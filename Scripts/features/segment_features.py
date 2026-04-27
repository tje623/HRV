#!/usr/bin/env python3
"""
ecgclean/features/segment_features.py — Step 3b: Segment-Level Feature Matrix

Computes the segment-level feature matrix consumed by the Stage 0 segment
quality classifier.  Each row corresponds to one 60-second segment and
carries features from five groups:

  1. HRV statistics (meanNN, sdNN, rmssd, pNN50, pNN20, etc.)
     signal-only: derived from inter-beat intervals (timing)

  2. RR roughness (fraction of large RR jumps, suspicious beat fraction)
     signal-only: derived from RR timing and physio constraint scores
     NOTE: artifact_fraction was removed — label-leakage.

  3. SQI_QRS (mean inter-beat morphology correlation among all valid beats)
     signal-only: uses all non-hard-filtered beats, no label filter.
     Previously filtered on label=="clean" which was leaky.

  4. Raw ECG amplitude statistics (median, IQR, p95, saturation fraction)
     signal-only: derived from raw ECG samples

  5. Label-free signal quality (new group)
     signal-only: segment_zcr, segment_spectral_entropy, segment_qrs_density,
     segment_flatline_fraction, segment_amplitude_range

Segments with insufficient data (< 5 beats, < 2 ECG samples) will have
NaN for features that require a minimum sample size.  Specifically:
  - HRV features (group 1): NaN if < 5 non-hard-filtered beats
  - RR roughness (group 2): NaN if < 2 non-hard-filtered beats
  - SQI_QRS (group 3): NaN if < 2 non-hard-filtered beats
  - ECG amplitude (group 4): NaN if < 1 ECG sample
  - Signal quality (group 5): NaN if < 4 ECG samples (welch/zcr minimum)

Downstream code handles imputation — these segments are NOT dropped.

Usage:
    python ecgclean/features/segment_features.py \\
        --processed-dir data/processed/ \\
        --output data/processed/segment_features.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SAMPLE_RATE_HZ,
    QRS_WINDOW_SAMPLES,
    ECG_BANDPASS_LOW_HZ,
    ECG_BANDPASS_HIGH_HZ,
    HR_MODAL_LOW_BPM,
    SEGMENT_DURATION_MS,
)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter, sosfilt, welch

# Import shared utility — works both as module and as standalone script
try:
    from ecgclean.features import pearson_corr_safe
except ImportError:
    def pearson_corr_safe(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[misc]
        if len(a) < 2 or len(b) < 2:
            return 0.0
        a_std = np.std(a)
        b_std = np.std(b)
        if a_std == 0.0 or b_std == 0.0:
            return 0.0
        r = np.corrcoef(a, b)[0, 1]
        if np.isnan(r) or np.isinf(r):
            return 0.0
        return float(r)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ecgclean.features.segment_features")

# SAMPLE_RATE_HZ imported from config (Polar H10 effective ECG sample rate = 130 Hz)

# Minimum thresholds for feature computation
_MIN_BEATS_HRV: int = 5
_MIN_BEATS_ROUGHNESS: int = 2
_MIN_BEATS_SQI: int = 2
_MIN_ECG_SAMPLES_QUALITY: int = 4  # minimum for ZCR / spectral features


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 1 — HRV features
# signal-only: derived from inter-beat timing only
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_hrv_features(rr_ms: np.ndarray) -> dict[str, float]:
    """Compute time-domain HRV statistics from an array of RR intervals.

    All features follow standard HRV nomenclature (Task Force 1996).

    Args:
        rr_ms: Array of RR intervals in milliseconds for one segment.
            Only non-hard-filtered beats should be included.

    Returns:
        Dict of HRV feature name → float value.  Returns NaN values if
        fewer than ``_MIN_BEATS_HRV`` intervals are provided.
    """
    nan_result = {
        "meanNN": np.nan, "sdNN": np.nan, "rmssd": np.nan,
        "pNN50": np.nan, "pNN20": np.nan, "minNN": np.nan,
        "maxNN": np.nan, "iqrNN": np.nan, "beat_count": 0.0,
    }

    # Filter out NaN
    rr = rr_ms[~np.isnan(rr_ms)]
    n = len(rr)

    if n < _MIN_BEATS_HRV:
        nan_result["beat_count"] = float(n)
        return nan_result

    meanNN = float(np.mean(rr))
    sdNN = float(np.std(rr, ddof=1)) if n > 1 else 0.0

    # RMSSD: root mean square of successive differences
    if n > 1:
        successive_diffs = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
        abs_diffs = np.abs(successive_diffs)
        pNN50 = float(np.sum(abs_diffs > 50) / len(successive_diffs))
        pNN20 = float(np.sum(abs_diffs > 20) / len(successive_diffs))
    else:
        rmssd = 0.0
        pNN50 = 0.0
        pNN20 = 0.0

    return {
        "meanNN": meanNN,
        "sdNN": sdNN,
        "rmssd": rmssd,
        "pNN50": pNN50,
        "pNN20": pNN20,
        "minNN": float(np.min(rr)),
        "maxNN": float(np.max(rr)),
        "iqrNN": float(np.percentile(rr, 75) - np.percentile(rr, 25)),
        "beat_count": float(n),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 2 — RR roughness
# signal-only: RR timing + physio constraint scores (no label dependency)
# artifact_fraction removed — was derived from beat labels (leaky).
# suspicious_beat_fraction kept — derived from physio constraint scores.
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_rr_roughness(
    rr_ms: np.ndarray,
    review_scores: np.ndarray,
) -> dict[str, float]:
    """Compute RR roughness and beat-quality fractions for a segment.

    Roughness measures how "jumpy" the RR sequence is — high roughness
    indicates noise, artifact, or ectopy.

    artifact_fraction was removed (label-leaky: counted beats labeled
    "artifact", but unannotated beats default to "clean").

    Args:
        rr_ms: RR intervals (ms) for all non-hard-filtered beats.
        review_scores: Review priority scores for all beats (physio-derived,
            not label-derived).

    Returns:
        Dict with roughness fractions and suspicious beat fraction.
    """
    nan_result = {
        "rr_roughness_100": np.nan,
        "rr_roughness_200": np.nan,
        "rr_roughness_300": np.nan,
        "suspicious_beat_fraction": np.nan,
    }

    rr = rr_ms[~np.isnan(rr_ms)]
    n_total = len(review_scores)

    if len(rr) < _MIN_BEATS_ROUGHNESS:
        if n_total > 0:
            nan_result["suspicious_beat_fraction"] = float(
                np.sum(review_scores > 0) / n_total
            )
        return nan_result

    # Successive absolute differences
    if len(rr) > 1:
        abs_diffs = np.abs(np.diff(rr))
        n_pairs = len(abs_diffs)
        rr_roughness_100 = float(np.sum(abs_diffs > 100) / n_pairs)
        rr_roughness_200 = float(np.sum(abs_diffs > 200) / n_pairs)
        rr_roughness_300 = float(np.sum(abs_diffs > 300) / n_pairs)
    else:
        rr_roughness_100 = 0.0
        rr_roughness_200 = 0.0
        rr_roughness_300 = 0.0

    suspicious_fraction = (
        float(np.sum(review_scores > 0) / n_total) if n_total > 0 else 0.0
    )

    return {
        "rr_roughness_100": rr_roughness_100,
        "rr_roughness_200": rr_roughness_200,
        "rr_roughness_300": rr_roughness_300,
        "suspicious_beat_fraction": suspicious_fraction,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 3 — SQI_QRS
# signal-only: morphology correlation, NO label filter.
# Previously filtered on label=="clean" which was leaky because unannotated
# beats default to "clean".  Now uses ALL non-hard-filtered beats.
# hard_filtered is structural (impossible timing), not label-derived.
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_sqi_qrs(
    windows: np.ndarray,
    hard_filtered: np.ndarray,
) -> float:
    """Compute Signal Quality Index based on QRS morphology consistency.

    SQI_QRS is the mean Pearson correlation between consecutive non-hard-
    filtered beat windows.  Uses ALL beats regardless of label — the prior
    "clean" filter was leaky because unannotated beats default to "clean".

    Hard-filtered beats are still excluded because they represent
    structurally impossible intervals (e.g., refractory violation), not
    an annotation judgment.

    High values (> 0.9) indicate clean signal; low values (< 0.7) suggest
    noise or artifact.

    Args:
        windows: (n_beats, QRS_WINDOW_SAMPLES) ECG windows for beats in
            this segment.
        hard_filtered: Boolean hard-filter flags (True = structurally
            impossible, always excluded).

    Returns:
        Mean inter-beat correlation, or NaN if insufficient valid beats.
    """
    valid_mask = ~hard_filtered
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < _MIN_BEATS_SQI:
        return np.nan

    correlations: list[float] = []
    for k in range(len(valid_idx) - 1):
        i, j = valid_idx[k], valid_idx[k + 1]
        r = pearson_corr_safe(
            windows[i].astype(np.float64),
            windows[j].astype(np.float64),
        )
        correlations.append(r)

    if len(correlations) == 0:
        return np.nan

    return float(np.mean(correlations))


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 4 — Raw ECG amplitude statistics
# signal-only: raw waveform amplitude features
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_ecg_amplitude_features(ecg_signal: np.ndarray) -> dict[str, float]:
    """Compute amplitude statistics from raw ECG samples for a segment.

    These capture baseline wander, saturation, and overall signal quality.

    Args:
        ecg_signal: 1-D ECG amplitude array for the segment.

    Returns:
        Dict with ecg_median, ecg_iqr, ecg_p95, ecg_saturation_fraction.
    """
    nan_result = {
        "ecg_median": np.nan,
        "ecg_iqr": np.nan,
        "ecg_p95": np.nan,
        "ecg_saturation_fraction": np.nan,
    }

    if len(ecg_signal) == 0:
        return nan_result

    ecg_median = float(np.median(ecg_signal))
    q25, q75 = np.percentile(ecg_signal, [25, 75])
    ecg_iqr = float(q75 - q25)

    abs_ecg = np.abs(ecg_signal)
    ecg_p95 = float(np.percentile(abs_ecg, 95))

    if ecg_p95 > 0:
        threshold = 0.95 * ecg_p95
        ecg_saturation_fraction = float(np.sum(abs_ecg > threshold) / len(ecg_signal))
    else:
        ecg_saturation_fraction = 0.0

    return {
        "ecg_median": ecg_median,
        "ecg_iqr": ecg_iqr,
        "ecg_p95": ecg_p95,
        "ecg_saturation_fraction": ecg_saturation_fraction,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 5 — Label-free signal quality (new)
# signal-only: all five features depend only on the raw ECG waveform and
# detected peak positions, never on beat labels.
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_label_free_quality(
    ecg_signal: np.ndarray,
    n_peaks: int,
) -> dict[str, float]:
    """Compute label-free segment-level signal quality features.

    These five features diagnose common failure modes (flatline, broadband
    noise, missing QRS) using only the raw ECG and peak count.

    Features:
        segment_zcr: Zero-crossing rate of the bandpass-filtered ECG.
            High (> 0.5) → noise/scribble.  Low (< 0.05) → flatline.
            Normal QRS signal falls in 0.05–0.35.

        segment_spectral_entropy: Shannon entropy (bits) of the power
            spectral density.  High → broadband noise.  Low → structured
            signal with dominant frequency components.

        segment_qrs_density: Detected peaks / expected peaks at
            HR_MODAL_LOW_BPM (85 BPM).  Value > 1.0 is plausible at higher
            HR.  Near 0 → detector found almost nothing = bad segment.

        segment_flatline_fraction: Fraction of 1-second windows whose
            std < max(10 µV, 5% of full-segment std).  Detects sensor-off
            periods, electrode disconnection, and saturated ADC output.

        segment_amplitude_range: 99th percentile minus 1st percentile of
            raw ECG amplitude.  Near 0 → flatline/DC.  Very large → motion
            artifact or saturation.

    Args:
        ecg_signal: 1-D ECG amplitude array for the segment.
        n_peaks: Number of detected R-peaks in this segment (from peaks table).

    Returns:
        Dict with the five feature values, all float.  NaN if the signal is
        too short (< _MIN_ECG_SAMPLES_QUALITY).
    """
    nan_result = {
        "segment_zcr": np.nan,
        "segment_spectral_entropy": np.nan,
        "segment_qrs_density": np.nan,
        "segment_flatline_fraction": np.nan,
        "segment_amplitude_range": np.nan,
    }

    if len(ecg_signal) < _MIN_ECG_SAMPLES_QUALITY:
        return nan_result

    ecg = ecg_signal.astype(np.float64)

    # ── segment_zcr ───────────────────────────────────────────────────────
    try:
        sos = butter(
            4,
            [ECG_BANDPASS_LOW_HZ, ECG_BANDPASS_HIGH_HZ],
            btype="bandpass",
            fs=SAMPLE_RATE_HZ,
            output="sos",
        )
        filtered = sosfilt(sos, ecg)
        signs = np.sign(filtered)
        signs[signs == 0] = 1.0
        zcr = float(np.mean(np.diff(signs) != 0))
    except Exception:
        zcr = np.nan

    # ── segment_spectral_entropy ──────────────────────────────────────────
    try:
        nperseg = min(256, len(ecg))
        _, psd = welch(ecg, fs=SAMPLE_RATE_HZ, nperseg=nperseg)
        psd_sum = psd.sum()
        if psd_sum > 0:
            psd_norm = psd / psd_sum
            psd_nz = psd_norm[psd_norm > 0]
            spectral_entropy = float(-np.sum(psd_nz * np.log2(psd_nz)))
        else:
            spectral_entropy = 0.0
    except Exception:
        spectral_entropy = np.nan

    # ── segment_qrs_density ───────────────────────────────────────────────
    expected_peaks = (SEGMENT_DURATION_MS / 1000.0) * (HR_MODAL_LOW_BPM / 60.0)
    qrs_density = float(n_peaks / expected_peaks) if expected_peaks > 0 else np.nan

    # ── segment_flatline_fraction ─────────────────────────────────────────
    window_samples = SAMPLE_RATE_HZ  # 1-second windows @ 130 Hz
    n_windows = len(ecg) // window_samples
    if n_windows == 0:
        flatline_frac: float = np.nan
    else:
        seg_std = float(np.std(ecg)) if len(ecg) > 1 else 0.0
        threshold = max(10.0, 0.05 * seg_std)
        flat_count = sum(
            float(np.std(ecg[i * window_samples:(i + 1) * window_samples])) < threshold
            for i in range(n_windows)
        )
        flatline_frac = float(flat_count) / float(n_windows)

    # ── segment_amplitude_range ───────────────────────────────────────────
    if len(ecg) >= 2:
        p1, p99 = np.percentile(ecg, [1, 99])
        amplitude_range = float(p99 - p1)
    else:
        amplitude_range = np.nan

    return {
        "segment_zcr": zcr,
        "segment_spectral_entropy": spectral_entropy,
        "segment_qrs_density": qrs_density,
        "segment_flatline_fraction": flatline_frac,
        "segment_amplitude_range": amplitude_range,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def compute_segment_feature_matrix(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ecg_samples_df: pd.DataFrame,
    ecg_windows: np.ndarray,
) -> pd.DataFrame:
    """Compute the complete segment-level feature matrix.

    Iterates over all segments, computes five feature groups per segment,
    and returns a DataFrame indexed by segment_idx.

    Feature contract: all output columns are signal-only (no label leakage).
    Specifically excluded:
      - artifact_fraction (was derived from beat labels)
      - f_imf_entropy/mean/variance (EMD features, dropped)

    Features that cannot be computed due to insufficient data are set to NaN.
    The following features may contain NaN:
      - HRV features (group 1): if segment has < 5 non-hard-filtered beats
      - RR roughness (group 2): if segment has < 2 non-hard-filtered beats
      - sqi_qrs (group 3): if segment has < 2 non-hard-filtered beats
      - ECG amplitude (group 4): if segment has 0 ECG samples
      - Signal quality (group 5): if segment has < 4 ECG samples

    Args:
        peaks_df: Canonical peaks table (peak_id, timestamp_ms, segment_idx).
        labels_df: Enriched labels table (peak_id, label, hard_filtered,
            rr_prev_ms, review_priority_score).
        ecg_samples_df: ECG samples table (timestamp_ms, ecg, segment_idx).
        ecg_windows: ECG window array (n_beats, QRS_WINDOW_SAMPLES), aligned
            to peaks_df row order.

    Returns:
        DataFrame indexed by segment_idx with all segment feature columns.
    """
    n_peaks = len(peaks_df)
    assert ecg_windows.shape[0] == n_peaks, (
        f"ecg_windows rows {ecg_windows.shape[0]} != peaks rows {n_peaks}"
    )

    # ── Merge peaks with labels ───────────────────────────────────────────
    merged = peaks_df[["peak_id", "segment_idx"]].merge(
        labels_df[["peak_id", "label", "hard_filtered", "rr_prev_ms",
                    "review_priority_score"]],
        on="peak_id",
        how="left",
    )
    # Ensure hard_filtered is bool
    if "hard_filtered" not in merged.columns:
        merged["hard_filtered"] = False
    merged["hard_filtered"] = merged["hard_filtered"].fillna(False).astype(bool)
    merged["review_priority_score"] = merged["review_priority_score"].fillna(0.0)

    # Attach row index in peaks_df for window lookup
    merged["_peak_row"] = np.arange(n_peaks)

    # Group ECG samples by segment
    ecg_by_seg = ecg_samples_df.groupby("segment_idx")

    # All segment indices (union of peaks and ECG sample segments)
    all_seg_idx = sorted(
        set(merged["segment_idx"].unique()) | set(ecg_samples_df["segment_idx"].unique())
    )

    logger.info(
        "Computing segment features for %d segments (%d peaks, %d ECG samples)",
        len(all_seg_idx), n_peaks, len(ecg_samples_df),
    )

    # Pre-group merged by segment_idx — avoids O(n_segments * n_peaks) boolean scan
    beats_by_seg = {
        seg: grp for seg, grp in merged.groupby("segment_idx", sort=False)
    }

    records: list[dict[str, float]] = []
    _empty_beats = merged.iloc[:0]  # zero-row DataFrame with correct columns

    for seg_idx in all_seg_idx:
        seg_int = int(seg_idx)
        row: dict[str, float] = {"segment_idx": float(seg_int)}

        # ── Get beats in this segment ─────────────────────────────────────
        seg_beats = beats_by_seg.get(seg_idx, _empty_beats)
        not_hard = seg_beats[~seg_beats["hard_filtered"]]

        rr_ms = not_hard["rr_prev_ms"].values.astype(np.float64)
        all_scores = seg_beats["review_priority_score"].values.astype(np.float64)
        hard_flags = seg_beats["hard_filtered"].values.astype(bool)

        # ── Beat windows for this segment ─────────────────────────────────
        beat_rows = seg_beats["_peak_row"].values.astype(int)
        seg_windows = (
            ecg_windows[beat_rows]
            if len(beat_rows) > 0
            else np.empty((0, QRS_WINDOW_SAMPLES))
        )

        # ── Feature group 1: HRV ──────────────────────────────────────────
        hrv = _compute_hrv_features(rr_ms)
        row.update(hrv)

        # ── Feature group 2: RR roughness (signal-only, no labels) ───────
        roughness = _compute_rr_roughness(rr_ms, all_scores)
        row.update(roughness)

        # ── Feature group 3: SQI_QRS (all non-hard-filtered beats) ───────
        row["sqi_qrs"] = _compute_sqi_qrs(seg_windows, hard_flags)

        # ── ECG signal for groups 4 & 5 ───────────────────────────────────
        if seg_idx in ecg_by_seg.groups:
            ecg_seg = ecg_by_seg.get_group(seg_idx)
            ecg_signal = ecg_seg["ecg"].values.astype(np.float64)
        else:
            ecg_signal = np.array([], dtype=np.float64)

        # ── Feature group 4: ECG amplitude (signal-only) ──────────────────
        amp_feats = _compute_ecg_amplitude_features(ecg_signal)
        row.update(amp_feats)

        # ── Feature group 5: Label-free signal quality (new) ─────────────
        quality_feats = _compute_label_free_quality(ecg_signal, len(seg_beats))
        row.update(quality_feats)

        records.append(row)

    # ── Assemble result ───────────────────────────────────────────────────
    result = pd.DataFrame(records)
    result["segment_idx"] = result["segment_idx"].astype(np.int32)
    result.set_index("segment_idx", inplace=True)

    # Enforce float32 for all non-int columns
    int_cols = {"beat_count"}
    for col in result.columns:
        if col in int_cols:
            result[col] = result[col].fillna(0).astype(np.int32)
        else:
            result[col] = result[col].astype(np.float32)

    logger.info(
        "Segment feature matrix complete: %d rows × %d columns",
        len(result), len(result.columns),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ECG WINDOW LOADER (shared with beat_features — inline for standalone CLI)
# ═══════════════════════════════════════════════════════════════════════════════


def _load_ecg_windows(
    peaks_df: pd.DataFrame,
    ecg_samples_df: pd.DataFrame,
    window_size: int = QRS_WINDOW_SAMPLES,
) -> np.ndarray:
    """Reconstruct ECG windows from ecg_samples for each peak.

    For each peak, extracts ``window_size`` samples centered on the peak
    timestamp from the ECG time series using binary search.

    Args:
        peaks_df: Peaks table with timestamp_ms.
        ecg_samples_df: ECG samples table with timestamp_ms and ecg.
        window_size: Number of ECG samples per window
            (default QRS_WINDOW_SAMPLES = 0.5 s @ 130 Hz).

    Returns:
        numpy array of shape (n_peaks, window_size), dtype float32.
    """
    n_peaks = len(peaks_df)
    windows = np.zeros((n_peaks, window_size), dtype=np.float32)

    ecg_ts = ecg_samples_df["timestamp_ms"].values.astype(np.int64)
    ecg_vals = ecg_samples_df["ecg"].values.astype(np.float32)
    n_ecg = len(ecg_ts)

    if n_ecg == 0:
        return windows

    peak_ts = peaks_df["timestamp_ms"].values.astype(np.int64)
    half = window_size // 2

    insert_idx = np.searchsorted(ecg_ts, peak_ts, side="left")

    for i in range(n_peaks):
        center = int(insert_idx[i])
        if center > 0 and center < n_ecg:
            if abs(ecg_ts[center - 1] - peak_ts[i]) < abs(ecg_ts[center] - peak_ts[i]):
                center = center - 1
        elif center >= n_ecg:
            center = n_ecg - 1

        start = center - half
        end = start + window_size

        src_start = max(0, start)
        src_end = min(n_ecg, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)

        if src_end > src_start:
            windows[i, dst_start:dst_end] = ecg_vals[src_start:src_end]

    return windows


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for segment feature computation."""
    parser = argparse.ArgumentParser(
        description="ECG Artifact Pipeline — Step 3b: Segment-Level Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        required=True,
        help="Directory containing peaks.parquet, labels.parquet, etc.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for segment_features.parquet "
            "(default: <processed-dir>/segment_features.parquet)"
        ),
    )
    parser.add_argument(
        "--chunk-segments",
        type=int,
        default=1000,
        help="Segments to process per chunk (default: 1000)",
    )
    args = parser.parse_args()

    proc = Path(args.processed_dir)
    ecg_samples_path = proc / "ecg_samples.parquet"
    for fname in ("peaks.parquet", "labels.parquet", "ecg_samples.parquet"):
        p = proc / fname
        if not p.exists():
            logger.error("Required file not found: %s", p)
            sys.exit(1)

    # Load small tables into RAM — never load ecg_samples.parquet directly
    logger.info("Loading peaks and labels from %s", proc)
    peaks_df = pd.read_parquet(proc / "peaks.parquet")
    labels_df = pd.read_parquet(proc / "labels.parquet")
    logger.info("Loaded: %d peaks, %d labels", len(peaks_df), len(labels_df))

    out_path = Path(args.output) if args.output else proc / "segment_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = args.chunk_segments

    # Sort peaks by segment_idx for O(log n) chunk slicing via searchsorted
    peaks_sorted = peaks_df.sort_values("segment_idx").reset_index(drop=True)
    seg_arr = peaks_sorted["segment_idx"].values
    all_segs = sorted(peaks_sorted["segment_idx"].unique())
    total_segs = len(all_segs)
    n_chunks = (total_segs + chunk_size - 1) // chunk_size

    logger.info(
        "Computing segment features: %d segments in %d chunks (chunk_size=%d) → %s",
        total_segs, n_chunks, chunk_size, out_path,
    )

    writer: pq.ParquetWriter | None = None

    for chunk_num, chunk_start in enumerate(range(0, total_segs, chunk_size), 1):
        chunk_segs = all_segs[chunk_start : chunk_start + chunk_size]
        seg_min = int(chunk_segs[0])
        seg_max = int(chunk_segs[-1])

        logger.info(
            "Chunk %d/%d: segments %d–%d", chunk_num, n_chunks, seg_min, seg_max
        )

        # Slice peaks for this segment range using binary search
        lo = int(np.searchsorted(seg_arr, seg_min, side="left"))
        hi = int(np.searchsorted(seg_arr, seg_max + 1, side="left"))
        chunk_peaks = peaks_sorted.iloc[lo:hi].copy().reset_index(drop=True)

        if len(chunk_peaks) == 0:
            logger.warning("No peaks for segments %d–%d — skipping", seg_min, seg_max)
            continue

        # Read ECG samples for this segment range via predicate pushdown
        ecg_chunk = pq.read_table(
            ecg_samples_path,
            filters=[
                ("segment_idx", ">=", seg_min),
                ("segment_idx", "<=", seg_max),
            ],
            columns=["timestamp_ms", "ecg", "segment_idx"],
        ).to_pandas().sort_values("timestamp_ms").reset_index(drop=True)

        # Build ECG windows for the peaks in this chunk
        chunk_windows = _load_ecg_windows(chunk_peaks, ecg_chunk)

        # Filter labels to only peaks in this chunk — avoids merging all 50M rows
        chunk_label_ids = set(chunk_peaks["peak_id"])
        chunk_labels = labels_df[labels_df["peak_id"].isin(chunk_label_ids)]

        # Compute all five feature groups for these segments
        result_chunk = compute_segment_feature_matrix(
            chunk_peaks, chunk_labels, ecg_chunk, chunk_windows
        )
        del ecg_chunk, chunk_windows

        # Stream-write to output parquet
        table = pa.Table.from_pandas(result_chunk, preserve_index=True)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()
        logger.info("Saved segment features → %s", out_path)
    else:
        logger.error("No segments processed — output not written")
        sys.exit(1)

    # ── Print summary (segment_features is small — safe to read back) ─────
    result = pd.read_parquet(out_path)
    print(f"\n{'=' * 70}")
    print(f"  Segment Feature Matrix Summary")
    print(f"{'=' * 70}")
    print(f"  Shape: {result.shape[0]} rows × {result.shape[1]} columns")
    print(f"\n  Columns ({len(result.columns)}):")
    for col in result.columns:
        print(f"    {col}: {result[col].dtype}")
    print(f"\n  NaN counts per column:")
    nan_counts = result.isna().sum()
    has_nan = False
    for col, cnt in nan_counts.items():
        if cnt > 0:
            has_nan = True
            print(f"    {col}: {cnt}")
    if not has_nan:
        print("    (none)")
    print(f"\n  First 3 rows:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(result.head(3).to_string())
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
