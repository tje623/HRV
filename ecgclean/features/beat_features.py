#!/usr/bin/env python3
"""
ecgclean/features/beat_features.py — Step 3a: Beat-Level Feature Matrix

Computes the full beat-level feature matrix consumed by the Stage 1 GBM
and downstream CNN / ensemble models.  Each row corresponds to one R-peak
in peaks.parquet and carries features from five groups:

  1. Legacy v6 features (RR stats + ECG window stats)
  2. Extended RR context (wider neighbourhood, local stats)
  3. QRS morphology similarity (template correlation)
  4. Physio constraint pass-through (flags from Step 2)
  5. Segment-level context (broadcast per segment)
  6. Label-free signal quality (raw waveform statistics, immune to label contamination)

All output columns are float32 or int32 — no object dtypes, no NaN.

Usage:
    python ecgclean/features/beat_features.py \
        --processed-dir data/processed/ \
        --output data/processed/beat_features.parquet
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import multiprocessing
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis as scipy_kurtosis

# Import shared utility — works both as module and as standalone script
try:
    from ecgclean.features import pearson_corr_safe
except ImportError:
    # Running as __main__ without package install — inline fallback
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
logger = logging.getLogger("ecgclean.features.beat_features")

# Minimum clean beats per segment for a usable local QRS template
_MIN_TEMPLATE_BEATS: int = 3

# Pan-Tompkins finds the MWI peak (biased ~2-8 samples toward the QRS upslope)
# not the R-peak apex. Snap the window center to argmax(|ecg|) within this radius.
_PEAK_SNAP_SAMPLES: int = 8   # ±8 samples = ±62ms at 130 Hz

# Fork-shared globals dict for multiprocessing workers
_G: dict = {}


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORISED PEARSON CORRELATION BATCH
# ═══════════════════════════════════════════════════════════════════════════════


def _pearson_batch(a: np.ndarray, b: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
    """Compute row-wise Pearson correlation between a and b in batches.

    Processes rows in batches of ``batch_size`` using einsum to stay
    cache-friendly and avoid large temporary arrays.

    Args:
        a: (n, m) float32 array.
        b: (n, m) float32 array, same shape as a.
        batch_size: Number of rows per einsum batch (default 50,000).

    Returns:
        (n,) float32 array of Pearson r values; 0.0 where denominator == 0.
    """
    n = a.shape[0]
    out = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        a_b = a[start:end].astype(np.float64)
        b_b = b[start:end].astype(np.float64)

        a_c = a_b - a_b.mean(axis=1, keepdims=True)
        b_c = b_b - b_b.mean(axis=1, keepdims=True)

        num = np.einsum("ij,ij->i", a_c, b_c)
        denom = (
            np.sqrt(np.einsum("ij,ij->i", a_c, a_c))
            * np.sqrt(np.einsum("ij,ij->i", b_c, b_c))
        )
        r = np.where(denom > 0, num / denom, 0.0)
        out[start:end] = r.astype(np.float32)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 1 — Legacy v6 features
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_legacy_features(
    rr_prev: np.ndarray,
    rr_next: np.ndarray,
    ecg_windows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the original v6 training features.

    These features must be computed identically to the original pipeline to
    preserve backward compatibility with existing model weights.

    Args:
        rr_prev: RR interval to previous beat (ms), shape (n,).
        rr_next: RR interval to next beat (ms), shape (n,).
        ecg_windows: ECG window samples, shape (n, 64).

    Returns:
        Dict of feature-name → float32 array.
    """
    n = len(rr_prev)

    with np.errstate(divide="ignore", invalid="ignore"):
        rr_ratio = np.where(
            (rr_next == 0) | np.isnan(rr_next),
            0.0,
            rr_prev / rr_next,
        )
    rr_diff = rr_prev - rr_next
    rr_mean = (rr_prev + rr_next) / 2.0

    window_mean = np.mean(ecg_windows, axis=1)
    window_std = np.std(ecg_windows, axis=1)
    window_ptp = np.ptp(ecg_windows, axis=1)
    window_min = np.min(ecg_windows, axis=1)
    window_max = np.max(ecg_windows, axis=1)

    return {
        "rr_prev": rr_prev.astype(np.float32),
        "rr_next": rr_next.astype(np.float32),
        "rr_ratio": rr_ratio.astype(np.float32),
        "rr_diff": rr_diff.astype(np.float32),
        "rr_mean": rr_mean.astype(np.float32),
        "window_mean": window_mean.astype(np.float32),
        "window_std": window_std.astype(np.float32),
        "window_ptp": window_ptp.astype(np.float32),
        "window_min": window_min.astype(np.float32),
        "window_max": window_max.astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 2 — Extended RR context
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_extended_rr(
    timestamps_ns: np.ndarray,
    rr_prev: np.ndarray,
    rr_next: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute extended RR context features from the sorted beat sequence.

    Features include 2-beat-back/forward intervals, 5-beat local statistics,
    and absolute delta measures.

    Args:
        timestamps_ns: Sorted int64 peak timestamps in nanoseconds.
        rr_prev: RR to previous beat (ms).  Already NaN-filled for edges.
        rr_next: RR to next beat (ms).  Already NaN-filled for edges.

    Returns:
        Dict of feature-name → float32 array.
    """
    n = len(timestamps_ns)
    ms_per_ns = 1e-6

    # ── rr_prev_2 / rr_next_2: two beats back/forward ────────────────────
    rr_all = np.diff(timestamps_ns.astype(np.float64)) * ms_per_ns  # (n-1,)

    rr_prev_2 = np.full(n, np.nan, dtype=np.float64)
    if n > 2:
        rr_prev_2[2:] = rr_all[:-1]

    rr_next_2 = np.full(n, np.nan, dtype=np.float64)
    if n > 2:
        rr_next_2[:-2] = rr_all[1:]

    # ── 5-beat local window (centered), vectorised with sliding_window_view ─
    # For beat i, the window is beats [i-2 .. i+2].
    # RR intervals involved: rr_all[i-2], rr_all[i-1], rr_all[i], rr_all[i+1]
    # That gives 4 RR values for each center beat.
    rr_local_mean_5 = np.full(n, np.nan, dtype=np.float64)
    rr_local_sd_5 = np.full(n, np.nan, dtype=np.float64)

    if n >= 5:
        # Pad rr_all by 2 on each side (edge mode) so every beat i maps to
        # a 4-element window without boundary checks.
        rr_padded = np.pad(rr_all.astype(np.float64), (2, 2), mode="edge")  # len n+3
        wins_4 = sliding_window_view(rr_padded, 4)  # shape (n, 4)
        rr_local_mean_5 = np.mean(wins_4, axis=1)
        rr_local_sd_5 = np.std(wins_4, axis=1)
    elif n > 1:
        # Fewer than 5 beats: use all available
        valid_rr = rr_all[~np.isnan(rr_all)]
        if len(valid_rr) > 0:
            rr_local_mean_5[:] = np.mean(valid_rr)
            rr_local_sd_5[:] = np.std(valid_rr) if len(valid_rr) > 1 else 0.0

    # ── Absolute deltas ──────────────────────────────────────────────────
    rr_abs_delta_prev = np.abs(rr_prev - rr_prev_2)
    rr_abs_delta_next = np.abs(rr_next - rr_next_2)

    # ── rr_delta_ratio_next (== rr_ratio by spec; kept as separate column) ─
    with np.errstate(divide="ignore", invalid="ignore"):
        rr_delta_ratio_next = np.where(
            (rr_next == 0) | np.isnan(rr_next),
            0.0,
            rr_prev / rr_next,
        )

    return {
        "rr_prev_2": rr_prev_2.astype(np.float32),
        "rr_next_2": rr_next_2.astype(np.float32),
        "rr_local_mean_5": rr_local_mean_5.astype(np.float32),
        "rr_local_sd_5": rr_local_sd_5.astype(np.float32),
        "rr_abs_delta_prev": rr_abs_delta_prev.astype(np.float32),
        "rr_abs_delta_next": rr_abs_delta_next.astype(np.float32),
        "rr_delta_ratio_next": rr_delta_ratio_next.astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 3 — QRS morphology similarity
# ═══════════════════════════════════════════════════════════════════════════════


def _build_segment_templates(
    ecg_windows: np.ndarray,
    segment_ids: np.ndarray,
    labels: np.ndarray,
    hard_filtered: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Build per-segment clean QRS templates and a global fallback template.

    For each segment, the template is the mean of all 64-sample windows
    belonging to clean, non-hard-filtered beats.  Segments with fewer
    than ``_MIN_TEMPLATE_BEATS`` clean beats use the global mean template.

    Args:
        ecg_windows: (n, 64) array of beat ECG windows.
        segment_ids: (n,) array of segment indices per beat.
        labels: (n,) array of label strings per beat.
        hard_filtered: (n,) boolean array of hard-filter status.

    Returns:
        Tuple of (segment_template_dict, global_template).
        segment_template_dict maps segment_idx → (64,) float64 array.
        global_template is the (64,) float64 fallback.
    """
    clean_mask = (labels == "clean") & (~hard_filtered)
    clean_windows = ecg_windows[clean_mask]

    # Global fallback template
    if len(clean_windows) >= 1:
        global_template = np.mean(clean_windows.astype(np.float64), axis=0)
    else:
        # Absolute fallback: mean of all windows
        global_template = np.mean(ecg_windows.astype(np.float64), axis=0)

    # Per-segment templates
    unique_segs = np.unique(segment_ids)
    seg_templates: dict[int, np.ndarray] = {}
    for seg_idx in unique_segs:
        seg_clean = clean_mask & (segment_ids == seg_idx)
        n_clean = int(seg_clean.sum())
        if n_clean >= _MIN_TEMPLATE_BEATS:
            seg_templates[int(seg_idx)] = np.mean(
                ecg_windows[seg_clean].astype(np.float64), axis=0
            )
        else:
            seg_templates[int(seg_idx)] = global_template

    return seg_templates, global_template


def _compute_qrs_similarity(
    ecg_windows: np.ndarray,
    segment_ids: np.ndarray,
    labels: np.ndarray,
    hard_filtered: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute QRS morphology similarity features for every beat.

    Features:
        - qrs_corr_to_template: correlation with segment-local clean template
        - qrs_corr_prev: correlation with previous beat's window
        - qrs_corr_next: correlation with next beat's window

    Args:
        ecg_windows: (n, 64) array of beat ECG windows.
        segment_ids: (n,) array of segment indices.
        labels: (n,) array of label strings.
        hard_filtered: (n,) boolean array.

    Returns:
        Dict of feature-name → float32 array.
    """
    n = len(ecg_windows)
    seg_templates, global_template = _build_segment_templates(
        ecg_windows, segment_ids, labels, hard_filtered
    )

    # Build per-beat template array (vectorised lookup)
    template_arr = np.stack(
        [seg_templates.get(int(s), global_template) for s in segment_ids]
    ).astype(np.float32)

    qrs_corr_template = _pearson_batch(ecg_windows.astype(np.float32), template_arr)

    qrs_corr_prev = np.zeros(n, dtype=np.float32)
    qrs_corr_next = np.zeros(n, dtype=np.float32)

    if n > 1:
        qrs_corr_prev[1:] = _pearson_batch(
            ecg_windows[1:].astype(np.float32),
            ecg_windows[:-1].astype(np.float32),
        )
        qrs_corr_next[:-1] = _pearson_batch(
            ecg_windows[:-1].astype(np.float32),
            ecg_windows[1:].astype(np.float32),
        )

    return {
        "qrs_corr_to_template": qrs_corr_template,
        "qrs_corr_prev": qrs_corr_prev,
        "qrs_corr_next": qrs_corr_next,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 4 — Physio constraint pass-through
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_physio_features(labels_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract physio constraint columns from labels, casting bools to int32.

    These come from Step 2 (physio_constraints.py) and are passed through
    directly as model features.

    Args:
        labels_df: Labels table aligned to the sorted peak order.

    Returns:
        Dict of feature-name → int32/float32 array.
    """
    bool_cols = [
        "physio_implausible",
        "pots_transition_candidate",
        "tachy_transition_candidate",
        "hr_suspicious_low",
        "hr_suspicious_high",
        "rr_suspicious_short",
        "rr_suspicious_long",
        "is_added_peak",
    ]
    features: dict[str, np.ndarray] = {}

    for col in bool_cols:
        if col in labels_df.columns:
            features[col] = labels_df[col].values.astype(np.int32)
        else:
            features[col] = np.zeros(len(labels_df), dtype=np.int32)

    if "review_priority_score" in labels_df.columns:
        features["review_priority_score"] = (
            labels_df["review_priority_score"]
            .fillna(0.0)
            .values.astype(np.float32)
        )
    else:
        features["review_priority_score"] = np.zeros(
            len(labels_df), dtype=np.float32
        )

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 5 — Segment-level context
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_segment_context(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    segment_quality_preds: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute per-segment aggregate features and return as a join-ready frame.

    Segment-level features are broadcast (repeated) to every beat in the
    segment during the final merge.

    Args:
        peaks_df: Sorted peaks table with segment_idx and peak_id.
        labels_df: Labels table with peak_id, label, rr_prev_ms.
        segment_quality_preds: Optional Stage 0 predictions with
            segment_idx and quality_pred columns.

    Returns:
        DataFrame indexed by segment_idx with columns:
        segment_artifact_fraction, segment_rr_sd,
        segment_clean_beat_count, segment_quality_pred.
    """
    # Merge labels onto peaks to get segment_idx
    merged = peaks_df[["peak_id", "segment_idx"]].merge(
        labels_df[["peak_id", "label", "rr_prev_ms"]],
        on="peak_id",
        how="left",
    )

    seg_groups = merged.groupby("segment_idx")

    artifact_frac = seg_groups.apply(
        lambda g: (g["label"] == "artifact").sum() / max(len(g), 1),
        include_groups=False,
    ).rename("segment_artifact_fraction")

    rr_sd = seg_groups["rr_prev_ms"].std().fillna(0.0).rename("segment_rr_sd")

    clean_count = seg_groups.apply(
        lambda g: (g["label"] == "clean").sum(),
        include_groups=False,
    ).rename("segment_clean_beat_count")

    seg_ctx = pd.DataFrame({
        "segment_artifact_fraction": artifact_frac.astype(np.float32),
        "segment_rr_sd": rr_sd.astype(np.float32),
        "segment_clean_beat_count": clean_count.astype(np.int32),
    })

    # Stage 0 predictions
    if segment_quality_preds is not None and "quality_pred" in segment_quality_preds.columns:
        pred_map = segment_quality_preds.set_index("segment_idx")["quality_pred"]
        seg_ctx["segment_quality_pred"] = (
            seg_ctx.index.map(pred_map).fillna(-1).astype(np.int32)
        )
    else:
        seg_ctx["segment_quality_pred"] = np.int32(-1)

    return seg_ctx


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 6 — Label-free signal quality
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_signal_quality_features(ecg_windows: np.ndarray) -> dict[str, np.ndarray]:
    """Label-free signal quality features.

    These measure properties of the raw ECG waveform window and are immune to
    label contamination — they depend only on the raw signal, never on labels
    or templates.

    Args:
        ecg_windows: (n, 64) float32 array of beat ECG windows.

    Returns:
        Dict of feature-name → float32 array.
    """
    # Work in float32 throughout (halves memory vs float64)
    w = ecg_windows.astype(np.float32)  # (n, 64)
    n = w.shape[0]

    # ── IQR: Q3 - Q1 (vectorised over all beats) ─────────────────────────
    pcts = np.percentile(w, [75, 25], axis=1)  # (2, n)
    window_iqr = (pcts[0] - pcts[1]).astype(np.float32)

    # ── Kurtosis (Fisher; normal distribution = 0) ────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        window_kurtosis = scipy_kurtosis(w, axis=1, fisher=True)
    window_kurtosis = np.nan_to_num(window_kurtosis, nan=0.0).astype(np.float32)

    # ── Zero-crossing rate on mean-subtracted window ──────────────────────
    w_centered = w - w.mean(axis=1, keepdims=True)          # (n, 64)
    signs = np.sign(w_centered)                              # (n, 64)
    signs[signs == 0] = 1                                    # treat 0 as positive
    zcr_counts = np.sum(signs[:, :-1] != signs[:, 1:], axis=1)  # (n,)
    window_zcr = (zcr_counts / 63.0).astype(np.float32)

    # ── R-peak SNR: |center sample| / (median(|window|) + eps) ───────────
    abs_w = np.abs(w)
    r_peak_snr = (abs_w[:, 32] / (np.median(abs_w, axis=1) + 1e-9)).astype(np.float32)

    # ── HF noise RMS: RMS of first-difference (crude high-pass) ──────────
    diffs = np.diff(w, axis=1)                               # (n, 63)
    window_hf_noise_rms = np.sqrt(np.mean(diffs ** 2, axis=1)).astype(np.float32)

    # ── Baseline wander slope: OLS via matrix form (no loop, no polyfit) ──
    xs = np.arange(64, dtype=np.float32)
    xs_c = xs - xs.mean()                                    # (64,) centred
    xs_var = float(np.sum(xs_c ** 2))                        # scalar
    w_c = w - w.mean(axis=1, keepdims=True)                  # (n, 64) centred
    if xs_var > 0:
        window_wander_slope = (w_c @ xs_c / xs_var).astype(np.float32)
    else:
        window_wander_slope = np.zeros(n, dtype=np.float32)

    # ── Energy ratio: QRS region (indices 24–40) vs total window ─────────
    qrs_energy = np.sum(w[:, 24:40] ** 2, axis=1)
    total_energy = np.sum(w ** 2, axis=1)
    window_energy_ratio = (qrs_energy / (total_energy + 1e-9)).astype(np.float32)

    return {
        "window_iqr":          window_iqr,
        "window_kurtosis":     window_kurtosis,
        "window_zcr":          window_zcr,
        "r_peak_snr":          r_peak_snr,
        "window_hf_noise_rms": window_hf_noise_rms,
        "window_wander_slope": window_wander_slope,
        "window_energy_ratio": window_energy_ratio,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NAN FILL — segment-median RR
# ═══════════════════════════════════════════════════════════════════════════════


def _fill_nan_with_segment_median(
    values: np.ndarray,
    segment_ids: np.ndarray,
) -> np.ndarray:
    """Fill NaN values in *values* with the per-segment median.

    If a segment has no valid values, falls back to the global median.
    If the global median is also NaN (all values NaN), fills with 0.0.

    Args:
        values: Float array possibly containing NaN.
        segment_ids: Array of segment indices, same length as values.

    Returns:
        Copy of values with NaNs replaced.
    """
    out = values.copy()
    nan_mask = np.isnan(out)
    if not nan_mask.any():
        return out

    # Global fallback
    global_med = float(np.median(out[~nan_mask])) if (~nan_mask).any() else 0.0

    # Per-segment medians and vectorised fill
    df = pd.DataFrame({"val": out, "seg": segment_ids})
    seg_meds = df.loc[~nan_mask].groupby("seg")["val"].median()
    nan_segs = segment_ids[nan_mask]
    fill_vals = pd.Series(nan_segs).map(seg_meds).fillna(global_med).values
    out[nan_mask] = fill_vals

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def compute_beat_feature_matrix(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ecg_windows: np.ndarray,
    segments_df: pd.DataFrame,
    segment_quality_preds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the complete beat-level feature matrix.

    Merges five feature groups into a single DataFrame indexed by peak_id.
    All output columns are float32 or int32.  NaN values are filled with
    segment-median RR intervals (for RR-derived features) or 0.0 (for
    correlation features).

    Args:
        peaks_df: Canonical peaks table (peak_id, timestamp_ns, segment_idx,
            source, is_added_peak).
        labels_df: Enriched labels table from physio_constraints (peak_id,
            label, rr_prev_ms, rr_next_ms, hard_filtered, and all soft flags).
        ecg_windows: ECG window array of shape (n_beats, 64), aligned to the
            row order of peaks_df.  Each row is 64 contiguous ECG samples
            centred on the R-peak.
        segments_df: Canonical segments table (segment_idx, quality_label).
        segment_quality_preds: Optional Stage 0 predictions DataFrame with
            columns (segment_idx, quality_pred).  If None, segment_quality_pred
            is set to -1 for all beats.

    Returns:
        DataFrame indexed by peak_id with all feature columns.  Row count
        equals len(peaks_df).
    """
    n = len(peaks_df)
    assert ecg_windows.shape[0] == n, (
        f"ecg_windows row count {ecg_windows.shape[0]} != peaks row count {n}"
    )

    logger.info("Computing beat feature matrix for %d beats", n)

    # ── Sort everything by timestamp for sequential features ──────────────
    sort_order = np.argsort(peaks_df["timestamp_ns"].values)
    peaks_sorted = peaks_df.iloc[sort_order].reset_index(drop=True)
    windows_sorted = ecg_windows[sort_order]

    # Align labels to sorted peaks by peak_id join
    label_map = labels_df.set_index("peak_id")
    labels_sorted = label_map.reindex(peaks_sorted["peak_id"]).reset_index()

    timestamps = peaks_sorted["timestamp_ns"].values.astype(np.int64)
    segment_ids = peaks_sorted["segment_idx"].values.astype(np.int32)

    # ── RR intervals (ms) from sorted timestamps ──────────────────────────
    ms_per_ns = 1e-6
    rr_prev = np.full(n, np.nan, dtype=np.float64)
    rr_next = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        diffs_ms = np.diff(timestamps.astype(np.float64)) * ms_per_ns
        rr_prev[1:] = diffs_ms
        rr_next[:-1] = diffs_ms

    # Fill NaN edges with segment median
    rr_prev = _fill_nan_with_segment_median(rr_prev, segment_ids)
    rr_next = _fill_nan_with_segment_median(rr_next, segment_ids)

    # ── Feature group 1: Legacy v6 ───────────────────────────────────────
    feats = _compute_legacy_features(rr_prev, rr_next, windows_sorted)

    # ── Feature group 2: Extended RR context ─────────────────────────────
    ext_rr = _compute_extended_rr(timestamps, rr_prev, rr_next)
    # Fill NaN in extended features with segment median
    for key in ("rr_prev_2", "rr_next_2", "rr_local_mean_5", "rr_local_sd_5",
                "rr_abs_delta_prev", "rr_abs_delta_next"):
        ext_rr[key] = _fill_nan_with_segment_median(
            ext_rr[key].astype(np.float64), segment_ids
        ).astype(np.float32)
    feats.update(ext_rr)

    # ── Feature group 3: QRS morphology similarity ───────────────────────
    label_vals = labels_sorted["label"].values.astype(str)
    hard_filt = (
        labels_sorted["hard_filtered"].values.astype(bool)
        if "hard_filtered" in labels_sorted.columns
        else np.zeros(n, dtype=bool)
    )
    qrs_feats = _compute_qrs_similarity(
        windows_sorted, segment_ids, label_vals, hard_filt
    )
    feats.update(qrs_feats)

    # ── Feature group 4: Physio constraint pass-through ──────────────────
    # Need is_added_peak from peaks, other flags from labels
    physio_labels = labels_sorted.copy()
    physio_labels["is_added_peak"] = peaks_sorted["is_added_peak"].values
    physio_feats = _extract_physio_features(physio_labels)
    feats.update(physio_feats)

    # ── Feature group 5: Segment-level context ───────────────────────────
    seg_ctx = _compute_segment_context(
        peaks_sorted, labels_sorted, segment_quality_preds
    )

    # ── Feature group 6: Label-free signal quality ────────────────────────
    sig_quality_feats = _compute_signal_quality_features(windows_sorted)
    feats.update(sig_quality_feats)

    # ── Assemble result DataFrame ─────────────────────────────────────────
    result = pd.DataFrame(feats)
    result["peak_id"] = peaks_sorted["peak_id"].values
    result["segment_idx"] = segment_ids

    # Join segment context
    result = result.merge(
        seg_ctx, left_on="segment_idx", right_index=True, how="left"
    )

    # Fill any remaining NaN in segment context columns
    for col in ("segment_artifact_fraction", "segment_rr_sd"):
        if col in result.columns:
            result[col] = result[col].fillna(0.0).astype(np.float32)
    if "segment_clean_beat_count" in result.columns:
        result["segment_clean_beat_count"] = (
            result["segment_clean_beat_count"].fillna(0).astype(np.int32)
        )
    if "segment_quality_pred" in result.columns:
        result["segment_quality_pred"] = (
            result["segment_quality_pred"].fillna(-1).astype(np.int32)
        )

    # Drop helper column, set index
    result.drop(columns=["segment_idx"], inplace=True)
    result.set_index("peak_id", inplace=True)

    # ── Final NaN sweep: replace any remaining NaN with 0 ─────────────────
    nan_counts_before = result.isna().sum()
    remaining_nan_cols = nan_counts_before[nan_counts_before > 0]
    if len(remaining_nan_cols) > 0:
        logger.warning(
            "Filling %d remaining NaN values across %d columns with 0",
            int(nan_counts_before.sum()),
            len(remaining_nan_cols),
        )
        result.fillna(0.0, inplace=True)

    # ── Enforce dtypes ────────────────────────────────────────────────────
    int_cols = [
        "physio_implausible", "pots_transition_candidate", "tachy_transition_candidate",
        "hr_suspicious_low", "hr_suspicious_high",
        "rr_suspicious_short", "rr_suspicious_long",
        "is_added_peak", "segment_clean_beat_count", "segment_quality_pred",
    ]
    for col in int_cols:
        if col in result.columns:
            result[col] = result[col].astype(np.int32)
    for col in result.columns:
        if col not in int_cols and result[col].dtype not in (np.int32, np.float32):
            result[col] = result[col].astype(np.float32)

    logger.info(
        "Beat feature matrix complete: %d rows × %d columns", len(result), len(result.columns)
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ECG WINDOW LOADER (for CLI standalone use)
# ═══════════════════════════════════════════════════════════════════════════════


def _load_ecg_windows_from_peaks_csv(
    peaks_df: pd.DataFrame,
    ecg_samples_df: pd.DataFrame,
    window_size: int = 64,
) -> np.ndarray:
    """Reconstruct ECG windows from ecg_samples for each peak.

    For each peak, extracts ``window_size`` samples centered on the peak
    timestamp from the ECG time series.  Uses binary search for fast
    nearest-sample lookup.

    Args:
        peaks_df: Peaks table with timestamp_ns.
        ecg_samples_df: ECG samples table with timestamp_ns and ecg.
        window_size: Number of ECG samples per window (default 64).

    Returns:
        numpy array of shape (n_peaks, window_size), dtype float32.
    """
    n_peaks = len(peaks_df)
    windows = np.zeros((n_peaks, window_size), dtype=np.float32)

    ecg_ts = ecg_samples_df["timestamp_ns"].values.astype(np.int64)
    ecg_vals = ecg_samples_df["ecg"].values.astype(np.float32)
    n_ecg = len(ecg_ts)

    if n_ecg == 0:
        logger.warning("No ECG samples available for window reconstruction")
        return windows

    peak_ts = peaks_df["timestamp_ns"].values.astype(np.int64)
    half = window_size // 2

    # Binary search for nearest sample to each peak
    insert_idx = np.searchsorted(ecg_ts, peak_ts, side="left")

    for i in range(n_peaks):
        center = int(insert_idx[i])
        # Refine: pick closer of left/right neighbour
        if center > 0 and center < n_ecg:
            if abs(ecg_ts[center - 1] - peak_ts[i]) < abs(ecg_ts[center] - peak_ts[i]):
                center = center - 1
        elif center >= n_ecg:
            center = n_ecg - 1

        # Snap to local amplitude maximum: Pan-Tompkins returns the MWI peak,
        # which is biased toward the QRS upslope. Correcting to argmax(|ecg|)
        # within ±_PEAK_SNAP_SAMPLES centres the window on the actual R-peak apex.
        snap_lo = max(0, center - _PEAK_SNAP_SAMPLES)
        snap_hi = min(n_ecg, center + _PEAK_SNAP_SAMPLES + 1)
        center  = snap_lo + int(np.argmax(np.abs(ecg_vals[snap_lo:snap_hi])))

        start = center - half
        end = start + window_size

        # Clip to valid range
        src_start = max(0, start)
        src_end = min(n_ecg, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)

        if src_end > src_start:
            windows[i, dst_start:dst_end] = ecg_vals[src_start:src_end]

    return windows


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIPROCESSING WORKER
# ═══════════════════════════════════════════════════════════════════════════════


def _process_chunk_worker(task: dict) -> tuple[int, str]:
    """Worker function for parallel chunk processing.

    Reads shared data from module-level ``_G`` dict (populated before forking),
    processes one chunk of segments, writes result to a checkpoint parquet file,
    and returns ``(chunk_num, ckpt_path_str)``.

    Args:
        task: Dict with keys:
            chunk_num (int), chunk_segs (list[int]), ckpt_path (str).

    Returns:
        Tuple of (chunk_num, ckpt_path_str).
    """
    chunk_num: int = task["chunk_num"]
    chunk_segs: list[int] = task["chunk_segs"]
    ckpt_path: str = task["ckpt_path"]

    peaks_sorted = _G["peaks_sorted"]
    seg_arr = _G["seg_arr"]
    labels_indexed = _G["labels_indexed"]
    segments_df = _G["segments_df"]
    seg_quality = _G["seg_quality"]
    ecg_samples_path = _G["ecg_samples_path"]

    seg_min = int(chunk_segs[0])
    seg_max = int(chunk_segs[-1])

    # O(log n) slice into pre-sorted peaks
    lo = int(np.searchsorted(seg_arr, seg_min, side="left"))
    hi = int(np.searchsorted(seg_arr, seg_max + 1, side="left"))
    chunk_peaks = peaks_sorted.iloc[lo:hi].copy()

    if len(chunk_peaks) == 0:
        # Write empty parquet so the checkpoint file exists
        empty_table = pa.table({})
        pq.write_table(empty_table, ckpt_path)
        return (chunk_num, ckpt_path)

    # Reindex labels from shared index
    chunk_labels = labels_indexed.reindex(chunk_peaks["peak_id"]).reset_index()

    # Read ECG samples for this segment range via predicate pushdown
    ecg_chunk = (
        pq.read_table(
            ecg_samples_path,
            filters=[
                ("segment_idx", ">=", max(0, seg_min - 1)),
                ("segment_idx", "<=", seg_max + 1),
            ],
            columns=["timestamp_ns", "ecg"],
        )
        .to_pandas()
        .sort_values("timestamp_ns")
        .reset_index(drop=True)
    )

    ecg_windows = _load_ecg_windows_from_peaks_csv(chunk_peaks, ecg_chunk)
    del ecg_chunk

    result_chunk = compute_beat_feature_matrix(
        chunk_peaks, chunk_labels, ecg_windows, segments_df, seg_quality
    )
    del ecg_windows

    table = pa.Table.from_pandas(result_chunk, preserve_index=True)
    pq.write_table(table, ckpt_path, compression="snappy")

    return (chunk_num, ckpt_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for beat feature computation."""
    parser = argparse.ArgumentParser(
        description="ECG Artifact Pipeline — Step 3a: Beat-Level Features",
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
        help="Output path for beat_features.parquet (default: <processed-dir>/beat_features.parquet)",
    )
    parser.add_argument(
        "--chunk-segments",
        type=int,
        default=5000,
        help=(
            "Number of 60-second segments to process per batch. "
            "Controls peak RAM: ~1 GB per 5000 segments. Default: 5000."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1) - 1)),
        help=(
            "Number of parallel worker processes. "
            "Default: min(8, max(1, cpu_count - 1))."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory for per-chunk checkpoint parquet files. "
            "Defaults to <output>.ckpt/. Existing checkpoints are skipped on resume."
        ),
    )
    parser.add_argument(
        "--segment-quality-preds",
        type=str,
        default=None,
        help="Optional path to segment_quality_preds.parquet for Stage 0 predictions.",
    )
    args = parser.parse_args()

    proc = Path(args.processed_dir)
    for fname in ("peaks.parquet", "labels.parquet", "segments.parquet", "ecg_samples.parquet"):
        p = proc / fname
        if not p.exists():
            logger.error("Required file not found: %s", p)
            sys.exit(1)

    # Load the small tables into RAM.  ecg_samples is intentionally NOT loaded
    # here — it is read in segment-range chunks below to avoid OOM.
    logger.info("Loading peaks, labels, segments from %s", proc)
    peaks_df = pd.read_parquet(proc / "peaks.parquet")
    labels_df = pd.read_parquet(proc / "labels.parquet")
    segments_df = pd.read_parquet(proc / "segments.parquet")
    logger.info(
        "Loaded: %d peaks, %d labels, %d segments",
        len(peaks_df), len(labels_df), len(segments_df),
    )

    ecg_samples_path = proc / "ecg_samples.parquet"

    # Optional segment quality predictions
    seg_quality: pd.DataFrame | None = None
    if args.segment_quality_preds:
        sqp = Path(args.segment_quality_preds)
        if sqp.exists():
            seg_quality = pd.read_parquet(sqp)
            logger.info("Loaded segment quality preds from %s", sqp)
        else:
            logger.warning("--segment-quality-preds path not found: %s", sqp)

    # Sort peaks by segment_idx once so each chunk is a contiguous slice.
    peaks_sorted = peaks_df.sort_values("segment_idx").reset_index(drop=True)
    seg_arr = peaks_sorted["segment_idx"].values
    labels_indexed = labels_df.set_index("peak_id")

    all_segs = sorted(peaks_sorted["segment_idx"].unique())
    total_segs = len(all_segs)
    chunk_size = args.chunk_segments

    out_path = Path(args.output) if args.output else proc / "beat_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else out_path.parent / (out_path.name + ".ckpt")
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (total_segs + chunk_size - 1) // chunk_size
    logger.info(
        "Processing %d segments in %d chunk(s) of up to %d segments each",
        total_segs, n_chunks, chunk_size,
    )

    # ── Scan for existing checkpoints (resume support) ────────────────────
    existing_ckpts: set[int] = set()
    for f in ckpt_dir.glob("chunk_?????.parquet"):
        try:
            existing_ckpts.add(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass

    if existing_ckpts:
        logger.info(
            "Found %d existing checkpoint(s); those chunks will be skipped",
            len(existing_ckpts),
        )

    # ── Build task list, skipping already-checkpointed chunks ────────────
    tasks: list[dict] = []
    for chunk_num, chunk_start in enumerate(range(0, total_segs, chunk_size), 1):
        ckpt_path = str(ckpt_dir / f"chunk_{chunk_num:05d}.parquet")
        if chunk_num in existing_ckpts:
            logger.info("[%d/%d] Skipping (checkpoint exists)", chunk_num, n_chunks)
            continue
        chunk_segs = all_segs[chunk_start: chunk_start + chunk_size]
        tasks.append({
            "chunk_num": chunk_num,
            "chunk_segs": chunk_segs,
            "ckpt_path": ckpt_path,
        })

    # ── Populate fork-shared globals ──────────────────────────────────────
    _G["peaks_sorted"] = peaks_sorted
    _G["seg_arr"] = seg_arr
    _G["labels_indexed"] = labels_indexed
    _G["segments_df"] = segments_df
    _G["seg_quality"] = seg_quality
    _G["ecg_samples_path"] = str(ecg_samples_path)

    # ── Dispatch tasks ────────────────────────────────────────────────────
    completed: list[tuple[int, str]] = []

    if args.workers > 1 and tasks:
        logger.info("Using %d worker processes (fork)", args.workers)
        mp_ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_ctx,
        ) as executor:
            future_to_task = {
                executor.submit(_process_chunk_worker, t): t for t in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                chunk_num, ckpt_path = future.result()
                logger.info(
                    "[%d/%d] Checkpoint written → %s",
                    chunk_num, n_chunks, ckpt_path,
                )
                completed.append((chunk_num, ckpt_path))
    else:
        if args.workers <= 1:
            logger.info("Running serially (workers=1)")
        for t in tasks:
            chunk_num, ckpt_path = _process_chunk_worker(t)
            logger.info(
                "[%d/%d] Checkpoint written → %s",
                chunk_num, n_chunks, ckpt_path,
            )
            completed.append((chunk_num, ckpt_path))

    # ── Assemble all checkpoints in order into final output ───────────────
    # Collect ALL chunk files (newly written + previously existing)
    all_ckpt_files: list[tuple[int, Path]] = []
    for f in ckpt_dir.glob("chunk_?????.parquet"):
        try:
            cnum = int(f.stem.split("_")[1])
            all_ckpt_files.append((cnum, f))
        except (IndexError, ValueError):
            pass
    all_ckpt_files.sort(key=lambda x: x[0])

    writer: pq.ParquetWriter | None = None
    total_beats = 0

    for cnum, ckpt_file in all_ckpt_files:
        tbl = pq.read_table(str(ckpt_file))
        if tbl.num_rows == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema, compression="snappy")
        writer.write_table(tbl)
        total_beats += tbl.num_rows
        logger.info("  assembled chunk %d: %d beats", cnum, tbl.num_rows)

    if writer is not None:
        writer.close()

    # ── Clean up checkpoint directory ─────────────────────────────────────
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    logger.info("Saved beat features → %s  (%d total beats)", out_path, total_beats)
    print(f"\n{'=' * 70}")
    print(f"  Beat Feature Matrix: {total_beats:,} beats written to {out_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
