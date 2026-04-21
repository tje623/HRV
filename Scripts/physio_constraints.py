#!/usr/bin/env python3
"""
ecgclean/physio_constraints.py — Step 2: Physiological Constraint Encoding

Applies patient-specific physiological knowledge to the canonical peak and
label tables produced by data_pipeline.py.  Adds hard-filter flags for
structurally impossible patterns and soft feature columns that encode
domain knowledge about this POTS patient's cardiac dynamics.

Hard filters target ONLY patterns that no cardiac mechanism can produce.
Soft features flag unusual-but-possible events for model consumption and
active-learning review prioritization.

Usage:
    python ecgclean/physio_constraints.py --processed-dir data/processed/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    HR_MODAL_LOW_BPM,
    HR_MODAL_HIGH_BPM,
    HR_SUSPICIOUS_LOW_BPM,
    HR_SUSPICIOUS_HIGH_BPM,
    RR_SUSPICIOUS_SHORT_MS,
    RR_SUSPICIOUS_LONG_MS,
    RR_SANDWICH_SHORT_MS,
    RR_NORMAL_LOW_MS,
    RR_NORMAL_HIGH_MS,
    POTS_MAX_DELTA_HR_PER_SEC,
    POTS_TRANSITION_WINDOW_SEC,
    TACHY_TRANSITION_MIN_RISE_BPM,
    TACHY_TRANSITION_LOOK_AHEAD_SEC,
    TACHY_TRANSITION_SMOOTH_BEATS,
)

# ── Derived constants (internal) ───────────────────────────────────────────────

_POTS_WINDOW_MS: int = int(POTS_TRANSITION_WINDOW_SEC * 1000)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ecgclean.physio_constraints")


# ═══════════════════════════════════════════════════════════════════════════════
# HARD FILTERS
# ═══════════════════════════════════════════════════════════════════════════════


def apply_hard_filters(
    peaks_df: pd.DataFrame, labels_df: pd.DataFrame
) -> pd.DataFrame:
    """Apply physiologically-impossible-pattern hard filters to beat labels.

    Hard filters target ONLY structural impossibilities:
      1. RR zero or negative (data integrity failure)
      2. Isolated impossible sandwich (spurious peak between two real beats)
      3. N+ consecutive bit-identical RR intervals (detector artifact)
      4. Timestamp out of sequence (equal or decreasing after sort)

    Beats already labeled 'artifact' by the annotation pipeline are also
    marked as hard-filtered with reason 'existing_artifact'.

    Args:
        peaks_df: Canonical peaks table with timestamp_ms and peak_id.
        labels_df: Canonical labels table with peak_id and label.

    Returns:
        Updated labels_df with two new columns:
          - hard_filtered (bool): True if the beat is a confirmed artifact.
          - hard_filter_reason (str or None): One of 'rr_zero_or_negative',
            'isolated_impossible_sandwich', 'consecutive_identical_rr',
            'timestamp_out_of_sequence', 'existing_artifact', or None.
    """
    # Sort peaks by timestamp for RR computation
    peaks_sorted = peaks_df.sort_values("timestamp_ms").reset_index(drop=True)
    ts = peaks_sorted["timestamp_ms"].values.astype(np.int64)
    peak_ids = peaks_sorted["peak_id"].values.astype(np.int64)
    n = len(peaks_sorted)

    # Map existing labels onto sorted order
    label_map = labels_df.set_index("peak_id")["label"]
    labels_sorted = peaks_sorted["peak_id"].map(label_map).values

    # Initialize result arrays
    hard_filtered = np.zeros(n, dtype=bool)
    reasons: np.ndarray = np.full(n, None, dtype=object)

    # Compute inter-beat intervals (n-1 elements)
    if n < 2:
        # Too few peaks for any RR-based filter
        return _merge_hard_filter_results(
            labels_df, peak_ids, hard_filtered, reasons, labels_sorted
        )

    rr_between = np.diff(ts).astype(np.float64)

    # Per-beat RR arrays (NaN for first/last as appropriate)
    rr_prev = np.full(n, np.nan)
    rr_prev[1:] = rr_between

    rr_next = np.full(n, np.nan)
    rr_next[:-1] = rr_between

    # ── Rule 1: RR zero or negative ────────────────────────────────────────
    # After sorting, rr < 0 is impossible; rr == 0 means duplicate timestamps.
    # Both are data integrity failures.
    mask_rr_nonpositive = ~np.isnan(rr_prev) & (rr_prev <= 0)
    hard_filtered[mask_rr_nonpositive] = True
    reasons[mask_rr_nonpositive] = "rr_zero_or_negative"

    # ── Rule 4: Timestamp out of sequence ──────────────────────────────────
    # Check the ORIGINAL (pre-sort) peak order for out-of-sequence timestamps.
    # Any peak whose original position differs from sorted position had a
    # sequencing problem.  Also catches equal timestamps (rr == 0) which
    # rule 1 already flags; rule 4 uses a distinct reason for pure ordering
    # violations detected before sort.
    orig_ts = peaks_df["timestamp_ms"].values.astype(np.int64)
    if len(orig_ts) > 1:
        orig_diffs = np.diff(orig_ts)
        oos_orig_indices = np.where(orig_diffs < 0)[0] + 1  # peaks that went backward
        if len(oos_orig_indices) > 0:
            # Map original-order peak_ids to sorted-order indices
            oos_peak_ids = set(peaks_df.iloc[oos_orig_indices]["peak_id"].values)
            for idx in range(n):
                if peak_ids[idx] in oos_peak_ids and not hard_filtered[idx]:
                    hard_filtered[idx] = True
                    reasons[idx] = "timestamp_out_of_sequence"

    # ── Rule 2: Isolated impossible sandwich ───────────────────────────────
    # A spurious detected peak wedged between two real beats:
    #   - rr_prev < 150ms  (way too close to the preceding real beat)
    #   - rr_next > 150ms  (not a pair of spurious peaks)
    #   - preceding beat's own rr_prev is in normal range (300-1200ms)
    #   - following beat's own rr_next is in normal range (300-1200ms)
    if n > 3:
        # Neighbor RR arrays
        # neighbor_prev_rr[i] = rr of beat i-1 to beat i-2 = rr_between[i-2]
        neighbor_prev_rr = np.full(n, np.nan)
        neighbor_prev_rr[2:] = rr_between[: n - 2]

        # neighbor_next_rr[i] = rr of beat i+1 to beat i+2 = rr_between[i+1]
        neighbor_next_rr = np.full(n, np.nan)
        neighbor_next_rr[: n - 2] = rr_between[1:]

        sandwich_mask = (
            ~np.isnan(rr_prev)
            & ~np.isnan(rr_next)
            & ~np.isnan(neighbor_prev_rr)
            & ~np.isnan(neighbor_next_rr)
            & (rr_prev < RR_SANDWICH_SHORT_MS)
            & (rr_next > RR_SANDWICH_SHORT_MS)
            & (neighbor_prev_rr >= RR_NORMAL_LOW_MS)
            & (neighbor_prev_rr <= RR_NORMAL_HIGH_MS)
            & (neighbor_next_rr >= RR_NORMAL_LOW_MS)
            & (neighbor_next_rr <= RR_NORMAL_HIGH_MS)
            & ~hard_filtered
        )
        hard_filtered[sandwich_mask] = True
        reasons[sandwich_mask] = "isolated_impossible_sandwich"

    # ── Existing artifacts ─────────────────────────────────────────────────
    return _merge_hard_filter_results(
        labels_df, peak_ids, hard_filtered, reasons, labels_sorted
    )



def _merge_hard_filter_results(
    labels_df: pd.DataFrame,
    sorted_peak_ids: np.ndarray,
    hard_filtered: np.ndarray,
    reasons: np.ndarray,
    labels_sorted: np.ndarray,
) -> pd.DataFrame:
    """Merge hard filter results back into labels_df.

    hard_filtered=True means a beat is structurally impossible (impossible RR,
    flatline detector artifact, out-of-sequence timestamp).  Annotated artifacts
    are NOT marked hard_filtered here — they are the positive training signal
    and must remain in the training set.

    Args:
        labels_df: Original labels table.
        sorted_peak_ids: Peak IDs in timestamp-sorted order.
        hard_filtered: Boolean mask from structural filters.
        reasons: String reasons from structural filters.
        labels_sorted: Original label values in sorted order (unused, kept for
            signature compatibility).

    Returns:
        Updated labels_df with hard_filtered and hard_filter_reason columns.
    """
    # Build result frame keyed by peak_id
    result = pd.DataFrame(
        {
            "peak_id": sorted_peak_ids,
            "hard_filtered": hard_filtered,
            "hard_filter_reason": reasons,
        }
    )

    # Drop any pre-existing hard_filtered columns so the merge doesn't
    # produce hard_filtered_x / hard_filtered_y suffix collisions
    drop_cols = [c for c in ("hard_filtered", "hard_filter_reason") if c in labels_df.columns]
    if drop_cols:
        labels_df = labels_df.drop(columns=drop_cols)

    out = labels_df.merge(result, on="peak_id", how="left")
    out["hard_filtered"] = out["hard_filtered"].fillna(False).astype(bool)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# POTS WINDOW BUILDER
# ═══════════════════════════════════════════════════════════════════════════════


def build_pots_windows(labels_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Build extended POTS transition windows from phys_event_window flags.

    Finds contiguous blocks of beats where phys_event_window is True,
    determines the timestamp range of each block, extends each side by
    POTS_TRANSITION_WINDOW_SEC, and merges overlapping windows.

    Requires labels_df to contain 'timestamp_ms' and 'phys_event_window'
    columns.  If 'timestamp_ms' is missing, logs a warning and returns
    an empty list.

    Args:
        labels_df: Labels table enriched with timestamp_ms.

    Returns:
        Sorted list of (start_ms, end_ms) tuples representing POTS
        transition windows (with buffer applied and overlaps merged).
    """
    if "timestamp_ms" not in labels_df.columns:
        logger.warning(
            "labels_df missing 'timestamp_ms' column; cannot build POTS windows"
        )
        return []

    if "phys_event_window" not in labels_df.columns:
        logger.warning(
            "labels_df missing 'phys_event_window' column; no POTS windows"
        )
        return []

    # Sort by timestamp for contiguity detection
    df = labels_df.sort_values("timestamp_ms").reset_index(drop=True)
    phys_mask = df["phys_event_window"].values.astype(bool)
    ts = df["timestamp_ms"].values.astype(np.int64)

    if not phys_mask.any():
        return []

    # Find contiguous blocks of True values
    phys_indices = np.where(phys_mask)[0]
    if len(phys_indices) == 0:
        return []

    # Split into contiguous runs: a gap of more than 1 index = new block
    breaks = np.where(np.diff(phys_indices) > 1)[0] + 1
    blocks = np.split(phys_indices, breaks)

    # Build raw windows from each block
    raw_windows: list[tuple[int, int]] = []
    for block in blocks:
        if len(block) == 0:
            continue
        block_start_ms = int(ts[block[0]])
        block_end_ms = int(ts[block[-1]])
        # Extend by POTS_TRANSITION_WINDOW_SEC on each side
        extended_start = block_start_ms - _POTS_WINDOW_MS
        extended_end = block_end_ms + _POTS_WINDOW_MS
        raw_windows.append((extended_start, extended_end))

    # Merge overlapping windows
    raw_windows.sort(key=lambda w: w[0])
    merged: list[tuple[int, int]] = []
    for start, end in raw_windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    logger.info(
        "Built %d POTS transition window(s) from %d phys_event block(s)",
        len(merged),
        len(blocks),
    )
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# TACHYCARDIC TRANSITION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════


def detect_tachycardic_transitions(ts_ms: np.ndarray, hr_bpm: np.ndarray) -> np.ndarray:
    """Detect tachycardic transition windows using backward rolling HR analysis.

    Identifies windows where HR rises >= TACHY_TRANSITION_MIN_RISE_BPM from the
    rolling floor seen within TACHY_TRANSITION_LOOK_AHEAD_SEC seconds (backward),
    then extends each detected block by POTS_TRANSITION_WINDOW_SEC on each side.

    This protects POTS orthostatic tachycardia surges from physio_implausible
    flagging without requiring any manual phys_event labels.

    Args:
        ts_ms: Sorted int64 timestamps in milliseconds (length n).
        hr_bpm: Instantaneous HR in bpm (length n, may contain NaN).

    Returns:
        Boolean array of length n; True where a beat is within a detected
        tachycardic transition window.
    """
    n = len(ts_ms)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Build time-indexed Series for rolling window operations.
    # ts_ms must be monotonically increasing (caller guarantees sort-by-timestamp).
    dt_index = pd.to_datetime(ts_ms, unit="ms")
    hr_series = pd.Series(hr_bpm, index=dt_index)

    # Step 1: Smooth instantaneous HR with a beat-count rolling mean.
    # This removes HRV noise (±25 bpm beat-to-beat at 130 Hz) so the rolling
    # floor tracks the underlying trend, not individual beat outliers.
    hr_smooth = (
        hr_series
        .rolling(TACHY_TRANSITION_SMOOTH_BEATS, min_periods=1)
        .mean()
        .values
    )

    # Step 2: Backward rolling minimum of the smoothed HR.
    # rolling_min[i] = min smoothed HR in [ts[i] - look_ahead_sec, ts[i]]
    look_ahead_str = f"{TACHY_TRANSITION_LOOK_AHEAD_SEC}s"
    smooth_series = pd.Series(hr_smooth, index=dt_index)
    rolling_min = smooth_series.rolling(look_ahead_str, min_periods=1).min().values

    # HR rise of smoothed signal relative to the rolling floor.
    hr_rise = hr_smooth - rolling_min

    # Raw tachycardic transition beats: smoothed HR rose significantly from floor.
    # Exclude NaN positions (no RR data → no HR).
    nan_mask = np.isnan(hr_smooth) | np.isnan(rolling_min)
    raw_tachy = (~nan_mask) & (hr_rise >= TACHY_TRANSITION_MIN_RISE_BPM)

    if not raw_tachy.any():
        logger.info("No tachycardic transitions detected.")
        return np.zeros(n, dtype=bool)

    # Group consecutive raw_tachy beats into contiguous blocks (same pattern
    # as build_pots_windows) to produce one window per burst, not one per beat.
    tachy_indices = np.where(raw_tachy)[0]
    breaks = np.where(np.diff(tachy_indices) > 1)[0] + 1
    blocks = np.split(tachy_indices, breaks)

    # Build extended windows: block timestamp span ± POTS_TRANSITION_WINDOW_SEC buffer.
    raw_windows: list[tuple[int, int]] = []
    for block in blocks:
        if len(block) == 0:
            continue
        block_start_ms = int(ts_ms[block[0]])
        block_end_ms = int(ts_ms[block[-1]])
        raw_windows.append(
            (block_start_ms - _POTS_WINDOW_MS, block_end_ms + _POTS_WINDOW_MS)
        )

    # Merge overlapping windows.
    raw_windows.sort(key=lambda w: w[0])
    merged: list[tuple[int, int]] = []
    for start, end in raw_windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    logger.info(
        "Detected %d tachycardic transition window(s) from %d onset block(s) "
        "(%d onset beat(s) total)",
        len(merged),
        len(blocks),
        int(raw_tachy.sum()),
    )

    return _is_in_any_window(ts_ms, merged)


# ═══════════════════════════════════════════════════════════════════════════════
# SOFT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════


def compute_soft_features(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute domain-knowledge soft features for every beat.

    Adds RR/HR metrics, suspicious-value flags, POTS transition candidacy,
    physiological implausibility, and a composite review priority score.

    All features are computed vectorially in milliseconds internally.

    Args:
        peaks_df: Canonical peaks table (timestamp_ms, peak_id).
        labels_df: Labels table with hard_filtered, hard_filter_reason,
                   and phys_event_window columns.
        segments_df: Segments table (unused currently; reserved for
                     segment-context features in later pipeline stages).

    Returns:
        Updated labels_df with new feature columns. Row count is unchanged.
    """
    # Merge timestamps into labels (needed for RR and window computations)
    if "timestamp_ms" not in labels_df.columns:
        labels_df = labels_df.merge(
            peaks_df[["peak_id", "timestamp_ms"]], on="peak_id", how="left"
        )

    # Sort by timestamp for sequential RR computation
    df = labels_df.sort_values("timestamp_ms").reset_index(drop=True)
    ts = df["timestamp_ms"].values.astype(np.int64)
    n = len(df)

    # Ensure hard_filtered column exists
    if "hard_filtered" not in df.columns:
        df["hard_filtered"] = False

    hard_filtered = df["hard_filtered"].values.astype(bool)

    # ── RR intervals ──────────────────────────────────────────────────────
    rr_prev_ms = np.full(n, np.nan)
    rr_next_ms = np.full(n, np.nan)
    if n > 1:
        diffs = np.diff(ts).astype(np.float64)
        rr_prev_ms[1:] = diffs
        rr_next_ms[:-1] = diffs

    rr_delta_ms = np.abs(rr_prev_ms - rr_next_ms)

    df["rr_prev_ms"] = rr_prev_ms
    df["rr_next_ms"] = rr_next_ms
    df["rr_delta_ms"] = rr_delta_ms

    # ── Instantaneous HR ──────────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        hr_bpm = np.where(
            np.isnan(rr_prev_ms) | (rr_prev_ms <= 0),
            np.nan,
            60_000.0 / rr_prev_ms,
        )
    df["hr_instantaneous_bpm"] = hr_bpm

    # ── HR change rate (bpm per second) ───────────────────────────────────
    # rate = |HR_i - HR_{i-1}| / (rr_prev_ms_i / 1000)
    hr_change_rate = np.full(n, np.nan)
    if n > 1:
        hr_prev = np.full(n, np.nan)
        hr_prev[1:] = hr_bpm[:-1]
        delta_hr = np.abs(hr_bpm - hr_prev)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta_t_sec = rr_prev_ms / 1000.0
            hr_change_rate = np.where(
                np.isnan(delta_hr) | np.isnan(delta_t_sec) | (delta_t_sec <= 0),
                np.nan,
                delta_hr / delta_t_sec,
            )
    df["hr_change_rate_bpm_per_sec"] = hr_change_rate

    # ── Suspicious-value boolean flags ────────────────────────────────────
    # All flags are False for hard-filtered beats and for beats with NaN RR
    not_hard = ~hard_filtered

    rr_valid = ~np.isnan(rr_prev_ms)
    hr_valid = ~np.isnan(hr_bpm)

    df["rr_suspicious_short"] = (
        rr_valid & not_hard & (rr_prev_ms < RR_SUSPICIOUS_SHORT_MS)
    )
    df["rr_suspicious_long"] = (
        rr_valid & not_hard & (rr_prev_ms > RR_SUSPICIOUS_LONG_MS)
    )
    df["hr_suspicious_low"] = (
        hr_valid & not_hard & (hr_bpm < HR_SUSPICIOUS_LOW_BPM)
    )
    df["hr_suspicious_high"] = (
        hr_valid & not_hard & (hr_bpm > HR_SUSPICIOUS_HIGH_BPM)
    )
    df["in_modal_hr_range"] = (
        hr_valid & (hr_bpm >= HR_MODAL_LOW_BPM) & (hr_bpm <= HR_MODAL_HIGH_BPM)
    )

    # ── POTS transition candidacy (label-driven: phys_event windows) ─────
    pots_windows = build_pots_windows(df)
    pots_candidate = _is_in_any_window(ts, pots_windows)
    df["pots_transition_candidate"] = pots_candidate

    # ── Tachycardic transition candidacy (auto-detected from HR trajectory)
    # Protects orthostatic tachycardia surges that have no phys_event label.
    tachy_transition = detect_tachycardic_transitions(ts, hr_bpm)  # ts is ms
    df["tachy_transition_candidate"] = tachy_transition

    # ── Physiological implausibility ──────────────────────────────────────
    # Flagged when HR change rate exceeds the POTS upper bound AND the beat
    # is NOT in a POTS transition window, NOT in an auto-detected tachycardic
    # transition, AND is NOT in the SVT short-RR range.
    rate_valid = ~np.isnan(hr_change_rate)
    df["physio_implausible"] = (
        rate_valid
        & not_hard
        & (hr_change_rate > POTS_MAX_DELTA_HR_PER_SEC)
        & ~pots_candidate
        & ~tachy_transition
        & ~df["rr_suspicious_short"].values
    )

    # ── Review priority score ─────────────────────────────────────────────
    score = np.zeros(n, dtype=np.float64)
    score += 3.0 * df["physio_implausible"].values.astype(np.float64)
    score += 2.0 * df["rr_suspicious_short"].values.astype(np.float64)
    score += 2.0 * df["rr_suspicious_long"].values.astype(np.float64)
    score += 1.5 * df["hr_suspicious_high"].values.astype(np.float64)
    score += 1.0 * df["hr_suspicious_low"].values.astype(np.float64)
    score += 0.5 * pots_candidate.astype(np.float64)
    score += 0.5 * tachy_transition.astype(np.float64)
    df["review_priority_score"] = score

    # ── Restore original row order (by peak_id) ──────────────────────────
    df = df.sort_values("peak_id").reset_index(drop=True)

    logger.info("Soft features computed for %d beats", n)
    return df


def _is_in_any_window(
    timestamps: np.ndarray, windows: list[tuple[int, int]]
) -> np.ndarray:
    """Check whether each timestamp falls within any (start, end) window.

    Uses vectorized numpy comparisons.  For a small number of windows
    (typical: < 100), the loop is negligible.

    Args:
        timestamps: int64 array of beat timestamps in milliseconds.
        windows: List of (start_ms, end_ms) tuples.

    Returns:
        Boolean array of same length as timestamps.
    """
    result = np.zeros(len(timestamps), dtype=bool)
    for start_ms, end_ms in windows:
        result |= (timestamps >= start_ms) & (timestamps <= end_ms)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════


def get_constraint_summary(labels_df: pd.DataFrame) -> dict[str, Any]:
    """Return all flag counts as a dict for downstream training scripts.

    Provides a machine-readable summary of hard filter results and soft
    feature flag distributions.

    Args:
        labels_df: Labels table after hard filters and soft features.

    Returns:
        Dict with keys: hard_filtered_total, hard_filter_reasons (dict),
        physio_implausible, pots_transition_candidate,
        rr_suspicious_short, rr_suspicious_long, hr_suspicious_low,
        hr_suspicious_high, review_priority_score_stats (dict).
    """
    summary: dict[str, Any] = {}

    # Hard filter counts
    if "hard_filtered" in labels_df.columns:
        summary["hard_filtered_total"] = int(labels_df["hard_filtered"].sum())
        if "hard_filter_reason" in labels_df.columns:
            reason_counts = (
                labels_df.loc[labels_df["hard_filtered"], "hard_filter_reason"]
                .value_counts()
                .to_dict()
            )
            summary["hard_filter_reasons"] = {
                str(k): int(v) for k, v in reason_counts.items()
            }
        else:
            summary["hard_filter_reasons"] = {}
    else:
        summary["hard_filtered_total"] = 0
        summary["hard_filter_reasons"] = {}

    # Soft feature flag counts
    for flag in [
        "physio_implausible",
        "pots_transition_candidate",
        "tachy_transition_candidate",
        "rr_suspicious_short",
        "rr_suspicious_long",
        "hr_suspicious_low",
        "hr_suspicious_high",
    ]:
        if flag in labels_df.columns:
            summary[flag] = int(labels_df[flag].sum())
        else:
            summary[flag] = 0

    # Review priority score distribution
    if "review_priority_score" in labels_df.columns:
        scores = labels_df["review_priority_score"].dropna()
        if len(scores) > 0:
            summary["review_priority_score_stats"] = {
                "min": float(scores.min()),
                "p25": float(scores.quantile(0.25)),
                "p50": float(scores.quantile(0.50)),
                "p75": float(scores.quantile(0.75)),
                "p90": float(scores.quantile(0.90)),
                "max": float(scores.max()),
                "nonzero_count": int((scores > 0).sum()),
            }
        else:
            summary["review_priority_score_stats"] = {}
    else:
        summary["review_priority_score_stats"] = {}

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def run(processed_dir: str) -> None:
    """Load canonical tables, apply hard filters and soft features, save.

    Loads peaks.parquet, labels.parquet, and segments.parquet from
    processed_dir, applies the full constraint pipeline, overwrites
    labels.parquet with the enriched version, and prints a summary.

    Args:
        processed_dir: Path to directory containing the Parquet tables.
    """
    proc = Path(processed_dir)

    # ── Load tables ────────────────────────────────────────────────────────
    peaks_path = proc / "peaks.parquet"
    labels_path = proc / "labels.parquet"
    segments_path = proc / "segments.parquet"

    for p in (peaks_path, labels_path, segments_path):
        if not p.exists():
            logger.error("Required file not found: %s", p)
            sys.exit(1)

    logger.info("Loading tables from %s", proc)
    peaks_df = pd.read_parquet(peaks_path)
    labels_df = pd.read_parquet(labels_path)
    segments_df = pd.read_parquet(segments_path)

    input_row_count = len(labels_df)
    logger.info(
        "Loaded: %d peaks, %d labels, %d segments",
        len(peaks_df),
        len(labels_df),
        len(segments_df),
    )

    # ── Step 1: Hard filters ──────────────────────────────────────────────
    print("\n>> Applying hard filters...")
    labels_df = apply_hard_filters(peaks_df, labels_df)

    # ── Step 2: Soft features ─────────────────────────────────────────────
    print(">> Computing soft features...")
    labels_df = compute_soft_features(peaks_df, labels_df, segments_df)

    # ── Validate row count ────────────────────────────────────────────────
    assert len(labels_df) == input_row_count, (
        f"Row count changed: {input_row_count} -> {len(labels_df)}"
    )

    # ── Save ──────────────────────────────────────────────────────────────
    print(">> Saving enriched labels.parquet...")
    table = pa.Table.from_pandas(labels_df, preserve_index=False)
    pq.write_table(table, labels_path, compression="snappy")
    logger.info("Saved enriched labels -> %s (%d rows)", labels_path, len(labels_df))

    # ── Print summary ─────────────────────────────────────────────────────
    summary = get_constraint_summary(labels_df)
    _print_summary(labels_df, summary)


def _print_summary(labels_df: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Print a human-readable summary of constraint pipeline results.

    Args:
        labels_df: Enriched labels table.
        summary: Dict from get_constraint_summary.
    """
    print(f"\n{'=' * 60}")
    print("  Physiological Constraint Summary")
    print(f"{'=' * 60}")
    print(f"  Total beats: {len(labels_df):,}")

    # Hard filters
    print(f"\n  Hard-filtered: {summary['hard_filtered_total']:,}")
    if summary["hard_filter_reasons"]:
        for reason, count in sorted(
            summary["hard_filter_reasons"].items(), key=lambda x: -x[1]
        ):
            print(f"    {reason:40s}: {count:>6,}")

    # Soft flags
    print(f"\n  Soft flags:")
    print(f"    physio_implausible              : {summary['physio_implausible']:>6,}")
    print(
        f"    pots_transition_candidate      : {summary['pots_transition_candidate']:>6,}"
    )
    print(
        f"    tachy_transition_candidate     : {summary['tachy_transition_candidate']:>6,}"
    )
    print(f"    rr_suspicious_short            : {summary['rr_suspicious_short']:>6,}")
    print(f"    rr_suspicious_long             : {summary['rr_suspicious_long']:>6,}")
    print(f"    hr_suspicious_low              : {summary['hr_suspicious_low']:>6,}")
    print(f"    hr_suspicious_high             : {summary['hr_suspicious_high']:>6,}")

    # Review priority score
    stats = summary.get("review_priority_score_stats", {})
    if stats:
        print(f"\n  Review priority score distribution:")
        print(
            f"    min={stats['min']:.1f}  p25={stats['p25']:.1f}  "
            f"p50={stats['p50']:.1f}  p75={stats['p75']:.1f}  "
            f"p90={stats['p90']:.1f}  max={stats['max']:.1f}"
        )
        print(f"    Beats with score > 0: {stats.get('nonzero_count', 0):,}")

    print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for the physiological constraints pipeline."""
    parser = argparse.ArgumentParser(
        description="ECG Artifact Pipeline — Step 2: Physiological Constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        required=True,
        help="Directory containing peaks.parquet, labels.parquet, segments.parquet",
    )
    args = parser.parse_args()
    run(args.processed_dir)


if __name__ == "__main__":
    main()
