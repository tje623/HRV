#!/usr/bin/env python3
"""
physio_events.py — Algorithmic detection of vagal arrests and RSA events.

Reads peaks.parquet and outputs two CSVs:
  vagal_arrests.csv  — single-beat pauses with gradual recovery
  rsa_events.csv     — rhythmic RR compression/expansion cycles

Usage:
    python ecgclean/physio_events.py
    python ecgclean/physio_events.py --arrest-min-pause-ms 1500 --rsa-min-amplitude-ms 80
    python ecgclean/physio_events.py --peaks /Volumes/xHRV/processed/peaks.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import find_peaks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# A gap between consecutive beats larger than this is a recording session boundary
# (e.g. device removed / between files), not a cardiac event.
FILE_BOUNDARY_MS: float = 10_000.0  # 10 seconds

# Vagal arrest detection
# ---
# A vagal arrest = a single beat that arrives substantially later than expected
# (pause ratio ≥ ARREST_MIN_RATIO vs local baseline), with gradual return to
# normal over the following beats.  Does NOT require any specific beat morphology.
ARREST_MIN_PAUSE_MS: float = 1_200.0   # absolute floor: beat must take ≥ 1.2 s to arrive
ARREST_MIN_RATIO: float = 1.8          # relative floor: beat must be ≥ 1.8× the local baseline RR
ARREST_BASELINE_WINDOW: int = 11       # rolling window for pre-arrest baseline (beats)
ARREST_MAX_RECOVERY_BEATS: int = 15    # max beats to look for return-to-baseline
ARREST_RECOVERY_THRESHOLD: float = 0.20  # RR within this fraction of baseline = "recovered"
ARREST_MAX_BASELINE_MS: float = 1_500.0  # skip beats where baseline is already very slow

# RSA detection
# ---
# RSA = rhythmic alternation between fast and slow RR at respiratory frequency.
# Detected by finding peaks (slow periods) and troughs (fast periods) in the
# RR deviation signal (RR minus local baseline) and grouping consecutive
# alternating extrema into events.
RSA_MIN_AMPLITUDE_MS: float = 60.0    # minimum peak-to-trough amplitude per half-cycle
RSA_MIN_CYCLES: int = 2               # minimum full cycles to call it RSA
RSA_MIN_HALF_PERIOD: int = 2          # minimum beats per half-cycle (compress or expand)
RSA_MAX_HALF_PERIOD: int = 14         # maximum beats per half-cycle
RSA_BASELINE_WINDOW: int = 27         # centered rolling window for baseline estimation


# ── Data loading ─────────────────────────────────────────────────────────────

def load_peaks(peaks_path: Path) -> pd.DataFrame:
    """Load peaks.parquet, sort chronologically, compute within-session RR intervals.

    Sets rr_ms = NaN at recording boundaries so downstream algorithms never
    measure the gap between two separate sessions as a cardiac interval.
    """
    logger.info("Loading peaks from %s ...", peaks_path)
    peaks = pq.read_table(
        peaks_path, columns=["peak_id", "timestamp_ns", "segment_idx"]
    ).to_pandas()
    logger.info("  %d peaks loaded", len(peaks))

    peaks = peaks.sort_values("timestamp_ns").reset_index(drop=True)
    peaks["rr_ms"] = peaks["timestamp_ns"].diff() / 1_000_000.0  # ns → ms

    # Mark file/session boundaries
    seg_delta = peaks["segment_idx"].diff().fillna(1)
    peaks["is_boundary"] = (
        peaks["rr_ms"].isna()
        | (peaks["rr_ms"] > FILE_BOUNDARY_MS)
        | (seg_delta < 0)
    )
    peaks.loc[peaks["is_boundary"], "rr_ms"] = np.nan

    n_boundaries = peaks["is_boundary"].sum()
    valid_rr = peaks["rr_ms"].dropna()
    logger.info(
        "  %d session boundaries | RR: median=%.0f ms, max=%.0f ms (within-session only)",
        n_boundaries, valid_rr.median(), valid_rr.max(),
    )
    return peaks


def _session_slices(peaks: pd.DataFrame) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each contiguous recording session."""
    boundaries = np.where(peaks["is_boundary"].values)[0].tolist()
    starts = [0] + boundaries
    ends   = boundaries + [len(peaks)]
    return [(s, e) for s, e in zip(starts, ends)
            if e - s >= RSA_MIN_HALF_PERIOD * RSA_MIN_CYCLES * 2 + RSA_BASELINE_WINDOW]


# ── Vagal arrest detection ────────────────────────────────────────────────────

def detect_vagal_arrests(
    peaks: pd.DataFrame,
    min_pause_ms: float = ARREST_MIN_PAUSE_MS,
    min_ratio: float = ARREST_MIN_RATIO,
) -> pd.DataFrame:
    """Detect vagal arrests: single-beat pauses with gradual recovery.

    Uses vectorized pandas operations for the candidate-selection pass
    (fast even over 54M rows), then a small Python loop only for the
    handful of actual candidates to characterise their recovery.

    Returns one row per detected arrest with:
      onset_peak_id, onset_time, rr_ms, local_baseline_ms, pause_ratio,
      recovery_beats, recovery_duration_ms
    """
    logger.info("Detecting vagal arrests  (min_pause=%.0f ms, min_ratio=%.2f)...",
                min_pause_ms, min_ratio)

    rr        = peaks["rr_ms"].values
    boundary  = peaks["is_boundary"].values
    peak_ids  = peaks["peak_id"].values
    timestamps = peaks["timestamp_ns"].values

    # ── Backward-looking baseline ──────────────────────────────────────────
    # shift(1) means "median of the PRECEDING window, excluding the current beat".
    # This prevents the pause itself from contaminating its own baseline.
    baseline = (
        pd.Series(rr)
        .rolling(ARREST_BASELINE_WINDOW, min_periods=3)
        .median()
        .shift(1)
        .values
    )

    # ── Vectorised candidate selection ────────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where((baseline > 0) & ~np.isnan(baseline), rr / baseline, np.nan)

    # Rolling max ratio over the preceding 4 beats: if any of the 4 preceding
    # beats were already elevated, this is sustained bradycardia, not an arrest.
    pre_max_ratio = (
        pd.Series(ratio)
        .rolling(4, min_periods=1)
        .max()
        .shift(1)
        .values
    )

    is_candidate = (
        ~boundary
        & ~np.isnan(rr)
        & ~np.isnan(baseline)
        & ~np.isnan(ratio)
        & (rr >= min_pause_ms)
        & (ratio >= min_ratio)
        & (baseline <= ARREST_MAX_BASELINE_MS)
        & (np.isnan(pre_max_ratio) | (pre_max_ratio <= 1.5))
    )

    candidate_idxs = np.where(is_candidate)[0]
    logger.info("  %d candidate pauses to characterise...", len(candidate_idxs))

    # ── Recovery characterisation (Python loop over small candidate set) ──
    arrests = []
    for i in candidate_idxs:
        rec_beats = 0
        rec_ms    = 0.0
        base_i    = float(baseline[i])
        for j in range(i + 1, min(len(rr), i + ARREST_MAX_RECOVERY_BEATS + 1)):
            if boundary[j] or np.isnan(rr[j]):
                break
            rec_beats += 1
            rec_ms    += float(rr[j])
            if base_i > 0 and rr[j] <= base_i * (1.0 + ARREST_RECOVERY_THRESHOLD):
                break  # back to baseline

        ts = int(timestamps[i])
        arrests.append({
            "onset_peak_id":       int(peak_ids[i]),
            "onset_timestamp_ns":  ts,
            "onset_time":          pd.Timestamp(ts, unit="ns").isoformat(),
            "rr_ms":               float(rr[i]),
            "local_baseline_ms":   float(base_i),
            "pause_ratio":         float(ratio[i]),
            "recovery_beats":      rec_beats,
            "recovery_duration_ms": rec_ms,
        })

    df = pd.DataFrame(arrests) if arrests else pd.DataFrame(
        columns=["onset_peak_id", "onset_timestamp_ns", "onset_time",
                 "rr_ms", "local_baseline_ms", "pause_ratio",
                 "recovery_beats", "recovery_duration_ms"]
    )
    logger.info("  ✓  %d vagal arrests detected", len(df))
    if len(df):
        logger.info("    RR:    %.0f – %.0f ms  (%.1f – %.1f s)",
                    df.rr_ms.min(), df.rr_ms.max(),
                    df.rr_ms.min()/1000, df.rr_ms.max()/1000)
        logger.info("    Ratio: %.2f – %.2fx  (vs local baseline)",
                    df.pause_ratio.min(), df.pause_ratio.max())
    return df


# ── RSA detection ─────────────────────────────────────────────────────────────

def detect_rsa_events(
    peaks: pd.DataFrame,
    min_amplitude_ms: float = RSA_MIN_AMPLITUDE_MS,
    min_cycles: int = RSA_MIN_CYCLES,
) -> pd.DataFrame:
    """Detect rhythmic RSA via alternating slow/fast RR extrema.

    Strategy per session:
      1. Compute local baseline (centered rolling median over RSA_BASELINE_WINDOW beats).
      2. Compute deviation = RR - baseline.
      3. Find peaks (slow periods) and troughs (fast periods) in the lightly
         smoothed deviation using scipy.signal.find_peaks.
      4. Walk through alternating extrema; accumulate consecutive valid half-cycles
         into RSA events, breaking when period or amplitude falls out of range.

    Returns one row per detected RSA event.
    """
    logger.info("Detecting RSA events  (min_amplitude=%.0f ms, min_cycles=%d)...",
                min_amplitude_ms, min_cycles)

    rr_all     = peaks["rr_ms"].values
    peak_ids   = peaks["peak_id"].values
    timestamps = peaks["timestamp_ns"].values

    # Global centered baseline (computed once for speed)
    baseline_all = (
        pd.Series(rr_all)
        .rolling(RSA_BASELINE_WINDOW, center=True, min_periods=5)
        .median()
        .values
    )
    dev_all = rr_all - baseline_all

    sessions = _session_slices(peaks)
    logger.info("  Processing %d recording sessions...", len(sessions))

    all_events: list[dict] = []

    for sess_start, sess_end in sessions:
        dev   = dev_all[sess_start:sess_end]
        pids  = peak_ids[sess_start:sess_end]
        ts    = timestamps[sess_start:sess_end]

        # Light smoothing to reduce single-beat noise; NaN edges → 0
        dev_s = (
            pd.Series(dev)
            .rolling(3, center=True, min_periods=1)
            .mean()
            .fillna(0)
            .values
        )

        # scipy.signal.find_peaks: height is half the target amplitude so each
        # *pair* of extrema totals at least min_amplitude_ms when averaged.
        height = min_amplitude_ms * 0.4
        pk, _ = find_peaks( dev_s,  height=height, distance=RSA_MIN_HALF_PERIOD)
        tr, _ = find_peaks(-dev_s,  height=height, distance=RSA_MIN_HALF_PERIOD)

        if len(pk) < min_cycles or len(tr) < min_cycles:
            continue

        # Merge slow-period peaks and fast-period troughs into one sorted list.
        # Each entry: (local_index, type, amplitude_magnitude)
        extrema: list[tuple[int, str, float]] = (
            [(int(i), "slow", float( dev_s[i])) for i in pk] +
            [(int(i), "fast", float(-dev_s[i])) for i in tr]
        )
        extrema.sort()

        # Walk through extrema, accumulating valid alternating runs.
        event_ex: list[tuple[int, str, float]] = []

        def _flush(ex: list) -> None:
            """Convert accumulated extrema to an event record if long enough."""
            n_half = len(ex) - 1
            n_full = n_half // 2
            if n_full < min_cycles:
                return
            s_i, e_i = ex[0][0], ex[-1][0]
            amps    = [(ex[k][2] + ex[k+1][2]) / 2.0 for k in range(n_half)]
            periods = [ex[k+1][0] - ex[k][0]         for k in range(n_half)]
            mean_amp = float(np.mean(amps))
            if mean_amp < min_amplitude_ms / 2.0:
                return
            ts_s, ts_e = int(ts[s_i]), int(ts[e_i])
            all_events.append({
                "start_peak_id":          int(pids[s_i]),
                "end_peak_id":            int(pids[e_i]),
                "start_time":             pd.Timestamp(ts_s, unit="ns").isoformat(),
                "end_time":               pd.Timestamp(ts_e, unit="ns").isoformat(),
                "duration_s":             (ts_e - ts_s) / 1e9,
                "beat_count":             e_i - s_i + 1,
                "n_cycles":               n_full,
                "mean_amplitude_ms":      mean_amp,
                "max_amplitude_ms":       float(max(amps)),
                "mean_half_period_beats": float(np.mean(periods)),
            })

        for idx, etype, amp in extrema:
            if not event_ex:
                event_ex.append((idx, etype, amp))
                continue

            prev_idx, prev_type, prev_amp = event_ex[-1]
            period = idx - prev_idx

            # Break conditions: same type, period out of range, or amplitude too small
            if (prev_type == etype
                    or period < RSA_MIN_HALF_PERIOD
                    or period > RSA_MAX_HALF_PERIOD
                    or (prev_amp + amp) / 2.0 < min_amplitude_ms * 0.4):
                _flush(event_ex)
                event_ex = [(idx, etype, amp)]
            else:
                event_ex.append((idx, etype, amp))

        _flush(event_ex)  # finalise any open event at end of session

    df = pd.DataFrame(all_events) if all_events else pd.DataFrame(
        columns=["start_peak_id", "end_peak_id", "start_time", "end_time",
                 "duration_s", "beat_count", "n_cycles",
                 "mean_amplitude_ms", "max_amplitude_ms", "mean_half_period_beats"]
    )
    logger.info("  ✓  %d RSA events detected", len(df))
    if len(df):
        logger.info("    Duration:  %d – %d beats",
                    df.beat_count.min(), df.beat_count.max())
        logger.info("    Amplitude: %.0f – %.0f ms",
                    df.mean_amplitude_ms.min(), df.mean_amplitude_ms.max())
        logger.info("    Cycles:    %d – %d full cycles per event",
                    df.n_cycles.min(), df.n_cycles.max())
    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Algorithmic detection of vagal arrests and RSA events from peaks.parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--peaks", type=Path,
                   default=Path("/Volumes/xHRV/processed/peaks.parquet"),
                   help="Path to peaks.parquet (default: /Volumes/xHRV/processed/peaks.parquet)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/Volumes/xHRV/Accessory/detected_events/"),
                   help="Directory for output CSVs (default: /Volumes/xHRV/Accessory/detected_events/)")
    p.add_argument("--arrest-min-pause-ms", type=float, default=ARREST_MIN_PAUSE_MS,
                   help=f"Min absolute RR (ms) for vagal arrest  (default: {ARREST_MIN_PAUSE_MS})")
    p.add_argument("--arrest-min-ratio", type=float, default=ARREST_MIN_RATIO,
                   help=f"Min RR/baseline ratio for vagal arrest  (default: {ARREST_MIN_RATIO})")
    p.add_argument("--rsa-min-amplitude-ms", type=float, default=RSA_MIN_AMPLITUDE_MS,
                   help=f"Min peak-to-trough amplitude for RSA (ms)  (default: {RSA_MIN_AMPLITUDE_MS})")
    p.add_argument("--rsa-min-cycles", type=int, default=RSA_MIN_CYCLES,
                   help=f"Min full cycles per RSA event  (default: {RSA_MIN_CYCLES})")
    args = p.parse_args()

    if not args.peaks.exists():
        logger.error("peaks.parquet not found: %s", args.peaks)
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    peaks = load_peaks(args.peaks)

    # ── Vagal arrests ──────────────────────────────────────────────────────
    arrests = detect_vagal_arrests(peaks, args.arrest_min_pause_ms, args.arrest_min_ratio)
    arrest_path = args.output_dir / "vagal_arrests.csv"
    arrests.to_csv(arrest_path, index=False)
    logger.info("Saved → %s", arrest_path)

    # ── RSA events ─────────────────────────────────────────────────────────
    rsa = detect_rsa_events(peaks, args.rsa_min_amplitude_ms, args.rsa_min_cycles)
    rsa_path = args.output_dir / "rsa_events.csv"
    rsa.to_csv(rsa_path, index=False)
    logger.info("Saved → %s", rsa_path)

    # ── Summary ────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Physio Event Detection — Summary")
    print(sep)
    print(f"  Total peaks analysed:      {len(peaks):>10,}")
    print()
    print(f"  Vagal arrests detected:    {len(arrests):>10,}")
    if len(arrests):
        print(f"    Pause range:             "
              f"{arrests.rr_ms.min():.0f} – {arrests.rr_ms.max():.0f} ms  "
              f"({arrests.rr_ms.min()/1000:.1f} – {arrests.rr_ms.max()/1000:.1f} s)")
        print(f"    Median pause ratio:      {arrests.pause_ratio.median():.2f}×")
        print(f"    Median recovery beats:   {arrests.recovery_beats.median():.1f}")
    print()
    print(f"  RSA events detected:       {len(rsa):>10,}")
    if len(rsa):
        print(f"    Duration range:          "
              f"{rsa.beat_count.min()} – {rsa.beat_count.max()} beats")
        print(f"    Amplitude range:         "
              f"{rsa.mean_amplitude_ms.min():.0f} – {rsa.mean_amplitude_ms.max():.0f} ms")
        print(f"    Total RSA beats:         {rsa.beat_count.sum():,}")
    print()
    print(f"  Output: {args.output_dir.resolve()}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
