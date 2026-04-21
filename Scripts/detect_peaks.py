#!/usr/bin/env python3
"""
ecgclean.detect_peaks
======================
R-peak detection for raw ECG CSV files.

Runs an ensemble of two detectors:
  1. Pan-Tompkins variant  (scipy: bandpass → diff → square → MWI)
  2. SWT-based detector    (PyWavelets: stationary wavelet transform energy)

Results from both are merged (union) with a configurable tolerance window,
then written as peak CSV files ready for data_pipeline.py.

Designed for non-stationary data with variable heart rate (e.g. POTS):
  - No global threshold — uses local percentile-adaptive thresholds per chunk
  - Chunked processing: files of any size processed in N-minute windows
    with overlap at boundaries to avoid missing peaks there
  - Multi-detector ensemble reduces missed beats without requiring retraining

Input format (same as existing ECG CSVs):
    DateTime (epoch ms integer)   ECG (float mV)
    1770893409028                 -1.701
    ...

Output (one peak CSV per input file, in --output-dir):
    peak_id (epoch ms)   source   is_added_peak
    1770893409028        detected False
    ...

Usage
-----
    python ecgclean/detect_peaks.py \\
        --ecg-dir path/to/raw_ecg/ \\
        --output-dir path/to/peaks/ \\
        [--fs 125] \\
        [--method ensemble|ptompkins|swt] \\
        [--chunk-min 5] \\
        [--refractory-ms 250] \\
        [--merge-tolerance-ms 50]
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, filtfilt, find_peaks

from config import (
    SAMPLE_RATE_HZ,
    REFRACTORY_MS,
    MERGE_TOLERANCE_MS,
    PEAK_CHUNK_MIN,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("ecgclean.detect_peaks")

# Pan-Tompkins / SWT find the peak of an energy envelope, which is biased
# toward the QRS upslope rather than the R-peak apex.  After detection,
# every candidate is snapped to argmax(|ecg|) within this window.
_SNAP_RADIUS_SEC: float = 0.060   # ±60 ms

def _snap_to_extremum(
    peaks: np.ndarray,
    ecg: np.ndarray,
    snap_radius: int,
) -> np.ndarray:
    """Snap each peak index to the nearest local |ECG| maximum.

    Works for both normal and inverted polarity because it uses absolute value.
    Uses the ORIGINAL (unfiltered) ECG so bandpass ringing doesn't shift the apex.

    Args:
        peaks: Sample indices from find_peaks (in the energy envelope).
        ecg:   Original ECG signal (same length; not bandpass-filtered).
        snap_radius: Maximum shift in samples.

    Returns:
        Snapped sample indices (same length as peaks).
    """
    if len(peaks) == 0:
        return peaks
    n = len(ecg)
    snapped = np.empty_like(peaks)
    for i, p in enumerate(peaks):
        lo = max(0, int(p) - snap_radius)
        hi = min(n, int(p) + snap_radius + 1)
        snapped[i] = lo + int(np.argmax(np.abs(ecg[lo:hi])))
    return snapped


# ═══════════════════════════════════════════════════════════════════════════ #
#  Detector implementations                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def _bandpass(ecg: np.ndarray, fs: float, lo: float = 5.0, hi: float = 18.0) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    lo_norm = max(lo / nyq, 0.001)
    hi_norm = min(hi / nyq, 0.999)
    b, a = butter(2, [lo_norm, hi_norm], btype="band")
    return filtfilt(b, a, ecg)


def detect_pantompkins(ecg: np.ndarray, fs: float, refractory_ms: float = 250.0) -> np.ndarray:
    """Pan-Tompkins variant R-peak detector.

    Steps:
        1. Bandpass 5-18 Hz
        2. First derivative
        3. Squaring
        4. Moving window integration (150 ms window)
        5. Adaptive threshold (local 70th percentile) + find_peaks
        6. Snap each index to nearest local |ECG| maximum (corrects MWI upslope bias)

    Returns sample indices of detected R-peaks.
    """
    filtered = _bandpass(ecg, fs)
    diff = np.gradient(filtered)
    squared = diff ** 2
    win = max(1, int(0.150 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    threshold = np.percentile(integrated, 70)
    min_distance = max(1, int(refractory_ms / 1000.0 * fs))
    peaks, _ = find_peaks(integrated, distance=min_distance, height=threshold)
    snap_radius = max(1, int(_SNAP_RADIUS_SEC * fs))
    return _snap_to_extremum(peaks, ecg, snap_radius)


def detect_swt(ecg: np.ndarray, fs: float, refractory_ms: float = 250.0) -> np.ndarray:
    """SWT-based R-peak detector.

    Uses levels 2 and 3 of a Stationary Wavelet Transform (db4 wavelet).
    At 125 Hz these levels capture ~8–31 Hz — the core QRS energy band.
    Detail coefficients are squared and summed to form an energy envelope.

    Returns sample indices of detected R-peaks.
    """
    # SWT requires input length divisible by 2^level
    level = 4
    divisor = 2 ** level
    orig_len = len(ecg)
    pad_len = int(np.ceil(orig_len / divisor)) * divisor
    padded = np.pad(_bandpass(ecg, fs), (0, pad_len - orig_len), mode="edge")

    # coeffs[i] = (cA_level_i+1, cD_level_i+1)  for i in 0..level-1
    # At 125 Hz: level 2 detail (D2) ≈ 15.6–31.25 Hz, level 3 detail (D3) ≈ 7.8–15.6 Hz
    # This covers the core QRS band of ~8-31 Hz
    coeffs = pywt.swt(padded, "db4", level=level)
    cD2 = coeffs[2][1][:orig_len]
    cD3 = coeffs[1][1][:orig_len]
    energy = cD2 ** 2 + cD3 ** 2

    threshold = np.percentile(energy, 75)
    min_distance = max(1, int(refractory_ms / 1000.0 * fs))
    peaks, _ = find_peaks(energy, distance=min_distance, height=threshold)
    snap_radius = max(1, int(_SNAP_RADIUS_SEC * fs))
    return _snap_to_extremum(peaks, ecg, snap_radius)


def _merge_peak_sets(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    tolerance_samples: int,
) -> np.ndarray:
    """Union of two peak sets, deduplicating within tolerance_samples."""
    if len(peaks_a) == 0:
        return peaks_b
    if len(peaks_b) == 0:
        return peaks_a
    combined = np.sort(np.concatenate([peaks_a, peaks_b]))
    kept = [combined[0]]
    for p in combined[1:]:
        if p - kept[-1] > tolerance_samples:
            kept.append(p)
    return np.array(kept, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Chunked file processing                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

def _detect_chunk(
    ecg: np.ndarray,
    fs: float,
    method: str,
    refractory_ms: float,
    merge_tolerance_ms: float,
) -> np.ndarray:
    if method == "ptompkins":
        return detect_pantompkins(ecg, fs, refractory_ms)
    elif method == "swt":
        return detect_swt(ecg, fs, refractory_ms)
    else:  # ensemble
        pa = detect_pantompkins(ecg, fs, refractory_ms)
        pb = detect_swt(ecg, fs, refractory_ms)
        tol = max(1, int(merge_tolerance_ms / 1000.0 * fs))
        return _merge_peak_sets(pa, pb, tol)


def process_file(
    ecg_path: Path,
    output_dir: Path,
    fs: float,
    method: str,
    chunk_min: float,
    overlap_sec: float,
    refractory_ms: float,
    merge_tolerance_ms: float,
) -> int:
    """Detect R-peaks in one ECG CSV and write a peak CSV."""
    log.info("Processing: %s", ecg_path.name)

    df = pd.read_csv(ecg_path)

    ts_col = next(
        (c for c in df.columns if c.lower() in ("datetime", "timestamp", "time")),
        df.columns[0],
    )
    ecg_col = next(
        (c for c in df.columns if c.lower() in ("ecg", "ecg_amplitude", "amplitude", "value")),
        df.columns[1],
    )

    timestamps_ms = df[ts_col].values.astype(np.int64)
    ecg_signal = df[ecg_col].values.astype(np.float64)

    valid = np.isfinite(ecg_signal)
    timestamps_ms = timestamps_ms[valid]
    ecg_signal = ecg_signal[valid]

    # Convert mV → µV (Polar H10 CSVs are in millivolts)
    ecg_signal = ecg_signal * 1000.0

    if len(ecg_signal) == 0:
        log.warning("  No valid samples in %s — skipping", ecg_path.name)
        return 0

    log.info("  %d samples  (%.1f hours)", len(ecg_signal), len(ecg_signal) / fs / 3600)

    # ── Chunked detection with boundary overlap ───────────────────────────
    chunk_samples = int(chunk_min * 60 * fs)
    overlap_samples = int(overlap_sec * fs)
    n = len(ecg_signal)
    all_peak_ts_ms: list[int] = []
    chunk_start = 0

    while chunk_start < n:
        chunk_end = min(chunk_start + chunk_samples, n)
        lo = max(0, chunk_start - overlap_samples)
        hi = min(n, chunk_end + overlap_samples)

        chunk_ecg = ecg_signal[lo:hi]
        chunk_ts  = timestamps_ms[lo:hi]

        if len(chunk_ecg) < int(0.5 * fs):
            chunk_start = chunk_end
            continue

        raw_peaks = _detect_chunk(chunk_ecg, fs, method, refractory_ms, merge_tolerance_ms)

        # Only keep peaks that fall inside the core (non-overlap) window
        core_lo = chunk_start - lo
        core_hi = chunk_end - lo
        for p in raw_peaks:
            if core_lo <= p < core_hi:
                all_peak_ts_ms.append(int(chunk_ts[p]))

        chunk_start = chunk_end

    if not all_peak_ts_ms:
        log.warning("  No peaks detected in %s", ecg_path.name)
        return 0

    # ── Global dedup + refractory enforcement ─────────────────────────────
    peak_ts = np.sort(np.array(all_peak_ts_ms, dtype=np.int64))

    # Dedup (boundary artifacts that slipped through)
    tol_ms = int(merge_tolerance_ms)
    kept = [peak_ts[0]]
    for t in peak_ts[1:]:
        if t - kept[-1] > tol_ms:
            kept.append(t)
    peak_ts = np.array(kept, dtype=np.int64)

    # Global refractory
    min_rr_ms = int(refractory_ms)
    final = [peak_ts[0]]
    for t in peak_ts[1:]:
        if t - final[-1] >= min_rr_ms:
            final.append(t)
    peak_ts = np.array(final, dtype=np.int64)

    med_rr = float(np.median(np.diff(peak_ts))) if len(peak_ts) > 1 else 0.0
    log.info(
        "  Detected %d peaks  (median HR ≈ %.0f bpm)",
        len(peak_ts),
        60_000 / med_rr if med_rr > 0 else 0,
    )

    # ── Write output ──────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "peak_id":       peak_ts,    # epoch ms — matches data_pipeline.py expectation
        "source":        "detected",
        "is_added_peak": False,
    })
    
    first_dt = dt.datetime.fromtimestamp(timestamps_ms[0] / 1000)
    last_dt = dt.datetime.fromtimestamp(timestamps_ms[-1] / 1000)
    
    def format_dt(d: dt.datetime) -> str:
        # e.g., 1-1, 7.52AM
        time_str = d.strftime("%-I.%M%p").upper()
        return f"{d.month}-{d.day}, {time_str}"
        
    out_filename = f"{format_dt(first_dt)} - {format_dt(last_dt)}.csv"
    out_path = output_dir / out_filename
    out_df.to_csv(out_path, index=False)
    log.info("  Saved → %s", out_path.name)
    
    # Set file creation and modification dates based on the recording's first and last rows
    if len(timestamps_ms) >= 2:
        try:
            # Get datetime objects for first and last timestamps
            first_dt = dt.datetime.fromtimestamp(timestamps_ms[0] / 1000)
            last_dt = dt.datetime.fromtimestamp(timestamps_ms[-1] / 1000)
            
            # Set access + modification time to the last row's timestamp
            os.utime(out_path, (last_dt.timestamp(), last_dt.timestamp()))
            
            # Set creation date on macOS to the first row's timestamp
            subprocess.run(
                ["SetFile", "-d", first_dt.strftime("%m/%d/%Y %H:%M:%S"), str(out_path)],
                check=True,
                capture_output=True
            )
            log.info("  Set file OS creation and modification timestamps successfully.")
        except Exception as time_e:
            log.warning(f"  Warning: Could not set OS file timestamps for {out_filename}: {time_e}")

    return len(peak_ts)


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="detect_peaks.py",
        description="Ensemble R-peak detector for non-stationary wearable ECG.",
    )
    p.add_argument("--ecg-dir", type=Path, required=True,
                   help="Directory of raw ECG CSV files")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write peak CSV files")
    p.add_argument("--fs", type=float, default=SAMPLE_RATE_HZ,
                   help="Sampling rate Hz (default: 125 for Polar H10 — do NOT use 130 or 256)")
    p.add_argument("--method", choices=["ensemble", "ptompkins", "swt"],
                   default="ensemble",
                   help="Detection method (default: ensemble)")
    p.add_argument("--chunk-min", type=float, default=PEAK_CHUNK_MIN,
                   help="Processing chunk size minutes (default: 5)")
    p.add_argument("--overlap-sec", type=float, default=5.0,
                   help="Overlap on each chunk boundary side in seconds (default: 5)")
    p.add_argument("--refractory-ms", type=float, default=REFRACTORY_MS,
                   help="Minimum RR interval ms (default: 250 → max 240 bpm)")
    p.add_argument("--merge-tolerance-ms", type=float, default=MERGE_TOLERANCE_MS,
                   help="Window for merging ensemble peaks ms (default: 50)")
    p.add_argument("--max-files", type=int, default=None,
                   help="Process only the first N ECG files (for debugging; default: all)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.ecg_dir.is_dir():
        log.error("ECG directory not found: %s", args.ecg_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(args.ecg_dir.glob("*.csv"))

    if not csv_files:
        log.error("No CSV files found in %s", args.ecg_dir)
        sys.exit(1)

    if args.max_files is not None:
        csv_files = csv_files[:args.max_files]
        log.info("--max-files %d: processing subset of %d file(s)", args.max_files, len(csv_files))

    log.info("Found %d ECG file(s) | method=%s | fs=%.0f Hz",
             len(csv_files), args.method, args.fs)

    total_peaks = 0
    for i, f in enumerate(csv_files, 1):
        log.info("[%d/%d] %s", i, len(csv_files), f.name)
        total_peaks += process_file(
            ecg_path=f,
            output_dir=args.output_dir,
            fs=args.fs,
            method=args.method,
            chunk_min=args.chunk_min,
            overlap_sec=args.overlap_sec,
            refractory_ms=args.refractory_ms,
            merge_tolerance_ms=args.merge_tolerance_ms,
        )

    log.info("Done. %d total peaks across %d file(s)", total_peaks, len(csv_files))


if __name__ == "__main__":
    main()
