#!/usr/bin/env python3
"""
ecgclean/data_pipeline.py — Step 1: Data Ingestion & Canonical Table Emission

Reads raw ECG CSVs, R-peak CSVs, and a directory of three annotation parquet
files (``beats.parquet``, ``segments.parquet``, ``bad_regions.parquet``) to
produce four canonical Parquet tables consumed by all downstream pipeline
stages:

  ecg_samples.parquet  — raw ECG time series with segment assignments
  peaks.parquet        — deduplicated R-peak catalog
  labels.parquet       — beat-level artifact/quality labels
  segments.parquet     — segment-level quality labels

Annotation input contract (``--annotations`` is a directory):

  beats.parquet
    Columns: timestamp_ms (int64), segment_idx (int32), label
    (clean | artifact | physio), subtype (auto | added | manual |
    spurious | interpolate), beat_training_eligible (bool).
    Scope: training-eligible beats only — beats from unreviewed
    or bad-region time windows are pre-excluded upstream.

  segments.parquet
    Columns: segment_idx (int32), start_ms, end_ms,
    review_status (reviewed | unreviewed),
    segment_quality_label (usable | unclean | unusable | unknown),
    beat_training_eligible_segment (bool),
    segment_quality_training_eligible (bool).

  bad_regions.parquet
    Columns: segment_idx, start_ms, end_ms. Each row is a bad
    time interval inside an otherwise-usable segment.

Beat label mapping (beats.parquet → labels.parquet):
  beats.label "clean"    → labels.label "clean"
  beats.label "artifact" → labels.label "artifact"  (subtype carries
                                                     spurious vs interpolate)
  beats.label "physio"   → labels.label "phys_event"
  Peaks not present in beats.parquet → label "unknown", subtype "",
  reviewed=False.

Segment quality mapping (segments.parquet → segments.parquet output):
  segment_quality_label "usable"   + RMSSD < 50  → quality_label "clean"
  segment_quality_label "usable"   + 50 ≤ RMSSD < 250 → quality_label "unknown"
  segment_quality_label "usable"   + RMSSD ≥ 250 → quality_label "noisy_ok"
  segment_quality_label "unclean"  → quality_label "noisy_ok"
  segment_quality_label "unusable" → quality_label "bad"
  segment_quality_label "unknown"  → quality_label "unknown"

Memory model: ECG CSVs are NEVER fully loaded into RAM.
  1. scan_recording_start_ns() — reads only the first row of each ECG file
     to find the global epoch start (min timestamp across all sources).
  2. stream_ecg_to_parquet()   — reads each file individually, computes
     segment_idx, writes rows to ecg_samples.parquet via ParquetWriter,
     and accumulates a lightweight {segment_idx: (min_ts, max_ts)} dict.
  The dict replaces the in-memory ecg_samples DataFrame everywhere
  build_segments() and validate_outputs() previously required it.

Usage:
    python data_pipeline.py \\
        --ecg-dir data/raw_ecg/ \\
        --peaks-dir data/peaks/ \\
        --annotations Data/Annotations/INPUT/ \\
        --output-dir data/processed/
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from calendar import timegm
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .utils.pipeline_logging import setup_logger, add_logging_args
except ImportError:
    from utils.pipeline_logging import setup_logger, add_logging_args

try:
    from .config import (
        SEGMENT_DURATION_MS,
        DEDUP_TOLERANCE_MS,
        ANNOTATION_MATCH_TOLERANCE_MS,
        MIN_VALID_TIMESTAMP_MS,
        ECG_CHUNK_SIZE,
    )
except ImportError:
    from config import (
        SEGMENT_DURATION_MS,
        DEDUP_TOLERANCE_MS,
        ANNOTATION_MATCH_TOLERANCE_MS,
        MIN_VALID_TIMESTAMP_MS,
        ECG_CHUNK_SIZE,
    )

logger = logging.getLogger("ecgclean.data_pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# TIMESTAMP PARSING
# ═══════════════════════════════════════════════════════════════════════════════


def parse_iso_to_ns(iso_str: str) -> int | None:
    """Parse an ISO 8601 datetime string to epoch milliseconds (UTC).

    Handles up to nanosecond precision in the fractional-seconds part.
    All timestamps are interpreted as UTC.

    Args:
        iso_str: ISO 8601 string, e.g. "2025-02-04T18:01:42.399369220"

    Returns:
        Epoch milliseconds as int64, or None if parsing fails.
    """
    try:
        # Split date from time on 'T' or space
        if "T" in iso_str:
            date_part, time_part = iso_str.split("T", 1)
        elif " " in iso_str:
            date_part, time_part = iso_str.split(" ", 1)
        else:
            return None

        # Strip timezone suffixes (treat everything as UTC)
        for suffix in ("Z", "+00:00"):
            if time_part.endswith(suffix):
                time_part = time_part[: -len(suffix)]

        # Separate fractional seconds
        if "." in time_part:
            time_main, frac = time_part.split(".", 1)
            # Pad or truncate to 9 digits (nanoseconds)
            frac_ns = int(frac.ljust(9, "0")[:9])
        else:
            time_main = time_part
            frac_ns = 0

        # Parse components
        year, month, day = (int(x) for x in date_part.split("-"))
        hour, minute, second = (int(x) for x in time_main.split(":"))

        # timegm interprets the tuple as UTC (unlike mktime which uses local TZ)
        epoch_s = timegm((year, month, day, hour, minute, second, 0, 0, 0))
        return epoch_s * 1000 + frac_ns // 1_000_000
    except Exception:
        return None


def parse_timestamp_to_ns(value: Any) -> int | None:
    """Convert a timestamp value of any supported format to epoch milliseconds.

    Supported formats:
      - int / np.integer: epoch milliseconds
      - float: epoch milliseconds (NaN → None)
      - str: ISO 8601 datetime, or numeric string of epoch ms

    Args:
        value: Raw timestamp in any supported format.

    Returns:
        Epoch milliseconds as int64, or None if conversion fails.
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        ns = parse_iso_to_ns(value)
        if ns is not None:
            return ns
        # Fallback: try as numeric string
        try:
            return int(float(value))
        except (ValueError, OverflowError):
            return None
    return None


def parse_timestamp_list(raw_list: list[Any]) -> np.ndarray:
    """Convert a list of mixed-format timestamps to a sorted int64 array of epoch ns.

    Values that cannot be parsed are skipped with a warning.

    Args:
        raw_list: List of timestamps in any supported format.

    Returns:
        Sorted numpy int64 array of epoch nanoseconds.
    """
    results: list[int] = []
    for val in raw_list:
        ns = parse_timestamp_to_ns(val)
        if ns is not None:
            results.append(ns)
        else:
            logger.warning("Could not parse timestamp value: %r", val)
    arr = np.array(results, dtype=np.int64)
    arr.sort()
    return arr


def timestamps_match_with_tolerance(
    query: np.ndarray,
    reference: np.ndarray,
    tolerance_ns: int = DEDUP_TOLERANCE_MS,
) -> np.ndarray:
    """Return a boolean mask: True for each query timestamp within tolerance of any reference.

    Uses binary search for O(n log m) performance.

    Args:
        query: Sorted int64 array of timestamps to check.
        reference: Sorted int64 array of reference timestamps.
        tolerance_ns: Maximum distance in milliseconds for a match.

    Returns:
        Boolean array of same length as query.
    """
    if len(reference) == 0:
        return np.zeros(len(query), dtype=bool)

    ref_sorted = np.sort(reference)
    idx = np.searchsorted(ref_sorted, query, side="left")

    idx_left = np.clip(idx - 1, 0, len(ref_sorted) - 1)
    idx_right = np.clip(idx, 0, len(ref_sorted) - 1)

    dist_left = np.abs(query - ref_sorted[idx_left])
    dist_right = np.abs(query - ref_sorted[idx_right])

    min_dist = np.minimum(dist_left, dist_right)
    return min_dist <= tolerance_ns


def nearest_indices_and_distances(
    query_ts: np.ndarray,
    reference_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest reference index, absolute distance, and signed delta.

    ``query_ts`` and ``reference_ts`` are millisecond epoch timestamps. The
    signed delta is ``nearest_reference_ts - query_ts``.
    """
    query_ts = query_ts.astype(np.int64, copy=False)
    reference_ts = reference_ts.astype(np.int64, copy=False)
    if len(reference_ts) == 0:
        return (
            np.full(len(query_ts), -1, dtype=np.int64),
            np.full(len(query_ts), np.iinfo(np.int64).max, dtype=np.int64),
            np.full(len(query_ts), np.iinfo(np.int64).max, dtype=np.int64),
        )

    idx = np.searchsorted(reference_ts, query_ts, side="left")
    idx_left = np.clip(idx - 1, 0, len(reference_ts) - 1)
    idx_right = np.clip(idx, 0, len(reference_ts) - 1)
    dist_left = np.abs(query_ts - reference_ts[idx_left])
    dist_right = np.abs(reference_ts[idx_right] - query_ts)
    use_left = dist_left <= dist_right
    nearest_idx = np.where(use_left, idx_left, idx_right).astype(np.int64)
    nearest_dist = np.where(use_left, dist_left, dist_right).astype(np.int64)
    signed_delta = (reference_ts[nearest_idx] - query_ts).astype(np.int64)
    return nearest_idx, nearest_dist, signed_delta


def timestamp_ms_to_iso(value: int | float | None) -> str:
    """Format epoch milliseconds for diagnostics CSVs."""
    if value is None or pd.isna(value):
        return ""
    try:
        return pd.Timestamp(int(value), unit="ms").isoformat()
    except (ValueError, OverflowError):
        return str(value)


def distance_bucket(distance_ms: int) -> str:
    """Human-readable nearest-peak distance bucket."""
    if distance_ms <= ANNOTATION_MATCH_TOLERANCE_MS:
        return f"<={ANNOTATION_MATCH_TOLERANCE_MS}ms"
    if distance_ms <= 160:
        return "81-160ms"
    if distance_ms <= 250:
        return "161-250ms"
    if distance_ms <= 500:
        return "251-500ms"
    if distance_ms <= 1000:
        return "501ms-1s"
    if distance_ms <= 60_000:
        return "1s-60s"
    return ">60s"


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV diagnostic, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def boolish(value: Any) -> bool:
    """Parse common bool-like values from parquet/pandas rows."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "t", "yes", "y"}


def timestamp_ms_to_local_display(value: int | float | None) -> str:
    """Format epoch milliseconds as m/d/yy h:mm:ss.000 am/pm."""
    if value is None or pd.isna(value):
        return ""
    try:
        ts = pd.Timestamp(int(value), unit="ms", tz="UTC").tz_convert("America/New_York")
    except (ValueError, OverflowError):
        return str(value)
    hour = ts.hour % 12 or 12
    suffix = "am" if ts.hour < 12 else "pm"
    return (
        f"{ts.month}/{ts.day}/{ts.year % 100:02d} "
        f"{hour}:{ts.minute:02d}:{ts.second:02d}.{int(ts.microsecond / 1000):03d} {suffix}"
    )


def collapse_segment_diagnostic_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse diagnostic segment rows into contiguous input-segment ranges."""
    if len(df) == 0 or "input_segment_idx" not in df.columns:
        return pd.DataFrame(
            columns=[
                "start_input_segment_idx", "end_input_segment_idx", "count",
                "start_ms", "start_iso", "start_local", "end_ms", "end_iso", "end_local",
            ]
        )

    rows = df.sort_values("input_segment_idx").reset_index(drop=True)
    group_id = (
        rows["input_segment_idx"]
        - pd.Series(np.arange(len(rows)), index=rows.index, dtype=np.int64)
    )
    ranges = (
        rows.assign(_group_id=group_id)
        .groupby("_group_id", sort=False)
        .agg(
            start_input_segment_idx=("input_segment_idx", "min"),
            end_input_segment_idx=("input_segment_idx", "max"),
            count=("input_segment_idx", "size"),
            start_ms=("start_ms", "min"),
            end_ms=("end_ms", "max"),
        )
        .reset_index(drop=True)
    )
    ranges["start_iso"] = ranges["start_ms"].map(timestamp_ms_to_iso)
    ranges["end_iso"] = ranges["end_ms"].map(timestamp_ms_to_iso)
    ranges["start_local"] = ranges["start_ms"].map(timestamp_ms_to_local_display)
    ranges["end_local"] = ranges["end_ms"].map(timestamp_ms_to_local_display)
    return ranges[
        [
            "start_input_segment_idx", "end_input_segment_idx", "count",
            "start_ms", "start_iso", "start_local", "end_ms", "end_iso", "end_local",
        ]
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


# Mapping from beats.parquet `label` values to canonical labels.parquet values.
BEAT_LABEL_MAP: dict[str, str] = {
    "clean": "clean",
    "artifact": "artifact",
    "physio": "phys_event",
}

# Mapping from segments.parquet `segment_quality_label` to canonical
# segments.parquet `quality_label` values consumed by segment_quality.py.
# Note: `usable` maps to `clean` here at the dict level, but build_segments()
# further splits `usable` segments by RMSSD into clean / noisy_ok / unknown.
SEGMENT_QUALITY_MAP: dict[str, str] = {
    "usable": "clean",
    "unclean": "noisy_ok",
    "unusable": "bad",
    "unknown": "unknown",
}


# RMSSD trinary banding for `usable` segments. The "clean" cutoff is
# deliberately conservative — the user empirically validated that <50 ms
# does not let noisy segments leak into the clean pool. The "noisy_ok"
# floor of 250 ms is the symmetric conservative choice on the other
# side. The middle band is intentionally discarded into "unknown" to
# protect class purity at both ends.
SEGMENT_RMSSD_CLEAN_MAX_MS: float = 50.0
SEGMENT_RMSSD_NOISY_MIN_MS: float = 250.0


def compute_segment_rmssd(peaks_df: pd.DataFrame) -> dict[int, float]:
    """Return {segment_idx → RMSSD in ms} computed over all peaks per segment.

    RMSSD is computed from the auto-detected peak series (post-dedup) as
    sqrt(mean(diff(RR)^2)), where RR_i = timestamp_ms[i+1] - timestamp_ms[i].
    A segment with fewer than 3 peaks (so fewer than 2 RR intervals and
    thus zero successive differences) gets NaN.

    No label filtering: artifact and clean peaks alike act as anchors,
    consistent with the design decision to keep RMSSD computation
    label-independent.
    """
    out: dict[int, float] = {}
    if len(peaks_df) == 0:
        return out
    for seg_idx, group in peaks_df.groupby("segment_idx"):
        seg_int = int(seg_idx)
        if len(group) < 3:
            out[seg_int] = float("nan")
            continue
        ts = np.sort(group["timestamp_ms"].to_numpy().astype(np.int64))
        rr = np.diff(ts).astype(np.float64)
        succ_diffs = np.diff(rr)
        if succ_diffs.size == 0:
            out[seg_int] = float("nan")
            continue
        out[seg_int] = float(np.sqrt(np.mean(succ_diffs ** 2)))
    return out


def classify_usable_segment_quality(rmssd_ms: float) -> str:
    """Map a `usable`-input segment's RMSSD to its output quality_label.

    - RMSSD < SEGMENT_RMSSD_CLEAN_MAX_MS (50)        → "clean"
    - SEGMENT_RMSSD_CLEAN_MAX_MS ≤ RMSSD <
      SEGMENT_RMSSD_NOISY_MIN_MS (250)               → "unknown" (discarded)
    - RMSSD ≥ SEGMENT_RMSSD_NOISY_MIN_MS (250)       → "noisy_ok"
    - NaN (insufficient peaks)                       → "unknown" (discarded)
    """
    if rmssd_ms is None or (isinstance(rmssd_ms, float) and math.isnan(rmssd_ms)):
        return "unknown"
    if rmssd_ms < SEGMENT_RMSSD_CLEAN_MAX_MS:
        return "clean"
    if rmssd_ms >= SEGMENT_RMSSD_NOISY_MIN_MS:
        return "noisy_ok"
    return "unknown"


def extract_bad_region_time_ranges(
    bad_regions_df: pd.DataFrame,
) -> list[tuple[int, int, int]]:
    """Return ``(segment_idx, start_ms, end_ms)`` tuples from bad_regions.parquet.

    Each row marks a time-bounded window within an otherwise-usable segment
    that is uninterpretable. Beats whose timestamp falls inside any window
    must be excluded from training (but may still be scored at inference).
    """
    if len(bad_regions_df) == 0:
        return []
    seg = bad_regions_df["segment_idx"].astype(np.int64).to_numpy()
    s = bad_regions_df["start_ms"].astype(np.int64).to_numpy()
    e = bad_regions_df["end_ms"].astype(np.int64).to_numpy()
    result = [(int(seg[i]), int(s[i]), int(e[i])) for i in range(len(seg))]
    n_segs = len({r[0] for r in result})
    logger.info(
        "Bad-region time ranges: %d window(s) across %d segment(s)",
        len(result), n_segs,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FILE LOADERS
# ═══════════════════════════════════════════════════════════════════════════════


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that exists in df."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _process_one_ecg_file(
    csv_path: Path,
    staging_path: Path,
    recording_start_ns: int,
) -> tuple[int, dict[int, tuple[int, int]], bool]:
    """Worker function: write one ECG CSV to its staging parquet (or read if already done).

    Must be a module-level function (not a closure) to be picklable for
    ProcessPoolExecutor.  Each call is fully independent — workers write to
    separate staging files with no shared state.

    Returns:
        (n_rows, seg_ranges_for_this_file, was_already_staged)
    """
    seg_ranges: dict[int, tuple[int, int]] = {}

    if staging_path.exists():
        try:
            tbl = pq.read_table(staging_path, columns=["timestamp_ms", "segment_idx"])
            ts_arr = tbl["timestamp_ms"].to_pandas().values
            seg_arr = tbl["segment_idx"].to_pandas().values
            for s in np.unique(seg_arr):
                s_int = int(s)
                mask = seg_arr == s
                seg_ranges[s_int] = (int(ts_arr[mask].min()), int(ts_arr[mask].max()))
            return len(ts_arr), seg_ranges, True
        except Exception:
            staging_path.unlink(missing_ok=True)

    # Not staged (or corrupt) — process from CSV in streaming chunks so that
    # large files (some are 2+ GB / 200M rows) don't exhaust worker RAM.
    # Each 500K-row chunk uses ~8 MB regardless of total file size.

    # ── Step 1: read header to find column names ───────────────────────────
    header_df = pd.read_csv(csv_path, nrows=0)
    ts_col  = _find_column(header_df, ["DateTime", "datetime", "timestamp", "Timestamp", "time", "Time"])
    if ts_col is None:
        ts_col = header_df.columns[0]
    ecg_col = _find_column(header_df, ["ECG", "ecg", "ecg_amplitude", "amplitude", "value"])
    if ecg_col is None:
        ecg_col = header_df.columns[1]

    # ── Step 2: detect timestamp format (string ISO vs numeric epoch ms) ──
    # Only needs the first row; reading 1k rows is cheap and avoids
    # mis-detecting on a malformed first line.
    _fmt_sample = pd.read_csv(csv_path, nrows=1)
    ts_is_string = isinstance(_fmt_sample[ts_col].iloc[0], str)

    # ── Step 3: stream full CSV in chunks → ParquetWriter ─────────────────
    schema  = pa.schema([
        ("timestamp_ms", pa.int64()),
        ("ecg",          pa.float32()),
        ("segment_idx",  pa.int32()),
    ])
    n_total  = 0
    n_bad_ts = 0

    try:
        with pq.ParquetWriter(staging_path, schema, compression="snappy") as writer:
            for chunk in pd.read_csv(csv_path, chunksize=ECG_CHUNK_SIZE):
                if ts_is_string:
                    _ts_ser  = chunk[ts_col].apply(parse_timestamp_to_ns)
                    _c_valid = _ts_ser.notna()
                    ts_c     = _ts_ser[_c_valid].values.astype(np.int64)
                    ecg_c    = pd.to_numeric(chunk.loc[_c_valid, ecg_col], errors="coerce").values.astype(np.float32)
                else:
                    ts_c   = chunk[ts_col].values.astype(np.int64)
                    _raw   = pd.to_numeric(chunk[ecg_col], errors="coerce").values
                    _fin   = np.isfinite(_raw)
                    ts_c   = ts_c[_fin]
                    ecg_c  = _raw[_fin].astype(np.float32)

                # Drop implausible timestamps
                _good = ts_c >= MIN_VALID_TIMESTAMP_MS
                if not _good.all():
                    n_bad_ts += int((~_good).sum())
                    ts_c  = ts_c[_good]
                    ecg_c = ecg_c[_good]

                if len(ts_c) == 0:
                    continue

                seg_c = ((ts_c - recording_start_ns) // SEGMENT_DURATION_MS).astype(np.int32)

                writer.write_table(pa.table({
                    "timestamp_ms": pa.array(ts_c,  type=pa.int64()),
                    "ecg":          pa.array(ecg_c, type=pa.float32()),
                    "segment_idx":  pa.array(seg_c, type=pa.int32()),
                }))

                # Accumulate seg_ranges without storing full arrays
                for s in np.unique(seg_c):
                    s_int = int(s)
                    mask  = seg_c == s
                    s_min = int(ts_c[mask].min())
                    s_max = int(ts_c[mask].max())
                    if s_int in seg_ranges:
                        old_lo, old_hi = seg_ranges[s_int]
                        seg_ranges[s_int] = (min(old_lo, s_min), max(old_hi, s_max))
                    else:
                        seg_ranges[s_int] = (s_min, s_max)

                n_total += len(ts_c)

    except Exception:
        staging_path.unlink(missing_ok=True)
        raise

    if n_bad_ts > 0:
        logger.warning(
            "  [timestamp] %s: dropped %d row(s) with timestamp < 2020-01-01",
            csv_path.name, n_bad_ts,
        )

    return n_total, seg_ranges, False


def scan_recording_start_ns(ecg_dir: Path, peaks_dir: Path) -> int:
    """Quick scan to find the global recording start (minimum timestamp).

    Reads only the first row of each ECG CSV (O(n_files) not O(total_rows))
    and the peak_id column of each peak CSV to find the absolute minimum
    timestamp across all sources.

    Args:
        ecg_dir: Directory containing raw ECG CSV files.
        peaks_dir: Directory containing R-peak CSV files.

    Returns:
        Global recording start as epoch nanoseconds.
    """
    min_ns: int | None = None

    ecg_files = sorted(ecg_dir.glob("*.csv"))
    if not ecg_files:
        logger.error("No ECG CSV files found in %s", ecg_dir)
        sys.exit(1)

    logger.info("Scanning %d ECG files for recording start...", len(ecg_files))
    for path in ecg_files:
        try:
            first = pd.read_csv(path, nrows=1)
            ts_col = _find_column(
                first,
                ["DateTime", "datetime", "timestamp", "Timestamp", "time", "Time"],
            )
            if ts_col is None:
                ts_col = first.columns[0]
            val = first[ts_col].iloc[0]
            ns = parse_timestamp_to_ns(val)
            if ns is not None and ns >= MIN_VALID_TIMESTAMP_MS and (min_ns is None or ns < min_ns):
                min_ns = ns
            elif ns is not None and ns < MIN_VALID_TIMESTAMP_MS:
                logger.warning(
                    "Ignoring implausible first-row timestamp in %s: %s (< 2020-01-01)",
                    path.name, pd.Timestamp(ns, unit="ms"),
                )
        except Exception as exc:
            logger.warning("Could not read first row of %s: %s", path.name, exc)

    for path in sorted(peaks_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path, usecols=["peak_id"])
            peak_min_ns = int(df["peak_id"].astype(np.int64).min())
            if peak_min_ns >= MIN_VALID_TIMESTAMP_MS and (min_ns is None or peak_min_ns < min_ns):
                min_ns = peak_min_ns
        except Exception as exc:
            logger.warning("Could not scan peak file %s: %s", path.name, exc)

    if min_ns is None:
        logger.error("Could not determine recording start — no readable files found")
        sys.exit(1)

    logger.info("Recording start: %s", pd.Timestamp(min_ns, unit="ms"))
    return min_ns


def _salvage_partial_parquet(
    partial_path: Path,
    csv_files: list[Path],
    staging_dir: Path,
) -> int:
    """Attempt to extract row groups from an incomplete parquet into staging files.

    Each write_table() call in stream_ecg_to_parquet() produces exactly one row
    group, so row_group[N] corresponds to csv_files[N].  This lets us recover
    already-processed files without re-reading their CSVs.

    If the file lacks a valid footer (common when a write was interrupted by a
    disk-full error), pq.ParquetFile() will raise and this function returns 0
    — the caller falls back to full reprocessing.

    Args:
        partial_path: Path to the incomplete ecg_samples.parquet.
        csv_files: Sorted list of ECG CSV paths (same order used when writing).
        staging_dir: Directory to write per-file staging parquets into.

    Returns:
        Number of row groups successfully salvaged.
    """
    if not partial_path.exists():
        logger.info("No partial parquet found at %s — starting fresh.", partial_path)
        return 0

    try:
        pf = pq.ParquetFile(partial_path)
        n_groups = pf.metadata.num_row_groups
    except Exception as exc:
        logger.warning(
            "Cannot read partial parquet (likely no footer after disk-full): %s. "
            "Will process all %d files from scratch.",
            exc, len(csv_files),
        )
        return 0

    logger.info(
        "Salvaging %d row group(s) from partial parquet: %s",
        n_groups, partial_path,
    )
    n_salvaged = 0
    for rg in range(min(n_groups, len(csv_files))):
        staging_path = staging_dir / (csv_files[rg].stem + ".parquet")
        if staging_path.exists():
            n_salvaged += 1
            continue
        try:
            table = pf.read_row_group(rg)
            pq.write_table(table, staging_path, compression="snappy")
            n_salvaged += 1
        except Exception as exc:
            logger.warning("Could not salvage row group %d: %s — stopping salvage.", rg, exc)
            break

    logger.info(
        "Salvaged %d/%d file(s) — remaining %d will be streamed from CSV.",
        n_salvaged, len(csv_files), len(csv_files) - n_salvaged,
    )
    return n_salvaged


def stream_ecg_to_parquet(
    ecg_dir: Path,
    output_path: Path,
    recording_start_ns: int,
    resume_partial: Path | None = None,
    n_workers: int = 1,
    max_files: int | None = None,
) -> dict[int, tuple[int, int]]:
    """Stream all ECG CSV files to a single Parquet file, one file at a time.

    Uses a two-phase approach to support crash-resumability:
      Phase 1 — per-file staging: each CSV is written to an individual parquet
        in <output_dir>/_ecg_staging/.  Files already staged are skipped, so a
        killed run can be resumed without re-processing completed files.
      Phase 2 — combine: all staging parquets are merged into ecg_samples.parquet
        in a single streaming pass (one staging file in RAM at a time).
      Cleanup: staging directory is removed after successful combine.

    If resume_partial is given, _salvage_partial_parquet() first extracts row
    groups from the old partial file into staging files before any CSV processing.

    Args:
        ecg_dir: Directory of raw ECG CSV files.
        output_path: Destination path for ecg_samples.parquet.
        recording_start_ns: Global recording start (epoch ns).
        resume_partial: Optional path to an incomplete ecg_samples.parquet from
            a previous run.  Row groups are salvaged into staging so their CSVs
            are not re-processed.

    Returns:
        seg_ranges: {segment_idx → (min_timestamp_ms, max_timestamp_ms)}.
        Used by build_segments() so the full Parquet never needs re-reading.
    """
    schema = pa.schema([
        pa.field("timestamp_ms", pa.int64()),
        pa.field("ecg", pa.float32()),
        pa.field("segment_idx", pa.int32()),
    ])

    csv_files = sorted(ecg_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No ECG CSV files found in %s", ecg_dir)
        sys.exit(1)

    if max_files is not None:
        csv_files = csv_files[:max_files]
        logger.info("--max-files %d: processing subset of %d file(s)", max_files, len(csv_files))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_path.parent / "_ecg_staging"
    staging_dir.mkdir(exist_ok=True)

    # Salvage row groups from a previous partial run before processing any CSVs
    if resume_partial is not None:
        _salvage_partial_parquet(resume_partial, csv_files, staging_dir)

    seg_ranges: dict[int, tuple[int, int]] = {}
    total_rows = 0

    # ── Phase 1: per-file staging (parallel, skips already-staged files) ──
    # Each worker is independent: reads/writes its own staging file, returns
    # its seg_ranges slice. Main process merges results as they complete.
    logger.info(
        "Phase 1: staging %d files with %d worker(s)...", len(csv_files), n_workers
    )
    jobs = {
        (i, path): staging_dir / (path.stem + ".parquet")
        for i, path in enumerate(csv_files, 1)
    }
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _process_one_ecg_file, path, staging_path, recording_start_ns
            ): (i, path)
            for (i, path), staging_path in jobs.items()
        }
        for future in as_completed(futures):
            i, path = futures[future]
            completed += 1
            try:
                n_rows, file_seg_ranges, was_staged = future.result()
            except Exception as exc:
                logger.error(
                    "[%d/%d] FAILED: %s — %s", i, len(csv_files), path.name, exc
                )
                raise

            status = "staged" if was_staged else "streamed"
            logger.info(
                "[%d/%d done] %s (%s, %d rows)",
                completed, len(csv_files), path.name, status, n_rows,
            )
            total_rows += n_rows

            # Merge this file's seg_ranges into the global dict
            for s, (mn, mx) in file_seg_ranges.items():
                if s in seg_ranges:
                    prev_mn, prev_mx = seg_ranges[s]
                    seg_ranges[s] = (min(prev_mn, mn), max(prev_mx, mx))
                else:
                    seg_ranges[s] = (mn, mx)

    # ── Phase 2: combine all staging files → ecg_samples.parquet ──────────
    # Reads one staging file at a time — RAM stays bounded regardless of total size.
    logger.info("Combining %d staging files → %s", len(csv_files), output_path)
    with pq.ParquetWriter(output_path, schema, compression="snappy") as writer:
        for sp in sorted(staging_dir.glob("*.parquet")):
            writer.write_table(pq.read_table(sp))

    # ── Phase 3: clean up staging directory ───────────────────────────────
    for sp in staging_dir.glob("*.parquet"):
        sp.unlink()
    staging_dir.rmdir()

    logger.info(
        "Wrote %d total ECG samples → %s  (%d segments)",
        total_rows, output_path, len(seg_ranges),
    )
    return seg_ranges


def load_peak_csvs(peaks_dir: Path) -> pd.DataFrame:
    """Load and concatenate all R-peak CSV files from a directory.

    Expected columns include peak_id (epoch ms), source, is_added_peak,
    segment_idx, label, and ecg_window_000..063.

    Args:
        peaks_dir: Path to directory containing one or more peak CSV files.

    Returns:
        Raw concatenated DataFrame with original columns.
    """
    csv_files = sorted(peaks_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No peak CSV files found in %s", peaks_dir)
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        logger.info("Loading peak file: %s", path.name)
        df = pd.read_csv(path)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    # Ensure peak_id is int64 (epoch ms in the source data)
    if "peak_id" in result.columns:
        result["peak_id"] = pd.to_numeric(result["peak_id"], errors="coerce")
        result.dropna(subset=["peak_id"], inplace=True)
        result["peak_id"] = result["peak_id"].astype(np.int64)
    else:
        logger.warning("No 'peak_id' column; deriving from DateTime")
        result["peak_id"] = result.iloc[:, 0].apply(
            lambda v: (parse_timestamp_to_ns(v) or 0)
        ).astype(np.int64)

    logger.info("Loaded %d peaks from %d file(s)", len(result), len(csv_files))
    return result


def load_annotation_inputs(
    annotations_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three annotation parquet files from the input directory.

    Expects ``annotations_path`` to be a directory containing exactly:
      - beats.parquet
          Columns: timestamp_ms, segment_idx, label, subtype,
          beat_training_eligible.
          `label` ∈ {clean, artifact, physio}.
          `subtype` ∈ {auto, added, manual, spurious, interpolate}.
      - segments.parquet
          Columns: segment_idx, start_ms, end_ms, review_status,
          segment_quality_label, beat_training_eligible_segment,
          segment_quality_training_eligible.
          `review_status` ∈ {reviewed, unreviewed}.
          `segment_quality_label` ∈ {usable, unclean, unusable, unknown}.
      - bad_regions.parquet
          Columns: segment_idx, start_ms, end_ms.

    The pipeline fails loudly (SystemExit) on any deviation from these
    schemas. No silent fallback to legacy columns or values.

    The legacy artifact_annotation.json input is rejected; pass a
    directory containing the three parquets instead.

    Args:
        annotations_path: Path to the annotation INPUT directory.

    Returns:
        ``(beats_df, segments_df, bad_regions_df)``. If the directory
        does not exist, all three are empty DataFrames with the expected
        columns (preserves the historical "no annotations yet" path).
    """
    if annotations_path.is_file() and annotations_path.suffix.lower() == ".json":
        raise SystemExit(
            f"--annotations was given a JSON file ({annotations_path}). "
            f"This pipeline now consumes the parquet annotation inputs. Pass a "
            f"directory containing beats.parquet, segments.parquet, and "
            f"bad_regions.parquet (e.g. Data/Annotations/INPUT/)."
        )

    beats_cols = ["timestamp_ms", "segment_idx", "label", "subtype", "beat_training_eligible"]
    seg_cols = [
        "segment_idx", "start_ms", "end_ms", "review_status",
        "segment_quality_label", "beat_training_eligible_segment",
        "segment_quality_training_eligible",
    ]
    br_cols = ["segment_idx", "start_ms", "end_ms"]

    allowed_review_status = {"reviewed", "unreviewed"}
    allowed_quality_label = {"usable", "unclean", "unusable", "unknown"}
    allowed_beat_label = {"clean", "artifact", "physio"}
    allowed_subtype = {"auto", "added", "manual", "spurious", "interpolate"}

    if not annotations_path.is_dir():
        logger.warning(
            "Annotation directory not found: %s — proceeding with empty inputs",
            annotations_path,
        )
        return (
            pd.DataFrame(columns=beats_cols),
            pd.DataFrame(columns=seg_cols),
            pd.DataFrame(columns=br_cols),
        )

    beats_path = annotations_path / "beats.parquet"
    segs_path = annotations_path / "segments.parquet"
    br_path = annotations_path / "bad_regions.parquet"

    missing = [p.name for p in (beats_path, segs_path, br_path) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Annotation directory {annotations_path} is missing required "
            f"file(s): {missing}. Expected beats.parquet, segments.parquet, "
            f"bad_regions.parquet."
        )

    beats_df = pd.read_parquet(beats_path)
    segments_df = pd.read_parquet(segs_path)
    bad_regions_df = pd.read_parquet(br_path)

    def _check_columns(name: str, df: pd.DataFrame, expected: list[str]) -> None:
        got = set(df.columns)
        exp = set(expected)
        missing_cols = exp - got
        extra_cols = got - exp
        if missing_cols or extra_cols:
            raise SystemExit(
                f"{name} schema mismatch. "
                f"Missing columns: {sorted(missing_cols) or 'none'}. "
                f"Unexpected columns: {sorted(extra_cols) or 'none'}. "
                f"Expected exactly: {sorted(exp)}."
            )

    _check_columns("beats.parquet", beats_df, beats_cols)
    _check_columns("segments.parquet", segments_df, seg_cols)
    _check_columns("bad_regions.parquet", bad_regions_df, br_cols)

    def _check_vocab(name: str, col: str, series: pd.Series, allowed: set[str]) -> None:
        if len(series) == 0:
            return
        actual = set(series.dropna().astype(str).unique())
        bad = actual - allowed
        if bad:
            raise SystemExit(
                f"{name}: column '{col}' contains unexpected value(s) "
                f"{sorted(bad)}. Allowed: {sorted(allowed)}."
            )

    _check_vocab("beats.parquet", "label", beats_df["label"], allowed_beat_label)
    _check_vocab("beats.parquet", "subtype", beats_df["subtype"], allowed_subtype)
    _check_vocab(
        "segments.parquet", "review_status",
        segments_df["review_status"], allowed_review_status,
    )
    _check_vocab(
        "segments.parquet", "segment_quality_label",
        segments_df["segment_quality_label"], allowed_quality_label,
    )

    logger.info(
        "Loaded annotations from %s: %d beats, %d segments, %d bad_regions",
        annotations_path, len(beats_df), len(segments_df), len(bad_regions_df),
    )
    if len(beats_df) > 0:
        logger.info(
            "Input beat label distribution: %s",
            beats_df["label"].value_counts().to_dict(),
        )
        logger.info(
            "Input beat subtype distribution: %s",
            beats_df["subtype"].value_counts().to_dict(),
        )
    if len(segments_df) > 0:
        logger.info(
            "Input segment_quality_label distribution: %s",
            segments_df["segment_quality_label"].value_counts().to_dict(),
        )

    return beats_df, segments_df, bad_regions_df


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════


def build_peaks(
    peak_csv_df: pd.DataFrame,
    beats_df: pd.DataFrame,
    recording_start_ns: int,
) -> pd.DataFrame:
    """Build the deduplicated peaks canonical table.

    Merges R-peaks from CSV with manually added peaks pulled from
    ``beats.parquet`` (rows where ``subtype == "added"``). Deduplicates
    within ``DEDUP_TOLERANCE_MS``, preferring annotation rows.

    Args:
        peak_csv_df: Raw peak DataFrame from load_peak_csvs.
        beats_df: ``beats.parquet`` content; rows with ``subtype == "added"``
            contribute manually-added peak timestamps.
        recording_start_ns: Epoch ns of recording start (for segment_idx).

    Returns:
        DataFrame with columns: peak_id (int64, auto-increment),
        timestamp_ms (int64), segment_idx (int32), source (str).
    """
    # ── Collect CSV peaks ──────────────────────────────────────────────────
    csv_ts_ns = peak_csv_df["peak_id"].values.astype(np.int64)

    csv_source = (
        peak_csv_df["source"].values
        if "source" in peak_csv_df.columns
        else np.full(len(csv_ts_ns), "detected", dtype=object)
    )

    records: list[dict[str, Any]] = []
    for i in range(len(csv_ts_ns)):
        src = str(csv_source[i]) if not pd.isna(csv_source[i]) else "detected"
        records.append(
            {
                "timestamp_ms": int(csv_ts_ns[i]),
                "source": src,
                "_origin": "csv",
            }
        )

    # ── Collect added peaks from beats.parquet ─────────────────────────────
    if (
        len(beats_df) > 0
        and "subtype" in beats_df.columns
        and "timestamp_ms" in beats_df.columns
    ):
        added_mask = (beats_df["subtype"] == "added").values
        added_ts = beats_df.loc[added_mask, "timestamp_ms"].astype(np.int64).values
        for ts in added_ts:
            records.append(
                {
                    "timestamp_ms": int(ts),
                    "source": "added",
                    "_origin": "annotation",
                }
            )

    # ── Build DataFrame and deduplicate ────────────────────────────────────
    peaks_df = pd.DataFrame(records)
    peaks_df.sort_values("timestamp_ms", inplace=True)
    peaks_df.reset_index(drop=True, inplace=True)

    # Dedup: within DEDUP_TOLERANCE_MS, keep annotation row (lower _priority)
    origin_priority = {"annotation": 0, "csv": 1}
    peaks_df["_priority"] = peaks_df["_origin"].map(origin_priority).fillna(1).astype(int)

    timestamps = peaks_df["timestamp_ms"].values
    keep_mask = np.ones(len(peaks_df), dtype=bool)
    priorities = peaks_df["_priority"].values

    i = 0
    while i < len(peaks_df):
        j = i + 1
        while j < len(peaks_df) and (timestamps[j] - timestamps[i]) <= DEDUP_TOLERANCE_MS:
            j += 1
        if j > i + 1:
            cluster_priorities = priorities[i:j]
            best_offset = int(np.argmin(cluster_priorities))
            for k in range(i, j):
                if k != i + best_offset:
                    keep_mask[k] = False
        i = j

    n_dupes = int((~keep_mask).sum())
    if n_dupes > 0:
        logger.info("Deduplicated %d peaks within %d ms tolerance", n_dupes, DEDUP_TOLERANCE_MS)

    peaks_df = peaks_df[keep_mask].copy()
    peaks_df.drop(columns=["_origin", "_priority"], inplace=True)

    peaks_df["segment_idx"] = (
        (peaks_df["timestamp_ms"] - recording_start_ns) // SEGMENT_DURATION_MS
    ).astype(np.int32)

    peaks_df.reset_index(drop=True, inplace=True)
    peaks_df.insert(0, "peak_id", np.arange(len(peaks_df), dtype=np.int64))

    peaks_df["timestamp_ms"] = peaks_df["timestamp_ms"].astype(np.int64)
    peaks_df["segment_idx"] = peaks_df["segment_idx"].astype(np.int32)

    logger.info(
        "Built peaks table: %d peaks (%d detected, %d added)",
        len(peaks_df),
        (peaks_df["source"] == "detected").sum(),
        (peaks_df["source"] == "added").sum(),
    )
    return peaks_df


def build_labels(
    peaks_df: pd.DataFrame,
    beats_df: pd.DataFrame,
    bad_region_ranges: list[tuple[int, int, int]] | None = None,
    diagnostics_dir: Path | None = None,
) -> pd.DataFrame:
    """Build beat-level labels by matching pipeline peaks against beats.parquet.

    Each peak in ``peaks_df`` is matched to its nearest beat in ``beats_df`` by
    ``timestamp_ms`` within ``ANNOTATION_MATCH_TOLERANCE_MS``. When matched,
    the canonical label and reviewed flag are inherited from the beat row;
    unmatched peaks default to ``label="unknown"``, ``reviewed=False`` (i.e.
    they are not training-eligible).

    Label mapping (from ``beats_df.label``):
      "clean"    → "clean"
      "artifact" → "artifact"   (includes legacy interpolation peaks)
      "physio"   → "phys_event"

    Reviewed flag:
      ``beats.parquet`` is pre-filtered to training-eligible rows only —
      unreviewed/revisit segments and beats inside bad regions are already
      excluded by the upstream annotation builder. Therefore
      ``reviewed = matched_to_beats_parquet AND beat_training_eligible``.

    Args:
        peaks_df: Canonical peaks table (with timestamp_ms and segment_idx).
        beats_df: ``beats.parquet`` content (training-eligible beats only).
        bad_region_ranges: Retained for call-site compatibility; bad-region
            beats are already scrubbed upstream so this parameter is ignored.
        diagnostics_dir: Optional directory for annotation/peak alignment CSVs.

    Returns:
        DataFrame with columns: peak_id (int64), segment_idx (int32),
        label (str), subtype (str, "" for unmatched/unknown peaks),
        reviewed (bool).
    """
    peak_ts = peaks_df["timestamp_ms"].values.astype(np.int64)
    n_peaks = len(peak_ts)

    # ── Match each peak to its nearest beat in beats.parquet ──────────────
    # Vectorized nearest-neighbour search: for each peak timestamp we look
    # up the closest beat timestamp; the match is accepted if the distance
    # is within ANNOTATION_MATCH_TOLERANCE_MS.
    labels = np.full(n_peaks, "unknown", dtype=object)
    subtypes = np.full(n_peaks, "", dtype=object)
    is_reviewed = np.zeros(n_peaks, dtype=bool)
    matched = np.zeros(n_peaks, dtype=bool)

    if len(beats_df) > 0 and "timestamp_ms" in beats_df.columns:
        beats_sorted = beats_df.reset_index().rename(columns={"index": "input_beat_row"})
        beats_sorted["input_beat_row"] = beats_sorted["input_beat_row"].astype(np.int64) + 1
        beats_sorted = beats_sorted.sort_values("timestamp_ms").reset_index(drop=True)
        beat_ts = beats_sorted["timestamp_ms"].values.astype(np.int64)
        beat_label = beats_sorted["label"].astype(str).values
        beat_subtype = (
            beats_sorted["subtype"].astype(str).values
            if "subtype" in beats_sorted.columns
            else np.full(len(beats_sorted), "", dtype=object)
        )
        if "beat_training_eligible" in beats_sorted.columns:
            beat_eligible = (
                beats_sorted["beat_training_eligible"].fillna(False).astype(bool).values
            )
        else:
            beat_eligible = np.ones(len(beats_sorted), dtype=bool)

        # Nearest-neighbour by binary search: pipeline peak -> input beat.
        nearest_idx, nearest_dist, _signed_delta = nearest_indices_and_distances(
            peak_ts, beat_ts
        )
        matched = nearest_dist <= ANNOTATION_MATCH_TOLERANCE_MS

        # Pull labels and eligibility from the matched beat rows
        matched_beat_label = beat_label[nearest_idx]
        matched_beat_eligible = beat_eligible[nearest_idx]
        matched_beat_subtype = beat_subtype[nearest_idx]

        # Apply BEAT_LABEL_MAP only to matched peaks; warn on unknown labels
        unknown_label_mask = matched & ~np.isin(
            matched_beat_label, list(BEAT_LABEL_MAP.keys())
        )
        if unknown_label_mask.any():
            unknown_vals = sorted(set(matched_beat_label[unknown_label_mask]))
            logger.warning(
                "Encountered %d matched beats with unrecognized label(s) %r — "
                "treating as 'clean'",
                int(unknown_label_mask.sum()), unknown_vals,
            )

        # Translate input labels to canonical labels for matched peaks
        for src, dst in BEAT_LABEL_MAP.items():
            mask = matched & (matched_beat_label == src)
            labels[mask] = dst
            subtypes[mask] = matched_beat_subtype[mask]

        is_reviewed = matched & matched_beat_eligible.astype(bool)

        n_matched = int(matched.sum())
        n_eligible = int(is_reviewed.sum())
        logger.info(
            "Labeled %d / %d pipeline peaks from annotation beats "
            "(tolerance %d ms); %d are training-eligible",
            n_matched, n_peaks, ANNOTATION_MATCH_TOLERANCE_MS, n_eligible,
        )

        # The peak->beat count above is not the same as asking whether every
        # annotation input beat has a nearby pipeline peak. Compute that
        # direction explicitly so the warning describes the actual problem.
        peaks_for_input = peaks_df.sort_values("timestamp_ms").reset_index(drop=True)
        input_nearest_idx, input_nearest_dist, input_delta = nearest_indices_and_distances(
            beat_ts, peaks_for_input["timestamp_ms"].values.astype(np.int64)
        )
        input_matched = input_nearest_dist <= ANNOTATION_MATCH_TOLERANCE_MS
        input_used_by_peak = np.zeros(len(beats_sorted), dtype=bool)
        input_used_by_peak[nearest_idx[matched]] = True
        input_status = np.full(
            len(beats_sorted),
            "peak_within_80_but_no_pipeline_peak_chose_this_beat",
            dtype=object,
        )
        input_status[~input_matched] = "no_peak_within_80"
        input_status[input_used_by_peak] = "contributes_to_label"
        unmatched_input = int((~input_matched).sum())
        logger.info(
            "Input beat-to-peak alignment: %d / %d input beats have a pipeline "
            "peak within %d ms",
            int(input_matched.sum()), len(beats_sorted), ANNOTATION_MATCH_TOLERANCE_MS,
        )
        status_df = beats_sorted.copy()
        status_df["input_match_status"] = input_status
        status_df["nearest_peak_id"] = peaks_for_input.loc[
            input_nearest_idx, "peak_id"
        ].to_numpy(dtype=np.int64)
        status_df["nearest_peak_timestamp_ms"] = peaks_for_input.loc[
            input_nearest_idx, "timestamp_ms"
        ].to_numpy(dtype=np.int64)
        status_df["nearest_peak_delta_ms"] = input_delta
        status_df["nearest_peak_abs_distance_ms"] = input_nearest_dist
        status_df["timestamp_iso"] = status_df["timestamp_ms"].map(timestamp_ms_to_iso)
        status_df["timestamp_local"] = status_df["timestamp_ms"].map(
            timestamp_ms_to_local_display
        )
        status_df["nearest_peak_timestamp_iso"] = status_df[
            "nearest_peak_timestamp_ms"
        ].map(timestamp_ms_to_iso)
        status_df["nearest_peak_timestamp_local"] = status_df[
            "nearest_peak_timestamp_ms"
        ].map(timestamp_ms_to_local_display)
        status_groups = (
            status_df.groupby(["label", "input_match_status"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["label", "input_match_status"])
        )
        logger.info(
            "Input beat match status by label:\n%s",
            status_groups.to_string(index=False),
        )
        if diagnostics_dir is not None:
            status_cols = [
                "input_beat_row", "input_match_status", "timestamp_ms",
                "timestamp_iso", "timestamp_local", "segment_idx", "label",
                "subtype", "nearest_peak_id",
                "nearest_peak_timestamp_ms", "nearest_peak_timestamp_iso",
                "nearest_peak_timestamp_local", "nearest_peak_delta_ms",
                "nearest_peak_abs_distance_ms",
            ]
            status_cols = [c for c in status_cols if c in status_df.columns]
            write_csv(
                status_df[status_cols],
                diagnostics_dir / "annotation_beat_match_status.csv",
            )
            write_csv(
                status_groups,
                diagnostics_dir / "annotation_beat_match_status_groups.csv",
            )
        if unmatched_input > 0:
            logger.warning(
                "%d input beat row(s) had no pipeline peak within %d ms — "
                "annotation INPUT may be out of sync with current peaks.parquet",
                unmatched_input, ANNOTATION_MATCH_TOLERANCE_MS,
            )
            unmatched = beats_sorted.loc[~input_matched].copy()
            unmatched["nearest_peak_id"] = peaks_for_input.loc[
                input_nearest_idx[~input_matched], "peak_id"
            ].to_numpy(dtype=np.int64)
            unmatched["nearest_peak_timestamp_ms"] = peaks_for_input.loc[
                input_nearest_idx[~input_matched], "timestamp_ms"
            ].to_numpy(dtype=np.int64)
            unmatched["nearest_peak_delta_ms"] = input_delta[~input_matched]
            unmatched["nearest_peak_abs_distance_ms"] = input_nearest_dist[~input_matched]
            unmatched["timestamp_iso"] = unmatched["timestamp_ms"].map(timestamp_ms_to_iso)
            unmatched["timestamp_local"] = unmatched["timestamp_ms"].map(
                timestamp_ms_to_local_display
            )
            unmatched["nearest_peak_timestamp_iso"] = unmatched[
                "nearest_peak_timestamp_ms"
            ].map(timestamp_ms_to_iso)
            unmatched["nearest_peak_timestamp_local"] = unmatched[
                "nearest_peak_timestamp_ms"
            ].map(timestamp_ms_to_local_display)

            group_cols = [
                c for c in ("subtype", "label")
                if c in unmatched.columns
            ]
            if group_cols:
                logger.warning(
                    "Unmatched input beats by source:\n%s",
                    unmatched.groupby(group_cols)
                    .size()
                    .sort_values(ascending=False)
                    .to_string(),
                )
            bucket_counts = (
                pd.Series(input_nearest_dist)
                .map(lambda d: distance_bucket(int(d)))
                .value_counts()
                .sort_index()
            )
            logger.info("Input beat nearest-peak distance buckets:\n%s", bucket_counts.to_string())

            if diagnostics_dir is not None:
                cols = [
                    "input_beat_row", "timestamp_ms", "timestamp_iso", "timestamp_local",
                    "segment_idx", "label", "subtype",
                    "nearest_peak_id",
                    "nearest_peak_timestamp_ms", "nearest_peak_timestamp_iso",
                    "nearest_peak_timestamp_local",
                    "nearest_peak_delta_ms", "nearest_peak_abs_distance_ms",
                ]
                cols = [c for c in cols if c in unmatched.columns]
                write_csv(
                    unmatched[cols].sort_values("nearest_peak_abs_distance_ms", ascending=False),
                    diagnostics_dir / "annotation_unmatched_beats.csv",
                )
                summary = (
                    unmatched.assign(
                        distance_bucket=unmatched["nearest_peak_abs_distance_ms"].map(
                            lambda d: distance_bucket(int(d))
                        )
                    )
                    .groupby(group_cols + ["distance_bucket"], dropna=False)
                    .size()
                    .reset_index(name="count")
                    if group_cols else pd.DataFrame()
                )
                if len(summary) > 0:
                    write_csv(summary, diagnostics_dir / "annotation_unmatched_beat_groups.csv")
                write_csv(
                    pd.DataFrame(
                        {
                            "distance_bucket": bucket_counts.index.astype(str),
                            "count": bucket_counts.values.astype(np.int64),
                        }
                    ),
                    diagnostics_dir / "annotation_peak_distance_buckets.csv",
                )
    else:
        logger.warning(
            "beats_df is empty — all %d peaks default to label='unknown', reviewed=False",
            n_peaks,
        )

    result = pd.DataFrame(
        {
            "peak_id": peaks_df["peak_id"].values.astype(np.int64),
            "segment_idx": peaks_df["segment_idx"].values.astype(np.int32),
            "label": labels,
            "subtype": subtypes,
            "reviewed": is_reviewed,
        }
    )

    logger.info(
        "Label distribution:\n%s", result["label"].value_counts().to_string()
    )
    logger.info(
        "reviewed=True: %d  |  reviewed=False: %d",
        int(result["reviewed"].sum()), int((~result["reviewed"]).sum()),
    )
    return result


def build_segments(
    seg_ranges: dict[int, tuple[int, int]],
    segments_input_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    recording_start_ns: int,
    diagnostics_dir: Path | None = None,
) -> pd.DataFrame:
    """Build segment-level quality labels from the input segments parquet.

    Output columns:
      segment_idx (int32)
      start_timestamp_ms, end_timestamp_ms (int64)
      review_status ∈ {reviewed, unreviewed}
      quality_label ∈ {clean, noisy_ok, bad, unknown}
      segment_rmssd_ms (float32, NaN when <3 peaks)
      beat_training_eligible_segment (bool, passes through from input)
      segment_quality_training_eligible (bool, recomputed from final
        quality_label)

    Mapping rules:
      usable + RMSSD < 50           → clean
      usable + 50 ≤ RMSSD < 250     → unknown (RMSSD purgatory; discarded)
      usable + RMSSD ≥ 250          → noisy_ok
      usable + RMSSD NaN            → unknown
      unclean                       → noisy_ok
      unusable                      → bad
      unknown (input)               → unknown

    segment_quality_training_eligible is True iff the OUTPUT
    quality_label is in {clean, noisy_ok, bad}. beat_training_eligible_segment
    is passed through unchanged from the input — the RMSSD-driven
    demotion to `unknown` is purely a segment-level concern.

    The input segments_input_df uses a GUI ordinal `segment_idx` while the
    pipeline's seg_ranges keys are wall-clock segment ordinals; matching is
    done by timestamp overlap, not raw id equality.
    """
    rmssd_by_pipe_seg = compute_segment_rmssd(peaks_df)

    quality_by_seg: dict[int, str] = {}
    overlap_by_seg: dict[int, int] = {}
    priority_by_quality = {"unknown": 0, "clean": 1, "noisy_ok": 2, "bad": 3}
    mapping_records: list[dict[str, Any]] = []

    if len(segments_input_df) > 0:
        for row in segments_input_df.itertuples(index=False):
            input_seg_idx = int(getattr(row, "segment_idx"))
            start_ms = int(getattr(row, "start_ms"))
            end_ms = int(getattr(row, "end_ms"))
            raw_label = str(getattr(row, "segment_quality_label"))
            review_status = str(getattr(row, "review_status", ""))

            wall_start_seg = int((start_ms - recording_start_ns) // SEGMENT_DURATION_MS)
            wall_end_seg = int(
                (max(start_ms, end_ms - 1) - recording_start_ns) // SEGMENT_DURATION_MS
            )
            best_pipe_seg: int | None = None
            max_overlap = 0
            for pipe_seg in range(wall_start_seg, wall_end_seg + 1):
                pipe_start = recording_start_ns + pipe_seg * SEGMENT_DURATION_MS
                pipe_end = recording_start_ns + (pipe_seg + 1) * SEGMENT_DURATION_MS
                overlap = max(0, min(end_ms, pipe_end) - max(start_ms, pipe_start))
                if overlap <= 0 or pipe_seg not in seg_ranges:
                    continue
                if overlap > max_overlap:
                    best_pipe_seg = pipe_seg
                    max_overlap = int(overlap)

            if raw_label == "usable":
                rmssd = rmssd_by_pipe_seg.get(best_pipe_seg, float("nan"))
                quality = classify_usable_segment_quality(rmssd)
            else:
                quality = SEGMENT_QUALITY_MAP.get(raw_label, "unknown")

            applies = best_pipe_seg is not None and quality != "unknown"
            applied_pipe_seg: int | None = None
            if applies:
                old_overlap = overlap_by_seg.get(best_pipe_seg, -1)
                old_quality = quality_by_seg.get(best_pipe_seg, "unknown")
                should_replace = max_overlap > old_overlap or (
                    max_overlap == old_overlap
                    and priority_by_quality[quality] > priority_by_quality[old_quality]
                )
                if should_replace:
                    quality_by_seg[best_pipe_seg] = quality
                    overlap_by_seg[best_pipe_seg] = max_overlap
                applied_pipe_seg = best_pipe_seg

            mapping_records.append({
                "input_segment_idx": input_seg_idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "review_status": review_status,
                "raw_segment_quality_label": raw_label,
                "canonical_quality_label": quality,
                "best_pipeline_segment_idx": best_pipe_seg,
                "best_overlap_ms": max_overlap,
                "applied_pipeline_segment_idx": applied_pipe_seg,
            })

        if diagnostics_dir is not None:
            write_csv(
                pd.DataFrame(mapping_records),
                diagnostics_dir / "annotation_segment_index_mapping.csv",
            )

    btes_by_seg: dict[int, bool] = {}
    review_by_seg: dict[int, str] = {}
    if len(segments_input_df) > 0 and "beat_training_eligible_segment" in segments_input_df.columns:
        for rec in mapping_records:
            pipe_seg = rec["best_pipeline_segment_idx"]
            if pipe_seg is None:
                continue
            input_idx = rec["input_segment_idx"]
            input_row = segments_input_df.loc[
                segments_input_df["segment_idx"] == input_idx
            ]
            if len(input_row) > 0:
                btes_by_seg[pipe_seg] = bool(
                    input_row["beat_training_eligible_segment"].iloc[0]
                )
            review_by_seg[pipe_seg] = rec["review_status"]

    records: list[dict[str, Any]] = []
    for seg_int in sorted(seg_ranges.keys()):
        start_ms, end_ms = seg_ranges[seg_int]
        quality = quality_by_seg.get(seg_int, "unknown")
        rmssd = rmssd_by_pipe_seg.get(seg_int, float("nan"))
        beat_eligible = btes_by_seg.get(seg_int, False)
        sq_eligible = quality in {"clean", "noisy_ok", "bad"}
        review = review_by_seg.get(seg_int, "unreviewed")

        records.append({
            "segment_idx": np.int32(seg_int),
            "start_timestamp_ms": np.int64(start_ms),
            "end_timestamp_ms": np.int64(end_ms),
            "review_status": review,
            "quality_label": quality,
            "segment_rmssd_ms": np.float32(rmssd),
            "beat_training_eligible_segment": bool(beat_eligible),
            "segment_quality_training_eligible": bool(sq_eligible),
        })

    result = pd.DataFrame(records)
    result["segment_idx"] = result["segment_idx"].astype(np.int32)
    result["start_timestamp_ms"] = result["start_timestamp_ms"].astype(np.int64)
    result["end_timestamp_ms"] = result["end_timestamp_ms"].astype(np.int64)
    result["segment_rmssd_ms"] = result["segment_rmssd_ms"].astype(np.float32)
    result["beat_training_eligible_segment"] = result["beat_training_eligible_segment"].astype(bool)
    result["segment_quality_training_eligible"] = result["segment_quality_training_eligible"].astype(bool)

    logger.info(
        "Segment quality distribution:\n%s",
        result["quality_label"].value_counts().to_string(),
    )
    if "segment_rmssd_ms" in result.columns:
        purg_mask = (
            (result["quality_label"] == "unknown")
            & (~result["segment_rmssd_ms"].isna())
            & (result["segment_rmssd_ms"] >= SEGMENT_RMSSD_CLEAN_MAX_MS)
            & (result["segment_rmssd_ms"] < SEGMENT_RMSSD_NOISY_MIN_MS)
        )
        logger.info(
            "RMSSD purgatory segments (50 ≤ RMSSD < 250 + usable): %d",
            int(purg_mask.sum()),
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def validate_outputs(
    ecg_seg_idxs: set[int],
    peaks: pd.DataFrame,
    labels: pd.DataFrame,
    segments: pd.DataFrame,
) -> bool:
    """Check referential integrity across all canonical tables.

    Checks performed:
      1. Every peak_id in labels exists in peaks
      2. Every segment_idx in segments exists in ecg_samples (via ecg_seg_idxs)
      3. No null peak_ids in labels
      4. No null timestamps in peaks
      5. Timestamp and peak_id columns are int64

    Args:
        ecg_seg_idxs: Set of segment_idx values present in ecg_samples.parquet.
            Pass set(seg_ranges.keys()) from stream_ecg_to_parquet().
        peaks: The peaks table.
        labels: The labels table.
        segments: The segments table.

    Returns:
        True if all checks pass, False otherwise.
    """
    ok = True

    # 1. labels.peak_id ⊆ peaks.peak_id
    label_pids = set(labels["peak_id"].unique())
    peak_pids = set(peaks["peak_id"].unique())
    orphans = label_pids - peak_pids
    if orphans:
        logger.error(
            "VALIDATION FAIL: %d peak_ids in labels not found in peaks", len(orphans)
        )
        ok = False
    else:
        logger.info("VALIDATION OK: All label peak_ids exist in peaks")

    # 2. segments.segment_idx ⊆ ecg_samples.segment_idx
    seg_in_segments = set(int(s) for s in segments["segment_idx"].unique())
    orphan_segs = seg_in_segments - ecg_seg_idxs
    if orphan_segs:
        logger.error(
            "VALIDATION FAIL: %d segment_idxs in segments not in ecg_samples",
            len(orphan_segs),
        )
        ok = False
    else:
        logger.info("VALIDATION OK: All segment_idxs in segments exist in ecg_samples")

    # 3. No null peak_ids
    null_pids = int(labels["peak_id"].isna().sum())
    if null_pids > 0:
        logger.error("VALIDATION FAIL: %d null peak_ids in labels", null_pids)
        ok = False
    else:
        logger.info("VALIDATION OK: No null peak_ids in labels")

    # 4. No null timestamps in peaks
    nulls = int(peaks["timestamp_ms"].isna().sum())
    if nulls > 0:
        logger.error("VALIDATION FAIL: %d null timestamp_ms in peaks", nulls)
        ok = False

    # 5. Dtype checks
    dtype_checks = [
        ("timestamp_ms", "peaks", peaks),
        ("peak_id", "peaks", peaks),
        ("peak_id", "labels", labels),
    ]
    for col, name, df in dtype_checks:
        if df[col].dtype != np.int64:
            logger.error(
                "VALIDATION FAIL: %s.%s dtype is %s, expected int64",
                name, col, df[col].dtype,
            )
            ok = False

    if ok:
        logger.info("All validation checks passed")
    else:
        logger.error("Some validation checks failed — review warnings above")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════


def print_table_summary(
    df: pd.DataFrame, table_name: str, label_col: str | None = None
) -> None:
    """Print a human-readable summary of a canonical table.

    Shows row count, timestamp range, and label distribution (if applicable).

    Args:
        df: The table DataFrame.
        table_name: Display name for the table.
        label_col: Optional column name containing categorical labels.
    """
    print(f"\n{'=' * 60}")
    print(f"  {table_name}")
    print(f"{'=' * 60}")
    print(f"  Rows: {len(df):,}")

    # Timestamp range
    for ts_col in ("timestamp_ms", "start_timestamp_ms"):
        if ts_col in df.columns:
            ts_min = df[ts_col].min()
            ts_max = df[ts_col if ts_col == "timestamp_ms" else "end_timestamp_ms"].max()
            try:
                dt_min = pd.Timestamp(ts_min, unit="ms")
                dt_max = pd.Timestamp(ts_max, unit="ms")
                print(f"  Time range: {dt_min} -> {dt_max}")
            except (ValueError, OverflowError):
                print(f"  Time range: {ts_min} -> {ts_max} (raw ms)")
            break

    # Label distribution
    if label_col and label_col in df.columns:
        print(f"  {label_col} distribution:")
        dist = df[label_col].value_counts()
        for val, count in dist.items():
            pct = 100.0 * count / len(df)
            print(f"    {str(val):20s}: {count:>8,} ({pct:5.1f}%)")

    print(f"{'-' * 60}")


def save_parquet(
    df: pd.DataFrame,
    output_dir: Path,
    table_name: str,
    label_col: str | None = None,
) -> Path:
    """Save a DataFrame as a Snappy-compressed Parquet file and print summary.

    Args:
        df: DataFrame to save.
        output_dir: Directory for the output file (created if needed).
        table_name: Filename stem (e.g. "ecg_samples" → "ecg_samples.parquet").
        label_col: Optional label column for summary display.

    Returns:
        Path to the written Parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{table_name}.parquet"

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")

    print_table_summary(df, table_name, label_col)
    logger.info("Saved %s -> %s (%d rows)", table_name, path, len(df))
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Parse CLI arguments, run the full ingestion pipeline, and save outputs."""
    parser = argparse.ArgumentParser(
        description="ECG Artifact Detection Pipeline — Step 1: Data Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ecg-dir",
        type=Path,
        required=True,
        help="Directory containing raw ECG CSV files",
    )
    parser.add_argument(
        "--peaks-dir",
        type=Path,
        required=True,
        help="Directory containing R-peak CSV files",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help=(
            "Path to the annotation INPUT directory containing beats.parquet, "
            "segments.parquet, and bad_regions.parquet (e.g. "
            "Data/Annotations/INPUT/). Legacy artifact_annotation.json is no "
            "longer accepted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for Parquet files (default: data/processed/)",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=None,
        help=(
            "Directory for annotation/peak/segment diagnostic CSVs "
            "(default: <output-dir>/diagnostics/data_pipeline)"
        ),
    )
    parser.add_argument(
        "--resume-partial",
        type=Path,
        default=None,
        help=(
            "Path to an incomplete ecg_samples.parquet from a previous interrupted run. "
            "Row groups will be salvaged into per-file staging parquets so their CSVs "
            "are not re-processed."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for ECG file processing (default: min(4, cpu_count))",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only the first N ECG files (for debugging; default: all)",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    global logger
    logger = setup_logger("data_pipeline", args=args, disable_log=args.no_log)
    logger.info("=== data_pipeline started ===")
    logger.debug(
        "Config: SEGMENT_DURATION_MS=%d  DEDUP_TOLERANCE_MS=%d  "
        "ANNOTATION_MATCH_TOLERANCE_MS=%d  MIN_VALID_TIMESTAMP_MS=%d  "
        "ECG_CHUNK_SIZE=%d",
        SEGMENT_DURATION_MS, DEDUP_TOLERANCE_MS, ANNOTATION_MATCH_TOLERANCE_MS,
        MIN_VALID_TIMESTAMP_MS, ECG_CHUNK_SIZE,
    )

    # ── Validate input paths ───────────────────────────────────────────────
    if not args.ecg_dir.is_dir():
        logger.error("ECG directory not found: %s", args.ecg_dir)
        sys.exit(1)
    if not args.peaks_dir.is_dir():
        logger.error("Peaks directory not found: %s", args.peaks_dir)
        sys.exit(1)

    # ── Quick scan: find global recording start ────────────────────────────
    # Reads only the first row of each ECG file + peak_id column of peak files.
    # O(n_files) instead of O(total_rows). Must run before streaming.
    print("\n>> Scanning for recording start timestamp...")
    recording_start_ns = scan_recording_start_ns(args.ecg_dir, args.peaks_dir)

    # ── Load peak CSVs (small: just timestamps + metadata) ────────────────
    # Peak files total ~50M rows × a few columns ≈ manageable in RAM.
    print("\n>> Loading peak CSV files...")
    peak_csv_df = load_peak_csvs(args.peaks_dir)

    # ── Load annotation parquets (beats, segments, bad_regions) ───────────
    beats_input_df, segments_input_df, bad_regions_df = load_annotation_inputs(
        args.annotations
    )

    # ── Stream ECG files → ecg_samples.parquet ────────────────────────────
    # Each file is processed independently; RAM use is bounded to one file
    # at a time. Returns seg_ranges dict for downstream use.
    print("\n>> Streaming ECG files → ecg_samples.parquet (no full-dataset concat)...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or (args.output_dir / "diagnostics" / "data_pipeline")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Diagnostics directory: %s", diagnostics_dir.resolve())
    ecg_samples_path = args.output_dir / "ecg_samples.parquet"
    seg_ranges = stream_ecg_to_parquet(
        args.ecg_dir, ecg_samples_path, recording_start_ns,
        resume_partial=args.resume_partial,
        n_workers=args.workers,
        max_files=args.max_files,
    )

    # ── Build peaks ────────────────────────────────────────────────────────
    print("\n>> Building peaks table...")
    peaks = build_peaks(peak_csv_df, beats_input_df, recording_start_ns)

    # ── Build labels ───────────────────────────────────────────────────────
    print("\n>> Building labels table...")
    bad_region_ranges = extract_bad_region_time_ranges(bad_regions_df)
    labels = build_labels(peaks, beats_input_df, bad_region_ranges, diagnostics_dir)

    # ── Build segments ─────────────────────────────────────────────────────
    # Manual segment classification from segments.parquet is authoritative.
    print("\n>> Building segments table...")
    segments = build_segments(
        seg_ranges=seg_ranges,
        segments_input_df=segments_input_df,
        peaks_df=peaks,
        recording_start_ns=recording_start_ns,
        diagnostics_dir=diagnostics_dir,
    )

    # ── Validate referential integrity ─────────────────────────────────────
    print("\n>> Validating referential integrity...")
    valid = validate_outputs(set(seg_ranges.keys()), peaks, labels, segments)

    # ── Save remaining Parquet files ───────────────────────────────────────
    # ecg_samples.parquet was already written by stream_ecg_to_parquet().
    print("\n>> Saving Parquet tables...")
    save_parquet(peaks, args.output_dir, "peaks")
    save_parquet(labels, args.output_dir, "labels", label_col="label")
    save_parquet(segments, args.output_dir, "segments", label_col="quality_label")

    # ── Final status ───────────────────────────────────────────────────────
    status = "PASSED" if valid else "FAILED (see warnings above)"
    logger.info(
        "=== data_pipeline complete: %d peaks, %d labels, %d segments | validation=%s ===",
        len(peaks), len(labels), len(segments), status,
    )
    logger.info("Output directory: %s", args.output_dir.resolve())
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete  |  Validation: {status}")
    print(f"  Output: {args.output_dir.resolve()}")
    print(f"{'=' * 60}\n")

    if not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
