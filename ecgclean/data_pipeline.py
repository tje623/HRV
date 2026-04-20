#!/usr/bin/env python3
"""
ecgclean/data_pipeline.py — Step 1: Data Ingestion & Canonical Table Emission

Reads raw ECG CSVs, R-peak CSVs, and artifact_annotation.json to produce
four canonical Parquet tables consumed by all downstream pipeline stages:

  ecg_samples.parquet  — raw ECG time series with segment assignments
  peaks.parquet        — deduplicated R-peak catalog
  labels.parquet       — beat-level artifact/quality labels
  segments.parquet     — segment-level quality labels

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
        --annotations data/annotations/artifact_annotation.json \\
        --output-dir data/processed/
"""

from __future__ import annotations

import argparse
import json
import logging
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

# ── Constants ──────────────────────────────────────────────────────────────────

SEGMENT_DURATION_NS: int = 60_000_000_000        # 60 seconds in nanoseconds
MS_TO_NS: int = 1_000_000                         # milliseconds → nanoseconds
DEDUP_TOLERANCE_NS: int = 10 * MS_TO_NS           # 10 ms dedup window
# Wider tolerance used when matching annotation timestamps to detected peaks.
# Annotations were recorded against pre-snap (MWI-biased) peak positions.
# After argmax snap, peaks can shift up to 60 ms (the snap radius).  We use
# 80 ms here to give a small margin beyond the snap radius.  Safe at all HRs:
# even 200 bpm has 300 ms R-R intervals, so 80 ms won't cross-match neighbours.
ANNOTATION_MATCH_TOLERANCE_NS: int = 80 * MS_TO_NS  # 80 ms

# Minimum plausible recording timestamp: 2020-01-01 00:00:00 UTC in nanoseconds.
# Any timestamp below this is treated as corrupt/invalid and filtered out.
# Prevents a single bad CSV row (e.g. epoch-0 ms) from collapsing
# recording_start_ns to 0, which would overflow int32 segment_idx for all
# 2025 timestamps (~29 billion >> int32 max of ~2.1 billion).
MIN_VALID_TIMESTAMP_NS: int = 1_577_836_800_000_000_000  # 2020-01-01 UTC in ns
# Rows per chunk when streaming large ECG CSVs.  Keeps per-worker RAM usage
# bounded to ~8 MB/chunk regardless of file size (some files are 2+ GB CSVs).
_ECG_CHUNK_SIZE: int = 500_000
ARTIFACT_FRACTION_BAD: float = 0.30               # >30% artifact → bad segment
MIN_VALIDATED_BEATS_CLEAN: int = 10               # minimum clean beats for "clean" segment

# ECG polarity correction: if median amplitude at dominant peaks < this → inverted.
# Applied per file in _process_one_ecg_file.
_ECG_INVERSION_THRESHOLD: float = -0.05   # mV
_ECG_INVERSION_WINDOW_SEC: float = 1.0    # split into 1-second windows for detection
_ECG_INVERSION_MIN_WINDOWS: int = 5       # need at least this many windows to decide

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ecgclean.data_pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# TIMESTAMP PARSING
# ═══════════════════════════════════════════════════════════════════════════════


def parse_iso_to_ns(iso_str: str) -> int | None:
    """Parse an ISO 8601 datetime string to epoch nanoseconds (UTC).

    Handles up to nanosecond precision in the fractional-seconds part.
    All timestamps are interpreted as UTC.

    Args:
        iso_str: ISO 8601 string, e.g. "2025-02-04T18:01:42.399369220"

    Returns:
        Epoch nanoseconds as int64, or None if parsing fails.
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
        return epoch_s * 1_000_000_000 + frac_ns
    except Exception:
        return None


def parse_timestamp_to_ns(value: Any) -> int | None:
    """Convert a timestamp value of any supported format to epoch nanoseconds.

    Supported formats:
      - int / np.integer: epoch milliseconds
      - float: epoch milliseconds (NaN → None)
      - str: ISO 8601 datetime, or numeric string of epoch ms

    Args:
        value: Raw timestamp in any supported format.

    Returns:
        Epoch nanoseconds as int64, or None if conversion fails.
    """
    if isinstance(value, (int, np.integer)):
        return int(value) * MS_TO_NS
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return int(value) * MS_TO_NS
    if isinstance(value, str):
        ns = parse_iso_to_ns(value)
        if ns is not None:
            return ns
        # Fallback: try as numeric string
        try:
            return int(float(value)) * MS_TO_NS
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
    tolerance_ns: int = DEDUP_TOLERANCE_NS,
) -> np.ndarray:
    """Return a boolean mask: True for each query timestamp within tolerance of any reference.

    Uses binary search for O(n log m) performance.

    Args:
        query: Sorted int64 array of timestamps to check.
        reference: Sorted int64 array of reference timestamps.
        tolerance_ns: Maximum distance in nanoseconds for a match.

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


# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def get_annotation_key(annotations: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the value for the first matching key in the annotations dict.

    Supports alternate key names between the spec and actual annotation files
    (e.g. 'manually_added_missed_peaks' vs 'added_r_peaks').

    Args:
        annotations: The parsed annotation JSON dict.
        *keys: One or more key names to try, in priority order.
        default: Value to return if no key is found (defaults to empty list).

    Returns:
        The value from the first matching key, or default.
    """
    for key in keys:
        if key in annotations:
            return annotations[key]
    if default is not None:
        return default
    logger.warning("None of keys %s found in annotations; using empty list", keys)
    return []


def parse_phys_event_windows(raw: list[Any]) -> list[tuple[int, int]]:
    """Parse tagged_physiological_events into (start_ns, end_ns) tuples.

    Handles two formats found in the wild:
      - List of dicts with start/end or start_time/end_time keys → window
      - List of bare epoch-ms timestamps → point event (start == end)

    Args:
        raw: Raw list from the annotation JSON.

    Returns:
        Sorted list of (start_ns, end_ns) tuples.
    """
    windows: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, dict):
            start_val = (
                item.get("start")
                or item.get("start_time")
                or item.get("start_timestamp")
            )
            end_val = (
                item.get("end")
                or item.get("end_time")
                or item.get("end_timestamp")
            )
            if start_val is not None and end_val is not None:
                start_ns = parse_timestamp_to_ns(start_val)
                end_ns = parse_timestamp_to_ns(end_val)
                if start_ns is not None and end_ns is not None:
                    windows.append((start_ns, end_ns))
                else:
                    logger.warning("Could not parse phys event dict timestamps: %r", item)
            else:
                logger.warning("Phys event dict missing start/end keys: %r", item)
        else:
            # Bare timestamp → point event
            ns = parse_timestamp_to_ns(item)
            if ns is not None:
                windows.append((ns, ns))
            else:
                logger.warning("Could not parse phys event timestamp: %r", item)
    windows.sort(key=lambda w: w[0])
    return windows


def _build_ann_to_pipeline_map(
    annotations: dict[str, Any],
    recording_start_ns: int,
) -> dict[int, int]:
    """Build a mapping from annotation segment index to pipeline segment_idx.

    The annotation tool numbers segments sequentially within the compiled file
    (0, 1, 2, …).  The pipeline assigns segment_idx as elapsed 60-second windows
    from recording_start_ns.  'segment_timestamps' in the annotation JSON maps
    each annotation index to its actual first_timestamp, allowing conversion.

    Args:
        annotations: Parsed annotation JSON dict.
        recording_start_ns: Pipeline recording start in epoch nanoseconds.

    Returns:
        Dict mapping annotation_seg_idx (int) → pipeline segment_idx (int).
    """
    mapping: dict[int, int] = {}
    for entry in annotations.get("segment_timestamps", []):
        ann_idx = entry.get("segment_idx")
        ts_str = entry.get("first_timestamp")
        if ann_idx is None or ts_str is None:
            continue
        try:
            ts_ns = int(pd.Timestamp(ts_str).value)
            pipeline_idx = (ts_ns - recording_start_ns) // SEGMENT_DURATION_NS
            mapping[int(ann_idx)] = int(pipeline_idx)
        except Exception:
            logger.warning("Could not parse segment_timestamps entry: %r", entry)
    return mapping


def extract_validated_segment_indices(
    annotations: dict[str, Any],
    recording_start_ns: int,
) -> set[int]:
    """Return the set of pipeline segment_idx values that were manually reviewed.

    The annotation tool numbers segments sequentially within the compiled file
    (seg_0, seg_1, …), while the pipeline assigns segment_idx as the number of
    complete 60-second windows elapsed since recording_start_ns.  These two
    numbering systems only coincide by accident.

    This function translates using 'segment_timestamps', which maps each
    annotation segment index to its first_timestamp.  The pipeline segment_idx
    is then recovered with the same formula the pipeline itself uses:

        pipeline_seg_idx = (first_timestamp_ns - recording_start_ns) // SEGMENT_DURATION_NS

    Args:
        annotations: The parsed annotation JSON dict.
        recording_start_ns: Epoch nanoseconds of the recording start, as
            computed by the pipeline (min of ECG and peak start timestamps).

    Returns:
        Set of pipeline integer segment_idx values that were reviewed.
        Empty set if the key is absent.
    """
    # ── Translate annotation indices → pipeline indices ──────────────────
    ann_to_pipeline = _build_ann_to_pipeline_map(annotations, recording_start_ns)

    if not ann_to_pipeline:
        logger.warning(
            "'segment_timestamps' absent or empty — falling back to raw index "
            "passthrough for validated_segments (indices may not match pipeline)"
        )

    # ── Convert validated_segments annotation indices to pipeline indices ──
    reviewed: set[int] = set()
    n_no_mapping = 0
    for seg in get_annotation_key(annotations, "validated_segments", default=[]):
        if isinstance(seg, (int, np.integer)):
            ann_idx = int(seg)
        elif isinstance(seg, str):
            try:
                ann_idx = int(seg.replace("seg_", ""))
            except ValueError:
                logger.warning("Could not parse validated segment identifier: %r", seg)
                continue
        else:
            continue

        if ann_to_pipeline:
            if ann_idx in ann_to_pipeline:
                reviewed.add(ann_to_pipeline[ann_idx])
            else:
                n_no_mapping += 1
        else:
            reviewed.add(ann_idx)  # fallback: pass through raw index

    if n_no_mapping:
        logger.warning(
            "%d validated segments had no entry in segment_timestamps — skipped",
            n_no_mapping,
        )

    if reviewed:
        logger.info(
            "Validated (reviewed) segments: %d annotation entries → %d pipeline indices",
            len(get_annotation_key(annotations, "validated_segments", default=[])),
            len(reviewed),
        )
    else:
        logger.warning(
            "No 'validated_segments' key found — all beats will be marked unreviewed"
        )
    return reviewed


def extract_bad_segment_indices(
    annotations: dict[str, Any],
    recording_start_ns: int,
) -> set[int]:
    """Collect all pipeline segment_idx values flagged as bad from annotation keys.

    Checks: bad_segments, bad_regions (by segment_idx), flagged_poor_quality_segments.
    All annotation segment indices are translated to pipeline segment_idx using
    segment_timestamps (same coordinate-system translation as validated_segments).

    Args:
        annotations: The parsed annotation JSON dict.
        recording_start_ns: Pipeline recording start in epoch nanoseconds.

    Returns:
        Set of pipeline integer segment_idx values flagged as bad.
    """
    ann_to_pipeline = _build_ann_to_pipeline_map(annotations, recording_start_ns)

    def _translate(ann_idx: int) -> int | None:
        if ann_to_pipeline:
            return ann_to_pipeline.get(ann_idx)
        return ann_idx  # fallback if no segment_timestamps

    bad: set[int] = set()

    for seg in get_annotation_key(annotations, "bad_segments", default=[]):
        if isinstance(seg, (int, np.integer)):
            p = _translate(int(seg))
            if p is not None:
                bad.add(p)

    for seg in get_annotation_key(
        annotations, "flagged_poor_quality_segments", default=[]
    ):
        if isinstance(seg, (int, np.integer)):
            ann_idx = int(seg)
        elif isinstance(seg, str):
            try:
                ann_idx = int(seg.replace("seg_", ""))
            except ValueError:
                logger.warning("Could not parse segment identifier: %r", seg)
                continue
        else:
            continue
        p = _translate(ann_idx)
        if p is not None:
            bad.add(p)

    return bad


def extract_bad_region_time_ranges(
    annotations: dict[str, Any],
    recording_start_ns: int,
) -> list[tuple[int, int, int]]:
    """Return (pipeline_seg_idx, start_ns, end_ns) tuples for all bad_regions.

    bad_regions are time-bounded windows within a segment that are
    uninterpretable.  Unlike bad_segments, they do NOT invalidate the whole
    segment — only beats whose timestamp falls inside the window are affected.

    Args:
        annotations: Parsed annotation JSON dict.
        recording_start_ns: Pipeline recording start in epoch nanoseconds.

    Returns:
        List of (pipeline_seg_idx, start_ns, end_ns) tuples, one per window.
        Malformed entries are silently dropped with a warning.
    """
    ann_to_pipeline = _build_ann_to_pipeline_map(annotations, recording_start_ns)
    result: list[tuple[int, int, int]] = []

    for region in get_annotation_key(annotations, "bad_regions", default=[]):
        if not isinstance(region, dict):
            continue
        ann_idx = region.get("segment_idx")
        start_str = region.get("start_time")
        end_str = region.get("end_time")
        if ann_idx is None or start_str is None or end_str is None:
            logger.warning("Malformed bad_region entry (skipped): %r", region)
            continue
        if ann_to_pipeline:
            pipeline_idx = ann_to_pipeline.get(int(ann_idx))
            if pipeline_idx is None:
                logger.warning(
                    "bad_region segment_idx=%d has no pipeline mapping (skipped)",
                    ann_idx,
                )
                continue
        else:
            pipeline_idx = int(ann_idx)  # fallback: pass-through raw index
        try:
            start_ns = int(pd.Timestamp(start_str).value)
            end_ns = int(pd.Timestamp(end_str).value)
        except Exception:
            logger.warning("Could not parse bad_region timestamps (skipped): %r", region)
            continue
        result.append((int(pipeline_idx), int(start_ns), int(end_ns)))

    if result:
        segs = len({r[0] for r in result})
        logger.info(
            "Bad-region time ranges: %d windows across %d segment(s)",
            len(result), segs,
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
            tbl = pq.read_table(staging_path, columns=["timestamp_ns", "segment_idx"])
            ts_arr = tbl["timestamp_ns"].to_pandas().values
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

    # ── Step 2: small sample to determine ts format, fs, and polarity ─────
    sample_df  = pd.read_csv(csv_path, nrows=10_000)
    sample_val = sample_df[ts_col].iloc[0]
    ts_is_string = isinstance(sample_val, str)

    # Estimate sampling frequency from first 1000 valid timestamp gaps
    if ts_is_string:
        _sample_ts = sample_df[ts_col].head(1000).apply(parse_timestamp_to_ns).dropna().values.astype(np.int64)
    else:
        _sample_ts = sample_df[ts_col].head(1000).values.astype(np.int64) * MS_TO_NS
    _fs_est = 60.0  # fallback
    if len(_sample_ts) > 1:
        _fs_est = 1e9 / float(np.median(np.diff(_sample_ts)))

    # Polarity check on the sample (10K rows ≈ 77 seconds at 130 Hz → 77 windows)
    if ts_is_string:
        _s_ts_ser = sample_df[ts_col].apply(parse_timestamp_to_ns)
        _s_valid  = _s_ts_ser.notna()
        _s_ts     = _s_ts_ser[_s_valid].values.astype(np.int64)
        _s_ecg    = pd.to_numeric(sample_df.loc[_s_valid, ecg_col], errors="coerce").values.astype(np.float32)
    else:
        _s_ts   = sample_df[ts_col].values.astype(np.int64) * MS_TO_NS
        _s_raw  = pd.to_numeric(sample_df[ecg_col], errors="coerce").values
        _s_fin  = np.isfinite(_s_raw)
        _s_ts   = _s_ts[_s_fin]
        _s_ecg  = _s_raw[_s_fin].astype(np.float32)
    _s_valid_ts = _s_ts >= MIN_VALID_TIMESTAMP_NS
    _s_ts  = _s_ts[_s_valid_ts]
    _s_ecg = _s_ecg[_s_valid_ts]

    invert = False
    _win = max(1, int(_ECG_INVERSION_WINDOW_SEC * _fs_est))
    _nw  = len(_s_ecg) // _win
    if _nw >= _ECG_INVERSION_MIN_WINDOWS:
        _dom = np.array([
            _s_ecg[w * _win + int(np.argmax(np.abs(_s_ecg[w * _win:(w + 1) * _win])))]
            for w in range(_nw)
        ])
        _dom_median = float(np.median(_dom))
        _dom_p10    = float(np.percentile(_dom, 10))
        _dom_p90    = float(np.percentile(_dom, 90))
        if _dom_median < _ECG_INVERSION_THRESHOLD:
            logger.info(
                "  [polarity] %s — INVERTING sign  "
                "(median_dominant=%.4f mV, p10=%.4f, p90=%.4f, windows=%d, threshold=%.4f)",
                csv_path.name, _dom_median, _dom_p10, _dom_p90, _nw, _ECG_INVERSION_THRESHOLD,
            )
            invert = True
            # ── Diagnostic PNG ────────────────────────────────────────────────
            # Save first ~10 seconds of ECG to a PNG so you can manually verify
            # the flip decision.  Written to <output_dir>/diagnostics/polarity/.
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                _diag_dir = staging_path.parent.parent / "diagnostics" / "polarity"
                _diag_dir.mkdir(parents=True, exist_ok=True)
                _plot_samples = min(len(_s_ecg), int(10 * _fs_est))
                _t_sec = np.arange(_plot_samples) / _fs_est
                fig, ax = plt.subplots(figsize=(14, 3))
                ax.plot(_t_sec, _s_ecg[:_plot_samples], lw=0.6, color="steelblue")
                ax.axhline(0, color="gray", lw=0.5, ls="--")
                ax.set_title(
                    f"{csv_path.name}  →  INVERTED (sign will be flipped)\n"
                    f"median_dominant={_dom_median:.4f} mV  "
                    f"p10={_dom_p10:.4f}  p90={_dom_p90:.4f}  "
                    f"windows={_nw}  threshold={_ECG_INVERSION_THRESHOLD}",
                    fontsize=9,
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ECG (mV)")
                ax.set_xlim(0, _t_sec[-1])
                fig.tight_layout()
                _png_path = _diag_dir / (csv_path.stem + "_polarity_check.png")
                fig.savefig(_png_path, dpi=100)
                plt.close(fig)
                logger.info("  [polarity] diagnostic PNG → %s", _png_path.name)
            except Exception as _png_err:
                logger.warning("  [polarity] could not save diagnostic PNG: %s", _png_err)
        else:
            logger.info(
                "  [polarity] %s — OK  "
                "(median_dominant=%.4f mV, p10=%.4f, p90=%.4f, windows=%d)",
                csv_path.name, _dom_median, _dom_p10, _dom_p90, _nw,
            )

    # ── Step 3: stream full CSV in chunks → ParquetWriter ─────────────────
    schema  = pa.schema([
        ("timestamp_ns", pa.int64()),
        ("ecg",          pa.float32()),
        ("segment_idx",  pa.int32()),
    ])
    n_total  = 0
    n_bad_ts = 0

    try:
        with pq.ParquetWriter(staging_path, schema, compression="snappy") as writer:
            for chunk in pd.read_csv(csv_path, chunksize=_ECG_CHUNK_SIZE):
                if ts_is_string:
                    _ts_ser  = chunk[ts_col].apply(parse_timestamp_to_ns)
                    _c_valid = _ts_ser.notna()
                    ts_c     = _ts_ser[_c_valid].values.astype(np.int64)
                    ecg_c    = pd.to_numeric(chunk.loc[_c_valid, ecg_col], errors="coerce").values.astype(np.float32)
                else:
                    ts_c   = chunk[ts_col].values.astype(np.int64) * MS_TO_NS
                    _raw   = pd.to_numeric(chunk[ecg_col], errors="coerce").values
                    _fin   = np.isfinite(_raw)
                    ts_c   = ts_c[_fin]
                    ecg_c  = _raw[_fin].astype(np.float32)

                # Drop implausible timestamps
                _good = ts_c >= MIN_VALID_TIMESTAMP_NS
                if not _good.all():
                    n_bad_ts += int((~_good).sum())
                    ts_c  = ts_c[_good]
                    ecg_c = ecg_c[_good]

                if len(ts_c) == 0:
                    continue

                # Convert mV → µV (Polar H10 CSVs are in millivolts)
                ecg_c = ecg_c * 1000.0

                if invert:
                    ecg_c = ecg_c * -1.0

                seg_c = ((ts_c - recording_start_ns) // SEGMENT_DURATION_NS).astype(np.int32)

                writer.write_table(pa.table({
                    "timestamp_ns": pa.array(ts_c,  type=pa.int64()),
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
            if ns is not None and ns >= MIN_VALID_TIMESTAMP_NS and (min_ns is None or ns < min_ns):
                min_ns = ns
            elif ns is not None and ns < MIN_VALID_TIMESTAMP_NS:
                logger.warning(
                    "Ignoring implausible first-row timestamp in %s: %s (< 2020-01-01)",
                    path.name, pd.Timestamp(ns, unit="ns"),
                )
        except Exception as exc:
            logger.warning("Could not read first row of %s: %s", path.name, exc)

    for path in sorted(peaks_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path, usecols=["peak_id"])
            peak_min_ns = int(df["peak_id"].astype(np.int64).min()) * MS_TO_NS
            if peak_min_ns >= MIN_VALID_TIMESTAMP_NS and (min_ns is None or peak_min_ns < min_ns):
                min_ns = peak_min_ns
        except Exception as exc:
            logger.warning("Could not scan peak file %s: %s", path.name, exc)

    if min_ns is None:
        logger.error("Could not determine recording start — no readable files found")
        sys.exit(1)

    logger.info("Recording start: %s", pd.Timestamp(min_ns, unit="ns"))
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
        seg_ranges: {segment_idx → (min_timestamp_ns, max_timestamp_ns)}.
        Used by build_segments() so the full Parquet never needs re-reading.
    """
    schema = pa.schema([
        pa.field("timestamp_ns", pa.int64()),
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
            lambda v: (parse_timestamp_to_ns(v) or 0) // MS_TO_NS
        ).astype(np.int64)

    logger.info("Loaded %d peaks from %d file(s)", len(result), len(csv_files))
    return result


def load_annotations(annotations_path: Path) -> dict[str, Any]:
    """Load artifact_annotation.json with graceful handling of missing keys.

    Logs warnings for expected keys that are not present. Never crashes on
    missing or unexpected structure.

    Args:
        annotations_path: Path to the JSON annotation file.

    Returns:
        Parsed dict (empty dict if file not found).
    """
    if not annotations_path.exists():
        logger.warning(
            "Annotation file not found: %s — proceeding with empty annotations",
            annotations_path,
        )
        return {}

    with open(annotations_path, "r") as f:
        data = json.load(f)

    # Check for expected keys and their known alternates
    expected_and_alts = {
        "artifacts": [],
        "validated_true_beats": ["validated_segments"],
        "bad_segments": [],
        "bad_regions": [],
        "flagged_poor_quality_segments": [],
        "manually_added_missed_peaks": ["added_r_peaks"],
        "tagged_physiological_events": [],
        "flagged_for_interpolation": [],
        "interpolated_replacements": [],
    }
    for key, alts in expected_and_alts.items():
        if key not in data:
            found_alt = any(a in data for a in alts)
            if found_alt:
                alt_name = next(a for a in alts if a in data)
                logger.info("Using alternate key '%s' for expected '%s'", alt_name, key)
            else:
                logger.warning("Annotation key '%s' not found (no alternates either)", key)

    logger.info("Loaded annotations with %d top-level keys", len(data))
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════


def build_peaks(
    peak_csv_df: pd.DataFrame,
    annotations: dict[str, Any],
    recording_start_ns: int,
) -> pd.DataFrame:
    """Build the deduplicated peaks canonical table.

    Merges R-peaks from CSV with manually added peaks from annotations.
    Deduplicates within 10ms tolerance, preferring annotation timestamps.

    Args:
        peak_csv_df: Raw peak DataFrame from load_peak_csvs.
        annotations: Parsed annotation JSON dict.
        recording_start_ns: Epoch ns of recording start (for segment_idx).

    Returns:
        DataFrame with columns: peak_id (int64, auto-increment),
        timestamp_ns (int64), segment_idx (int32), source (str),
        is_added_peak (bool).
    """
    # ── Collect CSV peaks ──────────────────────────────────────────────────
    csv_ts_ns = peak_csv_df["peak_id"].values.astype(np.int64) * MS_TO_NS

    csv_source = (
        peak_csv_df["source"].values
        if "source" in peak_csv_df.columns
        else np.full(len(csv_ts_ns), "detected", dtype=object)
    )
    csv_is_added = (
        peak_csv_df["is_added_peak"].values
        if "is_added_peak" in peak_csv_df.columns
        else np.full(len(csv_ts_ns), False, dtype=bool)
    )

    records: list[dict[str, Any]] = []
    for i in range(len(csv_ts_ns)):
        src = str(csv_source[i]) if not pd.isna(csv_source[i]) else "detected"
        added = bool(csv_is_added[i]) if not pd.isna(csv_is_added[i]) else False
        records.append(
            {
                "timestamp_ns": int(csv_ts_ns[i]),
                "source": src,
                "is_added_peak": added,
                "_origin": "csv",
            }
        )

    # ── Collect annotation added peaks (manually_added_missed_peaks) ───────
    added_raw = get_annotation_key(
        annotations, "manually_added_missed_peaks", "added_r_peaks", default=[]
    )
    added_ns = parse_timestamp_list(added_raw)
    for ts in added_ns:
        records.append(
            {
                "timestamp_ns": int(ts),
                "source": "added",
                "is_added_peak": True,
                "_origin": "annotation",
            }
        )

    # ── Collect interpolated replacement peaks (new timestamps) ────────────
    for item in get_annotation_key(annotations, "interpolated_replacements", default=[]):
        if isinstance(item, dict):
            ts_val = item.get("timestamp")
            if ts_val is not None:
                ns = parse_timestamp_to_ns(ts_val)
                if ns is not None:
                    records.append(
                        {
                            "timestamp_ns": int(ns),
                            "source": "detected",
                            "is_added_peak": False,
                            "_origin": "annotation",
                        }
                    )

    # ── Build DataFrame and deduplicate ────────────────────────────────────
    peaks_df = pd.DataFrame(records)
    peaks_df.sort_values("timestamp_ns", inplace=True)
    peaks_df.reset_index(drop=True, inplace=True)

    # Dedup: within 10ms tolerance, keep annotation version (lower _priority)
    origin_priority = {"annotation": 0, "csv": 1}
    peaks_df["_priority"] = peaks_df["_origin"].map(origin_priority).fillna(1).astype(int)

    timestamps = peaks_df["timestamp_ns"].values
    keep_mask = np.ones(len(peaks_df), dtype=bool)
    priorities = peaks_df["_priority"].values

    i = 0
    while i < len(peaks_df):
        j = i + 1
        while j < len(peaks_df) and (timestamps[j] - timestamps[i]) <= DEDUP_TOLERANCE_NS:
            j += 1
        if j > i + 1:
            # Cluster of peaks within tolerance — keep the highest-priority one
            cluster_priorities = priorities[i:j]
            best_offset = int(np.argmin(cluster_priorities))
            for k in range(i, j):
                if k != i + best_offset:
                    keep_mask[k] = False
        i = j

    n_dupes = int((~keep_mask).sum())
    if n_dupes > 0:
        logger.info("Deduplicated %d peaks within %d ms tolerance", n_dupes, DEDUP_TOLERANCE_NS // MS_TO_NS)

    peaks_df = peaks_df[keep_mask].copy()
    peaks_df.drop(columns=["_origin", "_priority"], inplace=True)

    # Assign segment_idx relative to recording start
    peaks_df["segment_idx"] = (
        (peaks_df["timestamp_ns"] - recording_start_ns) // SEGMENT_DURATION_NS
    ).astype(np.int32)

    # Auto-increment peak_id
    peaks_df.reset_index(drop=True, inplace=True)
    peaks_df.insert(0, "peak_id", np.arange(len(peaks_df), dtype=np.int64))

    # Enforce dtypes
    peaks_df["timestamp_ns"] = peaks_df["timestamp_ns"].astype(np.int64)
    peaks_df["segment_idx"] = peaks_df["segment_idx"].astype(np.int32)
    peaks_df["is_added_peak"] = peaks_df["is_added_peak"].astype(bool)

    logger.info(
        "Built peaks table: %d peaks (%d detected, %d added)",
        len(peaks_df),
        (peaks_df["source"] == "detected").sum(),
        (peaks_df["source"] == "added").sum(),
    )
    return peaks_df


def build_labels(
    peaks_df: pd.DataFrame,
    peak_csv_df: pd.DataFrame,
    annotations: dict[str, Any],
    validated_seg_idxs: set[int] | None = None,
    bad_region_ranges: list[tuple[int, int, int]] | None = None,
) -> pd.DataFrame:
    """Build beat-level labels with priority-based assignment.

    Label priority (highest wins):
      1. artifact       — from annotation 'artifacts' + CSV label=1
      2. interpolated   — from 'flagged_for_interpolation' / 'interpolated_replacements'
      3. missed_original — from 'manually_added_missed_peaks' / 'added_r_peaks'
      4. phys_event     — beat within a tagged_physiological_events window
      5. clean          — default label within reviewed segments

    Also computes:
      - phys_event_window (bool): True if the beat falls within any physiological
        event window, regardless of primary label.
      - reviewed (bool): True if the beat's segment appears in validated_seg_idxs.
        Only reviewed beats carry ground-truth labels and should be used for
        model training.  Beats in unreviewed segments default to label='clean'
        but reviewed=False, and must NOT be used as negative training examples.
      - in_bad_region (bool): True if the beat's timestamp falls inside any
        bad_region time window for its segment.  These beats are uninterpretable
        and must be excluded from model training (but NOT from inference).

    Args:
        peaks_df: Canonical peaks table (with timestamp_ns and segment_idx).
        peak_csv_df: Raw peak CSV DataFrame (with peak_id in ms and label).
        annotations: Parsed annotation JSON dict.
        validated_seg_idxs: Set of integer segment indices that were manually
            reviewed.  Pass the result of extract_validated_segment_indices().
            If None, all beats are marked reviewed=False.
        bad_region_ranges: List of (pipeline_seg_idx, start_ns, end_ns) tuples
            from extract_bad_region_time_ranges().  If None, in_bad_region is
            all False.

    Returns:
        DataFrame with columns: peak_id (int64), label (str),
        phys_event_window (bool), reviewed (bool), in_bad_region (bool).
    """
    peak_ts = peaks_df["timestamp_ns"].values.astype(np.int64)
    n_peaks = len(peak_ts)

    # ── Parse annotation timestamp arrays ──────────────────────────────────
    # Artifacts from annotation JSON
    annotation_artifact_ns = parse_timestamp_list(
        get_annotation_key(annotations, "artifacts", default=[])
    )

    # Artifacts from CSV label=1
    csv_artifact_ns = np.array([], dtype=np.int64)
    if "label" in peak_csv_df.columns and "peak_id" in peak_csv_df.columns:
        artifact_rows = peak_csv_df[peak_csv_df["label"] == 1]
        if len(artifact_rows) > 0:
            csv_artifact_ns = (
                artifact_rows["peak_id"].values.astype(np.int64) * MS_TO_NS
            )

    all_artifact_ns = np.unique(
        np.concatenate([annotation_artifact_ns, csv_artifact_ns])
    )

    # Interpolated peaks
    interp_ns_list: list[int] = []
    for ts in parse_timestamp_list(
        get_annotation_key(annotations, "flagged_for_interpolation", default=[])
    ):
        interp_ns_list.append(int(ts))
    for item in get_annotation_key(annotations, "interpolated_replacements", default=[]):
        if isinstance(item, dict):
            for field in ("peak_id", "timestamp"):
                val = item.get(field)
                if val is not None:
                    ns = parse_timestamp_to_ns(val)
                    if ns is not None:
                        interp_ns_list.append(ns)
    interp_ns = np.unique(np.array(interp_ns_list, dtype=np.int64))

    # Manually added peaks
    added_ns = parse_timestamp_list(
        get_annotation_key(
            annotations, "manually_added_missed_peaks", "added_r_peaks", default=[]
        )
    )

    # Physiological event windows
    phys_events = parse_phys_event_windows(
        get_annotation_key(annotations, "tagged_physiological_events", default=[])
    )

    # ── Vectorized matching ────────────────────────────────────────────────
    # Use ANNOTATION_MATCH_TOLERANCE_NS (80 ms) for all annotation-derived
    # labels.  Annotation timestamps were recorded against pre-snap peak
    # positions; the argmax snap shifts peaks up to 60 ms, so the default
    # 10 ms dedup tolerance would miss ~80 % of valid matches.
    is_artifact = timestamps_match_with_tolerance(
        peak_ts, all_artifact_ns, tolerance_ns=ANNOTATION_MATCH_TOLERANCE_NS
    )
    is_interp = timestamps_match_with_tolerance(
        peak_ts, interp_ns, tolerance_ns=ANNOTATION_MATCH_TOLERANCE_NS
    )
    is_added = (
        peaks_df["is_added_peak"].values
        | timestamps_match_with_tolerance(
            peak_ts, added_ns, tolerance_ns=ANNOTATION_MATCH_TOLERANCE_NS
        )
    )

    # Phys event windows: check if each peak falls within any window
    in_phys_window = np.zeros(n_peaks, dtype=bool)
    for start_ns, end_ns in phys_events:
        in_phys_window |= (peak_ts >= start_ns) & (peak_ts <= end_ns)

    # ── Assign labels: lowest priority first, higher overwrites ────────────
    labels = np.full(n_peaks, "clean", dtype=object)

    # Priority 4: phys_event (overwrites clean)
    labels[in_phys_window] = "phys_event"

    # Priority 3: missed_original (overwrites clean + phys_event)
    labels[is_added] = "missed_original"

    # Priority 2: interpolated (overwrites lower)
    labels[is_interp] = "interpolated"

    # Priority 1: artifact (overwrites everything)
    labels[is_artifact] = "artifact"

    # ── Reviewed flag ─────────────────────────────────────────────────────
    # A beat is "reviewed" only if its segment was manually annotated.
    # Unreviewed beats should not be used as negative training examples.
    if validated_seg_idxs is not None:
        seg_idxs = peaks_df["segment_idx"].values
        is_reviewed = np.array(
            [int(s) in validated_seg_idxs for s in seg_idxs], dtype=bool
        )
        n_reviewed = int(is_reviewed.sum())
        n_unreviewed = n_peaks - n_reviewed
        logger.info(
            "Reviewed beats: %d  |  Unreviewed (excluded from training): %d",
            n_reviewed,
            n_unreviewed,
        )
    else:
        is_reviewed = np.zeros(n_peaks, dtype=bool)
        logger.warning(
            "No validated_seg_idxs provided — all %d beats marked unreviewed", n_peaks
        )

    # ── in_bad_region flag ────────────────────────────────────────────────────
    # Vectorized: for each (seg_idx, start_ns, end_ns), mask beats in the window.
    seg_idxs = peaks_df["segment_idx"].values
    in_bad_region = np.zeros(n_peaks, dtype=bool)
    if bad_region_ranges:
        for br_seg, br_start, br_end in bad_region_ranges:
            mask = (
                (seg_idxs == br_seg)
                & (peak_ts >= br_start)
                & (peak_ts <= br_end)
            )
            in_bad_region |= mask
        n_in_bad_region = int(in_bad_region.sum())
        logger.info(
            "Beats inside bad_region windows: %d (in_bad_region=True)",
            n_in_bad_region,
        )

    result = pd.DataFrame(
        {
            "peak_id": peaks_df["peak_id"].values.astype(np.int64),
            "label": labels,
            "phys_event_window": in_phys_window,
            "reviewed": is_reviewed,
            "in_bad_region": in_bad_region,
        }
    )

    dist = result["label"].value_counts()
    logger.info("Label distribution:\n%s", dist.to_string())
    return result


def build_segments(
    seg_ranges: dict[int, tuple[int, int]],
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    annotations: dict[str, Any],
    recording_start_ns: int,
) -> pd.DataFrame:
    """Build segment-level quality labels.

    Quality label assignment:
      - bad: flagged in bad_segments/bad_regions/flagged_poor_quality_segments,
             OR more than 30% of beats in the segment are labeled 'artifact'
      - noisy_ok: has at least one artifact beat but not flagged as bad
      - clean: no artifact beats, no bad flags, at least 10 validated beats

    Args:
        seg_ranges: Per-segment timestamp ranges, {segment_idx: (min_ns, max_ns)},
            returned by stream_ecg_to_parquet(). Replaces the ecg_samples DataFrame.
        peaks_df: Canonical peaks table.
        labels_df: Canonical labels table.
        annotations: Parsed annotation JSON dict.
        recording_start_ns: Epoch ns of recording start.

    Returns:
        DataFrame with columns: segment_idx (int32), quality_label (str),
        start_timestamp_ns (int64), end_timestamp_ns (int64).
    """
    # Merge peaks with labels to get per-segment beat statistics
    peak_labels = peaks_df[["peak_id", "segment_idx"]].merge(
        labels_df[["peak_id", "label"]], on="peak_id", how="left"
    )

    seg_stats = peak_labels.groupby("segment_idx").agg(
        total_beats=("peak_id", "count"),
        artifact_beats=("label", lambda x: (x == "artifact").sum()),
        clean_beats=("label", lambda x: (x == "clean").sum()),
    )

    bad_seg_idxs = extract_bad_segment_indices(annotations, recording_start_ns)
    all_seg_idxs = sorted(seg_ranges.keys())

    records: list[dict[str, Any]] = []
    for seg_idx in all_seg_idxs:
        seg_int = int(seg_idx)

        # Timestamp range from pre-computed dict — no DataFrame needed
        if seg_int in seg_ranges:
            start_ns, end_ns = seg_ranges[seg_int]
        else:
            start_ns = recording_start_ns + seg_int * SEGMENT_DURATION_NS
            end_ns = recording_start_ns + (seg_int + 1) * SEGMENT_DURATION_NS

        # Beat statistics
        if seg_idx in seg_stats.index:
            total = int(seg_stats.loc[seg_idx, "total_beats"])
            n_artifact = int(seg_stats.loc[seg_idx, "artifact_beats"])
            n_clean = int(seg_stats.loc[seg_idx, "clean_beats"])
        else:
            total = 0
            n_artifact = 0
            n_clean = 0

        artifact_frac = n_artifact / total if total > 0 else 0.0

        # Quality assignment
        if seg_int in bad_seg_idxs or artifact_frac > ARTIFACT_FRACTION_BAD:
            quality = "bad"
        elif n_artifact > 0:
            quality = "noisy_ok"
        elif n_clean >= MIN_VALIDATED_BEATS_CLEAN:
            quality = "clean"
        else:
            quality = "noisy_ok"

        records.append(
            {
                "segment_idx": np.int32(seg_int),
                "quality_label": quality,
                "start_timestamp_ns": np.int64(start_ns),
                "end_timestamp_ns": np.int64(end_ns),
            }
        )

    result = pd.DataFrame(records)
    result["segment_idx"] = result["segment_idx"].astype(np.int32)
    result["start_timestamp_ns"] = result["start_timestamp_ns"].astype(np.int64)
    result["end_timestamp_ns"] = result["end_timestamp_ns"].astype(np.int64)

    dist = result["quality_label"].value_counts()
    logger.info("Segment quality distribution:\n%s", dist.to_string())
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
    nulls = int(peaks["timestamp_ns"].isna().sum())
    if nulls > 0:
        logger.error("VALIDATION FAIL: %d null timestamp_ns in peaks", nulls)
        ok = False

    # 5. Dtype checks
    dtype_checks = [
        ("timestamp_ns", "peaks", peaks),
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
    for ts_col in ("timestamp_ns", "start_timestamp_ns"):
        if ts_col in df.columns:
            ts_min = df[ts_col].min()
            ts_max = df[ts_col if ts_col == "timestamp_ns" else "end_timestamp_ns"].max()
            try:
                dt_min = pd.Timestamp(ts_min, unit="ns")
                dt_max = pd.Timestamp(ts_max, unit="ns")
                print(f"  Time range: {dt_min} -> {dt_max}")
            except (ValueError, OverflowError):
                print(f"  Time range: {ts_min} -> {ts_max} (raw ns)")
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
        help="Path to artifact_annotation.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for Parquet files (default: data/processed/)",
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
        default=min(4, (os.cpu_count() or 4)),
        help="Number of parallel workers for ECG file processing (default: min(4, cpu_count))",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only the first N ECG files (for debugging; default: all)",
    )
    args = parser.parse_args()

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

    # ── Load annotations ───────────────────────────────────────────────────
    annotations = load_annotations(args.annotations)

    # ── Stream ECG files → ecg_samples.parquet ────────────────────────────
    # Each file is processed independently; RAM use is bounded to one file
    # at a time. Returns seg_ranges dict for downstream use.
    print("\n>> Streaming ECG files → ecg_samples.parquet (no full-dataset concat)...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ecg_samples_path = args.output_dir / "ecg_samples.parquet"
    seg_ranges = stream_ecg_to_parquet(
        args.ecg_dir, ecg_samples_path, recording_start_ns,
        resume_partial=args.resume_partial,
        n_workers=args.workers,
        max_files=args.max_files,
    )

    # ── Build peaks ────────────────────────────────────────────────────────
    print("\n>> Building peaks table...")
    peaks = build_peaks(peak_csv_df, annotations, recording_start_ns)

    # ── Build labels ───────────────────────────────────────────────────────
    print("\n>> Building labels table...")
    validated_seg_idxs = extract_validated_segment_indices(annotations, recording_start_ns)
    bad_region_ranges = extract_bad_region_time_ranges(annotations, recording_start_ns)
    labels = build_labels(peaks, peak_csv_df, annotations, validated_seg_idxs, bad_region_ranges)

    # ── Build segments ─────────────────────────────────────────────────────
    # Uses seg_ranges dict instead of loading ecg_samples.parquet back into RAM.
    print("\n>> Building segments table...")
    segments = build_segments(seg_ranges, peaks, labels, annotations, recording_start_ns)

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
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete  |  Validation: {status}")
    print(f"  Output: {args.output_dir.resolve()}")
    print(f"{'=' * 60}\n")

    if not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
