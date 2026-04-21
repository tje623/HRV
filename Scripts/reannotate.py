#!/usr/bin/env python3
"""
reannotate.py — Interactive ECG browser for master annotation review.

Controls
--------
Default master mode:
    ] / →             Next annotated segment/window
    [ / ←             Previous annotated segment/window
    scroll            Zoom around cursor
    R                 Add current segment/window to revisit pile; hidden unless --r
    V                 Mark current window done; hidden unless --v
    Z                 Undo last JSON edit
    X                 Redo last undone JSON edit
    N                 Mark all negative current R-peaks in this window as artifacts
    G                 Jump to next abnormal RR interval in this window
    I                 Toggle interpolation mode
    O                 Mark whole segment bad; hidden from normal review
    P                 Toggle two-click bad-region mode
    F                 Toggle click-to-add phys_event mode
    A                 Toggle click-to-add missed_original mode (optional)
    Click             clean beat → artifact; artifact → clean
                      other annotated beat → remove annotation
                      empty waveform → add missed_original
    /                 Jump to next interpolated beat
    .                 Jump to next added/missed-original beat
    ,                 Jump to next artifact beat
    Space             Jump to next artifact/interpolated/added beat
    q                 Quit

Legacy marker mode (--marker-mode):
    ←  / →           Pan view 20%
    scroll            Zoom (vertical) or pan (horizontal)
    click             Click near an R-peak to select / deselect it
    B                 Confirm selected beats → type slot number + Enter to assign tag
    T                 Tag a segment region: click START → click END → type number + Enter
    X                 Enter delete-beat mode: click an annotated beat to remove it
                      Press X again (while in delete mode) to clear all SEGMENT
                      annotations visible in the current window
    D                 Delete the most recent annotation in the current window
    N                 Define / rename a tag slot: type 1-20 + Enter, then name + Enter
    Escape            Cancel current operation; clear beat selections in BROWSE
    Space / ]         Next marker (or next annotation window in --annotations-only mode)
    [                 Previous marker / window
    2 × 1–9           Double-press a legacy number to rename that legacy theme label
    2 × Enter         Export screenshots of all annotations (± 5-beat context)
    2 × Backspace     Clear ALL annotations (beats + segments) in visible window
    Q                 Save and quit

Tag assignment (in AWAIT BEAT THEME or AWAIT SEG THEME)
--------------------------------------------------------
    Type 1–20 + Enter  → assign new tag slot 1–20
    Backspace           → clear number buffer
    Escape              → cancel

    Slots 1–20 are new tags defined with N.
    Legacy themes 1–9 are shown in the "Legacy" box for reference (existing
    annotations remain visible) but cannot receive new assignments.

Output
------
    /Volumes/xHRV/Accessory/marker_annotations/
        segment_annotations.csv
        beat_annotations.csv
        theme_labels.json        ← legacy 1–9 theme names
        tag_labels.json          ← new tag slot names (1–20)
        screenshots/             ← PNG exports (double-press Enter)

Usage
-----
    python reannotate.py
    python reannotate.py --annotation-file v1_annotations.json --ecg-dir /Volumes/xHRV/ECG
    python reannotate.py --edit-file artifact_annotations_master_review.json
    python reannotate.py --start 42
    python reannotate.py --clean
    python reannotate.py --marker-mode --processed-dir /Volumes/xHRV/processed/

Legacy marker-mode examples:
    python marker_viewer.py --processed-dir /Volumes/xHRV/processed/
    python marker_viewer.py --processed-dir /Volumes/xHRV/processed/ --start 42
    python marker_viewer.py --processed-dir /Volumes/xHRV/processed/ --order asc
    python marker_viewer.py --processed-dir /Volumes/xHRV/processed/ --annotations-only
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
import time as _time
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import pandas as pd
except ModuleNotFoundError:  # The master annotation viewer does not require pandas.
    pd = None

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # Processed peaks are optional in master mode.
    pq = None

from config import (
    SAMPLE_RATE_HZ,
    PEAK_SNAP_SAMPLES,
    MARKER_CSV,
    ECG_DIR,
    PROCESSED_DIR,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MASTER_ANNOTATIONS = Path("/Volumes/xHRV/Annotations/V1/V1_Input.json")
DEFAULT_ECG_DIR = ECG_DIR
DEFAULT_PROCESSED_DIR = PROCESSED_DIR
DISPLAY_MS   = int(60 * 1000)
MIN_VIEW_MS  = int(10 * 1000)
MASTER_MIN_VIEW_MS = int(1 * 1000)
MAX_VIEW_MS  = int(120 * 1000)
ZOOM_FACTOR  = 1.15
RR_SHOW_MS   = int(30 * 1000)
EDIT_HIT_TEST_PIXELS = 44.0
CLEAN_HIT_TEST_PIXELS = 34.0
ADDED_R_PEAK_DEDUP_MS = 1
ARTIFACT_PEAK_MATCH_MS = 100
MISSED_PEAK_MATCH_MS = 10
MAX_UNDO_STATES = 100
LOCAL_TZ     = ZoneInfo("America/New_York")

_early_cutoff_dt = datetime(2025, 8, 1, tzinfo=LOCAL_TZ)
EARLY_CUTOFF_MS  = int(_early_cutoff_dt.astimezone(timezone.utc).timestamp() * 1000)
SEARCH_STEPS_RECENT = [int(5  * 60 * 1000)]
SEARCH_STEPS_EARLY  = [int(5  * 60 * 1000),
                        int(10 * 60 * 1000),
                        int(15 * 60 * 1000)]

OUTPUT_DIR = Path("/Volumes/xHRV/Annotations/V1/")

MASTER_LABEL_COLORS = {
    "artifact": "#ff4d6d",
    "interpolated": "#ffb703",
    "to_interpolate": "#ff7f11",
    "interpolation_preview": "#ffe066",
    "missed_original": "#b388ff",
    "phys_event": "#00d4ff",
    "reviewed": "#7bd88f",
}
MASTER_LABEL_MARKERS = {
    "artifact": "x",
    "interpolated": "ø",
    "to_interpolate": "D",
    "interpolation_preview": "o",
    "missed_original": "P",
    "phys_event": "*",
    "reviewed": "o",
}
MASTER_LABEL_NAMES = {
    "artifact": "Artifact",
    "interpolated": "Interpolated",
    "to_interpolate": "To_interpolate",
    "interpolation_preview": "Interpolation",
    "missed_original": "Added_peak",
    "phys_event": "Physio",
    "reviewed": "Clean",
}
POINT_LABEL_BY_FIELD = {
    "artifacts": "artifact",
    "interpolate_peaks": "interpolated",
    "physiological_events": "phys_event",
}
REVIEW_LABELS = {
    "artifact",
    "interpolated",
    "missed_original",
    "phys_event",
    "to_interpolate",
}
JUMP_LABELS = {
    "slash": "interpolated",
    "/": "interpolated",
    "period": "missed_original",
    ".": "missed_original",
    "comma": "artifact",
    ",": "artifact",
}
REVIEW_COMPLETED_KEY = "review_completed_windows"
ORPHAN_REVISIT_KEY = "revisit_windows"
TO_INTERPOLATE_KEY = "to_interpolate"
BAD_REGIONS_WITHIN_SEGMENTS_KEY = "bad_regions_within_segments"

# Legacy theme colors (theme_id 1-9, stored as-is in CSV)
THEME_COLORS = {
    1: "#ff6b6b", 2: "#ffd43b", 3: "#69db7c", 4: "#4fc3f7", 5: "#da77f2",
    6: "#ff922b", 7: "#38d9a9", 8: "#f783ac", 9: "#a9e34b",
}
DEFAULT_THEME_LABELS = {str(k): f"Theme {k}" for k in range(1, 10)}

# New tag slot colors (slot 1-20, stored as theme_id = TAG_OFFSET + slot)
TAG_OFFSET = 100   # theme_id 101 = slot 1 … 120 = slot 20
TAG_COLORS = {
     1: "#e63946",  2: "#f4a261",  3: "#2a9d8f",  4: "#457b9d",  5: "#a8dadc",
     6: "#e9c46a",  7: "#606c38",  8: "#bc6c25",  9: "#c77dff", 10: "#48cae4",
    11: "#f72585", 12: "#7209b7", 13: "#3a0ca3", 14: "#4cc9f0", 15: "#8338ec",
    16: "#fb5607", 17: "#ffbe0b", 18: "#06d6a0", 19: "#ef233c", 20: "#adb5bd",
}

SNAP_THRESHOLD_MS  = 2000
_DOUBLE_PRESS_SEC  = 0.40

# Annotations-only mode: cluster annotation timestamps within this window
_ANN_MERGE_MS = 60_000   # 60 s — merge centers within this span


# ── Peak snap helper ──────────────────────────────────────────────────────────

def _snap_peaks(
    pk_ns: np.ndarray,
    ecg_ts: np.ndarray,
    ecg_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap peak timestamps to the nearest local |ECG| maximum."""
    n = len(ecg_ts)
    out_ns = np.empty(len(pk_ns), dtype=np.int64)
    out_y  = np.zeros(len(pk_ns), dtype=np.float32)

    for i, ts in enumerate(pk_ns):
        pos = int(np.searchsorted(ecg_ts, ts))
        pos = min(max(pos, 0), n - 1)
        if 0 < pos < n:
            if abs(int(ecg_ts[pos - 1]) - int(ts)) < abs(int(ecg_ts[pos]) - int(ts)):
                pos = pos - 1
        lo   = max(0, pos - PEAK_SNAP_SAMPLES)
        hi   = min(n, pos + PEAK_SNAP_SAMPLES + 1)
        snap = lo + int(np.argmax(np.abs(ecg_vals[lo:hi])))
        out_ns[i] = ecg_ts[snap]
        out_y[i]  = ecg_vals[snap]

    return out_ns, out_y


def _nearest_points_to_ecg(
    point_ns: np.ndarray,
    ecg_ts: np.ndarray,
    ecg_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map timestamps to the nearest raw ECG sample without peak snapping."""
    if len(point_ns) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    if len(ecg_ts) == 0:
        return point_ns, np.zeros(len(point_ns), dtype=np.float32)

    out_ns = np.empty(len(point_ns), dtype=np.int64)
    out_y = np.empty(len(point_ns), dtype=np.float32)
    for i, ts_ns in enumerate(point_ns):
        pos = int(np.searchsorted(ecg_ts, ts_ns))
        if pos <= 0:
            nearest = 0
        elif pos >= len(ecg_ts):
            nearest = len(ecg_ts) - 1
        else:
            before = pos - 1
            after = pos
            nearest = (
                before
                if abs(int(ecg_ts[before]) - int(ts_ns))
                <= abs(int(ecg_ts[after]) - int(ts_ns))
                else after
            )
        out_ns[i] = ecg_ts[nearest]
        out_y[i] = ecg_vals[nearest]
    return out_ns, out_y


# ── Data helpers ──────────────────────────────────────────────────────────────

def _to_ms(dt_local: datetime) -> int:
    return int(dt_local.astimezone(timezone.utc).timestamp() * 1000)


def _load_markers() -> list[dict]:
    df = pd.read_csv(MARKER_CSV)
    col = df.columns[0]
    out = []
    for raw in df[col]:
        try:
            dt = datetime.strptime(str(raw).strip(), "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=LOCAL_TZ)
            out.append({"dt": dt, "ms": _to_ms(dt), "raw": str(raw).strip()})
        except ValueError:
            continue
    return out


def _load_segments(processed: Path) -> pd.DataFrame:
    return pq.read_table(
        processed / "segments.parquet",
        columns=["segment_idx", "start_timestamp_ms", "end_timestamp_ms"],
    ).to_pandas()


def _find_segment(marker_ms: int, segments: pd.DataFrame,
                  search_steps: list[int]) -> tuple[int | None, int]:
    for radius_ms in search_steps:
        lo, hi = marker_ms - radius_ms, marker_ms + radius_ms
        hits = segments[
            (segments["end_timestamp_ms"] >= lo) &
            (segments["start_timestamp_ms"] <= hi)
        ].copy()
        if len(hits) == 0:
            continue
        hits["mid_ms"] = (hits["start_timestamp_ms"] + hits["end_timestamp_ms"]) // 2
        hits["dist"]   = (hits["mid_ms"] - marker_ms).abs()
        best = hits.loc[hits["dist"].idxmin()]
        return int(best["segment_idx"]), int(best["dist"] / 1000)
    return None, -1


def _load_ecg(processed: Path, seg_idx: int,
              lo_ms: int, hi_ms: int) -> tuple[np.ndarray, np.ndarray]:
    tbl = pq.read_table(
        processed / "ecg_samples.parquet",
        filters=[
            ("segment_idx", "=", seg_idx),
            ("timestamp_ms", ">=", lo_ms),
            ("timestamp_ms", "<=", hi_ms),
        ],
        columns=["timestamp_ms", "ecg"],
    ).to_pandas().sort_values("timestamp_ms")
    return (tbl["timestamp_ms"].values.astype(np.int64),
            tbl["ecg"].values.astype(np.float32))


def _load_peaks(processed: Path, lo_ms: int, hi_ms: int) -> pd.DataFrame:
    tbl = pq.read_table(
        processed / "peaks.parquet",
        filters=[
            ("timestamp_ms", ">=", lo_ms),
            ("timestamp_ms", "<=", hi_ms),
        ],
        columns=["peak_id", "timestamp_ms"],
    ).to_pandas()
    tbl["peak_id"]      = tbl["peak_id"].astype(np.int64)
    tbl["timestamp_ms"] = tbl["timestamp_ms"].astype(np.int64)
    return tbl.sort_values("timestamp_ms").reset_index(drop=True)


# ── Master annotation / raw ECG helpers ──────────────────────────────────────

@dataclass
class ECGCsvFile:
    path: Path
    start_ms: int
    end_ms: int
    data_offset: int
    size: int


@dataclass
class MasterPoint:
    timestamp_ns: int
    label: str
    source: str
    peak_id: int | None = None
    raw_value: int | str | dict | None = None


@dataclass
class MasterWindow:
    start_ns: int
    end_ns: int
    title: str
    segment_idx: int | None = None
    points: list[MasterPoint] = field(default_factory=list)
    reviewed: bool = False
    bad: bool = False
    return_to_pile: bool = False
    bad_regions: list[tuple[int, int]] = field(default_factory=list)

    @property
    def center_ns(self) -> int:
        return (self.start_ns + self.end_ns) // 2


@dataclass
class MasterHistoryState:
    annotation_data: dict
    anchor_ns: int
    view_width_ns: int


def _parse_annotation_iso_ns(s: str) -> int:
    date, time_part = s.split("T")
    year, month, day = map(int, date.split("-"))
    if "." in time_part:
        hms, frac = time_part.split(".", 1)
    else:
        hms, frac = time_part, ""
    hour, minute, second = map(int, hms.split(":"))
    dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1_000 + int((frac + "000")[:3])


def _format_annotation_iso_ns(ms: int) -> str:
    whole, frac = divmod(int(ms), 1_000)
    dt = datetime.fromtimestamp(whole, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{frac:03d}"


def _ns_to_local_label(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _annotation_day_month(ts_ms: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m")


def _read_last_line(path: Path) -> bytes:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return b""
        block = min(size, 8192)
        f.seek(size - block)
        lines = f.read().splitlines()
    for line in reversed(lines):
        if line.strip():
            return line
    return b""


def _index_ecg_csvs(ecg_dir: Path) -> list[ECGCsvFile]:
    files: list[ECGCsvFile] = []
    for path in sorted(ecg_dir.glob("*.csv")):
        try:
            with open(path, "rb") as f:
                header = f.readline()
                first = f.readline()
                data_offset = len(header)
                f.seek(0, 2)
                size = f.tell()
            last = _read_last_line(path)
            if not first or not last:
                continue
            start_ms = int(first.split(b",", 1)[0])
            end_ms = int(last.split(b",", 1)[0])
        except (OSError, ValueError, IndexError):
            continue
        if end_ms < start_ms:
            continue
        files.append(ECGCsvFile(path, start_ms, end_ms, data_offset, size))
    return files


def _csv_seek_timestamp(f, meta: ECGCsvFile, target_ms: int) -> int:
    if target_ms <= meta.start_ms:
        return meta.data_offset
    if target_ms > meta.end_ms:
        return meta.size

    lo = meta.data_offset
    hi = meta.size
    best = meta.size
    while lo < hi:
        mid = (lo + hi) // 2
        f.seek(mid)
        if mid > meta.data_offset:
            f.readline()
        pos = f.tell()
        line = f.readline()
        if not line:
            hi = mid
            continue
        try:
            ts_ms = int(line.split(b",", 1)[0])
        except (ValueError, IndexError):
            lo = f.tell()
            continue
        if ts_ms < target_ms:
            lo = f.tell()
        else:
            best = pos
            hi = mid
    return best


def _load_raw_ecg(
    csv_index: list[ECGCsvFile],
    lo_ms: int,
    hi_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    ts_parts: list[np.ndarray] = []
    ecg_parts: list[np.ndarray] = []

    for meta in csv_index:
        if meta.end_ms < lo_ms or meta.start_ms > hi_ms:
            continue
        ts_vals: list[int] = []
        ecg_vals: list[float] = []
        try:
            with open(meta.path, "rb") as f:
                f.seek(_csv_seek_timestamp(f, meta, lo_ms))
                while True:
                    line = f.readline()
                    if not line:
                        break
                    try:
                        ts_raw, ecg_raw = line.strip().split(b",", 1)
                        ts_ms = int(ts_raw)
                    except (ValueError, IndexError):
                        continue
                    if ts_ms > hi_ms:
                        break
                    if ts_ms >= lo_ms:
                        ts_vals.append(ts_ms)
                        ecg_vals.append(float(ecg_raw))
        except OSError:
            continue
        if ts_vals:
            ts_parts.append(np.asarray(ts_vals, dtype=np.int64))
            ecg_parts.append(np.asarray(ecg_vals, dtype=np.float32))

    if not ts_parts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    ts = np.concatenate(ts_parts)
    ecg = np.concatenate(ecg_parts)
    order = np.argsort(ts)
    return ts[order], ecg[order]


def _load_processed_peaks(
    processed: Path | None,
    lo_ms: int,
    hi_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    if pq is None or processed is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    peak_path = processed / "peaks.parquet"
    if not peak_path.exists():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    try:
        table = pq.read_table(
            peak_path,
            filters=[
                ("timestamp_ms", ">=", lo_ms),
                ("timestamp_ms", "<=", hi_ms),
            ],
            columns=["peak_id", "timestamp_ms"],
        )
    except Exception as exc:
        print(f"WARNING: unable to load processed peaks: {exc}")
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if table.num_rows == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    peak_ids = np.asarray(table.column("peak_id").to_numpy(zero_copy_only=False),
                          dtype=np.int64)
    peak_ts = np.asarray(table.column("timestamp_ms").to_numpy(zero_copy_only=False),
                         dtype=np.int64)
    order = np.argsort(peak_ts)
    return peak_ids[order], peak_ts[order]


def _load_all_processed_peak_ts(processed: Path | None) -> np.ndarray:
    if pq is None or processed is None:
        return np.array([], dtype=np.int64)
    peak_path = processed / "peaks.parquet"
    if not peak_path.exists():
        return np.array([], dtype=np.int64)
    try:
        table = pq.read_table(peak_path, columns=["timestamp_ms"])
    except Exception as exc:
        print(f"WARNING: unable to load all processed peaks: {exc}")
        return np.array([], dtype=np.int64)
    if table.num_rows == 0:
        return np.array([], dtype=np.int64)
    peak_ts = np.asarray(table.column("timestamp_ms").to_numpy(zero_copy_only=False),
                         dtype=np.int64)
    return np.sort(peak_ts)


def _nearest_peak_within(
    peak_ts: np.ndarray,
    ts_ms: int,
    tolerance_ms: int = ARTIFACT_PEAK_MATCH_MS,
) -> int | None:
    if len(peak_ts) == 0:
        return None
    pos = int(np.searchsorted(peak_ts, ts_ms))
    candidates = []
    if pos < len(peak_ts):
        candidates.append(int(peak_ts[pos]))
    if pos > 0:
        candidates.append(int(peak_ts[pos - 1]))
    if not candidates:
        return None
    nearest = min(candidates, key=lambda candidate: abs(candidate - int(ts_ms)))
    if abs(nearest - int(ts_ms)) <= tolerance_ms:
        return nearest
    return None


def _has_peak_within(
    peak_ts: np.ndarray,
    ts_ms: int,
    tolerance_ms: int = ARTIFACT_PEAK_MATCH_MS,
) -> bool:
    return _nearest_peak_within(peak_ts, ts_ms, tolerance_ms) is not None


def _snap_points_to_ecg(
    point_ns: np.ndarray,
    ecg_ts: np.ndarray,
    ecg_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(point_ns) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    if len(ecg_ts) == 0:
        return point_ns, np.zeros(len(point_ns), dtype=np.float32)
    return _snap_peaks(point_ns.astype(np.int64), ecg_ts, ecg_vals)


class MasterAnnotationViewer:
    """Browse only master-annotation windows using raw ECG CSV files."""

    def __init__(
        self,
        annotation_file: Path,
        ecg_dir: Path,
        processed: Path | None = None,
        start_idx: int = 0,
        source_annotation_file: Path | None = None,
        include_validated: bool = False,
        include_revisit: bool = False,
        show_clean_only: bool = False,
    ) -> None:
        self.annotation_file = annotation_file
        self.source_annotation_file = source_annotation_file
        self.ecg_dir = ecg_dir
        self.processed = processed if processed and processed.exists() else None
        self.include_validated = include_validated
        self.include_revisit = include_revisit
        self.show_clean_only = show_clean_only
        self.annotation_data = json.loads(annotation_file.read_text())
        self.undo_stack: list[MasterHistoryState] = []
        self.redo_stack: list[MasterHistoryState] = []
        self.edit_mode = False
        self.pending_add_label: str | None = None
        self.pending_bad_region_start_ns: int | None = None
        self.jump_cursors: dict[str, int] = {}
        self.rawless_window_keys: set[str] = set()
        self.dirty = False
        self.last_edit_message = f"Editing copy: {annotation_file.name}"

        print(f"Indexing ECG CSVs in {ecg_dir} ...")
        self.csv_index = _index_ecg_csvs(ecg_dir)
        if not self.csv_index:
            print(f"ERROR: no usable ECG CSVs found in {ecg_dir}")
            sys.exit(1)

        self.segment_by_idx, self.segment_intervals = self._build_segments()
        self.segment_starts = [row[0] for row in self.segment_intervals]
        self.current_peak_ts = _load_all_processed_peak_ts(self.processed)
        dropped_artifacts, remapped_artifacts, dropped_added, stale_changed = (
            self._prune_stale_annotations_against_current_peaks()
        )
        if stale_changed:
            self._save_working_annotations()
            self.last_edit_message = (
                f"Pruned {dropped_artifacts} phantom artifact(s), "
                f"remapped {remapped_artifacts} artifact(s), "
                f"{dropped_added} no-longer-missed peak(s)"
            )
            print(self.last_edit_message)
        elif self.processed is not None and len(self.current_peak_ts) == 0:
            print("WARNING: current processed peaks unavailable; stale annotation "
                  "pruning skipped.")
        self.points = self._build_points()
        self.windows = self._build_windows()
        if not self.windows:
            print("ERROR: no matching windows found. Use --clean for clean-only "
                  "segments, --v for done windows, or --r for revisit windows.")
            sys.exit(1)

        self.window_idx = max(0, min(start_idx, len(self.windows) - 1))
        self.view_center_ns = self.windows[self.window_idx].center_ns
        self.view_width_ns = max(DISPLAY_MS, self.windows[self.window_idx].end_ns -
                                 self.windows[self.window_idx].start_ns)
        self.ecg_ts = np.array([], dtype=np.int64)
        self.ecg_vals = np.array([], dtype=np.float32)
        self.peak_ids = np.array([], dtype=np.int64)
        self.peak_ts = np.array([], dtype=np.int64)

        self.jump_targets = self._build_jump_targets()
        self._build_figure()
        self._go_to_window(self.window_idx)

    def _build_segments(
        self,
    ) -> tuple[dict[int, dict], list[tuple[int, int, int]]]:
        by_idx: dict[int, dict] = {}
        intervals: list[tuple[int, int, int]] = []
        for row in self.annotation_data.get("segment_timestamps", []):
            idx = int(row["segment_idx"])
            start_ns = _parse_annotation_iso_ns(row["first_timestamp"])
            end_ns = _parse_annotation_iso_ns(row["last_timestamp"])
            by_idx[idx] = {
                "segment_idx": idx,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "first_timestamp": row["first_timestamp"],
                "last_timestamp": row["last_timestamp"],
            }
            intervals.append((start_ns, end_ns, idx))
        intervals.sort()
        return by_idx, intervals

    def _segment_for_ts(self, ts_ns: int) -> int | None:
        pos = bisect_right(self.segment_starts, ts_ns) - 1
        for j in range(max(0, pos - 2), min(len(self.segment_intervals), pos + 3)):
            start_ns, end_ns, idx = self.segment_intervals[j]
            if start_ns <= ts_ns <= end_ns:
                return idx
        return None

    def _prune_stale_annotations_against_current_peaks(
        self,
    ) -> tuple[int, int, int, bool]:
        if len(getattr(self, "current_peak_ts", [])) == 0:
            return 0, 0, 0, False

        artifacts = [int(v) for v in self.annotation_data.get("artifacts", [])]
        matched_artifacts = []
        dropped_artifacts = 0
        remapped_artifacts = 0
        for peak_id in artifacts:
            matched_ms = _nearest_peak_within(
                self.current_peak_ts,
                peak_id,
                ARTIFACT_PEAK_MATCH_MS,
            )
            if matched_ms is None:
                dropped_artifacts += 1
            else:
                matched_peak_id = int(matched_ms)
                if matched_peak_id != peak_id:
                    remapped_artifacts += 1
                matched_artifacts.append(matched_peak_id)
        kept_artifacts = sorted(set(matched_artifacts))
        changed_artifacts = (
            dropped_artifacts > 0 or kept_artifacts != sorted(set(artifacts))
        )
        if changed_artifacts:
            self.annotation_data["artifacts"] = kept_artifacts

        excluded_ns = self._excluded_current_peak_ns()
        added = [str(v) for v in self.annotation_data.get("added_r_peaks", [])]
        kept_added = [
            ts for ts in added
            if (
                self._non_excluded_peak_near(
                    self.current_peak_ts,
                    _parse_annotation_iso_ns(ts),
                    excluded_ns,
                    MISSED_PEAK_MATCH_MS,
                )
                is None
            )
        ]
        dropped_added = len(added) - len(kept_added)
        if dropped_added:
            self.annotation_data["added_r_peaks"] = sorted(
                kept_added,
                key=_parse_annotation_iso_ns,
            )

        return (
            dropped_artifacts,
            remapped_artifacts,
            dropped_added,
            bool(changed_artifacts or dropped_added),
        )

    def _build_points(self) -> list[MasterPoint]:
        points: list[MasterPoint] = []
        for field_name, label in POINT_LABEL_BY_FIELD.items():
            for peak_id in self.annotation_data.get(field_name, []):
                peak_id = int(peak_id)
                points.append(MasterPoint(
                    timestamp_ns=peak_id,
                    peak_id=peak_id,
                    label=label,
                    source=field_name,
                    raw_value=peak_id,
                ))
        for ts in self.annotation_data.get("added_r_peaks", []):
            points.append(MasterPoint(
                timestamp_ns=_parse_annotation_iso_ns(ts),
                label="missed_original",
                source="added_r_peaks",
                raw_value=ts,
            ))
        for row in self.annotation_data.get(TO_INTERPOLATE_KEY, []):
            if not isinstance(row, dict):
                continue
            try:
                if row.get("peak_id") is not None:
                    peak_id = int(row["peak_id"])
                    timestamp_ns = peak_id
                else:
                    timestamp_ns = _parse_annotation_iso_ns(
                        str(row["original_timestamp"])
                    )
                    peak_id = None
            except (KeyError, TypeError, ValueError):
                continue
            points.append(MasterPoint(
                timestamp_ns=timestamp_ns,
                peak_id=peak_id,
                label="to_interpolate",
                source=TO_INTERPOLATE_KEY,
                raw_value=row,
            ))
        priority = {
            "to_interpolate": 0,
            "artifact": 1,
            "interpolated": 2,
            "missed_original": 3,
            "phys_event": 4,
        }
        return sorted(points, key=lambda p: (p.timestamp_ns, priority.get(p.label, 9)))

    def _review_window_key(self, window: MasterWindow) -> str:
        if window.segment_idx is not None:
            return f"segment:{window.segment_idx}"
        return f"orphan:{window.start_ns}:{window.end_ns}"

    def _completed_review_keys(self) -> set[str]:
        return {
            str(v) for v in self.annotation_data.get(REVIEW_COMPLETED_KEY, [])
        }

    def _is_review_completed(self, window: MasterWindow) -> bool:
        return self._review_window_key(window) in self._completed_review_keys()

    def _bad_segment_ids(self) -> set[int]:
        return {
            int(seg)
            for seg in self.annotation_data.get("bad_segments", [])
            if str(seg).lstrip("-").isdigit()
        }

    def _bad_regions_by_segment(self) -> dict[int, list[tuple[int, int]]]:
        regions: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for key in ("bad_regions", BAD_REGIONS_WITHIN_SEGMENTS_KEY):
            for row in self.annotation_data.get(key, []):
                if not isinstance(row, dict):
                    continue
                try:
                    seg_idx = int(row["segment_idx"])
                    start_ns = _parse_annotation_iso_ns(str(row["start_time"]))
                    end_ns = _parse_annotation_iso_ns(str(row["end_time"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if end_ns < start_ns:
                    start_ns, end_ns = end_ns, start_ns
                regions[seg_idx].append((start_ns, end_ns))
        for seg_idx, seg_regions in regions.items():
            regions[seg_idx] = sorted(seg_regions)
        return regions

    def _annotation_invalid_by_bad_area(
        self,
        ts_ns: int,
        segment_idx: int | None = None,
        bad_segments: set[int] | None = None,
        bad_regions_by_segment: dict[int, list[tuple[int, int]]] | None = None,
    ) -> bool:
        seg_idx = self._segment_for_ts(ts_ns) if segment_idx is None else segment_idx
        if seg_idx is None:
            return False
        bad_segments = self._bad_segment_ids() if bad_segments is None else bad_segments
        if int(seg_idx) in bad_segments:
            return True
        if bad_regions_by_segment is None:
            bad_regions_by_segment = self._bad_regions_by_segment()
        return any(
            start_ns <= int(ts_ns) <= end_ns
            for start_ns, end_ns in bad_regions_by_segment.get(int(seg_idx), [])
        )

    def _build_windows(self) -> list[MasterWindow]:
        points_by_segment: dict[int, list[MasterPoint]] = defaultdict(list)
        orphan_points: list[MasterPoint] = []
        for point in self.points:
            seg_idx = self._segment_for_ts(point.timestamp_ns)
            if seg_idx is None:
                orphan_points.append(point)
            else:
                points_by_segment[seg_idx].append(point)

        validated = {
            int(str(seg).split("_", 1)[1])
            for seg in self.annotation_data.get("validated_segments", [])
        }
        bad = self._bad_segment_ids()
        return_to_pile = {
            int(row["segment_idx"])
            for row in self.annotation_data.get("return_to_pile", [])
        }
        orphan_revisit = {
            str(v) for v in self.annotation_data.get(ORPHAN_REVISIT_KEY, [])
        }
        bad_regions_by_segment = self._bad_regions_by_segment()

        annotated_segments = set(points_by_segment) | set(bad_regions_by_segment)
        if self.show_clean_only:
            annotated_segments |= validated

        windows: list[MasterWindow] = []
        for seg_idx in sorted(annotated_segments):
            seg = self.segment_by_idx.get(seg_idx)
            if seg is None:
                continue
            title = f"Segment {seg_idx}"
            windows.append(MasterWindow(
                start_ns=seg["start_ns"],
                end_ns=seg["end_ns"],
                segment_idx=seg_idx,
                title=title,
                points=sorted(points_by_segment.get(seg_idx, []),
                              key=lambda p: p.timestamp_ns),
                reviewed=seg_idx in validated,
                bad=seg_idx in bad,
                return_to_pile=seg_idx in return_to_pile,
                bad_regions=bad_regions_by_segment.get(seg_idx, []),
            ))

        # Point-level annotations from orphan mini-files may not have matching
        # master segments. Keep them navigable as synthetic annotation windows.
        if orphan_points:
            clusters: list[list[MasterPoint]] = [[orphan_points[0]]]
            for point in orphan_points[1:]:
                if point.timestamp_ns - clusters[-1][-1].timestamp_ns <= _ANN_MERGE_MS:
                    clusters[-1].append(point)
                else:
                    clusters.append([point])
            for i, cluster in enumerate(clusters, start=1):
                start_ns = min(p.timestamp_ns for p in cluster) - DISPLAY_MS // 2
                end_ns = max(p.timestamp_ns for p in cluster) + DISPLAY_MS // 2
                window = MasterWindow(
                    start_ns=start_ns,
                    end_ns=end_ns,
                    segment_idx=None,
                    title=f"Orphan point window {i}",
                    points=cluster,
                )
                window.return_to_pile = (
                    self._review_window_key(window) in orphan_revisit
                )
                windows.append(window)

        # Segment review should only land on real master segments. Orphan point
        # annotations do not have segment bounds and can fall outside raw ECG
        # coverage, which makes the GUI draw markers over an empty waveform.
        windows = [w for w in windows if w.segment_idx is not None]
        windows = sorted(windows, key=lambda w: (w.start_ns, w.end_ns, w.segment_idx or -1))
        if self.show_clean_only:
            windows = [w for w in windows if self._is_clean_only_window(w)]
        else:
            windows = [w for w in windows if self._has_review_annotation(w)]
        windows = [w for w in windows if not w.bad]
        if not self.include_revisit:
            windows = [w for w in windows if not w.return_to_pile]
        if not self.include_validated:
            windows = [w for w in windows if not self._is_review_completed(w)]
        if getattr(self, "rawless_window_keys", None):
            windows = [
                w for w in windows
                if self._review_window_key(w) not in self.rawless_window_keys
            ]
        return windows

    def _has_review_annotation(self, window: MasterWindow) -> bool:
        return bool(window.bad_regions) or any(
            point.label in REVIEW_LABELS for point in window.points
        )

    def _is_clean_only_window(self, window: MasterWindow) -> bool:
        return (
            window.segment_idx is not None
            and window.reviewed
            and not self._has_review_annotation(window)
            and not window.bad
            and not window.bad_regions
        )

    def _build_jump_targets(self) -> dict[str, list[int]]:
        targets: dict[str, list[int]] = {label: [] for label in JUMP_LABELS.values()}
        windows = getattr(self, "windows", [])
        if windows:
            iterable = [point for window in windows for point in window.points]
        else:
            iterable = self.points
        for point in iterable:
            if point.label in targets:
                targets[point.label].append(point.timestamp_ns)
        for label in targets:
            targets[label] = sorted(set(targets[label]))
        return targets

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(28, 7), facecolor="#0d1117")
        gs = gridspec.GridSpec(1, 2, figure=self.fig,
                               width_ratios=[8, 1.5],
                               left=0.02, right=0.98,
                               top=0.88, bottom=0.10,
                               wspace=0.02)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_leg = self.fig.add_subplot(gs[1])
        self.ax.set_facecolor("#16213e")
        self.ax_leg.set_facecolor("#0d1117")
        self.ax_leg.axis("off")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _go_to_window(self, idx: int) -> None:
        if not self.windows:
            return
        desired = max(0, min(idx, len(self.windows) - 1))
        direction = 1 if desired >= self.window_idx else -1
        skipped = 0

        for start_idx, step in ((desired, direction), (desired - direction, -direction)):
            candidate = start_idx
            while 0 <= candidate < len(self.windows):
                self.window_idx = candidate
                window = self.windows[self.window_idx]
                self.view_center_ns = window.center_ns
                self.view_width_ns = max(DISPLAY_MS, window.end_ns - window.start_ns)
                self._load_window()
                if len(self.ecg_ts):
                    if skipped:
                        current_key = self._review_window_key(window)
                        self.windows = [
                            w for w in self.windows
                            if self._review_window_key(w)
                            not in self.rawless_window_keys
                        ]
                        self.window_idx = next(
                            (
                                i for i, w in enumerate(self.windows)
                                if self._review_window_key(w) == current_key
                            ),
                            self.window_idx,
                        )
                        self.jump_targets = self._build_jump_targets()
                        self.last_edit_message = (
                            f"Skipped {skipped} window(s) with no raw ECG samples"
                        )
                    self._draw()
                    return
                self.rawless_window_keys.add(self._review_window_key(window))
                skipped += 1
                candidate += step

        self.last_edit_message = "No visible windows have raw ECG samples"
        self._draw()

    def _go_to_time(self, ts_ns: int) -> None:
        starts = [w.start_ns for w in self.windows]
        pos = max(0, bisect_right(starts, ts_ns) - 1)
        candidates = list(range(max(0, pos - 2), min(len(self.windows), pos + 3)))
        containing = [
            i for i in candidates
            if self.windows[i].start_ns <= ts_ns <= self.windows[i].end_ns
        ]
        if containing:
            real_segments = [
                i for i in containing if self.windows[i].segment_idx is not None
            ]
            self._go_to_window(real_segments[0] if real_segments else containing[0])
            return
        centers = [abs(w.center_ns - ts_ns) for w in self.windows]
        self._go_to_window(int(np.argmin(centers)))

    def _jump_next(self, label: str) -> None:
        targets = self.jump_targets.get(label, [])
        if not targets:
            print(f"No {MASTER_LABEL_NAMES[label]} beats in master annotations.")
            return
        window = self.windows[self.window_idx]
        previous = self.jump_cursors.get(label)
        if previous is not None and window.start_ns <= previous <= window.end_ns:
            reference_ns = previous
        else:
            reference_ns = self.view_center_ns
        pos = bisect_right(targets, reference_ns)
        if pos >= len(targets):
            pos = 0
        self.jump_cursors[label] = targets[pos]
        self._go_to_time(targets[pos])

    def _jump_next_any(self) -> None:
        combined = sorted({
            ts
            for targets in self.jump_targets.values()
            for ts in targets
        })
        if not combined:
            print("No artifact, interpolated, or added beats in visible windows.")
            return
        previous = self.jump_cursors.get("any")
        window = self.windows[self.window_idx]
        if previous is not None and window.start_ns <= previous <= window.end_ns:
            reference_ns = previous
        else:
            reference_ns = self.view_center_ns
        pos = bisect_right(combined, reference_ns)
        if pos >= len(combined):
            pos = 0
        self.jump_cursors["any"] = combined[pos]
        self._go_to_time(combined[pos])

    def _load_window(self) -> None:
        half = self.view_width_ns // 2
        lo_ns = self.view_center_ns - half
        hi_ns = self.view_center_ns + half
        self.ecg_ts, self.ecg_vals = _load_raw_ecg(self.csv_index, lo_ns, hi_ns)
        self.peak_ids, self.peak_ts = _load_processed_peaks(self.processed, lo_ns, hi_ns)

    def _current_points(self, lo_ns: int, hi_ns: int) -> list[MasterPoint]:
        window = self.windows[self.window_idx]
        return [p for p in window.points if lo_ns <= p.timestamp_ns <= hi_ns]

    def _to_interpolate_original_ns(self, row: dict) -> int | None:
        try:
            if row.get("peak_id") is not None:
                return int(row["peak_id"])
            return _parse_annotation_iso_ns(str(row["original_timestamp"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _to_interpolate_replacement_ns(self, row: dict) -> int | None:
        try:
            replacement = row.get("replacement_timestamp")
            if not replacement:
                return None
            return _parse_annotation_iso_ns(str(replacement))
        except (TypeError, ValueError):
            return None

    def _clean_peak_mask(
        self,
        points: list[MasterPoint],
        window: MasterWindow,
        peak_ts: np.ndarray | None = None,
    ) -> np.ndarray:
        active_peak_ts = self.peak_ts if peak_ts is None else peak_ts
        if len(active_peak_ts) == 0:
            return np.zeros(len(active_peak_ts), dtype=bool)
        if window.segment_idx is None:
            half = self.view_width_ns // 2
            lo_ns = self.view_center_ns - half
            hi_ns = self.view_center_ns + half
        else:
            lo_ns = window.start_ns
            hi_ns = window.end_ns
        mask = (active_peak_ts >= lo_ns) & (active_peak_ts <= hi_ns)
        excluded_ns = self._excluded_current_peak_ns(points, active_peak_ts)
        if excluded_ns:
            mask &= ~np.isin(active_peak_ts, list(excluded_ns))
        return mask

    def _effective_rr_timestamps(
        self,
        points: list[MasterPoint],
        window: MasterWindow,
        peak_ts: np.ndarray | None = None,
    ) -> np.ndarray:
        active_peak_ts = self.peak_ts if peak_ts is None else peak_ts
        ts_parts: list[int] = []
        clean_mask = self._clean_peak_mask(points, window, active_peak_ts)
        if len(clean_mask):
            ts_parts.extend(int(ts) for ts in active_peak_ts[clean_mask])
        excluded_ns = self._excluded_current_peak_ns(points, active_peak_ts)
        ts_parts.extend(
            int(p.timestamp_ns)
            for p in points
            if p.label in {"interpolated", "missed_original", "phys_event"}
            and not (
                p.label == "missed_original"
                and self._non_excluded_peak_near(
                    active_peak_ts,
                    p.timestamp_ns,
                    excluded_ns,
                    MISSED_PEAK_MATCH_MS,
                ) is not None
            )
        )
        for point in points:
            if point.label != "to_interpolate" or not isinstance(point.raw_value, dict):
                continue
            replacement_ns = self._to_interpolate_replacement_ns(point.raw_value)
            if replacement_ns is not None and window.start_ns <= replacement_ns <= window.end_ns:
                ts_parts.append(int(replacement_ns))
        if not ts_parts:
            return np.array([], dtype=np.int64)
        ts_parts.sort()
        deduped = [ts_parts[0]]
        for ts_ns in ts_parts[1:]:
            if ts_ns - deduped[-1] > ADDED_R_PEAK_DEDUP_MS:
                deduped.append(ts_ns)
        return np.asarray(deduped, dtype=np.int64)

    @staticmethod
    def _rr_deviation_kind(rr_ms: int, neighbors: list[int]) -> str | None:
        if not neighbors:
            return None
        baseline = float(np.median(neighbors))
        diff = abs(float(rr_ms) - baseline)
        if baseline <= 0 or diff < 120:
            return None
        if rr_ms >= baseline * 1.25:
            return "long"
        if rr_ms <= baseline * 0.75:
            return "short"
        return None

    @staticmethod
    def _mix_hex_colors(start_hex: str, end_hex: str, amount: float) -> str:
        amount = max(0.0, min(1.0, float(amount)))
        start = start_hex.lstrip("#")
        end = end_hex.lstrip("#")
        rgb = []
        for i in range(0, 6, 2):
            a = int(start[i:i + 2], 16)
            b = int(end[i:i + 2], 16)
            rgb.append(round(a + (b - a) * amount))
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    @staticmethod
    def _rr_deviation_gradient(rr_ms: int, neighbors: list[int]) -> tuple[str | None, float]:
        if not neighbors:
            return None, 0.0
        baseline = float(np.median(neighbors))
        if baseline <= 0:
            return None, 0.0
        delta = float(rr_ms) - baseline
        if delta == 0:
            return None, 0.0
        relative_score = abs(delta / baseline) / 0.35
        absolute_score = abs(delta) / 220.0
        score = max(relative_score, absolute_score)
        return ("long" if delta > 0 else "short"), max(0.0, min(1.0, score))

    def _abnormal_rr_targets(
        self,
        beat_ts: np.ndarray,
    ) -> list[tuple[int, int, str]]:
        if len(beat_ts) < 2:
            return []
        beat_ts = np.asarray(sorted(int(ts) for ts in beat_ts), dtype=np.int64)
        rr_values = np.asarray([
            int(round(int(beat_ts[i + 1]) - int(beat_ts[i])))
            for i in range(len(beat_ts) - 1)
        ], dtype=np.int64)
        targets: list[tuple[int, int, str]] = []
        for i, rr_ms in enumerate(rr_values):
            neighbors = []
            if i > 0:
                neighbors.append(int(rr_values[i - 1]))
            if i + 1 < len(rr_values):
                neighbors.append(int(rr_values[i + 1]))
            kind = self._rr_deviation_kind(int(rr_ms), neighbors)
            if kind is None:
                continue
            mid_ns = (int(beat_ts[i]) + int(beat_ts[i + 1])) // 2
            targets.append((mid_ns, int(rr_ms), kind))
        return targets

    def _refresh_annotations(self, anchor_ns: int | None = None) -> None:
        anchor_ns = self.view_center_ns if anchor_ns is None else anchor_ns
        keep_width = self.view_width_ns
        self.points = self._build_points()
        self.windows = self._build_windows()
        self.jump_targets = self._build_jump_targets()
        if not self.windows:
            print("ERROR: no annotated windows remain.")
            sys.exit(1)

        starts = [w.start_ns for w in self.windows]
        pos = max(0, bisect_right(starts, anchor_ns) - 1)
        candidates = list(range(max(0, pos - 2), min(len(self.windows), pos + 3)))
        chosen = None
        containing = [
            i for i in candidates
            if self.windows[i].start_ns <= anchor_ns <= self.windows[i].end_ns
        ]
        if containing:
            real_segments = [
                i for i in containing if self.windows[i].segment_idx is not None
            ]
            chosen = real_segments[0] if real_segments else containing[0]
        if chosen is None:
            distances = [abs(w.center_ns - anchor_ns) for w in self.windows]
            chosen = int(np.argmin(distances))
        self.window_idx = chosen
        chosen_window = self.windows[self.window_idx]
        if not (chosen_window.start_ns <= self.view_center_ns <= chosen_window.end_ns):
            if chosen_window.start_ns <= anchor_ns <= chosen_window.end_ns:
                self.view_center_ns = anchor_ns
            else:
                self.view_center_ns = chosen_window.center_ns
        self.view_width_ns = keep_width
        self._load_window()

    def _granular_point_stats(self, timestamps_ns: list[int]) -> dict:
        counts_by_day: Counter[str] = Counter()
        counts_by_month: Counter[str] = Counter()
        counts_by_segment: Counter[int] = Counter()
        for ts_ns in timestamps_ns:
            day, month = _annotation_day_month(ts_ns)
            counts_by_day[day] += 1
            counts_by_month[month] += 1
            seg_idx = self._segment_for_ts(ts_ns)
            if seg_idx is not None:
                counts_by_segment[seg_idx] += 1

        month_values = list(counts_by_month.values())
        max_per_segment = max(counts_by_segment.values(), default=0)
        segments_with_max = [
            int(seg) for seg, count in sorted(counts_by_segment.items())
            if count == max_per_segment and max_per_segment
        ]
        return {
            "total": len(timestamps_ns),
            "unique_days": len(counts_by_day),
            "max_per_segment": int(max_per_segment),
            "segments_with_max": segments_with_max,
            "average_per_month": (
                round(float(len(timestamps_ns) / len(month_values)), 3)
                if month_values else 0
            ),
            "median_per_month": (
                float(np.median(month_values)) if month_values else 0
            ),
            "counts_by_month": dict(sorted(counts_by_month.items())),
            "counts_by_day": dict(sorted(counts_by_day.items())),
        }

    def _update_statistics(self) -> None:
        stats = self.annotation_data.setdefault("statistics", {})
        bad_segments = self._bad_segment_ids()
        bad_regions_by_segment = self._bad_regions_by_segment()

        def is_valid_annotation(
            ts_ns: int,
            segment_idx: int | None = None,
        ) -> bool:
            return not self._annotation_invalid_by_bad_area(
                ts_ns,
                segment_idx=segment_idx,
                bad_segments=bad_segments,
                bad_regions_by_segment=bad_regions_by_segment,
            )

        artifacts_ns_all = [
            int(peak_id)
            for peak_id in self.annotation_data.get("artifacts", [])
        ]
        interpolated_ns_all = [
            int(peak_id)
            for peak_id in self.annotation_data.get("interpolate_peaks", [])
        ]
        events_ns_all = [
            int(peak_id)
            for peak_id in self.annotation_data.get("physiological_events", [])
        ]
        added_ns_all = [
            _parse_annotation_iso_ns(ts)
            for ts in self.annotation_data.get("added_r_peaks", [])
        ]
        to_interpolate_ns_all = []
        to_interpolate_total = 0
        for row in self.annotation_data.get(TO_INTERPOLATE_KEY, []):
            if not isinstance(row, dict):
                continue
            try:
                if row.get("peak_id") is not None:
                    ts_ns = int(row["peak_id"])
                else:
                    ts_ns = _parse_annotation_iso_ns(str(row["original_timestamp"]))
                seg_idx = (
                    int(row["segment_idx"])
                    if row.get("segment_idx") is not None
                    else None
                )
            except (KeyError, TypeError, ValueError):
                continue
            to_interpolate_total += 1
            if is_valid_annotation(ts_ns, segment_idx=seg_idx):
                to_interpolate_ns_all.append(ts_ns)

        artifacts_ns = [ts for ts in artifacts_ns_all if is_valid_annotation(ts)]
        interpolated_ns = [
            ts for ts in interpolated_ns_all if is_valid_annotation(ts)
        ]
        events_ns = [ts for ts in events_ns_all if is_valid_annotation(ts)]
        added_ns = [ts for ts in added_ns_all if is_valid_annotation(ts)]

        stats["total_artifacts"] = len(artifacts_ns)
        stats["flagged_for_interpolation"] = len(interpolated_ns)
        stats[TO_INTERPOLATE_KEY] = len(to_interpolate_ns_all)
        stats["added_r_peaks"] = len(added_ns)
        stats["poor_quality_segments"] = len(self.annotation_data.get("bad_segments", []))
        stats["total_events"] = len(events_ns)
        stats["validated_segments"] = len(
            self.annotation_data.get("validated_segments", [])
        )
        stats[REVIEW_COMPLETED_KEY] = len(
            self.annotation_data.get(REVIEW_COMPLETED_KEY, [])
        )
        stats["return_pile_size"] = len(self.annotation_data.get("return_to_pile", []))
        orphan_revisit_size = len(self.annotation_data.get(ORPHAN_REVISIT_KEY, []))
        stats["orphan_revisit_windows"] = orphan_revisit_size
        stats["revisit_windows"] = (
            stats["return_pile_size"] + orphan_revisit_size
        )
        stats["bad_regions_within_segments"] = len(
            self.annotation_data.get(BAD_REGIONS_WITHIN_SEGMENTS_KEY, [])
        )
        stats["partial_bad_regions"] = (
            len(self.annotation_data.get("bad_regions", []))
            + stats["bad_regions_within_segments"]
        )
        stats["segments_with_partial_bad"] = len({
            int(row["segment_idx"])
            for key in ("bad_regions", BAD_REGIONS_WITHIN_SEGMENTS_KEY)
            for row in self.annotation_data.get(key, [])
            if isinstance(row, dict) and row.get("segment_idx") is not None
        })
        stats["invalidated_by_bad_area"] = {
            "artifacts": len(artifacts_ns_all) - len(artifacts_ns),
            "interpolations": len(interpolated_ns_all) - len(interpolated_ns),
            TO_INTERPOLATE_KEY: to_interpolate_total - len(to_interpolate_ns_all),
            "events": len(events_ns_all) - len(events_ns),
            "added": len(added_ns_all) - len(added_ns),
        }
        stats["granular"] = {
            "artifacts": self._granular_point_stats(artifacts_ns),
            "interpolations": self._granular_point_stats(interpolated_ns),
            TO_INTERPOLATE_KEY: self._granular_point_stats(to_interpolate_ns_all),
            "events": self._granular_point_stats(events_ns),
            "added": self._granular_point_stats(added_ns),
        }
        self.annotation_data["last_modified"] = datetime.now().isoformat()

    def _save_working_annotations(self) -> None:
        self._update_statistics()
        with open(self.annotation_file, "w") as f:
            json.dump(self.annotation_data, f, indent=2)
            f.write("\n")
        self.dirty = False

    def _make_history_state(
        self,
        anchor_ns: int | None = None,
        view_width_ns: int | None = None,
    ) -> MasterHistoryState:
        return MasterHistoryState(
            annotation_data=copy.deepcopy(self.annotation_data),
            anchor_ns=int(self.view_center_ns if anchor_ns is None else anchor_ns),
            view_width_ns=int(
                self.view_width_ns if view_width_ns is None else view_width_ns
            ),
        )

    def _apply_annotation_mutation(
        self,
        mutator,
        anchor_ns: int | None = None,
        view_width_ns: int | None = None,
    ) -> bool:
        before = self._make_history_state(anchor_ns, view_width_ns)
        changed = bool(mutator())
        if not changed:
            return False
        self.undo_stack.append(before)
        if len(self.undo_stack) > MAX_UNDO_STATES:
            self.undo_stack = self.undo_stack[-MAX_UNDO_STATES:]
        self.redo_stack.clear()
        return True

    def _restore_history_state(self, state: MasterHistoryState, message: str) -> None:
        self.annotation_data = copy.deepcopy(state.annotation_data)
        self.view_center_ns = state.anchor_ns
        self.view_width_ns = state.view_width_ns
        self.pending_add_label = None
        self.pending_bad_region_start_ns = None
        self._save_working_annotations()
        self._refresh_annotations(state.anchor_ns)
        self.last_edit_message = message
        self._draw()

    def _undo_last_edit(self) -> None:
        if not self.undo_stack:
            self.last_edit_message = "Nothing to undo"
            self._draw()
            return
        self.redo_stack.append(self._make_history_state())
        state = self.undo_stack.pop()
        self._restore_history_state(state, "Undid last edit")

    def _redo_last_edit(self) -> None:
        if not self.redo_stack:
            self.last_edit_message = "Nothing to redo"
            self._draw()
            return
        self.undo_stack.append(self._make_history_state())
        if len(self.undo_stack) > MAX_UNDO_STATES:
            self.undo_stack = self.undo_stack[-MAX_UNDO_STATES:]
        state = self.redo_stack.pop()
        self._restore_history_state(state, "Redid edit")

    def _mark_current_window_done(self) -> None:
        if not self.windows:
            return
        window = self.windows[self.window_idx]
        key = self._review_window_key(window)
        completed_set = {
            str(v) for v in self.annotation_data.get(REVIEW_COMPLETED_KEY, [])
        }
        if key in completed_set:
            self.last_edit_message = "Window already marked done"
            self._draw()
            return

        def mutate() -> bool:
            self.annotation_data[REVIEW_COMPLETED_KEY] = sorted({
                *completed_set,
                key,
            })
            if window.segment_idx is not None:
                validated = {
                    str(v) for v in self.annotation_data.get("validated_segments", [])
                }
                validated.add(f"segment_{int(window.segment_idx)}")
                self.annotation_data["validated_segments"] = sorted(
                    validated,
                    key=lambda v: int(str(v).split("_", 1)[1]),
                )
            return True

        self._apply_annotation_mutation(mutate, anchor_ns=window.center_ns)
        self.pending_add_label = None
        self.pending_bad_region_start_ns = None
        self.edit_mode = False
        self.last_edit_message = f"Marked {window.title} done"
        current_idx = self.window_idx
        self._save_working_annotations()

        self.points = self._build_points()
        self.windows = self._build_windows()
        self.jump_targets = self._build_jump_targets()
        if not self.windows:
            print("All visible annotation windows are marked done. Re-run with "
                  "--v to review completed windows.")
            plt.close(self.fig)
            return
        if self.include_validated:
            self.window_idx = min(current_idx, len(self.windows) - 1)
        else:
            self.window_idx = min(current_idx, len(self.windows) - 1)
        window = self.windows[self.window_idx]
        self.view_center_ns = window.center_ns
        self.view_width_ns = max(DISPLAY_MS, window.end_ns - window.start_ns)
        self._load_window()
        self._draw()

    def _mark_current_window_revisit(self) -> None:
        if not self.windows:
            return
        window = self.windows[self.window_idx]
        if window.segment_idx is not None:
            existing = {
                int(row["segment_idx"])
                for row in self.annotation_data.get("return_to_pile", [])
            }
            if window.segment_idx in existing:
                self.last_edit_message = f"{window.title} already in revisit pile"
                self._draw()
                return

            def mutate() -> bool:
                self.annotation_data.setdefault("return_to_pile", []).append({
                    "segment_idx": int(window.segment_idx),
                    "start_time": _format_annotation_iso_ns(window.start_ns),
                    "end_time": _format_annotation_iso_ns(window.end_ns),
                    "marked_at": datetime.now().isoformat(),
                })
                self.annotation_data["return_to_pile"] = sorted(
                    self.annotation_data["return_to_pile"],
                    key=lambda row: int(row["segment_idx"]),
                )
                return True

        else:
            key = self._review_window_key(window)
            existing = {
                str(v) for v in self.annotation_data.get(ORPHAN_REVISIT_KEY, [])
            }
            if key in existing:
                self.last_edit_message = f"{window.title} already in revisit pile"
                self._draw()
                return

            def mutate() -> bool:
                self.annotation_data.setdefault(ORPHAN_REVISIT_KEY, []).append(key)
                self.annotation_data[ORPHAN_REVISIT_KEY] = sorted({
                    str(v) for v in self.annotation_data[ORPHAN_REVISIT_KEY]
                })
                return True

        self._apply_annotation_mutation(mutate, anchor_ns=window.center_ns)
        self.pending_add_label = None
        self.pending_bad_region_start_ns = None
        self.edit_mode = False
        self.last_edit_message = f"Added {window.title} to revisit pile"
        current_idx = self.window_idx
        self._save_working_annotations()
        self.points = self._build_points()
        self.windows = self._build_windows()
        self.jump_targets = self._build_jump_targets()
        if not self.windows:
            print("All visible annotation windows are in the revisit pile. "
                  "Re-run with --r to review revisit windows.")
            plt.close(self.fig)
            return
        self.window_idx = min(current_idx, len(self.windows) - 1)
        window = self.windows[self.window_idx]
        self.view_center_ns = window.center_ns
        self.view_width_ns = max(DISPLAY_MS, window.end_ns - window.start_ns)
        self._load_window()
        self._draw()

    def _mark_current_window_bad(self) -> None:
        if not self.windows:
            return
        window = self.windows[self.window_idx]
        if window.segment_idx is None:
            self.last_edit_message = "Cannot mark orphan window as bad segment"
            self._draw()
            return
        existing = self._bad_segment_ids()
        if int(window.segment_idx) in existing:
            self.last_edit_message = f"{window.title} already marked bad"
            self._draw()
            return

        def mutate() -> bool:
            self.annotation_data["bad_segments"] = sorted({
                *existing,
                int(window.segment_idx),
            })
            return True

        self._apply_annotation_mutation(mutate, anchor_ns=window.center_ns)
        self.pending_add_label = None
        self.pending_bad_region_start_ns = None
        self.edit_mode = False
        self.last_edit_message = f"Marked {window.title} bad"
        current_idx = self.window_idx
        self._save_working_annotations()
        self.points = self._build_points()
        self.windows = self._build_windows()
        self.jump_targets = self._build_jump_targets()
        if not self.windows:
            print("All visible annotation windows are hidden or marked bad.")
            plt.close(self.fig)
            return
        self.window_idx = min(current_idx, len(self.windows) - 1)
        window = self.windows[self.window_idx]
        self.view_center_ns = window.center_ns
        self.view_width_ns = max(DISPLAY_MS, window.end_ns - window.start_ns)
        self._load_window()
        self._draw()

    def _mark_negative_current_peaks_artifacts(self) -> None:
        if not self.windows:
            return
        window = self.windows[self.window_idx]
        _, peak_ts = _load_processed_peaks(
            self.processed,
            window.start_ns,
            window.end_ns,
        )
        if len(peak_ts) == 0:
            self.last_edit_message = "No current pipeline peaks in this window"
            self._draw()
            return

        clean_mask = self._clean_peak_mask(window.points, window, peak_ts)
        clean_ts = peak_ts[clean_mask]
        if len(clean_ts) == 0:
            self.last_edit_message = "No unannotated current peaks to scan"
            self._draw()
            return

        ecg_ts, ecg_vals = _load_raw_ecg(
            self.csv_index,
            window.start_ns,
            window.end_ns,
        )
        if len(ecg_ts) == 0:
            self.last_edit_message = "No raw ECG available for negative peak scan"
            self._draw()
            return

        _, peak_y = _nearest_points_to_ecg(clean_ts, ecg_ts, ecg_vals)
        negative_peak_ids = sorted({
            int(ts_ns)
            for ts_ns, y in zip(clean_ts, peak_y)
            if float(y) < 0
        })
        existing = {int(v) for v in self.annotation_data.get("artifacts", [])}
        new_peak_ids = [pid for pid in negative_peak_ids if pid not in existing]
        if not new_peak_ids:
            self.last_edit_message = "No negative clean R-peaks found in this window"
            self._draw()
            return

        def mutate() -> bool:
            self.annotation_data.setdefault("artifacts", []).extend(new_peak_ids)
            self.annotation_data["artifacts"] = sorted({
                int(v) for v in self.annotation_data.get("artifacts", [])
            })
            return True

        self._apply_annotation_mutation(mutate, anchor_ns=self.view_center_ns)
        self.pending_add_label = None
        self.pending_bad_region_start_ns = None
        self.last_edit_message = (
            f"Marked {len(new_peak_ids)} negative R-peak(s) as artifact"
        )
        self._save_working_annotations()
        self._refresh_annotations(self.view_center_ns)
        self._draw()

    def _jump_next_abnormal_rr(self) -> None:
        if not self.windows:
            return
        window = self.windows[self.window_idx]
        _, peak_ts = _load_processed_peaks(
            self.processed,
            window.start_ns,
            window.end_ns,
        )
        beat_ts = self._effective_rr_timestamps(
            window.points,
            window,
            peak_ts=peak_ts,
        )
        targets = self._abnormal_rr_targets(beat_ts)
        if not targets:
            self.last_edit_message = "No abnormal RR intervals in this window"
            self._draw()
            return

        target_times = [target[0] for target in targets]
        previous = self.jump_cursors.get("rr_abnormal")
        if previous is not None and window.start_ns <= previous <= window.end_ns:
            reference_ns = previous
        else:
            reference_ns = self.view_center_ns
        pos = bisect_right(target_times, reference_ns)
        if pos >= len(targets):
            pos = 0

        target_ns, rr_ms, kind = targets[pos]
        self.jump_cursors["rr_abnormal"] = target_ns
        self.view_center_ns = target_ns
        self._load_window()
        self.last_edit_message = (
            f"Jumped to {kind} RR interval ({rr_ms}ms) near "
            f"{_ns_to_local_label(target_ns)}"
        )
        self._draw()

    def _remove_point_annotation(self, point: MasterPoint) -> bool:
        if point.label == "artifact" and point.peak_id is not None:
            values = self.annotation_data.setdefault("artifacts", [])
            self.annotation_data["artifacts"] = [
                int(v) for v in values if int(v) != int(point.peak_id)
            ]
            return True
        if point.label == "interpolated" and point.peak_id is not None:
            peak_id = int(point.peak_id)
            values = self.annotation_data.setdefault("interpolate_peaks", [])
            self.annotation_data["interpolate_peaks"] = [
                int(v) for v in values if int(v) != peak_id
            ]
            self.annotation_data["interpolated_replacements"] = [
                row for row in self.annotation_data.get("interpolated_replacements", [])
                if int(row.get("peak_id", -1)) != peak_id
            ]
            modes = self.annotation_data.get("interpolate_modes")
            if isinstance(modes, dict):
                modes.pop(str(peak_id), None)
            return True
        if point.label == "missed_original":
            target = str(point.raw_value) if point.raw_value is not None else None
            if target is not None:
                self.annotation_data["added_r_peaks"] = [
                    ts for ts in self.annotation_data.get("added_r_peaks", [])
                    if str(ts) != target
                ]
            else:
                self.annotation_data["added_r_peaks"] = [
                    ts for ts in self.annotation_data.get("added_r_peaks", [])
                    if abs(_parse_annotation_iso_ns(ts) - point.timestamp_ns)
                    > ADDED_R_PEAK_DEDUP_MS
                ]
            return True
        if point.label == "phys_event" and point.peak_id is not None:
            values = self.annotation_data.setdefault("physiological_events", [])
            self.annotation_data["physiological_events"] = [
                int(v) for v in values if int(v) != int(point.peak_id)
            ]
            return True
        if point.label == "to_interpolate":
            raw = point.raw_value if isinstance(point.raw_value, dict) else {}
            target_peak_id = raw.get("peak_id")
            target_original = raw.get("original_timestamp")
            kept = []
            removed = False
            for row in self.annotation_data.get(TO_INTERPOLATE_KEY, []):
                if not isinstance(row, dict):
                    kept.append(row)
                    continue
                same_peak = (
                    target_peak_id is not None
                    and row.get("peak_id") is not None
                    and int(row["peak_id"]) == int(target_peak_id)
                )
                same_original = (
                    target_peak_id is None
                    and target_original is not None
                    and str(row.get("original_timestamp")) == str(target_original)
                )
                if same_peak or same_original:
                    removed = True
                    continue
                kept.append(row)
            self.annotation_data[TO_INTERPOLATE_KEY] = kept
            if removed:
                self._recalculate_to_interpolate_gap_for_target(
                    point.timestamp_ns
                )
            return removed
        return False

    def _add_artifact_peak(self, timestamp_ms: int) -> bool:
        peak_id = int(timestamp_ms)
        artifacts = {int(v) for v in self.annotation_data.get("artifacts", [])}
        if peak_id in artifacts:
            return False
        self.annotation_data.setdefault("artifacts", []).append(peak_id)
        self.annotation_data["artifacts"] = sorted({
            int(v) for v in self.annotation_data.get("artifacts", [])
        })
        return True

    def _excluded_current_peak_ns(
        self,
        points: list[MasterPoint] | None = None,
        peak_ts: np.ndarray | None = None,
    ) -> set[int]:
        if peak_ts is None:
            peak_ts = self.current_peak_ts
        if points is None:
            points = [
                point for point in self._build_points()
                if point.label in {
                    "artifact", "interpolated", "phys_event", "to_interpolate"
                }
            ]

        excluded = set()
        for point in points:
            if point.label not in {
                "artifact", "interpolated", "phys_event", "to_interpolate"
            }:
                continue
            matched_ns = _nearest_peak_within(
                peak_ts,
                point.timestamp_ns,
                ARTIFACT_PEAK_MATCH_MS,
            )
            if matched_ns is not None:
                excluded.add(int(matched_ns))
        return excluded

    @staticmethod
    def _non_excluded_peak_near(
        peak_ts: np.ndarray,
        ts_ns: int,
        excluded_ns: set[int],
        tolerance_ns: int,
    ) -> int | None:
        matched_ns = _nearest_peak_within(peak_ts, ts_ns, tolerance_ns)
        if matched_ns is None:
            return None
        if int(matched_ns) in excluded_ns:
            return None
        return int(matched_ns)

    def _add_missed_original(self, ts_ns: int) -> bool:
        existing = [
            _parse_annotation_iso_ns(ts)
            for ts in self.annotation_data.get("added_r_peaks", [])
        ]
        if any(abs(ts - ts_ns) <= ADDED_R_PEAK_DEDUP_MS for ts in existing):
            return False
        self.annotation_data.setdefault("added_r_peaks", []).append(
            _format_annotation_iso_ns(ts_ns)
        )
        self.annotation_data["added_r_peaks"] = sorted(
            self.annotation_data["added_r_peaks"],
            key=_parse_annotation_iso_ns,
        )
        return True

    def _add_phys_event(self, ts_ms: int) -> bool:
        event_ms = int(ts_ms)
        events = {int(v) for v in self.annotation_data.get("physiological_events", [])}
        if event_ms in events:
            return False
        self.annotation_data.setdefault("physiological_events", []).append(event_ms)
        self.annotation_data["physiological_events"] = sorted({
            int(v) for v in self.annotation_data.get("physiological_events", [])
        })
        return True

    def _interpolation_target_ns_for_point(
        self,
        point: MasterPoint,
        peak_ts: np.ndarray,
    ) -> int | None:
        if point.label not in {"artifact", "interpolated", "to_interpolate"}:
            return None
        if point.peak_id is not None:
            matched = _nearest_peak_within(
                peak_ts,
                point.timestamp_ns,
                ARTIFACT_PEAK_MATCH_MS,
            )
            return int(matched) if matched is not None else int(point.timestamp_ns)
        if point.label == "to_interpolate":
            return int(point.timestamp_ns)
        return None

    def _to_interpolate_row_target_ns(
        self,
        row: dict,
        peak_ts: np.ndarray,
    ) -> int | None:
        try:
            if row.get("peak_id") is not None:
                peak_id_ms = int(row["peak_id"])
                matched = _nearest_peak_within(
                    peak_ts,
                    peak_id_ms,
                    ARTIFACT_PEAK_MATCH_MS,
                )
                return int(matched) if matched is not None else int(peak_id_ms)
            return _parse_annotation_iso_ns(str(row["original_timestamp"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _interpolation_anchor_ts(
        self,
        points: list[MasterPoint],
        peak_ts: np.ndarray,
        extra_artifact_targets: list[int] | None = None,
    ) -> list[int]:
        artifact_targets = [
            ns
            for point in points
            if (ns := self._interpolation_target_ns_for_point(point, peak_ts))
            is not None
        ]
        if extra_artifact_targets:
            artifact_targets.extend(int(ns) for ns in extra_artifact_targets)

        excluded_called_peaks = {
            int(ns)
            for ns in artifact_targets
            if _has_peak_within(peak_ts, ns, ARTIFACT_PEAK_MATCH_MS)
        }
        excluded_arr = np.asarray(sorted(excluded_called_peaks), dtype=np.int64)
        valid_ts = [
            int(ts)
            for ts in peak_ts
            if _nearest_peak_within(
                excluded_arr,
                int(ts),
                ARTIFACT_PEAK_MATCH_MS,
            ) is None
        ]
        valid_ts.extend(
            int(point.timestamp_ns)
            for point in points
            if point.label in {"missed_original", "phys_event"}
        )
        return sorted(set(valid_ts))

    @staticmethod
    def _neighbor_gap_for_target(
        target_ns: int,
        valid_ts: list[int],
    ) -> tuple[int, int] | None:
        valid_ts = sorted(set(int(v) for v in valid_ts))
        if len(valid_ts) < 2:
            return None
        pos = int(np.searchsorted(valid_ts, target_ns))
        if pos <= 0 or pos >= len(valid_ts):
            return None
        prev_valid = int(valid_ts[pos - 1])
        next_valid = int(valid_ts[pos])
        if next_valid <= prev_valid:
            return None
        return prev_valid, next_valid

    def _interpolation_replacements_for_gap(
        self,
        target_ns: int,
        valid_ts: list[int],
        row_targets: list[tuple[int, dict | None]],
    ) -> dict[int | None, int] | None:
        gap = self._neighbor_gap_for_target(target_ns, valid_ts)
        if gap is None:
            return None
        prev_valid, next_valid = gap
        group = [
            (int(ts), row)
            for ts, row in row_targets
            if prev_valid < int(ts) < next_valid
        ]
        if not group:
            return None
        group.sort(key=lambda item: item[0])
        gap_ns = next_valid - prev_valid
        replacements: dict[int | None, int] = {}
        for i, (_, row) in enumerate(group, start=1):
            replacement_ns = prev_valid + round(gap_ns * i / (len(group) + 1))
            replacements[id(row) if row is not None else None] = int(replacement_ns)
        return replacements

    def _interpolation_replacement_ns(
        self,
        target_ns: int,
        exclude_peak_ns: int | None = None,
        exclude_added_raw: str | None = None,
    ) -> int | None:
        window = self.windows[self.window_idx]
        _, peak_ts = _load_processed_peaks(
            self.processed,
            window.start_ns,
            window.end_ns,
        )

        points = window.points
        if exclude_added_raw is not None:
            points = [
                point for point in points
                if not (
                    point.label == "missed_original"
                    and str(point.raw_value) == str(exclude_added_raw)
                )
            ]

        extra_targets = []
        if exclude_peak_ns is not None:
            extra_targets.append(int(exclude_peak_ns))
        if exclude_added_raw is not None:
            extra_targets.append(int(target_ns))
        target_for_interpolation = int(target_ns)
        if extra_targets:
            nearest_target = min(
                extra_targets,
                key=lambda ts: abs(int(ts) - int(target_ns)),
            )
            if abs(int(nearest_target) - int(target_ns)) <= ARTIFACT_PEAK_MATCH_MS:
                target_for_interpolation = int(nearest_target)

        valid_ts = self._interpolation_anchor_ts(
            points,
            peak_ts,
            extra_artifact_targets=extra_targets,
        )
        row_targets = []
        for row in self.annotation_data.get(TO_INTERPOLATE_KEY, []):
            if not isinstance(row, dict):
                continue
            row_target = self._to_interpolate_row_target_ns(row, peak_ts)
            if row_target is not None:
                row_targets.append((row_target, row))
        row_targets.append((target_for_interpolation, None))
        replacements = self._interpolation_replacements_for_gap(
            target_for_interpolation,
            valid_ts,
            row_targets,
        )
        if replacements is None:
            return None
        return replacements.get(None)

    def _recalculate_to_interpolate_gap_replacements(
        self,
        target_ns: int,
        target_row: dict,
    ) -> int | None:
        window = self.windows[self.window_idx]
        _, peak_ts = _load_processed_peaks(
            self.processed,
            window.start_ns,
            window.end_ns,
        )
        points = [
            point for point in self._build_points()
            if window.start_ns <= point.timestamp_ns <= window.end_ns
        ]
        valid_ts = self._interpolation_anchor_ts(points, peak_ts)

        row_targets = []
        for row in self.annotation_data.get(TO_INTERPOLATE_KEY, []):
            if not isinstance(row, dict):
                continue
            row_target = self._to_interpolate_row_target_ns(row, peak_ts)
            if row_target is not None:
                row_targets.append((row_target, row))

        target_for_interpolation = self._to_interpolate_row_target_ns(
            target_row,
            peak_ts,
        )
        if target_for_interpolation is None:
            target_for_interpolation = int(target_ns)
        replacements = self._interpolation_replacements_for_gap(
            target_for_interpolation,
            valid_ts,
            row_targets,
        )
        if replacements is None:
            return None

        target_replacement = None
        for row in self.annotation_data.get(TO_INTERPOLATE_KEY, []):
            if not isinstance(row, dict):
                continue
            replacement_ns = replacements.get(id(row))
            if replacement_ns is None:
                continue
            row["replacement_timestamp"] = _format_annotation_iso_ns(replacement_ns)
            row["interpolation_method"] = "neighbor_even_spacing_ignore_artifacts"
            if row is target_row:
                target_replacement = replacement_ns
        return target_replacement

    def _recalculate_to_interpolate_gap_for_target(self, target_ns: int) -> bool:
        window = self.windows[self.window_idx]
        _, peak_ts = _load_processed_peaks(
            self.processed,
            window.start_ns,
            window.end_ns,
        )
        points = [
            point for point in self._build_points()
            if window.start_ns <= point.timestamp_ns <= window.end_ns
        ]
        valid_ts = self._interpolation_anchor_ts(points, peak_ts)

        row_targets = []
        for row in self.annotation_data.get(TO_INTERPOLATE_KEY, []):
            if not isinstance(row, dict):
                continue
            row_target = self._to_interpolate_row_target_ns(row, peak_ts)
            if row_target is not None:
                row_targets.append((row_target, row))

        replacements = self._interpolation_replacements_for_gap(
            int(target_ns),
            valid_ts,
            row_targets,
        )
        if replacements is None:
            return False

        changed = False
        for row in self.annotation_data.get(TO_INTERPOLATE_KEY, []):
            if not isinstance(row, dict):
                continue
            replacement_ns = replacements.get(id(row))
            if replacement_ns is None:
                continue
            row["replacement_timestamp"] = _format_annotation_iso_ns(replacement_ns)
            row["interpolation_method"] = "neighbor_even_spacing_ignore_artifacts"
            changed = True
        return changed

    def _add_to_interpolate_target(
        self,
        target_ns: int,
        source: str,
        peak_id: int | None = None,
        added_raw: str | None = None,
    ) -> int | None:
        rows = self.annotation_data.setdefault(TO_INTERPOLATE_KEY, [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            if peak_id is not None and row.get("peak_id") is not None:
                if int(row["peak_id"]) == int(peak_id):
                    return None
            if added_raw is not None:
                if (
                    str(row.get("original_timestamp")) == str(added_raw)
                    or str(row.get("removed_added_r_peak")) == str(added_raw)
                ):
                    return None

        before_artifacts = copy.deepcopy(self.annotation_data.get("artifacts", []))
        before_added = copy.deepcopy(self.annotation_data.get("added_r_peaks", []))
        before_rows = copy.deepcopy(rows)

        if peak_id is not None:
            self.annotation_data.setdefault("artifacts", []).append(int(peak_id))
            self.annotation_data["artifacts"] = sorted({
                int(v) for v in self.annotation_data.get("artifacts", [])
            })
        if added_raw is not None:
            self.annotation_data["added_r_peaks"] = [
                ts for ts in self.annotation_data.get("added_r_peaks", [])
                if str(ts) != str(added_raw)
            ]

        row = {
            "source": source,
            "original_timestamp": _format_annotation_iso_ns(target_ns),
            "replacement_timestamp": _format_annotation_iso_ns(target_ns),
            "interpolation_method": "neighbor_even_spacing_ignore_artifacts",
            "created_at": datetime.now().isoformat(),
        }
        if peak_id is not None:
            row["peak_id"] = int(peak_id)
        if added_raw is not None:
            row["removed_added_r_peak"] = str(added_raw)
        seg_idx = self._segment_for_ts(target_ns)
        if seg_idx is not None:
            row["segment_idx"] = int(seg_idx)

        rows.append(row)
        replacement_ns = self._recalculate_to_interpolate_gap_replacements(
            int(target_ns),
            row,
        )
        if replacement_ns is None:
            self.annotation_data["artifacts"] = before_artifacts
            self.annotation_data["added_r_peaks"] = before_added
            self.annotation_data[TO_INTERPOLATE_KEY] = before_rows
            return None

        self.annotation_data[TO_INTERPOLATE_KEY] = sorted(
            rows,
            key=lambda item: (
                self._to_interpolate_replacement_ns(item)
                if isinstance(item, dict)
                and self._to_interpolate_replacement_ns(item) is not None
                else self._to_interpolate_original_ns(item)
                if isinstance(item, dict)
                and self._to_interpolate_original_ns(item) is not None
                else 0
            ),
        )
        return replacement_ns

    def _nearest_waveform_timestamp(self, clicked_ms: int, lo_ms: int, event) -> int:
        if len(self.ecg_ts) == 0:
            return clicked_ms
        if event.x is not None and event.y is not None:
            x_sec = (self.ecg_ts - lo_ms) / 1000
            coords = self.ax.transData.transform(np.column_stack([x_sec, self.ecg_vals]))
            dists = np.hypot(coords[:, 0] - event.x, coords[:, 1] - event.y)
            return int(self.ecg_ts[int(np.argmin(dists))])
        pos = int(np.searchsorted(self.ecg_ts, clicked_ns))
        if pos <= 0:
            return int(self.ecg_ts[0])
        if pos >= len(self.ecg_ts):
            return int(self.ecg_ts[-1])
        before = int(self.ecg_ts[pos - 1])
        after = int(self.ecg_ts[pos])
        return before if abs(before - clicked_ns) <= abs(after - clicked_ns) else after

    def _nearest_waveform_points(
        self,
        point_ns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _nearest_points_to_ecg(point_ns, self.ecg_ts, self.ecg_vals)

    def _nearest_hit_index(
        self,
        ts_ns: np.ndarray,
        y_vals: np.ndarray,
        lo_ms: int,
        event,
        max_pixels: float = EDIT_HIT_TEST_PIXELS,
    ) -> int | None:
        if len(ts_ns) == 0 or event.x is None or event.y is None:
            return None
        x_sec = (ts_ns - lo_ms) / 1000
        coords = self.ax.transData.transform(np.column_stack([x_sec, y_vals]))
        dists = np.hypot(coords[:, 0] - event.x, coords[:, 1] - event.y)
        idx = int(np.argmin(dists))
        if float(dists[idx]) <= max_pixels:
            return idx
        return None

    def _clicked_annotation(
        self,
        points: list[MasterPoint],
        lo_ns: int,
        event,
    ) -> MasterPoint | None:
        if not points:
            return None
        sn_ns = np.empty(len(points), dtype=np.int64)
        sn_y = np.empty(len(points), dtype=np.float32)
        for label in [
            "artifact", "interpolated", "to_interpolate",
            "missed_original", "phys_event",
        ]:
            indices = [i for i, p in enumerate(points) if p.label == label]
            if not indices:
                continue
            ts_arr = np.asarray([points[i].timestamp_ns for i in indices],
                                dtype=np.int64)
            label_ns, label_y = self._nearest_waveform_points(ts_arr)
            for out_i, point_i in enumerate(indices):
                sn_ns[point_i] = label_ns[out_i]
                sn_y[point_i] = label_y[out_i]
        idx = self._nearest_hit_index(sn_ns, sn_y, lo_ns, event)
        return points[idx] if idx is not None else None

    def _clicked_clean_peak_ts(
        self,
        points: list[MasterPoint],
        lo_ns: int,
        event,
    ) -> int | None:
        window = self.windows[self.window_idx]
        clean_mask = self._clean_peak_mask(window.points, window)
        clean_ts = self.peak_ts[clean_mask]
        if len(clean_ts) == 0:
            return None
        sn_ns, sn_y = self._nearest_waveform_points(clean_ts)
        idx = self._nearest_hit_index(
            sn_ns, sn_y, lo_ns, event, max_pixels=CLEAN_HIT_TEST_PIXELS
        )
        return int(clean_ts[idx]) if idx is not None else None

    def _apply_interpolation_click(
        self,
        points: list[MasterPoint],
        lo_ns: int,
        event,
    ) -> None:
        clicked_point = self._clicked_annotation(points, lo_ns, event)
        target_ns = None
        peak_id = None
        added_raw = None
        source = None
        exclude_peak_ns = None
        exclude_added_raw = None

        if clicked_point is not None:
            if clicked_point.label == "to_interpolate":
                self.last_edit_message = "Already marked for interpolation"
                self._draw()
                return
            if clicked_point.label == "artifact" and clicked_point.peak_id is not None:
                target_ns = int(clicked_point.timestamp_ns)
                peak_id = int(clicked_point.peak_id)
                source = "artifact"
            elif clicked_point.label == "missed_original":
                target_ns = int(clicked_point.timestamp_ns)
                added_raw = str(clicked_point.raw_value)
                exclude_added_raw = added_raw
                source = "missed_original"

        if target_ns is None:
            clean_ts_ns = self._clicked_clean_peak_ts(points, lo_ns, event)
            if clean_ts_ns is not None:
                target_ns = int(clean_ts_ns)
                peak_id = int(clean_ts_ns)
                exclude_peak_ns = int(clean_ts_ns)
                source = "clean_peak"

        if target_ns is None or source is None:
            target_ns = lo_ns + int(event.xdata * 1000)
            source = "manual_interpolation"

        replacement_ns = self._interpolation_replacement_ns(
            target_ns,
            exclude_peak_ns=exclude_peak_ns,
            exclude_added_raw=exclude_added_raw,
        )
        if replacement_ns is None:
            self.last_edit_message = (
                "Could not interpolate; need surrounding non-excluded beats"
            )
            self._draw()
            return

        result = {"replacement_ns": replacement_ns}

        def mutate() -> bool:
            updated_ns = self._add_to_interpolate_target(
                target_ns,
                source,
                peak_id=peak_id,
                added_raw=added_raw,
            )
            if updated_ns is None:
                return False
            result["replacement_ns"] = updated_ns
            return True

        if self._apply_annotation_mutation(
            mutate,
            anchor_ns=target_ns,
        ):
            self.last_edit_message = (
                "Marked for interpolation; recalculated gap preview at "
                f"{_ns_to_local_label(result['replacement_ns'])}"
            )
            self._save_working_annotations()
            self._refresh_annotations(target_ns)
            self._draw()
            return

        self.last_edit_message = "Already marked for interpolation"
        self._draw()

    def _apply_missed_original_click(
        self,
        clicked_ns: int,
        lo_ns: int,
        points: list[MasterPoint],
        event,
    ) -> None:
        added_ns = self._nearest_waveform_timestamp(clicked_ns, lo_ns, event)
        blocking_peak_ms = self._non_excluded_peak_near(
            self.peak_ts,
            added_ns,
            self._excluded_current_peak_ns(points, self.peak_ts),
            MISSED_PEAK_MATCH_MS,
        )
        if blocking_peak_ms is not None:
            self.last_edit_message = (
                "Skipped missed_original; current peak already called near "
                f"{_ns_to_local_label(blocking_peak_ms)}"
            )
            self._draw()
            return
        if self._apply_annotation_mutation(
            lambda: self._add_missed_original(added_ns),
            anchor_ns=added_ns,
        ):
            self.last_edit_message = (
                f"Added missed_original near {_ns_to_local_label(added_ns)}"
            )
            self._save_working_annotations()
            self._refresh_annotations(added_ns)
            self._draw()
            return
        self.last_edit_message = "Skipped duplicate missed_original click"
        self._draw()

    def _apply_bad_region_click(self, clicked_ns: int) -> None:
        if not self.windows:
            return
        window = self.windows[self.window_idx]
        if window.segment_idx is None:
            self.last_edit_message = "Bad regions require a real segment"
            self._draw()
            return
        clicked_ns = max(window.start_ns, min(window.end_ns, int(clicked_ns)))
        if self.pending_bad_region_start_ns is None:
            self.pending_bad_region_start_ns = clicked_ns
            self.last_edit_message = (
                f"Bad region start set at {_ns_to_local_label(clicked_ns)}; "
                "click end point"
            )
            self._draw()
            return

        start_ns = min(self.pending_bad_region_start_ns, clicked_ns)
        end_ns = max(self.pending_bad_region_start_ns, clicked_ns)
        if end_ns <= start_ns:
            self.last_edit_message = "Bad region needs two distinct times"
            self._draw()
            return

        def mutate() -> bool:
            rows = self.annotation_data.setdefault(
                BAD_REGIONS_WITHIN_SEGMENTS_KEY,
                [],
            )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if (
                    int(row.get("segment_idx", -1)) == int(window.segment_idx)
                    and str(row.get("start_time")) == _format_annotation_iso_ns(start_ns)
                    and str(row.get("end_time")) == _format_annotation_iso_ns(end_ns)
                ):
                    return False
            rows.append({
                "segment_idx": int(window.segment_idx),
                "start_time": _format_annotation_iso_ns(start_ns),
                "end_time": _format_annotation_iso_ns(end_ns),
                "marked_at": datetime.now().isoformat(),
            })
            self.annotation_data[BAD_REGIONS_WITHIN_SEGMENTS_KEY] = sorted(
                rows,
                key=lambda row: (
                    int(row.get("segment_idx", -1)),
                    str(row.get("start_time", "")),
                    str(row.get("end_time", "")),
                ),
            )
            return True

        if self._apply_annotation_mutation(mutate, anchor_ns=start_ns):
            self.pending_bad_region_start_ns = None
            self.last_edit_message = (
                f"Marked bad region {_ns_to_local_label(start_ns)} - "
                f"{_ns_to_local_label(end_ns)}"
            )
            self._save_working_annotations()
            self._refresh_annotations((start_ns + end_ns) // 2)
            self._draw()
            return
        self.last_edit_message = "Skipped duplicate bad region"
        self._draw()

    def _apply_edit_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        half = self.view_width_ns // 2
        lo_ns = self.view_center_ns - half
        hi_ns = self.view_center_ns + half
        clicked_ns = lo_ns + int(event.xdata * 1000)
        points = self._current_points(lo_ns, hi_ns)

        if self.pending_add_label == "bad_region":
            self._apply_bad_region_click(clicked_ns)
            return

        if self.pending_add_label == "to_interpolate":
            self._apply_interpolation_click(points, lo_ns, event)
            return

        if self.pending_add_label == "phys_event":
            event_ns = self._nearest_waveform_timestamp(clicked_ns, lo_ns, event)
            if self._apply_annotation_mutation(
                lambda: self._add_phys_event(event_ns),
                anchor_ns=event_ns,
            ):
                self.last_edit_message = (
                    f"Added phys_event near {_ns_to_local_label(event_ns)}"
                )  # _ns_to_local_label works with ms values
                self._save_working_annotations()
                self._refresh_annotations(event_ns)
                self._draw()
                return
            self.last_edit_message = "Skipped duplicate phys_event click"
            self._draw()
            return

        if self.pending_add_label == "missed_original":
            self._apply_missed_original_click(clicked_ns, lo_ns, points, event)
            return

        clicked_point = self._clicked_annotation(points, lo_ns, event)
        if clicked_point is not None:
            if clicked_point.label in {
                "artifact", "interpolated", "to_interpolate",
                "missed_original", "phys_event"
            }:
                if self._apply_annotation_mutation(
                    lambda: self._remove_point_annotation(clicked_point),
                    anchor_ns=clicked_point.timestamp_ns,
                ):
                    self.last_edit_message = (
                        f"Removed {MASTER_LABEL_NAMES[clicked_point.label]} "
                        f"near {_ns_to_local_label(clicked_point.timestamp_ns)}"
                    )
                    self._save_working_annotations()
                    self._refresh_annotations(clicked_point.timestamp_ns)
                    self._draw()
                return
            self.last_edit_message = (
                f"{MASTER_LABEL_NAMES.get(clicked_point.label, clicked_point.label)} "
                "is not editable by click"
            )
            self._draw()
            return

        clean_ts_ns = self._clicked_clean_peak_ts(points, lo_ns, event)
        if clean_ts_ns is not None:
            if self._apply_annotation_mutation(
                lambda: self._add_artifact_peak(clean_ts_ns),
                anchor_ns=clean_ts_ns,
            ):
                self.last_edit_message = (
                    f"Marked clean beat as artifact near "
                    f"{_ns_to_local_label(clean_ts_ns)}"
                )
                self._save_working_annotations()
                self._refresh_annotations(clean_ts_ns)
                self._draw()
            return
        self._apply_missed_original_click(clicked_ns, lo_ns, points, event)

    def _draw_rr_labels(
        self,
        beat_ts: np.ndarray,
        lo_ns: int,
        hi_ns: int,
    ) -> None:
        if self.view_width_ns > DISPLAY_MS or len(beat_ts) < 2:
            return
        beat_ts = beat_ts[(beat_ts >= lo_ns) & (beat_ts <= hi_ns)]
        if len(beat_ts) < 2:
            return
        ylim = self.ax.get_ylim()
        y_rr = ylim[0] + 0.96 * (ylim[1] - ylim[0])
        rr_values = np.asarray([
            int(round(int(beat_ts[i + 1]) - int(beat_ts[i])))
            for i in range(len(beat_ts) - 1)
        ], dtype=np.int64)
        target_fontsize = 13.0
        min_fontsize = 6.5
        max_chars = max(len(str(int(v))) for v in rr_values)
        axis_px = max(1.0, float(self.ax.bbox.width))
        px_per_label = axis_px / max(1, len(rr_values))
        dpi_scale = float(self.fig.dpi) / 72.0
        estimated_target_px = target_fontsize * dpi_scale * (0.62 * max_chars) + 8
        if estimated_target_px <= px_per_label:
            rr_fontsize = target_fontsize
        else:
            rr_fontsize = max(
                min_fontsize,
                (px_per_label - 8) / max(1.0, dpi_scale * 0.62 * max_chars),
            )
        for i, rr_ms in enumerate(rr_values):
            neighbors = []
            if i > 0:
                neighbors.append(int(rr_values[i - 1]))
            if i + 1 < len(rr_values):
                neighbors.append(int(rr_values[i + 1]))
            kind, score = self._rr_deviation_gradient(int(rr_ms), neighbors)
            target_color = {
                "long": "#ff4d6d",
                "short": "#00d4ff",
            }.get(kind, "#ffd43b")
            color = self._mix_hex_colors("#ffd43b", target_color, score)
            weight = "bold" if score >= 0.55 else "normal"
            facecolor = self._mix_hex_colors("#0d1117", target_color, score * 0.35)
            alpha = 0.34 + (0.42 * score)
            mid_t = ((int(beat_ts[i]) + int(beat_ts[i + 1])) / 2 - lo_ns) / 1000
            self.ax.text(
                mid_t,
                y_rr,
                str(rr_ms),
                color=color,
                fontsize=rr_fontsize,
                ha="center",
                va="top",
                family="monospace",
                weight=weight,
                zorder=9,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": facecolor,
                    "edgecolor": "none",
                    "alpha": alpha,
                },
            )

    def _draw(self) -> None:
        window = self.windows[self.window_idx]
        half = self.view_width_ns // 2
        lo_ns = self.view_center_ns - half
        hi_ns = self.view_center_ns + half
        view_sec = self.view_width_ns / 1000

        self.ax.clear()
        self.ax.set_facecolor("#16213e")

        if len(self.ecg_ts):
            t_sec = (self.ecg_ts - lo_ns) / 1000
            self.ax.plot(t_sec, self.ecg_vals,
                         color="#4fc3f7", linewidth=0.55, alpha=0.92)
        else:
            self.ax.text(view_sec / 2, 0, "No raw ECG data in this window",
                         ha="center", va="center", color="#78909c", fontsize=14)

        ylim = self.ax.get_ylim()
        yspan = max(1.0, ylim[1] - ylim[0])

        seg_lo = max(0, (window.start_ns - lo_ns) / 1000)
        seg_hi = min(view_sec, (window.end_ns - lo_ns) / 1000)
        if window.bad:
            self.ax.axvspan(seg_lo, seg_hi, color="#ff4d6d", alpha=0.07, zorder=1)
        if window.return_to_pile:
            self.ax.axvspan(seg_lo, seg_hi, color="#b388ff", alpha=0.06, zorder=1)
        for start_ns, end_ns in window.bad_regions:
            x0 = max(0, (start_ns - lo_ns) / 1000)
            x1 = min(view_sec, (end_ns - lo_ns) / 1000)
            self.ax.axvspan(x0, x1, color="#ff4d6d", alpha=0.18, zorder=2)
        if (
            self.pending_add_label == "bad_region"
            and self.pending_bad_region_start_ns is not None
            and lo_ns <= self.pending_bad_region_start_ns <= hi_ns
        ):
            x0 = (self.pending_bad_region_start_ns - lo_ns) / 1000
            self.ax.axvline(
                x0,
                color="#ff4d6d",
                alpha=0.85,
                linewidth=1.8,
                linestyle="--",
                zorder=7,
            )

        points = self._current_points(lo_ns, hi_ns)

        if len(self.peak_ts):
            clean_mask = self._clean_peak_mask(window.points, window)
            clean_ts = self.peak_ts[clean_mask]
            if len(clean_ts):
                sn_ns, sn_y = self._nearest_waveform_points(clean_ts)
                self.ax.scatter((sn_ns - lo_ns) / 1000, sn_y,
                                color=MASTER_LABEL_COLORS["reviewed"],
                                s=72, marker=MASTER_LABEL_MARKERS["reviewed"],
                                alpha=0.72, zorder=4, linewidths=0.0)

        by_label: dict[str, list[MasterPoint]] = defaultdict(list)
        for point in points:
            by_label[point.label].append(point)
        for label in [
            "artifact", "interpolated", "to_interpolate",
            "missed_original", "phys_event",
        ]:
            label_points = by_label.get(label, [])
            if not label_points:
                continue
            ts_arr = np.asarray([p.timestamp_ns for p in label_points], dtype=np.int64)
            sn_ns, sn_y = self._nearest_waveform_points(ts_arr)
            marker = MASTER_LABEL_MARKERS[label]
            edge = "white" if marker not in ("x", "+") else MASTER_LABEL_COLORS[label]
            self.ax.scatter((sn_ns - lo_ns) / 1000, sn_y,
                            color=MASTER_LABEL_COLORS[label],
                            s=260 if label != "phys_event" else 360,
                            marker=marker, edgecolors=edge, linewidths=2.2,
                            zorder=8)

        preview_ns = []
        for point in points:
            if point.label != "to_interpolate" or not isinstance(point.raw_value, dict):
                continue
            replacement_ns = self._to_interpolate_replacement_ns(point.raw_value)
            if replacement_ns is not None and lo_ns <= replacement_ns <= hi_ns:
                preview_ns.append(replacement_ns)
        if preview_ns:
            preview_arr = np.asarray(preview_ns, dtype=np.int64)
            sn_ns, sn_y = self._nearest_waveform_points(preview_arr)
            self.ax.vlines(
                (preview_arr - lo_ns) / 1000,
                ylim[0],
                ylim[1],
                color=MASTER_LABEL_COLORS["interpolation_preview"],
                alpha=0.18,
                linestyles="dashed",
                linewidth=1.2,
                zorder=3,
            )
            self.ax.scatter(
                (sn_ns - lo_ns) / 1000,
                sn_y,
                facecolors="none",
                edgecolors=MASTER_LABEL_COLORS["interpolation_preview"],
                s=380,
                marker=MASTER_LABEL_MARKERS["interpolation_preview"],
                linewidths=2.4,
                zorder=9,
            )

        self._draw_rr_labels(
            self._effective_rr_timestamps(window.points, window),
            lo_ns,
            hi_ns,
        )

        self.ax.set_xlim(0, view_sec)
        self.ax.set_xlabel("seconds in window", color="#78909c", fontsize=8)
        tick_step = 10 if view_sec >= 60 else (5 if view_sec >= 30 else 2)
        tick_sec = np.arange(0, view_sec + 0.01, tick_step)
        tick_ns = [lo_ns + int(s * 1000) for s in tick_sec]
        self.ax.set_xticks(tick_sec)
        self.ax.set_xticklabels(
            [datetime.fromtimestamp(ns / 1000, tz=LOCAL_TZ).strftime("%H:%M:%S")
             for ns in tick_ns],
            color="#78909c", fontsize=7,
        )
        self.ax.tick_params(axis="y", colors="#546e7a", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#2a3a5c")

        status = []
        if window.reviewed:
            status.append("clean")
        if window.bad:
            status.append("bad segment")
        if window.return_to_pile:
            status.append("revisit")
        if window.bad_regions:
            status.append(f"{len(window.bad_regions)} partial bad region(s)")
        if self._is_review_completed(window):
            status.append("done")
        status_text = " | ".join(status) if status else "point annotations"
        mode_text = "browse"
        if self.pending_add_label == "bad_region":
            mode_text = (
                "BAD REGION END"
                if self.pending_bad_region_start_ns is not None
                else "BAD REGION START"
            )
        elif self.pending_add_label == "phys_event":
            mode_text = "PHYS EVENT MODE"
        elif self.pending_add_label == "missed_original":
            mode_text = "ADD PEAK MODE"
        elif self.pending_add_label == "to_interpolate":
            mode_text = "INTERPOLATE MODE"
        title = (
            f"{window.title}  {self.window_idx + 1}/{len(self.windows)}"
            f"  [{_ns_to_local_label(window.start_ns)} - {_ns_to_local_label(window.end_ns)}]"
            f"\n{status_text}  |  {mode_text}  |  {self.last_edit_message}"
            f"\n]/right next seg  [ /left prev seg  |  click empty adds peak  a add-mode  "
            f"p bad-region  f phys_event  i interp  o bad-seg  "
            f"n neg-artifact  g abnormal RR  r revisit  v done  z undo  x redo  |  "
            f"/ next interp  . next added  , next artifact  space next any  |  q quit"
        )
        self.fig.suptitle(title, color="#e0e0e0", fontsize=10, y=0.97)
        self._draw_legend(points)
        self.fig.canvas.draw()

    def _draw_legend(self, points: list[MasterPoint]) -> None:
        self.ax_leg.clear()
        self.ax_leg.set_facecolor("#0d1117")
        self.ax_leg.axis("off")
        y = 0.98
        self.ax_leg.text(0.5, y, "Master Labels", color="#e0e0e0", fontsize=8,
                         ha="center", va="top", weight="bold",
                         transform=self.ax_leg.transAxes)
        y -= 0.06
        window = self.windows[self.window_idx]
        counts = Counter(
            p.label for p in points
            if not self._annotation_invalid_by_bad_area(
                p.timestamp_ns,
                segment_idx=window.segment_idx,
            )
        )
        if len(self.peak_ts):
            clean_mask = self._clean_peak_mask(window.points, window)
            counts["reviewed"] = int(np.sum(clean_mask))
        counts["interpolation_preview"] = sum(
            1 for point in window.points
            if point.label == "to_interpolate"
            and isinstance(point.raw_value, dict)
            and (
                replacement_ns := self._to_interpolate_replacement_ns(point.raw_value)
            ) is not None
            and window.start_ns <= replacement_ns <= window.end_ns
            and not self._annotation_invalid_by_bad_area(
                point.timestamp_ns,
                segment_idx=window.segment_idx,
            )
        )
        for label in [
            "artifact", "interpolated", "to_interpolate",
            "interpolation_preview", "missed_original", "phys_event", "reviewed",
        ]:
            color = MASTER_LABEL_COLORS[label]
            marker = MASTER_LABEL_MARKERS[label]
            self.ax_leg.scatter([0.10], [y], color=color, s=90, marker=marker,
                                transform=self.ax_leg.transAxes)
            self.ax_leg.text(0.22, y, f"{MASTER_LABEL_NAMES[label]} ({counts[label]})",
                             color=color, fontsize=7, va="center",
                             transform=self.ax_leg.transAxes)
            y -= 0.045
        y -= 0.02
        self.ax_leg.text(0.5, y, "Navigation", color="#78909c", fontsize=7,
                         ha="center", va="top", transform=self.ax_leg.transAxes)
        y -= 0.045
        for line in [
            "]/right = next segment",
            "[/left = previous segment",
            "space = next any label",
            "/ = next interpolated",
            ". = next added beat",
            ", = next artifact",
            "click empty = add peak",
            "a = toggle add mode",
            "p = bad region start/end",
            "f = toggle phys_event",
            "i = toggle interpolate",
            "o = bad segment",
            "n = flag negative peaks",
            "g = next abnormal RR",
            "r = revisit pile",
            "v = mark window done",
            "z = undo",
            "x = redo",
            "click clean/artifact = toggle",
            "scroll = zoom",
            "q = quit",
        ]:
            self.ax_leg.text(0.04, y, line, color="#546e7a", fontsize=6.5,
                             va="top", family="monospace",
                             transform=self.ax_leg.transAxes)
            y -= 0.035
        if self.processed is None or pq is None:
            self.ax_leg.text(0.04, y - 0.02,
                             "clean markers need\nprocessed/peaks.parquet",
                             color="#ffb703", fontsize=6, va="top",
                             transform=self.ax_leg.transAxes)

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.step == 0:
            return
        scale = ZOOM_FACTOR ** (-event.step)
        new_width = int(self.view_width_ns * scale)
        new_width = max(MASTER_MIN_VIEW_MS, min(MAX_VIEW_MS, new_width))
        if new_width == self.view_width_ns:
            return
        if event.xdata is not None:
            view_sec = self.view_width_ns / 1000
            cursor_frac = max(0.0, min(1.0, event.xdata / view_sec))
            lo_ns = self.view_center_ns - self.view_width_ns // 2
            cursor_ns = lo_ns + int(cursor_frac * self.view_width_ns)
            new_lo_ns = cursor_ns - int(cursor_frac * new_width)
            self.view_center_ns = new_lo_ns + new_width // 2
        self.view_width_ns = new_width
        self._load_window()
        self._draw()

    def _on_key(self, event) -> None:
        k = event.key
        if k == " ":
            self._jump_next_any()
        elif k in ("]", "right"):
            self._go_to_window(self.window_idx + 1)
        elif k in ("[", "left"):
            self._go_to_window(self.window_idx - 1)
        elif k in JUMP_LABELS:
            self._jump_next(JUMP_LABELS[k])
        elif k == "p":
            if self.pending_add_label == "bad_region":
                self.pending_add_label = None
                self.pending_bad_region_start_ns = None
                self.last_edit_message = "Bad region mode disabled"
            else:
                self.pending_add_label = "bad_region"
                self.pending_bad_region_start_ns = None
                self.last_edit_message = "Bad region mode; click start point"
            self._draw()
        elif k == "f":
            if self.pending_add_label == "phys_event":
                self.pending_add_label = None
                self.pending_bad_region_start_ns = None
                self.last_edit_message = "Phys_event mode disabled"
            else:
                self.pending_add_label = "phys_event"
                self.pending_bad_region_start_ns = None
                self.last_edit_message = "Phys_event mode; click waveform to add"
            self._draw()
        elif k == "a":
            if self.pending_add_label == "missed_original":
                self.pending_add_label = None
                self.pending_bad_region_start_ns = None
                self.last_edit_message = "Add peak mode disabled"
            else:
                self.pending_add_label = "missed_original"
                self.pending_bad_region_start_ns = None
                self.last_edit_message = "Add peak mode; click waveform to add"
            self._draw()
        elif k == "i":
            if self.pending_add_label == "to_interpolate":
                self.pending_add_label = None
                self.pending_bad_region_start_ns = None
                self.last_edit_message = "Interpolation mode disabled"
            else:
                self.pending_add_label = "to_interpolate"
                self.pending_bad_region_start_ns = None
                self.last_edit_message = (
                    "Interpolation mode; click clean peak, artifact, or added peak"
                )
            self._draw()
        elif k == "r":
            self.pending_add_label = None
            self.pending_bad_region_start_ns = None
            self._mark_current_window_revisit()
        elif k == "v":
            self.pending_add_label = None
            self.pending_bad_region_start_ns = None
            self._mark_current_window_done()
        elif k == "o":
            self.pending_add_label = None
            self.pending_bad_region_start_ns = None
            self._mark_current_window_bad()
        elif k == "n":
            self.pending_add_label = None
            self.pending_bad_region_start_ns = None
            self._mark_negative_current_peaks_artifacts()
        elif k == "g":
            self.pending_add_label = None
            self.pending_bad_region_start_ns = None
            self._jump_next_abnormal_rr()
        elif k == "z":
            self.pending_add_label = None
            self.pending_bad_region_start_ns = None
            self._undo_last_edit()
        elif k == "x":
            self.pending_add_label = None
            self.pending_bad_region_start_ns = None
            self._redo_last_edit()
        elif k == "escape":
            if self.pending_add_label is not None:
                self.pending_add_label = None
                self.pending_bad_region_start_ns = None
                self.last_edit_message = "Canceled pending add"
            else:
                self.last_edit_message = "Nothing to cancel"
            self._draw()
        elif k == "q":
            plt.close(self.fig)

    def _on_click(self, event) -> None:
        self._apply_edit_click(event)

    def run(self) -> None:
        print(f"Loaded {len(self.windows)} annotated window(s) from {self.annotation_file}")
        if self.show_clean_only:
            print("Mode: clean-only windows (--clean)")
        else:
            print("Mode: review annotations only; clean-only windows hidden")
        if self.source_annotation_file is not None:
            print(f"Source preserved: {self.source_annotation_file}")
        print(f"Edits autosave to: {self.annotation_file}")
        print("Keys: ]/right next segment, [/left previous segment, "
              "space next any annotation, click empty waveform add peak, "
              "a add mode, p bad-region, f phys_event, i interpolate, o bad-seg, "
              "n negative peaks to artifacts, g abnormal RR, "
              "r revisit, v done, z undo, x redo, "
              "/ interpolated, . added, , artifact, q quit")
        plt.show()


def _load_or_init_themes() -> dict[str, str]:
    p = OUTPUT_DIR / "theme_labels.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(DEFAULT_THEME_LABELS, f, indent=2)
    return DEFAULT_THEME_LABELS.copy()


def _load_or_init_tags() -> dict[str, str]:
    """Load tag_labels.json (slot string → name). Returns empty dict if absent."""
    p = OUTPUT_DIR / "tag_labels.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _load_existing_annotations() -> tuple[list[dict], list[dict]]:
    segs, beats = [], []
    sp = OUTPUT_DIR / "segment_annotations.csv"
    bp = OUTPUT_DIR / "beat_annotations.csv"
    if sp.exists():
        with open(sp) as f:
            segs = list(csv.DictReader(f))
    if bp.exists():
        with open(bp) as f:
            beats = list(csv.DictReader(f))
    return segs, beats


# ── Annotator ─────────────────────────────────────────────────────────────────

class MarkerAnnotator:
    SEG_HEADER  = ["ann_id", "marker_idx", "marker_dt", "theme_id",
                   "view_start_ns", "view_end_ns", "annotated_at"]
    BEAT_HEADER = ["ann_id", "marker_idx", "marker_dt", "theme_id",
                   "peak_id", "peak_timestamp_ns", "annotated_at"]

    def __init__(self, processed: Path, start_idx: int = 0,
                 order: str = "desc", annotations_only: bool = False) -> None:
        self.processed         = processed
        self.markers           = _load_markers()
        if order == "desc":
            self.markers = list(reversed(self.markers))
        self.segments          = _load_segments(processed)
        self.themes            = _load_or_init_themes()
        self.tags              = _load_or_init_tags()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.seg_anns, self.beat_anns = _load_existing_annotations()
        self._next_ann_id = (
            max((int(a["ann_id"]) for a in self.seg_anns + self.beat_anns),
                default=0) + 1
        )

        # Annotations-only mode
        self.annotations_only  : bool            = annotations_only
        self.ann_windows       : list[int]       = []
        self.ann_window_idx    : int             = 0

        # Navigation state
        self.marker_idx        : int             = start_idx
        self.seg_idx           : int | None      = None
        self.view_center_ns    : int             = 0
        self.view_width_ns     : int             = DISPLAY_MS
        self.ecg_ts            : np.ndarray      = np.array([])
        self.ecg_vals          : np.ndarray      = np.array([])
        self.peaks_df          : pd.DataFrame    = pd.DataFrame()
        self.offset_sec        : int             = 0
        self.ecg_loaded        : bool            = False

        # Interaction state machine
        self.state             : str             = "BROWSE"
        self.selected_peaks    : list[int]       = []

        # Segment click-define state
        self._seg_start_ns     : int | None      = None
        self._seg_end_ns       : int | None      = None

        # Legacy theme rename state
        self._rename_theme_id  : int | None      = None
        self._rename_buffer    : str             = ""

        # New tag definition state
        self._new_tag_id_buffer   : str          = ""
        self._new_tag_name_buffer : str          = ""
        self._new_tag_pending_id  : int | None   = None

        # Tag assignment buffer (AWAIT_BEAT_THEME / AWAIT_SEG_THEME)
        self._tag_assign_buffer : str            = ""

        # Double-press detection
        self._last_key_time    : dict[str, float] = {}

        self._build_figure()

        if annotations_only:
            self.ann_windows = self._build_annotation_windows()
            if not self.ann_windows:
                print("No annotations found. Run without --annotations-only first.")
                sys.exit(0)
            print(f"Annotations-only mode: {len(self.ann_windows)} window(s) "
                  f"from {len(self.beat_anns)} beat + {len(self.seg_anns)} seg annotations.")
            self._go_to_annotation_window(0)
        else:
            self._go_to_marker(self.marker_idx)

    # ── Figure ────────────────────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(28, 7), facecolor="#0d1117")
        gs = gridspec.GridSpec(1, 2, figure=self.fig,
                               width_ratios=[8, 1.5],
                               left=0.02, right=0.98,
                               top=0.88, bottom=0.10,
                               wspace=0.02)
        self.ax     = self.fig.add_subplot(gs[0])
        self.ax_leg = self.fig.add_subplot(gs[1])
        self.ax.set_facecolor("#16213e")
        self.ax_leg.set_facecolor("#0d1117")
        self.ax_leg.axis("off")
        self.fig.canvas.mpl_connect("key_press_event",    self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event",        self._on_scroll)

    # ── Annotation color / label helpers ──────────────────────────────────────

    def _ann_color(self, theme_id: int) -> str:
        if theme_id > TAG_OFFSET:
            return TAG_COLORS.get(theme_id - TAG_OFFSET, "#aaaaaa")
        return THEME_COLORS.get(theme_id, "#ffffff")

    def _ann_short_label(self, theme_id: int) -> str:
        """Short label shown on the ECG plot (e.g. '#5' or 'L3')."""
        if theme_id > TAG_OFFSET:
            return f"#{theme_id - TAG_OFFSET}"
        return f"L{theme_id}"

    def _ann_full_label(self, theme_id: int) -> str:
        """Full label for logging / screenshot titles."""
        if theme_id > TAG_OFFSET:
            slot = theme_id - TAG_OFFSET
            name = self.tags.get(str(slot), f"Tag {slot}")
            return f"#{slot}: {name}"
        name = self.themes.get(str(theme_id), f"Theme {theme_id}")
        return f"L{theme_id}: {name}"

    # ── Navigation ────────────────────────────────────────────────────────────

    def _build_annotation_windows(self) -> list[int]:
        """Return sorted window-center timestamps (ns) for annotations-only mode.

        Clusters all annotation timestamps within _ANN_MERGE_MS of each other
        into a single window; center = midpoint of the cluster.
        """
        centers: list[int] = []
        for ann in self.beat_anns:
            centers.append(int(ann["peak_timestamp_ns"]))
        for ann in self.seg_anns:
            mid = (int(ann["view_start_ns"]) + int(ann["view_end_ns"])) // 2
            centers.append(mid)
        if not centers:
            return []
        centers.sort()
        clusters: list[list[int]] = [[centers[0]]]
        for c in centers[1:]:
            if c - clusters[-1][-1] <= _ANN_MERGE_MS:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return [(min(cl) + max(cl)) // 2 for cl in clusters]

    def _go_to_annotation_window(self, idx: int) -> None:
        if not self.ann_windows:
            return
        idx = max(0, min(idx, len(self.ann_windows) - 1))
        self.ann_window_idx  = idx
        center_ns            = self.ann_windows[idx]
        self.view_center_ns  = center_ns
        self.view_width_ns   = 60_000    # ±30 s flanks
        self.state           = "BROWSE"
        self.selected_peaks  = []
        steps = (SEARCH_STEPS_EARLY if center_ns < EARLY_CUTOFF_MS
                 else SEARCH_STEPS_RECENT)
        self.seg_idx, self.offset_sec = _find_segment(center_ns, self.segments, steps)
        self._load_window()
        self._draw()

    def _go_to_marker(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.markers):
            return
        self.marker_idx     = idx
        m                   = self.markers[idx]
        steps               = (SEARCH_STEPS_EARLY if m["ms"] < EARLY_CUTOFF_MS
                                else SEARCH_STEPS_RECENT)
        self.seg_idx, self.offset_sec = _find_segment(m["ns"], self.segments, steps)
        self.view_center_ns = m["ns"]
        self.state          = "BROWSE"
        self.selected_peaks = []
        self._load_window()
        self._draw()

    def _load_window(self) -> None:
        half = self.view_width_ns // 2
        lo   = self.view_center_ns - half
        hi   = self.view_center_ns + half

        if self.seg_idx is None:
            self.ecg_ts     = np.array([])
            self.ecg_vals   = np.array([])
            self.peaks_df   = pd.DataFrame(columns=["peak_id", "timestamp_ms"])
            self.ecg_loaded = False
            return

        overlapping = self.segments[
            (self.segments["end_timestamp_ms"] >= lo) &
            (self.segments["start_timestamp_ms"] <= hi)
        ]
        ts_parts, ecg_parts = [], []
        for seg in overlapping.itertuples():
            ts_s, ecg_s = _load_ecg(self.processed, seg.segment_idx, lo, hi)
            ts_parts.append(ts_s)
            ecg_parts.append(ecg_s)

        if ts_parts:
            self.ecg_ts   = np.concatenate(ts_parts)
            self.ecg_vals = np.concatenate(ecg_parts)
            order         = np.argsort(self.ecg_ts)
            self.ecg_ts   = self.ecg_ts[order]
            self.ecg_vals = self.ecg_vals[order]
            self.ecg_loaded = len(self.ecg_ts) > 20
        else:
            self.ecg_ts     = np.array([])
            self.ecg_vals   = np.array([])
            self.ecg_loaded = False

        self.peaks_df = _load_peaks(self.processed, lo, hi)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        half  = self.view_width_ns // 2
        lo_ns = self.view_center_ns - half
        hi_ns = self.view_center_ns + half

        self.ax.clear()
        self.ax.set_facecolor("#16213e")

        if self.ecg_loaded:
            t_sec = (self.ecg_ts - lo_ns) / 1000
            self.ax.plot(t_sec, self.ecg_vals,
                         color="#4fc3f7", linewidth=0.55, alpha=0.92)

            if len(self.peaks_df):
                pk_ns = self.peaks_df["timestamp_ms"].values.astype(np.int64)
                sn_ns, sn_y = _snap_peaks(pk_ns, self.ecg_ts, self.ecg_vals)
                self.ax.scatter((sn_ns - lo_ns) / 1000, sn_y,
                                color="#546e7a", s=18, zorder=4, alpha=0.7)

            if self.selected_peaks and len(self.peaks_df):
                sel = self.peaks_df[
                    self.peaks_df["peak_id"].isin(self.selected_peaks)
                ]
                if len(sel):
                    sel_ns = sel["timestamp_ms"].values.astype(np.int64)
                    sn_ns, sn_y = _snap_peaks(sel_ns, self.ecg_ts, self.ecg_vals)
                    self.ax.scatter((sn_ns - lo_ns) / 1000, sn_y,
                                    color="#ff6b6b", s=60, zorder=6,
                                    edgecolors="white", linewidths=0.5)
        else:
            self.ax.text(30, 0, "No ECG data in this window",
                         ha="center", va="center",
                         color="#546e7a", fontsize=14)

        view_sec = self.view_width_ns / 1000

        # Marker star reference line
        if self.markers:
            m          = self.markers[self.marker_idx]
            marker_off = (m["ms"] - lo_ns) / 1000
            if 0 <= marker_off <= view_sec:
                self.ax.axvline(marker_off, color="#ff6b6b", linewidth=1.4,
                                linestyle="--", alpha=0.8)
                self.ax.text(marker_off + view_sec * 0.005, self.ax.get_ylim()[1],
                             "★", color="#ff6b6b", fontsize=10, va="top", ha="left")

        # Segment-start anchor
        if self.state == "SEG_END" and self._seg_start_ns is not None:
            s_off = (self._seg_start_ns - lo_ns) / 1000
            if 0 <= s_off <= view_sec:
                self.ax.axvline(s_off, color="#69db7c", linewidth=2.2,
                                linestyle="-", alpha=0.9)
                self.ax.text(s_off + view_sec * 0.005, self.ax.get_ylim()[1],
                             "▶ start", color="#69db7c", fontsize=8,
                             va="top", ha="left")

        self._draw_existing_annotations(lo_ns, hi_ns,
                                        t_sec if self.ecg_loaded else None)

        # RR labels when zoomed in
        if (self.ecg_loaded and len(self.peaks_df) > 1
                and self.view_width_ns <= RR_SHOW_MS):
            ylim = self.ax.get_ylim()
            y_rr = ylim[0] + 0.97 * (ylim[1] - ylim[0])
            sorted_ts = np.sort(
                self.peaks_df["timestamp_ms"].values.astype(np.int64)
            )
            in_view = (sorted_ts >= lo_ns) & (sorted_ts <= hi_ns)
            view_ts = sorted_ts[in_view]
            for i in range(len(view_ts) - 1):
                rr_ms = int(round(view_ts[i + 1] - view_ts[i]))
                mid_t = ((view_ts[i] + view_ts[i + 1]) / 2 - lo_ns) / 1000
                self.ax.text(mid_t, y_rr, f"{rr_ms}",
                             color="#ffd43b", fontsize=7, ha="center", va="top",
                             family="monospace", zorder=9)

        self.ax.set_xlim(0, view_sec)
        self.ax.set_xlabel("seconds in window", color="#78909c", fontsize=8)
        tick_step = 10 if view_sec >= 60 else (5 if view_sec >= 30 else 2)
        tick_sec  = np.arange(0, view_sec + 0.01, tick_step)
        tick_ns   = [lo_ns + int(s * 1000) for s in tick_sec]
        self.ax.set_xticks(tick_sec)
        self.ax.set_xticklabels(
            [datetime.fromtimestamp(ns / 1000, tz=LOCAL_TZ).strftime("%H:%M:%S")
             for ns in tick_ns],
            color="#78909c", fontsize=7,
        )
        self.ax.tick_params(axis="y", colors="#546e7a", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#2a3a5c")

        # Title
        if self.annotations_only:
            dt_str = datetime.fromtimestamp(
                self.view_center_ns / 1000, tz=LOCAL_TZ
            ).strftime("%Y-%m-%d %H:%M:%S")
            title = (f"Ann Window {self.ann_window_idx + 1}/{len(self.ann_windows)}"
                     f"   [{dt_str}]   ±30 s context\n{self._state_hint()}")
        else:
            m          = self.markers[self.marker_idx]
            offset_str = (f"  ECG is {self.offset_sec}s from star"
                          if self.offset_sec > 0 else "  ECG overlaps star")
            title = (f"Marker {self.marker_idx + 1}/{len(self.markers)}"
                     f"   [{m['raw']}]{offset_str}\n{self._state_hint()}")
        self.fig.suptitle(title, color="#e0e0e0", fontsize=10, y=0.97)
        self._draw_legend()
        self.fig.canvas.draw()

    def _draw_existing_annotations(
        self, lo_ns: int, hi_ns: int, t_sec: np.ndarray | None
    ) -> None:
        ylim  = self.ax.get_ylim()
        yspan = ylim[1] - ylim[0]

        for ann in self.seg_anns:
            ann_lo = int(ann["view_start_ns"])
            ann_hi = int(ann["view_end_ns"])
            if ann_lo > hi_ns or ann_hi < lo_ns:
                continue
            x0  = max(0, (ann_lo - lo_ns) / 1000)
            x1  = min(self.view_width_ns / 1000, (ann_hi - lo_ns) / 1000)
            tid = int(ann["theme_id"])
            c   = self._ann_color(tid)
            lbl = self._ann_short_label(tid)
            self.ax.axvspan(x0, x1, alpha=0.08, color=c, zorder=2)
            self.ax.text((x0 + x1) / 2, ylim[1] - yspan * 0.04, lbl,
                         color=c, fontsize=7, ha="center", va="top", zorder=5)

        if t_sec is not None and len(self.peaks_df):
            for ann in self.beat_anns:
                pts_ns = int(ann["peak_timestamp_ns"])
                if not (lo_ns <= pts_ns <= hi_ns):
                    continue
                sn, sy = _snap_peaks(
                    np.array([pts_ns], dtype=np.int64),
                    self.ecg_ts, self.ecg_vals,
                )
                pt  = (sn[0] - lo_ns) / 1000
                py  = float(sy[0])
                tid = int(ann["theme_id"])
                c   = self._ann_color(tid)
                lbl = self._ann_short_label(tid)
                if self.state == "DELETE_SELECT":
                    self.ax.scatter([pt], [py], color="#ff4444", s=110, zorder=7,
                                    marker="x", edgecolors="#ff4444", linewidths=2.0)
                    self.ax.text(pt, py + yspan * 0.05, "✕",
                                 color="#ff4444", fontsize=6, ha="center", zorder=8)
                else:
                    self.ax.scatter([pt], [py], color=c, s=80, zorder=7,
                                    marker="D", edgecolors="white", linewidths=0.5)
                    self.ax.text(pt, py + yspan * 0.05, lbl,
                                 color=c, fontsize=6, ha="center", zorder=8)

    def _draw_legend(self) -> None:
        self.ax_leg.clear()
        self.ax_leg.set_facecolor("#0d1117")
        self.ax_leg.axis("off")

        y = 0.98

        # ── Tags section ──────────────────────────────────────────────────────
        self.ax_leg.text(0.5, y, "Tags", color="#e0e0e0", fontsize=8,
                         ha="center", va="top", weight="bold",
                         transform=self.ax_leg.transAxes)
        y -= 0.050

        defined = sorted(
            ((int(k), v) for k, v in self.tags.items() if v.strip()),
            key=lambda x: x[0],
        )
        if not defined:
            self.ax_leg.text(0.5, y, "(none — press N to define)",
                             color="#546e7a", fontsize=6, ha="center", va="top",
                             style="italic", transform=self.ax_leg.transAxes)
            y -= 0.038
        else:
            for slot, label in defined[:13]:   # show up to 13
                c = TAG_COLORS.get(slot, "#aaaaaa")
                self.ax_leg.scatter([0.08], [y], color=c, s=38, zorder=3,
                                    transform=self.ax_leg.transAxes)
                self.ax_leg.text(0.20, y, f"{slot}: {label}",
                                 color=c, fontsize=6.5, va="center",
                                 transform=self.ax_leg.transAxes)
                y -= 0.038
            if len(defined) > 13:
                self.ax_leg.text(0.5, y, f"… +{len(defined) - 13} more",
                                 color="#546e7a", fontsize=6, ha="center", va="center",
                                 transform=self.ax_leg.transAxes)
                y -= 0.034

        # ── Separator ─────────────────────────────────────────────────────────
        y -= 0.008
        self.ax_leg.text(0.5, y, "─" * 19, color="#2a3a5c", fontsize=6.5,
                         ha="center", va="center", transform=self.ax_leg.transAxes)
        y -= 0.030

        # ── Legacy section ────────────────────────────────────────────────────
        self.ax_leg.text(0.5, y, "Legacy (view only)", color="#78909c", fontsize=6.5,
                         ha="center", va="top", transform=self.ax_leg.transAxes)
        y -= 0.036
        for k in sorted(self.themes, key=int):
            c = THEME_COLORS.get(int(k), "#ffffff")
            self.ax_leg.scatter([0.08], [y], color=c, s=32, zorder=3, alpha=0.6,
                                transform=self.ax_leg.transAxes)
            self.ax_leg.text(0.20, y, f"{k}: {self.themes[k]}",
                             color=c, fontsize=6, va="center", alpha=0.6,
                             transform=self.ax_leg.transAxes)
            y -= 0.033

        # ── Separator ─────────────────────────────────────────────────────────
        y -= 0.008
        self.ax_leg.text(0.5, y, "─" * 19, color="#2a3a5c", fontsize=6.5,
                         ha="center", va="center", transform=self.ax_leg.transAxes)
        y -= 0.026

        # ── Mode + controls ───────────────────────────────────────────────────
        mode = "annotations-only" if self.annotations_only else "marker nav"
        self.ax_leg.text(0.5, y, f"[{mode}]", color="#ffd43b", fontsize=6,
                         ha="center", va="top", transform=self.ax_leg.transAxes)
        y -= 0.033

        controls = [
            "click=sel  B=confirm",
            "T=seg (click×2)",
            "# + ↩ = assign tag",
            "N = define tag",
            "X=del  2×X=clr segs",
            "2×⌫=clr window",
            "2×↩ = screenshots",
            "Esc=cancel/clear",
            "Spc / ] / [ = nav",
            "D=undo  Q=quit",
        ]
        for line in controls:
            self.ax_leg.text(0.04, y, line, color="#546e7a", fontsize=6,
                             va="top", transform=self.ax_leg.transAxes,
                             family="monospace")
            y -= 0.030

    def _state_hint(self) -> str:
        if self.state == "BROWSE":
            n   = len(self.selected_peaks)
            sel = f"  |  {n} selected — B: confirm" if n else ""
            return (f"← → pan  |  scroll↔↕  |  click peak = toggle{sel}"
                    f"  |  T: seg  |  X: del  |  N: new tag  |  Space: next")
        if self.state == "SEG_START":
            return "Click ECG to mark START of annotation region  |  Esc to cancel"
        if self.state == "SEG_END":
            return "Click ECG to mark END of annotation region  |  Esc to re-pick start"
        if self.state == "AWAIT_SEG_THEME":
            buf = self._tag_assign_buffer
            return (f"Type tag slot 1–20 + ↩ to assign  |  "
                    f"Buffer: '{buf}█'  |  ⌫ clear  |  Esc cancel")
        if self.state == "AWAIT_BEAT_THEME":
            buf = self._tag_assign_buffer
            return (f"Type tag slot 1–20 + ↩ to assign  |  "
                    f"Buffer: '{buf}█'  |  ⌫ clear  |  Esc cancel")
        if self.state == "DELETE_SELECT":
            return ("Click annotated beat (✕) to remove  |  "
                    "X again: clear all visible segments  |  Esc to exit")
        if self.state == "RENAME_THEME":
            return (f"Rename legacy theme {self._rename_theme_id}: "
                    f"'{self._rename_buffer}█'  |  ↩ confirm  |  Esc cancel")
        if self.state == "NEW_TAG_ID":
            return (f"New tag — type slot 1–20 + ↩  |  "
                    f"Buffer: '{self._new_tag_id_buffer}█'  |  Esc cancel")
        if self.state == "NEW_TAG_NAME":
            return (f"Slot {self._new_tag_pending_id} — type name + ↩  |  "
                    f"Buffer: '{self._new_tag_name_buffer}█'  |  Esc cancel")
        return ""

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax:
            return

        if event.button in ("scroll_left", "scroll_right"):
            direction = 1 if event.button == "scroll_right" else -1
            self.view_center_ns += int(self.view_width_ns * 0.15) * direction
            self._load_window()
            self._draw()
            return

        if event.step == 0:
            return
        scale     = ZOOM_FACTOR ** (-event.step)
        new_width = int(self.view_width_ns * scale)
        new_width = max(MIN_VIEW_MS, min(MAX_VIEW_MS, new_width))
        if new_width == self.view_width_ns:
            return

        if event.xdata is not None:
            view_sec    = self.view_width_ns / 1000
            cursor_frac = max(0.0, min(1.0, event.xdata / view_sec))
            lo_ns       = self.view_center_ns - self.view_width_ns // 2
            cursor_ns   = lo_ns + int(cursor_frac * self.view_width_ns)
            new_lo_ns   = cursor_ns - int(cursor_frac * new_width)
            self.view_center_ns = new_lo_ns + new_width // 2

        self.view_width_ns = new_width
        self._load_window()
        self._draw()

    def _on_key(self, event) -> None:
        k   = event.key
        now = _time.monotonic()

        # ── Double-press detection (BROWSE only) ──────────────────────────────
        if self.state == "BROWSE":
            prev   = self._last_key_time.get(k, 0.0)
            double = (now - prev) < _DOUBLE_PRESS_SEC
            self._last_key_time[k] = now

            if double and k in "123456789":
                # Rename a legacy theme label
                self._rename_theme_id = int(k)
                self._rename_buffer   = ""
                self.state            = "RENAME_THEME"
                self._draw()
                return

            if double and k == "enter":
                self._export_screenshots()
                return

            if double and k == "backspace":
                self._clear_window_annotations(beats=True, segments=True)
                return

        # ── BROWSE ────────────────────────────────────────────────────────────
        if self.state == "BROWSE":
            if k == "right":
                self.view_center_ns += self.view_width_ns // 5
                self._load_window(); self._draw()
            elif k == "left":
                self.view_center_ns -= self.view_width_ns // 5
                self._load_window(); self._draw()
            elif k == "t":
                self._seg_start_ns = None
                self.state = "SEG_START"
                self._draw()
            elif k == "x":
                self.state = "DELETE_SELECT"
                self._draw()
            elif k == "n":
                self._new_tag_id_buffer = ""
                self.state = "NEW_TAG_ID"
                self._draw()
            elif k == "b":
                if self.selected_peaks:
                    self._tag_assign_buffer = ""
                    self.state = "AWAIT_BEAT_THEME"
                    self._draw()
            elif k in (" ", "]"):
                if self.annotations_only:
                    self._go_to_annotation_window(self.ann_window_idx + 1)
                else:
                    self._go_to_marker(self.marker_idx + 1)
            elif k == "[":
                if self.annotations_only:
                    self._go_to_annotation_window(self.ann_window_idx - 1)
                else:
                    self._go_to_marker(self.marker_idx - 1)
            elif k == "d":
                self._delete_last_annotation()
            elif k == "escape":
                if self.selected_peaks:
                    self.selected_peaks = []
                    self._draw()
            elif k == "q":
                self._quit()

        # ── AWAIT SEG THEME ───────────────────────────────────────────────────
        elif self.state == "AWAIT_SEG_THEME":
            if k in "0123456789":
                self._tag_assign_buffer += k
                self._draw()
            elif k == "enter" and self._tag_assign_buffer:
                try:
                    num = int(self._tag_assign_buffer)
                except ValueError:
                    num = -1
                self._tag_assign_buffer = ""
                if 1 <= num <= 20:
                    self._save_segment_annotation(TAG_OFFSET + num)
                    self.state = "BROWSE"
                else:
                    print(f"  [T] Invalid slot '{num}' — must be 1-20.")
                self._draw()
            elif k == "backspace":
                self._tag_assign_buffer = self._tag_assign_buffer[:-1]
                self._draw()
            elif k == "escape":
                self._tag_assign_buffer = ""
                self.state = "BROWSE"
                self._draw()

        # ── AWAIT BEAT THEME ──────────────────────────────────────────────────
        elif self.state == "AWAIT_BEAT_THEME":
            if k in "0123456789":
                self._tag_assign_buffer += k
                self._draw()
            elif k == "enter" and self._tag_assign_buffer:
                try:
                    num = int(self._tag_assign_buffer)
                except ValueError:
                    num = -1
                self._tag_assign_buffer = ""
                if 1 <= num <= 20:
                    self._save_beat_annotations(TAG_OFFSET + num)
                    self.state          = "BROWSE"
                    self.selected_peaks = []
                else:
                    print(f"  [B] Invalid slot '{num}' — must be 1-20.")
                self._draw()
            elif k == "backspace":
                self._tag_assign_buffer = self._tag_assign_buffer[:-1]
                self._draw()
            elif k == "escape":
                self._tag_assign_buffer = ""
                self.state = "BROWSE"
                self._draw()

        # ── DELETE SELECT ─────────────────────────────────────────────────────
        elif self.state == "DELETE_SELECT":
            if k == "x":
                self._clear_window_annotations(beats=False, segments=True)
                self.state = "BROWSE"
            elif k == "escape":
                self.state = "BROWSE"
                self._draw()

        # ── SEG START ─────────────────────────────────────────────────────────
        elif self.state == "SEG_START":
            if k == "escape":
                self.state = "BROWSE"
                self._draw()

        # ── SEG END ───────────────────────────────────────────────────────────
        elif self.state == "SEG_END":
            if k == "escape":
                self._seg_start_ns = None
                self.state = "SEG_START"
                self._draw()

        # ── RENAME THEME (legacy) ─────────────────────────────────────────────
        elif self.state == "RENAME_THEME":
            if k == "enter":
                name = self._rename_buffer.strip()
                if name:
                    self.themes[str(self._rename_theme_id)] = name
                    p = OUTPUT_DIR / "theme_labels.json"
                    with open(p, "w") as f:
                        json.dump(self.themes, f, indent=2)
                    print(f"  [Rename] Legacy theme {self._rename_theme_id} → '{name}'")
                self.state = "BROWSE"
                self._draw()
            elif k == "escape":
                self.state = "BROWSE"
                self._draw()
            elif k == "backspace":
                self._rename_buffer = self._rename_buffer[:-1]
                self._draw()
            elif len(k) == 1:
                self._rename_buffer += k
                self._draw()

        # ── NEW TAG ID ────────────────────────────────────────────────────────
        elif self.state == "NEW_TAG_ID":
            if k in "0123456789":
                self._new_tag_id_buffer += k
                self._draw()
            elif k == "enter" and self._new_tag_id_buffer:
                try:
                    slot = int(self._new_tag_id_buffer)
                except ValueError:
                    slot = -1
                self._new_tag_id_buffer = ""
                if 1 <= slot <= 20:
                    self._new_tag_pending_id  = slot
                    self._new_tag_name_buffer = ""
                    self.state = "NEW_TAG_NAME"
                    self._draw()
                else:
                    print(f"  [N] Slot must be 1-20 (got {slot}).")
                    self.state = "BROWSE"
                    self._draw()
            elif k == "backspace":
                self._new_tag_id_buffer = self._new_tag_id_buffer[:-1]
                self._draw()
            elif k == "escape":
                self._new_tag_id_buffer = ""
                self.state = "BROWSE"
                self._draw()

        # ── NEW TAG NAME ──────────────────────────────────────────────────────
        elif self.state == "NEW_TAG_NAME":
            if k == "enter":
                name = self._new_tag_name_buffer.strip()
                if name and self._new_tag_pending_id is not None:
                    self.tags[str(self._new_tag_pending_id)] = name
                    p = OUTPUT_DIR / "tag_labels.json"
                    with open(p, "w") as f:
                        json.dump(self.tags, f, indent=2)
                    print(f"  [N] Tag slot {self._new_tag_pending_id} → '{name}'")
                self._new_tag_name_buffer = ""
                self._new_tag_pending_id  = None
                self.state = "BROWSE"
                self._draw()
            elif k == "escape":
                self._new_tag_name_buffer = ""
                self._new_tag_pending_id  = None
                self.state = "BROWSE"
                self._draw()
            elif k == "backspace":
                self._new_tag_name_buffer = self._new_tag_name_buffer[:-1]
                self._draw()
            elif len(k) == 1:
                self._new_tag_name_buffer += k
                self._draw()

    def _on_click(self, event) -> None:
        if self.state not in ("BROWSE", "DELETE_SELECT", "SEG_START", "SEG_END"):
            return
        if event.inaxes != self.ax or event.xdata is None:
            return

        lo_ns    = self.view_center_ns - self.view_width_ns // 2
        click_ns = lo_ns + int(event.xdata * 1000)

        if self.state == "BROWSE":
            if len(self.peaks_df) == 0:
                return
            dists     = (self.peaks_df["timestamp_ms"] - click_ns).abs()
            nearest_i = dists.idxmin()
            if dists[nearest_i] > SNAP_THRESHOLD_MS:
                return
            pid = int(self.peaks_df.at[nearest_i, "peak_id"])
            if pid in self.selected_peaks:
                self.selected_peaks.remove(pid)
            else:
                self.selected_peaks.append(pid)
            self._draw()

        elif self.state == "DELETE_SELECT":
            best_i, best_dist = None, SNAP_THRESHOLD_MS
            for i, ann in enumerate(self.beat_anns):
                dist = abs(int(ann["peak_timestamp_ns"]) - click_ns)
                if dist < best_dist:
                    best_dist = dist
                    best_i    = i
            if best_i is not None:
                ann = self.beat_anns.pop(best_i)
                print(f"  [X] Removed beat ann id={ann['ann_id']} "
                      f"(theme_id {ann['theme_id']})")
                self._rewrite_csv()
                self._draw()
            else:
                print("  [X] No annotated beat within snap range.")

        elif self.state == "SEG_START":
            self._seg_start_ns = click_ns
            self.state = "SEG_END"
            self._draw()

        elif self.state == "SEG_END":
            self._seg_end_ns = click_ns
            if self._seg_end_ns < self._seg_start_ns:
                self._seg_start_ns, self._seg_end_ns = (
                    self._seg_end_ns, self._seg_start_ns
                )
            self._tag_assign_buffer = ""
            self.state = "AWAIT_SEG_THEME"
            self._draw()

    # ── Annotation persistence ────────────────────────────────────────────────

    def _current_marker_info(self) -> tuple[int, str]:
        """Return (marker_idx_1indexed, marker_dt_str) for the current context."""
        if self.annotations_only:
            dt_str = datetime.fromtimestamp(
                self.view_center_ns / 1000, tz=LOCAL_TZ
            ).strftime("%Y-%m-%d %H:%M:%S")
            return self.ann_window_idx + 1, dt_str
        m = self.markers[self.marker_idx]
        return self.marker_idx + 1, m["raw"]

    def _save_segment_annotation(self, theme_id: int) -> None:
        midx, mdt = self._current_marker_info()
        row = {
            "ann_id":        self._next_ann_id,
            "marker_idx":    midx,
            "marker_dt":     mdt,
            "theme_id":      theme_id,
            "view_start_ns": self._seg_start_ns,
            "view_end_ns":   self._seg_end_ns,
            "annotated_at":  datetime.now(LOCAL_TZ).isoformat(),
        }
        self.seg_anns.append(row)
        self._next_ann_id += 1
        sp = OUTPUT_DIR / "segment_annotations.csv"
        write_header = not sp.exists()
        with open(sp, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.SEG_HEADER)
            if write_header:
                w.writeheader()
            w.writerow(row)
        dur_ms = self._seg_end_ns - self._seg_start_ns
        print(f"  [T] Segment [{self._ann_full_label(theme_id)}] "
              f"{dur_ms}ms → marker {midx}")

    def _save_beat_annotations(self, theme_id: int) -> None:
        midx, mdt = self._current_marker_info()
        now = datetime.now(LOCAL_TZ).isoformat()
        bp  = OUTPUT_DIR / "beat_annotations.csv"
        write_header = not bp.exists()
        with open(bp, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.BEAT_HEADER)
            if write_header:
                w.writeheader()
            for pid in self.selected_peaks:
                prow = self.peaks_df[self.peaks_df["peak_id"] == pid]
                if len(prow) == 0:
                    continue
                pts_ns = int(prow.iloc[0]["timestamp_ms"])
                row = {
                    "ann_id":            self._next_ann_id,
                    "marker_idx":        midx,
                    "marker_dt":         mdt,
                    "theme_id":          theme_id,
                    "peak_id":           pid,
                    "peak_timestamp_ns": pts_ns,
                    "annotated_at":      now,
                }
                self.beat_anns.append(row)
                self._next_ann_id += 1
                w.writerow(row)
        print(f"  [B] {len(self.selected_peaks)} beat(s) "
              f"[{self._ann_full_label(theme_id)}] → marker {midx}")

    def _delete_last_annotation(self) -> None:
        lo_ns = self.view_center_ns - self.view_width_ns // 2
        hi_ns = self.view_center_ns + self.view_width_ns // 2
        deleted = False
        for ann_list, atype in [(self.beat_anns, "beat"), (self.seg_anns, "seg")]:
            for i in range(len(ann_list) - 1, -1, -1):
                ann = ann_list[i]
                if atype == "beat":
                    if not (lo_ns <= int(ann["peak_timestamp_ns"]) <= hi_ns):
                        continue
                else:
                    if (int(ann["view_end_ns"]) < lo_ns
                            or int(ann["view_start_ns"]) > hi_ns):
                        continue
                ann_list.pop(i)
                print(f"  [D] Deleted {atype} ann id={ann['ann_id']}")
                deleted = True
                break
            if deleted:
                break
        if deleted:
            self._rewrite_csv()
            self._draw()
        else:
            print("  [D] No annotation in current window to delete.")

    def _rewrite_csv(self) -> None:
        sp = OUTPUT_DIR / "segment_annotations.csv"
        bp = OUTPUT_DIR / "beat_annotations.csv"
        if self.seg_anns:
            with open(sp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.SEG_HEADER)
                w.writeheader(); w.writerows(self.seg_anns)
        elif sp.exists():
            sp.unlink()
        if self.beat_anns:
            with open(bp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.BEAT_HEADER)
                w.writeheader(); w.writerows(self.beat_anns)
        elif bp.exists():
            bp.unlink()

    def _clear_window_annotations(
        self, beats: bool = True, segments: bool = True
    ) -> None:
        lo_ns = self.view_center_ns - self.view_width_ns // 2
        hi_ns = self.view_center_ns + self.view_width_ns // 2
        if beats:
            before = len(self.beat_anns)
            self.beat_anns = [
                a for a in self.beat_anns
                if not (lo_ns <= int(a["peak_timestamp_ns"]) <= hi_ns)
            ]
            print(f"  [Clear] {before - len(self.beat_anns)} beat ann(s) removed.")
        if segments:
            before = len(self.seg_anns)
            self.seg_anns = [
                a for a in self.seg_anns
                if not (int(a["view_start_ns"]) <= hi_ns
                        and int(a["view_end_ns"]) >= lo_ns)
            ]
            print(f"  [Clear] {before - len(self.seg_anns)} seg ann(s) removed.")
        self._rewrite_csv()
        self._draw()

    def _export_screenshots(self) -> None:
        """Export one PNG per (marker_idx, theme_id) group. Double-press Enter."""
        screenshot_dir = OUTPUT_DIR / "screenshots"
        screenshot_dir.mkdir(exist_ok=True)

        beat_groups: dict = defaultdict(list)
        for ann in self.beat_anns:
            key = (int(ann["marker_idx"]), int(ann["theme_id"]), ann["marker_dt"])
            beat_groups[key].append(ann)

        seg_groups: dict = defaultdict(list)
        for ann in self.seg_anns:
            key = (int(ann["marker_idx"]), int(ann["theme_id"]), ann["marker_dt"])
            seg_groups[key].append(ann)

        all_keys = sorted(set(beat_groups) | set(seg_groups))
        if not all_keys:
            print("  [Export] No annotations to screenshot.")
            return
        print(f"  [Export] Writing {len(all_keys)} screenshot(s) → {screenshot_dir} …")

        for key in all_keys:
            m_idx, theme_id, m_dt = key
            b_anns = beat_groups.get(key, [])
            s_anns = seg_groups.get(key, [])
            c      = self._ann_color(theme_id)
            label  = self._ann_full_label(theme_id)

            ts_pts = [int(a["peak_timestamp_ns"]) for a in b_anns]
            for a in s_anns:
                ts_pts += [int(a["view_start_ns"]), int(a["view_end_ns"])]
            if not ts_pts:
                continue

            pk_wide = _load_peaks(self.processed,
                                  min(ts_pts) - 8_000,
                                  max(ts_pts) + 8_000)
            if len(pk_wide) >= 2:
                sorted_ts = np.sort(pk_wide["timestamp_ms"].values.astype(np.int64))
                lo_i  = max(0, int(np.searchsorted(sorted_ts, min(ts_pts))) - 5)
                hi_i  = min(len(sorted_ts) - 1,
                            int(np.searchsorted(sorted_ts,
                                                max(ts_pts), side="right")) + 4)
                win_lo = int(sorted_ts[lo_i])
                win_hi = int(sorted_ts[hi_i])
            else:
                win_lo = min(ts_pts) - 5_000
                win_hi = max(ts_pts) + 5_000

            overlapping = self.segments[
                (self.segments["end_timestamp_ms"] >= win_lo) &
                (self.segments["start_timestamp_ms"] <= win_hi)
            ]
            if len(overlapping) == 0:
                print(f"  [Export] M{m_idx}/{label}: no ECG segment — skip")
                continue

            ts_parts, ecg_parts = [], []
            for seg in overlapping.itertuples():
                ts_s, ecg_s = _load_ecg(self.processed, seg.segment_idx,
                                        win_lo, win_hi)
                ts_parts.append(ts_s); ecg_parts.append(ecg_s)

            ecg_ts   = np.concatenate(ts_parts)
            ecg_vals = np.concatenate(ecg_parts)
            order    = np.argsort(ecg_ts)
            ecg_ts   = ecg_ts[order]
            ecg_vals = ecg_vals[order]

            if len(ecg_ts) < 10:
                print(f"  [Export] M{m_idx}/{label}: too little ECG — skip")
                continue

            pk_win = _load_peaks(self.processed, win_lo, win_hi)
            if len(pk_win):
                sn_all, sy_all = _snap_peaks(
                    pk_win["timestamp_ms"].values.astype(np.int64),
                    ecg_ts, ecg_vals,
                )
            else:
                sn_all = np.array([], dtype=np.int64)
                sy_all = np.array([], dtype=np.float32)

            fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0d1117")
            ax.set_facecolor("#16213e")
            t_sec = (ecg_ts - win_lo) / 1000
            ax.plot(t_sec, ecg_vals, color="#4fc3f7", linewidth=0.65, alpha=0.92)
            if len(sn_all):
                ax.scatter((sn_all - win_lo) / 1000, sy_all,
                           color="#546e7a", s=20, zorder=4, alpha=0.7)

            for ann in b_anns:
                sn, sy = _snap_peaks(
                    np.array([int(ann["peak_timestamp_ns"])], dtype=np.int64),
                    ecg_ts, ecg_vals,
                )
                ax.scatter([(sn[0] - win_lo) / 1000], [float(sy[0])],
                           color=c, s=100, zorder=7,
                           marker="D", edgecolors="white", linewidths=0.8)

            view_sec = (win_hi - win_lo) / 1000
            for ann in s_anns:
                x0 = (int(ann["view_start_ms"]) - win_lo) / 1000
                x1 = (int(ann["view_end_ms"])   - win_lo) / 1000
                ax.axvspan(max(0, x0), min(view_sec, x1), alpha=0.12, color=c)

            ax.set_xlim(0, view_sec)
            tick_step = 2 if view_sec <= 20 else 5
            tk = np.arange(0, view_sec + 0.01, tick_step)
            ax.set_xticks(tk)
            ax.set_xticklabels(
                [datetime.fromtimestamp((win_lo + s * 1000) / 1000,
                                        tz=LOCAL_TZ).strftime("%H:%M:%S")
                 for s in tk],
                color="#78909c", fontsize=7,
            )
            ax.set_xlabel("seconds", color="#78909c", fontsize=8)
            ax.tick_params(axis="y", colors="#546e7a", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a3a5c")
            fig.suptitle(
                f"M{m_idx}  ·  {label}  ·  {len(b_anns)} beat(s)  ·  {m_dt}",
                color="#e0e0e0", fontsize=9,
            )
            fig.tight_layout()
            m_dt_safe = m_dt.replace(":", "-").replace(" ", "_")
            tid_str   = (f"tag{theme_id - TAG_OFFSET}" if theme_id > TAG_OFFSET
                         else f"t{theme_id}")
            fname = screenshot_dir / f"m{m_idx:03d}_{tid_str}_{m_dt_safe}.png"
            fig.savefig(fname, dpi=120, bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)
            print(f"  [Export] {fname.name}")

        print(f"  [Export] Done — {len(all_keys)} screenshot(s) in {screenshot_dir}")

    def _quit(self) -> None:
        print(f"\nSaved {len(self.seg_anns)} segment ann(s) and "
              f"{len(self.beat_anns)} beat ann(s) to {OUTPUT_DIR}")
        plt.close(self.fig)

    def run(self) -> None:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Master annotation ECG viewer")
    parser.add_argument("--annotation-file", default=str(DEFAULT_MASTER_ANNOTATIONS),
                        help="Master legacy annotation JSON to browse")
    parser.add_argument("--edit-file", default=None,
                        help="Writable JSON copy for master-mode edits")
    parser.add_argument("--ecg-dir", default=str(DEFAULT_ECG_DIR),
                        help="Directory containing raw ECG CSV files")
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR),
                        help="Optional processed dir for peaks.parquet clean markers")
    parser.add_argument("--start", type=int, default=1,
                        help="Start at annotated window N (1-indexed)")
    parser.add_argument("--order", choices=["asc", "desc"], default="desc",
                        help="Marker mode only: asc = oldest first; desc = newest first")
    parser.add_argument("--annotations-only", action="store_true",
                        help="Marker mode only: navigate only existing CSV annotations")
    parser.add_argument("--marker-mode", action="store_true",
                        help="Run the old marker-centric annotator instead")
    parser.add_argument("--v", "--validated", dest="include_validated",
                        action="store_true",
                        help="Master mode only: include windows already marked done")
    parser.add_argument("--r", dest="include_revisit", action="store_true",
                        help="Master mode only: include revisit-pile windows")
    parser.add_argument("--clean", action="store_true",
                        help="Master mode only: show only legacy clean-only segments")
    args = parser.parse_args()

    if args.marker_mode:
        if pd is None or pq is None:
            print("ERROR: marker mode requires pandas and pyarrow.")
            sys.exit(1)
        processed = Path(args.processed_dir)
        if not processed.exists():
            print(f"ERROR: {processed} not found.")
            sys.exit(1)
        if not MARKER_CSV.exists():
            print(f"ERROR: Marker.csv not found at {MARKER_CSV}")
            sys.exit(1)

        print(f"Starting marker annotator from {MARKER_CSV}")
        MarkerAnnotator(
            processed,
            start_idx=args.start - 1,
            order=args.order,
            annotations_only=args.annotations_only,
        ).run()
        return

    annotation_file = Path(args.annotation_file)
    ecg_dir = Path(args.ecg_dir)
    processed = Path(args.processed_dir) if args.processed_dir else None
    if not annotation_file.exists():
        print(f"ERROR: annotation file not found: {annotation_file}")
        sys.exit(1)
    if not ecg_dir.exists():
        print(f"ERROR: ECG directory not found: {ecg_dir}")
        sys.exit(1)

    if args.edit_file:
        edit_file = Path(args.edit_file)
    else:
        edit_file = annotation_file.with_name(
            f"V1_Corrected.json"
        )
    if edit_file.resolve() == annotation_file.resolve():
        print("ERROR: --edit-file must be different from --annotation-file.")
        sys.exit(1)
    if not edit_file.exists():
        edit_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(annotation_file, edit_file)
        print(f"Created editable annotation copy: {edit_file}")
    else:
        print(f"Using existing editable annotation copy: {edit_file}")

    MasterAnnotationViewer(
        annotation_file=edit_file,
        ecg_dir=ecg_dir,
        processed=processed,
        start_idx=args.start - 1,
        source_annotation_file=annotation_file,
        include_validated=args.include_validated,
        include_revisit=args.include_revisit,
        show_clean_only=args.clean,
    ).run()


if __name__ == "__main__":
    main()
