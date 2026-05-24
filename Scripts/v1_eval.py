#!/usr/bin/env python3
"""
Recreated V1 ECG annotation GUI, now operating as a viewer over the V4
annotation parquets:

    /Volumes/xHRV/Data/Annotations/V4/segments.parquet
    /Volumes/xHRV/Data/Annotations/V4/beats.parquet
    /Volumes/xHRV/Data/Annotations/V4/bad_regions.parquet

The waveform and current peak tables are loaded from the processed parquets
under /Volumes/xHRV/Data/Processed (and the archived peaks parquet). The raw
ECG/peak directories are still accepted as CLI arguments for provenance.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds

plt = None
Rectangle = None

try:
    from config import (
        PHYSIO_CONSTRAINT_REVIEW_PEAKS_PQT,
        MARKER_REVIEW_RECOVERY_PQT,
        MARKER_CSV,
    )
except Exception:
    PHYSIO_CONSTRAINT_REVIEW_PEAKS_PQT = Path("/Volumes/xHRV/Diagnostics/physio_constraint_review_peaks.parquet")
    MARKER_REVIEW_RECOVERY_PQT = Path("/Volumes/xHRV/Diagnostics/marker_review_recovery_segments.parquet")
    MARKER_CSV = Path("/Volumes/xHRV/Data/Accessory/Marker.csv")


def require_matplotlib() -> None:
    global plt, Rectangle
    try:
        import matplotlib
    except ModuleNotFoundError:
        print(
            "Missing Python dependency: matplotlib. Run this with the same Python environment "
            "used for the prior ECG annotation GUIs, or install matplotlib into this environment.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        matplotlib.use("MacOSX")
    except Exception:
        pass

    import matplotlib.pyplot as pyplot
    from matplotlib.patches import Rectangle as MplRectangle

    plt = pyplot
    Rectangle = MplRectangle


ROOT = Path("/Volumes/xHRV")
V4_DIR = ROOT / "Data/Annotations/V4"
LEGACY_PROCESSED_DIR = ROOT / "Data/Processed"
CURRENT_PROCESSED_DIR = ROOT / "Processed"
PROCESSED_DIR = CURRENT_PROCESSED_DIR if CURRENT_PROCESSED_DIR.exists() else LEGACY_PROCESSED_DIR

OUT_SEGMENTS = V4_DIR / "segments.parquet"
OUT_BEATS = V4_DIR / "beats.parquet"
OUT_BAD = V4_DIR / "bad_regions.parquet"
ACTION_LOG = V4_DIR / "v1_eval_action_history.txt"

ECG_PARQUET = PROCESSED_DIR / "ecg_samples.parquet"
PEAKS_PARQUET = PROCESSED_DIR / "peaks.parquet"
PROCESSED_SEGMENTS_PARQUET = PROCESSED_DIR / "segments.parquet"

LOCAL_TZ = ZoneInfo("America/New_York")
DISPLAY_MS = 60_000
MIN_VIEW_MS = 2_500
MAX_VIEW_MS = 120_000
ZOOM_FACTOR = 1.25
CLICK_RADIUS_PX = 24.0
RR_CHANGE_FULL_SCALE = 0.55
DISPLAY_PEAK_SNAP_WINDOW_MS = 8
ANNOTATION_MATCH_TOLERANCE_MS = 8
ADDED_PEAK_SNAP_WINDOW_MS = 5
PRECISE_ADDED_PEAK_SNAP_WINDOW_MS = 1
WINDOW_CACHE_LIMIT = 48
MAX_RR_TEXT_LABELS = 220
DEFAULT_RMSSD_ARTIFACT_MS = 140.0
DEFAULT_MIN_RR_ARTIFACT_MS = 300
DEFAULT_PHYSIO_DELTA_MS = 350
DEFAULT_REVIEW_WINDOW_MS = 12_000
REVIEW_SHADE_MS = 130
MARKER_FLANK_SEGMENTS = 15
FEBRUARY_2025_START_MS = int(datetime(2025, 2, 1, tzinfo=timezone.utc).timestamp() * 1000)
MARCH_2025_START_MS = int(datetime(2025, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)

BG = "#17213d"
PANEL = "#0d111f"
ECG_COLOR = "#4bb8ef"
PEAK_COLOR = "#f4a340"
ARTIFACT_COLOR = "#ff4f73"
PHYSIO_COLOR = "#b46cff"
ADDED_EDGE = "#fff2cf"
BAD_REGION_COLOR = "#ff3d52"
PAC_PVC_COLOR = "#ff89d6"
PHYSIO_AFTEREFFECT_COLOR = "#5eead4"
ARRHYTHMIA_COLOR = "#f9e76e"
RR_BASE = (244, 207, 79)
RR_DECREASE = (255, 79, 115)
RR_INCREASE = (72, 202, 255)


SEGMENT_COLUMNS = [
    "segment_idx",
    "start_ms",
    "end_ms",
    "review_status",
    "segment_quality_label",
    "in_revisit_pile",
    "has_bad_region",
    "bad_region_count",
    "beat_training_eligible_segment",
    "segment_quality_training_eligible",
    "comment",
]

BEAT_COLUMNS = [
    "timestamp_ms",
    "segment_idx",
    "label",
    "subtype",
    "is_added_peak",
    "beat_training_eligible",
    "comment",
]

BAD_COLUMNS = [
    "segment_idx",
    "region_idx_within_segment",
    "start_ms",
    "end_ms",
    "original_annotation",
    "comment",
]

PARQUET_SCHEMAS = {
    "segments": pa.schema(
        [
            ("segment_idx", pa.int32()),
            ("start_ms", pa.int64()),
            ("end_ms", pa.int64()),
            ("review_status", pa.string()),
            ("segment_quality_label", pa.string()),
            ("in_revisit_pile", pa.bool_()),
            ("has_bad_region", pa.bool_()),
            ("bad_region_count", pa.int32()),
            ("beat_training_eligible_segment", pa.bool_()),
            ("segment_quality_training_eligible", pa.bool_()),
            ("comment", pa.string()),
        ]
    ),
    "beats": pa.schema(
        [
            ("timestamp_ms", pa.int64()),
            ("segment_idx", pa.int32()),
            ("label", pa.string()),
            ("subtype", pa.string()),
            ("is_added_peak", pa.bool_()),
            ("beat_training_eligible", pa.bool_()),
            ("comment", pa.string()),
        ]
    ),
    "bad": pa.schema(
        [
            ("segment_idx", pa.int32()),
            ("region_idx_within_segment", pa.int32()),
            ("start_ms", pa.int64()),
            ("end_ms", pa.int64()),
            ("original_annotation", pa.string()),
            ("comment", pa.string()),
        ]
    ),
}

def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def q(path: Path | str) -> str:
    return str(path).replace("'", "''")


def duckdb_bin() -> str:
    found = shutil.which("duckdb") or "/opt/homebrew/bin/duckdb"
    if not Path(found).exists():
        _die("duckdb CLI is required but was not found.")
    return found


DUCKDB = duckdb_bin()


def duck_json(sql: str) -> list[dict[str, Any]]:
    proc = subprocess.run(
        [DUCKDB, "-json", "-c", sql],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"duckdb query failed:\n{proc.stderr}\nSQL:\n{sql}")
    text = proc.stdout.strip()
    if not text:
        return []
    return json.loads(text)


def duck_exec(sql: str) -> None:
    proc = subprocess.run(
        [DUCKDB, "-c", sql],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"duckdb command failed:\n{proc.stderr}\nSQL:\n{sql}")


def parquet_columns(path: Path) -> list[str]:
    rows = duck_json(f"DESCRIBE SELECT * FROM read_parquet('{q(path)}')")
    return [str(r["column_name"]) for r in rows]


def parse_timestamp_ms(value: Any) -> int:
    if value is None:
        raise ValueError("missing timestamp")
    if isinstance(value, (int, float)):
        numeric = int(round(float(value)))
        if abs(numeric) > 10_000_000_000_000_000:
            return numeric // 1_000_000
        if abs(numeric) > 10_000_000_000_000:
            return numeric // 1_000
        return numeric
    text = str(value).strip()
    if not text:
        raise ValueError("blank timestamp")
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return parse_timestamp_ms(int(text))
    if text.endswith("Z"):
        text = text[:-1]
    if "." in text:
        head, tail = text.split(".", 1)
        digits = "".join(ch for ch in tail if ch.isdigit())
        text = f"{head}.{digits[:6].ljust(6, '0')}"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(round(dt.timestamp() * 1000))


def parse_segment_ref(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return int_or_none(value.get("segment_idx"))
    text = str(value)
    for prefix in ("segment:", "seg_", "segment_", "seg:"):
        if text.startswith(prefix):
            return int_or_none(text[len(prefix):])
    return int_or_none(text)


def segment_lookup_from_timestamps(segment_rows: list[dict[str, Any]]) -> tuple[list[int], dict[int, tuple[int, int]]]:
    ranges = {
        int(row["segment_idx"]): (int(row["start_ms"]), int(row["end_ms"]))
        for row in segment_rows
    }
    starts = sorted(ranges)
    return starts, ranges


def find_segment_for_ts(timestamp_ms: int, ordered_segment_ids: list[int], ranges: dict[int, tuple[int, int]]) -> int | None:
    lo = 0
    hi = len(ordered_segment_ids) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        seg_idx = ordered_segment_ids[mid]
        start_ms, end_ms = ranges[seg_idx]
        if timestamp_ms < start_ms:
            hi = mid - 1
        elif timestamp_ms > end_ms:
            lo = mid + 1
        else:
            return seg_idx
    return None


def quality_fields(quality: str, has_bad_region: bool) -> tuple[bool, bool]:
    if quality == "unusable":
        return False, True
    if has_bad_region or quality in {"partial", "usable", "unknown", ""}:
        return True, True
    return True, True


def normalize_segment_training_flags(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    is_revisit = str(row.get("review_status")) == "revisit" or bool_value(row.get("in_revisit_pile"))
    bad_count = int_or_none(row.get("bad_region_count")) or 0
    if is_revisit:
        row["review_status"] = "revisit"
        row["in_revisit_pile"] = True
        row["segment_quality_label"] = "partial" if bad_count != 0 else "unknown"
        row["beat_training_eligible_segment"] = False
        row["segment_quality_training_eligible"] = False
        return row
    row["in_revisit_pile"] = False
    if str(row.get("review_status")) == "unreviewed":
        row["beat_training_eligible_segment"] = False
        row["segment_quality_training_eligible"] = False
    return row


def split_revisit_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main: list[dict[str, Any]] = []
    revisit: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in rows:
        row = normalize_segment_training_flags(raw)
        seg_idx = int(row["segment_idx"])
        if seg_idx in seen:
            continue
        seen.add(seg_idx)
        if str(row.get("review_status")) == "revisit" or bool_value(row.get("in_revisit_pile")):
            revisit.append(row)
        else:
            main.append(row)
    main.sort(key=lambda r: int(r["segment_idx"]))
    revisit.sort(key=lambda r: int(r["segment_idx"]))
    return main, revisit


def ensure_inputs() -> None:
    for required in (OUT_SEGMENTS, OUT_BEATS, OUT_BAD):
        if not required.exists():
            _die(f"Missing required V4 parquet: {required}")


def load_parquet(path: Path) -> list[dict[str, Any]]:
    return duck_json(f"SELECT * FROM read_parquet('{q(path)}')")


def load_optional_parquet(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return normalize_rows(load_parquet(path), columns)


def load_marker_timestamps(path: Path) -> list[int]:
    if not path.exists():
        alt = path.with_name(path.name.lower())
        if alt.exists():
            path = alt
        else:
            _die(f"Missing marker CSV: {path}")
    timestamps: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            value = str(row[0]).strip()
            if not value:
                continue
            if row_idx == 0 and not value.lstrip("-").isdigit():
                continue
            timestamps.append(parse_timestamp_ms(value))
    return sorted(set(timestamps))


def merge_segment_rows(base_rows: list[dict[str, Any]], overlay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {int(row["segment_idx"]): dict(row) for row in base_rows}
    for row in overlay_rows:
        merged[int(row["segment_idx"])] = dict(row)
    return normalize_rows(sorted(merged.values(), key=lambda r: int(r["segment_idx"])), SEGMENT_COLUMNS)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "t", "yes", "y"}


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def normalize_rows(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        fixed = {col: row.get(col) for col in columns}
        if "subtype" in columns and fixed.get("subtype") is None and "original_annotation" in row:
            fixed["subtype"] = row.get("original_annotation")
        fixed["comment"] = fixed.get("comment") or ""
        out.append(fixed)
    return out


def normalize_processed_segment_row(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("segment_quality_label", row.get("quality_label", "unknown"))
    return {
        "segment_idx": int(row["segment_idx"]),
        "start_ms": int(row.get("start_ms", row.get("start_timestamp_ms"))),
        "end_ms": int(row.get("end_ms", row.get("end_timestamp_ms"))),
        "review_status": row.get("review_status") or "unreviewed",
        "segment_quality_label": label or "unknown",
        "in_revisit_pile": False,
        "has_bad_region": False,
        "bad_region_count": 0,
        "beat_training_eligible_segment": bool_value(row.get("beat_training_eligible_segment")),
        "segment_quality_training_eligible": bool_value(row.get("segment_quality_training_eligible")),
        "comment": "",
    }


def write_parquet(path: Path, rows: list[dict[str, Any]], columns: list[str], table: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v1_eval_", dir=str(path.parent)) as td:
        tmp_out = Path(td) / f"{table}.parquet"
        schema = PARQUET_SCHEMAS[table]
        cleaned_rows = []
        for row in rows:
            cleaned: dict[str, Any] = {}
            for field in schema:
                value = row.get(field.name)
                if pa.types.is_boolean(field.type):
                    value = bool_value(value)
                elif pa.types.is_integer(field.type):
                    value = int(value) if value not in {None, ""} else None
                elif pa.types.is_string(field.type):
                    value = "" if value is None else str(value)
                cleaned[field.name] = value
            cleaned_rows.append(cleaned)
        out_table = pa.Table.from_pylist(cleaned_rows, schema=schema)
        pq.write_table(out_table, tmp_out, compression="zstd")
        tmp_out.replace(path)


def ms_to_local(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def is_february_2025_ms(ms: int) -> bool:
    return FEBRUARY_2025_START_MS <= int(ms) < MARCH_2025_START_MS


def blend(base: tuple[int, int, int], target: tuple[int, int, int], amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    vals = [round(base[i] + (target[i] - base[i]) * amount) for i in range(3)]
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


@dataclass
class PeakPoint:
    peak_id: int | None
    timestamp_ms: int
    y: float
    source: str
    is_added_peak: bool
    label: str
    subtype: str = ""
    raw_timestamp_ms: int | None = None
    segment_idx: int | None = None


@dataclass
class SegmentMetrics:
    rr_count: int = 0
    rmssd_ms: float = 0.0
    min_rr_ms: int | None = None
    max_rr_increase_ms: int | None = None
    max_rr_change_pct: float = 0.0
    has_rr_drop_gt_20: bool = False
    has_rr_rise_gt_20_nonphysio: bool = False
    has_physio_rise: bool = False


def classify_segment_queue(
    has_annotations: bool,
    metrics: SegmentMetrics | None,
    rmssd_artifact_ms: float = DEFAULT_RMSSD_ARTIFACT_MS,
    min_rr_artifact_ms: int = DEFAULT_MIN_RR_ARTIFACT_MS,
    physio_delta_ms: int = DEFAULT_PHYSIO_DELTA_MS,
) -> str:
    if has_annotations:
        return "annotated"
    if metrics is None:
        return "likely-clean"
    if metrics.min_rr_ms is not None and metrics.min_rr_ms < min_rr_artifact_ms:
        return "likely-artifact"
    if metrics.rmssd_ms >= rmssd_artifact_ms:
        return "likely-artifact"
    if metrics.has_rr_drop_gt_20 or metrics.has_rr_rise_gt_20_nonphysio:
        return "likely-artifact"
    if metrics.has_physio_rise or (metrics.max_rr_increase_ms is not None and metrics.max_rr_increase_ms >= physio_delta_ms):
        return "likely-physio"
    return "likely-clean"


def low_amplitude_artifact_candidates(points: list[PeakPoint]) -> list[PeakPoint]:
    ordered = sorted([p for p in points if p.label != "artifact"], key=lambda p: p.timestamp_ms)
    if len(ordered) < 2:
        return []
    heights = [float(p.y) for p in ordered]
    avg_height = sum(heights) / len(heights)
    by_ts: dict[int, PeakPoint] = {}
    for idx, point in enumerate(ordered):
        height = heights[idx]
        low_vs_neighbors = False
        if 0 < idx < len(ordered) - 1:
            prev_height = heights[idx - 1]
            next_height = heights[idx + 1]
            low_vs_neighbors = height < 0.25 * prev_height and height < 0.25 * next_height
        low_vs_segment = height < 0.50 * avg_height
        if low_vs_neighbors or low_vs_segment:
            by_ts[point.timestamp_ms] = point
    return sorted(by_ts.values(), key=lambda p: p.timestamp_ms)


class ActionLogger:
    def __init__(self, path: Path, session_details: dict[str, Any] | None = None) -> None:
        self.path = path
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._set_append_only_flag()
        self.log("session_start", details=session_details or {})

    def _set_append_only_flag(self) -> None:
        chflags = shutil.which("chflags")
        if chflags is None:
            return
        subprocess.run(
            [chflags, "uappnd", str(self.path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def log(self, action: str, segment_idx: int | None = None, details: dict[str, Any] | None = None) -> None:
        details = details or {}
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "action": action,
            "segment_idx": segment_idx,
            "details": details,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        summary_bits = [f"action={action}"]
        if segment_idx is not None:
            summary_bits.append(f"segment={segment_idx}")
        for key in ("count", "restored_kind", "reason"):
            if key in details:
                summary_bits.append(f"{key}={details[key]}")
        print("[v1_eval_action] " + " ".join(summary_bits), flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        if not self.handle.closed:
            self.log("session_end")
            self.handle.close()


class MasterAnnotationViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        ensure_inputs()

        segment_rows = load_parquet(OUT_SEGMENTS)
        self.segments = normalize_rows(segment_rows, SEGMENT_COLUMNS)
        self.beats = normalize_rows(load_parquet(OUT_BEATS), BEAT_COLUMNS)
        self.bad_regions = normalize_rows(load_parquet(OUT_BAD), BAD_COLUMNS)
        self.segment_by_idx = {int(r["segment_idx"]): r for r in self.segments}
        self.processed_segment_by_idx: dict[int, dict[str, Any]] = {}
        self.timestamp_segment_cache: dict[int, int] = {}
        self.physio_review_table = None
        self.physio_review_total = 0
        self.physio_review_segment_cache: dict[int, dict[str, Any]] = {}
        self.physio_review_row_cache: dict[int, dict[str, Any]] = {}
        self.recovery_queue_mode = False
        self.marker_groups: dict[int, list[int]] = {}
        self.marker_order: list[int] = []
        self.marker_position: int = 0
        self.beats_by_segment: dict[int, list[dict[str, Any]]] = {}
        self.added_beats_by_timestamp: list[tuple[int, dict[str, Any]]] = []
        for row in self.beats:
            seg_idx = int(row["segment_idx"])
            self.beats_by_segment.setdefault(seg_idx, []).append(row)
        self._rebuild_added_beat_index()
        self.bad_by_segment: dict[int, list[dict[str, Any]]] = {}
        for row in self.bad_regions:
            seg_idx = int(row["segment_idx"])
            self.bad_by_segment.setdefault(seg_idx, []).append(row)
        self.physio_reviewed_segment_idxs = {
            int(row["segment_idx"])
            for row in self.segments
            if str(row.get("review_status") or "") == "reviewed"
        }

        self.segment_group_by_idx: dict[int, str] = {}
        self.segment_metrics: dict[int, SegmentMetrics] = {}
        if self._needs_segment_metrics():
            self.segment_metrics = self._load_segment_metrics()
        self._refresh_all_segment_groups()
        self.ecg_dataset = ds.dataset(ECG_PARQUET, format="parquet") if ECG_PARQUET.exists() else None
        self.peaks_dataset = ds.dataset(PEAKS_PARQUET, format="parquet") if PEAKS_PARQUET.exists() else None
        if getattr(args, "physio_review", False):
            self.physio_review_total = self._load_physio_review_index()

        self.index = max(0, int(args.start))
        if getattr(args, "marker_recovery", False):
            self.visible_segments = self._load_marker_recovery_queue()
            if not self.visible_segments:
                _die("No marker recovery rows found.")
            self.recovery_queue_mode = True
            self.index = min(self.index, len(self.visible_segments) - 1)
        elif getattr(args, "markers", False):
            self.visible_segments = self._load_marker_windows()
            if not self.visible_segments:
                _die("No marker windows found.")
            self._build_marker_navigation()
            start_marker_pos = self._find_next_marker_position(max(0, int(args.start)), 1)
            if start_marker_pos is None:
                _die("No unreviewed marker-center segments remain.")
            self.marker_position = start_marker_pos
            self.index = self._marker_center_index(self.marker_order[self.marker_position])
        elif getattr(args, "physio_review", False):
            if self.physio_review_total <= 0:
                _die("No physio review beats matched.")
            self.index = min(self.index, self.physio_review_total - 1)
            self.index = self._find_physio_review_index(self.index, 1, skip_reviewed=True)
            if self.index is None:
                _die("No unreviewed physio review beats remain.")
            self.visible_segments = [self._physio_review_row_at(self.index)]
        else:
            self.visible_segments = self._select_segments()
            if not self.visible_segments:
                _die("No segments matched the requested mode.")
            self.index = min(self.index, len(self.visible_segments) - 1)
        self.view_lo_ms: int | None = None
        self.view_hi_ms: int | None = None
        self.loaded_lo_ms: int | None = None
        self.loaded_hi_ms: int | None = None
        self.ecg_ms: list[int] = []
        self.ecg_y: list[float] = []
        self.current_peak_rows: list[dict[str, Any]] = []
        self.display_points: list[PeakPoint] = []
        self.window_cache: dict[tuple[int, int, int], tuple[list[int], list[float], list[dict[str, Any]]]] = {}
        self.window_cache_order: list[tuple[int, int, int]] = []

        self.undo_stack: list[dict[str, Any]] = []
        self.redo_stack: list[dict[str, Any]] = []
        self.mode = "browse"
        self.bad_region_start_ms: int | None = None
        self.physio_region_start_ms: int | None = None
        self.last_b_time = 0.0
        self.last_p_time = 0.0
        self.last_b_revert: tuple[int, str, bool, bool] | None = None
        self.comment_active = False
        self.comment_buffer = ""
        self.status = ""
        self.precise_add_mode = False
        self.back_stack: list[int] = []
        self.forward_stack: list[int] = []
        self.action_logger = ActionLogger(
            ACTION_LOG,
            {
                "script": str(Path(__file__)),
                "queue": self._requested_queue_group() or "default",
                "physio_review": bool(getattr(self.args, "physio_review", False)),
                "selected_segments": self._queue_size(),
                "log_path": str(ACTION_LOG),
            },
        )
        self._print_queue_summary()

        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.fig.patch.set_facecolor(PANEL)
        self.ax.set_facecolor(BG)
        self.fig.subplots_adjust(left=0.055, right=0.99, bottom=0.08, top=0.84)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self._load_window(reset_view=True)
        self._skip_empty_windows(direction=1)
        if not self.visible_segments:
            _die(self.status or "No non-empty ECG windows matched the requested mode. Use --show-empty to inspect empty rows.")
        self._draw()

    def _load_physio_review_index(self) -> int:
        if not getattr(self.args, "physio_review", False):
            return 0
        path = Path(PHYSIO_CONSTRAINT_REVIEW_PEAKS_PQT)
        if not path.exists():
            _die(f"Missing physio review peaks parquet: {path}")

        table = pq.read_table(
            path,
            columns=["segment_idx", "peak_id", "timestamp_ms", "assigned_label"],
        ).combine_chunks()
        original_total = table.num_rows
        february_skipped = 0
        timestamp = table.column("timestamp_ms")
        keep = pc.or_(
            pc.less(timestamp, FEBRUARY_2025_START_MS),
            pc.greater_equal(timestamp, MARCH_2025_START_MS),
        )
        table = table.filter(keep).combine_chunks()
        february_skipped = original_total - table.num_rows
        total = table.num_rows
        if total == 0:
            _die(f"No review beats found in physio review peaks parquet: {path}")
        if bool(getattr(self.args, "spread_queue", True)):
            table = self._spread_physio_review_table(table)
            total = table.num_rows
        self.physio_review_table = table
        suffix = f" ({february_skipped:,} February 2025 rows excluded)" if february_skipped else ""
        print(f"Loaded physio review index: {total:,} beats from {path}{suffix}", flush=True)
        return total

    def _load_marker_windows(self) -> list[dict[str, Any]]:
        timestamps = load_marker_timestamps(Path(MARKER_CSV))
        if not timestamps:
            return []
        marker_values = ",\n".join(
            f"({idx}, {int(ts)})" for idx, ts in enumerate(timestamps, start=1)
        )
        processed_rows = duck_json(
            f"""
            WITH markers(marker_idx, marker_timestamp_ms) AS (
                VALUES {marker_values}
            ),
            segments AS (
                SELECT
                    ROW_NUMBER() OVER (ORDER BY start_timestamp_ms, segment_idx) - 1 AS rn,
                    CAST(segment_idx AS INTEGER) AS segment_idx,
                    CAST(start_timestamp_ms AS BIGINT) AS start_timestamp_ms,
                    CAST(end_timestamp_ms AS BIGINT) AS end_timestamp_ms,
                    CAST(review_status AS VARCHAR) AS review_status,
                    CAST(quality_label AS VARCHAR) AS quality_label,
                    CAST(beat_training_eligible_segment AS BOOLEAN) AS beat_training_eligible_segment,
                    CAST(segment_quality_training_eligible AS BOOLEAN) AS segment_quality_training_eligible
                FROM read_parquet('{q(PROCESSED_SEGMENTS_PARQUET)}')
            ),
            marker_centers AS (
                SELECT
                    m.marker_idx,
                    m.marker_timestamp_ms,
                    s.rn AS center_rn,
                    s.segment_idx AS center_segment_idx
                FROM markers m
                JOIN LATERAL (
                    SELECT rn, segment_idx
                    FROM segments s
                    ORDER BY
                        CASE
                            WHEN m.marker_timestamp_ms BETWEEN s.start_timestamp_ms AND s.end_timestamp_ms THEN 0
                            ELSE 1
                        END,
                        ABS(((s.start_timestamp_ms + s.end_timestamp_ms) / 2) - m.marker_timestamp_ms)
                    LIMIT 1
                ) s ON TRUE
            ),
            marker_segments AS (
                SELECT
                    c.marker_idx,
                    c.marker_timestamp_ms,
                    c.center_segment_idx,
                    CAST(s.rn - c.center_rn AS INTEGER) AS marker_offset,
                    s.*
                FROM marker_centers c
                JOIN segments s
                    ON s.rn BETWEEN c.center_rn - {MARKER_FLANK_SEGMENTS}
                        AND c.center_rn + {MARKER_FLANK_SEGMENTS}
            )
            SELECT *
            FROM marker_segments
            ORDER BY marker_idx, marker_offset
            """
        )
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        skipped_reviewed = 0
        for raw in processed_rows:
            seg_idx = int(raw["segment_idx"])
            marker_idx = int(raw["marker_idx"])
            key = (marker_idx, seg_idx)
            if key in seen:
                continue
            seen.add(key)
            row = self.segment_by_idx.get(seg_idx)
            if row is None:
                row = normalize_processed_segment_row(raw)
            else:
                row = dict(row)
            if str(row.get("review_status") or "") == "reviewed":
                skipped_reviewed += 1
                continue
            row["_marker_idx"] = marker_idx
            row["_marker_timestamp_ms"] = int(raw["marker_timestamp_ms"])
            row["_marker_center_segment_idx"] = int(raw["center_segment_idx"])
            row["_marker_offset"] = int(raw["marker_offset"])
            rows.append(row)
        suffix = f" ({skipped_reviewed:,} already-reviewed segments excluded)" if skipped_reviewed else ""
        print(
            f"Loaded marker segments: {len(rows):,} rows from {len(timestamps):,} marker(s) "
            f"(±{MARKER_FLANK_SEGMENTS} segments each) via {MARKER_CSV}{suffix}",
            flush=True,
        )
        return rows

    def _load_marker_recovery_queue(self) -> list[dict[str, Any]]:
        path = Path(MARKER_REVIEW_RECOVERY_PQT)
        if not path.exists():
            _die(f"Missing marker recovery parquet: {path}")
        rows = load_parquet(path)
        out: list[dict[str, Any]] = []
        for raw in rows:
            seg_idx = int(raw["segment_idx"])
            row = self.segment_by_idx.get(seg_idx)
            if row is None:
                row = {col: raw.get(col) for col in SEGMENT_COLUMNS}
            else:
                row = dict(row)
            if str(row.get("review_status") or "") == "reviewed":
                continue
            if str(row.get("review_status") or "") == "revisit" or bool_value(row.get("in_revisit_pile")):
                continue
            for key in (
                "_marker_idx",
                "_marker_timestamp_ms",
                "_marker_center_segment_idx",
                "_marker_offset",
                "_direct_action_count",
                "_direct_actions",
            ):
                row[key] = raw.get(key)
            out.append(row)
        print(f"Loaded marker recovery queue: {len(out):,} rows from {path}", flush=True)
        return out

    def _build_marker_navigation(self) -> None:
        self.marker_groups = {}
        for idx, row in enumerate(self.visible_segments):
            marker_idx = int(row.get("_marker_idx") or 0)
            if marker_idx <= 0:
                continue
            self.marker_groups.setdefault(marker_idx, []).append(idx)
        for indices in self.marker_groups.values():
            indices.sort(key=lambda i: int_or_none(self.visible_segments[i].get("_marker_offset")) or 0)
        self.marker_order = sorted(self.marker_groups)

    def _marker_center_index(self, marker_idx: int) -> int:
        indices = self.marker_groups.get(int(marker_idx), [])
        if not indices:
            return 0
        return min(
            indices,
            key=lambda i: abs(int_or_none(self.visible_segments[i].get("_marker_offset")) or 0),
        )

    def _marker_center_row(self, marker_idx: int) -> dict[str, Any] | None:
        if int(marker_idx) not in self.marker_groups:
            return None
        return self.visible_segments[self._marker_center_index(int(marker_idx))]

    def _marker_center_done(self, marker_idx: int) -> bool:
        row = self._marker_center_row(marker_idx)
        if row is None:
            return True
        seg_idx = int(row["segment_idx"])
        live = self.segment_by_idx.get(seg_idx, row)
        return str(live.get("review_status")) == "reviewed" or bool_value(live.get("in_revisit_pile")) or str(live.get("review_status")) == "revisit"

    def _find_next_marker_position(self, start_pos: int, direction: int) -> int | None:
        if not self.marker_order:
            return None
        step = 1 if direction >= 0 else -1
        pos = max(0, min(len(self.marker_order) - 1, int(start_pos)))
        while 0 <= pos < len(self.marker_order):
            if not self._marker_center_done(self.marker_order[pos]):
                return pos
            pos += step
        return None

    def _spread_physio_review_table(self, table: Any) -> Any:
        total = int(table.num_rows)
        if total <= 2:
            return table
        step = max(total // 97, 1)
        while math.gcd(step, total) != 1:
            step += 1
        order = [(idx * step) % total for idx in range(total)]
        return table.take(pa.array(order, type=pa.int64())).combine_chunks()

    def _physio_review_segment_idx_at_index(self, idx: int) -> int:
        if self.physio_review_table is None:
            raise RuntimeError("physio review index has not been loaded")
        return int(self.physio_review_table.column("segment_idx")[idx].as_py())

    def _find_physio_review_index(self, start_idx: int, direction: int, skip_reviewed: bool = False) -> int | None:
        if self.physio_review_total <= 0:
            return None
        step = 1 if direction >= 0 else -1
        idx = max(0, min(self.physio_review_total - 1, int(start_idx)))
        while 0 <= idx < self.physio_review_total:
            seg_idx = self._physio_review_segment_idx_at_index(idx)
            if not skip_reviewed or seg_idx not in self.physio_reviewed_segment_idxs:
                return idx
            idx += step
        return None

    def _load_processed_segment_row(self, seg_idx: int) -> dict[str, Any] | None:
        cached = self.physio_review_segment_cache.get(seg_idx)
        if cached is not None:
            return cached
        if not PROCESSED_SEGMENTS_PARQUET.exists():
            return None
        rows = duck_json(
            f"""
            SELECT
                CAST(segment_idx AS INTEGER) AS segment_idx,
                CAST(start_timestamp_ms AS BIGINT) AS start_timestamp_ms,
                CAST(end_timestamp_ms AS BIGINT) AS end_timestamp_ms,
                CAST(review_status AS VARCHAR) AS review_status,
                CAST(quality_label AS VARCHAR) AS quality_label,
                CAST(beat_training_eligible_segment AS BOOLEAN) AS beat_training_eligible_segment,
                CAST(segment_quality_training_eligible AS BOOLEAN) AS segment_quality_training_eligible
            FROM read_parquet('{q(PROCESSED_SEGMENTS_PARQUET)}')
            WHERE segment_idx = {int(seg_idx)}
            LIMIT 1
            """
        )
        if not rows:
            return None
        row = normalize_processed_segment_row(rows[0])
        self.physio_review_segment_cache[seg_idx] = row
        return row

    def _physio_review_row_at(self, idx: int) -> dict[str, Any]:
        cached = self.physio_review_row_cache.get(idx)
        if cached is not None:
            return cached
        if self.physio_review_table is None:
            raise RuntimeError("physio review index has not been loaded")
        table = self.physio_review_table
        seg_idx = int(table.column("segment_idx")[idx].as_py())
        base = self.segment_by_idx.get(seg_idx) or self._load_processed_segment_row(seg_idx)
        if base is None:
            base = {
                "segment_idx": seg_idx,
                "start_ms": int(table.column("timestamp_ms")[idx].as_py()) - DISPLAY_MS // 2,
                "end_ms": int(table.column("timestamp_ms")[idx].as_py()) + DISPLAY_MS // 2,
                "review_status": "unreviewed",
                "segment_quality_label": "unknown",
                "in_revisit_pile": False,
                "has_bad_region": False,
                "bad_region_count": 0,
                "beat_training_eligible_segment": False,
                "segment_quality_training_eligible": False,
                "comment": "",
            }
        row = dict(base)
        row["_review_peak_id"] = int(table.column("peak_id")[idx].as_py())
        row["_review_peak_timestamp_ms"] = int(table.column("timestamp_ms")[idx].as_py())
        row["_review_assigned_label"] = str(table.column("assigned_label")[idx].as_py() or "")
        self.physio_review_row_cache[idx] = row
        if len(self.physio_review_row_cache) > WINDOW_CACHE_LIMIT * 4:
            for old_idx in sorted(self.physio_review_row_cache)[:WINDOW_CACHE_LIMIT]:
                self.physio_review_row_cache.pop(old_idx, None)
        return row

    def _load_processed_segment_index(self) -> dict[int, dict[str, Any]]:
        if not PROCESSED_SEGMENTS_PARQUET.exists():
            return {}
        rows = duck_json(
            f"""
            SELECT
                CAST(segment_idx AS INTEGER) AS segment_idx,
                CAST(start_timestamp_ms AS BIGINT) AS start_timestamp_ms,
                CAST(end_timestamp_ms AS BIGINT) AS end_timestamp_ms,
                CAST(review_status AS VARCHAR) AS review_status,
                CAST(quality_label AS VARCHAR) AS quality_label,
                CAST(beat_training_eligible_segment AS BOOLEAN) AS beat_training_eligible_segment,
                CAST(segment_quality_training_eligible AS BOOLEAN) AS segment_quality_training_eligible
            FROM read_parquet('{q(PROCESSED_SEGMENTS_PARQUET)}')
            ORDER BY start_timestamp_ms
            """
        )
        return {
            int(row["segment_idx"]): normalize_processed_segment_row(row)
            for row in rows
        }

    def _load_segment_metrics(self) -> dict[int, SegmentMetrics]:
        if not PEAKS_PARQUET.exists():
            return {}
        segments_sql = f"SELECT segment_idx, start_ms, end_ms FROM read_parquet('{q(OUT_SEGMENTS)}')"
        rows = duck_json(
            f"""
            WITH segments AS (
                {segments_sql}
            ),
            assigned_peaks AS (
                SELECT
                    s.segment_idx,
                    p.timestamp_ms
                FROM (
                    SELECT CAST(timestamp_ms AS BIGINT) AS timestamp_ms
                    FROM read_parquet('{q(PEAKS_PARQUET)}')
                    WHERE timestamp_ms BETWEEN (SELECT MIN(start_ms) FROM segments)
                        AND (SELECT MAX(end_ms) FROM segments)
                ) p
                ASOF JOIN segments s
                    ON p.timestamp_ms >= s.start_ms
                WHERE p.timestamp_ms <= s.end_ms
            ),
            ordered_peaks AS (
                SELECT
                    CAST(segment_idx AS INTEGER) AS segment_idx,
                    CAST(timestamp_ms AS BIGINT) AS timestamp_ms,
                    CAST(timestamp_ms AS BIGINT)
                        - LAG(CAST(timestamp_ms AS BIGINT)) OVER (
                            PARTITION BY CAST(segment_idx AS INTEGER)
                            ORDER BY CAST(timestamp_ms AS BIGINT)
                        ) AS rr_ms
                FROM assigned_peaks
            ),
            rr AS (
                SELECT
                    segment_idx,
                    timestamp_ms,
                    rr_ms,
                    LAG(rr_ms) OVER (
                        PARTITION BY segment_idx
                        ORDER BY timestamp_ms
                    ) AS prev_rr_ms,
                    rr_ms - LAG(rr_ms) OVER (
                        PARTITION BY segment_idx
                        ORDER BY timestamp_ms
                    ) AS delta_rr_ms
                FROM ordered_peaks
                WHERE rr_ms IS NOT NULL
            )
            SELECT
                segment_idx,
                COUNT(rr_ms) AS rr_count,
                MIN(rr_ms) AS min_rr_ms,
                COALESCE(SQRT(AVG(CASE WHEN delta_rr_ms IS NULL THEN NULL ELSE delta_rr_ms * delta_rr_ms END)), 0) AS rmssd_ms,
                COALESCE(MAX(delta_rr_ms), 0) AS max_rr_increase_ms,
                COALESCE(MAX(CASE
                    WHEN prev_rr_ms IS NULL OR prev_rr_ms <= 0 OR delta_rr_ms IS NULL THEN 0
                    ELSE ABS(delta_rr_ms::DOUBLE / prev_rr_ms::DOUBLE)
                END), 0) AS max_rr_change_pct,
                COALESCE(MAX(CASE
                    WHEN prev_rr_ms > 0 AND delta_rr_ms < 0
                        AND ABS(delta_rr_ms::DOUBLE / prev_rr_ms::DOUBLE) > 0.20 THEN 1
                    ELSE 0
                END), 0) AS has_rr_drop_gt_20,
                COALESCE(MAX(CASE
                    WHEN prev_rr_ms > 0 AND delta_rr_ms > 0 AND delta_rr_ms < {int(self.args.delta)}
                        AND delta_rr_ms::DOUBLE / prev_rr_ms::DOUBLE > 0.20 THEN 1
                    ELSE 0
                END), 0) AS has_rr_rise_gt_20_nonphysio,
                COALESCE(MAX(CASE
                    WHEN delta_rr_ms >= {int(self.args.delta)} THEN 1
                    ELSE 0
                END), 0) AS has_physio_rise
            FROM rr
            GROUP BY segment_idx
            """
        )
        metrics: dict[int, SegmentMetrics] = {}
        for row in rows:
            seg_idx = int(row["segment_idx"])
            metrics[seg_idx] = SegmentMetrics(
                rr_count=int(row.get("rr_count") or 0),
                rmssd_ms=float(row.get("rmssd_ms") or 0.0),
                min_rr_ms=int(row["min_rr_ms"]) if row.get("min_rr_ms") is not None else None,
                max_rr_increase_ms=(
                    int(row["max_rr_increase_ms"]) if row.get("max_rr_increase_ms") is not None else None
                ),
                max_rr_change_pct=float(row.get("max_rr_change_pct") or 0.0),
                has_rr_drop_gt_20=bool_value(row.get("has_rr_drop_gt_20")),
                has_rr_rise_gt_20_nonphysio=bool_value(row.get("has_rr_rise_gt_20_nonphysio")),
                has_physio_rise=bool_value(row.get("has_physio_rise")),
            )
        return metrics

    def _segment_has_annotations(self, seg_idx: int) -> bool:
        if self.bad_by_segment.get(seg_idx):
            return True
        if str(self.segment_by_idx.get(seg_idx, {}).get("comment") or "").strip():
            return True
        for row in self.beats_by_segment.get(seg_idx, []):
            label = str(row.get("label") or "clean")
            subtype = str(row.get("subtype") or "")
            if label != "clean" or bool_value(row.get("is_added_peak")) or subtype not in {"", "auto"}:
                return True
        return False

    def _refresh_segment_group(self, seg_idx: int) -> None:
        if not self.segment_metrics:
            self.segment_group_by_idx[seg_idx] = (
                "annotated" if self._segment_has_annotations(seg_idx) else "likely-clean"
            )
            return
        self.segment_group_by_idx[seg_idx] = classify_segment_queue(
            self._segment_has_annotations(seg_idx),
            self.segment_metrics.get(seg_idx),
            rmssd_artifact_ms=float(self.args.rmssd),
            min_rr_artifact_ms=int(self.args.min_rr),
            physio_delta_ms=int(self.args.delta),
        )

    def _refresh_all_segment_groups(self) -> None:
        for row in self.segments:
            self._refresh_segment_group(int(row["segment_idx"]))

    def _requested_queue_group(self) -> str | None:
        for flag, group in (
            ("annotated", "annotated"),
            ("likely_artifact", "likely-artifact"),
            ("likely_physio", "likely-physio"),
            ("likely_clean", "likely-clean"),
        ):
            if bool(getattr(self.args, flag, False)):
                return group
        return None

    def _needs_segment_metrics(self) -> bool:
        return self._requested_queue_group() in {"likely-artifact", "likely-physio", "likely-clean"}

    def _base_selected_segments(self) -> list[dict[str, Any]]:
        rows = sorted(self.segments, key=lambda r: int(r["start_ms"]))
        rows = [r for r in rows if not is_february_2025_ms(int(r["start_ms"]))]
        if self.args.reviewed:
            return [r for r in rows if str(r.get("review_status")) == "reviewed"]
        if self.args.usable:
            return [r for r in rows if str(r.get("segment_quality_label")) == "usable"]
        if self.args.unclean:
            return [r for r in rows if str(r.get("segment_quality_label")) == "unclean"]
        if self.args.unusable:
            return [r for r in rows if str(r.get("segment_quality_label")) == "unusable"]
        if self.args.physio:
            physio_segment_ids = {
                int(b["segment_idx"]) for b in self.beats
                if str(b.get("label")) == "physio"
            }
            return [r for r in rows if int(r["segment_idx"]) in physio_segment_ids]
        return rows

    def _spread_temporal_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not bool(getattr(self.args, "spread_queue", True)) or len(rows) <= 2:
            return rows
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in sorted(rows, key=lambda r: int(r["start_ms"])):
            key = datetime.fromtimestamp(int(row["start_ms"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            buckets.setdefault(key, []).append(row)
        ordered_keys = sorted(buckets)
        if len(ordered_keys) <= 1:
            return rows
        out: list[dict[str, Any]] = []
        remaining = True
        offset = 0
        while remaining:
            remaining = False
            for key in ordered_keys:
                bucket = buckets[key]
                if offset < len(bucket):
                    out.append(bucket[offset])
                    remaining = True
            offset += 1
        return out

    def _sort_segments_for_group(self, rows: list[dict[str, Any]], group: str | None) -> list[dict[str, Any]]:
        if group == "likely-clean":
            return sorted(
                rows,
                key=lambda r: (
                    self.segment_metrics.get(int(r["segment_idx"]), SegmentMetrics()).rmssd_ms,
                    int(r["start_ms"]),
                ),
            )
        if group == "likely-artifact":
            return sorted(
                rows,
                key=lambda r: (
                    self.segment_metrics.get(int(r["segment_idx"]), SegmentMetrics()).min_rr_ms
                    if self.segment_metrics.get(int(r["segment_idx"]), SegmentMetrics()).min_rr_ms is not None
                    else 1_000_000,
                    -self.segment_metrics.get(int(r["segment_idx"]), SegmentMetrics()).rmssd_ms,
                    int(r["start_ms"]),
                ),
            )
        if group == "likely-physio":
            return sorted(
                rows,
                key=lambda r: (
                    -(self.segment_metrics.get(int(r["segment_idx"]), SegmentMetrics()).max_rr_increase_ms or 0),
                    int(r["start_ms"]),
                ),
            )
        return rows

    def _select_segments(self) -> list[dict[str, Any]]:
        if getattr(self.args, "marker_recovery", False):
            rows = []
            for row in self.visible_segments:
                live = self.segment_by_idx.get(int(row["segment_idx"]), row)
                if str(live.get("review_status") or "") == "reviewed":
                    continue
                if str(live.get("review_status") or "") == "revisit" or bool_value(live.get("in_revisit_pile")):
                    continue
                rows.append(row)
            return rows
        if getattr(self.args, "markers", False):
            return list(self.visible_segments)
        if getattr(self.args, "physio_review", False):
            return [self._physio_review_row_at(self.index)]
        rows = self._base_selected_segments()
        group = self._requested_queue_group()
        if group is None:
            return self._spread_temporal_rows(rows)
        filtered = [r for r in rows if self.segment_group_by_idx.get(int(r["segment_idx"])) == group]
        return self._spread_temporal_rows(self._sort_segments_for_group(filtered, group))

    def _queue_counts(self) -> dict[str, int]:
        counts = {"annotated": 0, "likely-artifact": 0, "likely-physio": 0, "likely-clean": 0}
        for row in self._base_selected_segments():
            group = self.segment_group_by_idx.get(int(row["segment_idx"]), "likely-clean")
            counts[group] = counts.get(group, 0) + 1
        return counts

    def _print_queue_summary(self) -> None:
        if getattr(self.args, "physio_review", False):
            summary = f"Physio review queue: {self.physio_review_total:,} flagged beats"
            print(summary, flush=True)
            self.action_logger.log("physio_review_queue_summary", details={"beats": self.physio_review_total})
            return
        if not self.segment_metrics and self._requested_queue_group() is None:
            summary = f"Selected queue: {self._queue_size():,} segments"
            print(summary, flush=True)
            self.action_logger.log("queue_summary", details={"selected_segments": self._queue_size()})
            return
        counts = self._queue_counts()
        summary = (
            "Queue groups for current base selection: "
            f"--annotated {counts.get('annotated', 0)} | "
            f"--likely-artifact {counts.get('likely-artifact', 0)} | "
            f"--likely-physio {counts.get('likely-physio', 0)} | "
            f"--likely-clean {counts.get('likely-clean', 0)}"
        )
        print(summary, flush=True)
        self.action_logger.log("queue_summary", details=counts)

    def _queue_size(self) -> int:
        if getattr(self.args, "physio_review", False):
            return self.physio_review_total
        if getattr(self.args, "markers", False):
            return len(self.marker_order)
        return len(self.visible_segments)

    def _display_review_status(self, seg: dict[str, Any]) -> str:
        if getattr(self.args, "physio_review", False):
            row = self.segment_by_idx.get(int(seg["segment_idx"]), seg)
            return f"physio-{row.get('review_status') or 'unreviewed'}"
        return str(seg.get("review_status") or "unreviewed")

    def current_segment(self) -> dict[str, Any]:
        if getattr(self.args, "physio_review", False):
            return self._physio_review_row_at(self.index)
        return self.visible_segments[self.index]

    def _segment_idx(self) -> int:
        return int(self.current_segment()["segment_idx"])

    def _real_segment_idx_for_timestamp(self, timestamp_ms: int) -> int:
        cached = self.timestamp_segment_cache.get(int(timestamp_ms))
        if cached is not None:
            return cached
        if not PROCESSED_SEGMENTS_PARQUET.exists():
            return self._segment_idx()
        rows = duck_json(
            f"""
            SELECT CAST(segment_idx AS INTEGER) AS segment_idx
            FROM read_parquet('{q(PROCESSED_SEGMENTS_PARQUET)}')
            WHERE CAST(start_timestamp_ms AS BIGINT) <= {int(timestamp_ms)}
                AND CAST(end_timestamp_ms AS BIGINT) >= {int(timestamp_ms)}
            LIMIT 1
            """
        )
        if rows:
            seg_idx = int(rows[0]["segment_idx"])
        else:
            seg_idx = self._segment_idx()
        self.timestamp_segment_cache[int(timestamp_ms)] = seg_idx
        return seg_idx

    def _point_segment_idx(self, point: PeakPoint) -> int:
        if point.segment_idx is not None:
            return int(point.segment_idx)
        if getattr(self.args, "markers", False) or getattr(self.args, "marker_recovery", False):
            return self._real_segment_idx_for_timestamp(self._point_identity_ts(point))
        return self._segment_idx()

    def _load_window(self, reset_view: bool = False) -> None:
        seg = self.current_segment()
        seg_idx = int(seg["segment_idx"])
        start_ms = int(seg["start_ms"])
        end_ms = int(seg["end_ms"])
        load_start_ms, load_end_ms = self._window_bounds(seg, start_ms, end_ms)
        self.loaded_lo_ms = load_start_ms
        self.loaded_hi_ms = load_end_ms
        cache_key = (seg_idx, load_start_ms, load_end_ms)
        cached = self.window_cache.get(cache_key)
        if cached is not None:
            self.ecg_ms = list(cached[0])
            self.ecg_y = list(cached[1])
            self.current_peak_rows = [dict(r) for r in cached[2]]
            if reset_view or self.view_lo_ms is None or self.view_hi_ms is None:
                self._set_review_or_segment_view(seg, start_ms, end_ms)
            return

        self.ecg_ms, self.ecg_y = self._load_ecg(load_start_ms, load_end_ms)
        self.current_peak_rows = self._load_current_peaks(load_start_ms, load_end_ms)
        if not self.ecg_ms:
            fallback_ecg_ms, fallback_ecg_y = self._load_ecg_by_segment(seg_idx)
            if fallback_ecg_ms:
                self.ecg_ms, self.ecg_y = fallback_ecg_ms, fallback_ecg_y
                self.current_peak_rows = self._load_current_peaks_by_segment(seg_idx)
                start_ms = min(self.ecg_ms)
                end_ms = max(self.ecg_ms)
                seg["start_ms"] = start_ms
                seg["end_ms"] = end_ms
                if seg_idx in self.segment_by_idx:
                    self.segment_by_idx[seg_idx]["start_ms"] = start_ms
                    self.segment_by_idx[seg_idx]["end_ms"] = end_ms
                if not self.status:
                    self.status = f"Loaded Segment {seg_idx} by current segment_idx timestamps"
        if reset_view or self.view_lo_ms is None or self.view_hi_ms is None:
            self._set_review_or_segment_view(seg, start_ms, end_ms)
        self._cache_window(cache_key)

    def _set_review_or_segment_view(self, seg: dict[str, Any], start_ms: int, end_ms: int) -> None:
        self.view_lo_ms, self.view_hi_ms = self._window_bounds(seg, start_ms, end_ms)

    def _requested_window_ms(self, seg: dict[str, Any], start_ms: int, end_ms: int) -> int:
        requested = getattr(self.args, "window_size", None)
        if requested is not None:
            return max(MIN_VIEW_MS, int(round(float(requested) * 1000)))
        review_ts = int_or_none(seg.get("_review_peak_timestamp_ms"))
        if review_ts is not None:
            return max(DEFAULT_REVIEW_WINDOW_MS, MIN_VIEW_MS)
        return max(end_ms - start_ms, MIN_VIEW_MS)

    def _window_center_ms(self, seg: dict[str, Any], start_ms: int, end_ms: int) -> int:
        review_ts = int_or_none(seg.get("_review_peak_timestamp_ms"))
        if review_ts is not None:
            return review_ts
        if getattr(self.args, "markers", False) or getattr(self.args, "marker_recovery", False):
            return (start_ms + end_ms) // 2
        marker_ts = int_or_none(seg.get("_marker_timestamp_ms"))
        marker_offset = int_or_none(seg.get("_marker_offset"))
        if marker_ts is not None and marker_offset == 0:
            return marker_ts
        return (start_ms + end_ms) // 2

    def _window_bounds(self, seg: dict[str, Any], start_ms: int, end_ms: int) -> tuple[int, int]:
        width_ms = self._requested_window_ms(seg, start_ms, end_ms)
        center_ms = self._window_center_ms(seg, start_ms, end_ms)
        lo = center_ms - width_ms // 2
        hi = lo + width_ms
        return int(lo), int(hi)

    def _cache_window(self, cache_key: tuple[int, int, int]) -> None:
        self.window_cache[cache_key] = (
            list(self.ecg_ms),
            list(self.ecg_y),
            [dict(r) for r in self.current_peak_rows],
        )
        if cache_key in self.window_cache_order:
            self.window_cache_order.remove(cache_key)
        self.window_cache_order.append(cache_key)
        while len(self.window_cache_order) > WINDOW_CACHE_LIMIT:
            old = self.window_cache_order.pop(0)
            self.window_cache.pop(old, None)

    def _skip_empty_windows(self, direction: int) -> None:
        if getattr(self.args, "physio_review", False):
            return
        if not self.visible_segments:
            return
        step = 1 if direction >= 0 else -1
        while not self.ecg_ms and self.visible_segments:
            skipped = int(self.current_segment()["segment_idx"])
            self.visible_segments.pop(self.index)
            if not self.visible_segments:
                self.status = (
                    f"No ECG rows in remaining queue; last skipped Segment {skipped}. "
                    "Use --show-empty to inspect empty rows."
                )
                return
            if step < 0:
                self.index = max(0, self.index - 1)
            else:
                self.index = min(self.index, len(self.visible_segments) - 1)
            self._load_window(reset_view=True)
            self.status = f"Skipped empty Segment {skipped}"

    def _load_ecg(self, start_ms: int, end_ms: int) -> tuple[list[int], list[float]]:
        if self.ecg_dataset is None:
            return [], []
        table = self.ecg_dataset.to_table(
            columns=["timestamp_ms", "ecg"],
            filter=(ds.field("timestamp_ms") >= int(start_ms)) & (ds.field("timestamp_ms") <= int(end_ms)),
        )
        if table.num_rows == 0:
            return [], []
        table = table.take(pc.sort_indices(table, sort_keys=[("timestamp_ms", "ascending")])).combine_chunks()
        return (
            [int(v.as_py()) for v in table.column("timestamp_ms")],
            [float(v.as_py()) for v in table.column("ecg")],
        )

    def _load_ecg_by_segment(self, segment_idx: int) -> tuple[list[int], list[float]]:
        if self.ecg_dataset is None:
            return [], []
        table = self.ecg_dataset.to_table(
            columns=["timestamp_ms", "ecg"],
            filter=ds.field("segment_idx") == int(segment_idx),
        )
        if table.num_rows == 0:
            return [], []
        table = table.take(pc.sort_indices(table, sort_keys=[("timestamp_ms", "ascending")])).combine_chunks()
        return (
            [int(v.as_py()) for v in table.column("timestamp_ms")],
            [float(v.as_py()) for v in table.column("ecg")],
        )

    def _load_current_peaks(self, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        if self.peaks_dataset is None:
            return []
        table = self.peaks_dataset.to_table(
            columns=["peak_id", "timestamp_ms", "source", "segment_idx"],
            filter=(ds.field("timestamp_ms") >= int(start_ms)) & (ds.field("timestamp_ms") <= int(end_ms)),
        )
        if table.num_rows == 0:
            return []
        table = table.take(pc.sort_indices(table, sort_keys=[("timestamp_ms", "ascending")])).combine_chunks()
        rows = table.to_pylist()
        for row in rows:
            row["is_added_peak"] = False
        return rows

    def _load_current_peaks_by_segment(self, segment_idx: int) -> list[dict[str, Any]]:
        if self.peaks_dataset is None:
            return []
        table = self.peaks_dataset.to_table(
            columns=["peak_id", "timestamp_ms", "source", "segment_idx"],
            filter=ds.field("segment_idx") == int(segment_idx),
        )
        if table.num_rows == 0:
            return []
        table = table.take(pc.sort_indices(table, sort_keys=[("timestamp_ms", "ascending")])).combine_chunks()
        rows = table.to_pylist()
        for row in rows:
            row["is_added_peak"] = False
        return rows

    def _ecg_at(self, timestamp_ms: int) -> float:
        if not self.ecg_ms:
            return 0.0
        pos = bisect.bisect_left(self.ecg_ms, timestamp_ms)
        if pos <= 0:
            return self.ecg_y[0]
        if pos >= len(self.ecg_ms):
            return self.ecg_y[-1]
        before = self.ecg_ms[pos - 1]
        after = self.ecg_ms[pos]
        if abs(timestamp_ms - before) <= abs(after - timestamp_ms):
            return self.ecg_y[pos - 1]
        return self.ecg_y[pos]

    def _local_max_point(self, timestamp_ms: int, window_ms: int = DISPLAY_PEAK_SNAP_WINDOW_MS) -> tuple[int, float]:
        if not self.ecg_ms:
            return timestamp_ms, 0.0
        lo = bisect.bisect_left(self.ecg_ms, timestamp_ms - window_ms)
        hi = bisect.bisect_right(self.ecg_ms, timestamp_ms + window_ms)
        if lo >= hi:
            return timestamp_ms, self._ecg_at(timestamp_ms)
        best = max(range(lo, hi), key=lambda idx: self.ecg_y[idx])
        return int(self.ecg_ms[best]), float(self.ecg_y[best])

    def _annotation_for_timestamp(self, seg_idx: int, timestamp_ms: int, tol_ms: int = ANNOTATION_MATCH_TOLERANCE_MS) -> dict[str, Any] | None:
        rows = self.beats_by_segment.get(seg_idx, [])
        best: dict[str, Any] | None = None
        best_dist = tol_ms + 1
        for row in rows:
            dist = abs(int(row["timestamp_ms"]) - timestamp_ms)
            if dist <= tol_ms and dist < best_dist:
                best = row
                best_dist = dist
        return best

    def _all_segment_beat_rows(self, seg_idx: int) -> list[dict[str, Any]]:
        return self.beats_by_segment.setdefault(seg_idx, [])

    def _rebuild_added_beat_index(self) -> None:
        self.added_beats_by_timestamp = [
            (int(row["timestamp_ms"]), row)
            for row in self.beats
            if bool_value(row.get("is_added_peak"))
        ]
        self.added_beats_by_timestamp.sort(key=lambda item: item[0])

    def _visible_added_beat_rows(self, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
        if not getattr(self.args, "markers", False) and not getattr(self.args, "marker_recovery", False):
            return self._all_segment_beat_rows(self._segment_idx())
        lo = bisect.bisect_left(self.added_beats_by_timestamp, int(start_ms), key=lambda item: item[0])
        hi = bisect.bisect_right(self.added_beats_by_timestamp, int(end_ms), key=lambda item: item[0])
        return [row for _, row in self.added_beats_by_timestamp[lo:hi]]

    def _make_points(self) -> list[PeakPoint]:
        seg_idx = self._segment_idx()
        seg = self.current_segment()
        start_ms = self.view_lo_ms if self.view_lo_ms is not None else int(seg["start_ms"])
        end_ms = self.view_hi_ms if self.view_hi_ms is not None else int(seg["end_ms"])
        points: list[PeakPoint] = []
        seen_keys: set[tuple[int, str]] = set()

        for peak in self.current_peak_rows:
            ts = int(peak["timestamp_ms"])
            point_seg_idx = int_or_none(peak.get("segment_idx")) if (
                getattr(self.args, "markers", False) or getattr(self.args, "marker_recovery", False)
            ) else seg_idx
            if point_seg_idx is None:
                point_seg_idx = seg_idx
            ann = self._annotation_for_timestamp(point_seg_idx, ts)
            label = str(ann.get("label")) if ann else "clean"
            subtype = str(ann.get("subtype") or "") if ann else ""
            source = str(peak.get("source") or "called_peak")
            key = (ts, "current")
            seen_keys.add(key)
            display_ts, display_y = self._local_max_point(ts)
            points.append(
                PeakPoint(
                    peak_id=int(peak["peak_id"]) if peak.get("peak_id") is not None else None,
                    timestamp_ms=display_ts,
                    y=display_y,
                    source=source,
                    is_added_peak=bool_value(peak.get("is_added_peak")),
                    label=label,
                    subtype=subtype,
                    raw_timestamp_ms=ts,
                    segment_idx=point_seg_idx,
                )
            )

        for ann in self._visible_added_beat_rows(start_ms, end_ms):
            if not bool_value(ann.get("is_added_peak")):
                continue
            ts = int(ann["timestamp_ms"])
            key = (ts, "added")
            if key in seen_keys:
                continue
            display_ts, display_y = self._local_max_point(ts)
            points.append(
                PeakPoint(
                    peak_id=None,
                    timestamp_ms=display_ts,
                    y=display_y,
                    source="added_peak",
                    is_added_peak=True,
                    label=str(ann.get("label") or "clean"),
                    subtype=str(ann.get("subtype") or ""),
                    raw_timestamp_ms=ts,
                    segment_idx=int_or_none(ann.get("segment_idx")),
                )
            )

        points.sort(key=lambda p: p.timestamp_ms)
        return points

    def _current_segment_idx_or_none(self) -> int | None:
        if not self.visible_segments:
            return None
        try:
            return self._segment_idx()
        except Exception:
            return None

    def _snapshot(self) -> dict[str, Any]:
        return {
            "kind": "full",
            "segment_idx": self._current_segment_idx_or_none(),
            "segments": copy.deepcopy(self.segments),
            "beats": copy.deepcopy(self.beats),
            "bad_regions": copy.deepcopy(self.bad_regions),
        }

    def _segment_row_snapshot(self, seg_idx: int | None = None) -> dict[str, Any]:
        target_idx = int(seg_idx if seg_idx is not None else self._segment_idx())
        row = self.segment_by_idx.get(target_idx)
        return {
            "kind": "segment_row",
            "segment_idx": target_idx,
            "row": copy.deepcopy(row),
        }

    def _beat_segment_snapshot(self, seg_idx: int | None = None) -> dict[str, Any]:
        target_idx = int(seg_idx if seg_idx is not None else self._segment_idx())
        return {
            "kind": "beat_segment",
            "segment_idx": target_idx,
            "beat_rows": copy.deepcopy(self.beats_by_segment.get(target_idx, [])),
        }

    def _beat_segments_snapshot(self, seg_idxs: set[int] | list[int] | tuple[int, ...]) -> dict[str, Any]:
        target_idxs = sorted({int(seg_idx) for seg_idx in seg_idxs})
        return {
            "kind": "beat_segments",
            "segment_idxs": target_idxs,
            "beat_rows_by_segment": {
                int(seg_idx): copy.deepcopy(self.beats_by_segment.get(int(seg_idx), []))
                for seg_idx in target_idxs
            },
        }

    def _bad_segment_snapshot(self, seg_idx: int | None = None) -> dict[str, Any]:
        target_idx = int(seg_idx if seg_idx is not None else self._segment_idx())
        return {
            "kind": "bad_segment",
            "segment_idx": target_idx,
            "segment_row": copy.deepcopy(self.segment_by_idx.get(target_idx)),
            "bad_rows": copy.deepcopy(self.bad_by_segment.get(target_idx, [])),
        }

    def _snapshot_for_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        if entry.get("kind") == "segment_row":
            return self._segment_row_snapshot(int(entry["segment_idx"]))
        if entry.get("kind") == "beat_segment":
            return self._beat_segment_snapshot(int(entry["segment_idx"]))
        if entry.get("kind") == "beat_segments":
            return self._beat_segments_snapshot([int(v) for v in entry.get("segment_idxs", [])])
        if entry.get("kind") == "bad_segment":
            return self._bad_segment_snapshot(int(entry["segment_idx"]))
        return self._snapshot()

    def _restore(self, state: dict[str, Any]) -> None:
        target_idx = int_or_none(state.get("segment_idx"))
        if state.get("kind") == "segment_row":
            row = copy.deepcopy(state.get("row"))
            if row is None:
                if target_idx is not None:
                    self.segments = [r for r in self.segments if int(r["segment_idx"]) != target_idx]
            else:
                replaced = False
                for idx, existing in enumerate(self.segments):
                    if int(existing["segment_idx"]) == int(row["segment_idx"]):
                        self.segments[idx] = row
                        replaced = True
                        break
                if not replaced:
                    self.segments.append(row)
            self.segment_by_idx = {int(r["segment_idx"]): r for r in self.segments}
            if target_idx is not None:
                self._refresh_segment_group(target_idx)
        elif state.get("kind") == "beat_segment":
            if target_idx is not None:
                restored_rows = copy.deepcopy(state.get("beat_rows") or [])
                self.beats = [r for r in self.beats if int(r["segment_idx"]) != target_idx]
                self.beats.extend(restored_rows)
                if restored_rows:
                    self.beats_by_segment[target_idx] = restored_rows
                else:
                    self.beats_by_segment.pop(target_idx, None)
                self._rebuild_added_beat_index()
                self._refresh_segment_group(target_idx)
        elif state.get("kind") == "beat_segments":
            target_idxs = {int(v) for v in state.get("segment_idxs", [])}
            rows_by_segment = {
                int(k): copy.deepcopy(v)
                for k, v in (state.get("beat_rows_by_segment") or {}).items()
            }
            if target_idxs:
                self.beats = [r for r in self.beats if int(r["segment_idx"]) not in target_idxs]
                for seg_idx in sorted(target_idxs):
                    restored_rows = rows_by_segment.get(seg_idx, [])
                    self.beats.extend(restored_rows)
                    if restored_rows:
                        self.beats_by_segment[seg_idx] = restored_rows
                    else:
                        self.beats_by_segment.pop(seg_idx, None)
                    self._refresh_segment_group(seg_idx)
                self._rebuild_added_beat_index()
        elif state.get("kind") == "bad_segment":
            if target_idx is not None:
                restored_segment = copy.deepcopy(state.get("segment_row"))
                restored_bad = copy.deepcopy(state.get("bad_rows") or [])
                if restored_segment is None:
                    self.segments = [r for r in self.segments if int(r["segment_idx"]) != target_idx]
                    self.segment_by_idx.pop(target_idx, None)
                else:
                    replaced = False
                    for idx, existing in enumerate(self.segments):
                        if int(existing["segment_idx"]) == target_idx:
                            self.segments[idx] = restored_segment
                            replaced = True
                            break
                    if not replaced:
                        self.segments.append(restored_segment)
                    self.segment_by_idx[target_idx] = restored_segment
                self.bad_regions = [r for r in self.bad_regions if int(r["segment_idx"]) != target_idx]
                self.bad_regions.extend(restored_bad)
                if restored_bad:
                    self.bad_by_segment[target_idx] = restored_bad
                else:
                    self.bad_by_segment.pop(target_idx, None)
                self._refresh_segment_group(target_idx)
        else:
            self.segments = copy.deepcopy(state["segments"])
            self.beats = copy.deepcopy(state["beats"])
            self.bad_regions = copy.deepcopy(state["bad_regions"])
            self.segment_by_idx = {int(r["segment_idx"]): r for r in self.segments}
            self.beats_by_segment = {}
            for row in self.beats:
                self.beats_by_segment.setdefault(int(row["segment_idx"]), []).append(row)
            self._rebuild_added_beat_index()
            self.bad_by_segment = {}
            for row in self.bad_regions:
                self.bad_by_segment.setdefault(int(row["segment_idx"]), []).append(row)
            self.segment_group_by_idx = {}
            self._refresh_all_segment_groups()
        if getattr(self.args, "physio_review", False):
            if self.physio_review_total > 0:
                self.index = max(0, min(self.physio_review_total - 1, int(self.index)))
                self.visible_segments = [self._physio_review_row_at(self.index)]
            return
        selected = self._select_segments()
        if target_idx is not None:
            target = self.segment_by_idx.get(target_idx)
            if target is not None:
                target_pos = next(
                    (idx for idx, row in enumerate(selected) if int(row["segment_idx"]) == target_idx),
                    None,
                )
                if target_pos is None:
                    self.visible_segments = [target] + selected
                    self.index = 0
                else:
                    self.visible_segments = selected
                    self.index = target_pos
                return
        self.visible_segments = selected
        if self.visible_segments:
            self.index = min(self.index, len(self.visible_segments) - 1)

    def _push_undo(self) -> None:
        self.undo_stack.append(self._snapshot())
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _push_segment_undo(self) -> None:
        self.undo_stack.append(self._segment_row_snapshot())
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _push_beat_undo(self, seg_idx: int | None = None) -> None:
        self.undo_stack.append(self._beat_segment_snapshot(seg_idx))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _push_beat_multi_undo(self, seg_idxs: set[int] | list[int] | tuple[int, ...]) -> None:
        self.undo_stack.append(self._beat_segments_snapshot(seg_idxs))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _push_bad_undo(self, seg_idx: int | None = None) -> None:
        self.undo_stack.append(self._bad_segment_snapshot(seg_idx))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _save_all(self) -> None:
        self.segments = normalize_rows(self.segments, SEGMENT_COLUMNS)
        self.segment_by_idx = {int(r["segment_idx"]): r for r in self.segments}
        write_parquet(OUT_SEGMENTS, self.segments, SEGMENT_COLUMNS, "segments")
        write_parquet(OUT_BEATS, self.beats, BEAT_COLUMNS, "beats")
        write_parquet(OUT_BAD, self.bad_regions, BAD_COLUMNS, "bad")

    def _save_segment_outputs(self) -> None:
        self.segments = normalize_rows(self.segments, SEGMENT_COLUMNS)
        self.segment_by_idx = {int(r["segment_idx"]): r for r in self.segments}
        write_parquet(OUT_SEGMENTS, self.segments, SEGMENT_COLUMNS, "segments")

    def _save_beats_output(self) -> None:
        write_parquet(OUT_BEATS, self.beats, BEAT_COLUMNS, "beats")

    def _save_bad_output(self) -> None:
        write_parquet(OUT_BAD, self.bad_regions, BAD_COLUMNS, "bad")

    def _save_segment_and_bad_outputs(self) -> None:
        self._save_segment_outputs()
        self._save_bad_output()

    def _after_change(self, status: str, save: bool = True) -> None:
        self.status = status
        seg_idx = self._current_segment_idx_or_none()
        if seg_idx is not None:
            self._refresh_segment_group(seg_idx)
        if save:
            self._save_all()
        self._draw()

    def _after_segment_change(self, status: str, save: bool = True) -> None:
        self.status = status
        seg_idx = self._current_segment_idx_or_none()
        if seg_idx is not None:
            self._refresh_segment_group(seg_idx)
        if save:
            self._save_segment_outputs()
        self._draw()

    def _after_beat_change(self, status: str, save: bool = True) -> None:
        self.status = status
        seg_idx = self._current_segment_idx_or_none()
        if seg_idx is not None:
            self._refresh_segment_group(seg_idx)
        if save:
            self._save_beats_output()
        self._draw()

    def _log_action(self, action: str, details: dict[str, Any] | None = None, segment_idx: int | None = None) -> None:
        logger = getattr(self, "action_logger", None)
        if logger is None:
            return
        if segment_idx is None:
            segment_idx = self._current_segment_idx_or_none()
        logger.log(action, segment_idx=segment_idx, details=details or {})

    def _go(self, delta: int) -> None:
        self._go_physio_or_segment(delta, skip_reviewed=False)

    def _go_physio_or_segment(self, delta: int, skip_reviewed: bool = False) -> None:
        if not self.visible_segments:
            return
        if getattr(self.args, "markers", False):
            marker_idx = int(self.current_segment().get("_marker_idx") or 0)
            indices = self.marker_groups.get(marker_idx, [])
            if not indices:
                return
            pos = indices.index(self.index) if self.index in indices else min(
                range(len(indices)),
                key=lambda i: abs((int_or_none(self.visible_segments[indices[i]].get("_marker_offset")) or 0)),
            )
            new_pos = pos + (1 if delta >= 0 else -1)
            if new_pos < 0 or new_pos >= len(indices):
                side = "left" if delta < 0 else "right"
                self.status = f"Marker {marker_idx}: {side} edge"
                self._draw()
                return
            self.index = indices[new_pos]
            self.mode = "browse"
            self.bad_region_start_ms = None
            self.physio_region_start_ms = None
            self._load_window(reset_view=True)
            self._skip_empty_windows(direction=delta)
            self._draw()
            return
        if getattr(self.args, "physio_review", False):
            next_index = self._find_physio_review_index(self.index + delta, delta, skip_reviewed=skip_reviewed)
            if next_index is None:
                self.status = "No remaining unreviewed physio review beats" if skip_reviewed else "End of physio review queue"
                self._draw()
                return
            self.index = next_index
            self.visible_segments = [self._physio_review_row_at(self.index)]
            self.mode = "browse"
            self.bad_region_start_ms = None
            self.physio_region_start_ms = None
            self._load_window(reset_view=True)
            self._skip_empty_windows(direction=delta)
            self._draw()
            return
        if delta > 0 and self.forward_stack:
            target_idx = self.forward_stack.pop()
            target = self.segment_by_idx.get(target_idx)
            active = self._select_segments()
            if target is not None:
                ids = [int(r["segment_idx"]) for r in active]
                if target_idx in ids:
                    self.visible_segments = active
                    self.index = ids.index(target_idx)
                else:
                    self.visible_segments = [target] + active
                    self.index = 0
                self.mode = "browse"
                self.bad_region_start_ms = None
                self.physio_region_start_ms = None
                self._load_window(reset_view=True)
                self._skip_empty_windows(direction=1)
                self._log_action("navigate_forward_to_return_segment", {"from_key": "right"}, segment_idx=target_idx)
                self._draw()
                return
        if delta < 0 and self.back_stack:
            return_idx = self._current_segment_idx_or_none()
            target_idx = self.back_stack.pop()
            target = self.segment_by_idx.get(target_idx)
            if target is not None:
                if return_idx is not None and return_idx != target_idx:
                    self.forward_stack.append(return_idx)
                active = [r for r in self._select_segments() if int(r["segment_idx"]) != target_idx]
                self.visible_segments = [target] + active
                self.index = 0
                self.mode = "browse"
                self.bad_region_start_ms = None
                self.physio_region_start_ms = None
                self._load_window(reset_view=True)
                self._skip_empty_windows(direction=-1)
                self._log_action("navigate_back_to_recent_segment", {"from_key": "left"}, segment_idx=target_idx)
                self._draw()
                return
        current_idx = self._segment_idx()
        current_start = int(self.current_segment()["start_ms"])
        self.visible_segments = self._select_segments()
        if not self.visible_segments:
            self.status = "No remaining segments in this mode"
            self._draw()
            return
        ids = [int(r["segment_idx"]) for r in self.visible_segments]
        starts = [int(r["start_ms"]) for r in self.visible_segments]
        if current_idx in ids:
            base = ids.index(current_idx)
            self.index = max(0, min(len(self.visible_segments) - 1, base + delta))
        elif delta >= 0:
            self.index = min(bisect.bisect_right(starts, current_start), len(self.visible_segments) - 1)
        else:
            self.index = max(0, bisect.bisect_left(starts, current_start) - 1)
        self.mode = "browse"
        self.bad_region_start_ms = None
        self.physio_region_start_ms = None
        self._load_window(reset_view=True)
        self._skip_empty_windows(direction=delta)
        self._draw()

    def _advance_to_next_marker_center(self) -> None:
        if not getattr(self.args, "markers", False):
            self._advance_after_terminal_label()
            return
        next_pos = self._find_next_marker_position(self.marker_position + 1, 1)
        if next_pos is None:
            self.status = "No remaining unreviewed marker-center segments"
            self._draw()
            return
        self.marker_position = next_pos
        self.index = self._marker_center_index(self.marker_order[self.marker_position])
        self.mode = "browse"
        self.bad_region_start_ms = None
        self.physio_region_start_ms = None
        self._load_window(reset_view=True)
        self._skip_empty_windows(direction=1)
        self._draw()

    def _advance_after_terminal_label(self) -> None:
        seg_idx = self._segment_idx()
        old_start = int(self.current_segment()["start_ms"])
        self.visible_segments = self._select_segments()
        if not self.visible_segments:
            self._save_segment_outputs()
            self.status = "No remaining segments in this mode"
            self._draw()
            return
        starts = [int(r["start_ms"]) for r in self.visible_segments]
        self.index = min(bisect.bisect_right(starts, old_start), len(self.visible_segments) - 1)
        if int(self.visible_segments[self.index]["segment_idx"]) == seg_idx and self.index < len(self.visible_segments) - 1:
            self.index += 1
        self._load_window(reset_view=True)
        self._skip_empty_windows(direction=1)
        self._draw()

    def _set_segment_quality_fields(self, row: dict[str, Any]) -> None:
        normalized = normalize_segment_training_flags(row)
        row.update(normalized)
        if str(row.get("review_status")) == "revisit" or bool_value(row.get("in_revisit_pile")):
            return
        if str(row.get("review_status")) == "unreviewed":
            return
        quality = str(row.get("segment_quality_label") or "unknown")
        has_bad = bool_value(row.get("has_bad_region"))
        if quality == "unusable":
            row["beat_training_eligible_segment"] = False
            row["segment_quality_training_eligible"] = True
        elif has_bad:
            if quality in {"unknown", "", "usable"}:
                row["segment_quality_label"] = "partial"
            row["beat_training_eligible_segment"] = True
            row["segment_quality_training_eligible"] = True
        elif quality in {"usable", "partial", "unknown", ""}:
            row["beat_training_eligible_segment"] = True
            row["segment_quality_training_eligible"] = True

    def _persistent_current_segment(self) -> dict[str, Any]:
        seg = self.current_segment()
        seg_idx = int(seg["segment_idx"])
        row = self.segment_by_idx.get(seg_idx)
        if row is None:
            row = {col: seg.get(col) for col in SEGMENT_COLUMNS}
            self.segments.append(row)
            self.segment_by_idx[seg_idx] = row
        return row

    def _sync_current_marker_row(self, row: dict[str, Any]) -> None:
        if not (getattr(self.args, "markers", False) or getattr(self.args, "marker_recovery", False)) or not self.visible_segments:
            return
        marker_fields = {
            key: self.visible_segments[self.index].get(key)
            for key in ("_marker_idx", "_marker_timestamp_ms", "_marker_center_segment_idx", "_marker_offset")
        }
        self.visible_segments[self.index].update({col: row.get(col) for col in SEGMENT_COLUMNS})
        self.visible_segments[self.index].update(marker_fields)

    def _toggle_reviewed(self) -> None:
        self._push_segment_undo()
        seg = self._persistent_current_segment() if (
            getattr(self.args, "markers", False) or getattr(self.args, "marker_recovery", False)
        ) else self.current_segment()
        self.forward_stack.clear()
        if str(seg.get("review_status")) == "reviewed":
            before = dict(seg)
            seg["review_status"] = "unreviewed"
            self._sync_current_marker_row(seg)
            self._log_action(
                "segment_unmarked_reviewed",
                {"before": before, "after": dict(seg)},
                segment_idx=int(seg["segment_idx"]),
            )
            self._after_segment_change(f"Unmarked Segment {seg['segment_idx']} reviewed")
            return
        before = dict(seg)
        self.back_stack.append(int(seg["segment_idx"]))
        seg["review_status"] = "reviewed"
        seg["in_revisit_pile"] = False
        if str(seg.get("segment_quality_label") or "unknown") == "unknown":
            seg["segment_quality_label"] = "partial" if bool_value(seg.get("has_bad_region")) else "usable"
        self._set_segment_quality_fields(seg)
        self._sync_current_marker_row(seg)
        self._log_action(
            "segment_marked_reviewed",
            {"before": before, "after": dict(seg)},
            segment_idx=int(seg["segment_idx"]),
        )
        self._save_segment_outputs()
        self.status = f"Marked Segment {seg['segment_idx']} reviewed"
        self._advance_to_next_marker_center() if getattr(self.args, "markers", False) else self._advance_after_terminal_label()

    def _toggle_revisit(self) -> None:
        self._push_segment_undo()
        seg = self._persistent_current_segment() if (
            getattr(self.args, "markers", False) or getattr(self.args, "marker_recovery", False)
        ) else self.current_segment()
        self.forward_stack.clear()
        if bool_value(seg.get("in_revisit_pile")) or str(seg.get("review_status")) == "revisit":
            before = dict(seg)
            seg["in_revisit_pile"] = False
            seg["review_status"] = "unreviewed"
            self._sync_current_marker_row(seg)
            self._log_action(
                "segment_removed_from_revisit",
                {"before": before, "after": dict(seg)},
                segment_idx=int(seg["segment_idx"]),
            )
            self._after_segment_change(f"Removed Segment {seg['segment_idx']} from revisit")
            return
        before = dict(seg)
        self.back_stack.append(int(seg["segment_idx"]))
        seg["in_revisit_pile"] = True
        seg["review_status"] = "revisit"
        self._sync_current_marker_row(seg)
        self._log_action(
            "segment_added_to_revisit",
            {"before": before, "after": dict(seg)},
            segment_idx=int(seg["segment_idx"]),
        )
        self._save_segment_outputs()
        self.status = f"Added Segment {seg['segment_idx']} to revisit"
        self._advance_to_next_marker_center() if getattr(self.args, "markers", False) else self._advance_after_terminal_label()

    def _toggle_unusable(self) -> None:
        self._push_segment_undo()
        seg = self._persistent_current_segment() if getattr(self.args, "marker_recovery", False) else self.current_segment()
        self.forward_stack.clear()
        old = str(seg.get("segment_quality_label") or "unknown")
        old_beat = bool_value(seg.get("beat_training_eligible_segment"))
        old_seg = bool_value(seg.get("segment_quality_training_eligible"))
        self.last_b_revert = (int(seg["segment_idx"]), old, old_beat, old_seg)
        before = dict(seg)
        if old == "unusable":
            seg["segment_quality_label"] = "unknown"
            seg["beat_training_eligible_segment"] = True
            seg["segment_quality_training_eligible"] = True
            self._log_action(
                "segment_unmarked_unusable",
                {"before": before, "after": dict(seg)},
                segment_idx=int(seg["segment_idx"]),
            )
            self._after_segment_change(f"Removed unusable label from Segment {seg['segment_idx']}")
        else:
            self.back_stack.append(int(seg["segment_idx"]))
            seg["segment_quality_label"] = "unusable"
            seg["review_status"] = "reviewed"
            seg["in_revisit_pile"] = False
            seg["beat_training_eligible_segment"] = False
            seg["segment_quality_training_eligible"] = True
            self._sync_current_marker_row(seg)
            self._log_action(
                "segment_marked_unusable",
                {"before": before, "after": dict(seg)},
                segment_idx=int(seg["segment_idx"]),
            )
            if getattr(self.args, "marker_recovery", False):
                self._save_segment_outputs()
                self.status = f"Marked Segment {seg['segment_idx']} reviewed unusable"
                self._advance_after_terminal_label()
            else:
                self._after_segment_change(f"Marked Segment {seg['segment_idx']} unusable")

    def _restore_last_b_toggle(self) -> None:
        if not self.last_b_revert:
            return
        seg_idx, label, beat_ok, segment_ok = self.last_b_revert
        seg = self.segment_by_idx.get(seg_idx)
        if not seg:
            return
        before = dict(seg)
        seg["segment_quality_label"] = label
        seg["beat_training_eligible_segment"] = beat_ok
        seg["segment_quality_training_eligible"] = segment_ok
        self.last_b_revert = None
        self._log_action(
            "segment_unusable_toggle_reverted_for_bad_region_mode",
            {"before": before, "after": dict(seg)},
            segment_idx=seg_idx,
        )

    def _add_bad_region(self, start_ms: int, end_ms: int) -> None:
        seg = self._persistent_current_segment() if (
            getattr(self.args, "markers", False) or getattr(self.args, "marker_recovery", False)
        ) else self.current_segment()
        seg_idx = int(seg["segment_idx"])
        self._push_bad_undo(seg_idx)
        lo = min(start_ms, end_ms)
        hi = max(start_ms, end_ms)
        existing = self.bad_by_segment.setdefault(seg_idx, [])
        next_region = 1 + max([int_or_none(r.get("region_idx_within_segment")) or 0 for r in existing], default=0)
        row = {
            "segment_idx": seg_idx,
            "region_idx_within_segment": next_region,
            "start_ms": lo,
            "end_ms": hi,
            "original_annotation": "manual_bad_region",
            "comment": "",
        }
        self.bad_regions.append(row)
        existing.append(row)
        seg["has_bad_region"] = True
        seg["bad_region_count"] = len(existing)
        if str(seg.get("segment_quality_label") or "unknown") in {"unknown", "usable", ""}:
            seg["segment_quality_label"] = "partial"
        self._set_segment_quality_fields(seg)
        self._sync_current_marker_row(seg)
        self.mode = "browse"
        self.bad_region_start_ms = None
        self._refresh_segment_group(seg_idx)
        self._log_action("bad_region_added", {"region": dict(row), "segment_after": dict(seg)}, segment_idx=seg_idx)
        self.status = f"Added bad region to Segment {seg_idx}"
        self._save_segment_and_bad_outputs()
        self._draw()

    def _find_hit(self, event: Any) -> PeakPoint | None:
        if event.x is None or event.y is None:
            return None
        best: PeakPoint | None = None
        best_dist = CLICK_RADIUS_PX + 1
        for point in self.display_points:
            x = (point.timestamp_ms - int(self.current_segment()["start_ms"])) / 1000.0
            px, py = self.ax.transData.transform((x, point.y))
            dist = math.hypot(float(event.x) - px, float(event.y) - py)
            if dist <= CLICK_RADIUS_PX and dist < best_dist:
                best = point
                best_dist = dist
        return best

    def _point_identity_ts(self, point: PeakPoint) -> int:
        return int(point.raw_timestamp_ms if point.raw_timestamp_ms is not None else point.timestamp_ms)

    def _upsert_beat_label(self, point: PeakPoint, label: str, subtype: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        seg_idx = self._point_segment_idx(point)
        identity_ts = self._point_identity_ts(point)
        row = self._annotation_for_timestamp(seg_idx, identity_ts)
        before = copy.deepcopy(row) if row is not None else None
        if row is None:
            row = {
                "timestamp_ms": identity_ts,
                "segment_idx": seg_idx,
                "label": label,
                "subtype": subtype,
                "is_added_peak": bool(point.is_added_peak),
                "beat_training_eligible": True,
                "comment": "",
            }
            self.beats.append(row)
            self.beats_by_segment.setdefault(seg_idx, []).append(row)
        else:
            row["label"] = label
            row["subtype"] = subtype
            row["is_added_peak"] = bool(point.is_added_peak)
            row["beat_training_eligible"] = True
        return before, copy.deepcopy(row)

    def _remove_annotation_for_point(self, point: PeakPoint) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        seg_idx = self._point_segment_idx(point)
        identity_ts = self._point_identity_ts(point)
        rows = self.beats_by_segment.get(seg_idx, [])
        changes: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in rows:
            if abs(int(row["timestamp_ms"]) - identity_ts) <= ANNOTATION_MATCH_TOLERANCE_MS:
                before = copy.deepcopy(row)
                row["label"] = "clean"
                row["subtype"] = "added" if bool_value(row.get("is_added_peak")) else "auto"
                row["beat_training_eligible"] = True
                changes.append((before, copy.deepcopy(row)))
        # rows is built from self.beats, so the dict updates above already
        # mutate the canonical table.
        return changes

    def _toggle_artifact(self, point: PeakPoint) -> None:
        seg_idx = self._point_segment_idx(point)
        self._push_beat_undo(seg_idx)
        if point.label == "artifact" and point.subtype == "interpolate":
            changes = self._remove_annotation_for_point(point)
            self._refresh_segment_group(seg_idx)
            self._log_action(
                "interpolate_artifact_removed",
                {"point": point.__dict__, "changes": changes},
                segment_idx=seg_idx,
            )
            self._after_beat_change(f"Removed interpolate artifact at {point.timestamp_ms}")
        elif point.label == "artifact":
            before, after = self._upsert_beat_label(point, "artifact", "interpolate")
            self._refresh_segment_group(seg_idx)
            self._log_action(
                "artifact_changed_to_interpolate",
                {"point": point.__dict__, "before": before, "after": after},
                segment_idx=seg_idx,
            )
            self._after_beat_change(f"Marked interpolate artifact at {point.timestamp_ms}")
        else:
            before, after = self._upsert_beat_label(point, "artifact", "spurious")
            self._refresh_segment_group(seg_idx)
            self._log_action(
                "artifact_marked",
                {"point": point.__dict__, "before": before, "after": after},
                segment_idx=seg_idx,
            )
            self._after_beat_change(f"Marked artifact at {point.timestamp_ms}")

    def _toggle_physio(self, point: PeakPoint) -> None:
        seg_idx = self._point_segment_idx(point)
        self._push_beat_undo(seg_idx)
        if point.label == "physio":
            changes = self._remove_annotation_for_point(point)
            self._refresh_segment_group(seg_idx)
            self._log_action("physio_removed", {"point": point.__dict__, "changes": changes}, segment_idx=seg_idx)
            self._after_beat_change(f"Removed physio at {point.timestamp_ms}")
        else:
            before, after = self._upsert_beat_label(point, "physio", "manual")
            self._refresh_segment_group(seg_idx)
            self._log_action(
                "physio_marked",
                {"point": point.__dict__, "before": before, "after": after},
                segment_idx=seg_idx,
            )
            self._after_beat_change(f"Marked physio at {point.timestamp_ms}")

    def _review_metadata(self) -> dict[str, Any]:
        seg = self.current_segment()
        return {
            "review_peak_id": int_or_none(seg.get("_review_peak_id")),
            "review_peak_timestamp_ms": int_or_none(seg.get("_review_peak_timestamp_ms")),
            "review_assigned_label": str(seg.get("_review_assigned_label") or ""),
        }

    def _toggle_physio_review_label(self, point: PeakPoint, label: str, subtype: str, display: str) -> None:
        seg_idx = self._point_segment_idx(point)
        self._push_beat_undo(seg_idx)
        before, after = self._upsert_beat_label(point, label, subtype)
        action = f"physio_review_{label}_marked"
        self._log_action(
            action,
            {"point": point.__dict__, "before": before, "after": after},
            segment_idx=seg_idx,
        )
        self._after_beat_change(f"Marked {display} at {point.timestamp_ms}")

    def _mark_physio_segment_reviewed(self) -> None:
        seg = self.current_segment()
        seg_idx = int(seg["segment_idx"])
        self._push_segment_undo()
        before = copy.deepcopy(self.segment_by_idx.get(seg_idx))
        row = self.segment_by_idx.get(seg_idx)
        if row is None:
            row = {
                "segment_idx": seg_idx,
                "start_ms": int(seg["start_ms"]),
                "end_ms": int(seg["end_ms"]),
                "review_status": "reviewed",
                "segment_quality_label": str(seg.get("segment_quality_label") or "usable"),
                "in_revisit_pile": False,
                "has_bad_region": False,
                "bad_region_count": 0,
                "beat_training_eligible_segment": True,
                "segment_quality_training_eligible": True,
                "comment": "",
            }
            self.segments.append(row)
            self.segment_by_idx[seg_idx] = row
        else:
            row["start_ms"] = int(seg["start_ms"])
            row["end_ms"] = int(seg["end_ms"])
            row["review_status"] = "reviewed"
            if str(row.get("segment_quality_label") or "").strip() in {"", "unknown"}:
                row["segment_quality_label"] = "usable"
            row["in_revisit_pile"] = False
        self._set_segment_quality_fields(row)
        self._log_action(
            "physio_segment_marked_reviewed",
            {"before": before, "after": copy.deepcopy(row)},
            segment_idx=seg_idx,
        )
        self.physio_reviewed_segment_idxs.add(seg_idx)
        self._refresh_segment_group(seg_idx)
        self._save_segment_outputs()
        self.status = f"Marked physio Segment {seg_idx} reviewed"
        self._go_physio_or_segment(1, skip_reviewed=True)

    def _add_physio_region(self, start_ms: int, end_ms: int) -> None:
        _ = (start_ms, end_ms)
        self.mode = "browse"
        self.physio_region_start_ms = None
        self.status = "Physio regions are not saved until a single non-bad-region schema exists"
        self._log_action("physio_region_not_saved", {"reason": "no unified region schema"})
        self._draw()

    def _add_peak_at_click(self, event: Any) -> None:
        if event.xdata is None:
            return
        seg = self.current_segment()
        ts = int(int(seg["start_ms"]) + round(float(event.xdata) * 1000))
        lo = self.loaded_lo_ms if self.loaded_lo_ms is not None else int(seg["start_ms"])
        hi = self.loaded_hi_ms if self.loaded_hi_ms is not None else int(seg["end_ms"])
        ts = max(int(lo), min(int(hi), ts))
        snap_window = PRECISE_ADDED_PEAK_SNAP_WINDOW_MS if self.precise_add_mode else ADDED_PEAK_SNAP_WINDOW_MS
        ts, _ = self._local_max_point(ts, window_ms=snap_window)
        seg_idx = self._real_segment_idx_for_timestamp(ts) if (
            getattr(self.args, "markers", False) or getattr(self.args, "marker_recovery", False)
        ) else int(seg["segment_idx"])
        self._push_beat_undo(seg_idx)
        row = {
            "timestamp_ms": ts,
            "segment_idx": seg_idx,
            "label": "clean",
            "subtype": "added",
            "is_added_peak": True,
            "beat_training_eligible": True,
            "comment": "",
        }
        self.beats.append(row)
        self.beats_by_segment.setdefault(seg_idx, []).append(row)
        self.added_beats_by_timestamp.append((ts, row))
        self.added_beats_by_timestamp.sort(key=lambda item: item[0])
        self._refresh_segment_group(seg_idx)
        self._log_action("added_peak_created", {"row": dict(row)}, segment_idx=seg_idx)
        self._after_beat_change(f"Added peak at {ts}")

    def _auto_negative_artifacts(self) -> None:
        if not self.ecg_y:
            return
        points = [p for p in self._make_points() if p.label != "artifact"]
        if len(points) < 2:
            self.status = "Need at least 2 non-artifact peaks for amplitude artifact pass"
            self._draw()
            return
        candidates = low_amplitude_artifact_candidates(points)

        if not candidates:
            self.status = "No peaks met the <25% neighbors or <50% segment-average amplitude rules"
            self._draw()
            return
        seg_idxs = {self._point_segment_idx(point) for point in candidates}
        self._push_beat_multi_undo(seg_idxs)
        count = 0
        changes: list[dict[str, Any]] = []
        for point in candidates:
            before, after = self._upsert_beat_label(point, "artifact", "spurious")
            changes.append({"point": point.__dict__, "before": before, "after": after})
            count += 1
        self._refresh_segment_group(self._segment_idx())
        self._log_action(
            "auto_low_amplitude_artifacts_marked",
            {"count": count, "changes": changes},
            segment_idx=self._segment_idx(),
        )
        self._after_beat_change(f"Marked {count} low-amplitude peaks as artifacts")

    def on_click(self, event: Any) -> None:
        if event.inaxes != self.ax:
            return
        if self.comment_active:
            return
        if self.mode == "physio_region":
            if event.xdata is None:
                return
            seg = self.current_segment()
            ts = int(int(seg["start_ms"]) + round(float(event.xdata) * 1000))
            ts = max(int(seg["start_ms"]), min(int(seg["end_ms"]), ts))
            if self.physio_region_start_ms is None:
                self.physio_region_start_ms = ts
                self.status = f"Physio region start {ts}"
                self._draw()
            else:
                self._add_physio_region(self.physio_region_start_ms, ts)
            return
        if self.mode == "bad_region":
            if event.xdata is None:
                return
            seg = self.current_segment()
            ts = int(int(seg["start_ms"]) + round(float(event.xdata) * 1000))
            ts = max(int(seg["start_ms"]), min(int(seg["end_ms"]), ts))
            if self.bad_region_start_ms is None:
                self.bad_region_start_ms = ts
                self.status = f"Bad region start {ts}"
                self._draw()
            else:
                self._add_bad_region(self.bad_region_start_ms, ts)
            return

        point = self._find_hit(event)
        if self.mode == "pac_pvc":
            if point is not None:
                self._toggle_physio_review_label(point, "pac_pvc", "manual", "PAC/PVC")
            return
        if self.mode == "physio_aftereffect":
            if point is not None:
                self._toggle_physio_review_label(point, "physio_aftereffect", "manual", "physio aftereffect")
            return
        if self.mode == "arrhythmia_unspecified":
            if point is not None:
                self._toggle_physio_review_label(point, "arrhythmia_unspecified", "manual", "unspecified arrhythmia")
            return
        if self.mode == "physio":
            if point is not None:
                if getattr(self.args, "physio_review", False):
                    self._toggle_physio_review_label(point, "physio", "manual", "physio beat")
                else:
                    self._toggle_physio(point)
            return
        if point is not None:
            self._toggle_artifact(point)
            return
        self._add_peak_at_click(event)

    def on_scroll(self, event: Any) -> None:
        if self.comment_active or event.xdata is None:
            return
        seg = self.current_segment()
        seg_start = int(seg["start_ms"])
        bound_lo = self.loaded_lo_ms if self.loaded_lo_ms is not None else int(seg["start_ms"])
        bound_hi = self.loaded_hi_ms if self.loaded_hi_ms is not None else int(seg["end_ms"])
        lo = self.view_lo_ms if self.view_lo_ms is not None else bound_lo
        hi = self.view_hi_ms if self.view_hi_ms is not None else bound_hi
        center = seg_start + int(float(event.xdata) * 1000)
        current_width = hi - lo
        if event.button == "up":
            width = max(MIN_VIEW_MS, int(current_width / ZOOM_FACTOR))
        else:
            width = min(max(MAX_VIEW_MS, bound_hi - bound_lo), int(current_width * ZOOM_FACTOR))
        cursor_frac = 0.5 if current_width <= 0 else (center - lo) / current_width
        cursor_frac = max(0.05, min(0.95, cursor_frac))
        new_lo = int(center - width * cursor_frac)
        new_hi = new_lo + width
        if new_lo < bound_lo:
            shift = bound_lo - new_lo
            new_lo += shift
            new_hi += shift
        if new_hi > bound_hi:
            shift = new_hi - bound_hi
            new_lo -= shift
            new_hi -= shift
        new_lo = max(bound_lo, new_lo)
        new_hi = min(bound_hi, new_hi)
        self.view_lo_ms, self.view_hi_ms = new_lo, new_hi
        self._draw()

    def on_key(self, event: Any) -> None:
        key = event.key or ""
        if key in {" ", "spacebar"}:
            key = "space"
        if self.comment_active:
            self._handle_comment_key(key)
            return
        if key in {"left", "["}:
            self._go(-1)
        elif key in {"right", "]"}:
            self._go(1)
        elif key == "space":
            if getattr(self.args, "physio_review", False):
                self._mark_physio_segment_reviewed()
            else:
                self._toggle_reviewed()
        elif key == "z":
            if self.undo_stack:
                entry = self.undo_stack.pop()
                self.redo_stack.append(self._snapshot_for_entry(entry))
                self._restore(entry)
                if entry.get("kind") == "segment_row":
                    self._save_segment_outputs()
                elif entry.get("kind") in {"beat_segment", "beat_segments"}:
                    self._save_beats_output()
                elif entry.get("kind") == "bad_segment":
                    self._save_segment_and_bad_outputs()
                else:
                    self._save_all()
                self.status = "Undo"
                self._load_window(reset_view=False)
                self._log_action("undo", {"restored_kind": entry.get("kind")}, segment_idx=int_or_none(entry.get("segment_idx")))
                self._draw()
        elif key == "x":
            if self.redo_stack:
                entry = self.redo_stack.pop()
                self.undo_stack.append(self._snapshot_for_entry(entry))
                self._restore(entry)
                if entry.get("kind") == "segment_row":
                    self._save_segment_outputs()
                elif entry.get("kind") in {"beat_segment", "beat_segments"}:
                    self._save_beats_output()
                elif entry.get("kind") == "bad_segment":
                    self._save_segment_and_bad_outputs()
                else:
                    self._save_all()
                self.status = "Redo"
                self._load_window(reset_view=False)
                self._log_action("redo", {"restored_kind": entry.get("kind")}, segment_idx=int_or_none(entry.get("segment_idx")))
                self._draw()
        elif key == "m":
            self.status = "reviewed is now spacebar"
            self._draw()
        elif key == "r":
            self._toggle_revisit()
        elif key == "e":
            self.mode = "browse" if self.mode == "pac_pvc" else "pac_pvc"
            self.status = "PAC/PVC mode: click peak(s)"
            self._draw()
        elif key == "j":
            self.mode = "browse" if self.mode == "physio_aftereffect" else "physio_aftereffect"
            self.status = "Physio aftereffect mode: click peak(s)"
            self._draw()
        elif key == "y":
            self.mode = "browse" if self.mode == "arrhythmia_unspecified" else "arrhythmia_unspecified"
            self.status = "Unspecified arrhythmia mode: click peak(s)"
            self._draw()
        elif key == "p":
            now = time.monotonic()
            if now - self.last_p_time <= 0.45:
                self.mode = "physio_region"
                self.physio_region_start_ms = None
                self.status = "Physio region mode: click start, then end"
            else:
                self.mode = "browse" if self.mode == "physio" else "physio"
                self.status = "Physio beat mode: click peak(s)"
            self.last_p_time = now
            self._draw()
        elif key == "b":
            now = time.monotonic()
            if now - self.last_b_time <= 0.45:
                self._restore_last_b_toggle()
                self._save_segment_outputs()
                self.mode = "bad_region"
                self.bad_region_start_ms = None
                self.status = "Bad region mode: click start, then end"
                self._draw()
            else:
                self._toggle_unusable()
            self.last_b_time = now
        elif key == "v":
            self._auto_negative_artifacts()
        elif key == "a":
            self.precise_add_mode = not self.precise_add_mode
            self.status = f"precise add snap {'on' if self.precise_add_mode else 'off'}"
            self._draw()
        elif key == "c":
            self.comment_active = True
            self.comment_buffer = str(self.current_segment().get("comment") or "")
            self.status = "Comment mode"
            self._draw()
        elif key == "escape":
            self.mode = "browse"
            self.bad_region_start_ms = None
            self.physio_region_start_ms = None
            self.status = "browse mode"
            self._draw()
        elif key == "q":
            self._log_action("quit")
            self._save_all()
            self.action_logger.close()
            plt.close(self.fig)

    def _handle_comment_key(self, key: str) -> None:
        if key == "c":
            self._push_segment_undo()
            before = dict(self.current_segment())
            self.current_segment()["comment"] = self.comment_buffer
            self.comment_active = False
            self._log_action(
                "segment_comment_saved",
                {"before": before, "after": dict(self.current_segment())},
                segment_idx=self._segment_idx(),
            )
            self._after_segment_change("Saved segment comment")
            return
        if key == "backspace":
            self.comment_buffer = self.comment_buffer[:-1]
        elif key in {"enter", "return"}:
            self.comment_buffer += "\n"
        elif key == "space":
            self.comment_buffer += " "
        elif len(key) == 1:
            self.comment_buffer += key
        self._draw()

    def _rr_text_color(self, prev_rr: int | None, rr: int) -> str:
        if prev_rr is None or prev_rr <= 0:
            return blend(RR_BASE, RR_BASE, 0)
        delta = rr - prev_rr
        amount = min(1.0, abs(delta / prev_rr) / RR_CHANGE_FULL_SCALE)
        if delta < 0:
            return blend(RR_BASE, RR_DECREASE, amount)
        return blend(RR_BASE, RR_INCREASE, amount)

    def _draw_rr_labels(self, points: list[PeakPoint], lo_ms: int, hi_ms: int) -> None:
        rr_points = [p for p in points if p.label != "artifact"]
        rr_points.sort(key=lambda p: p.timestamp_ms)
        prev_rr: int | None = None
        seg_start = int(self.current_segment()["start_ms"])
        y_positions = (0.975, 0.925)
        trans = self.ax.get_xaxis_transform()
        visible_idx = 0
        intervals: list[tuple[PeakPoint, PeakPoint, int, int]] = []
        for left, right in zip(rr_points, rr_points[1:]):
            rr = right.timestamp_ms - left.timestamp_ms
            mid_ms = int((left.timestamp_ms + right.timestamp_ms) / 2)
            if mid_ms < lo_ms or mid_ms > hi_ms:
                prev_rr = rr
                continue
            intervals.append((left, right, rr, mid_ms))
            prev_rr = rr
        if len(intervals) > MAX_RR_TEXT_LABELS:
            stride = math.ceil(len(intervals) / MAX_RR_TEXT_LABELS)
            intervals = intervals[::stride]
        prev_rr = None
        for left, right, rr, mid_ms in intervals:
            mid_x = (mid_ms - seg_start) / 1000.0
            color = self._rr_text_color(prev_rr, rr)
            self.ax.text(
                mid_x,
                y_positions[visible_idx % len(y_positions)],
                str(int(rr)),
                color=color,
                fontsize=16,
                family="monospace",
                ha="center",
                va="top",
                transform=trans,
                fontweight="semibold",
                clip_on=True,
                bbox={"facecolor": "#070b13", "edgecolor": "none", "alpha": 0.58, "pad": 0.8},
            )
            visible_idx += 1
            prev_rr = rr

    def _draw_bad_regions(self) -> None:
        seg = self.current_segment()
        seg_idx = int(seg["segment_idx"])
        seg_start = int(seg["start_ms"])
        ymin, ymax = self.ax.get_ylim()
        for region in self.bad_by_segment.get(seg_idx, []):
            lo = (int(region["start_ms"]) - seg_start) / 1000.0
            width = (int(region["end_ms"]) - int(region["start_ms"])) / 1000.0
            patch = Rectangle((lo, ymin), width, ymax - ymin, facecolor=BAD_REGION_COLOR, alpha=0.18, edgecolor=None)
            self.ax.add_patch(patch)

    def _segment_metric_summary(self, seg_idx: int) -> str:
        if getattr(self.args, "markers", False):
            marker_ts = int_or_none(self.current_segment().get("_marker_timestamp_ms"))
            if marker_ts is not None:
                return f"marker {ms_to_local(marker_ts)}"
            return "marker"
        metrics = self.segment_metrics.get(seg_idx)
        group = self.segment_group_by_idx.get(seg_idx, "likely-clean")
        if metrics is None:
            return group
        min_rr = metrics.min_rr_ms if metrics.min_rr_ms is not None else "NA"
        max_delta = metrics.max_rr_increase_ms if metrics.max_rr_increase_ms is not None else "NA"
        pct = metrics.max_rr_change_pct * 100
        return f"{group} | RMSSD {metrics.rmssd_ms:.0f} | maxRR% {pct:.0f} | minRR {min_rr} | max+dRR {max_delta}"

    def _draw(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor(BG)
        if not self.visible_segments:
            self.display_points = []
            msg = self.status or "No remaining segments in this mode"
            self.ax.text(
                0.5,
                0.5,
                msg,
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                color="#e9eef8",
                fontsize=18,
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title("No segments available", color="#e9eef8", fontsize=14, pad=18)
            self.fig.canvas.draw_idle()
            return
        seg = self.current_segment()
        seg_start = int(seg["start_ms"])
        lo_ms = self.view_lo_ms if self.view_lo_ms is not None else int(seg["start_ms"])
        hi_ms = self.view_hi_ms if self.view_hi_ms is not None else int(seg["end_ms"])

        x = [(ms - seg_start) / 1000.0 for ms in self.ecg_ms]
        self.ax.plot(x, self.ecg_y, color=ECG_COLOR, linewidth=1.05, alpha=0.82)
        review_ts = int_or_none(seg.get("_review_peak_timestamp_ms"))
        if review_ts is not None:
            self.ax.axvspan(
                (review_ts - REVIEW_SHADE_MS - seg_start) / 1000.0,
                (review_ts + REVIEW_SHADE_MS - seg_start) / 1000.0,
                color="#fff2a8",
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
        if self.ecg_y:
            pad = max((max(self.ecg_y) - min(self.ecg_y)) * 0.15, 0.05)
            self.ax.set_ylim(min(self.ecg_y) - pad, max(self.ecg_y) + pad)

        self.display_points = self._make_points()
        clean_x: list[float] = []
        clean_y: list[float] = []
        added_x: list[float] = []
        added_y: list[float] = []
        art_x: list[float] = []
        art_y: list[float] = []
        interp_x: list[float] = []
        interp_y: list[float] = []
        phys_x: list[float] = []
        phys_y: list[float] = []
        pac_x: list[float] = []
        pac_y: list[float] = []
        after_x: list[float] = []
        after_y: list[float] = []
        arr_x: list[float] = []
        arr_y: list[float] = []
        for p in self.display_points:
            px = (p.timestamp_ms - seg_start) / 1000.0
            if p.label == "artifact" and p.subtype == "interpolate":
                interp_x.append(px); interp_y.append(p.y)
            elif p.label == "artifact":
                art_x.append(px); art_y.append(p.y)
            elif p.label == "pac_pvc":
                pac_x.append(px); pac_y.append(p.y)
            elif p.label == "physio_aftereffect":
                after_x.append(px); after_y.append(p.y)
            elif p.label == "arrhythmia_unspecified":
                arr_x.append(px); arr_y.append(p.y)
            elif p.label == "physio":
                phys_x.append(px); phys_y.append(p.y)
            elif p.is_added_peak:
                added_x.append(px); added_y.append(p.y)
            else:
                clean_x.append(px); clean_y.append(p.y)

        if clean_x:
            self.ax.scatter(clean_x, clean_y, s=82, c=PEAK_COLOR, edgecolors="#ffd79c", linewidths=1.0, zorder=5)
        if added_x:
            self.ax.scatter(added_x, added_y, s=110, c=PEAK_COLOR, edgecolors=ADDED_EDGE, linewidths=2.2, zorder=6)
        if art_x:
            self.ax.scatter(art_x, art_y, s=190, c=ARTIFACT_COLOR, marker="x", linewidths=3.6, zorder=7)
        if interp_x:
            self.ax.scatter(interp_x, interp_y, s=155, c=ARTIFACT_COLOR, marker="D", edgecolors="#ffd0d8", linewidths=1.5, zorder=7)
        if pac_x:
            self.ax.scatter(pac_x, pac_y, s=175, c=PAC_PVC_COLOR, marker="^", edgecolors="#ffe6f6", linewidths=1.2, zorder=8)
        if after_x:
            self.ax.scatter(after_x, after_y, s=150, c=PHYSIO_AFTEREFFECT_COLOR, marker="D", edgecolors="#dcfffb", linewidths=1.1, zorder=8)
        if arr_x:
            self.ax.scatter(arr_x, arr_y, s=155, c=ARRHYTHMIA_COLOR, marker="s", edgecolors="#fff9be", linewidths=1.1, zorder=8)
        if phys_x:
            self.ax.scatter(phys_x, phys_y, s=210, c=PHYSIO_COLOR, marker="*", edgecolors="#f3e8ff", linewidths=1.3, zorder=8)

        self.ax.set_xlim((lo_ms - seg_start) / 1000.0, (hi_ms - seg_start) / 1000.0)
        self._draw_bad_regions()
        self._draw_rr_labels(self.display_points, lo_ms, hi_ms)
        self.ax.tick_params(axis="both", colors="#9fb2d3", labelsize=9)
        self.ax.grid(True, alpha=0.14, color="#92a9d6", linewidth=0.7)
        for spine in self.ax.spines.values():
            spine.set_color("#334261")

        mode_bits = []
        if self.comment_active:
            mode_bits.append("COMMENT MODE")
        elif self.mode != "browse":
            mode_bits.append(self.mode.upper())
        if getattr(self.args, "markers", False):
            marker_offset = int_or_none(seg.get("_marker_offset"))
            marker_offset_text = f"{marker_offset:+d}" if marker_offset is not None else "NA"
            title = (
                f"Marker {seg.get('_marker_idx')} seg{marker_offset_text} "
                f"(Segment {seg['segment_idx']})  {self.marker_position + 1}/{self._queue_size()}  "
                f"[{ms_to_local(int(seg['start_ms']))} - {ms_to_local(int(seg['end_ms']))}]"
            )
        elif getattr(self.args, "marker_recovery", False):
            marker_offset = int_or_none(seg.get("_marker_offset"))
            marker_offset_text = f"{marker_offset:+d}" if marker_offset is not None else "NA"
            title = (
                f"Marker Recovery {self.index + 1}/{self._queue_size()}  "
                f"Marker {seg.get('_marker_idx')} seg{marker_offset_text} "
                f"(Segment {seg['segment_idx']})  "
                f"[{ms_to_local(int(seg['start_ms']))} - {ms_to_local(int(seg['end_ms']))}]"
            )
        else:
            title = (
                f"Segment {seg['segment_idx']}  {self.index + 1}/{self._queue_size()}  "
                f"[{ms_to_local(int(seg['start_ms']))} - {ms_to_local(int(seg['end_ms']))}]"
            )
        status = self.status
        if self.comment_active:
            status = f"comment: {self.comment_buffer[-120:]}"
        subtitle = (
            f"{' | '.join(mode_bits) if mode_bits else self._display_review_status(seg)} | "
            f"{str(seg.get('segment_quality_label'))} | "
            f"{self._segment_metric_summary(int(seg['segment_idx']))} | "
            f"add snap {'1ms' if self.precise_add_mode else f'{ADDED_PEAK_SNAP_WINDOW_MS}ms'} | {status}"
        )
        if review_ts is not None:
            review_bits = (
                f"review peak {seg.get('_review_peak_id')} | "
                f"{seg.get('_review_assigned_label') or ''}"
            )
            subtitle = f"{review_bits} | {subtitle}"
        if getattr(self.args, "marker_recovery", False):
            direct_actions = str(seg.get("_direct_actions") or "")
            if direct_actions:
                subtitle = f"actions {direct_actions} | {subtitle}"
        keys = (
            "left/right prev/next | z undo x redo | space reviewed | "
            "p physio, pp physio region | e PAC/PVC | j aftereffect | y arrhythmia | "
            "a add | b bad, bb bad region | v neg artifacts | c comment | q quit"
        )
        self.ax.set_title(f"{title}\n{subtitle}\n{keys}", color="#e9eef8", fontsize=14, pad=18)
        self.fig.subplots_adjust(left=0.055, right=0.99, bottom=0.08, top=0.84)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        try:
            plt.show()
        finally:
            logger = getattr(self, "action_logger", None)
            if logger is not None:
                logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4 ECG annotation viewer")
    parser.add_argument("--reviewed", action="store_true", help="Only show reviewed segments")
    parser.add_argument("--physio", action="store_true", help="Only show segments containing physio-labeled beats")
    parser.add_argument("--markers", action="store_true", help="Show real segment queues spanning ±15 segments around marker CSV timestamps")
    parser.add_argument("--marker-recovery", action="store_true", help="Review the recovered marker-session segments from Diagnostics without entering marker-center mode")
    parser.add_argument(
        "--physio-review",
        action="store_true",
        help="Review physio-constraint flagged beats from the diagnostics parquet, centered with a faint shaded beat overlay.",
    )
    parser.add_argument("--usable", action="store_true", help="Only show segments with segment_quality_label == 'usable'")
    parser.add_argument("--unclean", action="store_true", help="Only show segments with segment_quality_label == 'unclean'")
    parser.add_argument("--unusable", action="store_true", help="Only show segments with segment_quality_label == 'unusable'")
    parser.add_argument(
        "--chronological",
        dest="spread_queue",
        action="store_false",
        default=True,
        help="Show selected rows in chronological order instead of spreading them across dates.",
    )
    queue_group = parser.add_mutually_exclusive_group()
    queue_group.add_argument("--annotated", action="store_true", help="Only show segments that already contain beat/bad-region annotations")
    queue_group.add_argument("--likely-artifact", action="store_true", help="Only show unannotated segments with high RMSSD or RR < threshold")
    queue_group.add_argument("--likely-physio", action="store_true", help="Only show unannotated segments with a large positive delta-RR")
    queue_group.add_argument("--likely-clean", action="store_true", help="Only show unannotated segments not flagged by artifact/physio heuristics")
    parser.add_argument("--rmssd", type=float, default=DEFAULT_RMSSD_ARTIFACT_MS, help="RMSSD threshold for --likely-artifact")
    parser.add_argument("--min-rr", type=int, default=DEFAULT_MIN_RR_ARTIFACT_MS, help="RR intervals below this threshold flag --likely-artifact")
    parser.add_argument("--delta", type=int, default=DEFAULT_PHYSIO_DELTA_MS, help="Positive delta-RR threshold for --likely-physio")
    parser.add_argument("--start", type=int, default=0, help="0-based starting index within the selected queue")
    parser.add_argument(
        "--window-size",
        type=float,
        default=None,
        help="Visible/load window size in seconds, centered on the active segment/review beat. Default is the segment width; physio-review defaults to 12 seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for required in (ECG_PARQUET, PEAKS_PARQUET):
        if not required.exists():
            _die(f"Missing required file: {required}")
    ensure_inputs()
    require_matplotlib()
    viewer = MasterAnnotationViewer(args)
    viewer.show()


if __name__ == "__main__":
    main()
