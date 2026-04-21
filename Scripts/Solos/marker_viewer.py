#!/usr/bin/env python3
"""
marker_viewer.py — Interactive ECG browser and annotator for starred cardiac events.

Controls
--------
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
    python marker_viewer.py --processed-dir /Volumes/xHRV/processed/
    python marker_viewer.py --processed-dir /Volumes/xHRV/processed/ --start 42
    python marker_viewer.py --processed-dir /Volumes/xHRV/processed/ --order asc
    python marker_viewer.py --processed-dir /Volumes/xHRV/processed/ --annotations-only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time as _time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MARKER_CSV, SAMPLE_RATE_HZ

matplotlib.use("MacOSX")

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE  = SAMPLE_RATE_HZ
DISPLAY_MS   = 60_000
MIN_VIEW_MS  = 10_000
MAX_VIEW_MS  = 120_000
ZOOM_FACTOR  = 1.15
RR_SHOW_MS   = 30_000
LOCAL_TZ     = ZoneInfo("America/New_York")

_early_cutoff_dt = datetime(2025, 8, 1, tzinfo=LOCAL_TZ)
EARLY_CUTOFF_MS  = int(_early_cutoff_dt.astimezone(timezone.utc).timestamp() * 1000)
SEARCH_STEPS_RECENT = [5  * 60 * 1000]
SEARCH_STEPS_EARLY  = [5  * 60 * 1000,
                        10 * 60 * 1000,
                        15 * 60 * 1000]

OUTPUT_DIR = Path("/Volumes/xHRV/Accessory/marker_annotations")

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

SNAP_THRESHOLD_MS  = 2_000
PEAK_SNAP_SAMPLES  = 8
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
        return int(best["segment_idx"]), int(best["dist"] // 1000)
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
                   "view_start_ms", "view_end_ms", "annotated_at"]
    BEAT_HEADER = ["ann_id", "marker_idx", "marker_dt", "theme_id",
                   "peak_id", "peak_timestamp_ms", "annotated_at"]

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
        self.view_center_ms    : int             = 0
        self.view_width_ms     : int             = DISPLAY_MS
        self.ecg_ts            : np.ndarray      = np.array([])
        self.ecg_vals          : np.ndarray      = np.array([])
        self.peaks_df          : pd.DataFrame    = pd.DataFrame()
        self.offset_sec        : int             = 0
        self.ecg_loaded        : bool            = False

        # Interaction state machine
        self.state             : str             = "BROWSE"
        self.selected_peaks    : list[int]       = []

        # Segment click-define state
        self._seg_start_ms     : int | None      = None
        self._seg_end_ms       : int | None      = None

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

        Clusters all annotation timestamps within _ANN_MERGE_NS of each other
        into a single window; center = midpoint of the cluster.
        """
        centers: list[int] = []
        for ann in self.beat_anns:
            centers.append(int(ann["peak_timestamp_ms"]))
        for ann in self.seg_anns:
            mid = (int(ann["view_start_ms"]) + int(ann["view_end_ms"])) // 2
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
        center_ms            = self.ann_windows[idx]
        self.view_center_ms  = center_ms
        self.view_width_ms   = 60_000    # ±30 s flanks
        self.state           = "BROWSE"
        self.selected_peaks  = []
        steps = (SEARCH_STEPS_EARLY if center_ms < EARLY_CUTOFF_MS
                 else SEARCH_STEPS_RECENT)
        self.seg_idx, self.offset_sec = _find_segment(center_ms, self.segments, steps)
        self._load_window()
        self._draw()

    def _go_to_marker(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.markers):
            return
        self.marker_idx     = idx
        m                   = self.markers[idx]
        steps               = (SEARCH_STEPS_EARLY if m["ms"] < EARLY_CUTOFF_MS
                                else SEARCH_STEPS_RECENT)
        self.seg_idx, self.offset_sec = _find_segment(m["ms"], self.segments, steps)
        self.view_center_ms = m["ms"]
        self.state          = "BROWSE"
        self.selected_peaks = []
        self._load_window()
        self._draw()

    def _load_window(self) -> None:
        half = self.view_width_ms // 2
        lo   = self.view_center_ms - half
        hi   = self.view_center_ms + half

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
        half  = self.view_width_ms // 2
        lo_ms = self.view_center_ms - half
        hi_ms = self.view_center_ms + half

        self.ax.clear()
        self.ax.set_facecolor("#16213e")

        if self.ecg_loaded:
            t_sec = (self.ecg_ts - lo_ms) / 1000
            self.ax.plot(t_sec, self.ecg_vals,
                         color="#4fc3f7", linewidth=0.55, alpha=0.92)

            if len(self.peaks_df):
                pk_ms = self.peaks_df["timestamp_ms"].values.astype(np.int64)
                sn_ms, sn_y = _snap_peaks(pk_ms, self.ecg_ts, self.ecg_vals)
                self.ax.scatter((sn_ms - lo_ms) / 1000, sn_y,
                                color="#546e7a", s=18, zorder=4, alpha=0.7)

            if self.selected_peaks and len(self.peaks_df):
                sel = self.peaks_df[
                    self.peaks_df["peak_id"].isin(self.selected_peaks)
                ]
                if len(sel):
                    sel_ms = sel["timestamp_ms"].values.astype(np.int64)
                    sn_ms, sn_y = _snap_peaks(sel_ms, self.ecg_ts, self.ecg_vals)
                    self.ax.scatter((sn_ms - lo_ms) / 1000, sn_y,
                                    color="#ff6b6b", s=60, zorder=6,
                                    edgecolors="white", linewidths=0.5)
        else:
            self.ax.text(30, 0, "No ECG data in this window",
                         ha="center", va="center",
                         color="#546e7a", fontsize=14)

        view_sec = self.view_width_ms / 1000

        # Marker star reference line
        if self.markers:
            m          = self.markers[self.marker_idx]
            marker_off = (m["ms"] - lo_ms) / 1000
            if 0 <= marker_off <= view_sec:
                self.ax.axvline(marker_off, color="#ff6b6b", linewidth=1.4,
                                linestyle="--", alpha=0.8)
                self.ax.text(marker_off + view_sec * 0.005, self.ax.get_ylim()[1],
                             "★", color="#ff6b6b", fontsize=10, va="top", ha="left")

        # Segment-start anchor
        if self.state == "SEG_END" and self._seg_start_ms is not None:
            s_off = (self._seg_start_ms - lo_ms) / 1000
            if 0 <= s_off <= view_sec:
                self.ax.axvline(s_off, color="#69db7c", linewidth=2.2,
                                linestyle="-", alpha=0.9)
                self.ax.text(s_off + view_sec * 0.005, self.ax.get_ylim()[1],
                             "▶ start", color="#69db7c", fontsize=8,
                             va="top", ha="left")

        self._draw_existing_annotations(lo_ms, hi_ms,
                                        t_sec if self.ecg_loaded else None)

        # RR labels when zoomed in
        if (self.ecg_loaded and len(self.peaks_df) > 1
                and self.view_width_ms <= RR_SHOW_MS):
            ylim = self.ax.get_ylim()
            y_rr = ylim[0] + 0.97 * (ylim[1] - ylim[0])
            sorted_ts = np.sort(
                self.peaks_df["timestamp_ms"].values.astype(np.int64)
            )
            in_view = (sorted_ts >= lo_ms) & (sorted_ts <= hi_ms)
            view_ts = sorted_ts[in_view]
            for i in range(len(view_ts) - 1):
                rr_ms = int(round(view_ts[i + 1] - view_ts[i]))
                mid_t = ((view_ts[i] + view_ts[i + 1]) / 2 - lo_ms) / 1000
                self.ax.text(mid_t, y_rr, f"{rr_ms}",
                             color="#ffd43b", fontsize=7, ha="center", va="top",
                             family="monospace", zorder=9)

        self.ax.set_xlim(0, view_sec)
        self.ax.set_xlabel("seconds in window", color="#78909c", fontsize=8)
        tick_step = 10 if view_sec >= 60 else (5 if view_sec >= 30 else 2)
        tick_sec  = np.arange(0, view_sec + 0.01, tick_step)
        tick_ms   = [lo_ms + int(s * 1000) for s in tick_sec]
        self.ax.set_xticks(tick_sec)
        self.ax.set_xticklabels(
            [datetime.fromtimestamp(ms / 1000, tz=LOCAL_TZ).strftime("%H:%M:%S")
             for ms in tick_ms],
            color="#78909c", fontsize=7,
        )
        self.ax.tick_params(axis="y", colors="#546e7a", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#2a3a5c")

        # Title
        if self.annotations_only:
            dt_str = datetime.fromtimestamp(
                self.view_center_ms / 1000, tz=LOCAL_TZ
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
        self, lo_ms: int, hi_ms: int, t_sec: np.ndarray | None
    ) -> None:
        ylim  = self.ax.get_ylim()
        yspan = ylim[1] - ylim[0]

        for ann in self.seg_anns:
            ann_lo = int(ann["view_start_ms"])
            ann_hi = int(ann["view_end_ms"])
            if ann_lo > hi_ms or ann_hi < lo_ms:
                continue
            x0  = max(0, (ann_lo - lo_ms) / 1000)
            x1  = min(self.view_width_ms / 1000, (ann_hi - lo_ms) / 1000)
            tid = int(ann["theme_id"])
            c   = self._ann_color(tid)
            lbl = self._ann_short_label(tid)
            self.ax.axvspan(x0, x1, alpha=0.08, color=c, zorder=2)
            self.ax.text((x0 + x1) / 2, ylim[1] - yspan * 0.04, lbl,
                         color=c, fontsize=7, ha="center", va="top", zorder=5)

        if t_sec is not None and len(self.peaks_df):
            for ann in self.beat_anns:
                pts_ms = int(ann["peak_timestamp_ms"])
                if not (lo_ms <= pts_ms <= hi_ms):
                    continue
                sn, sy = _snap_peaks(
                    np.array([pts_ms], dtype=np.int64),
                    self.ecg_ts, self.ecg_vals,
                )
                pt  = (sn[0] - lo_ms) / 1000
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
            self.view_center_ms += int(self.view_width_ms * 0.15) * direction
            self._load_window()
            self._draw()
            return

        if event.step == 0:
            return
        scale     = ZOOM_FACTOR ** (-event.step)
        new_width = int(self.view_width_ms * scale)
        new_width = max(MIN_VIEW_MS, min(MAX_VIEW_MS, new_width))
        if new_width == self.view_width_ms:
            return

        if event.xdata is not None:
            view_sec    = self.view_width_ms / 1000
            cursor_frac = max(0.0, min(1.0, event.xdata / view_sec))
            lo_ms       = self.view_center_ms - self.view_width_ms // 2
            cursor_ms   = lo_ms + int(cursor_frac * self.view_width_ms)
            new_lo_ms   = cursor_ms - int(cursor_frac * new_width)
            self.view_center_ms = new_lo_ms + new_width // 2

        self.view_width_ms = new_width
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
                self.view_center_ms += self.view_width_ms // 5
                self._load_window(); self._draw()
            elif k == "left":
                self.view_center_ms -= self.view_width_ms // 5
                self._load_window(); self._draw()
            elif k == "t":
                self._seg_start_ms = None
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
                self._seg_start_ms = None
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

        lo_ms    = self.view_center_ms - self.view_width_ms // 2
        click_ms = lo_ms + int(event.xdata * 1000)

        if self.state == "BROWSE":
            if len(self.peaks_df) == 0:
                return
            dists     = (self.peaks_df["timestamp_ms"] - click_ms).abs()
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
                dist = abs(int(ann["peak_timestamp_ms"]) - click_ms)
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
            self._seg_start_ms = click_ms
            self.state = "SEG_END"
            self._draw()

        elif self.state == "SEG_END":
            self._seg_end_ms = click_ms
            if self._seg_end_ms < self._seg_start_ms:
                self._seg_start_ms, self._seg_end_ms = (
                    self._seg_end_ms, self._seg_start_ms
                )
            self._tag_assign_buffer = ""
            self.state = "AWAIT_SEG_THEME"
            self._draw()

    # ── Annotation persistence ────────────────────────────────────────────────

    def _current_marker_info(self) -> tuple[int, str]:
        """Return (marker_idx_1indexed, marker_dt_str) for the current context."""
        if self.annotations_only:
            dt_str = datetime.fromtimestamp(
                self.view_center_ms / 1000, tz=LOCAL_TZ
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
            "view_start_ms": self._seg_start_ms,
            "view_end_ms":   self._seg_end_ms,
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
        dur_ms = self._seg_end_ms - self._seg_start_ms
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
                pts_ms = int(prow.iloc[0]["timestamp_ms"])
                row = {
                    "ann_id":            self._next_ann_id,
                    "marker_idx":        midx,
                    "marker_dt":         mdt,
                    "theme_id":          theme_id,
                    "peak_id":           pid,
                    "peak_timestamp_ms": pts_ms,
                    "annotated_at":      now,
                }
                self.beat_anns.append(row)
                self._next_ann_id += 1
                w.writerow(row)
        print(f"  [B] {len(self.selected_peaks)} beat(s) "
              f"[{self._ann_full_label(theme_id)}] → marker {midx}")

    def _delete_last_annotation(self) -> None:
        lo_ms = self.view_center_ms - self.view_width_ms // 2
        hi_ms = self.view_center_ms + self.view_width_ms // 2
        deleted = False
        for ann_list, atype in [(self.beat_anns, "beat"), (self.seg_anns, "seg")]:
            for i in range(len(ann_list) - 1, -1, -1):
                ann = ann_list[i]
                if atype == "beat":
                    if not (lo_ms <= int(ann["peak_timestamp_ms"]) <= hi_ms):
                        continue
                else:
                    if (int(ann["view_end_ms"]) < lo_ms
                            or int(ann["view_start_ms"]) > hi_ms):
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
        lo_ms = self.view_center_ms - self.view_width_ms // 2
        hi_ms = self.view_center_ms + self.view_width_ms // 2
        if beats:
            before = len(self.beat_anns)
            self.beat_anns = [
                a for a in self.beat_anns
                if not (lo_ms <= int(a["peak_timestamp_ms"]) <= hi_ms)
            ]
            print(f"  [Clear] {before - len(self.beat_anns)} beat ann(s) removed.")
        if segments:
            before = len(self.seg_anns)
            self.seg_anns = [
                a for a in self.seg_anns
                if not (int(a["view_start_ms"]) <= hi_ms
                        and int(a["view_end_ms"]) >= lo_ms)
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

            ts_pts = [int(a["peak_timestamp_ms"]) for a in b_anns]
            for a in s_anns:
                ts_pts += [int(a["view_start_ms"]), int(a["view_end_ms"])]
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
                    np.array([int(ann["peak_timestamp_ms"])], dtype=np.int64),
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
    parser = argparse.ArgumentParser(description="Marker ECG annotator")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--start", type=int, default=1,
                        help="Start at marker N (1-indexed)")
    parser.add_argument("--order", choices=["asc", "desc"], default="desc",
                        help="asc = oldest first; desc = newest first (default)")
    parser.add_argument("--annotations-only", action="store_true",
                        help="Navigate only annotated ECG regions (± 30 s context)")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
