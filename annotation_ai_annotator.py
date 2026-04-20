#!/usr/bin/env python3
"""
annotation_ai_annotator.py — AI-assisted annotation of human-annotated cardiac events.

Beat annotations are CLUSTERED before being sent to Gemini: consecutive annotations
within --gap-sec of each other form a single event (PAC + Rebound = one call, not two).
Segment annotations remain individual events, rendered as shaded regions.

  1. Load cardiac beat and segment annotations (exclude artifact / bad-ECG theme_ids)
  2. Cluster beat annotations: new cluster when gap between consecutive timestamps > gap-sec
  3. For each cluster/segment: render dual-panel PNG (context ±45s | adaptive zoom)
     with RR interval ms overlaid as colour-coded text between R-peaks
  4. Send structured text payload to Gemini CLI; append result to event_ai_annotations.csv
  5. Re-running resumes automatically (event_id = first ann_id in cluster, stable key)

Usage:
    python annotation_ai_annotator.py
    python annotation_ai_annotator.py --max-calls 50
    python annotation_ai_annotator.py --flank-sec 10 --gap-sec 5
    python annotation_ai_annotator.py --flank-sec 10 --context-sec 45
    python annotation_ai_annotator.py --render-only   # PNGs only, no Gemini
    python annotation_ai_annotator.py --list-events   # show event table and exit
"""

import argparse
import csv
import datetime
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths & defaults ──────────────────────────────────────────────────────────

DEFAULT_ECG_PARQUET     = Path("/Volumes/xHRV/processed/ecg_samples.parquet")
DEFAULT_PEAKS_PARQUET   = Path("/Volumes/xHRV/processed/peaks.parquet")
DEFAULT_OUTPUT_CSV      = Path("/Volumes/xHRV/Accessory/marker_annotations/event_ai_annotations.csv")
DEFAULT_PNG_DIR         = Path("/Volumes/xHRV/Accessory/marker_annotations/event_ai_pngs")
DEFAULT_BEAT_ANNS_CSV   = Path("/Volumes/xHRV/Accessory/marker_annotations/beat_annotations.csv")
DEFAULT_SEG_ANNS_CSV    = Path("/Volumes/xHRV/Accessory/marker_annotations/segment_annotations.csv")
DEFAULT_TAG_LABELS_JSON = Path("/Volumes/xHRV/Accessory/marker_annotations/tag_labels.json")
DEFAULT_MODEL           = "gemini-3.1-pro"
DEFAULT_FLANK_BEATS   = 15    # beats shown on each side of the cluster (zoom panel)
DEFAULT_GAP_BEATS     = 4     # gap > N beats between annotations starts a new cluster
DEFAULT_CONTEXT_BEATS = 50    # beats shown on each side in wide-context panel (0 = single)
DEFAULT_DELAY_SEC       = 2.0
DEFAULT_MAX_CALLS       = 100

# ── Two annotation systems coexist in beat_annotations.csv ───────────────────
#
# V1  (raw theme_id 1–9):  original pass; meanings differ from V2 slot numbers.
#   1=Contraction  2=Expansion  3=Rhythmic/RSA/Vagal  4=Tachy↔Normal
#   5=PAC/PVC  6=BeatAnomaly  7=RSA-like  8=Artifact  9=CardiacAnomaly
#
# V2  (theme_id = 100 + slot, i.e. 101–120):  current system via marker_viewer.
#   slot meanings come from tag_labels.json (NormToNorm, Expansion, Contraction, …)
#
# V1 theme_id 8 = Artifact.  Even though tag_labels.json now maps slot 8 → "Triplet",
# those 427 V1 rows pre-date that change.  Exclude raw 8 as artifact.
# V2 theme_id 120 = Artifact (100+20).  V2 theme_id 118 = Bad ECG (100+18).
# Do NOT exclude 108 (V2 Triplet — genuine cardiac events).

# V1 tag names — historically accurate labels for raw theme_id 1–9.
# These OVERRIDE the V2 slot names that tag_labels.json would otherwise assign.
_V1_THEME_NAMES: dict[int, str] = {
    1: "V1:Contraction",
    2: "V1:Expansion",
    3: "V1:Rhythmic/RSA/Vagal",
    4: "V1:Tachy↔Normal",
    5: "V1:PAC/PVC",
    6: "V1:BeatAnomaly",
    7: "V1:RSA-like",
    # 8 = V1 Artifact — excluded, intentionally absent from this dict
    9: "V1:CardiacAnomaly",
}

_ARTIFACT_THEME_IDS:     frozenset[int] = frozenset({8, 120})   # V1 Artifact, V2 Artifact
_BAD_ECG_THEME_IDS:      frozenset[int] = frozenset({18, 118})  # V1 none, V2 Bad ECG
_NON_CARDIAC_THEME_IDS:  frozenset[int] = _ARTIFACT_THEME_IDS | _BAD_ECG_THEME_IDS

_NODE_CANDIDATES = [
    "/opt/homebrew/opt/node/bin/node",
    "/opt/homebrew/opt/node@23/bin/node",
    "/opt/homebrew/opt/node@24/bin/node",
    "/opt/homebrew/opt/node@25/bin/node",
    "/usr/local/bin/node",
]
_GEMINI_CLI = "/opt/homebrew/bin/gemini"

CSV_COLUMNS = [
    "event_id",              # first ann_id in cluster (stable resume key)
    "ann_type",              # "beat_cluster" or "seg"
    "event_center_datetime", # UTC human-readable
    "event_center_ns",       # UTC nanoseconds
    "human_tags",            # all unique tags in this cluster (semicolon-joined)
    "n_annotations",         # number of beat annotations in cluster
    "nearby_tags",           # non-cluster annotations within window (semicolon list)
    "event_type",
    "sub_type",
    "confidence",
    "rr_pattern",
    "hr_min_bpm",
    "hr_max_bpm",
    "key_features",
    "ecg_quality",
    "likely_physiological",
    "notes",
    "model_used",
    "processed_at",
    "png_path",
    "error",
]

# ── Tag label loader ──────────────────────────────────────────────────────────

def load_tag_names(tag_json: Path) -> dict[int, str]:
    """Build a theme_id → display name mapping.

    tag_labels.json defines V2 slot numbers (1–20).  The +100 auto-alias generates
    the V2 storage keys (101–120).  Raw theme_id 1–9 are V1 annotations whose
    meanings differ from V2 slot numbers, so they are overridden with historically
    accurate "V1:X" labels from _V1_THEME_NAMES.
    """
    tag_names: dict[int, str] = {}
    if tag_json.exists():
        with open(tag_json) as f:
            raw = json.load(f)
        tag_names = {int(k): v for k, v in raw.items()}
        for base_id, name in list(tag_names.items()):
            tag_names.setdefault(base_id + 100, f"L{base_id}:{name}")
    # V1 annotations (raw theme_id 1–9) have different meanings than V2 slot names.
    # Override so they carry historically accurate labels, not V2 slot names.
    tag_names.update(_V1_THEME_NAMES)
    return tag_names


# ── Annotation loading ────────────────────────────────────────────────────────

def load_cardiac_beat_annotations(beat_csv: Path, tag_names: dict[int, str]) -> pd.DataFrame:
    """Load beat_annotations.csv, cardiac-only rows, sorted by timestamp."""
    if not beat_csv.exists():
        logger.warning("beat_annotations.csv not found: %s", beat_csv)
        return pd.DataFrame()
    try:
        df = pd.read_csv(beat_csv)
    except Exception as e:
        logger.error("Could not read beat_annotations.csv: %s", e)
        return pd.DataFrame()
    df = df[~df["theme_id"].isin(_NON_CARDIAC_THEME_IDS)].copy()
    df["tag_name"] = df["theme_id"].map(lambda tid: tag_names.get(int(tid), f"tag{tid}"))
    df["peak_timestamp_ns"] = df["peak_timestamp_ns"].astype("int64")
    df = df.sort_values("peak_timestamp_ns").reset_index(drop=True)
    logger.info("Beat annotations: %d cardiac rows loaded", len(df))
    return df


def load_cardiac_seg_annotations(seg_csv: Path, tag_names: dict[int, str]) -> pd.DataFrame:
    """Load segment_annotations.csv, cardiac-only rows."""
    if not seg_csv.exists():
        logger.warning("segment_annotations.csv not found: %s", seg_csv)
        return pd.DataFrame()
    try:
        df = pd.read_csv(seg_csv)
    except Exception as e:
        logger.error("Could not read segment_annotations.csv: %s", e)
        return pd.DataFrame()
    df = df[~df["theme_id"].isin(_NON_CARDIAC_THEME_IDS)].copy()
    df["tag_name"] = df["theme_id"].map(lambda tid: tag_names.get(int(tid), f"tag{tid}"))
    df["view_start_ns"] = df["view_start_ns"].astype("int64")
    df["view_end_ns"]   = df["view_end_ns"].astype("int64")
    logger.info("Segment annotations: %d cardiac rows loaded", len(df))
    return df


# ── Event list builder ────────────────────────────────────────────────────────

def build_event_list(
    beat_df:  pd.DataFrame,
    seg_df:   pd.DataFrame,
    gap_sec:  float = 5.0,
) -> list[dict]:
    """Build one event per physiological occurrence.

    Beat annotations are grouped into clusters: a new cluster starts when
    consecutive annotation timestamps are more than gap_sec apart.  This turns
    e.g. [PAC, PAC, Rebound] — three annotations 0.8 s apart — into a single
    event rather than three separate Gemini calls.

    Segment annotations (shaded regions) each remain their own event.

    Event dict keys:
      event_id       : first ann_id in cluster (stable resume key across re-runs)
      ann_type       : "beat_cluster" | "seg"
      center_ns      : midpoint of cluster's timestamp range
      zoom_ns        : half-width of cluster span (0 for single-beat clusters)
      tag_name       : semicolon-joined unique tags in this cluster
      beat_anns      : list of annotation dicts (all beats in cluster)
      cluster_ids    : set of all ann_ids in cluster (for nearby-exclusion)
      span_start_ns  : (seg only) view_start_ns
      span_end_ns    : (seg only) view_end_ns
    """
    events: list[dict] = []

    # ── Beat clusters ─────────────────────────────────────────────────────────
    if len(beat_df):
        sorted_beats = beat_df.sort_values("peak_timestamp_ns").reset_index(drop=True)
        gap_ns = int(gap_sec * 1e9)
        current: list[dict] = []

        def _flush(cluster: list[dict]) -> None:
            if not cluster:
                return
            timestamps = [r["peak_timestamp_ns"] for r in cluster]
            t_min, t_max = min(timestamps), max(timestamps)
            center_ns = (t_min + t_max) // 2
            # unique tags preserving first-seen order
            seen: dict[str, None] = {}
            for r in cluster:
                seen[r["tag_name"]] = None
            tag_name = "; ".join(seen)
            events.append({
                "event_id":    cluster[0]["ann_id"],
                "ann_type":    "beat_cluster",
                "center_ns":   center_ns,
                "zoom_ns":     (t_max - t_min) // 2,   # half-span of cluster
                "tag_name":    tag_name,
                "beat_anns":   cluster,
                "cluster_ids": {r["ann_id"] for r in cluster},
            })

        for row in sorted_beats.itertuples(index=False):
            rec = row._asdict()
            rec["peak_timestamp_ns"] = int(rec["peak_timestamp_ns"])
            rec["ann_id"]            = int(rec["ann_id"])
            if not current or rec["peak_timestamp_ns"] - current[-1]["peak_timestamp_ns"] <= gap_ns:
                current.append(rec)
            else:
                _flush(current)
                current = [rec]
        _flush(current)

    # ── Segment events ────────────────────────────────────────────────────────
    if len(seg_df):
        for row in seg_df.itertuples(index=False):
            start_ns = int(row.view_start_ns)
            end_ns   = int(row.view_end_ns)
            mid_ns   = (start_ns + end_ns) // 2
            events.append({
                "event_id":      int(row.ann_id),
                "ann_type":      "seg",
                "center_ns":     mid_ns,
                "zoom_ns":       (end_ns - start_ns) // 2,
                "tag_name":      row.tag_name,
                "beat_anns":     [],
                "cluster_ids":   {int(row.ann_id)},
                "span_start_ns": start_ns,
                "span_end_ns":   end_ns,
            })

    events.sort(key=lambda e: e["center_ns"])

    n_beat_clusters = sum(1 for e in events if e["ann_type"] == "beat_cluster")
    n_seg           = sum(1 for e in events if e["ann_type"] == "seg")
    logger.info(
        "Beat annotations: %d → %d cluster(s) (gap %.0fs)  |  Segment events: %d  |  Total: %d",
        len(beat_df), n_beat_clusters, gap_sec, n_seg, len(events),
    )
    return events


def find_nearby_annotations(
    center_ns:    int,
    window_sec:   float,
    beat_df:      pd.DataFrame,
    seg_df:       pd.DataFrame,
    excluded_ids: set[int],     # all ann_ids already in this event's cluster
) -> list[dict]:
    """Return annotations within ±window_sec that are NOT part of this cluster.

    Each result dict: {t_start, t_end, tag, ann_type}
      beats: t_start == t_end (point)
      segs:  t_start/t_end span the annotated region
    """
    window_ns = int(window_sec * 1e9)
    results: list[dict] = []

    if len(beat_df):
        nearby = beat_df[
            (beat_df["peak_timestamp_ns"] >= center_ns - window_ns) &
            (beat_df["peak_timestamp_ns"] <= center_ns + window_ns) &
            (~beat_df["ann_id"].isin(excluded_ids))
        ]
        for row in nearby.itertuples(index=False):
            t = (int(row.peak_timestamp_ns) - center_ns) / 1e9
            results.append({"t_start": t, "t_end": t,
                            "tag": row.tag_name, "ann_type": "beat"})

    if len(seg_df):
        nearby_s = seg_df[
            (seg_df["view_start_ns"] <= center_ns + window_ns) &
            (seg_df["view_end_ns"]   >= center_ns - window_ns) &
            (~seg_df["ann_id"].isin(excluded_ids))
        ]
        for row in nearby_s.itertuples(index=False):
            t_start = (int(row.view_start_ns) - center_ns) / 1e9
            t_end   = (int(row.view_end_ns)   - center_ns) / 1e9
            results.append({"t_start": t_start, "t_end": t_end,
                            "tag": row.tag_name, "ann_type": "seg"})

    results.sort(key=lambda x: x["t_start"])
    return results


# ── Beats-based window helpers ────────────────────────────────────────────────

def compute_global_median_rr(peaks_path: Path) -> float:
    """Return global median RR interval in seconds from the full peaks file.

    Used only as a fallback / initial estimate for coarse fetch sizing and
    gap-beat → gap-second conversion.  Per-event zoom windows use actual local peaks.
    """
    try:
        tbl = pq.read_table(peaks_path, columns=["timestamp_ns"])
        ts  = tbl.to_pandas()["timestamp_ns"].sort_values().values
        if len(ts) < 2:
            return 0.75
        rr_s = np.diff(ts.astype("float64")) / 1e9
        rr_s = rr_s[(rr_s >= 0.20) & (rr_s <= 3.0)]   # 20 – 300 bpm
        return float(np.median(rr_s)) if len(rr_s) else 0.75
    except Exception as e:
        logger.warning("Could not compute global median RR: %s", e)
        return 0.75


def compute_zoom_from_peaks(
    center_ns:    int,
    c_start_ns:   int,   # earliest timestamp in the cluster
    c_end_ns:     int,   # latest  timestamp in the cluster
    peaks_df:     pd.DataFrame,
    n_beats:      int,
    fallback_rr:  float = 0.75,
) -> float:
    """Return zoom half-window (seconds) so that n_beats appear on each side of the cluster.

    Finds the n_beats-th peak to the left of c_start_ns and to the right of c_end_ns,
    then returns the larger of the two distances to center_ns.  This ensures the cluster
    itself is always fully visible with n_beats of context on both sides.
    """
    if peaks_df.empty or n_beats <= 0:
        return max(n_beats * fallback_rr, (c_end_ns - c_start_ns) / 2e9 + fallback_rr)

    ts          = peaks_df["timestamp_ns"].sort_values().values
    left_peaks  = ts[ts < c_start_ns]
    right_peaks = ts[ts > c_end_ns]

    left_ns  = left_peaks[-n_beats]  if len(left_peaks)  >= n_beats else (left_peaks[0]  if len(left_peaks)  else c_start_ns)
    right_ns = right_peaks[n_beats - 1] if len(right_peaks) >= n_beats else (right_peaks[-1] if len(right_peaks) else c_end_ns)

    left_sec  = (center_ns - left_ns)  / 1e9
    right_sec = (right_ns  - center_ns) / 1e9
    return max(left_sec, right_sec, fallback_rr)


# ── CLI discovery ─────────────────────────────────────────────────────────────

def find_node() -> str:
    for candidate in _NODE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Node.js binary not found. Tried: " + ", ".join(_NODE_CANDIDATES))


def find_gemini_cli() -> str:
    if not Path(_GEMINI_CLI).exists():
        raise FileNotFoundError(f"Gemini CLI not found at {_GEMINI_CLI}")
    return _GEMINI_CLI


# ── Resume logic ──────────────────────────────────────────────────────────────

def load_existing_results(output_csv: Path) -> set[int]:
    if not output_csv.exists():
        return set()
    done: set[int] = set()
    n_errors = 0
    with open(output_csv) as f:
        for row in csv.DictReader(f):
            eid = row.get("event_id")
            if not eid:
                continue
            if row.get("error", "").strip():
                n_errors += 1
            else:
                done.add(int(eid))
    logger.info(
        "Resuming: %d events already done, %d errored rows will be retried",
        len(done), n_errors,
    )
    return done


def purge_errored_rows(output_csv: Path) -> int:
    if not output_csv.exists():
        return 0
    kept: list[dict] = []
    removed = 0
    with open(output_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error", "").strip():
                removed += 1
            else:
                kept.append(row)
    if removed == 0:
        return 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in CSV_COLUMNS} for r in kept)
    logger.info("Purged %d errored row(s) from CSV (will be retried)", removed)
    return removed


# ── Data extraction ───────────────────────────────────────────────────────────

def extract_ecg_window(ecg_path: Path, t_ns: int, window_sec: float) -> pd.DataFrame:
    half = int(window_sec * 1e9)
    try:
        tbl = pq.read_table(
            ecg_path,
            columns=["timestamp_ns", "ecg"],
            filters=[
                ("timestamp_ns", ">=", t_ns - half),
                ("timestamp_ns", "<=", t_ns + half),
            ],
        )
        return tbl.to_pandas().sort_values("timestamp_ns").reset_index(drop=True)
    except Exception as e:
        logger.warning("ECG extraction failed: %s", e)
        return pd.DataFrame(columns=["timestamp_ns", "ecg"])


def extract_peaks_window(peaks_path: Path, t_ns: int, window_sec: float) -> pd.DataFrame:
    half = int(window_sec * 1e9)
    try:
        tbl = pq.read_table(
            peaks_path,
            columns=["peak_id", "timestamp_ns"],
            filters=[
                ("timestamp_ns", ">=", t_ns - half),
                ("timestamp_ns", "<=", t_ns + half),
            ],
        )
        df = tbl.to_pandas().sort_values("timestamp_ns").reset_index(drop=True)
        df["rr_ms"] = df["timestamp_ns"].diff() / 1_000_000.0
        return df
    except Exception as e:
        logger.warning("Peaks extraction failed: %s", e)
        return pd.DataFrame(columns=["peak_id", "timestamp_ns", "rr_ms"])


# ── PNG rendering ─────────────────────────────────────────────────────────────

def _draw_ecg_with_rr_text(
    ax,
    ecg_t:             np.ndarray,
    ecg_v:             np.ndarray,
    peaks_df:          pd.DataFrame,
    center_ns:         int,
    xlim:              float,
    tag_name:          str,
    nearby:            list[dict],
    _dark:             str,
    _text:             str,
    _grid:             str,
    show_rr_text:      bool  = True,
    primary_span:      tuple[float, float] | None = None,  # seg events
    cluster_beat_anns: list[dict] | None = None,           # beat_cluster events
    fontsize_rr:       float = 8.0,
) -> None:
    """Draw one ECG panel.

    Segments are drawn as filled spans (`axvspan`), beats as thin vertical lines.
    RR interval values are optionally printed as coloured text between consecutive peaks.
    No t=0 centre line is drawn — the annotation markers themselves locate the event.
    """
    ax.set_facecolor(_dark)
    ax.tick_params(colors=_text, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2d2d4e")
    ax.grid(True, color=_grid, linewidth=0.5, linestyle="--")

    ax.set_xlim(-xlim, xlim)

    # Y limits — establish first so span/text positions are correct
    if len(ecg_v):
        p1, p99 = np.percentile(ecg_v, [1, 99])
        margin = max(0.1, (p99 - p1) * 0.15)
        ymin = p1 - margin
        ymax = p99 + margin
    else:
        ymin, ymax = -1.0, 2.0
    ax.set_ylim(ymin, ymax)

    # ── Draw nearby annotations BEFORE the ECG so waveform sits on top ────────
    # Segment annotations: filled span across their full annotated region.
    # Beat annotations: a single vertical line.
    # Both: label text near the top of the axes.
    _seg_colors  = ["#9b59b6", "#1abc9c", "#e67e22", "#e74c3c", "#3498db"]
    _beat_color  = "#cc44ff"
    seg_color_idx = 0

    for ann in nearby:
        t_s   = ann["t_start"]
        t_e   = ann["t_end"]
        tag   = ann["tag"]
        atype = ann["ann_type"]

        if atype == "seg":
            clr = _seg_colors[seg_color_idx % len(_seg_colors)]
            seg_color_idx += 1
            # Clip span to visible range (axvspan clips automatically but we also
            # want the label to sit within the visible portion)
            visible_start = max(t_s, -xlim)
            visible_end   = min(t_e,  xlim)
            if visible_end > visible_start:
                ax.axvspan(visible_start, visible_end,
                           color=clr, alpha=0.18, zorder=0)
                # Boundary lines at the true edges (if inside the window)
                if -xlim <= t_s <= xlim:
                    ax.axvline(t_s, color=clr, lw=1.0, ls="--", alpha=0.7)
                if -xlim <= t_e <= xlim:
                    ax.axvline(t_e, color=clr, lw=1.0, ls="--", alpha=0.7)
                # Label at the centre of the VISIBLE portion
                label_x = (visible_start + visible_end) / 2
                ax.text(label_x, 0.97, tag,
                        transform=ax.get_xaxis_transform(),
                        color=clr, fontsize=8, ha="center", va="top",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=_dark,
                                  alpha=0.75, edgecolor=clr, linewidth=0.6),
                        clip_on=True)
        else:  # beat
            if -xlim <= t_s <= xlim:
                ax.axvline(t_s, color=_beat_color, lw=1.0, ls=":", alpha=0.75)
                ax.text(t_s, 0.97, tag,
                        transform=ax.get_xaxis_transform(),
                        color=_beat_color, fontsize=7.5, ha="center", va="top",
                        rotation=90,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor=_dark,
                                  alpha=0.75, edgecolor=_beat_color, linewidth=0.5),
                        clip_on=True)

    # ── Primary annotation(s): the event cluster itself ───────────────────────
    if primary_span is not None:
        # Segment event — orange filled region with boundary lines
        t_s, t_e = primary_span
        vis_s = max(t_s, -xlim)
        vis_e = min(t_e,  xlim)
        if vis_e > vis_s:
            ax.axvspan(vis_s, vis_e, color="#ff9944", alpha=0.20, zorder=1)
        if -xlim <= t_s <= xlim:
            ax.axvline(t_s, color="#ff9944", lw=1.5, alpha=0.9, zorder=2)
        if -xlim <= t_e <= xlim:
            ax.axvline(t_e, color="#ff9944", lw=1.5, alpha=0.9, zorder=2)
        label_x = (max(vis_s, -xlim) + min(vis_e, xlim)) / 2
        ax.text(label_x, 0.02, f"▶ {tag_name}",
                transform=ax.get_xaxis_transform(),
                color="#ff9944", fontsize=9, ha="center", va="bottom",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=_dark,
                          alpha=0.85, edgecolor="#ff9944", linewidth=0.8),
                clip_on=True)

    elif cluster_beat_anns:
        # Beat cluster — one orange line per beat in the cluster
        for ann in cluster_beat_anns:
            t_ann = (int(ann["peak_timestamp_ns"]) - center_ns) / 1e9
            if -xlim <= t_ann <= xlim:
                ax.axvline(t_ann, color="#ff9944", lw=1.4, alpha=0.85, zorder=2)
        # Single label centred on the cluster midpoint
        t_positions = [
            (int(a["peak_timestamp_ns"]) - center_ns) / 1e9
            for a in cluster_beat_anns
            if -xlim <= (int(a["peak_timestamp_ns"]) - center_ns) / 1e9 <= xlim
        ]
        if t_positions:
            label_x = (min(t_positions) + max(t_positions)) / 2
            ax.text(label_x, 0.02, f"▶ {tag_name}",
                    transform=ax.get_xaxis_transform(),
                    color="#ff9944", fontsize=9, ha="center", va="bottom",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=_dark,
                              alpha=0.85, edgecolor="#ff9944", linewidth=0.8),
                    clip_on=True)

    # ── ECG waveform (drawn after spans so it sits on top) ─────────────────────
    if len(ecg_t):
        ax.plot(ecg_t, ecg_v, lw=0.7, color="#4a9eff", alpha=0.92, zorder=3)

    # ── RR interval text between consecutive R-peaks ──────────────────────────
    if show_rr_text:
        valid = peaks_df.dropna(subset=["rr_ms"]).copy()
        valid["t_rel"] = (valid["timestamp_ns"].values - center_ns) / 1e9
        rr_y = ymin + (ymax - ymin) * 0.03

        for row in valid.itertuples(index=False):
            t_curr = row.t_rel
            rr     = row.rr_ms
            t_prev = t_curr - rr / 1000.0
            mid_t  = (t_prev + t_curr) / 2.0
            if abs(mid_t) > xlim:
                continue
            if rr > 1500:   clr = "#ff4444"
            elif rr > 1200: clr = "#ff8844"
            elif rr < 300:  clr = "#ff4444"
            elif rr < 400:  clr = "#ffaa00"
            else:           clr = "#f4d03f"
            ax.text(
                mid_t, rr_y, f"{rr:.0f}",
                color=clr, fontsize=fontsize_rr, ha="center", va="bottom",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", facecolor=_dark,
                          alpha=0.65, edgecolor="none"),
                clip_on=True,
            )

    ax.set_ylabel("ECG (mV)", color=_text, fontsize=10)
    ax.set_xlabel("Time relative to annotated event (s)", color=_text, fontsize=10)


def render_png(
    ecg_df:            pd.DataFrame,
    peaks_df:          pd.DataFrame,
    center_ns:         int,
    event_id:          int,
    center_datetime:   str,
    zoom_sec:          float,              # half-width of zoom panel
    output_path:       Path,
    tag_name:          str = "",
    nearby:            list[dict] | None = None,
    context_sec:       float | None = None,
    primary_span:      tuple[float, float] | None = None,
    cluster_beat_anns: list[dict] | None = None,
) -> None:
    """Render event PNG.

    Single-panel (ECG only) by default.  When context_sec > window_sec, the figure
    becomes side-by-side: wide ECG (±context_sec) | zoomed ECG (±window_sec).
    RR interval values are printed as coloured text between each pair of R-peaks.
    """
    _dark = "#0f0f1a"; _grid = "#1e1e38"; _text = "#a0a0c0"
    nearby = nearby or []

    ecg_t = (ecg_df["timestamp_ns"].values - center_ns) / 1e9 if len(ecg_df) else np.array([])
    ecg_v =  ecg_df["ecg"].values                              if len(ecg_df) else np.array([])

    dual = context_sec is not None and context_sec > zoom_sec

    if dual:
        fig, (ax_wide, ax_zoom) = plt.subplots(
            1, 2, figsize=(24, 7), facecolor=_dark,
            gridspec_kw={"wspace": 0.06},
        )
        # Wide panel: no RR text (beats too dense at ±45s)
        _draw_ecg_with_rr_text(
            ax_wide, ecg_t, ecg_v, peaks_df, center_ns,
            context_sec, tag_name, nearby, _dark, _text, _grid,
            show_rr_text=False,
            primary_span=primary_span,
            cluster_beat_anns=cluster_beat_anns,
            fontsize_rr=8.0,
        )
        # Zoom panel: RR text + full cluster rendering
        _draw_ecg_with_rr_text(
            ax_zoom, ecg_t, ecg_v, peaks_df, center_ns,
            zoom_sec, tag_name, nearby, _dark, _text, _grid,
            show_rr_text=True,
            primary_span=primary_span,
            cluster_beat_anns=cluster_beat_anns,
            fontsize_rr=9.0,
        )
        ax_wide.set_ylabel("ECG (mV)  — context", color=_text, fontsize=9)
        ax_zoom.set_ylabel("ECG (mV)  — zoom",    color=_text, fontsize=9)
        ax_zoom.tick_params(labelleft=False)
        title_suffix = f"[context ±{context_sec:.0f}s | zoom ±{zoom_sec:.0f}s]"
    else:
        fig, ax = plt.subplots(1, 1, figsize=(18, 7), facecolor=_dark)
        _draw_ecg_with_rr_text(
            ax, ecg_t, ecg_v, peaks_df, center_ns,
            zoom_sec, tag_name, nearby, _dark, _text, _grid,
            show_rr_text=True,
            primary_span=primary_span,
            cluster_beat_anns=cluster_beat_anns,
            fontsize_rr=9.0,
        )
        title_suffix = f"±{zoom_sec:.0f}s"

    n_beats = len(peaks_df.dropna(subset=["rr_ms"]))
    hr_med_str = "?"
    valid_rr = peaks_df["rr_ms"].dropna().values
    if len(valid_rr):
        hr_med_str = f"{60000 / float(np.median(valid_rr)):.0f}"

    fig.suptitle(
        f"Event #{event_id}  [{tag_name}]  —  {center_datetime} UTC  "
        f"({n_beats} beats,  median HR ≈ {hr_med_str} bpm)  {title_suffix}",
        color=_text, fontsize=12, y=0.99,
    )
    fig.patch.set_facecolor(_dark)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=_dark, bbox_inches="tight")
    plt.close(fig)


# ── Data text builder ─────────────────────────────────────────────────────────

def build_data_text(
    ecg_df:          pd.DataFrame,
    peaks_df:        pd.DataFrame,
    center_ns:       int,
    event_id:        int,
    center_datetime: str,
    window_sec:      float,
    tag_name:        str,
    ann_type:        str,
    nearby:          list[dict],
    beat_anns:       list[dict] | None = None,  # all beats in the cluster
    primary_span:    tuple[float, float] | None = None,
) -> str:
    """Build the stdin payload for the Gemini CLI."""
    lines: list[str] = []

    if ann_type == "beat_cluster":
        n = len(beat_anns) if beat_anns else 1
        lines.append(
            f"BEAT CLUSTER (event #{event_id}, {n} annotation{'s' if n != 1 else ''})  "
            f"—  {center_datetime} UTC"
        )
        lines.append(f"Patient tags in this cluster:  {tag_name}")
        lines.append(
            f"t=0 is the midpoint of the cluster's time span.  "
            f"Context window: ±{window_sec:.0f} s"
        )
    else:
        lines.append(
            f"SEGMENT ANNOTATION (event #{event_id})  —  {center_datetime} UTC"
        )
        lines.append(f"Patient tag:  {tag_name}")
        if primary_span:
            dur = primary_span[1] - primary_span[0]
            lines.append(
                f"This annotation covers a region from t={primary_span[0]:+.1f}s to "
                f"t={primary_span[1]:+.1f}s ({dur:.1f}s span).  "
                f"Context window: ±{window_sec:.0f} s"
            )
    lines.append("")

    # ── RR sequence ───────────────────────────────────────────────────────────
    valid = peaks_df.dropna(subset=["rr_ms"]).copy()
    valid["t_rel_s"] = (valid["timestamp_ns"].values - center_ns) / 1e9

    lines.append("=== RR INTERVAL SEQUENCE ===")
    lines.append("Format:  t(s)   RR(ms)   HR(bpm)   note")

    if len(valid):
        center_idx = (valid["t_rel_s"].abs()).idxmin()
        for row_idx, row in valid.iterrows():
            t   = row["t_rel_s"]
            rr  = row["rr_ms"]
            hr  = round(60_000 / rr, 1) if rr > 0 else 0
            note = "  ← THIS ANNOTATION" if row_idx == center_idx else ""
            if rr > 1200:
                note += f"  ← LONG PAUSE ({rr/1000:.2f}s)"
            elif rr < 350:
                note += "  ← VERY FAST"
            lines.append(f"  {t:+7.2f}   {rr:6.0f}   {hr:5.1f}{note}")
    else:
        lines.append("  (no peaks found in window)")

    lines.append("")

    # ── Statistics ────────────────────────────────────────────────────────────
    lines.append("=== STATISTICS ===")
    if len(valid) >= 2:
        rr_vals   = valid["rr_ms"].values
        t_vals    = valid["t_rel_s"].values
        median_rr = np.median(rr_vals)
        mean_rr   = np.mean(rr_vals)
        min_rr    = rr_vals.min()
        max_rr    = rr_vals.max()
        std_rr    = rr_vals.std()
        rmssd     = np.sqrt(np.mean(np.diff(rr_vals) ** 2))
        min_idx   = int(np.argmin(rr_vals))
        max_idx   = int(np.argmax(rr_vals))

        lines.append(f"  Total beats in window:  {len(rr_vals)}")
        lines.append(f"  Median RR:  {median_rr:.0f} ms  (≈ {60000/median_rr:.0f} bpm)")
        lines.append(f"  Mean RR:    {mean_rr:.0f} ms  (≈ {60000/mean_rr:.0f} bpm)")
        lines.append(f"  Min RR:     {min_rr:.0f} ms  (≈ {60000/min_rr:.0f} bpm)  at t={t_vals[min_idx]:+.1f}s")
        lines.append(f"  Max RR:     {max_rr:.0f} ms  (≈ {60000/max_rr:.0f} bpm)  at t={t_vals[max_idx]:+.1f}s")
        lines.append(f"  RR Std Dev: {std_rr:.0f} ms")
        lines.append(f"  RMSSD:      {rmssd:.0f} ms")

        w = window_sec
        pre  = rr_vals[t_vals < -5]
        peri = rr_vals[(t_vals >= -5) & (t_vals <= 5)]
        post = rr_vals[t_vals > 5]
        lines.append("")
        lines.append("  Zone breakdown:")
        for label, zone in [
            (f"Pre  (-{w:.0f}s to  -5s)", pre),
            ( "Peri  (-5s to  +5s)",       peri),
            (f"Post  (+5s to +{w:.0f}s)",  post),
        ]:
            if len(zone):
                lines.append(
                    f"    {label}: {len(zone):3d} beats, "
                    f"median RR={np.median(zone):.0f}ms (≈{60000/np.median(zone):.0f} bpm), "
                    f"range={zone.min():.0f}–{zone.max():.0f}ms"
                )
            else:
                lines.append(f"    {label}: no beats")
    else:
        lines.append("  Insufficient beats for statistics.")

    lines.append("")

    # ── ECG quality ───────────────────────────────────────────────────────────
    lines.append("=== ECG SIGNAL QUALITY ===")
    if len(ecg_df):
        ecg       = ecg_df["ecg"].values
        ecg_std   = ecg.std()
        ecg_min   = ecg.min()
        ecg_max   = ecg.max()
        ecg_ptp   = ecg_max - ecg_min
        noise_est = "low" if ecg_std < 0.15 else "moderate" if ecg_std < 0.35 else "high"
        polarity_flag = ""
        if abs(ecg_min) > ecg_max * 1.3 and ecg_max > 0:
            polarity_flag = "  ⚠ LIKELY POLARITY-INVERTED (negative peak dominates)"
        elif ecg_max < 0.05:
            polarity_flag = "  ⚠ FLAT / NO POSITIVE DEFLECTION (possible inversion or lead-off)"
        lines.append(f"  ECG samples:     {len(ecg)}")
        lines.append(f"  Signal std:      {ecg_std:.3f} mV  (noise estimate: {noise_est})")
        lines.append(
            f"  Amplitude range: {ecg_min:.2f} to {ecg_max:.2f} mV  "
            f"(peak-to-peak: {ecg_ptp:.2f} mV)"
        )
        lines.append(
            f"  Polarity:       {polarity_flag}" if polarity_flag
            else "  Polarity:        normal (positive R-peak dominant)"
        )
    else:
        lines.append("  No ECG samples extracted.")

    lines.append("")

    # ── Human annotations ─────────────────────────────────────────────────────
    lines.append("=== HUMAN ANNOTATIONS ===")
    if ann_type == "beat_cluster" and beat_anns:
        lines.append("Beats in this cluster (patient-tagged, each is part of this event):")
        for ann in sorted(beat_anns, key=lambda a: a["peak_timestamp_ns"]):
            t_rel = (int(ann["peak_timestamp_ns"]) - center_ns) / 1e9
            lines.append(f"  beat  t={t_rel:+7.2f}s  →  {ann['tag_name']}")
    elif ann_type == "seg":
        lines.append(
            f"Segment annotation (region, not a single beat): {tag_name}"
        )
        if primary_span:
            lines.append(
                f"  Region spans t={primary_span[0]:+.1f}s → t={primary_span[1]:+.1f}s  "
                f"({primary_span[1]-primary_span[0]:.1f}s total)"
            )
    if nearby:
        lines.append("")
        lines.append(f"Other annotations visible in this window (not part of this event):")
        for ann in nearby:
            if ann["ann_type"] == "beat":
                lines.append(f"  beat  t={ann['t_start']:+7.2f}s  →  {ann['tag']}")
            else:
                dur_s = ann["t_end"] - ann["t_start"]
                lines.append(
                    f"  seg   t={ann['t_start']:+7.2f}s → {ann['t_end']:+7.2f}s  "
                    f"({dur_s:.1f}s span)  →  {ann['tag']}"
                )
    else:
        lines.append("  No other annotations visible in this window.")

    return "\n".join(lines)


# ── Gemini CLI call ───────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
You are analysing a segment of a wearable ECG recording from a POTS \
(Postural Orthostatic Tachycardia Syndrome) patient.

The data above was extracted from a Polar H10 chest strap (130 Hz). \
t=0 is the EXACT timestamp of a single beat (or segment) that the patient \
manually tagged during ECG review.  This annotation IS the cardiac event — \
it is not an approximation.

PATIENT CONTEXT:
• Diagnosed POTS — postural tachycardia (HR often 100–180 bpm upright), extreme HR variability
• Frequently exhibits RSA: rhythmic RR compression → expansion repeating at respiratory rate
  (~12–20 cycles/min); amplitude ranges from 50 ms to 400+ ms per cycle
• "Vagal arrests": sudden single-beat pauses of 1–3+ seconds (one RR >> baseline), then \
gradual return over ~3–5 beats (e.g. 2000 ms → 1250 → 1000 → 750 → back to normal)
• PAC: early beat, slightly different morphology, short pre-beat RR, possible compensatory pause
• PVC: early beat, wide/bizarre QRS, usually full compensatory pause (long post-beat RR)
• Motion/EMG artifact: large irregular RR scatter unrelated to physiology
• Positional change: sustained step-change in HR (standing up = HR jumps 30+ bpm)

ANNOTATION SYSTEM — two naming schemes appear in the data:

V2 tags (prefix "L", stored as theme_id = 100+slot) — current system:
  L1=NormToNorm   L2=Contraction   L3=Expansion    L4=Rebound Beat  L5=PAC
  L6=PVC          L7=Couplet       L8=Triplet       L9=RSA           L10=Vagal Event
  L11=Rhythmic    L12=Tachy>Normal L13=Normal>Tachy
  L18=Bad ECG     L19=Cardiac Anomaly  L20=Artifact

V1 tags (prefix "V1:", stored as raw theme_id 1–9) — original system, DIFFERENT numbering:
  V1:Contraction         → single RR contraction event
  V1:Expansion           → single RR expansion / full RSA arc (segment)
  V1:Rhythmic/RSA/Vagal  → rhythmic RSA pattern or vagal tone
  V1:Tachy↔Normal        → HR transition in either direction (direction uncertain)
  V1:PAC/PVC             → ectopic beat or surrounding beat(s)
  V1:BeatAnomaly         → single beat that looked unusual, not further classified
  V1:RSA-like            → similar to rhythmic/RSA pattern
  V1:CardiacAnomaly      → unclassified cardiac anomaly

ECG POLARITY NOTE:
A small number of files were stored polarity-inverted (R-peaks point downward). \
The ECG SIGNAL QUALITY section flags this.  If flagged:
  • RR timings are STILL VALID; cardiac classification remains accurate
  • Set ecg_quality to "inverted" and note inversion in notes

Classify the primary cardiac event at t=0, guided by the patient's annotation label \
and the RR sequence.

Respond with ONLY a single valid JSON object — no markdown, no prose outside JSON:
{
  "event_type": "<rsa | vagal_arrest | pac | pvc | tachycardia_onset | bradycardia | artifact | positional_change | normal_sinus | unknown>",
  "sub_type": "<brief specific descriptor>",
  "confidence": <0.0–1.0>,
  "rr_pattern": "<1–2 sentence description of the RR series>",
  "hr_range_bpm": [<min_int>, <max_int>],
  "key_features": ["<feature1>", "<feature2>"],
  "ecg_quality": "<clean | noisy | degraded | inverted>",
  "likely_physiological": <true | false>,
  "notes": "<1–2 sentence clinical note; mention agreement or disagreement with patient label>"
}"""


def call_gemini_cli(
    data_text:   str,
    node:        str,
    gemini_cli:  str,
    model:       str,
    timeout_s:   int   = 120,
    max_retries: int   = 2,
    backoff_s:   float = 4.0,
) -> dict:
    cmd = [node, gemini_cli, "--model", model,
           "--prompt", _ANALYSIS_PROMPT, "--output-format", "json"]
    last_err = "Unknown error"
    response_text = ""
    for attempt in range(max_retries + 1):
        try:
            proc = subprocess.run(
                cmd,
                input=data_text.encode("utf-8", errors="replace"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace").strip()
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            if not stdout:
                last_err = f"Empty stdout. stderr: {stderr[-500:]}"
                raise RuntimeError(last_err)
            try:
                cli_json = json.loads(stdout)
            except json.JSONDecodeError as e:
                last_err = f"CLI JSON parse error: {e}. stdout: {stdout[:400]}"
                raise RuntimeError(last_err)
            if isinstance(cli_json, dict) and cli_json.get("error"):
                last_err = f"CLI reported error: {cli_json['error']}"
                raise RuntimeError(last_err)
            response_text = cli_json.get("response", "")
            if not response_text:
                last_err = f"No 'response' field. stderr: {stderr[-300:]}"
                raise RuntimeError(last_err)
            cleaned = re.sub(r"^```(?:json)?\s*", "", response_text.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_err = f"Model JSON parse error: {e}. response: {response_text[:400]}"
            logger.warning("Attempt %d — JSON parse error: %s", attempt + 1, e)
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            last_err = str(e)
            logger.warning("Attempt %d — %s", attempt + 1, str(e)[:120])
        if attempt < max_retries:
            time.sleep(backoff_s * (2 ** attempt))
    raise RuntimeError(f"All {max_retries + 1} attempts failed: {last_err}")


# ── CSV output ────────────────────────────────────────────────────────────────

def write_result(output_csv: Path, row: dict) -> None:
    write_header = not output_csv.exists()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="AI-assisted annotation of individual human-annotated cardiac events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ecg-parquet",   type=Path, default=DEFAULT_ECG_PARQUET)
    p.add_argument("--peaks-parquet", type=Path, default=DEFAULT_PEAKS_PARQUET)
    p.add_argument("--output",        type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--png-dir",       type=Path, default=DEFAULT_PNG_DIR)
    p.add_argument("--beat-anns",     type=Path, default=DEFAULT_BEAT_ANNS_CSV)
    p.add_argument("--seg-anns",      type=Path, default=DEFAULT_SEG_ANNS_CSV)
    p.add_argument("--tag-labels",    type=Path, default=DEFAULT_TAG_LABELS_JSON)
    p.add_argument("--model",         type=str,  default=DEFAULT_MODEL)
    p.add_argument("--flank-beats",   type=int,   default=DEFAULT_FLANK_BEATS, metavar="N",
                   help="Number of beats to show on each side of the cluster in the zoom panel "
                        "(default: %(default)s)")
    p.add_argument("--gap-beats",     type=int,   default=DEFAULT_GAP_BEATS, metavar="N",
                   help="Gap > N beats between annotations starts a new cluster "
                        "(default: %(default)s)")
    p.add_argument("--context-beats", type=int,   default=DEFAULT_CONTEXT_BEATS, metavar="N",
                   help="Beats shown on each side in the wide-context panel; 0 = single panel "
                        "(default: %(default)s)")
    p.add_argument("--max-calls",     type=int,   default=DEFAULT_MAX_CALLS)
    p.add_argument("--delay-sec",     type=float, default=DEFAULT_DELAY_SEC)
    p.add_argument("--timeout",       type=int,   default=120)
    p.add_argument("--render-only",   action="store_true",
                   help="Render PNGs without calling Gemini")
    p.add_argument("--list-events",   action="store_true",
                   help="Print the event table and exit")
    args = p.parse_args()

    if not args.render_only and not args.list_events:
        try:
            node_path   = find_node()
            gemini_path = find_gemini_cli()
            logger.info("Node:   %s", node_path)
            logger.info("Gemini: %s", gemini_path)
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)
    else:
        node_path = gemini_path = ""

    for path, label in [
        (args.ecg_parquet,   "ecg_samples.parquet"),
        (args.peaks_parquet, "peaks.parquet"),
        (args.beat_anns,     "beat_annotations.csv"),
    ]:
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    tag_names        = load_tag_names(args.tag_labels)
    beat_df          = load_cardiac_beat_annotations(args.beat_anns, tag_names)
    seg_df           = load_cardiac_seg_annotations(args.seg_anns, tag_names)
    global_median_rr = compute_global_median_rr(args.peaks_parquet)
    logger.info(
        "Global median RR: %.0f ms  (≈ %.0f bpm)  "
        "flank=%d beats  gap=%d beats  context=%d beats",
        global_median_rr * 1000, 60.0 / global_median_rr,
        args.flank_beats, args.gap_beats, args.context_beats,
    )
    gap_sec    = args.gap_beats * global_median_rr
    all_events = build_event_list(beat_df, seg_df, gap_sec=gap_sec)

    if args.list_events:
        print(f"\n{'ID':>6}  {'Type':4}  {'Center (UTC)':25}  Tag")
        print("-" * 70)
        for ev in all_events:
            dt = datetime.datetime.utcfromtimestamp(ev["center_ns"] / 1e9).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print(f"{ev['event_id']:>6}  {ev['ann_type']:4}  {dt:25}  {ev['tag_name']}")
        print(f"\nTotal: {len(all_events)} events")
        return

    purge_errored_rows(args.output)
    already_done = load_existing_results(args.output)
    to_process   = [ev for ev in all_events if ev["event_id"] not in already_done]
    to_process   = to_process[: args.max_calls]

    logger.info(
        "%d events total | %d already done | %d to process this run",
        len(all_events), len(already_done), len(to_process),
    )
    if not to_process:
        logger.info("All events processed. Nothing to do.")
        return

    n_ok = n_err = n_skip = 0

    for ev in to_process:
        event_id  = ev["event_id"]
        ann_type  = ev["ann_type"]
        center_ns = ev["center_ns"]
        tag_name  = ev["tag_name"]
        beat_anns = ev.get("beat_anns") or []
        n_anns    = len(beat_anns) if ann_type == "beat_cluster" else 1
        center_dt = datetime.datetime.utcfromtimestamp(center_ns / 1e9).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ── Cluster boundaries (for beats-based window computation) ──────────────
        if ann_type == "beat_cluster" and beat_anns:
            cluster_ts = [int(a["peak_timestamp_ns"]) for a in beat_anns]
            c_start_ns = min(cluster_ts)
            c_end_ns   = max(cluster_ts)
        else:  # seg
            c_start_ns = ev.get("span_start_ns", center_ns)
            c_end_ns   = ev.get("span_end_ns",   center_ns)

        # ── Coarse peaks: wide enough to count flank_beats on each side ──────────
        # Factor of 1.5 guards against HR variability (e.g. POTS tachycardia).
        coarse_sec   = max(args.context_beats, args.flank_beats * 3) * global_median_rr * 1.5
        peaks_coarse = extract_peaks_window(args.peaks_parquet, center_ns, coarse_sec)

        # ── Convert beat counts → seconds using actual local peaks ───────────────
        zoom_sec = compute_zoom_from_peaks(
            center_ns, c_start_ns, c_end_ns, peaks_coarse,
            args.flank_beats, global_median_rr,
        )
        ctx_raw  = compute_zoom_from_peaks(
            center_ns, c_start_ns, c_end_ns, peaks_coarse,
            args.context_beats, global_median_rr,
        ) if args.context_beats > 0 else None
        context_sec = ctx_raw if (ctx_raw is not None and ctx_raw > zoom_sec) else None

        fetch_sec = max(zoom_sec, context_sec or zoom_sec)

        # ── ECG + peaks filtered to actual render window ─────────────────────────
        ecg_df   = extract_ecg_window(args.ecg_parquet, center_ns, fetch_sec)
        fetch_ns = int(fetch_sec * 1e9)
        peaks_df = peaks_coarse[
            (peaks_coarse["timestamp_ns"] >= center_ns - fetch_ns) &
            (peaks_coarse["timestamp_ns"] <= center_ns + fetch_ns)
        ].copy().reset_index(drop=True)

        nearby = find_nearby_annotations(
            center_ns, zoom_sec, beat_df, seg_df,
            excluded_ids=ev["cluster_ids"],
        )
        # nearby_str for CSV: human-readable summary
        nearby_str = "; ".join(
            (f"t={a['t_start']:+.1f}s:{a['tag']}" if a["ann_type"] == "beat"
             else f"t={a['t_start']:+.1f}→{a['t_end']:+.1f}s:{a['tag']}")
            for a in nearby
        )

        # primary_span: relative seconds for segment events (for PNG rendering)
        primary_span: tuple[float, float] | None = None
        if ann_type == "seg":
            s_ns = ev.get("span_start_ns", center_ns)
            e_ns = ev.get("span_end_ns",   center_ns)
            primary_span = (
                (s_ns - center_ns) / 1e9,
                (e_ns - center_ns) / 1e9,
            )

        logger.info(
            "[%d/%d]  #%d  %s  %s  n=%d  zoom=±%.0fs(~%dB)  [%s]%s",
            n_ok + n_err + n_skip + 1, len(to_process),
            event_id, ann_type, center_dt, n_anns,
            zoom_sec, int(zoom_sec / global_median_rr),
            tag_name,
            f"  span={primary_span[0]:+.0f}→{primary_span[1]:+.0f}s" if primary_span
            else (f"  nearby: {nearby_str[:50]}" if nearby_str else ""),
        )

        if len(ecg_df) < 50:
            logger.warning("  Insufficient ECG data (%d samples) — skipping", len(ecg_df))
            write_result(args.output, {
                "event_id": event_id, "ann_type": ann_type,
                "event_center_datetime": center_dt, "event_center_ns": center_ns,
                "human_tags": tag_name, "n_annotations": n_anns,
                "nearby_tags": nearby_str,
                "error": f"insufficient_ecg ({len(ecg_df)} samples)",
                "processed_at": datetime.datetime.utcnow().isoformat(),
            })
            n_skip += 1
            continue

        png_path = args.png_dir / f"event_{event_id:05d}_{ann_type}.png"
        try:
            render_png(
                ecg_df, peaks_df, center_ns, event_id, center_dt,
                zoom_sec, png_path,
                tag_name=tag_name,
                nearby=nearby,
                context_sec=context_sec,
                primary_span=primary_span,
                cluster_beat_anns=beat_anns or None,
            )
            logger.info("  PNG → %s", png_path.name)
        except Exception as e:
            logger.warning("  PNG render failed (non-fatal): %s", e)
            png_path = Path("")

        if args.render_only:
            n_ok += 1
            continue

        data_text = build_data_text(
            ecg_df, peaks_df, center_ns, event_id, center_dt,
            zoom_sec, tag_name, ann_type, nearby,
            beat_anns=beat_anns or None,
            primary_span=primary_span,
        )

        try:
            result   = call_gemini_cli(
                data_text, node_path, gemini_path, args.model,
                timeout_s=args.timeout,
            )
            hr_range = result.get("hr_range_bpm", [None, None])
            logger.info(
                "  → %-20s | sub: %-30s | conf=%.2f | quality=%s",
                result.get("event_type", "?"),
                result.get("sub_type",   "?"),
                float(result.get("confidence", 0)),
                result.get("ecg_quality", "?"),
            )
            write_result(args.output, {
                "event_id":              event_id,
                "ann_type":              ann_type,
                "event_center_datetime": center_dt,
                "event_center_ns":       center_ns,
                "human_tags":            tag_name,
                "n_annotations":         n_anns,
                "nearby_tags":           nearby_str,
                "event_type":            result.get("event_type", ""),
                "sub_type":              result.get("sub_type", ""),
                "confidence":            result.get("confidence", ""),
                "rr_pattern":            result.get("rr_pattern", ""),
                "hr_min_bpm":            hr_range[0] if len(hr_range) > 0 else "",
                "hr_max_bpm":            hr_range[1] if len(hr_range) > 1 else "",
                "key_features":          "; ".join(str(f) for f in result.get("key_features", [])),
                "ecg_quality":           result.get("ecg_quality", ""),
                "likely_physiological":  result.get("likely_physiological", ""),
                "notes":                 result.get("notes", ""),
                "model_used":            args.model,
                "processed_at":          datetime.datetime.utcnow().isoformat(),
                "png_path":              str(png_path),
                "error":                 "",
            })
            n_ok += 1

        except Exception as e:
            logger.error("  Gemini call failed: %s", str(e)[:200])
            write_result(args.output, {
                "event_id": event_id, "ann_type": ann_type,
                "event_center_datetime": center_dt, "event_center_ns": center_ns,
                "human_tags": tag_name, "n_annotations": n_anns,
                "nearby_tags": nearby_str,
                "model_used": args.model,
                "processed_at": datetime.datetime.utcnow().isoformat(),
                "png_path": str(png_path),
                "error": str(e)[:300],
            })
            n_err += 1

        time.sleep(args.delay_sec)

    logger.info(
        "Done.  ok=%d  skipped=%d  errors=%d  (output: %s)",
        n_ok, n_skip, n_err, args.output,
    )


if __name__ == "__main__":
    main()
