#!/usr/bin/env python3
"""
marker_ai_annotator.py — AI-assisted annotation of Marker.csv events via Gemini CLI.

For each user-pressed marker in Marker.csv, this script:
  1. Extracts a ±window_sec ECG/RR window from the parquet files (predicate pushdown — fast)
  2. Renders a two-panel PNG (ECG waveform + RR series) and saves it for manual inspection
  3. Sends a structured text description of the RR sequence + ECG stats to the Gemini CLI
     (no API key required — uses existing CLI authentication)
  4. Parses the JSON response and appends to the output CSV

Progress is saved after every single CLI call.  Re-running resumes from where it left off.

Usage:
    python marker_ai_annotator.py
    python marker_ai_annotator.py --max-calls 50
    python marker_ai_annotator.py --window-sec 90
    python marker_ai_annotator.py --model gemini-3.1-pro-preview
    python marker_ai_annotator.py --render-only   # PNGs only, no Gemini calls
"""

import argparse
import csv
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths & defaults ──────────────────────────────────────────────────────────

DEFAULT_MARKERS_PATH    = Path("/Volumes/xHRV/Accessory/Marker.csv")
DEFAULT_ECG_PARQUET     = Path("/Volumes/xHRV/processed/ecg_samples.parquet")
DEFAULT_PEAKS_PARQUET   = Path("/Volumes/xHRV/processed/peaks.parquet")
DEFAULT_OUTPUT_CSV      = Path("/Volumes/xHRV/Accessory/marker_annotations/ai_annotations.csv")
DEFAULT_PNG_DIR         = Path("/Volumes/xHRV/Accessory/marker_annotations/ai_review_pngs")
DEFAULT_BEAT_ANNS_CSV   = Path("/Volumes/xHRV/Accessory/marker_annotations/beat_annotations.csv")
DEFAULT_SEG_ANNS_CSV    = Path("/Volumes/xHRV/Accessory/marker_annotations/segment_annotations.csv")
DEFAULT_TAG_LABELS_JSON = Path("/Volumes/xHRV/Accessory/marker_annotations/tag_labels.json")
DEFAULT_MODEL           = "gemini-3.1-pro-preview"
DEFAULT_WINDOW_SEC      = 60      # ±60s around each marker (2 min total visible)
DEFAULT_DELAY_SEC       = 2.0     # seconds between CLI calls
DEFAULT_MAX_CALLS       = 100

# theme_id values that represent artifact/noise rather than cardiac events —
# excluded from the Gemini human-annotation context section.
_ARTIFACT_THEME_IDS: frozenset[int] = frozenset({20, 108, 120})   # L8, 20, L20 — artifact beats
_BAD_ECG_THEME_IDS:  frozenset[int] = frozenset({18, 118})        # 18, L18 — bad ECG regions
_NON_CARDIAC_THEME_IDS: frozenset[int] = _ARTIFACT_THEME_IDS | _BAD_ECG_THEME_IDS

# Node + Gemini CLI — node isn't on subprocess PATH so we resolve it explicitly
_NODE_CANDIDATES = [
    "/opt/homebrew/opt/node/bin/node",
    "/opt/homebrew/opt/node@23/bin/node",
    "/opt/homebrew/opt/node@24/bin/node",
    "/opt/homebrew/opt/node@25/bin/node",
    "/usr/local/bin/node",
]
_GEMINI_CLI = "/opt/homebrew/bin/gemini"

CSV_COLUMNS = [
    "marker_index", "marker_datetime", "marker_timestamp_ns",
    "event_type", "sub_type", "confidence",
    "rr_pattern", "hr_min_bpm", "hr_max_bpm",
    "key_features", "ecg_quality", "likely_physiological", "notes",
    "model_used", "processed_at", "png_path", "error",
]

# ── Human annotation loader ───────────────────────────────────────────────────

def load_human_annotations(
    beat_csv: Path,
    seg_csv: Path,
    tag_json: Path,
) -> tuple[dict[int, str], list[dict]]:
    """Load human annotations for Gemini context lookup.

    The marker_idx in annotation CSVs is a *review-session key*, not a temporal
    proximity key — annotated beats can be anywhere in the 15-month ECG.
    We therefore index beat annotations by peak_timestamp_ns for O(1) lookup,
    and store segment annotations as a sorted list for range intersection.

    Returns:
        beat_by_ts:   peak_timestamp_ns (int) → tag_name (str)
                      Only cardiac-event tags; artifact/bad-ECG excluded.
        seg_anns:     list of {tag_name, view_start_ns, view_end_ns}
                      Only cardiac-event tags; artifact/bad-ECG excluded.
    """
    tag_names: dict[int, str] = {}
    if tag_json.exists():
        with open(tag_json) as f:
            raw = json.load(f)
        tag_names = {int(k): v for k, v in raw.items()}
        # Map legacy 100+ ids → "L{base}:{name}"
        for base_id, name in list(tag_names.items()):
            tag_names.setdefault(base_id + 100, f"L{base_id}:{name}")

    beat_by_ts: dict[int, str] = {}
    seg_anns:   list[dict]     = []

    if beat_csv.exists():
        try:
            ba = pd.read_csv(beat_csv)
            for _, row in ba.iterrows():
                tid = int(row["theme_id"])
                if tid in _NON_CARDIAC_THEME_IDS:
                    continue
                ts  = int(row["peak_timestamp_ns"])
                beat_by_ts[ts] = tag_names.get(tid, f"tag{tid}")
        except Exception as e:
            logger.warning("Could not load beat annotations: %s", e)

    if seg_csv.exists():
        try:
            sa = pd.read_csv(seg_csv)
            for _, row in sa.iterrows():
                tid = int(row["theme_id"])
                if tid in _NON_CARDIAC_THEME_IDS:
                    continue
                seg_anns.append({
                    "tag_name":      tag_names.get(tid, f"tag{tid}"),
                    "view_start_ns": int(row["view_start_ns"]),
                    "view_end_ns":   int(row["view_end_ns"]),
                })
        except Exception as e:
            logger.warning("Could not load segment annotations: %s", e)

    # ── Event centers: median annotation timestamp per marker_idx ────────────
    # These become the analysis window center instead of the imprecise star-press
    # timestamp.  ALL beat annotations are used for centering (including artifact
    # tags) because the user was looking at that ECG region regardless of tag.
    event_centers: dict[int, int] = {}
    if beat_csv.exists():
        try:
            ba_full = pd.read_csv(beat_csv)
            for mid, grp in ba_full.groupby("marker_idx"):
                event_centers[int(mid)] = int(grp["peak_timestamp_ns"].median())
        except Exception:
            pass

    if seg_csv.exists():
        try:
            sa_full = pd.read_csv(seg_csv)
            for mid, grp in sa_full.groupby("marker_idx"):
                if int(mid) not in event_centers:
                    midpoints = (grp["view_start_ns"] + grp["view_end_ns"]) / 2
                    event_centers[int(mid)] = int(midpoints.median())
        except Exception:
            pass

    logger.info(
        "Human annotations loaded: %d tagged beats, %d tagged segments, "
        "%d markers with annotation-derived event centers",
        len(beat_by_ts), len(seg_anns), len(event_centers),
    )
    return beat_by_ts, seg_anns, event_centers


# ── CLI discovery ─────────────────────────────────────────────────────────────

def find_node() -> str:
    """Return the path to the node binary, or raise if not found."""
    for candidate in _NODE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Node.js binary not found. Tried: " + ", ".join(_NODE_CANDIDATES)
    )


def find_gemini_cli() -> str:
    if not Path(_GEMINI_CLI).exists():
        raise FileNotFoundError(f"Gemini CLI not found at {_GEMINI_CLI}")
    return _GEMINI_CLI

# ── Marker loading ────────────────────────────────────────────────────────────

def load_markers(markers_path: Path, utc_offset_hours: float | None = None) -> pd.DataFrame:
    """Load Marker.csv and convert timestamps to UTC nanoseconds.

    ``utc_offset_hours``:
        • None (default) — auto-detect using the system's local timezone with
          full DST awareness (pytz / zoneinfo).  Works correctly for US/Eastern
          which alternates between UTC-5 (EST) and UTC-4 (EDT).
        • float — explicit fixed offset in hours (positive = east of UTC, e.g.
          pass 5.0 for a device that always recorded in UTC-5 regardless of DST).
          Use 0.0 to treat the CSV timestamps as already-UTC.
    """
    df = pd.read_csv(markers_path)
    dt_col = next(
        (c for c in df.columns if "time" in c.lower() or "date" in c.lower()),
        df.columns[0],
    )
    df = df.rename(columns={dt_col: "marker_datetime"})

    if utc_offset_hours is None:
        # DST-aware: use the local system timezone (the recording device and the
        # system running marker_viewer are both on the same machine / timezone).
        try:
            import zoneinfo
            local_tz = zoneinfo.ZoneInfo("America/New_York")   # adjust if needed
            def _to_utc_ns(dt_str: str) -> int:
                naive = datetime.datetime.fromisoformat(str(dt_str))
                aware = naive.replace(tzinfo=local_tz)
                return int(aware.timestamp() * 1e9)
        except ImportError:
            # Python < 3.9: fall back to dateutil
            try:
                from dateutil import tz as dateutil_tz
                local_tz = dateutil_tz.gettz("America/New_York")
                def _to_utc_ns(dt_str: str) -> int:
                    import dateutil.parser
                    naive = dateutil.parser.parse(str(dt_str))
                    aware = naive.replace(tzinfo=local_tz)
                    return int(aware.timestamp() * 1e9)
            except ImportError:
                logger.warning(
                    "Neither zoneinfo nor dateutil available; falling back to "
                    "system local time for Marker.csv timestamp conversion."
                )
                def _to_utc_ns(dt_str: str) -> int:
                    import time as _time
                    naive = datetime.datetime.fromisoformat(str(dt_str))
                    return int(naive.astimezone().timestamp() * 1e9)
        df["timestamp_ns"] = df["marker_datetime"].apply(_to_utc_ns)
    else:
        tz = datetime.timezone(datetime.timedelta(hours=utc_offset_hours))
        df["timestamp_ns"] = pd.to_datetime(df["marker_datetime"]).apply(
            lambda dt: int(dt.replace(tzinfo=tz).timestamp() * 1e9)
        )

    df = df.reset_index(drop=True)
    df.insert(0, "marker_index", df.index + 1)
    offset_desc = f"{utc_offset_hours:+.1f}h (fixed)" if utc_offset_hours is not None else "auto (system tz)"
    logger.info("Loaded %d markers from %s  (UTC offset: %s)", len(df), markers_path, offset_desc)
    return df[["marker_index", "marker_datetime", "timestamp_ns"]]


def load_existing_results(output_csv: Path) -> set[int]:
    """Return set of marker indices that completed successfully (no error column).

    Errored rows are NOT added to the done set so they are automatically
    retried on the next run.
    """
    if not output_csv.exists():
        return set()
    done: set[int] = set()
    n_errors = 0
    with open(output_csv) as f:
        for row in csv.DictReader(f):
            idx = row.get("marker_index")
            if not idx:
                continue
            if row.get("error", "").strip():
                n_errors += 1          # will be retried
            else:
                done.add(int(idx))     # success — skip
    logger.info(
        "Resuming: %d markers already done, %d errored rows will be retried",
        len(done), n_errors,
    )
    return done

# ── Data extraction ───────────────────────────────────────────────────────────

def extract_ecg_window(ecg_path: Path, t_ns: int, window_sec: float) -> pd.DataFrame:
    half = int(window_sec * 1e9)
    try:
        tbl = pq.read_table(
            ecg_path,
            columns=["timestamp_ns", "ecg"],
            filters=[("timestamp_ns", ">=", t_ns - half), ("timestamp_ns", "<=", t_ns + half)],
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
            filters=[("timestamp_ns", ">=", t_ns - half), ("timestamp_ns", "<=", t_ns + half)],
        )
        df = tbl.to_pandas().sort_values("timestamp_ns").reset_index(drop=True)
        df["rr_ms"] = df["timestamp_ns"].diff() / 1_000_000.0
        return df
    except Exception as e:
        logger.warning("Peaks extraction failed: %s", e)
        return pd.DataFrame(columns=["peak_id", "timestamp_ns", "rr_ms"])

# ── PNG rendering ─────────────────────────────────────────────────────────────

def render_png(
    ecg_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    t_marker_ns: int,
    marker_index: int,
    marker_datetime: str,
    window_sec: float,
    output_path: Path,
) -> None:
    ecg_t = (ecg_df["timestamp_ns"].values - t_marker_ns) / 1e9 if len(ecg_df) else np.array([])
    ecg_v = ecg_df["ecg"].values if len(ecg_df) else np.array([])
    peaks_t = (peaks_df["timestamp_ns"].values - t_marker_ns) / 1e9 if len(peaks_df) else np.array([])
    rr_v = peaks_df["rr_ms"].values if len(peaks_df) else np.array([])

    fig = plt.figure(figsize=(18, 8), facecolor="#0f0f1a")
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.35,
                           top=0.88, bottom=0.08, left=0.06, right=0.98)
    ax_ecg = fig.add_subplot(gs[0])
    ax_rr  = fig.add_subplot(gs[1])

    _dark = "#0f0f1a"; _grid = "#1e1e38"; _text = "#a0a0c0"
    for ax in (ax_ecg, ax_rr):
        ax.set_facecolor(_dark)
        ax.tick_params(colors=_text, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2d2d4e")
        ax.grid(True, color=_grid, linewidth=0.5, linestyle="--")

    if len(ecg_t):
        ax_ecg.plot(ecg_t, ecg_v, lw=0.7, color="#4a9eff", alpha=0.9)
    ax_ecg.axvline(0, color="#ff4444", lw=1.5, ls="--", alpha=0.9, label="Marker")
    ax_ecg.set_xlim(-window_sec, window_sec)
    ax_ecg.set_ylabel("ECG (mV)", color=_text, fontsize=10)
    ax_ecg.set_xlabel("Time relative to marker (s)", color=_text, fontsize=10)
    ax_ecg.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
                  edgecolor="#2d2d4e", labelcolor=_text)
    if len(ecg_v):
        p1, p99 = np.percentile(ecg_v, [1, 99])
        m = max(0.1, (p99 - p1) * 0.15)
        ax_ecg.set_ylim(p1 - m, p99 + m)

    valid_rr = rr_v[~np.isnan(rr_v)]
    valid_pt = peaks_t[~np.isnan(rr_v)]
    if len(valid_rr):
        x = valid_pt[1:] if len(valid_pt) > len(valid_rr) else valid_pt
        ax_rr.plot(x, valid_rr, lw=1.2, color="#2ecc71", marker="o", ms=2.5, alpha=0.85)
        baseline = float(np.median(valid_rr))
        ax_rr.axhline(baseline, color="#f39c12", lw=0.8, ls=":",
                      alpha=0.7, label=f"median {baseline:.0f} ms")
        ax_rr.axvline(0, color="#ff4444", lw=1.5, ls="--", alpha=0.9)
        ax_rr.set_xlim(-window_sec, window_sec)
        ax_rr.set_ylim(max(0, valid_rr.min() - 100), valid_rr.max() + 100)
        ax_rr.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
                     edgecolor="#2d2d4e", labelcolor=_text)
    ax_rr.set_ylabel("RR interval (ms)", color=_text, fontsize=10)
    ax_rr.set_xlabel("Time relative to marker (s)", color=_text, fontsize=10)

    n_beats = len(valid_rr)
    hr_med = round(60_000 / float(np.median(valid_rr)), 1) if len(valid_rr) else "?"
    fig.suptitle(
        f"Marker #{marker_index}  —  {marker_datetime}  "
        f"(±{window_sec:.0f}s,  {n_beats} beats,  median HR ≈ {hr_med} bpm)",
        color=_text, fontsize=11, y=0.96,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, facecolor=_dark)
    plt.close(fig)

# ── Data text builder ─────────────────────────────────────────────────────────

def build_data_text(
    ecg_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    t_marker_ns: int,
    marker_index: int,
    marker_datetime: str,
    window_sec: float,
    beat_by_ts: dict[int, str] | None = None,
    seg_anns: list[dict] | None = None,
    t_star_ns: int | None = None,      # original Polar star-press timestamp (may differ)
) -> str:
    """
    Build the stdin payload for the Gemini CLI: a structured text description
    of the RR interval sequence and ECG statistics around this marker event.
    Numerical sequences give the model more precision than a PNG.
    """
    lines: list[str] = []

    lines.append(f"MARKER #{marker_index}  —  {marker_datetime} (local time of star press)")
    if t_star_ns is not None and t_star_ns != t_marker_ns:
        star_offset_s = (t_marker_ns - t_star_ns) / 1e9
        lines.append(
            f"Analysis center: annotation-derived event center  "
            f"(star button pressed {star_offset_s:+.0f}s relative to t=0)"
        )
        lines.append(
            f"Context window: ±{window_sec:.0f}s around the annotated event  "
            f"(t=0 is the median of the patient's beat annotations, NOT the button press)"
        )
    else:
        lines.append(
            f"Context window: ±{window_sec:.0f}s  "
            f"(t=0 is the star button press — no beat annotations found for this marker)"
        )
    lines.append("")

    # ── RR sequence ───────────────────────────────────────────────────────────
    valid = peaks_df.dropna(subset=["rr_ms"]).copy()
    valid["t_rel_s"] = (valid["timestamp_ns"].values - t_marker_ns) / 1e9

    lines.append("=== RR INTERVAL SEQUENCE ===")
    lines.append("Format:  t(s)   RR(ms)   HR(bpm)   note")

    # Mark the single beat closest to t=0 as the "marker beat"
    if len(valid):
        marker_beat_idx = (valid["t_rel_s"].abs()).idxmin()

        for row_idx, row in valid.iterrows():
            t   = row["t_rel_s"]
            rr  = row["rr_ms"]
            hr  = round(60_000 / rr, 1) if rr > 0 else 0
            note = "  ← MARKER" if row_idx == marker_beat_idx else ""

            # Add visual separators for notable pauses or very fast beats
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
        rr_vals = valid["rr_ms"].values
        t_vals  = valid["t_rel_s"].values

        median_rr = np.median(rr_vals)
        mean_rr   = np.mean(rr_vals)
        min_rr    = rr_vals.min()
        max_rr    = rr_vals.max()
        std_rr    = rr_vals.std()
        diffs     = np.diff(rr_vals)
        rmssd     = np.sqrt(np.mean(diffs ** 2))

        min_idx = int(np.argmin(rr_vals))
        max_idx = int(np.argmax(rr_vals))

        lines.append(f"  Total beats in window:  {len(rr_vals)}")
        lines.append(f"  Median RR:  {median_rr:.0f} ms  (≈ {60000/median_rr:.0f} bpm)")
        lines.append(f"  Mean RR:    {mean_rr:.0f} ms  (≈ {60000/mean_rr:.0f} bpm)")
        lines.append(f"  Min RR:     {min_rr:.0f} ms  (≈ {60000/min_rr:.0f} bpm)  at t={t_vals[min_idx]:+.1f}s")
        lines.append(f"  Max RR:     {max_rr:.0f} ms  (≈ {60000/max_rr:.0f} bpm)  at t={t_vals[max_idx]:+.1f}s")
        lines.append(f"  RR Std Dev: {std_rr:.0f} ms")
        lines.append(f"  RMSSD:      {rmssd:.0f} ms")

        # Zone breakdown: pre / peri / post marker
        pre  = rr_vals[t_vals < -5]
        peri = rr_vals[(t_vals >= -5) & (t_vals <= 5)]
        post = rr_vals[t_vals > 5]

        lines.append("")
        lines.append("  Zone breakdown (pre/peri/post marker):")
        for label, zone in [("Pre  (-60s to -5s)", pre), ("Peri ( -5s to +5s)", peri), ("Post (+5s to +60s)", post)]:
            if len(zone):
                lines.append(f"    {label}: {len(zone):3d} beats, "
                              f"median RR={np.median(zone):.0f}ms (≈{60000/np.median(zone):.0f} bpm), "
                              f"range={zone.min():.0f}–{zone.max():.0f}ms")
            else:
                lines.append(f"    {label}: no beats")
    else:
        lines.append("  Insufficient beats for statistics.")

    lines.append("")

    # ── ECG quality ───────────────────────────────────────────────────────────
    lines.append("=== ECG SIGNAL QUALITY ===")
    if len(ecg_df):
        ecg = ecg_df["ecg"].values
        ecg_std  = ecg.std()
        ecg_min  = ecg.min()
        ecg_max  = ecg.max()
        ecg_ptp  = ecg_max - ecg_min
        noise_est = "low" if ecg_std < 0.15 else "moderate" if ecg_std < 0.35 else "high"

        # Polarity heuristic: an upright ECG has a positive R-peak dominant.
        # If the negative excursion is substantially larger than the positive,
        # the recording is likely polarity-inverted (pipeline bug on some files).
        polarity_flag = ""
        if abs(ecg_min) > ecg_max * 1.3 and ecg_max > 0:
            polarity_flag = "  ⚠ LIKELY POLARITY-INVERTED (negative peak dominates)"
        elif ecg_max < 0.05:
            polarity_flag = "  ⚠ FLAT / NO POSITIVE DEFLECTION (possible inversion or lead-off)"

        lines.append(f"  ECG samples:     {len(ecg)}")
        lines.append(f"  Signal std:      {ecg_std:.3f} mV  (noise estimate: {noise_est})")
        lines.append(f"  Amplitude range: {ecg_min:.2f} to {ecg_max:.2f} mV  (peak-to-peak: {ecg_ptp:.2f} mV)")
        if polarity_flag:
            lines.append(f"  Polarity:       {polarity_flag}")
        else:
            lines.append(f"  Polarity:        normal (positive R-peak dominant)")
    else:
        lines.append("  No ECG samples extracted.")

    # ── Human annotations (cardiac events only — artifact/bad-ECG excluded) ──
    # Beat annotations are indexed by exact peak_timestamp_ns; segment
    # annotations are filtered by time-window overlap.  Both use the actual
    # timestamps of the extracted data — NOT marker_idx — because the
    # annotation CSVs use marker_idx as a review-session key rather than
    # a temporal proximity key.
    MATCH_TOL_NS = 15_000_000   # 15ms tolerance for beat timestamp lookup
    window_ns    = int(window_sec * 1e9)
    win_start_ns = t_marker_ns - window_ns
    win_end_ns   = t_marker_ns + window_ns

    matched_beats: list[tuple[int, str]] = []   # (timestamp_ns, tag_name)
    if beat_by_ts and len(peaks_df) > 0:
        for ts in peaks_df["timestamp_ns"].values:
            ts_int = int(ts)
            # Exact match first, then ±tolerance
            tag = beat_by_ts.get(ts_int)
            if tag is None:
                for delta in range(-MATCH_TOL_NS, MATCH_TOL_NS + 1, 7_692_308):
                    tag = beat_by_ts.get(ts_int + delta)
                    if tag is not None:
                        break
            if tag is not None:
                matched_beats.append((ts_int, tag))

    matched_segs: list[dict] = []
    if seg_anns:
        for ann in seg_anns:
            if ann["view_end_ns"] >= win_start_ns and ann["view_start_ns"] <= win_end_ns:
                matched_segs.append(ann)

    lines.append("")
    if matched_beats or matched_segs:
        lines.append("=== EXISTING HUMAN ANNOTATIONS (cardiac events only) ===")
        lines.append("Note: these beats/segments were manually tagged by the patient/analyst.")
        lines.append("Artifact and bad-ECG tags are excluded here.")

        if matched_beats:
            lines.append("")
            lines.append("Beat-level tags:")
            for ts_int, tag in sorted(matched_beats):
                t_rel_s = (ts_int - t_marker_ns) / 1e9
                lines.append(f"  t={t_rel_s:+7.2f}s  →  {tag}")

        if matched_segs:
            lines.append("")
            lines.append("Segment-level tags:")
            for ann in sorted(matched_segs, key=lambda a: a["view_start_ns"]):
                t_start = (ann["view_start_ns"] - t_marker_ns) / 1e9
                t_end   = (ann["view_end_ns"]   - t_marker_ns) / 1e9
                lines.append(f"  {t_start:+7.2f}s → {t_end:+7.2f}s  →  {ann['tag_name']}")
    else:
        lines.append("=== EXISTING HUMAN ANNOTATIONS ===")
        lines.append("  None for this time window — marker not yet manually reviewed.")

    return "\n".join(lines)


# ── Gemini CLI call ───────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
You are analysing a segment of a wearable ECG recording from a POTS \
(Postural Orthostatic Tachycardia Syndrome) patient.

The data above was extracted from a Polar H10 chest strap (130 Hz).  \
The user pressed the Polar "star" marker button at t=0 because they noticed or felt something.

PATIENT CONTEXT:
• Diagnosed POTS — postural tachycardia (HR often 100–180 bpm upright), extreme HR variability
• Frequently exhibits RSA: rhythmic RR compression → expansion repeating at respiratory rate
  (~12–20 cycles/min); amplitude ranges from 50 ms to 400+ ms per cycle
• "Vagal arrests": sudden single-beat pauses of 1–3+ seconds (one RR >> baseline), then \
gradual return over ~3–5 beats (e.g. 2000ms → 1250 → 1000 → 750 → back to normal)
• PAC: early beat, slightly different morphology, short pre-beat RR, possible compensatory pause
• PVC: early beat, wide/bizarre QRS, usually full compensatory pause (long post-beat RR)
• Motion/EMG artifact: large irregular RR scatter unrelated to physiology
• Positional change: sustained step-change in HR (standing up = HR jumps 30+ bpm)
• The marker was pressed CONSCIOUSLY — the patient felt or noticed something at t=0

LEGACY ANNOTATION GUIDE (tags prefixed "L" in the human annotations section):
The patient previously annotated beats/segments with a 9-tag system (L1–L9).  \
These were NOT systematically re-labelled before this AI pass, so treat them as \
strong-but-not-guaranteed context clues:
  L1  → NormToNorm beat (reliable)
  L2  → Single beat: RR Contraction (shortening); Segment: full RSA Contraction→Expansion→Contraction arc (reliable)
  L3  → Rhythmic RSA cycle, or vagal event, or expansion phase — context dependent (moderate reliability)
  L4  → HR transition: Tachy→Normal OR Normal→Tachy (the patient ran out of tags and merged both; reliable that a transition occurred, direction uncertain)
  L5  → PAC, PVC, or immediately surrounding beat(s) (reliable)
  L6  → Unknown / unclear original intent (treat as weak signal only)
  L7  → Similar to L3 — rhythmic/RSA-related (moderate reliability)
  L8  → Artifact (ALWAYS means artifact; never a cardiac event)
  L9  → Cardiac anomaly / unusual event (reliable that something atypical occurred)

ECG POLARITY NOTE:
Due to a known pipeline bug, a small number of ECG files were stored with inverted \
polarity (R-peaks appear as downward deflections, S-waves as upward).  The ECG SIGNAL \
QUALITY section will flag "⚠ LIKELY POLARITY-INVERTED" when this is detected.  \
If flagged:
  • RR interval timings are STILL VALID (peak detection is polarity-robust)
  • Beat-to-beat cardiac event classification (PAC, vagal arrest, RSA) remains accurate
  • ECG morphology descriptions should note the inversion
  • Set ecg_quality to "degraded" and mention inversion in notes

Classify the primary cardiac event visible near the marker.

Respond with ONLY a single valid JSON object — no markdown fences, no prose outside JSON:
{
  "event_type": "<one of: rsa | vagal_arrest | pac | pvc | tachycardia_onset | bradycardia | artifact | positional_change | normal_sinus | unknown>",
  "sub_type": "<brief specific descriptor e.g. 'rhythmic_vagal_large_amplitude' or 'single_pause_with_3beat_recovery'>",
  "confidence": <0.0 to 1.0>,
  "rr_pattern": "<1–2 sentence description of what the RR series shows>",
  "hr_range_bpm": [<min_integer>, <max_integer>],
  "key_features": ["<feature1>", "<feature2>"],
  "ecg_quality": "<clean | noisy | degraded | inverted>",
  "likely_physiological": <true | false>,
  "notes": "<1–2 sentence clinical observation or uncertainty>"
}"""


def call_gemini_cli(
    data_text: str,
    node: str,
    gemini_cli: str,
    model: str,
    timeout_s: int = 120,
    max_retries: int = 2,
    backoff_s: float = 4.0,
) -> dict:
    """Call the Gemini CLI via subprocess, parse and return the model JSON.

    Mirrors the pattern from convos_gemCli.py:
      - data_text → stdin (the per-marker RR/ECG data)
      - _ANALYSIS_PROMPT → --prompt (appended after stdin by the CLI)
      - --output-format json → wraps response in {"response": "..."}
    """
    cmd = [
        node, gemini_cli,
        "--model", model,
        "--prompt", _ANALYSIS_PROMPT,
        "--output-format", "json",
    ]

    last_err: str = "Unknown error"
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

            # Parse CLI JSON wrapper
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
                last_err = f"No 'response' field in CLI output. stderr: {stderr[-300:]}"
                raise RuntimeError(last_err)

            # Strip markdown code fences if present (matching convos_gemCli.py)
            cleaned = response_text.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

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

def purge_errored_rows(output_csv: Path) -> int:
    """Rewrite the CSV keeping only rows that succeeded (empty error column).

    Called once at startup before any retries are appended, so successful
    retries never produce duplicate rows.  Returns count of rows removed.
    """
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
        description="AI-assisted annotation of Marker.csv events via Gemini CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--markers",         type=Path, default=DEFAULT_MARKERS_PATH)
    p.add_argument("--ecg-parquet",     type=Path, default=DEFAULT_ECG_PARQUET)
    p.add_argument("--peaks-parquet",   type=Path, default=DEFAULT_PEAKS_PARQUET)
    p.add_argument("--output",          type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--png-dir",         type=Path, default=DEFAULT_PNG_DIR)
    p.add_argument("--beat-anns",       type=Path, default=DEFAULT_BEAT_ANNS_CSV,
                   help="beat_annotations.csv (default: %(default)s)")
    p.add_argument("--seg-anns",        type=Path, default=DEFAULT_SEG_ANNS_CSV,
                   help="segment_annotations.csv (default: %(default)s)")
    p.add_argument("--tag-labels",      type=Path, default=DEFAULT_TAG_LABELS_JSON,
                   help="tag_labels.json (default: %(default)s)")
    p.add_argument("--model",         type=str,  default=DEFAULT_MODEL)
    p.add_argument("--window-sec",    type=float, default=DEFAULT_WINDOW_SEC)
    p.add_argument("--utc-offset-hours", type=float, default=None,
                   help="Timezone offset for Marker.csv timestamps in hours (e.g. 5 for EST, "
                        "4 for EDT).  Default: auto-detect from system local timezone (DST-aware).")
    p.add_argument("--max-calls",     type=int,  default=DEFAULT_MAX_CALLS)
    p.add_argument("--delay-sec",     type=float, default=DEFAULT_DELAY_SEC)
    p.add_argument("--timeout",       type=int,  default=120,
                   help="Per-call CLI timeout in seconds (default: 120)")
    p.add_argument("--render-only",   action="store_true",
                   help="Render PNGs without calling Gemini (preview/debug)")
    args = p.parse_args()

    # ── Resolve CLI paths ──────────────────────────────────────────────────
    if not args.render_only:
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

    # ── Validate inputs ────────────────────────────────────────────────────
    for path, label in [
        (args.markers,       "Marker.csv"),
        (args.ecg_parquet,   "ecg_samples.parquet"),
        (args.peaks_parquet, "peaks.parquet"),
    ]:
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    # ── Load markers and human annotations ────────────────────────────────
    purge_errored_rows(args.output)           # strip old errors before retry
    markers_df   = load_markers(args.markers, args.utc_offset_hours)
    already_done = load_existing_results(args.output)
    beat_by_ts, seg_anns, event_centers = load_human_annotations(
        args.beat_anns, args.seg_anns, args.tag_labels
    )
    to_process   = markers_df[~markers_df["marker_index"].isin(already_done)].head(args.max_calls)

    logger.info(
        "%d markers total | %d already done | %d to process this run",
        len(markers_df), len(already_done), len(to_process),
    )
    if len(to_process) == 0:
        logger.info("All markers processed. Nothing to do.")
        return

    n_ok = n_err = n_skip = 0

    for _, row in to_process.iterrows():
        idx      = int(row["marker_index"])
        dt_str   = str(row["marker_datetime"])
        t_star_ns = int(row["timestamp_ns"])   # Polar H10 button press (corrected tz)

        # Use annotation-derived event center when available — the star press can
        # be seconds to over a minute before/after the actual cardiac event, so
        # the annotated beats (which the patient placed precisely on the event)
        # are a far more accurate anchor for the analysis window.
        t_ann_ns = event_centers.get(idx)
        if t_ann_ns is not None:
            t_ns = t_ann_ns
            offset_s = (t_ann_ns - t_star_ns) / 1e9
            logger.info(
                "[%d/%d]  Marker #%d  %s  (annotation center, star offset %+.0fs)",
                idx, len(markers_df), idx, dt_str, offset_s,
            )
        else:
            t_ns = t_star_ns
            logger.info(
                "[%d/%d]  Marker #%d  %s  (no annotations — using star timestamp)",
                idx, len(markers_df), idx, dt_str,
            )

        # ── Extract data ────────────────────────────────────────────────────
        ecg_df   = extract_ecg_window(args.ecg_parquet,   t_ns, args.window_sec)
        peaks_df = extract_peaks_window(args.peaks_parquet, t_ns, args.window_sec)

        if len(ecg_df) < 100:
            logger.warning("  Insufficient ECG data (%d samples) — skipping", len(ecg_df))
            write_result(args.output, {
                "marker_index": idx, "marker_datetime": dt_str,
                "marker_timestamp_ns": t_ns,
                "error": f"insufficient_ecg ({len(ecg_df)} samples)",
                "processed_at": datetime.datetime.utcnow().isoformat(),
            })
            n_skip += 1
            continue

        # ── Render PNG (always — for manual inspection) ─────────────────────
        png_path = args.png_dir / f"marker_{idx:03d}.png"
        try:
            render_png(ecg_df, peaks_df, t_ns, idx, dt_str, args.window_sec, png_path)
            logger.info("  PNG → %s", png_path.name)
        except Exception as e:
            logger.warning("  PNG render failed (non-fatal): %s", e)
            png_path = Path("")

        if args.render_only:
            n_ok += 1
            continue

        # ── Build text payload for CLI stdin ────────────────────────────────
        data_text = build_data_text(
            ecg_df, peaks_df, t_ns, idx, dt_str, args.window_sec,
            beat_by_ts=beat_by_ts,
            seg_anns=seg_anns,
            t_star_ns=t_star_ns,
        )

        # ── Call Gemini CLI ─────────────────────────────────────────────────
        try:
            result = call_gemini_cli(
                data_text, node_path, gemini_path, args.model,
                timeout_s=args.timeout,
            )
            hr_range = result.get("hr_range_bpm", [None, None])
            logger.info(
                "  → %-20s | sub: %-35s | conf=%.2f | quality=%s",
                result.get("event_type", "?"),
                result.get("sub_type", "?"),
                float(result.get("confidence", 0)),
                result.get("ecg_quality", "?"),
            )
            write_result(args.output, {
                "marker_index":        idx,
                "marker_datetime":     dt_str,
                "marker_timestamp_ns": t_ns,
                "event_type":          result.get("event_type", ""),
                "sub_type":            result.get("sub_type", ""),
                "confidence":          result.get("confidence", ""),
                "rr_pattern":          result.get("rr_pattern", ""),
                "hr_min_bpm":          hr_range[0] if len(hr_range) > 0 else "",
                "hr_max_bpm":          hr_range[1] if len(hr_range) > 1 else "",
                "key_features":        "; ".join(str(f) for f in result.get("key_features", [])),
                "ecg_quality":         result.get("ecg_quality", ""),
                "likely_physiological": result.get("likely_physiological", ""),
                "notes":               result.get("notes", ""),
                "model_used":          args.model,
                "processed_at":        datetime.datetime.utcnow().isoformat(),
                "png_path":            str(png_path),
                "error":               "",
            })
            n_ok += 1

        except Exception as e:
            logger.error("  Gemini call failed: %s", str(e)[:200])
            write_result(args.output, {
                "marker_index": idx, "marker_datetime": dt_str,
                "marker_timestamp_ns": t_ns,
                "model_used": args.model,
                "processed_at": datetime.datetime.utcnow().isoformat(),
                "png_path": str(png_path),
                "error": str(e)[:300],
            })
            n_err += 1

        time.sleep(args.delay_sec)

    # ── Summary ────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Marker AI Annotator — Session Complete")
    print(sep)
    print(f"  Annotated:   {n_ok}")
    print(f"  Errors:      {n_err}")
    print(f"  Skipped:     {n_skip}  (no ECG data in window)")
    total_done = len(already_done) + n_ok
    remaining  = len(markers_df) - total_done
    print(f"  Overall:     {total_done} / {len(markers_df)} done"
          + (f"  (~{remaining / max(1, args.max_calls):.1f} more runs)" if remaining else "  ✓ complete"))
    print(f"\n  Output: {args.output.resolve()}")
    print(f"  PNGs:   {args.png_dir.resolve()}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
