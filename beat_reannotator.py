#!/usr/bin/env python3
"""
beat_reannotator.py — Web-based granular tag annotation for labelled beats.

Keyboard shortcuts:
    Space       advance to next beat (always saves)
    ← / →       previous / next beat (saves if changed)
    m           mirror: re-apply the previous beat's tags to this beat
    r / R       jump forward to next review_needed beat
    Shift+R     jump forward to next likely_artifact beat
    1–9         toggle tag at position 1–9
    01–09       toggle tag at position 10–18
    001–009     toggle tag at position 19–27
    1–5         immediately after toggling a tag: rate it 1–5
    Esc         clear shortcut buffer / cancel pending rating
    Q           quit and save

Usage:
    python beat_reannotator.py
    python beat_reannotator.py --queue-dir data/annotation_queues
    python beat_reannotator.py --output  data/annotation_queues/beat_tags.csv
    python beat_reannotator.py --port 7432
    python beat_reannotator.py --presorted data/annotation_queues/iteration_1/presorted.csv
"""

import argparse
import csv
import json
import logging
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Default tag pool (used only on first run; after that, tag_pool.json owns order) ──

DEFAULT_TAGS: list[dict] = [
    {"name": "low_amplitude",    "desc": "Low amplitude ECG signal"},
    {"name": "aberrant_waveform","desc": "Abnormal QRS morphology"},
    {"name": "irregular_rhythm", "desc": "Irregular RR intervals"},
    {"name": "pristine",         "desc": "Perfect — R-peak, waveform, everything"},
    {"name": "chaotic",          "desc": "Erratic but interpretable"},
    {"name": "uninterpretable",  "desc": "Severe distortion, segment unusable"},
    {"name": "rr_dominant",      "desc": "RR consistency salvages unreadable waveform"},
    {"name": "nearby_aberrance", "desc": "Nearby disturbed ECG, not affecting this beat"},
]


def get_tag_key(position: int) -> str:
    """Keyboard shortcut string for a 1-indexed position in the tag pool."""
    if position <= 9:
        return str(position)
    position -= 9
    prefix_count = (position - 1) // 9 + 1
    digit        = (position - 1) % 9 + 1
    return "0" * prefix_count + str(digit)


# ── Persistence helpers ───────────────────────────────────────────────────────

def load_tag_pool(path: Path) -> list[dict]:
    """Load ordered tag pool from disk; fall back to defaults on first run."""
    if path.exists():
        try:
            with open(path) as f:
                pool = json.load(f)
            if pool:
                return pool
        except Exception:
            pass
    return [dict(t) for t in DEFAULT_TAGS]


def save_tag_pool(path: Path, pool: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(pool, f, indent=2)


def load_tag_sets(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def save_tag_sets(path: Path, sets: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(sets, f, indent=2)


def load_all_beats(queue_dir: Path) -> list[dict]:
    queue_dirs = sorted(d for d in queue_dir.iterdir() if d.is_dir())
    beats: list[dict] = []
    for qd in queue_dirs:
        completed_csv = qd / "completed.csv"
        if not completed_csv.exists():
            continue
        with open(completed_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        loaded = skipped = 0
        for row in rows:
            pid = int(row["peak_id"])
            json_path = qd / f"beat_{pid:08d}.json"
            if not json_path.exists():
                skipped += 1
                continue
            try:
                with open(json_path) as jf:
                    data = json.load(jf)
            except Exception as e:
                logger.warning("  Could not load %s: %s", json_path.name, e)
                skipped += 1
                continue
            beats.append({
                "queue":         qd.name,
                "peak_id":       pid,
                "old_label":     row["label"],
                "timestamp_ns":  data.get("timestamp_ns", 0),
                "p_artifact":    data.get("p_artifact_ensemble", None),
                "disagreement":  data.get("disagreement", None),
                "rr_prev_ms":    data.get("rr_prev_ms", None),
                "rr_next_ms":    data.get("rr_next_ms", None),
                "r_peak_index":  data.get("r_peak_index_in_context", None),
                "context_ecg":   data.get("context_ecg", []),
                "context_ts_ns": data.get("context_timestamps_ns", []),
                "segment_idx":   data.get("segment_idx", None),
            })
            loaded += 1
        logger.info("  %-22s  %d beats  (%d skipped)", qd.name, loaded, skipped)
    beats.sort(key=lambda b: b["timestamp_ns"])
    logger.info("Total beats loaded: %d", len(beats))
    return beats


def apply_presort_order(beats: list[dict], presorted_path: Path) -> list[dict]:
    """Reorder beats and attach global_corr_clean + suggested_category from presort CSV.

    The presort CSV (produced by presort_reannotation_queue.py) has columns:
        peak_id, queue, timestamp_ns, global_corr_clean, suggested_category

    The function:
      1. Reads the CSV and builds a {peak_id → {global_corr_clean, suggested_category}} map.
      2. Reorders the beats list to match the CSV row order (clean_pristine first,
         review_needed middle, likely_artifact last).
      3. Attaches global_corr_clean and suggested_category to each beat dict.
      4. Appends any beats not mentioned in the CSV at the end (unchanged order).
    """
    rows: dict[int, dict] = {}
    order: list[int] = []
    with open(presorted_path) as f:
        for row in csv.DictReader(f):
            pid = int(row["peak_id"])
            rows[pid] = {
                "global_corr_clean":  float(row["global_corr_clean"])
                                      if row.get("global_corr_clean") not in (None, "")
                                      else None,
                "suggested_category": row.get("suggested_category") or None,
            }
            order.append(pid)

    by_pid = {b["peak_id"]: b for b in beats}
    presort_pids = set(order)

    reordered: list[dict] = []
    skipped = 0
    for pid in order:
        if pid not in by_pid:
            skipped += 1
            continue
        b = dict(by_pid[pid])
        b.update(rows[pid])
        reordered.append(b)

    extras = [b for b in beats if b["peak_id"] not in presort_pids]
    if extras:
        logger.info("  %d beat(s) not in presort CSV — appended at end", len(extras))
    if skipped:
        logger.warning("  %d peak_id(s) in presort CSV not found in loaded beats — skipped",
                       skipped)

    logger.info("Presort applied: %d beats reordered (+ %d extras)", len(reordered), len(extras))
    return reordered + extras


def load_existing_tags(output_path: Path) -> dict[tuple[str, int], list[dict]]:
    existing: dict[tuple[str, int], list[dict]] = {}
    if not output_path.exists():
        return existing
    with open(output_path) as f:
        for row in csv.DictReader(f):
            try:
                key = (row["queue"], int(row["peak_id"]))
                existing[key] = json.loads(row["tags"])
            except (KeyError, ValueError, json.JSONDecodeError):
                pass
    logger.info("Resumed: %d tag-sets loaded from %s", len(existing), output_path)
    return existing


# ── Session state ─────────────────────────────────────────────────────────────

class _State:
    def __init__(
        self,
        beats:         list[dict],
        existing_tags: dict,
        output_path:   Path,
        tag_pool_path: Path,
        tag_pool:      list[dict],
        tag_sets_path: Path,
        tag_sets:      list[dict],
    ) -> None:
        self.beats         = beats
        self.output_path   = output_path
        self.tag_pool_path = tag_pool_path
        self.tag_pool      = tag_pool
        self.tag_sets_path = tag_sets_path
        self.tag_sets      = tag_sets
        self.lock          = threading.Lock()
        self.beat_tags     = dict(existing_tags)
        self.shutdown_flag = threading.Event()

    def _key(self, idx: int) -> tuple[str, int]:
        b = self.beats[idx]
        return (b["queue"], b["peak_id"])

    def get_tags(self, idx: int) -> list[dict]:
        return self.beat_tags.get(self._key(idx), [])

    def set_tags(self, idx: int, tags: list[dict]) -> None:
        with self.lock:
            self.beat_tags[self._key(idx)] = tags
            self._append_csv(idx, tags)

    def _append_csv(self, idx: int, tags: list[dict]) -> None:
        b = self.beats[idx]
        write_header = not self.output_path.exists()
        with open(self.output_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["queue", "peak_id", "timestamp_ns",
                             "old_label", "tags", "annotated_at"])
            w.writerow([
                b["queue"], b["peak_id"], b["timestamp_ns"],
                b["old_label"], json.dumps(tags),
                datetime.now(timezone.utc).isoformat(),
            ])

    # ── Tag pool mutations ─────────────────────────────────────────────────────

    def all_tags(self) -> list[dict]:
        return self.tag_pool

    def add_tag(self, name: str, desc: str) -> None:
        with self.lock:
            self.tag_pool.append({"name": name, "desc": desc})
            save_tag_pool(self.tag_pool_path, self.tag_pool)

    def reorder_tags(self, names: list[str]) -> None:
        """Reorder pool to match the given name sequence."""
        with self.lock:
            by_name = {t["name"]: t for t in self.tag_pool}
            reordered = [by_name[n] for n in names if n in by_name]
            # Append any tags missing from the new order (shouldn't happen)
            seen = set(names)
            leftover = [t for t in self.tag_pool if t["name"] not in seen]
            self.tag_pool = reordered + leftover
            save_tag_pool(self.tag_pool_path, self.tag_pool)

    # ── Tag sets ───────────────────────────────────────────────────────────────

    def save_tag_set(self, name: str, tags: list[dict]) -> None:
        with self.lock:
            self.tag_sets = [s for s in self.tag_sets if s["name"] != name]
            self.tag_sets.append({"name": name, "tags": tags})
            save_tag_sets(self.tag_sets_path, self.tag_sets)

    def delete_tag_set(self, name: str) -> None:
        with self.lock:
            self.tag_sets = [s for s in self.tag_sets if s["name"] != name]
            save_tag_sets(self.tag_sets_path, self.tag_sets)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def first_untagged(self) -> int:
        for i in range(len(self.beats)):
            if not self.beat_tags.get(self._key(i)):
                return i
        return 0

    def stats(self) -> dict:
        total     = len(self.beats)
        annotated = sum(1 for i in range(total)
                        if self.beat_tags.get(self._key(i)))
        return {"total": total, "annotated": annotated,
                "remaining": total - annotated}


# ── HTTP handler ──────────────────────────────────────────────────────────────

def make_handler(state: _State):

    def _sanitize_tags(raw: list, valid_names: set) -> list[dict]:
        out = []
        for t in raw:
            if not isinstance(t, dict) or t.get("name") not in valid_names:
                continue
            r = t.get("rating")
            out.append({
                "name":   t["name"],
                "rating": r if isinstance(r, int) and 1 <= r <= 5 else None,
            })
        return out

    def _tagged_pool(pool):
        return [{**t, "key": get_tag_key(i + 1)} for i, t in enumerate(pool)]

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, html: str) -> None:
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            qs     = parse_qs(parsed.query)

            if parsed.path == "/":
                self._html(_build_html(state))

            elif parsed.path == "/api/beat":
                idx = int(qs.get("idx", ["0"])[0])
                idx = max(0, min(idx, len(state.beats) - 1))
                b   = state.beats[idx]
                self._json(200, {
                    "idx":           idx,
                    "total":         len(state.beats),
                    "queue":         b["queue"],
                    "peak_id":       b["peak_id"],
                    "old_label":     b["old_label"],
                    "p_artifact":    b["p_artifact"],
                    "disagreement":  b["disagreement"],
                    "rr_prev_ms":    b["rr_prev_ms"],
                    "rr_next_ms":    b["rr_next_ms"],
                    "r_peak_index":  b["r_peak_index"],
                    "context_ecg":   b["context_ecg"],
                    "context_ts_ns": b["context_ts_ns"],
                    "segment_idx":        b["segment_idx"],
                    "global_corr_clean":  b.get("global_corr_clean"),
                    "suggested_category": b.get("suggested_category"),
                    "current_tags":       state.get_tags(idx),
                    "stats":              state.stats(),
                })

            elif parsed.path == "/api/tags":
                self._json(200, {"tags": _tagged_pool(state.all_tags())})

            elif parsed.path == "/api/tagsets":
                self._json(200, {"sets": state.tag_sets})

            elif parsed.path == "/api/shutdown":
                self._json(200, {"ok": True})
                state.shutdown_flag.set()

            elif parsed.path == "/api/categories":
                cats = [b.get("suggested_category") for b in state.beats]
                self._json(200, {"categories": cats})

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length)) if length else {}

            if parsed.path == "/api/annotate":
                idx   = int(body.get("idx", 0))
                idx   = max(0, min(idx, len(state.beats) - 1))
                valid = {t["name"] for t in state.all_tags()}
                tags  = _sanitize_tags(body.get("tags", []), valid)
                state.set_tags(idx, tags)
                self._json(200, {"ok": True, "stats": state.stats()})

            elif parsed.path == "/api/tags/add":
                name = str(body.get("name", "")).strip()
                desc = str(body.get("desc", "")).strip()
                if not name:
                    self._json(400, {"error": "name required"})
                    return
                if name in {t["name"] for t in state.all_tags()}:
                    self._json(400, {"error": f"tag '{name}' already exists"})
                    return
                state.add_tag(name, desc)
                self._json(200, {"ok": True, "tags": _tagged_pool(state.all_tags())})

            elif parsed.path == "/api/tags/reorder":
                names = body.get("names", [])
                if isinstance(names, list):
                    state.reorder_tags([str(n) for n in names])
                self._json(200, {"ok": True, "tags": _tagged_pool(state.all_tags())})

            elif parsed.path == "/api/tagsets/save":
                name = str(body.get("name", "")).strip()
                if not name:
                    self._json(400, {"error": "name required"})
                    return
                valid = {t["name"] for t in state.all_tags()}
                tags  = _sanitize_tags(body.get("tags", []), valid)
                if not tags:
                    self._json(400, {"error": "no valid tags to save"})
                    return
                state.save_tag_set(name, tags)
                self._json(200, {"ok": True, "sets": state.tag_sets})

            elif parsed.path == "/api/tagsets/delete":
                name = str(body.get("name", "")).strip()
                state.delete_tag_set(name)
                self._json(200, {"ok": True, "sets": state.tag_sets})

            else:
                self.send_response(404)
                self.end_headers()

    return _Handler


# ── HTML / JS / CSS ───────────────────────────────────────────────────────────

def _build_html(state: _State) -> str:
    first_untagged   = state.first_untagged()
    initial_tag_pool = json.dumps([
        {**t, "key": get_tag_key(i + 1)}
        for i, t in enumerate(state.all_tags())
    ])
    initial_tag_sets = json.dumps(state.tag_sets)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Beat Tagger</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #0f0f1a; color: #e8e8f0;
  font-family: "SF Mono","Fira Code",monospace; font-size: 13px;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}}

/* ── Header ── */
#header {{
  display: flex; align-items: center; gap: 16px; padding: 8px 16px;
  background: #1a1a2e; border-bottom: 1px solid #2d2d4e; flex-shrink: 0;
}}
#beat-counter {{ font-size: 15px; font-weight: bold; color: #a0a0c0; }}
#progress-bar-wrap {{ flex: 1; height: 8px; background: #2d2d4e; border-radius: 4px; }}
#progress-bar {{ height: 8px; background: #4a90d9; border-radius: 4px; transition: width 0.2s; }}
#stats-text {{ font-size: 11px; color: #606080; white-space: nowrap; }}

/* ── Info bar (beat meta + ML suggestion across the full width) ── */
#info-bar {{
  display: flex; align-items: center; flex-shrink: 0;
  background: #11112a; border-bottom: 1px solid #2d2d4e;
  font-size: 11px; min-height: 32px; flex-wrap: wrap;
}}
.ib-item {{
  display: flex; align-items: center; gap: 4px;
  padding: 5px 12px; border-right: 1px solid #1e1e38; white-space: nowrap;
}}
.ib-key {{ color: #505070; }}
.ib-val {{ color: #c0c0e0; font-weight: bold; }}
#ib-ml {{
  margin-left: auto; display: flex; align-items: center; gap: 8px;
  padding: 4px 12px; border-left: 1px solid #1e1e38;
  border-radius: 0;
}}
#ib-ml-name  {{ font-weight: bold; font-size: 12px; }}
#ib-ml-reason {{ color: #5a5a7a; font-size: 10px; }}

/* ── Main layout ── */
#main {{ display: flex; flex: 1; overflow: hidden; }}
#plot-col {{ flex: 1; display: flex; flex-direction: column; min-width: 0; }}
#ecg-plot {{ flex: 1; }}

/* ── Side column ── */
#side-col {{
  width: 310px; min-width: 310px; background: #13132a;
  border-left: 1px solid #2d2d4e;
  display: flex; flex-direction: column; overflow: hidden;
}}
#tags-wrap {{
  flex: 1; overflow-y: auto; padding: 0 12px 8px; min-height: 0;
}}
#side-bottom {{
  flex-shrink: 0; padding: 8px 12px 10px;
  display: flex; flex-direction: column; gap: 8px;
  border-top: 1px solid #1e1e38;
}}
#status-bar {{
  flex-shrink: 0; padding: 6px 12px;
  background: #0a0a18; border-top: 1px solid #1e1e38;
  font-size: 11px; color: #505070; text-align: center;
}}

/* ── Section headers (tags + tag sets) ── */
.sec-header {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 9px 0 5px; position: sticky; top: 0;
  background: #13132a; z-index: 10;
}}
.sec-title {{ color: #6060a0; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }}
.sec-btn {{
  padding: 2px 8px; background: #2d2d4e; border: 1px solid #3d3d6e;
  border-radius: 4px; color: #8080b0; cursor: pointer; font-size: 11px;
  font-family: inherit; white-space: nowrap;
}}
.sec-btn:hover {{ background: #3d3d6e; color: #c0c0e0; }}
.sec-divider {{ height: 1px; background: #1e1e38; margin: 4px 0 0; }}

/* ── Inline add forms ── */
.add-form {{
  display: none; background: #1e1e38; border: 1px solid #3d3d6e;
  border-radius: 5px; padding: 8px; margin-bottom: 6px;
  flex-direction: column; gap: 6px;
}}
.add-form.open {{ display: flex; }}
.add-input {{
  width: 100%; padding: 5px 8px; background: #0f0f1a;
  border: 1px solid #2d2d4e; border-radius: 4px; color: #e0e0f0;
  font-family: inherit; font-size: 12px;
}}
.add-input:focus {{ outline: none; border-color: #4a9eff; }}
.add-btns {{ display: flex; gap: 6px; justify-content: flex-end; }}
.form-cancel {{
  padding: 4px 10px; border-radius: 4px; border: none; cursor: pointer;
  font-family: inherit; font-size: 12px; background: #2d2d4e; color: #9090b0;
}}
.form-submit {{
  padding: 4px 10px; border-radius: 4px; border: none; cursor: pointer;
  font-family: inherit; font-size: 12px; background: #4a9eff; color: #0f0f1a; font-weight: bold;
}}
.form-cancel:hover {{ background: #3d3d6e; }}
.form-submit:hover {{ background: #6ab0ff; }}

/* ── Tag buttons ── */
#tag-pool {{ display: flex; flex-direction: column; gap: 3px; }}
.tag-btn {{
  display: flex; align-items: flex-start; gap: 6px;
  padding: 5px 7px; border: 1px solid #2d2d4e; border-radius: 4px;
  cursor: pointer; background: #0f0f1a; user-select: none;
  transition: background 0.1s, border-color 0.1s;
}}
.tag-btn:hover    {{ background: #1a1a38; }}
.tag-btn.selected {{ border-color: #3a8aff; background: rgba(58,138,255,0.08); }}
.tag-btn.active   {{ border-color: #f39c12; background: rgba(243,156,18,0.10); }}
.tag-btn.dragging  {{ opacity: 0.35; }}
.tag-btn.drag-over {{ border-color: #f39c12; background: rgba(243,156,18,0.18); border-style: dashed; }}
.drag-handle {{
  color: #2d2d4e; font-size: 12px; cursor: grab; flex-shrink: 0;
  margin-top: 2px; line-height: 1; user-select: none;
}}
.drag-handle:hover {{ color: #4a4a6e; }}
.drag-handle:active {{ cursor: grabbing; }}
.tag-key {{
  padding: 1px 4px; border-radius: 3px; background: #2d2d4e;
  font-size: 10px; color: #5a5a8a; font-family: monospace;
  white-space: nowrap; flex-shrink: 0; margin-top: 2px;
  min-width: 20px; text-align: center;
}}
.tag-body {{ flex: 1; min-width: 0; }}
.tag-top  {{ display: flex; align-items: center; gap: 5px; min-height: 18px; }}
.tag-name {{
  font-size: 12px; color: #c0c0e0; flex: 1; min-width: 0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.tag-desc {{
  font-size: 10px; color: #404060; margin-top: 1px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.tag-check {{ font-size: 12px; color: #3a8aff; flex-shrink: 0; }}
.tag-star  {{ font-size: 11px; color: #f39c12; flex-shrink: 0; font-weight: bold; }}
.rating-bubbles {{ display: flex; gap: 3px; flex-shrink: 0; }}
.rating-bubble {{
  width: 18px; height: 18px; border-radius: 50%;
  background: #252545; color: #7070a0;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; cursor: pointer; border: 1px solid #353565;
  flex-shrink: 0; transition: background 0.1s;
}}
.rating-bubble:hover  {{ background: #353560; color: #f39c12; }}
.rating-bubble.active {{ background: #f39c12; color: #0f0f1a; border-color: #f39c12; }}

/* ── Tag sets ── */
#tag-sets-pool {{ display: flex; flex-direction: column; gap: 3px; margin-top: 2px; }}
.set-row {{
  display: flex; align-items: center; gap: 6px;
  padding: 5px 8px; border: 1px solid #2d2d4e; border-radius: 4px;
  background: #0f0f1a; cursor: default;
}}
.set-row:hover {{ background: #1a1a38; }}
.set-apply {{
  background: none; border: none; color: #4a9eff; cursor: pointer;
  font-size: 13px; padding: 0 2px; flex-shrink: 0; line-height: 1;
}}
.set-apply:hover {{ color: #6ab0ff; }}
.set-info {{ flex: 1; min-width: 0; cursor: pointer; }}
.set-name {{ font-size: 12px; color: #c0c0e0; display: block; }}
.set-preview {{
  font-size: 10px; color: #505070; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; display: block; margin-top: 1px;
}}
.set-delete {{
  background: none; border: none; color: #404060; cursor: pointer;
  font-size: 15px; padding: 0 2px; flex-shrink: 0; line-height: 1;
}}
.set-delete:hover {{ color: #e74c3c; }}
.sets-empty {{ font-size: 11px; color: #404060; padding: 6px 2px; font-style: italic; }}

/* ── Navigate ── */
#nav-row {{ display: flex; gap: 6px; margin-bottom: 6px; }}
.nav-btn {{
  flex: 1; padding: 7px 6px; border: 1px solid #2d2d4e; border-radius: 4px;
  background: #1a1a2e; color: #a0a0c0; cursor: pointer; font-family: inherit; font-size: 12px;
}}
.nav-btn:hover {{ background: #22224a; }}
#jump-row {{ display: flex; gap: 5px; }}
#jump-input {{
  flex: 1; padding: 5px 8px; background: #0f0f1a; border: 1px solid #2d2d4e;
  border-radius: 4px; color: #e0e0f0; font-family: inherit; font-size: 12px;
}}
#jump-go {{
  padding: 5px 10px; background: #2d2d4e; border: none; border-radius: 4px;
  color: #c0c0e0; cursor: pointer; font-family: inherit;
}}
#jump-go:hover {{ background: #3d3d6e; }}
</style>
</head>
<body>

<div id="header">
  <span id="beat-counter">Beat — / {len(state.beats)}</span>
  <div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>
  <span id="stats-text">Loading…</span>
</div>

<!-- Compact info bar: all beat metadata + ML suggestion in one horizontal strip -->
<div id="info-bar">
  <div class="ib-item"><span class="ib-key">Queue</span>  <span id="ib-queue" class="ib-val">—</span></div>
  <div class="ib-item"><span class="ib-key">Peak</span>   <span id="ib-pid"   class="ib-val">—</span></div>
  <div class="ib-item"><span class="ib-key">Old</span>    <span id="ib-old"   class="ib-val">—</span></div>
  <div class="ib-item"><span class="ib-key">p_art</span>  <span id="ib-part"  class="ib-val">—</span></div>
  <div class="ib-item"><span class="ib-key">Dis</span>    <span id="ib-dis"   class="ib-val">—</span></div>
  <div class="ib-item"><span class="ib-key">RR</span>     <span id="ib-rr"     class="ib-val">—</span></div>
  <div class="ib-item"><span class="ib-key">Corr</span>   <span id="ib-corr"   class="ib-val">—</span></div>
  <div class="ib-item"><span class="ib-key">Cat</span>    <span id="ib-suggest" class="ib-val">—</span></div>
  <div id="ib-ml">
    <span id="ib-ml-name">—</span>
    <span id="ib-ml-reason">—</span>
  </div>
</div>

<div id="main">
  <div id="plot-col"><div id="ecg-plot"></div></div>

  <div id="side-col">
    <div id="tags-wrap">

      <!-- ── Tags section ── -->
      <div class="sec-header">
        <span class="sec-title">Tags  (space to advance · m to mirror)</span>
        <button class="sec-btn" onclick="showAddTagForm()">+ Add</button>
      </div>
      <div id="add-tag-form" class="add-form">
        <input id="new-tag-name" class="add-input" placeholder="tag_name (underscores)" maxlength="40">
        <input id="new-tag-desc" class="add-input" placeholder="Description (optional)" maxlength="80">
        <div class="add-btns">
          <button class="form-cancel" onclick="cancelAddTagForm()">Cancel</button>
          <button class="form-submit" onclick="submitAddTagForm()">Add →</button>
        </div>
      </div>
      <div id="tag-pool"></div>

      <div class="sec-divider" style="margin-top:10px;"></div>

      <!-- ── Tag sets section ── -->
      <div class="sec-header">
        <span class="sec-title">Tag Sets</span>
        <button class="sec-btn" onclick="showSaveSetForm()">Save current</button>
      </div>
      <div id="save-set-form" class="add-form">
        <input id="new-set-name" class="add-input" placeholder="Set name" maxlength="40">
        <div class="add-btns">
          <button class="form-cancel" onclick="cancelSaveSetForm()">Cancel</button>
          <button class="form-submit" onclick="submitSaveSet()">Save →</button>
        </div>
      </div>
      <div id="tag-sets-pool"></div>

    </div><!-- #tags-wrap -->

    <div id="side-bottom">
      <div id="nav-row">
        <button class="nav-btn" onclick="navigate(-1)">← Prev</button>
        <button class="nav-btn" onclick="navigate(1)">Next →</button>
      </div>
      <div id="jump-row">
        <input id="jump-input" type="number" min="1" placeholder="Jump to #…">
        <button id="jump-go" onclick="jumpTo()">Go</button>
      </div>
    </div>

    <div id="status-bar">Space to advance · m to mirror previous · ← → navigate</div>
  </div>
</div>

<script>
// ── Injected from Python ──────────────────────────────────────────────────────
const TOTAL = {len(state.beats)};
let tagPool  = {initial_tag_pool};
let tagSets  = {initial_tag_sets};

// ── Session state ─────────────────────────────────────────────────────────────
let currentIdx    = {first_untagged};
let beatCache     = {{}};
let plotInitialized = false;
let selectedTags  = {{}};   // tagName → rating (null | int 1–5)
let activeTagName = null;   // tag currently showing rating bubbles
let prevTags      = {{}};   // selectedTags from the beat we just left
let keyBuffer     = '';     // accumulated leading zeros
let isDirty       = false;
let dragSrcIdx    = null;   // index of tag being dragged
let allCategories = [];     // suggested_category per beat index (from /api/categories)

const DEFAULT_STATUS = 'Space to advance · m to mirror · ← → navigate · Esc clear';

// ── Utilities ─────────────────────────────────────────────────────────────────
function esc(s) {{
  return String(s || '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// JS equivalent of Python's get_tag_key(position)
function getTagKey(pos) {{
  if (pos <= 9) return String(pos);
  pos -= 9;
  const zeros = Math.floor((pos - 1) / 9) + 1;
  const digit = (pos - 1) % 9 + 1;
  return '0'.repeat(zeros) + String(digit);
}}

function shortcutToPos(s) {{
  if (/^[1-9]$/.test(s)) return parseInt(s);
  const m = s.match(/^(0+)([1-9])$/);
  if (!m) return null;
  return 9 + (m[1].length - 1) * 9 + parseInt(m[2]);
}}

function setStatus(msg) {{
  document.getElementById('status-bar').textContent = msg;
}}

// ── ECG plot ──────────────────────────────────────────────────────────────────
function renderECG(beat) {{
  const ecg  = beat.context_ecg;
  const rIdx = beat.r_peak_index;
  if (!ecg || ecg.length === 0) return;
  let t;
  if (beat.context_ts_ns && beat.context_ts_ns.length === ecg.length) {{
    const t0 = beat.context_ts_ns[0];
    t = beat.context_ts_ns.map(v => (v - t0) / 1e9);
  }} else {{
    t = ecg.map((_, i) => i / 130.0);
  }}
  const rTime = rIdx != null ? t[rIdx] : null;
  const trace = {{ x: t, y: ecg, type: 'scatter', mode: 'lines',
                   line: {{ color: '#4a9eff', width: 1.2 }}, name: 'ECG' }};
  const shapes = rTime !== null ? [{{
    type: 'line', x0: rTime, x1: rTime, y0: 0, y1: 1,
    xref: 'x', yref: 'paper', line: {{ color: '#ff4444', width: 1.5, dash: 'dash' }},
  }}] : [];
  const layout = {{
    paper_bgcolor: '#0f0f1a', plot_bgcolor: '#0f0f1a',
    margin: {{ t: 30, b: 50, l: 55, r: 15 }},
    xaxis: {{ title: {{ text: 'Time (s)', font: {{ color: '#6060a0', size: 11 }} }},
              color: '#3d3d5e', gridcolor: '#1e1e38', zerolinecolor: '#2d2d4e' }},
    yaxis: {{ title: {{ text: 'ECG (mV)', font: {{ color: '#6060a0', size: 11 }} }},
              color: '#3d3d5e', gridcolor: '#1e1e38', zerolinecolor: '#2d2d4e' }},
    shapes, showlegend: false,
    annotations: rTime !== null ? [{{
      x: rTime, y: 1.05, xref: 'x', yref: 'paper',
      text: 'R', showarrow: false, font: {{ color: '#ff4444', size: 11 }},
    }}] : [],
  }};
  if (!plotInitialized) {{
    Plotly.newPlot('ecg-plot', [trace], layout, {{ responsive: true, displayModeBar: false }});
    plotInitialized = true;
  }} else {{
    Plotly.react('ecg-plot', [trace], layout);
  }}
}}

// ── Info bar (replaces old side-panel meta + suggestion) ─────────────────────
function computeStats(ecg, rIdx) {{
  if (!ecg || !ecg.length) return {{ std: 0, rAmp: 0, noise: 0 }};
  const n    = ecg.length;
  const mean = ecg.reduce((a, b) => a + b, 0) / n;
  const std  = Math.sqrt(ecg.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
  const rAmp = rIdx != null ? Math.abs(ecg[rIdx]) : 0;
  const el   = Math.floor(n * 0.2);
  const edge = [...ecg.slice(0, el), ...ecg.slice(n - el)];
  const em   = edge.reduce((a, b) => a + b, 0) / edge.length;
  const noise = Math.sqrt(edge.reduce((s, v) => s + (v - em) ** 2, 0) / edge.length);
  return {{ std, rAmp, noise }};
}}

const SUGGEST_COLORS = {{
  pristine:         '#2ecc71',
  low_amplitude:    '#3498db',
  aberrant_waveform:'#f39c12',
  chaotic:          '#e67e22',
  uninterpretable:  '#e74c3c',
}};

function suggestTag(beat) {{
  const {{ std, rAmp, noise }} = computeStats(beat.context_ecg, beat.r_peak_index);
  const p   = beat.p_artifact ?? 0.5;
  const old = beat.old_label  ?? '';
  if (p > 0.70 || old === 'artifact') {{
    if (noise < 0.04 && rAmp < 0.08)
      return {{ tag: 'chaotic',         reason: `Low-amp noise (σ=${{noise.toFixed(3)}})` }};
    return   {{ tag: 'uninterpretable', reason: `p_artifact=${{p.toFixed(3)}}, old=${{old}}` }};
  }}
  if (rAmp < 0.05)
    return {{ tag: 'low_amplitude',    reason: `R-amp=${{rAmp.toFixed(3)}} mV` }};
  if (noise > 0.12 || std > 0.18)
    return {{ tag: 'aberrant_waveform', reason: `noise=${{noise.toFixed(3)}}, std=${{std.toFixed(3)}}` }};
  return   {{ tag: 'pristine',         reason: `std=${{std.toFixed(3)}}, R-amp=${{rAmp.toFixed(3)}}` }};
}}

function renderInfoBar(beat) {{
  document.getElementById('ib-queue').textContent = beat.queue || '—';
  document.getElementById('ib-pid').textContent   = beat.peak_id ?? '—';
  document.getElementById('ib-old').textContent   = beat.old_label || '—';
  const fmt = (v, dec, suf = '') => v != null ? v.toFixed(dec) + suf : '—';
  const pa  = beat.p_artifact;
  document.getElementById('ib-part').textContent = pa != null ? pa.toFixed(3) : '—';
  document.getElementById('ib-part').style.color = pa == null ? '#8080a0'
    : pa > 0.7 ? '#e74c3c' : pa > 0.4 ? '#f39c12' : '#2ecc71';
  document.getElementById('ib-dis').textContent = fmt(beat.disagreement, 3);
  const rrp = beat.rr_prev_ms != null ? beat.rr_prev_ms.toFixed(0) : '—';
  const rrn = beat.rr_next_ms != null ? beat.rr_next_ms.toFixed(0) : '—';
  document.getElementById('ib-rr').textContent = `${{rrp}}→${{rrn}} ms`;
  const corr = beat.global_corr_clean;
  const ibCorr = document.getElementById('ib-corr');
  ibCorr.textContent = corr != null ? corr.toFixed(3) : '—';
  ibCorr.style.color = corr == null ? '#8080a0'
    : corr >= 0.8 ? '#2ecc71' : corr >= 0.5 ? '#f39c12' : '#e74c3c';
  const cat = beat.suggested_category;
  const CAT_COLORS = {{ clean_pristine: '#2ecc71', review_needed: '#f39c12', likely_artifact: '#e74c3c' }};
  const ibSuggest = document.getElementById('ib-suggest');
  ibSuggest.textContent = cat || '—';
  ibSuggest.style.color = CAT_COLORS[cat] || '#8080a0';
  const s   = suggestTag(beat);
  const col = SUGGEST_COLORS[s.tag] || '#7070a0';
  const ml  = document.getElementById('ib-ml');
  ml.style.background = col + '18';
  document.getElementById('ib-ml-name').textContent  = s.tag;
  document.getElementById('ib-ml-name').style.color  = col;
  document.getElementById('ib-ml-reason').textContent = s.reason;
}}

// ── Tag pool rendering (with drag-to-reorder) ─────────────────────────────────
function renderTagPool() {{
  const pool = document.getElementById('tag-pool');
  const scrollTop = pool.scrollTop;
  pool.innerHTML = '';

  tagPool.forEach((tag, idx) => {{
    const hasSel = Object.prototype.hasOwnProperty.call(selectedTags, tag.name);
    const rating = hasSel ? selectedTags[tag.name] : null;
    const isAct  = activeTagName === tag.name;

    const div = document.createElement('div');
    div.className = 'tag-btn' + (hasSel ? ' selected' : '') + (isAct ? ' active' : '');
    div.setAttribute('draggable', 'true');

    let rightHtml = '';
    if (isAct) {{
      rightHtml = '<span class="rating-bubbles">' +
        [1,2,3,4,5].map(n => {{
          const cls = 'rating-bubble' + (rating === n ? ' active' : '');
          return `<span class="${{cls}}" data-tag="${{esc(tag.name)}}" data-n="${{n}}">${{n}}</span>`;
        }}).join('') + '</span>';
    }} else if (hasSel && rating !== null) {{
      rightHtml = `<span class="tag-star">★${{rating}}</span>`;
    }} else if (hasSel) {{
      rightHtml = '<span class="tag-check">✓</span>';
    }}

    div.innerHTML = `
      <span class="drag-handle" title="Drag to reorder">⠿</span>
      <span class="tag-key">${{esc(tag.key)}}</span>
      <div class="tag-body">
        <div class="tag-top">
          <span class="tag-name">${{esc(tag.name)}}</span>
          ${{rightHtml}}
        </div>
        ${{!isAct ? `<div class="tag-desc">${{esc(tag.desc || '')}}</div>` : ''}}
      </div>`;

    // Click: toggle tag or rate
    div.addEventListener('click', e => {{
      if (e.target.classList.contains('rating-bubble')) {{
        const n    = parseInt(e.target.dataset.n);
        const name = e.target.dataset.tag;
        if (Object.prototype.hasOwnProperty.call(selectedTags, name)) {{
          selectedTags[name] = selectedTags[name] === n ? null : n;
          isDirty = true;
        }}
        if (activeTagName === name) activeTagName = null;
        renderTagPool();
        setStatus(DEFAULT_STATUS);
      }} else if (!e.target.classList.contains('drag-handle')) {{
        toggleTag(tag.name);
        setStatus(activeTagName
          ? `Rating "${{activeTagName}}": press 1–5 or click bubble · Esc to skip`
          : DEFAULT_STATUS);
      }}
    }});

    // Drag-to-reorder
    div.addEventListener('dragstart', e => {{
      dragSrcIdx = idx;
      activeTagName = null;
      e.dataTransfer.effectAllowed = 'move';
      // Delay adding class so drag preview is not dimmed
      setTimeout(() => div.classList.add('dragging'), 0);
    }});
    div.addEventListener('dragend', () => {{
      div.classList.remove('dragging');
      dragSrcIdx = null;
    }});
    div.addEventListener('dragover', e => {{
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      div.classList.add('drag-over');
    }});
    div.addEventListener('dragleave', e => {{
      // Only remove class if leaving the element entirely (not entering a child)
      if (!div.contains(e.relatedTarget)) div.classList.remove('drag-over');
    }});
    div.addEventListener('drop', e => {{
      e.preventDefault();
      div.classList.remove('drag-over');
      if (dragSrcIdx !== null && dragSrcIdx !== idx) {{
        const item = tagPool.splice(dragSrcIdx, 1)[0];
        tagPool.splice(idx, 0, item);
        // Recompute shortcut keys after reorder
        tagPool = tagPool.map((t, i) => ({{ ...t, key: getTagKey(i + 1) }}));
        renderTagPool();
        persistTagOrder();  // fire-and-forget save to server
      }}
      dragSrcIdx = null;
    }});

    pool.appendChild(div);
  }});

  pool.scrollTop = scrollTop;
}}

// ── Tag sets rendering ────────────────────────────────────────────────────────
function renderTagSets() {{
  const pool = document.getElementById('tag-sets-pool');
  pool.innerHTML = '';
  if (tagSets.length === 0) {{
    pool.innerHTML = '<div class="sets-empty">No saved sets yet</div>';
    return;
  }}
  tagSets.forEach(set => {{
    const div = document.createElement('div');
    div.className = 'set-row';
    const names   = set.tags.map(t => t.name);
    const preview = names.slice(0, 4).join(' · ') + (names.length > 4 ? ` +${{names.length - 4}}` : '');
    div.innerHTML = `
      <button class="set-apply" title="Apply this set">▶</button>
      <div class="set-info">
        <span class="set-name">${{esc(set.name)}}</span>
        <span class="set-preview">${{esc(preview)}}</span>
      </div>
      <button class="set-delete" title="Delete">×</button>`;
    div.querySelector('.set-apply').addEventListener('click', () => applyTagSet(set.name));
    div.querySelector('.set-info').addEventListener('click',  () => applyTagSet(set.name));
    div.querySelector('.set-delete').addEventListener('click', () => deleteTagSet(set.name));
    pool.appendChild(div);
  }});
}}

// ── Tag toggle / interaction ──────────────────────────────────────────────────
function toggleTag(name) {{
  if (Object.prototype.hasOwnProperty.call(selectedTags, name)) {{
    delete selectedTags[name];
    if (activeTagName === name) activeTagName = null;
  }} else {{
    selectedTags[name] = null;
    activeTagName = name;
  }}
  isDirty = true;
  renderTagPool();
}}

function applyTagSet(name) {{
  const set = tagSets.find(s => s.name === name);
  if (!set) return;
  selectedTags  = {{}};
  activeTagName = null;
  set.tags.forEach(t => {{ selectedTags[t.name] = t.rating ?? null; }});
  isDirty = true;
  renderTagPool();
  setStatus(`Applied set "${{name}}" · Space to advance`);
}}

async function deleteTagSet(name) {{
  const data = await fetch('/api/tagsets/delete', {{
    method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ name }}),
  }}).then(r => r.json());
  if (data.ok) {{ tagSets = data.sets; renderTagSets(); }}
}}

// ── Header stats ──────────────────────────────────────────────────────────────
function updateHeader(beat) {{
  const s   = beat.stats;
  const pct = s.total > 0 ? (s.annotated / s.total * 100).toFixed(1) : '0.0';
  document.getElementById('beat-counter').textContent =
    `Beat ${{(beat.idx + 1).toLocaleString()}} / ${{s.total.toLocaleString()}}`;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('stats-text').textContent =
    `${{s.annotated}} annotated · ${{s.remaining}} remaining (${{pct}}%)`;
}}

// ── Beat loading ──────────────────────────────────────────────────────────────
async function loadBeat(idx) {{
  idx = Math.max(0, Math.min(idx, TOTAL - 1));
  currentIdx = idx;
  let beat;
  if (beatCache[idx]) {{
    beat = beatCache[idx];
  }} else {{
    beat = await fetch('/api/beat?idx=' + idx).then(r => r.json());
    const ks = Object.keys(beatCache);
    if (ks.length > 20) delete beatCache[ks[0]];
    beatCache[idx] = beat;
  }}
  selectedTags  = {{}};
  activeTagName = null;
  isDirty       = false;
  (beat.current_tags || []).forEach(t => {{ selectedTags[t.name] = t.rating ?? null; }});
  renderECG(beat);
  renderInfoBar(beat);
  renderTagPool();
  updateHeader(beat);
  setStatus(DEFAULT_STATUS);
  if (idx + 1 < TOTAL && !beatCache[idx + 1])
    fetch('/api/beat?idx=' + (idx + 1)).then(r => r.json())
      .then(b => {{ beatCache[idx + 1] = b; }});
}}

// ── Save current beat's tags ──────────────────────────────────────────────────
async function saveCurrent() {{
  const tags = Object.entries(selectedTags).map(([name, rating]) => ({{ name, rating }}));
  const data = await fetch('/api/annotate', {{
    method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ idx: currentIdx, tags }}),
  }}).then(r => r.json());
  if (data.stats) {{
    const s   = data.stats;
    const pct = s.total > 0 ? (s.annotated / s.total * 100).toFixed(1) : '0.0';
    document.getElementById('stats-text').textContent =
      `${{s.annotated}} annotated · ${{s.remaining}} remaining (${{pct}}%)`;
    document.getElementById('progress-bar').style.width = pct + '%';
  }}
  delete beatCache[currentIdx];
  isDirty = false;
}}

async function persistTagOrder() {{
  await fetch('/api/tags/reorder', {{
    method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ names: tagPool.map(t => t.name) }}),
  }});
}}

// ── Navigation ────────────────────────────────────────────────────────────────
async function advance() {{
  prevTags = {{ ...selectedTags }};  // capture BEFORE overwriting
  await saveCurrent();
  keyBuffer = ''; activeTagName = null;
  await loadBeat(currentIdx + 1);
}}

async function navigate(delta) {{
  prevTags = {{ ...selectedTags }};  // capture BEFORE overwriting
  if (isDirty) await saveCurrent();
  keyBuffer = ''; activeTagName = null;
  await loadBeat(currentIdx + delta);
}}

function jumpTo() {{
  const v = parseInt(document.getElementById('jump-input').value, 10);
  if (!isNaN(v) && v >= 1 && v <= TOTAL) navigate(v - 1 - currentIdx);
}}

async function shutdown() {{
  if (isDirty) await saveCurrent();
  fetch('/api/shutdown');
  setStatus('Saving and shutting down…');
  setTimeout(() => window.close(), 800);
}}

// ── Keyboard ──────────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {{
  if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  const k = e.key;

  if (k === ' ')          {{ e.preventDefault(); advance(); return; }}
  if (k === 'ArrowLeft')  {{ navigate(-1); return; }}
  if (k === 'ArrowRight') {{ navigate(1);  return; }}
  if (k === 'q' || k === 'Q') {{ shutdown(); return; }}

  // Mirror previous beat's tags
  if (k === 'm' || k === 'M') {{
    if (Object.keys(prevTags).length > 0) {{
      selectedTags  = {{ ...prevTags }};
      activeTagName = null;
      isDirty = true;
      renderTagPool();
      setStatus('Mirrored tags from previous beat · Space to advance');
    }} else {{
      setStatus('No previous tags to mirror yet');
    }}
    return;
  }}

  // Jump shortcuts — Shift+R MUST be checked before plain r/R
  if (e.shiftKey && k === 'R') {{
    const next = allCategories.findIndex((c, i) => i > currentIdx && c === 'likely_artifact');
    if (next !== -1) {{ loadBeat(next); }}
    else {{ setStatus('No more likely_artifact beats after this position'); }}
    return;
  }}
  if (k === 'r' || k === 'R') {{
    const next = allCategories.findIndex((c, i) => i > currentIdx && c === 'review_needed');
    if (next !== -1) {{ loadBeat(next); }}
    else {{ setStatus('No more review_needed beats after this position'); }}
    return;
  }}

  if (k === 'Escape') {{
    keyBuffer = ''; activeTagName = null;
    renderTagPool(); setStatus(DEFAULT_STATUS); return;
  }}
  if (k === 'Backspace' && keyBuffer.length > 0) {{
    e.preventDefault();
    keyBuffer = keyBuffer.slice(0, -1);
    setStatus(keyBuffer ? `Buffer: ${{keyBuffer}}_ — type 1–9` : DEFAULT_STATUS);
    return;
  }}

  const d = parseInt(k, 10);
  if (isNaN(d)) return;

  // Rating mode: 1-5 rates the active tag
  if (activeTagName !== null && d >= 1 && d <= 5) {{
    selectedTags[activeTagName] = selectedTags[activeTagName] === d ? null : d;
    isDirty = true;
    activeTagName = null;
    renderTagPool(); setStatus(DEFAULT_STATUS); return;
  }}
  if (activeTagName !== null) {{
    activeTagName = null; renderTagPool();
  }}

  if (d === 0) {{
    keyBuffer += '0';
    setStatus(`Buffer: ${{keyBuffer}}_ — type digit 1–9`); return;
  }}

  const shortcut = keyBuffer + k;
  keyBuffer = '';
  const pos = shortcutToPos(shortcut);
  if (pos !== null && pos >= 1 && pos <= tagPool.length) {{
    toggleTag(tagPool[pos - 1].name);
    setStatus(activeTagName
      ? `Rating "${{activeTagName}}": press 1–5 or click bubble · Esc to skip`
      : DEFAULT_STATUS);
  }} else {{
    setStatus(DEFAULT_STATUS);
  }}
}});

document.getElementById('jump-input').addEventListener('keydown', e => {{
  if (e.key === 'Enter') jumpTo();
}});

// ── Add tag form ──────────────────────────────────────────────────────────────
function showAddTagForm() {{
  document.getElementById('add-tag-form').classList.add('open');
  document.getElementById('new-tag-name').focus();
}}
function cancelAddTagForm() {{
  document.getElementById('add-tag-form').classList.remove('open');
  document.getElementById('new-tag-name').value = '';
  document.getElementById('new-tag-desc').value = '';
}}
async function submitAddTagForm() {{
  const name = document.getElementById('new-tag-name').value.trim();
  const desc = document.getElementById('new-tag-desc').value.trim();
  if (!name) {{ document.getElementById('new-tag-name').focus(); return; }}
  const data = await fetch('/api/tags/add', {{
    method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ name, desc }}),
  }}).then(r => r.json());
  if (data.ok) {{
    tagPool = data.tags;
    cancelAddTagForm(); renderTagPool();
  }} else {{
    const inp = document.getElementById('new-tag-name');
    inp.style.borderColor = '#e74c3c';
    inp.title = data.error || 'Error';
    setTimeout(() => {{ inp.style.borderColor = ''; }}, 1500);
  }}
}}
document.getElementById('new-tag-name').addEventListener('input', e => {{
  e.target.value = e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_');
}});
document.getElementById('new-tag-name').addEventListener('keydown', e => {{
  if (e.key === 'Enter')  submitAddTagForm();
  if (e.key === 'Escape') cancelAddTagForm();
}});
document.getElementById('new-tag-desc').addEventListener('keydown', e => {{
  if (e.key === 'Enter')  submitAddTagForm();
  if (e.key === 'Escape') cancelAddTagForm();
}});

// ── Save tag set form ─────────────────────────────────────────────────────────
function showSaveSetForm() {{
  if (Object.keys(selectedTags).length === 0) {{
    setStatus('Select at least one tag before saving a set');
    return;
  }}
  document.getElementById('save-set-form').classList.add('open');
  document.getElementById('new-set-name').focus();
}}
function cancelSaveSetForm() {{
  document.getElementById('save-set-form').classList.remove('open');
  document.getElementById('new-set-name').value = '';
}}
async function submitSaveSet() {{
  const name = document.getElementById('new-set-name').value.trim();
  if (!name) {{ document.getElementById('new-set-name').focus(); return; }}
  const tags = Object.entries(selectedTags).map(([n, r]) => ({{ name: n, rating: r }}));
  const data = await fetch('/api/tagsets/save', {{
    method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ name, tags }}),
  }}).then(r => r.json());
  if (data.ok) {{
    tagSets = data.sets;
    cancelSaveSetForm(); renderTagSets();
    setStatus(`Saved set "${{name}}"`);
  }} else {{
    const inp = document.getElementById('new-set-name');
    inp.style.borderColor = '#e74c3c';
    inp.title = data.error || 'Error';
    setTimeout(() => {{ inp.style.borderColor = ''; }}, 1500);
  }}
}}
document.getElementById('new-set-name').addEventListener('keydown', e => {{
  if (e.key === 'Enter')  submitSaveSet();
  if (e.key === 'Escape') cancelSaveSetForm();
}});

// ── Boot ──────────────────────────────────────────────────────────────────────
renderTagPool();
renderTagSets();
loadBeat(currentIdx);
fetch('/api/categories').then(r => r.json()).then(d => {{ allCategories = d.categories || []; }});
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Web-based beat tag annotation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--queue-dir", type=Path,
                   default=Path("data/annotation_queues"))
    p.add_argument("--output", type=Path,
                   default=Path("data/annotation_queues/beat_tags.csv"))
    p.add_argument("--port", type=int, default=7432)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument(
        "--presorted", type=Path, default=None,
        metavar="PATH",
        help="CSV from presort_reannotation_queue.py — reorders beats by template "
             "correlation and attaches global_corr_clean / suggested_category",
    )
    args = p.parse_args()

    if not args.queue_dir.exists():
        logger.error("Queue directory not found: %s", args.queue_dir)
        sys.exit(1)

    tag_pool_path  = args.queue_dir / "tag_pool.json"
    tag_sets_path  = args.queue_dir / "tag_sets.json"
    tag_pool       = load_tag_pool(tag_pool_path)
    tag_sets       = load_tag_sets(tag_sets_path)

    if not (args.queue_dir / "tag_pool.json").exists():
        # First run: persist the defaults so future reorders are saved
        save_tag_pool(tag_pool_path, tag_pool)
        logger.info("Initialized tag pool with %d default tags → %s",
                    len(tag_pool), tag_pool_path)
    else:
        logger.info("Loaded %d tags from %s", len(tag_pool), tag_pool_path)
    if tag_sets:
        logger.info("Loaded %d tag set(s) from %s", len(tag_sets), tag_sets_path)

    logger.info("Scanning queue directories under %s …", args.queue_dir)
    beats = load_all_beats(args.queue_dir)
    if not beats:
        logger.error("No annotated beats found. Do completed.csv files exist?")
        sys.exit(1)

    if args.presorted is not None:
        if not args.presorted.exists():
            logger.error("Presort file not found: %s", args.presorted)
            sys.exit(1)
        logger.info("Applying presort order from %s …", args.presorted)
        beats = apply_presort_order(beats, args.presorted)

    existing_tags = load_existing_tags(args.output)
    state = _State(beats, existing_tags, args.output,
                   tag_pool_path, tag_pool, tag_sets_path, tag_sets)
    stats = state.stats()
    logger.info("Session: %d total · %d annotated · %d remaining",
                stats["total"], stats["annotated"], stats["remaining"])

    Handler = make_handler(state)
    server  = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url     = f"http://localhost:{args.port}"
    logger.info("Server at %s  (Ctrl-C or Q in browser to quit)", url)

    threading.Thread(target=server.serve_forever, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        state.shutdown_flag.wait()
    except KeyboardInterrupt:
        pass

    server.shutdown()
    final = state.stats()
    logger.info("Done. %d annotated · %d remaining.  Output: %s",
                final["annotated"], final["remaining"], args.output.resolve())


if __name__ == "__main__":
    main()
