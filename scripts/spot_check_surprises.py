#!/usr/bin/env python3
"""
scripts/spot_check_surprises.py — Minimal browser tool for spot-checking
~100 surprising auto-category disagreements.

Optional step — only needed if you want to manually verify the auto-categorizer
on the beats where it most confidently disagrees with the reviewed labels.

Input:  surprising_disagreements.csv (from validate_auto_categories.py)
Output: data/auto_categorization/spot_check_results.csv

Each beat shows: old label, auto-category, confidence, and a feature value table.
Decision per beat: (1) auto-category is correct  /  (2) wrong — correct is X  /  (3) skip

Usage
-----
    python scripts/spot_check_surprises.py \\
        --surprises  data/auto_categorization/surprising_disagreements.csv \\
        --output     data/auto_categorization/spot_check_results.csv \\
        --port       7433
"""

from __future__ import annotations

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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_CATEGORIES = [
    "pristine", "clean_normal", "clean_low_amplitude", "clean_noisy",
    "artifact_morphology", "artifact_noise", "artifact_general",
    "baseline_wander", "uncertain",
]

KEY_FEATURES = [
    "global_corr_clean", "r_peak_snr", "window_hf_noise_rms",
    "window_iqr", "window_wander_slope", "window_zcr",
    "window_kurtosis", "window_energy_ratio",
]


def _load_surprises(path: Path) -> list[dict]:
    beats = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            beats.append(dict(row))
    return beats


def _make_handler(beats: list[dict], results: dict, lock: threading.Lock,
                  shutdown_flag: threading.Event, output_path: Path):

    def _json(handler, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): pass

        def do_GET(self):
            parsed = urlparse(self.path)
            qs     = parse_qs(parsed.query)

            if parsed.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                body = _build_html(beats, results).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif parsed.path == "/api/beat":
                idx = int(qs.get("idx", ["0"])[0])
                idx = max(0, min(idx, len(beats) - 1))
                _json(self, 200, {
                    "idx":   idx,
                    "total": len(beats),
                    "beat":  beats[idx],
                    "decision": results.get(str(idx)),
                    "annotated": sum(1 for v in results.values() if v and v["decision"] != "skip"),
                })

            elif parsed.path == "/api/shutdown":
                _json(self, 200, {"ok": True})
                shutdown_flag.set()

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length)) if length else {}

            if parsed.path == "/api/decide":
                idx      = int(body.get("idx", 0))
                decision = str(body.get("decision", "skip"))   # correct / wrong / skip
                correct_cat = str(body.get("correct_category", ""))
                with lock:
                    results[str(idx)] = {
                        "decision":          decision,
                        "correct_category":  correct_cat,
                        "decided_at":        datetime.now(timezone.utc).isoformat(),
                    }
                    _save_results(beats, results, output_path)
                _json(self, 200, {"ok": True,
                                   "annotated": sum(1 for v in results.values()
                                                    if v and v["decision"] != "skip")})
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def _save_results(beats: list[dict], results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, beat in enumerate(beats):
        r = results.get(str(i))
        rows.append({
            "idx":               i,
            "peak_id":           beat.get("peak_id", ""),
            "old_label":         beat.get("label", beat.get("old_label", "")),
            "auto_category":     beat.get("auto_category", beat.get("category", "")),
            "confidence":        beat.get("category_confidence", ""),
            "decision":          r["decision"]         if r else "",
            "correct_category":  r["correct_category"] if r else "",
            "decided_at":        r["decided_at"]        if r else "",
        })
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _build_html(beats: list[dict], results: dict) -> str:
    n = len(beats)
    cats_json = json.dumps(ALL_CATEGORIES)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Spot Check Surprises</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0f0f1a; color: #c0c0e0; font-family: monospace; font-size: 13px; }}
#header {{ background: #1a1a2e; padding: 10px 18px; display: flex; align-items: center; gap: 20px; border-bottom: 1px solid #2d2d4e; }}
#header h1 {{ font-size: 15px; color: #8080d0; }}
#progress-wrap {{ flex: 1; height: 6px; background: #2d2d4e; border-radius: 3px; }}
#progress-bar {{ height: 100%; background: #4a9eff; border-radius: 3px; width: 0%; transition: width 0.3s; }}
#stats {{ color: #6060a0; font-size: 11px; }}
#main {{ max-width: 900px; margin: 24px auto; padding: 0 18px; }}
.info-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 18px; }}
.info-box {{ background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 6px; padding: 10px 14px; }}
.info-label {{ color: #5050a0; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }}
.info-val {{ font-size: 15px; font-weight: bold; margin-top: 4px; }}
.val-clean {{ color: #2ecc71; }}
.val-artifact {{ color: #e74c3c; }}
.val-uncertain {{ color: #f39c12; }}
table.feats {{ width: 100%; border-collapse: collapse; margin-bottom: 18px; font-size: 12px; }}
table.feats th {{ background: #1a1a2e; color: #6060a0; padding: 5px 10px; text-align: left; border-bottom: 1px solid #2d2d4e; }}
table.feats td {{ padding: 4px 10px; border-bottom: 1px solid #1e1e38; color: #a0a0c0; }}
table.feats tr:hover td {{ background: #1a1a2e; }}
.decision-row {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
button {{ background: #2d2d4e; border: 1px solid #3d3d6e; border-radius: 5px; color: #c0c0e0; cursor: pointer; padding: 8px 18px; font-family: monospace; font-size: 13px; }}
button:hover {{ background: #3d3d6e; }}
button.correct {{ border-color: #2ecc71; color: #2ecc71; }}
button.wrong   {{ border-color: #e74c3c; color: #e74c3c; }}
button.skip    {{ border-color: #6060a0; color: #6060a0; }}
button.decided {{ opacity: 0.55; }}
select {{ background: #1a1a2e; border: 1px solid #3d3d6e; color: #c0c0e0; border-radius: 4px; padding: 6px 10px; font-family: monospace; font-size: 13px; }}
#nav-row {{ display: flex; gap: 10px; margin-top: 14px; }}
#status {{ color: #5050a0; font-size: 11px; margin-top: 8px; min-height: 16px; }}
#wrong-row {{ display: none; align-items: center; gap: 10px; margin-top: 8px; }}
</style>
</head>
<body>
<div id="header">
  <h1>Spot-Check Surprises</h1>
  <div id="progress-wrap"><div id="progress-bar"></div></div>
  <span id="stats">0 / {n} decided</span>
</div>
<div id="main">
  <div class="info-grid">
    <div class="info-box"><div class="info-label">Beat #</div><div class="info-val" id="i-idx">—</div></div>
    <div class="info-box"><div class="info-label">Peak ID</div><div class="info-val" id="i-pid">—</div></div>
    <div class="info-box"><div class="info-label">Old label</div><div class="info-val" id="i-old">—</div></div>
    <div class="info-box"><div class="info-label">Auto-category (confidence)</div><div class="info-val" id="i-cat">—</div></div>
  </div>
  <table class="feats" id="feat-table"><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody></tbody></table>
  <div class="decision-row">
    <button class="correct" onclick="decide('correct')" title="[1]">[1] Correct</button>
    <button class="wrong"   onclick="showWrongRow()"    title="[2]">[2] Wrong — correct is…</button>
    <button class="skip"    onclick="decide('skip')"    title="[3]">[3] Skip</button>
  </div>
  <div id="wrong-row">
    <select id="cat-select">{_cat_options()}</select>
    <button class="wrong" onclick="decideWrong()">Confirm wrong</button>
    <button onclick="hideWrongRow()">Cancel</button>
  </div>
  <div id="nav-row">
    <button onclick="nav(-1)">← Prev</button>
    <button onclick="nav(1)">Next →</button>
    <button onclick="shutdown()" style="margin-left:auto">Q Quit &amp; Save</button>
  </div>
  <div id="status"></div>
</div>
<script>
const TOTAL = {n};
const CATS  = {cats_json};
let currentIdx = 0;
let beatCache  = {{}};

function catColor(c) {{
  if (!c) return '#8080a0';
  if (['pristine','clean_normal','clean_low_amplitude','clean_noisy'].includes(c)) return '#2ecc71';
  if (['artifact_morphology','artifact_noise','artifact_general','baseline_wander'].includes(c)) return '#e74c3c';
  return '#f39c12';
}}

function _catOptions() {{
  return CATS.map(c => `<option value="${{c}}">${{c}}</option>`).join('');
}}

async function loadBeat(idx) {{
  idx = Math.max(0, Math.min(idx, TOTAL - 1));
  currentIdx = idx;
  let d;
  if (beatCache[idx]) {{ d = beatCache[idx]; }}
  else {{
    d = await fetch('/api/beat?idx=' + idx).then(r => r.json());
    beatCache[idx] = d;
  }}
  const b = d.beat;
  document.getElementById('i-idx').textContent = (idx + 1) + ' / ' + TOTAL;
  document.getElementById('i-pid').textContent = b.peak_id || '—';
  const oldEl = document.getElementById('i-old');
  oldEl.textContent = b.label || b.old_label || '—';
  oldEl.className = 'info-val ' + (oldEl.textContent === 'artifact' ? 'val-artifact' : 'val-clean');
  const cat = b.auto_category || b.category || '—';
  const conf = b.category_confidence ? parseFloat(b.category_confidence).toFixed(3) : '';
  const catEl = document.getElementById('i-cat');
  catEl.textContent = cat + (conf ? ' (' + conf + ')' : '');
  catEl.style.color = catColor(cat);
  // Feature table
  const skip = new Set(['peak_id','label','old_label','auto_category','category','category_confidence','surprise_score','tree_path','leaf_id']);
  const tbody = document.querySelector('#feat-table tbody');
  tbody.innerHTML = '';
  for (const [k, v] of Object.entries(b)) {{
    if (skip.has(k)) continue;
    const tr = document.createElement('tr');
    const val = isNaN(v) ? v : parseFloat(v).toFixed(4);
    tr.innerHTML = `<td>${{k}}</td><td>${{val}}</td>`;
    tbody.appendChild(tr);
  }}
  // Show prior decision if any
  if (d.decision) {{
    const dd = d.decision;
    const msg = dd.decision === 'correct' ? '✓ Marked correct'
              : dd.decision === 'wrong'   ? '✗ Marked wrong → ' + dd.correct_category
              : '— Skipped';
    document.getElementById('status').textContent = msg;
  }} else {{
    document.getElementById('status').textContent = '';
  }}
  updateStats(d.annotated);
  hideWrongRow();
  // Prefetch next
  if (idx + 1 < TOTAL && !beatCache[idx + 1])
    fetch('/api/beat?idx=' + (idx + 1)).then(r => r.json()).then(b => {{ beatCache[idx + 1] = b; }});
}}

function updateStats(annotated) {{
  const pct = TOTAL > 0 ? (annotated / TOTAL * 100).toFixed(1) : '0';
  document.getElementById('stats').textContent = annotated + ' / ' + TOTAL + ' decided';
  document.getElementById('progress-bar').style.width = pct + '%';
}}

async function decide(decision, correctCat) {{
  const resp = await fetch('/api/decide', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{ idx: currentIdx, decision, correct_category: correctCat || '' }}),
  }}).then(r => r.json());
  updateStats(resp.annotated);
  delete beatCache[currentIdx];
  const msg = decision === 'correct' ? '✓ Marked correct'
            : decision === 'wrong'   ? '✗ Marked wrong → ' + correctCat
            : '— Skipped';
  document.getElementById('status').textContent = msg;
  hideWrongRow();
  if (currentIdx + 1 < TOTAL) loadBeat(currentIdx + 1);
}}

function showWrongRow() {{
  document.getElementById('wrong-row').style.display = 'flex';
}}
function hideWrongRow() {{
  document.getElementById('wrong-row').style.display = 'none';
}}
function decideWrong() {{
  const cat = document.getElementById('cat-select').value;
  decide('wrong', cat);
}}
function nav(delta) {{ loadBeat(currentIdx + delta); }}
async function shutdown() {{
  fetch('/api/shutdown');
  document.getElementById('status').textContent = 'Saving and shutting down…';
  setTimeout(() => window.close(), 800);
}}

document.addEventListener('keydown', e => {{
  if (['INPUT','SELECT'].includes(e.target.tagName)) return;
  if (e.key === '1') decide('correct');
  else if (e.key === '2') showWrongRow();
  else if (e.key === '3') decide('skip');
  else if (e.key === 'ArrowLeft')  nav(-1);
  else if (e.key === 'ArrowRight') nav(1);
  else if (e.key === 'q' || e.key === 'Q') shutdown();
  else if (e.key === 'Enter' && document.getElementById('wrong-row').style.display === 'flex') decideWrong();
  else if (e.key === 'Escape') hideWrongRow();
}});

loadBeat(0);
</script>
</body>
</html>"""


def _cat_options() -> str:
    return "".join(f'<option value="{c}">{c}</option>' for c in ALL_CATEGORIES)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="[OPTIONAL] Minimal browser tool for spot-checking ~100 auto-category disagreements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--surprises", type=Path,
                        default=Path("data/auto_categorization/surprising_disagreements.csv"),
                        help="CSV from validate_auto_categories.py")
    parser.add_argument("--output",    type=Path,
                        default=Path("data/auto_categorization/spot_check_results.csv"))
    parser.add_argument("--port",      type=int, default=7433)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.surprises.exists():
        logger.error("Surprises CSV not found: %s", args.surprises)
        sys.exit(1)

    beats = _load_surprises(args.surprises)
    logger.info("Loaded %d beats from %s", len(beats), args.surprises)

    results: dict = {}
    lock = threading.Lock()
    shutdown_flag = threading.Event()

    Handler = _make_handler(beats, results, lock, shutdown_flag, args.output)
    server  = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url     = f"http://localhost:{args.port}"
    logger.info("Server at %s", url)
    logger.info("Keyboard: [1]=correct  [2]=wrong  [3]=skip  [←/→]=navigate  [Q]=quit")

    threading.Thread(target=server.serve_forever, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        shutdown_flag.wait()
    except KeyboardInterrupt:
        pass

    server.shutdown()
    _save_results(beats, results, args.output)
    n_decided = sum(1 for v in results.values() if v and v["decision"] != "skip")
    logger.info("Done. %d / %d decided. Output → %s", n_decided, len(beats), args.output)


if __name__ == "__main__":
    main()
