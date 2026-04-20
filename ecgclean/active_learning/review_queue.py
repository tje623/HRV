#!/usr/bin/env python3
"""
ecgclean.active_learning.review_queue
======================================
Interactive terminal tool for reviewing exported annotation queue beats.

Displays each beat's ECG context window with matplotlib and accepts keyboard
shortcuts to label it.  Progress is saved after every beat so you can
stop and resume at any time.

Usage
-----
    python ecgclean/active_learning/review_queue.py \\
        --queue-dir data/annotation_queues/iteration_1/ \\
        --output data/annotation_queues/iteration_1/completed.csv

    # Resume at beat 329 (1-indexed position in sorted queue)
    python ecgclean/active_learning/review_queue.py \\
        --queue-dir data/annotation_queues/iteration_1/ \\
        --output data/annotation_queues/iteration_1/completed.csv \\
        --start-from 329

Controls
--------
    a          →  mark as artifact       (pending, not saved yet)
    c          →  mark as clean          (pending, not saved yet)
    i          →  mark as interpolated   (pending, not saved yet)
    p          →  mark as phys_event     (pending, not saved yet)
    m          →  mark as missed_original (pending, not saved yet)
    k          →  mark as skip           (pending, not saved yet)
    SPACE      →  confirm pending label and advance to next beat
    BACKSPACE  →  clear pending label (go back to existing / unlabeled)
    b / ←      →  go back to previous beat (no label change)
    0-9 + ENTER→  jump to beat N (1-indexed)
    q          →  quit and save progress

Options
-------
    --sort-by        Column to sort candidates by (default: composite_score desc)
    --sort-asc       Sort ascending instead of descending
    --start-from N   Start at beat N (1-indexed in sorted queue)
    --preview N      Show a grid preview of N beats before interactive mode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Matplotlib backend ────────────────────────────────────────────────────────
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
VALID_LABELS = {"clean", "artifact", "interpolated", "phys_event", "missed_original"}
SAMPLE_RATE_HZ = 130  # Polar H10 ECG sampling rate

# Label keys — pressing these sets a PENDING label (not saved until SPACE)
LABEL_KEYMAP = {
    "a": "artifact",
    "c": "clean",
    "i": "interpolated",
    "p": "phys_event",
    "m": "missed_original",
    "k": "__skip__",
}

LABEL_COLORS = {
    "artifact":         "#e74c3c",
    "clean":            "#2ecc71",
    "interpolated":     "#3498db",
    "phys_event":       "#9b59b6",
    "missed_original":  "#f39c12",
    "__skip__":         "#95a5a6",
    None:               "#546e7a",
}


# ═══════════════════════════════════════════════════════════════════════════ #
#  I/O helpers                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def _load_summary(queue_dir: Path) -> pd.DataFrame:
    csv_path = queue_dir / "queue_summary.csv"
    if not csv_path.exists():
        sys.exit(f"ERROR: queue_summary.csv not found in {queue_dir}")
    return pd.read_csv(csv_path)


def _load_json(queue_dir: Path, peak_id: int) -> dict:
    path = queue_dir / f"beat_{peak_id:08d}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_completed(output_path: Path) -> dict[int, str]:
    """Return {peak_id: label} for already-completed beats, including skips."""
    result: dict[int, str] = {}
    if output_path.exists():
        df = pd.read_csv(output_path)
        result.update(dict(zip(df["peak_id"].astype(int), df["label"].astype(str))))
    # Restore skipped beats so they show as already-handled on resume
    skip_path = output_path.with_name("skipped.csv")
    if skip_path.exists():
        skip_df = pd.read_csv(skip_path)
        if "peak_id" in skip_df.columns:
            for pid in skip_df["peak_id"].astype(int):
                result.setdefault(pid, "__skip__")
    return result


def _save_completed(output_path: Path, completed: dict[int, str]) -> None:
    # Save labeled beats (skips excluded — they have no label to record)
    rows = [(pid, lbl) for pid, lbl in completed.items() if not lbl.startswith("__")]
    df = pd.DataFrame(rows, columns=["peak_id", "label"])
    df.to_csv(output_path, index=False)
    # Persist skipped peak_ids separately so export scripts can exclude them
    skip_ids = [pid for pid, lbl in completed.items() if lbl == "__skip__"]
    skip_path = output_path.with_name("skipped.csv")
    pd.DataFrame({"peak_id": skip_ids}).to_csv(skip_path, index=False)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Rendering                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def _make_figure():
    fig = plt.figure(figsize=(14, 6.5), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 0.6, 0.5], hspace=0.08)
    ax_ecg   = fig.add_subplot(gs[0])
    ax_state = fig.add_subplot(gs[1])   # pending / confirmed label banner
    ax_info  = fig.add_subplot(gs[2])   # controls legend + progress
    for ax in (ax_ecg, ax_state, ax_info):
        ax.set_facecolor("#16213e")
        ax.axis("off")
    return fig, ax_ecg, ax_state, ax_info


def _render_beat(
    fig, ax_ecg, ax_state, ax_info,
    row: pd.Series,
    beat_json: dict,
    cursor_pos: int,       # 1-indexed display position
    total: int,
    completed_count: int,
    existing_label: str | None,
    pending_label: str | None,
    jump_buffer: str,
) -> None:
    ax_ecg.clear()
    ax_state.clear()
    ax_info.clear()
    for ax in (ax_ecg, ax_state, ax_info):
        ax.set_facecolor("#16213e")
        ax.axis("off")

    ecg   = beat_json.get("context_ecg", [])
    r_idx = beat_json.get("r_peak_index_in_context", 0)
    pid   = int(row["peak_id"])
    p_ens    = float(row.get("p_artifact_ensemble", 0))
    disagree = float(row.get("disagreement", 0))
    rr_prev  = row.get("rr_prev_ms")
    rr_next  = row.get("rr_next_ms")
    seg_idx  = int(row.get("segment_idx", -1))

    # ── ECG trace ─────────────────────────────────────────────────────────
    ax_ecg.set_facecolor("#16213e")
    ax_ecg.axis("on")
    if ecg:
        t = np.arange(len(ecg)) / SAMPLE_RATE_HZ
        ax_ecg.plot(t, ecg, color="#4fc3f7", linewidth=0.8, alpha=0.9)

        if 0 <= r_idx < len(ecg):
            ax_ecg.axvline(t[r_idx], color="#ff6b6b", linewidth=1.5,
                           linestyle="--", alpha=0.8)
            ax_ecg.scatter([t[r_idx]], [ecg[r_idx]], color="#ff6b6b", s=60, zorder=5)

        qrs_hw = 0.08
        ax_ecg.axvspan(t[r_idx] - qrs_hw, t[r_idx] + qrs_hw,
                       alpha=0.15, color="#ffeb3b")

        ax_ecg.set_xlim(t[0], t[-1])
        ax_ecg.set_ylabel("ECG (mV)", color="#b0bec5", fontsize=9)
        ax_ecg.tick_params(colors="#b0bec5", labelsize=8)
        for spine in ax_ecg.spines.values():
            spine.set_color("#37474f")
    else:
        ax_ecg.text(0.5, 0.5, "No ECG data available", transform=ax_ecg.transAxes,
                    ha="center", va="center", color="#ff6b6b", fontsize=14)

    rr_str = (f"RR: {rr_prev:.0f}→{rr_next:.0f} ms"
              if pd.notna(rr_prev) and pd.notna(rr_next) else "RR: N/A")
    title = (f"Beat #{cursor_pos}  |  peak_id={pid}  |  seg {seg_idx}  |  "
             f"p_artifact={p_ens:.3f}  disagree={disagree:.3f}  {rr_str}")
    ax_ecg.set_title(title, color="#eceff1", fontsize=10, pad=6)

    # ── Label state banner ────────────────────────────────────────────────
    ax_state.set_xlim(0, 1)
    ax_state.set_ylim(0, 1)
    ax_state.axis("off")

    if pending_label is not None:
        # A label is pending — show it highlighted, waiting for SPACE
        lbl_display = pending_label.lstrip("_")
        color = LABEL_COLORS.get(pending_label, "#eceff1")
        ax_state.add_patch(plt.Rectangle((0.02, 0.1), 0.96, 0.8,
                                         facecolor=color, alpha=0.25,
                                         transform=ax_state.transAxes))
        ax_state.text(0.5, 0.5,
                      f"PENDING: {lbl_display.upper()}  —  press SPACE to confirm, BACKSPACE to clear",
                      va="center", ha="center", color=color,
                      fontsize=11, fontweight="bold",
                      transform=ax_state.transAxes)
    elif existing_label is not None:
        # Already labeled — show saved label in muted style
        lbl_display = existing_label.lstrip("_")
        color = LABEL_COLORS.get(existing_label, "#eceff1")
        ax_state.text(0.5, 0.5,
                      f"saved: {lbl_display}  —  press a label key to change, SPACE to re-confirm",
                      va="center", ha="center", color=color,
                      fontsize=10, alpha=0.8,
                      transform=ax_state.transAxes)
    else:
        ax_state.text(0.5, 0.5,
                      "unlabeled  —  press a label key, then SPACE to confirm",
                      va="center", ha="center", color="#546e7a",
                      fontsize=10, style="italic",
                      transform=ax_state.transAxes)

    # ── Info bar ──────────────────────────────────────────────────────────
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    progress = cursor_pos / total
    ax_info.barh(0.5, progress, height=0.6, left=0,
                 color="#4fc3f7", alpha=0.25, transform=ax_info.transAxes)

    left_text = f"#{cursor_pos}/{total}  |  saved: {completed_count}"
    if jump_buffer:
        left_text += f"  |  goto: {jump_buffer}_"
    ax_info.text(0.01, 0.5, left_text, va="center", ha="left",
                 color="#eceff1", fontsize=8.5, transform=ax_info.transAxes)

    controls = "[a]rtifact [c]lean [i]nterp [p]hys [m]issed [k]skip  |  SPACE=confirm  BKSP=clear  b/←=back  0-9+ENTER=goto  q=quit"
    ax_info.text(0.5, 0.02, controls, va="bottom", ha="center",
                 color="#546e7a", fontsize=8, transform=ax_info.transAxes,
                 style="italic")

    fig.canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Core review loop                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def review_interactive(
    queue_dir: Path,
    output_path: Path,
    sort_by: str = "composite_score",
    sort_ascending: bool = False,
    start_from: int = 1,
) -> None:
    summary = _load_summary(queue_dir)

    if sort_by in summary.columns:
        summary = summary.sort_values(sort_by, ascending=sort_ascending).reset_index(drop=True)

    beats = summary.reset_index(drop=True)   # all beats, 0-indexed internally
    total = len(beats)
    completed = _load_completed(output_path)

    print(f"Queue: {total} beats  |  Already saved: {len(completed)}")

    # ── Mutable state cells (modified inside key handler closure) ──────────
    # cursor is 0-indexed internally; displayed as 1-indexed to the user
    cursor       = [max(0, min(start_from - 1, total - 1))]
    pending      = [None]   # pending label (str or None)
    nav_action   = [None]   # "advance", "back", "goto:<N>", "quit"
    jump_buf     = [""]     # digit accumulation for goto

    # ── Key handler ───────────────────────────────────────────────────────
    def on_key(event):
        key = event.key

        # Label keys — set pending without advancing
        if key in LABEL_KEYMAP:
            pending[0] = LABEL_KEYMAP[key]
            return

        # Spacebar — confirm pending and advance
        if key == " ":
            nav_action[0] = "advance"
            return

        # Backspace — clear pending
        if key in ("backspace", "delete"):
            pending[0] = None
            return

        # Back navigation
        if key in ("left", "b"):
            nav_action[0] = "back"
            return

        # Quit
        if key == "q":
            nav_action[0] = "quit"
            return

        # Digit accumulation for goto
        if key.isdigit():
            jump_buf[0] += key
            return

        # Enter commits goto
        if key in ("enter", "return") and jump_buf[0]:
            nav_action[0] = f"goto:{jump_buf[0]}"
            jump_buf[0] = ""
            return

        # Escape cancels goto buffer
        if key == "escape":
            jump_buf[0] = ""
            return

    fig, ax_ecg, ax_state, ax_info = _make_figure()
    plt.tight_layout()
    plt.ion()
    plt.show()
    fig.canvas.mpl_connect("key_press_event", on_key)

    last_rendered_state = [object()]  # sentinel — force first render

    while True:
        i = cursor[0]
        row = beats.iloc[i]
        pid = int(row["peak_id"])

        # Load JSON only when cursor changes
        current_state = (i, pending[0], jump_buf[0])
        if current_state != last_rendered_state[0]:
            beat_json = _load_json(queue_dir, pid)
            existing  = completed.get(pid)
            _render_beat(
                fig, ax_ecg, ax_state, ax_info,
                row, beat_json,
                cursor_pos=i + 1,
                total=total,
                completed_count=sum(1 for v in completed.values() if not v.startswith("__")),
                existing_label=existing,
                pending_label=pending[0],
                jump_buffer=jump_buf[0],
            )
            last_rendered_state[0] = current_state

        plt.pause(0.05)

        # Window was closed
        if not plt.fignum_exists(fig.number):
            nav_action[0] = "quit"

        # ── Handle pending-label change (re-render without action) ────────
        new_state = (i, pending[0], jump_buf[0])
        if new_state != last_rendered_state[0]:
            continue  # loop back to re-render

        # ── Handle navigation actions ─────────────────────────────────────
        if nav_action[0] is None:
            continue

        action = nav_action[0]
        nav_action[0] = None

        if action == "quit":
            print(f"\nQuitting at beat #{i + 1} (peak {pid}).  Progress saved.")
            break

        elif action == "back":
            if i > 0:
                cursor[0] -= 1
                pending[0] = None
                # Print the label we're stepping back over
                prev_pid = int(beats.iloc[cursor[0]]["peak_id"])
                prev_lbl = completed.get(prev_pid, "unlabeled")
                print(f"  ← back to beat #{cursor[0] + 1}  (peak {prev_pid}, was: {prev_lbl})")
            else:
                print("  Already at first beat.")

        elif action == "advance":
            # Save pending label if set; if not set and already labeled, keep existing
            if pending[0] is not None:
                completed[pid] = pending[0]
                _save_completed(output_path, completed)
                lbl = pending[0]
                color_code = (
                    "\033[91m" if lbl == "artifact"
                    else "\033[92m" if lbl == "clean"
                    else "\033[93m"
                )
                display_lbl = lbl.lstrip("_").upper()
                print(f"  [#{i+1:>3}/{total}] peak {pid:>8}  p={row['p_artifact_ensemble']:.3f}  "
                      f"{color_code}{display_lbl}\033[0m")
                pending[0] = None

            if i < total - 1:
                cursor[0] += 1
            else:
                print("\nReached last beat.")

        elif action.startswith("goto:"):
            raw = action.split(":", 1)[1]
            try:
                target = int(raw)  # 1-indexed
                new_idx = max(0, min(target - 1, total - 1))
                print(f"  → jumping to beat #{new_idx + 1} (peak {int(beats.iloc[new_idx]['peak_id'])})")
                cursor[0] = new_idx
                pending[0] = None
            except ValueError:
                print(f"  Invalid goto target: {raw!r}")

    plt.close(fig)

    # Final save (skip entries excluded)
    real_labels = {pid: lbl for pid, lbl in completed.items() if not lbl.startswith("__")}
    _save_completed(output_path, real_labels)

    print(f"\nDone.  {len(real_labels)} beats saved to {output_path}")
    if real_labels:
        from collections import Counter
        counts = Counter(real_labels.values())
        for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {lbl}: {cnt}")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Preview grid                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def preview_grid(queue_dir: Path, summary: pd.DataFrame, n: int = 25) -> None:
    """Show a grid of N ECG traces (sorted by composite_score) for a quick overview."""
    sample = summary.nlargest(n, "composite_score")
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.5), facecolor="#1a1a2e")
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        pid = int(row["peak_id"])
        beat_json = _load_json(queue_dir, pid)
        ecg   = beat_json.get("context_ecg", [])
        r_idx = beat_json.get("r_peak_index_in_context", 0)
        ax.set_facecolor("#16213e")
        if ecg:
            t = np.arange(len(ecg)) / SAMPLE_RATE_HZ
            ax.plot(t, ecg, color="#4fc3f7", linewidth=0.6)
            if 0 <= r_idx < len(ecg):
                ax.axvline(t[r_idx], color="#ff6b6b", linewidth=1, linestyle="--")
        ax.set_title(f"{pid}\np={row['p_artifact_ensemble']:.2f}",
                     color="#b0bec5", fontsize=7, pad=2)
        ax.tick_params(labelbottom=False, labelleft=False, colors="#555")
        for sp in ax.spines.values():
            sp.set_color("#37474f")

    for ax in axes[len(sample):]:
        ax.set_visible(False)

    fig.suptitle(f"Top-{n} highest composite_score candidates (overview)",
                 color="#eceff1", fontsize=12)
    plt.tight_layout()
    plt.show(block=True)


# ═══════════════════════════════════════════════════════════════════════════ #
#  CLI                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="review_queue.py",
        description="Interactive ECG beat annotation reviewer.",
    )
    p.add_argument("--queue-dir", required=True,
                   help="Directory containing queue_summary.csv and beat_*.json files")
    p.add_argument("--output", required=True,
                   help="Path to write completed.csv (auto-resumed if exists)")
    p.add_argument("--sort-by", default="composite_score",
                   help="Column to sort by (default: composite_score desc)")
    p.add_argument("--sort-asc", action="store_true",
                   help="Sort ascending instead of descending")
    p.add_argument("--start-from", type=int, default=1, metavar="N",
                   help="Start at beat N (1-indexed, default: 1)")
    p.add_argument("--preview", type=int, default=0, metavar="N",
                   help="Show a grid of N beats before interactive mode (default: 0 = skip)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    queue_dir = Path(args.queue_dir)
    output    = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.preview > 0:
        summary = _load_summary(queue_dir)
        print(f"Showing preview grid of {args.preview} highest-score beats...")
        preview_grid(queue_dir, summary, n=args.preview)

    review_interactive(
        queue_dir=queue_dir,
        output_path=output,
        sort_by=args.sort_by,
        sort_ascending=args.sort_asc,
        start_from=args.start_from,
    )


if __name__ == "__main__":
    main()
