#!/usr/bin/env python3
"""
Export top-N beats from ensemble_preds.parquet into a review queue directory
(queue_summary.csv + beat_*.json) that review_queue.py can load for annotation.

Streams ECG context from ecg_samples.parquet via PyArrow predicate pushdown —
never loads the full 48 GB file into memory.

Sort strategies (--sort-by):
    disagreement      |p_tabular - p_cnn| — where the two models disagree most
    p_artifact_cnn    CNN's highest-confidence artifact predictions — best for
                      finding the CNN's false positives and correcting its training
    p_artifact_ensemble  overall ensemble score
    uncertainty_ensemble  beats closest to the 0.5 decision boundary

Usage
-----
    cd "/Volumes/xHRV/Artifact Detector"
    source /Users/tannereddy/.envs/hrv/bin/activate

    # CNN false-positive correction queue (recommended for CNN retraining):
    python export_disagreement_queue.py \\
        --ensemble-preds /Volumes/xHRV/processed/ensemble_preds.parquet \\
        --peaks          /Volumes/xHRV/processed/peaks.parquet \\
        --labels         /Volumes/xHRV/processed/labels.parquet \\
        --ecg-samples    /Volumes/xHRV/processed/ecg_samples.parquet \\
        --output         data/annotation_queues/cnn_confident_500/ \\
        --n-beats        500 \\
        --sort-by        p_artifact_cnn

    # Disagreement queue (original):
    python export_disagreement_queue.py \\
        --ensemble-preds /Volumes/xHRV/processed/ensemble_preds.parquet \\
        --peaks          /Volumes/xHRV/processed/peaks.parquet \\
        --labels         /Volumes/xHRV/processed/labels.parquet \\
        --ecg-samples    /Volumes/xHRV/processed/ecg_samples.parquet \\
        --output         data/annotation_queues/disagreement_500/ \\
        --n-beats        500 \\
        --sort-by        disagreement

Then review with:
    python ecgclean/active_learning/review_queue.py \\
        --queue-dir  data/annotation_queues/cnn_confident_500/ \\
        --output     data/annotation_queues/cnn_confident_500/completed.csv \\
        --sort-by    p_artifact_cnn
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE_HZ

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

CONTEXT_WINDOW_SEC = 5.0
HALF_WINDOW_MS = int(CONTEXT_WINDOW_SEC / 2.0 * 1000)


def main() -> None:
    args = _build_parser().parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ensemble predictions (only the columns we need) ──────────────
    log.info("Loading ensemble predictions …")
    ens = pd.read_parquet(
        args.ensemble_preds,
        columns=["peak_id", "p_artifact_tabular", "p_artifact_cnn",
                 "p_artifact_ensemble", "disagreement", "uncertainty_ensemble"],
    )
    log.info("  %d total beats", len(ens))

    # ── 2. Select top-N by chosen sort column ────────────────────────────────
    sort_col = args.sort_by
    valid_cols = {"disagreement", "p_artifact_cnn", "p_artifact_ensemble", "uncertainty_ensemble"}
    if sort_col not in valid_cols:
        log.error("--sort-by must be one of %s, got %r", valid_cols, sort_col)
        sys.exit(1)
    # ── 2a. Exclude already-annotated beats ──────────────────────────────────
    pool = ens
    if args.exclude_annotated:
        excluded_ids: set[int] = set()
        for csv_path in args.exclude_annotated:
            p = Path(csv_path)
            if not p.exists():
                log.warning("--exclude-annotated file not found, skipping: %s", p)
            else:
                done = pd.read_csv(p)
                if "peak_id" in done.columns:
                    excluded_ids.update(done["peak_id"].astype(int).tolist())
            # Auto-load skipped.csv from the same directory
            skip_p = p.parent / "skipped.csv"
            if skip_p.exists():
                skip_df = pd.read_csv(skip_p)
                if "peak_id" in skip_df.columns:
                    n_before = len(excluded_ids)
                    excluded_ids.update(skip_df["peak_id"].astype(int).tolist())
                    log.info("  Loaded %d skipped beats from %s",
                             len(excluded_ids) - n_before, skip_p)
        if excluded_ids:
            before = len(pool)
            pool = pool[~pool["peak_id"].isin(excluded_ids)]
            log.info("Excluded %d already-annotated beats → %d remain",
                     before - len(pool), len(pool))

    if args.max_ensemble is not None:
        before = len(pool)
        pool = pool[pool["p_artifact_ensemble"] < args.max_ensemble]
        log.info("Filtered to p_artifact_ensemble < %.2f: %d → %d beats",
                 args.max_ensemble, before, len(pool))
    if args.min_cnn is not None:
        before = len(pool)
        pool = pool[pool["p_artifact_cnn"] >= args.min_cnn]
        log.info("Filtered to p_artifact_cnn >= %.2f: %d → %d beats",
                 args.min_cnn, before, len(pool))

    # Over-request to compensate for edge beats that will be filtered later.
    n_request = min(len(pool), int(args.n_beats * args.oversample))
    cand = pool.nlargest(n_request, sort_col).copy().reset_index(drop=True)
    log.info("Selected top %d candidates by %s (target %d after edge filtering)",
             len(cand), sort_col, args.n_beats)

    # Composite score (same formula as sampler.py, priority term = 0 here)
    cand["composite_score"] = (
        0.4 * cand["uncertainty_ensemble"].values
        + 0.4 * cand["disagreement"].values
    ).astype("float32")

    # ── 3. Attach timestamp_ms + segment_idx from peaks.parquet ──────────────
    log.info("Loading peaks …")
    peaks = pd.read_parquet(args.peaks, columns=["peak_id", "timestamp_ms", "segment_idx"])
    cand = cand.merge(peaks, on="peak_id", how="left")
    n_missing_seg = cand["segment_idx"].isna().sum()
    if n_missing_seg:
        log.warning("  %d beats have no segment_idx (missing from peaks.parquet)", n_missing_seg)
    cand["segment_idx"] = cand["segment_idx"].fillna(-1).astype("int64")
    cand["timestamp_ms"] = cand["timestamp_ms"].fillna(0).astype("int64")

    # ── 4. Attach label + RR intervals from labels.parquet ───────────────────
    log.info("Loading labels …")
    label_cols = ["peak_id", "label"]
    labels_full = pd.read_parquet(args.labels, columns=None)  # load all to check columns
    for col in ("rr_prev_ms", "rr_next_ms"):
        if col in labels_full.columns:
            label_cols.append(col)
    labels_sub = labels_full[label_cols].drop_duplicates("peak_id")
    cand = cand.merge(labels_sub, on="peak_id", how="left")
    cand["label"] = cand["label"].fillna("unlabeled").astype(str)

    # ── 5. Stream ECG context from ecg_samples.parquet per segment ───────────
    # Group candidates by segment_idx so we only read each segment once.
    ecg_path = args.ecg_samples
    seg_groups = cand.groupby("segment_idx", sort=True)
    n_segs = cand["segment_idx"].nunique()
    log.info("Fetching ECG context from %d unique segments …", n_segs)

    # Pre-allocate result containers
    context_ecg_store:   dict[int, list] = {}
    context_ts_store:    dict[int, list] = {}
    r_peak_idx_store:    dict[int, int]  = {}
    is_edge_store:       dict[int, bool] = {}  # True → peak too close to window edge

    for seg_num, (seg_idx, group) in enumerate(seg_groups, 1):
        if seg_idx < 0:
            for pid in group["peak_id"]:
                context_ecg_store[pid]  = []
                context_ts_store[pid]   = []
                r_peak_idx_store[pid]   = 0
                is_edge_store[pid]      = True
            continue

        if seg_num % 50 == 0 or seg_num == 1:
            log.info("  segment %d/%d  (seg_idx=%d)", seg_num, n_segs, seg_idx)

        # Predicate pushdown: read only this segment's rows
        try:
            tbl = pq.read_table(
                ecg_path,
                filters=[("segment_idx", "==", int(seg_idx))],
                columns=["timestamp_ms", "ecg", "segment_idx"],
            )
            ecg_chunk = tbl.to_pandas().sort_values("timestamp_ms").reset_index(drop=True)
        except Exception as exc:
            log.warning("  Failed to read segment %d: %s", seg_idx, exc)
            for pid in group["peak_id"]:
                context_ecg_store[pid]  = []
                context_ts_store[pid]   = []
                r_peak_idx_store[pid]   = 0
                is_edge_store[pid]      = True
            continue

        ecg_ts   = ecg_chunk["timestamp_ms"].values
        ecg_vals = ecg_chunk["ecg"].values

        for _, row in group.iterrows():
            pid   = int(row["peak_id"])
            ts_ms = int(row["timestamp_ms"])

            t_start = ts_ms - HALF_WINDOW_MS
            t_end   = ts_ms + HALF_WINDOW_MS
            mask    = (ecg_ts >= t_start) & (ecg_ts <= t_end)

            if mask.any():
                ctx_ecg = ecg_vals[mask].astype(float).tolist()
                ctx_ts  = ecg_ts[mask].astype(int).tolist()
                # Find timestamp-nearest sample, then snap to local amplitude
                # maximum within ±5 samples (~38 ms at 130 Hz). Corrects the
                # Pan-Tompkins upswing bias for GUI display only — peak_id
                # timestamps and all model inputs are unchanged.
                closest = int(np.argmin(np.abs(ecg_ts[mask] - ts_ms)))
                snap_lo = max(0, closest - 5)
                snap_hi = min(len(ctx_ecg), closest + 6)
                ctx_arr = np.array(ctx_ecg)
                r_idx   = snap_lo + int(np.argmax(ctx_arr[snap_lo:snap_hi]))
            else:
                ctx_ecg = []
                ctx_ts  = []
                r_idx   = 0

            n_ctx = len(ctx_ecg)
            margin = args.edge_margin
            on_edge = (n_ctx == 0
                       or r_idx < margin
                       or r_idx > n_ctx - 1 - margin)

            context_ecg_store[pid]  = ctx_ecg
            context_ts_store[pid]   = ctx_ts
            r_peak_idx_store[pid]   = r_idx
            is_edge_store[pid]      = on_edge

    # ── 6. Write JSON files + summary CSV ────────────────────────────────────
    n_edge   = sum(1 for v in is_edge_store.values() if v)
    n_usable = sum(1 for v in is_edge_store.values() if not v)
    log.info("Edge-beat filter: %d edge, %d usable (target %d)",
             n_edge, n_usable, args.n_beats)
    if n_usable < args.n_beats:
        log.warning("Only %d usable beats after edge filtering — "
                    "consider increasing --oversample (currently %.1f)",
                    n_usable, args.oversample)

    log.info("Writing JSON files …")
    summary_rows = []
    n_written = 0

    for _, row in cand.iterrows():
        if n_written >= args.n_beats:
            break
        pid      = int(row["peak_id"])
        ts_ms    = int(row["timestamp_ms"])
        seg_idx  = int(row["segment_idx"])
        cur_lbl  = str(row["label"])
        p_ens    = float(row["p_artifact_ensemble"])
        disagree = float(row["disagreement"])
        composite = float(row["composite_score"])
        rr_prev  = row.get("rr_prev_ms", float("nan"))
        rr_next  = row.get("rr_next_ms", float("nan"))

        if is_edge_store.get(pid, True):
            continue  # peak too close to window edge — skip silently

        ctx_ecg  = context_ecg_store.get(pid, [])
        ctx_ts   = context_ts_store.get(pid, [])
        r_idx    = r_peak_idx_store.get(pid, 0)

        rr_prev_out = None if (isinstance(rr_prev, float) and np.isnan(rr_prev)) else round(float(rr_prev), 2)
        rr_next_out = None if (isinstance(rr_next, float) and np.isnan(rr_next)) else round(float(rr_next), 2)

        record = {
            "peak_id":                  pid,
            "timestamp_ms":             ts_ms,
            "segment_idx":              seg_idx,
            "current_label":            cur_lbl,
            "p_artifact_ensemble":      round(p_ens, 6),
            "disagreement":             round(disagree, 6),
            "composite_score":          round(composite, 6),
            "rr_prev_ms":               rr_prev_out,
            "rr_next_ms":               rr_next_out,
            "context_ecg":              [round(v, 6) for v in ctx_ecg],
            "context_timestamps_ms":    ctx_ts,
            "r_peak_index_in_context":  r_idx,
        }

        json_path = out_dir / f"beat_{pid:08d}.json"
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

        summary_rows.append({
            "peak_id":                  pid,
            "timestamp_ms":             ts_ms,
            "segment_idx":              seg_idx,
            "current_label":            cur_lbl,
            "p_artifact_ensemble":      round(p_ens, 6),
            "p_artifact_tabular":       round(float(row["p_artifact_tabular"]), 6),
            "p_artifact_cnn":           round(float(row["p_artifact_cnn"]), 6),
            "disagreement":             round(disagree, 6),
            "composite_score":          round(composite, 6),
            "rr_prev_ms":               rr_prev_out,
            "rr_next_ms":               rr_next_out,
            "r_peak_index_in_context":  r_idx,
            "n_context_samples":        len(ctx_ecg),
        })
        n_written += 1

    summary_df = pd.DataFrame(summary_rows)
    csv_path = out_dir / "queue_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    n_with_ecg = sum(1 for pid in context_ecg_store if context_ecg_store[pid])
    print(f"\n{'=' * 72}")
    print("  Disagreement Queue Export")
    print(f"{'=' * 72}")
    print(f"  Beats exported:        {len(summary_rows)}")
    print(f"  Beats with ECG data:   {n_with_ecg}")
    print(f"  Beats missing ECG:     {len(summary_rows) - n_with_ecg}")
    print(f"  Sort column:           {sort_col}")
    print(f"  {sort_col} range:  {summary_df[sort_col].min():.4f} – {summary_df[sort_col].max():.4f}")
    print(f"  Output directory:      {out_dir}")
    print(f"\n  To review:")
    print(f"    python ecgclean/active_learning/review_queue.py \\")
    print(f"        --queue-dir  {out_dir} \\")
    print(f"        --output     {out_dir}/completed.csv \\")
    print(f"        --sort-by    {sort_col}")
    print(f"{'=' * 72}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="export_disagreement_queue.py",
        description="Export top-N disagreement beats to annotation queue.",
    )
    p.add_argument("--ensemble-preds", required=True,
                   help="Path to ensemble_preds.parquet")
    p.add_argument("--peaks", required=True,
                   help="Path to peaks.parquet")
    p.add_argument("--labels", required=True,
                   help="Path to labels.parquet")
    p.add_argument("--ecg-samples", required=True,
                   help="Path to ecg_samples.parquet (streamed via predicate pushdown)")
    p.add_argument("--output", required=True,
                   help="Output directory for queue_summary.csv + beat_*.json files")
    p.add_argument("--n-beats", type=int, default=500,
                   help="Number of beats to export (default: 500)")
    p.add_argument("--sort-by", default="disagreement",
                   choices=["disagreement", "p_artifact_cnn",
                            "p_artifact_ensemble", "uncertainty_ensemble"],
                   help="Column to rank beats by (default: disagreement)")
    p.add_argument("--max-ensemble", type=float, default=None, metavar="SCORE",
                   help="Only include beats where p_artifact_ensemble < SCORE. "
                        "Use 0.5 with --sort-by p_artifact_cnn to get CNN false "
                        "positive candidates (CNN confident, ensemble says clean).")
    p.add_argument("--min-cnn", type=float, default=None, metavar="SCORE",
                   help="Only include beats where p_artifact_cnn >= SCORE "
                        "(e.g. 0.7 to focus on CNN's high-confidence predictions).")
    p.add_argument("--exclude-annotated", nargs="+", default=None, metavar="CSV",
                   help="One or more completed.csv paths whose peak_ids will be "
                        "excluded. skipped.csv in the same directory is loaded "
                        "automatically.")
    p.add_argument("--edge-margin", type=int, default=30,
                   help="Exclude beats whose r_peak_index is within this many "
                        "samples of either end of the context window (default: 30 "
                        "≈ 230 ms at 130 Hz).")
    p.add_argument("--oversample", type=float, default=2.0,
                   help="Request this multiple of --n-beats candidates so edge "
                        "beats can be filtered without falling short (default: 2.0).")
    return p


if __name__ == "__main__":
    main()
