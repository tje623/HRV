#!/usr/bin/env python3
"""
ecgclean.active_learning.annotation_queue
==========================================
Serialize annotation candidates into a format consumable by the
annotation UI, and import completed annotations back.

Export produces:
    - One JSON file per candidate beat (with ECG context window)
    - A summary CSV with one row per candidate (no ECG waveform data)

Import reads back a completed CSV and validates against expected IDs.

CLI
---
    python ecgclean/active_learning/annotation_queue.py export \\
        --candidates ... --peaks ... --labels ... --ecg-samples ... --output ...

    python ecgclean/active_learning/annotation_queue.py import \\
        --completed ... --expected-ids ... --labels ... --al-iteration 1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `ecgclean` is importable when
# running this file directly (python ecgclean/active_learning/annotation_queue.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ecgclean.active_learning.sampler import record_labels

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_LABELS = {"clean", "artifact", "interpolated", "phys_event", "missed_original"}
SAMPLE_RATE_HZ = 125  # Polar H10 ECG sampling rate (empirically 8.000 ms/sample = 125 Hz)


# ===================================================================== #
#  Export                                                                #
# ===================================================================== #
def export_queue(
    candidates_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ecg_samples_df: pd.DataFrame,
    output_path: str,
    context_window_sec: float = 5.0,
) -> None:
    """Export annotation queue as JSON files + summary CSV.

    For each candidate beat, produces a JSON record with the ECG context
    window (raw samples ± context_window_sec/2 around the R-peak).

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Output of ``sampler.select_annotation_candidates()``.  Must have
        ``peak_id`` and at least ``p_artifact_ensemble``, ``disagreement``,
        ``composite_score``.
    peaks_df : pd.DataFrame
        Must have ``peak_id``, ``timestamp_ns``, ``segment_idx``.
    labels_df : pd.DataFrame
        Must have ``peak_id``, ``label``.  Used for ``current_label`` and
        ``rr_prev_ms`` / ``rr_next_ms``.
    ecg_samples_df : pd.DataFrame
        Must have ``timestamp_ns``, ``ecg``, ``segment_idx``.
    output_path : str
        Directory where output files are written.  Created if needed.
    context_window_sec : float
        Total context window in seconds (split equally before/after R-peak).
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Merge candidate info with peaks and labels ──────────────────
    cand = candidates_df.copy()

    # Ensure peak_id → timestamp_ns and segment_idx
    peak_cols = ["peak_id", "timestamp_ns", "segment_idx"]
    if "timestamp_ns" not in cand.columns or "segment_idx" not in cand.columns:
        cand = cand.merge(
            peaks_df[peak_cols].drop_duplicates("peak_id"),
            on="peak_id",
            how="left",
            suffixes=("", "_pk"),
        )
        # Resolve conflicts from merge
        for col in ["timestamp_ns", "segment_idx"]:
            alt = f"{col}_pk"
            if alt in cand.columns:
                cand[col] = cand[col].fillna(cand[alt])
                cand.drop(columns=[alt], inplace=True)

    # Merge in label info
    label_cols = ["peak_id", "label"]
    for col in ["rr_prev_ms", "rr_next_ms"]:
        if col in labels_df.columns:
            label_cols.append(col)
    label_merge = labels_df[label_cols].drop_duplicates("peak_id")
    cand = cand.merge(label_merge, on="peak_id", how="left", suffixes=("", "_lbl"))

    # ── Precompute ECG index for fast lookup ─────────────────────────
    half_window_ns = int(context_window_sec / 2.0 * 1e9)

    # Sort ECG samples by timestamp for efficient slicing
    ecg = ecg_samples_df.sort_values("timestamp_ns").reset_index(drop=True)
    ecg_ts = ecg["timestamp_ns"].values
    ecg_vals = ecg["ecg"].values
    ecg_seg = ecg["segment_idx"].values

    # ── Export each candidate ────────────────────────────────────────
    summary_rows = []
    n_exported = 0

    for _, row in cand.iterrows():
        pid = int(row["peak_id"])
        ts_ns = int(row.get("timestamp_ns", 0))
        seg_idx = int(row.get("segment_idx", -1))
        current_label = str(row.get("label", "unlabeled"))
        p_ens = float(row.get("p_artifact_ensemble", 0.0))
        disagree = float(row.get("disagreement", 0.0))
        composite = float(row.get("composite_score", 0.0))
        rr_prev = float(row.get("rr_prev_ms", np.nan))
        rr_next = float(row.get("rr_next_ms", np.nan))

        # ── Extract ECG context window ───────────────────────────────
        t_start = ts_ns - half_window_ns
        t_end = ts_ns + half_window_ns

        # Find ECG samples in same segment within the time window
        seg_mask = ecg_seg == seg_idx
        time_mask = (ecg_ts >= t_start) & (ecg_ts <= t_end)
        window_mask = seg_mask & time_mask

        context_ecg: list[float] = []
        context_ts: list[int] = []
        r_peak_idx = 0

        if window_mask.any():
            idx = np.where(window_mask)[0]
            context_ecg = ecg_vals[idx].astype(float).tolist()
            context_ts = ecg_ts[idx].astype(int).tolist()

            # Find R-peak index within context
            if len(context_ts) > 0:
                r_peak_idx = int(np.argmin(np.abs(np.array(context_ts) - ts_ns)))
        else:
            # No ECG data available for this beat
            context_ecg = []
            context_ts = []
            r_peak_idx = 0

        # ── Build JSON record ────────────────────────────────────────
        record = {
            "peak_id": pid,
            "timestamp_ns": ts_ns,
            "segment_idx": seg_idx,
            "current_label": current_label,
            "p_artifact_ensemble": round(p_ens, 6),
            "disagreement": round(disagree, 6),
            "composite_score": round(composite, 6),
            "rr_prev_ms": None if np.isnan(rr_prev) else round(rr_prev, 2),
            "rr_next_ms": None if np.isnan(rr_next) else round(rr_next, 2),
            "context_ecg": [round(v, 6) for v in context_ecg],
            "context_timestamps_ns": context_ts,
            "r_peak_index_in_context": r_peak_idx,
        }

        # Write JSON
        json_path = out_dir / f"beat_{pid:08d}.json"
        with open(json_path, "w") as f:
            json.dump(record, f, indent=2)

        # Summary row (no ECG waveform)
        summary_rows.append({
            "peak_id": pid,
            "timestamp_ns": ts_ns,
            "segment_idx": seg_idx,
            "current_label": current_label,
            "p_artifact_ensemble": round(p_ens, 6),
            "disagreement": round(disagree, 6),
            "composite_score": round(composite, 6),
            "rr_prev_ms": rr_prev if not np.isnan(rr_prev) else None,
            "rr_next_ms": rr_next if not np.isnan(rr_next) else None,
            "r_peak_index_in_context": r_peak_idx,
            "n_context_samples": len(context_ecg),
        })
        n_exported += 1

    # ── Write summary CSV ────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    csv_path = out_dir / "queue_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    log.info("Exported %d beats: %d JSONs + summary CSV → %s", n_exported, n_exported, out_dir)


# ===================================================================== #
#  Import                                                               #
# ===================================================================== #
def import_completed_annotations(
    completed_csv_path: str,
    expected_peak_ids: list[int],
) -> pd.DataFrame:
    """Read and validate completed annotations from CSV.

    Parameters
    ----------
    completed_csv_path : str
        CSV with columns ``peak_id`` (int) and ``label`` (str).
    expected_peak_ids : list[int]
        Peak IDs that were in the original annotation queue.  Any
        ``peak_id`` in the CSV that is *not* in this list triggers an error.

    Returns
    -------
    pd.DataFrame
        Columns: ``peak_id``, ``new_label``.  Ready for
        ``sampler.record_labels()``.

    Raises
    ------
    ValueError
        If any ``peak_id`` is unexpected or any ``label`` is invalid.
    FileNotFoundError
        If ``completed_csv_path`` does not exist.
    """
    path = Path(completed_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Completed annotations file not found: {path}")

    df = pd.read_csv(path)

    # ── Validate columns ─────────────────────────────────────────────
    required = {"peak_id", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required - set(df.columns)}.  "
                         f"Found: {list(df.columns)}")

    # ── Validate peak_ids ────────────────────────────────────────────
    expected_set = set(expected_peak_ids)
    unexpected = set(df["peak_id"]) - expected_set
    if unexpected:
        raise ValueError(
            f"Found {len(unexpected)} unexpected peak_id(s) not in the annotation queue: "
            f"{sorted(unexpected)[:20]}{'...' if len(unexpected) > 20 else ''}.  "
            f"All peak_ids must be from the original candidate list."
        )

    # ── Validate labels ──────────────────────────────────────────────
    invalid_labels = set(df["label"].unique()) - VALID_LABELS
    if invalid_labels:
        raise ValueError(
            f"Invalid label values: {invalid_labels}.  "
            f"Valid labels are: {VALID_LABELS}"
        )

    # ── Rename for sampler.record_labels ─────────────────────────────
    result = df[["peak_id", "label"]].copy()
    result = result.rename(columns={"label": "new_label"})

    log.info("Imported %d completed annotations from %s", len(result), path)
    return result


# ===================================================================== #
#  CLI                                                                  #
# ===================================================================== #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotation_queue.py",
        description="Export/import annotation queue for active learning.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── export ────────────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Export annotation queue")
    p_export.add_argument("--candidates", required=True, help="Path to AL candidates parquet")
    p_export.add_argument("--peaks", required=True, help="Path to peaks.parquet")
    p_export.add_argument("--labels", required=True, help="Path to labels.parquet")
    p_export.add_argument("--ecg-samples", required=True, help="Path to ecg_samples.parquet")
    p_export.add_argument("--output", required=True, help="Output directory")
    p_export.add_argument("--context-window", type=float, default=5.0, help="Context window (sec)")

    # ── import ────────────────────────────────────────────────────────
    p_import = sub.add_parser("import", help="Import completed annotations")
    p_import.add_argument("--completed", required=True, help="Path to completed.csv")
    p_import.add_argument("--expected-ids", required=True, help="Path to AL candidates parquet (for ID validation)")
    p_import.add_argument("--labels", required=True, help="Path to labels.parquet (will be updated)")
    p_import.add_argument("--al-iteration", type=int, required=True, help="AL iteration number")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "export":
        candidates = pd.read_parquet(args.candidates)
        peaks = pd.read_parquet(args.peaks)
        labels = pd.read_parquet(args.labels)
        ecg_samples = pd.read_parquet(args.ecg_samples)

        export_queue(
            candidates_df=candidates,
            peaks_df=peaks,
            labels_df=labels,
            ecg_samples_df=ecg_samples,
            output_path=args.output,
            context_window_sec=args.context_window,
        )

        # ── Summary ──────────────────────────────────────────────────
        print(f"\n{'=' * 72}")
        print("  Annotation Queue Export")
        print(f"{'=' * 72}")
        print(f"  Candidates exported: {len(candidates)}")
        print(f"  Output directory: {args.output}")
        out_dir = Path(args.output)
        n_json = len(list(out_dir.glob("beat_*.json")))
        csv_exists = (out_dir / "queue_summary.csv").exists()
        print(f"  JSON files: {n_json}")
        print(f"  Summary CSV: {'✓' if csv_exists else '✗'}")
        print(f"{'=' * 72}")

    elif args.command == "import":
        # Load expected peak_ids from the candidates parquet
        expected_df = pd.read_parquet(args.expected_ids)
        expected_ids = expected_df["peak_id"].tolist()

        # Import and validate
        annotations = import_completed_annotations(args.completed, expected_ids)

        # Record into labels.parquet
        record_labels(annotations, args.labels, args.al_iteration)

        print(f"\n{'=' * 72}")
        print("  Annotation Import Complete")
        print(f"{'=' * 72}")
        print(f"  Annotations imported: {len(annotations)}")
        print(f"  Labels updated: {args.labels}")
        print(f"  AL iteration: {args.al_iteration}")
        for lbl, cnt in annotations["new_label"].value_counts().items():
            print(f"    {lbl}: {cnt}")
        print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
