#!/usr/bin/env python3
"""
scripts/presort_reannotation_queue.py — Reannotation Queue Pre-Sorter

Pre-computes global template correlation for every beat in a JSON annotation
queue directory, sorts the beats by correlation score, and writes a CSV that
makes the reannotation session faster and more systematic.

Instead of reviewing beats in arbitrary file order, you work through:
  1. High-correlation beats first (likely clean_pristine — fast batch-confirms)
  2. Medium-correlation beats (review_needed — spend your attention here)
  3. Low-correlation beats (likely_artifact — fast batch-confirms)

The suggested_category column is a SUGGESTION only — the human confirms each
beat.  No labels are written or modified anywhere.

Queue JSON format (as used by ecgclean/active_learning/annotation_queue.py):
  {
    "peak_id":                 int,
    "timestamp_ns":            int,
    "segment_idx":             int,
    "current_label":           str,        # clean / artifact / missed_original / …
    "p_artifact_ensemble":     float,
    "disagreement":            float,
    "composite_score":         float,
    "rr_prev_ms":              float,
    "rr_next_ms":              float,
    "context_ecg":             [float, …], # variable-length ECG context (~650 samples)
    "context_timestamps_ns":   [int, …],
    "r_peak_index_in_context": int,        # index of R-peak within context_ecg
  }

Window extraction strategy:
  • Primary: extract 64 samples centred on r_peak_index_in_context from
    context_ecg (the JSON carries enough context).  Zero-pad at boundaries.
  • Fallback: for the rare beats with empty context_ecg (context_len == 0),
    load the 64-sample window from ecg_samples.parquet using timestamp_ns.

Pearson correlation is scale-invariant, so the mV units in the JSON context
(written before the µV conversion) produce identical correlation values to
computing against a µV template.

Usage
-----
    cd "/Volumes/xHRV/Artifact Detector"
    source /Users/tannereddy/.envs/hrv/bin/activate

    python scripts/presort_reannotation_queue.py \\
        --templates   data/templates/global_templates.joblib \\
        --queue-dir   data/annotation_queues/iteration_1/ \\
        --ecg-samples data/processed/ecg_samples.parquet \\
        --peaks       data/processed/peaks.parquet \\
        --output      data/reannotation_presorted.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.presort_reannotation_queue")

# ── Constants ─────────────────────────────────────────────────────────────────

_WINDOW_SIZE: int = 64
_PEAK_SNAP_SAMPLES: int = 8   # ±8 samples = ±62ms at 130 Hz — must match beat_features.py

# suggested_category thresholds
_THRESH_PRISTINE: float = 0.80   # corr > this → clean_pristine
_THRESH_ARTIFACT: float = 0.50   # corr < this → likely_artifact
# _THRESH_ARTIFACT ≤ corr ≤ _THRESH_PRISTINE → review_needed

# Score fields to carry forward from each JSON (in this order if present)
_SCORE_FIELDS: list[str] = [
    "p_artifact_ensemble",
    "p_artifact_tabular",
    "p_artifact_cnn",
    "disagreement",
    "composite_score",
    "uncertainty_score",
]

# Context fields to carry forward
_CONTEXT_FIELDS: list[str] = [
    "segment_idx",
    "rr_prev_ms",
    "rr_next_ms",
    "timestamp_ns",
]


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_window_from_context(
    context_ecg: list[float],
    r_peak_index: int,
    window_size: int = _WINDOW_SIZE,
    snap_samples: int = _PEAK_SNAP_SAMPLES,
) -> np.ndarray:
    """Extract a ``window_size``-sample window from a JSON context_ecg array.

    Centers the window on ``r_peak_index``, snaps to argmax(|ecg|) within
    ±snap_samples, and zero-pads at boundaries.

    Args:
        context_ecg: Variable-length ECG context as a Python list.
        r_peak_index: Index of the R-peak within context_ecg.
        window_size: Output window length (default 64).
        snap_samples: Argmax snap radius in samples (default 8).

    Returns:
        float32 array of shape (window_size,).  Zero-padded where the context
        does not extend far enough.
    """
    window = np.zeros(window_size, dtype=np.float32)

    n = len(context_ecg)
    if n == 0:
        return window

    context = np.asarray(context_ecg, dtype=np.float32)
    half    = window_size // 2

    # Clamp initial center to valid range
    center = max(0, min(n - 1, r_peak_index))

    # Snap to argmax(|ecg|) within ±snap_samples — same logic as beat_features.py
    snap_lo = max(0, center - snap_samples)
    snap_hi = min(n, center + snap_samples + 1)
    center  = snap_lo + int(np.argmax(np.abs(context[snap_lo:snap_hi])))

    src_start = max(0, center - half)
    src_end   = min(n, center + half)
    dst_start = src_start - (center - half)
    dst_end   = dst_start + (src_end - src_start)

    if src_end > src_start:
        window[dst_start:dst_end] = context[src_start:src_end]

    return window


def _extract_window_from_parquet(
    timestamp_ns: int,
    segment_idx: int,
    ecg_samples_path: Path,
    window_size: int = _WINDOW_SIZE,
    snap_samples: int = _PEAK_SNAP_SAMPLES,
) -> np.ndarray:
    """Fallback: extract a window from ecg_samples.parquet for one beat.

    Used only when context_ecg is empty in the JSON.  Loads a narrow ECG
    slice for the beat's segment (predicate pushdown), finds the nearest
    sample to timestamp_ns via binary search, then applies snap + extraction.

    Args:
        timestamp_ns: R-peak timestamp in nanoseconds.
        segment_idx: Segment index for predicate pushdown.
        ecg_samples_path: Path to ecg_samples.parquet.
        window_size: Output window length.
        snap_samples: Argmax snap radius.

    Returns:
        float32 array of shape (window_size,).
    """
    window = np.zeros(window_size, dtype=np.float32)

    try:
        ecg_df = (
            pq.read_table(
                ecg_samples_path,
                filters=[
                    ("segment_idx", ">=", max(0, segment_idx - 1)),
                    ("segment_idx", "<=", segment_idx + 1),
                ],
                columns=["timestamp_ns", "ecg"],
            )
            .to_pandas()
            .sort_values("timestamp_ns")
            .reset_index(drop=True)
        )
    except Exception as exc:
        logger.warning("ECG parquet fallback read failed: %s", exc)
        return window

    if len(ecg_df) == 0:
        return window

    ecg_ts   = ecg_df["timestamp_ns"].values.astype(np.int64)
    ecg_vals = ecg_df["ecg"].values.astype(np.float32)
    n_ecg    = len(ecg_ts)
    half     = window_size // 2

    insert_idx = int(np.searchsorted(ecg_ts, timestamp_ns, side="left"))
    center = insert_idx
    if 0 < center < n_ecg:
        if abs(ecg_ts[center - 1] - timestamp_ns) < abs(ecg_ts[center] - timestamp_ns):
            center -= 1
    elif center >= n_ecg:
        center = n_ecg - 1

    snap_lo = max(0, center - snap_samples)
    snap_hi = min(n_ecg, center + snap_samples + 1)
    center  = snap_lo + int(np.argmax(np.abs(ecg_vals[snap_lo:snap_hi])))

    src_start = max(0, center - half)
    src_end   = min(n_ecg, center + half)
    dst_start = src_start - (center - half)
    dst_end   = dst_start + (src_end - src_start)

    if src_end > src_start:
        window[dst_start:dst_end] = ecg_vals[src_start:src_end]

    return window


# ═══════════════════════════════════════════════════════════════════════════════
# PEARSON CORRELATION (vectorised, scale-invariant)
# ═══════════════════════════════════════════════════════════════════════════════

def _pearson_batch(windows: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Vectorised Pearson correlation of each row in *windows* vs *template*.

    Scale-invariant: mV and µV windows produce the same result.
    Zero-variance rows return 0.0 (no NaN).

    Args:
        windows: (n, L) float array of beat windows.
        template: (L,) template array.

    Returns:
        (n,) float32 array of Pearson r in [-1, 1].
    """
    w = windows.astype(np.float64)
    t = template.astype(np.float64)

    w_c = w - w.mean(axis=1, keepdims=True)
    t_c = t - t.mean()

    num   = (w_c * t_c).sum(axis=1)
    w_std = np.sqrt((w_c ** 2).sum(axis=1))
    t_std = float(np.sqrt((t_c ** 2).sum()))

    denom = w_std * t_std
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom > 0.0, num / denom, 0.0)

    result = np.where(np.isfinite(result), result, 0.0)
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def presort_queue(
    templates_path: Path,
    queue_dir: Path,
    ecg_samples_path: Path,
    peaks_path: Path,
    output_path: Path,
    pristine_threshold: float = _THRESH_PRISTINE,
    artifact_threshold: float = _THRESH_ARTIFACT,
) -> None:
    """Load queue JSONs, compute template correlations, and write sorted CSV.

    Args:
        templates_path: Path to global_templates.joblib.
        queue_dir: Directory containing beat_*.json files.
        ecg_samples_path: Path to ecg_samples.parquet (for empty-context fallback).
        peaks_path: Path to peaks.parquet (for metadata enrichment).
        output_path: Destination CSV file.
        pristine_threshold: Correlation above which a beat is suggested
            'clean_pristine' (default 0.80).
        artifact_threshold: Correlation below which a beat is suggested
            'likely_artifact' (default 0.50).
    """

    # ── Load template ─────────────────────────────────────────────────────────
    logger.info("Loading templates from %s", templates_path)
    tmpl = joblib.load(templates_path)
    template_clean: np.ndarray = tmpl["template_clean"].astype(np.float64)
    logger.info(
        "template_clean: built from %s clean beats at %s",
        tmpl.get("n_clean_beats", "?"),
        tmpl.get("built_at", "?"),
    )

    # ── Discover JSON files ───────────────────────────────────────────────────
    json_files = sorted(queue_dir.glob("beat_*.json"))
    n_files = len(json_files)
    if n_files == 0:
        logger.error("No beat_*.json files found in %s", queue_dir)
        sys.exit(1)
    logger.info("Found %d JSON files in %s", n_files, queue_dir)

    # ── Parse all JSONs ───────────────────────────────────────────────────────
    records: list[dict] = []
    windows_list: list[np.ndarray] = []
    fallback_needed: list[int] = []           # record indices needing parquet fallback

    for i, json_path in enumerate(json_files):
        try:
            beat = json.loads(json_path.read_text())
        except Exception as exc:
            logger.warning("Failed to parse %s: %s — skipping", json_path.name, exc)
            continue

        peak_id   = int(beat["peak_id"])
        ts_ns     = int(beat.get("timestamp_ns", 0))
        seg_idx   = int(beat.get("segment_idx", 0))
        ctx_ecg   = beat.get("context_ecg", [])
        r_idx     = int(beat.get("r_peak_index_in_context", 0))

        # Build flat record from scalar fields
        rec: dict = {
            "peak_id":        peak_id,
            "original_label": str(beat.get("current_label", "")),
        }
        for fld in _SCORE_FIELDS:
            if fld in beat:
                rec[fld] = float(beat[fld])
        for fld in _CONTEXT_FIELDS:
            if fld in beat:
                rec[fld] = beat[fld]

        records.append(rec)

        if len(ctx_ecg) == 0:
            # Needs ECG parquet fallback — store a placeholder
            windows_list.append(np.zeros(_WINDOW_SIZE, dtype=np.float32))
            fallback_needed.append(len(records) - 1)
            logger.debug("Empty context_ecg for peak_id=%d — will use parquet fallback", peak_id)
        else:
            windows_list.append(
                _extract_window_from_context(ctx_ecg, r_idx)
            )

    n_parsed = len(records)
    logger.info("Parsed %d / %d JSON files successfully", n_parsed, n_files)

    if n_parsed == 0:
        logger.error("No records parsed — nothing to output.")
        sys.exit(1)

    # ── Parquet fallback for empty-context beats ──────────────────────────────
    if fallback_needed:
        logger.info(
            "Loading ecg_samples.parquet for %d empty-context beat(s) …",
            len(fallback_needed),
        )
        for rec_idx in fallback_needed:
            rec = records[rec_idx]
            ts_ns   = int(rec.get("timestamp_ns", 0))
            seg_idx = int(rec.get("segment_idx", 0))
            windows_list[rec_idx] = _extract_window_from_parquet(
                ts_ns, seg_idx, ecg_samples_path
            )
            logger.debug(
                "  Fallback window for peak_id=%d via parquet", rec["peak_id"]
            )

    # ── Batch correlation ─────────────────────────────────────────────────────
    windows = np.stack(windows_list, axis=0)   # (n, 64)
    logger.info("Computing template correlations for %d beats …", n_parsed)
    corr = _pearson_batch(windows, template_clean)

    # ── Assign suggested_category ─────────────────────────────────────────────
    categories = np.where(
        corr > pristine_threshold,
        "clean_pristine",
        np.where(
            corr < artifact_threshold,
            "likely_artifact",
            "review_needed",
        ),
    )

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df["global_corr_clean"]  = corr.astype(np.float32)
    df["suggested_category"] = categories

    # ── Sort descending by correlation (highest = most likely clean, easiest first)
    df.sort_values("global_corr_clean", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Column ordering: key columns first, then scores, then context ─────────
    leading = ["peak_id", "original_label", "global_corr_clean", "suggested_category"]
    score_cols   = [c for c in _SCORE_FIELDS if c in df.columns]
    context_cols = [c for c in _CONTEXT_FIELDS if c in df.columns]
    other_cols   = [c for c in df.columns
                    if c not in leading + score_cols + context_cols]
    df = df[leading + score_cols + context_cols + other_cols]

    # ── Summary ───────────────────────────────────────────────────────────────
    cat_counts = df["suggested_category"].value_counts()
    label_counts = df["original_label"].value_counts()

    print()
    print("═" * 70)
    print("  REANNOTATION QUEUE PRE-SORT SUMMARY")
    print("═" * 70)
    print(f"  Queue directory : {queue_dir}")
    print(f"  Beats processed : {n_parsed:,}")
    print(f"  Output          : {output_path}")
    print()
    print("  ── Suggested categories (sort by global_corr_clean desc) ──────")
    for cat in ["clean_pristine", "review_needed", "likely_artifact"]:
        n = int(cat_counts.get(cat, 0))
        pct = 100 * n / n_parsed if n_parsed > 0 else 0
        thresh_note = {
            "clean_pristine":  f"corr > {pristine_threshold}",
            "review_needed":   f"{artifact_threshold} ≤ corr ≤ {pristine_threshold}",
            "likely_artifact": f"corr < {artifact_threshold}",
        }[cat]
        print(f"    {cat:<20}  {n:>5,}  ({pct:5.1f}%)   [{thresh_note}]")
    print()
    print("  ── Original label distribution ─────────────────────────────────")
    for lbl, cnt in label_counts.items():
        pct = 100 * cnt / n_parsed if n_parsed > 0 else 0
        print(f"    {str(lbl):<22}  {cnt:>5,}  ({pct:5.1f}%)")
    print()
    print("  ── global_corr_clean statistics ────────────────────────────────")
    print(f"    mean ± std : {corr.mean():.4f} ± {corr.std():.4f}")
    pcts = np.percentile(corr, [5, 25, 50, 75, 95])
    print(f"    p5 / p25   : {pcts[0]:.4f} / {pcts[1]:.4f}")
    print(f"    median     : {pcts[2]:.4f}")
    print(f"    p75 / p95  : {pcts[3]:.4f} / {pcts[4]:.4f}")
    print(f"    min / max  : {corr.min():.4f} / {corr.max():.4f}")
    if len(fallback_needed) > 0:
        print(f"\n  ⚠  {len(fallback_needed)} beat(s) used ecg_samples.parquet fallback "
              f"(empty context_ecg in JSON)")
    print()
    print("  ── Review strategy ─────────────────────────────────────────────")
    print("    1. Start at top of CSV (clean_pristine) — batch-confirm or flip quickly")
    print("    2. Middle section (review_needed) — spend annotation effort here")
    print("    3. Bottom (likely_artifact) — batch-confirm or flip quickly")
    print("    suggested_category is a hint only — you confirm every beat.")
    print("═" * 70)
    print()

    # ── Write output ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    logger.info("Saved → %s  (%d rows × %d columns)", output_path, len(df), len(df.columns))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-sort a reannotation queue by global template correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--templates",
        type=str,
        required=True,
        help="Path to global_templates.joblib",
    )
    parser.add_argument(
        "--queue-dir",
        type=str,
        required=True,
        help="Directory containing beat_*.json files",
    )
    parser.add_argument(
        "--ecg-samples",
        type=str,
        required=True,
        help="Path to ecg_samples.parquet (fallback for empty-context beats)",
    )
    parser.add_argument(
        "--peaks",
        type=str,
        required=True,
        help="Path to peaks.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/reannotation_presorted.csv",
        help="Output CSV path (default: data/reannotation_presorted.csv)",
    )
    parser.add_argument(
        "--pristine-threshold",
        type=float,
        default=_THRESH_PRISTINE,
        help=f"Correlation above which beats are suggested 'clean_pristine' "
             f"(default: {_THRESH_PRISTINE})",
    )
    parser.add_argument(
        "--artifact-threshold",
        type=float,
        default=_THRESH_ARTIFACT,
        help=f"Correlation below which beats are suggested 'likely_artifact' "
             f"(default: {_THRESH_ARTIFACT})",
    )

    args = parser.parse_args()

    # Validate required paths
    for flag, path in [
        ("--templates",   args.templates),
        ("--queue-dir",   args.queue_dir),
        ("--ecg-samples", args.ecg_samples),
        ("--peaks",       args.peaks),
    ]:
        if not Path(path).exists():
            logger.error("Path not found for %s: %s", flag, path)
            sys.exit(1)

    presort_queue(
        templates_path=Path(args.templates),
        queue_dir=Path(args.queue_dir),
        ecg_samples_path=Path(args.ecg_samples),
        peaks_path=Path(args.peaks),
        output_path=Path(args.output),
        pristine_threshold=args.pristine_threshold,
        artifact_threshold=args.artifact_threshold,
    )


if __name__ == "__main__":
    main()
