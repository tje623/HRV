#!/usr/bin/env python3
"""
ecgclean.active_learning.sampler
================================
Active-learning candidate selection for the ECG artifact detection pipeline.

Three complementary strategies identify the most informative unlabeled beats
to send for manual annotation:

    1. **margin**    – pure uncertainty sampling (p_ensemble near 0.5)
    2. **committee** – model disagreement (tabular vs. CNN differ)
    3. **priority**  – physiological suspicion (high review_priority_score)
    4. **combined**  – union of all three, ranked by composite score (default)

The ``combined`` composite score is:

    composite = 0.4 * uncertainty_ensemble
              + 0.4 * disagreement
              + 0.2 * (review_priority_score / max_priority)

Weight rationale (documented per spec requirement):
    - 0.4 uncertainty: Beats near the decision boundary are where the model
      will gain the most from a human label (classic uncertainty sampling).
    - 0.4 disagreement: When two architecturally different models disagree,
      the input likely exposes a blind spot in one of them.
    - 0.2 priority: Domain knowledge (POTS transitions, suspicious RR
      intervals) should nudge selection but not dominate, since the model
      may already handle many of these correctly.

CLI
---
    python ecgclean/active_learning/sampler.py select \\
        --ensemble-preds ... --labels ... --segments ... \\
        --segment-quality-preds ... --n-candidates 500 --output ...
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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
VALID_STRATEGIES = {"margin", "committee", "priority", "combined"}
VALID_LABELS = {"clean", "artifact", "interpolated", "phys_event", "missed_original"}


# ===================================================================== #
#  Core selection                                                       #
# ===================================================================== #
def select_annotation_candidates(
    ensemble_preds: pd.DataFrame,
    labels_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    segment_quality_preds: pd.DataFrame,
    n_candidates: int = 500,
    strategy: str = "combined",
    uncertainty_band: tuple[float, float] = (0.3, 0.7),
    disagreement_threshold: float = 0.3,
    al_iteration: int = 1,
) -> pd.DataFrame:
    """Select the most informative unlabeled beats for annotation.

    Parameters
    ----------
    ensemble_preds : pd.DataFrame
        Output of ``ensemble.fuse()`` — must contain ``peak_id``,
        ``p_artifact_ensemble``, ``p_artifact_tabular``, ``p_artifact_cnn``,
        ``disagreement``, ``uncertainty_ensemble``.
    labels_df : pd.DataFrame
        Existing labels.  Beats already labeled are *excluded* from
        the candidate pool.  Must contain ``peak_id``, ``label``,
        ``review_priority_score``, ``pots_transition_candidate``.
    segments_df : pd.DataFrame
        Segment metadata (``segment_idx``, etc.).  Used for context but
        segment quality filtering comes from *segment_quality_preds*.
    segment_quality_preds : pd.DataFrame
        Must contain ``segment_idx`` and ``quality_label``.  Beats in
        segments labeled ``bad`` are always excluded.
    n_candidates : int
        Maximum number of candidates to return.
    strategy : str
        One of ``"margin"``, ``"committee"``, ``"priority"``, ``"combined"``.
    uncertainty_band : tuple[float, float]
        ``(lo, hi)`` for the margin strategy.
    disagreement_threshold : float
        Minimum disagreement for the committee strategy.
    al_iteration : int
        Current active-learning iteration number (recorded in output).

    Returns
    -------
    pd.DataFrame
        Columns: peak_id, segment_idx, p_artifact_ensemble,
        p_artifact_tabular, p_artifact_cnn, disagreement,
        uncertainty_ensemble, review_priority_score, composite_score,
        selection_strategy, al_iteration.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")

    # ── Required columns check ───────────────────────────────────────
    ens_required = {
        "peak_id", "p_artifact_ensemble", "p_artifact_tabular",
        "p_artifact_cnn", "disagreement", "uncertainty_ensemble",
    }
    if not ens_required.issubset(ensemble_preds.columns):
        raise ValueError(f"ensemble_preds missing: {ens_required - set(ensemble_preds.columns)}")

    # ── Attach segment_idx via peaks.parquet ──────────────────────────
    # labels_df may not have segment_idx; try to get it from the
    # ensemble_preds parent dir or from peaks.parquet co-located with
    # labels.  We need segment_idx for filtering by segment quality.
    #
    # Strategy: join ensemble_preds with segment_quality_preds to find
    # bad segments.  We need peak_id → segment_idx.  Look for it in
    # labels_df first, then try to load peaks.parquet.
    if "segment_idx" not in labels_df.columns:
        # Try loading peaks.parquet from the same dir as labels
        # (This is the standard pipeline layout)
        log.info("labels_df lacks segment_idx — attempting to infer from peaks.parquet")
        peaks_candidates = [
            Path(labels_df.attrs.get("_source_path", "")).parent / "peaks.parquet",
        ]
        # Also check common locations
        for search_dir in [Path("data/processed"), Path(".")]:
            peaks_candidates.append(search_dir / "peaks.parquet")

        peaks_loaded = False
        for pp in peaks_candidates:
            if pp.exists():
                peaks_df = pd.read_parquet(pp, columns=["peak_id", "segment_idx"])
                labels_df = labels_df.merge(peaks_df, on="peak_id", how="left")
                log.info("Loaded segment_idx from %s", pp)
                peaks_loaded = True
                break

        if not peaks_loaded:
            log.warning("Could not find peaks.parquet — segment quality filtering disabled")
            labels_df["segment_idx"] = -1

    # ── Build working table ──────────────────────────────────────────
    # Start from ensemble preds and enrich with labels info
    work = ensemble_preds[list(ens_required)].copy()

    # Merge in label info (review_priority_score, pots_transition_candidate, label, segment_idx,
    # and reviewed if available)
    label_cols = ["peak_id"]
    for col in ["label", "reviewed", "review_priority_score", "pots_transition_candidate", "segment_idx"]:
        if col in labels_df.columns:
            label_cols.append(col)

    work = work.merge(labels_df[label_cols].drop_duplicates("peak_id"), on="peak_id", how="left")

    # Fill missing review_priority_score with 0
    if "review_priority_score" not in work.columns:
        work["review_priority_score"] = 0.0
    work["review_priority_score"] = work["review_priority_score"].fillna(0.0)

    # Fill missing pots_transition_candidate with False
    if "pots_transition_candidate" not in work.columns:
        work["pots_transition_candidate"] = False
    work["pots_transition_candidate"] = work["pots_transition_candidate"].fillna(False)

    log.info("Working table: %d beats total", len(work))

    # ── Exclude already-reviewed beats from the candidate pool ───────────
    # "Already-reviewed" means a human has actually inspected this beat.
    # Use the 'reviewed' column when present (reviewed=True → exclude).
    # Beats with reviewed=False carry a default label but were never
    # inspected — they ARE the active-learning candidate pool.
    # Fall back to label.notna() only when 'reviewed' is absent (old data).
    if "reviewed" in work.columns:
        already_labeled = work["reviewed"].fillna(False).astype(bool)
        n_labeled = int(already_labeled.sum())
        work = work[~already_labeled].copy()
        log.info(
            "Excluded %d reviewed beats, %d unreviewed remain as candidates",
            n_labeled,
            len(work),
        )
    elif "label" in work.columns:
        already_labeled = work["label"].notna()
        n_labeled = int(already_labeled.sum())
        work = work[~already_labeled].copy()
        log.info(
            "Excluded %d already-labeled beats (no 'reviewed' column), "
            "%d unlabeled remain",
            n_labeled,
            len(work),
        )
    else:
        log.info("No label column — treating all beats as unlabeled")

    # ── Exclude beats from 'bad' segments ────────────────────────────
    if "segment_idx" in work.columns and len(segment_quality_preds) > 0:
        bad_segments = set(
            segment_quality_preds.loc[
                segment_quality_preds["quality_label"] == "bad", "segment_idx"
            ]
        )
        if bad_segments:
            before = len(work)
            work = work[~work["segment_idx"].isin(bad_segments)].copy()
            log.info("Excluded %d beats from %d bad segments", before - len(work), len(bad_segments))
    else:
        log.info("No segment quality filtering applied")

    if len(work) == 0:
        log.warning("No unlabeled, non-bad beats remain — returning empty DataFrame")
        return _empty_candidates()

    # ── Strategy-specific selection ──────────────────────────────────
    if strategy == "margin":
        pool = work[
            (work["p_artifact_ensemble"] >= uncertainty_band[0])
            & (work["p_artifact_ensemble"] <= uncertainty_band[1])
        ].copy()
        pool["selection_strategy"] = "margin"

    elif strategy == "committee":
        pool = work[work["disagreement"] >= disagreement_threshold].copy()
        pool["selection_strategy"] = "committee"

    elif strategy == "priority":
        pool = work.nlargest(n_candidates, "review_priority_score").copy()
        pool["selection_strategy"] = "priority"

    elif strategy == "combined":
        # Merge all three pools
        margin_pool = work[
            (work["p_artifact_ensemble"] >= uncertainty_band[0])
            & (work["p_artifact_ensemble"] <= uncertainty_band[1])
        ].copy()
        margin_pool["selection_strategy"] = "margin"

        committee_pool = work[work["disagreement"] >= disagreement_threshold].copy()
        committee_pool["selection_strategy"] = "committee"

        priority_pool = work.nlargest(n_candidates, "review_priority_score").copy()
        priority_pool["selection_strategy"] = "priority"

        # Combine and deduplicate (keep first occurrence → preserves strategy tag)
        pool = pd.concat([margin_pool, committee_pool, priority_pool], ignore_index=True)
        pool = pool.drop_duplicates(subset="peak_id", keep="first")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    log.info("Strategy '%s' selected %d candidates before scoring", strategy, len(pool))

    if len(pool) == 0:
        log.warning("No candidates matched strategy '%s' — returning empty", strategy)
        return _empty_candidates()

    # ── Compute composite score ──────────────────────────────────────
    max_priority = pool["review_priority_score"].max()
    if max_priority == 0 or np.isnan(max_priority):
        max_priority = 1.0  # Avoid division by zero

    # Composite = 0.4 * uncertainty + 0.4 * disagreement + 0.2 * normalized priority
    # Weight rationale documented in module docstring.
    composite = (
        0.4 * pool["uncertainty_ensemble"].values
        + 0.4 * pool["disagreement"].values
        + 0.2 * (pool["review_priority_score"].values / max_priority)
    ).astype(np.float32)

    # ── Bonuses ──────────────────────────────────────────────────────
    # Beats in 'noisy_ok' segments get a 20% bonus — they are borderline
    if "segment_idx" in pool.columns and len(segment_quality_preds) > 0:
        noisy_ok_segments = set(
            segment_quality_preds.loc[
                segment_quality_preds["quality_label"] == "noisy_ok", "segment_idx"
            ]
        )
        noisy_ok_mask = pool["segment_idx"].isin(noisy_ok_segments).values
        composite = composite * np.where(noisy_ok_mask, 1.20, 1.0).astype(np.float32)

    # Beats with pots_transition_candidate get a 10% bonus
    if "pots_transition_candidate" in pool.columns:
        pots_mask = pool["pots_transition_candidate"].fillna(False).values.astype(bool)
        composite = composite * np.where(pots_mask, 1.10, 1.0).astype(np.float32)

    pool["composite_score"] = composite
    pool["al_iteration"] = al_iteration

    # ── Sort and truncate ────────────────────────────────────────────
    pool = pool.nlargest(n_candidates, "composite_score")

    # ── Select output columns ────────────────────────────────────────
    out_cols = [
        "peak_id", "segment_idx", "p_artifact_ensemble", "p_artifact_tabular",
        "p_artifact_cnn", "disagreement", "uncertainty_ensemble",
        "review_priority_score", "composite_score", "selection_strategy",
        "al_iteration",
    ]
    # Only include columns that exist
    out_cols = [c for c in out_cols if c in pool.columns]
    result = pool[out_cols].reset_index(drop=True)

    log.info("Returning %d annotation candidates (iteration %d)", len(result), al_iteration)
    return result


def _empty_candidates() -> pd.DataFrame:
    """Return an empty DataFrame with the expected candidate schema."""
    return pd.DataFrame(columns=[
        "peak_id", "segment_idx", "p_artifact_ensemble", "p_artifact_tabular",
        "p_artifact_cnn", "disagreement", "uncertainty_ensemble",
        "review_priority_score", "composite_score", "selection_strategy",
        "al_iteration",
    ])


# ===================================================================== #
#  Label recording (atomic update)                                      #
# ===================================================================== #
def record_labels(
    annotation_results: pd.DataFrame,
    labels_path: str,
    al_iteration: int,
) -> None:
    """Merge new annotations back into labels.parquet (atomically).

    Parameters
    ----------
    annotation_results : pd.DataFrame
        Must contain ``peak_id`` (int) and ``new_label`` (str).
    labels_path : str
        Path to the canonical ``labels.parquet``.
    al_iteration : int
        Iteration number to stamp on updated rows.

    Notes
    -----
    Atomic write: writes to a temp file first, then renames to the
    target path.  This prevents partial overwrites on crash.
    """
    required = {"peak_id", "new_label"}
    if not required.issubset(annotation_results.columns):
        raise ValueError(f"annotation_results missing: {required - set(annotation_results.columns)}")

    invalid = set(annotation_results["new_label"].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(f"Invalid labels: {invalid}.  Valid: {VALID_LABELS}")

    labels = pd.read_parquet(labels_path)
    log.info("Loaded %d existing labels from %s", len(labels), labels_path)

    # Ensure columns exist for AL metadata
    if "al_iteration" not in labels.columns:
        labels["al_iteration"] = np.nan
    if "uncertainty_score" not in labels.columns:
        labels["uncertainty_score"] = np.nan
    if "disagreement_score" not in labels.columns:
        labels["disagreement_score"] = np.nan

    # Update labels for matching peak_ids
    update_ids = set(annotation_results["peak_id"])
    mask = labels["peak_id"].isin(update_ids)
    n_updated = int(mask.sum())

    for _, row in annotation_results.iterrows():
        pid = row["peak_id"]
        idx = labels.index[labels["peak_id"] == pid]
        if len(idx) == 0:
            log.warning("peak_id %d not found in labels — skipping", pid)
            continue
        labels.loc[idx, "label"] = row["new_label"]
        labels.loc[idx, "al_iteration"] = al_iteration
        # Copy over uncertainty/disagreement if available
        if "uncertainty_ensemble" in annotation_results.columns:
            labels.loc[idx, "uncertainty_score"] = row.get("uncertainty_ensemble", np.nan)
        if "disagreement" in annotation_results.columns:
            labels.loc[idx, "disagreement_score"] = row.get("disagreement", np.nan)

    # Atomic write: temp file then rename
    out_path = Path(labels_path)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    labels.to_parquet(tmp_path, index=False, compression="snappy")
    tmp_path.rename(out_path)

    log.info("Updated %d labels (iteration %d), saved → %s", n_updated, al_iteration, out_path)


# ===================================================================== #
#  Iteration summary                                                    #
# ===================================================================== #
def get_iteration_summary(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize annotation progress by AL iteration.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Must contain ``al_iteration`` (may be NaN for original labels)
        and ``label``.

    Returns
    -------
    pd.DataFrame
        One row per iteration with counts and mean scores.
    """
    if "al_iteration" not in labels_df.columns:
        log.info("No al_iteration column — no active learning iterations recorded yet")
        return pd.DataFrame(columns=[
            "al_iteration", "n_labels", "label_distribution",
            "mean_uncertainty", "mean_disagreement",
        ])

    df = labels_df.copy()
    # Treat NaN al_iteration as iteration 0 (original labels)
    df["al_iteration"] = df["al_iteration"].fillna(0).astype(int)

    rows = []
    for it, grp in df.groupby("al_iteration"):
        dist = grp["label"].value_counts().to_dict()
        mean_unc = grp["uncertainty_score"].mean() if "uncertainty_score" in grp.columns else float("nan")
        mean_dis = grp["disagreement_score"].mean() if "disagreement_score" in grp.columns else float("nan")
        rows.append({
            "al_iteration": int(it),
            "n_labels": len(grp),
            "label_distribution": dist,
            "mean_uncertainty": mean_unc,
            "mean_disagreement": mean_dis,
        })

    return pd.DataFrame(rows)


# ===================================================================== #
#  CLI                                                                  #
# ===================================================================== #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sampler.py",
        description="Active-learning candidate selection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sel = sub.add_parser("select", help="Select annotation candidates")
    p_sel.add_argument("--ensemble-preds", required=True, help="Path to ensemble_preds.parquet")
    p_sel.add_argument("--labels", required=True, help="Path to labels.parquet")
    p_sel.add_argument("--segments", required=True, help="Path to segments.parquet")
    p_sel.add_argument("--segment-quality-preds", required=True, help="Path to segment_quality_preds.parquet")
    p_sel.add_argument("--n-candidates", type=int, default=500)
    p_sel.add_argument("--strategy", default="combined", choices=list(VALID_STRATEGIES))
    p_sel.add_argument("--al-iteration", type=int, default=1)
    p_sel.add_argument("--output", required=True, help="Output parquet path")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "select":
        ensemble_preds = pd.read_parquet(args.ensemble_preds)
        labels_df = pd.read_parquet(args.labels)
        # Store source path for segment_idx resolution
        labels_df.attrs["_source_path"] = args.labels
        segments_df = pd.read_parquet(args.segments)
        sqp = pd.read_parquet(args.segment_quality_preds)

        result = select_annotation_candidates(
            ensemble_preds=ensemble_preds,
            labels_df=labels_df,
            segments_df=segments_df,
            segment_quality_preds=sqp,
            n_candidates=args.n_candidates,
            strategy=args.strategy,
            al_iteration=args.al_iteration,
        )

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out, index=False, compression="snappy")
        log.info("Saved %d candidates → %s", len(result), out)

        # ── Summary ──────────────────────────────────────────────────
        print(f"\n{'=' * 72}")
        print("  Active Learning Candidate Selection")
        print(f"{'=' * 72}")
        print(f"  Strategy: {args.strategy}")
        print(f"  Iteration: {args.al_iteration}")
        print(f"  Candidates: {len(result)}")
        if len(result) > 0:
            print(f"\n  Strategy distribution:")
            for strat, cnt in result["selection_strategy"].value_counts().items():
                print(f"    {strat}: {cnt}")
            print(f"\n  Composite score: mean={result['composite_score'].mean():.4f}  "
                  f"max={result['composite_score'].max():.4f}")
            if "p_artifact_ensemble" in result.columns:
                print(f"  p_ensemble range: [{result['p_artifact_ensemble'].min():.4f}, "
                      f"{result['p_artifact_ensemble'].max():.4f}]")
        print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
