#!/usr/bin/env python3
"""
ecgclean.models.ensemble
========================
Stage 3: Ensemble fusion of tabular GBM and CNN beat-level artifact
predictions into a single calibrated probability.

Two fusion strategies are provided:
    1. Linear blend  – ``fuse()``  (weighted average, default alpha=0.5)
    2. Meta-classifier – ``train_meta_classifier()`` (logistic regression
       stacked on both model outputs plus their disagreement)

The best strategy is whichever yields the higher PR-AUC on the validation
labels.

CLI
---
    python ecgclean/models/ensemble.py fuse   --tabular-preds ... --cnn-preds ... --output ...
    python ecgclean/models/ensemble.py tune-alpha --tabular-preds ... --cnn-preds ... --labels ...
    python ecgclean/models/ensemble.py train-meta --tabular-preds ... --cnn-preds ... --labels ... --output ...
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ENSEMBLE_ALPHA, ENSEMBLE_THRESHOLD

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

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
# Valid artifact labels (same label vocabulary used across the pipeline)
# ---------------------------------------------------------------------------
ARTIFACT_LABELS = {"artifact"}
VALID_LABELS = {"clean", "artifact", "interpolated", "phys_event", "missed_original"}


# ===================================================================== #
#  Linear-blend ensemble                                                #
# ===================================================================== #
def fuse(
    tabular_preds: pd.DataFrame,
    cnn_preds: pd.DataFrame,
    alpha: float = ENSEMBLE_ALPHA,
    threshold: float = ENSEMBLE_THRESHOLD,
) -> pd.DataFrame:
    """Fuse tabular and CNN artifact probabilities via weighted average.

    Parameters
    ----------
    tabular_preds : pd.DataFrame
        Must contain ``peak_id`` and ``p_artifact_tabular``.
    cnn_preds : pd.DataFrame
        Must contain ``peak_id`` and ``p_artifact_cnn``.
    alpha : float
        Weight for the tabular model.  ``p_ensemble = alpha * p_tab + (1-alpha) * p_cnn``.
    threshold : float
        Decision boundary for ``predicted_artifact``.

    Returns
    -------
    pd.DataFrame
        Columns: peak_id, p_artifact_tabular, p_artifact_cnn,
        p_artifact_ensemble, disagreement, uncertainty_ensemble,
        predicted_artifact.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    # ── Merge on peak_id ─────────────────────────────────────────────
    required_tab = {"peak_id", "p_artifact_tabular"}
    required_cnn = {"peak_id", "p_artifact_cnn"}
    if not required_tab.issubset(tabular_preds.columns):
        raise ValueError(f"tabular_preds missing columns: {required_tab - set(tabular_preds.columns)}")
    if not required_cnn.issubset(cnn_preds.columns):
        raise ValueError(f"cnn_preds missing columns: {required_cnn - set(cnn_preds.columns)}")

    merged = tabular_preds[["peak_id", "p_artifact_tabular"]].merge(
        cnn_preds[["peak_id", "p_artifact_cnn"]],
        on="peak_id",
        how="inner",
    )
    if len(merged) == 0:
        raise ValueError("Inner join on peak_id produced 0 rows — check that IDs match")

    log.info("Fusing %d beats with alpha=%.2f, threshold=%.2f", len(merged), alpha, threshold)

    # ── Derived columns ──────────────────────────────────────────────
    p_tab = merged["p_artifact_tabular"].values.astype(np.float32)
    p_cnn = merged["p_artifact_cnn"].values.astype(np.float32)

    p_ens = (alpha * p_tab + (1.0 - alpha) * p_cnn).astype(np.float32)
    disagreement = np.abs(p_tab - p_cnn).astype(np.float32)
    uncertainty = (1.0 - 2.0 * np.abs(p_ens - 0.5)).astype(np.float32)

    merged["p_artifact_ensemble"] = p_ens
    merged["disagreement"] = disagreement
    merged["uncertainty_ensemble"] = uncertainty
    merged["predicted_artifact"] = p_ens >= threshold

    return merged


# ===================================================================== #
#  Alpha tuning (grid search)                                           #
# ===================================================================== #
def tune_alpha(
    tabular_preds: pd.DataFrame,
    cnn_preds: pd.DataFrame,
    labels_df: pd.DataFrame,
    alpha_grid: list[float] | None = None,
) -> dict:
    """Grid-search for the blend weight that maximizes PR-AUC.

    Parameters
    ----------
    tabular_preds, cnn_preds : pd.DataFrame
        Same format as ``fuse()``.
    labels_df : pd.DataFrame
        Must contain ``peak_id`` and ``label``.
    alpha_grid : list[float] | None
        Candidate alpha values.  Default ``[0.1, 0.2, ..., 0.9]``.

    Returns
    -------
    dict
        ``best_alpha``, ``best_pr_auc``, ``grid`` (list of dicts with
        ``alpha`` and ``pr_auc``).
    """
    if alpha_grid is None:
        alpha_grid = [round(x * 0.1, 1) for x in range(1, 10)]

    # Build binary target
    labels_sub = labels_df[["peak_id", "label"]].copy()
    labels_sub["y"] = labels_sub["label"].isin(ARTIFACT_LABELS).astype(int)

    n_pos = int(labels_sub["y"].sum())
    n_neg = int((labels_sub["y"] == 0).sum())
    log.info("tune_alpha: %d labeled beats (%d artifact, %d non-artifact)", len(labels_sub), n_pos, n_neg)

    if n_pos == 0 or n_neg == 0:
        log.warning("Single-class labels — PR-AUC is undefined.  Returning NaN for all alphas.")
        grid = [{"alpha": a, "pr_auc": float("nan")} for a in alpha_grid]
        return {"best_alpha": alpha_grid[len(alpha_grid) // 2], "best_pr_auc": float("nan"), "grid": grid}

    grid: list[dict] = []
    for a in alpha_grid:
        ens = fuse(tabular_preds, cnn_preds, alpha=a)
        ens_labeled = ens.merge(labels_sub[["peak_id", "y"]], on="peak_id", how="inner")
        if len(ens_labeled) == 0:
            grid.append({"alpha": a, "pr_auc": float("nan")})
            continue
        pr_auc = float(average_precision_score(ens_labeled["y"], ens_labeled["p_artifact_ensemble"]))
        grid.append({"alpha": a, "pr_auc": pr_auc})

    # ── Print table ──────────────────────────────────────────────────
    print("\n  Alpha Tuning Grid")
    print("  " + "─" * 28)
    for row in grid:
        flag = ""
        print(f"  alpha={row['alpha']:.1f}  PR-AUC={row['pr_auc']:.6f}{flag}")
    print("  " + "─" * 28)

    best = max(grid, key=lambda r: r["pr_auc"] if not np.isnan(r["pr_auc"]) else -1.0)
    log.info("Best alpha=%.1f  PR-AUC=%.6f", best["alpha"], best["pr_auc"])

    return {"best_alpha": best["alpha"], "best_pr_auc": best["pr_auc"], "grid": grid}


# ===================================================================== #
#  Meta-classifier (logistic regression stacking)                       #
# ===================================================================== #
def train_meta_classifier(
    tabular_preds: pd.DataFrame,
    cnn_preds: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_path: str,
) -> dict:
    """Train a logistic-regression meta-classifier on model outputs.

    Features: ``[p_artifact_tabular, p_artifact_cnn, disagreement]``.
    Target: ``label == "artifact"``.

    Parameters
    ----------
    tabular_preds, cnn_preds : pd.DataFrame
        Same format as ``fuse()``.
    labels_df : pd.DataFrame
        Must contain ``peak_id`` and ``label``.
    output_path : str
        Path to save the meta-classifier via ``joblib``.

    Returns
    -------
    dict
        ``pr_auc``, ``f1``, ``n_train``, ``n_artifact``.
    """
    # Build features
    ens = fuse(tabular_preds, cnn_preds)
    labels_sub = labels_df[["peak_id", "label"]].copy()
    labels_sub["y"] = labels_sub["label"].isin(ARTIFACT_LABELS).astype(int)

    merged = ens.merge(labels_sub[["peak_id", "y"]], on="peak_id", how="inner")
    n_pos = int(merged["y"].sum())
    n_neg = int((merged["y"] == 0).sum())
    log.info("Meta-classifier: %d labeled beats (%d artifact, %d non-artifact)", len(merged), n_pos, n_neg)

    feature_cols = ["p_artifact_tabular", "p_artifact_cnn", "disagreement"]
    X = merged[feature_cols].values.astype(np.float64)
    y = merged["y"].values

    if n_pos == 0 or n_neg == 0:
        log.warning("Single-class labels — cannot train meta-classifier. Saving dummy model.")
        # Save a dummy model that always predicts the majority class
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": None,
                "feature_cols": feature_cols,
                "is_dummy": True,
                "majority_class": 0 if n_pos == 0 else 1,
                "metrics": {"pr_auc": float("nan"), "f1": float("nan"), "n_train": len(merged), "n_artifact": n_pos},
            },
            out,
        )
        return {"pr_auc": float("nan"), "f1": float("nan"), "n_train": len(merged), "n_artifact": n_pos}

    # Train logistic regression with balanced class weights
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
    lr.fit(X, y)

    # Evaluate on the same labeled data (no separate val set at this stage;
    # the pipeline's temporal split happened upstream in stages 1–2a)
    y_prob = lr.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    pr_auc = float(average_precision_score(y, y_prob))
    f1 = float(f1_score(y, y_pred, zero_division=0.0))

    log.info("Meta-classifier  PR-AUC=%.4f  F1=%.4f", pr_auc, f1)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": lr,
            "feature_cols": feature_cols,
            "is_dummy": False,
            "metrics": {"pr_auc": pr_auc, "f1": f1, "n_train": len(merged), "n_artifact": n_pos},
        },
        out,
    )
    log.info("Saved meta-classifier → %s", out)

    return {"pr_auc": pr_auc, "f1": f1, "n_train": len(merged), "n_artifact": n_pos}


# ===================================================================== #
#  CLI                                                                  #
# ===================================================================== #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ensemble.py",
        description="Ensemble fusion of tabular + CNN artifact predictions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── fuse ──────────────────────────────────────────────────────────
    p_fuse = sub.add_parser("fuse", help="Create ensemble predictions")
    p_fuse.add_argument("--tabular-preds", required=True, help="Path to beat_tabular_preds.parquet")
    p_fuse.add_argument("--cnn-preds", default=None,
                        help="Path to beat_cnn_preds.parquet (omit when --tabular-only)")
    p_fuse.add_argument("--output", required=True, help="Output ensemble_preds.parquet")
    p_fuse.add_argument("--alpha", type=float, default=ENSEMBLE_ALPHA, help=f"Blend weight for tabular (default {ENSEMBLE_ALPHA})")
    p_fuse.add_argument("--threshold", type=float, default=ENSEMBLE_THRESHOLD, help=f"Decision threshold (default {ENSEMBLE_THRESHOLD})")
    p_fuse.add_argument("--tabular-only", action="store_true",
                        help="Skip CNN entirely — ensemble = tabular predictions, disagreement = 0.0")

    # ── tune-alpha ────────────────────────────────────────────────────
    p_tune = sub.add_parser("tune-alpha", help="Grid-search for best alpha")
    p_tune.add_argument("--tabular-preds", required=True)
    p_tune.add_argument("--cnn-preds", required=True)
    p_tune.add_argument("--labels", required=True, help="Path to labels.parquet")

    # ── train-meta ────────────────────────────────────────────────────
    p_meta = sub.add_parser("train-meta", help="Train logistic-regression meta-classifier")
    p_meta.add_argument("--tabular-preds", required=True)
    p_meta.add_argument("--cnn-preds", required=True)
    p_meta.add_argument("--labels", required=True)
    p_meta.add_argument("--output", required=True, help="Path to save meta-classifier (.joblib)")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "fuse":
        tab = pd.read_parquet(args.tabular_preds)

        if args.tabular_only:
            # Tabular-only mode: no CNN required.  Build ensemble columns from tabular
            # predictions directly so downstream consumers see the standard schema.
            result = tab[["peak_id", "p_artifact_tabular"]].copy()
            p_ens = result["p_artifact_tabular"].values.astype(np.float32)
            result["p_artifact_cnn"] = np.float32(0.0)
            result["p_artifact_ensemble"] = p_ens
            result["disagreement"] = np.float32(0.0)
            result["uncertainty_ensemble"] = (1.0 - 2.0 * np.abs(p_ens - 0.5)).astype(np.float32)
            result["predicted_artifact"] = p_ens >= args.threshold
            log.info("Tabular-only mode: %d beats, disagreement forced to 0.0", len(result))
        else:
            if args.cnn_preds is None:
                log.error("--cnn-preds is required unless --tabular-only is set")
                sys.exit(1)
            cnn = pd.read_parquet(args.cnn_preds)
            result = fuse(tab, cnn, alpha=args.alpha, threshold=args.threshold)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out, index=False, compression="snappy")
        log.info("Saved ensemble predictions → %s", out)

        # ── Summary ──────────────────────────────────────────────────
        n = len(result)
        n_art = int(result["predicted_artifact"].sum())
        print(f"\n{'=' * 72}")
        print("  Ensemble Predictions")
        print(f"{'=' * 72}")
        print(f"  Total beats: {n}")
        print(f"  Predicted artifact: {n_art} ({100.0 * n_art / max(n, 1):.1f}%)")
        print(f"  Predicted clean:    {n - n_art} ({100.0 * (n - n_art) / max(n, 1):.1f}%)")
        print(f"\n  p_artifact_ensemble stats:")
        print(f"    mean={result['p_artifact_ensemble'].mean():.6f}  "
              f"std={result['p_artifact_ensemble'].std():.6f}  "
              f"min={result['p_artifact_ensemble'].min():.6f}  "
              f"max={result['p_artifact_ensemble'].max():.6f}")
        print(f"\n  disagreement stats:")
        print(f"    mean={result['disagreement'].mean():.6f}  "
              f"std={result['disagreement'].std():.6f}  "
              f"max={result['disagreement'].max():.6f}")
        print(f"\n  First 5 rows:")
        print(result.head().to_string(index=False))
        print(f"{'=' * 72}")

    elif args.command == "tune-alpha":
        tab = pd.read_parquet(args.tabular_preds)
        cnn = pd.read_parquet(args.cnn_preds)
        labels = pd.read_parquet(args.labels)
        result = tune_alpha(tab, cnn, labels)
        print(f"\n  Best alpha: {result['best_alpha']}")
        print(f"  Best PR-AUC: {result['best_pr_auc']}")

    elif args.command == "train-meta":
        tab = pd.read_parquet(args.tabular_preds)
        cnn = pd.read_parquet(args.cnn_preds)
        labels = pd.read_parquet(args.labels)
        result = train_meta_classifier(tab, cnn, labels, args.output)
        print(f"\n  Meta-classifier results:")
        for k, v in result.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
