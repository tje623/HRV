#!/usr/bin/env python3
"""
scripts/validate_retrained_model.py — Old-vs-new tabular model comparison.

Loads both model artifacts, runs inference on the full merged feature set,
and prints a side-by-side report covering prediction counts, PR/ROC-AUC,
feature importances, agreement, and boundary-shift checks.  Fires WARNING
lines for any sanity-check failures and exits 0 (PASS) or 1 (FAIL).

The old model is always run with its own stored feature_columns — so it works
even when beat_features_merged.parquet contains more columns (41 features for
the new model vs 32 for the old).

Usage
-----
    cd "/Volumes/xHRV/Artifact Detector"
    source /Users/tannereddy/.envs/hrv/bin/activate

    python scripts/validate_retrained_model.py \\
        --old-model    models/beat_tabular_v2.joblib \\
        --new-model    models/beat_tabular_v3_merged.joblib \\
        --beat-features data/processed/beat_features_merged.parquet \\
        --labels        data/processed/labels.parquet \\
        --peaks         data/processed/peaks.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.validate_retrained_model")

# ── Label vocabulary (mirrors beat_artifact_tabular.py) ──────────────────────
ARTIFACT_LABELS = frozenset({"artifact"})
VALID_LABELS    = frozenset({"clean", "artifact", "interpolated",
                              "phys_event", "missed_original"})


# ── Model helpers ─────────────────────────────────────────────────────────────

def _load_artifact(path: Path) -> dict:
    if not path.exists():
        logger.error("Model file not found: %s", path)
        sys.exit(1)
    artifact = joblib.load(path)
    for key in ("model", "feature_columns", "train_medians", "optimal_threshold"):
        if key not in artifact:
            logger.error("Model artifact missing required key '%s': %s", key, path)
            sys.exit(1)
    return artifact


def _predict(features_df: pd.DataFrame, artifact: dict, label: str) -> tuple[pd.DataFrame, float]:
    """Run inference with a loaded model artifact.

    Only the feature columns the model was trained on are selected from
    features_df, so old and new models can share the same (wider) DataFrame.

    Returns
    -------
    preds : DataFrame with columns (peak_id, p_artifact, predicted)
    threshold : float — the optimal threshold used for the predicted column
    """
    clf           = artifact["model"]
    feature_cols: list[str]        = artifact["feature_columns"]
    train_medians: dict[str, float] = artifact["train_medians"]
    threshold: float               = artifact["optimal_threshold"]

    df = features_df.copy()
    if df.index.name == "peak_id":
        df = df.reset_index()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.error(
            "[%s] Model expects %d features; %d are missing from input: %s",
            label, len(feature_cols), len(missing), missing[:10],
        )
        sys.exit(1)

    # Median imputation — replicates _apply_median_imputation from
    # beat_artifact_tabular.py:  stored medians fill known NaN cols,
    # any remaining NaN filled with 0.0.
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            fill_val = float(train_medians.get(col, 0.0))
            X[col] = X[col].fillna(fill_val)

    proba    = clf.predict_proba(X.values)[:, 1].astype(np.float32)
    peak_ids = df["peak_id"].values.astype(np.int64)

    return pd.DataFrame({
        "peak_id":   peak_ids,
        "p_artifact": proba,
        "predicted":  proba >= threshold,
    }), threshold


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _compute_auc(preds: pd.DataFrame, labels_df: pd.DataFrame) -> dict:
    """PR-AUC and ROC-AUC computed on labeled beats only."""
    lab = labels_df.copy()
    if lab.index.name == "peak_id" and "peak_id" not in lab.columns:
        lab = lab.reset_index()

    merged = preds.merge(lab[["peak_id", "label"]], on="peak_id", how="inner")
    merged = merged[merged["label"].isin(VALID_LABELS)]

    n_labeled = len(merged)
    if n_labeled == 0:
        return dict(pr_auc=float("nan"), roc_auc=float("nan"),
                    n_labeled=0, n_artifact=0, n_clean=0)

    y      = merged["label"].isin(ARTIFACT_LABELS).astype(int).values
    scores = merged["p_artifact"].values
    n_pos  = int(y.sum())
    n_neg  = n_labeled - n_pos

    if n_pos == 0 or n_neg == 0:
        return dict(pr_auc=float("nan"), roc_auc=float("nan"),
                    n_labeled=n_labeled, n_artifact=n_pos, n_clean=n_neg)

    return dict(
        pr_auc   = float(average_precision_score(y, scores)),
        roc_auc  = float(roc_auc_score(y, scores)),
        n_labeled = n_labeled,
        n_artifact = n_pos,
        n_clean    = n_neg,
    )


def _top_features(artifact: dict, n: int = 20) -> list[tuple[str, float]]:
    """Return (feature_name, gain) pairs.

    Reads from stored val_metrics["top_features"] if present; otherwise
    falls back to recomputing from the LightGBM booster.
    """
    stored = (artifact.get("val_metrics") or {}).get("top_features")
    if stored:
        return [(r["feature"], float(r["gain"])) for r in stored[:n]]
    try:
        clf    = artifact["model"]
        cols   = artifact["feature_columns"]
        gains  = clf.booster_.feature_importance(importance_type="gain")
        pairs  = sorted(zip(cols, gains.tolist()), key=lambda x: -x[1])
        return pairs[:n]
    except Exception:
        return []


# ── Main validation logic ─────────────────────────────────────────────────────

def validate(
    old_model_path:      Path,
    new_model_path:      Path,
    beat_features_path:  Path,
    labels_path:         Path,
    peaks_path:          Path,          # loaded but reserved for future checks
) -> bool:
    """Run the full validation and print a report.

    Returns True if all sanity checks pass (PASS), False if any fire (FAIL).
    """
    # ── Load inputs ───────────────────────────────────────────────────────────
    logger.info("Loading features from %s", beat_features_path)
    features = pd.read_parquet(beat_features_path)
    n_total  = len(features)
    logger.info("Total beats in feature file: %d", n_total)

    logger.info("Loading labels from %s", labels_path)
    labels = pd.read_parquet(labels_path)
    if labels.index.name == "peak_id" and "peak_id" not in labels.columns:
        labels = labels.reset_index()

    # "reviewed" column may or may not be present depending on pipeline version
    has_reviewed_col = "reviewed" in labels.columns

    # All beats with an explicit valid label (used for PR/ROC-AUC)
    labeled_mask   = labels["label"].isin(VALID_LABELS)
    labeled_pids   = set(labels.loc[labeled_mask, "peak_id"].values)

    # Reviewed-clean beats: explicitly labeled as clean.
    # If a "reviewed" column exists, restrict to beats where reviewed is True.
    if has_reviewed_col:
        clean_mask = labeled_mask & (labels["label"] == "clean") & labels["reviewed"].astype(bool)
    else:
        clean_mask = labeled_mask & (labels["label"] == "clean")
    clean_reviewed_pids = set(labels.loc[clean_mask, "peak_id"].values)

    logger.info(
        "Labels: %d rows, %d with valid label, %d reviewed-clean",
        len(labels), len(labeled_pids), len(clean_reviewed_pids),
    )

    # peaks.parquet: loaded for potential future segment-level stratification
    logger.info("Loading peaks from %s", peaks_path)
    _peaks = pd.read_parquet(peaks_path)   # noqa: F841  (reserved for future use)

    logger.info("Loading old model: %s", old_model_path)
    old_art = _load_artifact(old_model_path)
    logger.info("Loading new model: %s", new_model_path)
    new_art = _load_artifact(new_model_path)

    # ── Predict ───────────────────────────────────────────────────────────────
    logger.info("Running old model (%d features) …", len(old_art["feature_columns"]))
    old_preds, old_thresh = _predict(features, old_art, "old")

    logger.info("Running new model (%d features) …", len(new_art["feature_columns"]))
    new_preds, new_thresh = _predict(features, new_art, "new")

    # ── Per-model aggregate stats ─────────────────────────────────────────────
    old_flagged  = int(old_preds["predicted"].sum())
    new_flagged  = int(new_preds["predicted"].sum())
    old_flag_pct = 100.0 * old_flagged / max(n_total, 1)
    new_flag_pct = 100.0 * new_flagged / max(n_total, 1)
    old_mean_p   = float(old_preds["p_artifact"].mean())
    new_mean_p   = float(new_preds["p_artifact"].mean())

    old_auc = _compute_auc(old_preds, labels)
    new_auc = _compute_auc(new_preds, labels)

    # ── Agreement metrics ─────────────────────────────────────────────────────
    cmp = old_preds[["peak_id", "predicted"]].merge(
        new_preds[["peak_id", "predicted"]],
        on="peak_id",
        suffixes=("_old", "_new"),
        how="inner",
    )
    n_cmp          = len(cmp)
    n_agree        = int((cmp["predicted_old"] == cmp["predicted_new"]).sum())
    n_c2a          = int((~cmp["predicted_old"] &  cmp["predicted_new"]).sum())  # clean→artifact
    n_a2c          = int(( cmp["predicted_old"] & ~cmp["predicted_new"]).sum())  # artifact→clean
    n_flipped      = n_c2a + n_a2c

    # Fraction of flipped beats that are labeled (reviewed) vs unlabeled
    if n_flipped > 0:
        flipped_pids         = set(cmp.loc[cmp["predicted_old"] != cmp["predicted_new"],
                                           "peak_id"].values)
        n_flipped_reviewed   = len(flipped_pids & labeled_pids)
        n_flipped_unreviewed = n_flipped - n_flipped_reviewed
    else:
        n_flipped_reviewed = n_flipped_unreviewed = 0

    # ── Boundary-shift check on reviewed-clean beats ──────────────────────────
    if clean_reviewed_pids:
        clean_cmp          = cmp[cmp["peak_id"].isin(clean_reviewed_pids)]
        n_clean_reviewed   = len(clean_cmp)
        n_clean_now_art    = int(clean_cmp["predicted_new"].sum())
        boundary_shift_pct = 100.0 * n_clean_now_art / max(n_clean_reviewed, 1)
    else:
        n_clean_reviewed = n_clean_now_art = 0
        boundary_shift_pct = 0.0

    # ── Feature importances ───────────────────────────────────────────────────
    old_feats = _top_features(old_art)
    new_feats = _top_features(new_art)

    # ── Print report ──────────────────────────────────────────────────────────
    W = 72

    def _fmt(v: float, dec: int = 4) -> str:
        return f"{v:.{dec}f}" if not np.isnan(v) else "N/A"

    print()
    print("═" * W)
    print("  MODEL VALIDATION REPORT")
    print("═" * W)
    print(f"  Old model  : {old_model_path.name}")
    print(f"               trained {old_art.get('trained_at', 'unknown')}")
    print(f"               {len(old_art['feature_columns'])} features")
    print(f"  New model  : {new_model_path.name}")
    print(f"               trained {new_art.get('trained_at', 'unknown')}")
    print(f"               {len(new_art['feature_columns'])} features")
    print()
    print(f"  {'Metric':<42}  {'Old':>11}  {'New':>11}")
    print(f"  {'─'*42}  {'─'*11}  {'─'*11}")
    print(f"  {'Total beats (full dataset)':<42}  {n_total:>11,}")
    print(f"  {'Optimal threshold':<42}  {old_thresh:>11.4f}  {new_thresh:>11.4f}")
    print(f"  {'Beats flagged as artifact':<42}  {old_flagged:>11,}  {new_flagged:>11,}")
    print(f"  {'  (% of total)':<42}  {old_flag_pct:>10.2f}%  {new_flag_pct:>10.2f}%")
    print(f"  {'Mean p_artifact (all beats)':<42}  {old_mean_p:>11.6f}  {new_mean_p:>11.6f}")
    print()
    print(f"  {'Labeled beats used for AUC':<42}  {old_auc['n_labeled']:>11,}  {new_auc['n_labeled']:>11,}")
    print(f"  {'  of which: artifact':<42}  {old_auc['n_artifact']:>11,}  {new_auc['n_artifact']:>11,}")
    print(f"  {'  of which: clean/other':<42}  {old_auc['n_clean']:>11,}  {new_auc['n_clean']:>11,}")
    print()
    delta_pr = new_auc["pr_auc"] - old_auc["pr_auc"]
    delta_roc = new_auc["roc_auc"] - old_auc["roc_auc"]
    pr_sign  = "+" if delta_pr  >= 0 else ""
    roc_sign = "+" if delta_roc >= 0 else ""
    print(f"  {'PR-AUC  (artifact class)  ← PRIMARY':<42}  "
          f"{_fmt(old_auc['pr_auc']):>11}  {_fmt(new_auc['pr_auc']):>11}")
    if not (np.isnan(old_auc["pr_auc"]) or np.isnan(new_auc["pr_auc"])):
        print(f"  {'  Δ PR-AUC (new − old)':<42}  {'':>11}  "
              f"{pr_sign}{delta_pr:>+10.4f}")
    print(f"  {'ROC-AUC':<42}  "
          f"{_fmt(old_auc['roc_auc']):>11}  {_fmt(new_auc['roc_auc']):>11}")
    if not (np.isnan(old_auc["roc_auc"]) or np.isnan(new_auc["roc_auc"])):
        print(f"  {'  Δ ROC-AUC (new − old)':<42}  {'':>11}  "
              f"{roc_sign}{delta_roc:>+10.4f}")

    print()
    print("─" * W)
    print("  AGREEMENT  (at each model's own optimal threshold)")
    print("─" * W)
    print(f"  Beats compared        : {n_cmp:,}")
    print(f"  Agreed (both same)    : {n_agree:,}  ({100.0*n_agree/max(n_cmp,1):.2f}%)")
    print(f"  clean → artifact      : {n_c2a:,}  ({100.0*n_c2a/max(n_cmp,1):.3f}%)")
    print(f"  artifact → clean      : {n_a2c:,}  ({100.0*n_a2c/max(n_cmp,1):.3f}%)")
    print(f"  Total flipped         : {n_flipped:,}")
    if n_flipped > 0:
        print(f"    reviewed (labeled)  : {n_flipped_reviewed:,}  "
              f"({100.0*n_flipped_reviewed/max(n_flipped,1):.1f}%)")
        print(f"    unreviewed          : {n_flipped_unreviewed:,}  "
              f"({100.0*n_flipped_unreviewed/max(n_flipped,1):.1f}%)")
    print()
    print(f"  Reviewed-clean beats       : {n_clean_reviewed:,}")
    print(f"  Now flagged by new model   : {n_clean_now_art:,}  ({boundary_shift_pct:.2f}%)")

    print()
    print("─" * W)
    print("  TOP 20 FEATURE IMPORTANCES  (gain)")
    print("─" * W)
    if not old_feats and not new_feats:
        print("  (not available — models may predate importance storage)")
    else:
        col_w = 26
        print(f"  {'#':<3}  {'OLD MODEL':<{col_w+9}}  {'NEW MODEL':<{col_w+9}}")
        print(f"  {'─'*3}  {'─'*(col_w+9)}  {'─'*(col_w+9)}")
        for i in range(max(len(old_feats), len(new_feats))):
            if i < len(old_feats):
                oname, ogain = old_feats[i]
                old_s = f"{oname[:col_w]:<{col_w}} {ogain:>8.0f}"
            else:
                old_s = " " * (col_w + 9)
            if i < len(new_feats):
                nname, ngain = new_feats[i]
                new_s = f"{nname[:col_w]:<{col_w}} {ngain:>8.0f}"
            else:
                new_s = ""
            print(f"  {i+1:<3}  {old_s}  {new_s}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    sanity_warnings: list[str] = []

    if new_flag_pct > 10.0:
        sanity_warnings.append(
            f"New model flags {new_flag_pct:.1f}% of all beats as artifact "
            f"(threshold > 10%) — possible miscalibration"
        )
    if new_flag_pct < 0.5:
        sanity_warnings.append(
            f"New model flags only {new_flag_pct:.2f}% of beats "
            f"(threshold < 0.5%) — possible undertrained"
        )
    if not (np.isnan(old_auc["pr_auc"]) or np.isnan(new_auc["pr_auc"])):
        if new_auc["pr_auc"] < old_auc["pr_auc"]:
            sanity_warnings.append(
                f"New model PR-AUC ({new_auc['pr_auc']:.4f}) < "
                f"old ({old_auc['pr_auc']:.4f}) — "
                f"retraining may have degraded performance"
            )
    if boundary_shift_pct > 5.0:
        sanity_warnings.append(
            f"{boundary_shift_pct:.1f}% of previously-clean reviewed beats "
            f"are now flagged as artifact (threshold > 5%) — "
            f"possible decision-boundary shift"
        )

    print()
    print("─" * W)
    print("  SANITY CHECKS")
    print("─" * W)
    if sanity_warnings:
        for w in sanity_warnings:
            print(f"  WARNING: {w}")
    else:
        print("  All checks passed — no warnings.")

    passed = len(sanity_warnings) == 0

    print()
    print("═" * W)
    if passed:
        print("  RESULT: PASS")
    else:
        print(f"  RESULT: FAIL — {len(sanity_warnings)} sanity warning(s) fired")
    print("═" * W)
    print()

    return passed


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare old vs new tabular model to guard against retraining regressions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--old-model",      type=Path, required=True,
                        help="Path to old model .joblib (e.g. beat_tabular_v2.joblib)")
    parser.add_argument("--new-model",      type=Path, required=True,
                        help="Path to new model .joblib (e.g. beat_tabular_v3_merged.joblib)")
    parser.add_argument("--beat-features",  type=Path, required=True,
                        help="beat_features_merged.parquet (41 features — new model uses all; "
                             "old model selects its own subset automatically)")
    parser.add_argument("--labels",         type=Path, required=True,
                        help="labels.parquet")
    parser.add_argument("--peaks",          type=Path, required=True,
                        help="peaks.parquet (loaded; reserved for future segment-level checks)")
    args = parser.parse_args()

    for flag, path in [
        ("--old-model",     args.old_model),
        ("--new-model",     args.new_model),
        ("--beat-features", args.beat_features),
        ("--labels",        args.labels),
        ("--peaks",         args.peaks),
    ]:
        if not path.exists():
            logger.error("File not found for %s: %s", flag, path)
            sys.exit(1)

    passed = validate(
        old_model_path     = args.old_model,
        new_model_path     = args.new_model,
        beat_features_path = args.beat_features,
        labels_path        = args.labels,
        peaks_path         = args.peaks,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
