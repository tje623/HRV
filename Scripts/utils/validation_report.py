#!/usr/bin/env python3
"""
Scripts/utils/validation_report.py — Full-dataset validation report (Prompt D).

Compares v6 (new) vs v1 (old) models on the full reviewed subset.
Run from /Volumes/xHRV with hrv venv active.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve,
)

PROCESSED = Path("Data/Processed")
MODELS = Path("Models")


def load_reviewed_with_preds(preds_path: Path) -> pd.DataFrame:
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    preds  = pd.read_parquet(preds_path)
    if preds.index.name == "peak_id":
        preds = preds.reset_index()

    keep = labels["reviewed"] | (labels["label"] == "artifact")
    reviewed = labels[keep].copy()
    reviewed["target"] = (reviewed["label"] == "artifact").astype(int)

    merged = reviewed.merge(preds, on="peak_id", how="inner")
    return merged


def prob_distribution_bins(proba: np.ndarray, n_bins: int = 10) -> None:
    edges = np.linspace(0, 1, n_bins + 1)
    print(f"\n  Probability distribution ({n_bins} equal bins):")
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        mask = (proba >= lo) & (proba <= hi if i == n_bins-1 else proba < hi)
        n = int(mask.sum())
        bar = "█" * min(40, int(40 * n / max(len(proba), 1)))
        print(f"    [{lo:.1f},{hi:.1f}): {n:>8,}  {bar}")


def report_model(label: str, merged: pd.DataFrame, prob_col: str,
                 threshold: float) -> dict:
    y = merged["target"].values
    p = merged[prob_col].values
    pr_auc  = float(average_precision_score(y, p))
    roc_auc = float(roc_auc_score(y, p))
    y_pred  = (p >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    prevalence = float(y.mean())

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Reviewed beats: {len(y):,}  (artifact: {y.sum():,}, {prevalence*100:.2f}%)")
    print(f"  PR-AUC:         {pr_auc:.4f}")
    print(f"  ROC-AUC:        {roc_auc:.4f}")
    print(f"  Threshold:      {threshold:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1:             {f1:.4f}")
    print(f"  Confusion matrix (threshold={threshold:.4f}):")
    print(f"    TN={tn:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TP={tp:,}")
    # ── Subtype-stratified artifact recall ───────────────────────────────
    if "subtype" in merged.columns:
        print(f"  Recall by subtype (threshold={threshold:.4f}):")
        for st in ("spurious", "interpolate"):
            mask = (merged["label"].values == "artifact") & (merged["subtype"].values == st)
            n = int(mask.sum())
            if n == 0:
                print(f"    {st:>11}: 0 examples")
                continue
            tp_st = int(y_pred[mask].sum())
            print(f"    {st:>11}: {tp_st} / {n} = {tp_st/n:.3f}")
    prob_distribution_bins(p)

    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "precision": precision,
            "recall": recall, "f1": f1, "threshold": threshold}


def main() -> None:
    import joblib

    # ── V1 model (old leaky schema) ──────────────────────────────────────
    v1_artifact = joblib.load(MODELS / "beat_tabular_v1.joblib")
    v1_thresh = v1_artifact.get("optimal_threshold", 0.5)
    # Load v1 preds from the BACKUP if they exist, else use hardcoded metrics.
    v1_path = PROCESSED / "beat_tabular_preds_v1_backup.parquet"
    if v1_path.exists():
        v1_merged = load_reviewed_with_preds(v1_path)
        v1_metrics = report_model("v1 model (old schema, 40 features)", v1_merged,
                                   "p_artifact_tabular", v1_thresh)
    else:
        print("\n[NOTE] v1 backup not found — using hardcoded v1 metrics from earlier run:")
        print("  PR-AUC: 0.7885,  ROC-AUC: 0.9875")
        v1_metrics = {"pr_auc": 0.7885, "roc_auc": 0.9875}

    # ── V6 model (new clean schema) ───────────────────────────────────────
    v6_artifact = joblib.load(MODELS / "beat_tabular_v6.joblib")
    v6_thresh = v6_artifact.get("optimal_threshold", 0.5)
    v6_merged = load_reviewed_with_preds(PROCESSED / "beat_tabular_preds.parquet")
    v6_metrics = report_model("v6 model (clean schema, 37 features)", v6_merged,
                               "p_artifact", v6_thresh)

    # ── Predicted prevalence over all 58M beats ───────────────────────────
    preds_full = pd.read_parquet(PROCESSED / "beat_tabular_preds.parquet")
    if preds_full.index.name == "peak_id":
        preds_full = preds_full.reset_index()
    p_all = preds_full["p_artifact"].values
    n_flagged = int((p_all >= v6_thresh).sum())
    pct = 100.0 * n_flagged / len(p_all)
    print(f"\n{'='*60}")
    print(f"  Full-dataset predicted prevalence (58M beats)")
    print(f"{'='*60}")
    print(f"  Total beats:      {len(p_all):,}")
    print(f"  Threshold:        {v6_thresh:.4f}")
    print(f"  Predicted artifact: {n_flagged:,}  ({pct:.2f}%)")
    if not (0.5 <= pct <= 5.0):
        print(f"  ⚠  Prevalence {pct:.2f}% outside expected 0.5–5% range — investigate")
    else:
        print(f"  ✓  Prevalence in reasonable range")

    # ── Comparison table ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Summary comparison table")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'PR-AUC':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*52}")
    print(f"  {'v1 (old, leaky schema)':<30} {v1_metrics['pr_auc']:>10.4f} "
          f"{v1_metrics['roc_auc']:>10.4f}")
    print(f"  {'v6 (new, clean schema)':<30} {v6_metrics['pr_auc']:>10.4f} "
          f"{v6_metrics['roc_auc']:>10.4f}")
    delta_pr  = v6_metrics['pr_auc']  - v1_metrics['pr_auc']
    delta_roc = v6_metrics['roc_auc'] - v1_metrics['roc_auc']
    print(f"  {'Δ (v6 − v1)':<30} {delta_pr:>+10.4f} {delta_roc:>+10.4f}")
    print(f"  {'-'*52}")

    if delta_pr > 0:
        print("\n  ✓ v6 improves on v1 in PR-AUC — feature cleanup had positive effect.")
    elif delta_pr > -0.02:
        print("\n  ~ v6 roughly matches v1 in PR-AUC — feature cleanup is neutral.")
    else:
        print("\n  ⚠ v6 underperforms v1 in PR-AUC — investigate feature loss.")


if __name__ == "__main__":
    main()
