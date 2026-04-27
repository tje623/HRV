#!/usr/bin/env python3
"""
Scripts/utils/eval_baselines.py — Baseline PR-AUC comparisons (Prompt C).

Computes three baselines on the validation reviewed subset and prints a
comparison table alongside the smoke_test LGBM model.

Run from /Volumes/xHRV with the hrv venv active:
    python Scripts/utils/eval_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROCESSED = Path("Data/Subsets/smoke_test/Processed")
MODELS = Path("Models")

_SEG_RULE_COLS = ["segment_zcr", "segment_qrs_density", "segment_spectral_entropy"]


def _impute(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(medians.get(col, 0.0))
    return df


def _load_reviewed_with_seg_features(
    val_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) reviewed beat rows with segment features joined.

    NaN segment features (short segments that couldn't compute Group 5 features)
    are filled with training-split medians to keep the rule-based baseline
    well-defined without leaking val information.
    """
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    peaks = pd.read_parquet(
        PROCESSED / "peaks.parquet", columns=["peak_id", "segment_idx"]
    )
    seg_feat = pd.read_parquet(PROCESSED / "segment_features.parquet")
    if seg_feat.index.name == "segment_idx":
        seg_feat = seg_feat.reset_index()

    # Reviewed filter (same logic as beat_artifact_tabular.py)
    keep = labels["reviewed"] | (labels["label"] == "artifact")
    reviewed = labels[keep].copy()
    reviewed["target"] = (reviewed["label"] == "artifact").astype(int)
    reviewed = reviewed[reviewed["label"] != "interpolated"]

    # Join segment_idx
    merged = reviewed.merge(peaks, on="peak_id", how="inner")

    # Temporal split — identical to beat_artifact_tabular.py
    unique_segs = sorted(merged["segment_idx"].unique())
    n_train = max(1, int(len(unique_segs) * (1 - val_fraction)))
    train_segs = set(unique_segs[:n_train])
    val_segs = set(unique_segs[n_train:])

    train_df = merged[merged["segment_idx"].isin(train_segs)].copy()
    val_df = merged[merged["segment_idx"].isin(val_segs)].copy()

    # Join segment features
    seg_cols = ["segment_idx"] + _SEG_RULE_COLS
    train_df = train_df.merge(seg_feat[seg_cols], on="segment_idx", how="left")
    val_df = val_df.merge(seg_feat[seg_cols], on="segment_idx", how="left")

    # Fill NaN segment features with training-split medians (no val leakage).
    # NaN arises for short/tail segments where Group 5 features couldn't be
    # computed (too few samples). Imputing with train median makes the rule
    # predict "normal" for those beats, which is conservative.
    train_medians_seg = {
        col: float(train_df[col].median()) if train_df[col].notna().any() else 0.0
        for col in _SEG_RULE_COLS
    }
    for col in _SEG_RULE_COLS:
        train_df[col] = train_df[col].fillna(train_medians_seg[col])
        val_df[col] = val_df[col].fillna(train_medians_seg[col])

    n_val_nan_filled = sum(
        int(val_df[col].isna().sum()) for col in _SEG_RULE_COLS
    )
    print(f"  Segment feature NaN fills (train medians): {train_medians_seg}")
    if n_val_nan_filled > 0:
        print(f"  WARNING: {n_val_nan_filled} remaining NaN after fill — check data")

    return train_df, val_df


def tune_spectral_entropy_threshold(train_df: pd.DataFrame) -> float:
    """Tune spectral_entropy upper-bound threshold on training set only.

    Higher spectral_entropy → noisier signal → more likely artifact.
    Grid searches thresholds to maximise PR-AUC on training split.
    """
    best_t, best_auc = 6.0, 0.0
    for t in np.linspace(3.0, 6.5, 70):
        scores = (train_df["segment_spectral_entropy"] > t).astype(float).values
        if scores.sum() == 0 or scores.sum() == len(scores):
            continue
        try:
            auc = average_precision_score(train_df["target"].values, scores)
        except ValueError:
            continue
        if auc > best_auc:
            best_auc, best_t = auc, t
    return float(best_t)


def predict_artifact_prevalence_full(
    clf: object, feature_cols: list[str], train_medians: dict[str, float],
    threshold: float,
) -> None:
    """Report predicted artifact prevalence over ALL 3.77M beats in the subset."""
    beat_feat = pd.read_parquet(PROCESSED / "beat_features.parquet")
    if beat_feat.index.name == "peak_id":
        beat_feat = beat_feat.reset_index()

    X = _impute(beat_feat[feature_cols], train_medians)
    proba = clf.predict_proba(X)[:, 1]
    predicted_artifact = int((proba >= threshold).sum())
    pct = 100.0 * predicted_artifact / len(proba)
    print(
        f"\nPredicted artifact prevalence over full subset "
        f"({len(proba):,} beats @ threshold={threshold:.4f}): "
        f"{predicted_artifact:,} ({pct:.2f}%)"
    )
    if pct < 0.1:
        print("  WARNING: Prevalence < 0.1% — model may be suppressing positives.")
    elif pct > 10.0:
        print("  WARNING: Prevalence > 10% — model may be over-predicting artifacts.")
    else:
        print("  OK: Prevalence in reasonable range (0.1%–10%)")


def main() -> None:
    sys.path.insert(0, "Scripts")

    print("Loading train/val reviewed subsets with segment features...")
    train_df, val_df = _load_reviewed_with_seg_features(val_fraction=0.2)

    y_val = val_df["target"].values
    prevalence = float(y_val.mean())
    n_total = len(y_val)
    n_artifact = int(y_val.sum())

    print(
        f"\nValidation reviewed subset: {n_total:,} beats, "
        f"{n_artifact} artifact ({prevalence * 100:.2f}%)"
    )
    print(
        f"Training reviewed subset:   {len(train_df):,} beats, "
        f"{int(train_df['target'].sum())} artifact "
        f"({100. * train_df['target'].mean():.2f}%)"
    )

    # ── Baseline 1: Random ────────────────────────────────────────────────
    # PR-AUC of a random scorer equals the positive rate in the eval set
    random_pr_auc = prevalence
    random_roc_auc = 0.5

    # ── Baseline 2: global_corr_clean — SKIPPED ───────────────────────────
    b2_pr_auc = float("nan")
    b2_roc_auc = float("nan")
    print(
        "\nBaseline 2 (global_corr_clean): SKIPPED "
        "— feature absent from beat_features schema"
    )

    # ── Baseline 3: Rule-based ────────────────────────────────────────────
    print("Tuning spectral entropy threshold on training split...")
    se_thresh = tune_spectral_entropy_threshold(train_df)
    print(f"  Best spectral_entropy threshold: {se_thresh:.3f}")

    rule_scores = (
        (val_df["segment_zcr"] > 0.5)
        | (val_df["segment_qrs_density"] < 0.3)
        | (val_df["segment_spectral_entropy"] > se_thresh)
    ).astype(float).values

    flagged = int(rule_scores.sum())
    print(
        f"  Rule flags {flagged}/{n_total} beats ({100. * flagged / max(n_total, 1):.1f}%)"
    )

    if rule_scores.sum() > 0 and y_val.sum() > 0:
        rule_pr_auc = float(average_precision_score(y_val, rule_scores))
        rule_roc_auc = float(roc_auc_score(y_val, rule_scores))
    else:
        rule_pr_auc = float("nan")
        rule_roc_auc = float("nan")
        print(
            "  NOTE: Rule flags 0 beats on val set — "
            "segment features for these segments are near-median (imputed). "
            "Rule-based PR-AUC is undefined."
        )

    # ── Stage 1 LGBM predictions ──────────────────────────────────────────
    model_path = MODELS / "smoke_test_beat_tabular.joblib"
    beat_feat = pd.read_parquet(PROCESSED / "beat_features.parquet")
    if beat_feat.index.name == "peak_id":
        beat_feat = beat_feat.reset_index()

    artifact = joblib.load(model_path)
    clf = artifact["model"]
    feature_cols = artifact["feature_columns"]
    train_medians = artifact["train_medians"]
    opt_threshold = artifact["optimal_threshold"]

    val_peak_ids = set(val_df["peak_id"].values)
    val_feat = beat_feat[beat_feat["peak_id"].isin(val_peak_ids)].copy()
    val_feat_imp = _impute(val_feat[["peak_id"] + feature_cols], train_medians)

    proba = clf.predict_proba(val_feat_imp[feature_cols])[:, 1]
    prob_series = pd.Series(proba, index=val_feat["peak_id"].values)
    y_lgbm = prob_series.reindex(val_df["peak_id"].values).values

    valid_mask = ~np.isnan(y_lgbm)
    if valid_mask.sum() > 0 and y_val[valid_mask].sum() > 0:
        lgbm_pr_auc = float(
            average_precision_score(y_val[valid_mask], y_lgbm[valid_mask])
        )
        lgbm_roc_auc = float(roc_auc_score(y_val[valid_mask], y_lgbm[valid_mask]))
    else:
        lgbm_pr_auc = float("nan")
        lgbm_roc_auc = float("nan")

    # ── Comparison table ──────────────────────────────────────────────────
    rows = [
        ("random baseline",              random_pr_auc,  random_roc_auc),
        ("rule-based on new features",   rule_pr_auc,    rule_roc_auc),
        ("global_corr_clean threshold",  b2_pr_auc,      b2_roc_auc),
        ("smoke_test beat tabular LGBM", lgbm_pr_auc,    lgbm_roc_auc),
    ]

    print(f"\n{'Model':<32} {'Val PR-AUC':>12} {'Val ROC-AUC':>12}")
    print("-" * 58)
    for name, pr, roc in rows:
        pr_str  = f"{pr:.4f}"  if not np.isnan(pr)  else "       n/a"
        roc_str = f"{roc:.4f}" if not np.isnan(roc) else "       n/a"
        print(f"{name:<32} {pr_str:>12} {roc_str:>12}")
    print("-" * 58)

    # ── Full-subset predicted prevalence (verification b) ─────────────────
    predict_artifact_prevalence_full(clf, feature_cols, train_medians, opt_threshold)

    # ── Recommendation ────────────────────────────────────────────────────
    print("\n=== RECOMMENDATION ===")
    if not np.isnan(lgbm_pr_auc) and not np.isnan(rule_pr_auc):
        delta = lgbm_pr_auc - rule_pr_auc
        if delta > 0.05:
            print(
                f"LGBM beats rule-based by +{delta:.4f} PR-AUC. "
                "Stage 1 is healthy → proceed to full dataset (Prompt D)."
            )
        elif delta > 0.0:
            print(
                f"LGBM marginally beats rule-based by +{delta:.4f} PR-AUC. "
                "Consider whether added complexity is worth it before scaling."
            )
        else:
            print(
                f"Rule-based matches or outperforms LGBM (delta={delta:+.4f}). "
                "Consider replacing Stage 1 with the rule-based detector — "
                "simpler, no retraining required, equivalent or better performance."
            )
    elif not np.isnan(lgbm_pr_auc) and np.isnan(rule_pr_auc):
        delta = lgbm_pr_auc - random_pr_auc
        print(
            f"Rule-based baseline is undefined (flagged 0 val beats after "
            f"NaN-fill imputation — val segments are all within normal feature range). "
            f"LGBM beats random baseline by +{delta:.4f} PR-AUC "
            f"({lgbm_pr_auc:.4f} vs {random_pr_auc:.4f}). "
        )
        if delta > 0.05:
            print(
                "LGBM provides meaningful lift over random → "
                "proceed to full dataset (Prompt D)."
            )
        else:
            print(
                "LGBM provides minimal lift over random. "
                "Investigate model quality before scaling to full dataset."
            )
    else:
        print("Insufficient data (NaN metrics). Check val set size and artifact count.")


if __name__ == "__main__":
    main()
