#!/usr/bin/env python3
"""
ecgclean/models/beat_artifact_tabular.py — Stage 1: Beat-Level Artifact Classifier

LightGBM binary classifier that assigns each R-peak an artifact probability
based on the extended beat feature matrix.  This is the primary beat-level
model in the pipeline; its output (``p_artifact_tabular``) feeds directly
into the ensemble and active learning sampler.

Label convention:
    label == "artifact"  → 1 (positive class)
    all other labels     → 0 (negative class)

Training notes:
    * Severe class imbalance (~1.3 % positive).  ``scale_pos_weight`` is set
      to ``n_negative / n_positive`` to compensate.
    * Temporal split by ``segment_idx`` — all beats from earlier segments go
      to train, later segments to val.  Random beat-level splits are
      **never** used because adjacent beats share autocorrelated physiology
      and electrode-contact quality.
    * Hard-filtered beats (caught by deterministic rules) are excluded from
      training but kept in the evaluation set to verify model agreement.
    * NaN imputation uses column medians from the training set only; medians
      are stored in the model artifact for identical treatment at inference.
    * PR-AUC on the artifact class is the primary evaluation metric.

Usage:
    # Train
    python ecgclean/models/beat_artifact_tabular.py train \\
        --beat-features data/processed/beat_features.parquet \\
        --labels data/processed/labels.parquet \\
        --segment-quality-preds data/processed/segment_quality_preds.parquet \\
        --output models/beat_tabular_v1.joblib

    # Predict
    python ecgclean/models/beat_artifact_tabular.py predict \\
        --beat-features data/processed/beat_features.parquet \\
        --model models/beat_tabular_v1.joblib \\
        --output data/processed/beat_tabular_preds.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LGBM_N_ESTIMATORS_BEAT,
    LGBM_LEARNING_RATE,
    LGBM_NUM_LEAVES_BEAT,
    LGBM_RANDOM_STATE,
    VAL_FRACTION,
    BEAT_BATCH_SIZE,
)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ecgclean.models.beat_artifact_tabular")

# ── Default LightGBM parameters ──────────────────────────────────────────────

# Features computed from existing artifact labels that cause training leakage.
# They encode the current label distribution directly, so the model learns to
# regurgitate label counts rather than detect artifacts from beat morphology.
# Excluded from training; stored feature_columns are kept consistent so
# prediction never sees them either (both sides of the model see the same set).
_LABEL_LEAKAGE_FEATURES: frozenset[str] = frozenset({
    "segment_artifact_fraction",
})

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "n_estimators": LGBM_N_ESTIMATORS_BEAT,
    "learning_rate": LGBM_LEARNING_RATE,
    "num_leaves": LGBM_NUM_LEAVES_BEAT,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "random_state": LGBM_RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_column_medians(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-column medians for NaN imputation.

    Only numeric columns are included.  Columns that are entirely NaN get
    a median of 0.0 (safe fallback — LightGBM can handle it, and a fully
    NaN column carries no signal anyway).

    Args:
        df: Feature DataFrame (may contain NaN).

    Returns:
        Dict mapping column name → median value (float).
    """
    medians: dict[str, float] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = df[col].median()
            medians[col] = float(med) if not np.isnan(med) else 0.0
    return medians


def _apply_median_imputation(
    df: pd.DataFrame,
    medians: dict[str, float],
) -> pd.DataFrame:
    """Fill NaN values using pre-computed column medians.

    Columns present in *medians* but missing from *df* are silently skipped.
    Columns present in *df* but absent from *medians* are filled with 0.0.

    Args:
        df: Feature DataFrame (may contain NaN).
        medians: Dict of column → median from the training set.

    Returns:
        Copy of *df* with NaN values replaced.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            fill_val = medians.get(col, 0.0)
            df[col] = df[col].fillna(fill_val)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD AND EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute precision, recall, and F1 at a given probability threshold.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        threshold: Classification threshold.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Compute calibration in *n_bins* equal-width probability bins.

    For each bin, reports the mean predicted probability, actual positive
    rate, and count of samples.

    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of equal-width bins (default 10).

    Returns:
        List of dicts, one per bin, with keys ``bin_range``, ``predicted``,
        ``actual``, and ``n``.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    calibration: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n > 0:
            mean_pred = float(y_prob[mask].mean())
            actual_rate = float(y_true[mask].mean())
        else:
            mean_pred = 0.0
            actual_rate = 0.0
        right_bracket = "]" if i == n_bins - 1 else ")"
        calibration.append(
            {
                "bin_range": f"[{lo:.1f}, {hi:.1f}{right_bracket}",
                "predicted": mean_pred,
                "actual": actual_rate,
                "n": n,
            }
        )
    return calibration


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train(
    beat_features_path: str,
    labels_path: str,
    segment_quality_preds_path: str,
    output_model_path: str,
    val_fraction: float = 0.2,
    exclude_bad_segments: bool = True,
    random_seed: int = 42,
) -> dict:
    """Train a LightGBM binary classifier for beat-level artifact detection.

    Loads the extended beat feature matrix, labels, and segment quality
    predictions, then trains a binary classifier (artifact vs clean).
    The model uses a temporal split by ``segment_idx``, early stopping,
    and PR-curve-based threshold tuning.

    The trained model, feature columns, imputation medians, and thresholds
    are saved as a single joblib artifact.

    Args:
        beat_features_path: Path to ``beat_features.parquet`` (indexed by
            peak_id, 32 feature columns).
        labels_path: Path to ``labels.parquet`` (contains peak_id, label,
            and hard_filtered columns).
        segment_quality_preds_path: Path to
            ``segment_quality_preds.parquet`` (contains segment_idx and
            quality_label columns).
        output_model_path: Destination for the joblib-serialised model
            artifact.
        val_fraction: Fraction of *segments* (by temporal order) to hold
            out for validation.  Default 0.2.
        exclude_bad_segments: If True, drop all beats from segments
            predicted ``bad`` by the Stage 0 classifier.  Default True.
        random_seed: Random seed for LightGBM reproducibility.

    Returns:
        Dict containing validation metrics: PR-AUC, ROC-AUC, per-threshold
        precision/recall/F1, confusion matrix, calibration, and top-20
        feature importances.

    Raises:
        FileNotFoundError: If any required input file is missing.
        AssertionError: If the temporal split invariant is violated.
    """
    # ── Load data ─────────────────────────────────────────────────────────
    bf_path = Path(beat_features_path)
    lb_path = Path(labels_path)
    sq_path = Path(segment_quality_preds_path)

    for fpath, name in [
        (bf_path, "Beat features"),
        (lb_path, "Labels"),
        (sq_path, "Segment quality preds"),
    ]:
        if not fpath.exists():
            raise FileNotFoundError(f"{name} not found: {fpath}")

    # peaks.parquet is in the same directory — provides peak_id → segment_idx
    peaks_path = bf_path.parent / "peaks.parquet"
    if not peaks_path.exists():
        raise FileNotFoundError(
            f"peaks.parquet not found at {peaks_path} — needed for "
            "segment_idx mapping.  It should be in the same directory as "
            "beat_features.parquet."
        )

    features_df = pd.read_parquet(bf_path)
    labels_df = pd.read_parquet(lb_path)
    seg_preds_df = pd.read_parquet(sq_path)
    peaks_df = pd.read_parquet(peaks_path)

    logger.info(
        "Loaded %d beat features, %d labels, %d segment preds, %d peaks",
        len(features_df),
        len(labels_df),
        len(seg_preds_df),
        len(peaks_df),
    )

    # ── Identify feature columns (before any joins) ───────────────────────
    if features_df.index.name == "peak_id":
        features_df = features_df.reset_index()
    feature_cols = [
        c for c in features_df.columns
        if c != "peak_id" and c not in _LABEL_LEAKAGE_FEATURES
    ]
    excluded_leaky = [c for c in features_df.columns if c in _LABEL_LEAKAGE_FEATURES]
    if excluded_leaky:
        logger.info(
            "Excluded label-leakage features from training: %s", excluded_leaky
        )
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # ── Merge: features + labels + peaks + segment quality ────────────────
    # Only select needed columns from each table to avoid name collisions
    label_cols = ["peak_id", "label"]
    if "hard_filtered" in labels_df.columns:
        label_cols.append("hard_filtered")
    if "reviewed" in labels_df.columns:
        label_cols.append("reviewed")
    if "in_bad_region" in labels_df.columns:
        label_cols.append("in_bad_region")
    merged = features_df.merge(
        labels_df[label_cols],
        on="peak_id",
        how="inner",
    )
    # hard_filtered may be absent when labels come from annotation sessions
    # (those CSVs only carry peak_id + label).  Default to False = include all.
    if "hard_filtered" not in merged.columns:
        merged["hard_filtered"] = False
    if "in_bad_region" not in merged.columns:
        merged["in_bad_region"] = False
    merged = merged.merge(
        peaks_df[["peak_id", "segment_idx"]].drop_duplicates(),
        on="peak_id",
        how="inner",
    )
    merged = merged.merge(
        seg_preds_df[["segment_idx", "quality_label"]].drop_duplicates(
            subset="segment_idx"
        ),
        on="segment_idx",
        how="left",  # keep beats even if their segment has no quality pred
    )
    # Missing quality predictions → treat as non-bad (not excluded)
    merged["quality_label"] = merged["quality_label"].fillna("clean")

    logger.info("Merged dataset: %d beats", len(merged))

    if len(merged) == 0:
        logger.error(
            "No beats matched across input files.  Check that peak_id "
            "values overlap between beat_features, labels, and peaks."
        )
        sys.exit(1)

    # ── Binary label ──────────────────────────────────────────────────────
    merged["target"] = (merged["label"] == "artifact").astype(int)

    n_artifact_total = int(merged["target"].sum())
    n_clean_total = len(merged) - n_artifact_total
    logger.info(
        "Label distribution: %d negative (%.1f%%), %d artifact (%.1f%%)",
        n_clean_total,
        100.0 * n_clean_total / max(len(merged), 1),
        n_artifact_total,
        100.0 * n_artifact_total / max(len(merged), 1),
    )

    # ── Exclude bad segments ──────────────────────────────────────────────
    # Only exclude non-artifact beats from bad segments. Unreviewed/clean-
    # labeled beats in bad segments can't be trusted as negatives. But
    # artifact-labeled beats in bad segments ARE trustworthy positive examples
    # and are rare — removing them halves the artifact training count.
    if exclude_bad_segments:
        bad_mask = merged["quality_label"] == "bad"
        excl_mask = bad_mask & (merged["target"] == 0)
        n_bad = int(excl_mask.sum())
        if n_bad > 0:
            n_bad_segments = int(merged.loc[bad_mask, "segment_idx"].nunique())
            n_artifacts_kept = int((bad_mask & (merged["target"] == 1)).sum())
            logger.info(
                "Excluding %d non-artifact beats from %d 'bad' segments "
                "(kept %d artifact-labeled beats from those segments)",
                n_bad,
                n_bad_segments,
                n_artifacts_kept,
            )
            merged = merged[~excl_mask].copy()

    # ── Exclude beats inside bad_region windows ───────────────────────────
    if "in_bad_region" in merged.columns:
        n_before = len(merged)
        merged = merged[~merged["in_bad_region"]].copy()
        n_excl = n_before - len(merged)
        if n_excl > 0:
            logger.info("Excluded %d beats inside bad_regions", n_excl)

    # ── Exclude interpolated beats ────────────────────────────────────────
    n_interp = int((merged["label"] == "interpolated").sum())
    if n_interp > 0:
        merged = merged[merged["label"] != "interpolated"].copy()
        logger.info("Excluded %d interpolated beats from training", n_interp)

    # ── Restrict to reviewed beats only ───────────────────────────────────
    # Only beats from manually annotated segments carry ground-truth labels.
    # Unreviewed beats have label='clean' by pipeline default but were never
    # inspected — using them as negative examples would corrupt the model.
    if "reviewed" in merged.columns:
        n_before = len(merged)
        merged = merged[merged["reviewed"]].copy()
        n_excluded = n_before - len(merged)
        if n_excluded > 0:
            logger.info(
                "Restricted to reviewed beats: %d excluded (unreviewed), %d remain",
                n_excluded,
                len(merged),
            )
    else:
        logger.warning(
            "No 'reviewed' column in labels — all beats used for training. "
            "Re-run data_pipeline.py to add reviewed labels."
        )

    # ── Temporal split by segment_idx ─────────────────────────────────────
    unique_segments = sorted(merged["segment_idx"].unique())
    n_segments = len(unique_segments)
    n_train_segments = max(1, int(n_segments * (1 - val_fraction)))

    train_segments = set(unique_segments[:n_train_segments])
    val_segments = set(unique_segments[n_train_segments:])

    all_train_df = merged[merged["segment_idx"].isin(train_segments)].copy()
    val_df = merged[merged["segment_idx"].isin(val_segments)].copy()

    # ── Assertion: temporal split integrity ───────────────────────────────
    if len(all_train_df) > 0 and len(val_df) > 0:
        max_train_seg = max(train_segments)
        min_val_seg = min(val_segments)
        assert max_train_seg < min_val_seg, (
            f"Temporal split violated: max train segment_idx ({max_train_seg}) "
            f"must be strictly less than min val segment_idx ({min_val_seg})"
        )

    logger.info(
        "Temporal split: %d train segments (%d beats), "
        "%d val segments (%d beats)",
        len(train_segments),
        len(all_train_df),
        len(val_segments),
        len(val_df),
    )

    # ── Exclude hard_filtered from TRAINING only (keep in val) ────────────
    train_df = all_train_df[~all_train_df["hard_filtered"]].copy()
    n_hard_filtered_excluded = len(all_train_df) - len(train_df)
    if n_hard_filtered_excluded > 0:
        logger.info(
            "Excluded %d hard_filtered beats from training set "
            "(kept in val for agreement verification)",
            n_hard_filtered_excluded,
        )

    # ── Warn on small artifact class ──────────────────────────────────────
    n_train_artifact = int(train_df["target"].sum())
    n_train_clean = len(train_df) - n_train_artifact

    if n_train_artifact < 100:
        logger.warning(
            "⚠ Artifact class has only %d training examples (< 100 recommended). "
            "Model performance will likely be poor.",
            n_train_artifact,
        )

    if n_train_artifact == 0:
        logger.warning(
            "⚠ No artifact examples in training set.  Model will produce "
            "only negative predictions.  Training proceeds for pipeline "
            "consistency."
        )

    # ── Class imbalance: scale_pos_weight ─────────────────────────────────
    if n_train_artifact > 0:
        scale_pos_weight = float(n_train_clean) / float(n_train_artifact)
    else:
        scale_pos_weight = 1.0
    logger.info("scale_pos_weight = %.2f", scale_pos_weight)

    # ── NaN imputation (train-only medians) ───────────────────────────────
    train_medians = _compute_column_medians(train_df[feature_cols])
    n_cols_with_nan = sum(1 for c in feature_cols if train_df[c].isna().any())
    logger.info(
        "Computed training medians for %d columns (%d had NaN)",
        len(train_medians),
        n_cols_with_nan,
    )

    X_train = _apply_median_imputation(train_df[feature_cols], train_medians)
    y_train = train_df["target"].values

    if len(val_df) > 0:
        X_val = _apply_median_imputation(val_df[feature_cols], train_medians)
        y_val = val_df["target"].values
    else:
        X_val = pd.DataFrame(columns=feature_cols)
        y_val = np.array([], dtype=int)

    # ── Train LightGBM ────────────────────────────────────────────────────
    params = DEFAULT_LGBM_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight
    params["random_state"] = random_seed

    # If validation set is single-class, AUC metric would fail — use only
    # binary_logloss for early stopping in that case
    val_has_both_classes = (
        len(y_val) > 0 and y_val.sum() > 0 and (y_val == 0).sum() > 0
    )
    if not val_has_both_classes and len(y_val) > 0:
        params["metric"] = "binary_logloss"
        logger.info(
            "Validation set has only one class — using binary_logloss "
            "only for early stopping (AUC requires both classes)"
        )

    logger.info(
        "Training LightGBM: %d estimators, early_stopping=50, lr=%.3f",
        params["n_estimators"],
        params["learning_rate"],
    )

    clf = lgb.LGBMClassifier(**params)

    if len(X_val) > 0:
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )
    else:
        clf.fit(X_train, y_train)

    best_iter = getattr(clf, "best_iteration_", None)
    logger.info(
        "Training complete.  Best iteration: %s",
        best_iter if best_iter is not None else "N/A (no early stopping)",
    )

    # ── Evaluation and threshold tuning ───────────────────────────────────
    metrics: dict[str, Any] = {}
    optimal_threshold = 0.5
    threshold_at_precision_90 = 0.5
    threshold_at_recall_90 = 0.5

    if len(X_val) > 0 and val_has_both_classes:
        proba_val = clf.predict_proba(X_val)[:, 1]

        # ── PR-AUC (PRIMARY METRIC) ──────────────────────────────────
        pr_auc = float(average_precision_score(y_val, proba_val))
        metrics["pr_auc"] = pr_auc

        # ── ROC-AUC ──────────────────────────────────────────────────
        roc_auc = float(roc_auc_score(y_val, proba_val))
        metrics["roc_auc"] = roc_auc

        # ── Precision-recall curve → threshold tuning ────────────────
        precision_arr, recall_arr, thresholds_arr = precision_recall_curve(
            y_val, proba_val
        )

        # Optimal F1 threshold (exclude sentinel point where recall=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_arr = np.where(
                (precision_arr[:-1] + recall_arr[:-1]) > 0,
                2.0
                * precision_arr[:-1]
                * recall_arr[:-1]
                / (precision_arr[:-1] + recall_arr[:-1]),
                0.0,
            )
        best_f1_idx = int(np.argmax(f1_arr))
        optimal_threshold = float(thresholds_arr[best_f1_idx])

        # Threshold for precision ≥ 0.90 (maximise recall subject to P ≥ 0.90)
        valid_p90 = precision_arr[:-1] >= 0.90
        if valid_p90.any():
            p90_indices = np.where(valid_p90)[0]
            # Among those meeting the precision constraint, pick highest recall
            best_p90 = p90_indices[int(np.argmax(recall_arr[:-1][p90_indices]))]
            threshold_at_precision_90 = float(thresholds_arr[best_p90])
        else:
            # No threshold achieves precision ≥ 0.90; use the highest threshold
            threshold_at_precision_90 = (
                float(thresholds_arr[-1]) if len(thresholds_arr) > 0 else 0.5
            )

        # Threshold for recall ≥ 0.90 (maximise precision subject to R ≥ 0.90)
        valid_r90 = recall_arr[:-1] >= 0.90
        if valid_r90.any():
            r90_indices = np.where(valid_r90)[0]
            best_r90 = r90_indices[
                int(np.argmax(precision_arr[:-1][r90_indices]))
            ]
            threshold_at_recall_90 = float(thresholds_arr[best_r90])
        else:
            # No threshold achieves recall ≥ 0.90; use the lowest threshold
            threshold_at_recall_90 = (
                float(thresholds_arr[0]) if len(thresholds_arr) > 0 else 0.5
            )

        # Metrics at each operating point
        metrics["at_optimal"] = _compute_metrics_at_threshold(
            y_val, proba_val, optimal_threshold
        )
        metrics["at_precision_90"] = _compute_metrics_at_threshold(
            y_val, proba_val, threshold_at_precision_90
        )
        metrics["at_recall_90"] = _compute_metrics_at_threshold(
            y_val, proba_val, threshold_at_recall_90
        )

        # Confusion matrix at optimal threshold
        y_pred_opt = (proba_val >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_val, y_pred_opt, labels=[0, 1])
        metrics["confusion_matrix"] = cm.tolist()

        # Calibration (10 equal-width bins)
        metrics["calibration"] = _compute_calibration(y_val, proba_val)

    elif len(X_val) > 0:
        # Validation set exists but has only one class — most metrics
        # cannot be computed meaningfully
        proba_val = clf.predict_proba(X_val)[:, 1]
        logger.warning(
            "Validation set has only one class — PR-AUC and threshold "
            "tuning skipped.  Defaults used (threshold=0.5)."
        )
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")
        metrics["at_optimal"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        metrics["at_precision_90"] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        metrics["at_recall_90"] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        metrics["confusion_matrix"] = []
        metrics["calibration"] = _compute_calibration(y_val, proba_val)
    else:
        logger.warning("No validation set — skipping all evaluation metrics")
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")
        metrics["at_optimal"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        metrics["at_precision_90"] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        metrics["at_recall_90"] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        metrics["confusion_matrix"] = []
        metrics["calibration"] = []

    # ── Feature importances (by gain) ─────────────────────────────────────
    gain_importances = clf.booster_.feature_importance(importance_type="gain")
    importance_pairs = sorted(
        zip(feature_cols, gain_importances.tolist()),
        key=lambda x: -x[1],
    )
    top_20 = importance_pairs[:20]
    metrics["top_features"] = [
        {"feature": name, "gain": imp} for name, imp in top_20
    ]

    # ── Save model artifact ───────────────────────────────────────────────
    trained_at = datetime.now(timezone.utc).isoformat()

    artifact: dict[str, Any] = {
        "model": clf,
        "feature_columns": feature_cols,
        "train_medians": train_medians,
        "optimal_threshold": optimal_threshold,
        "threshold_at_precision_90": threshold_at_precision_90,
        "threshold_at_recall_90": threshold_at_recall_90,
        "scale_pos_weight": scale_pos_weight,
        "val_metrics": metrics,
        "trained_at": trained_at,
    }

    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    logger.info("Saved model artifact → %s", out_path)

    # ── Print training summary ────────────────────────────────────────────
    n_val_artifact = int(y_val.sum()) if len(y_val) > 0 else 0
    n_val_clean = int((y_val == 0).sum()) if len(y_val) > 0 else 0

    _print_training_summary(
        n_total=len(merged),
        n_train=len(train_df),
        n_val=len(val_df),
        n_hard_filtered_excluded=n_hard_filtered_excluded,
        n_train_artifact=n_train_artifact,
        n_train_clean=n_train_clean,
        n_val_artifact=n_val_artifact,
        n_val_clean=n_val_clean,
        scale_pos_weight=scale_pos_weight,
        optimal_threshold=optimal_threshold,
        threshold_at_precision_90=threshold_at_precision_90,
        threshold_at_recall_90=threshold_at_recall_90,
        metrics=metrics,
        trained_at=trained_at,
    )

    return metrics


def _print_training_summary(
    n_total: int,
    n_train: int,
    n_val: int,
    n_hard_filtered_excluded: int,
    n_train_artifact: int,
    n_train_clean: int,
    n_val_artifact: int,
    n_val_clean: int,
    scale_pos_weight: float,
    optimal_threshold: float,
    threshold_at_precision_90: float,
    threshold_at_recall_90: float,
    metrics: dict[str, Any],
    trained_at: str,
) -> None:
    """Print a formatted summary of the training run.

    Prominently displays PR-AUC, threshold operating points, calibration,
    and feature importances.

    Args:
        n_total: Total beats after filtering.
        n_train: Training set size (after hard_filtered exclusion).
        n_val: Validation set size.
        n_hard_filtered_excluded: Beats removed from training due to
            hard_filtered flag.
        n_train_artifact: Artifact count in training set.
        n_train_clean: Clean count in training set.
        n_val_artifact: Artifact count in validation set.
        n_val_clean: Clean count in validation set.
        scale_pos_weight: Class imbalance weight used.
        optimal_threshold: Threshold maximising F1.
        threshold_at_precision_90: Threshold for P ≥ 0.90.
        threshold_at_recall_90: Threshold for R ≥ 0.90.
        metrics: Evaluation metrics dict.
        trained_at: ISO timestamp.
    """
    print(f"\n{'=' * 72}")
    print("  Stage 1 — Beat Artifact Tabular Classifier: Training Summary")
    print(f"{'=' * 72}")
    print(f"  Trained at: {trained_at}")
    print(f"  Total beats (after segment filtering): {n_total:,}")
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}  (temporal split by segment_idx)")
    if n_hard_filtered_excluded > 0:
        print(f"  Hard-filtered excluded from train: {n_hard_filtered_excluded:,}")

    print(f"\n  Training class distribution:")
    print(f"    clean/negative:  {n_train_clean:>8,}")
    print(f"    artifact/pos:    {n_train_artifact:>8,}")
    print(f"  scale_pos_weight:  {scale_pos_weight:.2f}")

    print(f"\n  Validation class distribution:")
    print(f"    clean/negative:  {n_val_clean:>8,}")
    print(f"    artifact/pos:    {n_val_artifact:>8,}")

    # PR-AUC displayed prominently
    pr_auc = metrics.get("pr_auc", float("nan"))
    roc_auc = metrics.get("roc_auc", float("nan"))
    print()
    print(f"  ╔═══════════════════════════════════════════╗")
    print(f"  ║  PR-AUC (artifact class):  {pr_auc:>8.4f}       ║  ← PRIMARY METRIC")
    print(f"  ║  ROC-AUC:                  {roc_auc:>8.4f}       ║")
    print(f"  ╚═══════════════════════════════════════════╝")

    # Threshold operating points
    print(f"\n  Threshold Operating Points:")
    for name, key, threshold in [
        ("Optimal F1", "at_optimal", optimal_threshold),
        ("Precision ≥ 0.90", "at_precision_90", threshold_at_precision_90),
        ("Recall ≥ 0.90", "at_recall_90", threshold_at_recall_90),
    ]:
        m = metrics.get(key, {})
        print(
            f"    {name:20s}: threshold={threshold:.4f}  "
            f"P={m.get('precision', 0):.4f}  R={m.get('recall', 0):.4f}  "
            f"F1={m.get('f1', 0):.4f}"
        )

    # Confusion matrix at optimal threshold
    cm = metrics.get("confusion_matrix", [])
    if cm:
        print(f"\n  Confusion Matrix at optimal threshold={optimal_threshold:.4f}:")
        print(f"    {'':>12s}  {'pred_neg':>10s}  {'pred_pos':>10s}")
        row_labels = ["actual_neg", "actual_pos"]
        for i, label in enumerate(row_labels):
            if i < len(cm):
                vals = "  ".join(f"{v:>10d}" for v in cm[i])
                print(f"    {label:>12s}  {vals}")

    # Calibration table
    calibration = metrics.get("calibration", [])
    if calibration:
        print(f"\n  Calibration (10 equal-width probability bins):")
        for entry in calibration:
            print(
                f"    {entry['bin_range']:>12s}: "
                f"predicted={entry['predicted']:.4f}, "
                f"actual={entry['actual']:.4f}, "
                f"n={entry['n']:>6,}"
            )

    # Feature importances — top 20 by gain
    top_features = metrics.get("top_features", [])
    if top_features:
        print(f"\n  Top {len(top_features)} Features (by gain):")
        for rank, entry in enumerate(top_features, 1):
            print(f"    {rank:>2d}. {entry['feature']:35s}  {entry['gain']:>12.1f}")

    print(f"{'=' * 72}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


def predict(
    beat_features: pd.DataFrame,
    model_path: str,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Run inference on beat features using a trained artifact classifier.

    Loads the model artifact, applies the stored median imputation, and
    returns per-beat artifact probabilities and binary predictions.

    If *threshold* is ``None``, the stored ``optimal_threshold`` (tuned on
    the validation set during training) is used.

    Args:
        beat_features: Beat feature DataFrame, indexed by ``peak_id``
            (or containing it as a column).  Must contain all feature
            columns the model was trained on.
        model_path: Path to the joblib-serialised model artifact produced
            by :func:`train`.
        threshold: Classification threshold.  If None, uses the stored
            optimal threshold.

    Returns:
        DataFrame with columns:
        - ``peak_id`` (int64)
        - ``p_artifact_tabular`` (float32)
        - ``predicted_artifact`` (bool)

    Raises:
        FileNotFoundError: If model_path does not exist.
        ValueError: If required feature columns are missing.
    """
    mpath = Path(model_path)
    if not mpath.exists():
        raise FileNotFoundError(f"Model not found: {mpath}")

    artifact = joblib.load(mpath)

    clf = artifact["model"]
    feature_cols: list[str] = artifact["feature_columns"]
    train_medians: dict[str, float] = artifact["train_medians"]
    stored_threshold: float = artifact["optimal_threshold"]
    trained_at: str = artifact.get("trained_at", "unknown")

    if threshold is None:
        threshold = stored_threshold

    logger.info(
        "Loaded model trained at %s (%d features, threshold=%.4f)",
        trained_at,
        len(feature_cols),
        threshold,
    )

    # ── Prepare features ──────────────────────────────────────────────────
    df = beat_features.copy()
    if df.index.name == "peak_id":
        df = df.reset_index()

    # Verify all required feature columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Feature mismatch: model expects columns {missing_cols} "
            f"which are missing from the input.  Model was trained at "
            f"{trained_at} with features: {feature_cols}"
        )

    peak_ids = df["peak_id"].values.astype(np.int64)

    # ── Impute and predict ────────────────────────────────────────────────
    X = _apply_median_imputation(df[feature_cols], train_medians)
    proba = clf.predict_proba(X)[:, 1].astype(np.float32)
    predicted = proba >= threshold

    result = pd.DataFrame(
        {
            "peak_id": peak_ids,
            "p_artifact_tabular": proba,
            "predicted_artifact": predicted,
        }
    )

    logger.info(
        "Predictions: %d beats — %d predicted artifact (%.2f%%)",
        len(result),
        int(predicted.sum()),
        100.0 * predicted.sum() / max(len(result), 1),
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# UNCERTAINTY SCORING
# ═══════════════════════════════════════════════════════════════════════════════


def get_uncertainty_scores(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Add an uncertainty column for active learning sampling.

    Uncertainty is defined as:

        uncertainty_tabular = 1 − 2 × |p_artifact_tabular − 0.5|

    This maps probabilities to [0, 1] where 1.0 = maximally uncertain
    (p = 0.5) and 0.0 = maximally confident (p = 0 or 1).

    The active learning sampler uses this score to prioritise beats where
    the model is least confident, directing human labelling effort where
    it will improve the model the most.

    Args:
        predictions_df: Output of :func:`predict`, must contain
            ``p_artifact_tabular`` column.

    Returns:
        Copy of *predictions_df* with ``uncertainty_tabular`` (float32)
        column appended.
    """
    df = predictions_df.copy()
    p = df["p_artifact_tabular"].values.astype(np.float64)
    df["uncertainty_tabular"] = (1.0 - 2.0 * np.abs(p - 0.5)).astype(
        np.float32
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _cli_train(args: argparse.Namespace) -> None:
    """Handle the ``train`` subcommand."""
    train(
        beat_features_path=args.beat_features,
        labels_path=args.labels,
        segment_quality_preds_path=args.segment_quality_preds,
        output_model_path=args.output,
        val_fraction=args.val_fraction,
        exclude_bad_segments=not args.no_exclude_bad_segments,
        random_seed=args.seed,
    )


def _cli_predict(args: argparse.Namespace) -> None:
    """Handle the ``predict`` subcommand — streams beat_features in batches.

    Loading the full beat_features_merged.parquet (54 M rows × 40+ cols)
    into RAM at once would consume 10-20 GB and crash the machine.  This
    implementation instead:
      1. Loads the model artifact (~MB).
      2. Opens beat_features.parquet for streaming via ParquetFile.
      3. Imputes and predicts each batch independently.
      4. Writes predictions incrementally via ParquetWriter.

    Peak RAM: model + one batch (~80 MB) + one output batch.
    """
    bf_path = Path(args.beat_features)
    if not bf_path.exists():
        logger.error("Beat features not found: %s", bf_path)
        sys.exit(1)

    # ── Load model artifact (small) ────────────────────────────────────────
    mpath = Path(args.model)
    if not mpath.exists():
        logger.error("Model not found: %s", mpath)
        sys.exit(1)

    artifact = joblib.load(mpath)
    clf = artifact["model"]
    feature_cols: list[str] = artifact["feature_columns"]
    train_medians: dict[str, float] = artifact["train_medians"]
    stored_threshold: float = artifact["optimal_threshold"]
    trained_at: str = artifact.get("trained_at", "unknown")

    threshold = args.threshold if args.threshold is not None else stored_threshold
    logger.info(
        "Loaded model trained at %s (%d features, threshold=%.4f)",
        trained_at,
        len(feature_cols),
        threshold,
    )

    # ── Inspect parquet schema without loading data ────────────────────────
    pf = pq.ParquetFile(bf_path)
    schema_names = set(pf.schema_arrow.names)
    n_total_rows = pf.metadata.num_rows

    missing_cols = [c for c in feature_cols if c not in schema_names]
    if missing_cols:
        raise ValueError(
            f"Feature mismatch: model expects {missing_cols} which are "
            f"missing from {bf_path}.  Model was trained at {trained_at} "
            f"with features: {feature_cols}"
        )

    # Only request the columns we actually need (peak_id + feature_cols)
    cols_to_read = list(
        dict.fromkeys(  # preserves order, deduplicates
            (["peak_id"] if "peak_id" in schema_names else []) + feature_cols
        )
    )

    logger.info(
        "Streaming tabular inference: %d rows, %d feature columns, "
        "batch_size=%d",
        n_total_rows,
        len(feature_cols),
        args.batch_size,
    )

    # ── Output setup ──────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_schema = pa.schema([
        pa.field("peak_id",             pa.int64()),
        pa.field("p_artifact_tabular",  pa.float32()),
        pa.field("predicted_artifact",  pa.bool_()),
        pa.field("uncertainty_tabular", pa.float32()),
    ])

    # Running stats (avoid accumulating all probabilities in RAM)
    total_beats = 0
    total_artifact = 0
    p_min = np.inf
    p_max = -np.inf
    p_sum = 0.0
    p_sq_sum = 0.0
    u_min = np.inf
    u_max = -np.inf
    u_sum = 0.0
    u_sq_sum = 0.0

    with pq.ParquetWriter(out_path, out_schema, compression="snappy") as writer:
        for batch in pf.iter_batches(
            batch_size=args.batch_size,
            columns=cols_to_read,
        ):
            df = batch.to_pandas()

            # Recover peak_id — iter_batches returns it as a column
            # (pandas index metadata is not applied to RecordBatch slices)
            if df.index.name == "peak_id":
                df = df.reset_index()
            if "peak_id" not in df.columns:
                logger.error(
                    "peak_id not found in batch (index=%r, columns=%s)",
                    df.index.name,
                    list(df.columns),
                )
                sys.exit(1)

            peak_ids = df["peak_id"].values.astype(np.int64)

            # Impute NaNs with training medians then predict
            X = _apply_median_imputation(df[feature_cols], train_medians)
            proba = clf.predict_proba(X)[:, 1].astype(np.float32)
            predicted = proba >= threshold

            # Uncertainty: 1 − 2|p − 0.5|  (0 = confident, 1 = maximally uncertain)
            p64 = proba.astype(np.float64)
            uncertainty = (1.0 - 2.0 * np.abs(p64 - 0.5)).astype(np.float32)

            # Accumulate stats
            total_beats += len(proba)
            total_artifact += int(predicted.sum())
            if len(proba) > 0:
                p_min = min(p_min, float(proba.min()))
                p_max = max(p_max, float(proba.max()))
                p_sum += float(proba.sum())
                p_sq_sum += float((p64 ** 2).sum())
                u_min = min(u_min, float(uncertainty.min()))
                u_max = max(u_max, float(uncertainty.max()))
                u_sum += float(uncertainty.sum())
                u_sq_sum += float((uncertainty.astype(np.float64) ** 2).sum())

            writer.write_table(
                pa.table(
                    {
                        "peak_id":             peak_ids,
                        "p_artifact_tabular":  proba,
                        "predicted_artifact":  predicted,
                        "uncertainty_tabular": uncertainty,
                    },
                    schema=out_schema,
                )
            )

            logger.info(
                "  %d / %d rows  (%.0f%%)",
                total_beats,
                n_total_rows,
                100.0 * total_beats / max(n_total_rows, 1),
            )

    logger.info("Saved predictions → %s", out_path)

    # ── Print summary ──────────────────────────────────────────────────────
    n_clean = total_beats - total_artifact
    pct_artifact = 100.0 * total_artifact / max(total_beats, 1)

    p_mean = p_sum / max(total_beats, 1)
    p_std = float(np.sqrt(max(0.0, p_sq_sum / max(total_beats, 1) - p_mean ** 2)))
    u_mean = u_sum / max(total_beats, 1)
    u_std = float(np.sqrt(max(0.0, u_sq_sum / max(total_beats, 1) - u_mean ** 2)))

    print(f"\n{'=' * 72}")
    print("  Beat Artifact Tabular Predictions")
    print(f"{'=' * 72}")
    print(f"  Total beats: {total_beats:,}")
    print(f"\n  Prediction distribution:")
    print(f"    clean (predicted):    {n_clean:>8,}  ({100.0 - pct_artifact:5.1f}%)")
    print(f"    artifact (predicted): {total_artifact:>8,}  ({pct_artifact:5.1f}%)")
    print(f"\n  Probability statistics (p_artifact_tabular):")
    print(
        f"    mean={p_mean:.4f}  std={p_std:.4f}  "
        f"min={p_min:.4f}  max={p_max:.4f}"
    )
    print(f"\n  Uncertainty statistics (uncertainty_tabular):")
    print(
        f"    mean={u_mean:.4f}  std={u_std:.4f}  "
        f"min={u_min:.4f}  max={u_max:.4f}"
    )
    print(f"\n  Output: {out_path}")
    print(f"{'=' * 72}\n")


def main() -> None:
    """CLI entry point with ``train`` and ``predict`` subcommands."""
    parser = argparse.ArgumentParser(
        description=(
            "ECG Artifact Pipeline — Stage 1: Beat-Level Artifact Classifier"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ─────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser(
        "train", help="Train the beat-level artifact classifier"
    )
    train_parser.add_argument(
        "--beat-features",
        type=str,
        required=True,
        help="Path to beat_features.parquet",
    )
    train_parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels.parquet (contains label and hard_filtered)",
    )
    train_parser.add_argument(
        "--segment-quality-preds",
        type=str,
        required=True,
        help="Path to segment_quality_preds.parquet",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the model artifact (.joblib)",
    )
    train_parser.add_argument(
        "--val-fraction",
        type=float,
        default=VAL_FRACTION,
        help=f"Fraction of segments for temporal validation (default: {VAL_FRACTION})",
    )
    train_parser.add_argument(
        "--no-exclude-bad-segments",
        action="store_true",
        default=False,
        help="Keep beats from 'bad' segments in training (default: exclude)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=LGBM_RANDOM_STATE,
        help=f"Random seed for LightGBM (default: {LGBM_RANDOM_STATE})",
    )

    # ── predict ───────────────────────────────────────────────────────────
    predict_parser = subparsers.add_parser(
        "predict", help="Run inference on beat features"
    )
    predict_parser.add_argument(
        "--beat-features",
        type=str,
        required=True,
        help="Path to beat_features.parquet",
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model artifact (.joblib)",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for predictions (.parquet)",
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (default: use stored optimal)",
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=BEAT_BATCH_SIZE,
        help=(
            f"Rows per streaming batch (default: {BEAT_BATCH_SIZE}). "
            "Decrease if RAM is tight."
        ),
    )

    args = parser.parse_args()

    if args.command == "train":
        _cli_train(args)
    elif args.command == "predict":
        _cli_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
