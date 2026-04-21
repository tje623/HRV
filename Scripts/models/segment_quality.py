#!/usr/bin/env python3
"""
ecgclean/models/segment_quality.py — Stage 0: Segment Quality Classifier

LightGBM multiclass classifier that gates segments into three quality tiers
before any beat-level model runs.  This is the first filter in the pipeline:
segments predicted as ``bad`` are excluded from all downstream training and
inference; segments predicted as ``noisy_ok`` may be included with relaxed
thresholds.

Label encoding:
    clean    → 0
    noisy_ok → 1
    bad      → 2

The model is trained with a *temporal* split (earlier segments → train,
later segments → validate) because adjacent 60-second windows share
autocorrelated physiological state and electrode-contact quality.  A random
split would leak this structure and produce optimistically biased metrics.

NaN imputation uses column medians computed from the training set only and
stored alongside the model for identical treatment at inference time.

Usage:
    # Train
    python ecgclean/models/segment_quality.py train \\
        --segment-features data/processed/segment_features.parquet \\
        --segments data/processed/segments.parquet \\
        --output models/segment_quality_v1.joblib

    # Predict
    python ecgclean/models/segment_quality.py predict \\
        --segment-features data/processed/segment_features.parquet \\
        --model models/segment_quality_v1.joblib \\
        --output data/processed/segment_quality_preds.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LGBM_N_ESTIMATORS_SEG,
    LGBM_LEARNING_RATE,
    LGBM_NUM_LEAVES_SEG,
    LGBM_RANDOM_STATE,
    VAL_FRACTION,
)

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ecgclean.models.segment_quality")

# ── Label mapping ─────────────────────────────────────────────────────────────

LABEL_ENCODER: dict[str, int] = {"clean": 0, "noisy_ok": 1, "bad": 2}
LABEL_DECODER: dict[int, str] = {v: k for k, v in LABEL_ENCODER.items()}

# ── Default LightGBM parameters ──────────────────────────────────────────────

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": LGBM_N_ESTIMATORS_SEG,
    "learning_rate": LGBM_LEARNING_RATE,
    "num_leaves": LGBM_NUM_LEAVES_SEG,
    "min_child_samples": 5,
    "class_weight": "balanced",
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
    import warnings
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
        Copy of df with NaN values replaced.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            fill_val = medians.get(col, 0.0)
            df[col] = df[col].fillna(fill_val)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train(
    segment_features_path: str,
    segments_labels_path: str,
    output_model_path: str,
    val_fraction: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train a LightGBM multiclass classifier for segment quality.

    Uses a temporal split: segments are sorted by ``segment_idx`` (which
    increases monotonically with recording time), and the first
    ``1 - val_fraction`` fraction becomes the training set.  This mirrors
    real deployment where the model must generalise to future segments.

    NaN values are imputed with column medians computed from the training
    split only.  The medians and feature column list are saved alongside
    the model for reproducible inference.

    Args:
        segment_features_path: Path to segment_features.parquet (indexed
            by segment_idx, 22 feature columns).
        segments_labels_path: Path to segments.parquet (contains
            segment_idx and quality_label columns).
        output_model_path: Destination for the joblib-serialised model
            artifact.
        val_fraction: Fraction of segments (by temporal order) to hold
            out for validation.  Default 0.2.
        random_seed: Random seed for LightGBM.  Does NOT affect the
            train/val split, which is purely temporal.

    Returns:
        Dict containing validation metrics: per-class precision/recall/F1,
        macro F1, confusion matrix, and top-15 feature importances.

    Raises:
        FileNotFoundError: If either input file is missing.
        AssertionError: Internal consistency checks.
    """
    # ── Load data ─────────────────────────────────────────────────────────
    sf_path = Path(segment_features_path)
    seg_path = Path(segments_labels_path)

    if not sf_path.exists():
        raise FileNotFoundError(f"Segment features not found: {sf_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Segments labels not found: {seg_path}")

    features_df = pd.read_parquet(sf_path)
    segments_df = pd.read_parquet(seg_path)

    logger.info(
        "Loaded %d segment features, %d segment labels",
        len(features_df), len(segments_df),
    )

    # ── Merge features with labels ────────────────────────────────────────
    # features_df is indexed by segment_idx; segments_df has segment_idx column
    if features_df.index.name == "segment_idx":
        features_df = features_df.reset_index()

    merged = features_df.merge(
        segments_df[["segment_idx", "quality_label"]],
        on="segment_idx",
        how="inner",
    )

    if len(merged) == 0:
        logger.error(
            "No segments matched between features (%d) and labels (%d). "
            "Check that segment_idx values overlap.",
            len(features_df), len(segments_df),
        )
        sys.exit(1)

    logger.info("Merged dataset: %d segments with both features and labels", len(merged))

    # Encode labels
    merged["label_int"] = merged["quality_label"].map(LABEL_ENCODER)
    unknown_labels = merged["quality_label"][merged["label_int"].isna()].unique()
    if len(unknown_labels) > 0:
        logger.warning("Unknown quality labels dropped: %s", unknown_labels)
        merged = merged.dropna(subset=["label_int"])
    merged["label_int"] = merged["label_int"].astype(int)

    # ── Temporal split ────────────────────────────────────────────────────
    merged = merged.sort_values("segment_idx").reset_index(drop=True)
    n_total = len(merged)
    n_train = max(1, int(n_total * (1 - val_fraction)))

    train_df = merged.iloc[:n_train].copy()
    val_df = merged.iloc[n_train:].copy()

    logger.info(
        "Temporal split: %d train (seg_idx ≤ %d), %d val (seg_idx ≥ %d)",
        len(train_df),
        int(train_df["segment_idx"].max()) if len(train_df) > 0 else -1,
        len(val_df),
        int(val_df["segment_idx"].min()) if len(val_df) > 0 else -1,
    )

    # Warn on small class counts
    for label_name, label_int in LABEL_ENCODER.items():
        train_count = int((train_df["label_int"] == label_int).sum())
        if train_count < 10:
            logger.warning(
                "Training set has only %d segments of class '%s' "
                "(< 10 recommended). Model may underperform on this class.",
                train_count, label_name,
            )

    # ── Identify feature columns ──────────────────────────────────────────
    meta_cols = {"segment_idx", "quality_label", "label_int"}
    feature_cols = [c for c in merged.columns if c not in meta_cols]
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # ── NaN imputation (train-only medians) ───────────────────────────────
    train_medians = _compute_column_medians(train_df[feature_cols])
    logger.info(
        "Computed training medians for %d columns (%d had NaN)",
        len(train_medians),
        sum(1 for c in feature_cols if train_df[c].isna().any()),
    )

    X_train = _apply_median_imputation(train_df[feature_cols], train_medians)
    y_train = train_df["label_int"].values

    if len(val_df) > 0:
        X_val = _apply_median_imputation(val_df[feature_cols], train_medians)
        y_val = val_df["label_int"].values
    else:
        X_val = pd.DataFrame(columns=feature_cols)
        y_val = np.array([], dtype=int)

    # ── Train LightGBM ────────────────────────────────────────────────────
    params = DEFAULT_LGBM_PARAMS.copy()
    params["random_state"] = random_seed

    logger.info("Training LightGBM with %d estimators...", params["n_estimators"])

    clf = lgb.LGBMClassifier(**params)

    if len(X_val) > 0:
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
        )
    else:
        clf.fit(X_train, y_train)

    logger.info("Training complete.")

    # ── Evaluation ────────────────────────────────────────────────────────
    metrics: dict[str, Any] = {}

    if len(X_val) > 0:
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, y_pred, labels=[0, 1, 2], zero_division=0.0,
        )
        per_class: dict[str, dict[str, float]] = {}
        for label_int, label_name in LABEL_DECODER.items():
            per_class[label_name] = {
                "precision": float(precision[label_int]),
                "recall": float(recall[label_int]),
                "f1": float(f1[label_int]),
                "support": int(support[label_int]),
            }
        metrics["per_class"] = per_class

        # Macro F1
        macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0.0))
        metrics["macro_f1"] = macro_f1

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report string (for printing)
        target_names = [LABEL_DECODER[i] for i in range(3)]
        report_str = classification_report(
            y_val, y_pred, labels=[0, 1, 2],
            target_names=target_names, zero_division=0.0,
        )
        metrics["classification_report"] = report_str
    else:
        logger.warning("No validation set — skipping evaluation metrics")
        metrics["per_class"] = {}
        metrics["macro_f1"] = float("nan")
        metrics["confusion_matrix"] = []
        metrics["classification_report"] = "(no validation set)"

    # Feature importances (by gain)
    importances = clf.feature_importances_
    importance_pairs = sorted(
        zip(feature_cols, importances.tolist()),
        key=lambda x: -x[1],
    )
    top_15 = importance_pairs[:15]
    metrics["top_features"] = [
        {"feature": name, "importance": imp} for name, imp in top_15
    ]

    # ── Save model artifact ───────────────────────────────────────────────
    trained_at = datetime.now(timezone.utc).isoformat()

    artifact = {
        "model": clf,
        "feature_columns": feature_cols,
        "label_encoder": LABEL_ENCODER.copy(),
        "train_medians": train_medians,
        "val_metrics": metrics,
        "trained_at": trained_at,
    }

    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    logger.info("Saved model artifact → %s", out_path)

    # ── Print training summary ────────────────────────────────────────────
    _print_training_summary(
        n_total, len(train_df), len(val_df),
        train_df["quality_label"].value_counts().to_dict(),
        val_df["quality_label"].value_counts().to_dict() if len(val_df) > 0 else {},
        metrics, trained_at,
    )

    return metrics


def _print_training_summary(
    n_total: int,
    n_train: int,
    n_val: int,
    train_dist: dict[str, int],
    val_dist: dict[str, int],
    metrics: dict[str, Any],
    trained_at: str,
) -> None:
    """Print a formatted summary of the training run.

    Args:
        n_total: Total number of segments.
        n_train: Training set size.
        n_val: Validation set size.
        train_dist: Class distribution in training set.
        val_dist: Class distribution in validation set.
        metrics: Evaluation metrics dict.
        trained_at: ISO timestamp of training completion.
    """
    print(f"\n{'=' * 70}")
    print("  Stage 0 — Segment Quality Classifier: Training Summary")
    print(f"{'=' * 70}")
    print(f"  Trained at: {trained_at}")
    print(f"  Total segments: {n_total}")
    print(f"  Train: {n_train}  |  Val: {n_val}  (temporal split)")
    print(f"\n  Training class distribution:")
    for label in ("clean", "noisy_ok", "bad"):
        count = train_dist.get(label, 0)
        print(f"    {label:12s}: {count:>6,}")
    print(f"\n  Validation class distribution:")
    for label in ("clean", "noisy_ok", "bad"):
        count = val_dist.get(label, 0)
        print(f"    {label:12s}: {count:>6,}")

    # Metrics
    if metrics.get("confusion_matrix"):
        print(f"\n  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"\n  Classification Report:")
        for line in metrics["classification_report"].split("\n"):
            print(f"    {line}")

        print(f"\n  Confusion Matrix (rows=true, cols=pred):")
        print(f"    {'':>12s}  {'clean':>8s}  {'noisy_ok':>8s}  {'bad':>8s}")
        cm = metrics["confusion_matrix"]
        for i, label in enumerate(["clean", "noisy_ok", "bad"]):
            if i < len(cm):
                row = cm[i]
                vals = "  ".join(f"{v:>8d}" for v in row)
                print(f"    {label:>12s}  {vals}")

    # Feature importances
    if metrics.get("top_features"):
        print(f"\n  Top 15 Features (by gain):")
        for rank, entry in enumerate(metrics["top_features"], 1):
            print(f"    {rank:>2d}. {entry['feature']:30s}  {entry['importance']:>10.1f}")

    print(f"{'=' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════


def predict(
    segment_features: pd.DataFrame,
    model_path: str,
) -> pd.DataFrame:
    """Run inference on segment features using a trained model.

    Loads the model artifact, verifies feature column compatibility,
    applies the stored median imputation, and returns class predictions
    with probabilities.

    Args:
        segment_features: Segment feature DataFrame, indexed by
            ``segment_idx`` (or containing it as a column).  Must have
            the same feature columns the model was trained on.
        model_path: Path to the joblib-serialised model artifact
            produced by :func:`train`.

    Returns:
        DataFrame with columns: segment_idx (int32), quality_pred (int),
        quality_label (str), p_clean (float32), p_noisy_ok (float32),
        p_bad (float32).

    Raises:
        FileNotFoundError: If model_path does not exist.
        ValueError: If required feature columns are missing from input.
    """
    mpath = Path(model_path)
    if not mpath.exists():
        raise FileNotFoundError(f"Model not found: {mpath}")

    artifact = joblib.load(mpath)

    clf = artifact["model"]
    feature_cols = artifact["feature_columns"]
    train_medians = artifact["train_medians"]
    trained_at = artifact.get("trained_at", "unknown")

    logger.info("Loaded model trained at %s with %d features", trained_at, len(feature_cols))

    # ── Prepare features DataFrame ────────────────────────────────────────
    df = segment_features.copy()
    if df.index.name == "segment_idx":
        df = df.reset_index()

    # Verify feature columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Feature mismatch: model expects columns {missing_cols} "
            f"which are missing from the input. Model was trained at "
            f"{trained_at} with features: {feature_cols}"
        )

    # Extract segment_idx before subsetting to features only
    segment_idx = df["segment_idx"].values.astype(np.int32)

    # ── Impute and predict ────────────────────────────────────────────────
    X = _apply_median_imputation(df[feature_cols], train_medians)

    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    # ── Build output DataFrame ────────────────────────────────────────────
    result = pd.DataFrame({
        "segment_idx": segment_idx,
        "quality_pred": y_pred.astype(int),
        "quality_label": [LABEL_DECODER[int(p)] for p in y_pred],
        "p_clean": y_proba[:, 0].astype(np.float32),
        "p_noisy_ok": y_proba[:, 1].astype(np.float32),
        "p_bad": y_proba[:, 2].astype(np.float32),
    })
    result["segment_idx"] = result["segment_idx"].astype(np.int32)

    logger.info(
        "Predictions: %d segments — %s",
        len(result),
        result["quality_label"].value_counts().to_dict(),
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MASK FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def get_bad_segment_mask(
    predictions_df: pd.DataFrame,
    include_noisy: bool = False,
) -> set[int]:
    """Return the set of segment indices predicted as bad (or worse).

    Used by downstream training scripts to exclude low-quality segments
    from beat-level feature matrices and model training.

    Args:
        predictions_df: Output of :func:`predict`, containing
            ``segment_idx`` and ``quality_pred`` columns.
        include_noisy: If True, also include segments predicted as
            ``noisy_ok`` (quality_pred == 1) in the mask.  Useful for
            conservative filtering during initial model development.

    Returns:
        Set of segment_idx values (ints) to exclude.
    """
    bad_mask = predictions_df["quality_pred"] == LABEL_ENCODER["bad"]

    if include_noisy:
        bad_mask = bad_mask | (predictions_df["quality_pred"] == LABEL_ENCODER["noisy_ok"])

    excluded = set(predictions_df.loc[bad_mask, "segment_idx"].astype(int).tolist())

    logger.info(
        "Segment mask: %d segments excluded (include_noisy=%s)",
        len(excluded), include_noisy,
    )

    return excluded


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _cli_train(args: argparse.Namespace) -> None:
    """Handle the ``train`` subcommand."""
    metrics = train(
        segment_features_path=args.segment_features,
        segments_labels_path=args.segments,
        output_model_path=args.output,
        val_fraction=args.val_fraction,
        random_seed=args.seed,
    )


def _cli_predict(args: argparse.Namespace) -> None:
    """Handle the ``predict`` subcommand."""
    sf_path = Path(args.segment_features)
    if not sf_path.exists():
        logger.error("Segment features not found: %s", sf_path)
        sys.exit(1)

    segment_features = pd.read_parquet(sf_path)
    logger.info("Loaded %d segment features from %s", len(segment_features), sf_path)

    result = predict(segment_features, args.model)

    # Save predictions
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(result, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    logger.info("Saved predictions → %s", out_path)

    # Print summary
    print(f"\n{'=' * 70}")
    print("  Segment Quality Predictions")
    print(f"{'=' * 70}")
    print(f"  Total segments: {len(result)}")
    print(f"\n  Prediction distribution:")
    for label in ("clean", "noisy_ok", "bad"):
        count = int((result["quality_label"] == label).sum())
        pct = 100.0 * count / max(len(result), 1)
        print(f"    {label:12s}: {count:>6,}  ({pct:5.1f}%)")
    print(f"\n  Probability statistics:")
    for col in ("p_clean", "p_noisy_ok", "p_bad"):
        vals = result[col]
        print(
            f"    {col:12s}: mean={vals.mean():.3f}  "
            f"std={vals.std():.3f}  "
            f"min={vals.min():.3f}  max={vals.max():.3f}"
        )

    # Mask summary
    bad_only = get_bad_segment_mask(result, include_noisy=False)
    bad_and_noisy = get_bad_segment_mask(result, include_noisy=True)
    print(f"\n  Segments to exclude:")
    print(f"    Bad only:          {len(bad_only):>6,}")
    print(f"    Bad + noisy_ok:    {len(bad_and_noisy):>6,}")

    print(f"\n  First 5 rows:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(result.head(5).to_string(index=False))
    print(f"{'=' * 70}\n")


def main() -> None:
    """CLI entry point with ``train`` and ``predict`` subcommands."""
    parser = argparse.ArgumentParser(
        description="ECG Artifact Pipeline — Stage 0: Segment Quality Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ─────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser(
        "train", help="Train the segment quality classifier"
    )
    train_parser.add_argument(
        "--segment-features", type=str, required=True,
        help="Path to segment_features.parquet",
    )
    train_parser.add_argument(
        "--segments", type=str, required=True,
        help="Path to segments.parquet (contains quality_label)",
    )
    train_parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the model artifact (.joblib)",
    )
    train_parser.add_argument(
        "--val-fraction", type=float, default=VAL_FRACTION,
        help=f"Fraction of segments for temporal validation (default: {VAL_FRACTION})",
    )
    train_parser.add_argument(
        "--seed", type=int, default=LGBM_RANDOM_STATE,
        help=f"Random seed for LightGBM (default: {LGBM_RANDOM_STATE})",
    )

    # ── predict ───────────────────────────────────────────────────────────
    predict_parser = subparsers.add_parser(
        "predict", help="Run inference on segment features"
    )
    predict_parser.add_argument(
        "--segment-features", type=str, required=True,
        help="Path to segment_features.parquet",
    )
    predict_parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model artifact (.joblib)",
    )
    predict_parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for predictions (.parquet)",
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
