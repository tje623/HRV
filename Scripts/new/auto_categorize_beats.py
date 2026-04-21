#!/usr/bin/env python3
"""
scripts/auto_categorize_beats.py — Feature-based beat auto-categorization.

Fits a shallow decision tree on reviewed labels, exports human-readable rules,
and applies them to the full dataset without any manual reannotation.

Subcommands
-----------
  fit     Train tree on reviewed labels → rules.json + tree.txt + tree_model.joblib
  apply   Apply rules to full dataset (chunked PyArrow streaming) → auto_categories.parquet
  encode  One-hot encode auto-categories and join to beat features → enriched parquet

Usage
-----
    python scripts/auto_categorize_beats.py fit \\
        --beat-features   data/processed/beat_features_merged.parquet \\
        --labels          data/processed/labels.parquet \\
        --output-rules    data/auto_categorization/rules.json \\
        --output-tree-viz data/auto_categorization/tree.txt \\
        --max-depth 6

    python scripts/auto_categorize_beats.py apply \\
        --beat-features data/processed/beat_features_merged.parquet \\
        --rules         data/auto_categorization/rules.json \\
        --output        data/processed/auto_categories.parquet

    python scripts/auto_categorize_beats.py encode \\
        --beat-features   data/processed/beat_features_merged.parquet \\
        --auto-categories data/processed/auto_categories.parquet \\
        --output          data/processed/beat_features_with_categories.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VAL_FRACTION, LGBM_RANDOM_STATE

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.auto_categorize_beats")

# ── Label vocabulary ──────────────────────────────────────────────────────────
CLEAN_REAL_LABELS = frozenset({"clean", "missed_original", "interpolated", "phys_event"})
ARTIFACT_LABEL    = "artifact"

# ── Category constants ────────────────────────────────────────────────────────
CLEAN_CATEGORIES    = frozenset({"pristine", "clean_normal", "clean_low_amplitude", "clean_noisy"})
ARTIFACT_CATEGORIES = frozenset({"artifact_morphology", "artifact_noise",
                                  "artifact_general", "baseline_wander"})

# Columns that are never features even if numeric
_NON_FEATURE = frozenset({"peak_id", "segment_idx", "label", "reviewed",
                           "hard_filtered", "y", "al_iteration"})
_NON_FEATURE_SUFFIXES = ("_id",)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _identify_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in _NON_FEATURE
        and not any(c.endswith(s) for s in _NON_FEATURE_SUFFIXES)
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _compute_medians(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for c in cols:
        med = df[c].median()
        result[c] = float(med) if pd.notna(med) else 0.0
    return result


def _impute(df: pd.DataFrame, medians: dict[str, float], cols: list[str]) -> pd.DataFrame:
    """Fill NaN in cols using stored medians (0.0 fallback for unknowns)."""
    df = df.copy()
    for c in cols:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(medians.get(c, 0.0))
    return df


def _temporal_split(df: pd.DataFrame, val_frac: float = VAL_FRACTION):
    """Split by segment_idx: earliest (1-val_frac) → train, latest val_frac → val."""
    segs = sorted(df["segment_idx"].dropna().unique())
    n_train = max(1, int(len(segs) * (1.0 - val_frac)))
    train_segs = set(segs[:n_train])
    mask = df["segment_idx"].isin(train_segs)
    return df[mask].copy(), df[~mask].copy()


def _print_metrics(y_true: np.ndarray, y_score: np.ndarray,
                   y_pred: np.ndarray, name: str) -> None:
    n     = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    acc   = float((y_true == y_pred).mean())
    cm    = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if n_pos > 0 and n_neg > 0:
        pr_auc  = float(average_precision_score(y_true, y_score))
        roc_auc = float(roc_auc_score(y_true, y_score))
    else:
        pr_auc = roc_auc = float("nan")
    print(f"\n  {name}")
    print(f"    n={n:,}  artifact={n_pos:,}  clean-real={n_neg:,}")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    PR-AUC   : {pr_auc:.4f}  ← PRIMARY")
    print(f"    ROC-AUC  : {roc_auc:.4f}")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"    Confusion matrix (actual→cols, pred→rows):")
        print(f"      pred clean | TN={tn:>8,}  FN={fn:>8,}")
        print(f"      pred artif | FP={fp:>8,}  TP={tp:>8,}")


# ── Tree traversal ────────────────────────────────────────────────────────────

def _get_leaf_paths(clf: DecisionTreeClassifier,
                    feature_names: list[str]) -> dict[int, list]:
    """Return {leaf_node_id: [(feat, op, threshold), ...]} for every leaf.

    op is ">" for the right (feature > threshold) branch,
    "<=" for the left (feature <= threshold) branch.
    """
    t      = clf.tree_
    left   = t.children_left
    right  = t.children_right
    feat   = t.feature
    thresh = t.threshold
    paths: dict[int, list] = {}

    def _recurse(node: int, path: list) -> None:
        if left[node] == right[node]:   # leaf: sentinel value is same for both
            paths[node] = list(path)
            return
        name = feature_names[feat[node]]
        tval = float(thresh[node])
        _recurse(left[node],  path + [(name, "<=", tval)])
        _recurse(right[node], path + [(name, ">",  tval)])

    _recurse(0, [])
    return paths


def _assign_category(
    leaf_id: int,                          # kept for potential future logging
    path: list[tuple[str, str, float]],
    n_clean: int,
    n_art: int,
) -> str:
    """Assign a human-readable category to a decision-tree leaf.

    Uses actual feature splits from the tree path (no hardcoded thresholds)
    combined with raw class counts at the leaf.

    Priority order matters — first match wins.
    """
    total = n_clean + n_art
    if total == 0:
        return "uncertain"
    clean_frac = n_clean / total
    art_frac   = n_art   / total

    # Quick lookup: {(feature_name, op): True}
    path_set: set[tuple[str, str]] = {(f, op) for f, op, _ in path}

    def _in(feature: str, op: str) -> bool:
        """Check if a (feature, op) pair appears anywhere in this leaf's path."""
        if (feature, op) in path_set:
            return True
        # window_zcr may appear as window_zero_crossing_rate in future versions
        if feature in {"window_zcr", "window_zero_crossing_rate"}:
            return (("window_zcr", op) in path_set
                    or ("window_zero_crossing_rate", op) in path_set)
        return False

    # 1. Pristine: template match + SNR split + high purity
    if _in("global_corr_clean", ">") and _in("r_peak_snr", ">") and clean_frac >= 0.95:
        return "pristine"
    # 2. Clean normal: template match + purity ≥ 90%
    if _in("global_corr_clean", ">") and clean_frac >= 0.90:
        return "clean_normal"
    # 3. Low amplitude: small IQR + predominantly clean
    if _in("window_iqr", "<=") and clean_frac >= 0.80:
        return "clean_low_amplitude"
    # 4. Noisy but clean: high HF noise + predominantly clean
    if _in("window_hf_noise_rms", ">") and clean_frac >= 0.70:
        return "clean_noisy"
    # 5. Baseline wander: high wander slope, or large IQR when not clean
    if _in("window_wander_slope", ">") or (_in("window_iqr", ">") and clean_frac < 0.70):
        return "baseline_wander"
    # 6. Artifact with poor template correlation
    if art_frac >= 0.70 and _in("global_corr_clean", "<="):
        return "artifact_morphology"
    # 7. Artifact with high zero-crossing rate (HF noise artifacts)
    if art_frac >= 0.70 and (_in("window_zcr", ">") or _in("window_zero_crossing_rate", ">")):
        return "artifact_noise"
    # 8. Generic artifact
    if art_frac >= 0.70:
        return "artifact_general"
    # 9. Mixed / ambiguous
    return "uncertain"


def _path_to_str(path: list[tuple[str, str, float]]) -> str:
    return " AND ".join(f"{f}{op}{t:.4g}" for f, op, t in path)


# ── Fit mode ──────────────────────────────────────────────────────────────────

def fit_mode(args: argparse.Namespace) -> None:
    feat_path     = Path(args.beat_features)
    labels_path   = Path(args.labels)
    rules_path    = Path(args.output_rules)
    tree_viz_path = Path(args.output_tree_viz)

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading features from %s", feat_path)
    features = pd.read_parquet(feat_path)
    if features.index.name == "peak_id":
        features = features.reset_index()

    logger.info("Loading labels from %s", labels_path)
    labels = pd.read_parquet(labels_path)
    if labels.index.name == "peak_id" and "peak_id" not in labels.columns:
        labels = labels.reset_index()

    # ── Non-negotiable: reviewed column must exist ────────────────────────────
    if "reviewed" not in labels.columns:
        logger.critical(
            "labels.parquet is missing the 'reviewed' column. "
            "Training on all labels would re-introduce the pre-catastrophe "
            "label contamination problem (unreviewed beats defaulting to clean). "
            "ABORTING."
        )
        sys.exit(1)

    # ── Load segment_idx from peaks.parquet (sibling file) ───────────────────
    peaks_path = feat_path.parent / "peaks.parquet"
    if not peaks_path.exists():
        logger.error(
            "peaks.parquet not found at %s — needed for temporal split", peaks_path
        )
        sys.exit(1)
    logger.info("Loading peaks (segment_idx) from %s", peaks_path)
    peaks = pd.read_parquet(peaks_path, columns=["peak_id", "segment_idx"])
    if peaks.index.name == "peak_id" and "peak_id" not in peaks.columns:
        peaks = peaks.reset_index()

    # ── Filter to reviewed beats only ─────────────────────────────────────────
    n_total = len(labels)
    rev = labels[labels["reviewed"].astype(bool)].copy()
    logger.info("Labels: %d total, %d reviewed (%.1f%%)",
                n_total, len(rev), 100.0 * len(rev) / max(n_total, 1))
    if len(rev) == 0:
        logger.error("No reviewed beats — cannot train.")
        sys.exit(1)

    # ── Build binary target ────────────────────────────────────────────────────
    if "hard_filtered" not in rev.columns:
        rev = rev.copy()
        rev["hard_filtered"] = False

    rev = rev.copy()
    rev["y"] = (
        rev["label"].isin({ARTIFACT_LABEL}) | rev["hard_filtered"].astype(bool)
    ).astype(int)

    n_clean = int((rev["y"] == 0).sum())
    n_art   = int((rev["y"] == 1).sum())
    logger.info("Class distribution — clean-real: %d  artifact: %d", n_clean, n_art)

    if n_clean == 0 or n_art == 0:
        logger.error("Single-class training set — tree cannot learn. Aborting.")
        sys.exit(1)

    # ── Merge features + labels + segment_idx ────────────────────────────────
    merged = (
        features
        .merge(rev[["peak_id", "y"]], on="peak_id", how="inner")
        .merge(peaks[["peak_id", "segment_idx"]].drop_duplicates("peak_id"),
               on="peak_id", how="left")
    )
    if merged["segment_idx"].isna().any():
        n_missing = int(merged["segment_idx"].isna().sum())
        logger.warning("%d peaks have no segment_idx — dropping from training", n_missing)
        merged = merged.dropna(subset=["segment_idx"])
    merged["segment_idx"] = merged["segment_idx"].astype(int)
    logger.info("After merge: %d beats", len(merged))

    # ── Feature columns ───────────────────────────────────────────────────────
    feature_cols = _identify_feature_cols(merged)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # ── Impute ────────────────────────────────────────────────────────────────
    medians = _compute_medians(merged, feature_cols)
    merged  = _impute(merged, medians, feature_cols)

    # ── Temporal split ────────────────────────────────────────────────────────
    train_df, val_df = _temporal_split(merged)
    logger.info("Temporal split — train: %d  val: %d", len(train_df), len(val_df))

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["y"].values.astype(int)
    X_val   = val_df[feature_cols].values.astype(np.float32) if len(val_df) else np.empty((0, len(feature_cols)))
    y_val   = val_df["y"].values.astype(int) if len(val_df) else np.array([], dtype=int)

    # ── Fit ───────────────────────────────────────────────────────────────────
    logger.info(
        "Fitting DecisionTreeClassifier(max_depth=%d, min_samples_leaf=100, "
        "class_weight=balanced) …",
        args.max_depth,
    )
    clf = DecisionTreeClassifier(
        max_depth        = args.max_depth,
        min_samples_leaf = 100,
        class_weight     = "balanced",
        random_state     = LGBM_RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    logger.info("Tree: %d nodes, %d leaves", clf.tree_.node_count, clf.get_n_leaves())

    # ── Metrics ───────────────────────────────────────────────────────────────
    print()
    print("═" * 68)
    print("  DECISION TREE FIT RESULTS")
    print("═" * 68)
    _print_metrics(y_train, clf.predict_proba(X_train)[:, 1], clf.predict(X_train), "Training set")
    if len(y_val) > 0:
        _print_metrics(y_val, clf.predict_proba(X_val)[:, 1], clf.predict(X_val), "Held-out (temporal val)")

    # Feature importances (all non-zero)
    print("\n  Feature importances (Gini decrease):")
    for fname, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]):
        if imp > 0:
            print(f"    {fname:<42}  {imp:.6f}")

    # ── Export tree text ───────────────────────────────────────────────────────
    tree_viz_path.parent.mkdir(parents=True, exist_ok=True)
    tree_text = export_text(clf, feature_names=list(feature_cols), max_depth=args.max_depth)
    tree_viz_path.write_text(tree_text)
    logger.info("Tree text → %s", tree_viz_path)
    lines = tree_text.splitlines()
    cap = 60
    print(f"\n  Tree text (first {cap} of {len(lines)} lines):")
    for line in lines[:cap]:
        print(f"    {line}")
    if len(lines) > cap:
        print(f"    … see {tree_viz_path} for full tree")

    # ── Leaf paths and raw class counts ───────────────────────────────────────
    leaf_paths     = _get_leaf_paths(clf, feature_cols)
    train_leaf_ids = clf.apply(X_train)

    leaf_raw: dict[int, dict[int, int]] = {}
    for lid, yi in zip(train_leaf_ids.tolist(), y_train.tolist()):
        if lid not in leaf_raw:
            leaf_raw[lid] = {0: 0, 1: 0}
        leaf_raw[lid][yi] += 1
    for lid in leaf_paths:
        if lid not in leaf_raw:
            leaf_raw[lid] = {0: 0, 1: 0}

    # ── Assign categories ─────────────────────────────────────────────────────
    leaf_to_cat: dict[int, str] = {}
    leaf_purity:  dict[int, float] = {}
    for lid, path in leaf_paths.items():
        c0 = leaf_raw[lid][0]
        c1 = leaf_raw[lid][1]
        cat = _assign_category(lid, path, c0, c1)
        leaf_to_cat[lid] = cat
        total = c0 + c1
        leaf_purity[lid] = float(max(c0, c1) / total) if total > 0 else 0.0

    cat_train_counts: dict[str, int] = {}
    for lid, cat in leaf_to_cat.items():
        n = leaf_raw[lid][0] + leaf_raw[lid][1]
        cat_train_counts[cat] = cat_train_counts.get(cat, 0) + n
    print("\n  Leaf-to-category assignment (training samples):")
    for cat, n in sorted(cat_train_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<32}  {n:>8,}")

    # ── Validation metrics dict ────────────────────────────────────────────────
    val_metrics: dict = {}
    if len(y_val) > 0:
        vp = clf.predict_proba(X_val)[:, 1]
        n_vpos = int(y_val.sum())
        n_vneg = len(y_val) - n_vpos
        val_metrics = {
            "n_val": len(y_val),
            "n_val_artifact": n_vpos,
            "n_val_clean": n_vneg,
            "pr_auc":  float(average_precision_score(y_val, vp)) if n_vpos > 0 and n_vneg > 0 else None,
            "roc_auc": float(roc_auc_score(y_val, vp)) if n_vpos > 0 and n_vneg > 0 else None,
        }

    # ── Build and save rules JSON ──────────────────────────────────────────────
    leaf_details = [
        {
            "leaf_id":       lid,
            "category":      leaf_to_cat[lid],
            "path_rules":    [{"feature": f, "op": op, "threshold": round(t, 8)}
                               for f, op, t in path],
            "class_support": {"clean-real": leaf_raw[lid][0], "artifact": leaf_raw[lid][1]},
            "purity":        round(leaf_purity[lid], 6),
            "n_samples":     leaf_raw[lid][0] + leaf_raw[lid][1],
            "path_str":      _path_to_str(path),
        }
        for lid, path in sorted(leaf_paths.items())
    ]

    rules = {
        "version":            "1.0",
        "built_at":           datetime.now(timezone.utc).isoformat(),
        "n_training_beats":   len(train_df),
        "n_val_beats":        len(val_df),
        "class_distribution": {"clean-real": n_clean, "artifact": n_art},
        "feature_columns":    feature_cols,
        "feature_medians":    {k: round(v, 8) for k, v in medians.items()},
        "max_depth":          args.max_depth,
        "n_leaves":           clf.get_n_leaves(),
        "val_metrics":        val_metrics,
        "leaf_to_category":   {str(k): v for k, v in leaf_to_cat.items()},
        "leaf_purity":        {str(k): round(v, 6) for k, v in leaf_purity.items()},
        "leaf_details":       leaf_details,
    }
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rules_path, "w") as f:
        json.dump(rules, f, indent=2, cls=_NpEncoder)
    logger.info("Rules JSON → %s", rules_path)

    # Save joblib model artifact
    model_path = rules_path.parent / "tree_model.joblib"
    joblib.dump(
        {"clf": clf, "feature_columns": feature_cols, "feature_medians": medians},
        model_path,
    )
    logger.info("Model artifact → %s", model_path)

    print()
    print("═" * 68)
    print(f"  Rules     → {rules_path}")
    print(f"  Tree text → {tree_viz_path}")
    print(f"  Model     → {model_path}")
    print("═" * 68)
    print()


# ── Apply mode ────────────────────────────────────────────────────────────────

def apply_mode(args: argparse.Namespace) -> None:
    rules_path  = Path(args.rules)
    feat_path   = Path(args.beat_features)
    output_path = Path(args.output)
    chunk_size  = args.chunk_size

    # ── Load rules ─────────────────────────────────────────────────────────────
    logger.info("Loading rules from %s", rules_path)
    with open(rules_path) as f:
        rules = json.load(f)

    feature_cols    = rules["feature_columns"]
    medians         = rules["feature_medians"]
    leaf_to_cat     = {int(k): v for k, v in rules["leaf_to_category"].items()}
    leaf_purity_map = {int(k): float(v) for k, v in rules["leaf_purity"].items()}

    model_path = rules_path.parent / "tree_model.joblib"
    if not model_path.exists():
        logger.error("tree_model.joblib not found at %s", model_path)
        sys.exit(1)
    logger.info("Loading model from %s", model_path)
    clf: DecisionTreeClassifier = joblib.load(model_path)["clf"]

    # Pre-build leaf → path string for the tree_path output column
    leaf_path_str = {lid: _path_to_str(p)
                     for lid, p in _get_leaf_paths(clf, feature_cols).items()}

    # ── Stream parquet ─────────────────────────────────────────────────────────
    logger.info("Streaming %s in chunks of %d …", feat_path, chunk_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema_out = pa.schema([
        ("peak_id",             pa.int64()),
        ("category",            pa.string()),
        ("leaf_id",             pa.int32()),
        ("category_confidence", pa.float32()),
        ("tree_path",           pa.string()),
    ])

    pf     = pq.ParquetFile(str(feat_path))
    writer = pq.ParquetWriter(str(output_path), schema=schema_out, compression="snappy")

    total_rows: int = 0
    cat_totals: dict[str, int] = {}

    try:
        for batch in pf.iter_batches(batch_size=chunk_size):
            df = batch.to_pandas()
            if "peak_id" not in df.columns:
                if df.index.name == "peak_id":
                    df = df.reset_index()
                else:
                    raise ValueError("peak_id not found in batch columns or index")

            peak_ids = df["peak_id"].values.astype(np.int64)

            # Impute each feature column
            for c in feature_cols:
                if c not in df.columns:
                    df[c] = medians.get(c, 0.0)
                elif df[c].isna().any():
                    df[c] = df[c].fillna(medians.get(c, 0.0))

            X        = df[feature_cols].values.astype(np.float32)
            leaf_ids = clf.apply(X).astype(np.int32)

            cats        = [leaf_to_cat.get(int(lid), "uncertain") for lid in leaf_ids]
            confidences = np.array([leaf_purity_map.get(int(lid), 0.0) for lid in leaf_ids],
                                   dtype=np.float32)
            paths_str   = [leaf_path_str.get(int(lid), "") for lid in leaf_ids]

            writer.write_table(pa.table({
                "peak_id":             pa.array(peak_ids,    type=pa.int64()),
                "category":            pa.array(cats,        type=pa.string()),
                "leaf_id":             pa.array(leaf_ids,    type=pa.int32()),
                "category_confidence": pa.array(confidences, type=pa.float32()),
                "tree_path":           pa.array(paths_str,   type=pa.string()),
            }))

            total_rows += len(df)
            for c in cats:
                cat_totals[c] = cat_totals.get(c, 0) + 1

            if total_rows % 5_000_000 < chunk_size:
                logger.info("  … %d rows processed", total_rows)
    finally:
        writer.close()

    logger.info("Wrote %d rows → %s", total_rows, output_path)

    print()
    print("═" * 60)
    print("  AUTO-CATEGORIZATION COMPLETE")
    print("═" * 60)
    print(f"  Total beats : {total_rows:,}")
    print()
    print(f"  {'Category':<32}  {'Count':>10}  {'%':>7}")
    print(f"  {'─'*32}  {'─'*10}  {'─'*7}")
    for cat in sorted(cat_totals, key=lambda c: -cat_totals[c]):
        n   = cat_totals[cat]
        pct = 100.0 * n / max(total_rows, 1)
        print(f"  {cat:<32}  {n:>10,}  {pct:>6.2f}%")
    print("═" * 60)
    print()


# ── Encode mode ───────────────────────────────────────────────────────────────

def encode_mode(args: argparse.Namespace) -> None:
    feat_path   = Path(args.beat_features)
    cats_path   = Path(args.auto_categories)
    output_path = Path(args.output)

    logger.info("Loading features from %s", feat_path)
    features = pd.read_parquet(feat_path)
    if features.index.name == "peak_id":
        features = features.reset_index()

    logger.info("Loading auto-categories from %s", cats_path)
    cats = pd.read_parquet(cats_path, columns=["peak_id", "category"])

    n_cats = cats["category"].nunique()
    logger.info("One-hot encoding %d unique categories …", n_cats)
    dummies = pd.get_dummies(cats["category"], prefix="cat", dtype=np.float32)
    cats_enc = pd.concat([cats[["peak_id"]], dummies], axis=1)

    enriched = features.merge(cats_enc, on="peak_id", how="left")
    cat_cols = [c for c in enriched.columns if c.startswith("cat_")]
    for col in cat_cols:
        enriched[col] = enriched[col].fillna(np.float32(0.0))

    logger.info(
        "Enriched: %d rows × %d columns (+%d category dummies)",
        len(enriched), len(enriched.columns), len(cat_cols),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Restore peak_id as index to match the input schema
    out_df = enriched.set_index("peak_id") if "peak_id" in enriched.columns else enriched
    pq.write_table(
        pa.Table.from_pandas(out_df, preserve_index=True),
        str(output_path),
        compression="snappy",
    )
    logger.info("Saved → %s", output_path)

    print()
    print(f"  Category dummies added: {cat_cols}")
    print(f"  Output: {output_path}  ({len(enriched):,} rows × {len(enriched.columns)} cols)")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature-based beat auto-categorization  (fit / apply / encode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # fit
    pf = sub.add_parser("fit", help="Fit decision tree on reviewed labels")
    pf.add_argument("--beat-features",    required=True)
    pf.add_argument("--labels",           required=True)
    pf.add_argument("--output-rules",     required=True)
    pf.add_argument("--output-tree-viz",  required=True)
    pf.add_argument("--max-depth",        type=int, default=6)

    # apply
    pa_ = sub.add_parser("apply", help="Apply rules to full dataset (chunked)")
    pa_.add_argument("--beat-features",  required=True)
    pa_.add_argument("--rules",          required=True)
    pa_.add_argument("--output",         required=True)
    pa_.add_argument("--chunk-size",     type=int, default=1_000_000)

    # encode
    pe = sub.add_parser("encode", help="One-hot encode auto-categories and join to features")
    pe.add_argument("--beat-features",   required=True)
    pe.add_argument("--auto-categories", required=True)
    pe.add_argument("--output",          required=True)

    args = parser.parse_args()
    if args.command == "fit":
        fit_mode(args)
    elif args.command == "apply":
        apply_mode(args)
    elif args.command == "encode":
        encode_mode(args)


if __name__ == "__main__":
    main()
