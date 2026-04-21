#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR="/Volumes/xHRV/Artifact Detector"
VENV="/Users/tannereddy/.envs/hrv/bin/activate"
DATA_DIR="${PROJECT_DIR}/data/processed"
AUTO_DIR="${PROJECT_DIR}/data/auto_categorization"
MODELS_DIR="${PROJECT_DIR}/models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

source "$VENV"
cd "$PROJECT_DIR"

echo "══════════════════════════════════════════════════════════════════════"
echo "  Retraining Pipeline (with Auto-Categories) — ${TIMESTAMP}"
echo "══════════════════════════════════════════════════════════════════════"

# ── Step 0: Verify prerequisites ─────────────────────────────────────────────
echo ""
echo "── Step 0: Checking prerequisites ──"
for f in \
    "${DATA_DIR}/beat_features_v2.parquet" \
    "${DATA_DIR}/global_template_features.parquet" \
    "${DATA_DIR}/labels.parquet" \
    "${DATA_DIR}/peaks.parquet" \
    "${DATA_DIR}/segment_quality_preds.parquet"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing required file: $f"
        exit 1
    fi
    echo "  ✓ $(basename "$f")"
done

# ── Step 1: Merge beat features + global template features ────────────────────
echo ""
echo "── Step 1: Merging beat features + global template features ──"
python scripts/merge_features_for_training.py \
    --beat-features            "${DATA_DIR}/beat_features_v2.parquet" \
    --global-template-features "${DATA_DIR}/global_template_features.parquet" \
    --output                   "${DATA_DIR}/beat_features_merged.parquet"

# ── Step 2: Fit auto-categorizer (skip if rules already exist) ────────────────
echo ""
mkdir -p "${AUTO_DIR}"
if [ ! -f "${AUTO_DIR}/rules.json" ]; then
    echo "── Step 2: Fitting auto-categorizer (reviewed labels only) ──"
    python scripts/auto_categorize_beats.py fit \
        --beat-features   "${DATA_DIR}/beat_features_merged.parquet" \
        --labels          "${DATA_DIR}/labels.parquet" \
        --output-rules    "${AUTO_DIR}/rules.json" \
        --output-tree-viz "${AUTO_DIR}/tree.txt" \
        --max-depth 6
else
    echo "── Step 2: Using existing auto-categorizer rules (${AUTO_DIR}/rules.json) ──"
fi

# ── Step 3: Apply auto-categorizer to training dataset ───────────────────────
echo ""
echo "── Step 3: Applying auto-categorizer ──"
python scripts/auto_categorize_beats.py apply \
    --beat-features "${DATA_DIR}/beat_features_merged.parquet" \
    --rules         "${AUTO_DIR}/rules.json" \
    --output        "${DATA_DIR}/auto_categories.parquet"

# ── Step 4: One-hot encode categories and join to feature matrix ──────────────
echo ""
echo "── Step 4: Encoding auto-categories into feature matrix ──"
python scripts/auto_categorize_beats.py encode \
    --beat-features   "${DATA_DIR}/beat_features_merged.parquet" \
    --auto-categories "${DATA_DIR}/auto_categories.parquet" \
    --output          "${DATA_DIR}/beat_features_with_categories.parquet"

# ── Step 5: Backup current models ─────────────────────────────────────────────
echo ""
echo "── Step 5: Backing up current models ──"
mkdir -p "${MODELS_DIR}/backup_${TIMESTAMP}"
for m in beat_tabular_v2.joblib beat_cnn_v1.pt beat_tabular_v3_merged.joblib; do
    if [ -f "${MODELS_DIR}/${m}" ]; then
        cp "${MODELS_DIR}/${m}" "${MODELS_DIR}/backup_${TIMESTAMP}/${m}"
        echo "  ✓ Backed up ${m}"
    else
        echo "  ⚠ ${m} not found (skipping)"
    fi
done

# ── Step 6: Retrain LightGBM on enriched features ────────────────────────────
# Option A: auto-categories as additional one-hot features (safer, backward-compatible)
# The LightGBM sees cat_pristine, cat_clean_normal, … as regular features alongside
# global_corr_clean, r_peak_snr, etc.  The binary target (artifact vs clean) is unchanged.
#
# TODO (Option B): multi-class target using auto_category as the prediction target.
#   This would require changes to beat_artifact_tabular.py and downstream scoring.
#   Deferred until Option A is validated.
echo ""
echo "── Step 6: Retraining LightGBM (tabular, with category features) ──"
echo "  Input:  ${DATA_DIR}/beat_features_with_categories.parquet"
echo "  Output: ${MODELS_DIR}/beat_tabular_v4_autocats.joblib"
python ecgclean/models/beat_artifact_tabular.py train \
    --beat-features         "${DATA_DIR}/beat_features_with_categories.parquet" \
    --labels                "${DATA_DIR}/labels.parquet" \
    --segment-quality-preds "${DATA_DIR}/segment_quality_preds.parquet" \
    --output                "${MODELS_DIR}/beat_tabular_v4_autocats.joblib"

# ── Step 7: Run tabular predictions ───────────────────────────────────────────
echo ""
echo "── Step 7: Running tabular predictions ──"
python ecgclean/models/beat_artifact_tabular.py predict \
    --beat-features "${DATA_DIR}/beat_features_with_categories.parquet" \
    --model         "${MODELS_DIR}/beat_tabular_v4_autocats.joblib" \
    --output        "${DATA_DIR}/beat_tabular_preds_v4.parquet"

# ── Step 8: CNN retraining (commented out — see retrain_with_merged_features.sh) ──
# Pre-catastrophe CNN PR-AUC was 0.4622.  Evaluate tabular-only first.
# Uncomment here and in Step 9 after the tabular results are validated.

# ── Step 9: Ensemble (tabular-only) ───────────────────────────────────────────
echo ""
echo "── Step 9: Running ensemble (tabular-only) ──"
python ecgclean/models/ensemble.py fuse \
    --tabular-preds "${DATA_DIR}/beat_tabular_preds_v4.parquet" \
    --output        "${DATA_DIR}/ensemble_preds_v4.parquet" \
    --tabular-only

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  DONE — ${TIMESTAMP}"
echo "  Tabular model : ${MODELS_DIR}/beat_tabular_v4_autocats.joblib"
echo "  Tabular preds : ${DATA_DIR}/beat_tabular_preds_v4.parquet"
echo "  Ensemble preds: ${DATA_DIR}/ensemble_preds_v4.parquet"
echo ""
echo "  Next: run validate_retrained_model.py to compare v4 vs the previous"
echo "  best model before switching ensemble_preds_v4 into production."
echo "══════════════════════════════════════════════════════════════════════"
