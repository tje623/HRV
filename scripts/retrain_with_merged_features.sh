#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR="/Volumes/xHRV/Artifact Detector"
VENV="/Users/tannereddy/.envs/hrv/bin/activate"
DATA_DIR="${PROJECT_DIR}/data/processed"
MODELS_DIR="${PROJECT_DIR}/models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

source "$VENV"
cd "$PROJECT_DIR"

echo "══════════════════════════════════════════════════════════════════════"
echo "  Retraining Pipeline — ${TIMESTAMP}"
echo "══════════════════════════════════════════════════════════════════════"

# ── Step 0: Verify prerequisites exist ─────────────────────────────────────
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

# ── Step 1: Merge features ────────────────────────────────────────────────
echo ""
echo "── Step 1: Merging beat features + global template features ──"
python scripts/merge_features_for_training.py \
    --beat-features            "${DATA_DIR}/beat_features_v2.parquet" \
    --global-template-features "${DATA_DIR}/global_template_features.parquet" \
    --output                   "${DATA_DIR}/beat_features_merged.parquet"

# ── Step 2: Backup current models ─────────────────────────────────────────
echo ""
echo "── Step 2: Backing up current models ──"
mkdir -p "${MODELS_DIR}/backup_${TIMESTAMP}"
for m in beat_tabular_v2.joblib beat_cnn_v1.pt; do
    if [ -f "${MODELS_DIR}/${m}" ]; then
        cp "${MODELS_DIR}/${m}" "${MODELS_DIR}/backup_${TIMESTAMP}/${m}"
        echo "  ✓ Backed up ${m}"
    else
        echo "  ⚠ ${m} not found (skipping backup)"
    fi
done

# ── Step 3: Retrain tabular model ─────────────────────────────────────────
echo ""
echo "── Step 3: Retraining LightGBM (tabular) ──"
echo "  Input features: ${DATA_DIR}/beat_features_merged.parquet"
echo "  Output model:   ${MODELS_DIR}/beat_tabular_v3_merged.joblib"
python ecgclean/models/beat_artifact_tabular.py train \
    --beat-features          "${DATA_DIR}/beat_features_merged.parquet" \
    --labels                 "${DATA_DIR}/labels.parquet" \
    --segment-quality-preds  "${DATA_DIR}/segment_quality_preds.parquet" \
    --output                 "${MODELS_DIR}/beat_tabular_v3_merged.joblib"

# ── Step 4: Run tabular predictions ───────────────────────────────────────
echo ""
echo "── Step 4: Running tabular predictions on full dataset ──"
python ecgclean/models/beat_artifact_tabular.py predict \
    --beat-features "${DATA_DIR}/beat_features_merged.parquet" \
    --model         "${MODELS_DIR}/beat_tabular_v3_merged.joblib" \
    --output        "${DATA_DIR}/beat_tabular_preds_v3.parquet"

# ── Step 5: CNN retraining ────────────────────────────────────────────────
# NOTE: The CNN is commented out intentionally.
# Pre-catastrophe CNN PR-AUC was 0.4622 — barely above random.
# With global_corr_clean boosting the tabular model, the CNN may be redundant.
# Evaluate tabular-only results first; uncomment here and in Step 6 if needed.

# echo ""
# echo "── Step 5: Retraining CNN ──"
# python ecgclean/models/beat_artifact_cnn.py train \
#     --beat-features          "${DATA_DIR}/beat_features_merged.parquet" \
#     --labels                 "${DATA_DIR}/labels.parquet" \
#     --segment-quality-preds  "${DATA_DIR}/segment_quality_preds.parquet" \
#     --ecg-samples            "${DATA_DIR}/ecg_samples.parquet" \
#     --output                 "${MODELS_DIR}/beat_cnn_v2_merged.pt"

# ── Step 6: Run ensemble ──────────────────────────────────────────────────
# Using --tabular-only: CNN was not retrained, so disagreement is set to 0.0.
# The output schema is identical to a two-model fuse (all expected columns
# are present); downstream tools (beat_reannotator, export_disagreement_queue)
# will work unchanged — zero disagreement simply means no beats are flagged
# as high-disagreement candidates.
#
# To switch to a real two-model ensemble after retraining the CNN:
#   Remove --tabular-only, add:
#     --cnn-preds "${DATA_DIR}/beat_cnn_preds_v2.parquet" --alpha 0.55
echo ""
echo "── Step 6: Running ensemble (tabular-only, CNN not yet retrained) ──"
python ecgclean/models/ensemble.py fuse \
    --tabular-preds "${DATA_DIR}/beat_tabular_preds_v3.parquet" \
    --output        "${DATA_DIR}/ensemble_preds_v3.parquet" \
    --tabular-only

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  DONE — ${TIMESTAMP}"
echo "  Tabular model : ${MODELS_DIR}/beat_tabular_v3_merged.joblib"
echo "  Tabular preds : ${DATA_DIR}/beat_tabular_preds_v3.parquet"
echo "  Ensemble preds: ${DATA_DIR}/ensemble_preds_v3.parquet"
echo "══════════════════════════════════════════════════════════════════════"
