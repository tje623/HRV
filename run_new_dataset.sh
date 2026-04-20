#!/bin/bash
# Full inference pipeline for new ECG dataset.
# Run from anywhere — uses absolute paths throughout.
# Usage: bash run_new_dataset.sh

set -e  # exit immediately if any step fails

cd "/Volumes/xHRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate

python ecgclean/detect_peaks.py \
  --ecg-dir "/Volumes/xHRV/ECG/" \
  --output-dir data/new_peaks/ \
  --method ensemble

python ecgclean/data_pipeline.py \
  --ecg-dir "/Volumes/xHRV/ECG/" \
  --peaks-dir data/new_peaks/ \
  --annotations data/annotations/artifact_annotations.json \
  --output-dir data/new_processed/

python ecgclean/physio_constraints.py \
  --processed-dir data/new_processed/

python ecgclean/features/beat_features.py \
  --processed-dir data/new_processed/ \
  --output data/new_processed/beat_features.parquet

python ecgclean/features/segment_features.py \
  --processed-dir data/new_processed/ \
  --output data/new_processed/segment_features.parquet

python ecgclean/models/segment_quality.py predict \
  --segment-features data/new_processed/segment_features.parquet \
  --model models/segment_quality_v1.joblib \
  --output data/new_processed/segment_quality_preds.parquet

python ecgclean/models/beat_artifact_tabular.py predict \
  --beat-features data/new_processed/beat_features.parquet \
  --model models/beat_tabular_v2.joblib \
  --output data/new_processed/beat_tabular_preds.parquet

python ecgclean/models/beat_artifact_cnn.py predict \
  --beat-features data/new_processed/beat_features.parquet \
  --ecg-samples data/new_processed/ecg_samples.parquet \
  --model models/beat_cnn_v1.pt \
  --output data/new_processed/beat_cnn_preds.parquet

python ecgclean/models/ensemble.py fuse \
  --tabular-preds data/new_processed/beat_tabular_preds.parquet \
  --cnn-preds data/new_processed/beat_cnn_preds.parquet \
  --output data/new_processed/ensemble_preds.parquet \
  --alpha 0.5

echo ""
echo "Done. Results: data/new_processed/ensemble_preds.parquet"
