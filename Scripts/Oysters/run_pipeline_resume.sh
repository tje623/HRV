#!/bin/bash
# Resume pipeline from data_pipeline.py (detect_peaks already done).
# Each step is skipped automatically if its output file already exists.
# Safe to re-run after any crash — only the failed step and everything after it will re-run.

set -e

cd "/Volumes/xHRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate

OUT="/Volumes/xHRV/processed"

run_if_missing() {
    local output_file="$1"
    shift
    if [ -f "$output_file" ]; then
        echo "[SKIP] Already exists: $output_file"
    else
        echo "[RUN] $*"
        "$@"
    fi
}

run_if_missing "$OUT/ecg_samples.parquet" \
  python ecgclean/data_pipeline.py \
    --ecg-dir "/Volumes/xHRV/ECG" \
    --peaks-dir data/new_peaks/ \
    --annotations data/annotations/artifact_annotations.json \
    --output-dir "$OUT" \
    --resume-partial "data/new_processed/ecg_samples.parquet" \
    --workers 10

run_if_missing "$OUT/physio_constraints_done.flag" \
  bash -c "python ecgclean/physio_constraints.py --processed-dir \"$OUT\" && touch \"$OUT/physio_constraints_done.flag\""

run_if_missing "$OUT/segment_features.parquet" \
  python ecgclean/features/segment_features.py \
    --processed-dir "$OUT" \
    --output "$OUT/segment_features.parquet"

run_if_missing "$OUT/segment_quality_preds.parquet" \
  python ecgclean/models/segment_quality.py predict \
    --segment-features "$OUT/segment_features.parquet" \
    --model models/segment_quality_v1.joblib \
    --output "$OUT/segment_quality_preds.parquet"

run_if_missing "$OUT/beat_features.parquet" \
  python ecgclean/features/beat_features.py \
    --processed-dir "$OUT" \
    --segment-quality-preds "$OUT/segment_quality_preds.parquet" \
    --output "$OUT/beat_features.parquet"

run_if_missing "$OUT/beat_tabular_preds.parquet" \
  python ecgclean/models/beat_artifact_tabular.py predict \
    --beat-features "$OUT/beat_features.parquet" \
    --model models/beat_tabular_v2.joblib \
    --output "$OUT/beat_tabular_preds.parquet"

run_if_missing "$OUT/beat_cnn_preds.parquet" \
  python ecgclean/models/beat_artifact_cnn.py predict \
    --beat-features "$OUT/beat_features.parquet" \
    --ecg-samples "$OUT/ecg_samples.parquet" \
    --model models/beat_cnn_v1.pt \
    --output "$OUT/beat_cnn_preds.parquet"

run_if_missing "$OUT/ensemble_preds.parquet" \
  python ecgclean/models/ensemble.py fuse \
    --tabular-preds "$OUT/beat_tabular_preds.parquet" \
    --cnn-preds "$OUT/beat_cnn_preds.parquet" \
    --output "$OUT/ensemble_preds.parquet" \
    --alpha 0.5

echo ""
echo "Done. Results: $OUT/ensemble_preds.parquet"
