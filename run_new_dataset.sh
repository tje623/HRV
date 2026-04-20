#!/bin/bash
# Incremental ingestion pipeline — new ECG files only.
#
# Workflow:
#   1. Put new Polar H10 CSV files in /Volumes/xHRV/ECG_new/
#   2. Run this script.
#   3. Script processes ONLY those new files (never re-reads the 80 GB archive).
#   4. Appends new predictions to /Volumes/xHRV/processed/ensemble_preds.parquet.
#   5. Move verified new CSVs to /Volumes/xHRV/ECG/ for archival.
#
# Usage:
#   bash run_new_dataset.sh
#   bash run_new_dataset.sh --ecg-dir /path/to/new/csvs        # custom staging dir
#   bash run_new_dataset.sh --skip-append                       # skip main-dataset merge
#
# Output goes to data/new_processed/  (overwritten each run).
# Main dataset at /Volumes/xHRV/processed/ is only touched at the append step.

set -e
cd "/Volumes/xHRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate

# ── Defaults ─────────────────────────────────────────────────────────────────
NEW_ECG_DIR="/Volumes/xHRV/ECG_new"
NEW_PROC="data/new_processed"
FULL_PROC="/Volumes/xHRV/processed"
SKIP_APPEND=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ecg-dir)      NEW_ECG_DIR="$2"; shift 2 ;;
    --skip-append)  SKIP_APPEND=1;    shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [ ! -d "$NEW_ECG_DIR" ]; then
  echo "ERROR: ECG staging directory not found: $NEW_ECG_DIR"
  echo "Put new CSV files there before running."
  exit 1
fi

N_NEW=$(ls "$NEW_ECG_DIR"/*.csv 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_NEW" -eq 0 ]; then
  echo "No CSV files found in $NEW_ECG_DIR — nothing to do."
  exit 0
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  Incremental ingestion: $N_NEW new CSV file(s) from $NEW_ECG_DIR"
echo "═══════════════════════════════════════════════════════════════════"

mkdir -p "$NEW_PROC"

# ── Stage 1: R-peak detection ─────────────────────────────────────────────────
echo "[1/10] Detecting R-peaks..."
python ecgclean/detect_peaks.py \
  --ecg-dir "$NEW_ECG_DIR" \
  --output-dir data/new_peaks/ \
  --method ensemble

# ── Stage 2: Build canonical Parquet tables ────────────────────────────────────
echo "[2/10] Building Parquet tables..."
python ecgclean/data_pipeline.py \
  --ecg-dir "$NEW_ECG_DIR" \
  --peaks-dir data/new_peaks/ \
  --annotations /Volumes/xHRV/data/annotations/artifact_annotations_final.json \
  --output-dir "$NEW_PROC/"

# ── Stage 3: Physio constraints ────────────────────────────────────────────────
echo "[3/10] Applying physio constraints..."
python ecgclean/physio_constraints.py \
  --processed-dir "$NEW_PROC/"

# ── Stage 4: Beat features ────────────────────────────────────────────────────
echo "[4/10] Extracting beat features..."
python ecgclean/features/beat_features.py \
  --processed-dir "$NEW_PROC/" \
  --output "$NEW_PROC/beat_features.parquet"

# ── Stage 5: Segment features + quality ───────────────────────────────────────
echo "[5/10] Extracting segment features..."
python ecgclean/features/segment_features.py \
  --processed-dir "$NEW_PROC/" \
  --output "$NEW_PROC/segment_features.parquet"

python ecgclean/models/segment_quality.py predict \
  --segment-features "$NEW_PROC/segment_features.parquet" \
  --model models/segment_quality_v1.joblib \
  --output "$NEW_PROC/segment_quality_preds.parquet"

# ── Stage 6: Global template correlation ──────────────────────────────────────
echo "[6/10] Computing global template correlations..."
python ecgclean/features/global_templates.py correlate \
  --templates data/templates/global_templates.joblib \
  --peaks "$NEW_PROC/peaks.parquet" \
  --ecg-samples "$NEW_PROC/ecg_samples.parquet" \
  --output "$NEW_PROC/global_template_features.parquet"

# ── Stage 7: Merge features ────────────────────────────────────────────────────
echo "[7/10] Merging features..."
python scripts/merge_features_for_training.py \
  --beat-features "$NEW_PROC/beat_features.parquet" \
  --global-template-features "$NEW_PROC/global_template_features.parquet" \
  --output "$NEW_PROC/beat_features_merged.parquet"

# ── Stage 8: Tabular inference ─────────────────────────────────────────────────
echo "[8/10] Running tabular model inference..."
python ecgclean/models/beat_artifact_tabular.py predict \
  --beat-features "$NEW_PROC/beat_features_merged.parquet" \
  --model models/beat_tabular_v5_postcorrection.joblib \
  --output "$NEW_PROC/beat_tabular_preds.parquet"

# ── Stage 9: CNN inference ─────────────────────────────────────────────────────
echo "[9/10] Running CNN inference..."
python ecgclean/models/beat_artifact_cnn.py predict \
  --beat-features "$NEW_PROC/beat_features_merged.parquet" \
  --ecg-samples "$NEW_PROC/ecg_samples.parquet" \
  --model models/beat_cnn_v3.pt \
  --output "$NEW_PROC/beat_cnn_preds.parquet"

# ── Stage 10: Ensemble fuse + append to main dataset ──────────────────────────
echo "[10/10] Fusing ensemble predictions..."
python ecgclean/models/ensemble.py fuse \
  --tabular-preds "$NEW_PROC/beat_tabular_preds.parquet" \
  --cnn-preds "$NEW_PROC/beat_cnn_preds.parquet" \
  --output "$NEW_PROC/ensemble_preds.parquet" \
  --alpha 0.55

if [ "$SKIP_APPEND" -eq 0 ]; then
  echo "Appending to main dataset..."
  python - <<PYEOF
import pandas as pd, sys

new_path  = "data/new_processed/ensemble_preds.parquet"
main_path = "/Volumes/xHRV/processed/ensemble_preds.parquet"

new  = pd.read_parquet(new_path)
main = pd.read_parquet(main_path)

# Guard against re-ingesting the same files twice
overlap = set(new["timestamp_ns"]) & set(main["timestamp_ns"])
if overlap:
    print(f"  WARNING: {len(overlap):,} timestamps already in main dataset — skipping duplicates.")
    new = new[~new["timestamp_ns"].isin(overlap)]

if len(new) == 0:
    print("  Nothing new to append — all timestamps already present.")
    sys.exit(0)

combined = (
    pd.concat([main, new], ignore_index=True)
    .sort_values("timestamp_ns")
    .reset_index(drop=True)
)
combined.to_parquet(main_path, index=False)
print(f"  Appended {len(new):,} beats. Main dataset now: {len(combined):,} beats.")
PYEOF
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Done. New predictions: $NEW_PROC/ensemble_preds.parquet"
if [ "$SKIP_APPEND" -eq 0 ]; then
  echo "  Appended to:           $FULL_PROC/ensemble_preds.parquet"
fi
echo ""
echo "  Next step: move ingested CSVs to archive:"
echo "    mv $NEW_ECG_DIR/*.csv /Volumes/xHRV/ECG/"
echo "═══════════════════════════════════════════════════════════════════"
