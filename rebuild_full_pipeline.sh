#!/usr/bin/env bash
# rebuild_full_pipeline.sh
# Rebuilds ecg_samples.parquet, peaks.parquet, and all derived parquets
# from scratch with three critical fixes applied:
#   1. detect_peaks.py now runs at correct --fs 130 (Polar H10 actual rate)
#   2. Pan-Tompkins + SWT both snap detected peaks to argmax(|ecg|) within ±60ms
#   3. data_pipeline.py detects and flips inverted ECG files before staging
#
# Safe to re-run: each rm is explicit; does NOT touch model files or annotations.
#
# Usage:
#   ./rebuild_full_pipeline.sh                  # full rebuild (all files)
#   ./rebuild_full_pipeline.sh --max-files 3    # debug: first 3 ECG files only

set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────────────
MAX_FILES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--max-files N]"
            exit 1
            ;;
    esac
done

# Build optional flags for python scripts
MAX_FILES_FLAG=""
if [ -n "$MAX_FILES" ]; then
    MAX_FILES_FLAG="--max-files $MAX_FILES"
    echo "  [DEBUG MODE] Processing first $MAX_FILES ECG file(s) only"
fi

cd "/Volumes/xHRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate

OUT="/Volumes/xHRV/processed"

if [ ! -d "$OUT" ]; then
    echo "ERROR: $OUT not found. Is xHRV mounted?"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           HRV Pipeline — Full Rebuild (fixes applied)           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Regenerate peak CSVs at correct sample rate ───────────────────
echo "=== Step 1/5: detect_peaks.py (--fs 130, argmax snap) ==="
rm -f data/new_peaks/*.csv
python ecgclean/detect_peaks.py \
    --ecg-dir /Volumes/xHRV/ECG \
    --output-dir data/new_peaks/ \
    --method ensemble \
    --fs 130 \
    $MAX_FILES_FLAG

# ── Step 2: Rebuild ECG parquets (polarity fix + corrected peaks) ─────────
echo ""
echo "=== Step 2/5: data_pipeline.py (polarity correction per file) ==="
rm -f "$OUT/ecg_samples.parquet" \
      "$OUT/peaks.parquet" \
      "$OUT/labels.parquet" \
      "$OUT/segments.parquet"
python ecgclean/data_pipeline.py \
    --ecg-dir /Volumes/xHRV/ECG \
    --peaks-dir data/new_peaks/ \
    --annotations data/annotations/artifact_annotations.json \
    --output-dir "$OUT" \
    --workers 10 \
    $MAX_FILES_FLAG

# ── Step 3: Physio constraints ────────────────────────────────────────────
echo ""
echo "=== Step 3/5: physio_constraints.py ==="
rm -f "$OUT/physio_constraints_done.flag"
python ecgclean/physio_constraints.py --processed-dir "$OUT"
touch "$OUT/physio_constraints_done.flag"

# ── Step 4: Beat features (corrected window centering via argmax snap) ─────
echo ""
echo "=== Step 4/5: beat_features.py (argmax window snap) ==="
rm -f "$OUT/beat_features.parquet"
python ecgclean/features/beat_features.py \
    --processed-dir "$OUT" \
    --output "$OUT/beat_features.parquet"

# ── Step 5: Segment features + quality predictions ────────────────────────
echo ""
echo "=== Step 5/5: segment_features.py + segment_quality predictions ==="
rm -f "$OUT/segment_features.parquet" "$OUT/segment_quality_preds.parquet"
python ecgclean/features/segment_features.py \
    --processed-dir "$OUT" \
    --output "$OUT/segment_features.parquet"
python ecgclean/models/segment_quality.py predict \
    --segment-features "$OUT/segment_features.parquet" \
    --model models/segment_quality_v1.joblib \
    --output "$OUT/segment_quality_preds.parquet"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Done. Annotate with marker_viewer.py, then run retrain_pipeline.sh ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
