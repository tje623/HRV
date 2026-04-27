#!/usr/bin/env bash
# rebuild_full_pipeline.sh
# Rebuilds ecg_samples.parquet, peaks.parquet, and all derived parquets from scratch.
#
# Usage:
#   bash Scripts/Oysters/rebuild_full_pipeline.sh               # full rebuild
#   bash Scripts/Oysters/rebuild_full_pipeline.sh --max-files 3 # first 3 files only
#   bash Scripts/Oysters/rebuild_full_pipeline.sh \
#       --ecg-dir Data/Subsets/foo/ECG \
#       --peaks-dir Data/Subsets/foo/Peaks \
#       --processed-dir Data/Subsets/foo/Processed

set -euo pipefail

# ── Resolve project root (grandparent of this script) ─────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# ── Activate virtualenv ────────────────────────────────────────────────────────
source /Users/tannereddy/.envs/hrv/bin/activate

# ── Defaults from config.py ────────────────────────────────────────────────────
_cfg_out="$(python - "${ROOT}" <<'PYEOF' 2>/dev/null
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "Scripts"))
try:
    from config import ECG_DIR, PEAKS_DIR, PROCESSED_DIR
    print(ECG_DIR)
    print(PEAKS_DIR)
    print(PROCESSED_DIR)
except Exception:
    sys.exit(1)
PYEOF
)" || true

if [ -n "$_cfg_out" ]; then
    ECG_DIR="$(echo "$_cfg_out" | sed -n '1p')"
    PEAKS_DIR="$(echo "$_cfg_out" | sed -n '2p')"
    PROCESSED_DIR="$(echo "$_cfg_out" | sed -n '3p')"
else
    ECG_DIR="${ROOT}/Data/ECG"
    PEAKS_DIR="${ROOT}/Data/Peaks"
    PROCESSED_DIR="${ROOT}/Processed"
fi

# Annotations file (load_annotations handles missing file gracefully)
ANNOTATIONS="${ROOT}/Data/Annotations/V1/V1.json"

# ── Parse arguments ────────────────────────────────────────────────────────────
MAX_FILES=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-files)     MAX_FILES="$2";     shift 2 ;;
        --ecg-dir)       ECG_DIR="$2";       shift 2 ;;
        --peaks-dir)     PEAKS_DIR="$2";     shift 2 ;;
        --processed-dir) PROCESSED_DIR="$2"; shift 2 ;;
        --annotations)   ANNOTATIONS="$2";   shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--max-files N] [--ecg-dir PATH] [--peaks-dir PATH] [--processed-dir PATH]"
            exit 1
            ;;
    esac
done

# ── Validate ───────────────────────────────────────────────────────────────────
if [ ! -d "$ECG_DIR" ]; then
    echo "ERROR: ECG directory not found: $ECG_DIR"
    exit 1
fi

mkdir -p "$PROCESSED_DIR"

# ── Build optional flags ───────────────────────────────────────────────────────
MAX_FILES_FLAG=""
if [ -n "$MAX_FILES" ]; then
    MAX_FILES_FLAG="--max-files $MAX_FILES"
    echo "  [DEBUG MODE] Processing first $MAX_FILES ECG file(s) only"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           HRV Pipeline — Full Rebuild                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "  ECG dir       : $ECG_DIR"
echo "  Peaks dir     : $PEAKS_DIR"
echo "  Processed dir : $PROCESSED_DIR"
echo ""

# ── Step 1: Regenerate peak CSVs ──────────────────────────────────────────────
echo "=== Step 1/5: detect_peaks.py ==="
PEAKS_OUT="${PROCESSED_DIR}/new_peaks"
mkdir -p "$PEAKS_OUT"
rm -f "${PEAKS_OUT}"/*.csv 2>/dev/null || true
python Scripts/detect_peaks.py \
    --ecg-dir "$ECG_DIR" \
    --output-dir "$PEAKS_OUT" \
    --method ensemble \
    --fs 130 \
    $MAX_FILES_FLAG

# ── Step 2: Rebuild ECG parquets ─────────────────────────────────────────────
echo ""
echo "=== Step 2/5: data_pipeline.py ==="
rm -f "${PROCESSED_DIR}/ecg_samples.parquet" \
      "${PROCESSED_DIR}/peaks.parquet" \
      "${PROCESSED_DIR}/labels.parquet" \
      "${PROCESSED_DIR}/segments.parquet"
python Scripts/data_pipeline.py \
    --ecg-dir "$ECG_DIR" \
    --peaks-dir "${PEAKS_DIR}" \
    --annotations "$ANNOTATIONS" \
    --output-dir "$PROCESSED_DIR" \
    --workers 10 \
    $MAX_FILES_FLAG

# ── Step 3: Physio constraints ────────────────────────────────────────────────
echo ""
echo "=== Step 3/5: physio_constraints.py ==="
rm -f "${PROCESSED_DIR}/physio_constraints_done.flag"
python Scripts/physio_constraints.py --processed-dir "$PROCESSED_DIR"
touch "${PROCESSED_DIR}/physio_constraints_done.flag"

# ── Step 4: Segment features + quality predictions ────────────────────────────
echo ""
echo "=== Step 4/5: segment_features.py + segment_quality predictions ==="
rm -f "${PROCESSED_DIR}/segment_features.parquet" \
      "${PROCESSED_DIR}/segment_quality_preds.parquet"
python Scripts/features/segment_features.py \
    --processed-dir "$PROCESSED_DIR" \
    --output "${PROCESSED_DIR}/segment_features.parquet"
python Scripts/models/segment_quality.py predict \
    --segment-features "${PROCESSED_DIR}/segment_features.parquet" \
    --model "${ROOT}/Models/segment_quality_v1.joblib" \
    --output "${PROCESSED_DIR}/segment_quality_preds.parquet"

# ── Step 5: Beat features ─────────────────────────────────────────────────────
echo ""
echo "=== Step 5/5: beat_features.py ==="
rm -f "${PROCESSED_DIR}/beat_features.parquet"
python Scripts/features/beat_features.py \
    --processed-dir "$PROCESSED_DIR" \
    --segment-quality-preds "${PROCESSED_DIR}/segment_quality_preds.parquet" \
    --output "${PROCESSED_DIR}/beat_features.parquet"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Done. Results in: ${PROCESSED_DIR}  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
