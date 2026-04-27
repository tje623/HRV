#!/usr/bin/env bash
# run_pipeline_logged.sh — Run the artifact-detection pipeline with per-step
# audit logs and a final summary.json.
#
# Usage:
#   bash Scripts/utils/run_pipeline_logged.sh --subset smoke_test
#   bash Scripts/utils/run_pipeline_logged.sh \
#       --processed-dir Data/Subsets/foo/Processed \
#       --ecg-dir Data/Subsets/foo/ECG \
#       --peaks-dir Data/Subsets/foo/Peaks
#
# Flags:
#   --subset <name>          Resolve ECG/Peaks/Processed under Data/Subsets/<name>/
#   --processed-dir <path>   Override processed output directory
#   --ecg-dir <path>         Override ECG input directory
#   --peaks-dir <path>       Override Peaks input directory
#   --annotations <path>     Annotation JSON (default: Data/Annotations/V1/V1.json)
#   --max-files <N>          Pass --max-files N to detect_peaks + data_pipeline

set -uo pipefail

# ── Resolve project root ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT"

# ── Activate virtualenv ────────────────────────────────────────────────────────
VENV="/Users/tannereddy/.envs/hrv/bin/activate"
[ -f "$VENV" ] && source "$VENV"

# ── Defaults ───────────────────────────────────────────────────────────────────
SUBSET_NAME=""
PROCESSED_DIR=""
ECG_DIR=""
PEAKS_DIR=""
ANNOTATIONS="${ROOT}/Data/Annotations/V1/V1.json"
MAX_FILES=""

# ── Parse arguments ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subset)        SUBSET_NAME="$2";    shift 2 ;;
        --processed-dir) PROCESSED_DIR="$2";  shift 2 ;;
        --ecg-dir)       ECG_DIR="$2";        shift 2 ;;
        --peaks-dir)     PEAKS_DIR="$2";      shift 2 ;;
        --annotations)   ANNOTATIONS="$2";    shift 2 ;;
        --max-files)     MAX_FILES="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Resolve paths from --subset ───────────────────────────────────────────────
if [ -n "$SUBSET_NAME" ]; then
    SUBSET_BASE="${ROOT}/Data/Subsets/${SUBSET_NAME}"
    [ -z "$ECG_DIR" ]       && ECG_DIR="${SUBSET_BASE}/ECG"
    [ -z "$PEAKS_DIR" ]     && PEAKS_DIR="${SUBSET_BASE}/Peaks"
    [ -z "$PROCESSED_DIR" ] && PROCESSED_DIR="${SUBSET_BASE}/Processed"
fi

if [ -z "$PROCESSED_DIR" ] || [ -z "$ECG_DIR" ] || [ -z "$PEAKS_DIR" ]; then
    echo "ERROR: Provide --subset <name> or all of --ecg-dir / --peaks-dir / --processed-dir"
    exit 1
fi

mkdir -p "$PROCESSED_DIR"

# ── Create run log directory ───────────────────────────────────────────────────
RUN_TS="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_LOG_DIR="${ROOT}/Docs/run_logs/${RUN_TS}"
mkdir -p "$RUN_LOG_DIR"

OVERALL_EXIT=0

# ── write_summary: aggregate step manifests → summary.json ────────────────────
write_summary() {
    local log_dir="$1"
    local overall="$2"
    python - "$log_dir" "$overall" <<'PYEOF'
import json, sys
from pathlib import Path
from datetime import datetime

log_dir = Path(sys.argv[1])
overall_exit = int(sys.argv[2])

manifests = sorted(log_dir.glob("*.manifest.json"))
steps = []
for mp in manifests:
    try:
        data = json.loads(mp.read_text())
    except Exception as exc:
        steps.append({"manifest": str(mp), "error": str(exc)})
        continue

    started = data.get("started_at_utc", "")
    finished = data.get("finished_at_utc", "")
    duration_s = None
    if started and finished:
        try:
            s = datetime.fromisoformat(started)
            f = datetime.fromisoformat(finished)
            duration_s = round((f - s).total_seconds(), 1)
        except Exception:
            pass

    output_sizes = {
        p["path"]: p.get("size_bytes")
        for p in data.get("outputs_after", [])
        if p.get("exists")
    }

    steps.append({
        "name": data.get("name"),
        "exit_code": data.get("exit_code"),
        "started_at_utc": started,
        "finished_at_utc": finished,
        "duration_s": duration_s,
        "output_sizes_bytes": output_sizes,
        "stdout_log": data.get("stdout_log"),
        "stderr_log": data.get("stderr_log"),
    })

summary = {
    "run_timestamp": log_dir.name,
    "overall_exit_code": overall_exit,
    "steps": steps,
}
out = log_dir / "summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(f"[summary] {out}")
PYEOF
}

# ── run_step: wrap one command with run_audit.py ──────────────────────────────
run_step() {
    local step_name="$1"
    shift
    echo "=== ${step_name} ==="
    python "${ROOT}/Scripts/utils/run_audit.py" \
        --name "$step_name" \
        --log-dir "$RUN_LOG_DIR" \
        -- "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  [FAILED] ${step_name} exited ${rc}"
        OVERALL_EXIT=$rc
    else
        echo "  [OK] ${step_name}"
    fi
    return $rc
}

# ── Optional --max-files flag ─────────────────────────────────────────────────
MF_FLAG=""
[ -n "$MAX_FILES" ] && MF_FLAG="--max-files ${MAX_FILES}"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  HRV Artifact Pipeline — Logged Run: ${RUN_TS}         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  ECG dir       : $ECG_DIR"
echo "  Peaks dir     : $PEAKS_DIR"
echo "  Processed dir : $PROCESSED_DIR"
echo "  Log dir       : $RUN_LOG_DIR"
echo ""

# ── Step 1: detect_peaks ───────────────────────────────────────────────────────
PEAKS_DETECT_OUT="${PROCESSED_DIR}/new_peaks"
mkdir -p "$PEAKS_DETECT_OUT"

if ! run_step "detect_peaks" \
    python "${ROOT}/Scripts/detect_peaks.py" \
        --ecg-dir "$ECG_DIR" \
        --output-dir "$PEAKS_DETECT_OUT" \
        --method ensemble \
        --fs 130 \
        $MF_FLAG; then
    write_summary "$RUN_LOG_DIR" "$OVERALL_EXIT"
    exit "$OVERALL_EXIT"
fi

# ── Step 2: data_pipeline ─────────────────────────────────────────────────────
if ! run_step "data_pipeline" \
    python "${ROOT}/Scripts/data_pipeline.py" \
        --ecg-dir "$ECG_DIR" \
        --peaks-dir "${PEAKS_DETECT_OUT}" \
        --annotations "$ANNOTATIONS" \
        --output-dir "$PROCESSED_DIR" \
        --workers 8 \
        $MF_FLAG; then
    write_summary "$RUN_LOG_DIR" "$OVERALL_EXIT"
    exit "$OVERALL_EXIT"
fi

# ── Step 3: physio_constraints ────────────────────────────────────────────────
if ! run_step "physio_constraints" \
    python "${ROOT}/Scripts/physio_constraints.py" \
        --processed-dir "$PROCESSED_DIR"; then
    write_summary "$RUN_LOG_DIR" "$OVERALL_EXIT"
    exit "$OVERALL_EXIT"
fi

# ── Step 4: segment_features ──────────────────────────────────────────────────
SEG_FEAT="${PROCESSED_DIR}/segment_features.parquet"
if ! run_step "segment_features" \
    python "${ROOT}/Scripts/features/segment_features.py" \
        --processed-dir "$PROCESSED_DIR" \
        --output "$SEG_FEAT"; then
    write_summary "$RUN_LOG_DIR" "$OVERALL_EXIT"
    exit "$OVERALL_EXIT"
fi

# ── Step 5: segment_quality_predict ───────────────────────────────────────────
SEG_QUAL="${PROCESSED_DIR}/segment_quality_preds.parquet"
SEG_MODEL="${ROOT}/Models/segment_quality_v1.joblib"
if ! run_step "segment_quality_predict" \
    python "${ROOT}/Scripts/models/segment_quality.py" predict \
        --segment-features "$SEG_FEAT" \
        --model "$SEG_MODEL" \
        --output "$SEG_QUAL"; then
    write_summary "$RUN_LOG_DIR" "$OVERALL_EXIT"
    exit "$OVERALL_EXIT"
fi

# ── Step 6: beat_features ─────────────────────────────────────────────────────
BEAT_FEAT="${PROCESSED_DIR}/beat_features.parquet"
if ! run_step "beat_features" \
    python "${ROOT}/Scripts/features/beat_features.py" \
        --processed-dir "$PROCESSED_DIR" \
        --segment-quality-preds "$SEG_QUAL" \
        --output "$BEAT_FEAT"; then
    write_summary "$RUN_LOG_DIR" "$OVERALL_EXIT"
    exit "$OVERALL_EXIT"
fi

# ── Step 7: beat_tabular_train ────────────────────────────────────────────────
BEAT_MODEL_OUT="${PROCESSED_DIR}/beat_tabular_trained.joblib"
if ! run_step "beat_tabular_train" \
    python "${ROOT}/Scripts/models/beat_artifact_tabular.py" train \
        --beat-features "$BEAT_FEAT" \
        --labels "${PROCESSED_DIR}/labels.parquet" \
        --segment-quality-preds "$SEG_QUAL" \
        --output "$BEAT_MODEL_OUT"; then
    write_summary "$RUN_LOG_DIR" "$OVERALL_EXIT"
    exit "$OVERALL_EXIT"
fi

# ── Step 8: beat_tabular_predict ──────────────────────────────────────────────
BEAT_PREDS="${PROCESSED_DIR}/beat_tabular_preds.parquet"
BEAT_MODEL="${ROOT}/Models/beat_tabular_v1.joblib"
run_step "beat_tabular_predict" \
    python "${ROOT}/Scripts/models/beat_artifact_tabular.py" predict \
        --beat-features "$BEAT_FEAT" \
        --model "$BEAT_MODEL" \
        --output "$BEAT_PREDS" || true

# ── Write summary ─────────────────────────────────────────────────────────────
write_summary "$RUN_LOG_DIR" "$OVERALL_EXIT"

echo ""
if [ "$OVERALL_EXIT" -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  ALL STEPS PASSED                                                    ║"
    echo "║  Logs: Docs/run_logs/${RUN_TS}/                   ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
else
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  PIPELINE FAILED (exit ${OVERALL_EXIT})                                     ║"
    echo "║  Logs: Docs/run_logs/${RUN_TS}/                   ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
fi
echo ""

exit "$OVERALL_EXIT"
