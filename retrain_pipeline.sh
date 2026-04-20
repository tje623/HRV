#!/usr/bin/env bash
# ============================================================
#  retrain_pipeline.sh
#  Full retraining pipeline after active learning sessions.
#
#  Steps:
#    0a. Re-run physio_constraints.py on full dataset
#        → adds tachy_transition_candidate, reduces physio_implausible false positives
#    0b. Patch updated physio columns into full-dataset beat_features.parquet
#        → avoids full beat_features.py re-run (~20-40 min streaming patch vs hours)
#    1. Merge all annotation queue completed.csv → combined labels.parquet
#    2. Extract ECG + features for newly labeled beats from external drive
#       and build a combined data/training/ directory
#    3. Retrain tabular model   (~2-5 min, LightGBM)
#    4. Retrain CNN model       (~1-2 h, Apple Silicon MPS)
#    5. Full-dataset inference  (tabular: ~5 min | CNN: ~1-2 h, MPS)
#    6. Re-run ensemble         (alpha=0.55 → tabular 55%, CNN 45%)
#
#  Prerequisites:
#    - External drive /Volumes/xHRV must be mounted
#    - hrv venv activated (or let the script do it)
#    - Run from any directory — script cds to the project root
# ============================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT="/Users/tannereddy/Desktop/Cabin/Python/Projects/hrv/Artifact Detector"
ENV="/Users/tannereddy/.envs/hrv/bin/activate"
PROCESSED_LOCAL="$PROJECT/data/processed"
PROCESSED_FULL="/Volumes/xHRV/processed"
QUEUES_DIR="$PROJECT/data/annotation_queues"
TRAIN_DIR="$PROJECT/data/training"   # combined training data written here
MODELS_DIR="$PROJECT/models"

cd "$PROJECT"
source "$ENV"
mkdir -p "$TRAIN_DIR" "$MODELS_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -d "$PROCESSED_FULL" ]; then
    echo "ERROR: External drive not mounted — $PROCESSED_FULL not found."
    echo "       Connect xHRV and re-run."
    exit 1
fi
if [ ! -f "$PROCESSED_LOCAL/labels.parquet" ]; then
    echo "ERROR: $PROCESSED_LOCAL/labels.parquet not found."
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          HRV Artifact Detector — Retraining Pipeline            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "  Project:        $PROJECT"
echo "  Full dataset:   $PROCESSED_FULL"
echo "  Training data:  $TRAIN_DIR"
echo "  Models out:     $MODELS_DIR"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 0a/6 — Re-run physio_constraints on full dataset"
echo "              (adds tachy_transition_candidate, fixes physio_implausible)"
echo "════════════════════════════════════════════════════════════════════"
# ══════════════════════════════════════════════════════════════════════════════

python3 ecgclean/physio_constraints.py \
    --processed-dir "$PROCESSED_FULL"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 0b/6 — Patch physio columns into full beat_features.parquet"
echo "              (streaming patch: no ECG re-extraction needed)"
echo "════════════════════════════════════════════════════════════════════"
# ══════════════════════════════════════════════════════════════════════════════

python3 - <<PYEOF
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

full     = Path("$PROCESSED_FULL")
bf_path  = full / "beat_features.parquet"
tmp_path = full / "beat_features_patched.parquet"

# Load only the physio columns from the freshly-updated labels.parquet.
# 3 cols × 50.5M rows ≈ ~600 MB in RAM — acceptable.
print("  Loading updated physio columns from labels.parquet ...")
new_physio = pq.read_table(
    full / "labels.parquet",
    columns=["peak_id", "physio_implausible", "tachy_transition_candidate",
             "review_priority_score"],
).to_pandas()
new_physio["peak_id"] = new_physio["peak_id"].astype("int64")
new_physio = new_physio.set_index("peak_id")
print(f"  Loaded {len(new_physio):,} physio rows")

# Build output schema: existing schema + tachy_transition_candidate if absent.
schema = pq.read_schema(bf_path)
if "tachy_transition_candidate" not in schema.names:
    schema = schema.append(pa.field("tachy_transition_candidate", pa.int32()))
    print("  Schema: added tachy_transition_candidate field")
else:
    print("  Schema: tachy_transition_candidate already present")

pf       = pq.ParquetFile(bf_path)
rows_done = 0
with pq.ParquetWriter(tmp_path, schema=schema, compression="snappy") as writer:
    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        if df.index.name == "peak_id":
            df = df.reset_index()
        df["peak_id"] = df["peak_id"].astype("int64")

        # Overwrite physio columns using fresh labels data.
        updated = new_physio[
            ["physio_implausible", "tachy_transition_candidate", "review_priority_score"]
        ].reindex(df["peak_id"])

        df["physio_implausible"]      = updated["physio_implausible"].values.astype(np.int32)
        df["tachy_transition_candidate"] = (
            updated["tachy_transition_candidate"].fillna(0).values.astype(np.int32)
        )
        df["review_priority_score"]   = (
            updated["review_priority_score"].fillna(0.0).values.astype(np.float32)
        )

        # Ensure all schema columns exist (fill any extras with 0).
        for field in schema:
            if field.name not in df.columns:
                df[field.name] = pa.array(
                    np.zeros(len(df), dtype=np.int32 if pa.types.is_integer(field.type)
                             else np.float32),
                    type=field.type,
                )

        writer.write_table(
            pa.Table.from_pandas(df[schema.names], preserve_index=False, schema=schema)
        )
        rows_done += len(df)
        if rows_done % 5_000_000 == 0:
            print(f"  Patched {rows_done:,} rows ...")

print(f"  Patch complete: {rows_done:,} total rows written")

# Atomic replace.
tmp_path.rename(bf_path)
print(f"  Replaced beat_features.parquet ({bf_path.stat().st_size / 1e6:,.0f} MB)")
PYEOF

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 1/6 — Merge annotation sessions → labels.parquet"
echo "════════════════════════════════════════════════════════════════════"
# ══════════════════════════════════════════════════════════════════════════════

python3 - <<PYEOF
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

queues_dir   = Path("$QUEUES_DIR")
orig_path    = Path("$PROCESSED_LOCAL/labels.parquet")
out_path     = Path("$TRAIN_DIR/labels.parquet")

# ── Collect all completed.csv across sessions ──────────────────────────────
rows = []
for session in sorted(queues_dir.iterdir()):
    completed = session / "completed.csv"
    if not completed.exists():
        continue
    df = pd.read_csv(completed)
    if "peak_id" not in df.columns or "label" not in df.columns:
        print(f"  SKIP {session.name}: unexpected columns {list(df.columns)}")
        continue
    # Drop skipped / internal markers
    df = df[~df["label"].astype(str).str.startswith("__")].copy()
    if len(df) == 0:
        continue
    df["_session"] = session.name
    rows.append(df[["peak_id", "label", "_session"]])
    print(f"  {session.name:<35s} {len(df):>5d} annotations")

if not rows:
    print("ERROR: No completed.csv files found — nothing to merge.")
    sys.exit(1)

new_df = pd.concat(rows, ignore_index=True)
# Last session listed wins when the same beat appears in multiple sessions
new_df = new_df.drop_duplicates(subset="peak_id", keep="last")[["peak_id", "label"]]
new_df["peak_id"] = new_df["peak_id"].astype("int64")

print(f"\n  New annotations (deduplicated): {len(new_df):,} beats")
print(new_df["label"].value_counts().rename("count").to_frame().to_string())

# ── Load original labels from data_pipeline ────────────────────────────────
orig = pd.read_parquet(orig_path)
if orig.index.name == "peak_id":
    orig = orig.reset_index()
orig = orig[["peak_id", "label"]].copy()
orig["peak_id"] = orig["peak_id"].astype("int64")
print(f"\n  Original labels: {len(orig):,} beats")
print(orig["label"].value_counts().rename("count").to_frame().to_string())

# ── Merge: new annotations override originals for the same peak_id ─────────
overlap = len(set(orig["peak_id"]) & set(new_df["peak_id"]))
print(f"\n  Overlapping peak_ids (new overrides original): {overlap}")

combined = pd.concat([
    orig[~orig["peak_id"].isin(set(new_df["peak_id"]))],
    new_df,
], ignore_index=True)
combined["peak_id"] = combined["peak_id"].astype("int64")

print(f"\n  Combined labels: {len(combined):,} beats")
print(combined["label"].value_counts().rename("count").to_frame().to_string())

pq.write_table(
    pa.Table.from_pandas(combined, preserve_index=False),
    out_path,
    compression="snappy",
)
print(f"\n  Saved → {out_path}")
PYEOF

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 2/6 — Build combined training data directory"
echo "             (streaming extraction — no full-dataset loads)"
echo "════════════════════════════════════════════════════════════════════"
# ══════════════════════════════════════════════════════════════════════════════

python3 - <<PYEOF
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

local     = Path("$PROCESSED_LOCAL")
full      = Path("$PROCESSED_FULL")
train_dir = Path("$TRAIN_DIR")

# ── Which peak_ids need to come from the external drive? ──────────────────
labels = pd.read_parquet(train_dir / "labels.parquet")
labels["peak_id"] = labels["peak_id"].astype("int64")
all_labeled_ids = set(labels["peak_id"].tolist())

# local peaks.parquet is small (2.4 MB) — safe to load
local_peaks = pd.read_parquet(local / "peaks.parquet")
if local_peaks.index.name == "peak_id":
    local_peaks = local_peaks.reset_index()
local_peaks["peak_id"] = local_peaks["peak_id"].astype("int64")
local_ids = set(local_peaks["peak_id"].tolist())

new_ids = all_labeled_ids - local_ids
print(f"Total labeled peaks:      {len(all_labeled_ids):,}")
print(f"Already in local data:    {len(all_labeled_ids & local_ids):,}")
print(f"Need external extraction: {len(new_ids):,}")

if len(new_ids) == 0:
    for fname in ["ecg_samples.parquet", "beat_features.parquet",
                  "peaks.parquet", "segment_quality_preds.parquet"]:
        dst = train_dir / fname
        if not dst.exists():
            shutil.copy2(str(local / fname), str(dst))
            print(f"  Copied {fname}")
    print("\nAll labeled peaks already in local data — nothing to extract.")

else:
    # ── 1. Stream full peaks.parquet to locate new peaks ──────────────────
    # Bounded memory: 1 M rows × 3 cols × 8 bytes = ~24 MB per batch
    print(f"\n[1/4] Scanning peaks.parquet for {len(new_ids):,} new peaks ...")
    pf_peaks = pq.ParquetFile(full / "peaks.parquet")
    needed   = set(new_ids)          # shrinks as we find peaks
    found_rows: list[pd.DataFrame] = []
    batches_scanned = 0
    for batch in pf_peaks.iter_batches(
        batch_size=1_000_000,
        columns=["peak_id", "timestamp_ns", "segment_idx"],
    ):
        df = batch.to_pandas()
        df["peak_id"] = df["peak_id"].astype("int64")
        hits = df[df["peak_id"].isin(needed)]
        if len(hits):
            found_rows.append(hits)
            needed -= set(hits["peak_id"].tolist())
        batches_scanned += 1
        if batches_scanned % 10 == 0:
            print(f"  {batches_scanned * 1_000_000:,} rows scanned, {len(needed):,} still needed")
        if not needed:
            print(f"  All peaks found — stopping early at batch {batches_scanned}")
            break

    new_peaks_df = (
        pd.concat(found_rows, ignore_index=True) if found_rows
        else pd.DataFrame(columns=["peak_id", "timestamp_ns", "segment_idx"])
    )
    found_ids   = set(new_peaks_df["peak_id"].tolist())
    missing_ids = new_ids - found_ids
    if missing_ids:
        print(f"  WARNING: {len(missing_ids)} labeled peak_ids not in full peaks.parquet")
    new_seg_ids = sorted(new_peaks_df["segment_idx"].unique().tolist())
    print(f"  Located {len(new_peaks_df):,} peaks across {len(new_seg_ids):,} segments")

    # ── 2. Stream ECG — write directly to ParquetWriter ───────────────────
    # Never accumulate the full combined ECG in RAM.
    # Write local ECG first (iter_batches), then append new segments.
    print(f"\n[2/4] Writing combined ecg_samples.parquet ...")
    ECG_COLS = ["timestamp_ns", "ecg", "segment_idx"]
    local_ecg_schema = pq.read_schema(local / "ecg_samples.parquet")
    write_cols = [f.name for f in local_ecg_schema if f.name in ECG_COLS]
    out_schema = pa.schema([local_ecg_schema.field(c) for c in write_cols])

    with pq.ParquetWriter(train_dir / "ecg_samples.parquet",
                          schema=out_schema, compression="snappy") as writer:
        # Pass 1: copy local ECG in streaming batches
        rows_local = 0
        for batch in pq.ParquetFile(local / "ecg_samples.parquet").iter_batches(
            batch_size=500_000, columns=write_cols
        ):
            writer.write_batch(batch)
            rows_local += batch.num_rows
        print(f"  Local ECG:  {rows_local:,} rows written")

        # Pass 2: append new segments from external drive (predicate pushdown)
        rows_new = 0
        BATCH_SEGS = 50          # ~50 segs × 7,800 samples × 12 B ≈ 5 MB each
        for i in range(0, len(new_seg_ids), BATCH_SEGS):
            batch_segs = new_seg_ids[i : i + BATCH_SEGS]
            tbl = pq.read_table(
                full / "ecg_samples.parquet",
                filters=[("segment_idx", "in", batch_segs)],
                columns=write_cols,
            )
            writer.write_table(tbl)
            rows_new += tbl.num_rows
            done = min(i + BATCH_SEGS, len(new_seg_ids))
            if done % 500 == 0 or done == len(new_seg_ids):
                print(f"  New ECG:    {done:,}/{len(new_seg_ids):,} segments")

    mb = (train_dir / "ecg_samples.parquet").stat().st_size / 1e6
    print(f"  ecg_samples.parquet → {mb:,.0f} MB  "
          f"({rows_local + rows_new:,} rows total)")

    # ── 3. Stream beat_features — write directly to ParquetWriter ─────────
    # local beat_features is small (10 MB) — load once, write, del.
    # full beat_features is huge — stream 500 K rows at a time, keep only hits.
    print(f"\n[3/4] Writing combined beat_features.parquet ...")
    local_bf = pd.read_parquet(local / "beat_features.parquet")
    if local_bf.index.name == "peak_id":
        local_bf = local_bf.reset_index()
    local_bf["peak_id"] = local_bf["peak_id"].astype("int64")

    bf_schema_full = pq.read_schema(full / "beat_features.parquet")
    # Use the full-dataset schema (superset); local schema might be older
    with pq.ParquetWriter(train_dir / "beat_features.parquet",
                          schema=bf_schema_full, compression="snappy") as writer:
        # Write local beats (fill any missing columns with 0)
        local_tbl = pa.Table.from_pandas(local_bf, preserve_index=False)
        for field in bf_schema_full:
            if field.name not in local_tbl.schema.names:
                local_tbl = local_tbl.append_column(
                    field.name,
                    pa.array([0] * len(local_tbl), type=field.type),
                )
        local_tbl = local_tbl.select(bf_schema_full.names)
        writer.write_table(local_tbl)
        print(f"  Local beats: {len(local_tbl):,} rows written")
        del local_bf, local_tbl

        # Stream full beat_features, emit only found_ids rows
        pf_bf    = pq.ParquetFile(full / "beat_features.parquet")
        needed_b = set(found_ids)   # peak_ids still to find
        rows_new_bf = 0
        for batch in pf_bf.iter_batches(batch_size=500_000):
            df = batch.to_pandas()
            if df.index.name == "peak_id":
                df = df.reset_index()
            df["peak_id"] = df["peak_id"].astype("int64")
            hits = df[df["peak_id"].isin(needed_b)]
            if len(hits):
                writer.write_table(
                    pa.Table.from_pandas(hits, preserve_index=False)
                    .select(bf_schema_full.names)
                )
                rows_new_bf += len(hits)
                needed_b -= set(hits["peak_id"].tolist())
            if not needed_b:
                break   # early exit

    mb = (train_dir / "beat_features.parquet").stat().st_size / 1e6
    print(f"  beat_features.parquet → {mb:,.0f} MB  ({rows_new_bf:,} new beats added)")

    # ── 4. Combine small files (all fit comfortably in RAM) ───────────────
    print(f"\n[4/4] Combining peaks.parquet + segment_quality_preds.parquet ...")

    # peaks
    shared = [c for c in local_peaks.columns if c in new_peaks_df.columns]
    combined_peaks = pd.concat([
        local_peaks[~local_peaks["peak_id"].isin(found_ids)],
        new_peaks_df[shared],
    ], ignore_index=True)
    pq.write_table(
        pa.Table.from_pandas(combined_peaks, preserve_index=False),
        train_dir / "peaks.parquet", compression="snappy",
    )
    print(f"  peaks.parquet → {len(combined_peaks):,} peaks")

    # segment_quality_preds — also small
    local_sqp = pd.read_parquet(local / "segment_quality_preds.parquet")
    local_sqp["segment_idx"] = local_sqp["segment_idx"].astype("int64")
    full_sqp  = pd.read_parquet(full / "segment_quality_preds.parquet")
    full_sqp["segment_idx"]  = full_sqp["segment_idx"].astype("int64")
    new_sqp   = full_sqp[full_sqp["segment_idx"].isin(set(new_seg_ids))]
    combined_sqp = pd.concat([
        local_sqp[~local_sqp["segment_idx"].isin(set(new_seg_ids))],
        new_sqp,
    ], ignore_index=True)
    pq.write_table(
        pa.Table.from_pandas(combined_sqp, preserve_index=False),
        train_dir / "segment_quality_preds.parquet", compression="snappy",
    )
    print(f"  segment_quality_preds.parquet → {len(combined_sqp):,} segments")

print("\n  Training directory contents:")
for f in sorted(Path("$TRAIN_DIR").glob("*.parquet")):
    mb = f.stat().st_size / 1e6
    print(f"    {f.name:<45s} {mb:8.1f} MB")
PYEOF

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 3/6 — Retrain tabular model (LightGBM, ~2-5 min)"
echo "════════════════════════════════════════════════════════════════════"
# ══════════════════════════════════════════════════════════════════════════════

python3 ecgclean/models/beat_artifact_tabular.py train \
    --beat-features         "$TRAIN_DIR/beat_features.parquet" \
    --labels                "$TRAIN_DIR/labels.parquet" \
    --segment-quality-preds "$TRAIN_DIR/segment_quality_preds.parquet" \
    --output                "$MODELS_DIR/beat_tabular_v3.joblib"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 4/6 — Retrain CNN model (PyTorch Lightning, MPS, ~1-2 h)"
echo "════════════════════════════════════════════════════════════════════"
# ══════════════════════════════════════════════════════════════════════════════

# peaks.parquet must live alongside beat_features.parquet —
# the train() function resolves it as bf_path.parent / "peaks.parquet"
python3 ecgclean/models/beat_artifact_cnn.py train \
    --beat-features         "$TRAIN_DIR/beat_features.parquet" \
    --labels                "$TRAIN_DIR/labels.parquet" \
    --ecg-samples           "$TRAIN_DIR/ecg_samples.parquet" \
    --segment-quality-preds "$TRAIN_DIR/segment_quality_preds.parquet" \
    --output                "$MODELS_DIR/beat_cnn_v2.pt"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 5/6 — Full-dataset inference (50.5M beats)"
echo "════════════════════════════════════════════════════════════════════"
# ══════════════════════════════════════════════════════════════════════════════

echo "  5a: Tabular predictions (~5 min) ..."
python3 ecgclean/models/beat_artifact_tabular.py predict \
    --beat-features "$PROCESSED_FULL/beat_features.parquet" \
    --model         "$MODELS_DIR/beat_tabular_v3.joblib" \
    --output        "$PROCESSED_FULL/beat_tabular_preds.parquet"

echo ""
echo "  5b: CNN predictions (~1-2 h on MPS) ..."
# peaks.parquet is resolved from beat_features parent: /Volumes/xHRV/processed/peaks.parquet
python3 ecgclean/models/beat_artifact_cnn.py predict \
    --beat-features "$PROCESSED_FULL/beat_features.parquet" \
    --ecg-samples   "$PROCESSED_FULL/ecg_samples.parquet" \
    --model         "$MODELS_DIR/beat_cnn_v2.pt" \
    --output        "$PROCESSED_FULL/beat_cnn_preds.parquet"

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  STEP 6/6 — Ensemble fusion  (alpha=0.55: tabular=55%, CNN=45%)"
echo "════════════════════════════════════════════════════════════════════"
# ══════════════════════════════════════════════════════════════════════════════

python3 ecgclean/models/ensemble.py fuse \
    --tabular-preds "$PROCESSED_FULL/beat_tabular_preds.parquet" \
    --cnn-preds     "$PROCESSED_FULL/beat_cnn_preds.parquet" \
    --output        "$PROCESSED_FULL/ensemble_preds.parquet" \
    --alpha         0.55

# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  DONE — Retraining pipeline complete.                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Models:"
echo "    Tabular:  $MODELS_DIR/beat_tabular_v3.joblib"
echo "    CNN:      $MODELS_DIR/beat_cnn_v2.pt"
echo ""
echo "  Predictions written to $PROCESSED_FULL:"
echo "    beat_tabular_preds.parquet"
echo "    beat_cnn_preds.parquet"
echo "    ensemble_preds.parquet"
echo ""
echo "  Next: export a new annotation queue with the retrained ensemble"
echo "    python export_disagreement_queue.py \\"
echo "        --ensemble-preds $PROCESSED_FULL/ensemble_preds.parquet \\"
echo "        --ecg-samples    $PROCESSED_FULL/ecg_samples.parquet \\"
echo "        --output-dir     data/annotation_queues/iteration_2 \\"
echo "        --sort-by        disagreement \\"
echo "        --n-beats        500 \\"
echo "        --exclude-annotated data/annotation_queues/*/completed.csv"
echo ""
