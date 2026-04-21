# ECG Artifact Pipeline: Overhaul Session Summary
**Date:** April 20, 2026  
**Repo:** [tje623/HRV](https://github.com/tje623/HRV)  
**Working directory:** `/Volumes/xHRV/Artifact Detector/`

---

## Project Context

This pipeline processes continuous Polar H10 ECG recordings from a POTS patient — currently ~14 months of data (~54 million beats, ~80 GB of raw CSVs). The goal is a per-beat artifact probability score (`p_artifact_ensemble`) stored in `/Volumes/xHRV/processed/ensemble_preds.parquet` that filters data for downstream HRV analysis.

**Key constraints that drive most design decisions:**
- **Data is growing continuously.** New Polar H10 recordings are added regularly. The pipeline must support incremental ingestion of new data without full re-runs of the entire 80+ GB dataset.
- **Severe class imbalance (~1.4% artifacts).** PR-AUC is the primary metric, not accuracy or F1.
- **Dataset too large for RAM.** All scripts must stream data in chunks; nothing is ever fully loaded at once.
- **External SSD.** Full processed dataset lives at `/Volumes/xHRV/processed/` (31 GB ECG parquet, 2.5 GB beat features). I/O bandwidth is a real constraint.
- **Hardware: M3 Mac with 36 GB unified memory, 12 cores.** MPS (Metal Performance Shaders) is the GPU backend for PyTorch. MLX is a separate Apple framework and was not used — PyTorch MPS is the correct choice for this codebase.

---

## Part 1 — Annotation Consolidation (`scripts/consolidate_annotations.py`)

### Problem
The source annotation file (`/Volumes/xHRV/historical/artifact_annotations_master_edited.json`) had several issues preventing clean pipeline ingestion:
- Two incompatible annotation key names from different GUI versions (`interpolate_peaks` vs `flagged_for_interpolation`, `physiological_events` vs `tagged_physiological_events`).
- Two separate keys for bad time-range windows: `bad_regions` (5 entries) and `bad_regions_within_segments` (54 entries), stored separately but representing the same concept.
- `validated_segments` stored as strings (`"seg_N"`) referring to annotation-tool segment indices, which are **completely different** from pipeline segment indices. The annotation tool numbers segments sequentially (0–38992) across actual recording data; the pipeline assigns `segment_idx = (timestamp_ns - recording_start_ns) // 60_000_000_000`, producing sparse indices with large gaps during non-recording periods. Example: annotation idx 3579 → pipeline idx 256070.
- The `return_to_pile` entries (beats removed from the validated pool) were stored as dicts with a `segment_idx` key, not bare integers.
- Trust filtering needed to be applied: `effective_validated = validated_segments − bad_segments − return_to_pile`.

**First attempt failed (>30% artifact drop):** An initial dict-based `pipeline_to_ann` lookup produced the wrong result because annotation segment boundaries (derived from `first_timestamp` / `last_timestamp` in `segment_timestamps`) do not align with pipeline segment boundaries. A peak timestamp in the "second half" of an annotation segment crosses into the next pipeline segment boundary, finding no entry in the lookup dict and being incorrectly classified as `not_validated`.

### Solution
Created `scripts/consolidate_annotations.py` with a `_SegmentIntervalLookup` class that performs **interval-based lookup using `bisect`** on sorted `first_ns` timestamps, checking both the candidate and its neighbor against `[first_ns, last_ns]` from `segment_timestamps`. This correctly handles cross-boundary timestamps.

**Output (`/Volumes/xHRV/data/annotations/artifact_annotations_final.json`):**
- Artifacts: 5973 → 4291 (28.1% drop — within expected range)
- Added R-peaks: 2457 → 1944
- Effective validated segments: 1780
- Unified bad_regions: 59 windows (merged from both source keys)
- Normalised key names throughout

---

## Part 2 — `data_pipeline.py`: Fix `bad_regions` Handling

### Problem
The pipeline's `extract_bad_segment_indices()` function was **collapsing bad_regions into full-segment exclusions** — if any sub-segment time window was marked bad, the entire 60-second segment was excluded. This was too aggressive and wasted clean data within the same segment outside the bad window.

### Solution
- Removed the loop that added `bad_regions` to the full-segment exclusion set.
- Added new function `extract_bad_region_time_ranges()` that translates annotation `segment_idx` → pipeline `segment_idx` and returns precise `(pipeline_idx, start_ns, end_ns)` tuples.
- Added `in_bad_region` boolean column to `build_labels()`, computed per-beat by checking whether each peak's timestamp falls within any bad-region window. This column flows downstream into training exclusion logic.

---

## Part 3 — Training Exclusions (`beat_artifact_tabular.py`, `beat_artifact_cnn.py`)

### Problem
Two categories of beats were being incorrectly included in model training:
1. **Beats inside `bad_regions`:** These are time windows explicitly marked as untrustworthy (sensor issues, gross motion artifacts). Including them as training examples corrupts the model.
2. **`interpolated` beats:** These are synthetic peaks inserted by the annotation tool to fill gaps. They are not real ECG beats and should never appear in training data.

### Solution
Added two sequential exclusion blocks in both `beat_artifact_tabular.py` and `beat_artifact_cnn.py`, applied after bad-segment exclusion and before the temporal train/val split:
```python
# Exclude beats inside bad_region windows
if "in_bad_region" in merged.columns:
    merged = merged[~merged["in_bad_region"]].copy()

# Exclude interpolated beats
merged = merged[merged["label"] != "interpolated"].copy()
```
Both models guard against missing columns (`in_bad_region` defaults to `False` if absent) for backward compatibility with older processed directories.

---

## Part 4 — `beat_features.py`: Vectorization + Multiprocessing + Checkpointing

### Problem
Running on the full 54M-beat dataset took 3.5+ hours single-threaded with near-0% CPU utilization. Three root causes:

**1. Python `for i in range(n_beats)` inner loops:**
- `_compute_signal_quality_features`: 7 per-beat calculations (IQR, kurtosis, ZCR, SNR, HF noise, wander slope, energy ratio) in a Python loop — effectively single-threaded numpy calls × 500K iterations per chunk.
- `_compute_qrs_similarity`: per-beat Pearson correlation with template and adjacent beats.
- `_compute_extended_rr`: per-beat 5-beat rolling window for `rr_local_mean_5`/`rr_local_sd_5`.
- `_fill_nan_with_segment_median`: per-element NaN fill loop.

**2. No multiprocessing.** Each of 113 chunks processed serially despite 12 available cores.

**3. No checkpointing.** Interrupting at 55/113 lost all progress with no way to resume.

**4. Kurtosis `RuntimeWarning`.** Fired for flat ECG windows (disconnected sensor, saturated ADC) where variance ≈ 0 makes kurtosis numerically undefined. The warning was not a data quality bug — it's an expected edge case — but it needed proper handling.

### Solutions

**Vectorization:**
- `_compute_signal_quality_features`: Replaced the per-beat loop entirely with bulk numpy/scipy array operations on the `(n, 64)` window matrix. Key operations: `np.percentile(w, [75,25], axis=1)`, `scipy_kurtosis(w, axis=1)`, vectorised sign/diff for ZCR, OLS slope via matrix multiply `(w_c @ xs_c) / xs_var`, batch `np.diff` for HF noise RMS.
- `_compute_qrs_similarity`: Added `_pearson_batch(a, b, batch_size=50_000)` using `np.einsum("ij,ij->i", ...)` for vectorised row-wise Pearson in 50K-row batches (memory-controlled). Template array built with `np.stack`; adjacent correlations computed on shifted slices.
- `_compute_extended_rr`: Replaced loop with `np.pad(rr_all, (2,2), mode="edge")` + `sliding_window_view(rr_padded, 4)` → `np.mean/std(wins_4, axis=1)`.
- `_fill_nan_with_segment_median`: Replaced per-element loop with pandas `groupby.median()` + `Series.map()` fill.

**Kurtosis fix:** Explicitly detect flat windows (`w.std(axis=1) < 1e-6`) and assign `0.0` directly, rather than silently suppressing the warning. Flat windows are almost always artifacts and `0.0` Fisher kurtosis is a reasonable imputation that the model can act on.

**Multiprocessing — Fork → Spawn (critical memory fix):**

*Initial implementation (wrong):* Used `multiprocessing.get_context("fork")`. With fork, each worker inherits the parent's **entire address space**. CPython's reference counting writes to every Python object on access, immediately defeating copy-on-write. With the parent using 16 GB and 8+ workers, this would OOM instantly.

*Correct implementation (spawn):* Switched to `multiprocessing.get_context("spawn")`. Workers start as fresh Python processes with zero inherited memory. Each worker receives only small task metadata (chunk range, file paths) plus two small DataFrames (`segments_df`, `seg_quality`). Workers load their own data slices from disk using PyArrow predicate pushdown:
- Peaks: `filters=[("segment_idx", ">=", seg_min), ("segment_idx", "<=", seg_max)]`
- Labels: `filters=[("timestamp_ns", ">=", ts_min), ("timestamp_ns", "<=", ts_max)]` (labels has `timestamp_ns` but not `segment_idx`; derived from chunk peaks' timestamp range)
- ECG: existing segment_idx predicate pushdown

Parent process now loads only the `segment_idx` column (~200 MB) to plan chunks, instead of full 54M-row peaks + labels DataFrames (~9 GB).

**Memory profile:** ~1.5–2 GB per worker. At 10 workers (default for 36 GB machine): ~15–20 GB total — well within budget.

**Checkpointing / Resume:**
- Each chunk writes to `<output>.ckpt/chunk_NNNNN.parquet` immediately on completion.
- On startup, existing checkpoint files are detected and their chunks skipped.
- On interrupt, restart with identical command — completed chunks are skipped automatically.
- On successful completion, all checkpoints are assembled in order into the final parquet, then the `.ckpt/` directory is deleted.

**New CLI flags:** `--workers` (default 10), `--checkpoint-dir`, `--segment-quality-preds`.

---

## Part 5 — CNN Inference: Vectorize `_extract_windows`

### Problem
`_extract_windows()` in `beat_artifact_cnn.py` contained two O(n_beats) Python loops:
1. **Per-beat window extraction:** One `np.interp` call per beat to reconstruct a 256-sample ECG window from raw timestamps. With ~2.7M beats per CNN inference chunk, this is ~2.7M Python iterations.
2. **Per-beat bandpass filtering:** One `sosfiltfilt(sos, windows[i])` call per beat. Each call initialises the filter state and processes 256 samples in isolation — massive overhead.

These loops were the dominant cost in full-dataset CNN inference.

### Solution
**Window extraction — vectorised per-segment:**  
Instead of iterating over each beat, iterate over unique segments (~thousands per chunk, not millions of beats). All beats within a segment are interpolated simultaneously using a single `np.interp` call on the flattened `(k×256,)` target timestamp array:
```python
target_ts = (t_starts[:, None] + offsets[None, :]).ravel()  # (k×256,)
flat = np.interp(target_ts, seg_ts, seg_ecg, left=0.0, right=0.0)
windows[beat_mask] = flat.reshape(-1, 256)
```

**Bandpass filtering — batched `sosfiltfilt`:**  
`sosfiltfilt(sos, X, axis=1)` processes all rows of `X` simultaneously. Applied in batches of 10,000 windows to control memory (~10 MB per batch):
```python
nonzero_idx = np.where(np.any(windows != 0, axis=1))[0]
for start in range(0, len(nonzero_idx), 10_000):
    idx = nonzero_idx[start:start + 10_000]
    filtered = sosfiltfilt(sos, windows[idx], axis=1)
    windows[idx[~np.any(np.isnan(filtered), axis=1)]] = filtered[...]
```
~100× faster per filter operation vs. the per-beat version.

---

## Part 6 — `run_new_dataset.sh`: Incremental Ingestion Overhaul

### Problem — Why This Matters
Data accrual is continuous. The user receives new Polar H10 recordings regularly. The old `run_new_dataset.sh` pointed `--ecg-dir` at `/Volumes/xHRV/ECG/` — the **entire 80+ GB archive** — meaning every new-data run re-processed all historical data from scratch. This was completely wrong and would take many hours per run for what should be a 20-40 minute incremental job.

Additionally, the script:
- Used stale model versions (`beat_tabular_v2`, `beat_cnn_v1`, `alpha=0.5`).
- Skipped the `global_templates.py correlate` and `merge_features_for_training.py` steps, producing feature vectors that didn't match what the current trained models expect.
- Dropped results on the floor — never appended them to the main `/Volumes/xHRV/processed/ensemble_preds.parquet`.

### Solution
**Staging directory pattern:** New ECG CSVs go into `/Volumes/xHRV/ECG_new/` (staging) rather than directly into `/Volumes/xHRV/ECG/` (archive). The script processes only files in `ECG_new/`, completely ignoring the historical archive.

**Full corrected pipeline sequence:**
1. `detect_peaks.py` on new files → `data/new_processed/`
2. `data_pipeline.py` on new files → canonical parquets for new data only
3. `physio_constraints.py` on `new_processed/`
4. `beat_features.py` on `new_processed/`
5. `segment_features.py` + `segment_quality.py predict`
6. `global_templates.py correlate` — **reuses existing template** from `data/templates/global_templates.joblib`, computes correlations for new beats only
7. `merge_features_for_training.py` — produces `beat_features_merged.parquet` for new data
8. `beat_artifact_tabular.py predict` (v5_postcorrection)
9. `beat_artifact_cnn.py predict` (v3, MPS-accelerated)
10. `ensemble.py fuse` (alpha=0.55)
11. **Timestamp-deduped append** to `/Volumes/xHRV/processed/ensemble_preds.parquet`

**Deduplication guard:** Before appending, the script checks for overlapping `timestamp_ns` values between new and existing predictions. Any overlapping timestamps are skipped, making the script idempotent — safe to re-run if the same files are accidentally staged twice.

**After processing:** Move verified CSVs from `ECG_new/` to `ECG/` for archival.

**CLI flags added:** `--ecg-dir` (override staging dir), `--skip-append` (dry-run without touching main dataset).

### Current-Session Exception
The 2.75 weeks of new data processed during this session were already placed directly in `/Volumes/xHRV/ECG/` (before the staging workflow was established). The initial raw ECG cleaning script has since been reconfigured to output to `ECG_new/` going forward. For the current batch, isolation of new files uses:
```bash
find /Volumes/xHRV/ECG/ -name "*.csv" \
  -newer "/Volumes/xHRV/ECG/3-24, 12.49PM_3-25, 4.27AM.csv" \
  -exec mv {} /Volumes/xHRV/ECG_new/ \;
bash run_new_dataset.sh
```

---

## Part 7 — GitHub Repository Setup

**Repo:** [tje623/HRV](https://github.com/tje623/HRV) (public)

All 53 scripts committed in the initial push:
- `ecgclean/` package (data pipeline, features, models, active learning)
- `scripts/` (consolidation, training utilities, validation tools)
- Root-level shell scripts and diagnostic tools
- `guides/` documentation
- `CLAUDE.md`

**`.gitignore` excludes:** All parquet/CSV/JSON data files, trained model weights (`.pt`, `.joblib`), annotation queues, lightning logs, checkpoint directories, `__pycache__`.

**Policy established:** All script edits are committed and pushed to the repo without needing to be asked. This is recorded in both `CLAUDE.md` and the project memory file.

---

## Files Created / Modified

| File | Action | Summary |
|---|---|---|
| `scripts/consolidate_annotations.py` | **Created** | Annotation consolidation with interval-based coordinate lookup |
| `/Volumes/xHRV/data/annotations/artifact_annotations_final.json` | **Created** | Consolidated annotation output (4291 artifacts, 1780 validated segs, 59 bad windows) |
| `ecgclean/data_pipeline.py` | **Modified** | Removed full-segment bad_region collapse; added `extract_bad_region_time_ranges()` + `in_bad_region` column |
| `ecgclean/models/beat_artifact_tabular.py` | **Modified** | Added `in_bad_region` and `interpolated` exclusions from training |
| `ecgclean/models/beat_artifact_cnn.py` | **Modified** | Same training exclusions; vectorized `_extract_windows` |
| `ecgclean/features/beat_features.py` | **Modified** | Full vectorization of inner loops; spawn-based multiprocessing with checkpointing; explicit flat-window kurtosis handling |
| `run_new_dataset.sh` | **Modified** | Staging-dir pattern; full correct pipeline; model version updates; append to main dataset |
| `CLAUDE.md` | **Modified** | Added GitHub commit policy |
| `.gitignore` | **Created** | Excludes data, weights, logs |
| `tje623/HRV` (GitHub) | **Created** | All scripts tracked in version control |

---

## Pipeline State at End of Session

**Completed on full dataset (`/Volumes/xHRV/processed/`):**
- `beat_features.parquet` ✓ (54M beats, took 3.5 h before multiprocessing fix; future runs ~20–30 min with 10 workers)
- `global_template_features.parquet` ✓
- `beat_features_merged.parquet` ✓
- `beat_tabular_preds.parquet` ✓ (model: `beat_tabular_v5_postcorrection.joblib`)

**In progress / pending:**
- `beat_artifact_cnn.py train` → `models/beat_cnn_v3.pt`
- `beat_artifact_cnn.py predict` → `beat_cnn_preds.parquet` (uses MPS GPU, vectorized `_extract_windows`)
- `ensemble.py fuse` → `ensemble_preds.parquet`
- `run_new_dataset.sh` for 2.75 weeks of new data

---

## Key Design Principles Established

1. **Never re-process historical data for new recordings.** Staging dir → incremental pipeline → append. The 80 GB archive is read-only after initial processing.
2. **Checkpoint every long-running script.** Scripts that take >5 minutes write per-chunk checkpoint files and resume on restart. Losing hours of progress to an interrupt is unacceptable at this dataset scale.
3. **Spawn, not fork, for multiprocessing.** Fork copies the entire parent address space; CPython reference counting defeats copy-on-write immediately. Spawn + file-path-based worker loading is the correct model for large-dataset parallelism.
4. **Vectorize before parallelizing.** Removing Python per-beat loops (replacing with numpy array ops) is typically a 50–200× speedup on its own, before any multiprocessing gain.
5. **Workers load their own data slices.** Rather than passing large DataFrames through IPC (expensive) or sharing via fork (OOM risk), each worker uses PyArrow predicate pushdown to read only its segment range from disk.
6. **Timestamp as the universal join key.** `timestamp_ns` is the only globally unique identifier across all sub-datasets. `peak_id` is local to each `processed/` directory and should not be used for cross-dataset joins.
