---
name: HRV Project — Full Technical Overview
description: Exhaustive reference for the entire HRV artifact detection project — every script, role, data flow, design choices, shell scripts, and current state. Load this at the start of any new context.
type: project
---

# HRV Project — Full Technical Overview

**Purpose:** This document is the complete reference for a personal long-term ECG artifact detection system built on 16 months of continuous Polar H10 recordings from a POTS patient. It is intended to bootstrap any new Claude session to full project awareness without re-reading source files.

---

## 1. Big Picture

The project has **two separate systems**:

### System A — Legacy HRV Analysis Pipeline (`/Users/tannereddy/Desktop/Cabin/Python/hrv/Pipeline/`)
The original set of scripts that compute HRV metrics from clean RR intervals. These produce Excel workbooks and graphs. They are **not actively being developed** — they are the *consumer* of the artifact detector's output.

### System B — Artifact Detector (`/Volumes/xHRV/Artifact Detector/`)
The ML pipeline being actively built. Its job is to produce a per-beat artifact probability score for every R-peak across 16 months of ECG data. The HRV pipeline (System A) then filters on that score.

**The entire point of System B:** Distinguish between genuine POTS physiology (extreme HR swings, vagal events, PVCs, PACs — all real) and signal artifacts (electrode pops, motion, EMG burst, lead-off) so HRV metrics aren't garbage.

---

## 2. Hardware & Data Source

- **Device:** Polar H10 chest strap ECG
- **Sample rate:** 130 Hz (NOT 256 Hz — this was a critical bug fixed earlier; many script defaults still say 256)
- **Format:** Raw CSV files, two columns: `DateTime` (epoch milliseconds integer), `ECG_uv` (float 𝜇V)
- **Volume:** 204 files, ~16 months of data, ~56 million beats total
- **Location:** `/Volumes/xAir/HRV/ECG/` (external SSD, must be mounted)
- **Naming convention:** `MM-DD, HH.MMam_MM-DD, HH.MMpm.csv` (start–end datetime span)

---

## 3. File System Layout

```
/Volumes/xAir/HRV/                          ← canonical data root (external SSD)
├── ECG/                                     ← 204 raw ECG CSVs (16 months)
├── processed/                               ← canonical parquet tables (outputs of rebuild)
│   ├── ecg_samples.parquet    (31 GB)       ← every ECG sample, all files merged
│   ├── peaks.parquet          (596 MB)      ← every R-peak, 56M+ rows
│   ├── labels.parquet         (921 MB)      ← per-beat quality labels + physio flags
│   ├── segments.parquet       (11 MB)       ← per-60s-segment quality labels
│   ├── beat_features.parquet  (2.5 GB)      ← 32-feature matrix per beat
│   ├── segment_features.parquet (31 MB)     ← 22-feature matrix per segment
│   ├── segment_quality_preds.parquet (8.5 MB)
│   ├── beat_tabular_preds.parquet (281 MB)
│   ├── beat_cnn_preds.parquet (556 MB)
│   └── ensemble_preds.parquet (753 MB)      ← FINAL OUTPUT for HRV analysis
├── Accessory/
│   ├── Marker.csv                           ← starred events from Polar app (200+ markers)
│   ├── DeviceHR.csv                         ← HR from device at 1Hz (not raw ECG)
│   ├── DeviceRR.csv                         ← RR from device at 1Hz (not raw ECG)
│   └── marker_annotations/                  ← OUTPUT of marker_viewer.py
│       ├── beat_annotations.csv
│       ├── segment_annotations.csv
│       ├── theme_labels.json
│       └── screenshots/                     ← PNG exports from double-press Enter

/Volumes/xAir/HRV/Artifact Detector/        ← project working directory (all scripts run from here)
├── ecgclean/                               ← main Python package
│   ├── detect_peaks.py
│   ├── data_pipeline.py
│   ├── physio_constraints.py
│   ├── features/
│   │   ├── beat_features.py
│   │   ├── segment_features.py
│   │   └── motif_features.py
│   ├── models/
│   │   ├── segment_quality.py
│   │   ├── beat_artifact_tabular.py
│   │   ├── beat_artifact_cnn.py
│   │   ├── pretrain_ssl.py
│   │   ├── segment_cnn_2d.py
│   │   └── ensemble.py
│   └── active_learning/
│       ├── sampler.py
│       ├── annotation_queue.py
│       └── review_queue.py
├── marker_viewer.py                        ← interactive ECG browser/annotator (separate from AL)
├── spot_check.py                           ← visual diagnostic for model disagreements
├── export_disagreement_queue.py            ← exports beats for review_queue from ensemble preds
├── fix_inverted_ecg.py                     ← standalone tool to correct inverted ECG segments
├── rebuild_full_pipeline.sh               ← full rebuild from raw ECG (hours)
├── retrain_pipeline.sh                    ← retrain models after new annotations
├── run_new_dataset.sh                     ← inference on new ECG files (no training)
├── run_pipeline_resume.sh                 ← resumable partial rebuild
├── data/
│   ├── new_peaks/                          ← peak CSVs output of detect_peaks.py
│   ├── annotations/
│   │   └── artifact_annotations.json       ← legacy manual annotations (from old GUI)
│   ├── annotation_queues/                  ← active learning queue directories
│   │   ├── iteration_1/                    ← 5,842 beats needing re-annotation (CURRENT)
│   │   ├── cnn_fp_500/, cnn_fp_750/, etc.  ← historical CNN false-positive queues
│   │   └── disagreement_500/
│   ├── training/                           ← combined training data (built by retrain_pipeline.sh)
│   ├── processed/                          ← local small processed data
│   └── motifs/
│       ├── qrs_motifs.joblib
│       └── rr_motifs.joblib
├── models/
│   ├── segment_quality_v1.joblib           ← segment noise gate (LightGBM)
│   ├── beat_tabular_v1.joblib              ← tabular beat classifier v1
│   ├── beat_tabular_v2.joblib              ← tabular beat classifier v2
│   ├── beat_tabular_v3.joblib              ← tabular beat classifier v3 (corrupted)
│   ├── beat_tabular_v5.joblib              ← tabular beat classifier v5 (current)
│   ├── beat_cnn_v1.pt                      ← CNN beat classifier v1
│   ├── beat_cnn_v2.pt                      ← CNN beat classifier v2 (corrupted)
│   ├── beat_cnn_v3.pt                      ← CNN beat classifier v3 (current; corrupted from problems across nearly all scripts in the pipeline that have somehow thus far gone unidentified)
│   ├── autoencoder_pretrained.pt           ← SSL pretrained encoder
│   ├── encoder_pretrained_weights.pt       ← extracted encoder weights
│   ├── segment_cnn2d_v1.pt                 ← 2D scalogram CNN for segments (corrupted; needs rewrite)
│   └── meta_classifier_v1.joblib           ← logistic stacking ensemble (alternative)
├── guides/
│   ├── PIPELINE.md                         ← concise CLI reference
│   ├── User Guide.md                       ← detailed step-by-step user guide
│   └── s4_Segment_Quality.md etc.          ← per-stage detailed design docs
└── terminal_logs/

/Users/tannereddy/Desktop/Cabin/Python/Projects/hrv/
├── Pipeline/                               ← legacy HRV analysis scripts
│   ├── hrv.py                              ← main HRV computation engine
│   ├── HRV2_Circadian.py                   ← circadian HRV analysis
│   ├── HRV3_Graphs.py                      ← graph generation
│   ├── HRV4_Instantaneous.py               ← instantaneous HRV
│   ├── CleanCSVs.py                        ← CSV preprocessing
│   └── config.py                           ← shared constants
└── Test/                                   ← test JSON files from old annotation approach
```

---

## 4. Every Script — Role, Inputs, Outputs, When Used

### 4.1 Core Pipeline Scripts (run in order)

---

#### `ecgclean/detect_peaks.py`
**Role:** R-peak detection on raw ECG files.
**When:** Step 0 — must run before data_pipeline.py. Also run via rebuild_full_pipeline.sh.
**Inputs:** `/Volumes/xAir/HRV/ECG/*.csv` (raw ECG CSVs)
**Outputs:** `data/new_peaks/*.csv` — one peak CSV per input file, columns: `peak_id` (epoch ms), `source`, `is_added_peak`
**Method:** Ensemble of two detectors:
  1. Pan-Tompkins variant (scipy bandpass → diff → square → moving window integration)
  2. SWT-based detector (PyWavelets stationary wavelet transform energy)
  Both detectors snap detected peaks to `argmax(|ECG|)` within ±60ms to correct MWI bias.
  Results merged (union) with 50ms tolerance window.
  Post-detection: global refractory enforcement (250ms minimum RR = 240 bpm max).
  Then: amplitude filter (added in Apr 2026) — removes false positives whose amplitude is <30% of neighbors or <20% of global median.
**Key flags:** `--fs 130` (CRITICAL — default is 256, wrong for Polar H10), `--method ensemble`, `--refractory-ms 250`
**Design choice:** Ensemble approach because Pan-Tompkins misses beats during rapid HR transitions common in POTS. Chunked processing (5-min windows with 5s overlap) keeps RAM bounded for large files.

---

#### `ecgclean/data_pipeline.py`
**Role:** Ingestion — converts raw CSVs + peak CSVs + annotations into canonical Parquet tables.
**When:** Step 1 — always runs first after detect_peaks.py.
**Inputs:** ECG CSVs, peak CSVs from detect_peaks.py, `artifact_annotations.json`
**Outputs (4 canonical tables):**
  - `ecg_samples.parquet` — all ECG samples with `timestamp_ns`, `ecg` (mV), `segment_idx`
  - `peaks.parquet` — all R-peaks with `peak_id`, `timestamp_ns`, `segment_idx`, `source`
  - `labels.parquet` — per-beat labels: `peak_id`, `label` (clean/artifact/phys_event/interpolated/missed_original)
  - `segments.parquet` — per-60s-segment quality: `segment_idx`, `quality_label`, `start_ns`, `end_ns`
**Key constants:**
  - `SEGMENT_DURATION_NS = 60s` — the pipeline divides the entire recording into consecutive 60-second segments
  - `DEDUP_TOLERANCE_NS = 10ms` — merges near-duplicate peaks
  - `ANNOTATION_MATCH_TOLERANCE_NS = 80ms` — matches legacy annotation timestamps to snapped peaks
  - `_ECG_INVERSION_THRESHOLD = -0.5 mV` — per-file polarity detection and correction
**Design choice:** Streaming architecture — never loads all 14 months of ECG into RAM. Uses ParquetWriter batching. Segment indexing is done relative to the global recording start timestamp.

---

#### `ecgclean/physio_constraints.py`
**Role:** Applies POTS-specific physiological knowledge to label beats with hard/soft flags.
**When:** Step 2 — right after data_pipeline.py. Also re-run in retrain_pipeline.sh (Step 0a).
**Inputs:** `labels.parquet`, `peaks.parquet` from processed dir
**Outputs:** Updated `labels.parquet` with 19 columns added including:
  - `hard_filtered` — TRUE only for structurally impossible patterns (zero/negative RR, "sandwich" peaks)
  - `physio_implausible` — soft flag for suspicious but possible patterns
  - `pots_transition_candidate` — within 45s of a detected tachycardic transition
  - `tachy_transition_candidate` — auto-detected HR rises ≥35 bpm (catches POTS orthostatic surge)
  - `review_priority_score` — composite score used by active learning sampler
**Key patient-specific constants:**
  - `HR_MODAL_LOW=90, HR_MODAL_HIGH=120 bpm` — this patient's normal resting HR range
  - `RR_SUSPICIOUS_SHORT_MS=220, RR_SUSPICIOUS_LONG_MS=3200` — soft flags (NOT hard filters)
  - `POTS_MAX_DELTA_HR_PER_SECOND=25.0` — fastest confirmed genuine HR drop
  - `TACHY_TRANSITION_MIN_RISE_BPM=35.0` — minimum rise to qualify as POTS surge
**CRITICAL design choice:** Hard filters are EXTREMELY conservative. Only fires on patterns that literally no cardiac mechanism can produce (e.g., a peak sandwiched between two other peaks at physiologically impossible spacing). The threshold was previously too aggressive and filtered real POTS beats.

---

#### `ecgclean/features/beat_features.py`
**Role:** Computes the 32-feature matrix that feeds all downstream ML models.
**When:** Step 3a — after physio_constraints.py. Can run in parallel with segment_features.py.
**Inputs:** `ecg_samples.parquet`, `peaks.parquet`, `labels.parquet`
**Outputs:** `beat_features.parquet` — 32 features per R-peak
**Feature groups:**
  1. Legacy v6 features: RR_prev, RR_next, RR_ratio, ECG window stats (mean, std, min, max, range)
  2. Extended RR context: 5-beat neighborhood stats, local HR, RR deviation from local mean
  3. QRS morphology similarity: Pearson correlation to per-segment template QRS
  4. Physio constraint pass-through: hard_filtered, physio_implausible, pots_transition_candidate, review_priority_score
  5. Segment context: segment quality label broadcast to all beats in segment
**Design choice:** `_PEAK_SNAP_SAMPLES=8` (±8 samples = ±62ms at 130Hz) snaps window center to `argmax(|ECG|)` to correct for Pan-Tompkins MWI bias. Template QRS is built only from ≥3 clean beats per segment.

---

#### `ecgclean/features/segment_features.py`
**Role:** Computes the 22-feature matrix for the segment noise gate classifier.
**When:** Step 3b — parallel with beat_features.py.
**Inputs:** `ecg_samples.parquet`, `peaks.parquet`, `labels.parquet`
**Outputs:** `segment_features.parquet` — 22 features per 60s segment
**Feature groups:**
  1. HRV statistics: meanNN, sdNN, rmssd, pNN50, pNN20 (standard HRV metrics on RR intervals)
  2. RR roughness: fraction of large RR jumps, artifact fraction, suspicious fraction
  3. SQI_QRS: mean inter-beat morphology correlation (template similarity across segment)
  4. EMD F-IMF statistics: entropy, mean, variance of first Intrinsic Mode Function (noise component)
  5. Raw ECG amplitude stats: median, IQR, p95, saturation fraction
**Note:** EMD (Empirical Mode Decomposition) is computationally expensive — slowest step in the pipeline.

---

#### `ecgclean/models/segment_quality.py`
**Role:** Stage 0 — segment noise gate. Classifies each 60s segment as `clean`, `noisy_ok`, or `bad`.
**When:** After segment_features.py. Run `train` once, then `predict` every rebuild.
**Inputs:** `segment_features.parquet`, `segments.parquet`
**Outputs:** `segment_quality_preds.parquet` (train also saves `models/segment_quality_v1.joblib`)
**Model:** LightGBM multiclass (3 classes). Temporal split (earlier segments → train, later → val). NaN imputation with training-set medians stored in model artifact.
**Design choice:** Segments predicted as `bad` are excluded from ALL downstream model training and inference. This is the holter-style noise gate — segment-level filtering is faster and more reliable than trying to classify every beat in a garbage segment.

---

#### `ecgclean/models/beat_artifact_tabular.py`
**Role:** Stage 1 — fast beat-level artifact classifier using tabular features.
**When:** Active learning loop Step 1. Primary workhorse model.
**Inputs:** `beat_features.parquet`, `labels.parquet`, `segment_quality_preds.parquet`
**Outputs:** `beat_tabular_preds.parquet` (p_artifact_tabular per beat), model `.joblib`
**Model:** LightGBM binary classifier. `scale_pos_weight = n_negative/n_positive` (~77x for 1.3% artifact rate). Temporal split only — NEVER random beat splits (adjacent beats share physiology). Hard-filtered beats excluded from training. PR-AUC is the primary metric (robust to class imbalance).
**Model versions:** v1 (Mar 17), v2 (Mar 18), v3 (Mar 25) — current is v3

---

#### `ecgclean/models/pretrain_ssl.py`
**Role:** Self-supervised pretraining — teaches the CNN encoder what ECG looks like before it sees labels.
**When:** One-time, before CNN training. Takes 30-90 minutes on M3.
**Inputs:** `peaks.parquet`, `ecg_samples.parquet`, `segment_quality_preds.parquet`
**Outputs:** `models/autoencoder_pretrained.pt`, `models/encoder_pretrained_weights.pt`
**Method:** Denoising autoencoder — corrupt clean ECG windows with synthetic noise (baseline wander, electrode pop, EMG burst, lead-off transient, motion, Gaussian), train to reconstruct clean signal. Encoder architecture is IDENTICAL to beat_artifact_cnn.py's CNN branch, enabling direct weight transfer.
**Design choice:** Self-supervised pretraining is used because the labeled artifact dataset is tiny (~2.2k beats out of 50M). The SSL task trains on ALL beats without needing labels, giving the CNN a huge head start.

---

#### `ecgclean/models/beat_artifact_cnn.py`
**Role:** Stage 2a — hybrid CNN beat classifier reading raw ECG waveform + tabular context.
**When:** After pretrain_ssl.py. Takes 1-3 hours on M3 MPS.
**Inputs:** `beat_features.parquet`, `labels.parquet`, `ecg_samples.parquet`, `segment_quality_preds.parquet`
**Outputs:** `beat_cnn_preds.parquet` (p_artifact_cnn per beat), model `.pt`
**Architecture:** 1D CNN branch (256-sample = ~2s ECG window) + 22-feature tabular branch → concatenated embeddings → 2-layer fusion head → binary output.
**Training:** PyTorch Lightning. BCEWithLogitsLoss + pos_weight. WeightedRandomSampler (15x oversampling of artifacts). EarlyStopping on val_pr_auc (patience=15). Cosine annealing LR. Temporal split.
**Noise augmentation:** Synthetic artifacts generated from clean beats to expand training data.
**Model versions:** v1 (Mar 17), v2 (Mar 25) — both exist but may be corrupted per retraining catastrophe

---

#### `ecgclean/models/ensemble.py`
**Role:** Stage 3 — combines tabular and CNN predictions into final artifact probability.
**When:** After both tabular and CNN predictions exist.
**Inputs:** `beat_tabular_preds.parquet`, `beat_cnn_preds.parquet`
**Outputs:** `ensemble_preds.parquet` — `p_artifact_ensemble`, `disagreement` per beat
**Methods:**
  - `fuse` — linear blend: `p_ensemble = alpha*p_tabular + (1-alpha)*p_cnn` (current alpha=0.55)
  - `tune-alpha` — grid search to find optimal blend weight
  - `train-meta` — logistic regression stacked on both scores (alternative)
**`disagreement` column:** `|p_tabular - p_cnn|` — high disagreement beats = most informative for annotation
**Current state:** ensemble_preds.parquet exists from Apr 3 rebuild. alpha=0.55 (tabular 55%, CNN 45%)

---

#### `ecgclean/features/motif_features.py`
**Role:** Discovers recurring ECG patterns and computes distance-based anomaly features.
**When:** Optional, anytime after beat_features.py. Not in the active rebuild pipeline.
**Inputs:** `beat_features.parquet`, `labels.parquet`
**Outputs:** `data/motifs/qrs_motifs.joblib`, `rr_motifs.joblib`, `motif_features.parquet`
**Method:** K-means clustering of beat windows (QRS morphology) and RR interval sequences. Anomaly score = distance to nearest cluster centroid.
**Status:** Motifs exist (qrs_motifs.joblib, rr_motifs.joblib) but motif_features.parquet may be stale.

---

#### `ecgclean/models/segment_cnn_2d.py`
**Role:** Alternative segment quality classifier using CWT scalogram images.
**When:** Optional parallel track, after segment_features.py.
**Inputs:** `ecg_samples.parquet`, `segments.parquet`
**Outputs:** `segment_cnn2d_preds.parquet`, `models/segment_cnn2d_v1.pt`
**Architecture:** 4 Conv2d/BN/ReLU blocks with MaxPool → AdaptiveAvgPool2d → 3-class softmax. Input: 64×64 Morlet scalogram image per 60s segment.
**Purpose:** Parallel quality signal — agreement with segment_quality.py is strong; disagreement flags segments for review.
**Model exists:** `models/segment_cnn2d_v1.pt` (Mar 17)

---

### 4.2 Active Learning Scripts

---

#### `ecgclean/active_learning/sampler.py`
**Role:** Selects the most informative unlabeled beats for human annotation.
**When:** After ensemble predictions exist, before each annotation round.
**Inputs:** `ensemble_preds.parquet`, `labels.parquet`, `segments.parquet`, `segment_quality_preds.parquet`
**Outputs:** `al_queue_iteration_N.parquet` — N candidate beats ranked by composite score
**Strategies:**
  - `margin` — beats near p=0.5 (maximum uncertainty)
  - `committee` — beats where tabular and CNN most disagree
  - `priority` — beats with high review_priority_score from physio_constraints
  - `combined` (default) — 0.4×uncertainty + 0.4×disagreement + 0.2×priority_normalized
**Why this composite:** Pure uncertainty misses beats that are certain to one model but wrong. Disagreement catches CNN blind spots. Priority catches POTS transition beats that models handle poorly.

---

#### `ecgclean/active_learning/annotation_queue.py`
**Role:** Serializes candidate beats to JSON for the annotation UI, imports results back.
**When:** After sampler.py (export), after review_queue.py (import).
**Export outputs:** One `beat_NNNNN.json` per candidate (ECG context window + features + current prediction), `queue_summary.csv`
**Import:** Reads `completed.csv` from review_queue.py, validates against expected IDs, updates `labels.parquet`

---

#### `ecgclean/active_learning/review_queue.py`
**Role:** Interactive terminal tool for manually labeling beats from a queue directory.
**When:** After annotation_queue.py export.
**Inputs:** Queue directory with beat JSON files and queue_summary.csv
**Outputs:** `completed.csv` in the queue directory
**Controls:** a=artifact, c=clean, i=interpolated, p=phys_event, m=missed_original, k=skip, SPACE=confirm, BACKSPACE=clear, b/←=back, 0-9+Enter=jump, q=quit
**Note:** This is the OLD annotation tool for active learning beats. `marker_viewer.py` is the NEW annotation tool for starred cardiac events.

---

### 4.3 Standalone Utility Scripts

---

#### `marker_viewer.py`
**Role:** Interactive ECG browser for annotating starred cardiac events from `Marker.csv`.
**When:** Used continuously during annotation phase. Run from `Artifact Detector/` directory.
**Launch command:** `python marker_viewer.py --processed-dir /Volumes/xAir/HRV/processed/`
**Inputs:** `Marker.csv` (200+ starred events from Polar app), `ecg_samples.parquet`, `peaks.parquet` from processed dir
**Outputs (to `/Volumes/xAir/HRV/Accessory/marker_annotations/`):**
  - `beat_annotations.csv` — beat-level annotations with theme_id and peak_id
  - `segment_annotations.csv` — region annotations with start_ns, end_ns, theme_id
  - `theme_labels.json` — theme names (editable in-app with double-press 1-9)
  - `screenshots/` — PNG exports per (marker_idx, theme_id)
**State machine:** BROWSE → click peaks → B to confirm → AWAIT_BEAT_THEME (press 1-9) → BROWSE
  T → SEG_START (click) → SEG_END (click) → AWAIT_SEG_THEME → BROWSE
  X → DELETE_SELECT (click annotated beat to remove, X again clears visible segments)
  Double-press 1-9: RENAME_THEME → type → Enter
  Double-press Enter: export screenshots
  Double-press Backspace: clear all annotations in window
**Key constants:**
  - `DISPLAY_NS = 60s` — default view window
  - `SAMPLE_RATE = 130` Hz
  - `SNAP_THRESHOLD_NS = 2s` — click snaps to nearest peak within ±2s
  - `_DOUBLE_PRESS_SEC = 0.40s` — detection window for double-press
  - `LOCAL_TZ = America/New_York`
**Current status:** ~25 of 200+ markers annotated. Annotation is slow — this is the bottleneck.

---

#### `spot_check.py`
**Role:** Visual diagnostic — shows the 25 most interesting beats in 8 disagreement categories.
**When:** After ensemble predictions. Use to understand model behavior.
**Inputs:** `beat_cnn_preds.parquet`, `beat_tabular_preds.parquet`, `ensemble_preds.parquet`, `ecg_samples.parquet`
**Shows:** CNN unsure→artifact, CNN unsure→clean, CNN certain artifact→ensemble clean, CNN certain clean→ensemble artifact, same 4 categories for tabular model
**Usage:** Close each matplotlib window to advance to next category.

---

#### `export_disagreement_queue.py`
**Role:** Exports top-N beats from ensemble predictions into review_queue format WITHOUT needing sampler.py.
**When:** Useful for targeted retraining — e.g., specifically fixing CNN false positives.
**Inputs:** `ensemble_preds.parquet`, `peaks.parquet`, `labels.parquet`, `ecg_samples.parquet`
**Outputs:** Queue directory with beat JSONs + `queue_summary.csv`
**Sort strategies:** `disagreement`, `p_artifact_cnn`, `p_artifact_ensemble`, `uncertainty_ensemble`
**Note:** Stream-efficient — never loads full 31GB ecg_samples.parquet; uses PyArrow predicate pushdown.

---

#### `fix_inverted_ecg.py`
**Role:** Standalone tool to detect and fix inverted ECG segments in ecg_samples.parquet.
**When:** One-time diagnostic if you suspect polarity issues. Now built into data_pipeline.py per-file.
**Modes:** `--report` (inspect only), `--fix` (write corrected parquet to `--output`)
**Note:** Polarity correction is NOW automatic in data_pipeline.py. This script is for manual correction of ecg_samples.parquet if needed after the fact.

---

### 4.4 Shell Scripts

---

#### `rebuild_full_pipeline.sh`
**Role:** Full rebuild from raw ECG — ALL 5 steps from scratch.
**Takes:** Hours (14 months of data)
**Run:** `bash "/Volumes/xAir/HRV/Artifact Detector/rebuild_full_pipeline.sh"`
**Steps:**
  1. detect_peaks.py (`--fs 130 --method ensemble`) → clears data/new_peaks/
  2. data_pipeline.py → clears 4 canonical parquets in /Volumes/xAir/HRV/processed/
  3. physio_constraints.py → updates labels.parquet
  4. beat_features.py → rebuilds beat_features.parquet
  5. segment_features.py + segment_quality.py predict → rebuilds segment parquets
**Does NOT rebuild:** model training, CNN predictions, ensemble, active learning
**Safe to re-run:** explicitly deletes only listed files, does not touch model files or annotations
**Last run:** Apr 3, 2026

---

#### `retrain_pipeline.sh`
**Role:** Full retraining of models after annotation sessions.
**Takes:** ~3-4 hours total (tabular: 5 min, CNN: 1-2h, inference: 1-2h)
**Run:** `bash "/Volumes/xAir/HRV/Artifact Detector/retrain_pipeline.sh"`
**Steps:**
  0a. Re-run physio_constraints.py on full dataset (adds tachy_transition_candidate)
  0b. Patch updated physio columns into full beat_features.parquet (streaming patch, ~20 min)
  1. Merge all annotation_queues/*/completed.csv → combined labels.parquet in data/training/
  2. Build combined training directory (extracts ECG for newly labeled beats from xAir)
  3. Retrain tabular model → models/beat_tabular_v3.joblib
  4. Retrain CNN model → models/beat_cnn_v2.pt
  5. Full-dataset inference (tabular ~5 min, CNN ~1-2h)
  6. Re-run ensemble (alpha=0.55)
**Note:** This script has hardcoded version numbers (v3, v2) — if you run it again, you need to edit those version numbers or they'll overwrite existing models.

---

#### `run_new_dataset.sh`
**Role:** Inference only on new ECG files (no retraining).
**When:** If you get a new batch of ECG files and want predictions without retraining.
**Note:** Uses default `--method ensemble` (no `--fs 130` flag — THIS IS A BUG, should add --fs 130)

---

#### `run_pipeline_resume.sh`
**Role:** Partial rebuild that can resume from any step.
**Status:** Less maintained than rebuild_full_pipeline.sh.

---

### 4.5 Legacy HRV Analysis Scripts (`/Users/tannereddy/Desktop/Cabin/Python/Projects/hrv/Pipeline/`)

These are the **consumers** of the artifact detector's output. They are NOT currently being actively developed.

- **`hrv.py`** — Main HRV computation. Uses IntervalLedger system to track processed time ranges. Computes standard HRV metrics (SDNN, RMSSD, pNN50, LF/HF, etc.) using neurokit2, nolds, antropy. Outputs to Excel workbooks.
- **`HRV2_Circadian.py`** — Circadian analysis of HRV across time of day.
- **`HRV3_Graphs.py`** — Graphing/visualization of HRV results.
- **`HRV4_Instantaneous.py`** — Instantaneous HRV computation.
- **`CleanCSVs.py`** — CSV preprocessing utilities.
- **`config.py`** — Shared constants for the HRV pipeline.

---

## 5. Data Flow (Full Dependency Chain)

```
Raw ECG CSVs (/Volumes/xAir/HRV/ECG/)
       ↓
detect_peaks.py → data/new_peaks/*.csv
       ↓
data_pipeline.py + artifact_annotations.json
       ↓
┌─────────────────────────────────┐
│ ecg_samples.parquet (31 GB)     │
│ peaks.parquet (596 MB)          │
│ labels.parquet (921 MB)         │
│ segments.parquet (11 MB)        │
└─────────────────────────────────┘
       ↓
physio_constraints.py
       ↓ (updates labels.parquet with 19 columns)
       │
       ├→ beat_features.py → beat_features.parquet (2.5 GB)
       │
       └→ segment_features.py → segment_features.parquet (31 MB)
              ↓
       segment_quality.py train + predict → segment_quality_preds.parquet
              ↓
       beat_artifact_tabular.py train + predict → beat_tabular_preds.parquet
              ↓
       pretrain_ssl.py → encoder_pretrained_weights.pt
              ↓
       beat_artifact_cnn.py train + predict → beat_cnn_preds.parquet
              ↓
       ensemble.py fuse → ensemble_preds.parquet ← FINAL OUTPUT
              ↓
       ┌─────────────────────────────────────────────────┐
       │ HRV Analysis Pipeline (hrv.py, HRV2, etc.)      │
       │ Filter: p_artifact_ensemble > threshold          │
       └─────────────────────────────────────────────────┘
```

**Active Learning Loop (separate from above):**
```
ensemble_preds.parquet
       ↓
sampler.py select → al_queue_iteration_N.parquet
       ↓
annotation_queue.py export → annotation_queues/iteration_N/
       ↓
review_queue.py → completed.csv
       ↓
annotation_queue.py import → updates labels.parquet
       ↓ (retrain from beat_artifact_tabular.py onward)
```

**Marker annotation workflow (separate from above):**
```
Marker.csv + ecg_samples.parquet + peaks.parquet
       ↓
marker_viewer.py
       ↓
Accessory/marker_annotations/{beat_annotations.csv, segment_annotations.csv}
```

---

## 6. Canonical Parquet Table Schemas

### `peaks.parquet`
- `peak_id`: int64 (epoch nanoseconds = timestamp of the R-peak)
- `timestamp_ns`: int64
- `segment_idx`: int32
- `source`: string (detected/added)
- `is_added_peak`: bool

### `labels.parquet`
- `peak_id`: int64
- `label`: string (clean/artifact/phys_event/interpolated/missed_original)
- 19 physio constraint columns including: `hard_filtered`, `physio_implausible`, `pots_transition_candidate`, `tachy_transition_candidate`, `review_priority_score`, `rr_prev_ms`, `rr_next_ms`, etc.

### `segments.parquet`
- `segment_idx`: int32
- `quality_label`: string (clean/noisy_ok/bad)
- `start_ns`, `end_ns`: int64
- `n_beats`, `n_artifact`, `artifact_fraction`, etc.

### `ecg_samples.parquet`
- `timestamp_ns`: int64
- `ecg`: float32 (mV)
- `segment_idx`: int32

### `ensemble_preds.parquet`
- `peak_id`: int64
- `p_artifact_tabular`: float32
- `p_artifact_cnn`: float32
- `p_artifact_ensemble`: float32 ← use this for HRV filtering
- `disagreement`: float32 (|tabular - cnn|)
- `is_artifact_ensemble`: bool (at optimal threshold)

---

## 7. Key Technical Decisions & Their Rationale

### Why 130 Hz, not 256 Hz?
Polar H10 actually samples at 130 Hz. Many scripts default to 256 Hz (historical error). Always pass `--fs 130` to detect_peaks.py. This was a critical bug — using 256 Hz caused the bandpass filter cutoffs to be wrong relative to actual QRS frequency content.

### Why temporal split (not random)?
Adjacent beats share autocorrelated physiology (same electrode contact, same activity state, same HR trajectory). Random splits leak this structure and produce optimistically biased PR-AUC. Beat-level random split from the same recording session is invalid.

### Why PR-AUC (not accuracy)?
Artifact rate is ~1.3% of all beats. Accuracy is meaningless — 98.7% accuracy by predicting "clean" for everything. PR-AUC measures the quality of the tradeoff between precision and recall specifically on the rare positive class.

### Why segment-level noise gate first?
Processing 50M beats is expensive. If a 60s segment is clearly garbage (disconnected electrode, lying on sensor), there's no point running beat-level models on any of the ~130 beats in it. Stage 0 provides a fast, cheap filter that eliminates entire garbage windows.

### Why not hard-filter aggressively?
The patient has genuine POTS physiology that looks like artifacts: HR drops from 170 to 60 bpm in 15 seconds (real), 220ms RR intervals during SVT (real), 3-second pauses during vagal episodes (real). Aggressive hard-filtering would eliminate genuine physiology. Only filter what is structurally impossible.

### Why self-supervised pretraining?
2,200 annotated artifact beats out of 50 million is not enough to train a CNN from scratch reliably. SSL pretraining uses all 50M beats (no labels) to learn ECG representations, then the labeled 2,200 fine-tune the classifier head. This is the same principle as ImageNet pretraining.

### Timestamp unit convention
- **epoch milliseconds (ms):** Raw CSV format, peak_id in peak CSVs from detect_peaks.py
- **epoch nanoseconds (ns):** Internal Parquet format (`peak_id` column in peaks.parquet = timestamp_ns × 1)
- **Conversion:** MS_TO_NS = 1,000,000. Be careful — a bug here creates peaks 1M× off in time.

---

## 8. Current State (as of Apr 2026)

### What exists and works:
- Full 14-month rebuild completed Apr 3, 2026 (all 5 rebuild_full_pipeline.sh steps)
- All canonical parquets up to date (ecg_samples, peaks, labels, segments, beat_features, segment_features, segment_quality_preds)
- Ensemble predictions exist (from models trained on ~Mar 25 annotation data)
- Amplitude filter added to detect_peaks.py (new since Apr 2026 rebuild)
- marker_viewer.py has 5 new features: click-select beats, enhanced deletion, rename themes, screenshot export, click-define segments

### Known Issues / Current Problems:
1. **Retraining catastrophe (from memory):** Both tabular v3 and CNN v2 models were corrupted during retraining. This means the ensemble_preds.parquet may be based on bad models. The 5,500 beats in `data/annotation_queues/iteration_1/` need to be re-annotated with a NEW 5-subcategory scheme before retraining.
2. **Annotation bottleneck:** Only ~25 of 200+ markers annotated in marker_viewer. This is the current primary work.
3. **R-peak quality:** Some false R-peaks visible in marker_viewer (T-waves, waveform features detected as peaks). The amplitude filter should help after this rebuild, but the peak catalog from before Apr 3 was worse.
4. **run_new_dataset.sh bug:** Missing `--fs 130` flag.
5. **retrain_pipeline.sh hardcoded versions:** Will overwrite v3/v2 models on next run; edit version numbers before running again.

### Annotation queue re-annotation plan:
The `iteration_1` queue (5,500 beats) must be re-annotated using a NEW 5-subcategory scheme:
- `clean_pristine` — textbook clean
- `clean_noisy` — clean beat in noisy context
- `clean_low_amplitude` — genuine beat, low amplitude
- `artifact_hard` — clear artifact
- `artifact_noise` — ambiguous/noisy artifact
A new review interface is needed for this re-annotation (the old review_queue.py only has 5 categories: clean/artifact/interpolated/phys_event/missed_original).

---

## 9. Python Environment

- **venv:** `/Users/tannereddy/.envs/hrv/bin/activate`
- **Activate:** `source /Users/tannereddy/.envs/hrv/bin/activate`
- **Key packages:** pandas, pyarrow, numpy, lightgbm, scikit-learn, joblib, PyEMD, torch, pytorch-lightning, scipy, pywavelets, scikit-image, matplotlib, neurokit2, nolds, antropy, openpyxl

---

## 10. Warp Terminal Note

Warp adds whitespace that corrupts copied multiline commands. **Always write commands to a .sh file and run the file, or use a single-line command.** Never paste multiline Python/bash directly into Warp.

---

## 11. Quick Reference — Most Common Commands

```bash
# Run marker_viewer for annotation
source /Users/tannereddy/.envs/hrv/bin/activate
python "/Volumes/xAir/HRV/Artifact Detector/marker_viewer.py" --processed-dir /Volumes/xAir/HRV/processed/

# Spot-check model disagreements
cd "/Volumes/xAir/HRV/Artifact Detector" && source /Users/tannereddy/.envs/hrv/bin/activate && python spot_check.py

# Full rebuild (hours)
bash "/Volumes/xAir/HRV/Artifact Detector/rebuild_full_pipeline.sh"

# Retrain models after annotation
bash "/Volumes/xAir/HRV/Artifact Detector/retrain_pipeline.sh"

# Review an annotation queue
cd "/Volumes/xAir/HRV/Artifact Detector"
source /Users/tannereddy/.envs/hrv/bin/activate
python ecgclean/active_learning/review_queue.py \
    --queue-dir data/annotation_queues/iteration_1/ \
    --output data/annotation_queues/iteration_1/completed.csv
```
