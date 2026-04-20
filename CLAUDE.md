# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A personal long-term ECG artifact detection system for 14 months of continuous Polar H10 recordings (~50.5M beats) from a POTS patient. The goal is a per-beat artifact probability score (`p_artifact_ensemble`) that filters data for downstream HRV analysis. The output is `/Volumes/xHRV/processed/ensemble_preds.parquet`.

**All scripts are run from this directory: `/Volumes/xHRV/Artifact Detector/`**

## Environment

```bash
source /Users/tannereddy/.envs/hrv/bin/activate
```

The external SSD must be mounted at `/Volumes/xHRV` for any full-dataset operation.

## Shell Scripts (common operations)

```bash
./retrain_pipeline.sh          # Retrain models after new annotations (1-2 h total)
./rebuild_full_pipeline.sh     # Full rebuild from raw ECG (hours, rarely needed)
./run_new_dataset.sh           # Inference only on new ECG files
./run_pipeline_resume.sh       # Resumable partial rebuild
```

## Running Individual Pipeline Stages

Scripts are invoked as `python ecgclean/<module>.py <subcommand> --args`. All stages below must run in order from the `Artifact Detector/` directory.

### One-time build (initial setup or full rebuild)
```bash
# 1. Build canonical Parquet tables from raw ECG + peaks + annotations
python ecgclean/data_pipeline.py --ecg-dir data/raw_ecg/ --peaks-dir data/peaks/ \
  --annotations data/annotations/artifact_annotations.json --output-dir data/processed/

# 2. Apply physiological hard/soft filters (updates labels.parquet)
python ecgclean/physio_constraints.py --processed-dir data/processed/

# 3. Extract features (run both in parallel — they're independent)
python ecgclean/features/beat_features.py --processed-dir data/processed/ \
  --output data/processed/beat_features.parquet
python ecgclean/features/segment_features.py --processed-dir data/processed/ \
  --output data/processed/segment_features.parquet

# 4. Segment noise gate
python ecgclean/models/segment_quality.py train \
  --segment-features data/processed/segment_features.parquet \
  --segments data/processed/segments.parquet --output models/segment_quality_v1.joblib
python ecgclean/models/segment_quality.py predict \
  --segment-features data/processed/segment_features.parquet \
  --model models/segment_quality_v1.joblib \
  --output data/processed/segment_quality_preds.parquet

# 5. SSL pretraining (30-90 min on M3 MPS, do before CNN training)
python ecgclean/models/pretrain_ssl.py pretrain \
  --peaks data/processed/peaks.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output-checkpoint models/autoencoder_pretrained.pt \
  --output-encoder models/encoder_pretrained_weights.pt
```

### Active learning loop (repeat each iteration, incrementing version numbers)
```bash
# Train tabular model
python ecgclean/models/beat_artifact_tabular.py train \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output models/beat_tabular_v3.joblib

# Predict (tabular)
python ecgclean/models/beat_artifact_tabular.py predict \
  --beat-features data/processed/beat_features.parquet \
  --model models/beat_tabular_v3.joblib \
  --output data/processed/beat_tabular_preds.parquet

# Train + predict CNN (1-3 h on M3 MPS)
python ecgclean/models/beat_artifact_cnn.py train \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output models/beat_cnn_v2.pt
python ecgclean/models/beat_artifact_cnn.py predict \
  --beat-features data/processed/beat_features.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --model models/beat_cnn_v2.pt --output data/processed/beat_cnn_preds.parquet

# Fuse into ensemble (alpha=0.55 → tabular 55%, CNN 45%)
python ecgclean/models/ensemble.py fuse \
  --tabular-preds data/processed/beat_tabular_preds.parquet \
  --cnn-preds data/processed/beat_cnn_preds.parquet \
  --output data/processed/ensemble_preds.parquet --alpha 0.55

# Find optimal blend weight
python ecgclean/models/ensemble.py tune-alpha \
  --tabular-preds data/processed/beat_tabular_preds.parquet \
  --cnn-preds data/processed/beat_cnn_preds.parquet \
  --labels data/processed/labels.parquet
```

### Active learning annotation workflow
```bash
# Select uncertain beats
python ecgclean/active_learning/sampler.py select \
  --ensemble-preds data/processed/ensemble_preds.parquet \
  --labels data/processed/labels.parquet \
  --segments data/processed/segments.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --n-candidates 500 --strategy combined --al-iteration 2 \
  --output data/processed/al_queue_iteration_2.parquet

# Export for annotation GUI
python ecgclean/active_learning/annotation_queue.py export \
  --candidates data/processed/al_queue_iteration_2.parquet \
  --peaks data/processed/peaks.parquet --labels data/processed/labels.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --output data/annotation_queues/iteration_2/

# Review interactively
python ecgclean/active_learning/review_queue.py \
  --queue-dir data/annotation_queues/iteration_2/ \
  --output data/annotation_queues/iteration_2/completed.csv

# Import back
python ecgclean/active_learning/annotation_queue.py import \
  --completed data/annotation_queues/iteration_2/completed.csv \
  --expected-ids data/processed/al_queue_iteration_2.parquet \
  --labels data/processed/labels.parquet --al-iteration 2
```

## Architecture

### Data locations
- **`data/`** — Local training subset (small; fits in RAM). Annotation queues, trained models, and the training-data amalgam used during `retrain_pipeline.sh` live here.
- **`/Volumes/xHRV/processed/`** — Full 50.5M-beat dataset (31 GB ECG Parquet, 2.5 GB beat features). Full-dataset inference always writes here.
- Raw ECG CSVs live at `/Volumes/xHRV/ECG/` (190 files, ~31 GB). Never loaded in full — always streamed.

### `ecgclean/` package structure
```
ecgclean/
  data_pipeline.py        # Raw ECG → canonical Parquet tables (4 outputs)
  detect_peaks.py         # R-peak detection (Pan-Tompkins + SWT wavelet ensemble)
  physio_constraints.py   # Hard/soft POTS physiology rules → 19 columns on labels.parquet
  physio_events.py        # Event marker parsing helpers
  features/
    beat_features.py      # 32 features per beat (RR context, QRS morphology, physio flags)
    segment_features.py   # 22 features per 60-s segment (HRV stats, SQI, EMD noise)
    motif_features.py     # QRS + RR pattern clustering → anomaly features
  models/
    segment_quality.py    # LightGBM multiclass: clean / noisy_ok / bad per segment
    beat_artifact_tabular.py  # LightGBM binary: artifact per beat (primary fast model)
    beat_artifact_cnn.py  # 1D CNN: raw 256-sample waveform + tabular features
    pretrain_ssl.py       # Denoising autoencoder (SSL backbone for CNN)
    segment_cnn_2d.py     # 2D CNN on CWT scalograms (parallel segment quality check)
    ensemble.py           # Linear blend of tabular + CNN; outputs disagreement score
  active_learning/
    sampler.py            # Uncertainty sampling (margin / committee / priority / combined)
    annotation_queue.py   # Export/import annotation JSON ↔ labels.parquet
    review_queue.py       # Interactive beat review TUI
```

### Pipeline flow
```
Raw ECG CSVs
    ↓ detect_peaks.py
    ↓ data_pipeline.py          → ecg_samples, peaks, labels, segments (Parquet)
    ↓ physio_constraints.py     → labels.parquet +19 columns
    ↓ beat_features.py          → beat_features.parquet (32 cols)
    ↓ segment_features.py       → segment_features.parquet (22 cols)
    ↓ segment_quality.py        → segment_quality_preds.parquet (noise gate)
    ↓ beat_artifact_tabular.py  → beat_tabular_preds.parquet
    ↓ [pretrain_ssl.py]         → encoder_pretrained_weights.pt
    ↓ beat_artifact_cnn.py      → beat_cnn_preds.parquet
    ↓ ensemble.py               → ensemble_preds.parquet  ← HRV analysis reads this
```

### Model versioning
Models are numbered sequentially: `beat_tabular_v1/v2/v3.joblib`, `beat_cnn_v1/v2.pt`. Increment on each AL retraining pass. The current production models are `beat_tabular_v3.joblib` and `beat_cnn_v2.pt`. Metrics are stored inside each `.joblib` artifact (see `PIPELINE.md` for the comparison snippet).

### Key design decisions
- **Class imbalance (~1.4% artifacts):** LightGBM uses `scale_pos_weight ≈ 77`; CNN uses synthetic noise augmentation to manufacture fake artifacts from clean beats.
- **Temporal train/val split:** Split by `segment_idx` (never random shuffle) to prevent data leakage across correlated beats.
- **Memory efficiency:** ECG and beat_features Parquets are always streamed in batches — never fully loaded. The `retrain_pipeline.sh` streaming patch pattern (Step 0b) is the canonical example.
- **PR-AUC is the primary metric** — not accuracy or F1 — because of the severe class imbalance. Target >0.85; AL stopping signal is <0.01–0.02 improvement per iteration.
- **For HRV filtering:** use the `recall ≥ 0.90` threshold (missing an artifact harms HRV more than a false positive). The `optimal_threshold` printed by tabular training is a starting point only.
- **Ensemble alpha:** currently 0.55 (55% tabular, 45% CNN). The `disagreement` score in `ensemble_preds.parquet` identifies beats where the two models disagree most — those are the best candidates for manual review.

### Annotation system note
Two incompatible label namespaces exist in older files. V1 raw labels: `1–9` (theme_id=8 = Artifact, theme_id=1 = Contraction). V2 labels: `100+` slot-based. The current pipeline uses V2. Do not conflate V1 theme_id values with V2 label values.

### Interactive / diagnostic tools
- `marker_viewer.py` — ECG browser/annotator using Polar Marker events
- `spot_check.py` — visual diagnostics for ensemble disagreements
- `export_disagreement_queue.py` — shortcut to extract high-disagreement beats
- `fix_inverted_ecg.py` — correct polarity-inverted ECG segments
- `annotation_ai_annotator.py`, `marker_ai_annotator.py` — AI-assisted annotation helpers
- `beat_reannotator.py` — re-annotate beats in bulk

## Detailed Documentation

Stage-by-stage guides are in `guides/`:
- `PIPELINE.md` — concise CLI reference
- `User Guide.md` — full step-by-step walkthrough
- `s4_Segment_Quality.md` through `s9_Pretrain_Ssl.md` — per-stage deep dives
