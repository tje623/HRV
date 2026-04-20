# ECG Artifact Detection Pipeline — User Guide
## What this is, what it needs, how to run it, what you get

---

## What This Pipeline Actually Does

You built a system that takes raw ECG recordings from your Polar H10 and automatically identifies which beats are artifacts (noise, electrode pops, motion) vs. genuine cardiac signal — including distinguishing between artifacts and your POTS physiology, which looks extreme but is real.

The end product is a **per-beat artifact probability score** for every beat in your recording. You use that score to filter your HRV analysis so it only runs on trustworthy data.

It does this in layers:
1. First throws out entire garbage segments (the noise gate)
2. Then scores individual beats with a fast gradient boosting model
3. Then scores them again with a neural network that reads the raw waveform shape
4. Then combines both scores into a final ensemble probability
5. Then tells you which beats you should go look at yourself (active learning)

---

## What You Need Before Running Anything

### Your Data Files
You need three types of files from your existing pipeline:

**1. ECG CSV files** — raw signal from your Polar H10
- Format: two columns — timestamp (epoch milliseconds) and ECG amplitude
- One file per recording session, or one big file — doesn't matter
- Put them in: `data/raw_ecg/`

**2. Peak CSV files** — your detected R-peaks with features
- These are the files your existing `anno_ml` pipeline produces
- Must have: timestamp column, `label` column (0=clean, 1=artifact), `segment_idx`, and `ecg_window_000` through `ecg_window_063`
- Put them in: `data/peaks/`

**3. artifact_annotation.json** — your manual annotation work
- The file you've been building with your annotation GUI
- Put it at: `data/annotations/artifact_annotation.json`

### Python Environment
Install everything you need:
```bash
pip install pandas pyarrow numpy lightgbm scikit-learn joblib PyEMD torch torchvision pytorch-lightning scipy pywavelets scikit-image
```

On your M3 Mac, PyTorch will automatically use the MPS GPU for the neural network steps. No configuration needed.

---

## Directory Structure to Create First

```bash
mkdir -p data/raw_ecg
mkdir -p data/peaks  
mkdir -p data/annotations
mkdir -p data/processed
mkdir -p data/motifs
mkdir -p data/annotation_queues
mkdir -p models
```

Then copy your files into the right places as described above.

---

## Run Order — Do These In Sequence

### STEP 1 — Build the canonical data tables
```bash
python ecgclean/data_pipeline.py \
  --ecg-dir data/raw_ecg/ \
  --peaks-dir data/peaks/ \
  --annotations data/annotations/artifact_annotations.json \
  --output-dir data/processed/
```
**What it does:** Reads all your raw files and produces 4 clean Parquet tables.
**Takes:** A few minutes for a year of data.
**Produces:**
- `data/processed/ecg_samples.parquet` — every ECG sample, timestamped
- `data/processed/peaks.parquet` — every R-peak with ID and segment
- `data/processed/labels.parquet` — beat-level labels (clean/artifact/phys_event/etc.)
- `data/processed/segments.parquet` — 60-second segment quality labels

---

### STEP 2 — Apply physiological constraints
```bash
python ecgclean/physio_constraints.py --processed-dir data/processed/
```
**What it does:** Flags beats that violate known POTS physiology rules. Hard filters catch structural impossibilities (zero RR, impossible sandwich patterns). Soft features flag suspicious HR values and POTS transition windows for the models to use.
**Takes:** 1-2 minutes.
**Produces:** Updated `data/processed/labels.parquet` with 19 columns including `hard_filtered`, `physio_implausible`, `pots_transition_candidate`, `review_priority_score`.

---

### STEP 3 — Compute feature matrices
Run both of these — they're independent and can run simultaneously in two terminals:

```bash
# Terminal 1
python ecgclean/features/beat_features.py \
  --processed-dir data/processed/ \
  --output data/processed/beat_features.parquet

# Terminal 2
python ecgclean/features/segment_features.py \
  --processed-dir data/processed/ \
  --output data/processed/segment_features.parquet
```
**What it does:** Computes the full feature set for each beat (32 features: RR context, QRS morphology similarity, physio flags, segment context) and each 60-second segment (22 features: HRV stats, SQI_QRS morphology consistency score, EMD noise statistics, amplitude stats).
**Takes:** 10-30 minutes for a year of data (the EMD computation is the slow part).
**Produces:**
- `data/processed/beat_features.parquet` — 32 features per beat
- `data/processed/segment_features.parquet` — 22 features per segment

---

### STEP 4 — Train the segment noise gate (Stage 0)
```bash
python ecgclean/models/segment_quality.py train \
  --segment-features data/processed/segment_features.parquet \
  --segments data/processed/segments.parquet \
  --output models/segment_quality_v1.joblib
```
Then run predictions:
```bash
python ecgclean/models/segment_quality.py predict \
  --segment-features data/processed/segment_features.parquet \
  --model models/segment_quality_v1.joblib \
  --output data/processed/segment_quality_preds.parquet
```
**What it does:** Trains a LightGBM model to classify each 60-second segment as `clean`, `noisy_ok`, or `bad`. Bad segments get masked out before any beat-level model runs — this is your Holter-style noise gate.
**Takes:** Under a minute to train.
**Produces:**
- `models/segment_quality_v1.joblib` — the trained model
- `data/processed/segment_quality_preds.parquet` — quality label + probabilities per segment

---

### STEP 5 — Train the beat tabular classifier (Stage 1)
```bash
python ecgclean/models/beat_artifact_tabular.py train \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output models/beat_tabular_v1.joblib
```
Then predict:
```bash
python ecgclean/models/beat_artifact_tabular.py predict \
  --beat-features data/processed/beat_features.parquet \
  --model models/beat_tabular_v1.joblib \
  --output data/processed/beat_tabular_preds.parquet
```
**What it does:** Trains a LightGBM model on your 32 beat-level features to classify each beat as artifact or clean. This is the fast, interpretable workhorse model. Uses your ~170k clean beats and ~2.2k artifact beats as training data. Handles class imbalance by weighting artifact beats ~77x more heavily.
**Takes:** 2-5 minutes to train.
**Key output to look at:** The training summary will print PR-AUC for the artifact class, a confusion matrix, and the top 20 most important features. **PR-AUC is the number that matters** — if it's above 0.85, the model is working well.
**Produces:**
- `models/beat_tabular_v1.joblib` — trained model with 3 operating thresholds
- `data/processed/beat_tabular_preds.parquet` — `p_artifact_tabular` per beat

---

### STEP 6 — Self-supervised pretraining (do this before the CNN)
This teaches the neural network what your ECG looks like before it sees any artifact labels.
```bash
python ecgclean/models/pretrain_ssl.py pretrain \
  --peaks data/processed/peaks.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output-checkpoint models/autoencoder_pretrained.pt \
  --output-encoder models/encoder_pretrained_weights.pt \
  --max-epochs 50 \
  --batch-size 1024
```
**What it does:** Trains a denoising autoencoder on ALL your ECG beats — labeled and unlabeled — without using any artifact labels. It learns to reconstruct clean ECG from artificially corrupted ECG. The learned encoder weights then initialize the CNN in Step 7, giving it a head start.
**Takes:** 30-90 minutes on your M3 (MPS GPU will be used automatically).
**Produces:**
- `models/encoder_pretrained_weights.pt` — the encoder weights to transfer to the CNN

To sanity-check the result (see ASCII waveform reconstructions):
```bash
python ecgclean/models/pretrain_ssl.py visualize \
  --checkpoint models/autoencoder_pretrained.pt \
  --ecg-samples data/processed/ecg_samples.parquet \
  --peaks data/processed/peaks.parquet \
  --n-examples 5
```

---

### STEP 7 — Train the CNN beat classifier (Stage 2a)
```bash
python ecgclean/models/beat_artifact_cnn.py train \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output models/beat_cnn_v1.pt
```
Then predict:
```bash
python ecgclean/models/beat_artifact_cnn.py predict \
  --beat-features data/processed/beat_features.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --model models/beat_cnn_v1.pt \
  --output data/processed/beat_cnn_preds.parquet
```
**What it does:** Trains a 1D convolutional neural network that reads the raw 256-sample ECG waveform around each beat AND the tabular RR/context features. It uses synthetic noise augmentation to compensate for having only 2.2k artifact examples — it manufactures thousands of fake artifact examples by corrupting clean beats with realistic noise patterns.
**Takes:** 1-3 hours on M3 with MPS (EarlyStopping will cut it short if it converges faster).
**Produces:**
- `models/beat_cnn_v1.pt` — trained CNN checkpoint
- `data/processed/beat_cnn_preds.parquet` — `p_artifact_cnn` per beat

---

### STEP 8 — Build the ensemble and get final scores
```bash
python ecgclean/models/ensemble.py fuse \
  --tabular-preds data/processed/beat_tabular_preds.parquet \
  --cnn-preds data/processed/beat_cnn_preds.parquet \
  --output data/processed/ensemble_preds.parquet \
  --alpha 0.5
```
To find the optimal blend weight:
```bash
python ecgclean/models/ensemble.py tune-alpha \
  --tabular-preds data/processed/beat_tabular_preds.parquet \
  --cnn-preds data/processed/beat_cnn_preds.parquet \
  --labels data/processed/labels.parquet
```
**What it does:** Combines the tabular model and CNN model scores into a single ensemble probability. Also computes a `disagreement` score — how much the two models disagree on each beat. High disagreement beats are the most interesting for manual review.
**Produces:**
- `data/processed/ensemble_preds.parquet` — final `p_artifact_ensemble` + `disagreement` per beat

**This is your primary output for downstream HRV analysis.** Filter beats where `p_artifact_ensemble > threshold` (use the optimal threshold printed during Step 5 training as a starting point).

---

### STEP 9 — Run the segment 2D CNN and motif discovery (parallel tracks)
These are independent and can run in any order after Step 3.

```bash
# 2D scalogram CNN for segment quality (parallel check on Stage 0)
python ecgclean/models/segment_cnn_2d.py train \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segments data/processed/segments.parquet \
  --output models/segment_cnn2d_v1.pt

python ecgclean/models/segment_cnn_2d.py predict \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segments data/processed/segments.parquet \
  --model models/segment_cnn2d_v1.pt \
  --output data/processed/segment_cnn2d_preds.parquet

# Compare the two segment quality models — disagreements are interesting
python ecgclean/models/segment_cnn_2d.py compare \
  --stage0-preds data/processed/segment_quality_preds.parquet \
  --cnn2d-preds data/processed/segment_cnn2d_preds.parquet
```

```bash
# Motif discovery — finds recurring ECG patterns, flags anomalies
python ecgclean/features/motif_features.py discover \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --output data/motifs/ \
  --n-qrs-clusters 12 \
  --n-rr-clusters 8

python ecgclean/features/motif_features.py compute \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --motifs data/motifs/ \
  --output data/processed/motif_features.parquet
```

---

### STEP 10 — Active learning: find beats to annotate next

After you have ensemble predictions, generate a prioritized review queue:
```bash
python ecgclean/active_learning/sampler.py select \
  --ensemble-preds data/processed/ensemble_preds.parquet \
  --labels data/processed/labels.parquet \
  --segments data/processed/segments.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --n-candidates 500 \
  --strategy combined \
  --al-iteration 1 \
  --output data/processed/al_queue_iteration_1.parquet
```

Export for your annotation GUI:
```bash
python ecgclean/active_learning/annotation_queue.py export \
  --candidates data/processed/al_queue_iteration_1.parquet \
  --peaks data/processed/peaks.parquet \
  --labels data/processed/labels.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --output data/annotation_queues/iteration_1/
```

After annotating, import results back:
```bash
python ecgclean/active_learning/annotation_queue.py import \
  --completed data/annotation_queues/iteration_1/completed.csv \
  --expected-ids data/processed/al_queue_iteration_1.parquet \
  --labels data/processed/labels.parquet \
  --al-iteration 1
```
Then retrain Steps 4-8 and repeat.

---

## Summary: Full Run Order

```
Step 1: data_pipeline.py          ← always first
Step 2: physio_constraints.py     ← always second
Step 3: beat_features.py          ← both in parallel
        segment_features.py       ←
Step 4: segment_quality.py train + predict
Step 5: beat_artifact_tabular.py train + predict
Step 6: pretrain_ssl.py           ← do before CNN
Step 7: beat_artifact_cnn.py train + predict
Step 8: ensemble.py fuse          ← needs steps 5+7
Step 9: segment_cnn_2d.py         ← anytime after step 3
        motif_features.py         ← anytime after step 3
Step 10: active learning          ← after step 8, ongoing
```

---

## What You End Up With

**For HRV analysis right now:** Use `data/processed/ensemble_preds.parquet`. Join it with your peaks on `peak_id`. Filter out beats where `p_artifact_ensemble` exceeds your chosen threshold (start with the `optimal_threshold` printed during Step 5 training). Run your HRV metrics on what remains.

**For ongoing improvement:** Every time you annotate a batch of uncertain beats (Step 10), retrain Steps 4-8. Your PR-AUC will climb with each iteration. You're not trying to annotate everything — just the 500 most uncertain beats each round.

**Models you'll want to keep:**
- `models/segment_quality_v1.joblib` — segment noise gate
- `models/beat_tabular_v1.joblib` — fast beat classifier (primary)
- `models/beat_cnn_v1.pt` — neural network beat classifier
- `models/encoder_pretrained_weights.pt` — pretrained encoder (reuse if you retrain CNN)

---

## Things That Will Probably Need Tuning

1. **The artifact threshold.** Step 5 prints three options (optimal F1, precision≥0.90, recall≥0.90). For HRV analysis where missing an artifact is worse than being conservative, start with the **recall≥0.90 threshold**.

2. **The ensemble alpha.** Run `tune-alpha` after Step 8 to find the optimal blend weight. If the tabular model significantly outperforms the CNN in PR-AUC, alpha will shift toward 1.0 (more tabular weight).

3. **n_clusters in motif discovery.** Start with 12 QRS clusters and 8 RR clusters. Look at the sparkline output — if multiple clusters look identical, reduce the number. If you see clusters that clearly mix different morphologies, increase it.

4. **POTS_TRANSITION_WINDOW_SEC in physio_constraints.py.** Currently 45 seconds. If the model is still incorrectly flagging beats immediately after POTS transitions, increase this. If it's being too permissive, reduce it.
