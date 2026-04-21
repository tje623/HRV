# ECG Artifact Detection Pipeline — v2 Final Architecture Specification
### Synthesized from initial design + literature review

---

## 0. Framing the Problem Correctly

This is **patient-specific, beat-level ECG signal quality and artifact detection** on nonstationary, single-lead, year-long data with extreme but genuine physiological dynamics (POTS). That combination is exactly where generic artifact detection fails — and exactly where the literature has developed the most targeted solutions.

The core insight from the literature review is that **no single model should be asked to do all of this at once.** High-performing pipelines decompose the problem into stages, each handling a narrower, better-conditioned task. The architecture here follows that pattern explicitly.

**What success looks like:** Stable, trustworthy downstream HRV metrics — not beat-perfect labels. The literature is explicit that discarding hopeless segments and accepting some uncertainty on borderline beats produces better downstream analysis than trying to salvage everything.

---

## 1. Diagnosis of v6

### What v6 Gets Right
- Beat-level RF with RR context and 64-sample morphological window is a solid foundation
- `class_weight="balanced"` acknowledges the imbalance problem
- Manual annotation corpus (~170k clean, ~2.2k artifact) is genuinely valuable

### Structural Problems

**1. One model doing too many things simultaneously:**
It must distinguish clean vs artifact, encode what is physiologically plausible for a POTS patient, and cope with day-to-day electrode/skin/posture nonstationarity — all at once, with only immediate-neighbor RR context.

**2. Temporal context is too shallow:**
A per-beat classifier that only sees one neighbor in each direction cannot distinguish "wild RR in an isolated bad beat" from "wild RR in the middle of a POTS transition block." The literature consistently shows that local segment context is critical.

**3. No noise gate:**
The model tries to classify beats in globally-hopeless segments. This is like trying to read individual words in a document that's been soaked in water. Those segments should be excluded before any beat-level model runs.

**4. Class imbalance is severe and largely unaddressed:**
~2,200 artifacts vs ~170,000 clean beats (~1.3% positive rate). `class_weight="balanced"` helps but is not sufficient; the model still sees artifacts as statistically irrelevant during tree splits.

**5. v6 ROC-AUC of 0.998 is almost certainly inflated:**
Adjacent beats within a segment are highly correlated. Random train/test splits leak correlated beats across the boundary, producing optimistic metrics that don't reflect real-world performance. All evaluation in the new system must use segment-level splits.

**6. Rich annotations are almost entirely unused:**
`bad_segments`, `bad_regions`, `flagged_poor_quality_segments`, `tagged_physiological_events`, `manually_added_missed_peaks`, `flagged_for_interpolation` — none of these inform the v6 model despite representing significant labeling effort.

---

## 2. Repository Layout

```
ecg_artifact_clean/
  data/
    raw_ecg/                      # original CSVs
    peaks/                        # R-peak detections
    annotations/                  # artifact_annotation.json and derivatives
    processed/                    # canonical Parquet tables
    motifs/                       # discovered RR and QRS motifs
  ecgclean/
    data_pipeline.py              # loaders → canonical tables
    physio_constraints.py         # domain knowledge filters and weak labels
    features/
      beat_features.py            # extended beat-level tabular features
      segment_features.py         # HRV + SQI_QRS + EMD + amplitude stats
      motif_features.py           # motif distance features
    models/
      segment_quality.py          # Stage 0: noise gate
      beat_artifact_tabular.py    # Stage 1: GBM beat classifier
      beat_artifact_cnn.py        # Stage 2a: hybrid 1D CNN
      segment_cnn_2d.py           # Stage 2b: scalogram CNN (segment-level)
      pretrain_ssl.py             # Stage 3: self-supervised pretraining
      ensemble.py                 # Stage 1 + Stage 2 fusion
    active_learning/
      sampler.py                  # uncertainty + disagreement + motif sampling
      annotation_queue.py         # queue serialization for annotation UI
  notebooks/
    01_exploration.ipynb
    02_segment_quality_eval.ipynb
    03_beat_model_eval.ipynb
    04_active_learning_audit.ipynb
    05_motif_analysis.ipynb
  legacy/                         # v6 code preserved for reference
```

---

## 3. Layer 1: Data — Canonical Tables

All downstream work reads from these four tables. Produced once by `data_pipeline.py`, saved as Parquet.

### `ecg_samples`
| Column | Type | Notes |
|---|---|---|
| `timestamp_ns` | int64 | Epoch nanoseconds (convert from ms in source) |
| `ecg` | float32 | Raw ECG amplitude |
| `segment_idx` | int32 | 60s segment index |

### `peaks`
| Column | Type | Notes |
|---|---|---|
| `peak_id` | int64 | Unique beat ID |
| `timestamp_ns` | int64 | R-peak timestamp |
| `segment_idx` | int32 | Segment membership |
| `source` | str | `detected` or `added` |
| `is_added_peak` | bool | From `manually_added_missed_peaks` |

### `labels` (beat-level)
| Label | Source in `artifact_annotation.json` |
|---|---|
| `clean` | `validated_true_beats` not otherwise flagged |
| `artifact` | `artifacts` list |
| `interpolated` | `flagged_for_interpolation` + `interpolated_replacements` |
| `missed_original` | `manually_added_missed_peaks` |
| `phys_event` | `tagged_physiological_events` |

### `segments` (segment-level quality labels)
| Label | Derivation |
|---|---|
| `bad` | Any of: `bad_segments`, `bad_regions`, `flagged_poor_quality_segments` |
| `noisy_ok` | Contains artifact beats but not globally flagged as bad |
| `clean` | No artifact flags, predominantly validated beats |

> This is where the year of annotation work gets cashed out into structured, reusable labels. Every field in `artifact_annotation.json` maps to exactly one place in these tables. Nothing is discarded.

---

## 4. Layer 2: Features

### 4.1 Beat-Level Features (Extended Tabular)

All existing features from v6 `features.py` are preserved as-is. The following are added.

#### Extended RR Context
For beat _i_, computed from the `peaks` table:

| Feature | Description |
|---|---|
| `rr_prev_2` | RR interval two beats back |
| `rr_next_2` | RR interval two beats forward |
| `rr_local_mean_5` | Mean RR over 5-beat symmetric window |
| `rr_local_sd_5` | SDNN over 5-beat symmetric window |
| `rr_abs_delta_prev` | `|RR_i − RR_{i-1}|` |
| `rr_abs_delta_next` | `|RR_i − RR_{i+1}|` |
| `rr_delta_ratio_next` | `RR_i / RR_{i+1}` |

#### Local QRS Morphology Similarity
For each segment, compute an average clean QRS template from beats labeled `clean`. Then:

| Feature | Description |
|---|---|
| `qrs_corr_to_template` | Pearson correlation between beat window and local segment template |
| `qrs_corr_prev` | Correlation between beat _i_ and beat _i−1_ windows |
| `qrs_corr_next` | Correlation between beat _i_ and beat _i+1_ windows |

*Literature basis: Campero Jurado 2023 — pairwise QRS correlation alone achieved 77% 3-class quality accuracy on single-lead patches, outperforming template-based and HRV-based SQIs.*

#### Physiological Constraint Features
Derived from domain knowledge about this patient's POTS dynamics (see §5):

| Feature | Description |
|---|---|
| `physio_implausible` | Binary: violates patient-specific HR/RR constraints with no plausible physiologic explanation |
| `pots_transition_candidate` | Binary: in a window consistent with observed POTS transition dynamics |
| `rr_exceeds_max_observed_clean` | Binary: RR interval exceeds the maximum seen in any firmly clean non-POTS segment |

#### Motif Distance Features (from §7)
| Feature | Description |
|---|---|
| `dist_to_nearest_qrs_motif` | DTW or Euclidean distance to nearest known QRS motif |
| `dist_to_nearest_rr_motif` | Distance from local RR subsequence to nearest RR motif |
| `nearest_motif_label` | Label of the nearest motif (`normal_sinus`, `pots_transition`, `anomaly`) |

#### Segment-Level Context (joined onto each beat)
| Feature | Description |
|---|---|
| `segment_artifact_fraction` | `(# artifact beats) / (# total beats)` in segment |
| `segment_rr_sd` | SDNN over entire segment |
| `segment_quality_pred` | Encoded output of Stage 0 segment quality model |
| `segment_sqi_qrs` | SQI_QRS score for the segment (see §4.2) |

---

### 4.2 Segment-Level Features (New)

Computed per 60s `segment_idx` to feed the Stage 0 quality model.

#### HRV Features (Kalpande et al. 2025 validated)
`meanNN`, `SDNN`, `RMSSD`, `pNN50`, `pNN20`, `minNN`, `maxNN`, `IQRNN`

#### RR Roughness
- Fraction of consecutive RR pairs where `|ΔRR| > 100ms`
- Fraction where `|ΔRR| > 200ms`
- Fraction where `|ΔRR| > 300ms`
- These thresholds should be tuned to the patient's known POTS transition rates to avoid flagging genuine physiology

#### SQI_QRS — First-Class Feature
Average Pearson correlation between *consecutive* QRS complex windows (no template required):

```
SQI_QRS = mean( corr(QRS_i, QRS_{i+1}) for i in segment )
```

*This is a first-class feature, not a footnote. Campero Jurado found SQI_QRS outperformed all other single SQIs on multi-day single-lead patch data. It is computed from R-peak locations and fixed windows — no additional signal processing required.*

#### EMD-F-IMF Statistics (Lee et al. 2012)
Apply Empirical Mode Decomposition to the segment's raw ECG. Extract the first intrinsic mode function (F-IMF), which is the high-frequency component capturing motion and muscle artifact:

- Square and normalize the F-IMF
- Compute over the segment: `f_imf_entropy` (Shannon entropy), `f_imf_mean`, `f_imf_variance`

*Literature basis: Lee et al. 2012 Holter artifact detector achieved 96.6% sensitivity, 94.7% specificity using only these three statistics. When added to a Holter AF detector, excluding flagged segments raised specificity from 74% to 85% with no sensitivity loss.*

#### Raw ECG Amplitude Statistics
- Median, IQR, 95th percentile of raw ECG over segment
- Fraction of samples at or near hardware saturation (detects electrode pop-offs)
- Segment-level bandpower in artifact-characteristic frequency bands (if EMD is implemented)

---

## 5. Physiological Constraint Encoding

This patient has POTS with documented extreme but genuine HR dynamics (e.g., 176→72 bpm drops in ~14 seconds). This domain knowledge is a significant asset that most generic pipelines lack. It should be encoded explicitly rather than left entirely to the model to discover.

### Hard Filters
Rules derived from known patient physiology:

1. **Maximum plausible instantaneous HR change:** If `|ΔRR|` between two consecutive beats exceeds what has ever been observed in any firmly clean POTS transition segment, flag as `physio_implausible`. This is a conservative upper bound, not a typical value.
2. **Minimum/maximum physiologically possible RR:** Absolute bounds (e.g., RR < 200ms or RR > 2500ms) are almost certainly artifact regardless of POTS context.
3. **Morphological impossibility:** Beats where the QRS window contains patterns that cannot be cardiac in origin (e.g., flat-line followed by a spike, sinusoidal waveform at AC frequency).

Hard filters generate **auto-artifact** labels without needing model inference. They are fast, interpretable, and immediately useful.

### Weak Labels / Soft Features
Rules that are informative but not definitive:

1. **POTS transition window:** A beat occurring within a window where HR is changing at a rate consistent with observed POTS dynamics — flag with `pots_transition_candidate = 1`. The beat-level model should learn to be *more permissive* about unusual RR values in these windows.
2. **Post-event normalization window:** Beats in the segment immediately after a POTS episode ends — elevated noise probability from residual motion.
3. **Time-of-day priors:** If segment-level noise is known to be worse during certain activity periods, encode that as a segment-level feature.

These become columns in the `labels` and `segments` tables, and features in the beat-level models.

---

## 6. Motif-Based Anomaly Detection

Given one subject and a year-long record, motif discovery is uniquely applicable here. Most patients don't have 8,760 hours of data from a single individual — you do.

### What to Discover
1. **QRS motifs:** Common waveform shapes in 64-128 sample windows around R-peaks. Expected motifs: normal sinus QRS, typical POTS-tachycardia QRS (slightly different morphology at high HR), artifact-distorted QRS shapes.
2. **RR motifs:** Common short RR subsequences (e.g., 10-beat windows). Expected: stable sinus rhythm, gradual POTS ramp-up, abrupt POTS drop, sleep-period bradycardia.

### How to Use Motifs

**As a weak label generator:**
- Beats whose QRS window is far from *all* known motifs are **anomaly candidates**
- Pipe anomaly candidates into the active learning review queue first, before uncertain beats from the classifier
- This is semi-automated pre-labeling: the model doesn't decide, but the anomaly flag tells you where to look

**As features:**
- `dist_to_nearest_qrs_motif` and `dist_to_nearest_rr_motif` (see §4.1) give the GBM and CNN explicit information about how unusual a beat is relative to the full history of this patient's ECG

**Implementation:**
- Start with SAX (Symbolic Aggregate approXimation) or a simple k-means clustering on the QRS windows and RR subsequences
- More sophisticated: use STUMPY (a fast Python library for matrix profile / motif discovery) on the RR time series
- The `motif_features.py` module generates distance features; the `motifs/` data directory stores discovered motifs with labels

---

## 7. Layer 3: Models

### Stage 0 — Segment-Level Quality Gate

**Purpose:** Mask globally bad segments before any beat-level model runs. This is the noise gate — the most important structural addition over v6.

- **Model:** LightGBM or XGBoost classifier
- **Input:** Full segment feature matrix (§4.2): HRV features + SQI_QRS + EMD-F-IMF stats + roughness metrics + amplitude stats
- **Labels:** `clean` / `noisy_ok` / `bad`
- **Output:** Per-segment quality prediction + probability score

Segments predicted `bad` are excluded from all downstream processing. Beats in `noisy_ok` segments are down-weighted during Stage 1 and Stage 2 training.

A segment-level RF baseline using only HRV features and SQI_QRS is a useful sanity check — Kalpande et al. achieved >93% accuracy within-dataset and ~90% cross-dataset with exactly this approach.

---

### Stage 1 — Beat-Level Tabular Classifier (GBM)

**Purpose:** Fast, interpretable, high-recall artifact detection. This is the primary production model.

- **Model:** LightGBM or XGBoost
- **Input:** Full extended beat-level feature set (§4.1)
- **Labels:** `clean` vs `artifact` (primary); optionally multi-class to separate `phys_event`
- **Class imbalance handling:**
  - `scale_pos_weight = (# clean) / (# artifact)` in XGBoost
  - Optionally: SMOTE oversampling of artifact class in training set only
  - Track PR-AUC as primary metric — ROC-AUC is insufficient at this imbalance ratio
- **Training scope:** Only beats from `clean` or `noisy_ok` segments
- **Validation:** Mandatory segment-level splits (never random beat splits)

**Output:** `p_artifact_tabular` per beat — calibrated probability, not hard label.

---

### Stage 2a — Beat-Level Hybrid 1D CNN

**Purpose:** Extract morphological features that the tabular model structurally cannot capture — waveform shape subtleties, artifact signatures that don't collapse into simple statistics, and patterns that only emerge from the raw waveform sequence.

#### Window Size
Expand to **250-300ms pre/post R-peak** (approximately 128-160 samples at 256Hz). This gives the CNN enough context to see the full QRS complex, P-wave onset, and T-wave, which is where the most diagnostic morphological information lives. The tabular features continue using the 64-sample window for backward compatibility.

#### Architecture

```
Input: ECG window [Batch, 1, 256 samples]
│
├── Conv1D(32, kernel=7, padding='same') → BatchNorm → ReLU
├── Conv1D(64, kernel=5, padding='same') → BatchNorm → ReLU → MaxPool(2)
├── Conv1D(128, kernel=5, padding='same') → BatchNorm → ReLU → MaxPool(2)
├── Conv1D(128, kernel=3, padding='same') → BatchNorm → ReLU → AdaptiveAvgPool
│         → CNN embedding vector [Batch, 128]
│
├── Tabular branch: [Batch, N_tabular]
│         → Linear(N, 64) → ReLU → Linear(64, 32) → ReLU
│         → Tabular embedding [Batch, 32]
│
└── Concatenate([CNN embedding, Tabular embedding]) → [Batch, 160]
          → Linear(160, 64) → ReLU → Dropout(0.3)
          → Linear(64, 1) → Sigmoid
          → p_artifact_cnn
```

Tabular branch inputs: RR context features, QRS correlations, motif distance features, physiological constraint flags, segment context. **Not** the raw window samples (those go through Conv layers).

#### Training Strategy — Noise Augmentation (Critical)

The class imbalance problem (~1.3% artifacts) is much better addressed by teaching the model what artifacts look like through augmentation than purely through reweighting. During training, apply to clean beats:

| Augmentation | Artifact Type Mimicked |
|---|---|
| Sinusoidal baseline drift (0.05-0.5 Hz, random amplitude) | Baseline wander from breathing/movement |
| Short-duration high-amplitude spike (1-3 samples) | Electrode pop-off |
| Low-frequency high-amplitude sinusoid burst (20-60 Hz envelope) | EMG / muscle artifact |
| Partial flatline followed by recovery | Lead-off transient |
| Random Gaussian noise scaled to 20-50% of QRS amplitude | General motion artifact |

*Literature basis: noise augmentation during training was used in both DeepBeat (PPG signal quality, F1 0.54→0.96) and the LVSD AI-ECG work to make classifiers robust to wearable-style contamination. It directly addresses the CNN's inability to learn from only 2,200 artifact examples.*

Apply augmentation to a fraction of clean beats (e.g., 5-15%) during each training epoch, creating synthetic artifact examples that vary across epochs. This effectively provides unlimited augmented artifact training samples.

#### Additional Training Details
- **Framework:** PyTorch Lightning (preferred) or Keras
- **Sampler:** `WeightedRandomSampler` to oversample true artifact beats 10-20x in addition to augmentation
- **Preprocessing:** Bandpass filter 3-40Hz on the input window (validated by Kalpande et al. and most wearable ECG pipelines)
- **Augmentation applied to:** Training set only — never validation or test
- **Training scope:** Same segment-level splits as Stage 1

---

### Stage 2b — Segment-Level 2D CNN on Scalograms (Optional but Recommended)

**Purpose:** A featureless, high-capacity segment quality classifier that learns noise signatures from time-frequency representations rather than engineered features. Particularly powerful for mixed-physiology data (alternating sinus/POTS) because it sees the joint time-frequency structure of transitions.

*Literature basis: Huerta et al. 2020 achieved ~92% clean/noisy classification on AF + sinus rhythm data using 2D CNN on wavelet scalogram images. The method requires no feature engineering — only segmentation and spectrogram computation.*

**Implementation:**
- Segment into 5-10s windows
- Compute **continuous wavelet transform (CWT) scalogram** or STFT spectrogram per window → 2D image
- Train a small 2D CNN (ResNet-18 or a 4-layer custom architecture) to classify `clean` / `noisy_ok` / `bad`
- Output: `p_bad_2d` per segment — use as an additional input to Stage 0 or as an independent quality signal

This can run as a parallel quality track alongside the Stage 0 engineered-feature model. Agreement between Stage 0 and Stage 2b on segment quality is a strong signal; disagreement flags segments for review.

---

### Stage 3 — Self-Supervised Pretraining

**Purpose:** Exploit the year-long unlabeled ECG corpus to learn a patient-specific representation of what this person's ECG looks like, before any artifact labels are introduced. This is the highest-leverage use of the data that has no labels attached to it.

*Literature basis: DeepBeat (Poh et al. 2020) pretrained a convolutional denoising autoencoder on ~1M simulated unlabeled PPG segments, then fine-tuned on 500k labeled signals. F1 for AF classification improved from 0.54 to 0.96. The key insight: unlabeled data encodes the distribution of normal; fine-tuning teaches the deviation.*

**Pretraining Task Options (in increasing complexity):**

1. **Denoising Autoencoder (simplest):** Corrupt a clean beat window (via the same augmentations from Stage 2a), train the network to reconstruct the original. Reconstruction error at inference time is a direct artifact score — high error = unusual/corrupted beat.

2. **Masked Prediction:** Mask random subsegments of the ECG window, train the model to predict the masked values from context. Forces the model to learn temporal structure.

3. **Contrastive Learning (SimCLR-style):** Generate two augmented views of the same beat, train an encoder to make them similar in embedding space while pushing different beats apart.

**Fine-Tuning:**
After pretraining on all available ECG (labeled + unlabeled), replace the decoder/projection head with a small artifact classification head and fine-tune on the labeled beats only. This should significantly reduce the number of labeled beats needed for high performance.

**Practical Note:** This is Stage 3 because it requires the earlier stages to be stable first. But it should be planned for from the beginning, meaning the CNN architecture in Stage 2a should be designed so its encoder weights can be initialized from the pretrained model.

---

### Ensemble — All Stages

Once Stage 1 and Stage 2a are trained:

```python
p_ensemble = alpha * p_tabular + (1 - alpha) * p_cnn
```

Start with `alpha = 0.5`. Tune alpha on the validation set. Graduate to a **logistic regression meta-classifier** on `[p_tabular, p_cnn]` if the linear blend is insufficient.

**The disagreement signal is a feature, not just a diagnostic:**
```python
disagreement = |p_tabular - p_cnn|
```
High disagreement cases are the most informative for active learning (see §8). They represent the model committee's honest uncertainty about a beat. Route all high-disagreement beats into the annotation queue first.

---

## 8. Active Learning Loop

### Core Protocol

1. Train current best model on all labeled beats
2. Run Stage 0 on new segments → mask `bad` segments
3. Run Stage 1 + Stage 2a on remaining beats → get `p_tabular`, `p_cnn`, `disagreement`
4. Build uncertainty pool from non-`bad` segments:
   - Primary: beats where `p_ensemble ∈ [0.3, 0.7]`
   - Secondary: beats with high `disagreement` score regardless of `p_ensemble`
5. Prioritize within pool:
   - Beats in `noisy_ok` segments (more borderline, more model-informative)
   - Beats near/within `tagged_physiological_events` windows (POTS transitions — highest clinical value)
   - High `dist_to_nearest_qrs_motif` (motif anomalies — likely either novel artifacts or novel physiology)
6. Serialize top-N beats to annotation review queue
7. After labeling K beats, retrain; log PR-AUC trajectory

### Annotation Queue Strategy (from Pasolli & Melgani 2010)
Three complementary selection strategies — use whichever combination produces fastest PR-AUC improvement per label:

- **Margin sampling:** Beats closest to decision boundary (`p ≈ 0.5`) — pure uncertainty
- **Query by committee:** Beats with highest `|p_tabular - p_cnn|` — disagreement-based uncertainty
- **Motif-anomaly enrichment:** Beats farthest from all known motifs — distribution coverage

### Labels Table Audit Trail
| Column | Description |
|---|---|
| `al_iteration` | Active learning round that generated this label (null = pre-existing) |
| `uncertainty_score` | `p_ensemble` at time of selection |
| `disagreement_score` | `|p_tabular - p_cnn|` at time of selection |
| `selection_strategy` | `margin`, `committee`, `motif_anomaly`, or `manual` |

---

## 9. Migration Path

### Step 1 — Normalize Data and Annotations
Write `data_pipeline.py`:
- Read peak CSVs, ECG CSVs, `artifact_annotation.json`
- Emit canonical `ecg_samples`, `peaks`, `labels`, `segments` tables
- Save to `data/processed/` as Parquet

### Step 2 — Encode Physiological Constraints
Write `physio_constraints.py`:
- Define patient-specific hard filters (absolute RR bounds, impossible morphology rules)
- Define POTS transition windows from `tagged_physiological_events`
- Generate `physio_implausible` and `pots_transition_candidate` columns in `labels`

### Step 3 — Port and Extend Features
- Move `compute_feature_matrix` from legacy `features.py` into `beat_features.py`
- Add extended RR context, QRS correlations, motif distance features, segment context joins
- Create `segment_features.py`: HRV + SQI_QRS + EMD-F-IMF + roughness + amplitude stats
- Validate feature distributions against v6 to catch regressions

### Step 4 — Motif Discovery
- Run k-means (or STUMPY matrix profile) on QRS windows and RR subsequences
- Label discovered clusters (`normal_sinus`, `pots_transition`, `anomaly`, etc.)
- Implement `motif_features.py` to generate distance features per beat

### Step 5 — Train Stage 0 Segment Gate
- Train LightGBM segment quality model on segment feature matrix
- Evaluate on held-out segments (segment-level splits)
- Verify it correctly masks known bad segments from annotations

### Step 6 — Train Stage 1 GBM
- Train beat-level LightGBM with extended features + segment gate outputs
- Segment-level train/val/test splits — flag any random splits as invalid
- Key metrics: PR-AUC and F1 for artifact class; compare to v6 on identical eval segments

### Step 7 — Train Stage 2a CNN
- Implement `beat_artifact_cnn.py` with hybrid architecture and noise augmentation strategy
- Train independently of Stage 1; benchmark independently
- Implement ensemble fusion in `ensemble.py`
- Audit disagreement cases manually — these drive the next annotation batch

### Step 8 — Train Stage 2b Scalogram CNN (Optional Parallel Track)
- Compute CWT scalograms for all segments
- Train 2D CNN for segment quality
- Compare against Stage 0 engineered-feature model; use disagreement as a quality signal

### Step 9 — Wire Active Learning Sampler
- Add uncertainty, disagreement, and motif-anomaly scoring to `sampler.py`
- Serialize priority queues to CSV for annotation UI
- Persist audit trail in labels table
- Track PR-AUC trajectory per AL iteration

### Step 10 — Self-Supervised Pretraining
- Implement denoising autoencoder pretraining in `pretrain_ssl.py`
- Pretrain on full unlabeled ECG corpus
- Fine-tune Stage 2a CNN from pretrained encoder weights
- Compare PR-AUC vs training from scratch — this is the key benchmark

---

## 10. Evaluation Standards

**All models evaluated on segment-level splits.** Never random beat splits.

| Metric | Role |
|---|---|
| **PR-AUC (artifact class)** | Primary — most informative at this imbalance ratio |
| **F1 at operating threshold** | Primary — threshold tuned on validation, reported on test |
| **Confusion matrix** | Mandatory — especially false negatives (missed artifacts) |
| ROC-AUC | Secondary only — can be misleadingly high at extreme imbalance |

**Baseline to beat:**
v6 RF at random-split metrics: precision ~0.84, recall ~0.73, F1 ~0.78, ROC-AUC 0.998
*These numbers are almost certainly inflated by random splits. Establish honest v6 baselines on segment-level splits first, then measure improvement from there.*

---

## 11. Division of Labor

| Responsibility | Owner |
|---|---|
| What counts as "clean" vs "hopeless" vs "physiologically interesting" | You |
| Patient-specific physiological constraint definitions (POTS bounds, hard filters) | You |
| Threshold decisions (uncertainty band, disagreement cutoff, motif distance cutoffs) | You |
| Failure case interpretation and annotation judgment calls | You |
| `data_pipeline.py` — loaders, converters, canonical table emission | Code LLM |
| `physio_constraints.py` — constraint rule implementation (from your spec) | Code LLM |
| Extended RR, QRS correlation, motif distance feature implementations | Code LLM |
| `segment_features.py` — HRV + SQI_QRS + EMD-F-IMF implementation | Code LLM |
| LightGBM/XGBoost training scripts with CLI/config | Code LLM |
| PyTorch Lightning CNN (Stage 2a) with noise augmentation pipeline | Code LLM |
| Scalogram generation + 2D CNN skeleton (Stage 2b) | Code LLM |
| Self-supervised denoising autoencoder (Stage 3) | Code LLM |
| Active learning sampler + annotation queue serialization | Code LLM |

---

## 12. What You Can Stop Worrying About

- **You do not need a transformer.** The literature's most competitive solutions are small CNNs or GBMs on medium-sized feature sets. Transformers add complexity without proportional benefit at this scale.
- **You do not need to classify every beat.** The literature explicitly discards MN-contaminated segments to improve downstream HRV and AF detection. Success is stable, trustworthy HRV metrics — not beat-perfect labels.
- **You do not need to annotate your entire corpus.** With active learning, motif-anomaly preselection, and Stage 3 self-supervised pretraining, a few tens of thousands of carefully chosen beats will outperform 175k randomly chosen ones.
- **The CNN class imbalance problem is solved by augmentation, not by finding more artifact examples.** The noise augmentation strategy in Stage 2a means the model has effectively unlimited artifact training examples — they're just synthetic. This is validated by the DeepBeat and LVSD AI-ECG results.

---

## References

- Kalpande et al. (ISWC 2025) — wearable ECG noise detection with HRV features, >93% accuracy
- Campero Jurado et al. (2023) — SQI_QRS on single-lead patches, 77% accuracy with pairwise QRS correlation alone
- Lee et al. (TBME 2012) — EMD-F-IMF Holter artifact detector, 96.6% sensitivity / 94.7% specificity
- Huerta et al. (2020, Entropy) — 2D CNN on wavelet scalograms, ~92% ECG quality classification
- Rahman et al. (Sci Rep 2026) — knowledge-based noise type classification + adaptive filtering
- Pasolli & Melgani (IEEE T-ITB 2010) — active learning for ECG beat classification (margin, committee sampling)
- Poh et al. (DeepBeat, 2020) — self-supervised pretraining on 1M unlabeled PPG segments, F1 0.54→0.96
- Sivaraks & Ratanamahatana (2015) — motif-based anomaly detection for ECG artifact/arrhythmia separation
