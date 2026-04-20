You are implementing Step 8 of an ECG artifact detection pipeline. Your task is to write two complete, production-ready Python files:
- `ecgclean/models/segment_cnn_2d.py`
- `ecgclean/features/motif_features.py`

These are two independent modules that can be developed and run in parallel. The 2D CNN is a second segment-quality classifier that operates on time-frequency images rather than engineered features. The motif module discovers recurring ECG patterns and generates distance-based features for the beat feature matrix.

---

## FILE 1: `ecgclean/models/segment_cnn_2d.py`

### Purpose
A segment-level quality classifier that learns noise signatures from wavelet scalogram images rather than hand-engineered features. It runs as a parallel track alongside Stage 0 (`segment_quality.py`). Agreement between the two models is a strong quality signal; disagreement flags segments for manual review.

This is a year-long single-lead ECG (Polar H10, ~256Hz) from a POTS patient. Segments are 60-second windows. The three quality labels are `clean`, `noisy_ok`, `bad`.

### Scalogram generation
```python
def compute_scalogram(
    ecg_segment: np.ndarray,
    sampling_rate: int = 256,
    image_size: tuple[int, int] = (64, 64)
) -> np.ndarray:
```

Compute a Continuous Wavelet Transform (CWT) scalogram using `pywt.cwt` with the `'morl'` (Morlet) wavelet. Use 64 logarithmically-spaced scales from 1 to 128. The output is a 2D array (scales x time). Resize to `image_size` using bilinear interpolation (`skimage.transform.resize`). Normalize to [0, 1] by dividing by the max absolute value (with epsilon=1e-8). Return as float32.

If the segment has fewer than 512 ECG samples, return a zero array of `image_size` shape and log a warning.

### Dataset class
```python
class SegmentScalogramDataset(torch.utils.data.Dataset):
```

- Takes `ecg_samples_df` and `segments_df`
- For each segment, extracts the raw ECG array from `ecg_samples_df`
- Computes scalogram on-the-fly (cache to disk in `data/processed/scalogram_cache/` keyed by segment_idx to avoid recomputation)
- Returns `(scalogram_tensor [1, H, W], label_tensor)` where scalogram is treated as a 1-channel image
- During training, apply light augmentation: random horizontal flip (time-reversal), random brightness/contrast jitter (±10%)

### Model architecture
```python
class SegmentQualityCNN2D(pl.LightningModule):
```

Use a small ResNet-18-style architecture adapted for 1-channel 64x64 inputs:

```
Input: [B, 1, 64, 64]
Conv2d(1, 16, 3, padding=1) → BN → ReLU
Conv2d(16, 32, 3, padding=1) → BN → ReLU → MaxPool2d(2) → [B, 32, 32, 32]
Conv2d(32, 64, 3, padding=1) → BN → ReLU → MaxPool2d(2) → [B, 64, 16, 16]
Conv2d(64, 128, 3, padding=1) → BN → ReLU → MaxPool2d(2) → [B, 128, 8, 8]
AdaptiveAvgPool2d(1) → Flatten → [B, 128]
Linear(128, 64) → ReLU → Dropout(0.3)
Linear(64, 3) → Softmax (3-class: clean/noisy_ok/bad)
```

- Loss: CrossEntropyLoss with class weights (inverse frequency)
- Optimizer: Adam, lr=1e-3, cosine annealing scheduler
- Monitor: macro F1 for model checkpoint

Training/validation split: temporal split by segment_idx (same rule as all other models).

### Agreement analysis function
```python
def compare_with_stage0(
    cnn2d_preds: pd.DataFrame,
    stage0_preds: pd.DataFrame
) -> pd.DataFrame:
```

Joins both prediction DataFrames on `segment_idx`. Returns a DataFrame with:
- `segment_idx`
- `quality_pred_stage0` (str)
- `quality_pred_cnn2d` (str)
- `agree` (bool)
- `both_predict_bad` (bool)
- `disagreement_flag` (bool): True if the two models disagree AND one of them predicts `bad`

Print a summary: overall agreement rate, confusion matrix between the two models.

### CLI
```
python ecgclean/models/segment_cnn_2d.py train \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segments data/processed/segments.parquet \
  --output models/segment_cnn2d_v1.pt

python ecgclean/models/segment_cnn_2d.py predict \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segments data/processed/segments.parquet \
  --model models/segment_cnn2d_v1.pt \
  --output data/processed/segment_cnn2d_preds.parquet

python ecgclean/models/segment_cnn_2d.py compare \
  --stage0-preds data/processed/segment_quality_preds.parquet \
  --cnn2d-preds data/processed/segment_cnn2d_preds.parquet
```

---

## FILE 2: `ecgclean/features/motif_features.py`

### Purpose
Discover recurring ECG waveform patterns (motifs) from the year-long recording and compute distance-based features for each beat. These features tell the GBM and CNN how unusual a beat is relative to the full history of this patient's ECG — a powerful signal for anomaly detection.

### QRS motif discovery
```python
def discover_qrs_motifs(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ecg_windows: np.ndarray,
    n_clusters: int = 12,
    use_clean_only: bool = True,
    random_seed: int = 42
) -> dict:
```

Cluster the 64-sample ECG windows using k-means (`sklearn.cluster.KMeans`). Use only beats with `label == "clean"` and `hard_filtered == False` if `use_clean_only=True`.

After clustering, assign human-readable labels to clusters by inspecting their centroid shapes. Use these heuristics:
- Mean HR of beats in cluster < 75 bpm AND centroid has clear P-wave and T-wave separation → `"normal_sinus_brady"`
- Mean HR 75-100 bpm, clean QRS morphology → `"normal_sinus"`
- Mean HR > 100 bpm, narrower QRS → `"sinus_tachycardia"`
- High intra-cluster variance (std of window_ptp > 0.3 * mean) → `"noisy_cluster"` — likely a catch-all for borderline beats
- Any cluster where >30% of beats come from `phys_event_window=True` → `"pots_transition"`

Return:
```python
{
    "centroids": np.ndarray,           # shape (n_clusters, 64)
    "cluster_labels": list[str],       # human-readable label per cluster
    "cluster_assignments": np.ndarray, # shape (n_clean_beats,) — cluster index per beat
    "kmeans_model": KMeans,
    "inertia": float,
    "peak_ids": np.ndarray             # peak_ids corresponding to cluster_assignments
}
```

Also implement:
```python
def discover_rr_motifs(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    window_size: int = 10,
    n_clusters: int = 8,
    random_seed: int = 42
) -> dict:
```

Slide a window of `window_size` consecutive RR intervals across the full RR series (from `rr_prev_ms`). Cluster these windows with k-means. Expected motif types: stable sinus, POTS ramp-up, POTS recovery, bradycardic rest, erratic/noisy. Return the same structure as `discover_qrs_motifs` with `centroids` shape `(n_clusters, window_size)`.

### Motif feature computation
```python
def compute_motif_features(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ecg_windows: np.ndarray,
    qrs_motif_dict: dict,
    rr_motif_dict: dict
) -> pd.DataFrame:
```

For each beat, compute:
- `dist_to_nearest_qrs_motif` (float32): Euclidean distance from this beat's 64-sample window to the nearest QRS centroid
- `nearest_qrs_motif_label` (str): human-readable label of the nearest QRS cluster
- `nearest_qrs_motif_idx` (int32): cluster index
- `dist_to_nearest_rr_motif` (float32): Euclidean distance from this beat's centered 10-beat RR window to the nearest RR centroid. For beats with fewer than 10 neighbors (start/end of recording), use available neighbors and zero-pad.
- `nearest_rr_motif_label` (str)
- `nearest_rr_motif_idx` (int32)
- `qrs_anomaly_score` (float32): `dist_to_nearest_qrs_motif / mean_dist_to_nearest_qrs_motif_across_clean_beats` — normalized so that 1.0 = average clean-beat distance, >1.0 = more anomalous than average
- `rr_anomaly_score` (float32): same normalization for RR motifs
- `is_qrs_anomaly` (bool): `qrs_anomaly_score > 2.0` (more than 2x average distance — a soft anomaly flag)
- `is_rr_anomaly` (bool): `rr_anomaly_score > 2.0`

### Save/load motif models
```python
def save_motifs(qrs_motif_dict: dict, rr_motif_dict: dict, output_dir: str) -> None:
def load_motifs(motif_dir: str) -> tuple[dict, dict]:
```

Save using joblib. Include a version key so that stale motif models can be detected.

### CLI
```
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

The `discover` CLI should print: cluster sizes, centroid HR ranges, intra-cluster variance, and the auto-assigned human-readable labels. Plot centroids to terminal using a simple ASCII representation (optional but nice: print centroid waveform as a sparkline using `▁▂▃▄▅▆▇█` characters scaled to amplitude).

---

## Requirements (both files)
- Type hints and docstrings on all functions
- `segment_cnn_2d.py`: `torch`, `pytorch_lightning`, `pywt`, `skimage`, `pandas`, `numpy`, `scikit-learn`
- `motif_features.py`: `pandas`, `numpy`, `scikit-learn`, `joblib` only
- Scalogram cache must be invalidated if `ecg_samples.parquet` modification time is newer than cache
- Motif discovery must be reproducible: same seed → same cluster assignments
- `nearest_qrs_motif_label` and `nearest_rr_motif_label` columns in motif features are string typed — downstream models will one-hot encode them; include a helper function `get_motif_dummies(motif_features_df) -> pd.DataFrame` that returns one-hot encoded columns ready to concatenate with the beat feature matrix

## Files available in working directory
- `data/processed/ecg_samples.parquet`, `peaks.parquet`, `labels.parquet`, `segments.parquet`
- `data/processed/beat_features.parquet`
- `ecgclean/features/beat_features.py` — read for ecg_windows loading pattern
- `ecg_artifact_pipeline_v2_FINAL.md`

Write both files completely. Do not abbreviate or use placeholder comments.
