You are implementing Step 6 of an ECG artifact detection pipeline. Your task is to write one complete, production-ready Python file: `ecgclean/models/beat_artifact_cnn.py`.

## Context

This is Stage 2a — the hybrid 1D CNN beat-level artifact classifier. It takes a 256-sample raw ECG window (expanded from v6's 64 samples) through convolutional layers, concatenates the resulting embedding with tabular features, and produces `p_artifact_cnn`. This model runs alongside the Stage 1 GBM; their outputs feed the ensemble.

This is a year-long single-lead ECG (Polar H10, ~256Hz) from a POTS patient. Class imbalance is severe (~1.3% artifact). The primary solution to imbalance is **noise augmentation** — corrupting clean beats with synthetic artifact patterns during training — not just reweighting. The noise augmentation strategy is described in detail below and is the most important part of this file.

Framework: PyTorch Lightning. The model must be runnable on CPU (M3 MacBook Pro) and MPS (Apple Silicon GPU).

## Architecture

```
Input: ECG window [Batch, 1, 256]    (256 samples @ ~256Hz ≈ 1 second window)

CNN Branch:
  Conv1d(1, 32, kernel_size=7, padding=3) → BatchNorm1d(32) → ReLU
  Conv1d(32, 64, kernel_size=5, padding=2) → BatchNorm1d(64) → ReLU → MaxPool1d(2)  → [B, 64, 128]
  Conv1d(64, 128, kernel_size=5, padding=2) → BatchNorm1d(128) → ReLU → MaxPool1d(2) → [B, 128, 64]
  Conv1d(128, 128, kernel_size=3, padding=1) → BatchNorm1d(128) → ReLU
  AdaptiveAvgPool1d(1) → Flatten → [B, 128]

Tabular Branch:
  Input: [B, N_tabular]
  Linear(N_tabular, 64) → ReLU → Linear(64, 32) → ReLU → [B, 32]

Fusion:
  Concat([CNN embedding, Tabular embedding]) → [B, 160]
  Linear(160, 64) → ReLU → Dropout(0.3)
  Linear(64, 1) → Sigmoid → p_artifact_cnn
```

The tabular branch receives these features (subset of beat_features.py output — NOT the raw window):
`rr_prev`, `rr_next`, `rr_ratio`, `rr_diff`, `rr_mean`, `rr_prev_2`, `rr_next_2`, `rr_local_mean_5`, `rr_local_sd_5`, `rr_abs_delta_prev`, `rr_abs_delta_next`, `qrs_corr_to_template`, `qrs_corr_prev`, `qrs_corr_next`, `physio_implausible`, `pots_transition_candidate`, `rr_suspicious_short`, `rr_suspicious_long`, `review_priority_score`, `segment_artifact_fraction`, `segment_rr_sd`, `segment_quality_pred`

## Noise Augmentation (Critical — implement completely)

During training, apply these augmentations to a randomly selected fraction (default 10%) of clean beats each epoch to create synthetic artifact examples. Each augmented beat is added to the batch with label=1 (artifact). This is the primary mechanism for addressing class imbalance.

Implement as a function:
```python
def augment_clean_beat_to_artifact(window: np.ndarray, artifact_type: str | None = None) -> np.ndarray:
```

If `artifact_type` is None, choose randomly from the following types with equal probability:

1. **`baseline_wander`**: Add a sinusoidal baseline: `A * sin(2π * f * t + φ)` where `A ~ Uniform(0.3, 1.5) * window.std()`, `f ~ Uniform(0.05, 0.5)` Hz, `φ ~ Uniform(0, 2π)`. Time vector assumes 256Hz.

2. **`electrode_pop`**: Insert a sharp spike at a random location: choose a random index `i`, set `window[i] = window[i] + A` where `A ~ Uniform(3, 8) * window.std()`, randomly positive or negative.

3. **`emg_burst`**: Add a burst of high-frequency noise over a random 30-80 sample window: `noise = A * np.random.randn(burst_len)` where `A ~ Uniform(0.5, 2.0) * window.std()`, placed at a random offset.

4. **`lead_off_transient`**: Set a contiguous block of 10-40 random samples to near-zero (multiply by `Uniform(0, 0.05)`), then add a recovery ramp back to the original amplitude over the following 10-20 samples.

5. **`motion_artifact`**: Add a low-frequency high-amplitude sinusoid: `A * sin(2π * f * t + φ)` where `A ~ Uniform(1.0, 3.0) * window.std()`, `f ~ Uniform(1.0, 5.0)` Hz.

6. **`gaussian_noise`**: Add `noise ~ Normal(0, A)` where `A ~ Uniform(0.2, 0.6) * window.std()`.

Also implement an augmentation-only function for genuine clean beats (applied to all clean beats with lower probability, 20%):
```python
def augment_clean_beat_preserve_label(window: np.ndarray) -> np.ndarray:
```
Apply one of: small Gaussian noise (sigma = 0.05 * std), random amplitude scaling (Uniform(0.9, 1.1)), or small time shift (roll by -2 to +2 samples). These keep the label as 0 (clean) but improve generalization.

## Dataset class
```python
class BeatDataset(torch.utils.data.Dataset):
```
- Takes: peaks_df, labels_df, ecg_samples_df, tabular_feature_columns list
- Constructs 256-sample windows around each R-peak from `ecg_samples_df` (linear interpolation at 256Hz, zero-pad if insufficient data)
- Applies bandpass preprocessing (3-40 Hz Butterworth, use scipy.signal) to raw windows before returning
- Applies `augment_clean_beat_to_artifact` during training (controlled by `training: bool` flag in __init__)
- Returns `(window_tensor, tabular_tensor, label_tensor)` tuples
- Windows are normalized per-beat: subtract mean, divide by std (with epsilon=1e-8 for zero-variance protection)

## Lightning module
```python
class BeatArtifactCNN(pl.LightningModule):
```

- `__init__(self, n_tabular_features, pos_weight, learning_rate=1e-3)`
- `forward(self, window, tabular) -> tensor`
- `training_step`, `validation_step`: use `BCELoss` with `pos_weight` for class imbalance
- `configure_optimizers`: Adam with cosine annealing LR scheduler, T_max=50
- Log: `train_loss`, `val_loss`, `val_pr_auc`, `val_f1_artifact` each epoch
- `val_pr_auc` is the primary metric for model selection (higher = better)

## Training function
```python
def train(
    beat_features_path: str,
    labels_path: str,
    ecg_samples_path: str,
    segment_quality_preds_path: str,
    output_model_path: str,
    val_fraction: float = 0.2,
    max_epochs: int = 100,
    batch_size: int = 512,
    augment_fraction: float = 0.10,
    random_seed: int = 42
) -> dict:
```

- Temporal split by segment_idx (same rule as Stage 1 — assert this)
- Exclude beats from `bad` segments
- Use `WeightedRandomSampler` to oversample artifact beats 15x in the training DataLoader
- Use `pl.Trainer` with:
  - `accelerator="auto"` (will use MPS on Apple Silicon automatically)
  - `max_epochs=max_epochs`
  - `EarlyStopping(monitor="val_pr_auc", patience=15, mode="max")`
  - `ModelCheckpoint(monitor="val_pr_auc", mode="max", save_top_k=1)`
- After training, load best checkpoint and run final evaluation on val set
- Save final model: `torch.save` the state dict + metadata dict (feature columns, tabular columns, normalization stats, val metrics, trained_at)

## Inference function
```python
def predict(
    beat_features: pd.DataFrame,
    ecg_samples_df: pd.DataFrame,
    model_path: str,
    batch_size: int = 1024
) -> pd.DataFrame:
```

Returns DataFrame with `peak_id`, `p_artifact_cnn` (float32).

## CLI
```
python ecgclean/models/beat_artifact_cnn.py train \
  --beat-features data/processed/beat_features.parquet \
  --labels data/processed/labels.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output models/beat_cnn_v1.pt

python ecgclean/models/beat_artifact_cnn.py predict \
  --beat-features data/processed/beat_features.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --model models/beat_cnn_v1.pt \
  --output data/processed/beat_cnn_preds.parquet
```

## Requirements
- `torch`, `pytorch_lightning`, `scipy`, `pandas`, `numpy`, `scikit-learn` only
- MPS/CPU/CUDA all supported via `accelerator="auto"`
- `augment_clean_beat_to_artifact` must be unit-testable: it takes a numpy array and returns a numpy array of the same shape
- No NaN in any tensor — add assertions in the Dataset __getitem__ that raise with a helpful message if NaN is found
- All random operations seeded via `pl.seed_everything(random_seed)` at the start of train()
- Print PR-AUC prominently in the final training summary

## Files available in working directory
- `data/processed/beat_features.parquet`
- `data/processed/labels.parquet`
- `data/processed/ecg_samples.parquet`
- `data/processed/segment_quality_preds.parquet`
- `ecgclean/models/beat_artifact_tabular.py` — read for split strategy reference
- `ecg_artifact_pipeline_v2_FINAL.md`

Write the complete file. Do not abbreviate or use placeholder comments. Every function and class must be fully implemented.
