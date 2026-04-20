You are implementing Step 9 (the final step) of an ECG artifact detection pipeline. Your task is to write one complete, production-ready Python file: `ecgclean/models/pretrain_ssl.py`.

## Context

This is Stage 3 — self-supervised pretraining on the full unlabeled ECG corpus. The goal is to learn a patient-specific representation of what this person's ECG looks like before any artifact labels are introduced. The pretrained encoder is then used to initialize the Stage 2a CNN (`beat_artifact_cnn.py`), reducing the number of labeled beats needed for high artifact detection performance.

This is a year-long single-lead ECG (Polar H10, ~256Hz) from a POTS patient with high vagal tone and documented extreme but genuine HR dynamics. The full recording is approximately 365 days × 24 hours × 3600 seconds × 256 samples = ~8 billion samples, but in practice the Polar H10 records intermittently. The usable corpus likely contains tens of millions of ECG samples — all of which are available for pretraining regardless of whether they have beat-level labels.

Framework: PyTorch Lightning. Must run efficiently on MPS (Apple Silicon M3).

## Architecture: Denoising Autoencoder

The pretraining task is denoising: corrupt a clean ECG window, train the encoder-decoder to reconstruct the original. At inference time, the encoder (convolutional backbone) is extracted and used to initialize Stage 2a.

### Encoder (shared with Stage 2a)
The encoder architecture must be **identical** to the CNN branch of `BeatArtifactCNN` in `beat_artifact_cnn.py`:

```
Input: [B, 1, 256]
Conv1d(1, 32, 7, padding=3) → BN → ReLU
Conv1d(32, 64, 5, padding=2) → BN → ReLU → MaxPool1d(2)  → [B, 64, 128]
Conv1d(64, 128, 5, padding=2) → BN → ReLU → MaxPool1d(2) → [B, 128, 64]
Conv1d(128, 128, 3, padding=1) → BN → ReLU
AdaptiveAvgPool1d(1) → Flatten → [B, 128]
```

### Decoder (pretraining only — discarded after pretraining)
```
Input: embedding [B, 128]
Linear(128, 128 * 16) → Reshape → [B, 128, 16]
ConvTranspose1d(128, 64, 4, stride=2, padding=1) → BN → ReLU  → [B, 64, 32]
ConvTranspose1d(64, 32, 4, stride=2, padding=1) → BN → ReLU   → [B, 32, 64]
ConvTranspose1d(32, 16, 4, stride=2, padding=1) → BN → ReLU   → [B, 16, 128]
ConvTranspose1d(16, 1, 4, stride=2, padding=1)                 → [B, 1, 256]
```

Loss: MSE between reconstructed window and the original clean window.

### Corruption function (training only)
```python
def corrupt_ecg_window(window: np.ndarray, corruption_level: float = 0.5) -> np.ndarray:
```

Apply one or more of the following corruptions, randomly chosen each call. `corruption_level` scales the intensity (0.0 = no corruption, 1.0 = full intensity):

1. **Gaussian noise**: Add `Normal(0, corruption_level * 0.3 * std)`
2. **Masking**: Zero out a contiguous block of `int(corruption_level * 64)` random samples
3. **Baseline wander**: Add `A * sin(2πft + φ)` where `A = corruption_level * std`, `f ~ Uniform(0.05, 0.5)` Hz
4. **Amplitude scaling**: Multiply by `Uniform(1 - 0.3*corruption_level, 1 + 0.3*corruption_level)`
5. **Combined** (apply 2-3 of the above simultaneously, with reduced individual intensities)

Use the same augmentation types as `beat_artifact_cnn.py` for consistency.

## Dataset class
```python
class PretrainDataset(torch.utils.data.Dataset):
```

- Loads all beats from `peaks.parquet` — labeled AND unlabeled, from all segments including `noisy_ok` (but NOT from segments predicted `bad` by the Stage 0 model, if available)
- Extracts 256-sample windows from `ecg_samples_df` using linear interpolation (same method as `BeatDataset` in `beat_artifact_cnn.py`)
- Applies bandpass preprocessing (3-40 Hz Butterworth) to raw windows
- Normalizes per-beat (subtract mean, divide by std, epsilon=1e-8)
- Returns `(corrupted_window_tensor, clean_window_tensor)` pairs — the model learns to reconstruct clean from corrupted
- Corruption is applied on-the-fly each epoch, so the same window gets different corruptions each time

## Lightning module
```python
class ECGDenoisingAutoencoder(pl.LightningModule):
```

- `__init__(self, corruption_level=0.5, learning_rate=1e-3)`
- `forward(self, corrupted_window) -> reconstructed_window`
- `training_step` / `validation_step`: MSE loss; also log `val_reconstruction_snr` (signal-to-noise ratio of reconstruction vs original, in dB: `10 * log10(var(original) / var(original - reconstructed))`)
- `configure_optimizers`: Adam with cosine annealing, T_max=50, eta_min=1e-5

## Encoder extraction and transfer
```python
def extract_encoder_weights(autoencoder_checkpoint_path: str) -> dict:
```

Loads the trained autoencoder. Extracts only the encoder (convolutional backbone) weights as a state dict. Returns a dict ready to be loaded into the CNN branch of `BeatArtifactCNN` via `model.cnn_branch.load_state_dict(encoder_weights, strict=True)`.

```python
def initialize_cnn_from_pretrained(
    beat_cnn_model: torch.nn.Module,
    encoder_weights: dict,
    freeze_encoder: bool = False
) -> torch.nn.Module:
```

Loads the encoder weights into the CNN branch of `BeatArtifactCNN`. If `freeze_encoder=True`, sets `requires_grad=False` for all encoder parameters (useful for initial fine-tuning). Returns the modified model. Print which layers were loaded and whether they are frozen.

## Training function
```python
def pretrain(
    peaks_path: str,
    ecg_samples_path: str,
    segment_quality_preds_path: str | None,
    output_checkpoint_path: str,
    output_encoder_path: str,
    val_fraction: float = 0.1,
    max_epochs: int = 50,
    batch_size: int = 1024,
    corruption_level: float = 0.5,
    random_seed: int = 42
) -> dict:
```

- Temporal split by segment_idx
- Use ALL available beats (labeled and unlabeled) except from `bad` segments
- Train the full autoencoder
- Save: full autoencoder checkpoint to `output_checkpoint_path`, encoder-only weights to `output_encoder_path`
- Return metrics dict with final val MSE loss and val reconstruction SNR

Print training summary including: total beats used, val MSE, val SNR, and the path to the encoder weights file that should be passed to `beat_artifact_cnn.py`.

## Reconstruction visualization
```python
def visualize_reconstruction(
    checkpoint_path: str,
    ecg_samples_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    n_examples: int = 5,
    output_path: str | None = None
) -> None:
```

Load the trained autoencoder. For `n_examples` randomly chosen beats, print an ASCII comparison of:
1. The original clean window
2. The corrupted input
3. The reconstructed output

Use the `▁▂▃▄▅▆▇█` sparkline characters (scale amplitude to 8 levels). This gives a quick sanity check that the decoder is actually learning to reconstruct rather than output the mean. If `output_path` is provided, also save as a matplotlib figure.

## Integration instructions (written as comments at the top of the file)

The file must begin with a multi-line comment block explaining:
1. How to run pretraining: command and expected runtime on M3 MacBook Pro
2. How to transfer weights to Stage 2a: the exact `initialize_cnn_from_pretrained` call sequence
3. What benchmark to check: compare Stage 2a PR-AUC trained from scratch vs from pretrained encoder; if pretrained is not better after 20 epochs, try `freeze_encoder=True` for the first 5 epochs then unfreeze

## CLI
```
python ecgclean/models/pretrain_ssl.py pretrain \
  --peaks data/processed/peaks.parquet \
  --ecg-samples data/processed/ecg_samples.parquet \
  --segment-quality-preds data/processed/segment_quality_preds.parquet \
  --output-checkpoint models/autoencoder_pretrained.pt \
  --output-encoder models/encoder_pretrained_weights.pt \
  --max-epochs 50 \
  --batch-size 1024

python ecgclean/models/pretrain_ssl.py visualize \
  --checkpoint models/autoencoder_pretrained.pt \
  --ecg-samples data/processed/ecg_samples.parquet \
  --peaks data/processed/peaks.parquet \
  --n-examples 5
```

## Requirements
- Type hints and docstrings on all functions
- `torch`, `pytorch_lightning`, `scipy`, `pandas`, `numpy` only
- The encoder architecture must be byte-for-byte identical to the CNN branch in `beat_artifact_cnn.py` — read that file before writing this one and copy the layer definitions exactly. This is not optional; weight transfer will silently fail if there is any shape mismatch.
- `accelerator="auto"` for MPS support
- `pl.seed_everything(random_seed)` at the start of `pretrain()`
- No NaN in any tensor — assert in Dataset __getitem__
- The encoder weights file must be loadable independently of the full autoencoder — it is a plain state dict, not a Lightning checkpoint

## Files available in working directory
- `data/processed/peaks.parquet`, `ecg_samples.parquet`, `segment_quality_preds.parquet`
- `ecgclean/models/beat_artifact_cnn.py` — **read this first** to copy the exact encoder architecture
- `ecg_artifact_pipeline_v2_FINAL.md`

Write the complete file. Do not abbreviate or use placeholder comments. Every function and class must be fully implemented.
