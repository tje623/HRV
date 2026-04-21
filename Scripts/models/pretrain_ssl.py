#!/usr/bin/env python3
"""
ecgclean/models/pretrain_ssl.py — Stage 3: Self-Supervised ECG Pretraining

Self-supervised denoising autoencoder for learning patient-specific ECG
representations from the full unlabeled corpus.  The encoder is
**architecturally identical** to the CNN branch of ``BeatArtifactCNN`` in
``beat_artifact_cnn.py``, allowing direct weight transfer after pretraining.

The pretraining task is *denoising*: corrupt a clean ECG window with
synthetic noise, then train the encoder–decoder to reconstruct the
original.  At transfer time, the encoder (convolutional backbone) is
extracted and used to initialize Stage 2a — significantly reducing the
number of labeled beats needed for high artifact-detection performance.

═══════════════════════════════════════════════════════════════════════════
  Integration instructions
═══════════════════════════════════════════════════════════════════════════

  1. Run pretraining (~5-15 min on M3 MacBook Pro with full corpus):

       python ecgclean/models/pretrain_ssl.py pretrain \\
         --peaks data/processed/peaks.parquet \\
         --ecg-samples data/processed/ecg_samples.parquet \\
         --segment-quality-preds data/processed/segment_quality_preds.parquet \\
         --output-checkpoint models/autoencoder_pretrained.pt \\
         --output-encoder models/encoder_pretrained_weights.pt \\
         --max-epochs 50 --batch-size 1024

  2. Transfer weights to Stage 2a:

       from ecgclean.models.pretrain_ssl import (
           extract_encoder_weights,
           initialize_cnn_from_pretrained,
       )
       from ecgclean.models.beat_artifact_cnn import BeatArtifactCNN

       encoder_weights = extract_encoder_weights(
           "models/encoder_pretrained_weights.pt"
       )
       model = BeatArtifactCNN(n_tabular_features=22)
       model = initialize_cnn_from_pretrained(
           model, encoder_weights, freeze_encoder=False
       )
       # Then train as usual …

  3. Benchmark: compare Stage 2a PR-AUC trained from scratch vs from
     pretrained encoder.  If pretrained is not better after 20 epochs,
     try ``freeze_encoder=True`` for the first 5 epochs then unfreeze:

       model = initialize_cnn_from_pretrained(model, encoder_weights,
                                              freeze_encoder=True)
       # train for 5 epochs …
       for p in model.cnn.parameters():
           p.requires_grad = True
       # continue training …

═══════════════════════════════════════════════════════════════════════════

Usage:
    # Pretrain
    python ecgclean/models/pretrain_ssl.py pretrain \\
        --peaks data/processed/peaks.parquet \\
        --ecg-samples data/processed/ecg_samples.parquet \\
        --segment-quality-preds data/processed/segment_quality_preds.parquet \\
        --output-checkpoint models/autoencoder_pretrained.pt \\
        --output-encoder models/encoder_pretrained_weights.pt \\
        --max-epochs 50 --batch-size 1024

    # Visualize reconstructions
    python ecgclean/models/pretrain_ssl.py visualize \\
        --checkpoint models/autoencoder_pretrained.pt \\
        --ecg-samples data/processed/ecg_samples.parquet \\
        --peaks data/processed/peaks.parquet \\
        --n-examples 5
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, Dataset

# ─── Project-root injection for direct script execution ─────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SAMPLE_RATE_HZ,
    WINDOW_SIZE_SAMPLES,
    CNN_LEARNING_RATE,
    CNN_MAX_EPOCHS,
    CNN_BATCH_SIZE,
    VAL_FRACTION,
    LGBM_RANDOM_STATE,
)

# ─── Logging ────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─── Constants (imported from config: SAMPLE_RATE_HZ, WINDOW_SIZE_SAMPLES) ──

# Sparkline characters for ASCII waveform rendering
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_windows(
    peak_timestamps_ms: np.ndarray,
    peak_segment_ids: np.ndarray,
    ecg_samples_df: pd.DataFrame,
) -> np.ndarray:
    """Extract and bandpass-filter ECG windows (WINDOW_SIZE_SAMPLES samples) for all peaks.

    For each peak, a 1-second window centred on the R-peak timestamp is
    constructed by linear interpolation of the raw ECG samples at 125 Hz.
    If no ECG data is available for a peak's segment, the window is
    all-zeros (zero-padded).

    After extraction, a 4th-order Butterworth bandpass filter (3–40 Hz)
    is applied to each window.

    Args:
        peak_timestamps_ms: Timestamp of each peak in nanoseconds.
        peak_segment_ids: Segment index for each peak.
        ecg_samples_df: Raw ECG samples with columns
            ``timestamp_ms``, ``ecg``, ``segment_idx``.

    Returns:
        Array of shape ``(n_beats, WINDOW_SIZE_SAMPLES)`` with filtered windows (float32).
    """
    n_beats = len(peak_timestamps_ms)
    windows = np.zeros((n_beats, WINDOW_SIZE_SAMPLES), dtype=np.float32)

    # Group ECG samples by segment for fast lookup
    ecg_by_seg: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if len(ecg_samples_df) > 0:
        for seg_idx, group in ecg_samples_df.groupby("segment_idx"):
            gs = group.sort_values("timestamp_ms")
            ecg_by_seg[int(seg_idx)] = (
                gs["timestamp_ms"].values.astype(np.int64),
                gs["ecg"].values.astype(np.float32),
            )

    sample_interval_ms = int(1000 / SAMPLE_RATE_HZ)  # 8 ms per sample at 125 Hz
    half_window = WINDOW_SIZE_SAMPLES // 2

    n_with_data = 0
    for i in range(n_beats):
        seg_idx = int(peak_segment_ids[i])
        if seg_idx not in ecg_by_seg:
            continue  # window stays all-zeros

        seg_ts, seg_ecg = ecg_by_seg[seg_idx]
        if len(seg_ts) < 2:
            continue  # need ≥2 points for interpolation

        peak_ts = int(peak_timestamps_ms[i])
        t_start = peak_ts - half_window * sample_interval_ms
        target_ts = (
            t_start + np.arange(WINDOW_SIZE_SAMPLES, dtype=np.int64) * sample_interval_ms
        )

        windows[i] = np.interp(
            target_ts.astype(np.float64),
            seg_ts.astype(np.float64),
            seg_ecg,
            left=0.0,
            right=0.0,
        ).astype(np.float32)
        n_with_data += 1

    logger.info(
        "Window extraction: %d / %d beats have ECG data", n_with_data, n_beats
    )

    # Bandpass filter (3–40 Hz, 4th-order Butterworth, zero-phase)
    sos = butter(4, [3.0, 40.0], btype="bandpass", fs=SAMPLE_RATE_HZ, output="sos")
    n_filtered = 0
    for i in range(n_beats):
        if np.any(windows[i] != 0):
            try:
                filtered = sosfiltfilt(sos, windows[i]).astype(np.float32)
                if not np.any(np.isnan(filtered)):
                    windows[i] = filtered
                    n_filtered += 1
            except ValueError:
                pass  # keep unfiltered on edge cases

    logger.info("Bandpass filtered: %d / %d windows", n_filtered, n_beats)
    return windows


# ═══════════════════════════════════════════════════════════════════════════════
# CORRUPTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def corrupt_ecg_window(
    window: np.ndarray, corruption_level: float = 0.5
) -> np.ndarray:
    """Apply random corruption(s) to an ECG window for denoising pretraining.

    Applies one or more of five corruption types: Gaussian noise, masking,
    baseline wander, amplitude scaling, or a combination of 2–3 of the above
    with reduced individual intensities.

    ``corruption_level`` scales the intensity (0.0 = no corruption,
    1.0 = full intensity).

    The same augmentation types used by ``beat_artifact_cnn.py`` are re-used
    here for consistency.

    Args:
        window: 1-D float32 array of shape ``(WINDOW_SIZE_SAMPLES,)``.
        corruption_level: Corruption intensity in [0, 1].

    Returns:
        Corrupted window (same shape, float32).  The original is not
        modified — a copy is made.
    """
    window = window.copy().astype(np.float32)
    n = len(window)
    sig_std = float(window.std())
    if sig_std < 1e-10:
        sig_std = 1e-3  # fallback for flat windows

    corruption_type = random.choice(
        ["gaussian", "masking", "baseline_wander", "amplitude_scaling", "combined"]
    )

    if corruption_type == "gaussian":
        noise_std = corruption_level * 0.3 * sig_std
        window += (np.random.randn(n) * noise_std).astype(np.float32)

    elif corruption_type == "masking":
        mask_len = int(corruption_level * 64)
        if mask_len > 0 and mask_len < n:
            start = np.random.randint(0, max(1, n - mask_len))
            window[start : start + mask_len] = 0.0

    elif corruption_type == "baseline_wander":
        amp = corruption_level * sig_std
        freq = np.random.uniform(0.05, 0.5)
        phase = np.random.uniform(0, 2.0 * np.pi)
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE_HZ
        window += (amp * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)

    elif corruption_type == "amplitude_scaling":
        lo = 1.0 - 0.3 * corruption_level
        hi = 1.0 + 0.3 * corruption_level
        scale = np.random.uniform(lo, hi)
        window *= scale

    elif corruption_type == "combined":
        # Apply 2–3 corruptions with reduced individual intensities
        n_corruptions = random.randint(2, 3)
        sub_types = random.sample(
            ["gaussian", "masking", "baseline_wander", "amplitude_scaling"],
            n_corruptions,
        )
        reduced_level = corruption_level * 0.5  # halve intensity for combined
        for sub_type in sub_types:
            if sub_type == "gaussian":
                noise_std = reduced_level * 0.3 * sig_std
                window += (np.random.randn(n) * noise_std).astype(np.float32)
            elif sub_type == "masking":
                mask_len = int(reduced_level * 64)
                if mask_len > 0 and mask_len < n:
                    start = np.random.randint(0, max(1, n - mask_len))
                    window[start : start + mask_len] = 0.0
            elif sub_type == "baseline_wander":
                amp = reduced_level * sig_std
                freq = np.random.uniform(0.05, 0.5)
                phase = np.random.uniform(0, 2.0 * np.pi)
                t = np.arange(n, dtype=np.float64) / SAMPLE_RATE_HZ
                window += (amp * np.sin(2.0 * np.pi * freq * t + phase)).astype(
                    np.float32
                )
            elif sub_type == "amplitude_scaling":
                lo = 1.0 - 0.3 * reduced_level
                hi = 1.0 + 0.3 * reduced_level
                scale = np.random.uniform(lo, hi)
                window *= scale

    return window


# ═══════════════════════════════════════════════════════════════════════════════
# SPARKLINE RENDERING
# ═══════════════════════════════════════════════════════════════════════════════


def _sparkline(values: np.ndarray, width: int = 40) -> str:
    """Render a 1-D signal as an ASCII sparkline using Unicode block chars.

    Downsamples or upsamples to ``width`` characters. Amplitude is mapped
    to 8 levels using ``▁▂▃▄▅▆▇█``.

    Args:
        values: 1-D numeric array.
        width: Number of characters in the output sparkline.

    Returns:
        Sparkline string of length ``width``.
    """
    if len(values) == 0:
        return "·" * width

    # Resample to target width
    indices = np.linspace(0, len(values) - 1, width).astype(int)
    resampled = np.array(values, dtype=np.float64)[indices]

    vmin, vmax = float(np.nanmin(resampled)), float(np.nanmax(resampled))
    if vmax - vmin < 1e-12:
        return SPARKLINE_CHARS[3] * width  # flat → mid-level

    # Map to 0–7 range
    normalised = (resampled - vmin) / (vmax - vmin)
    indices_8 = np.clip((normalised * 7.999).astype(int), 0, 7)
    return "".join(SPARKLINE_CHARS[i] for i in indices_8)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════


class PretrainDataset(Dataset):
    """PyTorch Dataset for self-supervised ECG pretraining.

    Loads ALL beats from ``peaks.parquet`` — labeled and unlabeled, from
    all segments except those predicted ``bad`` by Stage 0 (if available).

    Windows are pre-extracted, bandpass-filtered, and per-beat normalised
    (zero mean, unit std) during ``__init__``.  Corruption is applied
    on-the-fly each epoch so the same window gets different corruptions
    each time.

    Returns:
        ``(corrupted_window_tensor, clean_window_tensor)`` of shape
        ``(1, WINDOW_SIZE_SAMPLES)`` each (single channel).
    """

    def __init__(
        self,
        windows: np.ndarray,
        corruption_level: float = 0.5,
    ) -> None:
        """Initialise the pretrain dataset.

        Args:
            windows: Pre-extracted, bandpass-filtered, normalised ECG
                windows of shape ``(n_beats, WINDOW_SIZE_SAMPLES)``.
            corruption_level: Corruption intensity for the denoising task.
        """
        super().__init__()
        self.windows = windows.astype(np.float32)
        self.corruption_level = corruption_level

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = self.windows[idx].copy()
        corrupted = corrupt_ecg_window(clean, self.corruption_level)

        clean_t = torch.from_numpy(clean).unsqueeze(0)  # [1, WINDOW_SIZE_SAMPLES]
        corrupted_t = torch.from_numpy(corrupted).unsqueeze(0)  # [1, WINDOW_SIZE_SAMPLES]

        # Safety: no NaN allowed
        assert not torch.isnan(clean_t).any(), f"NaN in clean window at idx={idx}"
        assert not torch.isnan(corrupted_t).any(), f"NaN in corrupted window at idx={idx}"

        return corrupted_t, clean_t


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL — DENOISING AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════════════


class ECGDenoisingAutoencoder(pl.LightningModule):
    """Denoising autoencoder for self-supervised ECG representation learning.

    The **encoder** is architecturally identical to the CNN branch of
    ``BeatArtifactCNN`` in ``beat_artifact_cnn.py``::

        Input: [B, 1, WINDOW_SIZE_SAMPLES]
        Conv1d(1, 32, 7, padding=3) → BN → ReLU
        Conv1d(32, 64, 5, padding=2) → BN → ReLU → MaxPool1d(2)  → [B, 64, 128]
        Conv1d(64, 128, 5, padding=2) → BN → ReLU → MaxPool1d(2) → [B, 128, 64]
        Conv1d(128, 128, 3, padding=1) → BN → ReLU
        AdaptiveAvgPool1d(1) → Flatten → [B, 128]

    The **decoder** is used only during pretraining and discarded after::

        Input: embedding [B, 128]
        Linear(128, 128 * 16) → Reshape → [B, 128, 16]
        ConvTranspose1d(128, 64, 4, stride=2, padding=1) → BN → ReLU  → [B, 64, 32]
        ConvTranspose1d(64, 32, 4, stride=2, padding=1) → BN → ReLU   → [B, 32, 64]
        ConvTranspose1d(32, 16, 4, stride=2, padding=1) → BN → ReLU   → [B, 16, 128]
        ConvTranspose1d(16, 1, 4, stride=2, padding=1)                 → [B, 1, WINDOW_SIZE_SAMPLES]

    Loss: MSE between reconstructed and original clean window.

    Args:
        corruption_level: Corruption intensity for input corruption.
        learning_rate: Initial learning rate for Adam optimizer.
    """

    def __init__(
        self,
        corruption_level: float = 0.5,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.corruption_level = corruption_level
        self.learning_rate = learning_rate

        # ── Encoder: identical to BeatArtifactCNN.cnn ────────────────
        # IMPORTANT: This must be byte-for-byte identical to the CNN branch
        # in beat_artifact_cnn.py for weight transfer to work with strict=True
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # ── Decoder: pretraining only ────────────────────────────────
        self.decoder_linear = nn.Linear(128, 128 * 16)
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

        # Loss
        self.mse_loss = nn.MSELoss()

    def forward(self, corrupted_window: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode corrupted window, decode to reconstruction.

        Args:
            corrupted_window: Tensor of shape ``[B, 1, WINDOW_SIZE_SAMPLES]``.

        Returns:
            Reconstructed window tensor of shape ``[B, 1, WINDOW_SIZE_SAMPLES]``.
        """
        embedding = self.encoder(corrupted_window)  # [B, 128]
        decoded = self.decoder_linear(embedding)  # [B, 128*16]
        decoded = decoded.view(-1, 128, 16)  # [B, 128, 16]
        reconstructed = self.decoder_convs(decoded)  # [B, 1, WINDOW_SIZE_SAMPLES]
        return reconstructed

    def encode(self, window: torch.Tensor) -> torch.Tensor:
        """Extract the 128-dim embedding (encoder only, no decoder).

        Args:
            window: Tensor of shape ``[B, 1, WINDOW_SIZE_SAMPLES]``.

        Returns:
            Embedding tensor of shape ``[B, 128]``.
        """
        return self.encoder(window)

    def _compute_snr(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute signal-to-noise ratio (SNR) in dB.

        SNR = 10 * log10(var(original) / var(original - reconstructed))

        Args:
            original: Clean window tensor.
            reconstructed: Reconstructed window tensor.

        Returns:
            Scalar SNR in dB.
        """
        signal_var = torch.var(original)
        noise_var = torch.var(original - reconstructed)
        if noise_var < 1e-12:
            return torch.tensor(60.0, device=original.device)  # near-perfect
        snr = 10.0 * torch.log10(signal_var / noise_var + 1e-12)
        return snr

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step: reconstruct clean from corrupted.

        Args:
            batch: ``(corrupted, clean)`` tensors of shape ``[B, 1, WINDOW_SIZE_SAMPLES]``.
            batch_idx: Batch index.

        Returns:
            MSE loss.
        """
        corrupted, clean = batch
        reconstructed = self(corrupted)
        loss = self.mse_loss(reconstructed, clean)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step: MSE loss and reconstruction SNR.

        Args:
            batch: ``(corrupted, clean)`` tensors of shape ``[B, 1, WINDOW_SIZE_SAMPLES]``.
            batch_idx: Batch index.
        """
        corrupted, clean = batch
        reconstructed = self(corrupted)
        loss = self.mse_loss(reconstructed, clean)
        snr = self._compute_snr(clean, reconstructed)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_reconstruction_snr", snr, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Adam optimizer with cosine annealing scheduler.

        Returns:
            Optimizer configuration dict for PyTorch Lightning.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CNN_MAX_EPOCHS, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODER EXTRACTION & WEIGHT TRANSFER
# ═══════════════════════════════════════════════════════════════════════════════


def extract_encoder_weights(autoencoder_checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Extract encoder-only weights from a trained autoencoder checkpoint.

    Loads the trained autoencoder and returns only the encoder (convolutional
    backbone) weights as a state dict.  The returned dict is ready to be
    loaded into the CNN branch of ``BeatArtifactCNN``::

        model.cnn.load_state_dict(encoder_weights, strict=True)

    If the path ends with ``_weights.pt``, assumes it's already an
    encoder-only state dict (saved by ``pretrain()``).

    Args:
        autoencoder_checkpoint_path: Path to the autoencoder checkpoint
            (``.pt`` file) or encoder-only weights file.

    Returns:
        State dict mapping encoder layer names to tensors.
    """
    ckpt = torch.load(
        autoencoder_checkpoint_path, map_location="cpu", weights_only=False
    )

    # If this is already an encoder-only weights file
    if "encoder_state_dict" in ckpt:
        logger.info(
            "Loaded encoder weights directly from %s",
            autoencoder_checkpoint_path,
        )
        return ckpt["encoder_state_dict"]

    # Full autoencoder checkpoint — extract encoder prefix
    if "state_dict" in ckpt:
        full_sd = ckpt["state_dict"]
    else:
        full_sd = ckpt

    encoder_prefix = "encoder."
    encoder_sd: dict[str, torch.Tensor] = {}
    for key, value in full_sd.items():
        if key.startswith(encoder_prefix):
            # Strip the "encoder." prefix so keys match BeatArtifactCNN.cnn
            new_key = key[len(encoder_prefix) :]
            encoder_sd[new_key] = value

    if not encoder_sd:
        raise ValueError(
            f"No encoder weights found in checkpoint: {autoencoder_checkpoint_path}"
        )

    logger.info(
        "Extracted %d encoder parameters from %s",
        len(encoder_sd),
        autoencoder_checkpoint_path,
    )
    return encoder_sd


def initialize_cnn_from_pretrained(
    beat_cnn_model: torch.nn.Module,
    encoder_weights: dict[str, torch.Tensor],
    freeze_encoder: bool = False,
) -> torch.nn.Module:
    """Load pretrained encoder weights into a BeatArtifactCNN model.

    Loads the encoder weights into the ``cnn`` branch of the model.
    If ``freeze_encoder=True``, sets ``requires_grad=False`` for all
    encoder parameters (useful for initial fine-tuning epochs before
    unfreezing).

    Args:
        beat_cnn_model: A ``BeatArtifactCNN`` instance (or any model
            with a ``.cnn`` attribute that is an ``nn.Sequential``).
        encoder_weights: State dict from ``extract_encoder_weights()``.
        freeze_encoder: If ``True``, freeze all encoder parameters.

    Returns:
        The modified model (same instance, modified in-place).
    """
    # Load weights into the CNN branch
    beat_cnn_model.cnn.load_state_dict(encoder_weights, strict=True)
    logger.info("✓ Loaded pretrained encoder weights into model.cnn")

    # Report which layers were loaded
    for name, param in beat_cnn_model.cnn.named_parameters():
        status = "frozen" if freeze_encoder else "trainable"
        logger.info("  %-40s  shape=%-20s  %s", name, str(tuple(param.shape)), status)
        if freeze_encoder:
            param.requires_grad = False

    if freeze_encoder:
        n_frozen = sum(
            1 for p in beat_cnn_model.cnn.parameters() if not p.requires_grad
        )
        logger.info(
            "✓ Froze %d encoder parameters (set requires_grad=False)", n_frozen
        )

    return beat_cnn_model


# ═══════════════════════════════════════════════════════════════════════════════
# PRETRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def _prepare_windows(
    peaks_path: str,
    ecg_samples_path: str,
    segment_quality_preds_path: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load data, exclude bad segments, extract and normalise ECG windows.

    Args:
        peaks_path: Path to peaks.parquet.
        ecg_samples_path: Path to ecg_samples.parquet.
        segment_quality_preds_path: Optional path to segment_quality_preds.parquet
            to exclude beats from segments predicted as ``bad``.

    Returns:
        Tuple of (normalised_windows, segment_ids) where normalised_windows
        has shape ``(n_beats, WINDOW_SIZE_SAMPLES)`` and segment_ids has shape ``(n_beats,)``.
    """
    peaks_df = pd.read_parquet(peaks_path)
    ecg_samples_df = pd.read_parquet(ecg_samples_path)

    logger.info("Loaded: %d peaks, %d ECG samples", len(peaks_df), len(ecg_samples_df))

    # Exclude beats from segments predicted "bad"
    if segment_quality_preds_path is not None:
        seg_preds_path = Path(segment_quality_preds_path)
        if seg_preds_path.exists():
            seg_preds = pd.read_parquet(seg_preds_path)
            bad_segments = set(
                seg_preds.loc[
                    seg_preds["quality_pred"] == "bad", "segment_idx"
                ].tolist()
            )
            if bad_segments:
                n_before = len(peaks_df)
                peaks_df = peaks_df[~peaks_df["segment_idx"].isin(bad_segments)]
                logger.info(
                    "Excluded %d beats from %d 'bad' segments",
                    n_before - len(peaks_df),
                    len(bad_segments),
                )

    if len(peaks_df) == 0:
        logger.warning("No peaks after filtering — returning empty windows")
        return np.zeros((0, WINDOW_SIZE_SAMPLES), dtype=np.float32), np.array([], dtype=np.int64)

    # Extract windows
    windows = _extract_windows(
        peak_timestamps_ms=peaks_df["timestamp_ms"].values,
        peak_segment_ids=peaks_df["segment_idx"].values,
        ecg_samples_df=ecg_samples_df,
    )

    # Per-beat normalisation: subtract mean, divide by std
    eps = 1e-8
    for i in range(len(windows)):
        w = windows[i]
        mean = w.mean()
        std = w.std()
        if std > eps:
            windows[i] = (w - mean) / std
        else:
            windows[i] = w - mean  # flat window → zero-centred

    # Replace any remaining NaN with 0
    nan_mask = np.isnan(windows)
    if nan_mask.any():
        n_nan = nan_mask.sum()
        logger.warning("Replaced %d NaN values with 0 in windows", n_nan)
        windows[nan_mask] = 0.0

    segment_ids = peaks_df["segment_idx"].values.astype(np.int64)
    return windows, segment_ids


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
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run self-supervised pretraining of the denoising autoencoder.

    Performs temporal split by segment_idx (earlier segments → train,
    later → val).  Uses ALL available beats (labeled and unlabeled)
    except those from segments predicted ``bad`` by Stage 0.

    Saves:
        1. Full autoencoder checkpoint → ``output_checkpoint_path``
        2. Encoder-only weights → ``output_encoder_path``

    Args:
        peaks_path: Path to ``peaks.parquet``.
        ecg_samples_path: Path to ``ecg_samples.parquet``.
        segment_quality_preds_path: Optional path to
            ``segment_quality_preds.parquet`` to exclude bad segments.
        output_checkpoint_path: Where to save the full autoencoder checkpoint.
        output_encoder_path: Where to save the encoder-only weights.
        val_fraction: Fraction of segments for validation (temporal split).
        max_epochs: Maximum training epochs.
        batch_size: Training batch size.
        corruption_level: Corruption intensity for denoising task.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with keys: ``val_loss``, ``val_reconstruction_snr``,
        ``total_beats``, ``train_beats``, ``val_beats``.
    """
    pl.seed_everything(random_seed)

    # ── Prepare data ─────────────────────────────────────────────────
    windows, segment_ids = _prepare_windows(
        peaks_path, ecg_samples_path, segment_quality_preds_path
    )
    n_total = len(windows)

    if n_total == 0:
        logger.warning("No beats available for pretraining")
        return {
            "val_loss": float("nan"),
            "val_reconstruction_snr": float("nan"),
            "total_beats": 0,
            "train_beats": 0,
            "val_beats": 0,
        }

    # ── Temporal split by segment_idx ────────────────────────────────
    unique_segs = np.sort(np.unique(segment_ids))
    n_val_segs = max(1, int(len(unique_segs) * val_fraction))
    val_seg_set = set(unique_segs[-n_val_segs:].tolist())  # latest segments

    train_mask = np.array([s not in val_seg_set for s in segment_ids])
    val_mask = ~train_mask

    train_windows = windows[train_mask]
    val_windows = windows[val_mask]

    logger.info(
        "Temporal split: %d train, %d val (%d segments, %d val segments)",
        len(train_windows),
        len(val_windows),
        len(unique_segs),
        n_val_segs,
    )

    # ── Datasets & DataLoaders ───────────────────────────────────────
    train_ds = PretrainDataset(train_windows, corruption_level=corruption_level)
    val_ds = PretrainDataset(val_windows, corruption_level=corruption_level)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    has_val = len(val_windows) > 0
    val_dl: DataLoader | None = None
    if has_val:
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

    # ── Model ────────────────────────────────────────────────────────
    model = ECGDenoisingAutoencoder(
        corruption_level=corruption_level,
        learning_rate=CNN_LEARNING_RATE,
    )

    # ── Callbacks ────────────────────────────────────────────────────
    callbacks: list[pl.Callback] = []
    effective_epochs = max_epochs

    if has_val:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=10,
                verbose=True,
            )
        )
    else:
        # No validation — reduce epochs to prevent overfitting blindly
        effective_epochs = min(max_epochs, 5)
        logger.warning(
            "No validation set — reducing max_epochs to %d", effective_epochs
        )

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=effective_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=max(1, len(train_dl) // 5),
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # ── Extract metrics ──────────────────────────────────────────────
    val_loss = float("nan")
    val_snr = float("nan")

    if has_val and trainer.callback_metrics:
        if "val_loss" in trainer.callback_metrics:
            val_loss = float(trainer.callback_metrics["val_loss"])
        if "val_reconstruction_snr" in trainer.callback_metrics:
            val_snr = float(trainer.callback_metrics["val_reconstruction_snr"])

    # ── Save full autoencoder checkpoint ─────────────────────────────
    Path(output_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    full_ckpt = {
        "state_dict": model.state_dict(),
        "corruption_level": corruption_level,
        "learning_rate": CNN_LEARNING_RATE,
        "val_loss": val_loss,
        "val_reconstruction_snr": val_snr,
        "total_beats": n_total,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(full_ckpt, output_checkpoint_path)
    logger.info("Saved full autoencoder checkpoint → %s", output_checkpoint_path)

    # ── Save encoder-only weights ────────────────────────────────────
    Path(output_encoder_path).parent.mkdir(parents=True, exist_ok=True)
    encoder_sd = {k: v.cpu() for k, v in model.encoder.state_dict().items()}
    encoder_ckpt = {
        "encoder_state_dict": encoder_sd,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "val_loss": val_loss,
        "val_reconstruction_snr": val_snr,
        "total_beats_pretrained": n_total,
    }
    torch.save(encoder_ckpt, output_encoder_path)
    logger.info("Saved encoder-only weights → %s", output_encoder_path)

    # ── Summary ──────────────────────────────────────────────────────
    metrics = {
        "val_loss": val_loss,
        "val_reconstruction_snr": val_snr,
        "total_beats": n_total,
        "train_beats": int(train_mask.sum()),
        "val_beats": int(val_mask.sum()),
    }

    print()
    print("=" * 72)
    print("  Self-Supervised Pretraining Complete")
    print("=" * 72)
    print(f"  Total beats:          {n_total:,}")
    print(f"  Train / Val:          {int(train_mask.sum()):,} / {int(val_mask.sum()):,}")
    print(f"  Val MSE loss:         {val_loss:.6f}")
    print(f"  Val SNR (dB):         {val_snr:.2f}")
    print(f"  Autoencoder ckpt:     {output_checkpoint_path}")
    print(f"  Encoder weights:      {output_encoder_path}")
    print()
    print("  Next step: transfer weights to Stage 2a CNN with:")
    print(f"    encoder_weights = extract_encoder_weights('{output_encoder_path}')")
    print("    model = initialize_cnn_from_pretrained(beat_cnn_model, encoder_weights)")
    print("=" * 72)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════


def visualize_reconstruction(
    checkpoint_path: str,
    ecg_samples_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    n_examples: int = 5,
    output_path: str | None = None,
) -> None:
    """Visualise autoencoder reconstructions as ASCII sparklines.

    Loads the trained autoencoder and for ``n_examples`` randomly chosen
    beats, prints an ASCII comparison of the original clean window, the
    corrupted input, and the reconstructed output.

    If ``output_path`` is provided, also saves a matplotlib figure.

    Args:
        checkpoint_path: Path to the autoencoder checkpoint.
        ecg_samples_df: Raw ECG samples DataFrame.
        peaks_df: Peaks DataFrame.
        n_examples: Number of examples to visualise.
        output_path: Optional path to save a matplotlib figure.
    """
    # ── Load model ───────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ECGDenoisingAutoencoder(
        corruption_level=ckpt.get("corruption_level", 0.5),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    # ── Extract windows ──────────────────────────────────────────────
    windows = _extract_windows(
        peak_timestamps_ms=peaks_df["timestamp_ms"].values,
        peak_segment_ids=peaks_df["segment_idx"].values,
        ecg_samples_df=ecg_samples_df,
    )

    # Per-beat normalise
    eps = 1e-8
    for i in range(len(windows)):
        w = windows[i]
        std = w.std()
        if std > eps:
            windows[i] = (w - w.mean()) / std
        else:
            windows[i] = w - w.mean()

    # Filter to non-zero windows only
    nonzero_idx = [i for i in range(len(windows)) if np.any(windows[i] != 0)]
    if not nonzero_idx:
        # Fall back to all windows (including zero-padded)
        nonzero_idx = list(range(len(windows)))

    n_examples = min(n_examples, len(nonzero_idx))
    chosen = random.sample(nonzero_idx, n_examples)

    print()
    print("=" * 72)
    print("  Reconstruction Visualisation")
    print("=" * 72)

    fig_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for idx in chosen:
        clean = windows[idx].copy()
        corrupted = corrupt_ecg_window(clean, corruption_level=0.5)

        clean_t = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0)  # [1, 1, WINDOW_SIZE_SAMPLES]
        corrupted_t = torch.from_numpy(corrupted).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            reconstructed_t = model(corrupted_t)

        reconstructed = reconstructed_t.squeeze().numpy()

        # MSE and SNR for this example
        mse = float(np.mean((clean - reconstructed) ** 2))
        sig_var = float(np.var(clean))
        noise_var = float(np.var(clean - reconstructed))
        snr = 10.0 * np.log10(sig_var / (noise_var + 1e-12)) if noise_var > 1e-12 else 60.0

        print(f"\n  Beat index {idx}  (MSE={mse:.6f}, SNR={snr:.1f} dB)")
        print(f"    Original:      {_sparkline(clean, 50)}")
        print(f"    Corrupted:     {_sparkline(corrupted, 50)}")
        print(f"    Reconstructed: {_sparkline(reconstructed, 50)}")

        fig_data.append((clean, corrupted, reconstructed))

    print("=" * 72)

    # ── Optional matplotlib figure ───────────────────────────────────
    if output_path is not None and fig_data:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                n_examples, 3, figsize=(15, 3 * n_examples), squeeze=False
            )

            for row, (clean, corrupted, reconstructed) in enumerate(fig_data):
                t = np.arange(WINDOW_SIZE_SAMPLES) / SAMPLE_RATE_HZ * 1000  # ms

                axes[row, 0].plot(t, clean, "b-", linewidth=0.8)
                axes[row, 0].set_title(f"Original (beat {chosen[row]})")
                axes[row, 0].set_ylabel("Amplitude")

                axes[row, 1].plot(t, corrupted, "r-", linewidth=0.8)
                axes[row, 1].set_title("Corrupted")

                axes[row, 2].plot(t, reconstructed, "g-", linewidth=0.8)
                mse = float(np.mean((clean - reconstructed) ** 2))
                axes[row, 2].set_title(f"Reconstructed (MSE={mse:.4f})")

                for ax in axes[row]:
                    ax.set_xlabel("Time (ms)")

            plt.tight_layout()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved visualisation figure → %s", output_path)
        except ImportError:
            logger.warning("matplotlib not installed — skipping figure export")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Command-line interface for self-supervised pretraining."""
    parser = argparse.ArgumentParser(
        description="ECG self-supervised pretraining (denoising autoencoder)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── pretrain ─────────────────────────────────────────────────────
    p_pretrain = subparsers.add_parser(
        "pretrain", help="Pretrain denoising autoencoder on unlabeled ECG"
    )
    p_pretrain.add_argument(
        "--peaks", required=True, help="Path to peaks.parquet"
    )
    p_pretrain.add_argument(
        "--ecg-samples", required=True, help="Path to ecg_samples.parquet"
    )
    p_pretrain.add_argument(
        "--segment-quality-preds",
        default=None,
        help="Path to segment_quality_preds.parquet (optional, excludes bad segs)",
    )
    p_pretrain.add_argument(
        "--output-checkpoint",
        default="models/autoencoder_pretrained.pt",
        help="Output path for full autoencoder checkpoint",
    )
    p_pretrain.add_argument(
        "--output-encoder",
        default="models/encoder_pretrained_weights.pt",
        help="Output path for encoder-only weights",
    )
    p_pretrain.add_argument(
        "--val-fraction", type=float, default=0.1, help="Val segment fraction"
    )
    p_pretrain.add_argument(
        "--max-epochs", type=int, default=CNN_MAX_EPOCHS, help=f"Maximum training epochs (default: {CNN_MAX_EPOCHS})"
    )
    p_pretrain.add_argument(
        "--batch-size", type=int, default=1024, help="Training batch size"
    )
    p_pretrain.add_argument(
        "--corruption-level", type=float, default=0.5, help="Corruption intensity"
    )
    p_pretrain.add_argument(
        "--seed", type=int, default=LGBM_RANDOM_STATE, help=f"Random seed (default: {LGBM_RANDOM_STATE})"
    )

    # ── visualize ────────────────────────────────────────────────────
    p_viz = subparsers.add_parser(
        "visualize", help="Visualise autoencoder reconstructions"
    )
    p_viz.add_argument(
        "--checkpoint", required=True, help="Path to autoencoder checkpoint"
    )
    p_viz.add_argument(
        "--ecg-samples", required=True, help="Path to ecg_samples.parquet"
    )
    p_viz.add_argument(
        "--peaks", required=True, help="Path to peaks.parquet"
    )
    p_viz.add_argument(
        "--n-examples", type=int, default=5, help="Number of examples"
    )
    p_viz.add_argument(
        "--output", default=None, help="Optional output path for matplotlib figure"
    )

    args = parser.parse_args()

    if args.command == "pretrain":
        pretrain(
            peaks_path=args.peaks,
            ecg_samples_path=args.ecg_samples,
            segment_quality_preds_path=args.segment_quality_preds,
            output_checkpoint_path=args.output_checkpoint,
            output_encoder_path=args.output_encoder,
            val_fraction=args.val_fraction,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            corruption_level=args.corruption_level,
            random_seed=args.seed,
        )

    elif args.command == "visualize":
        ecg_samples_df = pd.read_parquet(args.ecg_samples)
        peaks_df = pd.read_parquet(args.peaks)
        visualize_reconstruction(
            checkpoint_path=args.checkpoint,
            ecg_samples_df=ecg_samples_df,
            peaks_df=peaks_df,
            n_examples=args.n_examples,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
