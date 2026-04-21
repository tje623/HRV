#!/usr/bin/env python3
"""
ecgclean.models.segment_cnn_2d
==============================
Segment-level quality classifier that learns noise signatures from CWT
(Continuous Wavelet Transform) scalogram images.

Runs as a parallel track alongside Stage 0 (``segment_quality.py``).
Agreement between the two models is a strong quality signal; disagreement
flags segments for manual review.

Architecture:
    Input [B, 1, 64, 64]  (1-channel 64×64 Morlet scalogram)
    → 4 Conv2d/BN/ReLU blocks with MaxPool
    → AdaptiveAvgPool2d(1) → [B, 128]
    → Linear head → 3-class softmax (clean / noisy_ok / bad)

CLI
---
    python ecgclean/models/segment_cnn_2d.py train   --ecg-samples ... --segments ... --output ...
    python ecgclean/models/segment_cnn_2d.py predict --ecg-samples ... --segments ... --model ... --output ...
    python ecgclean/models/segment_cnn_2d.py compare --stage0-preds ... --cnn2d-preds ...
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE_HZ, VAL_FRACTION, LGBM_RANDOM_STATE, CNN_MAX_EPOCHS

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pywt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize as skimage_resize
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUALITY_CLASSES = ["clean", "noisy_ok", "bad"]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(QUALITY_CLASSES)}
IDX_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_IDX.items()}
IMAGE_SIZE = (64, 64)
N_SCALES = 64
MIN_SAMPLES = 250  # Minimum ECG samples for a meaningful scalogram (2 s @ 125 Hz)


# ===================================================================== #
#  Scalogram computation                                                #
# ===================================================================== #
def compute_scalogram(
    ecg_segment: np.ndarray,
    sampling_rate: int = SAMPLE_RATE_HZ,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Compute a CWT Morlet scalogram from a raw ECG segment.

    Parameters
    ----------
    ecg_segment : np.ndarray
        1-D array of ECG voltage samples for one segment.
    sampling_rate : int
        Sample rate in Hz (default 125 for Polar H10 — do NOT use 130 or 256).
    image_size : tuple[int, int]
        (height, width) of output image.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``image_size``, values in [0, 1].
        Returns a zero array if the segment has < MIN_SAMPLES samples.
    """
    if len(ecg_segment) < MIN_SAMPLES:
        log.warning(
            "Segment has only %d samples (< %d = 2 s @ 125 Hz) — returning zero scalogram",
            len(ecg_segment),
            MIN_SAMPLES,
        )
        return np.zeros(image_size, dtype=np.float32)

    # 64 logarithmically-spaced scales from 1 to 128
    scales = np.logspace(np.log10(1), np.log10(128), num=N_SCALES)

    # CWT with Morlet wavelet
    coefficients, _ = pywt.cwt(
        ecg_segment.astype(np.float64),
        scales,
        "morl",
        sampling_period=1.0 / sampling_rate,
    )

    # coefficients shape: (n_scales, n_time)
    # Take absolute value for the scalogram (power)
    scalogram = np.abs(coefficients)

    # Resize to image_size using bilinear interpolation
    scalogram_resized = skimage_resize(
        scalogram,
        image_size,
        order=1,  # bilinear
        preserve_range=True,
        anti_aliasing=True,
    )

    # Normalize to [0, 1]
    max_val = np.max(np.abs(scalogram_resized))
    eps = 1e-8
    scalogram_resized = scalogram_resized / (max_val + eps)

    return scalogram_resized.astype(np.float32)


# ===================================================================== #
#  Scalogram disk cache                                                 #
# ===================================================================== #
def _cache_dir(base_dir: str | Path = "data/processed/scalogram_cache") -> Path:
    """Get or create the scalogram cache directory."""
    d = Path(base_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(segment_idx: int, ecg_samples_path: str | None) -> str:
    """Deterministic cache filename for a segment's scalogram."""
    # Include a hash of the ecg_samples file modification time so that
    # stale cache entries are automatically invalidated.
    mtime_hash = ""
    if ecg_samples_path and Path(ecg_samples_path).exists():
        mtime = os.path.getmtime(ecg_samples_path)
        mtime_hash = hashlib.md5(str(mtime).encode()).hexdigest()[:8]
    return f"seg_{segment_idx}_{mtime_hash}.npy"


def _load_or_compute_scalogram(
    segment_idx: int,
    ecg_samples_path: str,
    cache_base: str | Path = "data/processed/scalogram_cache",
) -> np.ndarray:
    """Load scalogram from cache or compute + cache it.

    Uses parquet predicate pushdown — never loads the full ECG table.
    """
    cache = _cache_dir(cache_base)
    key = _cache_key(segment_idx, ecg_samples_path)
    cache_file = cache / key

    if cache_file.exists():
        return np.load(cache_file).astype(np.float32)

    # Load only this segment's ECG via predicate pushdown
    table = pq.read_table(
        ecg_samples_path,
        filters=[("segment_idx", "==", segment_idx)],
        columns=["timestamp_ms", "ecg"],
    )
    seg_ecg = (
        table.to_pandas()
        .sort_values("timestamp_ms")["ecg"]
        .values.astype(np.float64)
    )

    scalogram = compute_scalogram(seg_ecg)
    # Atomic write — np.save appends .npy automatically, so name the temp
    # file with a stem suffix so it still ends in .npy (avoiding double ext).
    tmp_file = cache_file.parent / (cache_file.stem + "_tmp.npy")
    np.save(tmp_file, scalogram)
    os.replace(tmp_file, cache_file)
    return scalogram


def _prewarm_one(args: tuple) -> None:
    """Worker target for parallel cache pre-warm (must be module-level to pickle)."""
    seg_idx, ecg_samples_path, cache_base = args
    _load_or_compute_scalogram(seg_idx, ecg_samples_path, cache_base)


def _prewarm_cache(
    segment_indices: np.ndarray,
    ecg_samples_path: str,
    cache_base: str | Path = "data/processed/scalogram_cache",
) -> None:
    """Compute and cache scalograms for all segments in parallel.

    Uses ProcessPoolExecutor so CWT (CPU-bound) runs across all cores.
    Atomic writes mean workers never corrupt each other's files.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cache = _cache_dir(cache_base)
    missing = [
        int(s) for s in segment_indices
        if not (cache / _cache_key(int(s), ecg_samples_path)).exists()
    ]
    if not missing:
        log.info("Scalogram cache: all %d segments already cached", len(segment_indices))
        return

    n_workers = min(8, os.cpu_count() or 4)
    log.info(
        "Pre-warming scalogram cache: %d/%d segments to compute (%d workers)",
        len(missing), len(segment_indices), n_workers,
    )
    args = [(s, ecg_samples_path, str(cache_base)) for s in missing]
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_prewarm_one, a): a for a in args}
        for fut in as_completed(futures):
            fut.result()  # re-raise any worker exception immediately
            done += 1
            if done % 500 == 0 or done == len(missing):
                log.info("  Cache: %d/%d done", done, len(missing))
    log.info("Scalogram cache ready — %d segments cached", len(segment_indices))


# ===================================================================== #
#  Dataset                                                              #
# ===================================================================== #
class SegmentScalogramDataset(Dataset):
    """PyTorch Dataset for segment-level scalogram images.

    Parameters
    ----------
    ecg_samples_path : str
        Path to ecg_samples.parquet. Loaded per-segment via predicate
        pushdown — the full table is never held in memory.
    segments_df : pd.DataFrame
        Segment metadata with ``segment_idx``, ``quality_label``.
    training : bool
        If True, apply data augmentation (horizontal flip, brightness jitter).
    """

    def __init__(
        self,
        ecg_samples_path: str,
        segments_df: pd.DataFrame,
        training: bool = False,
    ) -> None:
        self.ecg_samples_path = ecg_samples_path
        self.segments_df = segments_df.reset_index(drop=True)
        self.training = training

        self.segment_indices = self.segments_df["segment_idx"].values
        self.labels = np.array([
            LABEL_TO_IDX.get(lbl, 1)  # default to noisy_ok
            for lbl in self.segments_df["quality_label"].values
        ], dtype=np.int64)

        log.info(
            "SegmentScalogramDataset: %d segments, training=%s",
            len(self), self.training,
        )

    def __len__(self) -> int:
        return len(self.segment_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seg_idx = int(self.segment_indices[idx])
        label = int(self.labels[idx])

        scalogram = _load_or_compute_scalogram(
            seg_idx,
            self.ecg_samples_path,
        )

        # ── Augmentation (training only) ─────────────────────────────
        if self.training:
            # Random horizontal flip (time reversal) with 50% probability
            if np.random.random() < 0.5:
                scalogram = np.flip(scalogram, axis=1).copy()

            # Random brightness/contrast jitter (±10%)
            brightness = 1.0 + np.random.uniform(-0.1, 0.1)
            scalogram = np.clip(scalogram * brightness, 0.0, 1.0).astype(np.float32)

        # Add channel dimension: [1, H, W]
        tensor = torch.from_numpy(scalogram).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return tensor, label_tensor


# ===================================================================== #
#  Model                                                                #
# ===================================================================== #
class SegmentQualityCNN2D(pl.LightningModule):
    """Small 2D CNN for segment quality classification from scalograms.

    Architecture follows a ResNet-18-style encoder (without skip connections)
    with 4 conv blocks → adaptive pool → 2-layer linear head → 3-class output.
    """

    def __init__(
        self,
        n_classes: int = 3,
        class_weights: torch.Tensor | None = None,
        lr: float = 1e-3,
        max_epochs: int = CNN_MAX_EPOCHS,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.lr = lr
        self.max_epochs = max_epochs

        # ── Encoder ──────────────────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → [B, 32, 32, 32]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → [B, 64, 16, 16]
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → [B, 128, 8, 8]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # → [B, 128, 1, 1]

        # ── Head ─────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

        # ── Loss ─────────────────────────────────────────────────────
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits [B, n_classes]."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.flatten(1)  # [B, 128]
        return self.head(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        preds = logits.argmax(dim=1)

        self.log(f"{stage}_loss", loss, prog_bar=True)

        # Macro F1 (compute on CPU)
        y_np = y.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        macro_f1 = f1_score(y_np, preds_np, average="macro", zero_division=0.0)
        self.log(f"{stage}_macro_f1", macro_f1, prog_bar=True)

        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict:
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ===================================================================== #
#  Training                                                             #
# ===================================================================== #
def train(
    ecg_samples_path: str,
    segments_path: str,
    output_model_path: str,
    max_epochs: int = CNN_MAX_EPOCHS,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_fraction: float = VAL_FRACTION,
) -> dict:
    """Train the 2D CNN segment quality classifier.

    Parameters
    ----------
    ecg_samples_path : str
        Path to ``ecg_samples.parquet``.
    segments_path : str
        Path to ``segments.parquet`` (must have ``quality_label``).
    output_model_path : str
        Where to save the trained model (``.pt``).
    max_epochs : int
        Maximum training epochs (default 50).
    batch_size : int
        Batch size (default 16).
    lr : float
        Learning rate (default 1e-3).
    val_fraction : float
        Fraction of segments for validation (temporal split, default 0.2).

    Returns
    -------
    dict
        Training summary metrics.
    """
    pl.seed_everything(LGBM_RANDOM_STATE)

    # ── Load data ────────────────────────────────────────────────────
    # ECG samples are NOT loaded into memory — scalograms are computed
    # per-segment via parquet predicate pushdown and cached to disk.
    segments = pd.read_parquet(segments_path)

    log.info("Loaded: %d segments (ECG samples streamed on demand)", len(segments))

    # Filter to segments that have quality_label
    if "quality_label" not in segments.columns:
        raise ValueError("segments.parquet must have 'quality_label' column")

    # ── Temporal split ───────────────────────────────────────────────
    seg_sorted = segments.sort_values("segment_idx").reset_index(drop=True)
    n = len(seg_sorted)
    split_idx = int(n * (1.0 - val_fraction))
    # With very few segments we may get 0 val — handle gracefully
    if n <= 1:
        split_idx = n  # All segments go to train
    else:
        split_idx = max(1, min(split_idx, n - 1))

    train_segs = seg_sorted.iloc[:split_idx].copy()
    val_segs = seg_sorted.iloc[split_idx:].copy()
    has_val = len(val_segs) > 0

    log.info(
        "Temporal split: %d train segments, %d val segments",
        len(train_segs), len(val_segs),
    )

    # ── Class distribution + weights ─────────────────────────────────
    label_counts = train_segs["quality_label"].value_counts()
    log.info("Train label distribution: %s", label_counts.to_dict())

    # Compute inverse-frequency class weights
    total = len(train_segs)
    weights = []
    for cls in QUALITY_CLASSES:
        cnt = label_counts.get(cls, 0)
        w = total / (len(QUALITY_CLASSES) * max(cnt, 1))
        weights.append(w)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    log.info("Class weights: %s", dict(zip(QUALITY_CLASSES, [f"{w:.2f}" for w in weights])))

    # ── Pre-warm scalogram cache (single-threaded, before workers start) ─
    all_seg_indices = seg_sorted["segment_idx"].values
    _prewarm_cache(all_seg_indices, ecg_samples_path)

    # ── Datasets ─────────────────────────────────────────────────────
    train_ds = SegmentScalogramDataset(ecg_samples_path, train_segs, training=True)
    val_ds = SegmentScalogramDataset(ecg_samples_path, val_segs, training=False)

    n_workers = 12
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, persistent_workers=True, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, persistent_workers=True, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = SegmentQualityCNN2D(
        n_classes=3,
        class_weights=class_weights,
        lr=lr,
        max_epochs=max_epochs,
    )
    log.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

    # ── Trainer ──────────────────────────────────────────────────────
    ckpt_dir = Path(output_model_path).parent / "cnn2d_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list = []
    if has_val:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="best-{epoch:02d}-{val_macro_f1:.4f}",
                monitor="val_macro_f1",
                mode="max",
                save_top_k=1,
            )
        )
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_macro_f1",
                patience=15,
                mode="max",
                verbose=True,
            )
        )
    else:
        log.warning("No validation segments — training without EarlyStopping / ModelCheckpoint")
        # Train for a small fixed number of epochs to avoid wasting time
        max_epochs = min(max_epochs, 5)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks if callbacks else None,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        logger=False,
    )

    log.info("Starting training: max_epochs=%d, batch_size=%d", max_epochs, batch_size)
    trainer.fit(model, train_dl, val_dl if has_val else None)

    # ── Load best checkpoint ─────────────────────────────────────────
    if has_val and trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        best_ckpt = trainer.checkpoint_callback.best_model_path
        log.info("Loading best checkpoint: %s", best_ckpt)
        model = SegmentQualityCNN2D.load_from_checkpoint(
            best_ckpt, n_classes=3, class_weights=class_weights,
        )

    # ── Final evaluation ─────────────────────────────────────────────
    model.eval()
    model.cpu()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_dl:
            logits = model(x.cpu())
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(y.numpy().tolist())

    val_metrics: dict = {
        "n_val": len(all_labels),
        "n_classes_in_val": len(set(all_labels)),
    }

    if len(set(all_labels)) > 1:
        macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0.0))
        val_metrics["macro_f1"] = macro_f1
    else:
        val_metrics["macro_f1"] = float("nan")

    # ── Save model artifact ──────────────────────────────────────────
    out = Path(output_model_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_classes": 3,
            "class_names": QUALITY_CLASSES,
            "image_size": IMAGE_SIZE,
            "val_metrics": val_metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        out,
    )
    log.info("Saved model artifact → %s", out)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  Segment 2D CNN Quality Classifier: Training Summary")
    print(f"{'=' * 72}")
    print(f"  Trained at:  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Segments: {len(segments)} total")
    print(f"  Train: {len(train_segs)}  |  Val: {len(val_segs)}")
    print(f"  Val macro F1: {val_metrics.get('macro_f1', float('nan')):.4f}")
    print(f"{'=' * 72}")

    return val_metrics


# ===================================================================== #
#  Prediction                                                           #
# ===================================================================== #
def predict(
    ecg_samples_path: str,
    segments_path: str,
    model_path: str,
    output_path: str,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Generate segment quality predictions from the 2D CNN.

    Parameters
    ----------
    ecg_samples_path, segments_path : str
        Paths to input Parquet files.
    model_path : str
        Path to the trained ``.pt`` model artifact.
    output_path : str
        Where to save predictions Parquet.
    batch_size : int
        Inference batch size.

    Returns
    -------
    pd.DataFrame
        Predictions with ``segment_idx``, ``quality_pred_cnn2d``,
        and per-class probabilities.
    """
    segments = pd.read_parquet(segments_path)

    # Load model
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = SegmentQualityCNN2D(n_classes=ckpt["n_classes"])
    # strict=False because class_weights buffer may be in state_dict
    # from training but not present in the freshly-created model
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.cpu()

    log.info(
        "Loaded CNN2D model (trained %s, %d classes)",
        ckpt.get("trained_at", "?"), ckpt["n_classes"],
    )

    # Dataset (no augmentation) — pre-warm cache then load in parallel
    _prewarm_cache(segments["segment_idx"].values, ecg_samples_path)
    ds = SegmentScalogramDataset(ecg_samples_path, segments, training=False)
    n_workers = min(8, os.cpu_count() or 4)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, persistent_workers=True, pin_memory=True,
    )

    # Predict
    all_probs = []
    with torch.no_grad():
        for x, _ in dl:
            logits = model(x.cpu())
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.numpy())

    if all_probs:
        probs_arr = np.concatenate(all_probs, axis=0)  # (n_segments, 3)
    else:
        probs_arr = np.zeros((0, 3), dtype=np.float32)

    pred_classes = probs_arr.argmax(axis=1)

    result = pd.DataFrame({
        "segment_idx": segments["segment_idx"].values,
        "quality_pred_cnn2d": [IDX_TO_LABEL[int(c)] for c in pred_classes],
        "p_clean_cnn2d": probs_arr[:, 0].astype(np.float32) if len(probs_arr) > 0 else [],
        "p_noisy_ok_cnn2d": probs_arr[:, 1].astype(np.float32) if len(probs_arr) > 0 else [],
        "p_bad_cnn2d": probs_arr[:, 2].astype(np.float32) if len(probs_arr) > 0 else [],
    })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False, compression="snappy")

    log.info("Saved %d segment predictions → %s", len(result), out)

    # ── Summary ──────────────────────────────────────────────────────
    n = len(result)
    print(f"\n{'=' * 72}")
    print("  Segment 2D CNN Predictions")
    print(f"{'=' * 72}")
    print(f"  Total segments: {n}")
    if n > 0:
        for cls in QUALITY_CLASSES:
            cnt = int((result["quality_pred_cnn2d"] == cls).sum())
            print(f"    {cls}: {cnt} ({100.0 * cnt / n:.1f}%)")
    print(f"{'=' * 72}")

    return result


# ===================================================================== #
#  Agreement analysis                                                   #
# ===================================================================== #
def compare_with_stage0(
    cnn2d_preds: pd.DataFrame,
    stage0_preds: pd.DataFrame,
) -> pd.DataFrame:
    """Compare 2D CNN predictions with Stage 0 (LightGBM) predictions.

    Parameters
    ----------
    cnn2d_preds : pd.DataFrame
        Must have ``segment_idx`` and ``quality_pred_cnn2d``.
    stage0_preds : pd.DataFrame
        Must have ``segment_idx`` and ``quality_label`` (Stage 0 prediction).

    Returns
    -------
    pd.DataFrame
        Comparison with ``segment_idx``, ``quality_pred_stage0``,
        ``quality_pred_cnn2d``, ``agree``, ``both_predict_bad``,
        ``disagreement_flag``.
    """
    # Normalize column names
    s0 = stage0_preds[["segment_idx"]].copy()
    if "quality_label" in stage0_preds.columns:
        s0["quality_pred_stage0"] = stage0_preds["quality_label"].values
    elif "quality_pred" in stage0_preds.columns:
        # Map numeric predictions to labels
        s0["quality_pred_stage0"] = [
            IDX_TO_LABEL.get(int(v), "noisy_ok") for v in stage0_preds["quality_pred"].values
        ]
    else:
        raise ValueError("stage0_preds must have 'quality_label' or 'quality_pred'")

    c2d = cnn2d_preds[["segment_idx", "quality_pred_cnn2d"]].copy()

    merged = s0.merge(c2d, on="segment_idx", how="inner")

    merged["agree"] = merged["quality_pred_stage0"] == merged["quality_pred_cnn2d"]
    merged["both_predict_bad"] = (
        (merged["quality_pred_stage0"] == "bad")
        & (merged["quality_pred_cnn2d"] == "bad")
    )
    merged["disagreement_flag"] = (
        ~merged["agree"]
        & (
            (merged["quality_pred_stage0"] == "bad")
            | (merged["quality_pred_cnn2d"] == "bad")
        )
    )

    # ── Print summary ────────────────────────────────────────────────
    n = len(merged)
    n_agree = int(merged["agree"].sum())
    n_both_bad = int(merged["both_predict_bad"].sum())
    n_flag = int(merged["disagreement_flag"].sum())

    print(f"\n{'=' * 72}")
    print("  Stage 0 vs 2D CNN Agreement Analysis")
    print(f"{'=' * 72}")
    print(f"  Segments compared: {n}")
    print(f"  Agreement rate: {n_agree}/{n} ({100.0 * n_agree / max(n, 1):.1f}%)")
    print(f"  Both predict bad: {n_both_bad}")
    print(f"  Disagreement flags (one says bad): {n_flag}")

    if n > 0:
        print(f"\n  Confusion matrix (rows=Stage0, cols=CNN2D):")
        s0_labels = merged["quality_pred_stage0"].values
        c2d_labels = merged["quality_pred_cnn2d"].values
        all_labels = sorted(set(s0_labels) | set(c2d_labels))
        cm = confusion_matrix(s0_labels, c2d_labels, labels=all_labels)
        # Header
        header = "            " + "  ".join(f"{l:>10}" for l in all_labels)
        print(f"  {header}")
        for i, row_label in enumerate(all_labels):
            row_str = "  ".join(f"{v:>10}" for v in cm[i])
            print(f"  {row_label:>10}  {row_str}")

    print(f"{'=' * 72}")

    return merged


# ===================================================================== #
#  CLI                                                                  #
# ===================================================================== #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segment_cnn_2d.py",
        description="2D CNN segment quality classifier from CWT scalograms.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ─────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train the 2D CNN")
    p_train.add_argument("--ecg-samples", required=True)
    p_train.add_argument("--segments", required=True)
    p_train.add_argument("--output", required=True, help="Output model .pt path")
    p_train.add_argument("--max-epochs", type=int, default=CNN_MAX_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-3)

    # ── predict ───────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Generate predictions")
    p_pred.add_argument("--ecg-samples", required=True)
    p_pred.add_argument("--segments", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--output", required=True)

    # ── compare ───────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="Compare with Stage 0 predictions")
    p_cmp.add_argument("--stage0-preds", required=True)
    p_cmp.add_argument("--cnn2d-preds", required=True)

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "train":
        train(
            ecg_samples_path=args.ecg_samples,
            segments_path=args.segments,
            output_model_path=args.output,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    elif args.command == "predict":
        predict(
            ecg_samples_path=args.ecg_samples,
            segments_path=args.segments,
            model_path=args.model,
            output_path=args.output,
        )

    elif args.command == "compare":
        s0 = pd.read_parquet(args.stage0_preds)
        c2d = pd.read_parquet(args.cnn2d_preds)
        result = compare_with_stage0(c2d, s0)


if __name__ == "__main__":
    main()
