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
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.pipeline_logging import setup_logger, add_logging_args
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
log = logging.getLogger("ecgclean.segment_cnn_2d")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUALITY_CLASSES = ["clean", "noisy_ok", "bad"]
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(QUALITY_CLASSES)}
IDX_TO_LABEL = {i: lbl for lbl, i in LABEL_TO_IDX.items()}
IMAGE_SIZE = (64, 64)
N_SCALES = 64
MIN_SAMPLES = 2 * SAMPLE_RATE_HZ  # Minimum ECG samples for a meaningful scalogram (2 s @ 130 Hz)


def _select_torch_device(device: str = "auto") -> torch.device:
    """Select an inference/training device with Apple Silicon support."""
    requested = device.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but PyTorch MPS is not available")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device {device!r}; expected auto, mps, cuda, or cpu")


def _lightning_accelerator(device: str = "auto") -> str:
    """Map CLI device names to PyTorch Lightning accelerator names."""
    requested = device.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "gpu"
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but PyTorch MPS is not available")
        return "mps"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available")
        return "gpu"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device {device!r}; expected auto, mps, cuda, or cpu")


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
        Sample rate in Hz (default 130 for Polar H10).
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
            "Segment has only %d samples (< %d = 2 s @ 130 Hz) — returning zero scalogram",
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


class ShardedScalogramCache:
    """Manifest-backed cache storing many scalograms per `.npy` shard."""

    MANIFEST_VERSION = 1

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = _cache_dir(base_dir)
        self.shard_dir = self.base_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.shard_dir / "manifest.json"
        self._manifest = self._load_manifest()
        self._shards = sorted(
            self._manifest.get("shards", []),
            key=lambda item: (int(item["min_segment_idx"]), int(item["max_segment_idx"])),
        )
        self._array_cache: dict[str, np.ndarray] = {}
        self._indices_cache: dict[str, np.ndarray] = {}

    def _empty_manifest(self) -> dict:
        return {
            "version": self.MANIFEST_VERSION,
            "image_size": list(IMAGE_SIZE),
            "shards": [],
        }

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return self._empty_manifest()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self._manifest, handle, indent=2, sort_keys=True)
        os.replace(tmp, self.manifest_path)

    def _candidate_shards(self, segment_idx: int) -> list[dict]:
        return [
            shard for shard in self._shards
            if int(shard["min_segment_idx"]) <= segment_idx <= int(shard["max_segment_idx"])
        ]

    def _load_indices(self, shard: dict) -> np.ndarray:
        name = shard["indices_file"]
        if name not in self._indices_cache:
            self._indices_cache[name] = np.load(self.shard_dir / name, mmap_mode="r")
        return self._indices_cache[name]

    def _load_array(self, shard: dict) -> np.ndarray:
        name = shard["data_file"]
        if name not in self._array_cache:
            if len(self._array_cache) >= 4:
                self._array_cache.pop(next(iter(self._array_cache)))
            self._array_cache[name] = np.load(self.shard_dir / name, mmap_mode="r")
        return self._array_cache[name]

    def locate(self, segment_idx: int) -> tuple[dict, int] | None:
        for shard in self._candidate_shards(segment_idx):
            indices = self._load_indices(shard)
            row = int(np.searchsorted(indices, segment_idx))
            if row < len(indices) and int(indices[row]) == segment_idx:
                return shard, row
        return None

    def has_segment(self, segment_idx: int) -> bool:
        return self.locate(int(segment_idx)) is not None

    def load(self, segment_idx: int) -> np.ndarray:
        located = self.locate(int(segment_idx))
        if located is None:
            raise KeyError(f"Segment {segment_idx} is not in the sharded scalogram cache")
        shard, row = located
        return np.asarray(self._load_array(shard)[row], dtype=np.float32)

    def write_shard(self, segment_indices: list[int], scalograms: list[np.ndarray]) -> None:
        if not segment_indices:
            return
        order = np.argsort(np.asarray(segment_indices, dtype=np.int64))
        indices = np.asarray(segment_indices, dtype=np.int64)[order]
        data = np.stack([scalograms[int(i)] for i in order]).astype(np.float32)
        if data.shape[1:] != IMAGE_SIZE:
            raise ValueError(f"Expected scalogram shape (*, {IMAGE_SIZE}), got {data.shape}")

        shard_id = len(self._manifest["shards"])
        data_file = f"shard_{shard_id:06d}_{int(indices[0])}_{int(indices[-1])}.npy"
        indices_file = f"shard_{shard_id:06d}_{int(indices[0])}_{int(indices[-1])}.indices.npy"

        data_tmp = self.shard_dir / f"{data_file}.tmp"
        indices_tmp = self.shard_dir / f"{indices_file}.tmp"
        with data_tmp.open("wb") as handle:
            np.save(handle, data)
        with indices_tmp.open("wb") as handle:
            np.save(handle, indices)
        os.replace(data_tmp, self.shard_dir / data_file)
        os.replace(indices_tmp, self.shard_dir / indices_file)

        shard = {
            "data_file": data_file,
            "indices_file": indices_file,
            "count": int(len(indices)),
            "min_segment_idx": int(indices[0]),
            "max_segment_idx": int(indices[-1]),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._manifest["shards"].append(shard)
        self._shards.append(shard)
        self._shards.sort(
            key=lambda item: (int(item["min_segment_idx"]), int(item["max_segment_idx"]))
        )
        self._save_manifest()


def _legacy_cache_files(cache: Path, segment_idx: int, ecg_samples_path: str | None) -> list[Path]:
    paths = [cache / f"seg_{segment_idx}.npy"]
    hashed = cache / _cache_key(segment_idx, ecg_samples_path)
    if hashed not in paths:
        paths.append(hashed)
    return paths


def _load_or_compute_scalogram(
    segment_idx: int,
    ecg_samples_path: str,
    cache_base: str | Path = "data/processed/scalogram_cache",
    sharded_cache: ShardedScalogramCache | None = None,
) -> np.ndarray:
    """Load scalogram from cache or compute + cache it.

    Uses parquet predicate pushdown — never loads the full ECG table.
    """
    cache = _cache_dir(cache_base)
    if sharded_cache is None:
        sharded_cache = ShardedScalogramCache(cache)

    if sharded_cache.has_segment(segment_idx):
        return sharded_cache.load(segment_idx)

    for cache_file in _legacy_cache_files(cache, segment_idx, ecg_samples_path):
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
    cache_file = cache / f"seg_{segment_idx}.npy"
    tmp_file = cache / f"seg_{segment_idx}_tmp.npy"
    np.save(tmp_file, scalogram)
    os.replace(tmp_file, cache_file)
    return scalogram


def _compute_scalogram_for_cache(args: tuple) -> tuple[int, np.ndarray]:
    """Worker: compute scalogram from a pre-loaded ECG array.

    ECG is passed in-process — no parquet I/O inside the worker, so all
    cores do pure CPU work (CWT) without fighting over the SSD.
    """
    seg_idx, ecg_array = args
    scalogram = compute_scalogram(ecg_array)
    return int(seg_idx), scalogram


def _prewarm_cache(
    segment_indices: np.ndarray,
    ecg_samples_path: str,
    cache_base: str | Path = "data/processed/scalogram_cache",
    batch_size: int = 2500,
) -> None:
    """Compute and cache scalograms: sequential I/O, then parallel CWT.

    Pattern: main process reads one batch of ECG from parquet (sequential,
    no SSD contention), hands pre-loaded arrays to workers who do pure
    CPU work (CWT). Workers never touch the parquet file.
    """
    from concurrent.futures import ProcessPoolExecutor

    cache = _cache_dir(cache_base)
    sharded_cache = ShardedScalogramCache(cache)

    missing = sorted(
        int(s) for s in segment_indices
        if (
            not sharded_cache.has_segment(int(s))
            and not any(
                legacy_file.exists()
                for legacy_file in _legacy_cache_files(cache, int(s), ecg_samples_path)
            )
        )
    )
    if not missing:
        log.info("Scalogram cache: all %d segments already cached", len(segment_indices))
        return

    n_workers = 12
    log.info(
        "Pre-warming scalogram cache: %d/%d segments to compute (%d workers, batch=%d)",
        len(missing), len(segment_indices), n_workers, batch_size,
    )

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch_start in range(0, len(missing), batch_size):
            batch = missing[batch_start : batch_start + batch_size]
            seg_min, seg_max = batch[0], batch[-1]

            # One sequential range-read covers the whole batch
            table = pq.read_table(
                ecg_samples_path,
                filters=[
                    ("segment_idx", ">=", seg_min),
                    ("segment_idx", "<=", seg_max),
                ],
                columns=["timestamp_ms", "ecg", "segment_idx"],
            )
            df = table.to_pandas()
            ecg_by_seg = {
                int(sid): grp.sort_values("timestamp_ms")["ecg"].values.astype(np.float64)
                for sid, grp in df.groupby("segment_idx")
            }
            del table, df

            worker_args = [
                (seg, ecg_by_seg.get(seg, np.array([], dtype=np.float64)))
                for seg in batch
            ]
            computed = list(pool.map(_compute_scalogram_for_cache, worker_args))
            sharded_cache.write_shard(
                [seg for seg, _ in computed],
                [scalogram for _, scalogram in computed],
            )

            done += len(batch)
            log.info("  Cache: %d/%d done", done, len(missing))

    log.info(
        "Scalogram cache ready — %d segments cached (%d sharded, legacy files still supported)",
        len(segment_indices), len(sharded_cache._shards),
    )


def repack_legacy_cache(
    segment_indices: np.ndarray,
    cache_base: str | Path = "data/processed/scalogram_cache",
    shard_size: int = 5000,
    ecg_samples_path: str | None = None,
) -> dict[str, int]:
    """Convert existing per-segment `.npy` cache files into shard files.

    This does not recompute scalograms and does not delete legacy files. It
    lets an existing cache move to the lower-overhead sharded layout safely.
    """
    cache = _cache_dir(cache_base)
    sharded_cache = ShardedScalogramCache(cache)
    pending_segments: list[int] = []
    pending_scalograms: list[np.ndarray] = []
    summary = {"repacked": 0, "already_sharded": 0, "missing_legacy": 0}

    for raw_seg in segment_indices:
        seg = int(raw_seg)
        if sharded_cache.has_segment(seg):
            summary["already_sharded"] += 1
            continue

        legacy_file = next(
            (
                path for path in _legacy_cache_files(cache, seg, ecg_samples_path)
                if path.exists()
            ),
            None,
        )
        if legacy_file is None:
            summary["missing_legacy"] += 1
            continue

        pending_segments.append(seg)
        pending_scalograms.append(np.load(legacy_file).astype(np.float32))
        if len(pending_segments) >= shard_size:
            sharded_cache.write_shard(pending_segments, pending_scalograms)
            summary["repacked"] += len(pending_segments)
            pending_segments = []
            pending_scalograms = []

    if pending_segments:
        sharded_cache.write_shard(pending_segments, pending_scalograms)
        summary["repacked"] += len(pending_segments)

    log.info(
        "Repacked scalogram cache: %d repacked, %d already sharded, %d missing legacy",
        summary["repacked"], summary["already_sharded"], summary["missing_legacy"],
    )
    return summary


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
        cache_base: str | Path = "data/processed/scalogram_cache",
    ) -> None:
        self.ecg_samples_path = ecg_samples_path
        self.segments_df = segments_df.reset_index(drop=True)
        self.training = training
        self.cache_base = cache_base
        self._sharded_cache: ShardedScalogramCache | None = None

        self.segment_indices = self.segments_df["segment_idx"].values
        self.labels = np.array([
            LABEL_TO_IDX.get(lbl, 1)  # default to noisy_ok
            for lbl in self.segments_df["quality_label"].values
        ], dtype=np.int64)

        log.info(
            "SegmentScalogramDataset: %d segments, training=%s",
            len(self), self.training,
        )

    def _get_sharded_cache(self) -> ShardedScalogramCache:
        if self._sharded_cache is None:
            self._sharded_cache = ShardedScalogramCache(self.cache_base)
        return self._sharded_cache

    def __len__(self) -> int:
        return len(self.segment_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seg_idx = int(self.segment_indices[idx])
        label = int(self.labels[idx])

        scalogram = _load_or_compute_scalogram(
            seg_idx,
            self.ecg_samples_path,
            cache_base=self.cache_base,
            sharded_cache=self._get_sharded_cache(),
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
    device: str = "auto",
    cache_base: str | Path = "data/processed/scalogram_cache",
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
    device : str
        Torch device for training: auto, mps, cuda, or cpu.
    cache_base : str | Path
        Scalogram cache directory.

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
    _prewarm_cache(all_seg_indices, ecg_samples_path, cache_base=cache_base)

    # ── Datasets ─────────────────────────────────────────────────────
    train_ds = SegmentScalogramDataset(
        ecg_samples_path, train_segs, training=True, cache_base=cache_base
    )
    val_ds = SegmentScalogramDataset(
        ecg_samples_path, val_segs, training=False, cache_base=cache_base
    )

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

    accelerator = _lightning_accelerator(device)
    log.info("Training accelerator: %s", accelerator)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks if callbacks else None,
        accelerator=accelerator,
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
    batch_size: int = 512,
    device: str = "auto",
    cache_base: str | Path = "data/processed/scalogram_cache",
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
    device : str
        Torch device for inference: auto, mps, cuda, or cpu.
    cache_base : str | Path
        Scalogram cache directory.

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
    infer_device = _select_torch_device(device)
    model.to(infer_device)

    log.info(
        "Loaded CNN2D model (trained %s, %d classes) on %s",
        ckpt.get("trained_at", "?"), ckpt["n_classes"], infer_device,
    )

    # Dataset (no augmentation) — pre-warm cache then load in parallel
    _prewarm_cache(segments["segment_idx"].values, ecg_samples_path, cache_base=cache_base)
    ds = SegmentScalogramDataset(
        ecg_samples_path, segments, training=False, cache_base=cache_base
    )
    n_workers = 10
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, persistent_workers=True,
        pin_memory=(infer_device.type == "cuda"),
    )

    # Predict
    all_probs = []
    with torch.inference_mode():
        for x, _ in dl:
            x = x.to(infer_device, non_blocking=(infer_device.type == "cuda"))
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

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
        description=(
            "Segment-level 2D CNN quality model built from ECG scalograms.\n\n"
            "Use `train` to build a `.pt` model artifact, `predict` to create a\n"
            "Parquet table of per-segment class probabilities, `repack-cache` to\n"
            "convert legacy per-segment scalogram `.npy` files into shard files,\n"
            "and `compare` to compare CNN predictions against Stage 0 outputs."
        ),
        epilog=(
            "Examples:\n"
            "  Train a model:\n"
            "    python /Volumes/xHRV/Scripts/models/segment_cnn_2d.py train \\\n"
            "      --ecg-samples /Volumes/xHRV/Data/Processed/ecg_samples.parquet \\\n"
            "      --segments /Volumes/xHRV/Data/Processed/segments.parquet \\\n"
            "      --output /Volumes/xHRV/Models/segment_cnn2d_v_current.pt \\\n"
            "      --device mps --batch-size 128\n\n"
            "  Generate predictions:\n"
            "    python /Volumes/xHRV/Scripts/models/segment_cnn_2d.py predict \\\n"
            "      --ecg-samples /Volumes/xHRV/Data/Processed/ecg_samples.parquet \\\n"
            "      --segments /Volumes/xHRV/Data/Processed/segments.parquet \\\n"
            "      --model /Volumes/xHRV/Models/segment_cnn2d_v_current.pt \\\n"
            "      --output /Volumes/xHRV/Data/Processed/segment_cnn2d_preds.parquet \\\n"
            "      --device mps --batch-size 512\n\n"
            "  Repack existing cache files:\n"
            "    python /Volumes/xHRV/Scripts/models/segment_cnn_2d.py repack-cache \\\n"
            "      --segments /Volumes/xHRV/Data/Processed/segments.parquet \\\n"
            "      --ecg-samples /Volumes/xHRV/Data/Processed/ecg_samples.parquet \\\n"
            "      --cache-base /Volumes/xHRV/Scripts/data/processed/scalogram_cache \\\n"
            "      --shard-size 5000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="{train,predict,repack-cache,compare}",
    )

    # ── train ─────────────────────────────────────────────────────────
    p_train = sub.add_parser(
        "train",
        help="Train the CNN and write a `.pt` model artifact",
        description=(
            "Train the segment quality CNN from raw ECG samples and segment labels.\n\n"
            "Required inputs:\n"
            "  --ecg-samples  Parquet with at least `segment_idx`, `timestamp_ms`, `ecg`\n"
            "  --segments     Parquet with at least `segment_idx`, `quality_label`\n"
            "  --output       Destination `.pt` model file\n\n"
            "Creates:\n"
            "  - the requested model artifact (`.pt`)\n"
            "  - `cnn2d_checkpoints/` beside the output model path\n"
            "  - scalogram cache files under `--cache-base`"
        ),
        epilog=(
            "Example:\n"
            "  python /Volumes/xHRV/Scripts/models/segment_cnn_2d.py train \\\n"
            "    --ecg-samples /Volumes/xHRV/Data/Processed/ecg_samples.parquet \\\n"
            "    --segments /Volumes/xHRV/Data/Processed/segments.parquet \\\n"
            "    --output /Volumes/xHRV/Models/segment_cnn2d_v_current.pt \\\n"
            "    --device mps --batch-size 128"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_train.add_argument(
        "--ecg-samples",
        required=True,
        help="Input ECG Parquet; must contain `segment_idx`, `timestamp_ms`, and `ecg`.",
    )
    p_train.add_argument(
        "--segments",
        required=True,
        help="Input segment Parquet; must contain `segment_idx` and `quality_label`.",
    )
    p_train.add_argument(
        "--output",
        required=True,
        help="Output `.pt` model path, for example `/Volumes/xHRV/Models/segment_cnn2d_v_current.pt`.",
    )
    p_train.add_argument(
        "--max-epochs",
        type=int,
        default=CNN_MAX_EPOCHS,
        help=f"Maximum training epochs. Default: {CNN_MAX_EPOCHS}.",
    )
    p_train.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size. Default: 16.",
    )
    p_train.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. Default: 1e-3.",
    )
    p_train.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Training device. `auto` prefers MPS on this Mac if available.",
    )
    p_train.add_argument(
        "--cache-base",
        default="data/processed/scalogram_cache",
        help="Scalogram cache directory. Default: `data/processed/scalogram_cache`.",
    )

    # ── predict ───────────────────────────────────────────────────────
    p_pred = sub.add_parser(
        "predict",
        help="Run inference and write a predictions Parquet",
        description=(
            "Run the trained CNN over every segment and write per-class probabilities.\n\n"
            "Required inputs:\n"
            "  --ecg-samples  Parquet with at least `segment_idx`, `timestamp_ms`, `ecg`\n"
            "  --segments     Parquet listing segments to score; must contain `segment_idx`\n"
            "  --model        Trained `.pt` artifact from `train`\n"
            "  --output       Destination predictions Parquet\n\n"
            "Creates:\n"
            "  - predictions Parquet with columns:\n"
            "    `segment_idx`, `quality_pred_cnn2d`, `p_clean_cnn2d`,\n"
            "    `p_noisy_ok_cnn2d`, `p_bad_cnn2d`"
        ),
        epilog=(
            "Example:\n"
            "  python /Volumes/xHRV/Scripts/models/segment_cnn_2d.py predict \\\n"
            "    --ecg-samples /Volumes/xHRV/Data/Processed/ecg_samples.parquet \\\n"
            "    --segments /Volumes/xHRV/Data/Processed/segments.parquet \\\n"
            "    --model /Volumes/xHRV/Models/segment_cnn2d_v_current.pt \\\n"
            "    --output /Volumes/xHRV/Data/Processed/segment_cnn2d_preds.parquet \\\n"
            "    --device mps --batch-size 512"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_pred.add_argument(
        "--ecg-samples",
        required=True,
        help="Input ECG Parquet; must contain `segment_idx`, `timestamp_ms`, and `ecg`.",
    )
    p_pred.add_argument(
        "--segments",
        required=True,
        help="Input segment Parquet to score; must contain `segment_idx`.",
    )
    p_pred.add_argument(
        "--model",
        required=True,
        help="Trained `.pt` model artifact from `train`.",
    )
    p_pred.add_argument(
        "--output",
        required=True,
        help="Output predictions Parquet, for example `/Volumes/xHRV/Data/Processed/segment_cnn2d_preds.parquet`.",
    )
    p_pred.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size. Default: 512.",
    )
    p_pred.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Inference device. `auto` prefers MPS on this Mac if available.",
    )
    p_pred.add_argument(
        "--cache-base",
        default="data/processed/scalogram_cache",
        help="Scalogram cache directory. Default: `data/processed/scalogram_cache`.",
    )

    # ── repack-cache ─────────────────────────────────────────────────
    p_repack = sub.add_parser(
        "repack-cache",
        help="Convert legacy per-segment cache files into shard files",
        description=(
            "Scan an existing scalogram cache for legacy `seg_<id>.npy` files and copy\n"
            "them into the new sharded cache layout without recomputing wavelets.\n\n"
            "Required inputs:\n"
            "  --segments     Parquet with `segment_idx` values to repack\n\n"
            "Creates:\n"
            "  - shard `.npy` files plus `manifest.json` under `--cache-base/shards`\n\n"
            "Does not:\n"
            "  - recompute scalograms\n"
            "  - delete old legacy cache files"
        ),
        epilog=(
            "Example:\n"
            "  python /Volumes/xHRV/Scripts/models/segment_cnn_2d.py repack-cache \\\n"
            "    --segments /Volumes/xHRV/Data/Processed/segments.parquet \\\n"
            "    --ecg-samples /Volumes/xHRV/Data/Processed/ecg_samples.parquet \\\n"
            "    --cache-base /Volumes/xHRV/Scripts/data/processed/scalogram_cache \\\n"
            "    --shard-size 5000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_repack.add_argument(
        "--segments",
        required=True,
        help="Segment Parquet whose `segment_idx` column defines which cache files to repack.",
    )
    p_repack.add_argument(
        "--cache-base",
        default="data/processed/scalogram_cache",
        help="Existing scalogram cache directory. Default: `data/processed/scalogram_cache`.",
    )
    p_repack.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="How many segments to pack per shard file. Default: 5000.",
    )
    p_repack.add_argument(
        "--ecg-samples",
        default=None,
        help="Optional ECG path used only to check old hashed legacy cache filenames.",
    )

    # ── compare ───────────────────────────────────────────────────────
    p_cmp = sub.add_parser(
        "compare",
        help="Compare Stage 0 predictions against CNN predictions",
        description=(
            "Join Stage 0 outputs and CNN predictions on `segment_idx` and print\n"
            "agreement statistics.\n\n"
            "Required inputs:\n"
            "  --stage0-preds  Parquet with `segment_idx` and either `quality_label` or `quality_pred`\n"
            "  --cnn2d-preds   Parquet produced by `predict`\n\n"
            "Creates:\n"
            "  - no new files\n"
            "  - console summary including agreement counts and confusion matrix"
        ),
        epilog=(
            "Example:\n"
            "  python /Volumes/xHRV/Scripts/models/segment_cnn_2d.py compare \\\n"
            "    --stage0-preds /Volumes/xHRV/Data/Processed/segment_quality_preds.parquet \\\n"
            "    --cnn2d-preds /Volumes/xHRV/Data/Processed/segment_cnn2d_preds.parquet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_cmp.add_argument(
        "--stage0-preds",
        required=True,
        help="Stage 0 predictions Parquet with `segment_idx` plus `quality_label` or `quality_pred`.",
    )
    p_cmp.add_argument(
        "--cnn2d-preds",
        required=True,
        help="CNN predictions Parquet produced by `predict`.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    add_logging_args(parser)
    args = parser.parse_args()
    global log
    log = setup_logger("segment_cnn_2d", args=args, disable_log=args.no_log)
    log.info("=== segment_cnn_2d started | command=%s ===", args.command)

    if args.command == "train":
        train(
            ecg_samples_path=args.ecg_samples,
            segments_path=args.segments,
            output_model_path=args.output,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            cache_base=args.cache_base,
        )

    elif args.command == "predict":
        predict(
            ecg_samples_path=args.ecg_samples,
            segments_path=args.segments,
            model_path=args.model,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
            cache_base=args.cache_base,
        )

    elif args.command == "compare":
        s0 = pd.read_parquet(args.stage0_preds)
        c2d = pd.read_parquet(args.cnn2d_preds)
        result = compare_with_stage0(c2d, s0)

    elif args.command == "repack-cache":
        segments = pd.read_parquet(args.segments, columns=["segment_idx"])
        summary = repack_legacy_cache(
            segments["segment_idx"].values,
            cache_base=args.cache_base,
            shard_size=args.shard_size,
            ecg_samples_path=args.ecg_samples,
        )
        print(
            "Repacked scalogram cache: "
            f"{summary['repacked']} repacked, "
            f"{summary['already_sharded']} already sharded, "
            f"{summary['missing_legacy']} missing legacy"
        )


if __name__ == "__main__":
    main()
