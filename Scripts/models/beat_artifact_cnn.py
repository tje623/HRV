#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SAMPLE_RATE_HZ,
    WINDOW_SIZE_SAMPLES,
    CNN_BATCH_SIZE,
    CNN_LEARNING_RATE,
    CNN_MAX_EPOCHS,
    CNN_DROPOUT,
    VAL_FRACTION,
    LGBM_RANDOM_STATE,
)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ecgclean.models.beat_artifact_cnn")


def _get_device() -> torch.device:
    """Return the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Constants (imported from config: SAMPLE_RATE_HZ, WINDOW_SIZE_SAMPLES)

TABULAR_COLUMNS: list[str] = [
    # RR interval context
    "rr_prev", "rr_next", "rr_ratio", "rr_diff", "rr_mean",
    "rr_prev_2", "rr_next_2",
    "rr_local_mean_5", "rr_local_sd_5",
    "rr_abs_delta_prev", "rr_abs_delta_next", "rr_delta_ratio_next",
    # QRS morphology + global template
    "qrs_corr_to_template", "qrs_corr_prev", "qrs_corr_next",
    "global_corr_clean",
    # Raw ECG window stats
    "window_mean", "window_std", "window_ptp", "window_min", "window_max",
    # Label-free signal quality
    "window_iqr", "window_kurtosis", "window_zcr",
    "r_peak_snr", "window_hf_noise_rms", "window_wander_slope",
    "window_energy_ratio",
    # Physiological constraint flags
    "physio_implausible",
    "pots_transition_candidate", "tachy_transition_candidate",
    "hr_suspicious_low", "hr_suspicious_high",
    "rr_suspicious_short", "rr_suspicious_long",
    "is_added_peak",
    # Segment-level context
    "review_priority_score",
    "segment_artifact_fraction", "segment_rr_sd",
    "segment_clean_beat_count", "segment_quality_pred",
]

ARTIFACT_TYPES: list[str] = [
    "baseline_wander",
    "electrode_pop",
    "emg_burst",
    "lead_off_transient",
    "motion_artifact",
    "gaussian_noise",
]


# ═══════════════════════════
# NOISE AUGMENTATION
# ═══════════════════════════


def augment_clean_beat_to_artifact(
    window: np.ndarray,
    artifact_type: str | None = None,
) -> np.ndarray:
    
    window = window.copy().astype(np.float32)
    n = len(window)
    sig_std = float(window.std())
    if sig_std < 1e-10:
        sig_std = 1e-3

    if artifact_type is None:
        artifact_type = random.choice(ARTIFACT_TYPES)

    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE_HZ

    if artifact_type == "baseline_wander":
        amp = np.random.uniform(0.3, 1.5) * sig_std
        freq = np.random.uniform(0.05, 0.5)
        phase = np.random.uniform(0, 2.0 * np.pi)
        window += (amp * np.sin(2.0 * np.pi * freq * t + phase)).astype(
            np.float32
        )

    elif artifact_type == "electrode_pop":
        idx = np.random.randint(0, n)
        amp = np.random.uniform(3.0, 8.0) * sig_std
        if np.random.random() < 0.5:
            amp = -amp
        window[idx] += amp

    elif artifact_type == "emg_burst":
        burst_len = np.random.randint(30, 81)
        offset = np.random.randint(0, max(1, n - burst_len))
        amp = np.random.uniform(0.5, 2.0) * sig_std
        noise = (amp * np.random.randn(burst_len)).astype(np.float32)
        end = min(offset + burst_len, n)
        window[offset:end] += noise[: end - offset]

    elif artifact_type == "lead_off_transient":
        block_len = np.random.randint(10, 41)
        ramp_len = np.random.randint(10, 21)
        max_start = max(0, n - block_len - ramp_len)
        start = np.random.randint(0, max(1, max_start + 1))

        block_end = min(start + block_len, n)
        ramp_end = min(block_end + ramp_len, n)

        ramp_target = float(window[min(ramp_end, n - 1)])

        scale = np.random.uniform(0.0, 0.05)
        window[start:block_end] *= scale

        actual_ramp = ramp_end - block_end
        if actual_ramp > 0:
            ramp_start_val = float(window[max(block_end - 1, 0)])
            ramp = np.linspace(ramp_start_val, ramp_target, actual_ramp)
            window[block_end:ramp_end] = ramp.astype(np.float32)

    elif artifact_type == "motion_artifact":
        amp = np.random.uniform(1.0, 3.0) * sig_std
        freq = np.random.uniform(1.0, 5.0)
        phase = np.random.uniform(0, 2.0 * np.pi)
        window += (amp * np.sin(2.0 * np.pi * freq * t + phase)).astype(
            np.float32
        )

    elif artifact_type == "gaussian_noise":
        amp = np.random.uniform(0.2, 0.6) * sig_std
        window += (np.random.randn(n) * amp).astype(np.float32)

    else:
        raise ValueError(f"Unknown artifact type: {artifact_type!r}")

    return window


def augment_clean_beat_preserve_label(window: np.ndarray) -> np.ndarray:
    window = window.copy().astype(np.float32)
    sig_std = float(window.std())
    n = len(window)

    aug = random.choice(["gaussian", "amplitude", "timeshift"])

    if aug == "gaussian":
        noise = (np.random.randn(n) * 0.05 * sig_std).astype(np.float32)
        window += noise
    elif aug == "amplitude":
        scale = np.random.uniform(0.9, 1.1)
        window *= scale
    elif aug == "timeshift":
        shift = np.random.randint(-2, 3)  # −2, −1, 0, +1, +2
        window = np.roll(window, shift)

    return window

# ═══════════════════════════
# SIGNAL PROCESSING UTILITIES
# ═══════════════════════════


def _extract_windows(
    peak_timestamps_ms: np.ndarray,
    peak_segment_ids: np.ndarray,
    ecg_samples_df: pd.DataFrame,
) -> np.ndarray:
    
    n_beats = len(peak_timestamps_ms)
    windows = np.zeros((n_beats, WINDOW_SIZE_SAMPLES), dtype=np.float32)

    ecg_by_seg: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if len(ecg_samples_df) > 0:
        for seg_idx, group in ecg_samples_df.groupby("segment_idx"):
            gs = group.sort_values("timestamp_ms")
            ecg_by_seg[int(seg_idx)] = (
                gs["timestamp_ms"].values.astype(np.int64),
                gs["ecg"].values.astype(np.float32),
            )

    sample_interval_ms = int(1000 / SAMPLE_RATE_HZ)  # ~7.7 ms per sample at 130 Hz
    half_window = WINDOW_SIZE_SAMPLES // 2

    # Window extraction: vectorised per-segment (not per-beat)
    # Outer loop is over unique segments (~few thousand per chunk), not over
    # individual beats (~millions).  All beats in a segment are interpolated
    # in a single np.interp call on the flattened (k×WINDOW_SIZE_SAMPLES,) target array.
    offsets = np.arange(WINDOW_SIZE_SAMPLES, dtype=np.int64) * sample_interval_ms  # (WINDOW_SIZE_SAMPLES,)
    n_with_data = 0

    for seg_idx, (seg_ts, seg_ecg) in ecg_by_seg.items():
        if len(seg_ts) < 2:
            continue
        beat_mask = peak_segment_ids == seg_idx
        if not beat_mask.any():
            continue

        peak_ts_seg = peak_timestamps_ms[beat_mask]          # (k,)
        t_starts = peak_ts_seg - half_window * sample_interval_ms  # (k,)
        target_ts = (t_starts[:, None] + offsets[None, :]).ravel()  # (k×WINDOW_SIZE_SAMPLES,)

        flat = np.interp(
            target_ts.astype(np.float64),
            seg_ts.astype(np.float64),
            seg_ecg,
            left=0.0,
            right=0.0,
        )
        windows[beat_mask] = flat.reshape(-1, WINDOW_SIZE_SAMPLES).astype(np.float32)
        n_with_data += int(beat_mask.sum())

    logger.info(
        "Window extraction: %d / %d beats have ECG data", n_with_data, n_beats
    )

    sos = butter(4, [3.0, 40.0], btype="bandpass", fs=SAMPLE_RATE_HZ, output="sos")
    _FILTER_BATCH = 10_000
    nonzero_idx = np.where(np.any(windows != 0, axis=1))[0]

    for start in range(0, len(nonzero_idx), _FILTER_BATCH):
        idx = nonzero_idx[start : start + _FILTER_BATCH]
        try:
            filtered = sosfiltfilt(sos, windows[idx], axis=1).astype(np.float32)
            valid = ~np.any(np.isnan(filtered), axis=1)
            windows[idx[valid]] = filtered[valid]
        except ValueError:
            pass

    logger.info(
        "Bandpass filtered: %d / %d windows",
        len(nonzero_idx),
        n_beats,
    )
    return windows


# ═══════════════════════════
# DATASET
# ═══════════════════════════


class BeatDataset(Dataset):
    """Pre-extracts and bandpass-filters all ECG windows (WINDOW_SIZE_SAMPLES samples) during
    ``__init__`` for efficiency."""

    def __init__(
        self,
        peak_ids: np.ndarray,
        peak_timestamps_ms: np.ndarray,
        peak_segment_ids: np.ndarray,
        tabular_features: np.ndarray,
        ecg_samples_df: pd.DataFrame,
        labels: np.ndarray | None = None,
        training: bool = False,
        augment_fraction: float = 0.10,
    ) -> None:
        self.peak_ids = peak_ids.astype(np.int64)
        self.n_beats = len(peak_ids)
        self.labels = (
            labels.astype(np.float32) if labels is not None
            else np.zeros(self.n_beats, dtype=np.float32)
        )
        self.tabular = np.nan_to_num(
            tabular_features, nan=0.0
        ).astype(np.float32)
        self.training = training
        self.augment_fraction = augment_fraction

        self.windows = _extract_windows(
            peak_timestamps_ms, peak_segment_ids, ecg_samples_df
        )

        logger.info(
            "BeatDataset: %d beats, training=%s, augment_fraction=%.2f",
            self.n_beats, self.training, self.augment_fraction,
        )

    def __len__(self) -> int:
        return self.n_beats

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.windows[idx].copy()
        label = float(self.labels[idx])
        tabular = self.tabular[idx].copy()

        if self.training and label == 0.0:
            if random.random() < self.augment_fraction:
                window = augment_clean_beat_to_artifact(window)
                label = 1.0
            elif random.random() < 0.20:
                window = augment_clean_beat_preserve_label(window)

        mu = float(window.mean())
        sigma = float(window.std()) + 1e-8
        window = (window - mu) / sigma

        window_t = torch.from_numpy(window).float().unsqueeze(0)  # [1, WINDOW_SIZE_SAMPLES]
        tabular_t = torch.from_numpy(tabular).float()
        label_t = torch.tensor(label, dtype=torch.float32)

        assert not torch.isnan(window_t).any(), (
            f"NaN in window tensor for peak_id={self.peak_ids[idx]}, idx={idx}"
        )
        assert not torch.isnan(tabular_t).any(), (
            f"NaN in tabular tensor for peak_id={self.peak_ids[idx]}, idx={idx}"
        )

        return window_t, tabular_t, label_t


# ═══════════════════════════
# LIGHTNING MODULE
# ═══════════════════════════


class BeatArtifactCNN(pl.LightningModule):

    def __init__(
        self,
        n_tabular_features: int,
        pos_weight: float = 1.0,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # CNN branch: [B, 1, WINDOW_SIZE_SAMPLES] → [B, 128]
        self.cnn = nn.Sequential(
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

        # Tabular branch: [B, N_tab] → [B, 32]
        self.tab = nn.Sequential(
            nn.Linear(n_tabular_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Fusion head: [B, 160] → [B, 1]
        self.head = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(64, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        self.val_preds: list[torch.Tensor] = []
        self.val_labels: list[torch.Tensor] = []

    def forward(
        self, window: torch.Tensor, tabular: torch.Tensor
    ) -> torch.Tensor:

        cnn_out = self.cnn(window)  # [B, 128]
        tab_out = self.tab(tabular)  # [B, 32]
        fused = torch.cat([cnn_out, tab_out], dim=1)  # [B, 160]
        logit = self.head(fused)  # [B, 1]
        return logit

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        window, tabular, label = batch
        logit = self(window, tabular).squeeze(-1)
        loss = self.loss_fn(logit, label)
        self.log(
            "train_loss", loss, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> None:
        window, tabular, label = batch
        logit = self(window, tabular).squeeze(-1)
        loss = self.loss_fn(logit, label)
        self.log(
            "val_loss", loss, prog_bar=True, on_epoch=True, on_step=False
        )
        proba = torch.sigmoid(logit)
        self.val_preds.append(proba.detach().cpu())
        self.val_labels.append(label.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self.val_preds:
            self.log("val_pr_auc", 0.0, prog_bar=True)
            self.log("val_f1_artifact", 0.0, prog_bar=True)
            return

        all_preds = torch.cat(self.val_preds).numpy()
        all_labels = torch.cat(self.val_labels).numpy()
        self.val_preds.clear()
        self.val_labels.clear()

        has_both = all_labels.sum() > 0 and (all_labels == 0).sum() > 0
        if has_both:
            pr_auc = float(average_precision_score(all_labels, all_preds))
            pred_binary = (all_preds >= 0.5).astype(int)
            f1 = float(f1_score(all_labels, pred_binary, zero_division=0.0))
        else:
            pr_auc = 0.0
            f1 = 0.0

        self.log("val_pr_auc", pr_auc, prog_bar=True)
        self.log("val_f1_artifact", f1, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CNN_MAX_EPOCHS
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ═══════════════════════════
# EVALUATION HELPER
# ═══════════════════════════


def _evaluate_model(
    model: BeatArtifactCNN,
    val_loader: DataLoader,
) -> dict[str, Any]:

    device = _get_device()
    model = model.to(device)
    model.eval()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            window, tabular, label = batch
            logit = model(window.to(device), tabular.to(device)).squeeze(-1)
            proba = torch.sigmoid(logit)
            all_preds.append(proba.cpu().numpy())
            all_labels.append(label.numpy())

    preds = np.concatenate(all_preds) if all_preds else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])

    metrics: dict[str, Any] = {
        "n_val": int(len(labels)),
        "n_artifact": int(labels.sum()) if len(labels) > 0 else 0,
        "mean_p_artifact": float(preds.mean()) if len(preds) > 0 else 0.0,
    }

    has_both = len(labels) > 0 and labels.sum() > 0 and (labels == 0).sum() > 0
    if has_both:
        metrics["pr_auc"] = float(average_precision_score(labels, preds))
        pred_binary = (preds >= 0.5).astype(int)
        metrics["f1_artifact"] = float(
            f1_score(labels, pred_binary, zero_division=0.0)
        )
    else:
        metrics["pr_auc"] = float("nan")
        metrics["f1_artifact"] = float("nan")

    return metrics


# ═══════════════════════════
# TRAINING
# ═══════════════════════════


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
    random_seed: int = 42,
    num_workers: int = 4,
) -> dict:
    
    pl.seed_everything(random_seed)

    bf_path = Path(beat_features_path)
    lb_path = Path(labels_path)
    ecg_path = Path(ecg_samples_path)
    sq_path = Path(segment_quality_preds_path)

    for fpath, name in [
        (bf_path, "Beat features"),
        (lb_path, "Labels"),
        (ecg_path, "ECG samples"),
        (sq_path, "Segment quality preds"),
    ]:
        if not fpath.exists():
            raise FileNotFoundError(f"{name} not found: {fpath}")

    peaks_path = bf_path.parent / "peaks.parquet"
    if not peaks_path.exists():
        raise FileNotFoundError(
            f"peaks.parquet not found at {peaks_path} — needed for "
            "timestamp and segment_idx mapping."
        )

    features_df = pd.read_parquet(bf_path)
    labels_df = pd.read_parquet(lb_path)
    ecg_samples_df = pd.read_parquet(ecg_path)
    seg_preds_df = pd.read_parquet(sq_path)
    peaks_df = pd.read_parquet(peaks_path)

    logger.info(
        "Loaded: %d features, %d labels, %d ECG samples, "
        "%d segment preds, %d peaks",
        len(features_df),
        len(labels_df),
        len(ecg_samples_df),
        len(seg_preds_df),
        len(peaks_df),
    )

    if features_df.index.name == "peak_id":
        features_df = features_df.reset_index()

    tabular_cols = [c for c in TABULAR_COLUMNS if c in features_df.columns]
    if len(tabular_cols) < len(TABULAR_COLUMNS):
        missing = set(TABULAR_COLUMNS) - set(tabular_cols)
        logger.warning("Missing tabular columns (will be zero): %s", missing)
    logger.info("Tabular columns (%d): %s", len(tabular_cols), tabular_cols)

    label_cols_cnn = ["peak_id", "label"]
    for _col in ("reviewed", "in_bad_region"):
        if _col in labels_df.columns:
            label_cols_cnn.append(_col)
    merged = features_df.merge(
        labels_df[label_cols_cnn],
        on="peak_id",
        how="inner",
    )
    if "in_bad_region" not in merged.columns:
        merged["in_bad_region"] = False
    merged = merged.merge(
        peaks_df[["peak_id", "segment_idx", "timestamp_ms"]].drop_duplicates(),
        on="peak_id",
        how="inner",
    )
    merged = merged.merge(
        seg_preds_df[["segment_idx", "quality_label"]].drop_duplicates(
            subset="segment_idx"
        ),
        on="segment_idx",
        how="left",
    )
    merged["quality_label"] = merged["quality_label"].fillna("clean")

    logger.info("Merged dataset: %d beats", len(merged))

    if len(merged) == 0:
        logger.error("No beats matched across input files.")
        sys.exit(1)

    merged["target"] = (merged["label"] == "artifact").astype(int)

    n_artifact_total = int(merged["target"].sum())
    n_clean_total = len(merged) - n_artifact_total
    logger.info(
        "Labels: %d clean (%.1f%%), %d artifact (%.1f%%)",
        n_clean_total,
        100.0 * n_clean_total / max(len(merged), 1),
        n_artifact_total,
        100.0 * n_artifact_total / max(len(merged), 1),
    )

    bad_mask = merged["quality_label"] == "bad"
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        logger.info(
            "Excluding %d beats from %d 'bad' segments",
            n_bad,
            int(merged.loc[bad_mask, "segment_idx"].nunique()),
        )
        merged = merged[~bad_mask].copy()

    if "in_bad_region" in merged.columns:
        n_before = len(merged)
        merged = merged[~merged["in_bad_region"]].copy()
        n_excl = n_before - len(merged)
        if n_excl > 0:
            logger.info("Excluded %d beats inside bad_regions", n_excl)

    n_interp = int((merged["label"] == "interpolated").sum())
    if n_interp > 0:
        merged = merged[merged["label"] != "interpolated"].copy()
        logger.info("Excluded %d interpolated beats from training", n_interp)

    if "reviewed" in merged.columns:
        n_before = len(merged)
        merged = merged[merged["reviewed"].astype(bool)].copy()
        n_excl = n_before - len(merged)
        if n_excl > 0:
            logger.info(
                "Restricted to reviewed beats: %d excluded (unreviewed), "
                "%d remain",
                n_excl,
                len(merged),
            )
    else:
        logger.warning(
            "No 'reviewed' column in labels — all beats used for training. "
            "Re-run data_pipeline.py to add reviewed labels."
        )

    unique_segments = sorted(merged["segment_idx"].unique())
    n_segments = len(unique_segments)
    n_train_seg = max(1, int(n_segments * (1 - val_fraction)))

    train_segments = set(unique_segments[:n_train_seg])
    val_segments = set(unique_segments[n_train_seg:])

    train_df = merged[merged["segment_idx"].isin(train_segments)].copy()
    val_df = merged[merged["segment_idx"].isin(val_segments)].copy()

    if len(train_df) > 0 and len(val_df) > 0:
        assert max(train_segments) < min(val_segments), (
            f"Temporal split violated: max train seg ({max(train_segments)}) "
            f"must be < min val seg ({min(val_segments)})"
        )

    logger.info(
        "Temporal split: %d train segments (%d beats), "
        "%d val segments (%d beats)",
        len(train_segments),
        len(train_df),
        len(val_segments),
        len(val_df),
    )

    n_train_pos = int(train_df["target"].sum())
    n_train_neg = len(train_df) - n_train_pos
    pos_weight = float(n_train_neg) / max(n_train_pos, 1)

    if n_train_pos == 0:
        logger.warning(
            "No artifact examples in training set. Model will learn "
            "only from augmented clean beats."
        )

    logger.info(
        "Train: %d clean, %d artifact → pos_weight=%.2f",
        n_train_neg,
        n_train_pos,
        pos_weight,
    )

    n_tabular = len(tabular_cols)

    train_dataset = BeatDataset(
        peak_ids=train_df["peak_id"].values,
        peak_timestamps_ms=train_df["timestamp_ms"].values,
        peak_segment_ids=train_df["segment_idx"].values,
        tabular_features=train_df[tabular_cols].values,
        ecg_samples_df=ecg_samples_df,
        labels=train_df["target"].values.astype(np.float32),
        training=True,
        augment_fraction=augment_fraction,
    )

    val_dataset = (
        BeatDataset(
            peak_ids=val_df["peak_id"].values,
            peak_timestamps_ms=val_df["timestamp_ms"].values,
            peak_segment_ids=val_df["segment_idx"].values,
            tabular_features=val_df[tabular_cols].values,
            ecg_samples_df=ecg_samples_df,
            labels=val_df["target"].values.astype(np.float32),
            training=False,
        )
        if len(val_df) > 0
        else None
    )

    artifact_mask = torch.from_numpy(train_dataset.labels == 1.0)
    sample_weights = torch.ones(len(train_dataset))
    sample_weights[artifact_mask] = 15.0

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=min(batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
        if val_dataset is not None
        else None
    )

    model = BeatArtifactCNN(
        n_tabular_features=n_tabular,
        pos_weight=pos_weight,
        learning_rate=CNN_LEARNING_RATE,
    )
    logger.info(
        "Model: %d parameters",
        sum(p.numel() for p in model.parameters()),
    )

    callbacks: list[pl.Callback] = []

    ckpt_dir = Path(output_model_path).parent / "cnn_checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        logger.info("Removed stale checkpoint directory: %s", ckpt_dir)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_pr_auc" if val_loader else None,
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_pr_auc:.4f}",
    )
    callbacks.append(checkpoint_cb)

    if val_loader is not None:
        early_stop_cb = EarlyStopping(
            monitor="val_pr_auc",
            patience=15,
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stop_cb)

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=False,
        enable_model_summary=False,
    )

    logger.info(
        "Starting training: max_epochs=%d, batch_size=%d, augment=%.0f%%",
        max_epochs,
        batch_size,
        augment_fraction * 100,
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_cb.best_model_path
    if best_path:
        logger.info("Loading best checkpoint: %s", best_path)
        best_ckpt = torch.load(best_path, weights_only=False, map_location=_get_device())
        model.load_state_dict(best_ckpt["state_dict"])
    else:
        logger.info("No checkpoint saved — using last model state")

    final_metrics: dict[str, Any] = {}
    if val_loader is not None:
        final_metrics = _evaluate_model(model, val_loader)
        logger.info("Final val metrics: %s", final_metrics)
    else:
        logger.warning("No validation set — skipping final evaluation")

    trained_at = datetime.now(timezone.utc).isoformat()

    artifact: dict[str, Any] = {
        "state_dict": model.cpu().state_dict(),
        "n_tabular_features": n_tabular,
        "tabular_columns": tabular_cols,
        "pos_weight": pos_weight,
        "val_metrics": final_metrics,
        "trained_at": trained_at,
    }

    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    logger.info("Saved model artifact → %s", out_path)

    _print_training_summary(
        n_total=len(merged),
        n_train=len(train_df),
        n_val=len(val_df),
        n_train_pos=n_train_pos,
        n_train_neg=n_train_neg,
        pos_weight=pos_weight,
        final_metrics=final_metrics,
        trained_at=trained_at,
        best_epoch=checkpoint_cb.best_model_score,
    )

    return final_metrics


def _print_training_summary(
    n_total: int,
    n_train: int,
    n_val: int,
    n_train_pos: int,
    n_train_neg: int,
    pos_weight: float,
    final_metrics: dict[str, Any],
    trained_at: str,
    best_epoch: Any,
) -> None:
    """Print a formatted training summary to stdout."""
    print(f"\n{'=' * 72}")
    print("  Stage 2a — Hybrid CNN Beat Artifact Classifier: Training Summary")
    print(f"{'=' * 72}")
    print(f"  Trained at:  {trained_at}")
    print(f"  Total beats: {n_total:,}")
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}  (temporal split)")

    print(f"\n  Training class distribution:")
    print(f"    clean/negative:  {n_train_neg:>8,}")
    print(f"    artifact/pos:    {n_train_pos:>8,}")
    print(f"  pos_weight:        {pos_weight:.2f}")

    pr_auc = final_metrics.get("pr_auc", float("nan"))
    f1 = final_metrics.get("f1_artifact", float("nan"))
    print()
    print(f"  ╔═══════════════════════════════════════════╗")
    print(f"  ║  PR-AUC (artifact class):  {pr_auc:>8.4f}       ║  ← PRIMARY METRIC")
    print(f"  ║  F1 (artifact, t=0.5):     {f1:>8.4f}       ║")
    print(f"  ╚═══════════════════════════════════════════╝")

    n_val_art = final_metrics.get("n_artifact", 0)
    mean_p = final_metrics.get("mean_p_artifact", 0.0)
    print(f"\n  Validation: {n_val:,} beats, {n_val_art} artifact")
    print(f"  Mean p_artifact: {mean_p:.6f}")

    if best_epoch is not None:
        print(f"  Best checkpoint val_pr_auc: {float(best_epoch):.4f}")

    print(f"{'=' * 72}\n")


# ═══════════════════════════
# INFERENCE
# ═══════════════════════════


def predict(
    beat_features: pd.DataFrame,
    ecg_samples_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    model_path: str,
    batch_size: int = 1024,
) -> pd.DataFrame:
    
    mpath = Path(model_path)
    if not mpath.exists():
        raise FileNotFoundError(f"Model not found: {mpath}")

    device = _get_device()
    artifact = torch.load(mpath, weights_only=False, map_location=device)

    n_tabular = artifact["n_tabular_features"]
    tabular_cols: list[str] = artifact["tabular_columns"]
    trained_at: str = artifact.get("trained_at", "unknown")

    logger.info(
        "Loaded CNN model (trained %s, %d tabular features, device=%s)",
        trained_at,
        n_tabular,
        device,
    )

    model = BeatArtifactCNN(
        n_tabular_features=n_tabular,
        pos_weight=artifact.get("pos_weight", 1.0),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    model.to(device)

    df = beat_features.copy()
    if df.index.name == "peak_id":
        df = df.reset_index()

    df = df.merge(
        peaks_df[["peak_id", "timestamp_ms", "segment_idx"]].drop_duplicates(),
        on="peak_id",
        how="inner",
    )

    for col in tabular_cols:
        if col not in df.columns:
            df[col] = 0.0

    dataset = BeatDataset(
        peak_ids=df["peak_id"].values,
        peak_timestamps_ms=df["timestamp_ms"].values,
        peak_segment_ids=df["segment_idx"].values,
        tabular_features=df[tabular_cols].values,
        ecg_samples_df=ecg_samples_df,
        labels=None,
        training=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, max(len(dataset), 1)),
        shuffle=False,
        num_workers=0,
    )

    all_probas: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            window, tabular, _ = batch
            logit = model(window.to(device), tabular.to(device)).squeeze(-1)
            proba = torch.sigmoid(logit).cpu().numpy()
            all_probas.append(proba)

    probas = (
        np.concatenate(all_probas).astype(np.float32)
        if all_probas
        else np.array([], dtype=np.float32)
    )

    result = pd.DataFrame(
        {
            "peak_id": dataset.peak_ids,
            "p_artifact_cnn": probas,
        }
    )

    n_pred_artifact = int((probas >= 0.5).sum()) if len(probas) > 0 else 0
    logger.info(
        "Predictions: %d beats — %d predicted artifact (%.2f%%)",
        len(result),
        n_pred_artifact,
        100.0 * n_pred_artifact / max(len(result), 1),
    )

    return result


# ═══════════════════════════
# CLI
# ═══════════════════════════


def _cli_train(args: argparse.Namespace) -> None:
    """Handle the ``train`` subcommand."""
    train(
        beat_features_path=args.beat_features,
        labels_path=args.labels,
        ecg_samples_path=args.ecg_samples,
        segment_quality_preds_path=args.segment_quality_preds,
        output_model_path=args.output,
        val_fraction=args.val_fraction,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        augment_fraction=args.augment_fraction,
        random_seed=args.seed,
        num_workers=args.num_workers,
    )


def _cli_predict(args: argparse.Namespace) -> None:
    """Handle the ``predict`` subcommand (streaming, low-memory)."""
    bf_path = Path(args.beat_features)
    ecg_path = Path(args.ecg_samples)

    if not bf_path.exists():
        logger.error("Beat features not found: %s", bf_path)
        sys.exit(1)
    if not ecg_path.exists():
        logger.error("ECG samples not found: %s", ecg_path)
        sys.exit(1)

    peaks_path = bf_path.parent / "peaks.parquet"
    if not peaks_path.exists():
        logger.error("peaks.parquet not found at %s", peaks_path)
        sys.exit(1)

    mpath = Path(args.model)
    if not mpath.exists():
        logger.error("Model not found: %s", mpath)
        sys.exit(1)

    device = _get_device()
    artifact = torch.load(mpath, weights_only=False, map_location=device)
    n_tabular = artifact["n_tabular_features"]
    tabular_cols: list[str] = artifact["tabular_columns"]
    trained_at: str = artifact.get("trained_at", "unknown")

    model = BeatArtifactCNN(
        n_tabular_features=n_tabular,
        pos_weight=artifact.get("pos_weight", 1.0),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    model.to(device)
    logger.info(
        "Loaded CNN model (trained %s, %d tabular features, device=%s)",
        trained_at, n_tabular, device,
    )

    # Load only the columns the saved model was trained on (up to 41 tabular cols).
    # Full beat_features parquet is ~5 GB — only read needed columns.
    # ecg_samples.parquet is NOT loaded here — it is streamed per chunk below.
    bf_schema_names = set(pq.read_schema(bf_path).names)
    needed_bf_cols = [
        c for c in (["peak_id"] + tabular_cols)
        if c in bf_schema_names
    ]
    beat_features = pq.read_table(bf_path, columns=needed_bf_cols).to_pandas()

    peaks_df = pq.read_table(
        peaks_path,
        columns=["peak_id", "timestamp_ms", "segment_idx"],
    ).to_pandas()

    logger.info(
        "Loaded %d beat features (%d cols), %d peaks",
        len(beat_features),
        len(beat_features.columns),
        len(peaks_df),
    )

    df = beat_features.copy()
    if df.index.name == "peak_id":
        df = df.reset_index()
    df = df.merge(
        peaks_df[["peak_id", "timestamp_ms", "segment_idx"]].drop_duplicates(),
        on="peak_id",
        how="inner",
    )
    for col in tabular_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df.sort_values("segment_idx").reset_index(drop=True)
    seg_arr = df["segment_idx"].values
    all_segs = sorted(df["segment_idx"].unique())
    total_segs = len(all_segs)
    chunk_size: int = args.chunk_segments
    batch_size: int = args.batch_size
    n_chunks = (total_segs + chunk_size - 1) // chunk_size

    logger.info(
        "Streaming CNN inference: %d segments, %d chunks (chunk=%d, batch=%d)",
        total_segs, n_chunks, chunk_size, batch_size,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_schema = pa.schema([
        pa.field("peak_id", pa.int64()),
        pa.field("p_artifact_cnn", pa.float32()),
    ])

    total_beats = 0
    total_predicted_artifact = 0

    with pq.ParquetWriter(out_path, out_schema, compression="snappy") as writer:
        for chunk_num, chunk_start in enumerate(range(0, total_segs, chunk_size), 1):
            chunk_segs = all_segs[chunk_start : chunk_start + chunk_size]
            seg_min, seg_max = int(chunk_segs[0]), int(chunk_segs[-1])

            lo = int(np.searchsorted(seg_arr, seg_min, side="left"))
            hi = int(np.searchsorted(seg_arr, seg_max + 1, side="left"))
            chunk_df = df.iloc[lo:hi]

            if len(chunk_df) == 0:
                continue

            logger.info(
                "CNN chunk %d/%d: segs %d–%d (%d beats)",
                chunk_num, n_chunks, seg_min, seg_max, len(chunk_df),
            )

            ecg_chunk = pq.read_table(
                ecg_path,
                filters=[
                    ("segment_idx", ">=", seg_min),
                    ("segment_idx", "<=", seg_max),
                ],
                columns=["timestamp_ms", "ecg", "segment_idx"],
            ).to_pandas().sort_values("timestamp_ms").reset_index(drop=True)

            # Extract and bandpass-filter ECG windows
            windows = _extract_windows(
                chunk_df["timestamp_ms"].values.astype(np.int64),
                chunk_df["segment_idx"].values.astype(np.int64),
                ecg_chunk,
            )
            del ecg_chunk

            tabular = np.nan_to_num(
                chunk_df[tabular_cols].values.astype(np.float32), nan=0.0
            )

            chunk_probas: list[float] = []
            for b in range(0, len(chunk_df), batch_size):
                win_b = windows[b : b + batch_size].copy()  # (B, WINDOW_SIZE_SAMPLES)
                tab_b = tabular[b : b + batch_size]

                mu = win_b.mean(axis=-1, keepdims=True)
                sigma = win_b.std(axis=-1, keepdims=True) + 1e-8
                win_b = (win_b - mu) / sigma

                win_t = torch.from_numpy(win_b).float().unsqueeze(1).to(device)  # [B, 1, WINDOW_SIZE_SAMPLES]
                tab_t = torch.from_numpy(tab_b).float().to(device)

                with torch.no_grad():
                    logit = model(win_t, tab_t).squeeze(-1)
                    proba = torch.sigmoid(logit).cpu().numpy()
                chunk_probas.extend(proba.tolist())

            del windows, tabular

            probas_arr = np.array(chunk_probas, dtype=np.float32)
            peak_ids = chunk_df["peak_id"].values.astype(np.int64)
            n_art = int((probas_arr >= 0.5).sum())
            total_beats += len(probas_arr)
            total_predicted_artifact += n_art

            logger.info(
                "  → %d beats, %d artifact (%.2f%%)",
                len(probas_arr), n_art, 100.0 * n_art / max(len(probas_arr), 1),
            )

            writer.write_table(
                pa.table(
                    {"peak_id": peak_ids, "p_artifact_cnn": probas_arr},
                    schema=out_schema,
                )
            )

    logger.info("Saved predictions → %s", out_path)

    pct = 100.0 * total_predicted_artifact / max(total_beats, 1)
    print(f"\n{'=' * 72}")
    print("  CNN Beat Artifact Predictions")
    print(f"{'=' * 72}")
    print(f"  Total beats: {total_beats:,}")
    print(f"\n  Prediction distribution:")
    print(f"    clean (p < 0.5):     {total_beats - total_predicted_artifact:>8,}  ({100 - pct:5.1f}%)")
    print(f"    artifact (p >= 0.5): {total_predicted_artifact:>8,}  ({pct:5.1f}%)")
    print(f"\n  Full statistics: see {out_path}")
    print(f"{'=' * 72}\n")


def main() -> None:
    """CLI entry point with ``train`` and ``predict`` subcommands."""
    parser = argparse.ArgumentParser(
        description=(
            "ECG Artifact Pipeline — Stage 2a: Hybrid CNN Beat Artifact Classifier"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    tp = subparsers.add_parser(
        "train", help="Train the hybrid CNN artifact classifier"
    )
    tp.add_argument(
        "--beat-features", type=str, required=True,
        help="Path to beat_features.parquet",
    )
    tp.add_argument(
        "--labels", type=str, required=True,
        help="Path to labels.parquet",
    )
    tp.add_argument(
        "--ecg-samples", type=str, required=True,
        help="Path to ecg_samples.parquet",
    )
    tp.add_argument(
        "--segment-quality-preds", type=str, required=True,
        help="Path to segment_quality_preds.parquet",
    )
    tp.add_argument(
        "--output", type=str, required=True,
        help="Output path for model artifact (.pt)",
    )
    tp.add_argument(
        "--val-fraction", type=float, default=VAL_FRACTION,
        help=f"Fraction of segments for validation (default: {VAL_FRACTION})",
    )
    tp.add_argument(
        "--max-epochs", type=int, default=CNN_MAX_EPOCHS,
        help=f"Maximum training epochs (default: {CNN_MAX_EPOCHS})",
    )
    tp.add_argument(
        "--batch-size", type=int, default=CNN_BATCH_SIZE,
        help=f"Batch size (default: {CNN_BATCH_SIZE})",
    )
    tp.add_argument(
        "--augment-fraction", type=float, default=0.10,
        help="Fraction of clean beats to corrupt per epoch (default: 0.10)",
    )
    tp.add_argument(
        "--seed", type=int, default=LGBM_RANDOM_STATE,
        help=f"Random seed (default: {LGBM_RANDOM_STATE})",
    )
    tp.add_argument(
        "--num-workers", type=int,
        default=min(4, max(1, (os.cpu_count() or 4) - 2)),
        help=(
            "DataLoader worker processes for data prefetching "
            "(default: min(4, cpu_count-2)).  Set 0 to disable."
        ),
    )

    # predict
    pp = subparsers.add_parser(
        "predict", help="Run inference on beat features"
    )
    pp.add_argument(
        "--beat-features", type=str, required=True,
        help="Path to beat_features.parquet",
    )
    pp.add_argument(
        "--ecg-samples", type=str, required=True,
        help="Path to ecg_samples.parquet",
    )
    pp.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model artifact (.pt)",
    )
    pp.add_argument(
        "--output", type=str, required=True,
        help="Output path for predictions (.parquet)",
    )
    pp.add_argument(
        "--batch-size", type=int, default=CNN_BATCH_SIZE,
        help=f"Inference batch size (default: {CNN_BATCH_SIZE})",
    )
    pp.add_argument(
        "--chunk-segments", type=int, default=2000,
        help="Segments to process per chunk (default: 2000)",
    )

    args = parser.parse_args()

    if args.command == "train":
        _cli_train(args)
    elif args.command == "predict":
        _cli_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
