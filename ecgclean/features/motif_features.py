#!/usr/bin/env python3
"""
ecgclean.features.motif_features
================================
Discover recurring ECG waveform patterns (motifs) via k-means clustering
and compute distance-based anomaly features for each beat.

Two motif families:
    1. **QRS motifs** — cluster 64-sample beat windows by morphology
    2. **RR motifs**  — cluster sliding windows of 10 consecutive RR intervals

These features tell downstream models how unusual each beat is relative
to the patient's full ECG history — a powerful signal for anomaly detection.

CLI
---
    python ecgclean/features/motif_features.py discover \\
        --beat-features ... --labels ... --output data/motifs/
    python ecgclean/features/motif_features.py compute \\
        --beat-features ... --labels ... --motifs data/motifs/ --output ...
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
MOTIF_VERSION = "1.0"
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"

# Ensure project root is on sys.path for loading beat_features window helper
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================== #
#  ECG window extraction (reuse pattern from beat_features.py)          #
# ===================================================================== #
def _load_ecg_windows(
    peaks_df: pd.DataFrame,
    ecg_samples_path: str,
    window_size: int = 64,
) -> np.ndarray:
    """Extract ECG windows centered on each peak.

    Uses the same binary-search algorithm as ``beat_features.py``.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Must have ``peak_id`` and ``timestamp_ns``.
    ecg_samples_path : str
        Path to ``ecg_samples.parquet``.
    window_size : int
        Number of samples per window (default 64).

    Returns
    -------
    np.ndarray
        Shape ``(n_peaks, window_size)``, dtype float32.
    """
    ecg_samples = pd.read_parquet(ecg_samples_path)
    n_peaks = len(peaks_df)
    windows = np.zeros((n_peaks, window_size), dtype=np.float32)

    ecg_ts = ecg_samples["timestamp_ns"].values.astype(np.int64)
    ecg_vals = ecg_samples["ecg"].values.astype(np.float32)
    n_ecg = len(ecg_ts)

    if n_ecg == 0:
        log.warning("No ECG samples available for window extraction")
        return windows

    peak_ts = peaks_df["timestamp_ns"].values.astype(np.int64)
    half = window_size // 2

    insert_idx = np.searchsorted(ecg_ts, peak_ts, side="left")

    for i in range(n_peaks):
        center = int(insert_idx[i])
        if center > 0 and center < n_ecg:
            if abs(ecg_ts[center - 1] - peak_ts[i]) < abs(ecg_ts[center] - peak_ts[i]):
                center -= 1
        elif center >= n_ecg:
            center = n_ecg - 1

        start = center - half
        end = start + window_size
        src_start = max(0, start)
        src_end = min(n_ecg, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)

        if src_end > src_start:
            windows[i, dst_start:dst_end] = ecg_vals[src_start:src_end]

    return windows


# ===================================================================== #
#  Sparkline rendering                                                  #
# ===================================================================== #
def _sparkline(values: np.ndarray, width: int = 40) -> str:
    """Render a 1-D signal as a sparkline using Unicode block chars.

    Parameters
    ----------
    values : np.ndarray
        1-D signal to render.
    width : int
        Character width of the sparkline.

    Returns
    -------
    str
        Unicode sparkline string.
    """
    if len(values) == 0:
        return ""
    # Resample to target width
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width).astype(int)
        values = values[indices]
    elif len(values) < width:
        indices = np.linspace(0, len(values) - 1, width)
        values = np.interp(indices, np.arange(len(values)), values)

    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        return SPARKLINE_CHARS[4] * width

    # Scale to [0, len(SPARKLINE_CHARS)-1]
    scaled = (values - vmin) / (vmax - vmin) * (len(SPARKLINE_CHARS) - 1)
    return "".join(SPARKLINE_CHARS[int(np.clip(v, 0, len(SPARKLINE_CHARS) - 1))] for v in scaled)


# ===================================================================== #
#  QRS motif discovery                                                  #
# ===================================================================== #
def discover_qrs_motifs(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ecg_windows: np.ndarray,
    n_clusters: int = 12,
    use_clean_only: bool = True,
    random_seed: int = 42,
) -> dict:
    """Cluster 64-sample QRS windows to discover recurring morphologies.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Must have ``peak_id``.  Row order matches ``ecg_windows``.
    labels_df : pd.DataFrame
        Must have ``peak_id``, ``label``, ``hard_filtered``,
        ``phys_event_window``.
    ecg_windows : np.ndarray
        Shape ``(n_beats, 64)``.
    n_clusters : int
        Number of k-means clusters (default 12).
    use_clean_only : bool
        If True, only cluster beats with ``label == "clean"``
        and ``hard_filtered == False``.
    random_seed : int
        For reproducibility.

    Returns
    -------
    dict
        ``centroids``, ``cluster_labels``, ``cluster_assignments``,
        ``kmeans_model``, ``inertia``, ``peak_ids``.
    """
    # ── Align peaks and labels ───────────────────────────────────────
    merged = peaks_df[["peak_id"]].copy()
    merged["_row"] = np.arange(len(merged))

    label_cols = ["peak_id", "label", "hard_filtered"]
    if "phys_event_window" in labels_df.columns:
        label_cols.append("phys_event_window")
    if "rr_prev_ms" in labels_df.columns:
        label_cols.append("rr_prev_ms")

    merged = merged.merge(labels_df[label_cols], on="peak_id", how="left")

    # ── Filter to clean beats ────────────────────────────────────────
    if use_clean_only:
        mask = (merged["label"] == "clean") & (~merged["hard_filtered"].fillna(False))
        selection = merged[mask].copy()
    else:
        selection = merged.copy()

    if len(selection) == 0:
        log.warning("No clean beats available for QRS motif discovery — using all beats")
        selection = merged.copy()

    row_indices = selection["_row"].values
    selected_windows = ecg_windows[row_indices]
    selected_peak_ids = selection["peak_id"].values

    log.info("QRS motif discovery: %d beats, %d clusters", len(selection), n_clusters)

    # Adjust n_clusters if fewer beats than clusters
    actual_k = min(n_clusters, len(selection))
    if actual_k < n_clusters:
        log.warning("Only %d beats available — reducing to %d clusters", len(selection), actual_k)

    # ── K-means ──────────────────────────────────────────────────────
    km = KMeans(n_clusters=actual_k, random_state=random_seed, n_init=10, max_iter=300)
    assignments = km.fit_predict(selected_windows)
    centroids = km.cluster_centers_  # (k, 64)

    # ── Auto-label clusters ──────────────────────────────────────────
    cluster_labels = []
    for ci in range(actual_k):
        cluster_mask = assignments == ci
        cluster_beats = selection[cluster_mask]
        cluster_windows = selected_windows[cluster_mask]

        n_beats = int(cluster_mask.sum())
        window_ptp = np.ptp(cluster_windows, axis=1)
        mean_ptp = float(np.mean(window_ptp)) if n_beats > 0 else 0.0
        std_ptp = float(np.std(window_ptp)) if n_beats > 0 else 0.0

        # Compute mean HR from rr_prev_ms
        if "rr_prev_ms" in cluster_beats.columns:
            rr_vals = cluster_beats["rr_prev_ms"].dropna().values
            mean_hr = 60000.0 / np.mean(rr_vals) if len(rr_vals) > 0 else 80.0
        else:
            mean_hr = 80.0

        # Check phys_event fraction
        if "phys_event_window" in cluster_beats.columns:
            phys_frac = float(cluster_beats["phys_event_window"].mean())
        else:
            phys_frac = 0.0

        # ── Heuristic labeling ───────────────────────────────────────
        if phys_frac > 0.3:
            label = "pots_transition"
        elif std_ptp > 0.3 * mean_ptp and mean_ptp > 0:
            label = "noisy_cluster"
        elif mean_hr < 75:
            label = "normal_sinus_brady"
        elif mean_hr > 100:
            label = "sinus_tachycardia"
        else:
            label = "normal_sinus"

        cluster_labels.append(label)

    log.info("QRS clusters: %s", dict(zip(range(actual_k), cluster_labels)))

    return {
        "centroids": centroids.astype(np.float32),
        "cluster_labels": cluster_labels,
        "cluster_assignments": assignments,
        "kmeans_model": km,
        "inertia": float(km.inertia_),
        "peak_ids": selected_peak_ids,
    }


# ===================================================================== #
#  RR motif discovery                                                   #
# ===================================================================== #
def discover_rr_motifs(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    window_size: int = 10,
    n_clusters: int = 8,
    random_seed: int = 42,
) -> dict:
    """Cluster sliding windows of consecutive RR intervals.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Must have ``peak_id``.
    labels_df : pd.DataFrame
        Must have ``peak_id`` and ``rr_prev_ms``.
    window_size : int
        Number of consecutive RR intervals per window (default 10).
    n_clusters : int
        Number of k-means clusters (default 8).
    random_seed : int
        For reproducibility.

    Returns
    -------
    dict
        Same structure as ``discover_qrs_motifs``, with
        ``centroids`` shape ``(n_clusters, window_size)``.
    """
    # ── Build RR series ──────────────────────────────────────────────
    merged = peaks_df[["peak_id"]].merge(
        labels_df[["peak_id", "rr_prev_ms"]], on="peak_id", how="left",
    )
    rr = merged["rr_prev_ms"].values.astype(np.float64)
    peak_ids_all = merged["peak_id"].values

    # Fill NaN with median for sliding window computation
    rr_median = np.nanmedian(rr)
    if np.isnan(rr_median):
        rr_median = 800.0  # fallback
    rr_filled = np.where(np.isnan(rr), rr_median, rr)

    # ── Sliding windows ──────────────────────────────────────────────
    n = len(rr_filled)
    if n < window_size:
        log.warning("Only %d beats — cannot form RR windows of size %d", n, window_size)
        # Return degenerate result
        km = KMeans(n_clusters=1, random_state=random_seed, n_init=1)
        single_window = rr_filled[:n]
        padded = np.zeros(window_size, dtype=np.float64)
        padded[:len(single_window)] = single_window
        km.fit(padded.reshape(1, -1))
        return {
            "centroids": km.cluster_centers_.astype(np.float32),
            "cluster_labels": ["insufficient_data"],
            "cluster_assignments": np.array([0]),
            "kmeans_model": km,
            "inertia": 0.0,
            "peak_ids": peak_ids_all[:1],
        }

    windows = []
    window_peak_ids = []

    for i in range(n - window_size + 1):
        w = rr_filled[i : i + window_size]
        windows.append(w)
        # Associate with the center beat of the window
        center_idx = i + window_size // 2
        window_peak_ids.append(peak_ids_all[center_idx])

    rr_windows = np.array(windows, dtype=np.float64)  # (n_windows, window_size)
    window_peak_ids = np.array(window_peak_ids)

    log.info("RR motif discovery: %d windows of size %d, %d clusters", len(rr_windows), window_size, n_clusters)

    actual_k = min(n_clusters, len(rr_windows))
    if actual_k < n_clusters:
        log.warning("Only %d RR windows — reducing to %d clusters", len(rr_windows), actual_k)

    # ── K-means ──────────────────────────────────────────────────────
    km = KMeans(n_clusters=actual_k, random_state=random_seed, n_init=10, max_iter=300)
    assignments = km.fit_predict(rr_windows)
    centroids = km.cluster_centers_  # (k, window_size)

    # ── Auto-label clusters ──────────────────────────────────────────
    cluster_labels = []
    for ci in range(actual_k):
        cluster_mask = assignments == ci
        cluster_windows = rr_windows[cluster_mask]
        n_wins = int(cluster_mask.sum())

        if n_wins == 0:
            cluster_labels.append("empty")
            continue

        mean_rr = float(np.mean(cluster_windows))
        std_rr = float(np.std(cluster_windows))
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 80.0

        # Check for ramp patterns (monotonically increasing/decreasing)
        diffs = np.diff(centroids[ci])
        frac_increasing = float(np.mean(diffs > 0))
        frac_decreasing = float(np.mean(diffs < 0))

        # ── Heuristic labeling ───────────────────────────────────────
        cv = std_rr / mean_rr if mean_rr > 0 else 0.0

        if cv > 0.25:
            label = "erratic_noisy"
        elif frac_decreasing > 0.7 and mean_hr > 80:
            label = "pots_ramp_up"  # RR decreasing → HR increasing
        elif frac_increasing > 0.7 and mean_hr < 90:
            label = "pots_recovery"  # RR increasing → HR decreasing
        elif mean_hr < 65:
            label = "bradycardic_rest"
        elif mean_hr > 100:
            label = "tachycardic"
        else:
            label = "stable_sinus"

        cluster_labels.append(label)

    log.info("RR clusters: %s", dict(zip(range(actual_k), cluster_labels)))

    return {
        "centroids": centroids.astype(np.float32),
        "cluster_labels": cluster_labels,
        "cluster_assignments": assignments,
        "kmeans_model": km,
        "inertia": float(km.inertia_),
        "peak_ids": window_peak_ids,
    }


# ===================================================================== #
#  Motif feature computation                                            #
# ===================================================================== #
def compute_motif_features(
    peaks_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ecg_windows: np.ndarray,
    qrs_motif_dict: dict,
    rr_motif_dict: dict,
) -> pd.DataFrame:
    """Compute distance-based motif features for every beat.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Must have ``peak_id``.
    labels_df : pd.DataFrame
        Must have ``peak_id``, ``rr_prev_ms``.
    ecg_windows : np.ndarray
        Shape ``(n_beats, 64)``.
    qrs_motif_dict : dict
        Output of ``discover_qrs_motifs()``.
    rr_motif_dict : dict
        Output of ``discover_rr_motifs()``.

    Returns
    -------
    pd.DataFrame
        One row per beat with columns: peak_id,
        dist_to_nearest_qrs_motif, nearest_qrs_motif_label,
        nearest_qrs_motif_idx, dist_to_nearest_rr_motif,
        nearest_rr_motif_label, nearest_rr_motif_idx,
        qrs_anomaly_score, rr_anomaly_score,
        is_qrs_anomaly, is_rr_anomaly.
    """
    n = len(peaks_df)
    peak_ids = peaks_df["peak_id"].values

    # ── QRS distances ────────────────────────────────────────────────
    qrs_centroids = qrs_motif_dict["centroids"]  # (k, 64)
    qrs_labels = qrs_motif_dict["cluster_labels"]

    # Compute distance from each beat window to each centroid
    # Using broadcasting: (n, 1, 64) - (1, k, 64) → (n, k)
    qrs_dists = np.linalg.norm(
        ecg_windows[:, np.newaxis, :] - qrs_centroids[np.newaxis, :, :],
        axis=2,
    )  # (n, k)

    nearest_qrs_idx = qrs_dists.argmin(axis=1)  # (n,)
    nearest_qrs_dist = qrs_dists[np.arange(n), nearest_qrs_idx]  # (n,)
    nearest_qrs_label = [qrs_labels[int(i)] for i in nearest_qrs_idx]

    # ── RR distances ─────────────────────────────────────────────────
    # Build RR windows for each beat (centered, zero-padded at edges)
    rr_window_size = rr_motif_dict["centroids"].shape[1]
    rr_centroids = rr_motif_dict["centroids"]  # (k, window_size)
    rr_labels = rr_motif_dict["cluster_labels"]

    # Get RR series
    merged = peaks_df[["peak_id"]].merge(
        labels_df[["peak_id", "rr_prev_ms"]], on="peak_id", how="left",
    )
    rr = merged["rr_prev_ms"].values.astype(np.float64)
    rr_median = np.nanmedian(rr)
    if np.isnan(rr_median):
        rr_median = 800.0
    rr_filled = np.where(np.isnan(rr), rr_median, rr)

    half_w = rr_window_size // 2

    rr_windows = np.zeros((n, rr_window_size), dtype=np.float64)
    for i in range(n):
        start = i - half_w
        end = start + rr_window_size
        # Clip to valid range and zero-pad
        src_start = max(0, start)
        src_end = min(n, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        rr_windows[i, dst_start:dst_end] = rr_filled[src_start:src_end]

    rr_dists = np.linalg.norm(
        rr_windows[:, np.newaxis, :] - rr_centroids[np.newaxis, :, :],
        axis=2,
    )  # (n, k)

    nearest_rr_idx = rr_dists.argmin(axis=1)
    nearest_rr_dist = rr_dists[np.arange(n), nearest_rr_idx]
    nearest_rr_label = [rr_labels[int(i)] for i in nearest_rr_idx]

    # ── Anomaly scores (normalized by mean clean-beat distance) ──────
    # Use clean-beat mean distance as baseline
    clean_mask_df = peaks_df[["peak_id"]].merge(
        labels_df[["peak_id", "label"]], on="peak_id", how="left",
    )
    clean_mask = (clean_mask_df["label"] == "clean").values

    if clean_mask.any():
        mean_qrs_dist_clean = float(np.mean(nearest_qrs_dist[clean_mask]))
        mean_rr_dist_clean = float(np.mean(nearest_rr_dist[clean_mask]))
    else:
        mean_qrs_dist_clean = float(np.mean(nearest_qrs_dist))
        mean_rr_dist_clean = float(np.mean(nearest_rr_dist))

    # Avoid division by zero
    mean_qrs_dist_clean = max(mean_qrs_dist_clean, 1e-8)
    mean_rr_dist_clean = max(mean_rr_dist_clean, 1e-8)

    qrs_anomaly_score = (nearest_qrs_dist / mean_qrs_dist_clean).astype(np.float32)
    rr_anomaly_score = (nearest_rr_dist / mean_rr_dist_clean).astype(np.float32)

    # ── Build output DataFrame ───────────────────────────────────────
    result = pd.DataFrame({
        "peak_id": peak_ids,
        "dist_to_nearest_qrs_motif": nearest_qrs_dist.astype(np.float32),
        "nearest_qrs_motif_label": nearest_qrs_label,
        "nearest_qrs_motif_idx": nearest_qrs_idx.astype(np.int32),
        "dist_to_nearest_rr_motif": nearest_rr_dist.astype(np.float32),
        "nearest_rr_motif_label": nearest_rr_label,
        "nearest_rr_motif_idx": nearest_rr_idx.astype(np.int32),
        "qrs_anomaly_score": qrs_anomaly_score,
        "rr_anomaly_score": rr_anomaly_score,
        "is_qrs_anomaly": qrs_anomaly_score > 2.0,
        "is_rr_anomaly": rr_anomaly_score > 2.0,
    })

    return result


# ===================================================================== #
#  One-hot encoding helper                                              #
# ===================================================================== #
def get_motif_dummies(motif_features_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the motif label columns.

    Parameters
    ----------
    motif_features_df : pd.DataFrame
        Must contain ``nearest_qrs_motif_label`` and
        ``nearest_rr_motif_label``.

    Returns
    -------
    pd.DataFrame
        One-hot columns ready to concatenate with the beat feature matrix.
        Column names: ``motif_qrs_{label}`` and ``motif_rr_{label}``.
    """
    dummies = pd.DataFrame(index=motif_features_df.index)

    if "nearest_qrs_motif_label" in motif_features_df.columns:
        qrs_dummies = pd.get_dummies(
            motif_features_df["nearest_qrs_motif_label"], prefix="motif_qrs",
        ).astype(np.float32)
        dummies = pd.concat([dummies, qrs_dummies], axis=1)

    if "nearest_rr_motif_label" in motif_features_df.columns:
        rr_dummies = pd.get_dummies(
            motif_features_df["nearest_rr_motif_label"], prefix="motif_rr",
        ).astype(np.float32)
        dummies = pd.concat([dummies, rr_dummies], axis=1)

    return dummies


# ===================================================================== #
#  Save / Load motif models                                             #
# ===================================================================== #
def save_motifs(
    qrs_motif_dict: dict,
    rr_motif_dict: dict,
    output_dir: str,
) -> None:
    """Save motif models to disk via joblib.

    Parameters
    ----------
    qrs_motif_dict, rr_motif_dict : dict
        Output of ``discover_qrs_motifs()`` and ``discover_rr_motifs()``.
    output_dir : str
        Directory to save into (created if needed).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"version": MOTIF_VERSION, **qrs_motif_dict},
        out / "qrs_motifs.joblib",
    )
    joblib.dump(
        {"version": MOTIF_VERSION, **rr_motif_dict},
        out / "rr_motifs.joblib",
    )
    log.info("Saved motif models → %s", out)


def load_motifs(motif_dir: str) -> tuple[dict, dict]:
    """Load motif models from disk.

    Parameters
    ----------
    motif_dir : str
        Directory containing ``qrs_motifs.joblib`` and ``rr_motifs.joblib``.

    Returns
    -------
    tuple[dict, dict]
        ``(qrs_motif_dict, rr_motif_dict)``.

    Raises
    ------
    FileNotFoundError
        If motif files don't exist.
    ValueError
        If version doesn't match.
    """
    d = Path(motif_dir)

    qrs_path = d / "qrs_motifs.joblib"
    rr_path = d / "rr_motifs.joblib"

    if not qrs_path.exists():
        raise FileNotFoundError(f"QRS motifs not found: {qrs_path}")
    if not rr_path.exists():
        raise FileNotFoundError(f"RR motifs not found: {rr_path}")

    qrs = joblib.load(qrs_path)
    rr = joblib.load(rr_path)

    # Version check
    for name, data in [("QRS", qrs), ("RR", rr)]:
        v = data.get("version", "?")
        if v != MOTIF_VERSION:
            log.warning(
                "%s motif version mismatch: expected %s, got %s — results may differ",
                name, MOTIF_VERSION, v,
            )

    log.info("Loaded motif models from %s", d)
    return qrs, rr


# ===================================================================== #
#  CLI                                                                  #
# ===================================================================== #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motif_features.py",
        description="ECG motif discovery and feature computation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── discover ──────────────────────────────────────────────────────
    p_disc = sub.add_parser("discover", help="Discover QRS and RR motifs")
    p_disc.add_argument("--beat-features", required=True, help="Path to beat_features.parquet")
    p_disc.add_argument("--labels", required=True, help="Path to labels.parquet")
    p_disc.add_argument("--output", required=True, help="Output directory for motif models")
    p_disc.add_argument("--n-qrs-clusters", type=int, default=12)
    p_disc.add_argument("--n-rr-clusters", type=int, default=8)

    # ── compute ───────────────────────────────────────────────────────
    p_comp = sub.add_parser("compute", help="Compute motif features for all beats")
    p_comp.add_argument("--beat-features", required=True)
    p_comp.add_argument("--labels", required=True)
    p_comp.add_argument("--motifs", required=True, help="Motif model directory")
    p_comp.add_argument("--output", required=True, help="Output motif_features.parquet")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "discover":
        # Load data
        bf_path = Path(args.beat_features)
        proc_dir = bf_path.parent

        peaks_df = pd.read_parquet(proc_dir / "peaks.parquet")
        labels_df = pd.read_parquet(args.labels)
        ecg_samples_path = str(proc_dir / "ecg_samples.parquet")

        log.info("Loaded: %d peaks, %d labels", len(peaks_df), len(labels_df))

        # Extract ECG windows
        ecg_windows = _load_ecg_windows(peaks_df, ecg_samples_path, window_size=64)
        log.info("ECG windows shape: %s", ecg_windows.shape)

        # ── Discover QRS motifs ──────────────────────────────────────
        qrs_motifs = discover_qrs_motifs(
            peaks_df, labels_df, ecg_windows,
            n_clusters=args.n_qrs_clusters,
        )

        # ── Discover RR motifs ───────────────────────────────────────
        rr_motifs = discover_rr_motifs(
            peaks_df, labels_df,
            n_clusters=args.n_rr_clusters,
        )

        # ── Save ─────────────────────────────────────────────────────
        save_motifs(qrs_motifs, rr_motifs, args.output)

        # ── Print summary ────────────────────────────────────────────
        print(f"\n{'=' * 72}")
        print("  QRS Motif Discovery")
        print(f"{'=' * 72}")
        print(f"  Beats clustered: {len(qrs_motifs['cluster_assignments'])}")
        print(f"  Clusters: {len(qrs_motifs['cluster_labels'])}")
        print(f"  Inertia: {qrs_motifs['inertia']:.2f}")
        print()

        for ci in range(len(qrs_motifs["cluster_labels"])):
            mask = qrs_motifs["cluster_assignments"] == ci
            n_beats = int(mask.sum())
            label = qrs_motifs["cluster_labels"][ci]
            centroid = qrs_motifs["centroids"][ci]
            spark = _sparkline(centroid)
            ptp = float(np.ptp(centroid))
            print(f"  [{ci:2d}] {label:25s}  n={n_beats:5d}  ptp={ptp:.4f}  {spark}")

        print(f"\n{'=' * 72}")
        print("  RR Motif Discovery")
        print(f"{'=' * 72}")
        print(f"  Windows clustered: {len(rr_motifs['cluster_assignments'])}")
        print(f"  Clusters: {len(rr_motifs['cluster_labels'])}")
        print(f"  Inertia: {rr_motifs['inertia']:.2f}")
        print()

        for ci in range(len(rr_motifs["cluster_labels"])):
            mask = rr_motifs["cluster_assignments"] == ci
            n_wins = int(mask.sum())
            label = rr_motifs["cluster_labels"][ci]
            centroid = rr_motifs["centroids"][ci]
            spark = _sparkline(centroid, width=20)
            mean_rr = float(np.mean(centroid))
            mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
            print(f"  [{ci:2d}] {label:20s}  n={n_wins:5d}  HR≈{mean_hr:.0f}bpm  {spark}")

        print(f"{'=' * 72}")

    elif args.command == "compute":
        bf_path = Path(args.beat_features)
        proc_dir = bf_path.parent

        peaks_df = pd.read_parquet(proc_dir / "peaks.parquet")
        labels_df = pd.read_parquet(args.labels)
        ecg_samples_path = str(proc_dir / "ecg_samples.parquet")

        # Load ECG windows
        ecg_windows = _load_ecg_windows(peaks_df, ecg_samples_path, window_size=64)

        # Load motifs
        qrs_motifs, rr_motifs = load_motifs(args.motifs)

        # Compute features
        result = compute_motif_features(
            peaks_df, labels_df, ecg_windows, qrs_motifs, rr_motifs,
        )

        # Save
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out, index=False, compression="snappy")
        log.info("Saved motif features → %s", out)

        # ── Summary ──────────────────────────────────────────────────
        n = len(result)
        print(f"\n{'=' * 72}")
        print("  Motif Feature Computation")
        print(f"{'=' * 72}")
        print(f"  Total beats: {n}")
        if n > 0:
            print(f"\n  QRS motif distribution:")
            for lbl, cnt in result["nearest_qrs_motif_label"].value_counts().items():
                print(f"    {lbl}: {cnt} ({100.0 * cnt / n:.1f}%)")
            print(f"\n  RR motif distribution:")
            for lbl, cnt in result["nearest_rr_motif_label"].value_counts().items():
                print(f"    {lbl}: {cnt} ({100.0 * cnt / n:.1f}%)")
            print(f"\n  QRS anomaly score: mean={result['qrs_anomaly_score'].mean():.4f}  "
                  f"max={result['qrs_anomaly_score'].max():.4f}")
            print(f"  RR anomaly score:  mean={result['rr_anomaly_score'].mean():.4f}  "
                  f"max={result['rr_anomaly_score'].max():.4f}")
            print(f"  QRS anomalies (>2σ): {result['is_qrs_anomaly'].sum()}")
            print(f"  RR anomalies (>2σ):  {result['is_rr_anomaly'].sum()}")

            # One-hot preview
            dummies = get_motif_dummies(result)
            print(f"\n  One-hot columns ({len(dummies.columns)}): {list(dummies.columns)}")

        print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
