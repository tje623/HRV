#!/usr/bin/env python3
"""
ecgclean.features.motif_features
================================
Discover recurring ECG waveform patterns (motifs) via k-means clustering
and compute distance-based anomaly features for each beat.

Two motif families:
    1. **QRS motifs** — cluster 63-sample (0.5 s @ 125 Hz) beat windows by
       morphology.
    2. **RR motifs**  — cluster sliding windows of 10 consecutive RR intervals.

These features tell downstream models how unusual each beat is relative
to the patient's full ECG history.

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
MOTIF_VERSION = "2.0"
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    SAMPLE_RATE_HZ,
    QRS_WINDOW_SAMPLES,
    HR_BRADY_BPM,
    HR_TACHY_BPM,
    RR_FALLBACK_MS,
    POTS_MIN_FRAC_DECREASING,
    POTS_MIN_FRAC_INCREASING,
    POTS_MIN_RR_DECLINE_MS,
    POTS_MIN_RR_RECOVERY_MS,
)


# ===================================================================== #
#  ECG window extraction                                                #
# ===================================================================== #
def _load_ecg_windows(
    peaks_df: pd.DataFrame,
    ecg_samples_path: str,
    window_size: int = QRS_WINDOW_SAMPLES,
) -> np.ndarray:
    """Extract ECG windows centered on each R-peak.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Must have ``peak_id`` and ``timestamp_ms``.
    ecg_samples_path : str
        Path to ``ecg_samples.parquet``.
    window_size : int
        Number of samples per window (default ``QRS_WINDOW_SAMPLES`` = 65).

    Returns
    -------
    np.ndarray
        Shape ``(n_peaks, window_size)``, dtype float32.
    """
    ecg_samples = pd.read_parquet(ecg_samples_path)
    n_peaks = len(peaks_df)
    windows = np.zeros((n_peaks, window_size), dtype=np.float32)

    ecg_ts = ecg_samples["timestamp_ms"].values.astype(np.int64)
    ecg_vals = ecg_samples["ecg"].values.astype(np.float32)
    n_ecg = len(ecg_ts)

    if n_ecg == 0:
        log.warning("No ECG samples available for window extraction")
        return windows

    peak_ts = peaks_df["timestamp_ms"].values.astype(np.int64)
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
    if len(values) == 0:
        return ""
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width).astype(int)
        values = values[indices]
    elif len(values) < width:
        indices = np.linspace(0, len(values) - 1, width)
        values = np.interp(indices, np.arange(len(values)), values)

    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-10:
        return SPARKLINE_CHARS[4] * width

    scaled = (values - vmin) / (vmax - vmin) * (len(SPARKLINE_CHARS) - 1)
    return "".join(SPARKLINE_CHARS[int(np.clip(v, 0, len(SPARKLINE_CHARS) - 1))] for v in scaled)


# ===================================================================== #
#  POTS transition detection (RR-dynamics only)                         #
# ===================================================================== #
def _detect_rr_pattern(centroid: np.ndarray) -> str:
    """Classify an RR-interval window centroid by its temporal dynamics.

    Detection is based solely on RR interval sequences — NOT on physiological
    event annotations (which mark PACs/PVCs/vagal events, not POTS transitions).

    A POTS ramp-up is identified by:
      - Majority of consecutive RR pairs are decreasing (HR accelerating)
      - Total RR contraction over the window exceeds POTS_MIN_RR_DECLINE_MS

    A POTS recovery is identified by:
      - Majority of consecutive RR pairs are increasing (HR decelerating)
      - Total RR expansion over the window exceeds POTS_MIN_RR_RECOVERY_MS
    """
    diffs = np.diff(centroid)
    frac_dec = float(np.mean(diffs < 0))
    frac_inc = float(np.mean(diffs > 0))
    total_decline = float(centroid[0] - centroid[-1])    # + = RR contracted
    total_recovery = float(centroid[-1] - centroid[0])   # + = RR expanded

    if (frac_dec > POTS_MIN_FRAC_DECREASING
            and total_decline > POTS_MIN_RR_DECLINE_MS):
        return "pots_ramp_up"

    if (frac_inc > POTS_MIN_FRAC_INCREASING
            and total_recovery > POTS_MIN_RR_RECOVERY_MS):
        return "pots_recovery"

    return ""   # no strong ramp pattern; caller applies HR-based label


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
    """Cluster QRS windows to discover recurring morphologies.

    Parameters
    ----------
    peaks_df : pd.DataFrame
        Must have ``peak_id``.  Row order matches ``ecg_windows``.
    labels_df : pd.DataFrame
        Must have ``peak_id``, ``label``, ``hard_filtered``.
        ``phys_event_window`` is intentionally NOT used here — those
        annotations mark PACs/PVCs/vagal events, not morphology clusters.
    ecg_windows : np.ndarray
        Shape ``(n_beats, QRS_WINDOW_SAMPLES)``.
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
    merged = peaks_df[["peak_id"]].copy()
    merged["_row"] = np.arange(len(merged))

    label_cols = ["peak_id", "label", "hard_filtered"]
    if "rr_prev_ms" in labels_df.columns:
        label_cols.append("rr_prev_ms")

    merged = merged.merge(labels_df[label_cols], on="peak_id", how="left")

    if use_clean_only:
        mask = (merged["label"] == "clean") & (~merged["hard_filtered"].fillna(False))
        selection = merged[mask].copy()
    else:
        selection = merged.copy()

    if len(selection) == 0:
        log.warning("No clean beats for QRS motif discovery — using all beats")
        selection = merged.copy()

    row_indices = selection["_row"].values
    selected_windows = ecg_windows[row_indices]
    selected_peak_ids = selection["peak_id"].values

    log.info("QRS motif discovery: %d beats, %d clusters", len(selection), n_clusters)

    actual_k = min(n_clusters, len(selection))
    if actual_k < n_clusters:
        log.warning("Only %d beats — reducing to %d clusters", len(selection), actual_k)

    km = KMeans(n_clusters=actual_k, random_state=random_seed, n_init=10, max_iter=300)
    assignments = km.fit_predict(selected_windows)
    centroids = km.cluster_centers_

    cluster_labels = []
    for ci in range(actual_k):
        cluster_mask = assignments == ci
        cluster_beats = selection[cluster_mask]
        cluster_windows = selected_windows[cluster_mask]

        n_beats_ci = int(cluster_mask.sum())
        window_ptp = np.ptp(cluster_windows, axis=1)
        mean_ptp = float(np.mean(window_ptp)) if n_beats_ci > 0 else 0.0
        std_ptp = float(np.std(window_ptp)) if n_beats_ci > 0 else 0.0

        if "rr_prev_ms" in cluster_beats.columns:
            rr_vals = cluster_beats["rr_prev_ms"].dropna().values
            mean_hr = (
                60000.0 / float(np.mean(rr_vals)) if len(rr_vals) > 0
                else 60000.0 / RR_FALLBACK_MS
            )
        else:
            mean_hr = 60000.0 / RR_FALLBACK_MS

        # Morphological quality check first — overrides HR-based labels.
        # High inter-beat amplitude variance within a cluster indicates noise.
        if std_ptp > 0.3 * mean_ptp and mean_ptp > 0:
            label = "noisy_cluster"
        # Patient-specific HR-based labels.
        # Note: HR > 130 BPM is common POTS physiology for this patient,
        # not an artifact.  HR regularly reaches 186–190+ BPM (e.g. showers).
        elif mean_hr < HR_BRADY_BPM:
            label = "bradycardic"
        elif mean_hr > HR_TACHY_BPM:
            label = "tachycardic"
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

    POTS transition windows are detected from RR dynamics alone —
    progressive contraction ≥ 150 ms over the window with ≥ 65% of
    consecutive pairs decreasing.

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
    merged = peaks_df[["peak_id"]].merge(
        labels_df[["peak_id", "rr_prev_ms"]], on="peak_id", how="left",
    )
    rr = merged["rr_prev_ms"].values.astype(np.float64)
    peak_ids_all = merged["peak_id"].values

    rr_median = np.nanmedian(rr)
    if np.isnan(rr_median):
        rr_median = RR_FALLBACK_MS
    rr_filled = np.where(np.isnan(rr), rr_median, rr)

    n = len(rr_filled)
    if n < window_size:
        log.warning("Only %d beats — cannot form RR windows of size %d", n, window_size)
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
        windows.append(rr_filled[i : i + window_size])
        window_peak_ids.append(peak_ids_all[i + window_size // 2])

    rr_windows = np.array(windows, dtype=np.float64)
    window_peak_ids = np.array(window_peak_ids)

    log.info(
        "RR motif discovery: %d windows of size %d, %d clusters",
        len(rr_windows), window_size, n_clusters,
    )

    actual_k = min(n_clusters, len(rr_windows))
    if actual_k < n_clusters:
        log.warning("Only %d RR windows — reducing to %d clusters", len(rr_windows), actual_k)

    km = KMeans(n_clusters=actual_k, random_state=random_seed, n_init=10, max_iter=300)
    assignments = km.fit_predict(rr_windows)
    centroids = km.cluster_centers_

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
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 60000.0 / RR_FALLBACK_MS
        cv = std_rr / mean_rr if mean_rr > 0 else 0.0

        centroid = centroids[ci]

        # POTS ramp/recovery detection from RR dynamics.
        ramp_label = _detect_rr_pattern(centroid)
        if ramp_label:
            cluster_labels.append(ramp_label)
            continue

        # High coefficient of variation → erratic/noisy RR pattern.
        if cv > 0.25:
            cluster_labels.append("erratic")
            continue

        # Patient-specific HR labels.
        # HR regularly exceeds 130 BPM for this POTS patient (routinely ~186
        # during showers, into the 190s otherwise) — "tachycardic" here means
        # a high-HR cluster, not an implausible beat.
        if mean_hr < HR_BRADY_BPM:
            cluster_labels.append("bradycardic")
        elif mean_hr > HR_TACHY_BPM:
            cluster_labels.append("tachycardic")
        else:
            cluster_labels.append("stable_sinus")

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
        Shape ``(n_beats, QRS_WINDOW_SAMPLES)``.
    qrs_motif_dict : dict
        Output of ``discover_qrs_motifs()``.
    rr_motif_dict : dict
        Output of ``discover_rr_motifs()``.

    Returns
    -------
    pd.DataFrame
        One row per beat.  Columns: ``peak_id``,
        ``dist_to_nearest_qrs_motif``, ``nearest_qrs_motif_label``,
        ``nearest_qrs_motif_idx``, ``dist_to_nearest_rr_motif``,
        ``nearest_rr_motif_label``, ``nearest_rr_motif_idx``,
        ``qrs_anomaly_score``, ``rr_anomaly_score``,
        ``is_qrs_anomaly``, ``is_rr_anomaly``.
    """
    n = len(peaks_df)
    peak_ids = peaks_df["peak_id"].values

    # ── QRS distances ────────────────────────────────────────────────
    qrs_centroids = qrs_motif_dict["centroids"]
    qrs_labels = qrs_motif_dict["cluster_labels"]

    qrs_dists = np.linalg.norm(
        ecg_windows[:, np.newaxis, :] - qrs_centroids[np.newaxis, :, :],
        axis=2,
    )

    nearest_qrs_idx = qrs_dists.argmin(axis=1)
    nearest_qrs_dist = qrs_dists[np.arange(n), nearest_qrs_idx]
    nearest_qrs_label = [qrs_labels[int(i)] for i in nearest_qrs_idx]

    # ── RR distances ─────────────────────────────────────────────────
    rr_window_size = rr_motif_dict["centroids"].shape[1]
    rr_centroids = rr_motif_dict["centroids"]
    rr_labels = rr_motif_dict["cluster_labels"]

    merged = peaks_df[["peak_id"]].merge(
        labels_df[["peak_id", "rr_prev_ms"]], on="peak_id", how="left",
    )
    rr = merged["rr_prev_ms"].values.astype(np.float64)
    rr_median = np.nanmedian(rr)
    if np.isnan(rr_median):
        rr_median = RR_FALLBACK_MS
    rr_filled = np.where(np.isnan(rr), rr_median, rr)

    half_w = rr_window_size // 2
    rr_windows = np.zeros((n, rr_window_size), dtype=np.float64)
    for i in range(n):
        start = i - half_w
        end = start + rr_window_size
        src_start = max(0, start)
        src_end = min(n, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        rr_windows[i, dst_start:dst_end] = rr_filled[src_start:src_end]

    rr_dists = np.linalg.norm(
        rr_windows[:, np.newaxis, :] - rr_centroids[np.newaxis, :, :],
        axis=2,
    )

    nearest_rr_idx = rr_dists.argmin(axis=1)
    nearest_rr_dist = rr_dists[np.arange(n), nearest_rr_idx]
    nearest_rr_label = [rr_labels[int(i)] for i in nearest_rr_idx]

    # ── Anomaly scores (normalized by mean clean-beat distance) ──────
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

    mean_qrs_dist_clean = max(mean_qrs_dist_clean, 1e-8)
    mean_rr_dist_clean = max(mean_rr_dist_clean, 1e-8)

    qrs_anomaly_score = (nearest_qrs_dist / mean_qrs_dist_clean).astype(np.float32)
    rr_anomaly_score = (nearest_rr_dist / mean_rr_dist_clean).astype(np.float32)

    return pd.DataFrame({
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


# ===================================================================== #
#  One-hot encoding helper                                              #
# ===================================================================== #
def get_motif_dummies(motif_features_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the motif label columns for use as model features."""
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
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": MOTIF_VERSION, **qrs_motif_dict}, out / "qrs_motifs.joblib")
    joblib.dump({"version": MOTIF_VERSION, **rr_motif_dict}, out / "rr_motifs.joblib")
    log.info("Saved motif models → %s", out)


def load_motifs(motif_dir: str) -> tuple[dict, dict]:
    d = Path(motif_dir)
    qrs_path = d / "qrs_motifs.joblib"
    rr_path = d / "rr_motifs.joblib"

    if not qrs_path.exists():
        raise FileNotFoundError(f"QRS motifs not found: {qrs_path}")
    if not rr_path.exists():
        raise FileNotFoundError(f"RR motifs not found: {rr_path}")

    qrs = joblib.load(qrs_path)
    rr = joblib.load(rr_path)

    for name, data in [("QRS", qrs), ("RR", rr)]:
        v = data.get("version", "?")
        if v != MOTIF_VERSION:
            log.warning(
                "%s motif version mismatch: expected %s, got %s — re-run discover",
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

    p_disc = sub.add_parser("discover", help="Discover QRS and RR motifs")
    p_disc.add_argument("--beat-features", required=True)
    p_disc.add_argument("--labels", required=True)
    p_disc.add_argument("--output", required=True, help="Output directory for motif models")
    p_disc.add_argument("--n-qrs-clusters", type=int, default=12)
    p_disc.add_argument("--n-rr-clusters", type=int, default=8)

    p_comp = sub.add_parser("compute", help="Compute motif features for all beats")
    p_comp.add_argument("--beat-features", required=True)
    p_comp.add_argument("--labels", required=True)
    p_comp.add_argument("--motifs", required=True)
    p_comp.add_argument("--output", required=True)

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "discover":
        bf_path = Path(args.beat_features)
        proc_dir = bf_path.parent

        peaks_df = pd.read_parquet(proc_dir / "peaks.parquet")
        labels_df = pd.read_parquet(args.labels)
        ecg_samples_path = str(proc_dir / "ecg_samples.parquet")

        log.info("Loaded: %d peaks, %d labels", len(peaks_df), len(labels_df))

        ecg_windows = _load_ecg_windows(peaks_df, ecg_samples_path)
        log.info("ECG windows shape: %s", ecg_windows.shape)

        qrs_motifs = discover_qrs_motifs(
            peaks_df, labels_df, ecg_windows,
            n_clusters=args.n_qrs_clusters,
        )
        rr_motifs = discover_rr_motifs(
            peaks_df, labels_df,
            n_clusters=args.n_rr_clusters,
        )

        save_motifs(qrs_motifs, rr_motifs, args.output)

        print(f"\n{'=' * 72}")
        print("  QRS Motif Discovery")
        print(f"{'=' * 72}")
        print(f"  Window size: {QRS_WINDOW_SAMPLES} samples = {QRS_WINDOW_SAMPLES / SAMPLE_RATE_HZ * 1000:.0f} ms @ {SAMPLE_RATE_HZ} Hz")
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
            print(f"  [{ci:2d}] {label:20s}  n={n_wins:5d}  HR≈{mean_hr:.0f} BPM  {spark}")

        print(f"{'=' * 72}")

    elif args.command == "compute":
        bf_path = Path(args.beat_features)
        proc_dir = bf_path.parent

        peaks_df = pd.read_parquet(proc_dir / "peaks.parquet")
        labels_df = pd.read_parquet(args.labels)
        ecg_samples_path = str(proc_dir / "ecg_samples.parquet")

        ecg_windows = _load_ecg_windows(peaks_df, ecg_samples_path)

        qrs_motifs, rr_motifs = load_motifs(args.motifs)

        result = compute_motif_features(
            peaks_df, labels_df, ecg_windows, qrs_motifs, rr_motifs,
        )

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out, index=False, compression="snappy")
        log.info("Saved motif features → %s", out)

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
            print(
                f"\n  QRS anomaly score: mean={result['qrs_anomaly_score'].mean():.4f}"
                f"  max={result['qrs_anomaly_score'].max():.4f}"
            )
            print(
                f"  RR anomaly score:  mean={result['rr_anomaly_score'].mean():.4f}"
                f"  max={result['rr_anomaly_score'].max():.4f}"
            )
            print(f"  QRS anomalies (>2×): {result['is_qrs_anomaly'].sum()}")
            print(f"  RR anomalies (>2×):  {result['is_rr_anomaly'].sum()}")

            dummies = get_motif_dummies(result)
            print(f"\n  One-hot columns ({len(dummies.columns)}): {list(dummies.columns)}")

        print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
