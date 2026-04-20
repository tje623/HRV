#!/usr/bin/env python3
"""
spot_check.py — 8-category disagreement diagnostic for CNN and tabular models.

For each model (CNN and tabular), shows the top 25 beats in four scenarios:
  A. Classified ARTIFACT   — model most UNSURE   (|p - 0.5| smallest)
  B. Classified CLEAN      — model most UNSURE   (|p - 0.5| smallest)
  C. Model CERTAIN ARTIFACT but classified CLEAN  (p highest, ens < 0.5)
  D. Model CERTAIN CLEAN   but classified ARTIFACT (p lowest, ens ≥ 0.5)

Close each window to advance to the next category.

Usage
-----
    cd "/Volumes/xHRV/Artifact Detector"
    source /Users/tannereddy/.envs/hrv/bin/activate
    python spot_check.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

matplotlib.use("MacOSX")

# ── Config ────────────────────────────────────────────────────────────────────
FULL        = Path("/Volumes/xHRV/processed")
CONTEXT_SEC = 2.5    # seconds either side of R-peak
SAMPLE_RATE = 130    # Hz — Polar H10
N_BEATS     = 25
COLS, ROWS  = 5, 5

# Each category: title, one-line subtitle, filter lambda, sort-key lambda,
# ascending (True = take smallest sort-key values first).
CATEGORIES: list[dict] = [
    # ── CNN ───────────────────────────────────────────────────────────────
    {
        "title":    "CNN UNSURE  →  Ensemble: ARTIFACT",
        "subtitle": "Ensemble ≥ 0.5 (artifact) · sorted by |p_cnn − 0.5| ascending",
        "model":    "CNN",
        "filter":   lambda df: df["p_artifact_ensemble"] >= 0.5,
        "sort_key": lambda df: (df["p_artifact_cnn"] - 0.5).abs(),
        "ascending": True,
    },
    {
        "title":    "CNN UNSURE  →  Ensemble: CLEAN",
        "subtitle": "Ensemble < 0.5 (clean) · sorted by |p_cnn − 0.5| ascending",
        "model":    "CNN",
        "filter":   lambda df: df["p_artifact_ensemble"] < 0.5,
        "sort_key": lambda df: (df["p_artifact_cnn"] - 0.5).abs(),
        "ascending": True,
    },
    {
        "title":    "CNN CERTAIN ARTIFACT  →  Ensemble: CLEAN",
        "subtitle": "Ensemble < 0.5 (clean) · sorted by p_cnn descending",
        "model":    "CNN",
        "filter":   lambda df: df["p_artifact_ensemble"] < 0.5,
        "sort_key": lambda df: df["p_artifact_cnn"],
        "ascending": False,
    },
    {
        "title":    "CNN CERTAIN CLEAN  →  Ensemble: ARTIFACT",
        "subtitle": "Ensemble ≥ 0.5 (artifact) · sorted by p_cnn ascending",
        "model":    "CNN",
        "filter":   lambda df: df["p_artifact_ensemble"] >= 0.5,
        "sort_key": lambda df: df["p_artifact_cnn"],
        "ascending": True,
    },
    # ── Tabular ───────────────────────────────────────────────────────────
    {
        "title":    "Tabular UNSURE  →  Ensemble: ARTIFACT",
        "subtitle": "Ensemble ≥ 0.5 (artifact) · sorted by |p_tab − 0.5| ascending",
        "model":    "Tabular",
        "filter":   lambda df: df["p_artifact_ensemble"] >= 0.5,
        "sort_key": lambda df: (df["p_artifact_tabular"] - 0.5).abs(),
        "ascending": True,
    },
    {
        "title":    "Tabular UNSURE  →  Ensemble: CLEAN",
        "subtitle": "Ensemble < 0.5 (clean) · sorted by |p_tab − 0.5| ascending",
        "model":    "Tabular",
        "filter":   lambda df: df["p_artifact_ensemble"] < 0.5,
        "sort_key": lambda df: (df["p_artifact_tabular"] - 0.5).abs(),
        "ascending": True,
    },
    {
        "title":    "Tabular CERTAIN ARTIFACT  →  Ensemble: CLEAN",
        "subtitle": "Ensemble < 0.5 (clean) · sorted by p_tab descending",
        "model":    "Tabular",
        "filter":   lambda df: df["p_artifact_ensemble"] < 0.5,
        "sort_key": lambda df: df["p_artifact_tabular"],
        "ascending": False,
    },
    {
        "title":    "Tabular CERTAIN CLEAN  →  Ensemble: ARTIFACT",
        "subtitle": "Ensemble ≥ 0.5 (artifact) · sorted by p_tab ascending",
        "model":    "Tabular",
        "filter":   lambda df: df["p_artifact_ensemble"] >= 0.5,
        "sort_key": lambda df: df["p_artifact_tabular"],
        "ascending": True,
    },
]

# ── ECG loading ───────────────────────────────────────────────────────────────

def _load_beats(sample_df: pd.DataFrame) -> list[dict]:
    """Load ECG context windows for a set of beats. Returns list of dicts."""
    context_ns = int(CONTEXT_SEC * 1e9)

    peak_id_list = sample_df["peak_id"].astype("int64").tolist()
    peaks_meta = pq.read_table(
        FULL / "peaks.parquet",
        filters=[("peak_id", "in", peak_id_list)],
        columns=["peak_id", "timestamp_ns", "segment_idx"],
    ).to_pandas()
    peaks_meta["peak_id"] = peaks_meta["peak_id"].astype("int64")

    df = sample_df.copy()
    df["peak_id"] = df["peak_id"].astype("int64")
    df = df.merge(peaks_meta, on="peak_id", how="inner")
    if len(df) == 0:
        return []

    seg_ids = df["segment_idx"].unique().tolist()
    ecg_df = pq.read_table(
        FULL / "ecg_samples.parquet",
        filters=[("segment_idx", "in", seg_ids)],
        columns=["timestamp_ns", "ecg", "segment_idx"],
    ).to_pandas().sort_values("timestamp_ns").reset_index(drop=True)

    beats = []
    for _, row in df.iterrows():
        seg_ecg = ecg_df[ecg_df["segment_idx"] == row["segment_idx"]]
        if len(seg_ecg) == 0:
            continue

        ts   = int(row["timestamp_ns"])
        mask = (
            (seg_ecg["timestamp_ns"] >= ts - context_ns)
            & (seg_ecg["timestamp_ns"] <= ts + context_ns)
        )
        ctx = seg_ecg[mask]
        if len(ctx) < 10:
            continue

        ctx_ts  = ctx["timestamp_ns"].values.astype(np.int64)
        ctx_ecg = ctx["ecg"].values.astype(np.float32)

        # Snap R-peak to local amplitude maximum ±5 samples (corrects P-T offset)
        closest = int(np.argmin(np.abs(ctx_ts - ts)))
        snap_lo = max(0, closest - 5)
        snap_hi = min(len(ctx_ecg), closest + 6)
        r_idx   = snap_lo + int(np.argmax(ctx_ecg[snap_lo:snap_hi]))

        beats.append({
            "peak_id": int(row["peak_id"]),
            "p_cnn":   float(row["p_artifact_cnn"]),
            "p_tab":   float(row["p_artifact_tabular"]),
            "p_ens":   float(row["p_artifact_ensemble"]),
            "ecg":     ctx_ecg,
            "r_idx":   r_idx,
        })

    return beats


# ── Plotting ──────────────────────────────────────────────────────────────────

# Header bar colours per model
_MODEL_COLORS = {"CNN": "#ce93d8", "Tabular": "#80cbc4"}

def _plot_grid(
    beats: list[dict],
    title: str,
    subtitle: str,
    model: str,
    cat_num: int,
) -> None:
    accent = _MODEL_COLORS.get(model, "#ffffff")

    fig = plt.figure(figsize=(22, 16), facecolor="#0d1117")
    fig.suptitle(
        f"[{cat_num}/8]  {title}\n{subtitle}   (showing {len(beats)} beats)",
        fontsize=12, fontweight="bold", color=accent, y=0.98,
    )

    gs = gridspec.GridSpec(
        ROWS, COLS, figure=fig,
        hspace=0.65, wspace=0.25,
        left=0.03, right=0.97, top=0.91, bottom=0.03,
    )

    for i, beat in enumerate(beats[:N_BEATS]):
        ax = fig.add_subplot(gs[i // COLS, i % COLS])
        ax.set_facecolor("#16213e")

        ecg = beat["ecg"]
        r   = beat["r_idx"]
        t   = np.arange(len(ecg)) / SAMPLE_RATE

        ax.plot(t, ecg, color="#4fc3f7", linewidth=0.7, alpha=0.9)

        if 0 <= r < len(ecg):
            ax.axvline(t[r], color="#ff6b6b", linewidth=1.2, linestyle="--", alpha=0.7)
            ax.scatter([t[r]], [ecg[r]], color="#ff6b6b", s=40, zorder=5)
            hw = 0.08
            ax.axvspan(
                max(0, t[r] - hw), min(t[-1], t[r] + hw),
                alpha=0.12, color="#ffeb3b",
            )

        cnn, tab, ens = beat["p_cnn"], beat["p_tab"], beat["p_ens"]
        if cnn >= 0.5 and tab >= 0.5:
            tc = "#ff6b6b"    # both say artifact — red
        elif cnn < 0.3 and tab < 0.3:
            tc = "#69db7c"    # both say clean — green
        else:
            tc = "#ffd43b"    # disagreement — yellow

        ax.set_title(
            f"cnn={cnn:.2f}  tab={tab:.2f}\nens={ens:.2f}",
            fontsize=7, pad=3, color=tc,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a3a5c")
            spine.set_linewidth(0.6)

    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not FULL.exists():
        print(f"ERROR: External drive not mounted — {FULL} not found.")
        sys.exit(1)

    print("Loading ensemble predictions ...")
    ens = pd.read_parquet(FULL / "ensemble_preds.parquet")
    print(f"  {len(ens):,} beats loaded\n")

    for cat_num, cat in enumerate(CATEGORIES, 1):
        title    = cat["title"]
        subtitle = cat["subtitle"]
        model    = cat["model"]

        pool = ens[cat["filter"](ens)].copy()
        n    = len(pool)
        print(f"[{cat_num}/8] {title}")
        print(f"         Pool: {n:,} qualifying beats", flush=True)

        if n == 0:
            print("         No matching beats — skipping.\n")
            continue

        # Sort by the diagnostic key and take the top N_BEATS (not random)
        sort_key_series = cat["sort_key"](pool)
        pool["_sort_key"] = sort_key_series.values
        pool = pool.sort_values("_sort_key", ascending=cat["ascending"])
        sample = pool.head(N_BEATS).drop(columns=["_sort_key"])

        print(f"         Loading ECG for {len(sample)} beats ...", flush=True)
        beats = _load_beats(sample)

        if not beats:
            print("         ECG data not found — skipping.\n")
            continue

        print(f"         Plotting — close window to advance.\n")
        _plot_grid(beats, title, subtitle, model, cat_num)

    print("Done.")


if __name__ == "__main__":
    main()
