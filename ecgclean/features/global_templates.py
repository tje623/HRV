#!/usr/bin/env python3
"""
ecgclean/features/global_templates.py — Global QRS Template Builder & Correlator

Two subcommands:

  build      Build a global clean QRS template from all manually-validated
             clean beats across the entire dataset and save it as a joblib
             dict.  The template is the mean of all verified-clean 64-sample
             ECG windows — it is label-contamination-safe because it only
             uses beats where reviewed == True AND label == "clean" AND
             hard_filtered == False.

  correlate  Load a saved template set and compute per-beat Pearson
             correlation to each template for all peaks.  Outputs a small
             parquet (peak_id, global_corr_clean) that can be joined onto
             beat_features.parquet before model training / inference.
             Processes 50M+ beats in segment-keyed chunks — never loads the
             full ECG parquet into memory.

Usage
-----
    # Build global template from verified-clean beats
    python ecgclean/features/global_templates.py build \\
        --peaks      data/processed/peaks.parquet \\
        --labels     data/processed/labels.parquet \\
        --ecg-samples data/processed/ecg_samples.parquet \\
        --output     data/templates/global_templates.joblib

    # Compute per-beat correlations (chunked, 50M-beat safe)
    python ecgclean/features/global_templates.py correlate \\
        --templates  data/templates/global_templates.joblib \\
        --peaks      data/processed/peaks.parquet \\
        --ecg-samples data/processed/ecg_samples.parquet \\
        --output     data/processed/global_template_features.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# pearson_corr_safe: import from package if available, else inline fallback.
# The fallback is intentionally identical to the canonical implementation in
# ecgclean/features/__init__.py so this file stays standalone-runnable.
try:
    from ecgclean.features import pearson_corr_safe
except ImportError:
    def pearson_corr_safe(a: np.ndarray, b: np.ndarray) -> float:  # type: ignore[misc]
        """Pearson correlation with zero-variance protection (inline fallback)."""
        if len(a) < 2 or len(b) < 2:
            return 0.0
        a_std = np.std(a)
        b_std = np.std(b)
        if a_std == 0.0 or b_std == 0.0:
            return 0.0
        r = np.corrcoef(a, b)[0, 1]
        if np.isnan(r) or np.isinf(r):
            return 0.0
        return float(r)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ecgclean.features.global_templates")

# ── Constants ─────────────────────────────────────────────────────────────────

# Pan-Tompkins finds the MWI peak (biased ~2-8 samples toward the QRS upslope),
# not the R-peak apex.  Snap the window center to argmax(|ecg|) within ±8
# samples.  Must match beat_features.py exactly.
_PEAK_SNAP_SAMPLES: int = 8   # ±8 samples = ±64ms at 125 Hz

_WINDOW_SIZE: int = 64
_SAMPLE_RATE_HZ: int = 125    # Polar H10 (empirically 8.000 ms/sample = 125 Hz) — do NOT use 130 or 256


# ═══════════════════════════════════════════════════════════════════════════════
# ECG WINDOW LOADER
# Copied from beat_features.py — kept here so this file is standalone-runnable
# without a package install.  Keep in sync with that file's version.
# ═══════════════════════════════════════════════════════════════════════════════


def _load_ecg_windows(
    peaks_df: pd.DataFrame,
    ecg_samples_df: pd.DataFrame,
    window_size: int = _WINDOW_SIZE,
) -> np.ndarray:
    """Reconstruct ECG windows from ecg_samples for each peak.

    For each peak, extracts ``window_size`` samples centered on the peak
    timestamp from the ECG time series.  Uses binary search for fast
    nearest-sample lookup, then snaps the center to argmax(|ecg|) within
    ±_PEAK_SNAP_SAMPLES to correct the Pan-Tompkins MWI bias.

    Args:
        peaks_df: Peaks table with ``timestamp_ns`` column.
        ecg_samples_df: ECG samples table with ``timestamp_ns`` and ``ecg``
            columns, sorted by timestamp_ns.
        window_size: Number of ECG samples per window (default 64).

    Returns:
        numpy array of shape (n_peaks, window_size), dtype float32.
        Rows for peaks that fall outside the ECG time range are zero-padded.
    """
    n_peaks = len(peaks_df)
    windows = np.zeros((n_peaks, window_size), dtype=np.float32)

    ecg_ts   = ecg_samples_df["timestamp_ns"].values.astype(np.int64)
    ecg_vals = ecg_samples_df["ecg"].values.astype(np.float32)
    n_ecg    = len(ecg_ts)

    if n_ecg == 0:
        logger.warning("No ECG samples available for window reconstruction")
        return windows

    peak_ts = peaks_df["timestamp_ns"].values.astype(np.int64)
    half    = window_size // 2

    # Binary search: for each peak timestamp find the closest ECG sample index
    insert_idx = np.searchsorted(ecg_ts, peak_ts, side="left")

    for i in range(n_peaks):
        center = int(insert_idx[i])
        # Refine: pick the closer of the left/right neighbour
        if 0 < center < n_ecg:
            if abs(ecg_ts[center - 1] - peak_ts[i]) < abs(ecg_ts[center] - peak_ts[i]):
                center = center - 1
        elif center >= n_ecg:
            center = n_ecg - 1

        # Snap to local amplitude maximum within ±_PEAK_SNAP_SAMPLES
        snap_lo = max(0, center - _PEAK_SNAP_SAMPLES)
        snap_hi = min(n_ecg, center + _PEAK_SNAP_SAMPLES + 1)
        center  = snap_lo + int(np.argmax(np.abs(ecg_vals[snap_lo:snap_hi])))

        start = center - half
        end   = start + window_size

        # Clip to valid range; zero-pad boundaries
        src_start = max(0, start)
        src_end   = min(n_ecg, end)
        dst_start = src_start - start
        dst_end   = dst_start + (src_end - src_start)

        if src_end > src_start:
            windows[i, dst_start:dst_end] = ecg_vals[src_start:src_end]

    return windows


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH CORRELATION HELPER
# Vectorised Pearson correlation of every row in `windows` against a single
# fixed `template`.  ~1000× faster than a Python loop over pearson_corr_safe.
# ═══════════════════════════════════════════════════════════════════════════════


def _pearson_batch(windows: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row in *windows* against *template*.

    Zero-variance rows (constant signal) return 0.0 rather than NaN.

    Args:
        windows: (n, L) array of beat windows.
        template: (L,) template to correlate against.

    Returns:
        (n,) float32 array of Pearson r values in [-1, 1].
    """
    w = windows.astype(np.float64)             # (n, L)
    t = template.astype(np.float64)            # (L,)

    w_centered = w - w.mean(axis=1, keepdims=True)
    t_centered = t - t.mean()

    numerator  = (w_centered * t_centered).sum(axis=1)               # (n,)
    w_norm     = np.sqrt((w_centered ** 2).sum(axis=1))              # (n,)
    t_norm     = float(np.sqrt((t_centered ** 2).sum()))

    denom = w_norm * t_norm
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom > 0.0, numerator / denom, 0.0)

    result = np.where(np.isfinite(result), result, 0.0)
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD MODE
# ═══════════════════════════════════════════════════════════════════════════════


def build_templates(
    peaks_path: Path,
    labels_path: Path,
    ecg_samples_path: Path,
    output_path: Path,
    chunk_segments: int = 5000,
) -> None:
    """Build and save a global clean QRS template.

    Selects beats where ``label == "clean"``, ``hard_filtered == False``,
    and (if available) ``reviewed == True``.  Extracts 64-sample ECG windows
    for those beats, computes the mean, and saves the result as a joblib dict.

    Args:
        peaks_path: Path to peaks.parquet.
        labels_path: Path to labels.parquet (output of physio_constraints.py).
        ecg_samples_path: Path to ecg_samples.parquet.
        output_path: Destination for the .joblib template file.
        chunk_segments: Number of 60-s segments to load per ECG read.
    """
    logger.info("=== BUILD: global QRS template ===")

    # ── Load peaks & labels ───────────────────────────────────────────────────
    logger.info("Loading peaks from %s", peaks_path)
    peaks_df = pd.read_parquet(peaks_path)
    logger.info("Loading labels from %s", labels_path)
    labels_df = pd.read_parquet(labels_path)

    logger.info("Peaks: %d total", len(peaks_df))
    logger.info("Labels: %d total", len(labels_df))

    # ── Filter to verified-clean beats ───────────────────────────────────────
    # Merge labels onto peaks via peak_id
    merged = peaks_df.merge(
        labels_df[["peak_id", "label", "hard_filtered"]
                  + (["reviewed"] if "reviewed" in labels_df.columns else [])],
        on="peak_id",
        how="inner",
    )

    if "reviewed" in merged.columns:
        clean_mask = (
            (merged["label"] == "clean")
            & (~merged["hard_filtered"].astype(bool))
            & (merged["reviewed"].astype(bool))
        )
        logger.info(
            "Filtering: label=='clean' AND ~hard_filtered AND reviewed==True"
        )
    else:
        logger.warning(
            "Column 'reviewed' not found in labels.parquet — "
            "including ALL label=='clean' beats that are not hard_filtered. "
            "Unreviewed beats may contaminate the template."
        )
        clean_mask = (
            (merged["label"] == "clean")
            & (~merged["hard_filtered"].astype(bool))
        )

    clean_peaks = merged[clean_mask].copy()
    n_clean = len(clean_peaks)

    if n_clean == 0:
        logger.error("No clean beats found — cannot build template. Exiting.")
        sys.exit(1)

    logger.info("Clean beats selected: %d", n_clean)

    # ── Extract ECG windows for clean beats (chunked by segment) ─────────────
    clean_peaks_sorted = clean_peaks.sort_values("segment_idx").reset_index(drop=True)
    seg_arr  = clean_peaks_sorted["segment_idx"].values
    all_segs = sorted(clean_peaks_sorted["segment_idx"].unique())
    n_segs   = len(all_segs)
    n_chunks = (n_segs + chunk_segments - 1) // chunk_segments

    logger.info(
        "Extracting windows across %d unique segment(s) in %d chunk(s)",
        n_segs, n_chunks,
    )

    all_windows: list[np.ndarray] = []

    for chunk_num, chunk_start in enumerate(range(0, n_segs, chunk_segments), 1):
        chunk_segs = all_segs[chunk_start: chunk_start + chunk_segments]
        seg_min    = int(chunk_segs[0])
        seg_max    = int(chunk_segs[-1])

        lo = int(np.searchsorted(seg_arr, seg_min, side="left"))
        hi = int(np.searchsorted(seg_arr, seg_max + 1, side="left"))
        chunk_peaks = clean_peaks_sorted.iloc[lo:hi].copy()

        if len(chunk_peaks) == 0:
            continue

        logger.info(
            "[%d/%d] Segments %d–%d | %d clean peaks",
            chunk_num, n_chunks, seg_min, seg_max, len(chunk_peaks),
        )

        # Predicate pushdown: only load ECG samples for these segments (+1 buffer)
        ecg_chunk = (
            pq.read_table(
                ecg_samples_path,
                filters=[
                    ("segment_idx", ">=", max(0, seg_min - 1)),
                    ("segment_idx", "<=", seg_max + 1),
                ],
                columns=["timestamp_ns", "ecg"],
            )
            .to_pandas()
            .sort_values("timestamp_ns")
            .reset_index(drop=True)
        )

        windows = _load_ecg_windows(chunk_peaks, ecg_chunk)
        del ecg_chunk

        all_windows.append(windows)

    # ── Compute global mean template ──────────────────────────────────────────
    combined = np.concatenate(all_windows, axis=0)   # (n_clean, 64)
    logger.info("Stacked %d windows of shape %s", combined.shape[0], combined.shape[1:])

    template_clean = combined.mean(axis=0).astype(np.float64)   # (64,)

    logger.info(
        "template_clean — shape: %s | min: %.4f | max: %.4f | std: %.4f",
        template_clean.shape,
        float(template_clean.min()),
        float(template_clean.max()),
        float(template_clean.std()),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "template_clean": template_clean,           # (64,) float64
        "n_clean_beats":  int(n_clean),             # int
        "built_at":       datetime.now().isoformat(),  # str
        "sample_rate_hz": _SAMPLE_RATE_HZ,          # int
        "window_size":    _WINDOW_SIZE,             # int
        "version":        "1.0",                    # str
    }

    # TODO: artifact prototype templates — add here once five-category
    # reannotation is complete.  Each subtype template is the mean of beats
    # labeled with that specific category:
    #
    #   "template_jitter"          — abrupt amplitude / timing jitter
    #   "template_baseline_wander" — slow drift contaminating the QRS window
    #   "template_low_amplitude"   — sub-threshold / stubby bumps
    #   "template_saturation"      — ADC-clipped / flat-topped peaks
    #
    # In the correlate step, add corresponding output columns:
    #   global_corr_jitter, global_corr_wander,
    #   global_corr_low_amplitude, global_corr_saturation
    #
    # Prerequisites: five-category reannotation must be imported into
    # labels.parquet before building these prototypes.

    joblib.dump(payload, output_path)
    logger.info("Saved template → %s", output_path)
    print(f"\n{'=' * 70}")
    print(f"  Global template built from {n_clean:,} clean beats")
    print(f"  Saved → {output_path}")
    print(f"{'=' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATE MODE
# ═══════════════════════════════════════════════════════════════════════════════


def correlate_templates(
    templates_path: Path,
    peaks_path: Path,
    ecg_samples_path: Path,
    output_path: Path,
    chunk_segments: int = 5000,
) -> None:
    """Compute per-beat Pearson correlation to each saved template.

    Processes all peaks in segment-keyed chunks — never loads the full ECG
    parquet into memory.  Writes a two-column parquet:
    ``(peak_id, global_corr_clean)``.

    Args:
        templates_path: Path to the .joblib file produced by ``build``.
        peaks_path: Path to peaks.parquet.
        ecg_samples_path: Path to ecg_samples.parquet.
        output_path: Destination parquet for the correlation features.
        chunk_segments: Number of 60-s segments to process per batch.
    """
    logger.info("=== CORRELATE: global template features ===")

    # ── Load templates ────────────────────────────────────────────────────────
    logger.info("Loading templates from %s", templates_path)
    tmpl = joblib.load(templates_path)

    template_clean: np.ndarray = tmpl["template_clean"]
    logger.info(
        "template_clean loaded — built from %d beats at %s",
        tmpl.get("n_clean_beats", "?"),
        tmpl.get("built_at", "?"),
    )
    logger.info(
        "template_clean stats — min: %.4f | max: %.4f | std: %.4f",
        float(template_clean.min()),
        float(template_clean.max()),
        float(template_clean.std()),
    )

    # Sanity check: template version compatibility
    tmpl_sr = tmpl.get("sample_rate_hz", None)
    if tmpl_sr is not None and int(tmpl_sr) != _SAMPLE_RATE_HZ:
        logger.warning(
            "Template sample_rate_hz=%d but pipeline expects %d — "
            "correlations may be meaningless",
            tmpl_sr, _SAMPLE_RATE_HZ,
        )

    # ── Load peaks ────────────────────────────────────────────────────────────
    logger.info("Loading peaks from %s", peaks_path)
    peaks_df = pd.read_parquet(peaks_path)
    logger.info("Peaks: %d total", len(peaks_df))

    # Sort once so each chunk is a contiguous slice (O(log n) boundaries)
    peaks_sorted = peaks_df.sort_values("segment_idx").reset_index(drop=True)
    seg_arr  = peaks_sorted["segment_idx"].values
    all_segs = sorted(peaks_sorted["segment_idx"].unique())
    total_segs = len(all_segs)
    n_chunks   = (total_segs + chunk_segments - 1) // chunk_segments

    logger.info(
        "Processing %d segment(s) in %d chunk(s) of up to %d segments",
        total_segs, n_chunks, chunk_segments,
    )

    # ── Output schema ─────────────────────────────────────────────────────────
    schema = pa.schema([
        ("peak_id",          pa.int64()),
        ("global_corr_clean", pa.float32()),
        # TODO: add global_corr_jitter, global_corr_wander,
        #       global_corr_low_amplitude, global_corr_saturation
        #       once artifact prototype templates are built.
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total_beats = 0

    # ── Chunk loop ────────────────────────────────────────────────────────────
    for chunk_num, chunk_start in enumerate(range(0, total_segs, chunk_segments), 1):
        chunk_segs = all_segs[chunk_start: chunk_start + chunk_segments]
        seg_min    = int(chunk_segs[0])
        seg_max    = int(chunk_segs[-1])

        lo = int(np.searchsorted(seg_arr, seg_min, side="left"))
        hi = int(np.searchsorted(seg_arr, seg_max + 1, side="left"))
        chunk_peaks = peaks_sorted.iloc[lo:hi].copy()

        if len(chunk_peaks) == 0:
            continue

        logger.info(
            "[%d/%d] Segments %d–%d | %d peaks",
            chunk_num, n_chunks, seg_min, seg_max, len(chunk_peaks),
        )

        # Predicate pushdown: +/-1 segment buffer so edge-peaks get full windows
        ecg_chunk = (
            pq.read_table(
                ecg_samples_path,
                filters=[
                    ("segment_idx", ">=", max(0, seg_min - 1)),
                    ("segment_idx", "<=", seg_max + 1),
                ],
                columns=["timestamp_ns", "ecg"],
            )
            .to_pandas()
            .sort_values("timestamp_ns")
            .reset_index(drop=True)
        )

        windows = _load_ecg_windows(chunk_peaks, ecg_chunk)
        del ecg_chunk

        # Vectorised Pearson correlation against template_clean
        corr_clean = _pearson_batch(windows, template_clean)
        del windows

        table = pa.table(
            {
                "peak_id":           pa.array(chunk_peaks["peak_id"].values, type=pa.int64()),
                "global_corr_clean": pa.array(corr_clean,                    type=pa.float32()),
            },
            schema=schema,
        )

        if writer is None:
            writer = pq.ParquetWriter(output_path, schema, compression="snappy")
        writer.write_table(table)

        total_beats += len(chunk_peaks)
        logger.info("  wrote %d beats (running total: %d)", len(chunk_peaks), total_beats)

    if writer is not None:
        writer.close()

    logger.info("Saved global template features → %s  (%d beats)", output_path, total_beats)
    print(f"\n{'=' * 70}")
    print(f"  Global Template Correlations: {total_beats:,} beats → {output_path}")
    print(f"{'=' * 70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point — dispatches to build or correlate subcommand."""
    parser = argparse.ArgumentParser(
        description="ECG Artifact Pipeline — Global QRS Template Builder & Correlator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")

    # ── build subcommand ──────────────────────────────────────────────────────
    build_p = subparsers.add_parser(
        "build",
        help="Build global QRS template from verified-clean beats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    build_p.add_argument(
        "--peaks",
        type=str,
        required=True,
        help="Path to peaks.parquet",
    )
    build_p.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels.parquet (output of physio_constraints.py)",
    )
    build_p.add_argument(
        "--ecg-samples",
        type=str,
        required=True,
        help="Path to ecg_samples.parquet",
    )
    build_p.add_argument(
        "--output",
        type=str,
        default="data/templates/global_templates.joblib",
        help="Output path for the .joblib template file (default: data/templates/global_templates.joblib)",
    )
    build_p.add_argument(
        "--chunk-segments",
        type=int,
        default=5000,
        help="Number of 60-s segments to load per ECG read (default: 5000)",
    )

    # ── correlate subcommand ──────────────────────────────────────────────────
    corr_p = subparsers.add_parser(
        "correlate",
        help="Compute per-beat template correlations for all peaks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    corr_p.add_argument(
        "--templates",
        type=str,
        required=True,
        help="Path to global_templates.joblib produced by the build subcommand",
    )
    corr_p.add_argument(
        "--peaks",
        type=str,
        required=True,
        help="Path to peaks.parquet",
    )
    corr_p.add_argument(
        "--ecg-samples",
        type=str,
        required=True,
        help="Path to ecg_samples.parquet",
    )
    corr_p.add_argument(
        "--output",
        type=str,
        default="data/processed/global_template_features.parquet",
        help="Output parquet path (default: data/processed/global_template_features.parquet)",
    )
    corr_p.add_argument(
        "--chunk-segments",
        type=int,
        default=5000,
        help="Number of 60-s segments to process per batch (default: 5000)",
    )

    # ── dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)

    if args.subcommand == "build":
        for label, path in [
            ("--peaks",       args.peaks),
            ("--labels",      args.labels),
            ("--ecg-samples", args.ecg_samples),
        ]:
            if not Path(path).exists():
                logger.error("File not found for %s: %s", label, path)
                sys.exit(1)
        build_templates(
            peaks_path=Path(args.peaks),
            labels_path=Path(args.labels),
            ecg_samples_path=Path(args.ecg_samples),
            output_path=Path(args.output),
            chunk_segments=args.chunk_segments,
        )

    elif args.subcommand == "correlate":
        for label, path in [
            ("--templates",   args.templates),
            ("--peaks",       args.peaks),
            ("--ecg-samples", args.ecg_samples),
        ]:
            if not Path(path).exists():
                logger.error("File not found for %s: %s", label, path)
                sys.exit(1)
        correlate_templates(
            templates_path=Path(args.templates),
            peaks_path=Path(args.peaks),
            ecg_samples_path=Path(args.ecg_samples),
            output_path=Path(args.output),
            chunk_segments=args.chunk_segments,
        )


if __name__ == "__main__":
    main()
