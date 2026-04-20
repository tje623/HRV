#!/usr/bin/env python3
"""
Detect and correct inverted ECG segments in ecg_samples.parquet.

A segment is considered inverted when the majority of detected R-peaks in that
segment have negative ECG amplitude — meaning the strap was worn upside down or
the signal polarity was flipped during recording.

Detection method: single streaming pass through ecg_samples.parquet, matching
each ECG sample to the known peak timestamps. For each segment, collects the ECG
amplitude at every R-peak position. If the median amplitude across those peaks is
below --inversion-threshold (default -0.05 mV), the segment is flagged as inverted.

Two modes:
    --report   (default) Print the list of inverted segments and exit.
    --fix      Correct the inverted segments (multiply ECG by -1) and write
               a corrected ecg_samples.parquet. Does NOT overwrite the
               original — writes to --output.

Usage
-----
    cd "/Volumes/xHRV/Artifact Detector"
    source /Users/tannereddy/.envs/hrv/bin/activate

    # Step 1: inspect what would be fixed
    python fix_inverted_ecg.py \\
        --ecg-samples /Volumes/xHRV/processed/ecg_samples.parquet \\
        --peaks       /Volumes/xHRV/processed/peaks.parquet \\
        --report

    # Step 2: write corrected parquet (original untouched)
    python fix_inverted_ecg.py \\
        --ecg-samples /Volumes/xHRV/processed/ecg_samples.parquet \\
        --peaks       /Volumes/xHRV/processed/peaks.parquet \\
        --fix \\
        --output      /Volumes/xHRV/processed/ecg_samples_fixed.parquet

    # Step 3: after verifying, swap the files:
    #   mv ecg_samples.parquet ecg_samples_original_backup.parquet
    #   mv ecg_samples_fixed.parquet ecg_samples.parquet
    # Then re-run the pipeline from beat_features.py onward.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# At 130 Hz one sample = 7,692,308 ns; allow ±1 sample.
PEAK_MATCH_TOLERANCE_NS = 8_000_000


def _detect_inverted_segments(
    ecg_samples_path: str,
    peaks_path: str,
    inversion_threshold: float,
    min_peaks: int,
) -> tuple[set[int], dict[int, float]]:
    """Return (inverted_segment_ids, {segment_idx: median_peak_amplitude}).

    Uses a single streaming pass through ecg_samples.parquet — no per-segment
    random seeks.

    Parameters
    ----------
    inversion_threshold : float
        Median ECG amplitude at detected peaks below this value → inverted.
    min_peaks : int
        Minimum number of matched peaks per segment to make a decision.
        Segments with fewer matches are skipped.
    """
    # ── Load all peak timestamps ───────────────────────────────────────────
    log.info("Loading peaks …")
    peaks_df = pd.read_parquet(peaks_path, columns=["peak_id", "timestamp_ns", "segment_idx"])
    if "timestamp_ns" not in peaks_df.columns:
        # Fallback: peak_id stores epoch ms
        peaks_df["timestamp_ns"] = peaks_df["peak_id"].astype(np.int64) * 1_000_000
    peaks_df["timestamp_ns"] = peaks_df["timestamp_ns"].astype(np.int64)
    peaks_df["segment_idx"]  = peaks_df["segment_idx"].astype(np.int32)

    # {segment_idx → sorted array of peak timestamps in ns}
    peak_ts_by_seg: dict[int, np.ndarray] = {}
    for seg_idx, grp in peaks_df.groupby("segment_idx"):
        peak_ts_by_seg[int(seg_idx)] = np.sort(
            grp["timestamp_ns"].values.astype(np.int64)
        )
    n_segs = len(peak_ts_by_seg)
    log.info("  %d peaks across %d segments", len(peaks_df), n_segs)
    del peaks_df

    # ── Single streaming pass: collect ECG amplitude at each peak ─────────
    # seg_amps[segment_idx] = list of ECG values sampled at peak positions
    seg_amps: dict[int, list[float]] = {s: [] for s in peak_ts_by_seg}

    pf = pq.ParquetFile(ecg_samples_path)
    total_rows = pf.metadata.num_rows
    approx_batches = max(1, total_rows // 500_000)
    log.info("Streaming %s rows (~%d batches) …",
             f"{total_rows:,}", approx_batches)

    for batch_idx, batch in enumerate(
        pf.iter_batches(
            batch_size=500_000,
            columns=["timestamp_ns", "ecg", "segment_idx"],
        )
    ):
        if batch_idx % 20 == 0:
            log.info("  batch %d / ~%d", batch_idx, approx_batches)

        df = batch.to_pandas()
        df["timestamp_ns"] = df["timestamp_ns"].astype(np.int64)
        df["ecg"]          = df["ecg"].astype(np.float32)
        df["segment_idx"]  = df["segment_idx"].astype(np.int32)

        for seg_val, seg_grp in df.groupby("segment_idx", sort=False):
            seg_idx = int(seg_val)
            if seg_idx not in peak_ts_by_seg:
                continue

            peak_ts = peak_ts_by_seg[seg_idx]   # all peaks for this segment
            ecg_ts  = seg_grp["timestamp_ns"].values
            ecg_val = seg_grp["ecg"].values

            # Sort ECG samples by timestamp (batch order may not be sorted)
            order  = np.argsort(ecg_ts, kind="stable")
            ecg_ts  = ecg_ts[order]
            ecg_val = ecg_val[order]

            batch_lo = ecg_ts[0]
            batch_hi = ecg_ts[-1]

            for pts in peak_ts:
                # Fast skip: peak is far outside the time range of this batch slice
                if pts < batch_lo - PEAK_MATCH_TOLERANCE_NS:
                    continue
                if pts > batch_hi + PEAK_MATCH_TOLERANCE_NS:
                    continue

                pos = int(np.searchsorted(ecg_ts, pts))
                best_dist = PEAK_MATCH_TOLERANCE_NS + 1
                best_amp  = 0.0
                for p in (pos - 1, pos):
                    if 0 <= p < len(ecg_ts):
                        d = abs(int(ecg_ts[p]) - int(pts))
                        if d < best_dist:
                            best_dist = d
                            best_amp  = float(ecg_val[p])
                if best_dist <= PEAK_MATCH_TOLERANCE_NS:
                    seg_amps[seg_idx].append(best_amp)

    # ── Compute per-segment medians ────────────────────────────────────────
    log.info("Computing per-segment medians …")
    inverted:   set[int]        = set()
    seg_median: dict[int, float] = {}
    for seg_idx, amps in seg_amps.items():
        if len(amps) < min_peaks:
            continue
        med = float(np.median(amps))
        seg_median[seg_idx] = med
        if med < inversion_threshold:
            inverted.add(seg_idx)

    log.info(
        "  %d segments with ≥%d matched peaks  |  %d inverted (%.2f%%)",
        len(seg_median),
        min_peaks,
        len(inverted),
        100.0 * len(inverted) / max(len(seg_median), 1),
    )
    return inverted, seg_median


def _fix_and_write(
    ecg_samples_path: str,
    inverted_segments: set[int],
    output_path: str,
) -> None:
    """Stream ecg_samples.parquet, flip ECG sign for inverted segments, write output."""
    log.info("Writing corrected parquet → %s", output_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(ecg_samples_path)
    writer:      pq.ParquetWriter | None = None
    n_fixed_rows = 0

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=500_000)):
        df = batch.to_pandas()

        mask = df["segment_idx"].isin(inverted_segments)
        if mask.any():
            df.loc[mask, "ecg"] = df.loc[mask, "ecg"] * -1.0
            n_fixed_rows += int(mask.sum())

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)

        if batch_idx % 20 == 0:
            log.info("  write batch %d  (%d rows corrected so far)",
                     batch_idx, n_fixed_rows)

    if writer is not None:
        writer.close()

    log.info(
        "Done. %d rows flipped across %d segments.",
        n_fixed_rows,
        len(inverted_segments),
    )


def main() -> None:
    args = _build_parser().parse_args()

    if not args.report and not args.fix:
        log.error("Specify --report (inspect) or --fix (write corrected parquet).")
        sys.exit(1)

    inverted, seg_median = _detect_inverted_segments(
        ecg_samples_path=args.ecg_samples,
        peaks_path=args.peaks,
        inversion_threshold=args.inversion_threshold,
        min_peaks=args.min_peaks,
    )

    # ── Report ────────────────────────────────────────────────────────────
    n_checked = len(seg_median)
    print(f"\n{'=' * 72}")
    print("  ECG Inversion Detection Report")
    print(f"{'=' * 72}")
    print(f"  Segments checked:    {n_checked:,}")
    print(f"  Inverted segments:   {len(inverted):,}  "
          f"({100.0 * len(inverted) / max(n_checked, 1):.2f}% of checked)")
    print(f"  Inversion threshold: {args.inversion_threshold} mV "
          f"(median ECG at R-peaks below this → inverted)")

    if inverted:
        amps = [seg_median[s] for s in sorted(inverted)]
        print(f"\n  Inverted segment_idx list (median peak amplitude):")
        for seg_idx in sorted(inverted):
            print(f"    segment {seg_idx:>8d}   median_peak_ecg = {seg_median[seg_idx]:+.4f} mV")
        print(f"\n  Most negative: {min(amps):+.4f} mV  |  "
              f"least negative: {max(amps):+.4f} mV")

    print(f"{'=' * 72}")

    if args.fix:
        if not inverted:
            log.info("No inverted segments found — nothing to fix.")
            return
        if not args.output:
            log.error("--fix requires --output path for the corrected parquet.")
            sys.exit(1)
        _fix_and_write(args.ecg_samples, inverted, args.output)
        orig = Path(args.ecg_samples)
        print(f"\n  Corrected parquet written → {args.output}")
        print(f"  Original untouched:        {args.ecg_samples}")
        print(f"\n  To swap in the fix:")
        print(f'    mv "{orig}" '
              f'"{orig.parent}/ecg_samples_original_backup.parquet"')
        print(f'    mv "{args.output}" "{orig}"')
        print(f"\n  Then re-run the pipeline from beat_features.py onward.")
        print(f"{'=' * 72}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fix_inverted_ecg.py",
        description="Detect and correct inverted ECG segments in ecg_samples.parquet.",
    )
    p.add_argument("--ecg-samples", required=True,
                   help="Path to ecg_samples.parquet")
    p.add_argument("--peaks", required=True,
                   help="Path to peaks.parquet")
    p.add_argument("--report", action="store_true",
                   help="Print inverted segments and exit (no changes written)")
    p.add_argument("--fix", action="store_true",
                   help="Write corrected ecg_samples.parquet to --output")
    p.add_argument("--output", default=None,
                   help="Output path for corrected parquet (required with --fix)")
    p.add_argument(
        "--inversion-threshold", type=float, default=-0.05,
        help="Median ECG at R-peaks below this → inverted (default: -0.05 mV)",
    )
    p.add_argument(
        "--min-peaks", type=int, default=3,
        help="Minimum matched peaks per segment to make a decision (default: 3)",
    )
    return p


if __name__ == "__main__":
    main()
