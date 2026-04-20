#!/usr/bin/env python3
"""
scripts/merge_features_for_training.py — Merge beat features with global
template correlation for LightGBM retraining.

Streams beat_features.parquet in batches (~80 MB each) to avoid loading
the full dataset (54 M rows × 40+ cols ≈ 10-15 GB) into RAM at once.

The global_template_features lookup (~700 MB) is loaded once into a
pd.Series and used for vectorised per-batch reindex lookups.  Each
enriched batch is written incrementally via pq.ParquetWriter.

Peak memory: ~700 MB (lookup) + ~80 MB (one batch) + ~80 MB (output batch).

Invariants enforced:
  • Output row count == input beat_features row count (streaming left join)
  • global_corr_clean dtype is float32
  • Missing correlations are filled with 0.0 (logged as WARNING)
  • peak_id column is preserved in the output

Usage
-----
    cd "/Volumes/xHRV/Artifact Detector"
    source /Users/tannereddy/.envs/hrv/bin/activate

    python scripts/merge_features_for_training.py \\
        --beat-features            /Volumes/xHRV/processed/beat_features.parquet \\
        --global-template-features /Volumes/xHRV/processed/global_template_features.parquet \\
        --output                   /Volumes/xHRV/processed/beat_features_merged.parquet

    # Local training subset (also fine — small enough to load fully, but
    # this script streams either way)
    python scripts/merge_features_for_training.py \\
        --beat-features            data/processed/beat_features.parquet \\
        --global-template-features data/processed/global_template_features.parquet \\
        --output                   data/processed/beat_features_merged.parquet
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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.merge_features_for_training")


def merge_features(
    beat_features_path: Path,
    global_template_features_path: Path,
    output_path: Path,
    batch_size: int = 500_000,
) -> None:
    """Left-join beat features with global template correlation.

    Streams beat_features in batches of `batch_size` rows.  The
    global_template_features lookup is loaded once and held in a
    pd.Series for O(n) vectorised reindex per batch.

    Args:
        beat_features_path: beat_features.parquet — must contain peak_id
            (either as a column or as the pandas index stored by
            preserve_index=True).
        global_template_features_path: global_template_features.parquet —
            columns (peak_id, global_corr_clean).
        output_path: Destination parquet path.
        batch_size: Rows per streaming batch.  Default 500 000 (≈ 80 MB
            at 40 float32 columns).
    """

    # ── Load lookup (global_template_features is small — 2 cols × 54 M rows
    #    ≈ 700 MB; negligible next to beat_features if fully loaded) ──────────
    logger.info(
        "Loading global template features from %s",
        global_template_features_path,
    )
    gtf = pq.read_table(
        global_template_features_path,
        columns=["peak_id", "global_corr_clean"],
    ).to_pandas()
    logger.info(
        "global_template_features: %d rows — building lookup ...", len(gtf)
    )

    global_corr_lookup = pd.Series(
        gtf["global_corr_clean"].values.astype(np.float32),
        index=gtf["peak_id"].values,
    )
    del gtf

    # Deduplicate index (should not happen, but be defensive)
    if not global_corr_lookup.index.is_unique:
        n_dup = int(global_corr_lookup.index.duplicated().sum())
        logger.warning(
            "global_template_features has %d duplicate peak_ids — keeping last",
            n_dup,
        )
        global_corr_lookup = global_corr_lookup[
            ~global_corr_lookup.index.duplicated(keep="last")
        ]

    # ── Inspect input schema (no data loaded) ─────────────────────────────
    pf = pq.ParquetFile(beat_features_path)
    in_schema = pf.schema_arrow
    n_input_rows = pf.metadata.num_rows
    n_input_cols = len(in_schema)

    if "peak_id" not in in_schema.names:
        logger.error(
            "peak_id column not found in beat_features schema.  "
            "Available columns: %s",
            in_schema.names,
        )
        sys.exit(1)

    logger.info(
        "beat_features: %d rows × %d columns — streaming in batches of %d",
        n_input_rows,
        n_input_cols,
        batch_size,
    )

    # ── Build output schema: all input fields + global_corr_clean ──────────
    # Drop any existing global_corr_clean so we always write a fresh value.
    out_fields = [f for f in in_schema if f.name != "global_corr_clean"]
    out_fields.append(pa.field("global_corr_clean", pa.float32()))
    out_schema = pa.schema(out_fields)

    # ── Stream → join → write ─────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_missing = 0
    gc_min = np.inf
    gc_max = -np.inf
    gc_sum = 0.0

    with pq.ParquetWriter(output_path, out_schema, compression="snappy") as writer:
        for batch in pf.iter_batches(batch_size=batch_size):
            tbl = pa.Table.from_batches([batch])

            # peak_id as numpy int64 (stored as a regular column in parquet)
            peak_id_arr = (
                tbl.column("peak_id")
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
            )

            # Vectorised lookup — NaN where peak_id not found
            corr_series = global_corr_lookup.reindex(peak_id_arr)
            missing_mask = corr_series.isna().values
            n_missing += int(missing_mask.sum())
            corr_vals = corr_series.fillna(0.0).values.astype(np.float32)

            # Running statistics (no need to accumulate all values)
            if len(corr_vals) > 0:
                gc_min = min(gc_min, float(corr_vals.min()))
                gc_max = max(gc_max, float(corr_vals.max()))
                gc_sum += float(corr_vals.sum())

            # Remove stale global_corr_clean (if input already had it)
            if "global_corr_clean" in tbl.schema.names:
                tbl = tbl.remove_column(
                    tbl.schema.get_field_index("global_corr_clean")
                )

            # Append new global_corr_clean column
            tbl = tbl.append_column(
                pa.field("global_corr_clean", pa.float32()),
                pa.array(corr_vals, type=pa.float32()),
            )

            # Write with exact output schema (enforces column order, strips
            # any stray pandas metadata from the original file)
            writer.write_table(
                pa.table(
                    {name: tbl.column(name) for name in out_schema.names},
                    schema=out_schema,
                )
            )

            n_processed += len(batch)
            logger.info(
                "  %d / %d rows  (%.0f%%)",
                n_processed,
                n_input_rows,
                100.0 * n_processed / max(n_input_rows, 1),
            )

    # ── Verify ────────────────────────────────────────────────────────────
    if n_processed != n_input_rows:
        logger.warning(
            "Row count mismatch: processed %d, expected %d",
            n_processed,
            n_input_rows,
        )

    if n_missing > 0:
        logger.warning(
            "%d peak_id(s) had no global_corr_clean entry — filled with 0.0",
            n_missing,
        )

    # ── Report ────────────────────────────────────────────────────────────
    gc_mean = gc_sum / max(n_processed, 1)

    print()
    print("═" * 60)
    print("  MERGE COMPLETE")
    print("═" * 60)
    print(f"  Rows     : {n_processed:,}  (input: {n_input_rows:,}  ✓ unchanged)")
    print(f"  Columns  : {len(out_schema)}  (was {n_input_cols}, +1 global_corr_clean)")
    print(f"  Missing  : {n_missing} filled with 0.0")
    print()
    print("  global_corr_clean  (float32)")
    print(f"    min    : {gc_min:.4f}")
    print(f"    mean   : {gc_mean:.4f}")
    print(f"    max    : {gc_max:.4f}")
    print()
    print(f"  Output   : {output_path}")
    print("═" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Left-join beat features with global template correlation "
            "(streaming — safe for full 80 GB dataset)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--beat-features",
        type=str,
        required=True,
        help="Path to beat_features.parquet",
    )
    parser.add_argument(
        "--global-template-features",
        type=str,
        required=True,
        help="Path to global_template_features.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/beat_features_merged.parquet",
        help="Output path (default: data/processed/beat_features_merged.parquet)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500_000,
        help=(
            "Rows per streaming batch (default: 500 000 ≈ 80 MB/batch at "
            "40 float32 columns).  Decrease if RAM is tight."
        ),
    )

    args = parser.parse_args()

    for flag, path in [
        ("--beat-features", args.beat_features),
        ("--global-template-features", args.global_template_features),
    ]:
        if not Path(path).exists():
            logger.error("File not found for %s: %s", flag, path)
            sys.exit(1)

    merge_features(
        beat_features_path=Path(args.beat_features),
        global_template_features_path=Path(args.global_template_features),
        output_path=Path(args.output),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
