#!/usr/bin/env python3
"""
scripts/merge_features_for_training.py — Merge beat features with global
template correlation for LightGBM retraining.

Left-joins beat_features_v2.parquet (40 feature columns, indexed by peak_id)
with global_template_features.parquet (peak_id, global_corr_clean) to produce
beat_features_merged.parquet — the file the retrained tabular model consumes.

The tabular model (beat_artifact_tabular.py line 311) auto-discovers all
non-peak_id columns as features, so the new global_corr_clean column is picked
up automatically with no changes to that script.

Invariants enforced:
  • Output row count == input beat_features row count (left join, no drops)
  • global_corr_clean dtype is float32
  • Missing correlations are filled with 0.0 (logged as WARNING)
  • peak_id remains the parquet index

Usage
-----
    cd "/Volumes/xHRV/Artifact Detector"
    source /Users/tannereddy/.envs/hrv/bin/activate

    python scripts/merge_features_for_training.py \\
        --beat-features            data/processed/beat_features_v2.parquet \\
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
) -> None:
    """Merge beat features with global template correlation and write parquet.

    Args:
        beat_features_path: beat_features_v2.parquet — indexed by peak_id.
        global_template_features_path: global_template_features.parquet —
            columns (peak_id, global_corr_clean).
        output_path: Destination parquet path.
    """

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading beat features from %s", beat_features_path)
    bf = pd.read_parquet(beat_features_path)

    if bf.index.name != "peak_id":
        if "peak_id" in bf.columns:
            bf = bf.set_index("peak_id")
        else:
            logger.error(
                "beat_features has neither peak_id index nor peak_id column — cannot merge"
            )
            sys.exit(1)

    n_input = len(bf)
    logger.info(
        "beat_features: %d rows × %d columns", n_input, len(bf.columns)
    )

    logger.info(
        "Loading global template features from %s", global_template_features_path
    )
    gtf = pd.read_parquet(
        global_template_features_path,
        columns=["peak_id", "global_corr_clean"],
    )
    logger.info("global_template_features: %d rows", len(gtf))

    # ── Left join on peak_id ──────────────────────────────────────────────────
    # Reset index so peak_id is a plain column, merge, then restore.
    merged = bf.reset_index().merge(
        gtf,
        on="peak_id",
        how="left",
    ).set_index("peak_id")

    # ── Invariant: row count must not change ──────────────────────────────────
    if len(merged) != n_input:
        logger.error(
            "Row count changed after merge: %d → %d — aborting",
            n_input, len(merged),
        )
        sys.exit(1)

    # ── Fill missing correlations ─────────────────────────────────────────────
    n_missing = int(merged["global_corr_clean"].isna().sum())
    if n_missing > 0:
        logger.warning(
            "%d peak_id(s) in beat_features have no global_corr_clean entry — "
            "filling with 0.0",
            n_missing,
        )
        merged["global_corr_clean"] = merged["global_corr_clean"].fillna(0.0)

    # ── Enforce float32 ───────────────────────────────────────────────────────
    merged["global_corr_clean"] = merged["global_corr_clean"].astype(np.float32)

    # ── Write ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(merged, preserve_index=True)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Saved → %s", output_path)

    # ── Report ────────────────────────────────────────────────────────────────
    gc = merged["global_corr_clean"].values
    print()
    print("═" * 60)
    print("  MERGE COMPLETE")
    print("═" * 60)
    print(f"  Rows     : {len(merged):,}  (input: {n_input:,}  ✓ unchanged)")
    print(f"  Columns  : {len(merged.columns)}  (was {len(bf.columns)}, +1 global_corr_clean)")
    print(f"  Missing  : {n_missing} filled with 0.0")
    print()
    print(f"  global_corr_clean  (float32)")
    print(f"    min    : {gc.min():.4f}")
    print(f"    mean   : {gc.mean():.4f}")
    print(f"    median : {float(np.median(gc)):.4f}")
    print(f"    max    : {gc.max():.4f}")
    print()
    print(f"  Output   : {output_path}")
    print("═" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Left-join beat features with global template correlation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--beat-features",
        type=str,
        required=True,
        help="Path to beat_features_v2.parquet",
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

    args = parser.parse_args()

    for flag, path in [
        ("--beat-features",            args.beat_features),
        ("--global-template-features", args.global_template_features),
    ]:
        if not Path(path).exists():
            logger.error("File not found for %s: %s", flag, path)
            sys.exit(1)

    merge_features(
        beat_features_path=Path(args.beat_features),
        global_template_features_path=Path(args.global_template_features),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
