#!/usr/bin/env python3
"""
Backfill beat_features.segment_quality_pred from peaks + segment_quality_preds.

This repairs beat feature files generated before segment_quality_preds.parquet
existed. It streams beat_features.parquet, so the ECG/window feature work does
not need to be rerun just to fix this one column.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
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
logger = logging.getLogger("scripts.utils.backfill_segment_quality_pred")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill segment_quality_pred in beat_features.parquet.",
    )
    parser.add_argument("--beat-features", required=True, help="Input beat_features.parquet")
    parser.add_argument("--peaks", required=True, help="peaks.parquet with peak_id, segment_idx")
    parser.add_argument(
        "--segment-quality-preds",
        required=True,
        help="segment_quality_preds.parquet with segment_idx, quality_pred",
    )
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace --beat-features after writing a temporary file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output file",
    )
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the remapped distribution without writing output",
    )
    return parser.parse_args()


def _output_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    beat_features = Path(args.beat_features)
    if args.dry_run:
        return None, None
    if args.in_place:
        tmp = beat_features.with_name(f"{beat_features.name}.tmp.{os.getpid()}.parquet")
        return beat_features, tmp
    if not args.output:
        raise SystemExit("Use --output, --in-place, or --dry-run.")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite to replace: {output}")
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}.parquet")
    return output, tmp


def _load_maps(peaks_path: Path, preds_path: Path) -> tuple[pd.Series, pd.Series]:
    logger.info("Loading peak_id -> segment_idx from %s", peaks_path)
    peaks = pd.read_parquet(peaks_path, columns=["peak_id", "segment_idx"])
    if peaks["peak_id"].duplicated().any():
        n_dup = int(peaks["peak_id"].duplicated().sum())
        raise SystemExit(f"peaks.parquet contains {n_dup} duplicate peak_id values")
    peak_to_segment = pd.Series(
        peaks["segment_idx"].to_numpy(),
        index=peaks["peak_id"].to_numpy(),
        name="segment_idx",
    )
    logger.info("Loaded %d peak mappings", len(peak_to_segment))

    logger.info("Loading segment_idx -> quality_pred from %s", preds_path)
    preds = pd.read_parquet(preds_path, columns=["segment_idx", "quality_pred"])
    if preds["segment_idx"].duplicated().any():
        n_dup = int(preds["segment_idx"].duplicated().sum())
        logger.warning(
            "segment_quality_preds has %d duplicate segment_idx values; keeping last",
            n_dup,
        )
        preds = preds.drop_duplicates("segment_idx", keep="last")
    segment_to_pred = pd.Series(
        preds["quality_pred"].astype("int32").to_numpy(),
        index=preds["segment_idx"].to_numpy(),
        name="quality_pred",
    )
    logger.info(
        "Loaded %d segment predictions; distribution=%s",
        len(segment_to_pred),
        segment_to_pred.value_counts(dropna=False).sort_index().to_dict(),
    )
    return peak_to_segment, segment_to_pred


def _replace_or_append_column(table: pa.Table, values: np.ndarray) -> pa.Table:
    field = pa.field("segment_quality_pred", pa.int32())
    column = pa.array(values, type=pa.int32())
    if "segment_quality_pred" in table.schema.names:
        idx = table.schema.get_field_index("segment_quality_pred")
        return table.set_column(idx, field, column)
    return table.append_column(field, column)


def main() -> None:
    args = _parse_args()
    beat_features_path = Path(args.beat_features)
    output_path, tmp_path = _output_paths(args)

    peak_to_segment, segment_to_pred = _load_maps(
        Path(args.peaks),
        Path(args.segment_quality_preds),
    )

    pf = pq.ParquetFile(beat_features_path)
    if "peak_id" not in pf.schema_arrow.names:
        raise SystemExit("beat_features.parquet must contain a physical peak_id column")

    writer: pq.ParquetWriter | None = None
    counts: Counter[int] = Counter()
    n_rows = 0
    missing_peaks = 0
    missing_segments = 0

    try:
        for batch_num, batch in enumerate(
            pf.iter_batches(batch_size=args.batch_size),
            start=1,
        ):
            table = pa.Table.from_batches([batch], schema=pf.schema_arrow)
            peak_ids = table.column("peak_id").combine_chunks().to_numpy(zero_copy_only=False)

            segment_ids = peak_to_segment.reindex(peak_ids)
            missing_peaks += int(segment_ids.isna().sum())

            mapped = segment_to_pred.reindex(segment_ids.to_numpy())
            missing_segments += int(mapped.isna().sum())
            values = mapped.fillna(-1).astype("int32").to_numpy()

            unique_vals, unique_counts = np.unique(values, return_counts=True)
            counts.update(
                {int(k): int(v) for k, v in zip(unique_vals, unique_counts)}
            )
            n_rows += len(values)

            if not args.dry_run:
                out_table = _replace_or_append_column(table, values)
                if writer is None:
                    assert tmp_path is not None
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(tmp_path, out_table.schema, compression="snappy")
                writer.write_table(out_table)

            logger.info(
                "Processed batch %d: %d total rows, current distribution=%s",
                batch_num,
                n_rows,
                dict(sorted(counts.items())),
            )
    finally:
        if writer is not None:
            writer.close()

    logger.info("Final segment_quality_pred distribution=%s", dict(sorted(counts.items())))
    logger.info("Rows with missing peak mapping: %d", missing_peaks)
    logger.info("Rows with missing segment prediction: %d", missing_segments)

    if not args.dry_run:
        assert output_path is not None and tmp_path is not None
        os.replace(tmp_path, output_path)
        logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
