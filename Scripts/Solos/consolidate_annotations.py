#!/usr/bin/env python3
"""
scripts/consolidate_annotations.py — Filter and consolidate annotation JSON.

Reads artifact_annotations_master_edited.json, applies trust rules:
  effective_validated = validated_segments − bad_segments − return_to_pile

Normalises key names (interpolate_peaks → flagged_for_interpolation, etc.),
merges bad_regions + bad_regions_within_segments, and writes a clean
artifact_annotations_final.json ready for data_pipeline.py.

Usage:
    python scripts/consolidate_annotations.py \\
        --input  /Volumes/xHRV/historical/artifact_annotations_master_edited.json \\
        --output /Volumes/xHRV/data/annotations/artifact_annotations_final.json \\
        [--peaks-parquet data/processed/peaks.parquet]
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SEGMENT_DURATION_MS, MIN_VALID_TIMESTAMP_MS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("consolidate_annotations")



# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_seg_str(s: object) -> int | None:
    """Parse 'seg_N' string or bare integer to annotation segment index."""
    if isinstance(s, (int, float)) and not (isinstance(s, float) and np.isnan(s)):
        return int(s)
    if isinstance(s, str):
        try:
            return int(s.replace("seg_", ""))
        except ValueError:
            return None
    return None


def _parse_timestamp_ms(val: object) -> int | None:
    """Convert epoch-ms int, ISO string, or numeric string → epoch milliseconds."""
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float):
        if np.isnan(val):
            return None
        return int(val)
    if isinstance(val, str):
        try:
            return int(pd.Timestamp(val).value) // 1_000_000
        except Exception:
            try:
                return int(float(val))
            except Exception:
                return None
    return None


def _get_recording_start_ms(peaks_parquet: Path) -> int:
    if not peaks_parquet.exists():
        logger.error(
            "peaks.parquet not found at %s — run data_pipeline.py first "
            "or specify --peaks-parquet",
            peaks_parquet,
        )
        sys.exit(1)
    df = pd.read_parquet(peaks_parquet, columns=["timestamp_ms"])
    val = int(df["timestamp_ms"].min())
    logger.info("Recording start: %s", pd.Timestamp(val, unit="ms"))
    return val


def _build_ann_to_pipeline(
    segment_timestamps: list,
    recording_start_ms: int,
) -> dict[int, int]:
    """Map annotation segment index → pipeline segment_idx via first_timestamp."""
    mapping: dict[int, int] = {}
    for entry in segment_timestamps:
        ann_idx = entry.get("segment_idx")
        ts_str = entry.get("first_timestamp")
        if ann_idx is None or ts_str is None:
            continue
        try:
            ts_ms = int(pd.Timestamp(ts_str).value) // 1_000_000
            pipeline_idx = (ts_ms - recording_start_ms) // SEGMENT_DURATION_MS
            mapping[int(ann_idx)] = int(pipeline_idx)
        except Exception:
            logger.warning("Could not parse segment_timestamps entry: %r", entry)
    return mapping


class _SegmentIntervalLookup:
    """Interval-based lookup: timestamp → annotation segment index.

    Each annotation segment covers [first_ns, last_ns].  Artifacts and other
    annotation timestamps may fall in the *second* pipeline-segment window of
    an annotation segment (because annotation boundaries and pipeline boundaries
    are offset), so a pure pipeline-idx→ann-idx dict misses them.  This class
    uses binary search on sorted first_ns values and a fallback last_ns check.
    """

    def __init__(self, segment_timestamps: list) -> None:
        intervals: list[tuple[int, int, int]] = []  # (first_ms, last_ms, ann_idx)
        for entry in segment_timestamps:
            ann_idx = entry.get("segment_idx")
            ft_str = entry.get("first_timestamp")
            lt_str = entry.get("last_timestamp")
            if ann_idx is None or ft_str is None:
                continue
            try:
                ft_ms = int(pd.Timestamp(ft_str).value) // 1_000_000
                lt_ms = (
                    int(pd.Timestamp(lt_str).value) // 1_000_000
                    if lt_str
                    else ft_ms + SEGMENT_DURATION_MS
                )
                intervals.append((ft_ms, lt_ms, int(ann_idx)))
            except Exception:
                pass
        intervals.sort(key=lambda x: x[0])
        self._first_ms = [iv[0] for iv in intervals]
        self._last_ms  = [iv[1] for iv in intervals]
        self._ann_idx  = [iv[2] for iv in intervals]

    def lookup(self, ts_ms: int) -> int | None:
        """Return annotation segment index for ts_ms, or None if unmatched."""
        # Candidate: the segment whose first_ms is ≤ ts_ms (search right boundary)
        pos = bisect.bisect_right(self._first_ms, ts_ms) - 1
        if pos < 0:
            return None
        # Check the candidate and its right neighbour (timestamp might land in
        # the overlap zone between two adjacent annotation segments)
        for idx in (pos, pos + 1):
            if 0 <= idx < len(self._ann_idx):
                if self._first_ms[idx] <= ts_ms <= self._last_ms[idx]:
                    return self._ann_idx[idx]
        return None


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate annotation JSON for pipeline consumption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to artifact_annotations_master_edited.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination path for artifact_annotations_final.json")
    parser.add_argument(
        "--peaks-parquet",
        type=Path,
        default=Path("data/processed/peaks.parquet"),
        help="Path to processed peaks.parquet (for recording_start_ns, "
             "used for bad_region coordinate translation only)",
    )
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    logger.info("Loading %s ...", args.input)
    with open(args.input, "r") as f:
        data = json.load(f)

    # ── Recording start (for bad_region pipeline coordinate translation) ──────
    recording_start_ms = _get_recording_start_ms(args.peaks_parquet)

    segment_timestamps = data.get("segment_timestamps", [])

    # ── Interval-based timestamp → annotation segment lookup ──────────────────
    seg_lookup = _SegmentIntervalLookup(segment_timestamps)

    # ── Build ann→pipeline map (only needed for bad_region translation) ───────
    ann_to_pipeline = _build_ann_to_pipeline(segment_timestamps, recording_start_ms)
    logger.info("Segment map: %d annotation ↔ pipeline entries", len(ann_to_pipeline))

    # ── Extract sets of annotation indices ────────────────────────────────────
    validated: set[int] = set()
    for s in data.get("validated_segments", []):
        idx = _parse_seg_str(s)
        if idx is not None:
            validated.add(idx)

    bad_segs: set[int] = set()
    for s in data.get("bad_segments", []):
        idx = _parse_seg_str(s)
        if idx is not None:
            bad_segs.add(idx)

    revisit: set[int] = set()
    for entry in data.get("return_to_pile", []):
        if isinstance(entry, dict):
            idx = entry.get("segment_idx")
            if idx is not None:
                revisit.add(int(idx))
        else:
            idx = _parse_seg_str(entry)
            if idx is not None:
                revisit.add(idx)

    effective_validated: set[int] = validated - bad_segs - revisit
    logger.info(
        "Segments: validated=%d  bad=%d  revisit=%d  → effective=%d",
        len(validated), len(bad_segs), len(revisit), len(effective_validated),
    )

    # ── Merge and index bad_regions (both sources) ────────────────────────────
    # bad_regions (5 entries) + bad_regions_within_segments (54 entries), no overlap.
    all_bad_regions_raw: list = (
        data.get("bad_regions", []) + data.get("bad_regions_within_segments", [])
    )
    # bad_region_intervals: pipeline_seg_idx → [(start_ns, end_ns)]
    bad_region_intervals: dict[int, list[tuple[int, int]]] = {}
    n_bad_region_skipped = 0
    for region in all_bad_regions_raw:
        if not isinstance(region, dict):
            continue
        ann_idx = region.get("segment_idx")
        start_str = region.get("start_time")
        end_str = region.get("end_time")
        if ann_idx is None or start_str is None or end_str is None:
            logger.warning("Malformed bad_region entry (skipped): %r", region)
            n_bad_region_skipped += 1
            continue
        pipeline_idx = ann_to_pipeline.get(int(ann_idx))
        if pipeline_idx is None:
            logger.warning("bad_region ann_idx=%d has no pipeline mapping (skipped)", ann_idx)
            n_bad_region_skipped += 1
            continue
        try:
            start_ms = int(pd.Timestamp(start_str).value) // 1_000_000
            end_ms = int(pd.Timestamp(end_str).value) // 1_000_000
        except Exception:
            logger.warning("Could not parse bad_region timestamps (skipped): %r", region)
            n_bad_region_skipped += 1
            continue
        bad_region_intervals.setdefault(pipeline_idx, []).append((start_ms, end_ms))

    logger.info(
        "Bad-region intervals: %d windows across %d segments (%d skipped)",
        sum(len(v) for v in bad_region_intervals.values()),
        len(bad_region_intervals),
        n_bad_region_skipped,
    )

    def _in_bad_region_by_ts(ts_ms: int) -> bool:
        """Check whether ts_ms falls inside any bad_region window."""
        # Derive pipeline seg from timestamp to look up intervals
        pipeline_seg = (ts_ms - recording_start_ms) // SEGMENT_DURATION_MS
        for s, e in bad_region_intervals.get(pipeline_seg, []):
            if s <= ts_ms <= e:
                return True
        # Also check adjacent pipeline seg (timestamp near boundary)
        for adjacent in (pipeline_seg - 1, pipeline_seg + 1):
            for s, e in bad_region_intervals.get(adjacent, []):
                if s <= ts_ms <= e:
                    return True
        return False

    # Drop-reason categoriser — uses interval lookup for annotation segment.
    def _drop_reason(
        ts_ms: int,
        check_bad_region: bool,
    ) -> str:
        ann_seg = seg_lookup.lookup(ts_ms)
        if ann_seg is None or ann_seg not in validated:
            return "not_validated"
        if ann_seg in bad_segs:
            return "in_bad_seg"
        if ann_seg in revisit:
            return "in_revisit"
        if check_bad_region and _in_bad_region_by_ts(ts_ms):
            return "in_bad_region"
        return "keep"

    # ── Filter artifacts (with bad_region exclusion) ──────────────────────────
    artifacts_raw = data.get("artifacts", [])
    artifacts_filtered = []
    drop_art = {"in_bad_seg": 0, "in_revisit": 0, "in_bad_region": 0, "not_validated": 0}
    for entry in artifacts_raw:
        ts_ms = _parse_timestamp_ms(entry)
        if ts_ms is None:
            drop_art["not_validated"] += 1
            continue
        reason = _drop_reason(ts_ms, check_bad_region=True)
        if reason == "keep":
            artifacts_filtered.append(entry)
        else:
            drop_art[reason] += 1

    # ── Filter added_r_peaks ──────────────────────────────────────────────────
    added_raw = data.get("added_r_peaks", [])
    added_filtered = []
    drop_added = {"in_bad_seg": 0, "in_revisit": 0, "in_bad_region": 0, "not_validated": 0}
    for entry in added_raw:
        ts_ms = _parse_timestamp_ms(entry)
        if ts_ms is None:
            drop_added["not_validated"] += 1
            continue
        reason = _drop_reason(ts_ms, check_bad_region=False)
        if reason == "keep":
            added_filtered.append(entry)
        else:
            drop_added[reason] += 1

    # ── Filter flagged_for_interpolation (GUI key: interpolate_peaks) ─────────
    interp_raw = data.get("interpolate_peaks", data.get("flagged_for_interpolation", []))
    interp_filtered = []
    drop_interp = {"in_bad_seg": 0, "in_revisit": 0, "in_bad_region": 0, "not_validated": 0}
    for entry in interp_raw:
        ts_ms = _parse_timestamp_ms(entry)
        if ts_ms is None:
            drop_interp["not_validated"] += 1
            continue
        reason = _drop_reason(ts_ms, check_bad_region=False)
        if reason == "keep":
            interp_filtered.append(entry)
        else:
            drop_interp[reason] += 1

    # ── Filter tagged_physiological_events (GUI key: physiological_events) ────
    physio_raw = data.get("physiological_events", data.get("tagged_physiological_events", []))
    physio_filtered = []
    drop_physio = {"in_bad_seg": 0, "in_revisit": 0, "in_bad_region": 0, "not_validated": 0}
    for entry in physio_raw:
        if isinstance(entry, dict):
            ts_val = (
                entry.get("start")
                or entry.get("start_time")
                or entry.get("timestamp")
            )
            ts_ms = _parse_timestamp_ms(ts_val)
        else:
            ts_ms = _parse_timestamp_ms(entry)
        if ts_ms is None:
            drop_physio["not_validated"] += 1
            continue
        reason = _drop_reason(ts_ms, check_bad_region=False)
        if reason == "keep":
            physio_filtered.append(entry)
        else:
            drop_physio[reason] += 1

    # ── Filter interpolated_replacements ──────────────────────────────────────
    interp_rep_raw = data.get("interpolated_replacements", [])
    interp_rep_filtered = []
    drop_irep = {"in_bad_seg": 0, "in_revisit": 0, "in_bad_region": 0, "not_validated": 0}
    for item in interp_rep_raw:
        if not isinstance(item, dict):
            continue
        # Try timestamp first (ISO), then peak_id (epoch ms)
        ts_val = item.get("timestamp") or item.get("peak_id")
        ts_ms = _parse_timestamp_ms(ts_val)
        if ts_ms is None:
            drop_irep["not_validated"] += 1
            continue
        reason = _drop_reason(ts_ms, check_bad_region=False)
        if reason == "keep":
            interp_rep_filtered.append(item)
        else:
            drop_irep[reason] += 1

    # ── Build output validated_segments (annotation index strings) ────────────
    eff_val_strings = [f"seg_{idx}" for idx in sorted(effective_validated)]

    # ── Build unified bad_regions for output (strip GUI-only fields) ──────────
    unified_bad_regions = []
    for region in all_bad_regions_raw:
        if isinstance(region, dict) and "segment_idx" in region:
            unified_bad_regions.append({
                "segment_idx": region["segment_idx"],
                "start_time":  region["start_time"],
                "end_time":    region["end_time"],
            })

    # ── Consolidation stats ───────────────────────────────────────────────────
    consolidation_stats = {
        "artifacts": {
            "before": len(artifacts_raw),
            "after": len(artifacts_filtered),
            "dropped": drop_art,
        },
        "added_r_peaks": {
            "before": len(added_raw),
            "after": len(added_filtered),
            "dropped": drop_added,
        },
        "flagged_for_interpolation": {
            "before": len(interp_raw),
            "after": len(interp_filtered),
            "dropped": drop_interp,
        },
        "tagged_physiological_events": {
            "before": len(physio_raw),
            "after": len(physio_filtered),
            "dropped": drop_physio,
        },
        "interpolated_replacements": {
            "before": len(interp_rep_raw),
            "after": len(interp_rep_filtered),
            "dropped": drop_irep,
        },
        "effective_validated_segments": len(effective_validated),
        "bad_region_windows_total": sum(len(v) for v in bad_region_intervals.values()),
    }

    # ── Build output JSON ─────────────────────────────────────────────────────
    output = {
        # Filtered annotation arrays (pipeline-expected key names)
        "artifacts":                    artifacts_filtered,
        "added_r_peaks":                added_filtered,
        "flagged_for_interpolation":    interp_filtered,
        "tagged_physiological_events":  physio_filtered,
        "interpolated_replacements":    interp_rep_filtered,
        # Filtered validated set
        "validated_segments":           eff_val_strings,
        # Preserved originals (used downstream)
        "bad_segments":                 data.get("bad_segments", []),
        "bad_regions":                  unified_bad_regions,
        "return_to_pile":               data.get("return_to_pile", []),
        # Coordinate translation (required by pipeline)
        "segment_timestamps":           segment_timestamps,
        # Audit trail
        "_consolidation_stats":         consolidation_stats,
    }

    # ── Write ─────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote → %s", args.output)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  Annotation Consolidation Summary")
    print(f"{'=' * 72}")
    for key in (
        "artifacts",
        "added_r_peaks",
        "flagged_for_interpolation",
        "tagged_physiological_events",
        "interpolated_replacements",
    ):
        s = consolidation_stats[key]
        b, a = s["before"], s["after"]
        d = s.get("dropped", {})
        drop_parts = [f"{k}={v}" for k, v in d.items() if v > 0]
        drop_str = "  ".join(drop_parts) if drop_parts else "none"
        print(f"  {key:42s}: {b:>5} → {a:>5}  (dropped: {drop_str})")
    print(
        f"  {'Effective validated segments':42s}: "
        f"{consolidation_stats['effective_validated_segments']}"
    )
    print(
        f"  {'Bad-region windows (merged)':42s}: "
        f"{consolidation_stats['bad_region_windows_total']}"
    )
    print(f"{'=' * 72}\n")

    # Sanity check: warn on unexpected drop rates
    for key in ("artifacts", "added_r_peaks", "flagged_for_interpolation",
                 "tagged_physiological_events", "interpolated_replacements"):
        s = consolidation_stats[key]
        b, a = s["before"], s["after"]
        if b > 0:
            drop_frac = (b - a) / b
            if drop_frac > 0.30:
                logger.warning(
                    "⚠ %s dropped >30%% (%d → %d, %.1f%%) — verify filters",
                    key, b, a, 100 * drop_frac,
                )
            elif drop_frac == 0.0:
                logger.warning(
                    "⚠ %s had 0 drops (%d items) — verify filters are active",
                    key, b,
                )


if __name__ == "__main__":
    main()
