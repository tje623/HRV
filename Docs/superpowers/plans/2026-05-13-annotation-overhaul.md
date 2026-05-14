# Annotation Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the ECG artifact-detection pipeline to consume the rebuilt annotation parquet inputs (`beats.parquet`, `segments.parquet`, `bad_regions.parquet`) with their new lean schema, add first-class handling of the `interpolate` artifact subtype (stop excluding it from training), and apply a trinary RMSSD-based clean/noisy_ok/unknown split to `usable` segments.

**Architecture:** Refactor in place. The schema-touching code lives in `data_pipeline.py` (loader, mappings, label builder, peak builder, segment builder). Downstream scripts get small surgical patches to retire the legacy `"interpolated"` past-tense label and to derive `is_added_peak` from `subtype` instead of from a persisted column. New segment-level `segment_rmssd_ms` is computed inside `build_segments()` from `peaks.parquet`. No new files except a minimal test directory.

**Tech Stack:** Python 3.11+, pandas, numpy, pyarrow. Pytest for unit tests (install once into the `~/.envs/hrv` venv). All runs use 12 cores by default (floor of 9 under memory pressure). Working dir: `/Volumes/xHRV`. Venv: `~/.envs/hrv`. Git: commits go to `tje623/HRV` on `main` after every task.

---

## File Structure

**Modified (existing):**
- `Scripts/data_pipeline.py` — schema, mappings, label/peak/segment builders, RMSSD computation
- `Scripts/features/beat_features.py` — derive `is_added_peak` feature from `labels.parquet` subtype
- `Scripts/models/beat_artifact_cnn.py` — remove interpolate exclusion
- `Scripts/models/beat_artifact_tabular.py` — remove interpolate exclusion
- `Scripts/models/ensemble.py` — purge `"interpolated"` from `VALID_LABELS`
- `Scripts/utils/validate_retrained_model.py` — purge `"interpolated"` from `VALID_LABELS`
- `Scripts/utils/eval_baselines.py` — drop `"interpolated"` filter, add subtype-stratified recall
- `Scripts/utils/validation_report.py` — drop `"interpolated"` filter, add subtype-stratified recall
- `Scripts/features/auto_categorize_beats.py` — purge `"interpolated"` from `CLEAN_REAL_LABELS`
- `Scripts/utils/validate_auto_categories.py` — purge `"interpolated"` from `CLEAN_REAL_LABELS`

**Created (new):**
- `Scripts/tests/__init__.py` — empty marker
- `Scripts/tests/conftest.py` — adds `Scripts/` to `sys.path` so `import data_pipeline` works
- `Scripts/tests/test_annotation_loader.py` — schema validation tests
- `Scripts/tests/test_mappings.py` — `BEAT_LABEL_MAP` and `SEGMENT_QUALITY_MAP` tests
- `Scripts/tests/test_segment_rmssd.py` — RMSSD computation and trinary banding tests
- `Scripts/tests/test_build_labels.py` — `subtype` propagation tests

**Out of scope (intentionally not touched):**
- `Scripts/utils/rebuild_v1_annotation_input.py`, `Scripts/utils/v1_eval.py`, `Scripts/utils/annotation_investigator.py` — V1 legacy utilities, will be left broken against the new schema.

---

## Task 0: Test infrastructure setup

**Files:**
- Create: `Scripts/tests/__init__.py`
- Create: `Scripts/tests/conftest.py`

- [ ] **Step 1: Install pytest into the project venv**

```bash
source ~/.envs/hrv/bin/activate && pip install pytest
```

Expected: `Successfully installed pytest-…` (or "already satisfied").

- [ ] **Step 2: Create the test directory marker**

Write `Scripts/tests/__init__.py` with this single line:

```python
# Pytest discovery root for Scripts/.
```

- [ ] **Step 3: Create `conftest.py` so tests can import pipeline modules**

Write `Scripts/tests/conftest.py`:

```python
"""Pytest configuration: make Scripts/ importable from test files.

Tests live under Scripts/tests/ and import modules like `data_pipeline`
directly. This conftest prepends Scripts/ to sys.path so those imports
resolve without a packaging/install step.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
```

- [ ] **Step 4: Verify pytest discovers the test directory**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/ --collect-only
```

Expected: `no tests ran` or `collected 0 items` (no test files yet — verifying discovery works).

- [ ] **Step 5: Commit and push**

```bash
cd /Volumes/xHRV && git add Scripts/tests/__init__.py Scripts/tests/conftest.py && git commit -m "$(cat <<'EOF'
test: add pytest discovery root for Scripts/

Adds Scripts/tests/conftest.py so test files can `import data_pipeline`
directly without packaging the project.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" && git push origin main
```

---

## Task 1: Update `BEAT_LABEL_MAP` and `SEGMENT_QUALITY_MAP`

**Files:**
- Modify: `Scripts/data_pipeline.py:373-387`
- Create: `Scripts/tests/test_mappings.py`

The new annotation schema replaces `partial` → `unclean` in `segment_quality_label`. `BEAT_LABEL_MAP` is already correct (`clean`/`artifact`/`physio`).

- [ ] **Step 1: Write the failing test**

Write `Scripts/tests/test_mappings.py`:

```python
"""Tests for BEAT_LABEL_MAP and SEGMENT_QUALITY_MAP in data_pipeline."""
from __future__ import annotations

import data_pipeline


def test_beat_label_map_covers_new_schema():
    """beats.parquet `label` values map to canonical labels.parquet values."""
    assert data_pipeline.BEAT_LABEL_MAP == {
        "clean": "clean",
        "artifact": "artifact",
        "physio": "phys_event",
    }


def test_segment_quality_map_uses_unclean_not_partial():
    """segments.parquet uses `unclean` (new) instead of `partial` (legacy)."""
    assert data_pipeline.SEGMENT_QUALITY_MAP == {
        "usable": "clean",
        "unclean": "noisy_ok",
        "unusable": "bad",
        "unknown": "unknown",
    }


def test_segment_quality_map_has_no_legacy_partial_key():
    assert "partial" not in data_pipeline.SEGMENT_QUALITY_MAP
```

- [ ] **Step 2: Run and confirm failure**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/test_mappings.py -v
```

Expected: `test_segment_quality_map_uses_unclean_not_partial` FAILS (legacy still has `"partial"`); the other two pass.

- [ ] **Step 3: Apply the mapping change**

Edit `Scripts/data_pipeline.py:382-387`:

```python
# Mapping from segments.parquet `segment_quality_label` to canonical
# segments.parquet `quality_label` values consumed by segment_quality.py.
# Note: `usable` maps to `clean` here at the dict level, but build_segments()
# further splits `usable` segments by RMSSD into clean / noisy_ok / unknown.
SEGMENT_QUALITY_MAP: dict[str, str] = {
    "usable": "clean",
    "unclean": "noisy_ok",
    "unusable": "bad",
    "unknown": "unknown",
}
```

- [ ] **Step 4: Run tests and confirm pass**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/test_mappings.py -v
```

Expected: All 3 pass.

- [ ] **Step 5: Commit and push**

```bash
cd /Volumes/xHRV && git add Scripts/data_pipeline.py Scripts/tests/test_mappings.py && git commit -m "$(cat <<'EOF'
refactor(data_pipeline): SEGMENT_QUALITY_MAP uses `unclean` not `partial`

The rebuilt annotation parquets use `unclean` as the segment quality
label (segments that contain bad-regions but are otherwise interpretable).
The legacy `partial` key is retired.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" && git push origin main
```

---

## Task 2: Rewrite `load_annotation_inputs` for the new schema

**Files:**
- Modify: `Scripts/data_pipeline.py:828-917`
- Create: `Scripts/tests/test_annotation_loader.py`

The loader currently expects legacy columns (`original_annotation`, `is_added_peak`, `in_revisit_pile`, `has_bad_region`, `bad_region_count`, `region_idx_within_segment`). It must be rewritten to expect the exact current schemas and fail loudly on deviations.

- [ ] **Step 1: Write the failing tests**

Write `Scripts/tests/test_annotation_loader.py`:

```python
"""Tests for load_annotation_inputs schema validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import data_pipeline


EXPECTED_BEATS_COLS = {
    "timestamp_ms", "segment_idx", "label", "subtype", "beat_training_eligible",
}
EXPECTED_SEGMENTS_COLS = {
    "segment_idx", "start_ms", "end_ms", "review_status",
    "segment_quality_label", "beat_training_eligible_segment",
    "segment_quality_training_eligible",
}
EXPECTED_BAD_REGIONS_COLS = {"segment_idx", "start_ms", "end_ms"}


def _write_minimal_inputs(d: Path) -> None:
    pd.DataFrame({
        "timestamp_ms": [1_000_000, 2_000_000],
        "segment_idx": [10, 10],
        "label": ["clean", "artifact"],
        "subtype": ["auto", "spurious"],
        "beat_training_eligible": [True, True],
    }).to_parquet(d / "beats.parquet")
    pd.DataFrame({
        "segment_idx": [10],
        "start_ms": [1_000_000],
        "end_ms": [2_000_000],
        "review_status": ["reviewed"],
        "segment_quality_label": ["usable"],
        "beat_training_eligible_segment": [True],
        "segment_quality_training_eligible": [True],
    }).to_parquet(d / "segments.parquet")
    pd.DataFrame({
        "segment_idx": pd.Series([], dtype="int32"),
        "start_ms": pd.Series([], dtype="int64"),
        "end_ms": pd.Series([], dtype="int64"),
    }).to_parquet(d / "bad_regions.parquet")


def test_loader_accepts_canonical_schemas(tmp_path):
    _write_minimal_inputs(tmp_path)
    beats, segs, brs = data_pipeline.load_annotation_inputs(tmp_path)
    assert set(beats.columns) == EXPECTED_BEATS_COLS
    assert set(segs.columns) == EXPECTED_SEGMENTS_COLS
    assert set(brs.columns) == EXPECTED_BAD_REGIONS_COLS


def test_loader_rejects_unknown_review_status(tmp_path):
    _write_minimal_inputs(tmp_path)
    segs = pd.read_parquet(tmp_path / "segments.parquet")
    segs["review_status"] = ["revisit"]  # legacy value, no longer accepted
    segs.to_parquet(tmp_path / "segments.parquet")
    with pytest.raises(SystemExit, match=r"review_status"):
        data_pipeline.load_annotation_inputs(tmp_path)


def test_loader_rejects_unknown_segment_quality_label(tmp_path):
    _write_minimal_inputs(tmp_path)
    segs = pd.read_parquet(tmp_path / "segments.parquet")
    segs["segment_quality_label"] = ["partial"]  # legacy value, no longer accepted
    segs.to_parquet(tmp_path / "segments.parquet")
    with pytest.raises(SystemExit, match=r"segment_quality_label"):
        data_pipeline.load_annotation_inputs(tmp_path)


def test_loader_rejects_unknown_beat_label(tmp_path):
    _write_minimal_inputs(tmp_path)
    beats = pd.read_parquet(tmp_path / "beats.parquet")
    beats["label"] = ["clean", "interpolated"]  # legacy past-tense value
    beats.to_parquet(tmp_path / "beats.parquet")
    with pytest.raises(SystemExit, match=r"label"):
        data_pipeline.load_annotation_inputs(tmp_path)


def test_loader_rejects_unknown_subtype(tmp_path):
    _write_minimal_inputs(tmp_path)
    beats = pd.read_parquet(tmp_path / "beats.parquet")
    beats["subtype"] = ["auto", "made_up_subtype"]
    beats.to_parquet(tmp_path / "beats.parquet")
    with pytest.raises(SystemExit, match=r"subtype"):
        data_pipeline.load_annotation_inputs(tmp_path)


def test_loader_rejects_missing_required_column(tmp_path):
    _write_minimal_inputs(tmp_path)
    beats = pd.read_parquet(tmp_path / "beats.parquet")
    beats = beats.drop(columns=["subtype"])
    beats.to_parquet(tmp_path / "beats.parquet")
    with pytest.raises(SystemExit, match=r"subtype"):
        data_pipeline.load_annotation_inputs(tmp_path)


def test_loader_still_rejects_json_input(tmp_path):
    p = tmp_path / "artifact_annotation.json"
    p.write_text("{}")
    with pytest.raises(SystemExit, match=r"parquet annotation inputs"):
        data_pipeline.load_annotation_inputs(p)
```

- [ ] **Step 2: Run and confirm failures**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/test_annotation_loader.py -v
```

Expected: Several tests FAIL (loader does not yet validate vocabularies and still references legacy columns).

- [ ] **Step 3: Rewrite `load_annotation_inputs`**

Replace `Scripts/data_pipeline.py:828-917` with:

```python
def load_annotation_inputs(
    annotations_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three annotation parquet files from the input directory.

    Expects ``annotations_path`` to be a directory containing exactly:
      - beats.parquet
          Columns: timestamp_ms, segment_idx, label, subtype,
          beat_training_eligible.
          `label` ∈ {clean, artifact, physio}.
          `subtype` ∈ {auto, added, manual, spurious, interpolate}.
      - segments.parquet
          Columns: segment_idx, start_ms, end_ms, review_status,
          segment_quality_label, beat_training_eligible_segment,
          segment_quality_training_eligible.
          `review_status` ∈ {reviewed, unreviewed}.
          `segment_quality_label` ∈ {usable, unclean, unusable, unknown}.
      - bad_regions.parquet
          Columns: segment_idx, start_ms, end_ms.

    The pipeline fails loudly (SystemExit) on any deviation from these
    schemas. No silent fallback to legacy columns or values.

    The legacy artifact_annotation.json input is rejected; pass a
    directory containing the three parquets instead.

    Args:
        annotations_path: Path to the annotation INPUT directory.

    Returns:
        ``(beats_df, segments_df, bad_regions_df)``. If the directory
        does not exist, all three are empty DataFrames with the expected
        columns (preserves the historical "no annotations yet" path).
    """
    if annotations_path.is_file() and annotations_path.suffix.lower() == ".json":
        raise SystemExit(
            f"--annotations was given a JSON file ({annotations_path}). "
            f"This pipeline now consumes the parquet annotation inputs. Pass a "
            f"directory containing beats.parquet, segments.parquet, and "
            f"bad_regions.parquet (e.g. Data/Annotations/INPUT/)."
        )

    beats_cols = ["timestamp_ms", "segment_idx", "label", "subtype", "beat_training_eligible"]
    seg_cols = [
        "segment_idx", "start_ms", "end_ms", "review_status",
        "segment_quality_label", "beat_training_eligible_segment",
        "segment_quality_training_eligible",
    ]
    br_cols = ["segment_idx", "start_ms", "end_ms"]

    allowed_review_status = {"reviewed", "unreviewed"}
    allowed_quality_label = {"usable", "unclean", "unusable", "unknown"}
    allowed_beat_label = {"clean", "artifact", "physio"}
    allowed_subtype = {"auto", "added", "manual", "spurious", "interpolate"}

    if not annotations_path.is_dir():
        logger.warning(
            "Annotation directory not found: %s — proceeding with empty inputs",
            annotations_path,
        )
        return (
            pd.DataFrame(columns=beats_cols),
            pd.DataFrame(columns=seg_cols),
            pd.DataFrame(columns=br_cols),
        )

    beats_path = annotations_path / "beats.parquet"
    segs_path = annotations_path / "segments.parquet"
    br_path = annotations_path / "bad_regions.parquet"

    missing = [p.name for p in (beats_path, segs_path, br_path) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Annotation directory {annotations_path} is missing required "
            f"file(s): {missing}. Expected beats.parquet, segments.parquet, "
            f"bad_regions.parquet."
        )

    beats_df = pd.read_parquet(beats_path)
    segments_df = pd.read_parquet(segs_path)
    bad_regions_df = pd.read_parquet(br_path)

    # Column-shape validation: exact set match, no extras, no missing.
    def _check_columns(name: str, df: pd.DataFrame, expected: list[str]) -> None:
        got = set(df.columns)
        exp = set(expected)
        missing_cols = exp - got
        extra_cols = got - exp
        if missing_cols or extra_cols:
            raise SystemExit(
                f"{name} schema mismatch. "
                f"Missing columns: {sorted(missing_cols) or 'none'}. "
                f"Unexpected columns: {sorted(extra_cols) or 'none'}. "
                f"Expected exactly: {sorted(exp)}."
            )

    _check_columns("beats.parquet", beats_df, beats_cols)
    _check_columns("segments.parquet", segments_df, seg_cols)
    _check_columns("bad_regions.parquet", bad_regions_df, br_cols)

    # Vocabulary validation: any unexpected categorical value is a hard error.
    def _check_vocab(name: str, col: str, series: pd.Series, allowed: set[str]) -> None:
        if len(series) == 0:
            return
        actual = set(series.dropna().astype(str).unique())
        bad = actual - allowed
        if bad:
            raise SystemExit(
                f"{name}: column '{col}' contains unexpected value(s) "
                f"{sorted(bad)}. Allowed: {sorted(allowed)}."
            )

    _check_vocab("beats.parquet", "label", beats_df["label"], allowed_beat_label)
    _check_vocab("beats.parquet", "subtype", beats_df["subtype"], allowed_subtype)
    _check_vocab(
        "segments.parquet", "review_status",
        segments_df["review_status"], allowed_review_status,
    )
    _check_vocab(
        "segments.parquet", "segment_quality_label",
        segments_df["segment_quality_label"], allowed_quality_label,
    )

    logger.info(
        "Loaded annotations from %s: %d beats, %d segments, %d bad_regions",
        annotations_path, len(beats_df), len(segments_df), len(bad_regions_df),
    )
    if len(beats_df) > 0:
        logger.info(
            "Input beat label distribution: %s",
            beats_df["label"].value_counts().to_dict(),
        )
        logger.info(
            "Input beat subtype distribution: %s",
            beats_df["subtype"].value_counts().to_dict(),
        )
    if len(segments_df) > 0:
        logger.info(
            "Input segment_quality_label distribution: %s",
            segments_df["segment_quality_label"].value_counts().to_dict(),
        )

    return beats_df, segments_df, bad_regions_df
```

- [ ] **Step 4: Update the module docstring (data_pipeline.py:14-44)**

Replace the multi-line annotation contract block in the module docstring (currently spanning lines 14-44 inside the triple-quoted docstring) with:

```
Annotation input contract (``--annotations`` is a directory):

  beats.parquet
    Columns: timestamp_ms (int64), segment_idx (int32), label
    (clean | artifact | physio), subtype (auto | added | manual |
    spurious | interpolate), beat_training_eligible (bool).
    Scope: training-eligible beats only — beats from unreviewed
    or bad-region time windows are pre-excluded upstream.

  segments.parquet
    Columns: segment_idx (int32), start_ms, end_ms,
    review_status (reviewed | unreviewed),
    segment_quality_label (usable | unclean | unusable | unknown),
    beat_training_eligible_segment (bool),
    segment_quality_training_eligible (bool).

  bad_regions.parquet
    Columns: segment_idx, start_ms, end_ms. Each row is a bad
    time interval inside an otherwise-usable segment.

Beat label mapping (beats.parquet → labels.parquet):
  beats.label "clean"    → labels.label "clean"
  beats.label "artifact" → labels.label "artifact"  (subtype carries
                                                     spurious vs interpolate)
  beats.label "physio"   → labels.label "phys_event"
  Peaks not present in beats.parquet → label "unknown", subtype "",
  reviewed=False.

Segment quality mapping (segments.parquet → segments.parquet output):
  segment_quality_label "usable"   + RMSSD < 50  → quality_label "clean"
  segment_quality_label "usable"   + 50 ≤ RMSSD < 250 → quality_label "unknown"
  segment_quality_label "usable"   + RMSSD ≥ 250 → quality_label "noisy_ok"
  segment_quality_label "unclean"  → quality_label "noisy_ok"
  segment_quality_label "unusable" → quality_label "bad"
  segment_quality_label "unknown"  → quality_label "unknown"
```

- [ ] **Step 5: Run tests and confirm pass**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/test_annotation_loader.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 6: Commit and push**

```bash
cd /Volumes/xHRV && git add Scripts/data_pipeline.py Scripts/tests/test_annotation_loader.py && git commit -m "$(cat <<'EOF'
refactor(data_pipeline): consume new annotation parquet schemas

Rewrites load_annotation_inputs to expect the rebuilt schemas exactly
(beats: 5 cols incl. subtype; segments: 7 cols, review in {reviewed,
unreviewed}, quality in {usable, unclean, unusable, unknown};
bad_regions: 3 cols). Validates column shape and value vocabularies
and fails loudly on any deviation. Module docstring updated to match.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" && git push origin main
```

---

## Task 3: Add `segment_rmssd_ms` computation helper

**Files:**
- Modify: `Scripts/data_pipeline.py` (add new function in the ANNOTATION HELPERS section after `extract_bad_region_time_ranges`, around line 410)
- Create: `Scripts/tests/test_segment_rmssd.py`

- [ ] **Step 1: Write the failing tests**

Write `Scripts/tests/test_segment_rmssd.py`:

```python
"""Tests for compute_segment_rmssd and classify_usable_segment_quality."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import data_pipeline


def test_rmssd_zero_for_perfectly_regular_rr():
    """Equal-spaced peaks → zero successive RR differences → RMSSD = 0."""
    peaks_df = pd.DataFrame({
        "peak_id": [1, 2, 3, 4, 5],
        "timestamp_ms": [0, 1000, 2000, 3000, 4000],
        "segment_idx": [7, 7, 7, 7, 7],
    })
    rmssd = data_pipeline.compute_segment_rmssd(peaks_df)
    assert 7 in rmssd
    assert math.isclose(rmssd[7], 0.0, abs_tol=1e-6)


def test_rmssd_nan_for_fewer_than_three_peaks():
    """A segment with 0, 1, or 2 peaks has no successive RR difference."""
    peaks_df = pd.DataFrame({
        "peak_id": [1, 2, 3],
        "timestamp_ms": [0, 1000, 5000],
        "segment_idx": [1, 1, 2],  # seg 1 has 2 peaks; seg 2 has 1
    })
    rmssd = data_pipeline.compute_segment_rmssd(peaks_df)
    assert math.isnan(rmssd[1])
    assert math.isnan(rmssd[2])


def test_rmssd_known_value():
    """RMSSD over RRs [1000, 1100, 900] → successive diffs [100, -200] → sqrt(mean([10000,40000])) ≈ 158.11."""
    peaks_df = pd.DataFrame({
        "peak_id": [1, 2, 3, 4],
        "timestamp_ms": [0, 1000, 2100, 3000],
        "segment_idx": [3, 3, 3, 3],
    })
    rmssd = data_pipeline.compute_segment_rmssd(peaks_df)
    assert math.isclose(rmssd[3], math.sqrt((100**2 + 200**2) / 2), rel_tol=1e-6)


def test_classify_clean_under_50():
    assert data_pipeline.classify_usable_segment_quality(0.0) == "clean"
    assert data_pipeline.classify_usable_segment_quality(49.9) == "clean"


def test_classify_unknown_in_middle_band():
    assert data_pipeline.classify_usable_segment_quality(50.0) == "unknown"
    assert data_pipeline.classify_usable_segment_quality(100.0) == "unknown"
    assert data_pipeline.classify_usable_segment_quality(249.9) == "unknown"


def test_classify_noisy_ok_at_or_above_250():
    assert data_pipeline.classify_usable_segment_quality(250.0) == "noisy_ok"
    assert data_pipeline.classify_usable_segment_quality(1000.0) == "noisy_ok"


def test_classify_nan_is_unknown():
    assert data_pipeline.classify_usable_segment_quality(float("nan")) == "unknown"
```

- [ ] **Step 2: Run and confirm failures**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/test_segment_rmssd.py -v
```

Expected: all FAIL (`compute_segment_rmssd` and `classify_usable_segment_quality` do not exist yet).

- [ ] **Step 3: Add the constants and helpers**

Insert into `Scripts/data_pipeline.py` immediately after the `SEGMENT_QUALITY_MAP` dict (around line 388):

```python
# RMSSD trinary banding for `usable` segments. The "clean" cutoff is
# deliberately conservative — the user empirically validated that <50 ms
# does not let noisy segments leak into the clean pool. The "noisy_ok"
# floor of 250 ms is the symmetric conservative choice on the other
# side. The middle band is intentionally discarded into "unknown" to
# protect class purity at both ends.
SEGMENT_RMSSD_CLEAN_MAX_MS: float = 50.0
SEGMENT_RMSSD_NOISY_MIN_MS: float = 250.0


def compute_segment_rmssd(peaks_df: pd.DataFrame) -> dict[int, float]:
    """Return {segment_idx → RMSSD in ms} computed over all peaks per segment.

    RMSSD is computed from the auto-detected peak series (post-dedup) as
    sqrt(mean(diff(RR)^2)), where RR_i = timestamp_ms[i+1] - timestamp_ms[i].
    A segment with fewer than 3 peaks (so fewer than 2 RR intervals and
    thus zero successive differences) gets NaN.

    No label filtering: artifact and clean peaks alike act as anchors,
    consistent with the design decision to keep RMSSD computation
    label-independent.
    """
    out: dict[int, float] = {}
    if len(peaks_df) == 0:
        return out
    for seg_idx, group in peaks_df.groupby("segment_idx"):
        seg_int = int(seg_idx)
        if len(group) < 3:
            out[seg_int] = float("nan")
            continue
        ts = np.sort(group["timestamp_ms"].to_numpy().astype(np.int64))
        rr = np.diff(ts).astype(np.float64)
        succ_diffs = np.diff(rr)
        if succ_diffs.size == 0:
            out[seg_int] = float("nan")
            continue
        out[seg_int] = float(np.sqrt(np.mean(succ_diffs ** 2)))
    return out


def classify_usable_segment_quality(rmssd_ms: float) -> str:
    """Map a `usable`-input segment's RMSSD to its output quality_label.

    - RMSSD < SEGMENT_RMSSD_CLEAN_MAX_MS (50)        → "clean"
    - SEGMENT_RMSSD_CLEAN_MAX_MS ≤ RMSSD <
      SEGMENT_RMSSD_NOISY_MIN_MS (250)               → "unknown" (discarded)
    - RMSSD ≥ SEGMENT_RMSSD_NOISY_MIN_MS (250)       → "noisy_ok"
    - NaN (insufficient peaks)                       → "unknown" (discarded)
    """
    if rmssd_ms is None or (isinstance(rmssd_ms, float) and math.isnan(rmssd_ms)):
        return "unknown"
    if rmssd_ms < SEGMENT_RMSSD_CLEAN_MAX_MS:
        return "clean"
    if rmssd_ms >= SEGMENT_RMSSD_NOISY_MIN_MS:
        return "noisy_ok"
    return "unknown"
```

Add `import math` to the imports at the top of the file if it isn't already there (check around line 66-95).

- [ ] **Step 4: Run tests and confirm pass**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/test_segment_rmssd.py -v
```

Expected: all 7 pass.

- [ ] **Step 5: Commit and push**

```bash
cd /Volumes/xHRV && git add Scripts/data_pipeline.py Scripts/tests/test_segment_rmssd.py && git commit -m "$(cat <<'EOF'
feat(data_pipeline): add compute_segment_rmssd + RMSSD trinary classifier

Adds compute_segment_rmssd(peaks_df) returning per-segment RMSSD in ms
(NaN when <3 peaks), and classify_usable_segment_quality(rmssd) mapping
to clean (<50) / unknown (50–250) / noisy_ok (≥250). Used by
build_segments to split `usable`-input segments into pure training
classes.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" && git push origin main
```

---

## Task 4: Rewrite `build_segments` for trinary RMSSD banding + eligibility recomputation

**Files:**
- Modify: `Scripts/data_pipeline.py:1348-1555` (and the call site at ~1859)

The current `build_segments` (a) still references the legacy `revisit` and `partial` values, (b) has no RMSSD path, (c) does not include `segment_rmssd_ms`, `beat_training_eligible_segment`, or recomputed `segment_quality_training_eligible` in its output.

- [ ] **Step 1: Extend the test file with build_segments behavior**

Append to `Scripts/tests/test_segment_rmssd.py`:

```python
def test_build_segments_usable_clean_under_50(tmp_path):
    """`usable` + RMSSD < 50 → quality_label=clean, eligibilities preserved."""
    # 60s segment with perfectly regular 1000ms RRs → RMSSD = 0
    peaks_df = pd.DataFrame({
        "peak_id": list(range(60)),
        "timestamp_ms": list(range(0, 60_000, 1_000)),
        "segment_idx": [0] * 60,
        "source": ["detected"] * 60,
    })
    seg_ranges = {0: (0, 60_000)}
    segments_input_df = pd.DataFrame({
        "segment_idx": [0],
        "start_ms": [0],
        "end_ms": [60_000],
        "review_status": ["reviewed"],
        "segment_quality_label": ["usable"],
        "beat_training_eligible_segment": [True],
        "segment_quality_training_eligible": [True],
    })
    out = data_pipeline.build_segments(
        seg_ranges=seg_ranges,
        segments_input_df=segments_input_df,
        peaks_df=peaks_df,
        recording_start_ns=0,
        diagnostics_dir=None,
    )
    assert list(out["quality_label"]) == ["clean"]
    assert math.isclose(out["segment_rmssd_ms"].iloc[0], 0.0, abs_tol=1e-6)
    assert bool(out["beat_training_eligible_segment"].iloc[0]) is True
    assert bool(out["segment_quality_training_eligible"].iloc[0]) is True


def test_build_segments_usable_in_purgatory_becomes_unknown(tmp_path):
    """`usable` + 50 ≤ RMSSD < 250 → quality_label=unknown, segment_quality_training_eligible flipped to False; beat eligibility preserved."""
    # Construct RRs whose RMSSD lands in [50, 250). With RRs alternating
    # 900/1100 ms, successive diffs are all ±200 ms → RMSSD = 200 ms.
    ts = []
    cur = 0
    for i in range(60):
        cur += 900 if i % 2 == 0 else 1100
        ts.append(cur)
    peaks_df = pd.DataFrame({
        "peak_id": list(range(len(ts))),
        "timestamp_ms": ts,
        "segment_idx": [0] * len(ts),
        "source": ["detected"] * len(ts),
    })
    seg_ranges = {0: (0, ts[-1])}
    segments_input_df = pd.DataFrame({
        "segment_idx": [0],
        "start_ms": [0],
        "end_ms": [ts[-1]],
        "review_status": ["reviewed"],
        "segment_quality_label": ["usable"],
        "beat_training_eligible_segment": [True],
        "segment_quality_training_eligible": [True],
    })
    out = data_pipeline.build_segments(
        seg_ranges=seg_ranges,
        segments_input_df=segments_input_df,
        peaks_df=peaks_df,
        recording_start_ns=0,
        diagnostics_dir=None,
    )
    assert list(out["quality_label"]) == ["unknown"]
    assert bool(out["beat_training_eligible_segment"].iloc[0]) is True  # PRESERVED
    assert bool(out["segment_quality_training_eligible"].iloc[0]) is False  # FLIPPED


def test_build_segments_unclean_input_becomes_noisy_ok():
    peaks_df = pd.DataFrame({
        "peak_id": [0, 1, 2, 3],
        "timestamp_ms": [0, 1000, 2000, 3000],
        "segment_idx": [0, 0, 0, 0],
        "source": ["detected"] * 4,
    })
    seg_ranges = {0: (0, 3000)}
    segments_input_df = pd.DataFrame({
        "segment_idx": [0],
        "start_ms": [0], "end_ms": [3000],
        "review_status": ["reviewed"],
        "segment_quality_label": ["unclean"],
        "beat_training_eligible_segment": [True],
        "segment_quality_training_eligible": [True],
    })
    out = data_pipeline.build_segments(
        seg_ranges=seg_ranges,
        segments_input_df=segments_input_df,
        peaks_df=peaks_df,
        recording_start_ns=0,
        diagnostics_dir=None,
    )
    assert list(out["quality_label"]) == ["noisy_ok"]
    assert bool(out["segment_quality_training_eligible"].iloc[0]) is True


def test_build_segments_unusable_input_becomes_bad():
    peaks_df = pd.DataFrame({
        "peak_id": [0, 1, 2, 3],
        "timestamp_ms": [0, 1000, 2000, 3000],
        "segment_idx": [0, 0, 0, 0],
        "source": ["detected"] * 4,
    })
    seg_ranges = {0: (0, 3000)}
    segments_input_df = pd.DataFrame({
        "segment_idx": [0],
        "start_ms": [0], "end_ms": [3000],
        "review_status": ["reviewed"],
        "segment_quality_label": ["unusable"],
        "beat_training_eligible_segment": [False],
        "segment_quality_training_eligible": [True],
    })
    out = data_pipeline.build_segments(
        seg_ranges=seg_ranges,
        segments_input_df=segments_input_df,
        peaks_df=peaks_df,
        recording_start_ns=0,
        diagnostics_dir=None,
    )
    assert list(out["quality_label"]) == ["bad"]
    assert bool(out["segment_quality_training_eligible"].iloc[0]) is True
```

- [ ] **Step 2: Run and confirm failures**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/test_segment_rmssd.py -v
```

Expected: the four `test_build_segments_*` tests FAIL — `build_segments` does not yet accept a `peaks_df` argument and does not produce the new columns.

- [ ] **Step 3: Replace `build_segments`**

Replace `Scripts/data_pipeline.py:1348-1555` with the following. The replacement reads the input `segments_input_df`, joins per-segment RMSSD computed from `peaks_df`, applies the trinary banding for `usable` inputs, and recomputes both eligibility booleans on the output:

```python
def build_segments(
    seg_ranges: dict[int, tuple[int, int]],
    segments_input_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    recording_start_ns: int,
    diagnostics_dir: Path | None = None,
) -> pd.DataFrame:
    """Build segment-level quality labels from the input segments parquet.

    Output columns:
      segment_idx (int32)
      start_timestamp_ms, end_timestamp_ms (int64)
      review_status ∈ {reviewed, unreviewed}
      quality_label ∈ {clean, noisy_ok, bad, unknown}
      segment_rmssd_ms (float32, NaN when <3 peaks)
      beat_training_eligible_segment (bool, passes through from input)
      segment_quality_training_eligible (bool, recomputed from final
        quality_label)

    Mapping rules:
      usable + RMSSD < 50           → clean
      usable + 50 ≤ RMSSD < 250     → unknown (RMSSD purgatory; discarded)
      usable + RMSSD ≥ 250          → noisy_ok
      usable + RMSSD NaN            → unknown
      unclean                       → noisy_ok
      unusable                      → bad
      unknown (input)               → unknown

    segment_quality_training_eligible is True iff the OUTPUT
    quality_label is in {clean, noisy_ok, bad}. beat_training_eligible_segment
    is passed through unchanged from the input — the RMSSD-driven
    demotion to `unknown` is purely a segment-level concern.

    The input segments_input_df uses a GUI ordinal `segment_idx` while the
    pipeline's seg_ranges keys are wall-clock segment ordinals; matching is
    done by timestamp overlap, not raw id equality.
    """
    # Precompute per-pipeline-segment RMSSD from peaks_df.
    rmssd_by_pipe_seg = compute_segment_rmssd(peaks_df)

    # Match each annotation row to its best-overlapping pipeline segment by time.
    quality_by_seg: dict[int, str] = {}
    overlap_by_seg: dict[int, int] = {}
    priority_by_quality = {"unknown": 0, "clean": 1, "noisy_ok": 2, "bad": 3}
    mapping_records: list[dict[str, Any]] = []

    if len(segments_input_df) > 0:
        for row in segments_input_df.itertuples(index=False):
            input_seg_idx = int(getattr(row, "segment_idx"))
            start_ms = int(getattr(row, "start_ms"))
            end_ms = int(getattr(row, "end_ms"))
            raw_label = str(getattr(row, "segment_quality_label"))
            review_status = str(getattr(row, "review_status", ""))

            wall_start_seg = int((start_ms - recording_start_ns) // SEGMENT_DURATION_MS)
            wall_end_seg = int(
                (max(start_ms, end_ms - 1) - recording_start_ns) // SEGMENT_DURATION_MS
            )
            best_pipe_seg: int | None = None
            max_overlap = 0
            for pipe_seg in range(wall_start_seg, wall_end_seg + 1):
                pipe_start = recording_start_ns + pipe_seg * SEGMENT_DURATION_MS
                pipe_end = recording_start_ns + (pipe_seg + 1) * SEGMENT_DURATION_MS
                overlap = max(0, min(end_ms, pipe_end) - max(start_ms, pipe_start))
                if overlap <= 0 or pipe_seg not in seg_ranges:
                    continue
                if overlap > max_overlap:
                    best_pipe_seg = pipe_seg
                    max_overlap = int(overlap)

            # Map raw label to canonical quality_label, applying RMSSD split
            # only on `usable` inputs.
            if raw_label == "usable":
                rmssd = rmssd_by_pipe_seg.get(best_pipe_seg, float("nan"))
                quality = classify_usable_segment_quality(rmssd)
            else:
                quality = SEGMENT_QUALITY_MAP.get(raw_label, "unknown")

            applies = best_pipe_seg is not None and quality != "unknown"
            applied_pipe_seg: int | None = None
            if applies:
                old_overlap = overlap_by_seg.get(best_pipe_seg, -1)
                old_quality = quality_by_seg.get(best_pipe_seg, "unknown")
                should_replace = max_overlap > old_overlap or (
                    max_overlap == old_overlap
                    and priority_by_quality[quality] > priority_by_quality[old_quality]
                )
                if should_replace:
                    quality_by_seg[best_pipe_seg] = quality
                    overlap_by_seg[best_pipe_seg] = max_overlap
                applied_pipe_seg = best_pipe_seg

            mapping_records.append({
                "input_segment_idx": input_seg_idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "review_status": review_status,
                "raw_segment_quality_label": raw_label,
                "canonical_quality_label": quality,
                "best_pipeline_segment_idx": best_pipe_seg,
                "best_overlap_ms": max_overlap,
                "applied_pipeline_segment_idx": applied_pipe_seg,
            })

        if diagnostics_dir is not None:
            write_csv(
                pd.DataFrame(mapping_records),
                diagnostics_dir / "annotation_segment_index_mapping.csv",
            )

    # Also track which annotation row owns each pipeline segment so we can
    # preserve `beat_training_eligible_segment` from the input.
    btes_by_seg: dict[int, bool] = {}
    if len(segments_input_df) > 0 and "beat_training_eligible_segment" in segments_input_df.columns:
        for rec in mapping_records:
            pipe_seg = rec["applied_pipeline_segment_idx"]
            if pipe_seg is None:
                continue
            input_idx = rec["input_segment_idx"]
            input_row = segments_input_df.loc[
                segments_input_df["segment_idx"] == input_idx
            ]
            if len(input_row) > 0:
                btes_by_seg[pipe_seg] = bool(
                    input_row["beat_training_eligible_segment"].iloc[0]
                )

    records: list[dict[str, Any]] = []
    for seg_int in sorted(seg_ranges.keys()):
        start_ms, end_ms = seg_ranges[seg_int]
        quality = quality_by_seg.get(seg_int, "unknown")
        rmssd = rmssd_by_pipe_seg.get(seg_int, float("nan"))
        beat_eligible = btes_by_seg.get(seg_int, False)
        sq_eligible = quality in {"clean", "noisy_ok", "bad"}

        # Pick up review_status from the mapping if this segment was matched;
        # otherwise default to "unreviewed".
        review = "unreviewed"
        for rec in mapping_records:
            if rec["applied_pipeline_segment_idx"] == seg_int:
                review = rec["review_status"]
                break

        records.append({
            "segment_idx": np.int32(seg_int),
            "start_timestamp_ms": np.int64(start_ms),
            "end_timestamp_ms": np.int64(end_ms),
            "review_status": review,
            "quality_label": quality,
            "segment_rmssd_ms": np.float32(rmssd),
            "beat_training_eligible_segment": bool(beat_eligible),
            "segment_quality_training_eligible": bool(sq_eligible),
        })

    result = pd.DataFrame(records)
    result["segment_idx"] = result["segment_idx"].astype(np.int32)
    result["start_timestamp_ms"] = result["start_timestamp_ms"].astype(np.int64)
    result["end_timestamp_ms"] = result["end_timestamp_ms"].astype(np.int64)
    result["segment_rmssd_ms"] = result["segment_rmssd_ms"].astype(np.float32)
    result["beat_training_eligible_segment"] = result["beat_training_eligible_segment"].astype(bool)
    result["segment_quality_training_eligible"] = result["segment_quality_training_eligible"].astype(bool)

    logger.info(
        "Segment quality distribution:\n%s",
        result["quality_label"].value_counts().to_string(),
    )
    n_purgatory = int(
        (result["quality_label"] == "unknown")
        & (~result["segment_rmssd_ms"].isna())
        & (result["segment_rmssd_ms"] >= SEGMENT_RMSSD_CLEAN_MAX_MS)
        & (result["segment_rmssd_ms"] < SEGMENT_RMSSD_NOISY_MIN_MS)
    ).sum() if "segment_rmssd_ms" in result.columns else 0
    logger.info("RMSSD purgatory segments (50 ≤ RMSSD < 250 + usable): %d", n_purgatory)
    return result
```

- [ ] **Step 4: Update the call site**

Find the existing call in `Scripts/data_pipeline.py` around line 1859:

```python
segments = build_segments(seg_ranges, segments_input_df, recording_start_ns, diagnostics_dir)
```

Replace with (note `peaks` is the already-built peaks DataFrame from the preceding `build_peaks(...)` call):

```python
segments = build_segments(
    seg_ranges=seg_ranges,
    segments_input_df=segments_input_df,
    peaks_df=peaks,
    recording_start_ns=recording_start_ns,
    diagnostics_dir=diagnostics_dir,
)
```

- [ ] **Step 5: Run all tests, confirm pass**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/ -v
```

Expected: all tests across all test files pass.

- [ ] **Step 6: Commit and push**

```bash
cd /Volumes/xHRV && git add Scripts/data_pipeline.py Scripts/tests/test_segment_rmssd.py && git commit -m "$(cat <<'EOF'
feat(data_pipeline): trinary RMSSD banding + recomputed eligibility in build_segments

Replaces build_segments to:
  - split `usable`-input segments by RMSSD into clean (<50) /
    unknown (50–250, discarded) / noisy_ok (≥250)
  - include `segment_rmssd_ms` in the output
  - recompute `segment_quality_training_eligible` from the final
    quality_label so RMSSD-purgatory rows are correctly flagged False
  - pass through `beat_training_eligible_segment` unchanged
  - drop legacy `revisit` / `partial` handling entirely

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" && git push origin main
```

---

## Task 5: Propagate `subtype` through `build_peaks` and `build_labels`; drop `is_added_peak` / `phys_event_window` / `in_bad_region`

**Files:**
- Modify: `Scripts/data_pipeline.py:925-1042` (build_peaks)
- Modify: `Scripts/data_pipeline.py:1045-1345` (build_labels)
- Create: `Scripts/tests/test_build_labels.py`

`is_added_peak` is no longer a persisted column in peaks.parquet — it is derived from `subtype=="added"` when needed. `subtype` flows from `beats.parquet` into `labels.parquet`. `phys_event_window` and `in_bad_region` are removed from `labels.parquet` (recoverable from `label=="phys_event"` and from the bad-region scrubbing already done upstream).

- [ ] **Step 1: Write the failing test**

Write `Scripts/tests/test_build_labels.py`:

```python
"""Tests for subtype propagation and column hygiene in build_peaks/build_labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_pipeline


def _toy_peaks_csv(timestamps_ms):
    return pd.DataFrame({
        "peak_id": [int(t) for t in timestamps_ms],
        "source": ["detected"] * len(timestamps_ms),
    })


def _toy_beats(rows):
    return pd.DataFrame(rows)


def test_build_peaks_does_not_emit_is_added_peak_column():
    csv = _toy_peaks_csv([1_000_000_000, 2_000_000_000])
    beats = _toy_beats([
        {"timestamp_ms": 1_500_000_000, "segment_idx": 0,
         "label": "clean", "subtype": "added", "beat_training_eligible": True},
    ])
    peaks = data_pipeline.build_peaks(csv, beats, recording_start_ns=0)
    assert "is_added_peak" not in peaks.columns
    assert "source" in peaks.columns
    # The added beat should still produce a peak row.
    assert (peaks["source"] == "added").sum() == 1


def test_build_labels_emits_subtype_and_drops_phys_event_window_and_in_bad_region():
    peaks = pd.DataFrame({
        "peak_id": [10, 20, 30, 40],
        "timestamp_ms": [1_000_000, 2_000_000, 3_000_000, 4_000_000],
        "segment_idx": [0, 0, 0, 0],
        "source": ["detected"] * 4,
    })
    beats = _toy_beats([
        {"timestamp_ms": 1_000_000, "segment_idx": 0,
         "label": "clean", "subtype": "auto", "beat_training_eligible": True},
        {"timestamp_ms": 2_000_000, "segment_idx": 0,
         "label": "artifact", "subtype": "spurious", "beat_training_eligible": True},
        {"timestamp_ms": 3_000_000, "segment_idx": 0,
         "label": "artifact", "subtype": "interpolate", "beat_training_eligible": True},
        # peak_id 40 has no matching beat -> label "unknown", subtype "".
    ])
    labels = data_pipeline.build_labels(peaks, beats, bad_region_ranges=None)
    assert "subtype" in labels.columns
    assert "phys_event_window" not in labels.columns
    assert "in_bad_region" not in labels.columns
    by_pid = labels.set_index("peak_id")
    assert by_pid.loc[10, "label"] == "clean"
    assert by_pid.loc[10, "subtype"] == "auto"
    assert by_pid.loc[20, "label"] == "artifact"
    assert by_pid.loc[20, "subtype"] == "spurious"
    assert by_pid.loc[30, "label"] == "artifact"
    assert by_pid.loc[30, "subtype"] == "interpolate"
    assert by_pid.loc[40, "label"] == "unknown"
    assert by_pid.loc[40, "subtype"] == ""


def test_build_labels_maps_physio_to_phys_event():
    peaks = pd.DataFrame({
        "peak_id": [10],
        "timestamp_ms": [1_000_000],
        "segment_idx": [0],
        "source": ["detected"],
    })
    beats = _toy_beats([
        {"timestamp_ms": 1_000_000, "segment_idx": 0,
         "label": "physio", "subtype": "manual", "beat_training_eligible": True},
    ])
    labels = data_pipeline.build_labels(peaks, beats, bad_region_ranges=None)
    assert labels.loc[labels["peak_id"] == 10, "label"].iloc[0] == "phys_event"
    assert labels.loc[labels["peak_id"] == 10, "subtype"].iloc[0] == "manual"
```

- [ ] **Step 2: Run and confirm failure**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/test_build_labels.py -v
```

Expected: failures — `peaks` still contains `is_added_peak`, `labels` still has `phys_event_window`/`in_bad_region`, and `labels` does not yet have `subtype`.

- [ ] **Step 3: Update `build_peaks` to derive added-status from `subtype` and drop `is_added_peak` from output**

In `Scripts/data_pipeline.py`, find the block at lines 974-990 that reads added peaks from beats_df using `is_added_peak`:

```python
    if (
        len(beats_df) > 0
        and "is_added_peak" in beats_df.columns
        and "timestamp_ms" in beats_df.columns
    ):
        added_mask = beats_df["is_added_peak"].fillna(False).astype(bool).values
        added_ts = beats_df.loc[added_mask, "timestamp_ms"].astype(np.int64).values
        for ts in added_ts:
            records.append(
                {
                    "timestamp_ms": int(ts),
                    "source": "added",
                    "is_added_peak": True,
                    "_origin": "annotation",
                }
            )
```

Replace with:

```python
    if (
        len(beats_df) > 0
        and "subtype" in beats_df.columns
        and "timestamp_ms" in beats_df.columns
    ):
        added_mask = (beats_df["subtype"] == "added").values
        added_ts = beats_df.loc[added_mask, "timestamp_ms"].astype(np.int64).values
        for ts in added_ts:
            records.append(
                {
                    "timestamp_ms": int(ts),
                    "source": "added",
                    "_origin": "annotation",
                }
            )
```

Also find lines 955-959 (the CSV `is_added_peak` column extraction) and the records.append() at 961-972, and the dtype enforcement at line 1034. Change them so `is_added_peak` is never put on the records dict and never enforced on `peaks_df`. The relevant edits:

Replace lines 955-972:

```python
    records: list[dict[str, Any]] = []
    for i in range(len(csv_ts_ns)):
        src = str(csv_source[i]) if not pd.isna(csv_source[i]) else "detected"
        records.append(
            {
                "timestamp_ms": int(csv_ts_ns[i]),
                "source": src,
                "_origin": "csv",
            }
        )
```

(removes the `csv_is_added` reading and the `is_added_peak` field from each record).

Delete the lines:

```python
    csv_is_added = (
        peak_csv_df["is_added_peak"].values
        if "is_added_peak" in peak_csv_df.columns
        else np.full(len(csv_ts_ns), False, dtype=bool)
    )
```

Delete line 1034:

```python
    peaks_df["is_added_peak"] = peaks_df["is_added_peak"].astype(bool)
```

Update the docstring (around lines 942-945) — change `is_added_peak (bool)` to remove that line so the returned columns are just `peak_id, timestamp_ms, segment_idx, source`.

- [ ] **Step 4: Update `build_labels` to add `subtype` and drop `phys_event_window`/`in_bad_region`**

In `Scripts/data_pipeline.py:1101-1143` (the beats-present branch), after `beat_label = beats_sorted["label"].astype(str).values`, add a parallel array for subtype:

```python
        beat_subtype = (
            beats_sorted["subtype"].astype(str).values
            if "subtype" in beats_sorted.columns
            else np.full(len(beats_sorted), "", dtype=object)
        )
```

After `matched_beat_label = beat_label[nearest_idx]` add:

```python
        matched_beat_subtype = beat_subtype[nearest_idx]
```

Above the existing `labels = np.full(...)` line near 1096-1098, add:

```python
    subtypes = np.full(n_peaks, "", dtype=object)
```

Inside the `for src, dst in BEAT_LABEL_MAP.items():` loop around 1137-1141, after `labels[mask] = dst`, add:

```python
            subtypes[mask] = matched_beat_subtype[mask]
```

Remove the `if dst == "phys_event": in_phys_window[mask] = True` line and the `in_phys_window = np.zeros(...)` initialization. Also remove the `in_bad_region` computation block (lines ~1312-1326).

Replace the result-DataFrame construction at lines 1328-1336 with:

```python
    result = pd.DataFrame(
        {
            "peak_id": peaks_df["peak_id"].values.astype(np.int64),
            "label": labels,
            "subtype": subtypes,
            "reviewed": is_reviewed,
        }
    )
```

Within the diagnostics CSV writing (around lines 1207-1220 and 1270-1283), wherever the code references `"original_annotation"` or `"is_added_peak"` in `status_cols` or `cols` lists, replace those tokens with `"subtype"` (and drop `"is_added_peak"` entirely). Likewise change the `group_cols` filter at lines 1250-1253 from `("original_annotation", "label", "is_added_peak")` to `("subtype", "label")`.

Update the function docstring at the top of `build_labels` to reflect the new return schema (drop `phys_event_window` and `in_bad_region`, add `subtype`).

- [ ] **Step 5: Update `beat_features.py` to derive `is_added_peak` feature from labels.parquet's `subtype`**

In `Scripts/features/beat_features.py:614`, replace:

```python
    physio_labels["is_added_peak"] = peaks_sorted["is_added_peak"].values
```

with:

```python
    # Derive the legacy `is_added_peak` feature from labels_sorted.subtype.
    # The peaks.parquet column was retired; the equivalent info now lives
    # in labels.parquet.subtype == "added".
    if "subtype" in labels_sorted.columns:
        physio_labels["is_added_peak"] = (
            labels_sorted["subtype"].astype(str).values == "added"
        )
    else:
        physio_labels["is_added_peak"] = np.zeros(len(labels_sorted), dtype=bool)
```

- [ ] **Step 6: Run all tests, confirm pass**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit and push**

```bash
cd /Volumes/xHRV && git add Scripts/data_pipeline.py Scripts/features/beat_features.py Scripts/tests/test_build_labels.py && git commit -m "$(cat <<'EOF'
refactor(data_pipeline): propagate subtype, drop is_added_peak/phys_event_window/in_bad_region

- peaks.parquet: drops is_added_peak column. The fact survives via
  source=="added" and labels.parquet.subtype=="added".
- labels.parquet: adds subtype (auto/added/manual/spurious/interpolate/""),
  drops phys_event_window (alias of label=="phys_event") and
  in_bad_region (bad-region beats are scrubbed upstream).
- beat_features.py: derives the is_added_peak feature from
  labels.subtype so the model feature matrix is unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" && git push origin main
```

---

## Task 6: Stop excluding `interpolate` beats from training

**Files:**
- Modify: `Scripts/models/beat_artifact_cnn.py:651-654`
- Modify: `Scripts/models/beat_artifact_tabular.py:460-464`

The current scripts filter out any beat with `label == "interpolated"` (the legacy past-tense value). In the new schema, interpolate beats appear as `label=="artifact"` + `subtype=="interpolate"` and should be **included** as artifact-positive examples.

- [ ] **Step 1: Open `Scripts/models/beat_artifact_cnn.py` to lines 650-655**

Current code:

```python
    n_interp = int((merged["label"] == "interpolated").sum())
    if n_interp:
        merged = merged[merged["label"] != "interpolated"].copy()
        logger.info("Excluded %d interpolated beats from training", n_interp)
```

Replace with:

```python
    # Interpolate-subtype artifacts are now INCLUDED as artifact-positive
    # examples. Log subtype composition of the artifact class for
    # evaluation traceability.
    if "subtype" in merged.columns:
        n_interp = int(((merged["label"] == "artifact") & (merged["subtype"] == "interpolate")).sum())
        n_spurious = int(((merged["label"] == "artifact") & (merged["subtype"] == "spurious")).sum())
        logger.info(
            "Artifact training set: %d spurious + %d interpolate "
            "(both included as artifact-positive)",
            n_spurious, n_interp,
        )
```

The unrelated comment on line 221 ("All beats in a segment are interpolated") refers to scalogram interpolation, not the legacy label — leave it untouched.

- [ ] **Step 2: Apply the same change to `Scripts/models/beat_artifact_tabular.py:460-464`**

Current code:

```python
    # ── Exclude interpolated beats ────────────────────────────────────────
    n_interp = int((merged["label"] == "interpolated").sum())
    if n_interp:
        merged = merged[merged["label"] != "interpolated"].copy()
        logger.info("Excluded %d interpolated beats from training", n_interp)
```

Replace with:

```python
    # Interpolate-subtype artifacts are now INCLUDED as artifact-positive
    # examples. Log subtype composition of the artifact class.
    if "subtype" in merged.columns:
        n_interp = int(((merged["label"] == "artifact") & (merged["subtype"] == "interpolate")).sum())
        n_spurious = int(((merged["label"] == "artifact") & (merged["subtype"] == "spurious")).sum())
        logger.info(
            "Artifact training set: %d spurious + %d interpolate "
            "(both included as artifact-positive)",
            n_spurious, n_interp,
        )
```

- [ ] **Step 3: Smoke-import both modules**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && python -c "
import sys; sys.path.insert(0, 'Scripts')
import models.beat_artifact_cnn
import models.beat_artifact_tabular
print('OK')
"
```

Expected: `OK`. (Catches syntax errors without running training.)

- [ ] **Step 4: Commit and push**

```bash
cd /Volumes/xHRV && git add Scripts/models/beat_artifact_cnn.py Scripts/models/beat_artifact_tabular.py && git commit -m "$(cat <<'EOF'
feat(models): include interpolate-subtype artifacts in training

Removes the legacy filter that dropped label=="interpolated" rows from
the artifact training set. Under the new annotation schema, those rows
appear as label=="artifact" + subtype=="interpolate" and are now
included as artifact-positive examples (~101 beats joining the ~10k
spurious examples).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" && git push origin main
```

---

## Task 7: Purge legacy `"interpolated"` label from VALID_LABELS / CLEAN_REAL_LABELS sets

**Files:**
- Modify: `Scripts/models/ensemble.py:48`
- Modify: `Scripts/utils/validate_retrained_model.py:48-50` (multi-line frozenset)
- Modify: `Scripts/utils/eval_baselines.py:56`
- Modify: `Scripts/utils/validation_report.py:29`
- Modify: `Scripts/features/auto_categorize_beats.py:71`
- Modify: `Scripts/utils/validate_auto_categories.py:38`

The past-tense label `"interpolated"` no longer exists. Remove it from every validation set and from any filtering predicate that excludes it.

- [ ] **Step 1: `ensemble.py:48`**

Current:

```python
VALID_LABELS = {"clean", "artifact", "interpolated", "phys_event", "missed_original"}
```

Replace with:

```python
VALID_LABELS = {"clean", "artifact", "phys_event", "missed_original"}
```

- [ ] **Step 2: `validate_retrained_model.py:48-49`**

Current code:

```python
VALID_LABELS    = frozenset({"clean", "artifact", "interpolated",
                              "phys_event", "missed_original"})
```

Replace with:

```python
VALID_LABELS    = frozenset({"clean", "artifact",
                              "phys_event", "missed_original"})
```

- [ ] **Step 3: `eval_baselines.py:56`**

Current:

```python
    reviewed = reviewed[reviewed["label"] != "interpolated"]
```

Delete this line entirely. (Interpolate beats now appear as artifact and should be evaluated like other artifacts.)

- [ ] **Step 4: `validation_report.py:28-29`**

Current code:

```python
    keep = (labels["reviewed"] | (labels["label"] == "artifact")) & \
           (labels["label"] != "interpolated")
```

Replace with:

```python
    keep = labels["reviewed"] | (labels["label"] == "artifact")
```

- [ ] **Step 5: `auto_categorize_beats.py:71`**

Current:

```python
CLEAN_REAL_LABELS = frozenset({"clean", "missed_original", "interpolated", "phys_event"})
```

Replace with:

```python
CLEAN_REAL_LABELS = frozenset({"clean", "missed_original", "phys_event"})
```

- [ ] **Step 6: `validate_auto_categories.py:38`**

Same pattern as Step 5:

```python
CLEAN_REAL_LABELS   = frozenset({"clean", "missed_original", "phys_event"})
```

- [ ] **Step 7: Verify no stray `"interpolated"` (the past-tense token) survives in any Scripts/ file**

```bash
cd /Volumes/xHRV && grep -rn '"interpolated"' Scripts/ --include='*.py' | grep -v 'Scripts/utils/v1_eval.py' | grep -v 'Scripts/utils/rebuild_v1_annotation_input.py' | grep -v 'Scripts/utils/annotation_investigator.py'
```

Expected: no output. Anything that prints is a missed reference (excluding the three out-of-scope V1 scripts).

- [ ] **Step 8: Smoke-import each modified module**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && python -c "
import sys; sys.path.insert(0, 'Scripts')
import models.ensemble
import utils.validate_retrained_model
import utils.eval_baselines
import utils.validation_report
import features.auto_categorize_beats
import utils.validate_auto_categories
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 9: Commit and push**

```bash
cd /Volumes/xHRV && git add Scripts/models/ensemble.py Scripts/utils/validate_retrained_model.py Scripts/utils/eval_baselines.py Scripts/utils/validation_report.py Scripts/features/auto_categorize_beats.py Scripts/utils/validate_auto_categories.py && git commit -m "$(cat <<'EOF'
refactor: purge legacy 'interpolated' beat label from validation sets

The past-tense 'interpolated' label is retired in favor of
label=='artifact' + subtype=='interpolate'. Removes it from VALID_LABELS
and CLEAN_REAL_LABELS in all consumer scripts, and drops the filter
that excluded those rows from eval_baselines and validation_report.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" && git push origin main
```

---

## Task 8: Add subtype-stratified recall reporting to eval scripts

**Files:**
- Modify: `Scripts/utils/eval_baselines.py`
- Modify: `Scripts/utils/validation_report.py`

After Task 7 these scripts treat all artifacts uniformly. We want a per-subtype recall slice so the 101 interpolate examples are observable in evaluation output.

- [ ] **Step 1: In `eval_baselines.py`, add a subtype-stratified recall block after the LGBM AUC computation**

Open `Scripts/utils/eval_baselines.py`. After the existing block that finishes at line ~233 (where `lgbm_pr_auc` and `lgbm_roc_auc` are assigned), and *before* the `# ── Comparison table ─────…` block at line ~235, insert:

```python
    # ── Subtype-stratified artifact recall (binary thresholded preds) ─────
    # Reports recall separately for `spurious` and `interpolate` so the
    # 101 interpolate artifacts are observable rather than averaged into
    # the ~10k spurious. val_df carries `subtype` from labels.parquet.
    if "subtype" in val_df.columns and valid_mask.sum() > 0:
        lgbm_pred = (y_lgbm[valid_mask] >= opt_threshold).astype(int)
        val_sub = val_df.iloc[np.where(valid_mask)[0]]
        print("\nArtifact recall by subtype (LGBM @ optimal threshold):")
        for st in ("spurious", "interpolate"):
            mask = (val_sub["label"].values == "artifact") & (val_sub["subtype"].values == st)
            n = int(mask.sum())
            if n == 0:
                print(f"  {st:>11}: 0 examples in val")
                continue
            tp = int(lgbm_pred[mask].sum())
            print(f"  {st:>11}: {tp} / {n} = {tp/n:.3f}")
```

- [ ] **Step 2: In `validation_report.py`, extend `report_model()` to log per-subtype recall**

Open `Scripts/utils/validation_report.py`. Inside `report_model()`, after the existing confusion-matrix prints at line ~74 (`print(f"    FN={fn:,}  TP={tp:,}")`) and **before** `prob_distribution_bins(p)` at line ~75, insert:

```python
    # ── Subtype-stratified artifact recall ───────────────────────────────
    if "subtype" in merged.columns:
        print(f"  Recall by subtype (threshold={threshold:.4f}):")
        for st in ("spurious", "interpolate"):
            mask = (merged["label"].values == "artifact") & (merged["subtype"].values == st)
            n = int(mask.sum())
            if n == 0:
                print(f"    {st:>11}: 0 examples")
                continue
            tp_st = int(y_pred[mask].sum())
            print(f"    {st:>11}: {tp_st} / {n} = {tp_st/n:.3f}")
```

- [ ] **Step 3: Smoke-import both modules**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && python -c "
import sys; sys.path.insert(0, 'Scripts')
import utils.eval_baselines
import utils.validation_report
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 4: Commit and push**

```bash
cd /Volumes/xHRV && git add Scripts/utils/eval_baselines.py Scripts/utils/validation_report.py && git commit -m "$(cat <<'EOF'
feat(eval): subtype-stratified artifact recall (spurious vs interpolate)

After folding interpolate beats into the artifact-positive class, log
per-subtype recall so the 101 interpolate examples are observable in
evaluation output (rather than averaged in with the 10k spurious).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" && git push origin main
```

---

## Task 9: End-to-end dry run + validation

**Files:**
- No code changes; runs `data_pipeline.py` against the live INPUT directory and inspects outputs.

- [ ] **Step 1: Run the full pipeline against the new annotations**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && python Scripts/data_pipeline.py \
    --annotations Data/Annotations/INPUT \
    --workers 12 2>&1 | tee logs/dp_$(date +%Y%m%d_%H%M%S).log | tail -120
```

(`--workers 12` is the actual flag exposed by `data_pipeline.py`'s argparse. For model scripts in later steps, check `--help` first since the flag name may differ.)

Expected behaviors in the log:
- Loader reports row counts that match the live data (approximately 187k beats, 40k segments, ~97 bad regions).
- No "schema mismatch" or "unexpected value" SystemExit.
- Final segment quality distribution shows `clean ≈ 32k`, `noisy_ok ≈ 2.5k`, `bad = 31`, `unknown` = remainder.
- "RMSSD purgatory segments" log line shows roughly 4,000–4,500.

- [ ] **Step 2: Inspect the output `labels.parquet` for `subtype` column**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && python -c "
import pyarrow.parquet as pq
import pandas as pd
t = pq.read_table('processed/labels.parquet', columns=['label','subtype','reviewed']).to_pandas()
print('schema:')
print(t.dtypes)
print()
print('label x subtype:')
print(pd.crosstab(t['label'], t['subtype']))
print()
print('reviewed True:', int(t['reviewed'].sum()))
"
```

(Adjust the path if `processed/labels.parquet` lives elsewhere — check the script output for the actual write location.)

Expected: `subtype` is present and populated; legacy columns `phys_event_window`, `in_bad_region`, `is_added_peak` are absent.

- [ ] **Step 3: Inspect output `segments.parquet`**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && python -c "
import pyarrow.parquet as pq
t = pq.read_table('processed/segments.parquet').to_pandas()
print(t.dtypes)
print()
print('quality_label counts:')
print(t['quality_label'].value_counts())
print()
print('rmssd summary:')
print(t['segment_rmssd_ms'].describe())
print()
print('segment_quality_training_eligible by quality_label:')
print(t.groupby('quality_label')['segment_quality_training_eligible'].agg(['sum','count']))
"
```

Expected: `segment_quality_training_eligible` is `True` for every `clean`/`noisy_ok`/`bad` row and `False` for every `unknown` row.

- [ ] **Step 4: Retrain `beat_artifact_tabular` and confirm interpolate counts in log**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && python Scripts/models/beat_artifact_tabular.py --workers 12 2>&1 | tee logs/bt_$(date +%Y%m%d_%H%M%S).log | grep -E "Artifact training set|recall on subtype"
```

Expected log lines: `Artifact training set: ~10000 spurious + ~101 interpolate (both included as artifact-positive)`. Eval section should show separate recall lines for `spurious` and `interpolate`.

- [ ] **Step 5: Retrain `beat_artifact_cnn` (same expectations)**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && python Scripts/models/beat_artifact_cnn.py --workers 12 2>&1 | tee logs/bc_$(date +%Y%m%d_%H%M%S).log | grep -E "Artifact training set|recall on subtype"
```

Same expectations as Step 4.

- [ ] **Step 6: Retrain segment_quality and confirm three-class training**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && python Scripts/models/segment_quality.py --workers 12 2>&1 | tee logs/sq_$(date +%Y%m%d_%H%M%S).log | tail -60
```

Expected: training data shape includes only segments with `segment_quality_training_eligible=True`. Class breakdown matches `clean ≈ 32k`, `noisy_ok ≈ 2.5k`, `bad = 31`. Training converges.

- [ ] **Step 7: Final sanity check — full pytest run**

```bash
cd /Volumes/xHRV && source ~/.envs/hrv/bin/activate && pytest Scripts/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit the run logs (optional but recommended)**

```bash
cd /Volumes/xHRV && git add logs/dp_*.log logs/bt_*.log logs/bc_*.log logs/sq_*.log 2>/dev/null; git commit -m "$(cat <<'EOF'
chore: archive end-to-end run logs for annotation-overhaul validation

Captures the data_pipeline + model retraining logs that validated the
new annotation schema, RMSSD trinary banding, and interpolate inclusion.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" 2>&1 | tail -3; git push origin main 2>&1 | tail -3
```

(If `logs/` is gitignored or doesn't exist, skip this step.)

---

## Verification summary

After all 9 tasks the pipeline reads exactly the three new parquet inputs, fails loudly on any schema or vocabulary drift, propagates `subtype` end-to-end, splits `usable` segments into clean (<50 ms RMSSD) / noisy_ok (≥250 ms) / unknown (50–250 ms purgatory + insufficient peaks), correctly recomputes `segment_quality_training_eligible` while preserving `beat_training_eligible_segment`, and includes the 101 interpolate beats as artifact-positive training examples. Per-subtype recall is logged in both eval scripts. The legacy `"interpolated"` past-tense label is purged from all in-scope validation sets.
