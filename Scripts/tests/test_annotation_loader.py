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
    segs["review_status"] = ["revisit"]
    segs.to_parquet(tmp_path / "segments.parquet")
    with pytest.raises(SystemExit, match=r"review_status"):
        data_pipeline.load_annotation_inputs(tmp_path)


def test_loader_rejects_unknown_segment_quality_label(tmp_path):
    _write_minimal_inputs(tmp_path)
    segs = pd.read_parquet(tmp_path / "segments.parquet")
    segs["segment_quality_label"] = ["partial"]
    segs.to_parquet(tmp_path / "segments.parquet")
    with pytest.raises(SystemExit, match=r"segment_quality_label"):
        data_pipeline.load_annotation_inputs(tmp_path)


def test_loader_rejects_unknown_beat_label(tmp_path):
    _write_minimal_inputs(tmp_path)
    beats = pd.read_parquet(tmp_path / "beats.parquet")
    beats["label"] = ["clean", "interpolated"]
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
