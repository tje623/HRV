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
