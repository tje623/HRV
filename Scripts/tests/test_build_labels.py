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
    assert "segment_idx" in labels.columns
    # segment_idx propagates from peaks_df
    assert by_pid.loc[10, "segment_idx"] == 0
    assert by_pid.loc[40, "segment_idx"] == 0
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
