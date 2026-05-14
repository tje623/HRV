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


def test_build_segments_usable_clean_under_50():
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


def test_build_segments_usable_in_purgatory_becomes_unknown():
    """`usable` + 50 ≤ RMSSD < 250 → quality_label=unknown, segment_quality_training_eligible flipped to False; beat eligibility preserved."""
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
    assert bool(out["beat_training_eligible_segment"].iloc[0]) is True
    assert bool(out["segment_quality_training_eligible"].iloc[0]) is False


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
