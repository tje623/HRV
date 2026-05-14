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
