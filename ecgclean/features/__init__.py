"""
ecgclean.features — Feature computation for ECG artifact detection pipeline.

Exports:
    compute_beat_feature_matrix   — beat-level feature matrix (Step 3a)
    compute_segment_feature_matrix — segment-level feature matrix (Step 3b)
    pearson_corr_safe             — Pearson correlation with zero-variance guard

Submodules:
    motif_features — QRS morphology & RR-interval motif discovery + anomaly features
"""

from __future__ import annotations

import numpy as np


def pearson_corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with zero-variance protection.

    Returns 0.0 if either input is constant (zero variance) rather than
    propagating NaN from ``np.corrcoef``.  This is the correct behavior
    for ECG windows that are clipped/saturated or consist of a single
    repeated sample.

    Args:
        a: 1-D numeric array.
        b: 1-D numeric array of the same length as *a*.

    Returns:
        Pearson r in [-1, 1], or 0.0 if computation is undefined.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    a_std = np.std(a)
    b_std = np.std(b)
    if a_std == 0.0 or b_std == 0.0:
        return 0.0
    # np.corrcoef returns a 2×2 matrix; [0,1] is the cross-correlation
    r = np.corrcoef(a, b)[0, 1]
    if np.isnan(r) or np.isinf(r):
        return 0.0
    return float(r)


from .beat_features import compute_beat_feature_matrix  # noqa: E402
from .segment_features import compute_segment_feature_matrix  # noqa: E402

__all__ = [
    "compute_beat_feature_matrix",
    "compute_segment_feature_matrix",
    "pearson_corr_safe",
]
