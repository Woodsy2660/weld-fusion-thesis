"""Base-plate estimation from a 1D height profile.

RANSAC plane/line fit is the configured primary method and is not yet
implemented. `outer_median_baseplate` is the documented fallback (median
of the outer `fraction` of each end of the profile), ported from the
previous geometry.profile.measure_geometry logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def outer_median_baseplate(profile: np.ndarray, fraction: float = 0.2) -> dict[str, float]:
    """Fallback: median of the first and last `fraction` of samples."""
    n = len(profile)
    k = max(1, int(n * fraction))
    left = float(np.median(profile[:k]))
    right = float(np.median(profile[-k:]))
    return {
        "base_level": (left + right) / 2.0,
        "left_base": left,
        "right_base": right,
        "misalignment_mm": abs(left - right),
    }


def ransac_baseplate(profile: np.ndarray, **kwargs: Any) -> dict[str, float]:
    """RANSAC fit of left and right base-plate segments.

    Contract: return the same dict keys as outer_median_baseplate.
    """
    raise NotImplementedError(
        "RANSAC base-plate fit is not implemented. "
        "Use outer_median_baseplate as the documented fallback "
        "(estimate_baseplate(..., method='outer_median'))."
    )


def estimate_baseplate(
    profile: np.ndarray,
    method: str = "outer_median",
    **kwargs: Any,
) -> dict[str, float]:
    """Dispatch. Default is the implemented fallback so callers can run offline."""
    if method == "ransac":
        return ransac_baseplate(profile, **kwargs)
    if method == "outer_median":
        return outer_median_baseplate(profile, **kwargs)
    raise ValueError(f"Unknown base-plate method: {method}")
