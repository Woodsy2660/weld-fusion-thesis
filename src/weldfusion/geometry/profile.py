"""Multi-slice, temporally averaged weld-bead profile extraction.

Single-row extraction discarded independent averaging that reduces D435
noise as sigma/sqrt(N). This replaces it: average N profile rows and,
when a temporal stack is provided, average T frames first.
"""

from __future__ import annotations

import numpy as np

from weldfusion.utils.config import load_named


def extract_profile(
    depth_map: np.ndarray,
    y_positions: int | list[int] | np.ndarray,
    *,
    smooth_kernel: int | None = None,
) -> np.ndarray:
    """Return a 1D height profile (mm), bead as a peak above the baseline.

    Args:
        depth_map: (H, W) in mm, or (T, H, W) temporal stack in mm.
        y_positions: row index or sequence of row indices to average.
        smooth_kernel: box-filter width; default from geometry.yaml.
    """
    if smooth_kernel is None:
        smooth_kernel = int(
            load_named("geometry")["profile_extraction"]["smoothing_kernel_size"]
        )

    arr = np.asarray(depth_map, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"depth_map must be (H,W) or (T,H,W), got shape {arr.shape}")

    temporal_mean = arr.mean(axis=0)  # (H, W)
    if isinstance(y_positions, (int, np.integer)):
        rows = temporal_mean[[int(y_positions)], :]
    else:
        idx = np.asarray(list(y_positions), dtype=int)
        if idx.size == 0:
            raise ValueError("y_positions must contain at least one row")
        rows = temporal_mean[idx, :]

    profile = rows.mean(axis=0)
    if smooth_kernel and smooth_kernel > 1:
        kernel = np.ones(smooth_kernel, dtype=np.float64) / float(smooth_kernel)
        profile = np.convolve(profile, kernel, mode="same")

    baseline = float(np.median(profile))
    return baseline - profile
