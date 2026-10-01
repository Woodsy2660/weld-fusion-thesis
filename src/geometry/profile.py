"""
Depth map to weld bead cross-section profile extraction.
Steps 2-4 of geometry pipeline.
"""
import numpy as np


def extract_profile(depth_map, y_position, smooth_kernel=5):
    """
    Extract a 1D height profile from a depth map at a given y position.

    Args:
        depth_map: 2D numpy array (H, W) in mm
        y_position: row index for the cross-section slice
        smooth_kernel: smoothing window size

    Returns:
        profile: 1D array of height values (mm)
    """
    raw_profile = depth_map[y_position, :].astype(np.float64)

    # Smooth to reduce noise
    kernel = np.ones(smooth_kernel) / smooth_kernel
    smoothed = np.convolve(raw_profile, kernel, mode="same")

    # Convert depth to height (camera distance - depth = surface height)
    # Invert so the weld bead appears as a peak above the baseline
    baseline = np.median(smoothed)
    height_profile = baseline - smoothed

    return height_profile


def measure_geometry(profile, pixel_pitch_mm=0.1):
    """
    Measure weld bead geometric features from a height profile.

    Args:
        profile: 1D height array in mm
        pixel_pitch_mm: lateral resolution (mm per pixel)

    Returns:
        dict with reinforcement_mm, undercut_mm, width_mm, misalignment_mm
    """
    # Detect base plate level (median of outer regions)
    n = len(profile)
    left_base = np.median(profile[:n // 5])
    right_base = np.median(profile[4 * n // 5:])
    base_level = (left_base + right_base) / 2

    # Reinforcement = max height above base
    reinforcement = np.max(profile) - base_level

    # Undercut = max depth below base (negative values at weld toes)
    undercut = base_level - np.min(profile)
    undercut = max(0, undercut)  # only count if below base

    # Width = distance between toe-to-toe (where profile crosses base level)
    above_base = profile > base_level
    if np.any(above_base):
        indices = np.where(above_base)[0]
        width = (indices[-1] - indices[0]) * pixel_pitch_mm
    else:
        width = 0.0

    # Misalignment = difference between left and right base plate levels
    misalignment = abs(left_base - right_base)

    return {
        "reinforcement_mm": float(reinforcement),
        "undercut_mm": float(undercut),
        "width_mm": float(width),
        "misalignment_mm": float(misalignment),
    }
