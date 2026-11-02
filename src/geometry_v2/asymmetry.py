"""
Plate asymmetry extraction from depth band data.

Asymmetry is the absolute difference between left and right plate heights,
measured from the weld centerline. High asymmetry indicates either:
  - Lack of fusion (LOF): one-sided heat input tilts the plate
  - Linear misalignment: plates were not aligned before welding

The depth sensor cannot distinguish between these causes, so both map to
the compound focal element {L,U}.

Usage:
    from src.geometry_v2.asymmetry import extract_asymmetry, AsymmetryMeasurement

    # From pre-computed band stats (configs/weld_selections.json extended)
    measurement = extract_asymmetry_from_stats(coupon_id, band_stats)

    # From raw depth data and weld line
    measurement = extract_asymmetry_from_depth(depth_mm, weld_line, color_shape)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AsymmetryMeasurement:
    """Plate asymmetry measurement for a single coupon.

    Attributes:
        coupon_id: Coupon identifier (e.g., "C01")
        asymmetry_mm: Absolute difference between left and right plate heights
        left_plate_mm: Median depth of left plate region (relative to reference)
        right_plate_mm: Median depth of right plate region
        crown_mm: Median depth at weld crown (center)
        n_valid_samples: Number of valid depth samples used
        is_valid: Whether measurement has sufficient valid samples
    """
    coupon_id: str
    asymmetry_mm: float
    left_plate_mm: float
    right_plate_mm: float
    crown_mm: float
    n_valid_samples: int
    is_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "coupon_id": self.coupon_id,
            "asymmetry_mm": self.asymmetry_mm,
            "left_plate_mm": self.left_plate_mm,
            "right_plate_mm": self.right_plate_mm,
            "crown_mm": self.crown_mm,
            "n_valid_samples": self.n_valid_samples,
            "is_valid": self.is_valid,
        }


def extract_asymmetry_from_band(
    band_depth: np.ndarray,
    y_coords: np.ndarray,
    coupon_id: str = "unknown",
    min_valid_fraction: float = 0.5,
) -> AsymmetryMeasurement:
    """
    Extract plate asymmetry from a 2D depth band array.

    Args:
        band_depth: 2D array (n_perp, n_along) of depth values in mm
        y_coords: 1D array of perpendicular distances from centerline (mm)
        coupon_id: Identifier for this coupon
        min_valid_fraction: Minimum fraction of valid (non-NaN) values required

    Returns:
        AsymmetryMeasurement with plate asymmetry and related statistics
    """
    # Define regions (same as extract_weld_band.py)
    center_mask = np.abs(y_coords) < 3          # Crown: ±3mm
    left_plate_mask = y_coords < -12            # Left plate: < -12mm
    right_plate_mask = y_coords > 12            # Right plate: > +12mm

    # Extract regions
    center_data = band_depth[center_mask, :]
    left_plate_data = band_depth[left_plate_mask, :]
    right_plate_data = band_depth[right_plate_mask, :]

    # Filter invalid values (sensor dropouts at ~-290mm)
    valid_range = (-50, 50)  # Valid depth range in mm from reference
    center_valid = center_data[(center_data > valid_range[0]) & (center_data < valid_range[1])]
    left_valid = left_plate_data[(left_plate_data > valid_range[0]) & (left_plate_data < valid_range[1])]
    right_valid = right_plate_data[(right_plate_data > valid_range[0]) & (right_plate_data < valid_range[1])]

    # Check valid fraction
    total_expected = band_depth.size
    total_valid = len(center_valid) + len(left_valid) + len(right_valid)
    valid_fraction = total_valid / total_expected if total_expected > 0 else 0

    if valid_fraction < min_valid_fraction or len(left_valid) < 10 or len(right_valid) < 10:
        return AsymmetryMeasurement(
            coupon_id=coupon_id,
            asymmetry_mm=float("nan"),
            left_plate_mm=float("nan"),
            right_plate_mm=float("nan"),
            crown_mm=float("nan"),
            n_valid_samples=total_valid,
            is_valid=False,
        )

    # Compute medians (robust to outliers)
    left_median = float(np.median(left_valid))
    right_median = float(np.median(right_valid))
    crown_median = float(np.median(center_valid)) if len(center_valid) > 0 else float("nan")

    # Asymmetry is absolute difference
    asymmetry = abs(left_median - right_median)

    return AsymmetryMeasurement(
        coupon_id=coupon_id,
        asymmetry_mm=asymmetry,
        left_plate_mm=left_median,
        right_plate_mm=right_median,
        crown_mm=crown_median,
        n_valid_samples=total_valid,
        is_valid=True,
    )


def extract_asymmetry_from_depth(
    depth_mm: np.ndarray,
    weld_line: tuple[tuple[int, int], tuple[int, int]],
    color_shape: tuple[int, int, int],
    coupon_id: str = "unknown",
    band_half_width_mm: float = 15.0,
) -> AsymmetryMeasurement:
    """
    Extract asymmetry directly from depth image and weld line.

    Args:
        depth_mm: 2D depth array in mm
        weld_line: ((x1, y1), (x2, y2)) in color image coordinates
        color_shape: Shape of color image (H, W, C) for coordinate scaling
        coupon_id: Identifier for this coupon
        band_half_width_mm: Half-width of band to extract (mm)

    Returns:
        AsymmetryMeasurement
    """
    (lx1, ly1), (lx2, ly2) = weld_line

    dx = lx2 - lx1
    dy = ly2 - ly1
    length = np.sqrt(dx**2 + dy**2)

    # Unit vectors
    perp = np.array([-dy, dx]) / length

    # Scale factors (color to depth)
    scale_x = depth_mm.shape[1] / color_shape[1]
    scale_y = depth_mm.shape[0] / color_shape[0]

    # Pixel size in mm (approximate from camera setup)
    px_to_mm = 0.25

    # Band dimensions
    band_half_width_px = int(band_half_width_mm / px_to_mm)
    n_along = int(length)

    # Create output array
    band_depth = np.full((2 * band_half_width_px + 1, n_along), np.nan)
    y_coords = np.linspace(-band_half_width_mm, band_half_width_mm, 2 * band_half_width_px + 1)

    # Extract depth values
    for i_along in range(n_along):
        t = i_along / n_along
        cx = lx1 + t * dx
        cy = ly1 + t * dy

        for i_perp, offset_px in enumerate(range(-band_half_width_px, band_half_width_px + 1)):
            px = int(cx + offset_px * perp[0])
            py = int(cy + offset_px * perp[1])

            dpx = int(px * scale_x)
            dpy = int(py * scale_y)

            if 0 <= dpx < depth_mm.shape[1] and 0 <= dpy < depth_mm.shape[0]:
                d = depth_mm[dpy, dpx]
                if 0 < d < 1000:  # Filter invalid sensor readings
                    band_depth[i_perp, i_along] = d

    return extract_asymmetry_from_band(band_depth, y_coords, coupon_id)


def load_precomputed_asymmetry(coupon_id: str) -> AsymmetryMeasurement | None:
    """
    Load pre-computed asymmetry from band analysis results.

    The band extraction script (scripts/extract_weld_band.py) computes
    asymmetry values that are displayed in the summary output. This function
    retrieves those values from the static mapping.

    Returns None if coupon not found.
    """
    # Pre-computed from running extract_weld_band.py on all 15 coupons
    # These values are from the summary output
    precomputed = {
        'C01': {'asymmetry': 4.62, 'crown': -0.76},
        'C02': {'asymmetry': 2.91, 'crown': -0.82},
        'C03': {'asymmetry': 1.78, 'crown': -0.69},
        'C04': {'asymmetry': 0.74, 'crown': 1.95},
        'C05': {'asymmetry': 1.60, 'crown': -1.33},
        'C06': {'asymmetry': 1.41, 'crown': 0.48},
        'C07': {'asymmetry': 5.31, 'crown': -1.31},
        'C08': {'asymmetry': 4.36, 'crown': -2.30},
        'C09': {'asymmetry': 4.65, 'crown': 0.42},
        'C10': {'asymmetry': 4.31, 'crown': -0.93},
        'C11': {'asymmetry': 0.74, 'crown': -1.02},
        'C12': {'asymmetry': 0.62, 'crown': -0.33},
        'C13': {'asymmetry': 4.41, 'crown': -1.42},
        'C14': {'asymmetry': 3.72, 'crown': -1.13},
        'C15': {'asymmetry': 3.35, 'crown': -0.57}
    }

    if coupon_id not in precomputed:
        return None

    stats = precomputed[coupon_id]
    return AsymmetryMeasurement(
        coupon_id=coupon_id,
        asymmetry_mm=stats['asymmetry'],
        left_plate_mm=float("nan"),  # Not stored in summary
        right_plate_mm=float("nan"),
        crown_mm=stats['crown'],
        n_valid_samples=1000,  # Approximate
        is_valid=True,
    )


def load_all_asymmetry() -> dict[str, AsymmetryMeasurement]:
    """Load asymmetry measurements for all coupons."""
    coupons = [f"C{i:02d}" for i in range(1, 16)]
    result = {}
    for coupon_id in coupons:
        meas = load_precomputed_asymmetry(coupon_id)
        if meas is not None:
            result[coupon_id] = meas
    return result
