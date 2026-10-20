"""
Plate isolation using height-above-support masking.

Task 1b Part B1: Create plate mask from height above support, not distance from camera.

Key insight: The plate sits on a support board. The plate top is at h ≈ 10mm above
the board, while the board is at h ≈ 0. A threshold at h > t/2 = 5mm separates them
with margin on both sides.

SCOPE: this module produces a plate MASK and nothing else. The board model is a
single constant depth (RANSAC over a scalar), not a plane -- adequate for a
5 mm threshold at the measured 1.5-2.8 deg tilt, and deliberately not a datum.
No measured height may reference the board; the plate-only datum lives in
src/geometry/measure.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    HAS_PYREALSENSE = True
except ImportError:
    HAS_PYREALSENSE = False


@dataclass
class PlateRegion:
    """Detected plate region with bounding box."""
    mask: np.ndarray           # Boolean mask in depth frame
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) in depth frame
    z_board_mm: float          # Fitted board depth
    valid_border_pixels: int   # Number of valid pixels in border band
    method: str                # "height_threshold" or "largest_connected"

    # For colour frame
    mask_colour: np.ndarray | None = None
    bbox_colour: tuple[int, int, int, int] | None = None


def fit_board_depth_ransac(
    depth_mm: np.ndarray,
    border_fraction: float = 0.10,
    ransac_iterations: int = 100,
    inlier_threshold_mm: float = 2.0,
) -> tuple[float, int]:
    """Fit the board (support surface) as ONE CONSTANT DEPTH using RANSAC over
    the depths of border pixels. Not a plane: there is no normal and no tilt.

    The border band is the outer fraction of the image, where we expect to see
    the support board rather than the plate.

    Returns:
        z_board_mm: The fitted board depth
        n_valid: Number of valid border pixels
    """
    h, w = depth_mm.shape
    border_h = int(h * border_fraction)
    border_w = int(w * border_fraction)

    # Create border mask
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_h, :] = True        # Top
    border_mask[-border_h:, :] = True       # Bottom
    border_mask[:, :border_w] = True        # Left
    border_mask[:, -border_w:] = True       # Right

    # Get valid depth values in border
    valid = ~np.isnan(depth_mm) & border_mask
    border_depths = depth_mm[valid]

    n_valid = len(border_depths)
    if n_valid < 100:
        # Not enough valid border pixels
        return float('nan'), n_valid

    # RANSAC over a scalar: the dominant constant depth, not a plane
    best_depth = np.median(border_depths)
    best_inliers = 0

    rng = np.random.default_rng(42)
    for _ in range(ransac_iterations):
        # Sample a single point
        idx = rng.integers(0, n_valid)
        candidate_depth = border_depths[idx]

        # Count inliers
        inliers = np.sum(np.abs(border_depths - candidate_depth) < inlier_threshold_mm)

        if inliers > best_inliers:
            best_inliers = inliers
            best_depth = candidate_depth

    # Refine with mean of inliers
    inlier_mask = np.abs(border_depths - best_depth) < inlier_threshold_mm
    z_board = np.mean(border_depths[inlier_mask])

    return z_board, n_valid


def create_plate_mask_from_height(
    depth_mm: np.ndarray,
    plate_thickness_mm: float = 10.0,
    border_fraction: float = 0.10,
    min_border_pixels: int = 5000,
) -> PlateRegion:
    """Create plate mask based on height above support surface.

    The test is h > t/2 where:
    - h = z_board - z (height above board)
    - t = plate thickness (10 mm)
    - t/2 = 5 mm threshold

    Args:
        depth_mm: Averaged depth frame in mm
        plate_thickness_mm: Plate thickness (default 10 mm)
        border_fraction: Fraction of image to use for border band
        min_border_pixels: Minimum valid pixels required in border

    Returns:
        PlateRegion with mask and metadata
    """
    h, w = depth_mm.shape

    # Fit board depth from border
    z_board, n_valid_border = fit_board_depth_ransac(
        depth_mm, border_fraction=border_fraction
    )

    height_threshold = plate_thickness_mm / 2  # 5 mm

    if n_valid_border < min_border_pixels or np.isnan(z_board):
        # Fallback: largest connected component of valid depth
        valid = ~np.isnan(depth_mm)

        # Label connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            valid.astype(np.uint8), connectivity=8
        )

        # Find largest component (excluding background label 0)
        if n_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = labels == largest_label
        else:
            mask = valid

        # Compute bounding box
        rows, cols = np.where(mask)
        if len(rows) > 0:
            x0, y0 = cols.min(), rows.min()
            x1, y1 = cols.max() + 1, rows.max() + 1
        else:
            x0, y0, x1, y1 = 0, 0, w, h

        return PlateRegion(
            mask=mask,
            bbox=(x0, y0, x1, y1),
            z_board_mm=z_board if not np.isnan(z_board) else np.nanmedian(depth_mm),
            valid_border_pixels=n_valid_border,
            method="largest_connected",
        )

    # Height above support
    height = z_board - depth_mm

    # Plate mask: height > threshold AND valid depth
    valid = ~np.isnan(depth_mm)
    mask = valid & (height > height_threshold)

    # Take largest connected component
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    if n_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = labels == largest_label

    # Morphological close to bridge weld bead and spatter (3mm element)
    # At ~320mm standoff with fx~640, 3mm ≈ 6 pixels
    kernel_px = max(3, int(3 * 640 / 320))  # ~6 px
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    # Compute bounding box
    rows, cols = np.where(mask)
    if len(rows) > 0:
        x0, y0 = cols.min(), rows.min()
        x1, y1 = cols.max() + 1, rows.max() + 1
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    return PlateRegion(
        mask=mask,
        bbox=(x0, y0, x1, y1),
        z_board_mm=z_board,
        valid_border_pixels=n_valid_border,
        method="height_threshold",
    )


def add_padding_to_bbox(
    bbox: tuple[int, int, int, int],
    padding_mm: float,
    depth_mm: float,
    fx: float,
    img_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Add padding (in mm) to a bounding box.

    Args:
        bbox: (x0, y0, x1, y1) in pixels
        padding_mm: Padding to add in mm
        depth_mm: Approximate depth in mm
        fx: Focal length in pixels
        img_shape: (height, width) of image

    Returns:
        Padded (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = bbox
    h, w = img_shape

    # Convert mm to pixels at this depth
    padding_px = int(padding_mm * fx / depth_mm)

    x0 = max(0, x0 - padding_px)
    y0 = max(0, y0 - padding_px)
    x1 = min(w, x1 + padding_px)
    y1 = min(h, y1 + padding_px)

    return x0, y0, x1, y1


def map_plate_region_to_colour(
    region: PlateRegion,
    depth_mm: np.ndarray,
    depth_intr: dict,
    colour_intr: dict,
    depth_to_colour_extr: dict,
) -> PlateRegion:
    """Map plate region from depth frame to colour frame.

    Updates region.mask_colour and region.bbox_colour.
    """
    from src.geometry.registration_v2 import (
        build_rs_intrinsics, build_rs_extrinsics,
        deproject_pixel_to_point, project_point_to_pixel,
        transform_point_to_point,
    )

    d_intr = build_rs_intrinsics(depth_intr)
    c_intr = build_rs_intrinsics(colour_intr)
    d2c_extr = build_rs_extrinsics(depth_to_colour_extr)

    colour_mask = np.zeros((c_intr.height, c_intr.width), dtype=bool)

    # Project each depth pixel to colour
    rows, cols = np.where(region.mask)

    for r, c in zip(rows, cols):
        if np.isnan(depth_mm[r, c]):
            continue

        # Deproject depth pixel to 3D
        depth_m = depth_mm[r, c] / 1000.0
        point = deproject_pixel_to_point(d_intr, [c, r], depth_m)

        # Transform to colour frame
        point_colour = transform_point_to_point(d2c_extr, point)

        # Project to colour image
        pixel = project_point_to_pixel(c_intr, point_colour)
        u, v = int(round(pixel[0])), int(round(pixel[1]))

        if 0 <= v < c_intr.height and 0 <= u < c_intr.width:
            colour_mask[v, u] = True

    # Dilate to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    colour_mask = cv2.dilate(colour_mask.astype(np.uint8), kernel).astype(bool)

    # Compute colour bounding box
    rows_c, cols_c = np.where(colour_mask)
    if len(rows_c) > 0:
        x0_c, y0_c = cols_c.min(), rows_c.min()
        x1_c, y1_c = cols_c.max() + 1, rows_c.max() + 1
    else:
        x0_c, y0_c, x1_c, y1_c = 0, 0, c_intr.width, c_intr.height

    region.mask_colour = colour_mask
    region.bbox_colour = (x0_c, y0_c, x1_c, y1_c)

    return region
