"""
Datum hole detection, pose solve, and cell boundary computation.

Registration converts pixel coordinates to the coupon coordinate system:
    s: along-weld axis (mm), origin at machined end face
    y: across-weld axis (mm), origin at weld centreline

The pose is determined from two datum holes drilled in the coupon:
    - Origin hole (Ø6): s=15mm, y=+50mm
    - Axis hole (Ø8): s=285mm, y=+50mm

The different diameters break end-for-end ambiguity.

Cross-check: plate edges are detected independently. If the edge-derived
and hole-derived origins disagree by more than the tolerance, the coupon
is flagged for review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "geometry.yaml"


@dataclass
class DatumHole:
    """Detected datum hole in pixel coordinates."""
    centre_px: tuple[float, float]  # (row, col) in image coordinates
    radius_px: float
    confidence: float               # Detection confidence [0, 1]
    estimated_diameter_mm: float    # Inferred from radius and scale


@dataclass
class CouponPose:
    """Pose of coupon in camera frame.

    Transforms pixel coordinates (row, col) to coupon coordinates (s, y).
    """
    # Translation: coupon origin in pixel coordinates
    origin_px: tuple[float, float]  # (row, col)

    # Rotation: angle from pixel x-axis to coupon s-axis (radians)
    theta_rad: float

    # Scale: mm per pixel (from datum hole separation)
    scale_mm_per_px: float

    # Quality metrics
    hole_separation_px: float       # Measured separation between holes
    hole_separation_mm: float       # Expected separation (from config)
    scale_error_pct: float          # Deviation from expected scale
    edge_crosscheck_error_mm: float # Disagreement with edge detection

    # Raw detections
    origin_hole: DatumHole
    axis_hole: DatumHole

    def px_to_coupon(self, row: float, col: float) -> tuple[float, float]:
        """Convert pixel coordinates to coupon coordinates (s, y)."""
        # Translate to origin
        dr = row - self.origin_px[0]
        dc = col - self.origin_px[1]

        # Rotate to align with coupon axes
        cos_t = math.cos(-self.theta_rad)
        sin_t = math.sin(-self.theta_rad)
        ds = dc * cos_t - dr * sin_t
        dy = dc * sin_t + dr * cos_t

        # Scale to mm
        s_mm = ds * self.scale_mm_per_px
        y_mm = dy * self.scale_mm_per_px

        return s_mm, y_mm

    def coupon_to_px(self, s_mm: float, y_mm: float) -> tuple[float, float]:
        """Convert coupon coordinates to pixel coordinates (row, col)."""
        # Scale to pixels
        ds = s_mm / self.scale_mm_per_px
        dy = y_mm / self.scale_mm_per_px

        # Rotate back
        cos_t = math.cos(self.theta_rad)
        sin_t = math.sin(self.theta_rad)
        dc = ds * cos_t - dy * sin_t
        dr = ds * sin_t + dy * cos_t

        # Translate from origin
        row = dr + self.origin_px[0]
        col = dc + self.origin_px[1]

        return row, col

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_px": self.origin_px,
            "theta_rad": self.theta_rad,
            "theta_deg": math.degrees(self.theta_rad),
            "scale_mm_per_px": self.scale_mm_per_px,
            "hole_separation_px": self.hole_separation_px,
            "hole_separation_mm": self.hole_separation_mm,
            "scale_error_pct": self.scale_error_pct,
            "edge_crosscheck_error_mm": self.edge_crosscheck_error_mm,
        }


@dataclass
class CellBounds:
    """Boundaries of a single cell in both coordinate systems."""
    cell_id: str
    cell_index: int              # 0-based index within coupon

    # Coupon coordinates (mm)
    s_start_mm: float
    s_end_mm: float
    s_centre_mm: float

    # Pixel coordinates (for extraction)
    col_start_px: int
    col_end_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cell_index": self.cell_index,
            "s_start_mm": self.s_start_mm,
            "s_end_mm": self.s_end_mm,
            "s_centre_mm": self.s_centre_mm,
            "col_start_px": self.col_start_px,
            "col_end_px": self.col_end_px,
        }


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load geometry configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_holes_in_depth(
    depth_mm: np.ndarray,
    config: dict | None = None,
) -> list[DatumHole]:
    """
    Detect datum holes in a depth image.

    Holes appear as regions with no valid depth (through-holes) or
    significantly different depth (if there's a backing surface).

    Args:
        depth_mm: Depth image in mm. Invalid pixels are NaN.
        config: Geometry config dict.

    Returns:
        List of detected holes, sorted by x-coordinate (left to right).
    """
    config = config or load_config()
    reg_cfg = config["registration"]

    # Expected hole diameters
    d_origin = reg_cfg["origin_hole"]["diameter_mm"]
    d_axis = reg_cfg["axis_hole"]["diameter_mm"]

    # Estimate scale from sensor config (approximate)
    sensor_cfg = config["sensor"]
    standoff = sensor_cfg["working_distance_mm"]
    depth_w = sensor_cfg["depth_resolution"][0]
    # D435i horizontal FOV ~87° at 1280 width
    fov_rad = math.radians(87)
    approx_scale = 2 * standoff * math.tan(fov_rad / 2) / depth_w  # mm/px

    # Expected hole radii in pixels
    r_origin_px = (d_origin / 2) / approx_scale
    r_axis_px = (d_axis / 2) / approx_scale

    # Find invalid regions (holes are through-holes, so depth is invalid)
    invalid_mask = np.isnan(depth_mm)

    # Label connected components of invalid regions
    labeled, n_labels = ndimage.label(invalid_mask)

    holes = []
    for label_id in range(1, n_labels + 1):
        # Get region properties
        region = (labeled == label_id)
        coords = np.argwhere(region)
        if len(coords) < 10:
            continue  # Too small to be a hole

        # Compute centroid and approximate radius
        centre = coords.mean(axis=0)  # (row, col)
        distances = np.sqrt(((coords - centre) ** 2).sum(axis=1))
        radius_px = np.percentile(distances, 90)

        # Check if this matches expected hole sizes
        area_px = len(coords)
        expected_area_origin = math.pi * r_origin_px**2
        expected_area_axis = math.pi * r_axis_px**2

        # Compute confidence based on area match
        area_ratio_origin = area_px / expected_area_origin if expected_area_origin > 0 else 0
        area_ratio_axis = area_px / expected_area_axis if expected_area_axis > 0 else 0

        # Accept if area is within 50-200% of expected
        if 0.5 < area_ratio_origin < 2.0 or 0.5 < area_ratio_axis < 2.0:
            # Determine which hole this is more likely to be
            if abs(area_ratio_origin - 1.0) < abs(area_ratio_axis - 1.0):
                estimated_d = d_origin
                confidence = 1.0 - abs(area_ratio_origin - 1.0)
            else:
                estimated_d = d_axis
                confidence = 1.0 - abs(area_ratio_axis - 1.0)

            holes.append(DatumHole(
                centre_px=(float(centre[0]), float(centre[1])),
                radius_px=float(radius_px),
                confidence=float(max(0, min(1, confidence))),
                estimated_diameter_mm=estimated_d,
            ))

    # Sort by column (left to right = low s to high s)
    holes.sort(key=lambda h: h.centre_px[1])

    return holes


def detect_plate_edges(
    depth_mm: np.ndarray,
    config: dict | None = None,
) -> tuple[float, float]:
    """
    Detect plate edges for cross-check.

    Returns the column positions (in pixels) of the left and right plate edges.
    """
    # Find columns where valid depth exists
    valid_mask = ~np.isnan(depth_mm)
    col_valid = valid_mask.any(axis=0)

    valid_cols = np.where(col_valid)[0]
    if len(valid_cols) < 10:
        return float("nan"), float("nan")

    left_edge = float(valid_cols[0])
    right_edge = float(valid_cols[-1])

    return left_edge, right_edge


def solve_pose(
    holes: list[DatumHole],
    config: dict | None = None,
    plate_edges: tuple[float, float] | None = None,
) -> CouponPose:
    """
    Solve coupon pose from detected datum holes.

    Requires exactly 2 holes. The smaller hole (Ø6) is the origin;
    the larger hole (Ø8) defines the s-axis direction.

    Args:
        holes: Detected datum holes (must be exactly 2).
        config: Geometry config dict.
        plate_edges: Optional (left, right) edge positions for cross-check.

    Returns:
        CouponPose for coordinate transformation.

    Raises:
        ValueError: If not exactly 2 holes detected.
    """
    config = config or load_config()
    reg_cfg = config["registration"]
    min_conf = reg_cfg["min_hole_confidence"]

    # Filter by confidence
    confident_holes = [h for h in holes if h.confidence >= min_conf]
    if len(confident_holes) != 2:
        raise ValueError(
            f"Expected 2 confident holes, found {len(confident_holes)}. "
            f"All holes: {[(h.estimated_diameter_mm, h.confidence) for h in holes]}"
        )

    # Identify origin (smaller) and axis (larger) holes
    if confident_holes[0].estimated_diameter_mm < confident_holes[1].estimated_diameter_mm:
        origin_hole, axis_hole = confident_holes[0], confident_holes[1]
    else:
        origin_hole, axis_hole = confident_holes[1], confident_holes[0]

    # Compute pixel separation
    dr = axis_hole.centre_px[0] - origin_hole.centre_px[0]
    dc = axis_hole.centre_px[1] - origin_hole.centre_px[1]
    separation_px = math.sqrt(dr**2 + dc**2)

    # Expected separation in mm
    s_origin = reg_cfg["origin_hole"]["s_mm"]
    s_axis = reg_cfg["axis_hole"]["s_mm"]
    separation_mm = s_axis - s_origin

    # Scale
    scale = separation_mm / separation_px if separation_px > 0 else 0.0

    # Angle (from pixel x-axis to coupon s-axis)
    theta = math.atan2(dr, dc)

    # Scale error
    sensor_cfg = config["sensor"]
    standoff = sensor_cfg["working_distance_mm"]
    depth_w = sensor_cfg["depth_resolution"][0]
    fov_rad = math.radians(87)
    expected_scale = 2 * standoff * math.tan(fov_rad / 2) / depth_w
    scale_error = 100 * (scale - expected_scale) / expected_scale if expected_scale > 0 else 0

    # Edge cross-check
    edge_error = float("nan")
    if plate_edges is not None and not any(math.isnan(e) for e in plate_edges):
        # Compute expected edge positions from pose
        # Left edge should be near s=0, right edge near s=coupon_length
        coupon_length = config["cells"]["coupon_length_mm"]
        left_expected_col = origin_hole.centre_px[1] - s_origin / scale if scale > 0 else 0
        right_expected_col = origin_hole.centre_px[1] + (coupon_length - s_origin) / scale if scale > 0 else 0

        left_error = abs(plate_edges[0] - left_expected_col) * scale
        right_error = abs(plate_edges[1] - right_expected_col) * scale
        edge_error = max(left_error, right_error)

    return CouponPose(
        origin_px=origin_hole.centre_px,
        theta_rad=theta,
        scale_mm_per_px=scale,
        hole_separation_px=separation_px,
        hole_separation_mm=separation_mm,
        scale_error_pct=scale_error,
        edge_crosscheck_error_mm=edge_error,
        origin_hole=origin_hole,
        axis_hole=axis_hole,
    )


def compute_cell_bounds(
    pose: CouponPose,
    coupon_id: str,
    config: dict | None = None,
) -> list[CellBounds]:
    """
    Compute cell boundaries in both coordinate systems.

    Excludes the first and last cells (arc start/stop transient zones).

    Args:
        pose: Solved coupon pose.
        coupon_id: Coupon identifier (e.g., "C07").
        config: Geometry config dict.

    Returns:
        List of CellBounds for all non-excluded cells.
    """
    config = config or load_config()
    cells_cfg = config["cells"]

    cell_width = cells_cfg["width_mm"]
    first_excluded = cells_cfg["first_excluded_mm"]
    last_excluded = cells_cfg["last_excluded_mm"]
    coupon_length = cells_cfg["coupon_length_mm"]

    # Compute cell boundaries
    s_start = first_excluded
    s_end = coupon_length - last_excluded
    n_cells = int((s_end - s_start) / cell_width)

    bounds = []
    for i in range(n_cells):
        s_cell_start = s_start + i * cell_width
        s_cell_end = s_cell_start + cell_width
        s_cell_centre = (s_cell_start + s_cell_end) / 2

        # Convert to pixel coordinates (using centre y=0 for column estimate)
        _, col_start = pose.coupon_to_px(s_cell_start, 0)
        _, col_end = pose.coupon_to_px(s_cell_end, 0)

        bounds.append(CellBounds(
            cell_id=f"{coupon_id}-{i:02d}",
            cell_index=i,
            s_start_mm=s_cell_start,
            s_end_mm=s_cell_end,
            s_centre_mm=s_cell_centre,
            col_start_px=int(round(col_start)),
            col_end_px=int(round(col_end)),
        ))

    return bounds


def register_coupon(
    depth_mm: np.ndarray,
    coupon_id: str,
    config: dict | None = None,
) -> tuple[CouponPose, list[CellBounds]]:
    """
    Full registration pipeline: detect holes, solve pose, compute cells.

    Args:
        depth_mm: Averaged depth image in mm. Invalid pixels are NaN.
        coupon_id: Coupon identifier.
        config: Geometry config dict.

    Returns:
        Tuple of (CouponPose, list of CellBounds).

    Raises:
        ValueError: If registration fails (holes not found, etc.).
    """
    config = config or load_config()

    # Detect datum holes
    holes = detect_holes_in_depth(depth_mm, config)

    # Detect plate edges for cross-check
    edges = detect_plate_edges(depth_mm, config)

    # Solve pose
    pose = solve_pose(holes, config, edges)

    # Check edge cross-check tolerance
    tolerance = config["registration"]["edge_crosscheck_tolerance_mm"]
    if not math.isnan(pose.edge_crosscheck_error_mm) and pose.edge_crosscheck_error_mm > tolerance:
        raise ValueError(
            f"Edge cross-check failed: error {pose.edge_crosscheck_error_mm:.2f}mm "
            f"> tolerance {tolerance:.2f}mm. Check datum holes or plate alignment."
        )

    # Compute cell bounds
    cells = compute_cell_bounds(pose, coupon_id, config)

    return pose, cells
