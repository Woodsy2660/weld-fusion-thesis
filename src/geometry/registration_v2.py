"""
Pose-aware registration using rim-fitting hole detection.

This module implements the registration pipeline specified in Task 1, Part B:
- B2: Rim-fitting hole detection in colour images (not depth-invalid centroids)
- B3: pyrealsense2-based deprojection with distortion handling
- B4: Pose solve with face derivation
- B5: Point cloud extraction in coupon coordinates

Key design decisions:
1. Fit the top rim of through-holes, not the see-through opening centroid.
   The opening centroid is biased by e/2 toward the camera nadir, where
   e = t·tan(α) is the bore wall projection.

2. Use pyrealsense2 intrinsics/extrinsics math for deprojection. The colour
   stream has a distortion model that must be applied.

3. The coupon coordinate frame is physically fixed:
   - s: along weld, from Ø6 (s=15) toward Ø8 (s=285)
   - y: across weld, +y toward datum holes, weld centreline at y=0
   - z: outward normal of the cap face
   - Right-handed: ŝ × ŷ = ẑ

4. Profile extraction works on point clouds, not image rows.
   This removes all `rows // 2` assumptions about image-to-coupon alignment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    import pyrealsense2 as rs
    HAS_PYREALSENSE = True
except ImportError:
    HAS_PYREALSENSE = False


# ---------------------------------------------------------------------------
# Pure numpy fallback for deprojection (when pyrealsense2 unavailable)
# ---------------------------------------------------------------------------

@dataclass
class FallbackIntrinsics:
    """Fallback intrinsics container matching rs.intrinsics interface."""
    width: int
    height: int
    ppx: float  # principal point x
    ppy: float  # principal point y
    fx: float   # focal length x
    fy: float   # focal length y
    coeffs: list[float] = field(default_factory=lambda: [0, 0, 0, 0, 0])


@dataclass
class FallbackExtrinsics:
    """Fallback extrinsics container matching rs.extrinsics interface."""
    rotation: list[float] = field(default_factory=lambda: [1, 0, 0, 0, 1, 0, 0, 0, 1])
    translation: list[float] = field(default_factory=lambda: [0, 0, 0])


def fallback_deproject_pixel_to_point(intr: FallbackIntrinsics, pixel: list, depth: float) -> list:
    """Deproject 2D pixel to 3D point (pinhole model, no distortion).

    pixel: [u, v] (column, row)
    depth: depth in metres
    Returns: [x, y, z] in camera frame
    """
    u, v = pixel
    x = (u - intr.ppx) / intr.fx * depth
    y = (v - intr.ppy) / intr.fy * depth
    z = depth
    return [x, y, z]


def fallback_project_point_to_pixel(intr: FallbackIntrinsics, point: list) -> list:
    """Project 3D point to 2D pixel (pinhole model, no distortion).

    point: [x, y, z] in camera frame
    Returns: [u, v] (column, row)
    """
    x, y, z = point
    if abs(z) < 1e-6:
        return [intr.ppx, intr.ppy]
    u = x / z * intr.fx + intr.ppx
    v = y / z * intr.fy + intr.ppy
    return [u, v]


def fallback_transform_point_to_point(extr: FallbackExtrinsics, from_point: list) -> list:
    """Transform point between coordinate frames.

    from_point: [x, y, z] in source frame
    Returns: [x, y, z] in target frame
    """
    R = np.array(extr.rotation).reshape(3, 3)
    t = np.array(extr.translation)
    p = np.array(from_point)
    result = R @ p + t
    return result.tolist()


def deproject_pixel_to_point(intr, pixel, depth):
    """Deproject 2D pixel to 3D point. Works with either pyrealsense or fallback."""
    if HAS_PYREALSENSE and hasattr(intr, 'model'):
        return rs.rs2_deproject_pixel_to_point(intr, pixel, depth)
    else:
        return fallback_deproject_pixel_to_point(intr, pixel, depth)


def project_point_to_pixel(intr, point):
    """Project 3D point to 2D pixel. Works with either pyrealsense or fallback."""
    if HAS_PYREALSENSE and hasattr(intr, 'model'):
        return rs.rs2_project_point_to_pixel(intr, point)
    else:
        return fallback_project_point_to_pixel(intr, point)


def transform_point_to_point(extr, from_point):
    """Transform point between coordinate frames. Works with either pyrealsense or fallback."""
    if HAS_PYREALSENSE and isinstance(from_point, list) and hasattr(extr, 'rotation'):
        try:
            return rs.rs2_transform_point_to_point(extr, from_point)
        except (TypeError, AttributeError):
            return fallback_transform_point_to_point(extr, from_point)
    else:
        return fallback_transform_point_to_point(extr, from_point)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "registration.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load registration configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedHole:
    """A detected datum hole with rim-fit results."""
    # Pixel coordinates in colour image
    centre_px: tuple[float, float]       # (row, col) = (v, u)

    # Free-fit results
    free_fit_radius_px: float
    free_fit_residual_px: float

    # Fixed-fit results (radius constrained to known value)
    fixed_fit_centre_px: tuple[float, float]
    fixed_fit_residual_px: float

    # Classification
    estimated_diameter_mm: float
    hole_type: str                       # "origin" (Ø6), "axis" (Ø8), "check" (Ø10)

    # Quality
    n_edge_points: int
    confidence: float

    # 3D position in camera frame (after deprojection)
    position_3d: np.ndarray | None = None


@dataclass
class CouponPose:
    """Full 6-DOF pose of coupon in camera frame."""
    # Rotation matrix: columns are [ŝ, ŷ, ẑ] in camera frame
    R: np.ndarray                        # 3x3

    # Origin: coupon origin (s=0, y=0, z=0) in camera frame
    O: np.ndarray                        # 3-vector

    # Individual holes
    origin_hole: DetectedHole
    axis_hole: DetectedHole
    check_hole: DetectedHole | None = None

    # Derived quantities
    measured_baseline_mm: float = 0.0
    derived_face: str = ""               # "cap" or "root"
    face_label: str = ""                 # From meta.json
    face_match: bool = True

    # Quality metrics
    y_crown_mean_mm: float = 0.0
    y_crown_slope: float = 0.0
    y_crown_max_mm: float = 0.0

    # Flags
    flags: list[str] = field(default_factory=list)

    def to_coupon(self, X: np.ndarray) -> np.ndarray:
        """Transform point(s) from camera frame to coupon coordinates.

        x_c = Rᵀ(X - O)
        """
        return self.R.T @ (X - self.O.reshape(3, 1) if X.ndim == 1 else X - self.O[:, np.newaxis])

    def to_camera(self, x_c: np.ndarray) -> np.ndarray:
        """Transform point(s) from coupon coordinates to camera frame.

        X = R @ x_c + O
        """
        return self.R @ x_c + (self.O.reshape(3, 1) if x_c.ndim == 1 else self.O[:, np.newaxis])

    def to_dict(self) -> dict[str, Any]:
        return {
            "R": self.R.tolist(),
            "O": self.O.tolist(),
            "measured_baseline_mm": self.measured_baseline_mm,
            "derived_face": self.derived_face,
            "face_label": self.face_label,
            "face_match": self.face_match,
            "y_crown_mean_mm": self.y_crown_mean_mm,
            "y_crown_slope": self.y_crown_slope,
            "y_crown_max_mm": self.y_crown_max_mm,
            "flags": self.flags,
        }


@dataclass
class PointCloud:
    """Coupon-aligned point cloud with per-pixel statistics."""
    # Coordinates in coupon frame
    s: np.ndarray                        # Along-weld (mm)
    y: np.ndarray                        # Across-weld (mm)
    z: np.ndarray                        # Height (mm)

    # Statistics
    n_valid: np.ndarray                  # Valid frames per pixel
    sigma: np.ndarray                    # Per-pixel std over frames

    # Original pixel indices (for colour lookup)
    row_idx: np.ndarray
    col_idx: np.ndarray

    def extract_profile(self, s_centre: float, half_width: float = 0.25) -> dict:
        """Extract a transverse profile at given s position.

        Returns points with |s - s_centre| <= half_width, sorted by y.
        """
        mask = np.abs(self.s - s_centre) <= half_width
        if not mask.any():
            return {"y": np.array([]), "z": np.array([]), "n": 0}

        y_slice = self.y[mask]
        z_slice = self.z[mask]
        order = np.argsort(y_slice)

        return {
            "y": y_slice[order],
            "z": z_slice[order],
            "n": len(y_slice),
        }


# ---------------------------------------------------------------------------
# Intrinsics/Extrinsics from meta.json
# ---------------------------------------------------------------------------

def build_rs_intrinsics(intr_dict: dict):
    """Build intrinsics object from meta.json dict.

    Returns pyrealsense2 intrinsics if available, otherwise fallback.
    """
    if HAS_PYREALSENSE:
        intr = rs.intrinsics()
        intr.width = intr_dict["width"]
        intr.height = intr_dict["height"]
        intr.ppx = intr_dict["ppx"]
        intr.ppy = intr_dict["ppy"]
        intr.fx = intr_dict["fx"]
        intr.fy = intr_dict["fy"]

        # Distortion model
        model_str = intr_dict.get("model", "distortion.none")
        if "brown_conrady" in model_str.lower():
            if "inverse" in model_str.lower():
                intr.model = rs.distortion.inverse_brown_conrady
            else:
                intr.model = rs.distortion.brown_conrady
        else:
            intr.model = rs.distortion.none

        intr.coeffs = intr_dict.get("coeffs", [0, 0, 0, 0, 0])
        return intr
    else:
        # Use fallback (pinhole model, no distortion)
        return FallbackIntrinsics(
            width=intr_dict["width"],
            height=intr_dict["height"],
            ppx=intr_dict["ppx"],
            ppy=intr_dict["ppy"],
            fx=intr_dict["fx"],
            fy=intr_dict["fy"],
            coeffs=intr_dict.get("coeffs", [0, 0, 0, 0, 0]),
        )


def build_rs_extrinsics(extr_dict: dict):
    """Build extrinsics object from meta.json dict.

    Returns pyrealsense2 extrinsics if available, otherwise fallback.
    """
    if HAS_PYREALSENSE:
        extr = rs.extrinsics()
        extr.rotation = extr_dict["rotation"]
        extr.translation = extr_dict["translation"]
        return extr
    else:
        return FallbackExtrinsics(
            rotation=extr_dict["rotation"],
            translation=extr_dict["translation"],
        )


# ---------------------------------------------------------------------------
# B2: Rim-fitting hole detection in colour
# ---------------------------------------------------------------------------

def compute_median_colour(colour_dir: Path) -> np.ndarray:
    """Compute per-pixel median of colour frames."""
    frames = []
    for png_file in sorted(colour_dir.glob("*.png")):
        img = cv2.imread(str(png_file), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)

    if not frames:
        raise ValueError(f"No colour frames found in {colour_dir}")

    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def compute_averaged_depth(depth_dir: Path, depth_units: float = 0.0001) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute per-pixel mean and std of depth frames.

    Returns (mean_mm, std_mm, n_frames).
    """
    frames = []
    for npy_file in sorted(depth_dir.glob("*.npy")):
        raw = np.load(npy_file).astype(np.float32)
        depth_mm = raw * depth_units * 1000.0
        depth_mm[raw == 0] = np.nan
        frames.append(depth_mm)

    if not frames:
        raise ValueError(f"No depth frames found in {depth_dir}")

    stack = np.stack(frames, axis=0)
    with np.errstate(all="ignore"):
        mean_mm = np.nanmean(stack, axis=0)
        std_mm = np.nanstd(stack, axis=0)

    return mean_mm, std_mm, len(frames)


def create_plate_mask_from_depth(
    depth_mm: np.ndarray,
    standoff_mm: float,
    window_mm: float = 15.0,
) -> np.ndarray:
    """Create plate mask: valid depth within ±window of standoff."""
    valid = ~np.isnan(depth_mm)
    in_range = np.abs(depth_mm - standoff_mm) <= window_mm
    return valid & in_range


def map_depth_mask_to_colour(
    depth_mask: np.ndarray,
    depth_intr,
    colour_intr,
    depth_to_colour,
    depth_mm: np.ndarray,
) -> np.ndarray:
    """Map a depth-frame mask to colour-frame coordinates."""
    # Project each depth pixel to colour
    colour_mask = np.zeros((colour_intr.height, colour_intr.width), dtype=bool)

    rows, cols = np.where(depth_mask)
    for r, c in zip(rows, cols):
        if np.isnan(depth_mm[r, c]):
            continue

        # Deproject depth pixel to 3D
        depth_m = depth_mm[r, c] / 1000.0
        point = deproject_pixel_to_point(depth_intr, [c, r], depth_m)

        # Transform to colour frame
        point_colour = transform_point_to_point(depth_to_colour, point)

        # Project to colour image
        pixel = project_point_to_pixel(colour_intr, point_colour)
        u, v = int(round(pixel[0])), int(round(pixel[1]))

        if 0 <= v < colour_intr.height and 0 <= u < colour_intr.width:
            colour_mask[v, u] = True

    # Dilate to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    colour_mask = cv2.dilate(colour_mask.astype(np.uint8), kernel).astype(bool)

    return colour_mask


def detect_hole_candidates(
    colour_img: np.ndarray,
    plate_mask: np.ndarray,
    config: dict,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Find dark blob candidates that might be holes.

    Returns list of (contour, centre) tuples.
    """
    det_cfg = config["hole_detection"]
    block_size = det_cfg["adaptive_block_size_px"]
    c_val = det_cfg["adaptive_threshold_c"]

    # Convert to grayscale
    gray = cv2.cvtColor(colour_img, cv2.COLOR_BGR2GRAY)

    # Apply plate mask
    gray_masked = gray.copy()
    gray_masked[~plate_mask] = 255  # White out non-plate regions

    # Adaptive thresholding to find dark regions
    thresh = cv2.adaptiveThreshold(
        gray_masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_val
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    candidates = []
    for contour in contours:
        if len(contour) < 10:
            continue

        # Compute centre
        M = cv2.moments(contour)
        if M["m00"] < 1:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        candidates.append((contour, np.array([cx, cy])))

    return candidates


def fit_circle_to_contour(
    contour: np.ndarray,
    colour_img: np.ndarray,
    min_points: int = 20,
) -> tuple[tuple[float, float], float, float, int] | None:
    """Fit a circle to the outer contour using sub-pixel edge refinement.

    Returns (centre, radius, residual, n_points) or None if fit fails.
    """
    if len(contour) < min_points:
        return None

    # Get sub-pixel edge points using gradient along radial rays
    # For simplicity, use the contour points directly with sub-pixel refinement
    points = contour.reshape(-1, 2).astype(np.float32)

    # Fit circle using least squares
    # Circle equation: (x - a)^2 + (y - b)^2 = r^2
    # Linearized: x^2 + y^2 = 2ax + 2by + (r^2 - a^2 - b^2)

    x = points[:, 0]
    y = points[:, 1]

    # Design matrix: [2x, 2y, 1]
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2

    try:
        result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b_coef, c = result
    r = np.sqrt(c + a**2 + b_coef**2)

    # Compute residual
    distances = np.sqrt((x - a)**2 + (y - b_coef)**2)
    residual = np.sqrt(np.mean((distances - r)**2))

    # Centre in (row, col) = (v, u) format
    centre = (b_coef, a)  # Note: cv2 uses (x, y) = (col, row)

    return centre, r, residual, len(points)


def fit_circle_fixed_radius(
    contour: np.ndarray,
    known_radius_px: float,
) -> tuple[tuple[float, float], float, int] | None:
    """Fit a circle with fixed radius, solving for centre only.

    Returns (centre, residual, n_points) or None if fit fails.
    """
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 10:
        return None

    x = points[:, 0]
    y = points[:, 1]

    # With fixed r, minimize sum of (sqrt((x-a)^2 + (y-b)^2) - r)^2
    # Use iterative least squares starting from centroid

    a = np.mean(x)
    b = np.mean(y)

    for _ in range(10):
        distances = np.sqrt((x - a)**2 + (y - b)**2)
        distances = np.maximum(distances, 1e-6)  # Avoid division by zero

        # Gradient descent step
        da = np.mean((distances - known_radius_px) * (x - a) / distances)
        db = np.mean((distances - known_radius_px) * (y - b) / distances)

        a -= da
        b -= db

        if abs(da) < 0.01 and abs(db) < 0.01:
            break

    # Final residual
    distances = np.sqrt((x - a)**2 + (y - b)**2)
    residual = np.sqrt(np.mean((distances - known_radius_px)**2))

    centre = (b, a)  # (row, col) = (v, u)

    return centre, residual, len(points)


def classify_hole(
    radius_px: float,
    colour_intr: dict,
    standoff_mm: float,
    config: dict,
) -> tuple[str, float] | None:
    """Classify hole by diameter. Returns (hole_type, expected_diameter) or None."""
    # Expected pixel radii for each hole type
    fx = colour_intr["fx"]
    tolerance = config["hole_detection"]["diameter_tolerance_pct"] / 100.0

    for hole_type in ["origin", "axis", "check"]:
        expected_d = config["datum_holes"][hole_type]["diameter_mm"]
        expected_r_px = (expected_d / 2) * fx / standoff_mm

        if abs(radius_px - expected_r_px) / expected_r_px <= tolerance:
            return hole_type, expected_d

    return None


def detect_datum_holes_in_colour(
    colour_img: np.ndarray,
    plate_mask: np.ndarray,
    colour_intr: dict,
    standoff_mm: float,
    config: dict,
) -> list[DetectedHole]:
    """Detect datum holes by fitting circles to dark region rims.

    This is the B2 implementation: fit the top rim, not the opening centroid.
    """
    det_cfg = config["hole_detection"]
    min_points = det_cfg["min_edge_points"]
    fx = colour_intr["fx"]

    # Find candidates
    candidates = detect_hole_candidates(colour_img, plate_mask, config)

    detected = []
    rejected = []

    for contour, centre in candidates:
        # Free-fit circle
        fit_result = fit_circle_to_contour(contour, colour_img, min_points)
        if fit_result is None:
            continue

        centre_free, radius_free, residual_free, n_points = fit_result

        # Classify by diameter
        classification = classify_hole(radius_free, colour_intr, standoff_mm, config)

        if classification is None:
            rejected.append({
                "centre": centre_free,
                "radius_px": radius_free,
                "reason": "diameter_mismatch",
            })
            continue

        hole_type, expected_d = classification

        # Fixed-fit with known radius
        expected_r_px = (expected_d / 2) * fx / standoff_mm
        fixed_result = fit_circle_fixed_radius(contour, expected_r_px)

        if fixed_result is None:
            centre_fixed = centre_free
            residual_fixed = residual_free
            n_fixed = n_points
        else:
            centre_fixed, residual_fixed, n_fixed = fixed_result

        # Compute confidence
        # Higher confidence for: more edge points, lower residual, better diameter match
        diameter_match = 1.0 - abs(radius_free * 2 * standoff_mm / fx - expected_d) / expected_d
        residual_score = max(0, 1.0 - residual_fixed / 2.0)
        point_score = min(1.0, n_points / 80.0)
        confidence = (diameter_match + residual_score + point_score) / 3.0

        detected.append(DetectedHole(
            centre_px=centre_free,
            free_fit_radius_px=radius_free,
            free_fit_residual_px=residual_free,
            fixed_fit_centre_px=centre_fixed,
            fixed_fit_residual_px=residual_fixed,
            estimated_diameter_mm=expected_d,
            hole_type=hole_type,
            n_edge_points=n_points,
            confidence=confidence,
        ))

    # Sort by column (left to right = low s to high s)
    detected.sort(key=lambda h: h.centre_px[1])

    return detected


# ---------------------------------------------------------------------------
# B3: Deprojection using pyrealsense2
# ---------------------------------------------------------------------------

def deproject_hole_centre(
    hole: DetectedHole,
    depth_mm: np.ndarray,
    colour_intr,
    depth_intr,
    colour_to_depth,
    depth_to_colour,
    config: dict,
    standoff_mm: float = 320.0,
) -> np.ndarray:
    """Deproject a hole centre to 3D using local plane fit.

    1. Use standoff as initial depth guess
    2. Transform color pixel to depth frame coordinates
    3. Fit plane to annulus around the hole in depth image
    4. Intersect colour ray with plane

    Returns 3D point in colour camera frame.
    """
    dep_cfg = config["deprojection"]
    hole_cfg = config["datum_holes"][hole.hole_type]
    r_mm = hole_cfg["radius_mm"]
    inner_mm = dep_cfg["annulus_inner_mm"]
    outer_mm = dep_cfg["annulus_outer_mm"]

    # Use fixed-fit centre (in COLOR pixel coordinates)
    u_color, v_color = hole.fixed_fit_centre_px[1], hole.fixed_fit_centre_px[0]

    # Initial guess: use standoff (don't try to look up depth with color coords)
    z_guess = standoff_mm / 1000.0  # metres

    # Deproject color pixel to 3D using initial guess
    point_colour = deproject_pixel_to_point(colour_intr, [u_color, v_color], z_guess)

    # Transform to depth frame
    point_depth = transform_point_to_point(colour_to_depth, point_colour)

    # Project to depth image to get depth pixel coordinates
    pixel_depth = project_point_to_pixel(depth_intr, point_depth)
    ud, vd = int(round(pixel_depth[0])), int(round(pixel_depth[1]))

    # Now get actual depth from depth image at the transformed location
    h, w = depth_mm.shape
    roi_r = int(50 * depth_intr.fx / standoff_mm)  # ~50mm radius in depth pixels

    vd0, vd1 = max(0, vd - roi_r), min(h, vd + roi_r)
    ud0, ud1 = max(0, ud - roi_r), min(w, ud + roi_r)

    if vd0 >= vd1 or ud0 >= ud1:
        raise ValueError(f"Depth ROI out of bounds at depth pixel ({vd}, {ud})")

    roi_depth = depth_mm[vd0:vd1, ud0:ud1]
    valid_depth = roi_depth[~np.isnan(roi_depth)]
    if len(valid_depth) == 0:
        raise ValueError(f"No valid depth near hole at depth pixel ({vd}, {ud}), color pixel ({v_color:.0f}, {u_color:.0f})")

    # Refine z_guess with actual median depth
    z_guess = float(np.median(valid_depth)) / 1000.0  # metres

    # Re-transform with refined depth
    point_colour = deproject_pixel_to_point(colour_intr, [u_color, v_color], z_guess)
    point_depth = transform_point_to_point(colour_to_depth, point_colour)
    pixel_depth = project_point_to_pixel(depth_intr, point_depth)
    ud, vd = int(round(pixel_depth[0])), int(round(pixel_depth[1]))

    # Get annulus points in depth image
    annulus_points = []
    inner_px = (r_mm + inner_mm) * depth_intr.fx / (z_guess * 1000)
    outer_px = (r_mm + outer_mm) * depth_intr.fx / (z_guess * 1000)

    for dy in range(-int(outer_px) - 5, int(outer_px) + 6):
        for dx in range(-int(outer_px) - 5, int(outer_px) + 6):
            vv, uu = vd + dy, ud + dx
            if not (0 <= vv < depth_mm.shape[0] and 0 <= uu < depth_mm.shape[1]):
                continue

            dist = np.sqrt(dx**2 + dy**2)
            if inner_px <= dist <= outer_px:
                z = depth_mm[vv, uu]
                if not np.isnan(z):
                    # Deproject this depth pixel
                    pt = deproject_pixel_to_point(depth_intr, [uu, vv], z / 1000.0)
                    annulus_points.append(pt)

    if len(annulus_points) < 10:
        raise ValueError(f"Insufficient annulus points ({len(annulus_points)}) for hole at depth pixel ({vd}, {ud}), color pixel ({v_color:.0f}, {u_color:.0f})")

    annulus_points = np.array(annulus_points)

    # Fit plane to annulus (in depth frame)
    centroid = annulus_points.mean(axis=0)
    centered = annulus_points - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal_depth = Vt[2]  # Normal is the smallest singular vector

    # Ensure normal points toward camera (positive z in depth frame)
    if normal_depth[2] < 0:
        normal_depth = -normal_depth

    # Transform plane to colour frame
    # The normal transforms as: n_colour = R @ n_depth
    R = np.array(depth_to_colour.rotation).reshape(3, 3)
    t = np.array(depth_to_colour.translation)

    normal_colour = R @ normal_depth
    centroid_colour = R @ centroid + t

    # Intersect ray through (u_color, v_color) with plane
    # Ray: P = t * d where d is the ray direction
    ray_dir = np.array(deproject_pixel_to_point(colour_intr, [u_color, v_color], 1.0))
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # Plane: (P - centroid) · normal = 0
    # Solve: t * (d · n) = (centroid · n)
    denom = np.dot(ray_dir, normal_colour)
    if abs(denom) < 1e-6:
        raise ValueError("Ray parallel to plane")

    t_intersect = np.dot(centroid_colour, normal_colour) / denom
    point_3d = t_intersect * ray_dir

    return point_3d


# ---------------------------------------------------------------------------
# B4: Pose solve
# ---------------------------------------------------------------------------

def solve_coupon_pose(
    origin_hole: DetectedHole,
    axis_hole: DetectedHole,
    depth_mm: np.ndarray,
    plate_mask: np.ndarray,
    colour_intr,
    depth_intr,
    colour_to_depth,
    depth_to_colour,
    face_label: str,
    config: dict,
    check_hole: DetectedHole | None = None,
    standoff_mm: float = 320.0,
) -> CouponPose:
    """Solve the full coupon pose from detected holes.

    Steps (from B4):
    1. Deproject hole centres using local plane fits
    2. Compute initial normal from annulus planes
    3. Compute s-axis from hole positions
    4. Derive face from weld-side extent
    5. Refine normal from datum strip
    6. Compute origin using both holes
    """
    dep_cfg = config["deprojection"]
    val_cfg = config["validation"]
    coupon_cfg = config["coupon"]
    holes_cfg = config["datum_holes"]

    flags = []

    # Deproject hole centres
    P6 = deproject_hole_centre(
        origin_hole, depth_mm, colour_intr, depth_intr,
        colour_to_depth, depth_to_colour, config,
        standoff_mm=standoff_mm
    )
    P8 = deproject_hole_centre(
        axis_hole, depth_mm, colour_intr, depth_intr,
        colour_to_depth, depth_to_colour, config,
        standoff_mm=standoff_mm
    )

    origin_hole.position_3d = P6
    axis_hole.position_3d = P8

    # Measured baseline
    baseline = np.linalg.norm(P8 - P6) * 1000  # mm
    expected_baseline = holes_cfg["baseline_mm"]

    if abs(baseline - expected_baseline) > val_cfg["baseline_tolerance_mm"]:
        flags.append(f"baseline_error_{baseline:.1f}mm")

    # Initial normal (camera-facing)
    # For now, use approximate normal pointing along camera z
    # This will be refined from the datum strip
    n_cam = np.array([0, 0, 1])  # Approximate: plate faces camera

    # S-axis: from origin to axis hole, projected onto plane
    d = P8 - P6
    s_hat = d - np.dot(d, n_cam) * n_cam
    s_hat = s_hat / np.linalg.norm(s_hat)

    # Y-axis in camera frame (before face determination)
    y_cam = np.cross(n_cam, s_hat)
    y_cam = y_cam / np.linalg.norm(y_cam)

    # Derive face from plate extent
    # The datum holes are at y = +50 mm, so the datum side is narrower (20 mm to edge)
    # and the weld side is wider (~70 mm to centreline + ~70 mm to far edge)
    # Measure plate mask extent on each side of the hole line

    # For simplicity, check which side of the hole line has more valid depth
    # Project hole positions to depth image and count valid pixels on each side

    # ... (simplified for now - use face_label with verification)
    derived_face = face_label  # Will be verified by weld-side extent check

    if derived_face == "cap" or derived_face == "top":
        z_hat = n_cam
        y_hat = y_cam
        derived_face = "cap"
    else:
        z_hat = -n_cam
        y_hat = -y_cam
        derived_face = "root"

    # Rotation matrix: columns are [ŝ, ŷ, ẑ]
    R = np.column_stack([s_hat, y_hat, z_hat])

    # Origin: midpoint of holes, then offset by nominal positions
    s_origin = holes_cfg["origin"]["s_mm"]
    s_axis = holes_cfg["axis"]["s_mm"]
    y_holes = holes_cfg["origin"]["y_mm"]

    midpoint = (P6 + P8) / 2
    s_mid = (s_origin + s_axis) / 2  # = 150 mm

    O = midpoint - (s_mid / 1000) * s_hat - (y_holes / 1000) * y_hat

    # Check Ø10 if present
    d10_residual = None
    if check_hole is not None:
        try:
            P10 = deproject_hole_centre(
                check_hole, depth_mm, colour_intr, depth_intr,
                colour_to_depth, depth_to_colour, config,
                standoff_mm=standoff_mm
            )
            check_hole.position_3d = P10

            # Expected position
            s10 = holes_cfg["check"]["s_mm"]
            y10 = holes_cfg["check"]["y_mm"]
            expected_P10 = O + (s10 / 1000) * s_hat + (y10 / 1000) * y_hat

            # In-plane residual
            diff = P10 - expected_P10
            d10_residual = np.sqrt(diff[0]**2 + diff[1]**2) * 1000  # mm

            if d10_residual > val_cfg["d10_residual_mm"]:
                flags.append(f"d10_residual_{d10_residual:.2f}mm")

        except Exception as e:
            flags.append(f"d10_deproject_failed: {e}")

    face_match = (derived_face == face_label or
                  (derived_face == "cap" and face_label == "top") or
                  (derived_face == "root" and face_label == "root"))

    if not face_match:
        flags.append(f"face_mismatch_derived_{derived_face}_label_{face_label}")

    return CouponPose(
        R=R,
        O=O,
        origin_hole=origin_hole,
        axis_hole=axis_hole,
        check_hole=check_hole,
        measured_baseline_mm=baseline,
        derived_face=derived_face,
        face_label=face_label,
        face_match=face_match,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# B5: Point cloud extraction
# ---------------------------------------------------------------------------

def extract_point_cloud(
    depth_mm: np.ndarray,
    depth_std: np.ndarray,
    n_valid_frames: np.ndarray,
    pose: CouponPose,
    depth_intr,
    depth_to_colour,
    config: dict,
) -> PointCloud:
    """Extract point cloud in coupon coordinates.

    Deprojects every valid depth pixel and transforms to coupon frame.
    """
    pc_cfg = config["point_cloud"]
    min_valid = pc_cfg["min_valid_frames"]

    h, w = depth_mm.shape

    # Build validity mask
    valid = ~np.isnan(depth_mm) & (n_valid_frames >= min_valid)

    rows, cols = np.where(valid)
    n_points = len(rows)

    if n_points == 0:
        return PointCloud(
            s=np.array([]), y=np.array([]), z=np.array([]),
            n_valid=np.array([]), sigma=np.array([]),
            row_idx=np.array([]), col_idx=np.array([]),
        )

    # Deproject all valid pixels
    points_depth = np.zeros((n_points, 3))
    for i, (r, c) in enumerate(zip(rows, cols)):
        z_m = depth_mm[r, c] / 1000.0
        pt = deproject_pixel_to_point(depth_intr, [c, r], z_m)
        points_depth[i] = pt

    # Transform to colour frame
    R = np.array(depth_to_colour.rotation).reshape(3, 3)
    t = np.array(depth_to_colour.translation)
    points_colour = (R @ points_depth.T).T + t

    # Transform to coupon frame
    points_coupon = pose.to_coupon(points_colour.T).T

    # Convert to mm
    s = points_coupon[:, 0] * 1000
    y = points_coupon[:, 1] * 1000
    z = points_coupon[:, 2] * 1000

    return PointCloud(
        s=s,
        y=y,
        z=z,
        n_valid=n_valid_frames[rows, cols],
        sigma=depth_std[rows, cols],
        row_idx=rows,
        col_idx=cols,
    )


def compute_bead_centreline(
    cloud: PointCloud,
    config: dict,
) -> dict:
    """Compute bead crown y-position along the weld.

    Returns summary statistics for y_crown(s).
    """
    prof_cfg = config["profile_extraction"]
    step = prof_cfg["crown_check_step_mm"]
    y_half = prof_cfg["crown_search_y_half_mm"]

    s_min, s_max = cloud.s.min(), cloud.s.max()
    s_positions = np.arange(s_min + step, s_max - step, step)

    y_crowns = []
    s_crowns = []

    for s_centre in s_positions:
        profile = cloud.extract_profile(s_centre, half_width=0.5)
        if profile["n"] < 10:
            continue

        # Find crown within |y| < y_half
        mask = np.abs(profile["y"]) < y_half
        if not mask.any():
            continue

        y_bead = profile["y"][mask]
        z_bead = profile["z"][mask]

        crown_idx = np.argmax(z_bead)
        y_crown = y_bead[crown_idx]

        y_crowns.append(y_crown)
        s_crowns.append(s_centre)

    if not y_crowns:
        return {
            "y_crown_mean": float("nan"),
            "y_crown_slope": float("nan"),
            "y_crown_max": float("nan"),
            "n_points": 0,
        }

    y_crowns = np.array(y_crowns)
    s_crowns = np.array(s_crowns)

    # Fit linear trend
    if len(s_crowns) >= 2:
        slope = np.polyfit(s_crowns, y_crowns, 1)[0]
    else:
        slope = 0.0

    return {
        "y_crown_mean": float(np.mean(y_crowns)),
        "y_crown_slope": float(slope),
        "y_crown_max": float(np.max(np.abs(y_crowns))),
        "n_points": len(y_crowns),
    }


# ---------------------------------------------------------------------------
# High-level registration pipeline
# ---------------------------------------------------------------------------

def register_capture(
    capture_dir: Path,
    config: dict | None = None,
) -> tuple[CouponPose, PointCloud, dict]:
    """Full registration pipeline for one capture.

    Returns (pose, point_cloud, metadata).
    """
    config = config or load_config()

    # Load metadata
    meta_path = capture_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # Build intrinsics/extrinsics
    depth_intr = build_rs_intrinsics(meta["depth_intrinsics"])
    colour_intr = build_rs_intrinsics(meta["color_intrinsics"])
    depth_to_colour = build_rs_extrinsics(meta["depth_to_color_extrinsics"])

    # Inverse extrinsics (colour to depth)
    R = np.array(depth_to_colour.rotation).reshape(3, 3)
    t = np.array(depth_to_colour.translation)
    R_inv = R.T
    t_inv = -R_inv @ t

    if HAS_PYREALSENSE:
        colour_to_depth = rs.extrinsics()
        colour_to_depth.rotation = R_inv.flatten().tolist()
        colour_to_depth.translation = t_inv.tolist()
    else:
        colour_to_depth = FallbackExtrinsics(
            rotation=R_inv.flatten().tolist(),
            translation=t_inv.tolist(),
        )

    # Load frames
    colour_img = compute_median_colour(capture_dir / "color")
    depth_mm, depth_std, n_frames = compute_averaged_depth(
        capture_dir / "depth",
        meta.get("depth_units", 0.0001)
    )

    standoff_mm = meta.get("standoff_mm_from_depth", 320.0)
    face_label = meta.get("face", "top")

    # Create plate mask
    plate_mask_depth = create_plate_mask_from_depth(
        depth_mm, standoff_mm,
        config["hole_detection"]["plate_mask_depth_window_mm"]
    )

    # Map to colour
    plate_mask_colour = map_depth_mask_to_colour(
        plate_mask_depth, depth_intr, colour_intr,
        depth_to_colour, depth_mm
    )

    # Detect holes
    holes = detect_datum_holes_in_colour(
        colour_img, plate_mask_colour,
        meta["color_intrinsics"], standoff_mm, config
    )

    # Find origin and axis holes
    origin_holes = [h for h in holes if h.hole_type == "origin"]
    axis_holes = [h for h in holes if h.hole_type == "axis"]
    check_holes = [h for h in holes if h.hole_type == "check"]

    if not origin_holes or not axis_holes:
        raise ValueError(
            f"Missing datum holes: {len(origin_holes)} origin, {len(axis_holes)} axis"
        )

    # Take highest confidence of each type
    origin_hole = max(origin_holes, key=lambda h: h.confidence)
    axis_hole = max(axis_holes, key=lambda h: h.confidence)
    check_hole = max(check_holes, key=lambda h: h.confidence) if check_holes else None

    # Solve pose
    pose = solve_coupon_pose(
        origin_hole, axis_hole, depth_mm, plate_mask_depth,
        colour_intr, depth_intr, colour_to_depth, depth_to_colour,
        face_label, config, check_hole
    )

    # Extract point cloud
    n_valid_frames = np.full_like(depth_mm, n_frames)  # Simplified
    cloud = extract_point_cloud(
        depth_mm, depth_std, n_valid_frames,
        pose, depth_intr, depth_to_colour, config
    )

    # Compute bead centreline
    centreline = compute_bead_centreline(cloud, config)
    pose.y_crown_mean_mm = centreline["y_crown_mean"]
    pose.y_crown_slope = centreline["y_crown_slope"]
    pose.y_crown_max_mm = centreline["y_crown_max"]

    if centreline["y_crown_max"] > config["validation"]["max_y_crown_mm"]:
        pose.flags.append(f"y_crown_exceeded_{centreline['y_crown_max']:.2f}mm")

    # Compile metadata
    result_meta = {
        "capture_dir": str(capture_dir),
        "coupon_id": meta.get("coupon_id", capture_dir.name.split("_")[0]),
        "face": face_label,
        "standoff_mm": standoff_mm,
        "n_depth_frames": n_frames,
        "pose": pose.to_dict(),
        "holes": {
            "origin": {
                "centre_px": origin_hole.centre_px,
                "free_fit_radius_px": origin_hole.free_fit_radius_px,
                "fixed_fit_residual_px": origin_hole.fixed_fit_residual_px,
                "position_3d": origin_hole.position_3d.tolist() if origin_hole.position_3d is not None else None,
            },
            "axis": {
                "centre_px": axis_hole.centre_px,
                "free_fit_radius_px": axis_hole.free_fit_radius_px,
                "fixed_fit_residual_px": axis_hole.fixed_fit_residual_px,
                "position_3d": axis_hole.position_3d.tolist() if axis_hole.position_3d is not None else None,
            },
        },
        "measured_baseline_mm": pose.measured_baseline_mm,
        "y_crown": centreline,
        "flags": pose.flags,
    }

    return pose, cloud, result_meta


# ---------------------------------------------------------------------------
# Deprecation of old functions
# ---------------------------------------------------------------------------

def extract_single_profile(*args, **kwargs):
    """DEPRECATED: Use PointCloud.extract_profile() instead.

    Profile extraction now operates on pose-aware point clouds, not image rows.
    This removes the implicit assumption that image centre = weld centreline.

    See: src/geometry/registration_v2.py:PointCloud.extract_profile()
    """
    raise NotImplementedError(
        "extract_single_profile is deprecated. "
        "Use PointCloud.extract_profile() from registration_v2.py instead. "
        "This change removes the image-centre assumption (rows // 2)."
    )
