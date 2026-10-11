"""
Depth map processing: averaging, noise estimation, profile extraction.

Key design decisions:
    1. Split-half noise estimation (not naive √N) — stereo noise is correlated
    2. Multiple transverse profiles per cell with quantile aggregation
    3. Tilt correction via plane-fit normal: d_true = Δz · cos(φ)
    4. Undercut measured at weld toes relative to parent metal datum
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


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load geometry configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Depth frame loading and temporal averaging
# ---------------------------------------------------------------------------

@dataclass
class DepthStack:
    """Stack of depth frames with temporal statistics."""
    mean_mm: np.ndarray          # Averaged depth in mm
    std_mm: np.ndarray           # Per-pixel standard deviation in mm
    n_frames: int                # Number of frames averaged
    valid_mask: np.ndarray       # True where depth is valid (not NaN)

    # Split-half noise estimation
    split_half_se_mm: float      # Standard error from split-half
    naive_se_mm: float           # Naive σ/√N estimate
    averaging_efficiency: float  # Ratio: naive / split-half (ideally 1.0)
    noise_floor_mm: float        # Estimated non-averaging floor


def load_depth_frames(
    frame_dir: Path,
    depth_units: float = 0.0001,
) -> list[np.ndarray]:
    """
    Load depth frames from a capture directory.

    Args:
        frame_dir: Directory containing 000.npy, 001.npy, etc.
        depth_units: Depth unit in meters (0.0001 = 100µm).

    Returns:
        List of depth frames in mm.
    """
    frames = []
    for f in sorted(frame_dir.glob("*.npy")):
        raw = np.load(f).astype(np.float32)
        # Convert to mm: raw * depth_units * 1000
        depth_mm = raw * depth_units * 1000.0
        # Mark invalid pixels (raw == 0 means no return)
        depth_mm[raw == 0] = np.nan
        frames.append(depth_mm)
    return frames


def compute_depth_stack(
    frames: list[np.ndarray],
    config: dict | None = None,
) -> DepthStack:
    """
    Compute temporal average and split-half noise estimate.

    Split-half estimation: compare mean of first half vs second half.
    SE = ½ · std(difference). This accounts for temporal correlation
    that naive √N ignores.

    Args:
        frames: List of depth frames in mm (NaN = invalid).
        config: Geometry config dict.

    Returns:
        DepthStack with averaged depth and noise estimates.
    """
    config = config or load_config()
    n = len(frames)
    if n == 0:
        raise ValueError("No frames to average")

    stack = np.stack(frames, axis=0)  # (N, H, W)

    # Overall mean and std
    with np.errstate(all="ignore"):
        mean_mm = np.nanmean(stack, axis=0)
        std_mm = np.nanstd(stack, axis=0)

    valid_mask = ~np.isnan(mean_mm)

    # Split-half noise estimation
    half = n // 2
    if half > 0:
        with np.errstate(all="ignore"):
            mean_first = np.nanmean(stack[:half], axis=0)
            mean_second = np.nanmean(stack[half:], axis=0)
            diff = mean_first - mean_second

        # SE = ½ · std(difference) over valid pixels
        valid_diff = diff[valid_mask & ~np.isnan(diff)]
        if len(valid_diff) > 0:
            split_half_se = 0.5 * np.std(valid_diff)
        else:
            split_half_se = float("nan")
    else:
        split_half_se = float("nan")

    # Naive estimate: median(std) / sqrt(N)
    valid_std = std_mm[valid_mask]
    if len(valid_std) > 0:
        median_std = np.nanmedian(valid_std)
        naive_se = median_std / math.sqrt(n)
    else:
        median_std = float("nan")
        naive_se = float("nan")

    # Averaging efficiency
    if not math.isnan(split_half_se) and split_half_se > 0:
        efficiency = naive_se / split_half_se
    else:
        efficiency = float("nan")

    # Estimate noise floor (the part that doesn't average out)
    # This is exposed via the split-half estimate
    unc_cfg = config.get("uncertainty", {})
    noise_floor_default = unc_cfg.get("noise_floor_mm_default", 0.015)
    noise_floor = max(split_half_se, noise_floor_default) if not math.isnan(split_half_se) else noise_floor_default

    return DepthStack(
        mean_mm=mean_mm,
        std_mm=std_mm,
        n_frames=n,
        valid_mask=valid_mask,
        split_half_se_mm=float(split_half_se),
        naive_se_mm=float(naive_se),
        averaging_efficiency=float(efficiency),
        noise_floor_mm=float(noise_floor),
    )


# ---------------------------------------------------------------------------
# Plane fitting and tilt correction
# ---------------------------------------------------------------------------

@dataclass
class PlaneFit:
    """Result of fitting a plane to parent metal regions."""
    normal: np.ndarray           # Unit normal vector (nx, ny, nz)
    d: float                     # Plane equation: nx*x + ny*y + nz*z + d = 0
    tilt_angle_deg: float        # Angle from horizontal
    residual_rms_mm: float       # RMS of fit residuals (= u_cal)
    n_points: int                # Number of points used in fit


def fit_plane_ransac(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    scale_mm_per_px: float,
    n_iterations: int = 100,
    inlier_threshold_mm: float = 0.5,
) -> PlaneFit:
    """
    Fit a plane to masked depth data using RANSAC.

    Args:
        depth_mm: Depth image in mm.
        mask: Boolean mask of pixels to include in fit.
        scale_mm_per_px: Lateral scale for converting pixel coords to mm.
        n_iterations: RANSAC iterations.
        inlier_threshold_mm: Distance threshold for inliers.

    Returns:
        PlaneFit with plane parameters and quality metrics.
    """
    # Get valid points
    valid = mask & ~np.isnan(depth_mm)
    rows, cols = np.where(valid)
    if len(rows) < 3:
        raise ValueError("Insufficient points for plane fit")

    # Convert to 3D coordinates (x=col, y=row, z=depth)
    x = cols.astype(np.float64) * scale_mm_per_px
    y = rows.astype(np.float64) * scale_mm_per_px
    z = depth_mm[valid].astype(np.float64)

    points = np.column_stack([x, y, z])
    n_points = len(points)

    best_inliers = 0
    best_plane = None

    rng = np.random.default_rng(42)
    for _ in range(n_iterations):
        # Sample 3 random points
        idx = rng.choice(n_points, size=3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm

        # Plane equation: n · (p - p1) = 0  =>  n · p = n · p1
        d = -np.dot(normal, p1)

        # Count inliers
        distances = np.abs(np.dot(points, normal) + d)
        n_inliers = np.sum(distances < inlier_threshold_mm)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_plane = (normal, d, distances)

    if best_plane is None:
        raise ValueError("RANSAC failed to find a valid plane")

    normal, d, distances = best_plane

    # Ensure normal points "up" (positive z component = toward camera)
    if normal[2] < 0:
        normal = -normal
        d = -d

    # Compute tilt angle (angle between normal and vertical)
    vertical = np.array([0, 0, 1])
    cos_tilt = np.dot(normal, vertical)
    tilt_rad = math.acos(np.clip(cos_tilt, -1, 1))
    tilt_deg = math.degrees(tilt_rad)

    # Compute RMS residual on inliers
    inlier_mask = distances < inlier_threshold_mm
    residuals = distances[inlier_mask]
    rms = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else float("nan")

    return PlaneFit(
        normal=normal,
        d=d,
        tilt_angle_deg=tilt_deg,
        residual_rms_mm=rms,
        n_points=int(best_inliers),
    )


def apply_tilt_correction(
    depth_mm: np.ndarray,
    plane_fit: PlaneFit,
) -> np.ndarray:
    """
    Apply tilt correction: d_true = Δz · cos(φ).

    Args:
        depth_mm: Depth image in mm.
        plane_fit: Fitted plane parameters.

    Returns:
        Corrected depth image in mm.
    """
    cos_phi = abs(plane_fit.normal[2])  # cos of tilt angle
    return depth_mm * cos_phi


# ---------------------------------------------------------------------------
# Profile extraction and undercut measurement
# ---------------------------------------------------------------------------

@dataclass
class ProfileMeasurement:
    """Measurements from a single transverse profile."""
    row_px: int                  # Row in image where profile was taken
    s_mm: float                  # Along-weld position

    # Undercut at each toe (depth below parent metal)
    undercut_left_mm: float      # Undercut at -y toe
    undercut_right_mm: float     # Undercut at +y toe

    # Other measurements
    reinforcement_mm: float      # Max height above parent metal
    bead_width_mm: float         # Toe-to-toe distance
    misalignment_mm: float       # Hi-lo between left and right parent metal

    # Validity
    n_valid: int                 # Valid pixels in this profile
    valid_fraction: float


def extract_single_profile(
    depth_mm: np.ndarray,
    col: int,
    s_mm: float,
    scale_mm_per_px: float,
    config: dict,
) -> ProfileMeasurement:
    """
    Extract measurements from a single transverse profile.

    A transverse profile is a column (all rows at fixed column), where:
    - Rows correspond to y positions (across weld, perpendicular to weld axis)
    - The column index corresponds to the s position (along weld)

    Args:
        depth_mm: Tilt-corrected depth image in mm. Standard convention:
                  rows = y (across weld), cols = s (along weld).
        col: Column index for this profile (s position).
        s_mm: Along-weld position in mm.
        scale_mm_per_px: Lateral scale.
        config: Geometry config dict.

    Returns:
        ProfileMeasurement with undercut, reinforcement, etc.
    """
    prof_cfg = config["profile_extraction"]
    smooth_k = prof_cfg["smoothing_kernel_px"]
    y_inner = prof_cfg["plane_fit_roi"]["y_inner_mm"]
    y_outer = prof_cfg["plane_fit_roi"]["y_outer_mm"]
    bead_half = prof_cfg["bead_roi"]["y_half_width_mm"]

    # Extract column (transverse profile) - all y values at this s position
    profile = depth_mm[:, col].copy()
    n_rows = len(profile)
    centre_row = n_rows // 2

    # Smooth profile
    if smooth_k > 1:
        kernel = np.ones(smooth_k) / smooth_k
        profile = np.convolve(profile, kernel, mode="same")

    # Convert row indices to y (mm from centreline)
    # Row indices: 0 at top of image, increasing downward
    # Convention: +y is "up" in coupon coords, so row decreases as y increases
    def row_to_y(row):
        return (centre_row - row) * scale_mm_per_px

    def y_to_row(y_mm):
        return centre_row - int(round(y_mm / scale_mm_per_px))

    # Define regions (in row indices)
    # -y (below centreline in coupon coords) = row > centre_row
    # +y (above centreline in coupon coords) = row < centre_row
    row_left_outer = y_to_row(-y_outer)   # -y outer = large row
    row_left_inner = y_to_row(-y_inner)   # -y inner
    row_right_inner = y_to_row(y_inner)   # +y inner
    row_right_outer = y_to_row(y_outer)   # +y outer = small row

    # Clamp to valid range
    row_left_outer = min(n_rows, max(0, row_left_outer))
    row_left_inner = min(n_rows, max(0, row_left_inner))
    row_right_inner = min(n_rows, max(0, row_right_inner))
    row_right_outer = min(n_rows, max(0, row_right_outer))

    # Extract parent metal regions
    # Left parent: -y region (rows from row_left_inner to row_left_outer)
    left_parent = profile[row_left_inner:row_left_outer]
    # Right parent: +y region (rows from row_right_outer to row_right_inner)
    right_parent = profile[row_right_outer:row_right_inner]

    # Compute parent metal datum levels
    valid_left = left_parent[~np.isnan(left_parent)]
    valid_right = right_parent[~np.isnan(right_parent)]

    if len(valid_left) < 3 or len(valid_right) < 3:
        # Insufficient data
        return ProfileMeasurement(
            row_px=col,  # Now stores col_px (s position)
            s_mm=s_mm,
            undercut_left_mm=float("nan"),
            undercut_right_mm=float("nan"),
            reinforcement_mm=float("nan"),
            bead_width_mm=float("nan"),
            misalignment_mm=float("nan"),
            n_valid=0,
            valid_fraction=0.0,
        )

    datum_left = float(np.median(valid_left))
    datum_right = float(np.median(valid_right))

    # Misalignment (hi-lo between plate halves)
    misalignment = abs(datum_left - datum_right)

    # Average datum for undercut/reinforcement reference
    datum_avg = (datum_left + datum_right) / 2

    # Bead region (in row indices)
    row_bead_left = y_to_row(-bead_half)   # -y side = larger row
    row_bead_right = y_to_row(bead_half)   # +y side = smaller row
    row_bead_left = min(n_rows, max(0, row_bead_left))
    row_bead_right = min(n_rows, max(0, row_bead_right))
    bead = profile[row_bead_right:row_bead_left]  # Slice from smaller to larger row
    valid_bead = ~np.isnan(bead)

    # Reinforcement: max height above datum (remember: smaller depth = higher surface)
    # Height above datum = datum - depth (positive if surface is closer to camera)
    if valid_bead.any():
        height_above_datum = datum_avg - bead
        reinforcement = float(np.nanmax(height_above_datum))
        reinforcement = max(0, reinforcement)
    else:
        reinforcement = float("nan")

    # Find toe positions (transition from bead to parent metal)
    # Look for where profile crosses datum level
    height_profile = datum_avg - profile

    # Left toe (-y side): search from bead center toward -y (increasing row)
    toe_left_row = row_bead_left
    for r in range(centre_row, row_left_inner):
        if 0 <= r < len(height_profile) and not np.isnan(height_profile[r]):
            if height_profile[r] < 0.1:  # Near datum level
                toe_left_row = r
                break

    # Right toe (+y side): search from bead center toward +y (decreasing row)
    toe_right_row = row_bead_right
    for r in range(centre_row, row_right_inner, -1):
        if 0 <= r < len(height_profile) and not np.isnan(height_profile[r]):
            if height_profile[r] < 0.1:  # Near datum level
                toe_right_row = r
                break

    # Bead width
    bead_width = abs(toe_left_row - toe_right_row) * scale_mm_per_px

    # Undercut: depth below datum at toes
    # Undercut is where the surface dips BELOW the parent metal
    # Look in a small region around each toe
    toe_search_px = max(3, int(2.0 / scale_mm_per_px))  # ~2mm region

    # Left toe undercut (-y side)
    left_toe_region = profile[max(0, toe_left_row - toe_search_px):min(n_rows, toe_left_row + toe_search_px)]
    if len(left_toe_region) > 0 and not np.all(np.isnan(left_toe_region)):
        # Undercut = how far below datum the surface goes
        depth_below_datum = left_toe_region - datum_left
        undercut_left = float(np.nanmax(depth_below_datum))  # Positive = below datum
        undercut_left = max(0, undercut_left)
    else:
        undercut_left = float("nan")

    # Right toe undercut (+y side)
    right_toe_region = profile[max(0, toe_right_row - toe_search_px):min(n_rows, toe_right_row + toe_search_px)]
    if len(right_toe_region) > 0 and not np.all(np.isnan(right_toe_region)):
        depth_below_datum = right_toe_region - datum_right
        undercut_right = float(np.nanmax(depth_below_datum))
        undercut_right = max(0, undercut_right)
    else:
        undercut_right = float("nan")

    # Validity
    n_valid = int(np.sum(~np.isnan(profile)))
    valid_fraction = n_valid / n_rows

    return ProfileMeasurement(
        row_px=col,  # Stores col_px (s position) for compatibility
        s_mm=s_mm,
        undercut_left_mm=undercut_left,
        undercut_right_mm=undercut_right,
        reinforcement_mm=reinforcement,
        bead_width_mm=bead_width,
        misalignment_mm=misalignment,
        n_valid=n_valid,
        valid_fraction=valid_fraction,
    )


@dataclass
class CellMeasurement:
    """Aggregated measurements for a cell from multiple profiles."""
    cell_id: str
    face: str
    s_centre_mm: float

    # Governing undercut (95th percentile across profiles and toes)
    undercut_mm: float
    undercut_toe: str            # "left" or "right"

    # Other measurements (max across profiles)
    reinforcement_mm: float
    misalignment_mm: float
    bead_width_mm: float

    # Validity
    n_profiles: int
    n_valid_profiles: int
    valid_fraction: float

    # Tilt info
    tilt_angle_deg: float

    # Raw profile data
    profile_measurements: list[ProfileMeasurement]


def extract_cell_measurements(
    depth_mm: np.ndarray,
    cell_col_start: int,
    cell_col_end: int,
    cell_id: str,
    face: str,
    s_centre_mm: float,
    scale_mm_per_px: float,
    plane_fit: PlaneFit,
    config: dict | None = None,
) -> CellMeasurement:
    """
    Extract measurements for a single cell from multiple transverse profiles.

    Args:
        depth_mm: Tilt-corrected depth image in mm.
        cell_col_start, cell_col_end: Column range for this cell.
        cell_id: Cell identifier.
        face: "top" or "root".
        s_centre_mm: Centre position along weld axis.
        scale_mm_per_px: Lateral scale.
        plane_fit: Fitted plane for tilt info.
        config: Geometry config dict.

    Returns:
        CellMeasurement with aggregated values.
    """
    config = config or load_config()
    prof_cfg = config["profile_extraction"]
    n_profiles = prof_cfg["profiles_per_cell"]
    quantile = prof_cfg["aggregation_quantile"]
    min_valid = prof_cfg["min_valid_fraction"]

    # Determine column positions for profiles (evenly spaced within cell s-range)
    # Standard image convention: cols = s (along weld), rows = y (across weld)
    col_start_clamped = max(0, cell_col_start)
    col_end_clamped = min(depth_mm.shape[1], cell_col_end)
    cols = np.linspace(col_start_clamped, col_end_clamped - 1, n_profiles, dtype=int)

    # Extract profiles (each profile is a column - all y values at a fixed s)
    profiles = []
    for col in cols:
        # Compute s position for this column
        s_mm = s_centre_mm + (col - (col_start_clamped + col_end_clamped) / 2) * scale_mm_per_px

        prof = extract_single_profile(
            depth_mm, col, s_mm, scale_mm_per_px, config
        )
        profiles.append(prof)

    # Filter valid profiles
    valid_profiles = [p for p in profiles if p.valid_fraction >= min_valid]
    n_valid = len(valid_profiles)

    if n_valid == 0:
        return CellMeasurement(
            cell_id=cell_id,
            face=face,
            s_centre_mm=s_centre_mm,
            undercut_mm=float("nan"),
            undercut_toe="",
            reinforcement_mm=float("nan"),
            misalignment_mm=float("nan"),
            bead_width_mm=float("nan"),
            n_profiles=n_profiles,
            n_valid_profiles=0,
            valid_fraction=0.0,
            tilt_angle_deg=plane_fit.tilt_angle_deg,
            profile_measurements=profiles,
        )

    # Aggregate undercuts with quantile
    left_undercuts = [p.undercut_left_mm for p in valid_profiles if not math.isnan(p.undercut_left_mm)]
    right_undercuts = [p.undercut_right_mm for p in valid_profiles if not math.isnan(p.undercut_right_mm)]

    undercut_left_q = float(np.percentile(left_undercuts, 100 * quantile)) if left_undercuts else float("nan")
    undercut_right_q = float(np.percentile(right_undercuts, 100 * quantile)) if right_undercuts else float("nan")

    # Governing undercut is the max of left and right
    if math.isnan(undercut_left_q) and math.isnan(undercut_right_q):
        governing_undercut = float("nan")
        governing_toe = ""
    elif math.isnan(undercut_right_q) or undercut_left_q >= undercut_right_q:
        governing_undercut = undercut_left_q
        governing_toe = "left"
    else:
        governing_undercut = undercut_right_q
        governing_toe = "right"

    # Other measurements: use max across valid profiles
    reinforcements = [p.reinforcement_mm for p in valid_profiles if not math.isnan(p.reinforcement_mm)]
    misalignments = [p.misalignment_mm for p in valid_profiles if not math.isnan(p.misalignment_mm)]
    widths = [p.bead_width_mm for p in valid_profiles if not math.isnan(p.bead_width_mm)]

    return CellMeasurement(
        cell_id=cell_id,
        face=face,
        s_centre_mm=s_centre_mm,
        undercut_mm=governing_undercut,
        undercut_toe=governing_toe,
        reinforcement_mm=float(max(reinforcements)) if reinforcements else float("nan"),
        misalignment_mm=float(max(misalignments)) if misalignments else float("nan"),
        bead_width_mm=float(np.median(widths)) if widths else float("nan"),
        n_profiles=n_profiles,
        n_valid_profiles=n_valid,
        valid_fraction=n_valid / n_profiles,
        tilt_angle_deg=plane_fit.tilt_angle_deg,
        profile_measurements=profiles,
    )
