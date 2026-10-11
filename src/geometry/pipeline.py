"""
Geometry modality pipeline: full flow from raw depth frames to BPA.

Orchestrates:
    1. Depth frame loading and temporal averaging (with split-half noise estimation)
    2. Coupon registration (datum hole detection, pose solve)
    3. Plane fit and tilt correction
    4. Per-cell profile extraction with quantile aggregation
    5. BPA construction for each cell

The output is a list of GeometryBPA objects ready for Dempster-Shafer fusion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .profile import (
    DepthStack,
    PlaneFit,
    CellMeasurement,
    load_depth_frames,
    compute_depth_stack,
    fit_plane_ransac,
    apply_tilt_correction,
    extract_cell_measurements,
)
from .registration import (
    CouponPose,
    CellBounds,
    register_coupon,
    detect_holes_in_depth,
    solve_pose,
    compute_cell_bounds,
    detect_plate_edges,
)
from .rules import (
    UncertaintyBudget,
    GeometryMeasurement,
    GeometryBPA,
    ValidationOutput,
    build_cell_bpa,
    assess_validation_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "geometry.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load geometry configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class FaceResult:
    """Processing result for one coupon face (top or root)."""
    face: str
    depth_stack: DepthStack
    plane_fit: PlaneFit
    cell_measurements: list[CellMeasurement]


@dataclass
class CouponResult:
    """Full processing result for one coupon."""
    coupon_id: str
    pose: CouponPose
    cell_bounds: list[CellBounds]
    face_results: dict[str, FaceResult]  # "top" and/or "root"
    cell_bpas: list[GeometryBPA]
    validation_outputs: list[ValidationOutput]
    uncertainty: UncertaintyBudget

    def to_dict(self) -> dict[str, Any]:
        return {
            "coupon_id": self.coupon_id,
            "pose": self.pose.to_dict(),
            "cell_bounds": [b.to_dict() for b in self.cell_bounds],
            "uncertainty": self.uncertainty.to_dict(),
            "cell_bpas": [bpa.to_dict() for bpa in self.cell_bpas],
        }


def compute_uncertainty_budget(
    depth_stacks: list[DepthStack],
    plane_fits: list[PlaneFit],
    config: dict | None = None,
) -> UncertaintyBudget:
    """
    Compute the four-term uncertainty budget from measured data.

    Args:
        depth_stacks: DepthStack objects from both faces.
        plane_fits: PlaneFit objects from both faces.
        config: Geometry config dict.

    Returns:
        UncertaintyBudget with measured values.
    """
    config = config or load_config()
    unc_cfg = config["uncertainty"]

    # u_noise: from split-half estimation (use max across faces)
    split_half_ses = [ds.split_half_se_mm for ds in depth_stacks if not math.isnan(ds.split_half_se_mm)]
    if split_half_ses:
        u_noise = max(split_half_ses)
    else:
        u_noise = unc_cfg.get("noise_floor_mm_default", 0.015)

    # u_quant: fixed from config (0.1mm / sqrt(12))
    u_quant = unc_cfg["u_quant_mm"]

    # u_cal: from plane fit residuals (use max across faces)
    residuals = [pf.residual_rms_mm for pf in plane_fits if not math.isnan(pf.residual_rms_mm)]
    if residuals:
        u_cal = max(residuals)
    else:
        u_cal = unc_cfg.get("u_cal_mm_default", 0.030)

    # u_align: from config (measured from repeat-placement captures)
    # This is measured during the capture session, not computed here
    u_align = unc_cfg.get("u_align_mm_default", 0.020)

    return UncertaintyBudget(
        u_noise=u_noise,
        u_quant=u_quant,
        u_cal=u_cal,
        u_align=u_align,
    )


def process_face(
    depth_mm: np.ndarray,
    face: str,
    pose: CouponPose,
    cell_bounds: list[CellBounds],
    config: dict | None = None,
) -> FaceResult:
    """
    Process a single coupon face: plane fit, tilt correction, profile extraction.

    Args:
        depth_mm: Averaged depth image in mm.
        face: "top" or "root".
        pose: Coupon pose from registration.
        cell_bounds: Cell boundaries in pixel coordinates.
        config: Geometry config dict.

    Returns:
        FaceResult with measurements for all cells.
    """
    config = config or load_config()
    prof_cfg = config["profile_extraction"]
    y_inner = prof_cfg["plane_fit_roi"]["y_inner_mm"]
    y_outer = prof_cfg["plane_fit_roi"]["y_outer_mm"]

    # Create mask for parent metal regions (for plane fit)
    h, w = depth_mm.shape
    parent_mask = np.zeros((h, w), dtype=bool)

    # Convert y bounds to columns for each row
    for row in range(h):
        # Find centre column for this row (approximate)
        centre_col = w // 2

        # Left parent metal region
        col_left_outer = max(0, centre_col - int(y_outer / pose.scale_mm_per_px))
        col_left_inner = max(0, centre_col - int(y_inner / pose.scale_mm_per_px))
        parent_mask[row, col_left_outer:col_left_inner] = True

        # Right parent metal region
        col_right_inner = min(w, centre_col + int(y_inner / pose.scale_mm_per_px))
        col_right_outer = min(w, centre_col + int(y_outer / pose.scale_mm_per_px))
        parent_mask[row, col_right_inner:col_right_outer] = True

    # Fit plane to parent metal
    plane_fit = fit_plane_ransac(
        depth_mm,
        parent_mask,
        pose.scale_mm_per_px,
        n_iterations=100,
        inlier_threshold_mm=0.5,
    )

    # Check tilt angle
    max_tilt = config.get("tilt_correction", {}).get("max_tilt_deg", 5.0)
    if plane_fit.tilt_angle_deg > max_tilt:
        # Log warning but continue (could raise if strict)
        pass

    # Apply tilt correction
    if config.get("tilt_correction", {}).get("enabled", True):
        depth_corrected = apply_tilt_correction(depth_mm, plane_fit)
    else:
        depth_corrected = depth_mm

    # Extract measurements for each cell
    cell_measurements = []
    for cell in cell_bounds:
        meas = extract_cell_measurements(
            depth_corrected,
            cell.col_start_px,
            cell.col_end_px,
            cell.cell_id,
            face,
            cell.s_centre_mm,
            pose.scale_mm_per_px,
            plane_fit,
            config,
        )
        cell_measurements.append(meas)

    # Create a minimal DepthStack for result (actual one computed in pipeline)
    # This is a placeholder; the real DepthStack is passed in from process_coupon
    depth_stack = DepthStack(
        mean_mm=depth_mm,
        std_mm=np.zeros_like(depth_mm),
        n_frames=1,
        valid_mask=~np.isnan(depth_mm),
        split_half_se_mm=float("nan"),
        naive_se_mm=float("nan"),
        averaging_efficiency=float("nan"),
        noise_floor_mm=config["uncertainty"]["noise_floor_mm_default"],
    )

    return FaceResult(
        face=face,
        depth_stack=depth_stack,
        plane_fit=plane_fit,
        cell_measurements=cell_measurements,
    )


def cell_measurement_to_geometry_measurement(
    cell_meas: CellMeasurement,
    toe: str,
) -> GeometryMeasurement:
    """
    Convert a CellMeasurement to a GeometryMeasurement for BPA construction.

    Args:
        cell_meas: CellMeasurement from profile extraction.
        toe: "left" or "right".

    Returns:
        GeometryMeasurement for rules.py.
    """
    if toe == "left":
        # Get left undercut from raw profiles
        undercuts = [
            p.undercut_left_mm
            for p in cell_meas.profile_measurements
            if not math.isnan(p.undercut_left_mm)
        ]
    else:
        undercuts = [
            p.undercut_right_mm
            for p in cell_meas.profile_measurements
            if not math.isnan(p.undercut_right_mm)
        ]

    undercut = float(np.percentile(undercuts, 95)) if undercuts else float("nan")

    return GeometryMeasurement(
        cell_id=cell_meas.cell_id,
        face=cell_meas.face,
        toe=toe,
        undercut_mm=undercut,
        reinforcement_mm=cell_meas.reinforcement_mm,
        misalignment_mm=cell_meas.misalignment_mm,
        bead_width_mm=cell_meas.bead_width_mm,
        n_valid_pixels=sum(p.n_valid for p in cell_meas.profile_measurements),
        valid_fraction=cell_meas.valid_fraction,
        tilt_angle_deg=cell_meas.tilt_angle_deg,
    )


def process_coupon(
    top_frames_dir: Path | None,
    root_frames_dir: Path | None,
    coupon_id: str,
    config: dict | None = None,
) -> CouponResult:
    """
    Full geometry pipeline for one coupon.

    Args:
        top_frames_dir: Directory with top-face depth frames (or None).
        root_frames_dir: Directory with root-face depth frames (or None).
        coupon_id: Coupon identifier (e.g., "C07").
        config: Geometry config dict.

    Returns:
        CouponResult with all measurements and BPAs.

    Raises:
        ValueError: If registration fails or no valid frames.
    """
    config = config or load_config()

    if top_frames_dir is None and root_frames_dir is None:
        raise ValueError("At least one face must have depth frames")

    # Load and average depth frames
    depth_stacks = {}
    if top_frames_dir is not None:
        top_frames = load_depth_frames(top_frames_dir)
        if top_frames:
            depth_stacks["top"] = compute_depth_stack(top_frames, config)

    if root_frames_dir is not None:
        root_frames = load_depth_frames(root_frames_dir)
        if root_frames:
            depth_stacks["root"] = compute_depth_stack(root_frames, config)

    if not depth_stacks:
        raise ValueError("No valid depth frames found")

    # Use top face for registration (preferred), fall back to root
    primary_face = "top" if "top" in depth_stacks else "root"
    primary_depth = depth_stacks[primary_face].mean_mm

    # Register coupon
    pose, cell_bounds = register_coupon(primary_depth, coupon_id, config)

    # Process each face
    face_results = {}
    plane_fits = []

    for face, stack in depth_stacks.items():
        result = process_face(
            stack.mean_mm,
            face,
            pose,
            cell_bounds,
            config,
        )
        # Update with actual depth stack
        result.depth_stack = stack
        face_results[face] = result
        plane_fits.append(result.plane_fit)

    # Compute uncertainty budget
    uncertainty = compute_uncertainty_budget(
        list(depth_stacks.values()),
        plane_fits,
        config,
    )

    # Build BPAs for each cell
    cell_bpas = []
    validation_outputs = []

    for cell in cell_bounds:
        # Collect measurements from all faces and toes for this cell
        measurements = []
        for face, result in face_results.items():
            cell_meas = next(
                (m for m in result.cell_measurements if m.cell_id == cell.cell_id),
                None,
            )
            if cell_meas is not None:
                # Create measurements for both toes
                measurements.append(
                    cell_measurement_to_geometry_measurement(cell_meas, "left")
                )
                measurements.append(
                    cell_measurement_to_geometry_measurement(cell_meas, "right")
                )

        if measurements:
            # Build BPA
            bpa = build_cell_bpa(measurements, uncertainty, config)
            cell_bpas.append(bpa)

            # Assess validation outputs
            validation = assess_validation_outputs(measurements, config)
            validation_outputs.append(validation)

    return CouponResult(
        coupon_id=coupon_id,
        pose=pose,
        cell_bounds=cell_bounds,
        face_results=face_results,
        cell_bpas=cell_bpas,
        validation_outputs=validation_outputs,
        uncertainty=uncertainty,
    )


def process_coupon_from_arrays(
    top_depth_mm: np.ndarray | None,
    root_depth_mm: np.ndarray | None,
    coupon_id: str,
    n_frames: int = 64,
    config: dict | None = None,
) -> CouponResult:
    """
    Process a coupon from pre-averaged depth arrays (for testing).

    This bypasses frame loading and temporal averaging for use with
    synthetic test data.

    Args:
        top_depth_mm: Averaged top-face depth in mm (or None).
        root_depth_mm: Averaged root-face depth in mm (or None).
        coupon_id: Coupon identifier.
        n_frames: Simulated number of averaged frames (for noise estimation).
        config: Geometry config dict.

    Returns:
        CouponResult with all measurements and BPAs.
    """
    config = config or load_config()

    if top_depth_mm is None and root_depth_mm is None:
        raise ValueError("At least one face must have depth data")

    # Create synthetic DepthStacks
    depth_stacks = {}
    unc_cfg = config["uncertainty"]

    for face, depth_mm in [("top", top_depth_mm), ("root", root_depth_mm)]:
        if depth_mm is not None:
            depth_stacks[face] = DepthStack(
                mean_mm=depth_mm.astype(np.float32),
                std_mm=np.zeros_like(depth_mm, dtype=np.float32),
                n_frames=n_frames,
                valid_mask=~np.isnan(depth_mm),
                split_half_se_mm=unc_cfg.get("noise_floor_mm_default", 0.015),
                naive_se_mm=unc_cfg.get("noise_floor_mm_default", 0.015),
                averaging_efficiency=1.0,
                noise_floor_mm=unc_cfg.get("noise_floor_mm_default", 0.015),
            )

    # Use top face for registration (preferred), fall back to root
    primary_face = "top" if "top" in depth_stacks else "root"
    primary_depth = depth_stacks[primary_face].mean_mm

    # Register coupon
    pose, cell_bounds = register_coupon(primary_depth, coupon_id, config)

    # Process each face
    face_results = {}
    plane_fits = []

    for face, stack in depth_stacks.items():
        result = process_face(
            stack.mean_mm,
            face,
            pose,
            cell_bounds,
            config,
        )
        result.depth_stack = stack
        face_results[face] = result
        plane_fits.append(result.plane_fit)

    # Compute uncertainty budget
    uncertainty = compute_uncertainty_budget(
        list(depth_stacks.values()),
        plane_fits,
        config,
    )

    # Build BPAs for each cell
    cell_bpas = []
    validation_outputs = []

    for cell in cell_bounds:
        measurements = []
        for face, result in face_results.items():
            cell_meas = next(
                (m for m in result.cell_measurements if m.cell_id == cell.cell_id),
                None,
            )
            if cell_meas is not None:
                measurements.append(
                    cell_measurement_to_geometry_measurement(cell_meas, "left")
                )
                measurements.append(
                    cell_measurement_to_geometry_measurement(cell_meas, "right")
                )

        if measurements:
            bpa = build_cell_bpa(measurements, uncertainty, config)
            cell_bpas.append(bpa)

            validation = assess_validation_outputs(measurements, config)
            validation_outputs.append(validation)

    return CouponResult(
        coupon_id=coupon_id,
        pose=pose,
        cell_bounds=cell_bounds,
        face_results=face_results,
        cell_bpas=cell_bpas,
        validation_outputs=validation_outputs,
        uncertainty=uncertainty,
    )
