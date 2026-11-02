#!/usr/bin/env python3
"""
Full geometry pipeline run: process all captures → summary.csv

Processes all captures in data/raw/S1, groups by coupon (C01-C15),
runs the geometry pipeline on each, and outputs summary.csv with
all cell BPAs.

Uses seeded registration (manual hole seeds) for robustness.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

from src.geometry.pipeline import (
    process_face,
    compute_uncertainty_budget,
    cell_measurement_to_geometry_measurement,
    load_config,
    CouponResult,
)
from src.geometry.profile import load_depth_frames, compute_depth_stack
from src.geometry.registration_seeded import load_seeds, register_capture_seeded
from src.geometry.registration import CellBounds
from src.geometry.rules import build_cell_bpa, assess_validation_outputs


class V2PoseAdapter:
    """Adapter to make v2 CouponPose compatible with pipeline.process_face()."""

    def __init__(self, v2_pose):
        self._v2 = v2_pose
        # Compute scale from hole positions
        # NOTE: Hole positions are in color image (1920x1080)
        # Scale to depth image (1280x720) = 2/3 scale
        color_to_depth_scale = 1280.0 / 1920.0
        origin_px = (v2_pose.origin_hole.centre_px[0] * color_to_depth_scale,
                     v2_pose.origin_hole.centre_px[1] * color_to_depth_scale)
        axis_px = (v2_pose.axis_hole.centre_px[0] * color_to_depth_scale,
                   v2_pose.axis_hole.centre_px[1] * color_to_depth_scale)
        baseline_px = np.sqrt(
            (axis_px[0] - origin_px[0])**2 + (axis_px[1] - origin_px[1])**2
        )
        baseline_mm = v2_pose.measured_baseline_mm if v2_pose.measured_baseline_mm > 0 else 270.0
        self.scale_mm_per_px = baseline_mm / baseline_px

    def __getattr__(self, name):
        return getattr(self._v2, name)


def compute_cell_bounds_from_v2_pose(pose, coupon_id: str, config: dict) -> list:
    """Compute cell bounds from v2 CouponPose (registration_v2 format).

    The v2 pose has 3D coordinates. We derive cell bounds in pixel space
    using the hole positions and measured baseline.

    NOTE: Hole positions are in color image coordinates (1920x1080).
    We scale to depth image coordinates (1280x720) = 2/3 scale.
    """
    cells_cfg = config["cells"]
    cell_width = cells_cfg["width_mm"]
    first_excluded = cells_cfg["first_excluded_mm"]
    last_excluded = cells_cfg["last_excluded_mm"]
    coupon_length = cells_cfg["coupon_length_mm"]

    # Get hole pixel positions from v2 pose (origin = d6, axis = d8)
    # These are in (row, col) format in COLOR image space (1920x1080)
    color_origin_px = pose.origin_hole.centre_px  # (row, col)
    color_axis_px = pose.axis_hole.centre_px      # (row, col)

    # Scale from color (1920x1080) to depth (1280x720)
    color_to_depth_scale = 1280.0 / 1920.0  # = 2/3
    origin_px = (color_origin_px[0] * color_to_depth_scale,
                 color_origin_px[1] * color_to_depth_scale)
    axis_px = (color_axis_px[0] * color_to_depth_scale,
               color_axis_px[1] * color_to_depth_scale)

    # Compute scale: pixels per mm along the s-axis
    # The holes are 270mm apart (baseline)
    baseline_px = np.sqrt(
        (axis_px[0] - origin_px[0])**2 + (axis_px[1] - origin_px[1])**2
    )
    baseline_mm = pose.measured_baseline_mm if pose.measured_baseline_mm > 0 else 270.0
    scale_px_per_mm = baseline_px / baseline_mm

    # Direction vector from origin to axis (unit)
    dx = axis_px[1] - origin_px[1]  # col direction
    dy = axis_px[0] - origin_px[0]  # row direction
    length = np.sqrt(dx**2 + dy**2)
    unit_s = np.array([dx / length, dy / length])  # (col, row) direction

    # Cell boundaries in s (mm)
    s_start = first_excluded
    s_end = coupon_length - last_excluded
    n_cells = int((s_end - s_start) / cell_width)

    bounds = []
    for i in range(n_cells):
        s_cell_start = s_start + i * cell_width
        s_cell_end = s_cell_start + cell_width
        s_cell_centre = (s_cell_start + s_cell_end) / 2

        # Convert s to pixel offset from origin hole
        # origin_hole is at s ~ 15mm (from config)
        origin_s_mm = 15.0  # From config: registration.datum_holes.origin.s_mm
        offset_start = (s_cell_start - origin_s_mm) * scale_px_per_mm
        offset_end = (s_cell_end - origin_s_mm) * scale_px_per_mm

        # Pixel positions (col)
        col_start = origin_px[1] + offset_start * unit_s[0]
        col_end = origin_px[1] + offset_end * unit_s[0]

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


DATA_DIR = Path("data/raw/S1")
OUTPUT_DIR = Path("results/geometry")
SEEDS_PATH = Path("configs/hole_seeds.csv")
WELD_AXIS_SEEDS_PATH = Path("configs/weld_axis_seeds.csv")


def parse_capture_name(name: str) -> dict | None:
    """Parse capture directory name: {coupon}_{face}_{repeat}."""
    match = re.match(r"^(C\d+|REF)_(top|root)_r(\d+)$", name)
    if not match:
        return None
    return {
        "coupon": match.group(1),
        "face": match.group(2),
        "repeat": int(match.group(3)),
    }


def find_captures(data_dir: Path) -> dict:
    """Group captures by coupon, returning primary (r0) capture directories."""
    coupons = defaultdict(dict)

    for capture_dir in sorted(data_dir.iterdir()):
        if not capture_dir.is_dir():
            continue

        parsed = parse_capture_name(capture_dir.name)
        if not parsed:
            continue

        # Skip REF captures (reference target)
        if parsed["coupon"] == "REF":
            continue

        # Use r0 as primary; store full capture directory (not just depth/)
        if parsed["repeat"] == 0:
            coupons[parsed["coupon"]][parsed["face"]] = capture_dir

    return dict(coupons)


def load_weld_axis_seeds(seeds_path: Path) -> dict:
    """Load weld axis seeds (centreline points) from CSV.

    Returns dict mapping capture_id -> [(u1, v1), (u2, v2)]
    """
    seeds = {}
    with open(seeds_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            capture_id = row["capture_id"]
            if capture_id not in seeds:
                seeds[capture_id] = []
            seeds[capture_id].append((int(row["px_u"]), int(row["px_v"])))
    return seeds


def process_coupon_seeded(
    coupon_id: str,
    faces: dict,
    hole_seeds: dict,
    weld_axis_seeds: dict,
    config: dict,
) -> CouponResult:
    """Process a single coupon using seeded registration."""
    from src.geometry.pipeline import (
        FaceResult, weld_axis_centre_rows
    )

    depth_stacks = {}
    face_results = {}
    plane_fits = []
    poses = {}
    first_cell_bounds = None  # Store cell bounds from first face

    # Process each face
    for face, capture_dir in faces.items():
        capture_id = capture_dir.name

        # Load depth frames
        depth_frames = load_depth_frames(capture_dir / "depth")
        if not depth_frames:
            continue
        depth_stack = compute_depth_stack(depth_frames, config)
        depth_stacks[face] = depth_stack

        # Get seeds for this capture
        capture_hole_seeds = hole_seeds.get(capture_id, [])
        if not capture_hole_seeds:
            raise ValueError(f"No hole seeds for {capture_id}")

        # Seeded registration (uses its own config from configs/registration.yaml)
        v2_pose, cloud, meta = register_capture_seeded(capture_dir, capture_hole_seeds)
        pose = V2PoseAdapter(v2_pose)  # Wrap for compatibility with process_face
        poses[face] = pose

        # Compute cell bounds from v2 pose (use first face's bounds for all)
        cell_bounds = compute_cell_bounds_from_v2_pose(v2_pose, coupon_id, config)
        if first_cell_bounds is None:
            first_cell_bounds = cell_bounds

        # Get weld axis seeds for centreline
        axis_seeds = weld_axis_seeds.get(capture_id, [])
        if len(axis_seeds) >= 2:
            centre_rows = weld_axis_centre_rows(axis_seeds[:2], depth_stack.mean_mm.shape[1])
        else:
            centre_rows = None

        # Process face
        result = process_face(
            depth_stack.mean_mm,
            face,
            pose,
            cell_bounds,
            config,
            centre_rows=centre_rows,
        )
        result.depth_stack = depth_stack
        face_results[face] = result
        plane_fits.append(result.plane_fit)

    if not face_results:
        raise ValueError("No valid faces processed")

    # Use cell bounds from first face
    pose = list(poses.values())[0]
    cell_bounds = first_cell_bounds

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


def run_full_pipeline():
    """Run geometry pipeline on all coupons and produce summary.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL GEOMETRY PIPELINE RUN (SEEDED REGISTRATION)")
    print("=" * 70)

    # Load seeds
    if not SEEDS_PATH.exists():
        print(f"ERROR: Hole seeds not found at {SEEDS_PATH}")
        return
    hole_seeds = load_seeds(SEEDS_PATH)
    print(f"Loaded hole seeds for {len(hole_seeds)} captures")

    weld_axis_seeds = {}
    if WELD_AXIS_SEEDS_PATH.exists():
        weld_axis_seeds = load_weld_axis_seeds(WELD_AXIS_SEEDS_PATH)
        print(f"Loaded weld axis seeds for {len(weld_axis_seeds)} captures")

    # Find all captures
    coupons = find_captures(DATA_DIR)
    print(f"\nFound {len(coupons)} coupons with captures")
    for coupon_id, faces in sorted(coupons.items()):
        face_list = ", ".join(sorted(faces.keys()))
        print(f"  {coupon_id}: {face_list}")

    print("\n" + "-" * 70)
    print("Processing coupons...")
    print("-" * 70)

    config = load_config()
    all_bpas = []
    errors = []

    for coupon_id, faces in sorted(coupons.items()):
        print(f"\n{coupon_id}:", end=" ")

        try:
            result = process_coupon_seeded(
                coupon_id, faces, hole_seeds, weld_axis_seeds, config
            )
            n_cells = len(result.cell_bpas)
            n_vacuous = sum(1 for bpa in result.cell_bpas if bpa.is_vacuous)
            print(f"{n_cells} cells ({n_vacuous} vacuous)")

            for bpa in result.cell_bpas:
                all_bpas.append({
                    "coupon_id": coupon_id,
                    "cell_id": bpa.cell_id,
                    "m_U": bpa.m_U,
                    "m_NPLC": bpa.m_NPLC,
                    "is_vacuous": bpa.is_vacuous,
                    "governing_measurand": bpa.governing_measurand,
                    "governing_undercut_mm": bpa.governing_undercut_mm,
                    "exceedance_prob": bpa.exceedance_prob,
                    "u_total_mm": bpa.u_total_mm,
                    **{f"p_{k}": v for k, v in (bpa.exceedance_by_measurand or {}).items()},
                })

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append((coupon_id, str(e)))

    # Write summary CSV
    csv_path = OUTPUT_DIR / "summary.csv"
    if all_bpas:
        # Collect all possible columns
        all_cols = set()
        for bpa in all_bpas:
            all_cols.update(bpa.keys())
        cols = sorted(all_cols)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_bpas)

        print(f"\n\nWrote {len(all_bpas)} cell BPAs to {csv_path}")

    # Write JSON with full details
    json_path = OUTPUT_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_coupons": len(coupons),
            "n_cells": len(all_bpas),
            "n_vacuous": sum(1 for b in all_bpas if b.get("is_vacuous")),
            "errors": errors,
            "bpas": all_bpas,
        }, f, indent=2, default=str)
    print(f"Wrote full results to {json_path}")

    if errors:
        print(f"\n{len(errors)} errors:")
        for coupon, err in errors:
            print(f"  {coupon}: {err}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    non_vacuous = [b for b in all_bpas if not b.get("is_vacuous")]
    if non_vacuous:
        m_U_vals = [b["m_U"] for b in non_vacuous]
        print(f"\nm({{U}}) distribution (non-vacuous cells):")
        print(f"  min:    {min(m_U_vals):.4f}")
        print(f"  median: {sorted(m_U_vals)[len(m_U_vals)//2]:.4f}")
        print(f"  max:    {max(m_U_vals):.4f}")
        print(f"  >0.5:   {sum(1 for v in m_U_vals if v > 0.5)} cells")

        # Count by governing measurand
        gov_counts = defaultdict(int)
        for b in non_vacuous:
            gov_counts[b.get("governing_measurand", "unknown")] += 1
        print(f"\nGoverning measurand counts:")
        for k, v in sorted(gov_counts.items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    run_full_pipeline()
