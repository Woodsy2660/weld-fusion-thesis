"""
Seeded registration pipeline.

Task 1b PRIMARY PATH: Use manual hole seeds plus gradient-based rim refinement.

Seeds replace B1-B3 (plate isolation, search windows, candidate detection).
B4 (rim fit) runs in a 12mm window around each seed. Every measured value
comes from the algorithm, not the click position.

Usage:
    python -m src.geometry.registration_seeded --capture C11_top_r0

Or programmatically:
    from src.geometry.registration_seeded import register_capture_seeded
    pose, cloud, meta = register_capture_seeded(capture_dir, seeds)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.geometry.registration_v2 import (
    DetectedHole,
    CouponPose,
    PointCloud,
    build_rs_intrinsics,
    build_rs_extrinsics,
    FallbackExtrinsics,
    compute_median_colour,
    compute_averaged_depth,
    deproject_hole_centre,
    solve_coupon_pose,
    extract_point_cloud,
    compute_bead_centreline,
    load_config,
    HAS_PYREALSENSE,
)

from src.geometry.rim_detector import (
    RimFitResult,
    detect_rim_from_seed,
)

from src.geometry.plate_isolation import (
    create_plate_mask_from_height,
    map_plate_region_to_colour,
    add_padding_to_bbox,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HoleSeed:
    """Manual seed for a datum hole."""
    capture_id: str
    hole_type: str  # "d6" or "d8"
    px_u: int       # Full-frame pixel column
    px_v: int       # Full-frame pixel row


def load_seeds(seeds_path: Path) -> dict[str, list[HoleSeed]]:
    """Load hole seeds from CSV.

    Returns dict mapping capture_id -> list of HoleSeed.
    """
    seeds = {}
    with open(seeds_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            capture_id = row["capture_id"]
            seed = HoleSeed(
                capture_id=capture_id,
                hole_type=row["hole"],
                px_u=int(row["px_u"]),
                px_v=int(row["px_v"]),
            )
            if capture_id not in seeds:
                seeds[capture_id] = []
            seeds[capture_id].append(seed)
    return seeds


def rim_fit_to_detected_hole(
    rim_result: RimFitResult,
    expected_diameter_mm: float,
    hole_type: str,
) -> DetectedHole:
    """Convert RimFitResult to DetectedHole for pose solve compatibility."""
    # RimFitResult uses (u, v) = (col, row) convention
    # DetectedHole uses (row, col) = (v, u) convention
    u_free, v_free = rim_result.free_fit_centre
    u_fixed, v_fixed = rim_result.fixed_fit_centre

    # Compute confidence from rim fit metrics
    # Higher contrast_ratio and more edge points = higher confidence
    contrast_score = min(1.0, rim_result.contrast_ratio / 3.0)
    point_score = min(1.0, rim_result.n_edge_points / 60.0)
    residual_score = max(0, 1.0 - rim_result.fixed_fit_residual / 5.0)
    confidence = (contrast_score + point_score + residual_score) / 3.0

    return DetectedHole(
        centre_px=(v_free, u_free),  # (row, col)
        free_fit_radius_px=rim_result.free_fit_axes[0] / 2,  # From major axis
        free_fit_residual_px=rim_result.fixed_fit_residual,
        fixed_fit_centre_px=(v_fixed, u_fixed),  # (row, col)
        fixed_fit_residual_px=rim_result.fixed_fit_residual,
        estimated_diameter_mm=rim_result.free_fit_diameter_mm,
        hole_type=hole_type,
        n_edge_points=rim_result.n_edge_points,
        confidence=confidence,
    )


def detect_hole_from_seed(
    colour_img: np.ndarray,
    seed: HoleSeed,
    depth_mm: float,
    fx: float,
    config: dict,
) -> tuple[DetectedHole, RimFitResult] | tuple[None, None]:
    """Detect hole using gradient-based rim fit around seed.

    Args:
        colour_img: Full-frame colour image
        seed: Manual hole seed
        depth_mm: Approximate depth at this location
        fx: Colour camera focal length
        config: Registration config

    Returns:
        (DetectedHole, RimFitResult) or (None, None) if detection fails
    """
    det_cfg = config["hole_detection"]
    holes_cfg = config["datum_holes"]

    # Map seed hole type to config hole type
    if seed.hole_type == "d6":
        expected_diameter_mm = holes_cfg["origin"]["diameter_mm"]
        hole_type = "origin"
    elif seed.hole_type == "d8":
        expected_diameter_mm = holes_cfg["axis"]["diameter_mm"]
        hole_type = "axis"
    else:
        raise ValueError(f"Unknown hole type: {seed.hole_type}")

    # Convert to grayscale
    gray = cv2.cvtColor(colour_img, cv2.COLOR_BGR2GRAY)

    # Run rim detector from seed
    window_radius_mm = det_cfg.get("window_radius_mm", 12.0)
    contrast_threshold = det_cfg.get("contrast_threshold", 1.5)

    rim_result = detect_rim_from_seed(
        gray=gray,
        seed_px=(seed.px_u, seed.px_v),
        expected_diameter_mm=expected_diameter_mm,
        depth_mm=depth_mm,
        fx=fx,
        window_radius_mm=window_radius_mm,
        contrast_threshold=contrast_threshold,
    )

    if rim_result is None or not rim_result.accepted:
        return None, None

    # Convert to DetectedHole
    detected = rim_fit_to_detected_hole(rim_result, expected_diameter_mm, hole_type)

    return detected, rim_result


def validate_hole_pair(
    origin_hole: DetectedHole,
    axis_hole: DetectedHole,
    depth_mm: np.ndarray,
    colour_intr,
    depth_intr,
    colour_to_depth,
    depth_to_colour,
    config: dict,
    standoff_mm: float = 320.0,
) -> tuple[float, list[str]]:
    """Validate the detected hole pair.

    Returns (measured_baseline_mm, flags).
    """
    holes_cfg = config["datum_holes"]
    val_cfg = config["validation"]
    det_cfg = config["hole_detection"]

    flags = []

    # Deproject holes to 3D
    try:
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
    except Exception as e:
        return 0.0, [f"deproject_failed: {e}"]

    # Compute baseline
    baseline_mm = float(np.linalg.norm(P8 - P6) * 1000)

    expected_baseline = holes_cfg["baseline_mm"]
    pair_gate_mm = det_cfg.get("pair_gate_mm", 10.0)
    baseline_tolerance = val_cfg["baseline_tolerance_mm"]

    # Loose gate check (±10 mm)
    if abs(baseline_mm - expected_baseline) > pair_gate_mm:
        flags.append(f"loose_gate_failed_{baseline_mm:.1f}mm")

    # Tight tolerance check (±0.5 mm)
    if abs(baseline_mm - expected_baseline) > baseline_tolerance:
        flags.append(f"baseline_tolerance_exceeded_{baseline_mm:.1f}mm")

    return baseline_mm, flags


def register_capture_seeded(
    capture_dir: Path,
    seeds: list[HoleSeed],
    config: dict | None = None,
) -> tuple[CouponPose, PointCloud, dict]:
    """Full registration pipeline using manual seeds.

    This is the PRIMARY path from Task 1b. Seeds provide the search windows,
    gradient-based rim fit measures the actual hole positions.

    Args:
        capture_dir: Path to capture directory
        seeds: List of HoleSeed objects for this capture
        config: Registration config (loaded if not provided)

    Returns:
        (pose, point_cloud, metadata)
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
        import pyrealsense2 as rs
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
    fx = meta["color_intrinsics"]["fx"]

    # Detect holes from seeds
    d6_seeds = [s for s in seeds if s.hole_type == "d6"]
    d8_seeds = [s for s in seeds if s.hole_type == "d8"]

    if not d6_seeds or not d8_seeds:
        raise ValueError(f"Missing seeds: {len(d6_seeds)} d6, {len(d8_seeds)} d8")

    # Use first seed of each type
    d6_seed = d6_seeds[0]
    d8_seed = d8_seeds[0]

    # Detect using rim fit
    origin_hole, origin_rim = detect_hole_from_seed(
        colour_img, d6_seed, standoff_mm, fx, config
    )
    axis_hole, axis_rim = detect_hole_from_seed(
        colour_img, d8_seed, standoff_mm, fx, config
    )

    if origin_hole is None:
        raise ValueError(f"Rim fit failed for d6 seed at ({d6_seed.px_u}, {d6_seed.px_v})")
    if axis_hole is None:
        raise ValueError(f"Rim fit failed for d8 seed at ({d8_seed.px_u}, {d8_seed.px_v})")

    # Validate pair
    baseline_mm, pair_flags = validate_hole_pair(
        origin_hole, axis_hole, depth_mm,
        colour_intr, depth_intr, colour_to_depth, depth_to_colour, config,
        standoff_mm=standoff_mm
    )

    # Create plate mask (for point cloud extraction)
    plate_region = create_plate_mask_from_height(depth_mm)
    plate_mask_depth = plate_region.mask

    # Solve pose
    pose = solve_coupon_pose(
        origin_hole, axis_hole, depth_mm, plate_mask_depth,
        colour_intr, depth_intr, colour_to_depth, depth_to_colour,
        face_label, config, check_hole=None,
        standoff_mm=standoff_mm
    )

    # Add pair validation flags
    pose.flags.extend(pair_flags)

    # Extract point cloud
    n_valid_frames = np.full_like(depth_mm, n_frames)
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
        "registration_method": "seeded",
        "seeds": {
            "d6": {"px_u": d6_seed.px_u, "px_v": d6_seed.px_v},
            "d8": {"px_u": d8_seed.px_u, "px_v": d8_seed.px_v},
        },
        "rim_fit": {
            "origin": {
                "contrast_ratio": origin_rim.contrast_ratio,
                "n_edge_points": origin_rim.n_edge_points,
                "free_fit_diameter_mm": origin_rim.free_fit_diameter_mm,
                "fixed_fit_residual": origin_rim.fixed_fit_residual,
                "accepted": bool(origin_rim.accepted),
            },
            "axis": {
                "contrast_ratio": axis_rim.contrast_ratio,
                "n_edge_points": axis_rim.n_edge_points,
                "free_fit_diameter_mm": axis_rim.free_fit_diameter_mm,
                "fixed_fit_residual": axis_rim.fixed_fit_residual,
                "accepted": bool(axis_rim.accepted),
            },
        },
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


def create_diagnostic_overlay(
    colour_img: np.ndarray,
    seeds: list[HoleSeed],
    origin_rim: RimFitResult | None,
    axis_rim: RimFitResult | None,
    config: dict,
) -> np.ndarray:
    """Create diagnostic overlay showing seeds, search windows, and rim fits.

    Returns BGR image with overlays.
    """
    overlay = colour_img.copy()
    det_cfg = config["hole_detection"]

    # Draw seeds and search windows
    for seed in seeds:
        # Seed point (cross)
        colour = (0, 255, 0) if seed.hole_type == "d6" else (0, 0, 255)
        cv2.drawMarker(
            overlay, (seed.px_u, seed.px_v), colour,
            cv2.MARKER_CROSS, 20, 2
        )

        # Window radius (approximate - would need depth/fx for exact)
        window_r_px = int(det_cfg.get("window_radius_mm", 12.0) * 2)  # Rough estimate
        cv2.circle(overlay, (seed.px_u, seed.px_v), window_r_px, colour, 1)

    # Draw rim fit results
    if origin_rim is not None:
        u, v = origin_rim.fixed_fit_centre
        cv2.circle(overlay, (int(u), int(v)), 10, (0, 255, 0), 2)
        # Draw fitted ellipse
        axes = (int(origin_rim.free_fit_axes[0] / 2), int(origin_rim.free_fit_axes[1] / 2))
        cv2.ellipse(
            overlay,
            (int(origin_rim.free_fit_centre[0]), int(origin_rim.free_fit_centre[1])),
            axes,
            origin_rim.free_fit_angle,
            0, 360, (0, 255, 0), 1
        )
        # Label
        cv2.putText(
            overlay,
            f"D6 CR={origin_rim.contrast_ratio:.2f}",
            (int(u) + 15, int(v)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    if axis_rim is not None:
        u, v = axis_rim.fixed_fit_centre
        cv2.circle(overlay, (int(u), int(v)), 10, (0, 0, 255), 2)
        # Draw fitted ellipse
        axes = (int(axis_rim.free_fit_axes[0] / 2), int(axis_rim.free_fit_axes[1] / 2))
        cv2.ellipse(
            overlay,
            (int(axis_rim.free_fit_centre[0]), int(axis_rim.free_fit_centre[1])),
            axes,
            axis_rim.free_fit_angle,
            0, 360, (0, 0, 255), 1
        )
        # Label
        cv2.putText(
            overlay,
            f"D8 CR={axis_rim.contrast_ratio:.2f}",
            (int(u) + 15, int(v)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

    return overlay


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Seeded registration pipeline")
    parser.add_argument("--capture", required=True, help="Capture directory name (e.g., C11_top_r0)")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw" / "S1"),
                        help="Data directory")
    parser.add_argument("--seeds", default=str(REPO_ROOT / "configs" / "hole_seeds.csv"),
                        help="Seeds CSV file")
    parser.add_argument("--output", default=str(REPO_ROOT / "results" / "geometry"),
                        help="Output directory")
    parser.add_argument("--overlay", action="store_true",
                        help="Save diagnostic overlay")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    capture_dir = data_dir / args.capture
    seeds_path = Path(args.seeds)
    output_dir = Path(args.output)

    if not capture_dir.exists():
        print(f"ERROR: Capture directory not found: {capture_dir}")
        return 1

    if not seeds_path.exists():
        print(f"ERROR: Seeds file not found: {seeds_path}")
        print("Run scripts/seed_holes.py first to create seeds.")
        return 1

    # Load seeds
    all_seeds = load_seeds(seeds_path)
    capture_seeds = all_seeds.get(args.capture, [])

    if not capture_seeds:
        print(f"ERROR: No seeds found for capture {args.capture}")
        return 1

    print(f"Found {len(capture_seeds)} seeds for {args.capture}")

    # Load config
    config = load_config()

    # Run registration
    try:
        pose, cloud, meta = register_capture_seeded(capture_dir, capture_seeds, config)
    except Exception as e:
        print(f"ERROR: Registration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print results
    print("\n" + "=" * 60)
    print(f"REGISTRATION RESULTS: {args.capture}")
    print("=" * 60)
    print(f"Measured baseline: {meta['measured_baseline_mm']:.2f} mm (expected 270.0)")
    print(f"Derived face: {pose.derived_face}")
    print(f"Face label: {pose.face_label}")
    print(f"Face match: {pose.face_match}")
    print(f"Y crown mean: {pose.y_crown_mean_mm:.2f} mm")
    print(f"Y crown max: {pose.y_crown_max_mm:.2f} mm")
    print(f"Point cloud: {len(cloud.s)} points")

    if pose.flags:
        print("\nFLAGS:")
        for flag in pose.flags:
            print(f"  - {flag}")
    else:
        print("\nNo flags (registration OK)")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.capture}_registration.json"
    with open(result_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nResults saved to: {result_path}")

    # Save overlay if requested
    if args.overlay:
        colour_img = compute_median_colour(capture_dir / "color")
        d6_seeds = [s for s in capture_seeds if s.hole_type == "d6"]
        d8_seeds = [s for s in capture_seeds if s.hole_type == "d8"]

        # Re-detect for overlay
        fx = meta["standoff_mm"] / meta["standoff_mm"]  # Will need proper fx
        origin_hole, origin_rim = detect_hole_from_seed(
            colour_img, d6_seeds[0], meta["standoff_mm"],
            config["datum_holes"]["origin"]["diameter_mm"] * 10, config  # Rough fx
        )
        axis_hole, axis_rim = detect_hole_from_seed(
            colour_img, d8_seeds[0], meta["standoff_mm"],
            config["datum_holes"]["axis"]["diameter_mm"] * 10, config
        )

        overlay = create_diagnostic_overlay(
            colour_img, capture_seeds, origin_rim, axis_rim, config
        )
        overlay_path = output_dir / f"{args.capture}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"Overlay saved to: {overlay_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
