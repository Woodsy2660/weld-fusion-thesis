#!/usr/bin/env python3
"""
Diagnostic script to visualize hole detection results.

Generates annotated images showing what the detector found vs what was expected.
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.geometry.registration_v2 import (
    load_config, compute_median_colour, compute_averaged_depth,
    create_plate_mask_from_depth, map_depth_mask_to_colour,
    detect_datum_holes_in_colour, detect_hole_candidates,
    build_rs_intrinsics, build_rs_extrinsics,
)


def diagnose_capture(capture_dir: Path, output_dir: Path, config: dict):
    """Diagnose hole detection on one capture."""
    print(f"\nDiagnosing: {capture_dir.name}")

    # Load metadata
    meta_path = capture_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # Build intrinsics
    depth_intr = build_rs_intrinsics(meta["depth_intrinsics"])
    colour_intr = build_rs_intrinsics(meta["color_intrinsics"])
    depth_to_colour = build_rs_extrinsics(meta["depth_to_color_extrinsics"])

    # Load frames
    colour_img = compute_median_colour(capture_dir / "color")
    depth_mm, _, _ = compute_averaged_depth(capture_dir / "depth", meta.get("depth_units", 0.0001))

    standoff_mm = meta.get("standoff_mm_from_depth", 320.0)

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

    # Find all candidates (before filtering)
    candidates = detect_hole_candidates(colour_img, plate_mask_colour, config)
    print(f"  Found {len(candidates)} dark blob candidates")

    # Detect holes (with filtering)
    holes = detect_datum_holes_in_colour(
        colour_img, plate_mask_colour,
        meta["color_intrinsics"], standoff_mm, config
    )
    print(f"  Classified {len(holes)} as holes:")
    for h in holes:
        print(f"    {h.hole_type}: centre=({h.centre_px[0]:.0f}, {h.centre_px[1]:.0f}), "
              f"r={h.free_fit_radius_px:.1f}px, conf={h.confidence:.2f}")

    # Create diagnostic image
    diag = colour_img.copy()

    # Draw plate mask boundary
    mask_edges = cv2.Canny(plate_mask_colour.astype(np.uint8) * 255, 100, 200)
    diag[mask_edges > 0] = [0, 255, 255]  # Yellow mask boundary

    # Draw all candidates in blue
    for contour, centre in candidates:
        cv2.drawContours(diag, [contour], 0, (255, 0, 0), 1)  # Blue
        cv2.circle(diag, (int(centre[0]), int(centre[1])), 3, (255, 0, 0), -1)

    # Draw detected holes
    colors = {"origin": (0, 255, 0), "axis": (0, 0, 255), "check": (255, 255, 0)}
    for h in holes:
        color = colors.get(h.hole_type, (255, 255, 255))
        cv2.circle(diag, (int(h.centre_px[1]), int(h.centre_px[0])),
                  int(h.free_fit_radius_px), color, 2)
        cv2.putText(diag, h.hole_type[0].upper(),
                   (int(h.centre_px[1]) - 10, int(h.centre_px[0]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw expected pixel positions (approximate)
    fx = meta["color_intrinsics"]["fx"]
    ppx = meta["color_intrinsics"]["ppx"]
    ppy = meta["color_intrinsics"]["ppy"]

    # Expected positions assuming camera is centered over coupon
    # This is approximate - just for visualization
    expected = {
        "origin": {"s_mm": 15, "y_mm": 50, "d_mm": 6},
        "axis": {"s_mm": 285, "y_mm": 50, "d_mm": 8},
    }

    # Add legend
    legend_y = 30
    cv2.putText(diag, "Blue: candidates", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(diag, "Green: origin (d6)", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(diag, "Red: axis (d8)", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(diag, "Yellow: check (d10)", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(diag, f"Standoff: {standoff_mm:.0f}mm", (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save
    output_path = output_dir / f"{capture_dir.name}_holes.png"
    cv2.imwrite(str(output_path), diag)
    print(f"  Saved: {output_path}")

    # Also save the thresholded mask for inspection
    gray = cv2.cvtColor(colour_img, cv2.COLOR_BGR2GRAY)
    gray_masked = gray.copy()
    gray_masked[~plate_mask_colour] = 255

    det_cfg = config["hole_detection"]
    thresh = cv2.adaptiveThreshold(
        gray_masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, det_cfg["adaptive_block_size_px"], det_cfg["adaptive_threshold_c"]
    )

    thresh_path = output_dir / f"{capture_dir.name}_thresh.png"
    cv2.imwrite(str(thresh_path), thresh)

    return len(holes)


def main():
    config = load_config()
    data_dir = REPO_ROOT / "data" / "raw" / "S1"
    output_dir = REPO_ROOT / "results" / "diagnostics" / "hole_detection"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Focus on a few key captures
    test_captures = ["C11_top_r0", "C11_root_r0", "C01_top_r0"]

    for name in test_captures:
        capture_dir = data_dir / name
        if capture_dir.exists():
            diagnose_capture(capture_dir, output_dir, config)


if __name__ == "__main__":
    main()
