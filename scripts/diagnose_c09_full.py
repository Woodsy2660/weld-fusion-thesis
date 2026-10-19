#!/usr/bin/env python3
"""
Full diagnostic for C09_top_r0 using actual validation code path.
"""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.geometry.registration_seeded import (
    load_seeds, load_config, compute_median_colour, compute_averaged_depth,
    detect_hole_from_seed, validate_hole_pair,
)
from src.geometry.registration_v2 import (
    build_rs_intrinsics, build_rs_extrinsics,
    FallbackExtrinsics,
    deproject_hole_centre,
)


def main():
    capture_id = "C09_top_r0"
    capture_dir = Path("data/raw/S1") / capture_id

    with open(capture_dir / "meta.json") as f:
        meta = json.load(f)

    config = load_config()
    seeds = load_seeds(Path("configs/hole_seeds.csv"))

    colour_img = compute_median_colour(capture_dir / "color")
    depth_mm, _, _ = compute_averaged_depth(
        capture_dir / "depth",
        meta.get("depth_units", 0.0001)
    )

    colour_intr = build_rs_intrinsics(meta["color_intrinsics"])
    depth_intr = build_rs_intrinsics(meta["depth_intrinsics"])
    depth_to_colour = build_rs_extrinsics(meta["depth_to_color_extrinsics"])

    R = np.array(depth_to_colour.rotation).reshape(3, 3)
    t = np.array(depth_to_colour.translation)
    R_inv = R.T
    t_inv = -R_inv @ t
    colour_to_depth = FallbackExtrinsics(
        rotation=R_inv.flatten().tolist(),
        translation=t_inv.tolist(),
    )

    standoff = meta.get("standoff_mm_from_depth", 320.0)
    fx = meta["color_intrinsics"]["fx"]

    print(f"Capture: {capture_id}")
    print(f"Standoff: {standoff:.1f}mm")
    print()

    # Get seeds
    capture_seeds = seeds[capture_id]
    d6_seed = [s for s in capture_seeds if s.hole_type == 'd6'][0]
    d8_seed = [s for s in capture_seeds if s.hole_type == 'd8'][0]

    print(f"D6 seed: ({d6_seed.px_u}, {d6_seed.px_v})")
    print(f"D8 seed: ({d8_seed.px_u}, {d8_seed.px_v})")

    # Detect holes
    d6_hole, d6_rim = detect_hole_from_seed(colour_img, d6_seed, standoff, fx, config)
    d8_hole, d8_rim = detect_hole_from_seed(colour_img, d8_seed, standoff, fx, config)

    print(f"\nD6 hole_type: {d6_hole.hole_type}")
    print(f"D6 fixed_fit_centre_px: {d6_hole.fixed_fit_centre_px}")
    print(f"D8 hole_type: {d8_hole.hole_type}")
    print(f"D8 fixed_fit_centre_px: {d8_hole.fixed_fit_centre_px}")

    # Deproject using actual function
    print("\n--- Deprojecting D6 (origin) ---")
    try:
        P6 = deproject_hole_centre(
            d6_hole, depth_mm, colour_intr, depth_intr,
            colour_to_depth, depth_to_colour, config,
            standoff_mm=standoff
        )
        print(f"D6 3D: [{P6[0]*1000:.1f}, {P6[1]*1000:.1f}, {P6[2]*1000:.1f}] mm")
    except Exception as e:
        print(f"D6 deproject FAILED: {e}")
        P6 = None

    print("\n--- Deprojecting D8 (axis) ---")
    try:
        P8 = deproject_hole_centre(
            d8_hole, depth_mm, colour_intr, depth_intr,
            colour_to_depth, depth_to_colour, config,
            standoff_mm=standoff
        )
        print(f"D8 3D: [{P8[0]*1000:.1f}, {P8[1]*1000:.1f}, {P8[2]*1000:.1f}] mm")
    except Exception as e:
        print(f"D8 deproject FAILED: {e}")
        P8 = None

    if P6 is not None and P8 is not None:
        baseline = np.linalg.norm(P8 - P6) * 1000
        print(f"\nBaseline: {baseline:.2f}mm (expect ~256mm)")
        print(f"D6 Z: {P6[2]*1000:.1f}mm")
        print(f"D8 Z: {P8[2]*1000:.1f}mm")

    # Also run validate_hole_pair
    print("\n--- Running validate_hole_pair ---")
    baseline_mm, flags = validate_hole_pair(
        d6_hole, d8_hole, depth_mm,
        colour_intr, depth_intr, colour_to_depth, depth_to_colour,
        config, standoff_mm=standoff
    )
    print(f"Baseline from validate_hole_pair: {baseline_mm:.2f}mm")
    print(f"Flags: {flags}")


if __name__ == "__main__":
    main()
