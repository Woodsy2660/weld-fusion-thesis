#!/usr/bin/env python3
"""
Step 2: Intrinsics audit for C11_top_r0.

Print fx, fy, ppx, ppy and distortion for every deprojection call,
comparing to meta.json values.
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
    FallbackIntrinsics, FallbackExtrinsics,
)


def main():
    capture_id = "C11_top_r0"
    capture_dir = Path("data/raw/S1") / capture_id

    # Load meta.json
    with open(capture_dir / "meta.json") as f:
        meta = json.load(f)

    print("=" * 70)
    print(f"INTRINSICS AUDIT: {capture_id}")
    print("=" * 70)

    print("\n### meta.json values ###")
    ci = meta["color_intrinsics"]
    di = meta["depth_intrinsics"]

    print(f"\nCOLOR INTRINSICS:")
    print(f"  fx     = {ci['fx']}")
    print(f"  fy     = {ci['fy']}")
    print(f"  ppx    = {ci['ppx']}")
    print(f"  ppy    = {ci['ppy']}")
    print(f"  width  = {ci['width']}")
    print(f"  height = {ci['height']}")
    if "coeffs" in ci:
        print(f"  coeffs = {ci['coeffs']}")

    print(f"\nDEPTH INTRINSICS:")
    print(f"  fx     = {di['fx']}")
    print(f"  fy     = {di['fy']}")
    print(f"  ppx    = {di['ppx']}")
    print(f"  ppy    = {di['ppy']}")
    print(f"  width  = {di['width']}")
    print(f"  height = {di['height']}")
    if "coeffs" in di:
        print(f"  coeffs = {di['coeffs']}")

    # Build intrinsics objects
    colour_intr = build_rs_intrinsics(ci)
    depth_intr = build_rs_intrinsics(di)

    print("\n### Built intrinsics objects ###")
    print(f"\ncolour_intr (passed to deproject_pixel_to_point for holes):")
    print(f"  fx     = {colour_intr.fx}")
    print(f"  fy     = {colour_intr.fy}")
    print(f"  ppx    = {colour_intr.ppx}")
    print(f"  ppy    = {colour_intr.ppy}")
    print(f"  width  = {colour_intr.width}")
    print(f"  height = {colour_intr.height}")

    print(f"\ndepth_intr (used for depth frame projections):")
    print(f"  fx     = {depth_intr.fx}")
    print(f"  fy     = {depth_intr.fy}")
    print(f"  ppx    = {depth_intr.ppx}")
    print(f"  ppy    = {depth_intr.ppy}")
    print(f"  width  = {depth_intr.width}")
    print(f"  height = {depth_intr.height}")

    # Check for HFOV issues
    print("\n### HFOV Analysis ###")
    # D435i nominal HFOV at 1280 depth = 87°
    # fx_nominal = width / (2 * tan(HFOV/2))
    import math
    hfov_87 = math.radians(87)
    fx_nominal_depth = 1280 / (2 * math.tan(hfov_87 / 2))
    fx_nominal_color = 1920 / (2 * math.tan(hfov_87 / 2))

    print(f"Nominal fx (HFOV=87°, depth 1280): {fx_nominal_depth:.3f}")
    print(f"Nominal fx (HFOV=87°, color 1920): {fx_nominal_color:.3f}")
    print(f"Actual depth fx:  {di['fx']:.3f}")
    print(f"Actual color fx:  {ci['fx']:.3f}")

    # Calculate what baseline we'd get with nominal vs actual
    expected_baseline = 270.0
    ratio = di['fx'] / fx_nominal_depth
    scaled_baseline = expected_baseline * ratio
    print(f"\nIf using actual intrinsics on nominal hole spacing:")
    print(f"  depth fx ratio (actual/nominal): {ratio:.4f}")
    print(f"  Would measure: {scaled_baseline:.1f}mm instead of {expected_baseline}mm")

    # The user's formula: 270 × 638.594/674.4 = 255.7
    # Let's check what 674.4 would be
    print(f"\nUser's hypothesis check:")
    print(f"  270 × {di['fx']:.3f} / 674.4 = {270 * di['fx'] / 674.4:.1f}mm")
    print(f"  (Observed baseline was ~256.6mm)")

    # Load and run detection
    config = load_config()
    seeds = load_seeds(Path("configs/hole_seeds.csv"))
    capture_seeds = seeds[capture_id]

    colour_img = compute_median_colour(capture_dir / "color")
    depth_mm, _, _ = compute_averaged_depth(
        capture_dir / "depth",
        meta.get("depth_units", 0.0001)
    )

    standoff = meta.get("standoff_mm_from_depth", 320.0)
    fx = ci["fx"]  # This is what gets passed to rim detection

    print(f"\n### Values passed to detect_hole_from_seed ###")
    print(f"  standoff_mm = {standoff}")
    print(f"  fx (color)  = {fx}")

    d6_seed = [s for s in capture_seeds if s.hole_type == 'd6'][0]
    d8_seed = [s for s in capture_seeds if s.hole_type == 'd8'][0]

    d6_hole, d6_rim = detect_hole_from_seed(colour_img, d6_seed, standoff, fx, config)
    d8_hole, d8_rim = detect_hole_from_seed(colour_img, d8_seed, standoff, fx, config)

    print(f"\n### Detection results ###")
    print(f"D6 seed: ({d6_seed.px_u}, {d6_seed.px_v})")
    print(f"D6 free-fit diameter: {d6_rim.free_fit_diameter_mm:.2f}mm (expect 6.0)")
    print(f"D6 fixed-fit centre: ({d6_rim.fixed_fit_centre[0]:.1f}, {d6_rim.fixed_fit_centre[1]:.1f})")

    print(f"\nD8 seed: ({d8_seed.px_u}, {d8_seed.px_v})")
    print(f"D8 free-fit diameter: {d8_rim.free_fit_diameter_mm:.2f}mm (expect 8.0)")
    print(f"D8 fixed-fit centre: ({d8_rim.fixed_fit_centre[0]:.1f}, {d8_rim.fixed_fit_centre[1]:.1f})")

    # Build extrinsics
    depth_to_colour = build_rs_extrinsics(meta["depth_to_color_extrinsics"])
    R = np.array(depth_to_colour.rotation).reshape(3, 3)
    t = np.array(depth_to_colour.translation)
    R_inv = R.T
    t_inv = -R_inv @ t

    colour_to_depth = FallbackExtrinsics(
        rotation=R_inv.flatten().tolist(),
        translation=t_inv.tolist(),
    )

    # Run validation
    baseline_mm, flags = validate_hole_pair(
        d6_hole, d8_hole, depth_mm,
        colour_intr, depth_intr, colour_to_depth, depth_to_colour,
        config, standoff_mm=standoff
    )

    print(f"\n### Baseline result ###")
    print(f"Measured baseline: {baseline_mm:.2f}mm")
    print(f"Expected baseline: 270.0mm")
    print(f"Error: {baseline_mm - 270.0:+.2f}mm")
    print(f"Flags: {flags}")


if __name__ == "__main__":
    main()
