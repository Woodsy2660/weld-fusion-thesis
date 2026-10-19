#!/usr/bin/env python3
"""
Step 3: Scale-independent checks.

For each face, compute:
(a) baseline / plate_length × 300 (expect 270 if holes are per drawing)
(b) C11 only: plate step height above board (expect 10.0mm)
(c) Plate width in 3D (expect 140mm)
"""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.geometry.registration_seeded import (
    load_seeds, load_config, compute_median_colour, compute_averaged_depth,
    detect_hole_from_seed, validate_hole_pair, create_plate_mask_from_height,
)
from src.geometry.registration_v2 import (
    build_rs_intrinsics, build_rs_extrinsics,
    FallbackIntrinsics, FallbackExtrinsics,
    deproject_pixel_to_point,
)
from src.geometry.plate_isolation import fit_board_depth_ransac


def compute_plate_dimensions(depth_mm, depth_intr, standoff_mm):
    """Compute plate dimensions in 3D from depth map."""
    h, w = depth_mm.shape

    # Find plate region (10mm above board)
    z_board, n_valid = fit_board_depth_ransac(depth_mm, border_fraction=0.1)

    if z_board is None:
        return None, None, None

    plate_threshold = z_board - 5.0  # Plate is ~10mm above board
    plate_mask = depth_mm < plate_threshold

    # Find bounding box of plate in pixels
    rows, cols = np.where(plate_mask)
    if len(rows) == 0:
        return None, None, None

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    # Deproject corners to 3D
    z_plate = float(np.nanmedian(depth_mm[plate_mask])) / 1000.0  # metres

    # Use depth frame for deprojection
    corner_tl = deproject_pixel_to_point(depth_intr, [col_min, row_min], z_plate)
    corner_tr = deproject_pixel_to_point(depth_intr, [col_max, row_min], z_plate)
    corner_bl = deproject_pixel_to_point(depth_intr, [col_min, row_max], z_plate)
    corner_br = deproject_pixel_to_point(depth_intr, [col_max, row_max], z_plate)

    # Plate length (along weld, horizontal in image)
    length_top = np.linalg.norm(np.array(corner_tr) - np.array(corner_tl)) * 1000
    length_bot = np.linalg.norm(np.array(corner_br) - np.array(corner_bl)) * 1000
    plate_length_mm = (length_top + length_bot) / 2

    # Plate width (across weld, vertical in image)
    width_left = np.linalg.norm(np.array(corner_bl) - np.array(corner_tl)) * 1000
    width_right = np.linalg.norm(np.array(corner_br) - np.array(corner_tr)) * 1000
    plate_width_mm = (width_left + width_right) / 2

    # Step height (board depth - plate depth)
    step_height_mm = z_board - float(np.nanmedian(depth_mm[plate_mask]))

    return plate_length_mm, plate_width_mm, step_height_mm


def main():
    config = load_config()
    seeds = load_seeds(Path("configs/hole_seeds.csv"))
    data_dir = Path("data/raw/S1")

    print("=" * 70)
    print("SCALE-INDEPENDENT CHECKS")
    print("=" * 70)
    print()
    print("Expected values:")
    print("  (a) baseline/plate_length × 300 = 270 (holes at s=15 and s=285)")
    print("  (b) Plate step height = 10.0mm (plate thickness)")
    print("  (c) Plate width = 140mm")
    print()

    results = []

    for capture_id in sorted(seeds.keys()):
        capture_dir = data_dir / capture_id
        if not capture_dir.exists():
            continue

        with open(capture_dir / "meta.json") as f:
            meta = json.load(f)

        depth_intr = build_rs_intrinsics(meta["depth_intrinsics"])
        colour_intr = build_rs_intrinsics(meta["color_intrinsics"])
        depth_to_colour = build_rs_extrinsics(meta["depth_to_color_extrinsics"])

        R = np.array(depth_to_colour.rotation).reshape(3, 3)
        t = np.array(depth_to_colour.translation)
        R_inv = R.T
        t_inv = -R_inv @ t
        colour_to_depth = FallbackExtrinsics(
            rotation=R_inv.flatten().tolist(),
            translation=t_inv.tolist(),
        )

        depth_mm, _, _ = compute_averaged_depth(
            capture_dir / "depth",
            meta.get("depth_units", 0.0001)
        )
        colour_img = compute_median_colour(capture_dir / "color")

        standoff = meta.get("standoff_mm_from_depth", 320.0)
        fx = meta["color_intrinsics"]["fx"]

        capture_seeds = seeds[capture_id]
        d6_seed = [s for s in capture_seeds if s.hole_type == 'd6'][0]
        d8_seed = [s for s in capture_seeds if s.hole_type == 'd8'][0]

        try:
            d6_hole, d6_rim = detect_hole_from_seed(colour_img, d6_seed, standoff, fx, config)
            d8_hole, d8_rim = detect_hole_from_seed(colour_img, d8_seed, standoff, fx, config)

            baseline_mm, flags = validate_hole_pair(
                d6_hole, d8_hole, depth_mm,
                colour_intr, depth_intr, colour_to_depth, depth_to_colour,
                config, standoff_mm=standoff
            )

            # Get plate dimensions
            plate_length, plate_width, step_height = compute_plate_dimensions(
                depth_mm, depth_intr, standoff
            )

            if plate_length and plate_length > 0:
                ratio = baseline_mm / plate_length * 300
            else:
                ratio = None

            result = {
                "capture_id": capture_id,
                "baseline_mm": baseline_mm,
                "plate_length_mm": plate_length,
                "plate_width_mm": plate_width,
                "step_height_mm": step_height,
                "baseline_ratio_x300": ratio,
            }
            results.append(result)

            # Print C11 specifically
            if capture_id.startswith("C11"):
                print(f"\n{capture_id}:")
                print(f"  Baseline: {baseline_mm:.2f}mm")
                print(f"  Plate length: {plate_length:.1f}mm" if plate_length else "  Plate length: N/A")
                print(f"  Plate width: {plate_width:.1f}mm (expect 140)" if plate_width else "  Plate width: N/A")
                print(f"  Step height: {step_height:.2f}mm (expect 10.0)" if step_height else "  Step height: N/A")
                if ratio:
                    print(f"  baseline/length × 300 = {ratio:.1f} (expect 270)")

        except Exception as e:
            print(f"{capture_id}: ERROR - {e}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baselines = [r["baseline_mm"] for r in results if r["baseline_mm"]]
    ratios = [r["baseline_ratio_x300"] for r in results if r["baseline_ratio_x300"]]
    widths = [r["plate_width_mm"] for r in results if r["plate_width_mm"]]
    heights = [r["step_height_mm"] for r in results if r["step_height_mm"]]

    print(f"\nBaseline (expect 270mm):")
    print(f"  Mean: {np.mean(baselines):.2f}mm, Std: {np.std(baselines):.2f}mm")

    print(f"\nBaseline/PlateLength × 300 (expect 270):")
    print(f"  Mean: {np.mean(ratios):.1f}, Std: {np.std(ratios):.1f}")

    print(f"\nPlate width (expect 140mm):")
    print(f"  Mean: {np.mean(widths):.1f}mm, Std: {np.std(widths):.1f}mm")

    print(f"\nStep height (expect 10.0mm):")
    print(f"  Mean: {np.mean(heights):.2f}mm, Std: {np.std(heights):.2f}mm")


if __name__ == "__main__":
    main()
