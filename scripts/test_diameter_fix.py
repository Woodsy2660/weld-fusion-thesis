#!/usr/bin/env python3
"""Test diameter fix and generate overlay visualization."""

import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.geometry.registration_seeded import (
    load_seeds, detect_hole_from_seed, load_config, compute_median_colour
)


def create_hole_overlay(colour_img, seed, rim_result, expected_d_mm, label):
    """Create zoomed overlay showing fitted circle."""
    # Crop region around seed
    crop_size = 100
    x0 = max(0, seed.px_u - crop_size)
    y0 = max(0, seed.px_v - crop_size)
    x1 = min(colour_img.shape[1], seed.px_u + crop_size)
    y1 = min(colour_img.shape[0], seed.px_v + crop_size)

    crop = colour_img[y0:y1, x0:x1].copy()

    # Scale up for visibility
    scale = 4
    crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale))

    # Draw seed point (yellow)
    seed_x = (seed.px_u - x0) * scale
    seed_y = (seed.px_v - y0) * scale
    cv2.circle(crop, (int(seed_x), int(seed_y)), 3, (0, 255, 255), -1)

    # Draw fitted circle (green for free fit)
    if rim_result.free_fit_centre:
        cx = (rim_result.free_fit_centre[0] - x0) * scale
        cy = (rim_result.free_fit_centre[1] - y0) * scale
        # Major axis from free fit
        r_px = rim_result.free_fit_axes[0] / 2 * scale  # axes are diameters
        cv2.circle(crop, (int(cx), int(cy)), int(r_px), (0, 255, 0), 2)

    # Draw fixed-radius circle (cyan)
    if rim_result.fixed_fit_centre:
        cx = (rim_result.fixed_fit_centre[0] - x0) * scale
        cy = (rim_result.fixed_fit_centre[1] - y0) * scale
        # Use expected diameter for fixed radius display
        cv2.circle(crop, (int(cx), int(cy)), 5, (255, 255, 0), -1)

    # Add text
    text = f"{label}: {rim_result.free_fit_diameter_mm:.2f}mm (expect {expected_d_mm}mm)"
    cv2.putText(crop, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(crop, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return crop


def main():
    config = load_config()
    seeds = load_seeds(Path('configs/hole_seeds.csv'))
    data_dir = Path('data/raw/S1')
    output_dir = Path('results/geometry/diameter_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    d6_errors = []
    d8_errors = []
    failures = []

    # Test C01_top_r0 with overlay
    test_capture = 'C01_top_r0'
    capture_dir = data_dir / test_capture

    with open(capture_dir / 'meta.json') as f:
        meta = json.load(f)

    colour_img = compute_median_colour(capture_dir / 'color')
    fx = meta['color_intrinsics']['fx']
    standoff = meta.get('standoff_mm_from_depth', 320.0)

    capture_seeds = seeds[test_capture]
    d6_seed = [s for s in capture_seeds if s.hole_type == 'd6'][0]
    d8_seed = [s for s in capture_seeds if s.hole_type == 'd8'][0]

    d6_hole, d6_rim = detect_hole_from_seed(colour_img, d6_seed, standoff, fx, config)
    d8_hole, d8_rim = detect_hole_from_seed(colour_img, d8_seed, standoff, fx, config)

    # Create overlays
    d6_overlay = create_hole_overlay(colour_img, d6_seed, d6_rim, 6.0, "D6")
    d8_overlay = create_hole_overlay(colour_img, d8_seed, d8_rim, 8.0, "D8")

    # Stack side by side
    combined = np.hstack([d6_overlay, d8_overlay])
    cv2.imwrite(str(output_dir / f'{test_capture}_diameter_overlay.png'), combined)
    print(f"Overlay saved to: {output_dir / f'{test_capture}_diameter_overlay.png'}")

    # Test all captures
    print("\nTesting all captures...")
    for capture_id in sorted(seeds.keys()):
        capture_dir = data_dir / capture_id
        if not capture_dir.exists():
            continue

        with open(capture_dir / 'meta.json') as f:
            meta = json.load(f)

        colour_img = compute_median_colour(capture_dir / 'color')
        fx = meta['color_intrinsics']['fx']
        standoff = meta.get('standoff_mm_from_depth', 320.0)

        capture_seeds = seeds[capture_id]
        d6_seed = [s for s in capture_seeds if s.hole_type == 'd6'][0]
        d8_seed = [s for s in capture_seeds if s.hole_type == 'd8'][0]

        try:
            d6_hole, d6_rim = detect_hole_from_seed(colour_img, d6_seed, standoff, fx, config)
            d8_hole, d8_rim = detect_hole_from_seed(colour_img, d8_seed, standoff, fx, config)

            d6_err = abs(d6_rim.free_fit_diameter_mm - 6.0) / 6.0 * 100
            d8_err = abs(d8_rim.free_fit_diameter_mm - 8.0) / 8.0 * 100
            d6_errors.append((capture_id, d6_rim.free_fit_diameter_mm, d6_err))
            d8_errors.append((capture_id, d8_rim.free_fit_diameter_mm, d8_err))

            if d6_err > 10 or d8_err > 10:
                print(f'{capture_id}: D6={d6_rim.free_fit_diameter_mm:.2f}mm ({d6_err:.1f}%), D8={d8_rim.free_fit_diameter_mm:.2f}mm ({d8_err:.1f}%) FAIL')
        except Exception as e:
            failures.append((capture_id, str(e)))

    print(f"\n{'='*60}")
    print("DIAMETER FIX RESULTS")
    print(f"{'='*60}")

    d6_vals = [e[2] for e in d6_errors]
    d8_vals = [e[2] for e in d8_errors]

    print(f"D6 errors: mean={sum(d6_vals)/len(d6_vals):.1f}%, max={max(d6_vals):.1f}%")
    print(f"D8 errors: mean={sum(d8_vals)/len(d8_vals):.1f}%, max={max(d8_vals):.1f}%")
    print(f"All within 10%: {all(e <= 10 for e in d6_vals + d8_vals)}")

    if failures:
        print(f"\nFailures: {failures}")

    # Show worst cases
    d6_sorted = sorted(d6_errors, key=lambda x: -x[2])[:5]
    d8_sorted = sorted(d8_errors, key=lambda x: -x[2])[:5]

    print("\nWorst D6:")
    for cap, diam, err in d6_sorted:
        print(f"  {cap}: {diam:.2f}mm ({err:.1f}%)")

    print("\nWorst D8:")
    for cap, diam, err in d8_sorted:
        print(f"  {cap}: {diam:.2f}mm ({err:.1f}%)")


if __name__ == "__main__":
    main()
