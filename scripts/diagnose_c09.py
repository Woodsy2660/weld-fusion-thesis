#!/usr/bin/env python3
"""
Diagnose C09_top_r0 D8 z=728mm issue.

Show the annulus pixels and depth values being used for plane fit.
"""

import json
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.geometry.registration_seeded import (
    load_seeds, load_config, compute_median_colour, compute_averaged_depth,
    detect_hole_from_seed,
)
from src.geometry.registration_v2 import (
    build_rs_intrinsics, build_rs_extrinsics,
    FallbackExtrinsics,
    deproject_pixel_to_point, project_point_to_pixel, transform_point_to_point,
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
    print(f"Depth image shape: {depth_mm.shape}")
    print()

    # Get D8 seed
    capture_seeds = seeds[capture_id]
    d8_seed = [s for s in capture_seeds if s.hole_type == 'd8'][0]

    print(f"D8 seed (color pixels): ({d8_seed.px_u}, {d8_seed.px_v})")

    # Detect hole
    d8_hole, d8_rim = detect_hole_from_seed(colour_img, d8_seed, standoff, fx, config)

    print(f"D8 fixed-fit centre (color): ({d8_rim.fixed_fit_centre[0]:.1f}, {d8_rim.fixed_fit_centre[1]:.1f})")

    # Now trace through the deprojection manually
    dep_cfg = config["deprojection"]
    # d8 maps to "axis" in config
    hole_cfg = config["datum_holes"]["axis"]
    r_mm = hole_cfg["diameter_mm"] / 2  # Config has diameter, need radius
    inner_mm = dep_cfg["annulus_inner_mm"]
    outer_mm = dep_cfg["annulus_outer_mm"]

    # Color pixel coords
    u_color, v_color = d8_hole.fixed_fit_centre_px[1], d8_hole.fixed_fit_centre_px[0]
    print(f"\nColor pixel (u, v): ({u_color:.1f}, {v_color:.1f})")

    # Initial guess
    z_guess = standoff / 1000.0
    print(f"Initial z_guess: {z_guess*1000:.1f}mm")

    # Deproject to 3D
    point_colour = deproject_pixel_to_point(colour_intr, [u_color, v_color], z_guess)
    print(f"3D point (color frame): {[f'{x*1000:.1f}' for x in point_colour]}mm")

    # Transform to depth frame
    point_depth = transform_point_to_point(colour_to_depth, point_colour)
    print(f"3D point (depth frame): {[f'{x*1000:.1f}' for x in point_depth]}mm")

    # Project to depth image
    pixel_depth = project_point_to_pixel(depth_intr, point_depth)
    ud, vd = int(round(pixel_depth[0])), int(round(pixel_depth[1]))
    print(f"Depth pixel (u, v): ({ud}, {vd})")

    # Check depth at this location
    h, w = depth_mm.shape
    roi_r = int(50 * depth_intr.fx / standoff)
    print(f"ROI radius: {roi_r}px")

    vd0, vd1 = max(0, vd - roi_r), min(h, vd + roi_r)
    ud0, ud1 = max(0, ud - roi_r), min(w, ud + roi_r)

    roi_depth = depth_mm[vd0:vd1, ud0:ud1]
    valid_depth = roi_depth[~np.isnan(roi_depth)]

    print(f"\nDepth ROI stats:")
    print(f"  Valid pixels: {len(valid_depth)}")
    print(f"  Min: {np.min(valid_depth):.1f}mm")
    print(f"  Max: {np.max(valid_depth):.1f}mm")
    print(f"  Median: {np.median(valid_depth):.1f}mm")
    print(f"  Mean: {np.mean(valid_depth):.1f}mm")
    print(f"  Std: {np.std(valid_depth):.1f}mm")

    # Refined z
    z_guess_refined = float(np.median(valid_depth)) / 1000.0
    print(f"\nRefined z_guess: {z_guess_refined*1000:.1f}mm")

    # Re-transform
    point_colour = deproject_pixel_to_point(colour_intr, [u_color, v_color], z_guess_refined)
    point_depth = transform_point_to_point(colour_to_depth, point_colour)
    pixel_depth = project_point_to_pixel(depth_intr, point_depth)
    ud, vd = int(round(pixel_depth[0])), int(round(pixel_depth[1]))
    print(f"Refined depth pixel (u, v): ({ud}, {vd})")

    # Get annulus points
    inner_px = (r_mm + inner_mm) * depth_intr.fx / (z_guess_refined * 1000)
    outer_px = (r_mm + outer_mm) * depth_intr.fx / (z_guess_refined * 1000)
    print(f"\nAnnulus inner/outer radii: {inner_px:.1f} / {outer_px:.1f} px")

    annulus_depths = []
    annulus_points = []

    for dy in range(-int(outer_px) - 5, int(outer_px) + 6):
        for dx in range(-int(outer_px) - 5, int(outer_px) + 6):
            vv, uu = vd + dy, ud + dx
            if not (0 <= vv < h and 0 <= uu < w):
                continue

            dist = np.sqrt(dx**2 + dy**2)
            if inner_px <= dist <= outer_px:
                z = depth_mm[vv, uu]
                if not np.isnan(z):
                    annulus_depths.append(z)
                    pt = deproject_pixel_to_point(depth_intr, [uu, vv], z / 1000.0)
                    annulus_points.append(pt)

    annulus_depths = np.array(annulus_depths)
    annulus_points = np.array(annulus_points)

    print(f"\nAnnulus depth stats:")
    print(f"  N points: {len(annulus_depths)}")
    print(f"  Min: {np.min(annulus_depths):.1f}mm")
    print(f"  Max: {np.max(annulus_depths):.1f}mm")
    print(f"  Median: {np.median(annulus_depths):.1f}mm")
    print(f"  Mean: {np.mean(annulus_depths):.1f}mm")
    print(f"  Std: {np.std(annulus_depths):.1f}mm")

    # Check for outliers
    median = np.median(annulus_depths)
    outliers = np.abs(annulus_depths - median) > 50  # >50mm from median
    print(f"  Outliers (>50mm from median): {np.sum(outliers)}")

    # Fit plane with SVD (current method)
    centroid = annulus_points.mean(axis=0)
    centered = annulus_points - centroid
    _, s, Vt = np.linalg.svd(centered)
    normal_depth = Vt[2]

    if normal_depth[2] < 0:
        normal_depth = -normal_depth

    print(f"\nSVD plane fit:")
    print(f"  Centroid (depth frame): {[f'{x*1000:.1f}' for x in centroid]}mm")
    print(f"  Normal: {normal_depth}")
    print(f"  Singular values: {s}")

    # Transform to color frame
    normal_colour = R @ normal_depth
    centroid_colour = R @ centroid + t

    print(f"  Centroid (color frame): {[f'{x*1000:.1f}' for x in centroid_colour]}mm")

    # Ray intersection
    ray_dir = np.array(deproject_pixel_to_point(colour_intr, [u_color, v_color], 1.0))
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    denom = np.dot(ray_dir, normal_colour)
    t_intersect = np.dot(centroid_colour, normal_colour) / denom
    point_3d = t_intersect * ray_dir

    print(f"\nRay intersection:")
    print(f"  Ray direction: {ray_dir}")
    print(f"  Denom (ray·normal): {denom:.6f}")
    print(f"  t_intersect: {t_intersect:.4f}")
    print(f"  Final 3D point: {[f'{x*1000:.1f}' for x in point_3d]}mm")
    print(f"  Final Z: {point_3d[2]*1000:.1f}mm (expect ~{standoff:.1f}mm)")

    # Save visualization
    output_dir = Path("results/geometry/c09_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create depth visualization around D8
    depth_vis = depth_mm.copy()
    depth_vis[np.isnan(depth_vis)] = 0
    depth_norm = ((depth_vis - 280) / 100 * 255).clip(0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # Draw annulus region
    cv2.circle(depth_color, (ud, vd), int(inner_px), (255, 255, 255), 1)
    cv2.circle(depth_color, (ud, vd), int(outer_px), (255, 255, 255), 1)
    cv2.circle(depth_color, (ud, vd), 3, (0, 255, 0), -1)

    cv2.imwrite(str(output_dir / "c09_d8_depth_annulus.png"), depth_color)
    print(f"\nSaved: {output_dir / 'c09_d8_depth_annulus.png'}")


if __name__ == "__main__":
    main()
