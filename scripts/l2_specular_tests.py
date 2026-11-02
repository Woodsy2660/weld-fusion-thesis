#!/usr/bin/env python3
"""
L2 diagnostic: Three cheap tests to distinguish real under-fill from specular artefact.

(a) Invalid-pixel fraction: bead band vs parent plate
(b) Proud-bead check: does a visibly convex bead read proud?
(c) Magnitude correlation: does dip correlate with brightness?

Usage:
    python scripts/l2_specular_tests.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.geometry.profile import load_depth_frames, compute_depth_stack, load_config
from src.geometry.registration_seeded import load_seeds, compute_median_colour

DATA_DIR = Path("data/raw/S1")
CONFIG = load_config()


def load_capture_data(capture_dir: Path) -> tuple:
    """Load depth stack and color image for a capture."""
    with open(capture_dir / "meta.json") as f:
        meta = json.load(f)

    depth_units = meta.get("depth_units", 0.0001)
    frames = load_depth_frames(capture_dir / "depth", depth_units)
    depth_stack = compute_depth_stack(frames, CONFIG)

    color_img = compute_median_colour(capture_dir / "color")

    return depth_stack, color_img, meta


def compute_region_masks(depth_mm: np.ndarray, meta: dict) -> tuple:
    """
    Create masks for bead band and parent plate regions.

    Returns:
        (bead_mask, plate_mask) - boolean arrays
    """
    H, W = depth_mm.shape

    # Approximate: bead is central band, plate is outer regions
    # Bead: ±12mm from centre (config: bead_roi.y_half_width_mm)
    # Plate: 15-40mm from centre (config: plane_fit_roi)

    # Use image centre as weld centreline approximation
    centre_row = H // 2

    # Scale: approximate from typical standoff and fx
    fx = meta.get("depth_intrinsics", {}).get("fx", 638)
    standoff = meta.get("standoff_mm_from_depth", 320)
    scale_mm_per_px = standoff / fx

    bead_half_px = int(12 / scale_mm_per_px)
    plate_inner_px = int(15 / scale_mm_per_px)
    plate_outer_px = int(40 / scale_mm_per_px)

    row_indices = np.arange(H)
    dist_from_centre = np.abs(row_indices - centre_row)

    bead_mask_1d = dist_from_centre <= bead_half_px
    plate_mask_1d = (dist_from_centre >= plate_inner_px) & (dist_from_centre <= plate_outer_px)

    bead_mask = np.tile(bead_mask_1d[:, np.newaxis], (1, W))
    plate_mask = np.tile(plate_mask_1d[:, np.newaxis], (1, W))

    return bead_mask, plate_mask


def test_a_invalid_pixel_fraction(captures: list) -> dict:
    """
    Test (a): Invalid-pixel fraction over bead band vs parent plate.

    Specular dropout produces invalid pixels that get filled by interpolation.
    A biased-but-valid reading is a measurement, not an artefact.
    """
    results = []

    for cap_id in captures:
        cap_dir = DATA_DIR / cap_id
        if not cap_dir.exists():
            continue

        try:
            depth_stack, _, meta = load_capture_data(cap_dir)
            depth_mm = depth_stack.mean_mm

            bead_mask, plate_mask = compute_region_masks(depth_mm, meta)

            # Invalid = NaN in the averaged depth
            invalid = np.isnan(depth_mm)

            bead_invalid_frac = np.sum(invalid & bead_mask) / np.sum(bead_mask)
            plate_invalid_frac = np.sum(invalid & plate_mask) / np.sum(plate_mask)

            results.append({
                "capture_id": cap_id,
                "bead_invalid_frac": float(bead_invalid_frac),
                "plate_invalid_frac": float(plate_invalid_frac),
                "ratio": float(bead_invalid_frac / plate_invalid_frac) if plate_invalid_frac > 0 else float("nan"),
            })

        except Exception as e:
            print(f"  {cap_id}: ERROR - {e}")

    return {"test": "invalid_pixel_fraction", "results": results}


def test_b_proud_bead(captures: list) -> dict:
    """
    Test (b): Do proud (visibly convex) beads read proud?

    If a physically proud bead reads as positive reinforcement,
    the sensor handles shiny weld metal correctly, and every trench
    elsewhere is real fill deficiency.
    """
    results = []

    for cap_id in captures:
        cap_dir = DATA_DIR / cap_id
        if not cap_dir.exists():
            continue

        try:
            depth_stack, _, meta = load_capture_data(cap_dir)
            depth_mm = depth_stack.mean_mm

            bead_mask, plate_mask = compute_region_masks(depth_mm, meta)

            # Compute signed reinforcement: plate_median - bead_median
            # Positive = bead is closer to camera = proud
            # Negative = bead is further from camera = concave/trench

            plate_depth = depth_mm[plate_mask & ~np.isnan(depth_mm)]
            bead_depth = depth_mm[bead_mask & ~np.isnan(depth_mm)]

            if len(plate_depth) > 0 and len(bead_depth) > 0:
                plate_median = float(np.median(plate_depth))
                bead_median = float(np.median(bead_depth))

                # Signed reinforcement: positive = proud, negative = concave
                signed_reinforcement_mm = plate_median - bead_median

                results.append({
                    "capture_id": cap_id,
                    "plate_median_mm": plate_median,
                    "bead_median_mm": bead_median,
                    "signed_reinforcement_mm": signed_reinforcement_mm,
                    "interpretation": "proud" if signed_reinforcement_mm > 0 else "concave",
                })

        except Exception as e:
            print(f"  {cap_id}: ERROR - {e}")

    return {"test": "proud_bead_check", "results": results}


def test_c_magnitude_correlation(captures: list) -> dict:
    """
    Test (c): Does dip magnitude correlate with bead brightness?

    If specular reflections cause false trenches, brighter regions
    should show deeper apparent concavities.
    """
    results = []
    all_brightness = []
    all_dip = []

    for cap_id in captures:
        cap_dir = DATA_DIR / cap_id
        if not cap_dir.exists():
            continue

        try:
            depth_stack, color_img, meta = load_capture_data(cap_dir)
            depth_mm = depth_stack.mean_mm

            bead_mask, plate_mask = compute_region_masks(depth_mm, meta)

            # Compute dip magnitude per cell (simplified: global for now)
            plate_depth = depth_mm[plate_mask & ~np.isnan(depth_mm)]
            bead_depth = depth_mm[bead_mask & ~np.isnan(depth_mm)]

            if len(plate_depth) > 0 and len(bead_depth) > 0:
                plate_median = float(np.median(plate_depth))
                bead_median = float(np.median(bead_depth))
                dip_mm = bead_median - plate_median  # Positive = bead is deeper = trench

                # Bead brightness (grayscale mean)
                gray = np.mean(color_img, axis=2) if color_img.ndim == 3 else color_img
                H_c, W_c = gray.shape[:2]
                H_d, W_d = depth_mm.shape

                # Resize bead mask to color image size
                scale_y = H_c / H_d
                scale_x = W_c / W_d

                # Simple: use central band of color image
                centre_row_c = H_c // 2
                bead_half_c = int(12 * H_c / (meta.get("standoff_mm_from_depth", 320) /
                                              meta.get("depth_intrinsics", {}).get("fx", 638) * H_d))
                bead_band_c = gray[centre_row_c - bead_half_c:centre_row_c + bead_half_c, :]
                bead_brightness = float(np.mean(bead_band_c))

                results.append({
                    "capture_id": cap_id,
                    "dip_mm": dip_mm,
                    "bead_brightness": bead_brightness,
                })

                all_brightness.append(bead_brightness)
                all_dip.append(dip_mm)

        except Exception as e:
            print(f"  {cap_id}: ERROR - {e}")

    # Compute correlation
    if len(all_brightness) > 3:
        r, p = stats.pearsonr(all_brightness, all_dip)
        correlation = {"pearson_r": float(r), "p_value": float(p), "n": len(all_brightness)}
    else:
        correlation = {"pearson_r": float("nan"), "p_value": float("nan"), "n": len(all_brightness)}

    return {"test": "magnitude_correlation", "results": results, "correlation": correlation}


def main():
    print("=" * 70)
    print("L2 DIAGNOSTIC: THREE CHEAP TESTS")
    print("=" * 70)

    # Get all captures
    captures = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("C")])
    print(f"\nAnalyzing {len(captures)} captures...")

    # Run tests
    print("\n" + "=" * 70)
    print("TEST (a): INVALID-PIXEL FRACTION")
    print("=" * 70)
    test_a = test_a_invalid_pixel_fraction(captures)

    bead_fracs = [r["bead_invalid_frac"] for r in test_a["results"]]
    plate_fracs = [r["plate_invalid_frac"] for r in test_a["results"]]
    ratios = [r["ratio"] for r in test_a["results"] if not np.isnan(r["ratio"])]

    print(f"\nBead band invalid fraction:   mean={np.mean(bead_fracs):.4f}, std={np.std(bead_fracs):.4f}")
    print(f"Parent plate invalid fraction: mean={np.mean(plate_fracs):.4f}, std={np.std(plate_fracs):.4f}")
    print(f"Ratio (bead/plate):            mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}")

    if np.mean(ratios) > 2.0:
        print("\n  → INTERPRETATION: Bead has 2x+ more invalid pixels than plate.")
        print("    This suggests specular dropout is occurring on the weld bead.")
    else:
        print("\n  → INTERPRETATION: Invalid pixel rates are comparable.")
        print("    Specular dropout is NOT a dominant factor.")

    # Show per-capture details
    print("\nPer-capture breakdown (first 10):")
    for r in test_a["results"][:10]:
        print(f"  {r['capture_id']}: bead={r['bead_invalid_frac']:.4f}, plate={r['plate_invalid_frac']:.4f}, ratio={r['ratio']:.2f}")

    print("\n" + "=" * 70)
    print("TEST (b): PROUD BEAD CHECK")
    print("=" * 70)
    test_b = test_b_proud_bead(captures)

    proud_count = sum(1 for r in test_b["results"] if r["interpretation"] == "proud")
    concave_count = sum(1 for r in test_b["results"] if r["interpretation"] == "concave")

    print(f"\nProud beads (positive reinforcement): {proud_count}")
    print(f"Concave beads (negative, trench):     {concave_count}")

    # Find most proud and most concave
    sorted_by_reinf = sorted(test_b["results"], key=lambda r: r["signed_reinforcement_mm"], reverse=True)

    print("\nMost PROUD beads (if sensor works, these should be visually convex):")
    for r in sorted_by_reinf[:5]:
        print(f"  {r['capture_id']}: {r['signed_reinforcement_mm']:+.2f} mm")

    print("\nMost CONCAVE beads (potential under-fill or specular artefact):")
    for r in sorted_by_reinf[-5:]:
        print(f"  {r['capture_id']}: {r['signed_reinforcement_mm']:+.2f} mm")

    if proud_count > 0:
        print(f"\n  → KEY CHECK: Visually inspect the {proud_count} 'proud' captures.")
        print("    If they have stacked-dime convex caps and read proud, the sensor")
        print("    handles shiny weld metal correctly and trenches elsewhere are real.")

    print("\n" + "=" * 70)
    print("TEST (c): MAGNITUDE CORRELATION")
    print("=" * 70)
    test_c = test_c_magnitude_correlation(captures)

    corr = test_c["correlation"]
    print(f"\nPearson r between bead brightness and dip magnitude: {corr['pearson_r']:.3f}")
    print(f"P-value: {corr['p_value']:.4f}")
    print(f"N samples: {corr['n']}")

    if abs(corr["pearson_r"]) > 0.5 and corr["p_value"] < 0.05:
        print("\n  → INTERPRETATION: Significant correlation detected.")
        print("    Brighter beads show deeper dips, suggesting specular artefact.")
    elif abs(corr["pearson_r"]) < 0.3:
        print("\n  → INTERPRETATION: Weak or no correlation.")
        print("    Dip magnitude does NOT correlate with brightness.")
        print("    This supports the interpretation that dips are real under-fill.")
    else:
        print("\n  → INTERPRETATION: Moderate correlation, inconclusive.")

    # Save results
    output_dir = Path("results/geometry/_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "test_a_invalid_pixels": test_a,
        "test_b_proud_bead": test_b,
        "test_c_correlation": test_c,
    }

    with open(output_dir / "l2_specular_tests.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'l2_specular_tests.json'}")

    # Summary conclusion
    print("\n" + "=" * 70)
    print("L2 CONCLUSION")
    print("=" * 70)
    print("""
Based on the three tests:

(a) Invalid-pixel fraction: If bead >> plate, specular dropout is occurring.
(b) Proud-bead check: If proud beads read proud, sensor is correct.
(c) Magnitude correlation: If dip correlates with brightness, it's an artefact.

The welds were NOT filled to plate surface (limited fabrication time).
A concavity over the bead band is therefore most likely a real measurement
of INCOMPLETELY FILLED GROOVE (ISO 5817 code 511).
""")


if __name__ == "__main__":
    main()
