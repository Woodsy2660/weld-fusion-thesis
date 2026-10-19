#!/usr/bin/env python3
"""
Test the existing hole detector to measure the baseline it produces.

Expected result (from B2 bias analysis):
- Baseline should be ~265.8 mm instead of 270 mm nominal
- This 1.6% error is due to fitting the see-through opening centroid,
  not the top rim.

Run: ~/weldenv/bin/python scripts/test_existing_detector.py
"""
import sys
import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.geometry.registration import detect_holes_in_depth, solve_pose, load_config


def load_capture_data(capture_dir: Path):
    """Load depth frames and metadata from a capture directory."""
    meta_path = capture_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    depth_dir = capture_dir / "depth"
    frames = []
    for npy_file in sorted(depth_dir.glob("*.npy")):
        raw = np.load(npy_file).astype(np.float32)
        depth_units = meta.get("depth_units", 0.0001)
        depth_mm = raw * depth_units * 1000.0
        depth_mm[raw == 0] = np.nan
        frames.append(depth_mm)

    return frames, meta


def compute_averaged_depth(frames):
    """Compute per-pixel mean depth."""
    stack = np.stack(frames, axis=0)
    with np.errstate(all="ignore"):
        mean_mm = np.nanmean(stack, axis=0)
    return mean_mm


def test_detector_on_capture(capture_dir: Path):
    """Test the existing detector on one capture."""
    print(f"\n{'='*60}")
    print(f"Testing: {capture_dir.name}")
    print(f"{'='*60}")

    frames, meta = load_capture_data(capture_dir)
    print(f"Loaded {len(frames)} depth frames")
    print(f"Standoff from meta: {meta.get('standoff_mm_from_depth', 'N/A')} mm")
    print(f"Face: {meta.get('face', 'N/A')}")

    depth_mm = compute_averaged_depth(frames)
    print(f"Depth shape: {depth_mm.shape}")

    config = load_config()

    # Run existing detector
    try:
        holes = detect_holes_in_depth(depth_mm, config)
        print(f"\nDetected {len(holes)} holes:")
        for h in holes:
            print(f"  - Centre (row, col): ({h.centre_px[0]:.1f}, {h.centre_px[1]:.1f})")
            print(f"    Radius: {h.radius_px:.1f} px")
            print(f"    Estimated diameter: {h.estimated_diameter_mm:.2f} mm")
            print(f"    Confidence: {h.confidence:.3f}")

        # Filter for the two most likely datum holes
        # We expect: one Ø6 and one Ø8, well separated
        d6_holes = [h for h in holes if h.estimated_diameter_mm == 6]
        d8_holes = [h for h in holes if h.estimated_diameter_mm == 8]

        print(f"\nFiltered: {len(d6_holes)} Ø6 candidates, {len(d8_holes)} Ø8 candidates")

        if d6_holes and d8_holes:
            # Find best pair by confidence
            best_d6 = max(d6_holes, key=lambda h: h.confidence)
            best_d8 = max(d8_holes, key=lambda h: h.confidence)

            print(f"\nBest Ø6: conf={best_d6.confidence:.3f}, pos=({best_d6.centre_px[0]:.1f}, {best_d6.centre_px[1]:.1f})")
            print(f"Best Ø8: conf={best_d8.confidence:.3f}, pos=({best_d8.centre_px[0]:.1f}, {best_d8.centre_px[1]:.1f})")

            # Compute pixel distance
            dr = best_d8.centre_px[0] - best_d6.centre_px[0]
            dc = best_d8.centre_px[1] - best_d6.centre_px[1]
            px_dist = math.sqrt(dr**2 + dc**2)

            # Use actual standoff from meta for scale
            standoff = meta.get("standoff_mm_from_depth", 320.0)
            depth_w = meta["depth_intrinsics"]["width"]
            fov_rad = math.radians(87)
            scale = 2 * standoff * math.tan(fov_rad / 2) / depth_w

            baseline = px_dist * scale

            print(f"\nBaseline calculation:")
            print(f"  Standoff: {standoff:.1f} mm (from meta.json)")
            print(f"  Scale: {scale:.4f} mm/px")
            print(f"  Pixel distance: {px_dist:.1f} px")
            print(f"  **MEASURED BASELINE: {baseline:.2f} mm**")
            print(f"  Nominal: 270 mm")
            print(f"  Error: {baseline - 270:.2f} mm ({(baseline - 270)/270*100:.2f}%)")

            return baseline
        else:
            print("ERROR: Could not find one Ø6 and one Ø8 hole")
            return None

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    data_dir = REPO_ROOT / "data" / "raw" / "S1"

    # Test on C11 first (the reference capture)
    c11_captures = sorted(data_dir.glob("C11_*_r0"))

    baselines = []
    for capture in c11_captures:
        baseline = test_detector_on_capture(capture)
        if baseline is not None:
            baselines.append((capture.name, baseline))

    if baselines:
        print(f"\n{'='*60}")
        print("SUMMARY - Existing Detector Baselines")
        print(f"{'='*60}")
        print(f"Nominal baseline: 270 mm (Ø6 at s=15, Ø8 at s=285)")
        print(f"Expected if bias hypothesis correct: ~265.8 mm")
        print()
        for name, b in baselines:
            delta = b - 270.0
            print(f"  {name}: {b:.2f} mm (delta: {delta:+.2f} mm)")

        mean_baseline = np.mean([b for _, b in baselines])
        print(f"\nMean baseline: {mean_baseline:.2f} mm")
        print(f"Mean error: {mean_baseline - 270:.2f} mm ({(mean_baseline - 270)/270*100:.2f}%)")


if __name__ == "__main__":
    main()
