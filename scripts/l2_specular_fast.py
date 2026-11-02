#!/usr/bin/env python3
"""
L2 diagnostic (FAST): Single-frame analysis to avoid slow multi-frame loading.
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

DATA_DIR = Path("data/raw/S1")


def load_single_depth(capture_dir: Path) -> tuple:
    """Load just the first depth frame and meta."""
    with open(capture_dir / "meta.json") as f:
        meta = json.load(f)

    depth_units = meta.get("depth_units", 0.0001)

    # Load only first frame
    first_npy = sorted((capture_dir / "depth").glob("*.npy"))[0]
    raw = np.load(first_npy).astype(np.float32)
    depth_mm = raw * depth_units * 1000.0
    depth_mm[raw == 0] = np.nan

    return depth_mm, meta


def load_single_color(capture_dir: Path) -> np.ndarray:
    """Load just the first color frame."""
    import cv2
    first_png = sorted((capture_dir / "color").glob("*.png"))[0]
    return cv2.imread(str(first_png))


def main():
    print("=" * 70)
    print("L2 DIAGNOSTIC (FAST SINGLE-FRAME)")
    print("=" * 70)

    captures = sorted([d.name for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("C") and "_r0" in d.name])
    print(f"\nAnalyzing {len(captures)} primary captures (r0 only)...")

    results_a = []  # Invalid pixel fractions
    results_b = []  # Signed reinforcement
    all_brightness = []
    all_dip = []

    for cap_id in captures:
        cap_dir = DATA_DIR / cap_id
        try:
            depth_mm, meta = load_single_depth(cap_dir)
            H, W = depth_mm.shape

            # Scale
            fx = meta.get("depth_intrinsics", {}).get("fx", 638)
            standoff = meta.get("standoff_mm_from_depth", 320)
            scale = standoff / fx

            centre_row = H // 2
            bead_half_px = int(12 / scale)
            plate_inner_px = int(15 / scale)
            plate_outer_px = int(40 / scale)

            # Create masks
            row_idx = np.arange(H)
            dist = np.abs(row_idx - centre_row)

            bead_rows = dist <= bead_half_px
            plate_rows = (dist >= plate_inner_px) & (dist <= plate_outer_px)

            # Test A: Invalid pixels
            invalid = np.isnan(depth_mm)
            bead_invalid = np.sum(invalid[bead_rows, :]) / (np.sum(bead_rows) * W)
            plate_invalid = np.sum(invalid[plate_rows, :]) / (np.sum(plate_rows) * W)
            ratio = bead_invalid / plate_invalid if plate_invalid > 0 else float("nan")

            results_a.append({
                "capture": cap_id,
                "bead_invalid": bead_invalid,
                "plate_invalid": plate_invalid,
                "ratio": ratio
            })

            # Test B: Signed reinforcement
            bead_depth = depth_mm[bead_rows, :]
            plate_depth = depth_mm[plate_rows, :]

            bead_med = np.nanmedian(bead_depth)
            plate_med = np.nanmedian(plate_depth)
            signed_reinf = plate_med - bead_med  # Positive = proud

            results_b.append({
                "capture": cap_id,
                "signed_reinf_mm": signed_reinf,
                "bead_median_mm": bead_med,
                "plate_median_mm": plate_med
            })

            # Test C: Brightness
            try:
                color = load_single_color(cap_dir)
                gray = np.mean(color, axis=2)
                H_c = gray.shape[0]
                centre_c = H_c // 2
                bead_band_c = int(12 * H_c / H / scale)
                brightness = np.mean(gray[centre_c - bead_band_c:centre_c + bead_band_c, :])
                dip = bead_med - plate_med  # Positive = trench
                all_brightness.append(brightness)
                all_dip.append(dip)
            except:
                pass

            print(f"  {cap_id}: invalid_ratio={ratio:.2f}, reinf={signed_reinf:+.2f}mm")

        except Exception as e:
            print(f"  {cap_id}: ERROR - {e}")

    # Summaries
    print("\n" + "=" * 70)
    print("TEST (a): INVALID-PIXEL FRACTION")
    print("=" * 70)

    ratios = [r["ratio"] for r in results_a if not np.isnan(r["ratio"])]
    bead_fracs = [r["bead_invalid"] for r in results_a]
    plate_fracs = [r["plate_invalid"] for r in results_a]

    print(f"Bead invalid:  mean={np.mean(bead_fracs):.4f}, std={np.std(bead_fracs):.4f}")
    print(f"Plate invalid: mean={np.mean(plate_fracs):.4f}, std={np.std(plate_fracs):.4f}")
    print(f"Ratio:         mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}")

    if np.mean(ratios) > 2.0:
        conclusion_a = "Bead has 2x+ more invalid pixels → specular dropout occurring"
    else:
        conclusion_a = "Invalid rates comparable → specular dropout NOT dominant"
    print(f"\n→ {conclusion_a}")

    print("\n" + "=" * 70)
    print("TEST (b): PROUD BEAD CHECK")
    print("=" * 70)

    proud = [r for r in results_b if r["signed_reinf_mm"] > 0]
    concave = [r for r in results_b if r["signed_reinf_mm"] <= 0]

    print(f"Proud beads (positive reinforcement):  {len(proud)}")
    print(f"Concave beads (negative/trench):       {len(concave)}")

    sorted_b = sorted(results_b, key=lambda r: r["signed_reinf_mm"], reverse=True)

    print("\nMost PROUD (visually convex? → sensor validation):")
    for r in sorted_b[:5]:
        print(f"  {r['capture']}: {r['signed_reinf_mm']:+.2f} mm")

    print("\nMost CONCAVE (under-fill or artefact?):")
    for r in sorted_b[-5:]:
        print(f"  {r['capture']}: {r['signed_reinf_mm']:+.2f} mm")

    if len(proud) > 0:
        conclusion_b = f"{len(proud)} proud captures exist. If visually convex, sensor is correct."
    else:
        conclusion_b = "No proud beads detected. All beads show concavity."
    print(f"\n→ {conclusion_b}")

    print("\n" + "=" * 70)
    print("TEST (c): MAGNITUDE CORRELATION")
    print("=" * 70)

    if len(all_brightness) > 3:
        r, p = stats.pearsonr(all_brightness, all_dip)
        print(f"Pearson r (brightness vs dip): {r:.3f}")
        print(f"P-value: {p:.4f}")
        print(f"N: {len(all_brightness)}")

        if abs(r) > 0.5 and p < 0.05:
            conclusion_c = "Significant correlation → specular artefact likely"
        elif abs(r) < 0.3:
            conclusion_c = "Weak/no correlation → dips are likely real under-fill"
        else:
            conclusion_c = "Moderate correlation → inconclusive"
    else:
        conclusion_c = "Insufficient data"
        r, p = float("nan"), float("nan")
    print(f"\n→ {conclusion_c}")

    # Final summary
    print("\n" + "=" * 70)
    print("L2 CONCLUSION")
    print("=" * 70)
    print(f"\n(a) Invalid pixels: {conclusion_a}")
    print(f"(b) Proud beads:    {conclusion_b}")
    print(f"(c) Correlation:    {conclusion_c}")

    # Save
    output = {
        "test_a": {"results": results_a, "conclusion": conclusion_a},
        "test_b": {"results": results_b, "conclusion": conclusion_b},
        "test_c": {"r": r, "p": p, "n": len(all_brightness), "conclusion": conclusion_c}
    }

    out_dir = Path("results/geometry/_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "l2_specular_fast.json", "w") as f:
        json.dump(output, f, indent=2, default=float)

    print(f"\nSaved: {out_dir / 'l2_specular_fast.json'}")


if __name__ == "__main__":
    main()
