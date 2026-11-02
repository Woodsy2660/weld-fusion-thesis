#!/usr/bin/env python3
"""
Crop vision patches from real coupon color images.

Uses the SAME cell boundaries as geometry pipeline (seeded registration)
to ensure vision and geometry agree on cell locations.

Output: data/processed/vision_coupon_patches/
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.geometry.registration_seeded import load_seeds, register_capture_seeded


REPO_ROOT = Path(__file__).parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "S1"
SEEDS_PATH = REPO_ROOT / "configs" / "hole_seeds.csv"
WELD_AXIS_SEEDS_PATH = REPO_ROOT / "configs" / "weld_axis_seeds.csv"
LABELS_PATH = REPO_ROOT / "data" / "labels_v1_2026-10-17.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "vision_coupon_patches"
DIAG_DIR = REPO_ROOT / "results" / "vision" / "_diagnostics"

# Crop parameters
BAND_WIDTH_MM = 30.0  # Fixed band across weld (±15mm from centreline)
MODEL_INPUT_SIZE = 224  # ResNet input size


def load_weld_axis_seeds(path: Path) -> dict:
    """Load weld axis seeds (two points defining centreline)."""
    seeds = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            capture_id = row["capture_id"]
            if capture_id not in seeds:
                seeds[capture_id] = []
            seeds[capture_id].append((int(row["px_u"]), int(row["px_v"])))
    return seeds


def compute_centreline(axis_seeds: list, width: int) -> np.ndarray:
    """Compute centreline row for each column from two axis seed points."""
    if len(axis_seeds) < 2:
        return None
    (u1, v1), (u2, v2) = axis_seeds[:2]
    cols = np.arange(width)
    return v1 + (v2 - v1) * (cols - u1) / (u2 - u1)


def letterbox_resize(img: np.ndarray, target_size: int) -> tuple:
    """Resize preserving aspect ratio, pad to square with edge replication.

    Returns (resized_img, pad_fraction).
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Edge replication padding (same as training)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_REPLICATE)

    pad_fraction = (pad_h + pad_w) / (2 * target_size)
    return padded, pad_fraction


def crop_cell(color_img: np.ndarray,
              col_start: int, col_end: int,
              centre_row: float, band_half_px: int) -> np.ndarray:
    """Crop a cell region from the color image."""
    h, w = color_img.shape[:2]

    # Ensure bounds are valid
    col_start = max(0, min(col_start, w - 1))
    col_end = max(0, min(col_end, w))
    row_start = max(0, int(centre_row - band_half_px))
    row_end = min(h, int(centre_row + band_half_px))

    if col_start >= col_end or row_start >= row_end:
        return None

    return color_img[row_start:row_end, col_start:col_end]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROP COUPON PATCHES FOR VISION")
    print("=" * 70)

    # Load seeds
    hole_seeds = load_seeds(SEEDS_PATH)
    axis_seeds = load_weld_axis_seeds(WELD_AXIS_SEEDS_PATH)
    print(f"Loaded hole seeds for {len(hole_seeds)} captures")
    print(f"Loaded axis seeds for {len(axis_seeds)} captures")

    # Load labels to know which cells we need
    labels = {}
    with open(LABELS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[(row["coupon_id"], int(row["cell_index"]))] = row
    print(f"Loaded {len(labels)} cell labels")

    # Find captures (use top face only for vision - better lighting)
    captures = {}
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.split("_")
        if len(parts) >= 3 and parts[1] == "top" and parts[2] == "r0":
            coupon = parts[0]
            if coupon.startswith("C") and coupon != "REF":
                captures[coupon] = d

    print(f"\nFound {len(captures)} coupon captures (top face)")

    # Process each capture
    all_patches = []
    missing = []

    for coupon_id, capture_dir in sorted(captures.items()):
        capture_id = capture_dir.name
        print(f"\n{coupon_id}:", end=" ")

        # Get hole seeds
        cap_seeds = hole_seeds.get(capture_id)
        if not cap_seeds:
            print("NO HOLE SEEDS")
            for i in range(10):
                missing.append((coupon_id, i, "no_hole_seeds"))
            continue

        # Register to get cell positions
        try:
            pose, cloud, meta = register_capture_seeded(capture_dir, cap_seeds)
        except Exception as e:
            print(f"REGISTRATION FAILED: {e}")
            for i in range(10):
                missing.append((coupon_id, i, f"registration_failed: {e}"))
            continue

        # Get hole positions in COLOR image coordinates
        origin_px = pose.origin_hole.centre_px  # (row, col)
        axis_px = pose.axis_hole.centre_px

        # Compute scale and direction (in color image space)
        baseline_px = np.sqrt(
            (axis_px[0] - origin_px[0])**2 + (axis_px[1] - origin_px[1])**2
        )
        baseline_mm = pose.measured_baseline_mm if pose.measured_baseline_mm > 0 else 270.0
        scale_px_per_mm = baseline_px / baseline_mm

        # Direction vector from D6 to D8
        dx = axis_px[1] - origin_px[1]  # col direction
        dy = axis_px[0] - origin_px[0]  # row direction
        length = np.sqrt(dx**2 + dy**2)
        unit_s = np.array([dx / length, dy / length])  # (col, row) direction

        # Band width in pixels
        band_half_px = int(BAND_WIDTH_MM / 2 * scale_px_per_mm)

        # Load median color image
        color_dir = capture_dir / "color"
        color_files = sorted(color_dir.glob("*.png"))
        if not color_files:
            print("NO COLOR IMAGES")
            for i in range(10):
                missing.append((coupon_id, i, "no_color_images"))
            continue

        # Use median of color frames for noise reduction
        frames = [cv2.imread(str(f)) for f in color_files]
        color_img = np.median(frames, axis=0).astype(np.uint8)
        h, w = color_img.shape[:2]

        # Get centreline from axis seeds or interpolate from holes
        cap_axis = axis_seeds.get(capture_id)
        if cap_axis and len(cap_axis) >= 2:
            centreline = compute_centreline(cap_axis, w)
        else:
            # Interpolate centreline from hole positions
            centreline = np.full(w, (origin_px[0] + axis_px[0]) / 2)

        # Cell parameters (same as geometry)
        cell_width_mm = 25.0
        first_excluded_mm = 25.0
        coupon_length_mm = 300.0
        last_excluded_mm = 25.0
        origin_s_mm = 15.0  # D6 is at s=15mm

        s_start = first_excluded_mm
        n_cells = 10

        cropped = 0
        for i in range(n_cells):
            s_cell_start = s_start + i * cell_width_mm
            s_cell_end = s_cell_start + cell_width_mm
            s_centre = (s_cell_start + s_cell_end) / 2

            # Convert to pixel positions
            offset_start = (s_cell_start - origin_s_mm) * scale_px_per_mm
            offset_end = (s_cell_end - origin_s_mm) * scale_px_per_mm
            offset_centre = (s_centre - origin_s_mm) * scale_px_per_mm

            col_start = int(origin_px[1] + offset_start * unit_s[0])
            col_end = int(origin_px[1] + offset_end * unit_s[0])
            col_centre = int(origin_px[1] + offset_centre * unit_s[0])

            # Get centre row at this position
            if 0 <= col_centre < w:
                centre_row = centreline[col_centre]
            else:
                centre_row = (origin_px[0] + axis_px[0]) / 2

            # Crop
            crop = crop_cell(color_img, min(col_start, col_end), max(col_start, col_end),
                            centre_row, band_half_px)

            if crop is None or crop.size == 0:
                missing.append((coupon_id, i, "crop_failed"))
                continue

            # Letterbox resize
            resized, pad_frac = letterbox_resize(crop, MODEL_INPUT_SIZE)

            # Save
            cell_id = f"{coupon_id}-{i:02d}"
            out_path = OUTPUT_DIR / f"{cell_id}.png"
            cv2.imwrite(str(out_path), resized)

            # Get label info
            label_info = labels.get((coupon_id, i), {})

            all_patches.append({
                "cell_id": cell_id,
                "coupon_id": coupon_id,
                "cell_index": i,
                "capture_id": capture_id,
                "patch_path": str(out_path.relative_to(REPO_ROOT)),
                "crop_col_start": min(col_start, col_end),
                "crop_col_end": max(col_start, col_end),
                "crop_row_start": int(centre_row - band_half_px),
                "crop_row_end": int(centre_row + band_half_px),
                "pad_fraction": pad_frac,
                "true_class": label_info.get("true_class", ""),
                "vision_target": label_info.get("vision_target", ""),
                "fold": label_info.get("fold", ""),
            })
            cropped += 1

        print(f"{cropped} patches")

    # Write manifest
    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        if all_patches:
            writer = csv.DictWriter(f, fieldnames=all_patches[0].keys())
            writer.writeheader()
            writer.writerows(all_patches)

    print(f"\n\nWrote {len(all_patches)} patches to {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")

    # Report missing
    if missing:
        print(f"\nMISSING PATCHES: {len(missing)}")
        for coupon, cell, reason in missing[:10]:
            print(f"  {coupon}-{cell:02d}: {reason}")

    # Generate contact sheet (20 random crops)
    if all_patches:
        np.random.seed(42)
        sample_idx = np.random.choice(len(all_patches), min(20, len(all_patches)), replace=False)
        samples = [all_patches[i] for i in sample_idx]

        # Create 4x5 grid
        grid_h, grid_w = 5, 4
        cell_size = MODEL_INPUT_SIZE
        contact = np.zeros((grid_h * cell_size, grid_w * cell_size, 3), dtype=np.uint8)

        for idx, patch in enumerate(samples):
            row, col = idx // grid_w, idx % grid_w
            img = cv2.imread(str(REPO_ROOT / patch["patch_path"]))
            if img is not None:
                contact[row*cell_size:(row+1)*cell_size,
                       col*cell_size:(col+1)*cell_size] = img
                # Add label
                label = f"{patch['cell_id']} {patch['true_class']}"
                cv2.putText(contact, label,
                           (col*cell_size + 5, row*cell_size + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        contact_path = DIAG_DIR / "coupon_patches_contact.png"
        cv2.imwrite(str(contact_path), contact)
        print(f"\nContact sheet: {contact_path}")

    # Summary by coupon
    print("\n" + "=" * 70)
    print("SUMMARY BY COUPON")
    print("=" * 70)
    coupon_counts = {}
    for p in all_patches:
        c = p["coupon_id"]
        coupon_counts[c] = coupon_counts.get(c, 0) + 1
    for c in sorted(coupon_counts.keys()):
        print(f"  {c}: {coupon_counts[c]} patches")


if __name__ == "__main__":
    main()
