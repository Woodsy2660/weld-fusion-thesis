#!/usr/bin/env python3
"""
Extract cell ROI patches from coupon images using weld line selections.

Creates properly cropped cell patches for retraining the vision model
on the actual experimental data rather than public corpus.

Output:
    data/processed/coupon_cell_patches/
        {vision_target}/
            {coupon_id}_{cell_idx}.png
    data/processed/coupon_cell_patches/manifest.csv
"""

import sys
from pathlib import Path
import json
import csv

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_weld_selections():
    """Load weld line selections."""
    with open(REPO_ROOT / "configs" / "weld_selections.json") as f:
        return json.load(f)


def load_labels():
    """Load ground truth labels."""
    labels = {}
    with open(REPO_ROOT / "results" / "thesis" / "data" / "labels_v2_2026-10-26.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['cell_id']] = {
                'coupon_id': row['coupon_id'],
                'cell_index': int(row['cell_index']),
                'zone': row['zone'],
                'condition': row['condition'],
                'true_class': row['true_class'],
                'vision_target': row['vision_target'],
                'fold': int(row['fold']),
            }
    return labels


def load_coupon_images(coupon_id, face='top'):
    """Load all frames for a coupon and average them."""
    coupon_dir = REPO_ROOT / "data" / "raw" / "S1" / f"{coupon_id}_{face}_r0"
    color_dir = coupon_dir / "color"

    if not color_dir.exists():
        return None

    img_files = sorted(color_dir.glob("*.png"))
    if not img_files:
        return None

    # Load middle frame (most stable)
    mid_idx = len(img_files) // 2
    return np.array(Image.open(img_files[mid_idx]))


def get_cell_bounds(weld_line, cell_idx, n_cells=10, band_width=60):
    """
    Get bounding box for a cell along the weld line.

    Returns: (x_min, y_min, x_max, y_max)
    """
    (x1, y1), (x2, y2) = weld_line

    # Direction along weld
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    # Unit vectors
    along = np.array([dx, dy]) / length
    perp = np.array([-dy, dx]) / length

    # Half band width
    hw = band_width / 2

    # Start and end fraction along weld
    t0 = cell_idx / n_cells
    t1 = (cell_idx + 1) / n_cells

    # Points along weld centerline
    p0 = np.array([x1, y1]) + t0 * length * along
    p1 = np.array([x1, y1]) + t1 * length * along

    # Four corners
    corners = np.array([
        p0 - hw * perp,
        p1 - hw * perp,
        p1 + hw * perp,
        p0 + hw * perp,
    ])

    # Bounding box
    x_min = int(corners[:, 0].min())
    x_max = int(corners[:, 0].max())
    y_min = int(corners[:, 1].min())
    y_max = int(corners[:, 1].max())

    return x_min, y_min, x_max, y_max


def extract_cell_patch(img, bounds, target_size=224):
    """Extract and resize a cell patch."""
    x_min, y_min, x_max, y_max = bounds
    h, w = img.shape[:2]

    # Clip to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # Extract patch
    patch = img[y_min:y_max, x_min:x_max]

    if patch.size == 0:
        return None

    # Resize to target size (preserving aspect ratio with padding)
    pil_img = Image.fromarray(patch)

    # Calculate scaling
    pw, ph = pil_img.size
    scale = min(target_size / pw, target_size / ph)
    new_w = int(pw * scale)
    new_h = int(ph * scale)

    # Resize
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # Pad to square
    result = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    result.paste(pil_img, (paste_x, paste_y))

    return np.array(result)


def main():
    print("=" * 60)
    print("EXTRACTING CELL ROI PATCHES")
    print("=" * 60)

    selections = load_weld_selections()
    labels = load_labels()

    # Output directory
    output_dir = REPO_ROOT / "data" / "processed" / "coupon_cell_patches"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirs for each vision target
    for target in ['{P}', '{N,U}', '{N,U,L}']:
        target_dir = output_dir / target.replace('{', '').replace('}', '').replace(',', '_')
        target_dir.mkdir(exist_ok=True)

    manifest = []
    stats = {'total': 0, 'success': 0, 'failed': 0}

    # Get unique coupons
    coupons = sorted(set(l['coupon_id'] for l in labels.values()))

    for coupon_id in coupons:
        print(f"\nProcessing {coupon_id}...")

        if coupon_id not in selections:
            print(f"  WARNING: No weld selection for {coupon_id}")
            continue

        sel = selections[coupon_id]
        weld_line = sel['weld_line']

        # Load image
        img = load_coupon_images(coupon_id)
        if img is None:
            print(f"  WARNING: No image for {coupon_id}")
            continue

        # Extract each cell
        for cell_idx in range(10):
            cell_id = f"{coupon_id}-{cell_idx:02d}"
            label = labels.get(cell_id)

            if not label:
                print(f"  WARNING: No label for {cell_id}")
                continue

            stats['total'] += 1

            # Get cell bounds
            bounds = get_cell_bounds(weld_line, cell_idx, band_width=60)

            # Extract patch
            patch = extract_cell_patch(img, bounds)

            if patch is None:
                print(f"  WARNING: Failed to extract {cell_id}")
                stats['failed'] += 1
                continue

            # Save patch
            vision_target = label['vision_target']
            target_dir_name = vision_target.replace('{', '').replace('}', '').replace(',', '_')
            patch_filename = f"{coupon_id}_{cell_idx:02d}.png"
            patch_path = output_dir / target_dir_name / patch_filename

            Image.fromarray(patch).save(patch_path)
            stats['success'] += 1

            # Add to manifest
            manifest.append({
                'patch_path': f"{target_dir_name}/{patch_filename}",
                'cell_id': cell_id,
                'coupon_id': coupon_id,
                'cell_index': cell_idx,
                'zone': label['zone'],
                'condition': label['condition'],
                'true_class': label['true_class'],
                'vision_target': vision_target,
                'fold': label['fold'],
            })

        print(f"  Extracted {sum(1 for m in manifest if m['coupon_id'] == coupon_id)} cells")

    # Save manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total cells: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nManifest: {manifest_path}")

    # Show distribution
    print("\nDistribution by vision target:")
    from collections import Counter
    target_counts = Counter(m['vision_target'] for m in manifest)
    for target, count in sorted(target_counts.items()):
        print(f"  {target}: {count}")

    print("\nDistribution by fold:")
    fold_counts = Counter(m['fold'] for m in manifest)
    for fold, count in sorted(fold_counts.items()):
        print(f"  Fold {fold}: {count}")


if __name__ == "__main__":
    main()
