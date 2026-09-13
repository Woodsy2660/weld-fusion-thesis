"""Generate contact sheet for discontinuity label review.

The 'discontinuity' label in weld-dataset is provisionally mapped to V3
(surface irregularity). This script generates a contact sheet of ~50 samples
for human review before training.

GATE: If the samples don't look like surface irregularities, adjust the
mapping in configs/class_map.yaml before proceeding to training.

Usage: python -m weldfusion.vision.review_discontinuity
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from weldfusion.utils.config import repo_root


def extract_discontinuity_crops(
    root: Path,
    max_samples: int = 50,
    crop_size: int = 150,
    context_factor: float = 1.5,
) -> list[tuple[np.ndarray, str, str]]:
    """Extract discontinuity crops from weld-dataset.

    Returns list of (crop_image, source_image, label) tuples.
    """
    data_dir = root / "data" / "weld-dataset" / "high_resolution_welds"
    if not data_dir.exists():
        raise FileNotFoundError(f"weld-dataset not found at {data_dir}")

    crops = []

    for json_path in sorted(data_dir.glob("*.json")):
        if len(crops) >= max_samples:
            break

        img_path = json_path.with_suffix(".jpg")
        if not img_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        for shape in data.get("shapes", []):
            if len(crops) >= max_samples:
                break

            if shape.get("shape_type") != "rectangle":
                continue
            if shape.get("label") != "discontinuity":
                continue

            points = shape["points"]
            x1, y1 = points[0]
            x2, y2 = points[1]
            x, y = int(min(x1, x2)), int(min(y1, y2))
            w, h = int(abs(x2 - x1)), int(abs(y2 - y1))

            if w <= 0 or h <= 0:
                continue

            # Expand with context
            cx, cy = x + w / 2, y + h / 2
            new_w = w * context_factor
            new_h = h * context_factor

            x1_crop = int(max(0, cx - new_w / 2))
            y1_crop = int(max(0, cy - new_h / 2))
            x2_crop = int(min(w_img, cx + new_w / 2))
            y2_crop = int(min(h_img, cy + new_h / 2))

            crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
            if crop.size == 0:
                continue

            # Resize to uniform size
            crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

            crops.append((crop, json_path.stem, "discontinuity"))

    return crops


def create_contact_sheet(
    crops: list[tuple[np.ndarray, str, str]],
    output_path: Path,
    cols: int = 10,
    cell_size: int = 150,
    padding: int = 2,
    label_height: int = 20,
) -> None:
    """Create a contact sheet from crops.

    Each cell shows the crop with source image name below.
    """
    n = len(crops)
    if n == 0:
        raise ValueError("No crops to display")

    rows = math.ceil(n / cols)

    cell_h = cell_size + label_height + padding * 2
    cell_w = cell_size + padding * 2

    sheet_h = rows * cell_h
    sheet_w = cols * cell_w

    # White background
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 255

    for idx, (crop, source, label) in enumerate(crops):
        row = idx // cols
        col = idx % cols

        x = col * cell_w + padding
        y = row * cell_h + padding

        # Place crop
        crop_resized = cv2.resize(crop, (cell_size, cell_size))
        sheet[y : y + cell_size, x : x + cell_size] = crop_resized

        # Add label
        text = f"{source[:15]}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        cv2.putText(
            sheet,
            text,
            (x, y + cell_size + 12),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    print(f"Contact sheet saved: {output_path}")
    print(f"  {n} discontinuity samples for review")
    print(f"  Size: {sheet_w}x{sheet_h} pixels")


def main() -> None:
    root = repo_root()

    print("Extracting discontinuity samples from weld-dataset...")
    crops = extract_discontinuity_crops(root, max_samples=50)

    if not crops:
        print("ERROR: No discontinuity samples found!")
        print("Check that data/weld-dataset/high_resolution_welds/ exists and contains LabelMe JSON files.")
        return

    output_path = root / "results" / "review" / "discontinuity_sample.png"
    create_contact_sheet(crops, output_path)

    print()
    print("=" * 60)
    print("REVIEW GATE")
    print("=" * 60)
    print("The 'discontinuity' label is provisionally mapped to V3 (surface irregularity).")
    print()
    print("Review the contact sheet and determine:")
    print("  1. Do these look like surface irregularities (bumps, dips, undercuts)?")
    print("  2. Or do they look like porosity (round holes) -> should map to V1?")
    print("  3. Or are they cracks -> should map to V2?")
    print("  4. Or are they mixed / unclassifiable -> should discard?")
    print()
    print("If the mapping is wrong, update configs/class_map.yaml before training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
