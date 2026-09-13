"""Patch converter: extract vision training patches from raw datasets.

Converts YOLO/LabelMe annotations into cropped patches organized by vision
head (V1-V4). This replaces remap_classes.py which incorrectly mapped to
Theta singletons.

Key requirements:
- Mapping is read from configs/class_map.yaml - hardcoding is a bug
- Unknown labels are a HARD ERROR, not a silent skip
- Negative samples (V4) are sized from the same distribution as positives
- Output manifest tracks source_image for grouped splitting

Usage: python -m weldfusion.vision.build_patches
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from weldfusion.frame import vision_head_order
from weldfusion.utils.config import load_named, repo_root


@dataclass
class Box:
    """Bounding box with metadata."""
    x: int
    y: int
    w: int
    h: int
    label: str
    source_image: str
    source_dataset: str
    is_negative: bool = False


@dataclass
class PatchRecord:
    """Record for manifest.csv."""
    patch_path: str
    head: str
    source_dataset: str
    source_image: str
    source_label: str
    box_x: int
    box_y: int
    box_w: int
    box_h: int
    is_negative: bool


def load_class_map() -> dict[str, dict[str, str]]:
    """Load source label -> head mapping from class_map.yaml."""
    return load_named("class_map")["sources"]


def parse_labelme_json(json_path: Path) -> list[Box]:
    """Parse LabelMe JSON annotation file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    source_image = json_path.stem

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape["label"]
        points = shape["points"]
        x1, y1 = points[0]
        x2, y2 = points[1]
        x, y = int(min(x1, x2)), int(min(y1, y2))
        w, h = int(abs(x2 - x1)), int(abs(y2 - y1))
        if w > 0 and h > 0:
            boxes.append(Box(
                x=x, y=y, w=w, h=h,
                label=label,
                source_image=source_image,
                source_dataset="weld-dataset",
            ))

    return boxes


def parse_yolo_txt(txt_path: Path, img_w: int, img_h: int, class_names: list[str]) -> list[Box]:
    """Parse YOLO txt annotation file."""
    boxes = []
    source_image = txt_path.stem

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            # Convert normalized to pixel coords
            x = int((cx - bw / 2) * img_w)
            y = int((cy - bh / 2) * img_h)
            w = int(bw * img_w)
            h = int(bh * img_h)

            if w > 0 and h > 0 and class_id < len(class_names):
                boxes.append(Box(
                    x=x, y=y, w=w, h=h,
                    label=class_names[class_id],
                    source_image=source_image,
                    source_dataset="welding-defect-kaggle",
                ))

    return boxes


def crop_with_context(
    img: np.ndarray,
    box: Box,
    context_factor: float = 1.75,
    target_size: int = 224,
) -> np.ndarray:
    """Crop box with context margin, resize to target size."""
    h_img, w_img = img.shape[:2]

    # Expand box by context factor
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    new_w = box.w * context_factor
    new_h = box.h * context_factor

    x1 = int(max(0, cx - new_w / 2))
    y1 = int(max(0, cy - new_h / 2))
    x2 = int(min(w_img, cx + new_w / 2))
    y2 = int(min(h_img, cy + new_h / 2))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)


def sample_negative_boxes(
    img_w: int,
    img_h: int,
    positive_boxes: list[Box],
    size_distribution: list[tuple[int, int]],
    n_samples: int,
    source_image: str,
    source_dataset: str,
    rng: random.Random,
) -> list[Box]:
    """Sample negative boxes from annotation-free regions.

    CRITICAL: sizes are drawn from the same distribution as positives.
    """
    negatives = []

    # Build occupied mask
    occupied = np.zeros((img_h, img_w), dtype=bool)
    for box in positive_boxes:
        y1, y2 = max(0, box.y), min(img_h, box.y + box.h)
        x1, x2 = max(0, box.x), min(img_w, box.x + box.w)
        occupied[y1:y2, x1:x2] = True

    attempts = 0
    max_attempts = n_samples * 20

    while len(negatives) < n_samples and attempts < max_attempts:
        attempts += 1

        # Sample size from positive distribution
        w, h = rng.choice(size_distribution)

        # Random position
        if w >= img_w or h >= img_h:
            continue
        x = rng.randint(0, img_w - w)
        y = rng.randint(0, img_h - h)

        # Check overlap with occupied regions
        region = occupied[y:y+h, x:x+w]
        if region.any():
            continue

        negatives.append(Box(
            x=x, y=y, w=w, h=h,
            label="_background",
            source_image=source_image,
            source_dataset=source_dataset,
            is_negative=True,
        ))

    return negatives


def iter_weld_dataset(root: Path) -> Iterator[tuple[Path, list[Box]]]:
    """Iterate over weld-dataset images and annotations."""
    data_dir = root / "data" / "weld-dataset" / "high_resolution_welds"
    if not data_dir.exists():
        print(f"Warning: {data_dir} not found")
        return

    for json_path in sorted(data_dir.glob("*.json")):
        img_path = json_path.with_suffix(".jpg")
        if not img_path.exists():
            continue
        boxes = parse_labelme_json(json_path)
        yield img_path, boxes


def iter_kaggle_dataset(root: Path) -> Iterator[tuple[Path, list[Box]]]:
    """Iterate over welding-defect-kaggle images and annotations."""
    # Class names from the dataset
    class_names = ["Bad Weld", "Good Weld", "Defect"]

    for version in ["The Welding Defect Dataset", "The Welding Defect Dataset - v2"]:
        base = root / "data" / "welding-defect-kaggle" / version / version
        for split in ["train", "valid", "test"]:
            img_dir = base / split / "images"
            lbl_dir = base / split / "labels"
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                txt_path = lbl_dir / (img_path.stem + ".txt")
                if not txt_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                boxes = parse_yolo_txt(txt_path, w, h, class_names)
                yield img_path, boxes


def build_patches(
    root: Path | None = None,
    seed: int = 42,
    target_size: int = 224,
    negative_multiplier: float = 1.5,
) -> Path:
    """Build vision patches from raw datasets.

    Returns path to manifest.csv.
    """
    root = root or repo_root()
    out_dir = root / "data" / "processed" / "vision_patches"
    class_map = load_class_map()
    heads = vision_head_order()
    rng = random.Random(seed)

    # Create output directories
    for head in heads:
        (out_dir / head).mkdir(parents=True, exist_ok=True)

    records: list[PatchRecord] = []
    size_distribution: list[tuple[int, int]] = []
    counts_by_head: dict[str, int] = defaultdict(int)
    patch_id = 0

    # Collect all boxes first to build size distribution
    all_images_boxes: list[tuple[Path, list[Box]]] = []

    print("Scanning weld-dataset...")
    for img_path, boxes in iter_weld_dataset(root):
        all_images_boxes.append((img_path, boxes))
        for box in boxes:
            size_distribution.append((box.w, box.h))

    print("Scanning welding-defect-kaggle...")
    for img_path, boxes in iter_kaggle_dataset(root):
        all_images_boxes.append((img_path, boxes))
        for box in boxes:
            size_distribution.append((box.w, box.h))

    if not size_distribution:
        raise RuntimeError("No boxes found in any dataset")

    print(f"Found {len(all_images_boxes)} images with {len(size_distribution)} boxes")

    # Process images
    for img_path, boxes in all_images_boxes:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        dataset = boxes[0].source_dataset if boxes else "unknown"
        source_image = img_path.stem
        dataset_map = class_map.get(dataset, {})

        # Process positive boxes
        positive_boxes = []
        for box in boxes:
            label = box.label

            # Strict mapping - unknown labels are errors
            if label not in dataset_map:
                raise ValueError(
                    f"Unknown label '{label}' in {dataset}. "
                    f"Add it to configs/class_map.yaml or mark as 'discard'."
                )

            head = dataset_map[label]
            if head == "discard":
                continue

            if head not in heads:
                raise ValueError(f"Invalid head '{head}' for label '{label}'")

            # Crop patch
            patch = crop_with_context(img, box, target_size=target_size)
            if patch is None:
                continue

            # Save patch
            patch_name = f"{patch_id:06d}.jpg"
            patch_path = out_dir / head / patch_name
            cv2.imwrite(str(patch_path), patch)

            records.append(PatchRecord(
                patch_path=f"{head}/{patch_name}",
                head=head,
                source_dataset=dataset,
                source_image=source_image,
                source_label=label,
                box_x=box.x,
                box_y=box.y,
                box_w=box.w,
                box_h=box.h,
                is_negative=False,
            ))

            counts_by_head[head] += 1
            positive_boxes.append(box)
            patch_id += 1

    # Sample negatives for V4
    print("Sampling negative patches for V4...")
    max_positive = max(counts_by_head.values()) if counts_by_head else 0
    target_negatives = int(max_positive * negative_multiplier)
    negatives_per_image = max(1, target_negatives // len(all_images_boxes))

    for img_path, boxes in all_images_boxes:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        source_image = img_path.stem
        dataset = boxes[0].source_dataset if boxes else "weld-dataset"

        neg_boxes = sample_negative_boxes(
            w_img, h_img, boxes, size_distribution,
            negatives_per_image, source_image, dataset, rng
        )

        for box in neg_boxes:
            patch = crop_with_context(img, box, target_size=target_size)
            if patch is None:
                continue

            patch_name = f"{patch_id:06d}.jpg"
            patch_path = out_dir / "V4" / patch_name
            cv2.imwrite(str(patch_path), patch)

            records.append(PatchRecord(
                patch_path=f"V4/{patch_name}",
                head="V4",
                source_dataset=dataset,
                source_image=source_image,
                source_label="_background",
                box_x=box.x,
                box_y=box.y,
                box_w=box.w,
                box_h=box.h,
                is_negative=True,
            ))

            counts_by_head["V4"] += 1
            patch_id += 1

    # Write manifest
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patch_path", "head", "source_dataset", "source_image",
            "source_label", "box_x", "box_y", "box_w", "box_h", "is_negative"
        ])
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "patch_path": rec.patch_path,
                "head": rec.head,
                "source_dataset": rec.source_dataset,
                "source_image": rec.source_image,
                "source_label": rec.source_label,
                "box_x": rec.box_x,
                "box_y": rec.box_y,
                "box_w": rec.box_w,
                "box_h": rec.box_h,
                "is_negative": rec.is_negative,
            })

    print(f"\nPatch counts by head:")
    for head in heads:
        print(f"  {head}: {counts_by_head[head]}")
    print(f"\nTotal: {len(records)} patches")
    print(f"Manifest: {manifest_path}")

    return manifest_path


def main() -> None:
    build_patches()


if __name__ == "__main__":
    main()
