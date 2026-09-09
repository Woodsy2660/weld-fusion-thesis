"""Remap public vision datasets onto the frame in configs/frame.yaml.

Source folder names are dataset-specific (this map). Target classes and
processed folder names come from the frame. Unmapped folders are discarded.
LoHi-WELD ships YOLO/COCO detection labels; this script still assumes
class-named image folders. A detection-to-folder converter is out of
scope for scaffolding.

Usage: python -m weldfusion.vision.remap_classes
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from weldfusion.frame import class_order, folder_names
from weldfusion.utils.config import repo_root

# Dataset-specific source folder names -> frame class letter.
# This is not the class list; the class list lives in frame.yaml.
SOURCE_FOLDER_TO_CLASS = {
    # LoHi-WELD
    "pores": "P",
    "discontinuities": "C",
    # Wijaya Kaggle
    "porosity": "P",
    "crack": "C",
    "good welding": "N",
    "Good Welding": "N",
    "bad welding": None,
    "excess reinforcement": None,  # geometric; measured by the geometry modality
    "spatters": None,
    # FINAL WELDING
    "microporous": "P",
    "lack of fusion": "L",
    "lack_of_fusion": "L",
    "crazing": "C",
    "good weld": "N",
    "good_weld": "N",
    "undercut": "U",
}

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42
RAW_DIRS = [
    "data/raw/vision/lohi_weld",
    "data/raw/vision/wijaya_kaggle",
    "data/raw/vision/final_welding",
]
OUTPUT_DIR = "data/processed/vision"


def collect_and_remap(root: Path | None = None) -> dict[str, list[Path]]:
    root = root or repo_root()
    targets = class_order()
    all_images = {cls: [] for cls in targets}

    for raw_dir in RAW_DIRS:
        raw_path = root / raw_dir
        if not raw_path.exists():
            print(f"Skipping {raw_dir} (not found)")
            continue
        for class_dir in raw_path.iterdir():
            if not class_dir.is_dir():
                continue
            source_class = class_dir.name
            target = SOURCE_FOLDER_TO_CLASS.get(source_class)
            if target is None:
                print(f"  Discarding class: {source_class}")
                continue
            if target not in all_images:
                raise RuntimeError(
                    f"SOURCE_FOLDER_TO_CLASS maps {source_class!r} to {target!r}, "
                    f"which is not in frame.yaml Theta={list(targets)}"
                )
            images = (
                list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.jpeg"))
            )
            all_images[target].extend(images)
            print(f"  {source_class} -> {target}: {len(images)} images")
    return all_images


def stratified_split(all_images: dict[str, list[Path]], root: Path | None = None) -> None:
    root = root or repo_root()
    folders = folder_names()
    random.seed(SEED)
    for cls, images in all_images.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }
        folder = folders[cls]
        for split_name, split_images in splits.items():
            dest = root / OUTPUT_DIR / split_name / folder
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, dest / img.name)
            print(f"  {folder}/{split_name}: {len(split_images)} images")


def main() -> None:
    print("Collecting and remapping classes from frame.yaml ...")
    all_images = collect_and_remap()
    print("\nClass totals:")
    for cls, imgs in all_images.items():
        print(f"  {cls}: {len(imgs)}")
    print("\nSplitting 80/10/10...")
    stratified_split(all_images)
    print("\nDone. Processed data in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
