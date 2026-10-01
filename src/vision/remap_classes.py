"""
SUPERSEDED -- kept for reference only. Do not run.

This was written against the originally planned layout (per-class folders under
data/raw/vision/lohi_weld, wijaya_kaggle, final_welding), which does not match
what is actually on disk. The real datasets are YOLO detection/segmentation
exports under data/Vision/, and its CLASS_MAP names ("pores", "good welding",
"crazing", ...) do not exist in any of them.

The working pipeline is:
    src/vision/convert_and_remap.py   annotations -> data/staged/
    src/vision/merge_dedup_split.py   -> data/merged/ -> data/processed/vision/

Original docstring follows.

Step 1-3 of vision pipeline: Download datasets, remap class labels
to {porosity, lack_of_fusion, crack, no_defect}, and split 80/10/10.

Run once after downloading raw datasets.
Usage: python src/vision/remap_classes.py
"""
import os
import shutil
import random
from pathlib import Path

# Class mapping from source datasets to target taxonomy
CLASS_MAP = {
    # LoHi-WELD
    "pores": "porosity",
    "discontinuities": "crack",
    # Wijaya Kaggle
    "porosity": "porosity",
    "crack": "crack",
    "good welding": "no_defect",
    "Good Welding": "no_defect",
    "bad welding": None,            # discard - ambiguous
    "excess reinforcement": None,   # discard - geometric defect
    "spatters": None,               # discard - cosmetic
    # FINAL WELDING
    "porosity": "porosity",
    "microporous": "porosity",
    "lack of fusion": "lack_of_fusion",
    "lack_of_fusion": "lack_of_fusion",
    "crack": "crack",
    "crazing": "crack",
    "good weld": "no_defect",
    "good_weld": "no_defect",
    # Add more mappings as you inspect the actual dataset folders
}

TARGET_CLASSES = ["porosity", "lack_of_fusion", "crack", "no_defect"]
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

RAW_DIRS = [
    "data/raw/vision/lohi_weld",
    "data/raw/vision/wijaya_kaggle",
    "data/raw/vision/final_welding",
]
OUTPUT_DIR = "data/processed/vision"


def collect_and_remap():
    """Walk raw dataset directories and copy images into remapped class folders."""
    all_images = {cls: [] for cls in TARGET_CLASSES}

    for raw_dir in RAW_DIRS:
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            print(f"Skipping {raw_dir} (not found)")
            continue

        for class_dir in raw_path.iterdir():
            if not class_dir.is_dir():
                continue

            source_class = class_dir.name
            target_class = CLASS_MAP.get(source_class)

            if target_class is None:
                print(f"  Discarding class: {source_class}")
                continue

            images = list(class_dir.glob("*.png")) + \
                     list(class_dir.glob("*.jpg")) + \
                     list(class_dir.glob("*.jpeg"))

            for img in images:
                all_images[target_class].append(img)

            print(f"  {source_class} -> {target_class}: {len(images)} images")

    return all_images


def stratified_split(all_images):
    """Split each class independently to maintain class balance."""
    random.seed(SEED)

    for cls, images in all_images.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest = Path(OUTPUT_DIR) / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)

            for img in split_images:
                shutil.copy2(img, dest / img.name)

            print(f"  {cls}/{split_name}: {len(split_images)} images")


if __name__ == "__main__":
    print("Collecting and remapping classes...")
    all_images = collect_and_remap()

    print("\nClass totals:")
    for cls, imgs in all_images.items():
        print(f"  {cls}: {len(imgs)}")

    print("\nSplitting 80/10/10...")
    stratified_split(all_images)
    print("\nDone. Processed data in:", OUTPUT_DIR)
