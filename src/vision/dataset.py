"""
Patch dataset and transforms for the vision modality.

Samples are per-instance crops produced by convert_and_remap.py, already
letterboxed to a square at write time (aspect preserved, edges replicated).
Split membership is read from the manifest rather than from directory layout,
because the split is GROUPED by source image -- the same head directory holds
patches belonging to train, val and test.
"""
import collections
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_patches_dir(cfg):
    """Resolve the patch root; relative paths resolve against the repo root."""
    root = Path(cfg["data"]["patches_dir"])
    return root if root.is_absolute() else REPO_ROOT / root


class PatchDataset(Dataset):
    """
    One row of manifest.csv per sample.

    Exposes `.targets` (head index per sample), `.classes` (head names) and
    `.groups` (source image per sample) so class weights and the grouped-split
    assertion can be computed without re-reading the manifest.
    """

    def __init__(self, manifest_path, split, head_names, transform=None,
                 patches_root=None, all_splits=False):
        self.head_names = list(head_names)
        self.classes = list(head_names)
        self.head_to_idx = {h: i for i, h in enumerate(self.head_names)}
        self.transform = transform
        self.root = Path(patches_root or Path(manifest_path).parent)

        self.rows, self.targets, self.groups = [], [], []
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                # all_splits is for feature extraction over the whole corpus
                # (grouping, embeddings) where split membership is irrelevant.
                if not all_splits and row.get("split") != split:
                    continue
                head = row["head"]
                if head not in self.head_to_idx:
                    raise ValueError(
                        f"{manifest_path}: head {head!r} is not one of "
                        f"{self.head_names}. Frame and manifest disagree."
                    )
                self.rows.append(row)
                self.targets.append(self.head_to_idx[head])
                self.groups.append(row["source_image"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row["patch_path"]) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[i]


def get_train_transforms(cfg):
    """
    Training augmentation.

    Only transforms that cannot change what a defect IS: geometry (flips,
    rotation) and illumination (brightness/contrast, mild blur). Vertical flip
    is valid here -- a weld bead has no canonical up.

    Deliberately absent:
      * hue and saturation jitter -- these are near-monochrome steel surfaces
        and a hue shift is unphysical;
      * aggressive scale jitter -- crop scale carries real signal about defect
        size, and patches already arrive letterboxed at image_size.
    """
    aug = cfg["augmentation"]
    blur = aug.get("gaussian_blur") or {}
    ops = [
        transforms.RandomHorizontalFlip(aug.get("horizontal_flip", 0.5)),
        transforms.RandomVerticalFlip(aug.get("vertical_flip", 0.5)),
        transforms.RandomRotation(aug["rotation_degrees"]),
        transforms.ColorJitter(brightness=aug.get("brightness", 0.0),
                               contrast=aug.get("contrast", 0.0)),
    ]
    if blur.get("p", 0):
        ops.append(transforms.RandomApply(
            [transforms.GaussianBlur(
                kernel_size=blur.get("kernel_size", 5),
                sigma=tuple(blur.get("sigma", [0.1, 1.0])))],
            p=blur["p"]))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["normalization"]["mean"],
                             std=cfg["normalization"]["std"]),
    ]
    return transforms.Compose(ops)


def get_eval_transforms(cfg):
    """
    Inference transforms: no augmentation, and no centre crop.

    Patches are already square and already the right size. A Resize(256) +
    CenterCrop(224) here would silently throw away the context margin that the
    cropper was careful to include.
    """
    size = cfg["data"]["image_size"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["normalization"]["mean"],
                             std=cfg["normalization"]["std"]),
    ])


def get_dataloaders(cfg, frame):
    """Build train/val/test loaders from the patch manifest."""
    patches = resolve_patches_dir(cfg)
    manifest = patches / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found. Run src/vision/convert_and_remap.py then "
            f"src/vision/merge_dedup_split.py first."
        )

    datasets = {
        split: PatchDataset(manifest, split, frame.head_names,
                            transform=(get_train_transforms(cfg) if split == "train"
                                       else get_eval_transforms(cfg)),
                            patches_root=patches)
        for split in ("train", "val", "test")
    }

    for split, ds in datasets.items():
        if len(ds) == 0:
            raise ValueError(
                f"split '{split}' is empty. Did merge_dedup_split.py run?"
            )

    # Fail loudly rather than silently training fewer heads than the frame
    # declares -- a head with no training samples can never be predicted, and
    # its BPA mass would be structurally zero.
    train_counts = collections.Counter(datasets["train"].targets)
    missing = [h for i, h in enumerate(frame.head_names)
               if train_counts[i] == 0]
    if missing:
        raise ValueError(
            f"heads {missing} have ZERO training patches but are declared in "
            f"configs/frame.yaml. Refusing to train a model that cannot "
            f"predict every focal element."
        )

    loaders = {
        split: DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=False,
        )
        for split, ds in datasets.items()
    }

    print(f"Patches root: {patches}")
    print(f"Heads: {frame.head_names}")
    for split, ds in datasets.items():
        counts = collections.Counter(ds.targets)
        per = "  ".join(f"{h}={counts[i]}" for i, h in enumerate(frame.head_names))
        print(f"  {split:<6} {len(ds):>7} patches from "
              f"{len(set(ds.groups)):>6} source images   {per}")

    return loaders, datasets


def compute_class_weights(dataset, device=None):
    """
    Inverse-frequency CrossEntropyLoss weights from the training split.

        w_c = N / (K * n_c)

    Normalised so the mean weight is 1, keeping the loss magnitude (and the
    usable learning rate) comparable to an unweighted run.
    """
    counts = collections.Counter(dataset.targets)
    k = len(dataset.classes)
    n = len(dataset.targets)
    weights = torch.tensor(
        [n / (k * counts[i]) if counts[i] else 0.0 for i in range(k)],
        dtype=torch.float32,
    )
    weights = weights / weights.mean()

    print("Class weights (inverse frequency):")
    for i, cls in enumerate(dataset.classes):
        print(f"  {cls:<6} n={counts[i]:<8} weight={weights[i]:.3f}")

    return weights.to(device) if device is not None else weights
