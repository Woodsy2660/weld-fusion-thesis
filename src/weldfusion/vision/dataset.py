"""PyTorch dataset and transforms for the vision modality.

Class order is taken from configs/frame.yaml, never ImageFolder's
alphabetical default (which would silently permute C,L,N,P,U).
"""

from __future__ import annotations

from torchvision import datasets, transforms

from weldfusion.frame import class_order, folder_names


def get_train_transforms(cfg: dict):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                cfg["data"]["image_size"],
                scale=tuple(cfg["augmentation"]["random_resized_crop"]["scale"]),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(cfg["augmentation"]["rotation_degrees"]),
            transforms.ColorJitter(
                brightness=cfg["augmentation"]["color_jitter"]["brightness"],
                contrast=cfg["augmentation"]["color_jitter"]["contrast"],
                saturation=cfg["augmentation"]["color_jitter"]["saturation"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg["normalization"]["mean"],
                std=cfg["normalization"]["std"],
            ),
        ]
    )


def get_eval_transforms(cfg: dict):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(cfg["data"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg["normalization"]["mean"],
                std=cfg["normalization"]["std"],
            ),
        ]
    )


def frame_class_to_idx() -> dict[str, int]:
    """Map processed-folder name -> index in frame order."""
    names = folder_names()
    return {names[c]: i for i, c in enumerate(class_order())}


class FrameImageFolder(datasets.ImageFolder):
    """ImageFolder with class_to_idx forced to frame.yaml order."""

    def find_classes(self, directory):
        folders = folder_names()
        classes = [folders[c] for c in class_order()]
        class_to_idx = {name: i for i, name in enumerate(classes)}
        return classes, class_to_idx


def get_dataloaders(cfg: dict):
    import torch

    train_ds = FrameImageFolder(cfg["data"]["train_dir"], transform=get_train_transforms(cfg))
    val_ds = FrameImageFolder(cfg["data"]["val_dir"], transform=get_eval_transforms(cfg))
    test_ds = FrameImageFolder(cfg["data"]["test_dir"], transform=get_eval_transforms(cfg))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    return train_loader, val_loader, test_loader
