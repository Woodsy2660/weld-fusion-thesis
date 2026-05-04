"""
PyTorch dataset and transforms for vision modality.
Steps 4-5: resize, augment, normalise.
"""
import torch
from torchvision import datasets, transforms


def get_train_transforms(cfg):
    """Training transforms: resize + augment + normalise."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            cfg["data"]["image_size"],
            scale=tuple(cfg["augmentation"]["random_resized_crop"]["scale"])
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
    ])


def get_eval_transforms(cfg):
    """Inference transforms: resize + normalise only. No augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg["data"]["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg["normalization"]["mean"],
            std=cfg["normalization"]["std"],
        ),
    ])


def get_dataloaders(cfg):
    """Create train/val/test dataloaders from processed directory."""
    train_ds = datasets.ImageFolder(
        cfg["data"]["train_dir"],
        transform=get_train_transforms(cfg),
    )
    val_ds = datasets.ImageFolder(
        cfg["data"]["val_dir"],
        transform=get_eval_transforms(cfg),
    )
    test_ds = datasets.ImageFolder(
        cfg["data"]["test_dir"],
        transform=get_eval_transforms(cfg),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
    )

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader
