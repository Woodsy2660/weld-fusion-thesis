"""Training loop for ResNet-50 vision classifier.

Usage: python -m weldfusion.vision.train --config configs/vision_train.yaml

Does not run as part of scaffolding: requires processed images on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from weldfusion.frame import class_order
from weldfusion.utils.config import load_yaml
from weldfusion.utils.provenance import write_manifest
from weldfusion.vision.dataset import get_dataloaders
from weldfusion.vision.model import build_model, freeze_backbone


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_train.yaml")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Frame classes: {list(class_order())}")

    train_loader, val_loader, _test_loader = get_dataloaders(cfg)
    model = build_model(
        pretrained=cfg["model"]["pretrained"] == "imagenet",
    ).to(device)
    if cfg["model"].get("freeze_layers"):
        freeze_backbone(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = None
    sched_cfg = cfg["training"].get("scheduler") or {}
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["epochs"],
            eta_min=sched_cfg.get("min_lr", 0.0),
        )

    best_val_acc = 0.0
    patience_counter = 0
    save_dir = Path("models/vision")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        print(
            f"Epoch {epoch + 1}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_resnet50.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    write_manifest(
        "results/manifests/vision_train.json",
        configs={"vision_train": cfg, "frame": {"classes": list(class_order())}},
        extra={"best_val_acc": best_val_acc},
    )


if __name__ == "__main__":
    main()
