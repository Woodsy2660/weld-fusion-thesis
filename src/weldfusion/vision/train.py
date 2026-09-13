"""Training loop for ResNet-50 vision classifier with grouped splitting.

The vision classifier outputs logits for 4 heads (V1-V4), NOT 5 Theta
singletons. Uses grouped K-fold by source_image to prevent leakage.

Key features:
- Grouped splitting by source_image (no patch-level random splits)
- Class-weighted cross-entropy for imbalance
- Active heads configuration (V2 may be empty)
- Saves raw logits for temperature scaling calibration
- Reports macro-F1, per-class precision/recall/F1

Usage: python -m weldfusion.vision.train
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)

from weldfusion.frame import vision_head_order, vision_head_count
from weldfusion.utils.config import load_named, repo_root
from weldfusion.utils.provenance import write_manifest, git_sha
from weldfusion.vision.dataset import PatchDataset
from weldfusion.vision.model import build_model, freeze_backbone, unfreeze_backbone
from weldfusion.vision.splits import (
    load_manifest,
    grouped_kfold_split,
    validate_no_image_leakage,
    get_fold_stats,
)


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Compute inverse frequency weights for class-weighted CE."""
    counts = Counter(labels)
    total = len(labels)

    weights = []
    for i in range(n_classes):
        count = counts.get(i, 0)
        if count == 0:
            weights.append(0.0)  # Zero weight for missing classes
        else:
            weights.append(total / (n_classes * count))

    return torch.tensor(weights, dtype=torch.float32)


def check_active_heads(df: pd.DataFrame, active_heads: list[str]) -> None:
    """Verify active heads have training data, fail loudly if empty."""
    head_order = vision_head_order()
    head_counts = df["head"].value_counts().to_dict()

    for head in head_order:
        count = head_counts.get(head, 0)
        if head in active_heads and count == 0:
            raise RuntimeError(
                f"FATAL: Active head {head} has ZERO training samples. "
                f"Either add data for {head} or remove it from vision.active_heads."
            )
        if head not in active_heads:
            print(f"  Note: Head {head} is INACTIVE (has {count} samples)")


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


def validate_with_logits(model, loader, criterion, device):
    """Validate and return all logits for calibration."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_logits = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    return (
        running_loss / total,
        correct / total,
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )


def compute_metrics(y_true, y_pred, head_names):
    """Compute per-class and macro metrics."""
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(head_names))), zero_division=0
    )

    metrics = {
        "macro_f1": float(np.mean(f1)),
        "per_class": {},
    }

    for i, head in enumerate(head_names):
        metrics["per_class"][head] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(head_names))))
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def train_fold(
    fold_idx: int,
    train_dataset: Subset,
    val_dataset: Subset,
    cfg: dict,
    device: torch.device,
    results_dir: Path,
    class_weights: torch.Tensor,
) -> dict:
    """Train one fold and save logits."""
    head_names = vision_head_order()
    n_heads = len(head_names)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4) if use_cuda else 0,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4) if use_cuda else 0,
        pin_memory=use_cuda,
    )

    model = build_model(n_heads=n_heads, pretrained=True).to(device)

    # Freeze backbone initially if configured
    freeze_epochs = cfg["model"].get("freeze_epochs", 0)
    if freeze_epochs > 0:
        freeze_backbone(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("min_lr", 1e-6),
    )

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(cfg["training"]["epochs"]):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs and freeze_epochs > 0:
            print(f"  Unfreezing backbone at epoch {epoch}")
            unfreeze_backbone(model)
            # Reset optimizer to include all params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["learning_rate"] / 10,  # Lower LR for fine-tuning
                weight_decay=cfg["training"]["weight_decay"],
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_logits, val_labels, val_preds = validate_with_logits(
            model, val_loader, criterion, device
        )
        scheduler.step()

        # Compute macro-F1
        metrics = compute_metrics(val_labels, val_preds, head_names)
        val_f1 = metrics["macro_f1"]

        print(
            f"  Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            best_metrics = metrics
            best_logits = val_logits
            best_labels = val_labels
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Save best model
    model_path = results_dir / f"model_fold{fold_idx}.pth"
    torch.save(best_model_state, model_path)

    # Save raw logits for calibration (REQUIRED)
    logits_path = results_dir / f"logits_fold{fold_idx}.npz"
    np.savez(
        logits_path,
        logits=best_logits,
        labels=best_labels,
        head_names=head_names,
    )

    # Save metrics
    metrics_path = results_dir / f"metrics_fold{fold_idx}.json"
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    return {
        "fold": fold_idx,
        "best_macro_f1": best_val_f1,
        "metrics": best_metrics,
        "model_path": str(model_path),
        "logits_path": str(logits_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_train.yaml")
    parser.add_argument("--manifest", default="data/processed/vision_patches/manifest.csv")
    args = parser.parse_args(argv)

    root = repo_root()
    cfg = load_named("vision_train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Vision heads: {vision_head_order()}")

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    df = load_manifest(manifest_path)
    print(f"Loaded {len(df)} patches from {len(df['source_image'].unique())} images")

    # Check active heads
    active_heads = cfg.get("active_heads", list(vision_head_order()))
    print(f"Active heads: {active_heads}")
    check_active_heads(df, active_heads)

    # Create dataset
    patches_dir = manifest_path.parent
    dataset = PatchDataset(patches_dir, df)

    # Grouped K-fold split
    n_folds = cfg["training"].get("n_folds", 5)
    splits = grouped_kfold_split(df, n_splits=n_folds, seed=cfg["training"]["seed"])
    validate_no_image_leakage(df, splits)

    fold_stats = get_fold_stats(df, splits)
    print(f"\nFold statistics:")
    for stat in fold_stats:
        print(f"  Fold {stat['fold']}: {stat['n_train_patches']} train, {stat['n_val_patches']} val, "
              f"{stat['n_train_images']} train images, {stat['n_val_images']} val images")

    # Compute class weights
    head_to_idx = {h: i for i, h in enumerate(vision_head_order())}
    all_labels = np.array([head_to_idx[h] for h in df["head"]])
    class_weights = compute_class_weights(all_labels, len(vision_head_order()))
    print(f"\nClass weights: {class_weights.tolist()}")

    # Results directory
    results_dir = root / "results" / "vision"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train each fold
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*60}")

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        result = train_fold(
            fold_idx, train_dataset, val_dataset, cfg, device, results_dir, class_weights
        )
        fold_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    macro_f1s = [r["best_macro_f1"] for r in fold_results]
    print(f"Macro-F1 across folds: {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")

    for r in fold_results:
        print(f"  Fold {r['fold']}: macro-F1 = {r['best_macro_f1']:.4f}")
        for head, metrics in r["metrics"]["per_class"].items():
            print(f"    {head}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    # Write run manifest
    inactive_heads = [h for h in vision_head_order() if h not in active_heads]
    manifest_data = {
        "git_sha": git_sha(),
        "config_hash": cfg.get("_hash", "unknown"),
        "seed": cfg["training"]["seed"],
        "active_heads": active_heads,
        "inactive_heads": inactive_heads,
        "class_weighting": "inverse_frequency",
        "n_folds": n_folds,
        "dataset_counts": df["head"].value_counts().to_dict(),
        "n_source_images": len(df["source_image"].unique()),
        "fold_results": fold_results,
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "std_macro_f1": float(np.std(macro_f1s)),
    }

    manifest_path = results_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"\nRun manifest: {manifest_path}")


if __name__ == "__main__":
    main()
