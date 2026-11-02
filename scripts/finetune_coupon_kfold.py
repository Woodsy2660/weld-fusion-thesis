#!/usr/bin/env python3
"""
Fine-tune vision model on real coupon patches with k=4 coupon-grouped folds.

For each fold i:
  - Train on folds != i (3 folds worth of coupons)
  - Validate/test on fold i

Reports per-fold macro-F1 and average across folds.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.vision.model import load_model
from src.utils.frame import load_frame


REPO_ROOT = Path(__file__).parents[1]
MODEL_PATH = REPO_ROOT / "models" / "vision" / "best_resnet18_ft-layer2.pth"
CONFIG_PATH = REPO_ROOT / "configs" / "vision_train.yaml"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "vision_coupon_patches" / "manifest.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "vision" / "coupon_eval"

# Fine-tuning hyperparameters (conservative for small dataset)
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 16
PATIENCE = 8


class CouponPatchDataset(Dataset):
    """Dataset for coupon patches with fold-based split."""

    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform
        self.target_to_idx = {"{P}": 0, "{N,U}": 1, "{N,U,L}": 2}

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        row = self.patches[i]
        img_path = REPO_ROOT / row["patch_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = self.target_to_idx[row["vision_target"]]
        return img, target


def get_transforms(cfg, train=True):
    """Get train or eval transforms."""
    size = cfg["data"]["image_size"]

    if train:
        aug = cfg["augmentation"]
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(aug.get("horizontal_flip", 0.5)),
            transforms.RandomVerticalFlip(aug.get("vertical_flip", 0.5)),
            transforms.RandomRotation(aug["rotation_degrees"]),
            transforms.ColorJitter(brightness=aug.get("brightness", 0.2),
                                   contrast=aug.get("contrast", 0.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["normalization"]["mean"],
                                 std=cfg["normalization"]["std"]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg["normalization"]["mean"],
                                 std=cfg["normalization"]["std"]),
        ])


def compute_class_weights(patches):
    """Compute inverse-frequency class weights."""
    target_to_idx = {"{P}": 0, "{N,U}": 1, "{N,U,L}": 2}
    counts = Counter(target_to_idx[p["vision_target"]] for p in patches)
    n = len(patches)
    k = 3
    weights = torch.tensor([n / (k * counts[i]) for i in range(k)], dtype=torch.float32)
    return weights / weights.mean()


def train_fold(fold_idx, train_patches, test_patches, cfg, device):
    """Train and evaluate on one fold."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}")
    print(f"{'='*60}")
    print(f"Train: {len(train_patches)} patches")
    print(f"Test:  {len(test_patches)} patches")

    # Datasets
    train_ds = CouponPatchDataset(train_patches, get_transforms(cfg, train=True))
    test_ds = CouponPatchDataset(test_patches, get_transforms(cfg, train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Class weights
    class_weights = compute_class_weights(train_patches).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    # Load pretrained model
    model = load_model(MODEL_PATH, num_heads=3, device=device)
    model.to(device)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training
    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1).cpu().numpy()
                y_true.extend(targets.numpy())
                y_pred.extend(preds)

        macro_f1 = f1_score(y_true, y_pred, average="macro")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: loss={train_loss/len(train_loader):.4f}, "
                  f"macro-F1={macro_f1:.4f}")

        # Early stopping
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and final evaluation
    model.load_state_dict(best_state)
    model.eval()

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(targets.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = (y_true == y_pred).mean()

    print(f"\nFold {fold_idx} best macro-F1: {best_f1:.4f}")
    print(f"Fold {fold_idx} final test:")
    print(classification_report(y_true, y_pred, target_names=["V1", "V3", "V4"], digits=4,
                                zero_division=0))

    # Save model
    model_path = OUTPUT_DIR / f"finetuned_fold{fold_idx}.pth"
    torch.save(best_state, model_path)
    print(f"Model saved: {model_path}")

    return {
        "fold": fold_idx,
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_probs": y_probs.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINE-TUNING WITH 4-FOLD COUPON-GROUPED CROSS-VALIDATION")
    print("=" * 70)

    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Load all patches
    patches = []
    with open(MANIFEST_PATH) as f:
        for row in csv.DictReader(f):
            patches.append(row)

    print(f"Total patches: {len(patches)}")

    # Group by fold
    folds = {}
    for p in patches:
        fold = p["fold"]
        if fold not in folds:
            folds[fold] = []
        folds[fold].append(p)

    print(f"Folds: {sorted(folds.keys())}")
    for f in sorted(folds.keys()):
        print(f"  Fold {f}: {len(folds[f])} patches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Run k-fold
    results = []
    for fold_idx in sorted(folds.keys()):
        test_patches = folds[fold_idx]
        train_patches = []
        for f in folds:
            if f != fold_idx:
                train_patches.extend(folds[f])

        fold_result = train_fold(fold_idx, train_patches, test_patches, cfg, device)
        results.append(fold_result)

    # Summary
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    macro_f1s = [r["macro_f1"] for r in results]
    accs = [r["accuracy"] for r in results]

    print("\nPer-fold macro-F1:")
    for r in results:
        print(f"  Fold {r['fold']}: {r['macro_f1']:.4f}")

    mean_f1 = np.mean(macro_f1s)
    std_f1 = np.std(macro_f1s)
    print(f"\nMean macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Mean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Compare to baselines
    trivial_f1 = 0.2338
    zeroshot_f1 = 0.2934

    print(f"\nBaseline comparison:")
    print(f"  Trivial (always V4):   {trivial_f1:.4f}")
    print(f"  Zero-shot transfer:    {zeroshot_f1:.4f}")
    print(f"  Fine-tuned (CV mean):  {mean_f1:.4f}")
    print(f"  Lift over trivial:     {mean_f1 - trivial_f1:+.4f}")
    print(f"  Lift over zero-shot:   {mean_f1 - zeroshot_f1:+.4f}")

    # Save results
    results_path = OUTPUT_DIR / "kfold_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "n_folds": len(results),
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "per_fold": results,
            "mean_macro_f1": float(mean_f1),
            "std_macro_f1": float(std_f1),
            "mean_accuracy": float(np.mean(accs)),
            "trivial_baseline": trivial_f1,
            "zeroshot_baseline": zeroshot_f1,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
