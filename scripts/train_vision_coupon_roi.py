#!/usr/bin/env python3
"""
Train vision model on coupon cell ROI patches with k-fold cross-validation.

Uses the correctly cropped cell patches from the weld ROI selections.
Implements 4-fold cross-validation grouped by coupon.

Output:
    results/vision/coupon_roi_train/
        fold{k}_model.pth
        fold{k}_results.json
        summary.json
        alpha_derivation.json
"""

import sys
from pathlib import Path
import json
import csv
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Vision targets (focal elements)
TARGETS = ['{P}', '{N,U}', '{N,U,L}']
TARGET_TO_IDX = {t: i for i, t in enumerate(TARGETS)}


class CellPatchDataset(Dataset):
    """Dataset for cell ROI patches."""

    def __init__(self, manifest_rows, data_dir, transform=None):
        self.rows = manifest_rows
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = self.data_dir / row['patch_path']

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = TARGET_TO_IDX[row['vision_target']]
        return img, label, row['cell_id']


def load_manifest():
    """Load the cell patches manifest."""
    manifest_path = REPO_ROOT / "data" / "processed" / "coupon_cell_patches" / "manifest.csv"
    rows = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['fold'] = int(row['fold'])
            rows.append(row)
    return rows


def get_transforms(train=True):
    """Get data transforms."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_model(num_classes=3, pretrained=True):
    """Create ResNet18 model for classification."""
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_cell_ids = []

    with torch.no_grad():
        for imgs, labels, cell_ids in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_cell_ids.extend(cell_ids)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'per_class_f1': f1_score(all_labels, all_preds, average=None).tolist(),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
    }

    return metrics, all_preds, all_labels, all_probs, all_cell_ids


def train_fold(fold_idx, train_rows, val_rows, test_rows, data_dir, output_dir,
               epochs=30, lr=0.001, batch_size=16, device='cuda'):
    """Train and evaluate for one fold."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}")
    print(f"{'='*60}")
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    # Show class distribution
    train_dist = Counter(r['vision_target'] for r in train_rows)
    print(f"Train distribution: {dict(train_dist)}")

    # Create datasets
    train_dataset = CellPatchDataset(train_rows, data_dir, get_transforms(train=True))
    val_dataset = CellPatchDataset(val_rows, data_dir, get_transforms(train=False))
    test_dataset = CellPatchDataset(test_rows, data_dir, get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    model = create_model(num_classes=3)
    model = model.to(device)

    # Class weights for imbalanced data
    class_counts = [train_dist.get(t, 1) for t in TARGETS]
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
    weights = weights / weights.sum() * len(TARGETS)
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0
    best_epoch = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / f"fold{fold_idx}_model.pth")

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.3f}, Val F1={val_metrics['macro_f1']:.3f}")

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(output_dir / f"fold{fold_idx}_model.pth"))
    test_metrics, test_preds, test_labels, test_probs, test_cell_ids = evaluate(
        model, test_loader, criterion, device)

    print(f"\nBest epoch: {best_epoch + 1}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")

    # Save results
    results = {
        'fold': fold_idx,
        'train_size': len(train_rows),
        'val_size': len(val_rows),
        'test_size': len(test_rows),
        'best_epoch': best_epoch + 1,
        'test_metrics': test_metrics,
        'predictions': [
            {
                'cell_id': cid,
                'true_label': int(tl),
                'pred_label': int(pl),
                'probs': prob.tolist(),
            }
            for cid, tl, pl, prob in zip(test_cell_ids, test_labels, test_preds, test_probs)
        ]
    }

    with open(output_dir / f"fold{fold_idx}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return test_metrics


def compute_alpha(macro_f1, trivial_f1=0.333):
    """Compute Shafer discounting alpha."""
    if macro_f1 <= trivial_f1:
        return 1.0
    skill = macro_f1 - trivial_f1
    alpha = 1 - skill / (1 - trivial_f1)
    return max(0, min(1, alpha))


def main():
    print("=" * 60)
    print("VISION MODEL TRAINING ON COUPON ROI PATCHES")
    print("=" * 60)

    # Set random seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    data_dir = REPO_ROOT / "data" / "processed" / "coupon_cell_patches"
    output_dir = REPO_ROOT / "results" / "vision" / "coupon_roi_train"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = load_manifest()
    print(f"Total samples: {len(manifest)}")

    # Get folds
    folds = sorted(set(r['fold'] for r in manifest))
    print(f"Folds: {folds}")

    # K-fold cross-validation
    all_metrics = []

    for test_fold in folds:
        # Split data
        # Test: current fold
        # Val: next fold (circular)
        # Train: remaining folds
        val_fold = folds[(folds.index(test_fold) + 1) % len(folds)]
        train_folds = [f for f in folds if f not in [test_fold, val_fold]]

        test_rows = [r for r in manifest if r['fold'] == test_fold]
        val_rows = [r for r in manifest if r['fold'] == val_fold]
        train_rows = [r for r in manifest if r['fold'] in train_folds]

        metrics = train_fold(
            fold_idx=test_fold,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=30,
            lr=0.001,
            batch_size=8,
            device=device,
        )

        all_metrics.append(metrics)

    # Aggregate results
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    macro_f1_scores = [m['macro_f1'] for m in all_metrics]
    accuracy_scores = [m['accuracy'] for m in all_metrics]

    mean_f1 = np.mean(macro_f1_scores)
    std_f1 = np.std(macro_f1_scores)
    mean_acc = np.mean(accuracy_scores)

    print(f"Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Per-fold F1: {[f'{f:.3f}' for f in macro_f1_scores]}")
    print(f"Accuracy: {mean_acc:.4f}")

    # Compute alpha
    # Trivial baseline: always predict majority class
    target_dist = Counter(r['vision_target'] for r in manifest)
    majority_class = max(target_dist, key=target_dist.get)
    trivial_acc = target_dist[majority_class] / len(manifest)
    # For F1, trivial is ~1/3 for 3-class balanced
    trivial_f1 = 1 / len(TARGETS)  # 0.333

    alpha = compute_alpha(mean_f1, trivial_f1)

    print(f"\nAlpha Derivation:")
    print(f"  Macro F1: {mean_f1:.4f}")
    print(f"  Trivial F1: {trivial_f1:.4f}")
    print(f"  Skill: {mean_f1 - trivial_f1:.4f}")
    print(f"  Alpha: {alpha:.4f}")
    print(f"  Evidence retained: {(1-alpha)*100:.1f}%")

    # Save summary
    summary = {
        'n_folds': len(folds),
        'n_samples': len(manifest),
        'class_distribution': dict(target_dist),
        'metrics': {
            'macro_f1_mean': mean_f1,
            'macro_f1_std': std_f1,
            'macro_f1_per_fold': macro_f1_scores,
            'accuracy_mean': mean_acc,
        },
        'alpha_derivation': {
            'f1_macro': mean_f1,
            'f1_trivial': trivial_f1,
            'skill': mean_f1 - trivial_f1,
            'alpha': alpha,
            'evidence_retained': 1 - alpha,
        },
        'comparison': {
            'old_alpha': 0.882,
            'new_alpha': alpha,
            'improvement': 0.882 - alpha,
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: OLD vs NEW")
    print("=" * 60)
    print(f"Old alpha (public corpus): 0.882")
    print(f"New alpha (coupon ROI):    {alpha:.3f}")
    print(f"Improvement:               {0.882 - alpha:.3f}")


if __name__ == "__main__":
    main()
