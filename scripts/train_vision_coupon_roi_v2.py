#!/usr/bin/env python3
"""
Train vision model on coupon cell ROI patches with enhanced augmentation.

Improvements over v1:
- MixUp augmentation for regularization
- RandAugment for automated augmentation policy
- Label smoothing for reduced overconfidence
- Focal loss option for class imbalance

Output:
    results/vision/coupon_roi_train_v2/
        fold{k}_model.pth
        fold{k}_results.json
        summary.json
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
import torch.nn.functional as F
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


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)

        # One-hot encode targets
        targets_one_hot = torch.zeros_like(log_preds)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Apply label smoothing
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Weighted loss
        if self.weight is not None:
            weight = self.weight[targets]
            loss = (-targets_smooth * log_preds).sum(dim=-1) * weight
        else:
            loss = (-targets_smooth * log_preds).sum(dim=-1)

        return loss.mean()


def mixup_data(x, y, alpha=0.4):
    """Apply MixUp augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CellPatchDataset(Dataset):
    """Dataset for cell ROI patches with enhanced augmentation."""

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


def get_transforms(train=True, use_randaugment=True):
    """Get data transforms with enhanced augmentation."""
    if train:
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05
            ),
        ]

        # Add RandAugment if available (PyTorch >= 1.9)
        if use_randaugment:
            try:
                transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
            except AttributeError:
                # RandAugment not available, use AutoAugment
                try:
                    transform_list.append(transforms.AutoAugment(
                        transforms.AutoAugmentPolicy.IMAGENET
                    ))
                except AttributeError:
                    pass  # Neither available, use basic augmentation

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])

        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_model(num_classes=3, pretrained=True):
    """Create ResNet18 model for classification."""
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

    # Add dropout before final layer for regularization
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.4):
    """Train for one epoch with optional MixUp."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_mixup and np.random.random() > 0.5:  # Apply MixUp 50% of the time
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, mixup_alpha)
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            # For accuracy, use the dominant label
            _, predicted = outputs.max(1)
            if lam > 0.5:
                correct += predicted.eq(labels_a).sum().item()
            else:
                correct += predicted.eq(labels_b).sum().item()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += labels.size(0)

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
               epochs=50, lr=0.0005, batch_size=16, device='cuda',
               use_mixup=True, use_label_smoothing=True):
    """Train and evaluate for one fold with enhanced techniques."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}")
    print(f"{'='*60}")
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    # Show class distribution
    train_dist = Counter(r['vision_target'] for r in train_rows)
    print(f"Train distribution: {dict(train_dist)}")

    # Create datasets with enhanced augmentation
    train_dataset = CellPatchDataset(train_rows, data_dir, get_transforms(train=True))
    val_dataset = CellPatchDataset(val_rows, data_dir, get_transforms(train=False))
    test_dataset = CellPatchDataset(test_rows, data_dir, get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model with dropout
    model = create_model(num_classes=3)
    model = model.to(device)

    # Class weights for imbalanced data
    class_counts = [train_dist.get(t, 1) for t in TARGETS]
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
    weights = weights / weights.sum() * len(TARGETS)
    weights = weights.to(device)

    # Choose loss function
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    # Separate criterion for evaluation (no smoothing)
    eval_criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_f1 = 0
    best_epoch = 0
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=use_mixup
        )
        val_metrics, _, _, _, _ = evaluate(model, val_loader, eval_criterion, device)

        scheduler.step()

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / f"fold{fold_idx}_model.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.3f}, Val F1={val_metrics['macro_f1']:.3f}")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(output_dir / f"fold{fold_idx}_model.pth"))
    test_metrics, test_preds, test_labels, test_probs, test_cell_ids = evaluate(
        model, test_loader, eval_criterion, device)

    print(f"\nBest epoch: {best_epoch + 1}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Per-class F1: {[f'{f:.3f}' for f in test_metrics['per_class_f1']]}")
    print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")

    # Save results
    results = {
        'fold': fold_idx,
        'train_size': len(train_rows),
        'val_size': len(val_rows),
        'test_size': len(test_rows),
        'best_epoch': best_epoch + 1,
        'test_metrics': test_metrics,
        'config': {
            'use_mixup': use_mixup,
            'use_label_smoothing': use_label_smoothing,
            'epochs': epochs,
            'lr': lr,
        },
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
    print("VISION MODEL TRAINING V2 - ENHANCED AUGMENTATION")
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
    output_dir = REPO_ROOT / "results" / "vision" / "coupon_roi_train_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = load_manifest()
    print(f"Total samples: {len(manifest)}")

    # Get folds
    folds = sorted(set(r['fold'] for r in manifest))
    print(f"Folds: {folds}")

    print("\nEnhancements enabled:")
    print("  - MixUp augmentation (alpha=0.4)")
    print("  - RandAugment/AutoAugment")
    print("  - Label smoothing (0.1)")
    print("  - Dropout (0.3)")
    print("  - AdamW with weight decay")
    print("  - Early stopping (patience=15)")

    # K-fold cross-validation
    all_metrics = []

    for test_fold in folds:
        # Split data
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
            epochs=50,
            lr=0.0005,
            batch_size=8,
            device=device,
            use_mixup=True,
            use_label_smoothing=True,
        )

        all_metrics.append(metrics)

    # Aggregate results
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY (V2)")
    print("=" * 60)

    macro_f1_scores = [m['macro_f1'] for m in all_metrics]
    accuracy_scores = [m['accuracy'] for m in all_metrics]

    mean_f1 = np.mean(macro_f1_scores)
    std_f1 = np.std(macro_f1_scores)
    mean_acc = np.mean(accuracy_scores)

    print(f"Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"Per-fold F1: {[f'{f:.3f}' for f in macro_f1_scores]}")
    print(f"Accuracy: {mean_acc:.4f}")

    # Compute alpha
    trivial_f1 = 1 / len(TARGETS)
    alpha = compute_alpha(mean_f1, trivial_f1)

    print(f"\nAlpha Derivation:")
    print(f"  Macro F1: {mean_f1:.4f}")
    print(f"  Trivial F1: {trivial_f1:.4f}")
    print(f"  Skill: {mean_f1 - trivial_f1:.4f}")
    print(f"  Alpha: {alpha:.4f}")
    print(f"  Evidence retained: {(1-alpha)*100:.1f}%")

    # Save summary
    summary = {
        'version': 'v2_enhanced_augmentation',
        'n_folds': len(folds),
        'n_samples': len(manifest),
        'class_distribution': dict(Counter(r['vision_target'] for r in manifest)),
        'enhancements': [
            'MixUp (alpha=0.4)',
            'RandAugment/AutoAugment',
            'Label smoothing (0.1)',
            'Dropout (0.3)',
            'AdamW with weight decay (0.01)',
            'Early stopping (patience=15)',
        ],
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
            'v1_alpha': 0.850,
            'v2_alpha': alpha,
            'improvement': 0.850 - alpha,
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: V1 vs V2")
    print("=" * 60)
    print(f"V1 alpha (basic aug):     0.850")
    print(f"V2 alpha (enhanced aug):  {alpha:.3f}")
    print(f"Improvement:              {0.850 - alpha:.3f}")


if __name__ == "__main__":
    main()
