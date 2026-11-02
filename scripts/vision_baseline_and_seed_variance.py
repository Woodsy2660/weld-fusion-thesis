#!/usr/bin/env python3
"""
Vision baseline fix and seed variance analysis.

Task 0: Compute correct majority-class macro-F1 baseline
Task 1: V2 seed variance (seeds 0,1,2,3,4)
Task 2: V1 seed variance (seeds 0,1,2,3,4)
Task 3: Variance analysis
Task 4: Porosity diagnosis
"""

import sys
from pathlib import Path
import json
import csv
import random
from collections import Counter, defaultdict
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Vision targets (focal elements)
TARGETS = ['{P}', '{N,U}', '{N,U,L}']
TARGET_TO_IDX = {t: i for i, t in enumerate(TARGETS)}
IDX_TO_TARGET = {i: t for t, i in TARGET_TO_IDX.items()}


def load_manifest():
    """Load the cell patches manifest."""
    manifest_path = REPO_ROOT / "data" / "processed" / "coupon_cell_patches" / "manifest.csv"
    rows = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['fold'] = int(row['fold'])
            row['label_idx'] = TARGET_TO_IDX[row['vision_target']]
            rows.append(row)
    return rows


def compute_majority_class_baseline(manifest):
    """
    Compute majority-class macro-F1 baseline using DummyClassifier.

    This matches the ultrasonic pipeline definition:
    sklearn.dummy.DummyClassifier(strategy='most_frequent')
    """
    # Get all labels
    y_true = [r['label_idx'] for r in manifest]

    # Create dummy classifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(y_true, y_true)  # X doesn't matter for most_frequent
    y_pred = dummy.predict(y_true)

    # Compute macro F1
    baseline_f1 = f1_score(y_true, y_pred, average='macro')

    return baseline_f1, y_pred[0]  # Return baseline and majority class


def compute_alpha(macro_f1, trivial_f1):
    """Compute Shafer discounting alpha with correct baseline."""
    if macro_f1 <= trivial_f1:
        return 1.0
    skill = macro_f1 - trivial_f1
    alpha = 1 - skill / (1 - trivial_f1)
    return max(0, min(1, alpha))


def task0_fix_baseline():
    """Task 0: Fix the trivial baseline computation."""
    print("=" * 70)
    print("TASK 0: FIX TRIVIAL BASELINE")
    print("=" * 70)

    manifest = load_manifest()

    # Distribution analysis
    print("\n### Label Distribution ###")
    target_counts = Counter(r['vision_target'] for r in manifest)
    for target in TARGETS:
        print(f"  {target}: {target_counts[target]}")
    print(f"  Total: {len(manifest)}")

    # Per-fold distribution
    print("\n### Per-Fold Distribution ###")
    fold_dists = defaultdict(Counter)
    for r in manifest:
        fold_dists[r['fold']][r['vision_target']] += 1

    for fold in sorted(fold_dists.keys()):
        dist = fold_dists[fold]
        print(f"  Fold {fold}: {dict(dist)}")

    # Porosity per fold with coupon info
    print("\n### Porosity Cells by Fold and Coupon ###")
    porosity_by_fold = defaultdict(list)
    for r in manifest:
        if r['vision_target'] == '{P}':
            porosity_by_fold[r['fold']].append((r['coupon_id'], r['cell_id']))

    for fold in sorted(porosity_by_fold.keys()):
        cells = porosity_by_fold[fold]
        coupons = set(c[0] for c in cells)
        print(f"  Fold {fold}: {len(cells)} cells from coupons {sorted(coupons)}")

    # Compute baseline
    print("\n### Baseline Computation ###")
    baseline_f1, majority_class = compute_majority_class_baseline(manifest)

    print(f"  Method: sklearn.dummy.DummyClassifier(strategy='most_frequent')")
    print(f"  Majority class: {IDX_TO_TARGET[majority_class]} (idx={majority_class})")
    print(f"  Majority-class macro-F1 baseline: {baseline_f1:.4f}")
    print(f"  Old baseline (1/3): 0.3333")

    # Recompute α for existing runs
    print("\n### α Recomputation for Existing Runs ###")
    existing_runs = [
        ("V1 run 1 (no seed)", 0.433, 0.850),
        ("V1 run 2 (seed=42)", 0.358, 0.963),
        ("V2 (seed=42)", 0.457, 0.815),
    ]

    print(f"\n  {'Run':<25} {'F1':>8} {'Old α':>8} {'New α':>8} {'Δα':>8}")
    print("  " + "-" * 57)

    results = []
    for name, f1, old_alpha in existing_runs:
        new_alpha = compute_alpha(f1, baseline_f1)
        delta = new_alpha - old_alpha
        print(f"  {name:<25} {f1:>8.4f} {old_alpha:>8.4f} {new_alpha:>8.4f} {delta:>+8.4f}")
        results.append({
            'run': name,
            'macro_f1': f1,
            'old_alpha': old_alpha,
            'new_alpha': new_alpha,
            'delta': delta
        })

    return {
        'baseline_f1': baseline_f1,
        'baseline_definition': 'majority-class macro-F1 via DummyClassifier(strategy=most_frequent)',
        'label_convention': 'Overfill→{N,U}, Misaligned→{N,U}, as per manifest.csv',
        'distribution': dict(target_counts),
        'fold_distributions': {k: dict(v) for k, v in fold_dists.items()},
        'porosity_by_fold': {k: len(v) for k, v in porosity_by_fold.items()},
        'porosity_coupons_by_fold': {k: sorted(set(c[0] for c in v)) for k, v in porosity_by_fold.items()},
        'recomputed_runs': results
    }


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
        return img, row['label_idx'], row['cell_id']


def get_v1_transforms(train=True):
    """V1 configuration transforms (basic augmentation)."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def get_v2_transforms(train=True):
    """V2 configuration transforms (enhanced augmentation)."""
    if train:
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ]
        try:
            transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
        except AttributeError:
            try:
                transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
            except AttributeError:
                pass
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


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing (V2 only)."""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)
        targets_one_hot = torch.zeros_like(log_preds)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        if self.weight is not None:
            weight = self.weight[targets]
            loss = (-targets_smooth * log_preds).sum(dim=-1) * weight
        else:
            loss = (-targets_smooth * log_preds).sum(dim=-1)
        return loss.mean()


def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation (V2 only)."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def create_v1_model(num_classes=3):
    """V1 model (no dropout)."""
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_v2_model(num_classes=3):
    """V2 model (with dropout)."""
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def train_and_evaluate_fold(config, seed, test_fold, manifest, data_dir, device, baseline_f1):
    """Train and evaluate a single fold with given configuration and seed."""
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    folds = sorted(set(r['fold'] for r in manifest))
    val_fold = folds[(folds.index(test_fold) + 1) % len(folds)]
    train_folds = [f for f in folds if f not in [test_fold, val_fold]]

    test_rows = [r for r in manifest if r['fold'] == test_fold]
    val_rows = [r for r in manifest if r['fold'] == val_fold]
    train_rows = [r for r in manifest if r['fold'] in train_folds]

    # Get transforms based on config
    if config == 'V1':
        train_transform = get_v1_transforms(train=True)
        test_transform = get_v1_transforms(train=False)
        model = create_v1_model()
        use_mixup = False
        use_label_smoothing = False
        lr = 0.001
        batch_size = 16
    else:  # V2
        train_transform = get_v2_transforms(train=True)
        test_transform = get_v2_transforms(train=False)
        model = create_v2_model()
        use_mixup = True
        use_label_smoothing = True
        lr = 0.0005
        batch_size = 8

    train_dataset = CellPatchDataset(train_rows, data_dir, train_transform)
    val_dataset = CellPatchDataset(val_rows, data_dir, test_transform)
    test_dataset = CellPatchDataset(test_rows, data_dir, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)

    # Class weights
    train_dist = Counter(r['vision_target'] for r in train_rows)
    class_counts = [train_dist.get(t, 1) for t in TARGETS]
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
    weights = weights / weights.sum() * len(TARGETS)
    weights = weights.to(device)

    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    eval_criterion = nn.CrossEntropyLoss(weight=weights)

    if config == 'V2':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 30
    best_val_f1 = 0
    best_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_mixup and np.random.random() > 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels)
                outputs = model(imgs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels)

        val_f1 = f1_score(val_labels, val_preds, average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Evaluate on test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    test_preds, test_labels, test_probs = [], [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels)
            test_probs.extend(probs.cpu().numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    per_class_f1 = f1_score(test_labels, test_preds, average=None)
    per_class_precision = precision_score(test_labels, test_preds, average=None, zero_division=0)
    per_class_recall = recall_score(test_labels, test_preds, average=None, zero_division=0)

    # Porosity-specific metrics
    porosity_mask = test_labels == 0
    porosity_support = porosity_mask.sum()
    porosity_pred_count = (test_preds == 0).sum()

    alpha = compute_alpha(macro_f1, baseline_f1)

    return {
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'alpha': alpha,
        'porosity_support': int(porosity_support),
        'porosity_pred_count': int(porosity_pred_count),
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist()
    }


def run_seed_variance(config, seeds, manifest, data_dir, device, baseline_f1):
    """Run seed variance experiment for a configuration."""
    folds = sorted(set(r['fold'] for r in manifest))

    results = {}
    for seed in seeds:
        print(f"\n  Seed {seed}:", end=" ", flush=True)
        seed_results = {}
        start_time = time.time()

        for fold in folds:
            print(f"F{fold}", end=" ", flush=True)
            fold_result = train_and_evaluate_fold(
                config, seed, fold, manifest, data_dir, device, baseline_f1
            )
            seed_results[fold] = fold_result

        elapsed = time.time() - start_time

        # Compute run-level metrics
        fold_f1s = [seed_results[f]['macro_f1'] for f in folds]
        run_mean_f1 = np.mean(fold_f1s)
        run_std_f1 = np.std(fold_f1s)

        # Compute run-level alpha
        run_alpha = compute_alpha(run_mean_f1, baseline_f1)

        results[seed] = {
            'fold_results': seed_results,
            'run_mean_f1': run_mean_f1,
            'run_std_f1': run_std_f1,
            'run_alpha': run_alpha,
            'elapsed_seconds': elapsed
        }

        print(f"| F1={run_mean_f1:.3f} α={run_alpha:.3f} ({elapsed:.1f}s)")

    return results


def main():
    print("=" * 70)
    print("VISION BASELINE FIX AND SEED VARIANCE ANALYSIS")
    print("=" * 70)
    print(f"Declared seeds: 0, 1, 2, 3, 4")
    print(f"Configurations: V1 (basic), V2 (enhanced)")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    data_dir = REPO_ROOT / "data" / "processed" / "coupon_cell_patches"
    output_dir = REPO_ROOT / "results" / "vision"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()

    # Task 0: Fix baseline
    baseline_results = task0_fix_baseline()
    baseline_f1 = baseline_results['baseline_f1']

    seeds = [0, 1, 2, 3, 4]

    # Task 1: V2 seed variance
    print("\n" + "=" * 70)
    print("TASK 1: V2 SEED VARIANCE")
    print("=" * 70)

    v2_results = run_seed_variance('V2', seeds, manifest, data_dir, device, baseline_f1)

    # Task 2: V1 seed variance
    print("\n" + "=" * 70)
    print("TASK 2: V1 SEED VARIANCE")
    print("=" * 70)

    v1_results = run_seed_variance('V1', seeds, manifest, data_dir, device, baseline_f1)

    # Task 3: Variance analysis
    print("\n" + "=" * 70)
    print("TASK 3: VARIANCE ANALYSIS")
    print("=" * 70)

    # Collect run-level F1 scores
    v1_run_f1s = [v1_results[s]['run_mean_f1'] for s in seeds]
    v2_run_f1s = [v2_results[s]['run_mean_f1'] for s in seeds]

    v1_between_seed_std = np.std(v1_run_f1s)
    v2_between_seed_std = np.std(v2_run_f1s)

    # Between-fold std averaged across seeds
    v1_fold_stds = [v1_results[s]['run_std_f1'] for s in seeds]
    v2_fold_stds = [v2_results[s]['run_std_f1'] for s in seeds]
    v1_avg_fold_std = np.mean(v1_fold_stds)
    v2_avg_fold_std = np.mean(v2_fold_stds)

    # Paired comparison
    differences = [v2_run_f1s[i] - v1_run_f1s[i] for i in range(len(seeds))]
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    print(f"\n### Run-Level Macro-F1 ###")
    print(f"  V1 mean: {np.mean(v1_run_f1s):.4f}, std: {v1_between_seed_std:.4f}")
    print(f"  V2 mean: {np.mean(v2_run_f1s):.4f}, std: {v2_between_seed_std:.4f}")

    print(f"\n### Between-Seed Standard Deviation ###")
    print(f"  V1: {v1_between_seed_std:.4f}")
    print(f"  V2: {v2_between_seed_std:.4f}")

    print(f"\n### Between-Fold Standard Deviation (averaged across seeds) ###")
    print(f"  V1: {v1_avg_fold_std:.4f}")
    print(f"  V2: {v2_avg_fold_std:.4f}")

    print(f"\n### V2 - V1 Paired Difference ###")
    print(f"  Mean difference: {mean_diff:+.4f}")
    print(f"  Std of differences: {std_diff:.4f}")

    # Resolvability statement
    print(f"\n### Resolvability Statement ###")
    if abs(mean_diff) < 2 * max(v1_between_seed_std, v2_between_seed_std):
        print(f"  The V1-V2 difference ({mean_diff:+.4f}) is NOT resolvable against")
        print(f"  the between-seed variation (V1: {v1_between_seed_std:.4f}, V2: {v2_between_seed_std:.4f}).")
        print(f"  No configuration comparison is resolvable at this sample size.")
        resolvable = False
    else:
        print(f"  The V1-V2 difference ({mean_diff:+.4f}) IS resolvable against")
        print(f"  the between-seed variation.")
        resolvable = True

    # Task 4: Porosity diagnosis
    print("\n" + "=" * 70)
    print("TASK 4: POROSITY DIAGNOSIS")
    print("=" * 70)

    print("\n### Porosity Performance by Fold and Seed ###")
    print(f"\n  {'Seed':<6} {'Fold':<6} {'Support':<8} {'Pred':<6} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print("  " + "-" * 58)

    porosity_results = defaultdict(list)
    folds = [1, 2, 3, 4]

    for seed in seeds:
        for fold in folds:
            v2_fold = v2_results[seed]['fold_results'][fold]
            support = v2_fold['porosity_support']
            pred_count = v2_fold['porosity_pred_count']
            prec = v2_fold['per_class_precision'][0]
            recall = v2_fold['per_class_recall'][0]
            f1 = v2_fold['per_class_f1'][0]

            print(f"  {seed:<6} {fold:<6} {support:<8} {pred_count:<6} {prec:<8.3f} {recall:<8.3f} {f1:<8.3f}")

            porosity_results[(fold, 'support')].append(support)
            porosity_results[(fold, 'recall')].append(recall)
            porosity_results[(fold, 'f1')].append(f1)

    print("\n### Porosity F1 by Fold (across all seeds) ###")
    for fold in folds:
        f1_values = porosity_results[(fold, 'f1')]
        support = porosity_results[(fold, 'support')][0]
        coupon_info = baseline_results['porosity_coupons_by_fold'].get(fold, [])
        print(f"  Fold {fold}: F1 = {np.mean(f1_values):.3f} ± {np.std(f1_values):.3f} "
              f"(support={support}, coupons={coupon_info})")

    # Check if fold difference persists
    fold1_f1s = porosity_results[(1, 'f1')]
    fold2_f1s = porosity_results[(2, 'f1')]

    print(f"\n### Fold 1 vs Fold 2 Porosity Comparison ###")
    print(f"  Fold 1 F1 across seeds: {fold1_f1s}")
    print(f"  Fold 2 F1 across seeds: {fold2_f1s}")

    if all(f1 == 0 for f1 in fold1_f1s) and any(f1 > 0 for f1 in fold2_f1s):
        print(f"  FINDING: Fold 1 porosity failure persists across all seeds.")
        print(f"  This indicates coupon-specific appearance, not seed variance.")

    # Alpha values for Task 5
    print("\n" + "=" * 70)
    print("ALPHA VALUES FOR TASK 5")
    print("=" * 70)

    v2_alphas = [v2_results[s]['run_alpha'] for s in seeds]
    print(f"\n  V2 α values: {[f'{a:.4f}' for a in v2_alphas]}")
    print(f"  V2 α min: {min(v2_alphas):.4f}")
    print(f"  V2 α median: {np.median(v2_alphas):.4f}")
    print(f"  V2 α max: {max(v2_alphas):.4f}")

    # Save results
    all_results = {
        'baseline': baseline_results,
        'v1_results': {str(k): v for k, v in v1_results.items()},
        'v2_results': {str(k): v for k, v in v2_results.items()},
        'variance_analysis': {
            'v1_run_f1s': v1_run_f1s,
            'v2_run_f1s': v2_run_f1s,
            'v1_between_seed_std': v1_between_seed_std,
            'v2_between_seed_std': v2_between_seed_std,
            'v1_avg_fold_std': v1_avg_fold_std,
            'v2_avg_fold_std': v2_avg_fold_std,
            'mean_diff_v2_minus_v1': mean_diff,
            'std_diff': std_diff,
            'resolvable': resolvable,
        },
        'alpha_range': {
            'min': min(v2_alphas),
            'median': float(np.median(v2_alphas)),
            'max': max(v2_alphas),
        }
    }

    with open(output_dir / "seed_variance_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {output_dir / 'seed_variance_results.json'}")

    # Print 5x4 matrices
    print("\n" + "=" * 70)
    print("5×4 MACRO-F1 MATRICES")
    print("=" * 70)

    print("\n### V1 Configuration ###")
    print(f"  {'Seed':<6}", end="")
    for fold in folds:
        print(f"{'F' + str(fold):<10}", end="")
    print(f"{'Mean':<10}{'Std':<10}")

    for seed in seeds:
        print(f"  {seed:<6}", end="")
        for fold in folds:
            f1 = v1_results[seed]['fold_results'][fold]['macro_f1']
            print(f"{f1:<10.4f}", end="")
        print(f"{v1_results[seed]['run_mean_f1']:<10.4f}{v1_results[seed]['run_std_f1']:<10.4f}")

    print("\n### V2 Configuration ###")
    print(f"  {'Seed':<6}", end="")
    for fold in folds:
        print(f"{'F' + str(fold):<10}", end="")
    print(f"{'Mean':<10}{'Std':<10}")

    for seed in seeds:
        print(f"  {seed:<6}", end="")
        for fold in folds:
            f1 = v2_results[seed]['fold_results'][fold]['macro_f1']
            print(f"{f1:<10.4f}", end="")
        print(f"{v2_results[seed]['run_mean_f1']:<10.4f}{v2_results[seed]['run_std_f1']:<10.4f}")


if __name__ == "__main__":
    main()
