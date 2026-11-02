# Vision Module Methodology Verification

## Run Identity

**Reported results**: `results/vision/vision_canonical_results.json`
**Training script**: `scripts/vision_baseline_and_seed_variance.py`
**Data manifest**: `data/processed/coupon_cell_patches/manifest.csv`
**Seeds declared**: 0, 1, 2, 3, 4 (all reported)
**Configurations**: V1 (basic), V2 (enhanced)

---

## Verification Table

| # | Item | Status | Actual Value | Evidence | Quote |
|---|------|--------|--------------|----------|-------|
| 0a | Input: letterboxing | **MATCH** | Aspect-ratio preserving resize with black padding | `scripts/extract_cell_roi_patches.py:130-147` | `"# Resize to target size (preserving aspect ratio with padding)"` |
| 0b | Input: 224×224 resize | **MATCH** | target_size=224 | `scripts/extract_cell_roi_patches.py:113` | `"def extract_cell_patch(img, bounds, target_size=224):"` |
| 0c | Input: ImageNet normalization | **MATCH** | [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] | `scripts/vision_baseline_and_seed_variance.py:194` | `"transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])"` |
| 1a | Network: architecture | **MISMATCH** | ResNet-18, NOT ResNet-50 | `scripts/vision_baseline_and_seed_variance.py:269` | `"model = models.resnet18(weights='IMAGENET1K_V1')"` |
| 1b | Network: pretrained weights | **MATCH** | ImageNet-1K V1 | `scripts/vision_baseline_and_seed_variance.py:269` | `"weights='IMAGENET1K_V1'"` |
| 1c | Network: trainable layers | **PARTIAL** | All layers trainable from epoch 0 (no freeze/unfreeze) | `scripts/vision_baseline_and_seed_variance.py:284-348` | No `requires_grad=False` anywhere |
| 2a | Training: optimizer V1 | **MATCH** | Adam, lr=0.001 | `scripts/vision_baseline_and_seed_variance.py:347` | `"optimizer = optim.Adam(model.parameters(), lr=lr)"` |
| 2b | Training: optimizer V2 | **MATCH** | AdamW, lr=0.0005, weight_decay=0.01 | `scripts/vision_baseline_and_seed_variance.py:344` | `"optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)"` |
| 2c | Training: scheduler V1 | **MATCH** | StepLR, step_size=10, gamma=0.5 | `scripts/vision_baseline_and_seed_variance.py:348` | `"scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)"` |
| 2d | Training: scheduler V2 | **MATCH** | CosineAnnealingWarmRestarts, T_0=10, T_mult=2 | `scripts/vision_baseline_and_seed_variance.py:345` | `"scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)"` |
| 2e | Training: epochs | **MATCH** | 30 with early stopping (patience=15) | `scripts/vision_baseline_and_seed_variance.py:350,353` | `"epochs = 30"`, `"patience = 15"` |
| 2f | Training: batch size V1 | **MATCH** | 16 | `scripts/vision_baseline_and_seed_variance.py:309` | `"batch_size = 16"` |
| 2g | Training: batch size V2 | **MATCH** | 8 | `scripts/vision_baseline_and_seed_variance.py:317` | `"batch_size = 8"` |
| 2h | Training: class weights | **MATCH** | Inverse-frequency, normalized | `scripts/vision_baseline_and_seed_variance.py:330-334` | `"weights = torch.tensor([1.0 / c for c in class_counts], ...)"` |
| 2i | Training: MixUp (V2 only) | **MATCH** | alpha=0.4, 50% probability | `scripts/vision_baseline_and_seed_variance.py:254-264,362` | `"def mixup_data(x, y, alpha=0.4):"` |
| 2j | Training: label smoothing (V2) | **MATCH** | smoothing=0.1 | `scripts/vision_baseline_and_seed_variance.py:235,337` | `"def __init__(self, smoothing=0.1, weight=None):"` |
| 2k | Training: dropout (V2) | **MATCH** | 0.3 on fc layer | `scripts/vision_baseline_and_seed_variance.py:277-280` | `"nn.Dropout(0.3)"` |
| 2l | Training: RandAugment (V2) | **MATCH** | num_ops=2, magnitude=9 | `scripts/vision_baseline_and_seed_variance.py:214` | `"transforms.RandAugment(num_ops=2, magnitude=9)"` |
| 3a | Calibration: temperature scaling | **NOT DONE** | null in results | `results/vision/vision_canonical_results.json:97-99` | `"temperature_per_seed": null` |
| 3b | Calibration: ECE | **NOT DONE** | null in results | `results/vision/vision_canonical_results.json:97-98` | `"ece_pre_calibration": null, "ece_post_calibration": null"` |
| 4a | Assignment: alpha formula | **MATCH** | α = 1 − (F1 − T)/(1 − T) | `scripts/vision_baseline_and_seed_variance.py:83-87` | `"alpha = 1 - (f1 - baseline_f1) / (1 - baseline_f1)"` |
| 4b | Assignment: baseline | **MATCH** | 0.2338 majority-class macro-F1 | `results/vision/vision_canonical_results.json:12-13` | `"value": 0.2338, "definition": "majority-class macro-F1"` |
| 5a | Verification: leakage check | **N/A** | Not applicable to coupon data | — | Coupon data uses fold-by-coupon, no sequence leakage possible |
| 5b | Verification: LOSO | **N/A** | 4-fold CV by coupon, not LOSO | — | `scripts/vision_baseline_and_seed_variance.py:293-295` groups by coupon |
| 5c | Verification: cosine audit | **NOT DONE** | No nearest-neighbor audit in seed variance pipeline | — | — |

---

## Critical Issues

### 1. Architecture Mismatch (Item 1a)

**Claim**: ResNet-50
**Actual**: ResNet-18

**Evidence**:
```python
# scripts/vision_baseline_and_seed_variance.py:269
model = models.resnet18(weights='IMAGENET1K_V1')
```

**Impact**: Thesis text must be updated to say ResNet-18.

---

### 2. No Backbone Freeze/Unfreeze (Item 1c)

**Claim** (per CLAUDE.md): "5-epoch backbone freeze then unfreeze"
**Actual**: All parameters trainable from epoch 0

**Evidence**: No `requires_grad=False` statements in training loop. The optimizer is initialized with all parameters:
```python
# scripts/vision_baseline_and_seed_variance.py:344
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
```

**Impact**: If thesis claims freeze/unfreeze, update to reflect actual training.

---

### 3. Calibration Not Implemented (Items 3a, 3b)

**Claim**: Temperature scaling fitted, ECE reported
**Actual**: Neither implemented in seed variance pipeline

**Evidence**:
```json
// results/vision/vision_canonical_results.json:96-102
"calibration": {
    "note": "ECE computation not implemented in current pipeline",
    "ece_pre_calibration": null,
    "ece_post_calibration": null,
    "temperature_per_seed": null
}
```

**Impact**:
- Thesis cannot claim calibrated probabilities for the reported runs
- Alpha values use raw softmax outputs, not temperature-scaled posteriors
- ECE figures cannot be reported for these runs

---

## Confirmed Matches

1. **Input pipeline**: Letterbox to 224×224 with ImageNet normalization ✓
2. **Pretrained weights**: ImageNet-1K V1 ✓
3. **Optimizer and scheduler**: V1 Adam/StepLR, V2 AdamW/CosineAnnealing ✓
4. **Regularization**: V2 uses MixUp, label smoothing, dropout, RandAugment ✓
5. **Class weights**: Inverse-frequency from training split ✓
6. **Alpha formula**: Shafer discounting with majority-class baseline ✓
7. **Cross-validation**: 4-fold grouped by coupon ✓

---

## Text Changes Required

### 1. Architecture
**Old**: "ResNet-50"
**New**: "ResNet-18"

### 2. Training Protocol (if claimed)
**Old**: "Backbone frozen for 5 epochs, then unfrozen"
**New**: "All layers trainable throughout training" (or omit this detail)

### 3. Calibration
**Old**: "Temperature scaling fitted on validation fold"
**New**: "Temperature scaling not applied; raw softmax posteriors used for BPA construction"

### 4. ECE
**Old**: Any ECE value
**New**: "ECE not computed for these runs" or implement ECE computation

---

## Appendix: Hyperparameter Table

### V1 Configuration (Basic Augmentation)

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 |
| Pretrained weights | ImageNet-1K V1 |
| Input size | 224×224 (letterbox) |
| Normalization | ImageNet mean/std |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Scheduler | StepLR (step=10, gamma=0.5) |
| Batch size | 16 |
| Epochs | 30 (early stopping patience=15) |
| Augmentation | HorizontalFlip, VerticalFlip, Rotation(15°), ColorJitter |
| Class weights | Inverse-frequency, normalized |

### V2 Configuration (Enhanced Augmentation)

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 |
| Pretrained weights | ImageNet-1K V1 |
| Input size | 224×224 (letterbox) |
| Normalization | ImageNet mean/std |
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Weight decay | 0.01 |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| Batch size | 8 |
| Epochs | 30 (early stopping patience=15) |
| Augmentation | V1 + RandomAffine, RandAugment(2,9), RandomErasing |
| MixUp | α=0.4, p=0.5 |
| Label smoothing | 0.1 |
| Dropout | 0.3 (fc layer) |
| Class weights | Inverse-frequency, normalized |

---

## Provenance

- **Verification date**: 2026-10-29
- **Training script**: `scripts/vision_baseline_and_seed_variance.py`
- **Results file**: `results/vision/vision_canonical_results.json`
- **Data manifest**: `data/processed/coupon_cell_patches/manifest.csv`
