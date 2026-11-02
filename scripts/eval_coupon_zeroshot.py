#!/usr/bin/env python3
"""
Zero-shot transfer evaluation on real coupon patches.

Uses the pretrained public-corpus model to predict on coupon patches
without any fine-tuning.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.vision.model import load_model
from src.utils.frame import load_frame


REPO_ROOT = Path(__file__).parents[1]
MODEL_PATH = REPO_ROOT / "models" / "vision" / "best_resnet18_ft-layer2.pth"
CONFIG_PATH = REPO_ROOT / "configs" / "vision_train.yaml"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "vision_coupon_patches" / "manifest.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "vision" / "coupon_eval"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ZERO-SHOT TRANSFER EVALUATION ON COUPON PATCHES")
    print("=" * 70)

    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Load frame
    frame = load_frame()
    head_names = frame.head_names  # ['V1', 'V3', 'V4']

    # Map vision_target strings to head indices
    # V1 = {P}, V3 = {N,U}, V4 = {N,U,L}
    target_to_idx = {
        "{P}": 0,       # V1
        "{N,U}": 1,     # V3
        "{N,U,L}": 2,   # V4
    }

    print(f"\nHead mapping:")
    for name, focal in zip(head_names, ["{P}", "{N,U}", "{N,U,L}"]):
        print(f"  {name}: {focal} -> index {target_to_idx[focal]}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Model: {MODEL_PATH}")

    model = load_model(MODEL_PATH, num_heads=3, device=device)
    model.eval()

    # Transforms (eval, no augmentation)
    size = cfg["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["normalization"]["mean"],
                             std=cfg["normalization"]["std"]),
    ])

    # Load manifest and run inference
    patches = []
    with open(MANIFEST_PATH) as f:
        for row in csv.DictReader(f):
            patches.append(row)

    print(f"\nEvaluating {len(patches)} patches...")

    y_true = []
    y_pred = []
    y_probs = []
    results = []

    with torch.no_grad():
        for row in patches:
            img_path = REPO_ROOT / row["patch_path"]
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

            true_target = row["vision_target"]
            true_idx = target_to_idx[true_target]

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_probs.append(probs.tolist())

            results.append({
                "cell_id": row["cell_id"],
                "true_class": row["true_class"],
                "vision_target": true_target,
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "pred_head": head_names[pred_idx],
                "probs": {h: float(p) for h, p in zip(head_names, probs)},
                "correct": pred_idx == true_idx,
            })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = (y_true == y_pred).mean()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    # Trivial baseline comparison
    trivial_macro_f1 = 0.2338
    print(f"\nTrivial baseline (always {'{N,U,L}'}): {trivial_macro_f1:.4f}")
    print(f"Lift over trivial: {macro_f1 - trivial_macro_f1:+.4f}")

    # Per-class metrics
    print("\nPer-head classification report:")
    print(classification_report(y_true, y_pred, target_names=head_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"       {' '.join(f'{h:>6}' for h in head_names)}")
    for i, row in enumerate(cm):
        print(f"{head_names[i]:>6} {' '.join(f'{v:>6}' for v in row)}")

    # Per-fold analysis
    print("\n" + "-" * 50)
    print("PER-FOLD ANALYSIS")
    print("-" * 50)

    folds = {}
    for r in results:
        fold = patches[results.index(r)]["fold"]
        if fold not in folds:
            folds[fold] = {"correct": 0, "total": 0}
        folds[fold]["total"] += 1
        if r["correct"]:
            folds[fold]["correct"] += 1

    for fold in sorted(folds.keys()):
        acc = folds[fold]["correct"] / folds[fold]["total"]
        print(f"Fold {fold}: {folds[fold]['correct']}/{folds[fold]['total']} = {acc:.3f}")

    # Save detailed results
    results_path = OUTPUT_DIR / "zeroshot_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": str(MODEL_PATH.name),
            "n_patches": len(patches),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "trivial_baseline": trivial_macro_f1,
            "lift_over_trivial": macro_f1 - trivial_macro_f1,
            "confusion_matrix": cm.tolist(),
            "head_names": head_names,
            "per_patch": results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Zero-shot macro-F1: {macro_f1:.4f}")
    print(f"Trivial baseline:   {trivial_macro_f1:.4f}")
    print(f"Lift:               {macro_f1 - trivial_macro_f1:+.4f}")

    if macro_f1 <= trivial_macro_f1:
        print("\nWARNING: Model performs at or below trivial baseline!")
    else:
        # Skill calculation for alpha derivation
        skill = (macro_f1 - trivial_macro_f1) / (1 - trivial_macro_f1)
        print(f"Skill = (F1 - trivial) / (1 - trivial) = {skill:.4f}")


if __name__ == "__main__":
    main()
