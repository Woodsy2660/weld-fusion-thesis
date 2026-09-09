"""Evaluate a trained vision checkpoint. Uses the shared metrics helper."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from weldfusion.evaluation.metrics import compute_metrics
from weldfusion.frame import class_order
from weldfusion.utils.config import load_yaml
from weldfusion.utils.provenance import write_manifest
from weldfusion.vision.dataset import FrameImageFolder, get_eval_transforms
from weldfusion.vision.model import load_model


def evaluate_on_dir(model, data_dir, cfg, device):
    transform = get_eval_transforms(cfg)
    dataset = FrameImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/vision/best_resnet50.pth")
    parser.add_argument("--config", default="configs/vision_train.yaml")
    parser.add_argument("--test-dir", default=None)
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device=str(device)).to(device)

    test_dir = args.test_dir or cfg["data"]["test_dir"]
    labels, preds = evaluate_on_dir(model, test_dir, cfg, device)
    names = list(class_order())
    metrics = compute_metrics(labels, preds, class_names=names)

    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["f1_macro"])
    print("Detection rate:", metrics["detection_rate"])
    print("FPR:", metrics["false_positive_rate"])
    print("Confusion matrix:\n", metrics["confusion_matrix"])

    save_dir = Path("results/tables")
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        save_dir / "vision_test_confusion.csv",
        metrics["confusion_matrix"],
        fmt="%d",
        delimiter=",",
    )
    write_manifest(
        "results/manifests/vision_eval.json",
        configs={"vision_train": cfg},
        extra={"model": args.model, "accuracy": metrics["accuracy"]},
    )


if __name__ == "__main__":
    main()
