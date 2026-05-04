"""
Evaluation script: confusion matrix, per-class metrics, Blender cross-domain test.
Usage: python src/vision/evaluate.py --model models/vision/best_resnet50.pth
"""
import argparse
import yaml
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.vision.model import load_model
from src.vision.dataset import get_eval_transforms

from torchvision import datasets


def evaluate_on_dir(model, data_dir, cfg, device):
    """Run evaluation on a directory of images."""
    transform = get_eval_transforms(cfg)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
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

    return np.array(all_labels), np.array(all_preds), dataset.classes


def plot_confusion_matrix(labels, preds, classes, save_path):
    """Generate and save confusion matrix figure."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved confusion matrix to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/vision/best_resnet50.pth")
    parser.add_argument("--config", default="configs/vision_train.yaml")
    parser.add_argument("--test-dir", default=None, help="Override test directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, num_classes=4, device=device).to(device)

    test_dir = args.test_dir or cfg["data"]["test_dir"]
    labels, preds, classes = evaluate_on_dir(model, test_dir, cfg, device)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=classes))

    save_dir = Path("results/vision/confusion_matrices")
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(labels, preds, classes, save_dir / "test_confusion.png")


if __name__ == "__main__":
    main()
