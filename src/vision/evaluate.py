"""
Evaluate a trained vision focal-element classifier.

Primary metric is MACRO-F1: with a heavily imbalanced head distribution,
accuracy can look excellent while the rarest focal element is never predicted
at all.

Usage:
    python src/vision/evaluate.py --model models/vision/best_resnet50_fold0.pth
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402
from src.vision.dataset import get_dataloaders  # noqa: E402
from src.vision.model import load_model  # noqa: E402


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    logits, labels = [], []
    for images, y in loader:
        out = model(images.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def plot_confusion_matrix(cm, head_names, frame, save_path, normalise=True):
    """Confusion matrix labelled with focal elements, not bare head names."""
    labels = [f"{h}\n{{{','.join(frame.focal_element(h))}}}" for h in head_names]
    data = cm.astype(float)
    fmt = "d"
    if normalise:
        with np.errstate(invalid="ignore", divide="ignore"):
            data = np.nan_to_num(data / data.sum(axis=1, keepdims=True))
        fmt = ".2f"
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", vmin=0,
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"label": "row-normalised" if normalise else "count"})
    ax.set_xlabel("Predicted focal element")
    ax.set_ylabel("True focal element")
    ax.set_title("Vision modality confusion matrix")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {save_path.relative_to(REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/vision/best_resnet50_fold0.pth")
    ap.add_argument("--config", default="configs/vision_train.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    cfg_path = REPO_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    frame = load_frame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    if not model_path.exists():
        sys.exit(f"ERROR: checkpoint not found: {model_path}")

    model = load_model(model_path, device=device, frame=frame).to(device)
    loaders, datasets = get_dataloaders(cfg, frame)

    print(f"Evaluating fold {args.fold} on the '{args.split}' split "
          f"({len(datasets[args.split])} patches from "
          f"{len(set(datasets[args.split].groups))} source images)")

    logits, labels = collect(model, loaders[args.split], device)
    preds = logits.argmax(1)

    m = compute_metrics(labels, preds, class_names=frame.head_names)

    print("\nRESULTS")
    print("=" * 78)
    print(f"MACRO-F1 {m['f1_macro']:.4f}   accuracy {m['accuracy']:.4f}   "
          f"precision {m['precision_macro']:.4f}   recall {m['recall_macro']:.4f}")
    print(f"\n{'head':<8}{'focal':<12}{'precision':>10}{'recall':>10}"
          f"{'f1':>10}{'support':>10}")
    print("-" * 78)
    for h in frame.head_names:
        r = m["report"].get(h)
        if r:
            els = "{" + ",".join(frame.focal_element(h)) + "}"
            print(f"{h:<8}{els:<12}{r['precision']:>10.4f}{r['recall']:>10.4f}"
                  f"{r['f1-score']:>10.4f}{int(r['support']):>10}")

    cm = m["confusion_matrix"]
    print("\nconfusion matrix (rows = true, cols = predicted):")
    print("        " + "".join(f"{h:>8}" for h in frame.head_names))
    for i, row in enumerate(cm):
        print(f"{frame.head_names[i]:<8}" + "".join(f"{int(v):>8}" for v in row))

    out = REPO_ROOT / "results" / "vision"
    plot_confusion_matrix(
        cm, frame.head_names, frame,
        out / "confusion_matrices" / f"{args.split}_fold{args.fold}.png")

    payload = {
        "split": args.split, "fold": args.fold,
        "macro_f1": float(m["f1_macro"]), "accuracy": float(m["accuracy"]),
        "precision_macro": float(m["precision_macro"]),
        "recall_macro": float(m["recall_macro"]),
        "per_head": {h: m["report"][h] for h in frame.head_names
                     if h in m["report"]},
        "confusion_matrix": cm.tolist(),
        "focal_elements": {h: frame.focal_element(h) for h in frame.head_names},
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"eval_{args.split}_fold{args.fold}.json", "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nWrote results/vision/eval_{args.split}_fold{args.fold}.json")


if __name__ == "__main__":
    main()
