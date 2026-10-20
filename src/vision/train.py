"""
Train the vision focal-element classifier.

The head count and the head->focal-element mapping come from configs/frame.yaml.
Nothing here hardcodes either. The model is a three-way softmax classifier
whose outputs map to V1 {P}, V3 {N,U}, V4 {N,U,L}.

Every macro metric is a THREE-CLASS macro. All three outputs are supported by the
data, so unlike fold 0 the macro average is a meaningful number.

Logits are persisted as PARQUET carrying the join keys -- patch_path,
source_frame, source_dataset, source_label, split, pad_fraction, box_area,
aspect_ratio -- because the fold-0 npz could not be joined to the manifest and
that blocked every geometry-conditioned analysis. npz artefacts from the
four-head run are incompatible and must not be read by this code.

Usage:
    python src/vision/train.py --config configs/vision_train.yaml
    python src/vision/train.py --arch resnet18
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402
from src.vision.dataset import compute_class_weights, get_dataloaders  # noqa: E402
from src.vision.model import build_model  # noqa: E402


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Deterministic pass; loader must be unshuffled for the logits to join."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    logits, labels_out = [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device)
        outputs = model(images)
        loss_sum += criterion(outputs, labels).item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        logits.append(outputs.float().cpu().numpy())
        labels_out.append(labels.cpu().numpy())
    return (loss_sum / total, correct / total,
            np.concatenate(logits), np.concatenate(labels_out))


def macro_f1(y_true, y_pred, head_names):
    m = compute_metrics(y_true, y_pred, class_names=head_names)
    return m["f1_macro"], m


def logits_frame(dataset, logits, labels, frame, split):
    """Logits plus every key needed to join back to the manifest."""
    rows = pd.DataFrame(dataset.rows)
    df = pd.DataFrame({
        "patch_path": rows["patch_path"],
        "source_frame": rows["source_frame"],
        "source_image": rows["source_image"],
        "source_dataset": rows["source_dataset"],
        "source_label": rows["source_label"],
        "split": split,
        "head_true": [frame.head_names[t] for t in labels],
        "head_pred": [frame.head_names[i] for i in logits.argmax(1)],
        "pad_fraction": rows["pad_fraction"].astype(float),
        "box_w": rows["box_w"].astype(float),
        "box_h": rows["box_h"].astype(float),
        "is_negative": rows["is_negative"].astype(int),
    })
    for i, h in enumerate(frame.head_names):
        df[f"logit_{h}"] = logits[:, i]
    df["box_area"] = df.box_w * df.box_h
    df["aspect_ratio"] = (np.maximum(df.box_w, df.box_h)
                          / np.maximum(np.minimum(df.box_w, df.box_h), 1e-9))
    df["correct"] = df.head_true == df.head_pred
    return df


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vision_train.yaml")
    ap.add_argument("--arch", default=None,
                    help="override model.architecture (resnet50 | resnet18)")
    ap.add_argument("--freeze-through", default=None,
                    help="override model.freeze_through (conv1 | layer1 | "
                         "layer2 | layer3 | layer4), so each run is "
                         "reproducible from the command line")
    ap.add_argument("--snapshot-epochs", type=int, nargs="*", default=[],
                    help="also save checkpoint + val/test logits at these "
                         "epochs (reporting only; never used for selection)")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="stop after this epoch WITHOUT changing the cosine "
                         "T_max, so the schedule matches a full run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None,
                    help="suffix for artefact names; defaults to the arch")
    args = ap.parse_args()

    cfg_path = (REPO_ROOT / args.config if not Path(args.config).is_absolute()
                else Path(args.config))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    arch = args.arch or cfg["model"]["architecture"]
    if args.freeze_through is not None:
        cfg["model"]["freeze_through"] = args.freeze_through
    tag = args.tag or (f"{arch}_ft-{args.freeze_through}"
                       if args.freeze_through else arch)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    frame = load_frame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print(f"VISION FOCAL-ELEMENT CLASSIFIER -- {arch}")
    print("=" * 78)
    print(f"Frame  : {frame}")
    print(f"Heads  : {frame.head_names} (n={frame.num_heads})")
    print(f"Device : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Seed   : {args.seed}")

    loaders, datasets = get_dataloaders(cfg, frame)

    freeze_through = cfg["model"].get("freeze_through")
    model = build_model(pretrained=cfg["model"]["pretrained"] == "imagenet",
                        frame=frame, architecture=arch,
                        freeze_through=freeze_through).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {arch}, frozen through {freeze_through!r} for the whole "
          f"run -- {trainable:,}/{total_p:,} parameters trainable, "
          f"{total_p - trainable:,} frozen")

    weights = None
    if cfg["training"].get("class_weights") == "inverse_frequency":
        weights = compute_class_weights(datasets["train"], device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Only the unfrozen parameters go to the optimizer; the frozen stages never
    # unfreeze, so there is no moment-continuity argument for including them.
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    scheduler = None
    if cfg["training"].get("scheduler", {}).get("type") == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs,
            eta_min=cfg["training"]["scheduler"].get("min_lr", 0.0))
        print(f"Scheduler: CosineAnnealingLR T_max={epochs} "
              f"eta_min={cfg['training']['scheduler'].get('min_lr', 0.0)}  "
              f"weight_decay={cfg['training']['weight_decay']}")

    results_dir = REPO_ROOT / "results" / "vision"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_dir = REPO_ROOT / "models" / "vision"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_dir / f"best_{tag}.pth"

    best_f1, patience, best_epoch = -1.0, 0, 0
    history, best_val = [], None

    for epoch in range(epochs):
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion,
                                          optimizer, device)
        va_loss, va_acc, va_logits, va_labels = evaluate(
            model, loaders["val"], criterion, device)
        f1, va_m = macro_f1(va_labels, va_logits.argmax(1), frame.head_names)
        va_f1_per_head = {h: float(va_m["report"][h]["f1-score"])
                          for h in frame.head_names if h in va_m["report"]}

        print(f"Epoch {epoch+1}/{epochs} lr={lr:.2e} | "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f} | macro-F1 {f1:.4f} | "
              + " ".join(f"{h} {v:.3f}" for h, v in va_f1_per_head.items()))
        history.append({"epoch": epoch + 1, "lr": lr, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_loss": va_loss,
                        "val_acc": va_acc, "val_macro_f1_three_head": f1,
                        "val_f1_per_head": va_f1_per_head})

        if scheduler is not None:
            scheduler.step()

        if epoch + 1 in args.snapshot_epochs:
            snap = f"{tag}_ep{epoch + 1}"
            torch.save(model.state_dict(), save_dir / f"snapshot_{snap}.pth")
            logits_frame(datasets["val"], va_logits, va_labels, frame, "val"
                         ).to_parquet(results_dir / f"logits_val_{snap}.parquet",
                                      index=False)
            _, _, sn_logits, sn_labels = evaluate(model, loaders["test"],
                                                  criterion, device)
            logits_frame(datasets["test"], sn_logits, sn_labels, frame, "test"
                         ).to_parquet(results_dir / f"logits_test_{snap}.parquet",
                                      index=False)
            print(f"Snapshot  : epoch {epoch + 1} -> logits_{{val,test}}_{snap}.parquet")

        if f1 > best_f1:
            best_f1, best_epoch, patience = f1, epoch + 1, 0
            torch.save(model.state_dict(), ckpt)
            best_val = logits_frame(datasets["val"], va_logits, va_labels,
                                    frame, "val")
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if args.stop_after and epoch + 1 >= args.stop_after:
            print(f"Stopping after epoch {epoch+1} (--stop-after)")
            break

    print(f"\nBest validation macro-F1 (three-class): {best_f1:.4f} "
          f"at epoch {best_epoch}")
    best_val.to_parquet(results_dir / f"logits_val_{tag}.parquet", index=False)
    print(f"Val logits : results/vision/logits_val_{tag}.parquet")

    # ---------------- test ----------------
    model.load_state_dict(torch.load(ckpt, map_location=device))
    te_loss, te_acc, te_logits, te_labels = evaluate(
        model, loaders["test"], criterion, device)
    f1_test, m = macro_f1(te_labels, te_logits.argmax(1), frame.head_names)

    test_df = logits_frame(datasets["test"], te_logits, te_labels, frame, "test")
    test_df.to_parquet(results_dir / f"logits_test_{tag}.parquet", index=False)
    print(f"Test logits: results/vision/logits_test_{tag}.parquet")

    print(f"\nTEST RESULTS -- {arch}")
    print("=" * 78)
    print(f"three-class macro-F1 {f1_test:.4f}   accuracy {te_acc:.4f}")
    print(f"{'head':<7}{'focal':<14}{'precision':>10}{'recall':>10}"
          f"{'f1':>10}{'support':>10}")
    print("-" * 78)
    rep = m["report"]
    for h in frame.head_names:
        if h in rep:
            r = rep[h]
            els = "{" + ",".join(frame.focal_element(h)) + "}"
            print(f"{h:<7}{els:<14}{r['precision']:>10.4f}{r['recall']:>10.4f}"
                  f"{r['f1-score']:>10.4f}{int(r['support']):>10}")
    print("-" * 78)
    print("confusion matrix (rows = true, cols = predicted):")
    print("        " + "".join(f"{h:>8}" for h in frame.head_names))
    for i, row in enumerate(m["confusion_matrix"]):
        print(f"{frame.head_names[i]:<8}" + "".join(f"{int(v):>8}" for v in row))

    metrics = {
        "architecture": arch,
        "macro_f1_three_head": float(f1_test),
        "accuracy": float(m["accuracy"]),
        "precision_macro_three_head": float(m["precision_macro"]),
        "recall_macro_three_head": float(m["recall_macro"]),
        "per_head": {h: rep[h] for h in frame.head_names if h in rep},
        "confusion_matrix": m["confusion_matrix"].tolist(),
        "head_names": frame.head_names,
        "focal_elements": {h: frame.focal_element(h) for h in frame.head_names},
        "best_val_macro_f1_three_head": float(best_f1),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
    }
    with open(results_dir / f"metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=1)

    import collections
    patch_counts = {}
    for split, ds in datasets.items():
        c = collections.Counter(ds.targets)
        patch_counts[split] = {h: c[i] for i, h in enumerate(frame.head_names)}

    split_summary_path = (REPO_ROOT / cfg["data"]["patches_dir"]
                          / "split_summary.json")
    split_summary = (json.loads(split_summary_path.read_text())
                     if split_summary_path.exists() else {})

    run = {
        "git_sha": git_sha(),
        "architecture": arch,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "seed": args.seed,
        "config": str(cfg_path.relative_to(REPO_ROOT)),
        "config_sha256_16": sha16(cfg_path),
        "frame_sha256_16": sha16(REPO_ROOT / "configs" / "frame.yaml"),
        "class_map_sha256_16": sha16(REPO_ROOT / "configs" / "class_map.yaml"),
        "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        "checkpoint_sha256_16": sha16(ckpt),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total_p),
        "freeze_through": freeze_through,
        "class_weights": ({h: float(weights[i])
                           for i, h in enumerate(frame.head_names)}
                          if weights is not None else None),
        "patch_counts_per_head": patch_counts,
        "source_frames_per_split": {s: len(set(ds.groups))
                                    for s, ds in datasets.items()},
        "split_approach": split_summary.get("approach"),
        "split_composition_drift_pp": (
            split_summary.get("acceptance", {}).get("max_test_drift_pp")),
        "test_macro_f1_three_head": float(f1_test),
    }
    with open(results_dir / f"run_manifest_{tag}.json", "w") as f:
        json.dump(run, f, indent=1)

    print(f"\nmetrics : results/vision/metrics_{tag}.json")
    print(f"manifest: results/vision/run_manifest_{tag}.json")


if __name__ == "__main__":
    main()
