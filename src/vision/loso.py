"""
Task D -- leave-one-source-out transfer estimate.

Within-dataset k-fold says nothing about transfer. The October RealSense coupon
images will resemble none of the three sources, so the honest estimate of how
this model behaves on unseen imagery is what happens when a whole source is
held out.

Three folds, each trained from ImageNet init on two sources and tested on the
third:

  1. hold out weld_dataset   -- test spans V1, V3, V4; full metrics
  2. hold out final_welding  -- test spans V1, V3, V4; full metrics
  3. hold out wijaya         -- DEGENERATE. Every wijaya patch is V4, so the
                                held-out test set has one class. Precision, F1
                                and any macro average are undefined; only V4
                                recall is reported, and the report says why.

Expected calibration error is reported per fold. Calibration degrades under
distribution shift, and this ECE is the empirical basis for choosing the
Shafer discount alpha in the BPA construction that follows.

Usage:
    python src/vision/loso.py --arch resnet18
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402
from src.vision.dataset import (  # noqa: E402
    PatchDataset, compute_class_weights, get_eval_transforms,
    get_train_transforms, resolve_patches_dir,
)
from src.vision.model import build_model  # noqa: E402
from src.vision.train import evaluate, logits_frame, train_one_epoch  # noqa: E402

SEED = 42
VAL_FRACTION = 0.1          # of the two retained sources
ECE_BINS = 15


def expected_calibration_error(logits, labels, n_bins=ECE_BINS):
    """
    Standard multi-class ECE over max-softmax confidence.

    Returns (ece, per-bin detail). Under distribution shift a model stays
    confident while accuracy falls, so this grows -- which is exactly the
    quantity the Shafer discount needs.
    """
    p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "confidence": None, "accuracy": None})
            continue
        c, a, w = conf[m].mean(), acc[m].mean(), m.mean()
        ece += w * abs(a - c)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "confidence": float(c), "accuracy": float(a)})
    return float(ece), bins


def write_fold_manifest(rows, holdout, seed, dest):
    """
    Rewrite the split column for one LOSO fold and save it, so PatchDataset can
    be reused unchanged.

    Frames stay whole: val is drawn at frame granularity from the two retained
    sources, never by patch.
    """
    rng = np.random.default_rng(seed)
    kept = [r for r in rows if r["source_dataset"] != holdout]
    held = [r for r in rows if r["source_dataset"] == holdout]

    frames = sorted({r["source_frame"] for r in kept})
    rng.shuffle(frames)
    n_val = max(1, int(len(frames) * VAL_FRACTION))
    val_frames = set(frames[:n_val])

    for r in kept:
        r["split"] = "val" if r["source_frame"] in val_frames else "train"
    for r in held:
        r["split"] = "test"

    out = kept + held
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    return len(kept), len(held), len(val_frames)


def build_loaders(manifest, cfg, frame, patches_root):
    dss, loaders = {}, {}
    for split in ("train", "val", "test"):
        tf = (get_train_transforms(cfg) if split == "train"
              else get_eval_transforms(cfg))
        ds = PatchDataset(manifest, split, frame.head_names, transform=tf,
                          patches_root=patches_root)
        dss[split] = ds
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=cfg["data"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=cfg["data"]["num_workers"], pin_memory=True)
    return loaders, dss


def run_fold(holdout, cfg, frame, arch, patches_root, rows, scratch, device):
    print("\n" + "=" * 78)
    print(f"LOSO FOLD -- holding out {holdout}")
    print("=" * 78)

    manifest = scratch / f"loso_{holdout}.csv"
    n_kept, n_held, n_val_frames = write_fold_manifest(
        [dict(r) for r in rows], holdout, SEED, manifest)
    loaders, dss = build_loaders(manifest, cfg, frame, patches_root)

    held_heads = Counter(dss["test"].targets)
    present = [frame.head_names[i] for i in sorted(held_heads)]
    print(f"train {len(dss['train'])} + val {len(dss['val'])} patches from the "
          f"other two sources ({n_val_frames} val frames)")
    print(f"held-out test: {len(dss['test'])} patches, heads present {present}")

    degenerate = len(present) < 2
    if degenerate:
        print(f"  DEGENERATE: the held-out set contains only {present[0]}. "
              f"Precision, F1 and macro averages are undefined here -- a "
              f"single-class test set makes every negative undefined, so only "
              f"recall is reported.")

    model = build_model(pretrained=cfg["model"]["pretrained"] == "imagenet",
                        frame=frame, architecture=arch,
                        freeze_through=cfg["model"].get("freeze_through")
                        ).to(device)
    weights = compute_class_weights(dss["train"], device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"])
    epochs = cfg["training"]["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs,
        eta_min=cfg["training"]["scheduler"].get("min_lr", 0.0))

    best_f1, patience, best_state, best_epoch = -1.0, 0, None, 0
    history = []
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion,
                                          optimizer, device)
        va_loss, va_acc, va_logits, va_labels = evaluate(
            model, loaders["val"], criterion, device)
        m = compute_metrics(va_labels, va_logits.argmax(1),
                            class_names=frame.head_names)
        f1 = m["f1_macro"]
        print(f"  epoch {epoch+1}/{epochs} train {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {va_loss:.4f}/{va_acc:.4f} | macro-F1 {f1:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_macro_f1": f1})
        scheduler.step()
        if f1 > best_f1:
            best_f1, best_epoch, patience = f1, epoch + 1, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print(f"  early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    te_loss, te_acc, te_logits, te_labels = evaluate(
        model, loaders["test"], criterion, device)
    ece, bins = expected_calibration_error(te_logits, te_labels)

    result = {
        "holdout": holdout,
        "degenerate": degenerate,
        "heads_present_in_holdout": present,
        "n_train": len(dss["train"]), "n_val": len(dss["val"]),
        "n_test": len(dss["test"]),
        "in_dist_val_macro_f1_three_head": float(best_f1),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "holdout_accuracy": float(te_acc),
        "expected_calibration_error": ece,
        "ece_bins": bins,
        "history": history,
    }

    preds = te_logits.argmax(1)
    if degenerate:
        head = present[0]
        idx = frame.head_names.index(head)
        recall = float((preds == idx).mean())
        result["holdout_recall_only"] = {head: recall}
        result["note"] = (
            f"All {len(dss['test'])} held-out patches are {head}. With one "
            f"class there are no true negatives, so precision, F1 and any "
            f"macro average are undefined. Recall is the only defined metric."
        )
        print(f"\n  held-out {head} recall: {recall:.4f}  "
              f"(accuracy {te_acc:.4f}, ECE {ece:.4f})")
        print(f"  precision / F1 / macro: UNDEFINED (single-class test set)")
    else:
        labels_present = [frame.head_names.index(h) for h in present]
        p, r, f, s = precision_recall_fscore_support(
            te_labels, preds, labels=labels_present, zero_division=0)
        result["holdout_per_head"] = {
            present[i]: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f[i]), "support": int(s[i])}
            for i in range(len(present))}
        macro = float(np.mean(f))
        result["holdout_macro_f1"] = macro
        result["holdout_macro_f1_note"] = (
            f"macro over the {len(present)} heads present in the held-out "
            f"source")
        result["transfer_gap"] = float(best_f1 - macro)
        print(f"\n  {'head':<6}{'precision':>11}{'recall':>9}{'f1':>9}"
              f"{'support':>9}")
        for i, h in enumerate(present):
            print(f"  {h:<6}{p[i]:>11.4f}{r[i]:>9.4f}{f[i]:>9.4f}{int(s[i]):>9}")
        print(f"  held-out macro-F1 {macro:.4f}   accuracy {te_acc:.4f}   "
              f"ECE {ece:.4f}")
        print(f"  TRANSFER GAP: in-distribution {best_f1:.4f} -> held-out "
              f"{macro:.4f}  ({best_f1-macro:+.4f})")

    df = logits_frame(dss["test"], te_logits, te_labels, frame, "test")
    df["holdout"] = holdout
    return result, df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/vision_train.yaml")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--sources", nargs="*",
                    default=["weld_dataset", "final_welding", "wijaya"])
    args = ap.parse_args()

    with open(REPO_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    frame = load_frame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    patches_root = resolve_patches_dir(cfg)
    with open(patches_root / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    scratch = REPO_ROOT / "results" / "vision" / "_loso_manifests"
    scratch.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"LEAVE-ONE-SOURCE-OUT -- {args.arch}")
    print("=" * 78)
    print(f"Frame: {frame}")
    counts = Counter(r["source_dataset"] for r in rows)
    print(f"sources: {dict(counts)}")

    results, frames = [], []
    for holdout in args.sources:
        res, df = run_fold(holdout, cfg, frame, args.arch, patches_root, rows,
                           scratch, device)
        results.append(res)
        frames.append(df)

    out_dir = REPO_ROOT / "results" / "vision"
    pd.concat(frames, ignore_index=True).to_parquet(
        out_dir / f"logits_loso_{args.arch}.parquet", index=False)

    payload = {"architecture": args.arch, "seed": SEED,
               "val_fraction_of_retained_sources": VAL_FRACTION,
               "ece_bins": ECE_BINS, "folds": results}
    with open(out_dir / "loso_metrics.json", "w") as f:
        json.dump(payload, f, indent=1)

    print("\n" + "=" * 78)
    print("TRANSFER SUMMARY")
    print("=" * 78)
    print(f"{'held out':<16}{'in-dist':>10}{'held-out':>11}{'gap':>9}"
          f"{'ECE':>9}{'n test':>9}")
    for r in results:
        if r["degenerate"]:
            print(f"{r['holdout']:<16}{r['in_dist_val_macro_f1_three_head']:>10.4f}"
                  f"{'n/a':>11}{'n/a':>9}{r['expected_calibration_error']:>9.4f}"
                  f"{r['n_test']:>9}   (single-class, recall only)")
        else:
            print(f"{r['holdout']:<16}{r['in_dist_val_macro_f1_three_head']:>10.4f}"
                  f"{r['holdout_macro_f1']:>11.4f}{-r['transfer_gap']:>+9.4f}"
                  f"{r['expected_calibration_error']:>9.4f}{r['n_test']:>9}")
    print(f"\n-> results/vision/loso_metrics.json")


if __name__ == "__main__":
    main()
