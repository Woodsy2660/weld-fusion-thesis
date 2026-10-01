"""
Step 0 -- rebuild fold-0 predictions with a patch identifier attached.

results/vision/logits_test_fold0.npz carries logits and labels but no patch id,
so it cannot be joined to the manifest and no geometry-conditioned analysis is
possible. This re-runs inference over val and test with a deterministic,
unshuffled loader and persists the join key alongside every logit.

The recovered test logits are asserted equal to the stored npz within 1e-5. If
they are not, the eval path is non-deterministic and every downstream number in
this diagnostic would be untrustworthy, so it stops rather than continuing.

Usage:
    python src/vision/diagnostics/predict_with_provenance.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402
from src.vision.dataset import (  # noqa: E402
    PatchDataset, get_eval_transforms, resolve_patches_dir,
)
from src.vision.model import load_model  # noqa: E402

TOLERANCE = 1e-5


@torch.no_grad()
def infer(model, dataset, device, batch_size=64):
    """Deterministic pass: shuffle=False, workers=0, order == manifest order."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    logits = []
    model.eval()
    for images, _ in loader:
        logits.append(model(images.to(device)).float().cpu().numpy())
    return np.concatenate(logits)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/vision_train.yaml")
    ap.add_argument("--checkpoint",
                    default="models/vision/best_resnet50_fold0.pth")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-dir", default="results/diagnostics")
    args = ap.parse_args()

    with open(REPO_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    frame = load_frame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patches = resolve_patches_dir(cfg)
    manifest_path = patches / "manifest.csv"

    print("=" * 78)
    print("STEP 0 -- predictions with provenance")
    print("=" * 78)
    print(f"Frame      : {frame}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")

    model = load_model(REPO_ROOT / args.checkpoint, device=device,
                       frame=frame).to(device)

    manifest = pd.read_csv(manifest_path)
    frames = []

    for split in ("val", "test"):
        ds = PatchDataset(manifest_path, split, frame.head_names,
                          transform=get_eval_transforms(cfg),
                          patches_root=patches)
        print(f"\n{split}: {len(ds)} patches")
        logits = infer(model, ds, device, cfg["data"]["batch_size"])

        rows = pd.DataFrame(ds.rows)
        assert len(rows) == len(logits), "row/logit length mismatch"

        out = pd.DataFrame({
            "patch_path": rows["patch_path"],
            "source_frame": rows["source_frame"],
            "source_image": rows["source_image"],
            "source_dataset": rows["source_dataset"],
            "source_label": rows["source_label"],
            "split": split,
            "head_true": [frame.head_names[t] for t in ds.targets],
            "head_pred": [frame.head_names[i] for i in logits.argmax(1)],
            "is_negative": rows["is_negative"].astype(int),
            "pad_fraction": rows["pad_fraction"].astype(float),
            "box_w": rows["box_w"].astype(float),
            "box_h": rows["box_h"].astype(float),
        })
        for i, h in enumerate(frame.head_names):
            out[f"logit_{h}"] = logits[:, i]

        out["box_area"] = out["box_w"] * out["box_h"]
        # Long side over short side, so 1.0 is square regardless of orientation
        # -- a 10x1 toe line and a 1x10 one are the same shape problem.
        out["aspect_ratio"] = (
            np.maximum(out["box_w"], out["box_h"])
            / np.maximum(np.minimum(out["box_w"], out["box_h"]), 1e-9)
        )
        frames.append(out)

        # ---- determinism check against the stored artefact ----
        stored_path = (REPO_ROOT / "results" / "vision"
                       / (f"logits_test_fold{args.fold}.npz" if split == "test"
                          else f"logits_fold{args.fold}.npz"))
        if stored_path.exists():
            stored = np.load(stored_path)
            ref = stored["logits"]
            if ref.shape != logits.shape:
                sys.exit(
                    f"FATAL: {stored_path.name} has shape {ref.shape} but this "
                    f"run produced {logits.shape}. The eval path has changed."
                )
            delta = np.abs(ref - logits).max()
            label_delta = int((stored["labels"] != np.array(ds.targets)).sum())
            print(f"  vs {stored_path.name}: max|delta| = {delta:.3e}, "
                  f"label mismatches = {label_delta}")
            if split == "test":
                if delta > TOLERANCE:
                    sys.exit(
                        f"FATAL: recovered test logits differ from "
                        f"{stored_path.name} by {delta:.3e} (> {TOLERANCE}). "
                        f"Something in the eval path is non-deterministic; every "
                        f"downstream number in this diagnostic would be unsafe. "
                        f"Stopping."
                    )
                if label_delta:
                    sys.exit(f"FATAL: {label_delta} label mismatches vs stored "
                             f"npz -- the manifest order has changed.")
                print(f"  OK: within {TOLERANCE}")
            else:
                # The val npz was written at the BEST epoch, not the final one,
                # so it is only expected to match when the checkpoint is that
                # same epoch. Reported, not asserted.
                print(f"  (val npz is written at the best epoch; informational)")

    df = pd.concat(frames, ignore_index=True)
    df["correct"] = df["head_true"] == df["head_pred"]

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / f"predictions_fold{args.fold}.parquet"
    df.to_parquet(pq, index=False)
    size_mb = pq.stat().st_size / 1e6
    print(f"\nWrote {pq.relative_to(REPO_ROOT)}  ({size_mb:.2f} MB)")
    if size_mb < 50:
        csv = out_dir / f"predictions_fold{args.fold}.csv"
        df.to_csv(csv, index=False)
        print(f"Wrote {csv.relative_to(REPO_ROOT)}  "
              f"({csv.stat().st_size/1e6:.2f} MB)")

    print(f"\nRows: {len(df)}  "
          f"(val {(df.split=='val').sum()}, test {(df.split=='test').sum()})")
    print(f"Test accuracy recomputed: "
          f"{df[df.split=='test'].correct.mean():.4f}")


if __name__ == "__main__":
    main()
