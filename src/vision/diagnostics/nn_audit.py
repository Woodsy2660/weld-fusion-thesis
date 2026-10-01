"""
Task F3 -- embedding nearest-neighbour audit.

Sequence stems catch near-duplicates that share a filename convention. This
catches the ones that do not: two frames of the same weld exported under
unrelated names land close together in a trained network's feature space even
when phash and filename identity both miss them.

For every test patch, find its nearest TRAIN patch by cosine distance. Compare
that distribution against a test->test control (nearest neighbour excluding
self), because some closeness is expected between any two crops of similar
welds -- what matters is whether test sits closer to train than to itself.

Then tile the 50 closest cross-split pairs. That contact sheet is the decisive
artefact: if those pairs are visibly the same weld a frame apart, the split is
leaking and no threshold argument is needed.

Usage:
    python src/vision/diagnostics/nn_audit.py --arch resnet50
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402
from src.vision.dataset import (  # noqa: E402
    PatchDataset, get_eval_transforms, resolve_patches_dir,
)
from src.vision.model import load_model  # noqa: E402

SEED = 42
TRAIN_SUBSAMPLE = 10000
NEAR_THRESHOLD = 0.05        # cosine distance below which two patches are
                             # treated as near-duplicates
T_NEAR_FRACTION = 0.02       # F5 threshold on the fraction below it
N_PAIRS = 50
TILE = 150


@torch.no_grad()
def features(model, dataset, device, batch_size=128):
    """L2-normalised penultimate activations, so dot product == cosine."""
    feats = []

    def hook(_m, _i, o):
        feats.append(o.detach().flatten(1).float().cpu())

    handle = model.avgpool.register_forward_hook(hook)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
    model.eval()
    for images, _ in loader:
        model(images.to(device, non_blocking=True))
    handle.remove()
    x = torch.cat(feats)
    return torch.nn.functional.normalize(x, dim=1)


def nearest(query, bank, device, exclude_self=False, chunk=512):
    """Nearest neighbour of each query row in bank, by cosine distance."""
    q, b = query.to(device), bank.to(device)
    best_d, best_i = [], []
    for s in range(0, len(q), chunk):
        sim = q[s:s + chunk] @ b.T
        if exclude_self:
            for r in range(sim.shape[0]):
                sim[r, s + r] = -2.0
        v, i = sim.max(dim=1)
        best_d.append((1.0 - v).cpu())
        best_i.append(i.cpu())
    return torch.cat(best_d).numpy(), torch.cat(best_i).numpy()


def plot_distributions(d_train, d_test, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    bins = np.linspace(0, max(0.6, float(np.percentile(
        np.concatenate([d_train, d_test]), 99))), 60)

    axes[0].hist(d_train, bins=bins, density=True, histtype="step",
                 linewidth=2, color="#A32E28", label="test -> nearest TRAIN")
    axes[0].hist(d_test, bins=bins, density=True, histtype="step",
                 linewidth=2, color="#2C5F8A",
                 label="test -> nearest TEST (control)")
    axes[0].axvline(NEAR_THRESHOLD, color="#333", linestyle=":", linewidth=1.2)
    axes[0].text(NEAR_THRESHOLD, axes[0].get_ylim()[1] * 0.92,
                 f"  {NEAR_THRESHOLD}", fontsize=9, color="#333")
    axes[0].set_xlabel("cosine distance to nearest neighbour")
    axes[0].set_ylabel("density")
    axes[0].set_title("Nearest-neighbour distance, test patches")
    axes[0].legend(fontsize=9)

    for d, c, lab in ((d_train, "#A32E28", "test -> train"),
                      (d_test, "#2C5F8A", "test -> test")):
        axes[1].plot(np.sort(d), np.linspace(0, 1, len(d)), linewidth=2,
                     color=c, label=lab)
    axes[1].axvline(NEAR_THRESHOLD, color="#333", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("cosine distance")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title("Empirical CDF")
    axes[1].set_xlim(0, bins[-1])
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pair_sheet(pairs, patches_root, out_path):
    """50 closest test<->train pairs, side by side, fully annotated."""
    cols, rows = 5, (len(pairs) + 4) // 5
    cell_w, cell_h = TILE * 2 + 12, TILE + 46
    header = 40
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + header),
                      (248, 248, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), f"{len(pairs)} closest test<->train pairs by cosine "
                       f"distance in penultimate feature space",
              fill=(15, 15, 20))
    draw.text((10, 23), "left = TEST patch, right = nearest TRAIN patch. "
                        "If these are the same weld a frame apart, the split "
                        "is leaking.", fill=(90, 95, 105))

    for i, p in enumerate(pairs):
        cx, cy = (i % cols) * cell_w, header + (i // cols) * cell_h
        for j, key in enumerate(("test_patch", "train_patch")):
            try:
                with Image.open(patches_root / p[key]) as im:
                    sheet.paste(im.convert("RGB").resize((TILE, TILE)),
                                (cx + j * (TILE + 12), cy))
            except Exception:
                draw.rectangle([cx + j * (TILE + 12), cy,
                                cx + j * (TILE + 12) + TILE, cy + TILE],
                               fill=(215, 215, 220))
        same = p["test_dataset"] == p["train_dataset"]
        draw.text((cx + 3, cy + TILE + 2),
                  f"d={p['cosine_distance']:.4f}"
                  f"{'  SAME SOURCE' if same else '  cross-source'}",
                  fill=(163, 46, 40) if p["cosine_distance"] < NEAR_THRESHOLD
                  else (45, 50, 58))
        draw.text((cx + 3, cy + TILE + 16),
                  f"test  {p['test_dataset']}/{p['test_label']} [{p['test_head']}]"[:52],
                  fill=(45, 50, 58))
        draw.text((cx + 3, cy + TILE + 30),
                  f"train {p['train_dataset']}/{p['train_label']} [{p['train_head']}]"[:52],
                  fill=(45, 50, 58))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", default="results/diagnostics")
    args = ap.parse_args()

    with open(REPO_ROOT / "configs" / "vision_train.yaml") as f:
        cfg = yaml.safe_load(f)
    frame = load_frame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patches_root = resolve_patches_dir(cfg)
    manifest = patches_root / "manifest.csv"
    ckpt = (Path(args.checkpoint) if args.checkpoint
            else REPO_ROOT / "models" / "vision" / f"best_{args.arch}.pth")
    if not ckpt.exists():
        sys.exit(f"ERROR: {ckpt} not found -- has Task C finished?")

    print("=" * 78)
    print(f"F3 -- embedding nearest-neighbour audit ({args.arch})")
    print("=" * 78)

    model = load_model(ckpt, device=device, frame=frame,
                       architecture=args.arch).to(device)

    test_ds = PatchDataset(manifest, "test", frame.head_names,
                           transform=get_eval_transforms(cfg),
                           patches_root=patches_root)
    train_ds = PatchDataset(manifest, "train", frame.head_names,
                            transform=get_eval_transforms(cfg),
                            patches_root=patches_root)

    rng = np.random.default_rng(SEED)
    n_train = len(train_ds)
    if n_train > TRAIN_SUBSAMPLE:
        idx = np.sort(rng.choice(n_train, TRAIN_SUBSAMPLE, replace=False))
        train_view = torch.utils.data.Subset(train_ds, idx.tolist())
        print(f"train subsampled to {TRAIN_SUBSAMPLE} of {n_train} "
              f"(seed {SEED})")
    else:
        idx = np.arange(n_train)
        train_view = train_ds

    print(f"extracting features: test {len(test_ds)}, train {len(train_view)}")
    f_test = features(model, test_ds, device)
    f_train = features(model, train_view, device)

    d_train, i_train = nearest(f_test, f_train, device)
    d_test, _ = nearest(f_test, f_test, device, exclude_self=True)

    frac = float((d_train < NEAR_THRESHOLD).mean())
    frac_ctrl = float((d_test < NEAR_THRESHOLD).mean())

    # __negative__ patches are annotation-free background. Two blank crops are
    # near-identical in feature space whether or not they come from the same
    # weld, so they inflate this statistic without evidencing leakage. Report
    # the annotated-defect subset separately -- that is the number that speaks
    # to whether real defect imagery spans the split.
    is_neg = np.array([r["source_label"] == "__negative__"
                       for r in test_ds.rows])
    frac_pos = (float((d_train[~is_neg] < NEAR_THRESHOLD).mean())
                if (~is_neg).any() else 0.0)
    frac_neg = (float((d_train[is_neg] < NEAR_THRESHOLD).mean())
                if is_neg.any() else 0.0)

    def describe(x):
        return {"mean": float(x.mean()), "median": float(np.median(x)),
                "p01": float(np.percentile(x, 1)),
                "p05": float(np.percentile(x, 5)),
                "p25": float(np.percentile(x, 25)),
                "p75": float(np.percentile(x, 75)),
                "min": float(x.min()), "max": float(x.max())}

    print(f"\n{'':<22}{'mean':>9}{'median':>9}{'p01':>9}{'p05':>9}{'min':>9}")
    for lab, x in (("test -> train", d_train), ("test -> test (ctrl)", d_test)):
        s = describe(x)
        print(f"{lab:<22}{s['mean']:>9.4f}{s['median']:>9.4f}"
              f"{s['p01']:>9.4f}{s['p05']:>9.4f}{s['min']:>9.4f}")

    print(f"\ntest patches with nearest-TRAIN cosine distance < "
          f"{NEAR_THRESHOLD}: {100*frac:.2f}%   <- headline")
    print(f"control, nearest-TEST below the same threshold : {100*frac_ctrl:.2f}%")
    print(f"  excluding __negative__ (annotated defects only): "
          f"{100*frac_pos:.2f}%  (n={int((~is_neg).sum())})")
    print(f"  __negative__ patches only                      : "
          f"{100*frac_neg:.2f}%  (n={int(is_neg.sum())})")
    verdict = "CONFIRMED" if frac >= T_NEAR_FRACTION else "CLEAN"
    print(f"threshold {100*T_NEAR_FRACTION:.0f}%  ->  {verdict}")

    plot_distributions(d_train, d_test,
                       REPO_ROOT / args.out_dir / "nn_distance_test_vs_train.png")
    print(f"-> {args.out_dir}/nn_distance_test_vs_train.png")

    # ---- closest pairs ----
    test_rows = pd.DataFrame(test_ds.rows)
    train_rows = pd.DataFrame(train_ds.rows).iloc[idx].reset_index(drop=True)
    order = np.argsort(d_train)[:N_PAIRS]
    pairs = []
    for t in order:
        tr = int(i_train[t])
        pairs.append({
            "cosine_distance": float(d_train[t]),
            "test_patch": test_rows.patch_path.iat[int(t)],
            "train_patch": train_rows.patch_path.iat[tr],
            "test_dataset": test_rows.source_dataset.iat[int(t)],
            "train_dataset": train_rows.source_dataset.iat[tr],
            "test_label": test_rows.source_label.iat[int(t)],
            "train_label": train_rows.source_label.iat[tr],
            "test_head": test_rows["head"].iat[int(t)],
            "train_head": train_rows["head"].iat[tr],
            "test_source_frame": test_rows.source_frame.iat[int(t)],
            "train_source_frame": train_rows.source_frame.iat[tr],
        })
    pair_sheet(pairs, patches_root,
               REPO_ROOT / "results" / "review" / "nn_pairs_test_train.png")
    print(f"-> results/review/nn_pairs_test_train.png  ({len(pairs)} pairs, "
          f"closest d={pairs[0]['cosine_distance']:.4f})")

    same_src = sum(1 for p in pairs if p["test_dataset"] == p["train_dataset"])
    same_head = sum(1 for p in pairs if p["test_head"] == p["train_head"])
    same_label = sum(1 for p in pairs if p["test_label"] == p["train_label"])
    print(f"   of the {len(pairs)} closest pairs: {same_src} share a source "
          f"dataset, {same_label} share a source label, {same_head} share a head")

    payload = {
        "architecture": args.arch, "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        "seed": SEED, "train_subsample": int(len(train_view)),
        "train_total": int(n_train), "n_test": int(len(test_ds)),
        "near_threshold": NEAR_THRESHOLD,
        "test_to_train": describe(d_train),
        "test_to_test_control": describe(d_test),
        "fraction_below_threshold": frac,
        "fraction_below_threshold_control": frac_ctrl,
        "fraction_below_threshold_annotated_defects_only": frac_pos,
        "fraction_below_threshold_negatives_only": frac_neg,
        "n_annotated_defect_test_patches": int((~is_neg).sum()),
        "n_negative_test_patches": int(is_neg.sum()),
        "f5_threshold": T_NEAR_FRACTION,
        "verdict": verdict,
        "closest_pairs": pairs,
        "closest_pairs_summary": {
            "same_source_dataset": same_src, "same_source_label": same_label,
            "same_head": same_head},
    }
    out = REPO_ROOT / args.out_dir / "nn_audit.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
