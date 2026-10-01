"""
Step 6 -- evidence pack for the three provisional class mappings.

Produces evidence only. configs/class_map.yaml is not touched; the human makes
the decisions.

  1. Contact sheets, 64 instances each, stratified by pad_fraction quartile and
     aspect quartile -- so the elongated and heavily padded cases are visible,
     which uniform sampling would bury. Each cell shows the original crop with
     the annotation box overlaid AND the letterboxed 224x224 patch exactly as
     the network receives it. The second is the one that matters: it shows
     whether the feature survives padding.
  2. Per-source-label recall inside each head.
  3. Penultimate-layer embedding of V3 test patches, PCA -> UMAP, coloured by
     source_label, with a silhouette score for discontinuity against the rest.
  4. Label inventory per source data.yaml with post-dedup counts, to test
     whether discontinuity is that dataset's catch-all class.

Usage:
    python src/vision/diagnostics/mapping_evidence.py
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
from src.vision.convert_and_remap import find_dir, load_yaml_names  # noqa: E402
from src.vision.dataset import (  # noqa: E402
    PatchDataset, get_eval_transforms, resolve_patches_dir,
)
from src.vision.model import load_model  # noqa: E402

SEED = 42
N_CELLS = 64
GRID = 8
IMG = 132          # each of the two images in a cell
TEXT_H = 34
CELL_W = IMG * 2 + 10
CELL_H = IMG + TEXT_H

TARGETS = [
    ("weld_dataset", "discontinuity"),          # provisional -> V3
    ("final_welding", "cluster of porosity"),   # provisional -> V1
    ("final_welding", "Crack"),                 # provisional -> V2
    ("final_welding", "Overlap"),               # control, V3
    ("final_welding", "Undercut"),              # control, V3
    ("weld_dataset", "stain"),                  # control, V4
    ("weld_dataset", "deposit"),                # control, V4
]


# ---------------------------------------------------------------------------
# 1. Contact sheets
# ---------------------------------------------------------------------------
def stratified_sample(df, n=N_CELLS, seed=SEED):
    """
    Sample across the pad x aspect quartile grid rather than uniformly, so the
    elongated and heavily padded instances are guaranteed representation.
    """
    rng = np.random.default_rng(seed)
    d = df.copy()
    for col, name in (("pad_fraction", "pq"), ("aspect_ratio", "aq")):
        try:
            d[name] = pd.qcut(d[col], 4, labels=False, duplicates="drop")
        except ValueError:
            d[name] = 0

    strata = list(d.groupby(["pq", "aq"]))
    if not strata:
        return d.head(n)
    per = max(1, n // len(strata))

    picked = []
    for _, sub in strata:
        k = min(per, len(sub))
        picked.append(sub.iloc[rng.choice(len(sub), k, replace=False)])
    out = pd.concat(picked)

    if len(out) < n:                       # top up from whatever is left
        rest = d.drop(out.index, errors="ignore")
        if len(rest):
            k = min(n - len(out), len(rest))
            out = pd.concat([out, rest.iloc[rng.choice(len(rest), k, replace=False)]])
    return out.head(n).sort_values(["pq", "aq"])


def build_contact_sheet(rows, dataset, label, head, patches_root, out_path):
    from src.vision.convert_and_remap import CONTEXT_MARGIN, expand_and_clip

    n = len(rows)
    cols = GRID
    nrows = (n + cols - 1) // cols
    header = 42
    sheet = Image.new("RGB", (cols * CELL_W, nrows * CELL_H + header),
                      (248, 248, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8),
              f"{dataset} / {label}  ->  {head}    {n} instances, stratified by "
              f"pad_fraction x aspect quartile", fill=(15, 15, 20))
    draw.text((10, 24),
              "left: original crop, annotation box in red   |   right: "
              "letterboxed 224x224 as the network receives it",
              fill=(90, 95, 105))

    for i, (_, r) in enumerate(rows.iterrows()):
        cx = (i % cols) * CELL_W
        cy = header + (i // cols) * CELL_H

        # --- left: original crop with the box drawn on it ---
        try:
            with Image.open(r.source_image) as im:
                im = im.convert("RGB")
                W, H = im.size
                bx, by, bw, bh = (float(r.box_x), float(r.box_y),
                                  float(r.box_w), float(r.box_h))
                xc, yc = (bx + bw / 2) / W, (by + bh / 2) / H
                x0, y0, x1, y1 = expand_and_clip(xc, yc, bw / W, bh / H, W, H,
                                                 CONTEXT_MARGIN)
                crop = im.crop((x0, y0, x1, y1))
                sc = min(IMG / max(1, crop.width), IMG / max(1, crop.height))
                nw, nh = max(1, int(crop.width * sc)), max(1, int(crop.height * sc))
                thumb = crop.resize((nw, nh), Image.BILINEAR)
                canvas = Image.new("RGB", (IMG, IMG), (228, 230, 234))
                canvas.paste(thumb, ((IMG - nw) // 2, (IMG - nh) // 2))
                d2 = ImageDraw.Draw(canvas)
                d2.rectangle(
                    [(IMG - nw) // 2 + (bx - x0) * sc,
                     (IMG - nh) // 2 + (by - y0) * sc,
                     (IMG - nw) // 2 + (bx - x0 + bw) * sc,
                     (IMG - nh) // 2 + (by - y0 + bh) * sc],
                    outline=(200, 40, 34), width=2)
                sheet.paste(canvas, (cx, cy))
        except Exception:
            draw.rectangle([cx, cy, cx + IMG, cy + IMG], fill=(215, 215, 220))

        # --- right: the actual patch the network sees ---
        try:
            with Image.open(patches_root / r.patch_path) as pm:
                sheet.paste(pm.convert("RGB").resize((IMG, IMG)), (cx + IMG + 10, cy))
        except Exception:
            draw.rectangle([cx + IMG + 10, cy, cx + 2 * IMG + 10, cy + IMG],
                           fill=(215, 215, 220))

        verdict = r.get("verdict", "train")
        colour = {"correct": (30, 105, 70), "WRONG": (163, 46, 40)}.get(
            verdict, (120, 126, 136))
        draw.text((cx + 3, cy + IMG + 2),
                  f"pad {float(r.pad_fraction):.2f}  ar {float(r.aspect_ratio):.1f}:1"
                  f"  {int(float(r.box_area))}px2", fill=(45, 50, 58))
        draw.text((cx + 3, cy + IMG + 16), f"fold0: {verdict}", fill=colour)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 2. Per-source-label recall
# ---------------------------------------------------------------------------
def per_source_label_recall(preds, heads):
    test = preds[preds.split == "test"]
    out = {}
    for h in heads:
        sub = test[test.head_true == h]
        rows = []
        for (ds, lab), g in sub.groupby(["source_dataset", "source_label"]):
            rows.append({
                "source": f"{ds}/{lab}",
                "support": int(len(g)),
                "recall": float(g.correct.mean()),
                "mean_pad_fraction": float(g.pad_fraction.mean()),
                "mean_aspect_ratio": float(g.aspect_ratio.mean()),
            })
        rows.sort(key=lambda r: -r["support"])
        # Weighted mean over the OTHER labels, so a label can be compared with
        # the head it sits in without comparing it to itself.
        for r in rows:
            others = [o for o in rows if o["source"] != r["source"]]
            tot = sum(o["support"] for o in others)
            r["head_recall_excluding_this_label"] = (
                float(sum(o["recall"] * o["support"] for o in others) / tot)
                if tot else None)
            r["delta_vs_rest_of_head"] = (
                r["recall"] - r["head_recall_excluding_this_label"]
                if r["head_recall_excluding_this_label"] is not None else None)
        out[h] = rows
    return out


# ---------------------------------------------------------------------------
# 3. Embedding
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, dataset, device, batch_size=64):
    """2048-d penultimate activations via a forward hook on avgpool."""
    feats = []

    def hook(_m, _i, o):
        feats.append(o.detach().flatten(1).float().cpu().numpy())

    handle = model.avgpool.register_forward_hook(hook)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
    model.eval()
    for images, _ in loader:
        model(images.to(device))
    handle.remove()
    return np.concatenate(feats)


def embed_v3(preds, cfg, frame, out_dir):
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    patches = resolve_patches_dir(cfg)
    manifest_path = patches / "manifest.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(REPO_ROOT / "models" / "vision" / "best_resnet50_fold0.pth",
                       device=device, frame=frame).to(device)

    ds = PatchDataset(manifest_path, "test", frame.head_names,
                      transform=get_eval_transforms(cfg), patches_root=patches)
    keep = [i for i, t in enumerate(ds.targets)
            if frame.head_names[t] == "V3"]
    sub = torch.utils.data.Subset(ds, keep)
    labels = np.array([ds.rows[i]["source_label"] for i in keep])
    print(f"  extracting 2048-d features for {len(keep)} V3 test patches")

    feats = extract_features(model, sub, device)
    pca = PCA(n_components=min(50, feats.shape[1], feats.shape[0]),
              random_state=SEED)
    reduced = pca.fit_transform(feats)
    print(f"  PCA -> {reduced.shape[1]}d, "
          f"explained variance {pca.explained_variance_ratio_.sum():.3f}")

    import umap
    emb = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=25,
                    min_dist=0.1).fit_transform(reduced)

    is_disc = labels == "discontinuity"
    result = {
        "n_v3_test_patches": int(len(keep)),
        "n_discontinuity": int(is_disc.sum()),
        "n_other_v3": int((~is_disc).sum()),
        "pca_components": int(reduced.shape[1]),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "label_counts": {k: int(v) for k, v in
                         pd.Series(labels).value_counts().items()},
    }
    if is_disc.any() and (~is_disc).any():
        binary = np.where(is_disc, "discontinuity", "other_V3")
        result["silhouette_discontinuity_vs_rest"] = {
            "pca50": float(silhouette_score(reduced, binary)),
            "umap2d": float(silhouette_score(emb, binary)),
        }
        print(f"  silhouette (discontinuity vs rest of V3): "
              f"PCA-50 {result['silhouette_discontinuity_vs_rest']['pca50']:.4f}, "
              f"UMAP-2D {result['silhouette_discontinuity_vs_rest']['umap2d']:.4f}")

    # plot
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    order = pd.Series(labels).value_counts().index.tolist()
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(order):
        m = labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=7, alpha=0.55,
                   label=f"{lab} (n={int(m.sum())})",
                   color=("#A32E28" if lab == "discontinuity" else cmap(i % 10)),
                   edgecolors="none", zorder=3 if lab == "discontinuity" else 2)
    ax.set_title("V3 {N,U} test patches, penultimate features\n"
                 "PCA-50 then UMAP, coloured by source label", fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    leg = ax.legend(fontsize=8.5, markerscale=2.2, loc="best")
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "v3_embedding_by_source_label.png", dpi=150)
    plt.close(fig)
    print(f"  -> {(out_dir/'v3_embedding_by_source_label.png').relative_to(REPO_ROOT)}")
    return result


# ---------------------------------------------------------------------------
# 4. Label inventory
# ---------------------------------------------------------------------------
def label_inventory(manifest):
    vision = find_dir(REPO_ROOT / "data", "vision")
    cmap = yaml.safe_load(
        open(REPO_ROOT / "configs" / "class_map.yaml"))["sources"]

    sources = {
        "final_welding": load_yaml_names(
            find_dir(vision, "FINAL WELDING DATASET") / "data.yaml"),
        "wijaya": load_yaml_names(
            find_dir(vision, "The Welding Defect Dataset") / "data.yaml"),
        "weld_dataset": ["pore", "deposit", "discontinuity", "stain"],
    }

    inv = {}
    for src, names in sources.items():
        counts = (manifest[manifest.source_dataset == src]
                  .source_label.value_counts().to_dict())
        total = sum(v for k, v in counts.items() if k != "__negative__")
        entries = []
        for nm in names:
            n = int(counts.get(nm, 0))
            entries.append({
                "class": nm,
                "mapped_to": cmap.get(src, {}).get(nm, "UNMAPPED"),
                "instances_post_dedup": n,
                "share_of_source": (n / total) if total else 0.0,
            })
        entries.sort(key=lambda e: -e["instances_post_dedup"])
        inv[src] = {
            "n_classes_in_data_yaml": len(names),
            "total_kept_instances": int(total),
            "classes": entries,
        }
    return inv


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/diagnostics")
    ap.add_argument("--sheet-dir", default="results/review/mapping")
    args = ap.parse_args()

    with open(REPO_ROOT / "configs" / "vision_train.yaml") as f:
        cfg = yaml.safe_load(f)
    frame = load_frame()
    out_dir = REPO_ROOT / args.out_dir
    sheet_dir = REPO_ROOT / args.sheet_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    patches_root = resolve_patches_dir(cfg)
    manifest = pd.read_csv(patches_root / "manifest.csv")
    manifest["box_area"] = manifest.box_w * manifest.box_h
    manifest["aspect_ratio"] = (
        np.maximum(manifest.box_w, manifest.box_h)
        / np.maximum(np.minimum(manifest.box_w, manifest.box_h), 1e-9))

    preds = pd.read_parquet(out_dir / "predictions_fold0.parquet")
    verdict = {r.patch_path: ("correct" if r.correct else "WRONG")
               for r in preds.itertuples()}
    manifest["verdict"] = manifest.patch_path.map(verdict).fillna("train")

    print("=" * 78)
    print("STEP 6 -- mapping evidence pack")
    print("=" * 78)

    # ---- 1. contact sheets ----
    print("\n[1/4] Contact sheets")
    sheets = []
    for ds, lab in TARGETS:
        rows = manifest[(manifest.source_dataset == ds)
                        & (manifest.source_label == lab)]
        if rows.empty:
            print(f"  {ds}/{lab}: no instances, skipped")
            continue
        head = rows["head"].mode().iat[0]
        sample = stratified_sample(rows)
        safe = lab.replace(" ", "-").replace("/", "-")
        path = build_contact_sheet(sample, ds, lab, head, patches_root,
                                   sheet_dir / f"contact_{ds}_{safe}.png")
        n_pred = int((sample.verdict != "train").sum())
        print(f"  {ds}/{lab:<22} {head}  {len(rows):>6} instances, sheet of "
              f"{len(sample)} ({n_pred} with fold-0 predictions) -> "
              f"{path.name}")
        sheets.append(str(path.relative_to(REPO_ROOT)))

    # ---- 2. per-source-label recall ----
    print("\n[2/4] Per-source-label recall (test split)")
    recall = per_source_label_recall(preds, frame.head_names)
    for h in frame.head_names:
        if not recall[h]:
            continue
        print(f"\n  {h}")
        print(f"    {'source':<34}{'n':>6}{'recall':>9}{'vs rest':>10}"
              f"{'pad':>8}{'aspect':>8}")
        for r in recall[h]:
            dv = r["delta_vs_rest_of_head"]
            print(f"    {r['source']:<34}{r['support']:>6}{r['recall']:>9.4f}"
                  f"{('--' if dv is None else f'{dv:+.4f}'):>10}"
                  f"{r['mean_pad_fraction']:>8.3f}"
                  f"{r['mean_aspect_ratio']:>8.2f}")

    # ---- 3. embedding ----
    print("\n[3/4] V3 embedding")
    emb = embed_v3(preds, cfg, frame, out_dir)

    # ---- 4. inventory ----
    print("\n[4/4] Label inventory (post-dedup)")
    inv = label_inventory(manifest)
    for src, d in inv.items():
        print(f"\n  {src}: {d['n_classes_in_data_yaml']} classes in data.yaml, "
              f"{d['total_kept_instances']} kept instances")
        for e in d["classes"]:
            print(f"    {e['class']:<26}{e['mapped_to']:>9}"
                  f"{e['instances_post_dedup']:>8}"
                  f"{100*e['share_of_source']:>8.1f}%")

    payload = {
        "seed": SEED,
        "contact_sheets": sheets,
        "per_source_label_recall": recall,
        "v3_embedding": emb,
        "label_inventory": inv,
    }
    with open(out_dir / "mapping_evidence.json", "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\n-> {(out_dir/'mapping_evidence.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
