"""
Task G1 -- build the source_sequence grouping key.

source_frame closes exact and filename-identity duplication. It does not close
sequence structure: 81.59% of test patches share a sequence stem with a train
frame, because consecutive video frames carry different names, different
source_frame values, and phash distances above 5.

source_sequence is the union-find over three relations, applied in order:

  1. the existing perceptual union-find (filename identity, MD5, phash <= 5),
     which is already encoded in source_frame -- every surviving image is its
     own group representative, so this relation contributes the identity
     partition and nothing more;
  2. SEQUENCE STEM, from sequence_audit.sequence_stem;
  3. EMBEDDING ADJACENCY -- connect two frames whose mean patch feature vectors
     are within cosine distance 0.05. This is the relation that addresses the
     phash recall ceiling (measured 0.1892) directly: features from a trained
     network are invariant to the rotation and shear that defeat phash.

Each relation's marginal contribution is reported, so it is visible which one
is doing the work.

Nothing is deleted. This changes the grouping key and therefore the split
assignment, not the corpus on disk.

Usage:
    python src/vision/diagnostics/build_sequence_groups.py --arch resnet50
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
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
from src.vision.diagnostics.nn_audit import features  # noqa: E402
from src.vision.diagnostics.sequence_audit import sequence_stem  # noqa: E402
from src.vision.model import load_model  # noqa: E402

SEED = 42
EMBED_THRESHOLD = 0.05


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            return True
        return False

    def groups(self):
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--threshold", type=float, default=EMBED_THRESHOLD)
    ap.add_argument("--exclude-negatives", action="store_true",
                    help="characterise a frame by its ANNOTATED patches only. "
                         "__negative__ crops are blank background and are "
                         "mutually near-identical regardless of which weld "
                         "they came from, so including them chains unrelated "
                         "frames into one component.")
    ap.add_argument("--skip-embedding", action="store_true",
                    help="relations 1 and 2 only; states so in the output")
    args = ap.parse_args()

    with open(REPO_ROOT / "configs" / "vision_train.yaml") as f:
        cfg = yaml.safe_load(f)
    frame = load_frame()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patches_root = resolve_patches_dir(cfg)
    manifest_path = patches_root / "manifest.csv"

    manifest = pd.read_csv(manifest_path)
    frames = manifest.drop_duplicates("source_frame")[
        ["source_frame", "source_image", "source_dataset"]].reset_index(drop=True)
    frame_ids = frames.source_frame.tolist()

    print("=" * 78)
    print("G1 -- building source_sequence")
    print("=" * 78)
    print(f"patches {len(manifest)}, source frames {len(frame_ids)}")

    uf = UnionFind(frame_ids)
    contributions = {}

    # ---- relation 1: existing perceptual union-find ----
    # source_frame already IS that union-find's representative, and
    # non-representatives were dropped from the corpus, so this relation merges
    # nothing further. Recorded for completeness rather than effect.
    merged1 = 0
    contributions["1_perceptual_union_find"] = {
        "merges": merged1,
        "note": "already encoded in source_frame; every surviving image is its "
                "own representative, so this contributes the identity partition",
    }
    print(f"\n[1] perceptual union-find (already in source_frame): "
          f"{merged1} further merges")

    # ---- relation 2: sequence stem ----
    stems = defaultdict(list)
    for r in frames.itertuples():
        stems[(r.source_dataset, sequence_stem(r.source_image)[0])].append(
            r.source_frame)
    merged2 = 0
    for members in stems.values():
        for m in members[1:]:
            if uf.union(members[0], m):
                merged2 += 1
    n_after_2 = len(uf.groups())
    contributions["2_sequence_stem"] = {
        "merges": merged2, "groups_after": n_after_2,
        "distinct_stems": len(stems),
    }
    print(f"[2] sequence stem: {merged2} merges, {len(frame_ids)} -> "
          f"{n_after_2} groups")

    # ---- relation 3: embedding adjacency ----
    if args.skip_embedding:
        contributions["3_embedding_adjacency"] = {
            "merges": 0, "skipped": True,
            "note": "run without --skip-embedding to include this relation"}
        print("[3] embedding adjacency: SKIPPED")
    else:
        ckpt = (Path(args.checkpoint) if args.checkpoint
                else REPO_ROOT / "models" / "vision" / f"best_{args.arch}.pth")
        if not ckpt.exists():
            sys.exit(f"ERROR: {ckpt} not found")
        model = load_model(ckpt, device=device, frame=frame,
                           architecture=args.arch).to(device)

        # One feature vector per FRAME: the mean of its patches, renormalised.
        ds = PatchDataset(manifest_path, None, frame.head_names,
                          transform=get_eval_transforms(cfg),
                          patches_root=patches_root, all_splits=True)
        print(f"[3] extracting features for {len(ds)} patches ...")
        feats = features(model, ds, device)

        rows = pd.DataFrame(ds.rows)
        order = {f: i for i, f in enumerate(frame_ids)}
        acc = torch.zeros(len(frame_ids), feats.shape[1])
        cnt = torch.zeros(len(frame_ids))
        skipped_neg = 0
        for i, sf in enumerate(rows.source_frame):
            if args.exclude_negatives and rows.source_label.iat[i] == "__negative__":
                skipped_neg += 1
                continue
            j = order[sf]
            acc[j] += feats[i]
            cnt[j] += 1
        if args.exclude_negatives:
            print(f"    excluded {skipped_neg} __negative__ patches from the "
                  f"frame means; {int((cnt == 0).sum())} frames have no "
                  f"annotated patch and keep a zero vector")
        frame_feats = torch.nn.functional.normalize(
            acc / cnt.clamp(min=1).unsqueeze(1), dim=1).to(device)

        merged3, pairs_found = 0, 0
        chunk = 512
        for s in range(0, len(frame_feats), chunk):
            sim = frame_feats[s:s + chunk] @ frame_feats.T
            close = (1.0 - sim) < args.threshold
            for r in range(close.shape[0]):
                i = s + r
                js = torch.nonzero(close[r], as_tuple=False).flatten().tolist()
                for j in js:
                    if j <= i:
                        continue
                    pairs_found += 1
                    if uf.union(frame_ids[i], frame_ids[j]):
                        merged3 += 1
        n_after_3 = len(uf.groups())
        contributions["3_embedding_adjacency"] = {
            "merges": merged3, "pairs_below_threshold": pairs_found,
            "threshold": args.threshold, "groups_after": n_after_3,
            "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        }
        print(f"[3] embedding adjacency (cosine < {args.threshold}): "
              f"{pairs_found} close pairs, {merged3} merges, "
              f"{n_after_2} -> {n_after_3} groups")

    # ---- assemble ----
    groups = uf.groups()
    rep = {}
    for root, members in groups.items():
        canonical = min(members)
        for m in members:
            rep[m] = canonical

    manifest["source_sequence"] = manifest.source_frame.map(rep)
    sizes = Counter(len(v) for v in groups.values())

    print(f"\nsource_sequence: {len(groups)} groups from {len(frame_ids)} "
          f"source frames "
          f"({100*len(groups)/len(frame_ids):.1f}% of the previous count)")
    print(f"group-size distribution (top 10): "
          f"{dict(sorted(sizes.items())[:10])}")
    largest = sorted(groups.values(), key=len, reverse=True)[:5]
    print("largest groups:")
    for g in largest:
        ds_of = frames.set_index("source_frame").source_dataset
        print(f"  n={len(g):<5} {ds_of.get(g[0], '?'):<15} "
              f"e.g. {Path(g[0]).name[:52]}")

    manifest.to_csv(manifest_path, index=False)
    print(f"\nmanifest updated with source_sequence: "
          f"{manifest_path.relative_to(REPO_ROOT)}")

    payload = {
        "seed": SEED,
        "n_source_frames": len(frame_ids),
        "n_sequence_groups": len(groups),
        "reduction_factor": len(frame_ids) / max(1, len(groups)),
        "relations": contributions,
        "group_size_distribution": {str(k): v for k, v in sorted(sizes.items())},
        "largest_groups": [
            {"size": len(g), "representative": g[0]} for g in largest],
    }
    out = patches_root / "sequence_groups.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
