"""
Stage 1 of the vision pipeline: annotations -> per-instance training patches.

For every annotation INSTANCE (not every image) this emits one patch, labelled
by that instance's own class mapped through configs/class_map.yaml onto the
four vision focal elements in configs/frame.yaml.

Why per-instance rather than per-image dominant class:
  * Scale. Inference runs on ~10 mm weld-cell crops. Training on whole weld
    images teaches features at the wrong scale.
  * Information. An image with five pores and one crack is not "porosity" --
    the crack is real evidence and a dominant-class vote throws it away.

Source images are perceptually deduplicated BEFORE cropping, so augmented
Roboflow siblings cannot leak an image's crops into two different splits.

Nothing here reads models/lohi-weld-baselines/ -- that tree holds YOLOv7
predictions, not ground truth.

Usage:
    python src/vision/convert_and_remap.py [--dry-run] [--limit N] [--workers N]
"""
import argparse
import csv
import hashlib
import re
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import (  # noqa: E402
    DISCARD, load_class_map, load_frame, resolve,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

MANIFEST_COLUMNS = [
    "patch_path", "head", "source_dataset", "source_image", "source_frame",
    "source_label", "box_x", "box_y", "box_w", "box_h", "is_negative",
    "pad_fraction",
]

# Context margin: the annotation box is expanded by this factor before
# cropping, so the patch carries surrounding bead for context. 1.5-2.0x.
CONTEXT_MARGIN = 1.75
MIN_BOX_PX = 8           # ignore degenerate annotation boxes
PATCH_SIZE = 224
JPEG_QUALITY = 92
PHASH_THRESHOLD = 5      # Hamming distance for "perceptually the same frame"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def find_dir(parent: Path, name: str):
    """Case-insensitive directory lookup (repo is used from Windows and WSL)."""
    exact = parent / name
    if exact.is_dir():
        return exact
    if not parent.is_dir():
        return None
    lowered = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == lowered:
            return child
    return None


def load_yaml_names(data_yaml: Path) -> list:
    """Read the `names:` list out of a Roboflow data.yaml."""
    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    nc = cfg.get("nc", len(names))
    if nc != len(names):
        raise ValueError(f"{data_yaml}: nc={nc} but {len(names)} names listed")
    return list(names)


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------
def parse_instances(label_path: Path):
    """
    Parse one YOLO-family annotation file into per-instance normalised boxes.

    Handles BOTH formats, mixed freely inside a single file:
        5 fields          -> "cls xc yc w h"                     (bbox)
        odd, > 5 fields   -> "cls x1 y1 x2 y2 ... xN yN"         (polygon)

    A polygon is reduced to its axis-aligned bounding box. Only field 0 is
    trusted for the class ID, so field count never matters for labelling.

    Returns (instances, n_malformed) where each instance is
    (class_id, xc, yc, w, h) in normalised [0,1] coordinates.
    """
    instances, malformed = [], 0
    try:
        text = label_path.read_text(errors="replace")
    except OSError:
        return instances, 1

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cid = int(float(parts[0]))
        except (ValueError, IndexError):
            malformed += 1
            continue

        vals = parts[1:]
        try:
            nums = [float(v) for v in vals]
        except ValueError:
            malformed += 1
            continue

        if len(nums) == 4:                                  # bbox
            xc, yc, w, h = nums
        elif len(nums) >= 6 and len(nums) % 2 == 0:         # polygon
            xs, ys = nums[0::2], nums[1::2]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        else:
            malformed += 1
            continue

        if w <= 0 or h <= 0:
            malformed += 1
            continue
        instances.append((cid, xc, yc, w, h))

    return instances, malformed


# ---------------------------------------------------------------------------
# Source enumeration -- one record per source image
# ---------------------------------------------------------------------------
def _record(img, source, split, instances, names, class_map):
    """Attach mapped heads to each instance. Unmapped classes raise."""
    mapped = []
    for cid, xc, yc, w, h in instances:
        if cid < 0 or cid >= len(names):
            continue                                  # counted by caller
        name = names[cid]
        head = resolve(class_map, source, name)
        mapped.append((name, head, xc, yc, w, h))
    return {"path": str(img), "source": source, "split": split,
            "instances": mapped}


def enumerate_roboflow(root, source, names, class_map, limit=None):
    """train|valid|test with images/ + labels/ subdirectories."""
    records, stats = [], Counter()
    for split in ("train", "valid", "test"):
        split_dir = find_dir(root, split)
        if split_dir is None:
            continue
        img_dir, lbl_dir = find_dir(split_dir, "images"), find_dir(split_dir, "labels")
        if img_dir is None:
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if limit:
            imgs = imgs[:limit]
        for img in imgs:
            stats["images"] += 1
            lbl = (lbl_dir / f"{img.stem}.txt") if lbl_dir else None
            if lbl is None or not lbl.exists():
                stats["missing_label"] += 1
                records.append(_record(img, source, split, [], names, class_map))
                continue
            inst, mal = parse_instances(lbl)
            stats["malformed"] += mal
            if not inst:
                stats["empty_label"] += 1
            for cid, *_ in inst:
                if cid < 0 or cid >= len(names):
                    stats["out_of_range"] += 1
            records.append(_record(img, source, split, inst, names, class_map))
    return records, stats


def enumerate_weld_dataset(root, class_map, limit=None):
    """Flat folders; each image has a sibling .yolo (bbox) and .json (LabelMe)."""
    names = ["pore", "deposit", "discontinuity", "stain"]
    records, stats = [], Counter()
    for subset in ("high_resolution_welds", "low_resolution_welds"):
        sub = find_dir(root, subset)
        if sub is None:
            continue
        imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if limit:
            imgs = imgs[:limit]
        for img in imgs:
            stats["images"] += 1
            lbl = img.with_suffix(".yolo")
            if not lbl.exists():
                stats["missing_label"] += 1
                records.append(_record(img, "weld_dataset", subset, [], names, class_map))
                continue
            inst, mal = parse_instances(lbl)
            stats["malformed"] += mal
            if not inst:
                stats["empty_label"] += 1
            for cid, *_ in inst:
                if cid < 0 or cid >= len(names):
                    stats["out_of_range"] += 1
            records.append(_record(img, "weld_dataset", subset, inst, names, class_map))
    return records, stats


# ---------------------------------------------------------------------------
# Deduplication -- exact (MD5) then perceptual (phash), at SOURCE IMAGE level
# ---------------------------------------------------------------------------
def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _hash_one(path):
    """Worker: (md5, phash-as-int) for one image."""
    import imagehash
    from PIL import Image
    try:
        digest = md5_of(path)
        with Image.open(path) as im:
            ph = imagehash.phash(im.convert("L"))
        return path, digest, int(str(ph), 16)
    except Exception:
        return path, None, None


def roboflow_stem(path):
    """
    Roboflow encodes the ORIGINAL frame in the filename:
        <original stem>.rf.<export hash>.jpg
    so every augmented export of one frame shares a stem. This is ground truth
    for "same source frame" and catches what phash cannot: rotation and shear
    push a sibling well past Hamming 5, so phash alone recovers only ~19% of
    the sibling pairs in this corpus (measured, precision 0.97 / recall 0.19).

    Returns None when the marker is absent, so non-Roboflow sources are never
    grouped by a coincidentally shared name.
    """
    name = Path(path).name
    if ".rf." not in name:
        return None
    return re.split(r"\.rf\.", name)[0]


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_groups(hashes: dict, threshold=PHASH_THRESHOLD):
    """
    Union-find over perceptual hashes, using 8-bit banding as an LSH prefilter.

    With an 8-band split of a 64-bit hash, any pair within `threshold` <= 5 bits
    must agree exactly on at least 3 bands, so scanning band buckets cannot miss
    a true near-duplicate while avoiding the full O(n^2) comparison.
    """
    parent = {p: p for p in hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Roboflow filename identity first -- it is exact where it applies.
    by_stem = defaultdict(list)
    for path in hashes:
        stem = roboflow_stem(path)
        if stem:
            by_stem[stem].append(path)
    for members in by_stem.values():
        for m in members[1:]:
            union(members[0], m)

    buckets = defaultdict(list)
    for path, h in hashes.items():
        for band in range(8):
            buckets[(band, (h >> (band * 8)) & 0xFF)].append(path)

    compared = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 2000:
            continue                       # a 2000-deep bucket is a flat image
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                key = (a, b) if a < b else (b, a)
                if key in compared:
                    continue
                compared.add(key)
                if hamming(hashes[a], hashes[b]) <= threshold:
                    union(a, b)

    groups = defaultdict(list)
    for path in hashes:
        groups[find(path)].append(path)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
def expand_and_clip(xc, yc, w, h, W, H, margin=CONTEXT_MARGIN):
    """Annotation box (normalised) -> expanded pixel box clipped to the image."""
    bw, bh = w * W, h * H
    cx, cy = xc * W, yc * H
    ew, eh = bw * margin, bh * margin
    x0 = max(0, int(round(cx - ew / 2)))
    y0 = max(0, int(round(cy - eh / 2)))
    x1 = min(W, int(round(cx + ew / 2)))
    y1 = min(H, int(round(cy + eh / 2)))
    return x0, y0, x1, y1


def letterbox(im, size=PATCH_SIZE):
    """
    Resize preserving aspect ratio, padding to square by edge replication.

    Stretching a 3.7:1 crop to a square destroys bead geometry, which is
    exactly the signal V3 depends on. Returns (image, pad_fraction) where
    pad_fraction is the share of the square that is padding.
    """
    from PIL import Image, ImageOps
    w, h = im.size
    if w == 0 or h == 0:
        return None, 1.0
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.BILINEAR)

    pad_l, pad_t = (size - nw) // 2, (size - nh) // 2
    pad_r, pad_b = size - nw - pad_l, size - nh - pad_t
    if pad_l or pad_t or pad_r or pad_b:
        # Edge replicate rather than a flat fill: a constant border injects a
        # hard synthetic edge that the network will happily learn.
        im = ImageOps.expand(im, (pad_l, pad_t, pad_r, pad_b))
        im = _replicate_edges(im, pad_l, pad_t, nw, nh)
    pad_fraction = 1.0 - (min(w, h) / max(w, h))
    return im, pad_fraction


def _replicate_edges(im, pad_l, pad_t, nw, nh):
    """Fill the letterbox bars by stretching the outermost row/column."""
    W, H = im.size
    content = im.crop((pad_l, pad_t, pad_l + nw, pad_t + nh))
    if pad_l:
        im.paste(content.crop((0, 0, 1, nh)).resize((pad_l, nh)), (0, pad_t))
    if pad_l + nw < W:
        right = W - (pad_l + nw)
        im.paste(content.crop((nw - 1, 0, nw, nh)).resize((right, nh)),
                 (pad_l + nw, pad_t))
    if pad_t:
        strip = im.crop((0, pad_t, W, pad_t + 1)).resize((W, pad_t))
        im.paste(strip, (0, 0))
    if pad_t + nh < H:
        bottom = H - (pad_t + nh)
        strip = im.crop((0, pad_t + nh - 1, W, pad_t + nh)).resize((W, bottom))
        im.paste(strip, (0, pad_t + nh))
    return im


def sample_negative_boxes(instances, W, H, size_pool, rng, n_wanted,
                          max_tries=40):
    """
    Sample annotation-free boxes for V4.

    Box sizes are drawn from the SAME pool as the positives. If negatives were
    systematically larger or smaller than positives the network would learn
    "crop size predicts class" and no metric in the report would reveal it.
    """
    boxes = []
    existing = []
    for _, _, xc, yc, w, h in instances:
        existing.append((
            (xc - w / 2) * W, (yc - h / 2) * H,
            (xc + w / 2) * W, (yc + h / 2) * H,
        ))

    for _ in range(n_wanted):
        for _ in range(max_tries):
            nw, nh = rng.choice(size_pool)
            bw, bh = max(MIN_BOX_PX, nw * W), max(MIN_BOX_PX, nh * H)
            if bw >= W or bh >= H:
                continue
            x0 = rng.uniform(0, W - bw)
            y0 = rng.uniform(0, H - bh)
            x1, y1 = x0 + bw, y0 + bh
            if any(not (x1 <= ex0 or x0 >= ex1 or y1 <= ey0 or y0 >= ey1)
                   for ex0, ey0, ex1, ey1 in existing):
                continue
            boxes.append((x0, y0, x1, y1))
            break
    return boxes


def process_record(args):
    """
    Worker: crop every mapped instance of one source image, plus its negatives.

    Returns (rows, counts, error). Runs in a subprocess -- the source images are
    decoded once here and all their crops emitted, so a 200-instance image is
    one decode rather than 200.
    """
    rec, out_root, size_pool, negatives_per_image, seed, patch_size = args
    from PIL import Image

    rows, counts = [], Counter()
    path = Path(rec["path"])
    rng = random.Random(f"{seed}:{path.name}")

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size

            for idx, (name, head, xc, yc, w, h) in enumerate(rec["instances"]):
                if head == DISCARD:
                    counts[f"discard::{name}"] += 1
                    continue
                x0, y0, x1, y1 = expand_and_clip(xc, yc, w, h, W, H)
                if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
                    counts["too_small"] += 1
                    continue
                patch, pad = letterbox(im.crop((x0, y0, x1, y1)), patch_size)
                if patch is None:
                    counts["too_small"] += 1
                    continue
                fname = f"{rec['source']}__{path.stem}__i{idx:04d}.jpg"
                dest = Path(out_root) / head / fname
                patch.save(dest, quality=JPEG_QUALITY)
                counts[head] += 1
                rows.append([
                    f"{head}/{fname}", head, rec["source"], str(path),
                    rec.get("frame_id", str(path)), name,
                    round((xc - w / 2) * W, 2), round((yc - h / 2) * H, 2),
                    round(w * W, 2), round(h * H, 2), 0, round(pad, 4),
                ])

            if negatives_per_image and size_pool:
                negs = sample_negative_boxes(
                    rec["instances"], W, H, size_pool, rng, negatives_per_image)
                for n, (x0, y0, x1, y1) in enumerate(negs):
                    patch, pad = letterbox(
                        im.crop((int(x0), int(y0), int(x1), int(y1))), patch_size)
                    if patch is None:
                        continue
                    fname = f"{rec['source']}__{path.stem}__neg{n:02d}.jpg"
                    dest = Path(out_root) / "V4" / fname
                    patch.save(dest, quality=JPEG_QUALITY)
                    counts["V4"] += 1
                    counts["negatives"] += 1
                    rows.append([
                        f"V4/{fname}", "V4", rec["source"], str(path),
                        rec.get("frame_id", str(path)), "__negative__",
                        round(x0, 2), round(y0, 2),
                        round(x1 - x0, 2), round(y1 - y0, 2), 1, round(pad, 4),
                    ])
    except Exception as exc:
        return [], counts, f"{path}: {exc}"

    return rows, counts, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate, map and dedup; write no patches")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap source images per split/subset (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--negatives-per-image", type=int, default=1,
                    help="annotation-free V4 patches sampled per source image")
    ap.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/processed/vision_patches")
    ap.add_argument("--skip-dedup", action="store_true",
                    help="diagnostic only; grouped splitting assumes dedup ran")
    ap.add_argument("--keep-augmented-siblings", action="store_true",
                    help="keep every augmented export instead of collapsing "
                         "each source frame to one representative. Siblings "
                         "still share a frame id, so the grouped split keeps "
                         "them on one side of the boundary and cannot leak. "
                         "Inflates apparent dataset size and class weights by "
                         "the export multiplier; effective sample size is "
                         "unchanged either way.")
    args = ap.parse_args()

    frame = load_frame()
    class_map = load_class_map(frame=frame)
    out_root = REPO_ROOT / args.out_dir

    data_root = REPO_ROOT / "data"
    vision_root = find_dir(data_root, "vision")
    if vision_root is None:
        sys.exit(f"ERROR: no vision data directory under {data_root}")

    print("=" * 78)
    print("CONVERT AND REMAP -> per-instance patches over vision focal elements")
    print("=" * 78)
    print(f"Frame       : {frame}")
    print(f"Heads       : {frame.head_names}  (n={frame.num_heads})")
    print(f"Vision root : {vision_root}")
    print(f"Patches out : {out_root}"
          f"{'   (DRY RUN)' if args.dry_run else ''}")
    print(f"Context margin {CONTEXT_MARGIN}x, letterboxed to "
          f"{args.patch_size}x{args.patch_size} (aspect preserved, edges "
          f"replicated)")
    print("NOTE: models/lohi-weld-baselines/ is NOT scanned -- it holds YOLOv7 "
          "predictions.")

    # ---------------- enumerate ----------------
    print("\n[1/5] Enumerating source images and mapping instance classes ...")
    records, stats = [], Counter()

    fw = find_dir(vision_root, "FINAL WELDING DATASET")
    if fw:
        names = load_yaml_names(fw / "data.yaml")
        r, s = enumerate_roboflow(fw, "final_welding", names, class_map, args.limit)
        records += r
        stats.update({f"final_welding/{k}": v for k, v in s.items()})
        print(f"  final_welding : {s['images']} images, {len(names)} classes")

    wd = find_dir(vision_root, "weld-dataset")
    if wd:
        r, s = enumerate_weld_dataset(wd, class_map, args.limit)
        records += r
        stats.update({f"weld_dataset/{k}": v for k, v in s.items()})
        print(f"  weld_dataset  : {s['images']} images, 4 classes")

    wj = find_dir(vision_root, "The Welding Defect Dataset")
    if wj:
        names = load_yaml_names(wj / "data.yaml")
        r, s = enumerate_roboflow(wj, "wijaya", names, class_map, args.limit)
        records += r
        stats.update({f"wijaya/{k}": v for k, v in s.items()})
        print(f"  wijaya        : {s['images']} images, {len(names)} classes")

    inst_by_head = Counter()
    inst_by_label = Counter()
    for rec in records:
        for name, head, *_ in rec["instances"]:
            inst_by_head[head] += 1
            inst_by_label[(rec["source"], name, head)] += 1
    print(f"  total source images : {len(records)}")
    print(f"  total instances     : {sum(inst_by_head.values())}")
    for head in frame.head_names + [DISCARD]:
        print(f"      {head:<9}: {inst_by_head.get(head, 0)}")

    # ---------------- dedup ----------------
    dup_report = {"exact_groups": 0, "exact_removed": 0,
                  "perceptual_groups": 0, "perceptual_removed": 0,
                  "cross_dataset": 0, "within_dataset": 0}
    conflicts = []
    dropped = set()
    # frame_id maps every source image to a canonical frame identity, so the
    # grouped split in merge_dedup_split.py cannot put two exports of the same
    # weld on opposite sides of the boundary.
    frame_id = {}

    if not args.skip_dedup:
        print(f"\n[2/5] Deduplicating {len(records)} SOURCE images "
              f"(MD5 exact, then phash Hamming<={PHASH_THRESHOLD}) ...")
        paths = [Path(r["path"]) for r in records]
        md5s, phashes = {}, {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (p, d, ph) in enumerate(ex.map(_hash_one, paths, chunksize=64)):
                if d is None:
                    continue
                md5s[p], phashes[p] = d, ph
                if (i + 1) % 5000 == 0:
                    print(f"      hashed {i+1}/{len(paths)}")

        by_md5 = defaultdict(list)
        for p, d in md5s.items():
            by_md5[d].append(p)
        exact = {k: sorted(v) for k, v in by_md5.items() if len(v) > 1}
        dup_report["exact_groups"] = len(exact)

        groups = near_duplicate_groups(phashes)
        dup_report["perceptual_groups"] = len(groups)

        src_of = {Path(r["path"]): r["source"] for r in records}
        heads_of = {Path(r["path"]): tuple(sorted(Counter(
            h for _, h, *_ in r["instances"]).items())) for r in records}

        # Label conflicts are detected on EXACT duplicates only, which is the
        # original semantics: identical bytes carrying different ground truth
        # is a genuine annotator contradiction, and every copy is dropped
        # rather than picking a winner.
        #
        # Perceptual siblings are NOT judged this way. Roboflow augmentation
        # rotates and crops, which legitimately changes which instances remain
        # in a frame -- a composition difference between augmented siblings is
        # expected, not contradictory. Treating it as a conflict would discard
        # thousands of usable images for no reason.
        for members in exact.values():
            label_sets = {heads_of[m] for m in members if m in heads_of}
            if len(label_sets) > 1:
                conflicts.append({
                    "members": [str(m) for m in members],
                    "sources": sorted({src_of[m] for m in members}),
                    "basis": "md5_identical_with_differing_labels",
                    "head_compositions": [
                        {"image": str(m), "heads": dict(heads_of[m])}
                        for m in members if m in heads_of
                    ],
                })
                dropped.update(members)

        # Perceptual dedup: keep one representative per group, drop the rest.
        # The representative is the member with the most annotated instances
        # (ties broken by path) so the richest copy of a frame survives.
        for members in groups.values():
            sources = {src_of[m] for m in members}
            if len(sources) > 1:
                dup_report["cross_dataset"] += 1
            else:
                dup_report["within_dataset"] += 1
            keep = max(members, key=lambda m: (sum(dict(heads_of.get(m, ())).values()), str(m)))
            for m in members:
                frame_id[m] = str(keep)        # siblings share a group key
            if not args.keep_augmented_siblings:
                dropped.update(m for m in members if m != keep)

        dup_report["perceptual_removed"] = len(dropped)
        # Group-size and per-source breakdown: phash can over-merge small,
        # low-contrast images (the ~150x40 strips especially), and a silent
        # 20% collapse of the corpus would be invisible without this.
        gsz = Counter(len(v) for v in groups.values())
        per_src = Counter()
        for members in groups.values():
            for m in members[1:]:
                per_src[src_of[m]] += 1
        subset_of = {Path(r["path"]): r["split"] for r in records}
        per_sub = Counter()
        for members in groups.values():
            for m in members[1:]:
                per_sub[f"{src_of[m]}/{subset_of[m]}"] += 1
        print(f"  group-size histogram       : {dict(sorted(gsz.items()))}")
        print(f"  siblings dropped by source : {dict(per_src)}")
        print(f"  siblings dropped by subset : {dict(per_sub.most_common(8))}")
        print(f"  exact-duplicate groups     : {len(exact)}")
        print(f"  perceptual-duplicate groups: {len(groups)}"
              f"  (within-dataset {dup_report['within_dataset']}, "
              f"cross-dataset {dup_report['cross_dataset']})")
        print(f"  source images dropped      : {len(dropped)}")
        print(f"  label-conflict groups      : {len(conflicts)} "
              f"(exact duplicates with differing labels; all copies dropped)")

        records = [r for r in records if Path(r["path"]) not in dropped]
        print(f"  source images surviving    : {len(records)}")

        # Head counts BEFORE dedup are misleading -- most of the corpus is
        # augmented re-exports of a much smaller set of frames. Recompute.
        post_head, post_label = Counter(), Counter()
        for rec in records:
            for name, head, *_ in rec["instances"]:
                post_head[head] += 1
                post_label[(rec["source"], name, head)] += 1
        print(f"\n  instances after dedup      : {sum(post_head.values())} "
              f"(was {sum(inst_by_head.values())})")
        for head in frame.head_names + [DISCARD]:
            before = inst_by_head.get(head, 0)
            after = post_head.get(head, 0)
            print(f"      {head:<9}: {after:>7}   (before dedup {before})")
        inst_by_head, inst_by_label = post_head, post_label

    if conflicts:
        merged_dir = REPO_ROOT / "data" / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        with open(merged_dir / "label_conflicts.json", "w") as f:
            json.dump(conflicts, f, indent=1)
        print(f"  -> data/merged/label_conflicts.json ({len(conflicts)} groups)")

    if args.dry_run:
        print("\nDRY RUN: no patches written.")
        _print_before_after(inst_by_label, frame)
        return

    # ---------------- crop ----------------
    print(f"\n[3/5] Cropping patches with {args.workers} workers ...")
    if out_root.exists():
        print(f"  clearing {out_root}")
        shutil.rmtree(out_root)
    for head in frame.head_names:
        (out_root / head).mkdir(parents=True, exist_ok=True)

    # Positive box sizes pool the negatives are drawn from.
    size_pool = [(w, h) for r in records for _, hd, _, _, w, h in r["instances"]
                 if hd != DISCARD]
    random.Random(args.seed).shuffle(size_pool)
    size_pool = size_pool[:20000] or [(0.1, 0.1)]

    for r in records:
        r["frame_id"] = frame_id.get(Path(r["path"]), r["path"])
    tasks = [(r, str(out_root), size_pool, args.negatives_per_image,
              args.seed, args.patch_size) for r in records]

    rows, counts, errors = [], Counter(), []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r, c, err in ex.map(process_record, tasks, chunksize=16):
            rows.extend(r)
            counts.update(c)
            if err:
                errors.append(err)
            done += 1
            if done % 2000 == 0:
                print(f"      {done}/{len(tasks)} images, "
                      f"{len(rows)} patches")

    # ---------------- manifest ----------------
    print(f"\n[4/5] Writing manifest ({len(rows)} patches) ...")
    manifest = out_root / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLUMNS)
        w.writerows(rows)

    with open(out_root / "dedup_report.json", "w") as f:
        json.dump(dup_report, f, indent=1)

    # ---------------- report ----------------
    print("\n[5/5] Results")
    print("=" * 78)
    print(f"{'head':<8}{'focal element':<16}{'patches':>10}")
    print("-" * 78)
    for head in frame.head_names:
        els = "{" + ",".join(frame.focal_element(head)) + "}"
        print(f"{head:<8}{els:<16}{counts.get(head, 0):>10}")
    print("-" * 78)
    print(f"{'TOTAL':<24}{sum(counts.get(h, 0) for h in frame.head_names):>10}")
    print(f"  of which negatives (V4): {counts.get('negatives', 0)}")
    if counts.get("too_small"):
        print(f"  skipped degenerate boxes: {counts['too_small']}")
    if errors:
        print(f"  WARNING {len(errors)} images failed; first: {errors[0]}")

    _print_before_after(inst_by_label, frame)

    print(f"\nManifest : {manifest.relative_to(REPO_ROOT)}")
    print(f"Patches  : {out_root.relative_to(REPO_ROOT)}/{{{','.join(frame.head_names)}}}/")
    print("\nNext: python src/vision/merge_dedup_split.py")


def _print_before_after(inst_by_label, frame):
    """
    Before/after: what the old 4-class taxonomy discarded vs what the focal
    element mapping now keeps.
    """
    OLD_KEPT = {                      # classes the previous taxonomy retained
        ("final_welding", "Porosity"), ("final_welding", "cluster of porosity"),
        ("final_welding", "Lack of Fusion"), ("final_welding", "Crack"),
        ("weld_dataset", "pore"), ("weld_dataset", "discontinuity"),
        ("wijaya", "Good Weld"),
    }
    print("\nBEFORE / AFTER -- instances recovered from discard")
    print("=" * 78)
    print(f"{'source':<15}{'class':<26}{'head':<6}{'instances':>10}  {'was':<9}")
    print("-" * 78)
    recovered = 0
    for (src, name, head), n in sorted(
            inst_by_label.items(), key=lambda kv: -kv[1]):
        was = "kept" if (src, name) in OLD_KEPT else "DISCARDED"
        if head == DISCARD:
            was_now = "discard"
        else:
            was_now = head
        if was == "DISCARDED" and head != DISCARD:
            recovered += n
        print(f"{src:<15}{name:<26}{was_now:<6}{n:>10}  {was:<9}")
    print("-" * 78)
    print(f"Instances recovered from discard by the remap: {recovered}")


if __name__ == "__main__":
    main()
