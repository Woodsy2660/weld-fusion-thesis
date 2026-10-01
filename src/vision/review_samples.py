"""
Contact sheets for the two class mappings that need human confirmation.

Both live in configs/class_map.yaml, so changing either is one line plus a
re-run of the conversion. Neither is decided here.

  1. weld_dataset `discontinuity` -> provisionally V3 {N,U}.
     If these are in fact cracks they belong in V2 {C}, and a compound-element
     assignment would be discarding real crack evidence.

  2. final_welding `Crack` -> V2 {C}.
     Surface-breaking cracks are uncommon in GMAW. The instance count is high
     enough to suspect the label covers something else (toe undercut, a
     linear slag line), in which case V3 is the correct home.

Usage:
    python src/vision/review_samples.py [--n 50] [--seed 42]
"""
import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.vision.convert_and_remap import (  # noqa: E402
    CONTEXT_MARGIN, IMAGE_EXTS, expand_and_clip, find_dir, letterbox,
    load_yaml_names, parse_instances,
)

TILE = 160
COLS = 10


def collect_instances(source, target_class, limit_scan=None):
    """Find every annotation instance of one source class, with its image."""
    vision = find_dir(REPO_ROOT / "data", "vision")
    found = []

    if source == "final_welding":
        root = find_dir(vision, "FINAL WELDING DATASET")
        names = load_yaml_names(root / "data.yaml")
        cid = names.index(target_class)
        for split in ("train", "valid", "test"):
            sd = find_dir(root, split)
            if sd is None:
                continue
            idir, ldir = find_dir(sd, "images"), find_dir(sd, "labels")
            imgs = sorted(p for p in idir.iterdir()
                          if p.suffix.lower() in IMAGE_EXTS)
            if limit_scan:
                imgs = imgs[:limit_scan]
            for im in imgs:
                lb = ldir / f"{im.stem}.txt"
                if not lb.exists():
                    continue
                inst, _ = parse_instances(lb)
                for c, xc, yc, w, h in inst:
                    if c == cid:
                        found.append((im, xc, yc, w, h))

    elif source == "weld_dataset":
        root = find_dir(vision, "weld-dataset")
        names = ["pore", "deposit", "discontinuity", "stain"]
        cid = names.index(target_class)
        for sub in ("high_resolution_welds", "low_resolution_welds"):
            sd = find_dir(root, sub)
            if sd is None:
                continue
            imgs = sorted(p for p in sd.iterdir()
                          if p.suffix.lower() in IMAGE_EXTS)
            if limit_scan:
                imgs = imgs[:limit_scan]
            for im in imgs:
                lb = im.with_suffix(".yolo")
                if not lb.exists():
                    continue
                inst, _ = parse_instances(lb)
                for c, xc, yc, w, h in inst:
                    if c == cid:
                        found.append((im, xc, yc, w, h))
    else:
        raise ValueError(source)

    return found


def contact_sheet(samples, out_path, title):
    """Tile crops into one reviewable PNG, cropped the way training will see them."""
    from PIL import Image, ImageDraw

    rows = (len(samples) + COLS - 1) // COLS
    header = 28
    sheet = Image.new("RGB", (COLS * TILE, rows * TILE + header), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(20, 20, 20))

    for i, (img_path, xc, yc, w, h) in enumerate(samples):
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                x0, y0, x1, y1 = expand_and_clip(xc, yc, w, h, W, H)
                patch, _ = letterbox(im.crop((x0, y0, x1, y1)), TILE)
        except Exception:
            continue
        if patch is None:
            continue
        sheet.paste(patch, ((i % COLS) * TILE, header + (i // COLS) * TILE))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-scan", type=int, default=None,
                    help="cap images scanned per split (faster, less coverage)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = REPO_ROOT / "results" / "review"
    written = []

    jobs = [
        ("weld_dataset", "discontinuity", "V3",
         "discontinuity_sample.png",
         "weld_dataset 'discontinuity' -> provisionally V3 {N,U}. "
         "If these are cracks they belong in V2 {C}."),
        ("final_welding", "Crack", "V2",
         "crack_sample.png",
         "final_welding 'Crack' -> V2 {C}. Surface-breaking cracks are "
         "uncommon in GMAW; if these are not cracks, V3 {N,U} is correct."),
    ]

    for source, cls, head, fname, blurb in jobs:
        print(f"\nScanning {source} for '{cls}' instances ...")
        found = collect_instances(source, cls, args.limit_scan)
        print(f"  found {len(found)} instances")
        if not found:
            print("  nothing to sample; skipped")
            continue
        rng.shuffle(found)
        sample = found[:args.n]
        path = contact_sheet(
            sample, out_dir / fname,
            f"{source} / {cls} -> {head}   ({len(sample)} of {len(found)}, "
            f"{CONTEXT_MARGIN}x context)")
        written.append((path, blurb))
        print(f"  wrote {path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 78)
    print("REVIEW REQUIRED -- these two mappings are PROVISIONAL")
    print("=" * 78)
    for path, blurb in written:
        print(f"\n  {path.relative_to(REPO_ROOT)}")
        print(f"      {blurb}")
    print("\nBoth mappings live in configs/class_map.yaml. Changing either is "
          "one line\nplus a re-run of convert_and_remap.py.")
    print("\nYou MUST confirm both before any result is written up. An "
          "unreviewed mapping\npropagates straight into the BPA and from there "
          "into the fused decision.")


if __name__ == "__main__":
    main()
