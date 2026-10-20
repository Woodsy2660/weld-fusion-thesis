"""
Propose weld-axis seeds: two colour-pixel points on the weld centreline per
capture, in the configs/hole_seeds.csv pattern.

PROPOSALS ONLY. Written with source=auto_proposed; a human confirms each one
against results/geometry/_diagnostics/weld_axis_seeds_sheet.png and changes
source to "manual" (editing px_v where wrong) before the seeds are trusted.

Method: at 30% and 70% of the way from the d6 to the d8 seed (in u), find the
one offset 35-65 mm from the hole line (standoff-scaled, either side) whose
40 px bands score highest jointly on brightness x local texture -- the MIG cap
is the brightest, most rippled feature -- then refine each point by +-12 px.

Usage:
    python scripts/propose_weld_axis_seeds.py
"""
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "raw" / "S1"
OUT_CSV = REPO_ROOT / "configs" / "weld_axis_seeds.csv"
SHEET = REPO_ROOT / "results" / "geometry" / "_diagnostics" / "weld_axis_seeds_sheet.png"


def main():
    holes = {}
    for r in csv.DictReader(open(REPO_ROOT / "configs" / "hole_seeds.csv")):
        holes.setdefault(r["capture_id"], {})[r["hole"]] = (int(r["px_u"]), int(r["px_v"]))
    rows, tiles = [], []
    for cid in sorted(holes):
        cap = RAW / cid
        pngs = sorted(glob.glob(str(cap / "color" / "*.png")))
        if not pngs or not (cap / "meta.json").exists():
            print(f"{cid}: no colour frames, skipped")
            continue
        meta = json.loads((cap / "meta.json").read_text())
        ci = meta["color_intrinsics"]
        img = np.median(np.stack([cv2.imread(p) for p in pngs[:5]]), 0).astype(np.uint8)
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float), (5, 5), 0)
        tex = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_64F)), (15, 15))
        score_img = cv2.blur(g, (61, 1)) * cv2.blur(tex, (61, 1))
        (u6, v6), (u8, v8) = holes[cid]["d6"], holes[cid]["d8"]
        px_per_mm = ci["fy"] / meta.get("standoff_mm_from_depth", 320.0)
        # One common offset from the hole line, scored jointly at both
        # columns (the weld runs ~parallel to the hole line), then each point
        # refined by +-12 px.
        us = [int(round(u6 + f * (u8 - u6))) for f in (0.3, 0.7)]
        vh = [v6 + f * (v8 - v6) for f in (0.3, 0.7)]
        best = None
        for side in (-1, 1):
            for d in range(int(35 * px_per_mm), int(65 * px_per_mm)):
                vs = [int(round(v + side * d)) for v in vh]
                if all(20 <= v < img.shape[0] - 20 for v in vs):
                    sc = sum(score_img[v - 20:v + 20, u].mean() for u, v in zip(us, vs))
                    if best is None or sc > best[0]:
                        best = (sc, vs)
        pts = []
        for name, u, v0 in zip("ab", us, best[1]):
            v = max(range(v0 - 12, v0 + 13), key=lambda vv: score_img[vv - 20:vv + 20, u].mean())
            pts.append((name, u, v))
            rows.append({"capture_id": cid, "point": name, "px_u": u,
                         "px_v": v, "source": "auto_proposed"})
        tile = img.copy()
        cv2.line(tile, pts[0][1:], pts[1][1:], (0, 255, 255), 2)
        for _, u, v in pts:
            cv2.circle(tile, (u, v), 10, (0, 0, 255), 2)
        for (u, v) in holes[cid].values():
            cv2.circle(tile, (u, v), 14, (255, 0, 0), 2)
        cv2.putText(tile, cid, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        tiles.append(cv2.resize(tile, (640, 360)))
        print(f"{cid}: a=({pts[0][1]},{pts[0][2]}) b=({pts[1][1]},{pts[1][2]})")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["capture_id", "point", "px_u", "px_v", "source"])
        w.writeheader()
        w.writerows(rows)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEET), sheet)
    print(f"-> {OUT_CSV.relative_to(REPO_ROOT)}\n-> {SHEET.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
