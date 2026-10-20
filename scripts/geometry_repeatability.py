"""
Repeatability of the geometry measurands across re-placement repeats.

Registers each repeat (fixed plate normal, registration_v2.fit_plate_normal),
measures every analysed cell two ways, and reports per-cell SD across repeats
and the pooled SD:

  before  binned per-profile extractor (measure.measure_capture, datum "linear")
          -- 0.5 mm bins, 5-bin smoothing, max/p95 over 10 profiles
  after   local surface fits (measure.localfit_cell)

Also tests the spatial-independence assumption directly on the parent plate:
SD of the repeat-to-repeat difference of window means, against window size,
compared with the 1/sqrt(N) an independent-noise model predicts.

Usage:
    python scripts/geometry_repeatability.py C05_top_r0 C05_top_r1 C05_top_r2
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
warnings.simplefilter("ignore", RuntimeWarning)

from src.geometry import measure  # noqa: E402
from src.geometry.registration_seeded import (  # noqa: E402
    load_seeds, register_capture_seeded,
)

RAW = REPO_ROOT / "data" / "raw" / "raw" / "S1"
OUT = REPO_ROOT / "results" / "geometry" / "_diagnostics"


def hole_s(pose, meta):
    tc = lambda P: pose.R.T @ (np.asarray(P) - pose.O) * 1000  # noqa: E731
    return (tc(meta["holes"]["origin"]["position_3d"])[0],
            tc(meta["holes"]["axis"]["position_3d"])[0])


def main():
    caps = sys.argv[1:]
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "geometry.yaml"))
    seeds = load_seeds(REPO_ROOT / "configs" / "hole_seeds.csv")
    axis = measure.load_weld_axis_seeds(REPO_ROOT / "configs" / "weld_axis_seeds.csv")
    use_axis = "--crown" not in sys.argv
    caps = [c for c in caps if not c.startswith("--")]
    before, after, grids = {}, {}, {}
    for cid in caps:
        pose, cloud, meta = register_capture_seeded(RAW / cid, seeds[cid])
        s6, s8 = hole_s(pose, meta)
        cl = None
        if use_axis:
            ci = json.loads((RAW / cid / "meta.json").read_text())["color_intrinsics"]
            cl = measure.seed_centre_line(pose, ci, axis[cid])
        cells, _, _, cf, _ = measure.measure_capture(cloud, pose, s6, s8, cfg,
                                                     centre_line=cl)
        print(f"  centre line ({cf['source']}): slope {cf['slope']:.4f}, "
              f"y0 {cf['intercept_mm']:.2f} mm", flush=True)
        h = measure.height_mm(cloud, pose)
        keep = np.abs(cloud.y) < 60
        s, y, h = cloud.s[keep], cloud.y[keep], h[keep]
        yr = y - (cf["slope"] * s + cf["intercept_mm"])
        before[cid] = {c["cell_index"]: c["linear"] for c in cells}
        after[cid] = {}
        for c in cells:
            L = c["linear"]
            after[cid][c["cell_index"]] = measure.localfit_cell(
                s, yr, h, c["s_start_mm"], c["s_end_mm"],
                L["toe_left_mm"], L["toe_right_mm"], cfg)
        # Plane-detrended parent-plate grid, 0.5 mm (y) x 1 mm (s).
        m = (s > 25) & (s < 275) & (np.abs(yr) > 15) & (np.abs(yr) < 45)
        A = np.column_stack([s[m], yr[m], np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(A, h[m], rcond=None)
        res = h[m] - A @ coef
        g, n = np.zeros((180, 250)), np.zeros((180, 250))
        iy = ((yr[m] + 45) / 0.5).astype(int)
        ix = (s[m] - 25).astype(int)
        ok = (iy >= 0) & (iy < 180) & (ix >= 0) & (ix < 250)
        np.add.at(g, (iy[ok], ix[ok]), res[ok])
        np.add.at(n, (iy[ok], ix[ok]), 1)
        grids[cid] = np.where(n > 0, g / np.maximum(n, 1), np.nan)
        print(f"{cid}: registered, {len(cells)} cells, tilt "
              f"{np.degrees(np.arccos(abs(pose.R[2, 2]))):.2f} deg", flush=True)

    report = {"captures": caps, "measurands": {}}
    print("\nPER-CELL SD ACROSS REPEATS (mm); pooled = sqrt(mean variance)")
    pairs = [
        ("misalignment", "misalignment_mm", "misalignment_mm"),
        ("misalignment (median of profiles)", "misalignment_median_mm", None),
        ("reinforcement", "reinforcement_mm", "reinforcement_signed_mm"),
        ("reinforcement (median of profiles)", "reinforcement_median_mm", None),
        ("undercut", "undercut_mm", "undercut_signed_mm"),
    ]
    cell_ids = sorted(set.intersection(*[set(before[c]) for c in caps]))
    for name, kb, ka in pairs:
        row = {}
        for label, src, key in (("before", before, kb), ("after", after, ka)):
            if key is None:
                continue
            vals = np.array([[np.nan if src[c].get(i) is None else src[c][i][key]
                              for c in caps] for i in cell_ids], float)
            sd = np.nanstd(vals, axis=1, ddof=1)
            row[label] = {"pooled_sd_mm": float(np.sqrt(np.nanmean(sd ** 2))),
                          "per_cell_sd_mm": [round(float(v), 3) for v in sd],
                          "cell_means_mm": [round(float(v), 3) for v in np.nanmean(vals, 1)]}
        report["measurands"][name] = row
        txt = "  ".join(f"{k} pooled {v['pooled_sd_mm']:.3f}" for k, v in row.items())
        print(f"  {name:<36} {txt}")
        for k, v in row.items():
            print(f"      {k:<6} per-cell {v['per_cell_sd_mm']}   means {v['cell_means_mm']}")

    pts = [after[c][i] for c in caps for i in cell_ids if after[c].get(i)]
    report["points_per_fit_min"] = {k: int(min(p[k] for p in pts))
                                    for k in ("n_datum", "n_crown", "n_toe_min")}
    print(f"\npoints per fit (minimum over cells): {report['points_per_fit_min']}")

    # Spatial independence test on the parent plate.
    print("\nPLATE WINDOW TEST: single-capture sigma of a window mean vs size")
    print("  window (y x s mm)   N cells   sigma_meas   sigma_if_independent")
    base = None
    curve = []
    for k in (1, 2, 4, 8, 16):
        diffs = []
        for a in range(len(caps)):
            for b in range(a + 1, len(caps)):
                A, B = grids[caps[a]], grids[caps[b]]
                hh, ww = (A.shape[0] // k) * k, (A.shape[1] // k) * k
                ma = np.nanmean(A[:hh, :ww].reshape(hh // k, k, ww // k, k), axis=(1, 3))
                mb = np.nanmean(B[:hh, :ww].reshape(hh // k, k, ww // k, k), axis=(1, 3))
                d = (ma - mb).ravel()
                diffs.append(d[np.isfinite(d)])
        sig = float(np.std(np.concatenate(diffs)) / np.sqrt(2))
        base = base or sig
        indep = base / np.sqrt(k * k)
        curve.append({"window_y_mm": 0.5 * k, "window_s_mm": 1.0 * k,
                      "n_cells": k * k, "sigma_mm": sig, "sigma_if_independent_mm": indep})
        print(f"  {0.5 * k:4.1f} x {1.0 * k:4.1f}         {k * k:5d}      {sig:.3f}        {indep:.3f}")
    report["plate_window_test"] = curve

    out = OUT / "repeatability_" + "_".join(caps) + ".json" if False else \
        OUT / f"repeatability_{caps[0].rsplit('_', 1)[0]}_{'axisseeds' if use_axis else 'crown'}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\n-> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
