"""
Phase 0 of the geometry task: test the tilt hypothesis. READ-ONLY -- applies no
correction and changes no pipeline output.

Hypothesis: the plate is tilted ~19 deg to the image plane, which would make the
step height read t/cos(theta) = 10.57 mm and a 2D baseline read
270*cos(theta) = 256 mm.

Per capture:
  Test 1  fit a 3D plane (RANSAC, then SVD on inliers) to the deprojected plate
          surface and to the board around it; report the angle between each
          normal and the optical axis. plate_isolation.py does NOT do this -- it
          fits a single constant board depth -- so this is new measurement.
  Test 3  rerun the seeded rim fit and report ellipse minor/major per hole.
  Also    perpendicular plate-to-board separation (plate plane to board plane,
          along the plate normal) against the axial step the existing
          scale_independent_checks.py reports (z_board - median plate z).

Usage:
    python scripts/tilt_check.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.geometry.plate_isolation import (  # noqa: E402
    create_plate_mask_from_height, fit_board_depth_ransac,
)
from src.geometry.registration_seeded import (  # noqa: E402
    detect_hole_from_seed, load_seeds,
)
from src.geometry.registration_v2 import (  # noqa: E402
    compute_averaged_depth, compute_median_colour, load_config,
)

RAW = REPO_ROOT / "data" / "raw" / "raw" / "S1"
OUT = REPO_ROOT / "results" / "geometry" / "_diagnostics"
RANSAC_TOL_MM = 0.5
RANSAC_ITERS = 400
SUBSAMPLE = 4


def deproject(depth_mm, mask, intr):
    v, u = np.nonzero(mask)
    v, u = v[::SUBSAMPLE], u[::SUBSAMPLE]
    z = depth_mm[v, u]
    ok = np.isfinite(z) & (z > 0)
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - intr["ppx"]) / intr["fx"] * z
    y = (v - intr["ppy"]) / intr["fy"] * z
    return np.column_stack([x, y, z])


def fit_plane(pts, rng):
    """RANSAC then SVD refit. Returns (unit normal with n_z > 0, centroid,
    inlier fraction, inlier RMS mm)."""
    best = None
    for _ in range(RANSAC_ITERS):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - s[0]) @ n) < RANSAC_TOL_MM
        if best is None or inl.sum() > best.sum():
            best = inl
    p = pts[best]
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    rms = float(np.sqrt(np.mean(((p - c) @ n) ** 2)))
    return n, c, float(best.mean()), rms


def angle_to_axis(n):
    return float(np.degrees(np.arccos(np.clip(abs(n[2]), -1, 1))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds(REPO_ROOT / "configs" / "hole_seeds.csv")
    config = load_config()
    rng = np.random.default_rng(0)
    rows = []
    for cap in sorted(p for p in RAW.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((cap / "meta.json").read_text())
        intr = meta["depth_intrinsics"]
        try:
            depth_mm, _, _ = compute_averaged_depth(cap / "depth",
                                                    meta.get("depth_units", 1e-4))
        except ValueError as e:
            rows.append({"capture_id": cap.name, "plane_error": repr(e)})
            print(cap.name, "SKIPPED:", e, flush=True)
            continue
        region = create_plate_mask_from_height(depth_mm)
        plate = region.mask.astype(bool)
        valid = np.isfinite(depth_mm) & (depth_mm > 0)
        board = valid & ~plate
        # Keep away from the plate edge, where the step itself would be fitted.
        import cv2
        k = np.ones((41, 41), np.uint8)
        board &= cv2.erode((~plate).astype(np.uint8), k).astype(bool)
        plate_core = cv2.erode(plate.astype(np.uint8), k).astype(bool)

        r = {"capture_id": cap.name}
        try:
            pp = deproject(depth_mm, plate_core, intr)
            n_p, c_p, f_p, rms_p = fit_plane(pp, rng)
            r.update(theta_plate_deg=angle_to_axis(n_p), plate_inlier_frac=f_p,
                     plate_rms_mm=rms_p,
                     plate_normal=[round(float(a), 5) for a in n_p])
            bp = deproject(depth_mm, board, intr)
            if len(bp) > 1000:
                n_b, c_b, f_b, rms_b = fit_plane(bp, rng)
                r.update(theta_board_deg=angle_to_axis(n_b),
                         board_inlier_frac=f_b, board_rms_mm=rms_b,
                         plate_board_normal_angle_deg=float(np.degrees(np.arccos(
                             np.clip(n_p @ n_b, -1, 1)))))
                # Perpendicular separation: board plane to plate centroid,
                # along the plate normal.
                r["t_perp_mm"] = float(abs((c_p - c_b) @ n_b))
            z_board, _ = fit_board_depth_ransac(depth_mm)
            r["t_axial_mm"] = float(z_board - np.nanmedian(depth_mm[plate]))
        except Exception as e:  # report, never hide
            r["plane_error"] = repr(e)

        cid = cap.name
        if cid in seeds:
            colour = compute_median_colour(cap / "color")
            fx = meta["color_intrinsics"]["fx"]
            standoff = meta.get("standoff_mm_from_depth", 320.0)
            for s in seeds[cid]:
                try:
                    _, rim = detect_hole_from_seed(colour, s, standoff, fx, config)
                    if rim is not None:
                        maj, mnr = rim.free_fit_axes
                        r[f"{s.hole_type}_minor_over_major"] = float(mnr / maj)
                        r[f"{s.hole_type}_n_edge"] = int(rim.n_edge_points)
                except Exception as e:
                    r[f"{s.hole_type}_rim_error"] = repr(e)
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in r.items() if k != "plate_normal"}, flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(OUT / "tilt_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {(OUT / 'tilt_check.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
