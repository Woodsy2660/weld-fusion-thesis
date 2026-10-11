"""
Independent witness for a RealSense capture.

Reads data/raw/<session>/<capture>/ directly from the .npy files and
deliberately imports NOTHING from src/. If this and the geometry pipeline
disagree, the disagreement is informative; that only holds while the two share
no code.

What it reports, per capture:
  depth scale and standoff actually achieved, against what meta.json claims
  valid fraction, single-frame noise, and the REAL averaging gain over 64
    frames (split-half), against the naive 1/sqrt(N)
  plane fits to the parent metal either side of the bead, separately and
    together, with the angular distortion between the two sides
  camera tilt
  transverse_profiles.png

Usage:
    python diagnose_capture.py ../data/raw/S1/C11_top_r0 [--out DIR]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# Plate geometry, from weld-docs/docs/03-data-and-labels.md. Used only to
# bound the search for the coupon within the scene -- no measurement depends
# on these being exact.
PLATE_LEN_MM = 300.0
PLATE_WID_MM = 140.0

# Parent-metal ROI either side of the weld centreline, from
# configs/geometry.yaml:profile_extraction.plane_fit_roi. Read here as
# literals because that config currently has unresolved merge conflicts and
# cannot be parsed; both values are echoed in the output so the coupling is
# visible.
ROI_Y_INNER_MM = 15.0
ROI_Y_OUTER_MM = 40.0

PLANE_INLIER_MM = 3.0
RANSAC_ITERS = 200
RANSAC_SEED = 0          # fixed: a diagnostic that moves between runs is useless


def sha256_16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def load_stack(capture: Path):
    meta = json.loads((capture / "meta.json").read_text())
    files = sorted((capture / "depth").glob("*.npy"))
    if not files:
        sys.exit(f"no depth frames in {capture/'depth'}")
    stack = np.stack([np.load(f) for f in files])          # (N, H, W) uint16
    return meta, stack, files


def deproject(mask, z_mm, K):
    ys, xs = np.nonzero(mask)
    z = z_mm[mask]
    X = (xs - K["ppx"]) * z / K["fx"]
    Y = (ys - K["ppy"]) * z / K["fy"]
    return np.c_[X, Y, z], ys, xs


def fit_plane(P):
    """Least-squares z = ax + by + c. Returns (coef, residuals, unit normal)."""
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    res = P[:, 2] - A @ coef
    n = np.array([-coef[0], -coef[1], 1.0])
    return coef, res, n / np.linalg.norm(n)


def ransac_plane(P, iters=RANSAC_ITERS, tol=PLANE_INLIER_MM, seed=RANSAC_SEED):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(P), 3, replace=False)
        A = np.c_[P[idx, 0], P[idx, 1], np.ones(3)]
        try:
            coef = np.linalg.solve(A, P[idx, 2])
        except np.linalg.LinAlgError:
            continue
        r = np.abs(P[:, 2] - (np.c_[P[:, 0], P[:, 1], np.ones(len(P))] @ coef))
        n_in = int((r < tol).sum())
        if best is None or n_in > best[0]:
            best = (n_in, coef)
    inliers = np.abs(P[:, 2] - (np.c_[P[:, 0], P[:, 1],
                                      np.ones(len(P))] @ best[1])) < tol
    return best[1], inliers


def angle_between(n1, n2):
    c = float(np.clip(abs(np.dot(n1, n2)), -1.0, 1.0))
    return np.degrees(np.arccos(c))


def analyse(capture: Path, out_dir: Path):
    meta, stack, files = load_stack(capture)
    K = meta["depth_intrinsics"]
    du_mm = meta["depth_units"] * 1000.0
    N, H, W = stack.shape

    r = {"capture": str(capture), "n_depth_frames": N,
         "depth_units_mm": du_mm,
         "standoff_mm_measured_in_meta": meta.get("standoff_mm_measured"),
         "usb_type": meta.get("usb_type"), "firmware": meta.get("firmware"),
         "sdk_version": meta.get("sdk_version"),
         "asic_temp_start_c": meta.get("asic_temperature_c_start"),
         "asic_temp_end_c": meta.get("asic_temperature_c_end"),
         "dropped_depth_frames": meta.get("dropped_depth_frames"),
         "post_processing_filters": meta.get("post_processing_filters")}

    z_all = stack.astype(np.float64) * du_mm
    valid = stack > 0
    r["valid_fraction_whole_frame"] = float(valid.mean())

    # ---- find the dominant near plane (the coupon), ignoring the floor ----
    z0 = z_all[0]
    v0 = valid[0]
    hist, edges = np.histogram(z0[v0], bins=np.arange(150, 1200, 10))
    mode_z = edges[int(np.argmax(hist))] + 5.0
    band = v0 & (np.abs(z0 - mode_z) < 40.0)
    r["modal_depth_mm"] = float(mode_z)

    P_band, ys_b, xs_b = deproject(band, z0, K)
    coef, inl = ransac_plane(P_band)
    P_in = P_band[inl]

    # Trim to a plate-sized window centred on the inlier centroid. The scene
    # contains jig and table at a similar depth; without this the "plane" is
    # fitted across all of it.
    cx, cy = np.median(P_in[:, 0]), np.median(P_in[:, 1])
    half_diag = 0.5 * np.hypot(PLATE_LEN_MM, PLATE_WID_MM) * 1.1
    keep = (np.abs(P_in[:, 0] - cx) < half_diag) & \
           (np.abs(P_in[:, 1] - cy) < half_diag)
    P_plate = P_in[keep]
    r["plate_candidate_points"] = int(len(P_plate))

    # ---- orient: long axis = s, short axis = transverse ----
    XY = P_plate[:, :2] - P_plate[:, :2].mean(0)
    _, _, Vt = np.linalg.svd(XY, full_matrices=False)
    e_s, e_t = Vt[0], Vt[1]
    s = XY @ e_s
    t = XY @ e_t
    r["extent_long_mm"] = float(s.max() - s.min())
    r["extent_transverse_mm"] = float(t.max() - t.min())

    # ---- median depth over the plate ----
    r["median_depth_mm"] = float(np.median(P_plate[:, 2]))
    r["lateral_scale_mm_per_px"] = float(r["median_depth_mm"] / K["fx"])
    r["lateral_scale_if_300mm"] = float(300.0 / K["fx"])

    # ---- locate the bead: transverse profile of height above the plane ----
    _, res_plate, n_plate = fit_plane(P_plate)
    r["camera_tilt_deg"] = float(angle_between(n_plate, np.array([0, 0, 1.0])))

    nb = 80
    tb = np.linspace(t.min(), t.max(), nb)
    idx = np.clip(np.digitize(t, tb) - 1, 0, nb - 1)
    prof = np.array([np.median(res_plate[idx == i]) if (idx == i).sum() > 40
                     else np.nan for i in range(nb)])
    if np.all(np.isnan(prof)):
        r["bead_found"] = False
        return r, None
    t_bead = float(tb[int(np.nanargmax(prof))])
    r["bead_found"] = True
    r["bead_transverse_position_mm"] = t_bead
    r["bead_peak_height_above_plane_mm"] = float(np.nanmax(prof))

    # ---- parent-metal planes either side ----
    dt = t - t_bead
    side_a = (dt > ROI_Y_INNER_MM) & (dt < ROI_Y_OUTER_MM)
    side_b = (dt < -ROI_Y_INNER_MM) & (dt > -ROI_Y_OUTER_MM)
    both = side_a | side_b
    r["roi_y_inner_mm"] = ROI_Y_INNER_MM
    r["roi_y_outer_mm"] = ROI_Y_OUTER_MM

    planes = {}
    for name, m in (("A", side_a), ("B", side_b), ("BOTH", both)):
        if m.sum() < 50:
            planes[name] = {"points": int(m.sum()), "rms_mm": None,
                            "normal": None,
                            "note": "too few points for a plane fit"}
            continue
        _, rr, nn = fit_plane(P_plate[m])
        planes[name] = {"points": int(m.sum()),
                        "rms_mm": float(np.sqrt(np.mean(rr ** 2))),
                        "normal": [float(x) for x in nn]}
    r["planes"] = planes

    if planes["A"]["normal"] and planes["B"]["normal"]:
        phi = angle_between(np.array(planes["A"]["normal"]),
                            np.array(planes["B"]["normal"]))
        r["angular_distortion_A_vs_B_deg"] = float(phi)
        r["implied_deviation_at_70mm_edge_mm"] = float(
            70.0 * np.tan(np.radians(phi)))

    # ---- temporal noise ----
    always = valid.all(axis=0)
    r["pixels_valid_in_all_frames"] = int(always.sum())
    if always.sum() > 1000:
        zs = z_all[:, always]                       # (N, M)
        per_px_std = zs.std(axis=0, ddof=1)
        r["median_per_pixel_std_mm"] = float(np.median(per_px_std))
        r["se_naive_mm"] = float(np.median(per_px_std) / np.sqrt(N))

        half = N // 2
        # Block halves: sensitive to drift through the capture.
        d_block = zs[:half].mean(0) - zs[half:].mean(0)
        se_block = float(np.median(np.abs(d_block)) * 1.4826 / 2.0)
        # Interleaved halves: cancels slow drift, isolates frame-to-frame noise.
        d_inter = zs[0::2].mean(0) - zs[1::2].mean(0)
        se_inter = float(np.median(np.abs(d_inter)) * 1.4826 / 2.0)
        r["se_split_half_block_mm"] = se_block
        r["se_split_half_interleaved_mm"] = se_inter
        r["achieved_reduction_factor_block"] = float(
            r["median_per_pixel_std_mm"] / se_block) if se_block > 0 else None
        r["achieved_reduction_factor_interleaved"] = float(
            r["median_per_pixel_std_mm"] / se_inter) if se_inter > 0 else None
        r["naive_reduction_factor"] = float(np.sqrt(N))

    figdata = {"t": t, "s": s, "res": res_plate, "t_bead": t_bead,
               "prof_t": tb, "prof": prof}
    return r, figdata


def plot_profiles(figdata, title, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, s, res = figdata["t"], figdata["s"], figdata["res"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))

    # Five transverse slices evenly along the weld axis.
    cuts = np.linspace(np.percentile(s, 10), np.percentile(s, 90), 5)
    for ax, s0 in zip(axes.ravel(), cuts):
        m = np.abs(s - s0) < 5.0
        if m.sum() < 50:
            ax.text(0.5, 0.5, "insufficient points", ha="center",
                    transform=ax.transAxes)
            continue
        order = np.argsort(t[m])
        ax.plot(t[m][order], res[m][order], ".", ms=1.6, color="#2C5F8A",
                alpha=0.5)
        ax.axvline(figdata["t_bead"], color="#A32E28", lw=1, ls="--")
        for sgn in (1, -1):
            ax.axvspan(figdata["t_bead"] + sgn * ROI_Y_INNER_MM,
                       figdata["t_bead"] + sgn * ROI_Y_OUTER_MM,
                       color="#2E6B4F", alpha=0.10)
        ax.set_title(f"s = {s0:.0f} mm  (n={m.sum()})", fontsize=9)
        ax.set_xlabel("transverse (mm)")
        ax.set_ylabel("height above plane (mm)")
        ax.grid(alpha=0.25)

    ax = axes.ravel()[5]
    ax.plot(figdata["prof_t"], figdata["prof"], "-", lw=2, color="#B8860B")
    ax.axvline(figdata["t_bead"], color="#A32E28", lw=1, ls="--")
    ax.set_title("median height vs transverse (bead locator)", fontsize=9)
    ax.set_xlabel("transverse (mm)")
    ax.set_ylabel("median height (mm)")
    ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--out", default="results/validation_C11")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    allr = {}
    for c in args.captures:
        cap = Path(c).resolve()
        print("=" * 74)
        print(cap.name)
        print("=" * 74)
        r, fig = analyse(cap, out_dir)
        allr[cap.name] = r
        for k, v in r.items():
            if k == "planes":
                for sname, p in v.items():
                    rms = "n/a" if p["rms_mm"] is None else f"{p['rms_mm']:.4f}"
                    print(f"  plane {sname:<5} rms {rms:>8} mm   "
                          f"points {p['points']}")
            elif isinstance(v, float):
                print(f"  {k:<42} {v:.4f}")
            else:
                print(f"  {k:<42} {v}")
        if fig is not None:
            png = out_dir / f"transverse_profiles_{cap.name}.png"
            plot_profiles(fig, cap.name, png)
            print(f"  -> {png.relative_to(repo)}")

    (out_dir / "part_a_diagnostic.json").write_text(json.dumps(allr, indent=1))
    print(f"\n-> {(out_dir/'part_a_diagnostic.json').relative_to(repo)}")


if __name__ == "__main__":
    main()
