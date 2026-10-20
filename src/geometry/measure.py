"""
Pose-aware per-cell weld geometry measurement on a registered point cloud.

Replaces the pose-blind `profile.extract_single_profile` (centreline hardcoded
to `n_rows // 2`, weld assumed vertical in the image) for the seeded pipeline.
Input is the coupon-frame PointCloud from `registration_seeded.register_capture_seeded`.

Reference surface: PLATE ONLY. The board never enters the measurement chain.
Each transverse profile gets a per-side linear datum fitted over the parent-
metal band (configs/geometry.yaml profile_extraction.plane_fit_roi, 15-40 mm
from the bead centre), evaluated at the toe (bead_roi.y_half_width_mm). Two
alternative datums are computed alongside for comparison only:

  "plane"   per-side plane fitted over the whole capture's parent band
  "median"  per-profile per-side level (legacy datum_mode: median)

Height convention: positive TOWARD THE CAMERA. registration_v2.solve_coupon_pose
sets the coupon z axis to +camera-z on top faces and -camera-z on root faces,
so its z is inverted on top faces; height is taken from camera depth instead.

Bead centre: located from the data per cell (crown of the cell's median
profile), then fitted linearly along s. The pose's y = 0 is defined by the
datum holes at y = 50 mm and inherits the lateral scale error recorded in
results/geometry/LIMITATIONS.md, so it is not trusted to place the bands.

Cell boundaries are proportional to the DETECTED hole span:
    s(s_nom) = p6 + (p8 - p6) * (s_nom - s6_nom) / (s8_nom - s6_nom)
which keeps the 25 mm / 12-cell scheme of docs/03-data-and-labels.md while any
uniform lateral scale error cancels.
"""
from __future__ import annotations

import numpy as np

BIN_MM = 0.5
SLAB_HALF_MM = 0.5
Y_SPAN_MM = 45.0
TOE_THRESHOLD_MM = 0.1      # as in the legacy extractor
TOE_SEARCH_MM = 2.0         # as in the legacy extractor
CENTRE_SEARCH_MM = 20.0
DATUMS = ("linear", "plane", "median")


def height_mm(cloud, pose):
    """Height toward the camera, mm, from coupon-frame z and the pose's z axis."""
    sign = np.sign(pose.R[2, 2]) if pose.R[2, 2] != 0 else 1.0
    return -sign * cloud.z


def cell_edges(s6, s8, geom_cfg):
    """Proportional cell boundaries in the detected-hole frame."""
    reg = geom_cfg["registration"]
    cells = geom_cfg["cells"]
    s6n, s8n = reg["origin_hole"]["s_mm"], reg["axis_hole"]["s_mm"]
    w, L = cells["width_mm"], cells["coupon_length_mm"]
    n = int(round(L / w))
    nominal = np.arange(n + 1) * w
    return s6 + (s8 - s6) * (nominal - s6n) / (s8n - s6n), n


def binned_profile(s, y, h, s_centre, y_centre=0.0):
    """Median height per BIN_MM bin of (y - y_centre), from a 1 mm slab."""
    m = np.abs(s - s_centre) <= SLAB_HALF_MM
    yy, hh = y[m] - y_centre, h[m]
    edges = np.arange(-Y_SPAN_MM, Y_SPAN_MM + BIN_MM, BIN_MM)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(yy, edges) - 1
    out = np.full(len(centres), np.nan)
    ok = (idx >= 0) & (idx < len(centres))
    for k in np.unique(idx[ok]):
        out[k] = np.median(hh[ok][idx[ok] == k])
    return centres, out


def smooth(v, k):
    if k <= 1:
        return v
    pad = k // 2
    vv = np.pad(v, pad, constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(vv, k)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(win, axis=1)


def _robust_line(x, v):
    """Least-squares line with one 3-MAD rejection pass. Returns (a, b, rms, n)."""
    ok = np.isfinite(v)
    x, v = x[ok], v[ok]
    if len(x) < 5 or np.ptp(x) < 1.0:
        return None
    a, b = np.polyfit(x, v, 1)
    r = v - (a * x + b)
    mad = np.median(np.abs(r - np.median(r))) * 1.4826
    keep = np.abs(r) <= max(3 * mad, 1e-6)
    if keep.sum() >= 5:
        a, b = np.polyfit(x[keep], v[keep], 1)
        r = v[keep] - (a * x[keep] + b)
    return float(a), float(b), float(np.sqrt(np.mean(r ** 2))), int(keep.sum())


def fit_side_plane(s, y, h, side, cfg, rng, tol=0.5, iters=300):
    """RANSAC plane h = a*s + b*y + c over one side's parent band."""
    roi = cfg["profile_extraction"]["plane_fit_roi"]
    m = (side * y >= roi["y_inner_mm"]) & (side * y <= roi["y_outer_mm"]) & np.isfinite(h)
    P = np.column_stack([s[m], y[m], h[m]])
    if len(P) < 100:
        return None
    best = None
    for _ in range(iters):
        q = P[rng.choice(len(P), 3, replace=False)]
        A = np.column_stack([q[:, 0], q[:, 1], np.ones(3)])
        try:
            coef = np.linalg.solve(A, q[:, 2])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(P[:, 0] * coef[0] + P[:, 1] * coef[1] + coef[2] - P[:, 2]) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]
    A = np.column_stack([Q[:, 0], Q[:, 1], np.ones(len(Q))])
    coef, *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)
    res = Q[:, 2] - A @ coef
    n = np.array([-coef[0], -coef[1], 1.0])
    return {"coef": coef.tolist(), "normal": (n / np.linalg.norm(n)).tolist(),
            "inlier_frac": float(best.mean()), "rms_mm": float(np.sqrt(np.mean(res ** 2))),
            "n_points": int(len(P))}


def measure_profile(yc, hp, cfg, datum, plane_lines=None):
    """
    Measurements from one binned profile (heights toward camera, y centred).

    datum: "linear" | "median" | "plane". For "plane", plane_lines gives
    {side: (a, b)} -- the per-side plane evaluated along this profile.
    Returns dict, NaN where the datum could not be formed.
    """
    pe = cfg["profile_extraction"]
    yi, yo = pe["plane_fit_roi"]["y_inner_mm"], pe["plane_fit_roi"]["y_outer_mm"]
    half = pe["bead_roi"]["y_half_width_mm"]
    hp = smooth(hp, pe["smoothing_kernel_px"])
    nan = dict(undercut_left=np.nan, undercut_right=np.nan, reinforcement=np.nan,
               misalignment=np.nan, bead_width=np.nan, datum_rms=np.nan,
               toe_left=np.nan, toe_right=np.nan, datum_left=np.nan, datum_right=np.nan)

    lines = {}
    for side in (-1, 1):
        band = (side * yc >= yi) & (side * yc <= yo)
        if datum == "plane":
            if plane_lines is None or plane_lines.get(side) is None:
                return nan
            a, b = plane_lines[side]
            v = hp[band] - (a * yc[band] + b)
            rms = float(np.sqrt(np.nanmean(v ** 2))) if np.isfinite(v).sum() else np.nan
            lines[side] = (a, b, rms)
        elif datum == "median":
            v = hp[band][np.isfinite(hp[band])]
            if len(v) < 5:
                return nan
            lv = float(np.median(v))
            lines[side] = (0.0, lv, float(np.sqrt(np.mean((v - lv) ** 2))))
        else:
            fit = _robust_line(yc[band], hp[band])
            if fit is None:
                return nan
            lines[side] = fit[:3]

    def ref(side, y):
        a, b, _ = lines[side]
        return a * y + b

    dL, dR = ref(-1, -half), ref(1, half)
    out = dict(datum_left=dL, datum_right=dR, misalignment=abs(dL - dR),
               datum_rms=float(np.mean([lines[-1][2], lines[1][2]])))

    # Reinforcement: max height above the line joining the two toe datums.
    bead = np.abs(yc) <= half
    joint = dL + (dR - dL) * (yc + half) / (2 * half)
    above = hp[bead] - joint[bead]
    out["reinforcement"] = (max(0.0, float(np.nanmax(above)))
                            if np.isfinite(above).any() else np.nan)

    # Toes: walk outward from the centre until height above that side's datum
    # drops below TOE_THRESHOLD_MM; default to the bead ROI edge.
    for side, key in ((-1, "left"), (1, "right")):
        hrel = hp - ref(side, yc)
        order = np.argsort(side * yc)
        toe = side * half
        for k in order:
            if side * yc[k] < 0 or side * yc[k] > yi:
                continue
            if np.isfinite(hrel[k]) and hrel[k] < TOE_THRESHOLD_MM:
                toe = yc[k]
                break
        out[f"toe_{key}"] = float(toe)
        win = np.abs(yc - toe) <= TOE_SEARCH_MM
        below = ref(side, yc[win]) - hp[win]
        out[f"undercut_{key}"] = (max(0.0, float(np.nanmax(below)))
                                  if np.isfinite(below).any() else np.nan)
    out["bead_width"] = out["toe_right"] - out["toe_left"]
    return out


def load_weld_axis_seeds(path):
    """{capture_id: [(px_u, px_v), (px_u, px_v)]} from configs/weld_axis_seeds.csv."""
    import csv
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["capture_id"], []).append((float(r["px_u"]), float(r["px_v"])))
    return out


def seed_centre_line(pose, colour_intr, pts_px):
    """
    Weld centreline y_c(s) = slope*s + intercept in the coupon frame, from two
    colour-pixel seeds: each pixel ray is intersected with the fitted plate
    plane (normal pose.R[:,2], through pose.O), then mapped with R.T.
    """
    n, O = pose.R[:, 2], pose.O
    sy = []
    for u, v in pts_px:
        d = np.array([(u - colour_intr["ppx"]) / colour_intr["fx"],
                      (v - colour_intr["ppy"]) / colour_intr["fy"], 1.0])
        X = d * (n @ O) / (n @ d)
        c = pose.R.T @ (X - O) * 1000.0
        sy.append((c[0], c[1]))
    (s1, y1), (s2, y2) = sy
    slope = (y2 - y1) / (s2 - s1)
    return {"slope": float(slope), "intercept_mm": float(y1 - slope * s1),
            "source": "weld_axis_seeds", "seed_points_sy_mm": [list(map(float, p)) for p in sy]}


def measure_capture(cloud, pose, s6, s8, geom_cfg, rng=None, centre_line=None):
    """
    All cells of one registered capture.

    Returns (cells, profiles, planes, centre_fit):
      cells     list of per-cell dicts (aggregated, per datum)
      profiles  dict of arrays for profiles.npz
      planes    per-side plate plane fits
    """
    rng = rng or np.random.default_rng(0)
    pe = geom_cfg["profile_extraction"]
    n_prof, q = pe["profiles_per_cell"], pe["aggregation_quantile"]
    s, y, h = cloud.s, cloud.y, height_mm(cloud, pose)
    keep = np.abs(y) < 60
    s, y, h = s[keep], y[keep], h[keep]

    edges, n_cells = cell_edges(s6, s8, geom_cfg)
    first_ex = int(round(geom_cfg["cells"]["first_excluded_mm"] / geom_cfg["cells"]["width_mm"]))
    last_ex = int(round(geom_cfg["cells"]["last_excluded_mm"] / geom_cfg["cells"]["width_mm"]))
    analysed = list(range(first_ex, n_cells - last_ex))

    # Bead centre per cell: crown of the cell's median raw profile.
    centres_s, centres_y = [], []
    for i in analysed:
        sc = 0.5 * (edges[i] + edges[i + 1])
        prof = [binned_profile(s, y, h, ss)[1]
                for ss in np.linspace(edges[i] + 2, edges[i + 1] - 2, 5)]
        yc, _ = binned_profile(s, y, h, sc)
        med = np.nanmedian(np.vstack(prof), axis=0)
        # Detrend with the outer bands so a sloping plate cannot move the crown.
        line = _robust_line(yc[np.abs(yc) >= 25], med[np.abs(yc) >= 25])
        rel = med - (line[0] * yc + line[1]) if line else med
        rel = smooth(rel, 11)
        win = np.abs(yc) <= CENTRE_SEARCH_MM
        if np.isfinite(rel[win]).any():
            centres_s.append(sc)
            centres_y.append(float(yc[win][np.nanargmax(rel[win])]))
    if len(centres_s) >= 2:
        ca, cb = np.polyfit(centres_s, centres_y, 1)
    else:
        ca, cb = 0.0, float(np.median(centres_y)) if centres_y else 0.0
    centre_fit = {"slope": float(ca), "intercept_mm": float(cb),
                  "cell_crowns_y_mm": centres_y, "source": "crown_argmax"}
    if centre_line is not None:
        # Seeds override the data-driven crown, which fails where the depth
        # reads the bead as a dip (results/geometry/LIMITATIONS.md, L2).
        centre_fit = dict(centre_line, crown_argmax_fit={
            "slope": float(ca), "intercept_mm": float(cb),
            "cell_crowns_y_mm": centres_y})
        ca, cb = centre_line["slope"], centre_line["intercept_mm"]
    yr = y - (ca * s + cb)

    span = (s >= edges[analysed[0]]) & (s <= edges[analysed[-1] + 1])
    planes = {side: fit_side_plane(s[span], yr[span], h[span], side, geom_cfg, rng)
              for side in (-1, 1)}
    if planes[-1] and planes[1]:
        n1, n2 = np.array(planes[-1]["normal"]), np.array(planes[1]["normal"])
        planes["half_plate_angle_deg"] = float(np.degrees(np.arccos(np.clip(n1 @ n2, -1, 1))))

    cells, prof_rows = [], []
    for i in analysed:
        rows = {d: [] for d in DATUMS}
        for ss in np.linspace(edges[i] + 1, edges[i + 1] - 1, n_prof):
            yc, hp = binned_profile(s, yr, h, ss)
            plane_lines = {}
            for side in (-1, 1):
                p = planes.get(side)
                plane_lines[side] = (None if p is None else
                                     (p["coef"][1], p["coef"][0] * ss + p["coef"][2]))
            for d in DATUMS:
                r = measure_profile(yc, hp, geom_cfg, d, plane_lines)
                rows[d].append(r)
            prof_rows.append((i, ss, hp))
        cell = {"cell_index": i, "s_start_mm": float(edges[i]),
                "s_end_mm": float(edges[i + 1])}
        for d in DATUMS:
            R = rows[d]
            valid = [r for r in R if np.isfinite(r["misalignment"])]
            frac = len(valid) / n_prof
            def agg(key, f):
                v = [r[key] for r in valid if np.isfinite(r[key])]
                return float(f(v)) if v else np.nan
            ucl = agg("undercut_left", lambda v: np.percentile(v, 100 * q))
            ucr = agg("undercut_right", lambda v: np.percentile(v, 100 * q))
            cell[d] = {
                "valid_fraction": frac,
                "undercut_left_mm": ucl, "undercut_right_mm": ucr,
                "undercut_mm": float(np.nanmax([ucl, ucr])) if np.isfinite([ucl, ucr]).any() else np.nan,
                "governing_toe": ("left" if (np.nan_to_num(ucl, nan=-1) >= np.nan_to_num(ucr, nan=-1)) else "right"),
                "reinforcement_mm": agg("reinforcement", np.max),
                "reinforcement_median_mm": agg("reinforcement", np.median),
                "misalignment_mm": agg("misalignment", np.max),
                "misalignment_median_mm": agg("misalignment", np.median),
                "bead_width_mm": agg("bead_width", np.median),
                "datum_rms_mm": agg("datum_rms", np.median),
                "toe_left_mm": agg("toe_left", np.median),
                "toe_right_mm": agg("toe_right", np.median),
            }
        cells.append(cell)

    profiles = {
        "cell_index": np.array([r[0] for r in prof_rows]),
        "s_mm": np.array([r[1] for r in prof_rows]),
        "y_mm": binned_profile(s, yr, h, 0.0)[0],
        "height_mm": np.vstack([r[2] for r in prof_rows]) if prof_rows else np.empty((0, 0)),
    }
    return cells, profiles, planes, centre_fit, edges


# ---------------------------------------------------------------------------
# Local surface fits (2026-10-20). Every quantity comes from a least-squares
# fit over tens to hundreds of points, never a single bin or pixel extreme.
# ---------------------------------------------------------------------------
CROWN_HALF_MM = 3.0
TOE_HALF_MM = 2.0
SEGMENT_MM = 5.0
MIN_TOE_POINTS = 20   # an under-determined quadratic extrapolates wildly


def _lstsq_reject(A, v):
    """Least squares with one 3-MAD rejection pass. Returns (coef, rms, n)."""
    if len(v) < A.shape[1] + 3:
        return None
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    r = v - A @ coef
    mad = np.median(np.abs(r - np.median(r))) * 1.4826
    keep = np.abs(r) <= max(3 * mad, 1e-6)
    if keep.sum() >= A.shape[1] + 3:
        coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        r = v[keep] - A[keep] @ coef
    return coef, float(np.sqrt(np.mean(r ** 2))), int(keep.sum())


def localfit_cell(s, yr, h, s0, s1, toe_left, toe_right, cfg):
    """
    Local-fit measurements for one cell (heights toward camera, y centred on
    the detected bead). Returns signed values so repeatability is not hidden by
    clipping at zero; undercut_mm / reinforcement_mm are the clipped forms.
    """
    pe = cfg["profile_extraction"]
    yi, yo = pe["plane_fit_roi"]["y_inner_mm"], pe["plane_fit_roi"]["y_outer_mm"]
    half = pe["bead_roi"]["y_half_width_mm"]
    sc = 0.5 * (s0 + s1)
    cell = (s >= s0) & (s < s1) & np.isfinite(h)
    out = {"n_datum": 0, "n_crown": 0, "n_toe_min": 0}

    planes = {}
    for side in (-1, 1):
        m = cell & (side * yr >= yi) & (side * yr <= yo)
        A = np.column_stack([s[m] - sc, yr[m], np.ones(m.sum())])
        fit = _lstsq_reject(A, h[m])
        if fit is None:
            return None
        planes[side] = fit[0]
        out[f"datum_rms_{'left' if side < 0 else 'right'}_mm"] = fit[1]
        out["n_datum"] += fit[2]

    def ref(side, ss, yy):
        a, b, c = planes[side]
        return a * (ss - sc) + b * yy + c

    dL, dR = ref(-1, sc, -half), ref(1, sc, half)
    out["misalignment_mm"] = float(abs(dL - dR))

    # Crown: quadratic in y (+ linear in s) over the bead centre.
    m = cell & (np.abs(yr) <= CROWN_HALF_MM)
    A = np.column_stack([np.ones(m.sum()), yr[m], yr[m] ** 2, s[m] - sc])
    fit = _lstsq_reject(A, h[m])
    if fit is None:
        return None
    a, b, c, _ = fit[0]
    ys = -b / (2 * c) if (c < 0 and abs(-b / (2 * c)) <= CROWN_HALF_MM) else 0.0
    joint = dL + (dR - dL) * (ys + half) / (2 * half)
    out["reinforcement_signed_mm"] = float(a + b * ys + c * ys ** 2 - joint)
    out["reinforcement_mm"] = max(0.0, out["reinforcement_signed_mm"])
    out["n_crown"] = fit[2]

    # Undercut: quadratic in y around each toe, per 5 mm segment along s;
    # depth below that side's datum at the fitted minimum within the window.
    worst, n_min = -np.inf, np.inf
    for side, toe in ((-1, toe_left), (1, toe_right)):
        if not np.isfinite(toe):
            continue
        for a0 in np.arange(s0, s1 - 1e-9, SEGMENT_MM):
            seg = cell & (s >= a0) & (s < a0 + SEGMENT_MM) & (np.abs(yr - toe) <= TOE_HALF_MM)
            A = np.column_stack([np.ones(seg.sum()), yr[seg] - toe, (yr[seg] - toe) ** 2])
            fit = _lstsq_reject(A, h[seg])
            if fit is None or fit[2] < MIN_TOE_POINTS:
                continue
            n_min = min(n_min, fit[2])
            yy = np.linspace(-TOE_HALF_MM, TOE_HALF_MM, 41)
            fitted = fit[0][0] + fit[0][1] * yy + fit[0][2] * yy ** 2
            depth = ref(side, a0 + SEGMENT_MM / 2, yy + toe) - fitted
            worst = max(worst, float(depth.max()))
    out["undercut_signed_mm"] = worst if np.isfinite(worst) else np.nan
    out["undercut_mm"] = max(0.0, worst) if np.isfinite(worst) else np.nan
    out["n_toe_min"] = int(n_min) if np.isfinite(n_min) else 0
    return out
