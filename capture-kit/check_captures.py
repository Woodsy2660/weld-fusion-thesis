#!/usr/bin/env python3
"""
Run this BEFORE breaking down the rig. A failure now is a 5-minute redo;
a failure tomorrow is impossible to fix.

Usage:  python check_captures.py data/raw/S1
"""

import json, pathlib, sys
import numpy as np
import cv2

# Acceptance thresholds — tune on the first coupon, then freeze.
MIN_VALID_DEPTH_FRACTION = 0.60   # over the whole plate region
MAX_SATURATED_FRACTION   = 0.001  # colour pixels at 255 in any channel
MAX_STANDOFF_ERROR_MM = 10.0    # declared vs depth-derived
MAX_DROPPED_FRAMES       = 0
MAX_TEMP_RISE_C          = 6.0
EXPECTED_DEPTH_UNITS     = 0.0001


def check_one(d: pathlib.Path):
    issues, info = [], {}
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return ["no meta.json"], {}
    meta = json.loads(meta_path.read_text())

    if abs(meta["depth_units"] - EXPECTED_DEPTH_UNITS) > 1e-9:
        issues.append(f"depth_units={meta['depth_units']}")
    if not meta["usb_type"].startswith("3"):
        issues.append(f"usb={meta['usb_type']}")
    if meta.get("post_processing_filters"):
        issues.append("filters were applied")
    if meta["dropped_depth_frames"] > MAX_DROPPED_FRAMES:
        issues.append(f"{meta['dropped_depth_frames']} dropped frames")

    rise = meta["asic_temperature_c_end"] - meta["asic_temperature_c_start"]
    info["temp_rise"] = rise
    if rise > MAX_TEMP_RISE_C:
        issues.append(f"ASIC temp rose {rise:.1f}C within one capture")

    dfiles = sorted((d / "depth").glob("*.npy"))
    cfiles = sorted((d / "color").glob("*.png"))
    info["n_depth"], info["n_color"] = len(dfiles), len(cfiles)
    if len(dfiles) != meta["n_depth"]:
        issues.append(f"{len(dfiles)} depth files, expected {meta['n_depth']}")
    if len(cfiles) != meta["n_color"]:
        issues.append(f"{len(cfiles)} colour files, expected {meta['n_color']}")
    if not dfiles or not cfiles:
        return issues, info

    stack = np.stack([np.load(f) for f in dfiles]).astype(np.float32)
    stack[stack == 0] = np.nan                      # 0 means no return

    valid = np.mean(~np.isnan(stack))
    info["valid_fraction"] = valid
    if valid < MIN_VALID_DEPTH_FRACTION:
        issues.append(f"valid depth only {valid:.1%}")

    with np.errstate(invalid="ignore"):
        per_px_std = np.nanstd(stack, axis=0) * meta["depth_units"] * 1000.0  # mm
    med_std = float(np.nanmedian(per_px_std))
    info["median_per_pixel_std_mm"] = med_std

    mean_depth = np.nanmean(stack, axis=0) * meta["depth_units"] * 1000.0
    info["median_depth_mm"] = float(np.nanmedian(mean_depth))
    if not (250 < info["median_depth_mm"] < 360):
        issues.append(f"median depth {info['median_depth_mm']:.0f} mm — standoff wrong?")

    # Declared vs measured standoff. The C11 session recorded 300.0 because
    # that is what was typed on the command line while the rig sat at 360 mm;
    # depth noise scales as z**2, so a 60 mm error is a 44% noise penalty that
    # nothing downstream can recover. A declared value is not a measurement.
    declared = meta.get("standoff_mm_measured")
    measured = meta.get("standoff_mm_from_depth")
    if measured is None:
        # Captures taken before the self-check existed: recompute it here so
        # the archive can still be audited.
        mid = np.load(dfiles[0])[300:420, 540:740].astype(float)
        mid = mid[mid > 0]
        measured = (float(np.median(mid)) * meta["depth_units"] * 1000.0
                    if mid.size else float("nan"))
        info["standoff_recomputed"] = True
    info["standoff_mm_declared"] = declared
    info["standoff_mm_from_depth"] = measured
    if declared is not None and np.isfinite(measured):
        if abs(measured - declared) > MAX_STANDOFF_ERROR_MM:
            issues.append(
                f"standoff declared {declared:.1f} mm but measured "
                f"{measured:.1f} mm ({measured - declared:+.1f})")

    img = cv2.imread(str(cfiles[0]))
    sat = float(np.mean(img.max(axis=2) >= 255))
    info["saturated_fraction"] = sat
    if sat > MAX_SATURATED_FRACTION:
        issues.append(f"{sat:.2%} saturated colour pixels — reduce exposure")
    info["mean_luma"] = float(img.mean())

    return issues, info


def main():
    root = pathlib.Path(sys.argv[1])
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dirs:
        sys.exit(f"nothing in {root}")

    print(f"{'capture':<22} {'dep':>4} {'col':>4} {'valid':>7} {'std/mm':>8} "
          f"{'z/mm':>7} {'sat':>7}  status")
    print("-" * 86)
    n_fail = 0
    for d in dirs:
        issues, info = check_one(d)
        status = "PASS" if not issues else "FAIL: " + "; ".join(issues)
        n_fail += bool(issues)
        print(f"{d.name:<22} {info.get('n_depth',0):>4} {info.get('n_color',0):>4} "
              f"{info.get('valid_fraction',0):>6.1%} "
              f"{info.get('median_per_pixel_std_mm',0):>8.3f} "
              f"{info.get('median_depth_mm',0):>7.1f} "
              f"{info.get('saturated_fraction',0):>6.2%}  {status}")

    print("-" * 86)
    print(f"{len(dirs)} captures, {n_fail} failing")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
