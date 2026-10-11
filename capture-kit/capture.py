#!/usr/bin/env python3
"""
RealSense D435i coupon capture — one coupon face per invocation.

Usage:
    python capture.py <coupon_id> <face> <session> <order_index> [--rep N]

Example:
    python capture.py C07 top S1 9
    python capture.py C07 top S1 9 --rep 1      # repeat placement, repeatability test
    python capture.py REF  top S1 0             # reference flat plate

Writes:
    data/raw/<session>/<coupon>_<face>_r<rep>/
        depth/000.npy ... 063.npy     uint16, raw, NO post-processing
        color/000.png ... 009.png     lossless PNG
        meta.json                     config, intrinsics, extrinsics, frame numbers
"""

import argparse, json, pathlib, sys, time
from datetime import datetime, timezone

import numpy as np
import cv2
import pyrealsense2 as rs

# ----------------------------------------------------------------------------
# FROZEN CAPTURE PARAMETERS — change only with a DECISIONS.md entry
# ----------------------------------------------------------------------------
N_DEPTH        = 64          # for 1/sqrt(N) temporal averaging
N_COLOR        = 10
N_WARMUP       = 30          # frames discarded; sensor is unstable at stream start
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30
COLOR_W, COLOR_H, COLOR_FPS = 1920, 1080, 30
DEPTH_UNITS    = 0.0001      # 100 um. NON-NEGOTIABLE. Device default 0.001 is fatal.
LASER_POWER    = 240.0       # set from the emitter A/B test — see SOP step 6
EXPOSURE_US    = 200.0       # set from the exposure procedure — see SOP step 5
WHITE_BALANCE  = 4500.0
GAIN           = 64.0
STANDOFF_TOLERANCE_MM = 10.0   # declared vs measured; beyond this, warn
OUT_ROOT       = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
# ----------------------------------------------------------------------------


def get_sensor(dev, name):
    for s in dev.query_sensors():
        if s.get_info(rs.camera_info.name) == name:
            return s
    raise RuntimeError(f"sensor not found: {name}")


def intrinsics_dict(vsp):
    i = vsp.get_intrinsics()
    return {"width": i.width, "height": i.height, "ppx": i.ppx, "ppy": i.ppy,
            "fx": i.fx, "fy": i.fy, "model": str(i.model), "coeffs": list(i.coeffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coupon_id")
    ap.add_argument("face", choices=["top", "root"])
    ap.add_argument("session")
    ap.add_argument("order_index", type=int)
    ap.add_argument("--rep", type=int, default=0, help="repeat placement index")
    ap.add_argument("--standoff", type=float, default=300.0, help="measured standoff mm")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    outdir = OUT_ROOT / args.session / f"{args.coupon_id}_{args.face}_r{args.rep}"
    if outdir.exists():
        sys.exit(f"REFUSING TO OVERWRITE: {outdir} already exists")
    (outdir / "depth").mkdir(parents=True)
    (outdir / "color").mkdir(parents=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)
    profile = pipe.start(cfg)
    streaming = True
    dev = profile.get_device()

    try:
        ds = dev.first_depth_sensor()
        cs = get_sensor(dev, "RGB Camera")

        # ORDER MATTERS: loading a visual preset resets depth_units.
        ds.set_option(rs.option.visual_preset, int(rs.rs400_visual_preset.high_accuracy))
        time.sleep(0.5)
        ds.set_option(rs.option.depth_units, DEPTH_UNITS)
        ds.set_option(rs.option.emitter_enabled, 1)
        ds.set_option(rs.option.laser_power, LASER_POWER)

        cs.set_option(rs.option.enable_auto_exposure, 0)
        cs.set_option(rs.option.enable_auto_white_balance, 0)
        cs.set_option(rs.option.exposure, EXPOSURE_US)
        cs.set_option(rs.option.white_balance, WHITE_BALANCE)
        cs.set_option(rs.option.gain, GAIN)

        # Hard assertion — this has silently failed before.
        actual_units = ds.get_option(rs.option.depth_units)
        if abs(actual_units - DEPTH_UNITS) > 1e-9:
            sys.exit(f"ABORT: depth_units is {actual_units}, expected {DEPTH_UNITS}")

        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        if not usb.startswith("3"):
            sys.exit(f"ABORT: USB link is {usb}, needs 3.x. Change cable or port.")

        temp_start = ds.get_option(rs.option.asic_temperature)

        for _ in range(N_WARMUP):
            pipe.wait_for_frames()

        # ---- acquire into RAM only; NO disk I/O inside this loop ----
        # A 1920x1080 PNG encode costs 30-80 ms against a 33 ms frame
        # interval, so writing here stalls the pipeline and the driver drops
        # frames. The C11 session lost 49 of 113 device frames this way.
        # Buffering costs 64 x 1.84 MB + 10 x 6.2 MB ~ 180 MB.
        depth_buf, color_buf, depth_nums, color_nums = [], [], [], []
        while len(depth_buf) < N_DEPTH or len(color_buf) < N_COLOR:
            frames = pipe.wait_for_frames()
            if len(depth_buf) < N_DEPTH:
                df = frames.get_depth_frame()
                if df:
                    depth_buf.append(np.asanyarray(df.get_data()).copy())
                    depth_nums.append(df.get_frame_number())
            if len(color_buf) < N_COLOR:
                cf = frames.get_color_frame()
                if cf:
                    color_buf.append(np.asanyarray(cf.get_data()).copy())
                    color_nums.append(cf.get_frame_number())

        temp_end = ds.get_option(rs.option.asic_temperature)

        # ---- standoff self-check, before anything is written ----
        # meta.json recorded 300.0 for the C11 session because that is what
        # was typed on the command line; the rig was actually at 358 mm. A
        # declared value must never stand in for a measurement.
        mid = depth_buf[0][300:420, 540:740].astype(float)
        mid = mid[mid > 0]
        measured_standoff = (float(np.median(mid)) * DEPTH_UNITS * 1000.0
                             if mid.size else float("nan"))
        if not (abs(measured_standoff - args.standoff) <= STANDOFF_TOLERANCE_MM):
            print(f"!! WARNING: declared {args.standoff:.1f} mm, "
                  f"measured {measured_standoff:.1f} mm - check the stand "
                  f"before continuing")

        # Streaming is finished; stop before touching the disk.
        pipe.stop()
        streaming = False

        for i, a in enumerate(depth_buf):
            np.save(outdir / "depth" / f"{i:03d}.npy", a)
        for i, a in enumerate(color_buf):
            cv2.imwrite(str(outdir / "color" / f"{i:03d}.png"), a,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])

        dprof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ext = dprof.get_extrinsics_to(cprof)

        # Dropped frames show up as gaps in the frame-number sequence.
        gaps = [b - a for a, b in zip(depth_nums, depth_nums[1:])]
        dropped = sum(g - 1 for g in gaps if g > 1)

        meta = {
            "coupon_id": args.coupon_id,
            "face": args.face,
            "capture_session": args.session,
            "capture_order_index": args.order_index,
            "repeat_index": args.rep,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "standoff_mm_measured": args.standoff,
            "standoff_mm_from_depth": measured_standoff,
            "operator_notes": args.note,

            "n_depth": N_DEPTH, "n_color": N_COLOR, "n_warmup": N_WARMUP,
            "depth_units": actual_units,
            "depth_scale_m": ds.get_depth_scale(),
            "visual_preset": "high_accuracy",
            "post_processing_filters": [],
            "emitter_enabled": 1, "laser_power": LASER_POWER,
            "exposure_us": EXPOSURE_US, "white_balance_k": WHITE_BALANCE, "gain": GAIN,
            "auto_exposure": 0, "auto_white_balance": 0,

            "depth_intrinsics": intrinsics_dict(dprof),
            "color_intrinsics": intrinsics_dict(cprof),
            "depth_to_color_extrinsics": {
                "rotation": list(ext.rotation), "translation": list(ext.translation)},

            "asic_temperature_c_start": temp_start,
            "asic_temperature_c_end": temp_end,
            "usb_type": usb,
            "firmware": dev.get_info(rs.camera_info.firmware_version),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "sdk_version": rs.__version__,
            "depth_frame_numbers": depth_nums,
            "color_frame_numbers": color_nums,
            "dropped_depth_frames": dropped,
        }
        with open(outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"OK  {outdir}  depth={n_d} color={n_c} dropped={dropped} "
              f"temp {temp_start:.1f}->{temp_end:.1f}C")
        if dropped:
            print("  !! DROPPED FRAMES — check cable/port before continuing")

    finally:
        # pipe.stop() already ran after acquisition on the success path.
        if streaming:
            pipe.stop()


if __name__ == "__main__":
    main()
