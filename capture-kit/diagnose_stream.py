"""
Separate "the rig is broken" from "capture.py configures it badly".

capture.py fails at its first wait_for_frames() even though pipe.start()
succeeded. That narrows the cause but does not identify it. This runs four
escalating stages and stops at the first failure; whichever stage breaks tells
you where the problem is.

  A  default profile, no options          -- is the camera streaming at all?
  B  capture.py's profiles, no options    -- is the 720p+1080p combination the problem?
  C  options set BEFORE start, then start -- does the option block break it?
  D  options set AFTER start (capture.py) -- reproduces the current failure

Reading the result:
  A fails              -> hardware, driver, cable, or another process holds the
                          camera. Close RealSense Viewer and retry before
                          anything else.
  A ok, B fails        -> USB bandwidth. The 1280x720 depth + 1920x1080 colour
                          pair at 30 fps is heavy; try a different port/cable,
                          or drop colour to 1280x720.
  B ok, C ok, D fails  -> capture.py's bug. It configures the sensor while
                          streaming; move the option block before pipe.start().
  everything ok        -> intermittent. Suspect cable seating or thermal state.

Usage (Windows, inside the capture venv):
    python diagnose_stream.py
"""
import sys
import time

import pyrealsense2 as rs

# Mirrors capture.py's FROZEN CAPTURE PARAMETERS.
DEPTH_W, DEPTH_H, DEPTH_FPS = 1280, 720, 30
COLOR_W, COLOR_H, COLOR_FPS = 1920, 1080, 30
DEPTH_UNITS = 0.0001
LASER_POWER = 240.0
EXPOSURE_US = 200.0
WHITE_BALANCE = 4500.0
GAIN = 64.0

TIMEOUT_MS = 5000
N_FRAMES = 5


def banner(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def get_sensor(dev, name):
    for s in dev.query_sensors():
        if s.get_info(rs.camera_info.name) == name:
            return s
    raise RuntimeError(f"sensor not found: {name}")


def device_report():
    banner("DEVICE")
    ctx = rs.context()
    devs = list(ctx.query_devices())
    if not devs:
        print("  NO DEVICE FOUND.")
        print("  -> camera not connected, driver not installed, or another")
        print("     process has it open. Close RealSense Viewer first.")
        return None
    for d in devs:
        for info in ("name", "serial_number", "firmware_version",
                     "usb_type_descriptor", "physical_port"):
            try:
                print(f"  {info:<22} {d.get_info(getattr(rs.camera_info, info))}")
            except Exception:
                pass
    return devs[0]


def supported_profiles(dev):
    banner("SUPPORTED PROFILES (what the device actually offers)")
    for s in dev.query_sensors():
        name = s.get_info(rs.camera_info.name)
        seen = set()
        for p in s.get_stream_profiles():
            try:
                v = p.as_video_stream_profile()
            except Exception:
                continue
            key = (str(p.stream_type()), v.width(), v.height(),
                   str(p.format()), p.fps())
            if key in seen:
                continue
            seen.add(key)
        wanted = [k for k in seen
                  if (k[1], k[2], k[4]) in {(DEPTH_W, DEPTH_H, DEPTH_FPS),
                                            (COLOR_W, COLOR_H, COLOR_FPS)}]
        print(f"  {name}: {len(seen)} profiles; "
              f"{len(wanted)} match the requested sizes")
        for k in sorted(wanted):
            print(f"      {k[0]:<22} {k[1]}x{k[2]} {k[3]:<14} {k[4]}fps")


def apply_options(dev, label):
    ds = dev.first_depth_sensor()
    cs = get_sensor(dev, "RGB Camera")
    print(f"  [{label}] visual_preset -> high_accuracy")
    ds.set_option(rs.option.visual_preset,
                  int(rs.rs400_visual_preset.high_accuracy))
    time.sleep(0.5)
    print(f"  [{label}] depth_units -> {DEPTH_UNITS}")
    ds.set_option(rs.option.depth_units, DEPTH_UNITS)
    ds.set_option(rs.option.emitter_enabled, 1)
    ds.set_option(rs.option.laser_power, LASER_POWER)
    cs.set_option(rs.option.enable_auto_exposure, 0)
    cs.set_option(rs.option.enable_auto_white_balance, 0)
    cs.set_option(rs.option.exposure, EXPOSURE_US)
    cs.set_option(rs.option.white_balance, WHITE_BALANCE)
    cs.set_option(rs.option.gain, GAIN)
    actual = ds.get_option(rs.option.depth_units)
    print(f"  [{label}] depth_units reads back {actual}")
    return abs(actual - DEPTH_UNITS) < 1e-9


def try_stream(label, cfg=None, options_before=False, options_after=False):
    print(f"\n--- {label} ---")
    pipe = rs.pipeline()
    cfg = cfg or rs.config()

    try:
        if options_before:
            # Configure the sensor while STOPPED, then start. This is the
            # ordering capture.py should probably use.
            ctx = rs.context()
            dev = list(ctx.query_devices())[0]
            apply_options(dev, "pre")

        profile = pipe.start(cfg)
        print("  pipe.start() OK")

        if options_after:
            apply_options(profile.get_device(), "post")

        for i in range(N_FRAMES):
            t0 = time.time()
            pipe.wait_for_frames(TIMEOUT_MS)
            print(f"  frame {i + 1}/{N_FRAMES} in {1000 * (time.time() - t0):.0f} ms")
        print(f"  RESULT: PASS")
        return True
    except Exception as exc:
        print(f"  RESULT: FAIL -- {type(exc).__name__}: {exc}")
        return False
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        time.sleep(1.0)   # let the device settle before the next stage


def main():
    dev = device_report()
    if dev is None:
        sys.exit(2)
    supported_profiles(dev)

    banner("STAGED STREAM TEST")

    full = rs.config()
    full.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    full.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, COLOR_FPS)

    results = {}
    results["A default profile, no options"] = try_stream("A")
    if results["A default profile, no options"]:
        results["B requested profiles, no options"] = try_stream("B", cfg=full)
        if results["B requested profiles, no options"]:
            results["C options BEFORE start"] = try_stream(
                "C", cfg=full, options_before=True)
            results["D options AFTER start (capture.py)"] = try_stream(
                "D", cfg=full, options_after=True)

    banner("SUMMARY")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    print("\nInterpretation:")
    if not results.get("A default profile, no options"):
        print("  A failed -> HARDWARE / DRIVER / ANOTHER PROCESS.")
        print("  Close RealSense Viewer, reseat the USB cable, try a port")
        print("  directly on the motherboard rather than a hub, and retry.")
    elif not results.get("B requested profiles, no options"):
        print("  A passed, B failed -> USB BANDWIDTH at the requested sizes.")
        print("  1280x720 depth + 1920x1080 colour at 30fps is a heavy pair.")
        print("  Try a different cable/port, or lower COLOR_W/COLOR_H.")
    elif results.get("C options BEFORE start") and \
            not results.get("D options AFTER start (capture.py)"):
        print("  C passed, D failed -> CODE BUG IN capture.py.")
        print("  Setting visual_preset and depth_units while streaming wedges")
        print("  the stream. Move the option block to before pipe.start().")
    elif all(results.values()):
        print("  Everything passed. The earlier failure was intermittent --")
        print("  suspect cable seating, a cold start, or a process that had")
        print("  the camera open at the time.")
    else:
        print("  Mixed result; see the stage that failed above.")


if __name__ == "__main__":
    main()
