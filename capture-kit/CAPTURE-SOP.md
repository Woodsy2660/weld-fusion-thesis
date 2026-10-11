# RealSense capture SOP

Run through this in order. Steps 1–7 happen once, at the start of the session.
Steps 8–10 are the per-coupon loop. Step 11 happens before you break down the rig.

---

## Phase 0 — setup and validation (once, ~45 min)

### 1. Physical rig

- Camera **normal to the plate**. A tilt becomes a systematic depth bias across the bead.
- Rigid mount. Copy stand or mag base — not a tripod on a workshop floor.
- Camera and coupon support clamped to the **same surface**, so 300 mm is a fixed
  mechanical dimension measured once with a depth gauge.
- **L-fence** for coupon location — two edge stops. Do NOT use dowels through the
  datum holes; they obscure the circles you need for the pose solve.
- Matte black background. Returns almost no IR, so the plate separates cleanly in
  depth and the edges are crisp in colour.
- **Kill the ambient light.** Blinds closed, or work at night. Ambient drifting
  through the session is the easiest confound to create and the hardest to remove.

### 2. Confirm the link

```
rs-enumerate-devices -s
```

Must report **USB 3.x**. If it says 2.1, change the cable or the port — a motherboard
rear port, not a hub, not a front panel. On USB 2 the camera drops frames silently,
and a dropped frame inside a 64-frame average looks like noise, not a fault.
`capture.py` aborts on a USB 2 link.

### 3. Firmware

Update in the Viewer before anything else. A firmware flash resets calibration, so
this must come before step 4.

### 4. Calibrate

In `realsense-viewer`:

1. **On-chip self-calibration** — More → On-Chip Calibration, flat textured target.
2. **Tare calibration at 300.0 mm** against a flat machined reference at the measured
   standoff. Removes the absolute depth offset that would otherwise read as
   reinforcement height.

Then **Depth Quality Tool** on a flat plate at 300 mm. Record all four numbers:

| Metric | What it gives you |
|---|---|
| Subpixel RMS | This *is* the δd assumed as 0.08 px in δz = z²δd/(fB). Replace the assumption. |
| Z RMS / plane-fit RMS | The u_cal term, measured |
| Z Accuracy % | Residual systematic bias after tare |
| Fill Rate | How often cells will map to Θ for insufficient data |

Write them into the session log. These are thesis numbers, not setup numbers.

### 5. Determine exposure

1. In the Viewer, enable auto-exposure, point at a **representative** coupon (one
   with a bright shiny bead, not the dullest one).
2. Let it settle, read the exposure value it lands on.
3. Disable auto-exposure, set that value manually.
4. Check the brightest part of the bead is not clipped and the shadowed toe is not
   crushed. Nudge until both hold.
5. **Write the final value into `EXPOSURE_US` in `capture.py` and never change it.**

Same for white balance.

### 6. Emitter A/B test

As-welded steel has plenty of natural texture, so the IR projector may help or may
cause specular dropout on shiny beads. Test it:

```
capture with laser_power = 0, 150, 240, 360
```

on the same coupon, and compare **fill rate in the weld region** and the per-pixel
std over a known-flat area. Pick the winner, set `LASER_POWER`, freeze it.

### 7. Disk space

```
depth: 1280 × 720 × 2 bytes = 1.84 MB/frame × 64 = 118 MB per capture
38 captures ≈ 4.5 GB depth + ~1.5 GB colour
```

Budget **8 GB** and check before you start.

---

## Phase 1 — capture loop

### 8. Randomise the order

```
python -c "import random;c=[f'C{i:02d}' for i in range(1,16)];random.shuffle(c);print(' '.join(c))"
```

Paste into `ORDER` in `run_session.sh`.

**This is not optional.** Capturing C01→C15 in order makes anything that drifts
through the session — ASIC temperature, ambient light, your own technique —
correlate with coupon ID. Coupon ID is the cross-validation fold group. A drift
correlated with the fold group is indistinguishable from a real effect, and nothing
in the analysis can separate them afterwards.

### 9. Run it

```
./run_session.sh
```

It prompts between coupons, captures the reference target at both ends, and runs the
checker at the finish.

### 10. The flip rule

**Always flip about the long (weld) axis. Never end-for-end.**

Then s is preserved and only y mirrors. Verification: after flipping, the **Ø6 hole
must still appear at the low-s end**. If it appears at the high-s end, the coupon was
flipped wrongly — reject the capture and redo it. Do not correct it in software.

Without this, root-face undercut measurements land on the wrong side of the weld, and
because undercut is roughly symmetric, it looks plausible while being wrong.

---

## Phase 2 — verify before breaking down

### 11. Run the checker

```
python check_captures.py data/raw/S1
```

Every row must say PASS. Anything red gets recaptured **while the rig is still set
up**. This is the step people skip and then lose a week to.

Also eyeball: the start-of-session and end-of-session REF captures should agree. If
they do not, the rig drifted during the session and you need to know that now.

---

## Capture count per coupon

| Item | Count | Why |
|---|---|---|
| Depth frames per face | **64** | δz/√N: 0.214 mm → 0.027 mm. Saved individually so you can compute the real per-pixel std rather than assume 1/√N holds. |
| Colour frames per face | **10** | Pick the sharpest, or average. Cheap insurance against a blurred frame. |
| Faces | **2** | Top and root. The depth stream sees only the presented surface; flipping doubles the geometric evidence for free. |
| **Per coupon** | **128 depth + 20 colour** | |

Session totals:

| | Captures | Depth frames |
|---|---|---|
| 15 coupons × 2 faces | 30 | 1920 |
| Repeatability: 3 coupons × 2 re-placements | 6 | 384 |
| Reference target, start and end | 2 | 128 |
| **Total** | **38** | **2432** |

About two seconds of streaming each; three hours all up including handling and checks.

### Why the repeatability captures matter

Your uncertainty budget currently *assumes* u_align = 0.020 mm for fixture
repeatability. Lifting three coupons out and re-placing them twice each gives you six
independent placements, from which the spread of the recovered pose **measures**
u_align instead.

That converts one of the four terms in

```
u_total = √(u_noise² + u_quant² + u_cal² + u_align²)
```

from an assumption into a measurement, for about fifteen minutes of extra work. The
Depth Quality Tool gives you u_cal and feeds u_noise; the step block gives you the
end-to-end check. Between them you can report a fully measured budget against the
predicted 0.078 mm — which is a stronger result than either number alone.

---

## What NOT to do

- **Do not capture through the Viewer.** Its exports are post-processed and it writes
  no metadata. Viewer is for steps 2–6 only.
- **Do not use `rs.align`.** It resamples depth and interpolates — exactly what
  turning off the filters was for. Register colour and depth independently to the
  coupon frame using the datum holes. The intrinsics and extrinsics are saved in
  `meta.json` if you ever need to align later.
- **Do not average frames at capture time.** Save all 64; average in analysis.
- **Do not enable any post-processing filter**, above all hole-filling. It
  interpolates across a genuine undercut groove and hands back a conforming profile.
- **Do not adjust exposure mid-session** to make a dull coupon look better. If you
  must, record it and treat that coupon as a separate acquisition condition.
- **Do not let couplant near a coupon before its optical capture is verified.**
  Irreversible.
