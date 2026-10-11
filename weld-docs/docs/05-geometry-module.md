# 05 — Geometry module

No training. The exceedance formula *is* the calibration. But the measurement
pipeline has design choices, and those must be frozen before results are seen.

## Sensor: Intel RealSense D435i

### Configuration — the four non-negotiables

```python
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

ds.set_option(rs.option.depth_units, 0.0001)   # 100 µm. Default 1000 µm is fatal.
ds.set_option(rs.option.visual_preset, 3)      # High Accuracy
ds.set_option(rs.option.emitter_enabled, 1)
cs.set_option(rs.option.enable_auto_exposure, 0)
cs.set_option(rs.option.enable_auto_white_balance, 0)
```

1. **Depth units 100 µm.** Device default is 1000 µm. At 1 mm quantisation the 0.5 mm
   undercut limit is unmeasurable. Assert `ds.get_option(rs.option.depth_units) ==
   0.0001` at the start of every capture — it does not always persist.
2. **High Accuracy preset.** Sparse, high-confidence pixels only. Missing data beats
   wrong data; missing data maps to Θ, which the framework handles natively.
3. **All post-processing filters OFF.** Especially hole-filling — it interpolates
   across a genuine undercut groove and returns a conforming profile.
4. **Auto-exposure and auto-white-balance off.** The vision model will learn the
   lighting, so the lighting must be identical across all 45 zones.

### Geometry at 300 mm standoff

**300 mm, not 200 mm.** The D435i MinZ is ~280 mm at 1280×720. The 200 mm figure in
earlier documents came from a D405 datasheet — different camera.

```
f_x = 640 / tan(43.5°) = 674 px
Δx  = 300 / 674 = 0.445 mm/px        depth lateral sampling
                 0.215 mm/px          colour lateral sampling
FOV = 569 × 333 mm                    whole coupon in one frame
```

### Depth uncertainty

```
δz = z² δd / (f B)
```

where z is standoff, δd is disparity uncertainty in pixels, f is focal length in
pixels and B = 50 mm is the stereo baseline. The z² dependence is why standoff
matters so much.

With δd = 0.08 px at z = 300 mm: **δz = 0.214 mm single-frame.**
Temporal averaging over N frames gives δz/√N, so **64 frames → 0.027 mm**.

**Caveat to carry into the write-up:** RealSense stereo noise is correlated frame to
frame, so the reduction will not be exactly 1/√N. Save the 64 frames individually
and compute the per-pixel standard deviation — that gives the real reduction factor
instead of the assumed one.

## Capture protocol

Per coupon face:

- 64 raw depth frames, saved as **individual `.npy` arrays** — not averaged on the fly.
- 10 colour frames.
- Both datum holes in frame.
- Then flip about the long axis and repeat for the root face.

Metadata per capture: coupon ID, face, timestamp, measured standoff, exposure, laser
power, ASIC temperature (`rs.option.asic_temperature` — the D435i drifts with die
temperature), USB link type, firmware version, SDK version.

Record the USB link type every session. On a USB 2 link the camera silently drops
frames rather than erroring, and a dropped frame inside a 64-frame average looks like
noise, not a fault.

## Calibration

1. **On-chip self-calibration** once after firmware update.
2. **Tare at 300.0 mm** against a flat machined reference. Removes the absolute depth
   offset that would otherwise masquerade as reinforcement height.
3. **Per-coupon plane fit** on the unwelded parent metal either side of the bead.
   The residual RMS *is* the `u_cal` term. Above ~0.03 mm indicates a mount problem,
   not a camera problem.
4. **Depth Quality Tool** at 300 mm on a flat plate: its *Subpixel RMS* output is
   literally the δd assumed as 0.08 px above. Replace the assumption with the
   measurement. Also gives Z accuracy, Z RMS and fill rate.

## Uncertainty budget

```
u_total = √(u_noise² + u_quant² + u_cal² + u_align²)
```

| Term | Value | Source |
|---|---|---|
| u_noise | 0.063 mm | Stereo noise after temporal and multi-row averaging |
| u_quant | 0.029 mm | 0.1 mm step ÷ √12 (uniform quantisation) |
| u_cal | 0.030 mm | Plane-fit residual |
| u_align | 0.020 mm | Fixture repeatability |
| **u_total** | **0.078 mm** | **Predicted** |

### Discrimination against ISO 5817, t = 10 mm

| Measurement | Limit | Ratio | Verdict |
|---|---|---|---|
| Undercut, Level B/C | 0.5 mm | 6.4 : 1 | Marginal — the declared limitation |
| Undercut, Level D | 1.0 mm | 12.8 : 1 | Comfortable |
| Reinforcement (b = 16) | 2.6 mm | 33 : 1 | Easy |
| Misalignment | 1.2 mm | 15 : 1 | Easy |

### The step block — highest-value outstanding experiment

A milled block with steps at 0.2 / 0.5 / 1.0 / 2.0 mm, measured through the full
pipeline at 300 mm. This replaces the computed 0.078 mm with a measured figure and
lets the thesis report **predicted versus measured**, which is a stronger result than
either number alone. Measure on both a machined and a sandblasted surface — mill
scale and spatter are the real-world case and will be worse.

## BPA construction

```
P(non-conformance) = Φ((d̂ − d_limit) / u_total)
```

where d̂ is the measured dimension, d_limit the ISO 5817 limit, and Φ the standard
normal CDF. This is the probability that the true dimension exceeds the limit given
a measurement with Gaussian uncertainty.

Mapped to mass:

```
m({U})       = P(non-conformance)        for undercut
m({N,P,L,C}) = 1 − P(non-conformance)
m(Θ)         = 1                          if the cell has too few valid depth pixels
```

## Frozen parameters — leakage control

The following are **set a priori from the ISO 5817 definition of undercut depth** and
must not be tuned against classification results:

- ROI width either side of the bead
- The datum surface the undercut depth is referenced to
- Plane-fit extent
- Minimum valid-pixel fraction for a cell to produce a non-vacuous BPA

Write them into a config file with a comment giving their justification, commit it,
and do not touch it afterwards. Tuning these while watching the confusion matrix is
leakage through the back door.
