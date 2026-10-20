# Geometry module — declared limitations

## L1. Lateral scale: bead width only (decision 2026-10-19)

The D6–D8 datum-hole baseline measures 256.5 mm against a nominal 270 mm
(C01_top_r0; 3D Euclidean distance, `registration_seeded.py:219`). The cause is
not established and is not investigated further.

**Why it does not propagate to the axial measurands.** The uint16 depth value is
computed by the camera firmware from its factory calibration; the pipeline only
multiplies it by `depth_units`. The pipeline's `fx` enters only the lateral
deprojection `x = (u - ppx) * z / fx`, so a lateral error rescales x and y but
not z. Every height is a difference between a measured point and a datum fitted
to points of the same profile; rescaling y changes the datum's slope but not its
predicted height at any measured point, so the height is unchanged.

| Quantity | Type | Affected |
|---|---|---|
| Undercut depth | axial | No |
| Reinforcement height | axial | No |
| Linear misalignment | axial | No |
| Cell index (proportional to detected D6–D8 span) | ratio | No |
| Bead width `b` | lateral | **Yes** |

`b` enters only the excess-reinforcement limit, `1 mm + 0.1b`. At b ≈ 14 mm a 5%
error is 0.7 mm, moving that limit by 0.07 mm against ≈ 2.4 mm: a 3% shift in one
limit of three.

**Second-order effect.** The coupon y = 0 is placed 50 mm from the hole line, so
it is offset from the true joint centre (measured 2.9–5.7 mm on C05_top). The
weld centreline is therefore taken from `configs/weld_axis_seeds.csv`, not from
y = 0 (`src/geometry/measure.py:seed_centre_line`).

## L2. Depth over the weld bead reads as a trench — OPEN, BLOCKING

The colour image shows a convex MIG cap; the D435 depth reads the bead band
**farther** from the camera than the parent plate:

- **Raw uint16, no transform** (C05_top_r0, three columns): the depth dip
  coincides with the bright bead band; ±25 px median 0.6–0.8 mm farther, deepest
  point 2–3 mm. Depth-to-colour translation is 14.69 mm in x and 0.04 mm in y,
  so the row alignment is exact to sub-pixel.
  `_diagnostics/sign_test_raw_depth_C05_top_r0.png`.
- **After fixing the registration normal** (2026-10-20, `fit_plate_normal`), the
  pipeline's own z gives bead-minus-plate −1.31 / −1.22 / −1.38 mm on r0/r1/r2,
  identical to the earlier sign-corrected figure. Not a sign error.
- **Local surface fit of the crown** (≥ 507 points per fit) gives reinforcement
  −0.17 to −1.67 mm in every C05_top cell.
- The feature follows the coupon across a 9.9 mm re-placement, so it is not
  fixed to camera pixels; a specular-material depth error would follow the
  coupon too. Not separable from existing data.

Until resolved, no reinforcement or undercut value from this sensor is valid,
and no geometry BPA is emitted.

## L3. Repeatability is set by spatially correlated noise

C05_top r0/r1/r2 (re-placement repeats, "repeat placement for u_align").

**Plate window test** (plane-detrended parent plate): SD of a window mean, one
capture, from the repeat-to-repeat difference:

| window (y × s) | cells | σ measured | σ if independent |
|---|---|---|---|
| 0.5 × 1 mm | 1 | 0.519 | 0.519 |
| 2 × 4 mm | 16 | 0.497 | 0.130 |
| 8 × 16 mm | 256 | 0.364 | 0.032 |

The noise is correlated over > 10 mm; averaging N neighbouring points does not
reduce it by √N.

**Measurands** (pooled SD across repeats, 10 cells):

| measurand | before (binned, max over profiles) | after (local fit + axis seeds) | limit | S/U after |
|---|---|---|---|---|
| misalignment | 0.875 | 0.275 | 1.0 | 3.6 : 1 |
| reinforcement | 0.924 | 0.493 | ≈ 2.4 | 4.9 : 1 (value invalid, L2) |
| undercut | 0.848 | 0.586 | 0.5 | 0.85 : 1 (value invalid, L2) |

Recorded in `configs/geometry.yaml` `uncertainty.measured_repeatability`, with
`use_for_bpa: false` until L2 is resolved.

## L4. Frozen 5-px smoothing kernel under-reads narrow undercut by 20%

`profile_extraction.smoothing_kernel_px: 5` (2.2 mm at 0.445 mm/px) attenuates
the 1.5 mm synthetic undercut groove: injected 0.2 / 0.5 / 0.8 mm reads
0.160 / 0.400 / 0.640; with 3 px or 1 px, exact. Previously hidden because the
vertical-weld plane mask (`pipeline.py:171-183`, retired 2026-10-20) inflated
u_cal to 0.24 mm and the test tolerance to 0.72 mm. The parameter is frozen
(weld-docs hard rule 6) and has not been changed;
`test_undercut_recovery[0.8]` now fails and is left failing.
