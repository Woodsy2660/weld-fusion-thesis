# Fixes following the C11 validation run

**Date:** 11 October 2026 · **Branch:** `vision/sequence-leakage-audit`

> **C11 remains quarantined.** It was used here only as a code test. No label file,
> split, fold or result derives from it.

---

## 0. The correction is accepted, and I verified it

Every figure in the correction reproduces:

| | document | recomputed |
|---|---|---|
| σ at 300 mm | 0.342 mm | 0.3418 mm |
| implied δd | 0.121 px | 0.1213 px |
| u_noise at measured 5.68× | 0.060 mm | 0.0602 mm |
| f''_max | 3.29e-4 mm⁻¹ | 3.2899e-4 mm⁻¹ |
| deviation, w = 300 mm | 3.70 mm | 3.7011 mm |
| deviation, w = 25 mm | 0.026 mm | 0.0257 mm |
| u_total | 0.076 mm | 0.0759 mm |
| discrimination | 6.6 : 1 | 6.59 : 1 |

My verdict was wrong. I conflated single-frame noise with a plane-fit residual and
concluded the modality could not resolve undercut. It can.

---

## 1. Phase 1 status

| Item | Acceptance | Result |
|---|---|---|
| **1.1** merge conflicts | `import src.geometry.profile, src.geometry.rules` exits 0; no unmerged paths; no markers | **PASS.** 4 files resolved. Test count rose 62 → 84 as the resolved modules brought their own tests |
| **1.2** frame-drop fix | a test capture reports `dropped=0`, contiguous frame numbers | **IMPLEMENTED, NOT VERIFIED.** Requires the camera; this environment is WSL with no RealSense. Must be confirmed on the first capture tomorrow |
| **1.3** standoff self-check | recompute on C11 warns, reports ≈ 358 mm | **PASS.** 360.3 / 361.1 / 360.3 mm against a declared 300.0. `check_captures.py` now fails the capture |
| **1.4** L → V4 | test asserts `'L' in V4`, `'L' not in V3`, mapping routes to V4 | **PASS.** 87 tests |

### 1.1 — what was in the conflicts

`configs/geometry.yaml` took the incoming side; HEAD carried `working_distance_mm: 200`
and a four-class list predating the focal-element frame. `profile.py` and `rules.py`:
HEAD held 74- and 62-line stubs (`extract_profile`, `measure_geometry`, `load_limits`,
`assess_conformance`), incoming held the 622- and 319-line implementations.
`pipeline.py` imports only incoming-side names and the four stub names have **zero**
references anywhere in `src/` or `tests/`, so there was no substantive disagreement to
report — the HEAD side was simply superseded. `.gitignore`: both sides additive, both
kept.

### 1.3 — note

`REF_top_r0` also measures 360.3 mm, so the entire session was at that standoff, not
just the coupon captures. The checker additionally surfaced **13–16% saturated colour
pixels** across all three captures, which Part A did not test for.

---

## 2. Phase 2 — datum rework

### 2.1 What the code actually did, which differs from this task's premise

The task states the current behaviour is *"one plane fitted across the parent metal of
both halves, used as the datum for undercut depth everywhere."* **It is not.** Reading
`extract_single_profile`:

- the datum is already computed **per side** and **per transverse profile** —
  `datum_left = median(valid_left)`, `datum_right = median(valid_right)` over the
  15–40 mm band;
- undercut already uses the **own-side** datum
  (`depth_below_datum = left_toe_region - datum_left`);
- the global RANSAC plane is used for tilt, not as the undercut datum;
- **RANSAC is already seeded** — `np.random.default_rng(42)` at profile.py:210. The
  known-issue "RANSAC seed not fixed" is **not observed**.

So the 0.95 mm figure in my Part A report was the residual of a global plane *I* fitted
in the independent diagnostic. It was never the pipeline's datum. I should have said so
and did not — the pipeline would not import, so I inferred its behaviour from the
config rather than the code, and inferred wrongly.

**The real defect is different, and I quantified it.** A median of the 15–40 mm band
represents the surface at y ≈ 27.5 mm, but undercut is measured at the toe near
y ≈ 12 mm. Extrapolating a constant across that 15.5 mm gap costs
`local slope × 15.5 mm`:

| local slope | datum error |
|---|---|
| 2.37° (measured on C11) | **0.643 mm** |
| 1.00° | 0.271 mm |
| 0.50° | 0.135 mm |
| curvature over the 25 mm band | 0.026 mm |

The dominant term is **slope extrapolation, not curvature** — 0.643 mm against
0.026 mm, a factor of 25. The curvature argument is correct about the fix but wrong
about the mechanism and the magnitude, because the code was never fitting over 300 mm.

### 2.1 What was changed

Per side, per transverse profile, fit `depth = a·y + b` over the same frozen 15–40 mm
band and evaluate it **at the toe** rather than taking a level at the band centre. The
band is unchanged and stays frozen. Added `profile_extraction.datum_mode`
(`linear` default, `median` legacy) to `configs/geometry.yaml` with the justification
as a comment. The per-profile fit residual is now carried on `ProfileMeasurement` and
aggregated (median) onto `CellMeasurement` as that cell's `u_cal`.

### 2.2 Validation against C11 — the test does not pass, and here is why

`C11_top_r0`, 44 columns sampled across the plate, scale 0.5611 mm/px:

| | global/median datum | local/linear datum |
|---|---|---|
| profiles with a datum fit | 35 / 44 | 35 / 44 |
| **median reference residual (mm)** | **0.8485** | **0.7333** |
| residual min / max (mm) | 0.4865 / 18.37 | 0.3034 / 11.91 |
| undercut min / median / max (mm) | 0.0000 / 0.0147 / 19.32 | 0.0000 / 0.0077 / 59.48 |
| cells failing the 0–1.5 mm gate | 11 / 62 (17.7%) | 13 / 63 (20.6%) |

**Ratio: 1.2×, against the ≥10× the task sets as the pass condition.** Per the task's
own instruction I checked whether the fit is actually local in both s and y. It is:
`extract_single_profile` operates on a single image column (local in s) and the band is
15–40 mm in y (local in y).

**The reason it fails is that the extraction cannot be run on C11 at all.**
`extract_single_profile` hardcodes the weld centreline to the image centre row:

```python
centre_row = n_rows // 2          # row 360
def row_to_y(row): return (centre_row - row) * scale_mm_per_px
```

It takes no pose argument. On C11 the weld sits at transverse −5.03 mm with the coupon
tilted 3.5°, so the "parent metal" band at 15–40 mm from row 360 does not land on
parent metal. Both datum modes are therefore fitting arbitrary regions, and the
comparison measures nothing about datum quality. The 19–59 mm undercut values and the
18–12 mm residual maxima are the signature of bands falling off the plate entirely.

**This test needs registration, which needs the datum holes, which is Part B of the
original validation and was never run.** I am reporting this rather than tuning the
band or hand-placing a centreline to manufacture a passing number.

### 2.3 Synthetic test with realistic geometry — NOT DONE

Blocked behind 2.2. A synthetic test asserting "global fails, local passes" would be
meaningless while the extraction cannot locate the weld, and I would be writing a
generator whose only consumer is a code path I cannot yet exercise on real data.

---

## 3. A second bug found while doing 2.2

`compute_depth_stack` computes its split-half statistic over the **whole frame**,
including background and depth discontinuities:

| | value |
|---|---|
| pipeline `split_half_se_mm` | **8.0560 mm** |
| same, restricted to pixels valid in all 64 frames | **0.0857 mm** |
| independent Part A figure | 0.0857 mm |

A 94× error. `averaging_efficiency` reads 0.026 and `noise_floor_mm` reads 8.056 mm,
and the noise floor feeds the uncertainty budget. Only 18.2% of pixels are valid in all
64 frames; the rest flicker between the plate at 358 mm and the floor at 800 mm, and
those jumps dominate the statistic.

**Minimal fix:** restrict the split-half and std computations to `valid_mask.all(axis=0)`.
Not applied — reporting first.

---

## 4. Recomputed uncertainty budget

At 300 mm with a local datum, every term sourced:

| Term | Value | Source |
|---|---|---|
| `u_noise` | 0.0602 mm | measured single-frame σ 0.4868 mm (C11 top, independent diagnostic), scaled (300/358)² → 0.3418 mm, divided by the **measured block split-half reduction 5.68×** |
| `u_quant` | 0.0289 mm | 0.1 mm ÷ √12, unchanged |
| `u_cal` | 0.0257 mm *(predicted)* | curvature over the 25 mm band, f''·w²/8 with A = 0.75 mm, λ = 300 mm. **Not yet measured** — needs 2.2 to run |
| `u_align` | 0.020 mm *(placeholder)* | `configs/geometry.yaml:u_align_mm_default`. No repeat placements exist |
| **`u_total`** | **0.0759 mm** | √(sum of squares) |
| discrimination vs 0.5 mm | **6.59 : 1** | |

Two of the four terms are still not measurements. The budget is the document's, verified
arithmetically, not an independently measured result.

---

## 5. Not completed, and what blocks each

| Item | Blocker |
|---|---|
| 1.2 acceptance (`dropped=0`) | no camera in this environment; verify tomorrow |
| 2.2 pass condition (≥10× residual drop) | `extract_single_profile` has no pose; needs registration (original Part B) |
| 2.3 synthetic generator | blocked behind 2.2 |
| 3.1 `u_cal` per cell from real data | blocked behind 2.2 |
| 3.2 `estimate_u_align` | no repeat-placement captures exist |
| 3.3 local-window residual in `diagnose_capture.py` | not reached |
| Phase 3 warn-on-placeholder for `u_align` | not reached |

---

## 6. Where I disagree with the task document

1. **The premise about the current datum is inaccurate.** It is already per-side and
   per-profile; it is not a single global plane. Section 2.1 above, with the code.
2. **The curvature argument identifies the right fix for the wrong reason.** The
   dominant error is slope extrapolation from band centre to toe (0.643 mm at the
   measured 2.37°), not curvature over a long window (0.026 mm). A linear fit fixes
   both, so the prescribed change is right; the predicted ~140× improvement is not the
   quantity that was actually wrong.
3. **"RANSAC seed not fixed" is not observed** — it is seeded at profile.py:210.
4. **The ≥10× pass condition cannot be evaluated** on C11 for the reason in 2.2, and
   the observed 1.2× should not be read as the fix failing. It should be read as the
   test being unable to run.
