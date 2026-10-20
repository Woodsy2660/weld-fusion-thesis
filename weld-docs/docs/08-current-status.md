# 08 — Current status

**Last updated: 2026-10-17. Update this file whenever you finish a piece of work.**

## Overall

Specimens are welded and collected. The ultrasonic instrument arrives in the week of
13 October. The vision pipeline has produced a significant negative result and needs
a clean retrain. The geometry module has hardware configured but no captures yet.
Fusion code is specified but cannot be validated end to end until three aligned BPA
tables exist.

Roughly four weeks remain, of which the last week is writing only.

## Module status

| Module | State | Notes |
|---|---|---|
| Frame / focal elements | **Done** | Three-way vision classifier settled; ceiling proposition proved |
| Vision | **Needs retrain** | Leakage finding complete; L→V4 remap outstanding; single-source retrain not yet run |
| Geometry | **Hardware ready, no data** | D435i configured at 300 mm; step block not yet made |
| Ultrasonic | **Blocked on instrument** | Prep work can and should proceed now |
| Fusion | **Specified, untested end to end** | Phase 2 implementable; needs real BPAs to validate |
| Evaluation | **Partially built** | Splits and metrics exist; clustered bootstrap and ceiling calculator outstanding |
| Registration | **Pipeline built, detection needs tuning** | Geometry correct (synthetic tests pass); hole detection picks up spatter |

## Done

- Frame of discernment and per-modality focal elements declared and justified.
- Two-modality ceiling proposition proved and written up.
- Sequence-leakage audit: 81.59% contamination quantified.
- LOSO transfer evaluation: macro-F1 0.43 / 0.22, ECE 0.34 / 0.68.
- Correct chance baseline computed (0.3023) on the actual class distribution.
- Crop-geometry confound cleared (joint-matched deltas −0.032 / −0.045).
- Letterbox geometry solved in closed form: `pad_fraction = 1 − 1/AR`.
- Stratified split reducing composition overlap 32.6% → 1.2%.
- Class-weight derivation corrected to train-split-only.
- 45 weld zones produced with logged parameters; coupons collected.
- D435i configuration settled, including the 300 mm standoff correction.
- Datum hole geometry corrected to y = 50 mm after the plate width was confirmed at
  70 mm per half.
- Documentation repository layout updated to match actual codebase structure
  (`weld-docs/CLAUDE.md`, `10-conventions.md`, main `CLAUDE.md`). V4 focal element
  corrected to `{N,U,L,C}`. Loose prompt files consolidated into `claude prompts/`.
  Credentials file added to `.gitignore`.

## In flight

- Drilling datum holes; stamping coupon IDs; punch-marking zone boundaries.
- Clean vision and depth capture of all coupons, **before** any couplant touches them.
- Cell-level label file construction from the welding parameter log.
- UT feature-extractor dry run against synthetic A-scans.

## Blocked

| Item | Blocked on |
|---|---|
| UT feature extraction on real data | Instrument arrival, week of 13 Oct |
| Fusion end-to-end validation | All three BPA tables existing |
| Measured uncertainty budget | Step block fabrication |

## Open questions requiring a decision

| # | Question | Status |
|---|---|---|
| 1 | Box-area 270 vs 267 px² discrepancy | Unresolved. Re-check `predictions_fold0.csv`. Do not settle by recency. |
| 2 | McNemar comparison family size | Proposed m = 4; needs confirming and writing into code as a constant |
| 3 | Radiography — booked or cut? | Objective 6 currently promises it. Confirm or remove the clause. |
| 4 | ISO 5817 Level B only, or B and C? | Objective 3 says B; undercut at B is marginal at 6.4:1 |
| 5 | Third datum hole | Optional; decide before drilling is finished |

## Known stale text in the thesis document

These will read badly against the actual results if not fixed:

- Objective 4 and the Scope section both state that ultrasonic instrumentation "was
  not secured". It has been hired. Reword to cover a reduced-scan-density outcome.
- Objective 6 promises radiographic examination — see open question 3.
- `system-configuration-v2.md` still records the datum hole position as y = +68 mm.
  Correct to y = 50 mm.

## Hard constraints to respect when planning

- The ultrasonic hire window cannot be moved or repeated.
- Couplant contaminates the surface and promotes flash rust, so **all optical capture
  must precede all ultrasonic testing**. This ordering is irreversible.
- Flash rust is already appearing on the as-welded surfaces. Capture early, and treat
  every coupon identically — selectively cleaning some would confound surface
  condition with capture order, which correlates with coupon, which is the fold group.
- **Experimental freeze 1 November.** After that, writing only.

---

## 2026-10-11 — First real capture validated against the pipelines

`results/validation_C11/REPORT.md`. Verdict: the geometry pipeline does **not** yet
produce trustworthy per-cell BPAs from real data.

**Blocked.** Four files carry unresolved merge conflicts (`configs/geometry.yaml`,
`src/geometry/profile.py`, `src/geometry/rules.py`, `.gitignore`). `src.geometry` does
not import. Validation Parts B (pipeline) and C (two-face consistency) could not run.

**Measured, from an independent diagnostic that imports nothing from `src/`:**

| | C11 top | C11 root | expected |
|---|---|---|---|
| standoff (mm) | 358.3 | 359.8 | 300 |
| single-frame noise (mm) | 0.487 | 0.549 | 0.214 |
| averaging reduction over 64 frames | 5.68× | 5.64× | 8× |
| parent-plane RMS (mm) | 0.95 | 0.84 | 0.030 |
| inter-plate angular distortion | 2.37° | n/a | < 0.5° |

**Next, in order:** resolve the merge; set the real standoff in `run_session.ps1` (the
300.0 is an untouched placeholder); capture the repeat placements so `u_align` becomes
measured; run the step block to re-anchor the uncertainty budget.

**Vision plumbing:** three-way classifier confirmed, `{C}` head genuinely absent.
`final_welding/Lack of Fusion → V3` still wrong, must be V4. Cell-crop and inference
checks blocked with Part B.

---

## 2026-10-11 (later) — Post-C11 fixes

`results/fixes_post_C11/REPORT.md`. Phase 1 complete bar one hardware check; Phase 2
blocked; Phase 3 partially blocked.

| | |
|---|---|
| merge conflicts resolved | 4 files, 87 tests pass |
| capture buffering | implemented, **unverified — needs the camera** |
| standoff self-check | working; C11 measured 360 mm against a declared 300 |
| L → V4 | fixed and tested |
| datum → per-side linear fit | implemented, **A/B could not run** |

**Blocking the geometry modality:** `extract_single_profile` hardcodes the weld
centreline to the image centre row and takes no pose, so it cannot be run on a real
capture without registration. This blocks the datum validation, per-cell `u_cal`, and
the synthetic test. Registration (original validation Part B) is now the critical path.

**Second bug:** `compute_depth_stack` computes split-half SE over the whole frame —
8.056 mm against 0.0857 mm restricted to always-valid pixels. Feeds `noise_floor_mm`.

Budget at 300 mm with a local datum: u_noise 0.0602 (measured), u_quant 0.0289,
u_cal 0.0257 (predicted), u_align 0.020 (placeholder) → **u_total 0.0759 mm, 6.59 : 1**.

---

## 2026-10-17 — Registration pipeline implemented, hole detection needs calibration

**What was built:**

- `src/geometry/registration_v2.py` — complete pose-aware registration pipeline
- `configs/registration.yaml` — frozen registration parameters (committed before data)
- `tests/test_registration_synthetic.py` — 8 synthetic tests, all pass
- Pure-numpy fallback deprojection (works without pyrealsense2)

**Design decisions recorded in DECISIONS.md:**

1. **Rim-fitting, not opening-centroid** — validated: opening centroid bias is e/2 toward
   camera nadir, unacceptable at ~1.3 mm for Ø6 at 15° off-axis.
2. **Coupon coordinate frame physically fixed** — origin at Ø6 rim centre, s toward Ø8,
   y toward datum holes, z outward normal.
3. **Profile extraction on point clouds** — `extract_single_profile()` deprecated;
   `PointCloud.extract_profile(s_mm)` replaces it.

**Synthetic tests verify the geometry:**

| Test | Result |
|---|---|
| Opening centroid bias matches e/2 | Pass |
| Rim fit recovers true centre | < 0.02 mm error |
| Baseline accurate to 0.5 mm | Pass |
| End-for-end invariant | Pass |
| Root face handled | Pass |

**Validation on real captures:**

Ran on 36 captures. All have warnings or failures.

| Status | Count | Cause |
|---|---|---|
| Fail (missing holes) | 5 | No candidates matched diameter |
| Warn (baseline wrong) | 31 | Spurious holes detected |

**Root cause:** Adaptive thresholding (`block_size=51, C=10`) picks up every dark region
(weld spatter, shadows, oxide). C11_top_r0: 333 candidates, 17 classified as "holes".

**Next steps:**

1. Review diagnostic images: `results/diagnostics/hole_detection/*.png`
2. Add shape quality filter (circularity, solidity)
3. Add position constraints (datum holes at y > 20 mm from centre)
4. Consider Canny → Hough circles instead of blob-based detection
5. Use colour information (machined bore walls are bright)

**Not blocked:** The geometry math is correct (synthetic tests prove it). Only detection
calibration remains.
