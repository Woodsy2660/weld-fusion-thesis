# Validation against the first real capture — coupon C11, session S1

**Date:** 11 October 2026 · **Captures:** `data/raw/S1/C11_top_r0`, `data/raw/S1/C11_root_r0`
**Device:** D435i s/n 036222070081, FW 5.17.3.10, SDK 2.58.4, USB 3.2

> **DATA QUARANTINE.** These two captures are a pipeline validation artefact. They are
> not training data, not evaluation data, and not part of the specimen dataset. They
> must never appear in a train split, test split, fold, calibration set or reported
> result. No label file has been created for them and none should be.

---

## 1. Verdict

**No.** The geometry pipeline cannot produce per-cell BPAs from this data at all,
because `src/geometry/profile.py`, `src/geometry/rules.py` and `configs/geometry.yaml`
contain unresolved Git merge-conflict markers and raise `SyntaxError` on import — Parts
B and C could not be executed. That is a repository state problem and is trivially
fixable. The **single biggest obstacle is not that**: it is that the capture was taken
at **358–360 mm standoff, not the frozen 300 mm**, and the resulting data cannot
resolve the quantity the modality exists to measure. Single-frame depth noise is
**0.487 mm** and the parent-metal plane-fit RMS is **0.95 mm**, against an ISO 5817
Level B undercut limit of **0.5 mm**. The datum surface is noisier than the defect.
Separately, the two parent-metal half-plates differ in orientation by **2.37°**, which
is 4.7× the 0.5° threshold at which a single global plane stops being a valid undercut
datum and implies **2.90 mm** of deviation at the 70 mm plate edge. Even with the merge
resolved, undercut measured against a global plane from this capture would be
dominated by plate distortion and noise, not by undercut.

---

## 2. Findings

| # | Sev | Finding | Evidence | Proposed fix |
|---|---|---|---|---|
| 1 | **blocker** | Three geometry files have unresolved merge conflicts; the module cannot be imported | `git status`: `UU configs/geometry.yaml`, `UU src/geometry/profile.py`, `UU src/geometry/rules.py`. `import src.geometry.pipeline` → `SyntaxError: invalid syntax (profile.py, line 1)`. Conflict markers at profile.py:1/76/699, rules.py:1/64/384, geometry.yaml:1/31/170 | Complete the merge. The HEAD side of `geometry.yaml` is the stale one — it carries `working_distance_mm: 200` (the known-wrong D405 figure) and a 4-class `classes:` list that predates the focal-element frame. The incoming side is correct. **Not fixed here, per rule 2.** |
| 2 | **blocker** | Capture taken at 358–360 mm, not the frozen 300 mm | Plate-plane median depth 358.3 mm (top) / 359.8 mm (root); modal scene depth 355 mm. `meta.json` records `standoff_mm_measured: 300.0` | The 300.0 is the untouched `$STANDOFF` placeholder in `run_session.ps1`. Measure and enter the real standoff; re-capture at 300 mm. Depth noise scales as z², so 358 mm costs a factor (358/300)² = 1.42 before anything else |
| 3 | **blocker** | Measurement noise exceeds the quantity being measured | Single-frame per-pixel σ 0.487 mm (top) / 0.549 mm (root). Parent-metal plane RMS 0.95 mm. Undercut limit 0.5 mm | Not fixable by averaging alone — see #5. Reduce standoff, and establish whether the surface finish is defeating the stereo matcher |
| 4 | **blocker** | Angular distortion between the two parent-metal half-plates is 2.37° | Side A normal vs side B normal, top face. 70·tan(2.37°) = **2.90 mm** at the plate edge | A single global plane is not a valid undercut datum for this coupon. Use a local per-cell, per-side reference. This is the known open issue in `DECISIONS.md` and this capture **confirms** it |
| 5 | important | Temporal averaging delivers 5.7–7.0×, not the assumed 8× | median σ 0.487 mm; SE split-half 0.0857 mm (block) / 0.0764 mm (interleaved) → reduction 5.68× / 6.37×. Naive √64 = 8× | Use the measured factor in the uncertainty budget. The doc already predicted this; it is now quantified |
| 6 | important | 49 of 113 device frames were dropped in transit | `dropped_depth_frames: 49` on both faces; frame numbers span 27→139 for 64 saved frames, 10 skip events, max gap 6 | 64 real frames were saved, so the average is valid, but it spans 3.8 s rather than 2.1 s while the ASIC warms 36→41 °C. Cause is almost certainly synchronous I/O in the capture loop (1080p PNG encode per frame). Buffer frames in RAM and write after the loop |
| 7 | important | Block and interleaved split-half disagree, indicating drift within the capture | 0.0857 vs 0.0764 mm, block 12% worse | Consistent with the 5 °C ASIC rise across a 3.8 s capture. Fixing #6 shortens the window and should reduce it |
| 8 | important | `final_welding/Lack of Fusion` is still mapped to V3 `{N,U}` | `load_class_map()['final_welding']['Lack of Fusion']` → `'V3'`; `'L' in frame.focal_element('V3')` → `False` | This is known-issue #3 in `00-START-HERE.md`, still unfixed. L is not in `{N,U}`. Must map to V4 `{N,U,L,C}` |
| 9 | important | Bead detection fails on the root face | Root bead located at transverse −62.2 mm with peak height 0.31 mm, against −5.0 mm / 1.99 mm on the top face. Side-B parent ROI collected 0 points, so no angular distortion could be computed for the root face | Either the root face has no raised bead (physically plausible) or the plate isolation failed — root transverse extent measured 212.7 mm against 150.5 mm on top and a 140 mm nominal plate. Inconclusive; see Open questions |
| 10 | minor | `u_align` is the config placeholder, not a measurement | `pipeline.py:131` → `unc_cfg.get("u_align_mm_default", 0.020)`. The comment on line 129 says "measured from repeat-placement captures", which is not what the code does | Either rename the key to make the placeholder obvious, or capture the repeat placements. `C11` has `r0` only — no repeat captures exist |
| 11 | minor | Derived lateral scale is 19.5% off the assumed value | 0.5611 mm/px at the real standoff, against 0.4698 mm/px at 300 mm (depth) and 0.2615 vs 0.2190 mm/px (colour) | Follows from #2. Any code that assumes 0.445 or 0.470 mm/px is wrong for this capture |
| 12 | minor | Parent metal is not flat at the scale of the measurement | Transverse profiles show 1–2 mm structured undulation across the ROI bands, with strong row-correlated striping | Partly #3, partly real plate distortion (#4). Needs the step-block experiment to separate sensor error from specimen shape |

---

## 3. Part A — independent diagnostic

`capture-kit/diagnose_capture.py`, written for this task. It reads the `.npy` files
directly and imports nothing from `src/`, so it remains an independent witness.

| Quantity | C11_top_r0 | C11_root_r0 | Expected |
|---|---|---|---|
| median depth (mm) | **358.3** | **359.8** | ≈ 300 |
| `standoff_mm_measured` in meta | 300.0 | 300.0 | the real value |
| lateral scale (mm/px) | **0.5611** | 0.5634 | ≈ 0.470 at 300 mm, fx 638.59 |
| valid fraction, whole frame | 0.2631 | 0.2893 | — |
| median per-pixel std (mm) | **0.4868** | 0.5491 | 0.214 at 300 mm |
| SE naive = std/√64 (mm) | 0.0609 | 0.0686 | 0.027 |
| SE split-half, block (mm) | 0.0857 | 0.0973 | — |
| SE split-half, interleaved (mm) | 0.0764 | 0.0788 | — |
| **achieved reduction factor** | **5.68** block / 6.37 interleaved | 5.64 / 6.97 | naive 8.00 |
| plane fit side A: rms / points | 0.9841 mm / 23,131 | 0.8360 mm / 21,868 | u_cal ≈ 0.030 mm |
| plane fit side B: rms / points | 0.8664 mm / 20,473 | **n/a / 0** | — |
| plane fit BOTH: rms / points | 0.9499 mm / 43,604 | 0.8360 mm / 21,868 | — |
| **angular distortion A vs B (deg)** | **2.3748** | not computable | < 0.5 |
| implied deviation at 70 mm edge | **2.9031 mm** | — | — |
| camera tilt (deg) | 3.5058 | 4.0621 | < 5 (config max) |
| plate extent long / transverse (mm) | 297.0 / 150.5 | 295.9 / 212.7 | 300 / 140 |
| ASIC temp start → end (°C) | 36 → 41 | 37 → 40 | — |
| dropped device frames | 49 | 49 | 0 |

### The transverse profiles, in words

`transverse_profiles_C11_top_r0.png`. **There is a recognisable bead** — a single clean
peak at transverse ≈ −5 mm rising 2–3 mm above the surrounding surface, present in all
five slices and unambiguous in the bead-locator panel. Its position is consistent
along the weld axis, so the plate isolation and axis assignment are working on the top
face.

**The parent metal is not flat.** Across the two green ROI bands the surface wanders by
1–2 mm in smooth, structured undulation — not noise, and far larger than the 0.030 mm
plane-fit residual the uncertainty budget assumes. The slice at s = −115 mm is a
monotonic ramp from −1 mm to +2 mm across 70 mm of transverse travel, which is the
2.37° inter-plate angle showing up directly.

**Strong row-correlated striping is visible throughout** — the points separate into
diagonal bands rather than a cloud. That is fixed-pattern stereo structure, and it is
why the interleaved split-half (which cancels drift but not fixed pattern) still sits
at 0.076 mm rather than the 0.027 mm the naive model predicts.

**No undercut groove is identifiable at either toe.** At 0.49 mm single-frame noise and
0.95 mm plane RMS, a 0.5 mm groove would be indistinguishable from the surrounding
undulation. This is a statement about the measurement, not about the coupon.

**Dropout is substantial** — 26–29% whole-frame validity, with large contiguous gaps
(s = 61 mm is missing the entire −60 to −25 mm span; s = 120 mm has gaps either side of
the bead).

---

## 4. Parts B and C — not executed

Both are blocked by finding #1. `src/geometry/pipeline.py` imports
`src/geometry/profile.py`, which begins with `<<<<<<< HEAD` and raises `SyntaxError`
before any code runs. No stage of B could be attempted — not registration, not cell
indexing, not profile extraction, not measurement, not the uncertainty budget, not BPA
construction. Part C depends on B's per-face output and is blocked transitively.

What this blocks, specifically, that the capture would otherwise have tested:

- **B1** datum hole detection, measured hole separation against the nominal 270 mm, the
  plate-edge cross-check, and false-positive rejection. The specular dropout concern is
  real and testable with this data — whole-frame validity is 26% and the bead region
  shows large dropout, which is exactly the blob population that can mimic a Ø6 hole.
- **B5** the uncertainty budget against real numbers. Part A already supplies two of the
  terms it would have used, and both are far from the configured values.
- **C1** the flip convention and the `y → −y` mirror. This is the check the two-face
  capture was taken for, and it remains untested.

---

## 5. Part D — vision plumbing

| Check | Result |
|---|---|
| 1. Colour loads at correct orientation/scale | 1920×1080, matches `color_intrinsics`. Derived scale **0.2615 mm/px** at the real 358.3 mm standoff (0.2190 at nominal 300 mm; the doc assumes 0.215) |
| 2. Cell crops land on the weld | **Could not test** — cell boundaries come from the geometry pipeline (blocked). No contact sheet produced |
| 3. Model accepts crop size, runs | **Could not test** — no cells to crop |
| 4. Output is a BPA over three heads, `{C}` absent | **PASS.** `frame.head_names` = `['V1','V3','V4']`, n = 3. Focal elements `V1:{P}`, `V3:{N,U}`, `V4:{N,U,L,C}`. No head equals `{C}` |
| 5. Confidence distribution not degenerate | **Could not test** — no inference run |
| 6. `Lack of Fusion → V4` mapping | **FAIL.** Still mapped to **V3 `{N,U}`**. `'L' in V3` is `False`, so the mapping asserts a focal element that does not contain the symbol it is labelling |

No accuracy number is reported, and none should be — the model is untrained against the
corrected frame and the LOSO result (macro-F1 0.43 / 0.22) says the public-corpus
weights do not transfer.

---

## 6. Figures

| File | Description |
|---|---|
| `transverse_profiles_C11_top_r0.png` | Five transverse slices along the weld axis plus the bead-locator trace, top face. Bead at −5 mm, parent ROI bands shaded |
| `transverse_profiles_C11_root_r0.png` | Same for the root face. Bead locator peaks at −62 mm with 0.31 mm amplitude — the detection failure in finding #9 |
| `part_a_diagnostic.json` | All Part A quantities, machine-readable |
| `run_manifest.json` | Commit, seed, input file hashes, package versions |

---

## 7. Open questions

1. **Is the root face genuinely beadless, or did plate isolation fail?** Root transverse
   extent measured 212.7 mm against 140 mm nominal and 150.5 mm on the top face, which
   suggests the isolation pulled in jig or table. Resolving this needs either the colour
   image inspected against the depth mask, or a capture with the coupon better separated
   in depth from its surroundings. I did not want to tune the isolation thresholds to
   make it work — that is the failure mode rule 1 prohibits.
2. **Why is single-frame noise 0.49 mm when the range-corrected prediction is 0.30 mm?**
   Even after the (358/300)² = 1.42 penalty, measurement is 1.6× worse than predicted.
   Candidates: surface finish defeating the stereo matcher, exposure at 200 µs being too
   short for this surface, or laser power 240 being wrong for this standoff. The
   emitter A/B test in SOP step 6 would separate these.
3. **What is the real standoff, and was the rig at 300 mm with a measurement error, or
   actually at 358 mm?** `standoff_mm_measured` is the untouched placeholder so it
   carries no information either way.
4. **Does any of the 2.37° inter-plate angle come from the fixture rather than the
   coupon?** A capture of the flat reference target in the same fixture would separate
   them. `REF_top_r0` exists in this session and was not analysed here.

---

## 8. Known-issue status

| Issue | Status | Evidence |
|---|---|---|
| Angular distortion invalidating a global plane datum | **CONFIRMED** | 2.3748° between parent half-plates, 4.7× the 0.5° threshold; 2.90 mm implied at the 70 mm edge |
| Datum detection in depth rather than colour (resolution, false positives) | **COULD NOT TEST** | Registration code is behind the merge-conflict blocker. The precondition is concerning: 26% whole-frame validity with large dropout blobs near the bead |
| `u_align` configured rather than measured | **CONFIRMED** | `pipeline.py:131` reads `u_align_mm_default` (0.020 mm). No repeat-placement captures exist — C11 has `r0` only |
| `u_noise` missing the non-averaging floor | **COULD NOT TEST** in the pipeline, but **CONFIRMED in principle** by Part A | Split-half gives 0.0764–0.0857 mm and by construction cancels fixed-pattern error. The visible row-correlated striping is exactly that error, so split-half alone under-reports. `noise_floor_mm_default: 0.015` in config is an assumption, not a measurement |
| Governing-undercut-across-both-faces logic absent | **COULD NOT TEST** | Blocked with Part C |
| RANSAC seed not fixed | **NOT OBSERVED** — no seeding found in `src/geometry/registration.py` (grep for `seed`/`random_state`/`default_rng` returns nothing), but the measurement code that would use RANSAC is behind the blocker, so this could not be confirmed end to end. `diagnose_capture.py` fixes its own seed at 0 |
| Depth units read from `meta.json` rather than hard-coded | **COULD NOT TEST** | No `depth_units` or `0.0001` reference found in `registration.py` or `pipeline.py`; the consuming code is in the conflicted `profile.py` |

---

## 9. What would make the next capture answer more

In priority order, and none of these are code changes:

1. Set the real standoff in `run_session.ps1` and verify it against the plate before
   capturing. The 300.0 placeholder is currently carrying a 58 mm error into the record.
2. Capture `REF_top_r0` analysis alongside the coupon to separate fixture tilt from
   plate distortion.
3. Take the repeat placements the protocol specifies (`C11` at `r1`, `r2`), so `u_align`
   becomes measured rather than assumed.
4. Run the step block. Part A shows the predicted uncertainty budget is wrong by more
   than an order of magnitude at this standoff; the budget needs a measured anchor
   before any undercut number is reportable.
