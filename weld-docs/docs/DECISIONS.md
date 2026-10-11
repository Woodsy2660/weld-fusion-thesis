# Decision log

Append-only. Newest at the top. Never delete an entry — supersede it with a new one
that references it.

Format: date, decision, reasoning, what it supersedes.

---

## 2026-10-09 — Datum hole lateral position corrected to y = 50 mm

Earlier specification put the holes at y = +68 mm, derived from an assumed 152.5 mm
coupon width. The plates are 70 mm per half. At y = 68 the Ø8 hole would have 2 mm of
edge distance and would break out. y = 50 gives 20 mm edge distance — 2.5 × D for the
Ø8. Marked 20 mm in from the outer edge, since the joint line is buried under the bead.

**Supersedes** the y = +68 figure in `system-configuration-v1.md` and `v2`.

---

## 2026-10-09 — Both datum holes in one plate half

Two holes fully determine in-plane pose (4 measurements, 3 unknowns, with the
separation as a scale check). They must be separated **along** the weld axis, because
angular precision goes as √2·σ_c/L in the baseline parallel to the axis of interest.
Both in the same half keeps them in one rigid piece if the coupon is later sectioned.
An optional third hole on the opposite half at mid-span gives an over-determined pose
with a residual, which is a quality check rather than a precision gain.

---

## 2026-10-08 — End cells excluded; n = 150

Arc start/stop craters are clearly visible at the plate ends in the specimen
photographs. Those cells represent an arc transient, not any defect recipe. Excluding
the first and last cell of each coupon removes 30 cells from 180.

---

## 2026-10-08 — Registration is optical-primary, mechanical-secondary

A dowel in a Ø8 clearance hole gives roughly ±1 mm of play, which over the 270 mm
span is ±1.06 mm at the far end. Optical centroid fitting gives ~0.17 mm. The fixture
therefore provides repeatable coarse seating; the fine pose comes from the hole
centroids in the image.

**Supersedes** the "mechanical first, optical confirms seating" statement in
`system-configuration-v1.md`.

---

## 2026-10-08 — Surface left as-welded; no selective cleaning

Flash rust appeared on the beads within hours. Cleaning some coupons and not others
would correlate surface condition with capture order, which correlates with coupon,
which is the cross-validation fold group — a textbook confound. As-welded is also the
realistic field condition.

---

## 2026-10-07 — Camera standoff 300 mm, not 200 mm

The D435i minimum depth is ~280 mm at 1280×720. The 200 mm figure came from a D405
datasheet. At 300 mm: 0.445 mm/px depth, 0.215 mm/px colour, 569 × 333 mm field of
view, single-frame δz = 0.214 mm recovering to 0.027 mm over 64 frames.

---

## 2026-10-06 — Vision trains on `final_welding` alone

The three aggregated corpora were originally unlabelled, segmented under different
conventions, and carry different acquisition geometry. Combining them bought sample
count at the cost of a label space that does not mean the same thing across sources,
which is what the LOSO result exposed. `final_welding` is also closest in appearance
to the real coupons. Matching the deployment distribution beats maximising n.

The original three-stage plan is reduced to two: a clean baseline, then an
initialisation ablation on real coupon images.

---

## 2026-10-05 — Class weights derived from the training split only

An earlier specification pinned weights to corpus-wide values. Weights, like every
other fitted quantity, must come from the split in use. For the current split,
`w = 35105/(3n)` gives 2.9677 / 0.7712 / 0.7319, normalising to
1.9914 / 0.5175 / 0.4911. The acceptance test is the derivation, not the numbers.

---

## 2026-10-04 — Cell width 25 mm; UT index 25 mm

The one-day instrument hire will not support a 10 mm index (~450 positions). At 25 mm
it is ~180. The cost is spatial resolution: a defect shorter than a cell is averaged
with sound metal, biasing L and U detection downward. Declared as a limitation.

---

## 2026-10-02 — UT primary probe angle 60°, not 70°

Normal incidence on a sidewall of bevel half-angle β requires θ = 90° − β. With
β = 30°, θ = 60°. A 70° beam strikes the sidewall 10° off normal and loses signal.
70° retained as secondary for near-surface coverage.

**Supersedes** an earlier recommendation of 70° as primary, which was wrong.

---

## 2026-10-01 — Lack of penetration dropped from the frame

Single-pass welding makes root condition a constant, not a controlled variable. L now
means lack of fusion only. The phrase "LoF/LoP" is retired.

---

## 2026-09-28 — Vision frame reduced to three heads

The `{C}` head is removed and C absorbed into `{N,U,L,C}`. The corpora support no
crack claim in either direction, and a head that cannot be trained is worse than an
honest ignorance assignment.

---

## 2026-09-25 — Chance baseline is 0.3023, not 0.33

An agent report claimed 0.33 by assuming balanced classes. The test set is not
balanced. Simulating uniform guessing against the actual class distribution gives
macro-F1 0.3023. The conclusion (source B at or below chance) held, but the reasoning
was wrong and had to be corrected.

---

## 2026-09-22 — Group public-corpus splits on `source_sequence`

A sequence audit found 81.59% of samples sharing a source sequence with a sample in
another split. All in-distribution figures from the prior pipeline are invalid. This
is reported as a methodological finding, not hidden.

---

## 2026-10-11 — First real capture (C11, S1) invalidates the 300 mm uncertainty budget

**Context.** Two captures of coupon C11 (top and root faces) were taken to validate the
geometry and vision pipelines end to end. Full analysis in
`results/validation_C11/REPORT.md`.

**What changed.**

1. **The 0.078 mm predicted `u_total` does not survive contact with real data, and the
   budget should not be quoted until re-anchored.** Measured single-frame depth noise is
   0.487 mm against 0.214 mm predicted, and the parent-metal plane-fit RMS — which *is*
   the `u_cal` term by definition — is 0.95 mm against 0.030 mm assumed. Part of this is
   standoff (the capture was at 358 mm, not 300 mm, and noise goes as z²), but the
   range-corrected prediction is still only 0.30 mm, so roughly 1.6× is unexplained.

2. **Temporal averaging delivers 5.7–7.0×, not 8×.** Measured by split-half on 64
   frames. The module doc anticipated this; it is now quantified. Block and interleaved
   split-half differ by 12%, which is drift within the capture window.

3. **The global-plane undercut datum is confirmed invalid for this coupon.** The two
   parent-metal half-plates differ in orientation by 2.37°, implying 2.90 mm of
   deviation at the 70 mm plate edge against a 0.5 mm undercut limit. This was listed as
   an open issue; it is now observed. A local per-cell, per-side reference is required.

4. **`u_align` remains a config placeholder.** No repeat-placement captures exist — C11
   was captured at `r0` only, so the protocol's repeatability step was skipped.

**Not changed.** No threshold, frozen parameter or config value was altered in response
to this data. Changing a frozen parameter to fit observed data is the failure this
project has already documented once.

**Blocking issue found.** `configs/geometry.yaml`, `src/geometry/profile.py`,
`src/geometry/rules.py` and `.gitignore` carry unresolved merge-conflict markers; the
geometry module raises `SyntaxError` on import. Parts B and C of the validation could
not run. The HEAD side of `geometry.yaml` is the stale one — it still contains
`working_distance_mm: 200`, the known-wrong D405 figure, and a four-class list
predating the focal-element frame.

**Still open.** `final_welding/Lack of Fusion` is mapped to V3 `{N,U}`. L is not in
`{N,U}`. This is known-issue #3 in `00-START-HERE.md` and remains unfixed.

---

## 2026-10-11 — Post-C11 fixes: datum, capture buffering, standoff check, L→V4

Full detail in `results/fixes_post_C11/REPORT.md`.

**Merge resolved.** `configs/geometry.yaml` took the incoming side (HEAD carried
`working_distance_mm: 200`, the D405 figure). `profile.py` and `rules.py` took the
incoming implementations; the HEAD stubs had zero references anywhere.

**Datum changed from a band median to a per-side fitted line.** `profile_extraction.
datum_mode: linear`. The previous datum was already per-side and per-profile — it was
*not* a global plane, contrary to the earlier report, which inferred the behaviour
from config because the module would not import. The real defect: a median of the
15–40 mm band represents y ≈ 27.5 mm while undercut is measured at the toe, y ≈ 12 mm.
Extrapolating a constant across that 15.5 mm costs `slope × 15.5 mm` = **0.643 mm** at
C11's measured 2.37° inter-plate angle, against a 0.5 mm limit. Curvature over the band
is only 0.026 mm, so the dominant term is slope, not curvature. A linear fit removes
the slope term. **Before/after could not be measured** — see below.

**Capture buffering.** Acquisition now fills RAM and writes after `pipe.stop()`. C11
lost 49 of 113 device frames to a 1920×1080 PNG encode inside a 33 ms loop. Measured
drop rate before: 49/113 (43%). After: **not yet measured — needs the camera.**

**Standoff self-check.** `standoff_mm_from_depth` is computed from the depth map and
written to `meta.json`; `check_captures.py` fails a capture differing from the declared
value by more than 10 mm. C11 declared 300.0 and measured 360.3 / 361.1 / 360.3 mm —
including the reference target, so the whole session was 60 mm out.

**L → V4.** `final_welding/Lack of Fusion` now maps to V4 `{N,U,L,C}`. L is not an
element of `{N,U}`. Known-issue #3, closed. A general test now asserts that no source
label naming a Θ symbol maps to a head excluding it.

**Two findings recorded, not fixed.**

1. `extract_single_profile` hardcodes the weld centreline to the image centre row and
   takes no pose. It cannot be run on a real capture until registration supplies one.
   This blocks the datum A/B, the per-cell `u_cal`, and the synthetic test.
2. `compute_depth_stack` computes split-half SE over the whole frame including
   background: 8.056 mm against 0.0857 mm when restricted to always-valid pixels, a
   94× error. `noise_floor_mm` carries it into the uncertainty budget.

**Not changed.** No threshold or frozen parameter was tuned to make anything pass.
