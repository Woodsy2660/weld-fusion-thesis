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
