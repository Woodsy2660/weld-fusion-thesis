# 03 — Specimens, registration, cells and labels

## Specimen set as actually built

Welded October 2026 with a qualified welder, parameters logged live.

| Property | Value |
|---|---|
| Coupons | 15 |
| Dimensions | 300 × ~140 × 10 mm (two plates, each half **70 mm** wide in y) |
| Joint | Single-V GMAW butt weld, Grade 250/350, flat 1G |
| Passes | **Single pass per zone** |
| Zones | 3 per coupon × 15 = **45 zones**, 100 mm each |

### Zone allocation

| Intended class | Zones | Mechanism |
|---|---|---|
| Sound (N) | 25 | Baseline recipe |
| Lack of fusion (L) | 8 | Low heat input, high travel speed, torch off-centre to sidewall |
| Porosity (P) | 7 | Shielding gas reduced or interrupted; contaminated surface |
| Undercut (U) | 3 | Excessive travel speed at high current, torch angled into the toe |
| Excess reinforcement | 1 | High wire feed speed, low travel speed |
| Misalignment | 1 | Deliberate hi-lo at setup |
| Lack of penetration | **0 — dropped** | Single pass makes root condition a constant |
| Crack (C) | 0 in the main set | See surrogates below |

**Excess reinforcement and misalignment are not members of Θ.** They are geometric
non-conformances. They exercise the geometry module's dimensional measurement path
and its mapping to `{U}` versus `{N,P,L,C}`, and are reported as geometric-measurement
validation, not as classification classes. Code must not try to classify them into Θ.

### Crack surrogates

| Surrogate | Serves | Form |
|---|---|---|
| Machined notch | UT class C | 1.0 × 3.0 mm, EDM or slitting saw, positions recorded |
| Bend-induced crack on offcuts | Vision class C appearance | Offcut welded then bent; **not** in the registered coupon set |

Expected class C contribution: on the order of **16 A-scans**. L-versus-C separation
is attemptable, not conclusive. Any code or write-up that implies otherwise is
overclaiming.

## Registration

### Datum holes

Drilled post-weld, in **one plate only** (both holes in the same half, so they stay
together if the coupon is later sectioned).

| Hole | Position | Purpose |
|---|---|---|
| Ø6 | s = 15 mm, y = +50 mm | Origin |
| Ø8 | s = 285 mm, y = +50 mm | Axis direction; different diameter breaks the end-for-end ambiguity |
| Ø10 (optional) | s = 150 mm, y = −50 mm | Third point for an over-determined pose with a residual |

`y = 50` is measured 20 mm in from the plate's outer edge (easier to mark than from
the buried joint line). Edge distance is 2.5 × D for the Ø8 — comfortable.

**The earlier specification of y = 68 mm was wrong** and came from an assumed
152.5 mm coupon width. The real half-width is 70 mm.

### Pose solve

Primary registration is **optical, from the hole centroids**, not mechanical. Fit a
circle to each hole's boundary pixels, take the centroids, solve translation and
rotation. The fixture provides repeatable coarse seating only — a dowel in a
clearance hole gives roughly ±1 mm, six times worse than the optical solve.

Error propagation, for reference:

```
σ_θ ≈ √2 σ_c / L                     angular uncertainty of the derived axis
σ_s ≈ √(σ_c² + (d σ_θ)²)             positional uncertainty at axial distance d
```

With σ_c = 0.1 mm and L = 270 mm, σ_s ≈ 0.173 mm at the far end.

**Cross-check against the plate edges.** Detect the plate ends independently; if the
edge-derived and hole-derived origins disagree by more than 2 mm, flag that coupon.
This is free and catches a mis-drilled hole or a short plate before it poisons a fold.

### Face flipping — a known bug source

The root-face image is a **mirror** of the top-face image.

**Convention: always flip about the long (weld) axis, never end-for-end.** Then
s is preserved and only y mirrors (`y → −y`). Verification: the Ø6 hole must still
appear at the low-s end. If it appears at the high-s end, the coupon was flipped
wrongly and the capture must be rejected, not corrected in software.

Without the mirror applied, every root-face undercut measurement lands on the wrong
side of the weld — and because undercut is roughly symmetric, it looks plausible
while being wrong.

### Measured, not assumed

Record per coupon: actual overall length, actual measured datum hole separation
(calipers), and the measured s-separation used for the per-image scale check.
Do not hard-code 300 mm or 270 mm anywhere.

## Cell structure

- Weld-axis coordinate `s ∈ [0, 300]` mm, origin at the machined end face.
- **Cell width 25 mm**, matched to the UT index pitch so one A-scan belongs to one cell.
- 12 cells per coupon nominal; **the first and last cell of each coupon are excluded**
  (arc start/stop transient, clearly visible in the specimen photographs).
- **n = 150 cells.**

The cell is the unit of analysis. BPAs are defined over it, ground truth attaches to
it, metrics are computed on it, the ablation compares it, the output map is made of it.

### Cell misassignment from registration error

```
P_mis = (2/w) ∫₀^(w/2) Φ(−d/σ) dd  ≈  0.80 σ / w
```

At σ = 0.173 mm and w = 25 mm this is 0.55% — under one cell in 150.

### Declared cost of the 25 mm cell

A defect shorter than 25 mm is averaged with sound metal in the same cell, biasing L
and U detection rates downward. The width was chosen to fit a one-day instrument
hire. This is a declared limitation, not a free choice.

## Labels

### The governing principle

**Label the parameter condition, not the defect.** You know what the welder was told
to do. Whether it produced the intended defect is a hypothesis. Some zones will come
back sound despite the deviation — the rate at which that happens is a reportable
result (yield per condition) and the basis of the label-noise figure.

### Rules

1. **Cell level, not zone level.** A 100 mm zone is four cells. If the lack of fusion
   ran for only 40 mm, two cells are defective and two are not.
2. **Intended and observed in separate columns.** Their disagreement rate *is* the
   label noise figure.
3. **Surface classes may be labelled visually** — undercut, excess reinforcement,
   misalignment, surface porosity. Visual inspection is the reference method under
   AS/NZS 1554.1. State this explicitly.
4. **Lack of fusion must NOT be labelled from a photograph.** It is subsurface. Its
   label comes from the parameter condition plus the UT result. This mirrors the
   frame, which forbids vision from asserting `{L}`.
5. **Crack positions are known by construction** (notch locations). No inference.
6. **Freeze and version the label file before training.** Never edit a label because
   a model disagreed with it.

### The circularity trap

If cells are labelled by looking at photographs and a vision model is then trained on
those labels, the model's accuracy measures agreement with the labeller's own visual
judgement, not with ground truth. Rules 3 and 4 above are what keep this honest:
surface features may be labelled visually because visual inspection *is* the reference
method for them; subsurface features may not.

### Schema

```
coupon_id              e.g. C07
actual_length_mm       measured, not assumed
face                   top | root
zone_id                e.g. C07-Z2
parameter_set_id       key into the welding parameter log
cell_index             0-based within coupon
s_start_mm, s_end_mm   in the datum-hole frame
intended_condition     from the run sheet
observed_surface_class visual assessment
surface_label_source   visual | gauge | section
observed_ut_class      from the A-scan
ut_label_source        ut | none
final_label            the label used for training/evaluation
label_confidence       high | medium | low
excluded_flag          bool
exclusion_reason       end_cell | no_coupling | insufficient_depth | other
notes
```

## Ground truth tiers

| Tier | Source | Coverage |
|---|---|---|
| 1 | Recorded weld parameters per zone | All cells |
| 1b | Radiography, if obtained | Full weld length |
| 2 | Destructive macro-section | Sampled cells only; validates Tier 1, quantifies label noise |
| U reference | Calibrated weld gauge or macro-section — **never the depth camera under test** | Undercut only |

Report the yield per condition. It is an honest result and the basis of the
label-noise figure.
