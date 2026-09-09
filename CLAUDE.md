# weldfusion — conventions for future sessions

Honours thesis: multi-modal Dempster–Shafer fusion for structural steel
butt-weld defect classification. Architecture is locked. Do not "simplify"
a modality to emit singleton masses over all of Θ.

## Frame of discernment

Single source of truth: `configs/frame.yaml`. No other module may hardcode
the class list.

```
Θ = {N, U, P, L, C}
  N  no defect / acceptable
  U  undercut exceeding AS/NZS 1554.1 and ISO 5817 Level B (rejectable)
  P  porosity
  L  lack of fusion / lack of penetration (merged)
  C  crack (machined notch surrogate experimentally)
```

## Per-modality focal elements

A modality must never assign mass to a class it cannot sense.
`frame.validate_modality_bpa` enforces this; tests will fail if it is bypassed.

| Modality | Allowed focals | Reading |
|----------|----------------|---------|
| Vision   | {P}, {C}, {N,U}, {N,U,L}, Θ | {N,U} = surface irregularity, depth unknown. {N,U,L} = surface clean (sound, sub-visible undercut, or internal LOF). |
| Geometry | {U}, {N,P,L,C}, Θ | {N,P,L,C} = profile conforms; says nothing about internals. |
| UT       | {P}, {L}, {C}, {N,U}, Θ | {N,U} = no subsurface indication; blind to surface geometry. |

Design property (tested):

```
{N,U,L} ∩ {N,P,L,C} ∩ {N,U} = {N}
```

Pairwise, using those three "looks OK to me" reports:

```
{N,U,L} ∩ {N,P,L,C} = {N,L}     vision+geometry cannot separate N from L
{N,U,L} ∩ {N,U}     = {N,U}     vision+UT cannot separate N from U
{N,P,L,C} ∩ {N,U}   = {N}       geometry+UT *does* isolate N
```

The last identity is algebraically true of the locked focals: a conforming
profile plus no subsurface indication already rules out U, P, L and C.
Vision still cannot isolate N on its clean-surface report. Do not change
the focals to force a slogan; change the tests if the claim is restated.

Competence partitions used by the algebraic ceiling (also in `frame.yaml`):

```
vision   {{P}, {C}, {N,U,L}}
geometry {{U}, {N,P,L,C}}
ut       {{P}, {L}, {C}, {N,U}}
```

Vision is coarsened: {N,U} vs {N,U,L} are alternative visual reports, not
nested measurements that isolate L.

## Cell geometry

- Coupon: 300 × 150 × 10 mm, 15 coupons, single-V GMAW, Grade 250/350.
- Weld-axis s ∈ [0, 300] mm, origin at the first scribed datum.
- Zones A/B/C: [0,100], [100,200], [200,300] mm.
- Cell width 10 mm. Guard band 10 mm at each zone boundary.
- Local centres in each zone: 15, 25, 35, 45, 55, 65, 75, 85 mm.
- 8 cells/zone, 24/coupon, 360 total.
- A value of s in a guard band bins to `None`. No cell spans a zone boundary.

## UT projection

Indication at sound path S, refracted angle θ, plate thickness t:

```
d      = S sin θ
z_leg1 = S cos θ
z_leg2 = 2t − S cos θ
s_flaw = s_probe + S sin θ
```

Bin by **s_flaw**, not s_probe. Beyond the second leg (`S cos θ > 2t`) raises.
Mis-registering by s_probe is ~27 mm at half-skip on a 70° probe.

## Geometry conformance

ISO 5817 Level B, t = 10 mm:

```
undercut        ≤ min(0.05 t, 0.5 mm)
excess metal    ≤ min(1 + 0.1 b, 5 mm)
misalignment    ≤ min(0.1 t + 0.2, 3 mm)
```

Not a hard threshold:

```
P(non-conforming) = Φ((d_measured − d_limit) / u_total)
u_total = √(u_cal² + u_noise² + u_align²)
```

Geometry BPA mass lives only on {U} and {N,P,L,C}. The old heuristic that
mapped dimensional non-conformance onto porosity / LOF / crack is gone.

D435 noise: `δz = (z² / (f B)) δd` with B = 50 mm, δd ≈ 0.08 px, f ≈ 447 px.
≈ 0.14 mm RMS at 200 mm. Average N rows: σ/√N (25 rows → ≈ 0.028 mm).

## Fusion

- Dempster on arbitrary subsets. Return K. K = 1 raises `ConflictError`.
- Primary rule: Murphy averaging (average, then combine the average with
  itself n−1 times). Dempster alone reproduces Zadeh's paradox; that is a
  regression test, not a surprise.
- Decision: pignistic `BetP(x) = Σ_{A∋x} m(A)/|A|`, normalised by `1 − m(∅)`.
- Ablation: {V}, {G}, {U}, {V,G}, {V,U}, {G,U}, {V,G,U}.
- No 0.05 uncertainty floor. Shafer discount `α_m = 1 − balanced_accuracy`
  fitted on inner validation folds only.

## Splits and evaluation

`GroupKFold` grouped by `coupon_id`. Nested: calibration on inner folds only.
Random cell-level splits are a bug. Metrics go through
`weldfusion.evaluation.metrics.compute_metrics` (including detection rate /
FPR via binary collapse `defect ⇔ pred ≠ N`). McNemar, bootstrap CIs,
Holm–Bonferroni across the seven comparisons. Algebraic ceiling is a
deliverable, not a nicety.

Independent U ground truth must not be the profilometer under test.

## Non-goals

- Do not train any model, download any dataset, or run any experiment
  unless the user explicitly asks.
- Do not fabricate results, metrics, figures, or example data that could
  be mistaken for real output.
- Do not write a README that claims completed work.
- Do not hardcode the class list anywhere except `configs/frame.yaml`.
- Do not add dependencies beyond: torch, torchvision, numpy, scipy,
  scikit-learn, xgboost, pyyaml, opencv-python, pytest.
- DEV-ONLY synthetic UT features must never be a source of reported results.

## Current status

**Implemented (deterministic, tested):**
frame, types, config hashing, provenance manifests, UT projection, fusion
cells, D435 uncertainty, ISO 5817 exceedance BPA, multi-slice profile
extraction, outer-median base-plate fallback, BPA construct (smallest
allowed focal), Shafer discount, Dempster (arbitrary subsets, returns K),
Murphy, pignistic, 7-subset ablation, temperature / isotonic / Platt /
reliability α, GroupKFold splits, shared metrics, McNemar / bootstrap /
Holm, algebraic ceiling, UT 6-feature extractor, physics-informed DEV-ONLY
synthetic generator, vision train/eval/infer/remap ported to the 5-class
frame (unrunnable without images).

**Stubbed (`NotImplementedError` + contract docstring):**
`registration/datums.py`, `ut/io_usm100.py`, `ut/train.py`,
`geometry/baseplate.ransac_baseplate`, `pipeline/run_coupon.py`,
`blender/scripts/generate_welds.py` (original scripts produced Figure 7 of
the preliminary report; recover or rebuild).

**Not in the repo:** datasets, checkpoints, results, notebooks, thesis LaTeX.

## Next implementation task

Blocked on data, not on maths:

1. Obtain a USM 100 PC export and implement `ut/io_usm100.py` against it.
2. Recover or rebuild the Blender generator (Figure 7).
3. Download the three vision datasets, inspect folder names, extend
   `SOURCE_FOLDER_TO_CLASS`, add a YOLO/COCO → folder converter if needed.
4. Implement RANSAC base-plate (`geometry/baseplate.py`).
5. Only then: train vision, train UT on *real* A-scans with `coupon_id`
   grouping, wire `pipeline/run_coupon.py`.
