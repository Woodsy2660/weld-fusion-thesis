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

## Vision classifier output (4 heads, NOT 5 classes)

The vision classifier outputs logits for **4 focal-element heads**, NOT 5
Theta singletons. This is critical - the classifier learns to assign mass to
compound subsets, not to individual defect classes.

```
V1 → {P}       surface-breaking porosity (round holes)
V2 → {C}       surface-breaking crack (linear features)
V3 → {N,U}     surface irregularity present, depth not measurable
V4 → {N,U,L}   surface clean - sound, sub-visible undercut, or internal LOF
```

Source of truth: `configs/frame.yaml` under `vision_heads`.

Training:
- Grouped K-fold by `source_image` (patches from same image stay together)
- Class-weighted cross-entropy (V4 >> V1 in count)
- V2 is expected to be empty (no crack training data) - excluded from active_heads
- Raw logits saved for temperature scaling calibration
- Macro-F1 as primary metric (accuracy meaningless at 42:1 imbalance)

Source label mapping (`configs/class_map.yaml`):
- weld-dataset: pore→V1, discontinuity→V3 (PROVISIONAL), stain/deposit→V4
- welding-defect-kaggle: "Good Weld"→V4, "Bad Weld"/"Defect"→discard

The `discontinuity` mapping requires human review: see
`results/review/discontinuity_sample.png` contact sheet.

## Cell geometry

- Coupon: 300 × 150 × 10 mm, 15 coupons, single-V GMAW, Grade 250/350.
- Weld-axis s ∈ [0, 300] mm, origin at the first scribed datum.
- Zones A/B/C: [0,100], [100,200], [200,300] mm.
- Cell width 10 mm. Guard band 10 mm at each zone boundary.
- Local centres in each zone: 15, 25, 35, 45, 55, 65, 75, 85 mm.
- 8 cells/zone, 24/coupon, 360 total.
- A value of s in a guard band bins to `None`. No cell spans a zone boundary.

## UT projection (CORRECTED)

In conventional angle-beam weld inspection, the probe sits on the parent
plate **beside** the weld with the beam directed **ACROSS** the weld. The
beam plane is perpendicular to the weld axis. The probe is scanned ALONG
the weld. Therefore:

```
Given:
    t_tof   time of flight (μs, pulse-echo round trip)
    c_s     shear wave velocity (3.23 mm/μs for steel)
    θ       refracted angle in the test piece
    t       plate thickness (mm)
    y_probe beam-exit offset from weld centreline (mm)
    s_probe along-weld position (mm)

Sound path (one-way):
    S = c_s × t_tof / 2

Along-weld coordinate:
    s_flaw = s_probe        (always; beam is perpendicular to weld axis)

Across-weld coordinate:
    y_flaw = y_probe − S × sin(θ)

Depth via triangle-wave fold:
    u = S × cos(θ)                    total vertical travel
    leg = floor(u / t) + 1            leg number (1-indexed)
    cycle = u mod (2t)
    z = cycle           if cycle ≤ t
    z = 2t − cycle      otherwise

z is always in [0, t].
```

The triangle-wave fold handles any number of legs, not just the first two.
UT observations bin directly on probe position `s_probe`.

## UT features (9 total)

Six amplitude-domain features (unchanged):
```
peak_amplitude_db, time_of_flight_us, signal_width_6db_us,
snr_db, backwall_echo_db, echo_count
```

Three spatial features (new):
```
flaw_depth_mm       z from the projection (most discriminative)
flaw_offset_mm      y_flaw across the weld
leg_number          integer (1, 2, …), as float for ML
```

Depth is physically discriminative:
- root region (z near t) → lack of penetration
- mid-wall (z ≈ t/2) → lack of fusion
- near scanning face (z small) → porosity or cracking

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

## Two-phase pipeline

### CellTable

The fusion cell is the unit of analysis. A `CellTable` (in `celltable.py`)
carries one row per cell. Every stage adds columns:

| Stage       | Columns                                                |
|-------------|--------------------------------------------------------|
| binning     | coupon_id, cell_id, zone, s_centre_mm                  |
| acquisition | rgb_patch_ref, depth_slice_ref, ascan_refs             |
| models      | vision_logits, geometry_measurements, ut_features      |
| calibration | vision_posterior, ut_posterior, geometry_exceedance    |
| bpa         | vision_bpa, geometry_bpa, ut_bpa                       |
| fusion      | fused_bpa, conflict_K, betp, predicted_label           |
| groundtruth | gt_label, gt_tier, gt_source                           |

### Phase 1: Batch (`pipeline/phase1_batch.py`)

Runs over ALL cells at once, per CV fold:
- Per-modality model inference (batched for GPU efficiency)
- Fit temperature scaling on inner validation fold ONLY
- Fit isotonic/Platt for UT on inner validation fold ONLY
- Estimate reliability α_m per modality on validation folds ONLY

Writes the model and calibration columns. MUST NOT see test-fold labels.

### Phase 2: Per-cell (`pipeline/phase2_percell.py`)

Runs per cell, independently (embarrassingly parallel):
- Calibrated posterior → mass over declared focal elements
- Shafer discounting by α_m
- Subset selection (ablation = column selection)
- Dempster combination, returning conflict K
- Murphy averaging
- Pignistic transform → predicted label

Writes the bpa and fusion columns.

### Orchestrator (`pipeline/run_coupon.py`)

Thin driver: bin → phase 1 → phase 2 → emit outputs.

## Reconstruction output (`output/reconstruction.py`)

From the completed CellTable:

1. **Per-coupon defect map**: predicted class and belief as f(s).
2. **3D surface model**: z(s, y) from the geometry modality (true 3D).
3. **Stacked subsurface sections**: per-cell (y, z) positions of UT
   indications, stacked along s.

**ACCURACY CAVEAT**: The surface model is true 3D. The subsurface sections
are 2D per cell, stacked along s — a single probe angle from one side is
NOT a tomographic reconstruction. Describe as "3D surface with stacked 2D
subsurface sections" or "2.5D". Do NOT call it a 3D reconstruction of the
weld volume.

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
  scikit-learn, xgboost, pyyaml, opencv-python, pytest, pandas.
- DEV-ONLY synthetic UT features must never be a source of reported results.

## Current status

**Implemented (deterministic, tested):**
frame, types, config hashing, provenance manifests, UT projection (corrected
across-weld physics, generalized triangle-wave fold, s_flaw = s_probe),
fusion cells, D435 uncertainty, ISO 5817 exceedance BPA, multi-slice profile
extraction, outer-median base-plate fallback, BPA construct (smallest
allowed focal), Shafer discount, Dempster (arbitrary subsets, returns K),
Murphy, pignistic, 7-subset ablation, temperature / isotonic / Platt /
reliability α, GroupKFold splits, shared metrics, McNemar / bootstrap /
Holm, algebraic ceiling, UT 9-feature extractor (6 amplitude + 3 spatial),
physics-informed DEV-ONLY synthetic generator (depth is causal), CellTable
columnar structure, two-phase pipeline (phase1_batch, phase2_percell),
reconstruction output with accuracy caveat.

**Vision pipeline (4-head focal elements, ready to train):**
- `build_patches.py`: extracts patches from weld-dataset + welding-defect-kaggle
- `dataset.py`: PatchDataset with head labels (V1-V4)
- `splits.py`: grouped K-fold by source_image, leakage validation
- `train.py`: class-weighted CE, macro-F1, saves logits for calibration
- `review_discontinuity.py`: generates contact sheet for V3 mapping review
- Patches built: 25,667 total (V1: 523, V2: 0, V3: 2977, V4: 22167)
- Source images: 3050 (no leakage between folds verified)

**Stubbed (`NotImplementedError` + contract docstring):**
`registration/datums.py`, `ut/io_usm100.py`, `ut/train.py`,
`geometry/baseplate.ransac_baseplate`, `pipeline/run_coupon.run_full_experiment`,
`blender/scripts/generate_welds.py`.

**Not in the repo:** datasets, checkpoints, results, notebooks, thesis LaTeX.

## Next implementation task

**Ready now:**
1. Train vision classifier: `python -m weldfusion.vision.train`
   - Review `results/review/discontinuity_sample.png` first
   - If V3 mapping looks wrong, update `configs/class_map.yaml` and rebuild patches

**Blocked on data/hardware:**
2. Obtain a USM 100 PC export and implement `ut/io_usm100.py` against it.
3. Recover or rebuild the Blender generator (Figure 7).
4. Implement RANSAC base-plate (`geometry/baseplate.py`).
5. Train UT on *real* A-scans with `coupon_id` grouping.
6. Wire the full experiment loop.
