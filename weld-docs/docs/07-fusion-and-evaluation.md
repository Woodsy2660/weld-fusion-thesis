# 07 — Fusion and evaluation

## Fusion pipeline

Per cell, in order:

1. Construct the three BPAs from the per-modality model outputs.
2. Apply Shafer discounting with `α_m = 1 − balanced_accuracy_m` from validation folds.
3. Combine: Murphy averaging (primary) or raw Dempster (ablation).
4. Record the conflict mass K.
5. Pignistic transform for the point decision.
6. Record `[Bel, Pl]` for every singleton.

Outputs per cell: predicted class, belief interval per class, conflict mass,
and which modalities contributed non-vacuous evidence.

## The seven-condition ablation

```
V, G, U, V+G, V+U, G+U, V+G+U
```

Implemented as **column selection over the Phase 1 output**, re-running only Phase 2.
Do not re-train models per condition. A modality excluded from a condition
contributes a vacuous BPA (`m(Θ) = 1`), which is the identity under Dempster's rule —
assert that this is true in a test.

### Against the algebraic ceiling

For each subset, compute the set of classes reachable by conjunction of that subset's
declared focal elements. This is **derived from the frame alone, before any data**.
Report empirical performance against it.

This separates two kinds of loss:
- **Structural loss** — the subset cannot reach the class, no classifier can fix it.
- **Classifier loss** — the subset can reach it and the models failed.

Reporting both is the novel part of the ablation. A V+G condition that fails to
separate L from C has not underperformed; it has hit its ceiling.

## Splits

| Data | Group by |
|---|---|
| Real specimen cells | `coupon_id` |
| Public image corpora | `source_sequence` |

Nested GroupKFold. Calibration (temperature scaling, isotonic regression, reliability
estimation for discounting) sees **inner folds only**.

## Metrics

- Per-class and macro precision, recall, F1.
- Confusion matrix.
- **Expected calibration error** — mandatory, not optional. Discounting depends on it.
- Detection rate and false positive rate via binary collapse (`ŷ ≠ N`).
- Conflict mass K distribution.

## Statistics — the part most likely to be done wrong

### McNemar's test

For paired comparisons between modality subsets:

```
χ² = (|b − c| − 1)² / (b + c)
```

where b and c are the discordant cell counts. The −1 is the continuity correction.

### Holm–Bonferroni

Sort p-values ascending, reject p₍ᵢ₎ if

```
p₍ᵢ₎ ≤ α / (m − i + 1)
```

stopping at the first failure.

**Declare the comparison family size m before computing anything.** The seven subsets
do not give seven independent comparisons. The defensible family is the
single-modality-added contrasts plus the final one:

```
V → V+G,  V → V+U,  G → G+U,  V+G → V+G+U        →  m = 4
```

Not m = 7. Write the chosen family into the results code as a constant with a comment.

### Clustered confidence intervals — do not skip this

Cells within a coupon are correlated. For clustered data with m cells per coupon and
intra-coupon correlation ρ:

```
Var(x̄) = (σ²/n) [1 + (m − 1) ρ]
```

The bracket is the **design effect** d_eff. Effective sample size is n/d_eff.

With n = 150, m = 10 and ρ = 0.5:

```
d_eff = 1 + 9(0.5) = 5.5
n_eff = 150 / 5.5 ≈ 27
```

**So the honest sample size for coupon-level effects is about 27, not 150.**

Consequences for the code:
- Estimate ρ from the data (one-way ANOVA on per-cell errors grouped by coupon) and
  report the measured design effect.
- Report **coupon-level clustered bootstrap CIs** alongside cell-level McNemar.
  Resample coupons with replacement, not cells.
- Cell-level McNemar alone is anti-conservative here. Reporting both, and saying why,
  turns a vulnerability into evidence of competence.

## Human baseline

- **Wall & Burch** sub-3 mm manual UT as the matched comparison.
- **Campbell et al. (2019)** as context only, caveated — different inspection method
  and defect population.

Do not compare the fused system directly to Campbell's 65% as if it were a like-for-like
benchmark.

## Required outputs

1. Per-coupon defect map: class and belief along s.
2. 3D surface model from depth.
3. Stacked 2D subsurface sections from the UT (y, z) projection.
4. Ablation table: seven subsets, empirical vs algebraic ceiling.
5. Conflict K distribution.
6. Predicted vs measured geometric uncertainty.
7. Sequence-leakage audit table.

**Describe (2)+(3) as "3D surface with stacked 2D subsurface sections", or 2.5D.**
One probe angle from one side is not a tomographic reconstruction. Do not let a
figure caption imply otherwise.

## Regression tests that must exist

- Zadeh's counterexample under raw Dempster produces the documented pathology.
- Murphy averaging does not.
- A vacuous BPA is the identity under combination.
- Depth-fold round trip recovers the known depth.
- Letterbox `pad_fraction == 1 − 1/AR` to numerical tolerance.
- Every BPA sums to 1 and has `m(∅) = 0`.
- Algebraic ceiling for `{V,G}` returns exactly `{N}, {U}, {P}` and does not contain
  `{L}` or `{C}`.
