# Vision Results Reconciliation

**Date:** 25 October 2026
**Auditor:** Claude Opus 4.5
**Scope:** Coupon-grouped k=4 cross-validation on real weld specimens

---

## Task 0: Label Convention Dependency

### Trivial Baseline Definition

**Value:** 0.2338
**Definition:** "most_frequent" — always predict the majority class V4 `{N,U,L}`
**Provenance:** Hardcoded constant at `scripts/finetune_coupon_kfold.py:291`

```python
# Compare to baselines
trivial_f1 = 0.2338
```

**NOT dynamically computed.** The UT pipeline uses `sklearn.dummy.DummyClassifier(strategy='most_frequent')` (`src/ut/train.py:64`), but the vision script hardcodes the constant.

### Verification

Current vision_target distribution (from `data/labels_v1_2026-10-17.csv`):

| Vision Target | Count | Fraction |
|---------------|-------|----------|
| `{P}` (V1)    | 26    | 17.3%    |
| `{N,U}` (V3)  | 43    | 28.7%    |
| `{N,U,L}` (V4)| 81    | 54.0%    |
| **Total**     | 150   | 100%     |

For "always predict V4":
- Class `{P}`: TP=0, FN=26 → F1 = 0
- Class `{N,U}`: TP=0, FN=43 → F1 = 0
- Class `{N,U,L}`: TP=81, FP=69, FN=0 → P=0.54, R=1.0 → F1 = 0.7013

**Macro-F1 = (0 + 0 + 0.7013)/3 = 0.2338** ✓

### Design-Doc Convention (Sensitivity)

Under design-doc where Overfill→N and Misaligned→N:

| Class | Current | Design-Doc |
|-------|---------|------------|
| N     | 50      | 63         |
| U     | 43      | 30         |
| P     | 26      | 26         |
| L     | 31      | 31         |

Vision targets under design-doc:
- `{P}` = 26
- `{N,U}` = 30
- `{N,U,L}` = 94

Trivial (always V4): F1_V4 = 2×0.627×1/(0.627+1) = 0.771
**Macro-F1 = 0.771/3 = 0.257**

**Impact on α:** Under design-doc, trivial rises from 0.234 to 0.257, so α would rise from 0.88 to 0.91. The 13-cell convention difference causes a 3-point shift in α.

---

## Task 1: Macro-F1 Discrepancy (0.324 vs 0.305)

### Resolution: 0.305 is a single-fold value, not a mean

**Source file:** `results/vision/coupon_eval/kfold_results.json`

| Fold | Macro-F1 |
|------|----------|
| 1    | **0.3053** |
| 2    | 0.2902   |
| 3    | 0.3321   |
| 4    | 0.3693   |
| **Mean** | **0.3242** |
| Std  | 0.0300   |

- **0.324** = `mean_macro_f1` (arithmetic mean of per-fold F1s)
- **0.305** = Fold 1's F1 (lowest fold)

There is no "0.305 mean" anywhere. If a document cites 0.305 as a mean, it is in error — that value is fold 1's macro-F1.

**Canonical value: macro-F1 = 0.324 ± 0.030 (mean ± std)**

---

## Task 2: ECE Discrepancy (0.14 vs 0.136)

### Resolution: 0.136 is the canonical post-calibration ECE

**Source file:** `results/vision/coupon_eval/calibration_results.json`

```json
{
  "pre_calibration_ece": 0.4223,
  "post_calibration_ece": 0.1360,
  "optimal_temperature": 10.0
}
```

- **0.136** = Actual post-calibration ECE
- **0.14** = Rounded approximation (appears nowhere in artefacts)

**Canonical value: ECE = 0.136 (post temperature scaling, T=10.0)**

Note: Pre-calibration ECE was 0.422 — the model was heavily overconfident before scaling.

---

## Task 3: Leakage Contradiction (81.59% vs 0%)

### Resolution: These describe DIFFERENT partitions

**81.59% — OLD partition (source_frame grouping)**
Source: `results/diagnostics/sequence_leakage_report.md`

> "test patches sharing a stem with train: 3,599 / 4,411 = **81.59%**"

This was the PUBLIC CORPUS split using `source_frame` grouping. The leakage was sequence-based: consecutive video frames of the same weld appeared in both train and test. The perceptual deduplication guard (`phash` distance ≤ 5) did not catch it because video frames differ pixel-by-pixel.

**0% — NEW partition (source_sequence grouping)**
Source: `results/vision/partition_search.json`

```json
"leakage_intersections": {
  "source_sequence": {"train&val": 0, "train&test": 0, "val&test": 0},
  "source_frame": {"train&val": 0, "train&test": 0, "val&test": 0},
  "source_image": {"train&val": 0, "train&test": 0, "val&test": 0}
}
```

This is the FIXED partition using `source_sequence` grouping, which ensures no sequence spans splits.

**NOT A CONTRADICTION.** The 81.59% was the problem; 0% is the fix. Both measurements are correct for their respective partitions.

**Coupon-data note:** The coupon k-fold (`kfold_results.json`) operates on 150 real-specimen cells with coupon-grouped folds. There is no sequence leakage in coupon data because patches are single-frame crops from independent physical coupons.

---

## Task 4: α_V Recomputation

### Formula

```
α = 1 - skill
skill = (F1_macro - F1_trivial) / (1 - F1_trivial)
```

### Computation

| Quantity | Value | Provenance |
|----------|-------|------------|
| F1_macro | 0.32421095140880174 | `kfold_results.json:mean_macro_f1` |
| F1_trivial | 0.2338 | `kfold_results.json:trivial_baseline` |
| skill | 0.11799915349621735 | computed |
| **α_V** | **0.8820008465037826** | computed |

Verification:
```
skill = (0.32421095140880174 - 0.2338) / (1 - 0.2338)
      = 0.09041095140880174 / 0.7662
      = 0.11799915349621735

alpha = 1 - 0.11799915349621735 = 0.8820008465037826
```

**Matches `alpha_derivation.json` exactly.**

### Sensitivity Check

| F1_trivial | α_V | Interpretation |
|------------|-----|----------------|
| 0.2338 (current) | 0.882 | 88.2% mass on Θ |
| 0.257 (design-doc) | 0.910 | 91.0% mass on Θ |
| 0.200 (hypothetical lower) | 0.845 | 84.5% mass on Θ |
| 0.300 (hypothetical higher) | 0.965 | 96.5% mass on Θ |

A ±0.03 change in F1_trivial produces roughly ±3–4 pp change in α.

---

## Summary of Faults

| Issue | Severity | Resolution |
|-------|----------|------------|
| Trivial baseline hardcoded, not computed | Low | Verified correct; recommend dynamic computation |
| 0.305 vs 0.324 | Medium | 0.305 was fold 1, not a mean — any document citing 0.305 as mean is in error |
| 0.14 vs 0.136 | Low | 0.14 is a rounding; canonical is 0.136 |
| 81.59% vs 0% leakage | None | Different partitions — both correct |

---

## Canonical Values for Thesis

| Metric | Value | Source |
|--------|-------|--------|
| Macro-F1 (CV mean) | **0.324 ± 0.030** | kfold_results.json |
| Trivial baseline | **0.234** | Majority-class F1 |
| ECE (post-calibration) | **0.136** | calibration_results.json |
| Temperature | **10.0** | calibration_results.json |
| α_V (discount factor) | **0.882** | alpha_derivation.json |
| Interpretation | 88.2% mass on Θ (ignorance) | — |
