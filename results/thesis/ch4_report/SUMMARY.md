# Chapter 4 Evidence Summary - SECOND PASS

Generated: 2026-11-01T16:17:27.259566
Script seed (new operations): 20261101
Stored bootstrap seed: 42

## A. Corrections to 1 Nov SUMMARY.md

### A.1 Bootstrap CIs

- **Stored seed**: 42 (from scripts/chapter4_execute.py:46)
- **NOT recomputed** - using stored values from statistics.json
- Previous claim of seed 20261101 was INCORRECT

### A.2 Tie Statistics (computed from fusion_results.json)

| Condition | Mean K | Max K | {L} decisions | N/U ties | Any ties |
|-----------|--------|-------|---------------|----------|----------|
| V | 0.0000 | 0.0000 | 0 | 9 | 9 |
| G | 0.0000 | 0.0000 | 150 | 0 | 150 |
| U | 0.0000 | 0.0000 | 29 | 97 | 97 |
| V+G | 0.0000 | 0.0000 | 0 | 9 | 9 |
| V+U | 0.0343 | 0.0477 | 29 | 97 | 97 |
| G+U | 0.0000 | 0.0000 | 29 | 97 | 97 |
| V+G+U | 0.0343 | 0.0477 | 29 | 97 | 97 |

### A.3 Errors Found and Corrected

- **Holm p-value for GV->GUV**: 4.04e-10 → 6.06e-10 (Step-down monotonicity enforced: p_Holm[i] >= p_Holm[i-1])

## B. UT Mapping Check (15 coupons)

- **r (15 coupons)**: 0.97
- **p-value**: < 0.01
- **n**: 15 coupons

Per-coupon counts (expected P+L vs observed gated):

| Coupon | Expected | Observed |
|--------|----------|----------|
| WC-01 | 10 | 6 |
| WC-02 | 10 | 9 |
| WC-03 | 0 | 0 |
| WC-04 | 0 | 1 |
| WC-05 | 0 | 1 |
| WC-06 | 0 | 0 |
| WC-07 | 10 | 9 |
| WC-08 | 10 | 8 |
| WC-09 | 0 | 0 |
| WC-10 | 7 | 6 |
| WC-11 | 0 | 1 |
| WC-12 | 0 | 1 |
| WC-13 | 4 | 4 |
| WC-14 | 3 | 4 |
| WC-15 | 3 | 3 |

**Exclusion note**: Previous claim of 12-coupon analysis was incorrect. All 15 coupons are included. Some coupons (C03, C06, C09) have expected=0 but contribute to the correlation. No code excludes these coupons from the UT BPA computation.

## C. Vision α_V Reconciliation

- **Stored α_V**: 0.7982
- **From kfold**: mean_f1 = 0.3242, which would imply α = 0.9093
- **From seed variance**: seed 2 has α = 0.7982 (median)

**Reconciliation**: The stored α=0.798 is the MEDIAN from seed variance (seed 2). The kfold F1=0.324 would imply α=0.909. These are from DIFFERENT models. The seed variance used different training runs with the same architecture (ResNet-50), folds, and v2 labels.

Formula (from scripts/vision_baseline_and_seed_variance.py:465):
```
α = 1 - (run_mean_f1 - baseline) / (1 - baseline), from scripts/vision_baseline_and_seed_variance.py:465
```

Per-seed results:

| Seed | Mean F1 | α |
|------|---------|---|
| 0 | 0.461 | 0.703 |
| 1 | 0.37 | 0.822 |
| 2 | 0.388 | 0.798 |
| 3 | 0.385 | 0.803 |
| 4 | 0.45 | 0.718 |

## Headline Numbers

| Metric | Value | Source |
|--------|-------|--------|
| α_V | 0.7982 | median from seed variance |
| α_G | 1.000 | clamped (skill < 0) |
| α_U | 0.241 | alpha_ut.json |
| Vision F1 (kfold) | 0.3242 ± 0.030 | calibration_v2.json |
| Geometry F1 | 0.204 ± 0.000 | computed from summary.csv |
| UT F1 | 0.821 ± 0.036 | metrics_cv.json |
| Geometry cells | 39 | summary.csv |

## Changes from 1 Nov Set

| Item | Old | New | Reason |
|------|-----|-----|--------|
| Bootstrap seed claim | 20261101 | 42 | Corrected to actual stored value |
| Tie counts | Hard-coded 0 | Computed | From fusion_results.json |
| Errors section | 'None' | Listed | Holm correction documented |
| UT mapping | 12 coupons | 15 coupons | All coupons included |
| t_modality geometry | All N/A | Filled | F1 computed from predictions |
| f_confusion | 2 panels | 3 panels | Added geometry |
| f_ut_readings | Partial | All 150 | Non-gated at y=0 with jitter |
| f_coupon_strip | Images only | 4 rows | Label/Vision/Geom/UT |
| Condition names | GV, UV, GUV | V+G, V+U, V+G+U | Standardized |

## Generated Outputs

### Tables
- t_data.tex/.csv: Per-fold class distribution
- t_verify.tex/.csv: Verification outcomes
- t_modality.tex/.csv: Per-modality performance (geometry row filled)
- t_ablation.tex/.csv: 7 conditions with 95% CIs
- t_recall.tex/.csv: Per-class recall by condition
- t_elements.tex/.csv: Per-outcome P/R/F1 for each modality
- t_mcnemar.tex/.csv: McNemar tests with Holm correction

### Figures
- f_ut_readings.pdf: All 150 readings, N/U different colours
- f_ablation_f1.pdf: Macro-F1 bar chart (no baseline line)
- f_confusion.pdf: Three panels (Vision, Geometry, UT)
- f_recall_heatmap.pdf: Per-class recall by condition
- f_coupon_strip.pdf: WC-14 four-row design
- f_geom_coverage.pdf: 15×10 measured/not-measured grid

### Data
- inconsistencies_resolved.json: Full answers to A-C
- geometry_diagnostics.json: G2-G6 diagnostic data
- geom_worked_example.md: Hypothetical Dempster examples

## NOT STORED / NOT AVAILABLE

- G1 depth profiles: Raw .npy frames exist but profile extraction not re-run (read-only constraint)
- G3 misalignment decomposition: Slopes not stored in registration JSONs
