# Chapter 4 Evidence Summary

Generated: 2026-10-31

## Files Generated

| File | Description |
|------|-------------|
| **Tables** | |
| t4_2_verify.tex/csv | T4.2 Verification checks (Vision/Geometry/UT) |
| t4_3_modality.tex/csv | T4.3 Modality results table |
| t4_4_ablation.tex/csv | T4.4 Ablation table (Dempster + Murphy CIs) |
| t4_5_mcnemar.tex/csv | T4.5 McNemar test results |
| **Figures** | |
| f4_2_coupon_strip.pdf/png | F4.2 Coupon strip (C14) showing all modalities |
| f4_3_confusion.pdf/png | F4.3 Confusion matrices by modality |
| f4_4_recall_heatmap.pdf/png | F4.4 Recall heatmap (NOT AVAILABLE - needs per-cell predictions) |
| f4_5_ablation_f1.pdf/png | F4.5 Ablation F1 with 95% CI |
| **JSON** | |
| s4_2_geometry.json | Geometry: governing measurand counts, misalignment stats, per-coupon worst |
| s4_3_equality.json | BPA equality test results (V+G vs V, etc.) |
| s4_4_neff.json | Effective sample size: rho=0.531, deff=5.78, n_eff=26 |
| t4_3_per_element_f1.json | Per-element F1 by modality |
| MANIFEST.json | File listing and sources |

## Headline Numbers

| Metric | Value | Source |
|--------|-------|--------|
| **Alpha values** | | |
| alpha_V | **0.798** | calibration_v2.json (median from 5-seed variance) |
| alpha_G | 1.00 | Deterministic (exceedance-based) |
| alpha_U | 0.241 | alpha_ut.json |
| **Modality performance** | | |
| Vision F1 | 0.324 +/- 0.030 | 4-fold CV on coupon patches |
| UT F1 | 0.821 | metrics_cv.json |
| Geometry cells measured | 39 | summary.csv |
| **Fusion (Dempster)** | | |
| V | 0.097 [0.029, 0.167] | |
| G | 0.086 [0.022, 0.143] | |
| U | 0.482 [0.355, 0.583] | |
| V+G | 0.097 [0.020, 0.168] | |
| V+U | 0.482 [0.348, 0.586] | |
| G+U | 0.482 [0.346, 0.587] | |
| V+G+U | 0.482 [0.344, 0.591] | |
| **Design effect** | | |
| rho (ICC) | 0.531 | One-way ANOVA by coupon |
| deff | 5.78 | 1 + 9*rho |
| n_eff | 26 | 150/deff |

## Contradictions to Expected Values

| Expected | Actual | Status |
|----------|--------|--------|
| alpha_V = 0.909 | alpha_V = 0.798 | **DIFFERS** - corrected to median from seed variance study |
| alpha_G = 1.00 | alpha_G = 1.00 | MATCHES |
| alpha_U = 0.241 | alpha_U = 0.241 | MATCHES |
| UT F1 = 0.821 | UT F1 = 0.821 | MATCHES |
| 39 geometry cells | 39 cells | MATCHES |
| F1 = 0.482 (all U conditions) | F1 = 0.482 | MATCHES |

## Gap Closures (2026-11-01)

| Gap | Answer |
|-----|--------|
| α_V provenance | Fusion used 0.798 from calibration_v2.json (matches t4_3_modality) |
| F4.4 recall | Available from fusion_results.json (150 cells with decision/true_class) |
| NN audit | 32.8% below 0.05 (public corpus, ResNet50); defects only: 29.4% |
| UT mapping | Pearson r = 0.732 (amplitude vs P/L), mean 54% (P/L) vs 10.5% (N/U) |

## NOT AVAILABLE Items

| Item | Reason |
|------|--------|
| UT depth vs SA*cos60 check | Depth verification not stored |
| Geometry plausibility check 0-2mm | Not implemented in pipeline |

## F4.2 Coupon Selection

- **Selected coupon**: C14
- **Rule**: Most distinct labels (3), then most geometry-measured cells (4)
- **Labels present**: L, U, N
- **Zones**: A=LOF (3 cells), B=Undercut (4 cells), C=Sound (3 cells)

## Geometry Statistics (s4_2_geometry.json)

All 39 measured cells have governing_measurand = "misalignment"

| Statistic | Value (mm) |
|-----------|------------|
| min | 0.77 |
| median | 3.75 |
| mean | 4.14 |
| 95th %ile | 7.84 |
| max | 10.08 |

Per-coupon worst cells:
- C02-04: 9.68 mm
- C15-01: 10.08 mm
- C06-03: 7.64 mm
- C09-03: 6.58 mm
- C10-02: 5.71 mm

U-call true labels: N=11, U=10, L=10, P=8 (all 39 cells output m({U})=1)

## McNemar Tests (T4.5)

| Comparison | b | c | chi2 | p_raw | p_Holm | Sig |
|------------|---|---|------|-------|--------|-----|
| V->GV | 0 | 0 | - | - | - | identical |
| V->UV | 77 | 15 | 40.45 | 2.0e-10 | 6.1e-10 | Yes |
| G->GU | 63 | 5 | 47.78 | 4.8e-12 | 1.9e-11 | Yes |
| GV->GUV | 77 | 15 | 40.45 | 2.0e-10 | 4.0e-10 | Yes |

Note: V->GV is identical (b=c=0) because G outputs vacuous for unmeasured cells,
so V+G = V for 111/150 cells, and the 39 measured cells all get m({U})=1.

## Scripts

- `scripts/generate_chapter4_evidence.py` - Tables and JSON
- `scripts/generate_chapter4_figures.py` - Figures

---

*All values computed from stored results without retraining or refitting.*
