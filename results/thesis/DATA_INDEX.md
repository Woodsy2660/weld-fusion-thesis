# Thesis Data Index

All supporting data files for Chapter 4 results discussion.

## Directory Structure

```
results/thesis/
├── tables/           # LaTeX table fragments
├── figures/          # PDF figures
├── data/             # Source data files
├── LIMITATIONS.md    # Known limitations
├── MANIFEST.json     # Full provenance
├── PHASE_A_DIAGNOSTIC.md
├── VACUITY_ANALYSIS.md
└── DATA_INDEX.md     # This file
```

## Data Files (`results/thesis/data/`)

### Labels
| File | Description |
|------|-------------|
| `labels_v2_2026-10-26.csv` | 150 cells with v2 convention (Overfill/Misaligned → N) |

### Vision Modality
| File | Description |
|------|-------------|
| `vision_bpas_v2.json` | Vision BPAs for all 150 cells (α_V = 0.909) |
| `calibration_v2.json` | Temperature scaling results (T = 100.0, hit boundary) |
| `kfold_results.json` | Per-fold predictions, y_true, y_pred, y_probs |
| `vision_canonical_results.json` | Consolidated vision metrics and audit status |

### Geometry Modality
| File | Description |
|------|-------------|
| `bpa_geometry.csv` | Canonical geometry BPAs (α_G = 1.0, 4-class naming) |
| `alpha_derivation.json` | Alpha computation: F1 = 0.204, clamped to 1.0 |
| `geometry_summary.csv` | Raw geometry measurements and exceedance probs |

### Ultrasonic Modality
| File | Description |
|------|-------------|
| `bpa_ut.csv` | UT BPAs for all 150 cells (α_U = 0.241) |
| `metrics_ut.json` | UT classification metrics (F1 = 0.821) |
| `ut_features.csv` | UT features including amplitude per cell |

### Fusion Results
| File | Description |
|------|-------------|
| `fusion_results.json` | All 7 subsets × 2 methods, per-cell results |
| `statistics.json` | Bootstrap CIs, McNemar tests, design effect |

## Key Numbers for Chapter 4

### Modality Performance (Table 4.3)
| Modality | F1 | Trivial | α | ECE |
|----------|-----|---------|---|-----|
| Vision | 0.324 ± 0.030 | 0.257 | 0.909 | 0.082 |
| Geometry | 0.204 | 0.426 | 1.000 | --- |
| Ultrasonic | 0.821 ± 0.036 | 0.255 | 0.241 | 0.108 |

### Fusion Results (Table 4.4)
| Subset | F1 | 95% CI |
|--------|-----|--------|
| V | 0.097 | [0.03, 0.17] |
| G | 0.086 | [0.02, 0.14] |
| U | 0.482 | [0.36, 0.58] |
| VG | 0.097 | [0.02, 0.17] |
| VU | 0.482 | [0.35, 0.59] |
| GU | 0.482 | [0.35, 0.59] |
| VGU | 0.482 | [0.34, 0.59] |

### Statistical Tests (Table 4.5)
| Comparison | χ² | p_Holm | Significant |
|------------|-----|--------|-------------|
| G → GU | 47.78 | <0.001 | Yes |
| V → VU | 40.45 | <0.001 | Yes |
| VG → VGU | 40.45 | <0.001 | Yes |
| V → VG | 0.00 | 1.000 | No |

### Design Effect
- ICC: 0.53
- Design effect: 5.78
- n_eff: 26

## Limitations Summary

1. **Geometry α clamped** — All 39 decided cells predict {U}, F1 < trivial
2. **26% geometry coverage** — 111/150 cells vacuous
3. **Vision T at boundary** — Calibration unconverged at T = 100
4. **No human baseline** — §4.5 cut
5. **Wide CIs** — n_eff ≈ 26 limits precision

## Figures (`results/thesis/figures/`)

| File | Description |
|------|-------------|
| `f4_1_ut_amplitude.pdf` | UT amplitude distribution by defect class |
| `f4_2_confusion.pdf` | Confusion matrix heatmap |
| `f4_3_geometry_coverage.pdf` | Geometry modality coverage (26% decided) |
| `f4_4_ablation.pdf` | Ablation study: F1 by modality subset |
| `f4_4_cellgrid_schematic.pdf` | Cell grid schematic with zones A-C and excluded regions |
| `f4_4_cellgrid_image.pdf` | Cell grid overlaid on actual coupon image |
| `f4_5_conflict.pdf` | Dempster-Shafer conflict distribution |
| `f4_6_ablation_schematic.pdf` | Ablation architecture: 3 modality BPAs → 7 fusion subsets |
| `f4_7_geometry_misalignment.pdf` | Misalignment measurements for 39 decided cells vs ISO limit |
| `f4_8_geometry_coverage_heatmap.pdf` | Coverage heatmap: decided vs vacuous by coupon/cell |
| `f4_9_geometry_class_distribution.pdf` | Ground truth distribution: decided vs vacuous cells |
| `f4_10_depth_images.png` | Raw depth images: decided vs vacuous coupons |
| `f4_11_cross_sections.png` | Weld cross-section profiles checking underfill |
| `f4_12_side_view.png` | Side view along weld centerline vs plates |
| `f4_13_vacuous_vs_decided.png` | 2D depth comparison: C04 (vacuous) vs C15 (decided) |
| `f4_14_bead_trench_artifact.pdf` | Trench artifact: weld reads as dip not crown |
| `f4_15_false_misalignment.pdf` | How trench artifact causes false misalignment |

## File Formats

- `.json` — JSON, human-readable
- `.csv` — CSV with headers
- `.tex` — LaTeX fragments (booktabs)
- `.pdf` — Vector PDF figures
