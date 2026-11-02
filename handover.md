# Handover Document - Weld Fusion Thesis

**Date:** 2026-10-30
**Branch:** vision/sequence-leakage-audit

## Overview

This thesis implements multi-modal weld defect assessment fusing vision (CNN), ultrasonic (EPOCH 600), and geometric (depth sensor) evidence under Dempster-Shafer theory.

## Frame of Discernment

**4-CLASS EXPERIMENTAL FRAME** (C removed - no crack specimens fabricated):
```
Theta = {N, U, P, L}
  N  no defect / acceptable
  U  geometric non-conformance to ISO 5817
  P  porosity
  L  lack of fusion
```

**v2 Label Distribution:**
- N: 63 cells
- U: 30 cells
- P: 26 cells
- L: 31 cells
- **Total: 150 cells** (15 coupons × 10 cells)

## Key Results (Chapter 4)

### Discounting Factors (α)
- **α_V = 0.798** (vision, median from 5-seed variance study)
- **α_G = 1.000** (geometry, clamped - all 39 measured cells predict {U})
- **α_U = 0.241** (ultrasonic, from F1 = 0.821)

### Fusion Performance
- **F1 = 0.482** for all conditions including U (α_U dominates)
- Murphy's rule shows slightly higher conflict (mean K = 0.081 vs Dempster's 0.034)
- Design effect = 5.78, n_eff = 26.0 (vs assumed 5.5/27)

### Key Findings
1. UT mapping correlation r = 0.9792 (close to documented r = 0.977)
2. Geometry fully discounted due to misalignment exceeding limits on all measured cells
3. 39/150 cells have geometry measurements (sensor positioning dependent)
4. 53/150 cells have gated UT indications (located reflectors)

## Repository Structure

```
weld-fusion-thesis/
├── configs/
│   ├── frame.yaml              # Frame of discernment (4-class)
│   ├── frame_5class.yaml       # 5-class frame (analytical ceiling)
│   ├── geometry.yaml           # Geometry processing config
│   └── class_map.yaml          # Source-to-head mapping
│
├── data/
│   ├── labels_v2_2026-10-26.csv    # Current labels (v2)
│   ├── processed/
│   │   └── coupon_cell_patches/    # Vision patches (150 cells)
│   └── raw/S1/                     # Raw depth frames
│
├── results/
│   ├── geometry/
│   │   ├── summary.csv             # Per-cell geometry results
│   │   └── bpa_geometry.csv        # Geometry BPAs
│   ├── ut/
│   │   ├── features.csv            # UT features (150 readings)
│   │   ├── metrics_cv.json         # CV metrics
│   │   └── bpa_ut.csv              # UT BPAs
│   ├── vision/
│   │   ├── seed_variance_results.json  # 5-seed variance analysis
│   │   └── logits_*.parquet        # Model outputs
│   └── thesis/ch4_evidence/        # Chapter 4 outputs
│       ├── scripts/                # Generation scripts
│       ├── t4_*.tex                # LaTeX tables (T4.1-T4.8)
│       ├── f4_*.pdf                # Figures (F4.1-F4.7)
│       └── s4_*.json               # Numbers for text
│
├── src/
│   ├── bpa/                        # BPA construction
│   ├── fusion/                     # D-S combination rules
│   ├── geometry/                   # Geometry pipeline
│   ├── ut/                         # Ultrasonic processing
│   ├── vision/                     # CNN training/inference
│   └── utils/                      # Frame loading, etc.
│
└── CLAUDE.md                       # Detailed project instructions
```

## Chapter 4 Evidence Files

### Tables (LaTeX)
- `t4_1_data.tex` - Fold distribution
- `t4_2_ut_cv.tex` - UT CV results
- `t4_3_ut_calibration.tex` - UT calibration
- `t4_4_vision_cv.tex` - Vision CV results
- `t4_5_alpha.tex` - Discounting factors
- `t4_6_ablation.tex` - Fusion ablation study
- `t4_7_mcnemar.tex` - McNemar matrix (statistical tests)
- `t4_8_declared.tex` - Declared comparisons

### Figures
- `f4_1_cell_examples.pdf` - One cell per class, 3 modalities
- `f4_2_ut_readings.pdf` - All 150 UT readings scatter plot
- `f4_3_ut_mapping.pdf` - UT mapping check
- `f4_4_confusion.pdf` - Confusion matrices
- `f4_5_recall_heatmap.pdf` - Per-class recall
- `f4_6_ablation_f1.pdf` - Ablation F1 comparison
- `f4_7_conflict.pdf` - Conflict coefficient distribution

### JSON Numbers
- `s4_1_numbers.json` - Data statistics + F4.1 cell selection
- `s4_2_ut.json` - UT verification (r = 0.9792)
- `s4_3_numbers.json` - Single-modality results
- `s4_4_equality.json` - Equality tests
- `s4_5_neff.json` - Effective sample size

## Selected Cells for F4.1

Selection rule: First geometry-measured cell per class in coupon order.

| Class | Cell ID | Vision | Geometry | UT |
|-------|---------|--------|----------|-----|
| N | C03-02 | {N,U,L} | Exc=1.0, misalignment | 12.6%, not located |
| U | C05-02 | {N,U} | Exc=1.0, misalignment | 4.8%, not located |
| P | C01-02 | {P} | Exc=1.0, misalignment | 22.1%, located, SP=11.7mm |
| L | C07-02 | {N,U,L} | Exc=0.999, misalignment | 81.9%, located, SP=21.9mm |

## Outstanding Work

1. **Write thesis Chapter 4 text** - All evidence is now generated
2. **Consider additional analysis** if needed:
   - Per-coupon breakdown
   - Sensitivity analysis on thresholds
   - Alternative combination rules

## Scripts for Regeneration

```bash
# Generate §4.1 outputs (fold table, numbers)
python3 results/thesis/ch4_evidence/scripts/generate_s4_1.py

# Generate F4.1 cell examples figure
python3 results/thesis/ch4_evidence/scripts/generate_f4_1.py

# Generate §4.2-§4.5 outputs (tables, figures, numbers)
python3 results/thesis/ch4_evidence/scripts/generate_s4_2_to_s4_5.py
```

## Environment

- WSL on Windows
- Python: `/usr/bin/python3` (system) or `~/weldenv/bin/python` (venv)
- PyTorch with CUDA 12.4 for training

## Important Notes

1. **Do not retrain models** - All results are from stored outputs
2. **Use v2 labels** - `data/labels_v2_2026-10-26.csv`
3. **Vision targets focal elements, not classes** - V1={P}, V3={N,U}, V4={N,U,L}
4. **Vision never predicts U alone** - Can see deviation but not measure severity
5. **Geometry always predicts {U}** - All measured cells exceed misalignment limits

## Contact

See `CLAUDE.md` for full project documentation and invariants.
