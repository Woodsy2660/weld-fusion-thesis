# Multi-modal fusion for steel butt-weld defect classification

Honours thesis (AMME4111/4112), University of Sydney. Author: Benjamin Wood.
Supervisor: Mehala Balamuri.

Evidence-theoretic fusion of RGB vision, 3D geometry, and ultrasonic A-scans
for post-fabrication inspection of structural steel butt welds against
AS/NZS 1554.1, with geometric limits from ISO 5817:2023 quality level B.

This repository is **scaffolding plus deterministic maths**. It does not contain
datasets, trained weights, experimental results, or a runnable end-to-end
pipeline.

## Frame of discernment

Θ = {N, U, P, L, C} — defined only in `configs/frame.yaml`.

Each modality assigns mass **only** over subsets it can physically resolve.
See `CLAUDE.md` for the focal-element table and the three-way conjunction
that isolates a sound weld.

## Status

| Layer | State |
|-------|--------|
| Frame, types, config hashing, manifests | Implemented, tested |
| Registration (cells, UT projection) | Implemented, tested |
| Geometry uncertainty + ISO 5817 exceedance BPA | Implemented, tested |
| Dempster, Murphy, pignistic, Shafer discount, 7-subset ablation | Implemented, tested |
| Calibration (temperature, isotonic, Platt, reliability α) | Implemented |
| Algebraic accuracy ceiling, GroupKFold splits, metrics, stats | Implemented, tested |
| Vision ResNet-50 train/eval/infer | Ported to 5-class frame; **unrunnable without data** |
| UT feature extractor | Implemented |
| UT RF/XGBoost training | Stub |
| USM 100 IO | Stub (export format unconfirmed) |
| RANSAC base-plate | Stub (outer-median fallback works) |
| Datum detection, `run_coupon` | Stub |
| Blender weld generator | Stub — original scripts produced Figure 7 of the preliminary report and must be recovered or rebuilt |
| Synthetic UT features | DEV-ONLY generator; must never be reported |

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Unix
pip install -e ".[dev]"
pytest
```

## Non-goals (this branch)

Do not train models, download datasets, or invent results. Random cell-level
splits are a bug. The 0.05 uncertainty floor is gone; ignorance is Shafer
discounting from validation-fold balanced accuracy.

## License

Academic thesis project. All rights reserved.
