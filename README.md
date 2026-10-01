# Multi-Modal Sensor Fusion for AI-Assisted Defect Classification in Structural Steel Butt Welds

**Author:** Benjamin Wood (SID: 530511030)  
**Supervisor:** Mehala Balamuri  
**Institution:** University of Sydney, School of AMME  
**Thesis:** AMME4111/4112 Honours Project 2025-2026

## Overview

This project investigates whether evidence-theoretic fusion of three sensing modalities — RGB surface vision, 3D geometric profiling, and ultrasonic A-scan analysis — yields higher defect detection rates than any single modality alone for post-fabrication inspection of structural steel butt welds assessed against AS/NZS 1554.1.

## Repository Structure

```
├── src/                    # Source code
│   ├── vision/             # ResNet-50 surface defect classifier
│   ├── geometry/           # Depth profile extraction + ISO 5817 rules
│   ├── ut/                 # A-scan feature extraction + Random Forest
│   ├── fusion/             # Dempster-Shafer evidence theory
│   └── utils/              # Shared utilities
├── configs/                # Training and pipeline configs
├── blender/                # Blender weld models and export scripts
├── notebooks/              # Exploration and analysis notebooks
├── results/                # Outputs, plots, tables
├── tests/                  # Unit and integration tests
├── thesis/                 # LaTeX and document files
└── data/                   # (gitignored) datasets and processed data
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data

Datasets are not included in this repository due to size. See `data/README.md` for download instructions.

## Modality Pipelines

| Modality | Model | Training Data | Code |
|----------|-------|---------------|------|
| Vision (RGB) | ResNet-50 | LoHi-WELD + Wijaya + FINAL WELDING | `src/vision/` |
| Geometry (depth) | Rules-based | Blender synthetic + abonyilab | `src/geometry/` |
| Ultrasonic (A-scan) | Random Forest / XGBoost | Synthetic features | `src/ut/` |
| Fusion | Dempster-Shafer | Combined outputs | `src/fusion/` |

## License

This repository is part of an academic thesis project. All rights reserved.
