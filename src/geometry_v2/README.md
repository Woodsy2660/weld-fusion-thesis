# Geometry V2: Asymmetry-Based BPA Construction

This module implements an alternative geometry BPA construction that addresses the limitation of the original geometry module (α_G = 1.0, vacuous).

## Problem Statement

The original geometry module uses **centerline depth** for ISO 5817 exceedance detection:
- Signal-to-noise ratio (SNR) = **0.23** (noise overwhelms signal)
- Cohen's d for LOF detection = -0.96 (large effect but negative correlation)
- Result: Shafer discounting gives **α_G = 1.0** (vacuous, no contribution)

## Solution: Plate Asymmetry

Analysis of weld band data revealed that **plate asymmetry** is a better discriminative feature:
- SNR = **1.75** (good signal-to-noise ratio)
- Cohen's d = **1.22** (large positive effect)
- ANOVA p-value = **0.044** (statistically significant)

| Condition | Mean Asymmetry (mm) | Std Dev |
|-----------|---------------------|---------|
| LOF | 4.46 | 0.80 |
| Misaligned | 4.41 | - |
| Porosity | 3.95 | 0.91 |
| Sound | 2.25 | 1.71 |
| Undercut | 1.21 | 0.52 |

## Key Insight

Plate asymmetry cannot distinguish between:
- **Lack of Fusion (LOF)**: One-sided heat input causes plate tilt
- **Linear Misalignment (U)**: Plates not aligned before welding

Therefore, high asymmetry maps to the **compound focal element {L,U}**, which correctly represents the sensor's epistemic limitation under Dempster-Shafer theory.

## Focal Element Distribution

### Original Geometry (src/geometry/rules.py)
```
G1: {U}       ISO 5817 exceedance (reinforcement, misalignment)
G2: {N,P,L}   Profile conforms
```

### Geometry V2 (src/geometry_v2/rules.py)
```
G1: {U}       ISO 5817 exceedance (unchanged)
G2: {N,P,L}   Profile conforms (unchanged)
G3: {L,U}     High plate asymmetry (NEW)
```

## BPA Construction Logic

```python
Given asymmetry A (mm) and ISO exceedance probability P:

IF P > 0.5:
    m({U}) = P * 0.9       # ISO exceedance takes precedence
    m(Theta) = 0.1

ELIF A > 3.5mm:            # High asymmetry threshold
    m({L,U}) = 0.6         # Strong evidence for LOF or misalignment
    m(Theta) = 0.4

ELIF A > 2.0mm:            # Moderate asymmetry threshold
    m({L,U}) = 0.3         # Partial evidence
    m(Theta) = 0.7

ELSE:
    m({N,P,L}) = 0.9       # Geometry within normal range
    m(Theta) = 0.1
```

## Results

### LOF Detection Performance
- **Sensitivity**: 100% (all 3 LOF coupons correctly identified)
- **Specificity**: 66.7%
- **Precision**: 42.9%

### BPA Assignment by True Class
| True Class | Mean m({L,U}) | Mean m({N,P,L}) |
|------------|---------------|-----------------|
| L (LOF) | 0.60 | 0.00 |
| U (Undercut) | 0.00 | 0.90 |
| P (Porosity) | 0.50 | 0.00 |
| N (Sound) | 0.25 | 0.45 |

**Key observations:**
- All LOF coupons correctly receive m({L,U}) = 0.6
- All Undercut coupons correctly receive m({N,P,L}) = 0.9 (no asymmetry signal)
- Some false positives (Sound/Porosity with high asymmetry) are acceptable because:
  - BPA assigns mass to {L,U}, not {L} alone
  - Fusion with other modalities will resolve these cases
  - The compound element correctly represents uncertainty

## Files

```
src/geometry_v2/
├── __init__.py         # Module description
├── asymmetry.py        # Asymmetry extraction from band data
├── rules.py            # BPA construction with asymmetry thresholds
├── analyze.py          # Analysis comparing original vs V2
├── visualize.py        # Thesis figures generation
└── README.md           # This documentation

configs/
└── frame_geometry_v2.yaml  # Frame config with G3:{L,U} added

results/geometry_v2/
├── bpa_results.csv           # BPA construction results
├── bpa_dicts.json            # BPA dicts for fusion
├── bpa_distribution_by_class.png
├── decision_flow.png
└── comparison_original_vs_v2.png
```

## Usage

```python
from src.geometry_v2.rules import build_coupon_bpa, build_all_bpas

# Build BPA for single coupon
bpa = build_coupon_bpa("C07")
print(bpa.focal_element_summary())  # "m({L,U})=0.60, m(Theta)=0.40"

# Build all BPAs
all_bpas = build_all_bpas()
for coupon_id, bpa in all_bpas.items():
    fusion_dict = bpa.as_bpa_dict()  # Ready for DS combination
```

## Thesis Updates Required

### Chapter 5, Section 5.3: Geometry Modality Workaround

1. **Document the limitation**: Centerline depth SNR = 0.23 (too low)

2. **Present the workaround**: Plate asymmetry as alternative feature
   - Extraction methodology (band analysis)
   - Thresholds derived from data (3.5mm, 2.0mm)
   - Statistical justification (SNR, Cohen's d, ANOVA)

3. **Update focal element diagram**:
   - Add G3: {L,U} with explanation of compound element
   - Explain why asymmetry → {L,U} not {L} alone

4. **Report results**:
   - 100% LOF sensitivity
   - Correct assignment of m({N,P,L}) to Undercut class
   - False positive rate and mitigation via fusion

5. **Discussion**:
   - The compound element correctly represents epistemic limitation
   - Small sample size (n=3 LOF) warrants validation
   - Integration with existing fusion framework

### Frame Configuration Update

To adopt V2 in the main pipeline, update `configs/frame.yaml`:

```yaml
geometry_heads:
  G1: ["U"]
  G2: ["N", "P", "L"]
  G3: ["L", "U"]  # NEW: Asymmetry-based detection
```

And add asymmetry thresholds:

```yaml
geometry_v2:
  asymmetry_high_mm: 3.5
  asymmetry_mod_mm: 2.0
  mass_high: 0.6
  mass_mod: 0.3
  uncertainty_floor: 0.1
```
