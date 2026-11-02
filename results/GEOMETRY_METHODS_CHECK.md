# Geometry Module Methodology Verification

## Run Identity

**Entry script**: `scripts/run_geometry_full.py`
**Config file**: `configs/geometry.yaml`
**Output files**: `results/geometry/summary.json`, `results/geometry/bpa_geometry.csv`
**Coupons processed**: C01–C15 (15 coupons, 150 cells)
**Registration method**: Seeded registration via `src/geometry/registration_seeded.py`

---

## Verification Table

| # | Item | Status | Actual Value | Evidence | Quote |
|---|------|--------|--------------|----------|-------|
| 0a | Run identity: entry script | **MATCH** | `scripts/run_geometry_full.py` | Commit ceaf2d6 | Uses `register_capture_seeded()` |
| 0b | Run identity: config | **MATCH** | `configs/geometry.yaml` | `profile.py:24` | `CONFIG_PATH = REPO_ROOT / "configs" / "geometry.yaml"` |
| 0c | Run identity: output files | **MATCH** | summary.json, bpa_geometry.csv | `results/geometry/` | 150 cells, 111 vacuous |
| 0d | Label convention: U meaning | **MATCH** | U = undercut OR reinforcement OR misalignment | `rules.py:6-10` | "configs/frame.yaml defines U as the DISJUNCTION undercut OR excess reinforcement OR linear misalignment" |
| 1.1 | Frames per face | **64** | 64 frames averaged per face | `results/geometry/C01_top_r0_registration.json:6` | `"n_depth_frames": 64` |
| 1.2 | 5.68× noise reduction | **DOCUMENTED** | Split-half SE vs naive σ/√N | `weld-docs/docs/08-current-status.md:109` | "averaging reduction over 64 frames: 5.68× / 5.64× / 8×" |
| 1.3 | Averaging method | **SPLIT-HALF** | Not naive √N | `src/geometry/profile.py:77-156` | `compute_depth_stack()`: "SE = ½ · std(difference)" |
| 1.4 | Invalid pixel handling | **MATCH** | raw==0 → NaN, nanmean | `profile.py:72-74, 104` | `depth_mm[raw == 0] = np.nan`; `np.nanmean(stack, axis=0)` |
| 2.1 | Plane-fit ROI | **MATCH** | y_inner=15mm, y_outer=40mm | `configs/geometry.yaml:73-74` | `y_inner_mm: 15`; `y_outer_mm: 40` |
| 2.2 | Datum mode | **LINEAR** | Per-side line fit, not median | `configs/geometry.yaml:93` | `datum_mode: linear` |
| 2.3 | RANSAC iterations | **100** | 100 iterations | `profile.py:177` | `n_iterations: int = 100` |
| 2.4 | RANSAC inlier threshold | **0.5 mm** | 0.5 mm | `profile.py:178` | `inlier_threshold_mm: float = 0.5` |
| 3.1 | Weld axis definition | **SEEDED** | From `configs/weld_axis_seeds.csv` | `scripts/run_geometry_full.py` | Uses `measure.seed_centre_line()` |
| 3.2 | Profiles per cell | **10** | 10 transverse profiles | `configs/geometry.yaml:50` | `profiles_per_cell: 10` |
| 3.3 | Aggregation quantile | **95th** | 0.95 for undercut | `configs/geometry.yaml:57` | `aggregation_quantile: 0.95` |
| 4.1 | Undercut definition | **MATCH** | Max of L/R toes, 95th %ile across profiles | `profile.py:647-660` | Governing undercut = max(left_q, right_q) |
| 4.2 | Reinforcement definition | **SIGNED** | Positive=excess (502), negative=underfill (511) | `profile.py:461-472` | "Return the SIGNED extremum: max for proud, min for concave" |
| 4.3 | Misalignment definition | **MATCH** | \|datum_left − datum_right\| at toes | `profile.py:443-444` | `misalignment = abs(datum_left - datum_right)` |
| 5.1 | Undercut limit (5011/5012) | **0.5 mm** | 0.05t, max 0.5mm; 0.5mm at t=10 | `configs/geometry.yaml:157-158` | `undercut_coef_t: 0.05`; `undercut_max_mm: 0.5` |
| 5.2 | Excess reinforcement limit (502) | **1+0.1b mm** | 1mm + 0.1×b, max 5mm | `configs/geometry.yaml:160-162` | `reinforcement_base_mm: 1.0`; `reinforcement_coef_b: 0.1` |
| 5.3 | Underfill limit (511) | **0.5 mm** | 0.05t, max 0.5mm; 0.5mm at t=10 | `configs/geometry.yaml:165-166` | `underfill_coef_t: 0.05`; `underfill_max_mm: 0.5` |
| 5.4 | Misalignment limit (507) | **1.0 mm** | 0.1t, max 3mm; 1.0mm at t=10 | `configs/geometry.yaml:168-169` | `misalignment_coef_t: 0.1`; `misalignment_max_mm: 3.0` |
| 6.1 | Exceedance formula | **MATCH** | P = Φ((d̂ − d_limit) / u_total) | `rules.py:224-230` | `z = (measured_mm - limit_mm) / u_total_mm; return float(norm.cdf(z))` |
| 6.2 | Active measurands | **3 of 4** | reinforcement_502, reinforcement_511, misalignment | `rules.py:315-332` | "Undercut is EXCLUDED from the U disjunction" |
| 6.3 | Undercut excluded reason | **u > limit** | u_undercut=0.586mm > 0.5mm limit | `rules.py:315-317` | "measured repeatability 0.586 mm exceeds 0.5 mm Level B limit; S/U ratio 0.85:1" |
| 6.4 | u_misalignment (used) | **0.275 mm** | Measured repeatability | `configs/geometry.yaml:140` | `misalignment_mm: 0.275` |
| 6.5 | u_undercut (excluded) | **0.586 mm** | Measured repeatability | `configs/geometry.yaml:138` | `undercut_mm: 0.586` |
| 6.6 | u_reinforcement | **0.493 mm** | Measured repeatability | `configs/geometry.yaml:139` | `reinforcement_mm: 0.493` |
| 6.7 | min_valid_fraction | **0.60** | 60% of ROI pixels must be valid | `configs/geometry.yaml:104` | `min_valid_fraction: 0.60` |
| 6.8 | Measured repeatability used | **YES** | `use_for_bpa: true` | `configs/geometry.yaml:136` | `use_for_bpa: true` |
| 7.1 | BPA columns | **MATCH** | m_U, m_NPLC (m_Theta implicit) | `results/geometry/summary.json` | `"m_U": ..., "m_NPLC": ...` |
| 7.2 | G2 focal element | **{N,P,L}** | 4-class frame (C removed) | `rules.py:18-19` | "G2 was {N,P,L,C} ... It is now {N,P,L}" |
| 7.3 | Alpha discount | **1.0 (clamped)** | F1 < trivial baseline | `results/geometry/alpha_derivation.json` | `"alpha_G": 1.0, "clamped": true` |
| 7.4 | F1 pooled | **0.204** | All decided cells predict {U} | `results/geometry/alpha_derivation.json` | `"f1_pooled": 0.2040816326530612` |
| 7.5 | Trivial baseline | **0.4265** | Majority-class macro-F1 | `results/geometry/alpha_derivation.json` | `"trivial_baseline": 0.4265` |
| 7.6 | Discounted BPA | **ALL VACUOUS** | α=1.0 → m(Θ)=1.0 for all cells | `results/geometry/bpa_geometry.csv` | All rows: `m_Theta=1.0` |
| 8.1 | Repeatability script | **MATCH** | `scripts/geometry_repeatability.py` | — | Processes C05_top r0/r1/r2 repeats |
| 8.2 | Repeatability coupon | **C05_top** | 3 re-placement repeats | `results/geometry/_diagnostics/repeatability_C05_top_axisseeds.json` | `"captures": ["C05_top_r0", "C05_top_r1", "C05_top_r2"]` |
| 8.3 | Repeatability cells | **10** | 10 cells pooled | `repeatability_C05_top_axisseeds.json:11-22` | 10 per-cell SD values listed |

---

## Critical Findings

### 1. Undercut Excluded from BPA Construction (Item 6.2–6.3)

**Status**: Correctly handled
**Reason**: Measured repeatability (0.586 mm) exceeds the Level B limit (0.5 mm). Signal-to-uncertainty ratio is 0.85:1, which is unsupportable.

**Evidence**:
```python
# rules.py:315-317
# NOTE: Undercut is EXCLUDED from the U disjunction (measured repeatability
# 0.586 mm exceeds 0.5 mm Level B limit; S/U ratio 0.85:1 is unsupportable).
```

**Impact**: Only reinforcement and misalignment drive m({U}). Undercut values are recorded for completeness but do not contribute to the BPA.

---

### 2. Alpha Clamped to 1.0 — Geometry Fully Discounted (Item 7.3–7.6)

**Status**: Expected given the data
**Reason**: F1 (0.204) < trivial baseline (0.4265), so α = 1.0 and all geometry evidence is discounted to m(Θ)=1.

**Evidence**:
```json
{
  "f1_pooled": 0.2040816326530612,
  "trivial_baseline": 0.4265,
  "alpha_G": 1.0,
  "clamped": true,
  "note": "Geometry predicts all {U} due to misalignment exceedance; F1 < trivial, alpha clamped to 1.0"
}
```

**Impact**: Geometry contributes no evidence to the fused BPA. All 150 cells have discounted m(Θ)=1.0.

---

### 3. Misalignment Dominates All Decided Cells (Item 7.4)

**Observation**: Of 39 non-vacuous cells, 39/39 have `governing_measurand: "misalignment"` with P(exceedance)≈1.0.

**Evidence** (from `summary.json`):
- Every non-vacuous cell shows `"governing_measurand": "misalignment"`
- All have `"p_misalignment": 1.0` or very close to 1.0
- The measured misalignment values (1.0–10.1 mm) far exceed the 1.0 mm limit

**Root Cause**: The weld specimens have significant linear misalignment (hi-lo between plate halves), likely from fit-up or distortion. With u_misalignment=0.275 mm and limit=1.0 mm, even 1.3 mm misalignment gives P(exceedance)≈1.0.

---

### 4. 74% Vacuous Rate (111/150 cells)

**Observation**: 111 of 150 cells (74%) map to vacuous BPA due to insufficient valid depth pixels.

**Evidence**:
```json
"n_vacuous": 111,
"n_cells": 150
```

**Root Cause**: The 60% `min_valid_fraction` threshold is not met in these cells, likely due to:
- Stereo occlusion in concave bead geometry
- Specular reflection from mill scale or weld spatter
- Depth discontinuities at bead/plate transitions

---

## Confirmed Matches

1. **Depth averaging**: 64 frames, split-half noise estimation ✓
2. **Plane-fit ROI**: 15–40 mm from centreline, linear datum mode ✓
3. **RANSAC parameters**: 100 iterations, 0.5 mm threshold ✓
4. **Profiles per cell**: 10, with 95th percentile aggregation ✓
5. **ISO 5817 limits**: All four correctly encoded for t=10 mm ✓
6. **Exceedance formula**: Φ((d̂ − limit) / u) ✓
7. **Measured repeatability**: u_undercut=0.586, u_reinforcement=0.493, u_misalignment=0.275 mm ✓
8. **Undercut exclusion**: Correctly excluded due to u > limit ✓
9. **Alpha derivation**: F1 < trivial baseline → α=1.0 clamped ✓

---

## Hyperparameter Summary

### Capture Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Sensor | Intel RealSense D435i | `configs/geometry.yaml:6` |
| Depth resolution | 1280×720 | `configs/geometry.yaml:7` |
| Working distance | 320 mm | `configs/geometry.yaml:9` |
| Depth frames per face | 64 | CAPTURE-SOP.md:152 |
| Depth units | 0.0001 (100 µm) | CLAUDE.md rule 4 |

### Processing Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Cell width | 25 mm | `configs/geometry.yaml:39` |
| Excluded zone (start/end) | 25 mm each | `configs/geometry.yaml:40-41` |
| Profiles per cell | 10 | `configs/geometry.yaml:50` |
| Aggregation quantile | 0.95 | `configs/geometry.yaml:57` |
| Smoothing kernel | 3 px | `configs/geometry.yaml:63` |
| Plane-fit ROI | 15–40 mm from centreline | `configs/geometry.yaml:73-74` |
| Datum mode | linear | `configs/geometry.yaml:93` |
| Min valid fraction | 0.60 | `configs/geometry.yaml:104` |
| RANSAC iterations | 100 | `profile.py:177` |
| RANSAC inlier threshold | 0.5 mm | `profile.py:178` |

### Uncertainty Budget (Measured Repeatability)

| Measurand | u (mm) | Limit (mm) | S/U Ratio | Used in BPA |
|-----------|--------|------------|-----------|-------------|
| Undercut | 0.586 | 0.5 | 0.85:1 | **NO** |
| Reinforcement | 0.493 | 1.0–5.0 | 2.0:1+ | Yes |
| Misalignment | 0.275 | 1.0 | 3.6:1 | Yes |

### ISO 5817 Level B Limits (t=10 mm)

| Imperfection | Formula | Limit at t=10 |
|--------------|---------|---------------|
| Undercut (5011/5012) | min(0.05t, 0.5) | 0.5 mm |
| Excess weld metal (502) | min(1 + 0.1b, 5) | 1 + 0.1b mm |
| Underfill (511) | min(0.05t, 0.5) | 0.5 mm |
| Misalignment (507) | min(0.1t, 3) | 1.0 mm |

---

## Provenance

- **Verification date**: 2026-10-29
- **Entry script**: `scripts/run_geometry_full.py`
- **Config file**: `configs/geometry.yaml`
- **Results files**: `results/geometry/summary.json`, `results/geometry/alpha_derivation.json`
- **Repeatability source**: `results/geometry/_diagnostics/repeatability_C05_top_axisseeds.json`
- **Repeatability script**: `scripts/geometry_repeatability.py`
