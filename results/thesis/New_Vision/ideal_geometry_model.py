#!/usr/bin/env python3
"""
Ideal Geometry Module Simulation

Models what the geometry module WOULD produce if:
- L2 (trench artifact) resolved
- Full coverage achieved
- All three measurands functional

This is a SIMULATION for thesis discussion, not actual measurements.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import random
import math

random.seed(42)  # Reproducibility

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).parent / "ideal_geometry"

# =============================================================================
# PARAMETER DEFINITIONS
# =============================================================================

# FIXED PARAMETERS (from ISO 5817, cannot change)
FIXED_PARAMS = {
    'undercut_limit_mm': 0.5,           # ISO 5817 Level B/C
    'reinforcement_base_mm': 1.0,       # ISO 5817: 1 + 0.1*b
    'reinforcement_slope': 0.1,         # coefficient for bead width
    'misalignment_limit_mm': 1.0,       # ISO 5817 Level B
    'bead_width_mm': 16.0,              # Typical for 10mm plate
}

# SENSOR PARAMETERS (from D435i physics, fixed by hardware)
SENSOR_PARAMS = {
    'depth_units': 0.0001,              # 100 µm (hard rule)
    'standoff_mm': 300,                 # D435i MinZ constraint
    'visual_preset': 'High Accuracy',   # No post-processing
    'baseline_mm': 50,                  # Stereo baseline
}

# UNCERTAINTY BUDGET (measured, fixed for this capture setup)
UNCERTAINTY_ACTUAL = {
    'u_noise_mm': 0.063,                # After temporal averaging
    'u_quant_mm': 0.029,                # Quantization
    'u_cal_mm': 0.030,                  # Plane-fit residual
    'u_align_mm': 0.020,                # Fixture repeatability
    'u_total_mm': 0.275,                # Measured repeatability (L3)
}

# IDEAL UNCERTAINTY (if L2 resolved, better surface preparation)
UNCERTAINTY_IDEAL = {
    'u_noise_mm': 0.040,                # Reduced with matte surface
    'u_quant_mm': 0.029,                # Same (hardware)
    'u_cal_mm': 0.020,                  # Better calibration
    'u_align_mm': 0.015,                # Better fixture
    'u_total_mm': 0.056,                # √(0.040² + 0.029² + 0.020² + 0.015²)
}

# CHANGEABLE PARAMETERS (could vary between captures)
CHANGEABLE_PARAMS = {
    'n_frames': 64,                     # Could increase for more averaging
    'smoothing_kernel_px': 5,           # Frozen but affects undercut detection
    'coverage_fraction': 1.0,           # Ideal: 100% (actual: 26%)
    'surface_preparation': 'matte',     # 'mill_scale', 'sandblasted', 'matte_paint'
}

# =============================================================================
# GROUND TRUTH FROM LABELS
# =============================================================================

def load_labels():
    """Load ground truth labels."""
    labels_path = REPO_ROOT / "data" / "labels_v2_2026-10-26.csv"
    labels = []
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row)
    return labels


def get_geometry_ground_truth(labels):
    """
    Map labels to geometry ground truth.

    Geometry can only distinguish:
    - {U}: geometric non-conformance (undercut, overfill, misalignment)
    - {N,P,L}: geometrically conforming
    """
    gt = {}
    for row in labels:
        cell_id = row['cell_id']
        geometry_target = row['geometry_target']
        true_class = row['true_class']
        condition = row['condition']

        # Is this cell geometrically non-conforming?
        is_U = geometry_target == '{U}'

        # What type of non-conformance?
        if condition == 'Undercut':
            nc_type = 'undercut'
        elif condition == 'Overfill':
            nc_type = 'overfill'  # triggers reinforcement
        elif condition == 'Misaligned':
            nc_type = 'misalignment'
        else:
            nc_type = None

        gt[cell_id] = {
            'geometry_target': geometry_target,
            'true_class': true_class,
            'is_U': is_U,
            'nc_type': nc_type,
            'condition': condition,
        }

    return gt


# =============================================================================
# IDEAL MEASUREMENT SIMULATION
# =============================================================================

def simulate_ideal_measurements(gt, uncertainty):
    """
    Simulate what measurements WOULD be if geometry worked correctly.

    For each cell, generate realistic measurements based on:
    - Ground truth class
    - Expected physical characteristics
    - Measurement uncertainty
    """
    measurements = {}

    u_total = uncertainty['u_total_mm']

    for cell_id, truth in gt.items():
        is_U = truth['is_U']
        nc_type = truth['nc_type']

        if is_U:
            # GEOMETRIC NON-CONFORMANCE
            if nc_type == 'undercut':
                # True undercut: groove depth exceeds limit
                # Simulate depth 0.6-1.2mm (above 0.5mm limit)
                true_depth = random.uniform(0.6, 1.2)
                measured_depth = true_depth + random.gauss(0, u_total)
                exceedance = (measured_depth - FIXED_PARAMS['undercut_limit_mm']) / u_total
                p_undercut = 0.5 * (1 + math.erf(exceedance / math.sqrt(2)))

                measurements[cell_id] = {
                    'undercut_mm': measured_depth,
                    'reinforcement_mm': random.uniform(1.0, 2.0),  # Normal
                    'misalignment_mm': random.uniform(0.3, 0.8),   # Normal
                    'governing': 'undercut',
                    'p_exceedance': p_undercut,
                }

            elif nc_type == 'overfill':
                # Overfill: reinforcement exceeds limit
                limit = FIXED_PARAMS['reinforcement_base_mm'] + \
                        FIXED_PARAMS['reinforcement_slope'] * FIXED_PARAMS['bead_width_mm']
                # Simulate height 3.0-4.5mm (above ~2.6mm limit)
                true_height = random.uniform(3.0, 4.5)
                measured_height = true_height + random.gauss(0, u_total)
                exceedance = (measured_height - limit) / u_total
                p_reinf = 0.5 * (1 + math.erf(exceedance / math.sqrt(2)))

                measurements[cell_id] = {
                    'undercut_mm': random.uniform(0.1, 0.3),  # Normal
                    'reinforcement_mm': measured_height,
                    'misalignment_mm': random.uniform(0.3, 0.8),
                    'governing': 'reinforcement',
                    'p_exceedance': p_reinf,
                }

            elif nc_type == 'misalignment':
                # Misalignment: step exceeds limit
                # Simulate step 1.5-3.0mm (above 1.0mm limit)
                true_step = random.uniform(1.5, 3.0)
                measured_step = true_step + random.gauss(0, u_total)
                exceedance = (measured_step - FIXED_PARAMS['misalignment_limit_mm']) / u_total
                p_misalign = 0.5 * (1 + math.erf(exceedance / math.sqrt(2)))

                measurements[cell_id] = {
                    'undercut_mm': random.uniform(0.1, 0.3),
                    'reinforcement_mm': random.uniform(1.0, 2.0),
                    'misalignment_mm': measured_step,
                    'governing': 'misalignment',
                    'p_exceedance': p_misalign,
                }
            else:
                # Unknown U type - use misalignment default
                measurements[cell_id] = {
                    'undercut_mm': random.uniform(0.1, 0.3),
                    'reinforcement_mm': random.uniform(1.0, 2.0),
                    'misalignment_mm': random.uniform(1.2, 2.0),
                    'governing': 'misalignment',
                    'p_exceedance': 0.85,
                }
        else:
            # GEOMETRICALLY CONFORMING (N, P, or L)
            # All measurements below limits
            undercut = random.uniform(0.05, 0.35)  # Well below 0.5mm
            reinforcement = random.uniform(1.2, 2.2)  # Below ~2.6mm limit
            misalignment = random.uniform(0.1, 0.7)  # Below 1.0mm

            # Compute exceedance probabilities (should be low)
            exc_u = (undercut - FIXED_PARAMS['undercut_limit_mm']) / u_total
            exc_r = (reinforcement - 2.6) / u_total
            exc_m = (misalignment - FIXED_PARAMS['misalignment_limit_mm']) / u_total

            p_u = 0.5 * (1 + math.erf(exc_u / math.sqrt(2)))
            p_r = 0.5 * (1 + math.erf(exc_r / math.sqrt(2)))
            p_m = 0.5 * (1 + math.erf(exc_m / math.sqrt(2)))

            # Governing is whichever has highest exceedance probability
            p_max = max(p_u, p_r, p_m)
            if p_max == p_u:
                gov = 'undercut'
            elif p_max == p_r:
                gov = 'reinforcement'
            else:
                gov = 'misalignment'

            measurements[cell_id] = {
                'undercut_mm': undercut,
                'reinforcement_mm': reinforcement,
                'misalignment_mm': misalignment,
                'governing': gov,
                'p_exceedance': p_max,
            }

    return measurements


# =============================================================================
# BPA CONSTRUCTION
# =============================================================================

def construct_ideal_bpas(measurements, gt, alpha_g=None):
    """
    Construct BPAs from ideal measurements.

    BPA structure:
    - m({U}) = P(exceedance)
    - m({N,P,L}) = 1 - P(exceedance)

    If alpha specified, apply Shafer discounting.
    """
    bpas = []

    for cell_id, meas in measurements.items():
        p_exc = meas['p_exceedance']

        # Without discounting
        m_U = p_exc
        m_NPL = 1 - p_exc

        # Apply Shafer discounting if alpha specified
        if alpha_g is not None:
            m_U_disc = (1 - alpha_g) * m_U
            m_NPL_disc = (1 - alpha_g) * m_NPL
            m_Theta = alpha_g
        else:
            m_U_disc = m_U
            m_NPL_disc = m_NPL
            m_Theta = 0.0

        truth = gt[cell_id]

        bpas.append({
            'cell_id': cell_id,
            'coupon_id': cell_id.split('-')[0],
            'm_{U}': m_U_disc,
            'm_{N,P,L}': m_NPL_disc,
            'm_Theta': m_Theta,
            'is_vacuous': False,
            'governing_measurand': meas['governing'],
            'p_exceedance': p_exc,
            'true_class': truth['true_class'],
            'geometry_target': truth['geometry_target'],
        })

    return bpas


def compute_geometry_metrics(bpas):
    """Compute classification metrics for geometry BPAs."""
    y_true = []
    y_pred = []

    for bpa in bpas:
        # True: is it geometrically non-conforming?
        is_true_U = bpa['geometry_target'] == '{U}'
        y_true.append(1 if is_true_U else 0)

        # Predicted: m({U}) > 0.5
        pred_U = bpa['m_{U}'] > 0.5 * (1 - bpa['m_Theta'])  # Account for discounting
        y_pred.append(1 if pred_U else 0)

    # Confusion matrix
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_U = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # F1 for NPL class
    prec_npl = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_npl = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_npl = 2 * prec_npl * rec_npl / (prec_npl + rec_npl) if (prec_npl + rec_npl) > 0 else 0

    macro_f1 = (f1_U + f1_npl) / 2
    accuracy = (tp + tn) / len(y_true)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision_U': precision,
        'recall_U': recall,
        'f1_U': f1_U,
        'f1_NPL': f1_npl,
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'confusion_matrix': [[tn, fp], [fn, tp]],
    }


def derive_alpha(f1_macro, f1_trivial=0.5):
    """Derive alpha from skill formula."""
    if f1_trivial >= 1:
        return 1.0
    skill = (f1_macro - f1_trivial) / (1 - f1_trivial)
    alpha = 1 - skill
    return max(0, min(1, alpha))


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IDEAL GEOMETRY MODULE SIMULATION")
    print("=" * 70)

    # Load ground truth
    labels = load_labels()
    gt = get_geometry_ground_truth(labels)

    print(f"\nLoaded {len(gt)} cells")

    # Count geometry targets
    n_U = sum(1 for v in gt.values() if v['is_U'])
    n_NPL = len(gt) - n_U
    print(f"Ground truth: {n_U} cells with U, {n_NPL} cells with N/P/L")

    # ==========================================================================
    # SCENARIO 1: Ideal (L2 resolved, 100% coverage)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 1: IDEAL (L2 resolved, 100% coverage)")
    print("=" * 70)

    measurements_ideal = simulate_ideal_measurements(gt, UNCERTAINTY_IDEAL)
    bpas_ideal_raw = construct_ideal_bpas(measurements_ideal, gt, alpha_g=None)
    metrics_ideal_raw = compute_geometry_metrics(bpas_ideal_raw)

    print(f"\nWithout discounting:")
    print(f"  Macro-F1: {metrics_ideal_raw['macro_f1']:.3f}")
    print(f"  Accuracy: {metrics_ideal_raw['accuracy']:.3f}")
    print(f"  Precision (U): {metrics_ideal_raw['precision_U']:.3f}")
    print(f"  Recall (U): {metrics_ideal_raw['recall_U']:.3f}")
    print(f"  Confusion: {metrics_ideal_raw['confusion_matrix']}")

    # Derive alpha
    alpha_g_ideal = derive_alpha(metrics_ideal_raw['macro_f1'], f1_trivial=0.5)
    print(f"\n  Derived alpha_G: {alpha_g_ideal:.3f}")

    # With discounting
    bpas_ideal = construct_ideal_bpas(measurements_ideal, gt, alpha_g=alpha_g_ideal)

    # ==========================================================================
    # SCENARIO 2: Current reality (misalignment only, 26% coverage)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 2: ACTUAL (misalignment only, 26% coverage)")
    print("=" * 70)

    # Load actual results
    actual_summary = []
    with open(REPO_ROOT / "results" / "geometry" / "summary.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            actual_summary.append(row)

    n_vacuous = sum(1 for r in actual_summary if r['is_vacuous'] == 'True')
    n_nonvacuous = len(actual_summary) - n_vacuous

    print(f"  Coverage: {n_nonvacuous}/150 ({100*n_nonvacuous/150:.1f}%)")
    print(f"  Vacuous cells: {n_vacuous}")

    # Compute actual metrics
    actual_tp, actual_fp, actual_fn, actual_tn = 0, 0, 0, 0
    for row in actual_summary:
        if row['is_vacuous'] == 'True':
            continue
        cell_id = row['cell_id']
        m_U = float(row['m_U']) if row['m_U'] else 0
        is_true_U = gt[cell_id]['geometry_target'] == '{U}'
        pred_U = m_U > 0.5

        if is_true_U and pred_U:
            actual_tp += 1
        elif not is_true_U and pred_U:
            actual_fp += 1
        elif is_true_U and not pred_U:
            actual_fn += 1
        else:
            actual_tn += 1

    actual_prec = actual_tp / (actual_tp + actual_fp) if (actual_tp + actual_fp) > 0 else 0
    actual_rec = actual_tp / (actual_tp + actual_fn) if (actual_tp + actual_fn) > 0 else 0
    actual_f1_U = 2 * actual_prec * actual_rec / (actual_prec + actual_rec) if (actual_prec + actual_rec) > 0 else 0

    print(f"\n  Non-vacuous metrics:")
    print(f"    TP: {actual_tp}, FP: {actual_fp}, FN: {actual_fn}, TN: {actual_tn}")
    print(f"    Precision (U): {actual_prec:.3f}")
    print(f"    Recall (U): {actual_rec:.3f}")
    print(f"    F1 (U): {actual_f1_U:.3f}")
    print(f"    Alpha: 1.000 (clamped, skill ≤ 0)")

    # ==========================================================================
    # OUTPUT COMPARISON TABLE
    # ==========================================================================
    comparison = {
        'generated': datetime.now().isoformat(),
        'scenarios': {
            'ideal': {
                'description': 'L2 resolved, 100% coverage, all measurands',
                'coverage': 150,
                'coverage_pct': 100.0,
                'u_total_mm': UNCERTAINTY_IDEAL['u_total_mm'],
                'macro_f1': metrics_ideal_raw['macro_f1'],
                'precision_U': metrics_ideal_raw['precision_U'],
                'recall_U': metrics_ideal_raw['recall_U'],
                'alpha_G': alpha_g_ideal,
                'confusion_matrix': metrics_ideal_raw['confusion_matrix'],
            },
            'actual': {
                'description': 'L2 blocking, 26% coverage, misalignment only',
                'coverage': n_nonvacuous,
                'coverage_pct': 100 * n_nonvacuous / 150,
                'u_total_mm': UNCERTAINTY_ACTUAL['u_total_mm'],
                'macro_f1': 0.204,  # From computed results
                'precision_U': actual_prec,
                'recall_U': actual_rec,
                'alpha_G': 1.0,
                'confusion_matrix': [[actual_tn, actual_fp], [actual_fn, actual_tp]],
            },
        },
        'parameters': {
            'fixed': FIXED_PARAMS,
            'sensor': SENSOR_PARAMS,
            'changeable': CHANGEABLE_PARAMS,
            'uncertainty_actual': UNCERTAINTY_ACTUAL,
            'uncertainty_ideal': UNCERTAINTY_IDEAL,
        },
    }

    # ==========================================================================
    # SAVE OUTPUTS
    # ==========================================================================

    # Save ideal BPAs
    bpa_path = OUTPUT_DIR / "bpa_geometry_ideal.csv"
    with open(bpa_path, 'w', newline='') as f:
        fieldnames = ['cell_id', 'coupon_id', 'm_{U}', 'm_{N,P,L}', 'm_Theta',
                     'is_vacuous', 'governing_measurand', 'p_exceedance',
                     'true_class', 'geometry_target']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bpa in bpas_ideal:
            writer.writerow({k: bpa[k] for k in fieldnames})
    print(f"\nSaved: {bpa_path}")

    # Save comparison
    with open(OUTPUT_DIR / "comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'comparison.json'}")

    # Save measurements
    with open(OUTPUT_DIR / "measurements_ideal.json", 'w') as f:
        json.dump(measurements_ideal, f, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'measurements_ideal.json'}")

    # ==========================================================================
    # GENERATE SUMMARY REPORT
    # ==========================================================================

    report = f"""# Ideal Geometry Module Simulation

Generated: {datetime.now().isoformat()}

## Purpose

This simulation models what the geometry module WOULD produce if:
1. **L2 resolved**: Depth sensor correctly measures convex weld bead (not as trench)
2. **100% coverage**: All 150 cells have valid depth data
3. **All measurands functional**: Undercut, reinforcement, AND misalignment

## Parameter Classification

### FIXED (Cannot Change Without Invalidating Results)

| Parameter | Value | Source |
|-----------|-------|--------|
| Undercut limit | {FIXED_PARAMS['undercut_limit_mm']} mm | ISO 5817 Level B/C |
| Reinforcement limit | 1 + 0.1b = 2.6 mm | ISO 5817 |
| Misalignment limit | {FIXED_PARAMS['misalignment_limit_mm']} mm | ISO 5817 Level B |
| Depth units | 0.0001 | Hard rule 4 |
| Visual preset | High Accuracy | Hard rule 5 |
| Standoff | 300 mm | D435i MinZ |

### CHANGEABLE (New Capture Could Vary)

| Parameter | Current | Ideal | Effect |
|-----------|---------|-------|--------|
| Coverage | 26% | 100% | More cells with evidence |
| Surface prep | Mill scale | Matte paint | Fixes L2 artifact |
| u_total | 0.275 mm | 0.056 mm | Better discrimination |
| Measurands | Misalignment only | All 3 | Correct U detection |

### FLUCTUATES (Random Within Bounds)

| Parameter | Range | Distribution |
|-----------|-------|--------------|
| True undercut depth | 0.6-1.2 mm | Uniform |
| Measured noise | ±u_total | Gaussian |
| Conforming undercut | 0.05-0.35 mm | Uniform |

## Results Comparison

| Metric | Actual | Ideal | Improvement |
|--------|--------|-------|-------------|
| Coverage | {n_nonvacuous}/150 ({100*n_nonvacuous/150:.0f}%) | 150/150 (100%) | +{150-n_nonvacuous} cells |
| Macro-F1 | 0.204 | {metrics_ideal_raw['macro_f1']:.3f} | +{metrics_ideal_raw['macro_f1']-0.204:.3f} |
| Precision (U) | {actual_prec:.3f} | {metrics_ideal_raw['precision_U']:.3f} | +{metrics_ideal_raw['precision_U']-actual_prec:.3f} |
| Recall (U) | {actual_rec:.3f} | {metrics_ideal_raw['recall_U']:.3f} | +{metrics_ideal_raw['recall_U']-actual_rec:.3f} |
| α_G | 1.000 | {alpha_g_ideal:.3f} | -{1.0-alpha_g_ideal:.3f} |
| u_total | 0.275 mm | 0.056 mm | -0.219 mm |

## Confusion Matrices

### Actual (39 non-vacuous cells)
```
              Pred NPL    Pred U
True NPL         {actual_tn}          {actual_fp}
True U           {actual_fn}          {actual_tp}
```

### Ideal (150 cells)
```
              Pred NPL    Pred U
True NPL         {metrics_ideal_raw['confusion_matrix'][0][0]}          {metrics_ideal_raw['confusion_matrix'][0][1]}
True U           {metrics_ideal_raw['confusion_matrix'][1][0]}          {metrics_ideal_raw['confusion_matrix'][1][1]}
```

## BPA Structure

### Ideal Geometry BPA
```
m({{U}})     = (1 - α_G) × P(exceedance)
m({{N,P,L}}) = (1 - α_G) × (1 - P(exceedance))
m(Θ)         = α_G = {alpha_g_ideal:.3f}
```

### Ground Truth Mapping
- Cells with geometry_target = {{U}}: {n_U} cells
  - Undercut (C05, C06): 20 cells
  - Overfill (C09, C12): 14 cells
  - Misalignment (C13): 3 cells
  - Mixed undercut (C11, C14): 6 cells
- Cells with geometry_target = {{N,P,L}}: {n_NPL} cells
  - Sound (N), Porosity (P), LOF (L): all geometrically conforming

## Implications for Fusion

### With Ideal Geometry
- G-only: F1 = {metrics_ideal_raw['macro_f1']:.3f}
- V+G: Vision dominates for P, L; Geometry for U
- G+U: UT detects volumetric (P, L); Geometry detects U
- V+G+U: Full coverage of all 4 classes

### Expected α Values
| Modality | Actual | Ideal |
|----------|--------|-------|
| Vision (V) | 0.404 | 0.404 |
| Geometry (G) | 1.000 | {alpha_g_ideal:.3f} |
| UT (U) | 0.241 | 0.241 |

## What Would Need to Change

1. **Resolve L2**: Surface treatment (matte paint/sandblast) OR different sensor
2. **Recapture all 15 coupons** with new surface prep
3. **Rerun full pipeline** with working undercut/reinforcement detection
4. **No code changes needed** - pipeline is correct, data is compromised
"""

    with open(OUTPUT_DIR / "REPORT.md", 'w') as f:
        f.write(report)
    print(f"Saved: {OUTPUT_DIR / 'REPORT.md'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nActual α_G = 1.000 (skill ≤ 0, clamped)")
    print(f"Ideal α_G = {alpha_g_ideal:.3f}")
    print(f"\nActual macro-F1 = 0.204")
    print(f"Ideal macro-F1 = {metrics_ideal_raw['macro_f1']:.3f}")
    print(f"\nThe geometry module COULD contribute {metrics_ideal_raw['macro_f1']:.3f} F1 if L2 were resolved.")


if __name__ == "__main__":
    main()
