"""
UT modality BPA construction.

Derives alpha from skill formula and constructs Dempster-Shafer BPAs.
Focal elements: {P}, {L}, {N,U}, Θ (total ignorance)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# UT focal elements
FOCAL_ELEMENTS = ['{L}', '{N,U}', '{P}', 'Theta']
CLASS_TO_FOCAL = {
    '{L}': '{L}',
    '{N,U}': '{N,U}',
    '{P}': '{P}',
}


def derive_alpha(f1_macro: float, f1_trivial: float) -> float:
    """
    Derive discount factor alpha from skill formula.

    alpha = 1 - (F1_macro - F1_trivial) / (1 - F1_trivial)

    Interpretation: alpha is the probability mass assigned to total ignorance (Theta).
    A perfect classifier (F1=1) gives alpha=0.
    A trivial classifier (F1=F1_trivial) gives alpha=1.
    """
    if f1_trivial >= 1:
        return 1.0
    skill = (f1_macro - f1_trivial) / (1 - f1_trivial)
    alpha = 1 - skill
    return max(0, min(1, alpha))  # Clamp to [0, 1]


def construct_bpa(
    proba: np.ndarray,
    classes: list,
    alpha: float,
) -> dict:
    """
    Construct BPA from classifier probabilities with alpha discount.

    The discounted BPA assigns mass (1-alpha) * p_i to each focal element
    and mass alpha to Theta (total ignorance).

    Args:
        proba: Probability vector for this sample [P(L), P(N,U), P(P)]
        classes: Class labels in order matching proba
        alpha: Discount factor for Theta

    Returns:
        BPA dict with focal elements as keys and masses as values
    """
    bpa = {}

    # Assign discounted mass to singletons
    for i, cls in enumerate(classes):
        focal = CLASS_TO_FOCAL.get(cls)
        if focal:
            bpa[focal] = (1 - alpha) * proba[i]

    # Assign alpha to Theta
    bpa['Theta'] = alpha

    return bpa


def validate_bpa(bpa: dict, tolerance: float = 1e-6) -> bool:
    """Validate that BPA sums to 1."""
    total = sum(bpa.values())
    return abs(total - 1.0) < tolerance


def main():
    predictions_path = REPO_ROOT / 'results' / 'ut' / 'cv_predictions_calibrated.json'
    metrics_path = REPO_ROOT / 'results' / 'ut' / 'metrics_cv.json'
    features_path = REPO_ROOT / 'results' / 'ut' / 'features.csv'
    output_dir = REPO_ROOT / 'results' / 'ut'

    print("=" * 60)
    print("UT BPA CONSTRUCTION")
    print("=" * 60)

    # Load metrics for alpha derivation
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    f1_macro = metrics['f1_macro_overall']
    f1_trivial = metrics['baseline_most_frequent_f1']

    print(f"\nF1-macro: {f1_macro:.4f}")
    print(f"F1-trivial: {f1_trivial:.4f}")

    # Derive alpha
    alpha = derive_alpha(f1_macro, f1_trivial)
    print(f"Alpha (discount): {alpha:.4f}")

    # Verify skill
    skill = 1 - alpha
    print(f"Skill: {skill:.4f}")

    # Load calibrated predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)

    y_proba = np.array(predictions['y_proba'])
    classes = predictions['classes']
    y_true = predictions['y_true']

    print(f"\nSamples: {len(y_true)}")
    print(f"Classes: {classes}")
    print(f"Focal elements: {FOCAL_ELEMENTS}")

    # Load features for cell_id mapping
    features_df = pd.read_csv(features_path)

    # Construct BPAs
    bpas = []
    validation_errors = []

    for i, (proba, true_class) in enumerate(zip(y_proba, y_true)):
        bpa = construct_bpa(proba, classes, alpha)

        # Validate
        if not validate_bpa(bpa):
            validation_errors.append(i)

        # Add metadata
        cell_id = features_df.iloc[i]['cell_id']
        coupon_id = features_df.iloc[i]['coupon_id']
        fold = features_df.iloc[i]['fold']

        bpas.append({
            'cell_id': cell_id,
            'coupon_id': coupon_id,
            'fold': int(fold),
            'true_class': true_class,
            'm_{L}': float(bpa['{L}']),
            'm_{N,U}': float(bpa['{N,U}']),
            'm_{P}': float(bpa['{P}']),
            'm_Theta': float(bpa['Theta']),
        })

    # Create DataFrame
    bpa_df = pd.DataFrame(bpas)

    # Validate all rows sum to 1
    mass_cols = ['m_{L}', 'm_{N,U}', 'm_{P}', 'm_Theta']
    row_sums = bpa_df[mass_cols].sum(axis=1)
    sum_errors = (abs(row_sums - 1.0) > 1e-6).sum()

    # Validate Theta mass equals alpha
    theta_errors = (abs(bpa_df['m_Theta'] - alpha) > 1e-6).sum()

    print("\n" + "=" * 60)
    print("BPA VALIDATION")
    print("=" * 60)
    print(f"Rows with sum != 1: {sum_errors}")
    print(f"Rows with m_Theta != alpha: {theta_errors}")
    print(f"Validation: {'PASSED' if sum_errors == 0 and theta_errors == 0 else 'FAILED'}")

    # Report mass distribution
    print("\nMass statistics:")
    for col in mass_cols:
        print(f"  {col}: mean={bpa_df[col].mean():.4f}, std={bpa_df[col].std():.4f}")

    # Save BPA CSV
    bpa_path = output_dir / 'bpa_ut.csv'
    bpa_df.to_csv(bpa_path, index=False)
    print(f"\nBPAs saved to: {bpa_path}")

    # Save alpha metadata
    alpha_meta = {
        'alpha': float(alpha),
        'skill': float(skill),
        'f1_macro': float(f1_macro),
        'f1_trivial': float(f1_trivial),
        'formula': 'alpha = 1 - (F1_macro - F1_trivial) / (1 - F1_trivial)',
        'focal_elements': FOCAL_ELEMENTS,
        'n_samples': len(bpas),
        'validation_passed': bool(sum_errors == 0 and theta_errors == 0),
    }

    alpha_path = output_dir / 'alpha_ut.json'
    with open(alpha_path, 'w') as f:
        json.dump(alpha_meta, f, indent=2)
    print(f"Alpha metadata saved to: {alpha_path}")

    return bpa_df, alpha_meta


if __name__ == '__main__':
    main()
