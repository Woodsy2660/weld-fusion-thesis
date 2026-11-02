#!/usr/bin/env python3
"""Analyze geometry feature quality and recommend workarounds.

Focuses on asymmetry as the discriminative feature since toe depth
readings are corrupted by sensor dropouts.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats


def main():
    base_dir = Path(__file__).parent.parent

    # Load weld selections with stats
    with open(base_dir / "configs" / "weld_selections.json") as f:
        selections = json.load(f)

    # Load labels
    labels_df = pd.read_csv(base_dir / "results" / "thesis" / "data" / "labels_v2_2026-10-26.csv")
    coupon_conditions = labels_df.groupby('coupon_id')['condition'].first().to_dict()
    coupon_class = labels_df.groupby('coupon_id')['true_class'].first().to_dict()

    # Re-run band extraction for proper asymmetry values
    # The asymmetry values from the band analysis (not affected by dropouts)
    # These were observed from running extract_weld_band.py
    band_stats = {
        'C01': {'asymmetry': 4.62, 'crown': -0.76},
        'C02': {'asymmetry': 2.91, 'crown': -0.82},
        'C03': {'asymmetry': 1.78, 'crown': -0.69},
        'C04': {'asymmetry': 0.74, 'crown': 1.95},
        'C05': {'asymmetry': 1.60, 'crown': -1.33},
        'C06': {'asymmetry': 1.41, 'crown': 0.48},
        'C07': {'asymmetry': 5.31, 'crown': -1.31},
        'C08': {'asymmetry': 4.36, 'crown': -2.30},
        'C09': {'asymmetry': 4.65, 'crown': 0.42},
        'C10': {'asymmetry': 4.31, 'crown': -0.93},
        'C11': {'asymmetry': 0.74, 'crown': -1.02},
        'C12': {'asymmetry': 0.62, 'crown': -0.33},
        'C13': {'asymmetry': 4.41, 'crown': -1.42},
        'C14': {'asymmetry': 3.72, 'crown': -1.13},
        'C15': {'asymmetry': 3.35, 'crown': -0.57}
    }

    # Build feature table
    data = []
    for coupon_id, bstats in band_stats.items():
        data.append({
            'coupon': coupon_id,
            'condition': coupon_conditions.get(coupon_id, 'Unknown'),
            'true_class': coupon_class.get(coupon_id, '?'),
            'asymmetry': bstats['asymmetry'],
            'crown': bstats['crown']
        })

    df = pd.DataFrame(data)

    print("="*70)
    print("GEOMETRY WORKAROUND ANALYSIS: ASYMMETRY AS DISCRIMINATIVE FEATURE")
    print("="*70)

    # 1. Asymmetry by condition
    print("\n1. PLATE ASYMMETRY BY CONDITION")
    print("-"*50)
    for cond in sorted(df['condition'].unique()):
        subset = df[df['condition'] == cond]['asymmetry']
        print(f"{cond:12s}: mean={subset.mean():.2f}mm, std={subset.std():.2f}mm, range=[{subset.min():.2f}, {subset.max():.2f}], n={len(subset)}")

    # 2. Asymmetry by true class
    print("\n2. ASYMMETRY BY TRUE CLASS")
    print("-"*50)
    for cls in ['N', 'U', 'P', 'L']:
        subset = df[df['true_class'] == cls]['asymmetry']
        if len(subset) > 0:
            print(f"{cls}: mean={subset.mean():.2f}mm, std={subset.std():.2f}mm, n={len(subset)}")

    # 3. ANOVA on asymmetry
    print("\n3. ANOVA: Can asymmetry distinguish classes?")
    print("-"*50)
    groups = [df[df['true_class'] == cls]['asymmetry'].values
              for cls in ['N', 'U', 'P', 'L'] if len(df[df['true_class'] == cls]) > 0]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"F-statistic: {f_stat:.3f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Significant (p<0.05): {'YES' if p_val < 0.05 else 'No'}")

    # 4. LOF vs Non-LOF separation
    print("\n4. EFFECT SIZE: LOF vs Non-LOF (Asymmetry)")
    print("-"*50)
    lof = df[df['true_class'] == 'L']['asymmetry']
    non_lof = df[df['true_class'] != 'L']['asymmetry']
    if len(lof) > 1 and len(non_lof) > 1:
        pooled_std = np.sqrt(((len(lof)-1)*lof.std()**2 + (len(non_lof)-1)*non_lof.std()**2) /
                            (len(lof) + len(non_lof) - 2))
        cohens_d = (lof.mean() - non_lof.mean()) / pooled_std
        print(f"LOF asymmetry: {lof.mean():.2f}mm (n={len(lof)})")
        print(f"Non-LOF asymmetry: {non_lof.mean():.2f}mm (n={len(non_lof)})")
        print(f"Cohen's d: {cohens_d:.2f}")
        print(f"Effect size: {'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'}")

    # 5. SNR for asymmetry
    print("\n5. SIGNAL-TO-NOISE RATIO (Asymmetry)")
    print("-"*50)
    within_class_var = df.groupby('true_class')['asymmetry'].var().mean()
    between_class_var = df.groupby('true_class')['asymmetry'].mean().var()
    snr_asymm = between_class_var / within_class_var if within_class_var > 0 else 0
    print(f"Within-class variance: {within_class_var:.4f}")
    print(f"Between-class variance: {between_class_var:.4f}")
    print(f"SNR (between/within): {snr_asymm:.2f}")

    # 6. Threshold classifier for LOF
    print("\n6. THRESHOLD CLASSIFIER: Detect LOF using asymmetry")
    print("-"*50)

    actual_lof = df['true_class'] == 'L'

    best_acc = 0
    best_thresh = 0
    best_sens = 0
    best_spec = 0

    for thresh in np.linspace(df['asymmetry'].min(), df['asymmetry'].max(), 50):
        pred_lof = df['asymmetry'] > thresh

        # Calculate metrics
        tp = ((pred_lof) & (actual_lof)).sum()
        tn = ((~pred_lof) & (~actual_lof)).sum()
        fp = ((pred_lof) & (~actual_lof)).sum()
        fn = ((~pred_lof) & (actual_lof)).sum()

        acc = (tp + tn) / len(df)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_sens = sens
            best_spec = spec

    print(f"Best threshold: {best_thresh:.2f}mm")
    print(f"Accuracy: {best_acc:.1%}")
    print(f"Sensitivity (LOF recall): {best_sens:.1%}")
    print(f"Specificity: {best_spec:.1%}")
    print(f"Trivial baseline: {(df['true_class'] != 'L').mean():.1%}")

    # 7. Multi-class decision rules
    print("\n7. PROPOSED GEOMETRY BPA CONSTRUCTION")
    print("-"*50)
    print("""
    Given asymmetry A and crown depth C (in mm):

    IF A > 3.5mm:
        m_G({L,U}) = 0.6   # High asymmetry suggests LOF or misalignment
        m_G(Theta) = 0.4   # Uncertainty remains

    ELSE IF A > 2.0mm:
        m_G({L,U}) = 0.3   # Moderate asymmetry
        m_G(Theta) = 0.7

    ELSE IF C < -1.5mm:
        m_G({U}) = 0.5     # Underfill beyond limit
        m_G(Theta) = 0.5

    ELSE IF C > 2.0mm:
        m_G({U}) = 0.4     # Excess reinforcement
        m_G({N}) = 0.2
        m_G(Theta) = 0.4

    ELSE:
        m_G({N,P,L}) = 0.3  # Geometry OK, can't distinguish
        m_G(Theta) = 0.7

    This maps asymmetry to {L,U} compound because:
    - LOF causes one-sided heat input -> plate tilts
    - Misalignment (U) also causes asymmetry
    - The sensor cannot distinguish LOF from misalignment
    """)

    # 8. Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    print(f"""
    FINDING: Asymmetry shows SNR = {snr_asymm:.2f} (vs 0.23 for centerline depth)

    OPTION A: PARTIAL DISCOUNTING
    ------------------------------
    Instead of alpha_G = 1.0 (vacuous), use:

        alpha_G = 1 - (F1_asymm - F1_trivial) / (1 - F1_trivial)

    With LOF detection at {best_acc:.0%} accuracy vs {(df['true_class'] != 'L').mean():.0%} trivial:
        alpha_G = 1 - ({best_acc:.2f} - {(df['true_class'] != 'L').mean():.2f}) / (1 - {(df['true_class'] != 'L').mean():.2f})
        alpha_G = {1 - (best_acc - (df['true_class'] != 'L').mean()) / (1 - (df['true_class'] != 'L').mean()):.2f}

    This allows geometry to contribute evidence at ~{(1 - (1 - (best_acc - (df['true_class'] != 'L').mean()) / (1 - (df['true_class'] != 'L').mean()))):.0%} reliability.

    OPTION B: COMPOUND FOCAL ELEMENT {'{L,U}'}
    ----------------------------------------
    The key insight: asymmetry discriminates BETWEEN defect modes
    but cannot distinguish LOF from misalignment. Map high asymmetry
    to the compound element {'{L,U}'} which correctly represents the
    sensor's epistemic limitation.

    OPTION C: KEEP alpha_G = 1.0
    ----------------------------
    Justification: With n=15 coupons and only n=3 LOF samples,
    the {best_acc:.0%} classifier accuracy may not generalize.
    Cross-validation would be needed to confirm.

    VERDICT: Option A or B are defensible given the data. The thesis
    should document both the limitation (centerline depth SNR=0.23)
    and the workaround (asymmetry-based BPA, SNR={snr_asymm:.2f}).
    """)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: GEOMETRY FEATURE COMPARISON")
    print("="*70)
    print(f"""
    {'Feature':<25} {'SNR':>8} {'LOF Cohen d':>12} {'Recommendation'}
    {'-'*65}
    {'Centerline depth':<25} {'0.23':>8} {'-0.96':>12}   Poor, keep alpha=1.0
    {'Plate asymmetry':<25} {snr_asymm:>8.2f} {cohens_d:>+12.2f}   Usable, revise BPA
    """)


if __name__ == "__main__":
    main()
