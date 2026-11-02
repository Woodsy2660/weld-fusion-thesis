#!/usr/bin/env python3
"""Analyze signal-to-noise ratio in geometry features to assess if
geometry can contribute meaningful evidence to fusion."""

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

    # Build feature table from band analysis
    # Need to re-extract - let's use centerline stats for now
    data = []
    for coupon_id, sel in selections.items():
        if 'stats' not in sel:
            continue
        s = sel['stats']
        data.append({
            'coupon': coupon_id,
            'condition': coupon_conditions.get(coupon_id, 'Unknown'),
            'true_class': coupon_class.get(coupon_id, '?'),
            'median_depth': s['median'],
            'std': s['std'],
            'min': s['min'],
            'max': s['max']
        })

    df = pd.DataFrame(data)

    print("="*60)
    print("GEOMETRY SIGNAL-TO-NOISE ANALYSIS")
    print("="*60)

    # Group by condition
    print("\n1. MEDIAN DEPTH BY CONDITION")
    print("-"*40)
    for cond in df['condition'].unique():
        subset = df[df['condition'] == cond]['median_depth']
        print(f"{cond:12s}: mean={subset.mean():+.2f}mm, std={subset.std():.2f}mm, n={len(subset)}")

    # Group by true class
    print("\n2. MEDIAN DEPTH BY TRUE CLASS")
    print("-"*40)
    for cls in ['N', 'U', 'P', 'L']:
        subset = df[df['true_class'] == cls]['median_depth']
        if len(subset) > 0:
            print(f"{cls}: mean={subset.mean():+.2f}mm, std={subset.std():.2f}mm, n={len(subset)}")

    # Statistical tests
    print("\n3. ANOVA: Can median_depth distinguish classes?")
    print("-"*40)
    groups = [df[df['true_class'] == cls]['median_depth'].values
              for cls in ['N', 'U', 'P', 'L'] if len(df[df['true_class'] == cls]) > 0]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"F-statistic: {f_stat:.3f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Significant (p<0.05): {'Yes' if p_val < 0.05 else 'No'}")

    # Effect size (Cohen's d) for LOF vs others
    print("\n4. EFFECT SIZE: LOF vs Non-LOF")
    print("-"*40)
    lof = df[df['true_class'] == 'L']['median_depth']
    non_lof = df[df['true_class'] != 'L']['median_depth']
    if len(lof) > 1 and len(non_lof) > 1:
        pooled_std = np.sqrt(((len(lof)-1)*lof.std()**2 + (len(non_lof)-1)*non_lof.std()**2) /
                            (len(lof) + len(non_lof) - 2))
        cohens_d = (lof.mean() - non_lof.mean()) / pooled_std
        print(f"LOF mean: {lof.mean():+.2f}mm (n={len(lof)})")
        print(f"Non-LOF mean: {non_lof.mean():+.2f}mm (n={len(non_lof)})")
        print(f"Cohen's d: {cohens_d:.2f}")
        print(f"Effect size: {'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'}")

    # Separability analysis
    print("\n5. CLASS SEPARABILITY (using median depth)")
    print("-"*40)

    # Try simple threshold classifier
    thresholds = np.linspace(df['median_depth'].min(), df['median_depth'].max(), 20)
    best_acc = 0
    best_thresh = 0

    for thresh in thresholds:
        # Predict: depth < thresh -> defect class
        pred_defect = df['median_depth'] < thresh
        actual_defect = df['true_class'].isin(['U', 'L', 'P'])
        acc = (pred_defect == actual_defect).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    print(f"Best threshold: {best_thresh:.2f}mm")
    print(f"Best accuracy: {best_acc:.1%}")
    print(f"Trivial baseline (majority): {(df['true_class'] == df['true_class'].mode()[0]).mean():.1%}")

    # SNR estimation
    print("\n6. SIGNAL-TO-NOISE RATIO")
    print("-"*40)

    within_class_var = df.groupby('true_class')['median_depth'].var().mean()
    between_class_var = df.groupby('true_class')['median_depth'].mean().var()

    snr = between_class_var / within_class_var if within_class_var > 0 else 0
    print(f"Within-class variance: {within_class_var:.4f}")
    print(f"Between-class variance: {between_class_var:.4f}")
    print(f"SNR (between/within): {snr:.2f}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if snr < 0.5:
        print("\n⚠️  LOW SNR: Geometry features have poor class separability")
        print("   The depth sensor noise overwhelms the defect signal.")
        print("   Recommendation: Keep α_G = 1.0 (no contribution)")
    elif snr < 1.0:
        print("\n⚡ MODERATE SNR: Some signal present but noisy")
        print("   Possible workarounds:")
        print("   1. Use robust features (asymmetry ratio, relative depths)")
        print("   2. Aggregate over multiple cells/captures")
        print("   3. Use partial α (e.g., α_G = 0.7) instead of 1.0")
    else:
        print("\n✓ GOOD SNR: Geometry features could contribute")
        print("   The BPA construction method needs revision:")
        print("   1. Map features to appropriate focal elements")
        print("   2. Use probabilistic thresholds, not binary")

    # Alternative feature analysis using asymmetry (if available)
    print("\n" + "="*60)
    print("ALTERNATIVE: ASYMMETRY-BASED CLASSIFICATION")
    print("="*60)

    # Would need to reload band analysis data for this
    print("\nNote: Run extract_weld_band.py and analyze asymmetry")
    print("LOF coupons showed asymmetry 3.7-5.3mm vs 1.6-1.8mm for others")
    print("This suggests asymmetry may be a better discriminative feature")


if __name__ == "__main__":
    main()
