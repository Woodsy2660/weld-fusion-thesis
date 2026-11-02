#!/usr/bin/env python3
"""
Visualize geometry V2 asymmetry-based BPA results.

Generates figures for thesis Chapter 5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.geometry_v2.rules import build_all_bpas, GeometryV2Config


def load_labels() -> pd.DataFrame:
    """Load ground truth labels."""
    labels_file = REPO_ROOT / "results" / "thesis" / "data" / "labels_v2_2026-10-26.csv"
    df = pd.read_csv(labels_file)
    return df.groupby('coupon_id').first().reset_index()


def create_bpa_distribution_figure():
    """Create figure showing BPA mass distribution by true class."""
    labels = load_labels()
    coupon_classes = dict(zip(labels['coupon_id'], labels['true_class']))

    bpas = build_all_bpas()

    # Build data for plotting
    data = []
    for coupon_id, bpa in bpas.items():
        data.append({
            'coupon': coupon_id,
            'true_class': coupon_classes.get(coupon_id, '?'),
            'asymmetry_mm': bpa.asymmetry_mm,
            'm_LU': bpa.m_LU,
            'm_NPL': bpa.m_NPL,
            'm_U': bpa.m_U,
            'm_Theta': bpa.m_Theta,
            'decision': bpa.decision,
        })

    df = pd.DataFrame(data)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Color map for classes
    colors = {'N': '#2ecc71', 'U': '#e74c3c', 'P': '#3498db', 'L': '#9b59b6'}

    # 1. Asymmetry by true class
    ax = axes[0]
    classes = ['N', 'U', 'P', 'L']
    x = np.arange(len(classes))
    width = 0.6

    means = [df[df['true_class'] == c]['asymmetry_mm'].mean() for c in classes]
    stds = [df[df['true_class'] == c]['asymmetry_mm'].std() for c in classes]
    bar_colors = [colors[c] for c in classes]

    bars = ax.bar(x, means, width, yerr=stds, color=bar_colors, alpha=0.7,
                  capsize=5, edgecolor='black', linewidth=1)

    ax.axhline(3.5, color='red', linestyle='--', linewidth=2, label='High threshold (3.5mm)')
    ax.axhline(2.0, color='orange', linestyle='--', linewidth=2, label='Mod threshold (2.0mm)')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('Plate Asymmetry (mm)', fontsize=11)
    ax.set_title('Asymmetry by True Class', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 7)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Mass on {L,U} by true class
    ax = axes[1]
    means_lu = [df[df['true_class'] == c]['m_LU'].mean() for c in classes]

    bars = ax.bar(x, means_lu, width, color=bar_colors, alpha=0.7,
                  edgecolor='black', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('Mass m({L,U})', fontsize=11)
    ax.set_title('Mass on {L,U} by True Class', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax.annotate('LOF correctly\ndetected',
                xy=(3, means_lu[3]), xytext=(2.5, 0.8),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, ha='center')

    # 3. Mass on {N,P,L} by true class
    ax = axes[2]
    means_npl = [df[df['true_class'] == c]['m_NPL'].mean() for c in classes]

    bars = ax.bar(x, means_npl, width, color=bar_colors, alpha=0.7,
                  edgecolor='black', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('Mass m({N,P,L})', fontsize=11)
    ax.set_title('Mass on {N,P,L} by True Class', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax.annotate('Undercut correctly\nassigned',
                xy=(1, means_npl[1]), xytext=(1.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, ha='center')

    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "geometry_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "bpa_distribution_by_class.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'bpa_distribution_by_class.png'}")

    plt.close()


def create_decision_flow_figure():
    """Create figure showing decision flow for BPA construction."""
    labels = load_labels()
    coupon_classes = dict(zip(labels['coupon_id'], labels['true_class']))
    coupon_conditions = dict(zip(labels['coupon_id'], labels['condition']))

    bpas = build_all_bpas()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort by asymmetry
    sorted_coupons = sorted(bpas.keys(), key=lambda c: bpas[c].asymmetry_mm, reverse=True)

    y_pos = np.arange(len(sorted_coupons))

    # Color by decision
    decision_colors = {
        'asymmetry_high': '#e74c3c',
        'asymmetry_mod': '#f39c12',
        'conforming': '#2ecc71',
        'vacuous': '#95a5a6',
    }

    # Bar chart of asymmetry
    asymmetries = [bpas[c].asymmetry_mm for c in sorted_coupons]
    bar_colors = [decision_colors[bpas[c].decision] for c in sorted_coupons]

    bars = ax.barh(y_pos, asymmetries, color=bar_colors, edgecolor='black', linewidth=0.5)

    # Threshold lines
    ax.axvline(3.5, color='red', linestyle='--', linewidth=2, label='High threshold (3.5mm)')
    ax.axvline(2.0, color='orange', linestyle='--', linewidth=2, label='Mod threshold (2.0mm)')

    # Labels
    ax.set_yticks(y_pos)
    labels_text = [f"{c} ({coupon_classes.get(c, '?')}) - {coupon_conditions.get(c, '?')}"
                   for c in sorted_coupons]
    ax.set_yticklabels(labels_text, fontsize=10)
    ax.set_xlabel('Plate Asymmetry (mm)', fontsize=12)
    ax.set_title('Geometry V2 Decision Flow: Asymmetry → BPA', fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='asymmetry_high → m({L,U})=0.6'),
        Patch(facecolor='#f39c12', label='asymmetry_mod → m({L,U})=0.3'),
        Patch(facecolor='#2ecc71', label='conforming → m({N,P,L})=0.9'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.set_xlim(0, 6)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "geometry_v2"
    fig.savefig(output_dir / "decision_flow.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'decision_flow.png'}")

    plt.close()


def create_comparison_figure():
    """Create figure comparing original vs V2 geometry focal elements."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original geometry focal elements
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Original Geometry\n(alpha_G = 1.0)', fontsize=14, fontweight='bold')

    # Draw focal elements
    from matplotlib.patches import Circle, FancyBboxPatch

    # G1: {U}
    rect1 = FancyBboxPatch((1, 6), 3, 2, boxstyle="round,pad=0.1",
                           facecolor='#e74c3c', alpha=0.5, edgecolor='black')
    ax.add_patch(rect1)
    ax.text(2.5, 7, 'G1: {U}', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 6.3, 'ISO 5817\nexceedance', ha='center', va='center', fontsize=9)

    # G2: {N,P,L}
    rect2 = FancyBboxPatch((5, 6), 4, 2, boxstyle="round,pad=0.1",
                           facecolor='#2ecc71', alpha=0.5, edgecolor='black')
    ax.add_patch(rect2)
    ax.text(7, 7, 'G2: {N,P,L}', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(7, 6.3, 'Conforming', ha='center', va='center', fontsize=9)

    # Status
    ax.text(5, 3, 'STATUS: Vacuous\nCenterline depth SNR = 0.23\nNo contribution to fusion',
            ha='center', va='center', fontsize=11, color='#c0392b',
            bbox=dict(boxstyle='round', facecolor='#fadbd8', edgecolor='#c0392b'))

    # V2 geometry focal elements
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Geometry V2\n(asymmetry-based)', fontsize=14, fontweight='bold')

    # G1: {U}
    rect1 = FancyBboxPatch((0.5, 6), 2.5, 2, boxstyle="round,pad=0.1",
                           facecolor='#e74c3c', alpha=0.5, edgecolor='black')
    ax.add_patch(rect1)
    ax.text(1.75, 7, 'G1: {U}', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.75, 6.3, 'exceedance', ha='center', va='center', fontsize=8)

    # G2: {N,P,L}
    rect2 = FancyBboxPatch((3.5, 6), 3, 2, boxstyle="round,pad=0.1",
                           facecolor='#2ecc71', alpha=0.5, edgecolor='black')
    ax.add_patch(rect2)
    ax.text(5, 7, 'G2: {N,P,L}', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 6.3, 'conforming', ha='center', va='center', fontsize=8)

    # G3: {L,U} - NEW
    rect3 = FancyBboxPatch((7, 6), 2.5, 2, boxstyle="round,pad=0.1",
                           facecolor='#9b59b6', alpha=0.5, edgecolor='black', linewidth=2)
    ax.add_patch(rect3)
    ax.text(8.25, 7, 'G3: {L,U}', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(8.25, 6.3, 'asymmetry', ha='center', va='center', fontsize=8)
    ax.text(9.7, 7.7, 'NEW', ha='center', va='center', fontsize=10, color='white',
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='#8e44ad'))

    # Status
    ax.text(5, 3, 'STATUS: Contributing\nAsymmetry SNR = 1.75\n100% LOF sensitivity',
            ha='center', va='center', fontsize=11, color='#27ae60',
            bbox=dict(boxstyle='round', facecolor='#d5f5e3', edgecolor='#27ae60'))

    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "geometry_v2"
    fig.savefig(output_dir / "comparison_original_vs_v2.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'comparison_original_vs_v2.png'}")

    plt.close()


def main():
    print("=" * 60)
    print("GEOMETRY V2 VISUALIZATION")
    print("=" * 60)

    create_bpa_distribution_figure()
    create_decision_flow_figure()
    create_comparison_figure()

    print("\nAll figures saved to results/geometry_v2/")


if __name__ == "__main__":
    main()
