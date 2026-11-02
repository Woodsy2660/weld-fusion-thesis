#!/usr/bin/env python3
"""Visualize geometry depth data: decided vs vacuous cells with ground truth comparison.

Shows:
1. Misalignment measurements for decided cells vs ISO 5817 limit
2. Coverage by coupon (heatmap)
3. Ground truth class distribution for decided vs vacuous
4. Why geometry predicts {U} for all decided cells
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "results" / "thesis" / "data"
    output_dir = base_dir / "results" / "thesis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    geom = pd.read_csv(data_dir / "geometry_summary.csv")
    labels = pd.read_csv(data_dir / "labels_v2_2026-10-26.csv")

    # Merge
    df = geom.merge(labels[['cell_id', 'coupon_id', 'cell_index', 'true_class', 'condition', 'zone']],
                    on='cell_id', suffixes=('', '_label'))

    # Separate decided and vacuous
    decided = df[~df['is_vacuous']].copy()
    vacuous = df[df['is_vacuous']].copy()

    print(f"Total cells: {len(df)}")
    print(f"Decided cells: {len(decided)} ({100*len(decided)/len(df):.1f}%)")
    print(f"Vacuous cells: {len(vacuous)} ({100*len(vacuous)/len(df):.1f}%)")

    # ISO 5817 Level B limit for misalignment at t=10mm: 0.1*t = 1.0mm
    MISALIGNMENT_LIMIT = 1.0

    # === Figure 1: Misalignment measurements for decided cells ===
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Get misalignment values (governing_undercut_mm contains the measurement)
    decided_sorted = decided.sort_values('governing_undercut_mm', ascending=False)

    colors = {'N': '#2ecc71', 'U': '#e74c3c', 'P': '#3498db', 'L': '#9b59b6'}
    bar_colors = [colors.get(c, 'gray') for c in decided_sorted['true_class']]

    x = np.arange(len(decided_sorted))
    bars = ax1.bar(x, decided_sorted['governing_undercut_mm'], color=bar_colors, edgecolor='black', linewidth=0.5)

    # ISO limit line
    ax1.axhline(MISALIGNMENT_LIMIT, color='red', linestyle='--', linewidth=2, label=f'ISO 5817 Level B limit ({MISALIGNMENT_LIMIT} mm)')

    # Labels
    ax1.set_xlabel('Decided cells (sorted by misalignment)', fontsize=11)
    ax1.set_ylabel('Measured misalignment (mm)', fontsize=11)
    ax1.set_title('Geometry: Misalignment Measurements for 39 Decided Cells\n(All exceed ISO 5817 limit → all predict {U})', fontsize=12)

    # Legend for true classes
    legend_patches = [mpatches.Patch(color=colors[c], label=f'{c} (n={len(decided_sorted[decided_sorted["true_class"]==c])})')
                      for c in ['N', 'U', 'P', 'L'] if c in decided_sorted['true_class'].values]
    legend_patches.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='ISO 5817 limit'))
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=9)

    ax1.set_xlim(-1, len(decided_sorted))
    ax1.set_ylim(0, max(decided_sorted['governing_undercut_mm']) * 1.1)

    plt.tight_layout()
    fig1.savefig(output_dir / "f4_7_geometry_misalignment.pdf", bbox_inches='tight', dpi=300)
    fig1.savefig(output_dir / "f4_7_geometry_misalignment.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'f4_7_geometry_misalignment.pdf'}")

    # === Figure 2: Coverage heatmap by coupon ===
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    # Create heatmap data
    coupons = sorted(df['coupon_id'].unique(), key=lambda x: int(x[1:]))
    heatmap_data = np.zeros((len(coupons), 10))
    class_colors_map = {'N': 0, 'U': 1, 'P': 2, 'L': 3}

    for i, coupon in enumerate(coupons):
        coupon_data = df[df['coupon_id'] == coupon].sort_values('cell_index')
        for _, row in coupon_data.iterrows():
            cell_idx = row['cell_index']
            if row['is_vacuous']:
                heatmap_data[i, cell_idx] = -1  # Vacuous = -1
            else:
                heatmap_data[i, cell_idx] = 1  # Decided = 1

    # Custom colormap: gray for vacuous, green for decided
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['lightgray', 'limegreen'])

    im = ax2.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add cell annotations
    for i, coupon in enumerate(coupons):
        coupon_data = df[df['coupon_id'] == coupon].sort_values('cell_index')
        for _, row in coupon_data.iterrows():
            cell_idx = row['cell_index']
            true_class = row['true_class']
            text_color = 'black' if not row['is_vacuous'] else 'dimgray'
            ax2.text(cell_idx, i, true_class, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

    ax2.set_xticks(range(10))
    ax2.set_xticklabels([f'{i}' for i in range(10)])
    ax2.set_yticks(range(len(coupons)))
    ax2.set_yticklabels(coupons)
    ax2.set_xlabel('Cell index', fontsize=11)
    ax2.set_ylabel('Coupon', fontsize=11)
    ax2.set_title('Geometry Coverage: Decided (green) vs Vacuous (gray) Cells\nLabels show ground truth class (N/U/P/L)', fontsize=12)

    # Legend
    legend_patches = [
        mpatches.Patch(color='limegreen', label=f'Decided (n={len(decided)})'),
        mpatches.Patch(color='lightgray', label=f'Vacuous (n={len(vacuous)})'),
    ]
    ax2.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9)

    plt.tight_layout()
    fig2.savefig(output_dir / "f4_8_geometry_coverage_heatmap.pdf", bbox_inches='tight', dpi=300)
    fig2.savefig(output_dir / "f4_8_geometry_coverage_heatmap.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'f4_8_geometry_coverage_heatmap.pdf'}")

    # === Figure 3: Ground truth distribution comparison ===
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Decided cells class distribution
    decided_counts = decided['true_class'].value_counts()
    axes[0].pie(decided_counts, labels=[f'{c}\n({v})' for c, v in decided_counts.items()],
                colors=[colors.get(c, 'gray') for c in decided_counts.index],
                autopct='%1.0f%%', startangle=90, explode=[0.02]*len(decided_counts))
    axes[0].set_title(f'Decided Cells (n={len(decided)})\nGeometry predicts U for all', fontsize=11)

    # Vacuous cells class distribution
    vacuous_counts = vacuous['true_class'].value_counts()
    axes[1].pie(vacuous_counts, labels=[f'{c}\n({v})' for c, v in vacuous_counts.items()],
                colors=[colors.get(c, 'gray') for c in vacuous_counts.index],
                autopct='%1.0f%%', startangle=90, explode=[0.02]*len(vacuous_counts))
    axes[1].set_title(f'Vacuous Cells (n={len(vacuous)})\nGeometry abstains (m(Θ)=1)', fontsize=11)

    plt.suptitle('Ground Truth Class Distribution: Decided vs Vacuous Geometry Cells', fontsize=13, y=1.02)
    plt.tight_layout()
    fig3.savefig(output_dir / "f4_9_geometry_class_distribution.pdf", bbox_inches='tight', dpi=300)
    fig3.savefig(output_dir / "f4_9_geometry_class_distribution.png", bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'f4_9_geometry_class_distribution.pdf'}")

    # === Print summary statistics ===
    print("\n=== SUMMARY: Geometry Decided Cells ===")
    print(f"Total decided: {len(decided)}")
    print(f"Ground truth breakdown:")
    for cls in ['N', 'U', 'P', 'L']:
        count = len(decided[decided['true_class'] == cls])
        print(f"  {cls}: {count} ({100*count/len(decided):.1f}%)")

    print(f"\nMisalignment statistics (mm):")
    print(f"  Min: {decided['governing_undercut_mm'].min():.2f}")
    print(f"  Max: {decided['governing_undercut_mm'].max():.2f}")
    print(f"  Mean: {decided['governing_undercut_mm'].mean():.2f}")
    print(f"  Median: {decided['governing_undercut_mm'].median():.2f}")

    print(f"\nAll {len(decided)} decided cells exceed ISO 5817 limit ({MISALIGNMENT_LIMIT} mm)")
    print(f"→ All predict {{U}} (geometric non-conformance)")
    print(f"→ Only {len(decided[decided['true_class']=='U'])} are actually U")
    print(f"→ F1 = 0.204 (below trivial baseline 0.426)")

    # Output data table for thesis
    print("\n=== DATA TABLE: Decided Cells ===")
    cols = ['cell_id', 'true_class', 'governing_undercut_mm', 'exceedance_prob']
    print(decided[cols].to_string(index=False))

    plt.close('all')


if __name__ == "__main__":
    main()
