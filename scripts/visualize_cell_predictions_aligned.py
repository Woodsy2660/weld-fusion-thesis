#!/usr/bin/env python3
"""
Visualize vision model cell predictions aligned to actual weld ROI.

Uses the weld line selections from configs/weld_selections.json to properly
overlay predictions on the correct weld region.

Usage:
    python scripts/visualize_cell_predictions_aligned.py [coupon_id]

    If coupon_id provided, shows only that coupon interactively.
    Otherwise, processes all coupons with existing selections.
"""

import sys
from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Patch
from matplotlib.collections import PatchCollection
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Colors for focal elements
FOCAL_COLORS = {
    '{P}': '#3498db',      # Blue - porosity
    '{N,U}': '#e74c3c',    # Red - geometric
    '{N,U,L}': '#2ecc71',  # Green - no evidence
    'Theta': '#95a5a6',    # Gray - ignorance
}

ZONE_COLORS = {
    'A': '#f39c12',
    'B': '#9b59b6',
    'C': '#1abc9c',
}


def load_weld_selections():
    """Load weld line selections."""
    with open(REPO_ROOT / "configs" / "weld_selections.json") as f:
        return json.load(f)


def load_bpas():
    """Load vision BPA predictions."""
    with open(REPO_ROOT / "results" / "vision" / "coupon_eval" / "vision_bpas_v2.json") as f:
        bpas = json.load(f)
    return {b['cell_id']: b for b in bpas}


def load_labels():
    """Load ground truth labels."""
    labels = {}
    with open(REPO_ROOT / "results" / "thesis" / "data" / "labels_v2_2026-10-26.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['cell_id']] = {
                'coupon_id': row['coupon_id'],
                'cell_index': int(row['cell_index']),
                'zone': row['zone'],
                'condition': row['condition'],
                'true_class': row['true_class'],
            }
    return labels


def load_coupon_image(coupon_id, face='top', frame_idx=5):
    """Load a color image from a coupon."""
    coupon_dir = REPO_ROOT / "data" / "raw" / "S1" / f"{coupon_id}_{face}_r0"
    color_dir = coupon_dir / "color"

    if not color_dir.exists():
        return None

    img_files = sorted(color_dir.glob("*.png"))
    if frame_idx < len(img_files):
        return np.array(Image.open(img_files[frame_idx]))
    return None


def get_dominant_focal(bpa):
    """Get dominant focal element."""
    focals = {
        '{P}': bpa['m_{P}'],
        '{N,U}': bpa['m_{N,U}'],
        '{N,U,L}': bpa['m_{N,U,L}'],
    }
    dominant = max(focals, key=focals.get)
    return dominant, focals[dominant]


def get_cell_polygons(weld_line, n_cells=10, band_width=40):
    """
    Calculate cell polygon vertices along the weld line.

    Args:
        weld_line: [(x1, y1), (x2, y2)] - start and end of weld
        n_cells: Number of cells along the weld
        band_width: Width of the band perpendicular to weld (pixels)

    Returns:
        List of polygon vertices for each cell
    """
    (x1, y1), (x2, y2) = weld_line

    # Direction along weld
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    # Unit vectors
    along = np.array([dx, dy]) / length
    perp = np.array([-dy, dx]) / length  # Perpendicular

    # Half band width
    hw = band_width / 2

    polygons = []
    for i in range(n_cells):
        # Start and end fraction along weld
        t0 = i / n_cells
        t1 = (i + 1) / n_cells

        # Points along weld centerline
        p0 = np.array([x1, y1]) + t0 * length * along
        p1 = np.array([x1, y1]) + t1 * length * along

        # Four corners of the cell polygon
        corners = [
            p0 - hw * perp,  # Bottom-left
            p1 - hw * perp,  # Bottom-right
            p1 + hw * perp,  # Top-right
            p0 + hw * perp,  # Top-left
        ]

        polygons.append(np.array(corners))

    return polygons


def visualize_single_coupon(coupon_id, selections, bpas, labels, ax=None,
                           show_legend=True, band_width=50):
    """Visualize predictions for a single coupon with aligned weld ROI."""

    if coupon_id not in selections:
        print(f"No weld selection for {coupon_id}")
        return False

    sel = selections[coupon_id]
    weld_line = sel['weld_line']

    # Load image
    img = load_coupon_image(coupon_id)
    if img is None:
        print(f"No image for {coupon_id}")
        return False

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        standalone = True
    else:
        standalone = False

    ax.imshow(img)

    # Get cell polygons along weld line
    polygons = get_cell_polygons(weld_line, n_cells=10, band_width=band_width)

    # Get condition for title
    first_cell = f"{coupon_id}-00"
    condition = labels.get(first_cell, {}).get('condition', 'Unknown')
    true_class = labels.get(first_cell, {}).get('true_class', '?')

    # Draw each cell
    for cell_idx, poly in enumerate(polygons):
        cell_id = f"{coupon_id}-{cell_idx:02d}"
        bpa = bpas.get(cell_id)
        label = labels.get(cell_id, {})
        zone = label.get('zone', '?')

        if bpa:
            dominant, mass = get_dominant_focal(bpa)
            color = FOCAL_COLORS[dominant]
            alpha = min(0.7, mass * 2 + 0.2)

            # Draw polygon
            polygon = Polygon(poly, facecolor=color, alpha=alpha,
                            edgecolor='white', linewidth=2)
            ax.add_patch(polygon)

            # Add cell number at center
            center = poly.mean(axis=0)
            ax.text(center[0], center[1], str(cell_idx),
                   ha='center', va='center', fontsize=10,
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

            # Add zone color indicator
            zone_color = ZONE_COLORS.get(zone, 'gray')
            zone_marker = Polygon(poly, facecolor='none',
                                 edgecolor=zone_color, linewidth=3, linestyle='--')
            ax.add_patch(zone_marker)

    # Draw weld line
    (x1, y1), (x2, y2) = weld_line
    ax.plot([x1, x2], [y1, y2], 'w-', linewidth=2, alpha=0.8)
    ax.plot([x1, x2], [y1, y2], 'k--', linewidth=1, alpha=0.5)

    ax.set_title(f'{coupon_id}: {condition} (GT: {true_class})',
                fontsize=12, fontweight='bold')
    ax.axis('off')

    if show_legend and standalone:
        legend_elements = [
            Patch(facecolor=FOCAL_COLORS['{P}'], alpha=0.7, label='{P} Porosity'),
            Patch(facecolor=FOCAL_COLORS['{N,U}'], alpha=0.7, label='{N,U} Geometric'),
            Patch(facecolor=FOCAL_COLORS['{N,U,L}'], alpha=0.7, label='{N,U,L} No evidence'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    if standalone:
        plt.tight_layout()
        output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"cell_predictions_{coupon_id}_aligned.png",
                   dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / f'cell_predictions_{coupon_id}_aligned.png'}")
        plt.close()

    return True


def visualize_all_coupons_aligned():
    """Create grid of all coupons with aligned predictions."""
    selections = load_weld_selections()
    bpas = load_bpas()
    labels = load_labels()

    # Get coupons with selections
    coupons = sorted([c for c in selections.keys() if c.startswith('C')])

    n_cols = 3
    n_rows = (len(coupons) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten()

    for idx, coupon_id in enumerate(coupons):
        visualize_single_coupon(coupon_id, selections, bpas, labels,
                               ax=axes[idx], show_legend=False)

    # Hide unused axes
    for idx in range(len(coupons), len(axes)):
        axes[idx].axis('off')

    # Legend
    legend_elements = [
        Patch(facecolor=FOCAL_COLORS['{P}'], alpha=0.7, label='Pred: {P} (Porosity)'),
        Patch(facecolor=FOCAL_COLORS['{N,U}'], alpha=0.7, label='Pred: {N,U} (Geometric)'),
        Patch(facecolor=FOCAL_COLORS['{N,U,L}'], alpha=0.7, label='Pred: {N,U,L} (No evidence)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.01), fontsize=12)

    plt.suptitle('Vision Model Cell Predictions (Aligned to Weld ROI)\n'
                '10 cells per coupon, colored by dominant focal element',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "cell_predictions_aligned_all.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cell_predictions_aligned_all.png'}")
    plt.close()


def interactive_single_coupon(coupon_id):
    """Interactive visualization for a single coupon."""
    selections = load_weld_selections()
    bpas = load_bpas()
    labels = load_labels()

    if coupon_id not in selections:
        print(f"No weld selection found for {coupon_id}")
        print(f"Available coupons: {sorted(selections.keys())}")
        return

    # Show the visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: Image with predictions
    visualize_single_coupon(coupon_id, selections, bpas, labels,
                           ax=axes[0], show_legend=True, band_width=50)

    # Right: BPA bar chart
    ax = axes[1]
    cells = [f"{coupon_id}-{i:02d}" for i in range(10)]
    x = np.arange(10)
    width = 0.8

    bottom = np.zeros(10)
    colors_order = ['{P}', '{N,U}', '{N,U,L}', 'Theta']
    keys_order = ['m_{P}', 'm_{N,U}', 'm_{N,U,L}', 'm_Theta']

    for focal, key in zip(colors_order, keys_order):
        masses = [bpas.get(cell_id, {}).get(key, 0) for cell_id in cells]
        ax.bar(x, masses, width, bottom=bottom, label=focal,
               color=FOCAL_COLORS[focal], alpha=0.8)
        bottom += np.array(masses)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(10)])
    ax.set_xlabel('Cell Index', fontsize=12)
    ax.set_ylabel('BPA Mass', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{coupon_id}: BPA Distribution per Cell', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')

    # Zone separators
    ax.axvline(2.5, color='black', linestyle='--', alpha=0.5)
    ax.axvline(6.5, color='black', linestyle='--', alpha=0.5)
    ax.text(1, 1.02, 'Zone A', ha='center', fontsize=10, color=ZONE_COLORS['A'])
    ax.text(4.5, 1.02, 'Zone B', ha='center', fontsize=10, color=ZONE_COLORS['B'])
    ax.text(8, 1.02, 'Zone C', ha='center', fontsize=10, color=ZONE_COLORS['C'])

    # Get ground truth info
    first_cell = f"{coupon_id}-00"
    condition = labels.get(first_cell, {}).get('condition', 'Unknown')
    true_class = labels.get(first_cell, {}).get('true_class', '?')

    plt.suptitle(f'{coupon_id}: {condition} (Ground Truth Class: {true_class})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"cell_predictions_{coupon_id}_detailed.png",
               dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'cell_predictions_{coupon_id}_detailed.png'}")

    plt.show()


def main():
    if len(sys.argv) > 1:
        coupon_id = sys.argv[1].upper()
        if not coupon_id.startswith('C'):
            coupon_id = 'C' + coupon_id.zfill(2)
        print(f"Processing single coupon: {coupon_id}")
        interactive_single_coupon(coupon_id)
    else:
        print("Processing all coupons with weld selections...")
        visualize_all_coupons_aligned()


if __name__ == "__main__":
    main()
