#!/usr/bin/env python3
"""
Visualize vision model cell-level predictions on validation coupons.

Shows how the model classifies each cell zone overlaid on the actual weld images.
"""

import sys
from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
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

# Zone colors for borders
ZONE_COLORS = {
    'A': '#f39c12',  # Orange
    'B': '#9b59b6',  # Purple
    'C': '#1abc9c',  # Teal
}


def load_bpas():
    """Load vision BPA predictions."""
    with open(REPO_ROOT / "results" / "vision" / "coupon_eval" / "vision_bpas_v2.json") as f:
        bpas = json.load(f)
    # Index by cell_id
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
                'vision_target': row['vision_target'],
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
    """Get the dominant focal element (excluding Theta)."""
    focals = {
        '{P}': bpa['m_{P}'],
        '{N,U}': bpa['m_{N,U}'],
        '{N,U,L}': bpa['m_{N,U,L}'],
    }
    dominant = max(focals, key=focals.get)
    return dominant, focals[dominant]


def create_single_coupon_prediction(coupon_id, bpas, labels, ax):
    """Create prediction overlay for a single coupon."""
    # Load image
    img = load_coupon_image(coupon_id)
    if img is None:
        ax.text(0.5, 0.5, f'{coupon_id}\nImage not found', ha='center', va='center')
        ax.axis('off')
        return

    h, w = img.shape[:2]
    ax.imshow(img)

    # Get condition for title
    first_cell = f"{coupon_id}-00"
    condition = labels.get(first_cell, {}).get('condition', 'Unknown')
    true_class = labels.get(first_cell, {}).get('true_class', '?')

    # Draw cell grid overlay
    n_cells = 10
    cell_width = w / n_cells

    # Cell positions (cells are along the weld, which runs horizontally)
    # Approximate weld region in the image
    weld_top = int(h * 0.35)
    weld_bottom = int(h * 0.65)
    weld_height = weld_bottom - weld_top

    for cell_idx in range(n_cells):
        cell_id = f"{coupon_id}-{cell_idx:02d}"
        bpa = bpas.get(cell_id)
        label = labels.get(cell_id, {})
        zone = label.get('zone', '?')

        x = cell_idx * cell_width
        y = weld_top

        if bpa:
            # Get dominant prediction
            dominant, mass = get_dominant_focal(bpa)
            color = FOCAL_COLORS[dominant]

            # Draw filled rectangle with alpha based on confidence
            alpha = min(0.7, mass * 2 + 0.3)  # Scale alpha with mass
            rect = Rectangle((x, y), cell_width, weld_height,
                            facecolor=color, alpha=alpha,
                            edgecolor='white', linewidth=1)
            ax.add_patch(rect)

            # Add cell number
            ax.text(x + cell_width/2, weld_top - 10, str(cell_idx),
                   ha='center', va='bottom', fontsize=8, color='white',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

            # Add zone indicator at bottom
            zone_color = ZONE_COLORS.get(zone, 'gray')
            zone_rect = Rectangle((x, weld_bottom), cell_width, 15,
                                  facecolor=zone_color, alpha=0.8)
            ax.add_patch(zone_rect)

    # Add zone labels
    ax.text(cell_width * 1, weld_bottom + 25, 'Zone A', ha='center', fontsize=9,
           color=ZONE_COLORS['A'], fontweight='bold')
    ax.text(cell_width * 4.5, weld_bottom + 25, 'Zone B', ha='center', fontsize=9,
           color=ZONE_COLORS['B'], fontweight='bold')
    ax.text(cell_width * 8, weld_bottom + 25, 'Zone C', ha='center', fontsize=9,
           color=ZONE_COLORS['C'], fontweight='bold')

    ax.set_title(f'{coupon_id}: {condition} (GT: {true_class})', fontsize=11, fontweight='bold')
    ax.axis('off')


def create_all_coupons_prediction_grid():
    """Create grid showing predictions for all 15 coupons."""
    bpas = load_bpas()
    labels = load_labels()

    # Get unique coupons
    coupons = sorted(set(l['coupon_id'] for l in labels.values()))

    fig, axes = plt.subplots(5, 3, figsize=(18, 25))
    axes = axes.flatten()

    for idx, coupon_id in enumerate(coupons):
        create_single_coupon_prediction(coupon_id, bpas, labels, axes[idx])

    # Legend
    legend_elements = [
        Patch(facecolor=FOCAL_COLORS['{P}'], label='Pred: {P} (Porosity)'),
        Patch(facecolor=FOCAL_COLORS['{N,U}'], label='Pred: {N,U} (Geometric)'),
        Patch(facecolor=FOCAL_COLORS['{N,U,L}'], label='Pred: {N,U,L} (No evidence)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.01), fontsize=11)

    plt.suptitle('Vision Model Cell Predictions on All 15 Coupons\n'
                '(Color = predicted focal element, overlay on weld region)',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "cell_predictions_all_coupons.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cell_predictions_all_coupons.png'}")
    plt.close()


def create_detailed_prediction_view():
    """Create detailed view showing predictions with BPA bars for select coupons."""
    bpas = load_bpas()
    labels = load_labels()

    # Select one coupon per condition
    conditions = {
        'Porosity': 'C01',
        'Sound': 'C03',
        'Undercut': 'C05',
        'LOF': 'C07',
        'Misaligned': 'C13',
    }

    fig, axes = plt.subplots(5, 2, figsize=(16, 20))

    for row, (condition, coupon_id) in enumerate(conditions.items()):
        # Left: Image with overlay
        ax_img = axes[row, 0]
        create_single_coupon_prediction(coupon_id, bpas, labels, ax_img)

        # Right: BPA bar chart for each cell
        ax_bar = axes[row, 1]

        cells = [f"{coupon_id}-{i:02d}" for i in range(10)]
        x = np.arange(10)
        width = 0.8

        # Stack bars
        bottom = np.zeros(10)
        colors_order = ['{P}', '{N,U}', '{N,U,L}', 'Theta']
        keys_order = ['m_{P}', 'm_{N,U}', 'm_{N,U,L}', 'm_Theta']

        for focal, key in zip(colors_order, keys_order):
            masses = []
            for cell_id in cells:
                bpa = bpas.get(cell_id, {})
                masses.append(bpa.get(key, 0))
            ax_bar.bar(x, masses, width, bottom=bottom, label=focal,
                      color=FOCAL_COLORS[focal], alpha=0.8)
            bottom += np.array(masses)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([str(i) for i in range(10)])
        ax_bar.set_xlabel('Cell Index')
        ax_bar.set_ylabel('BPA Mass')
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_title(f'{coupon_id} BPA Distribution per Cell', fontsize=11)

        # Add zone separators
        ax_bar.axvline(2.5, color='black', linestyle='--', alpha=0.5)
        ax_bar.axvline(6.5, color='black', linestyle='--', alpha=0.5)

        # Zone labels
        ax_bar.text(1, 1.02, 'Zone A', ha='center', fontsize=9, color=ZONE_COLORS['A'])
        ax_bar.text(4.5, 1.02, 'Zone B', ha='center', fontsize=9, color=ZONE_COLORS['B'])
        ax_bar.text(8, 1.02, 'Zone C', ha='center', fontsize=9, color=ZONE_COLORS['C'])

        if row == 0:
            ax_bar.legend(loc='upper right', fontsize=8)

    plt.suptitle('Vision Model Predictions: Image Overlay + BPA Distribution\n'
                '(5 Representative Coupons)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "cell_predictions_detailed.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cell_predictions_detailed.png'}")
    plt.close()


def create_prediction_accuracy_heatmap():
    """Create heatmap showing prediction accuracy by cell and condition."""
    bpas = load_bpas()
    labels = load_labels()

    # Map ground truth to expected dominant focal
    gt_to_focal = {
        'P': '{P}',
        'N': '{N,U,L}',  # Sound -> no defect evidence expected
        'U': '{N,U}',
        'L': '{N,U,L}',  # LOF is subsurface, vision can't see it
    }

    # Build accuracy matrix
    coupons = sorted(set(l['coupon_id'] for l in labels.values()))
    n_cells = 10

    accuracy_matrix = np.zeros((len(coupons), n_cells))
    prediction_matrix = np.empty((len(coupons), n_cells), dtype=object)

    for i, coupon_id in enumerate(coupons):
        for j in range(n_cells):
            cell_id = f"{coupon_id}-{j:02d}"
            bpa = bpas.get(cell_id)
            label = labels.get(cell_id, {})

            if bpa and label:
                true_class = label['true_class']
                expected_focal = gt_to_focal.get(true_class, '{N,U,L}')
                dominant, mass = get_dominant_focal(bpa)

                prediction_matrix[i, j] = dominant
                if dominant == expected_focal:
                    accuracy_matrix[i, j] = 1  # Correct
                elif true_class == 'L' and dominant == '{N,U,L}':
                    accuracy_matrix[i, j] = 0.5  # Acceptable for LOF
                else:
                    accuracy_matrix[i, j] = 0  # Incorrect

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('accuracy', ['#e74c3c', '#f39c12', '#2ecc71'])
    im = ax.imshow(accuracy_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add cell annotations
    for i in range(len(coupons)):
        for j in range(n_cells):
            pred = prediction_matrix[i, j]
            if pred:
                # Abbreviate focal element
                abbrev = {'P': 'P', 'N,U': 'NU', 'N,U,L': 'NUL'}
                pred_short = pred.replace('{', '').replace('}', '')
                pred_abbrev = abbrev.get(pred_short, pred_short)

                color = 'white' if accuracy_matrix[i, j] < 0.7 else 'black'
                ax.text(j, i, pred_abbrev, ha='center', va='center',
                       fontsize=8, color=color, fontweight='bold')

    # Labels
    ax.set_xticks(range(n_cells))
    ax.set_xticklabels([str(i) for i in range(n_cells)])
    ax.set_yticks(range(len(coupons)))

    # Y-axis labels with condition
    ylabels = []
    for coupon_id in coupons:
        cell_id = f"{coupon_id}-00"
        cond = labels.get(cell_id, {}).get('condition', '')
        true_cls = labels.get(cell_id, {}).get('true_class', '')
        ylabels.append(f"{coupon_id} ({cond[:3]}/{true_cls})")
    ax.set_yticklabels(ylabels)

    ax.set_xlabel('Cell Index', fontsize=12)
    ax.set_ylabel('Coupon (Condition/Class)', fontsize=12)

    # Zone separators
    ax.axvline(2.5, color='white', linewidth=2)
    ax.axvline(6.5, color='white', linewidth=2)

    # Zone labels
    ax.text(1, -1, 'Zone A', ha='center', fontsize=10, fontweight='bold')
    ax.text(4.5, -1, 'Zone B', ha='center', fontsize=10, fontweight='bold')
    ax.text(8, -1, 'Zone C', ha='center', fontsize=10, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Prediction Alignment')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Mismatch', 'Partial', 'Match'])

    ax.set_title('Vision Model Predictions vs Ground Truth\n'
                '(P=Porosity, NU=Geometric, NUL=No evidence)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "prediction_accuracy_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'prediction_accuracy_heatmap.png'}")
    plt.close()


def create_confusion_by_condition():
    """Create confusion analysis showing predictions by ground truth condition."""
    bpas = load_bpas()
    labels = load_labels()

    # Group predictions by condition
    by_condition = {}
    for cell_id, bpa in bpas.items():
        label = labels.get(cell_id, {})
        condition = label.get('condition', 'Unknown')
        if condition not in by_condition:
            by_condition[condition] = {'predictions': [], 'masses': []}

        dominant, mass = get_dominant_focal(bpa)
        by_condition[condition]['predictions'].append(dominant)
        by_condition[condition]['masses'].append({
            '{P}': bpa['m_{P}'],
            '{N,U}': bpa['m_{N,U}'],
            '{N,U,L}': bpa['m_{N,U,L}'],
        })

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    conditions = ['Porosity', 'Sound', 'Undercut', 'LOF', 'Overfill', 'Misaligned']

    for idx, condition in enumerate(conditions):
        ax = axes[idx]

        if condition not in by_condition:
            ax.text(0.5, 0.5, f'{condition}\nNo data', ha='center', va='center')
            ax.axis('off')
            continue

        data = by_condition[condition]
        preds = data['predictions']

        # Count predictions
        counts = {'{P}': 0, '{N,U}': 0, '{N,U,L}': 0}
        for p in preds:
            counts[p] += 1

        # Bar chart
        focals = list(counts.keys())
        vals = list(counts.values())
        colors = [FOCAL_COLORS[f] for f in focals]

        bars = ax.bar(focals, vals, color=colors, alpha=0.8, edgecolor='black')

        # Add count labels
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Count')
        ax.set_title(f'{condition}\n(n={len(preds)} cells)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.2 if vals else 10)

    plt.suptitle('Vision Model Predictions by Ground Truth Condition\n'
                '(Which focal element does the model predict for each defect type?)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "predictions_by_condition.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'predictions_by_condition.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("CELL-LEVEL PREDICTION VISUALIZATION")
    print("=" * 60)

    print("\n1. Creating all coupons prediction grid...")
    create_all_coupons_prediction_grid()

    print("\n2. Creating detailed prediction view...")
    create_detailed_prediction_view()

    print("\n3. Creating prediction accuracy heatmap...")
    create_prediction_accuracy_heatmap()

    print("\n4. Creating predictions by condition...")
    create_confusion_by_condition()

    print("\n" + "=" * 60)
    print("All figures saved to results/vision/thesis_figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
