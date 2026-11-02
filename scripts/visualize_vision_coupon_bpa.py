#!/usr/bin/env python3
"""
Visualize vision model classifications and BPA assignments for thesis coupons.

Shows real model outputs on the 15 experimental coupons including:
- Per-cell BPA distributions
- Confusion matrices
- Model performance metrics
"""

import sys
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_bpas():
    """Load vision BPA assignments for all cells."""
    with open(REPO_ROOT / "results" / "vision" / "coupon_eval" / "vision_bpas_v2.json") as f:
        return json.load(f)


def load_canonical_results():
    """Load canonical vision results."""
    with open(REPO_ROOT / "results" / "vision" / "coupon_eval" / "vision_canonical_results.json") as f:
        return json.load(f)


def load_labels():
    """Load ground truth labels."""
    labels_file = REPO_ROOT / "results" / "thesis" / "data" / "labels_v2_2026-10-26.csv"
    labels = {}
    with open(labels_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['cell_id']] = {
                'coupon_id': row['coupon_id'],
                'condition': row['condition'],
                'true_class': row['true_class'],
            }
    return labels


def create_bpa_heatmap():
    """Create heatmap showing BPA masses for all cells."""
    bpas = load_bpas()
    labels = load_labels()

    # Organize by coupon
    coupons = {}
    for bpa in bpas:
        cell_id = bpa['cell_id']
        coupon_id = cell_id.split('-')[0]
        if coupon_id not in coupons:
            coupons[coupon_id] = []
        coupons[coupon_id].append(bpa)

    # Sort coupons
    coupon_ids = sorted(coupons.keys())

    # Create figure
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    focal_colors = {
        'm_{P}': '#3498db',
        'm_{N,U}': '#e74c3c',
        'm_{N,U,L}': '#2ecc71',
        'm_Theta': '#95a5a6',
    }

    for idx, coupon_id in enumerate(coupon_ids):
        ax = axes[idx]

        cells = sorted(coupons[coupon_id], key=lambda x: int(x['cell_id'].split('-')[1]))
        n_cells = len(cells)

        # Get condition from labels
        first_cell = cells[0]['cell_id']
        condition = labels.get(first_cell, {}).get('condition', 'Unknown')
        true_class = labels.get(first_cell, {}).get('true_class', '?')

        # Stack bars for each cell
        x = np.arange(n_cells)
        width = 0.8

        bottom = np.zeros(n_cells)
        for focal, color in focal_colors.items():
            masses = [cell[focal] for cell in cells]
            ax.bar(x, masses, width, bottom=bottom, color=color, alpha=0.8)
            bottom += np.array(masses)

        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(n_cells)], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Cell', fontsize=9)
        ax.set_ylabel('BPA Mass', fontsize=9)
        ax.set_title(f'{coupon_id}\n{condition} (GT: {true_class})', fontsize=11, fontweight='bold')

        # Add grid
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axhline(0.25, color='black', linestyle=':', alpha=0.2, linewidth=0.5)
        ax.axhline(0.75, color='black', linestyle=':', alpha=0.2, linewidth=0.5)

    # Legend
    legend_elements = [
        Patch(facecolor='#3498db', label='m({P})'),
        Patch(facecolor='#e74c3c', label='m({N,U})'),
        Patch(facecolor='#2ecc71', label='m({N,U,L})'),
        Patch(facecolor='#95a5a6', label='m(Theta)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 0.02), fontsize=11)

    plt.suptitle('Vision BPA Assignments per Cell (15 Coupons × 10 Cells)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "coupon_bpa_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'coupon_bpa_heatmap.png'}")
    plt.close()


def create_performance_summary():
    """Create figure showing model performance metrics."""
    results = load_canonical_results()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Confusion matrix (aggregated)
    ax = axes[0, 0]
    cm_agg = np.zeros((3, 3))
    for fold_key in ['fold_1', 'fold_2', 'fold_3', 'fold_4']:
        cm_agg += np.array(results['confusion_matrices'][fold_key])

    classes = ['{P}', '{N,U}', '{N,U,L}']
    im = ax.imshow(cm_agg, cmap='Blues', aspect='auto')

    # Add text annotations
    for i in range(3):
        for j in range(3):
            val = int(cm_agg[i, j])
            color = 'white' if val > cm_agg.max() / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=14)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Aggregated Confusion Matrix\n(4-fold cross-validation)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Count')

    # 2. Per-fold F1 scores
    ax = axes[0, 1]
    fold_f1 = results['metrics']['per_fold_f1']
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4']
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, 4))

    bars = ax.bar(folds, fold_f1, color=colors, edgecolor='black', alpha=0.8)

    # Add mean line
    mean_f1 = results['metrics']['macro_f1_mean']
    ax.axhline(mean_f1, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_f1:.3f}')

    # Add trivial baseline
    trivial_f1 = results['baselines']['trivial_f1']
    ax.axhline(trivial_f1, color='orange', linestyle=':', linewidth=2, label=f'Trivial: {trivial_f1:.3f}')

    ax.set_ylabel('Macro F1', fontsize=12)
    ax.set_ylim(0, 0.5)
    ax.set_title('Per-Fold Macro F1 Scores', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')

    # Add value labels
    for bar, f1 in zip(bars, fold_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{f1:.3f}', ha='center', va='bottom', fontsize=10)

    # 3. Class distribution
    ax = axes[1, 0]
    class_counts = results['class_counts']
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(classes, counts, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Number of Cells', fontsize=12)
    ax.set_xlabel('Focal Element', fontsize=12)
    ax.set_title('Ground Truth Distribution', fontsize=12, fontweight='bold')

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Alpha derivation
    ax = axes[1, 1]
    ax.axis('off')

    alpha_info = results['alpha_derivation']
    text = f"""
Vision Modality: Shafer Discounting

F1_macro = {alpha_info['f1_macro']:.4f}
F1_trivial = {alpha_info['f1_trivial']:.4f}

Skill = F1_macro - F1_trivial
      = {alpha_info['skill']:.4f}

alpha_V = 1 - skill / (1 - F1_trivial)
        = 1 - {alpha_info['skill']:.4f} / {1 - alpha_info['f1_trivial']:.4f}
        = {alpha_info['alpha']:.4f}

Interpretation:
{alpha_info['alpha']*100:.1f}% of evidence mass is assigned to Theta (ignorance)
Only {(1-alpha_info['alpha'])*100:.1f}% of original mass is retained
"""

    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12,
           fontfamily='monospace', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#343a40', linewidth=2))

    plt.suptitle('Vision Model Performance on Experimental Coupons',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "vision_performance_summary.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'vision_performance_summary.png'}")
    plt.close()


def create_bpa_by_condition():
    """Show average BPA by ground truth condition."""
    bpas = load_bpas()
    labels = load_labels()

    # Group by condition
    by_condition = {}
    for bpa in bpas:
        cell_id = bpa['cell_id']
        label = labels.get(cell_id, {})
        condition = label.get('condition', 'Unknown')
        if condition not in by_condition:
            by_condition[condition] = []
        by_condition[condition].append(bpa)

    # Compute averages
    conditions = sorted(by_condition.keys())
    focals = ['m_{P}', 'm_{N,U}', 'm_{N,U,L}', 'm_Theta']
    focal_labels = ['{P}', '{N,U}', '{N,U,L}', 'Theta']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(conditions))
    width = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']

    for i, (focal, label, color) in enumerate(zip(focals, focal_labels, colors)):
        means = []
        stds = []
        for cond in conditions:
            masses = [b[focal] for b in by_condition[cond]]
            means.append(np.mean(masses))
            stds.append(np.std(masses))

        bars = ax.bar(x + i * width, means, width, yerr=stds, label=f'm({label})',
                     color=color, alpha=0.8, capsize=3)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel('Mean BPA Mass', fontsize=12)
    ax.set_xlabel('Ground Truth Condition', fontsize=12)
    ax.set_title('Vision BPA Distribution by Ground Truth Condition\n(Error bars: ±1 std)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "bpa_by_condition.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'bpa_by_condition.png'}")
    plt.close()


def create_sample_classification_figure():
    """Show example cells with their images and BPA assignments."""
    bpas = load_bpas()
    labels = load_labels()

    # Select representative examples: one from each condition
    conditions_to_show = ['Porosity', 'Sound', 'Undercut', 'LOF']
    examples = {}

    for bpa in bpas:
        cell_id = bpa['cell_id']
        label = labels.get(cell_id, {})
        condition = label.get('condition', 'Unknown')

        if condition in conditions_to_show and condition not in examples:
            # Find one with relatively clear prediction
            max_focal = max(bpa['m_{P}'], bpa['m_{N,U}'], bpa['m_{N,U,L}'])
            if max_focal > 0.02:  # Some signal beyond just Theta
                examples[condition] = bpa

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # First row: show the BPA stacked bars
    # Second row: show interpretation text

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
    focal_names = ['{P}', '{N,U}', '{N,U,L}', 'Theta']

    for col, cond in enumerate(conditions_to_show):
        bpa = examples.get(cond)
        if bpa is None:
            continue

        cell_id = bpa['cell_id']
        label = labels.get(cell_id, {})
        true_class = label.get('true_class', '?')

        # Top row: BPA bar
        ax = axes[0, col]
        masses = [bpa['m_{P}'], bpa['m_{N,U}'], bpa['m_{N,U,L}'], bpa['m_Theta']]

        bottom = 0
        for i, (m, color, fname) in enumerate(zip(masses, colors, focal_names)):
            ax.bar(0, m, bottom=bottom, color=color, width=0.6, alpha=0.8)
            if m > 0.05:
                ax.text(0, bottom + m/2, f'{m:.2f}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white' if m > 0.3 else 'black')
            bottom += m

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([])
        ax.set_ylabel('BPA Mass' if col == 0 else '', fontsize=11)
        ax.set_title(f'{cond}\n({cell_id}, GT: {true_class})', fontsize=12, fontweight='bold')

        # Bottom row: interpretation
        ax = axes[1, col]
        ax.axis('off')

        # Find dominant focal element
        focal_masses = [('m({P})', bpa['m_{P}']),
                       ('m({N,U})', bpa['m_{N,U}']),
                       ('m({N,U,L})', bpa['m_{N,U,L}'])]
        dominant = max(focal_masses, key=lambda x: x[1])

        probs = bpa['probs']
        text = f"""Softmax Probabilities:
V1 (P): {probs[0]:.3f}
V3 (N,U): {probs[1]:.3f}
V4 (N,U,L): {probs[2]:.3f}

BPA after α={bpa['alpha_V']:.2f} discount:
{dominant[0]} = {dominant[1]:.3f}
m(Theta) = {bpa['m_Theta']:.3f}

Interpretation:
{get_interpretation(bpa, true_class)}"""

        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=9,
               fontfamily='monospace', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    # Legend
    legend_elements = [Patch(facecolor=c, label=f'm({n})') for c, n in zip(colors, focal_names)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.02), fontsize=11)

    plt.suptitle('Example Cell Classifications with BPA Assignments',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "sample_classifications.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sample_classifications.png'}")
    plt.close()


def get_interpretation(bpa, true_class):
    """Generate interpretation text for a BPA."""
    probs = bpa['probs']
    max_prob_idx = np.argmax(probs)
    heads = ['V1', 'V3', 'V4']
    pred_head = heads[max_prob_idx]

    if bpa['m_Theta'] > 0.85:
        return "High ignorance (alpha > 0.85).\nVision provides weak evidence."
    elif pred_head == 'V1':
        return "Vision detects porosity.\nMass on {P}."
    elif pred_head == 'V3':
        return "Vision detects geometry issue.\nMass on {N,U}."
    else:
        return "No conclusive surface evidence.\nMass on {N,U,L}."


def main():
    print("=" * 60)
    print("VISION COUPON BPA VISUALIZATION")
    print("=" * 60)

    print("\n1. Creating BPA heatmap for all coupons...")
    create_bpa_heatmap()

    print("\n2. Creating performance summary figure...")
    create_performance_summary()

    print("\n3. Creating BPA by condition figure...")
    create_bpa_by_condition()

    print("\n4. Creating sample classification figure...")
    create_sample_classification_figure()

    print("\n" + "=" * 60)
    print("All figures saved to results/vision/thesis_figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
