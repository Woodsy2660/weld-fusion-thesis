#!/usr/bin/env python3
"""
Visualize vision model classifications and BPA assignments.

Shows sample images from each focal element head, the model's
predicted probabilities, and how those translate to BPA masses.

Focal elements (4-class experimental frame):
    V1: {P}       Surface-breaking porosity
    V3: {N,U}     Surface geometric anomaly (magnitude unknown)
    V4: {N,U,L}   No conclusive surface evidence
"""

import sys
from pathlib import Path
import random
import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Focal element definitions
HEAD_TO_FOCAL = {
    'V1': '{P}',
    'V3': '{N,U}',
    'V4': '{N,U,L}',
}

HEAD_DESCRIPTIONS = {
    'V1': 'Surface-breaking porosity',
    'V3': 'Geometric anomaly\n(magnitude unmeasured)',
    'V4': 'No conclusive evidence\n(includes LOF possibility)',
}

HEAD_COLORS = {
    'V1': '#3498db',  # Blue for porosity
    'V3': '#e74c3c',  # Red for geometric
    'V4': '#2ecc71',  # Green for no defect/ignorance
}


def load_manifest():
    """Load the vision patches manifest."""
    manifest_path = REPO_ROOT / "data" / "processed" / "vision_patches" / "manifest.csv"
    patches = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            patches.append(row)
    return patches


def load_image(patch_path):
    """Load a patch image."""
    full_path = REPO_ROOT / "data" / "processed" / "vision_patches" / patch_path
    if full_path.exists():
        return np.array(Image.open(full_path))
    return None


def softmax(logits):
    """Apply softmax to logits."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def compute_bpa(probs, uncertainty_floor=0.05):
    """
    Compute BPA from softmax probabilities.

    Maps:
        V1 prob -> m({P})
        V3 prob -> m({N,U})
        V4 prob -> m({N,U,L})
        uncertainty_floor -> m(Theta)
    """
    scale = 1.0 - uncertainty_floor
    return {
        '{P}': probs[0] * scale,
        '{N,U}': probs[1] * scale,
        '{N,U,L}': probs[2] * scale,
        'Theta': uncertainty_floor,
    }


def create_sample_grid(n_samples=3, seed=42):
    """
    Create a grid showing sample images from each head with predictions.
    """
    random.seed(seed)
    manifest = load_manifest()

    # Filter by head and split
    by_head = {'V1': [], 'V3': [], 'V4': []}
    for patch in manifest:
        head = patch['head']
        if head in by_head:
            by_head[head].append(patch)

    # Sample from each head
    samples = {}
    for head, patches in by_head.items():
        test_patches = [p for p in patches if p['split'] == 'test']
        if len(test_patches) >= n_samples:
            samples[head] = random.sample(test_patches, n_samples)
        elif test_patches:
            samples[head] = test_patches[:n_samples]
        else:
            samples[head] = patches[:n_samples]

    # Create figure
    n_heads = 3
    fig, axes = plt.subplots(n_heads, n_samples + 1, figsize=(4 * (n_samples + 1), 4 * n_heads))

    heads = ['V1', 'V3', 'V4']

    for row, head in enumerate(heads):
        # First column: head info
        ax = axes[row, 0]
        ax.axis('off')
        ax.set_facecolor(HEAD_COLORS[head])

        info_text = f"{head}\n{HEAD_TO_FOCAL[head]}\n\n{HEAD_DESCRIPTIONS[head]}"
        ax.text(0.5, 0.5, info_text, ha='center', va='center',
               fontsize=14, fontweight='bold', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Sample columns
        for col, patch in enumerate(samples.get(head, [])):
            ax = axes[row, col + 1]

            img = load_image(patch['patch_path'])
            if img is not None:
                ax.imshow(img)
                ax.axis('off')

                # Title with source info
                source = patch['source_label']
                if len(source) > 20:
                    source = source[:17] + '...'
                ax.set_title(f"GT: {source}", fontsize=10)

                # Add border color based on head
                for spine in ax.spines.values():
                    spine.set_edgecolor(HEAD_COLORS[head])
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            else:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
                ax.axis('off')

    plt.suptitle('Vision Model Training Data: Samples by Focal Element Head',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "sample_grid_by_head.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'sample_grid_by_head.png'}")
    plt.close()


def create_bpa_visualization():
    """
    Create visualization showing how model outputs map to BPA masses.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Example scenarios
    scenarios = [
        {
            'name': 'High Porosity Confidence',
            'logits': np.array([5.0, -2.0, -1.0]),
            'description': 'Model strongly predicts V1 (porosity)',
        },
        {
            'name': 'High Geometric Anomaly',
            'logits': np.array([-1.0, 6.0, 0.0]),
            'description': 'Model strongly predicts V3 (geometric)',
        },
        {
            'name': 'No Conclusive Evidence',
            'logits': np.array([-2.0, -1.0, 5.0]),
            'description': 'Model strongly predicts V4 (ignorance)',
        },
        {
            'name': 'Mixed: Porosity vs Geometric',
            'logits': np.array([3.0, 2.5, -1.0]),
            'description': 'Model uncertain between V1 and V3',
        },
        {
            'name': 'Mixed: All Uncertain',
            'logits': np.array([1.0, 0.8, 1.2]),
            'description': 'Model uncertain across all heads',
        },
        {
            'name': 'Geometric vs Ignorance',
            'logits': np.array([-2.0, 2.0, 2.5]),
            'description': 'Model uncertain between V3 and V4',
        },
    ]

    head_names = ['V1', 'V3', 'V4']
    focal_names = ['{P}', '{N,U}', '{N,U,L}', 'Theta']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // 3, idx % 3]

        logits = scenario['logits']
        probs = softmax(logits)
        bpa = compute_bpa(probs)

        # Create grouped bar chart
        x = np.arange(2)
        width = 0.35

        # Model outputs (softmax probabilities)
        bars1 = ax.bar(x[0] - 0.2, [1], width, label='Softmax', color='lightgray', alpha=0)

        # Create stacked bars for softmax
        bottom = 0
        for i, (head, p) in enumerate(zip(head_names, probs)):
            ax.bar(x[0], p, width, bottom=bottom, color=colors[i],
                   label=f'{head}: {p:.2f}', alpha=0.8)
            bottom += p

        # BPA masses
        bottom = 0
        focal_masses = [bpa['{P}'], bpa['{N,U}'], bpa['{N,U,L}'], bpa['Theta']]
        for i, (focal, m) in enumerate(zip(focal_names, focal_masses)):
            ax.bar(x[1], m, width, bottom=bottom, color=colors[i],
                   label=f'm({focal}): {m:.2f}', alpha=0.8)
            bottom += m

        ax.set_xticks(x)
        ax.set_xticklabels(['Softmax\nProbabilities', 'BPA\nMasses'], fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Mass', fontsize=10)
        ax.set_title(f"{scenario['name']}\n{scenario['description']}", fontsize=11, fontweight='bold')

        # Add annotations
        ax.axhline(1.0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    # Create legend separately
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='V1 → {P}'),
        Patch(facecolor=colors[1], label='V3 → {N,U}'),
        Patch(facecolor=colors[2], label='V4 → {N,U,L}'),
        Patch(facecolor=colors[3], label='Theta (uncertainty)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 0.02), fontsize=11)

    plt.suptitle('Vision Model: Softmax Probabilities → BPA Masses\n(uncertainty floor = 0.05)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "bpa_construction_examples.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'bpa_construction_examples.png'}")
    plt.close()


def create_head_distribution_figure():
    """
    Create figure showing the distribution of patches across heads.
    """
    manifest = load_manifest()

    # Count by head and split
    counts = {head: {'train': 0, 'val': 0, 'test': 0} for head in ['V1', 'V3', 'V4']}

    for patch in manifest:
        head = patch['head']
        split = patch['split']
        if head in counts and split in counts[head]:
            counts[head][split] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of total counts
    ax = axes[0]
    heads = ['V1', 'V3', 'V4']
    totals = [sum(counts[h].values()) for h in heads]
    colors_list = [HEAD_COLORS[h] for h in heads]

    bars = ax.bar(heads, totals, color=colors_list, alpha=0.8, edgecolor='black')

    # Add count labels
    for bar, total in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
               f'{total:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Patches', fontsize=12)
    ax.set_xlabel('Focal Element Head', fontsize=12)
    ax.set_title('Patch Distribution by Head', fontsize=14, fontweight='bold')

    # Add focal element labels
    for i, head in enumerate(heads):
        ax.text(i, -max(totals)*0.08, HEAD_TO_FOCAL[head], ha='center',
               fontsize=11, style='italic')

    ax.set_ylim(0, max(totals) * 1.15)

    # Stacked bar by split
    ax = axes[1]
    width = 0.6
    x = np.arange(len(heads))

    train_counts = [counts[h]['train'] for h in heads]
    val_counts = [counts[h]['val'] for h in heads]
    test_counts = [counts[h]['test'] for h in heads]

    ax.bar(x, train_counts, width, label='Train', color='#3498db', alpha=0.8)
    ax.bar(x, val_counts, width, bottom=train_counts, label='Val', color='#f39c12', alpha=0.8)
    ax.bar(x, test_counts, width, bottom=np.array(train_counts) + np.array(val_counts),
           label='Test', color='#2ecc71', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(heads, fontsize=12)
    ax.set_ylabel('Number of Patches', fontsize=12)
    ax.set_xlabel('Focal Element Head', fontsize=12)
    ax.set_title('Patch Distribution by Head and Split', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "head_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'head_distribution.png'}")
    plt.close()


def create_focal_element_diagram():
    """
    Create a diagram showing the vision focal element structure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'Vision Modality: Focal Element Structure',
           ha='center', fontsize=16, fontweight='bold')

    # Theta (frame of discernment)
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # Theta box
    theta_box = FancyBboxPatch((1, 5.5), 10, 3, boxstyle="round,pad=0.1",
                                facecolor='#ecf0f1', edgecolor='black', linewidth=2)
    ax.add_patch(theta_box)
    ax.text(6, 8, 'Theta = {N, U, P, L}', ha='center', fontsize=12, fontweight='bold')

    # V1: {P}
    v1_box = FancyBboxPatch((1.5, 6), 2, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#3498db', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(v1_box)
    ax.text(2.5, 7.1, 'V1', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(2.5, 6.5, '{P}', ha='center', fontsize=11, color='white')

    # V3: {N,U}
    v3_box = FancyBboxPatch((4.5, 6), 3, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#e74c3c', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(v3_box)
    ax.text(6, 7.1, 'V3', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(6, 6.5, '{N, U}', ha='center', fontsize=11, color='white')

    # V4: {N,U,L}
    v4_box = FancyBboxPatch((8, 6), 2.5, 1.5, boxstyle="round,pad=0.1",
                            facecolor='#2ecc71', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(v4_box)
    ax.text(9.25, 7.1, 'V4', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(9.25, 6.5, '{N, U, L}', ha='center', fontsize=11, color='white')

    # Descriptions below
    desc_y = 4.5
    ax.text(2.5, desc_y, 'Surface porosity\ndetected', ha='center', fontsize=10)
    ax.text(6, desc_y, 'Geometric anomaly\n(magnitude unknown)', ha='center', fontsize=10)
    ax.text(9.25, desc_y, 'No conclusive\nevidence', ha='center', fontsize=10)

    # BPA construction note
    ax.text(6, 2.5, 'BPA Construction:', ha='center', fontsize=12, fontweight='bold')
    ax.text(6, 1.8, 'm(focal) = softmax_prob × (1 - uncertainty_floor)', ha='center', fontsize=11)
    ax.text(6, 1.2, 'm(Theta) = uncertainty_floor = 0.05', ha='center', fontsize=11)

    # Key constraints
    ax.text(6, 0.5, 'Constraints: Vision NEVER predicts {U} or {L} alone (subsurface, unmeasurable)',
           ha='center', fontsize=10, style='italic', color='#7f8c8d')

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "focal_element_diagram.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'focal_element_diagram.png'}")
    plt.close()


def create_classification_examples():
    """
    Show real examples of correctly and incorrectly classified patches.
    """
    manifest = load_manifest()

    # Sample from each head - test set only
    random.seed(42)
    samples = {head: [] for head in ['V1', 'V3', 'V4']}

    for patch in manifest:
        if patch['split'] == 'test' and patch['head'] in samples:
            samples[patch['head']].append(patch)

    # Take 5 samples per head
    for head in samples:
        if len(samples[head]) > 5:
            samples[head] = random.sample(samples[head], 5)

    # Create grid
    n_per_head = 5
    fig, axes = plt.subplots(3, n_per_head, figsize=(3 * n_per_head, 10))

    heads = ['V1', 'V3', 'V4']

    for row, head in enumerate(heads):
        for col, patch in enumerate(samples[head][:n_per_head]):
            ax = axes[row, col]

            img = load_image(patch['patch_path'])
            if img is not None:
                ax.imshow(img)

            ax.axis('off')

            # Source label
            source = patch['source_label']
            if len(source) > 15:
                source = source[:12] + '...'

            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(HEAD_COLORS[head])
                spine.set_linewidth(4)
                spine.set_visible(True)

            if col == 0:
                ax.set_ylabel(f"{head}\n{HEAD_TO_FOCAL[head]}", fontsize=12,
                             fontweight='bold', rotation=0, labelpad=60)

            ax.set_title(source, fontsize=9)

    plt.suptitle('Vision Test Set: Sample Patches by Focal Element',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "test_samples_by_head.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'test_samples_by_head.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("VISION BPA VISUALIZATION")
    print("=" * 60)

    print("\n1. Creating head distribution figure...")
    create_head_distribution_figure()

    print("\n2. Creating focal element diagram...")
    create_focal_element_diagram()

    print("\n3. Creating BPA construction examples...")
    create_bpa_visualization()

    print("\n4. Creating sample grid by head...")
    create_sample_grid(n_samples=4)

    print("\n5. Creating test samples by head...")
    create_classification_examples()

    print("\n" + "=" * 60)
    print("All figures saved to results/vision/thesis_figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
