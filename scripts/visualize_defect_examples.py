#!/usr/bin/env python3
"""
Visualize example defect classifications for each defect type studied in thesis.

Shows actual weld images from experimental coupons for:
- Porosity
- Sound (no defect)
- Undercut
- LOF (Lack of Fusion)
- Overfill
- Misalignment
"""

import sys
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Representative coupons for each condition
CONDITION_COUPONS = {
    'Porosity': ['C01', 'C02'],
    'Sound': ['C03', 'C04'],
    'Undercut': ['C05', 'C06'],
    'LOF': ['C07', 'C08'],
    'Overfill': ['C09'],
    'Misaligned': ['C13'],
}

# Descriptions for each condition
CONDITION_DESCRIPTIONS = {
    'Porosity': 'Gas pores trapped in weld metal.\nSurface-breaking porosity visible as small holes.',
    'Sound': 'Acceptable weld with no significant defects.\nSmooth bead profile, good fusion.',
    'Undercut': 'Groove melted into base metal at weld toe.\nDepression along edge of weld bead.',
    'LOF': 'Lack of Fusion - incomplete bonding.\nOften subsurface, may show at weld toes.',
    'Overfill': 'Excess weld reinforcement.\nBead height exceeds ISO 5817 limit.',
    'Misaligned': 'Linear misalignment between plates.\nHeight difference between plate surfaces.',
}

# Frame of discernment mapping
CONDITION_TO_CLASS = {
    'Porosity': 'P',
    'Sound': 'N',
    'Undercut': 'U',
    'LOF': 'L',
    'Overfill': 'U',  # Overfill is geometric non-conformance
    'Misaligned': 'U',  # Misalignment is geometric non-conformance
}

# Colors for each condition
CONDITION_COLORS = {
    'Porosity': '#3498db',
    'Sound': '#2ecc71',
    'Undercut': '#e74c3c',
    'LOF': '#9b59b6',
    'Overfill': '#f39c12',
    'Misaligned': '#1abc9c',
}


def load_coupon_image(coupon_id, face='top', frame_idx=0):
    """Load a color image from a coupon."""
    coupon_dir = REPO_ROOT / "data" / "raw" / "S1" / f"{coupon_id}_{face}_r0"
    color_dir = coupon_dir / "color"

    if not color_dir.exists():
        return None

    # Get image files
    img_files = sorted(color_dir.glob("*.png"))
    if frame_idx < len(img_files):
        return np.array(Image.open(img_files[frame_idx]))
    return None


def create_defect_examples_grid():
    """Create a grid showing one example of each defect type."""
    conditions = ['Porosity', 'Sound', 'Undercut', 'LOF', 'Overfill', 'Misaligned']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, condition in enumerate(conditions):
        ax = axes[idx]

        # Get representative coupon
        coupons = CONDITION_COUPONS[condition]
        coupon_id = coupons[0]

        # Load image
        img = load_coupon_image(coupon_id)

        if img is not None:
            ax.imshow(img)

            # Add colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(CONDITION_COLORS[condition])
                spine.set_linewidth(5)
                spine.set_visible(True)
        else:
            ax.text(0.5, 0.5, f'Image not found\n{coupon_id}',
                   ha='center', va='center', fontsize=12)

        ax.axis('off')

        # Title with condition and class
        true_class = CONDITION_TO_CLASS[condition]
        ax.set_title(f'{condition} ({coupon_id})\nClass: {true_class}',
                    fontsize=14, fontweight='bold',
                    color=CONDITION_COLORS[condition])

    plt.suptitle('Defect Types Studied in Thesis\n(Representative Coupon Images)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "defect_examples_grid.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'defect_examples_grid.png'}")
    plt.close()


def create_defect_examples_detailed():
    """Create detailed figure with multiple views for each condition."""
    conditions = ['Porosity', 'Sound', 'Undercut', 'LOF', 'Overfill', 'Misaligned']

    fig = plt.figure(figsize=(20, 24))

    for row, condition in enumerate(conditions):
        coupons = CONDITION_COUPONS[condition]
        coupon_id = coupons[0]

        # Left: Full weld image
        ax1 = fig.add_subplot(6, 3, row * 3 + 1)
        img = load_coupon_image(coupon_id, frame_idx=5)  # Middle frame
        if img is not None:
            ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f'{condition}\n{coupon_id} - Top View', fontsize=12, fontweight='bold',
                     color=CONDITION_COLORS[condition])

        # Add colored border
        for spine in ax1.spines.values():
            spine.set_edgecolor(CONDITION_COLORS[condition])
            spine.set_linewidth(4)
            spine.set_visible(True)

        # Middle: Root view if available
        ax2 = fig.add_subplot(6, 3, row * 3 + 2)
        img_root = load_coupon_image(coupon_id, face='root', frame_idx=5)
        if img_root is not None:
            ax2.imshow(img_root)
            ax2.set_title(f'{coupon_id} - Root View', fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'Root view\nnot available', ha='center', va='center')
        ax2.axis('off')

        for spine in ax2.spines.values():
            spine.set_edgecolor(CONDITION_COLORS[condition])
            spine.set_linewidth(4)
            spine.set_visible(True)

        # Right: Description
        ax3 = fig.add_subplot(6, 3, row * 3 + 3)
        ax3.axis('off')

        true_class = CONDITION_TO_CLASS[condition]
        desc = CONDITION_DESCRIPTIONS[condition]

        text = f"""Condition: {condition}
True Class: {true_class}

{desc}

Coupons: {', '.join(coupons)}
"""
        ax3.text(0.1, 0.5, text, ha='left', va='center', fontsize=11,
                transform=ax3.transAxes, fontfamily='sans-serif',
                bbox=dict(boxstyle='round', facecolor=CONDITION_COLORS[condition],
                         alpha=0.2, edgecolor=CONDITION_COLORS[condition]))

    plt.suptitle('Weld Defect Types: Visual Examples from Experimental Coupons',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "defect_examples_detailed.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'defect_examples_detailed.png'}")
    plt.close()


def create_frame_of_discernment_figure():
    """Create figure showing the frame of discernment and class mapping."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(7, 11.5, 'Frame of Discernment: Θ = {N, U, P, L}',
           ha='center', fontsize=16, fontweight='bold')

    # Draw the four classes
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    classes = [
        ('N', 'No Defect\n(Sound)', '#2ecc71', 1.5),
        ('U', 'Geometric\nNon-conformance', '#e74c3c', 4.5),
        ('P', 'Porosity', '#3498db', 7.5),
        ('L', 'Lack of\nFusion', '#9b59b6', 10.5),
    ]

    for cls, desc, color, x in classes:
        box = FancyBboxPatch((x, 7), 2.5, 2.5, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(box)
        ax.text(x + 1.25, 8.7, cls, ha='center', va='center', fontsize=20,
               fontweight='bold', color='white')
        ax.text(x + 1.25, 7.8, desc, ha='center', va='center', fontsize=9,
               color='white')

    # Condition to class mappings
    ax.text(7, 5.5, 'Condition → Class Mapping', ha='center', fontsize=14, fontweight='bold')

    mappings = [
        ('Sound', 'N', '#2ecc71'),
        ('Undercut', 'U', '#e74c3c'),
        ('Overfill', 'U', '#e74c3c'),
        ('Misaligned', 'U', '#e74c3c'),
        ('Porosity', 'P', '#3498db'),
        ('LOF', 'L', '#9b59b6'),
    ]

    y_start = 4.5
    for i, (cond, cls, color) in enumerate(mappings):
        col = i % 3
        row = i // 3
        x = 2 + col * 4
        y = y_start - row * 1.5

        ax.text(x, y, f'{cond} → {cls}', ha='left', fontsize=12,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Note about U class
    ax.text(7, 1.5,
           'Note: Undercut, Overfill, and Misalignment all map to class U\n'
           '(geometric non-conformance to ISO 5817 Level B)',
           ha='center', fontsize=11, style='italic',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "frame_of_discernment.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'frame_of_discernment.png'}")
    plt.close()


def create_all_coupons_overview():
    """Create overview showing all 15 coupons organized by condition."""
    import csv

    # Load labels
    labels = {}
    with open(REPO_ROOT / "results" / "thesis" / "data" / "labels_v2_2026-10-26.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['coupon_id']
            if cid not in labels:
                labels[cid] = row['condition']

    # Group by condition
    by_condition = {}
    for cid, cond in labels.items():
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(cid)

    # Create figure
    fig = plt.figure(figsize=(20, 16))

    conditions = ['Sound', 'Porosity', 'Undercut', 'LOF', 'Misaligned']

    subplot_idx = 1
    for cond in conditions:
        coupons = sorted(by_condition.get(cond, []))

        for coupon_id in coupons:
            ax = fig.add_subplot(5, 5, subplot_idx)

            img = load_coupon_image(coupon_id, frame_idx=5)
            if img is not None:
                ax.imshow(img)

            ax.axis('off')

            true_class = CONDITION_TO_CLASS.get(cond, '?')
            ax.set_title(f'{coupon_id}\n{cond} ({true_class})', fontsize=10, fontweight='bold',
                        color=CONDITION_COLORS.get(cond, 'black'))

            # Colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(CONDITION_COLORS.get(cond, 'gray'))
                spine.set_linewidth(3)
                spine.set_visible(True)

            subplot_idx += 1

    # Fill remaining with empty
    while subplot_idx <= 25:
        ax = fig.add_subplot(5, 5, subplot_idx)
        ax.axis('off')
        subplot_idx += 1

    plt.suptitle('All 15 Experimental Coupons by Condition',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "all_coupons_overview.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'all_coupons_overview.png'}")
    plt.close()


def create_condition_comparison():
    """Create side-by-side comparison of different conditions."""
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    conditions = ['Porosity', 'Sound', 'Undercut', 'LOF', 'Overfill', 'Misaligned']

    for col, condition in enumerate(conditions):
        coupons = CONDITION_COUPONS[condition]
        coupon_id = coupons[0]

        # Top view
        ax = axes[0, col]
        img = load_coupon_image(coupon_id, face='top', frame_idx=5)
        if img is not None:
            ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{condition}\n({coupon_id})', fontsize=12, fontweight='bold',
                    color=CONDITION_COLORS[condition])

        for spine in ax.spines.values():
            spine.set_edgecolor(CONDITION_COLORS[condition])
            spine.set_linewidth(4)
            spine.set_visible(True)

        # Root view
        ax = axes[1, col]
        img = load_coupon_image(coupon_id, face='root', frame_idx=5)
        if img is not None:
            ax.imshow(img)
            ax.set_title('Root View', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        ax.axis('off')

        for spine in ax.spines.values():
            spine.set_edgecolor(CONDITION_COLORS[condition])
            spine.set_linewidth(4)
            spine.set_visible(True)

    # Row labels
    axes[0, 0].text(-0.15, 0.5, 'TOP', transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, va='center')
    axes[1, 0].text(-0.15, 0.5, 'ROOT', transform=axes[1, 0].transAxes,
                   fontsize=14, fontweight='bold', rotation=90, va='center')

    plt.suptitle('Defect Type Comparison: Top and Root Views',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_dir = REPO_ROOT / "results" / "vision" / "thesis_figures"
    fig.savefig(output_dir / "condition_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'condition_comparison.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("DEFECT EXAMPLES VISUALIZATION")
    print("=" * 60)

    print("\n1. Creating defect examples grid...")
    create_defect_examples_grid()

    print("\n2. Creating detailed defect examples...")
    create_defect_examples_detailed()

    print("\n3. Creating frame of discernment figure...")
    create_frame_of_discernment_figure()

    print("\n4. Creating all coupons overview...")
    create_all_coupons_overview()

    print("\n5. Creating condition comparison...")
    create_condition_comparison()

    print("\n" + "=" * 60)
    print("All figures saved to results/vision/thesis_figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
