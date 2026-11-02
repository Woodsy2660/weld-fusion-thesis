#!/usr/bin/env python3
"""
Generate F4.1 (cell examples grid) and F4.2 (UT readings scatter) for Chapter 4.

F4.1: 4×3 grid showing one cell per class (N, U, P, L) with:
    Column (a): Vision patch
    Column (b): Geometry exceedance probability (profile data not stored)
    Column (c): UT reading

F4.2: Scatter plot of all 150 UT readings (amplitude vs soundpath)

Generated: 2026-10-29
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.image import imread
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = PROJECT_ROOT / "results" / "thesis" / "ch4_evidence"
VISION_PATCHES = PROJECT_ROOT / "data" / "processed" / "coupon_cell_patches"

# Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
})

# Colors from spec
COLORS = {
    'vision': {'fill': '#d0e3fe', 'edge': '#4a73a8'},
    'geometry': {'fill': '#e2fae4', 'edge': '#5e8f6b'},
    'ultrasonic': {'fill': '#fbfdcd', 'edge': '#b59a2e'},
    'class': {
        'N': '#2ca02c',  # green
        'U': '#ff7f0e',  # orange
        'P': '#1f77b4',  # blue
        'L': '#d62728',  # red
    }
}

# Selected cells (first geometry-measured per class)
SELECTED_CELLS = {
    'N': 'C03-02',
    'U': 'C05-02',
    'P': 'C01-02',
    'L': 'C07-02',
}


def load_data():
    """Load all required data."""
    labels = pd.read_csv(PROJECT_ROOT / "data" / "labels_v2_2026-10-26.csv")
    geometry = pd.read_csv(PROJECT_ROOT / "results" / "geometry" / "summary.csv")
    ut = pd.read_csv(PROJECT_ROOT / "results" / "ut" / "features.csv")
    vision_manifest = pd.read_csv(VISION_PATCHES / "manifest.csv")
    return labels, geometry, ut, vision_manifest


def generate_f4_1(labels, geometry, ut, vision_manifest):
    """Generate F4.1: 4×3 cell examples grid."""
    fig, axes = plt.subplots(4, 3, figsize=(150/25.4, 120/25.4))  # 150mm × 120mm

    classes = ['N', 'U', 'P', 'L']
    col_titles = ['(a) Vision', '(b) Geometry', '(c) Ultrasonic']

    for row_idx, cls in enumerate(classes):
        cell_id = SELECTED_CELLS[cls]

        # Get data for this cell
        geom_row = geometry[geometry['cell_id'] == cell_id].iloc[0]
        ut_row = ut[ut['cell_id'] == cell_id].iloc[0]
        vision_row = vision_manifest[vision_manifest['cell_id'] == cell_id].iloc[0]

        # Column (a): Vision patch
        ax_vision = axes[row_idx, 0]
        patch_path = VISION_PATCHES / vision_row['patch_path']
        if patch_path.exists():
            img = imread(patch_path)
            ax_vision.imshow(img)
        ax_vision.set_xticks([])
        ax_vision.set_yticks([])
        ax_vision.set_ylabel(f'{cls}\n({cell_id})', rotation=0, ha='right', va='center',
                            fontweight='bold', color=COLORS['class'][cls])
        for spine in ax_vision.spines.values():
            spine.set_edgecolor(COLORS['vision']['edge'])
            spine.set_linewidth(1.5)

        # Column (b): Geometry exceedance
        ax_geom = axes[row_idx, 1]
        ax_geom.set_facecolor(COLORS['geometry']['fill'])
        for spine in ax_geom.spines.values():
            spine.set_edgecolor(COLORS['geometry']['edge'])
            spine.set_linewidth(1.5)

        exceedance = geom_row['exceedance_prob']
        governing = geom_row['governing_measurand']

        if pd.notna(exceedance) and not geom_row['is_vacuous']:
            # Draw exceedance bar
            bar_height = 0.4
            ax_geom.barh([0.5], [exceedance], height=bar_height,
                        color=COLORS['geometry']['edge'], alpha=0.7)
            ax_geom.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax_geom.set_xlim(0, 1.1)
            ax_geom.set_ylim(0, 1)
            ax_geom.set_xticks([0, 0.5, 1.0])
            ax_geom.set_xticklabels(['0', '0.5', '1.0'])
            ax_geom.set_yticks([])
            ax_geom.text(0.5, 0.85, f'P(exceed) = {exceedance:.3f}', ha='center',
                        va='center', fontsize=7, transform=ax_geom.transAxes)
            ax_geom.text(0.5, 0.15, f'({governing})', ha='center',
                        va='center', fontsize=6, style='italic', transform=ax_geom.transAxes)
        else:
            ax_geom.text(0.5, 0.5, 'Not measured', ha='center', va='center',
                        fontsize=8, style='italic', transform=ax_geom.transAxes)
            ax_geom.set_xticks([])
            ax_geom.set_yticks([])

        # Column (c): UT reading
        ax_ut = axes[row_idx, 2]
        ax_ut.set_facecolor(COLORS['ultrasonic']['fill'])
        for spine in ax_ut.spines.values():
            spine.set_edgecolor(COLORS['ultrasonic']['edge'])
            spine.set_linewidth(1.5)

        amplitude = ut_row['amplitude_pct']
        located = ut_row['located']
        soundpath = ut_row['soundpath_mm']

        # Draw amplitude bar
        bar_width = 0.4
        bar_color = COLORS['ultrasonic']['edge'] if located else '#888888'
        ax_ut.bar([0.5], [amplitude], width=bar_width, color=bar_color, alpha=0.7)
        ax_ut.axhline(y=20, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax_ut.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax_ut.set_xlim(0, 1)
        ax_ut.set_ylim(0, 120)
        ax_ut.set_xticks([])
        ax_ut.set_ylabel('% FSH', fontsize=7)
        ax_ut.set_yticks([0, 50, 100])

        status_txt = 'gated' if located else 'no gate'
        ax_ut.text(0.5, 0.92, f'{amplitude:.1f}%', ha='center', va='top',
                  fontsize=8, fontweight='bold', transform=ax_ut.transAxes)
        ax_ut.text(0.5, 0.05, f'({status_txt})', ha='center', va='bottom',
                  fontsize=6, style='italic', transform=ax_ut.transAxes)
        if located and soundpath > 0:
            ax_ut.text(0.5, 0.78, f'd={soundpath:.1f}mm', ha='center', va='top',
                      fontsize=6, transform=ax_ut.transAxes)

    # Column titles
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontweight='bold', pad=8)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / "f4_1_cell_examples.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(OUTPUT_DIR / "f4_1_cell_examples.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Generated: f4_1_cell_examples.pdf/.png")


def generate_f4_2(ut):
    """Generate F4.2: UT readings scatter plot."""
    fig, ax = plt.subplots(figsize=(112/25.4, 84/25.4))  # 112mm × 84mm

    # Separate by class
    for cls in ['N', 'U', 'P', 'L']:
        cls_data = ut[ut['true_class'] == cls]
        located = cls_data[cls_data['located'] == 1]
        not_located = cls_data[cls_data['located'] == 0]

        # Gated indications: filled markers
        if len(located) > 0:
            ax.scatter(located['soundpath_mm'], located['amplitude_pct'],
                      c=COLORS['class'][cls], marker='o', s=30, alpha=0.8,
                      label=f'{cls} (gated)', edgecolors='white', linewidths=0.5)

        # Not gated: open markers at x=0
        if len(not_located) > 0:
            # Jitter x slightly for visibility
            x_jitter = np.random.uniform(-0.5, 0.5, len(not_located))
            ax.scatter(x_jitter, not_located['amplitude_pct'],
                      c='none', marker='o', s=30, alpha=0.6,
                      edgecolors=COLORS['class'][cls], linewidths=1.5)

    # Reference lines
    ax.axhline(y=20, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Gate threshold (20%)')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Saturation (100%)')

    ax.set_xlabel('Sound path (mm)')
    ax.set_ylabel('Amplitude (% FSH)')
    ax.set_xlim(-2, 45)
    ax.set_ylim(0, 130)

    # Custom legend
    handles = []
    for cls in ['N', 'U', 'P', 'L']:
        handles.append(mpatches.Patch(color=COLORS['class'][cls], label=cls))
    handles.append(plt.Line2D([0], [0], color='gray', linestyle=':', label='Gate (20%)'))
    handles.append(plt.Line2D([0], [0], color='red', linestyle='--', label='Saturation'))

    ax.legend(handles=handles, loc='upper right', framealpha=0.9, ncol=2)

    # Annotation for non-gated readings
    ax.annotate('No gate\n(x=0)', xy=(0, 8), xytext=(5, 5),
               fontsize=7, ha='left', va='bottom',
               arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / "f4_2_ut_readings.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(OUTPUT_DIR / "f4_2_ut_readings.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Generated: f4_2_ut_readings.pdf/.png")


def update_numbers_json():
    """Update s4_1_numbers.json with selected cell IDs."""
    numbers_path = OUTPUT_DIR / "s4_1_numbers.json"
    with open(numbers_path, 'r') as f:
        numbers = json.load(f)

    numbers['f4_1_selected_cells'] = {
        'selection_rule': 'first geometry-measured cell in coupon then cell order',
        'cells': SELECTED_CELLS,
        'cell_details': {}
    }

    # Load data for details
    geometry = pd.read_csv(PROJECT_ROOT / "results" / "geometry" / "summary.csv")
    ut = pd.read_csv(PROJECT_ROOT / "results" / "ut" / "features.csv")

    for cls, cell_id in SELECTED_CELLS.items():
        geom_row = geometry[geometry['cell_id'] == cell_id].iloc[0]
        ut_row = ut[ut['cell_id'] == cell_id].iloc[0]

        numbers['f4_1_selected_cells']['cell_details'][cls] = {
            'cell_id': cell_id,
            'geometry': {
                'exceedance_prob': float(geom_row['exceedance_prob']) if pd.notna(geom_row['exceedance_prob']) else None,
                'governing_measurand': geom_row['governing_measurand'],
                'is_vacuous': bool(geom_row['is_vacuous'])
            },
            'ut': {
                'amplitude_pct': float(ut_row['amplitude_pct']),
                'located': bool(ut_row['located']),
                'soundpath_mm': float(ut_row['soundpath_mm']) if ut_row['located'] else None
            }
        }

    with open(numbers_path, 'w') as f:
        json.dump(numbers, f, indent=2)
    print("Updated: s4_1_numbers.json")


def main():
    print("Loading data...")
    labels, geometry, ut, vision_manifest = load_data()

    print("\nGenerating F4.1: Cell examples grid...")
    generate_f4_1(labels, geometry, ut, vision_manifest)

    print("\nGenerating F4.2: UT readings scatter...")
    generate_f4_2(ut)

    print("\nUpdating s4_1_numbers.json...")
    update_numbers_json()

    print("\nDone!")


if __name__ == "__main__":
    main()
