#!/usr/bin/env python3
"""
Generate F4.1: One cell per class through all three modalities.

Shows four rows (N, U, P, L) with three columns:
  (a) Vision patch
  (b) Geometry transverse profile (schematic with measurements)
  (c) UT reading (amplitude bar with gate reference)

Selected cells (first geometry-measured cell per class in coupon order):
  N: C03-02
  U: C05-02
  P: C01-02
  L: C07-02
"""

import json
import csv
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image

# Paths
REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "results" / "thesis" / "ch4_evidence"
PATCHES_DIR = REPO_ROOT / "data" / "processed" / "coupon_cell_patches"
GEOMETRY_SUMMARY_PATH = REPO_ROOT / "results" / "geometry" / "summary.csv"
UT_FEATURES_PATH = REPO_ROOT / "results" / "ut" / "features.csv"
S4_1_NUMBERS_PATH = OUTPUT_DIR / "s4_1_numbers.json"

# Selected cells
SELECTED_CELLS = {
    "N": "C03-02",
    "U": "C05-02",
    "P": "C01-02",
    "L": "C07-02",
}

# Class display order and colors
CLASS_ORDER = ["N", "U", "P", "L"]
CLASS_COLORS = {
    "N": "#4CAF50",  # Green
    "U": "#FF9800",  # Orange
    "P": "#2196F3",  # Blue
    "L": "#F44336",  # Red
}


def load_geometry_summary():
    """Load geometry summary data."""
    geom = {}
    with open(GEOMETRY_SUMMARY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            geom[row["cell_id"]] = row
    return geom


def load_ut_features():
    """Load UT features."""
    ut = {}
    with open(UT_FEATURES_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ut[row["cell_id"]] = row
    return ut


def load_s4_1_numbers():
    """Load s4_1_numbers.json for cell details."""
    with open(S4_1_NUMBERS_PATH) as f:
        return json.load(f)


def get_patch_path(cell_id: str) -> Path:
    """Get the path to a cell's vision patch."""
    # Parse cell_id
    coupon = cell_id.split("-")[0]
    cell_idx = int(cell_id.split("-")[1])

    # Check subdirectories
    for subdir in ["P", "N_U", "N_U_L"]:
        patch_path = PATCHES_DIR / subdir / f"{coupon}_{cell_idx:02d}.png"
        if patch_path.exists():
            return patch_path

    raise FileNotFoundError(f"Patch not found for {cell_id}")


def draw_geometry_panel(ax, cell_id: str, geom_data: dict, cls: str):
    """
    Draw a schematic geometry transverse profile.

    Shows:
    - Profile shape (symmetric for simplicity)
    - Misalignment measurement
    - Reference lines for toes
    - Exceedance probability annotation
    """
    # Get measurements
    exceedance = float(geom_data.get("exceedance_prob", "nan"))
    governing = geom_data.get("governing_measurand", "unknown")
    is_vacuous = geom_data.get("is_vacuous", "True") == "True"
    misalignment = float(geom_data.get("governing_undercut_mm", "nan")) if not is_vacuous else float("nan")

    ax.set_xlim(-15, 15)
    ax.set_ylim(-6, 4)
    ax.set_aspect("equal")

    if is_vacuous:
        # Show "No measurement" message
        ax.text(0, -1, "No geometry\nmeasurement", ha='center', va='center',
                fontsize=9, style='italic', color='#666666')
        ax.set_xlabel("Position (mm)", fontsize=8)
        ax.set_ylabel("Height (mm)", fontsize=8)
        ax.axhline(y=0, color='#cccccc', linestyle='-', linewidth=0.8)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-4, -2, 0, 2])
        return

    # Draw schematic profile
    # Parent metal (two plates with possible misalignment)
    y_left = 0  # Reference
    y_right = -min(misalignment, 3) if not np.isnan(misalignment) else 0  # Show misalignment

    # Left parent metal
    ax.fill_between([-15, -8], [y_left - 3, y_left - 3], [y_left, y_left],
                    color='#e0e0e0', edgecolor='#333333', linewidth=0.8)
    # Right parent metal
    ax.fill_between([8, 15], [y_right - 3, y_right - 3], [y_right, y_right],
                    color='#e0e0e0', edgecolor='#333333', linewidth=0.8)

    # Weld bead profile (raised above parent)
    x_weld = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])
    y_weld = np.array([y_left, y_left + 1.2, y_left + 1.8, y_left + 2.0, y_left + 2.2,
                       y_right + 2.0, y_right + 1.8, y_right + 1.2, y_right])
    ax.fill_between(x_weld, np.minimum(y_left, y_right) - 0.5, y_weld,
                    color='#ffcc80', edgecolor='#e65100', linewidth=1.0)

    # Reference line (datum)
    ax.axhline(y=y_left, color='#1976D2', linestyle='--', linewidth=0.8, label='Left datum')
    if abs(y_right - y_left) > 0.1:
        ax.axhline(y=y_right, color='#D32F2F', linestyle='--', linewidth=0.8, label='Right datum')

    # Misalignment annotation
    if not np.isnan(misalignment) and misalignment > 0.1:
        ax.annotate('', xy=(12, y_right), xytext=(12, y_left),
                    arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.0))
        ax.text(13, (y_left + y_right) / 2, f'{misalignment:.1f}mm',
                fontsize=7, va='center', ha='left')

    # Toe markers
    ax.plot([-8, -8], [y_left - 0.3, y_left + 0.3], 'k-', linewidth=1.5)
    ax.plot([8, 8], [y_right - 0.3, y_right + 0.3], 'k-', linewidth=1.5)

    # Exceedance probability annotation
    if not np.isnan(exceedance):
        ax.text(0, 3.5, f'P(exceed) = {exceedance:.3f}',
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#cccccc', alpha=0.9))

    ax.set_xlabel("Position (mm)", fontsize=8)
    ax.set_ylabel("Height (mm)", fontsize=8)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-4, -2, 0, 2])
    ax.tick_params(labelsize=7)


def draw_ut_panel(ax, cell_id: str, ut_data: dict, cls: str):
    """
    Draw UT reading panel showing amplitude bar with gate reference.
    """
    amplitude = float(ut_data.get("amplitude_pct", 0))
    located = int(ut_data.get("located", 0)) == 1
    soundpath = float(ut_data.get("soundpath_mm", 0)) if located else None

    # Set up axes
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 120)

    # Gate reference line (20%)
    ax.axhline(y=20, color='#D32F2F', linestyle='--', linewidth=1.0, label='Gate (20%)')
    ax.text(45, 22, '20%', fontsize=7, color='#D32F2F', ha='center')

    # Draw amplitude bar
    bar_width = 15
    bar_x = 15
    bar_color = CLASS_COLORS.get(cls, '#888888')

    ax.bar(bar_x, amplitude, width=bar_width, color=bar_color,
           edgecolor='#333333', linewidth=0.8, alpha=0.8)

    # Amplitude value on bar
    if amplitude > 15:
        ax.text(bar_x, amplitude - 5, f'{amplitude:.1f}%',
                ha='center', va='top', fontsize=9, fontweight='bold', color='white')
    else:
        ax.text(bar_x, amplitude + 3, f'{amplitude:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

    # Located/Not located indicator
    if located:
        status_text = "Located"
        status_color = '#4CAF50'
        if soundpath is not None:
            soundpath_text = f"SP: {soundpath:.1f}mm"
        else:
            soundpath_text = ""
    else:
        status_text = "Not located"
        status_color = '#9E9E9E'
        soundpath_text = ""

    ax.text(bar_x, 110, status_text, ha='center', va='bottom', fontsize=8,
            color=status_color, fontweight='bold')

    if soundpath_text:
        ax.text(bar_x, 103, soundpath_text, ha='center', va='bottom', fontsize=7,
                color='#666666')

    # Soundpath reference lines (20mm and 40mm)
    ax.axvline(x=35, color='#888888', linestyle=':', linewidth=0.5)
    ax.text(35, 5, '20mm', fontsize=6, color='#888888', ha='center', rotation=90)
    ax.axvline(x=42, color='#888888', linestyle=':', linewidth=0.5)
    ax.text(42, 5, '40mm', fontsize=6, color='#888888', ha='center', rotation=90)

    ax.set_ylabel("Amplitude (%)", fontsize=8)
    ax.set_xlabel("", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 120)


def generate_f4_1():
    """Generate the F4.1 figure."""
    # Load data
    geom = load_geometry_summary()
    ut = load_ut_features()

    # Figure setup - 4 rows x 3 columns
    fig_width = 180 / 25.4  # 180mm in inches
    fig_height = 160 / 25.4  # 160mm in inches

    fig, axes = plt.subplots(4, 3, figsize=(fig_width, fig_height))

    # Style setup
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['font.size'] = 9

    # Column labels
    col_labels = ["(a) Vision", "(b) Geometry", "(c) UT"]

    for row_idx, cls in enumerate(CLASS_ORDER):
        cell_id = SELECTED_CELLS[cls]

        # Row label (class)
        axes[row_idx, 0].text(-0.25, 0.5, f'{cls}:', transform=axes[row_idx, 0].transAxes,
                              fontsize=10, fontweight='bold', va='center', ha='right',
                              color=CLASS_COLORS[cls])

        # Column (a): Vision patch
        ax_vision = axes[row_idx, 0]
        try:
            patch_path = get_patch_path(cell_id)
            img = Image.open(patch_path)
            ax_vision.imshow(img)
            ax_vision.axis('off')
            ax_vision.set_title(cell_id, fontsize=8, pad=2)
        except FileNotFoundError as e:
            ax_vision.text(0.5, 0.5, f"Missing\n{cell_id}", ha='center', va='center',
                          transform=ax_vision.transAxes, fontsize=8)
            ax_vision.axis('off')

        # Column (b): Geometry profile
        ax_geom = axes[row_idx, 1]
        geom_data = geom.get(cell_id, {})
        draw_geometry_panel(ax_geom, cell_id, geom_data, cls)

        # Column (c): UT reading
        ax_ut = axes[row_idx, 2]
        ut_data = ut.get(cell_id, {})
        draw_ut_panel(ax_ut, cell_id, ut_data, cls)

    # Column headers
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].annotate(label, xy=(0.5, 1.15), xycoords='axes fraction',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)

    # Save
    fig.savefig(OUTPUT_DIR / "f4_1_cell_examples.pdf", format='pdf',
                bbox_inches='tight', pad_inches=0.05)
    fig.savefig(OUTPUT_DIR / "f4_1_cell_examples.png", format='png', dpi=300,
                bbox_inches='tight', pad_inches=0.05)

    plt.close(fig)

    print(f"Saved: {OUTPUT_DIR / 'f4_1_cell_examples.pdf'}")
    print(f"Saved: {OUTPUT_DIR / 'f4_1_cell_examples.png'}")

    return True


def main():
    print("Generating F4.1: Cell examples through three modalities")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    print("Selected cells:")
    for cls in CLASS_ORDER:
        cell_id = SELECTED_CELLS[cls]
        print(f"  {cls}: {cell_id}")
    print()

    success = generate_f4_1()

    if success:
        print("\nDone.")
    else:
        print("\nFailed.")


if __name__ == "__main__":
    main()
