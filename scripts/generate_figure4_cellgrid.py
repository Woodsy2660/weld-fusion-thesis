#!/usr/bin/env python3
"""Generate Figure 4: Cell grid on a coupon with zones A-C and excluded end cells.

Shows:
- Zones A, B, C (100mm each)
- 10 cells (cells 0-9) at 25mm pitch
- Excluded cells at ends (cells 0 and 9) hatched
- Axial coordinate s from datum hole end
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from pathlib import Path
from PIL import Image

# Configuration matching configs/geometry.yaml
CELL_WIDTH_MM = 25
FIRST_EXCLUDED_MM = 25
LAST_EXCLUDED_MM = 25
COUPON_LENGTH_MM = 300
N_CELLS = 10

# Zone definitions (approximate)
# Zones A, B, C each ~100mm but mapped to cells
# Cells 0-9 span 25mm to 275mm (250mm active region)
# Zone A: cells 1-3 (75mm) - but user says 100mm each
# Actually based on user description: 3 zones of 100mm = 300mm total
# With 25mm excluded at each end, active = 250mm
# But the zones span the full coupon nominally

def create_schematic_figure():
    """Create a schematic diagram of the cell grid without actual image."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Draw coupon outline
    coupon_rect = Rectangle((0, 0), COUPON_LENGTH_MM, 40,
                             linewidth=2, edgecolor='black',
                             facecolor='lightgray', zorder=1)
    ax.add_patch(coupon_rect)

    # Draw weld bead in center
    weld_rect = Rectangle((0, 17), COUPON_LENGTH_MM, 6,
                           linewidth=1, edgecolor='darkgray',
                           facecolor='silver', zorder=2)
    ax.add_patch(weld_rect)

    # Draw cell divisions
    for i in range(N_CELLS + 1):
        x = FIRST_EXCLUDED_MM + i * CELL_WIDTH_MM
        ax.axvline(x, color='blue', linestyle='--', linewidth=1, zorder=3)

    # Draw excluded region at start (0-25mm)
    excluded_start = Rectangle((0, 0), FIRST_EXCLUDED_MM, 40,
                                linewidth=1, edgecolor='red',
                                facecolor='none', hatch='///',
                                alpha=0.5, zorder=4)
    ax.add_patch(excluded_start)

    # Draw excluded region at end (275-300mm)
    excluded_end = Rectangle((COUPON_LENGTH_MM - LAST_EXCLUDED_MM, 0),
                             LAST_EXCLUDED_MM, 40,
                             linewidth=1, edgecolor='red',
                             facecolor='none', hatch='///',
                             alpha=0.5, zorder=4)
    ax.add_patch(excluded_end)

    # Cell labels
    for i in range(N_CELLS):
        x_center = FIRST_EXCLUDED_MM + (i + 0.5) * CELL_WIDTH_MM
        ax.text(x_center, -6, str(i), ha='center', va='top', fontsize=10, fontweight='bold')

    # Zone labels and brackets
    # Zone A: cells 0-2 (25-100mm) = 75mm
    # Zone B: cells 3-6 (100-200mm) = 100mm
    # Zone C: cells 7-9 (200-275mm) = 75mm
    # Actually, let's do equal zones conceptually spanning the active region

    zone_y = 47
    zone_height = 8

    # Zone A: 25-108.3mm
    zone_a_start = 25
    zone_a_end = 25 + (250/3)
    ax.annotate('', xy=(zone_a_start, zone_y), xytext=(zone_a_end, zone_y),
                arrowprops=dict(arrowstyle='|-|', color='darkblue', lw=2))
    ax.text((zone_a_start + zone_a_end)/2, zone_y + 5, 'Zone A',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkblue')

    # Zone B: 108.3-191.7mm
    zone_b_start = zone_a_end
    zone_b_end = zone_b_start + (250/3)
    ax.annotate('', xy=(zone_b_start, zone_y), xytext=(zone_b_end, zone_y),
                arrowprops=dict(arrowstyle='|-|', color='darkgreen', lw=2))
    ax.text((zone_b_start + zone_b_end)/2, zone_y + 5, 'Zone B',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkgreen')

    # Zone C: 191.7-275mm
    zone_c_start = zone_b_end
    zone_c_end = 275
    ax.annotate('', xy=(zone_c_start, zone_y), xytext=(zone_c_end, zone_y),
                arrowprops=dict(arrowstyle='|-|', color='darkred', lw=2))
    ax.text((zone_c_start + zone_c_end)/2, zone_y + 5, 'Zone C',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkred')

    # Excluded labels
    ax.text(FIRST_EXCLUDED_MM/2, 44, 'Excluded\n(arc start)',
            ha='center', va='bottom', fontsize=9, style='italic', color='red')
    ax.text(COUPON_LENGTH_MM - LAST_EXCLUDED_MM/2, 44, 'Excluded\n(arc stop)',
            ha='center', va='bottom', fontsize=9, style='italic', color='red')

    # Axial coordinate arrow
    ax.annotate('', xy=(280, -15), xytext=(20, -15),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(150, -22, 's (mm)', ha='center', va='top', fontsize=11, style='italic')

    # Datum hole indicator
    ax.plot(15, 20, 'ko', markersize=8)
    ax.text(15, 25, '⌀6mm\ndatum', ha='center', va='bottom', fontsize=8)

    # Axis hole indicator
    ax.plot(285, 20, 'ko', markersize=10)
    ax.text(285, 25, '⌀8mm', ha='center', va='bottom', fontsize=8)

    # Scale bar
    ax.plot([250, 300], [-28, -28], 'k-', lw=2)
    ax.text(275, -30, '50 mm', ha='center', va='top', fontsize=10)

    # Legend
    excluded_patch = mpatches.Patch(facecolor='none', edgecolor='red',
                                    hatch='///', label='Excluded (25 mm)')
    cell_line = plt.Line2D([0], [0], color='blue', linestyle='--',
                           label='Cell boundaries')
    ax.legend(handles=[excluded_patch, cell_line], loc='upper right',
              framealpha=0.9)

    ax.set_xlim(-10, 310)
    ax.set_ylim(-35, 70)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig


def create_image_based_figure(overlay_path: Path, output_path: Path):
    """Create figure overlaying cell grid on actual coupon image."""
    # Load image
    img = Image.open(overlay_path)
    img_array = np.array(img)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Display image
    ax.imshow(img_array)

    # Get image dimensions
    img_height, img_width = img_array.shape[:2]

    # Crop region estimate (focus on weld area)
    # Weld is roughly in the middle vertically
    weld_y_center = img_height * 0.45
    crop_height = img_height * 0.25

    ax.set_ylim(weld_y_center + crop_height, weld_y_center - crop_height)

    # The coupon spans most of the image width
    # Full coupon is 300mm: 25mm excluded + 250mm active (10 cells) + 25mm excluded
    # So the 10 cells span 250/300 = 83.3% of the coupon length
    coupon_left_px = img_width * 0.04   # Left edge of coupon in image
    coupon_right_px = img_width * 0.96  # Right edge of coupon in image
    coupon_width_px = coupon_right_px - coupon_left_px

    # Excluded regions are 25mm each out of 300mm total = 8.33% each end
    excluded_fraction = 25 / 300
    excluded_width_px = coupon_width_px * excluded_fraction

    # Active region (10 cells)
    active_left_px = coupon_left_px + excluded_width_px
    active_right_px = coupon_right_px - excluded_width_px
    active_width_px = active_right_px - active_left_px

    # Draw cell divisions within active region
    for i in range(N_CELLS + 1):
        x = active_left_px + (i / N_CELLS) * active_width_px
        ax.axvline(x, color='cyan', linestyle='--', linewidth=2, alpha=0.8)

    # Cell labels
    for i in range(N_CELLS):
        x_center = active_left_px + ((i + 0.5) / N_CELLS) * active_width_px
        ax.text(x_center, weld_y_center + crop_height - 20, str(i),
                ha='center', va='top', fontsize=12, fontweight='bold',
                color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Zone labels
    zone_y = weld_y_center - crop_height + 30

    # Zone A: cells 0-2
    zone_a_x = active_left_px + (1.5 / N_CELLS) * active_width_px
    ax.text(zone_a_x, zone_y, 'Zone A', ha='center', fontsize=14,
            fontweight='bold', color='yellow',
            bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.8))

    # Zone B: cells 3-6
    zone_b_x = active_left_px + (5 / N_CELLS) * active_width_px
    ax.text(zone_b_x, zone_y, 'Zone B', ha='center', fontsize=14,
            fontweight='bold', color='yellow',
            bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))

    # Zone C: cells 7-9
    zone_c_x = active_left_px + (8.5 / N_CELLS) * active_width_px
    ax.text(zone_c_x, zone_y, 'Zone C', ha='center', fontsize=14,
            fontweight='bold', color='yellow',
            bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8))

    # Excluded regions - hatch overlay (OUTSIDE the 10-cell grid)
    # Start excluded (before cell 0)
    excluded_start = Rectangle((coupon_left_px, weld_y_center - crop_height),
                               excluded_width_px, 2 * crop_height,
                               linewidth=2, edgecolor='red',
                               facecolor='red', alpha=0.3, hatch='///')
    ax.add_patch(excluded_start)

    # End excluded (after cell 9)
    excluded_end = Rectangle((active_right_px, weld_y_center - crop_height),
                             excluded_width_px, 2 * crop_height,
                             linewidth=2, edgecolor='red',
                             facecolor='red', alpha=0.3, hatch='///')
    ax.add_patch(excluded_end)

    ax.axis('off')
    plt.tight_layout()

    return fig


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "results" / "thesis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create schematic version (clean, no image dependency)
    print("Creating schematic cell grid figure...")
    fig_schematic = create_schematic_figure()
    fig_schematic.savefig(output_dir / "f4_4_cellgrid_schematic.pdf",
                          bbox_inches='tight', dpi=300)
    fig_schematic.savefig(output_dir / "f4_4_cellgrid_schematic.png",
                          bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir / 'f4_4_cellgrid_schematic.pdf'}")

    # Create image-based version using overlay
    overlay_dir = base_dir / "results" / "geometry" / "overlays"
    overlay_files = sorted(overlay_dir.glob("*_top_r0_overlay.png"))

    if overlay_files:
        # Use first available overlay
        overlay_path = overlay_files[0]
        print(f"Creating image-based figure using {overlay_path.name}...")

        fig_image = create_image_based_figure(overlay_path,
                                              output_dir / "f4_4_cellgrid_image.pdf")
        fig_image.savefig(output_dir / "f4_4_cellgrid_image.pdf",
                         bbox_inches='tight', dpi=300)
        fig_image.savefig(output_dir / "f4_4_cellgrid_image.png",
                         bbox_inches='tight', dpi=300)
        print(f"Saved: {output_dir / 'f4_4_cellgrid_image.pdf'}")

    plt.close('all')
    print("\nFigure 4 generation complete.")


if __name__ == "__main__":
    main()
