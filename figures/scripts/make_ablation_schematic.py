#!/usr/bin/env python3
"""
Generate ablation conditions schematic for Chapter 3 (§3.5, fig:ablation_schematic).

This is a METHODOLOGY figure showing the experimental design, not results.
No fitted values (alpha, F1, K) appear anywhere.

Output:
    figures/ablation_schematic.pdf
    figures/ablation_schematic.png (300 dpi)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle

# Font setup
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9

# Colors
COLORS = {
    'V': {'accent': '#4a73a8', 'fill': '#e8f0fe'},
    'G': {'accent': '#5e8f6b', 'fill': '#e6f4ea'},
    'U': {'accent': '#c08a4a', 'fill': '#fdf0e1'},
}
ANNOTATION_COLOR = '#8e867a'
EXCLUDED_FILL = '#f5f5f5'
EXCLUDED_EDGE = '#aaaaaa'

# Condition definitions: (label, V_included, G_included, U_included)
CONDITIONS = [
    ('V',     True,  False, False),
    ('G',     False, True,  False),
    ('U',     False, False, True),
    ('V+G',   True,  True,  False),
    ('V+U',   True,  False, True),
    ('G+U',   False, True,  True),
    ('V+G+U', True,  True,  True),
]


def draw_cylinder(ax, x, y, width, height, accent, fill):
    """Draw a cylinder (stored data)."""
    ellipse_h = height * 0.20

    # Body
    body = FancyBboxPatch(
        (x - width/2, y - height/2 + ellipse_h/2),
        width, height - ellipse_h,
        boxstyle="square,pad=0",
        facecolor=fill, edgecolor=accent, linewidth=0.8
    )
    ax.add_patch(body)

    # Top ellipse
    top = mpatches.Ellipse(
        (x, y + height/2 - ellipse_h/2),
        width, ellipse_h,
        facecolor=fill, edgecolor=accent, linewidth=0.8
    )
    ax.add_patch(top)

    # Bottom ellipse (half visible)
    bottom = mpatches.Arc(
        (x, y - height/2 + ellipse_h/2),
        width, ellipse_h,
        angle=0, theta1=180, theta2=360,
        edgecolor=accent, linewidth=0.8
    )
    ax.add_patch(bottom)


def draw_matrix_cell(ax, x, y, size, included, modality):
    """Draw a single matrix cell (included or excluded)."""
    half = size / 2

    if included:
        # Solid filled square
        rect = Rectangle(
            (x - half, y - half), size, size,
            facecolor=COLORS[modality]['accent'],
            edgecolor=COLORS[modality]['accent'],
            linewidth=0.6, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(x, y, r'$m^\alpha$', ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')
    else:
        # Dashed outline, light fill
        rect = Rectangle(
            (x - half, y - half), size, size,
            facecolor=EXCLUDED_FILL,
            edgecolor=EXCLUDED_EDGE,
            linewidth=0.6, linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(x, y, r'$m(\Theta)$', ha='center', va='center',
                fontsize=5.5, color='#888888')


def draw_dempster_box(ax, x, y, size):
    """Draw a Dempster combination box."""
    box = FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor='white', edgecolor='#333333', linewidth=0.7
    )
    ax.add_patch(box)
    ax.text(x, y, r'$\oplus$', ha='center', va='center',
            fontsize=10, color='#333333')


def main():
    # Figure dimensions: 160mm x 95mm
    fig_width = 160 / 25.4  # inches
    fig_height = 95 / 25.4

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')

    # Layout parameters
    # Cylinders in their own column, well-separated from the matrix
    cyl_x = 0.06
    cyl_width = 0.045
    cyl_height = 0.085
    cyl_spacing = 0.22
    cyl_top_y = 0.82

    # Matrix starts well to the right of the cylinders
    matrix_left = 0.36
    matrix_col_spacing = 0.08
    matrix_row_spacing = 0.075
    cell_size = 0.048
    matrix_top_y = 0.88

    dempster_x = 0.61
    dempster_size = 0.038

    output_x = 0.68

    # =========================================================================
    # LEFT: Cylinders (stored inputs) with labels below each
    # =========================================================================
    modalities = ['V', 'G', 'U']
    cyl_labels = [
        (r'Vision', r'$m_V^\alpha$'),
        (r'Geometry', r'$m_G^\alpha$'),
        (r'Ultrasonic', r'$m_U^\alpha$'),
    ]
    cyl_ys = [cyl_top_y - i * cyl_spacing for i in range(3)]

    for i, mod in enumerate(modalities):
        draw_cylinder(ax, cyl_x, cyl_ys[i], cyl_width, cyl_height,
                      COLORS[mod]['accent'], COLORS[mod]['fill'])

        # Labels below each cylinder
        name, mass = cyl_labels[i]
        ax.text(cyl_x, cyl_ys[i] - cyl_height/2 - 0.015,
                f'{name}  {mass}',
                ha='center', va='top', fontsize=7,
                color=COLORS[mod]['accent'], fontweight='bold')

    # Caption at the very bottom of the cylinder column
    ax.text(cyl_x, 0.12,
            'Discounted BPA per cell,\ncomputed once per fold',
            ha='center', va='top', fontsize=6.5, color='#555555',
            linespacing=1.15)

    # =========================================================================
    # CENTER: Inclusion matrix
    # =========================================================================
    col_xs = [matrix_left + i * matrix_col_spacing for i in range(3)]

    # Column headers
    for i, mod in enumerate(modalities):
        ax.text(col_xs[i], matrix_top_y + 0.02, mod,
                ha='center', va='bottom', fontsize=8,
                color=COLORS[mod]['accent'], fontweight='bold')

    # Dashed lines from cylinders to column headers (curved paths)
    for i, mod in enumerate(modalities):
        start_x = cyl_x + cyl_width/2 + 0.01
        start_y = cyl_ys[i]
        end_x = col_xs[i]
        end_y = matrix_top_y + 0.012

        # Curved dashed line
        ax.annotate('', xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle='arc3,rad=0.2',
                                    color=COLORS[mod]['accent'],
                                    linestyle='--',
                                    linewidth=0.5, alpha=0.5))

    # Matrix rows
    row_ys = [matrix_top_y - 0.05 - i * matrix_row_spacing for i in range(7)]

    # Draw band separators and labels on the far left of matrix
    band_label_x = matrix_left - 0.065

    # Single band (rows 0-2)
    single_top = row_ys[0] + cell_size/2 + 0.010
    single_bot = row_ys[2] - cell_size/2 - 0.006
    ax.plot([matrix_left - 0.030, col_xs[2] + cell_size/2 + 0.006],
            [single_bot, single_bot], color='#cccccc', linewidth=0.5)
    ax.text(band_label_x, (single_top + single_bot)/2, 'Single',
            ha='right', va='center', fontsize=6.5, color='#666666', rotation=90)

    # Pairs band (rows 3-5)
    pairs_top = row_ys[3] + cell_size/2 + 0.006
    pairs_bot = row_ys[5] - cell_size/2 - 0.006
    ax.plot([matrix_left - 0.030, col_xs[2] + cell_size/2 + 0.006],
            [pairs_bot, pairs_bot], color='#cccccc', linewidth=0.5)
    ax.text(band_label_x, (pairs_top + pairs_bot)/2, 'Pairs',
            ha='right', va='center', fontsize=6.5, color='#666666', rotation=90)

    # All three band (row 6)
    all_top = row_ys[6] + cell_size/2 + 0.006
    all_bot = row_ys[6] - cell_size/2 - 0.010
    ax.text(band_label_x, (all_top + all_bot)/2, 'All three',
            ha='right', va='center', fontsize=6.5, color='#666666', rotation=90)

    # Row labels and cells
    for row_idx, (cond_label, v_inc, g_inc, u_inc) in enumerate(CONDITIONS):
        y = row_ys[row_idx]

        # Row label
        ax.text(matrix_left - 0.038, y, cond_label,
                ha='right', va='center', fontsize=7.5, color='#444444')

        # Draw cells
        inclusions = [v_inc, g_inc, u_inc]
        for col_idx, (inc, mod) in enumerate(zip(inclusions, modalities)):
            draw_matrix_cell(ax, col_xs[col_idx], y, cell_size, inc, mod)

    # =========================================================================
    # RIGHT: Dempster combination and outputs
    # =========================================================================
    for row_idx in range(7):
        y = row_ys[row_idx]

        # Arrow from matrix row to Dempster box
        arrow_start = col_xs[2] + cell_size/2 + 0.010
        ax.annotate('', xy=(dempster_x - dempster_size/2 - 0.003, y),
                    xytext=(arrow_start, y),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                    lw=0.6, shrinkA=0, shrinkB=0))

        # Dempster box
        draw_dempster_box(ax, dempster_x, y, dempster_size)

        # Arrow to output
        ax.annotate('', xy=(output_x - 0.003, y),
                    xytext=(dempster_x + dempster_size/2 + 0.003, y),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                    lw=0.6, shrinkA=0, shrinkB=0))

        # Output label
        if row_idx == 0:
            ax.text(output_x, y, r'$m, K, \mathrm{BetP}, [\mathrm{Bel}, \mathrm{Pl}]$',
                    ha='left', va='center', fontsize=6.5, color='#333333')
        else:
            ax.text(output_x, y, r'$\prime\prime$',
                    ha='left', va='center', fontsize=8, color='#666666')

    # Murphy note under outputs
    ax.text(dempster_x, row_ys[6] - 0.050,
            'Murphy averaging computed\nin parallel as comparator',
            ha='center', va='top', fontsize=6, color='#777777',
            style='italic', linespacing=1.1)

    # =========================================================================
    # ANNOTATIONS (methods only)
    # =========================================================================

    # Annotation for V+G row (row 3): "Prop. 3.2: {L} unreachable"
    vg_y = row_ys[3]
    annot_x = 0.83
    ax.text(annot_x, vg_y, r'Prop. 3.2: $\{L\}$ unreachable',
            ha='left', va='center', fontsize=6.5, color=ANNOTATION_COLOR)
    # Leader line
    ax.plot([output_x + 0.105, annot_x - 0.005], [vg_y, vg_y],
            color=ANNOTATION_COLOR, linewidth=0.5, linestyle='-')

    # Bracket for rows containing U (rows 2, 4, 5, 6)
    u_rows = [2, 4, 5, 6]
    bracket_x = 0.96
    bracket_top = row_ys[u_rows[0]] + 0.016
    bracket_bot = row_ys[u_rows[-1]] - 0.016

    # Draw bracket
    ax.plot([bracket_x - 0.010, bracket_x], [bracket_top, bracket_top],
            color=ANNOTATION_COLOR, linewidth=0.6)
    ax.plot([bracket_x, bracket_x], [bracket_top, bracket_bot],
            color=ANNOTATION_COLOR, linewidth=0.6)
    ax.plot([bracket_x - 0.010, bracket_x], [bracket_bot, bracket_bot],
            color=ANNOTATION_COLOR, linewidth=0.6)

    # Bracket label
    ax.text(bracket_x + 0.008, (bracket_top + bracket_bot)/2,
            'Only conditions\nable to assert $\\{L\\}$',
            ha='left', va='center', fontsize=6, color=ANNOTATION_COLOR,
            linespacing=1.05)

    # Connect bracket to relevant rows with small ticks
    for row_idx in u_rows:
        y = row_ys[row_idx]
        ax.plot([bracket_x - 0.010, bracket_x - 0.005], [y, y],
                color=ANNOTATION_COLOR, linewidth=0.4)

    # =========================================================================
    # FOOTER
    # =========================================================================
    footer_text = (
        r"Excluded modality $=$ vacuous $m(\Theta)=1$, the identity of Dempster's rule "
        r"($\Theta \cap A = A$), so each row differs only in the modalities included."
    )
    ax.text(0.5, 0.025, footer_text,
            ha='center', va='bottom', fontsize=6.5, color='#555555',
            wrap=True)

    # =========================================================================
    # Save
    # =========================================================================
    plt.tight_layout(pad=0.2)

    output_dir = Path(__file__).resolve().parents[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / 'ablation_schematic.pdf'
    png_path = output_dir / 'ablation_schematic.png'

    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    plt.close(fig)


if __name__ == '__main__':
    main()
