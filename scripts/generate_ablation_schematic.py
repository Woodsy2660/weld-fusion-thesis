#!/usr/bin/env python3
"""Generate ablation schematic: three modality BPAs per cell feeding seven
combination branches, excluded modality shown as vacuous m(Θ) = 1.

Shows the data flow from per-cell BPAs through the 7 ablation subsets to
final fusion decisions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path


def create_ablation_schematic():
    """Create schematic showing ablation study data flow."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Colors for modalities
    colors = {
        'V': '#4477AA',  # Blue
        'G': '#228833',  # Green
        'U': '#EE6677',  # Red
        'vacuous': '#BBBBBB',  # Gray for vacuous
        'fusion': '#AA3377',  # Purple for fusion
        'decision': '#CCBB44'  # Yellow for decision
    }

    # === LEFT SIDE: Modality BPAs ===
    modality_x = 1
    modality_spacing = 2.5

    # Vision BPA
    v_box = FancyBboxPatch((modality_x - 0.8, 7), 1.6, 1.2,
                           boxstyle="round,pad=0.05",
                           facecolor=colors['V'], edgecolor='black',
                           linewidth=2, alpha=0.8)
    ax.add_patch(v_box)
    ax.text(modality_x, 7.6, 'Vision\nBPA', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(modality_x, 6.7, r'$m_V$', ha='center', va='top', fontsize=10, style='italic')

    # Geometry BPA
    g_box = FancyBboxPatch((modality_x - 0.8, 4.5), 1.6, 1.2,
                           boxstyle="round,pad=0.05",
                           facecolor=colors['G'], edgecolor='black',
                           linewidth=2, alpha=0.8)
    ax.add_patch(g_box)
    ax.text(modality_x, 5.1, 'Geometry\nBPA', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(modality_x, 4.2, r'$m_G$', ha='center', va='top', fontsize=10, style='italic')

    # Ultrasonic BPA
    u_box = FancyBboxPatch((modality_x - 0.8, 2), 1.6, 1.2,
                           boxstyle="round,pad=0.05",
                           facecolor=colors['U'], edgecolor='black',
                           linewidth=2, alpha=0.8)
    ax.add_patch(u_box)
    ax.text(modality_x, 2.6, 'Ultrasonic\nBPA', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(modality_x, 1.7, r'$m_U$', ha='center', va='top', fontsize=10, style='italic')

    # Alpha labels
    ax.text(modality_x + 1.2, 7.6, r'$\alpha_V = 0.91$', ha='left', va='center',
            fontsize=9, color=colors['V'])
    ax.text(modality_x + 1.2, 5.1, r'$\alpha_G = 1.00$', ha='left', va='center',
            fontsize=9, color=colors['G'])
    ax.text(modality_x + 1.2, 2.6, r'$\alpha_U = 0.24$', ha='left', va='center',
            fontsize=9, color=colors['U'])

    # === MIDDLE: Seven ablation branches ===
    subset_x = 6
    subsets = ['V', 'G', 'U', 'VG', 'VU', 'GU', 'VGU']
    subset_y_positions = [9, 7.5, 6, 4.5, 3, 1.5, 0]

    # Vacuous BPA reference
    ax.text(3.5, 9.8, r'Excluded modality $\rightarrow$ vacuous: $m(\Theta) = 1$',
            ha='center', va='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    for i, (subset, y) in enumerate(zip(subsets, subset_y_positions)):
        # Subset label box
        subset_box = FancyBboxPatch((subset_x - 0.6, y - 0.4), 1.2, 0.8,
                                    boxstyle="round,pad=0.03",
                                    facecolor='white', edgecolor='black',
                                    linewidth=1.5)
        ax.add_patch(subset_box)
        ax.text(subset_x, y, subset, ha='center', va='center',
                fontsize=12, fontweight='bold', family='monospace')

        # Draw input indicators for each modality
        indicator_x = subset_x - 1.5
        indicator_width = 0.4

        # Vision indicator
        v_active = 'V' in subset
        v_color = colors['V'] if v_active else colors['vacuous']
        v_indicator = Rectangle((indicator_x, y + 0.15), indicator_width, 0.2,
                                facecolor=v_color, edgecolor='black', linewidth=0.5)
        ax.add_patch(v_indicator)

        # Geometry indicator
        g_active = 'G' in subset
        g_color = colors['G'] if g_active else colors['vacuous']
        g_indicator = Rectangle((indicator_x, y - 0.1), indicator_width, 0.2,
                                facecolor=g_color, edgecolor='black', linewidth=0.5)
        ax.add_patch(g_indicator)

        # Ultrasonic indicator
        u_active = 'U' in subset
        u_color = colors['U'] if u_active else colors['vacuous']
        u_indicator = Rectangle((indicator_x, y - 0.35), indicator_width, 0.2,
                                facecolor=u_color, edgecolor='black', linewidth=0.5)
        ax.add_patch(u_indicator)

        # Draw connecting lines from source BPAs
        # Vision connection
        if v_active:
            ax.annotate('', xy=(indicator_x, y + 0.25),
                       xytext=(modality_x + 0.8, 7.6),
                       arrowprops=dict(arrowstyle='-', color=colors['V'],
                                      lw=1, alpha=0.6,
                                      connectionstyle='arc3,rad=0.1'))

        # Geometry connection
        if g_active:
            ax.annotate('', xy=(indicator_x, y),
                       xytext=(modality_x + 0.8, 5.1),
                       arrowprops=dict(arrowstyle='-', color=colors['G'],
                                      lw=1, alpha=0.6,
                                      connectionstyle='arc3,rad=0.05'))

        # Ultrasonic connection
        if u_active:
            ax.annotate('', xy=(indicator_x, y - 0.25),
                       xytext=(modality_x + 0.8, 2.6),
                       arrowprops=dict(arrowstyle='-', color=colors['U'],
                                      lw=1, alpha=0.6,
                                      connectionstyle='arc3,rad=-0.1'))

    # === RIGHT SIDE: Fusion and Decision ===
    fusion_x = 9

    # Fusion operation box
    fusion_box = FancyBboxPatch((fusion_x - 1, 3.5), 2, 2,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['fusion'], edgecolor='black',
                                linewidth=2, alpha=0.8)
    ax.add_patch(fusion_box)
    ax.text(fusion_x, 4.8, 'Dempster\nCombination', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(fusion_x, 3.9, r'$m_{1 \oplus 2 \oplus 3}$', ha='center', va='center',
            fontsize=10, color='white', style='italic')

    # Arrows from subsets to fusion
    for y in subset_y_positions:
        ax.annotate('', xy=(fusion_x - 1, 4.5),
                   xytext=(subset_x + 0.6, y),
                   arrowprops=dict(arrowstyle='->', color='gray',
                                  lw=1, alpha=0.5,
                                  connectionstyle='arc3,rad=0'))

    # Decision box
    decision_x = 12
    decision_box = FancyBboxPatch((decision_x - 1, 3.5), 2, 2,
                                  boxstyle="round,pad=0.05",
                                  facecolor=colors['decision'], edgecolor='black',
                                  linewidth=2, alpha=0.8)
    ax.add_patch(decision_box)
    ax.text(decision_x, 4.8, 'Pignistic', ha='center', va='center',
            fontsize=11, fontweight='bold', color='black')
    ax.text(decision_x, 4.2, 'Transform', ha='center', va='center',
            fontsize=11, fontweight='bold', color='black')
    ax.text(decision_x, 3.7, r'$BetP$', ha='center', va='center',
            fontsize=10, style='italic')

    # Arrow from fusion to decision
    ax.annotate('', xy=(decision_x - 1, 4.5),
               xytext=(fusion_x + 1, 4.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Output
    ax.annotate('', xy=(decision_x + 1.8, 4.5),
               xytext=(decision_x + 1, 4.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(decision_x + 2.2, 4.5, r'$\hat{y}$', ha='left', va='center',
            fontsize=14, fontweight='bold')

    # === LEGEND ===
    legend_elements = [
        mpatches.Patch(facecolor=colors['V'], edgecolor='black', label='Vision active'),
        mpatches.Patch(facecolor=colors['G'], edgecolor='black', label='Geometry active'),
        mpatches.Patch(facecolor=colors['U'], edgecolor='black', label='Ultrasonic active'),
        mpatches.Patch(facecolor=colors['vacuous'], edgecolor='black',
                      label=r'Vacuous $m(\Theta)=1$'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9, ncol=2)

    # === ANNOTATIONS ===
    # Title
    ax.text(7, 10.5, 'Ablation Study: Modality Fusion Architecture',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Per-cell note
    ax.text(1, 0.3, 'Per cell (×150)', ha='center', va='center',
            fontsize=9, style='italic', color='gray')

    # Seven subsets note
    ax.text(subset_x, -0.8, '7 ablation subsets', ha='center', va='center',
            fontsize=9, style='italic', color='gray')

    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-1.5, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "results" / "thesis" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating ablation schematic...")
    fig = create_ablation_schematic()

    output_pdf = output_dir / "f4_6_ablation_schematic.pdf"
    output_png = output_dir / "f4_6_ablation_schematic.png"

    fig.savefig(output_pdf, bbox_inches='tight', dpi=300)
    fig.savefig(output_png, bbox_inches='tight', dpi=300)

    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")

    plt.close('all')


if __name__ == "__main__":
    main()
