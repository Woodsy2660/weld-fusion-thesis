#!/usr/bin/env python3
"""
Generate Chapter 4 figures from stored results.

Figures:
- F4.2: Coupon strip showing all modalities on cell grid
- F4.3: Confusion matrices for each modality
- F4.4: Recall heatmap (per-class recall × conditions)
- F4.5: Ablation F1 bar chart with CI
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Style settings per spec
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8

# Output directory
OUT_DIR = Path("results/thesis/ch4_evidence")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Modality colors per spec
COLORS = {
    'vision': '#d0e3fe',
    'geometry': '#e2fae4',
    'ultrasonic': '#fbfdcd',
    'edge': '#333333'
}

# Figure dimensions (mm -> inches)
FULL_WIDTH = 150 / 25.4  # 150mm
HALF_WIDTH = 75 / 25.4   # 75mm
DPI = 300


def load_data():
    """Load all necessary data."""
    labels = pd.read_csv("data/labels_v2_2026-10-26.csv")
    geometry = pd.read_csv("results/geometry/summary.csv")
    ut_bpa = pd.read_csv("results/ut/bpa_ut.csv")
    ut_metrics = json.load(open("results/ut/metrics_cv.json"))
    vision_cal = json.load(open("results/vision/coupon_eval/calibration_v2.json"))
    fusion_stats = json.load(open("results/fusion/statistics.json"))

    return {
        'labels': labels,
        'geometry': geometry,
        'ut_bpa': ut_bpa,
        'ut_metrics': ut_metrics,
        'vision_cal': vision_cal,
        'fusion_stats': fusion_stats
    }


def select_coupon_for_f42(labels, geometry):
    """Select coupon with most distinct labels, then most geometry cells."""
    measured = geometry[geometry["is_vacuous"] == False]

    best_coupon = None
    best_distinct = 0
    best_geom_cells = 0

    for coupon_id in labels["coupon_id"].unique():
        coupon_labels = labels[labels["coupon_id"] == coupon_id]["true_class"]
        distinct = coupon_labels.nunique()
        geom_cells = len(measured[measured["coupon_id"] == coupon_id])

        if distinct > best_distinct or (distinct == best_distinct and geom_cells > best_geom_cells):
            best_coupon = coupon_id
            best_distinct = distinct
            best_geom_cells = geom_cells

    return best_coupon


# ============================================================================
# F4.2: Coupon strip showing all modalities on cell grid
# ============================================================================
def generate_f42_coupon_strip(data):
    """Generate coupon strip figure."""
    labels = data['labels']
    geometry = data['geometry']
    ut_bpa = data['ut_bpa']

    # Select coupon
    coupon_id = select_coupon_for_f42(labels, geometry)
    print(f"  F4.2: Using coupon {coupon_id}")

    # Get cells for this coupon
    coupon_labels = labels[labels["coupon_id"] == coupon_id].sort_values("cell_index")
    coupon_geom = geometry[geometry["coupon_id"] == coupon_id]
    coupon_ut = ut_bpa[ut_bpa["coupon_id"] == coupon_id]

    n_cells = len(coupon_labels)

    fig, axes = plt.subplots(4, 1, figsize=(FULL_WIDTH, 3), sharex=True)

    # Row 1: True labels
    ax = axes[0]
    class_colors = {'N': '#808080', 'U': '#ff9999', 'P': '#99ff99', 'L': '#9999ff'}
    for i, (_, row) in enumerate(coupon_labels.iterrows()):
        color = class_colors.get(row['true_class'], '#cccccc')
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor=COLORS['edge'], linewidth=0.5))
        ax.text(i + 0.5, 0.5, row['true_class'], ha='center', va='center', fontsize=7)
    ax.set_xlim(0, n_cells)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Label', fontsize=8)
    ax.set_yticks([])

    # Row 2: Vision target
    ax = axes[1]
    for i, (_, row) in enumerate(coupon_labels.iterrows()):
        target = row['vision_target']
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=COLORS['vision'],
                                    edgecolor=COLORS['edge'], linewidth=0.5))
        # Abbreviate target
        abbrev = target.replace('{', '').replace('}', '').replace(',', '')
        ax.text(i + 0.5, 0.5, abbrev, ha='center', va='center', fontsize=6)
    ax.set_xlim(0, n_cells)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Vision', fontsize=8)
    ax.set_yticks([])

    # Row 3: Geometry (exceedance or vacuous)
    ax = axes[2]
    for i, (_, row) in enumerate(coupon_labels.iterrows()):
        cell_id = row['cell_id']
        geom_row = coupon_geom[coupon_geom['cell_id'] == cell_id]
        if len(geom_row) > 0 and not geom_row.iloc[0]['is_vacuous']:
            m_u = geom_row.iloc[0]['m_U']
            if m_u > 0.5:
                label = f"U:{m_u:.2f}"
            else:
                label = "Θ"
            fc = COLORS['geometry']
        else:
            label = "vac"
            fc = '#f0f0f0'
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=fc,
                                    edgecolor=COLORS['edge'], linewidth=0.5))
        ax.text(i + 0.5, 0.5, label, ha='center', va='center', fontsize=6)
    ax.set_xlim(0, n_cells)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Geometry', fontsize=8)
    ax.set_yticks([])

    # Row 4: UT BPA (argmax)
    ax = axes[3]
    for i, (_, row) in enumerate(coupon_labels.iterrows()):
        cell_id = row['cell_id']
        ut_row = coupon_ut[coupon_ut['cell_id'] == cell_id]
        if len(ut_row) > 0:
            masses = {
                'L': ut_row.iloc[0]['m_{L}'],
                'NU': ut_row.iloc[0]['m_{N,U}'],
                'P': ut_row.iloc[0]['m_{P}']
            }
            top = max(masses, key=masses.get)
            label = f"{top}:{masses[top]:.2f}"
        else:
            label = "?"
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=COLORS['ultrasonic'],
                                    edgecolor=COLORS['edge'], linewidth=0.5))
        ax.text(i + 0.5, 0.5, label, ha='center', va='center', fontsize=6)
    ax.set_xlim(0, n_cells)
    ax.set_ylim(0, 1)
    ax.set_ylabel('UT', fontsize=8)
    ax.set_yticks([])
    ax.set_xticks(np.arange(n_cells) + 0.5)
    ax.set_xticklabels([f"{i}" for i in range(n_cells)], fontsize=7)
    ax.set_xlabel(f'Cell index ({coupon_id})', fontsize=8)

    plt.tight_layout()

    # Save
    fig.savefig(OUT_DIR / "f4_2_coupon_strip.pdf", dpi=DPI, bbox_inches='tight')
    fig.savefig(OUT_DIR / "f4_2_coupon_strip.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    return coupon_id


# ============================================================================
# F4.3: Confusion matrices for each modality
# ============================================================================
def generate_f43_confusion(data):
    """Generate confusion matrix figure for each modality."""

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.5))

    # Vision confusion matrix (from calibration_v2.json)
    vision_cm = np.array(data['vision_cal']['confusion_matrix'])
    vision_labels = ['{P}', '{N,U}', '{N,U,L}']

    ax = axes[0]
    im = ax.imshow(vision_cm, cmap='Blues', aspect='auto')
    for i in range(vision_cm.shape[0]):
        for j in range(vision_cm.shape[1]):
            ax.text(j, i, str(vision_cm[i, j]), ha='center', va='center', fontsize=8)
    ax.set_xticks(range(len(vision_labels)))
    ax.set_yticks(range(len(vision_labels)))
    ax.set_xticklabels(vision_labels, fontsize=7)
    ax.set_yticklabels(vision_labels, fontsize=7)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.set_title('Vision', fontsize=9)
    # Color bar for vision (single-hue)
    ax.patch.set_facecolor(COLORS['vision'])
    ax.patch.set_alpha(0.3)

    # Geometry: No true confusion matrix (outputs {U} or vacuous)
    ax = axes[1]
    # Create a simple summary: measured cells with m({U})=1 vs labels
    geom = data['geometry']
    labels = data['labels']
    measured = geom[geom['is_vacuous'] == False].merge(labels[['cell_id', 'true_class']], on='cell_id')
    if len(measured) > 0:
        # All measured cells output m({U})=1
        geom_summary = measured['true_class'].value_counts()
        classes = ['N', 'U', 'P', 'L']
        counts = [geom_summary.get(c, 0) for c in classes]

        bars = ax.bar(range(len(classes)), counts, color=COLORS['geometry'], edgecolor=COLORS['edge'])
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=7)
        ax.set_xlabel('True class', fontsize=8)
        ax.set_ylabel('Count (all predict U)', fontsize=8)
        ax.set_title('Geometry', fontsize=9)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(cnt), ha='center', va='bottom', fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No measured cells', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Geometry', fontsize=9)

    # UT confusion matrix
    ut_cm = np.array(data['ut_metrics']['confusion_matrix'])
    ut_labels = ['{L}', '{N,U}', '{P}']

    ax = axes[2]
    im = ax.imshow(ut_cm, cmap='YlGn', aspect='auto')
    for i in range(ut_cm.shape[0]):
        for j in range(ut_cm.shape[1]):
            ax.text(j, i, str(ut_cm[i, j]), ha='center', va='center', fontsize=8)
    ax.set_xticks(range(len(ut_labels)))
    ax.set_yticks(range(len(ut_labels)))
    ax.set_xticklabels(ut_labels, fontsize=7)
    ax.set_yticklabels(ut_labels, fontsize=7)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.set_title('Ultrasonic', fontsize=9)

    plt.tight_layout()

    fig.savefig(OUT_DIR / "f4_3_confusion.pdf", dpi=DPI, bbox_inches='tight')
    fig.savefig(OUT_DIR / "f4_3_confusion.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# F4.4: Recall heatmap (per-class recall × conditions, Dempster)
# ============================================================================
def generate_f44_recall_heatmap(data):
    """Generate recall heatmap."""
    # This requires per-class recall for each condition
    # The stored statistics only have F1, not per-class recall
    # We'll need to compute from fusion_results.json or note as NOT AVAILABLE

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 2.5))

    # Placeholder - would need per-condition per-class predictions to compute recall
    conditions = ['V', 'G', 'U', 'V+G', 'V+U', 'G+U', 'V+G+U']
    classes = ['N', 'U', 'P', 'L']

    # Create placeholder data (NOT AVAILABLE)
    ax.text(0.5, 0.5, 'Per-class recall data\nnot available in stored results\n\n'
            'Requires per-cell predictions\nfrom fusion_results.json',
            ha='center', va='center', transform=ax.transAxes, fontsize=9)
    ax.set_title('Recall Heatmap (Dempster)', fontsize=9)
    ax.axis('off')

    plt.tight_layout()

    fig.savefig(OUT_DIR / "f4_4_recall_heatmap.pdf", dpi=DPI, bbox_inches='tight')
    fig.savefig(OUT_DIR / "f4_4_recall_heatmap.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    print("  F4.4: NOTE - Recall heatmap requires per-cell predictions, marked as NOT AVAILABLE")


# ============================================================================
# F4.5: Ablation F1 bar chart with CI
# ============================================================================
def generate_f45_ablation_f1(data):
    """Generate ablation F1 bar chart with confidence intervals."""
    ci_results = data['fusion_stats']['ci_results']

    conditions = ['V', 'G', 'U', 'GV', 'UV', 'GU', 'GUV']
    labels = ['V', 'G', 'U', 'V+G', 'V+U', 'G+U', 'V+G+U']

    f1_vals = []
    ci_lower = []
    ci_upper = []

    for cond in conditions:
        key = f"{cond}_dempster"
        if key in ci_results:
            f1 = ci_results[key]['f1']
            ci = ci_results[key]['ci_95']
            f1_vals.append(f1)
            ci_lower.append(f1 - ci[0])
            ci_upper.append(ci[1] - f1)
        else:
            f1_vals.append(0)
            ci_lower.append(0)
            ci_upper.append(0)

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 2.5))

    x = np.arange(len(labels))
    bars = ax.bar(x, f1_vals, yerr=[ci_lower, ci_upper], capsize=3,
                  color='steelblue', edgecolor=COLORS['edge'], linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Macro-F1', fontsize=9)
    ax.set_xlabel('Condition', fontsize=9)
    ax.set_ylim(0, 0.7)

    # Add value labels
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
               f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()

    fig.savefig(OUT_DIR / "f4_5_ablation_f1.pdf", dpi=DPI, bbox_inches='tight')
    fig.savefig(OUT_DIR / "f4_5_ablation_f1.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    print("Loading data...")
    data = load_data()

    print("Generating F4.2 coupon strip...")
    coupon_id = generate_f42_coupon_strip(data)

    print("Generating F4.3 confusion matrices...")
    generate_f43_confusion(data)

    print("Generating F4.4 recall heatmap...")
    generate_f44_recall_heatmap(data)

    print("Generating F4.5 ablation F1...")
    generate_f45_ablation_f1(data)

    # Update manifest
    manifest_path = OUT_DIR / "MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"files": {}}

    manifest["files"]["f4_2_coupon_strip.pdf"] = f"Coupon strip figure (coupon {coupon_id})"
    manifest["files"]["f4_2_coupon_strip.png"] = f"Coupon strip figure PNG (coupon {coupon_id})"
    manifest["files"]["f4_3_confusion.pdf"] = "Modality confusion matrices"
    manifest["files"]["f4_3_confusion.png"] = "Modality confusion matrices PNG"
    manifest["files"]["f4_4_recall_heatmap.pdf"] = "Recall heatmap (NOT AVAILABLE - placeholder)"
    manifest["files"]["f4_4_recall_heatmap.png"] = "Recall heatmap PNG (NOT AVAILABLE - placeholder)"
    manifest["files"]["f4_5_ablation_f1.pdf"] = "Ablation F1 bar chart with CI"
    manifest["files"]["f4_5_ablation_f1.png"] = "Ablation F1 bar chart PNG"

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
