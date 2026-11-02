#!/usr/bin/env python3
"""
Generate all outputs for §4.1 Data Obtained.

Outputs:
    - t4_1_data.tex / t4_1_data.csv: Fold distribution table
    - f4_1_geometry_coverage.pdf / .png: Geometry coverage figure
    - s4_1_numbers.json: All computed numbers for text
    - MANIFEST.json: Source documentation

Uses v2 labels: data/labels_v2_2026-10-26.csv
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

# Paths
REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "results" / "thesis" / "ch4_evidence"
SCRIPTS_DIR = OUTPUT_DIR / "scripts"

LABELS_V2_PATH = REPO_ROOT / "data" / "labels_v2_2026-10-26.csv"
GEOMETRY_SUMMARY_PATH = REPO_ROOT / "results" / "geometry" / "summary.csv"
UT_FEATURES_PATH = REPO_ROOT / "results" / "ut" / "features.csv"
VISION_MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "coupon_cell_patches" / "manifest.csv"


def load_labels_v2():
    """Load v2 labels."""
    labels = {}
    with open(LABELS_V2_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["cell_id"]] = row
    return labels


def load_geometry_summary():
    """Load geometry summary."""
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


def load_vision_manifest():
    """Load vision patch manifest."""
    vision = {}
    with open(VISION_MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vision[row["cell_id"]] = row
    return vision


def generate_t4_1(labels, geometry, ut_features, vision_manifest):
    """Generate T4.1: Fold distribution table."""
    # Count by fold
    fold_data = defaultdict(lambda: {
        "N": 0, "U": 0, "P": 0, "L": 0,
        "cells": 0, "vision": 0, "geometry": 0, "ut": 0
    })

    for cell_id, lbl in labels.items():
        fold = int(lbl["fold"])
        true_class = lbl["true_class"]

        fold_data[fold]["cells"] += 1
        fold_data[fold][true_class] += 1

        # Vision: has a patch
        if cell_id in vision_manifest:
            fold_data[fold]["vision"] += 1

        # Geometry: is_vacuous == "False"
        if cell_id in geometry and geometry[cell_id]["is_vacuous"] == "False":
            fold_data[fold]["geometry"] += 1

        # UT: has a reading (located == 1)
        if cell_id in ut_features:
            fold_data[fold]["ut"] += 1  # All cells have UT readings

    # Generate LaTeX
    tex_lines = [
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Fold & N & U & P & L & Cells & Vision & Geometry & UT \\",
        r"\midrule",
    ]

    csv_rows = [["Fold", "N", "U", "P", "L", "Cells", "Vision", "Geometry", "UT"]]

    totals = {"N": 0, "U": 0, "P": 0, "L": 0, "cells": 0, "vision": 0, "geometry": 0, "ut": 0}

    for fold in sorted(fold_data.keys()):
        d = fold_data[fold]
        tex_lines.append(
            f"{fold} & {d['N']} & {d['U']} & {d['P']} & {d['L']} & "
            f"{d['cells']} & {d['vision']} & {d['geometry']} & {d['ut']} \\\\"
        )
        csv_rows.append([
            str(fold), str(d['N']), str(d['U']), str(d['P']), str(d['L']),
            str(d['cells']), str(d['vision']), str(d['geometry']), str(d['ut'])
        ])
        for key in totals:
            totals[key] += d[key]

    tex_lines.append(r"\midrule")
    tex_lines.append(
        f"Total & {totals['N']} & {totals['U']} & {totals['P']} & {totals['L']} & "
        f"{totals['cells']} & {totals['vision']} & {totals['geometry']} & {totals['ut']} \\\\"
    )
    csv_rows.append([
        "Total", str(totals['N']), str(totals['U']), str(totals['P']), str(totals['L']),
        str(totals['cells']), str(totals['vision']), str(totals['geometry']), str(totals['ut'])
    ])

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")

    # Write files
    with open(OUTPUT_DIR / "t4_1_data.tex", "w") as f:
        f.write("\n".join(tex_lines))

    with open(OUTPUT_DIR / "t4_1_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    return fold_data, totals


def generate_f4_1(labels, geometry):
    """Generate F4.1: Geometry coverage figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Font setup
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    # Colors
    GEOM_FILL = '#e2fae4'
    GEOM_EDGE = '#5e8f6b'

    # Build data structures
    coupon_cell_measured = defaultdict(dict)  # coupon -> {cell_index -> bool}
    coupons = sorted(set(lbl["coupon_id"] for lbl in labels.values()))

    for cell_id, lbl in labels.items():
        coupon = lbl["coupon_id"]
        cell_idx = int(lbl["cell_index"])
        measured = geometry.get(cell_id, {}).get("is_vacuous", "True") == "False"
        coupon_cell_measured[coupon][cell_idx] = measured

    # (a) Bar chart: fraction of coupons measured at each cell index
    cell_fractions = []
    for idx in range(10):
        measured_count = sum(1 for c in coupons if coupon_cell_measured[c].get(idx, False))
        cell_fractions.append(measured_count / len(coupons))

    # (b) Grid: 15 x 10, coupons as rows
    grid = np.zeros((len(coupons), 10))
    for i, coupon in enumerate(coupons):
        for j in range(10):
            if coupon_cell_measured[coupon].get(j, False):
                grid[i, j] = 1

    # Create figure: full width = 150mm
    fig_width = 150 / 25.4  # inches
    fig_height = 70 / 25.4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height),
                                    gridspec_kw={'width_ratios': [1, 1.3]})

    # Panel (a): Bar chart
    x = np.arange(10)
    bars = ax1.bar(x, [f * 100 for f in cell_fractions], color=GEOM_FILL,
                   edgecolor=GEOM_EDGE, linewidth=0.8)
    ax1.set_xlabel('Cell index')
    ax1.set_ylabel('Coupons measured (%)')
    ax1.set_xticks(x)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(-0.6, 9.6)

    # Zone boundaries
    ax1.axvline(x=2.5, color='#333333', linestyle='--', linewidth=0.6, alpha=0.7)
    ax1.axvline(x=6.5, color='#333333', linestyle='--', linewidth=0.6, alpha=0.7)

    # Zone labels
    ax1.text(1.0, 95, 'A', ha='center', fontsize=8, color='#666666')
    ax1.text(4.5, 95, 'B', ha='center', fontsize=8, color='#666666')
    ax1.text(8.0, 95, 'C', ha='center', fontsize=8, color='#666666')

    ax1.text(-0.02, 1.02, '(a)', transform=ax1.transAxes, fontsize=9, fontweight='bold',
             va='bottom', ha='right')

    # Panel (b): Grid
    # Create custom plot
    for i, coupon in enumerate(coupons):
        for j in range(10):
            if grid[i, j] == 1:
                rect = plt.Rectangle((j - 0.4, len(coupons) - 1 - i - 0.4), 0.8, 0.8,
                                      facecolor=GEOM_FILL, edgecolor=GEOM_EDGE, linewidth=0.6)
            else:
                rect = plt.Rectangle((j - 0.4, len(coupons) - 1 - i - 0.4), 0.8, 0.8,
                                      facecolor='white', edgecolor='#cccccc', linewidth=0.4)
            ax2.add_patch(rect)

    ax2.set_xlim(-0.6, 9.6)
    ax2.set_ylim(-0.6, len(coupons) - 0.4)
    ax2.set_aspect('equal')

    # Labels
    ax2.set_xticks(range(10))
    ax2.set_xlabel('Cell index')

    # WC labels on y-axis
    wc_labels = [f'WC-{int(c[1:]):02d}' for c in coupons]
    ax2.set_yticks(range(len(coupons)))
    ax2.set_yticklabels(wc_labels[::-1], fontsize=7)

    # Zone boundaries
    ax2.axvline(x=2.5, color='#333333', linestyle='--', linewidth=0.6, alpha=0.7)
    ax2.axvline(x=6.5, color='#333333', linestyle='--', linewidth=0.6, alpha=0.7)

    ax2.text(-0.02, 1.02, '(b)', transform=ax2.transAxes, fontsize=9, fontweight='bold',
             va='bottom', ha='right')

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / "f4_1_geometry_coverage.pdf", format='pdf',
                bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUTPUT_DIR / "f4_1_geometry_coverage.png", format='png', dpi=300,
                bbox_inches='tight', pad_inches=0.02)

    plt.close(fig)

    return cell_fractions, grid, coupons


def generate_s4_1_numbers(labels, geometry, ut_features, vision_manifest, fold_data, totals):
    """Generate s4_1_numbers.json with all computed values."""

    numbers = {
        "source_files": {
            "labels_v2": str(LABELS_V2_PATH.relative_to(REPO_ROOT)),
            "geometry_summary": str(GEOMETRY_SUMMARY_PATH.relative_to(REPO_ROOT)),
            "ut_features": str(UT_FEATURES_PATH.relative_to(REPO_ROOT)),
            "vision_manifest": str(VISION_MANIFEST_PATH.relative_to(REPO_ROOT)),
        },
        "label_distribution_v2": {
            "N": totals["N"],
            "U": totals["U"],
            "P": totals["P"],
            "L": totals["L"],
            "total": totals["cells"],
        },
    }

    # 1. Total measured geometry cells and count per fold
    geom_by_fold = {}
    geom_total = 0
    for fold in sorted(fold_data.keys()):
        geom_by_fold[f"fold_{fold}"] = fold_data[fold]["geometry"]
        geom_total += fold_data[fold]["geometry"]

    numbers["geometry_coverage"] = {
        "total_measured": geom_total,
        "total_vacuous": totals["cells"] - geom_total,
        "per_fold": geom_by_fold,
    }

    # 2. Measured fraction per cell index 0-9
    cell_index_data = defaultdict(lambda: {"total": 0, "measured": 0})
    coupons_with_zero = []

    for cell_id, lbl in labels.items():
        idx = int(lbl["cell_index"])
        coupon = lbl["coupon_id"]
        cell_index_data[idx]["total"] += 1
        if geometry.get(cell_id, {}).get("is_vacuous", "True") == "False":
            cell_index_data[idx]["measured"] += 1

    cell_fractions = {}
    for idx in range(10):
        frac = cell_index_data[idx]["measured"] / cell_index_data[idx]["total"]
        cell_fractions[f"index_{idx}"] = {
            "measured": cell_index_data[idx]["measured"],
            "total": cell_index_data[idx]["total"],
            "fraction": round(frac, 4),
        }

    # Find coupons with zero measured cells
    coupon_measured = defaultdict(int)
    for cell_id, lbl in labels.items():
        if geometry.get(cell_id, {}).get("is_vacuous", "True") == "False":
            coupon_measured[lbl["coupon_id"]] += 1

    all_coupons = sorted(set(lbl["coupon_id"] for lbl in labels.values()))
    coupons_with_zero = [c for c in all_coupons if coupon_measured[c] == 0]

    numbers["geometry_by_cell_index"] = {
        "fractions": cell_fractions,
        "coupons_with_zero_measured": coupons_with_zero,
    }

    # 3. Geometry coverage by v2 label with chi-squared test
    label_geom = defaultdict(lambda: {"measured": 0, "vacuous": 0})
    for cell_id, lbl in labels.items():
        true_class = lbl["true_class"]
        measured = geometry.get(cell_id, {}).get("is_vacuous", "True") == "False"
        if measured:
            label_geom[true_class]["measured"] += 1
        else:
            label_geom[true_class]["vacuous"] += 1

    # Chi-squared test
    observed = []
    for cls in ["N", "U", "P", "L"]:
        observed.append([label_geom[cls]["measured"], label_geom[cls]["vacuous"]])
    observed = np.array(observed)

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

    # Check for expected counts < 5
    expected_below_5 = []
    for i, cls in enumerate(["N", "U", "P", "L"]):
        for j, cat in enumerate(["measured", "vacuous"]):
            if expected[i, j] < 5:
                expected_below_5.append(f"{cls}_{cat}: {expected[i, j]:.2f}")

    numbers["geometry_by_label"] = {
        "counts": {
            "N": {"measured": label_geom["N"]["measured"], "vacuous": label_geom["N"]["vacuous"]},
            "U": {"measured": label_geom["U"]["measured"], "vacuous": label_geom["U"]["vacuous"]},
            "P": {"measured": label_geom["P"]["measured"], "vacuous": label_geom["P"]["vacuous"]},
            "L": {"measured": label_geom["L"]["measured"], "vacuous": label_geom["L"]["vacuous"]},
        },
        "percentages": {
            cls: round(100 * label_geom[cls]["measured"] /
                      (label_geom[cls]["measured"] + label_geom[cls]["vacuous"]), 1)
            for cls in ["N", "U", "P", "L"]
        },
        "chi_squared": {
            "statistic": round(chi2_stat, 3),
            "df": dof,
            "p_value": round(p_value, 4),
            "expected_counts": {
                "N_measured": round(expected[0, 0], 2),
                "N_vacuous": round(expected[0, 1], 2),
                "U_measured": round(expected[1, 0], 2),
                "U_vacuous": round(expected[1, 1], 2),
                "P_measured": round(expected[2, 0], 2),
                "P_vacuous": round(expected[2, 1], 2),
                "L_measured": round(expected[3, 0], 2),
                "L_vacuous": round(expected[3, 1], 2),
            },
            "expected_below_5": expected_below_5 if expected_below_5 else "none",
        },
    }

    # 4. UT data
    epoch_ids = [int(ut["epoch_id"]) for ut in ut_features.values()]
    located_count = sum(1 for ut in ut_features.values() if int(ut["located"]) == 1)
    saturated_count = sum(1 for ut in ut_features.values()
                          if float(ut["amplitude_pct"]) >= 100)

    numbers["ut_data"] = {
        "total_readings": len(ut_features),
        "epoch_id_range": [min(epoch_ids), max(epoch_ids)],
        "cells_with_gated_indication": located_count,
        "saturated_count": saturated_count,
        "dropped_reading": {
            "note": "ID 089 was an extra reading beyond cell 09 on WC-07",
            "source": "src/ut/features.py:90-123",
        },
    }

    # 5. Vision data
    vision_cells = set(vision_manifest.keys())
    label_cells = set(labels.keys())
    cells_without_patch = label_cells - vision_cells

    numbers["vision_data"] = {
        "cells_with_patch": len(vision_cells),
        "cells_without_patch": list(cells_without_patch) if cells_without_patch else "none",
    }

    # 6. Capture metadata - check if stored
    numbers["capture_metadata"] = {
        "captures_per_coupon": "NOT AVAILABLE - capture metadata not stored in results",
        "standoff_range": "NOT AVAILABLE - capture metadata not stored in results",
    }

    # Write JSON
    with open(OUTPUT_DIR / "s4_1_numbers.json", "w") as f:
        json.dump(numbers, f, indent=2)

    return numbers


def generate_manifest(numbers):
    """Generate MANIFEST.json."""
    manifest = {
        "section": "4.1",
        "title": "Data obtained",
        "generated": "2026-10-29",
        "outputs": {
            "t4_1_data.tex": {
                "description": "Fold distribution table (LaTeX)",
                "sources": [
                    "data/labels_v2_2026-10-26.csv",
                    "results/geometry/summary.csv",
                    "results/ut/features.csv",
                    "data/processed/coupon_cell_patches/manifest.csv",
                ],
            },
            "t4_1_data.csv": {
                "description": "Fold distribution table (CSV)",
                "sources": ["same as t4_1_data.tex"],
            },
            "f4_1_geometry_coverage.pdf": {
                "description": "Geometry coverage figure",
                "sources": [
                    "data/labels_v2_2026-10-26.csv",
                    "results/geometry/summary.csv",
                ],
            },
            "f4_1_geometry_coverage.png": {
                "description": "Geometry coverage figure (PNG)",
                "sources": ["same as f4_1_geometry_coverage.pdf"],
            },
            "s4_1_numbers.json": {
                "description": "All computed numbers for text",
                "sources": [
                    "data/labels_v2_2026-10-26.csv",
                    "results/geometry/summary.csv",
                    "results/ut/features.csv",
                    "data/processed/coupon_cell_patches/manifest.csv",
                ],
            },
        },
        "numbers": {
            "label_distribution": {
                "N": numbers["label_distribution_v2"]["N"],
                "U": numbers["label_distribution_v2"]["U"],
                "P": numbers["label_distribution_v2"]["P"],
                "L": numbers["label_distribution_v2"]["L"],
                "source": "data/labels_v2_2026-10-26.csv",
            },
            "geometry_measured": {
                "value": numbers["geometry_coverage"]["total_measured"],
                "source": "results/geometry/summary.csv:is_vacuous column",
            },
            "ut_gated_indications": {
                "value": numbers["ut_data"]["cells_with_gated_indication"],
                "source": "results/ut/features.csv:located column",
            },
            "ut_saturated": {
                "value": numbers["ut_data"]["saturated_count"],
                "source": "results/ut/features.csv:amplitude_pct >= 100",
            },
            "vision_cells": {
                "value": numbers["vision_data"]["cells_with_patch"],
                "source": "data/processed/coupon_cell_patches/manifest.csv",
            },
        },
    }

    with open(OUTPUT_DIR / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    print("Generating §4.1 Data Obtained outputs...")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load data
    print("Loading data...")
    labels = load_labels_v2()
    geometry = load_geometry_summary()
    ut_features = load_ut_features()
    vision_manifest = load_vision_manifest()

    # Verify label counts
    class_counts = defaultdict(int)
    for lbl in labels.values():
        class_counts[lbl["true_class"]] += 1
    print(f"Label distribution (v2): N={class_counts['N']}, U={class_counts['U']}, "
          f"P={class_counts['P']}, L={class_counts['L']}")

    # Generate T4.1
    print("Generating T4.1...")
    fold_data, totals = generate_t4_1(labels, geometry, ut_features, vision_manifest)
    print(f"  Total cells: {totals['cells']}")
    print(f"  Geometry measured: {totals['geometry']}")

    # Generate F4.1
    print("Generating F4.1...")
    cell_fractions, grid, coupons = generate_f4_1(labels, geometry)
    print(f"  Cell index fractions: {[f'{f:.2f}' for f in cell_fractions]}")

    # Generate s4_1_numbers.json
    print("Generating s4_1_numbers.json...")
    numbers = generate_s4_1_numbers(labels, geometry, ut_features, vision_manifest,
                                     fold_data, totals)

    # Generate MANIFEST.json
    print("Generating MANIFEST.json...")
    manifest = generate_manifest(numbers)

    print("\n=== SUMMARY ===")
    print(f"Label file: {LABELS_V2_PATH.relative_to(REPO_ROOT)}")
    print(f"Labels (v2): N={totals['N']}, U={totals['U']}, P={totals['P']}, L={totals['L']}")
    print(f"Total cells: {totals['cells']}")
    print(f"Vision cells: {totals['vision']}")
    print(f"Geometry measured: {totals['geometry']}")
    print(f"UT readings: {totals['ut']}")
    print(f"Chi-squared (geometry vs label): chi2={numbers['geometry_by_label']['chi_squared']['statistic']}, "
          f"p={numbers['geometry_by_label']['chi_squared']['p_value']}")

    print("\nOutputs written:")
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file() and f.suffix in ['.tex', '.csv', '.pdf', '.png', '.json']:
            print(f"  {f.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
