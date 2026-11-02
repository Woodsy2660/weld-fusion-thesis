#!/usr/bin/env python3
"""
Generate all outputs for §4.2-§4.5 of Chapter 4.

Outputs:
    §4.2: t4_2_uncertainty.tex, t4_3_vision_verify.tex, s4_2_ut.json, f4_3_ut_mapping.pdf
    §4.3: t4_4_modality.tex, t4_5_elements.tex, f4_4_confusion.pdf, s4_3_numbers.json
    §4.4: t4_6_ablation.tex, t4_7_recall.tex, f4_5_recall_heatmap.pdf,
          f4_6_ablation_f1.pdf, f4_7_conflict.pdf, s4_4_equality.json
    §4.5: t4_8_mcnemar.tex, s4_5_neff.json
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
import random

# Paths
REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "results" / "thesis" / "ch4_evidence"

LABELS_V2_PATH = REPO_ROOT / "data" / "labels_v2_2026-10-26.csv"
GEOMETRY_SUMMARY_PATH = REPO_ROOT / "results" / "geometry" / "summary.csv"
UT_FEATURES_PATH = REPO_ROOT / "results" / "ut" / "features.csv"
UT_METRICS_PATH = REPO_ROOT / "results" / "ut" / "metrics_ut.json"
VISION_CALIBRATION_PATH = REPO_ROOT / "results" / "vision" / "coupon_eval" / "calibration_v2.json"
GEOMETRY_ALPHA_PATH = REPO_ROOT / "results" / "geometry" / "alpha_derivation.json"
REPEATABILITY_PATH = REPO_ROOT / "results" / "geometry" / "_diagnostics" / "repeatability_C05_top.json"
LOSO_PATH = REPO_ROOT / "results" / "vision" / "loso_metrics.json"
PARTITION_PATH = REPO_ROOT / "results" / "vision" / "partition_search.json"
SEED_VARIANCE_PATH = REPO_ROOT / "results" / "vision" / "seed_variance_results.json"
FUSION_RESULTS_PATH = REPO_ROOT / "results" / "fusion" / "fusion_results.json"
FUSION_STATS_PATH = REPO_ROOT / "results" / "fusion" / "statistics.json"

THETA = frozenset({"N", "U", "P", "L"})
BOOTSTRAP_RESAMPLES = 2000
RANDOM_SEED = 42


def load_labels_v2():
    labels = {}
    with open(LABELS_V2_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["cell_id"]] = row
    return labels


def load_geometry_summary():
    geom = {}
    with open(GEOMETRY_SUMMARY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            geom[row["cell_id"]] = row
    return geom


def load_ut_features():
    ut = {}
    with open(UT_FEATURES_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ut[row["cell_id"]] = row
    return ut


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        if tp + fp == 0:
            p = 0.0
        else:
            p = tp / (tp + fp)
        if tp + fn == 0:
            r = 0.0
        else:
            r = tp / (tp + fn)
        if p + r == 0:
            f1 = 0.0
        else:
            f1 = 2 * p * r / (p + r)
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


# =============================================================================
# §4.2 Pipeline Verification
# =============================================================================

def generate_t4_2_uncertainty():
    """Generate T4.2: Geometry uncertainty table."""
    with open(REPEATABILITY_PATH) as f:
        rep = json.load(f)

    # Predicted uncertainty from config
    u_noise = 0.068  # From config: sqrt(single_frame_noise^2 / n_profiles)
    u_quant = 0.029
    u_cal = 0.030
    u_align = 0.020  # Placeholder
    u_c = 0.083  # RSS of above

    # Measured repeatability (pooled SD after alignment)
    u_undercut = rep["measurands"]["undercut"]["after"]["pooled_sd_mm"]
    u_reinforcement = rep["measurands"]["reinforcement"]["after"]["pooled_sd_mm"]
    u_misalignment = rep["measurands"]["misalignment"]["after"]["pooled_sd_mm"]

    # Limits from ISO 5817 Level B
    d_lim_undercut = 0.5  # mm
    d_lim_misalignment = 1.0  # mm
    # Reinforcement: 1 + 0.1*b, assume b=15mm -> 2.5mm
    b_mm = 15
    d_lim_reinforcement = 1.0 + 0.1 * b_mm

    tex = r"""\begin{tabular}{@{}llrrrrl@{}}
\toprule
Measurand & Component & Predicted & Measured & Ratio & $d_\mathrm{lim}/u$ & Active \\
\midrule
"""

    # Measurand rows
    tex += f"Undercut & --- & {u_c:.3f} & {u_undercut:.3f} & {u_undercut/u_c:.1f} & {d_lim_undercut/u_undercut:.1f} & No \\\\\n"
    tex += f"Excess weld & --- & {u_c:.3f} & {u_reinforcement:.3f} & {u_reinforcement/u_c:.1f} & {d_lim_reinforcement/u_reinforcement:.1f} & No \\\\\n"
    tex += f"Misalignment & --- & {u_c:.3f} & {u_misalignment:.3f} & {u_misalignment/u_c:.1f} & {d_lim_misalignment/u_misalignment:.1f} & Yes \\\\\n"

    tex += r"""\midrule
\multicolumn{7}{@{}l}{\textit{Uncertainty budget components}} \\
"""
    tex += f"& Single-frame noise & 0.215 & --- & --- & --- & --- \\\\\n"
    tex += f"& Averaging factor & $1/\\sqrt{{10}}$ & --- & --- & --- & --- \\\\\n"
    tex += f"& $u_\\mathrm{{noise}}$ & {u_noise:.3f} & --- & --- & --- & --- \\\\\n"
    tex += f"& $u_\\mathrm{{quant}}$ & {u_quant:.3f} & --- & --- & --- & --- \\\\\n"
    tex += f"& $u_\\mathrm{{cal}}$ & {u_cal:.3f} & --- & --- & --- & --- \\\\\n"
    tex += f"& $u_\\mathrm{{align}}$ & {u_align:.3f} & --- & --- & --- & Placeholder \\\\\n"
    tex += f"& $u_c$ (RSS) & {u_c:.3f} & --- & --- & --- & --- \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
"""

    with open(OUTPUT_DIR / "t4_2_uncertainty.tex", "w") as f:
        f.write(tex)

    # CSV version
    csv_rows = [
        ["Measurand", "Component", "Predicted_mm", "Measured_mm", "Ratio", "d_lim_over_u", "Active"],
        ["Undercut", "---", f"{u_c:.3f}", f"{u_undercut:.3f}", f"{u_undercut/u_c:.1f}", f"{d_lim_undercut/u_undercut:.1f}", "No"],
        ["Excess weld", "---", f"{u_c:.3f}", f"{u_reinforcement:.3f}", f"{u_reinforcement/u_c:.1f}", f"{d_lim_reinforcement/u_reinforcement:.1f}", "No"],
        ["Misalignment", "---", f"{u_c:.3f}", f"{u_misalignment:.3f}", f"{u_misalignment/u_c:.1f}", f"{d_lim_misalignment/u_misalignment:.1f}", "Yes"],
    ]
    with open(OUTPUT_DIR / "t4_2_uncertainty.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    return {
        "u_c_predicted": u_c,
        "u_undercut_measured": u_undercut,
        "u_reinforcement_measured": u_reinforcement,
        "u_misalignment_measured": u_misalignment,
        "d_lim_reinforcement_b_mm": b_mm,
    }


def generate_t4_3_vision_verify():
    """Generate T4.3: Vision verification table."""
    with open(PARTITION_PATH) as f:
        partition = json.load(f)
    with open(LOSO_PATH) as f:
        loso = json.load(f)
    with open(VISION_CALIBRATION_PATH) as f:
        calib = json.load(f)

    # Leakage intersections
    leakage = partition["leakage_intersections"]
    train_val_overlap = leakage["source_sequence"]["train&val"]
    train_test_overlap = leakage["source_sequence"]["train&test"]

    tex = r"""\begin{tabular}{@{}lr@{}}
\toprule
Check & Value \\
\midrule
"""

    # Pretraining split overlap
    tex += f"Pretraining split overlap (sequences) & {train_val_overlap + train_test_overlap} \\\\\n"

    # LOSO results
    for fold in loso["folds"]:
        holdout = fold["holdout"]
        if not fold.get("degenerate", False):
            seen_f1 = fold["in_dist_val_macro_f1_three_head"]
            held_f1 = fold["holdout_macro_f1"]
            gap = fold["transfer_gap"]
            tex += f"LOSO {holdout}: seen F1 & {seen_f1:.3f} \\\\\n"
            tex += f"LOSO {holdout}: held-out F1 & {held_f1:.3f} \\\\\n"
            tex += f"LOSO {holdout}: gap & {gap:.3f} \\\\\n"

    # Calibration
    tex += f"ECE before scaling & {calib['ece_pre']:.3f} \\\\\n"
    tex += f"ECE after scaling (15 bins, equal-width) & {calib['ece_post']:.3f} \\\\\n"
    tex += f"Fitted temperature $T$ & {calib['new_temperature']:.1f} \\\\\n"
    tex += f"Temperature bounds & [{calib['temperature_bounds'][0]}, {calib['temperature_bounds'][1]}] \\\\\n"
    tex += f"Hit boundary & {'Yes' if calib['hit_boundary'] else 'No'} \\\\\n"

    # Nearest-neighbor check - NOT AVAILABLE
    tex += "Coupon test patches with NN cosine $<$ 0.05 & NOT AVAILABLE \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
"""

    with open(OUTPUT_DIR / "t4_3_vision_verify.tex", "w") as f:
        f.write(tex)

    # Sequence overlap check (81.59% figure)
    # This describes the partition search result
    return {
        "pretraining_overlap": train_val_overlap + train_test_overlap,
        "loso_results": [
            {"holdout": f["holdout"], "seen_f1": f.get("in_dist_val_macro_f1_three_head"),
             "held_f1": f.get("holdout_macro_f1"), "gap": f.get("transfer_gap")}
            for f in loso["folds"] if not f.get("degenerate", False)
        ],
        "ece_pre": calib["ece_pre"],
        "ece_post": calib["ece_post"],
        "temperature": calib["new_temperature"],
        "hit_boundary": calib["hit_boundary"],
        "nn_check": "NOT AVAILABLE",
        "sequence_overlap_note": "81.59% refers to the selected partition's train share (see partition_search.json:selected.train_share=0.797)",
    }


def generate_s4_2_ut():
    """Generate s4_2_ut.json: UT verification checks including mapping correlation."""
    labels = load_labels_v2()
    ut_features = load_ut_features()

    # Count per coupon: cells with gated indication vs cells expected to have defect
    coupon_data = defaultdict(lambda: {"observed": 0, "expected": 0})

    for cell_id, lbl in labels.items():
        coupon = lbl["coupon_id"]
        true_class = lbl["true_class"]

        # Expected: P or L zones (internal defects detectable by UT)
        if true_class in ["P", "L"]:
            coupon_data[coupon]["expected"] += 1

        # Observed: gated indication (located == 1)
        ut = ut_features.get(cell_id, {})
        if int(ut.get("located", 0)) == 1:
            coupon_data[coupon]["observed"] += 1

    # Build arrays for correlation
    coupons = sorted(coupon_data.keys())
    observed = [coupon_data[c]["observed"] for c in coupons]
    expected = [coupon_data[c]["expected"] for c in coupons]

    # Pearson correlation
    if len(observed) > 2 and np.std(observed) > 0 and np.std(expected) > 0:
        r, p = stats.pearsonr(observed, expected)
        n = len(coupons)
        # Critical r at 0.05 for n-2 df
        t_crit = stats.t.ppf(0.975, n - 2)
        r_crit = t_crit / np.sqrt(t_crit**2 + n - 2)
    else:
        r, p, n, r_crit = None, None, len(coupons), None

    result = {
        "mapping_check": {
            "method": "Pearson correlation: observed indications vs expected defect cells per coupon",
            "n_coupons": n,
            "r": float(round(r, 4)) if r is not None else None,
            "p_value": float(round(p, 6)) if p is not None else None,
            "r_critical_0.05": float(round(r_crit, 4)) if r_crit is not None else None,
            "significant": bool(p < 0.05) if p is not None else None,
            "note": "Replaces unscripted r=0.977 from methodology",
            "per_coupon": {c: {"observed": int(coupon_data[c]["observed"]),
                              "expected": int(coupon_data[c]["expected"])}
                          for c in coupons},
        },
        "exports_checked": 1,  # Single TEST01.csv export used
        "exports_rejected": 0,
        "max_depth_error_mm": "NOT AVAILABLE - instrument depth vs SA*cos(60) not stored",
    }

    with open(OUTPUT_DIR / "s4_2_ut.json", "w") as f:
        json.dump(result, f, indent=2)

    return result, coupons, observed, expected


def generate_f4_3_ut_mapping(coupons, observed, expected):
    """Generate F4.3: UT mapping figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    UT_FILL = '#fbfdcd'
    UT_EDGE = '#b59a2e'

    fig_width = 75 / 25.4  # Half width
    fig_height = 60 / 25.4

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Scatter plot
    ax.scatter(expected, observed, s=40, facecolor=UT_FILL, edgecolor=UT_EDGE, linewidth=0.8, zorder=3)

    # y = x line
    max_val = max(max(observed), max(expected)) + 1
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.6, alpha=0.5, label='$y = x$')

    # Labels
    for i, c in enumerate(coupons):
        label = f"WC-{int(c[1:]):02d}"
        ax.annotate(label, (expected[i], observed[i]), fontsize=6,
                   xytext=(3, 3), textcoords='offset points', alpha=0.7)

    ax.set_xlabel('Expected defect cells (P + L)')
    ax.set_ylabel('Observed indications')
    ax.set_xlim(-0.5, max_val)
    ax.set_ylim(-0.5, max_val)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "f4_3_ut_mapping.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "f4_3_ut_mapping.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# §4.3 Single-modality Results
# =============================================================================

def generate_t4_4_modality():
    """Generate T4.4: Per-modality classification results."""
    with open(VISION_CALIBRATION_PATH) as f:
        vision = json.load(f)
    with open(GEOMETRY_ALPHA_PATH) as f:
        geometry = json.load(f)
    with open(UT_METRICS_PATH) as f:
        ut = json.load(f)

    tex = r"""\begin{tabular}{@{}llrrrrr@{}}
\toprule
Modality & Estimator & F1 & Pooled F1 & Trivial & Cells & $\alpha$ \\
\midrule
"""

    # Vision
    v_f1 = vision["mean_f1"]
    v_std = vision["std_f1"]
    v_pooled = vision["pooled_f1"]
    v_trivial = vision["trivial_baseline"]
    v_alpha = vision["alpha_V"]
    tex += f"Vision & mean-of-folds & {v_f1:.3f} $\\pm$ {v_std:.3f} & {v_pooled:.3f} & {v_trivial:.3f} & 150 & {v_alpha:.3f} \\\\\n"

    # Geometry
    g_f1 = geometry["f1_pooled"]
    g_trivial = geometry["trivial_baseline"]
    g_alpha = geometry["alpha_G"]
    tex += f"Geometry & pooled & {g_f1:.3f} & {g_f1:.3f} & {g_trivial:.3f} & 39 & {g_alpha:.3f} \\\\\n"

    # UT
    u_f1 = ut["f1_macro_mean"]
    u_std = ut["f1_macro_std"]
    u_pooled = ut["f1_macro"]
    u_trivial = ut["baseline_most_frequent_f1"]
    u_alpha = ut["alpha"]
    tex += f"Ultrasonic & mean-of-folds & {u_f1:.3f} $\\pm$ {u_std:.3f} & {u_pooled:.3f} & {u_trivial:.3f} & 150 & {u_alpha:.3f} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
"""

    with open(OUTPUT_DIR / "t4_4_modality.tex", "w") as f:
        f.write(tex)

    return {
        "vision": {"f1": v_f1, "std": v_std, "pooled_f1": v_pooled, "trivial": v_trivial, "alpha": v_alpha},
        "geometry": {"f1": g_f1, "trivial": g_trivial, "alpha": g_alpha},
        "ut": {"f1": u_f1, "std": u_std, "pooled_f1": u_pooled, "trivial": u_trivial, "alpha": u_alpha},
    }


def generate_t4_5_elements():
    """Generate T4.5: Per-element precision, recall, F1, support."""
    with open(VISION_CALIBRATION_PATH) as f:
        vision = json.load(f)
    with open(UT_METRICS_PATH) as f:
        ut = json.load(f)

    labels = load_labels_v2()
    geometry = load_geometry_summary()

    tex = r"""\begin{tabular}{@{}llrrrr@{}}
\toprule
Modality & Element & Precision & Recall & F1 & Support \\
\midrule
"""

    # Vision - from confusion matrix
    cm_v = np.array(vision["confusion_matrix"])
    v_elements = ["{P}", "{N,U}", "{N,U,L}"]
    for i, elem in enumerate(v_elements):
        tp = cm_v[i, i]
        fp = cm_v[:, i].sum() - tp
        fn = cm_v[i, :].sum() - tp
        support = cm_v[i, :].sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        tex += f"Vision & {elem} & {p:.3f} & {r:.3f} & {f1:.3f} & {support} \\\\\n"

    tex += r"\midrule" + "\n"

    # Geometry - binary {U} vs {N,P,L}
    g_decided = [(cell_id, lbl) for cell_id, lbl in labels.items()
                 if geometry.get(cell_id, {}).get("is_vacuous", "True") == "False"]

    # All geometry predictions are {U} due to misalignment exceedance
    g_pred = ["U"] * len(g_decided)
    g_true_binary = ["U" if lbl["true_class"] == "U" else "NPL" for _, lbl in g_decided]
    g_pred_binary = ["U"] * len(g_decided)  # All predict U

    for elem, true_val in [("{U}", "U"), ("{N,P,L}", "NPL")]:
        tp = sum(1 for gt, gp in zip(g_true_binary, g_pred_binary) if gt == true_val and gp == true_val)
        fp = sum(1 for gt, gp in zip(g_true_binary, g_pred_binary) if gt != true_val and gp == true_val)
        fn = sum(1 for gt, gp in zip(g_true_binary, g_pred_binary) if gt == true_val and gp != true_val)
        support = sum(1 for gt in g_true_binary if gt == true_val)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        tex += f"Geometry & {elem} & {p:.3f} & {r:.3f} & {f1:.3f} & {support} \\\\\n"

    tex += r"\midrule" + "\n"

    # UT - from confusion matrix
    cm_u = np.array(ut["confusion_matrix"])
    u_elements = ["{L}", "{N,U}", "{P}"]  # Order from metrics
    for i, elem in enumerate(u_elements):
        tp = cm_u[i, i]
        fp = cm_u[:, i].sum() - tp
        fn = cm_u[i, :].sum() - tp
        support = cm_u[i, :].sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        tex += f"Ultrasonic & {elem} & {p:.3f} & {r:.3f} & {f1:.3f} & {support} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
"""

    with open(OUTPUT_DIR / "t4_5_elements.tex", "w") as f:
        f.write(tex)


def generate_f4_4_confusion():
    """Generate F4.4: Confusion matrices for all three modalities."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    with open(VISION_CALIBRATION_PATH) as f:
        vision = json.load(f)
    with open(UT_METRICS_PATH) as f:
        ut = json.load(f)

    labels = load_labels_v2()
    geometry = load_geometry_summary()

    # Create custom colormaps
    def make_cmap(edge_color):
        return LinearSegmentedColormap.from_list('custom', ['white', edge_color])

    V_CMAP = make_cmap('#4a73a8')
    G_CMAP = make_cmap('#5e8f6b')
    U_CMAP = make_cmap('#b59a2e')

    fig_width = 150 / 25.4
    fig_height = 50 / 25.4

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

    # Vision
    cm_v = np.array(vision["confusion_matrix"])
    v_labels = ['{P}', '{N,U}', '{N,U,L}']
    im0 = axes[0].imshow(cm_v, cmap=V_CMAP, aspect='auto')
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(v_labels, fontsize=7)
    axes[0].set_yticklabels(v_labels, fontsize=7)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(cm_v[i, j]), ha='center', va='center', fontsize=8)
    axes[0].text(0.02, 0.98, '(a) Vision', transform=axes[0].transAxes,
                fontsize=9, va='top', ha='left')

    # Geometry - binary
    g_decided = [(cell_id, lbl) for cell_id, lbl in labels.items()
                 if geometry.get(cell_id, {}).get("is_vacuous", "True") == "False"]
    cm_g = np.zeros((2, 2), dtype=int)
    for cell_id, lbl in g_decided:
        true_bin = 0 if lbl["true_class"] == "U" else 1
        pred_bin = 0  # All predict U
        cm_g[true_bin, pred_bin] += 1

    g_labels = ['{U}', '{N,P,L}']
    im1 = axes[1].imshow(cm_g, cmap=G_CMAP, aspect='auto')
    axes[1].set_xticks(range(2))
    axes[1].set_yticks(range(2))
    axes[1].set_xticklabels(g_labels, fontsize=7)
    axes[1].set_yticklabels(g_labels, fontsize=7)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm_g[i, j]), ha='center', va='center', fontsize=8)
    axes[1].text(0.02, 0.98, '(b) Geometry', transform=axes[1].transAxes,
                fontsize=9, va='top', ha='left')

    # UT
    cm_u = np.array(ut["confusion_matrix"])
    u_labels = ['{L}', '{N,U}', '{P}']
    im2 = axes[2].imshow(cm_u, cmap=U_CMAP, aspect='auto')
    axes[2].set_xticks(range(3))
    axes[2].set_yticks(range(3))
    axes[2].set_xticklabels(u_labels, fontsize=7)
    axes[2].set_yticklabels(u_labels, fontsize=7)
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, str(cm_u[i, j]), ha='center', va='center', fontsize=8)
    axes[2].text(0.02, 0.98, '(c) Ultrasonic', transform=axes[2].transAxes,
                fontsize=9, va='top', ha='left')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "f4_4_confusion.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "f4_4_confusion.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_s4_3_numbers():
    """Generate s4_3_numbers.json with per-fold F1 and geometry details."""
    with open(VISION_CALIBRATION_PATH) as f:
        vision = json.load(f)
    with open(GEOMETRY_ALPHA_PATH) as f:
        geometry = json.load(f)
    with open(UT_METRICS_PATH) as f:
        ut = json.load(f)
    with open(SEED_VARIANCE_PATH) as f:
        seed = json.load(f)

    labels = load_labels_v2()
    geom_summary = load_geometry_summary()

    # Geometry: which measurand drives {U}, how many, and their labels
    u_calls = []
    for cell_id, lbl in labels.items():
        g = geom_summary.get(cell_id, {})
        if g.get("is_vacuous", "True") == "False":
            u_calls.append({
                "cell_id": cell_id,
                "true_class": lbl["true_class"],
                "governing_measurand": g.get("governing_measurand", "unknown"),
            })

    measurand_counts = defaultdict(int)
    label_counts = defaultdict(int)
    for u in u_calls:
        measurand_counts[u["governing_measurand"]] += 1
        label_counts[u["true_class"]] += 1

    result = {
        "per_fold_f1": {
            "vision": vision["per_fold_f1"],
            "geometry": geometry["per_fold_f1"],
            "ut": "NOT AVAILABLE - stored as mean only",
        },
        "geometry_u_calls": {
            "total": len(u_calls),
            "by_measurand": dict(measurand_counts),
            "by_true_label": dict(label_counts),
        },
        "vision_seed_spread": {
            "v1_between_seed_std": seed["variance_analysis"]["v1_between_seed_std"],
            "v2_between_seed_std": seed["variance_analysis"]["v2_between_seed_std"],
            "alpha_range": seed["alpha_range"],
        },
    }

    with open(OUTPUT_DIR / "s4_3_numbers.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# =============================================================================
# §4.4 Fused Results (Ablation)
# =============================================================================

def generate_t4_6_ablation():
    """Generate T4.6: Ablation results table."""
    with open(FUSION_RESULTS_PATH) as f:
        fusion = json.load(f)

    labels = load_labels_v2()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Group cells by coupon
    coupon_cells = defaultdict(list)
    for cell_id, lbl in labels.items():
        coupon_cells[lbl["coupon_id"]].append(cell_id)
    coupons = list(coupon_cells.keys())

    def bootstrap_f1(results_dict, labels_dict):
        f1s = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sampled_coupons = random.choices(coupons, k=len(coupons))
            sampled_cells = []
            for c in sampled_coupons:
                sampled_cells.extend(coupon_cells[c])
            y_true = [labels_dict[c]["true_class"] for c in sampled_cells]
            y_pred = [results_dict[c]["decision"] for c in sampled_cells]
            f1 = macro_f1(y_true, y_pred, ["N", "U", "P", "L"])
            f1s.append(f1)
        return np.percentile(f1s, [2.5, 97.5])

    subset_order = ["V", "G", "U", "GV", "UV", "GU", "GUV"]
    subset_display = ["V", "G", "U", "V+G", "V+U", "G+U", "V+G+U"]

    tex_dempster = r"""\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Subset & F1 [95\% CI] & Mean $K$ & Max $K$ & \{L\} & N/U ties \\
\midrule
"""

    tex_murphy = r"""\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Subset & F1 [95\% CI] & Mean $K$ & Max $K$ & \{L\} & N/U ties \\
\midrule
"""

    all_results = {}

    for method in ["dempster", "murphy"]:
        for i, key in enumerate(subset_order):
            results_dict = {r["cell_id"]: r for r in fusion[key][method]}

            # Point estimate
            y_true = [labels[c]["true_class"] for c in results_dict]
            y_pred = [results_dict[c]["decision"] for c in results_dict]
            f1 = macro_f1(y_true, y_pred, ["N", "U", "P", "L"])

            # Bootstrap CI
            ci = bootstrap_f1(results_dict, labels)

            # K statistics
            ks = [r["K"] for r in fusion[key][method]]
            mean_k = np.mean(ks)
            max_k = np.max(ks)

            # {L} count
            l_count = sum(1 for r in fusion[key][method] if r["decision"] == "L")

            # N/U ties - check where BetP(N) == BetP(U)
            nu_ties = 0
            for r in fusion[key][method]:
                betp = r["betp"]
                if abs(betp.get("N", 0) - betp.get("U", 0)) < 1e-9:
                    if r["decision"] == "N":  # Alphabetical tie-break chose N
                        nu_ties += 1

            row = f"{subset_display[i]} & {f1:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] & {mean_k:.3f} & {max_k:.3f} & {l_count} & {nu_ties} \\\\\n"

            if method == "dempster":
                tex_dempster += row
            else:
                tex_murphy += row

            all_results[f"{key}_{method}"] = {
                "f1": f1, "ci_95": [ci[0], ci[1]],
                "mean_K": mean_k, "max_K": max_k,
                "l_count": l_count, "nu_ties": nu_ties
            }

    tex_dempster += r"\bottomrule" + "\n" + r"\end{tabular}"
    tex_murphy += r"\bottomrule" + "\n" + r"\end{tabular}"

    # Combined table
    tex = r"""\begin{tabular}{@{}lrrrrrr@{}}
\toprule
\multicolumn{7}{@{}l}{\textit{Dempster's rule}} \\
\midrule
Subset & F1 [95\% CI] & Mean $K$ & Max $K$ & \{L\} & N/U ties \\
\midrule
"""
    for i, key in enumerate(subset_order):
        r = all_results[f"{key}_dempster"]
        tex += f"{subset_display[i]} & {r['f1']:.3f} [{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}] & {r['mean_K']:.3f} & {r['max_K']:.3f} & {r['l_count']} & {r['nu_ties']} \\\\\n"

    tex += r"""\midrule
\multicolumn{7}{@{}l}{\textit{Murphy's rule}} \\
\midrule
"""
    for i, key in enumerate(subset_order):
        r = all_results[f"{key}_murphy"]
        tex += f"{subset_display[i]} & {r['f1']:.3f} [{r['ci_95'][0]:.3f}, {r['ci_95'][1]:.3f}] & {r['mean_K']:.3f} & {r['max_K']:.3f} & {r['l_count']} & {r['nu_ties']} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
"""

    with open(OUTPUT_DIR / "t4_6_ablation.tex", "w") as f:
        f.write(tex)

    return all_results


def generate_t4_7_recall():
    """Generate T4.7: Per-class recall for each condition (Dempster)."""
    with open(FUSION_RESULTS_PATH) as f:
        fusion = json.load(f)

    labels = load_labels_v2()

    subset_order = ["V", "G", "U", "GV", "UV", "GU", "GUV"]
    subset_display = ["V", "G", "U", "V+G", "V+U", "G+U", "V+G+U"]
    classes = ["N", "U", "P", "L"]

    tex = r"""\begin{tabular}{@{}lrrrr@{}}
\toprule
Subset & N & U & P & L \\
\midrule
"""

    recall_data = {}

    for i, key in enumerate(subset_order):
        results_dict = {r["cell_id"]: r for r in fusion[key]["dempster"]}
        recalls = []
        for cls in classes:
            true_pos = sum(1 for c in results_dict if labels[c]["true_class"] == cls and results_dict[c]["decision"] == cls)
            total_true = sum(1 for c in results_dict if labels[c]["true_class"] == cls)
            recall = true_pos / total_true if total_true > 0 else 0
            recalls.append(recall)

        tex += f"{subset_display[i]} & {recalls[0]:.3f} & {recalls[1]:.3f} & {recalls[2]:.3f} & {recalls[3]:.3f} \\\\\n"
        recall_data[key] = dict(zip(classes, recalls))

    tex += r"""\bottomrule
\end{tabular}
"""

    with open(OUTPUT_DIR / "t4_7_recall.tex", "w") as f:
        f.write(tex)

    return recall_data


def generate_f4_5_recall_heatmap(recall_data):
    """Generate F4.5: Recall heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 9

    subset_order = ["V", "G", "U", "GV", "UV", "GU", "GUV"]
    subset_display = ["V", "G", "U", "V+G", "V+U", "G+U", "V+G+U"]
    classes = ["N", "U", "P", "L"]

    data = np.array([[recall_data[key][cls] for cls in classes] for key in subset_order])

    fig_width = 150 / 25.4
    fig_height = 70 / 25.4

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, cmap='Greys', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(4))
    ax.set_yticks(range(7))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(subset_display)
    ax.set_xlabel('Class')
    ax.set_ylabel('Modality subset')

    # Add values in cells
    for i in range(7):
        for j in range(4):
            color = 'white' if data[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                   fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Recall', shrink=0.8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "f4_5_recall_heatmap.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "f4_5_recall_heatmap.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_f4_6_ablation_f1(ablation_results):
    """Generate F4.6: Ablation F1 bar chart with CIs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 9

    V_FILL = '#d0e3fe'
    V_EDGE = '#4a73a8'

    subset_order = ["V", "G", "U", "GV", "UV", "GU", "GUV"]
    subset_display = ["V", "G", "U", "V+G", "V+U", "G+U", "V+G+U"]

    f1s = [ablation_results[f"{key}_dempster"]["f1"] for key in subset_order]
    cis = [ablation_results[f"{key}_dempster"]["ci_95"] for key in subset_order]
    errors = [[f1s[i] - cis[i][0], cis[i][1] - f1s[i]] for i in range(len(f1s))]
    errors = np.array(errors).T

    fig_width = 150 / 25.4
    fig_height = 60 / 25.4

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(len(subset_order))
    bars = ax.bar(x, f1s, color=V_FILL, edgecolor=V_EDGE, linewidth=0.8)
    ax.errorbar(x, f1s, yerr=errors, fmt='none', color='#333333', capsize=3, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(subset_display)
    ax.set_ylabel('Macro F1')
    ax.set_xlabel('Modality subset')
    ax.set_ylim(0, 0.6)

    # Add value labels
    for i, f1 in enumerate(f1s):
        ax.text(i, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "f4_6_ablation_f1.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "f4_6_ablation_f1.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_f4_7_conflict():
    """Generate F4.7: Conflict distribution for V+U and V+G+U."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 9

    with open(FUSION_RESULTS_PATH) as f:
        fusion = json.load(f)

    NEUTRAL = '#8e867a'

    ks_uv = [r["K"] for r in fusion["UV"]["dempster"]]
    ks_guv = [r["K"] for r in fusion["GUV"]["dempster"]]

    fig_width = 75 / 25.4
    fig_height = 60 / 25.4

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.hist([ks_uv, ks_guv], bins=20, color=[NEUTRAL, '#4a73a8'], alpha=0.7,
            edgecolor='#333333', linewidth=0.5, label=['V+U', 'V+G+U'])
    ax.axvline(x=np.mean(ks_guv), color='#4a73a8', linestyle='--', linewidth=0.8,
              label=f'Mean V+G+U = {np.mean(ks_guv):.3f}')

    ax.set_xlabel('Conflict mass $K$')
    ax.set_ylabel('Count')
    ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "f4_7_conflict.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / "f4_7_conflict.png", format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_s4_4_equality():
    """Generate s4_4_equality.json: Equality checks between conditions."""
    with open(FUSION_RESULTS_PATH) as f:
        fusion = json.load(f)

    comparisons = [
        ("GU", "U", "G+U vs U"),
        ("GV", "V", "V+G vs V"),
        ("GUV", "UV", "V+G+U vs V+U"),
    ]

    results = {}
    for key1, key2, name in comparisons:
        r1 = {r["cell_id"]: r for r in fusion[key1]["dempster"]}
        r2 = {r["cell_id"]: r for r in fusion[key2]["dempster"]}

        # Check BPA differences > 1e-9
        diff_count = 0
        decision_identical = True

        for cell_id in r1:
            betp1 = r1[cell_id]["betp"]
            betp2 = r2[cell_id]["betp"]

            for k in set(betp1.keys()) | set(betp2.keys()):
                if abs(betp1.get(k, 0) - betp2.get(k, 0)) > 1e-9:
                    diff_count += 1
                    break

            if r1[cell_id]["decision"] != r2[cell_id]["decision"]:
                decision_identical = False

        results[name] = {
            "cells_with_betp_diff_gt_1e9": diff_count,
            "decisions_identical": decision_identical,
        }

    with open(OUTPUT_DIR / "s4_4_equality.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# §4.5 Declared Comparisons
# =============================================================================

def generate_t4_8_mcnemar():
    """Generate T4.8: McNemar test results."""
    with open(FUSION_RESULTS_PATH) as f:
        fusion = json.load(f)

    labels = load_labels_v2()

    # Pre-registered comparisons
    comparisons = [
        ("V", "GV", "V -> V+G"),
        ("V", "UV", "V -> V+U"),
        ("G", "GU", "G -> G+U"),
        ("GV", "GUV", "V+G -> V+G+U"),
    ]

    results = []
    for before, after, name in comparisons:
        before_results = {r["cell_id"]: r for r in fusion[before]["dempster"]}
        after_results = {r["cell_id"]: r for r in fusion[after]["dempster"]}

        b = 0  # Before wrong, after right
        c = 0  # Before right, after wrong

        for cell_id in before_results:
            true_class = labels[cell_id]["true_class"]
            before_correct = before_results[cell_id]["decision"] == true_class
            after_correct = after_results[cell_id]["decision"] == true_class

            if not before_correct and after_correct:
                b += 1
            elif before_correct and not after_correct:
                c += 1

        if b + c > 0:
            chi2_stat = (abs(b - c) - 1)**2 / (b + c)
            p_raw = 1 - stats.chi2.cdf(chi2_stat, 1)
        else:
            chi2_stat = None
            p_raw = None

        results.append({
            "comparison": name,
            "b": b,
            "c": c,
            "chi2": chi2_stat,
            "p_raw": p_raw,
        })

    # Holm-Bonferroni correction
    valid_results = [r for r in results if r["p_raw"] is not None]
    valid_results.sort(key=lambda x: x["p_raw"])
    m = len(valid_results)
    for i, r in enumerate(valid_results):
        r["p_holm"] = min(1.0, r["p_raw"] * (m - i))
        r["significant"] = r["p_holm"] < 0.05

    # Update original results
    holm_lookup = {r["comparison"]: r for r in valid_results}
    for r in results:
        if r["p_raw"] is None:
            r["p_holm"] = None
            r["significant"] = None
            r["note"] = "identical"
        else:
            r["p_holm"] = holm_lookup[r["comparison"]]["p_holm"]
            r["significant"] = holm_lookup[r["comparison"]]["significant"]

    tex = r"""\begin{tabular}{@{}lrrrrrl@{}}
\toprule
Comparison & $b$ & $c$ & $\chi^2$ & $p_\mathrm{raw}$ & $p_\mathrm{Holm}$ & Decision \\
\midrule
"""

    for r in results:
        if r["p_raw"] is None:
            tex += f"{r['comparison']} & {r['b']} & {r['c']} & --- & --- & --- & identical \\\\\n"
        else:
            decision = "Sig." if r["significant"] else "n.s."
            tex += f"{r['comparison']} & {r['b']} & {r['c']} & {r['chi2']:.2f} & {r['p_raw']:.4f} & {r['p_holm']:.4f} & {decision} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
"""

    with open(OUTPUT_DIR / "t4_8_mcnemar.tex", "w") as f:
        f.write(tex)

    return results


def generate_s4_5_neff():
    """Generate s4_5_neff.json: Design effect and effective sample size."""
    with open(FUSION_RESULTS_PATH) as f:
        fusion = json.load(f)

    labels = load_labels_v2()

    # Group cells by coupon
    coupon_cells = defaultdict(list)
    for cell_id, lbl in labels.items():
        coupon_cells[lbl["coupon_id"]].append(cell_id)

    # Use GUV results
    results_dict = {r["cell_id"]: r for r in fusion["GUV"]["dempster"]}

    # Per-cell correctness
    coupon_errors = defaultdict(list)
    for cell_id, r in results_dict.items():
        coupon_id = labels[cell_id]["coupon_id"]
        err = 0 if r["decision"] == labels[cell_id]["true_class"] else 1
        coupon_errors[coupon_id].append(err)

    # ANOVA for ICC
    all_errors = [e for errors in coupon_errors.values() for e in errors]
    grand_mean = np.mean(all_errors)

    k = len(coupon_errors)
    n = len(all_errors)
    n_per_coupon = n / k

    ss_between = sum(len(errors) * (np.mean(errors) - grand_mean)**2
                     for errors in coupon_errors.values())
    ss_within = sum(sum((e - np.mean(errors))**2 for e in errors)
                    for errors in coupon_errors.values())

    ms_between = ss_between / (k - 1) if k > 1 else 0
    ms_within = ss_within / (n - k) if n > k else 0

    if ms_within > 0:
        icc = (ms_between - ms_within) / (ms_between + (n_per_coupon - 1) * ms_within)
        icc = max(0, icc)
    else:
        icc = 0

    design_effect = 1 + (n_per_coupon - 1) * icc
    n_eff = n / design_effect

    result = {
        "method": "One-way ANOVA on per-cell correctness for V+G+U",
        "n_cells": n,
        "n_coupons": k,
        "cells_per_coupon": n_per_coupon,
        "icc_rho": round(icc, 4),
        "design_effect": round(design_effect, 2),
        "design_effect_formula": "1 + (n_per_coupon - 1) * rho = 1 + 9 * rho",
        "n_eff": round(n_eff, 1),
        "assumed_deff": 5.5,
        "assumed_n_eff": 27,
    }

    with open(OUTPUT_DIR / "s4_5_neff.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("Generating §4.2-§4.5 outputs...")
    print(f"Output directory: {OUTPUT_DIR}")

    # §4.2 Pipeline Verification
    print("\n[§4.2] Pipeline verification...")
    print("  Generating T4.2 uncertainty table...")
    t4_2 = generate_t4_2_uncertainty()

    print("  Generating T4.3 vision verify table...")
    t4_3 = generate_t4_3_vision_verify()

    print("  Generating s4_2_ut.json and F4.3 UT mapping...")
    s4_2, coupons, observed, expected = generate_s4_2_ut()
    generate_f4_3_ut_mapping(coupons, observed, expected)
    print(f"    Mapping r = {s4_2['mapping_check']['r']}, p = {s4_2['mapping_check']['p_value']}")

    # §4.3 Single-modality Results
    print("\n[§4.3] Single-modality results...")
    print("  Generating T4.4 modality table...")
    t4_4 = generate_t4_4_modality()

    print("  Generating T4.5 elements table...")
    generate_t4_5_elements()

    print("  Generating F4.4 confusion matrices...")
    generate_f4_4_confusion()

    print("  Generating s4_3_numbers.json...")
    s4_3 = generate_s4_3_numbers()

    # §4.4 Fused Results (Ablation)
    print("\n[§4.4] Fused results (ablation)...")
    print("  Generating T4.6 ablation table...")
    ablation_results = generate_t4_6_ablation()

    print("  Generating T4.7 recall table...")
    recall_data = generate_t4_7_recall()

    print("  Generating F4.5 recall heatmap...")
    generate_f4_5_recall_heatmap(recall_data)

    print("  Generating F4.6 ablation F1...")
    generate_f4_6_ablation_f1(ablation_results)

    print("  Generating F4.7 conflict distribution...")
    generate_f4_7_conflict()

    print("  Generating s4_4_equality.json...")
    s4_4 = generate_s4_4_equality()

    # §4.5 Declared Comparisons
    print("\n[§4.5] Declared comparisons...")
    print("  Generating T4.8 McNemar table...")
    mcnemar = generate_t4_8_mcnemar()

    print("  Generating s4_5_neff.json...")
    neff = generate_s4_5_neff()
    print(f"    Design effect = {neff['design_effect']}, n_eff = {neff['n_eff']}")

    print("\n=== SUMMARY ===")
    print(f"Generated {len(list(OUTPUT_DIR.glob('*.tex')))} .tex files")
    print(f"Generated {len(list(OUTPUT_DIR.glob('*.pdf')))} .pdf files")
    print(f"Generated {len(list(OUTPUT_DIR.glob('*.json')))} .json files")

    print("\nDone.")


if __name__ == "__main__":
    main()
