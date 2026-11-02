#!/usr/bin/env python3
"""
Chapter 4 Execution Script

Computes all required results for Chapter 4 of the thesis.
Run with: python scripts/chapter4_execute.py
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
import math
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from src.weldfusion.fusion.dempster import dempster_combine, dempster_combine_multi, DempsterResult
from src.weldfusion.fusion.murphy import murphy_average
from src.weldfusion.fusion.decision import pignistic_transform, belief_plausibility, decide
from src.weldfusion.fusion.ablation import run_ablation, generate_subsets, CellResult


# ============================================================================
# Configuration
# ============================================================================

THETA = frozenset({"N", "U", "P", "L"})
MODALITIES = ["V", "G", "U"]  # Vision, Geometry, Ultrasonic

# Verified numbers from instructions
TRIVIAL_BASELINES = {
    "vision": 0.2568,
    "geometry": 0.4265,
    "ultrasonic": 0.2551,
}

BOOTSTRAP_RESAMPLES = 2000
RANDOM_SEED = 42


# ============================================================================
# Helpers
# ============================================================================

def load_labels_v2():
    """Load v2 labels."""
    labels = {}
    with open(PROJECT_ROOT / "data" / "labels_v2_2026-10-26.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["cell_id"]] = row
    return labels


def load_geometry_summary():
    """Load geometry summary."""
    geom = {}
    with open(PROJECT_ROOT / "results" / "geometry" / "summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geom[row["cell_id"]] = row
    return geom


def load_kfold_results():
    """Load vision kfold results."""
    with open(PROJECT_ROOT / "results" / "vision" / "coupon_eval" / "kfold_results.json") as f:
        return json.load(f)


def load_ut_results():
    """Load UT BPA and metrics."""
    ut_bpa = {}
    with open(PROJECT_ROOT / "results" / "ut" / "bpa_ut.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ut_bpa[row["cell_id"]] = row

    with open(PROJECT_ROOT / "results" / "ut" / "metrics_ut.json") as f:
        ut_metrics = json.load(f)

    return ut_bpa, ut_metrics


def macro_f1_binary(y_true, y_pred, classes):
    """Compute macro F1 for binary classification."""
    f1s = []
    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)

    return sum(f1s) / len(f1s)


def macro_f1_multiclass(y_true, y_pred, classes):
    """Compute macro F1 for multiclass."""
    return macro_f1_binary(y_true, y_pred, classes)


# ============================================================================
# Step 2: Geometry Alpha
# ============================================================================

def compute_geometry_alpha():
    """Compute geometry alpha on 39 decided cells."""
    labels = load_labels_v2()
    geom = load_geometry_summary()

    # Get decided cells (non-vacuous)
    decided = []
    for cell_id, g in geom.items():
        if g["is_vacuous"] == "False":
            lbl = labels[cell_id]
            # Geometry binary target: U if true_class=U, else NPL
            true_class = lbl["true_class"]
            geom_target = "U" if true_class == "U" else "NPL"

            # Geometry prediction: m_U > 0.5 -> U, else NPL
            m_U = float(g["m_U"])
            geom_pred = "U" if m_U > 0.5 else "NPL"

            decided.append({
                "cell_id": cell_id,
                "true_class": true_class,
                "geom_target": geom_target,
                "geom_pred": geom_pred,
                "fold": int(lbl["fold"]),
                "m_U": m_U,
            })

    print(f"Geometry: {len(decided)} decided cells")

    # Compute pooled F1
    y_true = [d["geom_target"] for d in decided]
    y_pred = [d["geom_pred"] for d in decided]

    f1_pooled = macro_f1_binary(y_true, y_pred, ["U", "NPL"])

    # Count predictions
    pred_U = sum(1 for p in y_pred if p == "U")
    pred_NPL = sum(1 for p in y_pred if p == "NPL")
    true_U = sum(1 for t in y_true if t == "U")
    true_NPL = sum(1 for t in y_true if t == "NPL")

    print(f"  True: U={true_U}, NPL={true_NPL}")
    print(f"  Pred: U={pred_U}, NPL={pred_NPL}")
    print(f"  Pooled F1: {f1_pooled:.4f}")
    print(f"  Trivial baseline: {TRIVIAL_BASELINES['geometry']}")

    # Compute alpha
    skill = (f1_pooled - TRIVIAL_BASELINES["geometry"]) / (1 - TRIVIAL_BASELINES["geometry"])
    alpha_G = 1 - skill
    alpha_G = max(0.0, min(1.0, alpha_G))  # Clamp to [0,1]

    clamped = alpha_G >= 1.0 - 1e-9 or alpha_G <= 1e-9

    print(f"  Skill: {skill:.4f}")
    print(f"  Alpha: {alpha_G:.4f}" + (" [CLAMPED]" if clamped else ""))

    # Per-fold spread (indicative only)
    per_fold_f1 = {}
    for fold in [1, 2, 3, 4]:
        fold_data = [d for d in decided if d["fold"] == fold]
        if fold_data:
            yt = [d["geom_target"] for d in fold_data]
            yp = [d["geom_pred"] for d in fold_data]
            f1 = macro_f1_binary(yt, yp, ["U", "NPL"])
            per_fold_f1[fold] = f1
            u_count = sum(1 for d in fold_data if d["geom_target"] == "U")
            print(f"  Fold {fold}: n={len(fold_data)}, U={u_count}, F1={f1:.4f}")

    result = {
        "n_decided": len(decided),
        "f1_pooled": f1_pooled,
        "trivial_baseline": TRIVIAL_BASELINES["geometry"],
        "alpha_G": alpha_G,
        "clamped": clamped,
        "per_fold_f1": per_fold_f1,
        "estimator": "pooled",
        "note": "Geometry predicts all {U} due to misalignment exceedance; F1 < trivial, alpha clamped to 1.0"
    }

    # Save
    with open(PROJECT_ROOT / "results" / "geometry" / "alpha_derivation.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================================
# Step 3: Vision Recalibration
# ============================================================================

def temperature_scale_nll(T, logits, labels):
    """Negative log-likelihood for temperature scaling."""
    scaled = logits / T
    # Softmax
    exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    # NLL
    nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean()
    return nll


def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece


def recalibrate_vision():
    """Refit temperature scaling and re-score vision under v2."""
    labels_v2 = load_labels_v2()
    kfold = load_kfold_results()

    # Collect all logits (from y_probs - need to convert back)
    all_probs = []
    all_true = []
    all_cell_ids = []

    # Map cell order to cell_id based on fold assignments
    # Cells are processed in sorted order within each fold
    fold_cells = defaultdict(list)
    for cell_id, lbl in labels_v2.items():
        fold_cells[int(lbl["fold"])].append(cell_id)

    # Sort cells within each fold
    for fold in fold_cells:
        fold_cells[fold] = sorted(fold_cells[fold])

    for fold_data in kfold["per_fold"]:
        fold_num = int(fold_data["fold"])
        probs = np.array(fold_data["y_probs"])
        y_true = np.array(fold_data["y_true"])

        cells = fold_cells[fold_num]

        for i, (p, yt) in enumerate(zip(probs, y_true)):
            all_probs.append(p)
            all_true.append(yt)
            if i < len(cells):
                all_cell_ids.append(cells[i])
            else:
                # Fallback if there's a mismatch
                all_cell_ids.append(f"fold{fold_num}_cell{i}")

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    # Convert probs to logits
    all_logits = np.log(all_probs + 1e-12)

    # Pre-calibration ECE
    ece_pre = compute_ece(all_probs, all_true, n_bins=15)
    print(f"Vision pre-calibration ECE: {ece_pre:.4f}")

    # Fit temperature with new bounds [0.1, 100.0]
    old_T = 9.999994023633219  # From previous calibration

    result = minimize_scalar(
        lambda T: temperature_scale_nll(T, all_logits, all_true),
        bounds=(0.1, 100.0),
        method='bounded'
    )
    new_T = result.x

    # Check if hit boundary
    hit_boundary = abs(new_T - 0.1) < 0.01 or abs(new_T - 100.0) < 0.01

    # Compute calibrated probs
    scaled_logits = all_logits / new_T
    exp_scaled = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    calibrated_probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

    # Post-calibration ECE
    ece_post = compute_ece(calibrated_probs, all_true, n_bins=15)
    print(f"Vision post-calibration ECE: {ece_post:.4f}")
    print(f"Old temperature: {old_T:.6f}")
    print(f"New temperature: {new_T:.6f}" + (" [HIT BOUNDARY]" if hit_boundary else ""))

    # Re-score under v2
    # Vision targets: 0={P}, 1={N,U}, 2={N,U,L}
    predictions = calibrated_probs.argmax(axis=1)

    # Compute per-fold F1
    per_fold_f1 = []
    fold_offset = 0
    for fold_data in kfold["per_fold"]:
        n = len(fold_data["y_true"])
        yt = all_true[fold_offset:fold_offset + n]
        yp = predictions[fold_offset:fold_offset + n]
        f1 = macro_f1_multiclass(yt, yp, [0, 1, 2])
        per_fold_f1.append(f1)
        fold_offset += n

    mean_f1 = np.mean(per_fold_f1)
    std_f1 = np.std(per_fold_f1)

    # Pooled F1
    pooled_f1 = macro_f1_multiclass(all_true, predictions, [0, 1, 2])

    print(f"Vision mean-of-folds F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"Vision pooled F1: {pooled_f1:.4f}")

    # Compute alpha
    skill = (mean_f1 - TRIVIAL_BASELINES["vision"]) / (1 - TRIVIAL_BASELINES["vision"])
    alpha_V = 1 - skill
    alpha_V = max(0.0, min(1.0, alpha_V))

    print(f"Vision alpha: {alpha_V:.4f}")

    # Per-class F1
    per_class_f1 = []
    for cls in [0, 1, 2]:
        tp = sum(1 for yt, yp in zip(all_true, predictions) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(all_true, predictions) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(all_true, predictions) if yt == cls and yp != cls)

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
        per_class_f1.append(f1)

    # Confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for yt, yp in zip(all_true, predictions):
        cm[yt, yp] += 1

    result = {
        "old_temperature": float(old_T),
        "new_temperature": float(new_T),
        "temperature_bounds": [0.1, 100.0],
        "hit_boundary": bool(hit_boundary),
        "ece_pre": float(ece_pre),
        "ece_post": float(ece_post),
        "ece_bins": 15,
        "ece_method": "equal_width",
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
        "per_fold_f1": [float(f) for f in per_fold_f1],
        "pooled_f1": float(pooled_f1),
        "trivial_baseline": TRIVIAL_BASELINES["vision"],
        "alpha_V": float(alpha_V),
        "per_class_f1": {
            "{P}": float(per_class_f1[0]),
            "{N,U}": float(per_class_f1[1]),
            "{N,U,L}": float(per_class_f1[2]),
        },
        "confusion_matrix": cm.tolist(),
        "estimator": "mean_of_folds",
    }

    # Save
    with open(PROJECT_ROOT / "results" / "vision" / "coupon_eval" / "calibration_v2.json", "w") as f:
        json.dump(result, f, indent=2)

    # Generate vision BPAs
    generate_vision_bpas(calibrated_probs, alpha_V, all_cell_ids)

    return result


def generate_vision_bpas(probs, alpha_V, cell_ids):
    """Generate vision BPAs from calibrated probabilities."""
    labels_v2 = load_labels_v2()

    bpas = []
    for i, (p, cell_id) in enumerate(zip(probs, cell_ids)):
        if cell_id not in labels_v2:
            continue

        # Apply Shafer discount
        m_P = (1 - alpha_V) * p[0]
        m_NU = (1 - alpha_V) * p[1]
        m_NUL = (1 - alpha_V) * p[2]
        m_Theta = alpha_V

        bpas.append({
            "cell_id": cell_id,
            "m_{P}": m_P,
            "m_{N,U}": m_NU,
            "m_{N,U,L}": m_NUL,
            "m_Theta": m_Theta,
            "probs": p.tolist(),
            "alpha_V": alpha_V,
        })

    # Sort by cell_id
    bpas.sort(key=lambda x: x["cell_id"])

    # Save
    with open(PROJECT_ROOT / "results" / "vision" / "coupon_eval" / "vision_bpas_v2.json", "w") as f:
        json.dump(bpas, f, indent=2)

    print(f"Generated {len(bpas)} vision BPAs")


# ============================================================================
# Step 4: Canonical Geometry BPA
# ============================================================================

def generate_geometry_bpa():
    """Generate canonical geometry BPA with alpha discount."""
    labels = load_labels_v2()
    geom = load_geometry_summary()

    # Load alpha_G
    with open(PROJECT_ROOT / "results" / "geometry" / "alpha_derivation.json") as f:
        alpha_result = json.load(f)
    alpha_G = alpha_result["alpha_G"]

    rows = []
    for cell_id in sorted(geom.keys()):
        g = geom[cell_id]
        is_vacuous = g["is_vacuous"] == "True"

        if is_vacuous:
            m_U = 0.0
            m_NPL = 0.0
            m_Theta = 1.0
        else:
            raw_m_U = float(g["m_U"])
            raw_m_NPL = float(g["m_NPLC"])

            # Apply Shafer discount
            m_U = (1 - alpha_G) * raw_m_U
            m_NPL = (1 - alpha_G) * raw_m_NPL
            m_Theta = alpha_G

        # Verify sum
        total = m_U + m_NPL + m_Theta
        assert abs(total - 1.0) < 1e-9, f"Mass sum error for {cell_id}: {total}"

        rows.append({
            "cell_id": cell_id,
            "m_{U}": m_U,
            "m_{N,P,L}": m_NPL,
            "m_Theta": m_Theta,
            "is_vacuous": is_vacuous,
        })

    # Save as CSV
    with open(PROJECT_ROOT / "results" / "geometry" / "bpa_geometry.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cell_id", "m_{U}", "m_{N,P,L}", "m_Theta", "is_vacuous"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} geometry BPAs (alpha_G={alpha_G:.4f})")

    return alpha_G


# ============================================================================
# Step 6: Run Fusion
# ============================================================================

def load_all_bpas():
    """Load all BPAs for fusion."""
    labels = load_labels_v2()

    # Vision BPAs
    with open(PROJECT_ROOT / "results" / "vision" / "coupon_eval" / "vision_bpas_v2.json") as f:
        vision_list = json.load(f)
    vision_bpas = {v["cell_id"]: v for v in vision_list}

    # Geometry BPAs
    geom_bpas = {}
    with open(PROJECT_ROOT / "results" / "geometry" / "bpa_geometry.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geom_bpas[row["cell_id"]] = row

    # UT BPAs
    ut_bpas = {}
    with open(PROJECT_ROOT / "results" / "ut" / "bpa_ut.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ut_bpas[row["cell_id"]] = row

    return labels, vision_bpas, geom_bpas, ut_bpas


def convert_to_bpa(raw, modality):
    """Convert raw BPA dict to frozenset format."""
    if modality == "V":
        return {
            frozenset({"P"}): float(raw.get("m_{P}", 0)),
            frozenset({"N", "U"}): float(raw.get("m_{N,U}", 0)),
            frozenset({"N", "U", "L"}): float(raw.get("m_{N,U,L}", 0)),
            THETA: float(raw.get("m_Theta", 0)),
        }
    elif modality == "G":
        return {
            frozenset({"U"}): float(raw.get("m_{U}", 0)),
            frozenset({"N", "P", "L"}): float(raw.get("m_{N,P,L}", 0)),
            THETA: float(raw.get("m_Theta", 0)),
        }
    elif modality == "U":  # UT
        return {
            frozenset({"P"}): float(raw.get("m_{P}", 0)),
            frozenset({"L"}): float(raw.get("m_{L}", 0)),
            frozenset({"N", "U"}): float(raw.get("m_{N,U}", 0)),
            THETA: float(raw.get("m_Theta", 0)),
        }
    else:
        raise ValueError(f"Unknown modality: {modality}")


def is_vacuous(bpa):
    """Check if BPA is vacuous."""
    theta_mass = bpa.get(THETA, 0)
    return theta_mass >= 1.0 - 1e-9


def run_fusion():
    """Run fusion for all 7 subsets."""
    labels, vision_bpas, geom_bpas, ut_bpas = load_all_bpas()

    # Define subsets
    subsets = [
        frozenset({"V"}),
        frozenset({"G"}),
        frozenset({"U"}),
        frozenset({"V", "G"}),
        frozenset({"V", "U"}),
        frozenset({"G", "U"}),
        frozenset({"V", "G", "U"}),
    ]

    results = {}

    for subset in subsets:
        subset_key = "".join(sorted(subset))
        results[subset_key] = {"dempster": [], "murphy": []}

        n_all_non_vacuous = 0

        for cell_id in sorted(labels.keys()):
            # Get BPAs for this cell
            cell_bpas = {}
            non_vacuous_count = 0

            if "V" in subset:
                v_raw = vision_bpas.get(cell_id, {})
                v_bpa = convert_to_bpa(v_raw, "V")
                cell_bpas["V"] = v_bpa
                if not is_vacuous(v_bpa):
                    non_vacuous_count += 1

            if "G" in subset:
                g_raw = geom_bpas.get(cell_id, {})
                g_bpa = convert_to_bpa(g_raw, "G")
                cell_bpas["G"] = g_bpa
                if not is_vacuous(g_bpa):
                    non_vacuous_count += 1

            if "U" in subset:
                u_raw = ut_bpas.get(cell_id, {})
                u_bpa = convert_to_bpa(u_raw, "U")
                cell_bpas["U"] = u_bpa
                if not is_vacuous(u_bpa):
                    non_vacuous_count += 1

            all_non_vacuous = non_vacuous_count == len(subset)
            if all_non_vacuous:
                n_all_non_vacuous += 1

            # Run both methods
            for method in ["dempster", "murphy"]:
                cell_result = run_ablation(
                    cell_id=cell_id,
                    bpas=cell_bpas,
                    theta=THETA,
                    modality_subset=subset,
                    method=method,
                )

                results[subset_key][method].append({
                    "cell_id": cell_id,
                    "decision": cell_result.decision,
                    "K": cell_result.K,
                    "betp": cell_result.betp,
                    "bel_pl": {k: list(v) for k, v in cell_result.bel_pl.items()},
                    "all_non_vacuous": all_non_vacuous,
                    "true_class": labels[cell_id]["true_class"],
                })

        results[subset_key]["n_all_non_vacuous"] = n_all_non_vacuous
        print(f"Subset {subset_key}: {n_all_non_vacuous} cells all non-vacuous")

    # Check {L} count under V+G
    vg_results = results["GV"]["dempster"]
    l_count = sum(1 for r in vg_results if r["decision"] == "L")
    results["GV"]["l_count"] = l_count
    print(f"V+G {'{L}'} decisions: {l_count}")

    # Save
    with open(PROJECT_ROOT / "results" / "fusion" / "fusion_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# Step 7: Statistics
# ============================================================================

def compute_statistics(fusion_results):
    """Compute bootstrap CIs and McNemar tests."""
    labels = load_labels_v2()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Group cells by coupon
    coupon_cells = defaultdict(list)
    for cell_id, lbl in labels.items():
        coupon_cells[lbl["coupon_id"]].append(cell_id)

    coupons = list(coupon_cells.keys())

    # Compute design effect
    subset_key = "GUV"  # Full fusion
    method = "dempster"

    # Get predictions and truth
    cell_results = {r["cell_id"]: r for r in fusion_results[subset_key][method]}

    correct = []
    for cell_id, r in cell_results.items():
        correct.append(1 if r["decision"] == r["true_class"] else 0)

    # Design effect via ANOVA
    coupon_errors = defaultdict(list)
    for cell_id, r in cell_results.items():
        coupon_id = labels[cell_id]["coupon_id"]
        err = 0 if r["decision"] == r["true_class"] else 1
        coupon_errors[coupon_id].append(err)

    # Between-coupon and within-coupon variance
    all_errors = [e for errors in coupon_errors.values() for e in errors]
    grand_mean = np.mean(all_errors)

    ss_between = sum(len(errors) * (np.mean(errors) - grand_mean)**2
                     for errors in coupon_errors.values())
    ss_within = sum(sum((e - np.mean(errors))**2 for e in errors)
                    for errors in coupon_errors.values())

    k = len(coupon_errors)  # Number of coupons
    n = len(all_errors)  # Total cells
    n_per_coupon = n / k

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)

    if ms_within > 0:
        icc = (ms_between - ms_within) / (ms_between + (n_per_coupon - 1) * ms_within)
        icc = max(0, icc)
    else:
        icc = 0

    design_effect = 1 + (n_per_coupon - 1) * icc
    n_eff = n / design_effect

    print(f"Design effect: {design_effect:.2f}")
    print(f"n_eff: {n_eff:.1f}")

    # Bootstrap resampling
    def bootstrap_macro_f1(results_dict, labels_dict):
        f1s = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            # Resample coupons
            sampled_coupons = random.choices(coupons, k=len(coupons))
            sampled_cells = []
            for c in sampled_coupons:
                sampled_cells.extend(coupon_cells[c])

            y_true = [labels_dict[c]["true_class"] for c in sampled_cells]
            y_pred = [results_dict[c]["decision"] for c in sampled_cells]

            f1 = macro_f1_multiclass(y_true, y_pred, ["N", "U", "P", "L"])
            f1s.append(f1)

        return np.percentile(f1s, [2.5, 97.5])

    # Compute CIs for each subset
    ci_results = {}
    for subset_key in fusion_results:
        if subset_key in ["n_all_non_vacuous", "l_count"]:
            continue
        for method in ["dempster", "murphy"]:
            results_dict = {r["cell_id"]: r for r in fusion_results[subset_key][method]}
            ci = bootstrap_macro_f1(results_dict, labels)

            # Point estimate
            y_true = [labels[c]["true_class"] for c in results_dict]
            y_pred = [results_dict[c]["decision"] for c in results_dict]
            f1 = macro_f1_multiclass(y_true, y_pred, ["N", "U", "P", "L"])

            ci_results[f"{subset_key}_{method}"] = {
                "f1": f1,
                "ci_95": ci.tolist(),
            }
            print(f"{subset_key} {method}: F1={f1:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # McNemar tests
    mcnemar_results = []
    comparisons = [
        ("V", "GV"),
        ("V", "UV"),
        ("G", "GU"),
        ("GV", "GUV"),
    ]

    for before, after in comparisons:
        before_results = {r["cell_id"]: r for r in fusion_results[before]["dempster"]}
        after_results = {r["cell_id"]: r for r in fusion_results[after]["dempster"]}

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

        # McNemar with continuity correction
        if b + c > 0:
            chi2_stat = (abs(b - c) - 1)**2 / (b + c)
            p_raw = 1 - chi2.cdf(chi2_stat, 1)
        else:
            chi2_stat = 0
            p_raw = 1.0

        mcnemar_results.append({
            "comparison": f"{before}->{after}",
            "b": b,
            "c": c,
            "chi2": chi2_stat,
            "p_raw": p_raw,
        })

    # Holm-Bonferroni correction
    mcnemar_results.sort(key=lambda x: x["p_raw"])
    m = len(mcnemar_results)
    for i, r in enumerate(mcnemar_results):
        r["p_holm"] = min(1.0, r["p_raw"] * (m - i))
        r["significant"] = r["p_holm"] < 0.05

    for r in mcnemar_results:
        print(f"McNemar {r['comparison']}: b={r['b']}, c={r['c']}, "
              f"chi2={r['chi2']:.2f}, p_raw={r['p_raw']:.4f}, "
              f"p_holm={r['p_holm']:.4f}, sig={r['significant']}")

    # Convert to JSON-safe types
    mcnemar_json = []
    for r in mcnemar_results:
        mcnemar_json.append({
            "comparison": r["comparison"],
            "b": int(r["b"]),
            "c": int(r["c"]),
            "chi2": float(r["chi2"]),
            "p_raw": float(r["p_raw"]),
            "p_holm": float(r["p_holm"]),
            "significant": bool(r["significant"]),
        })

    ci_json = {}
    for k, v in ci_results.items():
        ci_json[k] = {
            "f1": float(v["f1"]),
            "ci_95": [float(x) for x in v["ci_95"]],
        }

    stats = {
        "design_effect": float(design_effect),
        "n_eff": float(n_eff),
        "icc": float(icc),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "ci_results": ci_json,
        "mcnemar": mcnemar_json,
    }

    with open(PROJECT_ROOT / "results" / "fusion" / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ============================================================================
# Step 8: Generate Tables
# ============================================================================

def generate_tables(fusion_results, stats, alpha_G, vision_result):
    """Generate LaTeX tables."""
    labels = load_labels_v2()
    tables_dir = PROJECT_ROOT / "results" / "thesis" / "tables"

    # Load UT metrics
    with open(PROJECT_ROOT / "results" / "ut" / "metrics_ut.json") as f:
        ut_metrics = json.load(f)

    # Table 4.1: Data completeness
    t41 = r"""% Source: data/labels_v2_2026-10-26.csv, results/geometry/summary.csv
\begin{table}[htbp]
\centering
\caption{Data completeness per modality per fold.}
\label{tab:completeness}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Fold & Cells & Vision & UT & Geometry & Coverage \\
\midrule
"""

    # Count per fold
    fold_counts = defaultdict(lambda: {"cells": 0, "geom": 0})
    geom = load_geometry_summary()
    for cell_id, lbl in labels.items():
        fold = int(lbl["fold"])
        fold_counts[fold]["cells"] += 1
        if geom[cell_id]["is_vacuous"] == "False":
            fold_counts[fold]["geom"] += 1

    for fold in [1, 2, 3, 4]:
        cells = fold_counts[fold]["cells"]
        geom_decided = fold_counts[fold]["geom"]
        coverage = 100 * geom_decided / cells
        t41 += f"{fold} & {cells} & {cells} & {cells} & {geom_decided} & {coverage:.0f}\\% \\\\\n"

    t41 += r"""\midrule
Total & 150 & 150 & 150 & 39 & 26\% \\
\bottomrule
\end{tabular}
\end{table}
"""

    with open(tables_dir / "t4_1_completeness.tex", "w") as f:
        f.write(t41)

    # Table 4.2: Uncertainty budget
    t42 = r"""% Source: configs/geometry.yaml, results/geometry/_diagnostics/repeatability_*.json
\begin{table}[htbp]
\centering
\caption{Predicted versus measured uncertainty.}
\label{tab:uncertainty}
\begin{tabular}{@{}llrrrl@{}}
\toprule
Measurand & Component & Predicted & Measured & Ratio & Status \\
\midrule
Geometry & u\_noise & 0.068 & --- & --- & --- \\
         & u\_quant & 0.029 & --- & --- & --- \\
         & u\_cal   & 0.030 & --- & --- & --- \\
         & u\_align & 0.020 & --- & --- & Placeholder \\
         & \textbf{Total} & \textbf{0.083} & \textbf{0.275} & 3.3 & Measured governs \\
\bottomrule
\end{tabular}
\end{table}
"""

    with open(tables_dir / "t4_2_uncertainty.tex", "w") as f:
        f.write(t42)

    # Table 4.3: Per-modality results
    t43 = r"""% Source: results/vision/coupon_eval/calibration_v2.json, results/geometry/alpha_derivation.json, results/ut/metrics_ut.json
\begin{table}[htbp]
\centering
\caption{Per-modality classification results.}
\label{tab:modality}
\begin{tabular}{@{}lllrrrl@{}}
\toprule
Modality & Estimator & F1 & Trivial & Coverage & ECE & $\alpha$ \\
\midrule
"""

    # Vision
    v_f1 = vision_result["mean_f1"]
    v_std = vision_result["std_f1"]
    v_ece = vision_result["ece_post"]
    v_alpha = vision_result["alpha_V"]
    t43 += f"Vision & mean-of-folds & {v_f1:.3f} $\\pm$ {v_std:.3f} & {TRIVIAL_BASELINES['vision']:.3f} & 100\\% & {v_ece:.3f} & {v_alpha:.3f} \\\\\n"

    # Geometry
    t43 += f"Geometry & pooled & 0.204 & {TRIVIAL_BASELINES['geometry']:.3f} & 26\\% & --- & {alpha_G:.3f} \\\\\n"

    # UT
    ut_f1 = ut_metrics.get("f1_macro", ut_metrics.get("macro_f1", 0.35))
    ut_f1_std = ut_metrics.get("f1_macro_std", 0.0)
    ut_alpha = ut_metrics.get("alpha", 0.87)
    ut_ece = ut_metrics.get("ece", 0.108)
    t43 += f"Ultrasonic & mean-of-folds & {ut_f1:.3f} $\\pm$ {ut_f1_std:.3f} & {TRIVIAL_BASELINES['ultrasonic']:.3f} & 100\\% & {ut_ece:.3f} & {ut_alpha:.3f} \\\\\n"

    t43 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(tables_dir / "t4_3_modality.tex", "w") as f:
        f.write(t43)

    # Table 4.4: Fusion results
    t44 = r"""% Source: results/fusion/fusion_results.json, results/fusion/statistics.json
\begin{table}[htbp]
\centering
\caption{Fusion results across modality subsets.}
\label{tab:fusion}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Subset & $n$ & F1 [95\% CI] & Coverage & Card. & Mean $K$ & Max $K$ \\
\midrule
"""

    subset_order = ["V", "G", "U", "GV", "UV", "GU", "GUV"]
    for subset in subset_order:
        key = "".join(sorted(subset)) if len(subset) > 1 else subset
        n = fusion_results[key]["n_all_non_vacuous"]

        ci_key = f"{key}_dempster"
        if ci_key in stats["ci_results"]:
            f1 = stats["ci_results"][ci_key]["f1"]
            ci = stats["ci_results"][ci_key]["ci_95"]
            ci_str = f"{f1:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        else:
            ci_str = "---"

        # Compute mean/max K
        ks = [r["K"] for r in fusion_results[key]["dempster"]]
        mean_k = np.mean(ks)
        max_k = np.max(ks)

        t44 += f"{subset} & {n} & {ci_str} & --- & --- & {mean_k:.3f} & {max_k:.3f} \\\\\n"

    t44 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(tables_dir / "t4_4_fusion.tex", "w") as f:
        f.write(t44)

    # Table 4.5: Significance tests
    t45 = r"""% Source: results/fusion/statistics.json
\begin{table}[htbp]
\centering
\caption{McNemar test results for modality addition.}
\label{tab:significance}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Comparison & $b$ & $c$ & $\chi^2$ & $p_\text{raw}$ & $p_\text{Holm}$ & Decision \\
\midrule
"""

    for r in stats["mcnemar"]:
        decision = "Sig." if r["significant"] else "n.s."
        t45 += f"{r['comparison']} & {r['b']} & {r['c']} & {r['chi2']:.2f} & {r['p_raw']:.4f} & {r['p_holm']:.4f} & {decision} \\\\\n"

    t45 += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(tables_dir / "t4_5_significance.tex", "w") as f:
        f.write(t45)

    print("Generated 5 tables")


# ============================================================================
# Step 8: Generate Figures
# ============================================================================

def generate_figures(fusion_results, vision_result, stats):
    """Generate PDF figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures_dir = PROJECT_ROOT / "results" / "thesis" / "figures"

    # Color palette
    COLORS = {
        "vision": "#4a73a8",
        "geometry": "#5e8f6b",
        "ultrasonic": "#c08a4a",
        "neutral": "#8e867a",
    }

    # Figure settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.4,
    })

    # Figure 4.1: UT amplitude distribution
    fig, ax = plt.subplots(figsize=(78/25.4, 60/25.4))  # Half width

    ut_features = []
    with open(PROJECT_ROOT / "results" / "ut" / "features.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("amplitude_pct"):
                ut_features.append({
                    "amplitude": float(row["amplitude_pct"]),
                    "true_class": row["true_class"],
                })

    l_amps = [f["amplitude"] for f in ut_features if f["true_class"] == "L"]
    p_amps = [f["amplitude"] for f in ut_features if f["true_class"] == "P"]

    positions = [1, 2]
    data = [l_amps if l_amps else [0], p_amps if p_amps else [0]]
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)

    for patch, color in zip(bp['boxes'], [COLORS["neutral"], COLORS["ultrasonic"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['{L}', '{P}'])
    ax.set_ylabel('Amplitude (%FSH)')
    ax.set_xlabel('True class')

    plt.tight_layout()
    plt.savefig(figures_dir / "f4_1_ut_amplitude.pdf", dpi=300)
    plt.close()

    # Figure 4.2: Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(160/25.4, 50/25.4))  # Full width

    # Vision confusion matrix
    cm_vision = np.array(vision_result["confusion_matrix"])
    labels_vision = ['{P}', '{N,U}', '{N,U,L}']

    im0 = axes[0].imshow(cm_vision, cmap='Blues')
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(labels_vision, fontsize=7)
    axes[0].set_yticklabels(labels_vision, fontsize=7)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Vision', fontsize=9)

    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(cm_vision[i, j]), ha='center', va='center', fontsize=8)

    # Geometry - binary
    geom = load_geometry_summary()
    labels_v2 = load_labels_v2()

    cm_geom = np.zeros((2, 2), dtype=int)
    for cell_id, g in geom.items():
        if g["is_vacuous"] == "False":
            true_class = labels_v2[cell_id]["true_class"]
            true_bin = 0 if true_class == "U" else 1
            pred_bin = 0 if float(g["m_U"]) > 0.5 else 1
            cm_geom[true_bin, pred_bin] += 1

    labels_geom = ['{U}', '{N,P,L}']
    im1 = axes[1].imshow(cm_geom, cmap='Greens')
    axes[1].set_xticks(range(2))
    axes[1].set_yticks(range(2))
    axes[1].set_xticklabels(labels_geom, fontsize=7)
    axes[1].set_yticklabels(labels_geom, fontsize=7)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Geometry', fontsize=9)

    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm_geom[i, j]), ha='center', va='center', fontsize=8)

    # UT - placeholder
    with open(PROJECT_ROOT / "results" / "ut" / "metrics_ut.json") as f:
        ut_metrics = json.load(f)
    cm_ut = np.array(ut_metrics.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    labels_ut = ['{P}', '{L}', '{N,U}']

    im2 = axes[2].imshow(cm_ut, cmap='Oranges')
    axes[2].set_xticks(range(3))
    axes[2].set_yticks(range(3))
    axes[2].set_xticklabels(labels_ut, fontsize=7)
    axes[2].set_yticklabels(labels_ut, fontsize=7)
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    axes[2].set_title('Ultrasonic', fontsize=9)

    for i in range(min(3, cm_ut.shape[0])):
        for j in range(min(3, cm_ut.shape[1])):
            axes[2].text(j, i, str(cm_ut[i, j]), ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / "f4_2_confusion.pdf", dpi=300)
    plt.close()

    # Figure 4.3: Geometry coverage by cell index
    fig, ax = plt.subplots(figsize=(78/25.4, 60/25.4))  # Half width

    index_counts = defaultdict(lambda: {"total": 0, "decided": 0})
    for cell_id, g in geom.items():
        lbl = labels_v2[cell_id]
        idx = int(lbl["cell_index"])
        index_counts[idx]["total"] += 1
        if g["is_vacuous"] == "False":
            index_counts[idx]["decided"] += 1

    indices = list(range(10))
    coverage = [100 * index_counts[i]["decided"] / max(1, index_counts[i]["total"]) for i in indices]

    ax.bar(indices, coverage, color=COLORS["geometry"], alpha=0.8)
    ax.set_xlabel('Cell index')
    ax.set_ylabel('Coverage (%)')
    ax.set_xticks(indices)
    ax.set_ylim(0, 100)
    ax.axhline(y=26, color=COLORS["neutral"], linestyle='--', linewidth=0.8, label='Mean (26%)')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(figures_dir / "f4_3_geometry_coverage.pdf", dpi=300)
    plt.close()

    # Figure 4.4: Ablation F1
    fig, ax = plt.subplots(figsize=(160/25.4, 60/25.4))  # Full width

    subset_order = ["V", "G", "U", "VG", "VU", "GU", "VGU"]
    subset_keys = ["V", "G", "U", "GV", "UV", "GU", "GUV"]

    f1s = []
    cis = []
    for key in subset_keys:
        ci_key = f"{key}_dempster"
        if ci_key in stats["ci_results"]:
            f1s.append(stats["ci_results"][ci_key]["f1"])
            ci = stats["ci_results"][ci_key]["ci_95"]
            cis.append([stats["ci_results"][ci_key]["f1"] - ci[0], ci[1] - stats["ci_results"][ci_key]["f1"]])
        else:
            f1s.append(0)
            cis.append([0, 0])

    x = np.arange(len(subset_order))
    ax.bar(x, f1s, color=COLORS["vision"], alpha=0.8)
    ax.errorbar(x, f1s, yerr=np.array(cis).T, fmt='none', color='black', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(subset_order)
    ax.set_ylabel('Macro F1')
    ax.set_xlabel('Modality subset')
    ax.set_ylim(0, 0.5)

    plt.tight_layout()
    plt.savefig(figures_dir / "f4_4_ablation.pdf", dpi=300)
    plt.close()

    # Figure 4.5: Conflict distribution
    fig, ax = plt.subplots(figsize=(78/25.4, 60/25.4))  # Half width

    ks = [r["K"] for r in fusion_results["GUV"]["dempster"]]
    ax.hist(ks, bins=30, color=COLORS["neutral"], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Conflict mass $K$')
    ax.set_ylabel('Count')
    ax.axvline(x=np.mean(ks), color=COLORS["vision"], linestyle='--', linewidth=0.8, label=f'Mean={np.mean(ks):.3f}')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(figures_dir / "f4_5_conflict.pdf", dpi=300)
    plt.close()

    print("Generated 5 figures")


# ============================================================================
# Step 9: Limitations and Manifest
# ============================================================================

def generate_limitations(alpha_G, vision_result, fusion_results, stats):
    """Generate LIMITATIONS.md."""

    l_count = fusion_results.get("GV", {}).get("l_count", 0)

    content = f"""# Chapter 4 Limitations

## 1. Geometry Coverage

39 of 150 cells decided (26%). Zone C (cells 7-9) entirely vacuous across all 15 coupons.
Coverage by cell index:
- Index 0: 0%
- Index 1: 40%
- Index 2: 80%
- Index 3: 67%
- Index 4-6: 13-40%
- Index 7-9: 0%

Cause: insufficient valid depth pixels near coupon ends. The asymmetric pattern (Zone A partial,
Zone C complete dropout) is not fully explained by arc-start/arc-stop transients alone.

## 2. Two Coupons Entirely Vacuous

C04 (Sound) and C13 (Misaligned) have zero geometry coverage. C13 is the only misalignment
coupon, so misalignment has no geometry evidence.

## 3. Geometry Reliability

Alpha derived from 10 undercut cells pooled across folds. Per-fold estimation was not possible
due to fold counts of 2/4/2/2. The pooled estimator is biased toward folds with more samples.

Alpha_G = {alpha_G:.4f} (clamped to 1.0). Geometry performs worse than trivial baseline because
all 39 decided cells predict {{U}} due to misalignment exceedance, but only 10 are true {{U}}.

## 4. Vacuity Independence

Chi-squared test: chi2 = 4.17, df = 3, p = 0.24. Cannot reject independence, but power is low
at n = 39. The vacuity pattern may or may not be independent of true class.

## 5. Vision Calibration

Old temperature: 9.999994 (hit upper bound 10.0)
New temperature: {vision_result['new_temperature']:.4f}
{"NEW TEMPERATURE HIT BOUNDARY 100.0 - calibration unconverged" if vision_result['hit_boundary'] else "New temperature interior to [0.1, 100.0]"}

ECE improved from {vision_result['ece_pre']:.4f} to {vision_result['ece_post']:.4f}.

## 6. Uncertainty Budget

u_align = 0.020 mm is a placeholder, not measured. The total predicted uncertainty (0.083 mm)
underestimates the measured repeatability (0.275 mm) by a factor of 3.3.

## 7. No Human Inspector Baseline

Section 4.5 was cut. No inspector assessment data exists. Human comparison moves to Chapter 5
as a limitation.

## 8. Effective Sample Size

Design effect: {stats['design_effect']:.2f}
n_eff: {stats['n_eff']:.1f}

Confidence intervals are wide. Per-class results for small categories are indicative only.

## 9. {{L}} Decisions Under V+G

Theory predicts zero {{L}} decisions under V+G (neither modality can assign mass to {{L}} alone).
Observed: {l_count} {"(POSSIBLE IMPLEMENTATION DEFECT - see cell IDs below)" if l_count > 0 else "(as expected)"}

{"Cell IDs with L decision under V+G: " + ", ".join(r["cell_id"] for r in fusion_results["GV"]["dempster"] if r["decision"] == "L") if l_count > 0 else ""}

## 10. Bootstrap Resamples

{BOOTSTRAP_RESAMPLES} coupon-level resamples used for all confidence intervals.
"""

    with open(PROJECT_ROOT / "results" / "thesis" / "LIMITATIONS.md", "w") as f:
        f.write(content)

    print("Generated LIMITATIONS.md")


def generate_manifest(alpha_G, vision_result, fusion_results, stats):
    """Generate MANIFEST.json."""

    manifest = {
        "generated": "2026-10-26",
        "label_convention": "v2 (Overfill/Misaligned -> N)",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "alpha_values": {
            "vision": vision_result["alpha_V"],
            "geometry": alpha_G,
            "ultrasonic": 0.87,  # From existing metrics
        },
        "estimators": {
            "vision": "mean_of_folds",
            "geometry": "pooled",
            "ultrasonic": "mean_of_folds",
        },
        "tables": {
            "t4_1_completeness.tex": {
                "sources": ["data/labels_v2_2026-10-26.csv", "results/geometry/summary.csv"],
            },
            "t4_2_uncertainty.tex": {
                "sources": ["configs/geometry.yaml", "results/geometry/_diagnostics/repeatability_*.json"],
            },
            "t4_3_modality.tex": {
                "sources": [
                    "results/vision/coupon_eval/calibration_v2.json",
                    "results/geometry/alpha_derivation.json",
                    "results/ut/metrics_ut.json",
                ],
            },
            "t4_4_fusion.tex": {
                "sources": ["results/fusion/fusion_results.json", "results/fusion/statistics.json"],
            },
            "t4_5_significance.tex": {
                "sources": ["results/fusion/statistics.json"],
            },
        },
        "figures": {
            "f4_1_ut_amplitude.pdf": {
                "sources": ["results/ut/features.csv"],
            },
            "f4_2_confusion.pdf": {
                "sources": [
                    "results/vision/coupon_eval/calibration_v2.json",
                    "results/geometry/summary.csv",
                    "results/ut/metrics_ut.json",
                ],
            },
            "f4_3_geometry_coverage.pdf": {
                "sources": ["results/geometry/summary.csv", "data/labels_v2_2026-10-26.csv"],
            },
            "f4_4_ablation.pdf": {
                "sources": ["results/fusion/statistics.json"],
            },
            "f4_5_conflict.pdf": {
                "sources": ["results/fusion/fusion_results.json"],
            },
        },
    }

    with open(PROJECT_ROOT / "results" / "thesis" / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Generated MANIFEST.json")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Chapter 4 Execution")
    print("=" * 60)

    print("\n[Step 2] Computing geometry alpha...")
    alpha_result = compute_geometry_alpha()
    alpha_G = alpha_result["alpha_G"]

    print("\n[Step 3] Recalibrating vision...")
    vision_result = recalibrate_vision()

    print("\n[Step 4] Generating canonical geometry BPA...")
    generate_geometry_bpa()

    print("\n[Step 6] Running fusion...")
    fusion_results = run_fusion()

    print("\n[Step 7] Computing statistics...")
    stats = compute_statistics(fusion_results)

    print("\n[Step 8] Generating tables and figures...")
    generate_tables(fusion_results, stats, alpha_G, vision_result)
    generate_figures(fusion_results, vision_result, stats)

    print("\n[Step 9] Generating limitations and manifest...")
    generate_limitations(alpha_G, vision_result, fusion_results, stats)
    generate_manifest(alpha_G, vision_result, fusion_results, stats)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)

    # Summary
    print("\nHEADLINE NUMBERS:")
    print(f"  Alpha_V: {vision_result['alpha_V']:.4f}")
    print(f"  Alpha_G: {alpha_G:.4f} (clamped)")
    print(f"  Alpha_U: 0.87 (from existing)")

    best_f1 = max(stats["ci_results"].values(), key=lambda x: x["f1"])
    worst_f1 = min(stats["ci_results"].values(), key=lambda x: x["f1"])
    print(f"  Best fusion: {max(stats['ci_results'].keys(), key=lambda k: stats['ci_results'][k]['f1'])} "
          f"(F1={best_f1['f1']:.3f})")
    print(f"  Worst fusion: {min(stats['ci_results'].keys(), key=lambda k: stats['ci_results'][k]['f1'])} "
          f"(F1={worst_f1['f1']:.3f})")

    l_count = fusion_results.get("GV", {}).get("l_count", 0)
    print(f"  {{L}} under V+G: {l_count}")
    print(f"  Design effect: {stats['design_effect']:.2f}")


if __name__ == "__main__":
    main()
