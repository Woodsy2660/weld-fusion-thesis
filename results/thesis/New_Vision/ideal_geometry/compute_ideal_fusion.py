#!/usr/bin/env python3
"""
Compute fusion results with ideal geometry module.

Uses only standard library to avoid dependency issues.
"""

import json
import csv
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# Paths
BASE = Path(__file__).parent
THESIS_DIR = BASE.parent
ROOT = Path(__file__).resolve().parents[4]

# Frame elements
THETA = frozenset({"N", "U", "P", "L"})
CLASS_ORDER = ["N", "U", "P", "L"]

def load_csv(path):
    """Load CSV to list of dicts."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def parse_focal_element(s):
    """Parse '{N,U}' or 'Theta' to frozenset."""
    if s == "Theta" or s == "Θ" or not s:
        return THETA
    s = s.strip("{}")
    if not s:
        return THETA
    return frozenset(s.split(","))

def dempster_combine(bpa1, bpa2):
    """Combine two BPAs using Dempster's rule."""
    combined = defaultdict(float)
    conflict = 0.0

    for fe1, m1 in bpa1.items():
        for fe2, m2 in bpa2.items():
            intersection = fe1 & fe2
            mass = m1 * m2
            if not intersection:
                conflict += mass
            else:
                combined[intersection] += mass

    # Normalize
    if conflict >= 1.0:
        return {THETA: 1.0}

    norm = 1.0 - conflict
    if norm <= 0:
        return {THETA: 1.0}

    result = {}
    for fe, m in combined.items():
        result[fe] = m / norm

    return result

def pignistic_transform(bpa, theta):
    """Convert BPA to pignistic probability."""
    prob = defaultdict(float)

    for fe, m in bpa.items():
        if len(fe) > 0:
            share = m / len(fe)
            for elem in fe:
                prob[elem] += share

    return dict(prob)

def bpa_from_vision(cell_id, vision_dict):
    """Extract vision BPA for cell."""
    if cell_id not in vision_dict:
        return {THETA: 1.0}

    v = vision_dict[cell_id]
    bpa = {}
    for focal, mass in v.items():
        mass = float(mass)
        if mass <= 0:
            continue
        if focal == "m_Theta":
            bpa[THETA] = mass
        elif focal.startswith("m_"):
            fe = parse_focal_element(focal[2:])
            bpa[fe] = mass
    return bpa if bpa else {THETA: 1.0}

def bpa_from_geometry(cell_id, geom_rows):
    """Extract ideal geometry BPA for cell."""
    row = None
    for r in geom_rows:
        if r["cell_id"] == cell_id:
            row = r
            break

    if not row:
        return {THETA: 1.0}

    bpa = {}
    m_u = float(row.get("m_{U}", 0))
    m_npl = float(row.get("m_{N,P,L}", 0))
    m_theta = float(row.get("m_Theta", 0))

    if m_u > 0:
        bpa[frozenset({"U"})] = m_u
    if m_npl > 0:
        bpa[frozenset({"N", "P", "L"})] = m_npl
    if m_theta > 0:
        bpa[THETA] = m_theta

    return bpa if bpa else {THETA: 1.0}

def bpa_from_ut(cell_id, ut_rows):
    """Extract UT BPA for cell."""
    row = None
    for r in ut_rows:
        if r.get("cell_id") == cell_id:
            row = r
            break

    if not row:
        return {THETA: 1.0}

    bpa = {}
    for col, val in row.items():
        if col.startswith("m_") and val:
            try:
                mass = float(val)
            except:
                continue
            if mass <= 0:
                continue
            if "Theta" in col:
                bpa[THETA] = mass
            else:
                fe_str = col[2:].strip("{}")
                fe = frozenset(fe_str.split(","))
                bpa[fe] = mass

    return bpa if bpa else {THETA: 1.0}

def fuse_and_decide(bpa1, bpa2, bpa3=None):
    """Fuse 2 or 3 BPAs, return pignistic decision."""
    combined = dempster_combine(bpa1, bpa2)
    if bpa3:
        combined = dempster_combine(combined, bpa3)

    pignistic = pignistic_transform(combined, THETA)

    # Decision: max pignistic probability
    best_class = max(CLASS_ORDER, key=lambda c: pignistic.get(c, 0))
    return best_class, pignistic

def compute_metrics(y_true, y_pred, classes=CLASS_ORDER):
    """Compute precision, recall, F1 per class and macro."""
    # Build confusion matrix
    cm = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    metrics = {"per_class": {}}
    f1_scores = []

    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[t][c] for t in classes if t != c)
        fn = sum(cm[c][p] for p in classes if p != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        support = sum(cm[c][p] for p in classes)
        metrics["per_class"][c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
        f1_scores.append(f1)

    correct = sum(cm[c][c] for c in classes)
    total = sum(sum(cm[t][p] for p in classes) for t in classes)
    metrics["accuracy"] = correct / total if total > 0 else 0.0
    metrics["macro_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return metrics

def main():
    print("=" * 70)
    print("IDEAL GEOMETRY FUSION SIMULATION")
    print("=" * 70)

    # Load actual fusion results for comparison
    stats_path = ROOT / "results" / "thesis" / "ch4_report" / "source_data" / "statistics.json"
    actual_stats = json.loads(stats_path.read_text())

    # Load vision BPAs
    vision_bpa_path = ROOT / "results" / "vision" / "coupon_eval" / "vision_bpas.json"
    vision_bpas = json.loads(vision_bpa_path.read_text())

    # Load ideal geometry BPAs
    geom_rows = load_csv(BASE / "bpa_geometry_ideal.csv")

    # Load UT BPAs
    ut_bpa_path = ROOT / "results" / "ut" / "bpa_ut.csv"
    if ut_bpa_path.exists():
        ut_rows = load_csv(ut_bpa_path)
    else:
        # Fallback
        alt_path = THESIS_DIR.parent / "ch4_report" / "source_data" / "bpa_ut.csv"
        if alt_path.exists():
            ut_rows = load_csv(alt_path)
        else:
            ut_rows = []

    # Load labels
    labels_path = ROOT / "data" / "labels_v2_2026-10-26.csv"
    labels_rows = load_csv(labels_path)
    labels_map = {r["cell_id"]: r["true_class"] for r in labels_rows}

    # Get all cell_ids from geometry
    cell_ids = [r["cell_id"] for r in geom_rows]

    # Prepare results
    results = {
        "V_only": {"y_true": [], "y_pred": []},
        "G_ideal": {"y_true": [], "y_pred": []},
        "U_only": {"y_true": [], "y_pred": []},
        "VG_ideal": {"y_true": [], "y_pred": []},
        "VU": {"y_true": [], "y_pred": []},
        "GU_ideal": {"y_true": [], "y_pred": []},
        "VGU_ideal": {"y_true": [], "y_pred": []}
    }

    for cell_id in cell_ids:
        true_class = labels_map.get(cell_id)
        if not true_class:
            continue

        # Get BPAs
        v_bpa = bpa_from_vision(cell_id, vision_bpas)
        g_bpa = bpa_from_geometry(cell_id, geom_rows)
        u_bpa = bpa_from_ut(cell_id, ut_rows)

        # Single modality decisions (combine with vacuous)
        vacuous = {THETA: 1.0}
        v_dec, _ = fuse_and_decide(v_bpa, vacuous)
        g_dec, _ = fuse_and_decide(g_bpa, vacuous)
        u_dec, _ = fuse_and_decide(u_bpa, vacuous)

        # Fusion decisions
        vg_dec, _ = fuse_and_decide(v_bpa, g_bpa)
        vu_dec, _ = fuse_and_decide(v_bpa, u_bpa)
        gu_dec, _ = fuse_and_decide(g_bpa, u_bpa)
        vgu_dec, _ = fuse_and_decide(v_bpa, g_bpa, u_bpa)

        # Store results
        for key, dec in [
            ("V_only", v_dec), ("G_ideal", g_dec), ("U_only", u_dec),
            ("VG_ideal", vg_dec), ("VU", vu_dec), ("GU_ideal", gu_dec),
            ("VGU_ideal", vgu_dec)
        ]:
            results[key]["y_true"].append(true_class)
            results[key]["y_pred"].append(dec)

    # Compute metrics
    print(f"\nTotal cells: {len(results['V_only']['y_true'])}")
    print(f"Ground truth distribution: {Counter(results['V_only']['y_true'])}")

    print("\n" + "=" * 70)
    print("FUSION RESULTS WITH IDEAL GEOMETRY")
    print("=" * 70)

    fusion_metrics = {}
    for name, data in results.items():
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        metrics = compute_metrics(y_true, y_pred)
        fusion_metrics[name] = metrics

        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Macro-F1: {metrics['macro_f1']:.3f}")
        for c in CLASS_ORDER:
            pc = metrics['per_class'][c]
            print(f"    {c}: P={pc['precision']:.3f}, R={pc['recall']:.3f}, F1={pc['f1']:.3f} (n={pc['support']})")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: ACTUAL vs IDEAL GEOMETRY")
    print("=" * 70)

    mapping = {
        "V_only": "V",
        "G_ideal": "G",
        "U_only": "U",
        "VG_ideal": "V+G",
        "VU": "V+U",
        "GU_ideal": "G+U",
        "VGU_ideal": "V+G+U"
    }

    actual_fusion = actual_stats.get("fusion_combinations", {})

    comparison = []
    print(f"\n{'Configuration':<15} {'Actual F1':>12} {'Ideal F1':>12} {'Δ F1':>12}")
    print("-" * 55)

    for key, label in mapping.items():
        ideal_f1 = fusion_metrics[key]["macro_f1"]

        # Find actual
        actual_f1 = None
        if label == "V":
            actual_f1 = actual_stats.get("vision_macro_f1", 0)
        elif label == "G":
            actual_f1 = 0.204  # From ideal_geometry_model output
        elif label == "U":
            actual_f1 = actual_stats.get("ut_metrics", {}).get("f1_macro_mean", 0)
        else:
            for combo in actual_fusion.values():
                if combo.get("combination") == label:
                    actual_f1 = combo.get("macro_f1", 0)
                    break

        if actual_f1 is None:
            actual_f1 = 0.0

        delta = ideal_f1 - actual_f1
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        print(f"{label:<15} {actual_f1:>12.3f} {ideal_f1:>12.3f} {delta_str:>12}")

        comparison.append({
            "configuration": label,
            "actual_f1": actual_f1,
            "ideal_f1": ideal_f1,
            "delta_f1": delta
        })

    # Save full results
    output = {
        "generated": datetime.now().isoformat(),
        "description": "Fusion results with ideal geometry (L2 resolved, α_G = 0.000)",
        "alpha_values": {
            "vision": 0.404,
            "geometry_ideal": 0.000,
            "geometry_actual": 1.000,
            "ut": 0.241
        },
        "fusion_metrics": {},
        "comparison": comparison
    }

    for name, metrics in fusion_metrics.items():
        output["fusion_metrics"][name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class": metrics["per_class"]
        }

    output_path = BASE / "fusion_ideal_results.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {output_path}")

    # Generate thesis table format
    print("\n" + "=" * 70)
    print("TABLE 4.7 FORMAT: FUSION ABLATION WITH IDEAL GEOMETRY")
    print("=" * 70)

    print("\n| Configuration | N F1 | U F1 | P F1 | L F1 | Macro-F1 |")
    print("|---------------|------|------|------|------|----------|")
    for key in ["V_only", "G_ideal", "U_only", "VG_ideal", "VU", "GU_ideal", "VGU_ideal"]:
        m = fusion_metrics[key]
        pc = m["per_class"]
        label = mapping[key]
        print(f"| {label:<13} | {pc['N']['f1']:.3f}| {pc['U']['f1']:.3f}| {pc['P']['f1']:.3f}| {pc['L']['f1']:.3f}| {m['macro_f1']:.3f}    |")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: IDEAL GEOMETRY IMPACT")
    print("=" * 70)

    vgu_ideal = fusion_metrics["VGU_ideal"]["macro_f1"]
    vu_actual = None
    for combo in actual_fusion.values():
        if combo.get("combination") == "V+U":
            vu_actual = combo.get("macro_f1", 0)
            break
    if vu_actual is None:
        vu_actual = 0.664

    print(f"\nWith ideal geometry (α_G = 0.000):")
    print(f"  - G alone: F1 = {fusion_metrics['G_ideal']['macro_f1']:.3f} (perfect U detection)")
    print(f"  - V+G+U:   F1 = {vgu_ideal:.3f}")
    print(f"  - Actual V+U: F1 = {vu_actual:.3f}")
    print(f"  - Gain from ideal geometry: +{vgu_ideal - vu_actual:.3f}")

    print(f"\nKey insight: Geometry COULD be the highest-performing modality")
    print(f"for U-class detection if L2 trench artifact were resolved.")

    return output

if __name__ == "__main__":
    main()
