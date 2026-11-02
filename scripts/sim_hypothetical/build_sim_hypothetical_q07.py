#!/usr/bin/env python3
"""Simulated hypothetical vision at s=1, q=0.7, fused with measured UT and geometry.

The vision column is the seeded draw from simulate_ideal_vision (seed 20261101,
50 replicates). The representative replicate is the one whose V+U Dempster
macro-F1 is closest to the median. Ultrasonic masses are the measured masses
recovered by that script. Geometry masses are the measured discounted file
(alpha_G = 1, so every geometry BPA is vacuous).

Nothing here is a trained vision result. Outputs go only to
results/thesis/sim_hypothetical_q07/. Figure files carry the simulated /
hypothetical name. Figures are drawn without a caption, a title sentence, or a
stamp; axis labels and the marks already used by the chapter figure scripts stay.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

SIM_DIR = Path(__file__).resolve().parents[1] / "sim_ideal_vision"
sys.path.insert(0, str(SIM_DIR))

import simulate_ideal_vision as sim  # noqa: E402

Q_SKEW = 0.7
S_SKEW = 1.0
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
CLASSES = ("N", "U", "P", "L")
CONDITIONS = (
    ("V", "V", ("V",)),
    ("G", "G", ("G",)),
    ("U", "U", ("U",)),
    ("GV", "V+G", ("G", "V")),
    ("UV", "V+U", ("U", "V")),
    ("GU", "G+U", ("G", "U")),
    ("GUV", "V+G+U", ("G", "U", "V")),
)
DECLARED = (("V", "GV"), ("V", "UV"), ("G", "GU"), ("GV", "GUV"))
DECLARED_NAME = {
    ("V", "GV"): r"V $\to$ V+G",
    ("V", "UV"): r"V $\to$ V+U",
    ("G", "GU"): r"G $\to$ G+U",
    ("GV", "GUV"): r"V+G $\to$ V+G+U",
}


def repo_root() -> Path:
    return sim.find_repo_root()


def run_replicate_full(
    cells: list[dict],
    ut_bpa: list[dict],
    u_decisions: list[str],
    true: list[str],
    c_q: np.ndarray,
    s: float,
    f1_triv: float,
    rng: np.random.Generator,
) -> dict:
    """Same random calls as simulate_ideal_vision.run_replicate, plus the cell records."""
    targets = [cell["target"] for cell in cells]
    porosity = np.array([i for i, target in enumerate(targets) if target == "{P}"], dtype=int)
    n_flip = sim.hidden_count(len(porosity), s)
    if n_flip > 0:
        chosen = rng.choice(porosity, size=n_flip, replace=False)
        for index in np.atleast_1d(chosen):
            targets[int(index)] = "{N,U,L}"

    accuracy = float(np.mean([c_q[sim.OUTCOME_INDEX[target], sim.OUTCOME_INDEX[target]] for target in targets]))
    mu = float(np.clip(accuracy, 1e-6, 1.0 - 1e-6))
    alpha_par = mu * sim.BETA_CONCENTRATION
    beta_par = (1.0 - mu) * sim.BETA_CONCENTRATION

    preds: list[str] = []
    probs: list[dict[str, float]] = []
    for target in targets:
        row = c_q[sim.OUTCOME_INDEX[target]]
        drawn = int(rng.choice(3, p=row))
        pred = sim.OUTCOMES[drawn]
        confidence = float(np.clip(rng.beta(alpha_par, beta_par), sim.CONF_CLIP[0], sim.CONF_CLIP[1]))
        rest = (1.0 - confidence) / 2.0
        prob = {name: (confidence if name == pred else rest) for name in sim.OUTCOMES}
        scale = sum(prob.values())
        probs.append({name: value / scale for name, value in prob.items()})
        preds.append(pred)

    confidences = [prob[pred] for prob, pred in zip(probs, preds)]
    correct = [pred == target for pred, target in zip(preds, targets)]
    fold_scores = []
    for fold in (1, 2, 3, 4):
        idx = [i for i, cell in enumerate(cells) if cell["fold"] == fold]
        fold_scores.append(sim.macro_f1([targets[i] for i in idx], [preds[i] for i in idx], sim.OUTCOMES))
    f1_folds = float(np.mean(fold_scores))
    alpha = sim.alpha_from_f1(f1_folds, f1_triv)

    v_dec_d: list[str] = []
    v_dec_m: list[str] = []
    v_betp: list[dict[str, float]] = []
    v_k: list[float] = []
    uv_dec_d: list[str] = []
    uv_dec_m: list[str] = []
    uv_betp: list[dict[str, float]] = []
    uv_k: list[float] = []
    v_bpa: list[dict] = []
    for index, prob in enumerate(probs):
        bpa_v = sim.discount(prob, alpha)
        bet_v = sim.pignistic(bpa_v)
        v_dec_d.append(sim.decide(bet_v))
        alone_m, _ = sim.murphy([bpa_v])
        v_dec_m.append(sim.decide(sim.pignistic(alone_m)))
        v_betp.append(bet_v)
        v_k.append(0.0)
        fused_d, k_d = sim.dempster(bpa_v, ut_bpa[index])
        fused_m, _ = sim.murphy([bpa_v, ut_bpa[index]])
        bet_d = sim.pignistic(fused_d)
        uv_dec_d.append(sim.decide(bet_d))
        uv_dec_m.append(sim.decide(sim.pignistic(fused_m)))
        uv_betp.append(bet_d)
        uv_k.append(k_d)
        v_bpa.append(bpa_v)

    matrix = sim.confusion_counts(targets, preds)
    outcomes = sim.metrics_from_confusion(matrix)
    b_count, c_count = sim.discordant(true, u_decisions, uv_dec_d)
    return {
        "vision_f1_folds": f1_folds,
        "vision_f1_pooled": sim.macro_f1(targets, preds, sim.OUTCOMES),
        "outcome_p_recall": outcomes["{P}"]["recall"],
        "outcome_p_precision": outcomes["{P}"]["precision"],
        "ece": sim.expected_calibration_error(confidences, correct),
        "alpha": alpha,
        "outcomes": outcomes,
        "confusion": matrix,
        "V": sim.condition_metrics(true, v_dec_d, v_betp, v_k, v_dec_m),
        "UV": sim.condition_metrics(true, uv_dec_d, uv_betp, uv_k, uv_dec_m),
        "b": b_count,
        "c": c_count,
        "uv_f1_d": sim.macro_f1(true, uv_dec_d, sim.CLASSES),
        "preds": preds,
        "targets": targets,
        "probs": probs,
        "confidences": confidences,
        "v_bpa": v_bpa,
        "alpha_cell": alpha,
    }


def drop_zero(bpa: dict) -> dict:
    return {focal: mass for focal, mass in bpa.items() if mass > 1e-15}


def dempster_multi(bpas: list[dict]) -> tuple[dict, float]:
    usable = [drop_zero(bpa) for bpa in bpas if bpa]
    if not usable:
        return {sim.THETA: 1.0}, 0.0
    acc = dict(usable[0])
    total_k = 0.0
    for other in usable[1:]:
        acc, conflict = sim.dempster(acc, other)
        total_k = total_k + conflict - total_k * conflict
    return acc, total_k


def macro_f1_chapter(y_true: list[str], y_pred: list[str]) -> float:
    """Same four-class macro-F1 as scripts/chapter4_execute.py. A missing class scores 0."""
    scores = []
    for label in CLASSES:
        tp = sum(truth == label and pred == label for truth, pred in zip(y_true, y_pred))
        fp = sum(truth != label and pred == label for truth, pred in zip(y_true, y_pred))
        fn = sum(truth == label and pred != label for truth, pred in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(sum(scores) / len(scores))


def load_geometry(path: Path) -> dict[str, dict]:
    bpas = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            bpas[row["cell_id"]] = drop_zero(
                {
                    frozenset({"U"}): float(row["m_{U}"]),
                    frozenset({"N", "P", "L"}): float(row["m_{N,P,L}"]),
                    sim.THETA: float(row["m_Theta"]),
                }
            )
    return bpas


def fuse_cell(vision: dict, geometry: dict, ultrasonic: dict, mods: tuple[str, ...], method: str) -> dict:
    sources = {"V": vision, "G": geometry, "U": ultrasonic}
    bpas = [sources[name] for name in mods]
    if method == "dempster":
        combined, conflict = dempster_multi(bpas)
    elif method == "murphy":
        combined, conflict = sim.murphy([drop_zero(bpa) for bpa in bpas])
    else:
        raise ValueError(method)
    bet = sim.pignistic(combined)
    bel_pl = {}
    for label in CLASSES:
        belief, plausibility = sim.belief_plausibility(combined, frozenset({label}))
        bel_pl[label] = [belief, plausibility]
    return {
        "decision": sim.decide(bet),
        "K": float(conflict),
        "betp": {label: float(bet[label]) for label in CLASSES},
        "bel_pl": bel_pl,
        "nu_tie": sim.winners(bet) == {"N", "U"},
    }


def fuse_replicate(cells, rep, geom, ut_bpa) -> dict:
    fused = {key: {"dempster": [], "murphy": []} for key, _, _ in CONDITIONS}
    for index, cell in enumerate(cells):
        vision = rep["v_bpa"][index]
        geometry = geom.get(cell["cell_id"], {sim.THETA: 1.0})
        ultrasonic = ut_bpa[index]
        for key, _display, mods in CONDITIONS:
            for method in ("dempster", "murphy"):
                record = fuse_cell(vision, geometry, ultrasonic, mods, method)
                record["cell_id"] = cell["cell_id"]
                record["true_class"] = cell["true_class"]
                record["coupon_id"] = sim.coupon_of(cell["cell_id"])
                fused[key][method].append(record)
    return fused


def chapter_f1(rows: list[dict]) -> float:
    return macro_f1_chapter([row["true_class"] for row in rows], [row["decision"] for row in rows])


def bootstrap_cis(fused: dict, coupons: list[str], coupon_cells: dict[str, list[str]]) -> dict:
    random.seed(BOOTSTRAP_SEED)
    ci = {}
    for key, _display, _mods in CONDITIONS:
        for method in ("dempster", "murphy"):
            by_id = {row["cell_id"]: row for row in fused[key][method]}
            point = chapter_f1(list(by_id.values()))
            draws = []
            for _ in range(BOOTSTRAP_N):
                chosen = random.choices(coupons, k=len(coupons))
                sampled = [cell_id for coupon in chosen for cell_id in coupon_cells[coupon]]
                draws.append(
                    macro_f1_chapter(
                        [by_id[cell_id]["true_class"] for cell_id in sampled],
                        [by_id[cell_id]["decision"] for cell_id in sampled],
                    )
                )
            low, high = np.percentile(draws, [2.5, 97.5])
            ci[f"{key}_{method}"] = {"f1": float(point), "ci_95": [float(low), float(high)]}
    return ci


def design_effect(rows: list[dict]) -> dict:
    by_coupon = defaultdict(list)
    for row in rows:
        by_coupon[row["coupon_id"]].append(0 if row["decision"] == row["true_class"] else 1)
    groups = list(by_coupon.values())
    all_errors = [value for group in groups for value in group]
    grand = float(np.mean(all_errors))
    k = len(groups)
    n = len(all_errors)
    n_per = n / k
    ss_between = sum(len(group) * (float(np.mean(group)) - grand) ** 2 for group in groups)
    ss_within = sum(sum((value - float(np.mean(group))) ** 2 for value in group) for group in groups)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    if ms_within > 0:
        icc = (ms_between - ms_within) / (ms_between + (n_per - 1) * ms_within)
        icc = max(0.0, float(icc))
    else:
        icc = 0.0
    deff = 1.0 + (n_per - 1.0) * icc
    return {"icc": icc, "design_effect": deff, "n_eff": n / deff, "n": n, "k": k}


def holm(tests: list[dict]) -> list[dict]:
    ordered = sorted(tests, key=lambda row: (row["p_raw"], row["order"]))
    running = 0.0
    m = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, row["p_raw"] * (m - index))
        running = max(running, adjusted)
        row["p_holm"] = running
        row["significant"] = bool(running < 0.05)
    return tests


def mcnemar_tests(fused: dict) -> list[dict]:
    tests = []
    for order, (base, aug) in enumerate(DECLARED):
        before = {row["cell_id"]: row for row in fused[base]["dempster"]}
        after = {row["cell_id"]: row for row in fused[aug]["dempster"]}
        b = c = 0
        for cell_id, row in before.items():
            truth = row["true_class"]
            base_ok = row["decision"] == truth
            aug_ok = after[cell_id]["decision"] == truth
            if aug_ok and not base_ok:
                b += 1
            elif base_ok and not aug_ok:
                c += 1
        if b + c == 0:
            stat, p_raw = float("nan"), 1.0
        else:
            stat, p_raw = sim.mcnemar_stat(b, c)
        tests.append(
            {
                "comparison": f"{base}->{aug}",
                "base": base,
                "aug": aug,
                "b": int(b),
                "c": int(c),
                "chi2": stat,
                "p_raw": float(p_raw),
                "order": order,
            }
        )
    return holm(tests)


def rate_bundle(rows: list[dict]) -> dict[str, float]:
    """Screening rates on one decision list.

    Any defect: true class in {U, P, L} and the decision is not N.
    Volumetric: true class P and the decision is P.
    Undercut: true class U and the decision is U.
    False calls: true class N and the decision is not N.
    """

    def fraction(mask, hit) -> float:
        denom = sum(mask)
        if denom == 0:
            return float("nan")
        return sum(flag for flag, keep in zip(hit, mask) if keep) / denom

    truth = [row["true_class"] for row in rows]
    pred = [row["decision"] for row in rows]
    return {
        "any_defect": fraction([label != "N" for label in truth], [label != "N" for label in pred]),
        "volumetric": fraction([label == "P" for label in truth], [label == "P" for label in pred]),
        "undercut": fraction([label == "U" for label in truth], [label == "U" for label in pred]),
        "false_calls": fraction([label == "N" for label in truth], [label != "N" for label in pred]),
    }


def bootstrap_rates(fused: dict, coupons: list[str], coupon_cells: dict[str, list[str]]) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    out = {}
    for key, _display, _mods in CONDITIONS:
        by_id = {row["cell_id"]: row for row in fused[key]["dempster"]}
        point = rate_bundle(list(by_id.values()))
        bags = {name: [] for name in point}
        for _ in range(BOOTSTRAP_N):
            chosen = rng.choices(coupons, k=len(coupons))
            sampled = [by_id[cell_id] for coupon in chosen for cell_id in coupon_cells[coupon]]
            draw = rate_bundle(sampled)
            for name, value in draw.items():
                if value == value:
                    bags[name].append(value)
        packaged = {}
        for name, value in point.items():
            low, high = np.percentile(bags[name], [2.5, 97.5])
            packaged[name] = {
                "rate": float(value),
                "ci_95": [float(low), float(high)],
                "n_finite": len(bags[name]),
            }
        out[key] = packaged
    return out


def write_tex(path: Path, caption: str, lines: list[str]) -> None:
    if "sim_hypothetical" not in path.name:
        raise RuntimeError(path.name)
    text = "% Simulated: hypothetical. " + caption.strip() + "\n" + "\n".join(lines) + "\n"
    for char in text:
        if ord(char) > 127:
            raise RuntimeError(f"Non-ASCII {char!r} in {path.name}")
    path.write_text(text, encoding="ascii", newline="\n")


def tabular(spec: str, header: str, body: list[str]) -> list[str]:
    return [
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *[row + r" \\" for row in body],
        r"\bottomrule",
        r"\end{tabular}",
    ]


def pm(mean: float, sd: float, digits: int = 3) -> str:
    return rf"${mean:.{digits}f} \pm {sd:.{digits}f}$"


def fmt_p(p: float) -> str:
    if p != p:
        return "---"
    if p < 1e-3:
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / 10**exponent
        return rf"${mantissa:.2f}\times 10^{{{exponent}}}$"
    return f"{p:.4f}"


def fmt_chi(stat: float) -> str:
    if stat != stat:
        return "---"
    return f"{stat:.2f}"


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def geometry_scores(repo: Path) -> dict:
    summary = list(csv.DictReader((repo / "results" / "geometry" / "summary.csv").open(encoding="utf-8")))
    labels = list(csv.DictReader((repo / "data" / "labels_v2_2026-10-26.csv").open(encoding="utf-8")))
    label_map = {row["cell_id"]: row for row in labels}
    per_fold = defaultdict(lambda: {"y_true": [], "y_pred": []})
    y_true, y_pred = [], []
    cm = np.zeros((2, 2), dtype=int)
    for row in summary:
        if row["is_vacuous"] != "False":
            continue
        cell_id = row["cell_id"]
        if cell_id not in label_map:
            continue
        truth = "U" if label_map[cell_id]["true_class"] == "U" else "NPL"
        pred = "U" if float(row["m_U"]) > 0.5 else "NPL"
        y_true.append(truth)
        y_pred.append(pred)
        per_fold[label_map[cell_id]["fold"]]["y_true"].append(truth)
        per_fold[label_map[cell_id]["fold"]]["y_pred"].append(pred)
        cm[0 if truth == "U" else 1, 0 if pred == "U" else 1] += 1
    fold_f1 = [
        macro_f1_chapter_binary(per_fold[fold]["y_true"], per_fold[fold]["y_pred"])
        for fold in sorted(per_fold)
    ]
    return {
        "n": len(y_true),
        "mean": float(np.mean(fold_f1)),
        "std": float(np.std(fold_f1)),
        "pooled": macro_f1_chapter_binary(y_true, y_pred),
        "cm": cm,
        "summary": summary,
    }


def macro_f1_chapter_binary(y_true, y_pred) -> float:
    scores = []
    for label in ("U", "NPL"):
        tp = sum(truth == label and pred == label for truth, pred in zip(y_true, y_pred))
        fp = sum(truth != label and pred == label for truth, pred in zip(y_true, y_pred))
        fn = sum(truth == label and pred != label for truth, pred in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(sum(scores) / len(scores))


def draw_ut_readings(repo: Path, fusion_path: Path, out: Path) -> None:
    features = pd.read_csv(repo / "results" / "ut" / "features.csv").set_index("cell_id")
    labels = {
        row["cell_id"]: row["true_class"]
        for row in json.loads(fusion_path.read_text(encoding="utf-8"))["U"]["dempster"]
    }
    features["label"] = [labels[cell_id] for cell_id in features.index]
    fig, ax = plt.subplots(figsize=(160 * sim.MM, 78 * sim.MM))
    style = {
        "N": ("o", "#9E9E9E", "Sound"),
        "U": ("v", "#C0504D", "Undercut"),
        "P": ("s", "#C9A227", "Porosity"),
        "L": ("D", "#4A6FA5", "Lack of fusion"),
    }
    rng = np.random.default_rng(1)
    for key, (marker, colour, name) in style.items():
        subset = features[features.label == key]
        gated = subset[subset.located == 1]
        missed = subset[subset.located == 0]
        ax.scatter(
            gated.soundpath_mm,
            gated.amplitude_pct,
            marker=marker,
            s=22,
            facecolor=colour,
            edgecolor="#222222",
            linewidth=0.4,
            label=f"{name} ({len(subset)}; {len(gated)} gated)",
            zorder=3,
        )
        if len(missed):
            ax.scatter(
                rng.uniform(-3.2, -0.8, len(missed)),
                missed.amplitude_pct,
                marker=marker,
                s=14,
                facecolor="none",
                edgecolor=colour,
                linewidth=0.7,
                zorder=2,
            )
    ax.axhline(20, ls="--", lw=0.8, color="#333333")
    ax.text(44.5, 21, "gate 20%", ha="right", va="bottom", fontsize=7)
    ax.axhline(100, ls=":", lw=0.8, color="#333333")
    ax.text(44.5, 101, "full screen", ha="right", va="bottom", fontsize=7)
    for xpos, text in ((20, "half skip"), (40, "full skip")):
        ax.axvline(xpos, ls="-", lw=0.5, color="#BBBBBB", zorder=1)
        ax.text(xpos, 124, text, ha="center", va="bottom", fontsize=7)
    ax.axvspan(-3.6, -0.4, color="#F0F0F0", zorder=0)
    ax.text(-2, -6, "not\ngated", ha="center", va="top", fontsize=6.5)
    ax.set_xlim(-4, 45)
    ax.set_ylim(0, 130)
    ax.set_xlabel("Sound path (mm)")
    ax.set_ylabel("Peak amplitude (% FSH)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
    sim.save_figure(fig, out)


def focal_share(bpa: dict, names: dict[str, frozenset]) -> dict[str, float]:
    theta = bpa.get(sim.THETA, 0.0)
    scale = 1.0 - theta
    if scale <= 1e-12:
        return {name: 0.0 for name in names}
    return {name: bpa.get(focal, 0.0) / scale for name, focal in names.items()}


def draw_coupon(cells, rep, geom_summary, ut_bpa, out: Path) -> None:
    by_id = {cell["cell_id"]: index for index, cell in enumerate(cells)}
    geo = {row["cell_id"]: row for row in geom_summary}
    colours = {"N": "#9E9E9E", "U": "#E8A0A0", "P": "#F2C14E", "L": "#8DA0CB"}
    fills = {"Label": None, "Vision": "#D6E4F5", "Geometry": "#DCEFD9", "Ultrasonic": "#F5EBC8"}
    rows = ["Label", "Vision", "Geometry", "Ultrasonic"]
    vision_names = {"{P}": frozenset({"P"}), "{N,U}": frozenset({"N", "U"}), "{N,U,L}": frozenset({"N", "U", "L"})}
    ut_names = {"{P}": frozenset({"P"}), "{L}": frozenset({"L"}), "{N,U}": frozenset({"N", "U"})}
    fig, ax = plt.subplots(figsize=(160 * sim.MM, 52 * sim.MM))
    for i in range(10):
        cell_id = f"C14-{i:02d}"
        index = by_id[cell_id]
        label = cells[index]["true_class"]
        vision = focal_share(rep["v_bpa"][index], vision_names)
        vision_key = max(vision, key=vision.get)
        ultrasonic = focal_share(ut_bpa[index], ut_names)
        ut_key = max(ultrasonic, key=ultrasonic.get)
        geometry = geo[cell_id]
        vacuous = geometry["is_vacuous"] == "True"
        if vacuous:
            gtext = "n/m"
        else:
            gtext = f"$p_{{exc}}$\n{float(geometry['exceedance_prob']):.2f}"
        vals = {
            "Label": f"${label}$",
            "Vision": f"{vision_key}\n{vision[vision_key]:.2f}",
            "Geometry": gtext,
            "Ultrasonic": f"{ut_key}\n{ultrasonic[ut_key]:.2f}",
        }
        for j, name in enumerate(rows):
            y = len(rows) - 1 - j
            if name == "Label":
                face = colours[label]
            elif name == "Geometry" and vacuous:
                face = "white"
            else:
                face = fills[name]
            ax.add_patch(
                Rectangle(
                    (i, y),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="#333333",
                    lw=0.5,
                    hatch="///" if (name == "Geometry" and vacuous) else None,
                )
            )
            ax.text(
                i + 0.5,
                y + 0.5,
                vals[name],
                ha="center",
                va="center",
                fontsize=8.5 if name == "Label" else 6.5,
                linespacing=1.1,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none") if vals[name] == "n/m" else None,
            )
    for j, name in enumerate(rows):
        ax.text(-0.12, len(rows) - 0.5 - j, name, ha="right", va="center", fontsize=8)
    for i in range(10):
        ax.text(i + 0.5, -0.18, str(i), ha="center", va="top", fontsize=7)
    ax.text(5, -0.72, "Cell position", ha="center", va="top", fontsize=8)
    for boundary in (3, 7):
        ax.plot([boundary, boundary], [0, 4], color="black", lw=1.4)
    for xpos, zone in ((1.5, "A"), (5, "B"), (8.5, "C")):
        ax.text(xpos, 4.12, f"Zone {zone}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xlim(-1.7, 10.05)
    ax.set_ylim(-1.15, 4.55)
    ax.axis("off")
    sim.save_figure(fig, out)


def draw_confusion_modalities(rep_cm: np.ndarray, geom_cm: np.ndarray, ut_cm: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(160 * sim.MM, 58 * sim.MM))
    panels = (
        (rep_cm, ["{P}", "{N,U}", "{N,U,L}"], "Blues", "(a) Vision"),
        (geom_cm, ["{U}", "{N,P,L}"], "Greens", "(b) Geometry"),
        (ut_cm, ["{L}", "{N,U}", "{P}"], "Oranges", "(c) UT"),
    )
    for axis, (counts, ticks, cmap, title) in zip(axes, panels):
        image = axis.imshow(counts, cmap=cmap, aspect="equal")
        axis.set_xticks(range(len(ticks)), [sim.set_tex(tick) for tick in ticks], fontsize=7)
        axis.set_yticks(range(len(ticks)), [sim.set_tex(tick) for tick in ticks], fontsize=7)
        axis.set_xlabel("Predicted", fontsize=8)
        axis.set_ylabel("True", fontsize=8)
        peak = float(np.max(counts)) if counts.size else 0.0
        for row in range(counts.shape[0]):
            for col in range(counts.shape[1]):
                value = counts[row, col]
                axis.text(
                    col,
                    row,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > peak / 2 else "black",
                )
        axis.set_title(title, fontsize=9, loc="left", pad=3)
        image.set_clim(0, max(peak, 1))
    fig.tight_layout(w_pad=1.6)
    sim.save_figure(fig, out)


def draw_fusion_chapter(ci: dict, fused: dict, out: Path) -> None:
    keys = [key for key, _, _ in CONDITIONS]
    names = [rf"${display}$" if "+" not in display else "$" + display.replace("+", "{+}") + "$" for _, display, _ in CONDITIONS]
    # Keep the chapter script's math: V, G, U, V+G, V+U, G+U, V+G+U.
    names = [r"$V$", r"$G$", r"$U$", r"$V{+}G$", r"$V{+}U$", r"$G{+}U$", r"$V{+}G{+}U$"]
    f1 = [ci[f"{key}_dempster"]["f1"] for key in keys]
    low = [ci[f"{key}_dempster"]["ci_95"][0] for key in keys]
    high = [ci[f"{key}_dempster"]["ci_95"][1] for key in keys]
    colours = ["#C9A227" if "U" in key else "#BDBDBD" for key in keys]
    fig, (left, right) = plt.subplots(1, 2, figsize=(160 * sim.MM, 62 * sim.MM), gridspec_kw={"width_ratios": [1.15, 1]})
    xpos = np.arange(len(keys))
    left.bar(xpos, f1, color=colours, edgecolor="#333333", linewidth=0.5, width=0.7)
    left.errorbar(
        xpos,
        f1,
        yerr=[np.subtract(f1, low), np.subtract(high, f1)],
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
    )
    headroom = max(high) + 0.06
    for x, value, top in zip(xpos, f1, high):
        left.text(x, top + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    left.set_xticks(xpos, names, rotation=35, ha="right")
    left.set_ylabel(r"Macro-$F_1$")
    left.set_ylim(0, max(0.7, headroom))
    left.spines[["top", "right"]].set_visible(False)
    left.text(-0.16, 1.02, "(a)", transform=left.transAxes, fontsize=9)
    recall = np.zeros((4, len(keys)))
    for column, key in enumerate(keys):
        rows = fused[key]["dempster"]
        totals = Counter(row["true_class"] for row in rows)
        hits = Counter(row["true_class"] for row in rows if row["decision"] == row["true_class"])
        for row_index, label in enumerate(CLASSES):
            recall[row_index, column] = hits[label] / totals[label]
    image = right.imshow(recall, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for row in range(4):
        for col in range(len(keys)):
            value = recall[row, col]
            right.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.55 else "black")
    right.set_xticks(range(len(keys)), names, rotation=35, ha="right")
    right.set_yticks(range(4), [rf"${label}$" for label in CLASSES])
    right.set_ylabel("Label")
    colorbar = fig.colorbar(image, ax=right, fraction=0.046, pad=0.03)
    colorbar.set_label("Recall")
    colorbar.outline.set_linewidth(0.5)
    right.text(-0.22, 1.02, "(b)", transform=right.transAxes, fontsize=9)
    fig.tight_layout(w_pad=1.8)
    sim.save_figure(fig, out)
    return recall


def draw_sim_confusion(measured: np.ndarray, replicates: list[dict], out: Path) -> None:
    fig = plt.figure(figsize=(160 * sim.MM, 78 * sim.MM))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], left=0.08, right=0.98, bottom=0.16, top=0.90, wspace=0.38)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cax = fig.add_subplot(grid[0, 2])
    image = sim.draw_confusion(axes[0], measured, simulated=False)
    axes[0].set_title("(a) measured", fontsize=9, loc="left", pad=3)
    sim.draw_confusion(axes[1], sim.mean_confusion(replicates), simulated=True)
    axes[1].set_ylabel("")
    axes[1].set_title("(b) simulated", fontsize=9, loc="left", pad=3)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Row share")
    colorbar.outline.set_linewidth(0.5)
    sim.save_figure(fig, out)


def draw_sim_fusion(context: dict, pack: dict, replicates: list[dict], out: Path) -> None:
    measured = context["measured_conditions"]
    fig = plt.figure(figsize=(160 * sim.MM, 78 * sim.MM))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.22], left=0.07, right=0.98, bottom=0.22, top=0.90, wspace=0.40)
    axis = fig.add_subplot(outer[0, 0])
    inner = outer[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.10)
    heat = fig.add_subplot(inner[0, 0])
    cax = fig.add_subplot(inner[0, 1])
    groups = ((r"$V$", sim.COLOUR_V, "V"), (r"$U$", sim.COLOUR_U, "U"), (r"$V{+}U$", sim.COLOUR_FUSED, "UV"))
    xpos = np.arange(len(groups))
    width = 0.36
    for index, (label, colour, key) in enumerate(groups):
        if key == "U":
            axis.bar(xpos[index], measured[key]["f1_d"], width=width, color=colour, edgecolor="#222222", linewidth=0.5)
            continue
        axis.bar(xpos[index] - width / 2, measured[key]["f1_d"], width=width, color=colour, edgecolor="#222222", linewidth=0.5)
        mean, sd = pack[f"{key}:f1_d"]
        axis.bar(
            xpos[index] + width / 2,
            mean,
            width=width,
            color=colour,
            edgecolor="#222222",
            linewidth=0.5,
            hatch="///",
        )
        axis.errorbar(xpos[index] + width / 2, mean, yerr=sd, fmt="none", ecolor="#222222", elinewidth=0.7, capsize=2)
    axis.set_xticks(xpos, [label for label, _, _ in groups])
    axis.set_ylabel(r"Macro-$F_1$")
    axis.set_ylim(0.0, 1.0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor="0.75", edgecolor="#222222", label="Measured"),
            Patch(facecolor="0.75", edgecolor="#222222", hatch="///", label="Simulated"),
            Patch(facecolor=sim.COLOUR_U, edgecolor="#222222", label=r"$U$ reference"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )
    axis.set_title("(a)", fontsize=9, loc="left", pad=2)
    columns = []
    col_labels = []
    for key, label in (("V", r"$V$"), ("UV", r"$V{+}U$")):
        columns.append([measured[key]["recall"][lab] for lab in CLASSES])
        col_labels.append(label + " meas.")
        columns.append([float(np.mean([rep[key]["recall"][lab] for rep in replicates])) for lab in CLASSES])
        col_labels.append(label + " sim.")
    columns.insert(2, [measured["U"]["recall"][lab] for lab in CLASSES])
    col_labels.insert(2, r"$U$")
    matrix = np.asarray(columns, dtype=float).T
    image = heat.imshow(matrix, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            heat.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white" if value > 0.62 else "black")
    heat.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right")
    heat.set_yticks(range(len(CLASSES)), [rf"${lab}$" for lab in CLASSES])
    heat.set_ylabel("Label")
    heat.set_title("(b)", fontsize=9, loc="left", pad=2)
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Recall")
    colorbar.outline.set_linewidth(0.5)
    sim.save_figure(fig, out)


def draw_curve(sweep: dict[float, list[dict]], measured_f1: float, u_f1: float, skew_f1: float, out: Path) -> None:
    fig, axis = plt.subplots(figsize=(160 * sim.MM, 78 * sim.MM))
    points = []
    for q in sim.Q_GRID:
        reps = sweep[q]
        x_mean, _ = sim.mean_sd([rep["vision_f1_folds"] for rep in reps])
        y_f1, y_sd = sim.mean_sd([rep["UV"]["f1_d"] for rep in reps])
        y_rec, y_rec_sd = sim.mean_sd([rep["UV"]["recall"]["P"] for rep in reps])
        points.append((q, x_mean, y_f1, y_sd, y_rec, y_rec_sd))
    xs = [item[1] for item in points]
    y_f1 = np.asarray([item[2] for item in points])
    y_sd = np.asarray([item[3] for item in points])
    y_rec = np.asarray([item[4] for item in points])
    y_rec_sd = np.asarray([item[5] for item in points])
    axis.plot(xs, y_f1, color="0.2", marker="o", markersize=3.2, linewidth=1.0, label=r"V+U macro-$F_1$, s = 1")
    axis.fill_between(xs, y_f1 - y_sd, y_f1 + y_sd, color="0.2", alpha=0.12, linewidth=0)
    axis.plot(xs, y_rec, color="0.45", linestyle="--", marker="s", markersize=3.2, linewidth=1.0, label=r"V+U $P$ recall, s = 1")
    axis.fill_between(xs, y_rec - y_rec_sd, y_rec + y_rec_sd, color="0.45", alpha=0.08, linewidth=0)
    axis.axhline(u_f1, color=sim.COLOUR_U, linewidth=0.9, label="U alone")
    axis.axvline(measured_f1, color="black", linestyle=":", linewidth=1.0, label=r"Measured vision $F_1$")
    axis.axvline(skew_f1, color="black", linestyle="-.", linewidth=1.0, label=r"s = 1, q = 0.7")
    axis.set_xlabel(r"Simulated vision macro-$F_1$")
    axis.set_ylabel(r"V+U macro-$F_1$ or $P$ recall")
    axis.set_ylim(0.0, 1.02)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower left", bbox_to_anchor=(0.18, 0.02), fontsize=7, frameon=False, borderaxespad=0.0)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.96)
    sim.save_figure(fig, out)


def mass_record(bpa: dict) -> dict:
    return {
        "m_{P}": bpa.get(frozenset({"P"}), 0.0),
        "m_{N,U}": bpa.get(frozenset({"N", "U"}), 0.0),
        "m_{N,U,L}": bpa.get(frozenset({"N", "U", "L"}), 0.0),
        "m_Theta": bpa.get(sim.THETA, 0.0),
    }


def main() -> None:
    root = repo_root()
    out = root / "results" / "thesis" / "sim_hypothetical_q07"
    figures = out / "figures"
    tables = out / "tables"
    data = out / "data"
    for folder in (figures, tables, data):
        folder.mkdir(parents=True, exist_ok=True)

    args = sim.parse_args()
    labels, fusion, calibration = sim.load_inputs(args)
    context = sim.run_gate(labels, fusion, calibration)
    sim.print_gate(context["rows"])
    if not context["ok"]:
        raise SystemExit("Validation gate failed. No simulated hypothetical outputs were written.")

    rng = np.random.default_rng(sim.SEED)
    sweep: dict[float, list[dict]] = {}
    for q in sim.Q_GRID:
        c_q = sim.mix_confusion(context["c_real"], q)
        replicates = [
            run_replicate_full(
                context["cells"],
                context["ut_bpa"],
                context["u_decisions"],
                context["true"],
                c_q,
                S_SKEW,
                context["f1_triv"],
                rng,
            )
            for _ in range(sim.N_REPS)
        ]
        sweep[q] = replicates
        mean, sd = sim.mean_sd([rep["vision_f1_folds"] for rep in replicates])
        print(f"simulated s={S_SKEW:.1f} q={q:.1f}  vision macro-F1 {mean:.4f} +/- {sd:.4f}")

    replicates = sweep[Q_SKEW]
    vision_mean, vision_sd = sim.mean_sd([rep["vision_f1_folds"] for rep in replicates])
    if abs(vision_mean - 0.7968) > 0.0015 or abs(vision_sd - 0.0324) > 0.0015:
        raise SystemExit(f"Skewed vision F1 {vision_mean:.4f} +/- {vision_sd:.4f} does not match the seeded s=1 q=0.7 row.")
    pack = sim.level_pack(replicates)
    rep_index = sim.representative_index(replicates)
    rep = replicates[rep_index]
    print(f"representative replicate {rep_index}  V+U F1 {rep['uv_f1_d']:.4f}  b={rep['b']} c={rep['c']}  alpha={rep['alpha']:.4f}")

    geom = load_geometry(root / "results" / "geometry" / "bpa_geometry.csv")
    fused = fuse_replicate(context["cells"], rep, geom, context["ut_bpa"])
    # The simulator's own V and V+U Dempster decisions are the reference for this draw.
    v_mismatch = sum(row["decision"] != decision for row, decision in zip(fused["V"]["dempster"], [sim.decide(sim.pignistic(bpa)) for bpa in rep["v_bpa"]]))
    if v_mismatch:
        raise SystemExit(f"Vision fusion disagrees with the draw on {v_mismatch} cells.")

    coupons = sorted({cell["cell_id"].split("-", 1)[0] for cell in context["cells"]})
    # Prefer the label file's coupon id when it is present and complete.
    coupon_of = {}
    if "coupon_id" in labels.columns:
        for row in labels.to_dict(orient="records"):
            coupon_of[row["cell_id"].strip()] = row["coupon_id"].strip()
    for key in fused:
        for method in fused[key]:
            for row in fused[key][method]:
                row["coupon_id"] = coupon_of.get(row["cell_id"], row["coupon_id"])
    coupons = sorted({row["coupon_id"] for row in fused["V"]["dempster"]})
    coupon_cells = {coupon: [] for coupon in coupons}
    for row in fused["V"]["dempster"]:
        coupon_cells[row["coupon_id"]].append(row["cell_id"])

    print("Bootstrapping coupon clusters, seed 42, 2000 resamples.")
    ci = bootstrap_cis(fused, coupons, coupon_cells)
    effect = design_effect(fused["GUV"]["dempster"])
    tests = mcnemar_tests(fused)
    rates = bootstrap_rates(fused, coupons, coupon_cells)
    geom_stats = geometry_scores(root)

    ut_metrics = json.loads((root / "results" / "ut" / "metrics_cv.json").read_text(encoding="utf-8"))
    alpha_ut = json.loads((root / "results" / "ut" / "alpha_ut.json").read_text(encoding="utf-8"))
    inconsistencies = json.loads(
        (root / "results" / "thesis" / "ch4_evidence" / "data" / "inconsistencies_resolved.json").read_text(encoding="utf-8")
    )["answers"]

    g_decisions = [row["decision"] for row in fused["G"]["dempster"]]
    print(f"Geometry decisions: {Counter(g_decisions)}")
    for key, display, _mods in CONDITIONS:
        dem = fused[key]["dempster"]
        print(
            f"{display:8} F1 {ci[key + '_dempster']['f1']:.3f}  "
            f"Murphy {ci[key + '_murphy']['f1']:.3f}  "
            f"K {np.mean([row['K'] for row in dem]):.3f}  "
            f"L {sum(row['decision'] == 'L' for row in dem)}  "
            f"N/U {sum(row['nu_tie'] for row in dem)}  "
            f"correct {sum(row['decision'] == row['true_class'] for row in dem)}"
        )

    sim.setup_style()
    draw_ut_readings(root, args.fusion, figures / "sim_hypothetical_fig_res_ut_readings.pdf")
    draw_coupon(
        context["cells"],
        rep,
        geom_stats["summary"],
        context["ut_bpa"],
        figures / "sim_hypothetical_fig_res_coupon.pdf",
    )
    draw_confusion_modalities(
        rep["confusion"],
        geom_stats["cm"],
        np.asarray(ut_metrics["confusion_matrix"], dtype=float),
        figures / "sim_hypothetical_fig_res_confusion.pdf",
    )
    draw_fusion_chapter(ci, fused, figures / "sim_hypothetical_fig_res_fusion.pdf")
    draw_sim_confusion(context["measured_vision"]["confusion"], replicates, figures / "sim_hypothetical_fig_confusion.pdf")
    draw_sim_fusion(context, pack, replicates, figures / "sim_hypothetical_fig_fusion.pdf")
    draw_curve(
        sweep,
        context["measured_vision"]["f1_mean"],
        context["measured_conditions"]["U"]["f1_d"],
        vision_mean,
        figures / "sim_hypothetical_fig_curve.pdf",
    )

    # Tables that match the sim_vision pack, for this one skewed level.
    vision = context["measured_vision"]
    write_tex(
        tables / "sim_hypothetical_t_modality.tex",
        "s=1, q=0.7 vision outcomes over 50 replicates. Population SD. Ultrasonic row is measured.",
        tabular(
            "lcccccc",
            r"Source & macro-$F_1$ & pooled $F_1$ & $\{P\}$ recall & $\{P\}$ precision & ECE & $\alpha$",
            [
                "Measured vision & "
                + " & ".join(
                    [
                        pm(vision["f1_mean"], vision["f1_std"]),
                        f"{vision['pooled']:.3f}",
                        f"{vision['p_recall']:.3f}",
                        f"{vision['p_precision']:.3f}",
                        f"{vision['ece']:.3f}",
                        f"{vision['alpha']:.3f}",
                    ]
                ),
                "Simulated vision & "
                + " & ".join(
                    [
                        pm(*pack["f1"]),
                        pm(*pack["pooled"]),
                        pm(*pack["p_recall"]),
                        pm(*pack["p_precision"]),
                        pm(*pack["ece"]),
                        pm(*pack["alpha"]),
                    ]
                ),
            ],
        ),
    )
    element_rows = []
    for name in sim.OUTCOMES:
        element_rows.append(
            "Simulated & "
            + sim.set_tex(name)
            + " & "
            + " & ".join(pm(*pack[f"{name}:{metric}"]) for metric in ("precision", "recall", "f1", "support"))
        )
    write_tex(
        tables / "sim_hypothetical_t_elements.tex",
        "Outcome precision, recall and F1 for the simulated s=1, q=0.7 vision draw. Mean and population SD over 50 replicates.",
        tabular(
            "llcccc",
            r"Source & Outcome & Precision & Recall & $F_1$ & Support",
            element_rows,
        ),
    )
    ablation_rows = []
    for title, key in (("V", "V"), ("U", "U"), ("V+U", "UV")):
        if key == "U":
            measured = context["measured_conditions"]["U"]
            ablation_rows.append(
                f"{title} & measured & {measured['f1_d']:.3f} & {measured['f1_m']:.3f} & "
                f"{measured['mean_k']:.3f} & {measured['l_dec']} & {measured['nu_ties']}"
            )
            continue
        ablation_rows.append(
            f"{title} & simulated & {pm(*pack[key + ':f1_d'])} & {pm(*pack[key + ':f1_m'])} & "
            f"{pm(*pack[key + ':mean_k'])} & {pm(*pack[key + ':l_dec'], 1)} & {pm(*pack[key + ':nu_ties'], 1)}"
        )
    write_tex(
        tables / "sim_hypothetical_t_ablation.tex",
        "Decision macro-F1 for simulated vision and for simulated vision combined with measured ultrasonics. Mean and population SD over 50 replicates. The U row is the measured sensor.",
        tabular(
            "llccccc",
            r"Condition & Source & Dempster $F_1$ & Murphy $F_1$ & mean $K$ & $\{L\}$ & N/U ties",
            ablation_rows,
        ),
    )
    recall_rows = []
    for title, key in (("V", "V"), ("U", "U"), ("V+U", "UV")):
        if key == "U":
            recalls = context["measured_conditions"]["U"]["recall"]
            recall_rows.append(f"{title} & measured & " + " & ".join(f"{recalls[label]:.3f}" for label in CLASSES))
        else:
            recall_rows.append(
                f"{title} & simulated & " + " & ".join(pm(*pack[f"{key}:recall:{label}"]) for label in CLASSES)
            )
    write_tex(
        tables / "sim_hypothetical_t_recall.tex",
        "Per-class recall of the pignistic decision. Simulated rows are mean and population SD over 50 replicates.",
        tabular("llcccc", r"Condition & Source & N & U & P & L", recall_rows),
    )
    med_stat, med_p = sim.mcnemar_stat(rep["b"], rep["c"])
    mean_b, _ = sim.mean_sd([item["b"] for item in replicates])
    mean_c, _ = sim.mean_sd([item["c"] for item in replicates])
    mean_stat, mean_p = sim.mcnemar_stat(mean_b, mean_c)
    write_tex(
        tables / "sim_hypothetical_t_mcnemar.tex",
        "U to V+U on the simulated draw. Not one of the four declared chapter tests. Median replicate and the mean of the discordant counts.",
        tabular(
            "lcccccc",
            r"Aggregate & $b$ & $c$ & $\chi^2$ & $p$ & replicate & note",
            [
                f"Median replicate & {rep['b']} & {rep['c']} & {fmt_chi(med_stat)} & {fmt_p(med_p)} & {rep_index} & closest V+U $F_1$ to the median",
                f"Mean counts & {mean_b:.2f} & {mean_c:.2f} & {fmt_chi(mean_stat)} & {fmt_p(mean_p)} & 50 & population of replicates",
            ],
        ),
    )

    # Chapter-shaped tables from the representative draw plus measured geometry and UT.
    fold_counts = defaultdict(lambda: {"N": 0, "U": 0, "P": 0, "L": 0, "n": 0})
    cell_fold = {}
    for cell in context["cells"]:
        fold_counts[cell["fold"]][cell["true_class"]] += 1
        fold_counts[cell["fold"]]["n"] += 1
        cell_fold[cell["cell_id"]] = str(cell["fold"])
    geom_measured = defaultdict(int)
    for row in geom_stats["summary"]:
        if row["is_vacuous"] == "False" and row["cell_id"] in cell_fold:
            geom_measured[cell_fold[row["cell_id"]]] += 1
    data_rows = []
    csv_data = []
    for fold in (1, 2, 3, 4):
        stats = fold_counts[fold]
        data_rows.append(
            f"{fold} & {stats['N']} & {stats['U']} & {stats['P']} & {stats['L']} & {stats['n']} & "
            f"{stats['n']} & {geom_measured[str(fold)]} & {stats['n']}"
        )
        csv_data.append(
            {
                "fold": fold,
                "N": stats["N"],
                "U": stats["U"],
                "P": stats["P"],
                "L": stats["L"],
                "total": stats["n"],
                "vision_simulated": stats["n"],
                "geometry_measured": geom_measured[str(fold)],
                "ut_measured": stats["n"],
            }
        )
    write_tex(
        tables / "sim_hypothetical_t_res_data.tex",
        "Measured v2 labels per fold. Vision cells are the simulated assignment, which covers every cell. Geometry and UT counts are measured.",
        tabular(
            "lcccccccc",
            r"Fold & N & U & P & L & Total & Vision (sim.) & Geometry & UT",
            data_rows,
        ),
    )
    save_csv(tables / "sim_hypothetical_t_res_data.csv", csv_data)

    live = inconsistencies["geometry_repeatability"]["live_values"]
    dlim = inconsistencies["geometry_repeatability"]["d_lim_over_u"]
    ut_map = inconsistencies["ut_mapping_check"]
    ut_set = inconsistencies["ut_export_check"]["expected_settings"]
    verify_rows = [
        r"Vision (sim.) & Split leakage & not applicable to a seeded draw",
        r"Vision (sim.) & Held-out public source & not applicable to a seeded draw",
        r"Vision (sim.) & Nearest-neighbour audit & not applicable to a seeded draw",
        "Vision (sim.) & ECE, 15 bins & " + pm(*pack["ece"]),
        r"Vision (sim.) & $\alpha$ from fold-mean $F_1$ & " + pm(*pack["alpha"]),
        (
            "Geometry (measured) & repeatability $u$ (mm) & "
            f"mis={live['misalignment_mm']}, uc={live['undercut_mm']}, rf={live['reinforcement_mm']}"
        ),
        (
            r"Geometry (measured) & $d_{\mathrm{lim}}/u$ & "
            f"mis={dlim['misalignment']}, uc={dlim['undercut']}, rf={dlim['reinforcement']}"
        ),
        (
            "UT (measured) & instrument settings & "
            f"Vel={ut_set['Vel']}, Zero={ut_set['Zero']}, gate={ut_set['Gate_Level']}, angle={ut_set['Angle']}"
        ),
        (
            "UT (measured) & mapping $r$ & "
            f"$r={ut_map['r_15_coupon']}$, $n={ut_map['n_coupons']}$, $p{ut_map['p_15_coupon']}$"
        ),
    ]
    write_tex(
        tables / "sim_hypothetical_t_res_verify.tex",
        "Verification. Vision leakage, held-out source and nearest-neighbour rows are not properties of this draw. Geometry and UT rows are the measured checks.",
        tabular("llp{7.2cm}", r"Modality & Check & Outcome", verify_rows),
    )

    modality_rows = [
        "Vision (sim.) & 150 & "
        + pm(*pack["f1"])
        + " & "
        + f"{pack['pooled'][0]:.3f} & 0.257 & "
        + f"{pack['ece'][0]:.3f} & "
        + f"{pack['alpha'][0]:.3f}",
        (
            f"Geometry (measured) & {geom_stats['n']} & "
            f"{pm(geom_stats['mean'], geom_stats['std'])} & {geom_stats['pooled']:.3f} & 0.426 & N/A & 1.000"
        ),
        (
            f"UT (measured) & 150 & {pm(ut_metrics['f1_macro_mean'], ut_metrics['f1_macro_std'])} & "
            f"{ut_metrics['f1_macro_overall']:.3f} & {ut_metrics['baseline_most_frequent_f1']:.3f} & "
            f"{inconsistencies['ut_ece']['ece_uncalibrated']:.3f} & {alpha_ut['alpha']:.3f}"
        ),
    ]
    write_tex(
        tables / "sim_hypothetical_t_res_modality.tex",
        "Vision row is the s=1, q=0.7 simulator: fold-mean macro-F1, population SD over 50 replicates. Pooled F1, ECE and alpha in that row are the means. Geometry and UT are measured.",
        tabular(
            "lcccccc",
            r"Modality & Cells & $F_1$ (mean $\pm$ SD) & Pooled $F_1$ & Baseline & ECE & $\alpha$",
            modality_rows,
        ),
    )

    abl_rows = []
    abl_csv = []
    for key, display, _mods in CONDITIONS:
        dem = fused[key]["dempster"]
        mur = fused[key]["murphy"]
        dem_ci = ci[f"{key}_dempster"]
        mur_ci = ci[f"{key}_murphy"]
        mean_k = float(np.mean([row["K"] for row in dem]))
        mur_k = float(np.mean([row["K"] for row in mur]))
        l_count = sum(row["decision"] == "L" for row in dem)
        ties = sum(row["nu_tie"] for row in dem)
        abl_rows.append(
            f"{display} & {dem_ci['f1']:.3f} & [{dem_ci['ci_95'][0]:.3f}, {dem_ci['ci_95'][1]:.3f}] & "
            f"{mean_k:.3f} & {l_count} & {ties} & {mur_ci['f1']:.3f} & {mur_k:.3f}"
        )
        abl_csv.append(
            {
                "condition": display,
                "dempster_f1": f"{dem_ci['f1']:.6f}",
                "ci_low": f"{dem_ci['ci_95'][0]:.6f}",
                "ci_high": f"{dem_ci['ci_95'][1]:.6f}",
                "mean_K": f"{mean_k:.6f}",
                "L_decisions": l_count,
                "NU_ties": ties,
                "murphy_f1": f"{mur_ci['f1']:.6f}",
                "murphy_mean_K": f"{mur_k:.6f}",
            }
        )
    write_tex(
        tables / "sim_hypothetical_t_res_ablation.tex",
        f"Seven conditions on representative replicate {rep_index}. Dempster macro-F1 with coupon-bootstrap 95 percent intervals, seed {BOOTSTRAP_SEED}, {BOOTSTRAP_N} resamples. Geometry is the measured vacuous BPA.",
        tabular(
            "lccccccc",
            r"Condition & Dempster $F_1$ & 95\% CI & mean $K$ & $\{L\}$ & N/U ties & Murphy $F_1$ & Murphy $K$",
            abl_rows,
        ),
    )
    save_csv(tables / "sim_hypothetical_t_res_ablation.csv", abl_csv)

    dec_rows = []
    dec_csv = []
    for key, display, _mods in CONDITIONS:
        rows = fused[key]["dempster"]
        counts = Counter(row["decision"] for row in rows)
        correct = sum(row["decision"] == row["true_class"] for row in rows)
        dec_rows.append(
            f"{display} & {counts['N']} & {counts['U']} & {counts['P']} & {counts['L']} & {correct}"
        )
        dec_csv.append(
            {"condition": display, "N": counts["N"], "U": counts["U"], "P": counts["P"], "L": counts["L"], "correct": correct}
        )
    write_tex(
        tables / "sim_hypothetical_t_res_decisions.tex",
        f"Pignistic decisions on representative replicate {rep_index}. Alphabetical tie-break, so a two-way N/U tie is counted as N.",
        tabular("lccccc", r"Condition & N & U & P & L & Correct", dec_rows),
    )
    save_csv(tables / "sim_hypothetical_t_res_decisions.csv", dec_csv)

    det_rows = []
    det_csv = []
    for key, display, _mods in CONDITIONS:
        item = rates[key]
        bits = [display]
        flat = {"condition": display}
        for name in ("any_defect", "volumetric", "undercut", "false_calls"):
            rate = item[name]["rate"]
            low, high = item[name]["ci_95"]
            bits.append(f"{rate:.3f}")
            bits.append(f"[{low:.3f}, {high:.3f}]")
            flat[name] = f"{rate:.6f}"
            flat[name + "_low"] = f"{low:.6f}"
            flat[name + "_high"] = f"{high:.6f}"
        det_rows.append(" & ".join(bits))
        det_csv.append(flat)
    write_tex(
        tables / "sim_hypothetical_t_res_detection.tex",
        "Screening on the representative draw. Any defect: true in {U,P,L} and decision not N. Volumetric: true P and decision P. Undercut: true U and decision U. False calls: true N and decision not N. Intervals are coupon bootstraps, seed 42, 2000 resamples.",
        tabular(
            "lcccccccc",
            r"Condition & Any defect & 95\% CI & Volumetric & 95\% CI & Undercut & 95\% CI & False calls & 95\% CI",
            det_rows,
        ),
    )
    save_csv(tables / "sim_hypothetical_t_res_detection.csv", det_csv)

    mc_rows = []
    for (base, aug) in DECLARED:
        row = next(item for item in tests if item["base"] == base and item["aug"] == aug)
        mc_rows.append(
            f"{DECLARED_NAME[(base, aug)]} & {row['b']} & {row['c']} & {fmt_chi(row['chi2'])} & "
            f"{fmt_p(row['p_raw'])} & {fmt_p(row['p_holm'])}"
        )
    write_tex(
        tables / "sim_hypothetical_t_res_mcnemar.tex",
        (
            f"Four declared tests on representative replicate {rep_index}. "
            f"b is correct only after the added sensor. Holm uses m=4 with step-down monotonicity. "
            f"rho={effect['icc']:.3f}, design effect={effect['design_effect']:.2f}, n_eff={effect['n_eff']:.1f} "
            "from the V+G+U Dempster errors."
        ),
        tabular(
            "lccccc",
            r"Comparison & $b$ & $c$ & $\chi^2$ & $p$ & Holm $p$",
            mc_rows,
        ),
    )

    vision_rows = []
    for index, cell in enumerate(context["cells"]):
        record = mass_record(rep["v_bpa"][index])
        record.update(
            {
                "cell_id": cell["cell_id"],
                "true_class": cell["true_class"],
                "target": rep["targets"][index],
                "outcome": rep["preds"][index],
                "alpha_V": rep["alpha"],
                "replicate_index": rep_index,
            }
        )
        vision_rows.append(record)
    (data / "sim_hypothetical_vision_bpas.json").write_text(json.dumps(vision_rows, indent=2), encoding="utf-8")

    fusion_out = {
        "label": "simulated hypothetical vision, s=1, q=0.7, representative replicate",
        "seed": sim.SEED,
        "s": S_SKEW,
        "q": Q_SKEW,
        "n_replicates": sim.N_REPS,
        "replicate_index": rep_index,
        "alpha_V": rep["alpha"],
        "vision_f1_folds": rep["vision_f1_folds"],
        "ece": rep["ece"],
        "ut_source": "results/thesis/data/fusion_results.json recovered masses",
        "geometry_source": "results/geometry/bpa_geometry.csv",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_N,
    }
    for key, _display, _mods in CONDITIONS:
        fusion_out[key] = {}
        for method in ("dempster", "murphy"):
            fusion_out[key][method] = [
                {
                    "cell_id": row["cell_id"],
                    "true_class": row["true_class"],
                    "decision": row["decision"],
                    "K": row["K"],
                    "betp": row["betp"],
                    "bel_pl": row["bel_pl"],
                }
                for row in fused[key][method]
            ]
    (data / "sim_hypothetical_fusion_results.json").write_text(json.dumps(fusion_out, indent=2), encoding="utf-8")
    stats_out = {
        "label": "simulated hypothetical",
        "icc": effect["icc"],
        "design_effect": effect["design_effect"],
        "n_eff": effect["n_eff"],
        "bootstrap_resamples": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_results": ci,
        "mcnemar": [
            {name: (None if isinstance(value, float) and value != value else value) for name, value in row.items() if name != "order"}
            for row in tests
        ],
        "detection": rates,
        "replicate_index": rep_index,
        "fifty_replicate_pack": {
            key: {"mean": value[0], "sd": value[1]} for key, value in pack.items()
        },
    }
    (data / "sim_hypothetical_statistics.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    summary = out / "sim_hypothetical_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Simulated hypothetical vision, s = 1, q = 0.7",
                "",
                "This folder is a seeded simulation. It is not a measured vision result.",
                "The vision column is the s = 1, q = 0.7 draw from `scripts/sim_ideal_vision/simulate_ideal_vision.py`.",
                f"Seed {sim.SEED}, {sim.N_REPS} replicates, population SD.",
                f"The per-cell figures and the seven-condition tables use representative replicate {rep_index},",
                "chosen as the replicate whose V+U Dempster macro-F1 is closest to the median.",
                f"That replicate has vision fold-mean F1 {rep['vision_f1_folds']:.4f}, alpha {rep['alpha']:.4f}, ECE {rep['ece']:.4f}.",
                f"Across 50 replicates the vision fold-mean F1 is {vision_mean:.4f} +/- {vision_sd:.4f}.",
                "",
                "Ultrasonic masses are the measured masses recovered from `results/thesis/data/fusion_results.json`.",
                "Geometry masses are the measured file `results/geometry/bpa_geometry.csv` (alpha_G = 1).",
                "The measured UT decision macro-F1 on that fusion file is 0.482.",
                "A later UT label swap is not applied here.",
                "",
                "Trained-model checks (split leakage, a held-out public source, a nearest-neighbour audit)",
                "are not properties of this draw and are marked not applicable.",
                "",
                "Figures have no caption and no simulated stamp. The folder and the file names carry the label.",
                "Panel letters inside the chapter-style figures follow the existing chapter scripts.",
                "",
                f"V+G+U error correlation rho = {effect['icc']:.4f}, design effect = {effect['design_effect']:.3f}, n_eff = {effect['n_eff']:.2f}.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
