#!/usr/bin/env python3
"""Simulated ideal-vision benchmark for the weld-fusion thesis.

Only vision is simulated. Ultrasonic masses, labels, folds, discounting,
Dempster's rule, Murphy's rule, the pignistic decision and the alphabetical
tie-break are the measured pipeline. The script rebuilds that pipeline from
the stored belief and plausibility and stops if it does not match
``fusion_results.json``.

Every file it writes belongs under ``results/thesis/sim_vision/`` and starts
with ``sim_``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import chi2

SEED = 20261101
N_REPS = 50
Q_GRID = [round(i / 10, 1) for i in range(11)]
S_GRID = (1.0, 0.6)
TIE_TOL = 1e-9
K_TOL = 1e-6
MASS_TOL = 1e-6
BETA_CONCENTRATION = 20.0
CONF_CLIP = (0.34, 0.98)
N_BINS = 15
IDEAL_F1 = 0.65
IDEAL_P_RECALL = 0.70
BEAT_F1 = 0.02
BEAT_RECALL = 0.10
F1_TRIV_SPEC = 0.2568

# Reporting order follows the thesis recall table. Tie-break order is alphabetical.
CLASSES = ("N", "U", "P", "L")
TIE_ORDER = ("L", "N", "P", "U")
OUTCOMES = ("{P}", "{N,U}", "{N,U,L}")
OUTCOME_INDEX = {name: i for i, name in enumerate(OUTCOMES)}
OUTCOME_FOCAL = {
    "{P}": frozenset({"P"}),
    "{N,U}": frozenset({"N", "U"}),
    "{N,U,L}": frozenset({"N", "U", "L"}),
}
THETA = frozenset({"N", "U", "P", "L"})
FLAG_WORDS = {"overfill", "misaligned", "misalignment"}
FOLD_COUPONS = {
    1: ("C01", "C03", "C05", "C07"),
    2: ("C02", "C04", "C06", "C08"),
    3: ("C09", "C10", "C12"),
    4: ("C11", "C13", "C14", "C15"),
}
COUPON_FOLD = {coupon: fold for fold, coupons in FOLD_COUPONS.items() for coupon in coupons}
EXPECTED_SUPPORT = {"{P}": 26, "{N,U}": 43, "{N,U,L}": 81}

MM = 1.0 / 25.4
COLOUR_V = "#4A90D9"
COLOUR_U = "#C9A227"
COLOUR_FUSED = "#4A4A4A"


class GateFailure(Exception):
    """A validation check failed. The message is the mismatch to print."""


def find_repo_root() -> Path:
    seeds = [Path.cwd(), Path(__file__).resolve().parent]
    seen: list[Path] = []
    for seed in seeds:
        for candidate in [seed, *seed.parents]:
            if candidate in seen:
                continue
            seen.append(candidate)
            marker = candidate / "results" / "thesis" / "data" / "fusion_results.json"
            if marker.is_file():
                return candidate
    return Path.cwd()


def guard_output(out: Path, repo: Path) -> Path:
    """Allow the thesis simulation folder, or a directory outside results/."""
    resolved = out.resolve()
    results = (repo / "results").resolve()
    allowed = (repo / "results" / "thesis" / "sim_vision").resolve()
    try:
        resolved.relative_to(results)
        inside_results = True
    except ValueError:
        inside_results = False
    if inside_results and resolved != allowed:
        try:
            resolved.relative_to(allowed)
        except ValueError:
            raise SystemExit(
                f"Refusing to write into {resolved}. "
                f"Simulation outputs belong in {allowed}."
            )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_args() -> argparse.Namespace:
    repo = find_repo_root()
    data = repo / "results" / "thesis" / "data"
    parser = argparse.ArgumentParser(
        description="Simulate an ideal vision module and fuse it with measured ultrasonics."
    )
    parser.add_argument("--fusion", type=Path, default=data / "fusion_results.json")
    parser.add_argument("--labels", type=Path, default=data / "labels_v2_2026-10-26.csv")
    parser.add_argument("--calibration", type=Path, default=data / "calibration_v2.json")
    parser.add_argument("--out", type=Path, default=repo / "results" / "thesis" / "sim_vision")
    parser.add_argument(
        "--human-baseline",
        type=float,
        default=None,
        help="Optional horizontal reference on the curve figure. Omitted if unset.",
    )
    args = parser.parse_args()
    args.repo = repo
    return args


def coupon_of(cell_id: str) -> str:
    return cell_id.split("-", 1)[0]


def vision_target(true_class: str, condition: str, geometric: str) -> str:
    label = true_class.strip()
    flagged = condition.strip().lower() in FLAG_WORDS or geometric.strip().lower() in FLAG_WORDS
    if label == "P":
        return "{P}"
    if label == "U":
        return "{N,U}"
    if label == "L":
        return "{N,U,L}"
    if label == "N":
        return "{N,U}" if flagged else "{N,U,L}"
    raise GateFailure(f"Label {label!r} is outside {{N, U, P, L}}.")


def clean_bpa(raw: dict[frozenset[str], float]) -> dict[frozenset[str], float]:
    cleaned: dict[frozenset[str], float] = {}
    for focal, mass in raw.items():
        if mass < -1e-8:
            raise GateFailure(f"Negative mass {mass:.3e} on {set(focal)}.")
        if mass > 0.0:
            cleaned[focal] = float(mass)
    total = sum(cleaned.values())
    if abs(total - 1.0) > MASS_TOL:
        raise GateFailure(f"Recovered masses sum to {total:.8f}.")
    return cleaned


def recover_vision(bel_pl: dict) -> dict[frozenset[str], float]:
    """m(Theta) = Pl(P) - Bel(P), then the two compound vision focal elements."""
    bel_p = float(bel_pl["P"][0])
    pl_p = float(bel_pl["P"][1])
    pl_l = float(bel_pl["L"][1])
    pl_n = float(bel_pl["N"][1])
    m_theta = pl_p - bel_p
    m_nul = pl_l - m_theta
    m_nu = pl_n - m_nul - m_theta
    return clean_bpa(
        {
            OUTCOME_FOCAL["{P}"]: bel_p,
            OUTCOME_FOCAL["{N,U}"]: m_nu,
            OUTCOME_FOCAL["{N,U,L}"]: m_nul,
            THETA: m_theta,
        }
    )


def recover_ut(bel_pl: dict) -> dict[frozenset[str], float]:
    bel_p = float(bel_pl["P"][0])
    pl_p = float(bel_pl["P"][1])
    bel_l = float(bel_pl["L"][0])
    pl_n = float(bel_pl["N"][1])
    m_theta = pl_p - bel_p
    return clean_bpa(
        {
            frozenset({"P"}): bel_p,
            frozenset({"L"}): bel_l,
            OUTCOME_FOCAL["{N,U}"]: pl_n - m_theta,
            THETA: m_theta,
        }
    )


def dempster(
    bpa1: dict[frozenset[str], float], bpa2: dict[frozenset[str], float]
) -> tuple[dict[frozenset[str], float], float]:
    combined: dict[frozenset[str], float] = {}
    conflict = 0.0
    for left, m1 in bpa1.items():
        for right, m2 in bpa2.items():
            intersection = left & right
            mass = m1 * m2
            if not intersection:
                conflict += mass
            else:
                combined[intersection] = combined.get(intersection, 0.0) + mass
    if conflict >= 1.0 - 1e-12:
        raise RuntimeError(f"Total conflict K={conflict:.6f}.")
    scale = 1.0 - conflict
    return {focal: mass / scale for focal, mass in combined.items()}, conflict


def murphy(bpas: list[dict[frozenset[str], float]]) -> tuple[dict[frozenset[str], float], float]:
    """Average the masses, then combine that average with itself n - 1 times."""
    usable = [bpa for bpa in bpas if bpa]
    n = len(usable)
    if n == 0:
        return {THETA: 1.0}, 0.0
    if n == 1:
        return dict(usable[0]), 0.0
    keys: set[frozenset[str]] = set()
    for bpa in usable:
        keys.update(bpa)
    average = {key: sum(bpa.get(key, 0.0) for bpa in usable) / n for key in keys}
    average = {key: mass for key, mass in average.items() if mass > 0.0}
    result = dict(average)
    total_k = 0.0
    for _ in range(n - 1):
        result, conflict = dempster(result, average)
        total_k = total_k + conflict - total_k * conflict
    return result, total_k


def pignistic(bpa: dict[frozenset[str], float]) -> dict[str, float]:
    bet = {label: 0.0 for label in CLASSES}
    for focal, mass in bpa.items():
        if not focal:
            continue
        share = mass / len(focal)
        for label in focal:
            bet[label] += share
    return bet


def decide(bet: dict[str, float]) -> str:
    """Argmax of BetP. Ties within 1e-9 take the first alphabetical class."""
    top = max(bet[label] for label in TIE_ORDER)
    for label in TIE_ORDER:
        if abs(bet[label] - top) <= TIE_TOL:
            return label
    raise RuntimeError("BetP has no class.")


def winners(bet: dict[str, float]) -> set[str]:
    top = max(bet[label] for label in TIE_ORDER)
    return {label for label in TIE_ORDER if abs(bet[label] - top) <= TIE_TOL}


def belief_plausibility(bpa: dict[frozenset[str], float], hypothesis: frozenset[str]) -> tuple[float, float]:
    belief = 0.0
    plausibility = 0.0
    for focal, mass in bpa.items():
        if focal <= hypothesis:
            belief += mass
        if focal & hypothesis:
            plausibility += mass
    return belief, plausibility


def f1_one(y_true: list[str], y_pred: list[str], label: str) -> float | None:
    tp = fp = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if pred == label and truth == label:
            tp += 1
        elif pred == label:
            fp += 1
        elif truth == label:
            fn += 1
    if tp == 0 and fp == 0 and fn == 0:
        return None
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> float:
    """Unweighted mean of per-class F1. A class absent from both lists is skipped."""
    scores = []
    for label in labels:
        score = f1_one(y_true, y_pred, label)
        if score is not None:
            scores.append(score)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def class_recalls(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    recalls = {}
    for label in CLASSES:
        support = sum(truth == label for truth in y_true)
        hits = sum(truth == label and pred == label for truth, pred in zip(y_true, y_pred))
        recalls[label] = (hits / support) if support else float("nan")
    return recalls


def confusion_counts(y_true: list[str], y_pred: list[str]) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=float)
    for truth, pred in zip(y_true, y_pred):
        matrix[OUTCOME_INDEX[truth], OUTCOME_INDEX[pred]] += 1.0
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    rows = {}
    for index, name in enumerate(OUTCOMES):
        tp = float(matrix[index, index])
        support = float(matrix[index, :].sum())
        predicted = float(matrix[:, index].sum())
        fp = predicted - tp
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        score = (
            2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )
        rows[name] = {
            "precision": precision,
            "recall": recall,
            "f1": score,
            "support": support,
        }
    return rows


def expected_calibration_error(confidence: list[float], correct: list[bool], n_bins: int = N_BINS) -> float:
    """Equal-width ECE: bin weight times |accuracy - mean confidence|."""
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    error = 0.0
    for index in range(n_bins):
        lo, hi = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))
    return float(error)


def alpha_from_f1(f1: float, f1_triv: float) -> float:
    raw = 1.0 - (f1 - f1_triv) / (1.0 - f1_triv)
    return float(min(1.0, max(0.0, raw)))


def discount(prob: dict[str, float], alpha: float) -> dict[frozenset[str], float]:
    bpa: dict[frozenset[str], float] = {}
    if alpha > 0.0:
        bpa[THETA] = float(alpha)
    for name in OUTCOMES:
        mass = (1.0 - alpha) * prob[name]
        if mass > 0.0:
            bpa[OUTCOME_FOCAL[name]] = float(mass)
    total = sum(bpa.values())
    if abs(total - 1.0) > 1e-8:
        raise RuntimeError(f"Discounted masses sum to {total:.8f}.")
    return bpa


def discordant(y_true: list[str], base: list[str], augmented: list[str]) -> tuple[int, int]:
    """b: correct only under the augmented sensor. c: correct only under the base."""
    b = c = 0
    for truth, base_dec, aug_dec in zip(y_true, base, augmented):
        base_ok = base_dec == truth
        aug_ok = aug_dec == truth
        if aug_ok and not base_ok:
            b += 1
        elif base_ok and not aug_ok:
            c += 1
    return b, c


def mcnemar_stat(b: float, c: float) -> tuple[float, float]:
    """Continuity-corrected chi-square. Undefined when there are no discordant pairs."""
    total = b + c
    if total == 0:
        return float("nan"), float("nan")
    stat = (max(abs(b - c) - 1.0, 0.0) ** 2) / total
    return float(stat), float(chi2.sf(stat, 1))


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def max_interval_error(bpa: dict[frozenset[str], float], stored: dict) -> float:
    error = 0.0
    for label in CLASSES:
        belief, plausibility = belief_plausibility(bpa, frozenset({label}))
        error = max(
            error,
            abs(belief - float(stored[label][0])),
            abs(plausibility - float(stored[label][1])),
        )
    return error


def max_betp_error(bet: dict[str, float], stored: dict) -> float:
    return max(abs(bet[label] - float(stored[label])) for label in CLASSES)


def mix_confusion(c_real: np.ndarray, q: float) -> np.ndarray:
    mixed = (1.0 - q) * c_real + q * np.eye(3)
    mixed = np.clip(mixed, 0.0, None)
    return mixed / mixed.sum(axis=1, keepdims=True)


def hidden_count(n_porosity: int, s: float) -> int:
    """Nearest integer to (1 - s) times the porosity count. Half-integers round half up."""
    return int(np.floor((1.0 - s) * n_porosity + 0.5))


def condition_metrics(
    y_true: list[str],
    dec_d: list[str],
    betp_d: list[dict[str, float]],
    k_d: list[float],
    dec_m: list[str],
) -> dict:
    return {
        "f1_d": macro_f1(y_true, dec_d, CLASSES),
        "f1_m": macro_f1(y_true, dec_m, CLASSES),
        "mean_k": float(np.mean(k_d)),
        "l_dec": int(sum(decision == "L" for decision in dec_d)),
        "nu_ties": int(sum(winners(bet) == {"N", "U"} for bet in betp_d)),
        "recall": class_recalls(y_true, dec_d),
    }


def index_rule(fusion: dict, key: str, rule: str) -> dict[str, dict]:
    return {record["cell_id"]: record for record in fusion[key][rule]}


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, dict]:
    for path in (args.fusion, args.labels, args.calibration):
        if not path.is_file():
            raise SystemExit(f"Input not found: {path}")
    labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    fusion = json_load(args.fusion)
    calibration = json_load(args.calibration)
    return labels, fusion, calibration


def json_load(path: Path) -> dict:
    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_cells(labels: pd.DataFrame) -> list[dict]:
    required = {"cell_id", "true_class", "condition"}
    missing = required - set(labels.columns)
    if missing:
        raise GateFailure(f"Labels file is missing columns: {sorted(missing)}.")
    cells = []
    for row in labels.to_dict(orient="records"):
        cell_id = row["cell_id"].strip()
        coupon = coupon_of(cell_id)
        if coupon not in COUPON_FOLD:
            raise GateFailure(f"{cell_id} has coupon {coupon}, which is outside the four folds.")
        geometric = row.get("geometric_nonconformance", "")
        cells.append(
            {
                "cell_id": cell_id,
                "true_class": row["true_class"].strip(),
                "condition": row["condition"].strip(),
                "geometric": geometric.strip(),
                "fold": COUPON_FOLD[coupon],
                "file_fold": row.get("fold", "").strip(),
                "file_target": row.get("vision_target", "").strip(),
                "target": vision_target(row["true_class"], row["condition"], geometric),
            }
        )
    cells.sort(key=lambda cell: cell["cell_id"])
    return cells


def support_counts(cells: list[dict]) -> dict[str, int]:
    counts = {name: 0 for name in OUTCOMES}
    for cell in cells:
        counts[cell["target"]] += 1
    return counts


def print_gate(rows: list[tuple[str, bool, str]]) -> None:
    print("VALIDATION GATE")
    print(f"{'Check':<64} {'Status':<6} Detail")
    for name, ok, detail in rows:
        status = "PASS" if ok else "FAIL"
        print(f"{name:<64} {status:<6} {detail}")
    print("RESULT:", "PASS" if all(ok for _, ok, _ in rows) else "FAIL")


def add(rows: list[tuple[str, bool, str]], name: str, ok: bool, detail: str) -> None:
    rows.append((name, bool(ok), detail))


def run_gate(labels: pd.DataFrame, fusion: dict, calibration: dict) -> dict:
    """Rebuild the measured fusion. Return context for the simulation, or a failed gate."""
    rows: list[tuple[str, bool, str]] = []
    context: dict = {"rows": rows, "ok": False}

    try:
        cells = build_cells(labels)
    except GateFailure as exc:
        add(rows, "labels parse", False, str(exc))
        return context

    add(rows, "cell count", len(cells) == 150, f"{len(cells)} cells")
    folds_ok = True
    fold_bad = []
    for cell in cells:
        if cell["file_fold"] and cell["file_fold"] != str(cell["fold"]):
            folds_ok = False
            fold_bad.append(cell["cell_id"])
    add(
        rows,
        "folds match the declared coupons",
        folds_ok and {cell["fold"] for cell in cells} == {1, 2, 3, 4},
        "ok" if folds_ok else f"mismatch on {fold_bad[:8]}",
    )
    supports = support_counts(cells)
    support_ok = supports == EXPECTED_SUPPORT
    add(
        rows,
        "vision target supports",
        support_ok,
        ", ".join(f"{name}={supports[name]}" for name in OUTCOMES),
    )
    if "vision_target" in labels.columns:
        bad_targets = [cell["cell_id"] for cell in cells if cell["file_target"] != cell["target"]]
        add(
            rows,
            "target rule matches vision_target column",
            not bad_targets,
            "ok" if not bad_targets else f"{len(bad_targets)} rows, first {bad_targets[:8]}",
        )

    for key in ("V", "U", "UV"):
        if key not in fusion or "dempster" not in fusion[key] or "murphy" not in fusion[key]:
            add(rows, f"fusion key {key}", False, "missing dempster or murphy records")
            return context
    ids = [cell["cell_id"] for cell in cells]
    id_set = set(ids)
    id_ok = True
    id_detail = "150 shared cell ids"
    for key in ("V", "U", "UV"):
        for rule in ("dempster", "murphy"):
            found = {record["cell_id"] for record in fusion[key][rule]}
            if found != id_set:
                id_ok = False
                id_detail = f"{key}/{rule} has {len(found & id_set)} of 150 ids"
    add(rows, "fusion cell ids match labels", id_ok, id_detail)
    if not id_ok or len(cells) != 150 or not support_ok:
        return context

    indexed = {
        key: {rule: index_rule(fusion, key, rule) for rule in ("dempster", "murphy")}
        for key in ("V", "U", "UV")
    }
    label_ok = True
    label_bad = []
    for cell in cells:
        for key in ("V", "U", "UV"):
            stored = indexed[key]["dempster"][cell["cell_id"]]["true_class"]
            if stored != cell["true_class"]:
                label_ok = False
                label_bad.append(cell["cell_id"])
    add(
        rows,
        "fusion true_class matches labels",
        label_ok,
        "ok" if label_ok else f"mismatch on {label_bad[:8]}",
    )

    try:
        cm = np.asarray(calibration["confusion_matrix"], dtype=float)
        if cm.shape != (3, 3):
            raise GateFailure(f"Confusion matrix shape is {cm.shape}.")
        row_sums = [int(round(value)) for value in cm.sum(axis=1)]
        expected_rows = [EXPECTED_SUPPORT[name] for name in OUTCOMES]
        if row_sums != expected_rows:
            raise GateFailure(f"Confusion row sums are {row_sums}, expected {expected_rows}.")
        fold_f1 = np.asarray(calibration["per_fold_f1"], dtype=float)
        if fold_f1.shape != (4,):
            raise GateFailure("per_fold_f1 does not contain four folds.")
        mean_f1 = float(fold_f1.mean())
        std_f1 = float(fold_f1.std(ddof=0))
        if abs(mean_f1 - float(calibration["mean_f1"])) > 1e-12:
            raise GateFailure("mean_f1 does not equal the mean of per_fold_f1.")
        if abs(std_f1 - float(calibration["std_f1"])) > 1e-12:
            raise GateFailure("std_f1 is not the population SD of per_fold_f1.")
        triv = float(calibration["trivial_baseline"])
        if abs(triv - F1_TRIV_SPEC) > 1e-6:
            raise GateFailure(f"trivial_baseline is {triv}, not {F1_TRIV_SPEC}.")
        alpha_file = float(calibration["alpha_V"])
        alpha_formula = alpha_from_f1(mean_f1, triv)
        if abs(alpha_file - alpha_formula) > 1e-6:
            raise GateFailure(
                f"alpha_V is {alpha_file:.6f}; the discount formula gives {alpha_formula:.6f}."
            )
        if int(calibration["ece_bins"]) != N_BINS:
            raise GateFailure(f"ece_bins is {calibration['ece_bins']}, not {N_BINS}.")
        outcome_metrics = metrics_from_confusion(cm)
        pooled = float(np.mean([outcome_metrics[name]["f1"] for name in OUTCOMES]))
        if abs(pooled - float(calibration["pooled_f1"])) > 1e-6:
            raise GateFailure(
                f"Pooled macro-F1 from the confusion matrix is {pooled:.6f}, "
                f"file has {float(calibration['pooled_f1']):.6f}."
            )
        for name in OUTCOMES:
            stored_f1 = float(calibration["per_class_f1"][name])
            if abs(outcome_metrics[name]["f1"] - stored_f1) > 1e-6:
                raise GateFailure(f"{name} F1 does not match per_class_f1.")
    except (KeyError, GateFailure, TypeError, ValueError) as exc:
        add(rows, "calibration file contract", False, str(exc))
        return context
    add(rows, "calibration file contract", True, "folds, baseline, alpha, confusion, ECE bins")

    c_real = cm / cm.sum(axis=1, keepdims=True)
    try:
        vision_bpa = [recover_vision(indexed["V"]["dempster"][cell["cell_id"]]["bel_pl"]) for cell in cells]
        ut_bpa = [recover_ut(indexed["U"]["dempster"][cell["cell_id"]]["bel_pl"]) for cell in cells]
    except GateFailure as exc:
        add(rows, "mass recovery", False, str(exc))
        return context

    vision_theta = [bpa[THETA] for bpa in vision_bpa]
    ut_theta = [bpa[THETA] for bpa in ut_bpa]
    vision_theta_ok = (max(vision_theta) - min(vision_theta) < 1e-9) and (
        abs(float(np.mean(vision_theta)) - alpha_file) < 1e-6
    )
    add(
        rows,
        "vision m(Theta) equals alpha_V",
        vision_theta_ok,
        f"mean {float(np.mean(vision_theta)):.6f}",
    )
    ut_mean = float(np.mean(ut_theta))
    ut_ok = (max(ut_theta) - min(ut_theta) < 1e-9) and abs(ut_mean - 0.2406) < 5e-5
    add(
        rows,
        "UT m(Theta) constant and rounds to 0.2406",
        ut_ok,
        f"value {ut_mean:.6f}",
    )

    bel_err = 0.0
    for bpa, cell in zip(vision_bpa, cells):
        bel_err = max(bel_err, max_interval_error(bpa, indexed["V"]["dempster"][cell["cell_id"]]["bel_pl"]))
    for bpa, cell in zip(ut_bpa, cells):
        bel_err = max(bel_err, max_interval_error(bpa, indexed["U"]["dempster"][cell["cell_id"]]["bel_pl"]))
    add(rows, "recovered masses reproduce Bel and Pl", bel_err <= 1e-6, f"max abs error {bel_err:.3e}")

    true = [cell["true_class"] for cell in cells]
    rebuilt: dict[str, dict[str, dict]] = {}
    compare_err = {"decision": [], "K": 0.0, "betp": 0.0, "belpl": 0.0}
    for key, sources in (
        ("V", "vision"),
        ("U", "ut"),
        ("UV", "both"),
    ):
        rebuilt[key] = {}
        for rule in ("dempster", "murphy"):
            decisions = []
            k_values = []
            betps = []
            for index, cell in enumerate(cells):
                if sources == "vision":
                    bpa, conflict = murphy([vision_bpa[index]]) if rule == "murphy" else (vision_bpa[index], 0.0)
                elif sources == "ut":
                    bpa, conflict = murphy([ut_bpa[index]]) if rule == "murphy" else (ut_bpa[index], 0.0)
                elif rule == "dempster":
                    bpa, conflict = dempster(vision_bpa[index], ut_bpa[index])
                else:
                    bpa, conflict = murphy([vision_bpa[index], ut_bpa[index]])
                bet = pignistic(bpa)
                decision = decide(bet)
                decisions.append(decision)
                k_values.append(conflict)
                betps.append(bet)
                stored = indexed[key][rule][cell["cell_id"]]
                if decision != stored["decision"]:
                    compare_err["decision"].append(f"{key}/{rule}/{cell['cell_id']}")
                compare_err["K"] = max(compare_err["K"], abs(conflict - float(stored["K"])))
                compare_err["betp"] = max(compare_err["betp"], max_betp_error(bet, stored["betp"]))
                compare_err["belpl"] = max(
                    compare_err["belpl"], max_interval_error(bpa, stored["bel_pl"])
                )
            rebuilt[key][rule] = {"decision": decisions, "K": k_values, "betp": betps}

    n_dec = len(compare_err["decision"])
    add(
        rows,
        "decisions match V, U and V+U under both rules",
        n_dec == 0,
        "150 x 3 x 2" if n_dec == 0 else f"{n_dec} mismatches, first {compare_err['decision'][:8]}",
    )
    add(rows, "K matches within 1e-6", compare_err["K"] <= K_TOL, f"max abs error {compare_err['K']:.3e}")
    add(rows, "BetP matches within 1e-6", compare_err["betp"] <= 1e-6, f"max abs error {compare_err['betp']:.3e}")
    add(
        rows,
        "fused Bel and Pl match within 1e-6",
        compare_err["belpl"] <= 1e-6,
        f"max abs error {compare_err['belpl']:.3e}",
    )

    f1_ok = True
    f1_detail = []
    for key in ("V", "U", "UV"):
        for rule in ("dempster", "murphy"):
            got = macro_f1(true, rebuilt[key][rule]["decision"], CLASSES)
            stored_dec = [indexed[key][rule][cell["cell_id"]]["decision"] for cell in cells]
            stored_f1 = macro_f1(true, stored_dec, CLASSES)
            f1_detail.append(f"{key}/{rule} {got:.6f}")
            if abs(got - stored_f1) > 1e-12:
                f1_ok = False
    add(rows, "macro-F1 over {N, U, P, L} matches", f1_ok, "; ".join(f1_detail))

    def stored_decisions(key: str, rule: str) -> list[str]:
        return [indexed[key][rule][cell["cell_id"]]["decision"] for cell in cells]

    def stored_betps(key: str, rule: str) -> list[dict]:
        return [indexed[key][rule][cell["cell_id"]]["betp"] for cell in cells]

    for rule in ("dempster", "murphy"):
        n_p = sum(decision == "P" for decision in stored_decisions("V", rule))
        add(rows, f"V {rule} decides P on 141 cells", n_p == 141, f"{n_p}")
    for key in ("U", "UV"):
        for rule in ("dempster", "murphy"):
            n_l = sum(decision == "L" for decision in stored_decisions(key, rule))
            n_tie = sum(winners(bet) == {"N", "U"} for bet in stored_betps(key, rule))
            add(
                rows,
                f"{key} {rule} has 97 N/U ties and 29 L decisions",
                n_l == 29 and n_tie == 97,
                f"L={n_l}, N/U ties={n_tie}",
            )

    measured_conditions = {}
    for key in ("V", "U", "UV"):
        measured_conditions[key] = condition_metrics(
            true,
            stored_decisions(key, "dempster"),
            stored_betps(key, "dempster"),
            [float(indexed[key]["dempster"][cell["cell_id"]]["K"]) for cell in cells],
            stored_decisions(key, "murphy"),
        )
    b_count, c_count = discordant(
        true, stored_decisions("U", "dempster"), stored_decisions("UV", "dempster")
    )
    chi_stat, chi_p = mcnemar_stat(b_count, c_count)

    context.update(
        {
            "ok": all(ok for _, ok, _ in rows),
            "cells": cells,
            "vision_bpa": vision_bpa,
            "ut_bpa": ut_bpa,
            "c_real": c_real,
            "f1_triv": triv,
            "true": true,
            "u_decisions": stored_decisions("U", "dempster"),
            "measured_vision": {
                "f1_mean": mean_f1,
                "f1_std": std_f1,
                "pooled": pooled,
                "p_recall": outcome_metrics["{P}"]["recall"],
                "p_precision": outcome_metrics["{P}"]["precision"],
                "ece": float(calibration["ece_post"]),
                "alpha": alpha_file,
                "outcomes": outcome_metrics,
                "confusion": cm,
            },
            "measured_conditions": measured_conditions,
            "measured_mcnemar": {"b": b_count, "c": c_count, "chi2": chi_stat, "p": chi_p},
        }
    )
    return context


def run_replicate(
    cells: list[dict],
    ut_bpa: list[dict[frozenset[str], float]],
    u_decisions: list[str],
    true: list[str],
    c_q: np.ndarray,
    s: float,
    f1_triv: float,
    rng: np.random.Generator,
) -> dict:
    targets = [cell["target"] for cell in cells]
    porosity = np.array([i for i, target in enumerate(targets) if target == "{P}"], dtype=int)
    n_flip = hidden_count(len(porosity), s)
    if n_flip > 0:
        chosen = rng.choice(porosity, size=n_flip, replace=False)
        for index in np.atleast_1d(chosen):
            targets[int(index)] = "{N,U,L}"

    accuracy = float(np.mean([c_q[OUTCOME_INDEX[target], OUTCOME_INDEX[target]] for target in targets]))
    mu = float(np.clip(accuracy, 1e-6, 1.0 - 1e-6))
    alpha_par = mu * BETA_CONCENTRATION
    beta_par = (1.0 - mu) * BETA_CONCENTRATION

    preds: list[str] = []
    probs: list[dict[str, float]] = []
    for target in targets:
        row = c_q[OUTCOME_INDEX[target]]
        drawn = int(rng.choice(3, p=row))
        pred = OUTCOMES[drawn]
        confidence = float(rng.beta(alpha_par, beta_par))
        confidence = float(np.clip(confidence, CONF_CLIP[0], CONF_CLIP[1]))
        rest = (1.0 - confidence) / 2.0
        prob = {name: (confidence if name == pred else rest) for name in OUTCOMES}
        scale = sum(prob.values())
        probs.append({name: value / scale for name, value in prob.items()})
        preds.append(pred)

    confidences = [prob[pred] for prob, pred in zip(probs, preds)]
    correct = [pred == target for pred, target in zip(preds, targets)]
    fold_scores = []
    for fold in (1, 2, 3, 4):
        idx = [i for i, cell in enumerate(cells) if cell["fold"] == fold]
        fold_scores.append(
            macro_f1([targets[i] for i in idx], [preds[i] for i in idx], OUTCOMES)
        )
    f1_folds = float(np.mean(fold_scores))
    alpha = alpha_from_f1(f1_folds, f1_triv)

    v_dec_d: list[str] = []
    v_dec_m: list[str] = []
    v_betp: list[dict[str, float]] = []
    v_k: list[float] = []
    uv_dec_d: list[str] = []
    uv_dec_m: list[str] = []
    uv_betp: list[dict[str, float]] = []
    uv_k: list[float] = []
    for index, prob in enumerate(probs):
        bpa_v = discount(prob, alpha)
        bet_v = pignistic(bpa_v)
        v_dec_d.append(decide(bet_v))
        alone_m, _ = murphy([bpa_v])
        v_dec_m.append(decide(pignistic(alone_m)))
        v_betp.append(bet_v)
        v_k.append(0.0)
        fused_d, k_d = dempster(bpa_v, ut_bpa[index])
        fused_m, _ = murphy([bpa_v, ut_bpa[index]])
        bet_d = pignistic(fused_d)
        uv_dec_d.append(decide(bet_d))
        uv_dec_m.append(decide(pignistic(fused_m)))
        uv_betp.append(bet_d)
        uv_k.append(k_d)

    matrix = confusion_counts(targets, preds)
    outcomes = metrics_from_confusion(matrix)
    b_count, c_count = discordant(true, u_decisions, uv_dec_d)
    return {
        "vision_f1_folds": f1_folds,
        "vision_f1_pooled": macro_f1(targets, preds, OUTCOMES),
        "outcome_p_recall": outcomes["{P}"]["recall"],
        "outcome_p_precision": outcomes["{P}"]["precision"],
        "ece": expected_calibration_error(confidences, correct),
        "alpha": alpha,
        "outcomes": outcomes,
        "confusion": matrix,
        "V": condition_metrics(true, v_dec_d, v_betp, v_k, v_dec_m),
        "UV": condition_metrics(true, uv_dec_d, uv_betp, uv_k, uv_dec_m),
        "b": b_count,
        "c": c_count,
        "uv_f1_d": macro_f1(true, uv_dec_d, CLASSES),
    }


def simulate(context: dict) -> dict[float, dict[float, list[dict]]]:
    rng = np.random.default_rng(SEED)
    sweep: dict[float, dict[float, list[dict]]] = {s: {} for s in S_GRID}
    for s in S_GRID:
        for q in Q_GRID:
            c_q = mix_confusion(context["c_real"], q)
            replicates = [
                run_replicate(
                    context["cells"],
                    context["ut_bpa"],
                    context["u_decisions"],
                    context["true"],
                    c_q,
                    s,
                    context["f1_triv"],
                    rng,
                )
                for _ in range(N_REPS)
            ]
            sweep[s][q] = replicates
            f1_mean, f1_sd = mean_sd([rep["vision_f1_folds"] for rep in replicates])
            print(
                f"simulated s={s:.1f} q={q:.1f}  "
                f"vision macro-F1 {f1_mean:.4f} +/- {f1_sd:.4f}"
            )
    return sweep


def select_ideal(sweep_s: dict[float, list[dict]]) -> float | None:
    """Smallest q whose mean fold-macro-F1 and mean outcome-{P} recall clear the bars."""
    for q in Q_GRID:
        f1_mean, _ = mean_sd([rep["vision_f1_folds"] for rep in sweep_s[q]])
        recall_mean, _ = mean_sd([rep["outcome_p_recall"] for rep in sweep_s[q]])
        if f1_mean >= IDEAL_F1 and recall_mean >= IDEAL_P_RECALL:
            return q
    return None


def representative_index(replicates: list[dict]) -> int:
    scores = np.asarray([rep["uv_f1_d"] for rep in replicates], dtype=float)
    median = float(np.median(scores))
    return int(np.argmin(np.abs(scores - median)))


def crossing(sweep_s: dict[float, list[dict]], u_f1: float, u_recall: float) -> tuple[float, float] | None:
    """First q whose mean V+U margin clears both bars. Returns (q, vision F1)."""
    for q in Q_GRID:
        vision, _ = mean_sd([rep["vision_f1_folds"] for rep in sweep_s[q]])
        fused, _ = mean_sd([rep["UV"]["f1_d"] for rep in sweep_s[q]])
        recall, _ = mean_sd([rep["UV"]["recall"]["P"] for rep in sweep_s[q]])
        if (fused - u_f1) >= BEAT_F1 and (recall - u_recall) >= BEAT_RECALL:
            return q, vision
    return None


def pm_tex(mean: float, sd: float, digits: int = 3) -> str:
    return rf"${mean:.{digits}f} \pm {sd:.{digits}f}$"


def num_tex(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pm_md(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def fmt_p(p: float) -> str:
    if p != p:
        return "---"
    if p < 1e-4:
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / 10**exponent
        return rf"${mantissa:.2f}\times 10^{{{exponent}}}$"
    return f"{p:.4f}"


def fmt_chi(stat: float) -> str:
    if stat != stat:
        return "---"
    return f"{stat:.2f}"


def fmt_count_pair(b: float, c: float, as_int: bool) -> tuple[str, str]:
    if as_int:
        return str(int(round(b))), str(int(round(c)))
    return f"{b:.2f}", f"{c:.2f}"


def write_fragment(path: Path, caption: str, lines: list[str]) -> None:
    if not path.name.startswith("sim_"):
        raise RuntimeError(f"Output name must start with sim_: {path.name}")
    text = "% Simulated: " + caption.strip() + "\n" + "\n".join(lines) + "\n"
    for char in text:
        if ord(char) > 127:
            raise RuntimeError(f"Non-ASCII character {char!r} in {path.name}.")
    path.write_text(text, encoding="ascii", newline="\n")


def tex_table(spec: str, header: str, body: list[str]) -> list[str]:
    return [
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *[row + r" \\" for row in body],
        r"\bottomrule",
        r"\end{tabular}",
    ]


def level_pack(replicates: list[dict]) -> dict[str, tuple[float, float]]:
    def grab(fn) -> tuple[float, float]:
        return mean_sd([fn(rep) for rep in replicates])

    pack = {
        "f1": grab(lambda rep: rep["vision_f1_folds"]),
        "pooled": grab(lambda rep: rep["vision_f1_pooled"]),
        "p_recall": grab(lambda rep: rep["outcome_p_recall"]),
        "p_precision": grab(lambda rep: rep["outcome_p_precision"]),
        "ece": grab(lambda rep: rep["ece"]),
        "alpha": grab(lambda rep: rep["alpha"]),
    }
    for name in OUTCOMES:
        for metric in ("precision", "recall", "f1", "support"):
            pack[f"{name}:{metric}"] = grab(lambda rep, n=name, m=metric: rep["outcomes"][n][m])
    for key in ("V", "UV"):
        pack[f"{key}:f1_d"] = grab(lambda rep, k=key: rep[k]["f1_d"])
        pack[f"{key}:f1_m"] = grab(lambda rep, k=key: rep[k]["f1_m"])
        pack[f"{key}:mean_k"] = grab(lambda rep, k=key: rep[k]["mean_k"])
        pack[f"{key}:l_dec"] = grab(lambda rep, k=key: rep[k]["l_dec"])
        pack[f"{key}:nu_ties"] = grab(lambda rep, k=key: rep[k]["nu_ties"])
        for label in CLASSES:
            pack[f"{key}:recall:{label}"] = grab(lambda rep, k=key, lab=label: rep[k]["recall"][lab])
    return pack


def write_tables(out: Path, context: dict, ideal: dict | None, perfect: dict, ideal_q: float | None) -> None:
    vision = context["measured_vision"]
    measured_rows = [
        "Measured & "
        + " & ".join(
            [
                pm_tex(vision["f1_mean"], vision["f1_std"]),
                num_tex(vision["pooled"]),
                num_tex(vision["p_recall"]),
                num_tex(vision["p_precision"]),
                num_tex(vision["ece"]),
                num_tex(vision["alpha"]),
            ]
        )
    ]
    for label, pack in (("Ideal (sim)", ideal), ("Perfect (sim)", perfect)):
        if pack is None:
            measured_rows.append(label + " & " + " & ".join(["not reached"] * 6))
            continue
        measured_rows.append(
            label
            + " & "
            + " & ".join(
                [
                    pm_tex(*pack["f1"]),
                    pm_tex(*pack["pooled"]),
                    pm_tex(*pack["p_recall"]),
                    pm_tex(*pack["p_precision"]),
                    pm_tex(*pack["ece"]),
                    pm_tex(*pack["alpha"]),
                ]
            )
        )
    ideal_q_text = "not reached" if ideal_q is None else f"q = {ideal_q:.1f}, s = 0.6"
    write_fragment(
        out / "sim_t_modality.tex",
        "vision-module quality. Measured is the real sensor (macro-F1 is the mean and "
        "population SD over the four folds). Ideal "
        + f"({ideal_q_text}) and Perfect (q = 1, s = 1) are simulated: "
        "each other entry is the mean and population SD over 50 replicates.",
        tex_table(
            "lcccccc",
            r"Vision & Macro-$F_1$ & Pooled $F_1$ & $\{P\}$ recall & $\{P\}$ precision & ECE & $\alpha_V$",
            measured_rows,
        ),
    )

    element_rows = []
    for name in OUTCOMES:
        stats = vision["outcomes"][name]
        element_rows.append(
            " & ".join(
                [
                    "Measured",
                    set_tex(name),
                    num_tex(stats["precision"]),
                    num_tex(stats["recall"]),
                    num_tex(stats["f1"]),
                    str(int(round(stats["support"]))),
                ]
            )
        )
    if ideal is None:
        for name in OUTCOMES:
            element_rows.append(f"Ideal (sim) & {set_tex(name)} & not reached & not reached & not reached & not reached")
    else:
        for name in OUTCOMES:
            element_rows.append(
                " & ".join(
                    [
                        "Ideal (sim)",
                        set_tex(name),
                        pm_tex(*ideal[f"{name}:precision"]),
                        pm_tex(*ideal[f"{name}:recall"]),
                        pm_tex(*ideal[f"{name}:f1"]),
                        pm_tex(*ideal[f"{name}:support"], 1),
                    ]
                )
            )
    write_fragment(
        out / "sim_t_elements.tex",
        "per-outcome precision, recall, F1 and support for vision. "
        "Ideal is simulated"
        + (f" at q = {ideal_q:.1f}, s = 0.6." if ideal_q is not None else ", but the Ideal level was not reached."),
        tex_table(
            "llcccc",
            r"Version & Outcome & Precision & Recall & $F_1$ & Support",
            element_rows,
        ),
    )

    conditions = context["measured_conditions"]
    quantities = (
        ("f1_d", r"Dempster macro-$F_1$", 3, False),
        ("mean_k", r"Mean $K$", 3, False),
        ("l_dec", r"$\{L\}$ decisions", 1, True),
        ("nu_ties", r"N/U ties", 1, True),
        ("f1_m", r"Murphy macro-$F_1$", 3, False),
    )
    ablation_rows = []
    for key, title in (("V", r"$V$"), ("UV", r"$V{+}U$")):
        if key == "UV":
            ablation_rows.append(r"\midrule")
        for field, label, digits, is_count in quantities:
            measured = conditions[key][field]
            measured_tex = str(int(measured)) if is_count else num_tex(measured, digits)
            ideal_tex = "not reached" if ideal is None else pm_tex(*ideal[f"{key}:{field}"], digits)
            perfect_tex = pm_tex(*perfect[f"{key}:{field}"], digits)
            ablation_rows.append(
                f"{title} & {label} & {measured_tex} & {ideal_tex} & {perfect_tex}"
            )
    u_rows = [r"\midrule"]
    u_metrics = conditions["U"]
    for field, label, digits, is_count in quantities:
        value = u_metrics[field]
        shown = str(int(value)) if is_count else num_tex(value, digits)
        u_rows.append(rf"$U$ & {label} & \multicolumn{{3}}{{c}}{{{shown}}}")
    # Insert the U block between V and V+U. V is the first five data rows.
    body = ablation_rows[:5] + u_rows + ablation_rows[5:]
    # The midrule before UV was included in ablation_rows[5] when key switched.
    # body currently: 5 V rows, midrule + U rows, then midrule + UV rows.
    write_fragment(
        out / "sim_t_ablation.tex",
        "V, U and V+U. Ideal and Perfect vision are simulated; ultrasonic evidence is measured "
        "in every column. Mean K, L decisions and N/U ties are Dempster's rule. "
        "U is shown once because it does not depend on the vision simulation. "
        + (f"Ideal uses q = {ideal_q:.1f}, s = 0.6. " if ideal_q is not None else "Ideal was not reached. ")
        + "Perfect uses q = 1, s = 1.",
        [
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"Condition & Quantity & Measured & Ideal (sim) & Perfect (sim) \\",
            r"\midrule",
            *[row + r" \\" if not row.startswith(r"\midrule") else row for row in body],
            r"\bottomrule",
            r"\end{tabular}",
        ],
    )

    recall_rows = []
    for key, title in (("V", r"$V$"), ("U", r"$U$"), ("UV", r"$V{+}U$")):
        measured = conditions[key]["recall"]
        recall_rows.append(
            f"{title} & "
            + ("Reference" if key == "U" else "Measured")
            + " & "
            + " & ".join(num_tex(measured[label]) for label in CLASSES)
        )
        if key == "U":
            continue
        if ideal is None:
            recall_rows.append(
                f"{title} & Ideal (sim) & not reached & not reached & not reached & not reached"
            )
        else:
            recall_rows.append(
                f"{title} & Ideal (sim) & "
                + " & ".join(pm_tex(*ideal[f"{key}:recall:{label}"]) for label in CLASSES)
            )
    write_fragment(
        out / "sim_t_recall.tex",
        "per-class recall of the pignistic decision for measured vision and Ideal simulated vision. "
        "U is the measured reference and is identical for both columns.",
        tex_table(
            "llcccc",
            r"Condition & Version & N & U & P & L",
            recall_rows,
        ),
    )

def set_tex(name: str) -> str:
    """LaTeX for a focal element, also accepted by Matplotlib mathtext."""
    body = name[1:-1]
    return "$\\{" + body + "\\}$"


def mcnemar_row(version: str, aggregate: str, record: dict, as_int: bool) -> str:
    b_text, c_text = fmt_count_pair(record["b"], record["c"], as_int)
    return " & ".join(
        [version, aggregate, b_text, c_text, fmt_chi(record["chi2"]), fmt_p(record["p"])]
    )


def write_mcnemar(
    out: Path,
    context: dict,
    ideal_reps: list[dict] | None,
    perfect_reps: list[dict],
) -> None:
    rows = [mcnemar_row("Measured", "single", context["measured_mcnemar"], as_int=True)]
    for label, replicates in (("Ideal (sim)", ideal_reps), ("Perfect (sim)", perfect_reps)):
        if not replicates:
            rows.append(f"{label} & median replicate & not reached & not reached & --- & ---")
            rows.append(f"{label} & mean over replicates & not reached & not reached & --- & ---")
            continue
        index = representative_index(replicates)
        chosen = replicates[index]
        stat, p_value = mcnemar_stat(chosen["b"], chosen["c"])
        rows.append(
            mcnemar_row(
                label,
                "median replicate",
                {"b": chosen["b"], "c": chosen["c"], "chi2": stat, "p": p_value},
                as_int=True,
            )
        )
        b_mean = float(np.mean([rep["b"] for rep in replicates]))
        c_mean = float(np.mean([rep["c"] for rep in replicates]))
        stat_m, p_m = mcnemar_stat(b_mean, c_mean)
        rows.append(
            mcnemar_row(
                label,
                "mean over replicates",
                {"b": b_mean, "c": c_mean, "chi2": stat_m, "p": p_m},
                as_int=False,
            )
        )
    write_fragment(
        out / "sim_t_mcnemar.tex",
        "exploratory, not a declared comparison. Continuity-corrected McNemar test of "
        r"U $\to$ V+U. b is correct only under V+U. c is correct only under U. "
        "The median row is the replicate whose V+U Dempster macro-F1 is closest to the "
        "median. The mean row evaluates the same statistic on the mean counts.",
        [
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Version & Aggregate & $b$ & $c$ & $\chi^2$ & $p$ \\",
            r"\midrule",
            *[row + r" \\" for row in rows],
            r"\bottomrule",
            r"\end{tabular}",
        ],
    )


def write_tables_clean(
    out: Path,
    context: dict,
    ideal: dict | None,
    perfect: dict,
    ideal_q: float | None,
    ideal_reps: list[dict] | None,
    perfect_reps: list[dict],
) -> None:
    """Write the five tabular fragments. ``write_tables`` leaves McNemar to ``write_mcnemar``."""
    write_tables(out, context, ideal, perfect, ideal_q)
    write_mcnemar(out, context, ideal_reps, perfect_reps)


def setup_style() -> str:
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    chosen = "DejaVu Serif"
    for candidate in ("Times New Roman", "TeX Gyre Termes", "Liberation Serif"):
        if candidate in installed:
            chosen = candidate
            break
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [chosen, "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.4,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    return chosen


def stamp(fig: plt.Figure) -> None:
    fig.text(0.995, 0.012, "SIMULATED", ha="right", va="bottom", fontsize=7, color="0.45")


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(0.0, 1.05, text, transform=ax.transAxes, fontsize=9, ha="left", va="bottom")


def save_figure(fig: plt.Figure, path: Path) -> None:
    if not path.name.startswith("sim_"):
        raise RuntimeError(path.name)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def mean_confusion(replicates: list[dict]) -> np.ndarray:
    stack = np.stack([rep["confusion"] for rep in replicates], axis=0)
    return stack.mean(axis=0)


def draw_confusion(ax: plt.Axes, counts: np.ndarray, simulated: bool) -> plt.AxesImage:
    shares = counts / counts.sum(axis=1, keepdims=True)
    image = ax.imshow(shares, cmap="Greys", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks([0, 1, 2], [set_tex(name) for name in OUTCOMES])
    ax.set_yticks([0, 1, 2], [set_tex(name) for name in OUTCOMES])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True outcome")
    for row in range(3):
        for col in range(3):
            if simulated:
                label = f"{shares[row, col]:.2f}\n({counts[row, col]:.1f})"
            else:
                label = f"{shares[row, col]:.2f}\n({int(round(counts[row, col]))})"
            colour = "white" if shares[row, col] > 0.62 else "black"
            ax.text(col, row, label, ha="center", va="center", fontsize=7, color=colour)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return image


def plot_confusion(out: Path, measured_cm: np.ndarray, ideal_reps: list[dict] | None) -> None:
    fig = plt.figure(figsize=(160 * MM, 82 * MM))
    # Positions are fixed before the colour bar is added. Adjusting them afterwards
    # slides the axes back underneath the bar.
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.05],
        left=0.08,
        right=0.98,
        bottom=0.18,
        top=0.86,
        wspace=0.38,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cax = fig.add_subplot(grid[0, 2])
    image = draw_confusion(axes[0], measured_cm, simulated=False)
    panel_label(axes[0], "(a) measured")
    if ideal_reps is None:
        axes[1].axis("off")
        panel_label(axes[1], "(b) ideal not reached")
        cax.axis("off")
    else:
        draw_confusion(axes[1], mean_confusion(ideal_reps), simulated=True)
        axes[1].set_ylabel("")
        panel_label(axes[1], "(b) ideal, simulated")
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label("Row share")
        colorbar.outline.set_linewidth(0.5)
    stamp(fig)
    save_figure(fig, out / "sim_f_confusion.pdf")


def plot_fusion(
    out: Path,
    context: dict,
    ideal: dict | None,
    ideal_reps: list[dict] | None,
) -> None:
    fig = plt.figure(figsize=(160 * MM, 82 * MM))
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 1.22],
        left=0.07,
        right=0.98,
        bottom=0.24,
        top=0.86,
        wspace=0.40,
    )
    axis = fig.add_subplot(outer[0, 0])
    inner = outer[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.10)
    heat = fig.add_subplot(inner[0, 0])
    cax = fig.add_subplot(inner[0, 1])
    measured = context["measured_conditions"]
    groups = [
        ("$V$", COLOUR_V, "V"),
        ("$U$", COLOUR_U, "U"),
        ("$V{+}U$", COLOUR_FUSED, "UV"),
    ]
    x = np.arange(len(groups))
    width = 0.36
    for index, (label, colour, key) in enumerate(groups):
        if key == "U":
            axis.bar(
                x[index],
                measured[key]["f1_d"],
                width=width,
                color=colour,
                edgecolor="#222222",
                linewidth=0.5,
            )
            continue
        axis.bar(
            x[index] - width / 2,
            measured[key]["f1_d"],
            width=width,
            color=colour,
            edgecolor="#222222",
            linewidth=0.5,
        )
        if ideal is not None:
            mean, sd = ideal[f"{key}:f1_d"]
            axis.bar(
                x[index] + width / 2,
                mean,
                width=width,
                color=colour,
                edgecolor="#222222",
                linewidth=0.5,
                hatch="///",
            )
            axis.errorbar(
                x[index] + width / 2,
                mean,
                yerr=sd,
                fmt="none",
                ecolor="#222222",
                elinewidth=0.7,
                capsize=2,
            )
    axis.set_xticks(x, [label for label, _, _ in groups])
    axis.set_ylabel(r"Macro-$F_1$")
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor="0.75", edgecolor="#222222", label="Measured"),
            Patch(facecolor="0.75", edgecolor="#222222", hatch="///", label="Ideal (sim)"),
            Patch(facecolor=COLOUR_U, edgecolor="#222222", label=r"$U$ reference"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )
    panel_label(axis, "(a)")

    columns = []
    col_labels = []
    for key, label in (("V", r"$V$"), ("UV", r"$V{+}U$")):
        columns.append([measured[key]["recall"][lab] for lab in CLASSES])
        col_labels.append(label)
        if ideal_reps is not None:
            columns.append(
                [
                    float(np.mean([rep[key]["recall"][lab] for rep in ideal_reps]))
                    for lab in CLASSES
                ]
            )
            col_labels.append(label + " ideal")
    # Place U between V and V+U.
    u_column = [measured["U"]["recall"][lab] for lab in CLASSES]
    insert_at = 2
    columns.insert(insert_at, u_column)
    col_labels.insert(insert_at, r"$U$")
    matrix = np.asarray(columns, dtype=float).T
    image = heat.imshow(matrix, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            heat.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.62 else "black",
            )
    heat.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right")
    heat.set_yticks(range(len(CLASSES)), [rf"${lab}$" for lab in CLASSES])
    heat.set_ylabel("Label")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Recall")
    colorbar.outline.set_linewidth(0.5)
    panel_label(heat, "(b)")
    stamp(fig)
    save_figure(fig, out / "sim_f_fusion.pdf")


def plot_curve(
    out: Path,
    sweep: dict[float, dict[float, list[dict]]],
    measured_f1: float,
    u_f1: float,
    ideal_f1: float | None,
    human: float | None,
) -> None:
    fig, axis = plt.subplots(figsize=(160 * MM, 108 * MM))
    styles = {
        0.6: {"color": "black", "name": "s = 0.6"},
        1.0: {"color": "0.45", "name": "s = 1.0"},
    }
    for s, style in styles.items():
        points = []
        for q in Q_GRID:
            reps = sweep[s][q]
            x_mean, _ = mean_sd([rep["vision_f1_folds"] for rep in reps])
            y_f1, y_f1_sd = mean_sd([rep["UV"]["f1_d"] for rep in reps])
            y_rec, y_rec_sd = mean_sd([rep["UV"]["recall"]["P"] for rep in reps])
            points.append((x_mean, y_f1, y_f1_sd, y_rec, y_rec_sd))
        points.sort(key=lambda item: item[0])
        xs = [item[0] for item in points]
        y_f1 = np.asarray([item[1] for item in points])
        y_f1_sd = np.asarray([item[2] for item in points])
        y_rec = np.asarray([item[3] for item in points])
        y_rec_sd = np.asarray([item[4] for item in points])
        colour = style["color"]
        axis.plot(
            xs,
            y_f1,
            color=colour,
            linestyle="-",
            marker="o",
            markersize=3.2,
            linewidth=1.0,
            label=rf"V+U macro-$F_1$, {style['name']}",
        )
        axis.fill_between(xs, y_f1 - y_f1_sd, y_f1 + y_f1_sd, color=colour, alpha=0.12, linewidth=0)
        axis.plot(
            xs,
            y_rec,
            color=colour,
            linestyle="--",
            marker="s",
            markersize=3.2,
            linewidth=1.0,
            label=rf"V+U $P$ recall, {style['name']}",
        )
        axis.fill_between(xs, y_rec - y_rec_sd, y_rec + y_rec_sd, color=colour, alpha=0.08, linewidth=0)
    axis.axhline(u_f1, color=COLOUR_U, linewidth=0.9, label="U alone")
    if human is not None:
        axis.axhline(
            human,
            color="0.25",
            linestyle=(0, (6, 1.5)),
            linewidth=0.8,
            label="Human baseline",
        )
    axis.axvline(
        measured_f1,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=r"Measured vision $F_1$",
    )
    if ideal_f1 is not None:
        axis.axvline(
            ideal_f1,
            color="black",
            linestyle="-.",
            linewidth=1.0,
            label=r"Ideal vision $F_1$",
        )
    axis.set_xlabel(r"Simulated vision macro-$F_1$")
    axis.set_ylabel(r"V+U macro-$F_1$ or $P$ recall")
    axis.set_ylim(0.0, 1.02)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.34),
        ncol=2,
        fontsize=7,
        frameon=False,
        handlelength=3.8,
    )
    stamp(fig)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.68)
    save_figure(fig, out / "sim_f_curve.pdf")


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt_num_md(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def ece_failures(sweep: dict[float, dict[float, list[dict]]]) -> list[str]:
    failed = []
    for s in S_GRID:
        for q in Q_GRID:
            mean_ece, _ = mean_sd([rep["ece"] for rep in sweep[s][q]])
            if mean_ece >= 0.05:
                failed.append(f"s={s:.1f}, q={q:.1f}, mean ECE {mean_ece:.4f}")
    return failed


def write_summary(
    out: Path,
    context: dict,
    sweep: dict,
    ideal_q: float | None,
    ideal: dict | None,
    perfect: dict,
    font_name: str,
    human: float | None,
) -> None:
    vision = context["measured_vision"]
    u_metrics = context["measured_conditions"]["U"]
    n_porosity = sum(cell["target"] == "{P}" for cell in context["cells"])
    lines = [
        "# Simulated ideal vision",
        "",
        "Generated by `simulate_ideal_vision.py`. Every number below is computed from the",
        "input files or from the replicates drawn in this run. Rows marked Measured are the",
        "real module. Every other performance number is simulated.",
        "",
        f"Design: seed {SEED}, {N_REPS} replicates, q in {{0, 0.1, ..., 1}}, s in {{1.0, 0.6}}.",
        "Standard deviations across replicates use the population divisor n, the same",
        "convention as the stored vision fold SD. The random generator is numpy's Generator.",
        "Loop order is s = 1.0, then s = 0.6; q ascending; replicate 0 to 49. Inside a",
        "replicate the hidden-porosity subset is drawn first, then each cell in cell-id",
        "order draws an outcome and a confidence.",
        f"Figure font: {font_name}.",
        "",
        "## Validation gate",
        "",
        md_table(
            ["Check", "Status", "Detail"],
            [
                [name, "PASS" if ok else "FAIL", detail.replace("|", "/")]
                for name, ok, detail in context["rows"]
            ],
        ),
        "",
        "RESULT: " + ("PASS" if context["ok"] else "FAIL"),
        "",
        "## Simulation model",
        "",
        "The measured vision confusion matrix, with rows and columns ordered {P}, {N,U},",
        "{N,U,L}, is row-normalised and mixed with the identity:",
        "C(q) = (1 - q) C_real + q I, rows normalised again. q = 0 keeps the measured",
        "pattern of mistakes. q = 1 predicts the vision target on every cell.",
        "",
        "A porosity cell's vision target is {P}. Surface visibility s keeps that target.",
        "A fraction 1 - s of the porosity cells, a fixed count rounded to the nearest",
        f"integer and redrawn for every replicate, is treated as internal porosity. At s = 0.6",
        f"that count is {hidden_count(n_porosity, 0.6)} of {n_porosity} porosity cells, and",
        "those cells take the target {N,U,L}. At s = 1.0 none are hidden. Sound cells that",
        "are flagged overfill or misalignment already have target {N,U}; other sound cells",
        "and lack-of-fusion cells have target {N,U,L}. Undercut cells have target {N,U}.",
        "",
        "Each cell draws a hard outcome from its row of C(q). The confidence of that outcome",
        "is drawn from a Beta distribution with concentration 20 whose mean is the expected",
        "accuracy of C(q) on that replicate's targets. The mean is moved inside",
        "(1e-6, 1 - 1e-6) only so the Beta parameters stay positive. The draw is then",
        "clipped to [0.34, 0.98]. The other two outcomes share the remaining probability",
        "equally. Expected calibration error uses 15 equal-width bins and compares accuracy",
        "with the mean confidence in the bin.",
        "",
        "Macro-F1 is computed on the three vision outcomes inside each fold, then averaged",
        "over the four folds. That average sets alpha_V by the measured discount formula,",
        "using the vision trivial baseline stored in the calibration file. Mass on a vision",
        "outcome is (1 - alpha_V) times its probability, and the rest sits on the frame.",
        "Those masses are fused with the recovered ultrasonic masses. A single sensor is",
        "not combined with anything, so its conflict is zero. Two sensors use Dempster's",
        "rule, and Murphy's rule as the comparator: average the two mass functions, then",
        "combine the average with itself once.",
        "",
        "The decision is the class with the largest pignistic probability. A tie within",
        "1e-9 takes the first class in alphabetical order, L then N then P then U.",
        "Every focal element of vision and of ultrasonics that contains U also contains N,",
        "so those two pignistic probabilities stay equal and the tie-break does not return U.",
        "A perfect draw at q = 1 still predicts a compound vision target. It does not name",
        "the singleton {N} or {U}, because the camera is not given that target.",
        "",
        "Geometry is omitted. Its mass on the whole frame is 1 on every cell, and a vacuous",
        "source does not change Dempster's combination.",
        "",
        "## Reference levels",
        "",
        "Ideal is the smallest q at s = 0.6 whose mean vision macro-F1 is at least 0.65",
        "and whose mean recall of the vision outcome {P} is at least 0.70. Perfect is",
        "q = 1 and s = 1. The macro-F1 here is the mean over folds, then the mean over",
        "replicates. alpha_V is the mean of the replicate discounts.",
        "",
    ]
    ref_rows = []
    if ideal is None or ideal_q is None:
        ref_rows.append(["Ideal", "not reached", "0.6", "not reached", "not reached"])
    else:
        ref_rows.append(
            [
                "Ideal",
                f"{ideal_q:.1f}",
                "0.6",
                pm_md(*ideal["f1"], 4),
                pm_md(*ideal["alpha"], 4),
            ]
        )
    ref_rows.append(["Perfect", "1.0", "1.0", pm_md(*perfect["f1"], 4), pm_md(*perfect["alpha"], 4)])
    lines.append(md_table(["Level", "q", "s", "Vision macro-F1", "alpha_V"], ref_rows))
    lines.extend(["", "## Tables", ""])
    lines.extend(summary_tables(context, ideal, perfect, ideal_q, sweep))
    lines.extend(["", "## When V+U beats U alone", ""])
    lines.append(
        "The margin is judged on the mean over replicates. V+U macro-F1 must be at least "
        "0.02 above measured U alone, and V+U recall of class P must be at least 0.10 above "
        "measured U alone. Class-P recall is the porosity recall of the fused decision. "
        "It is not the vision module's recall of the outcome {P}, which is the quantity "
        "used to choose the Ideal level."
    )
    lines.append("")
    cross_rows = []
    for s in S_GRID:
        found = crossing(sweep[s], u_metrics["f1_d"], u_metrics["recall"]["P"])
        if found is None:
            cross_rows.append([f"{s:.1f}", "not reached", "not reached"])
        else:
            q_hit, vision_f1 = found
            cross_rows.append([f"{s:.1f}", f"{vision_f1:.4f}", f"{q_hit:.1f}"])
    lines.append(
        md_table(["s", "Minimum vision macro-F1", "q"], cross_rows)
    )
    lines.append("")
    lines.append(
        "The s = 0.6 row is the realistic surface-visibility case. The s = 1.0 row is the "
        "same sweep with every porosity cell left visible to the camera."
    )
    lines.extend(["", "## Calibration check", ""])
    failed = ece_failures(sweep)
    if not failed:
        lines.append("Mean ECE is below 0.05 at every q and at both visibility levels.")
    else:
        lines.append("Mean ECE is not below 0.05 at the following levels:")
        lines.append("")
        for item in failed:
            lines.append(f"- {item}")
        lines.append("")
        lines.append(
            "The confidence draw does not depend on whether the sampled outcome matches "
            "the target. Its spread covers more than one bin of width 1/15, so bins with "
            "different mean confidence have nearly the same accuracy. The check is reported "
            "and the draw is left as specified."
        )
    lines.append("")
    lines.append("ECE for the Ideal and Perfect rows is also reported in the modality table.")
    if human is not None:
        lines.extend(
            [
                "",
                "## Human baseline",
                "",
                f"The curve includes a horizontal line at the supplied value {human:.4f}. "
                "The script does not estimate a human baseline.",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Only vision is simulated. Ultrasonic masses, labels, folds, the discount rule,",
            "Dempster's rule, Murphy's rule, the pignistic decision and the alphabetical",
            "tie-break are the measured ones.",
            "",
            "The simulated errors follow the measured confusion pattern, scaled toward the",
            "identity as q increases. They are not a separate model of the mistakes a better",
            "camera would make, and they do not add failure modes the measured matrix does",
            "not already contain.",
            "",
            "The results show the capacity of this fusion framework when the vision sensor",
            "is cleaner. They are not the measured performance of the trained vision module.",
            "",
            "## Captions",
            "",
            "Paste each sentence into `\\caption{}`. The same sentence is the first comment",
            "line of the matching `.tex` file.",
            "",
        ]
    )
    for name in (
        "sim_t_modality.tex",
        "sim_t_elements.tex",
        "sim_t_ablation.tex",
        "sim_t_recall.tex",
        "sim_t_mcnemar.tex",
    ):
        first = (out / name).read_text(encoding="ascii").splitlines()[0]
        caption = first.replace("% Simulated: ", "", 1)
        lines.append(f"- `{name}`: Simulated: {caption}")
    lines.append("")
    (out / "sim_summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def summary_tables(context, ideal, perfect, ideal_q, sweep) -> list[str]:
    vision = context["measured_vision"]
    blocks = ["### Modality", ""]
    headers = ["Vision", "Macro-F1", "Pooled F1", "{P} recall", "{P} precision", "ECE", "alpha_V"]
    rows = [
        [
            "Measured",
            pm_md(vision["f1_mean"], vision["f1_std"]),
            fmt_num_md(vision["pooled"]),
            fmt_num_md(vision["p_recall"]),
            fmt_num_md(vision["p_precision"]),
            fmt_num_md(vision["ece"]),
            fmt_num_md(vision["alpha"]),
        ]
    ]
    for label, pack in (("Ideal (sim)", ideal), ("Perfect (sim)", perfect)):
        if pack is None:
            rows.append([label, "not reached", "not reached", "not reached", "not reached", "not reached", "not reached"])
        else:
            rows.append(
                [
                    label,
                    pm_md(*pack["f1"]),
                    pm_md(*pack["pooled"]),
                    pm_md(*pack["p_recall"]),
                    pm_md(*pack["p_precision"]),
                    pm_md(*pack["ece"]),
                    pm_md(*pack["alpha"]),
                ]
            )
    blocks.append(md_table(headers, rows))
    blocks.extend(["", "### Elements", ""])
    element_rows = []
    for name in OUTCOMES:
        stats = vision["outcomes"][name]
        element_rows.append(
            [
                "Measured",
                name,
                fmt_num_md(stats["precision"]),
                fmt_num_md(stats["recall"]),
                fmt_num_md(stats["f1"]),
                str(int(round(stats["support"]))),
            ]
        )
    if ideal is None:
        for name in OUTCOMES:
            element_rows.append(["Ideal (sim)", name, "not reached", "not reached", "not reached", "not reached"])
    else:
        for name in OUTCOMES:
            element_rows.append(
                [
                    "Ideal (sim)",
                    name,
                    pm_md(*ideal[f"{name}:precision"]),
                    pm_md(*ideal[f"{name}:recall"]),
                    pm_md(*ideal[f"{name}:f1"]),
                    pm_md(*ideal[f"{name}:support"], 1),
                ]
            )
    blocks.append(md_table(["Version", "Outcome", "Precision", "Recall", "F1", "Support"], element_rows))

    blocks.extend(["", "### Ablation", ""])
    ablation_rows = []
    quantities = (
        ("f1_d", "Dempster macro-F1", 3, False),
        ("mean_k", "Mean K", 3, False),
        ("l_dec", "{L} decisions", 1, True),
        ("nu_ties", "N/U ties", 1, True),
        ("f1_m", "Murphy macro-F1", 3, False),
    )
    conditions = context["measured_conditions"]

    def show(value: float, digits: int, is_count: bool) -> str:
        if is_count:
            return str(int(value))
        return fmt_num_md(value, digits)

    for key, title in (("V", "V"), ("U", "U (reference)"), ("UV", "V+U")):
        for field, label, digits, is_count in quantities:
            measured = show(conditions[key][field], digits, is_count)
            if key == "U":
                ablation_rows.append([title, label, measured, measured, measured])
            elif ideal is None:
                ablation_rows.append([title, label, measured, "not reached", pm_md(*perfect[f"{key}:{field}"], digits)])
            else:
                ablation_rows.append(
                    [
                        title,
                        label,
                        measured,
                        pm_md(*ideal[f"{key}:{field}"], digits),
                        pm_md(*perfect[f"{key}:{field}"], digits),
                    ]
                )
    blocks.append(md_table(["Condition", "Quantity", "Measured", "Ideal (sim)", "Perfect (sim)"], ablation_rows))
    blocks.append("")
    blocks.append("U is the measured module. The three U columns are the same reference.")

    blocks.extend(["", "### Recall", ""])
    recall_rows = []
    for key, title in (("V", "V"), ("U", "U"), ("UV", "V+U")):
        measured = conditions[key]["recall"]
        recall_rows.append(
            [title, "Reference" if key == "U" else "Measured"]
            + [fmt_num_md(measured[label]) for label in CLASSES]
        )
        if key == "U":
            continue
        if ideal is None:
            recall_rows.append([title, "Ideal (sim)", "not reached", "not reached", "not reached", "not reached"])
        else:
            recall_rows.append(
                [title, "Ideal (sim)"]
                + [pm_md(*ideal[f"{key}:recall:{label}"]) for label in CLASSES]
            )
    blocks.append(md_table(["Condition", "Version", "N", "U", "P", "L"], recall_rows))

    blocks.extend(["", "### McNemar", ""])
    blocks.append(
        "Exploratory, not a declared comparison. Comparison is U to V+U. "
        "b is correct only under V+U. c is correct only under U. "
        "Chi-square is the continuity-corrected statistic. The mean row uses the mean b and mean c."
    )
    blocks.append("")
    mcnemar_rows = []
    record = context["measured_mcnemar"]
    mcnemar_rows.append(
        ["Measured", "single", str(record["b"]), str(record["c"]), fmt_chi(record["chi2"]), fmt_p(record["p"]).replace("$", "").replace("\\times 10^{", "e").replace("}", "")]
    )
    # The LaTeX p format is awkward in Markdown. Format Markdown p-values directly.
    mcnemar_rows = [md_mcnemar("Measured", "single", record, True)]
    blocks.append("PLACEHOLDER_MCNEMAR")
    blocks.append("")
    blocks.extend(["### Quality sweep", ""])
    blocks.append(
        "Vision macro-F1 is the mean over folds. V+U columns are Dempster's rule. "
        "{P} recall is the vision outcome. P recall is class P under the fused decision."
    )
    blocks.append("")
    for s in S_GRID:
        blocks.append(f"s = {s:.1f}")
        blocks.append("")
        sweep_rows = []
        for q in Q_GRID:
            reps = sweep[s][q]
            sweep_rows.append(
                [
                    f"{q:.1f}",
                    pm_md(*mean_sd([rep["vision_f1_folds"] for rep in reps])),
                    pm_md(*mean_sd([rep["outcome_p_recall"] for rep in reps])),
                    pm_md(*mean_sd([rep["ece"] for rep in reps])),
                    pm_md(*mean_sd([rep["alpha"] for rep in reps])),
                    pm_md(*mean_sd([rep["UV"]["f1_d"] for rep in reps])),
                    pm_md(*mean_sd([rep["UV"]["recall"]["P"] for rep in reps])),
                ]
            )
        blocks.append(
            md_table(
                ["q", "Vision F1", "{P} recall", "ECE", "alpha_V", "V+U F1", "V+U P recall"],
                sweep_rows,
            )
        )
        blocks.append("")
    return blocks


def md_p(p: float) -> str:
    if p != p:
        return "---"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def md_mcnemar(version: str, aggregate: str, record: dict, as_int: bool) -> list[str]:
    if as_int:
        b_text, c_text = str(int(record["b"])), str(int(record["c"]))
    else:
        b_text, c_text = f"{record['b']:.2f}", f"{record['c']:.2f}"
    chi = "---" if record["chi2"] != record["chi2"] else f"{record['chi2']:.2f}"
    return [version, aggregate, b_text, c_text, chi, md_p(record["p"])]


def write_summary_mcnemar_fix(text: str, context, ideal_reps, perfect_reps) -> str:
    rows = [md_mcnemar("Measured", "single", context["measured_mcnemar"], True)]
    for label, replicates in (("Ideal (sim)", ideal_reps), ("Perfect (sim)", perfect_reps)):
        if not replicates:
            rows.append([label, "median replicate", "not reached", "not reached", "---", "---"])
            rows.append([label, "mean over replicates", "not reached", "not reached", "---", "---"])
            continue
        chosen = replicates[representative_index(replicates)]
        stat, p_value = mcnemar_stat(chosen["b"], chosen["c"])
        rows.append(md_mcnemar(label, "median replicate", {"b": chosen["b"], "c": chosen["c"], "chi2": stat, "p": p_value}, True))
        b_mean = float(np.mean([rep["b"] for rep in replicates]))
        c_mean = float(np.mean([rep["c"] for rep in replicates]))
        stat_m, p_m = mcnemar_stat(b_mean, c_mean)
        rows.append(
            md_mcnemar(
                label,
                "mean over replicates",
                {"b": b_mean, "c": c_mean, "chi2": stat_m, "p": p_m},
                False,
            )
        )
    table = md_table(["Version", "Aggregate", "b", "c", "chi-square", "p"], rows)
    return text.replace("PLACEHOLDER_MCNEMAR", table)


def main() -> None:
    args = parse_args()
    out = guard_output(args.out, args.repo)
    labels, fusion, calibration = load_inputs(args)
    context = run_gate(labels, fusion, calibration)
    print_gate(context["rows"])
    if not context["ok"]:
        print("Validation gate failed. No simulation outputs were written.")
        sys.exit(1)

    print("Gate passed. Simulating vision only. Ultrasonic masses stay measured.")
    sweep = simulate(context)
    ideal_q = select_ideal(sweep[0.6])
    ideal_reps = None if ideal_q is None else sweep[0.6][ideal_q]
    perfect_reps = sweep[1.0][1.0]
    ideal = None if ideal_reps is None else level_pack(ideal_reps)
    perfect = level_pack(perfect_reps)

    font_name = setup_style()
    write_tables_clean(out, context, ideal, perfect, ideal_q, ideal_reps, perfect_reps)
    plot_confusion(out, context["measured_vision"]["confusion"], ideal_reps)
    plot_fusion(out, context, ideal, ideal_reps)
    ideal_f1 = None if ideal is None else ideal["f1"][0]
    plot_curve(
        out,
        sweep,
        context["measured_vision"]["f1_mean"],
        context["measured_conditions"]["U"]["f1_d"],
        ideal_f1,
        args.human_baseline,
    )
    write_summary(out, context, sweep, ideal_q, ideal, perfect, font_name, args.human_baseline)
    summary_path = out / "sim_summary.md"
    summary_path.write_text(
        write_summary_mcnemar_fix(summary_path.read_text(encoding="utf-8"), context, ideal_reps, perfect_reps),
        encoding="utf-8",
        newline="\n",
    )
    expected = [
        "sim_t_modality.tex",
        "sim_t_elements.tex",
        "sim_t_ablation.tex",
        "sim_t_recall.tex",
        "sim_t_mcnemar.tex",
        "sim_f_confusion.pdf",
        "sim_f_confusion.png",
        "sim_f_fusion.pdf",
        "sim_f_fusion.png",
        "sim_f_curve.pdf",
        "sim_f_curve.png",
        "sim_summary.md",
    ]
    missing = [name for name in expected if not (out / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")
    for name in expected:
        if not name.startswith("sim_"):
            raise RuntimeError(name)
        if name.endswith(".tex"):
            first = (out / name).read_text(encoding="ascii").splitlines()[0]
            if not first.startswith("% Simulated:"):
                raise RuntimeError(f"{name} caption does not start with Simulated.")
    print(f"Wrote {len(expected)} files to {out}")


if __name__ == "__main__":
    main()
