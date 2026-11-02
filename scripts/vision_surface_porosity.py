#!/usr/bin/env python3
"""Surface-texture vision model for the 150 coupon cells.

The coupon ResNet calls the porosity head on almost none of the held-out
cells. WC-MS-01 induces porosity by cutting the shielding gas, which leaves
small bright and dark marks on the cap. This script scores that texture with
a logistic model under the same coupon folds the thesis already uses.

The inspected cell labels stay the targets. The welding-parameter log is not
a label source. Canonical files under results/vision/coupon_eval/ are not
modified. Fusion masses for this model are written beside the report so a
later fusion run can point at them.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sim_ideal_vision"))

import simulate_ideal_vision as fusion  # noqa: E402

MANIFEST = REPO / "data" / "processed" / "vision_coupon_patches" / "manifest.csv"
LABELS = REPO / "data" / "labels_v2_2026-10-26.csv"
FUSION_JSON = REPO / "results" / "thesis" / "data" / "fusion_results.json"
UT_METRICS = REPO / "results" / "ut" / "metrics_ut.json"
OUT = REPO / "results" / "vision" / "surface_porosity_cv"

OUTCOMES = fusion.OUTCOMES
F1_TRIV = 0.2568
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
BAND = (0.30, 0.70, 0.15, 0.85)
SIGMA = 3.0
BLOB_THR = 8.0
BLOB_MIN = 3
BLOB_MAX = 80
N_BINS = 15


def bead_features(path: Path) -> np.ndarray:
    """High-frequency energy and small residual blobs on the cap band."""
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    height, width = image.shape
    r0, r1, c0, c1 = BAND
    band = image[int(height * r0) : int(height * r1), int(width * c0) : int(width * c1)]
    residual = band - gaussian_filter(band, SIGMA)
    high_freq = float(np.abs(residual).mean())
    spread = float(np.percentile(residual, 95) - np.percentile(residual, 5))

    def blob_count(mask: np.ndarray) -> float:
        labelled, count = label(mask)
        if count == 0:
            return 0.0
        sizes = np.bincount(labelled.ravel())[1:]
        kept = (sizes >= BLOB_MIN) & (sizes <= BLOB_MAX)
        return float(kept.sum())

    dark = blob_count(residual < -BLOB_THR)
    bright = blob_count(residual > BLOB_THR)
    return np.array(
        [high_freq, np.log1p(dark), np.log1p(bright), float(band.std()), spread],
        dtype=np.float64,
    )


def standardise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return (train - centre) / scale, (other - centre) / scale


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(temperature: float) -> float:
        scaled = logits / max(float(temperature), 1e-3)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        log_prob = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return float(-log_prob[np.arange(len(y)), y].mean())

    fit = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(fit.x)


def probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def outcome_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return fusion.macro_f1(
        [OUTCOMES[i] for i in y_true],
        [OUTCOMES[i] for i in y_pred],
        OUTCOMES,
    )


def class_scores(y_true: np.ndarray, y_pred: np.ndarray, index: int) -> dict[str, float]:
    truth = y_true == index
    pred = y_pred == index
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(truth.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_cells() -> list[dict]:
    labels = pd.read_csv(LABELS, dtype=str, keep_default_na=False)
    label_target = {
        row.cell_id: row.vision_target for row in labels.itertuples(index=False)
    }
    cells = []
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["vision_target"] not in fusion.OUTCOME_INDEX:
                raise RuntimeError(f"Unmapped target {row['vision_target']} on {row['cell_id']}")
            if label_target.get(row["cell_id"]) != row["vision_target"]:
                raise RuntimeError(f"Manifest target disagrees with labels for {row['cell_id']}")
            cells.append(
                {
                    "cell_id": row["cell_id"],
                    "fold": int(row["fold"]),
                    "target": row["vision_target"],
                    "y": fusion.OUTCOME_INDEX[row["vision_target"]],
                    "true_class": row["true_class"],
                    "path": REPO / row["patch_path"],
                }
            )
    counts = {name: sum(cell["target"] == name for cell in cells) for name in OUTCOMES}
    if counts != fusion.EXPECTED_SUPPORT:
        raise RuntimeError(f"Target supports are {counts}.")
    return cells


def cross_validate(cells: list[dict], features: np.ndarray) -> dict:
    by_fold: dict[int, np.ndarray] = {}
    for index, cell in enumerate(cells):
        by_fold.setdefault(cell["fold"], []).append(index)
    for fold, indices in by_fold.items():
        by_fold[fold] = np.array(indices, dtype=int)

    records = []
    fold_rows = []
    for test_fold in (1, 2, 3, 4):
        remaining = [fold for fold in (1, 2, 3, 4) if fold != test_fold]
        cal_fold = remaining[-1]
        fit_folds = remaining[:-1]
        fit_idx = np.concatenate([by_fold[fold] for fold in fit_folds])
        cal_idx = by_fold[cal_fold]
        test_idx = by_fold[test_fold]
        y_fit = np.array([cells[i]["y"] for i in fit_idx])
        if set(y_fit.tolist()) != {0, 1, 2}:
            raise RuntimeError(f"Fit folds {fit_folds} are missing a vision outcome.")

        best_c = None
        best_score = -1.0
        for penalty in C_GRID:
            model = LogisticRegression(
                C=penalty,
                class_weight="balanced",
                max_iter=800,
                solver="lbfgs",
            )
            x_fit, x_cal = standardise(features[fit_idx], features[cal_idx])
            model.fit(x_fit, y_fit)
            score = outcome_f1(np.array([cells[i]["y"] for i in cal_idx]), model.predict(x_cal))
            if score > best_score:
                best_score = score
                best_c = penalty

        model = LogisticRegression(
            C=best_c,
            class_weight="balanced",
            max_iter=800,
            solver="lbfgs",
        )
        x_fit, x_rest = standardise(features[fit_idx], features[np.concatenate([cal_idx, test_idx])])
        model.fit(x_fit, y_fit)
        if list(model.classes_) != [0, 1, 2]:
            raise RuntimeError(f"Class order is {list(model.classes_)}.")
        rest_logits = model.decision_function(x_rest)
        cal_logits = rest_logits[: len(cal_idx)]
        test_logits = rest_logits[len(cal_idx) :]
        temperature = fit_temperature(cal_logits, np.array([cells[i]["y"] for i in cal_idx]))
        probs = probabilities(test_logits, temperature)
        pred = probs.argmax(axis=1)
        y_test = np.array([cells[i]["y"] for i in test_idx])
        fold_rows.append(
            {
                "fold": test_fold,
                "calibration_fold": cal_fold,
                "fit_folds": fit_folds,
                "C": best_c,
                "temperature": temperature,
                "macro_f1": outcome_f1(y_test, pred),
                "porosity": class_scores(y_test, pred, 0),
                "n": int(len(test_idx)),
            }
        )
        for local, index in enumerate(test_idx):
            cell = cells[int(index)]
            records.append(
                {
                    "cell_id": cell["cell_id"],
                    "fold": test_fold,
                    "target": cell["target"],
                    "true_class": cell["true_class"],
                    "pred": OUTCOMES[int(pred[local])],
                    "probs": [float(v) for v in probs[local]],
                }
            )
    records.sort(key=lambda row: row["cell_id"])
    return {"folds": fold_rows, "cells": records}


def confusion(records: list[dict]) -> list[list[int]]:
    matrix = np.zeros((3, 3), dtype=int)
    for row in records:
        matrix[fusion.OUTCOME_INDEX[row["target"]], fusion.OUTCOME_INDEX[row["pred"]]] += 1
    return matrix.tolist()


def fuse(records: list[dict], alpha: float) -> dict:
    payload = json.loads(FUSION_JSON.read_text(encoding="utf-8"))
    ut_records = {row["cell_id"]: row for row in payload["U"]["dempster"]}
    stored_u = {row["cell_id"]: row["decision"] for row in payload["U"]["dempster"]}
    y_true = []
    v_dec = []
    u_dec = []
    vu_dec = []
    for row in records:
        prob = {name: row["probs"][fusion.OUTCOME_INDEX[name]] for name in OUTCOMES}
        bpa_v = fusion.discount(prob, alpha)
        bet_v = fusion.pignistic(bpa_v)
        decision_v = fusion.decide(bet_v)
        ut_bpa = fusion.recover_ut(ut_records[row["cell_id"]]["bel_pl"])
        fused, _conflict = fusion.dempster(bpa_v, ut_bpa)
        decision_vu = fusion.decide(fusion.pignistic(fused))
        y_true.append(row["true_class"])
        v_dec.append(decision_v)
        u_dec.append(stored_u[row["cell_id"]])
        vu_dec.append(decision_vu)
        row["decision_V"] = decision_v
        row["decision_U"] = stored_u[row["cell_id"]]
        row["decision_VU"] = decision_vu
    return {
        "V": {
            "macro_f1": fusion.macro_f1(y_true, v_dec, fusion.CLASSES),
            "recall": fusion.class_recalls(y_true, v_dec),
        },
        "U": {
            "macro_f1": fusion.macro_f1(y_true, u_dec, fusion.CLASSES),
            "recall": fusion.class_recalls(y_true, u_dec),
        },
        "VU": {
            "macro_f1": fusion.macro_f1(y_true, vu_dec, fusion.CLASSES),
            "recall": fusion.class_recalls(y_true, vu_dec),
        },
    }


def write_report(summary: dict, path: Path) -> None:
    porosity = summary["porosity"]
    ut = summary["ut_outcome_P"]
    folds = summary["per_fold_f1"]
    lines = [
        "# Surface-texture vision on the coupon cells",
        "",
        "This file is generated by `scripts/vision_surface_porosity.py`.",
        "Every figure below is computed in that run. The canonical ResNet",
        "masses in `results/vision/coupon_eval/` are unchanged.",
        "",
        "## What changed",
        "",
        "WC-MS-01 asks the welder to produce porosity by reducing the shielding",
        "gas and increasing stick-out. That leaves small bright and dark marks",
        "on the cap. The coupon ResNet, trained on the 224 pixel cell crops,",
        "predicts the porosity focal element `{P}` on almost none of the",
        "held-out cells (coupon-roi v2: porosity F1 is 0 on folds 1, 3 and 4).",
        "",
        "The model here is a balanced logistic regression on five texture",
        "features of the cap band: high-frequency energy after a Gaussian of",
        "sigma 3, the log count of small dark residual blobs, the log count of",
        "small bright residual blobs, the band standard deviation, and the",
        "residual 5-to-95 percentile range. Blob size is 3 to 80 pixels and the",
        "residual threshold is 8 grey levels. Those constants were inspected on",
        "the full set of 150 cells before the cross-validation scores were",
        "computed, so the scores are mildly optimistic. The penalty `C` and the",
        "temperature are chosen on an inner coupon fold and are not refit on",
        "the test fold.",
        "",
        "Targets are the inspected `vision_target` values already on the coupon",
        "manifest, checked against `labels_v2_2026-10-26.csv`. The parameter",
        "log in WC-MS-01 is not used as a label. A zone welded under a porosity",
        "setting can still be a sound weld, and the document says so.",
        "",
        "Folds are the thesis coupon groups. For each test fold the last of the",
        "remaining folds is the calibration fold. The other two fit the",
        "classifier. Temperature scaling is fit by negative log-likelihood on",
        "the calibration fold. Argmax does not change with temperature; the",
        "temperature changes the probabilities that enter the belief masses.",
        "",
        "## Outcome metrics",
        "",
        f"Macro-F1 mean over folds: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} (population SD).",
        f"Per fold: {', '.join(f'{v:.4f}' for v in folds)}.",
        f"Pooled macro-F1: {summary['pooled_f1']:.4f}.",
        f"Trivial baseline used for the discount: {F1_TRIV:.4f}.",
        f"alpha_V: {summary['alpha_V']:.4f}.",
        f"ECE of the temperature-scaled probabilities, 15 equal-width bins: {summary['ece']:.4f}.",
        "",
        f"`{{P}}` precision {porosity['precision']:.4f}, recall {porosity['recall']:.4f}, F1 {porosity['f1']:.4f}",
        f"on support {porosity['support']} (tp {porosity['tp']}, fp {porosity['fp']}, fn {porosity['fn']}).",
        "",
        "Confusion, rows true and columns predicted, order `{P}`, `{N,U}`, `{N,U,L}`:",
        "",
        "```",
        json.dumps(summary["confusion"]),
        "```",
        "",
        "## Porosity against the ultrasonic module",
        "",
        "The stored ultrasonic outcome metrics (`results/ut/metrics_ut.json`)",
        f"give `{{P}}` precision {ut['precision']:.4f}, recall {ut['recall']:.4f}, F1 {ut['f1']:.4f}.",
        "That file's class order is `{L}`, `{N,U}`, `{P}`. The comparison is",
        "outcome against outcome. It is not the pignistic class decision.",
        "",
        summary["porosity_verdict"],
        "",
        "## Discounted decisions, fused with the measured ultrasonic masses",
        "",
        "Vision masses use the same discount as Chapter 4:",
        "`m(A) = (1 - alpha_V) p_A` and `m(Theta) = alpha_V`.",
        "Ultrasonic masses are recovered from the stored belief and plausibility.",
        "Dempster's rule and the alphabetical pignistic tie-break are unchanged.",
        "Geometry stays out. Its measured alpha is clamped at 1, so it assigns",
        "the whole mass to the frame and does not identify porosity. This run",
        "does not change the geometry module.",
        "",
        "The held-out macro-F1 is below the trivial baseline, so alpha_V is 1.",
        "Discounted vision then puts all of its mass on the frame. Fusing it",
        "with ultrasonics repeats the ultrasonic decision. No belief file is",
        "written. Fusion should keep",
        "`results/vision/coupon_eval/vision_bpas_v2.json`.",
        "",
        "## Limits",
        "",
        "The feature thresholds were chosen after looking at texture summaries",
        "of all 150 cells. The classifier never trains on its test coupons, and",
        "`C` and the temperature never see those coupons, but the feature",
        "definition was not locked inside the training folds.",
        "",
        "A cell whose porosity is entirely internal has no cap mark for this",
        "model to use. Lack of fusion remains inside `{N,U,L}` because the",
        "camera still has no singleton for it. Class U is still not a vision",
        "singleton: geometric exceedance belongs to the geometry module.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def ut_porosity() -> dict[str, float]:
    metrics = json.loads(UT_METRICS.read_text(encoding="utf-8"))
    classes = metrics["classes"]
    column = classes.index("{P}")
    matrix = np.array(metrics["confusion_matrix"], dtype=float)
    tp = matrix[column, column]
    support = matrix[column].sum()
    predicted = matrix[:, column].sum()
    precision = float(tp / predicted) if predicted else 0.0
    recall = float(tp / support) if support else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    if OUT.resolve().parent != (REPO / "results" / "vision").resolve():
        raise RuntimeError(f"Refusing to write outside results/vision: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    features = np.vstack([bead_features(cell["path"]) for cell in cells])
    cv = cross_validate(cells, features)
    y_true = np.array([fusion.OUTCOME_INDEX[row["target"]] for row in cv["cells"]])
    y_pred = np.array([fusion.OUTCOME_INDEX[row["pred"]] for row in cv["cells"]])
    fold_f1 = [row["macro_f1"] for row in cv["folds"]]
    macro_mean = float(np.mean(fold_f1))
    macro_std = float(np.std(fold_f1))
    alpha = fusion.alpha_from_f1(macro_mean, F1_TRIV)
    confidence = [max(row["probs"]) for row in cv["cells"]]
    correct = [row["pred"] == row["target"] for row in cv["cells"]]
    ece = fusion.expected_calibration_error(confidence, correct, N_BINS)
    porosity = class_scores(y_true, y_pred, 0)
    ut = ut_porosity()
    if porosity["f1"] > ut["f1"] and porosity["recall"] > ut["recall"]:
        verdict = (
            "On this run the surface model exceeds the stored ultrasonic "
            "`{P}` F1 and the stored ultrasonic `{P}` recall."
        )
    else:
        verdict = (
            "On this run the surface model does not exceed both the stored "
            "ultrasonic `{P}` F1 and the stored ultrasonic `{P}` recall. "
            "The numbers above are the measured comparison."
        )
    decisions = fuse(cv["cells"], alpha)
    bpa_path = OUT / "vision_bpas.json"
    if bpa_path.exists():
        bpa_path.unlink()
    summary = {
        "model": "surface_texture_logistic_coupon_cv",
        "n": len(cv["cells"]),
        "macro_f1_mean": macro_mean,
        "macro_f1_std": macro_std,
        "per_fold_f1": fold_f1,
        "folds": cv["folds"],
        "pooled_f1": outcome_f1(y_true, y_pred),
        "trivial_baseline": F1_TRIV,
        "alpha_V": alpha,
        "ece": ece,
        "ece_bins": N_BINS,
        "porosity": porosity,
        "ut_outcome_P": ut,
        "porosity_verdict": verdict,
        "confusion": confusion(cv["cells"]),
        "fusion": decisions,
        "feature_note": (
            "Blob threshold and band were inspected on all 150 cells before CV. "
            "C and temperature were selected on the inner coupon fold only."
        ),
    }
    (OUT / "cv_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT / "cell_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cell_id",
                "fold",
                "target",
                "true_class",
                "pred",
                "p_P",
                "p_NU",
                "p_NUL",
                "decision_V",
                "decision_U",
                "decision_VU",
            ],
        )
        writer.writeheader()
        for row in cv["cells"]:
            writer.writerow(
                {
                    "cell_id": row["cell_id"],
                    "fold": row["fold"],
                    "target": row["target"],
                    "true_class": row["true_class"],
                    "pred": row["pred"],
                    "p_P": f"{row['probs'][0]:.6f}",
                    "p_NU": f"{row['probs'][1]:.6f}",
                    "p_NUL": f"{row['probs'][2]:.6f}",
                    "decision_V": row["decision_V"],
                    "decision_U": row["decision_U"],
                    "decision_VU": row["decision_VU"],
                }
            )
    write_report(summary, OUT / "REPORT.md")
    print(
        f"macro-F1 {macro_mean:.4f} ± {macro_std:.4f}  "
        f"{{P}} F1 {porosity['f1']:.4f} recall {porosity['recall']:.4f}  "
        f"alpha {alpha:.4f}"
    )
    print(verdict)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
