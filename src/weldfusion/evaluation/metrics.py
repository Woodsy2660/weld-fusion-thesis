"""Shared evaluation metrics. All modality evaluate scripts import this helper."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from weldfusion.frame import class_order
from weldfusion.utils.config import load_named


def compute_metrics(
    y_true,
    y_pred,
    *,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Per-class P/R/F1, macro-F1, confusion matrix, detection rate, FPR.

    Detection rate and FPR use the binary collapse `defect <=> predicted != N`
    from evaluation.yaml.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    names = list(class_names) if class_names is not None else list(class_order())
    labels = list(range(len(names))) if _looks_like_indices(y_true, names) else names

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    detection, fpr = _binary_rates(y_true, y_pred, names, labels)

    per_class = {}
    for name in names:
        row = report.get(name, {})
        per_class[name] = {
            "precision": row.get("precision", 0.0),
            "recall": row.get("recall", 0.0),
            "f1": row.get("f1-score", 0.0),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm,
        "detection_rate": detection,
        "false_positive_rate": fpr,
        "report": report,
        "class_names": names,
    }


def _looks_like_indices(y, names: Sequence[str]) -> bool:
    y = np.asarray(y)
    if y.dtype.kind in "iu":
        return True
    return False


def _binary_rates(y_true, y_pred, names: Sequence[str], labels) -> tuple[float, float]:
    sound = load_named("evaluation")["binary_collapse"]["defect_is_not"]
    # Map whatever encoding we were given onto "is defect".
    if _looks_like_indices(y_true, names):
        try:
            sound_id = list(names).index(sound)
        except ValueError:
            sound_id = 0
        true_defect = np.asarray(y_true) != sound_id
        pred_defect = np.asarray(y_pred) != sound_id
    else:
        true_defect = np.asarray(y_true) != sound
        pred_defect = np.asarray(y_pred) != sound

    n_pos = int(true_defect.sum())
    n_neg = int((~true_defect).sum())
    tp = int((true_defect & pred_defect).sum())
    fp = int((~true_defect & pred_defect).sum())
    detection = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    return float(detection), float(fpr)
