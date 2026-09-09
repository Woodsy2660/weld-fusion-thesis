"""Isotonic regression (primary) and Platt scaling (alternative) for UT.

Both wrap scikit-learn. Fit on inner validation folds only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class IsotonicCalibrator:
    models: list[IsotonicRegression]
    classes: np.ndarray

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        calibrated = np.column_stack(
            [m.predict(scores[:, i]) for i, m in enumerate(self.models)]
        )
        calibrated = np.clip(calibrated, 1e-12, None)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return calibrated


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    """One-vs-rest isotonic regression on per-class scores (e.g. RF predict_proba)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    n_classes = scores.shape[1]
    models = []
    for i in range(n_classes):
        y = (labels == i).astype(float)
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(scores[:, i], y)
        models.append(ir)
    return IsotonicCalibrator(models=models, classes=np.arange(n_classes))


def apply_isotonic(calibrator: IsotonicCalibrator, scores: np.ndarray) -> np.ndarray:
    return calibrator.transform(scores)


@dataclass
class PlattCalibrator:
    model: LogisticRegression
    n_classes: int

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(scores, dtype=np.float64))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
    """Multinomial logistic (Platt-style) calibration on per-class scores."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    model = LogisticRegression(max_iter=1000)
    model.fit(scores, labels)
    return PlattCalibrator(model=model, n_classes=scores.shape[1])
