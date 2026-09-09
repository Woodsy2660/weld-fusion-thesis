"""Temperature scaling for the vision classifier.

Single scalar T fitted by minimising NLL on the inner validation fold.

    p_i = softmax(z_i / T)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = z / temperature
    z = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def _nll(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    t = max(float(temperature), 1e-6)
    log_p = np.log(softmax(logits, t) + 1e-15)
    return float(-log_p[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit T on an inner validation fold. Does not touch the test fold."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if len(logits) != len(labels):
        raise ValueError("logits and labels length mismatch")
    result = minimize_scalar(
        lambda t: _nll(t, logits, labels),
        bounds=(0.05, 16.0),
        method="bounded",
    )
    return float(result.x)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits, temperature)
