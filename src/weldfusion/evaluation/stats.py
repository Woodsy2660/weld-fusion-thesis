"""Paired tests, bootstrap CIs, Holm-Bonferroni correction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import binomtest


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on paired predictions vs the same ground truth.

    n01 = A wrong, B right; n10 = A right, B wrong.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    n01 = int((~a_ok & b_ok).sum())
    n10 = int((a_ok & ~b_ok).sum())
    n = n01 + n10
    if n == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(n01, n=n, p=0.5, alternative="two-sided").pvalue)
    return {"n01": n01, "n10": n10, "p_value": p_value}


def bootstrap_accuracy_diff(
    y_true,
    pred_a,
    pred_b,
    *,
    n_boot: int | None = None,
    seed: int | None = None,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI on accuracy(B) - accuracy(A)."""
    from weldfusion.utils.config import load_named

    cfg = load_named("evaluation")["statistics"]
    n_boot = int(cfg["bootstrap_n"] if n_boot is None else n_boot)
    seed = int(cfg["bootstrap_seed"] if seed is None else seed)

    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        acc_a = np.mean(pred_a[idx] == y_true[idx])
        acc_b = np.mean(pred_b[idx] == y_true[idx])
        diffs[i] = acc_b - acc_a
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    point = float(np.mean(pred_b == y_true) - np.mean(pred_a == y_true))
    return {
        "diff": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
    }


def holm_bonferroni(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> list[dict]:
    """Holm-Bonferroni adjusted decisions, preserving input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    adjusted = [0.0] * m
    stop = False
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        adj = min(1.0, p_values[i] * (m - rank))
        # Monotone: adjusted p must be non-decreasing in the sorted order.
        if rank > 0:
            adj = max(adj, adjusted[order[rank - 1]])
        adjusted[i] = adj
        if stop or p_values[i] > threshold:
            stop = True
            reject[i] = False
        else:
            reject[i] = True
    return [
        {
            "index": i,
            "p_value": float(p_values[i]),
            "p_adjusted": float(adjusted[i]),
            "reject": bool(reject[i]),
        }
        for i in range(m)
    ]
