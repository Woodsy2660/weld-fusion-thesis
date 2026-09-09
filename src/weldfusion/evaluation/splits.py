"""Grouped nested cross-validation by coupon_id.

No coupon may appear in both train and test of any fold. Calibration
(temperature, isotonic, reliability alpha) fits on inner folds only.
Random cell-level splits are a bug.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


def nested_group_kfold(
    groups: Sequence[Any],
    *,
    n_outer: int | None = None,
    n_inner: int | None = None,
) -> list[dict]:
    """Return outer folds with nested inner train/calibrate splits.

    Each fold is:
        {
          "train": ndarray of sample indices,
          "test": ndarray of sample indices,
          "inner": [{"train": ..., "calibrate": ...}, ...],
        }
    """
    from weldfusion.utils.config import load_named

    ev = load_named("evaluation")["cv"]
    n_outer = int(ev["n_outer"] if n_outer is None else n_outer)
    n_inner = int(ev["n_inner"] if n_inner is None else n_inner)

    groups = np.asarray(groups)
    n = len(groups)
    if n == 0:
        raise ValueError("groups is empty")
    n_unique = len(np.unique(groups))
    if n_unique < n_outer:
        raise ValueError(
            f"Need at least n_outer={n_outer} distinct groups, got {n_unique}"
        )

    indices = np.arange(n)
    outer = GroupKFold(n_splits=n_outer)
    folds = []
    for train_idx, test_idx in outer.split(indices, groups=groups):
        _assert_disjoint_groups(groups, train_idx, test_idx)
        inner_groups = groups[train_idx]
        n_inner_groups = len(np.unique(inner_groups))
        inner_splits = min(n_inner, n_inner_groups)
        inner_folds = []
        if inner_splits >= 2:
            inner = GroupKFold(n_splits=inner_splits)
            inner_x = np.arange(len(train_idx))
            for inner_tr, inner_cal in inner.split(inner_x, groups=inner_groups):
                tr = train_idx[inner_tr]
                cal = train_idx[inner_cal]
                _assert_disjoint_groups(groups, tr, cal)
                inner_folds.append({"train": tr, "calibrate": cal})
        folds.append({"train": train_idx, "test": test_idx, "inner": inner_folds})
    return folds


def _assert_disjoint_groups(groups: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
    overlap = set(groups[a]) & set(groups[b])
    if overlap:
        raise RuntimeError(
            f"Group leakage: {sorted(overlap)} appear in both sides of a split"
        )
