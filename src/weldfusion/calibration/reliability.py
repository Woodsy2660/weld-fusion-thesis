"""Shafer discounting parameter from validation-fold reliability.

    alpha_m = 1 - balanced_accuracy_m

Estimated on inner validation folds only. Never on the test fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def reliability_alpha(y_true, y_pred) -> float:
    """Return alpha in [0, 1]. Perfect balanced accuracy -> alpha=0 (no discount)."""
    ba = float(balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))
    alpha = 1.0 - ba
    return float(min(1.0, max(0.0, alpha)))
