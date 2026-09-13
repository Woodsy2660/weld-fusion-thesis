"""Construct a modality BPA from calibrated posteriors.

For most modalities: posterior over Theta singletons is mapped to the smallest
allowed focal element containing each singleton.

For vision: posterior is DIRECTLY over the 4 focal elements (V1-V4), NOT over
Theta singletons. Vision never predicts U or L alone.

There is no fixed uncertainty floor; ignorance is introduced only by
Shafer discounting (see bpa.discount).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from weldfusion.frame import (
    Focal,
    allowed_focals,
    as_focal,
    theta,
    validate_modality_bpa,
    vision_heads,
    vision_head_order,
)
from weldfusion.types import BPA


def smallest_allowed_focal(singleton: Focal, allowed: frozenset[Focal]) -> Focal:
    candidates = [A for A in allowed if singleton <= A]
    if not candidates:
        raise ValueError(
            f"No allowed focal contains {set(singleton)}; allowed={ [set(a) for a in allowed] }"
        )
    return min(candidates, key=lambda A: (len(A), tuple(sorted(A))))


def construct_bpa(
    posterior: Mapping[str, float],
    modality: str,
) -> BPA:
    """Map a calibrated posterior p(class) onto the modality's declared focals."""
    th = theta()
    missing = [c for c in th if c not in posterior]
    extra = [c for c in posterior if c not in th]
    if extra:
        raise ValueError(f"Posterior has classes outside Theta: {extra}")
    if missing:
        posterior = {**{c: 0.0 for c in missing}, **dict(posterior)}

    total = sum(float(v) for v in posterior.values())
    if total <= 0:
        raise ValueError("Posterior has non-positive mass")

    allowed = allowed_focals(modality)
    masses: dict[Focal, float] = {}
    for cls, p in posterior.items():
        share = float(p) / total
        if share == 0.0:
            continue
        focal = smallest_allowed_focal(as_focal(cls), allowed)
        masses[focal] = masses.get(focal, 0.0) + share

    # Drop exact zeros; validate_modality_bpa ignores them anyway.
    masses = {A: m for A, m in masses.items() if m > 0.0}
    validate_modality_bpa(masses, modality)
    return BPA(masses=masses, modality=modality)


def construct_vision_bpa(
    posterior: np.ndarray | Sequence[float],
) -> BPA:
    """Construct vision BPA from calibrated posteriors over the 4 heads.

    Vision outputs 4 logits corresponding to focal elements V1-V4, NOT 5
    Theta singletons. This function takes the softmax posterior over these
    heads and assigns mass directly to the corresponding focal elements.

    Parameters
    ----------
    posterior : array-like of shape (4,)
        Calibrated softmax probabilities for V1, V2, V3, V4 in order.

    Returns
    -------
    BPA
        Mass function over vision's declared focal elements.

    Raises
    ------
    ValueError
        If posterior length doesn't match number of vision heads.
    """
    heads = vision_heads()
    head_order = vision_head_order()

    posterior = np.asarray(posterior, dtype=np.float64)
    if len(posterior) != len(head_order):
        raise ValueError(
            f"Vision posterior length {len(posterior)} != {len(head_order)} heads"
        )

    total = posterior.sum()
    if total <= 0:
        raise ValueError("Vision posterior has non-positive total mass")

    masses: dict[Focal, float] = {}
    for head_name, prob in zip(head_order, posterior):
        focal = heads[head_name]
        share = float(prob) / total
        if share > 0.0:
            masses[focal] = masses.get(focal, 0.0) + share

    # Drop exact zeros
    masses = {A: m for A, m in masses.items() if m > 0.0}
    validate_modality_bpa(masses, "vision")
    return BPA(masses=masses, modality="vision")
