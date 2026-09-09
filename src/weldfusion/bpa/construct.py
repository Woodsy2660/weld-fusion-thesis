"""Construct a modality BPA from a calibrated posterior over Theta.

Each singleton class is assigned to the *smallest* allowed focal of that
modality that contains it. Masses of classes that share a focal are summed.
There is no fixed uncertainty floor; ignorance is introduced only by
Shafer discounting (see bpa.discount).
"""

from __future__ import annotations

from typing import Mapping

from weldfusion.frame import (
    Focal,
    allowed_focals,
    as_focal,
    theta,
    validate_modality_bpa,
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
