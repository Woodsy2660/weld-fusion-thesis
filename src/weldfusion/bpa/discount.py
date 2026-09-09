"""Shafer discounting by measured reliability alpha_m.

    m_alpha(A)     = (1 - alpha_m) * m(A)     for all A strictly inside Theta
    m_alpha(Theta) = alpha_m + (1 - alpha_m) * m(Theta)

alpha_m = 1 - balanced_accuracy_m, estimated on validation folds only
(see calibration.reliability). alpha = 0 is the identity; alpha = 1 is
the vacuous BPA.
"""

from __future__ import annotations

from weldfusion.frame import theta, validate_bpa
from weldfusion.types import BPA


def shafer_discount(bpa: BPA, alpha: float) -> BPA:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    th = theta()
    scale = 1.0 - alpha
    masses: dict = {}
    theta_mass = 0.0
    for focal, mass in bpa.masses.items():
        if focal == th:
            theta_mass += mass
        else:
            masses[focal] = scale * mass
    masses[th] = alpha + scale * theta_mass

    # Drop zeros that appear when alpha = 1.
    masses = {A: m for A, m in masses.items() if m > 0.0 or A == th}
    validate_bpa(masses)
    return BPA(masses=masses, modality=bpa.modality, alpha=alpha)
