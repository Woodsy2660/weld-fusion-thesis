"""Dempster's rule of combination for arbitrary subsets of Theta.

    K = sum_{B ∩ C = empty} m1(B) m2(C)
    m12(A) = (sum_{B ∩ C = A} m1(B) m2(C)) / (1 - K)   for A non-empty

K is returned alongside the combined BPA and is never discarded.
K == 1 raises ConflictError rather than dividing by zero.
"""

from __future__ import annotations

from weldfusion.frame import Focal, validate_bpa
from weldfusion.types import BPA

K_TOL = 1e-15


class ConflictError(ValueError):
    """Total conflict (K = 1): Dempster combination is undefined."""


def dempster_combine(
    m1: BPA | dict[Focal, float],
    m2: BPA | dict[Focal, float],
) -> tuple[BPA, float]:
    """Combine two BPAs. Returns (combined BPA, conflict mass K)."""
    masses1 = m1.masses if isinstance(m1, BPA) else m1
    masses2 = m2.masses if isinstance(m2, BPA) else m2

    unnorm: dict[Focal, float] = {}
    k = 0.0
    for B, mB in masses1.items():
        for C, mC in masses2.items():
            prod = mB * mC
            if prod == 0.0:
                continue
            inter = B & C
            if not inter:
                k += prod
            else:
                unnorm[inter] = unnorm.get(inter, 0.0) + prod

    if k >= 1.0 - K_TOL:
        raise ConflictError(f"Total conflict K={k:.6f}; Dempster combination is undefined")

    norm = 1.0 / (1.0 - k)
    combined = {A: v * norm for A, v in unnorm.items()}
    validate_bpa(combined)
    modality = None
    if isinstance(m1, BPA) and isinstance(m2, BPA) and m1.modality == m2.modality:
        modality = m1.modality
    return BPA(masses=combined, modality=modality), k
