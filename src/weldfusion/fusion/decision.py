"""Pignistic transform and decision rule.

    BetP(x) = sum_{A ∋ x} m(A) / |A| ,  normalised by 1 / (1 - m(empty))
"""

from __future__ import annotations

from weldfusion.frame import Focal, class_order, theta
from weldfusion.types import BPA


def pignistic(masses: BPA | dict[Focal, float]) -> dict[str, float]:
    raw = masses.masses if isinstance(masses, BPA) else masses
    empty = frozenset()
    m_empty = float(raw.get(empty, 0.0))
    denom = 1.0 - m_empty
    if denom <= 0:
        raise ValueError("Pignistic transform undefined: m(empty) >= 1")

    betp = {x: 0.0 for x in theta()}
    for A, m in raw.items():
        if not A:
            continue
        share = m / len(A)
        for x in A:
            if x in betp:
                betp[x] += share
    return {x: betp[x] / denom for x in class_order()}


def decide(masses: BPA | dict[Focal, float]) -> tuple[str, dict[str, float]]:
    """Argmax of BetP, breaking ties by frame order (first wins)."""
    betp = pignistic(masses)
    label = max(betp, key=lambda c: (betp[c], -class_order().index(c)))
    return label, betp
