"""
Decision functions for Dempster-Shafer theory.

Includes:
- Pignistic transform (BetP)
- Belief and Plausibility
- Decision rules
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from .dempster import BPA


def pignistic_transform(bpa: BPA, theta: FrozenSet[str]) -> Dict[str, float]:
    """
    Compute the pignistic probability distribution (BetP).

    For each singleton x in theta:
        BetP(x) = sum over A containing x of: m(A) / |A|

    This distributes mass on compound elements equally among their members.

    Args:
        bpa: Basic probability assignment
        theta: Frame of discernment (all singletons)

    Returns:
        Dictionary mapping each element in theta to its BetP
    """
    betp: Dict[str, float] = {x: 0.0 for x in theta}

    for focal, mass in bpa.items():
        cardinality = len(focal)
        if cardinality == 0:
            continue  # Skip empty set (should not occur in valid BPA)

        share = mass / cardinality
        for x in focal:
            if x in betp:
                betp[x] += share

    return betp


def belief_plausibility(bpa: BPA, hypothesis: FrozenSet[str]) -> Tuple[float, float]:
    """
    Compute Belief and Plausibility for a hypothesis.

    Bel(A) = sum of m(B) for all B subset of A
    Pl(A) = sum of m(B) for all B intersecting A

    Args:
        bpa: Basic probability assignment
        hypothesis: The hypothesis to evaluate

    Returns:
        Tuple of (Belief, Plausibility)
    """
    bel = 0.0
    pl = 0.0

    for focal, mass in bpa.items():
        # Belief: focal is subset of hypothesis
        if focal <= hypothesis:
            bel += mass

        # Plausibility: focal intersects hypothesis
        if focal & hypothesis:
            pl += mass

    return bel, pl


def decide(betp: Dict[str, float]) -> str:
    """
    Make a decision based on pignistic probabilities.

    Returns the singleton with maximum BetP (argmax).
    Ties are broken alphabetically for reproducibility.
    """
    if not betp:
        raise ValueError("Cannot decide from empty BetP")

    max_prob = max(betp.values())
    candidates = [x for x, p in betp.items() if p == max_prob]
    return min(candidates)  # Alphabetical tie-break


def set_valued_decision(bpa: BPA, theta: FrozenSet[str],
                        threshold: float = 0.0) -> FrozenSet[str]:
    """
    Make a set-valued decision including all elements with Pl > threshold.

    This returns the set of elements that cannot be ruled out.

    Args:
        bpa: Basic probability assignment
        theta: Frame of discernment
        threshold: Minimum plausibility to include (default 0)

    Returns:
        Set of elements with Pl > threshold
    """
    result = set()

    for x in theta:
        _, pl = belief_plausibility(bpa, frozenset({x}))
        if pl > threshold:
            result.add(x)

    return frozenset(result)
