"""
Dempster's rule of combination with conflict mass K.

This is the canonical implementation that RETURNS K.
The older src/fusion/dempster_shafer.py is deprecated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet


BPA = Dict[FrozenSet[str], float]


@dataclass
class DempsterResult:
    """Result of Dempster's combination including conflict mass."""
    bpa: BPA
    K: float  # Conflict mass before normalization


def _is_vacuous(bpa: BPA) -> bool:
    """Check if BPA is vacuous (all mass on Theta)."""
    if len(bpa) == 1:
        key = next(iter(bpa.keys()))
        return bpa[key] >= 1.0 - 1e-12
    return False


def dempster_combine(bpa1: BPA, bpa2: BPA, theta: FrozenSet[str] | None = None) -> DempsterResult:
    """
    Combine two BPAs using Dempster's rule of combination.

    Returns the combined BPA and the conflict mass K.

    Args:
        bpa1: First basic probability assignment
        bpa2: Second basic probability assignment
        theta: The frame of discernment (required if not inferable)

    Returns:
        DempsterResult with combined BPA and conflict K

    Raises:
        ValueError: If K = 1 (total conflict)
    """
    # Handle None inputs as vacuous
    if bpa1 is None or _is_vacuous(bpa1):
        # Vacuous acts as identity
        return DempsterResult(bpa=dict(bpa2) if bpa2 else {}, K=0.0)
    if bpa2 is None or _is_vacuous(bpa2):
        return DempsterResult(bpa=dict(bpa1), K=0.0)

    # Infer theta from largest focal element
    if theta is None:
        all_elements = set()
        for focal in list(bpa1.keys()) + list(bpa2.keys()):
            all_elements.update(focal)
        theta = frozenset(all_elements)

    # Compute unnormalized combination
    combined: Dict[FrozenSet[str], float] = {}
    K = 0.0  # Conflict mass

    for A, m1 in bpa1.items():
        for B, m2 in bpa2.items():
            intersection = A & B
            mass = m1 * m2

            if not intersection:  # Empty intersection = conflict
                K += mass
            else:
                combined[intersection] = combined.get(intersection, 0.0) + mass

    # Check for total conflict
    if K >= 1.0 - 1e-12:
        raise ValueError(f"Total conflict: K = {K}")

    # Normalize
    normalizer = 1.0 - K
    normalized: BPA = {}
    for focal, mass in combined.items():
        normalized[focal] = mass / normalizer

    return DempsterResult(bpa=normalized, K=K)


def dempster_combine_multi(bpas: list[BPA], theta: FrozenSet[str] | None = None) -> DempsterResult:
    """
    Combine multiple BPAs using Dempster's rule sequentially.

    Returns the combined BPA and the total conflict K.
    Note: K is accumulated across all pairwise combinations.
    """
    if not bpas:
        raise ValueError("Cannot combine empty list of BPAs")

    # Filter out None and vacuous BPAs
    non_vacuous = [b for b in bpas if b is not None and not _is_vacuous(b)]

    if not non_vacuous:
        # All vacuous - return vacuous
        if theta:
            return DempsterResult(bpa={theta: 1.0}, K=0.0)
        raise ValueError("All BPAs vacuous and theta not specified")

    if len(non_vacuous) == 1:
        return DempsterResult(bpa=dict(non_vacuous[0]), K=0.0)

    # Combine sequentially, tracking total K
    result = DempsterResult(bpa=dict(non_vacuous[0]), K=0.0)
    total_K = 0.0

    for bpa in non_vacuous[1:]:
        result = dempster_combine(result.bpa, bpa, theta)
        # Accumulate conflict (approximate - true K is more complex)
        total_K = total_K + result.K - total_K * result.K

    return DempsterResult(bpa=result.bpa, K=total_K)
