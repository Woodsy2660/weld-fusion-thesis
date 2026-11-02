"""
Murphy's averaging rule for combining multiple BPAs.

Murphy (2000) proposes averaging BPAs before applying Dempster's rule
to reduce the influence of conflicting evidence.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List

from .dempster import BPA, DempsterResult, dempster_combine


def average_bpas(bpas: List[BPA]) -> BPA:
    """
    Compute the arithmetic mean of multiple BPAs.

    All BPAs must be over the same frame, though focal elements may differ.
    """
    if not bpas:
        raise ValueError("Cannot average empty list of BPAs")

    n = len(bpas)
    averaged: Dict[FrozenSet[str], float] = {}

    for bpa in bpas:
        if bpa is None:
            continue
        for focal, mass in bpa.items():
            averaged[focal] = averaged.get(focal, 0.0) + mass / n

    return averaged


def murphy_average(bpas: List[BPA], theta: FrozenSet[str] | None = None) -> DempsterResult:
    """
    Combine BPAs using Murphy's averaging rule.

    Process:
    1. Average all BPAs to get a consensus BPA
    2. Apply Dempster's rule (n-1) times with the average

    Args:
        bpas: List of BPAs to combine
        theta: Frame of discernment

    Returns:
        DempsterResult with combined BPA and accumulated conflict K
    """
    if not bpas:
        raise ValueError("Cannot combine empty list of BPAs")

    # Filter out None BPAs
    valid_bpas = [b for b in bpas if b is not None]

    if not valid_bpas:
        if theta:
            return DempsterResult(bpa={theta: 1.0}, K=0.0)
        raise ValueError("All BPAs are None and theta not specified")

    if len(valid_bpas) == 1:
        return DempsterResult(bpa=dict(valid_bpas[0]), K=0.0)

    # Step 1: Average all BPAs
    avg_bpa = average_bpas(valid_bpas)

    # Step 2: Combine average with itself (n-1) times
    n = len(valid_bpas)
    result = DempsterResult(bpa=dict(avg_bpa), K=0.0)
    total_K = 0.0

    for _ in range(n - 1):
        result = dempster_combine(result.bpa, avg_bpa, theta)
        total_K = total_K + result.K - total_K * result.K

    return DempsterResult(bpa=result.bpa, K=total_K)
