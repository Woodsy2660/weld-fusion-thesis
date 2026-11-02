"""
Ablation study driver for fusion experiments.

Runs all 2^n - 1 non-empty subsets of modalities.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional

from .dempster import BPA, DempsterResult, dempster_combine_multi
from .murphy import murphy_average
from .decision import pignistic_transform, belief_plausibility, decide


@dataclass
class CellResult:
    """Fusion result for a single cell."""
    cell_id: str
    modalities: FrozenSet[str]
    method: str  # "dempster" or "murphy"

    # Combined BPA
    bpa: BPA
    K: float  # Conflict mass

    # Pignistic
    betp: Dict[str, float]
    decision: str

    # Intervals per singleton
    bel_pl: Dict[str, tuple]  # {element: (Bel, Pl)}

    # Metadata
    all_non_vacuous: bool  # True if all modalities in subset are non-vacuous


@dataclass
class AblationResult:
    """Results for one ablation condition."""
    modalities: FrozenSet[str]
    method: str

    cells: List[CellResult]
    n_all_non_vacuous: int  # Count where all modalities are non-vacuous

    # Aggregate metrics (computed after fusion)
    l_count_under_vg: Optional[int] = None  # {L} decisions when subset is {V,G}


def run_ablation(
    cell_id: str,
    bpas: Dict[str, BPA],  # {modality: BPA}
    theta: FrozenSet[str],
    modality_subset: FrozenSet[str],
    method: str = "dempster",
) -> CellResult:
    """
    Run fusion for a single cell with a given modality subset.

    Args:
        cell_id: Cell identifier
        bpas: Dictionary of BPAs keyed by modality name
        theta: Frame of discernment
        modality_subset: Which modalities to include
        method: "dempster" or "murphy"

    Returns:
        CellResult with all fusion outputs
    """
    # Collect BPAs for the subset
    subset_bpas = []
    non_vacuous_count = 0

    for mod in modality_subset:
        bpa = bpas.get(mod)
        if bpa is not None:
            # Check if vacuous (all mass on theta)
            is_vacuous = len(bpa) == 1 and theta in bpa and bpa[theta] >= 1.0 - 1e-12
            if not is_vacuous:
                non_vacuous_count += 1
            subset_bpas.append(bpa)
        else:
            # Missing modality = vacuous
            subset_bpas.append({theta: 1.0})

    all_non_vacuous = (non_vacuous_count == len(modality_subset))

    # Combine
    if method == "dempster":
        result = dempster_combine_multi(subset_bpas, theta)
    elif method == "murphy":
        result = murphy_average(subset_bpas, theta)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Pignistic transform
    betp = pignistic_transform(result.bpa, theta)
    decision = decide(betp)

    # Belief and Plausibility per singleton
    bel_pl = {}
    for x in theta:
        bel, pl = belief_plausibility(result.bpa, frozenset({x}))
        bel_pl[x] = (bel, pl)

    return CellResult(
        cell_id=cell_id,
        modalities=modality_subset,
        method=method,
        bpa=result.bpa,
        K=result.K,
        betp=betp,
        decision=decision,
        bel_pl=bel_pl,
        all_non_vacuous=all_non_vacuous,
    )


def generate_subsets(modalities: List[str]) -> List[FrozenSet[str]]:
    """Generate all non-empty subsets of modalities."""
    subsets = []
    for r in range(1, len(modalities) + 1):
        for combo in combinations(modalities, r):
            subsets.append(frozenset(combo))
    return subsets
