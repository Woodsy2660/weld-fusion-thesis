"""Phase 2: Per-cell fusion.

Runs per cell, independently (embarrassingly parallel):
    - Calibrated posterior -> mass over modality's declared focal elements
    - Shafer discounting by alpha_m
    - Subset selection (7-way ablation as column selection)
    - Dempster combination, returning conflict K
    - Murphy averaging
    - Pignistic transform -> predicted label

Writes the bpa and fusion columns to the CellTable.

This phase is fundamentally per-cell because each cell's fusion is
independent of all other cells. No spatial smoothing or CRF is applied
(that would break the independence assumption behind the McNemar test).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from weldfusion.bpa.construct import construct_bpa
from weldfusion.bpa.discount import shafer_discount
from weldfusion.celltable import CellTable
from weldfusion.frame import class_order, theta
from weldfusion.fusion.ablation import ablation_subsets
from weldfusion.fusion.decision import pignistic, decide
from weldfusion.fusion.dempster import ConflictError
from weldfusion.fusion.murphy import murphy_average
from weldfusion.types import BPA


def posterior_array_to_dict(posterior: np.ndarray) -> dict[str, float]:
    """Convert posterior array to dict keyed by class name."""
    classes = class_order()
    if len(posterior) != len(classes):
        raise ValueError(
            f"Posterior length {len(posterior)} != num classes {len(classes)}"
        )
    return {cls: float(p) for cls, p in zip(classes, posterior)}


def build_bpa_from_posterior(
    posterior: np.ndarray | None,
    modality: str,
    alpha: float,
) -> BPA | None:
    """Build discounted BPA from calibrated posterior array."""
    if posterior is None:
        return None

    posterior_dict = posterior_array_to_dict(posterior)
    bpa = construct_bpa(posterior_dict, modality)
    return shafer_discount(bpa, alpha)


def build_vision_bpa(
    posterior: np.ndarray | None,
    alpha: float,
) -> BPA | None:
    """Build discounted vision BPA from calibrated posterior."""
    return build_bpa_from_posterior(posterior, "vision", alpha)


def build_geometry_bpa(
    exceedance: float | None,
    alpha: float,
) -> BPA | None:
    """Build discounted geometry BPA from ISO 5817 exceedance probability.

    Geometry only has two focal elements (read from configs/frame.yaml):
        G1={U}: non-conforming (undercut)
        G2={N,P,L}: conforming (says nothing about internal defects)

    The exceedance probability p maps directly:
        m(G1) = p
        m(G2) = 1 - p

    NOTE: G2 was {N,P,L,C} in the 5-class frame. It is now {N,P,L} in the
    4-class experimental frame because crack surrogates could not be fabricated.
    """
    if exceedance is None:
        return None

    from weldfusion.frame import geometry_focal_elements

    p = float(exceedance)
    p = max(0.0, min(1.0, p))

    # Read focal elements from frame config
    focal_elements = geometry_focal_elements()
    g1_set = focal_elements["G1"]
    g2_set = focal_elements["G2"]

    masses = {
        g1_set: p,
        g2_set: 1.0 - p,
    }

    bpa = BPA(masses=masses, modality="geometry")
    bpa.validate()
    return shafer_discount(bpa, alpha)


def build_ut_bpa(
    posterior: np.ndarray | None,
    alpha: float,
) -> BPA | None:
    """Build discounted UT BPA from calibrated posterior."""
    return build_bpa_from_posterior(posterior, "ut", alpha)


def fuse_cell(
    vision_bpa: BPA | None,
    geometry_bpa: BPA | None,
    ut_bpa: BPA | None,
    subset: tuple[str, ...],
) -> tuple[BPA | None, float | None, dict[str, float] | None, str | None]:
    """Fuse BPAs for a single cell using the specified modality subset.

    The ablation is implemented as column selection: only BPAs from
    modalities in the subset are combined.

    Returns (fused_bpa, conflict_K, betp_dict, predicted_label).
    """
    # Collect BPAs for the requested subset
    bpas: list[BPA] = []
    if "vision" in subset and vision_bpa is not None:
        bpas.append(vision_bpa)
    if "geometry" in subset and geometry_bpa is not None:
        bpas.append(geometry_bpa)
    if "ut" in subset and ut_bpa is not None:
        bpas.append(ut_bpa)

    if not bpas:
        return None, None, None, None

    # Murphy averaging: average masses, then combine average with itself n-1 times
    try:
        fused, total_k = murphy_average(bpas)
    except ConflictError:
        # Total conflict - cannot fuse
        return None, 1.0, None, None

    # Pignistic transform and decision
    label, betp = decide(fused)

    return fused, total_k, betp, label


def process_cell(
    row_idx: int,
    table: CellTable,
    modality_alphas: dict[str, float],
    subset: tuple[str, ...],
) -> dict:
    """Process a single cell through Phase 2.

    Returns dict with all Phase 2 outputs for the cell.
    """
    # Get calibrated outputs from Phase 1
    vision_posterior = table["vision_posterior"][row_idx] if "vision_posterior" in table else None
    ut_posterior = table["ut_posterior"][row_idx] if "ut_posterior" in table else None
    geometry_exceedance = table["geometry_exceedance"][row_idx] if "geometry_exceedance" in table else None

    # Get alphas
    alpha_v = modality_alphas.get("vision", 0.0)
    alpha_g = modality_alphas.get("geometry", 0.0)
    alpha_u = modality_alphas.get("ut", 0.0)

    # Build per-modality BPAs
    vision_bpa = build_vision_bpa(vision_posterior, alpha_v)
    geometry_bpa = build_geometry_bpa(geometry_exceedance, alpha_g)
    ut_bpa = build_ut_bpa(ut_posterior, alpha_u)

    # Fuse
    fused_bpa, conflict_K, betp, predicted_label = fuse_cell(
        vision_bpa, geometry_bpa, ut_bpa, subset
    )

    return {
        "vision_bpa": vision_bpa,
        "geometry_bpa": geometry_bpa,
        "ut_bpa": ut_bpa,
        "fused_bpa": fused_bpa,
        "conflict_K": conflict_K,
        "betp": betp,
        "predicted_label": predicted_label,
    }


def run_phase2(
    table: CellTable,
    modality_alphas: dict[str, float],
    subset: tuple[str, ...] = ("vision", "geometry", "ut"),
) -> CellTable:
    """Run Phase 2 on all cells.

    Processes each cell independently (embarrassingly parallel).
    Adds BPA and fusion columns to the table.
    """
    n = len(table)

    # Pre-allocate output columns
    vision_bpas = []
    geometry_bpas = []
    ut_bpas = []
    fused_bpas = []
    conflict_Ks = []
    betps = []
    predicted_labels = []

    for i in range(n):
        result = process_cell(i, table, modality_alphas, subset)
        vision_bpas.append(result["vision_bpa"])
        geometry_bpas.append(result["geometry_bpa"])
        ut_bpas.append(result["ut_bpa"])
        fused_bpas.append(result["fused_bpa"])
        conflict_Ks.append(result["conflict_K"])
        betps.append(result["betp"])
        predicted_labels.append(result["predicted_label"])

    # Add columns
    return (
        table
        .add_column("vision_bpa", vision_bpas)
        .add_column("geometry_bpa", geometry_bpas)
        .add_column("ut_bpa", ut_bpas)
        .add_column("fused_bpa", fused_bpas)
        .add_column("conflict_K", conflict_Ks)
        .add_column("betp", betps)
        .add_column("predicted_label", predicted_labels)
    )


def run_all_ablations(
    table: CellTable,
    modality_alphas: dict[str, float],
) -> dict[str, CellTable]:
    """Run Phase 2 for all 7 ablation subsets.

    Returns dict mapping subset name to CellTable with fusion results.

    Ablation subsets (from configs/fusion.yaml):
        vision, geometry, ut, vision+geometry, vision+ut, geometry+ut, vision+geometry+ut
    """
    results = {}
    for subset in ablation_subsets():
        # Name by joining modality names
        subset_name = "+".join(subset)

        # Copy table for this ablation run
        ablation_table = CellTable()
        ablation_table._data = {k: list(v) for k, v in table._data.items()}
        ablation_table._n_rows = len(table)

        # Run Phase 2 with this subset
        results[subset_name] = run_phase2(ablation_table, modality_alphas, subset)

    return results
