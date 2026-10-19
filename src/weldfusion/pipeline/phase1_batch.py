"""Phase 1: Batch inference and calibration.

Runs over ALL cells at once, per CV fold:
    - Per-modality model inference (batched for GPU efficiency)
    - Fit temperature scaling on inner validation fold ONLY
    - Fit isotonic/Platt for UT on inner validation fold ONLY
    - Estimate reliability alpha_m per modality on validation folds ONLY

Writes the model and calibration columns to the CellTable.
MUST NOT see test-fold labels.

This phase is fundamentally batch-oriented because:
    1. GPU inference is efficient only in batches
    2. Temperature scaling fits one scalar T across a SET of predictions
    3. Reliability alpha_m is estimated across a FOLD, not per cell
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from weldfusion.calibration.reliability import reliability_alpha
from weldfusion.calibration.temperature import fit_temperature, apply_temperature
from weldfusion.calibration.isotonic import fit_isotonic, apply_isotonic
from weldfusion.celltable import CellTable
from weldfusion.frame import class_order


def run_vision_inference(
    table: CellTable,
    *,
    model_path: str | None = None,
) -> CellTable:
    """Run vision model inference on all cells in batch.

    Adds vision_logits column.

    This is a skeleton: actual inference requires trained weights and
    registered image patches.
    """
    n = len(table)
    n_classes = len(class_order())

    # Skeleton: would load model and run batched inference
    # For now, fill with None (no vision data available)
    logits = [None] * n

    if "vision_logits" in table:
        # Already computed, skip
        return table

    return table.add_column("vision_logits", logits)


def run_geometry_inference(
    table: CellTable,
    *,
    profile_data: dict | None = None,
) -> CellTable:
    """Extract geometry measurements for all cells.

    Adds geometry_measurements column containing per-cell dicts with:
        undercut_depth_mm, excess_metal_mm, misalignment_mm, etc.

    This is a skeleton: actual extraction requires depth maps.
    """
    n = len(table)

    # Skeleton: would extract from depth profiles
    measurements = [None] * n

    if "geometry_measurements" in table:
        return table

    return table.add_column("geometry_measurements", measurements)


def run_ut_inference(
    table: CellTable,
    *,
    ascan_data: dict | None = None,
    model_path: str | None = None,
) -> CellTable:
    """Run UT feature extraction and model inference for all cells.

    Adds ut_features column containing per-cell feature dicts.

    This is a skeleton: actual extraction requires A-scan data.
    """
    n = len(table)

    # Skeleton: would extract features from A-scans and run classifier
    features = [None] * n

    if "ut_features" in table:
        return table

    return table.add_column("ut_features", features)


def calibrate_vision(
    table: CellTable,
    train_indices: Sequence[int],
    train_labels: Sequence[int],
    *,
    temperature: float | None = None,
) -> tuple[CellTable, float]:
    """Apply temperature scaling to vision logits.

    If temperature is None, fits on the provided training indices.
    Returns (table with vision_posterior, fitted temperature).

    MUST be called with inner validation fold indices, NOT test fold.
    """
    n = len(table)
    posteriors = [None] * n

    if "vision_logits" not in table:
        raise ValueError("vision_logits column required before calibration")

    # Collect logits for fitting
    logits_for_fit = []
    labels_for_fit = []
    for idx, label in zip(train_indices, train_labels):
        logits = table["vision_logits"][idx]
        if logits is not None:
            logits_for_fit.append(logits)
            labels_for_fit.append(label)

    # Fit temperature if not provided
    if temperature is None and logits_for_fit:
        temperature = fit_temperature(
            np.array(logits_for_fit),
            np.array(labels_for_fit),
        )
    elif temperature is None:
        temperature = 1.0  # default if no data

    # Apply to all cells
    for i in range(n):
        logits = table["vision_logits"][i]
        if logits is not None:
            posteriors[i] = apply_temperature(logits, temperature)

    return table.add_column("vision_posterior", posteriors), temperature


def calibrate_ut(
    table: CellTable,
    train_indices: Sequence[int],
    train_labels: Sequence[int],
    *,
    isotonic_model: object | None = None,
) -> tuple[CellTable, object]:
    """Apply isotonic/Platt calibration to UT classifier outputs.

    If isotonic_model is None, fits on the provided training indices.
    Returns (table with ut_posterior, fitted model).

    MUST be called with inner validation fold indices, NOT test fold.
    """
    n = len(table)
    posteriors = [None] * n

    # Skeleton: would fit isotonic regression on UT classifier scores
    # For now, pass through None values

    if "ut_features" not in table:
        raise ValueError("ut_features column required before calibration")

    # Would fit and apply isotonic calibration here
    # Placeholder: return unchanged
    if isotonic_model is None:
        isotonic_model = object()  # placeholder

    return table.add_column("ut_posterior", posteriors), isotonic_model


def compute_geometry_exceedance(
    table: CellTable,
) -> CellTable:
    """Compute ISO 5817 exceedance probabilities from geometry measurements.

    Adds geometry_exceedance column (P(non-conforming) for undercut).
    """
    n = len(table)
    exceedances = [None] * n

    if "geometry_measurements" not in table:
        raise ValueError("geometry_measurements column required")

    # Skeleton: would compute Phi((d_measured - d_limit) / u_total)
    # For now, return None values

    return table.add_column("geometry_exceedance", exceedances)


def estimate_modality_alphas(
    table: CellTable,
    val_indices: Sequence[int],
    val_labels: Sequence[str],
) -> dict[str, float]:
    """Estimate Shafer discount alpha_m per modality on validation fold.

    alpha_m = 1 - balanced_accuracy (estimated on validation fold only).

    Returns dict mapping modality name to alpha value.
    MUST NOT be called on test fold.
    """
    # Skeleton: would compute balanced accuracy per modality and derive alpha
    # For now, return default discount factors

    alphas = {
        "vision": 0.15,    # placeholder: 85% balanced accuracy
        "geometry": 0.20,  # placeholder: 80% balanced accuracy
        "ut": 0.15,        # placeholder: 85% balanced accuracy
    }

    return alphas


def run_phase1(
    table: CellTable,
    *,
    train_indices: Sequence[int] | None = None,
    train_labels: Sequence[int] | None = None,
    val_indices: Sequence[int] | None = None,
    val_labels: Sequence[str] | None = None,
) -> tuple[CellTable, dict]:
    """Run the full Phase 1 pipeline.

    Returns (updated table, calibration_params dict).
    """
    # Model inference (batched)
    table = run_vision_inference(table)
    table = run_geometry_inference(table)
    table = run_ut_inference(table)

    calibration_params = {}

    # Calibration (requires validation labels)
    if train_indices is not None and train_labels is not None:
        table, temp = calibrate_vision(table, train_indices, train_labels)
        calibration_params["vision_temperature"] = temp

        table, iso_model = calibrate_ut(table, train_indices, train_labels)
        calibration_params["ut_isotonic_model"] = iso_model

    # Geometry exceedance (no calibration needed, pure physics)
    table = compute_geometry_exceedance(table)

    # Reliability alphas (requires validation labels)
    if val_indices is not None and val_labels is not None:
        alphas = estimate_modality_alphas(table, val_indices, val_labels)
        calibration_params["modality_alphas"] = alphas

    return table, calibration_params
