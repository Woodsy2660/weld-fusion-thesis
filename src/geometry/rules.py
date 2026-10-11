"""
ISO 5817 exceedance-based BPA construction for geometry modality.

The geometry modality resolves exactly one thing: whether the measured
undercut exceeds the ISO 5817 limit. It outputs a BPA over two focal elements:

    G1: {U}        measured undercut exceeds the limit (non-conforming)
    G2: {N,P,L,C}  profile conforms — says nothing about internal defects

Reinforcement and misalignment are measured and reported as validation outputs,
but they do NOT drive m({U}). Only undercut does.

The exceedance probability accounts for measurement uncertainty:

    P(non-conformance) = Φ((d̂ - d_limit) / u_total)

where Φ is the standard normal CDF. This is the probability that the true
undercut depth exceeds the limit, given a measurement with Gaussian uncertainty.

No ad-hoc ignorance floor is applied here. Ignorance enters via Shafer
discounting downstream, parameterised by measured modality reliability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import norm


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "geometry.yaml"


@dataclass
class UncertaintyBudget:
    """Four-term uncertainty budget for depth measurements.

    All values in mm.
    """
    u_noise: float    # From split-half estimation, not naive sqrt(N)
    u_quant: float    # Quantization: 0.1mm / sqrt(12)
    u_cal: float      # Plane-fit residual on flat reference
    u_align: float    # Fixture repeatability from repeat-placement

    @property
    def u_total(self) -> float:
        """Combined uncertainty (root sum of squares)."""
        return math.sqrt(
            self.u_noise**2 + self.u_quant**2 + self.u_cal**2 + self.u_align**2
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "u_noise_mm": self.u_noise,
            "u_quant_mm": self.u_quant,
            "u_cal_mm": self.u_cal,
            "u_align_mm": self.u_align,
            "u_total_mm": self.u_total,
        }


@dataclass
class GeometryMeasurement:
    """Measured geometric features for a single cell.

    undercut_mm is the governing measurement for BPA construction.
    reinforcement_mm and misalignment_mm are validation outputs only.
    """
    cell_id: str
    face: str                    # "top" or "root"
    toe: str                     # "left" or "right" (relative to weld axis)

    undercut_mm: float           # Depth below parent metal datum (governing)
    reinforcement_mm: float      # Height above parent metal datum
    misalignment_mm: float       # Hi-lo between plate halves
    bead_width_mm: float         # Toe-to-toe distance

    n_valid_pixels: int          # Valid depth pixels in measurement region
    valid_fraction: float        # Fraction of expected pixels that are valid
    tilt_angle_deg: float        # Plane-fit tilt for correction logging

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "face": self.face,
            "toe": self.toe,
            "undercut_mm": self.undercut_mm,
            "reinforcement_mm": self.reinforcement_mm,
            "misalignment_mm": self.misalignment_mm,
            "bead_width_mm": self.bead_width_mm,
            "n_valid_pixels": self.n_valid_pixels,
            "valid_fraction": self.valid_fraction,
            "tilt_angle_deg": self.tilt_angle_deg,
        }


@dataclass
class GeometryBPA:
    """BPA for geometry modality over focal elements {U} and {N,P,L,C}.

    If insufficient valid pixels, this is a vacuous BPA (m(Θ) = 1).
    """
    cell_id: str
    m_U: float                   # Mass on {U} — non-conforming
    m_NPLC: float                # Mass on {N,P,L,C} — conforming
    is_vacuous: bool             # True if insufficient data

    # Supporting data
    governing_undercut_mm: float # The undercut value that drove this BPA
    governing_face: str          # Which face had the governing undercut
    governing_toe: str           # Which toe had the governing undercut
    exceedance_prob: float       # P(true undercut > limit)
    u_total_mm: float            # Combined uncertainty used

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "m_U": self.m_U,
            "m_NPLC": self.m_NPLC,
            "is_vacuous": self.is_vacuous,
            "governing_undercut_mm": self.governing_undercut_mm,
            "governing_face": self.governing_face,
            "governing_toe": self.governing_toe,
            "exceedance_prob": self.exceedance_prob,
            "u_total_mm": self.u_total_mm,
        }

    def as_bpa_dict(self) -> dict[frozenset[str], float]:
        """Return as a proper BPA dict for fusion."""
        if self.is_vacuous:
            return {frozenset({"N", "U", "P", "L", "C"}): 1.0}
        return {
            frozenset({"U"}): self.m_U,
            frozenset({"N", "P", "L", "C"}): self.m_NPLC,
        }


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load geometry configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def exceedance_probability(
    measured_mm: float,
    limit_mm: float,
    u_total_mm: float,
) -> float:
    """
    Probability that the true value exceeds the limit, given measurement uncertainty.

    P(true > limit) = Φ((measured - limit) / u_total)

    where Φ is the standard normal CDF.

    Args:
        measured_mm: Measured undercut depth.
        limit_mm: ISO 5817 acceptance limit.
        u_total_mm: Combined measurement uncertainty.

    Returns:
        Exceedance probability in [0, 1].
    """
    if u_total_mm <= 0:
        # No uncertainty — deterministic comparison
        return 1.0 if measured_mm > limit_mm else 0.0

    z = (measured_mm - limit_mm) / u_total_mm
    return float(norm.cdf(z))


def build_cell_bpa(
    measurements: list[GeometryMeasurement],
    uncertainty: UncertaintyBudget,
    config: dict | None = None,
) -> GeometryBPA:
    """
    Build a BPA for one cell from measurements across faces and toes.

    The governing undercut is the maximum across both faces (top, root) and
    both toes (left, right). ISO 5817 judges by the worst case.

    Args:
        measurements: List of GeometryMeasurement for this cell (up to 4:
            top-left, top-right, root-left, root-right).
        uncertainty: Four-term uncertainty budget.
        config: Geometry config dict. Loaded from file if None.

    Returns:
        GeometryBPA for fusion.
    """
    config = config or load_config()
    limit_mm = config["acceptance_criteria"]["undercut_limit_mm"]
    min_valid = config["profile_extraction"]["min_valid_fraction"]

    # Check for sufficient data
    valid_measurements = [m for m in measurements if m.valid_fraction >= min_valid]
    if not valid_measurements:
        # Vacuous BPA — insufficient data
        return GeometryBPA(
            cell_id=measurements[0].cell_id if measurements else "unknown",
            m_U=0.0,
            m_NPLC=0.0,
            is_vacuous=True,
            governing_undercut_mm=float("nan"),
            governing_face="",
            governing_toe="",
            exceedance_prob=float("nan"),
            u_total_mm=uncertainty.u_total,
        )

    # Find governing undercut (maximum across all valid measurements)
    governing = max(valid_measurements, key=lambda m: m.undercut_mm)

    # Compute exceedance probability
    p_exceed = exceedance_probability(
        measured_mm=governing.undercut_mm,
        limit_mm=limit_mm,
        u_total_mm=uncertainty.u_total,
    )

    # Build BPA: m({U}) = P(exceed), m({N,P,L,C}) = 1 - P(exceed)
    # No ignorance floor here — that comes from Shafer discounting downstream.
    return GeometryBPA(
        cell_id=governing.cell_id,
        m_U=p_exceed,
        m_NPLC=1.0 - p_exceed,
        is_vacuous=False,
        governing_undercut_mm=governing.undercut_mm,
        governing_face=governing.face,
        governing_toe=governing.toe,
        exceedance_prob=p_exceed,
        u_total_mm=uncertainty.u_total,
    )


def validate_geometry_bpa(bpa: GeometryBPA) -> bool:
    """
    Validate that a geometry BPA satisfies the required invariants.

    - Masses sum to 1 (or BPA is vacuous with m(Θ) = 1)
    - No mass on singletons other than {U}
    - No mass on sets outside the declared focal elements
    """
    if bpa.is_vacuous:
        return True

    total = bpa.m_U + bpa.m_NPLC
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"BPA masses sum to {total}, expected 1.0")

    if not (0.0 <= bpa.m_U <= 1.0):
        raise ValueError(f"m({{U}}) = {bpa.m_U} outside [0, 1]")

    if not (0.0 <= bpa.m_NPLC <= 1.0):
        raise ValueError(f"m({{N,P,L,C}}) = {bpa.m_NPLC} outside [0, 1]")

    return True


# ---------------------------------------------------------------------------
# Validation outputs — not used for BPA, but reported for completeness
# ---------------------------------------------------------------------------

@dataclass
class ValidationOutput:
    """ISO 5817 conformance assessment for validation (not BPA construction)."""
    cell_id: str

    # Reinforcement assessment
    reinforcement_mm: float
    reinforcement_limit_mm: float
    reinforcement_conforms: bool

    # Misalignment assessment
    misalignment_mm: float
    misalignment_limit_mm: float
    misalignment_conforms: bool


def assess_validation_outputs(
    measurements: list[GeometryMeasurement],
    config: dict | None = None,
) -> ValidationOutput:
    """
    Assess reinforcement and misalignment conformance for validation.

    These do NOT affect m({U}). They are reported separately for the
    uncertainty validation and thesis discussion.
    """
    config = config or load_config()
    criteria = config["acceptance_criteria"]
    t = criteria["plate_thickness_mm"]

    # Use max values across all measurements
    max_reinf = max((m.reinforcement_mm for m in measurements), default=0.0)
    max_misal = max((m.misalignment_mm for m in measurements), default=0.0)
    max_width = max((m.bead_width_mm for m in measurements), default=16.0)

    # Reinforcement limit: h <= 1 + 0.1b, max 5mm
    reinf_limit = min(criteria["reinforcement_max_mm"], 1.0 + 0.1 * max_width)

    # Misalignment limit: h <= 0.1t + 0.2mm
    misal_limit = min(criteria["misalignment_max_mm"], 0.1 * t + 0.2)

    return ValidationOutput(
        cell_id=measurements[0].cell_id if measurements else "unknown",
        reinforcement_mm=max_reinf,
        reinforcement_limit_mm=reinf_limit,
        reinforcement_conforms=max_reinf <= reinf_limit,
        misalignment_mm=max_misal,
        misalignment_limit_mm=misal_limit,
        misalignment_conforms=max_misal <= misal_limit,
    )
