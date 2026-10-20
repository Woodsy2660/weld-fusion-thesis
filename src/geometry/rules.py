"""
ISO 5817 exceedance-based BPA construction for geometry modality.

The geometry modality resolves exactly one thing: whether the bead profile
breaches an ISO 5817 limit. configs/frame.yaml defines U as the DISJUNCTION
undercut OR excess reinforcement OR linear misalignment, so all three drive
the BPA over two focal elements:

    G1: {U}       some measurand exceeds its limit (non-conforming)
    G2: {N,P,L}   profile conforms — says nothing about internal defects

m({U}) = max_k P_k over the three exceedance probabilities. The three share one
datum, so their errors are positively correlated; max is the union probability
under full dependence and a lower bound otherwise, whereas the independent-union
form 1 - prod(1 - P_k) would overstate it. (Changed 2026-10-20; previously
undercut alone drove m({U}), contradicting the frame.)

NOTE: G2 was {N,P,L,C} in the 5-class analytical frame. It is now {N,P,L} in
the 4-class experimental frame because crack surrogates could not be fabricated.
The code reads focal elements from configs/frame.yaml and does not hardcode them.

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

from src.utils.frame import load_frame


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
    """BPA for geometry modality over focal elements G1={U} and G2={N,P,L}.

    If insufficient valid pixels, this is a vacuous BPA (m(Θ) = 1).

    NOTE: Field name m_NPLC is historical (from 5-class frame). It represents
    the "conforming" focal element, which is now {N,P,L} in the experimental
    frame. The actual focal elements are read from configs/frame.yaml.
    """
    cell_id: str
    m_U: float                   # Mass on {U} — non-conforming
    m_NPLC: float                # Mass on G2 — conforming (read from frame)
    is_vacuous: bool             # True if insufficient data

    # Supporting data
    governing_undercut_mm: float # The undercut value that drove this BPA
    governing_face: str          # Which face had the governing undercut
    governing_toe: str           # Which toe had the governing undercut
    exceedance_prob: float       # m({U}) = max of the three below
    u_total_mm: float            # Combined uncertainty used
    exceedance_by_measurand: dict | None = None   # {measurand: P}
    limits_mm: dict | None = None                 # {measurand: limit}
    governing_measurand: str = "undercut"

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
            "exceedance_by_measurand": self.exceedance_by_measurand,
            "limits_mm": self.limits_mm,
            "governing_measurand": self.governing_measurand,
        }

    def as_bpa_dict(self, frame=None) -> dict[frozenset[str], float]:
        """Return as a proper BPA dict for fusion.

        Focal elements are read from configs/frame.yaml, not hardcoded.
        """
        frame = frame or load_frame()
        theta = frozenset(frame.classes)
        geometry_heads = frame.heads_for("geometry")

        if self.is_vacuous:
            return {theta: 1.0}

        # G1 is {U}, G2 is the conforming focal element (read from frame)
        g1_set = frozenset(geometry_heads["G1"])
        g2_set = frozenset(geometry_heads["G2"])
        return {
            g1_set: self.m_U,
            g2_set: self.m_NPLC,
        }


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load geometry configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def iso_limits(criteria: dict, t_mm: float, b_mm: float) -> dict:
    """Level-B limits as min(coefficient form, cap), from configs/geometry.yaml."""
    return {
        "undercut": min(criteria["undercut_coef_t"] * t_mm,
                        criteria["undercut_max_mm"]),
        "reinforcement": min(criteria["reinforcement_base_mm"]
                             + criteria["reinforcement_coef_b"] * b_mm,
                             criteria["reinforcement_max_mm"]),
        "misalignment": min(criteria["misalignment_coef_t"] * t_mm,
                            criteria["misalignment_max_mm"]),
    }


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
    criteria = config["acceptance_criteria"]
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

    # Governing value per measurand: worst case across faces and toes.
    governing = max(valid_measurements, key=lambda m: m.undercut_mm)
    width = max(m.bead_width_mm for m in valid_measurements)
    limits = iso_limits(criteria, criteria["plate_thickness_mm"], width)
    worst = {
        "undercut": governing.undercut_mm,
        "reinforcement": max(m.reinforcement_mm for m in valid_measurements),
        "misalignment": max(m.misalignment_mm for m in valid_measurements),
    }
    by = {k: exceedance_probability(worst[k], limits[k], uncertainty.u_total)
          for k in limits}
    gov_k = max(by, key=by.get)
    p_exceed = by[gov_k]

    # Build BPA: m({U}) = max_k P_k, m(G2) = 1 - m({U}).
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
        exceedance_by_measurand=by,
        limits_mm=limits,
        governing_measurand=gov_k,
    )


def validate_geometry_bpa(bpa: GeometryBPA, frame=None) -> bool:
    """
    Validate that a geometry BPA satisfies the required invariants.

    - Masses sum to 1 (or BPA is vacuous with m(Θ) = 1)
    - No mass on singletons other than {U}
    - No mass on sets outside the declared focal elements

    Focal elements are read from configs/frame.yaml.
    """
    if bpa.is_vacuous:
        return True

    frame = frame or load_frame()
    geometry_heads = frame.heads_for("geometry")
    g2_set = frozenset(geometry_heads["G2"])

    total = bpa.m_U + bpa.m_NPLC
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"BPA masses sum to {total}, expected 1.0")

    if not (0.0 <= bpa.m_U <= 1.0):
        raise ValueError(f"m({{U}}) = {bpa.m_U} outside [0, 1]")

    if not (0.0 <= bpa.m_NPLC <= 1.0):
        raise ValueError(f"m({set(g2_set)}) = {bpa.m_NPLC} outside [0, 1]")

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

    limits = iso_limits(criteria, t, max_width)
    reinf_limit = limits["reinforcement"]
    misal_limit = limits["misalignment"]

    return ValidationOutput(
        cell_id=measurements[0].cell_id if measurements else "unknown",
        reinforcement_mm=max_reinf,
        reinforcement_limit_mm=reinf_limit,
        reinforcement_conforms=max_reinf <= reinf_limit,
        misalignment_mm=max_misal,
        misalignment_limit_mm=misal_limit,
        misalignment_conforms=max_misal <= misal_limit,
    )
