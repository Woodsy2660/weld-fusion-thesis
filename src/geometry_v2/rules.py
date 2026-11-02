"""
Geometry V2: Asymmetry-based BPA construction.

This module extends the original geometry BPA with plate asymmetry as a
discriminative feature. The key insight is that asymmetry (SNR=1.75)
discriminates LOF from other conditions, but cannot distinguish LOF from
misalignment. Therefore, high asymmetry maps to the compound {L,U}.

BPA Construction Logic:
-----------------------
1. If ISO 5817 exceedance probability > threshold:
   -> m({U}) based on exceedance (original logic from src/geometry/rules.py)

2. Elif plate asymmetry > high_threshold (3.5mm):
   -> m({L,U}) = 0.6, m(Theta) = 0.4
   (High confidence that either LOF or misalignment is present)

3. Elif plate asymmetry > mod_threshold (2.0mm):
   -> m({L,U}) = 0.3, m(Theta) = 0.7
   (Moderate confidence)

4. Else:
   -> m({N,P,L}) based on conformance
   (Geometry is within normal range)

The asymmetry thresholds are derived from band analysis:
- LOF coupons: mean 4.46mm (range 3.72-5.31mm)
- Non-LOF coupons: mean 2.60mm (range 0.62-4.65mm)
- Cohen's d = 1.22 (large effect)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.geometry_v2.asymmetry import AsymmetryMeasurement, load_precomputed_asymmetry


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "frame_geometry_v2.yaml"


@dataclass
class GeometryV2Config:
    """Configuration for geometry V2 BPA construction."""
    asymmetry_high_mm: float = 3.5
    asymmetry_mod_mm: float = 2.0
    mass_high: float = 0.6
    mass_mod: float = 0.3
    uncertainty_floor: float = 0.1

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_PATH) -> "GeometryV2Config":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        g2_cfg = cfg.get("geometry_v2", {})
        return cls(
            asymmetry_high_mm=g2_cfg.get("asymmetry_high_mm", 3.5),
            asymmetry_mod_mm=g2_cfg.get("asymmetry_mod_mm", 2.0),
            mass_high=g2_cfg.get("mass_high", 0.6),
            mass_mod=g2_cfg.get("mass_mod", 0.3),
            uncertainty_floor=g2_cfg.get("uncertainty_floor", 0.1),
        )


@dataclass
class GeometryV2BPA:
    """
    BPA for geometry V2 modality.

    Focal elements:
        G1: {U}     ISO 5817 exceedance
        G2: {N,P,L} Profile conforms
        G3: {L,U}   High plate asymmetry

    Only one non-Theta focal element carries mass at a time:
    - If exceedance detected: m({U}) from G1
    - Elif high asymmetry: m({L,U}) from G3
    - Else: m({N,P,L}) from G2
    """
    coupon_id: str

    # Masses on focal elements
    m_U: float           # G1: {U} - exceedance
    m_NPL: float         # G2: {N,P,L} - conforming
    m_LU: float          # G3: {L,U} - asymmetry
    m_Theta: float       # Uncertainty

    # Decision path
    decision: str        # "exceedance", "asymmetry_high", "asymmetry_mod", "conforming"
    is_vacuous: bool

    # Supporting data
    asymmetry_mm: float
    crown_mm: float
    exceedance_prob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "coupon_id": self.coupon_id,
            "m_U": self.m_U,
            "m_NPL": self.m_NPL,
            "m_LU": self.m_LU,
            "m_Theta": self.m_Theta,
            "decision": self.decision,
            "is_vacuous": self.is_vacuous,
            "asymmetry_mm": self.asymmetry_mm,
            "crown_mm": self.crown_mm,
            "exceedance_prob": self.exceedance_prob,
        }

    def as_bpa_dict(self) -> dict[frozenset[str], float]:
        """Return as a proper BPA dict for fusion."""
        if self.is_vacuous:
            return {frozenset(["N", "U", "P", "L"]): 1.0}

        bpa = {}
        theta = frozenset(["N", "U", "P", "L"])

        if self.m_U > 0:
            bpa[frozenset(["U"])] = self.m_U
        if self.m_NPL > 0:
            bpa[frozenset(["N", "P", "L"])] = self.m_NPL
        if self.m_LU > 0:
            bpa[frozenset(["L", "U"])] = self.m_LU
        if self.m_Theta > 0:
            bpa[theta] = bpa.get(theta, 0.0) + self.m_Theta

        return bpa

    def focal_element_summary(self) -> str:
        """Human-readable summary of the BPA."""
        parts = []
        if self.m_U > 0:
            parts.append(f"m({{U}})={self.m_U:.2f}")
        if self.m_NPL > 0:
            parts.append(f"m({{N,P,L}})={self.m_NPL:.2f}")
        if self.m_LU > 0:
            parts.append(f"m({{L,U}})={self.m_LU:.2f}")
        if self.m_Theta > 0:
            parts.append(f"m(Theta)={self.m_Theta:.2f}")
        return ", ".join(parts) if parts else "vacuous"


def build_asymmetry_bpa(
    asymmetry: AsymmetryMeasurement,
    exceedance_prob: float | None = None,
    config: GeometryV2Config | None = None,
) -> GeometryV2BPA:
    """
    Build a geometry V2 BPA from asymmetry measurement.

    Args:
        asymmetry: Plate asymmetry measurement
        exceedance_prob: ISO 5817 exceedance probability from original geometry
                        module (if available). If > 0.5, takes precedence.
        config: Configuration for thresholds and masses

    Returns:
        GeometryV2BPA for fusion
    """
    config = config or GeometryV2Config.from_yaml()

    # Vacuous if measurement invalid
    if not asymmetry.is_valid:
        return GeometryV2BPA(
            coupon_id=asymmetry.coupon_id,
            m_U=0.0,
            m_NPL=0.0,
            m_LU=0.0,
            m_Theta=1.0,
            decision="vacuous",
            is_vacuous=True,
            asymmetry_mm=float("nan"),
            crown_mm=float("nan"),
            exceedance_prob=exceedance_prob,
        )

    unc = config.uncertainty_floor
    a = asymmetry.asymmetry_mm

    # Decision hierarchy
    # 1. ISO 5817 exceedance takes precedence (original geometry module)
    if exceedance_prob is not None and exceedance_prob > 0.5:
        return GeometryV2BPA(
            coupon_id=asymmetry.coupon_id,
            m_U=exceedance_prob * (1 - unc),
            m_NPL=(1 - exceedance_prob) * (1 - unc),
            m_LU=0.0,
            m_Theta=unc,
            decision="exceedance",
            is_vacuous=False,
            asymmetry_mm=a,
            crown_mm=asymmetry.crown_mm,
            exceedance_prob=exceedance_prob,
        )

    # 2. High asymmetry -> m({L,U})
    if a > config.asymmetry_high_mm:
        m_lu = config.mass_high
        return GeometryV2BPA(
            coupon_id=asymmetry.coupon_id,
            m_U=0.0,
            m_NPL=0.0,
            m_LU=m_lu,
            m_Theta=1.0 - m_lu,
            decision="asymmetry_high",
            is_vacuous=False,
            asymmetry_mm=a,
            crown_mm=asymmetry.crown_mm,
            exceedance_prob=exceedance_prob,
        )

    # 3. Moderate asymmetry -> partial m({L,U})
    if a > config.asymmetry_mod_mm:
        m_lu = config.mass_mod
        return GeometryV2BPA(
            coupon_id=asymmetry.coupon_id,
            m_U=0.0,
            m_NPL=0.0,
            m_LU=m_lu,
            m_Theta=1.0 - m_lu,
            decision="asymmetry_mod",
            is_vacuous=False,
            asymmetry_mm=a,
            crown_mm=asymmetry.crown_mm,
            exceedance_prob=exceedance_prob,
        )

    # 4. Low asymmetry -> m({N,P,L})
    m_npl = 1.0 - unc
    return GeometryV2BPA(
        coupon_id=asymmetry.coupon_id,
        m_U=0.0,
        m_NPL=m_npl,
        m_LU=0.0,
        m_Theta=unc,
        decision="conforming",
        is_vacuous=False,
        asymmetry_mm=a,
        crown_mm=asymmetry.crown_mm,
        exceedance_prob=exceedance_prob,
    )


def build_coupon_bpa(coupon_id: str, config: GeometryV2Config | None = None) -> GeometryV2BPA:
    """
    Build geometry V2 BPA for a coupon using pre-computed asymmetry.

    This is the main entry point for BPA construction.

    Args:
        coupon_id: Coupon identifier (e.g., "C01")
        config: Configuration for thresholds

    Returns:
        GeometryV2BPA for fusion
    """
    asymmetry = load_precomputed_asymmetry(coupon_id)
    if asymmetry is None:
        # Return vacuous BPA
        return GeometryV2BPA(
            coupon_id=coupon_id,
            m_U=0.0,
            m_NPL=0.0,
            m_LU=0.0,
            m_Theta=1.0,
            decision="vacuous",
            is_vacuous=True,
            asymmetry_mm=float("nan"),
            crown_mm=float("nan"),
        )

    return build_asymmetry_bpa(asymmetry, config=config)


def build_all_bpas(config: GeometryV2Config | None = None) -> dict[str, GeometryV2BPA]:
    """Build geometry V2 BPAs for all coupons."""
    coupons = [f"C{i:02d}" for i in range(1, 16)]
    return {c: build_coupon_bpa(c, config) for c in coupons}
