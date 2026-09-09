"""ISO 5817 Level B geometric conformance as probability of exceedance.

Geometry assigns mass ONLY on {U} and {N,P,L,C}. It must never dump
dimensional non-conformance onto internal classes (porosity, LOF, crack).
That heuristic is invalid under the locked frame.

    P(non-conforming) = Phi((d_measured - d_limit) / u_total)

U in the frame is specifically rejectable undercut. Excess weld metal and
linear misalignment are still scored against ISO 5817 (returned as
diagnostics) but do not invent extra frame elements.
"""

from __future__ import annotations

from typing import Mapping

from scipy.stats import norm

from weldfusion.frame import as_focal, validate_modality_bpa
from weldfusion.types import BPA
from weldfusion.utils.config import load_named


def iso_limits(
    thickness_mm: float,
    weld_width_mm: float,
    *,
    cfg: dict | None = None,
) -> dict[str, float]:
    """Return undercut, reinforcement, and misalignment limits in mm."""
    limits = (cfg or load_named("geometry"))["acceptance_criteria"]["limits"]
    t = float(thickness_mm)
    b = float(weld_width_mm)
    undercut = min(limits["undercut_max_mm"], limits["undercut_ratio"] * t)
    reinforcement = min(
        limits["reinforcement_max_mm"],
        limits["reinforcement_offset_mm"] + limits["reinforcement_width_coeff"] * b,
    )
    misalignment = min(
        limits["misalignment_max_mm"],
        limits["misalignment_ratio"] * t + limits["misalignment_offset_mm"],
    )
    return {
        "undercut_mm": undercut,
        "reinforcement_mm": reinforcement,
        "misalignment_mm": misalignment,
    }


def exceedance_probability(measured_mm: float, limit_mm: float, u_total_mm: float) -> float:
    """Phi((d_measured - d_limit) / u_total). On-limit measurement yields 0.5."""
    if u_total_mm <= 0:
        raise ValueError("u_total_mm must be positive (zero-uncertainty on-limit is undefined)")
    return float(norm.cdf((measured_mm - limit_mm) / u_total_mm))


def conformance_bpa(
    measurements: Mapping[str, float],
    *,
    thickness_mm: float | None = None,
    u_total_mm: float | None = None,
    weld_width_mm: float | None = None,
) -> BPA:
    """Map undercut exceedance onto geometry focals {U} and {N,P,L,C}.

    `measurements` is the dict from measure_geometry(). Weld width defaults
    to the measured bead width so the reinforcement limit can be computed
    as a diagnostic; it does not enter the BPA.
    """
    geo = load_named("geometry")
    t = (
        float(thickness_mm)
        if thickness_mm is not None
        else float(geo["acceptance_criteria"]["plate_thickness_mm"])
    )
    if u_total_mm is None:
        from weldfusion.geometry.uncertainty import default_u_total_mm

        u_total_mm = default_u_total_mm()
    width = (
        float(weld_width_mm)
        if weld_width_mm is not None
        else float(measurements.get("width_mm", 0.0))
    )
    limits = iso_limits(t, width, cfg=geo)
    p_u = exceedance_probability(
        float(measurements["undercut_mm"]),
        limits["undercut_mm"],
        float(u_total_mm),
    )
    masses = {
        as_focal(["U"]): p_u,
        as_focal(["N", "P", "L", "C"]): 1.0 - p_u,
    }
    validate_modality_bpa(masses, "geometry")
    return BPA(masses=masses, modality="geometry")


def diagnostic_exceedances(
    measurements: Mapping[str, float],
    *,
    thickness_mm: float | None = None,
    u_total_mm: float | None = None,
) -> dict[str, float]:
    """ISO 5817 exceedance probabilities for all three dimensional checks."""
    geo = load_named("geometry")
    t = (
        float(thickness_mm)
        if thickness_mm is not None
        else float(geo["acceptance_criteria"]["plate_thickness_mm"])
    )
    if u_total_mm is None:
        from weldfusion.geometry.uncertainty import default_u_total_mm

        u_total_mm = default_u_total_mm()
    limits = iso_limits(t, float(measurements.get("width_mm", 0.0)), cfg=geo)
    return {
        name: exceedance_probability(
            float(measurements[key]), limits[key], float(u_total_mm)
        )
        for name, key in (
            ("undercut", "undercut_mm"),
            ("reinforcement", "reinforcement_mm"),
            ("misalignment", "misalignment_mm"),
        )
    }
