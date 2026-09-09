"""Geometry conformance: on-limit P=0.5, mass only on {U} and {N,P,L,C}."""

from __future__ import annotations

import pytest

from weldfusion.frame import as_focal, allowed_focals, validate_modality_bpa
from weldfusion.geometry.conformance import (
    conformance_bpa,
    exceedance_probability,
    iso_limits,
)


def test_on_limit_probability_is_half():
    assert exceedance_probability(0.5, 0.5, u_total_mm=0.05) == pytest.approx(0.5)


def test_iso_undercut_limit_at_t10():
    limits = iso_limits(thickness_mm=10.0, weld_width_mm=12.0)
    # min(0.5, 0.05*10) = 0.5
    assert limits["undercut_mm"] == pytest.approx(0.5)
    # min(5, 1+0.1*12) = 2.2
    assert limits["reinforcement_mm"] == pytest.approx(2.2)
    # min(3, 0.1*10+0.2) = 1.2
    assert limits["misalignment_mm"] == pytest.approx(1.2)


def test_conformance_mass_only_on_geometry_focals():
    bpa = conformance_bpa(
        {"undercut_mm": 0.5, "reinforcement_mm": 1.0, "width_mm": 12.0, "misalignment_mm": 0.1},
        thickness_mm=10.0,
        u_total_mm=0.05,
    )
    validate_modality_bpa(bpa.masses, "geometry")
    allowed = allowed_focals("geometry")
    for focal, mass in bpa.masses.items():
        if mass > 0:
            assert focal in allowed
    # On-limit undercut -> equal mass on {U} and {N,P,L,C}
    assert bpa.mass(as_focal("U")) == pytest.approx(0.5)
    assert bpa.mass(as_focal(["N", "P", "L", "C"])) == pytest.approx(0.5)
    assert as_focal("P") not in bpa.masses or bpa.mass(as_focal("P")) == 0.0
    assert as_focal("L") not in bpa.masses or bpa.mass(as_focal("L")) == 0.0
    assert as_focal("C") not in bpa.masses or bpa.mass(as_focal("C")) == 0.0


def test_deep_undercut_shifts_mass_to_u():
    bpa = conformance_bpa(
        {"undercut_mm": 1.5, "reinforcement_mm": 1.0, "width_mm": 12.0, "misalignment_mm": 0.0},
        thickness_mm=10.0,
        u_total_mm=0.05,
    )
    assert bpa.mass(as_focal("U")) > 0.9
