"""Hand-computed UT projection, including second-leg folding."""

from __future__ import annotations

import math

import pytest

from weldfusion.registration.ut_projection import ProjectionError, project_indication

T = 10.0
THETA = 70.0


def test_first_leg_hand_computed():
    # S=10 mm, 70 deg, t=10 mm, s_probe=0
    # d = 10 sin 70, z = 10 cos 70 (first leg, vertical < t)
    p = project_indication(10.0, THETA, T, s_probe_mm=0.0)
    assert p.leg == 1
    assert p.d_mm == pytest.approx(10.0 * math.sin(math.radians(THETA)))
    assert p.z_mm == pytest.approx(10.0 * math.cos(math.radians(THETA)))
    assert p.s_flaw_mm == pytest.approx(p.d_mm)


def test_second_leg_folding():
    # S=40 mm, 70 deg: vertical = 40 cos 70 ≈ 13.68 > t, < 2t
    # z = 2t - S cos theta
    p = project_indication(40.0, THETA, T, s_probe_mm=50.0)
    vertical = 40.0 * math.cos(math.radians(THETA))
    assert vertical > T
    assert vertical < 2 * T
    assert p.leg == 2
    assert p.z_mm == pytest.approx(2 * T - vertical)
    assert p.s_flaw_mm == pytest.approx(50.0 + 40.0 * math.sin(math.radians(THETA)))


def test_beyond_second_leg_raises():
    # S=60 mm, 70 deg: vertical ≈ 20.52 > 2t=20
    with pytest.raises(ProjectionError, match="beyond second leg"):
        project_indication(60.0, THETA, T, s_probe_mm=0.0)


def test_s_flaw_differs_from_s_probe_at_nonzero_angle():
    p = project_indication(25.0, 60.0, T, s_probe_mm=100.0)
    assert p.s_flaw_mm != p.s_probe_mm
    assert p.s_flaw_mm == pytest.approx(100.0 + 25.0 * math.sin(math.radians(60.0)))


def test_zero_angle_keeps_s():
    p = project_indication(5.0, 0.0, T, s_probe_mm=12.0)
    assert p.s_flaw_mm == pytest.approx(12.0)
    assert p.d_mm == pytest.approx(0.0)
    assert p.z_mm == pytest.approx(5.0)
