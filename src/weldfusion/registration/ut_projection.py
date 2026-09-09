"""Angle-beam UT trigonometric projection onto the weld-axis coordinate s.

For an indication at sound path S with refracted angle theta:

    surface offset from probe index point:  d = S * sin(theta)
    depth, first leg:                       z = S * cos(theta)
    depth, second leg (after one skip):     z = 2t - S * cos(theta)
    weld-axis position of the reflector:    s_flaw = s_probe + S * sin(theta)

s_flaw, NOT s_probe, is what the observation is binned by.
Indications beyond the second leg raise; they are not folded further.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class ProjectionError(ValueError):
    """Indication lies outside the modelled one-skip range."""


@dataclass(frozen=True)
class UTProjection:
    d_mm: float
    z_mm: float
    s_flaw_mm: float
    s_probe_mm: float
    sound_path_mm: float
    theta_deg: float
    leg: int


def project_indication(
    sound_path_mm: float,
    theta_deg: float,
    thickness_mm: float,
    s_probe_mm: float,
    *,
    max_leg: int = 2,
) -> UTProjection:
    """Project an A-scan indication onto (d, z, s_flaw).

    `theta_deg` is the refracted angle in the test piece, not the wedge angle.
    """
    if sound_path_mm < 0:
        raise ProjectionError("Sound path must be non-negative")
    if thickness_mm <= 0:
        raise ProjectionError("Thickness must be positive")

    theta = math.radians(theta_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    d = sound_path_mm * sin_t
    vertical = sound_path_mm * cos_t

    if vertical <= thickness_mm + 1e-12:
        z = vertical
        leg = 1
    elif vertical <= 2.0 * thickness_mm + 1e-12:
        z = 2.0 * thickness_mm - vertical
        leg = 2
    else:
        raise ProjectionError(
            f"Indication at S={sound_path_mm} mm, theta={theta_deg} deg "
            f"has vertical component {vertical:.3f} mm > 2t={2.0 * thickness_mm} mm "
            f"(beyond second leg); max_leg={max_leg}"
        )

    if leg > max_leg:
        raise ProjectionError(f"Leg {leg} exceeds max_leg={max_leg}")

    s_flaw = s_probe_mm + d
    return UTProjection(
        d_mm=d,
        z_mm=z,
        s_flaw_mm=s_flaw,
        s_probe_mm=s_probe_mm,
        sound_path_mm=sound_path_mm,
        theta_deg=theta_deg,
        leg=leg,
    )
