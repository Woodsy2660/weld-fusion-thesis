"""Depth-sensor uncertainty budget.

D435 active IR stereo, per-pixel RMS:

    delta_z = (z^2 / (f * B)) * delta_d

with B = 50 mm, delta_d ~ 0.08 px, f ~ 447 px at 848x480
(~0.14 mm RMS at 200 mm standoff). Averaging N independent profile rows
reduces this as sigma / sqrt(N).
"""

from __future__ import annotations

import math

from weldfusion.utils.config import load_named


def combine_uncertainty(*terms: float) -> float:
    """u_total = sqrt(u_cal^2 + u_noise^2 + u_align^2 + ...)."""
    return math.sqrt(sum(float(u) ** 2 for u in terms))


def realsense_d435_delta_z(
    z_mm: float,
    *,
    f_px: float | None = None,
    baseline_mm: float | None = None,
    delta_d_px: float | None = None,
) -> float:
    """Per-pixel depth RMS (mm) from the stereo matching model."""
    acq = load_named("acquisition")["realsense"]
    f_px = acq["f_px"] if f_px is None else f_px
    baseline_mm = acq["baseline_mm"] if baseline_mm is None else baseline_mm
    delta_d_px = acq["delta_d_px"] if delta_d_px is None else delta_d_px
    return (z_mm ** 2 / (f_px * baseline_mm)) * delta_d_px


def averaged_sigma(sigma: float, n: int) -> float:
    """Independent averaging: sigma / sqrt(N)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return sigma / math.sqrt(n)


def default_u_total_mm() -> float:
    """Combine calibration, (averaged) noise, and alignment terms from geometry.yaml."""
    u = load_named("geometry")["uncertainty"]
    return combine_uncertainty(u["u_cal_mm"], u["u_noise_mm"], u["u_align_mm"])
