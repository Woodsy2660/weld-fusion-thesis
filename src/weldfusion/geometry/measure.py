"""Weld-bead geometric measurements from a 1D height profile.

Ported from the previous geometry.profile.measure_geometry, with base-plate
estimation factored out. Reinforcement, undercut, width, misalignment.
"""

from __future__ import annotations

import numpy as np

from weldfusion.geometry.baseplate import estimate_baseplate
from weldfusion.utils.config import load_named


def measure_geometry(
    profile: np.ndarray,
    pixel_pitch_mm: float | None = None,
    *,
    baseplate_method: str = "outer_median",
) -> dict[str, float]:
    """Measure reinforcement, undercut, width, and misalignment (mm)."""
    if pixel_pitch_mm is None:
        pixel_pitch_mm = float(
            load_named("geometry")["profile_extraction"]["pixel_pitch_mm"]
        )

    base = estimate_baseplate(profile, method=baseplate_method)
    base_level = base["base_level"]

    reinforcement = float(np.max(profile) - base_level)
    undercut = max(0.0, float(base_level - np.min(profile)))

    above_base = profile > base_level
    if np.any(above_base):
        indices = np.where(above_base)[0]
        width = (indices[-1] - indices[0]) * float(pixel_pitch_mm)
    else:
        width = 0.0

    return {
        "reinforcement_mm": reinforcement,
        "undercut_mm": undercut,
        "width_mm": float(width),
        "misalignment_mm": float(base["misalignment_mm"]),
    }
