"""Weld-axis coordinate s, origin at the first scribed datum."""

from __future__ import annotations


def s_from_datum(distance_along_weld_mm: float, datum_s_mm: float = 0.0) -> float:
    """Convert a tape measurement from a scribed datum into weld-axis s."""
    return datum_s_mm + distance_along_weld_mm
