"""Scribed datum detection in RGB.

Contract: given an RGB image of a coupon with scribed datums at known
nominal spacing, return the pixel location (and hence s origin) of the
first datum. Used to register vision and geometry to the weld-axis
coordinate defined in registration.yaml.

Not implemented: requires labelled datum images and a detector that has
not been designed yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def detect_first_datum(image_path: str | Path) -> dict[str, Any]:
    """Return {pixel_xy, s_origin_mm, confidence} for the first scribed datum."""
    raise NotImplementedError(
        "Scribed datum detection in RGB is not implemented. "
        "Contract: image -> pixel location of the first datum, mapped to s=0."
    )
