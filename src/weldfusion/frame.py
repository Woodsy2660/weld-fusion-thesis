"""Frame of discernment and per-modality focal elements.

Every class list and allowed BPA support is read from configs/frame.yaml.
Hardcoding Theta elsewhere is a bug.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping

from weldfusion.utils.config import load_named

TOL = 1e-9

Focal = frozenset[str]
Masses = dict[Focal, float]


class FrameError(ValueError):
    """Invalid BPA or frame structure."""


@lru_cache(maxsize=1)
def _raw() -> dict:
    return load_named("frame")


def reload_frame() -> None:
    """Drop the cached frame (tests / config swaps)."""
    _raw.cache_clear()


def theta() -> Focal:
    return frozenset(_raw()["theta"])


def class_order() -> tuple[str, ...]:
    return tuple(_raw()["theta"])


def labels() -> dict[str, str]:
    return dict(_raw()["labels"])


def folder_names() -> dict[str, str]:
    return dict(_raw()["folder_names"])


def folder_to_class() -> dict[str, str]:
    return {folder: cls for cls, folder in folder_names().items()}


def as_focal(classes: Iterable[str] | str) -> Focal:
    if isinstance(classes, str):
        return frozenset({classes})
    return frozenset(classes)


def format_focal(focal: Focal) -> str:
    if focal == theta():
        return "Theta"
    order = {c: i for i, c in enumerate(class_order())}
    return "{" + ",".join(sorted(focal, key=lambda c: order.get(c, 10_000))) + "}"


def allowed_focals(modality: str) -> frozenset[Focal]:
    spec = _raw()["modalities"].get(modality)
    if spec is None:
        raise FrameError(f"Unknown modality: {modality}")
    return frozenset(as_focal(item) for item in spec["focal_elements"])


def competence_partition(modality: str) -> tuple[Focal, ...]:
    spec = _raw()["competence_partitions"].get(modality)
    if spec is None:
        raise FrameError(f"No competence partition for modality: {modality}")
    return tuple(as_focal(item) for item in spec)


def modalities() -> tuple[str, ...]:
    return tuple(_raw()["modalities"].keys())


def vision_heads() -> dict[str, Focal]:
    """Return mapping from head name (V1, V2, ...) to focal element.

    This is the SINGLE SOURCE OF TRUTH for what the vision classifier outputs.
    Vision outputs 4 heads, NOT 5 Theta singletons.
    """
    raw = _raw().get("vision_heads", {})
    return {name: as_focal(classes) for name, classes in raw.items()}


def vision_head_count() -> int:
    """Number of vision output heads (read from frame.yaml, never hardcoded)."""
    return len(vision_heads())


def vision_head_order() -> tuple[str, ...]:
    """Ordered head names (V1, V2, V3, V4)."""
    return tuple(sorted(vision_heads().keys()))


def is_subset(a: Focal, b: Focal) -> bool:
    return a <= b


def intersection(*focals: Focal) -> Focal:
    if not focals:
        return theta()
    out = focals[0]
    for f in focals[1:]:
        out = out & f
    return out


def validate_bpa(masses: Mapping[Focal, float], *, tol: float = TOL) -> None:
    """Masses sum to 1, are non-negative, and every focal is a subset of Theta."""
    th = theta()
    if not masses:
        raise FrameError("BPA is empty")
    total = 0.0
    for focal, mass in masses.items():
        if not isinstance(focal, frozenset):
            raise FrameError(f"Focal element keys must be frozensets, got {type(focal)}")
        if not focal <= th:
            raise FrameError(
                f"Focal {format_focal(focal)} is not a subset of Theta={format_focal(th)}"
            )
        if mass < -tol:
            raise FrameError(f"Negative mass {mass} on {format_focal(focal)}")
        total += float(mass)
    if abs(total - 1.0) > tol:
        raise FrameError(f"Masses sum to {total}, not 1 (tol={tol})")


def validate_modality_bpa(
    masses: Mapping[Focal, float],
    modality: str,
    *,
    tol: float = TOL,
) -> None:
    """validate_bpa plus: every non-zero mass sits on a declared focal of the modality."""
    validate_bpa(masses, tol=tol)
    allowed = allowed_focals(modality)
    for focal, mass in masses.items():
        if mass > tol and focal not in allowed:
            raise FrameError(
                f"Modality {modality} assigned mass {mass} to {format_focal(focal)}, "
                f"which is not in its declared focal set"
            )
