"""Dataclasses shared across every layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from weldfusion.frame import Focal, validate_bpa, validate_modality_bpa


@dataclass(frozen=True)
class Coupon:
    coupon_id: str
    thickness_mm: float
    length_mm: float
    zones: tuple[str, ...] = ("A", "B", "C")


@dataclass(frozen=True)
class Cell:
    coupon_id: str
    zone: str
    cell_id: str
    s_centre_mm: float
    s_width_mm: float

    @property
    def s_lo_mm(self) -> float:
        return self.s_centre_mm - self.s_width_mm / 2.0

    @property
    def s_hi_mm(self) -> float:
        return self.s_centre_mm + self.s_width_mm / 2.0

    def contains_s(self, s_mm: float) -> bool:
        return self.s_lo_mm <= s_mm < self.s_hi_mm


@dataclass
class Observation:
    cell: Cell
    modality: str
    payload: Any
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class BPA:
    masses: dict[Focal, float]
    modality: str | None = None
    alpha: float | None = None

    def validate(self) -> None:
        if self.modality is None:
            validate_bpa(self.masses)
        else:
            validate_modality_bpa(self.masses, self.modality)

    def mass(self, focal: Focal) -> float:
        return float(self.masses.get(focal, 0.0))


@dataclass
class GroundTruthRecord:
    coupon_id: str
    cell_id: str
    label: str
    tier: str
    source: str
    notes: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.coupon_id, self.cell_id)


def masses_from_mapping(raw: Mapping[Focal, float]) -> dict[Focal, float]:
    return {frozenset(k): float(v) for k, v in raw.items()}
