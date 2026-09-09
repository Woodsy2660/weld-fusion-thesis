"""Fusion cells along the weld axis, with guard bands at zone boundaries.

Cell centres within each 100 mm zone sit at local s = 15, 25, ..., 85 mm.
A value of s that falls in a 10 mm guard band at a zone boundary bins to None.
No cell spans a zone boundary.
"""

from __future__ import annotations

from weldfusion.types import Cell
from weldfusion.utils.config import load_named


def _reg() -> dict:
    return load_named("registration")


def n_cells_per_zone() -> int:
    return len(_reg()["cell"]["local_centres_mm"])


def n_cells_per_coupon() -> int:
    return n_cells_per_zone() * len(_reg()["zones"])


def n_cells_total() -> int:
    return n_cells_per_coupon() * int(_reg()["coupon"]["n_coupons"])


def cells_for_coupon(coupon_id: str) -> list[Cell]:
    cfg = _reg()
    width = float(cfg["cell"]["width_mm"])
    local_centres = [float(x) for x in cfg["cell"]["local_centres_mm"]]
    cells: list[Cell] = []
    for zone_name, (z_lo, _z_hi) in cfg["zones"].items():
        z_lo = float(z_lo)
        for i, local in enumerate(local_centres):
            centre = z_lo + local
            cell_id = f"{zone_name}{i + 1}"
            cells.append(
                Cell(
                    coupon_id=coupon_id,
                    zone=str(zone_name),
                    cell_id=cell_id,
                    s_centre_mm=centre,
                    s_width_mm=width,
                )
            )
    return cells


def all_cells(coupon_ids: list[str] | None = None) -> list[Cell]:
    cfg = _reg()
    if coupon_ids is None:
        n = int(cfg["coupon"]["n_coupons"])
        coupon_ids = [f"C{i:02d}" for i in range(1, n + 1)]
    out: list[Cell] = []
    for cid in coupon_ids:
        out.extend(cells_for_coupon(cid))
    return out


def bin_s(s_mm: float, coupon_id: str) -> Cell | None:
    """Bin an along-weld coordinate to a cell, or None if it falls in a guard band."""
    for cell in cells_for_coupon(coupon_id):
        if cell.contains_s(s_mm):
            return cell
    return None


def cell_spans_zone_boundary(cell: Cell) -> bool:
    cfg = _reg()
    z_lo, z_hi = (float(x) for x in cfg["zones"][cell.zone])
    # Half-open [s_lo, s_hi): the hi edge may equal the zone hi only if it is
    # not actually included. Guard bands keep s_hi strictly inside the zone.
    return cell.s_lo_mm < z_lo or cell.s_hi_mm > z_hi + 1e-9
