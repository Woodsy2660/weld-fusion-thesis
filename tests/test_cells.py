"""Fusion-cell geometry: 8 per zone, 24 per coupon, 360 total, guard bands."""

from __future__ import annotations

from weldfusion.registration.cells import (
    all_cells,
    bin_s,
    cell_spans_zone_boundary,
    cells_for_coupon,
    n_cells_per_coupon,
    n_cells_per_zone,
    n_cells_total,
)
from weldfusion.utils.config import load_named


def test_counts():
    assert n_cells_per_zone() == 8
    assert n_cells_per_coupon() == 24
    assert n_cells_total() == 360
    assert len(cells_for_coupon("C01")) == 24
    assert len(all_cells()) == 360


def test_no_cell_spans_zone_boundary():
    for cell in cells_for_coupon("C01"):
        assert not cell_spans_zone_boundary(cell)


def test_centres_match_spec():
    cells = cells_for_coupon("C01")
    a = [c for c in cells if c.zone == "A"]
    assert [c.s_centre_mm for c in a] == [15, 25, 35, 45, 55, 65, 75, 85]
    b = [c for c in cells if c.zone == "B"]
    assert [c.s_centre_mm for c in b] == [115, 125, 135, 145, 155, 165, 175, 185]


def test_guard_band_bins_to_none():
    # 10 mm guard at each zone boundary: [0,10), [90,110), [190,210), [290,300]
    assert bin_s(5.0, "C01") is None
    assert bin_s(90.0, "C01") is None
    assert bin_s(100.0, "C01") is None
    assert bin_s(105.0, "C01") is None
    assert bin_s(195.0, "C01") is None
    assert bin_s(295.0, "C01") is None


def test_cell_interior_bins():
    cell = bin_s(15.0, "C01")
    assert cell is not None
    assert cell.cell_id == "A1"
    assert cell.s_centre_mm == 15.0
    cell_b = bin_s(115.0, "C01")
    assert cell_b is not None
    assert cell_b.zone == "B"
    # Half-open: s=20 belongs to the next cell (centre 25), not centre 15.
    nxt = bin_s(20.0, "C01")
    assert nxt is not None
    assert nxt.s_centre_mm == 25.0


def test_coupon_length_matches_config():
    cfg = load_named("registration")
    assert cfg["coupon"]["length_mm"] == 300
    assert cfg["coupon"]["n_coupons"] == 15
    assert cfg["coupon"]["thickness_mm"] == 10
