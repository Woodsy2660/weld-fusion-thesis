"""Shafer discounting identities."""

from __future__ import annotations

import pytest

from weldfusion.bpa.discount import shafer_discount
from weldfusion.frame import as_focal, theta
from weldfusion.types import BPA


def _bpa():
    return BPA(
        masses={
            as_focal("P"): 0.4,
            as_focal(["N", "U"]): 0.3,
            theta(): 0.3,
        },
        modality="vision",
    )


def test_alpha_zero_is_identity():
    src = _bpa()
    out = shafer_discount(src, 0.0)
    for k, v in src.masses.items():
        assert out.masses[k] == pytest.approx(v)
    assert out.alpha == 0.0


def test_alpha_one_is_vacuous():
    out = shafer_discount(_bpa(), 1.0)
    assert out.mass(theta()) == pytest.approx(1.0)
    for k, v in out.masses.items():
        if k != theta():
            assert v == pytest.approx(0.0)


def test_masses_still_sum_to_one():
    for alpha in (0.0, 0.15, 0.5, 0.99, 1.0):
        out = shafer_discount(_bpa(), alpha)
        assert sum(out.masses.values()) == pytest.approx(1.0)


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        shafer_discount(_bpa(), -0.01)
    with pytest.raises(ValueError):
        shafer_discount(_bpa(), 1.01)
