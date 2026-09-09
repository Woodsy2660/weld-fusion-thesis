"""Murphy averaging does not exhibit Zadeh's paradox."""

from __future__ import annotations

import pytest

from weldfusion.frame import as_focal
from weldfusion.fusion.murphy import murphy_average
from weldfusion.types import BPA


def _bpa(pairs: dict) -> BPA:
    return BPA(masses={as_focal(k): v for k, v in pairs.items()})


def test_average_then_n_minus_one_combinations():
    m1 = _bpa({("P",): 1.0})
    m2 = _bpa({("P",): 1.0})
    fused, k = murphy_average([m1, m2])
    assert fused.mass(as_focal("P")) == pytest.approx(1.0)
    assert k == pytest.approx(0.0)


def test_single_bpa_is_identity():
    m = _bpa({("C",): 0.4, ("N", "U", "P", "L", "C"): 0.6})
    fused, k = murphy_average([m])
    assert k == 0.0
    assert fused.mass(as_focal("C")) == pytest.approx(0.4)


def test_murphy_does_not_exhibit_zadeh_paradox():
    # Same evidence as test_dempster.test_zadeh_paradox_reproduced.
    m1 = _bpa({("P",): 0.9, ("L",): 0.1})
    m2 = _bpa({("C",): 0.9, ("L",): 0.1})
    fused, _k = murphy_average([m1, m2])
    # Dempster would put mass 1 on L. Murphy keeps most mass on P and C.
    assert fused.mass(as_focal("L")) < 0.1
    assert fused.mass(as_focal("P")) == pytest.approx(fused.mass(as_focal("C")))
    assert fused.mass(as_focal("P")) > 0.4
