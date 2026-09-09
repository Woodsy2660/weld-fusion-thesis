"""Pignistic transform."""

from __future__ import annotations

import pytest

from weldfusion.frame import as_focal, class_order, theta
from weldfusion.fusion.decision import decide, pignistic
from weldfusion.types import BPA


def test_betp_sums_to_one():
    bpa = BPA(
        masses={
            as_focal("P"): 0.2,
            as_focal(["N", "U"]): 0.3,
            as_focal(["N", "U", "L"]): 0.1,
            theta(): 0.4,
        }
    )
    betp = pignistic(bpa)
    assert sum(betp.values()) == pytest.approx(1.0)
    assert list(betp) == list(class_order())


def test_compound_mass_splits_uniformly():
    bpa = BPA(masses={as_focal(["N", "U"]): 1.0})
    betp = pignistic(bpa)
    assert betp["N"] == pytest.approx(0.5)
    assert betp["U"] == pytest.approx(0.5)
    assert betp["P"] == pytest.approx(0.0)
    assert betp["L"] == pytest.approx(0.0)
    assert betp["C"] == pytest.approx(0.0)


def test_empty_set_normalisation():
    # Unnormalised mass with leftover on empty. Pignistic divides by 1-m(empty).
    masses = {
        frozenset(): 0.2,
        as_focal("P"): 0.4,
        as_focal("C"): 0.4,
    }
    betp = pignistic(masses)
    assert sum(betp.values()) == pytest.approx(1.0)
    assert betp["P"] == pytest.approx(0.5)
    assert betp["C"] == pytest.approx(0.5)


def test_decide_argmax():
    bpa = BPA(masses={as_focal("C"): 0.7, theta(): 0.3})
    label, betp = decide(bpa)
    assert label == "C"
    assert betp["C"] > betp["N"]
