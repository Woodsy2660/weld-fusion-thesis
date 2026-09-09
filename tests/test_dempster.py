"""Dempster combination: identities, conflict, Zadeh paradox (regression)."""

from __future__ import annotations

import pytest

from weldfusion.frame import as_focal
from weldfusion.fusion.dempster import ConflictError, dempster_combine
from weldfusion.types import BPA


def _bpa(pairs: dict, modality=None) -> BPA:
    return BPA(masses={as_focal(k): v for k, v in pairs.items()}, modality=modality)


def _close(a: BPA, b: BPA, tol=1e-10):
    keys = set(a.masses) | set(b.masses)
    for k in keys:
        assert a.masses.get(k, 0.0) == pytest.approx(b.masses.get(k, 0.0), abs=tol)


def test_commutativity():
    m1 = _bpa({("P",): 0.6, ("N", "U", "P", "L", "C"): 0.4})
    m2 = _bpa({("C",): 0.3, ("N", "U", "P", "L", "C"): 0.7})
    ab, k_ab = dempster_combine(m1, m2)
    ba, k_ba = dempster_combine(m2, m1)
    _close(ab, ba)
    assert k_ab == pytest.approx(k_ba)


def test_associativity():
    m1 = _bpa({("P",): 0.5, ("N", "U", "P", "L", "C"): 0.5})
    m2 = _bpa({("L",): 0.2, ("N", "U"): 0.3, ("N", "U", "P", "L", "C"): 0.5})
    m3 = _bpa({("N", "P", "L", "C"): 0.8, ("U",): 0.2})
    left, _ = dempster_combine(dempster_combine(m1, m2)[0], m3)
    right, _ = dempster_combine(m1, dempster_combine(m2, m3)[0])
    _close(left, right)


def test_vacuous_is_identity():
    theta = ("N", "U", "P", "L", "C")
    vacuous = _bpa({theta: 1.0})
    m = _bpa({("P",): 0.4, ("C",): 0.1, theta: 0.5})
    combined, k = dempster_combine(m, vacuous)
    _close(combined, m)
    assert k == pytest.approx(0.0)


def test_hand_worked_conflict():
    # m1({P})=0.6, m1(Theta)=0.4
    # m2({C})=0.5, m2(Theta)=0.5
    # Empty intersections: only {P} ∩ {C}.
    # K = 0.6 * 0.5 = 0.3
    # Unnormalised: {P}=0.6*0.5=0.3, {C}=0.4*0.5=0.2, Theta=0.4*0.5=0.2
    # Normalised by 1/0.7: {P}=3/7, {C}=2/7, Theta=2/7
    theta = ("N", "U", "P", "L", "C")
    m1 = _bpa({("P",): 0.6, theta: 0.4})
    m2 = _bpa({("C",): 0.5, theta: 0.5})
    combined, k = dempster_combine(m1, m2)
    assert k == pytest.approx(0.3)
    assert combined.mass(as_focal("P")) == pytest.approx(0.3 / 0.7)
    assert combined.mass(as_focal("C")) == pytest.approx(0.2 / 0.7)
    assert combined.mass(as_focal(theta)) == pytest.approx(0.2 / 0.7)


def test_total_conflict_raises():
    m1 = _bpa({("P",): 1.0})
    m2 = _bpa({("C",): 1.0})
    with pytest.raises(ConflictError, match="K="):
        dempster_combine(m1, m2)


def test_zadeh_paradox_reproduced():
    """Regression: Dempster assigns all mass to the small-agreement class.

    m1({a})=0.9, m1({b})=0.1
    m2({c})=0.9, m2({b})=0.1
    -> K=0.99, m({b})=1. This is why Murphy averaging is the primary rule.
    """
    m1 = BPA(masses={frozenset({"a"}): 0.9, frozenset({"b"}): 0.1})
    m2 = BPA(masses={frozenset({"c"}): 0.9, frozenset({"b"}): 0.1})
    # Zadeh focals are not subsets of the weld Theta; bypass validate_bpa
    # by calling the combination internals via a patched path: construct
    # masses that *are* in Theta by relabelling a,b,c -> P,L,C.
    m1 = _bpa({("P",): 0.9, ("L",): 0.1})
    m2 = _bpa({("C",): 0.9, ("L",): 0.1})
    combined, k = dempster_combine(m1, m2)
    assert k == pytest.approx(0.99)
    assert combined.mass(as_focal("L")) == pytest.approx(1.0)
    assert combined.mass(as_focal("P")) == pytest.approx(0.0, abs=1e-12)
    assert combined.mass(as_focal("C")) == pytest.approx(0.0, abs=1e-12)
