"""
Unit tests for the fusion module.

Tests hand-computed values from the specification.
"""

import pytest
import math
from src.weldfusion.fusion.dempster import dempster_combine, DempsterResult
from src.weldfusion.fusion.murphy import murphy_average
from src.weldfusion.fusion.decision import pignistic_transform, belief_plausibility


THETA = frozenset({"N", "U", "P", "L"})


class TestVacuousIdentity:
    """Vacuous identity: m combined with vacuous = m."""

    def test_vacuous_left(self):
        """Vacuous on left side."""
        vacuous = {THETA: 1.0}
        m = {frozenset({"P"}): 0.6, THETA: 0.4}

        result = dempster_combine(vacuous, m, THETA)

        assert abs(result.K) < 1e-12
        assert abs(result.bpa[frozenset({"P"})] - 0.6) < 1e-12
        assert abs(result.bpa[THETA] - 0.4) < 1e-12

    def test_vacuous_right(self):
        """Vacuous on right side."""
        vacuous = {THETA: 1.0}
        m = {frozenset({"L"}): 0.8, THETA: 0.2}

        result = dempster_combine(m, vacuous, THETA)

        assert abs(result.K) < 1e-12
        assert abs(result.bpa[frozenset({"L"})] - 0.8) < 1e-12
        assert abs(result.bpa[THETA] - 0.2) < 1e-12


class TestConflict:
    """
    Conflict test:
    m1({P}) = 0.8, m1(Theta) = 0.2
    m2({L}) = 0.9, m2(Theta) = 0.1

    Expected:
    K = 0.72
    m({P}) = 0.285714
    m({L}) = 0.642857
    m(Theta) = 0.071429
    """

    def test_conflict_K(self):
        m1 = {frozenset({"P"}): 0.8, THETA: 0.2}
        m2 = {frozenset({"L"}): 0.9, THETA: 0.1}

        result = dempster_combine(m1, m2, THETA)

        assert abs(result.K - 0.72) < 1e-6

    def test_conflict_masses(self):
        m1 = {frozenset({"P"}): 0.8, THETA: 0.2}
        m2 = {frozenset({"L"}): 0.9, THETA: 0.1}

        result = dempster_combine(m1, m2, THETA)

        assert abs(result.bpa[frozenset({"P"})] - 0.285714) < 1e-5
        assert abs(result.bpa[frozenset({"L"})] - 0.642857) < 1e-5
        assert abs(result.bpa[THETA] - 0.071429) < 1e-5


class TestPignisticCardinality:
    """
    Pignistic cardinality test:
    m({N,U,L}) = 0.6, m(Theta) = 0.4

    Expected:
    BetP(N) = BetP(U) = BetP(L) = 0.30
    BetP(P) = 0.10
    """

    def test_pignistic_values(self):
        bpa = {
            frozenset({"N", "U", "L"}): 0.6,
            THETA: 0.4,
        }

        betp = pignistic_transform(bpa, THETA)

        # m({N,U,L})=0.6 contributes 0.2 each to N, U, L
        # m(Theta)=0.4 contributes 0.1 each to N, U, P, L
        assert abs(betp["N"] - 0.30) < 1e-12
        assert abs(betp["U"] - 0.30) < 1e-12
        assert abs(betp["L"] - 0.30) < 1e-12
        assert abs(betp["P"] - 0.10) < 1e-12


class TestCommutativity:
    """Commutativity: m1 + m2 = m2 + m1."""

    def test_commutativity(self):
        m1 = {frozenset({"P"}): 0.5, frozenset({"N", "U"}): 0.3, THETA: 0.2}
        m2 = {frozenset({"L"}): 0.4, frozenset({"N"}): 0.2, THETA: 0.4}

        r1 = dempster_combine(m1, m2, THETA)
        r2 = dempster_combine(m2, m1, THETA)

        assert abs(r1.K - r2.K) < 1e-12

        for focal in set(r1.bpa.keys()) | set(r2.bpa.keys()):
            v1 = r1.bpa.get(focal, 0.0)
            v2 = r2.bpa.get(focal, 0.0)
            assert abs(v1 - v2) < 1e-12, f"Mismatch at {focal}: {v1} vs {v2}"


class TestAssociativity:
    """Associativity: (m1 + m2) + m3 = m1 + (m2 + m3)."""

    def test_associativity(self):
        m1 = {frozenset({"P"}): 0.4, THETA: 0.6}
        m2 = {frozenset({"L"}): 0.3, THETA: 0.7}
        m3 = {frozenset({"N"}): 0.2, THETA: 0.8}

        # (m1 + m2) + m3
        r12 = dempster_combine(m1, m2, THETA)
        r123_left = dempster_combine(r12.bpa, m3, THETA)

        # m1 + (m2 + m3)
        r23 = dempster_combine(m2, m3, THETA)
        r123_right = dempster_combine(m1, r23.bpa, THETA)

        for focal in set(r123_left.bpa.keys()) | set(r123_right.bpa.keys()):
            v_left = r123_left.bpa.get(focal, 0.0)
            v_right = r123_right.bpa.get(focal, 0.0)
            assert abs(v_left - v_right) < 1e-12, f"Mismatch at {focal}"


class TestBelPl:
    """Bel <= Pl on every focal element."""

    def test_bel_le_pl(self):
        bpa = {
            frozenset({"P"}): 0.3,
            frozenset({"N", "U"}): 0.25,
            frozenset({"L"}): 0.15,
            THETA: 0.3,
        }

        for x in THETA:
            bel, pl = belief_plausibility(bpa, frozenset({x}))
            assert bel <= pl + 1e-12, f"Bel > Pl for {x}: {bel} > {pl}"

    def test_bel_le_pl_compound(self):
        bpa = {
            frozenset({"P"}): 0.3,
            frozenset({"N", "U"}): 0.25,
            frozenset({"L"}): 0.15,
            THETA: 0.3,
        }

        for x in THETA:
            for y in THETA:
                if x != y:
                    hyp = frozenset({x, y})
                    bel, pl = belief_plausibility(bpa, hyp)
                    assert bel <= pl + 1e-12


class TestMassSum:
    """Combined BPA sums to 1.0."""

    def test_mass_sum(self):
        m1 = {frozenset({"P"}): 0.6, THETA: 0.4}
        m2 = {frozenset({"L"}): 0.5, frozenset({"N"}): 0.2, THETA: 0.3}

        result = dempster_combine(m1, m2, THETA)

        total = sum(result.bpa.values())
        assert abs(total - 1.0) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
