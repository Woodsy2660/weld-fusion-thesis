"""Algebraic accuracy ceiling from competence partitions."""

from __future__ import annotations

import pytest

from weldfusion.evaluation.ceiling import algebraic_ceiling, can_separate, refinement_for
from weldfusion.frame import as_focal


def test_geometry_cannot_separate_p_l_c():
    assert not can_separate(["geometry"], "P", "L")
    assert not can_separate(["geometry"], "P", "C")
    assert not can_separate(["geometry"], "L", "C")
    atoms = refinement_for(["geometry"])
    lumped = as_focal(["N", "P", "L", "C"])
    assert lumped in atoms


def test_vision_geometry_cannot_separate_n_from_l():
    assert not can_separate(["vision", "geometry"], "N", "L")
    atoms = refinement_for(["vision", "geometry"])
    assert as_focal(["N", "L"]) in atoms


def test_vision_ut_cannot_separate_n_from_u():
    assert not can_separate(["vision", "ut"], "N", "U")
    atoms = refinement_for(["vision", "ut"])
    assert as_focal(["N", "U"]) in atoms


def test_all_three_separate_all_five():
    mods = ["vision", "geometry", "ut"]
    classes = ["N", "U", "P", "L", "C"]
    for i, a in enumerate(classes):
        for b in classes[i + 1 :]:
            assert can_separate(mods, a, b), f"{a} vs {b}"
    atoms = refinement_for(mods)
    assert set(atoms) == {as_focal(c) for c in classes}


def test_uniform_prior_ceilings():
    # Uniform 0.2. Ceiling = 0.2 × number of atoms.
    assert algebraic_ceiling(["geometry"]) == pytest.approx(0.4)
    assert algebraic_ceiling(["vision", "geometry"]) == pytest.approx(0.8)
    assert algebraic_ceiling(["vision", "ut"]) == pytest.approx(0.8)
    assert algebraic_ceiling(["vision", "geometry", "ut"]) == pytest.approx(1.0)
