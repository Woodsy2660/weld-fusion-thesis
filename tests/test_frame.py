"""Frame of discernment and the core design property."""

from __future__ import annotations

import pytest

from weldfusion.frame import (
    FrameError,
    allowed_focals,
    as_focal,
    class_order,
    format_focal,
    theta,
    validate_bpa,
    validate_modality_bpa,
)


def _masses(pairs: dict[tuple[str, ...] | str, float]):
    out = {}
    for k, v in pairs.items():
        out[as_focal(k)] = v
    return out


def test_theta_from_yaml():
    assert class_order() == ("N", "U", "P", "L", "C")
    assert theta() == frozenset({"N", "U", "P", "L", "C"})


def test_masses_sum_to_one():
    validate_bpa(_masses({("N",): 0.4, ("P",): 0.6}))


def test_masses_not_summing_rejected():
    with pytest.raises(FrameError, match="sum"):
        validate_bpa(_masses({("N",): 0.4, ("P",): 0.4}))


def test_negative_mass_rejected():
    with pytest.raises(FrameError, match="Negative"):
        validate_bpa(_masses({("N",): 1.2, ("P",): -0.2}))


def test_focal_must_be_subset_of_theta():
    with pytest.raises(FrameError, match="not a subset"):
        validate_bpa({frozenset({"N", "X"}): 1.0})


def test_vision_rejects_geometry_only_focal():
    # {U} is not a vision focal.
    masses = _masses({("U",): 1.0})
    with pytest.raises(FrameError, match="vision"):
        validate_modality_bpa(masses, "vision")


def test_geometry_rejects_internal_class_singleton():
    masses = _masses({("P",): 1.0})
    with pytest.raises(FrameError, match="geometry"):
        validate_modality_bpa(masses, "geometry")


def test_ut_rejects_undercut_singleton():
    masses = _masses({("U",): 1.0})
    with pytest.raises(FrameError, match="ut"):
        validate_modality_bpa(masses, "ut")


def test_geometry_accepts_declared_focals():
    masses = _masses({("U",): 0.3, ("N", "P", "L", "C"): 0.7})
    validate_modality_bpa(masses, "geometry")


def test_three_way_conjunction_isolates_n():
    vis_clean = as_focal(["N", "U", "L"])
    geo_ok = as_focal(["N", "P", "L", "C"])
    ut_clear = as_focal(["N", "U"])
    assert vis_clean & geo_ok & ut_clear == as_focal("N")


def test_vision_pairs_do_not_isolate_n():
    vis_clean = as_focal(["N", "U", "L"])
    geo_ok = as_focal(["N", "P", "L", "C"])
    ut_clear = as_focal(["N", "U"])
    assert vis_clean & geo_ok == as_focal(["N", "L"])
    assert vis_clean & ut_clear == as_focal(["N", "U"])


def test_geometry_ut_clear_reports_intersect_to_n():
    """{N,P,L,C} ∩ {N,U} = {N} is algebraically true of the locked focals.

    A sound weld is isolated by geometry-conforming ∩ UT-clear without
    vision. Vision's clean-surface report still cannot isolate N from L
    (with geometry) or from U (with UT). See CLAUDE.md.
    """
    geo_ok = as_focal(["N", "P", "L", "C"])
    ut_clear = as_focal(["N", "U"])
    assert geo_ok & ut_clear == as_focal("N")


def test_allowed_focals_are_subsets_of_theta():
    th = theta()
    for modality in ("vision", "geometry", "ut"):
        for f in allowed_focals(modality):
            assert f <= th, format_focal(f)
