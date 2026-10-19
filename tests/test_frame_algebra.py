"""
Frame algebra: what the three modalities can and cannot resolve together.

These assert against configs/frame.yaml rather than hardcoded sets, so the
propositions are checked against the live config. If someone widens a focal
element, the test that depends on the widening fails here rather than silently
changing what the fused decision means.

The central claim being pinned is a thesis proposition: vision and geometry
together reach {N}, {U} and {P}, but cannot isolate L. Only UT has {L} as
a singleton focal element.

NOTE: Tests target the 4-class EXPERIMENTAL frame (Theta = {N, U, P, L}).
C was removed because no crack specimens could be fabricated. The 5-class
frame is retained in frame_5class.yaml for analytical ceiling derivation.
"""
import sys
from itertools import combinations
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bpa.construct import (  # noqa: E402
    BPAValidationError, modality_bpa, modality_focal_elements, theta_of,
    validate_modality_bpa,
)
from src.utils.frame import load_frame  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    return load_frame()


def sets_for(frame, modality):
    return list(modality_focal_elements(frame, modality).values())


# ---------------------------------------------------------------------------
# The three-head vision frame
# ---------------------------------------------------------------------------
def test_vision_has_three_heads_and_no_V2(frame):
    assert frame.head_names == ["V1", "V3", "V4"]
    assert frame.num_heads == 3
    assert "V2" not in frame.head_names


def test_head_names_are_not_renumbered(frame):
    """
    V3 and V4 keep their names although V2 is gone. Renumbering would silently
    invalidate every existing artefact, contact sheet and manifest.
    """
    assert set(frame.head_names) == {"V1", "V3", "V4"}


def test_V4_is_the_ignorance_element(frame):
    """
    V4 = {N,U,L} in the 4-class experimental frame.

    NOTE: This was {N,U,L,C} in the 5-class frame. C was removed from Theta
    because no crack specimens could be fabricated.
    """
    assert set(frame.focal_element("V4")) == {"N", "U", "L"}


def test_vision_still_never_asserts_U_or_L_alone(frame):
    """Vision cannot measure U against ISO 5817, and L is subsurface."""
    for head in frame.head_names:
        els = set(frame.focal_element(head))
        assert els != {"U"}, f"{head} asserts U alone"
        assert els != {"L"}, f"{head} asserts L alone"


def test_vision_singleton_P_is_still_legal(frame):
    """The rule is 'only declared focal elements', never 'no singletons'."""
    assert {"P"} in [set(frame.focal_element(h)) for h in frame.head_names]


# ---------------------------------------------------------------------------
# Proposition 1 -- three-way conjunction isolates the sound weld
# ---------------------------------------------------------------------------
def test_three_way_conjunction_isolates_N(frame):
    """
    {N,U,L} n {N,P,L} n {N,U} = {N}.

    In the 4-class frame (Theta = {N,U,P,L}).
    """
    vision_ignorance = frozenset(frame.focal_element("V4"))
    geometry_within = next(s for s in sets_for(frame, "geometry") if "N" in s)
    ut_nu = next(s for s in sets_for(frame, "ut") if s == frozenset({"N", "U"}))
    assert vision_ignorance & geometry_within & ut_nu == frozenset({"N"})


# ---------------------------------------------------------------------------
# Proposition 2 -- two modalities already reach N
# ---------------------------------------------------------------------------
def test_vision_deviation_plus_geometry_within_limits_gives_N(frame):
    """
    {N,U} n {N,P,L} = {N}. Vision saw a bead deviation; geometry measured it
    and found it inside the ISO 5817 limit. The weld is sound.
    """
    v3 = frozenset(frame.focal_element("V3"))
    geometry_within = next(s for s in sets_for(frame, "geometry") if "N" in s)
    assert v3 & geometry_within == frozenset({"N"})


# ---------------------------------------------------------------------------
# Proposition 3 -- the vision+geometry ceiling
# ---------------------------------------------------------------------------
def test_vision_and_geometry_reach_exactly_N_U_and_P(frame):
    """
    Over all pairwise intersections of a vision focal element with a geometry
    focal element, the reachable singletons are exactly {N}, {U} and {P}.

    This is the thesis proposition encoded executably: without UT, the fused
    decision can identify a sound weld, a geometric non-conformance and
    porosity, and nothing else.
    """
    reachable = set()
    for v in sets_for(frame, "vision"):
        for g in sets_for(frame, "geometry"):
            inter = v & g
            if len(inter) == 1:
                reachable.add(inter)
    assert reachable == {frozenset({"N"}), frozenset({"U"}), frozenset({"P"})}


def test_vision_and_geometry_cannot_isolate_L(frame):
    """
    V+G cannot isolate L (lack of fusion). Every focal element containing L
    also contains N, so BetP(L) <= BetP(N) for any V+G BPA.

    See tests/test_ceiling_proposition.py for the full algebraic proof.
    """
    for v in sets_for(frame, "vision"):
        for g in sets_for(frame, "geometry"):
            inter = v & g
            if "L" in inter:
                # L never appears without N
                assert "N" in inter, (
                    f"{set(v)} n {set(g)} = {set(inter)} contains L without N"
                )


def test_ut_is_what_isolates_L(frame):
    """UT has {L} as a singleton focal element - the only way to isolate L."""
    ut = sets_for(frame, "ut")
    assert frozenset({"L"}) in ut


# ---------------------------------------------------------------------------
# Proposition 4 -- disjoint elements feed the conflict mass
# ---------------------------------------------------------------------------
def test_vision_P_and_geometry_U_are_disjoint(frame):
    """
    {P} n {U} = empty. Vision saying porosity while geometry says undercut
    contributes to the conflict mass K in Dempster's rule, not to any focal
    element. A non-empty intersection here would mean the two modalities could
    never disagree, which would make K structurally zero.
    """
    p = frozenset({"P"})
    u = frozenset({"U"})
    assert p in sets_for(frame, "vision")
    assert u in sets_for(frame, "geometry")
    assert p & u == frozenset()


def test_every_modality_element_is_a_subset_of_theta(frame):
    theta = theta_of(frame)
    for modality in frame.modalities():
        for s in sets_for(frame, modality):
            assert s <= theta, f"{modality}: {set(s)} escapes Theta"
            assert s, f"{modality} declares an empty focal element"


def test_every_theta_symbol_is_reachable_by_some_modality(frame):
    """No symbol should be undetectable by the whole sensor suite."""
    covered = set()
    for modality in frame.modalities():
        for s in sets_for(frame, modality):
            covered |= set(s)
    assert covered == set(frame.classes)


# ---------------------------------------------------------------------------
# Per-modality BPA validation
# ---------------------------------------------------------------------------
def test_C_is_outside_theta(frame):
    """C was removed from the 4-class experimental frame."""
    assert "C" not in frame.classes


def test_L_singleton_is_accepted_for_ut(frame):
    """UT has U2 = {L} as a declared focal element."""
    ok = {frozenset({"L"}): 0.95, theta_of(frame): 0.05}
    assert validate_modality_bpa(ok, frame=frame, modality="ut") is True


def test_U_singleton_is_rejected_for_vision_but_accepted_for_geometry(frame):
    bad = {frozenset({"U"}): 0.9, theta_of(frame): 0.1}
    with pytest.raises(BPAValidationError):
        validate_modality_bpa(bad, frame=frame, modality="vision")
    assert validate_modality_bpa(bad, frame=frame, modality="geometry") is True


def test_P_singleton_is_accepted_for_vision(frame):
    ok = {frozenset({"P"}): 0.9, theta_of(frame): 0.1}
    assert validate_modality_bpa(ok, frame=frame, modality="vision") is True


def test_V4_element_is_valid_for_vision(frame):
    """
    V4 = {N,U,L} is the ignorance element in the 4-class frame.
    """
    valid = {frozenset({"N", "U", "L"}): 0.95, theta_of(frame): 0.05}
    assert validate_modality_bpa(valid, frame=frame, modality="vision") is True


def test_unknown_modality_raises(frame):
    with pytest.raises(NotImplementedError):
        validate_modality_bpa({theta_of(frame): 1.0}, frame=frame,
                              modality="radiography")


@pytest.mark.parametrize("modality", ["vision", "geometry", "ut"])
def test_modality_bpa_round_trips_through_its_own_validator(frame, modality):
    n = len(sets_for(frame, modality))
    bpa = modality_bpa([1.0 / n] * n, modality, frame=frame)
    assert abs(sum(bpa.values()) - 1.0) < 1e-9
    assert validate_modality_bpa(bpa, frame=frame, modality=modality) is True


@pytest.mark.parametrize("modality", ["vision", "geometry", "ut"])
def test_modality_bpa_reserves_theta(frame, modality):
    n = len(sets_for(frame, modality))
    bpa = modality_bpa([1.0 / n] * n, modality, frame=frame,
                       uncertainty_floor=0.07)
    assert bpa[theta_of(frame)] == pytest.approx(0.07)


def test_no_module_hardcodes_the_head_count(frame):
    """
    The model's output dimension must come from the frame. If someone pins it
    to 3 or 4, this catches it.
    """
    pytest.importorskip("torch")
    from src.vision.model import build_model
    model = build_model(pretrained=False, frame=frame)
    assert model.fc.out_features == frame.num_heads == 3
