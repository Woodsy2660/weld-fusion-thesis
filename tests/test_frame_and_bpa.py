"""
The frame is the single source of truth, and a modality may only put mass on
the focal elements it declares. These tests pin both.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bpa.construct import (  # noqa: E402
    BPAValidationError, validate_modality_bpa, vision_bpa, vision_focal_elements,
)
from src.utils.frame import (  # noqa: E402
    DISCARD, load_class_map, load_frame, resolve,
)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
def test_frame_declares_three_vision_heads():
    """V2 {C} was removed; V3 and V4 deliberately keep their names."""
    frame = load_frame()
    assert frame.head_names == ["V1", "V3", "V4"]
    assert frame.num_heads == 3


def test_focal_elements_match_the_thesis_definition():
    frame = load_frame()
    assert frame.focal_element("V1") == ["P"]
    assert set(frame.focal_element("V3")) == {"N", "U"}
    # Widened from {N,U,L} when the crack head was removed: with no V2, vision
    # can no longer assert "not a crack", so C re-enters the ignorance set.
    assert set(frame.focal_element("V4")) == {"N", "U", "L", "C"}
    assert "V2" not in frame.vision_heads


def test_theta_has_five_classes():
    assert set(load_frame().classes) == {"N", "U", "P", "L", "C"}


def test_L_appears_only_inside_the_ignorance_element():
    """Lack of fusion is subsurface; vision must never predict it alone."""
    frame = load_frame()
    for head in frame.head_names:
        els = set(frame.focal_element(head))
        if "L" in els:
            assert els == {"N", "U", "L", "C"}, (
                f"{head} contains L outside the ignorance element"
            )


def test_C_appears_only_inside_the_ignorance_element():
    """Since V2 was removed, C is in the same position as L."""
    frame = load_frame()
    for head in frame.head_names:
        els = set(frame.focal_element(head))
        if "C" in els:
            assert els == {"N", "U", "L", "C"}, (
                f"{head} contains C outside the ignorance element"
            )


def test_U_is_never_a_vision_singleton():
    """Vision sees deviation; only geometry can measure it against ISO 5817."""
    frame = load_frame()
    for head in frame.head_names:
        assert set(frame.focal_element(head)) != {"U"}


def test_head_order_is_sorted_for_deterministic_output_order():
    frame = load_frame()
    assert frame.head_names == sorted(frame.head_names)


# ---------------------------------------------------------------------------
# Class map
# ---------------------------------------------------------------------------
def test_every_mapped_target_is_a_head_or_discard():
    frame = load_frame()
    cmap = load_class_map(frame=frame)
    valid = set(frame.head_names) | {DISCARD}
    for source, mapping in cmap.items():
        for cls, target in mapping.items():
            assert target in valid, f"{source}/{cls} -> {target}"


def test_unmapped_class_raises_rather_than_silently_discarding():
    cmap = load_class_map()
    with pytest.raises(KeyError):
        resolve(cmap, "final_welding", "a class that does not exist")


def test_class_map_covers_every_class_in_each_source_data_yaml():
    """
    A class present in a source's data.yaml but absent from class_map.yaml
    would stop the conversion. Catch that here rather than 20 minutes in.
    """
    from src.vision.convert_and_remap import find_dir, load_yaml_names

    cmap = load_class_map()
    vision = find_dir(REPO_ROOT / "data", "vision")
    if vision is None:
        pytest.skip("vision data not present")

    for source, dirname in (("final_welding", "FINAL WELDING DATASET"),
                            ("wijaya", "The Welding Defect Dataset")):
        root = find_dir(vision, dirname)
        if root is None or not (root / "data.yaml").exists():
            continue
        for name in load_yaml_names(root / "data.yaml"):
            assert name in cmap[source], (
                f"{source}: class {name!r} from data.yaml is unmapped in "
                f"configs/class_map.yaml"
            )


# ---------------------------------------------------------------------------
# BPA construction and validation
# ---------------------------------------------------------------------------
def test_vision_bpa_sums_to_one_and_reserves_theta():
    frame = load_frame()
    bpa = vision_bpa([0.7, 0.2, 0.1], frame=frame, uncertainty_floor=0.05)
    assert abs(sum(bpa.values()) - 1.0) < 1e-9
    assert bpa[frozenset(frame.classes)] == pytest.approx(0.05)


def test_vision_bpa_only_uses_declared_focal_elements():
    frame = load_frame()
    bpa = vision_bpa([1 / 3] * 3, frame=frame)
    allowed = set(vision_focal_elements(frame).values()) | {frozenset(frame.classes)}
    assert set(bpa) <= allowed
    assert validate_modality_bpa(bpa, frame=frame) is True


def test_validate_rejects_mass_on_U_singleton():
    """The core invariant: vision cannot measure U on its own."""
    frame = load_frame()
    bad = {frozenset({"U"}): 0.95, frozenset(frame.classes): 0.05}
    with pytest.raises(BPAValidationError, match="singleton"):
        validate_modality_bpa(bad, frame=frame)


def test_validate_rejects_mass_on_L_singleton():
    """Lack of fusion is subsurface; a vision head can never assert it alone."""
    frame = load_frame()
    bad = {frozenset({"L"}): 0.95, frozenset(frame.classes): 0.05}
    with pytest.raises(BPAValidationError, match="singleton"):
        validate_modality_bpa(bad, frame=frame)


def test_validate_accepts_the_P_singleton_because_it_is_declared():
    """
    {P} is a singleton but IS a declared vision head. The rule is "only
    declared focal elements", not "no singletons" -- guard against someone
    tightening it the wrong way.

    {C} used to be accepted here too. It no longer is for vision, because V2
    was removed; see tests/test_frame_algebra.py for the vision-rejects /
    ut-accepts pair.
    """
    frame = load_frame()
    ok = {frozenset({"P"}): 0.95, frozenset(frame.classes): 0.05}
    assert validate_modality_bpa(ok, frame=frame) is True


def test_validate_rejects_undeclared_compound_element():
    frame = load_frame()
    bad = {frozenset({"P", "C"}): 0.95, frozenset(frame.classes): 0.05}
    with pytest.raises(BPAValidationError):
        validate_modality_bpa(bad, frame=frame)


def test_validate_rejects_symbols_outside_theta():
    frame = load_frame()
    bad = {frozenset({"X"}): 0.95, frozenset(frame.classes): 0.05}
    with pytest.raises(BPAValidationError, match="outside Theta"):
        validate_modality_bpa(bad, frame=frame)


def test_validate_rejects_masses_that_do_not_sum_to_one():
    frame = load_frame()
    bad = {frozenset({"P"}): 0.5, frozenset(frame.classes): 0.2}
    with pytest.raises(BPAValidationError, match="sum"):
        validate_modality_bpa(bad, frame=frame)


def test_validate_rejects_empty_set_mass():
    frame = load_frame()
    with pytest.raises(BPAValidationError, match="empty set"):
        validate_modality_bpa({frozenset(): 1.0}, frame=frame)


def test_vision_bpa_rejects_wrong_probability_count():
    with pytest.raises(ValueError):
        vision_bpa([0.25] * 4)      # the old four-head shape


# ---------------------------------------------------------------------------
# Model head count
# ---------------------------------------------------------------------------
def test_model_head_count_comes_from_the_frame():
    pytest.importorskip("torch")
    from src.vision.model import build_model
    frame = load_frame()
    model = build_model(pretrained=False)
    assert model.fc.out_features == frame.num_heads == 3
