"""
Basic probability assignment construction from calibrated modality outputs.

A modality does not get to put mass wherever it likes. Each one declares its
focal elements in configs/frame.yaml, and those declarations encode what the
sensor can physically resolve:

    V1  {P}      vision can see surface-breaking porosity
    V3  {N,U}    vision sees a geometric deviation but cannot measure whether
                 it breaches the ISO 5817 limit -- geometry decides that
    V4  {N,U,L}  the surface shows nothing conclusive; L lives only here,
                 because lack of fusion is subsurface and vision cannot see it

NOTE: V2 {C} was removed (precision 0.19, insufficient support). C is also
removed from the experimental frame entirely because no crack specimens could
be fabricated. The 5-class frame is retained in frame_5class.yaml for
analytical ceiling derivation.

Vision may never place mass on {U} alone or {L} alone. It may place mass on
{P}, which happens to be a singleton, because that is a declared head.
The invariant is "only declared focal elements", not "no singletons" --
validate_modality_bpa() enforces exactly that.

Focal elements are frozensets of Theta symbols. Theta itself carries the
uncertainty floor.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.frame import load_frame  # noqa: E402


class BPAValidationError(ValueError):
    """Raised when a modality assigns mass outside its declared focal elements."""


def theta_of(frame) -> frozenset:
    return frozenset(frame.classes)


def vision_focal_elements(frame) -> dict:
    """{head name: frozenset of Theta symbols} in classifier output order."""
    return {h: frozenset(frame.focal_element(h)) for h in frame.head_names}


def modality_focal_elements(frame, modality: str) -> dict:
    """
    {head: frozenset} for any modality declared in configs/frame.yaml.

    Each modality's legal focal elements encode what that sensor can physically
    resolve, which is why they differ.

    4-CLASS EXPERIMENTAL FRAME (Theta = {N, U, P, L}):
      vision    {P}, {N,U}, {N,U,L}       surface only; cannot measure U against
                                          ISO 5817, cannot see L at all
      geometry  {U}, {N,P,L}              profile only; resolves U and nothing
                                          else
      ut        {P}, {L}, {N,U}           subsurface; cannot tell a sound weld
                                          from a surface geometry problem

    NOTE: C was removed from the experimental frame because no crack specimens
    could be fabricated. The 5-class frame (frame_5class.yaml) is retained for
    analytical ceiling derivation.
    """
    return {h: frozenset(v) for h, v in frame.heads_for(modality).items()}


def modality_bpa(probabilities, modality, frame=None, uncertainty_floor=0.05):
    """
    Build a BPA for any modality from its calibrated per-head probabilities.

    Probabilities are in that modality's sorted head order.
    """
    frame = frame or load_frame()
    elements = modality_focal_elements(frame, modality)
    heads = list(elements)
    if len(probabilities) != len(heads):
        raise ValueError(
            f"expected {len(heads)} probabilities for {modality} heads "
            f"{heads}, got {len(probabilities)}"
        )
    total = float(sum(probabilities))
    if total <= 0:
        raise ValueError("probabilities sum to zero; cannot form a BPA")

    scale = 1.0 - uncertainty_floor
    bpa = {}
    for head, p in zip(heads, probabilities):
        key = elements[head]
        bpa[key] = bpa.get(key, 0.0) + (float(p) / total) * scale
    theta = theta_of(frame)
    bpa[theta] = bpa.get(theta, 0.0) + uncertainty_floor
    return bpa


def vision_bpa(probabilities, frame=None, uncertainty_floor=0.05):
    """
    Build a vision BPA from calibrated per-head probabilities.

    Args:
        probabilities: sequence of length frame.num_heads, in head order, as
            produced by softmax over the model's logits AFTER temperature
            scaling. Order matters and is the frame's head order.
        uncertainty_floor: mass reserved for Theta. Even a confident classifier
            is only one sensor, so some mass always stays uncommitted.

    Returns:
        dict {frozenset: mass}, summing to 1.
    """
    frame = frame or load_frame()
    heads = frame.head_names
    if len(probabilities) != len(heads):
        raise ValueError(
            f"expected {len(heads)} probabilities for heads {heads}, "
            f"got {len(probabilities)}"
        )

    total = float(sum(probabilities))
    if total <= 0:
        raise ValueError("probabilities sum to zero; cannot form a BPA")

    scale = 1.0 - uncertainty_floor
    elements = vision_focal_elements(frame)

    bpa = {}
    for head, p in zip(heads, probabilities):
        mass = (float(p) / total) * scale
        key = elements[head]
        # Two heads could in principle declare the same focal element; sum
        # rather than silently overwrite.
        bpa[key] = bpa.get(key, 0.0) + mass

    bpa[theta_of(frame)] = bpa.get(theta_of(frame), 0.0) + uncertainty_floor
    return bpa


def validate_modality_bpa(bpa, frame=None, modality="vision", tol=1e-6):
    """
    Check that a modality's BPA only commits mass where that modality is
    entitled to.

    PARAMETERISED BY MODALITY. The same focal element can be legal for one
    sensor and illegal for another: {C} is a legitimate UT head, because
    ultrasonic testing resolves cracks, but is rejected for vision, which since
    the three-head frame change has no crack head at all.

    The rule is "only this modality's declared focal elements", NOT "no
    singletons" -- {P} remains a legal vision singleton, and {U} a legal
    geometry singleton. Do not weaken it in that direction.

    Rejects:
      * focal elements outside the modality's declared heads (Theta excepted);
      * masses outside [0,1], or not summing to 1;
      * the empty set carrying mass.

    Returns True, or raises BPAValidationError.
    """
    frame = frame or load_frame()
    if modality not in frame.modality_heads:
        raise NotImplementedError(
            f"no focal-element declaration for modality {modality!r}; "
            f"add a '{modality}_heads' block to configs/frame.yaml before "
            f"constructing its BPA. Declared: {frame.modalities()}"
        )

    theta = theta_of(frame)
    allowed = set(modality_focal_elements(frame, modality).values()) | {theta}

    total = 0.0
    for key, mass in bpa.items():
        key = frozenset(key)
        if not key:
            raise BPAValidationError("mass assigned to the empty set")
        unknown = key - theta
        if unknown:
            raise BPAValidationError(
                f"focal element {set(key)} references symbols outside Theta: "
                f"{sorted(unknown)}"
            )
        if key not in allowed:
            singleton = " (a Theta singleton)" if len(key) == 1 else ""
            raise BPAValidationError(
                f"modality '{modality}' assigned mass to {set(key)}{singleton}, "
                f"which is not one of its declared focal elements "
                f"{[set(a) for a in sorted(allowed, key=sorted)]}. "
                f"A focal element legal for one modality is not legal for "
                f"another: {{C}} is a UT head, not a vision head."
            )
        if not (0.0 - tol <= mass <= 1.0 + tol):
            raise BPAValidationError(f"mass {mass} for {set(key)} outside [0,1]")
        total += mass

    if abs(total - 1.0) > 1e-3:
        raise BPAValidationError(f"masses sum to {total}, expected 1.0")
    return True
