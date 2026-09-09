"""Algebraic accuracy ceiling from competence partitions and the class prior.

For a subset of modalities, take the common refinement of their competence
partitions (configs/frame.yaml). Within each atom the best an oracle can
do is pick the class with the largest prior. Ceiling =

    sum_{atom} max_{x in atom} prior(x)

Two classes are separable iff no atom contains both.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from weldfusion.frame import Focal, class_order, competence_partition, theta
from weldfusion.utils.config import load_named


def common_refinement(partitions: Sequence[Sequence[Focal]]) -> list[Focal]:
    """Finest partition obtained by intersecting blocks of each input partition."""
    atoms: list[Focal] = [theta()]
    for part in partitions:
        nxt: list[Focal] = []
        for atom in atoms:
            for block in part:
                inter = atom & block
                if inter:
                    nxt.append(inter)
        atoms = nxt
    # Deduplicate while preserving contents.
    unique: dict[Focal, None] = {}
    for a in atoms:
        unique[a] = None
    return list(unique.keys())


def refinement_for(modalities: Iterable[str]) -> list[Focal]:
    parts = [competence_partition(m) for m in modalities]
    return common_refinement(parts)


def can_separate(modalities: Iterable[str], class_a: str, class_b: str) -> bool:
    a = class_a if isinstance(class_a, str) else next(iter(class_a))
    b = class_b if isinstance(class_b, str) else next(iter(class_b))
    for atom in refinement_for(modalities):
        if a in atom and b in atom:
            return False
    return True


def algebraic_ceiling(
    modalities: Iterable[str],
    prior: Mapping[str, float] | None = None,
) -> float:
    if prior is None:
        prior = load_named("evaluation")["prior"]
    missing = [c for c in theta() if c not in prior]
    if missing:
        raise ValueError(f"Prior missing classes {missing}")
    total = 0.0
    for atom in refinement_for(modalities):
        total += max(float(prior[x]) for x in atom)
    return total


def ceiling_table(prior: Mapping[str, float] | None = None) -> dict[tuple[str, ...], float]:
    from weldfusion.fusion.ablation import ablation_subsets

    return {subset: algebraic_ceiling(subset, prior=prior) for subset in ablation_subsets()}


def separable_pairs(modalities: Iterable[str]) -> list[tuple[str, str]]:
    order = class_order()
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            if can_separate(modalities, a, b):
                pairs.append((a, b))
    return pairs
