"""Seven-subset ablation over {V}, {G}, {U}, {V,G}, {V,U}, {G,U}, {V,G,U}."""

from __future__ import annotations

from weldfusion.fusion.decision import decide
from weldfusion.fusion.murphy import murphy_average
from weldfusion.types import BPA
from weldfusion.utils.config import load_named


def ablation_subsets() -> list[tuple[str, ...]]:
    cfg = load_named("fusion")
    return [tuple(s) for s in cfg["ablation_subsets"]]


def fuse_subset(
    bpas: dict[str, BPA],
    subset: tuple[str, ...] | list[str],
) -> dict:
    """Murphy-fuse the named modalities. Returns classification, BetP, K, BPA."""
    missing = [m for m in subset if m not in bpas]
    if missing:
        raise KeyError(f"No BPA supplied for modalities: {missing}")
    selected = [bpas[m] for m in subset]
    fused, k = murphy_average(selected)
    label, betp = decide(fused)
    return {
        "subset": tuple(subset),
        "classification": label,
        "betp": betp,
        "confidence": betp[label],
        "conflict_K": k,
        "bpa": fused,
    }


def run_ablation(bpas: dict[str, BPA]) -> list[dict]:
    return [fuse_subset(bpas, subset) for subset in ablation_subsets()]
