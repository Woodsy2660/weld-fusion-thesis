"""Murphy averaging (Murphy 2000).

Average the BPAs element-wise across n sources, then Dempster-combine
the average with itself n-1 times. Primary fusion rule because Dempster
alone reproduces Zadeh's paradox on highly conflicting evidence.
"""

from __future__ import annotations

from weldfusion.frame import Focal, validate_bpa
from weldfusion.fusion.dempster import dempster_combine
from weldfusion.types import BPA


def murphy_average(bpas: list[BPA | dict[Focal, float]]) -> tuple[BPA, float]:
    """Returns (fused BPA, last-step conflict K). n=1 returns the input with K=0."""
    if not bpas:
        raise ValueError("murphy_average requires at least one BPA")
    n = len(bpas)
    mass_list = [b.masses if isinstance(b, BPA) else b for b in bpas]
    if n == 1:
        validate_bpa(mass_list[0])
        src = bpas[0]
        modality = src.modality if isinstance(src, BPA) else None
        return BPA(masses=dict(mass_list[0]), modality=modality), 0.0

    keys: set[Focal] = set()
    for masses in mass_list:
        keys.update(masses.keys())
    avg = {k: sum(m.get(k, 0.0) for m in mass_list) / n for k in keys}
    validate_bpa(avg)

    result = BPA(masses=avg)
    k = 0.0
    for _ in range(n - 1):
        result, k = dempster_combine(result, avg)
    return result, k
