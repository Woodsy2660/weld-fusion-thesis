"""Parse a Krautkramer / Baker Hughes USM 100 PC export.

Open question: the on-disk format is unconfirmed (CSV vs proprietary
binary vs screenshot-derived). Do not implement a parser until a real
export file is in data/raw/ut/ and the columns (A-scan samples, probe
angle, index offset, units) are documented.

Contract: path -> iterable of raw A-scans with probe pose, so that
ut.features.extract_features and registration.ut_projection can run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def load_usm100_export(path: str | Path) -> Iterator[dict]:
    """Yield dicts with keys ascan, dt_us, theta_deg, s_probe_mm, coupon_id."""
    raise NotImplementedError(
        "USM 100 PC export format is unconfirmed. "
        "TODO: obtain a sample file from the instrument, document the "
        "column layout (A-scan samples, probe angle, index offset, units), "
        "and implement this parser against that sample. "
        f"Looked at path={path!s}."
    )
