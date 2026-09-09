"""End-to-end driver: one coupon in, per-cell fused labels out.

Contract:
    1. Load RGB, depth, and UT observations registered to weld-axis s.
    2. Bin each observation to a fusion cell (guard bands dropped).
    3. Build per-modality BPAs (vision temperature-scaled softmax -> construct;
       geometry Phi-exceedance; UT isotonic posterior -> construct).
    4. Shafer-discount by validation-fold alpha_m.
    5. Murphy-average across the requested ablation subset.
    6. Decide by pignistic transform.
    7. Write a provenance manifest next to the results.

This module is a skeleton. It does not run: capture, trained weights, and
ground truth are not in the repo.
"""

from __future__ import annotations

from typing import Any


def run_coupon(coupon_id: str, **kwargs: Any) -> list[dict]:
    """Return per-cell fused records for one coupon."""
    raise NotImplementedError(
        "run_coupon is a skeleton. Contract: coupon_id -> list of "
        "{cell_id, s_centre_mm, classification, betp, conflict_K, subset} "
        "for each fusion cell, plus a provenance manifest. "
        "Blocked on trained vision/UT weights, registered observations, "
        "and a ground-truth registry."
    )
