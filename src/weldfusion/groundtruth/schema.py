"""Ground-truth record validation.

Keyed by (coupon_id, cell_id). Two tiers plus an independent U reference
that must NOT be the profilometer under test.
"""

from __future__ import annotations

from weldfusion.frame import class_order
from weldfusion.types import GroundTruthRecord

VALID_TIERS = frozenset({"1", "1b", "2", "U_ref"})

# Sources that are the instrument under test for the U class.
PROFILOMETER_SOURCES = frozenset(
    {
        "profilometer",
        "realsense",
        "realsense_d435",
        "realsense_d415",
        "d435",
        "d415",
        "geometry",
        "depth",
    }
)


def validate_record(record: GroundTruthRecord) -> None:
    if record.label not in class_order():
        raise ValueError(
            f"Label {record.label!r} is not in Theta {list(class_order())}"
        )
    if record.tier not in VALID_TIERS:
        raise ValueError(f"Unknown tier {record.tier!r}; expected one of {sorted(VALID_TIERS)}")
    if record.tier == "U_ref" or (
        record.label == "U" and record.tier in {"2"}
    ):
        src = record.source.strip().lower()
        if src in PROFILOMETER_SOURCES:
            raise ValueError(
                "Independent U reference must not be the profilometer under test "
                f"(source={record.source!r})"
            )
