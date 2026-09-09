"""In-memory / YAML registry of ground-truth records, keyed by (coupon_id, cell_id)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import yaml

from weldfusion.groundtruth.schema import validate_record
from weldfusion.types import GroundTruthRecord


class GroundTruthRegistry:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], GroundTruthRecord] = {}

    def add(self, record: GroundTruthRecord) -> None:
        validate_record(record)
        self._records[record.key] = record

    def get(self, coupon_id: str, cell_id: str) -> GroundTruthRecord | None:
        return self._records.get((coupon_id, cell_id))

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[GroundTruthRecord]:
        return iter(self._records.values())

    def load(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or []
        for item in raw:
            self.add(GroundTruthRecord(**item))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "coupon_id": r.coupon_id,
                "cell_id": r.cell_id,
                "label": r.label,
                "tier": r.tier,
                "source": r.source,
                "notes": r.notes,
            }
            for r in self._records.values()
        ]
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
