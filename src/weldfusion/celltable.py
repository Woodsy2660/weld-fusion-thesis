"""CellTable: the columnar spine of the fusion pipeline.

The fusion cell is the unit of analysis:
- BPAs are defined over it
- Ground truth attaches to it
- Metrics are computed on it
- The ablation compares it
- The output map is made of it

A CellTable is created immediately after binning and carries one row per cell.
Every subsequent stage is a function that takes the table and returns it with
additional columns.

Columns, added in order by the stages that own them:

    STAGE           COLUMNS
    ─────────────────────────────────────────────────────────────────────────
    binning         coupon_id, cell_id, zone, s_centre_mm
    acquisition     rgb_patch_ref, depth_slice_ref, ascan_refs
    models          vision_logits, geometry_measurements, ut_features
    calibration     vision_posterior, ut_posterior, geometry_exceedance
    bpa             vision_bpa, geometry_bpa, ut_bpa
    fusion          fused_bpa, conflict_K, betp, predicted_label
    groundtruth     gt_label, gt_tier, gt_source

The table uses pandas DataFrame internally for efficient columnar operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

from weldfusion.frame import Focal, class_order, theta
from weldfusion.registration.cells import cells_for_coupon
from weldfusion.types import BPA, Cell


@dataclass
class CellTable:
    """Columnar table with one row per fusion cell.

    The table grows columns as it flows through pipeline stages.
    All operations that add columns return self for chaining.
    """

    _data: dict[str, list[Any]] = field(default_factory=dict)
    _n_rows: int = 0

    @classmethod
    def from_coupons(cls, coupon_ids: Sequence[str]) -> "CellTable":
        """Create a CellTable from a list of coupon IDs.

        Populates the binning columns: coupon_id, cell_id, zone, s_centre_mm.
        """
        table = cls()
        coupon_id_col: list[str] = []
        cell_id_col: list[str] = []
        zone_col: list[str] = []
        s_centre_col: list[float] = []

        for coupon_id in coupon_ids:
            for cell in cells_for_coupon(coupon_id):
                coupon_id_col.append(cell.coupon_id)
                cell_id_col.append(cell.cell_id)
                zone_col.append(cell.zone)
                s_centre_col.append(cell.s_centre_mm)

        table._data = {
            "coupon_id": coupon_id_col,
            "cell_id": cell_id_col,
            "zone": zone_col,
            "s_centre_mm": s_centre_col,
        }
        table._n_rows = len(coupon_id_col)
        return table

    @classmethod
    def from_cells(cls, cells: Sequence[Cell]) -> "CellTable":
        """Create a CellTable from a sequence of Cell objects."""
        table = cls()
        coupon_id_col = [c.coupon_id for c in cells]
        cell_id_col = [c.cell_id for c in cells]
        zone_col = [c.zone for c in cells]
        s_centre_col = [c.s_centre_mm for c in cells]

        table._data = {
            "coupon_id": coupon_id_col,
            "cell_id": cell_id_col,
            "zone": zone_col,
            "s_centre_mm": s_centre_col,
        }
        table._n_rows = len(cells)
        return table

    def __len__(self) -> int:
        return self._n_rows

    @property
    def columns(self) -> list[str]:
        return list(self._data.keys())

    def __getitem__(self, col: str) -> list[Any]:
        if col not in self._data:
            raise KeyError(f"Column '{col}' not found. Available: {self.columns}")
        return self._data[col]

    def __setitem__(self, col: str, values: Sequence[Any]) -> None:
        if len(values) != self._n_rows:
            raise ValueError(
                f"Column length {len(values)} != table length {self._n_rows}"
            )
        self._data[col] = list(values)

    def __contains__(self, col: str) -> bool:
        return col in self._data

    def add_column(self, name: str, values: Sequence[Any]) -> "CellTable":
        """Add a column. Returns self for chaining."""
        self[name] = values
        return self

    def row(self, idx: int) -> dict[str, Any]:
        """Get a single row as a dict."""
        if idx < 0 or idx >= self._n_rows:
            raise IndexError(f"Row index {idx} out of range [0, {self._n_rows})")
        return {col: values[idx] for col, values in self._data.items()}

    def iter_rows(self):
        """Iterate over rows as dicts."""
        for i in range(self._n_rows):
            yield self.row(i)

    def filter_by_coupon(self, coupon_id: str) -> "CellTable":
        """Return a new CellTable with only rows matching coupon_id."""
        mask = [cid == coupon_id for cid in self["coupon_id"]]
        return self._filter_by_mask(mask)

    def filter_by_indices(self, indices: Sequence[int]) -> "CellTable":
        """Return a new CellTable with only the specified row indices."""
        table = CellTable()
        idx_set = set(indices)
        for col, values in self._data.items():
            table._data[col] = [v for i, v in enumerate(values) if i in idx_set]
        table._n_rows = len(indices)
        return table

    def _filter_by_mask(self, mask: Sequence[bool]) -> "CellTable":
        table = CellTable()
        for col, values in self._data.items():
            table._data[col] = [v for v, m in zip(values, mask) if m]
        table._n_rows = sum(mask)
        return table

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe()")
        return pd.DataFrame(self._data)

    @classmethod
    def from_dataframe(cls, df: "pd.DataFrame") -> "CellTable":
        """Create from pandas DataFrame."""
        table = cls()
        table._data = {col: df[col].tolist() for col in df.columns}
        table._n_rows = len(df)
        return table

    def get_cells(self) -> list[Cell]:
        """Extract Cell objects from the table."""
        cells = []
        for i in range(self._n_rows):
            cells.append(
                Cell(
                    coupon_id=self["coupon_id"][i],
                    zone=self["zone"][i],
                    cell_id=self["cell_id"][i],
                    s_centre_mm=self["s_centre_mm"][i],
                    s_width_mm=10.0,  # from config, but hardcoded for simplicity
                )
            )
        return cells

    def get_coupon_ids(self) -> list[str]:
        """Get unique coupon IDs in order of first appearance."""
        seen = set()
        result = []
        for cid in self["coupon_id"]:
            if cid not in seen:
                seen.add(cid)
                result.append(cid)
        return result


# --- Column addition utilities for pipeline stages ---

def add_acquisition_refs(
    table: CellTable,
    *,
    rgb_refs: Sequence[str | None],
    depth_refs: Sequence[str | None],
    ascan_refs: Sequence[list[str] | None],
) -> CellTable:
    """Add acquisition reference columns."""
    return (
        table
        .add_column("rgb_patch_ref", rgb_refs)
        .add_column("depth_slice_ref", depth_refs)
        .add_column("ascan_refs", ascan_refs)
    )


def add_model_outputs(
    table: CellTable,
    *,
    vision_logits: Sequence[np.ndarray | None],
    geometry_measurements: Sequence[dict[str, float] | None],
    ut_features: Sequence[dict[str, float] | None],
) -> CellTable:
    """Add model output columns."""
    return (
        table
        .add_column("vision_logits", vision_logits)
        .add_column("geometry_measurements", geometry_measurements)
        .add_column("ut_features", ut_features)
    )


def add_calibration(
    table: CellTable,
    *,
    vision_posterior: Sequence[np.ndarray | None],
    ut_posterior: Sequence[np.ndarray | None],
    geometry_exceedance: Sequence[float | None],
) -> CellTable:
    """Add calibration output columns."""
    return (
        table
        .add_column("vision_posterior", vision_posterior)
        .add_column("ut_posterior", ut_posterior)
        .add_column("geometry_exceedance", geometry_exceedance)
    )


def add_bpas(
    table: CellTable,
    *,
    vision_bpa: Sequence[BPA | None],
    geometry_bpa: Sequence[BPA | None],
    ut_bpa: Sequence[BPA | None],
) -> CellTable:
    """Add per-modality BPA columns."""
    return (
        table
        .add_column("vision_bpa", vision_bpa)
        .add_column("geometry_bpa", geometry_bpa)
        .add_column("ut_bpa", ut_bpa)
    )


def add_fusion_results(
    table: CellTable,
    *,
    fused_bpa: Sequence[BPA | None],
    conflict_K: Sequence[float | None],
    betp: Sequence[dict[str, float] | None],
    predicted_label: Sequence[str | None],
) -> CellTable:
    """Add fusion result columns."""
    return (
        table
        .add_column("fused_bpa", fused_bpa)
        .add_column("conflict_K", conflict_K)
        .add_column("betp", betp)
        .add_column("predicted_label", predicted_label)
    )


def add_ground_truth(
    table: CellTable,
    *,
    gt_label: Sequence[str | None],
    gt_tier: Sequence[str | None],
    gt_source: Sequence[str | None],
) -> CellTable:
    """Add ground truth columns."""
    return (
        table
        .add_column("gt_label", gt_label)
        .add_column("gt_tier", gt_tier)
        .add_column("gt_source", gt_source)
    )


# --- Validation ---

def validate_binning_columns(table: CellTable) -> None:
    """Ensure required binning columns are present."""
    required = {"coupon_id", "cell_id", "zone", "s_centre_mm"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing binning columns: {missing}")


def validate_ready_for_fusion(table: CellTable) -> None:
    """Ensure BPA columns are present for fusion."""
    validate_binning_columns(table)
    bpa_cols = {"vision_bpa", "geometry_bpa", "ut_bpa"}
    missing = bpa_cols - set(table.columns)
    if missing:
        raise ValueError(f"Missing BPA columns: {missing}")
