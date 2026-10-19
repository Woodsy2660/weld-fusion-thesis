"""Reconstruction output from fusion results.

From the completed CellTable, emit:

1. Per-coupon defect map: predicted class and belief as a function of s.
2. 3D surface model from the geometry modality: z(s, y) over the coupon.
3. Stacked subsurface sections: for each cell, the (y, z) positions of UT
   indications, stacked along s.

ACCURACY CAVEAT:

The SURFACE is true 3D: a depth camera or laser profilometer captures the
actual surface geometry z(s, y) at millimetre resolution.

The SUBSURFACE is NOT a 3D reconstruction. A single-angle ultrasonic probe
scanning from one side produces a 2D section per cell, showing (y, z)
positions of indications along the beam plane. These sections are stacked
along the weld axis s to form a "2.5D" representation.

Do NOT describe the combined output as "3D reconstruction of the weld
volume". Call it "3D surface with stacked 2D subsurface sections" or "2.5D".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from weldfusion.celltable import CellTable
from weldfusion.frame import class_order
from weldfusion.registration.ut_projection import project_indication


@dataclass
class DefectMap:
    """Per-coupon defect map: predicted class and belief vs. s.

    Attributes
    ----------
    coupon_id : str
        Coupon identifier.
    s_centres : np.ndarray
        Along-weld coordinates of cell centres (mm).
    predicted_labels : list[str | None]
        Predicted defect class per cell.
    beliefs : list[dict[str, float] | None]
        Pignistic belief per class per cell.
    conflict_K : list[float | None]
        Conflict mass K per cell.
    """

    coupon_id: str
    s_centres: np.ndarray
    predicted_labels: list[str | None]
    beliefs: list[dict[str, float] | None]
    conflict_K: list[float | None]

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "coupon_id": self.coupon_id,
            "s_centres_mm": self.s_centres.tolist(),
            "predicted_labels": self.predicted_labels,
            "beliefs": self.beliefs,
            "conflict_K": self.conflict_K,
        }


@dataclass
class SurfaceModel:
    """3D surface model from geometry modality.

    This is true 3D: z(s, y) over the coupon surface, captured by
    depth camera or laser profilometer.

    Attributes
    ----------
    coupon_id : str
        Coupon identifier.
    s_coords : np.ndarray
        Along-weld coordinates (mm), shape (n_s,).
    y_coords : np.ndarray
        Across-weld coordinates (mm), shape (n_y,).
    z_surface : np.ndarray
        Surface height z(s, y) in mm, shape (n_s, n_y).
        NaN where no measurement available.
    """

    coupon_id: str
    s_coords: np.ndarray
    y_coords: np.ndarray
    z_surface: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "coupon_id": self.coupon_id,
            "s_coords_mm": self.s_coords.tolist(),
            "y_coords_mm": self.y_coords.tolist(),
            "z_surface_mm": np.where(
                np.isnan(self.z_surface), None, self.z_surface
            ).tolist(),
        }


@dataclass
class SubsurfaceSection:
    """2D subsurface section for a single cell.

    Each section shows (y, z) positions of UT indications along the
    beam plane at the cell's s coordinate. This is NOT a tomographic
    reconstruction - it is a 2D projection from a single probe angle.

    Attributes
    ----------
    coupon_id : str
        Coupon identifier.
    cell_id : str
        Cell identifier.
    s_centre : float
        Along-weld coordinate of cell centre (mm).
    theta_deg : float
        Probe refracted angle used for this section.
    indications : list of dict
        Each indication has: y_mm, z_mm, amplitude_db, leg.
    """

    coupon_id: str
    cell_id: str
    s_centre: float
    theta_deg: float
    indications: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "coupon_id": self.coupon_id,
            "cell_id": self.cell_id,
            "s_centre_mm": self.s_centre,
            "theta_deg": self.theta_deg,
            "indications": self.indications,
        }


def build_defect_map(table: CellTable, coupon_id: str) -> DefectMap:
    """Build a defect map for one coupon from a completed CellTable.

    The table must have columns: s_centre_mm, predicted_label, betp, conflict_K.
    """
    # Filter to this coupon
    coupon_table = table.filter_by_coupon(coupon_id)

    s_centres = np.array(coupon_table["s_centre_mm"])
    predicted_labels = list(coupon_table["predicted_label"]) if "predicted_label" in coupon_table else [None] * len(coupon_table)
    beliefs = list(coupon_table["betp"]) if "betp" in coupon_table else [None] * len(coupon_table)
    conflict_K = list(coupon_table["conflict_K"]) if "conflict_K" in coupon_table else [None] * len(coupon_table)

    return DefectMap(
        coupon_id=coupon_id,
        s_centres=s_centres,
        predicted_labels=predicted_labels,
        beliefs=beliefs,
        conflict_K=conflict_K,
    )


def build_surface_model(
    coupon_id: str,
    depth_data: dict[str, np.ndarray] | None = None,
) -> SurfaceModel:
    """Build a 3D surface model from depth camera data.

    This is a stub. Full implementation requires:
        - Registered depth maps per cell
        - Profile extraction
        - Surface interpolation

    Parameters
    ----------
    coupon_id : str
        Coupon identifier.
    depth_data : dict, optional
        Depth data keyed by cell_id. Each value is a depth slice.

    Returns
    -------
    SurfaceModel
        Surface model (empty if no data provided).
    """
    # Stub: return empty surface model
    # Full implementation would extract z(s, y) from depth slices
    return SurfaceModel(
        coupon_id=coupon_id,
        s_coords=np.array([]),
        y_coords=np.array([]),
        z_surface=np.array([]).reshape(0, 0),
    )


def build_subsurface_sections(
    table: CellTable,
    coupon_id: str,
    ut_indications: dict[str, list[dict]] | None = None,
    *,
    theta_deg: float = 70.0,
    thickness_mm: float = 10.0,
    y_probe_mm: float = 20.0,
    shear_velocity_mm_per_us: float = 3.23,
) -> list[SubsurfaceSection]:
    """Build 2D subsurface sections for all cells in a coupon.

    Each section shows UT indications projected to (y, z) coordinates.
    These are stacked along s to form the "2.5D" representation.

    IMPORTANT: This is NOT a tomographic reconstruction. A single probe
    angle from one side gives a 2D section, not a 3D volume.

    Parameters
    ----------
    table : CellTable
        Table with cell geometry.
    coupon_id : str
        Coupon to build sections for.
    ut_indications : dict, optional
        Mapping cell_id -> list of indication dicts with time_of_flight_us,
        amplitude_db. If None, returns empty sections.
    theta_deg : float
        Probe refracted angle.
    thickness_mm : float
        Plate thickness.
    y_probe_mm : float
        Probe beam-exit offset from weld centreline.
    shear_velocity_mm_per_us : float
        Shear wave velocity.

    Returns
    -------
    list of SubsurfaceSection
        One section per cell, in s order.
    """
    coupon_table = table.filter_by_coupon(coupon_id)

    sections = []
    for i in range(len(coupon_table)):
        cell_id = coupon_table["cell_id"][i]
        s_centre = coupon_table["s_centre_mm"][i]

        indications = []
        if ut_indications and cell_id in ut_indications:
            for ind in ut_indications[cell_id]:
                tof = ind.get("time_of_flight_us", 0.0)
                amp = ind.get("amplitude_db", 0.0)

                # Project indication to (y, z)
                proj = project_indication(
                    time_of_flight_us=tof,
                    theta_deg=theta_deg,
                    thickness_mm=thickness_mm,
                    s_probe_mm=s_centre,
                    y_probe_mm=y_probe_mm,
                    shear_velocity_mm_per_us=shear_velocity_mm_per_us,
                )

                indications.append({
                    "y_mm": proj.y_flaw_mm,
                    "z_mm": proj.z_flaw_mm,
                    "amplitude_db": amp,
                    "leg": proj.leg,
                })

        sections.append(SubsurfaceSection(
            coupon_id=coupon_id,
            cell_id=cell_id,
            s_centre=s_centre,
            theta_deg=theta_deg,
            indications=indications,
        ))

    # Sort by s_centre
    sections.sort(key=lambda sec: sec.s_centre)

    return sections


def build_reconstruction(
    table: CellTable,
    coupon_id: str,
    *,
    depth_data: dict[str, np.ndarray] | None = None,
    ut_indications: dict[str, list[dict]] | None = None,
    theta_deg: float = 70.0,
    thickness_mm: float = 10.0,
    y_probe_mm: float = 20.0,
) -> dict[str, Any]:
    """Build complete reconstruction output for one coupon.

    Returns dict with:
        defect_map : DefectMap
        surface_model : SurfaceModel
        subsurface_sections : list[SubsurfaceSection]

    ACCURACY CAVEAT: The subsurface sections are 2D per cell, stacked
    along s. This is "2.5D", not a true 3D reconstruction. Only the
    surface model is true 3D.
    """
    return {
        "defect_map": build_defect_map(table, coupon_id),
        "surface_model": build_surface_model(coupon_id, depth_data),
        "subsurface_sections": build_subsurface_sections(
            table, coupon_id, ut_indications,
            theta_deg=theta_deg,
            thickness_mm=thickness_mm,
            y_probe_mm=y_probe_mm,
        ),
        "accuracy_note": (
            "The surface model is true 3D: z(s, y) from the depth camera. "
            "The subsurface sections are 2D per cell, stacked along s. "
            "This is a '2.5D' representation, NOT a tomographic reconstruction. "
            "A single probe angle from one side cannot reconstruct the weld volume."
        ),
    }
