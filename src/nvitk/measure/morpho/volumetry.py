"""Per-component volumetry and anatomy provenance for the morphometrics summaries.

The centerline pipeline reports lengths, radii and curvature but never the raw
amount of vessel it measured. These helpers add voxel-count volume, mesh volume
and surface area to each ``tree_summary`` row so the workbook carries volumetry
alongside the morphometrics.

.. note::
   ``volume_mm3`` here deliberately duplicates :func:`nvitk.measure.volume.volume_mm3`
   instead of importing it. That function pulls in ``nvitk.core.backend`` and
   ``nvitk.types``, which would break the rule that :mod:`nvitk.measure.morpho`
   imports nothing heavier than numpy/scipy/vtk/pandas/nibabel — the Slicer
   ``MouseTOFMorphometrics`` module loads this package with stubbed ``nvitk``
   and ``nvitk.measure`` packages and would otherwise fail. It is one
   multiplication.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .anatomy_axes import MorphoContext
from .models import VesselInfo


def mask_volume_mm3(mask_bool: np.ndarray, spacing) -> tuple[int, float]:
    """``(n_voxels, volume_mm3)`` for a boolean component mask."""
    n_voxels = int(np.count_nonzero(mask_bool))
    voxel_mm3 = float(np.prod(np.asarray(spacing, dtype=float)[:3]))
    return n_voxels, float(n_voxels) * voxel_mm3


def surface_mass_properties(surface: Any) -> tuple[float, float]:
    """``(mesh_volume_mm3, surface_area_mm2)`` from a closed VTK polydata surface.

    Returns ``(nan, nan)`` when the surface is missing or VTK cannot evaluate it
    (open or non-manifold meshes).
    """
    if surface is None:
        return float("nan"), float("nan")
    try:
        import vtk

        mass = vtk.vtkMassProperties()
        mass.SetInputData(surface)
        mass.Update()
        return float(mass.GetVolume()), float(mass.GetSurfaceArea())
    except Exception:
        return float("nan"), float("nan")


def component_volumetry_fields(
    mask_cc: np.ndarray,
    spacing,
    surface: Any = None,
    *,
    skeleton_length_mm: float = float("nan"),
) -> dict:
    """Volumetry columns for one label/component, ready to merge into ``tree_summary``."""
    n_voxels, volume_mm3 = mask_volume_mm3(np.asarray(mask_cc).astype(bool), spacing)
    mesh_volume_mm3, surface_area_mm2 = surface_mass_properties(surface)

    equivalent_radius_mm = float("nan")
    length = float(skeleton_length_mm)
    if np.isfinite(length) and length > 0 and volume_mm3 > 0:
        equivalent_radius_mm = float(np.sqrt(volume_mm3 / (np.pi * length)))

    return {
        "n_voxels": int(n_voxels),
        "voxel_volume_mm3": float(np.prod(np.asarray(spacing, dtype=float)[:3])),
        "volume_mm3": float(volume_mm3),
        "volume_ul": float(volume_mm3),  # 1 mm^3 == 1 microlitre
        "volume_cc": float(volume_mm3) / 1000.0,
        "mesh_volume_mm3": mesh_volume_mm3,
        "surface_area_mm2": surface_area_mm2,
        "equivalent_radius_mm": equivalent_radius_mm,
    }


def anatomy_provenance_fields(
    ctx: Optional[MorphoContext],
    vessel_info: Optional[VesselInfo] = None,
) -> dict:
    """Columns recording which species/orientation actually drove root selection.

    A wrong species or a mislabelled NIfTI header otherwise shows up only as
    silently reversed centerlines; surfacing it in the workbook makes it visible.
    """
    if ctx is None:
        return {
            "species": "",
            "orientation_axcodes": "",
            "length_scale": 1.0,
            "root_rule": "",
            "root_rule_axis": "",
        }
    rule = str(getattr(vessel_info, "no_upstream_start", "") or "")
    return {
        "species": str(ctx.axes.species),
        "orientation_axcodes": str(ctx.axes.axcodes),
        "length_scale": float(ctx.length_scale),
        "root_rule": rule,
        "root_rule_axis": ctx.axes.describe_rule(rule) if rule else "",
    }


__all__ = [
    "anatomy_provenance_fields",
    "component_volumetry_fields",
    "mask_volume_mm3",
    "surface_mass_properties",
]
