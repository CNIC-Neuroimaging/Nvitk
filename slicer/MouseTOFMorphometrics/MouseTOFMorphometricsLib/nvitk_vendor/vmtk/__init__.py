# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/vmtk/__init__.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""VMTK (Vascular Modeling Toolkit) wrappers.

Thin, non-interactive wrappers over ``vmtkscripts`` / ``vtkvmtk`` with NumPy and ``vtkPolyData``
in and out:

- :mod:`nvitk_vendor.vmtk.centerlines` — seed-to-seed centerlines, skeleton-seeded networks, branch
  splitting, geometry, sections, bifurcation angles, wall-to-centerline distance;
- :mod:`nvitk_vendor.vmtk.surface` — smoothing, remeshing, decimation, subdivision, capping, kite
  removal, connectivity, curvature, normals, flow extensions, mass properties, marching cubes,
  surface → binary image;
- :mod:`nvitk_vendor.vmtk.image` — vessel enhancement (Frangi / Sato / VED);
- :mod:`nvitk_vendor.vmtk.flow` — 4D-flow particle tracing (pathlines; streamlines of one phase).

VMTK itself is imported lazily (:func:`require_vmtk` raises :class:`VmtkUnavailableError` with an
install hint); nvitk_vendor environments get the cp311 wheel from ``recipe/post-link.sh``.
"""

from ._core import (
    VmtkUnavailableError,
    require_vmtk,
    vmtk_available,
    vmtk_version,
)

__all__ = [
    "VmtkUnavailableError",
    "require_vmtk",
    "vmtk_available",
    "vmtk_version",
]
