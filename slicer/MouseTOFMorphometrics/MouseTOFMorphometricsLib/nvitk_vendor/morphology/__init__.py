"""Vendored ``nvitk.morphology`` subset.

``centerline``, ``mst_bridge`` and ``polyline_graph`` are copied verbatim by
``vendor_sync.py``. ``components`` and ``binary`` are hand-written stand-ins:
upstream they route through ``nvitk.types.Image``, which would pull SimpleITK,
dicom2nifti and the rest of the I/O stack into Slicer.

No heavy top-level imports here — submodules are imported on demand.
"""

from __future__ import annotations

__all__: list[str] = []
