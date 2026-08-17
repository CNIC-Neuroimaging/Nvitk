"""Vendored subset of nvitk needed for TOF morphometrics, for 3D Slicer.

The morphometrics algorithm modules under ``measure/`` and ``morphology/`` are
**copied verbatim** from the nvitk source tree by ``vendor_sync.py``; the only
transformation is rewriting the root package name ``nvitk`` → ``nvitk_vendor``,
so the vendored code stays numerically identical to what the napari GUI and the
qvtpy stage-7 pipeline run.

``core/`` and the small ``morphology`` helpers (``binary``, ``components``) are
**hand-written** NumPy-only stand-ins for nvitk's backend/logging layer, which
exists to support CuPy and is not wanted inside Slicer. They are never
overwritten by the sync script.

This package deliberately has no heavy top-level imports — ``nvitk``'s own
``__init__`` pulls in the whole toolkit, which is exactly what vendoring avoids.
See ``VENDORED.md`` for provenance and how to refresh.
"""

from __future__ import annotations

__all__: list[str] = []
