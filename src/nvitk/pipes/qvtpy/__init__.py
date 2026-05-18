"""Python-native QVT+ cerebrovascular pipeline (qvtpy).

Orchestrates DICOM acquisition, NIfTI conversion, eICAB CoW segmentation, 4D-flow
registration, centerline extraction, local complex-difference segmentation, LOC
placement, and phase-based hemodynamics. Entry points:

- :mod:`nvitk.pipes.qvtpy.run` — master CLI ``nvitk-qvtpy`` (local or SGE).
- :mod:`nvitk.pipes.qvtpy.run_flowshow` — interactive 4D-flow viewer ``nvitk-qvtpy-flowshow``.
- :mod:`nvitk.pipes.qvtpy.config` — host paths, stage directory names, cluster defaults.
- :mod:`nvitk.pipes.qvtpy.labels` — eICAB vs qvtpy multilabel id tables.

Per-subject results live under ``<output_root>/<subject>/qvtpy/<stage_dir>/`` (see
:data:`~nvitk.pipes.qvtpy.config.QVT_SUBDIR` and ``STAGE*_DIR`` constants).
"""

from __future__ import annotations

__all__ = []

