"""Re-export cross-section utilities from :mod:`nvitk.measure.cross_section`.

qvtpy stages 5–6 call :func:`segment_at_point`, :func:`cross_section_at_loc`, and
related helpers for oblique-plane segmentation at LOCs. This module exists so pipeline
code can import from ``nvitk.pipes.qvtpy.util.cross_section`` without duplicating APIs.
"""

from __future__ import annotations

from nvitk.measure.cross_section import *  # noqa: F403
