"""
Statmodels explorer — interactive mixed-effects and mediation modeling over the NVITK DB.

Description
-----------
Split into focused modules; import ``StatmodelsPanel`` / ``StatmodelsWindow`` from here.

``constants``        static choices and defaults
``theme``            dark palette, stylesheet, and the white-figure helper
``helpers``          repo access, formula parsing, checklist utilities
``frame_table``      analysis dataframe with column-anchored filtering
``derived``          derived-column editor (log, z-score, expressions)
``measurements``     multi-measurement selection and background frame loading
``report``           model info: stat chips over tabbed tables
``plot_view``        canvas, group include/exclude checklist, axis sliders
``mediation_panel``  mediation form and its worker
``window``           the explorer window itself
``panel``            the right-tab launcher
"""

from __future__ import annotations

from .constants import (
    PIPELINE_KIND_ASL,
    PIPELINE_KIND_FLAIR,
    PIPELINE_KIND_ITEMS,
    PIPELINE_KIND_QVTPY,
    PIPELINE_KIND_T1,
    PIPELINE_KIND_TOF,
)
from .panel import StatmodelsPanel
from .window import StatmodelsWindow

__all__ = [
    "PIPELINE_KIND_ASL",
    "PIPELINE_KIND_FLAIR",
    "PIPELINE_KIND_ITEMS",
    "PIPELINE_KIND_QVTPY",
    "PIPELINE_KIND_T1",
    "PIPELINE_KIND_TOF",
    "StatmodelsPanel",
    "StatmodelsWindow",
]
