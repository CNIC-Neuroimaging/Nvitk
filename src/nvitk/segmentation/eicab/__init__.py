"""eICAB Circle-of-Willis / TOF segmentation (local + SGE)."""

from __future__ import annotations

from . import config
from .runner import prune_eicab_outputs, resolve_eicab_tmp_dir, run_eicab, segmentation_outputs_to_keep

__all__ = [
    "config",
    "prune_eicab_outputs",
    "resolve_eicab_tmp_dir",
    "run_eicab",
    "segmentation_outputs_to_keep",
]
