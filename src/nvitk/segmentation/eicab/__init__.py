"""eICAB Circle-of-Willis / TOF segmentation (local + SGE)."""

from __future__ import annotations

from . import config
from .runner import eicab_tmp_dir, prune_eicab_outputs, run_eicab, segmentation_outputs_to_keep

__all__ = [
    "config",
    "eicab_tmp_dir",
    "prune_eicab_outputs",
    "run_eicab",
    "segmentation_outputs_to_keep",
]
