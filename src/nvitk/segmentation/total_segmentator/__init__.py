"""
TotalSegmentator integration: thin CLI wrapper + the class-to-label maps we use.

Usage::

    from nvitk.segmentation.total_segmentator import run_totalsegmentator, get_class_id

    run_totalsegmentator(
        input="/path/to/CT.nii.gz",
        output="/path/to/out_dir",
        task="total",
        roi_subset=["vertebrae_L4", "vertebrae_L3"],
    )
    l4_id = get_class_id("vertebrae_L4", "total")  # -> 28
"""

from __future__ import annotations

from .class_maps import (
    AVAILABLE_TASKS,
    CLASS_MAPS,
    get_class_id,
    get_class_map,
    get_class_name,
)
from .cli import main as totalseg_cli
from .runner import run_totalsegmentator

__all__ = [
    "AVAILABLE_TASKS",
    "CLASS_MAPS",
    "get_class_map",
    "get_class_id",
    "get_class_name",
    "run_totalsegmentator",
    "totalseg_cli",
]
