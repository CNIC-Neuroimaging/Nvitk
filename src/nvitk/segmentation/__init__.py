"""
Segmentation utilities: label-map primitives, hemisphere splitting, and
external segmentation engine wrappers (TotalSegmentator, eICAB).
"""

from __future__ import annotations

from . import eicab, hemisphere, labels, total_segmentator

__all__ = ["eicab", "labels", "hemisphere", "total_segmentator"]
