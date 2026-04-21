"""
Segmentation utilities: label-map primitives, hemisphere splitting, and
external segmentation engine wrappers (TotalSegmentator).
"""

from __future__ import annotations

from . import hemisphere, labels, total_segmentator

__all__ = ["labels", "hemisphere", "total_segmentator"]
