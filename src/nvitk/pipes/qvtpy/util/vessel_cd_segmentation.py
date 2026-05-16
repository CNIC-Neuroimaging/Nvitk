"""qvtpy facade for local CD vessel segmentation (see :mod:`nvitk.segmentation.local_cd`)."""

from __future__ import annotations

from nvitk.segmentation import local_cd as _local_cd
from nvitk.segmentation.local_cd import *  # noqa: F403

# Re-export private helpers used by notebooks/tests.
_bbox_with_padding = _local_cd._bbox_with_padding
_bbox_with_vessel_padding = _local_cd._bbox_with_vessel_padding
