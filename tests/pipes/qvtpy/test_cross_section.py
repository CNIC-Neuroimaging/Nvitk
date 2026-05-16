"""Tests for cross-section segmentation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from nvitk.pipes.qvtpy.util.cross_section import segment_in_plane


def test_segment_in_plane_disk() -> None:
    n = 41
    yy, xx = np.ogrid[:n, :n]
    cy, cx = (n - 1) / 2.0, (n - 1) / 2.0
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= 8**2
    mag = disk.astype(np.float64)
    cd = mag.copy()
    vel = mag.copy()
    mask, circ = segment_in_plane(mag, cd, vel)
    assert np.count_nonzero(mask) > 0
    assert circ >= 0.0
