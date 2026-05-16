"""Tests for venous geometry heuristics."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from nvitk.pipes.qvtpy.labels import NAME_LTSV, NAME_RTSV, NAME_SSSV, NAME_STRV
from nvitk.pipes.qvtpy.util.venous_heuristics import assign_venous_branches


def test_assign_four_synthetic_branches() -> None:
    shape = (80, 40, 80)
    vol = np.zeros(shape, dtype=bool)
    # SSSV: mid x, high z
    vol[38:42, 5:15, 50:75] = True
    # LTSV: left x
    vol[10:30, 5:12, 55:65] = True
    # RTSV: right x
    vol[50:70, 5:12, 55:65] = True
    # STRV: diagonal in y-z
    for t in range(20):
        vol[38:42, 5 + t, 40 + t] = True
    assigned = assign_venous_branches(vol, min_points=8)
    assert NAME_SSSV in assigned
    assert len(assigned) >= 2
