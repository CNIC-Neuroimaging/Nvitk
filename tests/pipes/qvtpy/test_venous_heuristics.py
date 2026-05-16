"""Tests for venous geometry heuristics."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from nvitk.pipes.qvtpy.labels import (
    NAME_LTSV,
    NAME_RTSV,
    NAME_SSSV,
    NAME_STRV,
    VENOUS_LABEL_BY_NAME,
    VENOUS_LABEL_LTSV,
    VENOUS_LABEL_RTSV,
    VENOUS_LABEL_SSSV,
    VENOUS_LABEL_STRV,
)
from nvitk.pipes.qvtpy.util.venous_heuristics import (
    assign_venous_branches,
    extract_branch_polylines,
    venous_name_to_label_id,
)


def test_venous_fixed_label_ids() -> None:
    assert venous_name_to_label_id(NAME_SSSV) == VENOUS_LABEL_SSSV == 31
    assert venous_name_to_label_id(NAME_STRV) == VENOUS_LABEL_STRV == 32
    assert venous_name_to_label_id(NAME_LTSV) == VENOUS_LABEL_LTSV == 33
    assert venous_name_to_label_id(NAME_RTSV) == VENOUS_LABEL_RTSV == 34
    assert VENOUS_LABEL_BY_NAME[NAME_SSSV] == 31


def test_assign_four_separate_branches() -> None:
    shape = (80, 40, 80)
    vol = np.zeros(shape, dtype=bool)
    vol[38:42, 5:15, 50:75] = True
    vol[10:30, 5:12, 55:65] = True
    vol[50:70, 5:12, 55:65] = True
    for t in range(20):
        vol[38:42, 5 + t, 40 + t] = True
    assigned = assign_venous_branches(vol, min_points=8)
    assert NAME_SSSV in assigned
    assert len(assigned) >= 2


def test_connected_sssv_rtsv_yields_both() -> None:
    """SSSV and RTSV in one CC should split at the junction, not merge as SSSV only."""
    shape = (80, 40, 80)
    vol = np.zeros(shape, dtype=bool)
    vol[38:42, 5:22, 35:70] = True
    vol[42:58, 5:12, 58:68] = True
    branches = extract_branch_polylines(vol, min_points=6)
    assert len(branches) >= 2
    assigned = assign_venous_branches(vol, min_points=6)
    assert NAME_SSSV in assigned
    assert NAME_RTSV in assigned


def test_missing_sinuses_ok() -> None:
    shape = (80, 40, 80)
    vol = np.zeros(shape, dtype=bool)
    vol[38:42, 5:15, 50:75] = True
    assigned = assign_venous_branches(vol, min_points=8)
    assert NAME_SSSV in assigned
    assert NAME_LTSV not in assigned
    assert NAME_RTSV not in assigned
    assert NAME_STRV not in assigned
