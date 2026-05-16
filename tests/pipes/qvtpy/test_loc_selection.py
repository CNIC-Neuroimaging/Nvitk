"""Tests for qvtpy LOC selection."""

from __future__ import annotations

import numpy as np

from nvitk.pipes.qvtpy.labels import QVTPY_LACA, QVTPY_LPCOMM
from nvitk.pipes.qvtpy.util.loc_selection import (
    pick_endpoint_indices,
    select_arterial_locs,
)


def test_pick_endpoint_indices_inset() -> None:
    out = pick_endpoint_indices(40, inset_frac=0.08, min_inset_pts=5)
    assert out is not None
    init_idx, fin_idx = out
    assert init_idx >= 5
    assert fin_idx <= 34
    assert init_idx < fin_idx


def test_qvtpy_dual_locs_for_aca() -> None:
    pts = np.stack(
        [np.linspace(0, 30, 40), np.zeros(40), np.linspace(0, 30, 40)],
        axis=1,
    )
    recs, meta = select_arterial_locs(
        {QVTPY_LACA: pts},
        strategy="qvtpy",
        endpoint_inset_frac=0.08,
    )
    assert len(recs) == 2
    assert {r.segment_id for r in recs} == {0, 1}
    assert meta["dual_loc_fallback_vessels"] == []


def test_midpoint_single_loc_for_comm() -> None:
    pts = np.stack(
        [np.linspace(0, 10, 20), np.zeros(20), np.linspace(0, 10, 20)],
        axis=1,
    )
    recs, _ = select_arterial_locs({QVTPY_LPCOMM: pts}, strategy="midpoint")
    assert len(recs) == 1
    assert recs[0].segment_id == 0
