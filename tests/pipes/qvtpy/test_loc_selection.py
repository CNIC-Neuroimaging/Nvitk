"""Tests for LOC selection helpers."""

from __future__ import annotations

import numpy as np

from nvitk.pipes.qvtpy.util.loc_selection import pick_masked_midpoint, split_into_parts, validate_sssv_strv_swap


def test_split_into_parts() -> None:
    pts = np.arange(30, dtype=np.float64).reshape(10, 3)
    parts = split_into_parts(pts, 3)
    assert len(parts) == 3
    assert sum(p.shape[0] for p in parts) == 10


def test_pick_masked_midpoint_prefers_mask() -> None:
    pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    idx = pick_masked_midpoint(pts, mask)
    assert idx == 1


def test_validate_sssv_strv_swap() -> None:
    sssv = np.stack([np.linspace(0, 0, 20), np.linspace(0, 19, 20), np.linspace(0, 0, 20)], axis=1)
    strv = np.stack([np.linspace(0, 0, 20), np.linspace(0, 19, 20), np.linspace(0, 19, 20)], axis=1)
    si, ti = validate_sssv_strv_swap(sssv, strv, 5, 5)
    assert isinstance(si, int) and isinstance(ti, int)
