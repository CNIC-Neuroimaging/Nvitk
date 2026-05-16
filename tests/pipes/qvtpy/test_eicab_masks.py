"""Tests for eICAB mask resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from nvitk.pipes.qvtpy.util.eicab_masks import resolve_eicab_mask


def test_resolve_cw_when_present(tmp_path: Path) -> None:
    d = tmp_path / "eicab"
    d.mkdir()
    cw = d / "subj_eICAB_CW.nii.gz"
    cw.write_bytes(b"")
    res = resolve_eicab_mask(d, preference="cw")
    assert res.used == "cw"
    assert res.path == cw
    assert not res.fallback


def test_fallback_wb_to_cw(tmp_path: Path) -> None:
    d = tmp_path / "eicab"
    d.mkdir()
    cw = d / "subj_eICAB_CW.nii.gz"
    cw.write_bytes(b"")
    res = resolve_eicab_mask(d, preference="wb")
    assert res.used == "cw"
    assert res.fallback
    assert res.path == cw


def test_missing_both_raises(tmp_path: Path) -> None:
    d = tmp_path / "eicab"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_eicab_mask(d, preference="cw")
