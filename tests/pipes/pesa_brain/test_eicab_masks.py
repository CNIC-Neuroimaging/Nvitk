"""Tests for black_blood eICAB CW/WB mask resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from nvitk.pipes.pesa_brain.black_blood.util.eicab_masks import resolve_eicab_mask


def test_resolve_eicab_mask_cw_preferred(tmp_path: Path) -> None:
    eicab_dir = tmp_path / "eicab"
    eicab_dir.mkdir()
    cw = eicab_dir / "subj_eICAB_CW.nii.gz"
    wb = eicab_dir / "subj_eICAB_WB.nii.gz"
    cw.write_bytes(b"cw")
    wb.write_bytes(b"wb")

    res = resolve_eicab_mask(eicab_dir, preference="cw")
    assert res.used == "cw"
    assert res.path == cw
    assert not res.fallback


def test_resolve_eicab_mask_wb_fallback_from_cw_request(tmp_path: Path) -> None:
    eicab_dir = tmp_path / "eicab"
    eicab_dir.mkdir()
    wb = eicab_dir / "subj_eICAB_WB.nii.gz"
    wb.write_bytes(b"wb")

    res = resolve_eicab_mask(eicab_dir, preference="cw")
    assert res.requested == "cw"
    assert res.used == "wb"
    assert res.fallback


def test_resolve_eicab_mask_neither_raises(tmp_path: Path) -> None:
    eicab_dir = tmp_path / "empty"
    eicab_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_eicab_mask(eicab_dir, preference="wb")
