"""Tests for IA_PET_V5 XNAT scan classification."""

from __future__ import annotations

import pandas as pd
import pytest

from nvitk.db.xnat_projects import (
    classify_scan_for_project,
    classify_scan_ia_pet_v5,
    session_modality_from_classifications,
)
from nvitk.db.xnat import (
    filter_subjects_by_asset_slots,
    list_scans_for_subject,
    list_subjects_for_project,
    project_subject_asset_slots,
)


@pytest.mark.parametrize(
    ("scan_id", "expected_seq"),
    [
        ("401", "DIXON_HEAD"),
        ("402", "DIXON_THORAX"),
        ("403", "DIXON_LEGS"),
    ],
)
def test_classify_dixon_by_scan_id(scan_id: str, expected_seq: str) -> None:
    out = classify_scan_ia_pet_v5(
        "mDIXON-Quant_BH",
        "usable",
        scan_id=scan_id,
    )
    assert out is not None
    assert out["sequence"] == expected_seq
    assert out["modality"] == "mr"


def test_classify_ct_pet() -> None:
    ct = classify_scan_ia_pet_v5("Body-Low Dose CT, iDose (4)", "usable")
    assert ct is not None
    assert ct["sequence"] == "CT"
    assert ct["modality"] == "ct"

    pet = classify_scan_ia_pet_v5("[DetailWB_CTAC] Vascular", "usable")
    assert pet is not None
    assert pet["sequence"] == "PET"
    assert pet["modality"] == "pt"


def test_classify_unusable_returns_none() -> None:
    assert classify_scan_ia_pet_v5("mDIXON-Quant_BH", "unusable", scan_id="401") is None


def test_classify_dixon_missing_region() -> None:
    assert classify_scan_ia_pet_v5("mDIXON-Quant_BH", "usable", scan_id="499") is None


def test_classify_for_project_dispatch() -> None:
    out = classify_scan_for_project(
        "IA_PET_V5",
        "mDIXON-Quant_BH",
        "usable",
        scan_id="401",
    )
    assert out is not None
    assert out["sequence"] == "DIXON_HEAD"


def test_session_modality_from_classifications() -> None:
    assert session_modality_from_classifications([{"modality": "mr"}]) == "mr"
    assert session_modality_from_classifications([{"modality": "ct"}]) == "ct"
    assert session_modality_from_classifications([{"modality": "pt"}]) == "pet"
    assert (
        session_modality_from_classifications(
            [{"modality": "ct"}, {"modality": "pt"}]
        )
        == "pet"
    )


class _FakeCatalog:
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self._tables = tables

    def table_exists(self, name: str) -> bool:
        return name in self._tables


class _FakeRepo:
    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self._tables = tables
        self.catalog = _FakeCatalog(tables)
        self.root = "/tmp/fake"

    def _load_table_frame(self, table: str, *, filters=None, use_sqlite=True):
        del use_sqlite
        df = self._tables.get(table, pd.DataFrame()).copy()
        if not filters or df.empty:
            return df
        for key, val in filters.items():
            if key in df.columns:
                df = df[df[key].astype(str) == str(val)]
        return df.reset_index(drop=True)


def test_list_subjects_and_scans() -> None:
    sessions = pd.DataFrame(
        [
            {
                "session_uid": "PESA_Brain:SUBJ1:EXP1",
                "subject_uid": "SUBJ1",
                "project_id": "PESA_Brain",
                "experiment_label": "EXP1",
            },
        ]
    )
    scans = pd.DataFrame(
        [
            {
                "scan_uid": "PESA_Brain:SUBJ1:EXP1:1",
                "session_uid": "PESA_Brain:SUBJ1:EXP1",
                "subject_uid": "SUBJ1",
                "scan_id": "1",
                "series_description": "TOF",
                "modality": "tof",
                "asset_slot": "tof",
                "local_cache_path": None,
                "quality": "usable",
            },
        ]
    )
    repo = _FakeRepo({"sessions": sessions, "scans": scans})
    assert list_subjects_for_project(repo, "PESA_Brain") == ["SUBJ1"]
    out = list_scans_for_subject(repo, "PESA_Brain", "SUBJ1")
    assert len(out) == 1
    assert out.iloc[0]["scan_id"] == "1"


def test_filter_subjects_by_asset_slots() -> None:
    sessions = pd.DataFrame(
        [
            {"session_uid": "IA:SUBJ1:E1", "subject_uid": "SUBJ1", "project_id": "IA_PET_V5"},
            {"session_uid": "IA:SUBJ2:E1", "subject_uid": "SUBJ2", "project_id": "IA_PET_V5"},
        ]
    )
    scans = pd.DataFrame(
        [
            {
                "session_uid": "IA:SUBJ1:E1",
                "subject_uid": "SUBJ1",
                "asset_slot": "ct",
            },
            {
                "session_uid": "IA:SUBJ1:E1",
                "subject_uid": "SUBJ1",
                "asset_slot": "pet",
            },
            {
                "session_uid": "IA:SUBJ2:E1",
                "subject_uid": "SUBJ2",
                "asset_slot": "ct",
            },
        ]
    )
    repo = _FakeRepo({"sessions": sessions, "scans": scans})
    slots_map = project_subject_asset_slots(repo, "IA_PET_V5")
    assert slots_map["SUBJ1"] == {"ct", "pet"}
    assert slots_map["SUBJ2"] == {"ct"}

    both = filter_subjects_by_asset_slots(slots_map, {"ct", "pet"}, match_all=True)
    assert both == ["SUBJ1"]

    any_pet = filter_subjects_by_asset_slots(slots_map, {"pet"}, match_all=False)
    assert any_pet == ["SUBJ1"]
