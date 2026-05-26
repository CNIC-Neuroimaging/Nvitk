"""Tests for local pipeline data presets."""

from __future__ import annotations

from pathlib import Path

import pytest

from nvitk.gui.pipeline_data_presets import (
    LocalAsset,
    get_pipeline_preset,
    list_local_assets,
    list_local_cohorts,
    list_local_subjects,
    list_pipeline_preset_ids,
    load_preset_roots,
    resolve_pesa_fat_batch,
)


def test_registry_lists_pipelines() -> None:
    ids = list_pipeline_preset_ids()
    assert "qvtpy" in ids
    assert "pesa_fat" in ids


def test_load_preset_roots_override(tmp_path: Path) -> None:
    d = tmp_path / "dicom"
    n = tmp_path / "nifti"
    r = tmp_path / "results"
    d.mkdir()
    n.mkdir()
    r.mkdir()
    roots = load_preset_roots(
        "qvtpy",
        dicom_root=d,
        nifti_root=n,
        results_root=r,
    )
    assert roots.dicom_root == d
    assert roots.layout == "flat"


def test_list_local_subjects_flat(tmp_path: Path) -> None:
    nroot = tmp_path / "nifti"
    (nroot / "PESA001").mkdir(parents=True)
    (nroot / "PESA002").mkdir(parents=True)
    roots = load_preset_roots(
        "qvtpy",
        dicom_root=tmp_path / "dicom",
        nifti_root=nroot,
        results_root=tmp_path / "results",
    )
    subs = list_local_subjects(roots)
    assert subs == ["PESA001", "PESA002"]


def test_pesa_fat_nested_cohort_layout(tmp_path: Path) -> None:
    """NIFTI_ROOT / 202602_Week1 / PESA* / volumes."""
    cohort = "202602_Week1"
    nroot = tmp_path / "nifti"
    (nroot / cohort / "PESA11471769").mkdir(parents=True)
    (nroot / cohort / "PESA996004").mkdir(parents=True)
    (nroot / "DIXON_500img" / "PESA116281").mkdir(parents=True)

    assert list_local_cohorts(nifti_root=nroot) == ["202602_Week1", "DIXON_500img"]

    roots = load_preset_roots(
        "pesa_fat",
        dicom_root=tmp_path / "dicom",
        nifti_root=nroot,
        results_root=tmp_path / "results",
        batch=cohort,
    )
    assert roots.batch == cohort
    assert list_local_subjects(roots) == ["PESA11471769", "PESA996004"]

    assets = list_local_assets(roots, "PESA11471769", include_dicom=False, include_results=False)
    assert not assets  # no files yet

    subj_dir = nroot / cohort / "PESA11471769"
    (subj_dir / "CT.nii.gz").write_bytes(b"")
    assets = list_local_assets(roots, "PESA11471769", include_dicom=False, include_results=False)
    assert len(assets) == 1
    assert assets[0].kind == "nifti"


def test_pesa_fat_auto_batch(tmp_path: Path) -> None:
    cohort = "202602_Week1"
    nroot = tmp_path / "nifti"
    (nroot / cohort / "PESA001").mkdir(parents=True)
    batch = resolve_pesa_fat_batch(
        nifti_root=nroot,
        dicom_root=tmp_path / "dicom",
        results_root=tmp_path / "results",
        batch=None,
    )
    assert batch == cohort


def test_list_local_subjects_respects_include_flags(tmp_path: Path) -> None:
    cohort = "202602_Week1"
    nroot = tmp_path / "nifti" / cohort
    (nroot / "PESA001").mkdir(parents=True)
    droot = tmp_path / "dicom" / cohort
    (droot / "PESA002").mkdir(parents=True)
    roots = load_preset_roots(
        "pesa_fat",
        dicom_root=tmp_path / "dicom",
        nifti_root=tmp_path / "nifti",
        results_root=tmp_path / "results",
        batch=cohort,
    )
    assert list_local_subjects(roots, include_dicom=False, include_nifti=True) == ["PESA001"]
    assert list_local_subjects(roots, include_dicom=True, include_nifti=False) == ["PESA002"]


def test_list_local_subjects_from_results_only(tmp_path: Path) -> None:
    cohort = "202602_Week1"
    subj = "PESA099"
    stage = tmp_path / "results" / cohort / "res_segmentation_ct" / subj / "CT"
    stage.mkdir(parents=True)
    (stage / "mask.nii.gz").write_bytes(b"")
    roots = load_preset_roots(
        "pesa_fat",
        dicom_root=tmp_path / "dicom",
        nifti_root=tmp_path / "nifti",
        results_root=tmp_path / "results",
        batch=cohort,
    )
    found = list_local_subjects(
        roots,
        include_dicom=False,
        include_nifti=False,
        include_results=True,
    )
    assert found == [subj]


def test_pesa_fat_preset_uses_batch() -> None:
    spec = get_pipeline_preset("pesa_fat")
    assert spec.show_batch is True
    assert spec.pesa_fat_layout is True
    assert spec.subject_globs == ("PESA*",)
