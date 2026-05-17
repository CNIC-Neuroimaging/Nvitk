"""Tests for stage0 convert rename to vwi_bb.nii.gz."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nvitk.pipes.pesa_brain.black_blood.stage0_convert import (
    VWI_BB_JSON,
    VWI_BB_NIFTI,
    convert_subject,
    vwi_bb_nifti_path,
)


def test_convert_subject_renames_to_vwi_bb(tmp_path: Path) -> None:
    subject = "PESA001"
    dicom_root = tmp_path / "DICOM"
    nifti_root = tmp_path / "NIFTI"
    (dicom_root / subject / "vwi_bb").mkdir(parents=True)

    def fake_dcm2nii(inp: str, out: str, **kwargs: object) -> list[str]:
        out_p = Path(out)
        out_p.mkdir(parents=True, exist_ok=True)
        nii = out_p / "ACC_strong_csAI_3D_BrainVIEW_T1W_MR_1701.nii.gz"
        js = out_p / "ACC_strong_csAI_3D_BrainVIEW_T1W_MR_1701.json"
        nii.write_bytes(b"nii")
        js.write_text("{}", encoding="utf-8")
        return [str(nii)]

    with patch(
        "nvitk.pipes.pesa_brain.black_blood.stage0_convert.dcm2nii",
        side_effect=fake_dcm2nii,
    ):
        out = convert_subject(
            subject,
            dicom_root=dicom_root,
            nifti_root=nifti_root,
            skip_existing=False,
        )

    assert out == vwi_bb_nifti_path(nifti_root, subject)
    assert out.name == VWI_BB_NIFTI
    assert (out.parent / VWI_BB_JSON).is_file()
