"""Filesystem layout helpers for the gPET (gpetpy) pipeline.

The pipeline operates on a small on-disk tree rooted at three top-level
directories (similar to PESA-Fat):

* ``DICOM_ROOT / <batch> / <subject> / <project> / <sequence> / ...`` (inputs)
* ``NIFTI_ROOT / <batch> / <subject> /`` (stage0 outputs + later inputs)
* ``RESULTS_ROOT / <batch> / gpetpy / <stage> / <subject> /`` (stage outputs)

Batch names are derived from XNAT session dates using
:func:`nvitk.pipes.pesa_fat.common.xnat_inputs.batch_from_session_date`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nvitk.pipes.pesa_fat.common.paths import (
    DEFAULT_DICOM_ROOT as _DEFAULT_DICOM_ROOT,
    DEFAULT_NIFTI_ROOT as _DEFAULT_NIFTI_ROOT,
    DEFAULT_RESULTS_ROOT as _DEFAULT_RESULTS_ROOT,
)


DEFAULT_DICOM_ROOT: Path = _DEFAULT_DICOM_ROOT
DEFAULT_NIFTI_ROOT: Path = _DEFAULT_NIFTI_ROOT
DEFAULT_RESULTS_ROOT: Path = _DEFAULT_RESULTS_ROOT


@dataclass(frozen=True)
class GpetLayout:
    """Resolved directories for a single gPET subject run."""

    batch: str
    subject: str
    dicom_root: Path = DEFAULT_DICOM_ROOT
    nifti_root: Path = DEFAULT_NIFTI_ROOT
    results_root: Path = DEFAULT_RESULTS_ROOT
    pipeline_subdir: str = "gpetpy"

    @property
    def dicom_dir(self) -> Path:
        return self.dicom_root / self.batch / self.subject

    @property
    def nifti_dir(self) -> Path:
        return self.nifti_root / self.batch / self.subject

    @property
    def results_dir(self) -> Path:
        return self.results_root / self.batch / self.pipeline_subdir

    def stage_dir(self, stage: str) -> Path:
        return self.results_dir / stage / self.subject

    # --- DICOM layout ---
    def dicom_project_sequence_dir(self, project_id: str, sequence: str) -> Path:
        return self.dicom_dir / str(project_id).strip() / str(sequence).strip()

    def dicom_ia_ct_dir(self) -> Path:
        return self.dicom_project_sequence_dir("IA_PET_V5", "CT")

    def dicom_ia_pet_dir(self) -> Path:
        return self.dicom_project_sequence_dir("IA_PET_V5", "PET")

    def dicom_pesabrain_t1_dir(self) -> Path:
        return self.dicom_project_sequence_dir("PESA_Brain", "3D_T1")

    # --- NIfTI canonical outputs ---
    def nifti_ct(self) -> Path:
        return self.nifti_dir / "CT.nii.gz"

    def nifti_pet(self) -> Path:
        return self.nifti_dir / "PT.nii.gz"

    def nifti_t1(self) -> Path:
        return self.nifti_dir / "T1.nii.gz"

    # --- Stage1 outputs ---
    def stage1_pet_brain(self) -> Path:
        return self.stage_dir("stage1") / "PT_brain.nii.gz"

    def stage1_brain_mask_pet(self) -> Path:
        return self.stage_dir("stage1") / "brain_mask_pet.nii.gz"

    def stage1_meta(self) -> Path:
        return self.stage_dir("stage1") / "crop_meta.json"

