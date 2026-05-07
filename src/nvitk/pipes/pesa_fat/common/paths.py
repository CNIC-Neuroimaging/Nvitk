"""Filesystem layout helpers for the PESA-Fat CT-PET / Dixon batches.

The pipelines operate on a flat on-disk tree rooted at three top-level
directories:

* ``DICOM_ROOT / <batch> / PESA* / ...``   (inputs)
* ``NIFTI_ROOT / <batch> / PESA* / ...``   (stage 0 outputs + inputs to later stages)
* ``RESULTS_ROOT / <batch> / res_<stage>/PESA* / ...``   (later-stage outputs)

All pipelines derive their per-subject paths from :class:`BatchLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_DICOM_ROOT   = Path("/data3/BIOIT_IMAGE/PESA_Fat/DATA/Visit-5-DIXON_PET-CT/DATA/DICOM")
# DEFAULT_NIFTI_ROOT   = Path("/data3/BIOIT_IMAGE/PESA_Fat/DATA/Visit-5-DIXON_PET-CT/DATA/NIFTI")
# DEFAULT_RESULTS_ROOT = Path("/data3/BIOIT_IMAGE/PESA_Fat/DATA/Visit-5-DIXON_PET-CT/RESULTS")
DEFAULT_NIFTI_ROOT   = Path("/home/imarcoss/DATA/BioIT/PESA-Fat/NIFTI")
DEFAULT_RESULTS_ROOT = Path("/home/imarcoss/DATA/BioIT/PESA-Fat/RESULTS")
DEFAULT_MODEL_ROOT   = Path("/data3/BIOIT_IMAGE/References/TotalSegmentator_v2/")

DEFAULT_NVITK_SRC_DIR = Path("/data3/BIOIT_IMAGE/nvitk/src")
DEFAULT_SGE_SCRIPTS_DIR = Path("/data3/BIOIT_IMAGE/PESA_Fat/DATA/Visit-5-DIXON_PET-CT/SCRIPTS_CLUSTER")

CLUSTER_HOST_ALIASES: dict[str, str] = {'samwise': '10.149.80.48'}

SUBJECT_GLOB = "PESA*"

NIFTI_EXTS: tuple[str, ...] = (".nii.gz", ".nii")
"""Accepted NIfTI extensions, preferring compressed output."""


def resolve_nii(parent: Path, stem: str) -> Path:
    """Return the existing ``parent/stem.nii[.gz]``, preferring ``.nii.gz``."""
    for ext in NIFTI_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Neither {parent / (stem + '.nii.gz')} nor {parent / (stem + '.nii')} exist"
    )


def resolve_nii_optional(parent: Path, stem: str) -> Path | None:
    """Variant of :func:`resolve_nii` returning ``None`` instead of raising."""
    for ext in NIFTI_EXTS:
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_subjects(value: str | None) -> list[str] | None:
    """Turn ``"PESA001,PESA002"`` into a list (or ``None`` when empty)."""
    if not value:
        return None
    items = [s.strip() for s in value.split(",") if s.strip()]
    return items or None


@dataclass(frozen=True)
class BatchLayout:
    """Resolved directories for a specific batch run."""

    batch: str
    dicom_root: Path
    nifti_root: Path
    results_root: Path
    model_root: Path

    @property
    def dicom_dir(self) -> Path:
        return self.dicom_root / self.batch

    @property
    def nifti_dir(self) -> Path:
        return self.nifti_root / self.batch

    @property
    def results_dir(self) -> Path:
        return self.results_root / self.batch

    @property
    def model_dir(self) -> Path:
        return self.model_root

    def stage_dir(self, stage_name: str) -> Path:
        """Return ``results_dir / f"res_{stage_name}"``."""
        return self.results_dir / f"res_{stage_name}"

    def subject_nifti_dir(self, subject: str) -> Path:
        return self.nifti_dir / subject

    def subject_dicom_dir(self, subject: str) -> Path:
        return self.dicom_dir / subject

    def subject_nifti_dirs(self) -> list[Path]:
        """Return all ``PESA*`` subject directories under ``nifti_dir`` (sorted)."""
        if not self.nifti_dir.exists():
            return []
        return sorted(
            d for d in self.nifti_dir.glob(SUBJECT_GLOB) if d.is_dir()
        )

    def subject_dicom_dirs(self) -> list[Path]:
        """Return all ``PESA*`` subject directories under ``dicom_dir`` (sorted)."""
        if not self.dicom_dir.exists():
            return []
        return sorted(
            d for d in self.dicom_dir.glob(SUBJECT_GLOB) if d.is_dir()
        )

    def iter_subjects(self) -> Iterator[str]:
        """Yield subject names (directory basenames) from the NIfTI layout."""
        for d in self.subject_nifti_dirs():
            yield d.name


def default_submit_script_path(batch: str) -> Path:
    """Return ``SCRIPTS_CLUSTER/submit_<batch>.sh`` under :data:`DEFAULT_SGE_SCRIPTS_DIR`."""
    return DEFAULT_SGE_SCRIPTS_DIR / f"submit_{batch}.sh"


def layout(
    batch: str,
    *,
    dicom_root: Path | str | None = None,
    nifti_root: Path | str | None = None,
    results_root: Path | str | None = None,
    model_root: Path | str | None = None,
) -> BatchLayout:
    """Build a :class:`BatchLayout` for ``batch``, falling back to defaults."""
    return BatchLayout(
        batch=batch,
        dicom_root=Path(dicom_root) if dicom_root else DEFAULT_DICOM_ROOT,
        nifti_root=Path(nifti_root) if nifti_root else DEFAULT_NIFTI_ROOT,
        results_root=Path(results_root) if results_root else DEFAULT_RESULTS_ROOT,
        model_root=Path(model_root) if model_root else DEFAULT_MODEL_ROOT,
    )


__all__ = [
    "BatchLayout",
    "CLUSTER_HOST_ALIASES",
    "DEFAULT_DICOM_ROOT",
    "DEFAULT_MODEL_ROOT",
    "DEFAULT_NVITK_SRC_DIR",
    "DEFAULT_NIFTI_ROOT",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_SGE_SCRIPTS_DIR",
    "SUBJECT_GLOB",
    "NIFTI_EXTS",
    "default_submit_script_path",
    "layout",
    "parse_subjects",
    "resolve_nii",
    "resolve_nii_optional",
]
