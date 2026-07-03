"""Filesystem layout helpers for qvtpy (local workstation vs cluster)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from nvitk.cluster import sge_json as _sj

# Cluster defaults (SGE binds).
DEFAULT_DICOM_ROOT = Path("/data_lab_MCC/imarcoss/LabMCC/DATA/DICOM")
DEFAULT_NIFTI_ROOT = Path("/data_lab_MCC/imarcoss/LabMCC/DATA/NIFTI")
DEFAULT_RESULTS_ROOT = Path("/data_lab_MCC/imarcoss/LabMCC/RESULTS/QVTPy")

# Workstation defaults (local machine).
LOCAL_DEFAULT_DICOM_ROOT = Path("/home/imarcoss/DATA/LabVF/PESA-Brain/DATA/DICOM")
LOCAL_DEFAULT_NIFTI_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DATA/NIFTI")
LOCAL_DEFAULT_RESULTS_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/RESULTS/res_QVTPy")

_ppaths = _sj.paths_section()
_ppipe_paths = _sj.pipeline_section("qvtpy_paths")

CLUSTER_HOST_ALIASES: dict[str, str] = {"samwise": "10.149.80.48"}
CLUSTER_HOST_ALIASES = _sj.merge_cluster_host_aliases(
    CLUSTER_HOST_ALIASES, _ppaths, _ppipe_paths
)


@dataclass(frozen=True)
class QvtpyPaths:
    """Resolved DICOM / NIfTI / results roots for one execution context."""

    dicom_root: Path
    nifti_root: Path
    results_root: Path

    def subject_dicom_dir(self, subject: str) -> Path:
        return self.dicom_root / subject

    def subject_nifti_dir(self, subject: str) -> Path:
        return self.nifti_root / subject


def _local_path_from_config(key: str, *, fallback: Path | None) -> Path:
    config_key = f"local_{key}"
    raw = _ppipe_paths.get(config_key)
    if fallback is not None:
        return Path(fallback)
    if raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    return {
        "dicom_root": LOCAL_DEFAULT_DICOM_ROOT,
        "nifti_root": LOCAL_DEFAULT_NIFTI_ROOT,
        "results_root": LOCAL_DEFAULT_RESULTS_ROOT,
    }[key]


def _cluster_path_from_config(key: str, *, fallback: Path | None) -> Path:
    config_key = f"cluster_{key}"
    raw = _ppipe_paths.get(config_key)
    if raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    if fallback is not None:
        return Path(fallback)
    return {
        "cluster_dicom_root": DEFAULT_DICOM_ROOT,
        "cluster_nifti_root": DEFAULT_NIFTI_ROOT,
        "cluster_results_root": DEFAULT_RESULTS_ROOT,
    }[key]


def layout_local(
    *,
    dicom_root: Path | str | None = None,
    nifti_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> QvtpyPaths:
    """Workstation layout for XNAT download and ``--submit local``."""
    return QvtpyPaths(
        dicom_root=_local_path_from_config(
            "dicom_root", fallback=Path(dicom_root) if dicom_root else None
        ),
        nifti_root=_local_path_from_config(
            "nifti_root", fallback=Path(nifti_root) if nifti_root else None
        ),
        results_root=_local_path_from_config(
            "results_root", fallback=Path(results_root) if results_root else None
        ),
    )


def layout_cluster(
    *,
    dicom_root: Path | str | None = None,
    nifti_root: Path | str | None = None,
    results_root: Path | str | None = None,
) -> QvtpyPaths:
    """Cluster-side layout for SGE Singularity binds."""
    return QvtpyPaths(
        dicom_root=_cluster_path_from_config(
            "cluster_dicom_root", fallback=Path(dicom_root) if dicom_root else None
        ),
        nifti_root=_cluster_path_from_config(
            "cluster_nifti_root", fallback=Path(nifti_root) if nifti_root else None
        ),
        results_root=_cluster_path_from_config(
            "cluster_results_root", fallback=Path(results_root) if results_root else None
        ),
    )


__all__ = [
    "CLUSTER_HOST_ALIASES",
    "DEFAULT_DICOM_ROOT",
    "DEFAULT_NIFTI_ROOT",
    "DEFAULT_RESULTS_ROOT",
    "LOCAL_DEFAULT_DICOM_ROOT",
    "LOCAL_DEFAULT_NIFTI_ROOT",
    "LOCAL_DEFAULT_RESULTS_ROOT",
    "QvtpyPaths",
    "layout_cluster",
    "layout_local",
]
