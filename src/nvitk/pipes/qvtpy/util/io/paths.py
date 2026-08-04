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
DEFAULT_TOTALSEG_MODEL_ROOT = Path("/data3/BIOIT_IMAGE/References/TotalSegmentator_v2/")

# Workstation defaults (local machine).
LOCAL_DEFAULT_DICOM_ROOT = Path("/home/imarcoss/DATA/LabVF/PESA-Brain/DATA/DICOM")
LOCAL_DEFAULT_NIFTI_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/DATA/NIFTI")
# LOCAL_DEFAULT_RESULTS_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/LabVF/PESA-Brain/RESULTS/res_QVTPy")
LOCAL_DEFAULT_RESULTS_ROOT = Path("/home/imarcoss/DATA/LabVF/res_QVTPy")
LOCAL_DEFAULT_TOTALSEG_MODEL_ROOT = Path("/home/imarcoss/NetVolumes/LAB_MCC/ai_models/imaging/TotalSegmentator/v2.0.0")

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
        """*subject*'s DICOM directory under this context's DICOM root."""
        return self.dicom_root / subject

    def subject_nifti_dir(self, subject: str) -> Path:
        """*subject*'s NIfTI directory under this context's NIfTI root."""
        return self.nifti_root / subject


def _local_path_from_config(key: str, *, fallback: Path | None) -> Path:
    """Resolve a local-workstation path setting *key* (e.g. ``"nifti_root"``): *fallback* if given,
    else the ``local_<key>`` value from ``sge.json``, else the hardcoded local default."""
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
        "model_root": LOCAL_DEFAULT_TOTALSEG_MODEL_ROOT,
    }[key]


def _cluster_path_from_config(key: str, *, fallback: Path | None) -> Path:
    """Resolve a cluster path setting *key*: the ``cluster_<key>`` value from ``sge.json`` if set,
    else *fallback*, else the hardcoded cluster default."""
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
        "model_root": DEFAULT_TOTALSEG_MODEL_ROOT,
    }[key]


def _local_totalseg_model_from_config(*, fallback: Path | None) -> Path:
    """Resolve the local-workstation TotalSegmentator model root."""
    return _local_path_from_config("model_root", fallback=fallback)


def _cluster_totalseg_model_from_config(*, fallback: Path | None) -> Path:
    """Resolve the cluster TotalSegmentator model root."""
    return _cluster_path_from_config("model_root", fallback=fallback)


def resolve_totalseg_model_dir(
    *,
    model_dir: Path | None = None,
    prefer_cluster: bool | None = None,
) -> Path:
    """Return TotalSegmentator weights root (qvtpy cluster/local layout)."""
    if model_dir is not None:
        return Path(model_dir).expanduser().resolve()
    env_home = os.environ.get("TOTALSEG_HOME_DIR", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    if prefer_cluster is None:
        prefer_cluster = os.environ.get("NVITK_CLUSTER", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
    if prefer_cluster:
        return _cluster_totalseg_model_from_config(fallback=None)
    return _local_totalseg_model_from_config(fallback=None)


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
    "DEFAULT_TOTALSEG_MODEL_ROOT",
    "LOCAL_DEFAULT_DICOM_ROOT",
    "LOCAL_DEFAULT_NIFTI_ROOT",
    "LOCAL_DEFAULT_RESULTS_ROOT",
    "LOCAL_DEFAULT_TOTALSEG_MODEL_ROOT",
    "QvtpyPaths",
    "layout_cluster",
    "layout_local",
    "resolve_totalseg_model_dir",
]
