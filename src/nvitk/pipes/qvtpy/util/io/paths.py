"""Filesystem layout helpers for qvtpy (local workstation vs cluster).

Every root comes from ``sge.json``'s ``pipelines.qvtpy_paths`` section — there are no
installation paths in this file. A root that is needed but not configured raises
:class:`~nvitk.core.config_paths.ConfigError` naming the key and the search path, rather than
silently resolving to somebody else's filesystem.

Configuration is read on use, not on import, so ``--config-dir`` and a mid-process
:func:`~nvitk.core.config_paths.set_config_dir` are both honoured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from nvitk.cluster import sge_json as _sj
from nvitk.core import config_paths, lazy_config

#: ``sge.json`` section holding this pipeline's data roots.
PIPELINE_PATHS_ID = "qvtpy_paths"


def _pipe_paths() -> dict:
    """The ``pipelines.qvtpy_paths`` block, read fresh (and cached) on each use."""
    return _sj.pipeline_section(PIPELINE_PATHS_ID)


def _resolve_host_aliases() -> dict[str, str]:
    """Cluster host aliases from ``paths.cluster_host_aliases`` plus this pipeline's overrides."""
    return _sj.merge_cluster_host_aliases({}, _sj.paths_section(), _pipe_paths())


def _opt_root(key: str):
    """The configured root *key* as a :class:`~pathlib.Path`, or ``None`` if it is unset.

    Deliberately does not raise. These names are read at import time as Click option defaults
    (``default=cfg.DEFAULT_NIFTI_ROOT``), so raising here would break ``--help`` on a machine
    with no configuration. ``None`` becomes "no default", and the error — naming the key and
    the search path — is raised by :func:`layout_local` / :func:`layout_cluster` when the value
    is actually needed to do work.
    """
    raw = _pipe_paths().get(key)
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expanduser(str(raw).strip()))


#: Roots exposed as module attributes for backwards compatibility. The *values* live only in
#: ``sge.json``; these are accessors, not defaults.
_RESOLVERS: dict[str, lazy_config.Resolver] = {
    "CLUSTER_HOST_ALIASES": _resolve_host_aliases,
    "DEFAULT_DICOM_ROOT": lambda: _opt_root("cluster_dicom_root"),
    "DEFAULT_NIFTI_ROOT": lambda: _opt_root("cluster_nifti_root"),
    "DEFAULT_RESULTS_ROOT": lambda: _opt_root("cluster_results_root"),
    "DEFAULT_TOTALSEG_MODEL_ROOT": lambda: _opt_root("cluster_model_root"),
    "LOCAL_DEFAULT_DICOM_ROOT": lambda: _opt_root("local_dicom_root"),
    "LOCAL_DEFAULT_NIFTI_ROOT": lambda: _opt_root("local_nifti_root"),
    "LOCAL_DEFAULT_RESULTS_ROOT": lambda: _opt_root("local_results_root"),
    "LOCAL_DEFAULT_TOTALSEG_MODEL_ROOT": lambda: _opt_root("local_model_root"),
}

__getattr__, __dir__ = lazy_config.module_getattr(_RESOLVERS, module_name=__name__)


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
    """A local-workstation root: *fallback* (the CLI flag) if given, else ``local_<key>``.

    The CLI flag wins here. Under ``--submit sge`` the opposite holds — see
    :func:`_cluster_path_from_config`.
    """
    if fallback is not None:
        return Path(fallback)
    raw = _pipe_paths().get(f"local_{key}")
    return Path(os.path.expanduser(str(config_paths.require(
        raw,
        key=f"pipelines.{PIPELINE_PATHS_ID}.local_{key}",
        hint="Set it, or pass the matching --*-root flag.",
    )).strip()))


def _cluster_path_from_config(key: str, *, fallback: Path | None) -> Path:
    """Resolve a cluster path setting *key* (e.g. ``"nifti_root"``): the ``cluster_<key>`` value
    from ``sge.json`` if set, else *fallback*, else the hardcoded cluster default.

    *key* is unprefixed, matching :func:`_local_path_from_config`. The prefix is added here.
    """
    raw = _pipe_paths().get(f"cluster_{key}")
    if raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    if fallback is not None:
        return Path(fallback)
    return Path(os.path.expanduser(str(config_paths.require(
        None,
        key=f"pipelines.{PIPELINE_PATHS_ID}.cluster_{key}",
        hint="Set it, or pass the matching --*-root flag.",
    ))))


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
            "dicom_root", fallback=Path(dicom_root) if dicom_root else None
        ),
        nifti_root=_cluster_path_from_config(
            "nifti_root", fallback=Path(nifti_root) if nifti_root else None
        ),
        results_root=_cluster_path_from_config(
            "results_root", fallback=Path(results_root) if results_root else None
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
