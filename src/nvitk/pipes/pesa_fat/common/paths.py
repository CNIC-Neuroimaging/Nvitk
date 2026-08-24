"""Filesystem layout helpers for the PESA-Fat CT-PET / Dixon batches.

The pipelines operate on a flat on-disk tree rooted at three top-level
directories:

* ``DICOM_ROOT / <batch> / PESA* / ...``   (inputs)
* ``NIFTI_ROOT / <batch> / PESA* / ...``   (stage 0 outputs + inputs to later stages)
* ``RESULTS_ROOT / <batch> / res_<stage>/PESA* / ...``   (later-stage outputs)

All pipelines derive their per-subject paths from :class:`BatchLayout`.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from nvitk.cluster import sge_json as _sj
from nvitk.core import config_paths, lazy_config

#: ``sge.json`` section holding this pipeline's data roots.
PIPELINE_PATHS_ID = "pesa_fat_paths"


def _pipe_paths() -> dict:
    """The ``pipelines.pesa_fat_paths`` block, read fresh (and cached) on each use."""
    return _sj.pipeline_section(PIPELINE_PATHS_ID)


def _opt_root(key: str):
    """The configured root *key* as a path, or ``None`` when unset.

    Does not raise: these names are read at import time as Click option defaults, so an
    unconfigured machine must still be able to print ``--help``. The actionable error comes
    from :func:`layout_local` / :func:`layout_cluster` when the value is needed.
    """
    raw = _pipe_paths().get(key)
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expanduser(str(raw).strip()))


_RESOLVERS: dict[str, lazy_config.Resolver] = {
    "DEFAULT_DICOM_ROOT": lambda: _opt_root("cluster_dicom_root"),
    "DEFAULT_NIFTI_ROOT": lambda: _opt_root("cluster_nifti_root"),
    "DEFAULT_RESULTS_ROOT": lambda: _opt_root("cluster_results_root"),
    "DEFAULT_MODEL_ROOT": lambda: _opt_root("cluster_model_root"),
    "LOCAL_DEFAULT_DICOM_ROOT": lambda: _opt_root("local_dicom_root"),
    "LOCAL_DEFAULT_NIFTI_ROOT": lambda: _opt_root("local_nifti_root"),
    "LOCAL_DEFAULT_RESULTS_ROOT": lambda: _opt_root("local_results_root"),
    "LOCAL_DEFAULT_MODEL_ROOT": lambda: _opt_root("local_model_root"),
    "DEFAULT_NVITK_SRC_DIR": lambda: _sj.resolve_nvitk_src_dir(),
    "DEFAULT_SGE_SCRIPTS_DIR": lambda: (
        _opt_path(_sj.paths_section().get("sge_scripts_dir"))
        or Path(tempfile.gettempdir()) / "nvitk-sge" / "scripts"
    ),
    "CLUSTER_HOST_ALIASES": lambda: _sj.merge_cluster_host_aliases(
        {}, _sj.paths_section(), _pipe_paths()
    ),
}


def _opt_path(value):
    """A configured value as an expanded path, or ``None`` when unset."""
    if value is None or not str(value).strip():
        return None
    return Path(os.path.expanduser(str(value).strip()))


__getattr__, __dir__ = lazy_config.module_getattr(_RESOLVERS, module_name=__name__)

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
        """This batch's DICOM directory."""
        return self.dicom_root / self.batch

    @property
    def nifti_dir(self) -> Path:
        """This batch's NIfTI directory."""
        return self.nifti_root / self.batch

    @property
    def results_dir(self) -> Path:
        """This batch's results directory."""
        return self.results_root / self.batch

    @property
    def model_dir(self) -> Path:
        """The TotalSegmentator model root (not batch-scoped)."""
        return self.model_root

    def stage_dir(self, stage_name: str) -> Path:
        """Return ``results_dir / f"res_{stage_name}"``."""
        return self.results_dir / f"res_{stage_name}"

    def subject_nifti_dir(self, subject: str) -> Path:
        """*subject*'s NIfTI directory within this batch."""
        return self.nifti_dir / subject

    def subject_dicom_dir(self, subject: str) -> Path:
        """*subject*'s DICOM directory within this batch."""
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
        dicom_root=_cluster_path_from_config(
            "cluster_dicom_root", fallback=Path(dicom_root) if dicom_root else None
        ),
        nifti_root=_cluster_path_from_config(
            "cluster_nifti_root", fallback=Path(nifti_root) if nifti_root else None
        ),
        results_root=_cluster_path_from_config(
            "cluster_results_root", fallback=Path(results_root) if results_root else None
        ),
        model_root=_cluster_path_from_config(
            "cluster_model_root", fallback=Path(model_root) if model_root else None
        ),
    )


def _local_path_from_config(key: str, *, fallback: Path | None) -> Path:
    """Resolve workstation root: CLI flag > ``local_*`` in sge.json > :data:`LOCAL_DEFAULT_*`."""
    if fallback is not None:
        return Path(fallback)
    raw = _pipe_paths().get(f"local_{key}")
    return Path(os.path.expanduser(str(config_paths.require(
        raw,
        key=f"pipelines.{PIPELINE_PATHS_ID}.local_{key}",
        hint="Set it, or pass the matching --*-root flag.",
    )).strip()))


def layout_local(
    batch: str,
    *,
    dicom_root: Path | str | None = None,
    nifti_root: Path | str | None = None,
    results_root: Path | str | None = None,
    model_root: Path | str | None = None,
) -> BatchLayout:
    """Workstation layout for XNAT download and ``--submit local``.

    Uses ``pipelines.pesa_fat_paths.local_*`` from ``.nvitk/sge.json`` when CLI
    ``--*-root`` flags are omitted (never the cluster ``cluster_*`` / ``DEFAULT_*`` paths).
    """
    return BatchLayout(
        batch=batch,
        dicom_root=_local_path_from_config(
            "dicom_root", fallback=Path(dicom_root) if dicom_root else None
        ),
        nifti_root=_local_path_from_config(
            "nifti_root", fallback=Path(nifti_root) if nifti_root else None
        ),
        results_root=_local_path_from_config(
            "results_root", fallback=Path(results_root) if results_root else None
        ),
        model_root=_local_path_from_config(
            "model_root", fallback=Path(model_root) if model_root else None
        ),
    )


def _cluster_path_from_config(key: str, *, fallback: Path | None) -> Path:
    """Resolve a cluster path setting *key*: the raw ``sge.json`` value if set, else *fallback*, else
    the hardcoded cluster default."""
    raw = _pipe_paths().get(key)
    if raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    if fallback is not None:
        return Path(fallback)
    return Path(os.path.expanduser(str(config_paths.require(
        None,
        key=f"pipelines.{PIPELINE_PATHS_ID}.{key}",
        hint="Set it, or pass the matching --*-root flag.",
    ))))


def layout_cluster(
    batch: str,
    *,
    dicom_root: Path | str | None = None,
    nifti_root: Path | str | None = None,
    results_root: Path | str | None = None,
    model_root: Path | str | None = None,
) -> BatchLayout:
    """Cluster-side :class:`BatchLayout` for SGE binds and SFTP upload targets.

    Reads ``pipelines.pesa_fat_paths`` from ``.nvitk/sge.json`` when set;
    otherwise falls back to CLI ``--*-root`` flags or :data:`DEFAULT_*_ROOT`.
    """
    ct_pet = _sj.pipeline_section("pesa_fat_ct_pet")
    model_fb = model_root
    if model_fb is None:
        raw_model = ct_pet.get("default_sge_model_root")
        if raw_model is not None and str(raw_model).strip():
            model_fb = Path(os.path.expanduser(str(raw_model).strip()))
    return BatchLayout(
        batch=batch,
        dicom_root=_cluster_path_from_config(
            "cluster_dicom_root", fallback=Path(dicom_root) if dicom_root else None
        ),
        nifti_root=_cluster_path_from_config(
            "cluster_nifti_root", fallback=Path(nifti_root) if nifti_root else None
        ),
        results_root=_cluster_path_from_config(
            "cluster_results_root", fallback=Path(results_root) if results_root else None
        ),
        model_root=_cluster_path_from_config(
            "cluster_model_root", fallback=Path(model_fb) if model_fb else None
        ),
    )


def group_subjects_by_batch(
    download_map: dict[str, tuple[str, dict]],
) -> dict[str, list[str]]:
    """Group XNAT download results ``{subject: (batch, paths)}`` by batch name."""
    by_batch: dict[str, list[str]] = {}
    for subject, (batch_name, _paths) in download_map.items():
        by_batch.setdefault(str(batch_name), []).append(str(subject))
    return {b: sorted(subs) for b, subs in sorted(by_batch.items())}


__all__ = [
    "BatchLayout",
    "CLUSTER_HOST_ALIASES",
    "DEFAULT_DICOM_ROOT",
    "LOCAL_DEFAULT_DICOM_ROOT",
    "LOCAL_DEFAULT_MODEL_ROOT",
    "LOCAL_DEFAULT_NIFTI_ROOT",
    "LOCAL_DEFAULT_RESULTS_ROOT",
    "DEFAULT_MODEL_ROOT",
    "DEFAULT_NVITK_SRC_DIR",
    "DEFAULT_NIFTI_ROOT",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_SGE_SCRIPTS_DIR",
    "SUBJECT_GLOB",
    "NIFTI_EXTS",
    "default_submit_script_path",
    "group_subjects_by_batch",
    "layout",
    "layout_cluster",
    "layout_local",
    "parse_subjects",
    "resolve_nii",
    "resolve_nii_optional",
]
