"""Pipeline filesystem presets for the Napari data browser (local mode)."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NIFTI_SUFFIXES = (".nii.gz", ".nii")

# PESA-Fat on disk:  NIFTI_ROOT / <cohort> / PESA* / *.nii.gz
#                    RESULTS_ROOT / <cohort> / res_* / PESA* / ...
PESA_FAT_SUBJECT_GLOBS: tuple[str, ...] = ("PESA*",)

_COHORT_WEEK_RE = re.compile(r"^\d{6}_Week\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class PipelinePresetSpec:
    """Registered pipeline with default data roots from its config module."""

    preset_id: str
    label: str
    config_module: str
    layout: str  # "flat" | "batch"
    subject_globs: tuple[str, ...]
    show_batch: bool
    default_batch: str
    pesa_fat_layout: bool = False


@dataclass
class PipelineRoots:
    """Resolved roots for browsing a pipeline on disk."""

    preset_id: str
    dicom_root: Path
    nifti_root: Path
    results_root: Path
    layout: str
    batch: str | None
    subject_globs: tuple[str, ...]
    pesa_fat_layout: bool = False

    def subject_dicom_dir(self, subject: str) -> Path:
        return self._subject_base(self.dicom_root, subject)

    def subject_nifti_dir(self, subject: str) -> Path:
        return self._subject_base(self.nifti_root, subject)

    def subject_results_base(self, subject: str) -> Path:
        if self.layout == "batch" and self.batch:
            return self.results_root / self.batch
        return self.results_root

    def _subject_base(self, root: Path, subject: str) -> Path:
        if self.layout == "batch" and self.batch:
            return root / self.batch / subject
        return root / subject


@dataclass(frozen=True)
class LocalAsset:
    """One openable file or DICOM directory for a subject."""

    kind: str  # dicom | nifti | results
    label: str
    path: Path


PRESET_REGISTRY: tuple[PipelinePresetSpec, ...] = (
    PipelinePresetSpec(
        "qvtpy",
        "QVTpy (4D flow)",
        "nvitk.pipes.qvtpy.config",
        layout="flat",
        subject_globs=("PESA*",),
        show_batch=False,
        default_batch="",
    ),
    PipelinePresetSpec(
        "bbtpy",
        "BBTpy (brain TOF)",
        "nvitk.pipes.bbtpy.config",
        layout="flat",
        subject_globs=("PESA*",),
        show_batch=False,
        default_batch="",
    ),
    PipelinePresetSpec(
        "pesa_fat",
        "PESA-Fat (Dixon + CT-PET)",
        "nvitk.pipes.pesa_fat.common.paths",
        layout="batch",
        subject_globs=PESA_FAT_SUBJECT_GLOBS,
        show_batch=True,
        default_batch="",
        pesa_fat_layout=True,
    ),
    PipelinePresetSpec(
        "pesa_fat_ctpet",
        "PESA-Fat CT-PET v5",
        "nvitk.pipes.pesa_fat.common.paths",
        layout="batch",
        subject_globs=PESA_FAT_SUBJECT_GLOBS,
        show_batch=True,
        default_batch="",
        pesa_fat_layout=True,
    ),
    PipelinePresetSpec(
        "pesa_fat_dixon",
        "PESA-Fat Dixon v5",
        "nvitk.pipes.pesa_fat.common.paths",
        layout="batch",
        subject_globs=PESA_FAT_SUBJECT_GLOBS,
        show_batch=True,
        default_batch="",
        pesa_fat_layout=True,
    ),
)


def list_pipeline_preset_ids() -> list[str]:
    return [p.preset_id for p in PRESET_REGISTRY]


def get_pipeline_preset(preset_id: str) -> PipelinePresetSpec:
    for spec in PRESET_REGISTRY:
        if spec.preset_id == preset_id:
            return spec
    known = ", ".join(list_pipeline_preset_ids())
    raise KeyError(f"Unknown pipeline preset {preset_id!r}; known: {known}")


def _path_from_config(mod: Any, name: str) -> Path | None:
    val = getattr(mod, name, None)
    if val is None:
        return None
    p = Path(val).expanduser()
    return p


def _is_cohort_dir_name(name: str) -> bool:
    """Cohort folder under NIFTI/DICOM/RESULTS roots (not a PESA subject)."""
    if not name or name.startswith(".") or name.startswith("res_"):
        return False
    if name.upper().startswith("PESA"):
        return False
    if _COHORT_WEEK_RE.match(name):
        return True
    if name.upper().startswith("DIXON_"):
        return True
    if name.startswith("Visit-"):
        return True
    return False


def list_local_cohorts(
    *,
    nifti_root: Path | str,
    dicom_root = None,
    results_root = None,
) -> list[str]:
    """Discover cohort folders (e.g. ``202602_Week1``) at the roots of a PESA-Fat tree."""
    found = set()
    for raw in (nifti_root, dicom_root, results_root):
        if raw is None:
            continue
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_dir() and _is_cohort_dir_name(path.name):
                found.add(path.name)
    return sorted(found)


def _cohort_has_pesa_subjects(nifti_root: Path, cohort: str) -> bool:
    base = nifti_root / cohort
    return bool(_list_subject_dirs(base, PESA_FAT_SUBJECT_GLOBS))


def resolve_pesa_fat_batch(
    *,
    nifti_root: Path,
    dicom_root: Path,
    results_root: Path,
    batch: str | None,
) -> str | None:
    """Pick a cohort folder when batch is missing or points at a non-existent path."""
    batch_val = (batch or "").strip() or None
    if batch_val and (nifti_root / batch_val).is_dir():
        return batch_val
    if batch_val and (results_root / batch_val).is_dir():
        return batch_val

    cohorts = list_local_cohorts(
        nifti_root=nifti_root,
        dicom_root=dicom_root,
        results_root=results_root,
    )
    for cohort in cohorts:
        if _cohort_has_pesa_subjects(nifti_root, cohort):
            return cohort
    return cohorts[0] if cohorts else batch_val


def load_preset_roots(
    preset_id: str,
    *,
    dicom_root = None,
    nifti_root = None,
    results_root = None,
    batch = None,
) -> PipelineRoots:
    """Load default roots from a pipeline config, with optional overrides."""
    spec = get_pipeline_preset(preset_id)
    mod = importlib.import_module(spec.config_module)
    d_root = Path(dicom_root).expanduser() if dicom_root else _path_from_config(mod, "DEFAULT_DICOM_ROOT")
    n_root = Path(nifti_root).expanduser() if nifti_root else _path_from_config(mod, "DEFAULT_NIFTI_ROOT")
    r_root = Path(results_root).expanduser() if results_root else _path_from_config(mod, "DEFAULT_RESULTS_ROOT")
    if d_root is None or n_root is None or r_root is None:
        raise ValueError(
            f"Pipeline {preset_id!r} is missing DEFAULT_DICOM_ROOT, DEFAULT_NIFTI_ROOT, "
            "or DEFAULT_RESULTS_ROOT in its config."
        )
    batch_val = (batch if batch is not None else spec.default_batch or "").strip() or None
    if spec.pesa_fat_layout:
        batch_val = resolve_pesa_fat_batch(
            nifti_root=n_root,
            dicom_root=d_root,
            results_root=r_root,
            batch=batch_val,
        )
    elif spec.layout == "batch" and not batch_val:
        batch_val = (spec.default_batch or "").strip() or None

    return PipelineRoots(
        preset_id=spec.preset_id,
        dicom_root=d_root,
        nifti_root=n_root,
        results_root=r_root,
        layout=spec.layout,
        batch=batch_val,
        subject_globs=spec.subject_globs,
        pesa_fat_layout=spec.pesa_fat_layout,
    )


def _is_subject_dir_name(name: str) -> bool:
    """Exclude pipeline stage folders and hidden entries."""
    if not name or name.startswith("."):
        return False
    if name.startswith("res_"):
        return False
    if name in ("per_subject", "assets"):
        return False
    return True


def _matches_subject_globs(name: str, globs: tuple[str, ...]) -> bool:
    if not _is_subject_dir_name(name):
        return False
    return any(Path(name).match(pattern) for pattern in globs)


def _list_subject_dirs(root: Path, globs: tuple[str, ...]) -> set[str]:
    """PESA* (etc.) folder names directly under *root*."""
    if not root.is_dir():
        return set()
    found = set()
    for pattern in globs:
        for path in root.glob(pattern):
            if path.is_dir() and _matches_subject_globs(path.name, globs):
                found.add(path.name)
    return found


def _batch_bases(roots: PipelineRoots) -> tuple[Path, Path, Path]:
    if roots.layout == "batch" and roots.batch:
        return (
            roots.nifti_root / roots.batch,
            roots.dicom_root / roots.batch,
            roots.results_root / roots.batch,
        )
    return roots.nifti_root, roots.dicom_root, roots.results_root


def _subjects_from_results_tree(r_base: Path, globs: tuple[str, ...]) -> set[str]:
    """Collect PESA* under ``<cohort>/res_*`` (skip ``per_subject`` leaves)."""
    found = set()
    if not r_base.is_dir():
        return found
    for stage in sorted(r_base.glob("res_*")):
        if not stage.is_dir():
            continue
        found.update(_list_subject_dirs(stage, globs))
        per_subj = stage / "per_subject"
        if per_subj.is_dir():
            found.update(_list_subject_dirs(per_subj, globs))
    return found


def list_local_subjects(
    roots: PipelineRoots,
    *,
    include_dicom = True,
    include_nifti = True,
    include_results = True,
) -> list[str]:
    """Discover subject folder names from enabled roots (independent of asset filters)."""
    n_base, d_base, r_base = _batch_bases(roots)
    globs = roots.subject_globs
    found = set()

    if include_nifti:
        found.update(_list_subject_dirs(n_base, globs))
    if include_dicom:
        found.update(_list_subject_dirs(d_base, globs))
    if include_results:
        if roots.layout == "batch" and roots.batch:
            found.update(_subjects_from_results_tree(r_base, globs))
        else:
            found.update(_subjects_from_results_tree(roots.results_root, globs))

    return sorted(found)


def _is_nifti_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii.gz") or name.endswith(".nii")


def _discover_dicom_assets(subject_dir: Path) -> list[LocalAsset]:
    if not subject_dir.is_dir():
        return []
    subdirs = sorted(d for d in subject_dir.iterdir() if d.is_dir())
    if subdirs:
        return [LocalAsset("dicom", d.name, d) for d in subdirs]
    if any(p.is_file() for p in subject_dir.iterdir()):
        return [LocalAsset("dicom", subject_dir.name, subject_dir)]
    return []


def _discover_nifti_assets(subject_dir: Path, *, max_files: int = 200) -> list[LocalAsset]:
    if not subject_dir.is_dir():
        return []
    assets = []
    for path in sorted(subject_dir.rglob("*")):
        if not path.is_file() or not _is_nifti_file(path):
            continue
        try:
            rel = path.relative_to(subject_dir)
            label = str(rel)
        except ValueError:
            label = path.name
        assets.append(LocalAsset("nifti", label, path))
        if len(assets) >= max_files:
            break
    return assets


def _discover_results_assets(roots: PipelineRoots, subject: str) -> list[LocalAsset]:
    assets = []
    base = roots.subject_results_base(subject)

    if roots.layout == "batch":
        stage_dirs = sorted(d for d in base.glob("res_*") if d.is_dir())
        for stage in stage_dirs:
            subj_dir = stage / subject
            if not subj_dir.is_dir():
                per_subj = stage / "per_subject" / subject
                if per_subj.is_dir():
                    subj_dir = per_subj
                else:
                    continue
            for path in sorted(subj_dir.rglob("*")):
                if path.is_file() and _is_nifti_file(path):
                    rel = path.relative_to(subj_dir)
                    assets.append(
                        LocalAsset("results", f"{stage.name}/{rel}", path)
                    )
    else:
        subj_dir = base / subject
        if subj_dir.is_dir():
            for path in sorted(subj_dir.rglob("*")):
                if path.is_file() and _is_nifti_file(path):
                    try:
                        rel = path.relative_to(subj_dir)
                        label = str(rel)
                    except ValueError:
                        label = path.name
                    assets.append(LocalAsset("results", label, path))

    return assets


def list_local_assets(
    roots: PipelineRoots,
    subject: str,
    *,
    include_dicom = True,
    include_nifti = True,
    include_results = True,
) -> list[LocalAsset]:
    """List openable DICOM series, NIfTI files, and result masks for *subject*."""
    out = []
    if include_dicom:
        out.extend(_discover_dicom_assets(roots.subject_dicom_dir(subject)))
    if include_nifti:
        out.extend(_discover_nifti_assets(roots.subject_nifti_dir(subject)))
    if include_results:
        out.extend(_discover_results_assets(roots, subject))
    return out
