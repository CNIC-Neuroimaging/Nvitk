"""Pipeline filesystem presets for the Napari data browser (local mode)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NIFTI_SUFFIXES = (".nii.gz", ".nii")
QVTPY_PIPELINE_BUNDLE_DIRS: tuple[str, ...] = ("eicab", "qvtpy")

def _dataset_config() -> Any:
    """The dataset config in force (its ``local_presets`` and ``cohort_dir_rules``)."""
    from nvitk.db.config.dataset_config import load_dataset_config

    return load_dataset_config()


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
    pesa_health_layout: bool = False
    #: Literal roots (``{"dicom", "nifti", "results"}``), for a pipeline without a config module.
    roots: tuple[tuple[str, str], ...] = ()


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
    pesa_health_layout: bool = False

    def subject_dicom_dir(self, subject: str) -> Path:
        """*subject*'s DICOM directory under this preset's DICOM root."""
        return self._subject_base(self.dicom_root, subject)

    def subject_nifti_dir(self, subject: str) -> Path:
        """*subject*'s NIfTI directory under this preset's NIfTI root."""
        return self._subject_base(self.nifti_root, subject)

    def subject_results_base(self, subject: str) -> Path:
        """Base results directory for *subject* (the batch subfolder for batch layouts)."""
        if self.layout == "batch" and self.batch:
            return self.results_root / self.batch
        return self.results_root

    def _subject_base(self, root: Path, subject: str) -> Path:
        """*subject*'s directory under *root*, nested under the current batch for batch layouts."""
        if self.layout == "batch" and self.batch:
            return root / self.batch / subject
        return root / subject


@dataclass(frozen=True)
class LocalAsset:
    """One openable file or DICOM directory for a subject."""

    kind: str  # dicom | nifti | results
    label: str
    path: Path


def _preset_from_config(entry: dict[str, Any]) -> PipelinePresetSpec:
    return PipelinePresetSpec(
        preset_id=str(entry["preset_id"]),
        label=str(entry.get("label") or entry["preset_id"]),
        config_module=str(entry.get("config_module") or ""),
        layout=str(entry.get("layout") or "flat"),
        subject_globs=tuple(str(g) for g in entry.get("subject_globs") or ("*",)),
        show_batch=bool(entry.get("show_batch", False)),
        default_batch=str(entry.get("default_batch") or ""),
        pesa_health_layout=bool(entry.get("pesa_health_layout", False)),
        roots=tuple((str(k), str(v)) for k, v in dict(entry.get("roots") or {}).items()),
    )


#: Pipelines the Data browser can walk on disk, from the dataset config's ``local_presets``.
PRESET_REGISTRY: tuple[PipelinePresetSpec, ...] = tuple(
    _preset_from_config(entry) for entry in _dataset_config().local_presets
)

# PESA-Health on disk:  NIFTI_ROOT / <cohort> / PESA* / *.nii.gz
#                    RESULTS_ROOT / <cohort> / res_* / PESA* / ...
PESA_HEALTH_SUBJECT_GLOBS: tuple[str, ...] = next(
    (p.subject_globs for p in PRESET_REGISTRY if p.pesa_health_layout), ("PESA*",)
)


def list_pipeline_preset_ids() -> list[str]:
    """Ids of every registered pipeline data-browser preset, in registry order."""
    return [p.preset_id for p in PRESET_REGISTRY]


def get_pipeline_preset(preset_id: str) -> PipelinePresetSpec:
    """Look up the registered :class:`PipelinePresetSpec` for *preset_id*; raises ``KeyError``
    (listing known ids) if unregistered."""
    for spec in PRESET_REGISTRY:
        if spec.preset_id == preset_id:
            return spec
    known = ", ".join(list_pipeline_preset_ids())
    raise KeyError(f"Unknown pipeline preset {preset_id!r}; known: {known}")


def _path_from_config(mod: Any, name: str) -> Path | None:
    """Read a path-valued attribute *name* off a pipeline config module *mod*, or ``None`` if unset."""
    val = getattr(mod, name, None)
    if val is None:
        return None
    p = Path(val).expanduser()
    return p


def _is_cohort_dir_name(name: str) -> bool:
    """Cohort folder under NIFTI/DICOM/RESULTS roots (not a subject), by the dataset config's
    ``cohort_dir_rules``."""
    return _dataset_config().is_cohort_dir_name(name)


def list_local_cohorts(
    *,
    nifti_root: Path | str,
    dicom_root = None,
    results_root = None,
) -> list[str]:
    """Discover cohort folders (e.g. ``202602_Week1``) at the roots of a PESA-Health tree."""
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
    """True if *cohort*'s NIfTI folder contains any PESA-Health subject directories."""
    base = nifti_root / cohort
    return bool(_list_subject_dirs(base, PESA_HEALTH_SUBJECT_GLOBS))


def resolve_pesa_health_batch(
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
    literal = dict(spec.roots)
    mod = importlib.import_module(spec.config_module) if spec.config_module else None

    def _default(key: str, attr: str) -> Path | None:
        if literal.get(key):
            return Path(literal[key]).expanduser()
        return _path_from_config(mod, attr) if mod is not None else None

    d_root = Path(dicom_root).expanduser() if dicom_root else _default("dicom", "DEFAULT_DICOM_ROOT")
    n_root = Path(nifti_root).expanduser() if nifti_root else _default("nifti", "DEFAULT_NIFTI_ROOT")
    r_root = Path(results_root).expanduser() if results_root else _default("results", "DEFAULT_RESULTS_ROOT")
    if d_root is None or n_root is None or r_root is None:
        raise ValueError(
            f"Pipeline {preset_id!r} is missing DEFAULT_DICOM_ROOT, DEFAULT_NIFTI_ROOT, "
            "or DEFAULT_RESULTS_ROOT in its config."
        )
    batch_val = (batch if batch is not None else spec.default_batch or "").strip() or None
    if spec.pesa_health_layout:
        batch_val = resolve_pesa_health_batch(
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
        pesa_health_layout=spec.pesa_health_layout,
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
    """True if *name* looks like a real subject directory (not junk) and matches any of *globs*."""
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
    """``(nifti, dicom, results)`` base directories, nested under the current batch for batch layouts."""
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
    """True if *path* has a ``.nii`` or ``.nii.gz`` extension."""
    name = path.name.lower()
    return name.endswith(".nii.gz") or name.endswith(".nii")


def _discover_dicom_assets(subject_dir: Path) -> list[LocalAsset]:
    """List DICOM assets under *subject_dir*: one per subdirectory (assumed one series each), or the
    directory itself if it directly contains files."""
    if not subject_dir.is_dir():
        return []
    subdirs = sorted(d for d in subject_dir.iterdir() if d.is_dir())
    if subdirs:
        return [LocalAsset("dicom", d.name, d) for d in subdirs]
    if any(p.is_file() for p in subject_dir.iterdir()):
        return [LocalAsset("dicom", subject_dir.name, subject_dir)]
    return []


def _discover_nifti_assets(subject_dir: Path, *, max_files: int = 200) -> list[LocalAsset]:
    """List every NIfTI file found anywhere under *subject_dir* (capped at *max_files*), labeled by
    their path relative to *subject_dir*."""
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
    """List *subject*'s pipeline output files under *roots* (batch-layout ``res_*`` stage
    subdirectories, including a ``per_subject`` fallback, or the flat results tree otherwise)."""
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
            bundle_dirs: set[str] = set()
            for bundle_name in QVTPY_PIPELINE_BUNDLE_DIRS:
                bundle = subj_dir / bundle_name
                if bundle.is_dir():
                    assets.append(LocalAsset("pipeline", bundle_name, bundle))
                    bundle_dirs.add(bundle_name)
            for path in sorted(subj_dir.rglob("*")):
                if not path.is_file() or not _is_nifti_file(path):
                    continue
                try:
                    top = path.relative_to(subj_dir).parts[0]
                except ValueError:
                    top = ""
                if top in bundle_dirs:
                    continue
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
