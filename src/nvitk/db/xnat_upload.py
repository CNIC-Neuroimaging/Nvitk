"""XNAT experiment-level resource upload helpers (pipeline-agnostic)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from nvitk.core.logger import Logger
from nvitk.db.xnat import _coalesce_attr, classify_scan

log = Logger()

DEFAULT_QVTPY_SCAN_SEQUENCES: frozenset[str] = frozenset(
    {"TOF", "4DFLOW_AP", "4DFLOW_RL", "4DFLOW_FH"}
)

_UPLOAD_IGNORE_DIR_NAMES = frozenset({"__pycache__", ".git", ".ipynb_checkpoints"})
_UPLOAD_IGNORE_SUFFIXES = (".tmp", ".pyc", ".pyo")


def _resolve_subject(project: Any, label: str) -> Any | None:
    subjects_map = getattr(project, "subjects", None) or {}
    if label in subjects_map:
        return subjects_map[label]
    for _key, subj in subjects_map.items():
        uid = str(_coalesce_attr(subj, "label", "id", "name") or _key)
        if uid == label:
            return subj
    return None


def _experiment_label(experiment: Any) -> str:
    return str(_coalesce_attr(experiment, "label", "id") or "").strip()


def _experiment_date(experiment: Any) -> Any:
    import pandas as pd

    return pd.to_datetime(_coalesce_attr(experiment, "date"), errors="coerce")


def _classify_experiment_sequences(
    experiment: Any,
    *,
    prefer_sequences: Iterable[str],
) -> set[str]:
    seq_set = {str(s).strip().upper() for s in prefer_sequences if str(s).strip()}
    found: set[str] = set()
    for scan in getattr(experiment, "scans", {}).values():
        series_description = str(
            _coalesce_attr(scan, "series_description", "type", "label") or ""
        )
        quality = str(_coalesce_attr(scan, "quality") or "")
        classification = classify_scan(series_description, quality)
        if classification is None:
            continue
        sequence = str(classification.get("sequence") or "").strip().upper()
        if sequence in seq_set:
            found.add(sequence)
    return found


def resolve_subject_experiment(
    project: Any,
    subject_label: str,
    *,
    prefer_sequences: Iterable[str] | None = None,
) -> tuple[Any, str]:
    """Return ``(experiment, experiment_label)`` for the best qvtpy MR session."""
    subject = _resolve_subject(project, subject_label)
    if subject is None:
        raise LookupError(
            f"Subject {subject_label!r} not found in XNAT project "
            f"{getattr(project, 'id', lambda: '?')()!r}"
        )

    seqs = prefer_sequences if prefer_sequences is not None else DEFAULT_QVTPY_SCAN_SEQUENCES
    best_exp: Any | None = None
    best_label = ""
    best_score = -1
    best_date = None

    for experiment in getattr(subject, "experiments", {}).values():
        found = _classify_experiment_sequences(experiment, prefer_sequences=seqs)
        score = len(found)
        if score <= 0:
            continue
        exp_label = _experiment_label(experiment)
        exp_date = _experiment_date(experiment)
        if score > best_score:
            best_exp = experiment
            best_label = exp_label
            best_score = score
            best_date = exp_date
            continue
        if score == best_score and best_exp is not None:
            if best_date is None or (exp_date is not None and exp_date > best_date):
                best_exp = experiment
                best_label = exp_label
                best_date = exp_date

    if best_exp is None or not best_label:
        raise LookupError(
            f"No experiment with qvtpy scans found for subject {subject_label!r}"
        )
    return best_exp, best_label


def iter_upload_files(local_dir: Path) -> list[Path]:
    """List files under *local_dir* excluding obvious junk."""
    if not local_dir.is_dir():
        return []
    files: list[Path] = []
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _UPLOAD_IGNORE_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in _UPLOAD_IGNORE_SUFFIXES:
            continue
        files.append(path)
    return files


def xnat_resource_has_files(experiment: Any, resource_label: str) -> bool:
    """Return True when *resource_label* exists on *experiment* and has files."""
    resources = getattr(experiment, "resources", None)
    if resources is None:
        return False
    try:
        resource = resources[resource_label]
    except (KeyError, TypeError, AttributeError):
        try:
            keys = list(resources.keys())
        except Exception:
            return False
        if resource_label not in keys:
            return False
        resource = resources[resource_label]
    try:
        if hasattr(resource, "exists") and not resource.exists():
            return False
    except Exception:
        pass
    try:
        files = resource.files()
    except Exception:
        return False
    return bool(files)


def upload_directory_to_xnat_resource(
    experiment: Any,
    resource_label: str,
    local_dir: Path | str,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> int:
    """Upload *local_dir* to an experiment resource via pyxnat ``put_dir``.

    Returns the number of files that would be / were uploaded.
    """
    root = Path(local_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local upload directory not found: {root}")

    files = iter_upload_files(root)
    n_files = len(files)
    if dry_run:
        log.info(
            f"[dry-run] would POST resource {resource_label!r} "
            f"<- {root} ({n_files} file(s), overwrite={overwrite})"
        )
        return n_files

    staging_root: Path | None = None
    try:
        upload_root = root
        junk_present = any(
            part in _UPLOAD_IGNORE_DIR_NAMES for part in root.parts
        ) or any(p.suffix.lower() in _UPLOAD_IGNORE_SUFFIXES for p in files)
        if junk_present:
            staging_root = Path(tempfile.mkdtemp(prefix="nvitk-xnat-upload-"))
            staging_dest = staging_root / root.name
            shutil.copytree(
                root,
                staging_dest,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "*.pyo",
                    "*.tmp",
                    ".git",
                    ".ipynb_checkpoints",
                ),
            )
            upload_root = staging_dest
            files = iter_upload_files(upload_root)

        resources = getattr(experiment, "resources", None)
        if resources is None:
            raise RuntimeError("Experiment has no resources collection")
        resource = resources[resource_label]
        log.info(
            f"POST resource {resource_label!r} <- {root} "
            f"({len(files)} file(s), overwrite={overwrite})"
        )
        resource.put_dir(str(upload_root), overwrite=overwrite, extract=True)
        return len(files)
    finally:
        if staging_root is not None and staging_root.is_dir():
            shutil.rmtree(staging_root, ignore_errors=True)


__all__ = [
    "DEFAULT_QVTPY_SCAN_SEQUENCES",
    "iter_upload_files",
    "resolve_subject_experiment",
    "upload_directory_to_xnat_resource",
    "xnat_resource_has_files",
]
