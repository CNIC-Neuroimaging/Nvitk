"""Read ``sge.json`` and merge its overlays onto Python defaults.

Where the file lives is decided by :mod:`nvitk.core.config_paths`; this module only knows how
to interpret its contents.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from nvitk.core import config_paths

SGE_JSON_NAME = "sge.json"


def sge_json_path() -> Path | None:
    """Locate ``sge.json``; see :func:`nvitk.core.config_paths.describe_search` for where."""
    return config_paths.config_file(SGE_JSON_NAME)


def load_sge_document() -> dict[str, Any]:
    """Parsed ``sge.json``, or an empty dict when it is absent or unreadable.

    Cached by :mod:`~nvitk.core.config_paths`, so the repeated ``paths_section()`` /
    ``defaults_section()`` / ``pipeline_section()`` calls throughout the codebase no longer
    re-read and re-parse the file each time.
    """
    return config_paths.load_json(SGE_JSON_NAME)


def paths_section() -> dict[str, Any]:
    """The ``paths`` block of ``sge.json`` (empty dict if absent)."""
    return dict(load_sge_document().get("paths", {}))


def resolve_nvitk_src_dir(*, fallback: Path | None = None) -> Path:
    """Cluster/host nvitk source tree from ``paths.nvitk_src_dir`` in ``sge.json``."""
    paths = paths_section()
    raw = paths.get("nvitk_src_dir")
    if raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    if fallback is not None:
        return fallback
    return Path(__file__).resolve().parents[1]


def gui_sge_job_root() -> str:
    """Default remote staging root for GUI SGE jobs (``paths.gui_sge_job_root``)."""
    raw = paths_section().get("gui_sge_job_root")
    if raw is None or not str(raw).strip():
        return ""
    return str(raw).strip().rstrip("/")


def resolve_nvitk_container(
    *, pipe: Mapping[str, Any] | None = None, fallback: Path | None = None
) -> Path | None:
    """Cluster nvitk Singularity image from pipeline override, ``sge.json``, or the registry.

    ``None`` when nothing is configured. There is deliberately no built-in image path: one
    would be specific to a single institution's filesystem, and silently returning it makes an
    unconfigured install fail later with a confusing "no such file" instead of saying which
    setting is missing. Callers that must have an image should pass the result through
    :func:`nvitk.core.config_paths.require`.
    """
    if pipe:
        for key in ("default_sge_container_root", "sge_container_root", "container_path"):
            raw = pipe.get(key)
            if raw is not None and str(raw).strip():
                return Path(os.path.expanduser(str(raw).strip()))
    paths = paths_section()
    raw = paths.get("nvitk_container")
    if raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    try:
        from nvitk.registry.containers import resolve_nvitk_cluster_sif

        reg_path = resolve_nvitk_cluster_sif()
        if reg_path is not None:
            return reg_path
    except Exception:
        pass
    return fallback


def defaults_section() -> dict[str, Any]:
    """The ``defaults`` block of ``sge.json`` (empty dict if absent)."""
    return dict(load_sge_document().get("defaults", {}))


def pipeline_section(pipeline_id: str) -> dict[str, Any]:
    """The ``pipelines[pipeline_id]`` block of ``sge.json`` (empty dict if absent)."""
    doc = load_sge_document()
    pipes = doc.get("pipelines")
    if not isinstance(pipes, dict):
        return {}
    raw = pipes.get(pipeline_id, {})
    return dict(raw) if isinstance(raw, dict) else {}


def merged_pipeline_flat(pipeline_id: str) -> dict[str, Any]:
    """``defaults`` shallow-updated by ``pipelines[pipeline_id]``."""
    out = defaults_section()
    out.update(pipeline_section(pipeline_id))
    return out


def _p(path_like: Any) -> Path:
    """Coerce a string/Path-like value to a user-expanded :class:`Path`."""
    return Path(os.path.expanduser(str(path_like)))


def resolve_log_err_dirs(
    *,
    paths: Mapping[str, Any],
    pipe: Mapping[str, Any],
    fallback_log: Path,
    fallback_err: Path,
) -> tuple[Path, Path]:
    """Resolve SGE log/err dirs: explicit paths, or ``sge_*_root`` + optional ``*_subdir``."""
    log_dir = pipe.get("sge_log_dir") or paths.get("sge_log_dir")
    err_dir = pipe.get("sge_err_dir") or paths.get("sge_err_dir")
    if log_dir:
        lg = _p(log_dir)
    else:
        root = paths.get("sge_log_root")
        sub = pipe.get("log_subdir") if "log_subdir" in pipe else pipe.get("sge_log_subdir")
        if root:
            r = _p(root)
            if sub is not None and str(sub).strip():
                lg = r / str(sub).strip()
            else:
                lg = r
        else:
            lg = fallback_log
    if err_dir:
        er = _p(err_dir)
    else:
        root = paths.get("sge_err_root")
        sub = pipe.get("err_subdir") if "err_subdir" in pipe else pipe.get("sge_err_subdir")
        if root:
            r = _p(root)
            if sub is not None and str(sub).strip():
                er = r / str(sub).strip()
            else:
                er = r
        else:
            er = fallback_err
    return lg, er


def merge_cluster_host_aliases(
    base: dict[str, str],
    paths: Mapping[str, Any],
    pipe: Mapping[str, Any],
) -> dict[str, str]:
    """Merge ``cluster_host_aliases`` maps from *paths* then *pipe* on top of *base* (later sources win)."""
    out = dict(base)
    for section in (paths, pipe):
        extra = section.get("cluster_host_aliases")
        if isinstance(extra, dict):
            for k, v in extra.items():
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
    return out
