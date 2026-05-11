"""Load optional repo-local ``.nvitk/sge.json`` and merge overlays onto Python defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def _find_repo_root() -> Path | None:
    """Ascend from this file until ``pyproject.toml`` + ``src/nvitk`` exist."""
    here = Path(__file__).resolve()
    for anc in [here.parent, *here.parents]:
        if (anc / "pyproject.toml").is_file() and (anc / "src" / "nvitk").is_dir():
            return anc
    return None


def sge_json_path() -> Path | None:
    root = _find_repo_root()
    if root is None:
        return None
    p = root / ".nvitk" / "sge.json"
    return p if p.is_file() else None


def load_sge_document() -> dict[str, Any]:
    """Return parsed ``sge.json`` or empty dict if missing / invalid."""
    path = sge_json_path()
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def paths_section() -> dict[str, Any]:
    return dict(load_sge_document().get("paths", {}))


def defaults_section() -> dict[str, Any]:
    return dict(load_sge_document().get("defaults", {}))


def pipeline_section(pipeline_id: str) -> dict[str, Any]:
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
    out = dict(base)
    for section in (paths, pipe):
        extra = section.get("cluster_host_aliases")
        if isinstance(extra, dict):
            for k, v in extra.items():
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
    return out
