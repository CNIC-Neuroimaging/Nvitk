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
    """Locate ``.nvitk/sge.json`` (repo, cwd, ``NVITK_HOME``, or ``NVITK_SGE_JSON``)."""
    candidates: list[Path] = []
    root = _find_repo_root()
    if root is not None:
        candidates.append(root / ".nvitk" / "sge.json")
    candidates.append(Path.cwd() / ".nvitk" / "sge.json")
    env_home = os.environ.get("NVITK_HOME", "").strip()
    if env_home:
        candidates.append(Path(env_home).expanduser() / ".nvitk" / "sge.json")
    env_json = os.environ.get("NVITK_SGE_JSON", "").strip()
    if env_json:
        candidates.append(Path(env_json).expanduser())
    seen: set[Path] = set()
    for p in candidates:
        try:
            key = p.resolve()
        except OSError:
            key = p
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    return None


def load_sge_document() -> dict[str, Any]:
    """Return parsed ``sge.json`` or empty dict if missing / invalid."""
    path = sge_json_path()
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        import warnings

        warnings.warn(f"Could not load {path}: {exc}", stacklevel=2)
        return {}


def paths_section() -> dict[str, Any]:
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


def resolve_nvitk_container(*, pipe: Mapping[str, Any] | None = None, fallback: Path | None = None) -> Path:
    """Cluster nvitk Singularity image from ``paths.nvitk_container`` or pipeline override."""
    if pipe:
        for key in ("default_sge_container_root", "sge_container_root", "container_path"):
            raw = pipe.get(key)
            if raw is not None and str(raw).strip():
                return Path(os.path.expanduser(str(raw).strip()))
    paths = paths_section()
    raw = paths.get("nvitk_container")
    if raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    if fallback is not None:
        return fallback
    return Path("/data3/BIOIT_IMAGE/Containers/nvitk_v2026.05.27.sif")


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
