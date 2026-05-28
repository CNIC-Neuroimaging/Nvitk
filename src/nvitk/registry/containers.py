"""Read ``registry/containers/registry/containers.json`` and resolve cluster SIF paths."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

_NVITK_CONTAINER_RE = re.compile(r"nvitk_v[\d.]+\.sif", re.IGNORECASE)
_DEFAULT_CLUSTER_ROOT = Path("/data3/BIOIT_IMAGE/Containers")
_NVITK_PROJECT = "nvitk"


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for anc in [here.parent, *here.parents]:
        if (anc / "pyproject.toml").is_file() and (anc / "src" / "nvitk").is_dir():
            return anc
    return None


def registry_path() -> Path | None:
    """Locate ``containers.json`` (canonical path or legacy symlink)."""
    root = _find_repo_root()
    candidates: list[Path] = []
    if root is not None:
        candidates.extend(
            [
                root / "registry" / "containers" / "registry" / "containers.json",
                root / "registry" / "containers.json",
            ]
        )
    candidates.append(
        Path(__file__).resolve().parents[3]
        / "registry"
        / "containers"
        / "registry"
        / "containers.json"
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_container_registry() -> dict[str, Any]:
    path = registry_path()
    if path is None:
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _project_entry(registry: dict[str, Any], name: str) -> dict[str, Any] | None:
    projects = registry.get("containers", {}).get("projects", {})
    if not isinstance(projects, dict):
        return None
    raw = projects.get(name)
    return raw if isinstance(raw, dict) else None


def _version_entry(project: dict[str, Any], version: str | None) -> tuple[str, dict[str, Any]] | None:
    versions = project.get("versions")
    if not isinstance(versions, dict) or not versions:
        return None
    ver = version or str(project.get("latest", "")).strip()
    if not ver:
        ver = sorted(versions.keys())[-1]
    entry = versions.get(ver)
    if not isinstance(entry, dict):
        return None
    return ver, entry


def default_cluster_sif_path(project_name: str, version: str) -> Path:
    """Default samwise deploy path when ``cluster_sif_path`` is absent."""
    return _DEFAULT_CLUSTER_ROOT / f"{project_name}_{version}.sif"


def resolve_cluster_sif_path(
    project_name: str,
    *,
    version: str | None = None,
    registry: dict[str, Any] | None = None,
) -> Path | None:
    """Cluster Singularity path for a project container (``cluster_sif_path`` or convention)."""
    reg = registry if registry is not None else load_container_registry()
    project = _project_entry(reg, project_name)
    if project is None:
        return None
    resolved = _version_entry(project, version)
    if resolved is None:
        return None
    ver, entry = resolved
    raw = entry.get("cluster_sif_path")
    if raw is not None and str(raw).strip():
        return Path(str(raw).strip())
    return default_cluster_sif_path(project_name, ver)


def resolve_nvitk_cluster_sif(*, version: str | None = None) -> Path | None:
    return resolve_cluster_sif_path(_NVITK_PROJECT, version=version)


def _is_nvitk_container_path(value: str) -> bool:
    return bool(_NVITK_CONTAINER_RE.search(value))


def sync_sge_nvitk_container(
    sge_path: Path,
    *,
    version: str | None = None,
    dry_run: bool = False,
) -> Path:
    """
    Set ``paths.nvitk_container`` and nvitk pipeline ``default_sge_container_root``
    from registry ``projects.nvitk.latest``.
    """
    cluster = resolve_nvitk_cluster_sif(version=version)
    if cluster is None:
        raise FileNotFoundError(
            "Could not resolve nvitk cluster SIF from container registry "
            f"(version={version!r})"
        )
    cluster_str = str(cluster)

    with open(sge_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"{sge_path} must be a JSON object")

    paths = doc.setdefault("paths", {})
    if not isinstance(paths, dict):
        paths = {}
        doc["paths"] = paths
    old_global = str(paths.get("nvitk_container", "")).strip()
    paths["nvitk_container"] = cluster_str

    pipelines = doc.get("pipelines")
    updated_pipes: list[str] = []
    if isinstance(pipelines, dict):
        for pipe_id, pipe in pipelines.items():
            if not isinstance(pipe, dict):
                continue
            key = "default_sge_container_root"
            cur = pipe.get(key)
            if cur is None:
                continue
            cur_s = str(cur).strip()
            if not cur_s:
                continue
            if _is_nvitk_container_path(cur_s) or cur_s == old_global:
                pipe[key] = cluster_str
                updated_pipes.append(str(pipe_id))

    if dry_run:
        return cluster

    sge_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sge_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return cluster


def sync_default_sge_json(
    *,
    version: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Sync repo ``.nvitk/sge.json`` from registry latest nvitk."""
    root = _find_repo_root()
    if root is None:
        raise FileNotFoundError("Could not locate repository root")
    sge_path = root / ".nvitk" / "sge.json"
    if not sge_path.is_file():
        raise FileNotFoundError(f"Missing {sge_path}")
    return sync_sge_nvitk_container(sge_path, version=version, dry_run=dry_run)
