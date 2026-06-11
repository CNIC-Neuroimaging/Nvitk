"""Locate ``.nvitk/settings.json`` and read the ``db`` block."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for anc in [here.parent, *here.parents]:
        if (anc / "pyproject.toml").is_file() and (anc / "src" / "nvitk").is_dir():
            return anc
    return None


def settings_json_path() -> Path | None:
    """Return the first existing ``.nvitk/settings.json`` on the search path."""
    candidates: list[Path] = []
    root = _find_repo_root()
    if root is not None:
        candidates.append(root / ".nvitk" / "settings.json")
    candidates.append(Path.cwd() / ".nvitk" / "settings.json")
    env_home = os.environ.get("NVITK_HOME", "").strip()
    if env_home:
        candidates.append(Path(env_home).expanduser() / ".nvitk" / "settings.json")
    env_json = os.environ.get("NVITK_SETTINGS_JSON", "").strip()
    if env_json:
        candidates.append(Path(env_json).expanduser())
    # Installed package layout: …/src/nvitk/db/settings_paths.py -> repo root
    pkg_parents = Path(__file__).resolve().parents
    if len(pkg_parents) >= 4:
        candidates.append(pkg_parents[3] / ".nvitk" / "settings.json")

    seen: set[Path] = set()
    for path in candidates:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def load_db_settings_block() -> dict[str, Any]:
    """Parse the ``db`` section from settings, or return ``{}``."""
    path = settings_json_path()
    if path is None:
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(doc, dict):
        return {}
    db = doc.get("db")
    return db if isinstance(db, dict) else {}


def configured_sge_dataset_root() -> Path | None:
    """Return ``db.sge_root`` from settings (path may not exist on this host)."""
    raw = load_db_settings_block().get("sge_root")
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expanduser(str(raw).strip()))


def sge_dataset_root_path(*, must_exist: bool = True) -> Path | None:
    """Resolve the cluster dataset root from env or ``db.sge_root``."""
    env_root = os.environ.get("NVITK_DATASET_ROOT", "").strip()
    if env_root:
        path = Path(os.path.expanduser(env_root))
        if must_exist and not path.is_dir():
            return None
        return path.resolve() if path.is_dir() else path

    configured = configured_sge_dataset_root()
    if configured is None:
        return None
    if must_exist and not configured.is_dir():
        return None
    return configured.resolve() if configured.is_dir() else configured
