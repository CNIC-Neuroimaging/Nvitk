"""Read ``settings.json`` and expose its ``dataset`` block.

Where the file lives is decided by :mod:`nvitk.core.config_paths`; this module only knows how
to interpret its contents::

    {
      "dataset": {
        "root": "...",              # the nvitk dataset this machine reads and writes
        "dvc_repo": "...",          # repository holding the DVC pointer files
        "dvc_remote_url": "...",    # where the DVC content lives
        "sge_root": "...",          # the dataset as seen from inside a cluster job
        "statmodels_root": "...",   # where the Stats GUI saves its models
        "raw_mount_root": "...",    # raw acquisition data, if not under the dataset
        "sqlite_index": true,
        "auto_scaffold": false
      }
    }

Atlas locations live in their own ``atlas.json`` (:mod:`nvitk.viz.atlas_sources`) and the XNAT
connection in ``xnat.json`` (:mod:`nvitk.db.xnat_config`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nvitk.core import config_paths

SETTINGS_JSON_NAME = "settings.json"

#: The block of ``settings.json`` holding the dataset settings.
DATASET_SECTION = "dataset"


def settings_json_path() -> Path | None:
    """Locate ``settings.json``; see :func:`nvitk.core.config_paths.describe_search` for where."""
    return config_paths.config_file(SETTINGS_JSON_NAME)


def load_settings_document() -> dict[str, Any]:
    """The whole parsed ``settings.json`` (``{}`` when absent)."""
    return config_paths.load_json(SETTINGS_JSON_NAME)


def load_dataset_settings_block() -> dict[str, Any]:
    """The ``dataset`` section of settings, or ``{}``."""
    block = load_settings_document().get(DATASET_SECTION)
    return block if isinstance(block, dict) else {}


def configured_dataset_root() -> Path | None:
    """``dataset.root`` from settings, expanded (``None`` when unset; the path may not exist)."""
    raw = load_dataset_settings_block().get("root")
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expanduser(str(raw).strip()))


def configured_sge_dataset_root() -> Path | None:
    """Return ``dataset.sge_root`` from settings (path may not exist on this host)."""
    raw = load_dataset_settings_block().get("sge_root")
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expanduser(str(raw).strip()))


def sge_dataset_root_path(*, must_exist: bool = True) -> Path | None:
    """Resolve the cluster dataset root from env or ``dataset.sge_root``."""
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
