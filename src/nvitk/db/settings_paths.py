"""Read ``settings.json`` and expose its ``db`` block.

Where the file lives is decided by :mod:`nvitk.core.config_paths`; this module only knows how
to interpret its contents.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nvitk.core import config_paths

SETTINGS_JSON_NAME = "settings.json"


def settings_json_path() -> Path | None:
    """Locate ``settings.json``; see :func:`nvitk.core.config_paths.describe_search` for where."""
    return config_paths.config_file(SETTINGS_JSON_NAME)


def load_settings_document() -> dict[str, Any]:
    """The whole parsed ``settings.json`` (``{}`` when absent).

    ``nvitk.viz.atlas_sources`` needs the ``atlas`` block, which this module previously did not
    expose — it re-opened and re-parsed the file itself. Exposing the document keeps a single
    reader.
    """
    return config_paths.load_json(SETTINGS_JSON_NAME)


def load_db_settings_block() -> dict[str, Any]:
    """The ``db`` section of settings, or ``{}``."""
    db = load_settings_document().get("db")
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
