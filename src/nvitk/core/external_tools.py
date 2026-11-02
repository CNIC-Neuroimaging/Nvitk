"""Where the external suites nvitk drives live: FreeSurfer, its license, FastSurfer, FSL.

The locations are remembered in ``external_tools.json`` beside the rest of the
configuration (the first existing nvitk configuration directory, else the XDG one),
so the GUI fields and the ``--freesurfer-home`` / ``--fs-license`` CLI options only
need to be given once::

    {"freesurfer_home": "~/freesurfer-7.4.1", "fs_license": "~/license.txt",
     "fastsurfer_home": "...", "fsldir": "..."}

Environment variables still win where a suite defines one (``FREESURFER_HOME``,
``FS_LICENSE``, ``FSLDIR``); each resolver documents its own order.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nvitk.core import config_paths

FILE_NAME = "external_tools.json"

#: Keys this file understands.
KEYS = ("freesurfer_home", "fs_license", "fastsurfer_home", "fsldir")


def settings_path() -> Path:
    """The settings file: an existing one if found, else where a new one is written."""
    found = config_paths.config_file(FILE_NAME)
    if found is not None and found.is_file():
        return found
    base = config_paths.config_dir() or config_paths.default_config_dir()
    return base / FILE_NAME


def load() -> dict[str, Any]:
    """The stored locations (empty when nothing is stored or the file is unreadable)."""
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(data) if isinstance(data, dict) else {}


def get_path(key: str) -> Path | None:
    """One stored location, expanded, or ``None``."""
    raw = load().get(key)
    if not raw:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(raw))))


def remember(**values: str | os.PathLike[str] | None) -> Path:
    """Store locations (``None`` or ``""`` clears one); returns the file written."""
    unknown = set(values) - set(KEYS)
    if unknown:
        raise KeyError(f"Unknown external-tool setting(s): {', '.join(sorted(unknown))}")
    data = load()
    for key, value in values.items():
        if value in (None, ""):
            data.pop(key, None)
        else:
            data[key] = str(value)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def env_prefix() -> Path:
    """The running environment's prefix (where nvitk installs the suites it manages)."""
    import sys

    return Path(sys.prefix)


def managed_dir(name: str) -> Path:
    """``$PREFIX/share/nvitk/<name>``: a suite nvitk installed into this environment."""
    return env_prefix() / "share" / "nvitk" / name


__all__ = ["FILE_NAME", "KEYS", "env_prefix", "get_path", "load", "managed_dir", "remember", "settings_path"]
