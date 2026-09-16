"""Per-user GUI preferences: the small, machine-local state the window remembers.

Separate from :mod:`nvitk.core.config_paths`, which resolves *site* configuration
(``sge.json``, ``settings.json``, ``xnat.json``) and is deliberately read-only.
This module owns the write side, and only for state the user never edits by hand:
a dock layout blob, and whatever else the window needs to reopen the way it was
closed.

Everything here fails soft. A preferences file is a convenience — a corrupt one,
an unwritable config directory or a layout saved by another Qt build must never
stop the application starting, so every entry point swallows its errors and
returns the neutral value instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Filename inside the config directory. Not one of ``config_paths``' known
#: names, so it is never picked up by the site-configuration search.
PREFS_FILE = "gui.json"

#: Key holding the base64 ``QMainWindow.saveState()`` blob.
DOCK_STATE_KEY = "dock_state"


def prefs_path() -> Path:
    """Where the preferences file lives, whether or not it exists yet."""
    from nvitk.core.config_paths import default_config_dir

    return default_config_dir() / PREFS_FILE


def load_prefs() -> dict[str, Any]:
    """The stored preferences, or ``{}`` when there are none to read."""
    try:
        path = prefs_path()
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — unreadable or corrupt is the same as absent
        return {}
    return data if isinstance(data, dict) else {}


def save_prefs(values: dict[str, Any]) -> bool:
    """Merge *values* into the stored preferences; ``True`` when it was written.

    Merged rather than replaced so two callers saving different keys do not erase
    each other, and written through a temporary file so an interrupted write
    leaves the previous preferences intact rather than a truncated file.
    """
    try:
        merged = load_prefs()
        merged.update(values)
        path = prefs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:  # noqa: BLE001 — an unwritable config dir is not fatal
        return False


def save_dock_state(window: Any) -> bool:
    """Store *window*'s dock layout. ``True`` when it was written."""
    try:
        from qtpy.QtCore import QByteArray  # noqa: F401 — import guard only

        encoded = bytes(window.saveState().toBase64()).decode("ascii")
    except Exception:  # noqa: BLE001
        return False
    return save_prefs({DOCK_STATE_KEY: encoded})


def restore_dock_state(window: Any) -> bool:
    """Put *window*'s docks back where they were. ``True`` when a layout applied.

    Must be called **after** every dock that should be restored has been added:
    ``restoreState`` matches docks by ``objectName`` and silently drops entries it
    cannot find. Napari's own restore runs while the viewer is being constructed,
    long before nvitk's docks exist, which is why this cannot rely on it.
    """
    encoded = str(load_prefs().get(DOCK_STATE_KEY) or "")
    if not encoded:
        return False
    try:
        from qtpy.QtCore import QByteArray

        return bool(window.restoreState(QByteArray.fromBase64(encoded.encode("ascii"))))
    except Exception:  # noqa: BLE001 — a layout from another Qt build is not
        # worth failing a launch over; the default arrangement is fine.
        return False


__all__ = [
    "DOCK_STATE_KEY",
    "PREFS_FILE",
    "load_prefs",
    "prefs_path",
    "restore_dock_state",
    "save_dock_state",
    "save_prefs",
]
