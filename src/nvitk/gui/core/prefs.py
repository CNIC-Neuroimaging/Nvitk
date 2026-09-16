"""GUI preferences: the machine-local state the window remembers between launches.

``gui.json`` lives alongside nvitk's other configuration and is found by the same
search (:mod:`nvitk.core.config_paths`), so an install that keeps its config in
``~/.nvitk`` does not end up with the settings there and the window layout
somewhere else. It differs from the rest in who writes it: ``sge.json`` and its
siblings are authored by a person, this one by the application. That is why
:func:`ensure_prefs_file` seeds it from the bundled template rather than asking
anyone to create it, and why ``config_paths.AUTHORED_CONFIG_FILES`` leaves it out
of "is this install configured?".

Everything here fails soft. A preferences file is a convenience — a corrupt one,
an unwritable config directory or a layout saved by another Qt build must never
stop the application starting, so every entry point swallows its errors and
returns the neutral value instead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: Filename inside the config directory, registered in ``config_paths``.
PREFS_FILE = "gui.json"

#: Key holding the base64 ``QMainWindow.saveState()`` blob.
DOCK_STATE_KEY = "dock_state"


def prefs_path() -> Path:
    """Where the preferences file lives, whether or not it exists yet.

    An existing one wins wherever the configuration search finds it — including
    through ``$NVITK_GUI_JSON`` or a ``--config-dir``. Otherwise it belongs in
    whichever directory already holds this install's configuration, and only
    falls back to the XDG default when there is no configuration at all.
    """
    from nvitk.core import config_paths

    # A per-file override names the file outright, whether or not it exists yet.
    direct = os.environ.get(config_paths.FILE_ENV_VARS[PREFS_FILE], "").strip()
    if direct:
        return Path(os.path.expanduser(direct))
    home = config_home()
    return (home or config_paths.default_config_dir()) / PREFS_FILE


def config_home() -> Path | None:
    """The directory holding the configuration this install actually reads.

    The directory of the first authored file the search resolves, rather than the
    first directory that happens to exist: an empty ``~/.config/nvitk`` beside a
    populated ``./.nvitk`` would otherwise put the window layout somewhere nvitk
    is not reading from. ``None`` when nothing is configured.
    """
    from nvitk.core import config_paths

    for name in config_paths.AUTHORED_CONFIG_FILES:
        found = config_paths.config_file(name)
        if found is not None:
            return found.parent
    return None


def is_configured() -> bool:
    """Whether this install has configuration a person wrote.

    A lone ``gui.json`` does not count: it means the GUI has run once, not that
    nvitk has been set up.
    """
    return config_home() is not None


def ensure_prefs_file() -> Path | None:
    """Seed ``gui.json`` from the bundled template; the path, or ``None``.

    Only for an install that is already configured. Creating it unconditionally
    would drop a file into a directory nobody has asked nvitk to own — and on a
    machine with no configuration at all, the layout is the least of what is
    missing. Returns the existing path untouched when there already is one.
    """
    if not is_configured():
        return None
    path = prefs_path()
    if path.is_file():
        return path
    try:
        from nvitk.cli.config_cmd import TEMPLATE_DIR

        template = TEMPLATE_DIR / PREFS_FILE.replace(".json", ".example.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if template.is_file():
            path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            path.write_text(json.dumps({DOCK_STATE_KEY: ""}, indent=2) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 — a seedable preference file is not a requirement
        return None
    return path


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
    "config_home",
    "ensure_prefs_file",
    "is_configured",
    "PREFS_FILE",
    "load_prefs",
    "prefs_path",
    "restore_dock_state",
    "save_dock_state",
    "save_prefs",
]
