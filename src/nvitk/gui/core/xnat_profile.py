"""The XNAT connection profile in use, found from the configuration and switchable from the GUI.

Every XNAT call opens its profile through :func:`nvitk.db.xnat_config.load_xnat_profile`, which
honours ``$NVITK_XNAT_CONFIG`` before the first ``xnat.{yaml,yml,json}`` found in the nvitk
configuration directories. Nothing has to be browsed for: :func:`current_profile` is that file.
:func:`set_profile` switches to another one for this process (setting the variable, so the
Data tab's downloads and the setup panel's sync both follow), remembers it among the recent
profiles (``gui.json``: ``xnat.recent_profiles``) and announces it on :func:`profile_bus`;
:func:`restore_last_profile` puts the last choice back at start-up.

:class:`XnatProfileButton` shows the profile (server and file) with a menu of every profile
found, *Choose a profile file…*, *Use the default* and *Open the folder*, as the dataset
button does for datasets. The password is never read here: only server, project and user.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from qtpy.QtCore import QObject, Qt, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QFileDialog, QMenu, QMessageBox, QToolButton, QWidget

ENV = "NVITK_XNAT_CONFIG"
RECENT_KEY = "xnat.recent_profiles"
LAST_KEY = "xnat.gui_profile"
MAX_RECENT = 8

#: ``$NVITK_XNAT_CONFIG`` as the process started (before a GUI switch set it).
_STARTUP_ENV: dict[str, str | None] = {}
_BUS: "_ProfileBus | None" = None


class _ProfileBus(QObject):
    #: The profile now in use (a path string; empty when none is found).
    changed = Signal(str)


def profile_bus() -> _ProfileBus:
    """The process-wide notifier of profile switches."""
    global _BUS
    if _BUS is None:
        _BUS = _ProfileBus()
    return _BUS


def _startup_env() -> str | None:
    if "env" not in _STARTUP_ENV:
        value = os.environ.get(ENV, "").strip()
        _STARTUP_ENV["env"] = value or None
    return _STARTUP_ENV["env"]


def _resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def found_profiles() -> list[str]:
    """Every XNAT profile file in the nvitk configuration directories, in lookup order."""
    from nvitk.db.xnat_config import _default_config_paths

    out: list[str] = []
    for candidate in _default_config_paths():
        try:
            if candidate.is_file():
                path = _resolved(candidate)
                if path not in out:
                    out.append(path)
        except OSError:
            continue
    return out


def default_profile() -> str | None:
    """The profile in force without a GUI switch: ``$NVITK_XNAT_CONFIG`` as launched, else the
    first one found."""
    env = _startup_env()
    if env and Path(env).expanduser().is_file():
        return _resolved(env)
    found = found_profiles()
    return found[0] if found else None


def current_profile() -> str | None:
    """The profile XNAT calls use now (``None``: none configured)."""
    _startup_env()
    env = os.environ.get(ENV, "").strip()
    if env:
        return _resolved(env) if Path(env).expanduser().is_file() else None
    return default_profile()


def recent_profiles() -> list[str]:
    """Profiles picked in the GUI before, newest first."""
    try:
        from nvitk.gui.core.prefs import load_prefs

        value = load_prefs().get(RECENT_KEY) or []
    except Exception:  # noqa: BLE001
        value = []
    return [str(v) for v in value if isinstance(v, str)]


def available_profiles() -> list[str]:
    """What the switch offers: the default, every profile found, then the recent ones."""
    out: list[str] = []
    for path in [default_profile(), *found_profiles(), *recent_profiles()]:
        if path and path not in out:
            out.append(path)
    return out


def profile_summary(path: str | Path | None) -> dict[str, str]:
    """``{"server", "host", "project", "user"}`` of the profile at *path* (empty strings for
    what it does not say, or when it cannot be read). Never the password."""
    out = {"server": "", "host": "", "project": "", "user": ""}
    if not path:
        return out
    try:
        from nvitk.db.xnat_config import _load_file

        data = _load_file(Path(path))
    except Exception:  # noqa: BLE001 (unreadable: shown by its file name only)
        return out
    if not isinstance(data, dict):
        return out
    nested = data.get("xnat") if isinstance(data.get("xnat"), dict) else None
    data = nested or data
    for key, source in (("server", "server"), ("project", "default_project"), ("user", "user")):
        value = data.get(source)
        out[key] = str(value).strip() if value is not None else ""
    server = out["server"]
    host = urlparse(server if "://" in server else f"//{server}").hostname if server else ""
    out["host"] = host or server
    return out


def _remember(path: str | None) -> None:
    try:
        from nvitk.gui.core.prefs import save_prefs

        values: dict[str, Any] = {LAST_KEY: path or ""}
        if path:
            values[RECENT_KEY] = ([path] + [p for p in recent_profiles() if p != path])[:MAX_RECENT]
        save_prefs(values)
    except Exception:  # noqa: BLE001
        pass


def set_profile(path: str | Path | None, *, remember: bool = True) -> str | None:
    """Use the profile at *path* for this process (``None``: back to the default); announces it.

    Raises ``ValueError`` when *path* is not a readable XNAT profile (JSON or YAML mapping).
    """
    _startup_env()
    if path is None or not str(path).strip():
        env = _startup_env()
        if env:
            os.environ[ENV] = env
        else:
            os.environ.pop(ENV, None)
        chosen = default_profile()
        if remember:
            _remember(None)
    else:
        resolved = Path(path).expanduser().resolve()
        try:
            from nvitk.db.xnat_config import _load_file

            data = _load_file(resolved)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{resolved} is not a readable XNAT profile ({exc}).") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{resolved} is not an XNAT profile (expected a mapping with 'server').")
        chosen = str(resolved)
        os.environ[ENV] = chosen
        if remember:
            _remember(chosen)
    profile_bus().changed.emit(chosen or "")
    return chosen


def restore_last_profile() -> str | None:
    """At start-up: the profile picked last time in the GUI, if it still exists and the
    environment does not already name one."""
    if _startup_env():
        return None
    try:
        from nvitk.gui.core.prefs import load_prefs

        last = str(load_prefs().get(LAST_KEY) or "").strip()
    except Exception:  # noqa: BLE001
        return None
    if last and Path(last).is_file():
        os.environ[ENV] = last
        return last
    return None


def _label(path: str | None) -> str:
    if not path:
        return "no XNAT profile"
    summary = profile_summary(path)
    return summary["host"] or Path(path).name


class XnatProfileButton(QToolButton):
    """The XNAT profile in use (its server), with a menu to switch to another one."""

    def __init__(self, parent: QWidget | None = None, *, show_name: bool = True) -> None:
        super().__init__(parent)
        from nvitk.gui.core.icons import icon

        self._show_name = show_name
        self.setIcon(icon("server"))
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon if show_name else Qt.ToolButtonIconOnly)
        self.setAutoRaise(True)
        menu = QMenu(self)
        menu.aboutToShow.connect(lambda: self._fill(menu))
        self.setMenu(menu)
        profile_bus().changed.connect(self._on_changed)
        self.refresh()

    def _on_changed(self, _path: str) -> None:
        self.refresh()

    def refresh(self) -> None:
        path = current_profile()
        if self._show_name:
            self.setText(_label(path))
        if path:
            summary = profile_summary(path)
            lines = [f"XNAT profile: {path}"]
            if summary["server"]:
                lines.append(f"Server: {summary['server']}")
            if summary["project"]:
                lines.append(f"Default project: {summary['project']}")
            others = len(available_profiles()) - 1
            lines.append(f"Click to switch ({others} other profile{'s' if others != 1 else ''} found)."
                         if others > 0 else "Click to choose another profile file.")
        else:
            lines = ["No XNAT profile found (xnat.json / xnat.yaml in the nvitk config directory).",
                     "Click to choose one."]
        self.setToolTip("\n".join(lines))

    def _fill(self, menu: QMenu) -> None:
        menu.clear()
        current = current_profile()
        head = menu.addAction(f"XNAT profile: {current or 'none'}")
        head.setEnabled(False)
        menu.addSeparator()
        default = default_profile()
        for path in available_profiles():
            host = profile_summary(path)["host"]
            label = f"{host or Path(path).name}   ({path})" + ("   · default" if path == default else "")
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(path == current)
            exists = Path(path).is_file()
            action.setEnabled(exists)
            if not exists:
                action.setText(label + "   · not found")
            action.triggered.connect(lambda _=False, p=path: self._switch(p if p != default else None))
        if not available_profiles():
            menu.addAction("(no profile found in the nvitk config directories)").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Choose a profile file…", self._choose)
        use_default = menu.addAction("Use the default", lambda: self._switch(None))
        use_default.setEnabled(current != default)
        open_folder = menu.addAction("Open the folder")
        open_folder.setEnabled(bool(current))
        open_folder.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(str(current)).parent))))

    def _choose(self) -> None:
        start = current_profile() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Choose an XNAT profile", str(Path(start).parent),
                                              "XNAT profile (*.json *.yaml *.yml);;All (*)")
        if path:
            self._switch(path)

    def _switch(self, path: str | None) -> None:
        try:
            set_profile(path)
        except ValueError as exc:
            QMessageBox.warning(self, "Not an XNAT profile", str(exc))


__all__ = [
    "XnatProfileButton",
    "available_profiles",
    "current_profile",
    "default_profile",
    "found_profiles",
    "profile_bus",
    "profile_summary",
    "recent_profiles",
    "restore_last_profile",
    "set_profile",
]
