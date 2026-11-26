"""The working dataset, switchable from the GUI: one root for the whole process.

Every panel opens the dataset through :func:`nvitk.db.repo.get_repo`, which honours
``$NVITK_DATASET_ROOT`` before ``settings.json``. :func:`set_root` sets that variable for this
process, remembers the folder among the recent ones (``gui.json``: ``db.recent_roots``) and
announces the change on :func:`dataset_bus`, so open panels (Data, QC, Analysis, Statmodels,
the import editor) re-open their dataset. :func:`restore_last_root` puts the last choice back at
start-up; *Use the settings default* clears it.

:class:`DatasetRootButton` is the switch: the current dataset's name, a menu of recent datasets,
*Choose a dataset folder…*, *Use the settings default* and *Open the folder*.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from qtpy.QtCore import QObject, Qt, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QFileDialog, QMenu, QMessageBox, QToolButton, QWidget

ENV = "NVITK_DATASET_ROOT"
RECENT_KEY = "db.recent_roots"
LAST_KEY = "db.gui_root"
MAX_RECENT = 8

_BUS: "_DatasetBus | None" = None
_SETTINGS_ROOT: dict[str, str | None] = {}


class _DatasetBus(QObject):
    #: The new dataset root (a path string).
    changed = Signal(str)


def dataset_bus() -> _DatasetBus:
    """The process-wide notifier of dataset switches."""
    global _BUS
    if _BUS is None:
        _BUS = _DatasetBus()
    return _BUS


def is_dataset(path: str | Path) -> bool:
    """Whether *path* holds an nvitk dataset (``catalog/repository.json``)."""
    root = Path(path).expanduser()
    return (root / "catalog" / "repository.json").is_file()


def settings_root() -> str | None:
    """The root ``settings.json`` (or the SGE / built-in default) gives, ignoring a GUI switch."""
    if "root" not in _SETTINGS_ROOT:
        saved = os.environ.pop(ENV, None)
        try:
            from nvitk.db.repo import get_repo

            _SETTINGS_ROOT["root"] = str(get_repo().root)
        except Exception:  # noqa: BLE001
            _SETTINGS_ROOT["root"] = None
        finally:
            if saved is not None:
                os.environ[ENV] = saved
    return _SETTINGS_ROOT["root"]


def current_root() -> str | None:
    """The dataset panels open now."""
    env = os.environ.get(ENV, "").strip()
    if env:
        return str(Path(env).expanduser().resolve())
    return settings_root()


def recent_roots() -> list[str]:
    """Recently used dataset folders, newest first."""
    try:
        from nvitk.gui.core.prefs import load_prefs

        value = load_prefs().get(RECENT_KEY) or []
    except Exception:  # noqa: BLE001
        value = []
    return [str(v) for v in value if isinstance(v, str)]


def _remember(root: str | None) -> None:
    try:
        from nvitk.gui.core.prefs import save_prefs

        values: dict[str, Any] = {LAST_KEY: root or ""}
        if root:
            values[RECENT_KEY] = ([root] + [r for r in recent_roots() if r != root])[:MAX_RECENT]
        save_prefs(values)
    except Exception:  # noqa: BLE001
        pass


def set_root(path: str | Path | None, *, remember: bool = True) -> str | None:
    """Make *path* the working dataset (``None``: back to the settings default); announces it.

    Raises ``ValueError`` when *path* is not a dataset folder.
    """
    if path is None or not str(path).strip():
        os.environ.pop(ENV, None)
        root = settings_root()
        if remember:
            _remember(None)
    else:
        resolved = Path(path).expanduser().resolve()
        if not is_dataset(resolved):
            raise ValueError(f"{resolved} is not an nvitk dataset (no catalog/repository.json).")
        root = str(resolved)
        os.environ[ENV] = root
        # An SGE switch would win over the variable in get_repo; a GUI choice is meant to.
        os.environ.pop("NVITK_SGE", None)
        if remember:
            _remember(root)
    dataset_bus().changed.emit(root or "")
    return root


def restore_last_root() -> str | None:
    """At start-up: the dataset picked last time in the GUI, if it still exists and the
    environment does not already name one."""
    if os.environ.get(ENV, "").strip():
        return None
    try:
        from nvitk.gui.core.prefs import load_prefs

        last = str(load_prefs().get(LAST_KEY) or "").strip()
    except Exception:  # noqa: BLE001
        return None
    if last and is_dataset(last):
        settings_root()  # remember the default before the variable hides it
        os.environ[ENV] = last
        return last
    return None


def _short(root: str | None) -> str:
    if not root:
        return "no dataset"
    path = Path(root)
    return path.name or str(path)


class DatasetRootButton(QToolButton):
    """The working dataset's name, with a menu to switch to another one."""

    def __init__(self, parent: QWidget | None = None, *, show_name: bool = True) -> None:
        super().__init__(parent)
        from nvitk.gui.core.icons import icon

        self._show_name = show_name
        self.setIcon(icon("database"))
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon if show_name else Qt.ToolButtonIconOnly)
        self.setAutoRaise(True)
        menu = QMenu(self)
        menu.aboutToShow.connect(lambda: self._fill(menu))
        self.setMenu(menu)
        dataset_bus().changed.connect(lambda _r: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        root = current_root()
        if self._show_name:
            self.setText(_short(root))
        self.setToolTip(f"Working dataset: {root or 'none'}\nClick to switch to another dataset.")

    def _fill(self, menu: QMenu) -> None:
        menu.clear()
        current = current_root()
        head = menu.addAction(f"Working dataset: {current or 'none'}")
        head.setEnabled(False)
        menu.addSeparator()
        default = settings_root()
        roots = [r for r in recent_roots() if r != default]
        for root in ([default] if default else []) + roots:
            label = f"{_short(root)}   ({root})" + ("   · settings default" if root == default else "")
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(root == current)
            exists = is_dataset(root)
            action.setEnabled(exists)
            if not exists:
                action.setText(label + "   · not found")
            action.triggered.connect(lambda _=False, r=root: self._switch(r if r != default else None))
        menu.addSeparator()
        menu.addAction("Choose a dataset folder…", self._choose)
        use_default = menu.addAction("Use the settings default", lambda: self._switch(None))
        use_default.setEnabled(os.environ.get(ENV, "").strip() != "")
        open_folder = menu.addAction("Open the folder")
        open_folder.setEnabled(bool(current))
        open_folder.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(current))))

    def _choose(self) -> None:
        start = current_root() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Choose an nvitk dataset folder", str(Path(start).parent))
        if folder:
            self._switch(folder)

    def _switch(self, root: str | None) -> None:
        try:
            set_root(root)
        except ValueError as exc:
            QMessageBox.warning(self, "Not a dataset", f"{exc}\n\nCreate one in Data ▸ Dataset & XNAT setup ▸ New dataset.")


__all__ = [
    "DatasetRootButton",
    "current_root",
    "dataset_bus",
    "is_dataset",
    "recent_roots",
    "restore_last_root",
    "set_root",
    "settings_root",
]
