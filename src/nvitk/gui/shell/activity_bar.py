"""Activity bar and module switching (see :mod:`nvitk.gui.shell`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import QActionGroup, QDockWidget, QSizePolicy, QToolBar, QWidget

from nvitk.gui.core import design
from nvitk.gui.core.icons import icon as make_icon


@dataclass(frozen=True)
class Module:
    """A group of panels behind one activity-bar icon."""

    key: str
    title: str
    icon: str
    docks: tuple[str, ...]
    hue: str = "grey"


MODULES: tuple[Module, ...] = (
    Module("data", "Data & QC", "database",
           ("nvitk:data", "nvitk:qc", "nvitk:atlases", "nvitk:dataset_config",
            "nvitk:dicom_browser"), "blue"),
    Module("tools", "Tools (imaging)", "tools", ("nvitk:tools",), "purple"),
    Module("labeling", "Labeling", "brush", ("nvitk:labeling", "nvitk:labels"), "green"),
    Module("mesh", "Mesh", "mesh", ("nvitk:mesh",), "orange"),
    Module("analysis", "Analysis & statistics", "chart", ("nvitk:analysis", "nvitk:statmodels"), "red"),
    Module("export", "Export & record", "export", ("nvitk:export", "nvitk:pipeline"), "cyan"),
    Module("inspect", "Inspect (properties, DICOM, spectral)", "inspect",
           ("nvitk:image_properties", "nvitk:display", "nvitk:dicom_tags", "nvitk:spectral"), "olive"),
)

#: Preferences key holding the active module ("" when folded away).
PREF_KEY = "gui.module"
#: Object name of the activity bar (saved with the window state).
BAR_NAME = "nvitk:activity"


class Shell:
    """Owns the activity bar and shows one module's docks at a time."""

    def __init__(self, viewer: Any, qt_main: Any) -> None:
        self._viewer = viewer
        self._main = qt_main
        self._active: str = ""
        self._bar = QToolBar("Modules", qt_main)
        self._bar.setObjectName(BAR_NAME)
        self._bar.setMovable(False)
        self._bar.setFloatable(False)
        self._bar.setOrientation(Qt.Vertical)
        self._bar.setIconSize(QSize(20, 20))
        self._bar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        qt_main.addToolBar(Qt.LeftToolBarArea, self._bar)
        self._group = QActionGroup(self._bar)
        self._group.setExclusive(False)
        self._actions: dict[str, Any] = {}
        for module in self.modules():
            action = self._bar.addAction(make_icon(module.icon, design.COLOR_MUTED, on_color=design.COLOR_ON_ACCENT), module.title)
            action.setCheckable(True)
            action.setToolTip(f"{module.title}  (click again to fold the panel away)")
            action.triggered.connect(lambda checked, k=module.key: self._clicked(k, checked))
            self._group.addAction(action)
            self._actions[module.key] = action
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._bar.addWidget(spacer)
        # At the foot: the compute backend (CPU / GPU) and CPU threads (shell/compute.py).
        from nvitk.gui.shell.compute import ComputeButton

        self.compute = ComputeButton(self._bar)
        self._bar.addWidget(self.compute)
        self._restyle()
        design.add_theme_listener(self._restyle)

    # structure
    def modules(self) -> list[Module]:
        """Modules with at least one dock present."""
        return [m for m in MODULES if self.docks_of(m.key)]

    def docks_of(self, key: str) -> list[QDockWidget]:
        """The docks of module *key* that exist, in module order."""
        module = next((m for m in MODULES if m.key == key), None)
        if module is None:
            return []
        found = {d.objectName(): d for d in self._main.findChildren(QDockWidget)}
        return [found[name] for name in module.docks if name in found]

    def module_of(self, object_name: str) -> str | None:
        """The module a dock belongs to."""
        return next((m.key for m in MODULES if object_name in m.docks), None)

    def active(self) -> str:
        """Key of the module on screen ("" when folded)."""
        return self._active

    def bar(self) -> QToolBar:
        """The activity bar."""
        return self._bar

    def arrange(self) -> None:
        """Put every module's docks in one tab group on the left, the layer list below them.

        Hidden docks keep their place in the group, so showing a module brings its panels
        back exactly there.
        """
        main = self._main
        docks = [d for m in self.modules() for d in self.docks_of(m.key)]
        if not docks:
            return
        first = docks[0]
        first.setFloating(False)
        main.addDockWidget(Qt.LeftDockWidgetArea, first)
        # Split before tabbing: Qt adds a dock split against a tabbed dock as one more tab.
        layer_list = self._layer_list_dock()
        if layer_list is not None:
            layer_list.setFloating(False)
            main.addDockWidget(Qt.LeftDockWidgetArea, layer_list)
            main.splitDockWidget(first, layer_list, Qt.Vertical)
        for dock in docks[1:]:
            dock.setFloating(False)
            main.tabifyDockWidget(first, dock)
        if layer_list is not None:
            try:
                main.resizeDocks([first, layer_list], [700, 260], Qt.Vertical)
            except Exception:  # noqa: BLE001
                pass

    def _layer_list_dock(self) -> QDockWidget | None:
        try:
            return self._viewer.window._qt_viewer.dockLayerList
        except Exception:  # noqa: BLE001
            return None

    # switching
    def _clicked(self, key: str, checked: bool) -> None:
        self.show_module(key if checked else "")

    def show_module(self, key: str, *, remember: bool = True) -> None:
        """Show module *key*'s panels and hide every other module's ("" folds them all)."""
        for module in self.modules():
            if module.key == key:
                continue
            for dock in self.docks_of(module.key):
                if not dock.isFloating():
                    dock.hide()
        own = self.docks_of(key)
        for dock in own:
            dock.show()
        if own:
            own[0].raise_()
        self._active = key if own else ""
        for module_key, action in self._actions.items():
            action.blockSignals(True)
            action.setChecked(module_key == self._active)
            action.blockSignals(False)
        if remember:
            try:
                from nvitk.gui.core.prefs import save_prefs

                save_prefs({PREF_KEY: self._active})
            except Exception:  # noqa: BLE001
                pass

    def show_dock(self, object_name: str) -> bool:
        """Switch to the module holding *object_name* (and leave raising it to the caller)."""
        key = self.module_of(object_name)
        if key is None:
            return False
        if key != self._active:
            self.show_module(key)
        return True

    def _restyle(self) -> None:
        for module in self.modules():
            action = self._actions.get(module.key)
            if action is not None:
                action.setIcon(make_icon(module.icon, design.COLOR_MUTED, on_color=design.COLOR_ON_ACCENT))
        self._bar.setStyleSheet(
            f"QToolBar#{BAR_NAME.replace(':', '_')} {{ }}"
            f"QToolBar {{ background-color: {design.COLOR_BG}; border: none;"
            f" border-right: 1px solid {design.COLOR_BORDER}; spacing: 2px; padding: 4px 2px; }}"
            "QToolBar QToolButton { border: none; border-radius: 6px; padding: 6px; margin: 1px; }"
            f"QToolBar QToolButton:hover {{ background-color: {design.COLOR_CONTROL_HOVER}; }}"
            f"QToolBar QToolButton:checked {{ background-color: {design.COLOR_ACCENT_DEEP}; }}"
        )


def stored_module(default: str = "tools") -> str:
    """The module the preferences say was on screen last ("" when it was folded)."""
    try:
        from nvitk.gui.core.prefs import load_prefs

        prefs = load_prefs()
        if PREF_KEY in prefs:
            return str(prefs.get(PREF_KEY) or "")
    except Exception:  # noqa: BLE001
        pass
    return default


def install_shell(viewer: Any) -> Shell:
    """Create the activity bar for *viewer*'s window (stored as ``viewer._nvitk_shell``)."""
    existing = getattr(viewer, "_nvitk_shell", None)
    if existing is not None:
        return existing
    shell = Shell(viewer, viewer.window._qt_window)
    viewer._nvitk_shell = shell
    return shell


__all__ = ["BAR_NAME", "MODULES", "Module", "PREF_KEY", "Shell", "install_shell", "stored_module"]
