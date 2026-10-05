"""Every panel a dock, movable between the main window and a workspace window.

Each nvitk panel (Tools, Labels, Data, QC, …) is its own dock, tabbed together on
the right by default, and Napari's own docks (layer list, layer controls, console,
plugins) are treated the same way. Any of them can be:

* popped out into a floating window and docked back (title-bar ⧉, or double-click);
* moved into the **workspace** — a second top-level window with no fixed
  content, where panels are docked, split and tabbed freely;
* hidden and shown again from the **Panels** menu, which both windows carry.

Qt cannot drag a dock from one ``QMainWindow`` to another, so moving between the
windows goes through the menu; inside a window, dragging works as usual.

The arrangement is remembered: the main window's ``saveState`` as before, plus
which docks live in the workspace, its own ``saveState`` and its geometry.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import QByteArray, QObject, Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAction,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMenu,
    QTabBar,
    QTabWidget,
    QWidget,
)

#: Preferences key for the workspace window's layout.
WORKSPACE_PREF_KEY = "workspace_layout"
WORKSPACE_OBJECT_NAME = "nvitkWorkspace"
LABELS_SHORTCUT = "Ctrl+Shift+L"

_DOCK_OPTIONS = (
    QMainWindow.DockOption.AnimatedDocks
    | QMainWindow.DockOption.AllowNestedDocks
    | QMainWindow.DockOption.AllowTabbedDocks
    | QMainWindow.DockOption.GroupedDragging
)


def make_panel_dock(
    viewer: Any,
    widget: QWidget,
    *,
    object_name: str,
    title: str,
    home_area: Any = Qt.RightDockWidgetArea,
    extras: list[QWidget] | None = None,
) -> QDockWidget:
    """A plain dock for an nvitk panel, with the nvitk pop-out title bar.

    Not added to any window: the caller places it. Plain rather than Napari's
    ``QtViewerDockWidget``, which rebuilds its title bar on every show and so
    would drop the nvitk controls (see ``attach_ortho_dock``).
    """
    from nvitk.gui.core.design import apply_theme
    from nvitk.gui.viz.left_dock import install_expand_button

    try:
        parent = viewer.window._qt_window
    except Exception:  # noqa: BLE001
        parent = None
    dock = QDockWidget(title, parent)
    dock.setObjectName(object_name)
    dock.setWidget(widget)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setMinimumWidth(120)
    dock._nvitk_home_area = home_area
    apply_theme(dock)
    install_expand_button(dock, title, extras=extras)
    return dock


class WorkspaceWindow(QMainWindow):
    """A second window that holds whichever panels are moved into it."""

    def __init__(self, manager: "PanelManager", parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self._manager = manager
        self._quitting = False
        self.setObjectName(WORKSPACE_OBJECT_NAME)
        self.setWindowTitle("nvitk — workspace")
        self.setDockOptions(_DOCK_OPTIONS)
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        self.tabifiedDockWidgetActivated.connect(lambda _dock: manager._tab_timer.start())
        self.setCentralWidget(self._make_hint())
        self.resize(1100, 750)
        menu = self.menuBar().addMenu("&Panels")
        manager.attach_menu(menu)

    @staticmethod
    def _make_hint() -> QLabel:
        hint = QLabel(
            "Empty workspace.\n\nUse Panels ▸ <panel> ▸ Move to workspace window "
            "to bring panels here, then drag them by their title bars to split or tab them."
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        return hint

    def sync_hint(self) -> None:
        """Show the empty-workspace hint only while there is nothing to show.

        Taken out rather than hidden: with no central widget the docks share the
        whole window, which is the point of a workspace.
        """
        has_docks = any(not d.isHidden() for d in self._manager.docks_in(self))
        central = self.centralWidget()
        if has_docks and central is not None:
            self.takeCentralWidget().deleteLater()
        elif not has_docks and central is None:
            self.setCentralWidget(self._make_hint())

    def closeEvent(self, event: Any) -> None:
        """Closing the workspace gives its panels back to the main window, so
        nothing is lost behind a window nobody can see. Quitting the app does not:
        the layout was already saved and is restored next launch."""
        if not self._quitting:
            self._manager.return_all()
        super().closeEvent(event)


class PanelManager(QObject):
    """Moves docks between the main window and the workspace, and remembers where."""

    def __init__(self, viewer: Any) -> None:
        main = viewer.window._qt_window
        super().__init__(main)
        self._viewer = viewer
        self.main: QMainWindow = main
        self.main.setDockOptions(self.main.dockOptions() | _DOCK_OPTIONS)
        # Tabs on top on the right, where the nvitk panels used to be a tab widget:
        # the same place to look for them as before.
        self.main.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)
        self._tab_timer = QTimer(self)
        self._tab_timer.setSingleShot(True)
        self._tab_timer.setInterval(0)
        self._tab_timer.timeout.connect(self.tune_tab_bars)
        self.main.tabifiedDockWidgetActivated.connect(lambda _dock: self._tab_timer.start())
        self.workspace: WorkspaceWindow | None = None
        self._labels_dock: QDockWidget | None = None
        self.theme_button: Any | None = None
        self.labels_action = QAction("Labels panel", self.main)
        self.labels_action.setShortcut(QKeySequence(LABELS_SHORTCUT))
        self.labels_action.setShortcutContext(Qt.ApplicationShortcut)
        self.labels_action.setToolTip("Show the label selection panel and put the cursor in its filter.")
        self.labels_action.triggered.connect(self.show_labels)
        # Owned by the main window, not by a menu: the menus are rebuilt every time
        # they open, and the shortcut must outlive that.
        self.main.addAction(self.labels_action)

    def tune_tab_bars(self) -> None:
        """Full panel names with scroll arrows, rather than "To…", "DI…".

        Qt creates a window's dock tab bars as docks get tabbed together, with
        eliding on: eleven panels in a sidebar then read as initials. Run again
        whenever docks move, since moving can create a new bar.
        """
        windows = [self.main] + ([self.workspace] if self.workspace is not None else [])
        for window in windows:
            for bar in window.findChildren(QTabBar):
                if bar.parentWidget() is not window:
                    continue  # a tab widget inside some panel, not a dock group
                bar.setElideMode(Qt.ElideNone)
                bar.setUsesScrollButtons(True)
                bar.setExpanding(False)

    def watch_dock(self, dock: QDockWidget) -> None:
        """Re-tune the tab bars whenever *dock* is docked, undocked or moved."""
        if getattr(dock, "_nvitk_tab_watch", False):
            return
        dock.dockLocationChanged.connect(lambda _area: self._tab_timer.start())
        dock.topLevelChanged.connect(lambda _floating: self._tab_timer.start())
        dock._nvitk_tab_watch = True

    def watch_all(self) -> None:
        """:meth:`watch_dock` every dock there is now, and tune the bars once."""
        for dock in self.all_docks():
            self.watch_dock(dock)
        self._tab_timer.start()

    # -- lookup ---------------------------------------------------------------

    def all_docks(self) -> list[QDockWidget]:
        """Every dock in either window (the workspace is a child of the main one)."""
        return [d for d in self.main.findChildren(QDockWidget) if d.objectName()]

    def owner(self, dock: QDockWidget) -> QMainWindow | None:
        """The window *dock* belongs to — floating docks included."""
        parent = dock.parentWidget()
        while parent is not None and not isinstance(parent, QMainWindow):
            parent = parent.parentWidget()
        return parent

    def docks_in(self, window: QMainWindow) -> list[QDockWidget]:
        """The docks *window* holds."""
        return [d for d in self.all_docks() if self.owner(d) is window]

    def find(self, object_name: str) -> QDockWidget | None:
        """The dock saved under *object_name*."""
        for dock in self.all_docks():
            if dock.objectName() == object_name:
                return dock
        return None

    def in_workspace(self, dock: QDockWidget) -> bool:
        return self.workspace is not None and self.owner(dock) is self.workspace

    # -- moving ---------------------------------------------------------------

    def ensure_workspace(self, *, show: bool = True) -> WorkspaceWindow:
        """The workspace window, created on first use."""
        if self.workspace is None:
            from nvitk.gui.core.design import apply_theme

            self.workspace = WorkspaceWindow(self, self.main)
            apply_theme(self.workspace)
        if show:
            self.workspace.show()
            self.workspace.raise_()
            self.workspace.activateWindow()
        return self.workspace

    @staticmethod
    def _home_area(dock: QDockWidget) -> Any:
        """Where *dock* goes when it comes back to the main window."""
        area = getattr(dock, "_nvitk_home_area", None)
        if area is None:
            area = getattr(dock, "qt_area", None)  # Napari's own docks
        return area if area is not None else Qt.RightDockWidgetArea

    def _move(self, dock: QDockWidget, target: QMainWindow, *, arrange: bool = True) -> None:
        """Re-home *dock* in *target*.

        With *arrange*, a panel coming into the workspace is split in beside the
        last one there, and one coming home is tabbed into the group on its home
        edge — so neither lands on top of another or as a sliver.
        """
        source = self.owner(dock)
        if source is target:
            dock.show()
            dock.raise_()
            return
        if source is not None:
            source.removeDockWidget(dock)
        dock.setParent(target)
        if target is self.workspace:
            peers = [d for d in self.docks_in(target) if d is not dock and not d.isFloating()]
            if arrange and peers:
                target.addDockWidget(target.dockWidgetArea(peers[-1]), dock)
                target.splitDockWidget(peers[-1], dock, Qt.Horizontal)
            else:
                target.addDockWidget(Qt.LeftDockWidgetArea, dock)
        else:
            area = self._home_area(dock)
            peers = [
                d for d in self.docks_in(target)
                if d is not dock and not d.isFloating() and target.dockWidgetArea(d) == area
                and not d.isHidden()
            ]
            target.addDockWidget(area, dock)
            if arrange and peers:
                target.tabifyDockWidget(peers[-1], dock)
        dock.setFloating(False)
        dock.show()
        dock.raise_()
        self.watch_dock(dock)
        self._tab_timer.start()
        if self.workspace is not None:
            self.workspace.sync_hint()

    def move_to_workspace(self, dock: QDockWidget) -> None:
        """Put *dock* in the workspace window, opening it if needed."""
        self._move(dock, self.ensure_workspace(show=True))

    def move_to_main(self, dock: QDockWidget) -> None:
        """Bring *dock* back to the main window."""
        self._move(dock, self.main)

    def move_all_to_workspace(self) -> None:
        """Every visible panel into the workspace."""
        workspace = self.ensure_workspace(show=True)
        for dock in self.docks_in(self.main):
            if not dock.isHidden():
                self._move(dock, workspace)

    def return_all(self) -> None:
        """Every panel in the workspace back to the main window."""
        if self.workspace is None:
            return
        for dock in self.docks_in(self.workspace):
            self._move(dock, self.main)

    def dock_all_floating(self) -> None:
        """Re-dock every floating panel in the window it belongs to."""
        for dock in self.all_docks():
            if dock.isFloating():
                dock.setFloating(False)

    @staticmethod
    def toggle_float(dock: QDockWidget) -> None:
        """Pop *dock* out, or dock it back."""
        toggle = getattr(dock, "_nvitk_float_toggle", None)
        if toggle is not None:
            toggle()
            return
        dock.setFloating(not dock.isFloating())
        dock.show()
        dock.raise_()

    def show_dock(self, dock: QDockWidget) -> None:
        """Make *dock* the one on screen in its tab group, and its window visible."""
        window = self.owner(dock)
        if window is self.workspace and self.workspace is not None:
            self.workspace.show()
        dock.show()
        dock.raise_()
        if self.workspace is not None:
            self.workspace.sync_hint()

    # -- labels ---------------------------------------------------------------

    def set_labels_dock(self, dock: QDockWidget) -> None:
        self._labels_dock = dock

    def show_labels(self) -> None:
        """Raise the Labels panel, wherever it is, and focus its filter."""
        dock = self._labels_dock
        if dock is None:
            return
        self.show_dock(dock)
        panel = getattr(dock, "_nvitk_labels_panel", None)
        if panel is not None:
            panel.focus_filter()

    # -- menu -----------------------------------------------------------------

    def attach_menu(self, menu: QMenu) -> None:
        """Fill *menu* from the live dock list each time it opens."""
        menu.aboutToShow.connect(lambda: self._rebuild_menu(menu))
        self._rebuild_menu(menu)

    @staticmethod
    def _title(dock: QDockWidget) -> str:
        title = dock.windowTitle() or dock.objectName()
        return title[:1].upper() + title[1:]

    def _rebuild_menu(self, menu: QMenu) -> None:
        menu.clear()
        menu.addAction(self.labels_action)
        menu.addSeparator()
        ws_open = self.workspace is not None and self.workspace.isVisible()
        open_ws = menu.addAction("Show workspace window" if self.workspace else "Open workspace window")
        open_ws.setEnabled(not ws_open)
        open_ws.triggered.connect(lambda: self.ensure_workspace(show=True))
        menu.addAction("Move all panels to workspace").triggered.connect(self.move_all_to_workspace)
        back = menu.addAction("Return all panels to main window")
        back.setEnabled(self.workspace is not None and bool(self.docks_in(self.workspace)))
        back.triggered.connect(self.return_all)
        floating = any(d.isFloating() for d in self.all_docks())
        dock_all = menu.addAction("Dock all floating panels")
        dock_all.setEnabled(floating)
        dock_all.triggered.connect(self.dock_all_floating)
        menu.addSeparator()

        docks = sorted(
            self.all_docks(),
            key=lambda d: (not d.objectName().startswith("nvitk"), self._title(d).lower()),
        )
        for dock in docks:
            where = "workspace" if self.in_workspace(dock) else "main"
            if dock.isFloating():
                where += ", floating"
            sub = menu.addMenu(f"{self._title(dock)}    ({where})")
            shown = sub.addAction("Shown")
            shown.setCheckable(True)
            shown.setChecked(not dock.isHidden())
            shown.toggled.connect(
                lambda on, d=dock: self.show_dock(d) if on else d.close()
            )
            pop = sub.addAction("Dock back in place" if dock.isFloating() else "Pop out")
            pop.triggered.connect(lambda _=False, d=dock: self.toggle_float(d))
            if self.in_workspace(dock):
                sub.addAction("Move to main window").triggered.connect(
                    lambda _=False, d=dock: self.move_to_main(d)
                )
            else:
                sub.addAction("Move to workspace window").triggered.connect(
                    lambda _=False, d=dock: self.move_to_workspace(d)
                )

        menu.addSeparator()
        from nvitk.gui.core.design import active_theme, toggle_theme

        theme = menu.addAction(
            "Switch to light theme" if active_theme() == "dark" else "Switch to dark theme"
        )
        # Through the title-bar toggle when there is one, so its label follows.
        button = self.theme_button
        theme.triggered.connect(button.click if button is not None else lambda: toggle_theme(self._viewer))

    # -- persistence ----------------------------------------------------------

    def layout_state(self) -> dict[str, Any]:
        """The workspace's part of the saved layout."""
        if self.workspace is None:
            return {"open": False, "docks": []}
        docks = [d.objectName() for d in self.docks_in(self.workspace)]
        return {
            "open": bool(self.workspace.isVisible()),
            "docks": docks,
            "state": bytes(self.workspace.saveState().toBase64()).decode("ascii"),
            "geometry": bytes(self.workspace.saveGeometry().toBase64()).decode("ascii"),
        }

    def save(self) -> bool:
        """Store the workspace layout in the GUI preferences."""
        from nvitk.gui.core.prefs import save_prefs

        try:
            return save_prefs({WORKSPACE_PREF_KEY: self.layout_state()})
        except Exception:  # noqa: BLE001 — never block a close on a preference
            return False

    def restore(self) -> bool:
        """Re-create the workspace the last session ended with.

        Must run after every dock exists and *before* the main window's
        ``restoreState``: the docks it claims are taken out of the main window
        first, so the main layout is restored around what is left.
        """
        from nvitk.gui.core.prefs import load_prefs

        saved = load_prefs().get(WORKSPACE_PREF_KEY)
        if not isinstance(saved, dict):
            return False
        names = [str(n) for n in saved.get("docks") or []]
        docks = [d for d in (self.find(n) for n in names) if d is not None]
        if not docks:
            return False
        workspace = self.ensure_workspace(show=False)
        for dock in docks:
            self._move(dock, workspace, arrange=False)
        try:
            state = str(saved.get("state") or "")
            if state:
                workspace.restoreState(QByteArray.fromBase64(state.encode("ascii")))
            geometry = str(saved.get("geometry") or "")
            if geometry:
                workspace.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
        except Exception:  # noqa: BLE001 — a layout from another Qt build
            pass
        workspace.sync_hint()
        if saved.get("open", True):
            workspace.show()
        return True

    def shutdown(self) -> None:
        """Called on application exit, after :meth:`save`: close the workspace
        without handing its panels back, so the app can quit."""
        if self.workspace is not None:
            self.workspace._quitting = True
            self.workspace.close()


def install_panel_manager(viewer: Any) -> PanelManager | None:
    """Create the manager and add its **Panels** menu to the main window."""
    try:
        manager = PanelManager(viewer)
        menu = manager.main.menuBar().addMenu("&Panels")
        manager.attach_menu(menu)
    except Exception:  # noqa: BLE001 — the GUI still works without it
        return None
    viewer._nvitk_panel_manager = manager
    return manager


__all__ = [
    "LABELS_SHORTCUT",
    "PanelManager",
    "WORKSPACE_PREF_KEY",
    "WorkspaceWindow",
    "install_panel_manager",
    "make_panel_dock",
]
