"""Every panel a dock, in the main window or in panel windows of its own.

Each nvitk panel (Imaging, Labels, DICOM browser, …) is its own dock, tabbed
together on the right by default, and Napari's own docks (layer list, layer
controls, console, plugins) and the orthogonal views are treated the same way.
Any of them can be:

* **popped out** into a *panel window* of its own — the title-bar ⧉, a
  double-click on its title or its **tab**, or the right-click menu — while the
  other panels stay where they are; ⧉ again puts it back at the tab it came from;
* **joined with other panels** in a panel window, on any screen: right-click a
  panel's tab or title bar ▸ *Move to window* ▸ a window (as a tab, or beside the
  panels there), or drag the panel by its title bar and drop it on that window
  (or on another popped-out panel, which makes a window of the two); inside a
  window, panels are tabbed and split by dragging, as in the main window;
* tabbed in with the nvitk panels — Napari's layer list and layer controls too —
  or sent back to Napari's left edge;
* hidden and shown again from the **Panels** menu, which every window carries.

Dragging a panel's title bar takes that panel alone, not its whole tab group
(Qt's ``GroupedDragging`` is off). Qt cannot drag a dock from one window into
another by itself; the drop on another window is recognised here when the drag
is Qt's own (nvitk's title bars). Napari's docks float with the system's title
bar, whose drags Qt never sees: for them, use the menu.

The arrangement is remembered: the main window's ``saveState`` as before, plus
every panel window — its panels, its own ``saveState`` and its geometry.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import QByteArray, QEvent, QObject, Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMenu,
    QTabBar,
    QTabWidget,
    QWidget,
)

#: Preferences key of the panel windows (a list, one entry per window).
WINDOWS_PREF_KEY = "panel_windows"
#: The single "workspace" window of earlier versions, read once and then cleared.
WORKSPACE_PREF_KEY = "workspace_layout"
WORKSPACE_OBJECT_NAME = "nvitkWorkspace"
LABELS_SHORTCUT = "Ctrl+Shift+L"

_DOCK_OPTIONS = (
    QMainWindow.DockOption.AnimatedDocks
    | QMainWindow.DockOption.AllowNestedDocks
    | QMainWindow.DockOption.AllowTabbedDocks
)
#: Off on purpose: with it, dragging a panel's title bar drags every panel tabbed
#: with it — the whole control panel pops out instead of the one asked for.
_GROUPED = QMainWindow.DockOption.GroupedDragging

TAB_HINT = ("Double-click a tab to pop that panel out into a window of its own; right-click it to "
            "move it into another window, beside or tabbed with the panels there.")


def move_tab_after(window: QMainWindow, anchor: QDockWidget, dock: QDockWidget) -> bool:
    """Put *dock*'s tab right after *anchor*'s in *window* (tabbing it in first if needed).

    ``tabifyDockWidget`` only ever appends, so *dock* and every tab that followed
    *anchor* are re-appended in order. The tab that was showing stays showing.
    """
    from qtpy.QtCore import QCoreApplication

    if anchor.isFloating() or dock.isFloating() or anchor is dock:
        return False
    if dock not in window.tabifiedDockWidgets(anchor):
        if not window.tabifiedDockWidgets(anchor) and window.dockWidgetArea(anchor) == Qt.NoDockWidgetArea:
            return False
        window.tabifyDockWidget(anchor, dock)
        QCoreApplication.processEvents()
        if dock not in window.tabifiedDockWidgets(anchor):
            return False
    bar = next(
        (tb for tb in window.findChildren(QTabBar)
         if {anchor.windowTitle(), dock.windowTitle()} <= {tb.tabText(i) for i in range(tb.count())}),
        None,
    )
    if bar is None:
        return False
    group = {d.windowTitle(): d for d in (anchor, *window.tabifiedDockWidgets(anchor))}
    titles = [bar.tabText(i) for i in range(bar.count())]
    current = group.get(bar.tabText(bar.currentIndex()))
    order = [group[t] for t in titles if t in group]
    trailing = [d for d in order[order.index(anchor) + 1:] if d is not dock]
    window.tabifyDockWidget(anchor, dock)
    for other in trailing:
        window.tabifyDockWidget(anchor, other)
    if current is not None:
        current.raise_()
    return True


#: Napari's docks that live on its left edge, layer controls above the layer list.
_NAPARI_LEFT = ("layer controls", "layer list")


class _TitleBarMenu(QObject):
    """A dock's title bar — nvitk's or Napari's: right-click opens the panel menu, and the
    end of a title-bar drag is checked for a drop on another window."""

    def __init__(self, manager: "PanelManager") -> None:
        super().__init__(manager)
        self._manager = manager

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 — Qt naming
        if not isinstance(obj, QDockWidget):
            return False
        kind = event.type()
        if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease) and event.button() == Qt.LeftButton:
            try:
                pos = event.globalPosition().toPoint()
            except AttributeError:
                pos = event.globalPos()
            if kind == QEvent.Type.MouseButtonPress:
                obj._nvitk_press_pos = pos
                return False
            start = getattr(obj, "_nvitk_press_pos", None)
            obj._nvitk_press_pos = None
            if start is not None and (pos - start).manhattanLength() >= QApplication.startDragDistance():
                # A drag, not a click: after Qt has finished its own drop (it may
                # have docked the panel itself), see whether it landed on another window.
                QTimer.singleShot(0, lambda d=obj, p=pos: self._manager.after_drag(d, p))
            return False
        if kind != QEvent.Type.ContextMenu:
            return False
        bar = obj.titleBarWidget()
        if bar is None or not bar.isVisible() or not bar.geometry().contains(event.pos()):
            return False  # a right-click inside the panel itself is the panel's business
        self._manager.tab_menu(obj).exec(event.globalPos())
        return True


class _TabBarPopOut(QObject):
    """Per-tab pop-out on the windows' dock tab bars: double-click, or the right-click menu."""

    def __init__(self, manager: "PanelManager") -> None:
        super().__init__(manager)
        self._manager = manager

    @staticmethod
    def _index(bar: QTabBar, event: Any) -> int:
        try:
            point = event.position().toPoint()
        except AttributeError:
            point = event.pos()
        return bar.tabAt(point)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 — Qt naming
        if not isinstance(obj, QTabBar):
            return False
        kind = event.type()
        if kind == QEvent.Type.MouseButtonDblClick and event.button() == Qt.LeftButton:
            dock = self._manager.dock_for_tab(obj, self._index(obj, event))
            if dock is not None:
                self._manager.pop_out(dock)
                return True
        elif kind == QEvent.Type.ContextMenu:
            dock = self._manager.dock_for_tab(obj, obj.tabAt(event.pos()))
            if dock is not None:
                self._manager.tab_menu(dock).exec(event.globalPos())
                return True
        return False


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


class PanelWindow(QMainWindow):
    """A top-level window holding whichever panels are put in it — on any screen."""

    def __init__(self, manager: "PanelManager", parent: QWidget | None = None, *, name: str = "") -> None:
        super().__init__(parent, Qt.Window)
        self._manager = manager
        self._quitting = False
        self.setObjectName(name or manager._next_window_name())
        self.setWindowTitle("nvitk")
        self.setDockOptions(_DOCK_OPTIONS)  # no GroupedDragging: one panel per drag
        self.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
        self.tabifiedDockWidgetActivated.connect(lambda _dock: manager._tab_timer.start())
        self.setCentralWidget(self._make_hint())
        self.resize(1100, 750)
        menu = self.menuBar().addMenu("&Panels")
        manager.attach_menu(menu)

    @staticmethod
    def _make_hint() -> QLabel:
        hint = QLabel(
            "Empty window.\n\nRight-click a panel's tab or title bar ▸ Move to window ▸ this window "
            "to bring it here — or drag it here by its title bar — then drag panels by their title "
            "bars to split or tab them."
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        return hint

    def sync_hint(self) -> None:
        """Show the empty-window hint only while there is nothing to show.

        Taken out rather than hidden: with no central widget the docks share the
        whole window.
        """
        has_docks = any(not d.isHidden() for d in self._manager.docks_in(self))
        central = self.centralWidget()
        if has_docks and central is not None:
            self.takeCentralWidget().deleteLater()
        elif not has_docks and central is None:
            self.setCentralWidget(self._make_hint())

    def sync_title(self) -> None:
        names = [self._manager._title(d) for d in self._manager.docks_in(self) if not d.isHidden()]
        text = " · ".join(names[:4]) + (f" +{len(names) - 4}" if len(names) > 4 else "")
        self.setWindowTitle(f"nvitk — {text}" if text else "nvitk — empty window")

    def closeEvent(self, event: Any) -> None:
        """Closing a panel window gives its panels back to the main window, so
        nothing is lost behind a window nobody can see. Quitting the app does not:
        the layout was already saved and is restored next launch."""
        if not self._quitting:
            self._manager.return_window(self)
        super().closeEvent(event)


#: The earlier name of :class:`PanelWindow`.
WorkspaceWindow = PanelWindow


class PanelManager(QObject):
    """Moves docks between the main window and panel windows, and remembers where."""

    def __init__(self, viewer: Any) -> None:
        main = viewer.window._qt_window
        super().__init__(main)
        self._viewer = viewer
        self.main: QMainWindow = main
        self.main.setDockOptions((self.main.dockOptions() | _DOCK_OPTIONS) & ~_GROUPED)
        self._tab_filter = _TabBarPopOut(self)
        self._title_filter = _TitleBarMenu(self)
        # Tabs on top on the right, where the nvitk panels used to be a tab widget:
        # the same place to look for them as before.
        self.main.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)
        self._tab_timer = QTimer(self)
        self._tab_timer.setSingleShot(True)
        self._tab_timer.setInterval(0)
        self._tab_timer.timeout.connect(self._refresh)
        self.main.tabifiedDockWidgetActivated.connect(lambda _dock: self._tab_timer.start())
        self.windows: list[PanelWindow] = []
        self._window_count = 0
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

    @property
    def workspace(self) -> PanelWindow | None:
        """The first panel window (the "workspace" of earlier versions)."""
        return self.windows[0] if self.windows else None

    def _refresh(self) -> None:
        self.tune_tab_bars()
        self._sync_windows()

    def _sync_windows(self) -> None:
        """Titles and hints of the panel windows; a window left empty closes itself."""
        for window in list(self.windows):
            if not self.docks_in(window):
                if getattr(window, "_nvitk_had_docks", False):
                    self._drop_window(window)
                else:
                    window.sync_hint()
                continue
            window.sync_hint()
            window.sync_title()

    def _drop_window(self, window: PanelWindow) -> None:
        if window in self.windows:
            self.windows.remove(window)
        window._quitting = True
        window.close()
        window.deleteLater()

    def _next_window_name(self) -> str:
        self._window_count += 1
        return f"nvitkPanelWindow{self._window_count}"

    def tune_tab_bars(self) -> None:
        """Full panel names with scroll arrows, rather than "To…", "DI…".

        Qt creates a window's dock tab bars as docks get tabbed together, with
        eliding on: eleven panels in a sidebar then read as initials. Run again
        whenever docks move, since moving can create a new bar.
        """
        for window in [self.main, *self.windows]:
            for bar in window.findChildren(QTabBar):
                if bar.parentWidget() is not window:
                    continue  # a tab widget inside some panel, not a dock group
                bar.setElideMode(Qt.ElideNone)
                bar.setUsesScrollButtons(True)
                bar.setExpanding(False)
                if not bar.property("nvitk_popout"):
                    bar.installEventFilter(self._tab_filter)
                    bar.setProperty("nvitk_popout", True)
                    bar.setToolTip(TAB_HINT)

    def watch_dock(self, dock: QDockWidget) -> None:
        """Re-tune the tab bars whenever *dock* is docked, undocked or moved."""
        if getattr(dock, "_nvitk_tab_watch", False):
            return
        dock.dockLocationChanged.connect(lambda _area: self._tab_timer.start())
        dock.topLevelChanged.connect(lambda _floating: self._tab_timer.start())
        dock.visibilityChanged.connect(lambda _visible: self._tab_timer.start())
        dock._nvitk_tab_watch = True
        dock.installEventFilter(self._title_filter)
        # The title bar's ⧉ and double-click go through here too, so a panel goes
        # into a window of its own, and comes back at the tab it left.
        dock._nvitk_managed_toggle = lambda d=dock: self.toggle_float(d)

    def watch_all(self) -> None:
        """:meth:`watch_dock` every dock there is now, and tune the bars once."""
        for dock in self.all_docks():
            self.watch_dock(dock)
        self._tab_timer.start()

    # -- lookup ---------------------------------------------------------------

    def all_docks(self) -> list[QDockWidget]:
        """Every dock in every window (the panel windows are children of the main one)."""
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

    def window_of(self, dock: QDockWidget) -> PanelWindow | None:
        """The panel window *dock* is in, or ``None`` (the main window)."""
        owner = self.owner(dock)
        return owner if isinstance(owner, PanelWindow) else None

    def in_workspace(self, dock: QDockWidget) -> bool:
        """In a panel window rather than the main one."""
        return self.window_of(dock) is not None

    # -- moving ---------------------------------------------------------------

    def new_window(self, *, show: bool = True, name: str = "", like: QWidget | None = None) -> PanelWindow:
        """A new, empty panel window (placed over *like* when given, else half the screen)."""
        from nvitk.gui.core.design import apply_theme

        window = PanelWindow(self, self.main, name=name)
        if name:
            # Restored windows keep their numbers; new ones go after the highest.
            digits = "".join(ch for ch in name if ch.isdigit())
            self._window_count = max(self._window_count, int(digits) if digits else 0)
        apply_theme(window)
        self.windows.append(window)
        if like is not None:
            window.setGeometry(like.geometry())
        else:
            screen = (self.main.screen() or QApplication.primaryScreen())
            if screen is not None:
                rect = screen.availableGeometry()
                window.setGeometry(rect.x() + rect.width() // 4, rect.y() + rect.height() // 6,
                                   max(rect.width() // 2, 520), max(int(rect.height() * 0.66), 400))
        if show:
            window.show()
            window.raise_()
            window.activateWindow()
        return window

    def ensure_workspace(self, *, show: bool = True) -> PanelWindow:
        """The first panel window, created when there is none."""
        window = self.workspace or self.new_window(show=show)
        if show:
            window.show()
            window.raise_()
        return window

    @staticmethod
    def _home_area(dock: QDockWidget) -> Any:
        """Where *dock* goes when it comes back to the main window."""
        area = getattr(dock, "_nvitk_home_area", None)
        if area is None:
            area = getattr(dock, "qt_area", None)  # Napari's own docks
        return area if area is not None else Qt.RightDockWidgetArea

    def _move(self, dock: QDockWidget, target: QMainWindow, *, arrange: bool = True,
              tab_with: QDockWidget | None = None) -> None:
        """Re-home *dock* in *target*.

        *tab_with*: put it in that dock's tab group. Otherwise, with *arrange*, a
        panel coming into a panel window is split in beside the last one there, and
        one coming home is tabbed into the group on its home edge — so neither
        lands on top of another or as a sliver.
        """
        source = self.owner(dock)
        if source is target and not dock.isFloating():
            if tab_with is not None and tab_with is not dock:
                target.tabifyDockWidget(tab_with, dock)
            dock.show()
            dock.raise_()
            return
        if source is not None and source is not target:
            source.removeDockWidget(dock)
        if dock.parentWidget() is not target:
            dock.setParent(target)
        dock.setFloating(False)
        if tab_with is not None and tab_with is not dock and self.owner(tab_with) is target \
                and not tab_with.isFloating():
            target.addDockWidget(target.dockWidgetArea(tab_with), dock)
            target.tabifyDockWidget(tab_with, dock)
        elif isinstance(target, PanelWindow):
            peers = [d for d in self.docks_in(target) if d is not dock and not d.isFloating() and not d.isHidden()]
            if arrange and peers:
                # Appended to the area, to the right of what is there: splitting next to
                # one panel would pull it out of the tab group it is in.
                target.addDockWidget(target.dockWidgetArea(peers[-1]), dock, Qt.Horizontal)
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
        dock.show()
        dock.raise_()
        self.watch_dock(dock)
        self._tab_timer.start()
        if isinstance(target, PanelWindow):
            target._nvitk_had_docks = True
            target.sync_hint()
            target.sync_title()
            target.show()
        sync_buttons = getattr(dock, "_nvitk_sync_buttons", None)
        if sync_buttons is not None:
            sync_buttons()

    def move_to_window(self, dock: QDockWidget, window: PanelWindow | None, *, tab: bool = True) -> PanelWindow:
        """Put *dock* in panel *window* (``None``: a new one) — as a tab of the panel showing
        there, or beside the panels there (*tab* false)."""
        if window is None or window not in self.windows:
            self._remember_tab(dock)
            window = self.new_window(show=True)
            self._move(dock, window)
            return window
        peers = [d for d in self.docks_in(window) if d is not dock and not d.isFloating() and not d.isHidden()]
        # The panel on show there (the current tab); a window not drawn yet shows none, so the first.
        showing = [d for d in peers if d.isVisible()] or peers
        self._move(dock, window, tab_with=(showing[0] if (tab and showing) else None))
        window.show()
        window.raise_()
        return window

    def move_to_workspace(self, dock: QDockWidget) -> None:
        """Put *dock* in the first panel window (a new one when there is none)."""
        self.move_to_window(dock, self.workspace, tab=False)

    def move_to_main(self, dock: QDockWidget) -> None:
        """Bring *dock* back to the main window — at the tab it left, when known."""
        self._move(dock, self.main)
        self._back_to_tab(dock)

    def move_all_to_workspace(self) -> None:
        """Every visible panel of the main window into one new panel window."""
        window = self.new_window(show=True)
        for dock in self.docks_in(self.main):
            if not dock.isHidden():
                self._remember_tab(dock)
                self._move(dock, window)

    def return_window(self, window: PanelWindow) -> None:
        """Every panel of *window* back to the main window (the window then closes)."""
        for dock in self.docks_in(window):
            self.move_to_main(dock)
        if window in self.windows:
            self.windows.remove(window)
            window.deleteLater()

    def return_all(self) -> None:
        """Every panel of every panel window back to the main window."""
        for window in list(self.windows):
            self.return_window(window)

    def dock_all_floating(self) -> None:
        """Re-dock every floating panel in the window it belongs to, at its old tab."""
        for dock in self.all_docks():
            if dock.isFloating():
                self.dock_back(dock)

    def toggle_float(self, dock: QDockWidget) -> None:
        """⧉: pop *dock* out into a window of its own, or bring it back where it was."""
        if dock.isFloating():
            self.dock_back(dock)
        elif self.window_of(dock) is not None:
            self.move_to_main(dock)
        else:
            self.pop_out(dock)

    def _tab_neighbours(self, dock: QDockWidget) -> tuple[QDockWidget | None, QDockWidget | None]:
        """The docks before and after *dock* in its tab bar (``None`` at the ends)."""
        window = self.owner(dock)
        if window is None:
            return None, None
        group = {d.windowTitle(): d for d in (dock, *window.tabifiedDockWidgets(dock))}
        for bar in window.findChildren(QTabBar):
            titles = [bar.tabText(i) for i in range(bar.count())]
            if bar.parentWidget() is window and dock.windowTitle() in titles:
                k = titles.index(dock.windowTitle())
                before = group.get(titles[k - 1]) if k > 0 else None
                after = group.get(titles[k + 1]) if k + 1 < len(titles) else None
                return before, after
        return None, None

    def _remember_tab(self, dock: QDockWidget) -> None:
        """Note where *dock* sits in the main window, to put it back there later."""
        if self.owner(dock) is self.main and not dock.isFloating():
            dock._nvitk_tab_before, dock._nvitk_tab_after = self._tab_neighbours(dock)
            dock._nvitk_was_area = self.main.dockWidgetArea(dock)

    def pop_out(self, dock: QDockWidget) -> PanelWindow:
        """Put *dock* in a panel window of its own (half the screen, centred); the
        panels it was tabbed with stay where they are. Other panels can join it."""
        window = self.window_of(dock)
        if window is not None and len(self.docks_in(window)) == 1:
            window.show()
            window.raise_()
            return window
        return self.move_to_window(dock, None)

    def _back_to_tab(self, dock: QDockWidget) -> None:
        """Put *dock* (now in the main window) back at the tab it left, when known."""
        window = self.main
        before = getattr(dock, "_nvitk_tab_before", None)
        after = getattr(dock, "_nvitk_tab_after", None)
        if before is not None and self.owner(before) is window and not before.isFloating() and before is not dock:
            move_tab_after(window, before, dock)
        elif after is not None and self.owner(after) is window and not after.isFloating() and after is not dock:
            # It was the first tab: tab it in, then move every other tab behind it.
            if dock not in window.tabifiedDockWidgets(after):
                window.tabifyDockWidget(after, dock)
            self._move_first(window, dock)
        elif self.is_napari_left(dock) and getattr(dock, "_nvitk_was_area", None) == Qt.LeftDockWidgetArea:
            self.send_home(dock)
        dock.show()
        dock.raise_()
        self._tab_timer.start()

    def dock_back(self, dock: QDockWidget) -> None:
        """Dock a floating panel back into its window, at the tab it came from."""
        if not dock.isFloating():
            return
        toggle = getattr(dock, "_nvitk_float_toggle", None)
        if toggle is not None:
            toggle()
        else:
            dock.setFloating(False)
        if self.owner(dock) is self.main:
            self._back_to_tab(dock)
        dock.show()
        dock.raise_()
        self._tab_timer.start()

    def _move_first(self, window: QMainWindow, dock: QDockWidget) -> None:
        """Make *dock* the first tab of its group."""
        group = [dock, *window.tabifiedDockWidgets(dock)]
        bar = next((tb for tb in window.findChildren(QTabBar)
                    if tb.parentWidget() is window and dock.windowTitle() in
                    [tb.tabText(i) for i in range(tb.count())]), None)
        if bar is None:
            return
        by_title = {d.windowTitle(): d for d in group}
        order = [by_title[bar.tabText(i)] for i in range(bar.count()) if bar.tabText(i) in by_title]
        current = by_title.get(bar.tabText(bar.currentIndex()))
        for other in order:
            if other is not dock:
                window.tabifyDockWidget(dock, other)
        if current is not None:
            current.raise_()

    # -- dropping a dragged panel on another window ------------------------------

    def drop_target(self, pos: Any, dragged: QDockWidget) -> tuple[QWidget | None, QDockWidget | None]:
        """``(window or floating panel, panel under the pointer)`` at screen *pos*, ignoring
        *dragged* — where a title-bar drag of *dragged* ended."""
        # Floating panels sit above the windows, the newest windows above the older.
        candidates: list[QWidget] = [d for d in self.all_docks() if d.isFloating() and d is not dragged]
        candidates += [*reversed(self.windows), self.main]
        for widget in candidates:
            if widget is dragged or not widget.isVisible():
                continue
            if not widget.frameGeometry().contains(pos):
                continue
            if isinstance(widget, QDockWidget):
                return widget, widget
            under = next((d for d in self.docks_in(widget) if d is not dragged and not d.isFloating()
                          and not d.isHidden() and not d.visibleRegion().isEmpty()
                          and d.rect().contains(d.mapFromGlobal(pos))), None)
            return widget, under
        return None, None

    def after_drag(self, dock: QDockWidget, pos: Any) -> bool:
        """A title-bar drag of *dock* ended at *pos*: when Qt left it floating over another
        nvitk window, put it there (tabbed with the panel under the pointer); over another
        popped-out panel, make a window of the two. Returns whether it moved."""
        if not dock.isFloating() or dock not in self.all_docks():
            return False
        target, under = self.drop_target(pos, dock)
        if target is None:
            return False
        if isinstance(target, QDockWidget):
            window = self.new_window(show=True, like=target)
            self._move(target, window)
            self._move(dock, window, tab_with=target)
            return True
        if target is self.main:
            if under is None:
                return False  # over the canvas: it stays where it was dropped, floating
            self._move(dock, self.main, tab_with=under)
            return True
        self._move(dock, target, tab_with=under)
        target.raise_()
        return True

    # -- the nvitk tab group ---------------------------------------------------

    def panel_group_anchor(self) -> QDockWidget | None:
        """A dock of the main window's nvitk tab group (the Imaging panel when it is there)."""
        candidates = [d for d in self.docks_in(self.main)
                      if d.objectName().startswith("nvitk:") and not d.isFloating() and not d.isHidden()
                      and self.main.dockWidgetArea(d) != Qt.NoDockWidgetArea]
        tools = next((d for d in candidates if d.objectName() == "nvitk:tools"), None)
        if tools is not None:
            return tools
        tabbed = [d for d in candidates if self.main.tabifiedDockWidgets(d)]
        return tabbed[0] if tabbed else (candidates[0] if candidates else None)

    def in_panel_group(self, dock: QDockWidget) -> bool:
        anchor = self.panel_group_anchor()
        return anchor is not None and not dock.isFloating() and self.owner(dock) is self.main and (
            dock is anchor or dock in self.main.tabifiedDockWidgets(anchor))

    def tab_with_panels(self, dock: QDockWidget) -> None:
        """Make *dock* (e.g. Napari's layer list) one more tab of the nvitk panels."""
        if self.window_of(dock) is not None:
            self._move(dock, self.main, arrange=False)
        anchor = self.panel_group_anchor()
        if dock.isFloating():
            dock.setFloating(False)
        if anchor is None or anchor is dock:
            self.main.addDockWidget(Qt.RightDockWidgetArea, dock)
        else:
            self.main.tabifyDockWidget(anchor, dock)
        dock.show()
        dock.raise_()
        self.watch_dock(dock)
        self._tab_timer.start()

    @staticmethod
    def is_napari_left(dock: QDockWidget) -> bool:
        return dock.objectName() in _NAPARI_LEFT

    def send_home(self, dock: QDockWidget) -> None:
        """Put one of Napari's left-edge docks back on the left (controls above the list)."""
        if self.window_of(dock) is not None:
            self._move(dock, self.main, arrange=False)
        if dock.isFloating():
            dock.setFloating(False)
        others = {d.objectName(): d for d in self.docks_in(self.main) if self.is_napari_left(d) and d is not dock}
        partner = others.get("layer list" if dock.objectName() == "layer controls" else "layer controls")
        partner_home = partner is not None and not partner.isFloating() and \
            self.main.dockWidgetArea(partner) == Qt.LeftDockWidgetArea and not self.in_panel_group(partner)
        # Split beside the partner only when it stands alone: Qt turns a split next to
        # a tabbed dock (the controls behind the orthogonal views, say) into one more tab.
        partner_alone = partner_home and not self.main.tabifiedDockWidgets(partner)
        if dock.objectName() == "layer list" and partner_alone:
            self.main.splitDockWidget(partner, dock, Qt.Vertical)
        else:
            self.main.addDockWidget(Qt.LeftDockWidgetArea, dock, Qt.Vertical)
            if dock.objectName() == "layer controls" and partner_alone:
                self.main.splitDockWidget(dock, partner, Qt.Vertical)
        dock.show()
        dock.raise_()
        self._tab_timer.start()

    def at_home(self, dock: QDockWidget) -> bool:
        return (not dock.isFloating() and self.owner(dock) is self.main
                and self.main.dockWidgetArea(dock) == Qt.LeftDockWidgetArea and not self.in_panel_group(dock))

    def dock_for_tab(self, bar: QTabBar, index: int) -> QDockWidget | None:
        """The dock shown by tab *index* of a window's dock tab bar."""
        if index < 0:
            return None
        title = bar.tabText(index)
        window = bar.parentWidget()
        matches = [d for d in self.all_docks()
                   if d.windowTitle() == title and self.owner(d) is window and not d.isFloating()]
        return matches[0] if matches else None

    # -- menus ----------------------------------------------------------------

    def _window_label(self, window: PanelWindow) -> str:
        names = [self._title(d) for d in self.docks_in(window)]
        return " · ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "") or "empty window"

    def _add_dock_actions(self, menu: QMenu, dock: QDockWidget) -> None:
        """The actions for one panel, shared by its tab / title-bar menu and the Panels menu."""
        title = self._title(dock)
        home = self.window_of(dock)
        if dock.isFloating():
            menu.addAction(f"Dock “{title}” back").triggered.connect(lambda: self.dock_back(dock))
        elif home is not None:
            menu.addAction("Back to the main window").triggered.connect(lambda: self.move_to_main(dock))
            if len(self.docks_in(home)) > 1:
                menu.addAction(f"Pop out “{title}” into its own window").triggered.connect(
                    lambda: self.move_to_window(dock, None))
        else:
            menu.addAction(f"Pop out “{title}” into its own window").triggered.connect(lambda: self.pop_out(dock))
        expand = getattr(dock, "_nvitk_expand_toggle", None)
        if expand is not None:
            menu.addAction(f"Fill the screen with “{title}”").triggered.connect(lambda: expand())
        others = [w for w in self.windows if w is not home and self.docks_in(w)]
        move = menu.addMenu("Move to window")
        for window in others:
            label = self._window_label(window)
            move.addAction(f"{label} — as a tab").triggered.connect(
                lambda _=False, w=window: self.move_to_window(dock, w, tab=True))
            move.addAction(f"{label} — beside").triggered.connect(
                lambda _=False, w=window: self.move_to_window(dock, w, tab=False))
        if others:
            move.addSeparator()
        move.addAction("A new window").triggered.connect(lambda: self.move_to_window(dock, None))
        if home is not None or dock.isFloating():
            move.addAction("The main window").triggered.connect(lambda: self.move_to_main(dock))
        menu.addSeparator()
        if not self.in_panel_group(dock):
            menu.addAction("Tab with the nvitk panels").triggered.connect(lambda: self.tab_with_panels(dock))
        if self.is_napari_left(dock) and not self.at_home(dock):
            menu.addAction("Back to napari's left side").triggered.connect(lambda: self.send_home(dock))

    def tab_menu(self, dock: QDockWidget) -> QMenu:
        """What a right-click on *dock*'s tab or title bar offers."""
        menu = QMenu(self.owner(dock) or self.main)
        self._add_dock_actions(menu, dock)
        if any(d.isFloating() for d in self.all_docks()):
            menu.addAction("Dock all floating panels").triggered.connect(self.dock_all_floating)
        if dock.features() & QDockWidget.DockWidgetFeature.DockWidgetClosable or dock.objectName().startswith("nvitk:"):
            menu.addSeparator()
            menu.addAction(f"Close “{self._title(dock)}”").triggered.connect(dock.close)
        return menu

    def show_dock(self, dock: QDockWidget) -> None:
        """Make *dock* the one on screen in its tab group, and its window visible."""
        window = self.window_of(dock)
        if window is not None:
            window.show()
        dock.show()
        dock.raise_()
        self._tab_timer.start()

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
        menu.addAction("New empty window").triggered.connect(lambda: self.new_window(show=True))
        menu.addAction("Move all panels to a new window").triggered.connect(self.move_all_to_workspace)
        back = menu.addAction("Return all panels to the main window")
        back.setEnabled(any(self.docks_in(w) for w in self.windows))
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
        numbering = {w: k for k, w in enumerate(self.windows, 1)}
        for dock in docks:
            window = self.window_of(dock)
            where = f"window {numbering.get(window, '?')}" if window is not None else "main"
            if dock.isFloating():
                where += ", floating"
            sub = menu.addMenu(f"{self._title(dock)}    ({where})")
            shown = sub.addAction("Shown")
            shown.setCheckable(True)
            shown.setChecked(not dock.isHidden())
            shown.toggled.connect(
                lambda on, d=dock: self.show_dock(d) if on else d.close()
            )
            self._add_dock_actions(sub, dock)

        menu.addSeparator()
        from nvitk.gui.core.design import active_theme, toggle_theme

        theme = menu.addAction(
            "Switch to light theme" if active_theme() == "dark" else "Switch to dark theme"
        )
        # Through the title-bar toggle when there is one, so its label follows.
        button = self.theme_button
        theme.triggered.connect(button.click if button is not None else lambda: toggle_theme(self._viewer))

    # -- persistence ----------------------------------------------------------

    @staticmethod
    def _encode(data: Any) -> str:
        return bytes(data.toBase64()).decode("ascii")

    def layout_state(self) -> list[dict[str, Any]]:
        """Every panel window: its panels, layout and geometry."""
        out = []
        for window in self.windows:
            docks = [d.objectName() for d in self.docks_in(window)]
            if not docks:
                continue
            out.append({
                "name": window.objectName(),
                "open": bool(window.isVisible()),
                "docks": docks,
                "state": self._encode(window.saveState()),
                "geometry": self._encode(window.saveGeometry()),
            })
        return out

    def save(self) -> bool:
        """Store the panel windows in the GUI preferences."""
        from nvitk.gui.core.prefs import save_prefs

        try:
            # The single-workspace entry of earlier versions is read once, then retired.
            return save_prefs({WINDOWS_PREF_KEY: self.layout_state(), WORKSPACE_PREF_KEY: {"docks": []}})
        except Exception:  # noqa: BLE001 — never block a close on a preference
            return False

    def restore(self) -> bool:
        """Re-create the panel windows the last session ended with.

        Must run after every dock exists and *before* the main window's
        ``restoreState``: the docks a window claims are taken out of the main
        window first, so the main layout is restored around what is left.
        """
        from nvitk.gui.core.prefs import load_prefs

        prefs = load_prefs()
        saved = prefs.get(WINDOWS_PREF_KEY)
        if not isinstance(saved, list):
            legacy = prefs.get(WORKSPACE_PREF_KEY)
            saved = [legacy] if isinstance(legacy, dict) and legacy.get("docks") else []
        restored = False
        for entry in saved:
            if not isinstance(entry, dict):
                continue
            docks = [d for d in (self.find(str(n)) for n in entry.get("docks") or []) if d is not None]
            docks = [d for d in docks if self.window_of(d) is None]
            if not docks:
                continue
            window = self.new_window(show=False, name=str(entry.get("name") or ""))
            for dock in docks:
                self._move(dock, window, arrange=False)
            try:
                state = str(entry.get("state") or "")
                if state:
                    window.restoreState(QByteArray.fromBase64(state.encode("ascii")))
                geometry = str(entry.get("geometry") or "")
                if geometry:
                    window.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
            except Exception:  # noqa: BLE001 — a layout from another Qt build
                pass
            window.sync_hint()
            window.sync_title()
            if entry.get("open", True):
                window.show()
            restored = True
        return restored

    def shutdown(self) -> None:
        """Called on application exit, after :meth:`save`: close the panel windows
        without handing their panels back, so the app can quit."""
        for window in list(self.windows):
            window._quitting = True
            window.close()


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
    "TAB_HINT",
    "WINDOWS_PREF_KEY",
    "move_tab_after",
    "PanelManager",
    "PanelWindow",
    "WORKSPACE_PREF_KEY",
    "WorkspaceWindow",
    "install_panel_manager",
    "make_panel_dock",
]
