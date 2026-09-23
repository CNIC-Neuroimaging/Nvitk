"""
Session tabs for the Statmodels explorer.

Description
-----------
One window, many independent workbenches. :class:`StatmodelsShell` is a tab bar over N
:class:`~nvitk.gui.panels.statmodels.window.StatmodelsWindow` pages, each holding its own
dataset query, frame recipe, model and plot — so a second hypothesis is a new tab rather than a
reload that costs the first one.

Switching is instant because nothing is rebuilt: every session is a live widget parked in a
``QStackedWidget``, so a tab change is a raise, not a reload. That is also why a page is built
eagerly rather than lazily — a tab that has to construct itself on first click is not a session
you can flick between while comparing two fits.

The tab bar's context menu is where sessions meet. Right-clicking a session *other* than the
current one offers to pull its dataframe across, in either of the two readings that are actually
useful — with its transformations still live, or as finished rows to build something new on. Both
run through :meth:`StatmodelsWindow.import_session`, which owns the semantics; this module only
asks the question and reports the answer.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QShortcut,
    QStyle,
    QTabBar,
    QTabWidget,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.gui.core.design import COLOR_CONTROL_HOVER, COLOR_MUTED, COLOR_TEXT
from nvitk.gui.core.geometry import fit_to_screen

from .constants import PIPELINE_KIND_QVTPY
from .theme import apply_dark_theme
from .window import StatmodelsWindow

log = Logger()


#: Edge of a tab-bar button, in pixels. A ``QPushButton`` honours an explicit stylesheet
#: geometry exactly — a ``QToolButton`` adds three pixels of frame to its size hint that no
#: ``setFixedSize`` takes back, leaving a button whose hint is bigger than the box it is drawn in.
_BUTTON_PX = 16


def _bar_button(glyph: str, tooltip: str) -> QPushButton:
    """One flat tab-bar button, styled to read on the dark chrome.

    Every geometry property is set explicitly rather than just the colours: a host stylesheet
    styles *all* its buttons — the explorer pads its own by 5×12 — and that is enough to leave a
    16 px button laid out and clickable but drawing blank, which is indistinguishable from
    missing. See ``nvitk.gui.viz.left_dock.install_expand_button``, which learned the same lesson.
    """
    button = QPushButton(glyph)
    button.setFlat(True)
    button.setAutoFillBackground(False)
    button.setToolTip(tooltip)
    button.setCursor(Qt.PointingHandCursor)
    button.setFixedSize(_BUTTON_PX, _BUTTON_PX)
    button.setStyleSheet(
        f"QPushButton {{ color: {COLOR_MUTED}; background: transparent; border: none;"
        f" padding: 0px; margin: 0px; font-size: 11px;"
        f" min-width: {_BUTTON_PX}px; max-width: {_BUTTON_PX}px;"
        f" min-height: {_BUTTON_PX}px; max-height: {_BUTTON_PX}px; }}"
        f"QPushButton:hover {{ color: {COLOR_TEXT};"
        f" background-color: {COLOR_CONTROL_HOVER}; border-radius: 3px; }}"
    )
    return button


class StatmodelsShell(QMainWindow):
    """A tabbed host for independent Statmodels sessions."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        initial_pipeline_kind: str = PIPELINE_KIND_QVTPY,
    ) -> None:
        """Open the shell on one clean session."""
        super().__init__(parent)
        self.setWindowTitle("nvitk Statmodels")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        fit_to_screen(self, 1700, 1000)
        apply_dark_theme(self)

        #: Pipeline kind new sessions open on. The launcher's dropdown writes here, so the choice
        #: made before opening the window still applies to the second tab and the tenth.
        self._pipeline_kind = str(initial_pipeline_kind or PIPELINE_KIND_QVTPY)
        #: Monotonic, not a count: renaming or closing tabs must not produce two "Session 3".
        self._counter = 0
        #: The session that currently owns the screen, by widget rather than by index — tabs are
        #: movable and closable, so an index does not keep pointing at the same session. Set
        #: before the tab widget exists because adding the first tab already fires
        #: ``currentChanged``.
        self._active: StatmodelsWindow | None = None

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(True)
        self._tabs.setTabsClosable(True)
        self._tabs.setElideMode(Qt.ElideRight)
        # Sessions accumulate, and a dozen elided tabs are still a dozen tabs wide.
        self._tabs.setUsesScrollButtons(True)
        self._tabs.tabCloseRequested.connect(self._on_close_tab)
        self._tabs.tabBarDoubleClicked.connect(self._on_rename_tab)
        self._tabs.currentChanged.connect(self._on_current_changed)

        bar = self._tabs.tabBar()
        bar.setContextMenuPolicy(Qt.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._on_tab_menu)

        add = _bar_button("+", "New session — a clean dataframe and model  (Ctrl+T)")
        add.clicked.connect(lambda *_: self.new_session())
        self._tabs.setCornerWidget(add, Qt.TopRightCorner)

        self.setCentralWidget(self._tabs)
        self._install_shortcuts()
        self.new_session()

    # ──────────────────────────────────────────────────────────────────────────
    # Sessions
    # ──────────────────────────────────────────────────────────────────────────
    def sessions(self) -> list[StatmodelsWindow]:
        """Every open session, in tab order."""
        pages = (self._tabs.widget(i) for i in range(self._tabs.count()))
        return [page for page in pages if isinstance(page, StatmodelsWindow)]

    def current_session(self) -> StatmodelsWindow | None:
        """The session on the visible tab, if there is one."""
        page = self._tabs.currentWidget()
        return page if isinstance(page, StatmodelsWindow) else None

    def new_session(
        self,
        *,
        title: str = "",
        pipeline_kind: str = "",
        select: bool = True,
    ) -> StatmodelsWindow:
        """
        Add a session with an empty dataframe and an unfitted model, and return it.

        Built with ``parent=None`` and only then handed to the tab widget: a child of this window
        with no layout to hold it would paint itself over the tab bar for as long as it took to
        get reparented.
        """
        kind = str(pipeline_kind or self._pipeline_kind)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            session = StatmodelsWindow(initial_pipeline_kind=kind, embedded=True)
        finally:
            QApplication.restoreOverrideCursor()

        self._counter += 1
        name = self._unique_title(title or f"Session {self._counter}")
        session.set_session_name(name)
        index = self._tabs.addTab(session, name)
        self._install_close_button(index, session)
        self._tabs.setTabToolTip(index, session.session_summary())
        if select:
            self._tabs.setCurrentIndex(index)
        return session

    def rename_session(self, session: StatmodelsWindow, name: str) -> str:
        """Retitle *session*'s tab, de-duplicating against the others, and return the name used.

        A no-op for a blank name, so a caller can offer
        :meth:`~nvitk.gui.panels.statmodels.window.StatmodelsWindow.suggested_title` without first
        checking whether the session has anything to suggest.
        """
        index = self._tabs.indexOf(session)
        wanted = str(name or "").strip()
        if index < 0 or not wanted or wanted == self._tabs.tabText(index):
            return self._tabs.tabText(index) if index >= 0 else ""
        unique = self._unique_title(wanted)
        self._tabs.setTabText(index, unique)
        session.set_session_name(unique)
        if index == self._tabs.currentIndex():
            self.setWindowTitle(f"nvitk Statmodels — {unique}")
        return unique

    def set_pipeline_kind(self, kind: str) -> None:
        """Point the current session at *kind*, and open later sessions on it too."""
        self._pipeline_kind = str(kind or PIPELINE_KIND_QVTPY)
        session = self.current_session()
        if session is not None:
            session.set_pipeline_kind(self._pipeline_kind)

    def show_maximized_floating(self) -> None:
        """Show, maximize, raise, and focus the shell."""
        self.show()
        self.showMaximized()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event: Any) -> None:
        """Give every session a chance to stop its workers before the pages are torn down.

        Qt destroys child widgets without delivering ``closeEvent`` to them, and a page's close
        handler is what joins its loader and mediation threads — threads that would otherwise
        outlive the widgets their signals are wired to.
        """
        for session in self.sessions():
            session.close()
        super().closeEvent(event)

    # ──────────────────────────────────────────────────────────────────────────
    # Tab bar
    # ──────────────────────────────────────────────────────────────────────────
    def _install_shortcuts(self) -> None:
        """Browser-style session keys, so a second model does not need the mouse."""
        bindings: list[tuple[str, Any]] = [
            ("Ctrl+T", lambda: self.new_session()),
            ("Ctrl+W", lambda: self._on_close_tab(self._tabs.currentIndex())),
            ("Ctrl+Tab", lambda: self._step(1)),
            ("Ctrl+Shift+Tab", lambda: self._step(-1)),
            ("Ctrl+PgDown", lambda: self._step(1)),
            ("Ctrl+PgUp", lambda: self._step(-1)),
        ]
        # Ctrl+1..9 by position, the way every tabbed application binds them.
        bindings += [
            (f"Ctrl+{n}", lambda index=n - 1: self._select(index)) for n in range(1, 10)
        ]
        # Parented to the shell, which is what keeps them alive past this call.
        for keys, slot in bindings:
            QShortcut(QKeySequence(keys), self).activated.connect(slot)

    def _step(self, delta: int) -> None:
        """Move *delta* tabs along, wrapping at both ends."""
        count = self._tabs.count()
        if count > 1:
            self._tabs.setCurrentIndex((self._tabs.currentIndex() + delta) % count)

    def _select(self, index: int) -> None:
        """Jump to a tab by position, ignoring positions that do not exist."""
        if 0 <= index < self._tabs.count():
            self._tabs.setCurrentIndex(index)

    def _install_close_button(self, index: int, session: StatmodelsWindow) -> None:
        """Replace Qt's stock close icon on this tab with one that reads on the dark chrome.

        The platform icon is a red or grey bitmap that belongs to no theme this application uses.
        """
        button = _bar_button("✕", "Close this session")
        # Resolved at click time, not bound now: tabs are movable, so the index this button was
        # installed at is not the index it will be clicked at.
        button.clicked.connect(lambda *_: self._on_close_tab(self._tabs.indexOf(session)))
        # Left on macOS, right everywhere else — the style knows, and putting it on the wrong
        # side leaves Qt's own button in place beside ours.
        side = QTabBar.ButtonPosition(
            QApplication.style().styleHint(QStyle.SH_TabBar_CloseButtonPosition)
        )
        self._tabs.tabBar().setTabButton(index, side, button)

    def _unique_title(self, wanted: str) -> str:
        """*wanted*, suffixed until no other tab already carries it."""
        taken = {self._tabs.tabText(i) for i in range(self._tabs.count())}
        if wanted not in taken:
            return wanted
        return next(f"{wanted} ({n})" for n in range(2, 1000) if f"{wanted} ({n})" not in taken)

    def _refresh_tooltips(self) -> None:
        """Re-read every session's summary onto its tab.

        Pulled on demand — on a tab change and before the context menu opens — rather than pushed
        from the sessions. A tab's tooltip is only ever read at those two moments, and a signal
        from each session for every frame recompute would be a great deal of wiring for it.
        """
        for index, session in enumerate(self.sessions()):
            self._tabs.setTabToolTip(index, session.session_summary())

    def _on_current_changed(self, index: int) -> None:
        """Hand the screen to the new session: its windows come back, the old one's go away."""
        previous, current = self._active, self.current_session()
        if previous is not None and previous is not current:
            # Only the outgoing session needs this. Any other session's windows were taken down
            # when *it* stopped being current, so two scans per switch is the whole cost however
            # many sessions are open.
            self._suspend(previous)
        if current is not None:
            current.restore_detached_windows()
        self._active = current

        self._refresh_tooltips()
        name = self._tabs.tabText(index) if index >= 0 else ""
        self.setWindowTitle(f"nvitk Statmodels — {name}" if name else "nvitk Statmodels")

    @staticmethod
    def _suspend(session: StatmodelsWindow) -> None:
        """Take *session*'s detached windows down, tolerating one already on its way out.

        Closing the current tab re-points the tab widget before the page is deleted, so the
        session arriving here can be one that no longer has a C++ object behind it.
        """
        try:
            session.suspend_detached_windows()
        except RuntimeError:
            pass

    def hideEvent(self, event: Any) -> None:
        """Take every session's detached windows off screen with the shell itself.

        Closing or hiding the shell otherwise leaves a floated dataframe panel behind on the
        desktop, with nothing left on screen to put it back.
        """
        for session in self.sessions():
            self._suspend(session)
        super().hideEvent(event)

    def showEvent(self, event: Any) -> None:
        """Bring the current session's detached windows back up with the shell."""
        super().showEvent(event)
        session = self.current_session()
        if session is not None:
            session.restore_detached_windows()

    def _on_rename_tab(self, index: int) -> None:
        """Rename a session from a double-click on its tab."""
        if index < 0:
            return
        session = self._tabs.widget(index)
        name, ok = QInputDialog.getText(
            self, "Rename session", "Session name:", text=self._tabs.tabText(index)
        )
        name = name.strip() if ok else ""
        if not name:
            return
        name = self._unique_title(name)
        self._tabs.setTabText(index, name)
        if isinstance(session, StatmodelsWindow):
            session.set_session_name(name)
        if index == self._tabs.currentIndex():
            self.setWindowTitle(f"nvitk Statmodels — {name}")

    def _on_close_tab(self, index: int) -> None:
        """Close a session, asking first if closing it would throw work away."""
        session = self._tabs.widget(index)
        if not isinstance(session, StatmodelsWindow):
            return
        name = self._tabs.tabText(index)
        if session.has_frame() or session.is_fitted():
            answer = QMessageBox.question(
                self,
                "Close session",
                f"Close “{name}”?\n\nIts dataframe and model are not saved.",
                QMessageBox.Close | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Close:
                return
        self._tabs.removeTab(index)
        # close() joins the background workers; deleteLater frees the figures and web views the
        # page was holding, which a removeTab on its own would leave alive and parentless.
        session.close()
        session.deleteLater()
        if self._tabs.count() == 0:
            # The shell without a session is a window with nothing in it and no way back.
            self.new_session()

    def _on_tab_menu(self, pos: Any) -> None:
        """Pop the session menu for whichever tab was right-clicked."""
        bar = self._tabs.tabBar()
        self._refresh_tooltips()
        menu = self._session_menu(bar.tabAt(pos))
        if menu is not None:
            menu.exec(bar.mapToGlobal(pos))

    def _session_menu(self, index: int) -> QMenu | None:
        """Build the session menu for the tab at *index*: import into the current one, duplicate,
        rename, close. Separate from :meth:`_on_tab_menu` so the menu can be built without a
        pointer, and inspected without blocking on ``exec``."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)

        if index < 0:
            menu.addAction("New session", lambda: self.new_session())
            return menu

        source = self._tabs.widget(index)
        if not isinstance(source, StatmodelsWindow):
            return None
        name = self._tabs.tabText(index)
        # A disabled action rather than addSection: the section label is drawn by the style, and
        # this application's QMenu stylesheet paints the separator without it — leaving a menu
        # whose entries say "import" without ever naming what from.
        header = menu.addAction(name)
        header.setEnabled(False)
        menu.addSeparator()

        current = self._tabs.currentIndex()
        if index != current and self.current_session() is not None:
            target = self._tabs.tabText(current)
            # Named by destination, so which way the frame travels is not something to work out
            # from which tab happened to be right-clicked.
            into = menu.addMenu(f"Import into “{target}”")
            into.setToolTipsVisible(True)
            recipe = into.addAction("Dataframe + transformations")
            recipe.setToolTip(
                f"Reproduce “{name}”'s table in “{target}”, carrying its measurements, "
                "combinations, derived columns, filters and reshape — all still editable there."
            )
            recipe.triggered.connect(
                lambda *_: self._import(index, StatmodelsWindow.IMPORT_RECIPE)
            )
            result = into.addAction("Dataframe as new raw data")
            result.setToolTip(
                f"Adopt “{name}”'s finished rows in “{target}” as raw data, without its recipe — "
                "for building a new set of transformations on top of its output."
            )
            result.triggered.connect(
                lambda *_: self._import(index, StatmodelsWindow.IMPORT_RESULT)
            )
            menu.addSeparator()

        duplicate = menu.addAction("Duplicate into a new session")
        duplicate.setToolTip(
            f"Open a copy of “{name}” — its frame, formula, engine and plot settings — to "
            "diverge from. The fit itself is not copied."
        )
        duplicate.triggered.connect(lambda *_: self._duplicate(index))

        menu.addSeparator()
        menu.addAction("Rename…", lambda: self._on_rename_tab(index))
        menu.addAction("Close session", lambda: self._on_close_tab(index))
        menu.addSeparator()
        menu.addAction("New session", lambda: self.new_session())
        return menu

    # ──────────────────────────────────────────────────────────────────────────
    # Transfers
    # ──────────────────────────────────────────────────────────────────────────
    def _import(self, index: int, mode: str) -> None:
        """Pull the session at *index* into the current one, confirming an overwrite first."""
        source = self._tabs.widget(index)
        target = self.current_session()
        if not isinstance(source, StatmodelsWindow) or target is None or source is target:
            return

        if target.has_frame() or target.is_fitted():
            answer = QMessageBox.question(
                self,
                "Replace this session's data",
                f"“{self._tabs.tabText(self._tabs.currentIndex())}” already has a dataframe.\n\n"
                f"Replace it with “{self._tabs.tabText(index)}”'s? Any fit here is dropped — the "
                "model settings and the plot are kept.",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return

        self._run_import(target, source, mode, failed="Import failed")

    def _duplicate(self, index: int) -> None:
        """Open a copy of the session at *index* in a new tab."""
        source = self._tabs.widget(index)
        if not isinstance(source, StatmodelsWindow):
            return
        clone = self.new_session(title=f"{self._tabs.tabText(index)} copy")
        copied = self._run_import(
            clone, source, StatmodelsWindow.IMPORT_CLONE, failed="Duplicate failed"
        )
        if not copied:
            # A half-copied session is not worth keeping: it would look like the original and
            # model something else.
            self._tabs.removeTab(self._tabs.indexOf(clone))
            clone.close()
            clone.deleteLater()

    def _run_import(
        self,
        target: StatmodelsWindow,
        source: StatmodelsWindow,
        mode: str,
        *,
        failed: str,
    ) -> bool:
        """Do the transfer under a wait cursor, reporting a failure rather than raising it."""
        error = ""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            target.import_session(source, mode=mode)
        except Exception as exc:
            log.debug("Session import (%s) failed", mode, exc_info=True)
            error = str(exc)
        finally:
            QApplication.restoreOverrideCursor()

        if error:
            QMessageBox.warning(self, failed, error)
            return False
        self._refresh_tooltips()
        return True


__all__ = ["StatmodelsShell"]
