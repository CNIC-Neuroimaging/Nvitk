"""Statmodels in the workbench's shape: a workflow rail, an Explore dock, frames from Analysis.

The Statmodels window grew as three tabbed docks on the left (Data, Model, Covariates), the
plot in the middle, Results on the right and the dataframe at the bottom, each closed and
re-opened from a View menu. This module gives it the layout of the rest of nvitk without
touching how any of those panels work:

* **Workflow rail**: a slim icon column at the window's left edge, like nvitk's activity bar.
  Model, Data and Covariates are its steps: clicking one opens the left column on it, and
  clicking the step already open folds the column away so the plot gets the room. Below them,
  Results, Dataframe and Explore toggle their docks, and Focus gives the plot the window.
  The rail is the only switch between the steps: the three docks are no longer tabbed, so the
  old Data / Covariates / Model tab bar is gone; one of them is shown at a time, in the same
  column (:func:`normalize_column` keeps it so after a layout restore, a reset or Focus).
* **Explore**: the Analysis module's :class:`~nvitk.gui.analysis.plot_builder.PlotBuilder` over
  the session's *working* frame (derived columns, filters, reshape and drops applied), so any
  column can be looked at (scatter, box, violin, histogram, heatmap, facets, a pandas query)
  before or beside a model, with no fit needed.
* **Frames from Analysis**: :func:`adopt_frame` makes any DataFrame a session's analysis
  frame, as a dropped spreadsheet is. The Analysis workbench uses it to send the table it shows
  (filtered) to a new Statmodels session.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import QDockWidget, QSizePolicy, QToolBar, QWidget

from nvitk.gui.core import design

#: Left-column steps of the rail: (dock key, icon, tooltip).
STEPS: tuple[tuple[str, str, str], ...] = (
    ("model", "chart", "Model: outcome, formula, family, random effects, fit"),
    ("data", "database", "Data: pipeline kind, measurements, covariates' sources, reload"),
    ("covariates", "sliders", "Covariates: clinical and imaging variables to add"),
)

#: Docks the rail toggles: (dock key, icon, tooltip).
TOGGLES: tuple[tuple[str, str, str], ...] = (
    ("results", "table", "Results: the fitted model's report"),
    ("frame", "grid", "Analysis dataframe"),
    ("explore", "plot_scatter", "Explore: plot any columns of the working frame"),
)

_LEFT = tuple(key for key, _i, _t in STEPS)


def install_relayout(window: Any) -> None:
    """Add the Explore dock and the workflow rail to a :class:`StatmodelsWindow`.

    Called while the window is being built, after its docks exist and before its View menu and
    default layout snapshot, so Explore is listed there and part of the default arrangement.
    """
    _install_explore(window)
    _install_rail(window)
    _install_single_column(window)


# Explore
def _install_explore(window: Any) -> None:
    from nvitk.gui.analysis.plot_builder import PlotBuilder

    builder = PlotBuilder()
    builder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    dock = QDockWidget("Explore", window)
    dock.setObjectName("statmodels_dock_explore")
    dock.setWidget(builder)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
    window.addDockWidget(Qt.RightDockWidgetArea, dock)
    results = window._docks.get("results")
    if results is not None:
        # Under Results, not tabbed with it: a tab bar made while the window is being built
        # outlived the layout restore of a session and was left painted over the plot.
        window.splitDockWidget(results, dock, Qt.Vertical)
    dock.hide()  # opened from the rail (or View); the plot keeps the room until then
    window._docks["explore"] = dock
    window._explore_builder = builder
    window._explore_dirty = True

    def on_visible(visible: bool) -> None:
        if visible and getattr(window, "_explore_dirty", False):
            sync_explore(window)

    dock.visibilityChanged.connect(on_visible)


def sync_explore(window: Any) -> None:
    """Show the session's working frame in Explore (now if it is on screen, else when it is)."""
    builder = getattr(window, "_explore_builder", None)
    dock = getattr(window, "_docks", {}).get("explore")
    if builder is None or dock is None:
        return
    if not dock.isVisible() or dock.visibleRegion().isEmpty():
        window._explore_dirty = True
        return
    frame = getattr(window, "_working_df", None)
    builder.set_frame(frame if isinstance(frame, pd.DataFrame) else pd.DataFrame())
    window._explore_dirty = False


# Rail
def _install_rail(window: Any) -> None:
    from nvitk.gui.core.icons import icon

    bar = QToolBar("Workflow", window)
    bar.setObjectName("statmodels_workflow_rail")
    bar.setOrientation(Qt.Vertical)
    bar.setMovable(False)
    bar.setFloatable(False)
    bar.setIconSize(QSize(18, 18))
    bar.setToolButtonStyle(Qt.ToolButtonIconOnly)
    window.addToolBar(Qt.LeftToolBarArea, bar)

    actions: dict[str, Any] = {}
    for key, name, tip in STEPS:
        if key not in window._docks:
            continue
        action = bar.addAction(tip.split(":")[0])
        action.setToolTip(tip)
        action.setCheckable(True)
        action.setData(name)
        action.triggered.connect(lambda _c=False, k=key: show_step(window, k))
        actions[key] = action
    bar.addSeparator()
    for key, name, tip in TOGGLES:
        dock = window._docks.get(key)
        if dock is None:
            continue
        action = bar.addAction(tip.split(":")[0])
        action.setToolTip(tip)
        action.setCheckable(True)
        action.setData(name)
        action.toggled.connect(lambda on, d=dock: (d.show(), d.raise_()) if on else d.hide())
        dock.visibilityChanged.connect(lambda _v, a=action, d=dock: _sync_toggle(a, d))
        actions[key] = action
    spacer = QWidget(bar)
    spacer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    bar.addWidget(spacer)
    focus_button = getattr(window, "_btn_focus", None)
    if focus_button is not None:
        focus = bar.addAction("Focus plot")
        focus.setToolTip(focus_button.toolTip())
        focus.setCheckable(True)
        focus.setData("maximize")
        focus.toggled.connect(focus_button.setChecked)
        focus_button.toggled.connect(lambda on: (focus.blockSignals(True), focus.setChecked(on), focus.blockSignals(False)))
        actions["focus"] = focus

    window._rail = bar
    window._rail_actions = actions
    window._rail_step = "data"

    for key in _LEFT:
        dock = window._docks.get(key)
        if dock is not None:
            dock.visibilityChanged.connect(lambda _v: _sync_steps(window))

    def restyle() -> None:
        for action in actions.values():
            action.setIcon(icon(str(action.data()), design.COLOR_TEXT, on_color=design.COLOR_ON_ACCENT))
        bar.setStyleSheet(
            f"QToolBar#statmodels_workflow_rail {{ background: {design.COLOR_SURFACE}; border: none;"
            f" border-right: 1px solid {design.COLOR_BORDER}; spacing: 2px; padding: 3px 2px; }}"
            "QToolBar#statmodels_workflow_rail QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 4px; }"
            f"QToolBar#statmodels_workflow_rail QToolButton:hover {{ background: {design.COLOR_CONTROL_HOVER}; }}"
            "QToolBar#statmodels_workflow_rail QToolButton:checked {"
            f" background: {design.COLOR_ACCENT_DEEP}; border-color: {design.COLOR_ACCENT}; }}"
        )

    restyle()
    design.add_theme_listener(restyle)
    _sync_steps(window)


def _sync_toggle(action: Any, dock: Any) -> None:
    on = not dock.isHidden()
    if action.isChecked() != on:
        action.blockSignals(True)
        action.setChecked(on)
        action.blockSignals(False)


def _raised(dock: Any) -> bool:
    """True when *dock* is open and, if tabbed, the tab in front."""
    return dock.isVisible() and not dock.visibleRegion().isEmpty()


def _sync_steps(window: Any) -> None:
    actions = getattr(window, "_rail_actions", {})
    left = [window._docks[k] for k in _LEFT if k in window._docks]
    column_open = any(not d.isHidden() for d in left)
    front = next((k for k in _LEFT if k in window._docks and _raised(window._docks[k])), None)
    if front is not None:
        window._rail_step = front
    for key in _LEFT:
        action = actions.get(key)
        if action is None:
            continue
        action.blockSignals(True)
        action.setChecked(column_open and key == window._rail_step)
        action.blockSignals(False)


def show_step(window: Any, key: str) -> None:
    """Open the left column on step *key*; fold it when *key* is the step already open."""
    docks = {k: window._docks[k] for k in _LEFT if k in window._docks}
    dock = docks.get(key)
    if dock is None:
        return
    if window._rail_step == key and not dock.isHidden():
        dock.hide()
    else:
        window._rail_step = key
        width = max((d.width() for d in docks.values() if not d.isHidden() and not d.isFloating()), default=0)
        for k, d in docks.items():
            if k != key and not d.isFloating():
                d.hide()
        dock.show()
        dock.raise_()
        if width > 0 and not dock.isFloating():
            # The step takes the column's width, not the one it had when it was last shown.
            window.resizeDocks([dock], [width], Qt.Horizontal)
    _sync_steps(window)


# One step at a time
def _untabify(window: Any) -> None:
    """Take the step docks out of any tab group (a restored layout from before had them tabbed)."""
    docks = [window._docks[k] for k in _LEFT if k in window._docks]
    if not any(set(window.tabifiedDockWidgets(d)) & set(docks) for d in docks if not d.isFloating()):
        return
    shown = {id(d): not d.isHidden() for d in docks}
    for d in docks:
        if d.isFloating():
            continue
        window.removeDockWidget(d)
        window.addDockWidget(Qt.LeftDockWidgetArea, d, Qt.Vertical)
        d.setVisible(shown[id(d)])


def normalize_column(window: Any) -> None:
    """At most one step dock showing in the left column, and none of them tabbed."""
    if getattr(window, "_column_normalizing", False):
        return
    window._column_normalizing = True
    try:
        _untabify(window)
        docks = {k: window._docks[k] for k in _LEFT if k in window._docks}
        shown = [k for k, d in docks.items() if not d.isHidden() and not d.isFloating()]
        if len(shown) > 1:
            keep = window._rail_step if window._rail_step in shown else shown[0]
            for k in shown:
                if k != keep:
                    docks[k].hide()
            window._rail_step = keep
    finally:
        window._column_normalizing = False
    _hide_stale_tab_bars(window)
    _sync_steps(window)


def _hide_stale_tab_bars(window: Any) -> None:
    """Hide the tab bar Qt keeps after the step docks were untabbed.

    QMainWindow does not delete a tab group's bar when the group goes; it parks it, still
    "visible", and a later relayout can paint it over the plot.
    """
    from qtpy.QtWidgets import QTabBar

    titles = {window._docks[k].windowTitle() for k in _LEFT if k in window._docks}
    docks = [window._docks[k] for k in _LEFT if k in window._docks]
    if any(set(window.tabifiedDockWidgets(d)) & set(docks) for d in docks):
        return
    for bar in window.findChildren(QTabBar):
        if bar.parent() is window and bar.count() and {bar.tabText(i) for i in range(bar.count())} <= titles:
            bar.hide()


def _install_single_column(window: Any) -> None:
    """Untabify the steps now, and after anything that may stack or tab them again."""
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QMainWindow

    docks = [window._docks[k] for k in _LEFT if k in window._docks]
    for d in docks:
        if not d.isFloating():
            window.removeDockWidget(d)
            window.addDockWidget(Qt.LeftDockWidgetArea, d, Qt.Vertical)
    window._rail_step = "data" if "data" in window._docks else (_LEFT[0] if docks else "")
    for k in _LEFT:
        dock = window._docks.get(k)
        if dock is not None:
            dock.setVisible(k == window._rail_step)
    _hide_stale_tab_bars(window)

    def restore_state(state: Any, version: int = 0) -> bool:
        done = QMainWindow.restoreState(window, state, version)
        normalize_column(window)
        return done

    # Python callers (session restore, Reset panel layout, Focus off) go through the instance.
    window.restoreState = restore_state
    reset = getattr(window, "_reset_dock_layout", None)
    if reset is not None:
        def reset_layout() -> None:
            reset()
            normalize_column(window)

        window._reset_dock_layout = reset_layout
    # Anything else that shows every dock (a View menu toggle, Focus off without a snapshot):
    # tidied once the event that showed them is over.
    pending = {"on": False}

    def later() -> None:
        if pending["on"]:
            return
        pending["on"] = True

        def run() -> None:
            pending["on"] = False
            normalize_column(window)

        QTimer.singleShot(0, run)

    for d in docks:
        d.visibilityChanged.connect(lambda _v: later())


def column_open(window: Any) -> bool:
    """Whether the left column (Data / Model / Covariates) is showing."""
    return any(not window._docks[k].isHidden() for k in _LEFT if k in window._docks)


# Frames from elsewhere
def adopt_frame(window: Any, frame: pd.DataFrame, *, label: str = "table") -> None:
    """Make *frame* the session's analysis frame, as a dropped spreadsheet would be.

    Derived columns and filters already defined are kept and re-applied; Reload data goes back to
    the dataset query.
    """
    from nvitk.stats.frame_ops import ensure_unique_columns

    if frame is None or frame.empty:
        raise ValueError("The table has no rows.")
    frame = ensure_unique_columns(frame.reset_index(drop=True), context=label)
    window._analysis_df = frame
    window._prebuilt_frame = False
    window._load_meta = {
        "source": "analysis-workbench",
        "path": "",
        "sheet": str(label),
        "n_rows": int(len(frame)),
        "covariates": [],
        "warnings": [],
    }
    window._measurements.set_diagnostics(window._load_meta)
    window._btn_reload.setText("Reload data")
    window._btn_reload.setToolTip("Discard the table sent from Analysis and re-run the dataset query.")
    window._recompute_frame(announce=False)
    window._status.setText(f"Analysis frame: {label} ({len(frame)} rows × {frame.shape[1]} columns), from the Analysis module.")


__all__ = ["STEPS", "TOGGLES", "adopt_frame", "column_open", "install_relayout", "normalize_column", "show_step", "sync_explore"]
