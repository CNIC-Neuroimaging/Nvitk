"""
Session tabs for the Statmodels explorer.

Covers the invariants that are not obvious from reading the code: that a session's dataframe and
recipe travel to another tab intact, that the *result* import spends the recipe instead of
replaying it, that two sessions never share state, and that the tab bar's own controls work after
a tab has been moved.

Runs offscreen. Every shell is closed at the end of its test — an offscreen Napari-adjacent Qt
suite segfaults at interpreter shutdown otherwise, after having passed.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy")
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication, QMessageBox, QTabBar, QWidget  # noqa: E402

from nvitk.stats.frame_ops import DerivedColumn, FilterRule  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the module; Qt allows no second one."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def shell(qapp):
    """A shell on one clean session, closed afterwards so its workers are joined."""
    pytest.importorskip("nvitk.gui.panels.statmodels.sessions")
    from nvitk.gui.panels.statmodels.sessions import StatmodelsShell

    window = StatmodelsShell()
    yield window
    window.close()


LONG_FRAME = pd.DataFrame(
    {
        "subject_uid": [f"s{i // 3:02d}" for i in range(12)],
        "territory": ["LICA", "RICA", "BASI"] * 4,
        "flow_mean": [10.0, 12.0, 5.0, 11.0, 13.0, 6.0, 30.0, 9.0, 4.0, 10.5, 12.5, 5.5],
        "age": [50] * 3 + [61] * 3 + [72] * 3 + [45] * 3,
    }
)


def _loaded(session, frame=LONG_FRAME):
    """Give *session* a frame plus a derived column and a filter, and return it."""
    session._analysis_df = frame.copy()
    session._derived = [
        DerivedColumn(name="log_flow", kind="transform", source="flow_mean", transform="log")
    ]
    session._chips.set_rules([FilterRule(column="flow_mean", kind="range", low=0.0, high=20.0)], [])
    session._column_types = {"territory": "Factor"}
    session._recompute_frame(announce=False)
    return session


def test_shell_opens_on_one_embedded_session(shell):
    """The window starts usable, and its page is a tab rather than a window of its own."""
    assert shell._tabs.count() == 1
    session = shell.current_session()
    assert session is not None
    assert session._embedded is True
    assert not session.isWindow()


def test_new_session_starts_clean(shell):
    """A new tab is a workbench, not a view of the last one."""
    _loaded(shell.current_session())
    fresh = shell.new_session()
    assert shell.current_session() is fresh
    assert fresh.working_frame() is None
    assert not fresh.has_frame()
    assert fresh._derived == [] and fresh._chips.rules() == []


def test_recipe_import_reproduces_the_frame_and_keeps_it_editable(shell):
    """The whole recipe travels, and the destination's copy is its own."""
    from nvitk.gui.panels.statmodels.window import StatmodelsWindow

    source = _loaded(shell.current_session())
    target = shell.new_session()
    target.import_session(source, mode=StatmodelsWindow.IMPORT_RECIPE)

    pd.testing.assert_frame_equal(target.working_frame(), source.working_frame())
    assert [d.name for d in target._derived] == ["log_flow"]
    assert [r.column for r in target._chips.rules()] == ["flow_mean"]
    assert target._column_types == {"territory": "Factor"}
    # The measurement picks came across with the rows, so the frame is not stale.
    assert target._btn_reload.text() == "Reload data"

    target._derived = []
    target._recompute_frame(announce=False)
    assert "log_flow" not in target.working_frame().columns
    assert "log_flow" in source.working_frame().columns, "sessions share state"


def test_result_import_adopts_the_table_as_raw_data(shell):
    """A finished table arrives without its recipe, and stays derivable."""
    from nvitk.gui.panels.statmodels.window import StatmodelsWindow

    source = _loaded(shell.current_session())
    target = shell.new_session()
    target.import_session(source, mode=StatmodelsWindow.IMPORT_RESULT)

    pd.testing.assert_frame_equal(target.working_frame(), source.working_frame())
    assert target._derived == [] and target._chips.rules() == [] and target._combinations == []
    assert not target._prebuilt_frame

    target._derived = [
        DerivedColumn(name="age_z", kind="transform", source="age", transform="zscore")
    ]
    target._recompute_frame(announce=False)
    assert "age_z" in target.working_frame().columns


def test_result_import_spends_the_reshape_rather_than_repeating_it(shell):
    """The wide pivot travels as a flag with the recipe, but not with its own output."""
    from nvitk.gui.panels.statmodels.window import StatmodelsWindow

    source = shell.current_session()
    source._analysis_df = LONG_FRAME.copy()
    source._wide_mode = True
    source._recompute_frame(announce=False)
    assert len(source.working_frame()) == 4, "one row per subject"

    replayed = shell.new_session()
    replayed.import_session(source, mode=StatmodelsWindow.IMPORT_RECIPE)
    assert replayed._wide_mode is True
    pd.testing.assert_frame_equal(replayed.working_frame(), source.working_frame())

    spent = shell.new_session()
    spent.import_session(source, mode=StatmodelsWindow.IMPORT_RESULT)
    assert spent._wide_mode is False
    pd.testing.assert_frame_equal(spent.working_frame(), source.working_frame())


def test_import_never_carries_the_fit(shell):
    """A result belongs to the frame it was fitted to."""
    from nvitk.gui.panels.statmodels.window import StatmodelsWindow

    source = _loaded(shell.current_session())
    source._last_result = object()
    assert source.is_fitted()

    target = shell.new_session()
    target.import_session(source, mode=StatmodelsWindow.IMPORT_RECIPE)
    assert not target.is_fitted()


def test_duplicate_copies_the_model_too(shell):
    """A duplicate is the whole session, frame recipe and formula alike."""
    source = _loaded(shell.current_session())
    source._formula.setPlainText("flow_mean ~ age")
    shell._duplicate(shell._tabs.indexOf(source))

    clone = shell.current_session()
    assert clone is not source
    assert clone._formula.toPlainText().strip() == "flow_mean ~ age"
    pd.testing.assert_frame_equal(clone.working_frame(), source.working_frame())


@pytest.mark.parametrize("mode", ["recipe", "result"])
def test_import_refuses_what_it_cannot_do(shell, mode):
    """Self-import and empty sources raise rather than half-succeeding."""
    session = _loaded(shell.current_session())
    with pytest.raises(ValueError):
        session.import_session(session, mode=mode)
    with pytest.raises(ValueError):
        session.import_session(shell.new_session(), mode=mode)


def test_tab_menu_names_its_destination(shell):
    """The import submenu is named after where the frame is going, not where it came from."""
    source = _loaded(shell.current_session())
    shell.rename_session(source, "hemodynamics")
    current = shell.new_session(title="perfusion")

    menu = shell._session_menu(shell._tabs.indexOf(source))
    labels = [action.text() for action in menu.actions() if action.text()]
    assert labels[0] == "hemodynamics", "the menu must name the session it acts on"

    submenu = next(a.menu() for a in menu.actions() if a.menu() is not None)
    assert submenu.title() == "Import into “perfusion”"
    assert [a.text() for a in submenu.actions()] == [
        "Dataframe + transformations",
        "Dataframe as new raw data",
    ]

    own = shell._session_menu(shell._tabs.indexOf(current))
    assert not any(a.menu() for a in own.actions()), "no import offer on the tab you are on"


def test_close_button_follows_its_own_tab(shell, monkeypatch):
    """Tabs are movable, so a close button must not close by remembered position."""
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Close))
    for name in ("second", "third"):
        shell.rename_session(shell.new_session(), name)
    shell.show()

    bar = shell._tabs.tabBar()
    side = next(s for s in (QTabBar.RightSide, QTabBar.LeftSide) if bar.tabButton(0, s) is not None)
    bar.moveTab(0, 2)
    moved = shell._tabs.tabText(2)
    bar.tabButton(2, side).click()
    assert moved not in [shell._tabs.tabText(i) for i in range(shell._tabs.count())]


def test_close_buttons_draw_their_glyph(shell):
    """A themed button squeezed blank by a host stylesheet is indistinguishable from a missing
    one, so this checks pixels rather than structure."""
    shell.show()
    bar = shell._tabs.tabBar()
    side = next(s for s in (QTabBar.RightSide, QTabBar.LeftSide) if bar.tabButton(0, s) is not None)
    button = bar.tabButton(0, side)

    assert button.sizeHint().width() <= button.width()
    assert button.sizeHint().height() <= button.height()
    image = button.grab().toImage()
    background = image.pixel(0, 0)
    ink = [
        (x, y)
        for x in range(image.width())
        for y in range(image.height())
        if image.pixel(x, y) != background
    ]
    assert ink, "the glyph never made it onto the button"
    xs, ys = [x for x, _ in ink], [y for _, y in ink]
    assert 0 < min(xs) and max(xs) < image.width() - 1, "glyph clipped horizontally"
    assert 0 < min(ys) and max(ys) < image.height() - 1, "glyph clipped vertically"


def test_shell_always_keeps_a_session(shell, monkeypatch):
    """Closing the last tab leaves a window with something in it."""
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Close))
    _loaded(shell.current_session())
    shell.new_session()
    while shell._tabs.count() > 1:
        shell._on_close_tab(0)
    shell._on_close_tab(0)
    assert shell._tabs.count() == 1
    assert not shell.current_session().has_frame()


# ──────────────────────────────────────────────────────────────────────────────
# Detached windows
# ──────────────────────────────────────────────────────────────────────────────
def _detached(session, title="popup"):
    """A separate window belonging to *session*, standing in for a floated dock or a dialog."""
    window = QWidget(session, Qt.Window)
    window.setWindowTitle(title)
    window.show()
    return window


def test_floated_dock_follows_its_session_off_screen(shell, qapp):
    """A floated dock is a top-level window, so Qt leaves it up when its page is switched away."""
    session = shell.current_session()
    shell.show()
    dock = session._docks["frame"]
    dock._nvitk_float_toggle()
    qapp.processEvents()
    assert dock.isFloating() and dock.isVisible()
    geometry = dock.geometry()

    shell.new_session()
    qapp.processEvents()
    assert not dock.isVisible(), "it stayed up over the new session"

    shell._tabs.setCurrentIndex(0)
    qapp.processEvents()
    assert dock.isVisible(), "it did not come back"
    assert dock.isFloating(), "it came back docked"
    assert dock.geometry() == geometry, "it came back somewhere else"


def test_any_detached_window_is_hidden_not_just_docks(shell, qapp):
    """The rule is about windows, not about the two kinds that exist today."""
    session = shell.current_session()
    shell.show()
    popup = _detached(session)
    qapp.processEvents()

    shell.new_session()
    qapp.processEvents()
    assert not popup.isVisible()

    shell._tabs.setCurrentIndex(0)
    qapp.processEvents()
    assert popup.isVisible()


def test_a_window_closed_while_its_session_was_away_stays_closed(shell, qapp):
    """Coming back restores what was taken down, not what the user shut in the meantime."""
    session = shell.current_session()
    shell.show()
    kept, closed = _detached(session, "kept"), _detached(session, "closed")
    qapp.processEvents()

    shell.new_session()
    qapp.processEvents()
    closed.close()
    qapp.processEvents()

    shell._tabs.setCurrentIndex(0)
    qapp.processEvents()
    assert kept.isVisible()
    assert not closed.isVisible()


def test_a_window_closed_while_its_session_was_live_is_not_resurrected(shell, qapp):
    """Hiding a panel is a decision; a round trip through another tab must not undo it."""
    session = shell.current_session()
    shell.show()
    popup = _detached(session)
    qapp.processEvents()
    popup.hide()

    shell.new_session()
    shell._tabs.setCurrentIndex(0)
    qapp.processEvents()
    assert not popup.isVisible()


def test_shell_takes_its_windows_off_screen_with_it(shell, qapp):
    """Hiding the shell must not leave a floated panel stranded with no way back."""
    session = shell.current_session()
    shell.show()
    popup = _detached(session)
    qapp.processEvents()

    shell.hide()
    qapp.processEvents()
    assert not popup.isVisible()

    shell.show()
    qapp.processEvents()
    assert popup.isVisible()


def test_suspending_twice_does_not_forget_what_to_restore(shell, qapp):
    """The shell suspends every session when hidden, including ones already off screen."""
    session = shell.current_session()
    shell.show()
    popup = _detached(session)
    qapp.processEvents()

    shell.new_session()          # session is suspended here
    qapp.processEvents()
    shell.hide()                 # ...and asked to suspend again, finding nothing visible
    qapp.processEvents()
    shell.show()
    qapp.processEvents()
    shell._tabs.setCurrentIndex(0)
    qapp.processEvents()
    assert popup.isVisible(), "the second suspend cleared the first one's list"


def test_shell_shrinks_to_a_small_screen(shell):
    """Several pages of docks must not put a floor under the window."""
    for _ in range(3):
        shell.new_session()
    shell.resize(900, 600)
    assert shell.minimumSizeHint().width() <= 900
    assert shell.minimumSizeHint().height() <= 600
