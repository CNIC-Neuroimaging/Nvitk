"""
Level ordering and the legend toggle on the distribution plots.

The default order is a natural sort, which is right for ``g0 … g3`` and wrong for anything whose
meaning is not alphabetical — a severity scale, a vessel sequence following the circulation, a
control group that belongs first.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from nvitk.stats.group_counts import ordered_levels

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from nvitk.stats.distribution_plots import (  # noqa: E402
    column_panels_static,
    column_plot_static,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TERRITORIES = ["BASI", "LICA", "RICA"]
#: Deliberately not alphabetical and not the frame's own order.
WANTED = ["RICA", "BASI", "LICA"]
MEANS = {"BASI": 6.0, "LICA": 10.0, "RICA": 12.0}


@pytest.fixture(scope="module")
def frame():
    """Three territories with clearly separated medians, so "by median" has a right answer."""
    rng = np.random.default_rng(6)
    n = 900
    out = pd.DataFrame(
        {
            "territory": rng.choice(TERRITORIES, n),
            "sex": rng.choice(["M", "F"], n),
            "grp": rng.choice(["g0", "g1", "g2", "g10"], n),
        }
    )
    out["flow_mean"] = [MEANS[t] for t in out["territory"]] + rng.normal(0, 1.0, n)
    return out


def _ticks(figure):
    """The x tick labels, without the count line underneath."""
    return [t.get_text().split("\n")[0] for t in figure.axes[0].get_xticklabels()]


# ──────────────────────────────────────────────────────────────────────────────
# The ordering itself
# ──────────────────────────────────────────────────────────────────────────────
def test_order_arranges_what_it_names():
    assert ordered_levels(["a", "b", "c"], ["c", "a", "b"]) == ["c", "a", "b"]


def test_no_order_is_the_identity():
    assert ordered_levels(["a", "b"], None) == ["a", "b"]
    assert ordered_levels(["a", "b"], []) == ["a", "b"]


def test_a_level_the_order_does_not_name_is_kept_after_it():
    """A saved order outlives the frame it was chosen on — a reload can add a level, and dropping
    it silently would hide data."""
    assert ordered_levels(["a", "b", "c"], ["c"]) == ["c", "a", "b"]


def test_a_level_the_frame_no_longer_has_is_ignored():
    """...and a filter can remove one."""
    assert ordered_levels(["a", "b"], ["z", "b", "a"]) == ["b", "a"]


# ──────────────────────────────────────────────────────────────────────────────
# Every plot path honours it
# ──────────────────────────────────────────────────────────────────────────────
def test_static_split_honours_the_order(frame):
    figure = column_plot_static(frame, "flow_mean", group="territory", level_order=WANTED)
    assert _ticks(figure) == WANTED


def test_static_panels_honour_the_order(frame):
    figure = column_panels_static(frame, "flow_mean", facet_by="territory", panel_order=WANTED)
    titles = [ax.get_title().split(" ")[0] for ax in figure.axes if ax.get_visible() and ax.get_title()]
    assert titles == WANTED


def test_interactive_split_honours_the_order(frame):
    pytest.importorskip("plotly")
    from nvitk.stats.interactive import column_plot

    figure = column_plot(frame, "flow_mean", group="territory", level_order=WANTED)
    assert [t.name.split("<br>")[0] for t in figure.data if t.name] == WANTED


def test_interactive_panels_honour_the_order(frame):
    pytest.importorskip("plotly")
    from nvitk.stats.interactive import column_panel_figure

    figure = column_panel_figure(frame, "flow_mean", facet_by="territory", panel_order=WANTED)
    titles = [a.text.split(" ")[0] for a in figure.layout.annotations if a.text]
    assert titles == WANTED


def test_both_backends_order_alike(frame):
    """A figure that reorders on one backend and not the other is two different figures."""
    pytest.importorskip("plotly")
    from nvitk.stats.interactive import column_plot

    static = column_plot_static(frame, "flow_mean", group="territory", level_order=WANTED)
    interactive = column_plot(frame, "flow_mean", group="territory", level_order=WANTED)
    assert _ticks(static) == [t.name.split("<br>")[0] for t in interactive.data if t.name]


# ──────────────────────────────────────────────────────────────────────────────
# Through the dialog
# ──────────────────────────────────────────────────────────────────────────────
pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(qapp, frame):
    """The distribution dialog, split by territory, closed afterwards."""
    from nvitk.gui.panels.statmodels.column_plot_dialog import ColumnPlotDialog

    widget = ColumnPlotDialog(None, frame=frame, column="flow_mean")
    widget.show()
    names = [widget._split.itemText(i) for i in range(widget._split.count())]
    widget._split.setCurrentIndex(names.index("territory"))
    qapp.processEvents()
    yield widget
    widget.close()


def test_the_dialog_applies_a_saved_order(dialog, qapp):
    dialog._orders["territory"] = WANTED
    dialog._redraw()
    qapp.processEvents()
    assert _ticks(dialog._static_figure) == WANTED


def test_the_order_survives_switching_columns_and_back(dialog, qapp):
    """Each column keeps its own order, so a detour through another split does not lose it."""
    dialog._orders["territory"] = WANTED
    dialog._redraw()
    names = [dialog._split.itemText(i) for i in range(dialog._split.count())]
    dialog._split.setCurrentIndex(names.index("sex"))
    qapp.processEvents()
    dialog._split.setCurrentIndex(names.index("territory"))
    qapp.processEvents()
    assert _ticks(dialog._static_figure) == WANTED


def test_the_button_says_whether_an_order_is_in_force(dialog, qapp):
    assert dialog._btn_order.isEnabled()
    assert dialog._btn_order.text() == "Order…"
    dialog._orders["territory"] = WANTED
    dialog._redraw()
    qapp.processEvents()
    assert dialog._btn_order.text() == "Order ✓"


def test_the_legend_toggle_hides_and_restores(dialog, qapp):
    """The overlaid kinds are the ones with a legend; violin names its levels on the axis."""
    index = [dialog._kind.itemData(i) for i in range(dialog._kind.count())].index("density")
    dialog._kind.setCurrentIndex(index)
    qapp.processEvents()

    def legend():
        found = [a.get_legend() for a in dialog._static_figure.axes if a.get_legend() is not None]
        assert found, "this kind should have a legend to toggle"
        return found[0]

    assert legend().get_visible()
    dialog._show_legend.setChecked(False)
    qapp.processEvents()
    assert not legend().get_visible()
    dialog._show_legend.setChecked(True)
    qapp.processEvents()
    assert legend().get_visible()


def test_the_legend_toggle_survives_a_redraw(dialog, qapp):
    """Every control redraws, so a toggle re-applied only on click would flick back on."""
    index = [dialog._kind.itemData(i) for i in range(dialog._kind.count())].index("density")
    dialog._kind.setCurrentIndex(index)
    dialog._show_legend.setChecked(False)
    qapp.processEvents()
    dialog._redraw()
    qapp.processEvents()
    found = [a.get_legend() for a in dialog._static_figure.axes if a.get_legend() is not None]
    assert found and not found[0].get_visible()


def test_the_controls_row_wraps_rather_than_clipping(qapp, frame):
    """A QHBoxLayout's minimum width is the sum of its children, which Qt enforces as the
    dialog's — so a row that outgrew the window used to clip its last controls."""
    from nvitk.gui.panels.statmodels.column_plot_dialog import ColumnPlotDialog

    for width in (1000, 700, 560):
        widget = ColumnPlotDialog(None, frame=frame, column="flow_mean")
        widget.resize(width, 700)
        widget.show()
        qapp.processEvents()
        qapp.processEvents()
        right = widget._btn_export.mapTo(widget, widget._btn_export.rect().topRight()).x()
        assert right <= widget.width(), f"clipped at {width}px"
        widget.close()


# ──────────────────────────────────────────────────────────────────────────────
# The reorder editor
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def editor(qapp, frame):
    """The reorder editor over the three territories."""
    from nvitk.gui.panels.statmodels.column_plot_dialog import LevelOrderDialog

    widget = LevelOrderDialog(
        None, column="territory", levels=TERRITORIES,
        frame=frame, value_column="flow_mean",
    )
    yield widget
    widget.close()


def test_preset_by_median(editor):
    """BASI 6, LICA 10, RICA 12 — ascending."""
    editor._apply_preset("median")
    assert editor.levels() == ["BASI", "LICA", "RICA"]


def test_preset_by_count(qapp, frame):
    """Most observations first."""
    from nvitk.gui.panels.statmodels.column_plot_dialog import LevelOrderDialog

    counts = frame["territory"].value_counts()
    widget = LevelOrderDialog(
        None, column="territory", levels=TERRITORIES, frame=frame, value_column="flow_mean"
    )
    widget._apply_preset("count")
    assert widget.levels() == list(counts.index)
    widget.close()


def test_preset_alpha_and_reverse(editor):
    editor._apply_preset("alpha")
    assert editor.levels() == sorted(TERRITORIES)
    editor._apply_preset("reverse")
    assert editor.levels() == sorted(TERRITORIES, reverse=True)


def test_preset_natural_sorts_g10_after_g2(qapp, frame):
    """The reason the default is a natural sort and not a string one."""
    from nvitk.gui.panels.statmodels.column_plot_dialog import LevelOrderDialog

    widget = LevelOrderDialog(
        None, column="grp", levels=["g10", "g2", "g0", "g1"],
        frame=frame, value_column="flow_mean",
    )
    widget._apply_preset("natural")
    assert widget.levels() == ["g0", "g1", "g2", "g10"]
    widget.close()


def test_the_arrows_move_one_place_and_keep_the_selection(editor):
    editor._list.setCurrentRow(2)
    editor._move(-1)
    assert editor.levels() == ["BASI", "RICA", "LICA"]
    assert editor._list.currentRow() == 1


def test_the_arrows_stop_at_the_ends(editor):
    before = editor.levels()
    editor._list.setCurrentRow(0)
    editor._move(-1)
    editor._list.setCurrentRow(len(before) - 1)
    editor._move(1)
    assert editor.levels() == before


# ──────────────────────────────────────────────────────────────────────────────
# Counts plots — a categorical column is not drawn as a distribution
# ──────────────────────────────────────────────────────────────────────────────
BINS = ["g0", "g1", "g2", "g3"]
#: Deliberately the reverse of the order the column was cut in.
BINS_WANTED = ["g3", "g2", "g1", "g0"]


@pytest.fixture(scope="module")
def binned():
    """A binned column whose rows arrive in a different order from its categories."""
    rng = np.random.default_rng(3)
    n = 958
    return pd.DataFrame(
        {
            "tacsctot_group": pd.Categorical(
                rng.choice(BINS, n, p=[0.57, 0.27, 0.09, 0.07]), categories=BINS
            ),
            "flow_mean": rng.normal(10.0, 2.0, n),
        }
    )


def _legend_levels(figure):
    """The legend entries, without the counts appended to them."""
    found = [a.get_legend() for a in figure.axes if a.get_legend() is not None]
    return [t.get_text().split(" ")[0] for t in found[0].get_texts()] if found else []


def test_counts_bars_follow_the_order(binned):
    """The reported bug: a categorical column takes the counts branch, which ignored the order."""
    figure = column_plot_static(
        binned, "tacsctot_group", kind="histogram",
        group="tacsctot_group", level_order=BINS_WANTED,
    )
    assert _ticks(figure) == BINS_WANTED


def test_counts_legend_follows_the_order(binned):
    """A legend that disagrees with the bars is worse than no legend."""
    figure = column_plot_static(
        binned, "tacsctot_group", kind="histogram",
        group="tacsctot_group", level_order=BINS_WANTED,
    )
    assert _legend_levels(figure) == BINS_WANTED


def test_counts_default_to_the_order_the_column_was_cut_in(binned):
    """Without an override a binned column keeps its categories, not the order its rows arrive
    in — and the legend has to agree with the bars there too."""
    figure = column_plot_static(binned, "tacsctot_group", kind="histogram", group="tacsctot_group")
    assert _ticks(figure) == BINS
    assert _legend_levels(figure) == BINS


def test_counts_order_without_a_split(binned):
    """No split still means axis groups: the counts are of the column's own levels."""
    figure = column_plot_static(
        binned, "tacsctot_group", kind="histogram", level_order=BINS_WANTED
    )
    assert _ticks(figure) == BINS_WANTED


def test_interactive_counts_follow_the_order(binned):
    """Plotly sorts a categorical axis by first appearance unless told otherwise."""
    pytest.importorskip("plotly")
    from nvitk.stats.interactive import column_plot

    figure = column_plot(
        binned, "tacsctot_group", kind="histogram",
        group="tacsctot_group", level_order=BINS_WANTED,
    )
    assert list(figure.layout.xaxis.categoryarray) == BINS_WANTED
    assert [t.name.split(" ")[0] for t in figure.data] == BINS_WANTED


def test_ordering_a_split_leaves_an_unrelated_x_axis_alone(binned):
    """One order feeds both axes, which is only safe because it ignores names it does not find."""
    pytest.importorskip("plotly")
    from nvitk.stats.interactive import column_plot

    figure = column_plot(binned, "flow_mean", group="tacsctot_group", level_order=BINS_WANTED)
    assert [t.name.split("<br>")[0] for t in figure.data if t.name] == BINS_WANTED


def test_base_levels_prefers_categories(binned):
    """The base a chosen order is applied on top of."""
    from nvitk.stats.distribution_plots import base_levels

    assert base_levels(binned["tacsctot_group"]) == BINS
    assert base_levels(pd.Series(["b", "a", "b"])) == ["b", "a"]


@pytest.fixture
def counts_dialog(qapp, binned):
    """The dialog on the binned column, split by itself — the reported configuration."""
    from nvitk.gui.panels.statmodels.column_plot_dialog import ColumnPlotDialog

    widget = ColumnPlotDialog(None, frame=binned, column="tacsctot_group")
    widget.show()
    names = [widget._split.itemText(i) for i in range(widget._split.count())]
    widget._split.setCurrentIndex(names.index("tacsctot_group"))
    kinds = [widget._kind.itemData(i) for i in range(widget._kind.count())]
    widget._kind.setCurrentIndex(kinds.index("histogram"))
    qapp.processEvents()
    yield widget
    widget.close()


def test_the_dialog_reorders_a_counts_plot(counts_dialog, qapp):
    counts_dialog._orders["tacsctot_group"] = BINS_WANTED
    counts_dialog._redraw()
    qapp.processEvents()
    assert _ticks(counts_dialog._static_figure) == BINS_WANTED


def test_the_status_line_lists_the_levels_as_the_axis_draws_them(counts_dialog, qapp):
    """Read against the figure, so a third order means reading it twice."""
    counts_dialog._orders["tacsctot_group"] = BINS_WANTED
    counts_dialog._redraw()
    qapp.processEvents()
    listed = counts_dialog._status.text().split("tacsctot_group:")[1]
    positions = [listed.index(level) for level in BINS_WANTED]
    assert positions == sorted(positions), listed


def test_order_is_offered_with_no_split_at_all(counts_dialog, qapp):
    """A categorical column is drawn as counts of its own levels whether or not it is split."""
    counts_dialog._facet_mode.setCurrentIndex(counts_dialog._facet_mode.findData(""))
    qapp.processEvents()
    assert counts_dialog._ordered_by() == "tacsctot_group"
    assert counts_dialog._btn_order.isEnabled()

    counts_dialog._orders["tacsctot_group"] = BINS_WANTED
    counts_dialog._redraw()
    qapp.processEvents()
    assert _ticks(counts_dialog._static_figure) == BINS_WANTED


def test_order_is_not_offered_for_a_bare_numeric_column(qapp, binned):
    """Nothing on that axis is a level."""
    from nvitk.gui.panels.statmodels.column_plot_dialog import ColumnPlotDialog

    widget = ColumnPlotDialog(None, frame=binned, column="flow_mean")
    widget.show()
    widget._facet_mode.setCurrentIndex(widget._facet_mode.findData(""))
    qapp.processEvents()
    assert widget._ordered_by() == ""
    assert not widget._btn_order.isEnabled()
    widget.close()
