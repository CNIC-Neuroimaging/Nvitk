"""
The N a distribution plot reports must be the N it is drawing.

Covers :mod:`nvitk.stats.group_counts` and the two backends that label figures through it. The
point of the shared module is that the Matplotlib figure and the Plotly one cannot disagree, so
most of these assert the *same* property against both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvitk.stats.group_counts import GroupCount, counts_note, displayed_counts

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from nvitk.stats.distribution_plots import (  # noqa: E402
    column_panels_static,
    column_plot_static,
)

#: Kinds that put each level on the categorical axis, and kinds that overlay them.
AXIS_KINDS = ("violin", "violin_points", "box", "box_points", "strip")
OVERLAY_KINDS = ("histogram", "density", "ecdf")


@pytest.fixture
def frame():
    """Three territories of unequal size, with missing values and filtered rows in two of them."""
    rng = np.random.default_rng(0)
    n = 60
    out = pd.DataFrame(
        {
            "subject_uid": [f"s{i:03d}" for i in range(n)],
            "territory": (["LICA"] * 25) + (["RICA"] * 20) + (["BASI"] * 15),
            "sex": (["M", "F"] * 30),
            "flow_mean": rng.normal(10.0, 2.0, n),
        }
    )
    out.loc[[3, 7, 40], "flow_mean"] = np.nan
    return out


@pytest.fixture
def excluded():
    """Two rows filtered out of LICA, three out of BASI."""
    mask = np.zeros(60, dtype=bool)
    mask[[1, 2, 50, 51, 52]] = True
    return mask


#: What the fixtures above come to: 25 LICA rows less 2 missing less 2 filtered, and so on.
EXPECTED = {"LICA": (21, 2), "RICA": (19, 0), "BASI": (12, 3)}


def _ticks(figure):
    """A figure's x tick labels, newlines flattened."""
    return [t.get_text().replace("\n", " ") for t in figure.axes[0].get_xticklabels()]


# ──────────────────────────────────────────────────────────────────────────────
# The counting itself
# ──────────────────────────────────────────────────────────────────────────────
def test_counts_separate_missing_from_filtered(frame, excluded):
    """A row with no value is not drawn; a filtered row is drawn greyed and counted apart."""
    counts = {
        c.level: c for c in displayed_counts(frame, "flow_mean", group="territory", excluded=excluded)
    }
    assert {k: (c.n, c.excluded) for k, c in counts.items()} == EXPECTED
    assert counts["LICA"].total == 23


def test_hiding_the_excluded_removes_them_from_the_count(frame, excluded):
    """What the figure is not drawing, the label must not claim."""
    counts = displayed_counts(
        frame, "flow_mean", group="territory", excluded=excluded, show_excluded=False
    )
    assert [(c.n, c.excluded) for c in counts] == [(n, 0) for n, _ in EXPECTED.values()]


def test_levels_order_and_absent_levels(frame, excluded):
    """The caller's draw order wins, and a level with nothing in it reports zero rather than
    being silently dropped."""
    counts = displayed_counts(
        frame, "flow_mean", group="territory", excluded=excluded,
        levels=["BASI", "LICA", "NOPE"],
    )
    assert [c.level for c in counts] == ["BASI", "LICA", "NOPE"]
    assert counts[-1] == GroupCount(level="NOPE", n=0, excluded=0)


def test_ungrouped_is_a_single_whole(frame, excluded):
    """Without a grouping there is one count: everything on the figure."""
    counts = displayed_counts(frame, "flow_mean", excluded=excluded)
    assert len(counts) == 1 and counts[0].level == ""
    assert counts[0].n == sum(n for n, _ in EXPECTED.values())


def test_missing_column_counts_nothing(frame):
    """A column that is not there is not an error — the caller has nothing to label."""
    assert displayed_counts(frame, "nope", group="territory") == []


def test_suffix_mentions_the_filtered_rows_only_when_there_are_any():
    """``n=21`` is the common case; the ``+2 excl`` is noise when it is zero."""
    assert GroupCount("LICA", 21).suffix() == "n=21"
    assert GroupCount("LICA", 21, 2).suffix() == "n=21 +2 excl"
    assert GroupCount("LICA", 21, 2).label(separator=" ") == "LICA (n=21 +2 excl)"


def test_counts_note_truncates_rather_than_running_off_the_window(frame, excluded):
    """A seventeen-vessel frame must not push the hint text off the dialog."""
    counts = [GroupCount(f"V{i}", i) for i in range(20)]
    note = counts_note(counts, total=GroupCount("", 190), group="vessel", limit=5)
    assert "+15 more" in note and note.startswith("n = 190")


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", AXIS_KINDS)
def test_static_puts_the_count_under_each_tick(frame, excluded, kind):
    """On the categorical kinds the level is an x tick, so that is where its N belongs."""
    figure = column_plot_static(
        frame, "flow_mean", kind=kind, group="territory", excluded_mask=excluded
    )
    ticks = _ticks(figure)
    for level, (n, excl) in EXPECTED.items():
        expected = f"{level} (n={n}" + (f" +{excl} excl)" if excl else ")")
        assert expected in ticks, (kind, ticks)


@pytest.mark.parametrize("kind", OVERLAY_KINDS)
def test_static_puts_the_count_in_the_legend(frame, excluded, kind):
    """On the overlaid kinds the level is only a legend entry, so that is where it goes."""
    figure = column_plot_static(
        frame, "flow_mean", kind=kind, group="territory", excluded_mask=excluded
    )
    legend = figure.axes[0].get_legend()
    texts = [t.get_text() for t in legend.get_texts()] if legend is not None else []
    for level, (n, _excl) in EXPECTED.items():
        assert any(level in t and f"n={n}" in t for t in texts), (kind, texts)


def test_static_pooled_n_is_the_sum_of_the_level_counts(frame, excluded):
    """Two numbers on one figure that do not add up is worse than one number."""
    figure = column_plot_static(
        frame, "flow_mean", kind="violin", group="territory", excluded_mask=excluded
    )
    summary = next(
        child.get_text() for child in figure.axes[0].texts if child.get_text().startswith("n = ")
    )
    assert summary.startswith(f"n = {sum(n for n, _ in EXPECTED.values())} +5 excl")


def test_static_panels_carry_their_own_n(frame, excluded):
    """Panels autoscale independently, so each has to say what it was drawn from."""
    figure = column_panels_static(
        frame, "flow_mean", facet_by="territory", excluded_mask=excluded
    )
    titles = [ax.get_title() for ax in figure.axes if ax.get_visible()]
    for level, (n, _excl) in EXPECTED.items():
        assert any(level in t and f"n={n}" in t for t in titles), titles


def test_static_ungrouped_is_left_alone(frame, excluded):
    """With no grouping there is no level to label, and the summary already gives the N."""
    figure = column_plot_static(frame, "flow_mean", kind="violin", excluded_mask=excluded)
    assert all("n=" not in tick for tick in _ticks(figure))


# ──────────────────────────────────────────────────────────────────────────────
# Plotly
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", ["violin", "box", "strip", "histogram", "density", "ecdf"])
def test_interactive_names_every_trace_with_its_count(frame, excluded, kind):
    """A Plotly trace name is both its legend entry and its categorical tick."""
    pytest.importorskip("plotly")
    from nvitk.stats.interactive import column_plot

    figure = column_plot(
        frame, "flow_mean", kind=kind, group="territory", excluded_mask=excluded
    )
    names = [t.name.replace("<br>", " ") for t in figure.data if t.name]
    for level, (n, _excl) in EXPECTED.items():
        assert any(level in name and f"n={n}" in name for name in names), (kind, names)


def test_interactive_count_equals_the_points_drawn(frame, excluded):
    """The whole claim, checked against the data actually handed to the renderer."""
    pytest.importorskip("plotly")
    from nvitk.stats.interactive import GREYED, column_plot

    figure = column_plot(
        frame, "flow_mean", kind="strip", group="territory", excluded_mask=excluded
    )
    for trace in figure.data:
        if not trace.name:
            continue
        level = trace.name.split("<br>")[0]
        kept, dropped = EXPECTED[level]
        # The coloured trace carries everything the level puts on the figure; the grey one is
        # drawn over the subset a filter removed. "n=21 +2 excl" is exactly that split.
        grey = getattr(trace.marker, "color", None) == GREYED
        assert len(trace.y) == (dropped if grey else kept + dropped), trace.name


def test_interactive_panels_carry_their_own_n(frame, excluded):
    """Same rule as the static panels, through the subplot titles."""
    pytest.importorskip("plotly")
    from nvitk.stats.interactive import column_panel_figure

    figure = column_panel_figure(
        frame, "flow_mean", facet_by="territory", excluded_mask=excluded
    )
    titles = [a.text for a in figure.layout.annotations if a.text]
    for level, (n, _excl) in EXPECTED.items():
        assert any(level in t and f"n={n}" in t for t in titles), titles


def test_both_backends_agree(frame, excluded):
    """The reason the counting lives in one module."""
    pytest.importorskip("plotly")
    from nvitk.stats.interactive import column_plot

    static = column_plot_static(
        frame, "flow_mean", kind="violin", group="territory", excluded_mask=excluded
    )
    interactive = column_plot(
        frame, "flow_mean", kind="violin", group="territory", excluded_mask=excluded
    )
    from_ticks = sorted(t.replace("\n", " ") for t in _ticks(static))
    from_traces = sorted(t.name.replace("<br>", " ") for t in interactive.data if t.name)
    assert from_ticks == from_traces
