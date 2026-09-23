"""
Pairwise contrasts and the significance brackets drawn from them.

The contrast arithmetic is checked against statsmodels' own ``t_test`` rather than against
recorded numbers: a linear combination of coefficients has one right answer, and matching it to
machine precision is worth more than any fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvitk.stats.pairwise import (
    ALPHA,
    format_p,
    series_colours,
    MODE_ALL,
    MODE_SIGNIFICANT,
    annotate_axes,
    bracket_layout,
    contrast_note,
    eligible,
    holm,
    level_positions,
    pairwise_contrasts,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

LEVELS = ["LICA", "RICA", "BASI", "LMCA", "RMCA", "SSS"]
#: BASI and SSS are genuinely lower; the four arterial levels are drawn from the *same* mean, so
#: "these must not differ" is true by construction rather than merely likely. A 0.4 gap at n≈50
#: is comfortably detectable, and asserting it away would be asserting the model is wrong.
MEANS = {"LICA": 10.0, "RICA": 10.0, "BASI": 6.0, "LMCA": 10.0, "RMCA": 10.0, "SSS": 4.0}


@pytest.fixture(scope="module")
def frame():
    """A cohort with one clearly separated pair of levels and four indistinguishable ones."""
    rng = np.random.default_rng(3)
    n = 300
    out = pd.DataFrame(
        {
            "subject_uid": [f"s{i % 40:02d}" for i in range(n)],
            "territory": rng.choice(LEVELS, n),
            "age_c": rng.normal(0.0, 1.0, n),
        }
    )
    out["flow"] = (
        [MEANS[t] for t in out["territory"]] + 0.5 * out["age_c"] + rng.normal(0, 1.0, n)
    )
    return out


@pytest.fixture(scope="module")
def fit(frame):
    """An OLS with the factor and one covariate."""
    smf = pytest.importorskip("statsmodels.formula.api")
    return smf.ols("flow ~ territory + age_c", data=frame).fit()


@pytest.fixture(scope="module")
def contrasts(fit, frame):
    """Every pairwise comparison of the six levels."""
    return pairwise_contrasts(fit, frame, factor="territory", levels=LEVELS)


def _positions():
    """Level → x index, as a categorical axis lays them out."""
    return {level: float(index) for index, level in enumerate(LEVELS)}


# ──────────────────────────────────────────────────────────────────────────────
# Holm
# ──────────────────────────────────────────────────────────────────────────────
def test_holm_matches_statsmodels():
    """One right answer, and it is not this module's to invent."""
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    raw = [0.001, 0.02, 0.04, 0.6, 0.0003]
    expected = multitest.multipletests(raw, method="holm")[1]
    assert np.allclose(holm(raw), expected)


def test_holm_passes_missing_through():
    """A comparison with no p-value must not consume a step of the correction."""
    out = holm([0.01, np.nan, 0.5])
    assert np.isnan(out[1])
    assert np.allclose(out[[0, 2]], holm([0.01, 0.5]))


def test_holm_is_monotone():
    """An adjusted p may never fall below one that preceded it in the step-down."""
    out = holm([0.04, 0.03, 0.02, 0.01])
    assert np.all(np.diff(out[np.argsort([0.04, 0.03, 0.02, 0.01])]) >= -1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# The contrasts themselves
# ──────────────────────────────────────────────────────────────────────────────
def test_every_pair_is_compared(contrasts):
    """Six levels are fifteen comparisons, each exactly once."""
    assert len(contrasts) == len(LEVELS) * (len(LEVELS) - 1) // 2
    pairs = {frozenset((row.a, row.b)) for row in contrasts.itertuples(index=False)}
    assert len(pairs) == len(contrasts)


def test_contrast_matches_statsmodels_t_test(contrasts, fit):
    """The estimate, its standard error and its p, against the same linear combination run
    through statsmodels itself."""
    names = list(fit.params.index)

    def vector(a: str, b: str) -> np.ndarray:
        out = np.zeros(len(names))
        for level, sign in ((a, 1.0), (b, -1.0)):
            term = f"territory[T.{level}]"
            if term in names:
                out[names.index(term)] += sign
        return out

    for row in contrasts.itertuples(index=False):
        reference = fit.t_test(vector(row.a, row.b))
        assert row.estimate == pytest.approx(float(np.ravel(reference.effect)[0]), abs=1e-9)
        assert row.se == pytest.approx(float(np.ravel(reference.sd)[0]), abs=1e-9)
        assert row.p_value == pytest.approx(float(reference.pvalue), abs=1e-9)


def test_the_separated_levels_are_the_significant_ones(contrasts):
    """A sanity check on the simulation: BASI and SSS differ from the arterial levels, and the
    arterial levels do not differ from each other."""
    arterial = {"LICA", "RICA", "LMCA", "RMCA"}
    for row in contrasts.itertuples(index=False):
        if {row.a, row.b} <= arterial:
            assert row.p_adj > ALPHA, (row.a, row.b, row.p_adj)
        else:
            assert row.p_adj < ALPHA, (row.a, row.b, row.p_adj)


def test_correction_follows_the_levels_asked_for(fit, frame):
    """Hiding levels must not leave the rest carrying the penalty of comparisons never made."""
    two = pairwise_contrasts(fit, frame, factor="territory", levels=["LICA", "BASI"])
    assert len(two) == 1
    assert two["p_adj"].iloc[0] == pytest.approx(two["p_value"].iloc[0])


def test_sorted_most_significant_first(contrasts):
    """The bracket cap keeps the head of the frame, so the order is load-bearing."""
    adjusted = contrasts["p_adj"].to_numpy()
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_missing_covariates_are_held_at_their_own_reference(fit, frame):
    """A caller that names no references still gets an adjusted comparison, not a patsy
    NameError for the covariate it did not mention."""
    out = pairwise_contrasts(fit, frame, factor="territory", levels=LEVELS)
    assert len(out) == 15 and out["se"].notna().all()


def test_a_factor_with_one_level_is_an_error(fit, frame):
    """Nothing to compare is a mistake worth naming, not an empty frame."""
    with pytest.raises(ValueError):
        pairwise_contrasts(fit, frame, factor="territory", levels=["LICA"])
    with pytest.raises(ValueError):
        pairwise_contrasts(fit, frame, factor="nope")


def test_mixedlm_goes_down_the_same_path(frame):
    """MixedLM's cov_params also covers the variance components; they are not in a fixed-effect
    contrast and must be subset away."""
    smf = pytest.importorskip("statsmodels.formula.api")
    fitted = smf.mixedlm("flow ~ territory + age_c", frame, groups=frame["subject_uid"]).fit()
    out = pairwise_contrasts(fitted, frame, factor="territory", levels=LEVELS)
    assert len(out) == 15
    assert out["se"].notna().all() and (out["se"] > 0).all()


# ──────────────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────────────
def test_level_positions_see_through_the_group_count_label():
    """The distribution plots put the N on a second line; the level is the first."""
    assert level_positions(["LICA\n(n=21 +2 excl)", "RICA\n(n=19)"], [0.0, 1.0]) == {
        "LICA": 0.0,
        "RICA": 1.0,
    }


def test_brackets_never_overlap_within_a_row(contrasts):
    """Two brackets on one row would draw as a single line spanning both."""
    brackets, _ = bracket_layout(contrasts, _positions(), mode=MODE_ALL, max_brackets=99)
    rows: dict[int, list[tuple[float, float]]] = {}
    for bracket in brackets:
        rows.setdefault(bracket.row, []).append((bracket.left, bracket.right))
    for spans in rows.values():
        spans.sort()
        for (_l1, r1), (l2, _r2) in zip(spans, spans[1:]):
            assert r1 < l2, spans


def test_significant_mode_drops_the_rest(contrasts):
    """'significant only' must contain no NS and no NA."""
    brackets, _ = bracket_layout(contrasts, _positions(), mode=MODE_SIGNIFICANT, max_brackets=99)
    assert brackets
    assert all(bracket.p_adj < ALPHA for bracket in brackets)
    assert all(bracket.stars not in {"NS", "NA"} for bracket in brackets)


def test_all_mode_keeps_the_non_significant_ones(contrasts):
    """'all' is the mode that answers 'was this tested at all'."""
    brackets, _ = bracket_layout(contrasts, _positions(), mode=MODE_ALL, max_brackets=99)
    assert len(brackets) == len(contrasts)
    assert any(bracket.stars == "NS" for bracket in brackets)


def test_the_cap_keeps_the_most_significant(contrasts):
    """A capped figure must not be a arbitrary sample of the comparisons."""
    brackets, omitted = bracket_layout(contrasts, _positions(), mode=MODE_ALL, max_brackets=4)
    assert len(brackets) == 4 and omitted == len(contrasts) - 4
    assert max(b.p_adj for b in brackets) <= contrasts["p_adj"].iloc[4]


def test_levels_absent_from_the_axis_are_not_drawn(contrasts):
    """A grouped display's panel holds some of the levels; a bracket to one it does not have
    would span the wrong tick."""
    brackets, _ = bracket_layout(
        contrasts, {"LICA": 0.0, "BASI": 1.0}, mode=MODE_ALL, max_brackets=99
    )
    assert len(brackets) == 1


def test_eligible_matches_what_gets_drawn(contrasts):
    """The status note counts through this, so it has to agree with the layout."""
    for mode in (MODE_ALL, MODE_SIGNIFICANT):
        brackets, omitted = bracket_layout(contrasts, _positions(), mode=mode, max_brackets=99)
        assert len(eligible(contrasts, mode)) == len(brackets) + omitted


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────
def test_annotate_axes_makes_room_without_moving_the_floor(contrasts, frame):
    """Brackets go above the data; raising the ceiling is fine, dropping the floor rescales it."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.plot(range(len(LEVELS)), [MEANS[level] for level in LEVELS])
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(LEVELS)
    before = ax.get_ylim()

    drawn, omitted = annotate_axes(ax, contrasts, mode=MODE_SIGNIFICANT)
    after = ax.get_ylim()
    assert drawn > 0
    assert after[1] > before[1]
    assert after[0] == pytest.approx(before[0])
    assert contrast_note(drawn, omitted, MODE_SIGNIFICANT).startswith(f"{drawn} pairwise")
    plt.close(figure)


def test_annotate_axes_is_a_no_op_without_matching_ticks(contrasts):
    """A continuous axis has no levels to span, and must not be annotated as though it had."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.plot([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    before = ax.get_ylim()
    drawn, _omitted = annotate_axes(ax, contrasts, mode=MODE_ALL)
    assert drawn == 0
    assert ax.get_ylim() == pytest.approx(before)
    plt.close(figure)


def test_plotly_gets_the_same_brackets(contrasts):
    """The interactive backend draws from the same layout, positioned by category index."""
    go = pytest.importorskip("plotly.graph_objects")
    from nvitk.stats.pairwise import annotate_plotly

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=LEVELS, y=[MEANS[level] for level in LEVELS]))
    drawn, _omitted = annotate_plotly(figure, contrasts, LEVELS, mode=MODE_SIGNIFICANT)

    expected, _ = bracket_layout(contrasts, _positions(), mode=MODE_SIGNIFICANT)
    assert drawn == len(expected)
    assert len(figure.layout.shapes) == drawn
    labels = sorted(a.text for a in figure.layout.annotations)
    assert labels == sorted(b.label().replace("\n", "<br>") for b in expected)


def test_note_says_when_nothing_reached_significance():
    """An empty 'significant only' figure has to say why it is empty."""
    empty = pd.DataFrame(columns=["a", "b", "p_adj", "stars"])
    assert "No pairwise comparison reached" in contrast_note(0, 0, MODE_SIGNIFICANT)
    assert bracket_layout(empty, _positions(), mode=MODE_SIGNIFICANT) == ([], 0)


# ──────────────────────────────────────────────────────────────────────────────
# Comparisons within a second factor
# ──────────────────────────────────────────────────────────────────────────────
#: A plaque effect that exists in one territory only — the shape of the model that prompted this.
BY_SLOPES = {"LICA": 0.0, "RICA": 0.35, "BASI": 0.0}


@pytest.fixture(scope="module")
def interaction():
    """A frame and an ``x * by`` fit whose effect lives in a single ``by`` level."""
    smf = pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(21)
    n = 800
    steps = {"g0": 0, "g1": 1, "g2": 2, "g3": 3}
    frame = pd.DataFrame(
        {
            "grp": rng.choice(list(steps), n),
            "territory": rng.choice(list(BY_SLOPES), n),
            "age_c": rng.normal(0.0, 1.0, n),
        }
    )
    frame["y"] = (
        [{"LICA": 10.0, "RICA": 10.0, "BASI": 6.0}[t] for t in frame["territory"]]
        + np.array(
            [BY_SLOPES[t] * steps[g] for t, g in zip(frame["territory"], frame["grp"])]
        )
        + 0.3 * frame["age_c"]
        + rng.normal(0, 1.0, n)
    )
    return frame, smf.ols("y ~ grp * territory + age_c", data=frame).fit()


def test_by_compares_within_each_level(interaction):
    """Four levels within each of three territories is eighteen comparisons, each labelled."""
    frame, fitted = interaction
    out = pairwise_contrasts(fitted, frame, factor="grp", by="territory")
    assert len(out) == 3 * 6
    assert set(out["by"]) == set(BY_SLOPES)
    for group, block in out.groupby("by"):
        assert len(block) == 6, group


def test_by_recovers_a_simple_effect_the_average_hides(interaction):
    """The whole reason a per-series bracket cannot use the averaged contrast."""
    frame, fitted = interaction

    def g0_vs_g3(table: pd.DataFrame) -> float:
        row = table.loc[
            ((table["a"] == "g0") & (table["b"] == "g3"))
            | ((table["a"] == "g3") & (table["b"] == "g0"))
        ]
        return float(row["estimate"].abs().iloc[0])

    within = pairwise_contrasts(fitted, frame, factor="grp", by="territory")
    rica = within.loc[within["by"] == "RICA"]
    lica = within.loc[within["by"] == "LICA"]

    # 3 steps x 0.35 in RICA, nothing in LICA.
    assert g0_vs_g3(rica) == pytest.approx(1.05, abs=0.3)
    assert g0_vs_g3(lica) == pytest.approx(0.0, abs=0.3)
    assert rica["p_adj"].min() < ALPHA
    # The averaged contrast dilutes it across three territories, which is the wrong claim to
    # put over a single curve.
    averaged = pairwise_contrasts(fitted, frame, factor="grp")
    assert g0_vs_g3(averaged) < g0_vs_g3(rica) / 2


def test_by_levels_restricts_both_the_grid_and_the_correction(interaction):
    """Hiding a series must not leave the others carrying its comparisons' penalty."""
    frame, fitted = interaction
    two = pairwise_contrasts(
        fitted, frame, factor="grp", by="territory", by_levels=["RICA", "LICA"]
    )
    assert set(two["by"]) == {"RICA", "LICA"}
    assert len(two) == 2 * 6
    full = pairwise_contrasts(fitted, frame, factor="grp", by="territory")
    # Same raw p, smaller family, so the adjusted p can only improve.
    key = ["by", "a", "b"]
    merged = two.merge(full, on=key, suffixes=("_two", "_full"))
    assert np.allclose(merged["p_value_two"], merged["p_value_full"])
    assert (merged["p_adj_two"] <= merged["p_adj_full"] + 1e-12).all()


def test_brackets_band_by_series_rather_than_interleaving(interaction):
    """A band is what makes "these are RICA's" readable once three series overlap."""
    frame, fitted = interaction
    out = pairwise_contrasts(fitted, frame, factor="grp", by="territory")
    positions = {level: float(i) for i, level in enumerate(["g0", "g1", "g2", "g3"])}
    brackets, _ = bracket_layout(out, positions, mode=MODE_ALL, max_brackets=99)

    rows_by_series: dict[str, set[int]] = {}
    for bracket in brackets:
        rows_by_series.setdefault(bracket.by, set()).add(bracket.row)
    bands = sorted(rows_by_series.items(), key=lambda kv: min(kv[1]))
    for (_a, first), (_b, second) in zip(bands, bands[1:]):
        assert max(first) < min(second), rows_by_series


def test_bracket_label_carries_the_stars_and_the_number():
    """Requested explicitly: the marker to read, the value to cite."""
    from nvitk.stats.pairwise import Bracket

    bracket = Bracket(left=0.0, right=1.0, row=0, stars="***", p_adj=0.0004, by="RICA")
    assert bracket.label() == "***\np=0.0004"
    assert bracket.label(show_p=False) == "***"
    assert Bracket(0.0, 1.0, 0, "", float("nan")).label() == "NA\np n/a"


def test_format_p_stays_short():
    """A wide number widens the bracket it sits on."""
    assert format_p(0.0) == "p<1e-4"
    assert format_p(0.0004) == "p=0.0004"
    assert format_p(0.5) == "p=0.5"
    assert format_p(float("nan")) == "p n/a"


def test_series_colours_reads_either_legend_spelling():
    """The plotters spell a legend entry both ways."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="#111111", label="RICA")
    ax.plot([0, 1], [1, 0], color="#222222", label="EMM territory=LICA")
    ax.legend()
    assert set(series_colours(ax)) == {"RICA", "LICA"}
    plt.close(figure)


def test_a_panel_only_gets_its_own_series_brackets(interaction):
    """The grouped display splits the series across panels; a bracket for a curve that is not
    on this panel would sit over nothing."""
    import matplotlib.pyplot as plt

    frame, fitted = interaction
    out = pairwise_contrasts(fitted, frame, factor="grp", by="territory")

    figure, ax = plt.subplots()
    ax.plot([0, 1, 2, 3], [1, 2, 3, 4], color="#111111", label="territory=RICA")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["g0", "g1", "g2", "g3"])
    ax.legend()

    drawn, _omitted = annotate_axes(ax, out, mode=MODE_ALL, max_brackets=99)
    assert drawn == 6, "only RICA's six comparisons belong on a RICA-only panel"
    plt.close(figure)
