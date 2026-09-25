"""
The confidence band has to describe the curve it is drawn around.

The plotted curve holds each categorical covariate at its **modal** level; ``emmeans`` averages
over every factor outside its specification with **equal weights**. On a model with a categorical
covariate those are different numbers, so the band came out displaced from the line by a constant
— the modal level's effect minus the mean of them — which reads as a band that is simply in the
wrong place.

These are numeric checks against the predicted curve rather than pixel checks on the figure: the
question is whether two computations agree, and the drawing is downstream of that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rpy2")

from nvitk.stats.r_mixedlm import _emmeans_band, r_backend_status  # noqa: E402
from nvitk.stats.r_robust import robust_backend_status  # noqa: E402

needs_lmrob = pytest.mark.skipif(
    not robust_backend_status().available, reason="needs R + robustbase"
)
needs_lme4 = pytest.mark.skipif(
    not r_backend_status().available, reason="needs R + pymer4 + lme4"
)

TERRITORIES = ["BASILAR", "LICA", "RICA"]
#: Unbalanced and with a real effect, so the modal level differs from the average.
EFFECT = {"BASILAR": 0.0, "LICA": 0.04, "RICA": 0.08}
FIXED = "log1p_pi + group_key"


@pytest.fixture(scope="module")
def frame():
    """A cohort where holding the factor at its mode and averaging over it differ."""
    rng = np.random.default_rng(11)
    n = 900
    out = pd.DataFrame(
        {
            "subject_uid": [f"s{i % 180:03d}" for i in range(n)],
            "group_key": rng.choice(TERRITORIES, n, p=[0.5, 0.3, 0.2]),
            "log1p_pi": rng.normal(0.55, 0.06, n),
        }
    )
    out["att_mean"] = (
        1.20
        + 0.12 * out["log1p_pi"]
        + [EFFECT[g] for g in out["group_key"]]
        + rng.normal(0, 0.06, n)
    )
    return out


@pytest.fixture(scope="module")
def grid(frame):
    """Five x values spanning the observed range."""
    return np.linspace(float(frame["log1p_pi"].min()), float(frame["log1p_pi"].max()), 5)


@pytest.fixture(scope="module")
def robust(frame):
    """A robust fit with one continuous and one categorical predictor."""
    from nvitk.stats.r_robust import fit_lmrob

    fit, _f, _meta = fit_lmrob(data=frame, formula=f"att_mean ~ {FIXED}")
    return fit


def _modal(frame: pd.DataFrame) -> str:
    return str(frame["group_key"].mode().iloc[0])


def _curve(fit, grid, level: str):
    """The population curve as the plotter predicts it: x varying, the factor at *level*."""
    from nvitk.stats.r_robust import lmrob_predict

    return lmrob_predict(fit, pd.DataFrame({"log1p_pi": grid, "group_key": level}))


# ──────────────────────────────────────────────────────────────────────────────
# The defect
# ──────────────────────────────────────────────────────────────────────────────
@needs_lmrob
def test_without_the_reference_the_band_is_displaced_by_a_constant(robust, frame, grid):
    """Pins the defect itself, so a future refactor that drops ``hold`` fails here rather than
    shipping a band that is quietly in the wrong place."""
    from nvitk.stats.r_robust import _lmrob_band

    bands = _lmrob_band(
        robust, x="log1p_pi", x_values=grid, group="", levels=[],
        continuous=True, fixed_formula=FIXED, ci_level=0.95,
    )
    offset = _curve(robust, grid, _modal(frame)) - bands[None]["emmean"].to_numpy()
    assert np.allclose(offset, offset[0], atol=1e-9), "not even a constant offset"
    assert abs(offset[0]) > 0.01, "this fixture no longer exercises the mismatch"


# ──────────────────────────────────────────────────────────────────────────────
# The fix
# ──────────────────────────────────────────────────────────────────────────────
@needs_lmrob
def test_the_band_centre_is_the_predicted_curve(robust, frame, grid):
    """What the figure claims: the interval belongs to the line it surrounds."""
    from nvitk.stats.r_robust import _lmrob_band

    bands = _lmrob_band(
        robust, x="log1p_pi", x_values=grid, group="", levels=[],
        continuous=True, fixed_formula=FIXED, ci_level=0.95,
        hold={"group_key": _modal(frame)},
    )
    assert np.allclose(
        bands[None]["emmean"].to_numpy(), _curve(robust, grid, _modal(frame)), atol=1e-9
    )


@needs_lmrob
def test_the_interval_straddles_its_centre(robust, frame, grid):
    from nvitk.stats.r_robust import _lmrob_band

    bands = _lmrob_band(
        robust, x="log1p_pi", x_values=grid, group="", levels=[],
        continuous=True, fixed_formula=FIXED, ci_level=0.95,
        hold={"group_key": _modal(frame)},
    )
    band = bands[None]
    assert (band["lower.CL"] < band["emmean"]).all()
    assert (band["upper.CL"] > band["emmean"]).all()


@needs_lmrob
def test_each_level_gets_its_own_curve_when_grouped(robust, frame, grid):
    """Pinning the other factors must not collapse the per-level split."""
    from nvitk.stats.r_robust import _lmrob_band

    bands = _lmrob_band(
        robust, x="log1p_pi", x_values=grid, group="group_key", levels=TERRITORIES,
        continuous=True, fixed_formula=FIXED, ci_level=0.95,
        hold={"group_key": _modal(frame)},
    )
    assert set(bands) == set(TERRITORIES)
    for level in TERRITORIES:
        assert np.allclose(
            bands[level]["emmean"].to_numpy(), _curve(robust, grid, level), atol=1e-9
        ), level


@needs_lmrob
def test_a_numeric_covariate_needs_no_pin(robust, frame, grid):
    """``emmeans`` already holds a continuous covariate at its mean, which is what the reference
    row does — so only the categoricals are named, and a numeric entry is ignored."""
    from nvitk.stats.r_robust import _lmrob_band

    with_number = _lmrob_band(
        robust, x="log1p_pi", x_values=grid, group="", levels=[],
        continuous=True, fixed_formula=FIXED, ci_level=0.95,
        hold={"group_key": _modal(frame), "log1p_pi": 0.5, "unused_numeric": 3.0},
    )
    without = _lmrob_band(
        robust, x="log1p_pi", x_values=grid, group="", levels=[],
        continuous=True, fixed_formula=FIXED, ci_level=0.95,
        hold={"group_key": _modal(frame)},
    )
    assert np.allclose(
        with_number[None]["emmean"].to_numpy(), without[None]["emmean"].to_numpy()
    )


@needs_lmrob
def test_a_column_the_formula_never_mentions_is_not_pinned(robust, frame, grid):
    """Asking emmeans for a variable its reference grid has no column for is an error, and the
    reference row carries every column of the frame."""
    from nvitk.stats.r_robust import _lmrob_band

    bands = _lmrob_band(
        robust, x="log1p_pi", x_values=grid, group="", levels=[],
        continuous=True, fixed_formula=FIXED, ci_level=0.95,
        hold={"group_key": _modal(frame), "subject_uid": "s000"},
    )
    assert bands is not None
    assert np.allclose(
        bands[None]["emmean"].to_numpy(), _curve(robust, grid, _modal(frame)), atol=1e-9
    )


@pytest.mark.slow
@needs_lme4
@needs_lmrob
def test_real_emmeans_accepts_the_extended_specification(frame, grid):
    """lme4 and MMRM go through emmeans itself rather than the design-matrix fallback, so the
    ``~ x | a * b`` spec has to be one emmeans takes — otherwise the band silently disappears."""
    from nvitk.stats.r_mixedlm import fit_lme4, lme4_predict

    model, _f, _meta = fit_lme4(
        data=frame, formula=f"att_mean ~ {FIXED} + (1 | subject_uid)"
    )
    bands = _emmeans_band(
        model, x="log1p_pi", x_values=grid, group="", levels=[],
        continuous=True, fixed_formula=FIXED, ci_level=0.95,
        hold={"group_key": _modal(frame)},
    )
    assert bands is not None, "emmeans rejected the spec and the band was lost"
    expected = lme4_predict(
        model,
        pd.DataFrame(
            {
                "log1p_pi": grid,
                "group_key": _modal(frame),
                "subject_uid": frame["subject_uid"].iloc[0],
            }
        ),
        use_random_effects=False,
    )
    assert np.allclose(bands[None]["emmean"].to_numpy(), expected, atol=1e-7)


# ──────────────────────────────────────────────────────────────────────────────
# What counts as one family of tests
# ──────────────────────────────────────────────────────────────────────────────
#: Enough levels that pooling them swamps a real effect — the reported case had seventeen.
MANY = [f"t{i:02d}" for i in range(17)]
BINS = ["g0", "g1", "g2", "g3"]


@pytest.fixture(scope="module")
def interaction():
    """A frame with one real effect, in one level of the second factor only."""
    smf = pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(5)
    rows = []
    for subject in range(120):
        plaque = BINS[subject % 4]
        for index, level in enumerate(MANY):
            rows.append(
                {
                    "territory": level,
                    "plaque": plaque,
                    "age_c": rng.normal(),
                    # The effect lives in t08 at g1 and nowhere else.
                    "flow_mean": 100 + 8 * index
                    + 14.0 * (level == "t08") * (plaque == "g1")
                    + rng.normal(0, 12),
                }
            )
    frame = pd.DataFrame(rows)
    return frame, smf.ols("flow_mean ~ plaque * territory + age_c", data=frame).fit()


def _pair(table, group, a, b):
    """The one comparison of *a* and *b* within *group*, whichever way round it came out."""
    rows = table.loc[
        (table["by"] == group)
        & (((table["a"] == a) & (table["b"] == b)) | ((table["a"] == b) & (table["b"] == a)))
    ]
    assert len(rows) == 1, (group, a, b)
    return rows.iloc[0]


def test_the_correction_family_is_the_series_not_the_figure(interaction):
    """The reported symptom: seventeen territories pooled into one family of 102 tests turned a
    real effect (raw p = 0.003) into p = 0.30, so every bracket read NS and "significant only"
    drew nothing."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = interaction
    table = pairwise_contrasts(fitted, frame, factor="plaque", by="territory")
    assert len(table) == len(MANY) * 6

    real = _pair(table, "t08", "g0", "g1")
    assert real["p_value"] < 0.01, "the fixture no longer contains a detectable effect"
    assert real["p_adj"] < 0.05, real["p_adj"]
    assert real["stars"] not in {"NS", "NA"}


def test_each_series_is_corrected_on_its_own(interaction):
    """Six comparisons per territory, so the adjusted p is the within-series step-down."""
    from nvitk.stats.pairwise import holm, pairwise_contrasts

    frame, fitted = interaction
    table = pairwise_contrasts(fitted, frame, factor="plaque", by="territory")
    for level, block in table.groupby("by"):
        expected = holm(block["p_value"].to_numpy())
        assert np.allclose(block["p_adj"].to_numpy(), expected), level


def test_without_a_by_the_whole_set_is_one_family(interaction):
    """No series means no per-series family; the comparisons are corrected together."""
    from nvitk.stats.pairwise import holm, pairwise_contrasts

    frame, fitted = interaction
    table = pairwise_contrasts(fitted, frame, factor="plaque")
    assert (table["by"] == "").all()
    assert np.allclose(table["p_adj"].to_numpy(), holm(table["p_value"].to_numpy()))


# ──────────────────────────────────────────────────────────────────────────────
# What a bracket is testing
# ──────────────────────────────────────────────────────────────────────────────
GRID = ["g0", "g1", "g2", "g3"]
PLACES = ["BASILAR", "LICA", "RICA", "LPCOMM"]


@pytest.fixture(scope="module")
def crossed():
    """A frame with an interaction living in one territory, and its OLS fit."""
    smf = pytest.importorskip("statsmodels.formula.api")
    rng = np.random.default_rng(5)
    level = {"BASILAR": 114.0, "LICA": 215.0, "RICA": 208.0, "LPCOMM": 27.0}
    rows = []
    for subject in range(150):
        plaque = GRID[subject % 4]
        for place in PLACES:
            rows.append(
                {
                    "territory": place,
                    "gr": plaque,
                    "age_c": rng.normal(),
                    "flow_mean": level[place]
                    + 14.0 * (place == "RICA") * (plaque == "g1")
                    + rng.normal(0, 12),
                }
            )
    frame = pd.DataFrame(rows)
    return frame, smf.ols("flow_mean ~ gr * territory + age_c", data=frame).fit()


def test_the_reference_level_is_found_by_elimination(crossed):
    """The level the coefficient table is silent about — which is what makes an interaction
    contrast computed here line up with the interaction row the table prints."""
    from nvitk.stats.pairwise import reference_level

    frame, fitted = crossed
    assert reference_level(fitted, frame, "territory") == "BASILAR"


def test_an_interaction_bracket_equals_the_table_s_interaction_coefficient(crossed):
    """The whole point of the basis: read a model's interactions off the figure."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    table = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction"
    )
    for place in PLACES:
        if place == "BASILAR":
            continue
        row = _pair(table, place, "g0", "g1")
        # ``a - b`` versus ``b - a`` flips the sign the table prints.
        estimate = row["estimate"] if row["a"] == "g1" else -row["estimate"]
        term = f"gr[T.g1]:territory[T.{place}]"
        assert estimate == pytest.approx(float(fitted.params[term]), abs=1e-9), place
        assert row["se"] == pytest.approx(float(fitted.bse[term]), abs=1e-9), place
        assert row["p_value"] == pytest.approx(float(fitted.pvalues[term]), abs=1e-9), place


def test_the_reference_series_gets_no_interaction_brackets(crossed):
    """Its own contrasts are zero by construction — it is the baseline, not a comparison."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    table = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction"
    )
    assert "BASILAR" not in set(table["by"])
    assert set(table["by"]) == set(PLACES) - {"BASILAR"}
    assert table.attrs["reference"] == "BASILAR"


def test_within_and_interaction_are_different_questions(crossed):
    """They disagree on a model with an interaction, which is why the choice exists — and why
    reading one off the other's figure was the reported confusion."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    within = pairwise_contrasts(fitted, frame, factor="gr", by="territory", basis="within")
    across = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction"
    )
    here = _pair(within, "RICA", "g0", "g1")["estimate"]
    versus = _pair(across, "RICA", "g0", "g1")["estimate"]
    assert abs(here - versus) > 1e-6


def test_an_interaction_basis_needs_a_second_factor(crossed):
    """With one grouping there is no interaction to test, and saying so beats an empty figure."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    with pytest.raises(ValueError, match="needs a second factor"):
        pairwise_contrasts(fitted, frame, factor="gr", basis="interaction")


def test_selecting_only_the_reference_is_refused(crossed):
    """Nothing left to measure against the baseline."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    with pytest.raises(ValueError, match="reference"):
        pairwise_contrasts(
            fitted, frame, factor="gr", by="territory",
            by_levels=["BASILAR"], basis="interaction",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Whether the p a bracket shows is corrected
# ──────────────────────────────────────────────────────────────────────────────
def test_raw_mode_reproduces_the_coefficient_table_exactly(crossed):
    """A regression table corrects nothing, so a corrected figure can never agree with it. This
    is the setting that makes them agree."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    table = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction", adjust="none"
    )
    for place in set(PLACES) - {"BASILAR"}:
        row = _pair(table, place, "g0", "g1")
        term = f"gr[T.g1]:territory[T.{place}]"
        assert row["p_adj"] == pytest.approx(float(fitted.pvalues[term]), abs=1e-12), place
        assert row["p_adj"] == pytest.approx(row["p_value"], abs=1e-12)


def test_holm_is_the_default_and_is_stricter(crossed):
    """Six comparisons per series is enough to turn a table's 0.0145 into 0.087."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    corrected = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction"
    )
    raw = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction", adjust="none"
    )
    assert corrected.attrs["adjust"] == "holm"
    assert raw.attrs["adjust"] == "none"

    key = ["by", "a", "b"]
    merged = corrected.merge(raw, on=key, suffixes=("_holm", "_raw"))
    assert len(merged) == len(corrected)
    assert (merged["p_adj_holm"] >= merged["p_adj_raw"] - 1e-12).all()
    assert (merged["p_adj_holm"] > merged["p_adj_raw"] + 1e-12).any()


def test_the_stars_follow_the_adjustment(crossed):
    """The marker is read off ``p_adj``, so it has to be restated when the adjustment changes."""
    from nvitk.stats.mixedlm import significance_stars
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    for setting in ("holm", "none"):
        table = pairwise_contrasts(
            fitted, frame, factor="gr", by="territory", basis="interaction", adjust=setting
        )
        for row in table.itertuples(index=False):
            assert row.stars == significance_stars(float(row.p_adj)), (setting, row.by)


def test_raw_mode_applies_to_the_within_basis_too(crossed):
    """The adjustment is a separate axis from what is being compared."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    table = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="within", adjust="none"
    )
    assert np.allclose(table["p_adj"].to_numpy(), table["p_value"].to_numpy())


def test_one_bracket_per_interaction_row_and_no_more(crossed):
    """An interaction *term* contrasts a level against the factor's own reference. Any other
    pair is the difference of two of them — a real contrast, but one the table never prints, so
    it cannot be cross-checked and does not belong to a basis that claims to show the model's
    interactions."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    rows = [t for t in fitted.params.index if ":" in t and t.startswith("gr[")]
    table = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction", adjust="none"
    )
    assert len(table) == len(rows)

    for row in table.itertuples(index=False):
        assert "g0" in {row.a, row.b}, (row.a, row.b)
        other = row.b if row.a == "g0" else row.a
        term = f"gr[T.{other}]:territory[T.{row.by}]"
        assert row.p_adj == pytest.approx(float(fitted.pvalues[term]), abs=1e-12), term


def test_the_within_basis_still_compares_every_pair(crossed):
    """The restriction belongs to the interaction basis alone — a simple effect between any two
    levels is a perfectly good question."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    table = pairwise_contrasts(fitted, frame, factor="gr", by="territory", basis="within")
    for _level, block in table.groupby("by"):
        assert len(block) == 6
        assert not block.apply(lambda r: "g0" in {r["a"], r["b"]}, axis=1).all()


def test_the_correction_family_shrinks_with_the_restriction(crossed):
    """Three comparisons per series, not six — the correction has to be over what is shown."""
    from nvitk.stats.pairwise import holm, pairwise_contrasts

    frame, fitted = crossed
    table = pairwise_contrasts(fitted, frame, factor="gr", by="territory", basis="interaction")
    for level, block in table.groupby("by"):
        assert len(block) == 3, level
        assert np.allclose(block["p_adj"].to_numpy(), holm(block["p_value"].to_numpy())), level


def test_the_all_pairs_basis_goes_beyond_the_table(crossed):
    """The switch back to every pair: a g1-versus-g2 comparison is the difference of two
    interaction coefficients, which is a real contrast and no table row."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    table = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction_all", adjust="none"
    )
    for _level, block in table.groupby("by"):
        assert len(block) == 6

    row = _pair(table, "RICA", "g1", "g2")
    expected = (
        fitted.params["gr[T.g2]:territory[T.RICA]"]
        - fitted.params["gr[T.g1]:territory[T.RICA]"]
    )
    estimate = row["estimate"] if row["a"] == "g2" else -row["estimate"]
    assert estimate == pytest.approx(float(expected), abs=1e-9)


def test_the_two_interaction_bases_agree_where_they_overlap(crossed):
    """The restriction removes comparisons; it must not change the ones that remain."""
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    restricted = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction", adjust="none"
    )
    everything = pairwise_contrasts(
        fitted, frame, factor="gr", by="territory", basis="interaction_all", adjust="none"
    )
    merged = restricted.merge(everything, on=["by", "a", "b"], suffixes=("_r", "_a"))
    assert len(merged) == len(restricted)
    assert np.allclose(merged["estimate_r"], merged["estimate_a"])
    assert np.allclose(merged["p_value_r"], merged["p_value_a"])


def test_the_reference_series_is_blank_in_both_interaction_bases(crossed):
    from nvitk.stats.pairwise import pairwise_contrasts

    frame, fitted = crossed
    for basis in ("interaction", "interaction_all"):
        table = pairwise_contrasts(fitted, frame, factor="gr", by="territory", basis=basis)
        assert "BASILAR" not in set(table["by"]), basis
