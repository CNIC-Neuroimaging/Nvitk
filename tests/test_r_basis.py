"""
Marginal means and contrasts for R models ``emmeans`` refuses.

Two reported failures are pinned here:

* ``'pairs' is not an exported object from 'namespace:emmeans'`` — ``pairs`` is an S3 method on
  ``emmGrid``, not an exported function, so ``emmeans::pairs`` cannot be called that way.
* ``Can't handle an object of class "lmrob"`` — ``emmeans`` registers no basis for ``robustbase``,
  which took the confidence band and the significance brackets down with it.

The fallback's correctness claim is that it reproduces ``emmeans``, so it is checked against
``emmeans`` on a model ``emmeans`` does accept rather than against recorded numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nvitk.stats.pairwise import pairwise_contrasts

pytest.importorskip("rpy2")

from nvitk.stats.r_mmrm import emmeans_available, mmrm_backend_status  # noqa: E402
from nvitk.stats.r_mixedlm import r_backend_status  # noqa: E402
from nvitk.stats.r_robust import robust_backend_status  # noqa: E402

needs_emmeans = pytest.mark.skipif(
    not emmeans_available(), reason="needs the R package emmeans"
)
needs_lmrob = pytest.mark.skipif(
    not robust_backend_status().available, reason="needs R + robustbase"
)
needs_mmrm = pytest.mark.skipif(
    not mmrm_backend_status().available, reason="needs R + mmrm"
)

LEVELS = ["LICA", "RICA", "BASI"]
#: LICA and RICA share a mean, so "these must not differ" is true by construction.
MEANS = {"LICA": 10.0, "RICA": 10.0, "BASI": 6.0}


@pytest.fixture(scope="module")
def frame():
    """A three-territory cohort with a covariate and a second factor."""
    rng = np.random.default_rng(5)
    n = 300
    out = pd.DataFrame(
        {
            "subject_uid": [f"s{i % 40:02d}" for i in range(n)],
            "territory": rng.choice(LEVELS, n),
            "sex": rng.choice(["M", "F"], n),
            "age_c": rng.normal(0.0, 1.0, n),
        }
    )
    out["log1p_pi"] = (
        [MEANS[t] for t in out["territory"]]
        + 0.6 * out["age_c"]
        + 0.4 * (out["sex"] == "M")
        + rng.normal(0, 1.0, n)
    )
    return out


@pytest.fixture(scope="module")
def lmrob_fit(frame):
    """A robust fit with an interaction — the model emmeans declines."""
    from nvitk.stats.r_robust import fit_lmrob

    fit, _frame, _meta = fit_lmrob(
        data=frame, formula="log1p_pi ~ territory * sex + age_c"
    )
    return fit


def _separated(contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split comparisons into the ones involving BASI and the ones that do not."""
    touches = (contrasts["a"] == "BASI") | (contrasts["b"] == "BASI")
    return contrasts.loc[touches], contrasts.loc[~touches]


# ──────────────────────────────────────────────────────────────────────────────
# The fallback reproduces emmeans
# ──────────────────────────────────────────────────────────────────────────────
@needs_emmeans
def test_basis_matches_emmeans_where_both_work(frame):
    """An ``lm`` is a model emmeans accepts, so the two can be compared directly — including
    emmeans' equal-weight averaging over the factor outside the specification."""
    from rpy2.robjects import default_converter, globalenv, pandas2ri
    from rpy2.robjects import r as R_
    from rpy2.robjects.conversion import localconverter

    from nvitk.stats.r_basis import linear_emmeans

    with localconverter(default_converter + pandas2ri.converter):
        globalenv["nvitk_test_df"] = frame
    fit = R_("lm(log1p_pi ~ territory * sex + age_c, data = nvitk_test_df)")

    mine = linear_emmeans(fit, "~ territory", ci_level=0.95).sort_values("territory")
    # Refitting inside the R expression rather than round-tripping the fitted object: emmeans
    # re-evaluates the model call, and a handle passed back through rpy2 loses its environment.
    with localconverter(default_converter + pandas2ri.converter):
        theirs = pd.DataFrame(
            R_(
                "as.data.frame(summary(emmeans::emmeans("
                "lm(log1p_pi ~ territory * sex + age_c, data = nvitk_test_df), ~territory),"
                " level = 0.95, infer = c(TRUE, FALSE)))"
            )
        )
    theirs = theirs.sort_values("territory")

    for column in ("emmean", "SE", "lower.CL", "upper.CL"):
        assert np.allclose(
            mine[column].to_numpy(float), theirs[column].to_numpy(float), atol=1e-9
        ), column


@needs_emmeans
def test_basis_contrasts_match_emmeans_where_both_work(frame):
    """The same check for the pairwise comparisons."""
    from rpy2.robjects import default_converter, globalenv, pandas2ri
    from rpy2.robjects import r as R_
    from rpy2.robjects.conversion import localconverter

    from nvitk.stats.r_basis import linear_pairs

    with localconverter(default_converter + pandas2ri.converter):
        globalenv["nvitk_test_df"] = frame
    fit = R_("lm(log1p_pi ~ territory * sex + age_c, data = nvitk_test_df)")

    mine = linear_pairs(fit, "territory").set_index("contrast").sort_index()
    with localconverter(default_converter + pandas2ri.converter):
        theirs = pd.DataFrame(
            R_(
                "as.data.frame(summary(emmeans::contrast(emmeans::emmeans("
                "lm(log1p_pi ~ territory * sex + age_c, data = nvitk_test_df), ~territory),"
                ' method = "pairwise", adjust = "none")))'
            )
        ).set_index("contrast").sort_index()

    assert list(mine.index) == list(theirs.index)
    for column in ("estimate", "SE", "p.value"):
        assert np.allclose(
            mine[column].to_numpy(float), theirs[column].to_numpy(float), atol=1e-9
        ), column


# ──────────────────────────────────────────────────────────────────────────────
# lmrob — the model emmeans declines
# ──────────────────────────────────────────────────────────────────────────────
@needs_lmrob
def test_lmrob_gets_marginal_means_and_intervals(lmrob_fit):
    """Reported as 'Can't handle an object of class lmrob', which took the 95% CI with it."""
    from nvitk.stats.r_robust import lmrob_emmeans

    out = lmrob_emmeans(lmrob_fit, "~ territory", ci_level=0.95)
    assert {"emmean", "SE", "df", "lower.CL", "upper.CL"} <= set(out.columns)
    assert len(out) == len(LEVELS)
    assert (out["SE"] > 0).all()
    assert (out["lower.CL"] < out["emmean"]).all()
    assert (out["upper.CL"] > out["emmean"]).all()


@needs_lmrob
def test_lmrob_gets_a_band_over_a_continuous_grid(lmrob_fit):
    """The confidence band asks for a marginal mean at each x, which is the same path."""
    from nvitk.stats.r_robust import lmrob_emmeans

    grid = np.linspace(-2.0, 2.0, 5)
    out = lmrob_emmeans(
        lmrob_fit, "~ age_c | territory", at_name="age_c", at_values=grid, ci_level=0.95
    )
    assert len(out) == len(grid) * len(LEVELS)
    assert (out["SE"] > 0).all()
    # A prediction interval is narrowest near the covariate's mean and widens away from it.
    per_x = out.groupby("age_c")["SE"].mean()
    assert per_x.loc[0.0] < per_x.loc[2.0]
    assert per_x.loc[0.0] < per_x.loc[-2.0]


@needs_lmrob
def test_lmrob_gets_pairwise_contrasts(lmrob_fit, frame):
    """The significance brackets, on the engine that could not have them."""
    out = pairwise_contrasts(lmrob_fit, frame, factor="territory", levels=LEVELS)
    assert len(out) == 3
    assert (out["se"] > 0).all()
    separated, same = _separated(out)
    assert (separated["p_adj"] < 0.05).all(), separated
    assert (same["p_adj"] > 0.05).all(), same


# ──────────────────────────────────────────────────────────────────────────────
# mmrm — the model emmeans accepts, once it is called correctly
# ──────────────────────────────────────────────────────────────────────────────
@needs_mmrm
@needs_emmeans
def test_mmrm_pairwise_contrasts(frame):
    """Reported as "'pairs' is not an exported object from 'namespace:emmeans'"."""
    from nvitk.stats.r_mmrm import fit_mmrm

    repeated = frame.copy()
    repeated["territory"] = pd.Categorical(repeated["territory"], categories=LEVELS)
    # One row per subject × territory, which is what an unstructured covariance needs.
    repeated = repeated.drop_duplicates(subset=["subject_uid", "territory"])

    fit, _frame, _meta = fit_mmrm(
        data=repeated,
        formula="log1p_pi ~ territory + age_c + sex + us(territory | subject_uid)",
        visit="territory",
        subject="subject_uid",
        structure="us",
    )
    out = pairwise_contrasts(fit, repeated, factor="territory", levels=LEVELS)
    assert len(out) == 3
    assert (out["se"] > 0).all() and out["p_value"].notna().all()
    # Satterthwaite degrees of freedom, not the residual count.
    assert out["df"].notna().all() and (out["df"] > 0).all()
    separated, same = _separated(out)
    assert (separated["p_adj"] < 0.05).all(), separated
    assert (same["p_adj"] > 0.05).all(), same


# ──────────────────────────────────────────────────────────────────────────────
# lme4 — the wrapper rpy2 cannot convert
# ──────────────────────────────────────────────────────────────────────────────
needs_lme4 = pytest.mark.skipif(
    not r_backend_status().available, reason="needs R + pymer4 + lme4"
)


def test_pymer4_is_unwrapped_to_its_r_object():
    """Reported as "Conversion 'py2rpy' not defined for objects of type ... lmer": the pymer4
    wrapper was handed to rpy2 instead of the R model it holds."""
    from nvitk.stats.pairwise import _r_object

    class FakeLmer:
        """Stands in for pymer4's model: the R object lives on one of two names."""

        def __init__(self, **attrs):
            for name, value in attrs.items():
                setattr(self, name, value)

    sentinel = object()
    assert _r_object(FakeLmer(r_model=sentinel)) is sentinel
    assert _r_object(FakeLmer(model_obj=sentinel)) is sentinel
    # ``model`` is not one of them — guessing it is what produced the reported error.
    wrapper = FakeLmer(model=sentinel)
    assert _r_object(wrapper) is wrapper


@pytest.mark.slow
@needs_lme4
@needs_emmeans
def test_lme4_contrasts_go_through_emmeans(frame):
    """End to end on a real pymer4 fit: emmeans should handle it, not the fallback."""
    from nvitk.stats.r_mixedlm import fit_lme4

    model, _f, _meta = fit_lme4(
        data=frame,
        formula="log1p_pi ~ territory + age_c + sex + (1 | subject_uid)",
    )
    out = pairwise_contrasts(model, frame, factor="territory", levels=LEVELS)
    assert len(out) == 3
    assert (out["se"] > 0).all() and out["p_value"].notna().all()
    separated, same = _separated(out)
    assert (separated["p_adj"] < 0.05).all(), separated
    assert (same["p_adj"] > 0.05).all(), same


# ──────────────────────────────────────────────────────────────────────────────
# Comparisons within a series, through the R engines
# ──────────────────────────────────────────────────────────────────────────────
@needs_lmrob
def test_lmrob_by_groups_through_the_design_matrix(frame):
    """The fallback has to carry ``by`` too, or a robust fit loses per-series brackets."""
    from nvitk.stats.r_robust import fit_lmrob

    fit, _f, _m = fit_lmrob(data=frame, formula="log1p_pi ~ territory * sex + age_c")
    out = pairwise_contrasts(fit, frame, factor="territory", by="sex", levels=LEVELS)
    assert set(out["by"]) == {"M", "F"}
    assert len(out) == 2 * 3
    assert (out["se"] > 0).all()
    for _group, block in out.groupby("by"):
        separated, _same = _separated(block)
        assert (separated["p_adj"] < 0.05).all(), block
