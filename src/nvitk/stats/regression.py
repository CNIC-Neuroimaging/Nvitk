"""
Single-level regression models — OLS, GLM, and non-linear least squares.

Description
-----------
Companions to :mod:`nvitk.stats.mixedlm` for data that does not need (or cannot support) a random
structure. Three engines, deliberately sharing as much of the surrounding machinery as possible:

``ols``
    Ordinary least squares over a patsy formula. The plain linear model.
``glm``
    Generalized linear model: a non-linear *link* between the predictors and the response
    (logistic for binary outcomes, Poisson/negative-binomial for counts, Gamma or inverse-Gaussian
    with a log link for skewed positive measures like flow or plaque volume). Still linear in its
    parameters, so the formula, the coefficient table and the marginal-means plot all carry over.
``nonlinear``
    True non-linear least squares: an explicit parametric curve (exponential decay, sigmoid, power
    law, …) fitted by iteration. This one does *not* share the formula workflow — it fits one
    response against one predictor — so it has its own result shape and plot.

Curvature *in the predictor* (splines, polynomials) is not an engine: those are patsy terms
(:data:`SPLINE_TERMS`) that work inside ``ols``, ``glm`` and MixedLM alike, because such a model is
still linear in its parameters.

Result conventions
------------------
``fit_ols`` and ``fit_glm`` return ``(result, model_df, metadata)`` exactly like
:func:`~nvitk.stats.mixedlm.fit_or_load_mixedlm`, so callers can treat all three interchangeably.
:func:`model_info_dict` renders any of them — including a MixedLM — into the structure
:func:`~nvitk.stats.mixedlm.render_mixedlm_info` and the GUI report panel already understand.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

from .mixedlm import (
    _coerce_object_numerics,
    _safe_col,
    formula_columns,
    mixedlm_coef_frame,
    mixedlm_info_dict,
    significance_stars,
)

log = Logger()

ANALYSIS_OLS = "ols"
ANALYSIS_GLM = "glm"
ANALYSIS_NONLINEAR = "nonlinear"


# ──────────────────────────────────────────────────────────────────────────────
# GLM families
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GlmFamily:
    """One GLM family, its available links, and when to reach for it."""

    key: str
    label: str
    links: tuple[str, ...]
    default_link: str
    description: str


GLM_FAMILIES: dict[str, GlmFamily] = {
    "gaussian": GlmFamily(
        "gaussian", "Gaussian (normal)", ("identity", "log", "inverse_power"), "identity",
        "Continuous, roughly symmetric outcome. With an identity link this is ordinary least "
        "squares; with a log link it models a multiplicative effect on the mean.",
    ),
    "binomial": GlmFamily(
        "binomial", "Binomial (logistic)", ("logit", "probit", "cloglog"), "logit",
        "Binary or proportion outcome. Coefficients on the logit link are log-odds; exponentiate "
        "them for odds ratios.",
    ),
    "poisson": GlmFamily(
        "poisson", "Poisson (counts)", ("log", "identity", "sqrt"), "log",
        "Counts or rates. Assumes the variance equals the mean — if the data are overdispersed, "
        "prefer the negative binomial.",
    ),
    "negativebinomial": GlmFamily(
        "negativebinomial", "Negative binomial (overdispersed counts)", ("log", "identity"), "log",
        "Counts whose variance exceeds their mean.",
    ),
    "gamma": GlmFamily(
        "gamma", "Gamma (skewed positive)", ("log", "inverse_power", "identity"), "log",
        "Strictly positive, right-skewed measures with roughly constant coefficient of variation — "
        "flow, volume, plaque burden. A log link makes the covariate effects multiplicative.",
    ),
    "inversegaussian": GlmFamily(
        "inversegaussian", "Inverse Gaussian (strongly skewed positive)", ("log", "inverse_squared", "identity"), "log",
        "Positive outcomes with heavier right skew than the Gamma handles well.",
    ),
}

_LINK_LABELS: dict[str, str] = {
    "identity": "identity",
    "log": "log",
    "logit": "logit",
    "probit": "probit",
    "cloglog": "complementary log-log",
    "sqrt": "square root",
    "inverse_power": "inverse (1/µ)",
    "inverse_squared": "inverse squared (1/µ²)",
}


def _build_family(family: str, link: str | None):
    """Instantiate a ``statsmodels`` family object for *family* / *link*."""
    import statsmodels.api as sm

    spec = GLM_FAMILIES.get(str(family).strip().lower())
    if spec is None:
        raise ValueError(
            f"Unknown GLM family {family!r}. Available: {', '.join(sorted(GLM_FAMILIES))}."
        )
    link_name = str(link or spec.default_link).strip().lower()
    if link_name not in spec.links:
        raise ValueError(
            f"Link {link_name!r} is not available for the {spec.label} family "
            f"(use one of {', '.join(spec.links)})."
        )
    link_obj = {
        "identity": sm.families.links.Identity,
        "log": sm.families.links.Log,
        "logit": sm.families.links.Logit,
        "probit": sm.families.links.Probit,
        "cloglog": sm.families.links.CLogLog,
        "sqrt": sm.families.links.Sqrt,
        "inverse_power": sm.families.links.InversePower,
        "inverse_squared": sm.families.links.InverseSquared,
    }[link_name]()
    family_obj = {
        "gaussian": sm.families.Gaussian,
        "binomial": sm.families.Binomial,
        "poisson": sm.families.Poisson,
        "negativebinomial": sm.families.NegativeBinomial,
        "gamma": sm.families.Gamma,
        "inversegaussian": sm.families.InverseGaussian,
    }[spec.key]
    return family_obj(link=link_obj), spec, link_name


# ──────────────────────────────────────────────────────────────────────────────
# Spline / polynomial terms
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SplineTerm:
    """A patsy term that lets a linear model bend."""

    key: str
    label: str
    template: str
    description: str


SPLINE_TERMS: tuple[SplineTerm, ...] = (
    SplineTerm(
        "bs", "B-spline (df)", "bs({col}, df={df})",
        "Piecewise-cubic spline with *df* degrees of freedom. The general-purpose choice for a "
        "smooth, possibly non-monotonic effect. df=3–5 is usually enough; more will chase noise.",
    ),
    SplineTerm(
        "cr", "Natural cubic spline (df)", "cr({col}, df={df})",
        "Cubic spline constrained to be linear beyond the outermost knots, which keeps the fit from "
        "flaring at the edges of the data.",
    ),
    SplineTerm(
        "poly2", "Quadratic", "{col} + I({col} ** 2)",
        "A single bend. Interpretable, but the fit at one end of the range is tied to the other.",
    ),
    SplineTerm(
        "poly3", "Cubic", "{col} + I({col} ** 2) + I({col} ** 3)",
        "Two bends. Beyond this, prefer a spline — high-order polynomials oscillate badly near the "
        "edges of the data.",
    ),
    SplineTerm(
        "log", "Log transform", "np.log({col})",
        "Diminishing returns: a fixed proportional change in the predictor shifts the outcome by a "
        "constant. Requires strictly positive values.",
    ),
)


def spline_term(kind: str, column: str, *, df: int = 4) -> str:
    """Render a :data:`SPLINE_TERMS` template, e.g. ``("bs", "age_c", df=4) -> "bs(age_c, df=4)"``."""
    for term in SPLINE_TERMS:
        if term.key == kind:
            return term.template.format(col=column, df=int(df))
    raise ValueError(f"Unknown spline term {kind!r}.")


# ──────────────────────────────────────────────────────────────────────────────
# Shared frame preparation
# ──────────────────────────────────────────────────────────────────────────────
def prepare_model_frame(
    data: pd.DataFrame,
    formula: str,
    *,
    required_columns: Sequence[str] | None = None,
    dropna_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Sanitize column names, coerce object-numerics and drop incomplete rows.

    The same preparation :func:`~nvitk.stats.mixedlm.fit_or_load_mixedlm` performs, factored out so
    every engine reports row counts the same way and the GUI's "dropped N incomplete rows" note
    means the same thing whichever model was fitted.
    """
    import re as _re

    df = data.copy()
    df.columns = [_safe_col(c) for c in df.columns]

    formula_cols = formula_columns(df.columns, formula)
    categorical_forced = set(_re.findall(r"C\(\s*([A-Za-z_][A-Za-z0-9_]*)", str(formula or "")))
    df = _coerce_object_numerics(df, [c for c in formula_cols if c not in categorical_forced])

    req = [c for c in (required_columns or []) if c]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    na_cols = list(dropna_columns) if dropna_columns is not None else sorted(set(req) | set(formula_cols))
    na_cols = [c for c in na_cols if c in df.columns]

    n_input = int(len(df))
    dropped_by_column: dict[str, int] = {}
    if na_cols:
        dropped_by_column = {c: int(df[c].isna().sum()) for c in na_cols if bool(df[c].isna().any())}
        df = df.dropna(subset=na_cols).reset_index(drop=True)
    if not len(df):
        detail = ", ".join(f"{c} ({n} missing)" for c, n in sorted(dropped_by_column.items()))
        raise ValueError(
            f"No complete rows left after dropping missing values in {na_cols} "
            f"(started from {n_input} rows)." + (f" Columns with missing values: {detail}." if detail else "")
        )

    meta = {
        "formula": formula,
        "n_rows_input": n_input,
        "n_rows": int(len(df)),
        "n_rows_dropped": n_input - int(len(df)),
        "dropna_columns": na_cols,
        "dropped_by_column": dropped_by_column,
        "loaded": False,
    }
    return df, meta


# ──────────────────────────────────────────────────────────────────────────────
# OLS and GLM
# ──────────────────────────────────────────────────────────────────────────────
def fit_ols(
    *,
    data: pd.DataFrame,
    formula: str,
    required_columns: Sequence[str] | None = None,
    dropna_columns: Sequence[str] | None = None,
    robust: str | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """
    Fit an ordinary least squares model.

    Parameters
    ----------
    robust : str, optional
        Heteroscedasticity-consistent covariance type (``"HC0"`` … ``"HC3"``). Worth setting when
        the residual spread grows with the fitted value; it changes the standard errors, not the
        coefficients.

    Returns
    -------
    (result, model_df, metadata)
        Shaped like :func:`~nvitk.stats.mixedlm.fit_or_load_mixedlm` so callers are interchangeable.
    """
    import statsmodels.formula.api as smf

    df, meta = prepare_model_frame(
        data, formula, required_columns=required_columns, dropna_columns=dropna_columns
    )
    kwargs = dict(fit_kwargs or {})
    if robust:
        kwargs["cov_type"] = robust
    result = smf.ols(formula, data=df).fit(**kwargs)
    meta.update({"engine": ANALYSIS_OLS, "robust": robust})
    return result, df, meta


def fit_glm(
    *,
    data: pd.DataFrame,
    formula: str,
    family: str = "gaussian",
    link: str | None = None,
    required_columns: Sequence[str] | None = None,
    dropna_columns: Sequence[str] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """
    Fit a generalized linear model.

    The link is where the non-linearity lives: predictors combine linearly on the link scale, and
    the fitted values are mapped back through its inverse. Coefficients are therefore effects *on
    the link scale* — log-odds for a logit link, log-ratios for a log link.
    """
    import statsmodels.formula.api as smf

    df, meta = prepare_model_frame(
        data, formula, required_columns=required_columns, dropna_columns=dropna_columns
    )
    family_obj, spec, link_name = _build_family(family, link)
    result = smf.glm(formula, data=df, family=family_obj).fit(**dict(fit_kwargs or {}))
    meta.update(
        {
            "engine": ANALYSIS_GLM,
            "family": spec.key,
            "family_label": spec.label,
            "link": link_name,
            "link_label": _LINK_LABELS.get(link_name, link_name),
        }
    )
    return result, df, meta


# ──────────────────────────────────────────────────────────────────────────────
# Non-linear least squares
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NonlinearModel:
    """A parametric curve fitted by iterative least squares."""

    key: str
    label: str
    expression: str
    params: tuple[str, ...]
    func: Callable[..., np.ndarray]
    initial: Callable[[np.ndarray, np.ndarray], list[float]]
    description: str


def _p0_exp_decay(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Amplitude from the observed range, rate from the x span, offset from the tail."""
    span = float(np.ptp(x)) or 1.0
    return [float(np.ptp(y)) or 1.0, 3.0 / span, float(np.min(y))]


NONLINEAR_MODELS: dict[str, NonlinearModel] = {
    "exp_decay": NonlinearModel(
        "exp_decay", "Exponential decay", "y = a·exp(−b·x) + c", ("a", "b", "c"),
        lambda x, a, b, c: a * np.exp(-b * x) + c,
        _p0_exp_decay,
        "Falls fast then flattens towards an asymptote c.",
    ),
    "exp_saturation": NonlinearModel(
        "exp_saturation", "Saturating exponential", "y = a·(1 − exp(−b·x)) + c", ("a", "b", "c"),
        lambda x, a, b, c: a * (1.0 - np.exp(-b * x)) + c,
        _p0_exp_decay,
        "Rises quickly then plateaus at a + c.",
    ),
    "logistic": NonlinearModel(
        "logistic", "Logistic / sigmoid", "y = L / (1 + exp(−k·(x − x₀))) + c", ("L", "k", "x0", "c"),
        lambda x, L, k, x0, c: L / (1.0 + np.exp(-k * (x - x0))) + c,
        lambda x, y: [float(np.ptp(y)) or 1.0, 1.0, float(np.median(x)), float(np.min(y))],
        "An S-curve between two plateaus, steepest at x₀.",
    ),
    "power": NonlinearModel(
        "power", "Power law", "y = a·x^b + c", ("a", "b", "c"),
        lambda x, a, b, c: a * np.power(np.clip(x, 1e-12, None), b) + c,
        lambda x, y: [float(np.ptp(y)) or 1.0, 1.0, float(np.min(y))],
        "Scale-free growth or decay. Needs positive x.",
    ),
    "michaelis_menten": NonlinearModel(
        "michaelis_menten", "Michaelis–Menten", "y = Vmax·x / (Km + x)", ("Vmax", "Km"),
        lambda x, Vmax, Km: Vmax * x / (Km + x),
        lambda x, y: [float(np.max(y)) or 1.0, float(np.median(x)) or 1.0],
        "Saturating uptake: rises to Vmax, half-maximal at Km.",
    ),
    "quadratic_peak": NonlinearModel(
        "quadratic_peak", "Quadratic with a peak", "y = a·(x − x₀)² + c", ("a", "x0", "c"),
        lambda x, a, x0, c: a * (x - x0) ** 2 + c,
        lambda x, y: [-1.0, float(np.median(x)), float(np.max(y))],
        "A single turning point at x₀ — use when you want to estimate *where* the optimum is.",
    ),
}


def fit_nonlinear(
    *,
    data: pd.DataFrame,
    x: str,
    y: str,
    model: str = "exp_decay",
    p0: Sequence[float] | None = None,
    maxfev: int = 20000,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """
    Fit an explicit parametric curve of *y* against *x* by non-linear least squares.

    Unlike the formula-driven engines this fits one predictor against one response, with no
    covariate adjustment and no grouping — the parameters are the model.

    Returns
    -------
    (result, model_df, metadata)
        *result* is a dict with ``params`` (a coefficient frame), ``predict`` (a callable),
        ``cov``, ``r_squared``, ``rmse``, ``n_obs`` and the model spec — not a statsmodels object,
        so it is handled separately by the reporting and plotting helpers.
    """
    from scipy.optimize import curve_fit
    from scipy.stats import norm

    spec = NONLINEAR_MODELS.get(str(model).strip().lower())
    if spec is None:
        raise ValueError(
            f"Unknown non-linear model {model!r}. Available: {', '.join(sorted(NONLINEAR_MODELS))}."
        )

    frame = data[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) <= len(spec.params):
        raise ValueError(
            f"{spec.label} has {len(spec.params)} parameters but only {len(frame)} complete "
            "observations — the fit is not identifiable."
        )
    xv = frame[x].to_numpy(dtype=float)
    yv = frame[y].to_numpy(dtype=float)

    start = list(p0) if p0 is not None else spec.initial(xv, yv)
    if len(start) != len(spec.params):
        raise ValueError(f"{spec.label} needs {len(spec.params)} starting values, got {len(start)}.")

    try:
        popt, pcov = curve_fit(spec.func, xv, yv, p0=start, maxfev=int(maxfev))
    except RuntimeError as exc:
        raise ValueError(
            f"{spec.label} did not converge ({exc}). Try different starting values, or a simpler "
            "curve."
        ) from exc

    fitted = spec.func(xv, *popt)
    residuals = yv - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    se = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, popt / se, np.nan)
    pvals = 2.0 * (1.0 - norm.cdf(np.abs(z)))

    params = pd.DataFrame(
        {
            "parameter": list(spec.params),
            "coef": popt,
            "std_err": se,
            "z": z,
            "p_value": pvals,
            "ci_low": popt - 1.959963985 * se,
            "ci_high": popt + 1.959963985 * se,
            "sig": [significance_stars(float(p)) for p in pvals],
        }
    )

    result = {
        "engine": ANALYSIS_NONLINEAR,
        "spec": spec,
        "popt": popt,
        "cov": pcov,
        "params": params,
        "predict": lambda grid, _f=spec.func, _p=popt: _f(np.asarray(grid, dtype=float), *_p),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rmse": float(np.sqrt(ss_res / len(frame))),
        "n_obs": int(len(frame)),
        "x": x,
        "y": y,
    }
    meta = {
        "engine": ANALYSIS_NONLINEAR,
        "formula": f"{y} ~ {spec.expression}",
        "model": spec.key,
        "n_rows_input": int(len(data)),
        "n_rows": int(len(frame)),
        "n_rows_dropped": int(len(data)) - int(len(frame)),
        "dropped_by_column": {},
        "dropna_columns": [x, y],
        "loaded": False,
    }
    log.info("%s fit: R²=%.4f RMSE=%.4g on n=%d.", spec.label, result["r_squared"], result["rmse"], len(frame))
    return result, frame, meta


def nonlinear_confidence_band(
    result: Mapping[str, Any],
    grid: np.ndarray,
    *,
    ci_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fitted curve and its confidence band over *grid*, by the delta method.

    The gradient of the curve with respect to each parameter is taken numerically, then
    ``se(x) = √(gᵀ Σ g)`` propagates the parameter covariance to the prediction.

    Returns
    -------
    (fitted, lower, upper)
    """
    from scipy.stats import norm

    spec: NonlinearModel = result["spec"]
    popt = np.asarray(result["popt"], dtype=float)
    cov = result.get("cov")
    grid = np.asarray(grid, dtype=float)
    fitted = spec.func(grid, *popt)

    if cov is None or not np.all(np.isfinite(cov)):
        return fitted, fitted, fitted

    # Numeric Jacobian: one column per parameter, stepped relative to its own scale.
    jac = np.empty((grid.size, popt.size), dtype=float)
    for i in range(popt.size):
        step = 1e-6 * max(abs(popt[i]), 1.0)
        up, down = popt.copy(), popt.copy()
        up[i] += step
        down[i] -= step
        jac[:, i] = (spec.func(grid, *up) - spec.func(grid, *down)) / (2.0 * step)

    var = np.einsum("ij,jk,ik->i", jac, np.asarray(cov, dtype=float), jac)
    se = np.sqrt(np.clip(var, 0.0, None))
    crit = float(norm.ppf(1.0 - (1.0 - float(ci_level)) / 2.0))
    return fitted, fitted - crit * se, fitted + crit * se


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
def _statsmodels_info_dict(result: Any, *, outcome_name: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Report structure for an OLS / GLM result, matching the MixedLM one."""
    header: dict[str, Any] = {
        "formula": getattr(getattr(result, "model", None), "formula", None),
        "n_obs": int(getattr(result, "nobs", 0)),
        "n_groups": None,
        "group_name": "Group",
        "outcome_name": outcome_name,
        "vc_group_name": "Variance Components",
    }
    if meta.get("engine") == ANALYSIS_GLM:
        header["family"] = meta.get("family_label")
        header["link"] = meta.get("link_label")

    fit_stats: dict[str, Any] = {}
    for attr in ("llf", "aic", "bic"):
        value = getattr(result, attr, None)
        if value is not None and np.isscalar(value):
            fit_stats[attr] = float(value)
    for attr, key in (("rsquared", "r_squared"), ("rsquared_adj", "r_squared_adj")):
        value = getattr(result, attr, None)
        if value is not None:
            fit_stats[key] = float(value)
    deviance = getattr(result, "deviance", None)
    if deviance is not None:
        fit_stats["deviance"] = float(deviance)
        null_dev = getattr(result, "null_deviance", None)
        if null_dev:
            # Explained deviance — the GLM analogue of R².
            fit_stats["pseudo_r_squared"] = float(1.0 - deviance / null_dev)
    scale = getattr(result, "scale", None)
    if scale is not None and np.isscalar(scale):
        fit_stats["scale"] = float(scale)
        fit_stats["resid_sd"] = float(np.sqrt(max(float(scale), 0.0)))
    fit_stats["converged"] = bool(getattr(getattr(result, "mle_retvals", {}) or {}, "get", lambda *_: True)("converged", True))

    return {
        "header": header,
        "fixed_effects": mixedlm_coef_frame(result),
        "random_effects": pd.DataFrame(columns=["component", "kind", "var", "sd"]),
        "cov_re": pd.DataFrame(),
        "fit_statistics": fit_stats,
        "has_vcomp": False,
    }


def model_info_dict(
    result: Any,
    *,
    outcome_name: str = "Outcome",
    group_name: str = "Group",
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Structured report for **any** engine — MixedLM, OLS, GLM or non-linear.

    Dispatches on the result type so the GUI report panel and
    :func:`~nvitk.stats.mixedlm.render_mixedlm_info` need to know nothing about which model was run.
    """
    meta = dict(meta or {})
    engine = str(meta.get("engine") or "")

    if engine == "lme4":
        from .r_mixedlm import lme4_info_dict

        return lme4_info_dict(result, outcome_name=outcome_name, meta=meta)

    if engine == "mmrm":
        from .r_mmrm import mmrm_info_dict

        return mmrm_info_dict(result, outcome_name=outcome_name, meta=meta)

    if engine == "sem":
        from .sem import sem_info_dict

        return sem_info_dict(result, outcome_name=outcome_name, group_name=group_name, meta=meta)

    if engine == "mrf":
        from .r_gam import mrf_info_dict

        return mrf_info_dict(result, outcome_name=outcome_name, group_name=group_name, meta=meta)

    if engine == "lmrob":
        from .r_robust import lmrob_info_dict

        return lmrob_info_dict(
            result, outcome_name=outcome_name, group_name=group_name, meta=meta
        )

    if isinstance(result, dict) and result.get("engine") == ANALYSIS_NONLINEAR:
        spec: NonlinearModel = result["spec"]
        return {
            "header": {
                "formula": f"{result['y']} ~ {spec.expression}",
                "n_obs": int(result["n_obs"]),
                "n_groups": None,
                "group_name": group_name,
                "outcome_name": result["y"],
                "vc_group_name": "Variance Components",
                "model": spec.label,
            },
            "fixed_effects": result["params"],
            "random_effects": pd.DataFrame(columns=["component", "kind", "var", "sd"]),
            "cov_re": pd.DataFrame(),
            "fit_statistics": {
                "r_squared": float(result["r_squared"]),
                "rmse": float(result["rmse"]),
                "converged": True,
            },
            "has_vcomp": False,
        }

    if engine in {ANALYSIS_OLS, ANALYSIS_GLM} or not hasattr(result, "fe_params"):
        return _statsmodels_info_dict(result, outcome_name=outcome_name, meta=meta)

    return mixedlm_info_dict(result, outcome_name=outcome_name, group_name=group_name)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def plot_nonlinear_fit(
    result: Mapping[str, Any],
    data: pd.DataFrame,
    *,
    errorbar: bool = False,
    ci_level: float = 0.95,
    include_points: bool = True,
    group: str | None = None,
    group_order: Sequence[str] | None = None,
    palette: str = "tab10",
    display: str = "overview",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> Any:
    """
    Scatter of the observations with the fitted curve, optionally with a confidence band.

    The curve is a single fit over all rows: *group* only colours the scatter, it does not fit one
    curve per group. Fitting per group would mean one non-linear fit per level, which the parameter
    table has no room to report honestly.

    Parameters
    ----------
    display : {"overview", "grouped"}
        ``"grouped"`` splits *group*'s levels into a grid of anatomical panels (see
        :mod:`nvitk.stats.region_groups`), each autoscaled to its own range. The fitted curve is the
        same single fit in every panel — which is exactly what makes the panels comparable: each one
        shows how its own regions sit against the common curve.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    spec: NonlinearModel = result["spec"]
    x, y = result["x"], result["y"]
    grouped = bool(group) and group in data.columns
    levels = (
        (list(group_order) if group_order else sorted(data[group].dropna().astype(str).unique()))
        if grouped else []
    )

    # One curve for the whole fit, evaluated over the full observed range so every panel shares it.
    grid = np.linspace(float(np.nanmin(data[x])), float(np.nanmax(data[x])), 300)
    fitted, lower, upper = nonlinear_confidence_band(result, grid, ci_level=ci_level)

    def draw_panel(ax: Any, panel_data: pd.DataFrame, panel_levels: Sequence[str], panel_title: str) -> None:
        """Scatter *panel_data* under the shared fitted curve."""
        if include_points:
            if panel_levels:
                colors = sns.color_palette(palette, n_colors=max(len(panel_levels), 3))
                for i, level in enumerate(panel_levels):
                    subset = panel_data.loc[panel_data[group].astype(str) == str(level)]
                    ax.scatter(subset[x], subset[y], s=18, alpha=0.45,
                               color=colors[i % len(colors)], label=str(level))
            else:
                ax.scatter(panel_data[x], panel_data[y], s=18, alpha=0.4, color="#4C72B0")

        if errorbar and not np.allclose(lower, upper):
            ax.fill_between(
                grid, lower, upper, color="black", alpha=0.12,
                label=f"{int(round(ci_level * 100))}% CI",
            )
        ax.plot(grid, fitted, color="black", lw=2.6, label=f"{spec.label}: {spec.expression}")

        ax.set_title(panel_title)
        ax.set_xlabel(x_label or x)
        ax.set_ylabel(y_label or y)
        ax.grid(True, axis="y", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            dedup: dict[str, Any] = {}
            for handle, label in zip(handles, labels):
                dedup.setdefault(label, handle)
            ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=9)

    display = str(display or "overview").strip().lower()
    if display not in {"overview", "grouped"}:
        raise ValueError("display must be one of: overview, grouped")
    heading = title or f"{spec.label}: {y} ~ f({x})   R²={result['r_squared']:.3f}"

    if display == "overview":
        fig, ax = plt.subplots(figsize=(10, 6))
        draw_panel(ax, data, levels, heading)
        panel_axes = [ax]
        fig.tight_layout()
    else:
        from nvitk.stats.region_groups import panel_grid, resolve_panels

        if not grouped:
            raise ValueError(
                "The grouped display needs a grouping column to split by; this fit has none. "
                "Use the Overview display."
            )
        panels = resolve_panels(levels, column=group or "group")
        fig, axes = panel_grid(len(panels), title=heading)
        panel_axes = []
        for ax, (panel, members) in zip(axes, panels.items()):
            sub = data.loc[data[group].astype(str).isin({str(v) for v in members})]
            if sub.empty:
                ax.text(0.5, 0.5, "No observations", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(panel)
                ax.set_axis_off()
                continue
            draw_panel(ax, sub, list(members), panel)
            panel_axes.append(ax)

    fig.linked_axes = panel_axes
    return fig


__all__ = [
    "ANALYSIS_GLM",
    "ANALYSIS_NONLINEAR",
    "ANALYSIS_OLS",
    "GLM_FAMILIES",
    "NONLINEAR_MODELS",
    "SPLINE_TERMS",
    "GlmFamily",
    "NonlinearModel",
    "SplineTerm",
    "fit_glm",
    "fit_nonlinear",
    "fit_ols",
    "model_info_dict",
    "nonlinear_confidence_band",
    "plot_nonlinear_fit",
    "prepare_model_frame",
    "spline_term",
]
