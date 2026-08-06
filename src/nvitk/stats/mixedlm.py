"""General mixed-effects modeling utilities built on top of statsmodels."""
# """TODO: GPU implementation Cupy + cuDF"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from nvitk.core.logger import Logger

log = Logger()


def _safe_col(name: str) -> str:
    """Sanitize a name into a valid patsy/statsmodels column identifier (non-alphanumeric chars → underscore)."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name))


def _formula_tokens(text: str) -> set[str]:
    """Identifier-like tokens in a patsy formula string (mirrors how statsmodels resolves formula terms)."""
    import tokenize
    from io import StringIO

    raw = str(text or "").strip()
    if not raw:
        return set()
    try:
        return {tok.string for tok in tokenize.generate_tokens(StringIO(raw).readline)}
    except Exception:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw))


def formula_columns(
    columns: Iterable[str],
    formula: str,
    *,
    re_formula: str | None = None,
    vc_formula: Mapping[str, str] | None = None,
    groups: str | None = None,
) -> list[str]:
    """
    Columns of *columns* referenced by the fixed-effects *formula* and the random-effects /
    variance-component specs.

    Used to align the NA-drop set with the rows patsy will actually keep: patsy silently drops rows
    with missing values in formula terms, while ``groups`` / ``exog_re`` keep their full length,
    which makes statsmodels raise ``Shape mismatch between endog/exog and extra 2d arrays``.
    """
    available = {str(c) for c in columns}
    tokens: set[str] = set()
    for text in (formula, re_formula, *(dict(vc_formula or {}).values())):
        tokens |= _formula_tokens(text or "")
    if groups:
        tokens.add(str(groups))
    return sorted(tokens & available)


def _coerce_object_numerics(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Coerce object/string columns that look numeric to ``float64``.

    Patsy treats object-dtype columns as categoricals (``Hematocrit[T.36.0]``, …) even when every
    cell is a float. That happens when clinical wide frames mix numeric and text variables into one
    ``value`` series before the pivot. Skip columns wrapped in ``C(...)`` by the caller — we only
    receive bare column names here.
    """
    out = df
    changed = False
    for col in columns:
        if col not in out.columns:
            continue
        series = out[col]
        if pd.api.types.is_numeric_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        # Only convert when the column is predominantly numeric (avoid turning IDs / groups into NaN).
        n_non_null = int(series.notna().sum())
        if n_non_null == 0:
            continue
        if int(numeric.notna().sum()) < max(1, int(0.9 * n_non_null)):
            continue
        if not changed:
            out = df.copy()
            changed = True
        out[col] = numeric
    return out


def _lighten(color: Any, amount: float = 0.55) -> tuple[float, float, float]:
    """
    Blend an RGB *color* towards white by *amount*.

    Used to give observed (unadjusted) means a lighter tone of their group's colour: the pairing
    stays obvious, but the model curve and the raw data are no longer the same ink.
    """
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    factor = min(max(float(amount), 0.0), 1.0)
    return (r + (1.0 - r) * factor, g + (1.0 - g) * factor, b + (1.0 - b) * factor)


def _z_critical(ci_level: float) -> float:
    """Two-sided normal critical value for a confidence level (0.95 → 1.959964…)."""
    from scipy.stats import norm

    level = min(max(float(ci_level), 0.5), 0.9999)
    return float(norm.ppf(1.0 - (1.0 - level) / 2.0))


_NATURAL_CHUNK_RE = re.compile(r"(\d+)")


def _natural_sort_key(value: Any) -> tuple:
    """
    Sort key that orders embedded numbers numerically: ``g0 < g1 < g2 < g10``.

    Plain string sorting puts ``g10`` before ``g2``; plain ``pd.unique`` order is not sorted at all.
    Numeric values sort ahead of strings so a mixed column still has a deterministic order.
    """
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return (0, float(value), "")
    text = str(value)
    # Split into alternating text / number chunks so comparison happens piecewise.
    chunks = tuple(
        (1, 0.0, chunk.lower()) if i % 2 == 0 else (0, float(chunk), "")
        for i, chunk in enumerate(_NATURAL_CHUNK_RE.split(text))
        if chunk != ""
    )
    return (1, 0.0, "") + chunks


def _prediction_standard_errors(
    result: Any,
    df: pd.DataFrame,
    *,
    x: str,
    x_values: np.ndarray,
    group: str,
    levels: Sequence[Any],
    covariate_refs: Mapping[str, Any],
) -> dict[str | None, np.ndarray]:
    """
    Delta-method standard error of the fixed-effects prediction along *x*, once per group level.

    Each level gets its own design matrix with the grouping column pinned to that level, so when the
    group appears in the formula (``C(territory)``) its interval reflects that level's own precision
    — a level with few observations gets a visibly wider band than a well-sampled one. When the
    group is only a random effect the design is identical across levels and every band comes out the
    same width, which is the honest answer: the fit holds no level-specific fixed-effect
    information.

    Returns
    -------
    dict
        ``{None: (prediction, se)}`` for the overall fixed-effects line, plus
        ``{str(level): (prediction, se)}`` per level. Levels whose design matrix cannot be built are
        simply absent. The prediction is on the link scale; the caller applies the inverse link.
    """
    import patsy

    design_info = result.model.data.design_info
    fe = model_params(result)
    cov = result.cov_params()
    # A MixedLM's cov_params spans the variance components too; keep the fixed-effects block.
    cov = cov.loc[fe.index, fe.index]

    def se_for(pins: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Prediction and its standard error along ``x_values`` with *pins* held constant."""
        grid = pd.DataFrame({x: x_values})
        for name, value in pins.items():
            if name != x:
                grid[name] = value
        grid = _match_grid_columns_to_df_dtypes(grid, df, [c for c in grid.columns if c in df.columns])
        design = patsy.build_design_matrices([design_info], grid, return_type="dataframe")[0]
        design = design.reindex(columns=fe.index, fill_value=0.0)
        prediction = (design @ fe).to_numpy(dtype=float)
        se = np.sqrt(np.clip(np.sum((design @ cov) * design, axis=1).to_numpy(dtype=float), 0.0, None))
        return prediction, se

    out: dict[str | None, tuple[np.ndarray, np.ndarray]] = {}
    base_refs = {k: v for k, v in covariate_refs.items() if k != x}
    grouped = bool(group) and group in df.columns and group != x

    # The black "fixed effect" line is drawn with every group contrast at zero — i.e. at the
    # grouping factor's reference level — so its band has to be evaluated there to match.
    overall_refs = dict(base_refs)
    if grouped and group not in overall_refs:
        reference_levels = sorted(df[group].dropna().astype(str).unique())
        if reference_levels:
            overall_refs[group] = reference_levels[0]
    try:
        out[None] = se_for(overall_refs)
    except Exception as exc:
        log.debug("No overall confidence band: %s", exc)

    if grouped:
        for level in levels:
            try:
                out[str(level)] = se_for({**base_refs, group: level})
            except Exception as exc:  # a level patsy cannot encode (unseen category)
                log.debug("No confidence band for %s=%s: %s", group, level, exc)
    if not out:
        raise ValueError(
            "The prediction grid could not be built for any level — pin every formula term that is "
            "neither the x axis nor the grouping column."
        )
    return out


def _match_grid_columns_to_df_dtypes(
    grid: pd.DataFrame,
    df_ref: pd.DataFrame,
    cols: Iterable[str],
) -> pd.DataFrame:
    """Align ``grid`` column dtypes with ``df_ref`` so patsy sees the same encodings."""
    out = grid.copy()
    for c in cols:
        if c not in out.columns or c not in df_ref.columns:
            continue
        ref = df_ref[c]
        if isinstance(ref.dtype, pd.CategoricalDtype):
            cats = list(ref.dtype.categories)
            out[c] = pd.Categorical(out[c], categories=cats, ordered=ref.dtype.ordered)
        elif pd.api.types.is_numeric_dtype(ref):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = out[c].astype(object)
    return out


def _map_sex_to_numeric(series: pd.Series) -> pd.Series:
    """Coerce a sex column (numeric, or text like ``\"M\"``/``\"female\"``) to ``1``=male / ``0``=female."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "male": 1,
        "m": 1,
        "1": 1,
        "female": 0,
        "f": 0,
        "0": 0,
    }
    return pd.to_numeric(s.map(mapping), errors="coerce")


def _default_fit_function(
    model_df: pd.DataFrame,
    *,
    formula: str,
    groups: str,
    re_formula: str = "1",
    vc_formula: dict[str, str] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
):
    """Default mixed-effects model fitter: ``statsmodels`` MixedLM via a formula/groups/random-effects spec."""
    import warnings

    # from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import statsmodels.formula.api as smf

    fit_kwargs = dict(fit_kwargs or {})
    model = smf.mixedlm(
        formula,
        data=model_df,
        groups=model_df[groups],
        re_formula=re_formula,
        vc_formula=vc_formula,
        # Let MixedLM drop incomplete rows from the design *and* from groups/exog_re/exog_vc
        # together. With the default ('none') only patsy drops them, and the length mismatch
        # surfaces as "Shape mismatch between endog/exog and extra 2d arrays given to model".
        missing="drop",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="statsmodels")
        result = model.fit(**fit_kwargs)
    return result


def fit_or_load_mixedlm(
    *,
    data: pd.DataFrame,
    formula: str,
    groups: str,
    model_path: str | Path | None = None,
    re_formula: str = "1",
    vc_formula: dict[str, str] | None = None,
    overwrite: bool = False,
    required_columns: list[str] | None = None,
    dropna_columns: list[str] | None = None,
    outcome_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    fit_fn: Callable[..., Any] | None = None,
    load_fn: Callable[[Path], Any] | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """
    Fit or load a statsmodels MixedLM result.

    Rows with missing values in any column referenced by *formula* / *re_formula* / *vc_formula*
    are dropped before fitting, so the returned ``model_df`` has the same grain as the fit and can
    be reused for prediction / plotting. Pass *dropna_columns* to restrict the NA-drop to an
    explicit column set instead.

    Returns
    -------
    (result, model_df, metadata)
    """
    from statsmodels.regression.mixed_linear_model import MixedLMResults

    if model_path is not None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    else: path = None
    df = data.copy()
    df.columns = [_safe_col(c) for c in df.columns]

    # Patsy formula tokens that are *not* wrapped in C() — coerce object-numerics among these so
    # Hematocrit/age/sex stay continuous even if the upstream wide frame upcast them to object.
    formula_cols = formula_columns(
        df.columns,
        formula,
        re_formula=re_formula,
        vc_formula=vc_formula,
        groups=groups,
    )
    # Keep explicit C(col) terms categorical: drop any name that appears inside C(...) in the formula.
    categorical_forced = set()
    for text in (formula, re_formula, *(dict(vc_formula or {}).values())):
        categorical_forced |= set(re.findall(r"C\(\s*([A-Za-z_][A-Za-z0-9_]*)", str(text or "")))
    coerce_cols = [c for c in formula_cols if c not in categorical_forced]
    df = _coerce_object_numerics(df, coerce_cols)

    req = list(required_columns or [])
    if groups not in req:
        req.append(groups)
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for MixedLM: {missing}")

    if outcome_transform is not None:
        df = outcome_transform(df)

    if dropna_columns is None:
        na_cols = sorted(set(req) | set(formula_cols))
    else:
        na_cols = list(dropna_columns)
    na_cols = [c for c in na_cols if c in df.columns]

    n_input = int(len(df))
    dropped_by_column: dict[str, int] = {}
    if na_cols:
        dropped_by_column = {
            c: int(df[c].isna().sum()) for c in na_cols if bool(df[c].isna().any())
        }
        df = df.dropna(subset=na_cols).reset_index(drop=True)
    if not len(df):
        detail = ", ".join(f"{c} ({n} missing)" for c, n in sorted(dropped_by_column.items()))
        raise ValueError(
            f"No complete rows left for MixedLM after dropping missing values in {na_cols} "
            f"(started from {n_input} rows)."
            + (f" Columns with missing values: {detail}." if detail else "")
        )

    metadata = {
        "model_path": str(path),
        "formula": formula,
        "groups": groups,
        "re_formula": re_formula,
        "vc_formula": vc_formula,
        "n_rows_input": n_input,
        "n_rows": int(len(df)),
        "n_rows_dropped": n_input - int(len(df)),
        "dropna_columns": na_cols,
        "dropped_by_column": dropped_by_column,
        "loaded": False,
    }

    if path is not None and path.exists() and not overwrite:
        if load_fn is not None:
            result = load_fn(path)
        else:
            result = MixedLMResults.load(path)
        metadata["loaded"] = True
        return result, df, metadata

    runner = fit_fn or _default_fit_function
    result = runner(
        df,
        formula=formula,
        groups=groups,
        re_formula=re_formula,
        vc_formula=vc_formula,
        fit_kwargs=fit_kwargs,
    )
    if path is not None: result.save(path)
    return result, df, metadata


# ──────────────────────────────────────────────────────────────────────────────
# Model reporting — structured extraction, then rendering
# ──────────────────────────────────────────────────────────────────────────────
def significance_stars(pval: float) -> str:
    """Conventional significance marker: ``***`` <0.001, ``**`` <0.01, ``*`` <0.05, ``.`` <0.1,
    ``NS`` otherwise, and ``""`` for a missing p-value."""
    if pval is None or np.isnan(pval):
        return ""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    if pval < 0.1:
        return "."
    return "NS"


def model_params(result: Any) -> pd.Series:
    """
    Fixed-effect coefficients of a fitted model.

    MixedLM exposes them as ``fe_params`` (``params`` there also carries the variance components);
    OLS and GLM use ``params``. One accessor so the reporting and plotting code does not care which
    engine produced the result.
    """
    fe = getattr(result, "fe_params", None)
    if fe is None:
        fe = getattr(result, "params", pd.Series(dtype=float))
    return fe


def model_random_effects(result: Any) -> Mapping[str, Any]:
    """Per-group random effects, or an empty mapping for models that have none."""
    return getattr(result, "random_effects", {}) or {}


def model_inverse_link(result: Any) -> Callable[[Any], Any]:
    """
    Map a linear predictor back to the response scale.

    Identity for OLS and MixedLM; the family's inverse link for a GLM, which is where a GLM's
    non-linearity lives — predictions are linear on the link scale and curved on the response scale.
    """
    family = getattr(getattr(result, "model", None), "family", None)
    link = getattr(family, "link", None)
    if link is None:
        return lambda values: values
    inverse = getattr(link, "inverse", None)
    if inverse is None:
        return lambda values: values
    return lambda values: inverse(np.asarray(values, dtype=float))


def mixedlm_coef_frame(result: Any, *, alpha: float = 0.05) -> pd.DataFrame:
    """
    Fixed-effects table of a fitted model (MixedLM, OLS or GLM).

    Returns
    -------
    pandas.DataFrame
        Columns ``parameter, coef, std_err, z, p_value, ci_low, ci_high, sig``. ``z`` and the
        confidence bounds are derived from the coefficient and its standard error (a Wald interval),
        so they are present even for results that do not expose ``conf_int``.
    """
    fe = model_params(result)
    fe_se = getattr(result, "bse_fe", getattr(result, "bse", pd.Series(dtype=float)))
    fe_p = getattr(result, "pvalues_fe", getattr(result, "pvalues", pd.Series(dtype=float)))
    # 1.959964… for the default alpha; computed so a caller can widen/narrow the interval.
    from scipy.stats import norm

    crit = float(norm.ppf(1.0 - alpha / 2.0))

    rows: list[dict[str, Any]] = []
    for param in fe.index:
        coef = float(fe[param])
        se = float(fe_se.get(param, np.nan))
        pval = float(fe_p.get(param, np.nan))
        rows.append(
            {
                "parameter": str(param),
                "coef": coef,
                "std_err": se,
                "z": coef / se if se else np.nan,
                "p_value": pval,
                "ci_low": coef - crit * se,
                "ci_high": coef + crit * se,
                "sig": significance_stars(pval),
            }
        )
    return pd.DataFrame(rows, columns=[
        "parameter", "coef", "std_err", "z", "p_value", "ci_low", "ci_high", "sig",
    ])


def mixedlm_random_effects_frame(result: Any) -> pd.DataFrame:
    """
    Variance structure of a MixedLM fit, one row per component.

    Returns
    -------
    pandas.DataFrame
        Columns ``component, kind, var, sd``. ``kind`` is ``"cov_re"`` for the group random-effects
        covariance diagonal, ``"vcomp"`` for variance components, and ``"residual"`` for the scale.
    """
    rows: list[dict[str, Any]] = []

    cov_re = getattr(result, "cov_re", None)
    if isinstance(cov_re, pd.DataFrame) and not cov_re.empty:
        for name in cov_re.index:
            var = float(cov_re.loc[name, name]) if name in cov_re.columns else np.nan
            rows.append(
                {
                    "component": str(name),
                    "kind": "cov_re",
                    "var": var,
                    "sd": float(np.sqrt(max(var, 0.0))) if np.isfinite(var) else np.nan,
                }
            )

    vcomp = getattr(result, "vcomp", None)
    if vcomp is not None:
        vc_names = getattr(result.model, "vc_names", [f"VC_{i}" for i in range(len(vcomp))])
        for name, var in zip(vc_names, vcomp):
            var_f = float(var)
            rows.append(
                {
                    "component": str(name),
                    "kind": "vcomp",
                    "var": var_f,
                    "sd": float(np.sqrt(max(var_f, 0.0))),
                }
            )

    if hasattr(result, "scale"):
        scale = float(result.scale)
        rows.append(
            {
                "component": "Residual",
                "kind": "residual",
                "var": scale,
                "sd": float(np.sqrt(max(scale, 0.0))),
            }
        )
    return pd.DataFrame(rows, columns=["component", "kind", "var", "sd"])


def mixedlm_group_coefficients(result: Any, *, factor: str = "Group") -> pd.DataFrame:
    """
    Per-group coefficients of a MixedLM: each level's own intercept and slopes.

    A random-slope model gives every group its own line, but those live in the random effects, not
    in the fixed-effects table — so a coefficient table showing one ``age_c`` row can read as if the
    slope were shared. This makes the per-level values explicit.

    Two numbers per term: the **total** (fixed effect + that level's random deviation), which is the
    line actually fitted for the group, and the **deviation** alone, which says how far the group
    sits from the population average.

    Returns
    -------
    pandas.DataFrame
        ``factor``, ``level``, then ``<term>`` and ``<term>_dev`` per random term. Empty when the
        model has no random effects.
    """
    random_effects = model_random_effects(result)
    if not random_effects:
        return pd.DataFrame(columns=["factor", "level"])

    fixed = model_params(result)
    rows: list[dict[str, Any]] = []
    for level, values in random_effects.items():
        row: dict[str, Any] = {"factor": factor, "level": str(level)}
        for term, deviation in dict(values).items():
            # statsmodels names the random intercept "Group"; its fixed counterpart is "Intercept".
            name = "Intercept" if str(term) in {"Group", "Intercept", "const"} else str(term)
            base = float(fixed.get(name, fixed.get("const", 0.0) if name == "Intercept" else 0.0))
            row[name] = base + float(deviation)
            row[f"{name}_dev"] = float(deviation)
        rows.append(row)
    return pd.DataFrame(rows)


def mixedlm_info_dict(
    result: Any,
    *,
    outcome_name: str = "Outcome",
    group_name: str = "Group",
    vc_group_name: str = "Variance Components",
) -> dict[str, Any]:
    """
    Everything :func:`print_mixedlm_info` reports, as structured data.

    This is the single source of truth for model reporting: :func:`render_mixedlm_info` turns it back
    into the classic fixed-width text, and the GUI renders the same dict as tables.

    Returns
    -------
    dict
        ``header``  — ``formula`` (``None`` when the model does not expose one), ``n_obs``,
        ``n_groups``, ``group_name``, ``outcome_name``, ``vc_group_name``;
        ``fixed_effects``  — :func:`mixedlm_coef_frame` output;
        ``random_effects`` — :func:`mixedlm_random_effects_frame` output;
        ``cov_re``         — the raw group covariance matrix (empty frame when absent);
        ``fit_statistics`` — ``llf``, ``aic``, ``bic``, ``converged``, ``scale``, ``resid_sd``
        (keys are omitted when the result does not expose them).

    Raises
    ------
    ValueError
        If *result* does not look like a ``MixedLMResults``.
    """
    required_attrs = ["fe_params", "summary", "nobs"]
    missing = [a for a in required_attrs if not hasattr(result, a)]
    if missing:
        raise ValueError(f"Object does not look like MixedLMResults. Missing attributes: {missing}")

    # ---- 1. Header: formula is best-effort, some fitters do not carry one ------
    try:
        formula = str(result.model.formula)
    except Exception:
        formula = None
    ngroups = getattr(result, "ngroups", None)
    if ngroups is None and hasattr(result, "random_effects"):
        ngroups = len(result.random_effects)

    header = {
        "formula": formula,
        "n_obs": int(result.nobs),
        "n_groups": int(ngroups) if ngroups is not None else None,
        "group_name": group_name,
        "outcome_name": outcome_name,
        "vc_group_name": vc_group_name,
    }

    # ---- 2. Fit statistics, each guarded so a partial result still reports -----
    fit_stats: dict[str, Any] = {}
    for attr in ("llf", "aic", "bic"):
        if hasattr(result, attr):
            fit_stats[attr] = float(getattr(result, attr))
    if hasattr(result, "converged"):
        fit_stats["converged"] = bool(result.converged)
    if hasattr(result, "scale"):
        scale = float(result.scale)
        fit_stats["scale"] = scale
        fit_stats["resid_sd"] = float(np.sqrt(max(scale, 0.0)))

    cov_re = getattr(result, "cov_re", pd.DataFrame())
    if not isinstance(cov_re, pd.DataFrame):
        cov_re = pd.DataFrame()

    return {
        "header": header,
        "fixed_effects": mixedlm_coef_frame(result),
        "random_effects": mixedlm_random_effects_frame(result),
        "cov_re": cov_re,
        "fit_statistics": fit_stats,
        "has_vcomp": getattr(result, "vcomp", None) is not None,
        "group_effects": mixedlm_group_coefficients(result, factor=group_name),
    }


def render_mixedlm_info(info: dict[str, Any]) -> str:
    """Render :func:`mixedlm_info_dict` output as the classic fixed-width MixedLM report."""
    header = dict(info.get("header") or {})
    group_name = str(header.get("group_name") or "Group")
    fit_stats = dict(info.get("fit_statistics") or {})

    buffer = io.StringIO()
    _w = lambda line="": buffer.write(f"{line}\n")

    _w("=" * 88)
    _w(f"Mixed Linear Model - {header.get('outcome_name', 'Outcome')}")
    _w("=" * 88)
    if header.get("formula"):
        _w(f"Formula: {header['formula']}")
    _w(f"Observations: {int(header.get('n_obs') or 0):,}")
    if header.get("n_groups") is not None:
        _w(f"{group_name} groups: {int(header['n_groups']):,}")
    _w()

    _w("Fixed effects")
    _w("-" * 88)
    _w(f"{'Parameter':<34}{'Coef':>12}{'Std.Err':>12}{'P-value':>12}{'Sig':>10}")
    for row in info["fixed_effects"].itertuples(index=False):
        _w(
            f"{row.parameter:<34}{row.coef:>12.4f}{row.std_err:>12.4f}"
            f"{row.p_value:>12.4g}{row.sig:>10}"
        )
    _w()

    cov_re = info.get("cov_re")
    if isinstance(cov_re, pd.DataFrame) and not cov_re.empty:
        _w(f"{group_name} random effects covariance")
        _w("-" * 88)
        _w(cov_re.to_string())
        _w()

    if info.get("has_vcomp"):
        re_frame = info["random_effects"]
        # ``vcomp`` is statsmodels' name for these; the R engines label them ``ranef``. Both belong
        # in this section — filtering on one spelling silently emptied it for the other.
        shown = re_frame[re_frame["kind"].isin(["vcomp", "ranef"])]
        _w(f"Variance components ({header.get('vc_group_name', 'Variance Components')})")
        _w("-" * 88)
        for row in shown.itertuples(index=False):
            _w(f"{row.component:<24} var={row.var:.6f}  sd={row.sd:.6f}")
        _w()

    if "scale" in fit_stats:
        _w(f"Residual variance: {fit_stats['scale']:.6f}")
        _w(f"Residual std dev: {fit_stats['resid_sd']:.6f}")
        _w()

    _w("Fit statistics")
    _w("-" * 88)
    for attr in ("llf", "aic", "bic"):
        if attr in fit_stats:
            _w(f"{attr.upper():<8}: {fit_stats[attr]:.4f}")
    if "converged" in fit_stats:
        _w(f"Converged: {fit_stats['converged']}")
    _w("=" * 88)
    return buffer.getvalue()


def print_mixedlm_info(
    result: Any,
    *,
    outcome_name: str = "Outcome",
    group_name: str = "Group",
    vc_group_name: str = "Variance Components",
    output_path: str | Path | None = None,
) -> str:
    """Print and optionally save a generic MixedLM summary report."""
    info = mixedlm_info_dict(
        result,
        outcome_name=outcome_name,
        group_name=group_name,
        vc_group_name=vc_group_name,
    )
    text = render_mixedlm_info(info)
    log.info(text)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text


def plot_mixedlm_params(
    *,
    result: Any,
    df_fit: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    mode: str = "auto",
    categorical_order: list[str] | None = None,
    group_order: list[str] | None = None,
    hue: str | None = None,
    hue_order: list[str] | None = None,
    palette: str | Mapping[str, str] = "tab10",
    include_points: bool = True,
    restrict_to_orders: bool = False,
    errorbar: bool = False,
    ci_level: float = 0.95,
    covariate_refs: dict[str, float] | None = None,
    display: str = "overview",
    output_path: str | Path | None = None,
    title: str = "MixedLM parameter plot",
    x_label: str | None = None,
    y_label: str | None = None,
) -> Any:
    """Generic plotting utility for continuous or grouped/categorical predictors.

    In ``mode="categorical"``, fixed-effects EMM uses a prediction grid at
    ``covariate_refs``. If the model formula includes a second categorical factor
    (e.g. ``C(x) * C(territory)``), pass that factor as ``hue`` or as ``group``
    when it differs from ``x``; the grid is then the factorial ``x`` × factor so
    patsy can evaluate all terms.

    Parameters
    ----------
    restrict_to_orders : bool
        By default ``group_order`` / ``hue_order`` / ``categorical_order`` only select which model
        curves are drawn — the raw observations still show every level. Set this to also subset
        ``df_fit`` to the listed levels, so hiding a group hides its points and lets the axes
        autoscale to what remains. The fit itself is untouched: the fixed-effect line is still the
        all-group estimate.
    errorbar : bool
        Draw the confidence interval of the model predictions. In categorical mode these are
        whiskers on each marginal mean, from the delta-method standard error of the fixed-effects
        prediction; in continuous mode a shaded band around the fixed-effect line.
    ci_level : float
        Confidence level for those intervals.
    include_points : bool
        Overlay the observed data. In continuous mode this is a scatter of individual rows; in
        categorical mode it is the **mean of y within each (x, hue) cell** — an unadjusted
        counterpart to the model's marginal means, drawn dashed and in a lighter tone.
    display : {"overview", "grouped"}
        ``"overview"`` draws every level on one pair of axes. ``"grouped"`` splits them into a grid
        of anatomical panels (carotids / anterior / posterior / venous for vessels, lobes for
        cortical parcels — see :mod:`nvitk.stats.region_groups`), each autoscaled to its own range.
        The model is untouched: the panels are a view of one fit, and the dashed population line is
        the same all-level estimate in each of them. Raises when no level maps to a known region.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df_fit.copy()
    covariate_refs = dict(covariate_refs or {})
    x_label = x_label or x
    y_label = y_label or y

    if x not in df.columns:
        raise ValueError(f"x column {x!r} is not in df_fit")
    if y not in df.columns:
        raise ValueError(f"y column {y!r} is not in df_fit")
    if group not in df.columns:
        raise ValueError(f"group column {group!r} is not in df_fit")

    if mode == "auto":
        mode = "categorical" if (categorical_order is not None or not pd.api.types.is_numeric_dtype(df[x])) else "continuous"
    if mode not in {"continuous", "categorical"}:
        raise ValueError("mode must be one of: auto, continuous, categorical")

    # ---- 0. Optionally subset the raw data to the requested levels --------------
    # Done before anything reads ``df`` so the scatter/pointplot cloud, the x-range and the
    # categorical tick order all follow the selection.
    if restrict_to_orders:
        if group_order is not None and group in df.columns:
            df = df.loc[df[group].astype(str).isin({str(g) for g in group_order})]
        if hue is not None and hue_order is not None and hue in df.columns:
            df = df.loc[df[hue].astype(str).isin({str(h) for h in hue_order})]
        if mode == "categorical" and categorical_order is not None and x in df.columns:
            df = df.loc[df[x].astype(str).isin({str(v) for v in categorical_order})]
        if df.empty:
            raise ValueError("No rows left after restricting to the selected levels.")

    # ---- 1. Resolve the categorical axis order once ----------------------------
    # Every panel of a grouped display must share one x axis, so this cannot be left to the
    # per-axes drawing where each subset would order its own observed levels.
    if mode == "categorical":
        categorical_order = _categorical_axis_order(df, x, categorical_order)

    display = str(display or "overview").strip().lower()
    if display not in {"overview", "grouped"}:
        raise ValueError("display must be one of: overview, grouped")

    common = dict(
        result=result,
        x=x,
        y=y,
        group=group,
        mode=mode,
        categorical_order=categorical_order,
        hue=hue,
        palette=palette,
        include_points=include_points,
        errorbar=errorbar,
        ci_level=ci_level,
        covariate_refs=covariate_refs,
    )

    if display == "overview":
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = _draw_mixedlm_axes(ax, df=df, group_order=group_order, hue_order=hue_order, **common)
        _finish_mixedlm_axes(ax, title=title, x_label=x_label, y_label=y_label)
        panel_axes = [ax]
    else:
        fig, panel_axes, errors = _draw_grouped_panels(
            df=df,
            group_order=group_order,
            hue_order=hue_order,
            title=title,
            x_label=x_label,
            y_label=y_label,
            common=common,
        )

    for name, message in errors.items():
        setattr(fig, name, message)
    # The plot pane rescales these together: panels of one model share a data space, and axis
    # sliders that moved only the first panel would make the small multiples incomparable.
    fig.linked_axes = panel_axes

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig


def _draw_grouped_panels(
    *,
    df: pd.DataFrame,
    group_order: list[str] | None,
    hue_order: list[str] | None,
    title: str,
    x_label: str,
    y_label: str,
    common: dict[str, Any],
) -> tuple[Any, list[Any], dict[str, str]]:
    """
    One axes per anatomical panel — carotids, anterior, posterior, venous — instead of one crowded
    axes for every level.

    Each panel is drawn from the rows of its own levels, so it autoscales to its own range: a venous
    ``log(PI)`` around −1.0 no longer flattens a carotid one around −0.2. The palette restarts in
    every panel, which is what keeps four curves legible where thirteen were not. The population
    (fixed-effect) line is unchanged by the split — it is the all-level estimate — so it repeats
    identically across panels and remains the common reference between them.
    """
    import matplotlib.pyplot as plt

    from nvitk.stats.region_groups import group_levels_into_panels

    group = common["group"]
    levels = group_order or sorted(df[group].dropna().astype(str).unique(), key=_natural_sort_key)
    panels = group_levels_into_panels(levels)
    if not panels:
        raise ValueError(
            f"None of the {len(levels)} {group!r} levels map to a known anatomical region, so there "
            f"are no groups to split into. Use the Overview display for this model."
        )

    n_cols = 1 if len(panels) == 1 else 2
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, grid = plt.subplots(
        n_rows, n_cols, figsize=(8.5 * n_cols, 4.8 * n_rows), squeeze=False
    )
    flat = [ax for row in grid for ax in row]

    errors: dict[str, str] = {}
    panel_axes: list[Any] = []
    for ax, (panel, panel_levels) in zip(flat, panels.items()):
        wanted = {str(v) for v in panel_levels}
        sub = df.loc[df[group].astype(str).isin(wanted)]
        if sub.empty:
            # A level can be selected but have no rows left after filtering; say so on the panel
            # rather than raising and losing the whole figure.
            ax.text(0.5, 0.5, "No observations", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(panel)
            ax.set_axis_off()
            continue
        errors.update(
            _draw_mixedlm_axes(
                ax,
                df=sub,
                group_order=list(panel_levels),
                # The hue is only restricted when it *is* the panelled column; an unrelated second
                # factor must keep all its levels or the panels stop being comparable.
                hue_order=list(panel_levels) if common.get("hue") == group else hue_order,
                **common,
            )
        )
        _finish_mixedlm_axes(ax, title=panel, x_label=x_label, y_label=y_label)
        panel_axes.append(ax)

    for ax in flat[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    # Constrained layout re-flows on every draw, which a grid with a suptitle needs; it also tells
    # the plot pane not to call tight_layout on top of it.
    fig.set_layout_engine("constrained")
    return fig, panel_axes, errors


def _categorical_axis_order(
    df: pd.DataFrame, x: str, categorical_order: list[str] | None
) -> list[Any]:
    """
    Order of the categorical x axis.

    ``pd.unique`` returns first-appearance order, which puts the levels in whatever sequence the
    rows happen to arrive in (g1, g2, g0, g3) — meaningless to read and different on every reload.
    """
    if categorical_order is not None:
        return list(categorical_order)
    if isinstance(df[x].dtype, pd.CategoricalDtype) and df[x].dtype.ordered:
        # A binned column already carries its intended order; honour it rather than re-sorting, so
        # labels like "low"/"medium"/"high" stay in sequence.
        present = set(df[x].dropna().astype(str))
        return [c for c in df[x].dtype.categories if str(c) in present]
    # Natural sort so g2 comes before g10.
    return sorted(pd.unique(df[x].dropna()), key=_natural_sort_key)


def _finish_mixedlm_axes(ax: Any, *, title: str, x_label: str, y_label: str) -> None:
    """Titles, grid and a de-duplicated legend — applied identically to every axes."""
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup: dict[str, Any] = {}
        for h, l in zip(handles, labels):
            if l not in dedup:
                dedup[l] = h
        ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=9)


def _draw_mixedlm_axes(
    ax: Any,
    *,
    result: Any,
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    mode: str,
    categorical_order: list[str] | None = None,
    group_order: list[str] | None = None,
    hue: str | None = None,
    hue_order: list[str] | None = None,
    palette: str | Mapping[str, str] = "tab10",
    include_points: bool = True,
    errorbar: bool = False,
    ci_level: float = 0.95,
    covariate_refs: dict[str, float] | None = None,
) -> dict[str, str]:
    """
    Draw one model plot onto *ax* — the body shared by the overview and the grouped panels.

    Everything that varies between panels arrives as an argument: *df* is already restricted to the
    panel's rows and *group_order* to its levels, which is what makes each panel autoscale and
    restart the palette. Nothing here reads the figure, so the caller owns titles, the legend and
    the layout.

    Returns
    -------
    dict
        ``{"emm_error": …}`` / ``{"ci_error": …}`` for failures that degrade the plot rather than
        abort it, so a caller can report why the model curves are missing.
    """
    import seaborn as sns

    covariate_refs = dict(covariate_refs or {})
    errors: dict[str, str] = {}

    if mode == "continuous":
        x_num = pd.to_numeric(df[x], errors="coerce")
        xmin, xmax = float(np.nanmin(x_num)), float(np.nanmax(x_num))
        x_line = np.linspace(xmin, xmax, 200)

        point_hue = hue if hue else group
        groups = group_order or sorted(df[group].dropna().astype(str).unique())
        colors = sns.color_palette(palette, n_colors=max(len(groups), 3))
        cmap = {g: colors[i % len(colors)] for i, g in enumerate(groups)}

        if include_points:
            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                hue=point_hue,
                hue_order=hue_order if hue else groups,
                palette=cmap if point_hue == group else palette,
                alpha=0.5,
                s=18,
                legend=False,
                ax=ax,
            )

        fe = model_params(result)
        re_dict = model_random_effects(result)
        b_int = float(fe.get("Intercept", fe.get("const", 0.0)))
        b_x = float(fe.get(x, 0.0))
        extra = 0.0
        for cov_name, cov_val in covariate_refs.items():
            # Categorical references (a level name) are only meaningful to the categorical branch's
            # patsy grid; here they would blow up on float().
            if isinstance(cov_val, (int, float, np.integer, np.floating)):
                extra += float(fe.get(cov_name, 0.0)) * float(cov_val)

        inverse = model_inverse_link(result)
        ax.plot(
            x_line,
            inverse(b_int + extra + b_x * x_line),
            color="black",
            lw=2.8,
            ls="--",
            label="Fixed effect",
        )

        # Fixed-effects prediction and its standard error along the line, per group. Needed both to
        # draw per-group bands and — for models with no random effects — to draw the per-group
        # lines at all.
        crit = _z_critical(ci_level)
        predictions: dict[str | None, tuple[np.ndarray, np.ndarray]] = {}
        try:
            predictions = _prediction_standard_errors(
                result,
                df,
                x=x,
                x_values=x_line,
                group=group,
                levels=groups,
                covariate_refs=covariate_refs,
            )
        except Exception as exc:
            if errorbar:
                log.warning("Could not build the confidence band: %s", exc)
                log.debug("CI band failure", exc_info=True)
                errors["ci_error"] = str(exc)

        if errorbar and predictions.get(None) is not None:
            _pred, overall_se = predictions[None]
            fixed_line = b_int + extra + b_x * x_line
            ax.fill_between(
                x_line,
                inverse(fixed_line - crit * overall_se),
                inverse(fixed_line + crit * overall_se),
                color="black",
                alpha=0.12,
                label=f"{int(round(ci_level * 100))}% CI (fixed effect)",
            )

        # Does the grouping factor actually appear in the model? If every level predicts the same
        # curve it does not, and drawing one identical line per level would be pure clutter.
        level_predictions = {k: v for k, v in predictions.items() if k is not None}
        group_is_in_model = len(level_predictions) > 1 and not all(
            np.allclose(next(iter(level_predictions.values()))[0], pred)
            for pred, _se in level_predictions.values()
        )

        for g in groups:
            re_vals = re_dict.get(g, re_dict.get(str(g), None))
            entry = level_predictions.get(str(g))
            if re_vals is not None:
                # MixedLM: the line is the fixed-effects prediction plus this group's random offset.
                re_int = float(re_vals.get("Group", re_vals.get("Intercept", 0.0)))
                re_x = float(re_vals.get(x, 0.0))
                eta = (b_int + extra + re_int) + (b_x + re_x) * x_line
            elif group_is_in_model and entry is not None:
                # OLS / GLM: no random effects, so the group's line *is* its fixed-effects
                # prediction — which only differs between levels when the formula contains the
                # grouping factor.
                eta = entry[0]
            else:
                continue
            ax.plot(x_line, inverse(eta), color=cmap[g], lw=2, alpha=0.9, label=f"{group}={g}")

            if errorbar and entry is not None:
                # Centred on this group's line, widened by the standard error of the fixed-effects
                # prediction *at this group's level*. For a MixedLM the random-effect offset that
                # positions the line carries no standard error from the fit, so it is not included.
                se_g = entry[1]
                ax.fill_between(
                    x_line,
                    inverse(eta - crit * se_g),
                    inverse(eta + crit * se_g),
                    color=cmap[g],
                    alpha=0.15,
                    linewidth=0,
                )

    else:
        order = _categorical_axis_order(df, x, categorical_order)
        # Second factor: interaction models need a full factorial x × (hue|group) grid
        # so patsy can evaluate e.g. C(x) * C(territory); a single-column grid raises
        # NameError for the missing factor.
        facet_col: str | None = None
        if hue is not None and hue != x:
            facet_col = hue
        elif group is not None and group != x:
            facet_col = group

        # ---- Colours, fixed once so the model curves and the observed means stay in step --------
        facet_order: list[Any] = []
        if facet_col is not None:
            if facet_col == hue and hue_order is not None:
                facet_order = list(hue_order)
            elif facet_col == group and group_order is not None:
                facet_order = list(group_order)
            else:
                facet_order = list(pd.unique(df[facet_col].dropna().astype(str)))
        base_colors = sns.color_palette(palette, n_colors=max(len(facet_order), 3))
        emm_colors = {str(lev): base_colors[i % len(base_colors)] for i, lev in enumerate(facet_order)}
        # Observed means get a lighter tone of their level's colour: same hue so the pairing is
        # obvious, lighter so the model curve is not confused with the raw data.
        point_colors = {lev: _lighten(c) for lev, c in emm_colors.items()}
        z_crit = _z_critical(ci_level)

        # EMM-like fixed-effects prediction grid at reference covariates.
        try:
            import patsy

            if facet_col is not None:
                rows = [(xv, fv) for xv in order for fv in facet_order]
                grid = pd.DataFrame(rows, columns=[x, facet_col])
            else:
                grid = pd.DataFrame({x: order})

            for cov_name, cov_val in covariate_refs.items():
                # Never overwrite the two columns the grid varies — a caller that pins every
                # non-outcome term would otherwise collapse the x axis or the facet to a constant.
                if cov_name in {x, facet_col}:
                    continue
                grid[cov_name] = cov_val

            grid = _match_grid_columns_to_df_dtypes(grid, df, [c for c in grid.columns if c in df.columns])
            X = patsy.build_design_matrices([result.model.data.design_info], grid, return_type="dataframe")[0]
            fe = model_params(result)
            cov = result.cov_params().loc[fe.index, fe.index]
            X = X.reindex(columns=fe.index, fill_value=0.0)
            pred = X @ fe
            se = np.sqrt(np.sum((X @ cov) * X, axis=1))
            # For a GLM the linear predictor lives on the link scale; map the estimate and both
            # interval endpoints back through the inverse link so the plot is on the scale of the
            # data. Transforming the endpoints (rather than building a symmetric interval around the
            # transformed mean) is what keeps the interval valid — on the response scale it becomes
            # asymmetric, which is correct for a log or logit link.
            inverse = model_inverse_link(result)
            emm = grid.copy()
            emm["estimate"] = inverse(pred)
            emm["lower"] = inverse(pred - z_crit * se)
            emm["upper"] = inverse(pred + z_crit * se)

            x_to_pos = {k: i for i, k in enumerate(order)}
            emm["_xi"] = emm[x].map(x_to_pos)

            if facet_col is None:
                sub = emm.sort_values("_xi")
                ax.plot(
                    sub["_xi"],
                    sub["estimate"],
                    color="black",
                    lw=2.6,
                    marker="o",
                    label="Fixed-effects EMM",
                )
                if errorbar:
                    # Whiskers rather than a band: on a discrete axis there is nothing between the
                    # ticks to interpolate, and a shaded ribbon would imply values that do not exist.
                    ax.errorbar(
                        sub["_xi"],
                        sub["estimate"],
                        yerr=[sub["estimate"] - sub["lower"], sub["upper"] - sub["estimate"]],
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.4,
                        capsize=5,
                        capthick=1.4,
                        alpha=0.8,
                        label=f"{int(round(ci_level * 100))}% CI",
                    )
            else:
                for i, lev in enumerate(facet_order):
                    sub = emm[emm[facet_col] == lev].sort_values("_xi")
                    if sub.empty:
                        continue
                    c = emm_colors[str(lev)]
                    ax.plot(
                        sub["_xi"],
                        sub["estimate"],
                        color=c,
                        lw=2.4,
                        marker="o",
                        label=f"EMM {facet_col}={lev}",
                    )
                    if errorbar:
                        ax.errorbar(
                            sub["_xi"],
                            sub["estimate"],
                            yerr=[sub["estimate"] - sub["lower"], sub["upper"] - sub["estimate"]],
                            fmt="none",
                            ecolor=c,
                            elinewidth=1.3,
                            capsize=4,
                            capthick=1.3,
                            alpha=0.85,
                        )
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels([str(v) for v in order])
        except Exception as e:
            # The plot still shows the raw observations; record why the model curves are missing so
            # a caller (the GUI status line) can say so instead of leaving a silently empty chart.
            log.warning("Could not evaluate the EMM grid: %s", e)
            log.debug("EMM grid failure", exc_info=True)
            errors["emm_error"] = str(e)

        if include_points:
            # These are *observed cell means*, not individual rows: seaborn's pointplot averages y
            # within each (x, hue) cell. They are drawn in a lighter tone of the level's colour so
            # they read as data rather than as a second model curve.
            point_hue = hue if hue is not None else (group if group != x else None)
            if point_hue == hue and hue_order is not None:
                point_hue_order = list(hue_order)
            elif point_hue == group and group_order is not None:
                point_hue_order = list(group_order)
            elif point_hue is not None:
                point_hue_order = list(pd.unique(df[point_hue].dropna().astype(str)))
            else:
                point_hue_order = None
            # Seaborn spaces dodged levels by dividing the slot width across them, which is a
            # division by zero when only one level is left (e.g. after hiding every other group).
            # Nothing to dodge apart in that case anyway.
            n_hue_levels = len(point_hue_order) if point_hue_order is not None else 0
            if point_hue_order is not None:
                point_palette = {
                    lev: point_colors.get(str(lev), _lighten(base_colors[i % len(base_colors)]))
                    for i, lev in enumerate(point_hue_order)
                }
            else:
                point_palette = None
            sns.pointplot(
                data=df,
                x=x,
                y=y,
                hue=point_hue,
                order=order,
                hue_order=point_hue_order,
                palette=point_palette if point_palette else palette,
                color=None if point_hue else _lighten(base_colors[0]),
                errorbar=None,
                dodge=bool(point_hue) and n_hue_levels > 1,
                linestyles="--",
                markers="s",
                legend=False,
                ax=ax,
            )
            # One legend entry explaining the lighter series, rather than leaving twice as many
            # lines on the plot as the legend accounts for.
            ax.plot(
                [], [],
                color=_lighten(base_colors[0]),
                lw=2.0,
                ls="--",
                marker="s",
                label="Observed mean (unadjusted)",
            )

    return errors


def build_mixedlm_frame_from_repo(
    repo: Any,
    *,
    source: str,
    value_columns: list[str],
    id_columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    rename_map: dict[str, str] | None = None,
    melt_to_long: bool = False,
    var_name: str = "variable",
    value_name: str = "value",
    dropna_columns: list[str] | None = None,
    sex_column: str | None = None,
    center_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Lightweight adapter from DataRepo query outputs to model-ready dataframes.
    """
    id_columns = list(id_columns or ["subject_uid"])
    rename_map = dict(rename_map or {})
    center_columns = list(center_columns or [])

    if source == "clinical":
        frame = repo.clinical(wide=True, filters=filters or {}, cohort_id=False)
    elif source == "image":
        frame = repo.image(wide=True, filters=filters or {}, cohort_id=False)
    else:
        frame = repo.get(source, filters=filters or {}, wide=False, cohort_id=False)

    df = frame.copy()
    df.columns = [_safe_col(c) for c in df.columns]
    if rename_map:
        safe_map = {_safe_col(k): _safe_col(v) for k, v in rename_map.items()}
        df = df.rename(columns=safe_map)
        id_columns = [safe_map.get(_safe_col(c), _safe_col(c)) for c in id_columns]
        value_columns = [safe_map.get(_safe_col(c), _safe_col(c)) for c in value_columns]

    needed = id_columns + value_columns
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in repo-derived frame: {missing}")

    if melt_to_long:
        out = df.melt(id_vars=id_columns, value_vars=value_columns, var_name=var_name, value_name=value_name)
    else:
        out = df[id_columns + value_columns].copy()

    if sex_column is not None and sex_column in out.columns:
        out[sex_column] = _map_sex_to_numeric(out[sex_column])

    for col in center_columns:
        if col in out.columns:
            nums = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_c"] = nums - float(np.nanmean(nums))

    if dropna_columns:
        safe_drop = [_safe_col(c) for c in dropna_columns if _safe_col(c) in out.columns]
        if safe_drop:
            out = out.dropna(subset=safe_drop)
    return out.reset_index(drop=True)


__all__ = [
    "build_mixedlm_frame_from_repo",
    "fit_or_load_mixedlm",
    "formula_columns",
    "mixedlm_coef_frame",
    "mixedlm_info_dict",
    "mixedlm_random_effects_frame",
    "plot_mixedlm_params",
    "print_mixedlm_info",
    "render_mixedlm_info",
    "significance_stars",
]

