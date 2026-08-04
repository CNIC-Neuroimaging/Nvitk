"""General mixed-effects modeling utilities built on top of statsmodels."""
# """TODO: GPU implementation Cupy + cuDF"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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


def print_mixedlm_info(
    result: Any,
    *,
    outcome_name: str = "Outcome",
    group_name: str = "Group",
    vc_group_name: str = "Variance Components",
    output_path: str | Path | None = None,
) -> str:
    """Print and optionally save a generic MixedLM summary report."""
    required_attrs = ["fe_params", "summary", "nobs"]
    missing = [a for a in required_attrs if not hasattr(result, a)]
    if missing:
        raise ValueError(f"Object does not look like MixedLMResults. Missing attributes: {missing}")

    buffer = io.StringIO()
    _w = lambda line="": buffer.write(f"{line}\n")

    _w("=" * 88)
    _w(f"Mixed Linear Model - {outcome_name}")
    _w("=" * 88)
    try:
        _w(f"Formula: {result.model.formula}")
    except Exception:
        pass
    _w(f"Observations: {int(result.nobs):,}")
    ngroups = getattr(result, "ngroups", None)
    if ngroups is None and hasattr(result, "random_effects"):
        ngroups = len(result.random_effects)
    if ngroups is not None:
        _w(f"{group_name} groups: {int(ngroups):,}")
    _w()

    fe = result.fe_params
    fe_se = getattr(result, "bse_fe", getattr(result, "bse", pd.Series(dtype=float)))
    fe_p = getattr(result, "pvalues_fe", getattr(result, "pvalues", pd.Series(dtype=float)))
    _w("Fixed effects")
    _w("-" * 88)
    _w(f"{'Parameter':<34}{'Coef':>12}{'Std.Err':>12}{'P-value':>12}{'Sig':>10}")
    for param in fe.index:
        coef = float(fe[param])
        se = float(fe_se.get(param, np.nan))
        pval = float(fe_p.get(param, np.nan))
        if np.isnan(pval):
            sig = ""
        elif pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        elif pval < 0.1:
            sig = "."
        else:
            sig = "NS"
        _w(f"{param:<34}{coef:>12.4f}{se:>12.4f}{pval:>12.4g}{sig:>10}")
    _w()

    cov_re = getattr(result, "cov_re", pd.DataFrame())
    if isinstance(cov_re, pd.DataFrame) and not cov_re.empty:
        _w(f"{group_name} random effects covariance")
        _w("-" * 88)
        _w(cov_re.to_string())
        _w()

    vcomp = getattr(result, "vcomp", None)
    if vcomp is not None:
        vc_names = getattr(result.model, "vc_names", [f"VC_{i}" for i in range(len(vcomp))])
        _w(f"Variance components ({vc_group_name})")
        _w("-" * 88)
        for name, var in zip(vc_names, vcomp):
            var_f = float(var)
            _w(f"{name:<24} var={var_f:.6f}  sd={np.sqrt(max(var_f, 0.0)):.6f}")
        _w()

    if hasattr(result, "scale"):
        scale = float(result.scale)
        _w(f"Residual variance: {scale:.6f}")
        _w(f"Residual std dev: {np.sqrt(max(scale, 0.0)):.6f}")
        _w()

    _w("Fit statistics")
    _w("-" * 88)
    for attr in ("llf", "aic", "bic"):
        if hasattr(result, attr):
            _w(f"{attr.upper():<8}: {float(getattr(result, attr)):.4f}")
    if hasattr(result, "converged"):
        _w(f"Converged: {bool(result.converged)}")
    _w("=" * 88)

    text = buffer.getvalue()
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
    errorbar: bool = False,
    covariate_refs: dict[str, float] | None = None,
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

    fig, ax = plt.subplots(figsize=(10, 6))

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

        fe = result.fe_params
        re_dict = getattr(result, "random_effects", {})
        b_int = float(fe.get("Intercept", fe.get("const", 0.0)))
        b_x = float(fe.get(x, 0.0))
        extra = 0.0
        for cov_name, cov_val in covariate_refs.items():
            extra += float(fe.get(cov_name, 0.0)) * float(cov_val)

        ax.plot(x_line, b_int + extra + b_x * x_line, color="black", lw=2.8, ls="--", label="Fixed effect")
        for g in groups:
            re_vals = re_dict.get(g, re_dict.get(str(g), None))
            if re_vals is None:
                continue
            re_int = float(re_vals.get("Group", re_vals.get("Intercept", 0.0)))
            re_x = float(re_vals.get(x, 0.0))
            y_line = (b_int + extra + re_int) + (b_x + re_x) * x_line
            ax.plot(x_line, y_line, color=cmap[g], lw=2, alpha=0.9, label=f"{group}={g}")
            # if errorbar:
            #     ax.fill_between(x_line, y_line - 1.96 * se, y_line + 1.96 * se, color=cmap[g], alpha=0.2)

    else:
        order = list(categorical_order) if categorical_order is not None else list(pd.unique(df[x].dropna()))
        # Second factor: interaction models need a full factorial x × (hue|group) grid
        # so patsy can evaluate e.g. C(x) * C(territory); a single-column grid raises
        # NameError for the missing factor.
        facet_col: str | None = None
        if hue is not None and hue != x:
            facet_col = hue
        elif group is not None and group != x:
            facet_col = group

        # EMM-like fixed-effects prediction grid at reference covariates.
        try:
            import patsy

            if facet_col is not None:
                if facet_col == hue:
                    facet_order = (
                        list(hue_order)
                        if hue_order is not None
                        else list(pd.unique(df[facet_col].dropna().astype(str)))
                    )
                elif facet_col == group:
                    facet_order = (
                        list(group_order)
                        if group_order is not None
                        else list(pd.unique(df[facet_col].dropna().astype(str)))
                    )
                else:
                    facet_order = list(pd.unique(df[facet_col].dropna().astype(str)))
                rows = [(xv, fv) for xv in order for fv in facet_order]
                grid = pd.DataFrame(rows, columns=[x, facet_col])
            else:
                grid = pd.DataFrame({x: order})

            for cov_name, cov_val in covariate_refs.items():
                grid[cov_name] = cov_val

            grid = _match_grid_columns_to_df_dtypes(grid, df, [c for c in grid.columns if c in df.columns])
            X = patsy.build_design_matrices([result.model.data.design_info], grid, return_type="dataframe")[0]
            fe = result.fe_params
            cov = result.cov_params().loc[fe.index, fe.index]
            X = X.reindex(columns=fe.index, fill_value=0.0)
            pred = X @ fe
            se = np.sqrt(np.sum((X @ cov) * X, axis=1))
            emm = grid.copy()
            emm["estimate"] = pred
            emm["lower"] = pred - 1.96 * se
            emm["upper"] = pred + 1.96 * se

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
                    ax.fill_between(
                        sub["_xi"],
                        sub["lower"],
                        sub["upper"],
                        color="gray",
                        alpha=0.2,
                        label="95% CI",
                    )
            else:
                colors = sns.color_palette(palette, n_colors=max(len(facet_order), 3))
                for i, lev in enumerate(facet_order):
                    sub = emm[emm[facet_col] == lev].sort_values("_xi")
                    if sub.empty:
                        continue
                    c = colors[i % len(colors)]
                    ax.plot(
                        sub["_xi"],
                        sub["estimate"],
                        color=c,
                        lw=2.4,
                        marker="o",
                        label=f"EMM {facet_col}={lev}",
                    )
                    if errorbar:
                        ax.fill_between(
                            sub["_xi"],
                            sub["lower"],
                            sub["upper"],
                            color=c,
                            alpha=0.12,
                        )
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels([str(v) for v in order])
        except Exception as e:
            import traceback
            traceback.print_exc()
            log.error(f"Error calculating EMM: {e}")

        if include_points:
            # Color raw observations like the EMM / territory lines, but keep them out of the
            # legend so it only lists the model curves.
            point_hue = hue if hue is not None else (group if group != x else None)
            if point_hue == hue and hue_order is not None:
                point_hue_order = list(hue_order)
            elif point_hue == group and group_order is not None:
                point_hue_order = list(group_order)
            elif point_hue is not None:
                point_hue_order = list(pd.unique(df[point_hue].dropna().astype(str)))
            else:
                point_hue_order = None
            sns.pointplot(
                data=df,
                x=x,
                y=y,
                hue=point_hue,
                order=order,
                hue_order=point_hue_order,
                palette=palette,
                errorbar=None,
                dodge=True if point_hue else False,
                legend=False,
                ax=ax,
            )

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

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig


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
    "fit_or_load_mixedlm",
    "formula_columns",
    "print_mixedlm_info",
    "plot_mixedlm_params",
    "build_mixedlm_frame_from_repo",
]

