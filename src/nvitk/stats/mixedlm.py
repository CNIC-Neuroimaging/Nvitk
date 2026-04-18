"""General mixed-effects modeling utilities built on top of statsmodels."""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd


def _safe_col(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name))


def _map_sex_to_numeric(series: pd.Series) -> pd.Series:
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

    req = list(required_columns or [])
    if groups not in req:
        req.append(groups)
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for MixedLM: {missing}")

    if outcome_transform is not None:
        df = outcome_transform(df)

    na_cols = list(dropna_columns or req)
    na_cols = [c for c in na_cols if c in df.columns]
    if na_cols:
        df = df.dropna(subset=na_cols).reset_index(drop=True)

    metadata = {
        "model_path": str(path),
        "formula": formula,
        "groups": groups,
        "re_formula": re_formula,
        "vc_formula": vc_formula,
        "n_rows": int(len(df)),
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
    with redirect_stdout(buffer):
        print("=" * 88)
        print(f"Mixed Linear Model - {outcome_name}")
        print("=" * 88)
        try:
            print(f"Formula: {result.model.formula}")
        except Exception:
            pass
        print(f"Observations: {int(result.nobs):,}")
        ngroups = getattr(result, "ngroups", None)
        if ngroups is None and hasattr(result, "random_effects"):
            ngroups = len(result.random_effects)
        if ngroups is not None:
            print(f"{group_name} groups: {int(ngroups):,}")
        print()

        fe = result.fe_params
        fe_se = getattr(result, "bse_fe", getattr(result, "bse", pd.Series(dtype=float)))
        fe_p = getattr(result, "pvalues_fe", getattr(result, "pvalues", pd.Series(dtype=float)))
        print("Fixed effects")
        print("-" * 88)
        print(f"{'Parameter':<34}{'Coef':>12}{'Std.Err':>12}{'P-value':>12}{'Sig':>10}")
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
            print(f"{param:<34}{coef:>12.4f}{se:>12.4f}{pval:>12.4g}{sig:>10}")
        print()

        cov_re = getattr(result, "cov_re", pd.DataFrame())
        if isinstance(cov_re, pd.DataFrame) and not cov_re.empty:
            print(f"{group_name} random effects covariance")
            print("-" * 88)
            print(cov_re.to_string())
            print()

        vcomp = getattr(result, "vcomp", None)
        if vcomp is not None:
            vc_names = getattr(result.model, "vc_names", [f"VC_{i}" for i in range(len(vcomp))])
            print(f"Variance components ({vc_group_name})")
            print("-" * 88)
            for name, var in zip(vc_names, vcomp):
                var_f = float(var)
                print(f"{name:<24} var={var_f:.6f}  sd={np.sqrt(max(var_f, 0.0)):.6f}")
            print()

        if hasattr(result, "scale"):
            scale = float(result.scale)
            print(f"Residual variance: {scale:.6f}")
            print(f"Residual std dev: {np.sqrt(max(scale, 0.0)):.6f}")
            print()

        print("Fit statistics")
        print("-" * 88)
        for attr in ("llf", "aic", "bic"):
            if hasattr(result, attr):
                print(f"{attr.upper():<8}: {float(getattr(result, attr)):.4f}")
        if hasattr(result, "converged"):
            print(f"Converged: {bool(result.converged)}")
        print("=" * 88)

    text = buffer.getvalue()
    print(text)
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
    covariate_refs: dict[str, float] | None = None,
    output_path: str | Path | None = None,
    title: str = "MixedLM parameter plot",
    x_label: str | None = None,
    y_label: str | None = None,
) -> Any:
    """Generic plotting utility for continuous or grouped/categorical predictors."""
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

        if include_points:
            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                hue=hue if hue else group,
                hue_order=hue_order if hue else group_order,
                palette=palette,
                alpha=0.5,
                s=18,
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
        groups = group_order or sorted(df[group].dropna().astype(str).unique())
        colors = sns.color_palette(palette, n_colors=max(len(groups), 3))
        cmap = {g: colors[i % len(colors)] for i, g in enumerate(groups)}
        for g in groups:
            re_vals = re_dict.get(g, re_dict.get(str(g), None))
            if re_vals is None:
                continue
            re_int = float(re_vals.get("Group", re_vals.get("Intercept", 0.0)))
            re_x = float(re_vals.get(x, 0.0))
            y_line = (b_int + extra + re_int) + (b_x + re_x) * x_line
            ax.plot(x_line, y_line, color=cmap[g], lw=2, alpha=0.9, label=f"{group}={g}")

    else:
        order = categorical_order or list(pd.unique(df[x].dropna()))
        # EMM-like fixed-effects prediction grid at reference covariates.
        try:
            import patsy

            grid = pd.DataFrame({x: order})
            for cov_name, cov_val in covariate_refs.items():
                grid[cov_name] = cov_val
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
            ax.plot(emm[x], emm["estimate"], color="black", lw=2.6, marker="o", label="Fixed-effects EMM")
            ax.fill_between(emm[x], emm["lower"], emm["upper"], color="gray", alpha=0.2, label="95% CI")
        except Exception:
            pass

        if include_points:
            sns.pointplot(
                data=df,
                x=x,
                y=y,
                hue=hue,
                order=order,
                hue_order=hue_order,
                palette=palette,
                errorbar=None,
                dodge=True if hue else False,
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
    "print_mixedlm_info",
    "plot_mixedlm_params",
    "build_mixedlm_frame_from_repo",
]

