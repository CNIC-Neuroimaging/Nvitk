"""Forest plots of a fitted model: what each variable contributes, for any engine.

Every engine's report carries the same coefficient table (``parameter, coef, std_err, p_value,
ci_low, ci_high``: :func:`nvitk.stats.regression.model_info_dict`), so one set of figures serves
MixedLM, lme4, MMRM, OLS, robust and generalised linear models, the MRF smooth, path models and the
Bayesian hierarchical model alike:

* **coefficients**: each estimate with its confidence (or credible) interval, on the outcome's
  scale; significant terms highlighted;
* **standardized**: the same estimates in standard deviations of the outcome per standard
  deviation of the predictor (a factor's levels per SD of the outcome), so an age effect in years
  and a haematocrit effect in % can be compared at all;
* **contribution**: one bar per *variable* (a factor's levels and an interaction's terms together),
  sized by its joint Wald chi-square (the test that the whole variable can be dropped), with its p;
  the variable ranking that a coefficient-by-coefficient forest cannot give for multi-level factors;
* **ratios**: ``exp(coef)`` on a log axis: odds ratios for a logit link, rate ratios for a log link.

Styling (fonts, palette, sizes) is the plot pane's (:mod:`nvitk.gui.panels.statmodels.plot_style`).
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

#: Figure kinds: key -> (picker label, needs a design / data to compute).
FOREST_KINDS: dict[str, str] = {
    "forest": "Forest · coefficients",
    "forest_std": "Forest · standardized effects",
    "forest_contrib": "Forest · variable contribution",
    "forest_ratio": "Forest · odds / rate ratios",
}
_INTERCEPT = {"intercept", "(intercept)", "const"}


def is_intercept(name: str) -> bool:
    return str(name).strip().lower() in _INTERCEPT


def term_of(parameter: str) -> str:
    """The variable a coefficient belongs to: a factor's level and an interaction's cell are folded
    into their term (``C(sex)[T.M]`` -> ``C(sex)``, ``age:C(sex)[T.M]`` -> ``age:C(sex)``,
    ``territory[T.LICA]`` -> ``territory``)."""
    parts = []
    for piece in str(parameter).split(":"):
        piece = re.sub(r"\[(T\.)?[^\]]*\]$", "", piece.strip())
        parts.append(piece)
    return ":".join(parts)


def _tidy(table: pd.DataFrame) -> pd.DataFrame:
    need = {"parameter", "coef"}
    if table is None or table.empty or not need <= set(table.columns):
        raise ValueError("This model has no coefficient table to plot.")
    out = table.copy()
    for col in ("coef", "std_err", "p_value", "ci_low", "ci_high"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan
    missing_ci = out["ci_low"].isna() & out["std_err"].notna()
    out.loc[missing_ci, "ci_low"] = out.loc[missing_ci, "coef"] - 1.959964 * out.loc[missing_ci, "std_err"]
    out.loc[missing_ci, "ci_high"] = out.loc[missing_ci, "coef"] + 1.959964 * out.loc[missing_ci, "std_err"]
    out["parameter"] = out["parameter"].astype(str)
    return out[out["coef"].notna()].reset_index(drop=True)


def _design_sd(result: Any, data: pd.DataFrame | None, parameters: Sequence[str],
               formula: str | None = None) -> dict[str, float]:
    """The standard deviation of each coefficient's design column: from the fitted design when the
    result keeps it, else rebuilt from the formula on *data*, else a numeric column of that name."""
    out: dict[str, float] = {}
    exog = getattr(getattr(result, "model", None), "exog", None)
    names = getattr(getattr(result, "model", None), "exog_names", None)
    if exog is not None and names is not None:
        sd = np.asarray(exog, dtype=float).std(axis=0, ddof=1)
        out.update({str(n): float(v) for n, v in zip(names, sd)})
    if formula and data is not None and len(out) < len(parameters):
        try:
            from patsy import dmatrix

            rhs = re.sub(r"\([^()]*\|[^()]*\)", "", formula.split("~", 1)[-1])
            rhs = re.sub(r"\+\s*\+", "+", rhs).strip().strip("+").strip() or "1"
            design = dmatrix(rhs, data, return_type="dataframe")
            for col in design.columns:
                out.setdefault(str(col), float(design[col].std(ddof=1)))
        except Exception:  # noqa: BLE001 (R-style names, terms patsy cannot read)
            pass
    if data is not None:
        for name in parameters:
            if name not in out and name in data.columns and pd.api.types.is_numeric_dtype(data[name]):
                out[name] = float(data[name].std(ddof=1))
    return out


def standardized_table(table: pd.DataFrame, *, result: Any = None, data: pd.DataFrame | None = None,
                       outcome: str | None = None, formula: str | None = None) -> tuple[pd.DataFrame, str]:
    """*table* rescaled to SD(y) per SD(x); returns it and a note on what could not be scaled."""
    tidy = _tidy(table)
    if data is None or not outcome or outcome not in data.columns:
        raise ValueError("Standardizing needs the outcome column in the model's data.")
    sd_y = float(pd.to_numeric(data[outcome], errors="coerce").std(ddof=1))
    if not np.isfinite(sd_y) or sd_y <= 0:
        raise ValueError("The outcome has no spread to standardize by.")
    sd_x = _design_sd(result, data, list(tidy["parameter"]), formula)
    factor = np.array([sd_x.get(p, np.nan) / sd_y for p in tidy["parameter"]])
    unscaled = [p for p, f in zip(tidy["parameter"], factor) if not np.isfinite(f) and not is_intercept(p)]
    factor = np.where(np.isfinite(factor), factor, 1.0 / sd_y)
    for col in ("coef", "ci_low", "ci_high", "std_err"):
        tidy[col] = tidy[col] * factor
    note = ""
    if unscaled:
        note = (f"{len(unscaled)} term(s) scaled by SD(y) only (no design column found): "
                + ", ".join(unscaled[:4]) + ("…" if len(unscaled) > 4 else ""))
    return tidy, note


def contribution_table(table: pd.DataFrame, *, result: Any = None, include_intercept: bool = False) -> pd.DataFrame:
    """One row per variable: its joint Wald chi-square (``b' V^-1 b`` over the variable's
    coefficients), degrees of freedom and p. Uses the result's covariance when it has one, else
    the diagonal (independent coefficients: the sum of their z squared)."""
    from scipy.stats import chi2

    tidy = _tidy(table)
    cov = None
    try:
        cov_fn = getattr(result, "cov_params", None)
        cov = cov_fn() if callable(cov_fn) else None
        if cov is not None and not isinstance(cov, pd.DataFrame):
            cov = None
    except Exception:  # noqa: BLE001
        cov = None
    rows = []
    tidy["term"] = tidy["parameter"].map(term_of)
    for term, group in tidy.groupby("term", sort=False):
        if not include_intercept and all(is_intercept(p) for p in group["parameter"]):
            continue
        b = group["coef"].to_numpy(dtype=float)
        names = list(group["parameter"])
        stat = np.nan
        exact = False
        if cov is not None and all(n in cov.index for n in names):
            v = cov.loc[names, names].to_numpy(dtype=float)
            try:
                stat = float(b @ np.linalg.solve(v, b))
                exact = True
            except np.linalg.LinAlgError:
                stat = np.nan
        if not np.isfinite(stat):
            se = group["std_err"].to_numpy(dtype=float)
            stat = float(np.nansum((b / se) ** 2)) if np.isfinite(se).any() else np.nan
        df = len(b)
        rows.append({"term": term, "chi2": stat, "df": df, "p_value": float(chi2.sf(stat, df)) if np.isfinite(stat) else np.nan,
                     "n_coef": df, "joint": exact, "max_abs_coef": float(np.nanmax(np.abs(b)))})
    return pd.DataFrame(rows)


def _short(name: str, width: int = 38) -> str:
    text = re.sub(r"C\(([^,)]+)(,[^)]*)?\)", r"\1", str(name))
    text = text.replace("[T.", "[")
    return text if len(text) <= width else text[: width - 1] + "…"


def _sort(tidy: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "effect":
        return tidy.sort_values("coef")
    if how == "abs":
        return tidy.assign(_a=tidy["coef"].abs()).sort_values("_a").drop(columns="_a")
    if how == "p":
        return tidy.sort_values("p_value", ascending=False)
    return tidy.iloc[::-1]


def forest_figure(
    table: pd.DataFrame,
    kind: str = "forest",
    *,
    result: Any = None,
    data: pd.DataFrame | None = None,
    outcome: str | None = None,
    formula: str | None = None,
    include_intercept: bool = False,
    sort: str = "model",
    alpha: float = 0.05,
    show_values: bool = True,
    max_terms: int = 60,
    title: str | None = None,
    interval_label: str = "95% CI",
) -> Any:
    """The forest figure of *kind* (:data:`FOREST_KINDS`) for a coefficient *table*."""
    import matplotlib.pyplot as plt

    if kind == "forest_contrib":
        return _contribution_figure(table, result=result, include_intercept=include_intercept, alpha=alpha,
                                    sort=sort, title=title)
    note = ""
    tidy = _tidy(table)
    x_label = "Estimate"
    ref = 0.0
    log_x = False
    if kind == "forest_std":
        tidy, note = standardized_table(tidy, result=result, data=data, outcome=outcome, formula=formula)
        x_label = "Standardized effect (SD of the outcome per SD of the predictor)"
    elif kind == "forest_ratio":
        for col in ("coef", "ci_low", "ci_high"):
            tidy[col] = np.exp(tidy[col].clip(-50, 50))
        x_label = "Ratio, exp(estimate) (odds ratio for logit links, rate ratio for log links)"
        ref, log_x = 1.0, True
    if not include_intercept:
        tidy = tidy[~tidy["parameter"].map(is_intercept)]
    if tidy.empty:
        raise ValueError("No coefficients to plot (only an intercept).")
    tidy = _sort(tidy, sort)
    dropped = 0
    if len(tidy) > max_terms:
        dropped = len(tidy) - max_terms
        tidy = tidy.assign(_r=tidy["p_value"].fillna(1)).nsmallest(max_terms, "_r").drop(columns="_r")
        tidy = _sort(tidy, sort)
    signif = (tidy["ci_low"] > ref) | (tidy["ci_high"] < ref)
    n = len(tidy)
    fig, ax, side = _forest_axes(n, show_values)
    pos = np.arange(n)
    colors = np.where(signif, "C3", "C0")
    lo = (tidy["coef"] - tidy["ci_low"]).to_numpy()
    hi = (tidy["ci_high"] - tidy["coef"]).to_numpy()
    for i in range(n):
        ax.errorbar(tidy["coef"].iloc[i], pos[i], xerr=[[max(lo[i], 0)], [max(hi[i], 0)]], fmt="none",
                    ecolor=colors[i], elinewidth=1.4, capsize=3)
    ax.scatter(tidy["coef"], pos, c=list(colors), s=36, zorder=3, edgecolors="white", linewidths=0.6)
    ax.axvline(ref, color="grey", linestyle="--", linewidth=0.9)
    ax.set_yticks(pos)
    ax.set_yticklabels([_short(p) for p in tidy["parameter"]])
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    if side is not None:
        for i, (_, row) in enumerate(tidy.iterrows()):
            p = row.get("p_value")
            ptxt = "" if not np.isfinite(p) else ("p<0.001" if p < 0.001 else f"p={p:.3f}")
            side.text(0.0, pos[i], f"{row['coef']:.3g} [{row['ci_low']:.3g}, {row['ci_high']:.3g}]  {ptxt}",
                      va="center", fontsize=8)
    from matplotlib.lines import Line2D

    ax.legend(handles=[Line2D([], [], marker="o", color="C3", linestyle="", label=f"{interval_label} excludes {ref:g}"),
                       Line2D([], [], marker="o", color="C0", linestyle="", label="includes it")],
              loc="lower right", fontsize=8, frameon=False)
    head = title or {"forest": "Coefficients", "forest_std": "Standardized effects",
                     "forest_ratio": "Odds / rate ratios"}.get(kind, "Forest plot")
    ax.set_title(f"{head} ({interval_label})")
    extra = []
    if dropped:
        extra.append(f"{dropped} term(s) with the largest p not shown")
    if note:
        extra.append(note)
    if extra:
        fig.text(0.01, 0.005, "; ".join(extra), fontsize=7, color="grey", ha="left", va="bottom")
    fig.forest_note = note
    return fig


def _forest_axes(n: int, show_values: bool) -> tuple[Any, Any, Any]:
    """The forest's axes and, with *show_values*, a text column beside it sharing its rows: laid
    out together, so neither the labels nor the numbers are clipped at the figure's edge."""
    import matplotlib.pyplot as plt

    height = max(2.4, 0.32 * n + 1.3)
    if not show_values:
        fig, ax = plt.subplots(figsize=(7, height), layout="constrained")
        return fig, ax, None
    fig, (ax, side) = plt.subplots(1, 2, figsize=(10, height), sharey=True, layout="constrained",
                                   gridspec_kw={"width_ratios": [3, 1.6]})
    side.set_xlim(0, 1)
    side.axis("off")
    side.tick_params(left=False, labelleft=False)
    side.set_title("estimate [interval]  p", fontsize=8, loc="left")
    return fig, ax, side


def _contribution_figure(table: pd.DataFrame, *, result: Any = None, include_intercept: bool = False,
                         alpha: float = 0.05, sort: str = "model", title: str | None = None) -> Any:
    import matplotlib.pyplot as plt

    contrib = contribution_table(table, result=result, include_intercept=include_intercept)
    if contrib.empty:
        raise ValueError("No variables to rank (only an intercept).")
    contrib = contrib.sort_values("chi2") if sort != "model" else contrib.iloc[::-1]
    n = len(contrib)
    fig, ax, side = _forest_axes(n, True)
    side.set_title("χ²  p", fontsize=8, loc="left")
    pos = np.arange(n)
    signif = contrib["p_value"] < alpha
    ax.barh(pos, contrib["chi2"], color=np.where(signif, "C3", "C0"), alpha=0.85)
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{_short(t)}  (df {d})" for t, d in zip(contrib["term"], contrib["df"])])
    for i, (_, row) in enumerate(contrib.iterrows()):
        p = row["p_value"]
        ptxt = "" if not np.isfinite(p) else ("p<0.001" if p < 0.001 else f"p={p:.3f}")
        side.text(0.0, pos[i], f"{row['chi2']:.2f}  {ptxt}", va="center", fontsize=8)
    ax.set_xlabel("Wald χ² of the variable (all its coefficients together)")
    approx = not bool(contrib["joint"].all())
    ax.set_title(title or "Variable contribution" + (" (diagonal approximation)" if approx else ""))
    fig.forest_note = ("Coefficient covariance not available for every variable: χ² is the sum of the "
                       "squared z of its coefficients (independent coefficients assumed)." if approx else "")
    return fig


__all__ = [
    "FOREST_KINDS",
    "contribution_table",
    "forest_figure",
    "is_intercept",
    "standardized_table",
    "term_of",
]
