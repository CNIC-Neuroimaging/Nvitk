"""
Model → plot geometry, so every engine renders through one interactive path.

Description
-----------
:mod:`~nvitk.stats.interactive` draws *frames*: observations, fitted lines, confidence ribbons,
marginal means. This module produces those frames from a fitted model, one adapter per engine, so
the renderer never learns what a MixedLM is and the engines never learn what Plotly is.

Every adapter returns the same :class:`PlotGeometry`, which is what makes the grouped anatomical
display, the QC grey-out and the hover payloads work identically whichever engine ran.

Why not reuse the matplotlib plotters
-------------------------------------
Those compute and draw in one pass — the numbers only ever exist inside a drawing call. Splitting
the computation out is what lets the same fit feed an interactive figure, a static one, or a table,
and it is why the geometry is returned rather than rendered.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

#: How many x values a fitted curve is evaluated at.
LINE_RESOLUTION = 160


@dataclass
class PlotGeometry:
    """Everything :func:`~nvitk.stats.interactive.model_plot` needs, engine-independent."""

    points: pd.DataFrame | None = None
    lines: pd.DataFrame | None = None
    bands: pd.DataFrame | None = None
    marginal_means: pd.DataFrame | None = None
    mode: str = "continuous"
    categorical_order: list[str] = field(default_factory=list)
    #: Populated when the model curves could not be built; the plot still shows the observations.
    error: str = ""

    def is_empty(self) -> bool:
        """Whether there is nothing at all to draw."""
        return all(
            f is None or f.empty
            for f in (self.points, self.lines, self.bands, self.marginal_means)
        )


def _x_grid(values: pd.Series, n: int = LINE_RESOLUTION) -> np.ndarray:
    """Evenly spaced grid spanning the observed x range."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        raise ValueError("The x column has no numeric values.")
    return np.linspace(float(numeric.min()), float(numeric.max()), n)


def _categorical_order(df: pd.DataFrame, x: str, order: Sequence[str] | None) -> list[str]:
    """Categorical x order — an explicit one, an ordered dtype's own, else a natural sort."""
    from .region_groups import natural_level_key

    if order is not None:
        return [str(v) for v in order]
    if isinstance(df[x].dtype, pd.CategoricalDtype) and df[x].dtype.ordered:
        present = set(df[x].dropna().astype(str))
        return [str(c) for c in df[x].dtype.categories if str(c) in present]
    return sorted({str(v) for v in df[x].dropna()}, key=natural_level_key)


def _reference_row(df: pd.DataFrame, exclude: Sequence[str]) -> dict[str, Any]:
    """Covariates held at their mean (numeric) or modal level (categorical)."""
    out: dict[str, Any] = {}
    for column in df.columns:
        if column in set(exclude):
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        out[column] = (
            float(pd.to_numeric(series, errors="coerce").mean())
            if pd.api.types.is_numeric_dtype(series) else series.mode().iloc[0]
        )
    return out


# ---------------------------------------------------------------------------
# statsmodels: MixedLM / OLS / GLM
# ---------------------------------------------------------------------------
def statsmodels_geometry(
    result: Any,
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str = "",
    mode: str = "auto",
    group_order: Sequence[str] | None = None,
    categorical_order: Sequence[str] | None = None,
    covariate_refs: Mapping[str, Any] | None = None,
    ci_level: float = 0.95,
    errorbar: bool = False,
) -> PlotGeometry:
    """
    Geometry for a statsmodels fit — MixedLM, OLS or GLM.

    Continuous mode gives one line per group (fixed effects plus that group's random offset, or its
    own fixed-effects prediction when the model has no random part) and the dashed population line.
    Categorical mode gives estimated marginal means from a patsy grid at *covariate_refs*.

    The slope and intercept of each line are carried alongside it so the renderer can put them in
    the hover box — reading a slope off a plot is otherwise guesswork.
    """
    from .mixedlm import (
        _prediction_standard_errors,
        _z_critical,
        model_inverse_link,
        model_params,
        model_random_effects,
    )

    for column in (x, y):
        if column not in df.columns:
            raise ValueError(f"Column {column!r} is not in the fitted frame.")
    if mode == "auto":
        mode = (
            "categorical"
            if categorical_order is not None or not pd.api.types.is_numeric_dtype(df[x])
            else "continuous"
        )

    grouped = bool(group) and group in df.columns and group != x
    levels = (
        [str(v) for v in group_order] if group_order is not None
        else sorted({str(v) for v in df[group].dropna()}) if grouped else []
    )
    geometry = PlotGeometry(points=df, mode=mode)

    if mode == "categorical":
        geometry.categorical_order = _categorical_order(df, x, categorical_order)
        geometry.marginal_means, geometry.error = _statsmodels_emm(
            result, df, x=x, group=group if grouped else "", levels=levels,
            order=geometry.categorical_order, covariate_refs=covariate_refs or {},
            ci_level=ci_level,
        )
        return geometry

    # ---- continuous ---------------------------------------------------------------
    grid = _x_grid(df[x])
    fixed = model_params(result)
    random = model_random_effects(result)
    inverse = model_inverse_link(result)
    refs = dict(covariate_refs or {})

    intercept = float(fixed.get("Intercept", fixed.get("const", 0.0)))
    slope = float(fixed.get(x, 0.0))
    offset = sum(
        float(fixed.get(name, 0.0)) * float(value)
        for name, value in refs.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    )

    rows: list[pd.DataFrame] = []
    band_rows: list[pd.DataFrame] = []
    population = intercept + offset + slope * grid
    rows.append(pd.DataFrame({
        group or "group": "", x: grid, y: inverse(population),
        "slope": slope, "intercept": intercept + offset,
    }))

    predictions: dict[str | None, tuple[np.ndarray, np.ndarray]] = {}
    try:
        predictions = _prediction_standard_errors(
            result, df, x=x, x_values=grid, group=group if grouped else "",
            levels=levels, covariate_refs=refs,
        )
    except Exception as exc:
        if errorbar:
            geometry.error = f"Confidence band unavailable: {exc}"
            log.debug("Prediction standard errors failed", exc_info=True)

    critical = _z_critical(ci_level)
    if errorbar and predictions.get(None) is not None:
        _pred, se = predictions[None]
        band_rows.append(pd.DataFrame({
            group or "group": "", x: grid,
            "lower": inverse(population - critical * se),
            "upper": inverse(population + critical * se),
        }))

    # Does the grouping factor move the prediction at all? If every level predicts the same curve
    # it is not in the model, and one identical line per level is pure clutter.
    level_predictions = {k: v for k, v in predictions.items() if k is not None}
    informative = len(level_predictions) > 1 and not all(
        np.allclose(next(iter(level_predictions.values()))[0], pred)
        for pred, _se in level_predictions.values()
    )

    for level in levels:
        effects = random.get(level, random.get(str(level)))
        entry = level_predictions.get(str(level))
        if effects is not None:
            level_intercept = intercept + offset + float(
                effects.get("Group", effects.get("Intercept", 0.0))
            )
            level_slope = slope + float(effects.get(x, 0.0))
            eta = level_intercept + level_slope * grid
        elif informative and entry is not None:
            eta = entry[0]
            level_slope, level_intercept = np.nan, np.nan
        else:
            continue
        rows.append(pd.DataFrame({
            group or "group": str(level), x: grid, y: inverse(eta),
            "slope": level_slope, "intercept": level_intercept,
        }))
        if errorbar and entry is not None:
            band_rows.append(pd.DataFrame({
                group or "group": str(level), x: grid,
                "lower": inverse(eta - critical * entry[1]),
                "upper": inverse(eta + critical * entry[1]),
            }))

    geometry.lines = pd.concat(rows, ignore_index=True) if rows else None
    geometry.bands = pd.concat(band_rows, ignore_index=True) if band_rows else None
    return geometry


def _statsmodels_emm(
    result: Any,
    df: pd.DataFrame,
    *,
    x: str,
    group: str,
    levels: Sequence[str],
    order: Sequence[str],
    covariate_refs: Mapping[str, Any],
    ci_level: float,
) -> tuple[pd.DataFrame | None, str]:
    """
    Estimated marginal means over the categorical x, from a patsy grid at reference covariates.

    An interaction model needs the full factorial ``x × group`` grid: a single-column grid raises a
    ``NameError`` for the missing factor and takes the marginal means down with it.
    """
    from .mixedlm import _match_grid_columns_to_df_dtypes, _z_critical, model_inverse_link, model_params

    try:
        import patsy

        facet = group if group and group != x else ""
        if facet:
            grid = pd.DataFrame(
                [(xv, fv) for xv in order for fv in levels], columns=[x, facet]
            )
        else:
            grid = pd.DataFrame({x: list(order)})
        for name, value in covariate_refs.items():
            if name not in {x, facet}:
                grid[name] = value
        grid = _match_grid_columns_to_df_dtypes(
            grid, df, [c for c in grid.columns if c in df.columns]
        )

        design = patsy.build_design_matrices(
            [result.model.data.design_info], grid, return_type="dataframe"
        )[0]
        fixed = model_params(result)
        covariance = result.cov_params().loc[fixed.index, fixed.index]
        design = design.reindex(columns=fixed.index, fill_value=0.0)
        estimate = design @ fixed
        se = np.sqrt(np.sum((design @ covariance) * design, axis=1))

        inverse = model_inverse_link(result)
        critical = _z_critical(ci_level)
        out = grid.copy()
        out["estimate"] = inverse(estimate)
        out["se"] = se
        out["lower"] = inverse(estimate - critical * se)
        out["upper"] = inverse(estimate + critical * se)
        if not facet:
            out[group or "group"] = ""
        return out, ""
    except Exception as exc:
        log.warning("Could not evaluate the marginal-means grid: %s", exc)
        log.debug("EMM grid failure", exc_info=True)
        return None, str(exc)


# ---------------------------------------------------------------------------
# R engines: lme4 and lmrob, which predict through R
# ---------------------------------------------------------------------------
def r_model_geometry(
    model: Any,
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str = "",
    mode: str = "auto",
    group_order: Sequence[str] | None = None,
    predict_fn: Any = None,
    band_fn: Any = None,
    fixed_formula: str = "",
    ci_level: float = 0.95,
    errorbar: bool = False,
) -> PlotGeometry:
    """
    Geometry for an engine whose parameters are named by R rather than by patsy.

    Predictions round-trip through R for the same reason the static plots do: R names a factor
    contrast ``territoryPCA`` where patsy names it ``C(territory)[T.PCA]``, so a locally rebuilt
    design matrix would silently zero every contrast.
    """
    if mode == "auto":
        mode = "continuous" if pd.api.types.is_numeric_dtype(df[x]) else "categorical"
    grouped = bool(group) and group in df.columns and group != x
    levels = (
        [str(v) for v in group_order] if group_order is not None
        else sorted({str(v) for v in df[group].dropna()}) if grouped else []
    )

    geometry = PlotGeometry(points=df, mode=mode)
    if mode == "categorical":
        geometry.categorical_order = _categorical_order(df, x, None)
        x_values = np.array(geometry.categorical_order, dtype=object)
    else:
        x_values = _x_grid(df[x])

    base = _reference_row(df, exclude=(x, y))

    def predict(level: str | None) -> np.ndarray | None:
        """Prediction along the grid for one level, or the population."""
        grid = pd.DataFrame([{**base, x: value} for value in x_values])
        if grouped and level is not None:
            grid[group] = str(level)
        try:
            return predict_fn(model, grid, use_random_effects=level is not None)
        except Exception as exc:
            log.debug("Prediction failed for %s=%s: %s", group, level, exc)
            return None

    rows: list[pd.DataFrame] = []
    population = predict(None)
    if population is not None:
        rows.append(pd.DataFrame({group or "group": "", x: x_values, y: population}))
    for level in levels:
        curve = predict(level)
        if curve is not None:
            rows.append(pd.DataFrame({group or "group": str(level), x: x_values, y: curve}))
    geometry.lines = pd.concat(rows, ignore_index=True) if rows else None

    if errorbar and band_fn is not None:
        bands = band_fn(
            model, x=x, x_values=x_values, group=group if grouped else "", levels=levels,
            continuous=mode == "continuous", fixed_formula=fixed_formula, ci_level=ci_level,
        )
        if bands is None:
            geometry.error = (
                "emmeans could not produce marginal means, so no confidence band is shown."
            )
        else:
            geometry.bands = _bands_from_emmeans(bands, x=x, group=group or "group")
    return geometry


def _bands_from_emmeans(
    bands: Mapping[Any, pd.DataFrame], *, x: str, group: str
) -> pd.DataFrame | None:
    """Flatten emmeans' per-level frames into the renderer's ``group / x / lower / upper`` shape."""
    from .r_mixedlm import _emmeans_columns

    rows: list[pd.DataFrame] = []
    for level, frame in bands.items():
        parts = _emmeans_columns(frame)
        if parts is None or x not in frame.columns:
            continue
        _estimate, lower, upper = parts
        rows.append(pd.DataFrame({
            group: "" if level is None else str(level),
            x: frame[x].to_numpy(),
            "lower": pd.to_numeric(frame[lower], errors="coerce").to_numpy(),
            "upper": pd.to_numeric(frame[upper], errors="coerce").to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True) if rows else None


# ---------------------------------------------------------------------------
# MMRM and non-linear, which already produce their geometry
# ---------------------------------------------------------------------------
def mmrm_geometry(
    emmeans_frame: pd.DataFrame, *, x: str, hue: str = ""
) -> PlotGeometry:
    """
    Geometry for an MMRM: its least-squares means are already the plot.

    ``emmeans`` names its columns by convention rather than by contract, so the estimate and the
    interval are located by trying the documented spellings in turn.
    """
    from .region_groups import natural_level_key

    frame = emmeans_frame.copy()
    estimate = next((c for c in ("emmean", "estimate", "response") if c in frame.columns), None)
    if estimate is None or x not in frame.columns:
        raise ValueError(
            f"Cannot plot: need {x!r} and an estimate column in {list(frame.columns)}."
        )
    lower = next((c for c in ("lower.CL", "asymp.LCL", "lower") if c in frame.columns), None)
    upper = next((c for c in ("upper.CL", "asymp.UCL", "upper") if c in frame.columns), None)
    se = next((c for c in ("SE", "se", "std.error") if c in frame.columns), None)

    out = pd.DataFrame({
        x: frame[x].astype(str),
        "estimate": pd.to_numeric(frame[estimate], errors="coerce"),
    })
    out[hue or "group"] = frame[hue].astype(str) if hue and hue in frame.columns else ""
    for source, target in ((lower, "lower"), (upper, "upper"), (se, "se")):
        if source:
            out[target] = pd.to_numeric(frame[source], errors="coerce")

    return PlotGeometry(
        marginal_means=out,
        mode="categorical",
        categorical_order=sorted(set(out[x]), key=natural_level_key),
    )


def nonlinear_geometry(
    result: Mapping[str, Any],
    data: pd.DataFrame,
    *,
    group: str = "",
    ci_level: float = 0.95,
    errorbar: bool = False,
) -> PlotGeometry:
    """
    Geometry for a non-linear fit: one curve over all rows, with the group only colouring points.

    Fitting per group would mean one non-linear fit per level, which the parameter table has no room
    to report honestly — so the single curve repeats and the points carry the grouping.
    """
    from .regression import nonlinear_confidence_band

    x, y = result["x"], result["y"]
    grid = _x_grid(data[x])
    fitted, lower, upper = nonlinear_confidence_band(result, grid, ci_level=ci_level)

    geometry = PlotGeometry(
        points=data,
        lines=pd.DataFrame({group or "group": "", x: grid, y: fitted}),
        mode="continuous",
    )
    if errorbar and not np.allclose(lower, upper):
        geometry.bands = pd.DataFrame(
            {group or "group": "", x: grid, "lower": lower, "upper": upper}
        )
    return geometry


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(
    geometry: PlotGeometry,
    *,
    x: str,
    y: str,
    group: str = "",
    display: str = "overview",
    hover_columns: Sequence[str] = (),
    excluded_mask: Any = None,
    show_excluded: bool = True,
    errorbar: bool = True,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> Any:
    """
    Render a :class:`PlotGeometry` as one figure, overview or anatomically panelled.

    The grouped display subsets every frame to each panel's levels and lays the panels out as
    subplots. The population line has a blank group, so it is kept in every panel — it is the same
    all-level estimate and repeating it is what makes the panels comparable.
    """
    from .interactive import model_plot, panel_grid_figure
    from .region_groups import panel_summary, resolve_panels

    common = dict(
        x=x, y=y, group=group, mode=geometry.mode,
        categorical_order=geometry.categorical_order or None,
        hover_columns=hover_columns, errorbar=errorbar, x_label=x_label, y_label=y_label,
    )

    if display != "grouped":
        return model_plot(
            points=geometry.points, lines=geometry.lines, bands=geometry.bands,
            marginal_means=geometry.marginal_means,
            excluded_mask=excluded_mask, show_excluded=show_excluded, title=title, **common,
        )

    levels = _geometry_levels(geometry, group)
    panels = resolve_panels(levels, column=group or "group")
    excluded = (
        pd.Series(np.asarray(excluded_mask, dtype=bool), index=geometry.points.index)
        if excluded_mask is not None and geometry.points is not None else None
    )

    figures: dict[str, Any] = {}
    for panel, members in panels.items():
        wanted = {str(v) for v in members}
        points = _subset(geometry.points, group, wanted)
        figures[panel] = model_plot(
            points=points,
            # A blank group is the population estimate; it belongs in every panel.
            lines=_subset(geometry.lines, group, wanted | {""}),
            bands=_subset(geometry.bands, group, wanted | {""}),
            marginal_means=_subset(geometry.marginal_means, group, wanted | {""}),
            excluded_mask=(
                excluded.loc[points.index].to_numpy()
                if excluded is not None and points is not None and not points.empty else None
            ),
            show_excluded=show_excluded, title="", **common,
        )
    figure = panel_grid_figure(figures, title=title)
    figure.layout.meta = {"panels": panel_summary(panels)}
    return figure


def _geometry_levels(geometry: PlotGeometry, group: str) -> list[str]:
    """Group levels present anywhere in the geometry, ignoring the population's blank."""
    if not group:
        return []
    seen: list[str] = []
    for frame in (geometry.lines, geometry.marginal_means, geometry.points):
        if frame is None or frame.empty or group not in frame.columns:
            continue
        for value in frame[group].astype(str):
            if value and value not in seen:
                seen.append(value)
    return seen


def _subset(frame: pd.DataFrame | None, group: str, keep: set[str]) -> pd.DataFrame | None:
    """Rows of *frame* whose group is in *keep*; unchanged when it carries no group column."""
    if frame is None or frame.empty or not group or group not in frame.columns:
        return frame
    return frame.loc[frame[group].astype(str).isin(keep)]


__all__ = [
    "LINE_RESOLUTION",
    "PlotGeometry",
    "mmrm_geometry",
    "nonlinear_geometry",
    "r_model_geometry",
    "render",
    "statsmodels_geometry",
]
