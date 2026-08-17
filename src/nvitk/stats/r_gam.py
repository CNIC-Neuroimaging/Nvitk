"""
Markov-random-field smoothing over the vessel adjacency graph, through R's ``mgcv``.

Description
-----------
Every model so far has treated territories as exchangeable levels. A fixed effect per territory
estimates each one from its own rows alone; a random intercept shrinks them all toward a common
mean, as if the basilar and the left MCA were two draws from one population of vessels. Neither uses
the fact that the basilar is *adjacent to* the vertebrals and the posterior cerebrals and to nothing
else.

A Markov random field does. ``s(territory, bs="mrf", xt=list(nb=...))`` penalises differences
between **neighbouring** vessels, so each territory's estimate borrows strength from the ones it is
anatomically connected to and not from the rest. The neighbourhood comes from
:func:`~nvitk.stats.vessel_network.neighbour_list`, and the smoothing parameter — how much
neighbours are pulled together — is estimated by REML rather than chosen.

What it does and does not model
-------------------------------
The graph is **undirected**. An MRF says "adjacent vessels should resemble one another"; it does not
say blood flows from the carotid to the middle cerebral. When the direction is the point, that is a
path model — see :mod:`~nvitk.stats.sem`. The two answer different questions and are worth running
together: the MRF gives a smoothed spatial field over the tree, the SEM gives the edges.

Requirements
------------
R with ``mgcv``, which ships with almost every R installation as a recommended package. Python:
``rpy2``. Nothing else.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger
from .frame_ops import _as_factor_preserving_order

log = Logger()

ANALYSIS_MRF = "mrf"

REQUIRED_R_PACKAGES: tuple[str, ...] = ("mgcv",)

INSTALL_HINT = (
    "mgcv ships with R as a recommended package; if it is genuinely missing:\n"
    "    R -e \"install.packages('mgcv', repos='https://cloud.r-project.org')\"\n"
    "and the Python bridge with:\n"
    "    pip install 'rpy2>=3.5'"
)

#: Response families ``mgcv::gam`` accepts, in the same shape as the other engines' pickers.
GAM_FAMILIES: dict[str, str] = {
    "gaussian": "gaussian — continuous",
    "Gamma(link=log)": "Gamma (log link) — skewed positive, e.g. flows and volumes",
    "poisson": "poisson — counts",
    "binomial": "binomial — binary",
    "scat": "scaled t — heavy-tailed, tolerates outliers",
}

#: Smoothing-parameter selection. REML is the default because it is the least prone to undersmooth.
GAM_METHODS: dict[str, str] = {
    "REML": "REML — restricted maximum likelihood (recommended)",
    "ML": "ML — comparable across models differing in fixed effects",
    "GCV.Cp": "GCV — generalised cross-validation, mgcv's historical default",
}


@dataclass(frozen=True)
class GamBackendStatus:
    """What the MRF engine found when it probed rpy2 → R → mgcv."""

    available: bool
    rpy2_version: str = ""
    r_version: str = ""
    r_packages: dict[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    reason: str = ""

    def summary(self) -> str:
        """One-line description for a status bar."""
        if not self.available:
            return f"unavailable — {self.reason}"
        parts = [f"rpy2 {self.rpy2_version}"]
        if self.r_version:
            parts.append(self.r_version.split("--")[0].strip())
        parts += [f"{k} {v}" for k, v in sorted(self.r_packages.items()) if v]
        return " · ".join(parts)

    def install_hint(self) -> str:
        """What to install to make the engine work."""
        return INSTALL_HINT


def gam_backend_status(*, check_r_packages: bool = True) -> GamBackendStatus:
    """Probe rpy2 → R → ``mgcv``. Never raises, and never imports R as an import side effect."""
    try:
        import rpy2  # noqa: F401
        import rpy2.situation as situation

        rpy2_version = str(getattr(rpy2, "__version__", "?"))
        r_version = str(situation.r_version_from_subprocess() or "")
    except Exception as exc:
        return GamBackendStatus(
            available=False, missing=("rpy2",),
            reason=f"rpy2 is not usable ({type(exc).__name__}: {exc}). It links Python to R.",
        )

    if not shutil.which("R"):
        return GamBackendStatus(
            available=False, rpy2_version=rpy2_version, missing=("R",),
            reason="No R interpreter on PATH.",
        )

    packages: dict[str, str] = {}
    if check_r_packages:
        try:
            import rpy2.robjects as ro

            installed = set(ro.r("rownames(installed.packages())"))
            for name in REQUIRED_R_PACKAGES:
                packages[name] = (
                    str(ro.r(f"as.character(packageVersion({name!r}))")[0])
                    if name in installed else ""
                )
        except Exception as exc:
            log.debug("Could not read R package versions: %s", exc)
        missing = [n for n in REQUIRED_R_PACKAGES if not packages.get(n)]
        if missing:
            return GamBackendStatus(
                available=False, rpy2_version=rpy2_version, r_version=r_version,
                r_packages={k: v for k, v in packages.items() if v},
                missing=tuple(missing),
                reason=f"R package(s) not installed: {', '.join(missing)}.",
            )

    return GamBackendStatus(
        available=True, rpy2_version=rpy2_version, r_version=r_version,
        r_packages={k: v for k, v in packages.items() if v},
    )


# ---------------------------------------------------------------------------
# R helpers
# ---------------------------------------------------------------------------
_R_HELPERS = """
.nvitk_mrf_fit <- function(data, formula, nb, region_col, family, method, select) {
  # The MRF basis indexes its penalty by factor level, so the levels must be exactly the names of
  # the neighbourhood list and in the same order. A mismatch is what produces mgcv's opaque
  # "mismatch between nb/polys supplied area names and data area names".
  data[[region_col]] <- factor(as.character(data[[region_col]]), levels = names(nb))
  # Resolve the family inside mgcv's namespace rather than the caller's: it sees both mgcv's own
  # families (scat, tw) and, through its imports, stats' (gaussian, Gamma). Plain eval() fails on
  # an R started without 'stats' attached, which is how some conda R builds come.
  fam <- eval(parse(text = family), envir = asNamespace("mgcv"))
  mgcv::gam(stats::as.formula(formula), data = data, family = fam,
            method = method, select = as.logical(select))
}

.nvitk_mrf_parametric <- function(fit) {
  s <- summary(fit)
  cf <- as.data.frame(s$p.table, check.names = FALSE)
  cf$parameter <- rownames(cf)
  rownames(cf) <- NULL
  cf
}

.nvitk_mrf_smooth <- function(fit) {
  s <- summary(fit)
  st <- as.data.frame(s$s.table, check.names = FALSE)
  st$smooth <- rownames(st)
  rownames(st) <- NULL
  st
}

.nvitk_mrf_stats <- function(fit) {
  s <- summary(fit)
  grab <- function(expr) tryCatch(as.numeric(expr), error = function(e) NA_real_)
  data.frame(
    deviance_explained = grab(s$dev.expl),
    r_squared_adj = grab(s$r.sq),
    scale = grab(s$scale),
    AIC = grab(stats::AIC(fit)),
    BIC = grab(stats::BIC(fit)),
    logLik = grab(as.numeric(stats::logLik(fit))),
    n_obs = grab(stats::nobs(fit)),
    edf_total = grab(sum(fit$edf)),
    converged = isTRUE(fit$converged),
    check.names = FALSE
  )
}

.nvitk_mrf_field <- function(fit, region_col, levels, newdata) {
  # The fitted spatial field: the smooth's contribution for each vessel, with everything else held
  # at the reference row. ``type = "terms"`` gives exactly that, plus its standard error.
  p <- stats::predict(fit, newdata = newdata, type = "terms", se.fit = TRUE)
  term <- paste0("s(", region_col, ")")
  cols <- colnames(p$fit)
  hit <- cols[grepl(region_col, cols, fixed = TRUE)][1]
  if (is.na(hit)) hit <- cols[1]
  data.frame(level = levels, effect = as.numeric(p$fit[, hit]),
             se = as.numeric(p$se.fit[, hit]), check.names = FALSE)
}
"""

_HELPERS_LOADED = False


def _ensure_helpers() -> None:
    """Define the R helper functions once per session."""
    global _HELPERS_LOADED
    if _HELPERS_LOADED:
        return
    from rpy2.robjects import r as R_

    R_(_R_HELPERS)
    _HELPERS_LOADED = True


def _converter():
    """rpy2 converter that moves pandas frames in both directions."""
    from rpy2.robjects import default_converter, pandas2ri
    from rpy2.robjects.conversion import localconverter

    return localconverter(default_converter + pandas2ri.converter)


def _call_r(name: str, *args) -> pd.DataFrame:
    """Call one of the helpers and bring its data frame back as pandas."""
    from rpy2.robjects import globalenv

    _ensure_helpers()
    with _converter():
        return pd.DataFrame(globalenv[name](*args))


def _r_neighbour_list(neighbours: Mapping[str, Sequence[str]]):
    """Convert a Python neighbour mapping to the named R list ``bs="mrf"`` expects."""
    from rpy2.robjects import ListVector, StrVector

    return ListVector({str(k): StrVector([str(v) for v in vs]) for k, vs in neighbours.items()})


def mrf_formula(
    outcome: str,
    *,
    region_column: str = "territory",
    covariates: Sequence[str] = (),
    k: int | None = None,
    extra_terms: Sequence[str] = (),
) -> str:
    """
    Build the ``gam`` formula for an MRF smooth over the vessel graph.

    ``k`` is the basis dimension: at most one fewer than the number of regions, since the smooth is
    identifiable only up to an intercept. Left unset it is chosen as ``n_regions - 1``, which lets
    the penalty rather than the basis size decide how much smoothing happens.
    """
    terms = [f's({region_column}, bs="mrf", xt=list(nb=nb)' + (f", k={int(k)})" if k else ")")]
    terms += [str(t) for t in covariates if str(t).strip()]
    terms += [str(t) for t in extra_terms if str(t).strip()]
    return f"{outcome} ~ " + " + ".join(terms)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_mrf(
    *,
    data: pd.DataFrame,
    formula: str = "",
    outcome: str = "",
    region_column: str = "territory",
    covariates: Sequence[str] = (),
    neighbours: Mapping[str, Sequence[str]] | None = None,
    family: str = "gaussian",
    method: str = "REML",
    select: bool = False,
    k: int | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """
    Fit a GAM with a Markov-random-field smooth over the vessel adjacency graph.

    Parameters
    ----------
    formula : str
        A complete ``gam`` formula, which must reference ``nb`` inside the MRF term. Leave empty to
        have one built from *outcome*, *region_column* and *covariates*.
    neighbours : mapping, optional
        ``{vessel: [neighbour, ...]}``. Defaults to
        :func:`~nvitk.stats.vessel_network.neighbour_list` restricted to the regions present.
    select : bool
        Add an extra penalty that can shrink a smooth to exactly zero, so the fit can conclude the
        graph structure explains nothing.

    Returns
    -------
    (fit, model_df, metadata)
        *fit* is the R ``gam`` object; *metadata* matches the other engines and additionally carries
        ``neighbours`` and ``levels``.
    """
    status = gam_backend_status()
    if not status.available:
        raise RuntimeError(f"The MRF engine is not available: {status.reason}\n\n{status.install_hint()}")

    if region_column not in data.columns:
        raise ValueError(f"Region column {region_column!r} is not in the frame.")

    from .vessel_network import canonical_node, neighbour_list

    df = data.copy()
    # The graph speaks canonical node ids; the frame speaks whatever the pipeline published.
    df["vessel_node"] = df[region_column].map(canonical_node)
    unmapped = sorted({str(r) for r, n in zip(df[region_column], df["vessel_node"]) if n is None})
    df = df.dropna(subset=["vessel_node"])
    if df.empty:
        raise ValueError(
            f"None of the {data[region_column].nunique()} {region_column!r} levels are recognized "
            f"vessels. An MRF needs a vessel-wise frame (grouping = 'vessel'), since the graph is "
            f"defined between vessels — a melted territory has no neighbours."
        )
    if unmapped:
        log.warning(
            "MRF: dropped %d region(s) with no place in the vessel graph — %s",
            len(unmapped), ", ".join(unmapped),
        )

    present = sorted(set(df["vessel_node"].astype(str)))
    graph = dict(neighbours) if neighbours is not None else neighbour_list(nodes=present)
    graph = {k_: [v for v in vs if v in graph] for k_, vs in graph.items()}
    isolated = sorted(n for n, nb in graph.items() if not nb)
    if len(graph) < 3:
        raise ValueError(
            f"An MRF needs at least three connected regions; only {len(graph)} of the graph's "
            f"vessels are in this frame."
        )

    outcome = outcome or (formula.split("~")[0].strip() if "~" in formula else "")
    if not outcome:
        raise ValueError("Give an outcome, or a formula of the form 'outcome ~ terms'.")
    # ``k`` cannot exceed the number of regions less one; mgcv errors rather than clamping.
    max_k = max(len(graph) - 1, 1)
    k = min(int(k), max_k) if k else max_k
    full_formula = formula or mrf_formula(
        outcome, region_column="vessel_node", covariates=covariates, k=k
    )
    # A user-written formula names the frame's own region column; the fit uses the canonical one.
    if formula:
        full_formula = re.sub(rf"\b{re.escape(region_column)}\b", "vessel_node", full_formula)

    needed = [c for c in _formula_columns(df.columns, full_formula) if c in df.columns]
    n_input = int(len(df))
    dropped_by_column = {c: int(df[c].isna().sum()) for c in needed if bool(df[c].isna().any())}
    df = df.dropna(subset=needed).reset_index(drop=True)
    if not len(df):
        raise ValueError(f"No complete rows left after dropping missing values in {needed}.")

    from .frame_ops import ensure_unique_columns

    df = ensure_unique_columns(df, context="analysis dataframe")
    for column in needed:
        if column != "vessel_node" and not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = _as_factor_preserving_order(df[column])
    df["vessel_node"] = pd.Categorical(df["vessel_node"].astype(str), categories=list(graph))

    with _converter():
        from rpy2.robjects import conversion

        r_data = conversion.get_conversion().py2rpy(df)
    _ensure_helpers()
    from rpy2.robjects import globalenv

    # ``nb`` is referenced by name inside the formula, so it has to exist in the environment mgcv
    # evaluates the smooth's ``xt`` argument in.
    globalenv["nb"] = _r_neighbour_list(graph)
    fit = globalenv[".nvitk_mrf_fit"](
        r_data, full_formula, globalenv["nb"], "vessel_node", family, method, select
    )

    meta = {
        "engine": ANALYSIS_MRF,
        "formula": full_formula,
        "fixed_formula": full_formula,
        "outcome": outcome,
        "region_column": region_column,
        "family": family,
        "method": method,
        "select": bool(select),
        "k": int(k),
        "neighbours": graph,
        "levels": list(graph),
        "isolated": isolated,
        "unmapped_regions": unmapped,
        "n_rows_input": n_input,
        "n_rows": int(len(df)),
        "n_rows_dropped": n_input - int(len(df)),
        "dropna_columns": needed,
        "dropped_by_column": dropped_by_column,
        "loaded": False,
        "backend": status.summary(),
    }
    log.info(
        "MRF fit: %s [%s, %s, k=%d over %d vessels] (n=%d)",
        full_formula, family, method, k, len(graph), len(df),
    )
    return fit, df, meta


def _formula_columns(columns: Sequence[str], formula: str) -> list[str]:
    """Frame columns a formula mentions, ignoring mgcv's own keywords."""
    reserved = {"s", "bs", "mrf", "xt", "list", "nb", "k", "te", "ti", "by", "log", "I"}
    available = {str(c) for c in columns}
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(formula or "")))
    return sorted((tokens - reserved) & available)


# ---------------------------------------------------------------------------
# Reading the fit
# ---------------------------------------------------------------------------
def mrf_coef_frame(fit: Any, *, ci_level: float = 0.95) -> pd.DataFrame:
    """Parametric (non-smooth) coefficients, in the shared report shape."""
    from .mixedlm import _z_critical, significance_stars

    raw = _call_r(".nvitk_mrf_parametric", fit)
    if raw.empty:
        return pd.DataFrame(
            columns=["parameter", "coef", "std_err", "z", "p_value", "ci_low", "ci_high", "sig"]
        )

    def pick(names: Sequence[str]) -> pd.Series:
        """First matching column, compared on a letters-and-digits normalization."""
        lookup = {"".join(c for c in str(x).lower() if c.isalnum()): x for x in raw.columns}
        for name in names:
            key = "".join(c for c in name.lower() if c.isalnum())
            if key in lookup:
                return pd.to_numeric(raw[lookup[key]], errors="coerce")
        return pd.Series(np.nan, index=raw.index)

    out = pd.DataFrame({"parameter": raw.get("parameter", raw.index).astype(str)})
    out["coef"] = pick(["Estimate"])
    out["std_err"] = pick(["Std. Error"])
    out["z"] = pick(["t value", "z value"])
    out["p_value"] = pick(["Pr(>|t|)", "Pr(>|z|)"])
    crit = _z_critical(ci_level)
    out["ci_low"] = out["coef"] - crit * out["std_err"]
    out["ci_high"] = out["coef"] + crit * out["std_err"]
    out["sig"] = [significance_stars(p) for p in out["p_value"]]
    return out.reset_index(drop=True)


def mrf_smooth_frame(fit: Any) -> pd.DataFrame:
    """
    The smooth term's summary: effective degrees of freedom and its test against a flat field.

    ``edf`` is what the MRF actually bought. An edf near 1 means the penalty collapsed the field
    almost to a constant — neighbouring vessels were pulled together so hard that the graph adds
    nothing over a single intercept. An edf near ``k`` means the vessels differ so much that the
    smoothing barely constrained them, which is a fixed effect per territory in all but name.
    """
    raw = _call_r(".nvitk_mrf_smooth", fit)
    if raw.empty:
        return pd.DataFrame(columns=["component", "kind", "edf", "ref_df", "statistic", "p_value"])
    rename = {"edf": "edf", "Ref.df": "ref_df", "F": "statistic", "Chi.sq": "statistic",
              "p-value": "p_value", "smooth": "component"}
    out = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    out["kind"] = "mrf smooth"
    for column in ("edf", "ref_df", "statistic", "p_value"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    keep = ["component", "kind", "edf", "ref_df", "statistic", "p_value"]
    return out.loc[:, [c for c in keep if c in out.columns]].reset_index(drop=True)


def mrf_field_frame(fit: Any, model_df: pd.DataFrame, meta: Mapping[str, Any]) -> pd.DataFrame:
    """
    The smoothed spatial field: each vessel's estimated departure from the overall level.

    This is the MRF's actual output — a value per vessel, shrunk toward its *neighbours* rather than
    toward the grand mean. Comparing it against a plain per-territory fixed effect shows how much
    the anatomy changed the estimate, which is the whole reason for running one.
    """
    levels = [str(v) for v in meta.get("levels", [])]
    if not levels:
        return pd.DataFrame(columns=["level", "effect", "se", "ci_low", "ci_high"])

    # One reference row per vessel: covariates at their mean (numeric) or modal level, so the field
    # is read at a common covariate setting and the vessels are comparable to each other.
    reference: dict[str, Any] = {}
    for column in model_df.columns:
        if column == "vessel_node":
            continue
        series = model_df[column].dropna()
        if series.empty:
            continue
        if pd.api.types.is_numeric_dtype(series):
            reference[column] = float(series.mean())
        else:
            reference[column] = series.mode().iloc[0]
    grid = pd.DataFrame([{**reference, "vessel_node": level} for level in levels])
    grid["vessel_node"] = pd.Categorical(grid["vessel_node"], categories=levels)

    from rpy2.robjects import StrVector, conversion, globalenv

    _ensure_helpers()
    with _converter():
        r_grid = conversion.get_conversion().py2rpy(grid)
        frame = pd.DataFrame(
            globalenv[".nvitk_mrf_field"](fit, "vessel_node", StrVector(levels), r_grid)
        )
    out = pd.DataFrame({
        "level": frame["level"].astype(str),
        "effect": pd.to_numeric(frame["effect"], errors="coerce"),
        "se": pd.to_numeric(frame["se"], errors="coerce"),
    })
    out["ci_low"] = out["effect"] - 1.959963985 * out["se"]
    out["ci_high"] = out["effect"] + 1.959963985 * out["se"]
    out["n_neighbours"] = [len(meta.get("neighbours", {}).get(v, [])) for v in out["level"]]
    return out


def mrf_info_dict(
    fit: Any,
    *,
    outcome_name: str = "Outcome",
    group_name: str = "Group",
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured report in the shape the GUI report panel expects from every engine."""
    meta = dict(meta or {})
    stats = _call_r(".nvitk_mrf_stats", fit)
    row = stats.iloc[0].to_dict() if not stats.empty else {}

    def number(key: str) -> float | None:
        """One statistic as a float, or ``None`` when R returned NA."""
        value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) else None

    smooth = mrf_smooth_frame(fit)
    graph = meta.get("neighbours", {}) or {}
    degrees = [len(v) for v in graph.values()]

    header = {
        "model": "MRF-smoothed GAM (mgcv)",
        "outcome": outcome_name,
        "group": group_name,
        "formula": meta.get("formula", ""),
        "family": meta.get("family", ""),
        "method": meta.get("method", ""),
        "n_obs": int(number("n_obs") or meta.get("n_rows", 0) or 0),
        "n_groups": len(graph),
        "converged": bool(row.get("converged", False)),
        "backend": meta.get("backend", ""),
    }
    fit_statistics = {
        "Deviance explained": number("deviance_explained"),
        "Adjusted R²": number("r_squared_adj"),
        "Scale estimate": number("scale"),
        "Total EDF": number("edf_total"),
        "AIC": number("AIC"),
        "BIC": number("BIC"),
        "Log-likelihood": number("logLik"),
        "Vessels in the graph": float(len(graph)) if graph else None,
        "Mean neighbours per vessel": float(np.mean(degrees)) if degrees else None,
    }

    note = ""
    if not smooth.empty and "edf" in smooth.columns:
        edf = float(pd.to_numeric(smooth["edf"], errors="coerce").iloc[0])
        k = float(meta.get("k", 0) or 0)
        if k and edf < 1.5:
            note = (
                f"The spatial field used {edf:.2f} effective degrees of freedom out of {k:.0f}: the "
                f"penalty collapsed it almost to a constant, so the vessel graph is adding little "
                f"over a single intercept."
            )
        elif k and edf > 0.9 * k:
            note = (
                f"The spatial field used {edf:.2f} of {k:.0f} effective degrees of freedom: the "
                f"vessels differ enough that smoothing barely constrained them, which is close to a "
                f"free effect per territory."
            )
    isolated = meta.get("isolated") or []
    if isolated:
        note = (note + "  " if note else "") + (
            f"{', '.join(isolated)} have no neighbour in the retained graph and were estimated "
            f"independently."
        )

    return {
        "header": header,
        "fixed_effects": mrf_coef_frame(fit),
        "random_effects": smooth.rename(columns={"edf": "var", "ref_df": "sd"}),
        "cov_re": pd.DataFrame(),
        "fit_statistics": {k: v for k, v in fit_statistics.items() if v is not None},
        "has_vcomp": not smooth.empty,
        "group_effects": pd.DataFrame(),
        "mrf_note": note,
    }


def plot_mrf_field(
    field: pd.DataFrame,
    *,
    node_labels: Mapping[str, str] | None = None,
    title: str = "Smoothed field over the vessel graph",
) -> Any:
    """
    The fitted field, vessel by vessel, with its confidence interval.

    Vessels are ordered by their estimated effect rather than alphabetically, so the gradient across
    the network is what the eye picks up. The annotation on each row is that vessel's neighbour
    count — a vessel with one neighbour is shrunk toward that single partner and its interval is
    correspondingly wide.
    """
    import matplotlib.pyplot as plt

    if field is None or field.empty:
        raise ValueError("No fitted field to plot.")

    frame = field.sort_values("effect").reset_index(drop=True)
    labels = dict(node_labels or {})
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.36 * len(frame) + 1.6)))
    y = np.arange(len(frame))

    ax.errorbar(
        frame["effect"], y,
        xerr=[frame["effect"] - frame["ci_low"], frame["ci_high"] - frame["effect"]],
        fmt="o", capsize=4, color="#4C72B0", ecolor="#4C72B0", ms=6,
    )
    ax.axvline(0, color="black", ls="--", lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(str(v), str(v)) for v in frame["level"]])
    if "n_neighbours" in frame.columns:
        for i, n in enumerate(frame["n_neighbours"]):
            ax.annotate(f"{int(n)} nb", (frame["ci_high"].iloc[i], i), fontsize=7,
                        alpha=0.65, xytext=(5, -3), textcoords="offset points")
    ax.set_xlabel("Smoothed effect (deviation from the overall level)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.linked_axes = [ax]
    return fig


def plot_mrf_graph(
    field: pd.DataFrame,
    neighbours: Mapping[str, Sequence[str]],
    *,
    node_labels: Mapping[str, str] | None = None,
    title: str = "Vessel adjacency graph",
) -> Any:
    """
    The adjacency graph itself, nodes coloured by their fitted effect.

    Shows what the smooth was actually working with: which vessels are connected, and whether the
    fitted field varies smoothly across the graph or jumps between neighbours. A sharp jump between
    two adjacent vessels is the interesting case — the penalty resisted it, so the data insisted.
    """
    import matplotlib.pyplot as plt

    nodes = list(neighbours)
    if not nodes:
        raise ValueError("No graph to draw.")
    values = dict(zip(field["level"].astype(str), pd.to_numeric(field["effect"], errors="coerce"))) \
        if field is not None and not field.empty else {}

    # A circular layout: no coordinates are meaningful here, and a circle keeps every edge visible.
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    positions = {node: (float(np.cos(a)), float(np.sin(a))) for node, a in zip(nodes, angles)}

    fig, ax = plt.subplots(figsize=(9, 9))
    for node, partners in neighbours.items():
        for partner in partners:
            if partner in positions and node < partner:
                x0, y0 = positions[node]
                x1, y1 = positions[partner]
                ax.plot([x0, x1], [y0, y1], color="#888888", lw=1.1, alpha=0.55, zorder=1)

    finite = [v for v in values.values() if pd.notna(v)]
    limit = max((abs(v) for v in finite), default=1.0) or 1.0
    labels = dict(node_labels or {})
    for node, (x, y) in positions.items():
        value = values.get(node, np.nan)
        colour = plt.get_cmap("RdBu_r")(0.5 if pd.isna(value) else 0.5 + 0.5 * value / limit)
        ax.scatter([x], [y], s=900, color=colour, edgecolor="#333333", zorder=2)
        ax.annotate(labels.get(node, node), (x, y), ha="center", va="center", fontsize=8, zorder=3)

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.linked_axes = []
    return fig


__all__ = [
    "ANALYSIS_MRF",
    "GAM_FAMILIES",
    "GAM_METHODS",
    "INSTALL_HINT",
    "REQUIRED_R_PACKAGES",
    "GamBackendStatus",
    "fit_mrf",
    "gam_backend_status",
    "mrf_coef_frame",
    "mrf_field_frame",
    "mrf_formula",
    "mrf_info_dict",
    "mrf_smooth_frame",
    "plot_mrf_field",
    "plot_mrf_graph",
]
