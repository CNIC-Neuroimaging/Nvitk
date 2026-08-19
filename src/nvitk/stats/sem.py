"""
Structural equation / path modelling over the cerebral vascular network.

Description
-----------
A stacked territory model asks "does flow differ between vessels?". A path model asks the question
the anatomy actually poses: **given how much arrives upstream, what determines how much leaves
downstream?** Writing the vasculature as a system of regressions —

.. code-block:: text

    basi ~ lva + rva + age_c + sex
    lpca ~ basi + lpcomm + age_c + sex
    lmca ~ lica + age_c + sex

— fits every junction simultaneously, respects that ``basi`` is both an outcome and a predictor, and
decomposes any exogenous effect into the part that travels *through* the network (indirect) and the
part that does not (direct). The mediation analysis already in this toolkit is the three-node
special case of exactly this.

Backends
--------
Two, probed independently and interchangeable at the call site:

``semopy``   pure Python, ``pip install semopy``. No R needed. Preferred default.
``lavaan``   through ``rpy2``, for parity with the R workflow and for the estimators and fit
             indices ``lavaan`` reports that ``semopy`` does not.

Both accept the same model syntax, which is what makes them swappable — that syntax is what
:func:`~nvitk.stats.vessel_network.sem_model_syntax` emits.

What a path coefficient means here
----------------------------------
These are cross-sectional measurements taken in one acquisition, so "upstream causes downstream" is
a hemodynamic assumption written into the model, not something the data establishes. The estimates
are associations *conditional on the assumed direction*. Reversing an edge will usually fit about as
well, so the topology has to come from anatomy — which is precisely why it is hard-coded in
:mod:`~nvitk.stats.vessel_network` rather than learned.
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

log = Logger()

ANALYSIS_SEM = "sem"

SEM_BACKENDS: tuple[str, ...] = ("semopy", "lavaan")

INSTALL_HINT = (
    "Install one of the two backends:\n"
    "    pip install semopy            # pure Python, no R required\n"
    "or, for the R route:\n"
    "    R -e \"install.packages('lavaan', repos='https://cloud.r-project.org')\"\n"
    "    pip install 'rpy2>=3.5'"
)

#: ``lavaan``'s estimators. ``semopy`` accepts the first three under the same names.
SEM_ESTIMATORS: dict[str, str] = {
    "ML": "Maximum likelihood — the default; assumes multivariate normality",
    "MLR": "ML with robust (Huber-White) standard errors — use when residuals are skewed",
    "GLS": "Generalised least squares",
    "WLS": "Weighted least squares — for ordinal or badly non-normal indicators",
}


@dataclass(frozen=True)
class SemBackendStatus:
    """What is available to fit a path model."""

    available: bool
    backends: tuple[str, ...] = ()
    versions: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    def summary(self) -> str:
        """One-line description for a status bar."""
        if not self.available:
            return f"unavailable — {self.reason}"
        return " · ".join(f"{name} {self.versions.get(name, '?')}" for name in self.backends)

    def install_hint(self) -> str:
        """What to install to make the engine work."""
        return INSTALL_HINT

    def preferred(self) -> str:
        """The backend to use when the caller does not care — semopy needs no R."""
        return self.backends[0] if self.backends else ""


def sem_backend_status() -> SemBackendStatus:
    """Probe both backends. Never raises, and never imports R as an import side effect."""
    backends: list[str] = []
    versions: dict[str, str] = {}

    try:
        import semopy

        backends.append("semopy")
        versions["semopy"] = str(getattr(semopy, "__version__", "?"))
    except Exception as exc:
        log.debug("semopy unavailable: %s", exc)

    try:
        import rpy2  # noqa: F401

        if shutil.which("R"):
            import rpy2.robjects as ro

            if "lavaan" in set(ro.r("rownames(installed.packages())")):
                backends.append("lavaan")
                versions["lavaan"] = str(ro.r("as.character(packageVersion('lavaan'))")[0])
                versions.setdefault("rpy2", str(getattr(rpy2, "__version__", "?")))
    except Exception as exc:
        log.debug("lavaan unavailable: %s", exc)

    if not backends:
        return SemBackendStatus(
            available=False,
            reason="neither semopy (Python) nor lavaan (R) is installed",
        )
    return SemBackendStatus(available=True, backends=tuple(backends), versions=versions)


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------
def resolve_network_syntax(syntax: str, columns: Sequence[str]) -> tuple[str, dict[str, str]]:
    """
    Rewrite vessel names in *syntax* onto the columns a network frame actually carries.

    A path model is typed by hand, and the published labels have several spellings for the same
    vessel — ``BASILAR``, ``BASI`` and ``basi`` are one artery, and a single model often mixes them.
    The network frame names its columns by canonical node, so every vessel token that is not already
    a column is resolved through :func:`~nvitk.stats.vessel_network.canonical_node` and rewritten
    when that lands on one.

    Tokens that are not vessels (``age_c``, ``sex``, the operators) are left untouched, as is any
    token that already matches a column — a real column always wins over a rename.

    Returns
    -------
    tuple of (str, dict)
        The rewritten syntax, and the ``{written: resolved}`` map of what changed.
    """
    from nvitk.stats.vessel_network import canonical_node

    available = {str(c) for c in columns}
    renames: dict[str, str] = {}

    def _resolve(match: "re.Match[str]") -> str:
        token = match.group(0)
        if token in available:
            return token
        node = canonical_node(token)
        if node and node in available:
            renames[token] = node
            return node
        return token

    # Comments may hold vessel names in prose; rewriting inside them would be noise.
    rewritten = []
    for line in syntax.splitlines():
        body, sep, comment = line.partition("#")
        rewritten.append(re.sub(r"[A-Za-z_][A-Za-z0-9_.]*", _resolve, body) + sep + comment)

    if renames:
        log.info(
            "Path model: resolved %d vessel name(s) onto the network frame (%s).",
            len(renames),
            ", ".join(f"{k}→{v}" for k, v in sorted(renames.items())[:6]),
        )
    return "\n".join(rewritten), renames


@dataclass
class SemSpec:
    """A path model: the syntax, the data it needs, and how to estimate it."""

    syntax: str
    backend: str = ""
    estimator: str = "ML"
    #: Standardize every observed variable before fitting. Flows in mL/min and ages in years differ
    #: by two orders of magnitude, which makes the raw coefficients incomparable and the optimiser
    #: badly conditioned; standardizing puts every path on the same "SD per SD" scale.
    standardize: bool = True
    #: Drop rows missing any modelled variable. ``False`` asks the backend for FIML instead.
    listwise: bool = True
    #: Grouping column for a multi-group fit — the same model estimated separately in each group,
    #: which is how you ask whether a path *differs* between them rather than assuming it does not.
    #: ``lavaan`` only; ``semopy`` has no multi-group support and the fit is refused rather than
    #: silently pooled, since a pooled estimate is not the question that was asked.
    group: str = ""

    def _statements(self) -> list[tuple[str, str, str]]:
        """
        ``(lhs, operator, rhs)`` for every non-comment line, longest operator first.

        Order matters: ``~~`` and ``=~`` both contain a character that would otherwise split as
        ``~``, so a naive scan reads ``f =~ x1 + x2`` as a regression of ``f =`` on ``x1``.
        """
        out: list[tuple[str, str, str]] = []
        for raw in self.syntax.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            for operator in ("=~", "~~", "~*~", "~"):
                if operator in line:
                    lhs, _, rhs = line.partition(operator)
                    out.append((lhs.strip(), operator, rhs.strip()))
                    break
        return out

    def latent(self) -> list[str]:
        """
        Latent variables — the left-hand sides of ``=~`` measurement lines.

        A latent is defined *by* the model, not measured: ``f1 =~ x1 + x2 + x3`` says the three
        indicators share a common cause called ``f1``. It is therefore not a column of the data, and
        every check that assumes "every name is a column" has to know the difference.
        """
        names: list[str] = []
        for lhs, operator, _rhs in self._statements():
            if operator == "=~" and lhs and lhs not in names:
                names.append(lhs)
        return names

    def tokens(self) -> list[str]:
        """Every identifier the syntax mentions, in order of first appearance."""
        names: list[str] = []
        for line in self.syntax.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", line):
                if token not in names:
                    names.append(token)
        return names

    def observed(self) -> list[str]:
        """
        Variables that must exist as columns — every token that is not a latent.

        This is what the frame has to supply. A latent appears in the syntax and *not* here, which
        is the whole reason the two are separated: validating a CFA against the columns would
        otherwise reject the factor it was written to estimate.
        """
        latents = set(self.latent())
        return [name for name in self.tokens() if name not in latents]

    def variables(self) -> list[str]:
        """
        Observed variables the syntax mentions — the columns the model needs.

        Kept as the historical name for :meth:`observed`; the path models that predate latent
        support call it, and for them the two are identical because there are no latents.
        """
        return self.observed()

    def endogenous(self) -> list[str]:
        """
        Variables the model gives an equation to — the left-hand sides of ``~``.

        A latent with indicators counts too: ``f1 =~ x1 + x2`` gives ``f1`` a measurement equation,
        so a residual covariance on it is a *residual* like any other and must not be reported as
        pairing a modelled variable with an unmodelled one.
        """
        names: list[str] = []
        for lhs, operator, _rhs in self._statements():
            if operator in {"~", "=~"} and lhs and lhs not in names:
                names.append(lhs)
        return names

    def covariances(self) -> list[tuple[str, str]]:
        """``(left, right)`` pairs from the residual-covariance (``~~``) lines."""
        return [
            (lhs, rhs)
            for lhs, operator, rhs in self._statements()
            if operator == "~~" and lhs and rhs
        ]

    def indicators(self) -> dict[str, list[str]]:
        """``{latent: [indicator, …]}`` from the ``=~`` measurement lines."""
        out: dict[str, list[str]] = {}
        for lhs, operator, rhs in self._statements():
            if operator != "=~" or not lhs:
                continue
            names = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", rhs)]
            out.setdefault(lhs, []).extend(n for n in names if n not in out.get(lhs, []))
        return out

    def validate(self, data: pd.DataFrame) -> str:
        """Empty when the model can be fitted against *data*, otherwise the reason it cannot."""
        if "~" not in self.syntax:
            return (
                "The model syntax contains no equation (nothing of the form 'y ~ x' or "
                "'factor =~ indicator1 + indicator2')."
            )

        # A latent needs enough indicators to be identified at all. Two is the practical floor and
        # only with other constraints; one is a rename of the indicator, which lavaan reports as a
        # non-convergence deep in the optimiser rather than as the modelling error it is.
        for latent, indicators in self.indicators().items():
            if len(indicators) < 2:
                return (
                    f"Latent variable '{latent}' has {len(indicators)} indicator"
                    f"{'' if len(indicators) == 1 else 's'}. A factor needs at least two to be "
                    f"identified — with one it is just that indicator under another name. Add "
                    f"indicators, or drop the '=~' line and use the observed variable directly."
                )

        # A residual covariance is only defined between two variables of the same kind: both with
        # equations of their own, or neither. Mixing them fails deep inside the backend with a bare
        # "'x' is not in list", which says nothing about what to change.
        endogenous = set(self.endogenous())
        for left, right in self.covariances():
            if (left in endogenous) != (right in endogenous):
                loose = right if left in endogenous else left
                paired = left if left in endogenous else right
                return (
                    f"'{left} ~~ {right}' pairs a modelled variable with an unmodelled one: "
                    f"{paired} has its own equation, {loose} does not. Residual covariances need "
                    f"both sides to be the same kind — give {loose} an equation (its bilateral "
                    f"counterpart usually has one), or drop the line."
                )

        missing = [v for v in self.variables() if v not in data.columns]
        if not missing:
            return ""

        # Separate the two reasons a name can be absent, because the fixes are different: a vessel
        # that is missing was never measured (or the frame is still long), whereas anything else is
        # a covariate that is simply not in this analysis frame.
        from nvitk.stats.vessel_network import canonical_node

        vessels = [v for v in missing if canonical_node(v)]
        others = [v for v in missing if not canonical_node(v)]
        parts: list[str] = []
        if vessels:
            parts.append(
                f"{', '.join(vessels)} — vessel(s) with no column. A path model needs one column "
                f"per vessel; the frame is pivoted for you, so this means the vessel carries no "
                f"measurement in this cohort. Drop it from the syntax or widen the region selection."
            )
        if others:
            parts.append(
                f"{', '.join(others)} — not in the frame. Covariates have to be subject-level "
                f"columns of the analysis dataframe."
            )
        return "The model mentions " + "  ".join(parts)


def prepare_sem_frame(
    data: pd.DataFrame, spec: SemSpec
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Reduce *data* to the modelled variables, coded numerically and optionally standardized.

    SEM works on a covariance matrix, so every variable has to be numeric: a two-level factor such
    as ``sex`` is dummy-coded here rather than being silently dropped by the backend. Anything with
    more than two levels is refused, because which contrast it should get is a modelling decision,
    not something to guess.

    Latent variables are *not* included — they are estimated, not supplied — and a multi-group
    column rides along untouched: it labels the rows rather than entering the covariance matrix, so
    dummy-coding or standardizing it would destroy the very grouping it is there to provide.
    """
    variables = [v for v in spec.variables() if v in data.columns]
    frame = data.loc[:, variables].copy()
    coding: dict[str, str] = {}

    group_column = str(spec.group or "")
    if group_column:
        if group_column not in data.columns:
            raise ValueError(
                f"{group_column!r} is not in the frame, so the model cannot be grouped by it."
            )
        if group_column in variables:
            raise ValueError(
                f"{group_column!r} is both the grouping column and a modelled variable. A "
                f"multi-group fit already estimates a separate model per level; naming it in the "
                f"syntax as well asks for its effect *within* each of its own levels, where it is "
                f"constant."
            )

    for column in variables:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        levels = sorted(series.dropna().astype(str).unique())
        if len(levels) == 2:
            frame[column] = (series.astype(str) == levels[1]).astype(float).where(series.notna())
            coding[column] = f"{levels[1]}=1, {levels[0]}=0"
        else:
            raise ValueError(
                f"{column!r} has {len(levels)} levels. A path model needs numeric variables; "
                f"recode it into indicator columns first and name them in the syntax."
            )

    n_input = len(frame)
    if spec.listwise:
        frame = frame.dropna()
    dropped = n_input - len(frame)
    if not len(frame):
        # Listwise deletion is decided by the *least* measured variable, so naming the sparse ones
        # turns an unactionable failure into a specific edit: drop those terms from the syntax.
        source = data.loc[:, variables]
        coverage = (source.notna().mean() * 100.0).sort_values()
        sparse = coverage.head(4)
        best = float(coverage.max()) if len(coverage) else 0.0
        raise ValueError(
            "No complete rows remain: no subject has every modelled variable measured. "
            "Listwise deletion is set by the sparsest term — "
            + ", ".join(f"{name} {pct:.0f}%" for name, pct in sparse.items())
            + f" (best covered: {best:.0f}%). Drop those terms from the syntax, or set "
            "listwise=False to estimate with FIML instead."
        )

    constant = [c for c in frame.columns if float(frame[c].std(ddof=0) or 0.0) == 0.0]
    if constant:
        raise ValueError(
            f"{', '.join(constant)} {'has' if len(constant) == 1 else 'have'} zero variance in the "
            f"retained rows, so no covariance can be estimated from {'it' if len(constant) == 1 else 'them'}."
        )

    scales = {c: float(frame[c].std(ddof=0)) for c in frame.columns}
    centres = {c: float(frame[c].mean()) for c in frame.columns}
    if spec.standardize:
        frame = (frame - pd.Series(centres)) / pd.Series(scales)

    # Attached after standardizing, and after the dropna that decided which rows survive, so the
    # labels line up with the rows they belong to and are never scaled.
    if group_column:
        frame[group_column] = data.loc[frame.index, group_column].astype(str)
        levels = sorted(frame[group_column].unique())
        if len(levels) < 2:
            raise ValueError(
                f"{group_column!r} has {len(levels)} level(s) in the retained rows — a multi-group "
                f"fit needs at least two groups to compare."
            )
        log.info("SEM: multi-group over %s (%s).", group_column, ", ".join(levels))

    meta = {
        "variables": variables,
        "latent": spec.latent(),
        "group": group_column,
        "coding": coding,
        "n_rows_input": n_input,
        "n_rows": int(len(frame)),
        "n_rows_dropped": int(dropped),
        "standardized": bool(spec.standardize),
        "scales": scales,
        "centres": centres,
    }
    return frame, meta


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_sem(
    *, data: pd.DataFrame, spec: SemSpec
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """
    Fit a path model with whichever backend is available.

    Returns
    -------
    (fit, model_df, metadata)
        *fit* is the backend's own object — a ``semopy.Model`` or an R ``lavaan`` fit — and
        *metadata* carries ``engine``, ``backend`` and the row accounting the other engines report.
    """
    status = sem_backend_status()
    if not status.available:
        raise RuntimeError(f"No SEM backend: {status.reason}\n\n{status.install_hint()}")

    backend = spec.backend or status.preferred()
    if backend not in status.backends:
        raise RuntimeError(
            f"Backend {backend!r} is not available (have: {', '.join(status.backends)}).\n\n"
            f"{status.install_hint()}"
        )

    problem = spec.validate(data)
    if problem:
        raise ValueError(problem)

    frame, prep = prepare_sem_frame(data, spec)
    fit = _fit_semopy(frame, spec) if backend == "semopy" else _fit_lavaan(frame, spec)

    meta = {
        "engine": ANALYSIS_SEM,
        "backend": backend,
        "backend_summary": status.summary(),
        "estimator": spec.estimator,
        "syntax": spec.syntax,
        "formula": spec.syntax.splitlines()[1] if "\n" in spec.syntax else spec.syntax,
        "loaded": False,
        **prep,
        "dropna_columns": prep["variables"],
        "dropped_by_column": {},
    }
    log.info(
        "SEM fit (%s, %s): %d variables over %d rows.",
        backend, spec.estimator, len(prep["variables"]), prep["n_rows"],
    )
    return fit, frame, meta


def _fit_semopy(frame: pd.DataFrame, spec: SemSpec) -> Any:
    """Fit through ``semopy``."""
    import semopy

    if spec.group:
        raise ValueError(
            f"A multi-group fit over {spec.group!r} needs the lavaan backend — semopy estimates one "
            f"model for the whole sample. Switch the backend to lavaan, or drop the grouping and "
            f"add {spec.group} as an ordinary covariate (which tests a mean shift, not whether the "
            f"paths themselves differ)."
        )
    model = semopy.Model(spec.syntax)
    # semopy names its objectives differently from lavaan; MLR has no direct equivalent, so it
    # falls back to ML rather than failing on a name the backend has never heard of.
    objective = {"ML": "MLW", "MLR": "MLW", "GLS": "GLS", "WLS": "WLS"}.get(spec.estimator, "MLW")
    model.fit(frame, obj=objective)
    return model


def _fit_lavaan(frame: pd.DataFrame, spec: SemSpec) -> Any:
    """Fit through R's ``lavaan``."""
    from rpy2.robjects import conversion, default_converter, globalenv, pandas2ri
    from rpy2.robjects import r as R_
    from rpy2.robjects.conversion import localconverter

    _ensure_lavaan_helpers()
    with localconverter(default_converter + pandas2ri.converter):
        r_data = conversion.get_conversion().py2rpy(frame.reset_index(drop=True))
    missing = "listwise" if spec.listwise else "fiml"
    return globalenv[".nvitk_sem_fit"](
        r_data, spec.syntax, spec.estimator, missing, spec.group
    )


_R_SEM_HELPERS = """
.nvitk_sem_fit <- function(data, syntax, estimator, missing, group) {
  args <- list(model = syntax, data = data, estimator = estimator, missing = missing)
  if (nzchar(group)) args$group <- group
  do.call(lavaan::sem, args)
}

.nvitk_sem_params <- function(fit) {
  p <- lavaan::parameterEstimates(fit, standardized = TRUE, ci = TRUE)
  as.data.frame(p)
}

.nvitk_sem_fitmeasures <- function(fit) {
  fm <- lavaan::fitMeasures(fit)
  data.frame(measure = names(fm), value = as.numeric(fm), check.names = FALSE)
}

.nvitk_sem_effects <- function(fit) {
  # Total and indirect effects need user-defined parameters to be declared in the syntax; when they
  # are absent this returns an empty frame rather than erroring.
  p <- lavaan::parameterEstimates(fit, standardized = TRUE)
  as.data.frame(p[p$op == ":=", , drop = FALSE])
}

.nvitk_sem_modindices <- function(fit, minimum) {
  # Saturated or just-identified models have no degrees of freedom left to free a parameter with,
  # and modificationIndices() errors rather than returning nothing. An empty frame is the honest
  # answer there: there is no misfit to attribute.
  out <- try(lavaan::modificationIndices(fit, sort. = TRUE, minimum.value = minimum),
             silent = TRUE)
  if (inherits(out, "try-error")) {
    return(data.frame(lhs = character(0), op = character(0), rhs = character(0),
                      mi = numeric(0), epc = numeric(0), sepc.all = numeric(0)))
  }
  as.data.frame(out)
}

.nvitk_sem_standardized <- function(fit) {
  as.data.frame(lavaan::standardizedSolution(fit))
}
"""

_LAVAAN_HELPERS_LOADED = False


def _ensure_lavaan_helpers() -> None:
    """Define the R helper functions once per session."""
    global _LAVAAN_HELPERS_LOADED
    if _LAVAAN_HELPERS_LOADED:
        return
    from rpy2.robjects import r as R_

    R_(_R_SEM_HELPERS)
    _LAVAAN_HELPERS_LOADED = True


# ---------------------------------------------------------------------------
# Reading the fit
# ---------------------------------------------------------------------------
def sem_paths_frame(
    fit: Any, *, backend: str = "", latent: Sequence[str] = ()
) -> pd.DataFrame:
    """
    Path estimates in one shape whichever backend produced them.

    Parameters
    ----------
    latent : sequence of str
        Names of the model's latent variables. Needed only for ``semopy``, which reports a
        measurement line as an ordinary regression of the indicator on the factor (``x1 ~ f1``)
        and so loses the distinction lavaan keeps. Passing the latents lets those rows be restored
        to ``f1 =~ x1``, which is what makes the two backends' tables genuinely interchangeable —
        without it a CFA fitted through semopy reports no loadings at all.

    Returns
    -------
    pandas.DataFrame
        ``lhs op rhs parameter coef std_err z p_value ci_low ci_high sig``, where ``op`` is ``~``
        for a regression path, ``~~`` for a covariance and ``=~`` for a latent loading.
    """
    from .mixedlm import significance_stars

    backend = backend or ("lavaan" if _looks_like_lavaan(fit) else "semopy")
    raw = _lavaan_params(fit) if backend == "lavaan" else _semopy_params(fit)
    if raw.empty:
        return pd.DataFrame(columns=[
            "lhs", "op", "rhs", "parameter", "coef", "std_err", "z", "p_value",
            "ci_low", "ci_high", "sig",
        ])

    factors = {str(name) for name in (latent or ())}
    if factors and backend != "lavaan":
        # A measurement line has a latent on the right and an *observed* indicator on the left.
        # ``f2 ~ f1`` has latents on both sides — that is a structural path between two factors, and
        # tagging it as a loading would claim f1 is measured by f2.
        is_loading = (
            (raw["op"].astype(str) == "~")
            & raw["rhs"].astype(str).isin(factors)
            & ~raw["lhs"].astype(str).isin(factors)
        )
        if is_loading.any():
            # Swap into lavaan's orientation: the factor is the common cause, so it is the lhs.
            lhs, rhs = raw.loc[is_loading, "lhs"].copy(), raw.loc[is_loading, "rhs"].copy()
            raw.loc[is_loading, "lhs"], raw.loc[is_loading, "rhs"] = rhs, lhs
            raw.loc[is_loading, "op"] = "=~"

    raw["parameter"] = raw["lhs"].astype(str) + " " + raw["op"].astype(str) + " " + raw["rhs"].astype(str)
    raw["sig"] = [significance_stars(p) for p in raw["p_value"]]
    order = ["lhs", "op", "rhs", "parameter", "coef", "std_err", "z", "p_value",
             "ci_low", "ci_high", "sig"]
    return raw.loc[:, [c for c in order if c in raw.columns]].reset_index(drop=True)


def _looks_like_lavaan(fit: Any) -> bool:
    """Whether *fit* is an R object rather than a semopy model."""
    return hasattr(fit, "rclass") or type(fit).__module__.startswith("rpy2")


def _semopy_params(fit: Any) -> pd.DataFrame:
    """semopy's ``inspect()`` table, renamed to the shared columns."""
    raw = fit.inspect(std_est=False)
    rename = {
        "lval": "lhs", "op": "op", "rval": "rhs", "Estimate": "coef",
        "Est. Std": "coef_std", "Std. Err": "std_err", "z-value": "z", "p-value": "p_value",
    }
    out = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    for column in ("coef", "std_err", "z", "p_value"):
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
    # semopy reports no interval, so build the Wald one rather than leaving the column empty.
    out["ci_low"] = out["coef"] - 1.959963985 * out["std_err"]
    out["ci_high"] = out["coef"] + 1.959963985 * out["std_err"]
    return out


def _lavaan_params(fit: Any) -> pd.DataFrame:
    """lavaan's ``parameterEstimates`` table, renamed to the shared columns."""
    from rpy2.robjects import default_converter, globalenv, pandas2ri
    from rpy2.robjects.conversion import localconverter

    _ensure_lavaan_helpers()
    with localconverter(default_converter + pandas2ri.converter):
        raw = pd.DataFrame(globalenv[".nvitk_sem_params"](fit))
    rename = {
        "est": "coef", "se": "std_err", "z": "z", "pvalue": "p_value",
        "ci.lower": "ci_low", "ci.upper": "ci_high", "std.all": "coef_std",
    }
    out = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns}).copy()
    for column in ("coef", "std_err", "z", "p_value", "ci_low", "ci_high"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def sem_modification_indices(
    fit: Any, *, backend: str = "", minimum: float = 3.84, limit: int = 25
) -> pd.DataFrame:
    """
    Parameters the model does **not** estimate, ranked by how much freeing each would improve fit.

    A poor global fit says the network as specified cannot reproduce the observed covariances; it
    does not say which edge is wrong. This does: each row is a path or covariance the model fixes at
    zero, with the χ² drop expected from letting it go free (``mi``) and the value it would take
    (``epc``, and ``sepc_all`` standardized).

    Read them as *hypotheses*, not fixes. ``minimum`` defaults to 3.84 — the 5% χ²(1) critical
    value — so the table only lists modifications that would be individually significant, but a
    model rebuilt by chasing this list is fitted to its own residuals and its p-values mean nothing.
    The useful reading is anatomical: a large index on a path the anatomy predicts is evidence the
    topology is missing an edge; one on an implausible path is evidence of nothing.

    Returns an empty frame for ``semopy`` (which does not compute them) and for a saturated model
    (which has no fixed parameter left to free).
    """
    columns = ["lhs", "op", "rhs", "parameter", "mi", "epc", "sepc_all"]
    backend = backend or ("lavaan" if _looks_like_lavaan(fit) else "semopy")
    if backend != "lavaan":
        log.info("Modification indices need the lavaan backend; semopy does not compute them.")
        return pd.DataFrame(columns=columns)

    from rpy2.robjects import default_converter, globalenv, pandas2ri
    from rpy2.robjects.conversion import localconverter

    _ensure_lavaan_helpers()
    with localconverter(default_converter + pandas2ri.converter):
        raw = pd.DataFrame(globalenv[".nvitk_sem_modindices"](fit, float(minimum)))
    if raw.empty:
        return pd.DataFrame(columns=columns)

    out = raw.rename(columns={"sepc.all": "sepc_all"}).copy()
    out["parameter"] = (
        out["lhs"].astype(str) + " " + out["op"].astype(str) + " " + out["rhs"].astype(str)
    )
    for column in ("mi", "epc", "sepc_all"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[:, [c for c in columns if c in out.columns]]
    return out.sort_values("mi", ascending=False).head(int(limit)).reset_index(drop=True)


def sem_standardized_solution(fit: Any, *, backend: str = "") -> pd.DataFrame:
    """
    Every parameter on the standardized (SD-per-SD) scale, with its own standard error.

    Different from the ``coef_std`` column :func:`sem_paths_frame` already carries: that one is a
    point rescaling of the unstandardized estimate, whereas ``lavaan``'s standardized solution
    propagates the uncertainty properly by the delta method, so its intervals are the ones to quote
    for a standardized coefficient.

    Empty for ``semopy``, which reports only the rescaled point estimate.
    """
    columns = ["lhs", "op", "rhs", "parameter", "est_std", "std_err", "z", "p_value",
               "ci_low", "ci_high"]
    backend = backend or ("lavaan" if _looks_like_lavaan(fit) else "semopy")
    if backend != "lavaan":
        return pd.DataFrame(columns=columns)

    from rpy2.robjects import default_converter, globalenv, pandas2ri
    from rpy2.robjects.conversion import localconverter

    _ensure_lavaan_helpers()
    with localconverter(default_converter + pandas2ri.converter):
        raw = pd.DataFrame(globalenv[".nvitk_sem_standardized"](fit))
    if raw.empty:
        return pd.DataFrame(columns=columns)

    out = raw.rename(columns={
        "est.std": "est_std", "se": "std_err", "pvalue": "p_value",
        "ci.lower": "ci_low", "ci.upper": "ci_high",
    }).copy()
    out["parameter"] = (
        out["lhs"].astype(str) + " " + out["op"].astype(str) + " " + out["rhs"].astype(str)
    )
    for column in ("est_std", "std_err", "z", "p_value", "ci_low", "ci_high"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.loc[:, [c for c in columns if c in out.columns]].reset_index(drop=True)


def sem_fit_measures(fit: Any, *, backend: str = "") -> dict[str, float]:
    """
    Global fit indices — how well the assumed topology reproduces the observed covariances.

    The conventional reading: ``CFI`` and ``TLI`` above 0.95, ``RMSEA`` below 0.06 and ``SRMR``
    below 0.08 describe a model consistent with the data. A poor fit here does not say which edge is
    wrong, only that the network as specified cannot reproduce what was measured — the modification
    indices are what point at the missing path.
    """
    backend = backend or ("lavaan" if _looks_like_lavaan(fit) else "semopy")
    wanted = ("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic", "npar")

    if backend == "lavaan":
        from rpy2.robjects import default_converter, globalenv, pandas2ri
        from rpy2.robjects.conversion import localconverter

        _ensure_lavaan_helpers()
        with localconverter(default_converter + pandas2ri.converter):
            frame = pd.DataFrame(globalenv[".nvitk_sem_fitmeasures"](fit))
        lookup = {
            str(m).lower(): float(v) for m, v in zip(frame["measure"], frame["value"])
        }
        return {k: lookup[k] for k in wanted if k in lookup and np.isfinite(lookup[k])}

    try:
        import semopy

        stats = semopy.calc_stats(fit)
        row = stats.iloc[0].to_dict() if isinstance(stats, pd.DataFrame) else dict(stats)
    except Exception as exc:
        log.debug("Could not compute semopy fit statistics: %s", exc)
        return {}
    lookup = {str(k).lower(): v for k, v in row.items()}
    out: dict[str, float] = {}
    for key in wanted:
        value = lookup.get(key, lookup.get(key.upper()))
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out[key] = number
    return out


def sem_info_dict(
    fit: Any,
    *,
    outcome_name: str = "network",
    group_name: str = "Group",
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured report in the shape the GUI report panel expects from every engine."""
    meta = dict(meta or {})
    backend = str(meta.get("backend") or "")
    paths = sem_paths_frame(fit, backend=backend, latent=meta.get("latent", ()))
    measures = sem_fit_measures(fit, backend=backend)

    regressions = paths.loc[paths["op"] == "~"] if "op" in paths.columns else paths
    covariances = paths.loc[paths["op"] == "~~"] if "op" in paths.columns else pd.DataFrame()
    loadings = paths.loc[paths["op"] == "=~"] if "op" in paths.columns else pd.DataFrame()

    latents = list(meta.get("latent", []))
    kind = "Measurement model" if (len(loadings) and not len(regressions)) else "Path model"
    if len(loadings) and len(regressions):
        kind = "Structural equation model"
    formula = f"{len(regressions)} structural path(s)"
    if len(loadings):
        formula += f", {len(loadings)} loading(s) on {len(latents)} latent(s)"
    formula += f" over {len(meta.get('variables', []))} observed variables"

    header = {
        "model": f"{kind} ({backend})",
        "outcome": outcome_name,
        "group": str(meta.get("group") or "") or group_name,
        "formula": formula,
        "estimator": meta.get("estimator", ""),
        "n_obs": int(meta.get("n_rows", 0) or 0),
        "n_groups": int(measures.get("ngroups", 0) or 0),
        "converged": bool(measures) or not regressions.empty,
        "backend": meta.get("backend_summary", ""),
        "standardized": bool(meta.get("standardized", False)),
    }

    fit_statistics = {
        "χ²": measures.get("chisq"),
        "df": measures.get("df"),
        "p(χ²)": measures.get("pvalue"),
        "CFI": measures.get("cfi"),
        "TLI": measures.get("tli"),
        "RMSEA": measures.get("rmsea"),
        "SRMR": measures.get("srmr"),
        "AIC": measures.get("aic"),
        "BIC": measures.get("bic"),
    }

    # The residual-covariance block is the closest analogue to a random-effects table: it is what
    # the structural part could not explain.
    random_effects = pd.DataFrame()
    if not covariances.empty:
        random_effects = pd.DataFrame({
            "component": covariances["parameter"].astype(str),
            "kind": "residual covariance",
            "var": pd.to_numeric(covariances["coef"], errors="coerce"),
            "sd": np.sqrt(pd.to_numeric(covariances["coef"], errors="coerce").abs()),
        })

    # The structural table is what the report leads with. When the model is a pure CFA there are no
    # structural paths at all, and leading with an empty table would read as a failed fit — so the
    # loadings take that slot instead, which is the thing that model actually estimated.
    fixed_effects = regressions if len(regressions) or loadings.empty else loadings

    return {
        "header": header,
        "fixed_effects": fixed_effects.reset_index(drop=True),
        "random_effects": random_effects,
        "cov_re": pd.DataFrame(),
        "fit_statistics": {k: v for k, v in fit_statistics.items() if v is not None},
        "has_vcomp": not random_effects.empty,
        "group_effects": pd.DataFrame(),
        "sem_syntax": meta.get("syntax", ""),
        "sem_loadings": loadings.reset_index(drop=True),
        "sem_modification_indices": _safe_modification_indices(fit, backend),
    }


def _safe_modification_indices(fit: Any, backend: str) -> pd.DataFrame:
    """Modification indices, or an empty frame — a diagnostic must never break the report."""
    try:
        return sem_modification_indices(fit, backend=backend)
    except Exception as exc:
        log.debug("Modification indices unavailable: %s", exc)
        return pd.DataFrame(columns=["lhs", "op", "rhs", "parameter", "mi", "epc", "sepc_all"])


# ---------------------------------------------------------------------------
# Effects along the network
# ---------------------------------------------------------------------------
def path_effects(
    paths: pd.DataFrame, *, source: str, target: str, max_depth: int = 6
) -> pd.DataFrame:
    """
    Decompose the effect of *source* on *target* into every route the fitted network provides.

    An indirect effect along a route is the product of its edge coefficients — the standard
    path-tracing rule — so an age effect that reaches the MCA only via the carotid shows up as one
    two-edge route, while a direct ``lmca ~ age_c`` term shows up as a route of length one.

    Only regression paths are traced; residual covariances are associations without a direction and
    have no product to take. Standard errors are not propagated: the delta method for a product of
    correlated estimates needs the full parameter covariance, which is backend-specific. Read the
    magnitudes here and take inference from the individual edges, or declare the product as a
    user-defined parameter in the syntax so the backend computes its interval properly.

    Returns
    -------
    pandas.DataFrame
        ``route``, ``n_edges``, ``effect``, one row per route, plus a final ``total`` row.
    """
    edges = paths.loc[paths["op"] == "~"] if "op" in paths.columns else paths
    if edges.empty:
        return pd.DataFrame(columns=["route", "n_edges", "effect"])

    # lhs ~ rhs means rhs → lhs, so successors of a node are the equations it appears in.
    successors: dict[str, list[tuple[str, float]]] = {}
    for _, row in edges.iterrows():
        successors.setdefault(str(row["rhs"]), []).append(
            (str(row["lhs"]), float(row["coef"]))
        )

    routes: list[dict[str, Any]] = []

    def walk(node: str, path: list[str], product: float, depth: int) -> None:
        """Depth-first path tracing; the graph is acyclic so no visited set is needed beyond depth."""
        if depth > max_depth:
            return
        for nxt, coefficient in successors.get(node, []):
            if nxt in path:          # a cycle would otherwise loop forever
                continue
            value = product * coefficient
            if nxt == target:
                routes.append({
                    "route": " → ".join([*path, nxt]),
                    "n_edges": len(path),
                    "effect": value,
                })
            else:
                walk(nxt, [*path, nxt], value, depth + 1)

    walk(source, [source], 1.0, 1)
    if not routes:
        return pd.DataFrame(columns=["route", "n_edges", "effect"])

    frame = pd.DataFrame(routes).sort_values(["n_edges", "route"]).reset_index(drop=True)
    total = pd.DataFrame([{
        "route": f"TOTAL {source} → {target}",
        "n_edges": pd.NA,
        "effect": float(frame["effect"].sum()),
    }])
    return pd.concat([frame, total], ignore_index=True)


def plot_sem_paths(
    paths: pd.DataFrame,
    *,
    max_paths: int = 30,
    title: str = "Path coefficients",
    op: str = "~",
) -> Any:
    """
    Forest plot of the structural paths, strongest first.

    Reading it: a path whose interval excludes zero is an edge the data supports at the assumed
    direction. Edge magnitudes are comparable to each other only when the fit was standardized —
    otherwise a path in mL/min per mL/min sits beside one in mL/min per year.

    Parameters
    ----------
    op : str
        Which kind of parameter to draw — ``"~"`` for structural paths, ``"=~"`` for factor
        loadings. Empty draws whatever the frame contains, which is what makes this reusable for a
        table that has already been filtered.
    """
    import matplotlib.pyplot as plt

    edges = paths.loc[paths["op"] == op] if (op and "op" in paths.columns) else paths
    if edges.empty:
        raise ValueError(f"No parameters with op={op!r} to plot." if op else "Nothing to plot.")

    frame = edges.reindex(
        edges["coef"].abs().sort_values(ascending=False).index
    ).head(max_paths).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.34 * len(frame) + 1.5)))
    y = np.arange(len(frame))
    coef = pd.to_numeric(frame["coef"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(frame.get("ci_low"), errors="coerce").to_numpy(float)
    high = pd.to_numeric(frame.get("ci_high"), errors="coerce").to_numpy(float)
    if np.isnan(low).all():
        low = high = coef

    significant = pd.to_numeric(frame.get("p_value"), errors="coerce").to_numpy(float) < 0.05
    colors = np.where(significant, "#4C72B0", "#999999")
    for i, (c, lo, hi, colour) in enumerate(zip(coef, low, high, colors)):
        ax.plot([lo, hi], [i, i], color=colour, lw=1.6, alpha=0.9)
        ax.plot([c], [i], marker="o", color=colour, ms=6)

    ax.axvline(0, color="black", ls="--", lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([str(p) for p in frame["parameter"]], fontsize=9)
    ax.set_xlabel("Path coefficient")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.linked_axes = [ax]
    return fig


def plot_sem_network(
    paths: pd.DataFrame,
    *,
    node_labels: Mapping[str, str] | None = None,
    title: str = "Fitted path model",
) -> Any:
    """
    The fitted network as a diagram: nodes laid out by depth, edges weighted by their coefficient.

    Edge width is the magnitude of the standardized path, colour its sign — blue for a positive
    relation, red for a negative one — and a dashed edge is one whose interval covers zero. It is a
    reading aid for the topology, not a substitute for the coefficient table.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    if "op" in paths.columns:
        structural = paths.loc[paths["op"] == "~"]
        loadings = paths.loc[paths["op"] == "=~"]
    else:
        structural, loadings = paths, paths.iloc[0:0]

    # A loading points from the factor to its indicator — the factor is the common cause, and
    # drawing it the other way round (indicator → factor) inverts the model's whole claim. lavaan
    # writes it as ``factor =~ indicator``, i.e. lhs → rhs, which is the opposite orientation from a
    # regression's ``outcome ~ predictor``, so the two are flipped into one convention here.
    latents = sorted(set(loadings["lhs"].astype(str))) if len(loadings) else []
    if len(loadings):
        flipped = loadings.rename(columns={"lhs": "rhs", "rhs": "lhs"})
        edges = pd.concat([structural, flipped], ignore_index=True)
    else:
        edges = structural
    edges = edges.loc[edges["lhs"].astype(str) != edges["rhs"].astype(str)]
    if edges.empty:
        raise ValueError("No structural paths to draw.")

    nodes = sorted(set(edges["lhs"].astype(str)) | set(edges["rhs"].astype(str)))
    incoming = {n: set(edges.loc[edges["lhs"].astype(str) == n, "rhs"].astype(str)) for n in nodes}

    # Layer by longest distance from a source, so upstream vessels sit left of downstream ones.
    depth: dict[str, int] = {}

    def depth_of(node: str, seen: frozenset[str] = frozenset()) -> int:
        """Longest path back to a node with no predecessors."""
        if node in depth:
            return depth[node]
        if node in seen or not incoming.get(node):
            return 0
        value = 1 + max(depth_of(p, seen | {node}) for p in incoming[node])
        depth[node] = value
        return value

    for node in nodes:
        depth[node] = depth_of(node)

    layers: dict[int, list[str]] = {}
    for node in nodes:
        layers.setdefault(depth[node], []).append(node)
    positions = {
        node: (level, i - (len(members) - 1) / 2)
        for level, members in layers.items()
        for i, node in enumerate(sorted(members))
    }

    fig, ax = plt.subplots(figsize=(2.6 * (max(layers) + 1) + 4, 1.5 * max(len(m) for m in layers.values()) + 3))
    scale = float(pd.to_numeric(edges["coef"], errors="coerce").abs().max() or 1.0)
    for _, row in edges.iterrows():
        source, target = str(row["rhs"]), str(row["lhs"])
        if source not in positions or target not in positions:
            continue
        coefficient = float(row["coef"])
        low = pd.to_numeric(pd.Series([row.get("ci_low")]), errors="coerce").iloc[0]
        high = pd.to_numeric(pd.Series([row.get("ci_high")]), errors="coerce").iloc[0]
        crosses_zero = bool(pd.notna(low) and pd.notna(high) and low <= 0 <= high)
        ax.add_patch(FancyArrowPatch(
            positions[source], positions[target],
            arrowstyle="-|>", mutation_scale=14,
            linewidth=0.8 + 3.4 * abs(coefficient) / scale,
            linestyle="--" if crosses_zero else "-",
            color="#C44E52" if coefficient < 0 else "#4C72B0",
            alpha=0.45 if crosses_zero else 0.85,
            shrinkA=18, shrinkB=18, connectionstyle="arc3,rad=0.08",
        ))

    labels = dict(node_labels or {})
    for node, (x, y) in positions.items():
        # The path-diagram convention: a latent variable is an ellipse, an observed one a box. It
        # is not decoration — the shape is what tells the reader which nodes were measured and which
        # the model invented.
        latent = node in latents
        ax.annotate(
            labels.get(node, node), (x, y), ha="center", va="center", fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.55" if latent else "square,pad=0.4",
                facecolor="#eef3fb" if latent else "white",
                edgecolor="#4C72B0" if latent else "#555555",
                linewidth=1.6 if latent else 1.0,
            ),
        )
    if latents:
        ax.annotate(
            "rounded / blue = latent variable", xy=(0.5, 0.0), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=8, color="#555555",
        )

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.linked_axes = []
    return fig


def plot_sem_loadings(
    paths: pd.DataFrame, *, max_paths: int = 40, title: str = "Factor loadings"
) -> Any:
    """
    Forest plot of the measurement model — how strongly each indicator loads on its factor.

    Reading it: a loading whose interval excludes zero is an indicator the factor accounts for. A
    factor whose loadings are all small is one the indicators do not actually share, whatever the
    global fit indices say — which is the check a CFA exists to make.
    """
    loadings = paths.loc[paths["op"] == "=~"] if "op" in paths.columns else paths
    if loadings.empty:
        raise ValueError(
            "This model has no measurement lines ('factor =~ indicator1 + indicator2'), so there "
            "are no loadings to plot."
        )
    return plot_sem_paths(loadings, max_paths=max_paths, title=title, op="=~")


def plot_sem_modification_indices(
    indices: pd.DataFrame, *, max_rows: int = 20, title: str = "Modification indices"
) -> Any:
    """
    Bar chart of the parameters that would most improve fit if freed, largest first.

    A reading aid for *where the model misfits*, not a to-do list — see
    :func:`sem_modification_indices` for why rebuilding a model from this table invalidates its
    own p-values.
    """
    import matplotlib.pyplot as plt

    if indices is None or indices.empty:
        raise ValueError(
            "No modification indices are available. They need the lavaan backend, and a model with "
            "degrees of freedom left — a saturated model has no fixed parameter to free."
        )

    frame = indices.head(int(max_rows)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.34 * len(frame) + 1.4)))
    y = np.arange(len(frame))
    values = pd.to_numeric(frame["mi"], errors="coerce").to_numpy(float)
    ax.barh(y, values, color="#4C72B0", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([str(p) for p in frame["parameter"]], fontsize=9)
    ax.set_xlabel("Modification index (expected χ² drop)")
    ax.axvline(3.84, color="#c44e52", ls="--", lw=1.2)
    ax.annotate("χ²(1) at p=0.05", xy=(3.84, len(frame) - 0.5), fontsize=7.5,
                color="#c44e52", va="top", ha="left")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.linked_axes = [ax]
    return fig


__all__ = [
    "ANALYSIS_SEM",
    "INSTALL_HINT",
    "SEM_BACKENDS",
    "SEM_ESTIMATORS",
    "SemBackendStatus",
    "SemSpec",
    "fit_sem",
    "path_effects",
    "plot_sem_loadings",
    "plot_sem_modification_indices",
    "plot_sem_network",
    "plot_sem_paths",
    "prepare_sem_frame",
    "sem_backend_status",
    "sem_fit_measures",
    "sem_info_dict",
    "sem_modification_indices",
    "sem_paths_frame",
    "sem_standardized_solution",
]
