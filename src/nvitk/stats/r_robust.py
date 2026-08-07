"""
Robust linear regression through R's ``robustbase::lmrob`` (MM-estimation).

Description
-----------
Ordinary least squares minimises squared residuals, so one badly segmented vessel or one mis-scaled
ASL frame moves every coefficient in the model. The usual answers are to hunt the outlier down by
hand or to drop it with an IQR filter — both of which mean deciding, before you have a model, which
observations are allowed to inform it.

``lmrob`` decides that as part of the fit. It estimates the regression by **MM-estimation**: an
initial S-estimate with a 50% breakdown point (half the data may be arbitrary before the estimate
does), refined by an M-step tuned for high efficiency at the Gaussian model. Each observation ends
up with a *robustness weight* between 0 and 1, and those weights are the diagnostic worth reading —
they tell you which subject × territory rows the model chose to discount, which is usually a list of
scans worth re-checking.

Why not ``statsmodels.RLM``
---------------------------
``RLM`` is M-estimation only. It downweights outlying *responses* but has a breakdown point of zero
against outlying *predictors*: a single high-leverage row — an implausible flow value used as a
covariate, say — still drags the fit wherever it likes. MM-estimation resists both. That difference
is the reason this module exists rather than a pure-Python path.

Requirements
------------
R: ``robustbase``, plus ``emmeans`` for marginal means and confidence bands. Python: ``rpy2``.

Conventions
-----------
Fitting goes through R's formula interface, so factor contrasts are named by R (``sexmale``) and not
by patsy (``C(sex)[T.male]``). That is why prediction and plotting round-trip through R, exactly as
for :mod:`~nvitk.stats.r_mixedlm`, rather than rebuilding a design matrix locally.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import shutil
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

ANALYSIS_LMROB = "lmrob"

REQUIRED_R_PACKAGES: tuple[str, ...] = ("robustbase",)
OPTIONAL_R_PACKAGES: tuple[str, ...] = ("emmeans",)

INSTALL_HINT = (
    "Install the R side with:\n"
    "    R -e \"install.packages(c('robustbase','emmeans'), repos='https://cloud.r-project.org')\"\n"
    "and the Python bridge with:\n"
    "    pip install 'rpy2>=3.5'"
)


# ---------------------------------------------------------------------------
# Estimator settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LmrobEstimator:
    """One ``lmrob`` estimation chain, as accepted by its ``method=`` argument."""

    key: str
    label: str
    description: str


#: ``lmrob`` estimation chains. Each letter is a step: S = initial high-breakdown estimate,
#: M = efficiency-tuned refinement, D = a design-adaptive scale, the second M another refinement.
LMROB_ESTIMATORS: dict[str, LmrobEstimator] = {
    "MM": LmrobEstimator(
        "MM", "MM (S → M)",
        "The classic MM-estimator: 50% breakdown from the S step, ~95% Gaussian efficiency from "
        "the M step. The default, and the right first choice.",
    ),
    "SMD": LmrobEstimator(
        "SMD", "SMD (S → M → D-scale)",
        "Adds a design-adaptive scale estimate, which corrects the residual scale for leverage. "
        "Better calibrated standard errors in small samples.",
    ),
    "SMDM": LmrobEstimator(
        "SMDM", "SMDM (S → M → D → M)",
        "The KS2014 recommendation: a second M-step after the design-adaptive scale. Slowest, and "
        "the best behaved when the design is unbalanced — which a per-territory frame usually is.",
    ),
    "S": LmrobEstimator(
        "S", "S only",
        "The initial high-breakdown estimate alone. Maximum resistance, low efficiency — useful to "
        "compare against, rarely to report.",
    ),
}

#: ``psi`` functions: how quickly a residual's influence is cut off as it grows.
LMROB_PSI: dict[str, str] = {
    "bisquare": "Tukey bisquare — redescending, zero weight past the tuning constant (default)",
    "optimal": "Optimal — minimises maximum bias for a given efficiency",
    "hampel": "Hampel — three-part redescending, gentler cut-off",
    "lqq": "LQQ — smooth redescending, the KS2014 default",
    "welsh": "Welsh — smooth, never exactly zero",
    "ggw": "Generalised Gauss-Weight — smooth redescending",
    "huber": "Huber — monotone, bounds influence but never redescends",
}

#: Preset control bundles from the robustbase literature. Empty means "use method/psi as given".
LMROB_SETTINGS: dict[str, str] = {
    "": "Custom — use the estimator and psi chosen above",
    "KS2011": "Koller & Stahel (2011) — SMDM with the lqq psi",
    "KS2014": "Koller & Stahel (2014) — KS2011 plus a more robust initial scale",
}


@dataclass(frozen=True)
class RobustBackendStatus:
    """What the robust engine found when it probed rpy2 → R → robustbase."""

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


def _r_package_versions(names: Sequence[str]) -> dict[str, str]:
    """Installed version of each R package, empty string when absent. Never raises."""
    try:
        import rpy2.robjects as ro

        installed = set(ro.r("rownames(installed.packages())"))
        out: dict[str, str] = {}
        for name in names:
            if name not in installed:
                out[name] = ""
                continue
            try:
                out[name] = str(ro.r(f"as.character(packageVersion({name!r}))")[0])
            except Exception:
                out[name] = "?"
        return out
    except Exception as exc:
        log.debug("Could not read R package versions: %s", exc)
        return {name: "" for name in names}


def robust_backend_status(*, check_r_packages: bool = True) -> RobustBackendStatus:
    """
    Probe rpy2 → R → ``robustbase``. Never raises, and never imports R as an import side effect.

    Independent of the lme4 and MMRM probes: ``robustbase`` has no shared dependency with either, so
    it can be available when they are not.
    """
    try:
        import rpy2  # noqa: F401
        import rpy2.situation as situation

        rpy2_version = str(getattr(rpy2, "__version__", "?"))
        r_version = str(situation.r_version_from_subprocess() or "")
    except Exception as exc:
        return RobustBackendStatus(
            available=False,
            missing=("rpy2",),
            reason=f"rpy2 is not usable ({type(exc).__name__}: {exc}). It links Python to R.",
        )

    if not shutil.which("R"):
        return RobustBackendStatus(
            available=False, rpy2_version=rpy2_version, missing=("R",),
            reason="No R interpreter on PATH.",
        )

    packages: dict[str, str] = {}
    if check_r_packages:
        packages = _r_package_versions([*REQUIRED_R_PACKAGES, *OPTIONAL_R_PACKAGES])
        missing = [name for name in REQUIRED_R_PACKAGES if not packages.get(name)]
        if missing:
            return RobustBackendStatus(
                available=False, rpy2_version=rpy2_version, r_version=r_version,
                r_packages={k: v for k, v in packages.items() if v},
                missing=tuple(missing),
                reason=f"R package(s) not installed: {', '.join(missing)}.",
            )

    return RobustBackendStatus(
        available=True, rpy2_version=rpy2_version, r_version=r_version,
        r_packages={k: v for k, v in packages.items() if v},
    )


# ---------------------------------------------------------------------------
# R helpers
# ---------------------------------------------------------------------------
_R_HELPERS = """
.nvitk_lmrob_fit <- function(data, formula, method, psi, setting, max_it, k_max, seed) {
  # A fixed seed matters here: the initial S-estimate is found by random resampling, so two runs on
  # the same data can otherwise return slightly different coefficients.
  if (!is.na(seed)) set.seed(as.integer(seed))
  ctrl <- if (nzchar(setting)) {
    robustbase::lmrob.control(setting = setting, max.it = as.integer(max_it),
                              k.max = as.integer(k_max))
  } else {
    robustbase::lmrob.control(method = method, psi = psi, max.it = as.integer(max_it),
                              k.max = as.integer(k_max))
  }
  robustbase::lmrob(stats::as.formula(formula), data = data, control = ctrl)
}

.nvitk_lmrob_coefs <- function(fit) {
  cf <- as.data.frame(summary(fit)$coefficients, check.names = FALSE)
  cf$parameter <- rownames(cf)
  rownames(cf) <- NULL
  cf
}

.nvitk_lmrob_weights <- function(fit) {
  w <- robustbase::weights(fit, type = "robustness")
  data.frame(row = seq_along(w), weight = as.numeric(w),
             residual = as.numeric(stats::residuals(fit)),
             fitted = as.numeric(stats::fitted(fit)),
             check.names = FALSE)
}

.nvitk_lmrob_stats <- function(fit) {
  s <- summary(fit)
  grab <- function(expr) tryCatch(as.numeric(expr), error = function(e) NA_real_)
  data.frame(
    scale = grab(fit$scale),
    r_squared = grab(s$r.squared),
    adj_r_squared = grab(s$adj.r.squared),
    df_residual = grab(fit$df.residual),
    n_obs = grab(length(stats::residuals(fit))),
    # ``rweights`` below 1e-8 are what robustbase itself reports as rejected observations.
    n_rejected = grab(sum(robustbase::weights(fit, type = "robustness") < 1e-8)),
    n_downweighted = grab(sum(robustbase::weights(fit, type = "robustness") < 0.5)),
    converged = isTRUE(fit$converged),
    iterations = grab(fit$iter),
    check.names = FALSE
  )
}

.nvitk_lmrob_predict <- function(fit, newdata) {
  as.numeric(stats::predict(fit, newdata = newdata))
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


def _formula_columns(columns: Sequence[str], formula: str) -> list[str]:
    """Frame columns a formula mentions."""
    import re

    available = {str(c) for c in columns}
    return sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(formula or ""))) & available)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_lmrob(
    *,
    data: pd.DataFrame,
    formula: str,
    method: str = "MM",
    psi: str = "bisquare",
    setting: str = "",
    max_iterations: int = 500,
    seed: int | None = 42,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """
    Fit a robust linear model by MM-estimation.

    Parameters
    ----------
    formula : str
        An R formula, e.g. ``"pi ~ age_c + sex + territory"``. There is no random part: ``lmrob``
        fits a single population regression. For clustered data — repeated territories within a
        subject — the coefficients stay valid but the standard errors ignore the clustering; use
        ``lme4`` or MMRM when the correlation is the point.
    method : str
        Estimation chain, see :data:`LMROB_ESTIMATORS`.
    psi : str
        Loss function, see :data:`LMROB_PSI`. Ignored when *setting* is given.
    setting : {"", "KS2011", "KS2014"}
        A published control preset. When set it overrides *method* and *psi*.
    seed : int or None
        The initial S-estimate resamples at random, so an unpinned seed makes the fit irreproducible
        at the last decimal. ``None`` leaves R's generator alone.

    Returns
    -------
    (fit, model_df, metadata)
        *fit* is the R ``lmrob`` object. *model_df* is the frame actually fitted, in the same row
        order as the robustness weights. *metadata* matches the other engines.
    """
    status = robust_backend_status()
    if not status.available:
        raise RuntimeError(
            f"The robust regression engine is not available: {status.reason}\n\n"
            f"{status.install_hint()}"
        )

    formula = str(formula or "").strip()
    if "~" not in formula:
        raise ValueError("A robust model needs a formula of the form 'outcome ~ terms'.")
    if "|" in formula:
        raise ValueError(
            "lmrob fits a single population regression and has no random-effects syntax. Drop the "
            "'(… | …)' term, or switch to the lme4 or MMRM engine."
        )
    method = str(method or "MM").strip() or "MM"
    if method not in LMROB_ESTIMATORS:
        raise ValueError(f"Unknown estimator {method!r}. Available: {', '.join(LMROB_ESTIMATORS)}.")
    psi = str(psi or "bisquare").strip() or "bisquare"
    if psi not in LMROB_PSI:
        raise ValueError(f"Unknown psi function {psi!r}. Available: {', '.join(LMROB_PSI)}.")
    setting = str(setting or "").strip()
    if setting and setting not in LMROB_SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}. Available: {', '.join(LMROB_SETTINGS)}.")

    needed = _formula_columns(data.columns, formula)
    n_input = int(len(data))
    dropped_by_column = {c: int(data[c].isna().sum()) for c in needed if bool(data[c].isna().any())}
    df = data.dropna(subset=needed).reset_index(drop=True) if needed else data.reset_index(drop=True)
    if not len(df):
        raise ValueError(f"No complete rows left after dropping missing values in {needed}.")

    # An MM-estimate needs enough clean rows to find a 50%-breakdown starting point; below roughly
    # 5 observations per parameter the S step is chasing noise.
    n_terms = max(len(needed) - 1, 1)
    if len(df) < 5 * n_terms:
        log.warning(
            "Robust fit on %d rows for ~%d terms — MM-estimation is unreliable at this ratio.",
            len(df), n_terms,
        )

    # R rejects duplicate column names, and string columns must become factors for contrasts to be
    # built the way the formula implies.
    from .frame_ops import ensure_unique_columns

    df = ensure_unique_columns(df, context="analysis dataframe")
    for column in needed:
        if column in df.columns and not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.Categorical(df[column].astype(str))

    with _converter():
        from rpy2.robjects import conversion

        r_data = conversion.get_conversion().py2rpy(df)
    _ensure_helpers()
    from rpy2.robjects import globalenv

    fit = globalenv[".nvitk_lmrob_fit"](
        r_data, formula, method, psi, setting,
        int(max_iterations), int(max_iterations),
        float("nan") if seed is None else int(seed),
    )

    meta = {
        "engine": ANALYSIS_LMROB,
        "formula": formula,
        "fixed_formula": formula,
        "method": method,
        "method_label": LMROB_ESTIMATORS[method].label,
        "psi": psi,
        "setting": setting,
        "seed": seed,
        "max_iterations": int(max_iterations),
        "n_rows_input": n_input,
        "n_rows": int(len(df)),
        "n_rows_dropped": n_input - int(len(df)),
        "dropna_columns": needed,
        "dropped_by_column": dropped_by_column,
        "loaded": False,
        "backend": status.summary(),
    }
    log.info(
        "Robust fit: %s [%s, psi=%s%s] (n=%d)",
        formula, method, psi, f", setting={setting}" if setting else "", len(df),
    )
    return fit, df, meta


# ---------------------------------------------------------------------------
# Reading the fit
# ---------------------------------------------------------------------------
_COEF_ALIASES: dict[str, tuple[str, ...]] = {
    "coef": ("Estimate", "estimate"),
    "std_err": ("Std. Error", "Std.Error", "std_error"),
    "z": ("t value", "t.value", "statistic"),
    "p_value": ("Pr(>|t|)", "p.value", "p_value"),
}


def _pick(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series | None:
    """First matching column, comparing on a letters-and-digits normalization of the name."""
    lookup = {"".join(ch for ch in str(c).lower() if ch.isalnum()): c for c in frame.columns}
    for name in names:
        key = "".join(ch for ch in str(name).lower() if ch.isalnum())
        if key in lookup:
            return frame[lookup[key]]
    return None


def lmrob_coef_frame(fit: Any, *, ci_level: float = 0.95) -> pd.DataFrame:
    """
    Coefficient table in the shared shape: ``parameter coef std_err z p_value ci_low ci_high sig``.

    The standard errors are the robust ones ``summary.lmrob`` reports — computed from the estimator's
    own sandwich-type covariance, not from an OLS residual variance.
    """
    from .mixedlm import _z_critical, significance_stars

    raw = _call_r(".nvitk_lmrob_coefs", fit)
    if raw.empty:
        return pd.DataFrame(
            columns=["parameter", "coef", "std_err", "z", "p_value", "ci_low", "ci_high", "sig"]
        )

    parameter = _pick(raw, ("parameter",))
    out = pd.DataFrame({"parameter": (parameter if parameter is not None else raw.index).astype(str)})
    for target, names in _COEF_ALIASES.items():
        series = _pick(raw, names)
        out[target] = pd.to_numeric(series, errors="coerce") if series is not None else np.nan

    crit = _z_critical(ci_level)
    out["ci_low"] = out["coef"] - crit * out["std_err"]
    out["ci_high"] = out["coef"] + crit * out["std_err"]
    out["sig"] = [significance_stars(p) for p in out["p_value"]]
    return out.reset_index(drop=True)


def lmrob_weights_frame(
    fit: Any, model_df: pd.DataFrame | None = None, *, key_columns: Sequence[str] = ()
) -> pd.DataFrame:
    """
    Per-observation robustness weights — the diagnostic that makes a robust fit worth running.

    A weight of 1 means the observation was treated as ordinary; 0 means the estimator rejected it
    outright. Joining the low-weight rows back to *model_df* names the subject and territory behind
    each one, which is normally a shortlist of scans to re-inspect rather than a statistical finding.

    Parameters
    ----------
    model_df : pandas.DataFrame, optional
        The frame returned by :func:`fit_lmrob`, in its original row order. Its *key_columns* are
        attached to the weights so each row is identifiable.
    key_columns : sequence of str
        Which columns to carry over — typically ``("subject_uid", "territory")``. Defaults to those
        two when they are present.

    Returns
    -------
    pandas.DataFrame
        ``weight``, ``residual``, ``fitted``, ``rejected``, plus the key columns, sorted by weight so
        the most heavily discounted observations come first.
    """
    frame = _call_r(".nvitk_lmrob_weights", fit)
    if frame.empty:
        return pd.DataFrame(columns=["weight", "residual", "fitted", "rejected"])

    out = pd.DataFrame({
        "weight": pd.to_numeric(frame.get("weight"), errors="coerce"),
        "residual": pd.to_numeric(frame.get("residual"), errors="coerce"),
        "fitted": pd.to_numeric(frame.get("fitted"), errors="coerce"),
    })
    out["rejected"] = out["weight"] < 1e-8

    if model_df is not None and len(model_df) == len(out):
        keys = [str(c) for c in (key_columns or ("subject_uid", "territory")) if c in model_df.columns]
        for column in reversed(keys):
            out.insert(0, column, model_df[column].astype(str).to_numpy())
    elif model_df is not None:
        # R drops rows the formula could not evaluate; without a 1:1 mapping, silently pairing the
        # two frames would attach the wrong subject to every weight.
        log.warning(
            "Robustness weights (%d) do not line up with the model frame (%d) — reporting them "
            "without subject/territory labels.", len(out), len(model_df),
        )
    return out.sort_values("weight", kind="stable").reset_index(drop=True)


def lmrob_fit_statistics(fit: Any) -> dict[str, Any]:
    """Scale, robust R², convergence and how many observations were discounted."""
    frame = _call_r(".nvitk_lmrob_stats", fit)
    if frame.empty:
        return {}
    row = frame.iloc[0]
    out: dict[str, Any] = {}
    for key in ("scale", "r_squared", "adj_r_squared", "df_residual", "n_obs",
                "n_rejected", "n_downweighted", "iterations"):
        value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        if pd.notna(value):
            out[key] = float(value)
    out["converged"] = bool(row.get("converged", False))
    return out


def lmrob_info_dict(
    fit: Any,
    *,
    outcome_name: str = "Outcome",
    group_name: str = "Group",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured report in the same shape the other engines return, for the GUI report panel."""
    meta = dict(meta or {})
    stats = lmrob_fit_statistics(fit)
    coefs = lmrob_coef_frame(fit)

    n_obs = int(stats.get("n_obs", meta.get("n_rows", 0)) or 0)
    n_rejected = int(stats.get("n_rejected", 0) or 0)
    n_down = int(stats.get("n_downweighted", 0) or 0)

    header = {
        "model": "Robust linear model (lmrob)",
        "outcome": outcome_name,
        "group": group_name,
        "formula": meta.get("formula", ""),
        "estimator": meta.get("method_label", meta.get("method", "MM")),
        "psi": meta.get("psi", ""),
        "setting": meta.get("setting", "") or "custom",
        "n_obs": n_obs,
        "n_groups": 0,
        "converged": bool(stats.get("converged", False)),
        "backend": meta.get("backend", ""),
    }
    fit_statistics = {
        "Robust residual scale": stats.get("scale"),
        "Robust R²": stats.get("r_squared"),
        "Adjusted robust R²": stats.get("adj_r_squared"),
        "Residual df": stats.get("df_residual"),
        "IRLS iterations": stats.get("iterations"),
        "Observations rejected (weight ≈ 0)": float(n_rejected),
        "Observations downweighted (weight < 0.5)": float(n_down),
    }

    note = ""
    if n_obs:
        note = (
            f"{n_down} of {n_obs} observations ({100 * n_down / n_obs:.1f}%) carry a robustness "
            f"weight below 0.5, of which {n_rejected} were rejected outright. See the Weights tab — "
            f"those rows are candidates for QC, not evidence about the model."
        )

    return {
        "header": header,
        "fixed_effects": coefs,
        "random_effects": pd.DataFrame(),
        "cov_re": pd.DataFrame(),
        "fit_statistics": {k: v for k, v in fit_statistics.items() if v is not None},
        "has_vcomp": False,
        "group_effects": pd.DataFrame(),
        "robust_weights_note": note,
    }


# ---------------------------------------------------------------------------
# Prediction, marginal means and plotting
# ---------------------------------------------------------------------------
def lmrob_predict(fit: Any, newdata: pd.DataFrame, *, use_random_effects: bool = False) -> np.ndarray:
    """
    Predictions at *newdata*, evaluated by R so factor contrasts keep R's own encoding.

    ``use_random_effects`` exists only to match the signature the shared plotting code expects; a
    robust linear model has no random effects, so it is ignored.
    """
    del use_random_effects
    _ensure_helpers()
    from rpy2.robjects import globalenv

    frame = newdata.reset_index(drop=True).copy()
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.Categorical(frame[column].astype(str))
    with _converter():
        from rpy2.robjects import conversion

        r_new = conversion.get_conversion().py2rpy(frame)
        return np.asarray(globalenv[".nvitk_lmrob_predict"](fit, r_new), dtype=float)


def lmrob_emmeans(
    fit: Any,
    specs: str,
    *,
    at_name: str = "",
    at_values: Sequence[float] | None = None,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """
    Estimated marginal means and their confidence intervals, from ``emmeans`` in R.

    Same signature and return shape as :func:`~nvitk.stats.r_mixedlm.lme4_emmeans`, so the shared
    plotting code can call either. It reuses that module's R helper, which takes any model
    ``emmeans`` supports — and ``emmeans`` supports ``lmrob`` directly. The one difference is that
    an ``lmrob`` fit *is* the R object, where a pymer4 model wraps one.

    Parameters
    ----------
    specs : str
        An emmeans specification formula: ``"~ territory"``, or ``"~ age_c | territory"``.
    at_name, at_values
        Evaluate a continuous predictor over a grid, for a confidence band along a line.
    """
    from rpy2.robjects import FloatVector, globalenv
    from rpy2.robjects import r as R_

    from . import r_mixedlm

    if not r_mixedlm._EMMEANS_HELPER_LOADED:
        R_(r_mixedlm._R_EMMEANS_HELPER)
        r_mixedlm._EMMEANS_HELPER_LOADED = True

    values = FloatVector(list(at_values) if at_values is not None else [])
    with _converter():
        return pd.DataFrame(
            globalenv[".nvitk_lme4_emmeans"](
                fit, str(specs), str(at_name), values, float(ci_level)
            )
        )


def _lmrob_band(
    fit: Any,
    *,
    x: str,
    x_values: Any,
    group: str,
    levels: Sequence[str],
    continuous: bool,
    fixed_formula: str,
    ci_level: float,
) -> dict[str | None, pd.DataFrame] | None:
    """Confidence bands from ``emmeans``, in the shape the shared plotter consumes."""
    from .r_mixedlm import _emmeans_band

    # The band logic is engine-independent — it only needs marginal means keyed by x and group — so
    # it is shared with lme4, with this module's emmeans call substituted in.
    return _emmeans_band(
        fit,
        x=x,
        x_values=x_values,
        group=group,
        levels=levels,
        continuous=continuous,
        fixed_formula=fixed_formula,
        ci_level=ci_level,
        emmeans_fn=lmrob_emmeans,
    )


def plot_lmrob_params(
    *,
    fit: Any,
    df_fit: pd.DataFrame,
    x: str,
    y: str,
    group: str = "",
    mode: str = "auto",
    group_order: Sequence[str] | None = None,
    restrict_to_orders: bool = False,
    include_points: bool = True,
    errorbar: bool = False,
    ci_level: float = 0.95,
    fixed_formula: str = "",
    palette: str = "tab10",
    display: str = "overview",
    title: str = "Robust linear model",
    x_label: str | None = None,
    y_label: str | None = None,
) -> Any:
    """
    Population and per-level curves for a robust fit, predicted through R.

    Visually identical to the other engines' plots — dashed black population curve, one coloured
    curve per level, observed data in a lighter tone — and it supports the same
    ``display="grouped"`` anatomical panelling. The per-level curves only separate when the grouping
    factor is in the formula: ``lmrob`` has no random effects to place them apart otherwise.
    """
    from .r_mixedlm import plot_lme4_params

    return plot_lme4_params(
        model=fit,
        df_fit=df_fit,
        x=x,
        y=y,
        group=group,
        mode=mode,
        group_order=group_order,
        restrict_to_orders=restrict_to_orders,
        include_points=include_points,
        errorbar=errorbar,
        ci_level=ci_level,
        fixed_formula=fixed_formula or "",
        palette=palette,
        display=display,
        title=title,
        x_label=x_label,
        y_label=y_label,
        predict_fn=lmrob_predict,
        band_fn=_lmrob_band,
        population_label="Robust fit (population)",
    )


def plot_lmrob_weights(
    weights: pd.DataFrame,
    *,
    label_column: str = "",
    threshold: float = 0.5,
    max_labels: int = 25,
    title: str = "Robustness weights",
) -> Any:
    """
    Weight against fitted value, with the discounted observations labelled.

    Reading it: a cloud at weight 1 with a short tail is a well-behaved fit; a long tail, or a
    cluster of low weights sharing one territory, says the model and that territory disagree — which
    is either a real subgroup or a segmentation problem, and worth knowing which.
    """
    import matplotlib.pyplot as plt

    if weights is None or weights.empty:
        raise ValueError("No robustness weights to plot.")

    fig, ax = plt.subplots(figsize=(10, 6))
    low = weights["weight"] < threshold
    ax.scatter(weights.loc[~low, "fitted"], weights.loc[~low, "weight"],
               s=20, alpha=0.5, color="#4C72B0", label=f"weight ≥ {threshold:g}")
    ax.scatter(weights.loc[low, "fitted"], weights.loc[low, "weight"],
               s=34, alpha=0.85, color="#C44E52", label=f"weight < {threshold:g}")

    # Label only the worst few: one text object per outlier is unreadable past a couple of dozen.
    if label_column and label_column in weights.columns:
        for _, row in weights.loc[low].head(max_labels).iterrows():
            ax.annotate(str(row[label_column]), (row["fitted"], row["weight"]),
                        fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")

    ax.axhline(threshold, color="grey", ls="--", lw=1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Fitted value")
    ax.set_ylabel("Robustness weight")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.linked_axes = [ax]
    return fig


__all__ = [
    "ANALYSIS_LMROB",
    "INSTALL_HINT",
    "LMROB_ESTIMATORS",
    "LMROB_PSI",
    "LMROB_SETTINGS",
    "OPTIONAL_R_PACKAGES",
    "REQUIRED_R_PACKAGES",
    "LmrobEstimator",
    "RobustBackendStatus",
    "fit_lmrob",
    "lmrob_coef_frame",
    "lmrob_emmeans",
    "lmrob_fit_statistics",
    "lmrob_info_dict",
    "lmrob_predict",
    "lmrob_weights_frame",
    "plot_lmrob_params",
    "plot_lmrob_weights",
    "robust_backend_status",
]
