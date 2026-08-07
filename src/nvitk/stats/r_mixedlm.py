"""
Mixed models through R's ``lme4``, via ``pymer4``.

Description
-----------
An optional alternative to :mod:`nvitk.stats.mixedlm`. Same models, different way of saying them:
``lme4`` puts the whole random structure inside the formula, where statsmodels splits it across
``groups`` / ``re_formula`` / ``vc_formula``.

.. code-block:: text

    statsmodels : formula="pi ~ age_c", groups="territory",
                  re_formula="1 + age_c", vc_formula={"subject": "0 + C(subject_uid)"}
    lme4        : pi ~ age_c + (1 + age_c | territory) + (1 | subject_uid)

One expression, harder to get wrong — which is the reason to want it. It also brings Satterthwaite
degrees of freedom and p-values (through ``lmerTest``), which statsmodels' MixedLM does not provide.

Availability
------------
Nothing here imports R at module load. :func:`r_backend_status` probes the whole chain — the Python
package, the R interpreter, and the R packages — and reports precisely what is missing, so the GUI
can offer the engine only when it will actually work and explain itself when it will not.

:func:`mixedlm_to_lme4_formula` and :func:`lme4_fixed_formula` are pure string manipulation and work
with or without R installed.

Supported ``pymer4`` versions
----------------------------
Written against **0.9.x**, which is close to a rewrite of 0.8:

===================  ==========================  ==================================
                     0.8.x                       0.9.x
===================  ==========================  ==================================
model class          ``Lmer`` / ``Glmer``        ``lmer`` / ``glmer`` (lowercase)
data frames          pandas                      **polars**
coefficient table    ``.coefs``                  ``.result_fit`` (easystats)
fit statistics       ``.AIC`` / ``.logLike``     ``.result_fit_stats`` (broom glance)
convergence          ``.warnings``               ``.convergence_status``
R model handle       ``.model_obj``              ``.r_model``
REML                 ``fit(REML=…)``             not exposed — always on for ``lmer``
===================  ==========================  ==================================

0.9 also grew a large R dependency surface (the easystats suite on top of lme4), which
:data:`REQUIRED_R_PACKAGES` tracks so the availability probe reports what is genuinely needed.

Column readers accept either generation's spelling where that is cheap; the fitting path targets
0.9 and raises a clear error on an incompatible install.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

from .mixedlm import significance_stars

log = Logger()

ANALYSIS_LME4 = "lme4"

# R packages pymer4 0.9 resolves eagerly at import time — every one of them must be present or
# ``import pymer4`` itself fails. lmerTest is what turns lme4's t-statistics into p-values; the
# insight/parameters/performance/report quartet is the easystats suite its reporting layer is
# built on. ``base``/``stats``/``utils`` are built into R and ``boot`` ships as a recommended
# package, so none of those are listed.
REQUIRED_R_PACKAGES: tuple[str, ...] = (
    "broom",
    "broom.mixed",
    "emmeans",
    "insight",
    "lme4",
    "lmerTest",
    "parameters",
    "performance",
    "report",
    "tibble",
)
OPTIONAL_R_PACKAGES: tuple[str, ...] = ()

INSTALL_HINT = (
    "conda install -c conda-forge r-broom r-broom.mixed r-emmeans r-insight r-lme4 "
    "r-lmertest r-parameters r-performance r-report r-tibble\n"
    "pip install pymer4          (with rpy2 already present)"
)

# A random-effects term: ``(1 | subject)``, ``(1 + age_c | territory)``, ``(0 + x || g)``.
_RANDOM_TERM_RE = re.compile(r"\(\s*[^()|]*\|\|?\s*[^()]*\)")


# ──────────────────────────────────────────────────────────────────────────────
# Availability
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RBackendStatus:
    """What is present, what is missing, and what to do about it."""

    available: bool
    pymer4_version: str = ""
    rpy2_version: str = ""
    r_version: str = ""
    r_packages: Mapping[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    reason: str = ""

    def install_hint(self) -> str:
        """Copy-pasteable commands for whatever is missing."""
        return INSTALL_HINT

    def summary(self) -> str:
        """One-line description for a status bar or a tooltip."""
        if self.available:
            packages = ", ".join(f"{k} {v}" for k, v in sorted(self.r_packages.items()))
            return f"pymer4 {self.pymer4_version} · R {self.r_version} · {packages}"
        return self.reason


def _r_package_versions(names: Sequence[str]) -> dict[str, str]:
    """Ask ``Rscript`` which of *names* are installed. Missing ones map to ``""``."""
    binary = shutil.which("Rscript")
    if not binary:
        return {name: "" for name in names}
    script = ";".join(
        f'cat("{name}=", if (requireNamespace("{name}", quietly=TRUE)) '
        f'as.character(packageVersion("{name}")) else "", "\\n", sep="")'
        for name in names
    )
    try:
        completed = subprocess.run(
            [binary, "-e", script], capture_output=True, text=True, timeout=60, check=False
        )
    except Exception as exc:
        log.debug("Rscript probe failed: %s", exc)
        return {name: "" for name in names}

    found: dict[str, str] = {name: "" for name in names}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if key.strip() in found:
                found[key.strip()] = value.strip()
    return found


def r_backend_status(*, check_r_packages: bool = True) -> RBackendStatus:
    """
    Probe the whole Python → rpy2 → R → lme4 chain.

    Never raises and never imports R as a side effect of importing this module: the GUI calls it to
    decide whether to offer the engine at all, and a broken R installation must not take the window
    down with it.

    Parameters
    ----------
    check_r_packages : bool
        Shell out to ``Rscript`` to check the R package versions. Skip it (a second or two) when you
        only need to know whether the Python side is importable.
    """
    missing: list[str] = []

    try:
        import pymer4  # noqa: F401

        pymer4_version = str(getattr(pymer4, "__version__", "?"))
    except Exception as exc:
        # pymer4 0.9 resolves its Python *and* R dependencies at import time, so this fires for a
        # missing R package (`tibble`) or a missing Python one (`formulae`) just as readily as for
        # an absent pymer4. Report the underlying message: "pymer4 is not installed" would be a
        # lie in every case but the first, and would send the reader off in the wrong direction.
        return RBackendStatus(
            available=False,
            missing=("pymer4",),
            reason=f"pymer4 could not be imported — {type(exc).__name__}: {exc}",
        )

    try:
        import rpy2  # noqa: F401
        import rpy2.situation as situation

        rpy2_version = str(getattr(rpy2, "__version__", "?"))
        r_version = str(situation.r_version_from_subprocess() or "")
    except Exception as exc:
        return RBackendStatus(
            available=False,
            pymer4_version=pymer4_version,
            missing=("rpy2",),
            reason=f"rpy2 is not usable ({type(exc).__name__}: {exc}). It links Python to R.",
        )

    if not shutil.which("R"):
        return RBackendStatus(
            available=False,
            pymer4_version=pymer4_version,
            rpy2_version=rpy2_version,
            missing=("R",),
            reason="No R interpreter on PATH.",
        )

    packages: dict[str, str] = {}
    if check_r_packages:
        packages = _r_package_versions([*REQUIRED_R_PACKAGES, *OPTIONAL_R_PACKAGES])
        missing = [name for name in REQUIRED_R_PACKAGES if not packages.get(name)]
        if missing:
            return RBackendStatus(
                available=False,
                pymer4_version=pymer4_version,
                rpy2_version=rpy2_version,
                r_version=r_version,
                r_packages={k: v for k, v in packages.items() if v},
                missing=tuple(missing),
                reason=f"R package(s) not installed: {', '.join(missing)}.",
            )

    return RBackendStatus(
        available=True,
        pymer4_version=pymer4_version,
        rpy2_version=rpy2_version,
        r_version=r_version,
        r_packages={k: v for k, v in packages.items() if v},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Formula translation
# ──────────────────────────────────────────────────────────────────────────────
def mixedlm_to_lme4_formula(
    formula: str,
    *,
    groups: str,
    re_formula: str = "1",
    vc_formula: Mapping[str, str] | None = None,
) -> str:
    """
    Translate a statsmodels MixedLM specification into one ``lme4`` formula.

    The three statsmodels arguments become bracketed random terms appended to the fixed part:

    * ``groups`` + ``re_formula`` → ``(<re_formula> | <groups>)``. ``re_formula="0"`` means no
      group-level random effect at all, so no term is emitted.
    * each ``vc_formula`` entry → ``(1 | <column>)``. A variance component written the statsmodels
      way, ``"0 + C(subject_uid)"``, is exactly lme4's ``(1 | subject_uid)``: one independent
      intercept per level.

    Examples
    --------
    >>> mixedlm_to_lme4_formula(
    ...     "pi ~ age_c", groups="territory", re_formula="1 + age_c",
    ...     vc_formula={"subject": "0 + C(subject_uid)"})
    'pi ~ age_c + (1 + age_c | territory) + (1 | subject_uid)'
    """
    # Strip any random terms already present, so converting a formula that has been converted before
    # replaces those terms instead of appending a second copy. lme4 accepts a repeated term but it
    # is not the model anyone means: two identical ``(1 | subject)`` terms fit the same variance
    # twice and drive the fit singular.
    fixed = lme4_fixed_formula(str(formula or "").strip())
    if not fixed:
        raise ValueError("A fixed-effects formula is required.")

    terms: list[str] = []
    re_spec = str(re_formula or "").strip()
    if groups and re_spec and re_spec not in {"0", "-1"}:
        terms.append(f"({re_spec} | {groups})")

    for _name, spec in dict(vc_formula or {}).items():
        column = _variance_component_column(spec)
        if column:
            terms.append(f"(1 | {column})")
        else:
            log.warning("Could not translate variance component %r to an lme4 term.", spec)

    # ``groups`` and a variance component can name the same column; emitting it twice is the same
    # mistake by another route.
    deduplicated = list(dict.fromkeys(terms))
    if len(deduplicated) < len(terms):
        log.warning(
            "Dropped %d duplicate random term(s) — the grouping factor and a variance component "
            "named the same column.",
            len(terms) - len(deduplicated),
        )
    if not deduplicated:
        raise ValueError(
            "The specification has no random effects, so it is not a mixed model — "
            "use the plain linear model instead."
        )
    return " + ".join([fixed, *deduplicated])


def _variance_component_column(spec: str) -> str:
    """Column named by a statsmodels variance-component formula (``"0 + C(subject_uid)"``)."""
    text = str(spec or "").strip()
    match = re.search(r"C\(\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if match:
        return match.group(1)
    # A bare column name, with any leading "0 +" / "-1 +" dropped.
    bare = re.sub(r"^\s*(0|-1)\s*\+\s*", "", text).strip()
    return bare if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", bare) else ""


def lme4_fixed_formula(formula: str) -> str:
    """
    Strip the random terms from an ``lme4`` formula, leaving the fixed part.

    Needed to build a patsy design matrix for prediction grids: R holds the model, but the marginal
    means and their standard errors are computed on the Python side from the fixed-effects design.

    >>> lme4_fixed_formula("pi ~ age_c + (1 + age_c | territory) + (1 | subject_uid)")
    'pi ~ age_c'
    """
    stripped = _RANDOM_TERM_RE.sub("", str(formula or ""))
    # Collapse the "+ +" and trailing "+" left behind by the removals.
    stripped = re.sub(r"\+\s*\+", "+", stripped)
    stripped = re.sub(r"\+\s*$", "", stripped).strip()
    stripped = re.sub(r"~\s*\+", "~", stripped).strip()
    if stripped.endswith("~"):
        stripped += " 1"
    return re.sub(r"\s+", " ", stripped)


def lme4_random_terms(formula: str) -> list[str]:
    """The bracketed random terms of an ``lme4`` formula, in order."""
    return [m.group(0) for m in _RANDOM_TERM_RE.finditer(str(formula or ""))]


def lme4_grouping_factors(formula: str) -> list[str]:
    """Grouping columns named on the right of each ``|`` in an ``lme4`` formula."""
    out: list[str] = []
    for term in lme4_random_terms(formula):
        _lhs, _, rhs = term.strip("() ").rpartition("|")
        name = rhs.strip()
        if name and name not in out:
            out.append(name)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────────────
def fit_lme4(
    *,
    data: pd.DataFrame,
    formula: str,
    family: str = "gaussian",
    reml: bool = True,
    control: str = "",
    dropna_columns: Sequence[str] | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """
    Fit ``lme4::lmer`` / ``glmer`` through ``pymer4``.

    Parameters
    ----------
    formula : str
        An lme4 formula, random terms included:
        ``pi ~ age_c + (1 + age_c | territory) + (1 | subject_uid)``.
    family : str
        ``gaussian`` uses ``lmer``; anything else (``binomial``, ``poisson``, ``gamma``,
        ``inverse_gaussian``) uses ``glmer``.
    reml : bool
        Restricted maximum likelihood. Leave it on for variance estimates; turn it off to compare
        models that differ in their *fixed* effects by likelihood, where REML fits are not
        comparable.
    control : str
        Raw ``lme4::lmerControl(...)`` string, for convergence trouble.

    Returns
    -------
    (model, model_df, metadata)
        *model* is the ``pymer4`` ``Lmer``; *metadata* matches the other engines so the GUI's
        row-count reporting is identical.

    Raises
    ------
    RuntimeError
        If the R backend is not available; the message names what is missing.
    """
    status = r_backend_status()
    if not status.available:
        raise RuntimeError(
            f"The R/lme4 engine is not available: {status.reason}\n\n{status.install_hint()}"
        )
    model_class, family_kwargs = _resolve_model_class(family)

    fixed = lme4_fixed_formula(formula)
    needed = _formula_columns(data.columns, formula)
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Formula references columns that are not in the frame: {missing}")

    na_cols = list(dropna_columns) if dropna_columns is not None else needed
    na_cols = [c for c in na_cols if c in data.columns]
    n_input = int(len(data))
    dropped_by_column = {c: int(data[c].isna().sum()) for c in na_cols if bool(data[c].isna().any())}
    df = data.dropna(subset=na_cols).reset_index(drop=True) if na_cols else data.copy()
    if not len(df):
        raise ValueError(f"No complete rows left after dropping missing values in {na_cols}.")

    # lme4 requires the grouping columns to be factors; a numeric subject id would be read as a
    # covariate and silently fit something else entirely.
    for factor in lme4_grouping_factors(formula):
        if factor in df.columns:
            df[factor] = df[factor].astype(str)

    # pymer4 0.9 is polars end to end: a pandas frame reaches ``.with_columns`` and dies.
    model = model_class(formula, data=_to_polars(df), **family_kwargs)
    if not reml and str(family or "gaussian") == "gaussian":
        log.warning(
            "pymer4 0.9 does not expose REML through its API — lmerTest::lmer's default (REML=TRUE) "
            "is used regardless. Fit with the statsmodels MixedLM engine if you need ML."
        )
    fit_kwargs: dict[str, Any] = {}
    if control:
        fit_kwargs["control"] = control
    model.fit(**fit_kwargs)

    meta = {
        "engine": ANALYSIS_LME4,
        "formula": formula,
        "fixed_formula": fixed,
        "family": str(family or "gaussian"),
        "reml": bool(reml),
        "grouping_factors": lme4_grouping_factors(formula),
        "n_rows_input": n_input,
        "n_rows": int(len(df)),
        "n_rows_dropped": n_input - int(len(df)),
        "dropna_columns": na_cols,
        "dropped_by_column": dropped_by_column,
        "loaded": False,
        "backend": status.summary(),
        "reml": bool(reml) and str(family or "gaussian") == "gaussian",
    }
    converged, message = _convergence(model)
    if not converged:
        meta["warnings"] = [message]
        log.warning("lme4: %s", message)
    log.info("lme4 fit: %s (n=%d)", formula, len(df))
    return model, df, meta


def _resolve_model_class(family: str) -> tuple[Any, dict[str, Any]]:
    """
    The ``pymer4`` model class for *family*, plus the kwargs its constructor needs.

    ``gaussian`` goes to ``lmer`` (via lmerTest, so the coefficients carry Satterthwaite p-values);
    every other family goes to ``glmer``.
    """
    import pymer4.models as models

    gaussian = str(family or "gaussian").strip().lower() in {"", "gaussian", "normal"}
    name = "lmer" if gaussian else "glmer"
    model_class = getattr(models, name, None)
    if model_class is None:
        # 0.8 spelled them ``Lmer`` / ``Glmer``; its data model differs enough that the rest of this
        # module would not work against it, so fail with something actionable.
        legacy = getattr(models, name.capitalize(), None)
        raise RuntimeError(
            "This pymer4 does not expose "
            f"pymer4.models.{name}"
            + (
                " — it looks like the 0.8 series, which this engine no longer supports. "
                "Upgrade with: pip install -U pymer4"
                if legacy is not None
                else f". Available: {', '.join(sorted(n for n in dir(models) if not n.startswith('_')))}"
            )
        )
    return model_class, ({} if gaussian else {"family": str(family)})


def _to_polars(frame: pd.DataFrame):
    """
    Convert a pandas frame to polars, which is what pymer4 0.9 works in.

    Column names are made unique first: polars refuses a frame with repeated names, and its error
    ("Pandas dataframe contains non-unique indices and/or column names") does not say which name is
    the problem.
    """
    import polars as pl

    from .frame_ops import ensure_unique_columns

    if isinstance(frame, pl.DataFrame):
        return frame
    # Categoricals (the binned columns) round-trip fine; object columns become strings.
    return pl.from_pandas(ensure_unique_columns(frame, context="analysis dataframe"))


def _to_pandas(frame: Any) -> pd.DataFrame:
    """Convert a polars frame (or anything already pandas) to pandas."""
    if isinstance(frame, pd.DataFrame):
        return frame
    to_pandas = getattr(frame, "to_pandas", None)
    if to_pandas is not None:
        return to_pandas()
    return pd.DataFrame(frame)


def _convergence(model: Any) -> tuple[bool, str]:
    """``(converged, message)`` across both pymer4 generations."""
    status = getattr(model, "convergence_status", None)
    if isinstance(status, tuple) and status:
        return bool(status[0]), str(status[1]) if len(status) > 1 else ""
    if status is not None:
        return bool(status), ""
    warnings = list(getattr(model, "warnings", None) or [])
    return (not warnings), "; ".join(str(w) for w in warnings)


def _formula_columns(columns: Sequence[str], formula: str) -> list[str]:
    """Frame columns an lme4 formula mentions, random terms included."""
    available = {str(c) for c in columns}
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(formula or "")))
    return sorted(tokens & available)


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
def _normalize(name: Any) -> str:
    """Reduce a column name to letters and digits, lowercased: ``"Std. Error"`` → ``"stderror"``."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Column aliases, held as *normalized* names so a spelling this code has never seen still matches.
# Every generation of pymer4, broom and easystats punctuates these differently — ``SE`` / ``Std.
# Error`` / ``std.error``, ``CI_low`` / ``2.5_ci`` / ``conf.low``, ``p`` / ``P-val`` / ``Pr(>|t|)``
# — and matching literal strings meant a near miss produced a silently empty column.
_COEF_ALIASES: dict[str, tuple[str, ...]] = {
    "parameter": ("parameter", "term", "name", "effect"),
    "coef": ("coefficient", "estimate", "coef", "beta", "value"),
    "std_err": ("se", "stderror", "stderr", "standarderror", "stddev"),
    "ci_low": ("cilow", "25ci", "conflow", "lower", "lowerci", "lowercl", "asymplcl", "q25"),
    "ci_high": ("cihigh", "975ci", "confhigh", "upper", "upperci", "uppercl", "asympucl", "q975"),
    "z": ("tstat", "zstat", "statistic", "tvalue", "zvalue", "t", "z"),
    "p_value": ("p", "pval", "pvalue", "prt", "prz", "prchisq"),
    "df": ("df", "dferror", "dfresidual", "dfresid"),
}


def _pick(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series | None:
    """
    Column of *frame* whose normalized name is in *names*.

    Exact matches win over normalized ones, and the alias order is honoured, so a frame carrying
    both ``p`` and ``p.value`` resolves deterministically.
    """
    for name in names:
        if name in frame.columns:
            return frame[name]
    normalized = {_normalize(c): c for c in reversed(list(frame.columns))}
    for name in names:
        column = normalized.get(_normalize(name))
        if column is not None:
            return frame[column]
    return None


def lme4_coef_frame(model: Any) -> pd.DataFrame:
    """
    ``pymer4``'s fixed-effects table normalized to this package's column names.

    0.9 puts it in ``.result_fit`` as a polars frame from easystats ``model_parameters()``, with the
    term names in a ``Parameter`` *column*; 0.8 put it in ``.coefs`` as a pandas frame with the terms
    in the index. Both are accepted.
    """
    raw = getattr(model, "result_fit", None)
    if raw is None:
        raw = getattr(model, "coefs", None)
    if raw is None:
        raise ValueError("The model has no coefficient table — did the fit run?")

    coefs = _to_pandas(raw)
    if coefs.empty:
        raise ValueError("The model's coefficient table is empty — did the fit run?")

    names = _pick(coefs, _COEF_ALIASES["parameter"])
    out = pd.DataFrame(
        {"parameter": [str(v) for v in (names if names is not None else coefs.index)]}
    )
    for target in ("coef", "std_err", "z", "p_value", "ci_low", "ci_high", "df"):
        series = _pick(coefs, _COEF_ALIASES[target])
        out[target] = np.nan if series is None else pd.to_numeric(series, errors="coerce").to_numpy()

    missing = [c for c in ("std_err", "z", "p_value", "ci_low", "ci_high") if out[c].isna().all()]
    if missing:
        log.debug(
            "Coefficient columns not found (%s); available: %s",
            ", ".join(missing),
            ", ".join(str(c) for c in coefs.columns),
        )
    _complete_coef_frame(out)

    out["sig"] = [significance_stars(float(p)) for p in out["p_value"]]
    return out[["parameter", "coef", "std_err", "z", "p_value", "ci_low", "ci_high", "sig", "df"]]


def _complete_coef_frame(out: pd.DataFrame) -> None:
    """
    Fill in whichever inference columns the source table did not carry, in place.

    These quantities determine one another, so a table holding any two of coefficient, standard
    error, test statistic and interval implies the rest. Reconstructing them beats showing a column
    of dashes — and every relation used here is exact, not an approximation:

    ``se = coef / z``  ·  ``se = (ci_high − ci_low) / 2·crit``  ·  ``z = coef / se``  ·
    ``p = 2·P(T > |z|)`` on the reported degrees of freedom, falling back to the normal.
    """
    from scipy.stats import norm, t as student_t

    coef, se, z = out["coef"], out["std_err"], out["z"]

    # Standard error from the statistic, or from the width of the interval.
    need = se.isna() & coef.notna() & z.notna() & (z != 0)
    if need.any():
        out.loc[need, "std_err"] = (coef[need] / z[need]).abs()
    se = out["std_err"]
    need = se.isna() & out["ci_low"].notna() & out["ci_high"].notna()
    if need.any():
        crit = np.where(out.loc[need, "df"].notna(),
                        student_t.ppf(0.975, out.loc[need, "df"].clip(lower=1e-6)), 1.959963985)
        out.loc[need, "std_err"] = (out.loc[need, "ci_high"] - out.loc[need, "ci_low"]) / (2 * crit)
    se = out["std_err"]

    need = out["z"].isna() & coef.notna() & se.notna() & (se != 0)
    if need.any():
        out.loc[need, "z"] = coef[need] / se[need]
    z = out["z"]

    # Two-sided p-value on the reported degrees of freedom; lmerTest's Satterthwaite df make this a
    # t-test, and without them the normal approximation is the honest fallback.
    need = out["p_value"].isna() & z.notna()
    if need.any():
        df = out.loc[need, "df"]
        out.loc[need, "p_value"] = np.where(
            df.notna(),
            2 * student_t.sf(np.abs(z[need]), df.clip(lower=1e-6).fillna(1.0)),
            2 * norm.sf(np.abs(z[need])),
        )

    # Wald interval, on the t distribution when the degrees of freedom are known.
    need = (out["ci_low"].isna() | out["ci_high"].isna()) & coef.notna() & se.notna()
    if need.any():
        crit = np.where(out.loc[need, "df"].notna(),
                        student_t.ppf(0.975, out.loc[need, "df"].clip(lower=1e-6)), 1.959963985)
        out.loc[need, "ci_low"] = coef[need] - crit * se[need]
        out.loc[need, "ci_high"] = coef[need] + crit * se[need]


def lme4_random_effects_frame(model: Any) -> pd.DataFrame:
    """
    ``pymer4``'s variance components normalized to component/kind/var/sd.

    0.9 fills ``.ranef_var`` from ``broom.mixed::tidy(effects="ran_pars")``, which reports **standard
    deviations**, not variances — the terms are named ``sd__(Intercept)``, ``cor__…``,
    ``sd__Observation``. Squaring the SD to recover the variance (rather than reading ``estimate`` as
    one) is the difference between reporting 0.09 and 0.0084. Correlation rows are kept but marked,
    since squaring those would be meaningless.
    """
    rows: list[dict[str, Any]] = []
    raw = getattr(model, "ranef_var", None)
    if raw is None:
        return pd.DataFrame(rows, columns=["component", "kind", "var", "sd"])

    frame = _to_pandas(raw)
    if frame.empty:
        return pd.DataFrame(rows, columns=["component", "kind", "var", "sd"])

    groups = _pick(frame, ("group", "grp"))
    terms = _pick(frame, ("term", "name"))
    estimates = _pick(frame, ("estimate", "std", "sd", "stddev", "var", "variance"))
    legacy_var = _pick(frame, ("var", "variance"))
    if estimates is None:
        log.debug(
            "Random-effects columns not recognised; available: %s",
            ", ".join(str(c) for c in frame.columns),
        )

    for i in range(len(frame)):
        group = str(groups.iloc[i]) if groups is not None else str(frame.index[i])
        term = str(terms.iloc[i]) if terms is not None else ""
        value = float(estimates.iloc[i]) if estimates is not None else np.nan

        is_correlation = term.startswith("cor__")
        label_term = re.sub(r"^(sd|cor)__", "", term)
        label = f"{group} · {label_term}" if label_term and label_term != group else group
        residual = group.lower() in {"residual", "observation"} or label_term == "Observation"

        if is_correlation:
            kind, variance, sd = "correlation", np.nan, np.nan
            label = f"{group} · cor({label_term})"
        elif legacy_var is not None and not term.startswith("sd__"):
            # 0.8 reported the variance directly.
            variance = float(legacy_var.iloc[i])
            sd = float(np.sqrt(max(variance, 0.0)))
            kind = "residual" if residual else "ranef"
        else:
            sd = value
            variance = float(value**2) if np.isfinite(value) else np.nan
            kind = "residual" if residual else "ranef"

        rows.append({"component": label, "kind": kind, "var": variance, "sd": sd})
    return pd.DataFrame(rows, columns=["component", "kind", "var", "sd"])


def _fit_statistic(model: Any, names: Sequence[str]) -> float | None:
    """Pull a scalar fit statistic from ``.result_fit_stats`` (0.9) or an attribute (0.8)."""
    stats = getattr(model, "result_fit_stats", None)
    if stats is not None:
        frame = _to_pandas(stats)
        for name in names:
            if name in frame.columns and len(frame):
                value = pd.to_numeric(frame[name], errors="coerce").iloc[0]
                if pd.notna(value):
                    return float(value)
    for name in names:
        value = getattr(model, name, None)
        if value is not None and np.isscalar(value):
            return float(value)
    return None


def lme4_group_coefficients(model: Any) -> pd.DataFrame:
    """
    Per-group coefficients of an ``lme4`` fit — each level's own intercept and slopes.

    ``lme4`` already computes both: ``coef()`` gives the total per level (what pymer4 stores as
    ``.fixef``) and ``ranef()`` the deviation from the population average (``.ranef``). With several
    grouping factors each is a mapping keyed by factor, so the ``factor`` column keeps them apart —
    a model with ``(1 + age_c | territory) + (1 | subject_uid)`` lists both here.

    Returns
    -------
    pandas.DataFrame
        ``factor``, ``level``, then ``<term>`` (total) and ``<term>_dev`` (deviation) per term.
        Empty when the model exposes neither.
    """
    totals = getattr(model, "fixef", None)
    deviations = getattr(model, "ranef", None)
    if totals is None and deviations is None:
        return pd.DataFrame(columns=["factor", "level"])

    # pymer4 keys both by grouping factor only when there are several; a single factor arrives as a
    # bare frame, so recover its name from the formula rather than labelling it "Group".
    try:
        factors = lme4_grouping_factors(str(getattr(model, "formula", "") or ""))
    except Exception:  # a malformed/absent formula must not cost us the table
        factors = []
    fallback = factors[0] if len(factors) == 1 else "Group"

    total_map = _group_effect_mapping(totals, fallback)
    deviation_map = _group_effect_mapping(deviations, fallback)

    frames: list[pd.DataFrame] = []
    for factor in dict.fromkeys([*total_map, *deviation_map]):
        total = total_map.get(factor)
        deviation = deviation_map.get(factor)
        reference = total if total is not None and not total.empty else deviation
        if reference is None or reference.empty:
            continue

        # Join on the level rather than by position: coef() and ranef() agree today, but a merge
        # keeps the two columns of a row describing the same group whatever order they arrive in.
        out = pd.DataFrame({"level": _levels_of(reference)})
        for term in [c for c in reference.columns if _normalize(c) != "level"]:
            for frame, suffix in ((total, ""), (deviation, "_dev")):
                if frame is None or term not in frame.columns:
                    continue
                values = pd.DataFrame({
                    "level": _levels_of(frame),
                    f"{term}{suffix}": pd.to_numeric(frame[term], errors="coerce").to_numpy(),
                })
                out = out.merge(values, on="level", how="left")
        out.insert(0, "factor", factor)
        frames.append(out)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["factor", "level"])


def _group_effect_mapping(value: Any, fallback: str) -> dict[str, pd.DataFrame]:
    """Normalize ``.fixef`` / ``.ranef`` to ``{factor: frame}``, whichever shape pymer4 returned."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): _to_pandas(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {str(i): _to_pandas(v) for i, v in enumerate(value)}
    return {fallback: _to_pandas(value)}


def _levels_of(frame: pd.DataFrame) -> list[str]:
    """Group level names: the ``level`` column pymer4 builds from R's row names, else the index."""
    for column in frame.columns:
        if _normalize(column) in {"level", "grp", "group"}:
            return [str(v) for v in frame[column]]
    return [str(v) for v in frame.index]


def lme4_info_dict(
    model: Any,
    *,
    outcome_name: str = "Outcome",
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report structure matching :func:`~nvitk.stats.mixedlm.mixedlm_info_dict`, from an ``Lmer``."""
    meta = dict(meta or {})
    factors = list(meta.get("grouping_factors") or [])

    # ``.ranef`` is one frame per grouping factor (a dict when there are several); each row is one
    # level, so its length is the number of groups.
    n_groups = None
    ranef = getattr(model, "ranef", None)
    if isinstance(ranef, dict) and ranef:
        n_groups = sum(len(_to_pandas(v)) for v in ranef.values())
    elif ranef is not None:
        try:
            n_groups = int(len(_to_pandas(ranef)))
        except Exception:
            n_groups = None

    fit_stats: dict[str, Any] = {}
    for key, names in (
        ("aic", ("AIC",)),
        ("bic", ("BIC",)),
        ("llf", ("logLik", "logLike")),
        ("resid_sd", ("sigma",)),
    ):
        value = _fit_statistic(model, names)
        if value is not None:
            fit_stats[key] = value
    fit_stats["converged"] = _convergence(model)[0]

    return {
        "header": {
            "formula": meta.get("formula") or getattr(model, "formula", None),
            "n_obs": int(meta.get("n_rows") or len(getattr(model, "data", []) or [])),
            "n_groups": n_groups,
            "group_name": " + ".join(factors) if factors else "Group",
            "outcome_name": outcome_name,
            "vc_group_name": "Random effects",
            "engine": "R · lme4",
            "backend": meta.get("backend", ""),
        },
        "fixed_effects": lme4_coef_frame(model),
        "random_effects": lme4_random_effects_frame(model),
        "cov_re": pd.DataFrame(),
        "fit_statistics": fit_stats,
        "has_vcomp": True,
        "group_effects": lme4_group_coefficients(model),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────
def lme4_fixef_vcov(model: Any) -> pd.DataFrame | None:
    """
    Fixed-effects covariance matrix, pulled out of the underlying R model.

    Needed for confidence bands on predictions. Returns ``None`` when it cannot be reached, so the
    caller can draw the curves without intervals rather than failing outright.
    """
    r_model = getattr(model, "r_model", None) or getattr(model, "model_obj", None)
    if r_model is None:
        return None
    try:
        from rpy2.robjects import r as R_
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import default_converter

        matrix = R_["as.matrix"](R_["vcov"](r_model))
        with localconverter(default_converter + pandas2ri.converter):
            values = np.asarray(matrix)
        names = list(lme4_coef_frame(model)["parameter"])
        if values.shape[0] != len(names):
            return None
        return pd.DataFrame(values, index=names, columns=names)
    except Exception as exc:
        log.debug("Could not read vcov from the R model: %s", exc)
        return None


# emmeans is asked for marginal means in R and hands back a tidy frame, which is the only route to a
# standard error for an lme4 prediction: ``predict.merMod`` returns none, and deriving one from the
# coefficient table would ignore the covariance between terms.
_R_EMMEANS_HELPER = """
.nvitk_lme4_emmeans <- function(model, specs, at_name, at_values, lvl) {
  # Satterthwaite rather than emmeans' Kenward-Roger default: KR needs pbkrtest, which is not part
  # of this engine's dependency set, and lmerTest (which is) provides Satterthwaite.
  suppressMessages(try(emmeans::emm_options(lmer.df = "satterthwaite"), silent = TRUE))
  spec_f <- stats::as.formula(specs)
  em <- if (nzchar(at_name)) {
    emmeans::emmeans(model, specs = spec_f, at = stats::setNames(list(at_values), at_name))
  } else {
    emmeans::emmeans(model, specs = spec_f)
  }
  as.data.frame(summary(em, level = lvl, infer = c(TRUE, FALSE)))
}
"""

_EMMEANS_HELPER_LOADED = False


def lme4_emmeans(
    model: Any,
    specs: str,
    *,
    at_name: str = "",
    at_values: Sequence[float] | None = None,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """
    Estimated marginal means and their confidence intervals, from ``emmeans`` in R.

    ``lme4`` gives predictions but no standard errors, so this is what makes a confidence band on an
    lme4 plot possible at all.

    Parameters
    ----------
    specs : str
        An emmeans specification formula: ``"~ territory"``, or ``"~ age_c | territory"`` to get one
        curve per level.
    at_name, at_values
        Evaluate a *continuous* predictor over a grid — ``at_name="age_c"`` with the x values the
        line is drawn at. Leave unset for a purely categorical specification.
    ci_level : float
        Confidence level for the returned interval.

    Returns
    -------
    pandas.DataFrame
        The grid columns plus ``emmean``, ``SE``, ``df``, ``lower.CL``, ``upper.CL``.
    """
    global _EMMEANS_HELPER_LOADED

    r_model = getattr(model, "r_model", None) or getattr(model, "model_obj", None)
    if r_model is None:
        raise ValueError("This model does not expose an underlying R object.")

    from rpy2.robjects import FloatVector, default_converter, globalenv, pandas2ri
    from rpy2.robjects import r as R_
    from rpy2.robjects.conversion import localconverter

    if not _EMMEANS_HELPER_LOADED:
        R_(_R_EMMEANS_HELPER)
        _EMMEANS_HELPER_LOADED = True

    values = FloatVector(list(at_values) if at_values is not None else [])
    with localconverter(default_converter + pandas2ri.converter):
        return pd.DataFrame(
            globalenv[".nvitk_lme4_emmeans"](
                r_model, str(specs), str(at_name), values, float(ci_level)
            )
        )


def _emmeans_band(
    model: Any,
    *,
    x: str,
    x_values: Sequence[Any],
    group: str,
    levels: Sequence[Any],
    continuous: bool,
    fixed_formula: str,
    ci_level: float,
    emmeans_fn: Any = None,
) -> dict[str | None, pd.DataFrame] | None:
    """
    Marginal means with intervals, one frame per group level (``None`` keyed for the population).

    The grouping factor is only asked for level by level when it appears in the *fixed* part of the
    formula. A factor that enters solely as a random effect has no marginal mean per level for
    emmeans to report — asking anyway just errors, so the population curve is used instead.

    ``emmeans_fn`` selects which engine's emmeans call to make; it defaults to lme4's. Everything
    else here is engine-independent, which is why the robust engine reuses this function rather than
    repeating the grouped/population logic.
    """
    grouped = bool(group) and group != x and bool(re.search(rf"\b{re.escape(group)}\b", fixed_formula))
    specs = f"~ {x}" + (f" | {group}" if grouped else "")
    try:
        frame = (emmeans_fn or lme4_emmeans)(
            model,
            specs,
            at_name=x if continuous else "",
            at_values=list(x_values) if continuous else None,
            ci_level=ci_level,
        )
    except Exception as exc:
        log.warning("Could not obtain marginal means from emmeans: %s", exc)
        log.debug("emmeans failure", exc_info=True)
        return None

    if not grouped or group not in frame.columns:
        return {None: frame}
    out: dict[str | None, pd.DataFrame] = {}
    for level in levels:
        subset = frame.loc[frame[group].astype(str) == str(level)]
        if not subset.empty:
            out[str(level)] = subset
    return out or {None: frame}


def _emmeans_columns(frame: pd.DataFrame) -> tuple[str, str, str] | None:
    """``(estimate, lower, upper)`` column names of an emmeans frame, whatever it called them."""
    estimate = next((c for c in ("emmean", "response", "estimate", "rate", "prob") if c in frame.columns), None)
    lower = next((c for c in ("lower.CL", "asymp.LCL", "lower.HPD", "lower") if c in frame.columns), None)
    upper = next((c for c in ("upper.CL", "asymp.UCL", "upper.HPD", "upper") if c in frame.columns), None)
    return (estimate, lower, upper) if estimate and lower and upper else None


def lme4_predict(
    model: Any,
    grid: pd.DataFrame,
    *,
    use_random_effects: bool = False,
) -> np.ndarray:
    """
    Predict on *grid* through ``pymer4``.

    ``use_random_effects`` selects between the population-level prediction and one that includes the
    fitted group deviations. 0.9 passes straight through to R's ``predict`` (where the switch is
    ``re.form=NA`` for population level); 0.8 spelled it ``use_rfx``. Each is tried in turn, so the
    call works on either.
    """
    predict = getattr(model, "predict", None)
    if predict is None:
        raise ValueError("This model object cannot predict.")

    # 0.9 wants polars; 0.8 wanted pandas. Hand over whatever the installed version accepts.
    for frame in (_to_polars(grid), grid):
        for kwargs in (
            {} if use_random_effects else {"re_form": _r_na()},
            {"use_rfx": use_random_effects, "verify_predictions": False},
            {"use_rfx": use_random_effects},
            {},
        ):
            try:
                return np.asarray(predict(frame, **kwargs), dtype=float)
            except TypeError:
                continue
            except Exception:
                break
    raise ValueError("Could not call predict() on this pymer4 version.")


def _r_na():
    """R's ``NA``, which ``predict(re.form=NA)`` needs for a population-level prediction."""
    from rpy2.robjects import NA_Logical

    return NA_Logical


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
# Why this does not reuse ``plot_mixedlm_params``: that function builds a patsy design matrix and
# indexes it by the model's parameter names. R names factor contrasts ``territoryPCA`` where patsy
# names them ``C(territory)[T.PCA]``, so the two would silently fail to line up and every contrast
# would be dropped to zero. Predicting through ``pymer4`` instead keeps R's own encoding.
def plot_lme4_params(
    *,
    model: Any,
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
    title: str = "lme4",
    x_label: str | None = None,
    y_label: str | None = None,
    predict_fn: Any = None,
    band_fn: Any = None,
    population_label: str = "Population (fixed effects)",
    excluded_points: pd.DataFrame | None = None,
) -> Any:
    """
    Population and per-group curves for an ``lme4`` fit, predicted through ``pymer4``.

    Mirrors :func:`~nvitk.stats.mixedlm.plot_mixedlm_params` visually — dashed black population
    curve, one coloured curve per group, observed data in a lighter tone — but gets its numbers from
    R rather than from a locally rebuilt design matrix.

    Parameters
    ----------
    errorbar : bool
        Draw confidence intervals. ``lme4``'s ``predict`` returns no standard error, so these come
        from :func:`lme4_emmeans` — marginal means computed by ``emmeans`` in R. A band appears per
        group when the grouping factor is in the fixed part of the formula, and one population band
        otherwise. When emmeans cannot supply them the curves are still drawn and the figure carries
        ``ci_error``.
    fixed_formula : str
        The formula's fixed part, used to decide whether the grouping factor has marginal means to
        report level by level.
    display : {"overview", "grouped"}
        ``"overview"`` puts every level on one pair of axes; ``"grouped"`` splits them into a grid of
        anatomical panels (see :mod:`nvitk.stats.region_groups`), each autoscaled to its own range.
        Both draw the same fit — the population curve and the emmeans intervals are computed once and
        repeat across panels.
    excluded_points : pandas.DataFrame, optional
        Observations a filter removed, drawn in grey beneath the kept ones. They are not in
        *df_fit* — the fit ran on the filtered frame — so they have to be supplied separately.
    predict_fn, band_fn, population_label
        Engine hooks. They default to ``lme4``'s, and exist so another R engine whose parameters are
        named by R rather than by patsy — :func:`~nvitk.stats.r_robust.plot_lmrob_params` — reuses
        this drawing code instead of copying it. Not needed for ordinary ``lme4`` use.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .mixedlm import _lighten, _natural_sort_key

    df = df_fit.copy()
    for column in (x, y):
        if column not in df.columns:
            raise ValueError(f"Column {column!r} is not in the fitted frame.")
    grouped = bool(group) and group in df.columns and group != x

    levels = (
        list(group_order)
        if group_order is not None
        else sorted(df[group].dropna().astype(str).unique(), key=_natural_sort_key)
        if grouped
        else []
    )
    if restrict_to_orders and grouped and levels:
        df = df.loc[df[group].astype(str).isin({str(v) for v in levels})]
        if df.empty:
            raise ValueError("No rows left after restricting to the selected levels.")

    if mode == "auto":
        mode = "continuous" if pd.api.types.is_numeric_dtype(df[x]) else "categorical"

    display = str(display or "overview").strip().lower()
    if display not in {"overview", "grouped"}:
        raise ValueError("display must be one of: overview, grouped")

    # A prediction grid varying x, with everything else held at a reference value: the mean for
    # numeric columns, the most frequent level for categorical ones. Built once from the whole
    # frame — a per-panel reference would pin covariates differently in each panel and the
    # population curve would stop being the same line everywhere.
    def reference_row() -> dict[str, Any]:
        row: dict[str, Any] = {}
        for column in df.columns:
            if column in {x, y}:
                continue
            series = df[column].dropna()
            if series.empty:
                continue
            if pd.api.types.is_numeric_dtype(series):
                row[column] = float(series.mean())
            else:
                row[column] = series.mode().iloc[0]
        return row

    base = reference_row()
    if mode == "continuous":
        x_values = np.linspace(float(df[x].min()), float(df[x].max()), 200)
    else:
        x_values = np.array(sorted(df[x].dropna().astype(str).unique(), key=_natural_sort_key), dtype=object)
    positions = np.arange(len(x_values)) if mode == "categorical" else x_values

    predict_fn = predict_fn or lme4_predict

    def predict_at(level: str | None) -> np.ndarray | None:
        """Prediction along the grid, for one group level or for the population."""
        grid = pd.DataFrame([{**base, x: value} for value in x_values])
        if grouped and level is not None:
            grid[group] = str(level)
        try:
            return predict_fn(model, grid, use_random_effects=level is not None)
        except Exception as exc:
            log.debug("%s prediction failed for %s=%s: %s", population_label, group, level, exc)
            return None

    errors: dict[str, str] = {}
    population = predict_at(None)

    # Intervals from emmeans, keyed by group level (or ``None`` for a single population band).
    # Computed once for every level; each panel then reads the keys it owns.
    bands: dict[str | None, pd.DataFrame] | None = None
    if errorbar:
        bands = (band_fn or _emmeans_band)(
            model,
            x=x,
            x_values=x_values,
            group=group if grouped else "",
            levels=levels,
            continuous=mode == "continuous",
            fixed_formula=fixed_formula or "",
            ci_level=ci_level,
        )
        if bands is None:
            errors["ci_error"] = (
                "emmeans could not produce marginal means for this model, so no confidence band is "
                "shown. Check that the R package 'emmeans' is installed."
            )

    def draw_panel(
        ax: Any, panel_df: pd.DataFrame, panel_levels: Sequence[str], panel_title: str
    ) -> None:
        """Draw the population curve and *panel_levels* onto *ax*, from *panel_df*'s observations."""
        colors = sns.color_palette(palette, n_colors=max(len(panel_levels), 3))
        cmap = {str(lev): colors[i % len(colors)] for i, lev in enumerate(panel_levels)}

        # Drawn first so the kept observations sit on top of them.
        if (
            include_points and excluded_points is not None and not excluded_points.empty
            and {x, y} <= set(excluded_points.columns)
        ):
            dropped = excluded_points
            if grouped and group in dropped.columns:
                dropped = dropped.loc[dropped[group].astype(str).isin({str(v) for v in panel_levels})]
            if not dropped.empty:
                if mode == "continuous":
                    ax.scatter(
                        pd.to_numeric(dropped[x], errors="coerce"),
                        pd.to_numeric(dropped[y], errors="coerce"),
                        s=18, alpha=0.45, color="#B0B0B0", linewidths=0, zorder=1,
                        label="excluded by filter",
                    )
                else:
                    positions_by_level = {str(v): i for i, v in enumerate(x_values)}
                    xi = dropped[x].astype(str).map(positions_by_level)
                    keep = xi.notna()
                    if bool(keep.any()):
                        rng = np.random.default_rng(int(keep.sum()))
                        ax.scatter(
                            xi[keep].to_numpy(float) + rng.uniform(-0.16, 0.16, int(keep.sum())),
                            pd.to_numeric(dropped.loc[keep, y], errors="coerce"),
                            s=16, alpha=0.4, color="#B0B0B0", linewidths=0, zorder=1,
                            label="excluded by filter",
                        )

        if include_points:
            if mode == "continuous":
                sns.scatterplot(
                    data=panel_df, x=x, y=y, hue=group if grouped else None,
                    hue_order=list(panel_levels) or None,
                    palette={k: _lighten(v) for k, v in cmap.items()} if grouped else None,
                    alpha=0.5, s=18, legend=False, ax=ax,
                )
            else:
                order = [str(v) for v in x_values]
                sns.pointplot(
                    data=panel_df, x=x, y=y, hue=group if grouped else None,
                    order=order, hue_order=list(panel_levels) or None,
                    palette={k: _lighten(v) for k, v in cmap.items()} if grouped else None,
                    errorbar=None, dodge=bool(grouped) and len(panel_levels) > 1,
                    linestyles="--", markers="s", legend=False, ax=ax,
                )
                ax.plot([], [], color=_lighten(colors[0]), lw=2.0, ls="--", marker="s",
                        label="Observed mean (unadjusted)")

        def draw_band(frame: pd.DataFrame, color: Any, label: str | None) -> None:
            """Shade (continuous) or whisker (categorical) one emmeans interval."""
            parts = _emmeans_columns(frame)
            if parts is None:
                return
            estimate, lower, upper = parts
            if mode == "continuous":
                sub = frame.sort_values(x)
                ax.fill_between(
                    pd.to_numeric(sub[x], errors="coerce"), sub[lower], sub[upper],
                    color=color, alpha=0.15, linewidth=0, label=label,
                )
            else:
                sub = frame.assign(_xi=frame[x].astype(str).map(
                    {str(v): i for i, v in enumerate(x_values)})).dropna(subset=["_xi"]).sort_values("_xi")
                ax.errorbar(
                    sub["_xi"], sub[estimate],
                    yerr=[sub[estimate] - sub[lower], sub[upper] - sub[estimate]],
                    fmt="none", ecolor=color, elinewidth=1.3, capsize=4, capthick=1.3,
                    alpha=0.85, label=label,
                )

        if population is not None:
            ax.plot(positions, population, color="black", lw=2.8, ls="--", label=population_label)
        if bands and None in bands:
            draw_band(bands[None], "black", f"{int(round(ci_level * 100))}% CI (emmeans)")

        for level in panel_levels:
            curve = predict_at(level)
            if curve is None:
                continue
            ax.plot(positions, curve, color=cmap[str(level)], lw=2, alpha=0.9,
                    marker="o" if mode == "categorical" else None, label=f"{group}={level}")
            if bands and str(level) in bands:
                draw_band(bands[str(level)], cmap[str(level)], None)
        if bands and None not in bands and any(str(lv) in bands for lv in panel_levels):
            # One legend entry for the whole family of per-group intervals.
            ax.plot([], [], color="grey", lw=6, alpha=0.3,
                    label=f"{int(round(ci_level * 100))}% CI (emmeans)")

        if mode == "categorical":
            ax.set_xticks(np.arange(len(x_values)))
            ax.set_xticklabels([str(v) for v in x_values])

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

    if display == "overview":
        fig, ax = plt.subplots(figsize=(10, 6))
        draw_panel(ax, df, levels, title)
        panel_axes = [ax]
        fig.tight_layout()
    else:
        from nvitk.stats.region_groups import panel_grid, resolve_panels

        panels = resolve_panels(levels, column=group)
        fig, axes = panel_grid(len(panels), title=title)
        panel_axes = []
        for ax, (panel, panel_levels) in zip(axes, panels.items()):
            sub = df.loc[df[group].astype(str).isin({str(v) for v in panel_levels})]
            if sub.empty:
                ax.text(0.5, 0.5, "No observations", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(panel)
                ax.set_axis_off()
                continue
            draw_panel(ax, sub, panel_levels, panel)
            panel_axes.append(ax)

    for name, message in errors.items():
        setattr(fig, name, message)
    fig.linked_axes = panel_axes
    return fig


__all__ = [
    "ANALYSIS_LME4",
    "INSTALL_HINT",
    "OPTIONAL_R_PACKAGES",
    "REQUIRED_R_PACKAGES",
    "RBackendStatus",
    "fit_lme4",
    "lme4_coef_frame",
    "lme4_emmeans",
    "lme4_fixed_formula",
    "lme4_fixef_vcov",
    "lme4_grouping_factors",
    "lme4_info_dict",
    "lme4_predict",
    "lme4_random_effects_frame",
    "lme4_random_terms",
    "mixedlm_to_lme4_formula",
    "plot_lme4_params",
    "r_backend_status",
]
