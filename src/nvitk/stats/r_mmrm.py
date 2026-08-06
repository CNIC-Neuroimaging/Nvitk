"""
Mixed models for repeated measures (MMRM), through R's ``mmrm`` package.

Description
-----------
An MMRM is not a random-effects model. Where a MixedLM says *"each subject has its own intercept,
and everything else is independent noise"*, an MMRM says *"the repeated measurements within a
subject are correlated, and here is the shape of that correlation"* — fitting a structured
covariance matrix directly instead of decomposing it into random effects plus residual.

For a subject × territory frame that matters. A random intercept imposes **compound symmetry**:
every pair of territories is assumed equally correlated. In reality MCA and ACA (both anterior) are
more alike than MCA and Basilar. An unstructured MMRM estimates all of those correlations rather
than assuming them away — at the cost of ``k(k+1)/2`` covariance parameters for ``k`` territories,
which needs a reasonable number of subjects to support.

.. code-block:: text

    MixedLM  : pi ~ age_c + sex,  groups=territory, vc={subject}
    lme4     : pi ~ age_c + sex + (1 | subject_uid)
    MMRM     : pi ~ age_c + sex + us(territory | subject_uid)

The bracketed term is a *covariance structure*, not a random effect: ``us`` unstructured, ``cs``
compound symmetry (the MixedLM equivalent), ``ar1`` first-order autoregressive, ``toep`` Toeplitz,
``ad`` ante-dependence, each with a heterogeneous-variance variant.

Requirements
------------
Python: **nothing beyond rpy2**, which is already needed for the lme4 engine. ``pymer4`` does not
wrap ``mmrm``, so this talks to R directly.

R: ``mmrm`` (which pulls in ``TMB``), plus ``emmeans`` for least-squares means.

Design note
-----------
Extraction happens *in R*, returning tidy data frames that rpy2 converts to pandas. Traversing S4/list
internals from Python would be far more brittle, and R's own accessors are the documented interface.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import shutil
import subprocess
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

from .mixedlm import significance_stars

log = Logger()

ANALYSIS_MMRM = "mmrm"

REQUIRED_R_PACKAGES: tuple[str, ...] = ("mmrm",)
OPTIONAL_R_PACKAGES: tuple[str, ...] = ("emmeans",)

INSTALL_HINT = (
    "No extra Python packages are needed — this uses rpy2 directly.\n"
    "conda install -c conda-forge r-mmrm r-emmeans\n"
    "  or:  R -e \"install.packages(c('mmrm','emmeans'), repos='https://cloud.r-project.org')\""
)


# ──────────────────────────────────────────────────────────────────────────────
# Covariance structures and degrees-of-freedom methods
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CovarianceStructure:
    """One ``mmrm`` covariance structure, and when it is the right choice."""

    key: str
    label: str
    n_parameters: str
    description: str


COVARIANCE_STRUCTURES: dict[str, CovarianceStructure] = {
    "us": CovarianceStructure(
        "us", "Unstructured", "k(k+1)/2",
        "Every variance and every pairwise correlation estimated freely. The default in clinical "
        "trials and the safest choice when the repeated dimension is not ordered — but it needs "
        "enough subjects to support all those parameters.",
    ),
    "cs": CovarianceStructure(
        "cs", "Compound symmetry", "2",
        "One variance and one correlation shared by every pair. Equivalent to a random intercept "
        "per subject, so this is the direct analogue of the MixedLM.",
    ),
    "csh": CovarianceStructure(
        "csh", "Compound symmetry, heterogeneous", "k+1",
        "One shared correlation, but each level keeps its own variance. Useful when territories "
        "differ in scale (flow in the MCA vs a communicating artery) but not in how they covary.",
    ),
    "ar1": CovarianceStructure(
        "ar1", "Autoregressive AR(1)", "2",
        "Correlation decays with distance between levels. Only meaningful when the repeated "
        "dimension is *ordered* — visits over time, not unordered territories.",
    ),
    "ar1h": CovarianceStructure(
        "ar1h", "AR(1), heterogeneous", "k+1",
        "Decaying correlation with a separate variance per level.",
    ),
    "toep": CovarianceStructure(
        "toep", "Toeplitz", "k",
        "Correlation depends only on the gap between levels, but each gap gets its own value. "
        "More flexible than AR(1), still assumes an ordering.",
    ),
    "toeph": CovarianceStructure(
        "toeph", "Toeplitz, heterogeneous", "2k-1",
        "Toeplitz correlations with a separate variance per level.",
    ),
    "ad": CovarianceStructure(
        "ad", "Ante-dependence", "k",
        "Correlation between neighbours estimated separately, more distant pairs implied by the "
        "chain. For ordered measurements whose correlation decays irregularly.",
    ),
    "adh": CovarianceStructure(
        "adh", "Ante-dependence, heterogeneous", "2k-1",
        "Ante-dependence with a separate variance per level.",
    ),
}

# Structures that assume the repeated dimension has a meaningful order.
ORDERED_STRUCTURES: frozenset[str] = frozenset({"ar1", "ar1h", "toep", "toeph", "ad", "adh"})


@dataclass(frozen=True)
class DfMethod:
    """A denominator degrees-of-freedom method."""

    key: str
    label: str
    description: str


DF_METHODS: dict[str, DfMethod] = {
    "Satterthwaite": DfMethod(
        "Satterthwaite", "Satterthwaite",
        "The usual choice. Adjusts the denominator degrees of freedom for the estimated covariance.",
    ),
    "Kenward-Roger": DfMethod(
        "Kenward-Roger", "Kenward-Roger",
        "Also inflates the standard errors to account for having estimated the covariance. Preferred "
        "for small samples; slower, and the standard errors will not match Satterthwaite's.",
    ),
    "Residual": DfMethod(
        "Residual", "Residual",
        "n − p. Simple and anti-conservative — only reasonable with many subjects.",
    ),
    "Between-Within": DfMethod(
        "Between-Within", "Between-within",
        "Splits the degrees of freedom by whether a term varies within or between subjects.",
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Availability
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MmrmBackendStatus:
    """What is present, what is missing, and what to do about it."""

    available: bool
    rpy2_version: str = ""
    r_version: str = ""
    r_packages: Mapping[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    reason: str = ""

    def install_hint(self) -> str:
        """Copy-pasteable commands for whatever is missing."""
        return INSTALL_HINT

    def summary(self) -> str:
        """One-line description for a status bar or tooltip."""
        if self.available:
            packages = ", ".join(f"{k} {v}" for k, v in sorted(self.r_packages.items()))
            return f"rpy2 {self.rpy2_version} · R {self.r_version} · {packages}"
        return self.reason


def _r_package_versions(names: Sequence[str]) -> dict[str, str]:
    """Ask ``Rscript`` which of *names* are installed; missing ones map to ``""``."""
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
    found = {name: "" for name in names}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            if key.strip() in found:
                found[key.strip()] = value.strip()
    return found


def mmrm_backend_status(*, check_r_packages: bool = True) -> MmrmBackendStatus:
    """
    Probe rpy2 → R → ``mmrm``. Never raises, and never imports R as an import side effect.

    Deliberately independent of the lme4 probe: MMRM needs neither pymer4 nor the easystats suite,
    so it can be available when the lme4 engine is not, and reporting them together would send you
    chasing packages you do not need.
    """
    try:
        import rpy2  # noqa: F401
        import rpy2.situation as situation

        rpy2_version = str(getattr(rpy2, "__version__", "?"))
        r_version = str(situation.r_version_from_subprocess() or "")
    except Exception as exc:
        return MmrmBackendStatus(
            available=False,
            missing=("rpy2",),
            reason=f"rpy2 is not usable ({type(exc).__name__}: {exc}). It links Python to R.",
        )

    if not shutil.which("R"):
        return MmrmBackendStatus(
            available=False, rpy2_version=rpy2_version, missing=("R",),
            reason="No R interpreter on PATH.",
        )

    packages: dict[str, str] = {}
    if check_r_packages:
        packages = _r_package_versions([*REQUIRED_R_PACKAGES, *OPTIONAL_R_PACKAGES])
        missing = [name for name in REQUIRED_R_PACKAGES if not packages.get(name)]
        if missing:
            return MmrmBackendStatus(
                available=False, rpy2_version=rpy2_version, r_version=r_version,
                r_packages={k: v for k, v in packages.items() if v},
                missing=tuple(missing),
                reason=f"R package(s) not installed: {', '.join(missing)}.",
            )

    return MmrmBackendStatus(
        available=True, rpy2_version=rpy2_version, r_version=r_version,
        r_packages={k: v for k, v in packages.items() if v},
    )


def emmeans_available() -> bool:
    """Whether ``emmeans`` is installed — least-squares means need it."""
    return bool(_r_package_versions(["emmeans"]).get("emmeans"))


# ──────────────────────────────────────────────────────────────────────────────
# Formula
# ──────────────────────────────────────────────────────────────────────────────
# A covariance term written inside the formula: ``us(territory | mri_id)``, ``ar1(v | g / s)``.
_COVARIANCE_TERM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(" + "|".join(sorted(COVARIANCE_STRUCTURES, key=len, reverse=True)) + r")"
    r"\s*\(\s*([^()|]+?)\s*\|\s*([^()]+?)\s*\)"
)


@dataclass(frozen=True)
class CovarianceTerm:
    """A covariance term parsed out of a formula."""

    structure: str
    visit: str
    subject: str
    group: str = ""
    text: str = ""

    def columns(self) -> list[str]:
        """Frame columns this term references."""
        return [c for c in (self.visit, self.group, self.subject) if c]


def parse_mmrm_covariance(formula: str) -> tuple[str, CovarianceTerm | None]:
    """
    Split an ``mmrm`` formula into its fixed part and its covariance term.

    Lets the term be written inline — ``… + us(territory | mri_id)`` — the way it is in R, instead
    of being assembled from separate controls. Returns the fixed formula with the term removed, plus
    the parsed term (``None`` when the formula has none).

    >>> parse_mmrm_covariance("pi ~ age_c + us(territory | mri_id)")[1].subject
    'mri_id'
    """
    text = str(formula or "")
    match = _COVARIANCE_TERM_RE.search(text)
    if match is None:
        return text.strip(), None

    structure, visit, right = match.group(1), match.group(2).strip(), match.group(3).strip()
    group = ""
    if "/" in right:
        group, _, subject = right.partition("/")
        group, subject = group.strip(), subject.strip()
    else:
        subject = right

    # Remove the term and tidy the '+' it leaves behind.
    fixed = text[: match.start()] + text[match.end() :]
    fixed = re.sub(r"\+\s*\+", "+", fixed)
    fixed = re.sub(r"\+\s*$", "", fixed).strip()
    fixed = re.sub(r"~\s*\+", "~ ", fixed).strip()
    fixed = re.sub(r"\s+", " ", fixed)
    if fixed.endswith("~"):
        fixed += " 1"

    return fixed, CovarianceTerm(
        structure=structure, visit=visit, subject=subject, group=group, text=match.group(0)
    )


def covariance_term(
    structure: str, visit: str, subject: str, *, group: str = ""
) -> str:
    """
    Render a covariance term on its own, e.g. ``us(territory | subject_uid)``.

    Split out from :func:`mmrm_formula` so a UI can offer the term for insertion into a formula the
    user is writing, rather than only assembling a whole formula.

    Parameters
    ----------
    group : str
        Optional grouping column for a *separate* covariance matrix per group, written
        ``us(visit | group / subject)``.
    """
    if structure not in COVARIANCE_STRUCTURES:
        raise ValueError(
            f"Unknown covariance structure {structure!r}. "
            f"Available: {', '.join(sorted(COVARIANCE_STRUCTURES))}."
        )
    if not visit or not subject:
        raise ValueError("Both a repeated-measures column and a subject column are required.")
    inner = f"{visit} | {group} / {subject}" if group else f"{visit} | {subject}"
    return f"{structure}({inner})"


def mmrm_formula(
    fixed: str,
    *,
    visit: str,
    subject: str,
    structure: str = "us",
    group: str = "",
) -> str:
    """
    Append an ``mmrm`` covariance term to a fixed-effects formula.

    >>> mmrm_formula("pi ~ age_c + sex", visit="territory", subject="subject_uid")
    'pi ~ age_c + sex + us(territory | subject_uid)'

    Parameters
    ----------
    group : str
        Optional grouping column for a *separate* covariance matrix per group, written
        ``us(visit | group / subject)`` — e.g. one covariance per treatment arm.
    """
    fixed = str(fixed or "").strip()
    if not fixed or "~" not in fixed:
        raise ValueError("A fixed-effects formula with a '~' is required.")
    return f"{fixed} + {covariance_term(structure, visit, subject, group=group)}"


def validate_mmrm_data(
    data: pd.DataFrame, *, visit: str, subject: str, structure: str = "us"
) -> list[str]:
    """
    Problems that would make an ``mmrm`` fit fail or mislead, as human-readable messages.

    ``mmrm`` requires at most one observation per ``(subject, visit)`` cell — the covariance matrix
    is indexed by visit, so a duplicate has no place to go and R raises a message that does not
    point at the data. Checking here means the error names the offending subjects.
    """
    problems: list[str] = []
    for column in (visit, subject):
        if column not in data.columns:
            problems.append(f"Column {column!r} is not in the frame.")
    if problems:
        return problems

    duplicated = data.duplicated(subset=[subject, visit], keep=False)
    if duplicated.any():
        examples = (
            data.loc[duplicated, [subject, visit]]
            .astype(str)
            .agg(" / ".join, axis=1)
            .drop_duplicates()
            .head(4)
            .tolist()
        )
        problems.append(
            f"{int(duplicated.sum())} rows share a ({subject}, {visit}) cell, which mmrm cannot "
            f"represent — e.g. {', '.join(examples)}. Aggregate to one row per cell first."
        )

    n_levels = int(data[visit].nunique())
    n_subjects = int(data[subject].nunique())
    if n_levels < 2:
        problems.append(f"{visit!r} has only {n_levels} level — there is nothing repeated to model.")
    if structure == "us":
        n_parameters = n_levels * (n_levels + 1) // 2
        if n_subjects < n_parameters:
            problems.append(
                f"An unstructured covariance over {n_levels} levels needs {n_parameters} parameters "
                f"but there are only {n_subjects} subjects; the fit will be unstable. Consider "
                "compound symmetry or a Toeplitz structure."
            )
    if structure in ORDERED_STRUCTURES:
        problems.append(
            f"{COVARIANCE_STRUCTURES[structure].label} assumes {visit!r} is ordered — check that its "
            "level order is meaningful, since the correlation is modelled as a function of distance."
        )
    return problems


# ──────────────────────────────────────────────────────────────────────────────
# R bridge
# ──────────────────────────────────────────────────────────────────────────────
# Extraction lives in R and returns tidy data frames: rpy2 converts those cleanly, whereas walking
# the fit object's internals from Python would break on every upstream refactor.
_R_HELPERS = """
.nvitk_mmrm_fit <- function(data, formula, reml, method, factor_cols) {
  # mmrm requires the repeated and subject variables to be *factors*, not character vectors --
  # rpy2 hands over plain characters, and mmrm rejects those with
  # "Time point variable 'x' must be a factor". Coerce here so the requirement is met whatever
  # the Python-side conversion produced.
  for (col in factor_cols) {
    if (col %in% names(data) && !is.factor(data[[col]])) {
      data[[col]] <- factor(data[[col]])
    }
  }
  mmrm::mmrm(formula = stats::as.formula(formula), data = data,
             reml = reml, method = method)
}

.nvitk_mmrm_coefs <- function(fit) {
  cf <- as.data.frame(summary(fit)$coefficients, check.names = FALSE)
  cf$parameter <- rownames(cf)
  rownames(cf) <- NULL
  cf
}

.nvitk_mmrm_covariance <- function(fit) {
  v <- as.matrix(mmrm::VarCorr(fit))
  out <- as.data.frame(v, check.names = FALSE)
  out$level <- rownames(v)
  rownames(out) <- NULL
  out
}

.nvitk_mmrm_stats <- function(fit) {
  grab <- function(expr) tryCatch(as.numeric(expr), error = function(e) NA_real_)
  conv <- tryCatch(mmrm::component(fit, "convergence"), error = function(e) NA)
  data.frame(
    AIC = grab(stats::AIC(fit)),
    BIC = grab(stats::BIC(fit)),
    logLik = grab(stats::logLik(fit)),
    n_obs = grab(mmrm::component(fit, "n_obs")),
    n_subjects = grab(mmrm::component(fit, "n_subjects")),
    converged = isTRUE(is.na(conv)) || isTRUE(all(conv == 0)),
    check.names = FALSE
  )
}

.nvitk_mmrm_emmeans <- function(fit, specs) {
  em <- emmeans::emmeans(fit, specs = stats::as.formula(specs))
  as.data.frame(em)
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
    func = globalenv[name]
    with _converter():
        return pd.DataFrame(func(*args))


# ──────────────────────────────────────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────────────────────────────────────
def fit_mmrm(
    *,
    data: pd.DataFrame,
    formula: str,
    visit: str = "",
    subject: str = "",
    structure: str = "us",
    group: str = "",
    method: str = "Satterthwaite",
    reml: bool = True,
    strict: bool = True,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """
    Fit a mixed model for repeated measures.

    Parameters
    ----------
    formula : str
        Either the full R formula with the covariance term written inline —
        ``"pi ~ age_c + us(territory | mri_id)"`` — or the fixed part alone, in which case the term
        is assembled from *visit*, *subject*, *structure* and *group*. An inline term always wins:
        if you wrote one, that is the model, and the separate arguments are ignored.
    method : str
        Denominator degrees of freedom; see :data:`DF_METHODS`.
    reml : bool
        Restricted maximum likelihood. Leave on unless comparing fixed-effects specifications by
        likelihood.
    strict : bool
        Refuse to fit when :func:`validate_mmrm_data` finds a hard problem. Set ``False`` to
        downgrade those to warnings.

    Returns
    -------
    (fit, model_df, metadata)
        *fit* is the R ``mmrm`` object; *metadata* matches the other engines.
    """
    status = mmrm_backend_status()
    if not status.available:
        raise RuntimeError(
            f"The MMRM engine is not available: {status.reason}\n\n{status.install_hint()}"
        )

    fixed_formula, term = parse_mmrm_covariance(formula)
    if term is not None:
        structure, visit, subject, group = term.structure, term.visit, term.subject, term.group
        full_formula = str(formula).strip()
    else:
        if not visit or not subject:
            raise ValueError(
                "No covariance term found in the formula, and no repeated / subject column was "
                "given. Either write the term inline, e.g. '+ us(territory | subject_uid)', or "
                "choose the columns."
            )
        full_formula = mmrm_formula(
            fixed_formula, visit=visit, subject=subject, structure=structure, group=group
        )

    needed = [c for c in {visit, subject, *_formula_columns(data.columns, fixed_formula)} if c in data.columns]
    n_input = int(len(data))
    dropped_by_column = {c: int(data[c].isna().sum()) for c in needed if bool(data[c].isna().any())}
    df = data.dropna(subset=needed).reset_index(drop=True) if needed else data.copy()
    if not len(df):
        raise ValueError(f"No complete rows left after dropping missing values in {needed}.")

    # mmrm needs the repeated and subject columns as factors. pandas Categorical is what rpy2 maps
    # to an R factor — a plain string column arrives as a character vector, which mmrm rejects with
    # "Time point variable 'x' must be a factor". The R helper coerces again as a backstop, so this
    # holds whatever rpy2 version is installed.
    factor_columns = [c for c in (visit, subject, group) if c and c in df.columns]
    for column in factor_columns:
        df[column] = pd.Categorical(df[column].astype(str))

    problems = validate_mmrm_data(df, visit=visit, subject=subject, structure=structure)
    hard = [p for p in problems if "assumes" not in p and "will be unstable" not in p]
    for problem in problems:
        log.warning("MMRM: %s", problem)
    if strict and hard:
        raise ValueError("MMRM cannot fit this frame:\n  - " + "\n  - ".join(hard))

    # R rejects repeated column names just as polars does, so make them unique before crossing over.
    from .frame_ops import ensure_unique_columns

    df = ensure_unique_columns(df, context="analysis dataframe")
    with _converter():
        from rpy2.robjects import conversion

        r_data = conversion.get_conversion().py2rpy(df)
    _ensure_helpers()
    from rpy2.robjects import globalenv

    from rpy2.robjects import StrVector

    fit = globalenv[".nvitk_mmrm_fit"](
        r_data, full_formula, reml, method, StrVector(factor_columns)
    )

    meta = {
        "engine": ANALYSIS_MMRM,
        "formula": full_formula,
        "fixed_formula": fixed_formula,
        "inline_term": term.text if term is not None else "",
        "visit": visit,
        "subject": subject,
        "structure": structure,
        "structure_label": COVARIANCE_STRUCTURES[structure].label,
        "group": group,
        "method": method,
        "reml": bool(reml),
        "n_rows_input": n_input,
        "n_rows": int(len(df)),
        "n_rows_dropped": n_input - int(len(df)),
        "dropna_columns": needed,
        "dropped_by_column": dropped_by_column,
        "loaded": False,
        "backend": status.summary(),
    }
    if problems:
        meta["warnings"] = problems
    log.info("MMRM fit: %s [%s, %s df] (n=%d)", full_formula, structure, method, len(df))
    return fit, df, meta


def _formula_columns(columns: Sequence[str], formula: str) -> list[str]:
    """Frame columns a formula mentions."""
    import re

    available = {str(c) for c in columns}
    return sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(formula or ""))) & available)


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
_COEF_ALIASES: dict[str, tuple[str, ...]] = {
    "coef": ("Estimate",),
    "std_err": ("Std. Error", "Std.Error", "Std..Error"),
    "df": ("df",),
    "z": ("t value", "t.value", "t..value", "statistic"),
    "p_value": ("Pr(>|t|)", "Pr...t..", "p.value", "p"),
}


def _pick(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series | None:
    """First present column among *names*."""
    for name in names:
        if name in frame.columns:
            return frame[name]
    return None


def mmrm_coef_frame(fit: Any) -> pd.DataFrame:
    """Fixed-effects table, normalized to this package's column names."""
    from scipy.stats import norm  # noqa: F401  (kept for parity with the other engines)

    coefs = _call_r(".nvitk_mmrm_coefs", fit)
    if coefs.empty:
        raise ValueError("The MMRM fit produced no coefficients.")

    names = _pick(coefs, ("parameter",))
    out = pd.DataFrame({"parameter": [str(v) for v in (names if names is not None else coefs.index)]})
    for target in ("coef", "std_err", "df", "z", "p_value"):
        series = _pick(coefs, _COEF_ALIASES[target])
        out[target] = np.nan if series is None else pd.to_numeric(series, errors="coerce").to_numpy()

    # mmrm reports t-statistics with estimated degrees of freedom, so the interval is a t interval
    # rather than the normal one the other engines use.
    from scipy.stats import t as student_t

    crit = np.where(
        np.isfinite(out["df"]), student_t.ppf(0.975, np.clip(out["df"], 1e-6, None)), 1.959963985
    )
    out["ci_low"] = out["coef"] - crit * out["std_err"]
    out["ci_high"] = out["coef"] + crit * out["std_err"]
    out["sig"] = [significance_stars(float(p)) for p in out["p_value"]]
    return out[["parameter", "coef", "std_err", "z", "p_value", "ci_low", "ci_high", "sig", "df"]]


def mmrm_covariance_matrix(fit: Any) -> pd.DataFrame:
    """The estimated covariance matrix of the repeated measures, indexed by level."""
    frame = _call_r(".nvitk_mmrm_covariance", fit)
    if "level" in frame.columns:
        frame = frame.set_index("level")
        frame.index.name = None
    return frame


def mmrm_correlation_matrix(fit: Any) -> pd.DataFrame:
    """
    The covariance matrix rescaled to correlations.

    This is the thing an unstructured MMRM is *for*: it says how alike two territories actually are,
    which a random intercept assumes rather than estimates.
    """
    covariance = mmrm_covariance_matrix(fit)
    sd = np.sqrt(np.diag(covariance.to_numpy(dtype=float)))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance.to_numpy(dtype=float) / np.outer(sd, sd)
    return pd.DataFrame(correlation, index=covariance.index, columns=covariance.columns)


def mmrm_random_effects_frame(fit: Any) -> pd.DataFrame:
    """
    The covariance structure rendered as component/kind/var/sd rows.

    Lets an MMRM reuse the shared report panel's "Random effects" tab: the diagonal becomes one
    variance per level, and the off-diagonal correlations follow.
    """
    covariance = mmrm_covariance_matrix(fit)
    values = covariance.to_numpy(dtype=float)
    levels = [str(i) for i in covariance.index]

    rows: list[dict[str, Any]] = []
    for i, level in enumerate(levels):
        variance = float(values[i, i])
        rows.append(
            {
                "component": level,
                "kind": "variance",
                "var": variance,
                "sd": float(np.sqrt(max(variance, 0.0))),
            }
        )
    sd = np.sqrt(np.diag(values))
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            denominator = sd[i] * sd[j]
            rows.append(
                {
                    "component": f"cor({levels[i]}, {levels[j]})",
                    "kind": "correlation",
                    "var": np.nan,
                    "sd": float(values[i, j] / denominator) if denominator else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=["component", "kind", "var", "sd"])


def mmrm_info_dict(
    fit: Any,
    *,
    outcome_name: str = "Outcome",
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report structure matching the other engines, so the shared panel renders it unchanged."""
    meta = dict(meta or {})
    stats = _call_r(".nvitk_mmrm_stats", fit)

    def stat(name: str) -> float | None:
        if name in stats.columns and len(stats):
            value = pd.to_numeric(stats[name], errors="coerce").iloc[0]
            return float(value) if pd.notna(value) else None
        return None

    fit_stats: dict[str, Any] = {}
    for key, column in (("aic", "AIC"), ("bic", "BIC"), ("llf", "logLik")):
        value = stat(column)
        if value is not None:
            fit_stats[key] = value
    converged = stats["converged"].iloc[0] if "converged" in stats.columns and len(stats) else True
    fit_stats["converged"] = bool(converged)

    n_subjects = stat("n_subjects")
    n_obs = stat("n_obs") or meta.get("n_rows") or 0

    return {
        "header": {
            "formula": meta.get("formula"),
            "n_obs": int(n_obs),
            "n_groups": int(n_subjects) if n_subjects else None,
            "group_name": meta.get("subject", "Subject"),
            "outcome_name": outcome_name,
            "vc_group_name": f"Covariance ({meta.get('structure_label', meta.get('structure', ''))})",
            "engine": "R · mmrm",
            "structure": meta.get("structure_label"),
            "df_method": meta.get("method"),
            "backend": meta.get("backend", ""),
        },
        "fixed_effects": mmrm_coef_frame(fit),
        "random_effects": mmrm_random_effects_frame(fit),
        "cov_re": mmrm_covariance_matrix(fit),
        "fit_statistics": fit_stats,
        "has_vcomp": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Least-squares means and plotting
# ──────────────────────────────────────────────────────────────────────────────
def mmrm_emmeans(fit: Any, specs: str) -> pd.DataFrame:
    """
    Least-squares means from ``emmeans``, as a tidy frame.

    Parameters
    ----------
    specs : str
        An emmeans specification formula, e.g. ``"~ territory"`` or
        ``"~ tacsctot_group | territory"``.

    Returns
    -------
    pandas.DataFrame
        The grid columns plus ``emmean``, ``SE``, ``df``, ``lower.CL``, ``upper.CL``.
    """
    if not emmeans_available():
        raise RuntimeError(
            "Least-squares means need the R package 'emmeans'.\n"
            "conda install -c conda-forge r-emmeans"
        )
    return _call_r(".nvitk_mmrm_emmeans", fit, specs)


def plot_mmrm_emmeans(
    emmeans_frame: pd.DataFrame,
    *,
    x: str,
    hue: str | None = None,
    errorbar: bool = True,
    palette: str = "tab10",
    title: str = "MMRM least-squares means",
    y_label: str = "Estimated marginal mean",
) -> Any:
    """
    Least-squares means with their confidence intervals, drawn like the other categorical plots.

    Whiskers rather than a ribbon: the x axis is a set of levels, and there is nothing between them
    to interpolate.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from .mixedlm import _natural_sort_key

    frame = emmeans_frame.copy()
    estimate = next((c for c in ("emmean", "estimate", "response") if c in frame.columns), None)
    if estimate is None or x not in frame.columns:
        raise ValueError(f"Cannot plot: need {x!r} and an estimate column in {list(frame.columns)}.")
    lower = next((c for c in ("lower.CL", "asymp.LCL", "lower") if c in frame.columns), None)
    upper = next((c for c in ("upper.CL", "asymp.UCL", "upper") if c in frame.columns), None)

    order = sorted(frame[x].dropna().astype(str).unique(), key=_natural_sort_key)
    positions = {level: i for i, level in enumerate(order)}

    fig, ax = plt.subplots(figsize=(10, 6))
    levels = (
        sorted(frame[hue].dropna().astype(str).unique(), key=_natural_sort_key)
        if hue and hue in frame.columns
        else [None]
    )
    colors = sns.color_palette(palette, n_colors=max(len(levels), 3))

    for i, level in enumerate(levels):
        subset = frame if level is None else frame.loc[frame[hue].astype(str) == level]
        subset = subset.assign(_xi=subset[x].astype(str).map(positions)).sort_values("_xi")
        color = "black" if level is None else colors[i % len(colors)]
        ax.plot(
            subset["_xi"], subset[estimate], marker="o", lw=2.4, color=color,
            label=None if level is None else f"{hue}={level}",
        )
        if errorbar and lower and upper:
            ax.errorbar(
                subset["_xi"], subset[estimate],
                yerr=[subset[estimate] - subset[lower], subset[upper] - subset[estimate]],
                fmt="none", ecolor=color, elinewidth=1.3, capsize=4, capthick=1.3, alpha=0.85,
            )

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.25)
    if any(level is not None for level in levels):
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_mmrm_correlation(
    fit: Any,
    *,
    levels: Sequence[str] | None = None,
    title: str = "Estimated correlation between levels",
) -> Any:
    """
    Heatmap of the estimated correlation matrix.

    Worth looking at whenever the structure is unstructured: it shows directly whether the compound
    symmetry a random-intercept model would have assumed is anywhere near the truth.

    Parameters
    ----------
    levels : sequence of str, optional
        Show only these levels, in this order. The correlations are still *estimated* over every
        level — this only restricts the display.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    correlation = mmrm_correlation_matrix(fit)
    if levels:
        keep = [str(v) for v in levels if str(v) in correlation.index]
        if not keep:
            raise ValueError("None of the selected levels are in the fitted covariance matrix.")
        correlation = correlation.loc[keep, keep]

    n = len(correlation)
    # Level names can be long ("Internal Carotid Arteries"), so size from the label width as well as
    # the cell count, and cap it: an oversized figure scaled into the canvas is what pushes the tick
    # labels off the edge.
    label_width = max((len(str(i)) for i in correlation.index), default=8)
    width = float(np.clip(n * 0.8 + label_width * 0.11, 6.0, 16.0))
    height = float(np.clip(n * 0.7 + 1.5, 5.0, 13.0))

    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        correlation,
        annot=n <= 12,
        fmt=".2f",
        annot_kws={"fontsize": 8} if n > 8 else None,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"label": "correlation", "shrink": 0.8},
    )
    ax.set_title(title)
    # Angled column labels and horizontal row labels: with more than a handful of levels the default
    # horizontal ticks overlap each other and run past the axes.
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    # Constrained layout re-runs on every draw, so the labels stay inside the figure when the Qt
    # canvas is resized — tight_layout only solves it for the size at creation time.
    fig.set_layout_engine("constrained")
    return fig


__all__ = [
    "ANALYSIS_MMRM",
    "COVARIANCE_STRUCTURES",
    "DF_METHODS",
    "INSTALL_HINT",
    "OPTIONAL_R_PACKAGES",
    "ORDERED_STRUCTURES",
    "REQUIRED_R_PACKAGES",
    "CovarianceStructure",
    "CovarianceTerm",
    "DfMethod",
    "MmrmBackendStatus",
    "emmeans_available",
    "fit_mmrm",
    "mmrm_backend_status",
    "mmrm_coef_frame",
    "mmrm_correlation_matrix",
    "mmrm_covariance_matrix",
    "mmrm_emmeans",
    "mmrm_formula",
    "parse_mmrm_covariance",
    "mmrm_info_dict",
    "mmrm_random_effects_frame",
    "plot_mmrm_correlation",
    "plot_mmrm_emmeans",
    "validate_mmrm_data",
]
