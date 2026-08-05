"""
Mediation analysis — three engines over one common result shape.

Description
-----------
Given an exposure ``x``, a mediator ``m`` and an outcome ``y``, mediation decomposes the effect of
``x`` on ``y`` into an *indirect* path through ``m`` (``a·b``) and a *direct* path (``c'``). Three
engines are provided, trading statistical fidelity against runtime:

``mixedlm_bootstrap``
    Fits the mediator and outcome models as MixedLMs (``y`` nested in ``group_col``, subjects as a
    variance component) and resamples **whole subjects** with replacement. Respects the
    subject × territory nesting of the analysis frames, at the cost of ``2 × n_boot`` model fits.
``pingouin_by_level``
    ``pingouin.mediation_analysis`` run separately within each level of a grouping column. Fast and
    gives a per-territory picture, but each fit is an OLS that ignores the nesting.
``statsmodels_parametric``
    ``statsmodels.stats.mediation.Mediation`` over the pooled frame with OLS mediator/outcome models.
    Standard and quick, but pooling subject × territory rows treats them as independent, so its
    intervals are anti-conservative for these data.

All three funnel into :func:`mediation_result_frame`, a tidy table with columns
``path, coef, ci_low, ci_high, pval, n, engine, level`` (plus the ``CI2.5`` / ``CI97.5`` aliases the
notebook plots use), so the plotting helpers and the GUI treat them interchangeably.

Conventions
-----------
Paths are labeled ``a`` (x → m), ``b`` (m → y | x), ``c'`` (x → y | m, the direct effect),
``Indirect`` (a·b), ``Direct`` (= c'), and ``Total`` (indirect + direct). Bootstrap p-values are
two-sided percentile p-values: ``2 · min(P(θ ≤ 0), P(θ ≥ 0))``.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from nvitk.core.logger import Logger

from .mixedlm import fit_or_load_mixedlm

log = Logger()

MEDIATION_ENGINES: tuple[str, ...] = (
    "mixedlm_bootstrap",
    "pingouin_by_level",
    "statsmodels_parametric",
)

ENGINE_LABELS: dict[str, str] = {
    "mixedlm_bootstrap": "MixedLM subject-cluster bootstrap",
    "pingouin_by_level": "pingouin, per grouping level",
    "statsmodels_parametric": "statsmodels Mediation (pooled OLS)",
}

# Caveat shown next to results from engines that ignore the subject/territory nesting.
POOLED_OLS_NOTE = (
    "This engine fits pooled OLS models: repeated (subject × territory) rows are treated as "
    "independent observations, so the standard errors and intervals are anti-conservative. Use the "
    "MixedLM bootstrap for inference that respects the nesting."
)

# Canonical column order of every tidy mediation table.
RESULT_COLUMNS = ("path", "coef", "ci_low", "ci_high", "pval", "n", "engine", "level")

_PATH_ORDER = ("a", "b", "c'", "Indirect", "Direct", "Total")


# ──────────────────────────────────────────────────────────────────────────────
# Specification
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MediationSpec:
    """
    One mediation analysis: which columns play which role, and how to estimate it.

    Parameters
    ----------
    x, m, y : str
        Exposure, mediator, outcome column names. These must be bare frame columns — a transformed
        term belongs in a derived column, not here, because the engines look coefficients up by name.
    covariates : tuple of str
        Adjusted for in both the mediator and the outcome model.
    group_col : str
        Nesting level: the MixedLM ``groups`` for ``mixedlm_bootstrap``, the split column for
        ``pingouin_by_level``. Ignored by ``statsmodels_parametric``.
    subject_col : str
        Cluster resampled by ``mixedlm_bootstrap`` and used as its variance component.
    n_boot : int
        Bootstrap draws (``n_rep`` for ``statsmodels_parametric``).
    """

    x: str
    m: str
    y: str
    covariates: tuple[str, ...] = ()
    group_col: str = "territory"
    subject_col: str = "subject_uid"
    engine: str = "mixedlm_bootstrap"
    n_boot: int = 500
    seed: int = 42
    ci: float = 0.95

    def formulas(self) -> tuple[str, str]:
        """``(mediator_formula, outcome_formula)`` — ``m ~ x + covars`` and ``y ~ m + x + covars``."""
        covars = list(self.covariates)
        formula_m = " + ".join([self.x, *covars])
        formula_y = " + ".join([self.m, self.x, *covars])
        return f"{self.m} ~ {formula_m}", f"{self.y} ~ {formula_y}"

    def required_columns(self) -> list[str]:
        """Every column the analysis touches, de-duplicated in a stable order."""
        names = [self.x, self.m, self.y, *self.covariates]
        if self.engine == "mixedlm_bootstrap":
            names += [self.group_col, self.subject_col]
        elif self.engine == "pingouin_by_level":
            names.append(self.group_col)
        return list(dict.fromkeys(n for n in names if n))

    def validate(self, df: pd.DataFrame | None = None) -> str:
        """Return why this spec cannot be run, or ``""`` if it is usable."""
        if not (self.x and self.m and self.y):
            return "Exposure (X), mediator (M) and outcome (Y) must all be set."
        if len({self.x, self.m, self.y}) < 3:
            return "Exposure, mediator and outcome must be three different columns."
        if self.engine not in MEDIATION_ENGINES:
            return f"Unknown engine {self.engine!r}."
        if df is not None:
            missing = [c for c in self.required_columns() if c not in df.columns]
            if missing:
                return f"Columns not in the analysis frame: {', '.join(missing)}."
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "x": self.x,
            "m": self.m,
            "y": self.y,
            "covariates": list(self.covariates),
            "group_col": self.group_col,
            "subject_col": self.subject_col,
            "engine": self.engine,
            "n_boot": int(self.n_boot),
            "seed": int(self.seed),
            "ci": float(self.ci),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MediationSpec":
        """Rebuild from :meth:`to_dict` output, tolerating missing keys."""
        return cls(
            x=str(data.get("x") or ""),
            m=str(data.get("m") or ""),
            y=str(data.get("y") or ""),
            covariates=tuple(str(c) for c in (data.get("covariates") or ())),
            group_col=str(data.get("group_col") or "territory"),
            subject_col=str(data.get("subject_col") or "subject_uid"),
            engine=str(data.get("engine") or "mixedlm_bootstrap"),
            n_boot=int(data.get("n_boot", 500)),
            seed=int(data.get("seed", 42)),
            ci=float(data.get("ci", 0.95)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Engine (a) — MixedLM subject-cluster bootstrap
# ──────────────────────────────────────────────────────────────────────────────
def _bootstrap_summary(values: np.ndarray, *, q_lo: float, q_hi: float) -> dict[str, Any]:
    """Percentile CI, mean and two-sided bootstrap p-value of a draw distribution."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "ci_low": float(np.percentile(finite, q_lo)),
        "ci_high": float(np.percentile(finite, q_hi)),
        "p_value": 2 * min(float(np.mean(finite <= 0)), float(np.mean(finite >= 0))),
        "n": int(finite.size),
    }


def bootstrap_mediation_mixedlm_by_subject(
    df: pd.DataFrame,
    *,
    formula_m: str,
    formula_y: str,
    x: str,
    m: str,
    y: str,
    group_col: str = "territory",
    subject_col: str = "subject_uid",
    required_columns: Sequence[str] | None = None,
    n_boot: int = 500,
    seed: int = 42,
    re_formula: str = "1",
    vc_formula: dict[str, str] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    ci: float = 0.95,
    return_dist: bool = True,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """
    Cluster-bootstrap mediation for MixedLM models, resampling whole subjects.

    Strategy
    --------
    1. Fit both models once on the observed data for the point estimates
       (``a = coef of x in M``, ``b = coef of m in Y``, ``c' = coef of x in Y``).
    2. For each draw, resample *subjects* with replacement and keep **all** of each subject's rows
       (its territories), relabeling the ids so a repeated subject forms a distinct variance-component
       group. Refit both models and record ``a·b``, ``c'`` and their sum.
    3. Summarize each path by its percentile interval and a two-sided bootstrap p-value.

    Resampling whole subjects — rather than rows — is what keeps the within-subject correlation in
    the resampled data; row-wise resampling would understate the uncertainty.

    Parameters
    ----------
    formula_m, formula_y : str
        Mediator and outcome model formulas, e.g. ``"pi ~ pp + age_c + sex"`` and
        ``"att_mean ~ pi + pp + age_c + sex"``. See :meth:`MediationSpec.formulas`.
    vc_formula : dict, optional
        Variance components. Defaults to ``{"subject": f"0 + C({subject_col})"}``.
    progress : callable, optional
        Called as ``progress(done, total)`` after every draw.
    should_cancel : callable, optional
        Polled every draw; when it returns ``True`` the run stops early and the result carries
        ``cancelled=True`` with whatever draws completed.

    Returns
    -------
    dict
        ``point_estimate`` (a, b, c_prime, indirect, direct, total), ``bootstrap`` (per-path
        ``mean``/``ci_low``/``ci_high``/``p_value``/``n``), ``dist`` (draw arrays, when
        *return_dist*), ``n_boot``, ``n_failed_draws``, ``cancelled``.

    Raises
    ------
    ValueError
        If the observed-data fits do not expose *x* / *m* as named parameters — which happens when a
        term is wrapped (``C(x)``) or transformed in the formula. The message lists what is available.
    """
    fit_kwargs = dict(fit_kwargs or {})
    vc_formula = dict(vc_formula or {"subject": f"0 + C({subject_col})"})
    required_columns = list(required_columns or [])
    rng = np.random.default_rng(seed)

    def _fit_pair(data: pd.DataFrame) -> tuple[Any, Any]:
        """Fit the mediator and outcome MixedLMs on *data*."""
        res_m, _, _ = fit_or_load_mixedlm(
            model_path=None,
            data=data,
            formula=formula_m,
            groups=group_col,
            re_formula=re_formula,
            vc_formula=vc_formula,
            overwrite=True,
            required_columns=required_columns,
            fit_kwargs=fit_kwargs,
        )
        res_y, _, _ = fit_or_load_mixedlm(
            model_path=None,
            data=data,
            formula=formula_y,
            groups=group_col,
            re_formula=re_formula,
            vc_formula=vc_formula,
            overwrite=True,
            required_columns=required_columns,
            fit_kwargs=fit_kwargs,
        )
        return res_m, res_y

    # ---- 1. Point estimates on the observed data ------------------------------
    res_m0, res_y0 = _fit_pair(df)
    if x not in res_m0.params.index:
        raise ValueError(
            f"Exposure {x!r} is not a named parameter of the mediator model. "
            f"Available: {list(res_m0.params.index)}. Wrapped or transformed terms (C(x), log(x)) "
            "cannot be read back by name — use a derived column instead."
        )
    if m not in res_y0.params.index or x not in res_y0.params.index:
        raise ValueError(
            f"Mediator {m!r} / exposure {x!r} are not named parameters of the outcome model. "
            f"Available: {list(res_y0.params.index)}."
        )

    a0 = float(res_m0.params[x])
    b0 = float(res_y0.params[m])
    cprime0 = float(res_y0.params[x])
    ind0 = a0 * b0

    # ---- 2. Cluster bootstrap over subjects ------------------------------------
    # Index the frame by subject once: rebuilding each draw with a boolean scan per subject would be
    # quadratic in the number of subjects and dominate the runtime.
    by_subject: dict[Any, pd.DataFrame] = {
        sid: part for sid, part in df.groupby(subject_col, sort=False)
    }
    subjects = np.array(list(by_subject.keys()), dtype=object)

    draws: dict[str, list[float]] = {"a": [], "b": [], "c_prime": [], "indirect": [], "total": []}
    n_failed = 0
    cancelled = False

    for draw in range(int(n_boot)):
        if should_cancel is not None and should_cancel():
            cancelled = True
            log.info("Mediation bootstrap cancelled after %d of %d draws.", draw, n_boot)
            break

        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        parts = []
        for i, sid in enumerate(sampled):
            part = by_subject[sid].copy()
            # Distinct id per copy, so a subject drawn twice contributes two VC groups rather than
            # collapsing into one over-weighted cluster.
            part[subject_col] = f"{sid}__boot{i}"
            parts.append(part)
        df_boot = pd.concat(parts, ignore_index=True)

        try:
            res_m, res_y = _fit_pair(df_boot)
            a = float(res_m.params[x])
            b = float(res_y.params[m])
            cprime = float(res_y.params[x])
        except Exception as exc:  # MixedLM does not converge on every resample
            n_failed += 1
            log.debug("Mediation bootstrap draw %d failed: %s", draw, exc)
        else:
            draws["a"].append(a)
            draws["b"].append(b)
            draws["c_prime"].append(cprime)
            draws["indirect"].append(a * b)
            draws["total"].append(a * b + cprime)

        if progress is not None:
            progress(draw + 1, int(n_boot))

    # ---- 3. Summarize ----------------------------------------------------------
    alpha = 1.0 - float(ci)
    q_lo, q_hi = 100 * (alpha / 2.0), 100 * (1.0 - alpha / 2.0)
    dist = {k: np.asarray(v, dtype=float) for k, v in draws.items()}
    dist["direct"] = dist["c_prime"]

    out: dict[str, Any] = {
        "point_estimate": {
            "a": a0,
            "b": b0,
            "c_prime": cprime0,
            "indirect": ind0,
            "direct": cprime0,
            "total": ind0 + cprime0,
        },
        "bootstrap": {
            key: _bootstrap_summary(dist[key], q_lo=q_lo, q_hi=q_hi)
            for key in ("a", "b", "c_prime", "indirect", "direct", "total")
        },
        "n_boot": int(n_boot),
        "n_failed_draws": int(n_failed),
        "cancelled": cancelled,
        "ci": float(ci),
    }
    if return_dist:
        out["dist"] = dist
    if n_failed:
        log.warning(
            "Mediation bootstrap: %d of %d draws failed to converge and were skipped.",
            n_failed,
            n_boot,
        )
    return out


def mediation_boot_to_stats_df(boot_res: dict[str, Any], *, include_total: bool = True) -> pd.DataFrame:
    """Convert :func:`bootstrap_mediation_mixedlm_by_subject` output into a tidy path table."""
    pe = boot_res["point_estimate"]
    bs = boot_res["bootstrap"]

    rows: list[tuple[str, float, dict[str, Any]]] = [
        ("a", pe["a"], bs["a"]),
        ("b", pe["b"], bs["b"]),
        ("c'", pe["c_prime"], bs["c_prime"]),
        ("Indirect", pe["indirect"], bs["indirect"]),
        ("Direct", pe["direct"], bs["direct"]),
    ]
    if include_total and "total" in pe and "total" in bs:
        rows.append(("Total", pe["total"], bs["total"]))

    frame = pd.DataFrame(
        [
            {
                "path": path,
                "coef": float(coef),
                "ci_low": float(summary["ci_low"]),
                "ci_high": float(summary["ci_high"]),
                "pval": float(summary["p_value"]),
                "n": int(summary["n"]),
            }
            for path, coef, summary in rows
        ]
    )
    return _finalize_result_frame(frame, engine="mixedlm_bootstrap")


# ──────────────────────────────────────────────────────────────────────────────
# Engine (b) — pingouin, per grouping level
# ──────────────────────────────────────────────────────────────────────────────
def _canonical_pingouin_path(path: str, *, x: str, m: str, y: str) -> str:
    """
    Relabel pingouin's regression-style path names.

    pingouin names its rows after the models it fitted (``pi ~ X``, ``Y ~ pi``). The first is the
    ``a`` path and maps cleanly. The second does **not** map to ``b``: pingouin reports the
    coefficient of M in ``Y ~ M + covariates``, without X, whereas the ``b`` that multiplies into the
    indirect effect comes from ``Y ~ M + X + covariates``. On a 4000-row simulation with a true
    ``b = 0.5155`` adjusted, pingouin's row reads ``0.7402`` — so it is labeled ``b (unadjusted)``
    and kept out of the default forest, where sitting next to Indirect/Direct would invite reading
    ``a × b`` off the plot and getting the wrong number.
    """
    text = str(path).strip()
    normalized = text.replace(" ", "")
    if normalized in {f"{m}~X", f"{m}~{x}"}:
        return "a"
    if normalized in {f"Y~{m}", f"{y}~{m}"}:
        return "b (unadjusted)"
    return text


def pingouin_mediation(
    df: pd.DataFrame,
    *,
    x: str,
    m: str,
    y: str,
    covar: Sequence[str] = (),
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Single ``pingouin.mediation_analysis`` run, normalized to the tidy path table."""
    import pingouin as pg

    # pingouin seeds some bootstrap paths from the NumPy global RNG.
    np.random.seed(int(seed))
    raw = pg.mediation_analysis(
        data=df,
        x=x,
        m=m,
        y=y,
        covar=list(covar),
        n_boot=int(n_boot),
        alpha=float(alpha),
        seed=int(seed),
    )
    # ``return_dist=True`` would make this a (stats, dist) tuple; be tolerant either way.
    stats = raw[0] if isinstance(raw, tuple) else raw

    frame = pd.DataFrame(
        {
            "path": stats["path"].astype(str).map(lambda p: _canonical_pingouin_path(p, x=x, m=m, y=y)),
            "coef": pd.to_numeric(stats["coef"], errors="coerce"),
            "ci_low": pd.to_numeric(stats.get("CI[2.5%]", stats.get("CI2.5")), errors="coerce"),
            "ci_high": pd.to_numeric(stats.get("CI[97.5%]", stats.get("CI97.5")), errors="coerce"),
            "pval": pd.to_numeric(stats.get("pval"), errors="coerce"),
            "n": int(len(df)),
        }
    )
    return _finalize_result_frame(frame, engine="pingouin_by_level")


def pingouin_mediation_by_level(
    df: pd.DataFrame,
    *,
    x: str,
    m: str,
    y: str,
    covar: Sequence[str] = (),
    level_col: str = "territory",
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
    min_rows: int | None = None,
    dropna: bool = True,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Run :func:`pingouin_mediation` separately within each level of *level_col*.

    Levels with too few complete rows to fit the models are skipped: with *min_rows* unset the
    threshold is ``max(20, 5 + 3·len(covar))``, which scales with the number of parameters.

    Returns
    -------
    (by_level, summary)
        *by_level* maps each level to its tidy path table; *summary* has one row per level with the
        indirect / direct / total effects and their intervals, sorted by indirect effect.
    """
    covar = list(covar)
    needed = list(dict.fromkeys([level_col, x, m, y, *covar]))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not in the frame: {missing}")

    data = df[needed].copy()
    if dropna:
        data = data.dropna()
    threshold = int(min_rows) if min_rows is not None else max(20, 5 + 3 * len(covar))

    by_level: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    groups = list(data.groupby(level_col, sort=True))

    for i, (level, part) in enumerate(groups):
        if should_cancel is not None and should_cancel():
            log.info("Per-level mediation cancelled after %d of %d levels.", i, len(groups))
            break
        if len(part) < threshold:
            skipped.append(f"{level} (n={len(part)})")
            if progress is not None:
                progress(i + 1, len(groups))
            continue
        try:
            stats = pingouin_mediation(
                part, x=x, m=m, y=y, covar=covar, n_boot=n_boot, seed=seed, alpha=alpha
            )
        except Exception as exc:
            log.warning("Mediation failed for %s=%s: %s", level_col, level, exc)
            if progress is not None:
                progress(i + 1, len(groups))
            continue

        stats = stats.assign(level=str(level))
        by_level[str(level)] = stats

        row: dict[str, Any] = {"level": str(level), "n": int(len(part))}
        for path, prefix in (("Indirect", "indirect"), ("Direct", "direct"), ("Total", "total")):
            hit = stats.loc[stats["path"] == path]
            if hit.empty:
                continue
            first = hit.iloc[0]
            row[prefix] = float(first["coef"])
            row[f"{prefix}_lo"] = float(first["ci_low"])
            row[f"{prefix}_hi"] = float(first["ci_high"])
            row[f"{prefix}_p"] = float(first["pval"])
        rows.append(row)
        if progress is not None:
            progress(i + 1, len(groups))

    summary = pd.DataFrame(rows)
    if not summary.empty and "indirect" in summary.columns:
        summary = summary.sort_values("indirect").reset_index(drop=True)
    if skipped:
        log.warning(
            "Mediation skipped %d level(s) with fewer than %d complete rows: %s",
            len(skipped),
            threshold,
            ", ".join(skipped),
        )
        summary.attrs["skipped"] = skipped
    return by_level, summary


# ──────────────────────────────────────────────────────────────────────────────
# Engine (c) — statsmodels Mediation (pooled OLS)
# ──────────────────────────────────────────────────────────────────────────────
def statsmodels_mediation(
    df: pd.DataFrame,
    *,
    x: str,
    m: str,
    y: str,
    covariates: Sequence[str] = (),
    method: str = "parametric",
    n_rep: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    ``statsmodels.stats.mediation.Mediation`` with OLS mediator and outcome models.

    The models are fitted on the pooled frame, so repeated (subject × territory) rows count as
    independent observations — see :data:`POOLED_OLS_NOTE`, which is attached to the returned frame
    as ``frame.attrs["note"]``.

    Parameters
    ----------
    method : {"parametric", "bootstrap"}
        Passed to ``Mediation.fit``.
    """
    import statsmodels.api as sm
    from statsmodels.stats.mediation import Mediation

    covariates = list(covariates)
    needed = list(dict.fromkeys([x, m, y, *covariates]))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not in the frame: {missing}")
    data = df[needed].dropna().copy()
    if data.empty:
        raise ValueError("No complete rows for the mediation model.")

    rhs_m = " + ".join([x, *covariates])
    rhs_y = " + ".join([m, x, *covariates])
    # Mediation takes *unfitted* model instances and refits them per replication.
    mediator_model = sm.OLS.from_formula(f"{m} ~ {rhs_m}", data=data)
    outcome_model = sm.OLS.from_formula(f"{y} ~ {rhs_y}", data=data)

    np.random.seed(int(seed))
    summary = Mediation(outcome_model, mediator_model, x, m).fit(
        method=str(method), n_rep=int(n_rep)
    ).summary()

    # ``summary`` is indexed by ACME/ADE/Total effect/Prop. mediated with control/treated/average
    # variants; map the averages onto the canonical path names and keep the rest as-is.
    rename = {
        "ACME (average)": "Indirect",
        "ADE (average)": "Direct",
        "Total effect": "Total",
    }
    rows: list[dict[str, Any]] = []
    for label, row in summary.iterrows():
        rows.append(
            {
                "path": rename.get(str(label), str(label)),
                "coef": float(row.get("Estimate", np.nan)),
                "ci_low": float(row.get("Lower CI bound", np.nan)),
                "ci_high": float(row.get("Upper CI bound", np.nan)),
                "pval": float(row.get("P-value", np.nan)),
                "n": int(len(data)),
            }
        )
    frame = _finalize_result_frame(pd.DataFrame(rows), engine="statsmodels_parametric")
    frame.attrs["note"] = POOLED_OLS_NOTE
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Common result shape
# ──────────────────────────────────────────────────────────────────────────────
def _finalize_result_frame(frame: pd.DataFrame, *, engine: str, level: str | None = None) -> pd.DataFrame:
    """Give *frame* the canonical column set, ordering and legacy CI aliases."""
    out = frame.copy()
    out["engine"] = engine
    if "level" not in out.columns:
        out["level"] = level
    for col in RESULT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    # Sort the recognized paths into their conventional order, keeping any extras behind them.
    rank = {p: i for i, p in enumerate(_PATH_ORDER)}
    out["_rank"] = out["path"].map(lambda p: rank.get(str(p), len(rank)))
    out = out.sort_values(["_rank", "path"], kind="stable").drop(columns="_rank")
    out = out[list(RESULT_COLUMNS)].reset_index(drop=True)
    # Notebook-facing aliases; the plotting helpers accept either spelling.
    out["CI2.5"] = out["ci_low"]
    out["CI97.5"] = out["ci_high"]
    return out


def mediation_result_frame(
    result: Any,
    *,
    engine: str,
    level: str | None = None,
) -> pd.DataFrame:
    """
    Normalize any engine's output to the tidy path table.

    Accepts the bootstrap result *dict* from :func:`bootstrap_mediation_mixedlm_by_subject` or an
    already-tidy frame from the pingouin / statsmodels engines.
    """
    if isinstance(result, dict):
        return mediation_boot_to_stats_df(result)
    if isinstance(result, pd.DataFrame):
        return _finalize_result_frame(result, engine=engine, level=level)
    raise TypeError(f"Cannot build a mediation result frame from {type(result).__name__}.")


def run_mediation(
    df: pd.DataFrame,
    spec: MediationSpec,
    *,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """
    Run the engine named by *spec* and return a uniform result bundle.

    Returns
    -------
    dict
        ``spec``, ``engine``, ``paths`` (the tidy table), ``by_level`` and ``summary`` (per-level
        engine only), ``raw`` (the engine's native output), and ``note`` (a caveat to surface, or
        ``""``).
    """
    problem = spec.validate(df)
    if problem:
        raise ValueError(problem)

    bundle: dict[str, Any] = {
        "spec": spec,
        "engine": spec.engine,
        "by_level": None,
        "summary": None,
        "note": "",
    }

    if spec.engine == "mixedlm_bootstrap":
        formula_m, formula_y = spec.formulas()
        raw = bootstrap_mediation_mixedlm_by_subject(
            df,
            formula_m=formula_m,
            formula_y=formula_y,
            x=spec.x,
            m=spec.m,
            y=spec.y,
            group_col=spec.group_col,
            subject_col=spec.subject_col,
            required_columns=spec.required_columns(),
            n_boot=spec.n_boot,
            seed=spec.seed,
            ci=spec.ci,
            progress=progress,
            should_cancel=should_cancel,
        )
        bundle["raw"] = raw
        bundle["paths"] = mediation_boot_to_stats_df(raw)
        if raw.get("cancelled"):
            bundle["note"] = (
                f"Cancelled: summarizing the {raw['bootstrap']['indirect']['n']} draws that "
                "completed."
            )
        elif raw.get("n_failed_draws"):
            bundle["note"] = (
                f"{raw['n_failed_draws']} of {raw['n_boot']} draws failed to converge and were "
                "skipped."
            )

    elif spec.engine == "pingouin_by_level":
        by_level, summary = pingouin_mediation_by_level(
            df,
            x=spec.x,
            m=spec.m,
            y=spec.y,
            covar=spec.covariates,
            level_col=spec.group_col,
            n_boot=spec.n_boot,
            seed=spec.seed,
            alpha=1.0 - spec.ci,
            progress=progress,
            should_cancel=should_cancel,
        )
        bundle["raw"] = by_level
        bundle["by_level"] = by_level
        bundle["summary"] = summary
        # Pooled table too, so the Paths tab always has something to show.
        bundle["paths"] = pingouin_mediation(
            df.dropna(subset=spec.required_columns()),
            x=spec.x,
            m=spec.m,
            y=spec.y,
            covar=spec.covariates,
            n_boot=spec.n_boot,
            seed=spec.seed,
            alpha=1.0 - spec.ci,
        )
        bundle["note"] = POOLED_OLS_NOTE
        skipped = list(summary.attrs.get("skipped", [])) if summary is not None else []
        if skipped:
            bundle["note"] += f" Skipped levels with too few rows: {', '.join(skipped)}."

    else:  # statsmodels_parametric
        paths = statsmodels_mediation(
            df,
            x=spec.x,
            m=spec.m,
            y=spec.y,
            covariates=spec.covariates,
            n_rep=spec.n_boot,
            seed=spec.seed,
        )
        bundle["raw"] = paths
        bundle["paths"] = paths
        bundle["note"] = POOLED_OLS_NOTE

    return bundle


def render_mediation_info(bundle: Mapping[str, Any]) -> str:
    """Fixed-width text report of a :func:`run_mediation` bundle, for the report panel's Raw tab."""
    import io

    spec: MediationSpec = bundle["spec"]
    buffer = io.StringIO()
    _w = lambda line="": buffer.write(f"{line}\n")

    _w("=" * 88)
    _w(f"Mediation analysis — {ENGINE_LABELS.get(spec.engine, spec.engine)}")
    _w("=" * 88)
    _w(f"X (exposure): {spec.x}")
    _w(f"M (mediator): {spec.m}")
    _w(f"Y (outcome) : {spec.y}")
    if spec.covariates:
        _w(f"Covariates  : {', '.join(spec.covariates)}")
    if spec.engine in {"mixedlm_bootstrap", "pingouin_by_level"}:
        _w(f"Grouping    : {spec.group_col}")
    if spec.engine == "mixedlm_bootstrap":
        _w(f"Subject     : {spec.subject_col}")
    _w(f"Draws       : {spec.n_boot}   seed={spec.seed}   CI={spec.ci:.0%}")
    formula_m, formula_y = spec.formulas()
    _w(f"Mediator model: {formula_m}")
    _w(f"Outcome model : {formula_y}")
    _w()

    _w("Paths")
    _w("-" * 88)
    _w(f"{'Path':<14}{'Coef':>14}{'CI low':>14}{'CI high':>14}{'p':>14}{'n':>10}")
    for row in bundle["paths"].itertuples(index=False):
        _w(
            f"{str(row.path):<14}{row.coef:>14.5g}{row.ci_low:>14.5g}"
            f"{row.ci_high:>14.5g}{row.pval:>14.4g}{int(row.n):>10}"
        )
    _w()

    summary = bundle.get("summary")
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        _w(f"Indirect effect by {spec.group_col}")
        _w("-" * 88)
        _w(summary.to_string(index=False))
        _w()

    if bundle.get("note"):
        _w("Note")
        _w("-" * 88)
        _w(str(bundle["note"]))
    _w("=" * 88)
    return buffer.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def _ci_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Pick the CI column spelling present in *frame* (canonical or notebook-legacy)."""
    if "ci_low" in frame.columns and "ci_high" in frame.columns:
        return "ci_low", "ci_high"
    return "CI2.5", "CI97.5"


def plot_mediation_forest(
    stats_df: pd.DataFrame,
    *,
    ax: Any | None = None,
    title: str = "Mediation path estimates",
    order: Sequence[str] = _PATH_ORDER,
) -> Any:
    """Forest plot of path coefficients with their confidence intervals."""
    import matplotlib.pyplot as plt

    lo_col, hi_col = _ci_columns(stats_df)
    frame = stats_df.copy()
    keep = [p for p in order if p in set(frame["path"].astype(str))]
    if keep:
        frame = frame[frame["path"].astype(str).isin(keep)].copy()
        frame["path"] = pd.Categorical(frame["path"].astype(str), categories=keep, ordered=True)
        frame = frame.sort_values("path")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.7 * len(frame))))
    else:
        fig = ax.figure

    ypos = np.arange(len(frame))
    coef = frame["coef"].to_numpy(float)
    lo = frame[lo_col].to_numpy(float)
    hi = frame[hi_col].to_numpy(float)
    ax.errorbar(
        coef,
        ypos,
        xerr=np.vstack([coef - lo, hi - coef]),
        fmt="o",
        color="#4C72B0",
        ecolor="#4C72B0",
        capsize=4,
    )
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(frame["path"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Effect size (coef with CI)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_indirect_bootstrap(
    dist: Sequence[float],
    *,
    ci_low: float | None = None,
    ci_high: float | None = None,
    ax: Any | None = None,
    bins: int = 50,
) -> Any:
    """Histogram + KDE of the bootstrap distribution of the indirect effect, with CI markers."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    values = np.asarray(dist, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("The bootstrap distribution is empty — no draw converged.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.hist(values, bins=bins, alpha=0.35, color="#4C72B0", density=True)
    if values.size > 2 and np.ptp(values) > 0:  # a KDE needs some spread
        xs = np.linspace(values.min(), values.max(), 300)
        ax.plot(xs, gaussian_kde(values)(xs), color="#1F4E79", lw=2, label="KDE")

    ci_low = float(np.percentile(values, 2.5)) if ci_low is None else float(ci_low)
    ci_high = float(np.percentile(values, 97.5)) if ci_high is None else float(ci_high)
    ax.axvline(0, color="black", ls="--", lw=1.5, label="0 (null)")
    ax.axvline(ci_low, color="#C44E52", ls="--", lw=2, label=f"CI low: {ci_low:.4g}")
    ax.axvline(ci_high, color="#55A868", ls="--", lw=2, label=f"CI high: {ci_high:.4g}")

    ax.set_title("Bootstrap distribution of the indirect effect (a·b)")
    ax.set_xlabel("Indirect effect")
    ax.set_ylabel("Density")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    return fig


def plot_indirect_by_level(
    summary_df: pd.DataFrame,
    *,
    ax: Any | None = None,
    level_col: str = "level",
) -> Any:
    """Forest plot of the indirect effect within each grouping level."""
    import matplotlib.pyplot as plt

    if summary_df is None or summary_df.empty:
        raise ValueError("No per-level results to plot.")
    frame = summary_df.sort_values("indirect").copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(frame))))
    else:
        fig = ax.figure

    ypos = np.arange(len(frame))
    coef = frame["indirect"].to_numpy(float)
    lo = frame["indirect_lo"].to_numpy(float)
    hi = frame["indirect_hi"].to_numpy(float)
    ax.errorbar(
        coef,
        ypos,
        xerr=np.vstack([coef - lo, hi - coef]),
        fmt="o",
        capsize=4,
        color="#4C72B0",
        ecolor="#4C72B0",
    )
    ax.axvline(0, color="black", ls="--", lw=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(frame[level_col].astype(str))
    ax.set_xlabel("Indirect effect (a·b) with CI")
    ax.set_title(f"Indirect effect by {level_col}")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def _residualize(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Residuals of ``y`` regressed on ``X`` (with an intercept), for partial-regression plots."""
    import statsmodels.api as sm

    design = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, design, missing="drop").fit().resid


def plot_partial_paths_mediation(
    df: pd.DataFrame,
    *,
    x: str,
    m: str,
    y: str,
    covars: Sequence[str] = (),
    figsize: tuple[float, float] = (12, 5),
) -> Any:
    """
    Partial-regression plots of the ``a`` and ``b`` paths.

    Left: ``m`` against ``x``, both residualized on the covariates. Right: ``y`` against ``m``, both
    residualized on ``x`` + covariates — the adjusted relationship the ``b`` coefficient describes.
    These are OLS partial plots, offered as a visual check rather than as the mixed-model estimate.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    covars = list(covars)
    data = df[[x, m, y, *covars]].dropna().copy()
    if data.empty:
        raise ValueError("No complete rows for the partial-path plots.")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ---- a path: residual(m | covars) vs residual(x | covars) ------------------
    if covars:
        rx = _residualize(data[x], data[covars])
        rm = _residualize(data[m], data[covars])
    else:
        rx, rm = data[x], data[m]
    sns.regplot(x=rx, y=rm, ax=axes[0], scatter_kws={"alpha": 0.6}, line_kws={"color": "#C44E52"})
    axes[0].set_title(f"a path (adjusted): {x} → {m}")
    axes[0].set_xlabel(f"{x} residualized on covariates")
    axes[0].set_ylabel(f"{m} residualized on covariates")

    # ---- b path: residual(y | x+covars) vs residual(m | x+covars) --------------
    rm_x = _residualize(data[m], data[[x, *covars]])
    ry_x = _residualize(data[y], data[[x, *covars]])
    sns.regplot(x=rm_x, y=ry_x, ax=axes[1], scatter_kws={"alpha": 0.6}, line_kws={"color": "#55A868"})
    axes[1].set_title(f"b path (adjusted): {m} → {y} | {x}")
    axes[1].set_xlabel(f"{m} residualized on {x}+covariates")
    axes[1].set_ylabel(f"{y} residualized on {x}+covariates")

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Backwards compatibility
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_mediation_by_subject(
    df,
    formula_m,
    formula_y,
    model_cols,
    n_boot=500,
    seed=42,
):
    """
    Mediation bootstrap (a*b) resampling at the subject level.

    Deprecated
    ----------
    Kept for the ``nvitk.stats.mediation`` catalog entry and existing notebooks. It hardcodes
    ``x="pi"``, ``m="pp"`` and ``groups="territory"``; prefer
    :func:`bootstrap_mediation_mixedlm_by_subject`, which takes them as arguments.

    Returns:
        dict with:
            - indirect_dist (np.array)
            - ci_low, ci_high
            - mean
            - p_value (two-sided bootstrap)
    """
    log.warning(
        "bootstrap_mediation_by_subject is deprecated; use bootstrap_mediation_mixedlm_by_subject."
    )
    result = bootstrap_mediation_mixedlm_by_subject(
        df,
        formula_m=formula_m,
        formula_y=formula_y,
        x="pi",
        m="pp",
        y=str(formula_y).split("~", 1)[0].strip(),
        group_col="territory",
        subject_col="subject_uid",
        required_columns=list(model_cols or []),
        n_boot=n_boot,
        seed=seed,
    )
    indirect = result["dist"]["indirect"]
    summary = result["bootstrap"]["indirect"]
    return {
        "indirect_dist": indirect,
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "mean": summary["mean"],
        "p_value": summary["p_value"],
    }


__all__ = [
    "ENGINE_LABELS",
    "MEDIATION_ENGINES",
    "POOLED_OLS_NOTE",
    "RESULT_COLUMNS",
    "MediationSpec",
    "bootstrap_mediation_by_subject",
    "bootstrap_mediation_mixedlm_by_subject",
    "mediation_boot_to_stats_df",
    "mediation_result_frame",
    "pingouin_mediation",
    "pingouin_mediation_by_level",
    "plot_indirect_bootstrap",
    "plot_indirect_by_level",
    "plot_mediation_forest",
    "plot_partial_paths_mediation",
    "render_mediation_info",
    "run_mediation",
    "statsmodels_mediation",
]
