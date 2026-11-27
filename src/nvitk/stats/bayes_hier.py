"""Bayesian hierarchical (multilevel) linear models, by Gibbs sampling.

The same models the mixed-model engines fit, written the same way (lme4 syntax), but estimated as
a posterior rather than a point:

    flow_mean ~ age_c + sex + C(territory) + (1 | subject_uid) + (1 + age_c | territory)

* every coefficient gets a full posterior: its mean, a credible interval that is a probability
  statement about the coefficient (95 % sure it lies there, given the model and the priors), and
  the probability of direction (how sure the effect is positive, or negative);
* the group effects (one per subject, per territory...) are *partially pooled*: each is shrunk
  toward its population mean by exactly as much as its own data are uncertain, which is what a
  hierarchical model is for: a territory measured in three subjects borrows strength from the
  others instead of reporting a wild mean;
* nothing is approximated by asymptotics: small groups, few subjects and variance components
  near zero (where REML struggles and reports 0) are handled by the posterior itself.

Model (``y`` continuous)::

    y = X b + sum_k Z_k u_k + e,      e ~ N(0, s2)
    u_k[g] ~ N(0, S_k)                 (one vector per level g of grouping factor k)
    b_j ~ N(0, (prior_scale * sd(y) / sd(x_j))^2),  intercept ~ N(mean(y), (10 sd(y))^2)
    S_k ~ InvWishart(q_k + 1, diag(sd(y)^2 / 2 / sd(w)^2)),   s2 ~ InvGamma(1, sd(y)^2 / 2)

The priors are weakly informative and *scaled to the data* (rstanarm's autoscale), so they mean the
same thing whatever the units: a coefficient is not expected to move y by more than a few standard
deviations per standard deviation of x. Every full conditional is conjugate, so each step is an
exact draw (no tuning, no divergences) and the sampler is vectorised over groups.

Diagnostics follow Vehtari et al. (2021): split R-hat (< 1.01 means the chains agree) and bulk
effective sample size per parameter; WAIC and the Bayesian R^2 summarise the fit.

No external dependency (NumPy, SciPy, patsy). Gaussian outcomes only: a binary or count outcome
belongs to a GLM, or to lme4's glmer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

ENGINE = "bayes"

_RANDOM_TERM = re.compile(r"\(\s*([^()|]*?)\s*(\|\|?)\s*([^()|]+?)\s*\)")


# --------------------------------------------------------------------------- formula


@dataclass
class RandomTerm:
    """One ``(expr | group)`` term."""

    expr: str
    group: str
    uncorrelated: bool = False

    @property
    def label(self) -> str:
        return f"({self.expr} {'||' if self.uncorrelated else '|'} {self.group})"


def parse_formula(formula: str) -> tuple[str, str, list[RandomTerm]]:
    """``(outcome, fixed right-hand side, random terms)`` of an lme4-style *formula*."""
    if "~" not in formula:
        raise ValueError("The formula needs an outcome: y ~ x + (1 | group).")
    lhs, rhs = formula.split("~", 1)
    terms = [RandomTerm(m.group(1).strip() or "1", m.group(3).strip(), m.group(2) == "||")
             for m in _RANDOM_TERM.finditer(rhs)]
    fixed = _RANDOM_TERM.sub("", rhs)
    fixed = re.sub(r"\+\s*\+", "+", fixed)
    fixed = re.sub(r"^\s*\+|\+\s*$", "", fixed.strip()).strip() or "1"
    return lhs.strip(), fixed, terms


def _group_columns(group: str) -> list[str]:
    return [g.strip() for g in group.split(":") if g.strip()]


# --------------------------------------------------------------------------- result


@dataclass
class BayesHierResult:
    """A fitted Bayesian hierarchical model: posterior draws and their summaries.

    Exposes the attributes the statsmodels-shaped helpers read (``params``, ``bse``,
    ``pvalues``, ``fe_params``...), so the coefficient table, the anatomical maps and the forest
    plots work on it like on any other engine.
    """

    formula: str
    outcome: str
    fixed_names: list[str]
    #: ``(chains, draws, p)`` fixed-effect draws.
    beta: np.ndarray
    #: ``(chains, draws)`` residual variance draws.
    sigma2: np.ndarray
    terms: list[RandomTerm]
    #: Per term: random-design column names, group levels, ``(chains, draws, G, q)`` effects,
    #: ``(chains, draws, q, q)`` covariance draws.
    term_columns: list[list[str]]
    term_levels: list[list[str]]
    term_effects: list[np.ndarray]
    term_cov: list[np.ndarray]
    n_obs: int
    y: np.ndarray
    #: ``(draws_kept, n)`` posterior-predictive draws (thinned).
    y_rep: np.ndarray
    log_lik: np.ndarray
    fitted_mean: np.ndarray
    settings: dict[str, Any] = field(default_factory=dict)
    design_info: Any = None
    summary_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # statsmodels-shaped views
    @property
    def params(self) -> pd.Series:
        return pd.Series(self.beta.reshape(-1, self.beta.shape[-1]).mean(axis=0), index=self.fixed_names)

    fe_params = params

    @property
    def bse(self) -> pd.Series:
        return pd.Series(self.beta.reshape(-1, self.beta.shape[-1]).std(axis=0, ddof=1), index=self.fixed_names)

    bse_fe = bse

    @property
    def pvalues(self) -> pd.Series:
        """Two-sided posterior tail probability, 2 * min(P(b > 0), P(b < 0)): small when the
        posterior sits away from zero. Not a frequentist p-value; the maps and stars read it."""
        draws = self.beta.reshape(-1, self.beta.shape[-1])
        pos = (draws > 0).mean(axis=0)
        return pd.Series(np.clip(2 * np.minimum(pos, 1 - pos), 1.0 / max(draws.shape[0], 1), 1.0), index=self.fixed_names)

    pvalues_fe = pvalues

    def cov_params(self) -> pd.DataFrame:
        draws = self.beta.reshape(-1, self.beta.shape[-1])
        return pd.DataFrame(np.cov(draws, rowvar=False).reshape(len(self.fixed_names), -1),
                            index=self.fixed_names, columns=self.fixed_names)

    @property
    def nobs(self) -> int:
        return int(self.n_obs)


# --------------------------------------------------------------------------- sampling


def _inv_wishart(rng: np.random.Generator, df: float, scale: np.ndarray) -> np.ndarray:
    """One draw of an inverse-Wishart(df, scale) matrix."""
    q = scale.shape[0]
    precision = np.linalg.inv(scale)
    chol = np.linalg.cholesky((precision + precision.T) / 2)
    # Bartlett decomposition of a Wishart(df, precision) draw, then inverted.
    a = np.zeros((q, q))
    for i in range(q):
        a[i, i] = np.sqrt(rng.chisquare(df - i))
        a[i, :i] = rng.standard_normal(i)
    w = chol @ a @ a.T @ chol.T
    return np.linalg.inv(w)


def _batched_mvn(rng: np.random.Generator, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """One draw per row of ``N(means[g], covs[g])`` (``(G, q)`` and ``(G, q, q)``)."""
    chol = np.linalg.cholesky((covs + np.swapaxes(covs, 1, 2)) / 2)
    z = rng.standard_normal(means.shape)
    return means + np.einsum("gij,gj->gi", chol, z)


#: Largest joint system (fixed effects + every group effect) drawn as one dense Gaussian block.
JOINT_LIMIT = 2500


def _joint_step(rng, wtw, wty, x, z_mats, groups_idx, levels, offsets, prior_prec, prior_mean, iw_scales, covs, s2):
    """Fixed and group effects drawn together, then each term's covariance."""
    from scipy.linalg import cho_factor, cho_solve, solve_triangular

    p = x.shape[1]
    dims = wtw.shape[0]
    prec = wtw / s2
    prec[np.diag_indices(p)] += prior_prec
    for k, z in enumerate(z_mats):
        q, g = z.shape[1], len(levels[k])
        inv_cov = np.linalg.inv(covs[k])
        base = offsets[k]
        idx = base + np.arange(g)[:, None, None] * q
        rows = (idx + np.arange(q)[None, :, None]).repeat(q, axis=2)
        cols = (idx + np.arange(q)[None, None, :]).repeat(q, axis=1)
        prec[rows, cols] += inv_cov[None, :, :]
    rhs = wty / s2
    rhs[:p] += prior_prec * prior_mean
    lower, _ = cho_factor(prec, lower=True, check_finite=False)
    mean = cho_solve((lower, True), rhs, check_finite=False)
    theta = mean + solve_triangular(lower, rng.standard_normal(dims), lower=True, trans="T", check_finite=False)
    beta = theta[:p]
    u, contrib, new_covs = [], [], []
    for k, z in enumerate(z_mats):
        q, g = z.shape[1], len(levels[k])
        uk = theta[offsets[k]:offsets[k] + g * q].reshape(g, q)
        u.append(uk)
        contrib.append(np.einsum("nq,nq->n", z, uk[groups_idx[k]]))
        new_covs.append(_inv_wishart(rng, q + 1 + g, iw_scales[k] + uk.T @ uk))
    return beta, u, new_covs, contrib


def _blocked_step(rng, x, y, xtx, z_mats, groups_idx, levels, ztz, prior_prec, prior_mean, iw_scales, u, covs,
                  contrib, s2):
    """Fixed effects, then each term's group effects (vectorised over levels), then covariances."""
    resid = y - (np.sum(contrib, axis=0) if contrib else 0.0)
    prec = xtx / s2 + np.diag(prior_prec)
    cov = np.linalg.inv(prec)
    mean = cov @ (x.T @ resid / s2 + prior_prec * prior_mean)
    beta = rng.multivariate_normal(mean, (cov + cov.T) / 2)
    xb = x @ beta
    u, contrib, covs = list(u), list(contrib), list(covs)
    for k, z in enumerate(z_mats):
        idx = groups_idx[k]
        partial = y - xb - (np.sum(contrib, axis=0) - contrib[k])
        q = z.shape[1]
        zr = np.zeros((len(levels[k]), q))
        np.add.at(zr, idx, z * partial[:, None])
        inv_cov = np.linalg.inv(covs[k])
        post_cov = np.linalg.inv(ztz[k] / s2 + inv_cov[None, :, :])
        post_mean = np.einsum("gij,gj->gi", post_cov, zr / s2)
        u[k] = _batched_mvn(rng, post_mean, post_cov)
        contrib[k] = np.einsum("nq,nq->n", z, u[k][idx])
        covs[k] = _inv_wishart(rng, q + 1 + len(levels[k]), iw_scales[k] + u[k].T @ u[k])
    return beta, u, covs, contrib


def fit_bayes_hier(
    data: pd.DataFrame,
    formula: str,
    *,
    chains: int = 4,
    draws: int = 1000,
    warmup: int = 1000,
    prior_scale: float = 2.5,
    seed: int | None = 1234,
    ppc_draws: int = 200,
    progress: Any = None,
) -> tuple[BayesHierResult, pd.DataFrame, dict[str, Any]]:
    """Fit *formula* (lme4 syntax) to *data* by Gibbs sampling; returns ``(result, model frame, meta)``."""
    from patsy import dmatrices, dmatrix

    outcome, fixed_rhs, terms = parse_formula(formula)
    used = set(re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", formula))
    for term in terms:
        used.update(_group_columns(term.group))
    columns = [c for c in data.columns if str(c) in used]
    frame = data[columns].dropna().copy()
    if frame.empty:
        raise ValueError("No complete rows for the model's columns.")
    y_mat, x_mat = dmatrices(f"{outcome} ~ {fixed_rhs}", frame, return_type="dataframe", NA_action="drop")
    frame = frame.loc[y_mat.index]
    y = np.asarray(y_mat.iloc[:, 0], dtype=float)
    x = np.asarray(x_mat, dtype=float)
    names = [str(c) for c in x_mat.columns]
    n, p = x.shape
    if n <= p:
        raise ValueError(f"{n} complete rows for {p} fixed effects: not enough data.")
    sd_y = float(np.std(y, ddof=1)) or 1.0
    mean_y = float(np.mean(y))

    # random designs
    z_mats, groups_idx, levels, t_cols = [], [], [], []
    expanded: list[RandomTerm] = []
    for term in terms:
        gcols = _group_columns(term.group)
        missing = [g for g in gcols if g not in frame.columns]
        if missing:
            raise ValueError(f"Grouping column not in the data: {', '.join(missing)}")
        key = frame[gcols].astype(str).agg(":".join, axis=1) if len(gcols) > 1 else frame[gcols[0]].astype(str)
        codes, uniques = pd.factorize(key, sort=True)
        w = dmatrix(term.expr if term.expr not in ("", "1") else "1", frame, return_type="dataframe")
        if term.expr.strip() == "0":
            continue
        parts = [[c] for c in w.columns] if term.uncorrelated else [list(w.columns)]
        for cols in parts:
            z_mats.append(np.asarray(w[cols], dtype=float))
            groups_idx.append(codes)
            levels.append([str(u) for u in uniques])
            t_cols.append([str(c) for c in cols])
            expanded.append(RandomTerm(" + ".join(str(c) for c in cols), term.group, term.uncorrelated))

    # priors
    sd_x = x.std(axis=0, ddof=1)
    prior_sd = np.where(sd_x > 1e-12, prior_scale * sd_y / np.where(sd_x > 1e-12, sd_x, 1.0), 10 * sd_y)
    prior_mean = np.zeros(p)
    for j, name in enumerate(names):
        if name == "Intercept":
            prior_mean[j], prior_sd[j] = mean_y, 10 * sd_y
    prior_prec = 1.0 / prior_sd ** 2
    a0, b0 = 1.0, sd_y ** 2 / 2
    iw_scales = []
    for z in z_mats:
        sd_w = z.std(axis=0, ddof=1)
        sd_w = np.where(sd_w > 1e-12, sd_w, 1.0)
        # An intercept column is constant: its scale is the outcome's own.
        iw_scales.append(np.diag(sd_y ** 2 / 2 / sd_w ** 2))

    xtx = x.T @ x
    k_terms = len(z_mats)
    # per-group cross products, fixed across iterations
    ztz = []
    for z, idx, lv in zip(z_mats, groups_idx, levels):
        q = z.shape[1]
        out = np.zeros((len(lv), q, q))
        np.add.at(out, idx, z[:, :, None] * z[:, None, :])
        ztz.append(out)

    # Joint block: the fixed effects and every group effect drawn together from one Gaussian. The
    # intercept and the random intercepts are strongly correlated a posteriori; drawn one block
    # after the other they crawl (R-hat 1.5 on a 60-subject example), drawn jointly they mix in a
    # few iterations. Dense, so used while the system stays small; larger models fall back to the
    # term-by-term sampler.
    from scipy import sparse

    dims = p + sum(len(lv) * z.shape[1] for z, lv in zip(z_mats, levels))
    joint = dims <= JOINT_LIMIT
    if joint:
        blocks = [sparse.csr_matrix(x)]
        offsets = [p]
        for z, idx, lv in zip(z_mats, groups_idx, levels):
            q, g = z.shape[1], len(lv)
            rows_i = np.repeat(np.arange(n), q)
            cols_i = (idx[:, None] * q + np.arange(q)[None, :]).ravel()
            blocks.append(sparse.csr_matrix((z.ravel(), (rows_i, cols_i)), shape=(n, g * q)))
            offsets.append(offsets[-1] + g * q)
        w_all = sparse.hstack(blocks).tocsr()
        wtw = (w_all.T @ w_all).toarray()
        wty = w_all.T @ y

    rng = np.random.default_rng(seed)
    total = warmup + draws
    beta_draws = np.zeros((chains, draws, p))
    s2_draws = np.zeros((chains, draws))
    u_draws = [np.zeros((chains, draws, len(lv), z.shape[1])) for z, lv in zip(z_mats, levels)]
    cov_draws = [np.zeros((chains, draws, z.shape[1], z.shape[1])) for z in z_mats]
    keep_every = max(1, (chains * draws) // max(ppc_draws, 1))
    y_rep, log_lik_rows, mu_sum = [], [], np.zeros(n)
    counter = 0

    for c in range(chains):
        # dispersed starts: chains that agree from different places are what R-hat checks
        beta = np.linalg.lstsq(x, y, rcond=None)[0] + rng.normal(0, 0.1, p) * prior_sd / prior_scale
        s2 = sd_y ** 2 * rng.uniform(0.5, 1.5)
        u = [np.zeros((len(lv), z.shape[1])) for z, lv in zip(z_mats, levels)]
        covs = [np.diag(np.diag(s)) * 2 * rng.uniform(0.5, 1.5) for s in iw_scales]
        contrib = [np.zeros(n) for _ in range(k_terms)]
        for it in range(total):
            if joint:
                beta, u, covs, contrib = _joint_step(
                    rng, wtw, wty, x, z_mats, groups_idx, levels, offsets, prior_prec, prior_mean, iw_scales, covs, s2,
                )
            else:
                beta, u, covs, contrib = _blocked_step(
                    rng, x, y, xtx, z_mats, groups_idx, levels, ztz, prior_prec, prior_mean, iw_scales, u, covs,
                    contrib, s2,
                )
            mu = x @ beta + (np.sum(contrib, axis=0) if k_terms else 0.0)
            ss = float(np.sum((y - mu) ** 2))
            s2 = 1.0 / rng.gamma(a0 + n / 2, 1.0 / (b0 + ss / 2))
            if it >= warmup:
                d = it - warmup
                beta_draws[c, d] = beta
                s2_draws[c, d] = s2
                for k in range(k_terms):
                    u_draws[k][c, d] = u[k]
                    cov_draws[k][c, d] = covs[k]
                mu_sum += mu
                ll = -0.5 * (np.log(2 * np.pi * s2) + (y - mu) ** 2 / s2)
                if counter % keep_every == 0:
                    y_rep.append(mu + rng.normal(0, np.sqrt(s2), n))
                log_lik_rows.append(ll)
                counter += 1
            if progress is not None and it % 200 == 0:
                progress(f"chain {c + 1}/{chains}: {it}/{total}")

    log_lik = np.asarray(log_lik_rows)
    result = BayesHierResult(
        formula=formula, outcome=outcome, fixed_names=names, beta=beta_draws, sigma2=s2_draws,
        terms=expanded, term_columns=t_cols, term_levels=levels, term_effects=u_draws, term_cov=cov_draws,
        n_obs=n, y=y, y_rep=np.asarray(y_rep), log_lik=log_lik, fitted_mean=mu_sum / max(counter, 1),
        settings={"chains": chains, "draws": draws, "warmup": warmup, "prior_scale": prior_scale, "seed": seed},
        design_info=x_mat.design_info,
    )
    result.summary_table = posterior_summary(result)
    model_df = frame.copy()
    model_df["_fitted"] = result.fitted_mean
    grouping = [t.group for t in expanded]
    meta = {
        "engine": ENGINE, "formula": formula, "sampler": "joint" if joint else "blocked", "n_rows": int(n), "fixed_formula": f"{outcome} ~ {fixed_rhs}", "outcome": outcome,
        "grouping_factors": list(dict.fromkeys(grouping)), "chains": chains, "draws": draws, "warmup": warmup,
        "prior_scale": prior_scale, "seed": seed,
    }
    return result, model_df, meta


# --------------------------------------------------------------------------- diagnostics


def _split_rhat(chains: np.ndarray) -> float:
    """Split R-hat of one parameter's ``(chains, draws)`` draws."""
    c, d = chains.shape
    if d < 4:
        return float("nan")
    half = d // 2
    parts = np.concatenate([chains[:, :half], chains[:, half:2 * half]], axis=0)
    m, n = parts.shape
    means = parts.mean(axis=1)
    within = parts.var(axis=1, ddof=1).mean()
    between = n * means.var(ddof=1)
    if within <= 0:
        return 1.0
    var_plus = (n - 1) / n * within + between / n
    return float(np.sqrt(var_plus / within))


def _ess(chains: np.ndarray) -> float:
    """Effective sample size of ``(chains, draws)`` draws (Geyer's initial monotone sequence)."""
    c, n = chains.shape
    if n < 4:
        return float(c * n)
    centred = chains - chains.mean(axis=1, keepdims=True)
    var = centred.var(axis=1).mean()
    if var <= 0:
        return float(c * n)
    acov = np.zeros(n)
    for chain in centred:
        f = np.fft.rfft(chain, n=2 * n)
        acov += np.fft.irfft(f * np.conj(f))[:n] / n
    acov /= c
    rho = acov / acov[0]
    total, t = 0.0, 1
    prev = np.inf
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        total += pair
        prev = pair
        t += 2
    tau = 1 + 2 * total
    return float(c * n / max(tau, 1e-9))


def _interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo = (1 - level) / 2
    return float(np.quantile(draws, lo)), float(np.quantile(draws, 1 - lo))


def posterior_summary(result: BayesHierResult, *, level: float = 0.95) -> pd.DataFrame:
    """The fixed effects' posterior: mean, sd, credible interval, probability of direction,
    R-hat and bulk ESS, in the columns every engine's coefficient table has."""
    from .mixedlm import significance_stars

    rows = []
    for j, name in enumerate(result.fixed_names):
        chains = result.beta[:, :, j]
        flat = chains.ravel()
        lo, hi = _interval(flat, level)
        pos = float((flat > 0).mean())
        p_tail = max(2 * min(pos, 1 - pos), 1.0 / flat.size)
        sd = float(flat.std(ddof=1))
        rows.append({
            "parameter": name, "coef": float(flat.mean()), "std_err": sd,
            "z": float(flat.mean() / sd) if sd else np.nan, "p_value": p_tail,
            "ci_low": lo, "ci_high": hi, "sig": significance_stars(p_tail) if (lo > 0 or hi < 0) else "",
            "pd": max(pos, 1 - pos), "rhat": _split_rhat(chains), "ess_bulk": _ess(chains),
        })
    return pd.DataFrame(rows)


def variance_components(result: BayesHierResult, *, level: float = 0.95) -> pd.DataFrame:
    """Posterior of every variance component (and the residual), as standard deviations."""
    rows = []
    for term, cols, cov in zip(result.terms, result.term_columns, result.term_cov):
        for i, col in enumerate(cols):
            sd = np.sqrt(cov[:, :, i, i])
            lo, hi = _interval(sd.ravel(), level)
            rows.append({"component": term.group, "kind": col, "var": float((sd ** 2).mean()), "sd": float(sd.mean()),
                         "sd_low": lo, "sd_high": hi, "rhat": _split_rhat(sd), "ess_bulk": _ess(sd)})
        for i in range(len(cols)):
            for j in range(i):
                corr = cov[:, :, i, j] / np.sqrt(cov[:, :, i, i] * cov[:, :, j, j])
                lo, hi = _interval(corr.ravel(), level)
                rows.append({"component": term.group, "kind": f"corr({cols[i]}, {cols[j]})", "var": np.nan,
                             "sd": float(corr.mean()), "sd_low": lo, "sd_high": hi, "rhat": _split_rhat(corr),
                             "ess_bulk": _ess(corr)})
    sd = np.sqrt(result.sigma2)
    lo, hi = _interval(sd.ravel(), level)
    rows.append({"component": "Residual", "kind": "", "var": float(result.sigma2.mean()), "sd": float(sd.mean()),
                 "sd_low": lo, "sd_high": hi, "rhat": _split_rhat(sd), "ess_bulk": _ess(sd)})
    return pd.DataFrame(rows)


def group_effects(result: BayesHierResult, term_index: int = 0, column: int = 0, *, level: float = 0.95) -> pd.DataFrame:
    """Posterior of each level's effect for one random term (the caterpillar)."""
    effects = result.term_effects[term_index][..., column]  # (chains, draws, G)
    flat = effects.reshape(-1, effects.shape[-1])
    lo = np.quantile(flat, (1 - level) / 2, axis=0)
    hi = np.quantile(flat, 1 - (1 - level) / 2, axis=0)
    return pd.DataFrame({"level": result.term_levels[term_index], "mean": flat.mean(axis=0), "ci_low": lo, "ci_high": hi})


def waic(result: BayesHierResult) -> dict[str, float]:
    """WAIC from the pointwise log-likelihood draws (lower is better)."""
    ll = result.log_lik
    lppd = float(np.sum(np.log(np.mean(np.exp(ll - ll.max(axis=0)), axis=0)) + ll.max(axis=0)))
    p_waic = float(np.sum(ll.var(axis=0, ddof=1)))
    return {"waic": -2 * (lppd - p_waic), "lppd": lppd, "p_waic": p_waic}


def bayes_r2(result: BayesHierResult) -> tuple[float, float, float]:
    """Gelman's Bayesian R^2: posterior mean and 95 % interval of var(fit) / (var(fit) + s2)."""
    fit_var = np.var(result.fitted_mean)
    r2 = fit_var / (fit_var + result.sigma2.ravel())
    return float(r2.mean()), float(np.quantile(r2, 0.025)), float(np.quantile(r2, 0.975))


def bayes_info_dict(result: BayesHierResult, *, outcome_name: str = "", meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """The report every engine produces (header, coefficients, random effects, fit statistics)."""
    meta = dict(meta or {})
    fe = result.summary_table if not result.summary_table.empty else posterior_summary(result)
    vc = variance_components(result)
    w = waic(result)
    r2, r2_lo, r2_hi = bayes_r2(result)
    worst_rhat = float(np.nanmax(list(fe["rhat"]) + list(vc["rhat"]))) if not fe.empty else float("nan")
    min_ess = float(np.nanmin(list(fe["ess_bulk"]) + list(vc["ess_bulk"]))) if not fe.empty else float("nan")
    s = result.settings
    return {
        "header": {
            "formula": result.formula, "n_obs": int(result.n_obs),
            "n_groups": len(result.term_levels[0]) if result.term_levels else None,
            "group_name": result.terms[0].group if result.terms else "—",
            "groups_detail": ", ".join(f"{t.group}: {len(lv)}" for t, lv in zip(result.terms, result.term_levels)),
            "outcome_name": outcome_name or result.outcome, "vc_group_name": "Variance components (posterior sd)",
            "model": (f"Bayesian hierarchical linear model — Gibbs, {s.get('chains')} chains × {s.get('draws')} draws "
                      f"(+{s.get('warmup')} warm-up), prior scale {s.get('prior_scale')}"),
        },
        "fixed_effects": fe,
        "random_effects": vc,
        "cov_re": pd.DataFrame(),
        "fit_statistics": {
            "bayes_r2": r2, "bayes_r2_ci": f"[{r2_lo:.3f}, {r2_hi:.3f}]", "waic": w["waic"], "p_waic": w["p_waic"],
            "max_rhat": worst_rhat, "min_ess_bulk": min_ess,
            "converged": bool(worst_rhat < 1.01 and min_ess > 100),
        },
        "has_vcomp": True,
        "notes": [
            "Intervals are 95 % equal-tailed credible intervals; p_value is the two-sided posterior tail "
            "probability 2·min(P(b>0), P(b<0)), and pd the probability of direction.",
            "R-hat < 1.01 and bulk ESS > 400 per parameter mean the chains agree and the summaries are precise."
            if worst_rhat < 1.01 else
            f"⚠ R-hat up to {worst_rhat:.3f}: the chains disagree. Run more draws (or simplify the random part).",
        ],
    }


# --------------------------------------------------------------------------- figures


def _flat_param_draws(result: BayesHierResult, names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    out = {}
    for j, name in enumerate(result.fixed_names):
        if names is None or name in names:
            out[name] = result.beta[:, :, j]
    for term, cols, cov in zip(result.terms, result.term_columns, result.term_cov):
        for i, col in enumerate(cols):
            label = f"sd({col} | {term.group})"
            if names is None or label in names:
                out[label] = np.sqrt(cov[:, :, i, i])
    if names is None or "sigma" in names:
        out["sigma"] = np.sqrt(result.sigma2)
    return out


def plot_trace(result: BayesHierResult, *, max_params: int = 8, include_intercept: bool = False) -> Any:
    """Trace (draws over iterations, one colour per chain) and posterior density per parameter."""
    import matplotlib.pyplot as plt

    params = _flat_param_draws(result)
    names = [k for k in params if include_intercept or k != "Intercept"][:max_params]
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 1.6 * len(names) + 0.6), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1]})
    for row, name in zip(axes, names):
        draws = params[name]
        for c in range(draws.shape[0]):
            row[0].plot(draws[c], linewidth=0.6, alpha=0.8, label=f"chain {c + 1}")
            row[1].hist(draws[c], bins=40, histtype="step", density=True)
        row[0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        rhat = _split_rhat(draws)
        row[0].set_title(f"R-hat {rhat:.3f} · ESS {_ess(draws):.0f}", fontsize=8, loc="right")
        row[1].set_yticks([])
    axes[0][0].legend(fontsize=7, ncol=4, loc="upper left")
    axes[-1][0].set_xlabel("iteration (after warm-up)")
    fig.suptitle("Trace plots")
    return fig


def plot_posterior_densities(result: BayesHierResult, *, include_intercept: bool = False, level: float = 0.95) -> Any:
    """One posterior density per fixed effect, its credible interval shaded, zero marked."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    names = [n for n in result.fixed_names if include_intercept or n != "Intercept"]
    cols = min(3, max(1, len(names)))
    rows = int(np.ceil(len(names) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.3 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        flat = result.beta[:, :, result.fixed_names.index(name)].ravel()
        lo, hi = _interval(flat, level)
        grid = np.linspace(flat.min(), flat.max(), 256)
        dens = gaussian_kde(flat)(grid) if flat.std() > 0 else np.zeros_like(grid)
        ax.plot(grid, dens, color="C0")
        inside = (grid >= lo) & (grid <= hi)
        ax.fill_between(grid[inside], dens[inside], alpha=0.3, color="C0")
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        pd_ = max((flat > 0).mean(), (flat < 0).mean())
        ax.set_title(f"{name}\npd {pd_:.3f}", fontsize=8)
        ax.set_yticks([])
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    fig.suptitle(f"Posterior densities ({level:.0%} credible interval shaded)")
    return fig


def plot_caterpillar(result: BayesHierResult, term_index: int = 0, column: int = 0, *, max_levels: int = 60) -> Any:
    """Each level's group effect with its credible interval, sorted: who sits above or below the
    population, and how sure the model is."""
    import matplotlib.pyplot as plt

    if not result.terms:
        raise ValueError("The model has no group-level terms: add one, e.g. (1 | subject_uid).")
    table = group_effects(result, term_index, column).sort_values("mean")
    shown = table if len(table) <= max_levels else pd.concat([table.head(max_levels // 2), table.tail(max_levels // 2)])
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.18 * len(shown) + 1)))
    pos = np.arange(len(shown))
    excl = (shown["ci_low"] > 0) | (shown["ci_high"] < 0)
    ax.hlines(pos, shown["ci_low"], shown["ci_high"], color=np.where(excl, "C3", "C0"), linewidth=1.2)
    ax.scatter(shown["mean"], pos, color=np.where(excl, "C3", "C0"), s=12, zorder=3)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(pos)
    ax.set_yticklabels(shown["level"], fontsize=7 if len(shown) > 30 else 8)
    term = result.terms[term_index]
    ax.set_xlabel(f"{result.term_columns[term_index][column]} effect (deviation from the population)")
    note = f" — {len(table)} levels, the {max_levels} most extreme shown" if len(table) > max_levels else ""
    ax.set_title(f"Group effects: {term.group}{note}")
    return fig


def plot_ppc(result: BayesHierResult, *, n_curves: int = 60) -> Any:
    """Posterior predictive check: the observed outcome's density over densities of data
    simulated from the fitted model. A model that fits draws data that look like the observed."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(6.5, 4))
    lo = min(result.y.min(), result.y_rep.min()) if result.y_rep.size else result.y.min()
    hi = max(result.y.max(), result.y_rep.max()) if result.y_rep.size else result.y.max()
    grid = np.linspace(lo, hi, 256)
    for rep in result.y_rep[:n_curves]:
        ax.plot(grid, gaussian_kde(rep)(grid), color="C0", alpha=0.12, linewidth=0.8)
    ax.plot(grid, gaussian_kde(result.y)(grid), color="black", linewidth=2, label="observed")
    ax.plot([], [], color="C0", alpha=0.5, label="simulated from the posterior")
    ax.set_xlabel(result.outcome)
    ax.set_yticks([])
    ax.legend()
    ax.set_title("Posterior predictive check")
    return fig


def plot_marginal_effect(result: BayesHierResult, data: pd.DataFrame, x: str, *, group: str | None = None,
                         include_points: bool = True, level: float = 0.95) -> Any:
    """Population-level prediction of the outcome against *x* (others at their reference: numeric
    covariates at their mean, factors at their first level), with its credible band; one curve per
    level of *group* when the fixed part contains it."""
    import matplotlib.pyplot as plt
    from patsy import build_design_matrices

    if x not in data.columns:
        raise ValueError(f"No column {x!r}.")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base = {}
    for col in data.columns:
        series = data[col]
        if col in (x, group):
            continue
        if pd.api.types.is_numeric_dtype(series):
            base[col] = float(series.mean())
        elif series.notna().any():
            base[col] = series.dropna().mode().iloc[0]
    numeric_x = pd.api.types.is_numeric_dtype(data[x])
    xs = np.linspace(float(data[x].min()), float(data[x].max()), 60) if numeric_x else sorted(data[x].dropna().unique(), key=str)
    groups = [None]
    # A level-specific curve: through the fixed part when the factor is a fixed effect, else (a
    # grouping factor of a random term) the population curve plus that level's posterior effect,
    # its partially pooled line.
    fixed_terms = [str(t) for t in getattr(result.design_info, "term_names", [])]
    in_fixed = bool(group) and any(group in t for t in fixed_terms)
    random_k = next((k for k, t in enumerate(result.terms) if group and t.group == group), None)
    if group and group in data.columns and (in_fixed or random_k is not None):
        groups = sorted(data[group].dropna().unique(), key=str)[:12]
    draws = result.beta.reshape(-1, result.beta.shape[-1])
    for k, g in enumerate(groups):
        grid = pd.DataFrame({x: xs})
        for col, val in base.items():
            grid[col] = val
        if g is not None:
            grid[group] = g
        try:
            (design,) = build_design_matrices([result.design_info], grid, return_type="dataframe")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Cannot predict over {x}: {exc}") from exc
        pred = draws @ np.asarray(design, dtype=float).T
        if g is not None and not in_fixed and random_k is not None:
            levels = result.term_levels[random_k]
            if str(g) in levels:
                li = levels.index(str(g))
                effects = result.term_effects[random_k][:, :, li, :].reshape(-1, len(result.term_columns[random_k]))
                for ci, col in enumerate(result.term_columns[random_k]):
                    if col == "Intercept":
                        z = np.ones(len(xs))
                    elif col == x and numeric_x:
                        z = np.asarray(xs, dtype=float)
                    elif col in base and isinstance(base[col], float):
                        z = np.full(len(xs), base[col])
                    else:
                        continue
                    pred = pred + effects[:, [ci]] * z[None, :]
        mean = pred.mean(axis=0)
        lo = np.quantile(pred, (1 - level) / 2, axis=0)
        hi = np.quantile(pred, 1 - (1 - level) / 2, axis=0)
        color = f"C{k % 10}"
        label = str(g) if g is not None else "posterior mean"
        if numeric_x:
            ax.plot(xs, mean, color=color, label=label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.2)
        else:
            pos = np.arange(len(xs)) + (k - (len(groups) - 1) / 2) * 0.12
            ax.errorbar(pos, mean, yerr=[mean - lo, hi - mean], fmt="o", color=color, label=label, capsize=3)
    if include_points:
        y = result.outcome if result.outcome in data.columns else None
        if y is not None and numeric_x:
            ax.scatter(data[x], data[y], s=8, color="grey", alpha=0.35, zorder=0, label="observed")
    if not numeric_x:
        ax.set_xticks(np.arange(len(xs)))
        ax.set_xticklabels([str(v) for v in xs], rotation=30, ha="right")
    ax.set_xlabel(x)
    ax.set_ylabel(result.outcome)
    ax.set_title(f"Posterior prediction of {result.outcome} vs {x} ({level:.0%} credible band)")
    ax.legend(fontsize=8)
    return fig


__all__ = [
    "BayesHierResult",
    "ENGINE",
    "RandomTerm",
    "bayes_info_dict",
    "bayes_r2",
    "fit_bayes_hier",
    "group_effects",
    "parse_formula",
    "plot_caterpillar",
    "plot_marginal_effect",
    "plot_posterior_densities",
    "plot_ppc",
    "plot_trace",
    "posterior_summary",
    "variance_components",
    "waic",
]
