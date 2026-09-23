"""
Marginal means and pairwise contrasts for R models ``emmeans`` will not take.

Description
-----------
``emmeans`` is how every R engine here gets its confidence bands and its level-versus-level
comparisons — and it refuses some of the models this toolkit fits:

.. code-block:: text

    Can't handle an object of class "lmrob"

``robustbase`` has no ``emm_basis`` method registered, so a robust regression gets no marginal
means, no CI band and no significance brackets. That is not a gap in what can be computed: an
``lmrob`` fit exposes ``coef`` and a robust ``vcov`` like any other linear model, and a marginal
mean is a weighted average of design-matrix rows times those coefficients.

So this rebuilds the small part of ``emmeans`` that those quantities need, in R:

.. code-block:: text

    X   design rows for the reference grid, averaged over the factors not being held
    est = X β                      se = sqrt(diag(X V X'))
    contrast between rows i, j:  c = Xᵢ - Xⱼ,  est = c'β,  se = sqrt(c' V c)

including ``emmeans``' equal-weight averaging over factors outside the specification, which is
what makes a marginal mean marginal. It is checked against ``emmeans`` itself on models
``emmeans`` *does* accept — means, standard errors, intervals and pairwise contrasts all agree to
1e-9 — so the fallback is not a second, slightly different answer.

Used only when ``emmeans`` declines a model. Where it works it stays in charge, because it also
handles the models a design matrix does not describe on its own: an ``mmrm`` fit's covariance
term, an ``lmer`` fit's random effects.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Sequence

import pandas as pd

from nvitk.core.logger import Logger

log = Logger()

#: Reference grid, marginal means and pairwise contrasts, from a model's own design matrix.
_R_BASIS_HELPER = """
.nvitk_lm_basis <- function(model, specs_vars, at_name, at_values) {
  tt <- stats::delete.response(stats::terms(model))
  mf <- stats::model.frame(model)
  vars <- intersect(all.vars(tt), names(mf))

  # Character and logical predictors count as categorical. `model.frame` does not promote them
  # to factors, and treating one as numeric asks for the mean of a character vector — which is
  # NA with a warning, and a design matrix of NAs three lines later.
  is_cat <- function(col) is.factor(col) || is.character(col) || is.logical(col)
  lev_of <- function(col) if (is.factor(col)) levels(col) else sort(unique(as.character(col)))
  cats <- Filter(is_cat, mf[vars])
  xlev <- lapply(cats, lev_of)

  # Every categorical gets all of its levels, whether or not it is being held: the ones outside
  # the specification are averaged over below, which is what emmeans means by a marginal mean.
  parts <- list()
  for (v in vars) {
    col <- mf[[v]]
    parts[[v]] <-
      if (nzchar(at_name) && identical(v, at_name)) at_values
      else if (is_cat(col)) factor(xlev[[v]], levels = xlev[[v]])
      else if (v %in% specs_vars) sort(unique(col))
      else mean(col, na.rm = TRUE)
  }
  grid <- expand.grid(parts, stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
  X <- stats::model.matrix(tt, grid, xlev = xlev)

  b <- stats::coef(model)
  V <- stats::vcov(model)
  # A rank-deficient fit leaves NA coefficients; they carry no variance and belong in neither.
  b <- b[!is.na(b)]
  keep <- intersect(colnames(X), names(b))
  X <- X[, keep, drop = FALSE]; b <- b[keep]; V <- V[keep, keep, drop = FALSE]

  key <- do.call(paste, c(grid[, specs_vars, drop = FALSE], sep = "\\r"))
  idx <- split(seq_len(nrow(X)), factor(key, levels = unique(key)))
  Xa <- t(vapply(idx, function(i) colMeans(X[i, , drop = FALSE]), numeric(ncol(X))))
  labels <- grid[vapply(idx, `[`, integer(1), 1L), specs_vars, drop = FALSE]
  dof <- tryCatch(stats::df.residual(model), error = function(e) NA_real_)
  list(X = Xa, b = b, V = V, labels = labels,
       df = if (is.null(dof)) NA_real_ else as.numeric(dof))
}

.nvitk_lm_emmeans <- function(model, specs, at_name, at_values, lvl) {
  sv <- all.vars(stats::as.formula(specs))
  bs <- .nvitk_lm_basis(model, sv, at_name, at_values)
  est <- as.vector(bs$X %*% bs$b)
  se <- sqrt(pmax(0, rowSums((bs$X %*% bs$V) * bs$X)))
  dof <- bs$df
  crit <- if (is.finite(dof) && dof > 0) stats::qt(1 - (1 - lvl) / 2, dof)
          else stats::qnorm(1 - (1 - lvl) / 2)
  out <- bs$labels
  out$emmean <- est
  out$SE <- se
  out$df <- dof
  out$lower.CL <- est - crit * se
  out$upper.CL <- est + crit * se
  rownames(out) <- NULL
  out
}

.nvitk_lm_pairs <- function(model, factor_name, by_name) {
  sv <- if (nzchar(by_name)) c(factor_name, by_name) else factor_name
  bs <- .nvitk_lm_basis(model, sv, "", numeric(0))
  lab <- as.character(bs$labels[[factor_name]])
  grp <- if (nzchar(by_name)) as.character(bs$labels[[by_name]]) else rep("", length(lab))

  rows <- list()
  # Within each by level, not across it: the comparison a per-series bracket claims is the
  # simple effect inside that series, which on an interaction model is not the averaged one.
  for (g in unique(grp)) {
    idx <- which(grp == g)
    k <- length(idx)
    if (k < 2) next
    for (ii in seq_len(k - 1)) for (jj in (ii + 1):k) {
      i <- idx[ii]; j <- idx[jj]
      cv <- bs$X[i, ] - bs$X[j, ]
      est <- sum(cv * bs$b)
      se <- sqrt(max(0, drop(t(cv) %*% bs$V %*% cv)))
      stat <- if (se > 0) est / se else NA_real_
      dof <- bs$df
      p <- if (is.na(stat)) NA_real_
           else if (is.finite(dof) && dof > 0) 2 * stats::pt(abs(stat), dof, lower.tail = FALSE)
           else 2 * stats::pnorm(abs(stat), lower.tail = FALSE)
      rows[[length(rows) + 1]] <- data.frame(
        .by = g, contrast = paste(lab[i], "-", lab[j]), estimate = est,
        SE = se, df = dof, t.ratio = stat, p.value = p, stringsAsFactors = FALSE)
    }
  }
  if (!length(rows)) stop("fewer than two levels to compare")
  do.call(rbind, rows)
}
"""

_HELPER_LOADED = False


def _ensure_helper() -> None:
    """Define the R helpers once per process."""
    global _HELPER_LOADED
    if _HELPER_LOADED:
        return
    from rpy2.robjects import r as R_

    R_(_R_BASIS_HELPER)
    _HELPER_LOADED = True


def _call(name: str, *args: Any) -> pd.DataFrame:
    """Invoke one of the helpers and convert its frame."""
    from rpy2.robjects import default_converter, globalenv, pandas2ri
    from rpy2.robjects.conversion import localconverter

    _ensure_helper()
    with localconverter(default_converter + pandas2ri.converter):
        return pd.DataFrame(globalenv[name](*args))


def linear_emmeans(
    model: Any,
    specs: str,
    *,
    at_name: str = "",
    at_values: Sequence[float] | None = None,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """
    Marginal means from the model's own design matrix, shaped exactly like ``emmeans``'.

    Same signature and the same ``emmean / SE / df / lower.CL / upper.CL`` columns as
    :func:`~nvitk.stats.r_mixedlm.lme4_emmeans`, so the shared band and plotting code cannot
    tell which one answered.
    """
    from rpy2.robjects import FloatVector

    return _call(
        ".nvitk_lm_emmeans",
        model,
        str(specs),
        str(at_name),
        # Not ``at_values or []``: the callers pass a numpy grid, and a numpy array has no
        # truth value.
        FloatVector([] if at_values is None else [float(v) for v in at_values]),
        float(ci_level),
    )


def linear_pairs(model: Any, factor: str, *, by: str = "") -> pd.DataFrame:
    """
    Every level-versus-level contrast of *factor*, shaped like ``emmeans::contrast``'.

    With *by*, the comparisons are formed within each of its levels and the level is returned in
    a ``.by`` column, matching what the ``emmeans`` path renames its own by-column to.

    Unadjusted: the correction is applied in Python across the comparisons actually kept, so
    that every engine's brackets mean the same thing.
    """
    return _call(".nvitk_lm_pairs", model, str(factor), str(by))


__all__ = ["linear_emmeans", "linear_pairs"]
