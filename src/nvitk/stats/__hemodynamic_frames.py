"""Build long analysis tables from ``melt_imaging_territories`` + clinical wide frames."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

__all__ = [
    "aggregate_territory_measurements",
    "merge_subject_covariates",
    "build_analysis_df_from_repo_frames",
]


def aggregate_territory_measurements(
    territory_df: pd.DataFrame,
    variable_ids: Sequence[str],
    *,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    For each ``variable_id``, mean (or other ``agg``) ``value`` within
    ``(subject_uid, territory)``.

    Expects columns: ``subject_uid``, ``territory``, ``variable_id``, ``value``.
    """
    if territory_df.empty:
        return pd.DataFrame(columns=["subject_uid", "territory", *variable_ids])

    required = {"subject_uid", "territory", "variable_id", "value"}
    missing = required - set(territory_df.columns)
    if missing:
        raise KeyError(f"territory_df missing columns: {sorted(missing)}")

    parts: list[pd.DataFrame] = []
    for var in variable_ids:
        sub = territory_df.loc[territory_df["variable_id"] == var, ["subject_uid", "territory", "value"]].copy()
        if sub.empty:
            continue
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        part = sub.groupby(["subject_uid", "territory"], as_index=False, sort=False).agg(
            **{str(var): ("value", agg)}
        )
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=["subject_uid", "territory", *variable_ids])

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["subject_uid", "territory"], how="outer")
    return out


def merge_subject_covariates(
    long_df: pd.DataFrame,
    clinical_wide: pd.DataFrame,
    covariate_cols: Iterable[str],
    *,
    subject_key: str = "subject_uid",
    dedupe: str = "first",
) -> pd.DataFrame:
    """
    Attach subject-level covariates (one row per ``subject_key`` in ``clinical_wide``).

    ``long_df`` must include ``subject_key`` and typically ``territory`` plus outcome
    columns from :func:`aggregate_territory_measurements`.
    """
    cov_cols = list(dict.fromkeys([subject_key, *covariate_cols]))
    missing = [c for c in cov_cols if c not in clinical_wide.columns]
    if missing:
        raise KeyError(f"clinical_wide missing columns: {missing}")

    cov = clinical_wide[cov_cols].drop_duplicates(subset=[subject_key], keep=dedupe)
    return long_df.merge(cov, on=subject_key, how="inner")


def build_analysis_df_from_repo_frames(
    territory_df: pd.DataFrame,
    clinical_wide: pd.DataFrame,
    *,
    imaging_variable_ids: Sequence[str],
    covariate_cols: Sequence[str],
    subject_key: str = "subject_uid",
    dedupe_clinical: str = "first",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    One row per ``(subject_uid, territory)`` with aggregated imaging measures and
    merged covariates — same grain as ``cacs.ipynb`` after melt (long format).

    Parameters
    ----------
    territory_df
        Output of :func:`nvitk.stats.melt_imaging_territories`.
    clinical_wide
        Wide clinical / joined table (e.g. ``repo.join([clinical, plaque, cognitive])``).
    imaging_variable_ids
        ``variable_id`` values to aggregate (e.g. ``(\"mean_cbf\", \"pi\")``).
    covariate_cols
        Columns to copy from ``clinical_wide`` (must exist; typically ``tacsctot_group``,
        ``age_at_mri``, ``sex``, …).
    """
    wide = aggregate_territory_measurements(territory_df, imaging_variable_ids, agg=agg)
    return merge_subject_covariates(
        wide,
        clinical_wide,
        covariate_cols,
        subject_key=subject_key,
        dedupe=dedupe_clinical,
    )
