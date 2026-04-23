"""Build long analysis tables from ``melt_imaging_territories`` + clinical wide frames."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ._vessel_territory_map import parse_wide_image_column

__all__ = [
    "aggregate_territory_measurements",
    "merge_subject_covariates",
    "build_analysis_df_from_repo_frames",
    "build_analysis_df_from_territory_definitions",
    "index_wide_image_columns_by_region_variable",
]


def _map_sex_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "male": 1,
        "m": 1,
        "1": 1,
        "female": 0,
        "f": 0,
        "0": 0,
    }
    return pd.to_numeric(s.map(mapping), errors="coerce")


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

    if 'sex' in clinical_wide.columns and clinical_wide['sex'].dtype in [str, object]:
        clinical_wide['sex'] = _map_sex_to_numeric(clinical_wide['sex'])

    cov = clinical_wide[cov_cols].drop_duplicates(subset=[subject_key], keep=dedupe)
    return long_df.merge(cov, on=subject_key, how="inner")


def index_wide_image_columns_by_region_variable(
    df: pd.DataFrame,
    *,
    id_cols: Sequence[str] | None = None,
    include_frame_index: bool = False,
) -> dict[tuple[str, str], list[str]]:
    """
    Map ``(region_id, variable_id)`` pairs to wide column names parseable by
    :func:`~nvitk.stats.parse_wide_image_column`.

    ``id_cols`` defaults to ``(\"subject_uid\",)`` when present in ``df``; otherwise
    no id columns are assumed.
    """
    if id_cols is None:
        id_cols = ("subject_uid",) if "subject_uid" in df.columns else ()
    id_set = set(id_cols)
    out: dict[tuple[str, str], list[str]] = {}
    for col in df.columns:
        if col in id_set:
            continue
        pw = parse_wide_image_column(col)
        if pw is None:
            continue
        if not include_frame_index and pw.frame_suffix is not None:
            continue
        key = (pw.region_id, pw.variable_id)
        out.setdefault(key, []).append(col)
    return out


def _rowwise_agg_across_columns(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    agg: str,
) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index)
    num = df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce")
    if agg == "mean":
        return num.mean(axis=1)
    if agg == "median":
        return num.median(axis=1)
    if agg == "sum":
        return num.sum(axis=1)
    if agg == "min":
        return num.min(axis=1)
    if agg == "max":
        return num.max(axis=1)
    raise ValueError(f"unsupported agg={agg!r}; use mean|median|sum|min|max")


def build_analysis_df_from_territory_definitions(
    image_df_wide: pd.DataFrame,
    clinical_wide: pd.DataFrame,
    territory_definitions: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    flow_vars: Sequence[str],
    asl_vars: Sequence[str],
    covariate_cols: Sequence[str],
    subject_key: str = "subject_uid",
    dedupe_clinical: str = "first",
    agg: str = "mean",
    include_frame_index: bool = False,
) -> pd.DataFrame:
    """
    One row per ``(subject_key, territory)`` using an **explicit** per-territory
    region list for 4D flow and ASL.

    Unlike :func:`melt_imaging_territories` + :func:`build_analysis_df_from_repo_frames`,
    the same ASL ``region_id`` (e.g. ``ctx-whole-brain``) may contribute to **more
    than one** analysis territory (e.g. ``ICA-WB`` and ``Venous-WB``) with
    different flow region groupings, because territories are not derived from a
    single inverted ``region → territory`` map.

    Parameters
    ----------
    territory_definitions
        ``{territory_name: {\"flow\": (region_ids...), \"asl\": (region_ids...)}}``.
        Either ``flow`` or ``asl`` may be omitted or empty; missing variables for
        that territory are filled with NaN. Region IDs must match ``region_id`` in
        wide column names after :func:`~nvitk.stats.parse_wide_image_column`.
    flow_vars, asl_vars
        ``variable_id`` tokens to pull from wide columns (e.g. ``flow_mean``,
        ``mean_cbf``).
    include_frame_index
        When True, include ``flow_tseries`` (and similar) columns with ``_fN``
        suffixes in the column index; default False matches :func:`melt_imaging_territories`.
    """
    if subject_key not in image_df_wide.columns:
        raise KeyError(f"{subject_key!r} not in image_df_wide")
    col_index = index_wide_image_columns_by_region_variable(
        image_df_wide,
        id_cols=(subject_key,),
        include_frame_index=include_frame_index,
    )

    chunks: list[pd.DataFrame] = []
    for territory, spec in territory_definitions.items():
        flow_regions = tuple(spec.get("flow", ()) or ())
        asl_regions = tuple(spec.get("asl", ()) or ())
        block = image_df_wide[[subject_key]].copy()
        block["territory"] = territory
        for var in flow_vars:
            cols = [c for r in flow_regions for c in col_index.get((r, var), [])]
            block[var] = _rowwise_agg_across_columns(image_df_wide, cols, agg=agg)
        for var in asl_vars:
            cols = [c for r in asl_regions for c in col_index.get((r, var), [])]
            block[var] = _rowwise_agg_across_columns(image_df_wide, cols, agg=agg)
        chunks.append(block)

    if not chunks:
        cols = [subject_key, "territory", *flow_vars, *asl_vars]
        wide = pd.DataFrame(columns=cols)
    else:
        wide = pd.concat(chunks, ignore_index=True)

    return merge_subject_covariates(
        wide,
        clinical_wide,
        covariate_cols,
        subject_key=subject_key,
        dedupe=dedupe_clinical,
    )


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
