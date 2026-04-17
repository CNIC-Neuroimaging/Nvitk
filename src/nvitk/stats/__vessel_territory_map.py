"""Wide ``DataRepo.image`` columns → territory labels for PESA-style imaging.

Maps ``{region_id}_{variable_id}`` wide keys (and optional ``_fN`` frame suffixes
for ``flow_tseries``) to coarse vascular territories for **4D flow** and **ASL**
vascular atlas parcels. **Desikan** ``ctx-*`` ASL regions are labeled
``Desikan/Cortical``. **T1** / **FLAIR WMH** use atlas or CSV ``region_id`` values
from your dataset; this module does not assign them to ICA/venous territories.

``IMAGE_VARIABLE_IDS`` mirrors ``dataset/nvitk-dataset/catalog/variables.json``
(image domain) at last sync — refresh manually if the catalog changes.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import pandas as pd

__all__ = [
    "IMAGE_VARIABLE_IDS",
    "IMAGING_VARIABLE_TERRITORY_RULE",
    "FLOW_REGION_ID_TO_TERRITORY",
    "TERRITORY_FLOW_REGIONS",
    "TERRITORY_ASL_V8_REGIONS",
    "REGION_TO_TERRITORY_FLOW",
    "REGION_TO_TERRITORY_ASL_V8",
    "ParsedWideColumn",
    "parse_wide_image_column",
    "asl_vascular_parcel_to_territory",
    "asl_region_id_to_territory",
    "melt_imaging_territories",
]


# dataset/nvitk-dataset/catalog/variables.json — domain == "image"
IMAGE_VARIABLE_IDS: dict[str, tuple[str, ...]] = {
    "4dflow": (
        "basilar",
        "basilar_pi",
        "flow_mean",
        "flow_tseries",
        "pi",
        "psf",
        "left_aca",
        "left_aca_pi",
        "left_communicating",
        "left_communicating_pi",
        "left_ica",
        "left_ica_pi",
        "left_mca",
        "left_mca_pi",
        "left_pca",
        "left_pca_pi",
        "left_transverse",
        "left_transverse_pi",
        "right_aca",
        "right_aca_pi",
        "right_communicating",
        "right_communicating_pi",
        "right_ica",
        "right_ica_pi",
        "right_mca",
        "right_mca_pi",
        "right_pca",
        "right_pca_pi",
        "right_transverse",
        "right_transverse_pi",
        "sagital_sinus",
        "sagital_sinus_pi",
        "straight_sinus",
        "straight_sinus_pi",
        "tcbf",
    ),
    "asl": (
        "att_cov",
        "att_mean",
        "att_median",
        "cov_cbf",
        "mean_cbf",
        "left_aca_0",
        "left_aca_8",
        "left_basilar_0",
        "left_basilar_8",
        "left_mca_0",
        "left_mca_8",
        "left_pca_0",
        "left_pca_8",
        "right_aca_0",
        "right_aca_8",
        "right_basilar_0",
        "right_basilar_8",
        "right_mca_0",
        "right_mca_8",
        "right_pca_0",
        "right_pca_8",
        "watershed_0",
        "watershed_8",
        "mri_id_1",
    ),
    "t1": (
        "t1_cortical_volume",
        "t1_subcortical_volume",
    ),
    "flair": (
        "wmh_dist",
        "wmh_freq",
        "wmh_les",
        "wmh_reg",
    ),
}

IMAGING_VARIABLE_TERRITORY_RULE: dict[str, str] = {
    "basilar": "flow_region_table",
    "basilar_pi": "flow_region_table",
    "flow_mean": "flow_region_table",
    "pi": "flow_region_table",
    "flow_tseries": "flow_region_table",
    "psf": "flow_region_table",
    "tcbf": "global_no_territory",
    "mean_cbf": "asl_region_vascular_or_desikan",
    "cov_cbf": "asl_desikan_only_in_this_dataset",
    "att_mean": "asl_region_vascular_or_desikan",
    "att_median": "asl_region_vascular_or_desikan",
    "att_cov": "asl_native_inspect_regions",
    "left_aca_0": "asl_vascular_parcel_dict",
    "left_aca_8": "asl_vascular_parcel_dict",
    "right_aca_0": "asl_vascular_parcel_dict",
    "right_aca_8": "asl_vascular_parcel_dict",
    "left_mca_0": "asl_vascular_parcel_dict",
    "left_mca_8": "asl_vascular_parcel_dict",
    "right_mca_0": "asl_vascular_parcel_dict",
    "right_mca_8": "asl_vascular_parcel_dict",
    "left_pca_0": "asl_vascular_parcel_dict",
    "left_pca_8": "asl_vascular_parcel_dict",
    "right_pca_0": "asl_vascular_parcel_dict",
    "right_pca_8": "asl_vascular_parcel_dict",
    "left_basilar_0": "asl_vascular_parcel_dict",
    "left_basilar_8": "asl_vascular_parcel_dict",
    "right_basilar_0": "asl_vascular_parcel_dict",
    "right_basilar_8": "asl_vascular_parcel_dict",
    "watershed_0": "asl_vascular_parcel_dict",
    "watershed_8": "asl_vascular_parcel_dict",
    "mri_id_1": "non_measurement",
    "t1_cortical_volume": "t1_atlas_runtime_regions",
    "t1_subcortical_volume": "t1_atlas_runtime_regions",
    "wmh_dist": "wmh_csv_runtime_regions",
    "wmh_freq": "wmh_csv_runtime_regions",
    "wmh_les": "wmh_csv_runtime_regions",
    "wmh_reg": "wmh_csv_runtime_regions",
}

FLOW_REGION_ID_TO_TERRITORY: dict[str, str] = {
    "lica": "Internal Carotid Arteries",
    "rica": "Internal Carotid Arteries",
    "left_ica": "Internal Carotid Arteries",
    "right_ica": "Internal Carotid Arteries",
    "sssv": "Venous Drainage",
    "strv": "Venous Drainage",
    "ltsv": "Venous Drainage",
    "rtsv": "Venous Drainage",
    "sagital_sinus": "Venous Drainage",
    "straight_sinus": "Venous Drainage",
    "lmca": "Anterior Circulation",
    "rmca": "Anterior Circulation",
    "laca": "Anterior Circulation",
    "raca": "Anterior Circulation",
    "left_mca": "Anterior Circulation",
    "right_mca": "Anterior Circulation",
    "left_aca": "Anterior Circulation",
    "right_aca": "Anterior Circulation",
    "basi": "Posterior Circulation",
    "lpca": "Posterior Circulation",
    "rpca": "Posterior Circulation",
    "left_pca": "Posterior Circulation",
    "right_pca": "Posterior Circulation",
    "lcomm": "Circle of Willis / Communicating",
    "rcomm": "Circle of Willis / Communicating",
    "left_communicating": "Circle of Willis / Communicating",
    "right_communicating": "Circle of Willis / Communicating",
    "left_transverse": "Venous Drainage",
    "right_transverse": "Venous Drainage",
    "basilar": "Posterior Circulation",
}

TERRITORY_FLOW_REGIONS: dict[str, tuple[str, ...]] = {
    "Internal Carotid Arteries": ("lica", "rica"),
    "Venous Drainage": ("sssv", "strv", "ltsv", "rtsv"),
    "Anterior Circulation": ("lmca", "rmca", "laca", "raca"),
    "Posterior Circulation": ("basi", "lpca", "rpca"),
}

TERRITORY_ASL_V8_REGIONS: dict[str, tuple[str, ...]] = {
    "Internal Carotid Arteries": (),
    "Venous Drainage": (),
    "Anterior Circulation": (
        "left_mca_8",
        "right_mca_8",
        "left_aca_8",
        "right_aca_8",
    ),
    "Posterior Circulation": (
        "left_basilar_8",
        "right_basilar_8",
        "left_pca_8",
        "right_pca_8",
    ),
    "Watershed": ("watershed_0", "watershed_8"),
}

REGION_TO_TERRITORY_ASL_V8: dict[str, str] = {
    r: t for t, regs in TERRITORY_ASL_V8_REGIONS.items() for r in regs
}

# Primary synonym table; short tokens from TERRITORY_FLOW_REGIONS merged in.
REGION_TO_TERRITORY_FLOW: dict[str, str] = dict(FLOW_REGION_ID_TO_TERRITORY)
for _territory, _regs in TERRITORY_FLOW_REGIONS.items():
    for _r in _regs:
        REGION_TO_TERRITORY_FLOW.setdefault(_r, _territory)

# Longest suffix first so ``mean_cbf`` wins over embedded shorter tokens.
_WIDE_IMAGE_VARIABLE_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        {
            "t1_subcortical_volume",
            "t1_cortical_volume",
            "flow_tseries",
            "att_median",
            "mean_cbf",
            "att_mean",
            "cov_cbf",
            "att_cov",
            "flow_mean",
            "wmh_reg",
            "wmh_les",
            "wmh_freq",
            "wmh_dist",
            "pi",
            "psf",
            "tcbf",
        },
        key=lambda s: (-len(s), s),
    )
)


class ParsedWideColumn(NamedTuple):
    """``region_id`` + ``variable_id`` from a wide image column name."""

    region_id: str
    variable_id: str
    frame_suffix: str | None  # e.g. ``"f1"`` when the column ends with ``_f1``


def _extract_region_var_frame(key: str) -> tuple[str, str, str | None] | None:
    """Match ``{region}_{variable}`` or ``{region}_{variable}_{fN}``."""
    for var in _WIDE_IMAGE_VARIABLE_SUFFIXES:
        suf = "_" + var
        if key.endswith(suf):
            reg = key[: -len(suf)]
            if reg:
                return reg, var, None
        m = re.match(rf"^(.+){re.escape(suf)}_(f\d+)$", key, flags=re.IGNORECASE)
        if m:
            return m.group(1), var, m.group(2).lower()
    return None


def _strip_known_prefix(col: str) -> str:
    """
    Drop repeated leading ``{modality}_`` / ``{4dflow_vN}_`` segments produced by
    multi-modality / multi-pipeline wide pivots (see ``_compose_image_wide_keys``).

    Only ``4dflow``, ``4dflow_v{n}``, ``asl``, ``t1``, and ``flair`` are removed,
    so tokens like ``left_`` in ``left_mca_8_mean_cbf`` are never stripped.
    """
    body = str(col)
    for _ in range(8):
        m = re.match(r"^(4dflow_v\d+|4dflow|asl|t1|flair)_(.+)$", body, flags=re.IGNORECASE)
        if not m:
            break
        body = m.group(2)
    return body


def parse_wide_image_column(column: str) -> ParsedWideColumn | None:
    """
    Parse a wide image column name into region, variable, and optional frame
    index suffix (``f1``, ``f2``, … for ``flow_tseries``).
    """
    body = _strip_known_prefix(str(column))
    got = _extract_region_var_frame(body)
    if got is None:
        return None
    region_id, variable_id, frame_suffix = got
    return ParsedWideColumn(region_id, variable_id, frame_suffix)


def asl_vascular_parcel_to_territory(region_id: str) -> str | None:
    """Map ASL vascular parcel ``region_id`` (``…_0`` / ``…_8`` / ``…_12``) to a coarse territory."""
    if region_id.startswith("watershed_"):
        return "Watershed"
    if "mca_" in region_id or "aca_" in region_id:
        return "Anterior Circulation"
    if "pca_" in region_id or "basilar_" in region_id:
        return "Posterior Circulation"
    return None


def asl_region_id_to_territory(region_id: str) -> str | None:
    """
    Resolve ASL ``region_id`` to a display territory: vascular-8 table,
    vascular parcel heuristics, then Desikan ``ctx-*``.
    """
    t = REGION_TO_TERRITORY_ASL_V8.get(region_id)
    if t is not None:
        return t
    t2 = asl_vascular_parcel_to_territory(region_id)
    if t2 is not None:
        return t2
    if region_id.startswith("ctx-") or region_id.startswith("ctx_"):
        return "Desikan/Cortical"
    return None


def melt_imaging_territories(
    df: pd.DataFrame,
    *,
    id_cols: list[str] | tuple[str, ...] | None = None,
    flow_vars: list[str] | tuple[str, ...] | None = None,
    asl_vars: list[str] | tuple[str, ...] | None = None,
    unmapped_label: str = "Unmapped",
    include_frame_index: bool = False,
) -> pd.DataFrame:
    """
    Long table: one row per ``id_cols`` × wide imaging column, with ``territory``
    and ``modality_group`` (``flow`` / ``asl``).

    Columns that do not parse as ``{region}_{variable}`` image wide keys are
    omitted (clinical / cognitive / unknown).
    """
    _id = ("subject_uid",) if id_cols is None else tuple(id_cols)
    missing = [c for c in _id if c not in df.columns]
    if missing:
        raise KeyError(f"id_cols not in df: {missing}")

    _flow = ("flow_mean", "pi") if flow_vars is None else tuple(flow_vars)
    _asl = (
        "mean_cbf",
        "att_mean",
        "att_median",
        "cov_cbf",
        "att_cov",
    ) if asl_vars is None else tuple(asl_vars)

    id_list = list(_id)
    image_cols: list[str] = []
    parsed_meta: list[ParsedWideColumn | None] = []
    for col in df.columns:
        if col in _id:
            continue
        pw = parse_wide_image_column(col)
        parsed_meta.append(pw)
        if pw is None:
            continue
        if pw.variable_id in _flow or pw.variable_id in _asl:
            image_cols.append(col)

    if not image_cols:
        cols = list(_id) + ["territory", "modality_group", "region_id", "variable_id"]
        if include_frame_index:
            cols.append("frame_suffix")
        cols.append("value")
        return pd.DataFrame(columns=cols)

    long = df.melt(id_vars=id_list, value_vars=image_cols, var_name="_column", value_name="value")
    rows: list[dict] = []
    for _, row in long.iterrows():
        col = str(row["_column"])
        pw = parse_wide_image_column(col)
        if pw is None:
            continue
        var = pw.variable_id
        if var in _flow:
            territory = REGION_TO_TERRITORY_FLOW.get(pw.region_id)
            group = "flow"
        elif var in _asl:
            territory = asl_region_id_to_territory(pw.region_id)
            group = "asl"
        else:
            continue
        if territory is None:
            territory = unmapped_label
        rec = {k: row[k] for k in id_list}
        rec["territory"] = territory
        rec["modality_group"] = group
        rec["region_id"] = pw.region_id
        rec["variable_id"] = var
        if include_frame_index:
            rec["frame_suffix"] = pw.frame_suffix
        rec["value"] = row["value"]
        rows.append(rec)

    out = pd.DataFrame.from_records(rows)
    return out
