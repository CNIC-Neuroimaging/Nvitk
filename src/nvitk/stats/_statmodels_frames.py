"""Long-form image measurements → Statmodels analysis frames.

Uses ``DataRepo.image(..., wide=False)`` so region/variable columns stay
explicit (avoids single-variable wide pivots that drop ``_{variable}`` suffixes).
"""

from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd

from nvitk.pipes.qvtpy.common.morpho_db_publish import _SCALAR_VARS as _MORPHO_SCALAR_VARS

GroupingMode = Literal[
    "vessel",
    "tree",
    "hemisphere",
    "region",
    "territory",
]

PipelineKind = Literal["qvtpy", "asl", "t1", "flair", "tof"]

KIND_MODALITY: dict[str, str] = {
    "qvtpy": "4dflow",
    "asl": "asl",
    "t1": "t1",
    "flair": "flair",
    "tof": "tof",
}

# Vessel-wise LOC hemodynamics (per branch / LOC).
QVTPY_VESSEL_FEATURES: tuple[str, ...] = ("flow_mean", "pi", "ri")
# Tree-level hemodynamics (per arterial root: L_ICA / R_ICA / Basilar).
QVTPY_TREE_FEATURES: tuple[str, ...] = (
    "pwv",
    "pwv_fielding_xcor",
    "pitc_slope",
    "pitc_intercept",
)
ASL_FEATURES: tuple[str, ...] = ("mean_cbf", "cov_cbf", "att_mean", "att_cov")
T1_FEATURES: tuple[str, ...] = ("t1_cortical_volume", "t1_subcortical_volume")
FLAIR_FEATURES: tuple[str, ...] = ("wmh_dist", "wmh_freq", "wmh_les", "wmh_reg")
TOF_FEATURES: tuple[str, ...] = tuple(_MORPHO_SCALAR_VARS.keys())

FEATURE_ALIASES: dict[str, str] = {
    "flow": "flow_mean",
    "pwv_bjornfoot": "pwv",
    "pwv_fielding": "pwv_fielding_xcor",
    "pitc": "pitc_slope",
    "mean_cbf": "mean_cbf",
    "att_mean": "att_mean",
    "cov_cbf": "cov_cbf",
    "att_cov": "att_cov",
}

_LR_PREFIX_RE = re.compile(r"^(left_|right_|l_|r_)", re.IGNORECASE)
_SIDE_LETTER_RE = re.compile(r"^([LR])([A-Z][A-Z0-9_]*)$", re.IGNORECASE)
_TREE_ICA = frozenset({"L_ICA", "R_ICA", "LICA", "RICA", "LEFT_ICA", "RIGHT_ICA"})
_TREE_BASILAR = frozenset({"BASILAR", "BASI"})


def resolve_feature_id(feature: str) -> str:
    """Resolve a friendly feature alias (e.g. ``\"flow\"``) to its canonical ``variable_id`` (e.g. ``\"flow_mean\"``)."""
    text = str(feature).strip()
    return FEATURE_ALIASES.get(text, text)


def features_for_kind(kind: str) -> list[str]:
    """List the available feature ``variable_id``s for a pipeline kind (qvtpy/asl/t1/flair/tof)."""
    kind = str(kind).strip().lower()
    if kind == "qvtpy":
        return [*QVTPY_VESSEL_FEATURES, *QVTPY_TREE_FEATURES]
    if kind == "asl":
        return list(ASL_FEATURES)
    if kind == "t1":
        return list(T1_FEATURES)
    if kind == "flair":
        return list(FLAIR_FEATURES)
    if kind == "tof":
        return list(TOF_FEATURES)
    return []


def feature_family(kind: str, variable_id: str) -> str:
    """Return ``vessel`` / ``tree`` / ``region`` for grouping UI defaults."""
    vid = resolve_feature_id(variable_id)
    kind = str(kind).strip().lower()
    if kind == "qvtpy":
        if vid in QVTPY_TREE_FEATURES:
            return "tree"
        return "vessel"
    if kind in {"asl", "t1", "flair"}:
        return "region"
    if kind == "tof":
        return "vessel"
    return "region"


def grouping_choices_for(kind: str, variable_id: str) -> list[tuple[str, str]]:
    """Ordered ``(label, key)`` grouping options for the current modality/feature."""
    family = feature_family(kind, variable_id)
    kind = str(kind).strip().lower()
    if kind == "qvtpy" and family == "tree":
        return [
            ("By tree (L_ICA / R_ICA / Basilar)", "tree"),
            ("By hemisphere (avg L/R ICA; Basilar kept)", "hemisphere"),
        ]
    if kind == "qvtpy" and family == "vessel":
        return [
            ("Vessel-wise (region_id)", "vessel"),
            ("By hemisphere (avg L/R pairs)", "hemisphere"),
        ]
    if kind == "tof":
        return [
            ("Vessel-wise (eICAB vessel)", "vessel"),
            ("By hemisphere (avg L/R pairs)", "hemisphere"),
        ]
    if kind == "asl":
        return [
            ("Atlas region", "region"),
            ("Vascular territory (melted)", "territory"),
        ]
    # t1 / flair: keep published regions
    return [("Region", "region")]


def region_to_hemisphere_pair_key(region_id: str) -> str:
    """Collapse L/R counterparts to one key (ICA, MCA, …); midline stays itself."""
    raw = str(region_id).strip()
    if not raw:
        return raw
    su = raw.upper().replace("-", "_")
    if su in _TREE_ICA:
        return "ICA"
    if su in _TREE_BASILAR:
        return "Basilar"
    for prefix in ("LEFT_", "RIGHT_", "L_", "R_"):
        if su.startswith(prefix) and len(su) > len(prefix):
            return su[len(prefix) :].lower()
    m = _SIDE_LETTER_RE.match(su)
    if m:
        return m.group(2).upper() if len(m.group(2)) <= 4 else m.group(2).lower()
    stripped = _LR_PREFIX_RE.sub("", raw)
    return stripped if stripped else raw


def assign_group_key(
    region_id: str,
    *,
    grouping: str,
    kind: str = "qvtpy",
    variable_id: str = "flow_mean",
) -> str:
    """Map a published ``region_id`` to the analysis ``group_key``."""
    rid = str(region_id).strip()
    grouping = str(grouping).strip().lower()
    if grouping in {"vessel", "tree", "region"}:
        return rid
    if grouping == "hemisphere":
        return region_to_hemisphere_pair_key(rid)
    if grouping == "territory":
        from nvitk.stats._vessel_territory_map import (
            REGION_TO_TERRITORY_FLOW,
            asl_region_id_to_territory,
        )

        kind = str(kind).strip().lower()
        if kind == "asl":
            return asl_region_id_to_territory(rid) or "Unmapped"
        return REGION_TO_TERRITORY_FLOW.get(rid) or REGION_TO_TERRITORY_FLOW.get(
            rid.lower(), "Unmapped"
        )
    return rid


def atlas_for_request(kind: str, variable_id: str, atlas: str | None) -> str | None:
    """Atlas argument for ``DataRepo.image`` (ASL vascular/desikan; T1 cortical/subcortical)."""
    kind = str(kind).strip().lower()
    vid = resolve_feature_id(variable_id)
    if kind == "asl":
        return (atlas or "vascular-8").strip() or "vascular-8"
    if kind == "t1":
        if vid == "t1_subcortical_volume":
            return "subcortical"
        return "cortical"
    return None


def build_long_analysis_frame(
    repo: Any,
    *,
    pipeline_kind: str,
    pipeline: str,
    feature: str,
    grouping: str,
    atlas: str | None = None,
    clinical_vars: list[str] | None = None,
    cognitive_vars: list[str] | None = None,
) -> pd.DataFrame:
    """Load long-form image rows, assign group keys, aggregate, attach covariates."""
    from nvitk.stats._hemodynamic_frames import (
        aggregate_territory_measurements,
        build_analysis_df_from_repo_frames,
    )

    kind = str(pipeline_kind).strip().lower()
    modality = KIND_MODALITY.get(kind)
    if modality is None:
        raise ValueError(f"Unknown pipeline kind {pipeline_kind!r}.")

    variable_id = resolve_feature_id(feature)
    # ``latest`` is only an alias on some pipelines (e.g. 4dflow_v3). Treat it as
    # "catalog default for this modality" so ASL / T1 / FLAIR / TOF resolve too.
    raw_pipeline = (pipeline or "").strip()
    if not raw_pipeline or raw_pipeline.lower() in {"latest", "default", "current", "last"}:
        pipeline_sel: str | None = None
    else:
        pipeline_sel = raw_pipeline
    clinical_vars = list(clinical_vars or [])
    cognitive_vars = list(cognitive_vars or [])

    image_kwargs: dict[str, Any] = {
        "modality": modality,
        "pipeline": pipeline_sel,
        "variables": [variable_id],
        "wide": False,
    }
    atlas_sel = atlas_for_request(kind, variable_id, atlas)
    if atlas_sel:
        image_kwargs["atlas"] = atlas_sel

    image = repo.image(**image_kwargs)
    if image is None or image.empty:
        pipe_txt = pipeline_sel if pipeline_sel else "catalog-default"
        raise ValueError(
            f"No {modality!r} image measurements for variable={variable_id!r} "
            f"(pipeline={pipe_txt!r}"
            + (f", atlas={atlas_sel!r}" if atlas_sel else "")
            + ")."
        )

    required = {"subject_uid", "region_id", "variable_id"}
    missing = required - set(image.columns)
    if missing:
        raise ValueError(f"image_measurements missing columns: {sorted(missing)}")

    long = image.copy()
    if "value" not in long.columns:
        if "value_num" in long.columns:
            long["value"] = pd.to_numeric(long["value_num"], errors="coerce")
        elif "value_text" in long.columns:
            long["value"] = pd.to_numeric(long["value_text"], errors="coerce")
        else:
            raise ValueError("image_measurements has no value / value_num column.")
    else:
        long["value"] = pd.to_numeric(long["value"], errors="coerce")

    long = long.loc[long["variable_id"].astype(str) == variable_id].copy()
    if long.empty:
        raise ValueError(f"No rows for variable_id={variable_id!r} after filtering.")

    long["territory"] = [
        assign_group_key(
            rid, grouping=grouping, kind=kind, variable_id=variable_id
        )
        for rid in long["region_id"].astype(str)
    ]
    long["region_id"] = long["region_id"].astype(str)
    long["modality_group"] = kind

    # Covariates
    frames: list[pd.DataFrame] = []
    if clinical_vars:
        try:
            clinical = repo.clinical(variables=clinical_vars, wide=True)
            if clinical is not None and not clinical.empty:
                frames.append(clinical)
        except Exception:
            pass
    if cognitive_vars:
        try:
            cognitive = repo.cognitive(variables=cognitive_vars, wide=True)
            if cognitive is not None and not cognitive.empty:
                frames.append(cognitive)
        except Exception:
            pass

    covariates = pd.DataFrame()
    if frames:
        covariates = frames[0]
        for frame in frames[1:]:
            if "subject_uid" not in frame.columns or "subject_uid" not in covariates.columns:
                continue
            overlap = set(covariates.columns) & set(frame.columns) - {"subject_uid"}
            frame_use = frame.drop(
                columns=[c for c in overlap if c in frame.columns], errors="ignore"
            )
            covariates = covariates.merge(frame_use, on="subject_uid", how="outer")

    covariate_vars = list(dict.fromkeys([*clinical_vars, *cognitive_vars]))
    present = [c for c in covariate_vars if c in covariates.columns] if not covariates.empty else []

    if covariates.empty or not present:
        wide = aggregate_territory_measurements(long, [variable_id])
    else:
        wide = build_analysis_df_from_repo_frames(
            long,
            covariates,
            imaging_variable_ids=[variable_id],
            covariate_cols=present,
        )

    wide["group_key"] = wide["territory"].astype(str)

    # Friendly outcome aliases used in default formulas
    if variable_id in wide.columns:
        if variable_id == "flow_mean" and "flow" not in wide.columns:
            wide["flow"] = wide[variable_id]
        alias_inv = {v: k for k, v in FEATURE_ALIASES.items() if k != v}
        alias = alias_inv.get(variable_id)
        if alias and alias not in wide.columns:
            wide[alias] = wide[variable_id]

    if "patient_id" not in wide.columns and "subject_uid" in wide.columns:
        wide["patient_id"] = wide["subject_uid"]
    if "Hematocrit" not in wide.columns and "hematocrit" in wide.columns:
        wide = wide.rename(columns={"hematocrit": "Hematocrit"})
    if "age" in wide.columns and "age_c" not in wide.columns:
        age = pd.to_numeric(wide["age"], errors="coerce")
        wide["age_c"] = age - age.mean(skipna=True)
    elif "age_at_mri" in wide.columns and "age_c" not in wide.columns:
        age = pd.to_numeric(wide["age_at_mri"], errors="coerce")
        wide["age_c"] = age - age.mean(skipna=True)

    return wide


__all__ = [
    "ASL_FEATURES",
    "FEATURE_ALIASES",
    "FLAIR_FEATURES",
    "KIND_MODALITY",
    "QVTPY_TREE_FEATURES",
    "QVTPY_VESSEL_FEATURES",
    "T1_FEATURES",
    "TOF_FEATURES",
    "assign_group_key",
    "atlas_for_request",
    "build_long_analysis_frame",
    "feature_family",
    "features_for_kind",
    "grouping_choices_for",
    "region_to_hemisphere_pair_key",
    "resolve_feature_id",
]
