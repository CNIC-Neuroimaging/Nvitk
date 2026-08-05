"""Long-form image measurements → Statmodels analysis frames.

Uses ``DataRepo.image(..., wide=False)`` so region/variable columns stay
explicit (avoids single-variable wide pivots that drop ``_{variable}`` suffixes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy.common.morpho_db_publish import _SCALAR_VARS as _MORPHO_SCALAR_VARS

log = Logger()

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

# Subject-level attributes live on the ``subjects`` entity table rather than in
# ``clinical_measurements``, so they are not catalog "variables" and never show up in
# ``catalog.variable_entries(domain="clinical")``. They are offered as covariates alongside the
# clinical variables and merged on ``subject_uid``.
SUBJECT_TABLE = "subjects"
_SUBJECT_ATTRIBUTE_EXCLUDED = frozenset(
    {
        "subject_uid",
        "primary_patient_id",
        "primary_seqn",
        "notes",
        "source_batch_id",
        "source_file",
        "source_sheet",
        "updated_at",
    }
)

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
    # ``territory`` is offered on every vessel-level modality because it is the only grouping whose
    # keys are shared with ASL — it is what makes a joint 4D-flow + perfusion frame possible.
    if kind == "qvtpy" and family == "tree":
        return [
            ("By tree (L_ICA / R_ICA / Basilar)", "tree"),
            ("By hemisphere (avg L/R ICA; Basilar kept)", "hemisphere"),
            ("Vascular territory (shared with ASL)", "territory"),
        ]
    if kind == "qvtpy" and family == "vessel":
        return [
            ("Vessel-wise (region_id)", "vessel"),
            ("By hemisphere (avg L/R pairs)", "hemisphere"),
            ("Vascular territory (shared with ASL)", "territory"),
        ]
    if kind == "tof":
        return [
            ("Vessel-wise (eICAB vessel)", "vessel"),
            ("By hemisphere (avg L/R pairs)", "hemisphere"),
            ("Vascular territory (shared with ASL)", "territory"),
        ]
    if kind == "asl":
        return [
            ("Atlas region", "region"),
            ("By hemisphere (MCA / ACA / PCA / ICA / Basilar)", "hemisphere"),
            ("Vascular territory (shared with 4D flow)", "territory"),
        ]
    # t1 / flair: keep published regions
    return [("Region", "region")]


# Canonical spelling of every hemisphere key. Applied last in
# :func:`region_to_hemisphere_pair_key` so the same vessel reaches the same key whichever way it was
# published — ``LMCA``, ``LEFT_MCA`` and ASL's ``left_mca_8`` must all become ``MCA``, or a joint
# 4D-flow + perfusion frame silently fails to align.
_HEMISPHERE_CANONICAL: dict[str, str] = {
    "mca": "MCA",
    "aca": "ACA",
    "pca": "PCA",
    "ica": "ICA",
    "basilar": "Basilar",
    "basi": "Basilar",
    "ba": "Basilar",
    "va": "VA",
    "vertebral": "VA",
    "pcomm": "pcomm",
    "comm": "pcomm",
    "communicating": "pcomm",
    "posterior_communicating": "pcomm",
    "acomm": "acomm",
    "anterior_communicating": "acomm",
    "tsv": "TSV",
    "transverse": "TSV",
    "sssv": "SSSV",
    "strv": "STRV",
    "watershed": "Watershed",
}

# ASL region ids carry the smoothing kernel as a suffix (``left_mca_8``, ``watershed_0``).
_ASL_SMOOTHING_SUFFIX_RE = re.compile(r"_(0|8|12)$")


def _canonical_hemisphere_key(token: str) -> str:
    """Canonical spelling for a side-stripped vessel *token*, or the token unchanged."""
    return _HEMISPHERE_CANONICAL.get(str(token).strip().lower(), token)


def region_to_hemisphere_pair_key(region_id: str) -> str:
    """
    Collapse L/R counterparts to one key (ICA, MCA, …); midline stays itself.

    The result is canonicalized through :data:`_HEMISPHERE_CANONICAL`, so spellings that differ only
    in separator or case land on the same key. That is what lets a 4D-flow measurement grouped by
    hemisphere join an ASL one on ``(subject_uid, territory)``.
    """
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
            return _canonical_hemisphere_key(su[len(prefix) :])
    m = _SIDE_LETTER_RE.match(su)
    if m:
        token = m.group(2)
        canonical = _canonical_hemisphere_key(token)
        if canonical != token:
            return canonical
        return token.upper() if len(token) <= 4 else token.lower()
    stripped = _LR_PREFIX_RE.sub("", raw)
    return _canonical_hemisphere_key(stripped) if stripped else raw


def asl_region_id_to_hemisphere(region_id: str) -> str:
    """
    Collapse an ASL atlas region to the same vessel key the 4D-flow hemisphere grouping produces.

    ASL vascular parcels are published per side and per smoothing kernel (``left_mca_8``,
    ``right_pca_8``, ``watershed_8``). Stripping the kernel suffix and the side prefix yields
    ``MCA`` / ``ACA`` / ``PCA`` / ``ICA`` / ``Basilar`` — exactly the keys
    :func:`region_to_hemisphere_pair_key` gives for ``LMCA`` / ``RMCA`` / …, so the two modalities
    can be joined vessel by vessel rather than only at the coarse territory level.

    Watershed parcels have no 4D-flow counterpart and keep their own ``Watershed`` key; Desikan
    parcels fall back to a side-stripped parcel name.
    """
    raw = str(region_id).strip()
    if not raw:
        return raw
    token = _ASL_SMOOTHING_SUFFIX_RE.sub("", raw)
    if "watershed" in token.lower():
        return "Watershed"
    return region_to_hemisphere_pair_key(token)


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
        if str(kind).strip().lower() == "asl":
            return asl_region_id_to_hemisphere(rid)
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


def clinical_measurement_variable_ids(repo: Any) -> set[str]:
    """``variable_id``s that currently have at least one row in ``clinical_measurements``.

    Used to prefer the measurements table over the sparse ``subjects`` entity columns when both
    expose the same name (e.g. ``sex`` after ``import_sex.py``). Catalog registration alone is not
    enough: a Statmodels window opened before the import keeps a stale catalog and would otherwise
    keep routing ``sex`` through ``subjects``.
    """
    try:
        if not repo.catalog.table_exists("clinical_measurements"):
            return set()
        frame = repo.get("clinical_measurements", columns=["variable_id"], cohort_id=False)
    except Exception:
        return set()
    if frame is None or frame.empty or "variable_id" not in frame.columns:
        return set()
    return {str(v) for v in frame["variable_id"].dropna().unique()}


def subject_attribute_entries(repo: Any) -> list[dict[str, Any]]:
    """
    Subject-level covariates (``sex``, …) available on the ``subjects`` table, in the same
    ``{"variable_id", "label", "domain"}`` shape as :meth:`Catalog.variable_entries`.

    Bookkeeping / identifier / timestamp columns are excluded, as are names already available as
    clinical measurements (catalog entry **or** rows in ``clinical_measurements``) — the measurement
    table always wins over the entity column.
    """
    try:
        if not repo.catalog.table_exists(SUBJECT_TABLE):
            return []
        columns = dict(repo.catalog.get_table(SUBJECT_TABLE).columns or {})
    except Exception:
        return []

    try:
        registered = {
            str(e.get("variable_id")) for e in repo.catalog.variable_entries(domain="clinical")
        }
    except Exception:
        registered = set()
    registered |= clinical_measurement_variable_ids(repo)

    entries: list[dict[str, Any]] = []
    for name, dtype in columns.items():
        col = str(name)
        if col in _SUBJECT_ATTRIBUTE_EXCLUDED or col in registered:
            continue
        if "datetime" in str(dtype).lower():
            continue
        entries.append(
            {
                "variable_id": col,
                "label": f"{col} (subject)",
                "domain": "clinical",
                "table": SUBJECT_TABLE,
            }
        )
    return entries


def subject_attribute_columns(repo: Any) -> list[str]:
    """``variable_id``s of the subject-level covariates returned by :func:`subject_attribute_entries`."""
    return [str(e["variable_id"]) for e in subject_attribute_entries(repo)]


def _subject_attribute_frame(repo: Any, names: list[str]) -> pd.DataFrame:
    """One row per ``subject_uid`` with the requested ``subjects``-table columns."""
    if not names:
        return pd.DataFrame()
    try:
        frame = repo.get(SUBJECT_TABLE, columns=["subject_uid", *names])
    except Exception:
        return pd.DataFrame()
    if frame is None or frame.empty or "subject_uid" not in frame.columns:
        return pd.DataFrame()
    keep = ["subject_uid", *[c for c in names if c in frame.columns]]
    return frame[keep].drop_duplicates(subset=["subject_uid"], keep="first")


# ──────────────────────────────────────────────────────────────────────────────
# Composite territory labels
# ──────────────────────────────────────────────────────────────────────────────
COMPOSITE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "TCBF": {
        "label": "TCBF — total cerebral blood flow (RICA + LICA + BASILAR)",
        "description": (
            "Total cerebral blood flow: the summed mean flow of both internal carotid arteries and "
            "the basilar artery, i.e. everything entering the cranium."
        ),
        # Matched case-insensitively against ``region_id``; synonyms cover the published spellings.
        "regions": (
            ("rica", "right_ica", "r_ica"),
            ("lica", "left_ica", "l_ica"),
            ("basilar", "basi", "ba"),
        ),
        "agg": "sum",
        "kinds": ("qvtpy",),
        "features": ("flow_mean",),
    },
}


def composites_for(kind: str, variable_id: str) -> list[tuple[str, str]]:
    """``(name, label)`` composite territories available for a pipeline kind / feature."""
    vid = resolve_feature_id(variable_id)
    kind = str(kind).strip().lower()
    return [
        (name, str(spec["label"]))
        for name, spec in COMPOSITE_DEFINITIONS.items()
        if kind in spec["kinds"] and vid in spec["features"]
    ]


def _composite_rows(long: pd.DataFrame, name: str, *, value_column: str = "value") -> pd.DataFrame:
    """
    Build the long-form rows for one composite territory.

    Each component is resolved per subject and the components are combined with the definition's
    aggregation. A subject missing **any** component is skipped rather than summed over what it has:
    a partial TCBF is not a smaller TCBF, it is a wrong one, and silently emitting it would bias
    every model that uses the label.

    Returns
    -------
    pandas.DataFrame
        Long-form rows (``subject_uid``, ``region_id``, ``variable_id``, ``value``, …) carrying the
        composite as their ``region_id``, or an empty frame when no subject has every component.
    """
    definition = COMPOSITE_DEFINITIONS[name]
    components: tuple[tuple[str, ...], ...] = definition["regions"]

    keys = long["region_id"].astype(str).str.strip().str.lower()
    per_component: list[pd.Series] = []
    for synonyms in components:
        wanted = {s.lower() for s in synonyms}
        part = long.loc[keys.isin(wanted)]
        if part.empty:
            log.warning(
                "Composite %s: no rows for component %s — the label cannot be built.",
                name,
                synonyms[0],
            )
            return long.iloc[0:0]
        # One value per subject; a duplicated vessel (repeat scan) is averaged first.
        per_component.append(
            pd.to_numeric(part[value_column], errors="coerce")
            .groupby(part["subject_uid"])
            .mean()
        )

    stacked = pd.concat(per_component, axis=1)
    complete = stacked.dropna()
    n_partial = len(stacked) - len(complete)
    if n_partial:
        log.warning(
            "Composite %s: skipped %d of %d subjects missing at least one component "
            "(a partial sum would understate the total).",
            name,
            n_partial,
            len(stacked),
        )
    if complete.empty:
        return long.iloc[0:0]

    values = complete.sum(axis=1) if definition["agg"] == "sum" else complete.mean(axis=1)
    rows = pd.DataFrame(
        {
            "subject_uid": complete.index,
            "region_id": name,
            "variable_id": long["variable_id"].iloc[0],
            value_column: values.to_numpy(),
        }
    )
    log.info("Composite %s: built for %d subjects.", name, len(rows))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Analysis-frame building blocks
# ──────────────────────────────────────────────────────────────────────────────
def build_feature_territory_frame(
    repo: Any,
    *,
    pipeline_kind: str,
    pipeline: str,
    feature: str,
    grouping: str,
    atlas: str | None = None,
    agg: str = "mean",
    column: str | None = None,
    composites: Sequence[str] = (),
) -> pd.DataFrame:
    """
    One image measurement, aggregated to one row per ``(subject_uid, territory)``.

    This is the covariate-free half of :func:`build_long_analysis_frame`: it loads the long-form
    rows for a single ``variable_id``, assigns the ``territory`` group key implied by *grouping*, and
    aggregates within each ``(subject_uid, territory)`` cell. Keeping it separate is what lets
    several measurements from *different* pipelines be joined on a shared key before any covariate
    merge happens — attaching covariates per measurement would compound their inner merge.

    Parameters
    ----------
    pipeline_kind : {"qvtpy", "asl", "t1", "flair", "tof"}
        Selects the imaging modality via :data:`KIND_MODALITY`.
    pipeline : str
        Pipeline id, or one of ``latest``/``default``/``current``/``last`` for the catalog default.
    feature : str
        Feature name or alias; resolved through :func:`resolve_feature_id`.
    grouping : GroupingMode
        How ``region_id`` collapses into ``territory`` (see :func:`assign_group_key`).
    agg : str
        Aggregation applied within each ``(subject_uid, territory)`` cell.
    column : str, optional
        Output column name. Defaults to the resolved ``variable_id``.
    composites : sequence of str
        Names from :data:`COMPOSITE_DEFINITIONS` to append as extra territory labels, e.g.
        ``("TCBF",)``. They are computed from the raw ``region_id`` rows, so they are unaffected by
        *grouping*, and appear as additional levels of ``territory``.

    Returns
    -------
    pandas.DataFrame
        Columns ``subject_uid``, ``territory``, and the measurement column.
        ``frame.attrs["region_ids"]`` lists the source region ids, which lets callers work out
        whether a different grouping would produce keys compatible with another measurement.
    """
    from nvitk.stats._hemodynamic_frames import aggregate_territory_measurements

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

    long["region_id"] = long["region_id"].astype(str)
    source_region_ids = sorted(long["region_id"].unique())

    # ---- Composite labels are summed over raw vessels, so they must be built before grouping ----
    for name in composites:
        if name not in COMPOSITE_DEFINITIONS:
            log.warning("Unknown composite territory %r — skipped.", name)
            continue
        extra = _composite_rows(long, name)
        if not extra.empty:
            long = pd.concat([long, extra], ignore_index=True)

    long["territory"] = [
        assign_group_key(
            rid, grouping=grouping, kind=kind, variable_id=variable_id
        )
        for rid in long["region_id"].astype(str)
    ]
    # A composite is its own label whatever the grouping — it is already an aggregate, so collapsing
    # it into a territory (or leaving it "Unmapped") would defeat the point of asking for it.
    composite_names = {n for n in composites if n in COMPOSITE_DEFINITIONS}
    if composite_names:
        is_composite = long["region_id"].isin(composite_names)
        long.loc[is_composite, "territory"] = long.loc[is_composite, "region_id"]
    long["modality_group"] = kind

    wide = aggregate_territory_measurements(long, [variable_id], agg=agg)
    out_col = str(column or variable_id)
    if out_col != variable_id and variable_id in wide.columns:
        wide = wide.rename(columns={variable_id: out_col})
    wide.attrs["region_ids"] = source_region_ids
    return wide


def collapse_visits_to_subject(
    frame: pd.DataFrame,
    *,
    policy: str = "latest",
    subject_key: str = "subject_uid",
    visit_key: str = "visit_id",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Collapse a per-visit measurement frame to one row per subject, combining across visits.

    ``clinical_measurements`` pivots on ``(subject_uid, visit_id)``, so a cohort whose variables were
    collected at different visits gets one row per visit with the *other* visits' variables blank.
    Anything downstream that assumes one row per subject then keeps a single visit and silently
    discards the rest — which is how requesting carotid plaque (visit 3) alongside the rest of the
    clinical panel (visit 4) can null out the panel.

    Each column is filled from the first non-missing value in visit order, so variables measured at
    different visits combine instead of competing.

    Parameters
    ----------
    policy : {"latest", "earliest"}
        Which visit wins for a variable recorded at more than one. ``latest`` takes the most recent.

    Returns
    -------
    (collapsed, provenance)
        *provenance* maps each column to the visits that contributed a value, so a caller can tell
        the user which variables came from where.
    """
    if frame.empty or subject_key not in frame.columns or visit_key not in frame.columns:
        return frame, {}
    if not frame[subject_key].duplicated().any():
        return frame.drop(columns=[visit_key], errors="ignore"), {}

    value_columns = [c for c in frame.columns if c not in {subject_key, visit_key}]
    provenance = {
        column: sorted(
            str(v) for v in frame.loc[frame[column].notna(), visit_key].dropna().unique()
        )
        for column in value_columns
    }

    ordered = frame.sort_values([subject_key, visit_key], kind="stable")
    grouped = ordered.groupby(subject_key, sort=False)[value_columns]
    # ``first``/``last`` skip missing values, which is exactly the "combine across visits" rule.
    collapsed = (grouped.last() if policy == "latest" else grouped.first()).reset_index()

    log.info(
        "Collapsed %d covariate rows across %s to %d subjects (%s visit wins per variable).",
        len(frame),
        visit_key,
        len(collapsed),
        policy,
    )
    return collapsed, provenance


def resolve_covariate_frame(
    repo: Any,
    *,
    clinical_vars: list[str] | None = None,
    cognitive_vars: list[str] | None = None,
    visit_policy: str = "latest",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Subject-level covariate frame assembled from the clinical / subjects / cognitive tables.

    Each source is collapsed to one row per subject (see :func:`collapse_visits_to_subject`) *before*
    the sources are merged, so variables recorded at different visits combine rather than
    multiplying rows or shadowing one another.

    Returns
    -------
    (covariates, present)
        *covariates* is one row per ``subject_uid`` (outer-merged across sources, duplicated columns
        dropped from the right frame); *present* lists the requested names that actually resolved.
        ``covariates.attrs["visit_provenance"]`` maps each column to the visits it came from.
        A source that raises is skipped silently — a missing cognitive table must not block a model
        that only needs clinical covariates.
    """
    clinical_vars = list(clinical_vars or [])
    cognitive_vars = list(cognitive_vars or [])

    # Covariates. Prefer ``clinical_measurements`` whenever the requested name has rows there
    # (covers a stale in-memory catalog after ``import_sex.py``). Only fall back to the
    # ``subjects`` entity table for names that exist solely as subject attributes.
    clinical_present = clinical_measurement_variable_ids(repo)
    subject_attrs = set(subject_attribute_columns(repo)) - clinical_present
    subject_vars = [v for v in clinical_vars if v in subject_attrs and v not in clinical_present]
    clinical_vars = [v for v in clinical_vars if v not in subject_vars]

    frames: list[pd.DataFrame] = []
    provenance: dict[str, list[str]] = {}

    def add(frame: pd.DataFrame | None) -> None:
        """Collapse a source to one row per subject, then queue it for merging."""
        if frame is None or frame.empty:
            return
        collapsed, source_provenance = collapse_visits_to_subject(frame, policy=visit_policy)
        provenance.update(source_provenance)
        frames.append(collapsed)

    if clinical_vars:
        try:
            add(repo.clinical(variables=clinical_vars, wide=True))
        except Exception as exc:
            log.debug("Clinical covariates unavailable: %s", exc)
    if subject_vars:
        add(_subject_attribute_frame(repo, subject_vars))
    if cognitive_vars:
        try:
            add(repo.cognitive(variables=cognitive_vars, wide=True))
        except Exception as exc:
            log.debug("Cognitive covariates unavailable: %s", exc)

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

    covariate_vars = list(dict.fromkeys([*clinical_vars, *subject_vars, *cognitive_vars]))
    present = [c for c in covariate_vars if c in covariates.columns] if not covariates.empty else []
    covariates.attrs["visit_provenance"] = {k: v for k, v in provenance.items() if k in present}
    return covariates, present


def finalize_analysis_frame(
    wide: pd.DataFrame,
    repo: Any,
    *,
    primary_variable_id: str | None = None,
) -> pd.DataFrame:
    """
    Add the conveniences every Statmodels frame is expected to carry.

    In order: ``group_key`` (a copy of ``territory``, used as the default MixedLM grouping), the
    friendly outcome aliases the default formulas reference (``flow`` for ``flow_mean``, and the
    inverse of :data:`FEATURE_ALIASES`), ``patient_id``, the ``hematocrit`` → ``Hematocrit`` rename,
    numeric coercion of catalog-numeric covariates, and the mean-centered ``age_c``.

    Parameters
    ----------
    primary_variable_id : str, optional
        Measurement to alias. With several measurements loaded, only the first one gets aliases —
        the aliases exist for the legacy single-feature default formulas.
    """
    wide["group_key"] = wide["territory"].astype(str)

    # Friendly outcome aliases used in default formulas
    variable_id = primary_variable_id
    if variable_id and variable_id in wide.columns:
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

    # Belt-and-suspenders: numeric clinical/cognitive covariates can arrive as object dtype when the
    # wide pivot mixed them with text variables. Coerce them here so the preview table and formulas
    # see floats (Patsy otherwise expands Hematocrit into Hematocrit[T.36.0], …).
    try:
        numeric_ids = {
            str(e.get("variable_id"))
            for e in repo.catalog.variable_entries(domain="clinical")
            + repo.catalog.variable_entries(domain="cognitive")
            if str(e.get("value_kind") or "").strip().lower() in {"numeric", "float", "int", "integer", "number"}
        }
    except Exception:
        numeric_ids = set()
    numeric_ids.update({"Hematocrit", "hematocrit", "age", "age_at_mri", "age_c", "sex", "bmi", "weight", "height"})
    for col in list(wide.columns):
        if col in numeric_ids or col.lower() in {v.lower() for v in numeric_ids}:
            if not pd.api.types.is_numeric_dtype(wide[col]):
                wide[col] = pd.to_numeric(wide[col], errors="coerce")

    if "age" in wide.columns and "age_c" not in wide.columns:
        age = pd.to_numeric(wide["age"], errors="coerce")
        wide["age_c"] = age - age.mean(skipna=True)
    elif "age_at_mri" in wide.columns and "age_c" not in wide.columns:
        age = pd.to_numeric(wide["age_at_mri"], errors="coerce")
        wide["age_c"] = age - age.mean(skipna=True)

    return wide


# ──────────────────────────────────────────────────────────────────────────────
# Measurement specs and multi-measurement frames
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MeasurementSpec:
    """
    One image measurement to load into an analysis frame.

    A spec is self-contained: it names its own pipeline kind, pipeline, feature, atlas and grouping,
    so measurements from different modalities can be combined as long as their *grouping* produces
    commensurate ``territory`` keys (e.g. qvtpy ``hemisphere`` → ``MCA``/``ACA``/``PCA`` against ASL
    ``territory`` → the same vascular-territory labels).

    Parameters
    ----------
    alias : str, optional
        Output column name. Defaults to ``resolve_feature_id(feature)``. Must be a valid Python
        identifier: :func:`~nvitk.stats.mixedlm.fit_or_load_mixedlm` sanitizes column names, so a
        non-identifier alias would silently diverge between the analysis frame and the model frame.
    """

    pipeline_kind: str = "qvtpy"
    pipeline: str = "latest"
    feature: str = "flow_mean"
    grouping: str = "vessel"
    atlas: str | None = None
    alias: str | None = None
    agg: str = "mean"
    composites: tuple[str, ...] = ()

    def column(self) -> str:
        """Name this measurement takes in the analysis frame."""
        return str(self.alias or resolve_feature_id(self.feature))

    def label(self) -> str:
        """One-line description for list widgets, e.g. ``"qvtpy · latest · pi · hemisphere → pi"``."""
        parts = [self.pipeline_kind, self.pipeline, self.feature, self.grouping]
        if self.atlas:
            parts.append(self.atlas)
        if self.composites:
            parts.append("+" + "+".join(self.composites))
        return f"{' · '.join(str(p) for p in parts)}  →  {self.column()}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "pipeline_kind": self.pipeline_kind,
            "pipeline": self.pipeline,
            "feature": self.feature,
            "grouping": self.grouping,
            "atlas": self.atlas,
            "alias": self.alias,
            "agg": self.agg,
            "composites": list(self.composites),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeasurementSpec":
        """Rebuild from :meth:`to_dict` output, tolerating missing keys."""
        return cls(
            pipeline_kind=str(data.get("pipeline_kind") or "qvtpy"),
            pipeline=str(data.get("pipeline") or "latest"),
            feature=str(data.get("feature") or "flow_mean"),
            grouping=str(data.get("grouping") or "vessel"),
            atlas=(str(data["atlas"]) if data.get("atlas") else None),
            alias=(str(data["alias"]) if data.get("alias") else None),
            agg=str(data.get("agg") or "mean"),
            composites=tuple(str(c) for c in (data.get("composites") or ())),
        )


def _keys_for_grouping(spec: MeasurementSpec, region_ids: Sequence[str], grouping: str) -> set[str]:
    """Territory keys *spec*'s region ids would produce under *grouping*."""
    variable_id = resolve_feature_id(spec.feature)
    return {
        assign_group_key(rid, grouping=grouping, kind=spec.pipeline_kind, variable_id=variable_id)
        for rid in region_ids
    }


def _suggest_compatible_groupings(
    first: MeasurementSpec,
    first_regions: Sequence[str],
    second: MeasurementSpec,
    second_regions: Sequence[str],
) -> str:
    """
    Name a grouping combination that would actually make two measurements joinable.

    Rather than guessing, this replays :func:`assign_group_key` over both measurements' real region
    ids for every grouping the UI offers, and reports the pairing with the largest key overlap. That
    matters because the intuitive advice is often wrong: qvtpy ``hemisphere`` yields ``MCA``/``ACA``
    while ASL ``territory`` yields ``Anterior Circulation`` — only ``territory`` on *both* sides
    lines up.
    """
    if not first_regions or not second_regions:
        return ""

    first_options = [key for _label, key in grouping_choices_for(first.pipeline_kind, first.feature)]
    second_options = [key for _label, key in grouping_choices_for(second.pipeline_kind, second.feature)]

    best: tuple[int, str, str] | None = None
    for g1 in first_options:
        keys1 = _keys_for_grouping(first, first_regions, g1) - {"Unmapped"}
        for g2 in second_options:
            shared = keys1 & (_keys_for_grouping(second, second_regions, g2) - {"Unmapped"})
            if shared and (best is None or len(shared) > best[0]):
                best = (len(shared), g1, g2)
    if best is None:
        return ""

    _n, g1, g2 = best
    if g1 == first.grouping and g2 == second.grouping:
        return ""
    return (
        f"Set {first.column()!r} to grouping {g1!r} and {second.column()!r} to {g2!r} — "
        f"that pairing shares {_n} key(s)."
    )


def _coerce_specs(measurements: Sequence[MeasurementSpec | Mapping[str, Any]]) -> list[MeasurementSpec]:
    """Accept either :class:`MeasurementSpec` instances or plain dicts (as saved in configs)."""
    out: list[MeasurementSpec] = []
    for entry in measurements:
        out.append(entry if isinstance(entry, MeasurementSpec) else MeasurementSpec.from_dict(entry))
    return out


def build_multi_feature_analysis_frame(
    repo: Any,
    *,
    measurements: Sequence[MeasurementSpec | Mapping[str, Any]],
    clinical_vars: list[str] | None = None,
    cognitive_vars: list[str] | None = None,
    join: str = "inner",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Analysis frame combining one or more image measurements plus subject covariates.

    Each measurement is loaded independently by :func:`build_feature_territory_frame` and the
    results are joined on ``(subject_uid, territory)``. With ``join="inner"`` (the default) only
    subject/territory cells present in *every* measurement survive, which is the right grain for a
    model like ``att_mean ~ pi + …``: every row then has all its predictors observed.

    Covariates are merged **once**, after the measurement join. The covariate merge is an inner
    merge on ``subject_uid``, so attaching it per measurement would apply it repeatedly.

    Parameters
    ----------
    join : {"inner", "outer", "left"}
        How successive measurement frames are combined.

    Returns
    -------
    (frame, meta)
        *meta* carries ``measurements`` (per-spec row/subject/territory counts), ``key_overlap``
        (fraction of measurement 0's keys retained after each join), ``n_rows``, ``covariates``, and
        ``warnings`` — a list of human-readable problems worth surfacing in a UI, most importantly a
        disjoint ``territory`` vocabulary between two pipelines.

    Raises
    ------
    ValueError
        If *measurements* is empty or two specs would write the same output column.
    """
    from nvitk.stats._hemodynamic_frames import merge_subject_covariates

    specs = _coerce_specs(measurements)
    if not specs:
        raise ValueError("At least one measurement is required.")

    # ---- 1. Reject colliding output columns before doing any I/O ---------------
    seen: dict[str, int] = {}
    for i, spec in enumerate(specs):
        col = spec.column()
        if col in seen:
            raise ValueError(
                f"Measurements {seen[col] + 1} and {i + 1} both produce column {col!r}. "
                "Give one of them a distinct alias."
            )
        seen[col] = i

    meta: dict[str, Any] = {"measurements": [], "warnings": [], "join": join}

    # ---- 2. Load each measurement on its own grouping --------------------------
    frames: list[pd.DataFrame] = []
    for spec in specs:
        frame = build_feature_territory_frame(
            repo,
            pipeline_kind=spec.pipeline_kind,
            pipeline=spec.pipeline,
            feature=spec.feature,
            grouping=spec.grouping,
            atlas=spec.atlas,
            agg=spec.agg,
            column=spec.column(),
            composites=spec.composites,
        )
        frames.append(frame)
        meta["measurements"].append(
            {
                "spec": spec,
                "column": spec.column(),
                "n_rows": int(len(frame)),
                "n_subjects": int(frame["subject_uid"].nunique()) if len(frame) else 0,
                "territories": sorted(frame["territory"].astype(str).unique()) if len(frame) else [],
                "region_ids": list(frame.attrs.get("region_ids") or []),
            }
        )

    # ---- 3. Join on the shared (subject_uid, territory) key --------------------
    wide = frames[0]
    base_keys = set(map(tuple, wide[["subject_uid", "territory"]].astype(str).to_numpy())) if len(wide) else set()
    for spec, frame in zip(specs[1:], frames[1:]):
        base_terr = set(meta["measurements"][0]["territories"])
        this_terr = set(frame["territory"].astype(str).unique()) if len(frame) else set()
        if base_terr and this_terr and not (base_terr & this_terr):
            fix = _suggest_compatible_groupings(
                specs[0], meta["measurements"][0].get("region_ids") or [],
                spec, list(frame.attrs.get("region_ids") or []),
            )
            meta["warnings"].append(
                f"{spec.column()!r} has no territory in common with "
                f"{specs[0].column()!r} ({sorted(this_terr)[:4]} vs {sorted(base_terr)[:4]}). "
                + (fix or "No grouping combination makes these two measurements comparable.")
            )
        wide = wide.merge(frame, on=["subject_uid", "territory"], how=join)

    if base_keys:
        kept = set(map(tuple, wide[["subject_uid", "territory"]].astype(str).to_numpy())) if len(wide) else set()
        meta["key_overlap"] = len(kept & base_keys) / len(base_keys)
    else:
        meta["key_overlap"] = 0.0

    if wide.empty:
        meta["warnings"].append(
            "The measurement join produced no rows — the selected measurements share no "
            "(subject, territory) cell."
        )

    # ---- 4. Covariates once, then the shared finalization ----------------------
    covariates, present = resolve_covariate_frame(
        repo, clinical_vars=clinical_vars, cognitive_vars=cognitive_vars
    )
    if present and not covariates.empty and not wide.empty:
        before = len(wide)
        wide = merge_subject_covariates(wide, covariates, present)
        if len(wide) < before:
            meta["warnings"].append(
                f"Covariate merge dropped {before - len(wide)} of {before} rows "
                "(subjects missing from the covariate tables)."
            )
    meta["covariates"] = present

    # Mixing visits is a legitimate thing to want (plaque at one visit, cognition at another) but it
    # is a modeling decision, so say so rather than letting it happen quietly.
    provenance = dict(covariates.attrs.get("visit_provenance") or {})
    meta["visit_provenance"] = provenance
    visits_used = {v for visits in provenance.values() for v in visits}
    if len(visits_used) > 1:
        by_visit: dict[str, list[str]] = {}
        for column, visits in sorted(provenance.items()):
            by_visit.setdefault(" + ".join(visits), []).append(column)
        detail = "; ".join(f"{visit}: {', '.join(cols)}" for visit, cols in sorted(by_visit.items()))
        meta["warnings"].append(
            f"Covariates come from more than one visit and were combined per subject ({detail}). "
            "Each variable keeps its own visit's value."
        )

    wide = finalize_analysis_frame(wide, repo, primary_variable_id=specs[0].column())
    meta["n_rows"] = int(len(wide))
    for message in meta["warnings"]:
        log.warning("Analysis frame: %s", message)
    return wide, meta


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
    """Load long-form image rows, assign group keys, aggregate, attach covariates.

    Single-measurement convenience wrapper over :func:`build_multi_feature_analysis_frame`.
    """
    frame, _meta = build_multi_feature_analysis_frame(
        repo,
        measurements=[
            MeasurementSpec(
                pipeline_kind=pipeline_kind,
                pipeline=pipeline,
                feature=feature,
                grouping=grouping,
                atlas=atlas,
            )
        ],
        clinical_vars=clinical_vars,
        cognitive_vars=cognitive_vars,
    )
    return frame


__all__ = [
    "ASL_FEATURES",
    "FEATURE_ALIASES",
    "FLAIR_FEATURES",
    "KIND_MODALITY",
    "QVTPY_TREE_FEATURES",
    "QVTPY_VESSEL_FEATURES",
    "T1_FEATURES",
    "TOF_FEATURES",
    "MeasurementSpec",
    "assign_group_key",
    "atlas_for_request",
    "build_feature_territory_frame",
    "build_long_analysis_frame",
    "build_multi_feature_analysis_frame",
    "collapse_visits_to_subject",
    "feature_family",
    "finalize_analysis_frame",
    "resolve_covariate_frame",
    "features_for_kind",
    "grouping_choices_for",
    "asl_region_id_to_hemisphere",
    "composites_for",
    "region_to_hemisphere_pair_key",
    "resolve_feature_id",
    "subject_attribute_columns",
    "subject_attribute_entries",
    "clinical_measurement_variable_ids",
]
