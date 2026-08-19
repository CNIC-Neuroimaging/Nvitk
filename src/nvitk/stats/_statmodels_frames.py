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
QVTPY_VESSEL_FEATURES: tuple[str, ...] = ("flow_mean", "pi", "ri", "psf", "flow_tseries")

#: Variables stored one row per cardiac frame rather than one row per vessel. They load as a *set*
#: of columns — ``flow_tseries``, ``flow_tseries_f1``, … — because the waveform is the measurement;
#: averaging it back to a scalar just recomputes ``flow_mean``.
TIMESERIES_FEATURES: frozenset[str] = frozenset({"flow_tseries"})


def is_timeseries_feature(variable_id: str) -> bool:
    """Whether *variable_id* is stored per cardiac frame (see :data:`TIMESERIES_FEATURES`)."""
    return resolve_feature_id(variable_id) in TIMESERIES_FEATURES
# Tree-level hemodynamics (per arterial root: L_ICA / R_ICA / Basilar).
QVTPY_TREE_FEATURES: tuple[str, ...] = (
    "pwv",
    "pwv_fielding_xcor",
    "pitc_slope",
    "pitc_intercept",
)
ASL_FEATURES: tuple[str, ...] = ("mean_cbf", "cov_cbf", "att_mean", "att_cov")
#: FreeSurfer measurements, one variable per quantity. Ordered by how often they are modelled:
#: grey-matter volume and cortical thickness first, then the rest of the surface morphometry, then
#: the subcortical intensity statistics. ``t1_cortical_volume`` / ``t1_subcortical_volume`` are the
#: pre-``import_t1_regions`` variables, kept last so saved configs that name them still resolve.
T1_FEATURES: tuple[str, ...] = (
    # ---- cortical parcels (Desikan-Killiany, lateralized) ---------------------
    "t1_gray_volume",
    "t1_thickness_avg",
    "t1_thickness_std",
    "t1_surface_area",
    "t1_num_vertices",
    "t1_mean_curvature",
    "t1_gaussian_curvature",
    "t1_folding_index",
    "t1_curvature_index",
    # ---- subcortical structures and whole-brain scalars -----------------------
    # eTIV is *not* a variable of its own: it is one region of ``t1_volume_mm3``, alongside
    # BrainSegVol, CortexVol and the rest. Reach it with the measurement's region filter
    # (``regions=("etiv",)``) rather than by publishing a parallel variable that would have to be
    # kept in step with this one.
    "t1_volume_mm3",
    "t1_num_voxels",
    "t1_intensity_mean",
    "t1_intensity_std",
    "t1_intensity_min",
    "t1_intensity_max",
    "t1_intensity_range",
    "t1_index_unitless",
    # ---- superseded ------------------------------------------------------------
    "t1_cortical_volume",
    "t1_subcortical_volume",
)
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
        # ``lobe`` is offered for the Desikan atlas and is a real reduction in multiplicity
        # (68 parcels → 8 lobes). The pipeline publishes its own lobe rows, so this grouping uses
        # those rather than re-averaging parcels — see ``_select_region_granularity``.
        return [
            ("Atlas region", "region"),
            ("By lobe (Desikan; uses the published lobe rows)", "lobe"),
            ("By hemisphere (MCA / ACA / PCA / ICA / Basilar)", "hemisphere"),
            ("Vascular territory (shared with 4D flow)", "territory"),
        ]
    if kind == "t1":
        # Cortical parcels are lateralized and subcortical structures mostly are too, so both a
        # left/right pairing and a lobe pooling are meaningful — and each is a real reduction in
        # multiplicity: 68 parcels → 34 pairs → 6 lobes.
        return [
            ("Region (parcel / structure)", "region"),
            ("By hemisphere (avg L/R pairs)", "hemisphere"),
            ("By lobe (Desikan / aseg panel)", "lobe"),
        ]
    # flair: keep published regions
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


def _region_match_key(region_id: Any) -> str:
    """Spelling-insensitive key for matching a region filter against published ``region_id``s."""
    return re.sub(r"[^0-9a-z]+", "", str(region_id).strip().lower())


def available_region_ids(
    repo: Any,
    *,
    pipeline_kind: str,
    feature: str,
    pipeline: str = "latest",
) -> list[str]:
    """
    Published ``region_id``s for one measurement, for populating a region picker.

    Reads ``image_measurements`` directly rather than through :meth:`DataRepo.image` so the list is
    what the table actually holds — a catalog default pipeline or a cohort restriction would
    otherwise hide regions the user can legitimately select. Never raises: an unreadable table just
    yields an empty list, and the picker falls back to free text.
    """
    variable_id = resolve_feature_id(feature)
    try:
        frame = repo.get("image_measurements", cohort_id=False)
    except Exception as exc:
        log.debug("Could not list regions for %s: %s", variable_id, exc)
        return []
    if frame is None or frame.empty or "variable_id" not in frame.columns:
        return []

    rows = frame.loc[frame["variable_id"].astype(str) == variable_id]
    wanted = str(pipeline).strip().lower()
    if wanted not in {"", "latest", "any", "default", "current", "last"} and "pipeline_id" in rows.columns:
        selected = rows.loc[rows["pipeline_id"].astype(str) == str(pipeline)]
        if not selected.empty:
            rows = selected
    if rows.empty or "region_id" not in rows.columns:
        return []
    return sorted({str(r) for r in rows["region_id"].dropna().unique()})


def t1_region_to_hemisphere_pair_key(region_id: str) -> str:
    """
    Drop the hemisphere prefix from a FreeSurfer region so L/R counterparts share a key.

    ``left_precuneus`` and ``right_precuneus`` both become ``precuneus``; midline structures
    (``brain_stem``, ``csf``) and the whole-head scalars (``etiv``) are returned unchanged.

    Kept separate from :func:`region_to_hemisphere_pair_key` because that one is built for vessels:
    it canonicalizes through the vessel table and upper-cases short names, which would turn
    ``left_bankssts`` into ``BANKSSTS`` and leave the hemisphere grouping spelled unlike every other
    T1 grouping. Parcel names have no canonical short form, so stripping the side is all there is
    to do.

    Examples
    --------
    >>> t1_region_to_hemisphere_pair_key("left_precuneus")
    'precuneus'
    >>> t1_region_to_hemisphere_pair_key("brain_stem")
    'brain_stem'
    """
    raw = str(region_id).strip()
    if not raw:
        return raw
    lowered = raw.lower()
    for prefix in ("left_", "right_", "lh_", "rh_", "ctx_lh_", "ctx_rh_", "l_", "r_"):
        if lowered.startswith(prefix) and len(lowered) > len(prefix):
            return lowered[len(prefix):]
    for suffix in ("_left", "_right", "_lh", "_rh"):
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            return lowered[: -len(suffix)]
    return lowered


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
        if str(kind).strip().lower() == "t1":
            return t1_region_to_hemisphere_pair_key(rid)
        return region_to_hemisphere_pair_key(rid)
    if grouping == "lobe":
        # Side-qualified: a lobe grouping that averages the two hemispheres together cannot answer
        # a lateralized question, and asymmetry is most of what a lobe-level analysis looks for.
        from nvitk.stats.region_groups import region_lobe_key

        return region_lobe_key(rid)
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


def subject_image_annotation_entries(repo: Any) -> list[dict[str, Any]]:
    """
    Image variables holding **one value per subject** rather than one per vessel.

    These are the manual QC annotations — ``cow_config`` / ``venous_config``, see
    :mod:`nvitk.db.qvtpy_anatomy` — which describe the subject's anatomy as a whole and so carry no
    ``region_id``. They cannot be measurements in an analysis frame (there is no region to join on),
    but they are exactly what a model wants as a subject-level factor, so they are offered
    alongside the clinical covariates.

    Recognized by ``scope == "subject"`` on the catalog entry, plus the known annotation ids —
    a dataset whose catalog was written by an older run may carry the rows without the flag.
    """
    try:
        from nvitk.db.qvtpy_anatomy import ANATOMY_CONFIG_VARIABLES
    except Exception:  # pragma: no cover - stats must import without the db extras
        ANATOMY_CONFIG_VARIABLES = {}  # type: ignore[assignment]

    try:
        registered = repo.catalog.variable_entries(domain="image")
    except Exception:
        registered = []

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in registered:
        variable_id = str(entry.get("variable_id") or "").strip()
        if not variable_id or variable_id in seen:
            continue
        is_subject_scoped = str(entry.get("scope") or "").strip().lower() == "subject"
        if not (is_subject_scoped or variable_id in ANATOMY_CONFIG_VARIABLES):
            continue
        seen.add(variable_id)
        entries.append(
            {
                "variable_id": variable_id,
                "label": f"{entry.get('label') or variable_id} (subject)",
                "domain": "clinical",
                "table": "image_measurements",
            }
        )
    for variable_id, var in ANATOMY_CONFIG_VARIABLES.items():
        if variable_id in seen:
            continue
        seen.add(str(variable_id))
        entries.append(
            {
                "variable_id": str(variable_id),
                "label": f"{var.label} (subject)",
                "domain": "clinical",
                "table": "image_measurements",
            }
        )
    return entries


def subject_image_annotation_columns(repo: Any) -> list[str]:
    """``variable_id``s of the subject-level image annotations offered as covariates."""
    return [str(e["variable_id"]) for e in subject_image_annotation_entries(repo)]


def _subject_image_annotation_frame(repo: Any, names: list[str]) -> pd.DataFrame:
    """
    One row per ``subject_uid`` with the requested subject-level image annotations as columns.

    Read long and pivoted here rather than through ``repo.image(wide=True)``: the wide pivot keys on
    ``region_id``, which these rows deliberately leave empty. Values come from ``value_text`` with a
    numeric fallback, and a subject annotated twice keeps the last non-empty value.
    """
    if not names:
        return pd.DataFrame()
    try:
        frame = repo.get(
            "image_measurements",
            cohort_id=False,
            filters={"variable_id": list(names)},
        )
    except Exception as exc:
        log.debug("Subject image annotations unavailable: %s", exc)
        return pd.DataFrame()
    if frame is None or frame.empty or "variable_id" not in frame.columns:
        return pd.DataFrame()

    text = frame["value_text"] if "value_text" in frame.columns else pd.Series(pd.NA, index=frame.index)
    numeric = frame["value_num"] if "value_num" in frame.columns else pd.Series(pd.NA, index=frame.index)
    values = text.astype("object").where(text.notna(), numeric)
    tidy = pd.DataFrame(
        {
            "subject_uid": frame["subject_uid"].astype("string"),
            "variable_id": frame["variable_id"].astype("string"),
            "value": values,
        }
    ).dropna(subset=["subject_uid", "value"])
    if tidy.empty:
        return pd.DataFrame()
    wide = (
        tidy.drop_duplicates(subset=["subject_uid", "variable_id"], keep="last")
        .pivot(index="subject_uid", columns="variable_id", values="value")
        .reset_index()
    )
    wide.columns.name = None
    keep = ["subject_uid", *[c for c in names if c in wide.columns]]
    return wide[keep]


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
#: Which published granularity each grouping wants — see :func:`_select_region_granularity`.
_GROUPING_GRANULARITY: dict[str, str] = {
    "region": "parcel",
    "vessel": "parcel",
    "tree": "parcel",
    "lobe": "lobe",
    "hemisphere": "hemisphere",
    # A vascular territory is the mean of the parcels inside it. Averaging the parcels *and* the
    # whole-brain summary that contains them weights the summary as though it were one more parcel.
    "territory": "parcel",
}


def _select_region_granularity(
    long: pd.DataFrame,
    *,
    grouping: str,
    variable_id: str,
    explicit_regions: bool,
) -> pd.DataFrame:
    """
    Keep one granularity of published region, so an aggregate never sits beside its own members.

    The ASL and T1 tables publish several granularities in the same ``region_id`` column:
    ``ctx-lh-precuneus`` (a parcel), ``ctx-Left-Parietal-Lobe`` (the lobe that contains it),
    ``ctx-left-hemisphere`` and ``ctx-whole-brain``. Loading all of them as levels of one factor
    means a model compares a parcel against a sum it is part of, and the brain map paints a lobe on
    top of the parcels it pools — both wrong, and neither announces itself.

    So the grouping decides the granularity:

    ``region`` / ``vessel``
        parcels only — the finest level, and what "atlas region" means.
    ``lobe`` / ``hemisphere``
        the **published** aggregate when the pipeline provides one, because that is the pipeline's
        own volume-weighted summary; parcels re-averaged unweighted are a different (worse) number.
        Falls back to deriving from parcels when no aggregate row exists.
    ``territory``
        parcels, because a territory is the mean of the parcels inside it — pooling the parcels
        *and* the whole-brain summary that contains them weights that summary as one more parcel.

    An explicit ``regions=`` filter always wins — naming an id is an unambiguous request for it.
    """
    from nvitk.stats.region_groups import region_granularity

    wanted = _GROUPING_GRANULARITY.get(str(grouping).strip().lower())
    if wanted is None or explicit_regions or long.empty:
        return long

    granularity = long["region_id"].astype(str).map(region_granularity)
    counts = granularity.value_counts().to_dict()
    if len(counts) <= 1:
        return long

    keep_class = wanted if counts.get(wanted) else "parcel"
    kept = long.loc[granularity == keep_class].copy()
    if kept.empty:
        return long

    dropped = {k: v for k, v in counts.items() if k != keep_class}
    log.info(
        "%s: %d published region id(s) are %s-level; keeping those for grouping=%r and dropping "
        "%s. A parcel and a sum containing it cannot be levels of the same factor.",
        variable_id, len(kept["region_id"].unique()), keep_class, grouping,
        ", ".join(f"{n} {k}" for k, n in sorted(dropped.items())),
    )
    return kept


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
    regions: Sequence[str] = (),
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
    regions : sequence of str
        Keep only these published ``region_id``s. Empty means all. Matching is case-insensitive and
        ignores ``-``/``_`` differences, because the same structure is spelled several ways across
        importers.

        The filter runs *after* composites are built, so naming ``TCBF`` here keeps the composite
        while dropping the vessels it was summed from — which is the point of asking for it. It is
        what makes a whole-head scalar usable: ``t1_volume_mm3`` publishes 61 regions, of which
        ``etiv`` is one, and a model wanting head size wants that row and no other.

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

    # ---- Region pre-filter, after composites so a composite label can be selected ---------------
    if regions:
        wanted = {_region_match_key(r) for r in regions if str(r).strip()}
        keep = long["region_id"].map(lambda r: _region_match_key(r) in wanted)
        matched = sorted({str(r) for r, k in zip(long["region_id"], keep) if k})
        unmatched = sorted(
            r for r in regions
            if _region_match_key(r) not in {_region_match_key(m) for m in matched}
        )
        if unmatched:
            log.warning(
                "Region filter: %s not present in %s (available: %s%s).",
                ", ".join(unmatched), variable_id,
                ", ".join(source_region_ids[:8]),
                "…" if len(source_region_ids) > 8 else "",
            )
        long = long.loc[keep].copy()
        if long.empty:
            raise ValueError(
                f"The region filter {list(regions)!r} matched none of {variable_id}'s "
                f"{len(source_region_ids)} region(s). Available: "
                f"{', '.join(source_region_ids[:12])}"
                f"{'…' if len(source_region_ids) > 12 else ''}."
            )
        log.info(
            "Region filter on %s: kept %d of %d region(s) — %s.",
            variable_id, len(matched), len(source_region_ids), ", ".join(matched[:8]),
        )

    # ---- Granularity: never mix a parcel with a sum that already contains it --------------------
    long = _select_region_granularity(
        long, grouping=grouping, variable_id=variable_id, explicit_regions=bool(regions)
    )

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

    # A time-resolved variable becomes one column per cardiac frame; collapsing it into a single
    # number would reproduce ``flow_mean`` and throw away the waveform that made it worth loading.
    if is_timeseries_feature(variable_id) and "frame_index" in long.columns:
        from nvitk.stats._hemodynamic_frames import aggregate_territory_frames, frame_columns

        wide = aggregate_territory_frames(long, variable_id, agg=agg)
        out_col = str(column or variable_id)
        if out_col != variable_id:
            wide = wide.rename(
                columns={
                    name: name.replace(variable_id, out_col, 1)
                    for name in frame_columns(wide, variable_id)
                }
            )
        n_frames = len(frame_columns(wide, out_col))
        log.info(
            "%s: %d cardiac frame(s) loaded as %s … %s.",
            variable_id, n_frames, out_col, f"{out_col}_f{n_frames - 1}" if n_frames > 1 else out_col,
        )
        wide.attrs["region_ids"] = source_region_ids
        wide.attrs["frame_columns"] = frame_columns(wide, out_col)
        return wide

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
    Subject-level covariate frame assembled from the clinical / subjects / image / cognitive tables.

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
    # ``subjects`` entity table for names that exist solely as subject attributes, and to
    # ``image_measurements`` for the subject-level manual annotations (``cow_config``, …).
    clinical_present = clinical_measurement_variable_ids(repo)
    annotation_names = set(subject_image_annotation_columns(repo)) - clinical_present
    annotation_vars = [v for v in clinical_vars if v in annotation_names]
    clinical_vars = [v for v in clinical_vars if v not in annotation_vars]
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
    if annotation_vars:
        add(_subject_image_annotation_frame(repo, annotation_vars))
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

    covariate_vars = list(
        dict.fromkeys([*clinical_vars, *subject_vars, *annotation_vars, *cognitive_vars])
    )
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
    # A subject-grain frame has no territory column — each region became its own variable. There is
    # then exactly one group, and saying so keeps every downstream consumer (plot facets, mixed-model
    # grouping defaults, summaries) working off one column name instead of branching on the grain.
    if "territory" in wide.columns:
        wide["group_key"] = wide["territory"].astype(str)
    elif "group_key" not in wide.columns:
        wide["group_key"] = "subject"

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
    #: Restrict the measurement to these published ``region_id``s before grouping. Empty means all.
    #: Applied *after* composites are built, so a composite label (``TCBF``) can be named here and
    #: still be computed from every vessel that feeds it.
    regions: tuple[str, ...] = ()

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
        if self.regions:
            shown = ", ".join(self.regions[:3])
            more = f" +{len(self.regions) - 3}" if len(self.regions) > 3 else ""
            parts.append(f"[{shown}{more}]")
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
            "regions": list(self.regions),
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
            regions=tuple(str(r) for r in (data.get("regions") or ())),
        )


def _keys_for_grouping(spec: MeasurementSpec, region_ids: Sequence[str], grouping: str) -> set[str]:
    """Territory keys *spec*'s region ids would produce under *grouping*."""
    variable_id = resolve_feature_id(spec.feature)
    return {
        assign_group_key(rid, grouping=grouping, kind=spec.pipeline_kind, variable_id=variable_id)
        for rid in region_ids
    }


def _join_on_subject(
    specs: Sequence[MeasurementSpec],
    frames: Sequence[pd.DataFrame],
    meta: dict[str, Any],
    *,
    join: str = "inner",
) -> pd.DataFrame:
    """
    Combine measurements on ``subject_uid`` alone, spreading each one's territories into columns.

    The territory column is dropped rather than kept, because after this join there is no single
    territory a row belongs to — that is the point of the grain. Anything downstream that wants to
    group by region works off the ``@territory`` suffix instead.
    """
    wide: pd.DataFrame | None = None
    produced: dict[str, list[str]] = {}
    for spec, frame in zip(specs, frames):
        block = subject_wide_frame(frame, column=spec.column())
        if block.empty:
            meta["warnings"].append(f"{spec.column()!r} contributed no rows to the subject join.")
            continue
        produced[spec.column()] = list(block.attrs.get("value_columns") or block.columns)
        wide = block if wide is None else wide.join(block, how=join)

    if wide is None:
        meta["key_overlap"] = 0.0
        meta["warnings"].append("No measurement produced any subject-level column.")
        return pd.DataFrame(columns=["subject_uid"])

    base_subjects = set(frames[0]["subject_uid"].astype(str)) if len(frames[0]) else set()
    kept = set(wide.index.astype(str))
    meta["key_overlap"] = len(kept & base_subjects) / len(base_subjects) if base_subjects else 0.0
    meta["subject_columns"] = produced
    meta["grain"] = "subject"

    total = sum(len(v) for v in produced.values())
    meta["warnings"].append(
        f"Joined on subject: {len(wide)} row(s), {total} measurement column(s) "
        f"({'; '.join(f'{k} → {len(v)}' for k, v in produced.items())}). There is no 'territory' "
        f"column in this grain — a model cannot carry a territory term, and each region is its own "
        f"variable."
    )
    if wide.empty:
        meta["warnings"].append(
            "No subject has all of the selected measurements. Try join='outer' to keep partial "
            "subjects."
        )
    return wide.reset_index()


def subject_wide_frame(
    frame: pd.DataFrame, *, column: str, region_column: str = "territory"
) -> pd.DataFrame:
    """
    Spread one measurement's territories across columns, leaving one row per subject.

    Turns the long ``(subject, territory, value)`` shape into ``subject`` plus one column per
    territory, named ``value__territory`` — ``flow_mean__LICA``, ``mean_cbf__left_mca_8``. A
    measurement with a single territory keeps its plain name, since ``mean_cbf__ctx_whole_brain``
    carries no more information than ``mean_cbf`` when that is the only one there is.

    The double underscore is not decoration: these names go into patsy and R formulas, which accept
    only ``[A-Za-z_][A-Za-z0-9_]*``. A separator like ``@`` or ``.`` would read better and produce
    columns no model could reference without quoting.

    This is what makes cross-modality models possible. Territories are only comparable *within* a
    parcellation: a 4D-flow vessel, an ASL vascular territory and a FreeSurfer parcel name different
    kinds of thing, so joining them on a shared territory key is not a constraint to relax but a
    question with no answer. Joining on the subject instead asks something that does have one —
    "does this subject's whole-brain perfusion track their carotid flow" — and puts both on one row.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``subject_uid``. ``frame.attrs["value_columns"]`` lists the columns produced.
    """
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    if region_column not in frame.columns:
        # Already one row per subject — nothing to spread.
        out = frame.set_index("subject_uid")[[column]]
        out.attrs["value_columns"] = [column]
        return out

    territories = [str(t) for t in frame[region_column].dropna().unique()]
    wide = frame.pivot_table(
        index="subject_uid", columns=region_column, values=column, aggfunc="mean"
    )
    if len(territories) <= 1:
        wide.columns = [column]
    else:
        wide.columns = [f"{column}__{_identifier_fragment(c)}" for c in wide.columns]
    wide.attrs["value_columns"] = list(wide.columns)
    return wide


#: Columns of an analysis frame that hold the *same* region label under different names.
#: :func:`finalize_analysis_frame` copies ``territory`` into ``group_key`` so downstream consumers
#: have one name to rely on, which means a frame carries the region twice — and a formula may name
#: either. Anything that changes one of them (a reference level, a recode) has to change all of
#: them, or the model silently uses the untouched copy.
REGION_ALIAS_COLUMNS: tuple[str, ...] = ("territory", "group_key", "region_id")


def region_alias_columns(frame: pd.DataFrame, column: str) -> list[str]:
    """
    Columns of *frame* carrying the same region labels as *column*, itself included.

    Membership is decided by comparing values rather than by trusting the names: a frame where
    ``group_key`` was melted from something else, or where the user renamed a column, must not have
    an unrelated column recoded underneath it.
    """
    if frame is None or column not in frame.columns:
        return [column]
    reference = frame[column].astype(str)
    out = [column]
    for candidate in REGION_ALIAS_COLUMNS:
        if candidate == column or candidate not in frame.columns:
            continue
        other = frame[candidate].astype(str)
        if len(other) == len(reference) and other.equals(reference):
            out.append(candidate)
    return out


def subject_measurement_families(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    ``{measurement: [regions]}`` for the ``value__region`` columns of a subject-grain frame.

    Reads the naming :func:`subject_wide_frame` produced, so a frame can be melted back without
    being told which columns belong together.
    """
    families: dict[str, list[str]] = {}
    for column in df.columns:
        name = str(column)
        head, sep, tail = name.partition("__")
        if sep and head and tail:
            families.setdefault(head, []).append(tail)
    return {k: sorted(v) for k, v in sorted(families.items())}


def melt_subject_frame(
    df: pd.DataFrame,
    *,
    family: str,
    region_column: str = "territory",
) -> pd.DataFrame:
    """
    Melt one measurement's ``value__region`` columns of a subject-grain frame back to long.

    This is the shape a model needs when the *region* is a term rather than a set of separate
    variables. A frame joined on the subject can hold measurements from unrelated parcellations —
    a 4D-flow vessel beside a whole-head eTIV — but ``flow_mean ~ psqeduca * territory`` cannot be
    written against it, because there is no territory column to interact with. Melting one family
    restores that column while leaving every *other* column repeated down the rows, so the eTIV and
    the covariates ride along and stay usable as predictors.

    Only the named family moves. Melting two at once would need their regions to correspond, which
    is exactly the assumption the subject grain exists to avoid.

    Parameters
    ----------
    family : str
        Measurement prefix, e.g. ``flow_mean`` for ``flow_mean__LICA`` … ``flow_mean__TCBF``.
    region_column : str
        Name for the recovered region column.

    Returns
    -------
    pandas.DataFrame
        One row per (subject × region of *family*), with *family* as the value column and
        ``group_key`` set to the region so plots and grouped models pick it up.

    Raises
    ------
    ValueError
        When no column carries the ``family__`` prefix — melting would otherwise silently produce
        an empty frame.
    """
    prefix = f"{family}__"
    value_columns = [c for c in df.columns if str(c).startswith(prefix)]
    if not value_columns:
        available = ", ".join(subject_measurement_families(df)) or "none"
        raise ValueError(
            f"No column starts with {prefix!r}, so there is nothing to melt. "
            f"Measurement families in this frame: {available}."
        )
    if family in df.columns:
        raise ValueError(
            f"{family!r} is already a column of this frame; melting would produce two columns of "
            f"that name. Rename the measurement's output column first."
        )

    id_columns = [c for c in df.columns if c not in value_columns]
    long = df.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name=region_column,
        value_name=family,
    )
    long[region_column] = long[region_column].astype(str).str.slice(len(prefix))
    long["group_key"] = long[region_column].astype(str)

    missing = int(long[family].isna().sum())
    if missing:
        log.info(
            "Melted %s into %d row(s) over %d region(s); %d cell(s) are empty (a subject without "
            "that region).",
            family, len(long), len(value_columns), missing,
        )
    # ``melt`` emits one contiguous block per value column, so the first rows of the result are all
    # the same region. A capped table preview then shows a handful of regions and looks as though
    # the rest were lost. Interleaving by subject makes the preview representative of the whole.
    sort_keys = [c for c in ("subject_uid", region_column) if c in long.columns]
    if sort_keys:
        long = long.sort_values(sort_keys, kind="stable")
    return long.reset_index(drop=True)


def _identifier_fragment(value: Any) -> str:
    """Reduce a territory label to something usable inside a formula identifier."""
    token = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return token or "region"


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



#: Per-vessel QC variables the qvtpy stage 9 publishes into ``image_measurements``.
QC_VESSEL_VARIABLES: tuple[str, ...] = (
    "qc_flow_plausible", "qc_hypoplastic", "qc_conservation", "qc_score", "qc_flag",
)


def attach_vessel_qc(
    repo: Any,
    wide: pd.DataFrame,
    *,
    spec: Any,
    variables: Sequence[str] = QC_VESSEL_VARIABLES,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Merge the published per-vessel QC metrics onto an analysis frame.

    They live in ``image_measurements`` keyed on ``(subject_uid, region_id)`` — the same grain as a
    measurement — but they are not measurements anyone selects, so nothing would load them. Without
    this the QC filters are permanently greyed out on a dataset that has them, which is exactly the
    dataset where they matter.

    The region key is collapsed with the **same** :func:`assign_group_key` the measurement used, so
    a melted frame (territory or hemisphere grouping) gets the QC averaged over the vessels that
    were melted into each key. That is meaningful for the scores; for the flags it becomes the
    *fraction* of the melted vessels that failed, which is why the melted case is reported.

    Returns
    -------
    (frame, attached)
        The frame with any QC columns added, and their names. Both unchanged/empty when the dataset
        carries none — a dataset that has not run stage 9 is the normal case, not an error.
    """
    if wide.empty or "subject_uid" not in wide.columns or "territory" not in wide.columns:
        return wide, []
    try:
        rows = repo.get("image_measurements", cohort_id=False)
    except Exception as exc:
        log.debug("Could not read image_measurements for the QC metrics: %s", exc)
        return wide, []
    if rows is None or rows.empty or "variable_id" not in rows.columns:
        return wide, []

    wanted = [v for v in variables if v in set(rows["variable_id"].astype(str))]
    if not wanted:
        return wide, []
    qc = rows.loc[rows["variable_id"].astype(str).isin(wanted)]
    if qc.empty or not {"subject_uid", "region_id", "value_num"} <= set(qc.columns):
        return wide, []

    kind = str(getattr(spec, "pipeline_kind", "qvtpy") or "qvtpy")
    grouping = str(getattr(spec, "grouping", "vessel") or "vessel")
    keys = qc["region_id"].astype(str).map(
        lambda r: assign_group_key(r, grouping=grouping, kind=kind)
    )
    table = (
        qc.assign(territory=keys.astype(str))
        .pivot_table(
            index=["subject_uid", "territory"], columns="variable_id",
            values="value_num", aggfunc="mean",
        )
        .reset_index()
    )
    table["subject_uid"] = table["subject_uid"].astype(str)
    table["territory"] = table["territory"].astype(str)

    out = wide.copy()
    out["subject_uid"] = out["subject_uid"].astype(str)
    out["territory"] = out["territory"].astype(str)
    # Never overwrite a column the frame already has — a derived one of the same name wins.
    attached = [c for c in wanted if c not in out.columns]
    if not attached:
        return wide, []
    merged = out.merge(
        table.loc[:, ["subject_uid", "territory", *attached]],
        on=["subject_uid", "territory"], how="left", validate="many_to_one",
    )
    log.info(
        "Attached %d QC metric(s) on the %r grouping: %s",
        len(attached), grouping, ", ".join(attached),
    )
    return merged, attached


def build_multi_feature_analysis_frame(
    repo: Any,
    *,
    measurements: Sequence[MeasurementSpec | Mapping[str, Any]],
    clinical_vars: list[str] | None = None,
    cognitive_vars: list[str] | None = None,
    join: str = "inner",
    grain: str = "territory",
    attach_qc: bool = True,
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
    grain : {"territory", "subject"}
        What a row is. ``"territory"`` keeps the long shape and joins on the shared
        ``(subject, territory)`` cell — right when the measurements share a parcellation, and the
        only way to fit a model with a territory term. ``"subject"`` gives one row per subject and
        spreads each measurement's territories into ``value@territory`` columns, which is the only
        way to relate measurements whose parcellations name different kinds of region: an ASL
        whole-brain CBF against a 4D-flow vessel, or a FreeSurfer parcel against either.
    attach_qc : bool
        Merge the published per-vessel autoQC metrics (``qc_flow_plausible``, ``qc_score``, …) onto
        the frame. They are what the QC filter presets read, so turning this off greys those out —
        but it also keeps five columns nobody asked for out of a frame built for something else,
        and skips their lookup on a dataset where the stage never ran.

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
            regions=spec.regions,
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

    # ---- 3. Join, either on the shared (subject, territory) cell or per subject -
    if grain == "subject":
        wide = _join_on_subject(specs, frames, meta, join=join)
    else:
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
                    + (fix or "These parcellations name different kinds of region, so no grouping "
                              "makes them share a key — switch the join grain to 'subject' to put "
                              "them side by side as one row per subject instead.")
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
                "(subject, territory) cell. If they come from different parcellations (a vessel "
                "against a cortical parcel, say), that is expected: switch the join grain to "
                "'subject'."
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

    # The published per-vessel QC metrics, on the same key the measurements were loaded on. They
    # are what the analysis dataframe's QC filters read, and nothing else would load them.
    if not attach_qc:
        meta["qc_columns"] = []
    elif not wide.empty and grain == "subject":
        # The vessel QC metrics are per (subject, vessel); with no territory column there is no row
        # to attach them to, and averaging them over a subject would turn a per-vessel gate into a
        # number that gates nothing. Filter on the territory grain, then switch.
        meta["qc_columns"] = []
    elif not wide.empty:
        wide, qc_columns = attach_vessel_qc(repo, wide, spec=specs[0])
        meta["qc_columns"] = qc_columns
        if qc_columns and specs[0].grouping not in {"vessel", "tree", "region"}:
            meta["warnings"].append(
                f"The QC metrics were averaged over the vessels melted into each "
                f"{specs[0].grouping!r} key. The scores stay meaningful; a flag becomes the "
                f"fraction of that key's vessels that failed."
            )
    else:
        meta["qc_columns"] = []

    wide = finalize_analysis_frame(wide, repo, primary_variable_id=specs[0].column())

    # pandas tolerates repeated column names; polars and R do not, and they fail with a message
    # that names neither the column nor the frame. Catch it here, where the report can say which
    # column it was, rather than at fit time in someone else's error string.
    from .frame_ops import ensure_unique_columns

    before = list(wide.columns)
    wide = ensure_unique_columns(wide, context="analysis frame")
    dropped = [c for c in dict.fromkeys(before) if before.count(c) > 1]
    if dropped:
        meta["warnings"].append(
            f"Duplicate column(s) in the analysis frame were reduced to one: {', '.join(dropped)}. "
            "Check whether a covariate shares a name with a measurement."
        )

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
    "t1_region_to_hemisphere_pair_key",
    "resolve_feature_id",
    "subject_attribute_columns",
    "subject_attribute_entries",
    "subject_image_annotation_columns",
    "subject_image_annotation_entries",
    "clinical_measurement_variable_ids",
]
