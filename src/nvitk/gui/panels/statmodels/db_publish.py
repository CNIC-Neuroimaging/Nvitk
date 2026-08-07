"""
Publish a derived analysis column back into the dataset as a first-class variable.

Description
-----------
A derived column — ``log_pi``, ``cp_group``, ``asl_vs_flow_diff`` — lives only inside one Statmodels
session. Recomputing it in the next session, or in a notebook, means retyping the definition and
hoping it matches. Writing it back to the dataset makes it a variable like any other: it appears in
the covariate lists, it can be loaded as a measurement, and it carries provenance saying which
column and which expression produced it.

Which table
-----------
A derived value belongs wherever its **source** lives, because that is what determines the key it is
measured on:

=========================  ==========================  ===========================================
Source                     Table                       Key
=========================  ==========================  ===========================================
image measurement          ``image_measurements``      ``subject_uid`` × ``region_id``
clinical variable          ``clinical_measurements``   ``subject_uid`` (× ``visit_id``)
cognitive variable         ``cognitive_measurements``  ``subject_uid`` (× ``visit_id``)
=========================  ==========================  ===========================================

The row building and the upsert are :mod:`nvitk.db.derived_measurements`' job; this module only
works out *which* of its entry points applies and assembles the aggregate frame it expects.

Region identity
---------------
The analysis frame is keyed on the **melted** territory, not on the published ``region_id``: a frame
built with ``grouping="territory"`` has one row for "Anterior Circulation", aggregated over four
vessels. There is no single region id to attribute that value to, so the group key is written as the
``region_id`` and the grouping that produced it is recorded in ``region_label`` and in the
provenance. A value derived from a vessel-wise frame round-trips exactly; one derived from a melted
frame round-trips to the same melted key, which is the honest answer rather than inventing a
component vessel.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.derived_measurements import (
    DerivedClinicalMeasurementSpec,
    DerivedImageMeasurementSpec,
    DerivedVariableRegistration,
    build_clinical_measurement_rows,
    build_image_measurement_rows,
    publish_derived_measurements,
)
from nvitk.stats.frame_ops import DerivedColumn

log = Logger()

IMAGE_TABLE = "image_measurements"
CLINICAL_TABLE = "clinical_measurements"
COGNITIVE_TABLE = "cognitive_measurements"

#: Tables a derived column can be written to, with the domain the catalog records for it.
PUBLISH_TABLES: tuple[tuple[str, str, str], ...] = (
    ("Image measurements — one row per subject × region", IMAGE_TABLE, "image"),
    ("Clinical measurements — one row per subject", CLINICAL_TABLE, "clinical"),
    ("Cognitive measurements — one row per subject", COGNITIVE_TABLE, "cognitive"),
)

#: A variable_id has to survive being used as a formula term, so it must be a Python identifier.
VARIABLE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class PublishTarget:
    """Where a derived column goes and what identifies each of its rows."""

    table: str
    domain: str
    key_columns: tuple[str, ...]
    #: Why this table was chosen, shown in the dialog so a wrong guess is visible before writing.
    reason: str = ""
    modality: str = ""
    pipeline_id: str = ""
    grouping: str = ""

    def is_image(self) -> bool:
        """Whether rows carry a region, and therefore go to ``image_measurements``."""
        return self.table == IMAGE_TABLE


def resolve_publish_target(
    column: str,
    *,
    derived: Sequence[DerivedColumn],
    measurement_columns: Mapping[str, Any] | None = None,
    clinical_columns: Sequence[str] = (),
    cognitive_columns: Sequence[str] = (),
    frame_columns: Sequence[str] = (),
) -> PublishTarget:
    """
    Work out which table a derived column belongs to, by following it back to its source.

    Resolution walks the derived chain: a column derived from a column derived from ``pi`` is still
    an image measurement. An expression that mixes domains cannot be attributed, so it falls back to
    the image table when the frame is region-keyed and to clinical otherwise — and says so in
    :attr:`PublishTarget.reason`, because that is exactly the case where the caller should look.

    Parameters
    ----------
    measurement_columns : mapping
        ``{column: MeasurementSpec}`` for the image measurements currently loaded.
    clinical_columns, cognitive_columns
        Covariate columns resolved from those domains.
    frame_columns
        Columns of the analysis frame, used to detect whether it is region-keyed at all.
    """
    measurement_columns = dict(measurement_columns or {})
    clinical = {str(c) for c in clinical_columns}
    cognitive = {str(c) for c in cognitive_columns}
    by_name = {d.name: d for d in derived}
    region_keyed = any(c in set(frame_columns) for c in ("territory", "group_key", "region_id"))

    def sources_of(name: str, seen: frozenset[str] = frozenset()) -> set[str]:
        """Base (non-derived) columns *name* ultimately depends on."""
        if name in seen:
            return set()
        spec = by_name.get(name)
        if spec is None:
            return {name}
        seen = seen | {name}
        if spec.kind in {"transform", "bins"} and spec.source:
            return sources_of(spec.source, seen)
        # An expression can name any number of columns; only the identifiers that are real columns
        # of the frame count as sources.
        known = set(by_name) | set(frame_columns)
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", spec.expression or "")) & known
        out: set[str] = set()
        for token in sorted(tokens):
            out |= sources_of(token, seen)
        return out or {name}

    roots = sources_of(str(column))
    image_roots = sorted(roots & set(measurement_columns))
    clinical_roots = sorted(roots & clinical)
    cognitive_roots = sorted(roots & cognitive)

    if image_roots and not (clinical_roots or cognitive_roots):
        spec = measurement_columns[image_roots[0]]
        return PublishTarget(
            table=IMAGE_TABLE,
            domain="image",
            key_columns=("subject_uid", "region_id", "variable_id"),
            reason=f"derived from the image measurement {image_roots[0]!r}",
            modality=str(getattr(spec, "pipeline_kind", "") or ""),
            pipeline_id=str(getattr(spec, "pipeline", "") or "latest"),
            grouping=str(getattr(spec, "grouping", "") or ""),
        )
    if clinical_roots and not (image_roots or cognitive_roots):
        return PublishTarget(
            table=CLINICAL_TABLE, domain="clinical",
            key_columns=("subject_uid", "visit_id", "variable_id"),
            reason=f"derived from the clinical variable {clinical_roots[0]!r}",
        )
    if cognitive_roots and not (image_roots or clinical_roots):
        return PublishTarget(
            table=COGNITIVE_TABLE, domain="cognitive",
            key_columns=("subject_uid", "visit_id", "variable_id"),
            reason=f"derived from the cognitive variable {cognitive_roots[0]!r}",
        )

    mixed = sorted(image_roots + clinical_roots + cognitive_roots)
    if mixed and region_keyed:
        spec = measurement_columns.get(image_roots[0]) if image_roots else None
        return PublishTarget(
            table=IMAGE_TABLE, domain="image",
            key_columns=("subject_uid", "region_id", "variable_id"),
            reason=(
                f"derived from more than one domain ({', '.join(mixed)}); stored per region because "
                f"the analysis frame is region-keyed — change the table if that is wrong"
            ),
            modality=str(getattr(spec, "pipeline_kind", "") or "") if spec else "",
            pipeline_id=str(getattr(spec, "pipeline", "") or "latest") if spec else "latest",
            grouping=str(getattr(spec, "grouping", "") or "") if spec else "",
        )
    return PublishTarget(
        table=CLINICAL_TABLE, domain="clinical",
        key_columns=("subject_uid", "visit_id", "variable_id"),
        reason=(
            f"could not attribute {column!r} to a loaded measurement or covariate"
            if not mixed else f"derived from {', '.join(mixed)}"
        ),
    )


@dataclass
class PublishRequest:
    """Everything the write needs, as edited in the dialog."""

    column: str
    variable_id: str
    label: str
    table: str
    domain: str
    unit: str = ""
    aliases: tuple[str, ...] = ()
    modality: str = ""
    pipeline_id: str = "latest"
    grouping: str = ""
    region_column: str = "territory"
    definition: str = ""
    register: bool = True
    extra_provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> str:
        """Empty when the request can be written, otherwise the reason it cannot."""
        if not VARIABLE_ID_RE.match(self.variable_id or ""):
            return (
                "The variable id must start with a letter or underscore and contain only letters, "
                "digits and underscores — it is used as a formula term."
            )
        if self.table not in {IMAGE_TABLE, CLINICAL_TABLE, COGNITIVE_TABLE}:
            return f"Unknown table {self.table!r}."
        return ""


def build_publish_frame(df: pd.DataFrame, request: PublishRequest) -> pd.DataFrame:
    """
    Reduce the analysis frame to the rows that will be written.

    Image rows keep one value per ``(subject_uid, region)``; clinical and cognitive rows collapse to
    one value per subject, since a subject-level variable repeated across territories is the same
    number written N times. Rows whose value is missing are dropped — writing NaN would shadow a real
    value on the next read.

    Raises
    ------
    ValueError
        When the frame lacks ``subject_uid``, the value column, or (for image rows) the region
        column, or when nothing survives.
    """
    column = request.column
    if column not in df.columns:
        raise ValueError(f"Column {column!r} is not in the analysis dataframe.")
    if "subject_uid" not in df.columns:
        raise ValueError(
            "The analysis dataframe has no 'subject_uid' column, so there is nothing to key the "
            "rows on. Reload the data before publishing."
        )

    is_categorical = isinstance(df[column].dtype, pd.CategoricalDtype) or not (
        pd.api.types.is_numeric_dtype(df[column])
    )
    keep = ["subject_uid", column]
    if request.table == IMAGE_TABLE:
        region = request.region_column
        if region not in df.columns:
            raise ValueError(
                f"Region column {region!r} is not in the analysis dataframe; an image measurement "
                f"needs one to identify its rows."
            )
        keep.append(region)

    out = df.loc[:, [c for c in dict.fromkeys(keep) if c in df.columns]].copy()
    out = out.dropna(subset=[column])
    if out.empty:
        raise ValueError(f"Every value of {column!r} is missing — nothing to write.")

    if request.table == IMAGE_TABLE:
        out = out.rename(columns={request.region_column: "region_id"})
        out["region_id"] = out["region_id"].astype(str)
        # ``region_label`` records *how* the region key was formed, so a value stored against
        # "Anterior Circulation" is not mistaken for a published vessel id later.
        out["region_label"] = request.grouping or "analysis-frame key"
        out = out.drop_duplicates(subset=["subject_uid", "region_id"], keep="first")
    else:
        # One row per subject: a clinical value repeated across a subject's territories is one fact.
        # Unless it is not — a value that genuinely varies within a subject is region-level data
        # heading for a subject-level table, and taking the first row would silently discard the
        # rest. Say so rather than writing a number nobody can reproduce.
        varying = out.groupby("subject_uid")[column].nunique(dropna=True)
        n_varying = int((varying > 1).sum())
        if n_varying:
            log.warning(
                "%r takes more than one value within %d subject(s); only the first row per subject "
                "is written to %s. Publish it to the image table instead if it is region-level.",
                column, n_varying, request.table,
            )
        n_before = len(out)
        out = out.drop_duplicates(subset=["subject_uid"], keep="first")
        if n_before != len(out):
            log.info(
                "Collapsed %d analysis rows to %d subject-level rows for %r.",
                n_before, len(out), column,
            )
        if "visit_id" in df.columns:
            visits = df.loc[out.index, "visit_id"] if out.index.isin(df.index).all() else None
            if visits is not None:
                out["visit_id"] = visits.astype(str).to_numpy()

    values = out[column]
    if is_categorical:
        # A binned column is a label, not a number; ``value_text`` is where the schema keeps those.
        out["value_num"] = pd.NA
        out["value_text"] = values.astype(str)
    else:
        out["value_num"] = pd.to_numeric(values, errors="coerce")
    return out.reset_index(drop=True)


def publish_derived_column(
    repo: Any,
    df: pd.DataFrame,
    request: PublishRequest,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Write a derived analysis column into the dataset.

    Delegates the actual row construction and upsert to
    :mod:`nvitk.db.derived_measurements`, so the rows land with the same schema, provenance and
    catalog registration as any importer-produced variable.

    Parameters
    ----------
    dry_run : bool
        Build and validate everything, but do not touch the dataset. Returns the same summary with
        ``written=False``, which is what the dialog previews.

    Returns
    -------
    dict
        ``{"rows": DataFrame, "n_rows": int, "n_subjects": int, "table": str, "written": bool}``.
    """
    problem = request.validate()
    if problem:
        raise ValueError(problem)

    agg = build_publish_frame(df, request)
    value_kind = "text" if agg["value_num"].isna().all() else "float"

    if request.table == IMAGE_TABLE:
        spec = DerivedImageMeasurementSpec(
            variable_id=request.variable_id,
            modality=request.modality or "derived",
            pipeline_id=request.pipeline_id or "latest",
            source_file="statmodels",
            source_sheet="derived_columns",
            source_column=request.column,
            value_kind=value_kind,
            unit=request.unit or None,
            pipeline_name=f"statmodels_{request.variable_id}",
        )
        rows = build_image_measurement_rows(agg, spec)
        if value_kind == "text":
            rows["value_text"] = agg["value_text"].astype("string").to_numpy()
        registration = DerivedVariableRegistration.from_image_spec(
            spec, label=request.label or request.variable_id,
            aliases=list(request.aliases) or None,
        )
    else:
        spec = DerivedClinicalMeasurementSpec(
            variable_id=request.variable_id,
            source_file="statmodels",
            source_sheet="derived_columns",
            source_column=request.column,
            value_kind=value_kind,
            unit=request.unit or None,
        )
        rows = build_clinical_measurement_rows(agg, spec)
        if value_kind == "text":
            rows["value_text"] = agg["value_text"].astype("string").to_numpy()
        registration = DerivedVariableRegistration(
            variable_id=request.variable_id,
            domain=request.domain,
            table=request.table,
            label=request.label or request.variable_id,
            source_column=request.column,
            source_file="statmodels",
            source_sheet="derived_columns",
            aliases=list(request.aliases) or None,
            value_kind=value_kind,
            unit=request.unit or None,
        )

    summary = {
        "rows": rows,
        "n_rows": int(len(rows)),
        "n_subjects": int(agg["subject_uid"].nunique()),
        "table": request.table,
        "value_kind": value_kind,
        "written": False,
    }
    if dry_run:
        return summary

    provenance = {
        "importer": "statmodels_derived_column",
        "source_column": request.column,
        "definition": request.definition,
        "grouping": request.grouping,
        **dict(request.extra_provenance),
    }
    publish_derived_measurements(
        repo,
        rows,
        table=request.table,
        register=registration if request.register else None,
        provenance={k: v for k, v in provenance.items() if v},
        upsert_key_columns=list(_key_columns_for(request.table)),
    )
    summary["written"] = True
    log.ok(
        "Published %r as %s.%s — %d rows over %d subjects.",
        request.column, request.table, request.variable_id,
        summary["n_rows"], summary["n_subjects"],
    )
    return summary


def _key_columns_for(table: str) -> tuple[str, ...]:
    """Upsert key for a table: what makes two rows the same measurement."""
    if table == IMAGE_TABLE:
        return ("subject_uid", "region_id", "variable_id", "frame_index")
    return ("subject_uid", "visit_id", "variable_id")


__all__ = [
    "CLINICAL_TABLE",
    "COGNITIVE_TABLE",
    "IMAGE_TABLE",
    "PUBLISH_TABLES",
    "VARIABLE_ID_RE",
    "PublishRequest",
    "PublishTarget",
    "build_publish_frame",
    "publish_derived_column",
    "resolve_publish_target",
]
