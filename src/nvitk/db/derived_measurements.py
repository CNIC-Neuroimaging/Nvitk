"""Build long-form measurement rows, upsert tables, and register catalog variables for derived pipelines.

Typical flow after computing an aggregate dataframe (one row per entity with the new
``value_num`` or ``value_text``):

1. :func:`build_image_measurement_rows` or :func:`build_clinical_measurement_rows`
2. :meth:`DataRepo.upsert_table`
3. :func:`register_derived_variable` or :func:`publish_derived_measurements`

Example (Peak Systolic Flow style)::

    spec = DerivedImageMeasurementSpec(
        variable_id=\"psf\",
        modality=\"4dflow\",
        pipeline_id=\"4dflow_v2\",
        source_file=\"derived\",
        source_sheet=\"psf\",
        source_column=\"psf\",
    )
    rows = build_image_measurement_rows(agg, spec)
    publish_derived_measurements(
        repo,
        rows,
        table=\"image_measurements\",
        register=DerivedVariableRegistration(
            variable_id=\"psf\",
            label=\"Peak systolic flow (max over flow_tseries frames)\",
            aliases=[\"psf\", \"PSF\", \"peak_systolic_flow\"],
            modality=\"4dflow\",
            value_kind=\"float\",
            unit=\"mL/min\",
        ),
        provenance={\"importer\": \"derive_psf\", \"parent_variable\": \"flow_tseries\"},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import pandas as pd

from .repo import DataRepo
from .storage import utc_now_iso


@dataclass(frozen=True)
class DerivedImageMeasurementSpec:
    """Metadata for long-form ``image_measurements`` rows built from an aggregate frame."""

    variable_id: str
    modality: str
    pipeline_id: str
    source_file: str
    source_sheet: str
    source_column: str
    value_column: str = "value_num"
    pipeline_name: str | None = None
    value_kind: str = "float"
    unit: str | None = None
    source_batch_id: str | None = None
    source_table: str | None = None
    qc_status: str | None = None
    source_asset: str | None = None
    measured_at: Any = None  # default NaT; set to pd.Timestamp per project rules
    default_frame_index: int | None = None  # if None, column is pd.NA (nullable Int64)


@dataclass(frozen=True)
class DerivedClinicalMeasurementSpec:
    """Metadata for long-form ``clinical_measurements`` rows built from an aggregate frame."""

    variable_id: str
    source_file: str
    source_sheet: str
    source_column: str
    value_column: str = "value_num"
    value_kind: str = "float"
    unit: str | None = None
    source_batch_id: str | None = None
    source_table: str | None = None
    measured_at: Any = None


@dataclass
class DerivedVariableRegistration:
    """Catalog entry for :meth:`DataRepo.register_variables` (requires variable_id, domain, table)."""

    variable_id: str
    domain: str = "image"
    table: str = "image_measurements"
    label: str | None = None
    source_column: str | None = None
    source_file: str | None = None
    source_sheet: str | None = None
    aliases: list[str] | None = None
    modality: str | None = None
    value_kind: str | None = None
    unit: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_catalog_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "variable_id": self.variable_id,
            "domain": self.domain,
            "table": self.table,
        }
        for key in (
            "label",
            "source_column",
            "source_file",
            "source_sheet",
            "aliases",
            "modality",
            "value_kind",
            "unit",
        ):
            val = getattr(self, key)
            if val is not None:
                entry[key] = val
        entry.update(self.extra)
        return entry

    @classmethod
    def from_image_spec(
        cls,
        spec: DerivedImageMeasurementSpec,
        *,
        label: str,
        aliases: list[str] | None = None,
        **extra: Any,
    ) -> DerivedVariableRegistration:
        """Fill registration source fields from a :class:`DerivedImageMeasurementSpec`."""
        return cls(
            variable_id=spec.variable_id,
            domain="image",
            table="image_measurements",
            label=label,
            source_column=spec.source_column,
            source_file=spec.source_file,
            source_sheet=spec.source_sheet,
            aliases=aliases,
            modality=spec.modality,
            value_kind=spec.value_kind,
            unit=spec.unit,
            extra=dict(extra),
        )


def _series_na_string(n: int) -> pd.Series:
    return pd.Series([pd.NA] * n, dtype="string")


def _optional_string_col(agg: pd.DataFrame, name: str, n: int) -> pd.Series:
    if name in agg.columns:
        return agg[name].astype("string")
    return _series_na_string(n)


def build_image_measurement_rows(
    agg: pd.DataFrame,
    spec: DerivedImageMeasurementSpec,
) -> pd.DataFrame:
    """Map an aggregate frame to ``image_measurements`` columns (one derived variable).

    ``agg`` must include ``subject_uid`` and the column named by ``spec.value_column``.
    Optional columns copied through when present: ``session_id``, ``region_id``,
    ``region_label``, ``frame_index``.
    """
    if agg.empty:
        return pd.DataFrame()
    if "subject_uid" not in agg.columns:
        raise ValueError("agg must include a 'subject_uid' column.")
    if spec.value_column not in agg.columns:
        raise ValueError(f"agg is missing value column {spec.value_column!r}.")

    n = len(agg)
    vn = pd.to_numeric(agg[spec.value_column], errors="coerce")
    pipeline_name = spec.pipeline_name or f"derived_{spec.variable_id}"
    source_table = spec.source_table or f"{spec.source_file}::{spec.source_sheet}"

    frame_index: pd.Series
    if "frame_index" in agg.columns:
        frame_index = pd.to_numeric(agg["frame_index"], errors="coerce").astype("Int64")
    elif spec.default_frame_index is not None:
        frame_index = pd.Series([spec.default_frame_index] * n, dtype="Int64")
    else:
        frame_index = pd.Series([pd.NA] * n, dtype="Int64")

    measured_at = spec.measured_at
    if measured_at is None:
        measured_ts = pd.Series([pd.NaT] * n, dtype="datetime64[ns]")
    else:
        measured_ts = pd.Series([measured_at] * n, dtype="datetime64[ns]")

    batch = spec.source_batch_id
    if batch is None:
        batch_s = _series_na_string(n)
    else:
        batch_s = pd.Series([batch] * n, dtype="string")

    out = pd.DataFrame(
        {
            "subject_uid": agg["subject_uid"].astype("string"),
            "session_id": _optional_string_col(agg, "session_id", n),
            "modality": pd.Series([spec.modality] * n, dtype="string"),
            "region_id": _optional_string_col(agg, "region_id", n),
            "region_label": _optional_string_col(agg, "region_label", n),
            "frame_index": frame_index,
            "variable_id": pd.Series([spec.variable_id] * n, dtype="string"),
            "value_num": vn.astype("float64"),
            "value_text": _series_na_string(n),
            "unit": pd.Series([spec.unit if spec.unit is not None else pd.NA] * n, dtype="string"),
            "value_kind": pd.Series([spec.value_kind] * n, dtype="string"),
            "pipeline_name": pd.Series([pipeline_name] * n, dtype="string"),
            "pipeline_id": pd.Series([spec.pipeline_id] * n, dtype="string"),
            "qc_status": pd.Series([spec.qc_status if spec.qc_status is not None else pd.NA] * n, dtype="string"),
            "source_asset": pd.Series([spec.source_asset if spec.source_asset is not None else pd.NA] * n, dtype="string"),
            "source_table": pd.Series([source_table] * n, dtype="string"),
            "source_file": pd.Series([spec.source_file] * n, dtype="string"),
            "source_sheet": pd.Series([spec.source_sheet] * n, dtype="string"),
            "source_column": pd.Series([spec.source_column] * n, dtype="string"),
            "source_batch_id": batch_s,
            "measured_at": measured_ts,
            "updated_at": pd.Series([utc_now_iso()] * n, dtype="string"),
        }
    )
    return out


def build_clinical_measurement_rows(
    agg: pd.DataFrame,
    spec: DerivedClinicalMeasurementSpec,
) -> pd.DataFrame:
    """Map an aggregate frame to ``clinical_measurements`` columns (one derived variable)."""
    if agg.empty:
        return pd.DataFrame()
    if "subject_uid" not in agg.columns:
        raise ValueError("agg must include a 'subject_uid' column.")
    if spec.value_column not in agg.columns:
        raise ValueError(f"agg is missing value column {spec.value_column!r}.")

    n = len(agg)
    vn = pd.to_numeric(agg[spec.value_column], errors="coerce")
    source_table = spec.source_table or f"{spec.source_file}::{spec.source_sheet}"

    measured_at = spec.measured_at
    if measured_at is None:
        measured_ts = pd.Series([pd.NaT] * n, dtype="datetime64[ns]")
    else:
        measured_ts = pd.Series([measured_at] * n, dtype="datetime64[ns]")

    batch = spec.source_batch_id
    if batch is None:
        batch_s = _series_na_string(n)
    else:
        batch_s = pd.Series([batch] * n, dtype="string")

    visit = _optional_string_col(agg, "visit_id", n)

    out = pd.DataFrame(
        {
            "subject_uid": agg["subject_uid"].astype("string"),
            "visit_id": visit,
            "variable_id": pd.Series([spec.variable_id] * n, dtype="string"),
            "value_num": vn.astype("float64"),
            "value_text": _series_na_string(n),
            "unit": pd.Series([spec.unit if spec.unit is not None else pd.NA] * n, dtype="string"),
            "value_kind": pd.Series([spec.value_kind] * n, dtype="string"),
            "source_table": pd.Series([source_table] * n, dtype="string"),
            "source_file": pd.Series([spec.source_file] * n, dtype="string"),
            "source_sheet": pd.Series([spec.source_sheet] * n, dtype="string"),
            "source_column": pd.Series([spec.source_column] * n, dtype="string"),
            "source_batch_id": batch_s,
            "measured_at": measured_ts,
        }
    )
    return out


def register_derived_variable(
    repo: DataRepo,
    entry: DerivedVariableRegistration | Mapping[str, Any],
) -> None:
    """Register one variable in the dataset catalog (merge semantics by default)."""
    if isinstance(entry, DerivedVariableRegistration):
        repo.register_variables([entry.to_catalog_entry()])
    else:
        repo.register_variables([dict(entry)])


def publish_derived_measurements(
    repo: DataRepo,
    df: pd.DataFrame,
    *,
    table: Literal["image_measurements", "clinical_measurements", "cognitive_measurements"],
    register: DerivedVariableRegistration | Mapping[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    build_sqlite_index: bool = True,
    upsert_key_columns: list[str] | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Upsert derived measurement rows and optionally register catalog metadata.

    Pass the dataframe returned by :func:`build_image_measurement_rows` or
    :func:`build_clinical_measurement_rows` (the latter also matches the
    ``cognitive_measurements`` schema). When ``register`` is set, that entry
    is written with :func:`register_derived_variable` after a successful upsert.
    """
    if df.empty and not allow_empty:
        raise ValueError("df is empty; nothing to publish (or pass allow_empty=True).")
    combined = repo.upsert_table(
        table,
        df,
        key_columns=upsert_key_columns,
        provenance=provenance,
        build_sqlite_index=build_sqlite_index,
    )
    if register is not None:
        register_derived_variable(repo, register)
    return combined
