"""Staged spreadsheet import: edit a table, map its columns, check, then write.

The one-shot importer (:func:`nvitk.db.importers.import_measurements_from_source`) names every
column after itself, cannot give image columns a region or a pipeline version, and overwrites only
rows from a file of the same name. This module is what the GUI's import editor builds on:

* **Table edits** (:func:`apply_ops`): rename, drop / keep, filter rows, replace values, scale,
  compute a column, drop rows with no subject... kept as a list of operations, so an import can
  be replayed on the next delivery of the same spreadsheet (a *recipe*).
* **Column plan** (:class:`ColumnPlan`, :func:`guess_plan`): what each column is (the subject,
  session, visit or date column, a value, or skipped) and, for values, the variable it feeds, the
  region (territory) for image measurements, the unit and the value kind. :func:`split_names`
  splits names like ``FreqTot`` into a variable and a region.
* **Import settings** (:class:`ImportPlan`): clinical / cognitive / image, modality and pipeline
  version (a new id registers a new version in ``measurement_pipelines.json``), the default visit,
  the batch id, and the write mode:

  - ``replace``: a value for the same subject, visit (or pipeline), variable and region replaces
    the one in the dataset, whatever file that came from;
  - ``missing_only``: only values the dataset does not have yet are added;
  - ``keep_both``: the previous behaviour, rows are told apart by source file, sheet and column.

* :func:`build_rows` turns the table and plan into measurement rows, :func:`check_import` reports
  what an import would do (new / replaced / unchanged rows, new variables and regions, unknown
  subjects) without writing, and :func:`apply_import` writes rows, variables, the pipeline entry
  and the recipe, then rebuilds the query index.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .storage import normalize_variable_id, utc_now_iso

#: Measurement kinds and their tables.
KIND_TABLES = {
    "clinical": "clinical_measurements",
    "cognitive": "cognitive_measurements",
    "image": "image_measurements",
    # The subject registry: one row per subject, one column per identifier (nvitk.db.subjects).
    "subjects": "subject_ids",
}

#: What to do with subject ids the dataset does not know (measurement imports).
UNKNOWN_POLICIES = ("auto", "new", "skip")
UNKNOWN_LABELS = {
    "auto": "New subjects if the ids are subject uids, else skip",
    "new": "Add them as new subjects",
    "skip": "Skip their rows",
}

#: What a column can be.
ROLES = ("value", "subject", "session", "visit", "date", "skip")

#: Write modes (see the module docstring).
MODES = ("replace", "missing_only", "keep_both")
MODE_LABELS = {
    "replace": "Replace matching values",
    "missing_only": "Only add missing values",
    "keep_both": "Keep both (rows told apart by source file)",
}

#: Value kinds a column can be forced to.
VALUE_KINDS = ("auto", "numeric", "text")

#: Ready-made name splits for :func:`split_names` (regex with ``variable`` / ``region`` groups).
NAME_PATTERNS = {
    "CamelCase: Variable + Region (FreqTot)": r"^(?P<variable>[A-Z]?[a-z0-9]+)(?P<region>[A-Z].*)$",
    "variable_region (freq_tot)": r"^(?P<variable>.+)_(?P<region>[^_]+)$",
    "region_variable (tot_freq)": r"^(?P<region>[^_]+)_(?P<variable>.+)$",
    "variable.region (freq.tot)": r"^(?P<variable>[^.]+)\.(?P<region>.+)$",
    "Whole name is the region": r"^(?P<region>.+)$",
}


# --------------------------------------------------------------------------- table edits


def apply_op(df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
    """Apply one table edit *op* (``{"op": name, ...}``) to *df*; returns a new frame."""
    kind = op.get("op")
    out = df
    if kind == "rename":
        out = df.rename(columns={op["from"]: op["to"]})
    elif kind == "drop":
        out = df.drop(columns=[c for c in op["columns"] if c in df.columns])
    elif kind == "keep":
        out = df[[c for c in op["columns"] if c in df.columns]]
    elif kind == "query":
        out = df.query(str(op["expr"]), engine="python")
    elif kind == "filter":
        from .frame_filters import filter_mask

        out = df[filter_mask(df, dict(op.get("filters") or {}))]
    elif kind == "dropna":
        cols = [c for c in op.get("columns") or [] if c in df.columns] or None
        out = df.dropna(subset=cols, how="all" if op.get("how") == "all" else "any")
        if cols:
            text = out[cols].astype("string").apply(lambda s: s.str.strip())
            out = out[~(text == "").any(axis=1)]
    elif kind == "replace":
        column = op["column"]
        out = df.copy()
        series = out[column]
        find, repl = op.get("find", ""), op.get("replace", "")
        if op.get("regex"):
            out[column] = series.astype("string").str.replace(str(find), str(repl), regex=True)
        else:
            text = series.astype("string")
            out[column] = series.where(text != str(find), repl if repl != "" else pd.NA)
    elif kind == "scale":
        column = op["column"]
        out = df.copy()
        out[column] = pd.to_numeric(out[column], errors="coerce") * float(op.get("factor", 1.0)) + float(op.get("offset", 0.0))
    elif kind == "compute":
        out = df.copy()
        out[op["column"]] = df.eval(str(op["expr"]), engine="python")
    elif kind == "cast":
        column = op["column"]
        out = df.copy()
        if op.get("to") == "numeric":
            numeric = pd.to_numeric(out[column], errors="coerce")
            # Decimal commas ("1,5"), value by value: a column can mix both.
            comma = pd.to_numeric(out[column].astype("string").str.replace(",", ".", regex=False), errors="coerce")
            out[column] = numeric.fillna(comma).astype("float64")
        else:
            out[column] = out[column].astype("string")
    elif kind == "strip":
        column = op["column"]
        out = df.copy()
        text = out[column].astype("string").str.strip()
        if op.get("prefix"):
            text = text.str.removeprefix(str(op["prefix"]))
        if op.get("suffix"):
            text = text.str.removesuffix(str(op["suffix"]))
        out[column] = text
    elif kind == "duplicate":
        out = df.copy()
        out[op["to"]] = df[op["column"]]
    elif kind == "dedupe":
        cols = [c for c in op.get("columns") or [] if c in df.columns] or None
        out = df.drop_duplicates(subset=cols, keep="last")
    elif kind == "delete_rows":
        positions = sorted({int(i) for i in op.get("rows") or [] if 0 <= int(i) < len(df)})
        out = df.drop(index=df.index[positions])
    elif kind == "set":
        out = df.copy()
        out.iat[int(op["row"]), list(df.columns).index(op["column"])] = op.get("value")
    else:
        raise ValueError(f"Unknown table edit: {kind!r}")
    return out.reset_index(drop=True)


#: Edits tied to one file's rows (a typed cell, deleted rows): not replayed by a recipe.
ROW_SPECIFIC_OPS = frozenset({"set", "delete_rows"})


def apply_ops(df: pd.DataFrame, ops: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Replay *ops* on *df* in order (the recipe's table edits)."""
    out = df.reset_index(drop=True)
    for op in ops:
        out = apply_op(out, op)
    return out


def describe_op(op: dict[str, Any]) -> str:
    """A short human description of one edit, for the history list."""
    kind = op.get("op")
    if kind == "rename":
        return f"Rename {op['from']} → {op['to']}"
    if kind in ("drop", "keep"):
        cols = list(op.get("columns") or [])
        return f"{'Drop' if kind == 'drop' else 'Keep only'} {', '.join(map(str, cols[:4]))}{'…' if len(cols) > 4 else ''}"
    if kind == "query":
        return f"Rows where {op['expr']}"
    if kind == "filter":
        from .frame_filters import describe_filter

        return "Keep rows: " + "; ".join(describe_filter(c, f) for c, f in (op.get("filters") or {}).items())
    if kind == "dropna":
        return f"Drop rows with no {', '.join(map(str, op.get('columns') or ['value']))}"
    if kind == "replace":
        return f"{op['column']}: {op.get('find')!s} → {op.get('replace')!s}"
    if kind == "scale":
        return f"{op['column']} × {op.get('factor', 1)}" + (f" + {op['offset']}" if op.get("offset") else "")
    if kind == "compute":
        return f"{op['column']} = {op['expr']}"
    if kind == "cast":
        return f"{op['column']} as {op.get('to')}"
    if kind == "strip":
        return f"Trim {op['column']}"
    if kind == "duplicate":
        return f"Copy {op['column']} → {op['to']}"
    if kind == "dedupe":
        return "Drop duplicate rows"
    if kind == "delete_rows":
        return f"Delete {len(op.get('rows') or [])} row(s)"
    if kind == "set":
        return f"{op['column']}[{int(op['row']) + 1}] = {op.get('value')}"
    return str(kind)


# --------------------------------------------------------------------------- plan


@dataclass
class ColumnPlan:
    """What one column of the staged table is, and where its values go."""

    column: str
    role: str = "value"
    variable_id: str = ""
    region_id: str = ""
    region_label: str = ""
    unit: str = ""
    label: str = ""
    value_kind: str = "auto"
    frame_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImportPlan:
    """The settings of one import (besides the columns)."""

    kind: str = "clinical"
    source_name: str = ""
    sheet: str = "0"
    batch_id: str = ""
    modality: str = ""
    pipeline_id: str = ""
    pipeline_name: str = ""
    make_default_pipeline: bool = False
    visit_id: str = ""
    mode: str = "replace"
    register_subject_ids: bool = True
    #: Which identifier the subject column holds: ``auto`` (the best-matching namespace),
    #: ``subject_uid`` or a ``subject_ids`` namespace (``mri_id``, ``patient_id``...).
    subject_namespace: str = "auto"
    #: Subject ids the dataset does not know: one of :data:`UNKNOWN_POLICIES`.
    unknown_subjects: str = "auto"
    #: Registry imports: the prefix of generated uids when no column gives them.
    uid_prefix: str = "S"
    columns: list[ColumnPlan] = field(default_factory=list)

    @property
    def table(self) -> str:
        return KIND_TABLES[self.kind]

    @property
    def is_registry(self) -> bool:
        return self.kind == "subjects"

    def value_columns(self) -> list[ColumnPlan]:
        return [c for c in self.columns if c.role == "value" and c.variable_id.strip()]

    def role_column(self, role: str) -> str | None:
        return next((c.column for c in self.columns if c.role == role), None)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["columns"] = [c.to_dict() for c in self.columns]
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImportPlan":
        data = dict(payload)
        columns = [ColumnPlan(**{k: v for k, v in c.items() if k in ColumnPlan.__dataclass_fields__})
                   for c in data.pop("columns", [])]
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known, columns=columns)


def logical_keys(kind: str) -> list[str]:
    """The columns that say "the same value" for a measurement, whatever file it came from."""
    if kind == "image":
        return ["subject_uid", "modality", "region_id", "frame_index", "variable_id", "pipeline_id"]
    return ["subject_uid", "visit_id", "variable_id"]


def _candidates() -> dict[str, list[str]]:
    from nvitk.db.config.dataset_config import load_dataset_config

    cfg = load_dataset_config()
    return {
        "subject": ["subject_uid", *cfg.subject_columns],
        "session": list(cfg.session_columns),
        "visit": list(cfg.visit_columns),
        "date": list(cfg.date_columns),
        "clinical_meta": list(cfg.clinical_metadata_columns),
    }


def _first(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {normalize_variable_id(c): c for c in columns}
    for cand in candidates:
        hit = lookup.get(normalize_variable_id(cand))
        if hit is not None:
            return hit
    return None


def guess_plan(df: pd.DataFrame, kind: str, *, repo: Any = None, source_name: str = "") -> list[ColumnPlan]:
    """A first column plan for *df*: id / visit / session / date columns found by name (the dataset
    config's candidates), identifier-like and ``Unnamed:`` columns skipped, every other column a
    value named after an existing variable when the catalog knows the name (or an alias), else
    after itself."""
    from nvitk.db.importers import _is_identifier_column, _parse_image_wide_column

    if kind == "subjects":
        return _guess_registry_columns(df, repo)
    columns = [str(c) for c in df.columns]
    cands = _candidates()
    roles: dict[str, str] = {}
    for role in ("subject", "session", "visit", "date"):
        hit = _first([c for c in columns if c not in roles], cands[role])
        if hit is not None:
            roles[hit] = role
    if "subject" not in roles.values():
        # No known subject column: the first identifier-like one (``codigo``, ``ID``...) is taken.
        hit = next((c for c in columns if c not in roles and _is_identifier_column(c)), None)
        if hit is None:
            hit = next((c for c in columns if c not in roles and normalize_variable_id(c) in {"id", "codigo", "code", "sujeto"}), None)
        if hit is not None:
            roles[hit] = "subject"
    meta = {normalize_variable_id(c) for c in cands["clinical_meta"]} if kind == "image" else set()
    domain = "image" if kind == "image" else kind
    known: set[str] = set()
    catalog = getattr(repo, "catalog", None)
    if catalog is not None:
        try:
            known = {str(e["variable_id"]) for e in catalog.variable_entries(domain=domain)}
        except Exception:  # noqa: BLE001
            known = set()

    def resolver(name: str) -> str | None:
        """The registered variable *name* is (by id, alias, source column or label), if any."""
        if not known:
            return None
        got = catalog.resolve_variable_ids([name], domain=domain)[0]
        return got if got in known else None

    plans: list[ColumnPlan] = []
    for column in columns:
        if column in roles:
            plans.append(ColumnPlan(column, role=roles[column]))
            continue
        skip = column.startswith("Unnamed:") or _is_identifier_column(column) or normalize_variable_id(column) in meta
        if skip:
            plans.append(ColumnPlan(column, role="skip"))
            continue
        variable, region, region_label = normalize_variable_id(column), "", ""
        if kind == "image" and source_name:
            v, r, rl = _parse_image_wide_column(source_name, column)
            variable, region, region_label = v, r or "", rl or ""
        variable = resolver(column) or resolver(variable) or variable
        plans.append(ColumnPlan(column, variable_id=variable, region_id=region, region_label=region_label, label=column))
    return plans


def _guess_registry_columns(df: pd.DataFrame, repo: Any) -> list[ColumnPlan]:
    """Column plan of a subject registry (:func:`nvitk.db.subjects.guess_registry_plan`): the uid
    column as Subject, identifiers / attributes / the cohort as values named after their namespace."""
    from nvitk.db.subjects import SubjectIndex, guess_registry_plan

    index = None
    if repo is not None:
        try:
            index = SubjectIndex.load(repo)
        except Exception:  # noqa: BLE001
            index = None
    guess = guess_registry_plan(df, index)
    plans: list[ColumnPlan] = []
    for column in (str(c) for c in df.columns):
        if column == guess.uid_column:
            plans.append(ColumnPlan(column, role="subject", label=column))
        elif column in guess.id_columns:
            ns = guess.id_columns[column]
            plans.append(ColumnPlan(column, variable_id=ns, label=column))
        elif column in guess.attribute_columns:
            plans.append(ColumnPlan(column, variable_id=guess.attribute_columns[column], label=column))
        elif column == guess.cohort_column:
            plans.append(ColumnPlan(column, variable_id="cohort", label=column))
        else:
            plans.append(ColumnPlan(column, role="skip", label=column))
    return plans


def split_names(plans: list[ColumnPlan], pattern: str, *, columns: Iterable[str] | None = None,
                normalize: bool = True, region_labels: dict[str, str] | None = None) -> list[str]:
    """Set variable / region of the value columns whose name matches *pattern* (a regex with
    ``variable`` and/or ``region`` named groups); returns the columns it changed.

    ``FreqTot`` under ``^(?P<variable>[A-Z]?[a-z0-9]+)(?P<region>[A-Z].*)$`` becomes variable
    ``freq`` and region ``tot``. *region_labels* maps region ids to labels (``tot`` → "Total brain").
    """
    rx = re.compile(pattern)
    wanted = set(columns) if columns is not None else None
    changed = []
    for plan in plans:
        if plan.role != "value" or (wanted is not None and plan.column not in wanted):
            continue
        m = rx.match(plan.column)
        if not m:
            continue
        groups = m.groupdict()
        if groups.get("variable"):
            plan.variable_id = normalize_variable_id(groups["variable"]) if normalize else groups["variable"]
        if groups.get("region"):
            region = groups["region"]
            plan.region_id = normalize_variable_id(region) if normalize else region
            plan.region_label = (region_labels or {}).get(plan.region_id, plan.region_label or region)
        changed.append(plan.column)
    return changed


# --------------------------------------------------------------------------- rows


def _values(series: pd.Series, kind: str) -> tuple[pd.Series, pd.Series, str]:
    from nvitk.db.importers import _coerce_numeric, _series_value_payload

    if kind == "numeric":
        return _coerce_numeric(series), pd.Series([pd.NA] * len(series), dtype="string"), "numeric"
    if kind == "text":
        text = series.astype("string").where(series.notna(), pd.NA)
        return pd.Series([np.nan] * len(series), dtype="float64"), text, "text"
    return _series_value_payload(series)


def _string_column(df: pd.DataFrame, column: str | None, default: Any = pd.NA) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series([default] * len(df), dtype="string")
    out = df[column]
    if pd.api.types.is_float_dtype(out):
        # 4.0 read from Excel → "4", not "4.0".
        as_int = out.dropna()
        if len(as_int) and bool((as_int == as_int.round()).all()):
            out = out.astype("Int64")
    return out.astype("string").str.strip().replace("", pd.NA)


def resolve_subject_column(df: pd.DataFrame, plan: ImportPlan, index: Any) -> tuple[pd.Series, Any]:
    """``(subject_uid per row, Resolution)`` for the plan's subject column, through the dataset's
    identifiers (:class:`~nvitk.db.subjects.SubjectIndex`): a sheet may name its subjects by any of
    them (an MRI id, a patient code), never only by ``subject_uid``. Unknown ids follow
    ``plan.unknown_subjects``; ids belonging to several subjects stay unresolved."""
    from nvitk.db.subjects import UID

    subject_col = plan.role_column("subject")
    raw = _string_column(df, subject_col)
    resolution = index.resolve(raw, plan.subject_namespace or "auto", column=subject_col)
    uids = resolution.uids.copy()
    policy = plan.unknown_subjects or "auto"
    if policy == "auto":
        policy = "new" if resolution.namespace == UID or index.n_subjects == 0 else "skip"
    if policy == "new" and resolution.unknown:
        unknown = set(resolution.unknown)
        fill = raw.where(raw.isin(unknown))
        uids = uids.fillna(fill)
    resolution.policy = policy
    return uids.astype("string"), resolution


def build_rows(df: pd.DataFrame, plan: ImportPlan, *, index: Any = None) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    """The measurement rows, variable entries and problems for *df* under *plan* (nothing written).

    With *index* (the dataset's :class:`~nvitk.db.subjects.SubjectIndex`) the subject column is
    resolved to ``subject_uid`` (:func:`resolve_subject_column`); without, its values are taken as
    they are."""
    from nvitk.db.importers import _parse_datetime_series

    problems: list[str] = []
    subject_col = plan.role_column("subject")
    if subject_col is None or subject_col not in df.columns:
        return pd.DataFrame(), [], ["No subject column: mark the column holding the subject ids as Subject."]
    values = plan.value_columns()
    if not values:
        return pd.DataFrame(), [], ["No value columns: give at least one column a variable."]
    if plan.kind == "image" and not plan.modality.strip():
        problems.append("Image measurements need a modality.")
    if not plan.batch_id.strip():
        problems.append("Give the import a batch id (its version tag).")
    missing = [c.column for c in values if c.column not in df.columns]
    if missing:
        return pd.DataFrame(), [], [f"Columns no longer in the table: {', '.join(missing)}"]
    subjects = _string_column(df, subject_col)
    blank = int(subjects.isna().sum())
    if blank:
        problems.append(f"{blank} row(s) have no subject id and are skipped.")
    if index is not None:
        subjects, resolution = resolve_subject_column(df, plan, index)
        problems.append(resolution.summary())
        if resolution.unknown:
            problems.append(f"{len(resolution.unknown):,} unknown id(s): "
                            + ("added as new subjects." if resolution.policy == "new" else "their rows are skipped."))
    date_col = plan.role_column("date")
    measured = _parse_datetime_series(df[date_col]) if date_col else pd.Series([pd.NaT] * len(df), dtype="datetime64[ns]")
    source_file = plan.source_name or "spreadsheet"
    source_table = f"{Path(source_file).stem}::{plan.sheet}"
    domain = plan.kind
    frames: list[pd.DataFrame] = []
    entries: list[dict[str, Any]] = []
    seen_keys: dict[tuple, str] = {}
    for col in values:
        num, text, value_kind = _values(df[col.column], col.value_kind)
        variable_id = normalize_variable_id(col.variable_id) or col.variable_id.strip()
        key = (variable_id, col.region_id.strip(), col.frame_index)
        if key in seen_keys:
            problems.append(f"Columns {seen_keys[key]} and {col.column} both feed {variable_id}"
                            f"{' / ' + col.region_id if col.region_id else ''}: the last one wins.")
        seen_keys[key] = col.column
        base = {
            "subject_uid": subjects,
            "variable_id": variable_id,
            "value_num": num,
            "value_text": text,
            "unit": pd.Series([col.unit.strip() or pd.NA] * len(df), dtype="string"),
            "value_kind": value_kind,
            "source_table": source_table,
            "source_file": source_file,
            "source_sheet": str(plan.sheet),
            "source_column": col.column,
            "source_batch_id": plan.batch_id.strip(),
            "measured_at": measured,
        }
        if plan.kind == "image":
            region = col.region_id.strip()
            frame = pd.DataFrame({
                **base,
                "session_id": _string_column(df, plan.role_column("session"), "").fillna(""),
                "modality": plan.modality.strip().lower(),
                "region_id": pd.Series([region or pd.NA] * len(df), dtype="string"),
                "region_label": pd.Series([(col.region_label.strip() or region) or pd.NA] * len(df), dtype="string"),
                "frame_index": pd.Series([col.frame_index if col.frame_index is not None else pd.NA] * len(df), dtype="Int64"),
                "pipeline_name": plan.pipeline_name.strip() or Path(source_file).stem,
                "pipeline_id": pd.Series([plan.pipeline_id.strip() or pd.NA] * len(df), dtype="string"),
                "qc_status": pd.Series([pd.NA] * len(df), dtype="string"),
                "source_asset": pd.Series([pd.NA] * len(df), dtype="string"),
            })
        else:
            visits = _string_column(df, plan.role_column("visit"))
            default_visit = plan.visit_id.strip()
            if default_visit:
                visits = visits.fillna(default_visit)
            frame = pd.DataFrame({**base, "visit_id": visits})
            n_no_visit = int(frame["visit_id"].isna().sum())
            if n_no_visit:
                problems.append(f"{col.column}: {n_no_visit} row(s) have no visit; give a default visit.")
        frame = frame[frame["subject_uid"].notna() & (frame["value_num"].notna() | frame["value_text"].notna())]
        frames.append(frame)
        entry = {
            "variable_id": variable_id,
            "domain": domain,
            "table": plan.table,
            "source_column": col.column,
            "source_file": source_file,
            "source_sheet": str(plan.sheet),
            "aliases": [col.column],
            "label": col.label.strip() or col.column,
            "value_kind": value_kind,
        }
        if col.unit.strip():
            entry["unit"] = col.unit.strip()
        if plan.kind == "image":
            entry["modality"] = plan.modality.strip().lower()
        entries.append(entry)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not rows.empty:
        keys = logical_keys(plan.kind)
        dup = rows.duplicated(subset=keys, keep="last")
        if bool(dup.any()):
            problems.append(f"{int(dup.sum())} row(s) repeat a subject/{'region' if plan.kind == 'image' else 'visit'}"
                            "/variable already in this table: the last one is kept.")
            rows = rows[~dup].reset_index(drop=True)
    return rows, entries, problems


# --------------------------------------------------------------------------- check / apply


@dataclass
class ImportCheck:
    """What an import would do; nothing is written to get it."""

    rows: pd.DataFrame
    entries: list[dict[str, Any]]
    problems: list[str]
    n_new: int = 0
    n_replace: int = 0
    n_same: int = 0
    n_skipped_existing: int = 0
    n_other_sources: int = 0
    unknown_subjects: list[str] = field(default_factory=list)
    new_variables: list[str] = field(default_factory=list)
    new_regions: dict[str, list[str]] = field(default_factory=dict)
    new_pipeline: bool = False
    per_variable: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: How the subject column resolved (measurement imports).
    resolution: Any = None
    #: What a subject-registry import would do (:class:`~nvitk.db.subjects.RegistryResult`).
    registry: Any = None

    @property
    def ok(self) -> bool:
        if self.registry is not None:
            return bool(self.registry.new_subjects or self.registry.updated_subjects)
        return not self.rows.empty and not any(p.startswith(("No ", "Columns no longer", "Image measurements need", "Give the import"))
                                               for p in self.problems)

    def summary(self, plan: ImportPlan) -> str:
        if self.registry is not None:
            return "\n".join([self.registry.summary(), *self.problems])
        if self.rows.empty:
            return "Nothing to import.\n" + "\n".join(self.problems)
        lines = [f"{len(self.rows):,} value(s) for {self.rows['subject_uid'].nunique():,} subject(s) → {plan.table}."]
        if plan.mode == "missing_only":
            lines.append(f"New: {self.n_new:,}; already in the dataset (left alone): {self.n_skipped_existing:,}.")
        elif plan.mode == "keep_both":
            lines.append(f"New: {self.n_new:,}; replacing rows of this same file: {self.n_replace:,}"
                         f"; unchanged: {self.n_same:,}.")
            if self.n_other_sources:
                lines.append(f"{self.n_other_sources:,} value(s) also exist from another file: both copies are kept, "
                             "and wide tables show only one of them.")
        else:
            lines.append(f"New: {self.n_new:,}; replacing a different value: {self.n_replace:,}; unchanged: {self.n_same:,}.")
        if self.new_variables:
            lines.append("New variables: " + ", ".join(self.new_variables[:12]) + ("…" if len(self.new_variables) > 12 else ""))
        for variable, regions in list(self.new_regions.items())[:8]:
            lines.append(f"New regions for {variable}: {', '.join(regions[:10])}{'…' if len(regions) > 10 else ''}")
        if self.new_pipeline:
            lines.append(f"New pipeline version {plan.pipeline_id} ({plan.modality}) is registered"
                         f"{' as the default' if plan.make_default_pipeline else ''}.")
        if self.unknown_subjects:
            lines.append(f"{len(self.unknown_subjects)} subject id(s) are not in the dataset yet "
                         f"(e.g. {', '.join(self.unknown_subjects[:3])}).")
        lines += self.problems
        return "\n".join(lines)


def _keys(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    """One string per row joining *keys* (vectorised: millions of rows in a second or two)."""
    parts = _key_frame(frame, keys)
    out = pd.Series([""] * len(frame), index=frame.index, dtype="string")
    for i, key in enumerate(keys):
        out = parts[key] if i == 0 else out + "\x1f" + parts[key]
    return out.astype(str)


def _key_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for key in keys:
        if key not in frame.columns:
            out[key] = ""
        elif key == "frame_index":
            out[key] = pd.to_numeric(frame[key], errors="coerce").astype("Int64").astype("string").fillna("")
        else:
            out[key] = frame[key].astype("string").str.strip().fillna("")
    return out


def _value_text(frame: pd.DataFrame) -> pd.Series:
    num = pd.to_numeric(frame.get("value_num"), errors="coerce")
    text = frame.get("value_text")
    out = num.map(lambda v: "" if pd.isna(v) else f"{float(v):.10g}")
    if text is not None:
        out = out.where(out != "", text.astype("string").fillna(""))
    return out.astype("string")


def _existing(repo: Any, plan: ImportPlan, variables: list[str]) -> pd.DataFrame:
    if not repo.catalog.table_exists(plan.table):
        return pd.DataFrame()
    filters = {"variable_id": variables} if variables else None
    try:
        frame = repo.get(plan.table, filters=filters, cohort_id=False, use_sqlite=False)
    except Exception:  # noqa: BLE001 (a filter this table cannot take)
        frame = repo.get(plan.table, cohort_id=False, use_sqlite=False)
    if not frame.empty and variables:
        frame = frame[frame["variable_id"].astype("string").isin(variables)]
    return frame


def _subject_index(repo: Any) -> Any:
    """The dataset's subject identifiers, or ``None`` without a dataset."""
    if repo is None:
        return None
    from nvitk.db.subjects import SubjectIndex

    try:
        return SubjectIndex.load(repo)
    except Exception:  # noqa: BLE001 (a dataset without the id tables)
        return None


def registry_plan(df: pd.DataFrame, plan: ImportPlan) -> Any:
    """The :class:`~nvitk.db.subjects.RegistryPlan` a ``subjects`` import plan describes: the
    Subject column gives new subjects their uid, every value column is an identifier whose
    namespace is its Variable (``sex`` / ``birth_date`` are kept as attributes, ``cohort`` as the
    cohort the subject joins)."""
    from nvitk.db.subjects import RegistryPlan, namespace_of

    out = RegistryPlan(uid_column=plan.role_column("subject") or "", uid_prefix=plan.uid_prefix or "S",
                       batch_id=plan.batch_id.strip(), source_name=plan.source_name, sheet=str(plan.sheet),
                       cohort_id="")
    for col in plan.columns:
        if col.role != "value" or col.column not in df.columns:
            continue
        ns = namespace_of(col.variable_id or col.column)
        if ns in ("sex", "birth_date"):
            out.attribute_columns[col.column] = ns
        elif ns in ("cohort", "cohort_id"):
            out.cohort_column = col.column
        else:
            out.id_columns[col.column] = ns
    return out


def _check_registry(repo: Any, df: pd.DataFrame, plan: ImportPlan) -> ImportCheck:
    from nvitk.db.subjects import SubjectIndex, plan_registry

    rplan = registry_plan(df, plan)
    problems = []
    if not rplan.id_columns and not rplan.uid_column:
        problems.append("No identifier columns: mark the uid column as Subject and each identifier as a value "
                        "named after its namespace (mri_id, patient_id…).")
    if not plan.batch_id.strip():
        problems.append("Give the import a batch id (its version tag).")
    index = SubjectIndex.load(repo) if repo is not None else SubjectIndex()
    result = plan_registry(index, df, rplan)
    check = ImportCheck(rows=result.id_rows, entries=[], problems=problems, registry=result)
    check.n_new = len(result.id_rows)
    return check


def _apply_registry(repo: Any, df: pd.DataFrame, plan: ImportPlan, *, ops: list[dict[str, Any]] | None,
                    rebuild_index: bool, recipe_name: str, say: Any) -> dict[str, Any]:
    from nvitk.db.subjects import register_subjects

    say("Registering the subjects and their identifiers…")
    result = register_subjects(repo, df, registry_plan(df, plan), rebuild_index=rebuild_index)
    out: dict[str, Any] = {"table": "subject_ids", "rows": int(len(result.id_rows)), "written": int(len(result.id_rows)),
                           "new_subjects": len(result.new_subjects), "updated_subjects": len(result.updated_subjects),
                           "conflicts": len(result.conflicts), "summary": result.summary()}
    if recipe_name:
        out["recipe"] = str(save_recipe(repo, recipe_name, plan, ops or []))
    return out


def check_import(repo: Any, df: pd.DataFrame, plan: ImportPlan) -> ImportCheck:
    """Build the rows and compare them with the dataset (reads, never writes)."""
    if plan.is_registry:
        return _check_registry(repo, df, plan)
    index = _subject_index(repo)
    rows, entries, problems = build_rows(df, plan, index=index)
    check = ImportCheck(rows=rows, entries=entries, problems=problems)
    if index is not None and plan.role_column("subject"):
        check.resolution = resolve_subject_column(df, plan, index)[1]
    if rows.empty or repo is None:
        return check
    variables = sorted(rows["variable_id"].astype(str).unique())
    existing = _existing(repo, plan, variables)
    keys = logical_keys(plan.kind)
    new_keys = _keys(rows, keys)
    new_vals = _value_text(rows)
    if not existing.empty:
        old_keys = _keys(existing, keys)
        old_vals = dict(zip(old_keys, _value_text(existing)))
        exists = new_keys.isin(set(old_keys))
        if plan.mode == "keep_both":
            source_keys = keys + ["source_file", "source_sheet", "source_column"]
            old_src = set(_keys(existing, source_keys))
            same_src = _keys(rows, source_keys).isin(old_src)
            same_val = pd.Series([old_vals.get(k) == v for k, v in zip(new_keys, new_vals)], index=rows.index)
            check.n_same = int((same_src & same_val).sum())
            check.n_replace = int((same_src & ~same_val).sum())
            check.n_new = int((~same_src).sum())
            check.n_other_sources = int((exists & ~same_src).sum())
        else:
            same = pd.Series([old_vals.get(k) == v for k, v in zip(new_keys, new_vals)], index=rows.index) & exists
            check.n_same = int(same.sum())
            check.n_replace = int((exists & ~same).sum())
            check.n_new = int((~exists).sum())
            if plan.mode == "missing_only":
                check.n_skipped_existing = int(exists.sum())
                check.n_replace = check.n_same = 0
    else:
        check.n_new = len(rows)
    known = {str(e["variable_id"]) for e in repo.catalog.variable_entries(domain=plan.kind)}
    check.new_variables = [v for v in variables if v not in known]
    if plan.kind == "image":
        old_regions: dict[str, set[str]] = {}
        if not existing.empty:
            sub = existing[existing["modality"].astype("string").str.lower() == plan.modality.strip().lower()]
            for variable, group in sub.groupby(sub["variable_id"].astype(str)):
                old_regions[variable] = set(group["region_id"].astype("string").fillna("").astype(str))
        for variable, group in rows.groupby(rows["variable_id"].astype(str)):
            regions = sorted(set(group["region_id"].astype("string").fillna("").astype(str)) - old_regions.get(variable, set()))
            if variable not in check.new_variables and regions:
                check.new_regions[variable] = [r or "(none)" for r in regions]
        if plan.pipeline_id.strip():
            check.new_pipeline = plan.pipeline_id.strip() not in repo.catalog.all_pipeline_ids()
        else:
            check.problems.append("No pipeline version: these rows will not match queries that ask for one.")
    try:
        subjects = repo.get("subjects", columns=["subject_uid"], cohort_id=False)
        known_subjects = set(subjects["subject_uid"].astype(str))
        if known_subjects:
            check.unknown_subjects = sorted(set(rows["subject_uid"].astype(str)) - known_subjects)
    except Exception:  # noqa: BLE001
        pass
    group_cols = ["variable_id", "region_id"] if plan.kind == "image" else ["variable_id"]
    per = rows.groupby(group_cols, dropna=False).agg(n=("subject_uid", "size"), subjects=("subject_uid", "nunique"),
                                                     mean=("value_num", "mean")).reset_index()
    check.per_variable = per
    return check


def register_pipeline(repo: Any, pipeline_id: str, modality: str, *, pipeline_name: str = "",
                      make_default: bool = False) -> bool:
    """Add *pipeline_id* to ``measurement_pipelines.json`` (as the default of *modality* when
    *make_default*); returns whether the manifest changed."""
    catalog = repo.catalog
    if make_default:
        return bool(catalog.register_default_pipeline({"pipeline_id": pipeline_id, "modality": modality,
                                                       "aliases": ["latest"], "pipeline_name": pipeline_name or None}))
    if pipeline_id in catalog.all_pipeline_ids():
        return False
    import copy

    from .storage import write_json

    if catalog.pipelines_manifest_path is None:
        raise RuntimeError("The dataset has no pipelines manifest (repository.json: pipelines_manifest).")
    payload = copy.deepcopy(catalog.pipelines_manifest)
    entry = {"pipeline_id": pipeline_id, "modality": modality, "is_default": False, "aliases": []}
    if pipeline_name:
        entry["pipeline_name"] = pipeline_name
    payload.setdefault("pipelines", []).append(entry)
    catalog.validate_pipelines_manifest(payload)
    write_json(catalog.pipelines_manifest_path, payload)
    catalog.pipelines_manifest = payload
    return True


def _variable_entries_for_write(repo: Any, entries: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    """New variables get their full entry; existing ones only gain the column as an alias (their
    label, source and unit stay as registered)."""
    known = {str(e["variable_id"]): e for e in repo.catalog.variable_entries(domain=domain)}
    out = []
    merged: dict[str, dict[str, Any]] = {}
    for entry in entries:
        vid = entry["variable_id"]
        if vid in known:
            item = merged.setdefault(vid, {"variable_id": vid, "domain": known[vid].get("domain", domain),
                                           "table": known[vid].get("table", entry["table"]), "aliases": []})
            item["aliases"] = sorted(set(item["aliases"]) | set(entry.get("aliases") or []))
        else:
            item = merged.get(vid)
            if item is None:
                merged[vid] = dict(entry)
            else:
                item["aliases"] = sorted(set(item.get("aliases") or []) | set(entry.get("aliases") or []))
    out.extend(merged.values())
    return out


def apply_import(repo: Any, df: pd.DataFrame, plan: ImportPlan, *, ops: list[dict[str, Any]] | None = None,
                 rebuild_index: bool = True, recipe_name: str = "", progress: Any = None) -> dict[str, Any]:
    """Write the import: pipeline entry, measurement rows (per the write mode), variables, subject
    ids, the recipe; then the query index. Returns counts."""
    say = progress or (lambda _m: None)
    if plan.is_registry:
        return _apply_registry(repo, df, plan, ops=ops, rebuild_index=rebuild_index, recipe_name=recipe_name, say=say)
    index = _subject_index(repo)
    rows, entries, problems = build_rows(df, plan, index=index)
    if rows.empty:
        raise ValueError("; ".join(problems) or "Nothing to import.")
    table = plan.table
    result: dict[str, Any] = {"table": table, "rows": int(len(rows))}
    if plan.kind == "image" and plan.pipeline_id.strip():
        say("Registering the pipeline version…")
        result["pipeline_registered"] = register_pipeline(
            repo, plan.pipeline_id.strip(), plan.modality.strip().lower(),
            pipeline_name=plan.pipeline_name.strip(), make_default=plan.make_default_pipeline)
    say(f"Writing {len(rows):,} rows to {table}…")
    provenance = {"importer": "nvitk staged import", "source_file": plan.source_name, "source_batch_id": plan.batch_id,
                  "mode": plan.mode, "imported_at": utc_now_iso()}
    keys = logical_keys(plan.kind)
    if plan.mode == "keep_both":
        repo.upsert_table(table, rows, provenance=provenance)
        result["written"] = int(len(rows))
    else:
        rows = repo._enforce_measurement_columns(table, rows) if hasattr(repo, "_enforce_measurement_columns") else rows
        existing = repo.get(table, cohort_id=False, use_sqlite=False) if repo.catalog.table_exists(table) else pd.DataFrame()
        if existing.empty:
            combined = rows
            result["written"] = int(len(rows))
        else:
            # Only rows of the imported variables can match (variable_id is part of the key).
            variables = set(rows["variable_id"].astype(str))
            candidate = existing["variable_id"].astype(str).isin(variables).to_numpy()
            new_keys = set(_keys(rows, keys))
            hit = pd.Series(False, index=existing.index)
            if candidate.any():
                hit.loc[candidate] = _keys(existing.loc[candidate], keys).isin(new_keys).to_numpy()
            if plan.mode == "replace":
                result["replaced"] = int(hit.sum())
                combined = pd.concat([existing[~hit], rows], ignore_index=True)
                result["written"] = int(len(rows))
            else:  # missing_only
                have = set(_keys(existing.loc[candidate], keys)) if candidate.any() else set()
                fresh = rows[~_keys(rows, keys).isin(have).to_numpy()]
                combined = pd.concat([existing, fresh], ignore_index=True)
                result["written"] = int(len(fresh))
        repo.write_table(table, combined, provenance=provenance)
    say("Registering the variables…")
    repo.register_variables(_variable_entries_for_write(repo, entries, plan.kind))
    if index is not None:
        added = sorted({str(u) for u in rows["subject_uid"].dropna().astype(str)} - set(index._uids))
        result["new_subjects"] = len(added)
    if plan.register_subject_ids:
        try:
            from nvitk.db.importers import harvest_subject_ids_from_frame

            subject_col = plan.role_column("subject")
            id_frame = df.copy()
            # The resolved subject, not the id the sheet used: the sheet's id columns (the one it
            # named subjects by included) are recorded as that subject's identifiers.
            id_frame["subject_uid"] = (resolve_subject_column(df, plan, index)[0] if index is not None
                                       else _string_column(df, subject_col))
            harvest_subject_ids_from_frame(repo, id_frame, source_path=Path(plan.source_name or "spreadsheet"),
                                           sheet_name=str(plan.sheet), source_batch_id=plan.batch_id,
                                           id_source="nvitk_import")
        except Exception as exc:  # noqa: BLE001 (an id table this dataset lacks)
            result["subject_ids_error"] = str(exc)
    if result.get("new_subjects"):
        # Unknown ids taken as new subjects: in the subject list, like the ones already there.
        try:
            from nvitk.db.importers import rebuild_subjects_table

            rebuild_subjects_table(repo)
        except Exception as exc:  # noqa: BLE001
            result["subjects_error"] = str(exc)
    if recipe_name:
        result["recipe"] = str(save_recipe(repo, recipe_name, plan, ops or []))
    if rebuild_index:
        say("Rebuilding the query index…")
        repo.build_sqlite_index()
    return result


# --------------------------------------------------------------------------- recipes


def recipes_dir(repo: Any) -> Path:
    """Where the dataset keeps import recipes: ``<root>/catalog/import_recipes``."""
    return Path(repo.root) / "catalog" / "import_recipes"


def _recipe_file(repo: Any, name: str) -> Path:
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip()).strip("_") or "recipe"
    return recipes_dir(repo) / f"{stem}.json"


def save_recipe(repo: Any, name: str, plan: ImportPlan, ops: list[dict[str, Any]], *, header: int = 0) -> Path:
    """Store *plan* and the table edits *ops* as a recipe the next import of this layout can reuse."""
    path = _recipe_file(repo, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    replayable = [op for op in ops if op.get("op") not in ROW_SPECIFIC_OPS]
    payload = {"name": name, "saved_at": utc_now_iso(), "header": int(header), "ops": replayable, "plan": plan.to_dict()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def list_recipes(repo: Any) -> list[str]:
    """Names of the dataset's saved recipes."""
    folder = recipes_dir(repo)
    if not folder.is_dir():
        return []
    names = []
    for path in sorted(folder.glob("*.json")):
        try:
            names.append(str(json.loads(path.read_text(encoding="utf-8")).get("name") or path.stem))
        except Exception:  # noqa: BLE001
            continue
    return names


def load_recipe(repo: Any, name: str) -> dict[str, Any]:
    """A recipe: ``{"name", "ops", "plan": ImportPlan, "header"}``."""
    path = _recipe_file(repo, name)
    if not path.exists():
        for candidate in recipes_dir(repo).glob("*.json"):
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if payload.get("name") == name:
                path = candidate
                break
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["plan"] = ImportPlan.from_dict(payload.get("plan") or {})
    return payload


def known_regions(repo: Any, modality: str = "") -> pd.DataFrame:
    """Distinct ``(modality, variable_id, region_id, region_label, pipeline_id)`` of the image table."""
    if not repo.catalog.table_exists("image_measurements"):
        return pd.DataFrame(columns=["modality", "variable_id", "region_id", "region_label", "pipeline_id"])
    cols = ["modality", "variable_id", "region_id", "region_label", "pipeline_id"]
    filters = {"modality": modality} if modality else None
    try:
        frame = repo.get("image_measurements", columns=cols, filters=filters, cohort_id=False)
    except Exception:  # noqa: BLE001
        frame = repo.get("image_measurements", columns=cols, cohort_id=False)
    return frame.drop_duplicates().reset_index(drop=True)


__all__ = [
    "ColumnPlan",
    "ImportCheck",
    "ImportPlan",
    "KIND_TABLES",
    "MODES",
    "MODE_LABELS",
    "NAME_PATTERNS",
    "ROLES",
    "ROW_SPECIFIC_OPS",
    "VALUE_KINDS",
    "apply_import",
    "apply_op",
    "apply_ops",
    "build_rows",
    "check_import",
    "describe_op",
    "guess_plan",
    "known_regions",
    "list_recipes",
    "load_recipe",
    "logical_keys",
    "recipes_dir",
    "register_pipeline",
    "save_recipe",
    "split_names",
]
