"""Who a row is about: one ``subject_uid`` per subject, every other identifier pointing at it.

A dataset keeps each subject once, in ``subjects`` (``subject_uid``), and every identifier the
subject goes by elsewhere (the patient id, the MRI id, the XNAT label, the study number, a
hospital code...) as a row of ``subject_ids``: ``(subject_uid, id_namespace, id_value)``. A
spreadsheet, an XNAT project or a folder of scans may name the subjects by any of those, so
nothing that brings data in may take the id it finds as the ``subject_uid``: it resolves it
first.

* :class:`SubjectIndex` holds the dataset's identifiers and resolves values to subjects, in one
  namespace or, with ``"auto"``, in whichever namespace the values match best (a column named
  ``mri_id`` holding MRI ids resolves through the ``mri_id`` namespace; one named ``codigo``
  holding patient ids, through ``patient_id``). Values are compared loosely: case, spacing, an
  Excel ``.0`` and leading zeros of a number do not matter.
* :func:`register_subjects` builds or extends the dataset's subject list from a *registry*: one
  row per subject, one column per identifier. A row whose identifiers already belong to a subject
  adds the ones it lacks; a row nobody knows becomes a new subject; a row whose identifiers
  belong to two different subjects is a conflict and is reported, never merged.
* :func:`resolve_subject_label`: the one-value form the XNAT and folder indexers use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from .storage import normalize_string, normalize_variable_id, utc_now_iso

#: The namespace of the ``subject_uid`` itself.
UID = "subject_uid"
#: ``subject_ids`` namespaces that are attributes, not identifiers (never used to resolve).
ATTRIBUTE_NAMESPACES = frozenset({"sex", "birth_date", "fechanacimiento", "age"})
#: Where a registry's rows say they came from (``subject_ids.id_source``).
REGISTRY_SOURCE = "subject_registry"

_NUMBER = re.compile(r"^[+-]?\d+(\.0+)?$")


def id_key(value: Any) -> str | None:
    """The form identifiers are compared in: trimmed, case-folded, inner spaces collapsed; a
    number loses an Excel ``.0`` and its leading zeros (``"00123"``, ``123.0`` and ``"123"`` are
    one id)."""
    text = normalize_string(value)
    if text is None:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip().casefold()
    if not text or text in {"nan", "none", "null", "na", "n/a"}:
        return None
    if _NUMBER.match(text):
        text = text.split(".")[0].lstrip("+")
        negative = text.startswith("-")
        digits = text.lstrip("-").lstrip("0") or "0"
        text = f"-{digits}" if negative else digits
    return text


#: Column names meaning a namespace nvitk itself writes (XNAT sync records ``xnat_subject``).
NAMESPACE_ALIASES = {"xnat_label": "xnat_subject", "xnat": "xnat_subject", "xnat_id": "xnat_subject",
                     "xnat_subject_label": "xnat_subject", "xnat_subject_id": "xnat_subject"}
#: Words that make a registry column an identifier.
_ID_WORDS = frozenset({"id", "uid", "ids", "label", "code", "codigo", "nhc", "seqn", "number", "num", "nr", "dni",
                       "mrn", "accession"})


def namespace_of(column: str) -> str:
    """The ``subject_ids`` namespace a column named *column* holds (``"MRI ID"`` -> ``mri_id``,
    ``"XNAT label"`` -> ``xnat_subject``, the namespace the XNAT sync records)."""
    ns = normalize_variable_id(str(column)) or str(column).strip().lower()
    return NAMESPACE_ALIASES.get(ns, ns)


def looks_like_identifier(column: str) -> bool:
    """Whether a registry column holds an identifier (``mri_id``, ``XNAT label``, ``Codigo``...)."""
    from .importers import _is_identifier_column

    if _is_identifier_column(str(column)):
        return True
    words = set(re.split(r"[_\W]+", normalize_variable_id(str(column)) or ""))
    return bool(words & _ID_WORDS)


@dataclass
class Resolution:
    """How a column of ids resolved (see :meth:`SubjectIndex.resolve`)."""

    #: The ``subject_uid`` per row; ``<NA>`` where unknown (or ambiguous).
    uids: pd.Series
    namespace: str
    n_rows: int = 0
    n_matched: int = 0
    #: Ids no subject has (in the namespace used).
    unknown: list[str] = field(default_factory=list)
    #: Ids that belong to more than one subject: ``{id: [uids]}``; their rows are not resolved.
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    #: Match counts of the namespaces tried (``"auto"``), best first.
    tried: list[tuple[str, int]] = field(default_factory=list)
    #: What the importer did with the unknown ids (``new`` / ``skip``), once decided.
    policy: str = ""

    def summary(self) -> str:
        parts = [f"Subject ids read as “{self.namespace}”: {self.n_matched:,} of {self.n_rows:,} row(s) match a subject"]
        if self.unknown:
            parts.append(f"{len(self.unknown):,} id(s) match none (e.g. {', '.join(self.unknown[:3])})")
        if self.ambiguous:
            first = next(iter(self.ambiguous.items()))
            parts.append(f"{len(self.ambiguous):,} id(s) belong to several subjects and are left out "
                         f"(e.g. {first[0]} → {', '.join(first[1][:3])})")
        return "; ".join(parts) + "."


class SubjectIndex:
    """The dataset's identifiers, ready to resolve (read once; cheap to keep for an import)."""

    def __init__(self, ids: pd.DataFrame | None = None, uids: Iterable[str] = ()) -> None:
        #: namespace -> id key -> uids
        self._by_ns: dict[str, dict[str, set[str]]] = {}
        #: every subject_uid the dataset has
        self._uids: set[str] = {str(u) for u in uids if normalize_string(u) is not None}
        ids = ids if ids is not None else pd.DataFrame()
        if not ids.empty and {"subject_uid", "id_namespace", "id_value"} <= set(ids.columns):
            frame = ids[["subject_uid", "id_namespace", "id_value"]].dropna()
            for uid, ns, value in frame.itertuples(index=False):
                uid = normalize_string(uid)
                key = id_key(value)
                if uid is None or key is None:
                    continue
                self._uids.add(str(uid))
                ns = str(ns)
                if ns in ATTRIBUTE_NAMESPACES:
                    continue
                self._by_ns.setdefault(ns, {}).setdefault(key, set()).add(str(uid))
        own = self._by_ns.setdefault(UID, {})
        for uid in self._uids:
            key = id_key(uid)
            if key is not None:
                own.setdefault(key, set()).add(uid)

    @classmethod
    def load(cls, repo: Any) -> "SubjectIndex":
        """The index of *repo*'s ``subject_ids`` and ``subjects``."""
        def table(name: str, columns: list[str]) -> pd.DataFrame:
            try:
                if not repo.catalog.table_exists(name):
                    return pd.DataFrame()
                frame = repo.get(name, cohort_id=False, use_sqlite=False)
            except TypeError:
                frame = repo.get(name)
            except Exception:  # noqa: BLE001 (an empty or new dataset)
                return pd.DataFrame()
            return frame[[c for c in columns if c in frame.columns]] if not frame.empty else frame

        ids = table("subject_ids", ["subject_uid", "id_namespace", "id_value"])
        subjects = table("subjects", ["subject_uid"])
        uids = subjects["subject_uid"].dropna().astype(str).tolist() if not subjects.empty else []
        return cls(ids, uids)

    # what is known

    @property
    def n_subjects(self) -> int:
        return len(self._uids)

    def has_subject(self, uid: Any) -> bool:
        return normalize_string(uid) is not None and str(normalize_string(uid)) in self._uids

    def namespaces(self) -> list[tuple[str, int]]:
        """``(namespace, number of ids)`` of the identifiers known, ``subject_uid`` first."""
        out = [(ns, len(keys)) for ns, keys in self._by_ns.items() if keys]
        out.sort(key=lambda item: (item[0] != UID, -item[1], item[0]))
        return out

    def lookup(self, value: Any, namespace: str = "any") -> set[str]:
        """The subjects *value* names in *namespace* (``"any"``: in every namespace)."""
        key = id_key(value)
        if key is None:
            return set()
        if namespace == "any":
            found: set[str] = set()
            for keys in self._by_ns.values():
                found |= keys.get(key, set())
            return found
        return set(self._by_ns.get(namespace, {}).get(key, set()))

    def match_counts(self, values: Iterable[Any]) -> list[tuple[str, int]]:
        """How many of the distinct *values* each namespace knows, best first."""
        keys = {k for k in (id_key(v) for v in values) if k is not None}
        counts = [(ns, sum(1 for k in keys if k in known)) for ns, known in self._by_ns.items()]
        counts = [c for c in counts if c[1] > 0]
        counts.sort(key=lambda item: (-item[1], item[0] != UID))
        return counts

    def resolve(self, values: pd.Series | Iterable[Any], namespace: str = "auto", *,
                column: str | None = None) -> Resolution:
        """Each of *values* as a ``subject_uid`` (``<NA>`` when unknown or ambiguous).

        *namespace* ``"auto"`` picks the namespace that knows the most of the values: the one
        named after *column* when it ties, then ``subject_uid``. In an empty dataset (nothing
        known) it is ``subject_uid``: every id is unknown.
        """
        series = pd.Series(values) if not isinstance(values, pd.Series) else values
        tried: list[tuple[str, int]] = []
        if namespace in ("", "auto", None):
            tried = self.match_counts(series.dropna().unique())
            own = namespace_of(column) if column else None
            namespace = UID
            if tried:
                best = tried[0][1]
                tops = [ns for ns, n in tried if n == best]
                namespace = own if own in tops else (UID if UID in tops else tops[0])
        table = self._by_ns.get(namespace, {})
        out: list[Any] = []
        unknown: dict[str, None] = {}
        ambiguous: dict[str, list[str]] = {}
        for value in series:
            key = id_key(value)
            if key is None:
                out.append(pd.NA)
                continue
            hits = table.get(key, set())
            if len(hits) == 1:
                out.append(next(iter(hits)))
            else:
                out.append(pd.NA)
                text = str(normalize_string(value))
                if hits:
                    ambiguous[text] = sorted(hits)
                else:
                    unknown[text] = None
        uids = pd.Series(out, index=series.index, dtype="string")
        n_rows = int(series.map(lambda v: id_key(v) is not None).sum())
        return Resolution(uids=uids, namespace=namespace, n_rows=n_rows, n_matched=int(uids.notna().sum()),
                          unknown=list(unknown), ambiguous=ambiguous, tried=tried)


def resolve_subject_label(index: SubjectIndex, label: Any, *, prefer: Iterable[str] = ()) -> str:
    """The ``subject_uid`` an XNAT label or a folder name stands for: the subject one of the
    *prefer* namespaces (then any) names unambiguously, else the label itself (a new subject)."""
    text = str(normalize_string(label) or label)
    for ns in (*prefer, UID):
        hits = index.lookup(text, ns)
        if len(hits) == 1:
            return next(iter(hits))
    hits = index.lookup(text, "any")
    return next(iter(hits)) if len(hits) == 1 else text


# --------------------------------------------------------------------------- registry


@dataclass
class RegistryPlan:
    """How a registry sheet maps onto subjects."""

    #: Column whose value is the ``subject_uid`` of a new subject (``""``: :attr:`uid_prefix` + a number).
    uid_column: str = ""
    #: Identifier columns -> namespace (the uid column may be listed too).
    id_columns: dict[str, str] = field(default_factory=dict)
    #: Attribute columns -> namespace (``sex``, ``birth_date``): kept in ``subject_ids`` too.
    attribute_columns: dict[str, str] = field(default_factory=dict)
    cohort_column: str = ""
    cohort_id: str = ""
    uid_prefix: str = "S"
    batch_id: str = ""
    source_name: str = ""
    sheet: str = "0"


@dataclass
class RegistryResult:
    """What a registry import does (or did)."""

    new_subjects: list[str] = field(default_factory=list)
    updated_subjects: list[str] = field(default_factory=list)
    unchanged_subjects: list[str] = field(default_factory=list)
    id_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    cohort_rows: list[tuple[str, str]] = field(default_factory=list)
    #: ``(row number, reason)`` of the rows left out.
    conflicts: list[tuple[int, str]] = field(default_factory=list)
    skipped_blank: int = 0

    def summary(self) -> str:
        lines = [f"{len(self.new_subjects):,} new subject(s), {len(self.updated_subjects):,} existing subject(s) "
                 f"get new identifiers, {len(self.unchanged_subjects):,} already complete."]
        if not self.id_rows.empty:
            per = self.id_rows.groupby("id_namespace").size().sort_values(ascending=False)
            lines.append("Identifiers added: " + ", ".join(f"{ns} {n:,}" for ns, n in per.items()))
        if self.cohort_rows:
            lines.append(f"{len(self.cohort_rows):,} cohort membership(s).")
        if self.skipped_blank:
            lines.append(f"{self.skipped_blank:,} row(s) carry no identifier and are skipped.")
        if self.conflicts:
            lines.append(f"{len(self.conflicts):,} row(s) are left out, their identifiers disagree with the dataset:")
            lines += [f"  row {r}: {why}" for r, why in self.conflicts[:12]]
            if len(self.conflicts) > 12:
                lines.append(f"  … and {len(self.conflicts) - 12} more")
        return "\n".join(lines)


def _next_uid(prefix: str, taken: set[str]) -> str:
    n = 1
    width = max(4, len(str(len(taken) + 1)))
    while True:
        uid = f"{prefix}{n:0{width}d}"
        if uid not in taken:
            return uid
        n += 1


def plan_registry(index: SubjectIndex, frame: pd.DataFrame, plan: RegistryPlan) -> RegistryResult:
    """What :func:`register_subjects` would do with *frame* (reads nothing else, writes nothing)."""
    result = RegistryResult()
    now = utc_now_iso()
    id_cols = dict(plan.id_columns)
    if plan.uid_column and plan.uid_column not in id_cols:
        id_cols[plan.uid_column] = namespace_of(plan.uid_column)
    taken = set(index._uids)
    rows: list[dict[str, Any]] = []
    #: ids given to subjects earlier in this same sheet (a sheet may repeat a subject)
    claimed: dict[tuple[str, str], str] = {}
    known_pairs = {(ns, key): uids for ns, keys in index._by_ns.items() for key, uids in keys.items()}
    for number, (_i, row) in enumerate(frame.iterrows(), start=2):
        pairs = []
        for column, ns in id_cols.items():
            if column not in frame.columns:
                continue
            value = normalize_string(row.get(column))
            if value is not None and id_key(value) is not None:
                pairs.append((str(ns), str(value)))
        if not pairs:
            result.skipped_blank += 1
            continue
        owners: set[str] = set()
        for ns, value in pairs:
            key = id_key(value)
            owners |= known_pairs.get((ns, key), set())
            if (ns, key) in claimed:
                owners.add(claimed[(ns, key)])
        if plan.uid_column:
            uid_value = normalize_string(row.get(plan.uid_column))
            if uid_value is not None and index.has_subject(uid_value):
                owners.add(str(uid_value))
        if len(owners) > 1:
            result.conflicts.append((number, "its identifiers belong to different subjects ("
                                     + ", ".join(sorted(owners)[:4]) + ")"))
            continue
        if owners and plan.uid_column:
            # The uid column is the subject's identity: a row naming another value there than the
            # subject its other ids belong to already has is two subjects, not one.
            owner = next(iter(owners))
            uid_ns = namespace_of(plan.uid_column)
            uid_value = normalize_string(row.get(plan.uid_column))
            if uid_value is not None:
                key = id_key(uid_value)
                held = {k for (ns, k), uids in known_pairs.items() if ns == uid_ns and owner in uids}
                held |= {k for (ns, k), uid in claimed.items() if ns == uid_ns and uid == owner}
                if owner != str(uid_value) and held and key not in held:
                    result.conflicts.append((number, f"{plan.uid_column} {uid_value} but its other identifiers "
                                                     f"belong to {owner}"))
                    continue
        if owners:
            uid = next(iter(owners))
            is_new = uid not in index._uids and uid not in result.new_subjects
        else:
            raw = normalize_string(row.get(plan.uid_column)) if plan.uid_column else None
            uid = str(raw) if raw is not None else _next_uid(plan.uid_prefix or "S", taken)
            is_new = True
        added = 0
        for ns, value in pairs:
            key = id_key(value)
            if uid in known_pairs.get((ns, key), set()) or claimed.get((ns, key)) == uid:
                continue
            claimed[(ns, key)] = uid
            rows.append({"subject_uid": uid, "id_namespace": ns, "id_value": value, "id_source": REGISTRY_SOURCE,
                         "source_file": plan.source_name or "registry", "source_sheet": str(plan.sheet),
                         "is_primary": bool(plan.uid_column) and ns == namespace_of(plan.uid_column),
                         "source_batch_id": plan.batch_id, "updated_at": now})
            added += 1
        for column, ns in plan.attribute_columns.items():
            value = normalize_string(row.get(column)) if column in frame.columns else None
            if value is None:
                continue
            if ns == "birth_date":
                parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
                value = parsed.date().isoformat() if not pd.isna(parsed) else value
            if index.lookup(value, ns) and uid in index.lookup(value, ns):
                continue
            rows.append({"subject_uid": uid, "id_namespace": ns, "id_value": str(value), "id_source": REGISTRY_SOURCE,
                         "source_file": plan.source_name or "registry", "source_sheet": str(plan.sheet),
                         "is_primary": False, "source_batch_id": plan.batch_id, "updated_at": now})
            added += 1
        cohort = (normalize_string(row.get(plan.cohort_column)) if plan.cohort_column else None) or plan.cohort_id
        if cohort:
            result.cohort_rows.append((str(cohort), uid))
        taken.add(uid)
        if is_new:
            if uid not in result.new_subjects:
                result.new_subjects.append(uid)
        elif added:
            if uid not in result.updated_subjects:
                result.updated_subjects.append(uid)
        elif uid not in result.unchanged_subjects and uid not in result.updated_subjects:
            result.unchanged_subjects.append(uid)
    result.id_rows = pd.DataFrame(rows)
    return result


def register_subjects(repo: Any, frame: pd.DataFrame, plan: RegistryPlan, *, rebuild_index: bool = True) -> RegistryResult:
    """Add the registry's subjects and identifiers to *repo* (see the module); conflicting rows are
    left out and reported. Rebuilds ``subjects`` from the identifiers, then the query index."""
    index = SubjectIndex.load(repo)
    result = plan_registry(index, frame, plan)
    provenance = {"importer": "nvitk subject registry", "source_file": plan.source_name, "source_batch_id": plan.batch_id}
    if not result.id_rows.empty:
        repo.upsert_table("subject_ids", result.id_rows, provenance=provenance)
    if result.cohort_rows:
        from .importers import upsert_cohort_membership_for_subjects

        by_cohort: dict[str, list[str]] = {}
        for cohort, uid in result.cohort_rows:
            by_cohort.setdefault(cohort, []).append(uid)
        for cohort, uids in by_cohort.items():
            upsert_cohort_membership_for_subjects(repo, cohort, sorted(set(uids)), membership_source=REGISTRY_SOURCE,
                                                  source_batch_id=plan.batch_id or "subject_registry")
    if result.new_subjects or result.updated_subjects:
        from .importers import rebuild_subjects_table

        rebuild_subjects_table(repo)
    if rebuild_index:
        try:
            repo.build_sqlite_index()
        except Exception:  # noqa: BLE001 (a dataset without the index yet)
            pass
    return result


def guess_registry_plan(frame: pd.DataFrame, index: SubjectIndex | None = None) -> RegistryPlan:
    """A first registry plan: identifier-like columns as namespaces (the dataset's own spelling when
    it knows them), ``sex`` / birth-date columns as attributes, a cohort column, and the uid column:
    ``subject_uid`` if present, else the column matching the most existing subjects, else the first
    identifier column."""
    plan = RegistryPlan()
    for column in frame.columns:
        ns = namespace_of(column)
        if ns in ("sex", "sexo", "gender", "genero"):
            plan.attribute_columns[str(column)] = "sex"
        elif ns in ("birth_date", "fechanacimiento", "fecha_nacimiento", "date_of_birth", "dob"):
            plan.attribute_columns[str(column)] = "birth_date"
        elif ns in ("cohort", "cohort_id", "cohorte", "group", "grupo"):
            plan.cohort_column = str(column)
        elif ns == UID or looks_like_identifier(str(column)):
            plan.id_columns[str(column)] = UID if ns == UID else ns
    if UID in plan.id_columns.values():
        plan.uid_column = next(c for c, ns in plan.id_columns.items() if ns == UID)
    elif plan.id_columns:
        best, best_n = "", -1
        if index is not None and index.n_subjects:
            for column in plan.id_columns:
                n = sum(1 for v in frame[column].dropna().unique() if index.lookup(v, UID))
                if n > best_n:
                    best, best_n = column, n
        plan.uid_column = best if best_n > 0 else next(iter(plan.id_columns))
    return plan


__all__ = [
    "ATTRIBUTE_NAMESPACES",
    "REGISTRY_SOURCE",
    "RegistryPlan",
    "RegistryResult",
    "Resolution",
    "SubjectIndex",
    "UID",
    "guess_registry_plan",
    "id_key",
    "namespace_of",
    "plan_registry",
    "register_subjects",
    "resolve_subject_label",
]
