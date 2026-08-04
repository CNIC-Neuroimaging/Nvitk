"""
Dataset catalog: ``repository.json``, table manifests, variable registry, scaffold helpers.

:class:`DatasetCatalog` is the source of truth for paths and schemas under the dataset root.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from nvitk.core.exceptions import ValidationError

from .exceptions import ManifestError, TableNotFoundError
from .storage import (
    COGNITIVE_MEASUREMENT_COLUMNS,
    coerce_bool,
    empty_dataframe,
    infer_manifest_dtypes,
    normalize_variable_id,
    read_json,
    utc_now_iso,
    write_json,
    write_parquet_table,
)


@dataclass(frozen=True)
class TableDefinition:
    """Static description of one Parquet-backed table (columns, keys, wide pivot settings)."""

    name: str
    path: Path
    kind: str
    description: str = ""
    key_columns: tuple[str, ...] = ()
    index_columns: tuple[str, ...] = ()
    wide_index_columns: tuple[str, ...] = ()
    wide_key_columns: tuple[str, ...] = ()
    value_columns: tuple[str, ...] = ()
    columns: dict[str, str] = field(default_factory=dict)
    row_count: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


class DatasetCatalog:
    """
    Load and update ``catalog/repository.json``, resolve table paths, variables, and pipelines.

    Use :meth:`get_table` for :class:`TableDefinition` rows and :meth:`list_tables` for names.
    """

    REQUIRED_REPOSITORY_KEYS = {
        "schema_version",
        "dataset_name",
        "table_root",
        "cache_root",
        "tables_manifest",
        "variables_manifest",
        "sqlite_index_path",
    }
    REQUIRED_TABLE_KEYS = {"path", "kind", "columns"}
    REQUIRED_VARIABLE_KEYS = {"variable_id", "domain", "table"}

    def __init__(self, root: str | Path):
        """Load the catalog manifests under *root* (raises ``ManifestError`` if ``repository.json``
        is missing)."""
        self.root = Path(root).expanduser().resolve()
        self.repository_path = self.root / "catalog" / "repository.json"
        if not self.repository_path.exists():
            raise ManifestError(f"Dataset repository manifest not found: {self.repository_path}")

        self.repository_manifest: dict[str, Any] = {}
        self.tables_manifest: dict[str, Any] = {}
        self.variables_manifest: dict[str, Any] = {}
        self.pipelines_manifest: dict[str, Any] = {"schema_version": "1.0", "pipelines": []}
        self.pipelines_manifest_path: Path | None = None
        self._tables: dict[str, TableDefinition] = {}
        self.refresh()

    @classmethod
    def create_scaffold(cls, root: str | Path) -> Path:
        """Create a minimal dataset tree at *root* by copying the packaged catalog templates
        (manifests + schema) and creating empty ``tables/``/``cache/`` directories."""
        destination = Path(root).expanduser().resolve()
        package_root = Path(__file__).resolve().parents[3]
        template_root = package_root / "dataset" / "catalog"
        if not template_root.exists():
            template_root = package_root / "dataset" / "nvitk-dataset" / "catalog"
        destination.mkdir(parents=True, exist_ok=True)

        catalog_files = (
            "repository.json",
            "tables.json",
            "variables.json",
            "measurement_pipelines.json",
        )
        for name in catalog_files:
            source = template_root / name
            target = destination / "catalog" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.copy2(source, target)

        schema_src = template_root / "schema"
        schema_dst = destination / "catalog" / "schema"
        if schema_src.is_dir():
            schema_dst.mkdir(parents=True, exist_ok=True)
            for schema_file in schema_src.glob("*.json"):
                shutil.copy2(schema_file, schema_dst / schema_file.name)

        (destination / "tables").mkdir(parents=True, exist_ok=True)
        (destination / "cache").mkdir(parents=True, exist_ok=True)
        return destination

    def refresh(self) -> None:
        """Re-read all manifest JSON files from disk and rebuild the in-memory :class:`TableDefinition`
        cache. Called after every catalog mutation to keep this instance consistent with disk."""
        self.repository_manifest = read_json(self.repository_path)
        self.validate_repository_manifest(self.repository_manifest)

        self.tables_manifest_path = self.root / self.repository_manifest["tables_manifest"]
        self.variables_manifest_path = self.root / self.repository_manifest["variables_manifest"]

        self.tables_manifest = read_json(self.tables_manifest_path)
        self.variables_manifest = read_json(self.variables_manifest_path)
        self.validate_tables_manifest(self.tables_manifest)
        self.validate_variables_manifest(self.variables_manifest)

        pipelines_rel = self.repository_manifest.get("pipelines_manifest")
        if pipelines_rel:
            self.pipelines_manifest_path = self.root / pipelines_rel
            self.pipelines_manifest = read_json(self.pipelines_manifest_path)
            self.validate_pipelines_manifest(self.pipelines_manifest)
        else:
            self.pipelines_manifest_path = None
            self.pipelines_manifest = {"schema_version": "1.0", "pipelines": []}

        self._tables = {}
        for name, payload in self.tables_manifest["tables"].items():
            self._tables[name] = TableDefinition(
                name=name,
                path=self.root / payload["path"],
                kind=payload["kind"],
                description=payload.get("description", ""),
                key_columns=tuple(payload.get("key_columns", [])),
                index_columns=tuple(payload.get("index_columns", [])),
                wide_index_columns=tuple(payload.get("wide_index_columns", [])),
                wide_key_columns=tuple(payload.get("wide_key_columns", [])),
                value_columns=tuple(payload.get("value_columns", [])),
                columns=dict(payload.get("columns", {})),
                row_count=payload.get("row_count"),
                provenance=dict(payload.get("provenance", {})),
            )

    @property
    def sqlite_index_path(self) -> Path:
        """Absolute path to the SQLite query-cache database declared in the repository manifest."""
        return self.root / self.repository_manifest["sqlite_index_path"]

    @property
    def table_root(self) -> Path:
        """Absolute path to the directory holding this dataset's Parquet table files."""
        return self.root / self.repository_manifest["table_root"]

    def list_tables(self) -> list[str]:
        """Sorted names of every table registered in the manifest."""
        return sorted(self._tables)

    def get_table(self, name: str) -> TableDefinition:
        """Return *name*'s :class:`TableDefinition`, raising ``TableNotFoundError`` if unregistered."""
        try:
            return self._tables[name]
        except KeyError as exc:
            raise TableNotFoundError(f"Unknown dataset table: {name}") from exc

    def table_exists(self, name: str) -> bool:
        """True if *name* is a registered table."""
        return name in self._tables

    def ensure_table_definition(self, name: str, *, clone_from: str) -> Path:
        """
        Register *name* in ``tables.json`` by cloning manifest metadata from *clone_from*,
        pointing at ``<table_root>/{name}.parquet``. Creates an empty Parquet file if missing.

        No-op (returns existing Parquet path) when *name* is already registered.
        """
        if self.table_exists(name):
            return self.get_table(name).path

        template = self.tables_manifest.setdefault("tables", {}).get(clone_from)
        if template is None:
            raise TableNotFoundError(f"Cannot clone table definition: unknown template {clone_from!r}")

        table_root = self.repository_manifest.get("table_root", "tables").strip().rstrip("/")
        rel_path = f"{table_root}/{name}.parquet"
        new_payload: dict[str, Any] = {
            "path": rel_path,
            "kind": template.get("kind", "derived"),
            "columns": dict(template.get("columns", {})),
        }
        for optional in (
            "description",
            "key_columns",
            "index_columns",
            "wide_index_columns",
            "wide_key_columns",
            "value_columns",
        ):
            if optional in template:
                new_payload[optional] = template[optional]
        if "provenance" in template:
            new_payload["provenance"] = dict(template["provenance"])

        new_payload["row_count"] = 0
        new_payload["last_updated"] = utc_now_iso()

        dest = self.root / rel_path
        columns_map: dict[str, str] = dict(new_payload["columns"])
        if not columns_map:
            raise ManifestError(f"Template table {clone_from!r} has no columns; cannot clone schema.")
        write_parquet_table(dest, empty_dataframe(columns_map))

        tables = self.tables_manifest.setdefault("tables", {})
        tables[name] = new_payload
        self.tables_manifest["last_updated"] = utc_now_iso()
        write_json(self.tables_manifest_path, self.tables_manifest)
        self.refresh()
        return dest

    def ensure_cognitive_measurements_table(self) -> Path:
        """Register ``cognitive_measurements`` with the canonical long-form schema."""
        if self.table_exists("cognitive_measurements"):
            return self.get_table("cognitive_measurements").path

        columns = {
            "subject_uid": "string",
            "visit_id": "string",
            "variable_id": "string",
            "value_num": "float64",
            "value_text": "string",
            "unit": "string",
            "value_kind": "string",
            "source_table": "string",
            "source_file": "string",
            "source_sheet": "string",
            "source_column": "string",
            "source_batch_id": "string",
            "measured_at": "datetime64[ns]",
        }
        if tuple(columns) != COGNITIVE_MEASUREMENT_COLUMNS:
            columns = {name: columns.get(name, "string") for name in COGNITIVE_MEASUREMENT_COLUMNS}

        table_root = self.repository_manifest.get("table_root", "tables").strip().rstrip("/")
        rel_path = f"{table_root}/cognitive_measurements.parquet"
        new_payload: dict[str, Any] = {
            "path": rel_path,
            "kind": "measurements",
            "description": "Long-form cognitive test variables linked to subjects and optional visits.",
            "key_columns": [
                "subject_uid",
                "visit_id",
                "variable_id",
                "source_file",
                "source_sheet",
                "source_column",
            ],
            "index_columns": [
                "subject_uid",
                "visit_id",
                "variable_id",
                "source_file",
                "source_sheet",
                "source_column",
            ],
            "wide_index_columns": ["subject_uid", "visit_id"],
            "wide_key_columns": ["variable_id"],
            "value_columns": ["value_num", "value_text"],
            "columns": columns,
            "row_count": 0,
            "last_updated": utc_now_iso(),
        }
        dest = self.root / rel_path
        write_parquet_table(dest, empty_dataframe(columns))
        tables = self.tables_manifest.setdefault("tables", {})
        tables["cognitive_measurements"] = new_payload
        self.tables_manifest["last_updated"] = utc_now_iso()
        write_json(self.tables_manifest_path, self.tables_manifest)
        self.refresh()
        return dest

    def list_pipelines(self, modality: str | None = None) -> list[dict[str, Any]]:
        """All pipeline entries, or just those matching *modality* (case-insensitive) if given."""
        entries = list(self.pipelines_manifest.get("pipelines", []))
        if modality is None:
            return entries
        m = str(modality).strip().lower()
        return [e for e in entries if e.get("modality") and str(e["modality"]).strip().lower() == m]

    def default_pipeline_id(self, modality: str) -> str | None:
        """The ``pipeline_id`` flagged ``is_default`` for *modality*, or ``None`` if none is set."""
        m = str(modality).strip().lower()
        for entry in self.pipelines_manifest.get("pipelines", []):
            if not entry.get("modality") or str(entry["modality"]).strip().lower() != m:
                continue
            if coerce_bool(entry.get("is_default")):
                return str(entry["pipeline_id"])
        return None

    def pipeline_ids_for_role(self, role: str) -> list[str]:
        """``pipeline_id`` values for every pipeline entry tagged with *role*."""
        return [
            str(e["pipeline_id"])
            for e in self.pipelines_manifest.get("pipelines", [])
            if str(e.get("role") or "") == role
        ]

    def all_pipeline_ids(self) -> set[str]:
        """Set of every registered ``pipeline_id`` across all modalities."""
        return {str(e["pipeline_id"]) for e in self.pipelines_manifest.get("pipelines", []) if e.get("pipeline_id")}

    def _pipeline_entries_for_modality(self, modality: str | None) -> list[dict[str, Any]]:
        """Pipeline entries for *modality* (case-insensitive), or all entries if *modality* is unset."""
        entries = list(self.pipelines_manifest.get("pipelines", []))
        if modality is None or not str(modality).strip():
            return entries
        m = str(modality).strip().lower()
        return [e for e in entries if e.get("modality") and str(e["modality"]).strip().lower() == m]

    @staticmethod
    def _normalize_pipeline_tokens(selector: str | int | Iterable[str | int] | None) -> list[str]:
        """Coerce a pipeline *selector* (a single str/int or an iterable of them) into a flat list of
        non-empty stripped string tokens."""
        if selector is None:
            return []
        if isinstance(selector, (str, int)):
            raw = [selector]
        else:
            raw = list(selector)
        out: list[str] = []
        for t in raw:
            if isinstance(t, int):
                s = str(t).strip()
            else:
                s = str(t).strip()
            if s:
                out.append(s)
        return out

    def resolve_pipeline_selector(
        self,
        selector: str | int | Iterable[str | int] | None,
        *,
        modality: str | None = None,
    ) -> list[str] | None:
        """
        Return None to apply catalog defaults per modality in ``DataRepo.image``.
        Otherwise return an explicit ``pipeline_id`` list.

        Aliases from ``measurement_pipelines.json`` (including integers like ``1`` / ``2``) resolve
        within the pipelines for ``modality`` when provided; without ``modality``, ambiguous aliases
        raise ``ManifestError``.
        """
        tokens = self._normalize_pipeline_tokens(selector)
        if not tokens:
            return None

        if len(tokens) == 1 and tokens[0].lower() == "legacy":
            ids = self.pipeline_ids_for_role("legacy")
            if not ids:
                raise ManifestError("No pipelines with role 'legacy' registered in measurement_pipelines.json.")
            return ids

        mod_scope = str(modality).strip() if modality and str(modality).strip() else None
        candidates = self._pipeline_entries_for_modality(mod_scope) if mod_scope else list(
            self.pipelines_manifest.get("pipelines", [])
        )
        cand_ids = {str(e["pipeline_id"]) for e in candidates if e.get("pipeline_id")}

        # alias_key -> list of pipeline_ids (detect ambiguity when len > 1)
        alias_buckets: dict[str, list[str]] = {}
        for e in candidates:
            pid = str(e["pipeline_id"])
            keys: set[str] = {pid.lower()}
            for a in e.get("aliases") or []:
                keys.add(str(a).strip().lower())
            for k in keys:
                alias_buckets.setdefault(k, []).append(pid)

        def resolve_one(token: str) -> str:
            """Resolve a single pipeline selector token to a canonical ``pipeline_id`` (exact id match,
            then alias/short-id lookup), raising ``ManifestError`` if ambiguous or unknown."""
            tl = token.lower()
            # Exact pipeline_id match (case-insensitive) within candidates
            for pid in cand_ids:
                if pid.lower() == tl:
                    return pid
            # Alias / short id
            if tl in alias_buckets:
                hits = list(dict.fromkeys(alias_buckets[tl]))
                if len(hits) == 1:
                    return hits[0]
                raise ManifestError(
                    f"Ambiguous pipeline selector {token!r} matches multiple pipeline_id values {hits!r}; "
                    "pass modality=... to choose a pipeline."
                )
            known = sorted(self.all_pipeline_ids())
            raise ManifestError(f"Unknown pipeline_id or alias: {token!r}. Known pipeline_id values: {known}")

        resolved = [resolve_one(t) for t in tokens]
        return list(dict.fromkeys(resolved))

    def validate_pipelines_manifest(self, payload: dict[str, Any]) -> None:
        """Validate the pipelines manifest structure and reject duplicate ``is_default`` pipelines within
        the same modality; raises ``ManifestError`` on any violation."""
        pipelines = payload.get("pipelines")
        if not isinstance(pipelines, list):
            raise ManifestError("Pipelines manifest must define a 'pipelines' list.")
        defaults_per_mod: dict[str, str] = {}
        for entry in pipelines:
            if not isinstance(entry, dict):
                raise ManifestError("Each pipeline entry must be an object.")
            pid = entry.get("pipeline_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ManifestError("Each pipeline must have a non-empty pipeline_id.")
            mod = entry.get("modality")
            if mod is not None and (not isinstance(mod, str) or not str(mod).strip()):
                raise ManifestError(f"Invalid modality for pipeline {pid!r}.")
            if coerce_bool(entry.get("is_default")) and mod is not None:
                key = str(mod).strip().lower()
                if key in defaults_per_mod:
                    raise ManifestError(
                        f"Multiple is_default pipelines for modality {mod!r}: "
                        f"{defaults_per_mod[key]!r} and {pid!r}."
                    )
                defaults_per_mod[key] = str(pid)

    def variable_entries(self, *, domain: str | None = None, table: str | None = None) -> list[dict[str, Any]]:
        """Registered variable entries, optionally filtered by ``domain`` and/or ``table``."""
        entries = list(self.variables_manifest.get("variables", []))
        if domain is not None:
            entries = [item for item in entries if item.get("domain") == domain]
        if table is not None:
            entries = [item for item in entries if item.get("table") == table]
        return entries

    def resolve_variable_ids(self, requested: str | Iterable[str] | None, *, domain: str | None = None) -> list[str]:
        """Resolve *requested* names/aliases/source-column names/labels to canonical ``variable_id``
        values (case- and normalization-insensitive), leaving any unmatched entries unchanged."""
        if requested is None:
            return []

        values = [requested] if isinstance(requested, str) else list(requested)
        if not values:
            return []

        entries = self.variable_entries(domain=domain)
        alias_map: dict[str, str] = {}

        def register_alias(token: str | None, canonical: str) -> None:
            """Record *token* (raw, lower-cased, and normalized forms) as an alias for *canonical*."""
            if token is None:
                return
            text = str(token).strip()
            if not text:
                return
            alias_map[text] = canonical
            alias_map[text.lower()] = canonical
            normalized = normalize_variable_id(text)
            if normalized:
                alias_map[normalized] = canonical

        for entry in entries:
            canonical = str(entry["variable_id"])
            register_alias(canonical, canonical)
            source_column = entry.get("source_column")
            if source_column:
                register_alias(str(source_column), canonical)
            for alias in entry.get("aliases", []):
                register_alias(str(alias), canonical)
            label = entry.get("label")
            if label:
                register_alias(str(label), canonical)
            for meta_key in ("export_name", "original_name"):
                meta = entry.get(meta_key)
                if meta:
                    register_alias(str(meta), canonical)

        resolved: list[str] = []
        for item in values:
            text = str(item).strip()
            if not text:
                resolved.append(text)
                continue
            candidate = (
                alias_map.get(text)
                or alias_map.get(text.lower())
                or alias_map.get(normalize_variable_id(text))
            )
            resolved.append(candidate if candidate is not None else text)
        return resolved

    def register_variables(self, entries: list[dict[str, Any]], *, merge: bool = True) -> None:
        """Add or update *entries* in the variables manifest (merging into existing entries with the same
        ``variable_id`` when ``merge`` is True, otherwise overwriting), then persist and reload."""
        current = self.variables_manifest.setdefault("variables", [])
        lookup = {item["variable_id"]: dict(item) for item in current}

        for entry in entries:
            self._validate_variable_entry(entry)
            variable_id = str(entry["variable_id"])
            if merge and variable_id in lookup:
                merged = lookup[variable_id]
                merged = self._merge_variable_entry(merged, entry)
                lookup[variable_id] = merged
            else:
                lookup[variable_id] = dict(entry)

        self.variables_manifest["schema_version"] = self.variables_manifest.get("schema_version", "1.0")
        self.variables_manifest["last_updated"] = utc_now_iso()
        self.variables_manifest["variables"] = sorted(lookup.values(), key=lambda item: str(item["variable_id"]))
        write_json(self.variables_manifest_path, self.variables_manifest)
        self.refresh()

    def _merge_variable_entry(self, existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        """Merge *new* onto *existing*: union+de-dupe ``aliases``, and for every other key skip empty
        strings/lists/None in *new* so a partial update never blanks out existing metadata."""
        merged = dict(existing)
        for key, value in new.items():
            if key == "aliases":
                existing_aliases = [str(item) for item in merged.get("aliases", []) if str(item).strip()]
                new_aliases = [str(item) for item in value or [] if str(item).strip()]
                seen: set[str] = set()
                aliases: list[str] = []
                for alias in existing_aliases + new_aliases:
                    if alias not in seen:
                        seen.add(alias)
                        aliases.append(alias)
                if aliases:
                    merged["aliases"] = aliases
                continue

            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, list) and not value:
                continue
            merged[key] = value
        return merged

    def update_table_schema(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        provenance: dict[str, Any] | None = None,
        path: str | None = None,
    ) -> None:
        """Update (or create) *name*'s entry in the tables manifest from *df*'s inferred dtypes, row
        count, and optional *provenance*/``path``, then persist and reload."""
        tables = self.tables_manifest.setdefault("tables", {})
        payload = tables.get(name)
        if payload is None:
            payload = {
                "path": path or str((Path(self.repository_manifest["table_root"]) / f"{name}.parquet").as_posix()),
                "kind": "derived",
                "columns": {},
            }
            tables[name] = payload
        elif path is not None:
            payload["path"] = path

        payload["columns"] = infer_manifest_dtypes(df)
        payload["row_count"] = int(len(df))
        payload["last_updated"] = utc_now_iso()
        if provenance:
            merged_provenance = dict(payload.get("provenance", {}))
            merged_provenance.update(provenance)
            payload["provenance"] = merged_provenance

        self.tables_manifest["last_updated"] = utc_now_iso()
        write_json(self.tables_manifest_path, self.tables_manifest)
        self.refresh()

    def clear_table(self, name: str) -> None:
        """Delete the Parquet file for ``name`` if present and set ``row_count`` to 0 in the manifest."""
        payload = self.tables_manifest.setdefault("tables", {}).get(name)
        if payload is None:
            raise TableNotFoundError(f"Unknown dataset table: {name}")
        path = self.root / payload["path"]
        if path.exists():
            path.unlink()
        payload["row_count"] = 0
        payload["last_updated"] = utc_now_iso()
        self.tables_manifest["last_updated"] = utc_now_iso()
        write_json(self.tables_manifest_path, self.tables_manifest)
        self.refresh()

    def validate_repository_manifest(self, payload: dict[str, Any]) -> None:
        """Raise ``ManifestError`` if *payload* is missing any ``REQUIRED_REPOSITORY_KEYS``."""
        missing = sorted(self.REQUIRED_REPOSITORY_KEYS - payload.keys())
        if missing:
            raise ManifestError(f"Repository manifest missing keys: {missing}")

    def validate_tables_manifest(self, payload: dict[str, Any]) -> None:
        """Raise ``ManifestError`` if *payload*'s ``tables`` object or any entry is malformed."""
        if "tables" not in payload or not isinstance(payload["tables"], dict):
            raise ManifestError("Tables manifest must define a 'tables' object.")

        for name, entry in payload["tables"].items():
            missing = sorted(self.REQUIRED_TABLE_KEYS - entry.keys())
            if missing:
                raise ManifestError(f"Table '{name}' is missing keys: {missing}")
            if not isinstance(entry["columns"], dict):
                raise ManifestError(f"Table '{name}' columns must be a mapping.")

    def validate_variables_manifest(self, payload: dict[str, Any]) -> None:
        """Raise ``ManifestError`` if *payload* lacks a ``variables`` list, or if any entry is invalid."""
        variables = payload.get("variables")
        if not isinstance(variables, list):
            raise ManifestError("Variables manifest must define a 'variables' list.")
        for entry in variables:
            self._validate_variable_entry(entry)

    def _validate_variable_entry(self, entry: dict[str, Any]) -> None:
        """Raise ``ManifestError``/``ValidationError`` if *entry* is missing required keys or has a
        blank ``variable_id``."""
        missing = sorted(self.REQUIRED_VARIABLE_KEYS - entry.keys())
        if missing:
            raise ManifestError(f"Variable entry missing keys: {missing}")
        variable_id = entry.get("variable_id")
        if not isinstance(variable_id, str) or not variable_id.strip():
            raise ValidationError("Variable identifiers must be non-empty strings.")
