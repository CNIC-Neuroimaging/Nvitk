from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from nvitk.core.exceptions import ValidationError

from .exceptions import ManifestError, TableNotFoundError
from .storage import infer_manifest_dtypes, normalize_variable_id, read_json, utc_now_iso, write_json


@dataclass(frozen=True)
class TableDefinition:
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
        self.root = Path(root).expanduser().resolve()
        self.repository_path = self.root / "catalog" / "repository.json"
        if not self.repository_path.exists():
            raise ManifestError(f"Dataset repository manifest not found: {self.repository_path}")

        self.repository_manifest: dict[str, Any] = {}
        self.tables_manifest: dict[str, Any] = {}
        self.variables_manifest: dict[str, Any] = {}
        self._tables: dict[str, TableDefinition] = {}
        self.refresh()

    @classmethod
    def create_scaffold(cls, root: str | Path) -> Path:
        destination = Path(root).expanduser().resolve()
        template_root = Path(__file__).resolve().parents[3] / "dataset"
        destination.mkdir(parents=True, exist_ok=True)

        for relative in [
            Path("README.md"),
            Path("catalog") / "repository.json",
            Path("catalog") / "tables.json",
            Path("catalog") / "variables.json",
            Path("catalog") / "schema" / "repository.schema.json",
            Path("catalog") / "schema" / "tables.schema.json",
            Path("catalog") / "schema" / "variables.schema.json",
        ]:
            source = template_root / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        (destination / "tables").mkdir(parents=True, exist_ok=True)
        (destination / "cache").mkdir(parents=True, exist_ok=True)
        return destination

    def refresh(self) -> None:
        self.repository_manifest = read_json(self.repository_path)
        self.validate_repository_manifest(self.repository_manifest)

        self.tables_manifest_path = self.root / self.repository_manifest["tables_manifest"]
        self.variables_manifest_path = self.root / self.repository_manifest["variables_manifest"]

        self.tables_manifest = read_json(self.tables_manifest_path)
        self.variables_manifest = read_json(self.variables_manifest_path)
        self.validate_tables_manifest(self.tables_manifest)
        self.validate_variables_manifest(self.variables_manifest)

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
        return self.root / self.repository_manifest["sqlite_index_path"]

    @property
    def table_root(self) -> Path:
        return self.root / self.repository_manifest["table_root"]

    def list_tables(self) -> list[str]:
        return sorted(self._tables)

    def get_table(self, name: str) -> TableDefinition:
        try:
            return self._tables[name]
        except KeyError as exc:
            raise TableNotFoundError(f"Unknown dataset table: {name}") from exc

    def table_exists(self, name: str) -> bool:
        return name in self._tables

    def variable_entries(self, *, domain: str | None = None, table: str | None = None) -> list[dict[str, Any]]:
        entries = list(self.variables_manifest.get("variables", []))
        if domain is not None:
            entries = [item for item in entries if item.get("domain") == domain]
        if table is not None:
            entries = [item for item in entries if item.get("table") == table]
        return entries

    def resolve_variable_ids(self, requested: str | Iterable[str] | None, *, domain: str | None = None) -> list[str]:
        if requested is None:
            return []

        values = [requested] if isinstance(requested, str) else list(requested)
        if not values:
            return []

        entries = self.variable_entries(domain=domain)
        alias_map: dict[str, str] = {}

        def register_alias(token: str | None, canonical: str) -> None:
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

    def validate_repository_manifest(self, payload: dict[str, Any]) -> None:
        missing = sorted(self.REQUIRED_REPOSITORY_KEYS - payload.keys())
        if missing:
            raise ManifestError(f"Repository manifest missing keys: {missing}")

    def validate_tables_manifest(self, payload: dict[str, Any]) -> None:
        if "tables" not in payload or not isinstance(payload["tables"], dict):
            raise ManifestError("Tables manifest must define a 'tables' object.")

        for name, entry in payload["tables"].items():
            missing = sorted(self.REQUIRED_TABLE_KEYS - entry.keys())
            if missing:
                raise ManifestError(f"Table '{name}' is missing keys: {missing}")
            if not isinstance(entry["columns"], dict):
                raise ManifestError(f"Table '{name}' columns must be a mapping.")

    def validate_variables_manifest(self, payload: dict[str, Any]) -> None:
        variables = payload.get("variables")
        if not isinstance(variables, list):
            raise ManifestError("Variables manifest must define a 'variables' list.")
        for entry in variables:
            self._validate_variable_entry(entry)

    def _validate_variable_entry(self, entry: dict[str, Any]) -> None:
        missing = sorted(self.REQUIRED_VARIABLE_KEYS - entry.keys())
        if missing:
            raise ManifestError(f"Variable entry missing keys: {missing}")
        variable_id = entry.get("variable_id")
        if not isinstance(variable_id, str) or not variable_id.strip():
            raise ValidationError("Variable identifiers must be non-empty strings.")
