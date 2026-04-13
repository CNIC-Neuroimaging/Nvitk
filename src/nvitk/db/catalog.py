from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from nvitk.core.exceptions import ValidationError

from .exceptions import ManifestError, TableNotFoundError
from .storage import coerce_bool, infer_manifest_dtypes, normalize_variable_id, read_json, utc_now_iso, write_json


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
        self.pipelines_manifest: dict[str, Any] = {"schema_version": "1.0", "pipelines": []}
        self.pipelines_manifest_path: Path | None = None
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
            Path("catalog") / "measurement_pipelines.json",
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

    def list_pipelines(self, modality: str | None = None) -> list[dict[str, Any]]:
        entries = list(self.pipelines_manifest.get("pipelines", []))
        if modality is None:
            return entries
        m = str(modality).strip().lower()
        return [e for e in entries if e.get("modality") and str(e["modality"]).strip().lower() == m]

    def default_pipeline_id(self, modality: str) -> str | None:
        m = str(modality).strip().lower()
        for entry in self.pipelines_manifest.get("pipelines", []):
            if not entry.get("modality") or str(entry["modality"]).strip().lower() != m:
                continue
            if coerce_bool(entry.get("is_default")):
                return str(entry["pipeline_id"])
        return None

    def pipeline_ids_for_role(self, role: str) -> list[str]:
        return [
            str(e["pipeline_id"])
            for e in self.pipelines_manifest.get("pipelines", [])
            if str(e.get("role") or "") == role
        ]

    def all_pipeline_ids(self) -> set[str]:
        return {str(e["pipeline_id"]) for e in self.pipelines_manifest.get("pipelines", []) if e.get("pipeline_id")}

    def _pipeline_entries_for_modality(self, modality: str | None) -> list[dict[str, Any]]:
        entries = list(self.pipelines_manifest.get("pipelines", []))
        if modality is None or not str(modality).strip():
            return entries
        m = str(modality).strip().lower()
        return [e for e in entries if e.get("modality") and str(e["modality"]).strip().lower() == m]

    @staticmethod
    def _normalize_pipeline_tokens(selector: str | int | Iterable[str | int] | None) -> list[str]:
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
