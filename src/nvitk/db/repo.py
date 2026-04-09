from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .catalog import DatasetCatalog, TableDefinition
from .filters import apply_filters, ensure_list, merge_filters
from .sqlite_index import SQLiteIndex
from .storage import coerce_dataframe_to_manifest, empty_dataframe, read_parquet_table, utc_now_iso, write_parquet_table


def _default_dataset_root() -> Path:
    env_root = os.getenv("NVITK_DATASET_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "dataset"


class DataRepo:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        use_sqlite: bool = False,
        auto_scaffold: bool = False,
    ):
        dataset_root = Path(root or _default_dataset_root()).expanduser().resolve()
        if auto_scaffold and not (dataset_root / "catalog" / "repository.json").exists():
            DatasetCatalog.create_scaffold(dataset_root)

        self.root = dataset_root
        self.catalog = DatasetCatalog(dataset_root)
        self.sqlite = SQLiteIndex(self.catalog.sqlite_index_path)
        self.use_sqlite = use_sqlite

    def list_tables(self) -> list[str]:
        return self.catalog.list_tables()

    def get(
        self,
        table: str,
        *,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        wide: bool = False,
        use_sqlite: bool | None = None,
    ) -> pd.DataFrame:
        definition = self.catalog.get_table(table)
        effective_sqlite = self.use_sqlite if use_sqlite is None else use_sqlite

        if effective_sqlite and self.sqlite.exists():
            try:
                df = self.sqlite.query_table(table, columns=columns, filters=filters)
            except Exception:
                df = self._read_table(definition, columns=columns, filters=filters)
        else:
            df = self._read_table(definition, columns=columns, filters=filters)

        df = coerce_dataframe_to_manifest(df, definition.columns)
        if wide:
            if table in {"clinical_measurements", "image_measurements"}:
                df = self._resolve_measurement_values(df)
            return self._to_wide(df, definition)
        return df.reset_index(drop=True)

    def clinical(
        self,
        *,
        variables: str | Iterable[str] | None = None,
        filters: dict[str, Any] | None = None,
        wide: bool = False,
        use_sqlite: bool | None = None,
    ) -> pd.DataFrame:
        resolved_variables = self.catalog.resolve_variable_ids(variables, domain="clinical")
        merged = merge_filters(filters, {"variable_id": resolved_variables} if resolved_variables else None)
        df = self.get("clinical_measurements", filters=merged, use_sqlite=use_sqlite)
        return self._prepare_measurements(df, wide=wide, table_name="clinical_measurements")

    def image(
        self,
        *,
        modality: str | None = None,
        variables: str | Iterable[str] | None = None,
        regions: str | Iterable[str] | None = None,
        filters: dict[str, Any] | None = None,
        wide: bool = False,
        use_sqlite: bool | None = None,
    ) -> pd.DataFrame:
        resolved_variables = self.catalog.resolve_variable_ids(variables, domain="image")
        merged = merge_filters(
            filters,
            {"variable_id": resolved_variables} if resolved_variables else None,
            {"modality": modality} if modality else None,
            {"region_id": ensure_list(regions)} if regions else None,
        )
        df = self.get("image_measurements", filters=merged, use_sqlite=use_sqlite)
        return self._prepare_measurements(df, wide=wide, table_name="image_measurements")

    def assets(self, *, filters: dict[str, Any] | None = None, use_sqlite: bool | None = None) -> pd.DataFrame:
        return self.get("assets", filters=filters, use_sqlite=use_sqlite)

    def join(
        self,
        frames: Iterable[pd.DataFrame],
        *,
        on: str | list[str] = "subject_uid",
        how: str = "left",
    ) -> pd.DataFrame:
        dataframes = [frame.copy() for frame in frames]
        if not dataframes:
            return pd.DataFrame()
        keys = [on] if isinstance(on, str) else list(on)
        result = dataframes[0]
        for frame in dataframes[1:]:
            result = result.merge(frame, on=keys, how=how)
        return result

    def write_table(
        self,
        table: str,
        df: pd.DataFrame,
        *,
        provenance: dict[str, Any] | None = None,
        build_sqlite_index: bool = False,
    ) -> Path:
        definition = self.catalog.get_table(table)
        write_parquet_table(definition.path, df)
        merged_provenance = {"written_at": utc_now_iso()}
        if provenance:
            merged_provenance.update(provenance)
        self.catalog.update_table_schema(table, df, provenance=merged_provenance, path=self._relative_path(definition.path))
        if build_sqlite_index:
            self.build_sqlite_index(tables=[table])
        return definition.path

    def upsert_table(
        self,
        table: str,
        df: pd.DataFrame,
        *,
        key_columns: list[str] | None = None,
        provenance: dict[str, Any] | None = None,
        build_sqlite_index: bool = False,
    ) -> pd.DataFrame:
        definition = self.catalog.get_table(table)
        existing = self.get(table)
        combined = pd.concat([existing, df], ignore_index=True)
        keys = key_columns or list(definition.key_columns)
        if keys:
            present_keys = [column for column in keys if column in combined.columns]
            if present_keys:
                combined = combined.drop_duplicates(subset=present_keys, keep="last")
        self.write_table(table, combined, provenance=provenance, build_sqlite_index=build_sqlite_index)
        return combined

    def build_sqlite_index(self, *, tables: list[str] | None = None) -> Path:
        return self.sqlite.build(self.catalog, tables=tables)

    def register_variables(self, entries: list[dict[str, Any]]) -> None:
        self.catalog.register_variables(entries)

    def _read_table(
        self,
        definition: TableDefinition,
        *,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not definition.path.exists():
            manifest_columns = definition.columns
            selected_columns = {
                column: dtype
                for column, dtype in manifest_columns.items()
                if columns is None or column in columns
            }
            return empty_dataframe(selected_columns)

        df = read_parquet_table(definition.path, columns=columns)
        return apply_filters(df, filters)

    def _prepare_measurements(self, df: pd.DataFrame, *, wide: bool, table_name: str) -> pd.DataFrame:
        df = self._resolve_measurement_values(df)
        if wide:
            definition = self.catalog.get_table(table_name)
            return self._to_wide(df, definition)
        return df.reset_index(drop=True)

    def _resolve_measurement_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df.copy()
        value_num = out["value_num"] if "value_num" in out.columns else None
        value_text = out["value_text"] if "value_text" in out.columns else None
        if value_num is not None and value_text is not None:
            out["value"] = value_num.where(value_num.notna(), value_text)
        elif value_num is not None:
            out["value"] = value_num
        elif value_text is not None:
            out["value"] = value_text
        return out

    def _to_wide(self, df: pd.DataFrame, definition: TableDefinition) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        index_columns = [column for column in definition.wide_index_columns if column in df.columns]
        key_columns = [column for column in definition.wide_key_columns if column in df.columns]
        if not index_columns or not key_columns or "value" not in df.columns:
            return df.copy()

        tmp = df.copy()
        tmp["_wide_key"] = self._compose_wide_keys(tmp, key_columns)
        wide = tmp.pivot_table(index=index_columns, columns="_wide_key", values="value", aggfunc="first")
        wide.columns = [str(column) for column in wide.columns]
        return wide.reset_index()

    def _compose_wide_keys(self, df: pd.DataFrame, key_columns: list[str]) -> pd.Series:
        return (
            df[key_columns]
            .astype("string")
            .fillna("")
            .agg(lambda row: "__".join(item for item in row if item), axis=1)
            .astype("string")
        )

    def _relative_path(self, path: Path) -> str:
        return str(path.relative_to(self.root).as_posix())
