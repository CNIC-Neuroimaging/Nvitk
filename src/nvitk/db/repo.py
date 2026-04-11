from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .asl_atlases import regions_for_atlas
from .catalog import DatasetCatalog, TableDefinition
from .exceptions import FilterError
from .filters import apply_filters, ensure_list, merge_filters
from .sqlite_index import SQLiteIndex
from .storage import coerce_bool, coerce_dataframe_to_manifest, empty_dataframe, normalize_variable_id, read_parquet_table, utc_now_iso, write_parquet_table


def _default_dataset_root() -> Path:
    env_root = os.getenv("NVITK_DATASET_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "dataset"


def _pick_value_column_for_spec(spec: Any) -> str:
    if isinstance(spec, Mapping):
        for raw_op in spec:
            op = str(raw_op).strip().lower().lstrip("$")
            if op in {"contains"}:
                return "value_text"
        return "value_num"
    if isinstance(spec, str):
        return "value_text"
    return "value_num"


def _entity_keys_from_frame(df: pd.DataFrame) -> set[tuple[str, ...] | str]:
    if df.empty:
        return set()
    if "visit_id" in df.columns:
        return set(
            zip(
                df["subject_uid"].astype("string").fillna(""),
                df["visit_id"].astype("string").fillna(""),
            )
        )
    return set(df["subject_uid"].astype("string").fillna(""))


def _frame_matches_entity_keys(df: pd.DataFrame, keys: set[tuple[str, ...] | str]) -> pd.Series:
    if "visit_id" in df.columns:
        tuples = zip(df["subject_uid"].astype("string").fillna(""), df["visit_id"].astype("string").fillna(""))
        return pd.Series([t in keys for t in tuples], index=df.index)
    return df["subject_uid"].astype("string").fillna("").isin(keys)


def _image_wide_single_variable_from_request(variables: str | Iterable[str] | None) -> bool:
    """True when ``variables`` names exactly one measurement (wide columns = region only)."""
    if variables is None:
        return False
    if isinstance(variables, str):
        return bool(str(variables).strip())
    items = list(variables)
    if len(items) != 1:
        return False
    return bool(str(items[0]).strip())


def _compose_image_wide_keys(df: pd.DataFrame, *, single_variable: bool) -> pd.Series:
    """Build short pivot keys: ``<region>_<variable>``, or ``<region>`` when ``single_variable``."""
    idx = df.index
    m = (
        df["modality"].astype("string").fillna("")
        if "modality" in df.columns
        else pd.Series("", index=idx, dtype="string")
    )
    p = (
        df["pipeline_id"].astype("string").fillna("")
        if "pipeline_id" in df.columns
        else pd.Series("", index=idx, dtype="string")
    )
    r = (
        df["region_id"].astype("string").fillna("")
        if "region_id" in df.columns
        else pd.Series("", index=idx, dtype="string")
    )
    r = r.replace("", pd.NA).fillna("unknown")
    v = (
        df["variable_id"].astype("string").fillna("")
        if "variable_id" in df.columns
        else pd.Series("", index=idx, dtype="string")
    )
    v = v.replace("", pd.NA).fillna("unknown")

    fi_num = (
        pd.to_numeric(df["frame_index"], errors="coerce")
        if "frame_index" in df.columns
        else pd.Series(pd.NA, index=idx, dtype="float64")
    )
    frame_suffix = pd.Series("", index=idx, dtype="string")
    mask_f = fi_num.notna() & (fi_num != 0)
    frame_suffix.loc[mask_f] = "_f" + fi_num.loc[mask_f].round().astype("Int64").astype(str)

    m_nz = m.replace("", pd.NA)
    p_nz = p.replace("", pd.NA)
    multi_mod = m_nz.nunique(dropna=True) > 1
    multi_pipe = p_nz.nunique(dropna=True) > 1

    prefix = pd.Series("", index=idx, dtype="string")
    if multi_mod:
        prefix = m.astype("string") + "_"
    if multi_pipe:
        prefix = prefix.astype("string") + p.astype("string") + "_"

    r_s = r.astype("string")
    if single_variable:
        keys = prefix.astype("string") + r_s + frame_suffix.astype("string")
    else:
        keys = prefix.astype("string") + r_s + "_" + v.astype("string") + frame_suffix.astype("string")
    return keys.astype("string")


def _apply_variable_value_filters(df: pd.DataFrame, specs: list[tuple[str, Any]]) -> pd.DataFrame:
    if df.empty or not specs:
        return df
    key_sets: list[set[tuple[str, ...] | str]] = []
    for variable_id, spec in specs:
        sub = df[df["variable_id"].astype("string") == str(variable_id)]
        if sub.empty:
            return df.iloc[0:0].copy()
        value_col = _pick_value_column_for_spec(spec)
        if value_col not in sub.columns:
            return df.iloc[0:0].copy()
        filtered = apply_filters(sub, {value_col: spec})
        if filtered.empty:
            return df.iloc[0:0].copy()
        key_sets.append(_entity_keys_from_frame(filtered))
    allowed = set.intersection(*key_sets)
    if not allowed:
        return df.iloc[0:0].copy()
    return df.loc[_frame_matches_entity_keys(df, allowed)].copy()


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

    def _measurement_column_names(self, table_name: str) -> set[str]:
        return set(self.catalog.get_table(table_name).columns.keys())

    def _split_measurement_filters(
        self,
        filters: dict[str, Any] | None,
        *,
        domain: str,
        table_name: str,
    ) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
        if not filters:
            return {}, []
        columns = self._measurement_column_names(table_name)
        structural: dict[str, Any] = {}
        variable_specs: list[tuple[str, Any]] = []
        for key, spec in filters.items():
            if not isinstance(key, str) or not key.strip():
                raise FilterError("Filter columns must be non-empty strings.")
            k = key.strip()
            if k in columns:
                structural[k] = spec
                continue
            resolved = self.catalog.resolve_variable_ids([k], domain=domain)[0]
            variable_specs.append((str(resolved), spec))
        return structural, variable_specs

    def _load_table_frame(
        self,
        table: str,
        *,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        use_sqlite: bool | None = None,
        force_parquet: bool = False,
    ) -> pd.DataFrame:
        definition = self.catalog.get_table(table)
        effective_sqlite = (self.use_sqlite if use_sqlite is None else use_sqlite) and not force_parquet

        if effective_sqlite and self.sqlite.exists():
            try:
                df = self.sqlite.query_table(table, columns=columns, filters=filters)
            except Exception:
                df = self._read_table(definition, columns=columns, filters=filters)
        else:
            df = self._read_table(definition, columns=columns, filters=filters)

        return coerce_dataframe_to_manifest(df, definition.columns)

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
            image_sv = False if table == "image_measurements" else None
            return self._to_wide(df, definition, image_wide_single_variable=image_sv)
        return df.reset_index(drop=True)

    def clinical(
        self,
        *,
        variables: str | Iterable[str] | None = None,
        filters: dict[str, Any] | None = None,
        wide: bool = True,
        use_sqlite: bool | None = None,
    ) -> pd.DataFrame:
        resolved_variables = self.catalog.resolve_variable_ids(variables, domain="clinical")
        structural, var_specs = self._split_measurement_filters(filters, domain="clinical", table_name="clinical_measurements")
        merged = merge_filters(structural, {"variable_id": resolved_variables} if resolved_variables else None)
        force_parquet = bool(var_specs)
        df = self._load_table_frame(
            "clinical_measurements",
            filters=merged,
            use_sqlite=use_sqlite,
            force_parquet=force_parquet,
        )
        if var_specs:
            df = _apply_variable_value_filters(df, var_specs)
        return self._prepare_measurements(df, wide=wide, table_name="clinical_measurements")

    def image(
        self,
        *,
        modality: str | None = None,
        variables: str | Iterable[str] | None = None,
        regions: str | Iterable[str] | None = None,
        atlas: str | None = None,
        filters: dict[str, Any] | None = None,
        wide: bool = True,
        use_sqlite: bool | None = None,
        pipeline: str | Iterable[str] | None = None,
        data_version: int | None = None,
    ) -> pd.DataFrame:
        if data_version is not None:
            warnings.warn(
                "data_version is deprecated; use pipeline='legacy' or pipeline=None with catalog defaults.",
                DeprecationWarning,
                stacklevel=2,
            )
            if data_version == 1:
                pipeline = "legacy"
            elif data_version == 2:
                pipeline = None
            else:
                raise FilterError("data_version must be None, 1, or 2.")

        filters_norm = dict(filters or {})
        if "pipeline_version" in filters_norm and "pipeline_id" not in filters_norm:
            filters_norm["pipeline_id"] = filters_norm.pop("pipeline_version")

        if atlas is not None:
            if modality is None or str(modality).strip().lower() != "asl":
                raise FilterError("Parameter 'atlas' requires modality='asl'.")
            atlas_regions = regions_for_atlas(atlas)
            user_regions = [normalize_variable_id(r) for r in ensure_list(regions)] if regions else []
            if user_regions:
                intersection = sorted(set(atlas_regions) & set(user_regions))
                if not intersection:
                    raise FilterError("'regions' and 'atlas' have no region_id in common.")
                region_filter = intersection
            else:
                region_filter = atlas_regions
        else:
            region_filter = ensure_list(regions) if regions else None

        resolved_variables = self.catalog.resolve_variable_ids(variables, domain="image")
        structural, var_specs = self._split_measurement_filters(
            filters_norm, domain="image", table_name="image_measurements"
        )
        merged = merge_filters(
            structural,
            {"variable_id": resolved_variables} if resolved_variables else None,
            {"modality": modality} if modality else None,
            {"region_id": region_filter} if region_filter else None,
        )

        has_pipeline_in_filters = "pipeline_id" in structural
        explicit_pipeline_ids: list[str] | None = None
        if not has_pipeline_in_filters:
            explicit_pipeline_ids = self.catalog.resolve_pipeline_selector(pipeline)
            if explicit_pipeline_ids is not None:
                merged = merge_filters(merged, {"pipeline_id": explicit_pipeline_ids})
            elif modality is not None:
                dpid = self.catalog.default_pipeline_id(str(modality))
                if dpid:
                    merged = merge_filters(merged, {"pipeline_id": dpid})

        post_filter_defaults = (
            not has_pipeline_in_filters
            and explicit_pipeline_ids is None
            and modality is None
            and pipeline is None
        )

        force_parquet = bool(var_specs) or post_filter_defaults

        df = self._load_table_frame(
            "image_measurements",
            filters=merged,
            use_sqlite=use_sqlite,
            force_parquet=force_parquet,
        )
        if var_specs:
            df = _apply_variable_value_filters(df, var_specs)
        if post_filter_defaults:
            df = self._filter_image_rows_to_catalog_defaults(df)
        return self._prepare_measurements(
            df,
            wide=wide,
            table_name="image_measurements",
            image_wide_single_variable=_image_wide_single_variable_from_request(variables),
        )

    def _filter_image_rows_to_catalog_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep rows where each modality with a catalog default matches that pipeline_id; other modalities pass through."""
        if df.empty or "modality" not in df.columns:
            return df
        if "pipeline_id" not in df.columns:
            return df
        defaults: dict[str, str] = {}
        for e in self.catalog.pipelines_manifest.get("pipelines", []):
            if not e.get("modality") or not coerce_bool(e.get("is_default")):
                continue
            defaults[str(e["modality"]).strip().lower()] = str(e["pipeline_id"])

        def row_ok(row: pd.Series) -> bool:
            m = str(row.get("modality", "")).strip().lower() if pd.notna(row.get("modality")) else ""
            if m not in defaults:
                return True
            pid = row.get("pipeline_id")
            if pd.isna(pid) or str(pid).strip() == "":
                return False
            return str(pid) == defaults[m]

        mask = df.apply(row_ok, axis=1)
        return df.loc[mask].copy()

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

    def drop_table(self, name: str, *, remove_sqlite: bool = True) -> None:
        self.catalog.clear_table(name)
        if remove_sqlite and self.sqlite.exists():
            self.sqlite.db_path.unlink()

    def drop_all_tables(self, *, remove_sqlite: bool = True) -> None:
        for table_name in self.catalog.list_tables():
            self.drop_table(table_name, remove_sqlite=False)
        if remove_sqlite and self.sqlite.exists():
            self.sqlite.db_path.unlink()

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
        df = coerce_dataframe_to_manifest(df, definition.columns)
        return apply_filters(df, filters)

    def _prepare_measurements(
        self,
        df: pd.DataFrame,
        *,
        wide: bool,
        table_name: str,
        image_wide_single_variable: bool | None = None,
    ) -> pd.DataFrame:
        df = self._resolve_measurement_values(df)
        if wide:
            definition = self.catalog.get_table(table_name)
            kw: dict[str, Any] = {}
            if table_name == "image_measurements":
                kw["image_wide_single_variable"] = (
                    False if image_wide_single_variable is None else image_wide_single_variable
                )
            return self._to_wide(df, definition, **kw)
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

    def _to_wide(
        self,
        df: pd.DataFrame,
        definition: TableDefinition,
        *,
        image_wide_single_variable: bool | None = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        index_columns = [column for column in definition.wide_index_columns if column in df.columns]
        key_columns = [column for column in definition.wide_key_columns if column in df.columns]
        if not index_columns or not key_columns or "value" not in df.columns:
            return df.copy()

        tmp = df.copy()
        # pandas pivot_table drops all rows when any index level is entirely NA/NaN (common for
        # optional session_id / visit_id). Normalize missing index values so pivot preserves rows.
        for column in index_columns:
            series = tmp[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                tmp[column] = series.astype("string").fillna("")
            else:
                tmp[column] = series.astype("string").fillna("")
        if definition.name == "image_measurements":
            tmp["_wide_key"] = _compose_image_wide_keys(
                tmp,
                single_variable=bool(image_wide_single_variable),
            )
        else:
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
