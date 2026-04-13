from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .asl_atlases import regions_for_atlas
from .catalog import DatasetCatalog, TableDefinition
from .exceptions import FilterError
from .filters import apply_filters, ensure_list, merge_filters
from .xnat_config import XnatConnectionConfig, load_xnat_profile, resolve_xnat_connection
from .sqlite_index import SQLiteIndex
from .storage import coerce_bool, coerce_dataframe_to_manifest, empty_dataframe, normalize_variable_id, read_parquet_table, utc_now_iso, write_parquet_table


def _default_dataset_root() -> Path:
    env_root = os.getenv("NVITK_DATASET_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "dataset"


def _root_path_for_imread(path: str | Path, *, asset_type: str) -> Path:
    """Directory or file path to pass to :func:`nvitk.io.imageio.imread` for ``force_type``."""
    p = Path(path)
    t = str(asset_type).strip().lower()
    if t == "dicom":
        if p.is_dir():
            return p
        return p.parent
    return p


def _dedupe_imread_jobs(jobs: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[Path, str]] = []
    for p, ft in jobs:
        key = (p.resolve().as_posix(), str(ft).strip().lower())
        if key not in seen:
            seen.add(key)
            out.append((p, ft))
    return out


def _imread_stack_from_jobs(jobs: list[tuple[Path, str]]) -> Any:
    from nvitk.io.imageio import imread

    if not jobs:
        return []
    outs: list[Any] = []
    for p, ft in jobs:
        outs.append(imread(str(p), force_type=str(ft).strip().lower()))
    if len(outs) == 1:
        return outs[0]
    return outs


def _dicom_cache_dir_has_files(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for child in directory.iterdir():
        if not child.is_file():
            continue
        low = child.name.lower()
        if low.startswith("."):
            continue
        suf = child.suffix.lower()
        if suf in {".dcm", ".dicom", ".ima", ".img"} or suf == "":
            return True
    return False


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
        use_sqlite: bool = True,
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
        use_sqlite: bool | None = True,
        force_parquet: bool = False,
    ) -> pd.DataFrame:
        definition = self.catalog.get_table(table)
        effective_sqlite = (self.use_sqlite if use_sqlite is None else use_sqlite) and not force_parquet

        if effective_sqlite and self.sqlite.exists():
            try:
                df = self.sqlite.query_table(table, columns=columns, filters=filters)
            except Exception:
                warnings.warn(f"Error querying table {table} with SQLite index. Falling back to parquet read.", stacklevel=2)
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
        use_sqlite: bool | None = True,
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
        use_sqlite: bool | None = True,
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
        use_sqlite: bool | None = True,
        pipeline: str | int | Iterable[str | int] | None = None,
    ) -> pd.DataFrame:
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
            explicit_pipeline_ids = self.catalog.resolve_pipeline_selector(pipeline, modality=modality)
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

    def _assets_to_wide(self, df: pd.DataFrame, *, value_column: str = "asset_path") -> pd.DataFrame:
        """One row per ``subject_uid``; column names are ``asset_slot`` values (paths in cells by default)."""
        if df.empty:
            return df.copy()
        if "subject_uid" not in df.columns:
            return df.copy()
        if "asset_slot" not in df.columns:
            return df[["subject_uid"]].drop_duplicates().reset_index(drop=True)
        if value_column not in df.columns:
            return df.copy()

        work = df.copy()
        slot = work["asset_slot"].astype("string")
        mask = slot.notna() & (slot.str.strip() != "")
        work = work.loc[mask]
        if work.empty:
            return pd.DataFrame(columns=["subject_uid"])

        if "updated_at" in work.columns:
            work = work.sort_values("updated_at", ascending=False, na_position="last")
        work = work.drop_duplicates(subset=["subject_uid", "asset_slot"], keep="first")

        wide = work.pivot_table(
            index="subject_uid",
            columns="asset_slot",
            values=value_column,
            aggfunc="first",
        )
        wide.columns = [str(c) for c in wide.columns]
        return wide.reset_index()

    def assets(
        self,
        *,
        filters: dict[str, Any] | None = None,
        modality: str | None = None,
        asset_type: str | None = None,
        resource_label: str | None = None,
        subject_uid: str | None = None,
        session_uid: str | None = None,
        pipeline_id: str | None = None,
        asset_slot: str | None = None,
        source: str | None = None,
        wide: bool = False,
        use_sqlite: bool | None = True,
        value: str = "asset_path",
        get_image: bool = False,
    ) -> pd.DataFrame | Any:
        merged = merge_filters(
            dict(filters or {}),
            {"modality": modality} if modality else None,
            {"asset_type": asset_type} if asset_type else None,
            {"resource_label": resource_label} if resource_label else None,
            {"subject_uid": subject_uid} if subject_uid else None,
            {"session_uid": session_uid} if session_uid else None,
            {"pipeline_id": pipeline_id} if pipeline_id else None,
            {"asset_slot": asset_slot} if asset_slot else None,
            {"source": source} if source else None,
        )
        if get_image:
            if wide:
                raise FilterError("get_image=True is incompatible with wide=True; use wide=False.")
            if not modality or not str(modality).strip():
                raise FilterError("get_image=True requires modality=... (non-empty).")
            if not asset_type or not str(asset_type).strip():
                raise FilterError("get_image=True requires asset_type=... (non-empty).")

        df = self._load_table_frame("assets", filters=merged, use_sqlite=use_sqlite)
        df = df.reset_index(drop=True)

        if get_image:
            if df.empty:
                return []
            if value not in df.columns:
                raise FilterError(f"Unknown value column for asset images: {value!r}.")

            jobs: list[tuple[Path, str]] = []
            for i in range(len(df)):
                row = df.iloc[i]
                raw = row[value]
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    raise FilterError(f"Asset row has no path in column {value!r}.")
                force = row["asset_type"] if "asset_type" in row.index and pd.notna(row.get("asset_type")) else asset_type
                if force is None or (isinstance(force, float) and pd.isna(force)):
                    force = asset_type
                ft = str(force).strip().lower() if force is not None else ""
                if not ft:
                    raise FilterError("Cannot determine asset_type for imread(force_type=...).")
                jobs.append((_root_path_for_imread(str(raw), asset_type=ft), ft))

            jobs = _dedupe_imread_jobs(jobs)
            return _imread_stack_from_jobs(jobs)

        if not wide:
            return df
        if value not in df.columns:
            raise FilterError(f"Unknown value column for assets wide form: {value!r}.")
        return self._assets_to_wide(df, value_column=value)

    def asset(
        self,
        *,
        filters: dict[str, Any] | None = None,
        modality: str | None = None,
        asset_type: str | None = None,
        resource_label: str | None = None,
        subject_uid: str | None = None,
        session_uid: str | None = None,
        pipeline_id: str | None = None,
        asset_slot: str | None = None,
        source: str | None = None,
        wide: bool = True,
        use_sqlite: bool | None = True,
        value: str = "asset_path",
        get_image: bool = False,
    ) -> pd.DataFrame | Any:
        """Assets as a wide table (one row per subject) by default; see :meth:`assets`."""
        return self.assets(
            filters=filters,
            modality=modality,
            asset_type=asset_type,
            resource_label=resource_label,
            subject_uid=subject_uid,
            session_uid=session_uid,
            pipeline_id=pipeline_id,
            asset_slot=asset_slot,
            source=source,
            wide=wide,
            use_sqlite=use_sqlite,
            value=value,
            get_image=get_image,
        )

    def scans(
        self,
        *,
        filters: dict[str, Any] | None = None,
        scan_uid: str | None = None,
        session_uid: str | None = None,
        subject_uid: str | None = None,
        scan_id: str | None = None,
        modality: str | None = None,
        orientation: str | None = None,
        resource_label: str | None = None,
        asset_slot: str | None = None,
        quality: str | None = None,
        series_description: str | None = None,
        source_batch_id: str | None = None,
        use_sqlite: bool | None = True,
        path_column: str = "local_cache_path",
        get_image: bool = False,
        asset_type: str | None = None,
        download_scan_path: str | Path | None = None,
        xnat_config: XnatConnectionConfig | None = None,
        skip_existing_download: bool = True,
    ) -> pd.DataFrame | Any:
        """Query the ``scans`` inventory table with optional keyword filters (like :meth:`assets`)."""
        merged = merge_filters(
            dict(filters or {}),
            {"scan_uid": scan_uid} if scan_uid else None,
            {"session_uid": session_uid} if session_uid else None,
            {"subject_uid": subject_uid} if subject_uid else None,
            {"scan_id": scan_id} if scan_id else None,
            {"modality": modality} if modality else None,
            {"orientation": orientation} if orientation else None,
            {"resource_label": resource_label} if resource_label else None,
            {"asset_slot": asset_slot} if asset_slot else None,
            {"quality": quality} if quality else None,
            {"series_description": series_description} if series_description else None,
            {"source_batch_id": source_batch_id} if source_batch_id else None,
        )
        if get_image:
            if not modality or not str(modality).strip():
                raise FilterError("get_image=True requires modality=... (non-empty).")
            if not asset_type or not str(asset_type).strip():
                raise FilterError("get_image=True requires asset_type=... (non-empty).")

        df = self._load_table_frame("scans", filters=merged, use_sqlite=use_sqlite)
        df = df.reset_index(drop=True)

        if get_image:
            if path_column not in df.columns:
                raise FilterError(f"Unknown path column for scan images: {path_column!r}.")
            if df.empty:
                return []
            ft = str(asset_type).strip().lower()
            jobs: list[tuple[Path, str]] = []
            need_download: list[pd.Series] = []
            for i in range(len(df)):
                row = df.iloc[i]
                raw = row[path_column]
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    need_download.append(row)
                    continue
                jobs.append((_root_path_for_imread(str(raw), asset_type=ft), ft))

            if need_download:
                cfg = xnat_config
                if cfg is None:
                    try:
                        cfg = resolve_xnat_connection(load_xnat_profile())
                    except ValueError as exc:
                        raise FilterError(
                            "get_image=True with missing local_cache_path requires XNAT credentials: "
                            "pass xnat_config=... or set XNAT_SERVER / XNAT_PROJECT (and auth), "
                            "or use an NVITK XNAT profile."
                        ) from exc

                from nvitk.io.imageio import imread

                from .xnat import connect_xnat, download_scan_dicoms, resolve_xnat_scan_from_scan_row

                if ft != "dicom":
                    raise FilterError(
                        "On-demand download from XNAT in scans(get_image=True) is implemented for "
                        "asset_type='dicom' only; sync NIfTI assets or set local_cache_path."
                    )

                use_persistent_cache = download_scan_path is not None and str(download_scan_path).strip()
                persistent_base = Path(download_scan_path).expanduser().resolve() if use_persistent_cache else None
                if persistent_base is not None:
                    persistent_base.mkdir(parents=True, exist_ok=True)

                ephemeral_images: list[Any] = []

                with connect_xnat(cfg) as xsession:
                    for row in need_download:
                        try:
                            scan_obj = resolve_xnat_scan_from_scan_row(xsession, row.to_dict())
                        except (LookupError, ValueError) as exc:
                            raise FilterError(f"Could not resolve XNAT scan for downloads: {exc}") from exc

                        if persistent_base is not None:
                            subj = str(row.get("subject_uid") or "unknown")
                            slot = str(row.get("asset_slot") or "scan").strip() or "scan"
                            sid = str(row.get("scan_id") or "unknown")
                            dest = persistent_base / subj / slot / sid
                            dest.mkdir(parents=True, exist_ok=True)
                            if skip_existing_download and _dicom_cache_dir_has_files(dest):
                                pass
                            else:
                                download_scan_dicoms(scan_obj, dest)
                            jobs.append((_root_path_for_imread(dest, asset_type=ft), ft))
                        else:
                            with tempfile.TemporaryDirectory(prefix="nvitk_xnat_scan_") as tmp:
                                dest = Path(tmp)
                                download_scan_dicoms(scan_obj, dest)
                                ephemeral_images.append(
                                    imread(str(_root_path_for_imread(dest, asset_type=ft)), force_type=ft)
                                )

                if ephemeral_images:
                    disk_out: list[Any] = []
                    for p, fft in _dedupe_imread_jobs(jobs):
                        disk_out.append(imread(str(p), force_type=str(fft).strip().lower()))
                    combined = disk_out + ephemeral_images
                    if len(combined) == 1:
                        return combined[0]
                    return combined

            if not jobs:
                raise FilterError(
                    f"No usable paths in column {path_column!r} for the selected scans "
                    "(expected non-null local download/cache directories)."
                )
            jobs = _dedupe_imread_jobs(jobs)
            return _imread_stack_from_jobs(jobs)

        return df

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
