"""
High-level access to the on-disk dataset: catalog tables, measurements, assets, scans, and writes.

:class:`DataRepo` resolves the dataset root (``NVITK_DATASET_ROOT`` or the ``dataset/`` tree
next to the package), opens :class:`~nvitk.db.catalog.DatasetCatalog` and
:class:`~nvitk.db.sqlite_index.SQLiteIndex`, and exposes filtered queries with optional cohort
scoping (see ``DEFAULT_COHORT_ID``).
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from nvitk.core.logger import Logger

from .asl_atlases import regions_for_atlas
from .t1_atlases import regions_for_t1_atlas
from .catalog import DatasetCatalog, TableDefinition
from .exceptions import FilterError, SettingsError
from .filters import apply_filters, ensure_list, merge_filters
from .xnat_config import XnatConnectionConfig, load_xnat_profile, resolve_xnat_connection
from .sqlite_index import SQLiteIndex
from .storage import (
    MEASUREMENT_TABLE_COLUMNS,
    coerce_bool,
    coerce_dataframe_to_manifest,
    empty_dataframe,
    normalize_variable_id,
    read_parquet_table,
    restrict_to_manifest_columns,
    utc_now_iso,
    write_parquet_table,
)

log = Logger()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset root & defaults
# ──────────────────────────────────────────────────────────────────────────────


def _default_dataset_root() -> Path:
    """Resolve the dataset root: ``NVITK_DATASET_ROOT`` if set, else ``<repo>/dataset/nvitk-dataset``."""
    env_root = os.getenv("NVITK_DATASET_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "dataset" / "nvitk-dataset"


def _local_dataset_root_from_settings(db: dict[str, Any]) -> str | Path:
    """Pick the dataset root from a settings ``db`` block: ``local_fallback_root`` first, then ``root``,
    then :func:`_default_dataset_root`."""
    fallback = db.get("local_fallback_root")
    if fallback is not None and str(fallback).strip():
        log.info("Using local root: %s", fallback)
        return fallback
    root = db.get("root")
    if root is not None and str(root).strip():
        log.info("Using remote root: %s", root)
        return root
    return _default_dataset_root()


def get_repo(
    *,
    prefer_sge: bool | None = None,
    root: Path | str | None = None,
    use_sqlite: bool | None = None,
    auto_scaffold: bool | None = None,
) -> DataRepo:
    """Open :class:`DataRepo` using env, SGE root, or workstation settings."""
    from .settings_paths import load_db_settings_block, sge_dataset_root_path

    db = load_db_settings_block()
    sqlite_index = db.get("sqlite_index", True) if use_sqlite is None else use_sqlite
    auto_scaff = db.get("auto_scaffold", False) if auto_scaffold is None else auto_scaffold

    if root is not None:
        return DataRepo(
            root=Path(root).expanduser().resolve(),
            use_sqlite=sqlite_index,
            auto_scaffold=auto_scaff,
        )

    use_sge = prefer_sge
    if use_sge is None:
        use_sge = os.environ.get("NVITK_SGE", "").lower() in ("1", "true", "yes")

    if use_sge:
        sge_root = sge_dataset_root_path(must_exist=True)
        if sge_root is not None:
            log.info("Using SGE dataset root: %s", sge_root)
            return DataRepo(root=sge_root, use_sqlite=sqlite_index, auto_scaffold=auto_scaff)

    env_root = os.getenv("NVITK_DATASET_ROOT", "").strip()
    if env_root:
        return DataRepo(
            root=Path(env_root).expanduser().resolve(),
            use_sqlite=sqlite_index,
            auto_scaffold=auto_scaff,
        )

    local_root = _local_dataset_root_from_settings(db)
    return DataRepo(
        root=Path(local_root).expanduser().resolve(),
        use_sqlite=sqlite_index,
        auto_scaffold=auto_scaff,
    )


def get_repo_from_settings(return_xnat_config: bool = False) -> DataRepo | XnatConnectionConfig:
    """Get :class:`DataRepo` from settings file, optionally with :class:`XnatConnectionConfig`."""
    try:
        from .settings_paths import load_db_settings_block

        db = load_db_settings_block()
        if not db:
            raise SettingsError("No db settings found (.nvitk/settings.json missing or empty)")

        repo = get_repo(use_sqlite=db.get("sqlite_index"), auto_scaffold=db.get("auto_scaffold"))
        if return_xnat_config:
            if "xnat_config" in db:
                _net_file, _urs, _pwd = None, None, None
                try:
                    _net_file = db["xnat_config"]["netrc_file"]
                except Exception:
                    _urs, _pwd = db["xnat_config"]["user"], db["xnat_config"]["password"]
                finally:
                    if _net_file is None and (_urs is None or _pwd is None):
                        raise SettingsError("XNAT config requires netrc_file or user and password")
                return repo, XnatConnectionConfig(
                    server=db["xnat_config"]["server"],
                    project=db["xnat_config"]["project"],
                    netrc_file=_net_file,
                    user=_urs,
                    password=_pwd,
                    verify=db["xnat_config"]["verify"],
                )
        return repo
    except SettingsError:
        raise
    except Exception as e:
        import traceback

        log.warning(traceback.format_exc())
        raise SettingsError(f"Error getting repo from settings: {e}") from e


# Default cohort for API queries when ``cohort_id`` is omitted (see :meth:`DataRepo._resolve_cohort`).
DEFAULT_COHORT_ID = "PESA-Brain"


# ──────────────────────────────────────────────────────────────────────────────
# Local paths & imread job helpers
# ──────────────────────────────────────────────────────────────────────────────


def _root_path_for_imread(path: str | Path, *, asset_type: str) -> Path:
    """Directory or file path to pass to :func:`nvitk.io.imageio.imread` for ``force_type``."""
    p = Path(path)
    t = str(asset_type).strip().lower()
    if t == "dicom":
        if p.is_dir():
            return p
        return p.parent
    return p


def _nifti_files_in_dir(directory: Path) -> list[Path]:
    """Sorted list of ``.nii`` / ``.nii.gz`` files directly under ``directory``."""
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for child in directory.iterdir():
        if not child.is_file():
            continue
        low = child.name.lower()
        if low.startswith("."):
            continue
        if low.endswith(".nii.gz") or low.endswith(".nii"):
            out.append(child)
    return sorted(out, key=lambda p: p.name.lower())


def _nifti_cache_dir_has_files(directory: Path) -> bool:
    """True if *directory* contains at least one cached ``.nii``/``.nii.gz`` file."""
    return bool(_nifti_files_in_dir(directory))


def _imread_jobs_for_scan_path(raw: str | Path, *, asset_type: str) -> list[tuple[Path, str]]:
    """One or more ``(path, force_type)`` jobs for a local cache path (expands NIfTI directories)."""
    p = Path(str(raw)).expanduser().resolve()
    ft = str(asset_type).strip().lower()
    if ft == "nifti":
        if p.is_file():
            return [(p, "nifti")]
        if p.is_dir():
            files = _nifti_files_in_dir(p)
            if not files:
                raise FilterError(f"No NIfTI (.nii/.nii.gz) files found under {p!r}.")
            return [(f, "nifti") for f in files]
        raise FilterError(f"Invalid NIfTI path: {raw!r}")
    if ft == "dicom":
        return [(_root_path_for_imread(p, asset_type="dicom"), "dicom")]
    return [(p, ft)]


def _dedupe_imread_jobs(jobs: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Drop duplicate ``(path, force_type)`` jobs, keyed on resolved absolute path and lower-cased type,
    preserving first-seen order."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[Path, str]] = []
    for p, ft in jobs:
        key = (p.resolve().as_posix(), str(ft).strip().lower())
        if key not in seen:
            seen.add(key)
            out.append((p, ft))
    return out


def _metadata_json_path_for_nifti(nifti_path: Path) -> Path | None:
    """If a sibling JSON sidecar exists next to ``nifti_path``, return its path (for ``imread(metadata_json=...)``)."""
    from nvitk.io.readers.nifti import nifti_metadata_json_path

    jp = nifti_metadata_json_path(nifti_path)
    return jp if jp.is_file() else None


def _imread_kwargs_with_nifti_sidecar(path: Path, force_type: str, base: dict[str, Any]) -> dict[str, Any]:
    """Merge ``imread`` kwargs; for NIfTI, set ``metadata_json`` when a per-file sidecar exists (overrides generic ``metadata_json``)."""
    kw = dict(base)
    if str(force_type).strip().lower() == "nifti":
        mj = _metadata_json_path_for_nifti(path)
        if mj is not None:
            kw["metadata_json"] = str(mj)
    return kw


def _imread_stack_from_jobs(jobs: list[tuple[Path, str]], **imread_kwargs: Any) -> Any:
    """Read every ``(path, force_type)`` job with :func:`nvitk.io.imageio.imread`, attaching a NIfTI
    sidecar's ``metadata_json`` when present. Returns a single image for one job, otherwise a list."""
    from nvitk.io.imageio import imread

    if not jobs:
        return []
    outs: list[Any] = []
    base_kw = dict(imread_kwargs)
    for p, ft in jobs:
        ft_l = str(ft).strip().lower()
        kw = _imread_kwargs_with_nifti_sidecar(p, ft_l, base_kw)
        outs.append(imread(str(p), force_type=ft_l, **kw))
    if len(outs) == 1:
        return outs[0]
    return outs


def _dicom_cache_dir_has_files(directory: Path) -> bool:
    """True if *directory* contains at least one non-hidden file that looks like a DICOM slice
    (``.dcm``/``.dicom``/``.ima``/``.img`` extension, or no extension at all)."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Measurement entity keys & variable-value filters
# ──────────────────────────────────────────────────────────────────────────────


def _pick_value_column_for_spec(spec: Any) -> str:
    """Choose which measurement value column a filter *spec* should be applied against: ``value_text``
    for string specs or a ``$contains``-style operator, ``value_num`` otherwise."""
    if isinstance(spec, Mapping):
        for raw_op in spec:
            op = str(raw_op).strip().lower().lstrip("$")
            if op in {"contains"}:
                return "value_text"
        return "value_num"
    if isinstance(spec, str):
        return "value_text"
    return "value_num"


def _normalize_visit_like_for_key(series: pd.Series) -> pd.Series:
    """Normalize visit_id / session_id so 4, 4.0, and '4' compare as the same key part."""
    num = pd.to_numeric(series, errors="coerce")
    out = series.astype("string").fillna("")
    intlike = num.notna() & (num == num.round())
    out = out.copy()
    out.loc[intlike] = num.loc[intlike].round().astype("Int64").astype(str)
    return out


def _normalize_frame_index_for_key(series: pd.Series, *, index: pd.Index) -> pd.Series:
    """Format ``frame_index`` as a string for entity-key composition; non-zero values become their
    rounded integer string, missing/zero values become ``""`` (frame 0 is treated as "no frame")."""
    fi = pd.to_numeric(series, errors="coerce")
    fr = pd.Series("", index=index, dtype="string")
    mask = fi.notna() & (fi != 0)
    fr.loc[mask] = fi.loc[mask].round().astype("Int64").astype(str)
    return fr


def _measurement_entity_tuple_series(df: pd.DataFrame) -> pd.Series:
    """One hashable tuple per row identifying a measurement row (excluding ``variable_id``)."""
    if df.empty:
        return pd.Series(dtype=object)
    idx = df.index
    parts: list[pd.Series] = [df["subject_uid"].astype("string").fillna("")]
    if "visit_id" in df.columns:
        parts.append(_normalize_visit_like_for_key(df["visit_id"]))
    elif "session_id" in df.columns:
        parts.append(_normalize_visit_like_for_key(df["session_id"]))
    for col in ("modality", "pipeline_id", "region_id"):
        if col in df.columns:
            parts.append(df[col].astype("string").fillna(""))
    if "frame_index" in df.columns:
        parts.append(_normalize_frame_index_for_key(df["frame_index"], index=idx))
    if len(parts) == 1:
        return parts[0]
    return pd.Series(
        pd.MultiIndex.from_arrays([p.array for p in parts]).tolist(),
        index=idx,
        dtype=object,
    )


def _entity_keys_from_frame(df: pd.DataFrame) -> set[tuple[str, ...] | str]:
    """Set of distinct entity keys (see :func:`_measurement_entity_tuple_series`) present in *df*."""
    if df.empty:
        return set()
    s = _measurement_entity_tuple_series(df)
    return set(s.tolist())


def _frame_matches_entity_keys(df: pd.DataFrame, keys: set[tuple[str, ...] | str]) -> pd.Series:
    """Boolean mask over *df* rows whose entity key is in *keys*."""
    s = _measurement_entity_tuple_series(df)
    return s.isin(keys)


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


def _ordered_unique(values: Iterable[str]) -> list[str]:
    """De-duplicate *values*, keeping first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _resolve_requested_variables_with_alias_map(
    catalog: DatasetCatalog,
    requested: str | Iterable[str] | None,
    *,
    domain: str,
) -> tuple[list[str], dict[str, str]]:
    """Resolve *requested* variable names/aliases to canonical ids via ``catalog``. Returns the ordered,
    de-duplicated canonical ids plus a ``{canonical: originally_requested_alias}`` map."""
    if requested is None:
        return [], {}
    raw_values = [requested] if isinstance(requested, str) else list(requested)
    cleaned = [str(item).strip() for item in raw_values if str(item).strip()]
    if not cleaned:
        return [], {}
    resolved = catalog.resolve_variable_ids(cleaned, domain=domain)
    alias_map: dict[str, str] = {}
    ordered_canonicals: list[str] = []
    for raw, canonical in zip(cleaned, resolved):
        c = str(canonical).strip()
        if not c:
            continue
        ordered_canonicals.append(c)
        if c not in alias_map:
            alias_map[c] = raw
    return _ordered_unique(ordered_canonicals), alias_map


def _resolve_variable_id_filter_spec(
    catalog: DatasetCatalog,
    *,
    domain: str,
    spec: Any,
) -> tuple[Any, dict[str, str]]:
    """
    Resolve aliases inside a structural ``variable_id`` filter spec to canonical ids.
    Returns ``(resolved_spec, canonical_to_requested_alias)``.
    """

    alias_map: dict[str, str] = {}

    def resolve_tokens(values: list[Any]) -> list[str]:
        """Resolve a list of raw variable tokens to canonical ids, recording aliases as a side effect."""
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if not cleaned:
            return []
        resolved = catalog.resolve_variable_ids(cleaned, domain=domain)
        out: list[str] = []
        for raw, canonical in zip(cleaned, resolved):
            c = str(canonical).strip()
            if not c:
                continue
            out.append(c)
            if c not in alias_map:
                alias_map[c] = raw
        return _ordered_unique(out)

    if isinstance(spec, Mapping):
        out_spec: dict[str, Any] = {}
        for op, value in spec.items():
            if isinstance(value, (list, tuple, set)):
                out_spec[op] = resolve_tokens(list(value))
            elif isinstance(value, str):
                resolved_one = resolve_tokens([value])
                out_spec[op] = resolved_one[0] if resolved_one else value
            else:
                out_spec[op] = value
        return out_spec, alias_map

    if isinstance(spec, (list, tuple, set)):
        return resolve_tokens(list(spec)), alias_map
    if isinstance(spec, str):
        resolved_one = resolve_tokens([spec])
        return (resolved_one[0] if resolved_one else spec), alias_map
    return spec, alias_map


def _rename_image_wide_column_with_alias(name: str, canonical_to_alias: dict[str, str]) -> str:
    """Rewrite a wide-pivot image-measurement column name so its trailing canonical variable id
    (optionally followed by a ``_f<frame>`` suffix) is replaced by the requested alias."""
    if not canonical_to_alias:
        return name
    ordered = sorted(canonical_to_alias.items(), key=lambda item: len(item[0]), reverse=True)
    for canonical, alias in ordered:
        if alias == canonical:
            continue
        if name.endswith(f"_{canonical}"):
            return name[: -len(canonical)] + alias
        match = re.match(rf"^(.*)_{re.escape(canonical)}(_f\d+)$", name)
        if match:
            return f"{match.group(1)}_{alias}{match.group(2)}"
    return name


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
    """Restrict *df* to entities (subject/visit/... tuples) whose measurement value satisfies every
    ``(variable_id, filter_spec)`` in *specs*, evaluating each variable's filter against its own rows
    before intersecting entity keys back onto the full frame."""
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
        if filtered.empty and value_col == "value_num" and "value_text" in sub.columns:
            tmp = sub.copy()
            tmp["value_num"] = pd.to_numeric(tmp["value_text"], errors="coerce")
            filtered = apply_filters(tmp, {value_col: spec})
        if filtered.empty:
            return df.iloc[0:0].copy()
        key_sets.append(_entity_keys_from_frame(filtered))
    allowed = set.intersection(*key_sets)
    if not allowed:
        return df.iloc[0:0].copy()
    return df.loc[_frame_matches_entity_keys(df, allowed)].copy()


# ──────────────────────────────────────────────────────────────────────────────
# DataRepo
# ──────────────────────────────────────────────────────────────────────────────


class DataRepo:
    """
    Dataset-facing API: load tables, clinical/image measurements, assets, scans, and writes.

    Queries prefer the SQLite index when ``use_sqlite`` is True and the index file exists;
    otherwise fall back to Parquet. Cohort filtering uses ``cohort_membership`` when present
    unless ``cohort_id=False`` (disable) or an explicit cohort id is passed.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        use_sqlite: bool = True,
        auto_scaffold: bool = False,
    ):
        """
        Parameters
        ----------
        root
            Dataset root directory; defaults to ``NVITK_DATASET_ROOT`` or the ``dataset/`` folder.
        use_sqlite
            When True, use :class:`~nvitk.db.sqlite_index.SQLiteIndex` for fast filtered reads.
        auto_scaffold
            If True and ``catalog/repository.json`` is missing, create a minimal dataset scaffold.
        """
        dataset_root = Path(root or _default_dataset_root()).expanduser().resolve()
        if auto_scaffold and not (dataset_root / "catalog" / "repository.json").exists():
            DatasetCatalog.create_scaffold(dataset_root)

        self.root = dataset_root
        self.catalog = DatasetCatalog(dataset_root)
        self.sqlite = SQLiteIndex(self.catalog.sqlite_index_path)
        self.use_sqlite = use_sqlite

    def list_tables(self) -> list[str]:
        """Registered table names from the catalog manifest."""
        return self.catalog.list_tables()

    def _resolve_cohort(
        self,
        cohort_id: str | bool | None,
        filters: Mapping[str, Any] | None,
    ) -> tuple[str | bool, dict[str, Any]]:
        """Return ``(effective_cohort, filters_without_cohort_id)``. ``False`` = do not filter by cohort."""
        f = dict(filters or {})
        from_filters = f.pop("cohort_id", None)
        c = cohort_id
        if c is None:
            c = from_filters
        if c is False:
            return False, f
        if c is None or (isinstance(c, str) and not str(c).strip()):
            return DEFAULT_COHORT_ID, f
        return str(c).strip(), f

    def _cohort_subject_uid_set(self, cohort_id: str) -> set[str] | None:
        """Subjects in ``cohort_membership`` for ``cohort_id``; ``None`` if table missing or empty (no filtering)."""
        if not self.catalog.table_exists("cohort_membership"):
            return None
        try:
            cm = self._load_table_frame(
                "cohort_membership",
                filters={"cohort_id": cohort_id},
                use_sqlite=None,
            )
        except Exception:
            return None
        if cm.empty or "subject_uid" not in cm.columns:
            return None
        return set(cm["subject_uid"].astype("string").str.strip())

    def _filter_dataframe_by_cohort(self, df: pd.DataFrame, cohort_id: str) -> pd.DataFrame:
        """Restrict *df* to rows whose ``subject_uid`` belongs to *cohort_id*; no-op if cohort membership
        is unavailable or *df* has no ``subject_uid`` column."""
        if df.empty or "subject_uid" not in df.columns:
            return df
        uids = self._cohort_subject_uid_set(cohort_id)
        if uids is None:
            return df
        s = df["subject_uid"].astype("string").fillna("")
        return df.loc[s.isin(uids)].copy()

    def _measurement_column_names(self, table_name: str) -> set[str]:
        """Column names declared in the catalog manifest for *table_name*."""
        return set(self.catalog.get_table(table_name).columns.keys())

    def _split_measurement_filters(
        self,
        filters: dict[str, Any] | None,
        *,
        domain: str,
        table_name: str,
    ) -> tuple[dict[str, Any], list[tuple[str, Any]], dict[str, str]]:
        """Split a user-supplied ``filters`` dict into structural column filters (applied directly to the
        measurement table) and per-variable value filters (applied against ``value_num``/``value_text``
        rows for a specific ``variable_id``), resolving any variable aliases along the way. Returns
        ``(structural_filters, [(variable_id, spec), ...], canonical_to_alias)``."""
        if not filters:
            return {}, [], {}
        columns = self._measurement_column_names(table_name)
        definition = self.catalog.get_table(table_name)
        reserved = set(definition.key_columns) | set(definition.index_columns)
        canonical_var_ids = {str(e["variable_id"]) for e in self.catalog.variable_entries(domain=domain)}
        structural: dict[str, Any] = {}
        variable_specs: list[tuple[str, Any]] = []
        alias_map: dict[str, str] = {}
        for key, spec in filters.items():
            if not isinstance(key, str) or not key.strip():
                raise FilterError("Filter columns must be non-empty strings.")
            k = key.strip()
            if k in reserved:
                if k == "variable_id":
                    resolved_spec, resolved_aliases = _resolve_variable_id_filter_spec(
                        self.catalog, domain=domain, spec=spec
                    )
                    structural[k] = resolved_spec
                    for canonical, requested in resolved_aliases.items():
                        if canonical not in alias_map:
                            alias_map[canonical] = requested
                else:
                    structural[k] = spec
                continue
            if k == "variable_id":
                resolved_spec, resolved_aliases = _resolve_variable_id_filter_spec(
                    self.catalog, domain=domain, spec=spec
                )
                structural[k] = resolved_spec
                for canonical, requested in resolved_aliases.items():
                    if canonical not in alias_map:
                        alias_map[canonical] = requested
                continue
            resolved = self.catalog.resolve_variable_ids([k], domain=domain)[0]
            if resolved in canonical_var_ids:
                variable_specs.append((str(resolved), spec))
                continue
            if k in columns:
                structural[k] = spec
                continue
            variable_specs.append((str(resolved), spec))
        return structural, variable_specs, alias_map

    def _rename_measurement_wide_columns(
        self,
        df: pd.DataFrame,
        *,
        table_name: str,
        canonical_to_alias: dict[str, str],
    ) -> pd.DataFrame:
        """Rename wide-pivot columns from canonical variable ids back to the aliases the caller requested
        (image-measurement columns via suffix rewriting, other tables via a direct column rename)."""
        if df.empty or not canonical_to_alias:
            return df
        out = df.copy()
        rename_map: dict[str, str] = {}
        if table_name == "image_measurements":
            for column in out.columns:
                renamed = _rename_image_wide_column_with_alias(str(column), canonical_to_alias)
                if renamed != column:
                    rename_map[column] = renamed
        else:
            for canonical, alias in canonical_to_alias.items():
                if canonical != alias and canonical in out.columns:
                    rename_map[canonical] = alias
        if rename_map:
            out = out.rename(columns=rename_map)
        return out

    def _load_table_frame(
        self,
        table: str,
        *,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        use_sqlite: bool | None = True,
        force_parquet: bool = False,
    ) -> pd.DataFrame:
        """Load *table* via SQLite (if enabled/available) with a Parquet fallback on query failure,
        coercing the result to the manifest's declared column dtypes."""
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
        cohort_id: str | bool | None = None,
    ) -> pd.DataFrame:
        """
        Load *table* as a DataFrame with optional column projection, filters, and wide pivot.

        For ``clinical_measurements`` / ``image_measurements`` / ``cognitive_measurements``,
        ``wide=True`` resolves values and pivots to one column per variable (and entity keys);
        see catalog wide definitions.
        """
        definition = self.catalog.get_table(table)
        effective_sqlite = self.use_sqlite if use_sqlite is None else use_sqlite
        cohort_eff, clean_filters = self._resolve_cohort(cohort_id, filters)

        if effective_sqlite and self.sqlite.exists():
            try:
                df = self.sqlite.query_table(table, columns=columns, filters=clean_filters)
            except Exception:
                df = self._read_table(definition, columns=columns, filters=clean_filters)
        else:
            df = self._read_table(definition, columns=columns, filters=clean_filters)

        df = coerce_dataframe_to_manifest(df, definition.columns)
        if cohort_eff is not False:
            df = self._filter_dataframe_by_cohort(df, str(cohort_eff))
        if wide:
            if table in {"clinical_measurements", "image_measurements", "cognitive_measurements"}:
                df = self._resolve_measurement_values(df)
            image_sv = False if table == "image_measurements" else None
            return self._to_wide(df, definition, image_wide_single_variable=image_sv)
        return df.reset_index(drop=True)

    def cognitive(
        self,
        *,
        variables: str | Iterable[str] | None = None,
        filters: dict[str, Any] | None = None,
        wide: bool = True,
        use_sqlite: bool | None = True,
        cohort_id: str | bool | None = None,
    ) -> pd.DataFrame:
        """
        Query ``cognitive_measurements`` like :meth:`clinical` but with ``domain='cognitive'``.
        """
        if not self.catalog.table_exists("cognitive_measurements"):
            return pd.DataFrame()
        cohort_eff, filters_clean = self._resolve_cohort(cohort_id, filters)
        resolved_variables, req_aliases = _resolve_requested_variables_with_alias_map(
            self.catalog, variables, domain="cognitive"
        )
        structural, var_specs, filter_aliases = self._split_measurement_filters(
            filters_clean, domain="cognitive", table_name="cognitive_measurements"
        )
        alias_map: dict[str, str] = dict(req_aliases)
        for canonical, requested in filter_aliases.items():
            if canonical not in alias_map:
                alias_map[canonical] = requested
        merged = merge_filters(structural, {"variable_id": resolved_variables} if resolved_variables else None)
        force_parquet = bool(var_specs)
        df = self._load_table_frame(
            "cognitive_measurements",
            filters=merged,
            use_sqlite=use_sqlite,
            force_parquet=force_parquet,
        )
        if var_specs:
            df = _apply_variable_value_filters(df, var_specs)
        if cohort_eff is not False:
            df = self._filter_dataframe_by_cohort(df, str(cohort_eff))
        out = self._prepare_measurements(df, wide=wide, table_name="cognitive_measurements")
        if wide:
            out = self._rename_measurement_wide_columns(
                out, table_name="cognitive_measurements", canonical_to_alias=alias_map
            )
        return out

    def clinical(
        self,
        *,
        variables: str | Iterable[str] | None = None,
        filters: dict[str, Any] | None = None,
        wide: bool = True,
        use_sqlite: bool | None = True,
        cohort_id: str | bool | None = None,
    ) -> pd.DataFrame:
        """
        Query ``clinical_measurements`` with variable resolution, optional value filters, and cohort scope.

        Non-reserved *filters* keys are treated as variable ids with comparison specs (see
        :func:`~nvitk.db.filters.apply_filters`). Set ``wide=False`` for long (tidy) form.
        """
        cohort_eff, filters_clean = self._resolve_cohort(cohort_id, filters)
        resolved_variables, req_aliases = _resolve_requested_variables_with_alias_map(
            self.catalog, variables, domain="clinical"
        )
        structural, var_specs, filter_aliases = self._split_measurement_filters(
            filters_clean, domain="clinical", table_name="clinical_measurements"
        )
        alias_map: dict[str, str] = dict(req_aliases)
        for canonical, requested in filter_aliases.items():
            if canonical not in alias_map:
                alias_map[canonical] = requested
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
        if cohort_eff is not False:
            df = self._filter_dataframe_by_cohort(df, str(cohort_eff))
        out = self._prepare_measurements(df, wide=wide, table_name="clinical_measurements")
        if wide:
            out = self._rename_measurement_wide_columns(
                out, table_name="clinical_measurements", canonical_to_alias=alias_map
            )
        return out

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
        cohort_id: str | bool | None = None,
    ) -> pd.DataFrame:
        """
        Query ``image_measurements`` with modality, regions, atlas (ASL or T1), pipeline selection, and filters.

        When neither modality nor pipeline filters are given, rows are restricted to catalog
        default pipelines per modality where defined. Use ``atlas=`` with ``modality='asl'`` or
        ``modality='t1'`` to expand *regions* from a named atlas preset.
        """
        cohort_eff, filters_norm = self._resolve_cohort(cohort_id, filters)
        if "pipeline_version" in filters_norm and "pipeline_id" not in filters_norm:
            filters_norm["pipeline_id"] = filters_norm.pop("pipeline_version")

        if atlas is not None:
            mod_l = str(modality).strip().lower() if modality is not None else ""
            if mod_l == "asl":
                atlas_regions = regions_for_atlas(atlas)
            elif mod_l == "t1":
                atlas_regions = regions_for_t1_atlas(atlas)
            else:
                raise FilterError("Parameter 'atlas' requires modality='asl' or modality='t1'.")
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

        resolved_variables, req_aliases = _resolve_requested_variables_with_alias_map(
            self.catalog, variables, domain="image"
        )
        structural, var_specs, filter_aliases = self._split_measurement_filters(
            filters_norm, domain="image", table_name="image_measurements"
        )
        alias_map: dict[str, str] = dict(req_aliases)
        for canonical, requested in filter_aliases.items():
            if canonical not in alias_map:
                alias_map[canonical] = requested
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
        if cohort_eff is not False:
            df = self._filter_dataframe_by_cohort(df, str(cohort_eff))
        out = self._prepare_measurements(
            df,
            wide=wide,
            table_name="image_measurements",
            image_wide_single_variable=_image_wide_single_variable_from_request(variables),
        )
        if wide:
            out = self._rename_measurement_wide_columns(
                out, table_name="image_measurements", canonical_to_alias=alias_map
            )
        return out

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
            """Keep the row unless its modality has a catalog default pipeline that it doesn't match."""
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
        imread_kwargs: dict[str, Any] | None = None,
        cohort_id: str | bool | None = None,
    ) -> pd.DataFrame | Any:
        """
        Query the ``assets`` table; optionally pivot to one row per ``subject_uid`` (``wide=True``).

        With ``get_image=True``, resolve paths to :func:`nvitk.io.imageio.imread` jobs (NIfTI
        directories expand to multiple volumes; JSON sidecars are picked up when present).
        ``imread_kwargs`` is forwarded to ``imread``; ``value`` selects the path column to read.
        """
        cohort_eff, filters_clean = self._resolve_cohort(cohort_id, filters)
        merged = merge_filters(
            dict(filters_clean),
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
        if cohort_eff is not False:
            df = self._filter_dataframe_by_cohort(df, str(cohort_eff))

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
            return _imread_stack_from_jobs(jobs, **(imread_kwargs or {}))

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
        imread_kwargs: dict[str, Any] | None = None,
        cohort_id: str | bool | None = None,
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
            imread_kwargs=imread_kwargs,
            cohort_id=cohort_id,
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
        imread_kwargs: dict[str, Any] | None = None,
        cohort_id: str | bool | None = None,
    ) -> pd.DataFrame | Any:
        """
        Query the ``scans`` table; with ``get_image=True``, load volumes from ``path_column``.

        Missing local paths can trigger XNAT download when ``xnat_config`` or env/profile resolves;
        use ``download_scan_path`` for a persistent cache directory. ``asset_type`` must be
        ``dicom`` or ``nifti`` for on-demand download; ``imread_kwargs`` passes through to ``imread``.
        """
        cohort_eff, filters_clean = self._resolve_cohort(cohort_id, filters)
        merged = merge_filters(
            dict(filters_clean),
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
        if cohort_eff is not False:
            df = self._filter_dataframe_by_cohort(df, str(cohort_eff))

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
                jobs.extend(_imread_jobs_for_scan_path(raw, asset_type=ft))

            if need_download:
                cfg = xnat_config
                if cfg is None:
                    try:
                        cfg = resolve_xnat_connection(load_xnat_profile())
                        log.info(f"Connected to XNAT: {cfg.server} / {cfg.project}")
                    except ValueError as exc:
                        raise FilterError(
                            "get_image=True with missing local_cache_path requires XNAT credentials: "
                            "pass xnat_config=... or set XNAT_SERVER / XNAT_PROJECT (and auth), "
                            "or use an NVITK XNAT profile."
                        ) from exc

                from nvitk.io.imageio import imread

                from .xnat import (
                    connect_xnat,
                    download_scan_dicoms,
                    download_scan_niftis,
                    resolve_xnat_scan_from_scan_row,
                )

                if ft not in ("dicom", "nifti"):
                    raise FilterError(
                        "On-demand download from XNAT in scans(get_image=True) is implemented for "
                        "asset_type='dicom' or 'nifti' only; sync other assets or set local_cache_path."
                    )

                use_persistent_cache = download_scan_path is not None and str(download_scan_path).strip()
                persistent_base = Path(download_scan_path).expanduser().resolve() if use_persistent_cache else None
                if persistent_base is not None:
                    persistent_base.mkdir(parents=True, exist_ok=True)

                ephemeral_images: list[Any] = []
                read_kw = imread_kwargs or {}

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
                            if ft == "dicom":
                                if skip_existing_download and _dicom_cache_dir_has_files(dest):
                                    pass
                                else:
                                    download_scan_dicoms(scan_obj, dest)
                                jobs.extend(_imread_jobs_for_scan_path(dest, asset_type=ft))
                            else:
                                if skip_existing_download and _nifti_cache_dir_has_files(dest):
                                    pass
                                else:
                                    download_scan_niftis(scan_obj, dest)
                                nifti_paths = _nifti_files_in_dir(dest)
                                if not nifti_paths:
                                    raise FilterError(f"No NIfTI files available under {dest!r} after download.")
                                for npth in nifti_paths:
                                    jobs.append((npth, "nifti"))
                        else:
                            with tempfile.TemporaryDirectory(prefix="nvitk_xnat_scan_") as tmp:
                                dest = Path(tmp)
                                if ft == "dicom":
                                    download_scan_dicoms(scan_obj, dest)
                                    ephemeral_images.append(
                                        imread(
                                            str(_root_path_for_imread(dest, asset_type=ft)),
                                            force_type=ft,
                                            **read_kw,
                                        )
                                    )
                                else:
                                    download_scan_niftis(scan_obj, dest)
                                    nifti_paths = _nifti_files_in_dir(dest)
                                    if not nifti_paths:
                                        raise FilterError(
                                            f"No NIfTI files extracted to temporary directory {dest!r}."
                                        )
                                    for npth in nifti_paths:
                                        kw = _imread_kwargs_with_nifti_sidecar(npth, "nifti", read_kw)
                                        ephemeral_images.append(
                                            imread(str(npth), force_type="nifti", **kw)
                                        )

                if ephemeral_images:
                    disk_out: list[Any] = []
                    for p, fft in _dedupe_imread_jobs(jobs):
                        fft_l = str(fft).strip().lower()
                        kw = _imread_kwargs_with_nifti_sidecar(p, fft_l, read_kw)
                        disk_out.append(imread(str(p), force_type=fft_l, **kw))
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
            return _imread_stack_from_jobs(jobs, **(imread_kwargs or {}))

        return df

    def join(
        self,
        frames: Iterable[pd.DataFrame],
        *,
        on: str | list[str] = "subject_uid",
        how: str = "left",
        cohort_id: str | bool | None = None,
    ) -> pd.DataFrame:
        """
        Merge multiple DataFrames on *on* (default ``subject_uid``); optionally filter by cohort after join.
        """
        dataframes = [frame.copy() for frame in frames]
        if not dataframes:
            return pd.DataFrame()
        keys = [on] if isinstance(on, str) else list(on)
        result = dataframes[0]
        for frame in dataframes[1:]:
            result = result.merge(frame, on=keys, how=how)
        cohort_eff, _ = self._resolve_cohort(cohort_id, None)
        if cohort_eff is not False and "subject_uid" in result.columns:
            result = self._filter_dataframe_by_cohort(result, str(cohort_eff))
        return result

    def _enforce_measurement_columns(self, table: str, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict *df* to the columns allowed for a known measurement *table*; pass through unchanged
        for tables not listed in ``MEASUREMENT_TABLE_COLUMNS``."""
        allowed = MEASUREMENT_TABLE_COLUMNS.get(table)
        if allowed is None:
            return df
        return restrict_to_manifest_columns(df, allowed)

    def write_table(
        self,
        table: str,
        df: pd.DataFrame,
        *,
        provenance: dict[str, Any] | None = None,
        build_sqlite_index: bool = False,
    ) -> Path:
        """
        Write *df* to the table's Parquet file and refresh catalog schema metadata.

        Set ``build_sqlite_index=True`` to rebuild the SQLite index for *table* after the write.
        """
        definition = self.catalog.get_table(table)
        df = self._enforce_measurement_columns(table, df)
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
        """
        Append *df* to the existing table and drop duplicates on *key_columns* (default: manifest keys).

        Always reads the current Parquet (not SQLite) so sequential upserts without an
        intervening index rebuild do not drop recently written rows.

        Returns the combined frame after :meth:`write_table`.
        """
        definition = self.catalog.get_table(table)
        df = self._enforce_measurement_columns(table, df)
        # SQLite may lag Parquet when build_sqlite_index=False between upserts.
        existing = self.get(table, cohort_id=False, use_sqlite=False)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = self._enforce_measurement_columns(table, combined)
        keys = key_columns or list(definition.key_columns)
        if keys:
            present_keys = [column for column in keys if column in combined.columns]
            if present_keys:
                combined = combined.drop_duplicates(subset=present_keys, keep="last")
        self.write_table(table, combined, provenance=provenance, build_sqlite_index=build_sqlite_index)
        return combined

    def build_sqlite_index(self, *, tables: list[str] | None = None) -> Path:
        """Rebuild the SQLite index from Parquet (all tables or the subset *tables*)."""
        return self.sqlite.build(self.catalog, tables=tables)

    def register_variables(self, entries: list[dict[str, Any]]) -> None:
        """Register variable definitions in the catalog manifest (clinical/image domains)."""
        self.catalog.register_variables(entries)

    def drop_table(self, name: str, *, remove_sqlite: bool = True) -> None:
        """Clear *name* from the catalog; optionally delete the SQLite DB file if ``remove_sqlite``."""
        self.catalog.clear_table(name)
        if remove_sqlite and self.sqlite.exists():
            self.sqlite.db_path.unlink()

    def drop_all_tables(self, *, remove_sqlite: bool = True) -> None:
        """Clear every registered table; optionally remove the SQLite index file once at the end."""
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
        """Read *definition*'s Parquet file (an empty typed frame if it doesn't exist yet), coerce to the
        manifest schema, and apply *filters*."""
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
        """Resolve ``value_num``/``value_text`` into a unified ``value`` column and, if ``wide``, pivot to
        one column per variable via :meth:`_to_wide`; otherwise return the long frame with a fresh index."""
        df = self._resolve_measurement_values(df)
        if wide:
            definition = self.catalog.get_table(table_name)
            kw: dict[str, Any] = {}
            if table_name == "image_measurements":
                kw["image_wide_single_variable"] = (
                    False if image_wide_single_variable is None else image_wide_single_variable
                )
            out = self._to_wide(df, definition, **kw)
            return self._coerce_wide_measurement_dtypes(out, table_name=table_name)
        return df.reset_index(drop=True)

    def _coerce_wide_measurement_dtypes(self, df: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
        """
        Restore numeric dtypes on wide-pivoted measurement columns.

        ``_resolve_measurement_values`` builds a single ``value`` series from ``value_num`` with a
        ``value_text`` fallback, which upcasts the whole series to ``object`` whenever *any* text
        variable is in the query. The subsequent pivot then leaves numeric variables (e.g.
        ``hematocrit``) as object-dtype floats, and Patsy treats them as categoricals
        (``Hematocrit[T.36.0]``, …). Coerce columns whose catalog ``value_kind`` is numeric back to
        float64.
        """
        if df.empty:
            return df
        numeric_kinds = {"numeric", "float", "int", "integer", "number"}
        domain = {
            "clinical_measurements": "clinical",
            "cognitive_measurements": "cognitive",
            "image_measurements": "image",
        }.get(table_name)
        try:
            entries = self.catalog.variable_entries(domain=domain) if domain else self.catalog.variable_entries()
        except Exception:
            entries = []
        by_id = {str(e.get("variable_id")): e for e in entries if e.get("variable_id")}

        out = df.copy()
        for column in out.columns:
            name = str(column)
            entry = by_id.get(name)
            if entry is None and table_name == "image_measurements":
                # image wide keys are ``{region}_{variable}`` / similar — match by suffix variable_id
                for vid, ent in by_id.items():
                    if name == vid or name.endswith(f"_{vid}"):
                        entry = ent
                        break
            if entry is None:
                continue
            kind = str(entry.get("value_kind") or "").strip().lower()
            if kind not in numeric_kinds:
                continue
            if pd.api.types.is_numeric_dtype(out[column]):
                continue
            out[column] = pd.to_numeric(out[column], errors="coerce")
        return out

    def _resolve_measurement_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a unified ``value`` column, preferring ``value_num`` and falling back to ``value_text``
        row-wise where the numeric value is missing."""
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
        """Pivot a long measurement frame to wide form: one row per entity (``wide_index_columns``) and
        one column per composed key (region/variable/frame, see :func:`_compose_image_wide_keys` for
        ``image_measurements`` or :meth:`_compose_wide_keys` otherwise), values from the ``value`` column."""
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
        """Join *key_columns* per row with ``__`` (skipping empty parts) to form wide-pivot column keys."""
        return (
            df[key_columns]
            .astype("string")
            .fillna("")
            .agg(lambda row: "__".join(item for item in row if item), axis=1)
            .astype("string")
        )

    def _relative_path(self, path: Path) -> str:
        """*path* expressed relative to the dataset root, as a POSIX-style string for catalog storage."""
        return str(path.relative_to(self.root).as_posix())
