"""
Filesystem helpers for the dataset layer: JSON, Parquet, dtype coercion, variable id normalization.

Shared by the catalog, SQLite index, and import pipelines.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# JSON & time
# ──────────────────────────────────────────────────────────────────────────────


def utc_now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string (no microseconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object from *path*."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write *payload* as indented JSON; creates parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, ensure_ascii=True)
        handle.write("\n")


# ──────────────────────────────────────────────────────────────────────────────
# Manifest dtypes & DataFrames
# ──────────────────────────────────────────────────────────────────────────────


def infer_manifest_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """Map column names to dtype strings as stored in table manifests."""
    return {column: str(dtype) for column, dtype in df.dtypes.items()}


def manifest_dtype_to_pandas(dtype_name: str) -> str:
    """Map manifest dtype strings to pandas dtypes used when coercing frames."""
    normalized = dtype_name.lower()
    if normalized in {"string", "string[python]"}:
        return "string"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized.startswith("int"):
        return "Int64"
    if normalized.startswith("float"):
        return "float64"
    if normalized.startswith("datetime"):
        return "datetime64[ns]"
    return "object"


def empty_dataframe(columns: dict[str, str]) -> pd.DataFrame:
    """Build an empty DataFrame with columns typed per manifest *columns*."""
    data: dict[str, pd.Series] = {}
    for column, dtype_name in columns.items():
        dtype = manifest_dtype_to_pandas(dtype_name)
        if dtype == "datetime64[ns]":
            data[column] = pd.Series([], dtype="datetime64[ns]")
        else:
            data[column] = pd.Series([], dtype=dtype)
    return pd.DataFrame(data)


def coerce_dataframe_to_manifest(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Align *df* to manifest columns: ``pipeline_id`` vs ``pipeline_version``, then cast dtypes."""
    if df.empty:
        out = df.copy()
        if "pipeline_id" in columns and "pipeline_id" not in out.columns:
            if "pipeline_version" in out.columns:
                out["pipeline_id"] = out["pipeline_version"]
            else:
                out["pipeline_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
        if "pipeline_id" in columns and "pipeline_version" in out.columns:
            out = out.drop(columns=["pipeline_version"])
        return out

    out = df.copy()
    # Canonical column is pipeline_id; older Parquet may use pipeline_version or omit both.
    if "pipeline_id" in columns and "pipeline_id" not in out.columns:
        if "pipeline_version" in out.columns:
            out["pipeline_id"] = out["pipeline_version"]
        else:
            out["pipeline_id"] = pd.Series(pd.NA, index=out.index, dtype="string")

    if "pipeline_id" in columns and "pipeline_version" in out.columns:
        out = out.drop(columns=["pipeline_version"])

    for column, dtype_name in columns.items():
        if column not in out.columns:
            continue
        dtype = manifest_dtype_to_pandas(dtype_name)
        if dtype == "string":
            out[column] = out[column].astype("string")
        elif dtype == "boolean":
            out[column] = out[column].astype("boolean")
        elif dtype == "Int64":
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
        elif dtype == "float64":
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
        elif dtype == "datetime64[ns]":
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Parquet
# ──────────────────────────────────────────────────────────────────────────────


def read_parquet_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read Parquet with pyarrow; optional column projection."""
    return pd.read_parquet(Path(path), columns=columns, engine="pyarrow")


def write_parquet_table(path: str | Path, df: pd.DataFrame) -> None:
    """Write *df* to Parquet (no index); creates parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False, engine="pyarrow")


# ──────────────────────────────────────────────────────────────────────────────
# Values & identifiers
# ──────────────────────────────────────────────────────────────────────────────


def coerce_bool(value: Any) -> bool:
    """Parse manifest/catalog boolean flags from bool, int, or common string tokens."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_variable_id(value: str) -> str:
    """Lowercase snake_case token for matching column names, export names, and UI labels."""
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def normalize_string(value: Any) -> str | None:
    """Strip strings; map NaN/empty to None."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def ensure_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast existing *columns* to pandas ``string`` dtype."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype("string")
    return out


def json_dumps(value: Any) -> str:
    """Stable JSON for hashing or sidecars (sorted keys, ASCII)."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)
