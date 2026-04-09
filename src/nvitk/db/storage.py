from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, ensure_ascii=True)
        handle.write("\n")


def infer_manifest_dtypes(df: pd.DataFrame) -> dict[str, str]:
    return {column: str(dtype) for column, dtype in df.dtypes.items()}


def manifest_dtype_to_pandas(dtype_name: str) -> str:
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
    data: dict[str, pd.Series] = {}
    for column, dtype_name in columns.items():
        dtype = manifest_dtype_to_pandas(dtype_name)
        if dtype == "datetime64[ns]":
            data[column] = pd.Series([], dtype="datetime64[ns]")
        else:
            data[column] = pd.Series([], dtype=dtype)
    return pd.DataFrame(data)


def coerce_dataframe_to_manifest(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
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


def read_parquet_table(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(path), columns=columns, engine="pyarrow")


def write_parquet_table(path: str | Path, df: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False, engine="pyarrow")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def ensure_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype("string")
    return out


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)
