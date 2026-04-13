from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from .exceptions import FilterError

_SEQUENCE_TYPES = (list, tuple, set, frozenset, pd.Index)
_SUPPORTED_OPS = {
    "eq",
    "ne",
    "in",
    "not_in",
    "gt",
    "ge",
    "lt",
    "le",
    "contains",
    "is_null",
    "not_null",
}


@dataclass(frozen=True)
class FilterCondition:
    column: str
    op: str
    value: Any = None


def _is_multi_value(value: Any) -> bool:
    return isinstance(value, _SEQUENCE_TYPES)


def _normalize_op(op: str) -> str:
    normalized = op.strip().lower().lstrip("$")
    aliases = {
        "gte": "ge",
        '>=': 'ge',
        ">": "gt",
        "lte": "le",
        "<=": "le",
        "<": "lt",
        "isin": "in",
        "==": "eq",
        "!=": "ne",
        "nin": "not_in",
        "notin": "not_in",
        "null": "is_null",
        "isnull": "is_null",
        'Null': "is_null",
        'NULL': "is_null",
        "notnull": "not_null",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _SUPPORTED_OPS:
        raise FilterError(f"Unsupported filter operator: {op}")
    return normalized


def normalize_filters(filters: Mapping[str, Any] | None) -> dict[str, list[FilterCondition]]:
    if not filters:
        return {}

    normalized: dict[str, list[FilterCondition]] = {}
    for column, spec in filters.items():
        if not isinstance(column, str) or not column.strip():
            raise FilterError("Filter columns must be non-empty strings.")

        column_name = column.strip()
        conditions: list[FilterCondition] = []

        if isinstance(spec, FilterCondition):
            conditions = [FilterCondition(column_name, _normalize_op(spec.op), spec.value)]
        elif isinstance(spec, Mapping):
            for raw_op, value in spec.items():
                op = _normalize_op(str(raw_op))
                conditions.append(FilterCondition(column_name, op, value))
        elif _is_multi_value(spec):
            conditions = [FilterCondition(column_name, "in", list(spec))]
        elif spec is None:
            conditions = [FilterCondition(column_name, "is_null", None)]
        else:
            conditions = [FilterCondition(column_name, "eq", spec)]

        normalized[column_name] = conditions
    return normalized


def _mask_for_condition(series: pd.Series, condition: FilterCondition) -> pd.Series:
    op = condition.op
    value = condition.value

    if op == "eq":
        if value is None:
            return series.isna()
        return series == value
    if op == "ne":
        if value is None:
            return series.notna()
        return series != value
    if op == "in":
        values = list(value or [])
        if not values:
            return pd.Series(False, index=series.index)
        return series.isin(values)
    if op == "not_in":
        values = list(value or [])
        if not values:
            return pd.Series(True, index=series.index)
        return ~series.isin(values)
    if op == "gt":
        return series > value
    if op == "ge":
        return series >= value
    if op == "lt":
        return series < value
    if op == "le":
        return series <= value
    if op == "contains":
        return series.astype("string").str.contains(str(value), case=False, na=False)
    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return series.notna()
    raise FilterError(f"Unsupported filter operator: {op}")


def apply_filters(df: pd.DataFrame, filters: Mapping[str, Any] | None) -> pd.DataFrame:
    if df.empty or not filters:
        return df.copy()

    normalized = normalize_filters(filters)
    mask = pd.Series(True, index=df.index)

    for column, conditions in normalized.items():
        resolved_column = column
        if column not in df.columns:
            if column == "pipeline_id" and "pipeline_version" in df.columns:
                resolved_column = "pipeline_version"
            else:
                raise FilterError(f"Column '{column}' is not available in the selected table.")
        series = df[resolved_column]
        column_mask = pd.Series(True, index=df.index)
        for condition in conditions:
            column_mask &= _mask_for_condition(series, condition)
        mask &= column_mask

    return df.loc[mask].copy()


def escape_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def build_sql_where(filters: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
    normalized = normalize_filters(filters)
    if not normalized:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []

    for column, conditions in normalized.items():
        escaped = escape_identifier(column)
        for condition in conditions:
            op = condition.op
            value = condition.value

            if op == "eq":
                clauses.append(f"{escaped} = ?")
                params.append(value)
            elif op == "ne":
                clauses.append(f"{escaped} != ?")
                params.append(value)
            elif op == "in":
                values = list(value or [])
                if not values:
                    clauses.append("1 = 0")
                else:
                    placeholders = ", ".join("?" for _ in values)
                    clauses.append(f"{escaped} IN ({placeholders})")
                    params.extend(values)
            elif op == "not_in":
                values = list(value or [])
                if values:
                    placeholders = ", ".join("?" for _ in values)
                    clauses.append(f"{escaped} NOT IN ({placeholders})")
                    params.extend(values)
            elif op == "gt":
                clauses.append(f"{escaped} > ?")
                params.append(value)
            elif op == "ge":
                clauses.append(f"{escaped} >= ?")
                params.append(value)
            elif op == "lt":
                clauses.append(f"{escaped} < ?")
                params.append(value)
            elif op == "le":
                clauses.append(f"{escaped} <= ?")
                params.append(value)
            elif op == "contains":
                clauses.append(f"LOWER(CAST({escaped} AS TEXT)) LIKE ?")
                params.append(f"%{str(value).lower()}%")
            elif op == "is_null":
                clauses.append(f"{escaped} IS NULL")
            elif op == "not_null":
                clauses.append(f"{escaped} IS NOT NULL")
            else:
                raise FilterError(f"Unsupported filter operator for SQLite queries: {op}")

    return " AND ".join(clauses), params


def merge_filters(*filters: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in filters:
        if not item:
            continue
        merged.update(item)
    return merged


def ensure_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]
