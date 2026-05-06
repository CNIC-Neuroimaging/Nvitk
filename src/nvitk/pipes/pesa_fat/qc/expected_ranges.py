"""Expected value ranges for QC table highlighting (placeholders until populated).

Keys match spreadsheet column names from stage-3 ``column_order()`` (excluding ``pesa_id``).
Values are ``(min_expected, max_expected)``. Use ``float('nan')`` for either bound to mean
\"no limit\" on that side; if either bound is non-finite, the cell is not highlighted.
"""

from __future__ import annotations

import math

from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg

_NAN = float("nan")


def _ctpet_measurement_columns() -> list[str]:
    cols: list[str] = []
    for spec in ct_cfg.SUV_SPECS:
        for suffix, _ in ct_cfg.SUV_STATS:
            cols.append(f"{spec.column_prefix}_{suffix}")
    for spec in ct_cfg.VOL_SPECS:
        cols.append(spec.column)
        cols.append(spec.column.replace("_VOL", "_NSlices"))
    return cols


def _dixon_measurement_columns() -> list[str]:
    cols: list[str] = []
    for spec in dx_cfg.MEASURE_SPECS:
        for metric in spec.metrics:
            cols.append(f"{spec.prefix}_{metric}")
    return cols


def _placeholder_map(columns: list[str]) -> dict[str, tuple[float, float]]:
    return {c: (_NAN, _NAN) for c in columns}


EXPECTED_RANGES_CTPET: dict[str, tuple[float, float]] = _placeholder_map(
    _ctpet_measurement_columns()
)
EXPECTED_RANGES_DIXON: dict[str, tuple[float, float]] = _placeholder_map(
    _dixon_measurement_columns()
)


def cell_out_of_range(
    column: str,
    value: float | int | None,
    ranges: dict[str, tuple[float, float]],
) -> bool:
    """Return True if *value* is outside ``ranges[column]`` when both bounds are finite."""
    if column == "pesa_id" or column not in ranges:
        return False
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    lo, hi = ranges[column]
    if not math.isfinite(lo) or not math.isfinite(hi):
        return False
    return v < lo or v > hi


__all__ = [
    "EXPECTED_RANGES_CTPET",
    "EXPECTED_RANGES_DIXON",
    "cell_out_of_range",
]
