"""Expected value thresholds for QC table highlighting.

Keys match spreadsheet column names from stage-3 ``column_order()`` (excluding ``pesa_id``).
Values are ``(floor, warn_high, bad_high)``:

- red if value < floor (when floor finite)
- orange if value > warn_high (when warn_high finite) AND value <= bad_high (when bad_high finite)
- red if value > bad_high (when bad_high finite)

Use ``None`` or ``nan`` to disable any threshold.
"""

from __future__ import annotations

import math

from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg

_NAN = float("nan")
Threshold = tuple[float | None, float | None, float | None]


def _ctpet_measurement_columns() -> list[str]:
    """All CT-PET stage-3 measurement column names (SUV stats per ROI, plus volume/slice-count
    columns), for building the expected-range placeholder map."""
    cols: list[str] = []
    for spec in ct_cfg.SUV_SPECS:
        for suffix, _ in ct_cfg.SUV_STATS:
            cols.append(f"{spec.column_prefix}_{suffix}")
    for spec in ct_cfg.VOL_SPECS:
        cols.append(spec.column)
        cols.append(spec.column.replace("_VOL", "_NSlices"))
    return cols


def _dixon_measurement_columns() -> list[str]:
    """All Dixon stage-3 measurement column names (metric per ROI spec), for building the expected-
    range placeholder map."""
    cols: list[str] = []
    for spec in dx_cfg.MEASURE_SPECS:
        for metric in spec.metrics:
            cols.append(f"{spec.prefix}_{metric}")
    return cols


def _placeholder_map(columns: list[str]) -> dict[str, Threshold]:
    """Build a ``{column: (None, None, None)}`` map (disabled thresholds) for every column."""
    return {c: (None, None, None) for c in columns}


EXPECTED_RANGES_CTPET: dict[str, Threshold] = _placeholder_map(_ctpet_measurement_columns())
EXPECTED_RANGES_DIXON: dict[str, Threshold] = _placeholder_map(_dixon_measurement_columns())

# ---------------------------------------------------------------------------
# Concrete thresholds requested (v2 incremental)
# ---------------------------------------------------------------------------

# Fat SUV thresholds (all SUV stats)
_FAT_ROIS = ("GRASA_V", "GRASA_SC", "GRASA_V_BATCH", "GRASA_SC_BATCH")
_SUV_SUFFIXES = tuple(suf for suf, _ in ct_cfg.SUV_STATS)  # SUVMAX, SUVmean, SUV95p, SUV99p
for roi in _FAT_ROIS:
    for suf in _SUV_SUFFIXES:
        EXPECTED_RANGES_CTPET[f"{roi}_{suf}"] = (0.0, 5.0, 6.0)

# Dixon FF thresholds (exclude bone narrow BN_*)
for col in list(EXPECTED_RANGES_DIXON.keys()):
    if not str(col).endswith("_FF"):
        continue
    if str(col).startswith("DIXON_BN_"):
        continue
    EXPECTED_RANGES_DIXON[col] = (0.0, 50.0, 100.0)


def cell_level(
    column: str,
    value: float | int | None,
    ranges: dict[str, Threshold],
) -> str | None:
    """Return 'warn' | 'bad' | None for the given cell."""
    if column == "pesa_id" or column not in ranges:
        return None
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None

    floor, warn_high, bad_high = ranges[column]

    def _finite(x: float | None) -> bool:
        """True if *x* is a set, finite threshold value."""
        return x is not None and math.isfinite(float(x))

    if _finite(floor) and v < float(floor):
        return "bad"
    if _finite(bad_high) and v > float(bad_high):
        return "bad"
    if _finite(warn_high) and v > float(warn_high):
        # warn zone only makes sense when we also have a bad_high, but keep warn regardless
        return "warn"
    return None


__all__ = [
    "EXPECTED_RANGES_CTPET",
    "EXPECTED_RANGES_DIXON",
    "cell_level",
]
