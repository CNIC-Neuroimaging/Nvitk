#!/usr/bin/env python3
"""Import FreeSurfer cortical and subcortical region metrics into ``image_measurements``.

Description
-----------
Both T1 workbooks are **long**: one row per ``(subject, session, side, measurement, region)``::

    ID           side  measurement  region      value    subject      session

so ``measurement`` names the variable, ``region`` (with ``side``) names the region, and ``value``
is the number. The earlier importer read these sheets as *wide* — one region per column — which
turned the column headers into region ids and kept one arbitrary row per subject. This reads the
layout it actually has.

What each measurement becomes
-----------------------------
One variable per measurement rather than one lumped "volume", because they are different
quantities in different units: a cortical thickness in mm and a grey-matter volume in mm³ cannot
share a variable id and still be interpretable.

``Index`` and ``SegId`` are skipped. They are FreeSurfer's internal row number and segmentation
label id — bookkeeping that happens to be numeric, not measurements of anything.

Region ids
----------
Cortical parcels are lateralized, so ``side`` is folded into the id: ``left_precuneus``,
``right_precuneus``. Whole-brain scalars (``BrainSegVol``, ``eTIV``, ``MeanThickness``) have no
side and keep their bare name. ``region_label`` preserves the original spelling.

Note on eTIV
------------
``eTIV`` arrives here as ``t1_volume_mm3`` at region ``etiv``. ``import_t1_etiv.py`` additionally
publishes it as a dedicated ``t1_etiv_volume`` scalar — the same number under a name that is easier
to reach for as a normalization denominator. Running both is intentional; running neither leaves
you without head-size correction.

Examples::

    # Report what would be written, per variable
    python scripts/database/import_t1_regions.py

    # Write both workbooks
    python scripts/database/import_t1_regions.py --write

    # Only cortical thickness and grey-matter volume
    python scripts/database/import_t1_regions.py --measurements ThickAvg,GrayVol --write
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import re
from pathlib import Path
from typing import Any, Sequence

import click
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.derived_measurements import (
    DerivedImageMeasurementSpec,
    DerivedVariableRegistration,
    build_image_measurement_rows,
    publish_derived_measurements,
)
from nvitk.db.importers import read_tabular_source

log = Logger()

_RAW = Path(
    "/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/"
    "PESA-Brain/DB/raw"
)
DEFAULT_CORTICAL = _RAW / "PESABrain_T1_cortical_regs_20250605.xlsx"
DEFAULT_SUBCORTICAL = _RAW / "PESABrain_T1_subcortical_regs_20250605.xlsx"

T1_PIPELINE_ID = "t1_volumetry_v1"

#: ``source measurement -> (variable_id, unit, label)``. Measurements that mean the same thing map
#: to the same variable even when the workbooks spell them differently: ``Thickness_mm``
#: (MeanThickness) is an average cortical thickness like ``ThickAvg``, and ``SurfArea_mm2``
#: (WhiteSurfArea) is a surface area like ``SurfArea``, so each pair shares a variable and is told
#: apart by its region.
MEASUREMENT_MAP: dict[str, tuple[str, str | None, str]] = {
    # ---- cortical, per parcel -------------------------------------------------
    "GrayVol": ("t1_gray_volume", "mm3", "Cortical grey-matter volume"),
    "ThickAvg": ("t1_thickness_avg", "mm", "Cortical thickness (mean)"),
    "ThickStd": ("t1_thickness_std", "mm", "Cortical thickness (SD within parcel)"),
    "SurfArea": ("t1_surface_area", "mm2", "Cortical surface area"),
    "NumVert": ("t1_num_vertices", "count", "Surface vertex count"),
    "MeanCurv": ("t1_mean_curvature", "1/mm", "Integrated rectified mean curvature"),
    "GausCurv": ("t1_gaussian_curvature", "1/mm2", "Integrated rectified Gaussian curvature"),
    "FoldInd": ("t1_folding_index", None, "Folding index"),
    "CurvInd": ("t1_curvature_index", None, "Intrinsic curvature index"),
    # ---- whole-brain scalars --------------------------------------------------
    "Volume_mm3": ("t1_volume_mm3", "mm3", "Segmented volume"),
    "Thickness_mm": ("t1_thickness_avg", "mm", "Cortical thickness (mean)"),
    "SurfArea_mm2": ("t1_surface_area", "mm2", "Cortical surface area"),
    # ---- subcortical, per structure -------------------------------------------
    "NVoxels": ("t1_num_voxels", "count", "Segmented voxel count"),
    "normMean": ("t1_intensity_mean", None, "Normalized intensity (mean)"),
    "normStdDev": ("t1_intensity_std", None, "Normalized intensity (SD)"),
    "normMin": ("t1_intensity_min", None, "Normalized intensity (min)"),
    "normMax": ("t1_intensity_max", None, "Normalized intensity (max)"),
    "normRange": ("t1_intensity_range", None, "Normalized intensity (range)"),
    # Ratios and QC counters (BrainSegVol-to-eTIV, SurfaceHoles, …).
    "unitless": ("t1_index_unitless", None, "Dimensionless FreeSurfer index"),
}

#: FreeSurfer bookkeeping: a row number and a segmentation label id. Numeric, but not measurements.
SKIP_MEASUREMENTS: frozenset[str] = frozenset({"Index", "SegId"})

#: The catalog's default key for ``image_measurements`` does **not** include ``session_id``, so two
#: T1 sessions of the same subject collide and the earlier one is dropped. Four subjects here have a
#: repeat scan; keying on the session keeps both, which is the point of storing a session at all.
UPSERT_KEY: list[str] = [
    "subject_uid", "session_id", "modality", "region_id", "frame_index",
    "variable_id", "pipeline_id", "source_file", "source_sheet", "source_column",
]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _region_token(side: Any, region: Any) -> str:
    """``('left', 'precuneus') -> 'left_precuneus'``; unsided regions keep their bare name."""
    name = re.sub(r"[^0-9A-Za-z]+", "_", str(region or "").strip()).strip("_").lower()
    if not name:
        return ""
    hemisphere = str(side or "").strip().lower()
    if hemisphere in {"left", "lh", "l"}:
        return f"left_{name}"
    if hemisphere in {"right", "rh", "r"}:
        return f"right_{name}"
    return name


def read_t1_long(path: Path, *, sheet: str = "Sheet1") -> pd.DataFrame:
    """
    Normalize one T1 workbook to ``subject_uid`` / ``session_id`` / ``measurement`` / ``region_id``
    / ``region_label`` / ``value_num``.

    Raises
    ------
    ValueError
        When the sheet lacks the long-format columns — importing it as anything else is what
        produced column-header region ids last time, so the shape is checked rather than assumed.
    """
    raw = read_tabular_source(path, sheet_name=sheet)
    lower = {str(c).strip().lower(): c for c in raw.columns}
    required = ("measurement", "region", "value", "subject")
    missing = [c for c in required if c not in lower]
    if missing:
        raise ValueError(
            f"{path.name} is not in the long T1 layout — missing {', '.join(missing)}. "
            f"Columns present: {', '.join(str(c) for c in raw.columns)}."
        )

    side = raw[lower["side"]] if "side" in lower else pd.Series("", index=raw.index)
    frame = pd.DataFrame({
        "subject_uid": raw[lower["subject"]].astype("string"),
        "session_id": (
            raw[lower["session"]].astype("string") if "session" in lower
            else pd.Series(pd.NA, index=raw.index, dtype="string")
        ),
        "measurement": raw[lower["measurement"]].astype(str).str.strip(),
        "region_label": raw[lower["region"]].astype("string"),
        "region_id": [_region_token(s, r) for s, r in zip(side, raw[lower["region"]])],
        "value_num": pd.to_numeric(raw[lower["value"]], errors="coerce"),
    })

    before = len(frame)
    frame = frame.dropna(subset=["subject_uid", "value_num"])
    frame = frame[frame["region_id"].astype(bool)]
    log.info(
        "%s: %d row(s) → %d usable; %d subject(s), %d measurement(s), %d region(s).",
        path.name, before, len(frame), frame["subject_uid"].nunique(),
        frame["measurement"].nunique(), frame["region_id"].nunique(),
    )
    return frame.reset_index(drop=True)


def plan(frame: pd.DataFrame, *, only: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-measurement summary of what would be imported, and under which variable."""
    rows: list[dict[str, Any]] = []
    for measurement, block in frame.groupby("measurement", sort=True):
        if measurement in SKIP_MEASUREMENTS:
            rows.append({"measurement": measurement, "variable_id": "(skipped)",
                         "unit": "", "rows": len(block), "regions": block["region_id"].nunique()})
            continue
        if only and measurement not in set(only):
            continue
        mapped = MEASUREMENT_MAP.get(measurement)
        if mapped is None:
            rows.append({"measurement": measurement, "variable_id": "(unmapped)",
                         "unit": "", "rows": len(block), "regions": block["region_id"].nunique()})
            continue
        variable, unit, _label = mapped
        rows.append({"measurement": measurement, "variable_id": variable, "unit": unit or "",
                     "rows": len(block), "regions": block["region_id"].nunique()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
def build_rows(
    frame: pd.DataFrame, *, path: Path, sheet: str, only: Sequence[str] | None = None
) -> tuple[pd.DataFrame, list[DerivedVariableRegistration]]:
    """``image_measurements`` rows for every mapped measurement, plus their catalog entries."""
    blocks: list[pd.DataFrame] = []
    registrations: dict[str, DerivedVariableRegistration] = {}
    wanted = set(only) if only else None

    for measurement, block in frame.groupby("measurement", sort=True):
        if measurement in SKIP_MEASUREMENTS:
            continue
        if wanted is not None and measurement not in wanted:
            continue
        mapped = MEASUREMENT_MAP.get(measurement)
        if mapped is None:
            log.warning(
                "Measurement %r is not in MEASUREMENT_MAP and was skipped (%d row(s)). Add it "
                "there to import it.", measurement, len(block),
            )
            continue
        variable, unit, label = mapped

        rows = build_image_measurement_rows(
            block.loc[:, ["subject_uid", "session_id", "region_id", "region_label", "value_num"]],
            DerivedImageMeasurementSpec(
                variable_id=variable,
                modality="t1",
                pipeline_id=T1_PIPELINE_ID,
                pipeline_name=path.stem,
                source_file=path.name,
                source_sheet=sheet,
                # The source column is always 'value'; the measurement name is what distinguishes
                # the variables, and it belongs in the upsert key so two measurements sharing a
                # variable id (Volume_mm3 vs Thickness_mm) cannot overwrite each other.
                source_column=str(measurement),
                unit=unit,
                source_batch_id="import_t1_regions",
            ),
        )
        blocks.append(rows)
        registrations.setdefault(
            variable,
            DerivedVariableRegistration(
                variable_id=variable,
                domain="image",
                table="image_measurements",
                modality="t1",
                label=label,
                unit=unit,
                value_kind="float",
                source_file=path.name,
                source_sheet=sheet,
                source_column=str(measurement),
            ),
        )

    if not blocks:
        return pd.DataFrame(), []
    return pd.concat(blocks, ignore_index=True), list(registrations.values())


def publish(
    repo: Any,
    rows: pd.DataFrame,
    registrations: Sequence[DerivedVariableRegistration],
    *,
    path: Path,
    build_index: bool,
) -> None:
    """Upsert one workbook's rows in a single pass, registering each variable afterwards."""
    if rows.empty:
        return
    publish_derived_measurements(
        repo,
        rows,
        table="image_measurements",
        provenance={"importer": "import_t1_regions", "source_file": path.name},
        build_sqlite_index=build_index,
        upsert_key_columns=UPSERT_KEY,
    )
    for registration in registrations:
        repo.register_variables([registration.to_catalog_entry()])
    log.ok("%s: wrote %d row(s) over %d variable(s).", path.name, len(rows), len(registrations))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command("import-t1-regions")
@click.option(
    "--cortical",
    type=click.Path(path_type=Path),
    default=DEFAULT_CORTICAL,
    show_default=False,
    help="Cortical workbook. Pass '' to skip it.",
)
@click.option(
    "--subcortical",
    type=click.Path(path_type=Path),
    default=DEFAULT_SUBCORTICAL,
    show_default=False,
    help="Subcortical workbook. Pass '' to skip it.",
)
@click.option("--sheet", default="Sheet1", show_default=True, help="Worksheet to read.")
@click.option(
    "--measurements",
    default=None,
    help="Comma-separated source measurements to import (e.g. 'ThickAvg,GrayVol'). Omit for all.",
)
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the one configured in .nvitk/settings.json.",
)
@click.option(
    "--write/--dry-run",
    default=False,
    show_default="--dry-run",
    help="Actually write to the dataset. The default only reports what would be written.",
)
def main(
    cortical: Path | None,
    subcortical: Path | None,
    sheet: str,
    measurements: str | None,
    dataset: Path | None,
    write: bool,
) -> None:
    """Import the FreeSurfer cortical/subcortical long-format exports."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    only = [m.strip() for m in (measurements or "").split(",") if m.strip()] or None
    sources = [
        (Path(p), label)
        for p, label in ((cortical, "cortical"), (subcortical, "subcortical"))
        if p and str(p)
    ]
    if not sources:
        raise click.ClickException("Nothing to import — both workbooks were skipped.")

    try:
        repo = _open_repo(dataset)
    except Exception as exc:
        raise click.ClickException(f"Could not open the dataset ({exc}).") from exc

    total = 0
    for i, (path, label) in enumerate(sources):
        if not path.is_file():
            raise click.ClickException(f"{label} workbook not found: {path}")
        frame = read_t1_long(path, sheet=sheet)

        click.echo(f"\n{label} — {path.name}")
        click.echo(plan(frame, only=only).to_string(index=False))

        rows, registrations = build_rows(frame, path=path, sheet=sheet, only=only)
        total += len(rows)
        if write:
            # Index once, after the last workbook: it is rebuilt from Parquet either way.
            publish(repo, rows, registrations, path=path, build_index=(i == len(sources) - 1))

    click.echo("")
    if write:
        click.echo(f"Wrote {total} row(s) to image_measurements.")
    else:
        click.echo(f"Dry run — {total} row(s) would be written. Re-run with --write to apply.")


if __name__ == "__main__":
    main()
