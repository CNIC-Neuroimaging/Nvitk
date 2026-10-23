"""TOF morphometrics results in the measurement results window.

A morphometrics run (the GUI *Morphometrics* tool, or a stage-7 folder loaded for viewing) is
shown as five results, like every other measurement:

``Morphometrics — vessels``
    one row per vessel with the stage-7 report's columns (``morphometrics_report.csv``): the
    selected features and the high-confidence lesion flags, the same table the pipeline writes and
    the database imports. Built from the run's tables when the file is missing (older runs).
``Morphometrics — all vessel measures``
    one row per vessel: every column of ``vessel_summary.csv`` (calibre, end-window diameters,
    bifurcation blanking, tortuosity of the first / last N mm, curvature integrals, direction
    angles, branching, lesion counts and the worst lesion's severity / confidence…), then the
    measures computed before the compact tables existed — the cow_morpho path metrics of
    ``00_Path_Summary`` aggregated per vessel, the label volumetry and the literature tortuosity
    workbook. A name that already means something else in ``vessel_summary.csv`` gets the suffix
    ``_legacy`` (``tortuosity_dm_legacy`` is cow's length-weighted mean over segments; the
    unsuffixed one is the main path's). Both vessel tables are label rows: *Colour labels by*
    colours the measured labels **and** the vessel surfaces by any column.
``Morphometrics — lesions``
    one row per stenosis / enlargement: location, length, severity (taper %, diameter and area %
    against the proximal reference) and confidence with its flags. Clicking a row shows that
    lesion in the viewer.
``Morphometrics — segments``
    one row per non-overlapping segment.
``Morphometrics — subject``
    circle-of-Willis variants, completeness, left/right asymmetry and totals.

Older outputs without the compact tables still give the vessel table (from ``00_Path_Summary``).
"""

from __future__ import annotations

import re
import weakref
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# The per-vessel merge is shared with the pipeline's report (Qt-free); re-exported here.
from nvitk.measure.morpho.export_utils.report import (  # noqa: F401
    LEGACY_PATH_COLUMNS,
    REPORT_CSV,
    TORTUOSITY_COLUMNS,
    VOLUMETRY_COLUMNS,
    _read_csv,
    morphometrics_report,
    vessel_table,
)

#: Columns that name a row rather than measure it.
_IDENTITY = ("case_id", "label", "vessel_name", "full_name", "component_id", "side", "pair", "territory")

VESSELS_TITLE = "Morphometrics — vessels"
#: Every per-vessel measure (the vessels table shows the report's columns only).
VESSELS_ALL_TITLE = "Morphometrics — all vessel measures"
LESIONS_TITLE = "Morphometrics — lesions"
SEGMENTS_TITLE = "Morphometrics — segments"
SUBJECT_TITLE = "Morphometrics — subject"
#: Field of the surface layer that holds the colouring by a measurement.
SURFACE_VALUE_FIELD = "measurement (by label)"


def _is_count(column: str) -> bool:
    return (column.startswith(("n_", "total_n_")) or column.endswith(("_n", "_count", "_present", "n_points"))
            or column in ("bend_count", "n_voxels", "n_paths", "n_components"))


def _cell(column: str, value: Any) -> Any:
    """A payload cell: bools as such, counts as ints, missing numbers blank."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        if not np.isfinite(v):
            return ""
        return int(round(v)) if _is_count(column) and v.is_integer() else v
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _unique(key: str, used: set[str]) -> str:
    out, k = key, 2
    while out in used:
        out, k = f"{key} ({k})", k + 1
    used.add(out)
    return out


def vessel_payload(
    table: pd.DataFrame, key_of: Callable[[int, str], str] | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """``(rows, row → label id)`` for the results window."""
    rows: dict[str, dict[str, Any]] = {}
    row_labels: dict[str, int] = {}
    if table.empty:
        return rows, row_labels
    columns = [c for c in table.columns if c not in _IDENTITY]
    if "present" in columns:
        columns.remove("present")
        columns.insert(0, "present")
    used: set[str] = set()
    for _, r in table.sort_values("label").iterrows() if "label" in table.columns else table.iterrows():
        name = str(r.get("vessel_name", "") or "")
        lid = int(r["label"]) if "label" in r and pd.notna(r["label"]) else 0
        key = _unique(key_of(lid, name) if key_of is not None and lid else (name or f"label {lid}"), used)
        rows[key] = {c: _cell(c, r.get(c)) for c in columns}
        if lid:
            row_labels[key] = lid
    return rows, row_labels


#: Lesion columns shown, as ``table column → csv column``.
_LESION_COLUMNS: dict[str, str] = {
    "vessel": "vessel_name", "kind": "kind", "segment": "segment",
    "from origin (mm)": "start_s_from_vessel_origin_mm", "length (mm)": "length_mm",
    "taper %": "taper_percent", "diameter %": "diameter_percent", "area %": "area_percent",
    "reference": "reference", "reference r (mm)": "reference_radius_mm",
    "proximal r (mm)": "reference_proximal_radius_mm", "distal r (mm)": "reference_distal_radius_mm",
    "extreme r (mm)": "extreme_radius_mm",
    "confidence": "confidence", "class": "confidence_class", "flags": "confidence_flags",
}


def lesion_payload(lesions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for i, (_, r) in enumerate(lesions.iterrows(), start=1):
        key = f"{i}. {r.get('vessel_name', '')} {r.get('kind', '')}".strip()
        rows[key] = {col: _cell(src, r.get(src)) for col, src in _LESION_COLUMNS.items() if src in lesions.columns}
    return rows


def segment_payload(segments: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    skip = set(_IDENTITY) | {"path_id"}
    columns = [c for c in segments.columns if c not in skip]
    for _, r in segments.iterrows():
        key = _unique(f"{r.get('vessel_name', '')} · {r.get('segment', '')}", used)
        rows[key] = {c: _cell(c, r.get(c)) for c in columns}
    return rows


def subject_payload(subject: pd.DataFrame) -> dict[str, Any]:
    if subject.empty:
        return {}
    row = subject.iloc[0]
    return {c: _cell(c, row[c]) for c in subject.columns if c != "case_id"}


def surface_colourer(surface_layer: Any) -> Callable[..., None] | None:
    """An ``on_colour`` callback colouring *surface_layer*'s vessels like their labels."""
    if surface_layer is None:
        return None
    ref = weakref.ref(surface_layer)

    def on_colour(column: str | None, values: dict[int, float], colormap: str,
                  limits: tuple[float, float] | None) -> None:
        from nvitk.gui.labels.visibility import label_value_rgba
        from nvitk.gui.mesh.layers import add_field, fields_of, set_display

        layer = ref()
        if layer is None:
            return
        labels = fields_of(layer).get("label")
        if labels is None:
            return
        if column is None:
            set_display(layer, mode="field", field="label")
            return
        colours, _lo, _hi = label_value_rgba(np.unique(np.asarray(labels).astype(int)), values, colormap,
                                             limits=limits)
        add_field(layer, SURFACE_VALUE_FIELD, np.asarray(labels, dtype=np.float32),
                  categories={k: tuple(float(c) for c in v) for k, v in colours.items()}, show=True)

    return on_colour


def lesion_highlighter(viewer: Any, lesion_layer: Any, keys: list[str]) -> Callable[[str | None], None] | None:
    """An ``on_select`` callback: the clicked lesion is selected, brought into view and shown in the
    *Lesion inspector* (its calibre profile, :mod:`nvitk.gui.viz.lesion_inspector`)."""
    if lesion_layer is None:
        return None
    ref = weakref.ref(lesion_layer)
    index = {k: i for i, k in enumerate(keys)}

    def on_select(key: str | None) -> None:
        layer = ref()
        if layer is None:
            return
        if key is None or key not in index:
            layer.selected_data = set()
            return
        i = index[key]
        shown = np.asarray(getattr(layer, "shown", np.ones(len(layer.data), dtype=bool)), dtype=bool)
        if i < len(shown) and shown[i]:  # a hypoplasia (absent vessel) has no place to go to
            layer.selected_data = {i}
            world = np.asarray(layer.data_to_world(np.asarray(layer.data[i], dtype=float)), dtype=float)
            nd = len(viewer.dims.point)
            for axis, value in zip(range(nd - len(world), nd), world):
                viewer.dims.set_point(axis, float(value))
            try:
                viewer.camera.center = tuple(float(v) for v in world[-3:])
            except Exception:  # noqa: BLE001
                pass
        else:
            layer.selected_data = set()
        try:
            from nvitk.gui.viz.lesion_inspector import inspect_lesion

            inspect_lesion(viewer, i)
        except Exception:  # noqa: BLE001 — the selection already happened
            pass

    return on_select


def show_morphometrics_results(
    viewer: Any,
    stage7_dir: Path | str,
    *,
    label_layer: Any | None = None,
    surface_layer: Any | None = None,
    lesion_layer: Any | None = None,
    key_of: Callable[[int, str], str] | None = None,
    subtitle: str = "",
) -> dict[str, int]:
    """Show a morphometrics run in the results window (module docstring); returns row counts.

    *key_of* ``(label id, vessel name) → row name`` names the vessel rows (default: the vessel
    name). With *label_layer*, *Colour labels by* colours it — and *surface_layer*'s vessels —
    by any vessel column; *lesion_layer* is where clicked lesions are shown.
    """
    from nvitk.gui.viz.results_window import show_results

    d = Path(stage7_dir)
    subtitle = subtitle or str(d)
    counts = {"vessels": 0, "lesions": 0, "segments": 0}

    subject = subject_payload(_read_csv(d / "subject_summary.csv"))
    if subject:
        show_results(viewer, SUBJECT_TITLE, subject, subtitle=subtitle)

    segments = _read_csv(d / "segments.csv")
    if not segments.empty:
        rows = segment_payload(segments)
        counts["segments"] = len(rows)
        show_results(viewer, SEGMENTS_TITLE, rows, subtitle=subtitle, row_header="Segment")

    lesions = _read_csv(d / "lesions.csv")
    if not lesions.empty:
        from nvitk.gui.viz.lesion_inspector import LESION_CLASS_COLOURS, confidence_filter_controls

        rows = lesion_payload(lesions)
        counts["lesions"] = len(rows)
        classes = lesions.get("confidence_class", pd.Series([""] * len(lesions))).astype(str).tolist()
        tints = {key: LESION_CLASS_COLOURS[c] for key, c in zip(rows, classes) if c in LESION_CLASS_COLOURS}
        show_results(
            viewer, LESIONS_TITLE, rows, subtitle=subtitle, row_header="Lesion",
            row_colours=tints,
            controls=(lambda: confidence_filter_controls(viewer)) if lesion_layer is not None else None,
            note=("Severity: taper % against the healthy taper reference; diameter / area % against the "
                  "proximal reference. Confidence ∈ [0, 1] (high ≥ 0.7) with the penalties in flags. "
                  "Rows: green high, yellow moderate, red low confidence; the filter above hides the "
                  "less confident lesions in the viewer. Click a row to show the lesion and its calibre "
                  "profile (Lesion inspector); in the viewer, hover a lesion point for its card and click it "
                  "to inspect."),
            on_select=lesion_highlighter(viewer, lesion_layer, list(rows)),
        )

    table = vessel_table(d)
    rows, row_labels = vessel_payload(table, key_of)
    if rows:
        show_results(
            viewer, VESSELS_ALL_TITLE, rows, subtitle=subtitle, row_header="Vessel",
            note=("Every per-vessel measure of the run. Colour labels by: any column colours the measured "
                  "labels and the vessel surfaces. Columns ending in _legacy are cow_morpho's path metrics, "
                  "whose names the compact table uses for something else (see M-Morphometrics.md)."),
            label_layer=label_layer if row_labels else None, row_labels=row_labels,
            on_colour=surface_colourer(surface_layer),
        )
    # The main table: the stage-7 report's columns, as the pipeline writes them and the DB imports them.
    report_csv = d / REPORT_CSV
    report = _read_csv(report_csv) if report_csv.is_file() else morphometrics_report(d)
    rows, row_labels = vessel_payload(report, key_of)
    counts["vessels"] = len(rows)
    if rows:
        show_results(
            viewer, VESSELS_TITLE, rows, subtitle=subtitle, row_header="Vessel",
            note=("The stage-7 report (morphometrics_report.csv), the columns the pipeline publishes to the "
                  "database: the selected features and the high-confidence lesion flags. Every measure is in "
                  f"'{VESSELS_ALL_TITLE}'. Colour labels by: any column colours the measured labels and the "
                  "vessel surfaces."),
            label_layer=label_layer if row_labels else None, row_labels=row_labels,
            on_colour=surface_colourer(surface_layer),
        )
    return counts


def default_key_of(layer: Any) -> Callable[[int, str], str]:
    """Row names as the other measurement tables write them: ``"3 — Left ICA"`` with the label's
    own name, else with the vessel name from the topology, else ``"3"``."""

    def key_of(lid: int, vessel: str) -> str:
        name = ""
        try:
            from nvitk.gui.labels.catalog import label_name

            name = str(label_name(layer, int(lid)) or "")
        except Exception:  # noqa: BLE001
            name = ""
        if not name and vessel and not re.fullmatch(r"label_?\d+", vessel):
            name = vessel
        return f"{int(lid)} — {name}" if name else str(int(lid))

    return key_of


__all__ = [
    "LEGACY_PATH_COLUMNS",
    "TORTUOSITY_COLUMNS",
    "VOLUMETRY_COLUMNS",
    "default_key_of",
    "lesion_payload",
    "segment_payload",
    "show_morphometrics_results",
    "subject_payload",
    "surface_colourer",
    "vessel_payload",
    "vessel_table",
]
