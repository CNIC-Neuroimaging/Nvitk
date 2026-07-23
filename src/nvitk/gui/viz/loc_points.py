"""Load QVTpy ``locs.csv`` and display as Napari Points layers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.gui.core.spatial import layer_affine
from nvitk.gui.viz.layers import init_points_layer_style, install_points_style_sync

LOC_CSV_COLUMNS: tuple[str, ...] = (
    "vessel_id",
    "vessel_name",
    "segment_id",
    "loc_role",
    "centerline_index",
    "i",
    "j",
    "k",
    "centerline_x",
    "centerline_y",
    "centerline_z",
    "tangent_x",
    "tangent_y",
    "tangent_z",
    "loc_circularity",
    "loc_cross_section_area_mm2",
)

LOC_LAYER_NAME = "LOCs"
DEFAULT_LOC_POINT_SIZE = 2.0
DEFAULT_LOC_FACE_COLOR = "#ff0000"
DEFAULT_LOC_SYMBOL = "o"


def load_locs_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read stage-5 style ``locs.csv`` into a list of row dicts."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"LOC CSV not found: {p}")
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows in {p}")
    return rows


def locs_to_napari_points(
    rows: list[dict[str, Any]],
    reference_layer: Any | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Build Napari point coordinates and per-point feature columns.

    Prefers sub-voxel ``centerline_x/y/z`` when present; falls back to ``i/j/k``.
    """
    coords = []
    features = {c: [] for c in LOC_CSV_COLUMNS}

    for row in rows:
        if all(k in row and str(row.get(k, "")).strip() for k in ("centerline_x", "centerline_y", "centerline_z")):
            coords.append(
                np.array(
                    [
                        float(row["centerline_x"]),
                        float(row["centerline_y"]),
                        float(row["centerline_z"]),
                    ],
                    dtype=np.float64,
                )
            )
        else:
            coords.append(
                np.array([float(row["i"]), float(row["j"]), float(row["k"])], dtype=np.float64)
            )

        for col in LOC_CSV_COLUMNS:
            if col in row:
                val = row[col]
                if col in ("vessel_id", "segment_id", "centerline_index"):
                    features[col].append(int(float(val)))
                elif col in ("i", "j", "k"):
                    features[col].append(int(float(val)))
                elif col in ("loc_circularity", "loc_cross_section_area_mm2"):
                    features[col].append(float(val))
                else:
                    features[col].append(str(val))

    data = np.asarray(coords, dtype=np.float64)
    feat_out = {k: np.asarray(v) for k, v in features.items()}
    return data, feat_out


def add_locs_layer(
    viewer: Any,
    rows: list[dict[str, Any]],
    *,
    reference_layer = None,
    name = LOC_LAYER_NAME,
) -> Any:
    """Add or replace a Points layer for LOC rows."""
    coords, features = locs_to_napari_points(rows, reference_layer)
    for lyr in list(viewer.layers):
        if lyr.name == name:
            viewer.layers.remove(lyr)
    kwargs = {
        "size": DEFAULT_LOC_POINT_SIZE,
        "face_color": DEFAULT_LOC_FACE_COLOR,
        "symbol": DEFAULT_LOC_SYMBOL,
        "border_width": 0,
        "border_width_is_relative": False,
    }
    if reference_layer is not None:
        aff = layer_affine(reference_layer)
        if aff is not None:
            kwargs["affine"] = aff
    layer = viewer.add_points(coords, name=name, features=features, **kwargs)
    init_points_layer_style(
        layer,
        size=DEFAULT_LOC_POINT_SIZE,
        symbol=DEFAULT_LOC_SYMBOL,
        face_color=DEFAULT_LOC_FACE_COLOR,
    )
    install_points_style_sync(layer, sync_face_color=True)
    return layer


def remove_locs_layer(viewer: Any, name: str = LOC_LAYER_NAME) -> None:
    for lyr in list(viewer.layers):
        if lyr.name == name:
            viewer.layers.remove(lyr)
