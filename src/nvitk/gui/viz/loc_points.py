"""Load QVTpy ``locs.csv`` and display as Napari Points layers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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

#: Snap radius (voxels) for binding a cross-section pick to a stage-5 LOC pose.
LOC_SNAP_DISTANCE_VOX: float = 2.5


@dataclass(frozen=True)
class LocPose:
    """Stage-5 LOC geometry: the exact pose stage 6 measures flow at."""

    vessel_id: int
    vessel_name: str
    loc_role: str
    center: np.ndarray  # (3,) voxel coords
    tangent: np.ndarray  # (3,) unit tangent from locs.csv
    centerline_index: int | None = None

    def label(self) -> str:
        """Short display label including optional role (init/fin)."""
        role = str(self.loc_role or "").strip()
        if role:
            return f"{self.vessel_name} [{role}]"
        return self.vessel_name


def parse_loc_poses(rows: Sequence[Mapping[str, Any]]) -> list[LocPose]:
    """Build :class:`LocPose` entries from stage-5 ``locs.csv`` row dicts."""
    out: list[LocPose] = []
    for row in rows:
        name = str(row.get("vessel_name") or "").strip()
        if not name:
            continue
        try:
            if all(
                str(row.get(k, "")).strip()
                for k in ("centerline_x", "centerline_y", "centerline_z")
            ):
                center = np.array(
                    [
                        float(row["centerline_x"]),
                        float(row["centerline_y"]),
                        float(row["centerline_z"]),
                    ],
                    dtype=np.float64,
                )
            else:
                center = np.array(
                    [float(row["i"]), float(row["j"]), float(row["k"])],
                    dtype=np.float64,
                )
            tangent = np.array(
                [
                    float(row["tangent_x"]),
                    float(row["tangent_y"]),
                    float(row["tangent_z"]),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue
        n = float(np.linalg.norm(tangent))
        if n < 1e-9 or not np.all(np.isfinite(center)):
            continue
        tangent = tangent / n
        try:
            vessel_id = int(float(row.get("vessel_id") or 0))
        except (TypeError, ValueError):
            vessel_id = 0
        try:
            cl_idx_raw = row.get("centerline_index")
            cl_idx = (
                int(float(cl_idx_raw))
                if cl_idx_raw not in (None, "")
                else None
            )
        except (TypeError, ValueError):
            cl_idx = None
        out.append(
            LocPose(
                vessel_id=vessel_id,
                vessel_name=name,
                loc_role=str(row.get("loc_role") or "").strip(),
                center=center,
                tangent=tangent,
                centerline_index=cl_idx,
            )
        )
    return out


def nearest_loc_pose(
    poses: Sequence[LocPose],
    xyz: np.ndarray,
    *,
    max_distance_vox: float = LOC_SNAP_DISTANCE_VOX,
) -> LocPose | None:
    """Closest LOC within *max_distance_vox*, or ``None`` if none are near enough."""
    if not poses:
        return None
    p = np.asarray(xyz, dtype=np.float64).reshape(3)
    max_d2 = float(max_distance_vox) ** 2
    best: LocPose | None = None
    best_d2 = float("inf")
    for loc in poses:
        d2 = float(np.sum((loc.center - p) ** 2))
        if d2 < best_d2 and d2 <= max_d2:
            best = loc
            best_d2 = d2
    return best


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
    reference_layer=None,
    name=LOC_LAYER_NAME,
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
    """Remove the LOC points layer named *name* from *viewer*, if present."""
    for lyr in list(viewer.layers):
        if lyr.name == name:
            viewer.layers.remove(lyr)


__all__ = [
    "DEFAULT_LOC_FACE_COLOR",
    "DEFAULT_LOC_POINT_SIZE",
    "DEFAULT_LOC_SYMBOL",
    "LOC_CSV_COLUMNS",
    "LOC_LAYER_NAME",
    "LOC_SNAP_DISTANCE_VOX",
    "LocPose",
    "add_locs_layer",
    "load_locs_csv",
    "locs_to_napari_points",
    "nearest_loc_pose",
    "parse_loc_poses",
    "remove_locs_layer",
]
