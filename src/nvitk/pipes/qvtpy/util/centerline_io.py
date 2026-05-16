"""Load ordered vessel centerlines from stage-3 NIfTI outputs (no NPZ)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.io.imageio import imread
from nvitk.morphology.centerline import compute_centerlines

setup(globals())

CENTERLINES_MASK_NIFTI = "centerlines_mask.nii.gz"
CENTERLINE_META_JSON = "centerline_meta.json"


def centerlines_mask_path(stage3_dir: Path) -> Path:
    return Path(stage3_dir) / CENTERLINES_MASK_NIFTI


def centerline_meta_path(stage3_dir: Path) -> Path:
    return Path(stage3_dir) / CENTERLINE_META_JSON


def load_centerline_meta(stage3_dir: Path) -> dict[str, Any]:
    p = centerline_meta_path(stage3_dir)
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _polyline_for_label(
    mask: Any,
    label_id: int,
    *,
    min_points: int,
) -> Any | None:
    """Ordered (N,3) polyline for voxels with *label_id* in the centerline mask."""
    m = to_numpy(mask).astype(np.int32, copy=False)
    roi = m == int(label_id)
    if not roi.any():
        return None
    cl = compute_centerlines(
        m,
        centerline_mask=roi,
        labels=[int(label_id)],
        min_points=int(min_points),
    )
    return cl.get(int(label_id))


def load_arterial_centerlines(
    stage3_dir: Path,
    *,
    min_points: int = 5,
    meta: dict[str, Any] | None = None,
) -> dict[int, Any]:
    """Arterial centerlines keyed by eICAB label id."""
    stage3_dir = Path(stage3_dir)
    meta = meta or load_centerline_meta(stage3_dir)
    mask_path = centerlines_mask_path(stage3_dir)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing {mask_path}")
    mask = as_backend_array(imread(mask_path).data).astype(np.int32, copy=False)
    arterial: dict[int, Any] = {}
    for lid in meta.get("arterial_labels", []):
        lid = int(lid)
        pts = _polyline_for_label(mask, lid, min_points=min_points)
        if pts is not None:
            arterial[lid] = pts
    return arterial


def load_venous_centerlines(
    stage3_dir: Path,
    *,
    min_points: int = 5,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Venous centerlines keyed by vessel name (SSSV, STRV, …)."""
    stage3_dir = Path(stage3_dir)
    meta = meta or load_centerline_meta(stage3_dir)
    mask_path = centerlines_mask_path(stage3_dir)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing {mask_path}")
    mask = as_backend_array(imread(mask_path).data).astype(np.int32, copy=False)
    name_to_id = {str(k): int(v) for k, v in (meta.get("venous_label_by_name") or {}).items()}
    venous: dict[str, Any] = {}
    for name, lid in name_to_id.items():
        pts = _polyline_for_label(mask, lid, min_points=min_points)
        if pts is not None:
            venous[name] = pts
    return venous


def load_centerlines(
    stage3_dir: Path,
    *,
    min_points: int = 5,
) -> tuple[dict[int, Any], dict[str, Any], dict[str, Any]]:
    """Return ``(arterial, venous, meta)`` from stage-3 NIfTI + JSON meta."""
    meta = load_centerline_meta(stage3_dir)
    arterial = load_arterial_centerlines(stage3_dir, min_points=min_points, meta=meta)
    venous = load_venous_centerlines(stage3_dir, min_points=min_points, meta=meta)
    return arterial, venous, meta


__all__ = [
    "CENTERLINE_META_JSON",
    "CENTERLINES_MASK_NIFTI",
    "centerline_meta_path",
    "centerlines_mask_path",
    "load_arterial_centerlines",
    "load_centerline_meta",
    "load_centerlines",
    "load_venous_centerlines",
]
