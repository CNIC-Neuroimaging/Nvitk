"""Load and write ordered vessel centerlines from stage-3 / stage-4 outputs.

**Inputs**

- ``centerlines_mask.nii.gz`` + ``centerline_meta.json`` (stage 3).
- Optional ``centerlines_mask_4dflow.nii.gz`` from stage-4 ``seg_4dflow`` skeletonization.

**Outputs**

- Polylines keyed by qvtpy arterial label id or venous name (SSSV, …).
- Rasterized multilabel masks via :func:`rasterize_centerlines_mask`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.io.imageio import imread, imsave
from nvitk.morphology.centerline import compute_centerlines
from nvitk.pipes.qvtpy.labels import (
    QVTPY_ARTERIAL_LABEL_IDS,
    QVTPY_BASILAR,
    QVTPY_LACA,
    QVTPY_LICA,
    QVTPY_LMCA,
    QVTPY_LPCA,
    QVTPY_RACA,
    QVTPY_RICA,
    QVTPY_RMCA,
    QVTPY_RPCA,
)
from nvitk.pipes.qvtpy.util.venous_heuristics import venous_name_to_label_id

setup(globals())

# Child → parent label for proximal seed when regenerating CLs from seg_4dflow.
_ARTERIAL_PARENT_LABEL: dict[int, int] = {
    QVTPY_LMCA: QVTPY_LICA,
    QVTPY_RMCA: QVTPY_RICA,
    QVTPY_LACA: QVTPY_LICA,
    QVTPY_RACA: QVTPY_RICA,
    QVTPY_LPCA: QVTPY_BASILAR,
    QVTPY_RPCA: QVTPY_BASILAR,
}

# ---------------------------------------------------------------------------
# Artifact names
# ---------------------------------------------------------------------------

CENTERLINES_MASK_NIFTI = "centerlines_mask.nii.gz"
CENTERLINE_META_JSON = "centerline_meta.json"

CENTERLINES_MASK_SEG_NIFTI = "centerlines_mask_4dflow.nii.gz"
CENTERLINE_SEG_META_JSON = "centerlines_seg_meta.json"


# ---------------------------------------------------------------------------
# Path helpers + meta JSON
# ---------------------------------------------------------------------------


def centerlines_mask_path(stage_dir: Path, *, from_segmentation: bool = False) -> Path:
    """Path to the centerline multilabel NIfTI in *stage_dir*."""
    name = CENTERLINES_MASK_SEG_NIFTI if from_segmentation else CENTERLINES_MASK_NIFTI
    return Path(stage_dir) / name


def centerline_meta_path(stage_dir: Path, *, from_segmentation: bool = False) -> Path:
    """Path to the JSON sidecar listing arterial / venous label ids."""
    name = CENTERLINE_SEG_META_JSON if from_segmentation else CENTERLINE_META_JSON
    return Path(stage_dir) / name


def load_centerline_meta(stage_dir: Path, *, from_segmentation: bool = False) -> dict[str, Any]:
    """Parse centerline JSON metadata from *stage_dir*."""
    p = centerline_meta_path(stage_dir, from_segmentation=from_segmentation)
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Rasterize polylines → multilabel mask
# ---------------------------------------------------------------------------


def rasterize_centerlines_mask(
    shape: tuple[int, int, int],
    arterial: dict[int, Any],
    venous: dict[str, Any] | None = None,
    *,
    venous_label_by_name: dict[str, int] | None = None,
) -> np.ndarray:
    """Voxel mask with qvtpy label id on each centerline point."""
    mask = np.zeros(shape, dtype=np.int32)
    for vid, pts in sorted(arterial.items()):
        p = as_backend_array(pts)
        for row in p:
            i, j, k = (
                int(round(float(row[0]))),
                int(round(float(row[1]))),
                int(round(float(row[2]))),
            )
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                mask[i, j, k] = int(vid)
    if venous:
        for name, pts in venous.items():
            lid = int(
                (venous_label_by_name or {}).get(name)
                or venous_name_to_label_id(name)
            )
            p = as_backend_array(pts)
            for row in p:
                i, j, k = (
                    int(round(float(row[0]))),
                    int(round(float(row[1]))),
                    int(round(float(row[2]))),
                )
                if (
                    0 <= i < shape[0]
                    and 0 <= j < shape[1]
                    and 0 <= k < shape[2]
                    and mask[i, j, k] == 0
                ):
                    mask[i, j, k] = lid
    return mask


def _parent_contact_seed(
    seg: np.ndarray,
    child_id: int,
    parent_id: int,
) -> np.ndarray | None:
    """Voxel near the child/parent interface (proximal trunk seed)."""
    seg_np = to_numpy(seg).astype(np.int32, copy=False)
    child = seg_np == int(child_id)
    parent = seg_np == int(parent_id)
    if not np.any(child) or not np.any(parent):
        return None
    try:
        from scipy import ndimage

        parent_d = ndimage.binary_dilation(parent, iterations=2)
        contact = child & parent_d
    except Exception:
        contact = child & parent
    if np.any(contact):
        return np.mean(np.argwhere(contact).astype(np.float64), axis=0)
    child_coords = np.argwhere(child).astype(np.float64)
    parent_centroid = np.mean(np.argwhere(parent).astype(np.float64), axis=0)
    d2 = np.sum((child_coords - parent_centroid.reshape(1, 3)) ** 2, axis=1)
    return child_coords[int(np.argmin(d2))]


def export_centerlines_from_segmentation(
    seg: np.ndarray,
    out_dir: Path,
    *,
    metadata: dict[str, Any] | None = None,
    min_points: int = 3,
    venous_polylines: dict[str, Any] | None = None,
    venous_label_by_name: dict[str, int] | None = None,
    prefer_polylines: dict[int, Any] | None = None,
) -> tuple[Path, Path]:
    """Build centerlines from ``seg_4dflow`` and write mask + meta under *out_dir*.

    *prefer_polylines* (typically stage-3 arterial CLs) bias branched vessels so
    proximal trunks are not dropped by a distal-only graph diameter. When absent,
    MCA/ACA/PCA use a parent-contact seed from the segmentation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    shape = tuple(int(s) for s in seg_np.shape[:3])
    prefer = {int(k): v for k, v in (prefer_polylines or {}).items() if v is not None}

    label_ids = sorted(int(v) for v in np.unique(seg_np) if int(v) > 0)
    arterial_polylines: dict[int, Any] = {}
    for lid in label_ids:
        if int(lid) not in QVTPY_ARTERIAL_LABEL_IDS:
            continue
        prefs = prefer.get(int(lid))
        if prefs is None:
            parent = _ARTERIAL_PARENT_LABEL.get(int(lid))
            if parent is not None:
                seed = _parent_contact_seed(seg_np, int(lid), int(parent))
                if seed is not None:
                    prefs = seed.reshape(1, 3)
        cl = compute_centerlines(
            seg_np,
            labels=[int(lid)],
            min_points=int(min_points),
            prefer_points_by_label={int(lid): prefs} if prefs is not None else None,
        )
        pts = cl.get(int(lid))
        if pts is not None:
            arterial_polylines[int(lid)] = pts

    venous_out = venous_polylines or {}
    mask = rasterize_centerlines_mask(
        shape,
        arterial_polylines,
        venous_out,
        venous_label_by_name=venous_label_by_name,
    )
    mask_path = centerlines_mask_path(out_dir, from_segmentation=True)
    meta_path = centerline_meta_path(out_dir, from_segmentation=True)
    imsave(mask_path, mask, metadata=dict(metadata or {}))
    meta = {
        "source": "seg_4dflow",
        "arterial_labels": sorted(arterial_polylines.keys()),
        "venous_vessels": list(venous_out.keys()),
        "venous_label_by_name": venous_label_by_name or {},
        "n_arterial_points": int(sum(p.shape[0] for p in arterial_polylines.values())),
        "n_venous_points": int(sum(p.shape[0] for p in venous_out.values())),
        "centerlines_mask_nifti": str(mask_path),
        "prefer_polylines_labels": sorted(prefer.keys()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return mask_path, meta_path


# ---------------------------------------------------------------------------
# Polyline extraction from multilabel centerline mask
# ---------------------------------------------------------------------------


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
    stage_dir: Path,
    *,
    min_points: int = 3,
    meta: dict[str, Any] | None = None,
    from_segmentation: bool = False,
) -> dict[int, Any]:
    """Arterial centerlines keyed by qvtpy label id."""
    stage_dir = Path(stage_dir)
    meta = meta or load_centerline_meta(stage_dir, from_segmentation=from_segmentation)
    mask_path = centerlines_mask_path(stage_dir, from_segmentation=from_segmentation)
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
    stage_dir: Path,
    *,
    min_points: int = 3,
    meta: dict[str, Any] | None = None,
    from_segmentation: bool = False,
) -> dict[str, Any]:
    """Venous centerlines keyed by vessel name (SSSV, STRV, …)."""
    stage_dir = Path(stage_dir)
    meta = meta or load_centerline_meta(stage_dir, from_segmentation=from_segmentation)
    mask_path = centerlines_mask_path(stage_dir, from_segmentation=from_segmentation)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing {mask_path}")
    mask = as_backend_array(imread(mask_path).data).astype(np.int32, copy=False)
    name_to_id = {str(k): int(v) for k, v in (meta.get("venous_label_by_name") or {}).items()}
    venous: dict[str, Any] = {}
    for name in meta.get("venous_vessels", []):
        lid = name_to_id.get(str(name))
        if lid is None:
            continue
        pts = _polyline_for_label(mask, int(lid), min_points=min_points)
        if pts is not None:
            venous[str(name)] = pts
    return venous


def load_centerlines(
    stage3_dir: Path,
    *,
    min_points: int = 3,
    stage4_dir: Path | None = None,
    truncate_length_ratio: float = 0.6,
) -> tuple[dict[int, Any], dict[str, Any], dict[str, Any]]:
    """Return ``(arterial, venous, meta)``.

    Arterial polylines prefer stage-4 segmentation centerlines when available; venous
    still come from stage-3 (not present in ``seg_4dflow``). Stage-3 is used when a
    stage-4 label is missing, or when the stage-4 polyline is much shorter than
    stage-3 (truncated diameter path), controlled by *truncate_length_ratio*.
    """
    stage3_dir = Path(stage3_dir)
    meta3 = load_centerline_meta(stage3_dir)
    venous = load_venous_centerlines(stage3_dir, min_points=min_points, meta=meta3)

    s4_cl = Path(stage4_dir) if stage4_dir is not None else None
    if s4_cl is not None and centerlines_mask_path(s4_cl, from_segmentation=True).is_file():
        meta_cl = load_centerline_meta(s4_cl, from_segmentation=True)
        arterial = load_arterial_centerlines(
            s4_cl, min_points=min_points, meta=meta_cl, from_segmentation=True
        )
        # Recover missing / truncated arterial CLs from stage-3.
        s3_arterial = load_arterial_centerlines(
            stage3_dir, min_points=min_points, meta=meta3
        )
        ratio = float(truncate_length_ratio)
        for lid, pts3 in s3_arterial.items():
            if pts3 is None:
                continue
            n3 = int(np.asarray(pts3).shape[0])
            if n3 < int(min_points):
                continue
            pts4 = arterial.get(int(lid))
            if pts4 is None:
                arterial[int(lid)] = pts3
                continue
            n4 = int(np.asarray(pts4).shape[0])
            if n4 < int(min_points) or (ratio > 0.0 and n4 < ratio * n3):
                arterial[int(lid)] = pts3
        return arterial, venous, meta_cl

    arterial = load_arterial_centerlines(stage3_dir, min_points=min_points, meta=meta3)
    return arterial, venous, meta3


__all__ = [
    "CENTERLINE_META_JSON",
    "CENTERLINE_SEG_META_JSON",
    "CENTERLINES_MASK_NIFTI",
    "CENTERLINES_MASK_SEG_NIFTI",
    "centerline_meta_path",
    "centerlines_mask_path",
    "export_centerlines_from_segmentation",
    "load_arterial_centerlines",
    "load_centerline_meta",
    "load_centerlines",
    "load_venous_centerlines",
    "rasterize_centerlines_mask",
]
