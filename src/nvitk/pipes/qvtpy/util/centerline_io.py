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
from nvitk.morphology.centerline import compute_centerline_branches, compute_centerlines
from nvitk.pipes.qvtpy.labels import (
    QVTPY_ARTERIAL_LABEL_IDS,
    QVTPY_BASILAR,
    QVTPY_BRANCHED_LABEL_IDS,
    QVTPY_LACA,
    QVTPY_LICA,
    QVTPY_LMCA,
    QVTPY_LPCA,
    QVTPY_MCA_IDS,
    QVTPY_ACA_IDS,
    QVTPY_PCA_IDS,
    QVTPY_RACA,
    QVTPY_RICA,
    QVTPY_RMCA,
    QVTPY_RPCA,
    QVTPY_SMALL_ARTERIAL_IDS,
    qvtpy_branch_names,
    qvtpy_branch_parent_label,
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

# MCA/ACA/PCA/comm: keep shorter skeletons when regenerating from segmentation / eICAB.
_SMALL_BRANCH_MIN_POINTS = 2
_SMALL_CENTERLINE_LABEL_IDS = QVTPY_MCA_IDS | QVTPY_ACA_IDS | QVTPY_SMALL_ARTERIAL_IDS
_DEFAULT_SMOOTH_WINDOW = 5
_DEFAULT_SMOOTH_SPLINE = True

# ---------------------------------------------------------------------------
# Artifact names
# ---------------------------------------------------------------------------

CENTERLINES_MASK_NIFTI = "centerlines_mask.nii.gz"
CENTERLINE_META_JSON = "centerline_meta.json"

CENTERLINES_MASK_SEG_NIFTI = "centerlines_mask_4dflow.nii.gz"
CENTERLINE_SEG_META_JSON = "centerlines_seg_meta.json"
CENTERLINE_SEG_BRANCHES_JSON = "centerlines_seg_branches.json"

# Named arterial branches: parent qvtpy label id -> [(branch_name, (N,3) points), ...].
# The first entry per label is the trunk (main path).
ArterialBranches = dict


def main_path_of(branches: list[tuple[str, Any]]) -> Any | None:
    """Trunk (main) polyline of a per-label branch list, or ``None`` if empty."""
    if not branches:
        return None
    return branches[0][1]


def arterial_main_paths(arterial: dict[int, list[tuple[str, Any]]]) -> dict[int, Any]:
    """Legacy view: parent label id -> trunk polyline (drops side branches)."""
    out: dict[int, Any] = {}
    for lid, branches in arterial.items():
        pts = main_path_of(branches)
        if pts is not None:
            out[int(lid)] = pts
    return out


def flatten_branches(
    arterial: dict[int, list[tuple[str, Any]]],
) -> dict[str, Any]:
    """Name-keyed view: branch name -> polyline (across all parent labels)."""
    out: dict[str, Any] = {}
    for branches in arterial.values():
        for name, pts in branches:
            out[str(name)] = pts
    return out


def parent_label_of(branch_name: str) -> int | None:
    """Parent qvtpy label id for a branch name like ``LMCA-M2a``."""
    return qvtpy_branch_parent_label(branch_name)


CENTERLINE_BRANCH_VTP_DIR = "branch_vtps"


def export_branch_vtps(
    arterial: dict[int, list[tuple[str, Any]]],
    out_dir: Path,
) -> list[Path]:
    """Write one ``.vtp`` polyline per named branch (for tortuosity metrics).

    Files are named ``{parent_label}_{branch_name}.vtp`` (e.g. ``6_LMCA-M2a.vtp``)
    so :func:`nvitk.measure.morpho.export_utils.compute_tortuosity_metrics.load_centerlines_from_vtp_folder`
    keys each metrics row by the branch name. Points are the (voxel-space)
    centerline coordinates. Best-effort: returns an empty list when VTK is
    unavailable.
    """
    try:
        from nvitk.measure.morpho.surface import build_polyline_polydata, save_vtp
    except Exception as exc:  # noqa: BLE001
        log = None
        try:
            from nvitk.core.logger import Logger

            log = Logger()
        except Exception:
            pass
        if log is not None:
            log.warning(f"branch VTP export skipped (VTK unavailable: {exc})")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for lid, branches in sorted(arterial.items()):
        for name, pts in branches:
            pts_np = to_numpy(pts).astype(float).reshape(-1, 3)
            if pts_np.shape[0] < 2:
                continue
            poly = build_polyline_polydata(pts_np, [])
            path = out_dir / f"{int(lid)}_{str(name)}.vtp"
            save_vtp(poly, str(path))
            written.append(path)
    return written


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


def smooth_centerline_polyline(
    points: Any,
    *,
    window: int = _DEFAULT_SMOOTH_WINDOW,
    use_spline: bool = _DEFAULT_SMOOTH_SPLINE,
) -> Any:
    """Light smoothing of an ordered centerline polyline (voxel coords).

    Uses a centered moving average; optionally resamples with a mild spline
    (``s>0``) so small distal branches remain continuous without sharp kinks.
    """
    import numpy as np

    pts = to_numpy(points).astype(np.float64, copy=False)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        return as_backend_array(pts)

    w = max(3, int(window) | 1)  # odd window
    if pts.shape[0] < w:
        w = pts.shape[0] if pts.shape[0] % 2 == 1 else max(3, pts.shape[0] - 1)
    if w >= 3 and pts.shape[0] >= w:
        pad = w // 2
        padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(w, dtype=np.float64) / float(w)
        smoothed = np.stack(
            [np.convolve(padded[:, d], kernel, mode="valid") for d in range(3)],
            axis=1,
        )
    else:
        smoothed = pts

    if use_spline and smoothed.shape[0] >= 4:
        try:
            from scipy.interpolate import splprep, splev

            # Mild smoothing factor proportional to polyline length.
            s = float(smoothed.shape[0]) * 0.35
            tck, _ = splprep(smoothed.T, s=s, k=min(3, smoothed.shape[0] - 1))
            u = np.linspace(0.0, 1.0, smoothed.shape[0])
            smoothed = np.stack(splev(u, tck), axis=1)
        except Exception:
            pass
    return as_backend_array(smoothed.astype(np.float32, copy=False))


def centerlines_from_segmentation(
    seg: np.ndarray,
    *,
    min_points: int = 3,
    min_branch_points: int = 3,
    prefer_polylines: dict[int, Any] | None = None,
    labels: list[int] | None = None,
    smooth: bool = True,
    smooth_window: int = _DEFAULT_SMOOTH_WINDOW,
) -> dict[int, list[tuple[str, Any]]]:
    """Named arterial branch polylines from ``seg_4dflow``.

    Returns ``{parent_label: [(branch_name, (N,3) points), ...]}`` with the trunk
    first. Branched territories (MCA/ACA/PCA) are decomposed into a trunk plus
    bifurcation side branches (named e.g. ``LMCA-M1`` / ``LMCA-M2a`` via
    :func:`~nvitk.pipes.qvtpy.labels.qvtpy_branch_names`); ICA/basilar/comm/vertebral
    stay single-path (one branch = the bare vessel name).

    *prefer_polylines* (typically stage-3 arterial CLs) bias branched vessels so
    proximal trunks are not dropped by a distal-only graph diameter. When absent,
    MCA/ACA/PCA use a parent-contact seed from the segmentation. Optional light
    smoothing is applied per branch by default.
    """
    import numpy as np

    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    prefer = {int(k): v for k, v in (prefer_polylines or {}).items() if v is not None}
    if labels is None:
        label_ids = sorted(int(v) for v in np.unique(seg_np) if int(v) > 0)
    else:
        label_ids = sorted(int(v) for v in labels)

    branch_ids = QVTPY_MCA_IDS | QVTPY_ACA_IDS | QVTPY_PCA_IDS
    arterial: dict[int, list[tuple[str, Any]]] = {}
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
        lid_min = (
            min(int(min_points), _SMALL_BRANCH_MIN_POINTS)
            if int(lid) in _SMALL_CENTERLINE_LABEL_IDS
            else int(min_points)
        )
        if int(lid) in QVTPY_BRANCHED_LABEL_IDS:
            br = compute_centerline_branches(
                seg_np,
                labels=[int(lid)],
                min_points=int(lid_min),
                min_branch_points=int(min_branch_points),
                prefer_points_by_label={int(lid): prefs} if prefs is not None else None,
            )
            paths = br.get(int(lid)) or []
        else:
            cl = compute_centerlines(
                seg_np,
                labels=[int(lid)],
                min_points=int(lid_min),
                prefer_points_by_label={int(lid): prefs} if prefs is not None else None,
            )
            pts = cl.get(int(lid))
            paths = [pts] if pts is not None else []
        if not paths:
            continue
        names = qvtpy_branch_names(int(lid), len(paths))
        named: list[tuple[str, Any]] = []
        for name, pts in zip(names, paths):
            if pts is None:
                continue
            if smooth:
                pts = smooth_centerline_polyline(pts, window=smooth_window)
            named.append((str(name), pts))
        if named:
            arterial[int(lid)] = named
    return arterial


def export_centerlines_from_segmentation(
    seg: np.ndarray,
    out_dir: Path,
    *,
    metadata: dict[str, Any] | None = None,
    min_points: int = 3,
    venous_polylines: dict[str, Any] | None = None,
    venous_label_by_name: dict[str, int] | None = None,
    prefer_polylines: dict[int, Any] | None = None,
    smooth: bool = True,
    smooth_window: int = _DEFAULT_SMOOTH_WINDOW,
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

    arterial = centerlines_from_segmentation(
        seg_np,
        min_points=int(min_points),
        prefer_polylines=prefer_polylines,
        smooth=bool(smooth),
        smooth_window=int(smooth_window),
    )

    # The multilabel NIfTI can only carry one int per voxel, so every branch of a
    # vessel paints the parent label (viz/back-compat). Branch identity + points
    # are persisted separately in the branches sidecar JSON below.
    arterial_for_mask = {
        int(lid): np.vstack([to_numpy(pts) for _n, pts in branches])
        for lid, branches in arterial.items()
        if branches
    }
    venous_out = venous_polylines or {}
    mask = rasterize_centerlines_mask(
        shape,
        arterial_for_mask,
        venous_out,
        venous_label_by_name=venous_label_by_name,
    )
    mask_path = centerlines_mask_path(out_dir, from_segmentation=True)
    meta_path = centerline_meta_path(out_dir, from_segmentation=True)
    branches_path = out_dir / CENTERLINE_SEG_BRANCHES_JSON
    imsave(mask_path, mask, metadata=dict(metadata or {}))

    branch_records: list[dict[str, Any]] = []
    arterial_branches_meta: list[dict[str, Any]] = []
    n_arterial_points = 0
    for lid, branches in sorted(arterial.items()):
        for gen_idx, (name, pts) in enumerate(branches):
            pts_np = to_numpy(pts).astype(float)
            n_arterial_points += int(pts_np.shape[0])
            branch_records.append(
                {
                    "branch_name": str(name),
                    "parent_label": int(lid),
                    "generation": int(gen_idx),
                    "is_trunk": bool(gen_idx == 0),
                    "n_points": int(pts_np.shape[0]),
                    "points": pts_np.tolist(),
                }
            )
            arterial_branches_meta.append(
                {
                    "branch_name": str(name),
                    "parent_label": int(lid),
                    "is_trunk": bool(gen_idx == 0),
                    "n_points": int(pts_np.shape[0]),
                }
            )
    branches_path.write_text(
        json.dumps({"source": "seg_4dflow", "branches": branch_records}, indent=2),
        encoding="utf-8",
    )

    # Per-branch VTP polylines for tortuosity metrics (best-effort; needs VTK).
    branch_vtp_paths = export_branch_vtps(arterial, out_dir / CENTERLINE_BRANCH_VTP_DIR)

    meta = {
        "source": "seg_4dflow",
        "arterial_labels": sorted(arterial.keys()),
        "arterial_branches": arterial_branches_meta,
        "branches_json": str(branches_path),
        "venous_vessels": list(venous_out.keys()),
        "venous_label_by_name": venous_label_by_name or {},
        "n_arterial_points": int(n_arterial_points),
        "n_venous_points": int(sum(p.shape[0] for p in venous_out.values())),
        "centerlines_mask_nifti": str(mask_path),
        "branch_vtp_dir": str(out_dir / CENTERLINE_BRANCH_VTP_DIR),
        "n_branch_vtps": int(len(branch_vtp_paths)),
        "prefer_polylines_labels": sorted(prefer.keys()),
        "smooth": bool(smooth),
        "smooth_window": int(smooth_window),
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


def load_arterial_branches(
    stage_dir: Path,
    *,
    min_points: int = 3,
    meta: dict[str, Any] | None = None,
    from_segmentation: bool = False,
) -> dict[int, list[tuple[str, Any]]]:
    """Named arterial branches keyed by parent qvtpy label id.

    Prefers the ``centerlines_seg_branches.json`` sidecar (multi-branch, named).
    Falls back to reconstructing one trunk polyline per label from the multilabel
    centerline mask (wrapped as ``[(vessel_name, pts)]``) when the sidecar is
    absent (e.g. stage-3, or older stage-4 outputs).
    """
    from nvitk.pipes.qvtpy.labels import qvtpy_vessel_name

    stage_dir = Path(stage_dir)
    meta = meta or load_centerline_meta(stage_dir, from_segmentation=from_segmentation)
    branches_path = stage_dir / CENTERLINE_SEG_BRANCHES_JSON
    if from_segmentation and branches_path.is_file():
        doc = json.loads(branches_path.read_text(encoding="utf-8"))
        out: dict[int, list[tuple[str, Any]]] = {}
        for rec in doc.get("branches", []):
            pts = as_backend_array(
                np.asarray(rec.get("points", []), dtype=np.float32)
            )
            if int(np.asarray(pts).shape[0]) < int(min_points):
                continue
            lid = int(rec.get("parent_label"))
            out.setdefault(lid, []).append((str(rec.get("branch_name")), pts))
        if out:
            return out

    # Fallback: single trunk polyline per label from the mask.
    mask_path = centerlines_mask_path(stage_dir, from_segmentation=from_segmentation)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing {mask_path}")
    mask = as_backend_array(imread(mask_path).data).astype(np.int32, copy=False)
    fallback: dict[int, list[tuple[str, Any]]] = {}
    for lid in meta.get("arterial_labels", []):
        lid = int(lid)
        pts = _polyline_for_label(mask, lid, min_points=min_points)
        if pts is not None:
            fallback[lid] = [(qvtpy_vessel_name(lid), pts)]
    return fallback


def load_arterial_centerlines(
    stage_dir: Path,
    *,
    min_points: int = 3,
    meta: dict[str, Any] | None = None,
    from_segmentation: bool = False,
) -> dict[int, Any]:
    """Arterial trunk (main-path) polylines keyed by qvtpy label id.

    Legacy single-polyline view: returns only the trunk of each vessel. Use
    :func:`load_arterial_branches` for the full named bifurcation set.
    """
    branches = load_arterial_branches(
        stage_dir,
        min_points=min_points,
        meta=meta,
        from_segmentation=from_segmentation,
    )
    return arterial_main_paths(branches)


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


def load_centerlines_branches(
    stage3_dir: Path,
    *,
    min_points: int = 3,
    stage4_dir: Path | None = None,
    truncate_length_ratio: float = 0.6,
) -> tuple[dict[int, list[tuple[str, Any]]], dict[str, Any], dict[str, Any]]:
    """Return ``(arterial_branches, venous, meta)`` (named multi-branch arterial).

    Like :func:`load_centerlines` but keeps every named branch per vessel. Stage-4
    branches are preferred when present; a vessel whose stage-4 trunk is missing or
    much shorter than stage-3 (see *truncate_length_ratio*) falls back to the
    stage-3 single polyline (wrapped as the vessel's trunk).
    """
    stage3_dir = Path(stage3_dir)
    meta3 = load_centerline_meta(stage3_dir)
    venous = load_venous_centerlines(stage3_dir, min_points=min_points, meta=meta3)

    s4_cl = Path(stage4_dir) if stage4_dir is not None else None
    if s4_cl is not None and centerlines_mask_path(s4_cl, from_segmentation=True).is_file():
        meta_cl = load_centerline_meta(s4_cl, from_segmentation=True)
        arterial = load_arterial_branches(
            s4_cl, min_points=min_points, meta=meta_cl, from_segmentation=True
        )
        s3_arterial = load_arterial_branches(
            stage3_dir, min_points=min_points, meta=meta3
        )
        ratio = float(truncate_length_ratio)
        for lid, branches3 in s3_arterial.items():
            pts3 = main_path_of(branches3)
            if pts3 is None:
                continue
            n3 = int(np.asarray(pts3).shape[0])
            if n3 < int(min_points):
                continue
            branches4 = arterial.get(int(lid))
            pts4 = main_path_of(branches4) if branches4 else None
            if pts4 is None:
                arterial[int(lid)] = branches3
                continue
            n4 = int(np.asarray(pts4).shape[0])
            if n4 < int(min_points) or (ratio > 0.0 and n4 < ratio * n3):
                arterial[int(lid)] = branches3
        return arterial, venous, meta_cl

    arterial = load_arterial_branches(stage3_dir, min_points=min_points, meta=meta3)
    return arterial, venous, meta3


__all__ = [
    "CENTERLINE_META_JSON",
    "CENTERLINE_SEG_META_JSON",
    "CENTERLINE_SEG_BRANCHES_JSON",
    "CENTERLINE_BRANCH_VTP_DIR",
    "CENTERLINES_MASK_NIFTI",
    "CENTERLINES_MASK_SEG_NIFTI",
    "arterial_main_paths",
    "centerline_meta_path",
    "centerlines_from_segmentation",
    "centerlines_mask_path",
    "export_branch_vtps",
    "export_centerlines_from_segmentation",
    "flatten_branches",
    "load_arterial_branches",
    "load_arterial_centerlines",
    "load_centerline_meta",
    "load_centerlines",
    "load_centerlines_branches",
    "load_venous_centerlines",
    "main_path_of",
    "parent_label_of",
    "rasterize_centerlines_mask",
    "smooth_centerline_polyline",
]
