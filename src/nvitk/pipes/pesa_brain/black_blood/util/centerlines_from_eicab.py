"""eICAB CW mask → BB relabel, optional warp, centerlines, rasterized mask."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import as_backend_array
from nvitk.io.imageio import imread, imsave
from nvitk.morphology.centerline import compute_centerlines
from nvitk.pipes.pesa_brain.black_blood.labels import (
    BB_ARTERIAL_LABEL_IDS,
    bb_vessel_name,
    relabel_eicab_to_bb,
)
from nvitk.registration.fsl.flirt import flirt_apply_rigid

CENTERLINES_MASK_NIFTI = "centerlines_mask.nii.gz"
CENTERLINE_META_JSON = "centerline_meta.json"
EICAB_BB_IN_TOF_NIFTI = "eicab_bb_in_tof.nii.gz"


@dataclass(frozen=True)
class CenterlineArtifacts:
    centerlines: dict[int, np.ndarray]
    centerlines_mask_path: Path
    centerline_meta_path: Path
    eicab_bb_path: Path


def rasterize_centerlines_mask(
    shape: tuple[int, int, int],
    centerlines: dict[int, np.ndarray],
) -> np.ndarray:
    """Voxel mask with arterial label id on centerline points."""
    mask = np.zeros(shape, dtype=np.int32)
    for vid, pts in sorted(centerlines.items()):
        p = as_backend_array(pts)
        for row in p:
            i, j, k = (
                int(round(float(row[0]))),
                int(round(float(row[1]))),
                int(round(float(row[2]))),
            )
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                mask[i, j, k] = int(vid)
    return mask


def _grids_match(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> bool:
    return tuple(int(x) for x in a_shape[:3]) == tuple(int(x) for x in b_shape[:3])


def build_centerlines_from_eicab(
    eicab_cw_path: Path,
    tof_ref_path: Path,
    out_dir: Path,
    *,
    transform_mat: Path | None = None,
    min_points: int = 5,
    skip_existing: bool = False,
) -> CenterlineArtifacts:
    """Relabel eICAB CW, warp to TOF grid if needed, compute and write centerlines."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / CENTERLINES_MASK_NIFTI
    meta_path = out_dir / CENTERLINE_META_JSON
    bb_path = out_dir / EICAB_BB_IN_TOF_NIFTI

    if skip_existing and mask_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cl_img = imread(mask_path)
        clm = as_backend_array(cl_img.data).astype(np.int32, copy=False)
        centerlines = compute_centerlines(
            clm,
            centerline_mask=clm > 0,
            labels=sorted(int(v) for v in np.unique(clm) if int(v) > 0),
            min_points=min_points,
        )
        return CenterlineArtifacts(
            centerlines=centerlines,
            centerlines_mask_path=mask_path,
            centerline_meta_path=meta_path,
            eicab_bb_path=bb_path if bb_path.is_file() else bb_path,
        )

    ref_img = imread(tof_ref_path)
    ref_shape = tuple(int(x) for x in ref_img.data.shape[:3])

    labels_native = as_backend_array(imread(eicab_cw_path).data).astype(np.int32, copy=False)
    warped_path = out_dir / "_eicab_warped_tmp.nii.gz"
    if _grids_match(labels_native.shape, ref_shape):
        labels_in_ref = labels_native
        warped = False
    else:
        if transform_mat is None or not Path(transform_mat).is_file():
            raise FileNotFoundError(
                "eICAB mask grid differs from TOF_resampled; stage1 matrix required."
            )
        flirt_apply_rigid(
            eicab_cw_path,
            tof_ref_path,
            transform_mat,
            warped_path,
            interp="nearestneighbour",
        )
        labels_in_ref = np.rint(
            as_backend_array(imread(warped_path).data)
        ).astype(np.int32, copy=False)
        warped = True

    bb_labels = relabel_eicab_to_bb(labels_in_ref)
    imsave(bb_path, bb_labels, metadata=dict(ref_img.metadata or {}))

    centerlines = compute_centerlines(
        bb_labels,
        labels=sorted(BB_ARTERIAL_LABEL_IDS),
        min_points=min_points,
    )
    cl_mask = rasterize_centerlines_mask(ref_shape, centerlines)
    imsave(mask_path, cl_mask, metadata=dict(ref_img.metadata or {}))

    meta: dict[str, Any] = {
        "eicab_cw": str(eicab_cw_path),
        "tof_reference": str(tof_ref_path),
        "warped_eicab": warped,
        "transform_matrix": str(transform_mat) if transform_mat else None,
        "labels": {
            str(lid): {
                "name": bb_vessel_name(lid),
                "n_points": int(centerlines[lid].shape[0]) if lid in centerlines else 0,
            }
            for lid in sorted(BB_ARTERIAL_LABEL_IDS)
            if lid in centerlines
        },
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if warped and warped_path.is_file():
        warped_path.unlink(missing_ok=True)

    return CenterlineArtifacts(
        centerlines=centerlines,
        centerlines_mask_path=mask_path,
        centerline_meta_path=meta_path,
        eicab_bb_path=bb_path,
    )
