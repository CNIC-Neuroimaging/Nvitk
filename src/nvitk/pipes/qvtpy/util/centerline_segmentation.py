"""Assemble 3D vessel segmentation from cross-sectional masks along centerlines."""

from __future__ import annotations

from typing import Literal

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.pipes.qvtpy.util.cross_section import CrossSectionResult, segment_along_polyline
from nvitk.pipes.qvtpy.util.mask_cleaning import clean_multilabel_islands

AssemblyMode = Literal["voxel", "mesh"]


def _stamp_plane_mask(
    vol: np.ndarray,
    result: CrossSectionResult,
    label: int,
    *,
    radius_vox: float,
) -> None:
    """Write 2D mask voxels into 3D *vol* at oblique plane coordinates."""
    from nvitk.transform.oblique import oblique_slice

    mask = result.mask_2d.astype(bool, copy=False)
    if not np.any(mask):
        return
    res = int(mask.shape[0])
    lin = np.linspace(-float(radius_vox), float(radius_vox), res)
    yy, xx = np.meshgrid(lin, lin, indexing="ij")
    center = result.center_xyz
    u, v = result.u, result.v
    for iy in range(res):
        for ix in range(res):
            if not mask[iy, ix]:
                continue
            off = np.array([xx[iy, ix], yy[iy, ix], 0.0], dtype=np.float64)
            xyz = center + off[0] * u + off[1] * v
            i, j, k = int(np.round(xyz[0])), int(np.round(xyz[1])), int(np.round(xyz[2]))
            if 0 <= i < vol.shape[0] and 0 <= j < vol.shape[1] and 0 <= k < vol.shape[2]:
                vol[i, j, k] = int(label)


def _interpolate_stations(
    results: list[CrossSectionResult],
    label: int,
    vol: np.ndarray,
    *,
    interp_level: int,
    radius_vox: float,
) -> None:
    if interp_level <= 0 or len(results) < 2:
        for r in results:
            _stamp_plane_mask(vol, r, label, radius_vox=radius_vox)
        return
    n_blend = int(interp_level)
    for a, b in zip(results[:-1], results[1:]):
        _stamp_plane_mask(vol, a, label, radius_vox=radius_vox)
        for t in range(1, n_blend + 1):
            alpha = float(t) / float(n_blend + 1)
            center = (1.0 - alpha) * a.center_xyz + alpha * b.center_xyz
            tangent = (1.0 - alpha) * a.tangent + alpha * b.tangent
            # Reuse mask shape from a; blend masks linearly for stamping
            blend_mask = ((1.0 - alpha) * a.mask_2d.astype(np.float32) + alpha * b.mask_2d.astype(np.float32)) > 0.5
            blended = CrossSectionResult(
                mask_2d=blend_mask,
                area_mm2=(1.0 - alpha) * a.area_mm2 + alpha * b.area_mm2,
                circularity=(1.0 - alpha) * a.circularity + alpha * b.circularity,
                center_xyz=center,
                tangent=tangent,
                u=(1.0 - alpha) * a.u + alpha * b.u,
                v=(1.0 - alpha) * a.v + alpha * b.v,
                pixel_spacing_mm=a.pixel_spacing_mm,
                plane_res=a.plane_res,
            )
            _stamp_plane_mask(vol, blended, label, radius_vox=radius_vox)
    _stamp_plane_mask(vol, results[-1], label, radius_vox=radius_vox)


def assemble_voxel_segmentation(
    shape: tuple[int, int, int],
    vessels: dict[int | str, np.ndarray],
    *,
    mag: np.ndarray,
    cd: np.ndarray,
    vel_mag: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    label_for_key: dict[int | str, int],
    stride: int = 1,
    radius_vox: float = 10.0,
    seg_interp_level: int = 0,
    cross_section_res: int = 0,
    plane_interp_order: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build multilabel volume by stamping cross-sections along each polyline."""
    vol = np.zeros(shape, dtype=np.int32)
    stats: dict[str, float] = {}
    for key, pts in vessels.items():
        lid = int(label_for_key[key])
        results = segment_along_polyline(
            pts,
            mag=mag,
            cd=cd,
            vel_mag=vel_mag,
            voxel_spacing=voxel_spacing,
            stride=stride,
            radius_vox=radius_vox,
            cross_section_res=cross_section_res,
            plane_interp_order=plane_interp_order,
        )
        if not results:
            continue
        _interpolate_stations(results, lid, vol, interp_level=seg_interp_level, radius_vox=radius_vox)
        areas = [r.area_mm2 for r in results]
        stats[f"mean_area_mm2_{key}"] = float(np.mean(areas)) if areas else 0.0
        stats[f"n_stations_{key}"] = float(len(results))
    return vol, stats


def assemble_mesh_segmentation(
    shape: tuple[int, int, int],
    vessels: dict[int | str, np.ndarray],
    *,
    mag: np.ndarray,
    cd: np.ndarray,
    vel_mag: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    label_for_key: dict[int | str, int],
    stride: int = 2,
    radius_vox: float = 10.0,
    cross_section_res: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Mesh assembly via distance transform from stamped sparse points (VTK-free fallback)."""
    from scipy import ndimage as scipy_ndi

    sparse = np.zeros(shape, dtype=np.uint8)
    vol, stats = assemble_voxel_segmentation(
        shape,
        vessels,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        label_for_key=label_for_key,
        stride=max(1, int(stride)),
        radius_vox=radius_vox,
        seg_interp_level=0,
        cross_section_res=cross_section_res,
    )
    for lid in sorted(set(label_for_key.values())):
        roi = vol == lid
        if not np.any(roi):
            continue
        dist = scipy_ndi.distance_transform_edt(~roi)
        # Fill within ~radius_vox of stamped voxels
        fill = dist <= float(radius_vox) * 1.2
        sparse[fill] = np.maximum(sparse[fill], np.uint8(lid))
    out = sparse.astype(np.int32, copy=False)
    return out, stats


def postprocess_segmentation(
    seg: np.ndarray,
    *,
    min_fraction: float = 0.005,
) -> np.ndarray:
    return to_numpy(clean_multilabel_islands(seg, min_fraction=min_fraction))


__all__ = [
    "assemble_mesh_segmentation",
    "assemble_voxel_segmentation",
    "postprocess_segmentation",
]
