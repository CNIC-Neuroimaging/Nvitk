"""
Ureter exclusion mask for CT-PET (PESA-Fat ct-pet-v5 conventions).

Builds a 3D ureter tube mask from registered CT + SUV, kidney/bladder masks,
minimum-cost path (skimage MCP), B-spline resampling, and EDT cylinder.

I/O: nvitk.io (not SimpleITK). Arrays: backend ``np`` / ``ndi`` after
``setup(globals())``; skimage/scipy.interpolate run on ``to_numpy`` slices
of the cost volume and path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(os.path.abspath('')).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import nvitk as nv
from nvitk.core import setup

from nvitk.transform import resample_mask_to_pet
from nvitk.segmentation.labels import get_label 

from nvitk.core import as_backend_array, to_numpy

setup(globals())

# splprep, splev = scipy.interpolate.splprep, scipy.interpolate.splev
from scipy.interpolate import splprep, splev
from skimage.graph import route_through_array

log = Logger()


# ---------------------------------------------------------------------------
# Anchors (heuristic)
# ---------------------------------------------------------------------------


def _convex_hull_slicewise_z(vol_uint8: np.ndarray) -> np.ndarray:
    """Same convention as stage2: hull each slice along the last axis."""
    try:
        from skimage.morphology import convex_hull_image
    except Exception:
        return vol_uint8.copy()
    vol_uint8 = to_numpy(vol_uint8)
    out = vol_uint8.astype(np.uint8, copy=True)
    for i in range(out.shape[-1]):
        sl = out[..., i]
        if sl.any():
            out[..., i] = convex_hull_image(sl)
    return as_backend_array(out)

def anchor_kidney_pelvis_concavity(
    kidney_mask: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    midline_x_index: float | None = None,
    medial_half_width_vox: float | None = None,
    structure: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Anchor in (hull − kidney) in each axial slice, 3D largest CC, centroid in mm.
    midline_x_index: voxel X index of sagittal midline; if set with medial_half_width_vox,
    keep only voxels with |x - midline| <= medial_half_width_vox (kidney side — pass
    per-side mask upstream instead if you already split L/R).
    """
    m = as_backend_array(kidney_mask).astype(bool)
    if not m.any():
        raise ValueError("empty kidney_mask")
    hull = _convex_hull_slicewise_z(m.astype(np.uint8)) > 0
    cavity = hull & ~m
    if midline_x_index is not None and medial_half_width_vox is not None:
        xs = np.arange(m.shape[0], dtype=np.float32)[:, None, None]
        cavity &= np.abs(xs - midline_x_index) <= medial_half_width_vox
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)
    lab, n = ndi.label(cavity, structure=structure)
    if n < 1:
        raise ValueError("no cavity voxels after hull − mask")
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    lid = int(sizes.argmax())
    coords = np.argwhere(lab == lid).astype(np.float64)
    return as_backend_array(coords.mean(axis=0))

def anchor_bladder_mid_posterior(
    bladder_mask: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    axis_z: int = -1,
    axis_ap: int = 1,
    z_trim_fraction: float = 0.25,
    posterior_q_low: float = 0.55,
    posterior_q_high: float = 0.90,
    structure: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Central Z slab (drop top/bottom fractions of occupied Z), then keep voxels whose
    A–P coordinate lies between posterior quantiles on axis_ap.

    axis_z: index of inferior–superior axis in the array (default last dim, like stage2 loop).
    axis_ap: index of the axis along which “posterior” is *increasing* index (verify on your data).
    """
    m = as_backend_array(bladder_mask).astype(bool)
    if not m.any():
        raise ValueError("empty bladder_mask")

    if axis_z < 0:
        axis_z = m.ndim + axis_z
    if axis_ap < 0:
        axis_ap = m.ndim + axis_ap

    z_idx = np.where(m.any(axis=tuple(i for i in range(m.ndim) if i != axis_z)))[0]
    if z_idx.size == 0:
        raise ValueError("no bladder extent along axis_z")

    z_min, z_max = int(z_idx.min()), int(z_idx.max())
    span = z_max - z_min + 1
    trim = max(1, int(span * z_trim_fraction))
    z_lo, z_hi = z_min + trim, z_max - trim
    if z_lo > z_hi:
        z_lo, z_hi = z_min, z_max

    band = np.zeros_like(m, dtype=bool)
    sl = [slice(None)] * m.ndim
    sl[axis_z] = slice(z_lo, z_hi + 1)
    band[tuple(sl)] = True
    cand = m & band
    if not cand.any():
        cand = m

    ap = np.where(cand)[axis_ap].astype(np.float64)
    lo = float(np.quantile(ap, posterior_q_low))
    hi = float(np.quantile(ap, posterior_q_high))
    if hi <= lo:
        lo, hi = float(ap.min()), float(ap.max())

    mask_ap = np.zeros_like(cand, dtype=bool)
    ap_full = np.arange(m.shape[axis_ap], dtype=np.float64)
    ap_sl = [slice(None)] * m.ndim
    ap_sl[axis_ap] = (ap_full >= lo) & (ap_full <= hi)
    mask_ap[tuple(ap_sl)] = True
    region = cand & mask_ap
    if not region.any():
        region = cand

    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)
    lab, n = ndi.label(region, structure=structure)
    if n < 1:
        raise ValueError("no voxels in bladder anchor region")

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    lid = int(sizes.argmax())
    coords = np.argwhere(lab == lid).astype(np.float64)
    return coords.mean(axis=0)


# ---------------------------------------------------------------------------
# Cost map
# ---------------------------------------------------------------------------


def build_cost_volume(
    suv: Any,
    hu: Any,
    p_end_zyx: Any,
    spacing_xyz_mm: Tuple[float, float, float],
    *,
    w_pet: float = 1.0,
    w_ct: float = 1.0,
    w_dist: float = 0.02,
    suv_clip: float = 15.0,
    eps: float = 1e-3,
    bone_hu: float = 250.0,
    air_hu: float = -500.0,
    penalty_high: float = 1e3,
    gaussian_sigma_vox: float = 1.0,
) -> Any:
    suv_c = np.clip(suv, 0.0, suv_clip)
    pet_term = w_pet / (suv_c + eps)

    pen = np.zeros_like(hu, dtype=np.float64)
    pen = np.where(hu > bone_hu, penalty_high, pen)
    pen = np.where(hu < air_hu, penalty_high, pen)
    pen = ndi.gaussian_filter(pen, sigma=gaussian_sigma_vox, mode="nearest")
    ct_term = w_ct * pen

    zz, yy, xx = np.indices(hu.shape)
    sx, sy, sz = spacing_xyz_mm
    pts = np.stack(
        [zz.astype(np.float64) * sz, yy.astype(np.float64) * sy, xx.astype(np.float64) * sx],
        axis=-1,
    )
    pe = np.array(p_end_zyx) * np.array([sz, sy, sx], dtype=np.float64)
    dist = np.sqrt(np.sum((pts - pe.reshape(1, 1, 1, 3)) ** 2, axis=-1))
    dist_term = w_dist * dist

    c = pet_term.astype(np.float64) + ct_term + dist_term
    c = np.clip(c, 1e-6, None)
    return c


# ---------------------------------------------------------------------------
# Path + spline + tube
# ---------------------------------------------------------------------------


def minimum_cost_path_zyx(cost: Any, start_zyx: Tuple[int, int, int], end_zyx: Tuple[int, int, int]) -> Any:
    from skimage.graph import route_through_array

    c_np = to_numpy(cost, copy=False)
    path, _ = route_through_array(c_np, start_zyx, end_zyx, fully_connected=True, geometric=True)
    return as_backend_array(path).astype(np.int32)


def spline_resample_zyx(
    path_zyx: Any,
    n_points: int,
    s_smooth: float,
    bounds_lo: Any,
    bounds_hi: Any,
) -> Any:
    from scipy.interpolate import splprep, splev

    p = to_numpy(path_zyx, copy=True)
    if p.shape[0] < 4:
        rep = 4 - p.shape[0]
        p = np.vstack([p, np.repeat(p[-1:], rep, axis=0)])

    pts = p.T
    tck, _u = splprep(pts, s=float(s_smooth), k=3)
    u_new = np.linspace(0, 1, int(n_points), dtype=np.float64)
    z, y, x = splev(u_new, tck)
    out = np.stack([z, y, x], axis=1)
    out = np.clip(out, bounds_lo, bounds_hi)
    return as_backend_array(out)


def line_mask_from_path(shape: Tuple[int, ...], path_zyx_float: Any) -> Any:
    mask = np.zeros(shape, dtype=np.uint8)
    pi = np.round(as_backend_array(path_zyx_float)).astype(np.int32)
    for i in range(pi.shape[0]):
        z, y, x = int(pi[i, 0]), int(pi[i, 1]), int(pi[i, 2])
        z = max(0, min(shape[0] - 1, z))
        y = max(0, min(shape[1] - 1, y))
        x = max(0, min(shape[2] - 1, x))
        mask[z, y, x] = 1
    return mask


def edt_tube_mm(line_mask: Any, spacing_xyz_mm: Tuple[float, float, float], radius_mm: float) -> Any:
    inv = as_backend_array(line_mask == 0)
    dist_mm = ndi.distance_transform_edt(inv, sampling=spacing_xyz_mm)
    return (dist_mm <= radius_mm).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_ureter_segmentation(
    ct: Image,
    pet_suv: Image,
    kidney_r: Image,
    kidney_l: Image,
    bladder: Image,
    *,
    radius_mm: float = 6.0,
    w_pet: float = 1.0,
    w_ct: float = 1.0,
    w_dist: float = 0.02,
    spline_s: float = 5.0,
) -> Any:
    """Assume CT, PET SUV, and masks share the same grid and ``axes`` (e.g. XYZ)."""
    hu = ct.data
    suv = pet_suv.data
    km_r = kidney_r.data
    km_l = kidney_l.data
    bm = bladder.data

    sp = ct.spacing or pet_suv.spacing
    if sp is None or len(sp) < 3:
        raise ValueError("CT (or PET) Image must carry spacing in metadata")
    spacing_xyz = (float(sp[0]), float(sp[1]), float(sp[2]))

    ureter_data = np.zeros_like(hu, dtype=np.uint8)
    for km in [km_r, km_l]:
        p_start = anchor_kidney_pelvis_concavity(km, spacing=spacing_xyz)
        p_end   = anchor_bladder_mid_posterior(bm, spacing=spacing_xyz)
        start   = tuple(int(round(float(x))) for x in as_backend_array(p_start))
        end     = tuple(int(round(float(x))) for x in as_backend_array(p_end))

        cost = build_cost_volume(suv, hu, p_start, spacing_xyz, w_pet=w_pet, w_ct=w_ct, w_dist=w_dist)
        path = minimum_cost_path_zyx(cost, start, end)

        n_pts = max(2 * int(path.shape[0]), 64)
        lo = np.array([0, 0, 0], dtype=np.float64)
        hi = np.array([hu.shape[0] - 1, hu.shape[1] - 1, hu.shape[2] - 1], dtype=np.float64)
        path_sp = spline_resample_zyx(path, n_pts, spline_s, lo, hi)

        line = line_mask_from_path(hu.shape, path_sp)
        _ureter_data = edt_tube_mm(line, spacing_xyz, radius_mm)

        ureter_data = np.where(_ureter_data > 0, _ureter_data, ureter_data)

    mask_ureter = ct.with_data(ureter_data.astype(np.uint8))

    return mask_ureter, path, path_sp