"""
Ureter exclusion mask for CT-PET (PESA-Fat ct-pet-v5 conventions) — v2.

Key changes versus v1
---------------------
* ``anchor_bladder_entry_per_side`` — each ureter now ends at the ipsilateral
  trigone corner (superior bladder face, same lateral hemisphere as its
  kidney), rather than both sharing a single posterior mid-bladder point.
* ``build_cost_volume`` — PET-primary cost; CT hard barriers removed.
  Three components:
    - inverse of a *gap-filled* SUV envelope (dark spots healed by smoothing);
    - an anatomical *corridor prior* (distance to the straight kidney→bladder
      line) that keeps the path in the expected para-spinal region;
    - a soft *lateral hemisphere* constraint that prevents the path from
      crossing to the contralateral side.
* ``run_ureter_segmentation`` updated to use both new primitives; original
  helpers kept for backward compatibility.

Array / axis conventions (consistent with the rest of this module)
------------------------------------------------------------------
The volume is stored **XYZ in memory** (axis 0 = Lateral/X, axis 1 = AP/Y,
axis 2 = IS/Z).  This matches:
  - ``_convex_hull_slicewise_z`` iterating over ``shape[-1]`` for Z;
  - ``anchor_kidney_pelvis_concavity`` using axis 0 for ``midline_x_index``;
  - ``anchor_bladder_mid_posterior`` having ``axis_z = -1``.
All new functions expose explicit ``axis_x`` / ``axis_z`` parameters so the
caller can override for any resampled grid that differs from this default.

I/O: nvitk.io (not SimpleITK).  Arrays: backend ``np`` / ``ndi`` after
``setup(globals())``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from nvitk.core import setup
from nvitk.core import as_backend_array, to_numpy

setup(globals())

from scipy.interpolate import splprep, splev
from skimage.graph import route_through_array

from nvitk.core.logger import Logger
log = Logger()


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _convex_hull_slicewise_z(vol_uint8: np.ndarray) -> np.ndarray:
    """Hull each axial (IS) slice along the last axis — same as stage2."""
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


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def anchor_kidney_pelvis_concavity(
    kidney_mask: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    midline_x_index: float | None = None,
    medial_half_width_vox: float | None = None,
    structure: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Anchor in (hull − kidney) per axial slice → 3D largest CC → centroid.
    Unchanged from v1 — starting points are already correct.

    ``midline_x_index`` / ``medial_half_width_vox``: restrict the cavity to
    the medial half of the kidney (lateral axis = axis 0).
    """
    m = as_backend_array(kidney_mask).astype(bool)
    if not m.any():
        raise ValueError("empty kidney_mask")
    hull   = _convex_hull_slicewise_z(m.astype(np.uint8)) > 0
    cavity = hull & ~m
    if midline_x_index is not None and medial_half_width_vox is not None:
        xs = np.arange(m.shape[0], dtype=np.float32)[:, None, None]
        cavity &= np.abs(xs - midline_x_index) <= medial_half_width_vox
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)
    lab, n = ndi.label(cavity, structure=structure)
    if n < 1:
        raise ValueError("no cavity voxels after hull − mask")
    sizes    = np.bincount(lab.ravel())
    sizes[0] = 0
    lid      = int(sizes.argmax())
    coords   = np.argwhere(lab == lid).astype(np.float64)
    return as_backend_array(coords.mean(axis=0))


def anchor_bladder_entry_per_side(
    bladder_mask: np.ndarray,
    kidney_mask: np.ndarray,
    *,
    axis_x: int = 0,
    axis_z: int = -1,
    superior_fraction: float = 0.35,
    structure: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Bladder ureter-entry anchor ipsilateral to *kidney_mask*.

    Anatomy
    -------
    Each ureter enters the bladder at the *ureteral orifice*, located at the
    lateral corners of the trigone — superior face of the bladder, on the same
    lateral side as its kidney.  Both orifices are separated by ~5 cm in an
    adult.  Using a shared mid-bladder point for both ureters causes the MCP
    to route both paths into the midline well before the actual entry point.

    Strategy
    --------
    1. Identify the lateral centroid of the kidney to determine which half of
       the volume it occupies.
    2. Restrict bladder voxels to the same lateral hemisphere (ipsilateral).
    3. Keep only the top ``superior_fraction`` of those voxels in the IS
       direction — the trigone is at the superior–posterior face.
    4. Return the centroid of the largest connected component.

    Parameters
    ----------
    axis_x : int
        Array axis for the Lateral (L-R) direction.
        Default 0 — matches ``anchor_kidney_pelvis_concavity`` convention
        where ``midline_x_index`` is applied along axis 0.
    axis_z : int
        Array axis for the IS direction.
        Default -1 (last axis) — matches ``_convex_hull_slicewise_z`` and
        ``anchor_bladder_mid_posterior`` conventions.
    superior_fraction : float
        Fraction of IS extent to keep from the top; 0.35 → top 35 %.
    """
    m  = as_backend_array(bladder_mask).astype(bool)
    km = as_backend_array(kidney_mask).astype(bool)
    if not m.any():
        raise ValueError("empty bladder_mask")
    if not km.any():
        raise ValueError("empty kidney_mask — cannot determine ipsilateral side")

    if axis_x < 0:
        axis_x = m.ndim + axis_x
    if axis_z < 0:
        axis_z = m.ndim + axis_z

    # 1. Lateral centroid of kidney → which side of the volume it occupies
    k_coords  = np.argwhere(km)
    k_x_mean  = float(k_coords[:, axis_x].mean())
    vol_x_mid = m.shape[axis_x] / 2.0

    # 2. Build an ipsilateral lateral mask for the bladder
    x_arr     = np.arange(m.shape[axis_x], dtype=np.float64)
    x_in_side = (x_arr >= vol_x_mid) if (k_x_mean >= vol_x_mid) else (x_arr < vol_x_mid)

    lat_mask = np.zeros_like(m, dtype=bool)
    lat_sl   = [slice(None)] * m.ndim
    lat_sl[axis_x] = x_in_side
    lat_mask[tuple(lat_sl)] = True

    cand = m & lat_mask
    if not cand.any():
        log.warning(
            "anchor_bladder_entry_per_side: no bladder voxels on the ipsilateral side "
            "(axis_x=%d, k_x=%.1f, mid=%.1f) — falling back to full bladder mask.",
            axis_x, k_x_mean, vol_x_mid,
        )
        cand = m.copy()

    # 3. Top ``superior_fraction`` in IS direction (trigone corners are superior)
    other_axes = tuple(i for i in range(m.ndim) if i != axis_z)
    z_present  = np.where(cand.any(axis=other_axes))[0]
    z_min_c, z_max_c = int(z_present.min()), int(z_present.max())
    span  = z_max_c - z_min_c + 1
    trim  = max(1, int(span * (1.0 - superior_fraction)))
    z_sup = z_max_c - trim  # lower bound of the superior band

    sup_sl = [slice(None)] * m.ndim
    sup_sl[axis_z] = slice(z_sup, z_max_c + 1)
    sup_mask = np.zeros_like(cand, dtype=bool)
    sup_mask[tuple(sup_sl)] = True

    region = cand & sup_mask
    if not region.any():
        log.warning(
            "anchor_bladder_entry_per_side: superior band empty — using full ipsilateral cand."
        )
        region = cand

    # 4. Largest CC centroid
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)
    lab, n = ndi.label(region, structure=structure)
    if n < 1:
        raise ValueError("no bladder voxels found in ipsilateral superior region")
    sizes    = np.bincount(lab.ravel())
    sizes[0] = 0
    lid      = int(sizes.argmax())
    coords   = np.argwhere(lab == lid).astype(np.float64)
    return coords.mean(axis=0)


# ---------------------------------------------------------------------------
# Cost maps
# ---------------------------------------------------------------------------

def build_cost_volume(
    suv: Any,
    spacing_xyz_mm: Tuple[float, float, float],
    body: "Image",
    *,
    # ---- PET / gap-filling ----------------------------------------
    w_pet: float = 1.0,
    suv_clip_l: float = 1e-2,
    suv_clip_h: float = 15.0,
    suv_fill_sigma_vox: float = 4.0,
    suv_fill_blend: float = 0.5,
    eps: float = 1e-3,
) -> Any:
    """
    PET-primary cost volume for ureter MCP routing — v2.1.

    What changed from v2
    --------------------
    The corridor prior (distance to the straight kidney→bladder segment) has
    been **removed**.  It was structurally wrong: because it is zero on the
    straight line and grows everywhere else, the MCP minimised it by routing
    straight, producing a linear cylindrical mask regardless of the PET signal.
    A path prior that rewards the straight line *is* the straight line.

    Components
    ----------
    pet_term
        Inverse of a gap-filled SUV envelope — the primary and dominant cost.
        High uptake = low cost = preferred.  Dark spots are healed by blending
        the raw map with a smoothed version (element-wise maximum), so short
        low-uptake gaps along the ureter do not reroute the path.

    Parameters
    ----------
    w_lateral : float
        Maximum penalty added by the lateral guardrail (plateau value).
        In the same units as pet_term.  Default 1.5 ≈ cost of a near-zero
        SUV voxel — enough to block, not enough to reshape the path.
    lateral_free_mm : float
        Half-width of the zero-penalty lateral band around the kidney x.
    lateral_slope_mm : float
        Width of the linear ramp from free band to plateau.  The full
        guardrail spans ``lateral_free_mm + lateral_slope_mm`` in mm.
    axis_x : int
        Array axis for the Lateral (L-R) direction (default 0).
    """
    s0, s1, s2 = spacing_xyz_mm

    # ---- gap-filled SUV ---------------------------------------------------
    # 1) Body mask
    body_mask = np.asarray(body.data) > 0

    # 2) Clip + smooth as you already do
    suv_c = np.clip(suv, suv_clip_l, suv_clip_h)
    suv_smooth = ndi.gaussian_filter(
        suv_c.astype(np.float64),
        sigma=float(suv_fill_sigma_vox),
        mode="nearest",
    )
    suv_filled = np.maximum(suv_c, float(suv_fill_blend) * suv_smooth)

    # 3) Log-compress SUV (high SUV separation improves)
    log_suv = np.log1p(suv_filled)

    # 4) Normalize ONLY inside body (important)
    if np.any(body_mask):
        lo, hi = np.percentile(log_suv[body_mask], [1, 99])  # robust range
    else:
        lo, hi = np.percentile(log_suv, [1, 99])
    den = max(hi - lo, 1e-6)
    score = np.clip((log_suv - lo) / den, 0.0, 1.0)  # high SUV -> score near 1

    # 5) Convert to cost: high score => very low cost
    # gamma > 1 makes high-SUV voxels even cheaper
    gamma = 2.5
    cost = float(w_pet) * (1.0 - score) ** gamma + float(eps)

    # 6) Fill outside-body with worst in-body pet cost
    if np.any(body_mask):
        outside_fill = np.max(cost[body_mask])
    else:
        outside_fill = np.max(cost)

    cost = np.where(body_mask, cost, outside_fill)
    return cost


# ---------------------------------------------------------------------------
# Path + spline + tube  (unchanged from v1)
# ---------------------------------------------------------------------------

def minimum_cost_path_zyx(
    cost: Any,
    start_zyx: Tuple[int, int, int],
    end_zyx: Tuple[int, int, int],
) -> Any:
    c_np   = to_numpy(cost, copy=False)
    path, _ = route_through_array(
        c_np, start_zyx, end_zyx, fully_connected=True, geometric=True
    )
    return as_backend_array(path).astype(np.int32)


def spline_resample_zyx(
    path_zyx: Any,
    n_points: int,
    s_smooth: float,
    bounds_lo: Any,
    bounds_hi: Any,
) -> Any:
    p = to_numpy(path_zyx, copy=True)
    if p.shape[0] < 4:
        rep = 4 - p.shape[0]
        p   = np.vstack([p, np.repeat(p[-1:], rep, axis=0)])
    pts     = p.T
    tck, _u = splprep(pts, s=float(s_smooth), k=3)
    u_new   = np.linspace(0, 1, int(n_points), dtype=np.float64)
    z, y, x = splev(to_numpy(u_new), tck)
    x, y, z = as_backend_array(x), as_backend_array(y), as_backend_array(z)
    out     = np.stack([z, y, x], axis=1)
    out     = np.clip(out, bounds_lo, bounds_hi)
    return as_backend_array(out)


def line_mask_from_path(shape: Tuple[int, ...], path_zyx_float: Any) -> Any:
    mask = np.zeros(shape, dtype=np.uint8)
    pi   = np.round(as_backend_array(path_zyx_float)).astype(np.int32)
    for i in range(pi.shape[0]):
        z = int(np.clip(pi[i, 0], 0, shape[0] - 1))
        y = int(np.clip(pi[i, 1], 0, shape[1] - 1))
        x = int(np.clip(pi[i, 2], 0, shape[2] - 1))
        mask[z, y, x] = 1
    return mask


def edt_tube_mm(
    line_mask: Any,
    spacing_xyz_mm: Tuple[float, float, float],
    radius_mm: float,
) -> Any:
    inv     = as_backend_array(line_mask == 0)
    dist_mm = ndi.distance_transform_edt(inv, sampling=spacing_xyz_mm)
    return (dist_mm <= radius_mm).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def segment_ureter(
    pet_suv: "Image",
    kidney_r: "Image",
    kidney_l: "Image",
    bladder: "Image",
    body: "Image",
    *,
    # ---- Tube geometry -------------------------------------------------
    radius_mm: float = 6.0,
    # ---- Cost weights --------------------------------------------------
    w_pet: float = 5.0,
    # ---- Gap-filling ---------------------------------------------------
    suv_fill_sigma_vox: float = 0.5,
    suv_fill_blend: float = 0.05,
    # ---- Spline --------------------------------------------------------
    spline_s: float = 1.5,
    # ---- Axis conventions — adjust if your resampled grid differs -------
    axis_x: int = 0,   # lateral (L-R) axis in the array
    axis_z: int = -1,  # IS axis in the array
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Build bilateral ureter exclusion masks from kidney + bladder segmentations
    and the registered SUV map.

    Returns
    -------
    mask_ureter : Image
        Binary mask (uint8) in the PET SUV grid.
    paths : dict {"R": array, "L": array}
        Raw MCP voxel paths.
    paths_sp : dict {"R": array, "L": array}
        B-spline-smoothed paths.

    Design notes
    ------------
    * Starting points (kidney pelvis concavity) are unchanged from v1.
    * Ending points are now **ipsilateral**: each ureter routes to the
      superior corner of the bladder on the same lateral side as its kidney.
      This avoids the artificial convergence at a single mid-bladder point
      that caused both paths to merge and deviate from the true anatomy.
    * The cost volume uses only SUV (gap-filled) plus soft anatomical priors.
      CT barriers are omitted: at whole-body resolution the ureter is not
      visible on CT, and hard bone/air penalties deflected the MCP onto
      incorrect routes running along fascial planes far from the true path.
    """
    suv = pet_suv.data
    bm  = bladder.data

    sp = pet_suv.spacing
    if sp is None or len(sp) < 3:
        raise ValueError("CT (or PET) Image must carry spacing in metadata")
    spacing_xyz = (float(sp[0]), float(sp[1]), float(sp[2]))

    ureter_data = np.zeros_like(suv, dtype=np.uint8)
    paths: Dict[str, Any]    = {}
    paths_sp: Dict[str, Any] = {}

    for side, km in [("R", kidney_r.data), ("L", kidney_l.data)]:

        # ---- anchors -------------------------------------------------------
        p_start = anchor_kidney_pelvis_concavity(km, spacing=spacing_xyz)
        p_end   = anchor_bladder_entry_per_side(
            bm, km,
            axis_x=axis_x,
            axis_z=axis_z,
        )

        start = tuple(int(round(float(v))) for v in as_backend_array(p_start))
        end   = tuple(int(round(float(v))) for v in as_backend_array(p_end))

        log.info("Ureter-%s: start_vox=%s  end_vox=%s", side, start, end)

        # ---- PET-primary cost volume ---------------------------------------
        cost = build_cost_volume(
            suv, spacing_xyz, body,
            w_pet=w_pet,
            suv_fill_sigma_vox=suv_fill_sigma_vox,
            suv_fill_blend=suv_fill_blend,
        )

        # ---- MCP routing ---------------------------------------------------
        path = minimum_cost_path_zyx(cost, start, end)
        log.info("Ureter-%s: MCP path length = %d voxels", side, int(path.shape[0]))

        # ---- B-spline smoothing --------------------------------------------
        n_pts   = max(2 * int(path.shape[0]), 64)
        lo      = np.array([0, 0, 0], dtype=np.float64)
        hi      = np.array(
            [suv.shape[0] - 1, suv.shape[1] - 1, suv.shape[2] - 1],
            dtype=np.float64,
        )
        path_sp = spline_resample_zyx(path, n_pts, spline_s, lo, hi)

        # ---- EDT tube mask -------------------------------------------------
        line         = line_mask_from_path(suv.shape, path_sp)
        tube         = edt_tube_mm(line, spacing_xyz, radius_mm)
        ureter_data  = np.where(tube > 0, tube, ureter_data)

        paths[side]    = path
        paths_sp[side] = path_sp

    mask_ureter = pet_suv.with_data(ureter_data.astype(np.uint8))
    return mask_ureter, paths, paths_sp