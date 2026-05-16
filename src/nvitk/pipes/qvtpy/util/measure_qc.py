"""QC figures for stage-6 LOC cross-section measurements."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.measure.cross_section import segment_at_point
from nvitk.morphology.centerline import centerline_tangents
from nvitk.transform.oblique import oblique_slice
from nvitk.core.backend import setup

setup(globals())

def _project_cl_to_plane(
    pts: np.ndarray,
    center_xyz: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map 3D centerline points to in-plane row/col indices (for scatter)."""
    rel = as_backend_array(pts) - as_backend_array(center_xyz).reshape(1, 3)
    pu = rel @ u.reshape(3)
    pv = rel @ v.reshape(3)
    return pu, pv


def save_loc_cross_section_qc_png(
    out_path: Path,
    *,
    cd: np.ndarray,
    centerline_pts: np.ndarray,
    loc_index: int,
    center_xyz: np.ndarray,
    tangent: np.ndarray,
    vessel_name: str,
    segment_id: int,
    loc_role: str,
    voxel_spacing: tuple[float, float, float],
    radius_vox: float,
    cross_section_res: int = 0,
    plane_interp_order: int = 1,
) -> None:
    """Save CD oblique slice with segmentation mask and centerline context."""
    pts = to_numpy(centerline_pts).astype(np.float64)
    if pts.shape[0] < 1:
        return
    # loc_index = int(np.clip(loc_index, 0, pts.shape[0] - 1))
    loc_index = int(max(0, min(loc_index, pts.shape[0] - 1)))
    tangents = centerline_tangents(pts, k_half=2)
    tang = tangents[loc_index] if tangent is None else tangent

    xs = segment_at_point(
        center_xyz,
        tang,
        mag=cd,
        cd=cd,
        vel_mag=cd,
        voxel_spacing=voxel_spacing,
        radius_vox=radius_vox,
        cross_section_res=cross_section_res,
        plane_interp_order=plane_interp_order,
    )
    cd_sl = oblique_slice(
        cd,
        center_xyz=xs.center_xyz,
        u_xyz=xs.u,
        v_xyz=xs.v,
        radius_vox=radius_vox,
        res=xs.plane_res,
        order=int(plane_interp_order),
    )
    cd_np = as_backend_array(cd_sl).astype(np.float64)
    mask = xs.mask_2d.astype(bool)

    pu, pv = _project_cl_to_plane(pts, xs.center_xyz, xs.u, xs.v)
    loc_pu, loc_pv = float(pu[loc_index]), float(pv[loc_index])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(to_numpy(cd_np), cmap="gray", origin="lower")
    ax.contour(to_numpy(mask), levels=[0.5], colors=["cyan"], linewidths=1.0)
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask, 2] = 1.0
    rgba[mask, 3] = 0.25
    ax.imshow(to_numpy(rgba), origin="lower")

    i0 = max(0, loc_index - 2)
    i1 = min(pts.shape[0], loc_index + 3)
    ax.scatter(to_numpy(pu[i0:i1]), to_numpy(pv[i0:i1]), c="yellow", s=18, label="CL ±2")
    ax.scatter(to_numpy(loc_pu), to_numpy(loc_pv), c="red", s=60, marker="x", label="LOC")
    ax.set_title(
        f"{vessel_name} seg={segment_id} role={loc_role} idx={loc_index} "
        f"area={xs.area_mm2:.2f} mm² res={xs.plane_res}"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


__all__ = ["save_loc_cross_section_qc_png"]
