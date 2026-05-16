"""QC figures for stage-6 LOC cross-section measurements."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.measure.cross_section import ThrAlgorithm, cross_section_at_loc
from nvitk.morphology.centerline import centerline_tangents
from nvitk.transform.oblique import oblique_slice

setup(globals())


def _plot_cross_section_panel(
    ax: plt.Axes,
    *,
    cd_sl: np.ndarray,
    mask: np.ndarray,
    title: str,
    is_loc: bool,
) -> None:
    cd_np = as_backend_array(cd_sl).astype(np.float64)
    m = as_backend_array(mask).astype(bool)
    ax.imshow(to_numpy(cd_np), cmap="gray", origin="lower")
    if np.any(m):
        ax.contour(to_numpy(m), levels=[0.5], colors=["cyan"], linewidths=1.0)
        rgba = np.zeros((*m.shape, 4))
        rgba[m, 2] = 1.0
        rgba[m, 3] = 0.25
        ax.imshow(to_numpy(rgba), origin="lower")
    if is_loc:
        ax.plot(
            [to_numpy(m).shape[1] / 2.0],
            [to_numpy(m).shape[0] / 2.0],
            marker="x",
            color="red",
            markersize=10,
            markeredgewidth=2,
        )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def save_loc_cross_section_qc_png(
    out_path: Path,
    *,
    cd: np.ndarray,
    mag: np.ndarray,
    vel_mag: np.ndarray,
    centerline_pts: np.ndarray,
    loc_index: int,
    vessel_name: str,
    segment_id: int,
    loc_role: str,
    voxel_spacing: tuple[float, float, float],
    radius_vox: float,
    cross_section_res: int = 0,
    plane_interp_order: int = 1,
    measure_resegment: bool = True,
    thr_algorithm: ThrAlgorithm = "lsthr",
    volume_seg: np.ndarray | None = None,
    volume_label_id: int = 0,
) -> None:
    """Five-panel row: CL indices loc-2 … loc+2 with CD and mask contour each."""
    pts = as_backend_array(centerline_pts).astype(np.float64)
    if pts.shape[0] < 1:
        return
    loc_index = int(max(0, min(loc_index, pts.shape[0] - 1)))
    tangents = centerline_tangents(pts, k_half=2)

    offsets = (-2, -1, 0, 1, 2)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    loc_xs = None

    for ax, offset in zip(axes, offsets):
        idx = int(max(0, min(loc_index + offset, pts.shape[0] - 1)))
        center = pts[idx]
        tang = tangents[idx]
        xs = cross_section_at_loc(
            center,
            tang,
            mag=mag,
            cd=cd,
            vel_mag=vel_mag,
            voxel_spacing=voxel_spacing,
            radius_vox=radius_vox,
            cross_section_res=cross_section_res,
            plane_interp_order=plane_interp_order,
            measure_resegment=measure_resegment,
            thr_algorithm=thr_algorithm,
            volume_seg=volume_seg,
            volume_label_id=volume_label_id,
        )
        if offset == 0:
            loc_xs = xs
        cd_sl = oblique_slice(
            cd,
            center_xyz=xs.center_xyz,
            u_xyz=xs.u,
            v_xyz=xs.v,
            radius_vox=radius_vox,
            res=xs.plane_res,
            order=int(plane_interp_order),
        )
        role = "LOC" if offset == 0 else f"idx {idx}"
        title = f"{role}\narea={xs.area_mm2:.2f} mm²"
        _plot_cross_section_panel(
            ax,
            cd_sl=cd_sl,
            mask=xs.mask_2d,
            title=title,
            is_loc=(offset == 0),
        )

    assert loc_xs is not None
    fig.suptitle(
        f"{vessel_name} seg={segment_id} role={loc_role} "
        f"loc_idx={loc_index} res={loc_xs.plane_res} thr={thr_algorithm}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


__all__ = ["save_loc_cross_section_qc_png"]
