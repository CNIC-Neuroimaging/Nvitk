"""Shared LOC-wise hemodynamic measurement (stage-6 core)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.measure.cross_section import (
    ThrAlgorithm,
    cross_section_at_loc,
    flow_series_ml_s,
    masked_plane_velocity_series,
)
from nvitk.measure.hemodynamics import (
    mean_velocity_mm_s,
    pulsatility_index,
    resistivity_index,
)
from nvitk.pipes.qvtpy.labels import qvtpy_vessel_name

setup(globals())


def measure_loc_row(
    row: dict[str, Any],
    *,
    mag: np.ndarray,
    cd: np.ndarray,
    vel_mag: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    cross_section_radius_vox: float = 10.0,
    measure_resegment: bool = False,
    thr_algorithm: ThrAlgorithm = "lsthr",
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
    cs_supersampling: bool = False,
    volume_seg: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """PI/RI and time series for one LOC row (stage-5 ``locs.csv`` format)."""
    vid = int(row["vessel_id"])
    vname = (row.get("vessel_name") or "").strip() or qvtpy_vessel_name(vid)
    center = np.array(
        [
            float(row["centerline_x"]),
            float(row["centerline_y"]),
            float(row["centerline_z"]),
        ],
        dtype=np.float64,
    )
    tang = np.array(
        [float(row["tangent_x"]), float(row["tangent_y"]), float(row["tangent_z"])],
        dtype=np.float64,
    )
    xs = cross_section_at_loc(
        center,
        tang,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        radius_vox=cross_section_radius_vox,
        cross_section_res=cross_section_res,
        plane_interp_order=int(cross_section_plane_interp),
        cs_supersampling=cs_supersampling,
        measure_resegment=measure_resegment,
        thr_algorithm=thr_algorithm,
        volume_seg=volume_seg,
        volume_label_id=vid,
    )
    area_mm2 = float(xs.area_mm2)
    vel_ts = masked_plane_velocity_series(
        vx,
        vy,
        vz,
        xs,
        plane_interp_order=int(cross_section_plane_interp),
    )
    flow_ts = flow_series_ml_s(vel_ts, area_mm2)
    flow_2d = flow_ts.reshape(1, -1)
    nt = int(vx.shape[3])

    rec: dict[str, float | int | str] = {
        "vessel_id": vid,
        "vessel_name": vname,
        "loc_cross_section_radius_vox": float(cross_section_radius_vox),
        "loc_cross_section_area_mm2": float(area_mm2),
        "loc_mean_velocity_mm_s": float(abs(mean_velocity_mm_s(vel_ts))),
        "loc_mean_flow_ml_s": float(abs(np.mean(flow_ts))),
        "loc_pi": float(pulsatility_index(flow_2d)[0]),
        "loc_ri": float(resistivity_index(flow_2d)[0]),
    }
    for t in range(nt):
        rec[f"loc_velocity_mm_s_t{t}"] = float(abs(vel_ts[t]))
        rec[f"loc_flow_ml_s_t{t}"] = float(abs(flow_ts[t]))
    return rec


def run_loc_measurements(
    loc_rows: list[dict[str, Any]],
    *,
    mag: np.ndarray,
    cd: np.ndarray,
    vel_mag: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    cross_section_radius_vox: float = 10.0,
    measure_resegment: bool = False,
    thr_algorithm: ThrAlgorithm = "lsthr",
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
    cs_supersampling: bool = False,
    volume_seg: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    """Measure all LOCs; returns rows suitable for ``loc_measurements.csv``."""
    if vx.ndim != 4:
        raise ValueError("Expected 4D phase volumes for LOC-wise time series.")
    out: list[dict[str, float | int | str]] = []
    for row in loc_rows:
        out.append(
            measure_loc_row(
                row,
                mag=mag,
                cd=cd,
                vel_mag=vel_mag,
                vx=vx,
                vy=vy,
                vz=vz,
                voxel_spacing=voxel_spacing,
                cross_section_radius_vox=cross_section_radius_vox,
                measure_resegment=measure_resegment,
                thr_algorithm=thr_algorithm,
                cross_section_res=cross_section_res,
                cross_section_plane_interp=cross_section_plane_interp,
                cs_supersampling=cs_supersampling,
                volume_seg=volume_seg,
            )
        )
    return out


__all__ = ["measure_loc_row", "run_loc_measurements"]
