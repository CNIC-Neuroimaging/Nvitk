"""Precompute the 4D-flow hemodynamics (qvtpy stage 6) of the example subject for ``hemo_manim.py``.

- The velocity field: the AP / RL / FH phase volumes through
  :func:`nvitk.measure.hemodynamics.velocity_mm_s_from_phases` (stage 6's own conversion), at
  every voxel of the stage-4 segmentation and every cardiac frame.
- Particle pathlines: particles advected through that time-resolved field (trilinear in space,
  linear and periodic in time, RK2), born at the arterial inlets and throughout the vessels. A
  visualisation aid, not a measurement (stage 6 traces no particles): near the wall a gentle pull
  towards the lumen's centre (at most ~10 % of a typical arterial speed) keeps them inside the
  2-voxel-dilated segmentation through the bends that 4D flow's 0.8 mm voxels resolve coarsely.
- LOC measurements: :func:`nvitk.pipes.qvtpy.util.loc.loc_measure.measure_loc_row` on every
  stage-5 LOC with the example run's settings (``measure_meta.json``), checked against the shipped
  ``loc_measurements.csv``; and, per LOC, the through-plane velocity sampled on the measuring
  plane at each frame (the picture of what is integrated).
- The dense profile: :func:`nvitk.pipes.qvtpy.util.hemodynamics.vessel_hemodynamics.compute_vessel_hemodynamics`
  with ``collect_plot_data``: every station's position, distance from the root, PI, quality,
  PWV delay, and each root's PITC fit, PWV and the branches' damping index.

Run (nvitk environment)::

    python scripts/anim/hemo_precompute.py -o scripts/anim/hemo_stages.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

#: A complete example (eicab/, 4DFlow/, qvtpy/stage2-6); the stenosis renders pass --root.
DEFAULT_ROOT = Path("/home/imarcoss/placeholder/animations/fetal example")
#: Recorded particle frames per cardiac cycle (the scene plays one cycle in CYCLE_S seconds of
#: video at 30 fps: 4 s -> 120).
FRAMES_PER_CYCLE = 120
CYCLES = 3
WARMUP_CYCLES = 1
SUBSTEPS = 10
N_PARTICLES = 5000
PLANE_N = 72


def _u8(vol: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    v = np.asarray(vol, np.float32)
    lo, hi = np.percentile(v[v > 0], [lo_pct, hi_pct])
    return (np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


class Field:
    """The periodic velocity field in voxel-index units per second on a crop."""

    def __init__(self, vx, vy, vz, spacing, tr, lo):
        self.v = [np.ascontiguousarray(a / s, np.float32) for a, s in zip((vx, vy, vz), spacing)]
        self.nt = vx.shape[3]
        self.tr = float(tr)
        self.period = self.nt * self.tr
        self.lo = np.asarray(lo, float)

    def __call__(self, p: np.ndarray, t: float) -> np.ndarray:
        """Velocity (index/s) at index points *p* (n, 3, full-volume indices) at time *t* (s)."""
        ph = (t % self.period) / self.tr
        f0 = int(np.floor(ph)) % self.nt
        f1 = (f0 + 1) % self.nt
        w = ph - np.floor(ph)
        q = (p - self.lo).T
        out = np.empty((len(p), 3), np.float32)
        for c in range(3):
            a = ndi.map_coordinates(self.v[c][..., f0], q, order=1, mode="nearest")
            b = ndi.map_coordinates(self.v[c][..., f1], q, order=1, mode="nearest")
            out[:, c] = a * (1 - w) + b * w
        return out


def trace_particles(field: Field, inside: np.ndarray, lo: np.ndarray, inlets: np.ndarray, spread: np.ndarray,
                    rng: np.random.Generator, pull: tuple[np.ndarray, ...] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advect a pool of particles; a dead one is reborn (half at the inlets, half anywhere in the
    vessels). Returns per recorded frame: positions (index), speed (index/s), age (frames)."""
    n = N_PARTICLES

    def spawn(k: int) -> np.ndarray:
        at_inlet = rng.random(k) < 0.55
        src = np.where(at_inlet[:, None], inlets[rng.integers(0, len(inlets), k)],
                       spread[rng.integers(0, len(spread), k)])
        return src + rng.uniform(-0.45, 0.45, (k, 3))

    def ok(p: np.ndarray) -> np.ndarray:
        q = np.rint(p - lo).astype(int)
        good = np.all((q >= 0) & (q < np.array(inside.shape)), axis=1)
        res = np.zeros(len(p), bool)
        res[good] = inside[tuple(q[good].T)]
        return res

    def nudge(p: np.ndarray) -> np.ndarray:
        if pull is None:
            return 0.0
        qq = (p - lo).T
        return np.stack([ndi.map_coordinates(g, qq, order=1, mode="nearest") for g in pull], 1)

    p = spawn(n)
    age = np.zeros(n, np.int32)
    outside = np.zeros(n, np.int32)
    slow = np.zeros(n, np.int32)
    total = (WARMUP_CYCLES + CYCLES) * FRAMES_PER_CYCLE
    dt_frame = field.period / FRAMES_PER_CYCLE
    h = dt_frame / SUBSTEPS
    max_age = int(1.4 * FRAMES_PER_CYCLE)
    pos_rec, spd_rec, age_rec = [], [], []
    t = 0.0
    for fr in range(total):
        for _ in range(SUBSTEPS):
            k1 = field(p, t) + nudge(p)
            k2 = field(p + 0.5 * h * k1, t + 0.5 * h) + nudge(p + 0.5 * h * k1)
            p = p + h * k2
            t += h
        v = field(p, t)
        spd = np.linalg.norm(v, axis=1)
        slow = np.where(spd < 8.0, slow + 1, 0)  # < ~6 mm/s for a while: stagnant
        outside = np.where(ok(p), 0, outside + 1)
        age += 1
        dead = (outside > 2) | (slow > 24) | (age > max_age)
        if dead.any():
            k = int(dead.sum())
            p[dead] = spawn(k)
            age[dead] = 0
            slow[dead] = 0
            outside[dead] = 0
        if fr >= WARMUP_CYCLES * FRAMES_PER_CYCLE:
            pos_rec.append(p.astype(np.float32).copy())
            spd_rec.append(spd.astype(np.float32))
            age_rec.append(np.minimum(age, 255).astype(np.uint8))
    return np.stack(pos_rec), np.stack(spd_rec), np.stack(age_rec)


def compute(root: Path) -> tuple[dict, dict]:
    from nvitk.core.backend import set_global_backend

    set_global_backend("numpy")
    from nvitk.measure.cross_section import plane_basis_from_tangent
    from nvitk.measure.hemodynamics import _mad_outlier_mask, velocity_mm_s_from_phases, weighted_linear_fit
    from nvitk.pipes.qvtpy.labels import QVTPY_ARTERIAL_ID_TO_NAME
    from nvitk.pipes.qvtpy.util.hemodynamics.vessel_hemodynamics import compute_vessel_hemodynamics
    from nvitk.pipes.qvtpy.util.loc.loc_measure import measure_loc_row

    q = root / "qvtpy"
    flow = root / "4DFlow"
    mm = json.loads((q / "stage6_measure" / "measure_meta.json").read_text())
    tr = float(mm["temporal_resolution_s"])

    def ph(axis: str) -> np.ndarray:
        f = next((flow / axis).glob("*_ph.nii.gz"))
        return np.asanyarray(nib.load(f).dataobj).astype(np.float64)

    vx, vy, vz = velocity_mm_s_from_phases(ph("AP"), ph("RL"), ph("FH"))
    vx, vy, vz = (np.asarray(a) for a in (vx, vy, vz))
    nt = vx.shape[3]
    seg_img = nib.load(q / "stage4_4dflow_segmentation" / "seg_4dflow.nii.gz")
    seg = np.asanyarray(seg_img.dataobj).astype(np.int32)
    aff = seg_img.affine
    spacing = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
    cd = np.asanyarray(nib.load(flow / "ComplexDifference_3D.nii.gz").dataobj).astype(np.float64)
    angio = np.asanyarray(nib.load(flow / "Angiography_3D.nii.gz").dataobj).astype(np.float64)
    vel_mag = np.abs(cd)  # stage 6's fallback without VelocityMagnitude_3D

    # ---- the field at the segmentation's voxels
    sidx = np.argwhere(seg > 0)
    sel = tuple(sidx.T)
    vel = np.stack([vx[sel], vy[sel], vz[sel]], -1)  # (n, nt, 3) mm/s along the index axes
    ica_up = float(np.mean(vz[seg == 1].max(axis=-1)))

    # ---- LOC measurements, as stage 6
    locs = pd.read_csv(q / "stage5_loc_generation" / "locs.csv")
    shipped = pd.read_csv(q / "stage6_measure" / "loc_measurements.csv")
    rows, planes, pmasks = [], [], []
    for _, r in locs.iterrows():
        rec = measure_loc_row(r.to_dict(), mag=angio, cd=cd, vel_mag=vel_mag, vx=vx, vy=vy, vz=vz,
                              voxel_spacing=spacing, cross_section_radius_vox=float(mm["cross_section_radius_vox"]),
                              measure_resegment=bool(mm["measure_resegment"]),
                              thr_algorithm=mm["measure_thr_algorithm"],
                              cross_section_plane_interp=int(mm["cross_section_plane_interp"]),
                              cs_supersampling=bool(mm["cs_supersampling"]), volume_seg=seg)
        rows.append(rec)
        # the through-plane velocity on the measuring plane, per frame (display)
        c = r[["centerline_x", "centerline_y", "centerline_z"]].to_numpy(float)
        t = r[["tangent_x", "tangent_y", "tangent_z"]].to_numpy(float)
        t = t / np.linalg.norm(t)
        u, v = plane_basis_from_tangent(t)
        R = float(mm["cross_section_radius_vox"]) * 0.6
        g = np.linspace(-R, R, PLANE_N)
        gu, gv = np.meshgrid(g, g, indexing="xy")
        pts = c + gu[..., None] * u + gv[..., None] * v  # (N, N, 3) index coords
        crd = pts.reshape(-1, 3).T
        lab_pl = ndi.map_coordinates(seg, crd, order=0, mode="constant").reshape(PLANE_N, PLANE_N)
        m = lab_pl == int(r["vessel_id"])
        cc, nlab = ndi.label(m)
        if nlab > 1:
            mid = cc[PLANE_N // 2, PLANE_N // 2]
            m = cc == (mid if mid > 0 else np.argmax(np.bincount(cc.ravel())[1:]) + 1)
        series = np.empty((nt, PLANE_N, PLANE_N), np.float32)
        for k in range(nt):
            comp = [ndi.map_coordinates(a[..., k], crd, order=1, mode="nearest") for a in (vx, vy, vz)]
            series[k] = (comp[0] * t[0] + comp[1] * t[1] + comp[2] * t[2]).reshape(PLANE_N, PLANE_N)
        planes.append(series)
        pmasks.append(m)
    meas = pd.DataFrame(rows)
    agree = float(np.nanmax(np.abs(meas["loc_mean_flow_ml_s"].to_numpy() - shipped["loc_mean_flow_ml_s"].to_numpy())))

    # ---- dense profile (PITC / PWV / damping)
    hemo = compute_vessel_hemodynamics(
        None, volume_seg=seg, cd=cd, mag=angio, vel_mag=vel_mag, vx=vx, vy=vy, vz=vz, voxel_spacing=spacing,
        temporal_resolution_s=tr, stride=int(mm["pitc_stride"]), radius_vox=float(mm["cross_section_radius_vox"]),
        quality_thresh=float(mm["pitc_quality_thresh"]), quality_metric=str(mm["pitc_quality_metric"]),
        measure_resegment=bool(mm["pitc_measure_resegment"]), label_constrain=bool(mm["pitc_label_constrain"]),
        thr_algorithm=mm["measure_thr_algorithm"], plane_interp_order=int(mm["cross_section_plane_interp"]),
        cs_supersampling=bool(mm["cs_supersampling"]), collect_plot_data=True)
    st_rows, regions = [], {}
    for rid, g in hemo.geometry_by_region.items():
        ri = len(regions)
        regions[rid] = {"index": ri, "pitc_slope": g.pitc_slope, "pitc_intercept": g.pitc_intercept,
                        "pitc_r2": g.pitc_r2, "pitc_n": g.pitc_n, "global_pi": g.global_pi,
                        "quality_thresh": g.quality_thresh, "pwv_bjornfoot_m_s": g.pwv_bjornfoot_m_s,
                        "pwv_fielding_m_s": g.pwv_fielding_m_s, "pwv_n_stations": g.pwv_n_stations,
                        "root_label": g.root_label}
        for vname, vg in g.vessels.items():
            for s in vg.stations:
                st_rows.append([ri, s.vessel_id, s.centerline_x, s.centerline_y, s.centerline_z, s.distance_mm, s.pi,
                                s.quality, s.area_mm2, float(s.used_for_pwv), s.pwv_xcor_time_s,
                                s.pwv_weight_area])
    stations = np.asarray(st_rows, np.float32)
    # ---- PWV by cross-correlation (Fielding): the delays stage 6 plots, filtered as pwv_fielding_xcor does
    # (Q > thresh, area weight > 0, MAD outliers out), then the area-weighted line delay = d / PWV
    pwv_pts = {}
    for rid, pd_ in hemo.region_plot_data.items():
        d = np.asarray(pd_["pwv_distance_mm"], float)
        t = np.asarray(pd_["pwv_xcor_time_s"], float)
        w = np.asarray(pd_["pwv_weight_area"], float)
        keep = np.isfinite(t) & np.isfinite(d) & (w > 0)
        if keep.sum() >= 4:
            keep &= _mad_outlier_mask(t)
        slope = icpt = float("nan")
        if keep.sum() >= 2:
            fit = weighted_linear_fit(d[keep], t[keep], w[keep])
            slope, icpt = float(fit["slope"]), float(fit["intercept"])
        ed = np.asarray(pd_["pwv_excluded_distance_mm"], float)
        et = np.asarray(pd_["pwv_excluded_xcor_time_s"], float)
        pwv_pts[rid] = np.concatenate([
            np.stack([d, t, w, keep.astype(float)], 1),           # 1 = in the fit, 0 = MAD outlier
            np.stack([ed, et, np.zeros_like(ed), np.full_like(ed, -1.0)], 1),  # -1 = Q <= thresh
        ]).astype(np.float32)
        regions[rid].update({"pwv_xcor_slope_s_per_mm": slope, "pwv_xcor_intercept_s": icpt,
                             "pwv_xcor_n_fit": int(keep.sum()),
                             "pwv_xcor_m_s_refit": 1.0 / slope / 1000.0 if np.isfinite(slope) and slope else float("nan")})
    damping = {r["region_id"]: r.get("damping_index") for r in hemo.summary_rows if r.get("row_kind") == "branch"}

    # ---- particles
    inside = ndi.binary_dilation(seg > 0, iterations=2)
    lo = np.maximum(np.argwhere(inside).min(0) - 3, 0)
    hi = np.minimum(np.argwhere(inside).max(0) + 4, seg.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    fld = Field(vx[crop], vy[crop], vz[crop], spacing, tr, lo)
    inlets = []
    for lid in (1, 2, 3, 13, 14):
        pts = np.argwhere(seg == lid)
        if len(pts):
            zmin = pts[:, 2].min()
            inlets.append(pts[pts[:, 2] <= zmin + 2])
    inlets = np.concatenate(inlets).astype(float)
    spread = np.argwhere(seg > 0).astype(float)
    # the wall pull: up to 25 voxels/s (~20 mm/s) up the distance-to-wall gradient, fading to 0 at
    # 1.5 voxels inside the lumen
    edt = ndi.distance_transform_edt(ndi.binary_dilation(seg > 0, iterations=2)[crop])
    core = ndi.distance_transform_edt(seg[crop] > 0)
    grad = np.gradient(ndi.gaussian_filter(edt, 0.8))
    norm = np.sqrt(sum(g * g for g in grad)) + 1e-6
    w = 25.0 * np.clip(1.0 - core / 1.5, 0.0, 1.0)
    pull = tuple((g / norm * w).astype(np.float32) for g in grad)
    pos, spd, age = trace_particles(fld, inside[crop], lo, inlets, spread, np.random.default_rng(0), pull)
    pos_w = (pos @ aff[:3, :3].T + aff[:3, 3]).astype(np.float16)
    spd_mm = (spd * float(np.mean(spacing))).astype(np.float16)

    # ---- volumes for the MIP
    cd8 = _u8(cd, 50.0, 99.9)
    names = {int(k): v for k, v in QVTPY_ARTERIAL_ID_TO_NAME.items() if k > 0}
    arrays = {
        "seg_vox": sidx.astype(np.int16), "seg_lab": seg[sel].astype(np.int8), "affine": aff,
        "vel": vel.astype(np.float16),
        "p_pos": pos_w, "p_speed": spd_mm, "p_age": age,
        "loc_planes": np.stack(planes).astype(np.float16), "loc_masks": np.stack(pmasks),
        "loc_rows": locs[["vessel_id", "centerline_x", "centerline_y", "centerline_z", "tangent_x", "tangent_y",
                          "tangent_z"]].to_numpy(np.float32),
        "loc_flow": meas[[f"loc_flow_ml_s_t{k}" for k in range(nt)]].to_numpy(np.float32),
        "loc_vel": meas[[f"loc_velocity_mm_s_t{k}" for k in range(nt)]].to_numpy(np.float32),
        "loc_scalar": meas[["loc_cross_section_area_mm2", "loc_mean_velocity_mm_s", "loc_mean_flow_ml_s", "loc_pi",
                            "loc_ri"]].to_numpy(np.float32),
        "stations": stations, "cd8": cd8,
        **{f"pwv_pts_{rid}": v for rid, v in pwv_pts.items()},
    }
    meta = {
        "spacing": list(spacing), "nt": int(nt), "tr_s": tr, "period_s": nt * tr,
        "frames_per_cycle": FRAMES_PER_CYCLE, "cycles": CYCLES, "n_particles": N_PARTICLES,
        "plane_radius_vox": float(mm["cross_section_radius_vox"]) * 0.6,
        "loc_names": locs["vessel_name"].tolist(), "loc_roles": locs["loc_role"].tolist(),
        "names": {str(k): v for k, v in names.items()},
        "regions": regions, "damping": damping,
        "station_cols": ["region", "vessel_id", "x", "y", "z", "distance_mm", "pi", "quality", "area_mm2",
                         "used_for_pwv", "pwv_xcor_time_s", "pwv_weight_area"],
        "loc_flow_max_abs_diff_vs_shipped_ml_s": agree,
        "ica_peak_vz_mean_mm_s": ica_up,
    }
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Example folder with 4DFlow/ and qvtpy/.")
    args = ap.parse_args()
    arrays, meta = compute(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta, default=str)), **arrays)
    print(json.dumps({k: meta[k] for k in ("regions", "loc_flow_max_abs_diff_vs_shipped_ml_s", "ica_peak_vz_mean_mm_s",
                                           "period_s")}, default=str, indent=1))
    print({k: v.shape for k, v in arrays.items()})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
