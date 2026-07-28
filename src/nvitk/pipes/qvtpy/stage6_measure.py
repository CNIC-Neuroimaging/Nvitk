"""qvtpy stage 6: LOC-wise velocity / PI / RI from PC phase volumes.

**Inputs**

- Stage-5 ``locs.csv``, AP/RL/FH phase NIfTIs (:func:`~nvitk.io.conversors.phase2volume.discover_phase_inputs`),
  optional stage-4 ``seg_4dflow`` when not re-segmenting in-plane.

**Outputs**

- ``loc_measurements.csv``, optional cross-section QC PNGs under ``stage6_measure/``.
"""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import Any, Literal, TextIO

import click
import numpy as np

import nvitk
from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.pipes.qvtpy.util.io.sge_backend import (
    sge_backend_cli_args,
    sge_qvtpy_stage_resources,
    sge_stage_extra_env,
    sge_stage_use_nv,
)
from nvitk.core.logger import Logger
from nvitk.io.conversors.phase2volume import discover_phase_inputs
from nvitk.io.imageio import imread
from nvitk.measure.hemodynamics import velocity_mm_s_from_phases
from nvitk.pipes.qvtpy import config as cfg
from nvitk.measure.cross_section import ThrAlgorithm
from nvitk.pipes.qvtpy.util.centerline.centerline_io import (
    flatten_branches,
    load_arterial_branches,
    load_arterial_centerlines,
    load_centerlines,
)
from nvitk.pipes.qvtpy.util.loc.loc_measure import run_loc_measurements
from nvitk.pipes.qvtpy.util.hemodynamics.measure_qc import save_loc_cross_section_qc_png
from nvitk.pipes.qvtpy.util.hemodynamics.vessel_hemodynamics import compute_vessel_hemodynamics
from nvitk.measure.hemodynamics import QUALITY_THRESH_DEFAULT
from nvitk.pipes.qvtpy.labels import qvtpy_vessel_name

setup(globals())

log = Logger()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_RADIUS_VOX = 10.0


# ---------------------------------------------------------------------------
# Path helpers + contrast / phase I/O
# ---------------------------------------------------------------------------


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage3_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _stage5_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE5_LOC_DIR


def _stage4_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE4_SEG_DIR


def _stage6_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE6_MEASURE_DIR


def _load_seg_4dflow(output_root: Path, subject: str) -> np.ndarray:
    seg_p = _stage4_dir(output_root, subject) / "seg_4dflow.nii.gz"
    if not seg_p.is_file():
        raise FileNotFoundError(f"Missing {seg_p} (run stage4; required when --no-measure-resegment)")
    return as_backend_array(imread(seg_p).data).astype(np.int32)


def _voxel_spacing(ap_img_path: Path) -> tuple[float, float, float]:
    ap_img = imread(ap_img_path)
    sp = ap_img.spacing
    if sp is not None and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    aff = ap_img.affine
    if aff is not None:
        a = to_numpy(aff).astype(np.float64)
        return (
            float(np.linalg.norm(a[:3, 0])),
            float(np.linalg.norm(a[:3, 1])),
            float(np.linalg.norm(a[:3, 2])),
        )
    return (1.0, 1.0, 1.0)


# Cardiac phase spacing is typically tens of ms. Used only to sanity-check
# RR / n_phases (or NominalInterval) estimates — never to validate NIfTI t_res.
_CARDIAC_FRAME_MIN_S = 0.015
_CARDIAC_FRAME_MAX_S = 0.25


def _plausible_cardiac_frame_s(tr: float) -> bool:
    return _CARDIAC_FRAME_MIN_S <= float(tr) <= _CARDIAC_FRAME_MAX_S


def _n_cardiac_phases(meta: dict[str, Any], n_timepoints: int | None) -> int | None:
    """Number of RR-normalized cardiac bins (prefer volume shape over DICOM tags)."""
    if n_timepoints is not None and int(n_timepoints) > 1:
        return int(n_timepoints)
    for key in ("CardiacNumberOfImages", "NumberOfTemporalPositions"):
        val = meta.get(key)
        if val is None:
            continue
        try:
            n = int(round(float(val)))
        except (TypeError, ValueError):
            continue
        if n > 1:
            return n
    return None


def _cardiac_frame_duration_s(
    meta: dict[str, Any],
    *,
    n_timepoints: int | None = None,
) -> tuple[float | None, str]:
    """Resolve cardiac-frame duration (s) for RR-normalized phase bins.

    PESA / Q-flow volumes store a fixed number of frames (e.g. 15) that always
    span one cardiac cycle, so ``Δt = RR / n_phases``. NIfTI ``t_res`` /
    ``pixdim[4]`` and DICOM ``RepetitionTime`` are MRI RF timing and must not
    be used. ``FrameTime`` is also unreliable when bins are RR-normalized.

    Priority: ``HeartRate`` → ``NominalInterval`` → (last resort) ``FrameTime``.

    Returns ``(duration_s, source)``; ``duration_s`` is ``None`` when unresolved.
    """
    def _pos(key: str, *, scale: float = 1.0) -> float | None:
        val = meta.get(key)
        if val is None:
            return None
        try:
            out = float(val) * float(scale)
        except (TypeError, ValueError):
            return None
        return out if out > 0 else None

    n_t = _n_cardiac_phases(meta, n_timepoints)

    # 1) RR / n_phases from HeartRate (correct for normalized cardiac bins).
    hr = _pos("HeartRate")
    if hr is not None and hr > 10.0 and n_t is not None:
        tr = (60.0 / hr) / float(n_t)
        if _plausible_cardiac_frame_s(tr):
            return tr, f"HeartRate/{n_t}"

    # 2) Nominal RR interval (ms) / n_phases when HeartRate is missing.
    nominal = _pos("NominalInterval", scale=1e-3)
    if nominal is not None and n_t is not None:
        tr = nominal / float(n_t)
        if _plausible_cardiac_frame_s(tr):
            return tr, f"NominalInterval/{n_t}"

    # 3) FrameTime only if no RR-based estimate is available.
    frame_time = _pos("FrameTime", scale=1e-3)
    if frame_time is not None and _plausible_cardiac_frame_s(frame_time):
        log.warning(
            "Using FrameTime as cardiac Δt; prefer HeartRate/n_phases for "
            "RR-normalized bins"
        )
        return frame_time, "FrameTime"

    # Explicitly ignore t_res / TemporalResolution / RepetitionTime.
    for key in ("temporal_resolution", "t_res", "TemporalResolution", "RepetitionTime"):
        if meta.get(key) is not None:
            log.warning(
                f"Ignoring {key}={meta.get(key)!r} for cardiac frame duration "
                "(RR-normalized bins: use HeartRate/n_phases, not MRI t_res/TR)"
            )
            break

    return None, "none"


def _phase_temporal_resolution_s(ap_phase_path: Path) -> float | None:
    """Cardiac-frame duration (s) = RR / n_phases from phase metadata / sidecars.

    Does not use NIfTI ``t_res`` (often MRI ``RepetitionTime``). Returns ``None``
    when HeartRate / NominalInterval / FrameTime cannot resolve Δt, so PWV is
    skipped rather than using a wrong clock.
    """
    ap_img = imread(ap_phase_path)
    meta = dict(ap_img.metadata or {})
    data = ap_img.data
    n_t = int(data.shape[3]) if getattr(data, "ndim", 0) >= 4 else None
    tr, source = _cardiac_frame_duration_s(meta, n_timepoints=n_t)
    if tr is not None:
        log.info(
            f"stage6 cardiac frame duration: {tr * 1e3:.3f} ms "
            f"(source={source}, n_t={n_t})"
        )
        return tr

    # Sidecars often carry HeartRate when the NIfTI header only has MRI TR.
    for js in sorted(ap_phase_path.parent.glob("*.json")):
        try:
            data_js = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data_js, dict):
            continue
        tr, source = _cardiac_frame_duration_s(data_js, n_timepoints=n_t)
        if tr is not None:
            log.info(
                f"stage6 cardiac frame duration: {tr * 1e3:.3f} ms "
                f"(source={source} via {js.name}, n_t={n_t})"
            )
            return tr
    return None


def _load_contrast(nifti_root: Path, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = nifti_root / subject / "4DFlow"
    cd_p = sub / "ComplexDifference_3D.nii.gz"
    if not cd_p.is_file():
        cd_p = sub / "ComplexDifference_3D.nii"
    cd = as_backend_array(imread(cd_p).data).astype(np.float64)
    mag = cd
    angio = sub / "Angiography_3D.nii.gz"
    if angio.is_file():
        mag = as_backend_array(imread(angio).data).astype(np.float64)
    vel = np.abs(cd)
    vmag = sub / "VelocityMagnitude_3D.nii.gz"
    if vmag.is_file():
        vel = as_backend_array(imread(vmag).data).astype(np.float64)
    return mag, cd, vel


_PITC_PROFILE_FIELDS = [
    "root_region_id",
    "vessel_id",
    "vessel_name",
    "station_index",
    "distance_mm",
    "pi",
    "quality",
    "quality_metric",
    "area_mm2",
    "circularity",
    "flow_mean_ml_s",
]

_VESSEL_HEMO_FIELDS = [
    "region_id",
    "region_label",
    "row_kind",
    "pitc_slope",
    "pitc_intercept",
    "pitc_r2",
    "pitc_n",
    "global_pi",
    "pwv_bjornfoot_m_s",
    "pwv_fielding_m_s",
    "pwv_r_fielding",
    "pwv_n_stations",
    "damping_index",
]


def _run_vessel_hemodynamics(
    subject: str,
    *,
    out_dir: Path,
    output_root: Path,
    volume_seg: np.ndarray,
    cd: np.ndarray,
    mag: np.ndarray,
    vel_mag: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    voxel_spacing: tuple[float, float, float],
    temporal_resolution_s: float | None,
    radius_vox: float,
    pitc_stride: int,
    pitc_quality_thresh: float,
    pitc_quality_metric: str,
    pitc_measure_resegment: bool,
    pitc_label_constrain: bool,
    measure_thr_algorithm: ThrAlgorithm,
    cross_section_res: int,
    cross_section_plane_interp: int,
    cs_supersampling: bool,
    summary: dict[str, int],
    save_plots: bool = False,
    seg_metadata: dict | None = None,
) -> None:
    """Sample dense PITC/PWV and write ``pitc_profile.csv`` + ``vessel_hemodynamics.csv``."""
    if not bool(np.any(volume_seg)):
        log.warning(f"[{subject}] stage6 PITC: empty stage-4 segmentation; skipping")
        return
    # Prefer stage-3 arterial polylines (same bias as stage-4 CL export) and load
    # venous polylines for flow_waveforms.png.
    waveform_centerlines: dict[int, np.ndarray] = {}
    prefer_arterial: dict[int, np.ndarray] = {}
    s3 = _stage3_dir(output_root, subject)
    s4 = _stage4_dir(output_root, subject)
    try:
        arterial, venous, meta = load_centerlines(s3, min_points=3, stage4_dir=s4)
        prefer_arterial = {int(k): to_numpy(v) for k, v in arterial.items()}
        from nvitk.pipes.qvtpy.util.centerline.venous_heuristics import venous_name_to_label_id

        venous_ids = {
            k: int(v) for k, v in (meta.get("venous_label_by_name") or {}).items()
        }
        for name, poly in venous.items():
            lid = venous_name_to_label_id(str(name), venous_ids)
            if lid is not None:
                waveform_centerlines[int(lid)] = to_numpy(poly)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[{subject}] stage6: stage3/4 centerlines unavailable ({exc})")
    hemo = compute_vessel_hemodynamics(
        waveform_centerlines or None,
        volume_seg=volume_seg,
        prefer_polylines=prefer_arterial or None,
        cd=cd,
        mag=mag,
        vel_mag=vel_mag,
        vx=vx,
        vy=vy,
        vz=vz,
        voxel_spacing=voxel_spacing,
        temporal_resolution_s=temporal_resolution_s,
        stride=int(pitc_stride),
        radius_vox=float(radius_vox),
        quality_thresh=float(pitc_quality_thresh),
        quality_metric=str(pitc_quality_metric),
        measure_resegment=bool(pitc_measure_resegment),
        label_constrain=bool(pitc_label_constrain),
        thr_algorithm=measure_thr_algorithm,
        cross_section_res=int(cross_section_res),
        plane_interp_order=int(cross_section_plane_interp),
        cs_supersampling=bool(cs_supersampling),
        # Always collect plot/geometry data so QC can reload pipeline results
        # without recomputing in the GUI (PNGs remain gated by save_plots).
        collect_plot_data=True,
    )

    with (out_dir / "pitc_profile.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_PITC_PROFILE_FIELDS)
        w.writeheader()
        for row in hemo.profile_rows:
            w.writerow({k: row.get(k, "") for k in _PITC_PROFILE_FIELDS})

    with (out_dir / "vessel_hemodynamics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_VESSEL_HEMO_FIELDS)
        w.writeheader()
        for row in hemo.summary_rows:
            w.writerow({k: row.get(k, "") for k in _VESSEL_HEMO_FIELDS})

    summary["n_profile_stations"] = len(hemo.profile_rows)
    summary["n_regions"] = sum(1 for r in hemo.summary_rows if r.get("row_kind") == "root")
    log.info(
        f"[{subject}] stage6 PITC/PWV: {summary['n_profile_stations']} stations, "
        f"{summary['n_regions']} roots -> vessel_hemodynamics.csv"
    )

    try:
        from nvitk.pipes.qvtpy.util.hemodynamics.hemo_viz_io import write_hemo_viz_bundle

        bundle_path = write_hemo_viz_bundle(out_dir, hemo)
        if bundle_path is not None:
            summary["hemo_viz_bundle"] = bundle_path.name
            log.info(f"[{subject}] stage6: wrote {bundle_path.name} for QC reload")
    except Exception as exc:  # noqa: BLE001 - bundle is optional for stage success
        log.warning(f"[{subject}] stage6: failed to write hemo_viz_bundle ({exc})")

    if save_plots:
        _save_measurement_plots(
            subject,
            out_dir=out_dir,
            hemo=hemo,
            volume_seg=hemo.volume_seg if hemo.volume_seg is not None else volume_seg,
            seg_metadata=seg_metadata,
        )


def _save_measurement_plots(
    subject: str,
    *,
    out_dir: Path,
    hemo,
    volume_seg: np.ndarray,
    seg_metadata: dict | None,
) -> None:
    """Render PITC/PWV/Bjornfoot-QC/flow figures and per-region PITC masks."""
    from nvitk.pipes.qvtpy.util.hemodynamics.measure_plots import (
        plot_bjornfoot_qc_figure,
        plot_flow_waveforms,
        plot_pitc_figure,
        plot_pwv_figure,
        save_pitc_region_masks,
    )

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_pitc_figure(hemo.region_plot_data, plots_dir / "pitc.png")
        plot_pwv_figure(hemo.region_plot_data, plots_dir / "pwv.png")
        plot_bjornfoot_qc_figure(
            hemo.region_plot_data, plots_dir / "pwv_bjornfoot_qc.png"
        )
        plot_flow_waveforms(
            hemo.region_plot_data,
            plots_dir / "flow_waveforms.png",
            all_label_waveforms=hemo.all_label_waveforms,
        )
    except Exception as exc:  # noqa: BLE001 - plotting must not fail the stage
        import traceback
        log.warning(traceback.format_exc())
        log.warning(f"[{subject}] stage6 measurement plots failed: {exc}")
    try:
        masks = save_pitc_region_masks(
            volume_seg, out_dir / "pitc_masks", metadata=seg_metadata
        )
        log.info(f"[{subject}] stage6 PITC branch masks: {len(masks)} region(s)")
    except Exception as exc:  # noqa: BLE001
        import traceback
        log.warning(traceback.format_exc())
        log.warning(f"[{subject}] stage6 PITC mask export failed: {exc}")


# ---------------------------------------------------------------------------
# Stage 6: per-LOC flow, PI, RI
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
    cross_section_radius_vox: float = _DEFAULT_RADIUS_VOX,
    measure_resegment: bool = False,
    measure_thr_algorithm: ThrAlgorithm = "lsthr",
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
    cs_supersampling: bool = False,
    write_cross_section_qc: bool = True,
    skip_pitc: bool = False,
    pitc_stride: int = 1,
    pitc_quality_thresh: float = QUALITY_THRESH_DEFAULT,
    pitc_quality_metric: str = "stdv_from_mean",
    pitc_measure_resegment: bool = True,
    pitc_label_constrain: bool = True,
    save_plots: bool = False,
) -> Path:
    """Measure flow metrics at each LOC; return stage-6 output directory."""
    # ---- Prerequisites: stage5 locs.csv --------------------------------------
    loc_csv = _stage5_dir(output_root, subject) / "locs.csv"
    if not loc_csv.is_file():
        raise FileNotFoundError(f"Missing {loc_csv} (run stage5)")
    out_dir = _stage6_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    meas_csv = out_dir / "loc_measurements.csv"
    if skip_existing and meas_csv.is_file():
        log.info(f"[{subject}] stage6 measure: skip -> {out_dir}")
        return out_dir

    # ---- Phase volumes → mm/s velocity time series ---------------------------
    inputs = discover_phase_inputs(nifti_root / subject)
    voxel_spacing = _voxel_spacing(inputs.ap_phase_path)
    temporal_resolution_s = _phase_temporal_resolution_s(inputs.ap_phase_path)
    if temporal_resolution_s is None:
        log.warning(
            f"[{subject}] stage6: no temporal resolution in phase metadata; "
            "PWV will be skipped for this subject"
        )
    mag, cd, vel_mag = _load_contrast(nifti_root, subject)
    volume_seg: np.ndarray | None = None
    if not measure_resegment:
        volume_seg = _load_seg_4dflow(output_root, subject)

    ap = as_backend_array(imread(inputs.ap_phase_path).data).astype(np.float64)
    rl = as_backend_array(imread(inputs.rl_phase_path).data).astype(np.float64)
    fh = as_backend_array(imread(inputs.fh_phase_path).data).astype(np.float64)
    vx, vy, vz = velocity_mm_s_from_phases(ap, rl, fh)
    if vx.ndim != 4:
        raise ValueError("Expected 4D phase volumes for LOC-wise time series.")
    nt = int(vx.shape[3])

    vel_cols = [f"loc_velocity_mm_s_t{t}" for t in range(nt)]
    flow_cols = [f"loc_flow_ml_s_t{t}" for t in range(nt)]
    fieldnames = [
        "vessel_id",
        "vessel_name",
        "loc_cross_section_radius_vox",
        "loc_cross_section_area_mm2",
        "loc_mean_velocity_mm_s",
        "loc_mean_flow_ml_s",
        "loc_pi",
        "loc_ri",
        *vel_cols,
        *flow_cols,
    ]

    # ---- Optional centerlines for QC -------------------------------------------
    arterial_cls: dict[int, np.ndarray] | None = None
    branch_cls_by_name: dict[str, np.ndarray] = {}
    qc_dir = out_dir / "cross-sections"
    if write_cross_section_qc:
        s3 = _stage3_dir(output_root, subject)
        s4 = _stage4_dir(output_root, subject)
        try:
            arterial_cls = {
                int(k): to_numpy(v)
                for k, v in load_arterial_centerlines(s3, min_points=3).items()
            }
        except FileNotFoundError:
            log.warning(f"[{subject}] stage6 QC: missing stage3 centerlines, skipping PNGs")
        # Resolve QC polylines per named branch from the stage-4 branch sidecar.
        try:
            branch_cls_by_name = {
                str(name): to_numpy(pts)
                for name, pts in flatten_branches(
                    load_arterial_branches(s4, min_points=3, from_segmentation=True)
                ).items()
            }
        except FileNotFoundError:
            branch_cls_by_name = {}

    # ---- Per-LOC: resegment, masked-plane flow, PI / RI ----------------------
    loc_rows: list[dict[str, str]] = []
    with loc_csv.open(newline="", encoding="utf-8") as fh:
        loc_rows = list(csv.DictReader(fh))

    rows_out = run_loc_measurements(
        loc_rows,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        vx=vx,
        vy=vy,
        vz=vz,
        voxel_spacing=voxel_spacing,
        cross_section_radius_vox=cross_section_radius_vox,
        measure_resegment=measure_resegment,
        thr_algorithm=measure_thr_algorithm,
        cross_section_res=cross_section_res,
        cross_section_plane_interp=cross_section_plane_interp,
        cs_supersampling=cs_supersampling,
        volume_seg=volume_seg,
    )
    qc_paths: list[str] = []
    for row, _rec in zip(loc_rows, rows_out):
        vid = int(row["vessel_id"])
        vname = (row.get("vessel_name") or "").strip() or qvtpy_vessel_name(vid)

        qc_poly = branch_cls_by_name.get(vname)
        if qc_poly is None and arterial_cls is not None:
            qc_poly = arterial_cls.get(vid)
        if qc_poly is not None:
            seg_id = int(row.get("segment_id") or 0)
            loc_role = str(row.get("loc_role") or "mid")
            cl_idx = int(row.get("centerline_index") or 0)
            qc_name = f"{vname}_seg{seg_id}.png"
            qc_path = qc_dir / qc_name
            try:
                save_loc_cross_section_qc_png(
                    qc_path,
                    cd=cd,
                    mag=mag,
                    vel_mag=vel_mag,
                    centerline_pts=qc_poly,
                    loc_index=cl_idx,
                    vessel_name=vname,
                    segment_id=seg_id,
                    loc_role=loc_role,
                    voxel_spacing=voxel_spacing,
                    radius_vox=cross_section_radius_vox,
                    cross_section_res=cross_section_res,
                    plane_interp_order=int(cross_section_plane_interp),
                    cs_supersampling=cs_supersampling,
                    measure_resegment=measure_resegment,
                    thr_algorithm=measure_thr_algorithm,
                    volume_seg=volume_seg,
                    volume_label_id=vid,
                )
                qc_paths.append(str(qc_path.relative_to(out_dir)))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                log.warning(f"[{subject}] QC PNG failed for {vname}: {exc}")

    # ---- Write loc_measurements.csv ------------------------------------------
    with meas_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    # ---- Dense PITC / PWV vessel hemodynamics --------------------------------
    pitc_summary: dict[str, int] = {"n_profile_stations": 0, "n_regions": 0}
    if not skip_pitc:
        seg_metadata: dict | None = None
        if save_plots:
            seg_p = _stage4_dir(output_root, subject) / "seg_4dflow.nii.gz"
            if seg_p.is_file():
                seg_metadata = dict(imread(seg_p).metadata or {})
        try:
            _run_vessel_hemodynamics(
                subject,
                out_dir=out_dir,
                output_root=output_root,
                volume_seg=volume_seg if volume_seg is not None else _load_seg_4dflow(output_root, subject),
                cd=cd,
                mag=mag,
                vel_mag=vel_mag,
                vx=vx,
                vy=vy,
                vz=vz,
                voxel_spacing=voxel_spacing,
                temporal_resolution_s=temporal_resolution_s,
                radius_vox=cross_section_radius_vox,
                pitc_stride=pitc_stride,
                pitc_quality_thresh=pitc_quality_thresh,
                pitc_quality_metric=pitc_quality_metric,
                pitc_measure_resegment=pitc_measure_resegment,
                pitc_label_constrain=pitc_label_constrain,
                measure_thr_algorithm=measure_thr_algorithm,
                cross_section_res=cross_section_res,
                cross_section_plane_interp=cross_section_plane_interp,
                cs_supersampling=cs_supersampling,
                summary=pitc_summary,
                save_plots=save_plots,
                seg_metadata=seg_metadata,
            )
        except Exception as exc:
            import traceback
            log.warning(traceback.format_exc())
            log.warning(f"[{subject}] stage6 PITC/PWV failed: {exc}")

    (out_dir / "measure_meta.json").write_text(
        json.dumps(
            {
                "n_rows": len(rows_out),
                "n_timepoints": nt,
                "temporal_resolution_s": temporal_resolution_s,
                "pitc_enabled": (not skip_pitc),
                "pitc_stride": int(pitc_stride),
                "pitc_quality_thresh": float(pitc_quality_thresh),
                "pitc_quality_metric": str(pitc_quality_metric),
                "pitc_measure_resegment": bool(pitc_measure_resegment),
                "pitc_label_constrain": bool(pitc_label_constrain),
                "pitc_n_profile_stations": pitc_summary["n_profile_stations"],
                "pitc_n_regions": pitc_summary["n_regions"],
                "save_plots": bool(save_plots),
                "measure_resegment": bool(measure_resegment),
                "cs_supersampling": bool(cs_supersampling),
                "measure_thr_algorithm": str(measure_thr_algorithm),
                "cross_section_radius_vox": float(cross_section_radius_vox),
                "cross_section_plane_interp": int(cross_section_plane_interp),
                "reported_flow_velocity_as_magnitude": True,
                "cross_section_qc_dir": "cross-sections",
                "cross_section_qc_pngs": qc_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"[{subject}] stage6 measure -> {meas_csv}")

    try:
        from nvitk.pipes.qvtpy.common.db_publish import maybe_publish_stage6_on_sge

        maybe_publish_stage6_on_sge(subject_uid=subject, stage6_dir=out_dir)
    except Exception as exc:  # noqa: BLE001 - DB publish must never fail the stage
        log.warning(f"[{subject}] stage6 DB publish hook skipped: {exc}")

    return out_dir


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _subject_sge_spec(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    cross_section_radius_vox: float = _DEFAULT_RADIUS_VOX,
    measure_resegment: bool = False,
    measure_thr_algorithm: str = "lsthr",
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
    cs_supersampling: bool = False,
    skip_pitc: bool = False,
    pitc_stride: int = 1,
    pitc_quality_thresh: float = QUALITY_THRESH_DEFAULT,
    pitc_quality_metric: str = "stdv_from_mean",
    pitc_measure_resegment: bool = True,
    pitc_label_constrain: bool = True,
    save_plots: bool = False,
    backend: str = "gpu",
) -> tuple[StageSpec, ClusterPaths]:
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    parts = [
        *python_module_argv("nvitk.pipes.qvtpy.stage6_measure"),
        *sge_backend_cli_args(backend),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
        "--cross-section-radius-vox",
        str(float(cross_section_radius_vox)),
        "--cross-section-res",
        str(int(cross_section_res)),
        "--cross-section-plane-interp",
        str(int(cross_section_plane_interp)),
        "--measure-thr-algorithm",
        shlex.quote(str(measure_thr_algorithm)),
        "--pitc-stride",
        str(int(pitc_stride)),
        "--pitc-quality-thresh",
        str(float(pitc_quality_thresh)),
        "--pitc-quality-metric",
        shlex.quote(str(pitc_quality_metric)),
    ]
    if cs_supersampling:
        parts.append("--cs-supersampling")
    if skip_pitc:
        parts.append("--skip-pitc")
    if save_plots:
        parts.append("--save-plots")
    if skip_existing:
        parts.append("--skip-existing")
    if not measure_resegment:
        parts.append("--no-measure-resegment")
    if not pitc_measure_resegment:
        parts.append("--no-pitc-measure-resegment")
    if not pitc_label_constrain:
        parts.append("--no-pitc-label-constrain")
    python_cmd = " ".join(parts)
    paths = ClusterPaths(
        src=src_p,
        container=container,
        models=None,
        data_root=nifti_root,
        output_root=output_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    spec = StageSpec(
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage6_{subject}",
        python_cmd=python_cmd,
        resources=sge_qvtpy_stage_resources(backend),
        binds=binds,
        use_nv=sge_stage_use_nv(backend),
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    return spec, paths


def build_subject_sge_command(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    cross_section_radius_vox: float = _DEFAULT_RADIUS_VOX,
    measure_resegment: bool = False,
    measure_thr_algorithm: str = "lsthr",
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
    cs_supersampling: bool = False,
    skip_pitc: bool = False,
    pitc_stride: int = 1,
    pitc_quality_thresh: float = QUALITY_THRESH_DEFAULT,
    pitc_quality_metric: str = "stdv_from_mean",
    pitc_measure_resegment: bool = True,
    pitc_label_constrain: bool = True,
    save_plots: bool = False,
    backend: str = "gpu",
) -> str:
    """Return the host shell command for one stage6 array/SGE task."""
    from nvitk.cluster.sge import build_singularity_command

    spec, paths = _subject_sge_spec(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        cross_section_radius_vox=cross_section_radius_vox,
        measure_resegment=measure_resegment,
        measure_thr_algorithm=measure_thr_algorithm,
        cross_section_res=cross_section_res,
        cross_section_plane_interp=cross_section_plane_interp,
        cs_supersampling=cs_supersampling,
        skip_pitc=skip_pitc,
        pitc_stride=pitc_stride,
        pitc_quality_thresh=pitc_quality_thresh,
        pitc_quality_metric=pitc_quality_metric,
        pitc_measure_resegment=pitc_measure_resegment,
        pitc_label_constrain=pitc_label_constrain,
        save_plots=save_plots,
        backend=backend,
    )
    return build_singularity_command(spec, paths)


def submit_subject_sge(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    hold_jid: str | None = None,
    emit: TextIO | None = None,
    cross_section_radius_vox: float = _DEFAULT_RADIUS_VOX,
    measure_resegment: bool = False,
    measure_thr_algorithm: str = "lsthr",
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
    cs_supersampling: bool = False,
    skip_pitc: bool = False,
    pitc_stride: int = 1,
    pitc_quality_thresh: float = QUALITY_THRESH_DEFAULT,
    pitc_quality_metric: str = "stdv_from_mean",
    pitc_measure_resegment: bool = True,
    pitc_label_constrain: bool = True,
    save_plots: bool = False,
    backend: str = "gpu",
) -> str:
    """Emit or submit one stage-6 SGE job. Returns qsub job id."""
    spec, paths = _subject_sge_spec(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        cross_section_radius_vox=cross_section_radius_vox,
        measure_resegment=measure_resegment,
        measure_thr_algorithm=measure_thr_algorithm,
        cross_section_res=cross_section_res,
        cross_section_plane_interp=cross_section_plane_interp,
        cs_supersampling=cs_supersampling,
        skip_pitc=skip_pitc,
        pitc_stride=pitc_stride,
        pitc_quality_thresh=pitc_quality_thresh,
        pitc_quality_metric=pitc_quality_metric,
        pitc_measure_resegment=pitc_measure_resegment,
        pitc_label_constrain=pitc_label_constrain,
        save_plots=save_plots,
        backend=backend,
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage6-measure")
@backend_click_option()
@click.option("--subject", required=True)
@click.option("--nifti-root", type=click.Path(path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--cross-section-radius-vox", type=float, default=_DEFAULT_RADIUS_VOX, show_default=True)
@click.option(
    "--measure-resegment/--no-measure-resegment",
    default=False,
    show_default=True,
    help="Recompute in-plane segmentation at each LOC (default off).",
)
@click.option(
    "--measure-thr-algorithm",
    type=click.Choice(["lsthr", "lthr", "otsu"], case_sensitive=False),
    default="lsthr",
    show_default=True,
    help="In-plane threshold when --measure-resegment (ignored if resegment off).",
)
@click.option("--cross-section-res", type=int, default=0, show_default=True)
@click.option("--cross-section-plane-interp", type=int, default=1, show_default=True)
@click.option(
    "--cs-supersampling/--no-cs-supersampling",
    default=False,
    show_default=True,
    help="Supersample oblique cross-section grid (~4×); default is native voxel sampling.",
)
@click.option(
    "--skip-pitc/--no-skip-pitc",
    default=False,
    show_default=True,
    help="Skip dense PITC/PWV vessel hemodynamics (default: run).",
)
@click.option("--pitc-stride", type=int, default=1, show_default=True,
              help="Centerline sampling stride (voxels) for the dense PITC profile.")
@click.option(
    "--pitc-quality-thresh",
    type=float,
    default=QUALITY_THRESH_DEFAULT,
    show_default=True,
    help="Cross-section quality threshold (0-4) for PITC/PWV inclusion.",
)
@click.option(
    "--pitc-quality-metric",
    type=click.Choice(["stdv_from_mean", "waveform"], case_sensitive=False),
    default="stdv_from_mean",
    show_default=True,
    help="Per-station Q metric for PITC/PWV weighting.",
)
@click.option(
    "--pitc-measure-resegment/--no-pitc-measure-resegment",
    default=True,
    show_default=True,
    help="In-plane MAG+CD+VEL resegmentation at each PITC station (QVTplus-style).",
)
@click.option(
    "--pitc-label-constrain/--no-pitc-label-constrain",
    default=True,
    show_default=True,
    help="Intersect in-plane mask with seg_4dflow vessel label.",
)
@click.option(
    "--save-plots/--no-save-plots",
    default=False,
    show_default=True,
    help="Render paper-style PITC/PWV/flow figures + per-region PITC branch masks.",
)
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    cross_section_radius_vox: float,
    measure_resegment: bool,
    measure_thr_algorithm: str,
    cross_section_res: int,
    cross_section_plane_interp: int,
    cs_supersampling: bool,
    skip_pitc: bool,
    pitc_stride: int,
    pitc_quality_thresh: float,
    pitc_quality_metric: str,
    pitc_measure_resegment: bool,
    pitc_label_constrain: bool,
    save_plots: bool,
) -> None:
    Logger()
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        cross_section_radius_vox=cross_section_radius_vox,
        measure_resegment=measure_resegment,
        measure_thr_algorithm=measure_thr_algorithm.lower(),
        cross_section_res=cross_section_res,
        cross_section_plane_interp=cross_section_plane_interp,
        cs_supersampling=cs_supersampling,
        skip_pitc=skip_pitc,
        pitc_stride=pitc_stride,
        pitc_quality_thresh=pitc_quality_thresh,
        pitc_quality_metric=pitc_quality_metric.lower(),
        pitc_measure_resegment=pitc_measure_resegment,
        pitc_label_constrain=pitc_label_constrain,
        save_plots=save_plots,
    )


__all__ = ["main", "run_subject", "build_subject_sge_command", "submit_subject_sge"]


if __name__ == "__main__":
    main()
