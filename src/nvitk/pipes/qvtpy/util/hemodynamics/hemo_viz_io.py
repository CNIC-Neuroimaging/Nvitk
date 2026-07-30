"""Serialize / reload stage-6 PITC–PWV plot data and Napari geometry.

QC must display the hemodynamics figures and summary values that were written
by qvtpy stage 6, not a fresh GUI recompute.  When ``collect_plot_data`` runs,
stage 6 writes ``hemo_viz_bundle.json`` next to the CSVs; the GUI loads that
bundle (or reconstructs a PITC-only view from the CSVs + saved PNGs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.pipes.qvtpy.util.hemodynamics.vessel_hemodynamics import (
    RegionGeometryViz,
    StationGeometryViz,
    VesselGeometryViz,
    VesselHemodynamicsResult,
)

HEMO_VIZ_BUNDLE_NAME = "hemo_viz_bundle.json"

_ARRAY_TAG = "__ndarray__"
_NAN_TAG = "__nan__"
_INF_TAG = "__inf__"
_NINF_TAG = "__ninf__"


def _encode_scalar(x: Any) -> Any:
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v):
            return {_NAN_TAG: True}
        if math.isinf(v):
            return {_INF_TAG: True} if v > 0 else {_NINF_TAG: True}
        return v
    if x is None or isinstance(x, str):
        return x
    return x


def _decode_scalar(obj: Any) -> Any:
    if obj is None:
        return float("nan")
    if isinstance(obj, dict):
        if _NAN_TAG in obj:
            return float("nan")
        if _INF_TAG in obj:
            return float("inf")
        if _NINF_TAG in obj:
            return float("-inf")
        if _ARRAY_TAG in obj:
            shape = tuple(int(s) for s in (obj.get("shape") or []))
            dtype = obj.get("dtype") or "float64"
            raw = obj.get("data") or []
            vals = [_decode_scalar(v) for v in raw]
            arr = np.asarray(vals, dtype=dtype)
            if shape:
                arr = arr.reshape(shape)
            return arr
    return obj


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        arr = to_numpy(obj)
        flat = arr.reshape(-1)
        if np.issubdtype(arr.dtype, np.floating):
            data = []
            for v in flat:
                fv = float(v)
                if math.isnan(fv):
                    data.append(None)
                elif math.isinf(fv):
                    data.append({_INF_TAG: True} if fv > 0 else {_NINF_TAG: True})
                else:
                    data.append(fv)
        else:
            data = [_encode_scalar(v) for v in flat]
        return {
            _ARRAY_TAG: True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data": data,
        }
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return _encode_scalar(obj)


def _from_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        if _ARRAY_TAG in obj or any(t in obj for t in (_NAN_TAG, _INF_TAG, _NINF_TAG)):
            return _decode_scalar(obj)
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


def _station_to_dict(st: StationGeometryViz) -> dict[str, Any]:
    return {
        "vessel_id": int(st.vessel_id),
        "vessel_name": str(st.vessel_name),
        "station_index": int(st.station_index),
        "centerline_x": float(st.centerline_x),
        "centerline_y": float(st.centerline_y),
        "centerline_z": float(st.centerline_z),
        "distance_mm": float(st.distance_mm),
        "pi": float(st.pi),
        "quality": float(st.quality),
        "area_mm2": float(st.area_mm2),
        "used_for_pwv": bool(st.used_for_pwv),
        "pwv_weight_area": float(st.pwv_weight_area),
        "pwv_weight_quality": float(st.pwv_weight_quality),
        "pwv_xcor_time_s": float(st.pwv_xcor_time_s),
        "pwv_time_to_upstroke_s": float(st.pwv_time_to_upstroke_s),
        "pwv_bjornfoot_weighted_rms": float(st.pwv_bjornfoot_weighted_rms),
        "pwv_bjornfoot_delay_residual_s": float(st.pwv_bjornfoot_delay_residual_s),
        "pwv_bjornfoot_waveform_corr": float(st.pwv_bjornfoot_waveform_corr),
    }


def _station_from_dict(d: dict[str, Any]) -> StationGeometryViz:
    return StationGeometryViz(
        vessel_id=int(d["vessel_id"]),
        vessel_name=str(d["vessel_name"]),
        station_index=int(d["station_index"]),
        centerline_x=float(d["centerline_x"]),
        centerline_y=float(d["centerline_y"]),
        centerline_z=float(d["centerline_z"]),
        distance_mm=float(d["distance_mm"]),
        pi=float(d["pi"]),
        quality=float(d["quality"]),
        area_mm2=float(d["area_mm2"]),
        used_for_pwv=bool(d.get("used_for_pwv", False)),
        pwv_weight_area=float(d.get("pwv_weight_area", float("nan"))),
        pwv_weight_quality=float(d.get("pwv_weight_quality", float("nan"))),
        pwv_xcor_time_s=float(d.get("pwv_xcor_time_s", float("nan"))),
        pwv_time_to_upstroke_s=float(d.get("pwv_time_to_upstroke_s", float("nan"))),
        pwv_bjornfoot_weighted_rms=float(
            d.get("pwv_bjornfoot_weighted_rms", float("nan"))
        ),
        pwv_bjornfoot_delay_residual_s=float(
            d.get("pwv_bjornfoot_delay_residual_s", float("nan"))
        ),
        pwv_bjornfoot_waveform_corr=float(
            d.get("pwv_bjornfoot_waveform_corr", float("nan"))
        ),
    )


def _geometry_to_dict(region: RegionGeometryViz) -> dict[str, Any]:
    vessels = {}
    for name, ves in region.vessels.items():
        vessels[str(name)] = {
            "vessel_id": int(ves.vessel_id),
            "vessel_name": str(ves.vessel_name),
            "polyline_oriented": _to_jsonable(ves.polyline_oriented),
            "distance_offset_mm": float(ves.distance_offset_mm),
            "stations": [_station_to_dict(s) for s in ves.stations],
        }
    return {
        "region_id": str(region.region_id),
        "root_label": int(region.root_label),
        "root_polyline_oriented": _to_jsonable(region.root_polyline_oriented),
        "root_init_xyz": _to_jsonable(region.root_init_xyz),
        "root_fin_xyz": _to_jsonable(region.root_fin_xyz),
        "pitc_slope": float(region.pitc_slope),
        "pitc_intercept": float(region.pitc_intercept),
        "pitc_r2": float(region.pitc_r2),
        "pitc_n": int(region.pitc_n),
        "global_pi": float(region.global_pi),
        "quality_thresh": float(region.quality_thresh),
        "pwv_bjornfoot_m_s": region.pwv_bjornfoot_m_s,
        "pwv_fielding_m_s": region.pwv_fielding_m_s,
        "pwv_r_fielding": region.pwv_r_fielding,
        "pwv_n_stations": int(region.pwv_n_stations),
        "vessels": vessels,
        "excluded_segments": [_to_jsonable(s) for s in region.excluded_segments],
        "root_init_extra_xyz": [_to_jsonable(p) for p in region.root_init_extra_xyz],
    }


def _geometry_from_dict(d: dict[str, Any]) -> RegionGeometryViz:
    vessels: dict[str, VesselGeometryViz] = {}
    for name, vd in (d.get("vessels") or {}).items():
        poly = _from_jsonable(vd["polyline_oriented"])
        vessels[str(name)] = VesselGeometryViz(
            vessel_id=int(vd["vessel_id"]),
            vessel_name=str(vd["vessel_name"]),
            polyline_oriented=np.asarray(poly, dtype="float64"),
            distance_offset_mm=float(vd.get("distance_offset_mm") or 0.0),
            stations=[_station_from_dict(s) for s in (vd.get("stations") or [])],
        )
    extras = [
        np.asarray(_from_jsonable(p), dtype="float64").reshape(3)
        for p in (d.get("root_init_extra_xyz") or [])
    ]
    excluded = [
        np.asarray(_from_jsonable(s), dtype="float64")
        for s in (d.get("excluded_segments") or [])
    ]
    return RegionGeometryViz(
        region_id=str(d["region_id"]),
        root_label=int(d["root_label"]),
        root_polyline_oriented=np.asarray(
            _from_jsonable(d["root_polyline_oriented"]), dtype="float64"
        ),
        root_init_xyz=np.asarray(_from_jsonable(d["root_init_xyz"]), dtype="float64").reshape(
            3
        ),
        root_fin_xyz=np.asarray(_from_jsonable(d["root_fin_xyz"]), dtype="float64").reshape(3),
        pitc_slope=float(d.get("pitc_slope") or 0.0),
        pitc_intercept=float(d.get("pitc_intercept") or 0.0),
        pitc_r2=float(d.get("pitc_r2") or 0.0),
        pitc_n=int(d.get("pitc_n") or 0),
        global_pi=float(d.get("global_pi") or 0.0),
        quality_thresh=float(d.get("quality_thresh") or 2.5),
        pwv_bjornfoot_m_s=d.get("pwv_bjornfoot_m_s", ""),
        pwv_fielding_m_s=d.get("pwv_fielding_m_s", ""),
        pwv_r_fielding=d.get("pwv_r_fielding", ""),
        pwv_n_stations=int(d.get("pwv_n_stations") or 0),
        vessels=vessels,
        excluded_segments=excluded,
        root_init_extra_xyz=extras,
    )


def _normalize_vessel_waveforms(region: dict[str, Any]) -> None:
    """JSON object keys are strings; restore integer vessel labels."""
    wf = region.get("vessel_waveforms")
    if not isinstance(wf, dict):
        return
    fixed: dict[int, Any] = {}
    for k, v in wf.items():
        try:
            lid = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            for arr_key in ("mean", "std"):
                if arr_key in v and not isinstance(v[arr_key], np.ndarray):
                    v[arr_key] = np.asarray(v[arr_key], dtype="float64")
        fixed[lid] = v
    region["vessel_waveforms"] = fixed


def write_hemo_viz_bundle(out_dir: Path, hemo: VesselHemodynamicsResult) -> Path | None:
    """Write ``hemo_viz_bundle.json`` for QC reload. Returns path or None."""
    if not hemo.region_plot_data:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "region_plot_data": _to_jsonable(hemo.region_plot_data),
        "geometry_by_region": _to_jsonable(
            {rid: _geometry_to_dict(geom) for rid, geom in hemo.geometry_by_region.items()}
        ),
    }
    path = out_dir / HEMO_VIZ_BUNDLE_NAME
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def load_hemo_viz_bundle(
    stage6_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, RegionGeometryViz]] | None:
    """Load plot data + geometry from a stage-6 directory, or None if missing."""
    path = Path(stage6_dir) / HEMO_VIZ_BUNDLE_NAME
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    plot_data = _from_jsonable(raw.get("region_plot_data") or {})
    if not isinstance(plot_data, dict) or not plot_data:
        return None
    for region in plot_data.values():
        if isinstance(region, dict):
            _normalize_vessel_waveforms(region)
    geometry: dict[str, RegionGeometryViz] = {}
    geom_raw = _from_jsonable(raw.get("geometry_by_region") or {})
    if isinstance(geom_raw, dict):
        for rid, gd in geom_raw.items():
            if isinstance(gd, dict):
                geometry[str(rid)] = _geometry_from_dict(gd)
    return plot_data, geometry


def reconstruct_region_plot_data_from_csvs(
    stage6_dir: Path,
    *,
    quality_thresh: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Build PITC-capable ``region_plot_data`` from stage-6 CSVs (no PWV timing)."""
    import csv

    stage6_dir = Path(stage6_dir)
    profile_path = stage6_dir / "pitc_profile.csv"
    hemo_path = stage6_dir / "vessel_hemodynamics.csv"
    if not profile_path.is_file() or not hemo_path.is_file():
        return {}

    meta_path = stage6_dir / "measure_meta.json"
    q_thresh = float(quality_thresh) if quality_thresh is not None else 2.5
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            q_thresh = float(meta.get("pitc_quality_thresh") or q_thresh)
        except Exception:  # noqa: BLE001
            pass

    roots: dict[str, dict[str, Any]] = {}
    with hemo_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("row_kind") or "") != "root":
                continue
            rid = str(row.get("region_id") or "").strip()
            if not rid:
                continue

            def _f(key: str, default: float = float("nan")) -> float:
                raw = row.get(key, "")
                if raw is None or str(raw).strip() == "":
                    return default
                try:
                    return float(raw)
                except ValueError:
                    return default

            def _opt(key: str) -> Any:
                raw = row.get(key, "")
                if raw is None or str(raw).strip() == "":
                    return ""
                try:
                    return float(raw)
                except ValueError:
                    return str(raw)

            roots[rid] = {
                "region_id": rid,
                "pitc_slope": _f("pitc_slope"),
                "pitc_intercept": _f("pitc_intercept"),
                "global_pi": _f("global_pi"),
                "quality_thresh": q_thresh,
                "pwv_bjornfoot_m_s": _opt("pwv_bjornfoot_m_s"),
                "pwv_fielding_m_s": _opt("pwv_fielding_m_s"),
                "distance_mm": [],
                "pi": [],
                "quality": [],
            }

    with profile_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rid = str(row.get("root_region_id") or "").strip()
            if rid not in roots:
                continue
            try:
                roots[rid]["distance_mm"].append(float(row["distance_mm"]))
                roots[rid]["pi"].append(float(row["pi"]))
                roots[rid]["quality"].append(float(row["quality"]))
            except (KeyError, TypeError, ValueError):
                continue

    out: dict[str, dict[str, Any]] = {}
    for rid, data in roots.items():
        out[rid] = {
            "region_id": rid,
            "distance_mm": np.asarray(data["distance_mm"], dtype="float64"),
            "pi": np.asarray(data["pi"], dtype="float64"),
            "quality": np.asarray(data["quality"], dtype="float64"),
            "pitc_slope": data["pitc_slope"],
            "pitc_intercept": data["pitc_intercept"],
            "global_pi": data["global_pi"],
            "quality_thresh": data["quality_thresh"],
            "pwv_bjornfoot_m_s": data["pwv_bjornfoot_m_s"],
            "pwv_fielding_m_s": data["pwv_fielding_m_s"],
            # Empty PWV arrays → interactive PWV tabs fall back to saved PNGs.
            "pwv_distance_mm": np.asarray([], dtype="float64"),
            "pwv_xcor_time_s": np.asarray([], dtype="float64"),
            "pwv_time_to_upstroke_s": np.asarray([], dtype="float64"),
            "pwv_excluded_distance_mm": np.asarray([], dtype="float64"),
            "pwv_excluded_xcor_time_s": np.asarray([], dtype="float64"),
            "pwv_excluded_time_to_upstroke_s": np.asarray([], dtype="float64"),
            "pwv_weight_area": np.asarray([], dtype="float64"),
            "pwv_weight_quality": np.asarray([], dtype="float64"),
            "pwv_bjornfoot_weighted_rms": np.asarray([], dtype="float64"),
            "pwv_bjornfoot_delay_residual_s": np.asarray([], dtype="float64"),
            "pwv_bjornfoot_waveform_corr": np.asarray([], dtype="float64"),
            "vessel_waveforms": {},
        }
    return out


def stage6_saved_plot_paths(stage6_dir: Path) -> dict[str, Path]:
    """Map plot kind → PNG path under ``stage6/plots/`` when present."""
    plots = Path(stage6_dir) / "plots"
    mapping = {
        "pitc": plots / "pitc.png",
        "pwv": plots / "pwv.png",
        "bjornfoot": plots / "pwv_bjornfoot_qc.png",
    }
    return {k: p for k, p in mapping.items() if p.is_file()}


def load_measure_meta(stage6_dir: Path) -> dict[str, Any]:
    """Load ``measure_meta.json`` if present."""
    path = Path(stage6_dir) / "measure_meta.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def hemo_params_from_measure_meta(stage6_dir: Path) -> dict[str, Any]:
    """GUI / compute kwargs matching the stage-6 run that wrote *stage6_dir*."""
    meta = load_measure_meta(stage6_dir)
    return {
        "stride": int(meta.get("pitc_stride") or 1),
        "cross_section_radius_vox": float(meta.get("cross_section_radius_vox") or 10.0),
        "quality_thresh": float(meta.get("pitc_quality_thresh") or 2.5),
        "quality_metric": str(meta.get("pitc_quality_metric") or "stdv_from_mean"),
        "measure_resegment": bool(meta.get("pitc_measure_resegment", False)),
        "label_constrain": bool(meta.get("pitc_label_constrain", True)),
        "thr_algorithm": str(meta.get("measure_thr_algorithm") or "otsu"),
        "cross_section_plane_interp": int(meta.get("cross_section_plane_interp") or 1),
        "cs_supersampling": bool(meta.get("cs_supersampling", True)),
        "temporal_resolution_s": meta.get("temporal_resolution_s"),
        "station_point_size": 2.5,
        "root_region": "All",
    }


def load_saved_root_summaries(stage6_dir: Path) -> dict[str, dict[str, Any]]:
    """Root-row PITC/PWV values from ``vessel_hemodynamics.csv`` (pipeline truth)."""
    import csv

    path = Path(stage6_dir) / "vessel_hemodynamics.csv"
    if not path.is_file():
        return {}

    def _opt_float(raw: Any) -> Any:
        if raw is None or str(raw).strip() == "":
            return ""
        try:
            return float(raw)
        except (TypeError, ValueError):
            return str(raw)

    def _float_or(raw: Any, default: float = float("nan")) -> float:
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    out: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("row_kind") or "") != "root":
                continue
            rid = str(row.get("region_id") or "").strip()
            if not rid:
                continue
            out[rid] = {
                "pitc_slope": _float_or(row.get("pitc_slope")),
                "pitc_intercept": _float_or(row.get("pitc_intercept")),
                "pitc_r2": _float_or(row.get("pitc_r2")),
                "pitc_n": int(_float_or(row.get("pitc_n"), 0.0)),
                "global_pi": _float_or(row.get("global_pi")),
                "pwv_bjornfoot_m_s": _opt_float(row.get("pwv_bjornfoot_m_s")),
                "pwv_fielding_m_s": _opt_float(row.get("pwv_fielding_m_s")),
                "pwv_r_fielding": _opt_float(row.get("pwv_r_fielding")),
                "pwv_n_stations": int(_float_or(row.get("pwv_n_stations"), 0.0)),
            }
    return out


def apply_saved_hemo_summaries(
    plot_data: dict[str, dict[str, Any]],
    regions: list[RegionGeometryViz],
    summaries: dict[str, dict[str, Any]],
    *,
    quality_thresh: float | None = None,
) -> None:
    """Overwrite reported slopes / PWV with pipeline CSV values (in place)."""
    if not summaries:
        return
    for rid, data in plot_data.items():
        saved = summaries.get(str(rid))
        if not saved or not isinstance(data, dict):
            continue
        for key in (
            "pitc_slope",
            "pitc_intercept",
            "global_pi",
            "pwv_bjornfoot_m_s",
            "pwv_fielding_m_s",
        ):
            if key in saved:
                data[key] = saved[key]
        if quality_thresh is not None:
            data["quality_thresh"] = float(quality_thresh)

    for region in regions:
        saved = summaries.get(str(region.region_id))
        if not saved:
            continue
        region.pitc_slope = float(saved.get("pitc_slope", region.pitc_slope) or 0.0)
        region.pitc_intercept = float(
            saved.get("pitc_intercept", region.pitc_intercept) or 0.0
        )
        region.pitc_r2 = float(saved.get("pitc_r2", region.pitc_r2) or 0.0)
        region.pitc_n = int(saved.get("pitc_n", region.pitc_n) or 0)
        region.global_pi = float(saved.get("global_pi", region.global_pi) or 0.0)
        region.pwv_bjornfoot_m_s = saved.get(
            "pwv_bjornfoot_m_s", region.pwv_bjornfoot_m_s
        )
        region.pwv_fielding_m_s = saved.get(
            "pwv_fielding_m_s", region.pwv_fielding_m_s
        )
        region.pwv_r_fielding = saved.get("pwv_r_fielding", region.pwv_r_fielding)
        region.pwv_n_stations = int(
            saved.get("pwv_n_stations", region.pwv_n_stations) or 0
        )
        if quality_thresh is not None:
            region.quality_thresh = float(quality_thresh)


def plot_data_has_interactive_pwv(plot_data: dict[str, dict[str, Any]]) -> bool:
    """True when PWV timing arrays are present for diagnostics controls."""
    for data in plot_data.values():
        if not isinstance(data, dict):
            continue
        if np.asarray(data.get("pwv_distance_mm", [])).size:
            return True
    return False


def load_stage6_hemo_for_qc(
    stage6_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[RegionGeometryViz], dict[str, Path]]:
    """Load saved hemodynamics for QC: plot data, geometry list, PNG fallbacks.

    Prefers ``hemo_viz_bundle.json``; otherwise reconstructs PITC-only data from
    CSVs. Callers that need full interactive PWV/station controls should recompute
    with :func:`~nvitk.pipes.qvtpy.util.hemodynamics.vessel_hemodynamics.compute_vessel_hemodynamics`
    and then :func:`apply_saved_hemo_summaries`.
    """
    stage6_dir = Path(stage6_dir)
    pngs = stage6_saved_plot_paths(stage6_dir)
    summaries = load_saved_root_summaries(stage6_dir)
    meta = load_measure_meta(stage6_dir)
    q_thresh = meta.get("pitc_quality_thresh")
    quality_thresh = float(q_thresh) if q_thresh is not None else None
    bundle = load_hemo_viz_bundle(stage6_dir)
    if bundle is not None:
        plot_data, geometry_map = bundle
        regions = list(geometry_map.values())
        apply_saved_hemo_summaries(
            plot_data, regions, summaries, quality_thresh=quality_thresh
        )
        return plot_data, regions, pngs
    plot_data = reconstruct_region_plot_data_from_csvs(stage6_dir)
    apply_saved_hemo_summaries(
        plot_data, [], summaries, quality_thresh=quality_thresh
    )
    return plot_data, [], pngs


__all__ = [
    "HEMO_VIZ_BUNDLE_NAME",
    "apply_saved_hemo_summaries",
    "hemo_params_from_measure_meta",
    "load_hemo_viz_bundle",
    "load_measure_meta",
    "load_saved_root_summaries",
    "load_stage6_hemo_for_qc",
    "plot_data_has_interactive_pwv",
    "reconstruct_region_plot_data_from_csvs",
    "stage6_saved_plot_paths",
    "write_hemo_viz_bundle",
]
