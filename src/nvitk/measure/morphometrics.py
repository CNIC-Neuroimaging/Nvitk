"""TOF Circle-of-Willis morphometrics (cow_morpho port, nvitk-native centerlines).

Public entry points for stage-7 and external callers. Core algorithms live under
:mod:`nvitk.measure.morpho` with configuration in :mod:`nvitk.measure.morphometrics_config`.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np

from nvitk.measure.morpho.export_utils.compute_tortuosity_metrics import run as run_tortuosity_metrics
from nvitk.measure.morpho.export_utils.generate_radius_histograms import run as run_radius_histograms
from nvitk.measure.morpho.preprocess_taubin import (
    load_segmentation,
    preprocess_segmentation,
    save_segmentation,
)
from nvitk.measure.morpho.run_case import N_WORKERS, run_case
from nvitk.measure.morpho.topology_io import (
    TOPOLOGY_NONE,
    default_qvtpy_topology_path,
    load_topology,
    resolve_topology_path,
)
from nvitk.measure.morphometrics_config import MorphometricsConfig, default_morphometrics_config

__all__ = [
    "MorphometricsConfig",
    "default_morphometrics_config",
    "run_case",
    "run_morphometrics_case",
]


def _smooth_taubin(
    seg_path: Path,
    out_dir: Path,
    *,
    taubin_iters: int,
    taubin_lambda: float,
    taubin_mu: float,
    keep_largest: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = seg_path.name.replace(".nii.gz", "").replace(".nii", "")
    output_path = out_dir / f"{stem}_taubin.nii.gz"
    if output_path.is_file():
        return output_path

    seg, affine, header = load_segmentation(str(seg_path))
    spacing = header.get_zooms()[:3]
    smoothed = preprocess_segmentation(
        seg=seg,
        keep_largest=keep_largest,
        taubin_iters=taubin_iters,
        taubin_lambda=taubin_lambda,
        taubin_mu=taubin_mu,
    )
    save_segmentation(str(output_path), smoothed, affine, header)

    report = {
        "input": str(seg_path),
        "output": str(output_path),
        "spacing_mm": [float(x) for x in spacing],
        "taubin_iters": int(taubin_iters),
        "taubin_lambda": float(taubin_lambda),
        "taubin_mu": float(taubin_mu),
        "keep_largest_component": bool(keep_largest),
        "labels": [int(x) for x in np.unique(seg) if x != 0],
    }
    report_path = out_dir / f"{stem}_taubin_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def _resolve_mapping(
    *,
    mapping: dict | None,
    mapping_json: str | None,
) -> tuple[dict | None, str | None]:
    """Resolve topology mapping for morphometrics.

    Returns ``(mapping_dict_or_empty, mapping_json_path_or_None)``.
    An empty dict means vessel-wise only (no topology awareness).
    """
    if mapping is not None:
        return mapping, mapping_json

    token = None if mapping_json is None else str(mapping_json).strip()
    if token is not None and token.lower() == TOPOLOGY_NONE:
        return {}, None
    if token is not None and token != "":
        path = resolve_topology_path(token)
        if path is None:
            return {}, None
        return load_topology(path) or {}, str(path)

    # Default: qvtpy / eICAB topology JSON (same content as former coded mapping).
    path = default_qvtpy_topology_path()
    return load_topology(path) or {}, str(path)


def run_morphometrics_case(
    seg_path: str | Path,
    out_dir: str | Path,
    *,
    mapping: dict | None = None,
    mapping_json: str | None = None,
    case_out_dir_override: str | Path | None = None,
    n_workers: int | None = None,
    config: MorphometricsConfig | None = None,
    input_already_smoothed: bool = False,
    skip_if_excel_exists: bool = False,
) -> Path:
    """Run full TOF morphometrics for one multilabel segmentation NIfTI.

    Topology resolution order:
    1. Explicit ``mapping`` dict
    2. ``mapping_json`` path / basename under ``measure/morpho/topology/``
    3. ``mapping_json="none"`` → vessel-wise only (no topology)
    4. Default ``qvtpy_topology.json``

    Returns path to ``case_metrics_donut_tree.xlsx``.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    cfg = config or default_morphometrics_config()
    seg_p = Path(seg_path).expanduser().resolve()
    out_p = Path(out_dir).expanduser().resolve()
    case_dir = Path(case_out_dir_override).expanduser().resolve() if case_out_dir_override else out_p
    case_dir.mkdir(parents=True, exist_ok=True)

    excel_path = case_dir / "case_metrics_donut_tree.xlsx"
    if skip_if_excel_exists and excel_path.is_file():
        pipeline_skipped = True
    else:
        pipeline_skipped = False

    topo, topo_json = _resolve_mapping(mapping=mapping, mapping_json=mapping_json)
    workers = int(n_workers) if n_workers is not None else (cfg.n_workers or N_WORKERS)

    if not pipeline_skipped:
        if input_already_smoothed:
            pipeline_seg = seg_p
        else:
            pipeline_seg = _smooth_taubin(
                seg_p,
                case_dir,
                taubin_iters=cfg.taubin_iters,
                taubin_lambda=cfg.taubin_lambda,
                taubin_mu=cfg.taubin_mu,
                keep_largest=cfg.keep_largest_component_taubin,
            )

        run_case(
            seg_path=str(pipeline_seg),
            out_dir=str(case_dir.parent),
            mapping_json=topo_json,
            mapping=topo if topo is not None else {},
            case_out_dir_override=str(case_dir),
            n_workers=workers,
        )

    if cfg.run_tortuosity:
        centerlines_dir = case_dir / "centerlines"
        if centerlines_dir.is_dir():
            try:
                run_tortuosity_metrics(
                    input_path=str(centerlines_dir),
                    output_csv=str(centerlines_dir / "tortuosity_metrics.csv"),
                    output_xlsx=str(centerlines_dir / "tortuosity_metrics.xlsx"),
                    save_pointwise_sheets=False,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING [tortuosity]: {exc}")
                traceback.print_exc()

    if cfg.run_histograms and excel_path.is_file():
        try:
            run_radius_histograms(
                excel_path=str(excel_path),
                output_dir=str(case_dir / "radius_histograms"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING [histograms]: {exc}")
            traceback.print_exc()

    if not excel_path.is_file():
        raise FileNotFoundError(f"Morphometrics Excel not produced: {excel_path}")
    return excel_path
