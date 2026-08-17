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

from nvitk.measure.morpho.anatomy_axes import SPECIES_CHOICES, normalize_species
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
    TopologyMeta,
    default_eicab_topology_path,
    load_topology,
    load_topology_meta,
    resolve_topology_path,
)
from nvitk.measure.morphometrics_config import MorphometricsConfig, default_morphometrics_config

__all__ = [
    "SPECIES_CHOICES",
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
    """Run Taubin smoothing (:mod:`measure.morpho.preprocess_taubin`) on one segmentation and write the result."""
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

    labels = [int(x) for x in np.unique(seg) if x != 0]
    voxel_mm3 = float(np.prod(np.asarray(spacing, dtype=float)))
    report = {
        "input": str(seg_path),
        "output": str(output_path),
        "spacing_mm": [float(x) for x in spacing],
        "voxel_volume_mm3": voxel_mm3,
        "taubin_iters": int(taubin_iters),
        "taubin_lambda": float(taubin_lambda),
        "taubin_mu": float(taubin_mu),
        "keep_largest_component": bool(keep_largest),
        "labels": labels,
        # Raw vs smoothed volume per label: the morphometrics workbook measures
        # the smoothed mask, so this is where the difference is recorded.
        "raw_volume_mm3": {
            str(l): float(np.count_nonzero(seg == l)) * voxel_mm3 for l in labels
        },
        "smoothed_volume_mm3": {
            str(l): float(np.count_nonzero(smoothed == l)) * voxel_mm3 for l in labels
        },
    }
    report_path = out_dir / f"{stem}_taubin_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def _resolve_mapping(
    *,
    mapping: dict | None,
    mapping_json: str | None,
) -> tuple[dict | None, str | None, TopologyMeta]:
    """Resolve topology mapping for morphometrics.

    Returns ``(mapping_dict_or_empty, mapping_json_path_or_None, topology_meta)``.
    An empty dict means vessel-wise only (no topology awareness); the meta then
    carries its defaults (human, unscaled).
    """
    if mapping is not None:
        return mapping, mapping_json, load_topology_meta(mapping_json)

    token = None if mapping_json is None else str(mapping_json).strip()
    if token is not None and token.lower() == TOPOLOGY_NONE:
        return {}, None, TopologyMeta()
    if token is not None and token != "":
        path = resolve_topology_path(token)
        if path is None:
            return {}, None, TopologyMeta()
        return load_topology(path) or {}, str(path), load_topology_meta(path)

    # Default: eICAB topology (TOF label IDs). Not the 4D-flow qvtpy_topology.json.
    path = default_eicab_topology_path()
    return load_topology(path) or {}, str(path), load_topology_meta(path)


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
    species: str | None = None,
    axes_override: str | None = None,
    length_scale: float | None = None,
) -> Path:
    """Run full TOF morphometrics for one multilabel segmentation NIfTI.

    Topology resolution order:
    1. Explicit ``mapping`` dict
    2. ``mapping_json`` path / basename under ``measure/morpho/topology/``
    3. ``mapping_json="none"`` → vessel-wise only (no topology)
    4. Default ``eicab_topology.json``

    Species resolution order (drives how ``no_upstream_start`` rules resolve onto
    array axes — a mouse is a quadruped, so ``caudal`` lands on scanner A/P
    rather than S/I):
    1. Explicit ``species`` argument (``"human"`` / ``"mouse"``)
    2. ``config.species`` when it is not ``"auto"``
    3. The topology JSON's ``_meta.species``
    4. ``"human"``

    ``axes_override`` (axis codes such as ``"LSA"``) and ``length_scale`` follow
    the same order, defaulting to the topology's ``_meta``.

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

    topo, topo_json, topo_meta = _resolve_mapping(mapping=mapping, mapping_json=mapping_json)
    workers = int(n_workers) if n_workers is not None else (cfg.n_workers or N_WORKERS)

    requested_species = species if species is not None else cfg.species
    resolved_species = normalize_species(requested_species, fallback=topo_meta.species)
    resolved_axes_override = (
        axes_override
        if axes_override is not None
        else (cfg.axes_override if cfg.axes_override is not None else topo_meta.axes_override)
    )
    resolved_length_scale = float(
        length_scale if length_scale is not None else topo_meta.length_scale
    )

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
            species=resolved_species,
            axes_override=resolved_axes_override,
            length_scale=resolved_length_scale,
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
