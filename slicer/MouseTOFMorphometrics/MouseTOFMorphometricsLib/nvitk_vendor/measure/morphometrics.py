# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/measure/morphometrics.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""TOF Circle-of-Willis morphometrics (cow_morpho port).

Public entry points for stage-7 and external callers. Core algorithms live under
:mod:`nvitk_vendor.measure.morpho`; parameters are named, mask-specific profiles in
:mod:`nvitk_vendor.measure.morphometrics_config` (default ``eicab_cw``: eICAB Circle-of-Willis masks,
cow_morpho's tuned values, VMTK centerlines).
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np

from nvitk_vendor.measure.morpho.anatomy_axes import SPECIES_CHOICES, normalize_species
from nvitk_vendor.measure.morpho.export_utils.compute_tortuosity_metrics import run as run_tortuosity_metrics
from nvitk_vendor.measure.morpho.export_utils.generate_radius_histograms import run as run_radius_histograms
from nvitk_vendor.measure.morpho.preprocess_taubin import (
    load_segmentation,
    preprocess_segmentation,
    save_segmentation,
)
from nvitk_vendor.measure.morpho.run_case import N_WORKERS, run_case
from nvitk_vendor.measure.morpho.topology_io import (
    TOPOLOGY_NONE,
    TopologyMeta,
    default_eicab_topology_path,
    load_topology,
    load_topology_meta,
    resolve_topology_path,
)
from nvitk_vendor.measure.morphometrics_config import (
    CENTERLINE_BACKENDS,
    DEFAULT_PROFILE,
    PROFILES,
    MorphometricsConfig,
    default_morphometrics_config,
    morphometrics_config,
)

__all__ = [
    "CENTERLINE_BACKENDS",
    "PROFILES",
    "SPECIES_CHOICES",
    "MorphometricsConfig",
    "default_morphometrics_config",
    "morphometrics_config",
    "run_case",
    "run_morphometrics_case",
]

PROVENANCE_FILE = "morphometrics_provenance.json"


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


def _versions() -> dict:
    """Versions of the libraries that shape the numbers (for the provenance file)."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("nvitk", "vmtk", "vtk", "numpy", "scipy", "scikit-image"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = ""
    return out


def _git_revision() -> str:
    """Short git revision of the nvitk_vendor source tree (``""`` outside a checkout)."""
    import subprocess

    try:
        res = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return res.stdout.strip() if res.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def _write_provenance(case_dir: Path, *, seg_path: Path, pipeline_seg: Path, cfg: MorphometricsConfig,
                      topology_json: str | None, species: str, input_already_smoothed: bool) -> None:
    """Record what produced a case's outputs: profile, parameters, backend, inputs, versions."""
    params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.params.items()}
    payload = {
        "profile": cfg.profile,
        "centerline_backend": cfg.centerline_backend,
        "input_mask": str(seg_path),
        "pipeline_input": str(pipeline_seg),
        "input_already_smoothed": bool(input_already_smoothed),
        "taubin": {"iters": cfg.taubin_iters, "lambda": cfg.taubin_lambda, "mu": cfg.taubin_mu,
                   "keep_largest_component": cfg.keep_largest_component_taubin},
        "topology": topology_json,
        "species": species,
        "versions": _versions(),
        "git_revision": _git_revision(),
        "parameters": params,
    }
    (case_dir / PROVENANCE_FILE).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_morphometrics_case(
    seg_path: str | Path,
    out_dir: str | Path,
    *,
    profile: str | None = None,
    config: MorphometricsConfig | None = None,
    centerline_backend: str | None = None,
    mapping: dict | None = None,
    mapping_json: str | None = None,
    case_out_dir_override: str | Path | None = None,
    n_workers: int | None = None,
    input_already_smoothed: bool = False,
    skip_if_excel_exists: bool = False,
    species: str | None = None,
    axes_override: str | None = None,
    length_scale: float | None = None,
    case_id: str | None = None,
) -> Path:
    """Run full TOF morphometrics for one multilabel segmentation NIfTI.

    *case_id* labels the output rows (default: the smoothed segmentation's file stem).

    Configuration: *config*, else the named *profile* (default ``eicab_cw``);
    *centerline_backend* (``vmtk`` / ``native`` / ``raycast``) overrides the profile's backend.

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

    Returns path to ``case_metrics_donut_tree.xlsx``; the compact tables (``vessel_summary.csv``,
    ``segments.csv``, ``lesions.csv``, ``subject_summary.csv``) and ``morphometrics_provenance.json``
    are written next to it.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    cfg = config or morphometrics_config(profile or DEFAULT_PROFILE)
    if centerline_backend:
        cfg = cfg.with_params(CENTERLINE_BACKEND=str(centerline_backend))
        if cfg.centerline_backend not in CENTERLINE_BACKENDS:
            raise ValueError(f"centerline_backend must be one of {CENTERLINE_BACKENDS}")
    seg_p = Path(seg_path).expanduser().resolve()
    out_p = Path(out_dir).expanduser().resolve()
    case_dir = Path(case_out_dir_override).expanduser().resolve() if case_out_dir_override else out_p
    case_dir.mkdir(parents=True, exist_ok=True)

    excel_path = case_dir / "case_metrics_donut_tree.xlsx"
    pipeline_skipped = bool(skip_if_excel_exists and excel_path.is_file())

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
            config=cfg,
            case_id=case_id,
        )
        _write_provenance(
            case_dir, seg_path=seg_p, pipeline_seg=pipeline_seg, cfg=cfg, topology_json=topo_json,
            species=resolved_species, input_already_smoothed=input_already_smoothed,
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
    # The final report, last: it reads the tortuosity metrics written just above.
    try:
        from nvitk_vendor.measure.morpho.export_utils.report import REPORT_FEATURES, write_morphometrics_report

        report_csv = write_morphometrics_report(case_dir)
        print(f"  [report] {report_csv.name} (+ 'Report' sheet) → {case_dir}")
        prov = case_dir / PROVENANCE_FILE
        if prov.is_file():
            payload = json.loads(prov.read_text(encoding="utf-8"))
            payload["report"] = {"file": report_csv.name, "features": list(REPORT_FEATURES),
                                 "lesions": "high confidence only"}
            prov.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — the run's tables are written; the report is a summary
        print(f"  WARNING [report]: {exc}")
        traceback.print_exc()
    return excel_path
