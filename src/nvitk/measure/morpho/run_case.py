"""Parallel execution layer for the refactor_clean pipeline.

Each label/component is an independent job dispatched to a subprocess worker.
Within a job, root-to-terminal paths are still processed serially because VMTK
validation uses already-accepted centerlines from the same vessel as spatial
references.

VTK/VMTK are not reliably fork-safe, so workers are spawned (clean interpreter)
rather than forked.  Lower N_WORKERS if VMTK becomes unstable.
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

import nvitk.measure.morphometrics_config as _config
from nvitk.measure.morphometrics_config import MorphometricsConfig, active_config, apply_config
from nvitk.measure.morpho.anatomy_axes import MorphoContext, resolve_anatomical_axes
from nvitk.measure.morpho.io_utils import (
    clean_generated_vtp_dir,
    clean_legacy_centerline_vtp_names,
    ensure_dir,
    load_mapping,
    load_multilabel_nifti,
    safe_filename,
    safe_sheet_name,
    vessel_sheet_sort_key,
)
from nvitk.measure.morpho.caliber import set_extra_detection_junctions
from nvitk.measure.morpho.labels_util import (
    split_partner_junctions,
    split_partner_labels,
    connected_components,
    empty_vessel_info,
    keep_largest_component,
    keep_largest_component_per_label,
    resolve_labels_to_process,
)
from nvitk.measure.morpho.orchestration import process_component_tree_vmtk
from nvitk.measure.morpho.surface import set_world_transform
from nvitk.measure.morpho.export_utils.vessel_tables import (
    add_hypoplasia_columns,
    component_tables,
    fetal_pca_rows,
    hypoplasia_rows,
    locate_a1_hypoplasia,
    subject_summary,
    vessel_summary,
)
from nvitk.measure.morpho.export_utils.summaries import (
    branchpoint_export_dataframe,
    compute_volumetry_summary,
    compute_hemispheric_summary,
    compute_lr_asymmetry,
    compute_vessel_summary_row,
    path_summary_export_dataframe,
    recursive_tree_segment_summary_rows,
    tree_summary_export_dataframe,
)

# Conservative default: VMTK + VTK are memory-hungry.
N_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))


def _headless() -> None:
    """Force a non-interactive matplotlib backend in each worker (no display available)."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass


# Worker (must be a top-level function to be picklable for spawn)

def _worker(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Subprocess entry point: run the full VMTK morphometrics pipeline for one label/component.

    Must stay a top-level function (picklable) since workers are spawned, not
    forked: VTK/VMTK are not reliably fork-safe. Catches all exceptions and
    returns them as a result dict (``ok=False``) so one failed job doesn't kill
    the pool.
    """
    _headless()
    (
        label, component_id, mask_cc,
        multilabel, spacing, vessel_info, mapping,
        dirs, case_id, ctx,
    ) = args
    # Spawned workers start from the module defaults: install this run's profile and output frame.
    if ctx.config is not None:
        apply_config(ctx.config)
    set_world_transform(ctx.affine, spacing)

    try:
        print(f"\n=== Label {label} / component {component_id} ===")
        print(f"  Component voxels: {int(mask_cc.sum())}")
        # A split vessel (M1 / M2, A1 / A2, P1 / P2): the bifurcations it shares with its other
        # half, from the merged mask, bound its lesion measures like its own junctions.
        partners = split_partner_labels(vessel_info.name, mapping)
        extra_junctions = split_partner_junctions(mask_cc, multilabel, partners, spacing)
        set_extra_detection_junctions(extra_junctions)
        if len(extra_junctions):
            print(f"  Junctions shared with {partners} (merged mask): {len(extra_junctions)}")
        (
            path_results, point_sheets, tree_summary, branch_df,
            region_summaries, region_sheets, recursive_segments, donut_loop_df,
            split_results,
        ) = process_component_tree_vmtk(
            label=label,
            component_id=component_id,
            mask_cc=mask_cc,
            multilabel=multilabel,
            spacing=spacing,
            vessel_info=vessel_info,
            mapping=mapping,
            centerline_dir=dirs["centerline_dir"] if _config.SAVE_CENTERLINES else None,
            centerline_radius_dir=dirs["centerline_radius_dir"] if _config.SAVE_CENTERLINES and _config.SAVE_CENTERLINE_RADIUS else None,
            region_centerline_dir=None,
            surface_dir=dirs["surface_dir"] if _config.SAVE_SURFACES else None,
            ctx=ctx,
        )

        # Non-overlapping rows are the primary output; the measured root->terminal
        # paths re-traverse shared trunks and are kept only for traceability.
        path_summaries = []
        for res in split_results:
            res["case_id"] = case_id
            path_summaries.append(compute_vessel_summary_row(case_id, label, vessel_info, res))

        root_to_terminal_summaries = []
        for res in path_results:
            res["case_id"] = case_id
            root_to_terminal_summaries.append(
                compute_vessel_summary_row(case_id, label, vessel_info, res)
            )

        tree_summary["case_id"] = case_id
        for row in region_summaries:
            row["case_id"] = case_id

        seg_summaries = recursive_tree_segment_summary_rows(
            case_id=case_id, label=label, component_id=component_id,
            vessel_info=vessel_info, segments=recursive_segments,
        )

        compact = component_tables(
            case_id=case_id, label=label, component_id=component_id, vessel_info=vessel_info,
            path_results=path_results, split_results=split_results, branch_df=branch_df,
            tree_summary=tree_summary, axes=ctx.axes, extra_junctions=extra_junctions,
        )

        if not branch_df.empty:
            branch_df.insert(0, "case_id", case_id)
        if not donut_loop_df.empty:
            donut_loop_df.insert(0, "case_id", case_id)
        for df in point_sheets.values():
            df["case_id"] = case_id
        for df in region_sheets.values():
            df["case_id"] = case_id

        return {
            "ok": True,
            "label": label, "component_id": component_id,
            "path_summaries": path_summaries,
            "root_to_terminal_summaries": root_to_terminal_summaries,
            "tree_summary": tree_summary,
            "tree_region_summaries": region_summaries,
            "tree_segment_summaries": seg_summaries,
            "branch_df": branch_df,
            "donut_loop_df": donut_loop_df,
            "point_sheets": point_sheets,
            "region_sheets": region_sheets,
            "compact": compact,
        }
    except Exception as e:
        return {
            "ok": False,
            "label": label, "component_id": component_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        set_extra_detection_junctions(None)


# Helpers

def _output_dirs(case_out_dir: str) -> Dict[str, Optional[str]]:
    """Compute the (unconditional) output subdirectory paths for one case, per config flags."""
    return {
        "centerline_dir": os.path.join(case_out_dir, "centerlines"),
        "centerline_radius_dir": (
            os.path.join(case_out_dir, "centerlines_radius")
            if _config.SAVE_CENTERLINE_RADIUS else None
        ),
        "region_centerline_dir": None,
        "surface_dir": os.path.join(case_out_dir, "surfaces"),
    }


def _setup_output_dirs(case_out_dir: str) -> Dict[str, Optional[str]]:
    """Create the enabled output subdirectories and clear any stale generated VTPs from a prior run."""
    dirs = _output_dirs(case_out_dir)
    ensure_dir(case_out_dir)
    if _config.SAVE_CENTERLINES:
        ensure_dir(dirs["centerline_dir"])
        clean_generated_vtp_dir(dirs["centerline_dir"])
        if _config.SAVE_CENTERLINE_RADIUS:
            ensure_dir(dirs["centerline_radius_dir"])
            clean_generated_vtp_dir(dirs["centerline_radius_dir"])
    if _config.SAVE_SURFACES:
        ensure_dir(dirs["surface_dir"])
        clean_generated_vtp_dir(dirs["surface_dir"])
    return dirs


def label_input_volumetry(multilabel: np.ndarray, labels: List[int], spacing) -> Dict[int, dict]:
    """Voxel count / volume / component count per label for the mask handed to the pipeline.

    Reported alongside the measured per-component volumetry so the gap between
    them is visible: components too small to skeletonize never reach a
    ``tree_summary`` row.
    """
    voxel_mm3 = float(np.prod(np.asarray(spacing, dtype=float)[:3]))
    out: Dict[int, dict] = {}
    for label in labels:
        mask = multilabel == label
        n_voxels = int(np.count_nonzero(mask))
        if not n_voxels:
            continue
        out[int(label)] = {
            "n_voxels": n_voxels,
            "volume_mm3": float(n_voxels) * voxel_mm3,
            "n_components": int(len(connected_components(mask)))
            if _config.PROCESS_ALL_CONNECTED_COMPONENTS else 1,
        }
    return out


def _make_jobs(
    multilabel: np.ndarray,
    labels: List[int],
    mapping: dict,
    dirs: dict,
    case_id: str,
    spacing: np.ndarray,
    ctx: MorphoContext,
) -> List[Tuple]:
    """Build the per-(label, component) argument tuples that :func:`_worker` will process.

    Components with fewer than 2 voxels are skipped (too small for a centerline).
    """
    jobs = []
    for label in labels:
        if not (multilabel == label).any():
            continue
        comps = (
            connected_components(multilabel == label)
            if _config.PROCESS_ALL_CONNECTED_COMPONENTS
            else [keep_largest_component(multilabel == label)]
        )
        for component_id, mask_cc in enumerate(comps, start=1):
            if int(mask_cc.sum()) < 2:
                continue
            vessel_info = mapping.get(label, empty_vessel_info(label))
            jobs.append((
                int(label), int(component_id), mask_cc.astype(bool),
                multilabel, spacing, vessel_info, mapping, dirs, case_id, ctx,
            ))
    return jobs


def unique_sheet_names(keys: List[str], prefix: str = "P") -> Dict[str, str]:
    """Map each key to a unique Excel sheet name (<= 31 characters): ``P001_<key...>``.

    Truncating long path ids to Excel's 31-character limit made different paths collide (and
    overwrite one another's sheet); the numeric prefix keeps them apart, and the full ids are listed
    in the ``Sheet_Index`` sheet.
    """
    out: Dict[str, str] = {}
    used: set = set()
    for i, key in enumerate(keys, start=1):
        base = f"{prefix}{i:03d}_" + safe_sheet_name(str(key))
        name = base[:31]
        k = 2
        while name.lower() in used:
            suffix = f"~{k}"
            name = base[:31 - len(suffix)] + suffix
            k += 1
        used.add(name.lower())
        out[key] = name
    return out


def _first(df: pd.DataFrame, column: str):
    return df[column].iloc[0] if column in df.columns and len(df) else None


def _write_excel(
    case_out_dir: str,
    path_summaries: List[dict],
    tree_summaries: List[dict],
    tree_region_summaries: List[dict],
    branch_dfs: List[pd.DataFrame],
    donut_loop_dfs: List[pd.DataFrame],
    vessel_sheets: Dict[str, pd.DataFrame],
    input_volumetry: Optional[Dict[int, dict]] = None,
    tree_segment_summaries: Optional[List[dict]] = None,
    root_to_terminal_summaries: Optional[List[dict]] = None,
    compact: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    """Assemble every result table into ``case_metrics_donut_tree.xlsx`` (+ CSVs).

    Sheet order: the compact tables first; ``Vessel_Summary`` (one row per vessel), ``Segments``,
    ``Lesions``, ``Subject`` (also written as ``vessel_summary.csv``, ``segments.csv``,
    ``lesions.csv``, ``subject_summary.csv``), then the detailed cow_morpho-style sheets.
    ``00_Path_Summary`` holds the **non-overlapping** segment rows: each piece of vessel is measured
    once. The raw root->terminal paths, which re-traverse every shared trunk, are kept in
    ``07_Root_To_Terminal_Paths`` for traceability: summing their lengths overestimates the
    vessel tree. Point-wise tables follow as ``P###_...`` sheets (``Sheet_Index`` maps them to path
    ids) and as ``points/<path_id>.csv``.
    """
    compact = compact or {}
    path_summary_df = pd.DataFrame(path_summaries)
    tree_segment_df = pd.DataFrame(tree_segment_summaries or [])
    root_to_terminal_df = pd.DataFrame(root_to_terminal_summaries or [])
    tree_summary_df = pd.DataFrame(tree_summaries)
    branchpoints_df = pd.concat(branch_dfs, ignore_index=True) if branch_dfs else pd.DataFrame()
    lr_df = compute_lr_asymmetry(path_summary_df)
    hemisphere_df = compute_hemispheric_summary(path_summary_df)
    volumetry_df = compute_volumetry_summary(tree_summary_df, input_volumetry)

    keys = sorted(vessel_sheets, key=vessel_sheet_sort_key)
    sheet_names = unique_sheet_names(keys)
    assert len(set(n.lower() for n in sheet_names.values())) == len(keys), "duplicate point-sheet names"

    out_xlsx = os.path.join(case_out_dir, "case_metrics_donut_tree.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for sheet, key in (("Vessel_Summary", "vessels"), ("Segments", "segments"),
                           ("Lesions", "lesions"), ("Subject", "subject")):
            df = compact.get(key)
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet, index=False)
        path_summary_export_dataframe(path_summary_df).to_excel(writer, sheet_name="00_Path_Summary", index=False)
        if not tree_summary_df.empty:
            tree_summary_export_dataframe(tree_summary_df).to_excel(writer, sheet_name="01_Tree_Summary", index=False)
        if not branchpoints_df.empty:
            branchpoint_export_dataframe(branchpoints_df).to_excel(writer, sheet_name="02_Branchpoints", index=False)
        if not lr_df.empty:
            lr_df.to_excel(writer, sheet_name="03_LR_Asymmetry", index=False)
        if not hemisphere_df.empty:
            hemisphere_df.to_excel(writer, sheet_name="05_Hemisphere", index=False)
        if not tree_segment_df.empty:
            tree_segment_df.to_excel(writer, sheet_name="04_Tree_Segments", index=False)
        if not volumetry_df.empty:
            volumetry_df.to_excel(writer, sheet_name="06_Volumetry", index=False)
        if not root_to_terminal_df.empty:
            # path_id links each row to its Sheet_Index entry and points/*.csv.
            path_summary_export_dataframe(root_to_terminal_df, keep=("path_id",)).to_excel(
                writer, sheet_name="07_Root_To_Terminal_Paths", index=False
            )
        if keys:
            pd.DataFrame({
                "sheet": [sheet_names[k] for k in keys],
                "path_id": keys,
                "label": [_first(vessel_sheets[k], "label") for k in keys],
                "vessel_name": [_first(vessel_sheets[k], "vessel_name") for k in keys],
                "points_csv": [os.path.join("points", safe_filename(k) + ".csv") for k in keys],
            }).to_excel(writer, sheet_name="Sheet_Index", index=False)
        for key in keys:
            vessel_sheets[key].to_excel(writer, sheet_name=sheet_names[key], index=False)

    if keys:
        points_dir = os.path.join(case_out_dir, "points")
        ensure_dir(points_dir)
        for key in keys:
            vessel_sheets[key].to_csv(os.path.join(points_dir, safe_filename(key) + ".csv"), index=False)
    # Plain CSVs alongside the workbook (no openpyxl needed downstream: Slicer UI, scripts, DB).
    for name, df in (("vessel_summary.csv", compact.get("vessels")), ("segments.csv", compact.get("segments")),
                     ("lesions.csv", compact.get("lesions")), ("subject_summary.csv", compact.get("subject")),
                     ("volumetry.csv", volumetry_df)):
        if df is not None and not df.empty:
            df.to_csv(os.path.join(case_out_dir, name), index=False)
    return out_xlsx


# Public API

def run_case(
    seg_path: str,
    out_dir: str,
    mapping_json: Optional[str] = None,
    mapping: Optional[dict] = None,
    case_out_dir_override: Optional[str] = None,
    n_workers: Optional[int] = None,
    species: Optional[str] = None,
    axes_override: Optional[str] = None,
    length_scale: float = 1.0,
    config: Optional[MorphometricsConfig] = None,
    case_id: Optional[str] = None,
) -> str:
    """Process one segmentation file in parallel.  Returns the Excel path.

    *case_id* labels every output row (default: the segmentation's file stem; give the subject
    id when several subjects share one file name, as stage 7's ``TOF_eICAB_CW.nii.gz`` do).

    *config* (default: the ``eicab_cw`` profile) is installed for the run (here and in every
    spawned worker), and the previous values are restored afterwards. *species* /
    *axes_override* / *length_scale* describe the subject the label ids belong to; they are
    normally taken from the topology JSON's ``_meta`` block by
    :func:`nvitk.measure.morphometrics.run_morphometrics_case`. They control how
    ``no_upstream_start`` rules resolve onto array axes and how the human-calibrated millimetre
    thresholds are rescaled.
    """
    from nvitk.measure.morphometrics_config import default_morphometrics_config

    config = config or default_morphometrics_config()
    with active_config(config):
        try:
            return _run_case(
                seg_path, out_dir, mapping_json=mapping_json, mapping=mapping,
                case_out_dir_override=case_out_dir_override, n_workers=n_workers, species=species,
                axes_override=axes_override, length_scale=length_scale, config=config,
                case_id=case_id,
            )
        finally:
            set_world_transform(None, None)


def _run_case(
    seg_path: str,
    out_dir: str,
    *,
    mapping_json: Optional[str],
    mapping: Optional[dict],
    case_out_dir_override: Optional[str],
    n_workers: Optional[int],
    species: Optional[str],
    axes_override: Optional[str],
    length_scale: float,
    config: MorphometricsConfig,
    case_id: Optional[str] = None,
) -> str:
    _headless()
    n_workers = int(n_workers) if n_workers is not None else int(N_WORKERS)

    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    multilabel, affine, spacing = load_multilabel_nifti(seg_path)
    ctx = MorphoContext(
        axes=resolve_anatomical_axes(affine, species=species, axes_override=axes_override),
        length_scale=float(length_scale),
        config=config,
        affine=tuple(map(tuple, np.asarray(affine, dtype=float))),
    )
    set_world_transform(ctx.affine, spacing)
    if _config.BRIDGE_LABEL_GAPS_BEFORE_CENTERLINES:
        from nvitk.morphology.mst_bridge import fill_multilabel_gaps_mst

        multilabel = fill_multilabel_gaps_mst(
            multilabel,
            close_radius=int(_config.BRIDGE_LABEL_CLOSE_RADIUS),
            bridge_max_gap=int(_config.BRIDGE_LABEL_MAX_GAP_VOXELS),
            bridge_radius=1,
        )
        print(f"  Bridged nearby same-label gaps (max_gap={int(_config.BRIDGE_LABEL_MAX_GAP_VOXELS)} vx).")
    if not _config.PROCESS_ALL_CONNECTED_COMPONENTS:
        multilabel = keep_largest_component_per_label(multilabel)

    labels_all = sorted(int(x) for x in np.unique(multilabel) if x != 0)
    if not labels_all:
        raise ValueError("No non-zero labels found in segmentation.")
    labels = resolve_labels_to_process(labels_all)

    if mapping is not None:
        resolved_mapping = mapping
    elif mapping_json and os.path.exists(mapping_json):
        resolved_mapping = load_mapping(mapping_json)
    else:
        from nvitk.measure.morpho.topology_io import load_eicab_topology

        resolved_mapping = load_eicab_topology()

    case_id = case_id or os.path.basename(seg_path).replace(".nii.gz", "").replace(".nii", "")
    case_out_dir = case_out_dir_override or os.path.join(out_dir, case_id)
    dirs = _setup_output_dirs(case_out_dir)

    input_volumetry = label_input_volumetry(multilabel, labels, spacing)
    jobs = _make_jobs(multilabel, labels, resolved_mapping, dirs, case_id, spacing, ctx)

    print("Parallel pipeline")
    print(f"  Case    : {case_id}")
    print(f"  Profile : {config.profile} (centerline backend: {config.centerline_backend})")
    print(f"  Labels  : {labels}")
    print(f"  Jobs    : {len(jobs)} label/component task(s)")
    print(f"  Workers : {n_workers}")
    print(f"  Output  : {case_out_dir}")
    print(f"  Anatomy : {ctx.describe()}")

    if n_workers <= 1:
        results = [_worker(args) for args in jobs]
    else:
        mp_context = get_context("spawn")
        results = []
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as pool:
            future_to_job = {pool.submit(_worker, args): (args[0], args[1]) for args in jobs}
            for future in as_completed(future_to_job):
                label, component_id = future_to_job[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "ok": False, "label": label, "component_id": component_id,
                        "error": str(e), "traceback": traceback.format_exc(),
                    }
                results.append(result)
                status = "done" if result.get("ok") else "ERROR"
                print(f"  [{status}] label {label} component {component_id}")

    results.sort(key=lambda r: (int(r.get("label", 0)), int(r.get("component_id", 0))))

    path_summaries, tree_summaries, tree_region_summaries = [], [], []
    root_to_terminal_summaries, tree_segment_summaries = [], []
    branch_dfs, donut_loop_dfs = [], []
    vessel_sheets: Dict[str, pd.DataFrame] = {}
    vessel_rows, segment_rows, lesion_rows = [], [], []
    errors = []

    for result in results:
        if not result.get("ok"):
            errors.append(result)
            print(f"  ERROR label {result['label']} component {result['component_id']}: {result['error']}")
            print(result["traceback"])
            continue
        path_summaries.extend(result["path_summaries"])
        root_to_terminal_summaries.extend(result.get("root_to_terminal_summaries", []))
        tree_summaries.append(result["tree_summary"])
        tree_region_summaries.extend(result["tree_region_summaries"])
        tree_segment_summaries.extend(result.get("tree_segment_summaries", []))
        if not result["branch_df"].empty:
            branch_dfs.append(result["branch_df"])
        if not result["donut_loop_df"].empty:
            donut_loop_dfs.append(result["donut_loop_df"])
        vessel_sheets.update(result["point_sheets"])
        compact = result.get("compact") or {}
        vessel_rows.extend(compact.get("vessels", []))
        segment_rows.extend(compact.get("segments", []))
        lesion_rows.extend(compact.get("lesions", []))

    vessels_df = vessel_summary(vessel_rows, lesion_rows, mapping=resolved_mapping)
    if not vessels_df.empty:
        vessels_df["case_id"] = case_id
    # Absent main vessels -> "hypoplasia" lesions (no location), flagged per vessel too.
    hypoplasia = hypoplasia_rows(vessels_df, resolved_mapping, input_volumetry)
    # An A1 hypoplasia is shown where the missing A1 would run (ICA end -> AComm / other A1).
    locate_a1_hypoplasia(hypoplasia, multilabel, resolved_mapping, spacing)
    lesion_rows.extend(hypoplasia)
    vessels_df = add_hypoplasia_columns(vessels_df, hypoplasia)
    # A P2 fed from the ICA through the PComm, its P1 absent or hypoplastic -> "fetal" lesions.
    fetal = fetal_pca_rows(multilabel, resolved_mapping, vessels_df, hypoplasia, spacing)
    lesion_rows.extend(fetal)
    compact_tables = {
        "vessels": vessels_df,
        "segments": pd.DataFrame(segment_rows),
        "lesions": _lesions_with_world_position(pd.DataFrame(lesion_rows)),
        "subject": subject_summary(vessels_df, fetal),
    }
    out_xlsx = _write_excel(
        case_out_dir, path_summaries, tree_summaries, tree_region_summaries,
        branch_dfs, donut_loop_dfs, vessel_sheets, input_volumetry,
        tree_segment_summaries=tree_segment_summaries,
        root_to_terminal_summaries=root_to_terminal_summaries,
        compact=compact_tables,
    )

    if _config.SAVE_CENTERLINES:
        clean_legacy_centerline_vtp_names(dirs["centerline_dir"])
        if _config.SAVE_CENTERLINE_RADIUS:
            clean_legacy_centerline_vtp_names(dirs["centerline_radius_dir"])

    print(f"\n  Done. Excel → {out_xlsx}")
    print(f"  Jobs OK: {len(results) - len(errors)}/{len(results)}")
    if errors:
        print(f"  Failed : {len(errors)}")

    return out_xlsx


def _lesions_with_world_position(lesions: pd.DataFrame) -> pd.DataFrame:
    """Add the lesion midpoint in scanner RAS (the frame of the VTPs) next to ``x/y/z_mm``."""
    if lesions.empty or not {"x_mm", "y_mm", "z_mm"} <= set(lesions.columns):
        return lesions
    from nvitk.measure.morpho.surface import to_world

    ras = to_world(lesions[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float))
    out = lesions.copy()
    out["x_ras"], out["y_ras"], out["z_ras"] = ras[:, 0], ras[:, 1], ras[:, 2]
    return out


if __name__ == "__main__":
    raise SystemExit(
        "Use nvitk.pipes.qvtpy.stage7_morphometrics (or nvitk.measure.morphometrics.run_morphometrics_case) "
        "with --output-root and --subject; paths are resolved via qvtpy config."
    )
