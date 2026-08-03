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
from nvitk.measure.morpho.io_utils import (
    clean_generated_vtp_dir,
    clean_legacy_centerline_vtp_names,
    ensure_dir,
    load_mapping,
    load_multilabel_nifti,
    safe_sheet_name,
    vessel_sheet_sort_key,
)
from nvitk.measure.morpho.labels_util import (
    connected_components,
    empty_vessel_info,
    keep_largest_component,
    keep_largest_component_per_label,
    resolve_labels_to_process,
)
from nvitk.measure.morpho.orchestration import process_component_tree_vmtk
from nvitk.measure.morpho.export_utils.summaries import (
    branchpoint_export_dataframe,
    build_vessel_points_dataframe,
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
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Worker (must be a top-level function to be picklable for spawn)
# ---------------------------------------------------------------------------

def _worker(args: Tuple[Any, ...]) -> Dict[str, Any]:
    _headless()
    (
        label, component_id, mask_cc,
        multilabel, spacing, vessel_info, mapping,
        dirs, case_id,
    ) = args

    try:
        print(f"\n=== Label {label} / component {component_id} ===")
        print(f"  Component voxels: {int(mask_cc.sum())}")
        (
            path_results, point_sheets, tree_summary, branch_df,
            region_summaries, region_sheets, recursive_segments, donut_loop_df,
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
        )

        path_summaries = []
        for res in path_results:
            res["case_id"] = case_id
            path_summaries.append(compute_vessel_summary_row(case_id, label, vessel_info, res))

        tree_summary["case_id"] = case_id
        for row in region_summaries:
            row["case_id"] = case_id

        seg_summaries = recursive_tree_segment_summary_rows(
            case_id=case_id, label=label, component_id=component_id,
            vessel_info=vessel_info, segments=recursive_segments,
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
            "tree_summary": tree_summary,
            "tree_region_summaries": region_summaries,
            "tree_segment_summaries": seg_summaries,
            "branch_df": branch_df,
            "donut_loop_df": donut_loop_df,
            "point_sheets": point_sheets,
            "region_sheets": region_sheets,
        }
    except Exception as e:
        return {
            "ok": False,
            "label": label, "component_id": component_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dirs(case_out_dir: str) -> Dict[str, Optional[str]]:
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


def _make_jobs(
    multilabel: np.ndarray,
    labels: List[int],
    mapping: dict,
    dirs: dict,
    case_id: str,
    spacing: np.ndarray,
) -> List[Tuple]:
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
                multilabel, spacing, vessel_info, mapping, dirs, case_id,
            ))
    return jobs


def _write_excel(
    case_out_dir: str,
    path_summaries: List[dict],
    tree_summaries: List[dict],
    tree_region_summaries: List[dict],
    branch_dfs: List[pd.DataFrame],
    donut_loop_dfs: List[pd.DataFrame],
    vessel_sheets: Dict[str, pd.DataFrame],
) -> str:
    path_summary_df = pd.DataFrame(path_summaries)
    tree_summary_df = pd.DataFrame(tree_summaries)
    branchpoints_df = pd.concat(branch_dfs, ignore_index=True) if branch_dfs else pd.DataFrame()
    lr_df = compute_lr_asymmetry(path_summary_df)
    hemisphere_df = compute_hemispheric_summary(path_summary_df)

    out_xlsx = os.path.join(case_out_dir, "case_metrics_donut_tree.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        path_summary_export_dataframe(path_summary_df).to_excel(writer, sheet_name="00_Path_Summary", index=False)
        if not tree_summary_df.empty:
            tree_summary_export_dataframe(tree_summary_df).to_excel(writer, sheet_name="01_Tree_Summary", index=False)
        if not branchpoints_df.empty:
            branchpoint_export_dataframe(branchpoints_df).to_excel(writer, sheet_name="02_Branchpoints", index=False)
        if not lr_df.empty:
            lr_df.to_excel(writer, sheet_name="03_LR_Asymmetry", index=False)
        if not hemisphere_df.empty:
            hemisphere_df.to_excel(writer, sheet_name="05_Hemisphere", index=False)
        for sheet_name in sorted(vessel_sheets, key=vessel_sheet_sort_key):
            vessel_sheets[sheet_name].to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return out_xlsx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_case(
    seg_path: str,
    out_dir: str,
    mapping_json: Optional[str] = None,
    mapping: Optional[dict] = None,
    case_out_dir_override: Optional[str] = None,
    n_workers: Optional[int] = None,
) -> str:
    """Process one segmentation file in parallel.  Returns the Excel path."""
    _headless()
    n_workers = int(n_workers) if n_workers is not None else int(N_WORKERS)

    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    multilabel, _affine, spacing = load_multilabel_nifti(seg_path)
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
        from nvitk.measure.morpho.topology_io import load_qvtpy_topology

        resolved_mapping = load_qvtpy_topology()

    case_id = os.path.basename(seg_path).replace(".nii.gz", "").replace(".nii", "")
    case_out_dir = case_out_dir_override or os.path.join(out_dir, case_id)
    dirs = _setup_output_dirs(case_out_dir)

    jobs = _make_jobs(multilabel, labels, resolved_mapping, dirs, case_id, spacing)

    print(f"Parallel pipeline")
    print(f"  Case    : {case_id}")
    print(f"  Labels  : {labels}")
    print(f"  Jobs    : {len(jobs)} label/component task(s)")
    print(f"  Workers : {n_workers}")
    print(f"  Output  : {case_out_dir}")

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
    branch_dfs, donut_loop_dfs = [], []
    vessel_sheets: Dict[str, pd.DataFrame] = {}
    errors = []

    for result in results:
        if not result.get("ok"):
            errors.append(result)
            print(f"  ERROR label {result['label']} component {result['component_id']}: {result['error']}")
            print(result["traceback"])
            continue
        path_summaries.extend(result["path_summaries"])
        tree_summaries.append(result["tree_summary"])
        tree_region_summaries.extend(result["tree_region_summaries"])
        if not result["branch_df"].empty:
            branch_dfs.append(result["branch_df"])
        if not result["donut_loop_df"].empty:
            donut_loop_dfs.append(result["donut_loop_df"])
        vessel_sheets.update(result["point_sheets"])

    out_xlsx = _write_excel(
        case_out_dir, path_summaries, tree_summaries, tree_region_summaries,
        branch_dfs, donut_loop_dfs, vessel_sheets,
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


if __name__ == "__main__":
    raise SystemExit(
        "Use nvitk.pipes.qvtpy.stage7_morphometrics (or nvitk.measure.morphometrics.run_morphometrics_case) "
        "with --output-root and --subject; paths are resolved via qvtpy config."
    )
