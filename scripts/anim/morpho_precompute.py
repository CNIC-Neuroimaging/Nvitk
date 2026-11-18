"""Precompute the TOF morphometrics (qvtpy stage 7) of the example subject for ``morpho_manim.py``.

Runs :func:`nvitk.measure.morphometrics.run_morphometrics_case` on the post-processed eICAB mask
(``TOF_eICAB_WB_pp``, as stage 7 does: eICAB topology, profile ``eicab_cw``, VMTK centerlines)
unless ``--morpho-dir`` already holds its outputs, and gathers what the animation draws:

- every voxel of the mask (index + label) and its affine;
- the marching-cubes surface of each vessel (``surfaces/*.vtp``, scanner RAS);
- every root→terminal path's point table (``points/*.csv``): position, arc length, cross-section
  and inscribed-sphere radii, curvature, the siphon mask, the robust reference radius and the
  deviation from it, the lesion flags;
- on the showcase path (the left ICA), a few cross-sections cut exactly as
  :func:`nvitk.measure.morpho.surface.compute_cross_section_radius` cuts them (plane normal from
  :func:`~nvitk.measure.morpho.surface.cross_section_tangents`, the surface's contour, the 36 rays of
  ``ray_median`` and their first wall crossings);
- the vessel summary, the lesion table (the most severe stenosis is shown on its path) and the
  provenance;
- for the circle-of-Willis variants step, a second subject (``--variants-mask``, the fetal
  example): its surfaces, voxels and lesion table (absent / hypoplastic segments, fetal PCA).

Run (nvitk environment)::

    python scripts/anim/morpho_precompute.py -o scripts/anim/morpho_stages.npz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

DEFAULT_MASK = Path("/home/imarcoss/placeholder/animations/stenosis example/eicab/TOF_eICAB_WB_pp.nii.gz")
#: The circle-of-Willis variants step shows another subject: the fetal example (absent right A1,
#: thread-like right P1, fetal right PCA).
DEFAULT_VARIANTS_MASK = Path("/home/imarcoss/placeholder/animations/fetal example/eicab/TOF_eICAB_WB_pp.nii.gz")
SHOWCASE = "LICA"
N_SECTIONS = 6
POINT_COLS = ["x_ras", "y_ras", "z_ras", "s_mm", "radius_mm", "maximum_inscribed_sphere_radius_mm",
              "curvature_1_per_mm", "siphon_mask_point", "robust_radius_mm", "robust_reference_radius_mm",
              "deviation_percent", "is_stenotic", "is_enlarged"]
SUMMARY_COLS = ["vessel_name", "label", "present", "main_length_mm", "main_chord_length_mm", "tortuosity_dm",
                "diameter_mean_mm", "diameter_min_mm", "diameter_max_mm", "volume_mm3", "surface_area_mm2",
                "curvature_mean_1_per_mm", "n_bifurcations", "n_stenosis", "n_enlargement", "hypoplasia"]


def run_morphometrics(mask: Path, out_dir: Path) -> None:
    from nvitk.measure.morpho.topology_io import default_eicab_topology_path
    from nvitk.measure.morphometrics import run_morphometrics_case

    run_morphometrics_case(mask, out_dir, mapping_json=str(default_eicab_topology_path()),
                           case_out_dir_override=out_dir, n_workers=12, input_already_smoothed=True,
                           case_id="example")


def read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtk.util import numpy_support

    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(str(path))
    r.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(r.GetOutput())
    tri.Update()
    poly = tri.GetOutput()
    pts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)
    cells = numpy_support.vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return pts, cells.astype(np.int32)


def cross_sections(surf_path: Path, pts: np.ndarray, at: np.ndarray) -> list[dict]:
    """The cut of the vessel surface by the cross-section plane at the points *at* (indices into
    *pts*), as the morphometrics cut it, with the 36 rays' first wall crossings."""
    import vtk
    from vtk.util import numpy_support

    from nvitk.measure.morpho import surface as msurf
    from nvitk.measure.morphometrics_config import apply_config, morphometrics_config

    apply_config(morphometrics_config("eicab_cw"))  # the stage-7 profile's cut (1 mm tangent window, 36 rays)
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(str(surf_path))
    r.Update()
    surface = r.GetOutput()
    out = []
    if True:
        tangents = msurf.cross_section_tangents(pts)
        k = int(msurf._cfg.CROSS_SECTION_RAYS)
        for i in at:
            t = tangents[i]
            plane = vtk.vtkPlane()
            plane.SetOrigin(*map(float, pts[i]))
            plane.SetNormal(*map(float, t))
            cutter = vtk.vtkCutter()
            cutter.SetInputData(surface)
            cutter.SetCutFunction(plane)
            stripper = vtk.vtkStripper()
            stripper.SetInputConnection(cutter.GetOutputPort())
            stripper.Update()
            res = stripper.GetOutput()
            allp = numpy_support.vtk_to_numpy(res.GetPoints().GetData()).astype(float)
            loops = []
            lines = res.GetLines()
            lines.InitTraversal()
            ids = vtk.vtkIdList()
            while lines.GetNextCell(ids):
                loops.append([ids.GetId(j) for j in range(ids.GetNumberOfIds())])
            radius = msurf.ray_median_radius(allp, loops, pts[i].astype(float), t)
            # the rays' first crossings, as ray_median_radius finds them
            u, v = msurf._plane_basis(t)
            rel = allp - pts[i]
            xy = np.column_stack([rel @ u, rel @ v])
            a_, e_ = [], []
            for loop in loops:
                idx = np.asarray(loop, int)
                if idx[0] != idx[-1]:
                    idx = np.append(idx, idx[0])
                a_.append(xy[idx[:-1]])
                e_.append(xy[idx[1:]])
            a = np.concatenate(a_)
            e = np.concatenate(e_) - a
            theta = 2.0 * np.pi * np.arange(k) / k
            d = np.column_stack([np.cos(theta), np.sin(theta)])
            denom = d[:, :1] * e[None, :, 1] - d[:, 1:] * e[None, :, 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / denom
                ss = (a[None, :, 0] * d[:, 1:] - a[None, :, 1] * d[:, :1]) / denom
            hit = (np.abs(denom) > 1e-12) & (tt > 1e-9) & (ss >= 0.0) & (ss < 1.0)
            first = np.where(hit, tt, np.inf).min(axis=1)
            ends = pts[i] + (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v) * np.where(
                np.isfinite(first), first, 0.0)[:, None]
            # the loop around the point (the one the median circle sits in): the loop the rays hit most
            out.append({"index": int(i), "origin": pts[i].tolist(), "normal": t.tolist(), "u": u.tolist(),
                        "v": v.tolist(), "radius": float(radius),
                        "ray_len": np.where(np.isfinite(first), first, np.nan).tolist(),
                        "ray_end": ends.tolist(), "loops": [allp[np.asarray(lp, int)].tolist() for lp in loops]})
    return out


def surfaces(morpho_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every vessel surface of a stage-7 run: vertices, faces, per-vertex eICAB label."""
    verts, faces, vlab, off = [], [], [], 0
    for f in sorted((morpho_dir / "surfaces").glob("*.vtp")):
        m = re.match(r"(\d+)_", f.name)
        if not m:
            continue
        p, c = read_surface(f)
        verts.append(p)
        faces.append(c + off)
        vlab.append(np.full(len(p), int(m.group(1)), np.int16))
        off += len(p)
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(vlab)


def variants(mask_path: Path, morpho_dir: Path) -> tuple[dict, dict]:
    """The variants step's subject: its surfaces, voxels, lesion table (the variants) and summary."""
    if not (morpho_dir / "vessel_summary.csv").is_file():
        print(f"running the morphometrics into {morpho_dir} (VMTK, a few minutes)")
        run_morphometrics(mask_path, morpho_dir)
    img = nib.load(mask_path)
    lab = np.asanyarray(img.dataobj).astype(np.int32)
    idx = np.argwhere(lab > 0)
    v, f, vl = surfaces(morpho_dir)
    arrays = {"v_vox": idx.astype(np.int16), "v_lab": lab[tuple(idx.T)].astype(np.int8), "v_mesh_v": v,
              "v_mesh_f": f, "v_mesh_lab": vl}
    lesions = pd.read_csv(morpho_dir / "lesions.csv")
    meta = {"affine": img.affine.tolist(),
            "lesions": json.loads(lesions.iloc[:, :16].to_json(orient="records")),
            "summary": json.loads(pd.read_csv(morpho_dir / "vessel_summary.csv").reindex(columns=SUMMARY_COLS)
                                  .to_json(orient="records"))}
    return arrays, meta


def compute(mask_path: Path, morpho_dir: Path) -> tuple[dict, dict]:
    if not (morpho_dir / "vessel_summary.csv").is_file():
        print(f"running the morphometrics into {morpho_dir} (VMTK, a few minutes)")
        run_morphometrics(mask_path, morpho_dir)
    img = nib.load(mask_path)
    lab = np.asanyarray(img.dataobj).astype(np.int32)
    idx = np.argwhere(lab > 0)

    mesh_v, mesh_f, mesh_lab = surfaces(morpho_dir)

    # paths
    rows, meta_paths = [], []
    for k, f in enumerate(sorted((morpho_dir / "points").glob("*.csv"))):
        d = pd.read_csv(f)
        if len(d) < 3:
            continue
        a = d.reindex(columns=POINT_COLS).to_numpy(float)
        rows.append(np.c_[a, np.full(len(a), len(meta_paths)), np.full(len(a), int(d["label"].iloc[0]))])
        meta_paths.append({"file": f.name, "vessel": str(d["vessel_name"].iloc[0]), "label": int(d["label"].iloc[0]),
                           "n": int(len(d)), "length_mm": float(d["s_mm"].max())})
    paths = np.concatenate(rows).astype(np.float32)

    # the showcase path: the longest path of the showcase vessel
    cand = [i for i, p in enumerate(meta_paths) if p["vessel"] == SHOWCASE]
    show = max(cand, key=lambda i: meta_paths[i]["n"])
    sel = paths[:, len(POINT_COLS)] == show
    sp = paths[sel][:, :3].astype(float)
    n = len(sp)
    at = np.linspace(0.12 * n, 0.88 * n, N_SECTIONS).astype(int)
    lid = meta_paths[show]["label"]
    surf_file = next((morpho_dir / "surfaces").glob(f"{lid}_*.vtp"))
    sections = cross_sections(surf_file, sp, at)

    summary = pd.read_csv(morpho_dir / "vessel_summary.csv").reindex(columns=SUMMARY_COLS)
    lesions = pd.read_csv(morpho_dir / "lesions.csv")
    # the lesion to show: the most severe stenosis (else enlargement) and the path it was found on
    lesion_show = None
    for kind in ("stenosis", "enlargement"):
        cand_l = lesions[lesions["kind"] == kind]
        if len(cand_l):
            row = cand_l.sort_values("diameter_percent", ascending=False).iloc[0]
            stems = [p["file"].rsplit(".", 1)[0] for p in meta_paths]
            pid = str(row["path_id"])
            k = next((i for i, st in enumerate(stems) if st == pid or st.endswith(pid) or pid.endswith(st)), None)
            if k is None:  # fall back to the vessel's longest path
                same = [i for i, p in enumerate(meta_paths) if p["label"] == int(row["label"])]
                k = max(same, key=lambda i: meta_paths[i]["n"]) if same else None
            keep = [c for c in ("label", "vessel_name", "kind", "segment", "path_id", "point_start", "point_end",
                                "start_s_from_vessel_origin_mm", "length_mm", "taper_percent", "diameter_percent",
                                "area_percent", "reference_radius_mm", "extreme_radius_mm", "confidence_class",
                                "x_mm", "y_mm", "z_mm") if c in row.index]
            lesion_show = {**json.loads(row[keep].to_json()), "path_index": k}
            break
    prov = json.loads((morpho_dir / "morphometrics_provenance.json").read_text())
    arrays = {
        "vox": idx.astype(np.int16),
        "lab": lab[tuple(idx.T)].astype(np.int8),
        "mesh_v": mesh_v,
        "mesh_f": mesh_f,
        "mesh_lab": mesh_lab,
        "paths": paths,
    }
    meta = {
        "affine": img.affine.tolist(),
        "spacing_mm": [float(s) for s in img.header.get_zooms()[:3]],
        "point_cols": POINT_COLS,
        "paths": meta_paths,
        "showcase": show,
        "lesion_show": lesion_show,
        "sections": sections,
        "summary": json.loads(summary.to_json(orient="records")),
        "lesions": json.loads(lesions.iloc[:, :16].to_json(orient="records")),
        "profile": prov.get("profile") or prov.get("config", {}).get("profile"),
        "centerline_backend": prov.get("centerline_backend") or prov.get("backend"),
    }
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--mask", type=Path, default=DEFAULT_MASK, help="Post-processed eICAB mask (WB_pp).")
    ap.add_argument("--morpho-dir", type=Path, default=Path("/tmp/nvitk_anim_morpho"),
                    help="Stage-7 output folder (run there when it holds no vessel_summary.csv).")
    ap.add_argument("--variants-mask", type=Path, default=DEFAULT_VARIANTS_MASK,
                    help="The subject the circle-of-Willis variants step shows.")
    ap.add_argument("--variants-morpho-dir", type=Path, default=Path("/tmp/nvitk_anim_morpho_variants"),
                    help="Its stage-7 output folder (run there when empty).")
    args = ap.parse_args()
    arrays, meta = compute(args.mask, args.morpho_dir)
    va, vm = variants(args.variants_mask, args.variants_morpho_dir)
    arrays.update(va)
    meta["variants"] = vm
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta, default=str)), **arrays)
    print({k: v.shape for k, v in arrays.items()})
    print(json.dumps({k: meta[k] for k in ("showcase", "lesion_show", "profile", "centerline_backend")}, default=str))
    print("variants:", [(r["vessel_name"], r["kind"]) for r in meta["variants"]["lesions"]])
    print([(s["index"], round(s["radius"], 2)) for s in meta["sections"]])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
