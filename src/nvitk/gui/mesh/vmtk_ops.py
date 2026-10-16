"""VMTK operations of the Mesh panel (and the experimental EXPIGEO ray-cast skeleton).

Each wraps :mod:`nvitk.vmtk` (or :mod:`nvitk.morphology.raycast_skeleton`) for the active surface
layer: surfaces go to VMTK as ``vtkPolyData`` in world millimetres and come back as new layers,
per-vertex fields, centerline overlays and tables. Appended to
:data:`nvitk.gui.mesh.operations.OPERATIONS`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.gui.mesh.layers import add_field, add_point_cloud_layer, add_polylines_layer
from nvitk.gui.tools.registry import ParamSpec
from nvitk.types import Mesh, PointCloud

_P = ParamSpec


def _poly(ctx: Any) -> Any:
    """The active surface as a ``vtkPolyData`` (PyVista)."""
    from nvitk.meshlab.convert import to_pyvista

    return to_pyvista(ctx.mesh())


def _mesh(poly: Any) -> Mesh:
    """A ``vtkPolyData`` back as a :class:`Mesh` (triangulated surface)."""
    import pyvista as pv

    from nvitk.meshlab.convert import from_pyvista

    return from_pyvista(pv.wrap(poly))


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.{digits}g}" if np.isfinite(f) else "–"


# ──────────────────────────────────────────────────────────────────────────────
# Clean, smooth, remesh
# ──────────────────────────────────────────────────────────────────────────────


def run_vmtk_smooth(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.vmtk.surface import smooth

    method = str(ctx.params.get("vmtk_smooth_method") or "taubin")
    it = int(ctx.params.get("vmtk_smooth_iter") or 30)
    out = _mesh(smooth(_poly(ctx), method=method, iterations=it,
                       passband=float(ctx.params.get("vmtk_passband") or 0.1),
                       relaxation=float(ctx.params.get("vmtk_relaxation") or 0.01)))
    ctx.put_mesh(out, "vmtk_smooth")
    return OpResult(f"VMTK {method} smoothing of {ctx.name}: {it} iterations.")


def run_vmtk_remesh(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.vmtk.surface import mean_edge_length, remesh

    edge = float(ctx.params.get("vmtk_edge") or 0.0)
    out_poly = remesh(_poly(ctx), target_edge_length=edge or None,
                      iterations=int(ctx.params.get("vmtk_remesh_iter") or 10),
                      preserve_boundary=bool(ctx.params.get("vmtk_keep_boundary", False)))
    ctx.put_mesh(_mesh(out_poly), "vmtk_remesh")
    return OpResult(f"VMTK remesh of {ctx.name}: {out_poly.GetNumberOfCells()} triangles, "
                    f"mean edge {mean_edge_length(out_poly):.3g} mm.")


def run_vmtk_decimate(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.vmtk.surface import decimate

    src = ctx.mesh()
    out = _mesh(decimate(_poly(ctx), target_reduction=float(ctx.params.get("vmtk_reduction") or 0.5)))
    ctx.put_mesh(out, "vmtk_decimated")
    return OpResult(f"VMTK decimation of {ctx.name}: {src.n_faces} → {out.n_faces} faces.")


def run_vmtk_subdivide(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.vmtk.surface import subdivide

    src = ctx.mesh()
    out = _mesh(subdivide(_poly(ctx), subdivisions=int(ctx.params.get("vmtk_subdiv") or 1),
                          method=str(ctx.params.get("vmtk_subdiv_method") or "butterfly")))
    ctx.put_mesh(out, "vmtk_subdivided")
    return OpResult(f"VMTK subdivision of {ctx.name}: {src.n_faces} → {out.n_faces} faces.")


def run_vmtk_cap(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.vmtk.surface import cap

    poly, entity = cap(_poly(ctx), method=str(ctx.params.get("vmtk_cap_method") or "centerpoint"))
    out = _mesh(poly)
    layer = ctx.put_mesh(out, "capped")
    n_caps = int(len(np.unique(entity[entity > 1])))
    if n_caps and layer is not None and len(entity) == out.n_faces:
        # Faces → per-vertex wall/cap id (max over each vertex's faces) for colouring.
        vid = np.zeros(len(out.vertices), dtype=np.float32)
        np.maximum.at(vid, out.faces.ravel(), np.repeat(entity.astype(np.float32), 3))
        add_field(layer, "cap id", vid, show=False)
    return OpResult(f"Capped {ctx.name}: {n_caps} opening(s) closed.")


def run_vmtk_kites(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.vmtk.surface import remove_kites

    src = ctx.mesh()
    out = _mesh(remove_kites(_poly(ctx), size_factor=float(ctx.params.get("vmtk_kite_factor") or 0.1)))
    ctx.put_mesh(out, "no_kites")
    return OpResult(f"Removed marching-cubes kites from {ctx.name}: {src.n_faces} → {out.n_faces} faces.")


# ──────────────────────────────────────────────────────────────────────────────
# Measure
# ──────────────────────────────────────────────────────────────────────────────


def run_vmtk_curvature(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.vmtk.surface import curvature

    kind = str(ctx.params.get("vmtk_curv_kind") or "mean")
    values = curvature(_poly(ctx), kind=kind, absolute=bool(ctx.params.get("vmtk_curv_abs")),
                       median_filter=bool(ctx.params.get("vmtk_curv_median", True)))
    add_field(ctx.layer, f"{kind} curvature (VMTK)", values.astype(np.float32))
    v = values[np.isfinite(values)]
    return OpResult(f"{kind.capitalize()} curvature of {ctx.name}: median {_fmt(np.median(v))} 1/mm, "
                    f"p5–p95 {_fmt(np.percentile(v, 5))}…{_fmt(np.percentile(v, 95))}.")


def run_vmtk_mass(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.vmtk.surface import mass_properties

    props = mass_properties(_poly(ctx))
    table = {
        "surface area (mm²)": _fmt(props["surface_area_mm2"], 5),
        "volume (mm³)": _fmt(props["volume_mm3"], 5),
        "shape index": _fmt(props["shape_index"], 4),
        "centre of mass (mm)": ", ".join(_fmt(c, 4) for c in props["center_of_mass"]),
    }
    return OpResult(f"{ctx.name}: area {table['surface area (mm²)']} mm², volume {table['volume (mm³)']} mm³.",
                    table=table, title=f"VMTK mass properties: {ctx.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Vessels & tubes
# ──────────────────────────────────────────────────────────────────────────────


def _network(ctx: Any) -> Any:
    from nvitk.meshlab.vessels import mesh_vmtk_network

    return mesh_vmtk_network(ctx.mesh(), voxel_size=float(ctx.params.get("vmtk_seed_voxel") or 0.0))


def _overlay_network(ctx: Any, net: Any, suffix: str) -> list[Any]:
    """Branch polylines + radius-coloured centerline points for a VMTK network."""
    pts = np.vstack([b.points for b in net.branches])
    radius = np.concatenate([b.radius for b in net.branches])
    group = np.concatenate([np.full(len(b.points), b.group_id) for b in net.branches])
    span = float(np.ptp(ctx.mesh().vertices, axis=0).max())
    cloud = PointCloud(points=pts, metadata={"name": f"{ctx.name}_{suffix}"},
                       point_data={"radius": radius, "branch": group.astype(np.float32)})
    return [lyr for lyr in (
        add_polylines_layer(ctx.viewer, [b.points for b in net.branches], name=f"{ctx.name}_{suffix}_lines",
                            width=max(span / 400.0, 0.1)),
        add_point_cloud_layer(ctx.viewer, cloud, name=f"{ctx.name}_{suffix}", color_by="radius",
                              size=max(span / 120.0, 0.3)),
    ) if lyr is not None]


def run_vmtk_branches(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult, _gather
    from nvitk.meshlab.convert import to_pyvista
    from nvitk.vmtk.centerlines import bifurcation_geometry, branch_clip

    net = _network(ctx)
    layers = _overlay_network(ctx, net, "vmtk_branches")
    if bool(ctx.params.get("vmtk_clip_surface", True)) and len(net.branches) > 1:
        groups = branch_clip(to_pyvista(ctx.mesh()), net.split)
        add_field(ctx.layer, "branch (VMTK)", groups.astype(np.float32))
    _gather(ctx, layers, f"{ctx.name} VMTK branches")
    table: dict[str, dict[str, Any]] = {}
    for b in net.branches:
        table[f"group {b.group_id}" + (" (bifurcation)" if b.blanking else "")] = {
            "length (mm)": _fmt(b.length),
            "mean Ø (mm)": _fmt(2.0 * np.nanmean(b.radius)),
            "min Ø (mm)": _fmt(2.0 * np.nanmin(b.radius)),
            "centerlines": ", ".join(str(c) for c in b.centerline_ids),
        }
    if len(net.branches) > 1:
        for k, bif in enumerate(bifurcation_geometry(net.split), 1):
            angles = ", ".join(f"g{d['group_id']}: {d['in_plane_angle_deg']:.0f}°/{d['out_of_plane_angle_deg']:.0f}°"
                               for d in bif["branches"] if d["role"] == "daughter")
            table[f"bifurcation {k}"] = {"length (mm)": f"group {bif['group_id']}", "mean Ø (mm)": angles}
    n_v = len(net.vessel_branches)
    return OpResult(f"VMTK: {n_v} branch(es), {len(net.bifurcations)} bifurcation piece(s) from "
                    f"{len(net.targets) + 1} end points.", table=table, title=f"VMTK branches: {ctx.name}",
                    row_header="Group")


def run_vmtk_sections(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult, _gather
    from nvitk.meshlab.convert import to_pyvista
    from nvitk.vmtk._core import point_array, points_of
    from nvitk.vmtk.centerlines import centerline_sections

    net = _network(ctx)
    cl, _sections = centerline_sections(to_pyvista(ctx.mesh()), net.centerlines)
    pts = points_of(cl)
    fields = {
        "section area (mm²)": point_array(cl, "CenterlineSectionArea"),
        "min size (mm)": point_array(cl, "CenterlineSectionMinSize"),
        "max size (mm)": point_array(cl, "CenterlineSectionMaxSize"),
        "shape (min/max)": point_array(cl, "CenterlineSectionShape"),
    }
    span = float(np.ptp(ctx.mesh().vertices, axis=0).max())
    cloud = PointCloud(points=pts, metadata={"name": f"{ctx.name}_sections"},
                       point_data={k: v.astype(np.float32) for k, v in fields.items()})
    lyr = add_point_cloud_layer(ctx.viewer, cloud, name=f"{ctx.name}_sections", color_by="section area (mm²)",
                                size=max(span / 120.0, 0.3))
    _gather(ctx, [lyr] if lyr is not None else [], f"{ctx.name} VMTK sections")
    area = fields["section area (mm²)"]
    ok = np.isfinite(area)
    table = {name: {"median": _fmt(np.nanmedian(v)), "min": _fmt(np.nanmin(v)), "max": _fmt(np.nanmax(v))}
             for name, v in fields.items()}
    return OpResult(f"VMTK sections along {len(pts)} centerline points ({int(ok.sum())} closed).", table=table,
                    title=f"VMTK centerline sections: {ctx.name}", row_header="Measure")


def run_vmtk_distance(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.meshlab.convert import to_pyvista
    from nvitk.vmtk.centerlines import distance_to_centerlines

    net = _network(ctx)
    relative = bool(ctx.params.get("vmtk_dist_relative"))
    d = distance_to_centerlines(to_pyvista(ctx.mesh()), net.centerlines, use_radius=relative)
    name = "distance / radius (VMTK)" if relative else "distance to centerline (VMTK)"
    add_field(ctx.layer, name, d.astype(np.float32))
    return OpResult(f"{name} on {ctx.name}: median {_fmt(np.nanmedian(d))}"
                    + ("" if relative else " mm") + ".")


def run_vmtk_flow_extensions(ctx: Any) -> Any:
    from nvitk.gui.mesh.operations import OpResult
    from nvitk.meshlab.convert import to_pyvista
    from nvitk.vmtk._core import points_of
    from nvitk.vmtk.centerlines import centerlines_between_points
    from nvitk.vmtk.surface import cap, flow_extensions

    surface = to_pyvista(ctx.mesh())
    capped, entity = cap(surface, method="simple")
    caps = [int(e) for e in np.unique(entity) if e > 1]
    if len(caps) < 2:
        raise ValueError("Flow extensions need an open surface with at least two openings (clip the ends first).")
    # Cap centres become the centerline seeds (inlet = the largest opening).
    pts = points_of(capped)
    centres, areas = [], []
    for e in caps:
        cells = np.flatnonzero(entity == e)
        ids = sorted({capped.GetCell(int(c)).GetPointId(j) for c in cells for j in range(3)})
        centres.append(pts[ids].mean(axis=0))
        areas.append(len(cells))
    order = np.argsort(areas)[::-1]
    centres = np.asarray(centres)[order]
    cl = centerlines_between_points(capped, centres[0], centres[1:], resampling_step=None)
    out = _mesh(flow_extensions(surface, cl, extension_ratio=float(ctx.params.get("vmtk_ext_ratio") or 10.0)))
    ctx.put_mesh(out, "flow_extensions")
    return OpResult(f"Flow extensions on {len(caps)} opening(s) of {ctx.name} "
                    f"({len(points_of(cl))} centerline points).")


def run_expigeo(ctx: Any) -> Any:
    """Experimental: EXPIGEO ray-cast skeleton + geometric descriptors on the surface."""
    import vtk

    from nvitk.gui.mesh.operations import OpResult, _gather
    from nvitk.meshlab.convert import to_pyvista
    from nvitk.morphology.raycast_skeleton import (
        RaycastParams,
        expigeo_descriptors,
        raycast_skeleton,
        surface_points_normals,
    )

    surface = to_pyvista(ctx.mesh())
    if bool(ctx.params.get("expigeo_smooth", True)):
        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputData(surface)
        sm.SetNumberOfIterations(15)
        sm.SetPassBand(0.1)
        sm.NormalizeCoordinatesOn()
        sm.Update()
        surface = sm.GetOutput()
    pts, nrm = surface_points_normals(surface)
    params = RaycastParams(
        n_rays=int(ctx.params.get("expigeo_rays") or 8),
        theta_max_deg=float(ctx.params.get("expigeo_theta") or 15.0),
    )
    skel = raycast_skeleton(pts, nrm, params)
    desc = expigeo_descriptors(skel, ctx.mesh().vertices)
    for name, values in desc.surface_fields().items():
        add_field(ctx.layer, f"EXPIGEO {name}", values.astype(np.float32), show=name == "sphericity")
    span = float(np.ptp(ctx.mesh().vertices, axis=0).max())
    cloud = PointCloud(points=skel.vertices, metadata={"name": f"{ctx.name}_raycast_skeleton"},
                       point_data={"radius": skel.radius, "termination_likelihood": desc.termination_likelihood,
                                   "sphericity": desc.sphericity, "blockage": desc.blockage})
    layers = [add_polylines_layer(ctx.viewer, skel.polylines(), name=f"{ctx.name}_raycast_lines",
                                  width=max(span / 400.0, 0.1)),
              add_point_cloud_layer(ctx.viewer, cloud, name=f"{ctx.name}_raycast_skeleton", color_by="radius",
                                    size=max(span / 120.0, 0.3))]
    _gather(ctx, [lyr for lyr in layers if lyr is not None], f"{ctx.name} EXPIGEO")
    table = {
        name: {"median": _fmt(np.nanmedian(v)), "p95": _fmt(np.nanpercentile(v, 95))}
        for name, v in desc.surface_fields().items()
    }
    return OpResult(
        f"Ray-cast skeleton (experimental): {len(skel.centerline_points)} centerline points → "
        f"{len(skel.vertices)} vertices, {len(skel.leaves)} leaves; descriptors added as fields.",
        table=table, title=f"EXPIGEO descriptors: {ctx.name}", row_header="Descriptor",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Registration in the Mesh panel
# ──────────────────────────────────────────────────────────────────────────────


def vmtk_operations(op_factory: Any, categories: dict[str, str]) -> tuple[Any, ...]:
    """The VMTK ``MeshOp`` entries (built with the panel's ``_op`` factory)."""
    c = categories
    seed_voxel = _P("vmtk_seed_voxel", "Seed skeleton voxel (mm, 0 = auto)", "float", 0.0, min=0.0, max=10.0)
    return (
        op_factory("vmtk_cap", c["clean"], "Cap open ends (VMTK)", ("mesh",),
                   (_P("vmtk_cap_method", "Cap", "choice", "centerpoint",
                       choices=("simple", "centerpoint", "smooth", "annular")),),
                   run_vmtk_cap, "vmtksurfacecapper: close every open boundary; each cap gets its own id "
                   "(field “cap id”, 1 = wall).", replaceable=True),
        op_factory("vmtk_kites", c["clean"], "Remove marching-cubes kites (VMTK)", ("mesh",),
                   (_P("vmtk_kite_factor", "Size factor", "float", 0.1, min=0.0, max=1.0),),
                   run_vmtk_kites, "vmtksurfacekiteremoval: collapse the small 'kite' triangle pairs left by "
                   "marching cubes.", replaceable=True),
        op_factory("vmtk_smooth", c["smooth"], "Smooth (VMTK Taubin / Laplace)", ("mesh",),
                   (_P("vmtk_smooth_method", "Method", "choice", "taubin", choices=("taubin", "laplace")),
                    _P("vmtk_smooth_iter", "Iterations", "int", 30, min=1, max=1000),
                    _P("vmtk_passband", "Taubin pass band", "float", 0.1, min=0.001, max=2.0),
                    _P("vmtk_relaxation", "Laplace relaxation", "float", 0.01, min=0.0001, max=1.0)),
                   run_vmtk_smooth, "vmtksurfacesmoothing: Taubin (windowed-sinc, volume preserving) or Laplacian.",
                   replaceable=True),
        op_factory("vmtk_remesh", c["remesh"], "Isotropic remesh (VMTK)", ("mesh",),
                   (_P("vmtk_edge", "Target edge length (mm, 0 = keep mean)", "float", 0.0, min=0.0, max=50.0),
                    _P("vmtk_remesh_iter", "Iterations", "int", 10, min=1, max=100),
                    _P("vmtk_keep_boundary", "Preserve boundary edges", "bool", False)),
                   run_vmtk_remesh, "vmtksurfaceremeshing: regular triangles of a target edge length "
                   "(CFD-quality surface).", replaceable=True),
        op_factory("vmtk_decimate", c["remesh"], "Decimate (VMTK)", ("mesh",),
                   (_P("vmtk_reduction", "Remove fraction of triangles", "float", 0.5, min=0.01, max=0.99),),
                   run_vmtk_decimate, "vmtksurfacedecimation (quadric decimation).", replaceable=True),
        op_factory("vmtk_subdivide", c["remesh"], "Subdivide (VMTK)", ("mesh",),
                   (_P("vmtk_subdiv", "Subdivisions", "int", 1, min=1, max=4),
                    _P("vmtk_subdiv_method", "Method", "choice", "butterfly", choices=("linear", "butterfly", "loop"))),
                   run_vmtk_subdivide, "vmtksurfacesubdivision.", replaceable=True),
        op_factory("vmtk_curvature", c["measure"], "Surface curvature (VMTK)", ("mesh",),
                   (_P("vmtk_curv_kind", "Curvature", "choice", "mean", choices=("mean", "gaussian", "maximum", "minimum")),
                    _P("vmtk_curv_abs", "Absolute value", "bool", False),
                    _P("vmtk_curv_median", "Median-filter over each one-ring", "bool", True)),
                   run_vmtk_curvature, "vmtksurfacecurvature: per-vertex curvature (1/mm) as a field."),
        op_factory("vmtk_mass", c["measure"], "Area / volume / shape index (VMTK)", ("mesh",), (),
                   run_vmtk_mass, "vmtksurfacemassproperties: surface area, enclosed volume, shape index and "
                   "centre of mass."),
        op_factory("vmtk_branches", c["vessels"], "Branches & bifurcation angles (VMTK)", ("mesh",),
                   (seed_voxel, _P("vmtk_clip_surface", "Colour the surface by branch", "bool", True)),
                   run_vmtk_branches, "VMTK centerlines from the widest end to every other end, split into branches "
                   "(vmtkbranchextractor) with the maximum-inscribed-sphere radius; the surface is split by branch "
                   "(vmtkbranchclipper) and each bifurcation's in-plane / out-of-plane daughter angles are listed "
                   "(vmtkbifurcationreferencesystems / vectors)."),
        op_factory("vmtk_sections", c["vessels"], "Centerline sections (VMTK)", ("mesh",), (seed_voxel,),
                   run_vmtk_sections, "vmtkcenterlinesections: lumen section area, min / max size and shape at "
                   "every centerline point."),
        op_factory("vmtk_distance", c["vessels"], "Distance to centerlines (VMTK)", ("mesh",),
                   (seed_voxel, _P("vmtk_dist_relative", "Relative to the inscribed radius", "bool", False)),
                   run_vmtk_distance, "vmtkdistancetocenterlines: wall-to-centerline distance per vertex (or "
                   "normalised by the local inscribed radius: >1 bulges, <1 narrowings)."),
        op_factory("vmtk_flow_extensions", c["vessels"], "Flow extensions (VMTK)", ("mesh",),
                   (_P("vmtk_ext_ratio", "Extension length (× opening radius)", "float", 10.0, min=0.5, max=100.0),),
                   run_vmtk_flow_extensions, "vmtkflowextensions: extrude every opening along the centerline "
                   "direction (inflow / outflow tubes for CFD). Needs an open surface (clip the ends first)."),
        op_factory("expigeo", c["vessels"], "Ray-cast skeleton + EXPIGEO descriptors (experimental)", ("mesh",),
                   (_P("expigeo_rays", "Rays per surface point (K)", "int", 8, min=3, max=64),
                    _P("expigeo_theta", "Cone half-angle (°)", "float", 15.0, min=1.0, max=60.0),
                    _P("expigeo_smooth", "Smooth the surface first", "bool", True)),
                   run_expigeo, "EXPIGEO (Layachi et al., MICCAI 2026): rays cast inward from every surface point "
                   "give the centerline and radius; voxelised, MST-linked and pruned into a skeleton. Descriptors "
                   "per vertex: termination likelihood, blockage, geometric sphericity, radius, radius-gradient "
                   "magnitude and projected flow. Experimental."),
    )


__all__ = ["vmtk_operations"]
