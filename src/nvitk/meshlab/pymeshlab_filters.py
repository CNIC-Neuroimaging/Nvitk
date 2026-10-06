"""MeshLab's filters (through PyMeshLab) on nvitk meshes and point clouds.

`PyMeshLab <https://pymeshlab.readthedocs.io>`_ exposes MeshLab's ~280 filters:
screened Poisson and ball-pivoting reconstruction, isotropic remeshing, quadric
decimation, Poisson-disk sampling, geodesic distances, curvature, booleans,
repairs of non-manifold geometry and many more. This module

- converts :class:`~nvitk.types.Mesh` / :class:`~nvitk.types.PointCloud` to a
  PyMeshLab mesh and back (normals, the per-vertex fields as custom attributes,
  one field as MeshLab's per-vertex *quality*, the per-vertex selection);
- runs any filter by name and sorts what it produced: new meshes, the changed
  input, a new per-vertex quality or selection, and returned measures
  (:func:`run_meshlab_filter`);
- lists the filters that make sense for medical surfaces, grouped and with typed
  parameters (:func:`meshlab_filters`), for user interfaces. Photogrammetry
  filters (cameras, rasters, textures, vertex colours) and those that need an
  OpenGL context are left out.

The listing comes from PyMeshLab itself (names, parameter types and defaults,
enum choices), cached under ``~/.cache/nvitk`` per PyMeshLab version.

Example::

    from nvitk.meshlab.pymeshlab_filters import run_meshlab_filter

    res = run_meshlab_filter(mesh, "meshing_isotropic_explicit_remeshing",
                             {"targetlen": "1%", "iterations": 5})
    remeshed = res.current

    res = run_meshlab_filter(cloud, "generate_surface_reconstruction_screened_poisson", {"depth": 8})
    surface = res.meshes[0]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError
from nvitk.types import Mesh, PointCloud

#: Online documentation of one filter.
DOCS_URL = "https://pymeshlab.readthedocs.io/en/latest/filter_list.html#{name}"

#: Name a field gets in MeshLab when it is not a valid identifier there.
_SELECTED = "nvitk_selected"

#: Filters left out of :func:`meshlab_filters` (substrings of their names):
#: photogrammetry (cameras, rasters, textures, colours), file and project I/O,
#: OpenGL-backed filters, quad / polygon meshes and the MeshSet bookkeeping.
_EXCLUDE = (
    "camera", "raster", "texcoord", "texmap", "texture", "sketchfab", "nxs_", "snapshot", "_gpu",
    "iso_parametrization", "voronoi_atlas", "set_mesh_name", "delete_", "load_", "save_", "export_",
    "apply_color_", "compute_color_", "set_color_", "ambient_occlusion", "depth_complexity",
    "overlapping_meshes", "global_alignment", "merging_visible", "set_matrix", "cylindrical_unwrapping",
    "vertex_attribute_seam", "generate_copy_of_current_mesh", "quad_mesh", "tri_to_quad", "poly_to_tri",
    "polygon", "by_grammar", "dust_accumulation", "fractal_terrain", "by_function_per_face",
    "texcoord", "new_custom_point_attribute", "from_selected_vertices", "coord_by_function",
    "volumetric_obscurance", "shape_diameter_function", "craters", "noisy_isosurface",
    "compute_scalar_from_camera", "point_cloud_movement_over_mesh", "cubic_stylization",
    "compute_mls_projection_", "voronoi_scaffolding",
)

#: ``(group, name prefixes)``; the first match wins. The order is the menu order.
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("MeshLab · Create", ("create_",)),
    ("MeshLab · Cleaning & repair", ("meshing_remove_", "meshing_repair_", "meshing_merge_close_vertices",
                                     "meshing_close_holes", "meshing_snap_", "meshing_re_orient_",
                                     "meshing_invert_face_orientation")),
    ("MeshLab · Remeshing & simplification", ("meshing_decimation_", "meshing_isotropic_", "meshing_surface_subdivision_",
                                              "meshing_edge_flip_", "meshing_refine_", "generate_resampled_uniform_mesh",
                                              "meshing_cut_along_crease_edges")),
    ("MeshLab · Smoothing & deformation", ("apply_coord_", "apply_normal_smoothing", "apply_normal_unsharp",
                                           "apply_normal_normalization", "compute_coord_")),
    ("MeshLab · Reconstruction & point clouds", ("generate_surface_reconstruction_", "generate_alpha_",
                                                 "generate_marching_cubes_", "compute_mls_projection_",
                                                 "compute_normal_for_point_clouds", "generate_simplified_point_cloud",
                                                 "apply_normal_point_cloud_smoothing", "generate_convex_hull",
                                                 "compute_curvature_and_color_")),
    ("MeshLab · Sampling", ("generate_sampling_", "generate_voronoi_")),
    ("MeshLab · Curvature, normals & quality", ("compute_curvature_", "compute_normal_", "compute_scalar_by_discrete",
                                                "compute_scalar_by_aspect", "compute_scalar_by_function",
                                                "compute_scalar_by_scalar_harmonic", "compute_scalar_transfer",
                                                "apply_scalar_", "compute_new_custom_scalar")),
    ("MeshLab · Measures & distances", ("get_", "compute_scalar_by_distance_", "compute_scalar_by_geodesic_",
                                        "compute_scalar_by_heat_geodesic_", "compute_scalar_by_border_distance",
                                        "generate_polyline_")),
    ("MeshLab · Booleans & pieces", ("generate_boolean_", "generate_splitting_by_connected_components",
                                     "generate_solid_wireframe", "transfer_attributes_per_vertex")),
    ("MeshLab · Selection", ("compute_selection_", "apply_selection_", "set_selection_", "meshing_remove_selected_",
                             "generate_from_selected_", "generate_plane_fitting_to_selection")),
    ("MeshLab · Transform & align", ("compute_matrix_", "apply_matrix_")),
)

#: Filters that run on point clouds (checked one by one on PyMeshLab 2025.7); the
#: others need triangles, and a few of them crash on a cloud instead of refusing it.
_POINT_CLOUD_FILTERS = {
    "apply_coord_random_displacement", "apply_matrix_flip_or_swap_axis", "apply_matrix_freeze",
    "apply_matrix_inverse", "apply_normal_normalization_per_vertex", "apply_normal_point_cloud_smoothing",
    "apply_scalar_clamping_per_vertex", "apply_scalar_saturation_per_vertex", "compute_curvature_and_color_apss_per_vertex",
    "compute_curvature_and_color_rimls_per_vertex", "compute_custom_radius_scalar_attribute_per_vertex",
    "compute_matrix_by_principal_axis", "compute_matrix_from_rotation", "compute_matrix_from_scaling_or_normalization",
    "compute_matrix_from_translation", "compute_matrix_from_translation_rotation_scale",
    "compute_new_custom_scalar_attribute_per_vertex", "compute_normal_by_function_per_vertex",
    "compute_normal_for_point_clouds", "compute_scalar_by_distance_from_another_mesh_per_vertex",
    "compute_scalar_by_distance_from_point_cloud_per_vertex", "compute_scalar_by_function_per_vertex",
    "compute_selection_point_cloud_outliers", "compute_selection_by_condition_per_vertex",
    "compute_selection_by_scalar_per_vertex", "meshing_remove_selected_vertices",
    "generate_alpha_shape", "generate_convex_hull", "generate_marching_cubes_apss", "generate_marching_cubes_rimls",
    "generate_sampling_clustered_vertex", "generate_sampling_element", "generate_simplified_point_cloud",
    "generate_surface_reconstruction_ball_pivoting", "generate_surface_reconstruction_screened_poisson",
    "generate_surface_reconstruction_vcg", "generate_voronoi_filtering", "get_geometric_measures",
    "get_hausdorff_distance", "get_scalar_histogram_per_vertex", "get_scalar_statistics_per_vertex",
    "get_topological_measures",
}

#: Friendly titles and one-line descriptions for the filters most used on anatomy.
_FRIENDLY: dict[str, tuple[str, str]] = {
    "generate_surface_reconstruction_screened_poisson": (
        "Screened Poisson reconstruction",
        "MeshLab's reference Poisson surface from an oriented point cloud (Kazhdan); octree depth sets the detail."),
    "generate_surface_reconstruction_ball_pivoting": (
        "Ball-pivoting reconstruction",
        "Rolls a ball over the points and keeps the triangles it rests on: interpolates the points, keeps holes."),
    "generate_surface_reconstruction_vcg": (
        "VCG volumetric reconstruction", "Volumetric merging of range points into a surface."),
    "generate_alpha_wrap": (
        "Alpha wrap (watertight shrink-wrap)",
        "A guaranteed watertight, manifold surface wrapped around any mesh or point soup (CGAL alpha wrap)."),
    "generate_alpha_shape": ("Alpha shape", "Delaunay-based shape of the points without simplices larger than alpha."),
    "generate_convex_hull": ("Convex hull", "The convex hull of the vertices / points."),
    "meshing_isotropic_explicit_remeshing": (
        "Isotropic remeshing",
        "Even, well-shaped triangles of a target edge length (split, collapse, flip, relax, reproject)."),
    "meshing_decimation_quadric_edge_collapse": (
        "Quadric edge-collapse decimation",
        "Fewer triangles with the least shape change (Garland–Heckbert), optionally keeping borders and topology."),
    "meshing_decimation_clustering": ("Clustering decimation", "Vertex clustering on a grid: fast, coarse."),
    "meshing_close_holes": ("Close holes", "Fills open borders up to a maximum hole size (in edges)."),
    "meshing_repair_non_manifold_edges": ("Repair non-manifold edges", "Removes or splits faces on edges shared by more than two faces."),
    "meshing_repair_non_manifold_vertices": ("Repair non-manifold vertices", "Splits vertices whose faces form separate fans."),
    "meshing_merge_close_vertices": ("Merge close vertices", "Welds vertices closer than a threshold."),
    "meshing_remove_duplicate_vertices": ("Remove duplicate vertices", "Merges vertices with identical coordinates."),
    "meshing_remove_duplicate_faces": ("Remove duplicate faces", "Removes faces repeated with the same vertices."),
    "meshing_remove_null_faces": ("Remove zero-area faces", "Removes degenerate faces."),
    "meshing_remove_unreferenced_vertices": ("Remove unreferenced vertices", "Drops vertices no face uses."),
    "meshing_remove_connected_component_by_diameter": ("Remove small pieces (by size)", "Drops connected pieces smaller than a diameter."),
    "meshing_remove_connected_component_by_face_number": ("Remove small pieces (by faces)", "Drops connected pieces with few faces."),
    "meshing_re_orient_faces_coherently": ("Orient faces coherently", "Makes the face orientation consistent across each piece."),
    "meshing_invert_face_orientation": ("Invert faces", "Flips every face (normals point the other way)."),
    "meshing_surface_subdivision_loop": ("Loop subdivision", "Smooth refinement: each triangle into four."),
    "meshing_surface_subdivision_butterfly": ("Butterfly subdivision", "Interpolating refinement: keeps the original vertices."),
    "meshing_surface_subdivision_midpoint": ("Midpoint subdivision", "Splits edges at their midpoints (shape unchanged)."),
    "generate_resampled_uniform_mesh": ("Uniform mesh resampling", "Rebuilds the surface on a regular voxel grid (offset, watertight)."),
    "apply_coord_taubin_smoothing": ("Taubin smoothing", "λ|μ smoothing without shrinkage."),
    "apply_coord_hc_laplacian_smoothing": ("HC Laplacian smoothing", "Laplacian smoothing that pulls vertices back toward the original (less shrinkage)."),
    "apply_coord_laplacian_smoothing": ("Laplacian smoothing", "Classic umbrella smoothing (shrinks)."),
    "apply_coord_two_steps_smoothing": ("Two-step feature-preserving smoothing", "Smooths the normals first, then fits the vertices to them: keeps sharp features."),
    "apply_coord_laplacian_smoothing_surface_preserving": ("Surface-preserving Laplacian", "Smooths while keeping the shape within an angle threshold."),
    "apply_coord_unsharp_mask": ("Unsharp mask (geometry)", "Sharpens geometric detail."),
    "apply_coord_random_displacement": ("Random displacement", "Adds noise to the vertices (testing robustness)."),
    "generate_sampling_poisson_disk": ("Poisson-disk sampling", "Evenly spaced samples on the surface (blue noise), or a subset of a point cloud."),
    "generate_sampling_montecarlo": ("Monte-Carlo sampling", "Random area-weighted samples on the surface."),
    "generate_sampling_stratified_triangle": ("Stratified triangle sampling", "Samples spread across each triangle."),
    "generate_sampling_volumetric": ("Volumetric sampling", "Samples inside the volume enclosed by the surface."),
    "generate_sampling_clustered_vertex": ("Clustered vertex sampling", "One representative vertex per grid cell."),
    "generate_simplified_point_cloud": ("Simplify point cloud", "Poisson-disk subset of a point cloud with a given number of points."),
    "compute_normal_for_point_clouds": ("Point-cloud normals", "Fits normals from the nearest neighbours and orients them."),
    "compute_normal_per_vertex": ("Vertex normals", "Recomputes per-vertex normals (area / angle weighted)."),
    "compute_curvature_principal_directions_per_vertex": (
        "Principal curvatures", "Principal curvature values and directions (several estimators) into the quality."),
    "compute_scalar_by_discrete_curvature_per_vertex": (
        "Discrete curvature", "Mean, Gaussian, RMS or absolute curvature per vertex (into the quality)."),
    "compute_scalar_by_aspect_ratio_per_face": ("Triangle quality", "Aspect ratio of each face (mesh quality)."),
    "compute_scalar_by_geodesic_distance_from_given_point_per_vertex": (
        "Geodesic distance from a click", "Distance along the surface from the clicked point."),
    "compute_scalar_by_geodesic_distance_from_selection_per_vertex": (
        "Geodesic distance from the selection", "Distance along the surface from the selected vertices."),
    "compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex": (
        "Heat-method geodesic distance", "Fast geodesic distance from the selection (heat method)."),
    "compute_scalar_by_border_distance_per_vertex": ("Distance from the border", "Geodesic distance from the open borders."),
    "compute_scalar_by_distance_from_another_mesh_per_vertex": (
        "Distance from another mesh", "Per-vertex distance (signed or not) to a reference surface."),
    "get_hausdorff_distance": ("Hausdorff distance", "Hausdorff (and mean / RMS) distance between two surfaces."),
    "get_geometric_measures": ("Geometric measures", "Area, volume, centre of mass, inertia, principal axes."),
    "get_topological_measures": ("Topological measures", "Vertices, edges, faces, boundary edges, components, genus, manifoldness, holes."),
    "generate_boolean_union": ("Boolean union", "Union of two closed surfaces."),
    "generate_boolean_intersection": ("Boolean intersection", "Intersection of two closed surfaces."),
    "generate_boolean_difference": ("Boolean difference", "The first surface minus the second."),
    "generate_boolean_xor": ("Boolean XOR", "Symmetric difference of two closed surfaces."),
    "generate_splitting_by_connected_components": ("Split into pieces", "One mesh per connected piece."),
    "generate_polyline_from_planar_section": ("Planar section (polyline)", "The intersection with an axis-aligned plane."),
    "compute_matrix_by_icp_between_meshes": ("ICP alignment", "Rigid alignment of this mesh onto a reference."),
    "compute_matrix_by_principal_axis": ("Align to principal axes", "Rotates the mesh onto its principal axes."),
    "compute_selection_by_self_intersections_per_face": ("Select self-intersecting faces", "Selects faces that cross other faces."),
    "compute_selection_by_non_manifold_edges_per_face": ("Select non-manifold edges", "Selects faces on non-manifold edges."),
    "compute_selection_from_mesh_border": ("Select border", "Selects the faces / vertices on open borders."),
    "compute_selection_by_condition_per_vertex": ("Select by condition", "Selects vertices where a formula holds (x, y, z, q, nx, …, fields)."),
    "compute_selection_by_scalar_per_vertex": ("Select by quality range", "Selects vertices whose quality lies in a range."),
    "compute_selection_point_cloud_outliers": ("Select outliers", "Selects points far from their neighbours (LoOP)."),
    "meshing_remove_selected_vertices": ("Delete selected vertices", "Removes the selected vertices and their faces."),
    "meshing_remove_selected_vertices_and_faces": ("Delete selected vertices and faces", "Removes the selection."),
    "meshing_remove_selected_faces": ("Delete selected faces", "Removes the selected faces."),
    "apply_selection_dilatation": ("Dilate selection", "Grows the selection by one ring."),
    "apply_selection_erosion": ("Erode selection", "Shrinks the selection by one ring."),
    "apply_selection_inverse": ("Invert selection", "Selects what was not selected."),
}

#: Friendlier names for frequent (terse) MeshLab parameter names.
_PARAM_LABELS: dict[str, str] = {
    "targetlen": "Target edge length", "iterations": "Iterations", "iteration": "Iterations",
    "featuredeg": "Feature angle (°)", "maxsurfdist": "Max distance from the surface", "checksurfdist": "Check the surface distance",
    "adaptive": "Adaptive", "selectedonly": "Selected only", "selected": "Selected only",
    "depth": "Octree depth", "fulldepth": "Full depth", "samplespernode": "Samples per node",
    "pointweight": "Interpolation weight", "preclean": "Pre-clean the points", "scale": "Scale",
    "targetfacenum": "Target number of faces", "targetperc": "Target fraction (0 = use the number)",
    "qualitythr": "Quality threshold", "preserveboundary": "Keep the borders", "boundaryweight": "Border weight",
    "preservenormal": "Keep normals (no flips)", "preservetopology": "Keep the topology", "optimalplacement": "Optimal placement",
    "planarquadric": "Planar simplification", "autoclean": "Clean afterwards", "maxholesize": "Largest hole (edges)",
    "samplenum": "Number of samples", "radius": "Radius", "ballradius": "Ball radius", "clustering": "Clustering radius",
    "creasethr": "Crease angle (°)", "deletefaces": "Delete the old faces", "threshold": "Threshold", "stepsmoothnum": "Smoothing steps",
    "cotangentweight": "Cotangent weights", "boundary": "Smooth the borders too", "k": "Neighbours", "smoothiter": "Smoothing iterations",
    "flipflag": "Flip normals", "viewpos": "Viewpoint", "startpoint": "Start point", "maxdistance": "Max distance",
    "sampledmesh": "Sampled mesh", "targetmesh": "Target mesh", "sourcemesh": "Source mesh", "first_mesh": "First mesh",
    "second_mesh": "Second mesh", "refmesh": "Reference mesh", "referencemesh": "Reference mesh", "measuremesh": "Measured mesh",
    "signeddist": "Signed distance", "lambda_": "Lambda", "mu": "Mu", "curvaturetype": "Curvature", "method": "Method",
    "cellsize": "Cell size", "offset": "Offset", "mergeclosevert": "Merge close vertices", "multisample": "Multisample",
    "absdist": "Absolute distance", "exactnum": "Exact number of samples", "subsample": "Subsample", "refineflag": "Refine",
    "splitflag": "Split edges", "collapseflag": "Collapse edges", "swapflag": "Flip edges", "smoothflag": "Relax",
    "reprojectflag": "Reproject", "alpha": "Alpha", "offset_": "Offset", "minval": "Min", "maxval": "Max", "inclusive": "Inclusive",
    "condselect": "Condition", "q": "Quality formula", "normalthr": "Normal angle threshold (°)", "normaliter": "Normal iterations",
    "fititer": "Fitting iterations", "smoothnum": "Smoothing steps", "neighbours": "Neighbours", "useinputmesh": "Use the input mesh",
    "sampleradius": "Sample spacing", "bestsamplepoolsize": "Best-sample pool size", "maxfaces": "Max faces",
    "removeunref": "Remove unreferenced vertices", "planeaxis": "Plane axis", "planeoffset": "Plane offset", "relativeto": "Relative to",
    "createsectionsurface": "Also make the section surface", "splitsurfacewithsection": "Split the surface at the section",
    "savesample": "Add the sample points as layers", "samplevert": "Sample the vertices", "sampleedge": "Sample the edges",
    "samplefauxedge": "Sample the faux edges", "sampleface": "Sample the faces", "maxdist": "Max distance",
    "visiblelayer": "Merge all visible layers", "cgdepth": "Conjugate-gradient depth", "iters": "Solver iterations",
    "confidence": "Quality as confidence", "threads": "Threads", "alllayers": "Apply to all layers",
    "freeze": "Bake the transform into the vertices", "subdiv": "Subdivision level", "viewpoint": "Viewpoint",
    "filterscale": "Filter scale", "projectionaccuracy": "Projection accuracy", "maxprojectioniters": "Max projection iterations",
    "transfer_face_color": "Keep face colours", "transfer_face_quality": "Keep face quality",
    "transfer_vert_color": "Keep vertex colours", "transfer_vert_quality": "Keep vertex quality",
    "selection": "Selected only", "onselected": "Selected only", "selectiononly": "Selected only", "onlyselected": "Selected only",
    "weight": "Weight", "weightorig": "Weight of the original", "radiusvariance": "Radius variance", "angle": "Angle (°)",
    "angledeg": "Angle (°)", "randomseed": "Random seed", "delta": "Step", "bestsampleflag": "Best-sample heuristic",
    "bestsamplepool": "Best-sample pool size", "exactnumflag": "Exact number of samples", "exactnumtolerance": "Tolerance (exact number)",
    "sigman": "Normal sharpness (σn)", "maxrefittingiters": "Max refitting iterations", "resolution": "Grid resolution",
    "customaxis": "Custom axis", "customcenter": "Custom centre", "histmin": "Histogram min", "histmax": "Histogram max",
    "areaweighted": "Area weighted", "binnum": "Bins", "minq": "Min quality", "maxq": "Max quality", "allfaces": "All faces",
    "allverts": "All vertices", "nbneighbors": "Neighbours", "rotaxis": "Rotation axis", "axisx": "X", "axisy": "Y", "axisz": "Z",
    "mincomponentsize": "Min piece size (faces)", "mincomponentdiag": "Min piece diameter", "newfaceselected": "Select the new faces",
    "selfintersection": "Avoid self-intersections", "refinehole": "Refine the filled holes",
    "refineholeedgelen": "Edge length in filled holes", "forceflip": "Force the flip", "vertdispratio": "Vertex displacement ratio",
    "rays": "Rays", "pointnum": "Number of points", "size": "Size", "loopweight": "Weighting scheme", "pthreshold": "Threshold",
}

#: String parameters that are formulas, file names or labels rather than choices.
_FREE_TEXT = {
    "a", "b", "g", "r", "q", "x", "y", "z", "u", "v", "u0", "u1", "u2", "v0", "v1", "v2", "x_expr", "y_expr", "z_expr",
    "expr", "condselect", "grammar", "attr_name", "name", "newname", "textname", "texturename", "imagefilename",
    "csvfilename", "exportfile", "importfile", "input_file", "output_file", "sketchfabkeycode", "tags", "title",
    "description", "tfslist",
}

#: Integer parameters that index a mesh of the MeshSet.
_MESH_PARAMS = {
    "basemesh", "coloredmesh", "controlmesh", "first_mesh", "measuremesh", "proxymesh", "referencemesh", "refinemesh",
    "refmesh", "sampledmesh", "samples_mesh", "second_mesh", "sourcemesh", "target_mesh", "targetmesh", "vertexmesh",
}

#: 3-vector parameters that are naturally a point clicked on the surface.
_PICK_PARAMS = {"startpoint"}

#: Filters that need oriented per-point normals on a point cloud (estimated when missing).
_NEEDS_NORMALS = ("surface_reconstruction", "marching_cubes_", "curvature_and_color_", "normal_point_cloud_smoothing")

#: Filters that read MeshLab's per-vertex quality (the field shown is passed as quality to these only).
_READS_QUALITY = re.compile(r"apply_scalar_|by_scalar|get_scalar_|scalar_transfer|vertices_by_scalar")


@dataclass(frozen=True)
class FilterParam:
    """One parameter of a MeshLab filter.

    *kind*: ``bool``, ``int``, ``float``, ``percent`` (a length: ``"1%"`` of the
    bounding-box diagonal, or a number in mm), ``enum`` (*choices*), ``text``,
    ``vec3``, ``mesh`` (another mesh of the MeshSet), ``matrix`` or ``color``.
    """

    name: str
    kind: str
    default: Any
    label: str = ""
    choices: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class FilterInfo:
    """A MeshLab filter: its PyMeshLab *name*, a *label*, its *group* and parameters."""

    name: str
    label: str
    group: str
    params: tuple[FilterParam, ...]
    description: str = ""
    #: ``()`` for generators (create_*), else what it can run on.
    inputs: tuple[str, ...] = ("mesh", "points")
    #: Name of the 3-vector parameter a click on the surface fills, if any.
    pick_param: str = ""

    @property
    def url(self) -> str:
        return DOCS_URL.format(name=self.name)


@dataclass
class MeshLabResult:
    """What a filter run produced."""

    #: New meshes / point clouds the filter added (reconstructions, samplings, pieces…).
    meshes: list[Mesh | PointCloud] = field(default_factory=list)
    #: The input after the filter, when the filter changed its geometry.
    current: Mesh | PointCloud | None = None
    #: Per-vertex quality of the input after the filter, when it changed (curvature, distances…).
    quality: np.ndarray | None = None
    #: Per-vertex selection of the input after the filter (0 / 1), when it changed.
    selected: np.ndarray | None = None
    #: Values the filter returned (measures).
    values: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Availability
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def pymeshlab_available() -> bool:
    """``True`` when PyMeshLab can be imported."""
    try:
        import pymeshlab  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def require_pymeshlab() -> Any:
    """Import and return ``pymeshlab``, or raise an install hint."""
    try:
        import pymeshlab
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailableError("MeshLab filters need PyMeshLab: pip install pymeshlab") from exc
    return pymeshlab


def pymeshlab_version() -> str:
    try:
        from importlib.metadata import version

        return version("pymeshlab")
    except Exception:  # noqa: BLE001
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────────────────────


def _identifier(name: str) -> str:
    """A field name MeshLab's formula parser accepts."""
    ident = re.sub(r"\W+", "_", str(name)).strip("_") or "field"
    return ident if not ident[0].isdigit() else f"f_{ident}"


def to_pymeshlab(obj: Mesh | PointCloud, *, quality: np.ndarray | str | None = None) -> Any:
    """A PyMeshLab mesh from *obj*: geometry, normals and *quality* (array or field name).

    The fields become custom attributes only once the mesh is in a MeshSet
    (:func:`add_to_meshset`): attributes added to a free-standing PyMeshLab mesh
    do not survive ``MeshSet.add_mesh`` intact, and a formula using them crashes.
    """
    pml = require_pymeshlab()
    if isinstance(obj, PointCloud):
        pts = np.ascontiguousarray(obj.points, dtype=np.float64)
        kwargs: dict[str, Any] = {"vertex_matrix": pts}
        nrm = obj.normals
        if nrm is not None and np.shape(nrm) == pts.shape:
            kwargs["v_normals_matrix"] = np.ascontiguousarray(nrm, dtype=np.float64)
        n = len(pts)
    else:
        kwargs = {"vertex_matrix": np.ascontiguousarray(obj.vertices, dtype=np.float64),
                  "face_matrix": np.ascontiguousarray(obj.faces, dtype=np.int32)}
        n = obj.n_vertices
    if isinstance(quality, str):
        quality = obj.point_data.get(quality)
    if quality is not None and len(quality) == n:
        q = np.nan_to_num(np.asarray(quality, dtype=np.float64), nan=0.0)
        kwargs["v_scalar_array"] = np.ascontiguousarray(q)
    return pml.Mesh(**kwargs)


def add_to_meshset(ms: Any, obj: Mesh | PointCloud, name: str = "mesh", *,
                   quality: np.ndarray | str | None = None) -> int:
    """Add *obj* to the MeshSet *ms* with its 1-D fields as custom per-vertex
    attributes (formula-safe names, usable in MeshLab's expressions); returns its id."""
    ms.add_mesh(to_pymeshlab(obj, quality=quality), name)
    m = ms.current_mesh()
    n = m.vertex_number()
    for key, arr in obj.point_data.items():
        arr = np.asarray(arr)
        if arr.ndim == 1 and len(arr) == n and np.issubdtype(arr.dtype, np.number):
            try:
                m.add_vertex_custom_scalar_attribute(np.nan_to_num(arr.astype(np.float64)), _identifier(key))
            except Exception:  # noqa: BLE001 — attributes are a convenience
                pass
    return ms.current_mesh_id()


def from_pymeshlab(m: Any, *, metadata: dict[str, Any] | None = None) -> Mesh | PointCloud:
    """A :class:`Mesh` (or :class:`PointCloud` when it has no faces) from a PyMeshLab mesh.

    Its quality comes along as ``point_data["quality"]``, normals of a point cloud
    as ``point_data["normals"]``; a polyline (edges, no faces) keeps its edges in
    ``metadata["edges"]``.
    """
    meta = dict(metadata or {})
    try:
        verts = np.asarray(m.transformed_vertex_matrix(), dtype=float)
    except Exception:  # noqa: BLE001
        verts = np.asarray(m.vertex_matrix(), dtype=float)
    data: dict[str, np.ndarray] = {}
    if m.has_vertex_scalar():
        data["quality"] = np.asarray(m.vertex_scalar_array(), dtype=float)
    faces = np.asarray(m.face_matrix(), dtype=np.int64) if m.face_number() else np.zeros((0, 3), np.int64)
    if not len(faces):
        try:
            nrm = np.asarray(m.vertex_normal_matrix(), dtype=float)
            if nrm.shape == verts.shape and np.isfinite(nrm).all() and np.abs(nrm).sum():
                data["normals"] = nrm
        except Exception:  # noqa: BLE001
            pass
        if m.edge_number():
            meta["edges"] = np.asarray(m.edge_matrix(), dtype=np.int64)
        return PointCloud(points=verts, metadata=meta, point_data=data)
    return Mesh(verts, faces, metadata=meta, point_data=data)


def _selection_of(m: Any) -> np.ndarray:
    try:
        return np.asarray(m.vertex_selection_array(), dtype=bool)
    except Exception:  # noqa: BLE001
        return np.zeros(m.vertex_number(), dtype=bool)


# ──────────────────────────────────────────────────────────────────────────────
# Running a filter
# ──────────────────────────────────────────────────────────────────────────────


def _value(pml: Any, kind: str, value: Any) -> Any:
    """A parameter value as PyMeshLab wants it."""
    if kind == "percent":
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("%"):
                return pml.PercentageValue(float(text[:-1] or 0))
            return pml.PureValue(float(text))
        return pml.PureValue(float(value))
    if kind == "vec3":
        if isinstance(value, str):
            value = [float(v) for v in re.split(r"[,;\s]+", value.strip()) if v]
        return np.asarray(value, dtype=np.float64).reshape(3)
    if kind == "matrix":
        if isinstance(value, str):
            value = [float(v) for v in re.split(r"[,;\s]+", value.strip()) if v]
        return np.asarray(value, dtype=np.float64).reshape(4, 4)
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    return value


def run_meshlab_filter(
    obj: Mesh | PointCloud | None,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    quality: np.ndarray | str | None = None,
    selected: np.ndarray | None = None,
) -> MeshLabResult:
    """Apply MeshLab filter *name* to *obj* (``None`` for generators like ``create_sphere``).

    *params* are the filter's parameters; a ``percent`` one may be ``"1%"`` (of
    the bounding-box diagonal) or a number (mm); a ``mesh`` one may be another
    :class:`Mesh` / :class:`PointCloud` (added to the MeshSet), with ``None`` or
    ``"active"`` meaning *obj*. *quality* is MeshLab's per-vertex quality (an
    array or a field name of *obj*), *selected* the per-vertex selection.
    Parameters left out keep MeshLab's defaults, computed on *obj*.
    """
    pml = require_pymeshlab()
    info = _info_or_none(name)
    kinds = {p.name: p.kind for p in info.params} if info is not None else {}
    params = dict(params or {})
    if isinstance(obj, PointCloud) and obj.normals is None and any(k in name for k in _NEEDS_NORMALS):
        from nvitk.meshlab.pointcloud import estimate_normals

        obj = estimate_normals(obj, orient="propagate")
    if name.startswith("generate_boolean_"):
        # MeshLab's booleans need closed surfaces with outward faces.
        from nvitk.meshlab.cleaning import ensure_outward

        obj = ensure_outward(obj) if isinstance(obj, Mesh) else obj
        params = {k: ensure_outward(v) if isinstance(v, Mesh) else v for k, v in params.items()}
    ms = pml.MeshSet()
    ms.set_verbosity(False)
    sel_in = None
    if obj is not None:
        add_to_meshset(ms, obj, "input", quality=quality)
        if selected is not None and len(selected) == ms.current_mesh().vertex_number() and np.any(selected):
            sel_in = np.asarray(selected, dtype=bool)
            # Rebuild the selection from a custom attribute through MeshLab's own condition filter.
            ms.current_mesh().add_vertex_custom_scalar_attribute(sel_in.astype(np.float64), _SELECTED)
            ms.apply_filter("compute_selection_by_condition_per_vertex", condselect=f"{_SELECTED} > 0.5")
            try:
                ms.apply_filter("compute_selection_transfer_vertex_to_face", inclusive=False)
            except Exception:  # noqa: BLE001
                pass
    kwargs: dict[str, Any] = {}
    extra: list[int] = []
    for key, value in params.items():
        kind = kinds.get(key, "")
        if kind == "mesh" or isinstance(value, (Mesh, PointCloud)):
            if value is None or (isinstance(value, str) and value.strip().lower() in ("", "active", "(active layer)")):
                kwargs[key] = 0
                continue
            if not isinstance(value, (Mesh, PointCloud)):
                raise ValueError(f"{key}: give a Mesh / PointCloud, or 'active'.")
            add_to_meshset(ms, value, f"ref{len(extra) + 1}")
            extra.append(ms.current_mesh_id())
            kwargs[key] = ms.current_mesh_id()
            continue
        if not kind and isinstance(value, str) and value.strip().endswith("%"):
            kind = "percent"
        kwargs[key] = _value(pml, kind, value) if kind else value
    for key, kind in kinds.items():
        if kind == "mesh" and key not in kwargs and obj is not None:
            kwargs[key] = 0  # MeshLab's own default is not always the current mesh
    if obj is not None:
        ms.set_current_mesh(0)
    before_ids = set(_mesh_ids(ms))
    if obj is not None:
        m0 = ms.mesh(0)
        v0 = np.asarray(m0.vertex_matrix()).copy()
        f0 = np.asarray(m0.face_matrix()).copy() if m0.face_number() else np.zeros((0, 3))
        q0 = np.asarray(m0.vertex_scalar_array()).copy() if m0.has_vertex_scalar() else None
        s0 = _selection_of(m0) if sel_in is None else sel_in

    values = ms.apply_filter(name, **kwargs) or {}

    flat: dict[str, Any] = {}
    for key, val in dict(values).items():
        val = _plain(val)
        if isinstance(val, dict):
            flat.update({f"{key}_{k}": v for k, v in val.items()})
        else:
            flat[key] = val
    res = MeshLabResult(values=flat)
    meta = dict(getattr(obj, "metadata", {}) or {}) if obj is not None else {"space": "world"}
    meta.pop("edges", None)
    for mid in [i for i in _mesh_ids(ms) if i not in before_ids]:
        out = from_pymeshlab(ms.mesh(mid), metadata=meta)
        if (isinstance(out, Mesh) and out.n_vertices) or (isinstance(out, PointCloud) and out.n_points):
            res.meshes.append(out)
    if obj is not None and ms.mesh_id_exists(0):
        m0 = ms.mesh(0)
        v1 = np.asarray(m0.transformed_vertex_matrix())
        f1 = np.asarray(m0.face_matrix()) if m0.face_number() else np.zeros((0, 3))
        changed = v1.shape != v0.shape or f1.shape != f0.shape or not np.array_equal(v1, v0) \
            or not np.array_equal(f1, f0)
        if changed:
            res.current = from_pymeshlab(m0, metadata=meta)
        if m0.has_vertex_scalar():
            q1 = np.asarray(m0.vertex_scalar_array(), dtype=float)
            if q0 is None or q1.shape != q0.shape or not np.allclose(q1, q0, equal_nan=True):
                res.quality = q1
        s1 = _selection_of(m0)
        if s1.shape != np.shape(s0) or not np.array_equal(s1, s0):
            res.selected = s1.astype(np.float32)
    return res


def reads_quality(name: str) -> bool:
    """``True`` when filter *name* uses MeshLab's per-vertex quality as input."""
    return bool(_READS_QUALITY.search(name))


def meshlab_defaults(obj: Mesh | PointCloud | None, name: str) -> dict[str, Any]:
    """The parameter values MeshLab would use for *obj* (sizes, counts and lengths
    follow the mesh), as the catalog's kinds expect them; lengths stay relative
    (``"1%"``) as in the catalog."""
    info = _info_or_none(name)
    if info is None or obj is None:
        return {}
    pml = require_pymeshlab()
    ms = pml.MeshSet()
    ms.add_mesh(to_pymeshlab(obj))
    try:
        values = ms.filter_parameter_values(name)
    except Exception:  # noqa: BLE001 — fall back to the catalog's defaults
        return {}
    out: dict[str, Any] = {}
    for p in info.params:
        if p.name not in values or p.kind in ("percent", "mesh", "matrix", "color", "text"):
            continue
        val = values[p.name]
        try:
            if p.kind == "enum":
                idx = int(val) if not isinstance(val, str) else (p.choices.index(val) if val in p.choices else 0)
                out[p.name] = p.choices[idx] if 0 <= idx < len(p.choices) else p.default
            elif p.kind == "vec3":
                out[p.name] = ", ".join(f"{float(v):.6g}" for v in np.asarray(val, dtype=float).ravel()[:3])
            elif p.kind == "bool":
                out[p.name] = bool(val)
            elif p.kind == "int":
                out[p.name] = int(val)
            elif p.kind == "float":
                out[p.name] = float(val)
        except Exception:  # noqa: BLE001
            continue
    return out


def _mesh_ids(ms: Any) -> list[int]:
    """Ids of the meshes in *ms* (they are not always 0..n-1 after deletions)."""
    ids, mid, found = [], 0, 0
    while found < ms.mesh_number() and mid < ms.mesh_number() + 1024:
        if ms.mesh_id_exists(mid):
            ids.append(mid)
            found += 1
        mid += 1
    return ids


def _plain(value: Any) -> Any:
    """Filter outputs as plain Python / NumPy values."""
    if type(value).__name__ == "BoundingBox":
        return {"min": np.asarray(value.min()).tolist(), "max": np.asarray(value.max()).tolist(),
                "diagonal": float(value.diagonal())}
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 16 else value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


# ──────────────────────────────────────────────────────────────────────────────
# The filter catalog
# ──────────────────────────────────────────────────────────────────────────────


def _cache_path() -> Path:
    """Per PyMeshLab version and per filter selection (editing :data:`_EXCLUDE` rebuilds it)."""
    import hashlib

    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "nvitk"
    key = hashlib.sha1("|".join(sorted(_EXCLUDE)).encode()).hexdigest()[:8]
    return root / f"pymeshlab-filters-{pymeshlab_version()}-{key}.json"


def _group_of(name: str) -> str:
    for group, prefixes in _GROUPS:
        if name.startswith(prefixes):
            return group
    return "MeshLab · Other"


def _humanize(name: str) -> str:
    text = name
    for prefix in ("meshing_", "generate_", "compute_", "apply_", "get_", "create_", "set_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.replace("_per_vertex", "").replace("_per_face", " (faces)").replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _param_label(name: str, kind: str) -> str:
    label = _PARAM_LABELS.get(name) or name.replace("_", " ").strip().capitalize()
    if kind == "percent":
        label += " (% or mm)"
    return label


def _parse_param(fname: str, pname: str, ty: str, text: str, enums: dict[str, list[str]]) -> FilterParam | None:
    ty = ty.strip()
    text = text.strip()
    if pname in _MESH_PARAMS and ty.startswith("int"):
        return FilterParam(pname, "mesh", "active", _param_label(pname, "mesh"))
    if ty == "bool":
        return FilterParam(pname, "bool", text == "True", _param_label(pname, "bool"))
    if ty == "int":
        try:
            return FilterParam(pname, "int", int(float(text)), _param_label(pname, "int"))
        except ValueError:
            return None
    if ty.startswith("float"):
        lo = hi = None
        bounds = re.search(r"\[min:\s*([-\d.e+]+);\s*max:\s*([-\d.e+]+)\]", text)
        if bounds:
            lo, hi = float(bounds.group(1)), float(bounds.group(2))
            text = text[: bounds.start()].strip()
        try:
            return FilterParam(pname, "float", float(text), _param_label(pname, "float"), min=lo, max=hi)
        except ValueError:
            return None
    if ty == "PercentageValue":
        return FilterParam(pname, "percent", text if text.endswith("%") else f"{text}%", _param_label(pname, "percent"))
    if ty == "str":
        value = text.strip("'\"")
        choices = enums.get(f"{fname}.{pname}")
        if choices:
            return FilterParam(pname, "enum", value if value in choices else choices[0],
                               _param_label(pname, "enum"), choices=tuple(choices))
        return FilterParam(pname, "text", value, _param_label(pname, "text"))
    if "float64[3]" in ty:
        nums = [float(v) for v in re.findall(r"[-\d.e+]+", text)][:3] or [0.0, 0.0, 0.0]
        return FilterParam(pname, "vec3", ", ".join(f"{v:g}" for v in nums), _param_label(pname, "vec3"))
    if "float64[4, 4]" in ty:
        return FilterParam(pname, "matrix", "1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1", _param_label(pname, "matrix"))
    if ty == "Color":
        return None  # colours are left at MeshLab's default
    return None


def _introspect_here() -> dict[str, Any]:
    """Parameter declarations and enum choices, read from PyMeshLab (prints to fd 1)."""
    import ctypes

    pml = require_pymeshlab()
    libc = ctypes.CDLL(None)

    def _capture(fn: Any, *args: Any, **kwargs: Any) -> str:
        sys.stdout.flush()
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            saved = os.dup(1)
            os.dup2(tmp.fileno(), 1)
            try:
                fn(*args, **kwargs)
            finally:
                libc.fflush(None)
                os.dup2(saved, 1)
                os.close(saved)
            tmp.seek(0)
            return tmp.read().decode("utf-8", "replace")

    tetra = (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
             np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32))
    decl: dict[str, list[list[str]]] = {}
    enums: dict[str, list[str]] = {}
    for fname in pml.filter_list():
        if any(s in fname for s in _EXCLUDE):
            continue
        txt = _capture(pml.print_filter_parameter_list, fname)
        params = re.findall(r"^\t(\w+) : (.+?) = (.*)$", txt, re.M)
        decl[fname] = [list(p) for p in params]
        for pname, ty, _dv in params:
            if ty.strip() != "str" or pname in _FREE_TEXT:
                continue
            ms = pml.MeshSet()
            for _ in range(2):
                ms.add_mesh(pml.Mesh(*tetra))
            try:
                _capture(ms.apply_filter, fname, **{pname: "\u0001nvitk-probe"})
            except Exception as exc:  # noqa: BLE001
                found = re.search(r"Possible values are (.*)", str(exc))
                if found:
                    enums[f"{fname}.{pname}"] = re.findall(r"'([^']*)'", found.group(1))
    return {"version": pymeshlab_version(), "decl": decl, "enums": enums}


def _introspect() -> dict[str, Any]:
    """:func:`_introspect_here` in a child process (it redirects the process's stdout)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "catalog.json"
        subprocess.run([sys.executable, "-m", "nvitk.meshlab.pymeshlab_filters", "--catalog", str(out)],
                       check=True, capture_output=True, timeout=600)
        return json.loads(out.read_text())


@lru_cache(maxsize=1)
def _catalog_raw() -> dict[str, Any]:
    path = _cache_path()
    try:
        raw = json.loads(path.read_text())
        if raw.get("version") == pymeshlab_version():
            return raw
    except Exception:  # noqa: BLE001
        pass
    raw = _introspect()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
    except Exception:  # noqa: BLE001 — an unwritable cache only costs time
        pass
    return raw


@lru_cache(maxsize=1)
def _catalog() -> tuple[FilterInfo, ...]:
    raw = _catalog_raw()
    order = {g: i for i, (g, _p) in enumerate(_GROUPS)}
    infos = []
    for fname, params in raw["decl"].items():
        if any(x in fname for x in _EXCLUDE):
            continue
        parsed = tuple(p for p in (_parse_param(fname, *row, raw["enums"]) for row in params) if p is not None)
        group = _group_of(fname)
        label, desc = _FRIENDLY.get(fname, (_humanize(fname), ""))
        inputs: tuple[str, ...] = () if fname.startswith("create_") else (
            ("mesh", "points") if fname in _POINT_CLOUD_FILTERS else ("mesh",))
        pick = next((p.name for p in parsed if p.kind == "vec3" and p.name in _PICK_PARAMS), "")
        infos.append(FilterInfo(fname, label, group, parsed, desc, inputs, pick))
    infos.sort(key=lambda f: (order.get(f.group, len(order)), f.name not in _FRIENDLY, f.label.lower()))
    return tuple(infos)


def meshlab_filters() -> list[FilterInfo]:
    """The MeshLab filters offered for nvitk surfaces (empty without PyMeshLab)."""
    if not pymeshlab_available():
        return []
    return list(_catalog())


def meshlab_groups() -> list[str]:
    """The groups of :func:`meshlab_filters`, in menu order."""
    seen: list[str] = []
    for info in meshlab_filters():
        if info.group not in seen:
            seen.append(info.group)
    return seen


def _info_or_none(name: str) -> FilterInfo | None:
    try:
        return next((f for f in meshlab_filters() if f.name == name), None)
    except Exception:  # noqa: BLE001 — running a filter does not need the catalog
        return None


def meshlab_filter(name: str) -> FilterInfo:
    """The :class:`FilterInfo` of filter *name*."""
    info = _info_or_none(name)
    if info is None:
        raise KeyError(f"No MeshLab filter {name!r} (or it is not offered).")
    return info


__all__ = [
    "DOCS_URL",
    "FilterInfo",
    "FilterParam",
    "MeshLabResult",
    "add_to_meshset",
    "meshlab_defaults",
    "reads_quality",
    "from_pymeshlab",
    "meshlab_filter",
    "meshlab_filters",
    "meshlab_groups",
    "pymeshlab_available",
    "pymeshlab_version",
    "require_pymeshlab",
    "run_meshlab_filter",
    "to_pymeshlab",
]


if __name__ == "__main__":  # pragma: no cover — the catalog child process
    if len(sys.argv) == 3 and sys.argv[1] == "--catalog":
        Path(sys.argv[2]).write_text(json.dumps(_introspect_here()))
