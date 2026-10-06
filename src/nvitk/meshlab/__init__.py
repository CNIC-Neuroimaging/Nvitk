"""Meshes and point clouds: build, clean, smooth, remesh, measure, align — and over time.

The MeshLab / ParaView basics for nvitk's surface types
(:class:`~nvitk.types.Mesh`, :class:`~nvitk.types.PointCloud`,
:class:`~nvitk.types.MeshSeries`):

- :mod:`~nvitk.meshlab.marching_cubes` — surfaces from masks and label maps;
- :mod:`~nvitk.meshlab.io` — STL, OBJ, OFF, PLY, VTK/VTP, GIfTI, XYZ/CSV/PCD, series + ``.pvd``;
- :mod:`~nvitk.meshlab.cleaning` — duplicates, degenerate faces, components, holes, orientation;
- :mod:`~nvitk.meshlab.smoothing` — Laplacian and Taubin;
- :mod:`~nvitk.meshlab.remeshing` — decimation, subdivision, clipping, convex hull, voxel remesh;
- :mod:`~nvitk.meshlab.measure` — area, volume, curvature, distances, cross-sections;
- :mod:`~nvitk.meshlab.transform` — affine transforms, principal axes, ICP;
- :mod:`~nvitk.meshlab.pointcloud` — sampling, downsampling, outliers, normals, reconstruction;
- :mod:`~nvitk.meshlab.voxelize` — closed meshes back to masks on any image grid;
- :mod:`~nvitk.meshlab.temporal` — 3D+t series: per-frame surfaces, tracking, motion, smoothing;
- :mod:`~nvitk.meshlab.topology` — edges, boundaries, components, Euler characteristic, genus;
- :mod:`~nvitk.meshlab.vessels` — centerlines, diameter profiles, stenosis, bifurcations, thickness;
- :mod:`~nvitk.meshlab.edit` — local edits around a point: erase, keep a piece, push/pull, smooth;
- :mod:`~nvitk.meshlab.sampling` — image values on surfaces (probe), signed distance maps;
- :mod:`~nvitk.meshlab.pymeshlab_filters` — MeshLab's filters through PyMeshLab (optional).
"""

from .cleaning import (
    clean,
    fill_holes,
    flip_normals,
    keep_components,
    merge_vertices,
    orient_faces,
    remove_small_components,
    split_components,
)
from .convert import from_pyvista, mesh_to_point_cloud, to_pyvista
from .io import read_mesh, read_mesh_series, read_point_cloud, read_pvd, read_surface, write_mesh_series, write_surface
from .marching_cubes import (
    marching_cubes_binary,
    marching_cubes_multilabel,
    mesh_from_image,
)
from .measure import (
    cross_section,
    curvature,
    distance_to_surface,
    mesh_metrics,
    point_cloud_metrics,
    principal_axes,
    surface_area,
    surface_distance_stats,
    volume,
)
from .pointcloud import (
    estimate_normals,
    orient_normals,
    points_to_mask,
    poisson_reconstruct,
    points_from_mask,
    random_downsample,
    reconstruct_surface,
    remove_outliers,
    sample_surface,
    voxel_downsample,
)
from .remeshing import clip_plane, convex_hull, decimate, isotropic_remesh, remesh_voxel, subdivide
from .smoothing import laplacian_smooth, taubin_smooth
from .temporal import (
    interpolate_frames,
    propagate_mesh,
    series_from_image,
    series_metrics,
    smooth_in_time,
    vertex_displacement,
)
from .topology import genus, is_watertight
from .edit import (
    connected_selection,
    delete_selected,
    erase_points,
    erase_sphere,
    grow_selection,
    keep_piece,
    keep_selected,
    piece_at,
    sculpt,
    smooth_spot,
)
from .pymeshlab_filters import meshlab_filters, pymeshlab_available, run_meshlab_filter
from .transform import align_icp, align_principal_axes, apply_affine, compose_affine, icp, transform_surface
from .sampling import sample_image, sample_on_surface, signed_distance_image
from .vessels import (
    bifurcations,
    branch_map,
    branch_metrics,
    branch_profiles,
    mesh_centerlines,
    radius_map,
    thickness_map,
    vessel_section_at,
)
from .voxelize import mesh_to_mask

__all__ = [
    "branch_map",
    "connected_selection",
    "delete_selected",
    "grow_selection",
    "keep_selected",
    "isotropic_remesh",
    "meshlab_filters",
    "pymeshlab_available",
    "run_meshlab_filter",
    "compose_affine",
    "erase_points",
    "erase_sphere",
    "keep_piece",
    "piece_at",
    "radius_map",
    "sculpt",
    "smooth_spot",
    "bifurcations",
    "branch_metrics",
    "branch_profiles",
    "mesh_centerlines",
    "orient_normals",
    "points_to_mask",
    "poisson_reconstruct",
    "sample_image",
    "sample_on_surface",
    "signed_distance_image",
    "thickness_map",
    "vessel_section_at",
    "align_icp",
    "align_principal_axes",
    "apply_affine",
    "clean",
    "clip_plane",
    "convex_hull",
    "cross_section",
    "curvature",
    "decimate",
    "distance_to_surface",
    "estimate_normals",
    "fill_holes",
    "flip_normals",
    "from_pyvista",
    "genus",
    "icp",
    "interpolate_frames",
    "is_watertight",
    "keep_components",
    "laplacian_smooth",
    "marching_cubes_binary",
    "marching_cubes_multilabel",
    "merge_vertices",
    "mesh_from_image",
    "mesh_metrics",
    "mesh_to_mask",
    "mesh_to_point_cloud",
    "orient_faces",
    "point_cloud_metrics",
    "points_from_mask",
    "principal_axes",
    "propagate_mesh",
    "random_downsample",
    "read_mesh",
    "read_mesh_series",
    "read_point_cloud",
    "read_pvd",
    "read_surface",
    "reconstruct_surface",
    "remesh_voxel",
    "remove_outliers",
    "remove_small_components",
    "sample_surface",
    "series_from_image",
    "series_metrics",
    "smooth_in_time",
    "split_components",
    "subdivide",
    "surface_area",
    "surface_distance_stats",
    "taubin_smooth",
    "to_pyvista",
    "transform_surface",
    "vertex_displacement",
    "volume",
    "voxel_downsample",
    "write_mesh_series",
    "write_surface",
]
