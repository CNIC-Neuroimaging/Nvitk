# Types & Transform

## The `Image` and surface containers

`nvitk.types.image.Image` holds voxel data (NumPy or CuPy), spatial metadata (spacing,
affine, orientation), optional named axes, and DICOM tags — the object every I/O, filter,
and measurement function in the toolkit passes around.

```{code-block} python
from nvitk.io import imread

img = imread("study/ct.nii.gz", backend="numpy")
print(img.axes, img.shape, img.modality, img.orientation)
```

`nvitk.types.mesh` holds the surface types:

| Type | Holds |
|---|---|
| `Mesh` | Triangles (`vertices`, `faces`), per-vertex `point_data` and per-face `cell_data` (curvature, distances, normals), metadata (affine, spacing, label id, `space`). Areas, normals, bounds as properties. |
| `PointCloud` | Points with per-point data (normals, labels, intensities). |
| `MeshSeries` | A mesh per time frame plus frame times; `shared_topology` when the frames correspond vertex by vertex. |

Meshes built from images are in world millimetres; they are napari Surface data in
{doc}`the GUI <../gui/index>`.

## Mesh reconstruction

`nvitk.meshlab` builds a `Mesh` from a binary or multi-label `Image` via marching cubes:

```{code-block} python
from nvitk.io import imread
from nvitk.meshlab import mesh_from_image

img = imread("mask.nii.gz")
mesh = mesh_from_image(img)  # binary; use multilabel=True for label maps
```

| Function | Purpose |
|---|---|
| `mesh_from_image` | High-level entry point — binary or multi-label. |
| `marching_cubes_binary` / `marching_cubes_multilabel` | Lower-level marching-cubes implementations (outward-oriented faces). |

## Working with meshes and point clouds

The rest of `nvitk.meshlab` is the MeshLab / ParaView toolbox for those types:

| Module | Purpose |
|---|---|
| `io` | `read_surface` / `write_surface` (STL, OBJ, OFF, PLY, VTK/VTP, GIfTI, XYZ/CSV, PCD), `read_mesh_series` / `write_mesh_series` / `read_pvd` for time series. |
| `cleaning` | `clean` (weld duplicates, drop degenerate/duplicate faces), `keep_components` / `split_components`, `fill_holes`, `orient_faces`, `flip_normals`. |
| `smoothing` | `taubin_smooth` (volume-preserving), `laplacian_smooth`, `smooth_point_data`. |
| `remeshing` | `decimate`, `subdivide`, `isotropic_remesh` (MeshLab), `clip_plane`, `clip_box`, `convex_hull`, `remesh_voxel`. |
| `measure` | `mesh_metrics`, `surface_area`, `volume`, `curvature`, `distance_to_surface`, `surface_distance_stats`, `cross_section`, `principal_axes`. |
| `transform` | `apply_affine`, `compose_affine`, `transform_surface`, `align_principal_axes`, `icp` / `align_icp`. |
| `pointcloud` | `sample_surface`, `points_from_mask`, `points_to_mask`, `voxel_downsample`, outlier removal, `estimate_normals` / `orient_normals` (minimum-spanning-tree propagation), `reconstruct_surface` (`poisson`, `imls`, `implicit`, `alpha_shape`, `convex_hull`). |
| `voxelize` | `mesh_to_mask`: a closed surface back onto any image grid. |
| `sampling` | `sample_image` (image values at world points), `sample_on_surface` (optionally averaged along the normal, inward or outward), `signed_distance_image`. |
| `vessels` | `mesh_centerlines` (skeleton → smoothed branches), `branch_profiles` (area, perimeter, min / max / equivalent diameter, circularity along each branch), `branch_metrics` (length, tortuosity, curvature, stenosis), `bifurcations`, `branch_map` (nearest branch per vertex), `vessel_section_at`, `thickness_map`, `radius_map`. |
| `edit` | Local, point-driven edits: `erase_sphere`, `piece_at` / `keep_piece`, `sculpt`, `smooth_spot`, `erase_points`; per-vertex selections: `delete_selected`, `keep_selected`, `grow_selection`, `connected_selection`, `faces_of_selection`. |
| `pymeshlab_filters` | MeshLab's filters through PyMeshLab: `run_meshlab_filter` (any filter by name on a `Mesh` / `PointCloud`), `meshlab_filters` (the grouped, typed catalog the GUI builds its forms from), `meshlab_defaults`, `to_pymeshlab` / `from_pymeshlab`. |
| `temporal` | `series_from_image` (3D+t masks), `series_metrics`, `propagate_mesh` (vertex correspondence over time), `vertex_displacement`, `smooth_in_time`, `interpolate_frames`. |
| `topology` | Edges, boundary loops, components, Euler characteristic, genus, `is_watertight`. |

```{code-block} python
from nvitk.io import imread
import nvitk.meshlab as ml

mask = imread("lv_4d.nii.gz")                       # 3D+t
series = ml.series_from_image(mask, smooth_iterations=10)
tracked = ml.propagate_mesh(series)                 # shared topology
volumes = ml.series_metrics(tracked)["volume"]      # mm³ per frame
motion = ml.vertex_displacement(tracked)            # (T, N) mm
```

Surface reconstruction from points defaults to Poisson reconstruction on a grid sized
from the point spacing (`resolution=0`); normals are estimated and oriented consistently first,
so open or noisy clouds still give one closed surface. `trim` removes the parts of it
farther from the points than that many typical point gaps. `method="screened_poisson"` and
`"ball_pivoting"` run MeshLab's versions (ball pivoting interpolates the points and keeps
holes open).

```{code-block} python
cloud = ml.points_from_mask(mask_3d, max_points=20000)
surface = ml.reconstruct_surface(cloud, method="poisson")

vessel = ml.mesh_from_image(lumen_mask)                 # binary mask → one Mesh
branches = ml.mesh_centerlines(vessel)
profiles = ml.branch_profiles(vessel, branches, every=1.0)   # one section per mm
table = [ml.branch_metrics(b) for b in branches]             # length, tortuosity, stenosis…
```

### MeshLab filters

`nvitk.meshlab.pymeshlab_filters` runs any of MeshLab's filters on nvitk types. The mesh's
fields travel as custom per-vertex attributes (usable in MeshLab's formulas; spaces and symbols
become `_`, so *wall thickness* is `wall_thickness`), one of them
as MeshLab's *quality* if asked, and the result is sorted into new meshes, the changed
input, a new quality or selection, and returned measures:

```{code-block} python
from nvitk.meshlab.pymeshlab_filters import run_meshlab_filter

res = run_meshlab_filter(mesh, "meshing_isotropic_explicit_remeshing", {"targetlen": "1%"})
remeshed = res.current

res = run_meshlab_filter(mesh, "get_hausdorff_distance", {"sampledmesh": "active", "targetmesh": other})
res.values["mean"], res.quality                  # mm; per-vertex distance on `mesh`

res = run_meshlab_filter(cloud, "generate_surface_reconstruction_screened_poisson", {"depth": 9})
surface = res.meshes[0]                          # normals are estimated when the cloud has none
```

Lengths are `"1%"` (of the bounding-box diagonal) or plain numbers (mm); a filter's other
mesh is passed as a `Mesh` / `PointCloud`. Parameters left out keep the defaults MeshLab
computes for the input.

## Geometric transforms

`nvitk.transform` covers resampling and orientation:

| Module | Purpose |
|---|---|
| `resampling` | `resample_to`, `resample_pet_to_mask`, `resample_mask_to_pet` — grid-to-grid resampling. |
| `isotropy` | Resample to isotropic voxel spacing. |
| `reorient` | Canonicalize axis order/orientation. |
| `rotate` / `rotation` | Rotation about arbitrary axes, incl. Z-rotation helpers used by several pipelines. |
| `swap_axes` | Axis-order manipulation. |
| `oblique` | Oblique slice extraction. |
| `threaded` | Slab-parallel `affine_transform` / `map_coordinates` on the shared worker pool (`map_coordinates` bit-identical to SciPy; slab layout independent of the thread count). `resample_to(..., threaded=True)` opts in. |
| `fourier` | K-space: `kspace` / `inverse_kspace` (centred, orthonormal; 3D or 2D slice-wise; 3D+t per frame), display components, radial `kspace_filter`. CPU (`scipy.fft`, multi-threaded) or GPU (CuPy). |
| `temporal` | 3D+t: `extract_frame`, `temporal_projection` (incl. time-to-peak and AUC), `time_intensity_curve`, `stack_frames`. |
| `interpolation` | `resample_axes` (up/down-sample any axes, time included, by factor / spacing / size; shape-based for masks; field of view kept), `block_reduce_axes`, `interpolate_mask_slices` (fill between sparsely annotated slices), `fill_missing_slices`. |

```{code-block} python
from nvitk.transform import isotropy, resample_pet_to_mask

pet_iso = isotropy(pet)
pet_on_mask = resample_to(pet_iso, mask)
```

```{seealso}
Full generated reference: [`nvitk.types`](../autoapi/nvitk/types/index),
[`nvitk.transform`](../autoapi/nvitk/transform/index),
[`nvitk.meshlab`](../autoapi/nvitk/meshlab/index).
```
