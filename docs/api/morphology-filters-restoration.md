# Morphology, Filters & Restoration

Three complementary modules for preparing and cleaning up binary/label masks and raw
intensity volumes before measurement.

## Morphology

`nvitk.morphology` — binary/label operations, connected components, and centerline
extraction:

| Module | Purpose |
|---|---|
| `binary` | Dilate, erode, open, close, fill-holes, and related binary ops. |
| `components` | Connected-component labeling and filtering. |
| `centerline` | Skeleton/centerline extraction from tubular structures (vessels); every entry point takes `method=` (below). |
| `skeleton_methods` | The skeletonization methods: `vmtk`, `native` (scikit-image thinning, re-centred) and `raycast` (experimental). `centerline_network()` returns sub-voxel branch polylines with radii; `method_skeleton()` rasterizes them to a 1-voxel skeleton so voxel-graph tools work with any method. |
| `centering` | Re-centres skeleton paths on the lumen (cross-section centroid + inscribed-radius ascent) and measures the inscribed radius as the distance to the surface. The GUI's *Centre centerline on the lumen* tools (Tools and Mesh panels) apply it to any centerline layer. |
| `raycast_skeleton` | **Experimental.** EXPIGEO ray-cast skeletonization (inverted-normal ray probing, spatial-median vertices, MST, spur pruning) and its geometric descriptors (radius gradient, termination likelihood, blockage, sphericity, projected flow). |
| `centerline_siphon` | ICA siphon-specific centerline correction — see the worked methodology in the repo's `notebooks/exploration/pesabrain-anatomy/methods.md`. |
| `polyline_graph`, `mst_bridge` | Graph representations of centerlines and minimum-spanning-tree bridging between disconnected segments. |

```{code-block} bash
nvitk-morph dilate -i mask.nii.gz -o mask_dil.nii.gz --footprint 2
```

### Skeletonization methods

| Method | How | Default for |
|---|---|---|
| `vmtk` | VMTK Voronoi centerlines on the mask's surface, seeded at the skeleton end points (needs {doc}`vmtk`); `vmtk_coverage="full"` (default) traces every branch, loop and separate piece, `"tree"` one run from the widest end | GUI tools, TOF morphometrics |
| `native` | scikit-image 3D thinning; networks are re-centred on the lumen | library calls, so the validated 4D-flow qvtpy stages keep their outputs |
| `raycast` | EXPIGEO ray probing of the surface (Layachi et al., MICCAI 2026) | nothing: experimental, opt-in |

```{code-block} python
from nvitk.morphology.skeleton_methods import centerline_network, method_skeleton

net = centerline_network(mask, method="vmtk", spacing=(0.5, 0.5, 0.5))  # branches, radii, junctions
skel = method_skeleton(mask, method="raycast", spacing=(0.5, 0.5, 0.5))  # 1-voxel skeleton
```

## Filters

`nvitk.filters` — intensity/structure filters, exposed as submodules rather than a flat
function list:

| Module | Purpose |
|---|---|
| `sliding_threshold` | Local adaptive thresholding. |
| `hessian` | Hessian-based structure filters. |
| `jerman` | Jerman vesselness filter. |
| `snakes` | Active-contour (snakes) segmentation refinement. |

```{code-block} bash
nvitk-filter sliding-threshold -i cd.nii.gz -o mask.nii.gz --dim 3d
```

## Restoration

`nvitk.restoration` — denoising and correction of raw intensity volumes:

| Module | Purpose |
|---|---|
| `bilateral` | Edge-preserving bilateral denoising (CPU/GPU). |
| `n4_bias` | N4 bias-field correction (ANTs-backed). |
| `mri_super_resolution` | ANTsPyNet-backed MRI super-resolution. |
| `_cuda_kernels` | Raw CUDA kernels for the GPU restoration paths — compiled lazily on first call, so importing `nvitk.restoration` never requires a CUDA-capable CuPy. |

```{code-block} bash
nvitk-restore bilateral -i pet.nii.gz -o pet_denoised.nii.gz --backend gpu
```

```{seealso}
Full generated reference: [`nvitk.morphology`](../autoapi/nvitk/morphology/index),
[`nvitk.filters`](../autoapi/nvitk/filters/index),
[`nvitk.restoration`](../autoapi/nvitk/restoration/index).
```
