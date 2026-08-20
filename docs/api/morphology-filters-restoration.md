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
| `centerline` | Skeleton/centerline extraction from tubular structures (vessels). |
| `centerline_siphon` | ICA siphon-specific centerline correction — see the worked methodology in the repo's `notebooks/exploration/pesabrain-anatomy/methods.md`. |
| `polyline_graph`, `mst_bridge` | Graph representations of centerlines and minimum-spanning-tree bridging between disconnected segments. |

```{code-block} bash
nvitk-morph dilate -i mask.nii.gz -o mask_dil.nii.gz --footprint 2
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
