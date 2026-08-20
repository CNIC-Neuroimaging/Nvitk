# Types & Transform

## The `Image` and `Mesh` containers

`nvitk.types.image.Image` holds voxel data (NumPy or CuPy), spatial metadata (spacing,
affine, orientation), optional named axes, and DICOM tags — the object every I/O, filter,
and measurement function in the toolkit passes around.

```{code-block} python
from nvitk.io import imread

img = imread("study/ct.nii.gz", backend="numpy")
print(img.axes, img.shape, img.modality, img.orientation)
```

`nvitk.types.mesh.Mesh` stores vertices, faces, and metadata (affine, spacing, label id) for
surface reconstructions — used directly as napari Surface-layer data in {doc}`the GUI
<../gui/index>`.

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
| `marching_cubes_binary` / `marching_cubes_multilabel` | Lower-level marching-cubes implementations. |

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
