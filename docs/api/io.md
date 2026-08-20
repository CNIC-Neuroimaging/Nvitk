# I/O

`nvitk.io` reads and writes every image format the toolkit works with, dispatching on file
extension or an explicit `force_type`, and converting between acquisition formats and NIfTI.

## Reading, writing, and displaying

```{code-block} python
from nvitk.io import imread, imsave, imshow

img = imread("study/pet", force_type="dicom", backend="gpu")
imshow(img, axis=2, index="mid")
imsave("out/pet_copy.nii.gz", img)
```

| Function | Purpose |
|---|---|
| `imread` | Reader dispatch by extension or `force_type` (`nifti`, `dicom`, `tiff`, `mha`, `pil`, `nd2`, ...) — returns an {class}`~nvitk.types.image.Image`. |
| `imsave` | Writer dispatch, mirrors `imread`'s format set. |
| `imshow` | Slice view, orthogonal view, mosaic, or animation, for quick inspection. |
| `convert_image`, `swapaxes` | Format conversion and axis-order helpers. |

Per-format implementations live in `nvitk.io.readers` (`dicom`, `mha`, `nd2`, `nifti`,
`pil`, `tiff`) and `nvitk.io.writers` (mirroring the same set).

## Format conversors

CLI-facing conversion tools, each also a registered entry point:

| Command | Module | Converts |
|---|---|---|
| `dcm2nii` | `nvitk.io.conversors.dcm2nii` | DICOM → NIfTI (incl. RT structs, tissue segmentation, Zeiss-specific handling) |
| `stl2nifti` | `nvitk.io.conversors.stl2nifti` | Surface mesh (STL) → labelmap/NIfTI |
| `phase2volume` | `nvitk.io.conversors.phase2volume` | Phase-contrast MRI → velocity volume and derivatives |
| `nikon2nifti` | `nvitk.io.conversors.nikon2nifti` | Nikon microscopy → NIfTI |

## ANTs interop

`nvitk.io.ants_bridge` provides `require_ants`/`require_antspynet` guards and
`to_ants_image` conversion, used wherever a tool needs to hand an `Image` off to ANTsPy or
ANTsPyNet (see {doc}`registration` and {doc}`segmentation`).

```{seealso}
Full generated reference: [`nvitk.io`](../autoapi/nvitk/io/index).
```
