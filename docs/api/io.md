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
| `imread` | Reader dispatch by extension or `force_type` (`nifti`, `dicom`, `tiff`, `mha`, `pil`, `nd2`, `b2nd`, `pkl`, ...) — returns an {class}`~nvitk.types.image.Image`. |
| `imsave` | Writer dispatch, mirrors `imread`'s format set. |
| `imshow` | Slice view, orthogonal view, mosaic, or animation, for quick inspection. |
| `convert_image`, `swapaxes` | Format conversion and axis-order helpers. |

Per-format implementations live in `nvitk.io.readers` (`b2nd`, `dicom`, `mha`, `nd2`,
`nifti`, `pil`, `pkl`, `tiff`) and `nvitk.io.writers` (which cover `nifti`, `mha`, `tiff`, `pil`).

## Preprocessed cases (`.b2nd` / `.pkl`)

nnU-Net and nnssl store preprocessed volumes as a Blosc2 array shaped `(C, Z, Y, X)` beside a
`.pkl` sidecar holding the source SimpleITK geometry and the crop/resample bookkeeping — for
example the ToPBrain outputs under
`$nnssl_preprocessed/<Dataset>/<plans>_<config>/.../<session>/TOF.b2nd`. `read_b2nd` reassembles
the world geometry those two files only imply: it takes the target spacing from the
`*Plans*.json` it finds by walking up to the folder named by a configuration's `data_identifier`,
shifts the origin by the crop bounding box, and returns the volume in `XYZ` order with an RAS
affine — so a preprocessed case overlays exactly on the scan it came from.

```{code-block} python
from nvitk.io import imread

img = imread(".../pesa_tof-PESA1521/ses-1/TOF.b2nd")   # (X, Y, Z), affine + orientation
img = imread(".../ses-1")                              # every .b2nd in the folder
img = imread(".../ses-1/TOF.pkl")                      # sidecar → its companion array
```

A length-1 channel axis is dropped by default (`squeeze_channel=False` keeps it as `XYZC`, and
`channel=i` decompresses just one channel of a multi-modality case). The raw sidecar survives in
`img.metadata["preprocessing_properties"]`, with `spacing_source` recording whether the spacing
came from the plans file, the sidecar, or a shape-ratio fallback. In the GUI these files open
like any other volume — file dialog, drag & drop, or `open_paths_with_nvitk`.

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
