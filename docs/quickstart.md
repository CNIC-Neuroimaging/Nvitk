# Quickstart

## The dual NumPy/CuPy backend

Every nvitk module is written once against a backend-aware `np`/`ndi`/`scipy` proxy. Call
{func}`~nvitk.core.setup` once per module (or script) to inject those proxies, then switch
backends per-call, per-scope, or process-wide — no branching in your own code.

```python
from nvitk.core import setup, using, get_current_backend

setup(globals())  # injects backend-aware np, ndi, scipy into this module's globals
print(get_current_backend())  # "numpy" by default, or "cupy" if NVITK_BACKEND=cupy

with using("cupy"):
    x = np.asarray([1, 2, 3])  # runs on GPU inside this scope only
```

See {doc}`api/core-backend` for the full backend API (`using_backend`, `set_global_backend`,
array-conversion helpers, and the `NVITK_BACKEND`/`NVITK_CUDA_DEVICE` environment variables).

## Reading, processing, and saving an image

`nvitk.io`'s `imread`/`imsave` dispatch on file extension (or an explicit `force_type`) and
return an {class}`~nvitk.types.image.Image` — a container for voxel data plus spacing,
affine, DICOM tags, and orientation.

```python
from nvitk.core import setup
from nvitk.io import imread, imsave
from nvitk.measure import volume_cc, Measurer

setup(globals())

img = imread("study/pet", force_type="dicom", backend="gpu")
mask = imread("mask.nii.gz")

print(volume_cc(mask))
summary = Measurer(img, mask).volume() | Measurer(img, mask).suv(kinds=("bw",))
imsave("out/pet_copy.nii.gz", img)
```

## The same operations from the command line

Every image-processing module also ships a CLI entry point accepting `-i`/`-o` and an
optional `--submit local|sge` for cluster dispatch (cluster defaults live in
`.nvitk/sge.json`):

```bash
nvitk-restore bilateral -i pet.nii.gz -o pet_denoised.nii.gz --backend gpu
nvitk-morph dilate -i mask.nii.gz -o mask_dil.nii.gz --footprint 2
nvitk-filter sliding-threshold -i cd.nii.gz -o mask.nii.gz --dim 3d
nvitk-measure volume -i mask.nii.gz -o vol.txt
nvitk-transform resample -i pet.nii.gz -r ct.nii.gz -o pet_on_ct.nii.gz
```

Run `pyhelp` for an interactive, searchable tree of every registered command (`pyhelp
--no-interactive` for a flat listing in CI/scripts) — see {doc}`api/cli-catalog`.

## Opening the GUI instead

```bash
nvitk-gui
```

Opens a Napari workbench with the same tool catalog as a form-driven dock, plus mesh
reconstruction, a dedicated statistics workbench, and pipeline export — see
{doc}`gui/index`.

## Where to next

- {doc}`api/index` — the full library reference, organized by topic.
- {doc}`gui/index` and {doc}`stats-gui/index` — the two GUI workbenches.
- {doc}`pipelines/index` — end-to-end batch pipelines (PESA-Fat, QVTPy).
