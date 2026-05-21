# Nvitk

## Neuro-Vascular Imaging ToolKit

## Installation (conda recommended)

This project depends on packages that are **only available via conda** (for example `mrtrix3`, `freeimage`, and `libstdcxx-ng`). From the repository root:

```bash
./install.sh cpu    # env name: nvitk
./install.sh gpu    # env name: gpu-nvitk (CUDA 13 stack, see below)
```

Activate with `conda activate nvitk` or `conda activate gpu-nvitk`.

> **Note:** The CPU profile does not install a CUDA-enabled PyTorch or CuPy. Use `./install.sh gpu` if you need GPU acceleration.

For a hierarchical catalog of CLI entry points, run **`pyhelp`** after activation (interactive tree on a TTY; use `pyhelp --no-interactive` for CI/pipes).

### GPU / CUDA (CUDA 13)

The GPU environment targets **CUDA 13** (PyTorch wheels from the `cu130` index and `cupy-cuda13x`; see `env/environment-gpu.yml`).

---

## User guide

| Section | What it covers |
|--------|----------------|
| [Computing backend](#computing-backend-numpy-and-cupy) | `using()`, env vars, `setup()` |
| [The `Image` container](#the-image-container) | Voxels, metadata, axes |
| [Command-line tools](#command-line-tools) | Scripts from `pyproject.toml` / `pyhelp` |
| [Image module CLIs](#image-module-clis) | `nvitk-morph`, `nvitk-restore`, filters, measure, transform |
| [nvitk-gui](#nvitk-gui) | Napari workbench for quick tool tests |
| [Mesh reconstruction](#mesh-reconstruction) | `nvitk.types.Mesh`, `nvitk.meshlab` |
| [Image I/O](#image-io-read-save-and-display) | `imread`, `imsave`, `imshow` |
| [Measurements](#measurements-and-metrics) | Volumes, SUV, overlap, `Measurer` |
| [Segmentation](#segmentation) | Labels, hemispheres, TotalSegmentator |
| [Geometric transforms](#geometric-transforms) | Resampling, isotropy, rotation |
| [PESA fat pipelines](#pesa-fat-pipelines) | Batch CT / PET / Dixon CLIs |

### Computing backend (NumPy and CuPy)

- **Environment:** `NVITK_BACKEND` (`auto`, `numpy`, `cupy`, ...), optional `NVITK_CUDA_DEVICE`, `NVITK_WARN_ON_FALLBACK`.
- **Scoped switch:** `with using("cupy"):` temporarily forces GPU backend.
- **Proxy setup:** `setup(globals())` injects backend-aware `np`, `ndi`, and `scipy`.

```python
from nvitk.core import setup, using, get_current_backend

setup(globals())
print(get_current_backend())

with using("cupy"):
    x = np.asarray([1, 2, 3])
```

### The `Image` container

`nvitk.types.image.Image` holds voxel **data** (NumPy or CuPy), **metadata** (spacing, affine, DICOM tags), optional **axes**, and **orientation**.

```python
from nvitk.io import imread

img = imread("study/ct.nii.gz", backend="numpy")
print(img.axes, img.shape, img.modality, img.orientation)
```

### Command-line tools

Same list as **`pyhelp`** and `src/nvitk/util/list_cli_commands.py` parse from `pyproject.toml`:

| Command | Role |
|---------|------|
| `dcm2nii` | DICOM -> NIfTI |
| `stl2nifti` | STL -> NIfTI |
| `phase2volume` | Phase data -> volume |
| `nikon2nifti` | Nikon -> NIfTI |
| `nvitk-totalseg` | TotalSegmentator wrapper (local / SGE) |
| `nvitk-eicab` | eICAB TOF / Circle-of-Willis segmentation (local / SGE) |
| `pyhelp` | List commands + backend reminder |
| `nvitk-pesa-fat` | PESA fat batch driver |
| `nvitk-pesa-fat-ctpet` | PESA fat CT/PET pipeline |
| `nvitk-pesa-fat-dixon` | PESA fat Dixon pipeline |

```bash
pyhelp                    # interactive colored tree; Enter on a command to select it
pyhelp --no-interactive   # full static tree (colors)
pyhelp --flat             # legacy flat listing
# Bash: put the selected command on your current prompt line (recommended)
source scripts/pyhelp-shell.bash && pyhelp-select
# Or permanent wrapper:
#   alias pyhelp='eval "$(pyhelp --shell 2>/dev/tty)"'
# Capture only: cmd=$(pyhelp --pick 2>/dev/tty)
nvitk-pesa-fat-ctpet --help
```

**Image processing** commands are grouped under *Image Processing* in `pyhelp` (conversion, segmentation, registration, filters, morphology, restoration, measure, transform). **Pipelines** (PESA-fat, qvtpy, …) and **General** (`pyhelp`, `nvitk-gui`) are separate top-level sections.

| Command | Role |
|---------|------|
| `nvitk-morph` | Morphology (dilate, centerline, siphon-correct, …) |
| `nvitk-restore` | Restoration (bilateral) |
| `nvitk-filter` | Filters (sliding-threshold) |
| `nvitk-measure` | Metrics (volume, SUV, dice, surface) |
| `nvitk-transform` | Resample, isotropy, oblique slice |
| `nvitk-gui` | Napari GUI for prototyping (optional extra) |

### Image module CLIs

Module CLIs accept `-i` / `-o` paths and optional `--submit local|sge` (same SGE pattern as segmentation pipelines). Configure cluster defaults in `.nvitk/sge.json` under `pipelines.image_tools`.

```bash
nvitk-restore bilateral -i pet.nii.gz -o pet_denoised.nii.gz --backend gpu
nvitk-morph dilate -i mask.nii.gz -o mask_dil.nii.gz --footprint 2
nvitk-filter sliding-threshold -i cd.nii.gz -o mask.nii.gz --dim 3d
nvitk-measure volume -i mask.nii.gz -o vol.txt
nvitk-transform resample -i pet.nii.gz -r ct.nii.gz -o pet_on_ct.nii.gz
```

### nvitk-gui

Install the GUI extra, then launch the Napari workbench:

```bash
pip install -e ".[gui]"
nvitk-gui
```

Open images (drag-drop or Ctrl+O) via **nvitk.io** (NIfTI, DICOM folders, TIFF, MHA, ND2, …) — not Napari’s default imageio reader. Pick tools from the catalog dock, toggle GPU for supported filters, reconstruct meshes from masks, and export a simple pipeline JSON from the *Pipeline* tab.

After `pip install -e .`, Napari also registers the **nvitk-io** reader plugin (higher priority than builtins for `.nii.gz`).

### Mesh reconstruction

```python
from nvitk.io import imread
from nvitk.meshlab import mesh_from_image

img = imread("mask.nii.gz")
mesh = mesh_from_image(img)  # binary; use multilabel=True for label maps
```

`nvitk.types.Mesh` stores vertices, faces, and metadata (affine, spacing, label id) for Napari surface layers.

### Image I/O: read, save, and display

- Import from `nvitk.io`: `imread`, `imsave`, `imshow`.
- Reader/writer chosen by extension or `force_type` (`nifti`, `dicom`, `tiff`, `mha`, `pil`, `nd2`, ...).
- `imshow` supports slice view, orthogonal view, mosaic, and animation.

```python
from nvitk.io import imread, imsave, imshow

img = imread("study/pet", force_type='dicom', backend="gpu")
imshow(img, axis=2, index="mid")
imsave("out/pet_copy.nii.gz", img)
```

### Measurements and metrics

Includes volume, intensity, SUV, voxel overlap (Dice/Jaccard), surface metrics, and radiomics. Use `Measurer` for chained workflows.

```python
from nvitk.measure import volume_cc, masked_stats, Measurer

vol = volume_cc(mask)
stats = masked_stats(pet, mask, stats=("mean", "max"))
summary = Measurer(pet, mask).volume() | Measurer(pet, mask).suv(kinds=("bw",))
```

### Segmentation

`nvitk.segmentation` includes `labels`, `hemisphere`, `total_segmentator`, and `eicab`.

```bash
nvitk-totalseg   --input ct.nii.gz   --output out/totalseg   --task total   --submit local   --device gpu
nvitk-eicab      --input tof.nii.gz  --output out/eicab       --submit local --container /path/to/eicab.sif
```

### Geometric transforms

Includes `isotropy`, affine resampling (`resample_to`, `resample_pet_to_mask`, `resample_mask_to_pet`), and Z-rotation helpers.

```python
from nvitk.transform import isotropy, resample_pet_to_mask

pet_iso = isotropy(pet)
pet_on_mask = resample_to(pet_iso, mask)
```

### PESA fat pipelines

The three `nvitk-pesa-fat*` commands call `run_batch`, `ct_pet_v5.run`, and `dixon_v5.run`.

```bash
nvitk-pesa-fat --help
nvitk-pesa-fat-ctpet --help
nvitk-pesa-fat-dixon --help
```
