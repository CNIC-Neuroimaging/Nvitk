# Main GUI (napari)

`nvitk-gui` is a full [napari](https://napari.org) workbench built on nvitk: nearly the
entire {doc}`CLI tool catalog <../api/cli-catalog>` is available as a form-driven dock
alongside napari's own layer viewer, plus mesh reconstruction, a session pipeline recorder,
and its own image reader that takes priority over napari's built-ins.

```{code-block} bash
nvitk-gui
```

```{code-block} bash
pip install -e ".[gui]"   # if installing the GUI extra from a pixi/dev checkout
```

## Layout

The main window (`nvitk.gui.app.run_app`) is napari's viewer plus a right-hand dock of tabs:

| Tab | What it's for |
|---|---|
| **Tools** | The full tool catalog (below), form-driven via magicgui. |
| **Data** | Dataset/subject browser over a `DataRepo` ({doc}`../api/db`). |
| **QC** | Quality-control review panels for pipeline outputs. |
| **Statmodels** | Launches {doc}`the Stats GUI <../stats-gui/index>` as a floating window. |
| **Image properties** | Spacing, affine, orientation, and other metadata for the active layer. |
| **DICOM tags** | DICOM header inspection. |
| **Mesh** | Reconstructs a `Mesh` from the active binary/label layer via marching cubes and adds it as a napari Surface layer. |
| **Layers** | Layer management, a "record pipeline steps" toggle, and the CT display window picker (below). |
| **Export** | Layer export to disk. |
| **Pipeline** | Writes the recorded step sequence (open/mesh/export/...) as JSON. |

Keybindings: <kbd>Ctrl</kbd>+<kbd>T</kbd> transpose axes, <kbd>Ctrl</kbd>+<kbd>O</kbd> open,
<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>S</kbd> save the active layer. A bottom dock streams
the shared nvitk logger's output.

## CT display windows

CT is the one modality with a physically calibrated intensity scale — Hounsfield units are fixed
by definition (−1000 air, 0 water) — so a *fixed* display window shows the same tissue contrast
on every scan. Napari's default is per-layer auto-contrast, which for CT is actively unhelpful:
two scans of the same anatomy look different because one happened to include more bone in the
field of view.

The **Layers** tab therefore carries a window picker backed by
{mod}`nvitk.viz.ct_windows`, a registry of standard windows stored as level/width in HU:

| Window | Level / Width | For |
|---|---|---|
| Brain | 40 / 80 | Grey-white differentiation; the head-CT default |
| CT angiography | 300 / 600 | Contrast-filled lumen against wall and tissue |
| Stroke / posterior fossa | 35 / 30 | Narrow window for early infarct |
| Subdural | 70 / 200 | Extra-axial collection against adjacent bone |
| Bone | 500 / 2000 | Cortical detail and fractures |
| Soft tissue, Mediastinum, Lung, Liver, Full range | — | General review |

Choosing a preset applies it immediately to the selected Image layer. Level and width can also
be typed directly, in which case the picker switches to *Custom* — unless the values happen to
match a registered window, when it snaps back to that name. **Apply to all image layers** windows
the whole viewer at once, and **Auto** restores napari's per-layer min/max.

```{note}
Only CT is offered a window. MR intensities are arbitrary units with no fixed zero, so an HU
range is meaningless there — the picker detects this from layer modality metadata (falling back
to the intensity range, since CT goes well below zero and MR magnitude data never does), disables
itself, and says why. "Apply to all" skips non-CT layers rather than blanking them.
```

The registry is display-only and never modifies voxels. For intensity rescaling that feeds a
model, see {mod}`nvitk.normalization.intensity`.

## Tool catalog

`nvitk.gui.tools.registry` defines every tool as a `GuiToolSpec` (id, category, parameter
spec, whether it needs a reference layer or 3D data, and its run mode), merged with the
pipeline shortcuts from `nvitk.gui.pipeline.catalog`. **91 tools across 11 categories**,
each backed by the same functions documented in the {doc}`Main API Reference <../api/index>`:

| Category | Count | Examples |
|---|---|---|
| Restoration | 3 | Bilateral filter, N4 bias correction, MRI super-resolution |
| Filters | 6 | Sliding threshold, Hessian, Jerman vesselness, snakes, mask keep-inside/outside |
| Morphology | 11 | Dilate/erode/open/close, fill holes, connected components, ICA siphon correction, mask genus |
| Centerline | 3 | Detect/cut junctions, convert to polyline |
| Segmentation | 24 | Label ops, mask boolean algebra, region growing, blood flood, ANTsPyNet brain/vessel/DKT, TotalSegmentator, eICAB |
| Visualization | 8 | PET/SUV hotspots, 4D-flow vectors/streamlines, vessel cross-sections, hemodynamics, TOF morphometrics |
| Transform | 8 | Volume projection, reorient, rotate, swap axes, isotropy, resample, oblique slice |
| Registration | 6 | FLIRT rigid/apply, ANTsPy register/apply, FireANTs register/apply |
| Measure | 16 | QVTPy LOCs, LOC/mask hemodynamics, volume, morphometrics, Dice/Jaccard, SUV stats |
| Lab | 1 | Mouse TOF Circle-of-Willis interactive session |
| Pipelines | 5 | PESA-Fat CT-PET, PESA-Fat DIXON, QVTPy, BBTPy, GPETPy — each opens a CLI form for the corresponding pipeline command |

The dock (`nvitk.gui.tools.dock`) wires the category/operation form to a label picker (shown
for label-like layers), a TotalSegmentator ROI checklist (shown only for that tool), a
pipeline-CLI form (for the Pipelines category), the GPU toggle, and a "Run SGE" button
(enabled per-tool via `is_sge_capable`).

## GPU toggle

A single "GPU computing: ON/OFF" button (`nvitk.gui.tools.gpu_toggle`) calls the same
{doc}`process-wide backend switch <../api/core-backend>` as `nvitk-gui --backend`, falling
back to CPU with a log warning if no CUDA/CuPy is available. It's a global switch, not
per-filter — only tools with an actual GPU code path benefit.

## The `nvitk-io` reader plugin

nvitk registers itself as a [napari plugin](https://napari.org/stable/plugins/index.html)
(`napari.yaml`), reading `.nii`/`.nii.gz`/`.mha`/`.mhd`/`.tif`/`.tiff`/`.nd2`/`.png`/`.jpg`/
`.jpeg`/`.bmp`/`.gif`/`.dcm` and whole DICOM-series directories through nvitk's own `imread`
rather than napari's default `imageio`-based reader. Programmatic/plugin-manager opens use
ordinary npe2 filename-pattern matching; for interactive drag-and-drop and File → Open,
`nvitk.gui.app.install_nvitk_io` additionally patches napari's `QtViewer._qt_open` so nvitk's
reader gets first refusal, falling back to napari's built-in reader only if nvitk can't open
the file.

## Command reference

```{eval-rst}
.. click:: nvitk.gui.main:main
   :prog: nvitk-gui
   :nested: full
```

```{seealso}
Full generated reference: [`nvitk.gui`](../autoapi/nvitk/gui/index).
```
