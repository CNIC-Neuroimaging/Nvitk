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

## 3D+t volumes

A 4D image — a dynamic CT, a perfusion or cine series, a cardiac phase stack from
`dcm2nii --stack-phases` — opens as a **time-first** layer: napari sees a `T x X x Y x Z`
view of the `X x Y x Z x T` file (no copy) with a block-diagonal 5x5 affine, the time step on
the first axis and the file's full spatial affine on the other three.

Why it matters: napari aligns layers by their *trailing* dimensions. Time last put a 3D mask of
the same patient on world axes (Y, Z, T), so overlays never lined up and inflated the time
slider; and 4D layers had to drop their (oblique) affine to keep the slider sane. Time-first,
the spatial axes are the trailing three — exactly where every 3D layer sits — so a 3D
segmentation, a 3D tool output or a CT of another phase overlays the 4D volume voxel for voxel,
and nothing couples time and space.

- It opens like a 3D volume (axial, 2D) on its first frame; the time slider's play button is a
  cine. The slider runs in seconds when the frame interval is known.
- Tools get the array as shown (`TXYZ`, axes labelled accordingly); 3D results of a 3D+t layer
  land on its spatial grid. Export writes the file's own `XYZT` order and 4x4 affine back.
- 4D-flow phase arithmetic reads the file order through
  `nvitk.gui.core.spatial.layer_source_order`, so the flow tools work unchanged.

## Orthogonal views

The **Orthogonal views** dock shows axial / coronal / sagittal through one crosshair, drawing
every visible layer on the bound layer's grid (off-grid layers are resampled once and cached).

- **3D+t**: a 4D layer binds like a 3D one; a *Time* slider appears above the views and stays in
  step with napari's time slider both ways, so playing the cine drives the views and the 3D
  planes too. The label shows the frame time, the cardiac phase, or — for a monoenergetic stack —
  the energy in keV.
- **3D slices**: *Show the three slices in 3D* puts the cuts on the canvas; *…and the slice
  image* draws them with the **same window, colormap and gamma** as the layer (labels keep
  their colours) and follows later edits — they used to appear in napari's default grey range.
- **Per-layer control**: a table lists every drawn layer with two checkboxes — *3D slice*
  (which layers get a slice image on the planes; default the bound one) and *Cut* (which layers
  the *See inside* cut opens; default all). Keep a mask whole while the CT around it is cut away,
  or the other way round.

## Performance (CPU threads)

The views' host work — slicing and compositing every layer, resampling off-grid layers, oblique
and CPR reformats, FFTs — runs on nvitk's shared worker pool ({mod}`nvitk.core.parallel`).
The **CPU: N / M threads** button in the Tools dock sets the budget: default **75 % of the cores**,
presets 25/50/75 %/all, stored in `gui.json` (`"performance"`) and overridden per session by
`$NVITK_WORKERS`. The same dialog toggles napari's experimental *asynchronous slicing*.

Measured on a 512x512x416 float64 CT with a mask and an off-grid layer: a three-view redraw
dropped from 8.2 s to 18 ms. Most of that was not threading but a slicing fix — `np.take`
copied the whole Fortran-ordered NIfTI array for every slice; the views now index views of it.
Resampling the off-grid layer: 13.8 s on one thread, 0.67 s on 24.

## Time & frequency tools

| Tool | What it does |
|---|---|
| **3D+t: extract time frame** | One frame (or the one on screen) as a 3D layer. |
| **3D+t: temporal projection** | max / mean / min / std / sum / median, plus **ttp** (time to peak, s) and **auc** maps. |
| **3D+t: time–intensity curve** | The curve at the cursor voxel or over a mask label, in a window that collects curves and saves CSV — bolus tracking, enhancement. On a monoenergetic stack (`dcm2nii --stack-energies`) it is the spectral curve, HU vs keV. |
| **3D+t: stack layers** | Same-grid 3D layers → one 3D+t layer, ordered by energy for monoenergetic results or by cardiac phase when known. |
| **K-space (FFT)** | Spatial FFT (3D, or 2D slice by slice; per time point for 3D+t) shown as log-magnitude (or phase/real/imag) on the same grid. The complex data stays attached to the layer. |
| **Inverse FFT** | Back to image space from the active k-space layer, optionally keeping or removing the region painted in a mask over k-space. |
| **K-space filter** | Radial low/high/band-pass/band-stop with a Hann, Gaussian, Butterworth or ideal edge. |

All run on the GPU when GPU computing is on; the FFTs use the worker pool on the CPU.

## Quick tools: brightness / contrast

The search-bar quick operations that set a window — *Brightness / contrast*, *Rescale
intensity*, *Find contours* — take **absolute intensities** (HU for CT) on sliders spanning the
layer's data range, each with a box to type an exact value. They start on the robust 1–99 %
window expressed in intensity units, and the live preview is reverted if the dialog is cancelled.

## Tool catalog

`nvitk.gui.tools.registry` defines every tool as a `GuiToolSpec` (id, category, parameter
spec, whether it needs a reference layer or 3D data, and its run mode), merged with the
pipeline shortcuts from `nvitk.gui.pipeline.catalog`. **117 tools across 12 categories**,
each backed by the same functions documented in the {doc}`Main API Reference <../api/index>`:

| Category | Count | Examples |
|---|---|---|
| Restoration | 3 | Bilateral filter, N4 bias correction, MRI super-resolution |
| Filters | 19 | Sliding threshold, Hessian, Jerman vesselness, snakes, mask keep-inside/outside |
| Morphology | 11 | Dilate/erode/open/close, fill holes, connected components, ICA siphon correction, mask genus |
| Centerline | 3 | Detect/cut junctions, convert to polyline |
| Segmentation | 25 | Label ops, mask boolean algebra, region growing, blood flood, ANTsPyNet brain/vessel/DKT, TotalSegmentator, eICAB |
| Registration | 6 | FLIRT rigid/apply, ANTsPy register/apply, FireANTs register/apply |
| Visualization | 12 | PET/SUV hotspots, 4D-flow vectors/streamlines, vessel cross-sections, hemodynamics, TOF morphometrics |
| Transform | 8 | Volume projection, reorient, rotate, swap axes, isotropy, resample, oblique slice |
| Time & frequency | 7 | 3D+t frame / projection / time curve / stacking, k-space FFT, inverse FFT, k-space filter |
| Measure | 19 | QVTPy LOCs, LOC/mask hemodynamics, volume, morphometrics, Dice/Jaccard, SUV stats |
| Lab | 1 | Mouse TOF Circle-of-Willis interactive session |
| Pipelines | 3 | Pipeline CLI shortcuts |

The dock (`nvitk.gui.tools.dock`) wires the category/operation form to a label picker (shown
for label-like layers), a TotalSegmentator ROI checklist (shown only for that tool), a
pipeline-CLI form (for the Pipelines category), the GPU toggle, the CPU-threads button, and a "Run SGE" button
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
