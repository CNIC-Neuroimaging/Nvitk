# Main GUI (napari)

`nvitk-gui` is a full [napari](https://napari.org) workbench built on nvitk: nearly the
entire {doc}`CLI tool catalog <../api/cli-catalog>` is available as a form-driven dock
alongside napari's own layer viewer, plus mesh and point-cloud tools, a session pipeline recorder,
and its own image reader that takes priority over napari's built-ins.

```{code-block} bash
nvitk-gui
```

```{code-block} bash
pip install -e ".[gui]"   # if installing the GUI extra from a pixi/dev checkout
```

## Layout

The main window (`nvitk.gui.app.run_app`) is napari's viewer plus a right-hand group of
panels, each its own dock, tabbed together:

| Panel | What it's for |
|---|---|
| **Imaging** | The full tool catalog (below), form-driven via magicgui, with the tool search bar. The theme toggle sits on its title bar. |
| **Labels** | Label selection for any label layer, whatever tool is selected (below). |
| **Meshlab** | Meshes and point clouds, MeshLab/ParaView style (below): surfaces from labels, click-to-edit, clean/repair, smoothing, remeshing, alignment, measurements, vessel centerlines, image ↔ mesh ↔ points conversion, colouring, mesh time series. |
| **Image properties** | Spacing, affine, orientation, and other metadata for the active layer. |
| **DICOM tags** | DICOM header inspection. |
| **Data** | Dataset/subject browser over a `DataRepo` ({doc}`../api/db`). |
| **QC** | Quality-control review panels for pipeline outputs. |
| **Statmodels** | Launches {doc}`the Stats GUI <../stats-gui/index>` as a floating window. |
| **Export** | Layer export to disk. |
| **Layers** | Layer management, a "record pipeline steps" toggle, and the CT display window picker (below). |
| **Pipeline** | Writes the recorded step sequence (open/mesh/export/...) as JSON. |

Keybindings: <kbd>Ctrl</kbd>+<kbd>T</kbd> transpose axes, <kbd>Ctrl</kbd>+<kbd>O</kbd> open,
<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>S</kbd> save the active layer,
<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>L</kbd> show the Labels panel with the cursor in its
filter. A bottom dock streams the shared nvitk logger's output.

### Panels and the workspace window

Every panel, nvitk's and napari's own (layer list, layer controls, console, plugin
docks), is a dock (`nvitk.gui.core.workspace`):

- **Pop out / dock back**: the ⧉ button on an nvitk panel's title bar, or a double-click
  on it; napari's docks keep their own float button. ⛶ fills the screen with a panel.
- **Panels menu** (in the menu bar of both windows): show or hide any panel, pop it out,
  or move it between the main window and the **workspace**. The workspace is a second
  window with no fixed content, where panels are split and tabbed freely by dragging
  their title bars. Qt cannot drag a dock from one window to another, so moving between
  the two windows goes through this menu.
- Closing the workspace gives its panels back to the main window.

The layout is remembered in `gui.json`: the main window's dock state, plus the
workspace's panels, dock state, geometry and whether it was open (`workspace_layout`).
A layout saved before panels became separate docks (`dock_layout_version` < 2) keeps
napari's docks and the orthogonal views where they were and regroups the nvitk panels
as tabs once.

### Layer folders

The 📁 button beside napari's new-layer buttons groups the selected layers into a
**folder**; folders nest (*New subfolder from selected layers* in a folder's menu). A
folder's header sits above its first layer, its layers are indented beneath it:

- the ▾ / ▸ arrow folds and unfolds it, the box shows or hides every layer in it;
- a click on the header selects all its layers (so napari's delete, duplicate… act on
  the folder), a double click renames it, the right button opens its menu (rename,
  move the selection in, ungroup, delete with its layers);
- dragging a layer between two layers of a folder puts it in that folder, dragging it
  out takes it out, and dragging a folded folder moves the whole folder.

Membership is the layer's `metadata["nvitk_folder"]` path (`"CT/Masks"`); folders exist
through their layers. *Show folders in the layer list* in the 📁 menu switches back to the
flat list without forgetting them (`nvitk.gui.core.layer_folders`).

### Label selection

The **Labels** panel lists the labels of a label layer: Labels layers and Image masks
with few integer values. It follows the active layer, or the one picked in its *Layer*
list. Each label has a checkbox to show or hide it and a colour dot to recolour it.
The *Mapping* names the ids (eICAB, TotalSegmentator, …), and the filter box narrows
long vocabularies by name or id. **All** and **None** act on the labels the filter
shows.

Each label layer's row in napari's layer list also carries a **▾** button. It unfolds the
row to list that layer's labels underneath it, and **▴** folds it back. Each line shows a
show/hide box, the label's colour as drawn on the canvas, and its name. Click a line to
show or hide that label; <kbd>Alt</kbd>+click shows only that label (and again brings the
others back). Click the colour dot to recolour a label. **All** and **None** act on the
whole layer, and **Panel** opens the full Labels panel on it. A list longer than twelve
labels scrolls with the mouse wheel.

Names come from the layer's own vocabulary: the one picked for it in a picker, otherwise
the one guessed from its name, path and contents. Each layer keeps its own, so moving
between two segmentations never names one with the other's labels. Every picker (this
panel and the layer list) edits the same per-layer filter, so they always agree.

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

### Sagittal, coronal and single-slice series

napari slices a layer assuming its data axis *i* runs along world axis *i*. A file stored in
another order (a sagittal MR volume, a coronal cine, any non-axial single slice) breaks that.
In the axial view, a coronal plane is seen edge-on, and napari raises `Singular matrix` on
every repaint. Such files open **world-ordered**: the same voxels, read in world axis order
(a view, no copy), with every flip and obliquity kept in the affine.

A 2D image with a real position (localizer, surview, a single DICOM slice) opens as a
**one-voxel-thick 3D layer at that position**. It overlays the volume where the planes
cross, instead of sitting on the viewer's last two axes in coordinates of its own. Export
writes the file's own axis order (2D stays 2D) and affine back; the image properties show
the axes as displayed. Colour captures (dose sheets, tracker graphs) and plain 2D images
without a position keep the ordinary 2D path.

## Orthogonal views

The **Orthogonal views** dock opens with the window (tabbed with napari's layer controls; a
position you give it is remembered) and shows axial / coronal / sagittal through one crosshair,
drawing every visible layer on the bound layer's grid (off-grid layers are resampled once and
cached). It follows the active layer only onto real volumes. Selecting a localizer, a
single-slice series or a colour capture keeps the volume already shown rather than rebinding
to a one-voxel grid.

- **Display follows the layers**: colormap, window, gamma, opacity, blending — and
  **interpolation**. Each layer is magnified with its napari `interpolation2d` (napari's default,
  `nearest`, stays blocky; `linear`/`cubic` smooth), labels always nearest as on the canvas; an
  oblique plane is resliced with the matching spline order (0 / 1 / 3), and an off-grid layer set to
  `nearest` is resampled nearest. The 3D slice planes copy the interpolation too.
- **Zoom**: Ctrl+wheel, 10 % per wheel notch in proportion to the wheel's travel (touchpads and
  high-resolution wheels no longer race through the range); double-click resets. The plain wheel
  steps one slice per notch the same way.

- **3D+t**: a 4D layer binds like a 3D one; a *Time* slider appears above the views and stays in
  step with napari's time slider both ways, so playing the cine drives the views and the 3D
  planes too. **▶ Play / ⏸ Pause** beside it runs the cine from the panel itself (looping, rate in
  fps), moving napari's slider with it. The label shows the frame time, the cardiac phase, or — for
  a monoenergetic stack — the energy in keV.
- **3D slices**: *Show the three slices in 3D* puts the cuts on the canvas; *…and the slice
  image* draws them with the **same window, colormap and gamma** as the layer (labels keep
  their colours) and follows later edits — they used to appear in napari's default grey range.
- **Per-layer control**: a table lists every drawn layer with two checkboxes — *3D slice*
  (which layers get a slice image on the planes; default the bound one) and *Cut* (which layers
  the *See inside* cut opens; default all). Keep a mask whole while the CT around it is cut away,
  or the other way round.

The 3D card of the orthogonal views reports how many layers are drawn and how many were
resampled onto the active layer's grid (the names are in its tooltip). *Resample layers on
other grids* can be turned off to draw only same-grid layers. *Orientation figure on the 3D
canvas* shows a person with R/L, A/P and H/F arrows — the marker of the QC reports' 3D
renders — in the corner of the 3D view, turning with the camera
(`nvitk.gui.viz.orientation_overlay`). Both choices are remembered in `gui.json`.

## Performance (CPU threads)

The views' host work — slicing and compositing every layer, resampling off-grid layers, oblique
and CPR reformats, FFTs — runs on nvitk's shared worker pool ({mod}`nvitk.core.parallel`).
The **CPU: N / M threads** button in the Imaging dock sets the budget: default **75 % of the cores**,
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
| Registration | 6 | FLIRT register (6/7/9/12 DOF, cost, search range) / apply; ANTsPy register (every `type_of_transform`, stage metrics, iteration schedules, masks) / apply; FireANTs register (moments → rigid → affine → greedy / SyN chains on the GPU) / apply. Each can also warp further layers with the new transform. |
| Visualization | 12 | PET/SUV hotspots, 4D-flow vectors/streamlines, vessel cross-sections, hemodynamics, TOF morphometrics |
| Transform | 8 | Volume projection, reorient, rotate, swap axes, isotropy, resample, oblique slice |
| Interpolation | 4 | Up/down-sample selected axes (factor, spacing or size; shape-based for masks), block downsampling, interpolate masks between annotated slices, fill missing slices |
| Time & frequency | 8 | 3D+t frame / projection / time curve / stacking / frame interpolation, k-space FFT, inverse FFT, k-space filter |
| Measure | 19 | QVTPy LOCs, LOC/mask hemodynamics, volume, morphometrics, Dice/Jaccard, SUV stats |
| Lab | 1 | Mouse TOF Circle-of-Willis interactive session |
| Pipelines | 3 | Pipeline CLI shortcuts |

Tools that act on labels run on the labels currently **shown** on a label layer — the
selection made in the **Labels** panel or from the layer's ▾ in the layer list — or on the
ids typed in the form's *Label id(s)* field. The dock (`nvitk.gui.tools.dock`) adds a
TotalSegmentator ROI checklist (shown only for that tool), a pipeline-CLI form (for the
Pipelines category; it takes the dock's spare height), the GPU toggle, the CPU-threads
button, and a "Run SGE" button (enabled per-tool via `is_sge_capable`).

Every parameter a tool declares in the registry gets a widget, whether or not the form
declares one by hand (`nvitk.gui.tools.panel._add_registry_widgets`), laid out in the
order the tool lists them.

## Meshlab panel

`nvitk.gui.mesh` puts {doc}`nvitk.meshlab <../api/types-transform>` in a dock, the tab
after **Labels**. The active layer decides what can run: a Surface (mesh), a Points layer,
an image / labels layer, or a mesh time series.

| Category | Operations |
|---|---|
| Create | Surfaces from labels / mask, isosurface of an image, convex hull |
| Edit (click on the surface) | Erase around a click (or keep only inside), keep / delete the piece under a click, push / pull the surface, smooth around a click |
| Select (on the surface) | Brush, screen box, piece under a click; grow / shrink, invert, all, clear; delete, keep or copy the selection |
| Clean & repair | Merge duplicates and drop degenerate faces, keep largest / drop small / split pieces, fill holes, orient faces |
| Smooth | Taubin (volume-preserving) or Laplacian |
| Remesh | Decimate, subdivide (Loop / butterfly), isotropic remesh (even triangles), remesh through voxels (watertight), clip at a click |
| Transform & align | Translate / rotate / scale, ICP onto another surface or cloud, principal axes |
| Measure | Area, volume, sphericity, topology (holes, genus), curvature maps, cross-section at a click |
| Vessels & tubes | Centerlines and diameter profile (per-branch length, tortuosity, curvature, min / max diameter, stenosis; bifurcation angles — click a row of the table to see that branch or junction), vessel cross-section at a click, local thickness map |
| Compare | Distance maps and ASSD / Hausdorff |
| Convert (image ↔ mesh ↔ points) | Mesh → mask or signed-distance image on any grid, mesh → points (vertices or uniform samples), mask → points, points → mask (dots or filled), points → mesh, sample an image onto a surface or cloud |
| Point cloud | Surface reconstruction (Poisson, MeshLab's screened Poisson and ball pivoting, IMLS, implicit, alpha shape, hull), voxel / random downsampling, outlier removal, normals |
| Time (3D+t) | Area and volume over time (plot, ejection fraction), track a surface through time, displacement / speed maps, temporal smoothing, frame interpolation, extract a frame |

**Surfaces from labels.** The labels of the active layer are listed with their names and
colours; tick the ones to mesh (filter, *All*, *None*). The output can be one layer per
label (gathered in a layer folder, each in its label's colour), one layer coloured by
label, one merged surface, or groups. For groups, tick a group's labels, press *Add ticked
labels as a group*, name it, and go on with the next; the groups are also plain text
(`Left: 1,2 ; Right: 3-5`). A 3D+t mask gives time series.

**Picking a point.** Operations that need a point (cross-sections, clip, the edits) do not
read the cursor. *Run* arms them (the button turns into *Cancel picking*), and the next
click in the viewer runs them: in 3D the click is cast into the scene and lands on the
front-most face of the active surface (or the nearest point of a Points layer); dragging
still rotates the camera. A red *picked point* marker shows where it landed. *Undo edit*
puts back what the last edit changed — a crop, an erase, a move of several layers — up to 20
steps back.

**Selecting.** The *Select* operations keep a per-vertex (or per-point) selection, drawn red
over grey. *Brush*: Run, then drag on the surface to paint (mode *add* or *remove*, a radius
in mm); a drag that starts off the surface still turns the camera; *Done selecting* ends it.
*Box*: drag a rectangle over the view — only what faces you, or straight through — adding,
removing or replacing; the camera waits until Done (the wheel still zooms). *Piece under a
click* takes a whole connected piece. The selection can be grown or shrunk by rings,
inverted or cleared (the layer's previous colouring comes back), and the selected part
deleted, kept alone or copied into a new layer. MeshLab's selection-based filters (delete
selected, dilate / erode, geodesic distance from the selection…) start from it and say so
when nothing is selected.

**Display.** It acts on every selected surface, points and series layer at once (a layer
without the chosen field keeps its colouring). *Colour by* a solid colour (colour picker) or any per-vertex field an
operation attached (curvature, distance, diameter, thickness, sampled image values,
labels, displacement) or the x / y / z coordinates, with a colormap and an automatic
(2nd–98th percentile) or manual range. Opacity, shading, wireframe and face normals for
surfaces; size, symbol and 3D shading for points. Fields of a time series follow its frames.

**Move by hand.** Translate, rotate and scale with a live preview — every selected layer
together, about their common centre; *Apply* bakes the move into the vertices (one undo
step), *Reset* drops it. Works on surfaces, points and whole time series. Operations
themselves run on one layer: with several selected, select the one to work on alone.

**Results tables.** Where a table row stands for something in the viewer, clicking it shows
it: a centerline branch or bifurcation is highlighted on the surface (amber over grey) and
on the centerline (red), and the camera centres on it; a frame of *Measure over time* jumps
the series to that frame. *Clear highlight*, another result or closing the window puts the
colouring back.

**MeshLab filters.** With [PyMeshLab](https://pymeshlab.readthedocs.io) installed (a
dependency of nvitk), the categories below the separator hold MeshLab's own filters — about
170 of them, one operation each: *Create*, *Cleaning & repair* (non-manifold repair, close
holes, small pieces…), *Remeshing & simplification* (isotropic remeshing, quadric
decimation, subdivision, uniform resampling…), *Smoothing & deformation* (HC Laplacian,
two-step, surface-preserving…), *Reconstruction & point clouds* (screened Poisson, ball
pivoting, alpha wrap, VCG, MLS marching cubes…), *Sampling* (Poisson disk, Monte Carlo,
volumetric…), *Curvature, normals & quality*, *Measures & distances* (Hausdorff, geometric
and topological measures, geodesic distance from a click…), *Booleans & pieces*,
*Selection* and *Transform & align*. Their forms are built from MeshLab's parameters, with
the defaults MeshLab computes for the active mesh; lengths take `1%` (of the bounding-box
diagonal) or a number in mm; a filter on two meshes asks for the other layer. What a filter
produces comes back as new layers (several in a folder), the changed surface (*Replace the
active layer* writes it in place, undoable), a field to colour by (MeshLab's per-vertex
*quality*: curvature, distances…), a *selected* field (red) that the next selection-based
filter (delete, dilate, geodesic from the selection…) starts from, and a table of measures.
The field shown in *Display* is what MeshLab sees as quality for filters that read it
(select by quality range, scalar smoothing…). Filters for photogrammetry (cameras,
rasters, textures, vertex colours) and those needing an OpenGL context are not listed. The
filter list is read from PyMeshLab once per version and cached in `~/.cache/nvitk`.

**Search.** Every Meshlab operation, native or MeshLab's own, is in the Imaging tab's search bar and
the <kbd>Alt</kbd>+<kbd>F</kbd> palette (“isotropic”, “ball pivot”, “hausdorff”,
“geodesic”…): choosing one brings the Meshlab tab forward with it selected.

Meshes it adds are in world millimetres. A time series is one Surface layer that follows
napari's time slider when a 3D+t image is open, or the panel's own frame slider and play
button otherwise. *Open…* reads STL, OBJ, OFF, PLY, VTK/VTP, GIfTI, XYZ and PCD (a ParaView
`.pvd` as a series); *Save…* writes them (a series as one file per frame plus a `.pvd`).
The same files open with <kbd>Ctrl</kbd>+<kbd>O</kbd> or by dropping them on the viewer.

## GPU toggle

A single "GPU computing: ON/OFF" button (`nvitk.gui.tools.gpu_toggle`) calls the same
{doc}`process-wide backend switch <../api/core-backend>` as `nvitk-gui --backend`, falling
back to CPU with a log warning if no CUDA/CuPy is available. It's a global switch, not
per-filter — only tools with an actual GPU code path benefit.

## The `nvitk-io` reader plugin

nvitk registers itself as a [napari plugin](https://napari.org/stable/plugins/index.html)
(`napari.yaml`), reading `.nii`/`.nii.gz`/`.mha`/`.mhd`/`.tif`/`.tiff`/`.nd2`/`.png`/`.jpg`/
`.jpeg`/`.bmp`/`.gif`/`.dcm` and whole DICOM-series directories through nvitk's own `imread`
(and meshes / point clouds — `.stl`/`.obj`/`.off`/`.ply`/`.vtk`/`.vtp`/`.gii`/`.xyz`/`.pcd` —
through `nvitk.meshlab.io`)
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
