# nvitk Slicer modules

Scripted loadable modules for [3D Slicer](https://www.slicer.org/), meant to be
loaded via **Additional module paths** (no Slicer rebuild / CMake extension).

## Modules

| Folder | Module title | Category |
|--------|--------------|----------|
| `MouseTOFCoW/` | Mouse TOF CoW | nvitk |
| `MouseTOFMorphometrics/` | TOF CoW Morphometrics | nvitk |

The two chain: **Mouse TOF CoW** produces `{volume}_tof_cow_trees` (labels 1/2/3), which
**TOF CoW Morphometrics** measures using `mouse_root_topology.json`.

### Mouse TOF CoW

Self-contained port of the Napari Lab Mouse TOF CoW recipe:

1. **Stage 1** — Slicer **N4ITKBiasFieldCorrection** CLI → Frangi/hysteresis blood flood → CCs
2. **Stage 2** — left-click CCs in a slice view, assign to Left ICA / Right ICA / Basilar
3. **Finalize** — multilabel blood-flood expand → `{volume}_tof_cow_trees` (labels 1/2/3)

**Does not import nvitk or ANTsPy.** Compute lives under `MouseTOFCoW/MouseTOFCoWLib/`.

#### Dependencies

Already in Slicer: `numpy`, `scipy`, built-in **N4ITKBiasFieldCorrection**.

Install once into Slicer’s Python (module button **Install optional deps**, or console):

```python
import slicer
slicer.util.pip_install("scikit-image scikit-learn")
```

Works on Windows Slicer builds (no antspyx wheels).

### TOF CoW Morphometrics

Measures a multilabel vessel labelmap: Taubin smoothing → skeleton → centerlines →
radius / tortuosity / stenosis / enlargement → **per-label volumetry**. Outputs
`case_metrics_donut_tree.xlsx` (incl. sheet `06_Volumetry`), `volumetry.csv`, and
centerline / surface VTPs, and shows the tables plus 3D models inside Slicer.

**Does not import nvitk.** Like Mouse TOF CoW, compute lives under
`MouseTOFMorphometrics/MouseTOFMorphometricsLib/` — nvitk's install requirements (antspyx,
TotalSegmentator, nnU-Net, a pinned SimpleITK) cannot go into Slicer's Python. The
difference from the CoW module: the algorithm modules are **copied verbatim**, not
simplified, so the measurements are exactly those of the upstream pipeline
(verified byte-for-byte on the mouse test case).

#### Vendored pipeline

```text
MouseTOFMorphometricsLib/
├── deps.py             # dependency check + Install dependencies button backend
├── morphometrics.py    # facade over the vendored pipeline
├── mrml_io.py          # labelmap ↔ NIfTI, result VTPs → model nodes
├── results.py          # workbook/CSV → UI tables
├── vendor_sync.py      # regenerates nvitk_vendor/ from an nvitk checkout
└── nvitk_vendor/       # generated; see VENDORED.md for provenance + file hashes
    ├── core/           # hand-written NumPy-only stand-ins for nvitk.core
    ├── measure/        # verbatim: morphometrics.py, morphometrics_config.py, morpho/
    └── morphology/     # verbatim: centerline, mst_bridge, polyline_graph
                        # hand-written: binary, components
```

`nvitk_vendor/` is treated as **generated code**. The only transformation is the root
package rename `nvitk` → `nvitk_vendor`, which keeps every file diffable against upstream.
Hand-written stand-ins replace nvitk's CuPy backend, Rich logger and `Image` type — these
are never overwritten by the sync script.

Refresh the copy after changing the morphometrics pipeline:

```bash
python slicer/MouseTOFMorphometrics/MouseTOFMorphometricsLib/vendor_sync.py
```

`--check` exits non-zero when the vendored copy has drifted (useful in CI);
`--src /path/to/src` points at a checkout elsewhere.

#### Species and axis handling

Topology JSONs may declare a `_meta` block (`species`, `length_scale`, `axes_override`).
`mouse_root_topology.json` declares `species: "mouse"`, and each vessel's
`no_upstream_start` is `"caudal"`. Anatomical directions are resolved against the image
**affine** (not a hardcoded axis), and a mouse is treated as a quadruped — so `caudal`
lands on the scanner A/P axis rather than S/I. The run prints and records the resolution,
e.g. `species=mouse axcodes=LPS ... length_scale=0.15`; check it in the status line or in
the `species` / `orientation_axcodes` / `root_rule_axis` columns of `01_Tree_Summary`.

`length_scale` (0.15 for mouse) rescales the human-calibrated minimum path length, minimum
tree-arm length and spur-pruning thresholds — without it a mouse run discards nearly every
path. The caliber detectors (`STENOSIS_*`, `ENLARGEMENT_*`, taper and siphon windows) are
**not** rescaled: treat those columns as uncalibrated for mouse data.

#### Dependencies

Already in Slicer: `numpy`, `scipy`, `vtk`, `matplotlib`.

Four packages must be added once. Use the module's **Dependencies → Install dependencies**
button (it lists exactly what is missing and disables itself when satisfied), or the console:

```python
import slicer
slicer.util.pip_install("pandas nibabel scikit-image openpyxl")
```

`Re-check` re-runs the import test after installing. Until it passes, the topology/species
combos and **Run morphometrics** stay disabled.

The run is always serial: the pipeline parallelises with spawned subprocesses, which are not
reliable inside Slicer's embedded Python.

**Input already Taubin-smoothed** is on by default, so the labelmap is measured as-is —
segmentations coming from Mouse TOF CoW are already clean, and smoothing shrinks the mask.
Uncheck it to smooth first.

## Setup

1. Open **Edit → Application Settings → Modules**
2. Under **Additional module paths**, add the **module folder** (not its parent):

   ```text
   ~/nvitk/slicer/MouseTOFCoW
   ~/nvitk/slicer/MouseTOFMorphometrics
   ```

   On Windows, e.g. `C:\path\to\nvitk\slicer\MouseTOFCoW`

3. Restart Slicer
4. Open **Mouse TOF CoW** / **TOF CoW Morphometrics** under category **nvitk**
5. If needed, click the module's install-dependencies button once

Confirm:

```python
print("MouseTOFCoW" in slicer.util.moduleNames())
print("MouseTOFMorphometrics" in slicer.util.moduleNames())
print(getattr(slicer.modules, "n4itkbiasfieldcorrection", None) is not None)
```

## Usage — Mouse TOF CoW

1. Load a 3D TOF scalar volume
2. Select it → **Run Stage 1**
3. Left-click a connected component in any slice view (highlight turns yellow)
4. **Add CC to tree** for the current tree (Left ICA → Right ICA → Basilar)
5. **Tree done** when finished with each tree; after Basilar, expand runs automatically
6. **Deselect** clears the pending highlight; **Cancel** ends Stage 2 and leaves the CC labelmap

Outputs:

- `{name}_tof_cow_cc` — Stage-1 connected components
- `{name}_tof_cow_trees` — final trees (`1` Left ICA, `2` Right ICA, `3` Basilar)

## Usage — TOF CoW Morphometrics

1. Select the multilabel labelmap (e.g. `{name}_tof_cow_trees`)
2. **Topology JSON** → `mouse_root_topology.json` for mouse; **Species** → `auto`
3. Optionally set **Output directory** (empty ⇒ a temporary folder)
4. **Run morphometrics**

Outputs, under the case directory:

- `case_metrics_donut_tree.xlsx`:
  - `00_Path_Summary` — **non-overlapping** vessel segments, one row per piece of vessel
  - `01_Tree_Summary` — per label/component, incl. the resolved `species` /
    `orientation_axcodes` / `root_rule_axis`, `centerline_total_length_mm` and
    `unique_skeleton_graph_length_mm`
  - `02_Branchpoints`, `03_LR_Asymmetry`, `04_Tree_Segments`, `05_Hemisphere`, `06_Volumetry`
  - `07_Root_To_Terminal_Paths` — the raw measured paths (see the caveat below)
  - per-vessel point sheets
- `volumetry.csv` — the `06_Volumetry` table, openpyxl-free
- `centerlines/`, `centerlines_radius/`, `surfaces/` — VTPs, loaded into the 3D view
- `radius_histograms/`, `centerlines/tortuosity_metrics.xlsx`
- `{labelmap}.nii.gz` — the exact labelmap that was measured (after gap bridging)
- `*_taubin_report.json` — raw vs smoothed volume per label (only when smoothing ran)

#### Result models are placed in RAS

The pipeline works in a scaled voxel-index frame: VTP points are `voxel_index * spacing`
with the origin at `(0, 0, 0)` and no direction cosines. Loaded as-is they would sit in a
corner of the volume, rotated away from the labelmap. The module maps them back with
`affine @ diag(1/spacing)` — read from the NIfTI it handed the pipeline — and bakes the
transform into the points, so the models are plain RAS geometry sitting on the
segmentation, with no transform node left in the scene.

#### Non-overlapping centerlines

VMTK is seeded root→terminal, so every path of a branching vessel re-traverses the shared
proximal trunk — summing those path lengths multiply-counts it (on a human eICAB case,
4155 mm of "path length" over 1800 mm of actual vessel, and 54% of the exported centerline
geometry duplicated). The pipeline therefore splits the measured paths back into unique
skeleton-edge segments, and deduplicates whatever cannot be split (loop/donut components,
whose cyclic skeleton has no tree decomposition).

`00_Path_Summary` holds that non-overlapping set: each piece of vessel appears exactly
once, so `length_mm` sums to the real tree length and matches
`01_Tree_Summary.centerline_total_length_mm` and the exported VTPs. Stenosis and
enlargement flags are **not** re-detected per segment — their reference radius is
established over the whole parent vessel and often lies proximal to the segment — the
full-path detections are re-aggregated instead.

`07_Root_To_Terminal_Paths` keeps the raw measured paths for traceability. **Do not sum its
lengths**; `01_Tree_Summary.centerline_paths_total_length_with_shared_trunks_mm` reports
that inflated figure under a name that says so.

The **Per vessel** tab joins per-label mask volumetry (voxel count, volume mm³/µL, mesh
volume, surface area, equivalent radius) to length-weighted centerline metrics; **Volumetry**
shows the full per-label table including the pipeline-input volume, so smoothing/pruning
losses stay visible; **Per segment** lists the individual non-overlapping segments.

## Note on N4

Stage 1 calls Slicer’s **N4ITKBiasFieldCorrection** CLI (`shrinkFactor=2`, default spline distance).
This avoids in-process SimpleITK N4 crashes. Results can differ slightly from ANTsPy N4 used in the Napari Lab; Frangi/hysteresis/expand parameters still match the Lab tool.
