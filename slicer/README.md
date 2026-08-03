# nvitk Slicer modules

Scripted loadable modules for [3D Slicer](https://www.slicer.org/), meant to be
loaded via **Additional module paths** (no Slicer rebuild / CMake extension).

## Modules

| Folder | Module title | Category |
|--------|--------------|----------|
| `MouseTOFCoW/` | Mouse TOF CoW | nvitk |

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

## Setup

1. Open **Edit → Application Settings → Modules**
2. Under **Additional module paths**, add the **module folder** (not its parent):

   ```text
   /home/imarcoss/nvitk/slicer/MouseTOFCoW
   ```

   On Windows, e.g. `C:\path\to\nvitk\slicer\MouseTOFCoW`

3. Restart Slicer
4. Open **Mouse TOF CoW** under category **nvitk**
5. If needed, click **Install optional deps (skimage, sklearn)** once

Confirm:

```python
print("MouseTOFCoW" in slicer.util.moduleNames())
print(getattr(slicer.modules, "n4itkbiasfieldcorrection", None) is not None)
```

## Usage

1. Load a 3D TOF scalar volume
2. Select it → **Run Stage 1**
3. Left-click a connected component in any slice view (highlight turns yellow)
4. **Add CC to tree** for the current tree (Left ICA → Right ICA → Basilar)
5. **Tree done** when finished with each tree; after Basilar, expand runs automatically
6. **Deselect** clears the pending highlight; **Cancel** ends Stage 2 and leaves the CC labelmap

Outputs:

- `{name}_tof_cow_cc` — Stage-1 connected components
- `{name}_tof_cow_trees` — final trees (`1` Left ICA, `2` Right ICA, `3` Basilar)

## Note on N4

Stage 1 calls Slicer’s **N4ITKBiasFieldCorrection** CLI (`shrinkFactor=2`, default spline distance).
This avoids in-process SimpleITK N4 crashes. Results can differ slightly from ANTsPy N4 used in the Napari Lab; Frangi/hysteresis/expand parameters still match the Lab tool.
