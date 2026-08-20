```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
api/index
gui/index
stats-gui/index
pipelines/index
```

# nvitk

**Neuro-Vascular Imaging ToolKit** — a research toolkit for neurological and vascular
biomedical image processing (CT, PET, MRI/MRA): I/O, filtering, restoration, segmentation,
registration, imaging metrics, mesh reconstruction, statistics, and full research pipelines,
with an optional Napari-based GUI and a NumPy/CuPy dual backend for CPU/GPU execution.

Developed at [CNIC](https://www.cnic.es/) for intracranial and vascular research, including
4D-flow MRI hemodynamics, TOF morphometrics, and whole-body PET/CT quantification.

[![Conda Version](https://anaconda.org/cnic/nvitk/badges/version.svg)](https://anaconda.org/cnic/nvitk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ignacio-ms/Nvitk/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](installation)

```{code-block} bash
conda install -c cnic -c conda-forge -c bioconda -c mrtrix3 -c ejolly nvitk
```

::::{grid} 1 2 2 3
:gutter: 3
:class-container: nvitk-home-grid

:::{grid-item-card} {octicon}`rocket` Installation
:link: installation
:link-type: doc
Install via conda (recommended, ready-to-use) or set up a pixi development environment
from a clone.
:::

:::{grid-item-card} {octicon}`play` Quickstart
:link: quickstart
:link-type: doc
The dual NumPy/CuPy backend, the `Image` container, and your first read → process → save
pipeline.
:::

:::{grid-item-card} {octicon}`book` Main API Reference
:link: api/index
:link-type: doc
The core library: I/O, transforms, morphology, filters, restoration, registration,
segmentation, measurements, statistics, and the CPU/GPU backend.
:::

:::{grid-item-card} {octicon}`device-desktop` Napari GUI
:link: gui/index
:link-type: doc
`nvitk-gui` — a Napari workbench with a 91-tool catalog, GPU toggle, mesh reconstruction,
and pipeline export.
:::

:::{grid-item-card} {octicon}`graph` Stats GUI
:link: stats-gui/index
:link-type: doc
`nvitk-statsmodels` — mixed models, MMRM, SEM, mediation analysis, and cohort-level
brain/vascular plotting.
:::

:::{grid-item-card} {octicon}`workflow` Pipelines
:link: pipelines/index
:link-type: doc
End-to-end batch pipelines: PESA-Fat (CT/PET, DIXON) and QVTPy (4D-flow hemodynamics),
local or SGE-cluster execution.
:::
::::

## Highlights

- **One codebase, two backends** — write code once against `nvitk.core`'s NumPy/CuPy proxy
  (`setup()`, `using()`, `get_current_backend()`) and switch between CPU and GPU execution
  per-call, per-scope, or process-wide, without branching your code.
- **A real GUI, not a demo** — the Napari workbench (`nvitk-gui`) exposes nearly the entire
  tool catalog through a form-driven dock, with its own prioritized NIfTI/DICOM/TIFF/ND2
  reader plugin.
- **A dedicated stats workbench** — `nvitk-statsmodels` runs standalone (no Napari/GPU
  context needed) for mixed-effects models, MMRM, robust regression, SEM/network models,
  and cluster-bootstrap mediation analysis over cohort measurement tables.
- **Cluster-aware pipelines** — PESA-Fat and QVTPy run identically `--submit local` or
  `--submit sge`, with per-subject array-job dispatch and dependency chaining.

## Where to next

- New to nvitk? Start with {doc}`installation` then {doc}`quickstart`.
- Looking for a specific function or class? Jump straight to the
  {doc}`Main API Reference <api/index>`.
- Want to run a full cohort pipeline? See {doc}`pipelines/index`.
