# Main API Reference

The core `nvitk` library, organized by topic. Each page below is a curated overview of one
area — for the exhaustive, auto-generated reference of every module, class, and function,
see the [full `nvitk` package index](../autoapi/nvitk/index) (built directly from source via
static analysis, always in sync with the code).

```{toctree}
:maxdepth: 1
:hidden:

core-backend
io
types-transform
morphology-filters-restoration
registration
segmentation
measure
stats
viz
db
cluster-registry
cli-catalog
util
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`cpu` Core & Backend
:link: core-backend
:link-type: doc
The NumPy/CuPy dual backend — `setup()`, `using()`, and everything that makes the rest of
the library CPU/GPU-agnostic.
:::

:::{grid-item-card} {octicon}`file-media` I/O
:link: io
:link-type: doc
`imread`/`imsave`/`imshow`, format conversors (DICOM, STL, phase-contrast, Nikon), and
per-format readers/writers.
:::

:::{grid-item-card} {octicon}`package` Types & Transform
:link: types-transform
:link-type: doc
The `Image` and `Mesh` containers, mesh reconstruction, and geometric transforms (resample,
reorient, rotate, isotropy).
:::

:::{grid-item-card} {octicon}`pulse` Morphology, Filters & Restoration
:link: morphology-filters-restoration
:link-type: doc
Binary/label morphology, centerline extraction, vesselness/threshold filters, denoising and
bias-field correction.
:::

:::{grid-item-card} {octicon}`git-compare` Registration
:link: registration
:link-type: doc
FSL FLIRT, ANTsPy, and FireANTs (GPU) registration.
:::

:::{grid-item-card} {octicon}`list-unordered` Segmentation
:link: segmentation
:link-type: doc
Label-map operations, brain extraction, vessel/DKT segmentation, and the TotalSegmentator /
eICAB engine wrappers.
:::

:::{grid-item-card} {octicon}`graph` Measure
:link: measure
:link-type: doc
Volume, intensity, SUV, overlap, surface, and radiomics metrics, plus the vascular
morphometrics subsystem.
:::

:::{grid-item-card} {octicon}`beaker` Statistics
:link: stats
:link-type: doc
Mixed models, mediation, SEM, and the Python/R statistical engines behind the Stats GUI.
:::

:::{grid-item-card} {octicon}`eye` Visualization
:link: viz
:link-type: doc
Brain/vascular atlas rendering, flow streamlines, and PET hotspot visualization helpers.
:::

:::{grid-item-card} {octicon}`database` Database & XNAT
:link: db
:link-type: doc
`DataRepo`, the local dataset catalog, and XNAT sync/upload.
:::

:::{grid-item-card} {octicon}`server` Cluster & Registry
:link: cluster-registry
:link-type: doc
SGE submission helpers and the Singularity container/model registry.
:::

:::{grid-item-card} {octicon}`terminal` CLI Catalog
:link: cli-catalog
:link-type: doc
How `pyhelp` and `nvitk-gui`'s tool dock discover and describe every command.
:::

:::{grid-item-card} {octicon}`tools` Utilities
:link: util
:link-type: doc
Logging, ANSI colors, and other small shared helpers.
:::
::::

```{note}
`nvitk.graphs` and `nvitk.normalization` are reserved namespaces for planned functionality —
they currently expose no public API.
```
