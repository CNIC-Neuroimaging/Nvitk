# VMTK

`nvitk.vmtk` wraps the [Vascular Modeling Toolkit](http://www.vmtk.org) (VMTK 1.5.2 on VTK 9.6.2).
VMTK is imported lazily: `nvitk` imports without it. Functions that need it raise
`VmtkUnavailableError` rather than falling back silently. Surfaces and centerlines are
`vtkPolyData` (PyVista objects work too) in millimetres. Volumes are NumPy arrays plus voxel
spacing.

```{code-block} python
from nvitk.vmtk import vmtk_available, vmtk_version

vmtk_available(), vmtk_version()      # (True, "1.5.2")
```

| Module | Purpose |
|---|---|
| `centerlines` | `centerlines_between_points` (cow_morpho's `vmtkCenterlines` + `vmtkCenterlineGeometry` call), `centerline_network` (one VMTK run from a source to every end point, split into branches with `vmtkBranchExtractor`; `joined_branches()` meets them at the bifurcation centres), `full_centerline_network` (every branch, loop and separate piece of a vessel mask — below), `mask_endpoints`, geometry / attributes / smoothing / resampling, `centerline_sections`, `distance_to_centerlines`, `branch_clip`, `bifurcation_geometry`. |
| `surface` | Taubin / Laplace smoothing, isotropic remeshing, decimation, subdivision, capping (with cap ids), marching-cubes kite removal, connectivity, curvature, normals, flow extensions, mass properties, marching cubes, surface → binary image. |
| `image` | `vessel_enhancement`: multiscale Frangi, Sato, or vesselness-enhancing diffusion (VED / VED-M). |
| `flow` | 4D-flow particle tracing with `vtkvmtkStaticTemporalStreamTracer`: steady **streamlines** at one cardiac phase, or periodic **pathlines** through the cycle, from planar or volume seeds inside a mask. Physical units (mm, s); speeds in mm/s. |

```{code-block} python
from nvitk.morphology.centering import mask_surface
from nvitk.vmtk.centerlines import centerline_network, mask_endpoints

ends, radii = mask_endpoints(mask, spacing)
net = centerline_network(mask_surface(mask, spacing), ends, endpoint_radii=radii)
for points, radius in net.joined_branches():
    ...
```

```{code-block} python
from nvitk.vmtk.flow import FlowTraceParams, trace_flow

traces = trace_flow(velocity_xyzt3, mask, FlowTraceParams(trace_mode="pathlines", dt_seconds=0.04,
                                                          max_length=1.0), spacing=spacing)
```

## Full networks

`centerline_network` traces a shortest-path tree from one source. It skips one segment of every
loop and reaches nothing outside the source's connected piece. `full_centerline_network(surface,
mask, spacing)` covers all of it. Per connected piece of the surface, it makes:
1. a run from the widest end point;
2. further runs from the end points nearest whatever skeleton branch is still uncovered, keeping
   only the new stretches;
3. a run between the two ends of any branch still missed;
4. where VMTK cannot trace a branch, the skeleton branch re-centred on the lumen.

A piece that thinning erases entirely (scikit-image does this to some straight, flat-ended
tubes) is traced between the ends of its main axis. Each branch records its `source` (`tree`,
`loop`, `gap` or `skeleton`). On a whole circle-of-Willis tree it covers 97 % of the skeleton.

```{code-block} python
from nvitk.morphology.centering import mask_surface
from nvitk.vmtk.centerlines import full_centerline_network

net = full_centerline_network(mask_surface(mask, spacing), mask, spacing)
net.branches, net.radii, net.sources, net.junctions
```

## Installation

conda-forge's VMTK build for Python 3.11 needs NumPy < 2, and upstream publishes wheels only for
CPython ≥ 3.12. nvitk therefore installs a **cp311 wheel of VMTK 1.5.2** built from the upstream
sdist against `vtk==9.6.2`, the same VTK pin the upstream wheels use. See {doc}`../installation`.

## Quirks worth knowing

- VMTK scripts log to stdout; the wrappers silence them (`LogOn = 0`).
- `vmtkSurfaceRemeshing` can leave unreferenced points. `remesh()` cleans them; left in place they
  read as zero curvature and break connectivity.
- The stream tracer's `TerminalSpeed` is a ratio, not mm/s. `FlowTraceParams.min_speed` drops slow
  seeds instead.
- At each bifurcation, VMTK blanks a stretch of every centerline. `VmtkNetwork.vessel_branches`
  therefore stop short of one another, `bifurcations` are the blanked pieces, and
  `joined_branches()` reconnects them (matching each branch against every centerline's blanked
  piece, not just the longest one).
- A mask component is not always one marching-cubes surface: voxels touching only at an edge or
  a corner give separate pieces. The full network therefore works per surface piece.

```{seealso}
Full generated reference: [`nvitk.vmtk`](../autoapi/nvitk/vmtk/index).
```
