# ICA siphon centerline correction — methodology

This document describes `nvitk.morphology.centerline_siphon` — the
topological correction we apply to Internal Carotid Artery (ICA) centerlines
on TOF MRA so that the polyline follows the full cavernous siphon curl
instead of shortcutting through the false donut closure.

The exploration notebook (`eicab_reseg.ipynb`) was the testbed; this module
is the productionised, backend-aware port (NumPy and CuPy).

---

## 1. Problem statement

The ICA has a hairpin-shaped **cavernous siphon** just before the bifurcation
into MCA + ACA. At our TOF resolution the two limbs of that hairpin often sit
within partial-volume distance of each other, so the segmentation merges them:
the topologically correct **hook** (a tube, β₁ = 0) becomes a **donut / torus**
(β₁ ≥ 1). When `compute_centerlines` extracts a centerline through such a mask,
the longest-path (skeleton-diameter) walker shortcuts across the false bridge
and skips most of the curl — visible as a centerline endpoint that lands inside
the loop instead of at the bifurcation tip.

We want the centerline to:

- traverse the full vessel from **skull-base entry** to **bifurcation tip**,
- bypass the false bridge by going **around the curl** (not through it),
- never modify the underlying segmentation mask (mask repair is left to other
  downstream code — this module only fixes the polyline).

---

## 2. Topological observation that drives the fix

For any donut mask, its 26-connected skeleton has at least one cycle (β₁ of the
skeleton graph is ≥ β₁ of the mask). That cycle is composed of **two arcs**:

| arc | length | what it is |
| --- | ------ | ---------- |
| bridge | short chord across the donut hole | the false closure created by partial-volume |
| curl   | long perimeter around the donut hole | the true vessel path |

The arc lengths are **invariant** under any orientation / head-tilt — there is
no axis-aligned prior to get wrong. The bridge can be found purely by
graph-topology, no Y / Z heuristics:

1. Find the cycle (`networkx.minimum_cycle_basis`, fallback `cycle_basis`).
2. Walk it in cyclic order (we rebuild the order because nx does **not**
   contractually return it).
3. Pick two **anchor nodes** on the cycle — they are the points the curl arc
   terminates on, so we keep them:
   - **Degree-3+ junctions** on the cycle (real branch points where the
     stem / tip attach to the loop). If there are >2 such junctions, pick the
     pair with the largest Z-spread.
   - Otherwise, if there is one junction: that junction **plus** the cycle's
     max-Z voxel (anatomical apex / "future tip leaf" of the curl).
   - Otherwise (pure floating cycle, no junction): the cycle's min-Z and
     max-Z voxels.
4. The cycle splits into two arcs at the anchors → drop the **shorter** arc
   from the skeleton (the mask is untouched).

The remaining skeleton is the curl + stem + tip; on a tree it has at least two
degree-1 leaves. We pick:

- `base` = **min-Z degree-1 leaf** (skull-base entry, lowest in the brain),
- `tip`  = **max-Z degree-1 leaf** (ICA bifurcation, highest in the brain).

`networkx.shortest_path(G, base, tip)` is unique on a tree, so the corrected
centerline is fully determined: base → curl → tip, with no shortcut and the
endpoints guaranteed to be at the vessel extremes.

This is what `prune_skeleton_shortest_arc` + `compute_corrected_centerline`
implement.

---

## 3. Public API

```python
from nvitk.morphology import (
    GenusReport,
    SiphonCorrectionResult,
    compute_corrected_centerline,
    compute_mask_genus,
    correct_siphon_centerlines,
    prune_skeleton_shortest_arc,
)
```

### `compute_mask_genus(mask, *, label_name="vessel", connectivity=1)`

Standalone topology probe. Returns `GenusReport(label_name, n_voxels,
n_components, euler_chi, beta0, beta1, skeleton_cycles, skeleton_voxels)` —
`report.suspect` is `True` iff `beta1 > 0` (a donut handle exists).
Inputs may be NumPy or CuPy; the report is JSON-serialisable via `to_dict()`.

### `prune_skeleton_shortest_arc(mask, *, label_name="vessel")`

Skeletonize → graph → split each cycle at its anchors → drop the shorter arc.
Returns `(sk_pruned, bridge_voxels, info)` — `sk_pruned` is a backend array
(NumPy or CuPy depending on the active backend), `bridge_voxels` is a list of
`(i, j, k)` tuples, and `info["cycles"]` records the per-cycle arc lengths,
anchors and Y/Z ranges.

### `compute_corrected_centerline(mask, *, label_name="vessel")`

End-to-end primitive for a **single** binary mask. Prunes the bridge as above,
then traces the `min-Z leaf → max-Z leaf` shortest path on the pruned
skeleton. Returns `(path, sk_pruned, info)` with `path` shape `(N, 3)`
float32 in voxel coords, ordered base → tip.

### `correct_siphon_centerlines(tof, vessel_mask, *, correction_ids=(0, 1), out_dir=None, save_qc=False, min_points=3)`

The top-level driver — **same order as `eicab_reseg.ipynb` Cells 4–8**:

1. Seed centerlines from the input multilabel `vessel_mask`.
2. Per ICA in `correction_ids`: local Otsu on TOF inside the seed-CL bbox,
   2-iter erosion, optional `repair_ica_donut_3d` when β₁ > 0.
3. `compute_corrected_centerline` on the **repaired Otsu mask** (not the raw
   eICAB label).
4. Default `compute_centerlines` for all other labels.

The input segmentation volume is not written back; only centerline / bridge
outputs are saved. NIfTI writes use `_imsave_like_reference(..., tof)` so
affine, zooms, and axes match the TOF `Image` metadata (same as notebook
`save_mask_nii(..., ref=tof)`).

Outputs (when `out_dir` is provided):

| file | content |
| ---- | ------- |
| `corrected_centerlines.nii.gz` | per-label centerline mask (corrected for `correction_ids`, default for the rest) — saved with the TOF affine/header. |
| `removed_bridges.nii.gz`       | per-label bridge-voxel mask (same label IDs as `correction_ids`) — saved with the TOF affine/header. |
| `siphon_correction.json`       | per-label metadata (cycle/anchor info, base/tip, warnings). |
| `qc_siphon_correction.png`     | 3D matplotlib QC overlay (only when `save_qc=True`). |

Returns:

```python
{
  "centerlines": {label_id: (N, 3) backend array, ...},
  "bridges":     {label_id: [(i, j, k), ...]},
  "details":     {label_id: SiphonCorrectionResult.to_dict()},
  "corrected_centerlines_mask": (X, Y, Z) int32 backend array,
  "removed_bridges_mask":       (X, Y, Z) int32 backend array,
  "output_paths": {"centerlines": ..., "bridges": ..., ...},
}
```

`correction_ids` defaults to `(0, 1)` (the project convention for RICA / LICA).
It accepts any label IDs present in `vessel_mask`.

---

## 4. Quick reference (algorithm walk-through)

For each `lid` in `correction_ids`:

1. `roi = vessel_mask == lid`
2. `sk = skeletonize_binary(roi)` — 26-conn 3D skeleton (CPU, scikit-image).
3. Build a 26-connected `networkx.Graph` from `sk`.
4. `cycles = nx.minimum_cycle_basis(G)` (fallback `cycle_basis`).
5. For each cycle:
   - Walk it in cyclic order (defensive, since nx ordering is not
     contractual).
   - Anchors = junctions on the cycle (degree > 2 in G), or min/max-Z if
     there are < 2 junctions.
   - Split the cycle into two arcs at the anchors; drop the **shorter** arc
     voxels from `sk` (record them as bridge voxels).
6. `G' = skeleton_to_graph(sk_pruned)`; pick `base = min-Z degree-1 leaf`,
   `tip = max-Z degree-1 leaf` (fallback: min/max-Z over all nodes).
7. `path = nx.shortest_path(G', base, tip)` → ordered `(N, 3)` polyline.
8. Rasterise all centerlines (including the unchanged default ones for
   non-corrected labels) into the centerline mask; rasterise the bridge
   voxels (per label) into the bridges mask. Both volumes are saved with the
   TOF metadata so their NIfTI affines match the input image space.

Why "shortest arc" is the bridge:

- The bridge is **by anatomical definition** a short chord across the donut
  hole — the curl is the long way around. So `len(arc1) < len(arc2)`
  ⇒ `arc1` is the bridge.
- Anchoring on degree-3+ junctions ensures we never cut into the stem / tip
  branches: anchors are the literal points where stem and tip leave the loop.

Why min-Z / max-Z leaves are the right endpoints:

- After bridge pruning, the skeleton is a tree on its main connected
  component. A tree's diameter (what `compute_centerlines` would pick) is
  not guaranteed to terminate at the anatomical extremes — it can land at
  the broken bridge stub.
- min-Z / max-Z leaves are exactly the skull-base entry and bifurcation tip
  (Z is the inf↔sup axis in our acquisitions). Pinning the centerline
  there makes the polyline fully reproducible across subjects.

---

## 5. Backend (NumPy / CuPy) handling

The module registers `np`, `scipy`, `ndi` via
`nvitk.core.backend.setup(globals())`, so heavy array ops (e.g. label
rasterisation, ROI extraction) follow the active backend. CPU-only
libraries (`scikit-image` skeleton, `networkx` graphs, `matplotlib`,
`marching_cubes` if invoked) are fed via `to_numpy(...)` and their results
are wrapped back into the active backend via `as_backend_array(...)`.

Usage:

```python
import nvitk as nv
from nvitk.core.backend import using
from nvitk.morphology import correct_siphon_centerlines

tof = nv.imread("TOF.nii.gz", backend="cupy")
mask = nv.imread("vessel_mask.nii.gz", backend="cupy")

with using("cupy"):
    res = correct_siphon_centerlines(
        tof, mask,
        correction_ids=(0, 1),
        out_dir="out/",
        save_qc=True,
    )
```

The CPU and CuPy paths are bit-identical on the centerline / bridge masks
for the same inputs (verified against the PESA15689521 subject:
`np.array_equal(cpu_mask, gpu_mask) == True`).

`networkx` operates on Python tuple keys, not arrays, so the graph walks are
always CPU regardless of backend; the only cost of GPU is the host↔device
copy of the skeleton (a few thousand voxels per ICA — negligible).

---

## 6. Affine / image-space preservation

Outputs are written via `nvitk.io.imsave(..., metadata=dict(tof_img.metadata))`.
The TOF Image's metadata carries the `affine` (read by the NIfTI reader from
the sform / qform); the writer (`nvitk.io.writers.nifti.write_nifti`) consumes
`metadata["affine"]` and zooms (`x_res`, `y_res`, `z_res`) directly. Smoke test
on PESA15689521 confirms `np.allclose(tof.affine, output.affine) == True` for
both `corrected_centerlines.nii.gz` and `removed_bridges.nii.gz`.

If you pass a raw NumPy / CuPy array as `vessel_mask` without an `Image`
wrapper, the function logs a warning and falls back to an identity affine for
the outputs (because the array has no metadata to inherit). Use
`nv.imread(path)` or wrap arrays in `nvitk.types.Image(data=arr, metadata={
"affine": <4x4>, "x_res": ..., ...})` to keep the image space.

---

## 7. Reference numbers (PESA15689521)

For comparison, this is what the algorithm produces on the dev subject:

| label | cycle len | bridge | curl | base (min-Z) | tip (max-Z) | corrected pts |
|-------|----------:|-------:|-----:|--------------|-------------|--------------:|
| LICA (id=1) | 43 | **5** | 36 | (148, 228, **z=24**) | (147, 237, **z=63**) | 81 |
| RICA (id=2) | 42 | **4** | 36 | (200, 225, **z=27**) | (202, 236, **z=64**) | 82 |

Before the correction, both centerlines were ~50 pts and the max-Z endpoint
landed inside the loop (the diameter walker picked the bridge stub as the
"farthest" node). After the correction the centerlines span the full Z extent
of each vessel (24→63 and 27→64) and route around the curl apex.

---

## 8. Limits

- The "shorter arc = bridge" rule relies on the anatomical convention that the
  curl is the long way around. If a subject's siphon is so warped that the
  curl is shorter than the bridge, the heuristic fails. We have not seen this
  in our cohort; if it shows up, the per-cycle `info` log records both arc
  lengths so it's trivial to flag.
- Endpoint selection assumes Z is the inf↔sup axis. If you re-orient the
  volume (e.g. transpose to a non-standard view), pass a re-oriented mask
  too; the module only knows about array indices, not anatomical labels.
- The module **does not** repair the segmentation mask itself. The donut
  mask remains β₁ = 1 on disk. If you want a topologically clean mask for
  downstream consumers, run the (notebook-only) `repair_ica_donut_3d` step
  before calling this module — but the centerline correction is independent
  of that and works on the donut directly.

---

## 9. Where to look next

- Source: `src/nvitk/morphology/centerline_siphon.py`.
- Public re-exports: `src/nvitk/morphology/__init__.py`.
- Original exploration: `notebooks/exploration/pesabrain-anatomy/eicab_reseg.ipynb`.
- Genus / topology helpers used by this module: `compute_mask_genus`,
  `nvitk.morphology.skeletonize_binary`, `nvitk.morphology.label_connected`.
