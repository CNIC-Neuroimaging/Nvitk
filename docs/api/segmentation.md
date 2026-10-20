# Segmentation

`nvitk.segmentation` covers label-map operations and several segmentation engines, from
lightweight in-process algorithms to external-tool wrappers.

## Label-map operations

| Module | Purpose |
|---|---|
| `labels` | Generic label-map manipulation (relabel, merge, select). |
| `interactive` | Region growing for manual segmentation (the Labeling tab's tools): `GrowSession` (an image prepared once — smoothed, on the backend — for the magic wand, the adaptive confidence-connected flood and the Frangi vesselness flood from seeds), `trace_vessel_path` (minimal path through the vesselness between clicked points), `path_radii` / `tube_from_path` (a tube around it, its radius measured from the lumen), `path_intensity_fill` (the voxels out to a reach from the path whose intensity lies in the path's own range, connected to it), `adaptive_brush_keep` (the smart brush's Otsu / tolerance split). |
| `mask_ops` | Boolean algebra over masks (union, intersection, subtract, xor, complement), and keeping or removing an image's voxels by a mask's labels (`apply_mask_to_image`: fill with a value, the image minimum or NaN; margin in mm; 3D mask over 3D+t images). |
| `region_growing` | Seeded region growing. |
| `blood_flood` | Flood-fill style vessel/blood-pool segmentation. |
| `hull_edt` | Convex-hull / Euclidean-distance-transform based segmentation helpers. |
| `protrusion_filter` | Removes small protrusions from a label. |

## ANTsPyNet-backed segmentation

| Module | Purpose |
|---|---|
| `brain_extraction` | Skull-stripping. |
| `mra_vessel` | MRA vessel segmentation. |
| `dkt` | Desikan-Killiany-Tourville cortical parcellation. |
| `hemisphere` | Left/right hemisphere split. |
| `mouse_brain` | Mouse-specific brain segmentation. |

## External-engine wrappers

These shell out to (or containerize) a separate segmentation tool rather than reimplementing
it — see {doc}`../installation` for what needs to be on `PATH`.

| Command | Package | Purpose |
|---|---|---|
| `nvitk-totalseg` | `nvitk.segmentation.total_segmentator` | Wraps the `TotalSegmentator` CLI (local or SGE, CPU or GPU) rather than importing its Python API directly. |
| `nvitk-eicab` | `nvitk.segmentation.eicab` | Runs eICAB (TOF / Circle-of-Willis segmentation) via Singularity. |
| `nvitk-fastsurfer` | `nvitk.segmentation.fastsurfer` | Runs FastSurfer from the tree installed in the environment (see below). |

## FastSurfer

[FastSurfer](https://deep-mi.org/FastSurfer/) is the deep-learning counterpart of FreeSurfer.
nvitk installs **v2.5.4** — the newest release that runs on nvitk's Python 3.11 (v2.6 needs
3.12) — into `$CONDA_PREFIX/share/nvitk/fastsurfer`, with its network weights, and runs it with
the environment's own Python and PyTorch. Every module is a `nvitk-fastsurfer` subcommand and a
GUI tool (Tools ▸ *FastSurfer*):

| Subcommand | What it produces | Needs |
|---|---|---|
| `seg` | `aparc.DKTatlas+aseg` (subcortical + DKT cortex), aseg, brain mask, PV-corrected stats; by default also the three modules below | T1 |
| `cereb` | CerebNet cerebellar lobules / vermis | a segmented subject (or `--t1`) |
| `hypothal` | HypVINN hypothalamic sub-structures (`--t2` improves it) | a segmented subject (or `--t1`) |
| `cc` | corpus callosum + fornix segmentation and shape metrics | a segmented subject (or `--t1`) |
| `surf`, `run`, `long` | recon-surf surfaces, thickness, surface parcellation; the full pipeline; longitudinal | FreeSurfer 7.4.1 + license |
| `segstats`, `reduce-to-aseg`, `conform` | label statistics, aparc+aseg → aseg, FastSurfer's conforming | — |

`--native-image` (the GUI's default) writes every result on the input's own grid. Label maps use
FreeSurfer's ids (cerebellum 601-631, hypothalamus 961-984, CC 250-255) and open in the GUI named
and coloured from FreeSurferColorLUT. The surface and longitudinal pipelines need FreeSurfer 7.4.1
and a (free) FreeSurfer license: `nvitk-fastsurfer install-freesurfer [--license FILE]` downloads
it (≈9.5 GB, resumable) outside the environment, or point to an existing one with
`nvitk-fastsurfer set-freesurfer DIR` / `set-license FILE`. `nvitk-fastsurfer info` reports
what is installed.

Every processing subcommand also runs on the cluster with `--submit sge` (plus `--emit-script`,
`--no-remote`, `--dry-run`, `--direct-submit` as for the other nvitk CLIs, and `--sge-gpu/--sge-cpu`):
the folders its path arguments live in are bound into the nvitk container (`/nvitk/ext/<n>`) and the
paths rewritten, so they must be paths the cluster sees too. The container brings FastSurfer and a
pruned FreeSurfer 7.4.1; the license is bound from `--fs-license` (or the local configuration).
`nvitk-fsl` works the same way (`nvitk-fsl bet in.nii.gz out --submit sge`).

## PET-specific

[`nvitk.segmentation.pet.ureter_segmentation`](../autoapi/ureter_segmentation/index) — ureter
segmentation for whole-body PET/CT quantification pipelines (see {doc}`../pipelines/index`).

```{seealso}
Full generated reference: [`nvitk.segmentation`](../autoapi/nvitk/segmentation/index).
```
