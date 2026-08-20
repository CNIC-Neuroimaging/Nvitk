# Segmentation

`nvitk.segmentation` covers label-map operations and several segmentation engines, from
lightweight in-process algorithms to external-tool wrappers.

## Label-map operations

| Module | Purpose |
|---|---|
| `labels` | Generic label-map manipulation (relabel, merge, select). |
| `mask_ops` | Boolean algebra over masks (union, intersection, keep-inside/outside). |
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

## PET-specific

[`nvitk.segmentation.pet.ureter_segmentation`](../autoapi/ureter_segmentation/index) — ureter
segmentation for whole-body PET/CT quantification pipelines (see {doc}`../pipelines/index`).

```{seealso}
Full generated reference: [`nvitk.segmentation`](../autoapi/nvitk/segmentation/index).
```
