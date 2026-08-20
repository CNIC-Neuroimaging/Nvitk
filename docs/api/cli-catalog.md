# CLI Catalog

Every nvitk command is a registered entry point (`[project.scripts]` in `pyproject.toml`).
`pyhelp` discovers and renders all of them from that same source, so this page and `pyhelp`
never drift apart.

```{code-block} bash
pyhelp                    # interactive tree; Enter on a command selects it and prints --help
pyhelp --no-interactive   # full static tree (colors)
pyhelp --flat             # legacy flat listing
```

`nvitk.cli.catalog` (`CatalogNode`, `ToolEntry`, `build_catalog_tree`,
`parse_pyproject_scripts`) parses `pyproject.toml` directly and builds the tree that both
`pyhelp` (`nvitk.util.list_cli_commands`, `nvitk.util.pyhelp_tree`) and
{doc}`nvitk-gui's Tools dock <../gui/index>` render from — the GUI's tool catalog is a
superset that adds parameter forms on top of the same underlying command list.

`nvitk.cli` also hosts the module-level CLI implementations themselves (`ants`, `fireants`,
`morphology`, `restoration`, `filters`, `measure`, `transform`, `segmentation`) — the click
commands backing the entry points in the table below.

## Full command reference

| Command | Role |
|---|---|
| **Conversion** | |
| `dcm2nii` | DICOM → NIfTI |
| `stl2nifti` | Surface mesh → labelmap/NIfTI |
| `phase2volume` | Phase-contrast MRI → velocity volume |
| `nikon2nifti` | Nikon microscopy → NIfTI |
| **Segmentation** | |
| `nvitk-totalseg` | TotalSegmentator wrapper (local/SGE, CPU/GPU) |
| `nvitk-eicab` | eICAB TOF / Circle-of-Willis segmentation |
| `nvitk-seg` | General segmentation entry point |
| **Registration** | |
| `nvitk-flirt` | FSL FLIRT wrapper |
| `nvitk-ants` | ANTsPy registration |
| `nvitk-fireants` | FireANTs (GPU) registration |
| **Image module CLIs** | |
| `nvitk-morph` | Morphology |
| `nvitk-restore` | Restoration |
| `nvitk-filter` | Filters |
| `nvitk-measure` | Metrics |
| `nvitk-transform` | Geometric transforms |
| **PESA-Fat pipeline** — see {doc}`../pipelines/pesa-fat` | |
| `nvitk-pesa-fat` | Batch driver |
| `nvitk-pesa-fat-ctpet` | CT/PET pipeline |
| `nvitk-pesa-fat-dixon` | Dixon pipeline |
| `nvitk-pesa-fat-hotspot` | Hotspot viewer |
| `nvitk-pesa-fat-qc` | QC report |
| `nvitk-pesa-fat-qc-portal` | QC review portal |
| `nvitk-pesa-fat-sync-measurements` | Publish measurements to the DB |
| **PESA-Brain cohort pipelines** — see {doc}`../pipelines/qvtpy` | |
| `nvitk-bbtpy` | BBT-py batch driver |
| `nvitk-qvtpy` | 4D-flow hemodynamics pipeline |
| `nvitk-qvtpy-flowshow` | 4D-flow interactive viewer |
| `nvitk-qvtpy-xnat-upload` | XNAT upload |
| `nvitk-qvtpy-autoqc` | Automated QC |
| **Database / sync** | |
| `nvitk-xnat-sync` | XNAT dataset sync |
| `nvitk-xnat-pipeline-sync` | XNAT pipeline-resource sync |
| **GUI** — see {doc}`../gui/index` and {doc}`../stats-gui/index` | |
| `nvitk-gui` | Napari workbench |
| `nvitk-statsmodels` | Statistical modeling workbench |
| **General** | |
| `pyhelp` | This catalog, interactively |

```{seealso}
Full generated reference: [`nvitk.cli`](../autoapi/nvitk/cli/index),
[`nvitk.util`](../autoapi/nvitk/util/index).
```
