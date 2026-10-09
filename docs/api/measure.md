# Measure

`nvitk.measure` exposes every quantitative metric in the toolkit through two complementary
APIs: standalone functional primitives, and a `Measurer` orchestrator for chaining several
measurements over the same image/mask pair.

```{code-block} python
from nvitk.measure import volume_cc, masked_stats, Measurer

vol = volume_cc(mask)
stats = masked_stats(pet, mask, stats=("mean", "max"))
summary = Measurer(pet, mask).volume() | Measurer(pet, mask).suv(kinds=("bw",))
```

| Module | Purpose |
|---|---|
| `volume` | Voxel/physical volume. |
| `intensity` | Masked intensity statistics. |
| `suv` | Standardized Uptake Value (PET) computation. |
| `voxel` | Voxel-overlap metrics — Dice, Jaccard. |
| `surface` | Surface-based metrics (Hausdorff distance, surface area). |
| `radiomics` | Radiomics feature extraction (PyRadiomics-backed). |
| `compare` | Paired-image comparison utilities. |
| `hemodynamics`, `mask_hemodynamics` | Flow/pressure-derived hemodynamic indices used by the {doc}`QVTPy pipeline <../pipelines/qvtpy>`. |
| `cross_section` | Oblique-plane cross-section extraction — the shared primitive behind QVTPy's per-station flow measurements. |
| `measurer` | The `Measurer` orchestrator class itself. |
| `voxelwise` | Cohort voxelwise GLM with permutation FWE correction (FSL `randomise`) — see below. |

## Vascular morphometrics (`nvitk.measure.morpho`)

A large, dedicated subsystem for Circle-of-Willis-style vascular morphometrics (caliber,
tortuosity, stenosis, tree topology) — the engine behind {doc}`the QVTPy pipeline's stage 7
<../pipelines/qvtpy>` and the standalone TOF morphometrics tools. Key pieces:

| Module | Purpose |
|---|---|
| `anatomy`, `anatomy_axes`, `labels_util` | Anatomical region/label definitions. |
| `centerlines`, `skeleton` | Centerline/skeleton extraction feeding the morphometrics. |
| `caliber`, `metrics` | Per-segment caliber and derived morphometric measures. |
| `tree_regions`, `tree_segments`, `topology_io` | Vessel-tree topology construction and I/O. |
| `donut_loops`, `geometry`, `surface` | Cross-section geometry helpers. |
| `orchestration`, `run_case` | Full-case orchestration entry points. |
| `export_utils/` | Tortuosity metric export, radius histograms, PVSM scene generation, summary tables. |
| `topology/*.json` | Per-cohort topology definitions (eICAB, mouse-root, QVTPy). |

## Voxelwise analysis (`nvitk.measure.voxelwise`)

Every other measurement in this module is *region-wise*: one number per subject × parcel or vessel.
That answers "which region" and cannot answer "**where** in this region", and it presumes the
parcellation is the right unit. `voxelwise` fits the same GLM at every voxel instead, estimating
the null by permutation with FSL's `randomise`, so the result is a family-wise-error-corrected map
rather than a table.

The images and the design matrix are deliberately independent — images come from a flat directory
of spatially normalised volumes, the design from database measurements — so a cross-modal question
is the normal case, not a special one:

```{code-block} python
from nvitk.measure.voxelwise import run_voxelwise

result = run_voxelwise(
    "/data/ASL_MNI", "results/asl_vs_lmca",
    include="*_s8_*",
    cohort="4dflow_v3",                       # subjects that pipeline published
    evs=["flow_mean__LMCA", "age_at_mri", "sex"],
    contrasts=["+flow_mean__LMCA:mca_positive"],
    n_perm=5000, tfce=True,
)
print(result.summary())
```

| Piece | Purpose |
|---|---|
| `fsl_backend_status()` | Probe `FSLDIR`, `fslmerge` and `randomise` **separately** — they ship in different conda packages, so an image can have one and not the other. Never raises. |
| `resolve_cohort_images()` | Glob a flat directory, auto-detect the session id in each name, resolve it to a `subject_uid` through `subject_ids`. Returns an **ordered** list. |
| `cohort_subjects()` / `available_cohorts()` | The `--cohort` filter: subjects with measurements under a given `pipeline_id`. Distinct from `DataRepo`'s `cohort_id` membership filter. |
| `validate_common_space()` | Refuse a stack whose volumes are not on one grid, naming the first offender. |
| `VoxelwiseDesign` | EVs, contrasts, demeaning, rank checks, and FSL VEST writers (`design.mat` / `design.con`) written directly — no `Text2Vest` needed. |
| `merge_4d()` / `run_randomise()` | `fslmerge -t` and `randomise` (or `randomise_parallel`, run locally — see below). |
| `PreFilter` / `apply_prefilters()` | Subject inclusion rules on measurement values (`flow_mean__LICA>=15`), applied before the design is validated. |
| `run_voxelwise()` | The one-call path: resolve → validate → design → merge → randomise. |
| `validate_randomise_inputs()` / `run_randomise_direct()` | The other path: a 4D stack and `design.mat` / `.con` / `.fts` built elsewhere, checked against each other from their headers, then `randomise`. |
| `load_voxelwise_result()` | Read a finished results folder with no FSL, no database and no re-run — what both viewers use. t-contrast and F-test maps are numbered separately; `names_for(kind)` gives the names a map kind is addressed by. |

### The ordering contract

Row *i* of the design matrix must be volume *i* of the 4D stack. `resolve_cohort_images` returns an
ordered list and everything downstream is indexed off it; `align_design_to_images` drops a subject
from *both* sides at once. A transposed or shuffled design still runs and still produces a
plausible-looking map, so this is never inferred, only carried — and `manifest.json` records the
final order so a result can be re-checked later.

### Filename → subject

Filenames carry an *acquisition* id, not a subject id, and which kind depends on who exported the
directory: `BMRI100102` under `mr_id`, `IA004754` under `session`, and so on. Real cohorts mix them
in one folder, so no single regex covers a directory.

Rather than ask for one, every letters-then-digits token in the filename is checked against the
`mr_id` / `mri_id` / `session` namespaces, and whichever token is a registered code wins. All three
namespaces resolve onto the same `subject_uid`, which is the key `--cohort` and the design frame
are both written against — the standardisation has to happen *before* the cohort intersection, or
the filter silently removes everything.

`--id-pattern` overrides this with an explicit regex, for an id glued to neighbouring text with no
delimiter, or a name where auto-detection would be ambiguous. Hidden files (`.nii`, macOS `._*`
forks) are skipped: they turn up on network shares and have no stem to carry an id.

### One row per subject

A voxelwise design has exactly one row per subject, so two files claiming the same `subject_uid`
have to be reconciled. `--on-duplicate` decides:

| Value | Behaviour |
|---|---|
| `error` (default) | Report **every** conflict at once and stop. Which session is kept changes the result, so this is the caller's call. |
| `skip` | Drop those subjects entirely — no arbitrary choice. |
| `first` / `last` | Keep one, by filename order. |

When most subjects are duplicated the same number of times, the error says so and points at
`--include` instead: that is one image in several derivatives (`_s0`, `_s8`, `_s12`), not a cohort
with repeat scans, and a duplicate policy would silently pick a smoothing level.

### Subject set

`images found ∩ cohort ∩ prefilters ∩ design-frame complete cases`, applied in that order with a
count logged at each step. A cohort that silently removes half the images is otherwise
indistinguishable from a bad include pattern.

### Prefilters

`--prefilter` narrows the *cohort* on a measurement value, before the design is validated:

```bash
--prefilter 'flow_mean__LICA>=15' --prefilter 'Hematocrit>=42'
```

Repeatable, combining with AND, and each rule logs its own before/after count — a rule that removes
almost everything is otherwise indistinguishable from a mis-typed column. Operators: `>=`, `<=`,
`!=`, `==`, `>`, `<` (a bare `=` is read as `==`). A non-numeric value compares as text and only
supports `==` / `!=`.

The column **need not be an EV**: keeping only subjects with a patent left ICA while modelling the
right one is a cohort decision, not a covariate, and a column named only in a prefilter is pulled
into the frame for the purpose. A missing value never satisfies a comparison, so a subject with no
measurement in the tested column is excluded rather than silently kept.

Rules are recorded in `manifest.json` alongside the pre-filter subject count, so a result says what
cohort produced it.

### Prepared 4D stack and design files

When the stack and the design already exist — built in FSL's Glm GUI, in another tool, or with
`nvitk-voxelwise design` and `merge` — `randomise` runs on them directly, with no database and no
cohort resolution:

```{code-block} bash
nvitk-voxelwise randomise -i stack_4d.nii.gz -d design.mat -t design.con -f design.fts \
  -m brain_mask.nii.gz -o results/vw_glm --n-perm 5000 --parallel
```

Before anything runs (or uploads), the files are checked against each other from their headers and
every mismatch is reported at once: design rows against stack volumes, contrast weights against
design columns, `.fts` columns against t-contrasts (0/1 only, no empty F-test), the mask's grid
against the stack's, and each VEST header against the rows that actually follow `/Matrix`. A design
built for a *different* stack of the same length cannot be detected, so the volume count is echoed
for a person to recognise.

There is **no default mask** here, unlike `run`: a prepared stack need not be in MNI space. The
design files are copied into the results folder as `design.mat` / `.con` / `.fts`, and
`manifest.json` records `"mode": "prepared"`, the input paths and the contrast and F-test names
(from `/ContrastName<i>`, else `c1…` / `f1…`). Its subject list is empty — a prepared stack carries
no record of which volume is which subject.

F-tests produce `…fstat<N>` maps alongside the `…tstat<N>` ones. The two are numbered separately, so
every viewer lists F-test names for an F map and contrast names for a t map.

### `randomise_parallel` on one machine

`randomise_parallel` splits the permutations into fragments and hands them to `fsl_sub` — a cluster
submitter. Run locally, FSL's `fsl_sub` either follows site configuration and queues the fragments
(returning before any map exists) or starts **one per core**. Every fragment holds the whole 4D
stack in memory (about twice its float32 size: ~1 GB for 120 subjects at 2 mm, ~4.4 GB for 500), so
32 cores means 32 copies, which is what took workstations down.

`--parallel` therefore runs `randomise_parallel` against a throwaway `FSLDIR` overlay in which
`fsl_sub` is nvitk's own runner (`nvitk/measure/fsl_sub_local.sh`, derived from FSL 6.0's bash
`fsl_sub`). The FSL installation is never modified. The runner:

- runs fragments on this machine only — never `qsub`, whatever is configured;
- caps how many run at once, and pins each to one thread;
- stops the merge step if any fragment failed. `randomise_parallel` ignores the fragment step's
  exit code, so otherwise an incomplete set of permutations would be merged into plausible
  p-values.

How many fragments run at once, first match wins:

| Source | Notes |
|---|---|
| `--jobs N` (GUI: *jobs*) | Explicit. Warns if the estimated memory exceeds what is free. |
| `FSLSUB_PARALLEL=N` | The variable FSL's own `fsl_sub` reads, so one setting caps both. `0` or unset means "choose". |
| automatic | The cores available (`NSLOTS` inside an SGE job, else the affinity mask), lowered until the fragments' estimated memory fits in 80 % of what is free. |

`--seed` is not passed on: `randomise_parallel` seeds fragment *k* with *k*, so a parallel run is
reproducible regardless. `--n-perm 0` (exhaustive) needs plain `randomise`. Paths containing spaces
are refused, since `randomise_parallel` re-expands its arguments unquoted. As FSL itself warns, a
design with fewer unique permutations than requested (a small one-sample test, say) should run
without `--parallel`.

A re-run into the same output root first removes what the previous run with that root left: the
merge step appends to the permutation-distribution files, and a stale map from a design with more
contrasts would otherwise be read as part of the new result.

### Reading the maps

`randomise` writes corrected p-values as **1 − p**, so a value above 0.95 means p < 0.05. Bright is
*evidence*, not effect size. {py:mod}`nvitk.stats.voxelwise_map` holds the figure builders shared by
the napari tool and the Statmodels `Display → Voxelwise` pane, so the two viewers cannot disagree
about the threshold, the colormap or the caption.

The threshold is a **window**, not a single cut: `lo`/`hi` on the 1 − p scale, default 0.95 → 1.0.
Narrowing the top (0.95 → 0.99) isolates the marginal shell from the voxels that pass
overwhelmingly — and because that hides the *strongest* voxels too, the caption says so rather than
reading as a plain threshold.

Three views, each with its own controls: the cortical **surface** (hemisphere, view angles, pial or
inflated, and a rotatable 3-D form), the **glass brain** (every nilearn projection including the
four-panel `lyrz`), and **orthogonal slices** (any slicer mode, with cut coordinates in mm driven by
sliders clamped to the map's own bounding box, or an integer montage). The registries differ per
builder — `lyrz` is glass-only, `mosaic` slice-only — so the mode list follows the view.

The napari GUI adds a third view: a 3-D scene ({py:mod}`nvitk.gui.viz.voxelwise_3d`, configured
from **Visualization → Voxelwise 3D scene**) putting the suprathreshold voxels inside a translucent
brain shell, as iso-surfaces or as points coloured by value. The shell is drawn in both modes.

It draws **any** map `randomise` wrote, not only the corrected ones, and the threshold follows the
kind: a 1 − p map is windowed one-sided on the value, a `tstat` map two-sided on `|value|` so a
negative effect is not silently dropped. Defaults come from the map's own distribution, since a
signed statistic has no conventional cut.

The shell resolves through three sources, because the obvious one is usually absent — `run_voxelwise`
only writes `mask.nii.gz` into the results folder when it *derived* the mask itself, so a run with
`--mask` leaves none behind:

1. `mask.nii.gz` beside the maps,
2. the path recorded in `manifest.json`, if still readable here,
3. the MNI152 brain mask resampled onto the result's grid.

### CLI

```{code-block} bash
nvitk-voxelwise cohorts          # which pipeline ids --cohort accepts, with subject counts
nvitk-voxelwise status           # is FSL usable here, and which binary is missing if not
nvitk-voxelwise design  …        # write design.mat / design.con and report the intersection
nvitk-voxelwise merge   …        # write the 4D stack, in subject order
nvitk-voxelwise run     …        # the whole analysis; --submit local|sge
nvitk-voxelwise randomise …      # randomise on a prepared stack + design.mat/.con/.fts; --submit local|sge
nvitk-voxelwise report <dir>     # summarise a finished results folder
nvitk-voxelwise fetch <dir> --to # pull a finished cluster result back over SFTP
```

### Running on the cluster

`--submit sge` never shells out to a local `qsub` — a workstation does not have one. It follows the
same path as the `qvtpy` and `pesa_fat` pipelines: write the driver script locally, publish it over
SFTP, run it over SSH, and report the job ids.

`--from-source` says where the inputs are:

| Value | Behaviour |
|---|---|
| `local` (default) | The cohort is resolved **here** — include pattern, cohort, prefilters, complete cases — and only the volumes the design actually keeps are uploaded to `<output>/inputs/`, along with the mask. A 4000-file directory typically sends a few hundred. |
| `sge` | `--image-dir` and `--mask` are already cluster paths. They are bound in place and nothing is transferred. |

`-o` is always a cluster-visible root under `--submit sge`, so it is used verbatim; the uploaded
inputs live inside it and one analysis is one self-contained directory.

`randomise --submit sge` follows the same path. With `--from-source local` the stack and design are
checked here first, then the stack is uploaded under its own name (skipped when the cluster copy
has the same size) and the design files and mask under fixed names (always re-sent — two designs
over the same subjects differ in values, not in size). With `--from-source sge` the paths must be
absolute; each input's folder is mounted at its own path.

`--parallel` on the cluster requests `-pe smp N` slots for the one job — `--jobs`, else
`sge_pe_smp` in `pipelines.voxelwise`, else 4 — and the fragments run inside it on exactly those
slots, through `NSLOTS`. They are never submitted as separate jobs: the compute node runs the
container, which has no `qsub`. On most SGE setups `sge_h_vmem` applies **per slot**.

Credentials come from `NVITK_SGE_SSH_HOST` / `_USER` / `_PASSWORD`, then `--remote-host` /
`--remote-user`, then a prompt. The env vars alone are sufficient, which is what lets the napari
tool submit through a subprocess that has no tty.

```{code-block} bash
nvitk-voxelwise run --image-dir /local/CBF_MNI --include '*_s8*' --cohort 4dflow_v3 \
  --ev flow_mean__RMCA --ev age_at_mri --ev sex --ev Hematocrit \
  --contrast '0,1,0,0,0:rmca_positive' --prefilter 'flow_mean__RMCA>15' \
  --mask /local/atlas/brain_mni.nii \
  -o /data_lab_MCC/…/RESULTS/Voxelwise/vwRMCA --submit sge --from-source local

nvitk-voxelwise fetch /data_lab_MCC/…/RESULTS/Voxelwise/vwRMCA --to ./vwRMCA
```

`--dry-run` prints the binds, the qsub line and the upload count without transferring or
submitting; `--emit-script PATH --no-remote` writes the script for you to run yourself.
`fetch` pulls the maps, manifest and design by default and leaves the 4D stack on the cluster —
several GB that nothing local reads (`--all` takes everything).

```{warning}
nvitk Singularity images before `v2026.10.08` install `fsl-base`, `fsl-flirt`, `fsl-avwutils` and
`fsl-warpfns` but **not** `fsl-randomise`: `fslmerge` is present, `randomise` is not, and
`--submit sge` fails on the queue at the randomise step. From `v2026.10.08` the image template adds
`fsl-randomise`. Local runs are unaffected, and `fsl_backend_status()` reports the gap by name.
```

```{seealso}
Full generated reference: [`nvitk.measure`](../autoapi/nvitk/measure/index).
```
