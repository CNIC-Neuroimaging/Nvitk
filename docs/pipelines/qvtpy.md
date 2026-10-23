# QVTPy

4D-flow MRI hemodynamics and TOF morphometrics — a from-scratch Python port of the
measurement concepts in the original MATLAB [QVTplus](https://github.com/ABI-Animus-Laboratory)
tool (format-compatible LOC output, not a wrapper around it: `nvitk.pipes.qvtpy` needs no
MATLAB).

```{code-block} bash
nvitk-qvtpy --subject-dir /data/subj01 --stages stage0_c,stage1,stage2,stage3,stage4,stage5,stage6 \
  --submit local
```

## Stages

`DEFAULT_STAGES` = `stage0_c,stage1,stage2,stage3,stage4,stage5,stage6` — stages `4t`, `7`,
`8`, and `9` are opt-in.

| Stage | Alias | What it does |
|---|---|---|
| `stage0_d` | `download` | XNAT → DICOM (local only). |
| `stage0_c` | `convert`, `stage0` | DICOM → NIfTI + reorg + optional `phase2volume` derivatives. |
| `stage1` | `eicab` | eICAB on `TOF/TOF.nii.gz`, then the eICAB post-process (below). |
| `stage2` | `registration` | eICAB TOF (resampled) → 4D-flow reference, **rigid FSL FLIRT**. |
| `stage3` | `centerline` | eICAB in 4D-flow space + arterial/venous centerline extraction. |
| `stage4` | `segmentation` | Complex-difference crop per vessel + threshold + optional region growing → `seg_4dflow`. |
| `stage4t` | `seg_t` | Same, per `ComplexDifference_4D` frame → `seg_4dflow_4d` (opt-in). |
| `stage5` | `loc` | QVTplus-style Location-of-Comparison (LOC) CSV, arterial + venous. |
| `stage6` | `measure` | Per-LOC masked-plane flow / PI / RI from the phase volumes — see {doc}`qvtpy-hemodynamics`. |
| `stage7` | `morphometrics`, `morpho` | TOF eICAB morphometrics — caliber/tortuosity/stenosis — see {doc}`qvtpy-morphometrics`. |
| `stage8_xnat_upload` | `xnat_upload` | Uploads `eicab/` + `qvtpy/` results to the XNAT session (also standalone as `nvitk-qvtpy-xnat-upload`; requires stage 2–7 complete). |
| `stage9_autoqc` | `autoqc`, `qc` | DB-only QC scoring of published measurements — see {doc}`qvtpy-autoqc`. |

## Stage 1: the eICAB post-process

After eICAB, `--post-process-eicab` (on by default, in `nvitk-qvtpy` and in
`python -m nvitk.pipes.qvtpy.stage1_eicab`) fits eICAB's circle-of-Willis labels to the TOF
(`nvitk.pipes.qvtpy.util.eicab.eicab_postprocess`, library steps in
`nvitk.segmentation.vessel_refine`). It writes `TOF_eICAB_CW_pp.nii.gz`,
`centerlines_mask_pp.nii.gz` and `eicab_postprocess.json` beside eICAB's outputs, which stay
untouched; stage 7 measures the `_pp` mask by default. In order:

1. **Anterior choroidals dropped** (LACHA / RACHA, labels 17 / 18).
2. **Local re-segmentation.** eICAB's masks are thicker than the lumen. Each label is re-drawn at
   its local half maximum, `T = bg + 0.5 (core − bg)`:
   - `core` is the TOF on the label's skeleton and `bg` the TOF 1–3 mm around it, both spread
     along the vessel (σ 2 mm). Only that vessel's own intensities count, never the whole
     volume's, and the threshold is applied to the image lightly denoised (σ 0.3 mm).
   - The other labels are barriers. A label grows by at most half its radius (≤ 1 mm).
   - A label too faint for a half maximum (contrast < 0.5) keeps its mask.
3. **ICAs and basilar to the image floor.** eICAB's box cuts them at its lower face while the lumen
   goes on (the petrous ICA, the vertebrobasilar junction).
   - Each trunk follows the minimum-cost path, cheap along bright tubes, through a multi-scale
     Frangi tree from its end down to a vessel cross-section in the image's last slices.
   - The path starts from the label's lowest band and from its lowest skeleton tip (the open end).
     It may run up to 5 mm above the label's lowest voxel, because the petrous ICA often leaves
     eICAB's label sideways at the bottom of its bend before descending.
   - Bright lumen (above the trunk's half maximum) counts as tree, which bridges the Frangi gaps
     at junctions and tight bends. Frangi fades at the volume's edge, so the last 3 mm accept any
     voxel above the trunk's gate.
   - The lumen around the path is drawn at the same local half maximum. On 32 PESA subjects
     every ICA reaches the bottom slices, except one whose eICAB labels are misplaced.
     Before 2026-10-23, 4 right ICAs stopped 4–39 mm short with a flat cut.
   - The basilar follows up to two branches down. A second branch is kept when at least 4 mm of
     it runs apart from the first (beyond twice the trunk radius) along a vessel (mean step cost
     ≤ 2.5 of 1–5). It no longer needs to cost less than 2.5× the first: the second vertebral
     often reaches the floor two or three times farther away. Two branches: 19 / 32 subjects
     (9 before). The 4D-flow pipeline's vertebral logic
     (`util/segmentation/vertebral_split.split_vertebral_from_basilar`: a centerline Y below the
     confluence, sides from the ICAs) then splits them off as **LVA / RVA**. These are new labels
     19 / 20: stage 7's topology knows them, and stages 2–4 drop them, since they grow their own.
   - A trunk with no path to the floor falls back to following the tree as one tube.
4. **Distal growth** (`--eicab-pp-distal`, on by default). The ACAs, MCAs, PCAs (P2) and SCAs grow
   along the multi-scale Frangi tree beyond eICAB's box, as the blood flood floods its markers:
   - all at once and in competition: each voxel goes to the vessel whose front (geodesic, cheaper
     along vessel centres) reaches it first, up to 60 mm, and the other labels are barriers;
   - **ACAs**: the fronts advance together, step for step, so each A2 is reached first by its own
     ACA. Each side of the AComm starts that side's front, since an A2 often begins at the AComm.
     Neither ACA may grow more than 5 mm past a midline plane oriented as the head's mid-sagittal
     plane (from the head outline) and anchored at the AComm.
5. **Siphon correction** (`--eicab-pp-siphon`, on by default). Where an ICA siphon's limbs touch
   and close a loop, only the contact is cut (`nvitk.morphology.siphon_cut`: pretess, then an
   intensity-weighted minimum cut), as *ICA siphon centerline correction* does in the GUI.

`--no-post-process-eicab` skips the post-process (stage 7 then measures eICAB's mask);
`--only-pp` runs it alone on existing eICAB outputs. The GUI runs the same steps, each
switchable, as *Segmentation ▸ eICAB post-process*. It follows nvitk's backend (`--backend`,
`NVITK_BACKEND`): the filters, distance transforms and the Frangi Hessian run on the GPU with CuPy,
about 16 s per subject instead of ~38 s on the CPU (PESA10017225). The minimum-cost paths, the
skeleton, the GMM tree and the siphon cuts stay on the host. The GPU mask differs from the CPU one
by a few float32 rounding voxels (Dice ≥ 0.996 on most vessels, 0.96 on one small SCA); the CPU
backend reproduces the earlier output exactly. The Frangi filter is nvitk's own, the same measure
as scikit-image's, about 6× faster on the CPU.

**Reusing an existing eICAB mask** (`--reuse-eicab`, off by default; `nvitk-qvtpy` and
`stage1_eicab`, local and SGE). Each subject's line in the log says which branch it took:
- **Mask found**: the subject's eICAB folder already holds `*_eICAB_CW` and `TOF_resampled`.
  eICAB is skipped and only the post-process runs.
- **No mask**: eICAB runs, by default in **CW-only** mode. eICAB has no option for this, so
  nvitk's `sitecustomize` hook inside the container (`NVITK_EICAB_CW_ONLY`) stops it once the CW
  mask and `TOF_resampled` are written. The CW centerline/metrics and the whole-brain (WB) flood
  fill are skipped, about 20 min per subject on the cluster. The CW mask is identical to a full
  run's (checked voxel for voxel on PESA10017225).
- **On SGE**: the check runs at submit time, where a found mask gives a post-process-only job
  without a GPU, and again inside the eICAB job at run time.
- **WB mask**: `--eicab-wb/--no-eicab-wb` overrides whether eICAB runs its WB flood fill (on by
  default, off for the `--reuse-eicab` fallback). Without a WB mask, stage 3's venous exclusion
  uses the CW labels.
- **With `--skip-processed`**: that check counts stage 1 done as soon as any eICAB NIfTI exists,
  so it skips the subject before `--reuse-eicab` can add a missing post-process.

What it changes downstream:
- Stage 7 measures longer ICAs (to the image floor), the vertebrals as their own vessels, much
  larger ACA / MCA / PCA / SCA trees, and diameters ~15–25 % smaller.
- Stage 7 takes longer: ~285 s instead of ~60 s on PESA17808400 with 6 workers.
- Stage 7's lesion thresholds were calibrated on eICAB's masks ({doc}`qvtpy-morphometrics`).

Known limits:
- The competition can only continue the vessels eICAB started. In one test subject both A2s leave
  the left half of a wide AComm and both go to the left ACA.
- When the basilar shows no Y above the floor, the vertebrals stay part of it.

## Local vs. SGE dispatch

- **`--submit local`** — an in-process loop over subjects and stages.
- **`--submit sge`** — one array job per subject (`qsub -t`, task = pending stage,
  concurrency-limited with done-markers), chunked/"drip-fed" via
  `--sge-subject-chunk-size` to stay under per-user SGE job caps, with optional SSH remote
  execution.

Stage 0 can also pull directly from XNAT (`--from-source xnat`, with
`--save-dicoms/--no-save-dicoms` controlling whether downloaded DICOMs persist). Cluster
defaults live in `.nvitk/sge.json` under `pipelines.qvtpy` / `pipelines.qvtpy_paths` — see
{doc}`../api/cluster-registry`.

## External tool notes

| Tool | Used here? |
|---|---|
| **FSL FLIRT** | Yes — stage 2's registration is a direct `nipype` FSL FLIRT call. Needs FSL on `PATH`, see {doc}`../installation`. |
| **FreeSurfer / Desikan atlas** | No — despite being a general nvitk prerequisite (used elsewhere for ASL/T1/brain-parcellation work, see {doc}`../stats-gui/index`), QVTPy itself has no FreeSurfer dependency. |
| **SPM12 / MATLAB QVTplus** | No — `nvitk.pipes.qvtpy` reimplements the measurement concepts in Python independently. |

## Related commands

`nvitk-qvtpy-flowshow` opens an interactive 4D-flow viewer ({func}`nvitk.viz.flowshow`) over
the AP/RL/FH phase NIfTIs and a multilabel vessel mask, searching a fallback order for the
mask (up to the stage-4 `seg_4dflow.nii.gz` output) and optionally loading `locs.csv` +
`centerlines_mask.nii.gz` when pointed at a full pipeline output root.

## Command reference

```{eval-rst}
.. click:: nvitk.pipes.qvtpy.run:main
   :prog: nvitk-qvtpy
   :nested: full

.. click:: nvitk.pipes.qvtpy.run_flowshow:main
   :prog: nvitk-qvtpy-flowshow
   :nested: full

.. click:: nvitk.pipes.qvtpy.xnat_upload:main
   :prog: nvitk-qvtpy-xnat-upload
   :nested: full

.. click:: nvitk.pipes.qvtpy.stage9_autoqc:main
   :prog: nvitk-qvtpy-autoqc
   :nested: full
```

```{seealso}
Full generated reference: [`nvitk.pipes.qvtpy`](../autoapi/nvitk/pipes/qvtpy/index). Detailed
measurement math: {doc}`qvtpy-hemodynamics`, {doc}`qvtpy-morphometrics`, {doc}`qvtpy-autoqc`.
```
