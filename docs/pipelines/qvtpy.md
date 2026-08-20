# QVTPy

4D-flow MRI hemodynamics and TOF morphometrics — a from-scratch Python port of the
measurement concepts in the original MATLAB [QVTplus](https://github.com/ABI-Animus-Laboratory)
tool (format-compatible LOC output, not a wrapper around it: `nvitk.pipes.qvtpy` has no code
dependency on the vendored `qvtplus/` MATLAB tree).

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
| `stage1` | `eicab` | eICAB on `TOF/TOF.nii.gz`. |
| `stage2` | `registration` | eICAB TOF (resampled) → 4D-flow reference, **rigid FSL FLIRT**. |
| `stage3` | `centerline` | eICAB in 4D-flow space + arterial/venous centerline extraction. |
| `stage4` | `segmentation` | Complex-difference crop per vessel + threshold + optional region growing → `seg_4dflow`. |
| `stage4t` | `seg_t` | Same, per `ComplexDifference_4D` frame → `seg_4dflow_4d` (opt-in). |
| `stage5` | `loc` | QVTplus-style Location-of-Comparison (LOC) CSV, arterial + venous. |
| `stage6` | `measure` | Per-LOC masked-plane flow / PI / RI from the phase volumes — see {doc}`qvtpy-hemodynamics`. |
| `stage7` | `morphometrics`, `morpho` | TOF eICAB morphometrics — caliber/tortuosity/stenosis — see {doc}`qvtpy-morphometrics`. |
| `stage8_xnat_upload` | `xnat_upload` | Uploads `eicab/` + `qvtpy/` results to the XNAT session (also standalone as `nvitk-qvtpy-xnat-upload`; requires stage 2–7 complete). |
| `stage9_autoqc` | `autoqc`, `qc` | DB-only QC scoring of published measurements — see {doc}`qvtpy-autoqc`. |

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
| **SPM12 / MATLAB QVTplus** | No code dependency — the vendored `src/nvitk/pipes/qvtplus/` tree is the original MATLAB reference implementation kept for provenance; `nvitk.pipes.qvtpy` reimplements the measurement concepts in Python independently. |

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
