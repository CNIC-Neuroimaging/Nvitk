# PESA-Fat

Whole-body CT/PET and Dixon MRI fat-quantification pipelines (v5). `nvitk-pesa-fat` is the
batch driver: shared DICOM→NIfTI conversion, then either or both of two independent
sub-pipelines (CT-PET, Dixon), then an optional local QC report.

```{code-block} bash
nvitk-pesa-fat --input-dir /data/raw --output-dir /data/pesa_fat \
  --pipelines ct-pet-v5,dixon-v5 --submit local
```

## Stages

| Stage | Command | What it does |
|---|---|---|
| 0 | (inside `nvitk-pesa-fat`) | Shared DICOM → NIfTI conversion and renaming (`common/stage0_convert.py`). |
| 1 (CT-PET) | (inside `nvitk-pesa-fat-ctpet`) | TotalSegmentator on CT, all configured tasks run sequentially → `res_segmentation_ct/<SUBJECT>/CT/<task>.nii.gz`. |
| 2 (CT-PET) | " | Post-processing into `MO`/`FAT`/`BODY`/`ORGANS`/`MUSCLES.nii.gz` (v5: muscles kept hemisphere-split). |
| 3 (CT-PET) | " | SUV + volume measurement — PET resampled onto each mask grid, written to a per-subject Excel. |
| 1–3 (Dixon) | `nvitk-pesa-fat-dixon` | Mirrors the CT-PET stage shape with a `--dixon-regions` selector instead of CT tasks. |
| 4 | `nvitk-pesa-fat-qc` | Static per-subject + batch-index HTML QC report. |
| — | `nvitk-pesa-fat-qc-portal` | Interactive review portal over those reports (static HTML + Excel/DB-backed review state). |
| — | `nvitk-pesa-fat-sync-measurements` | Reads per-subject stage-3 Excel and upserts long-form rows into the `image_measurements` DB table. |
| — | `nvitk-pesa-fat-hotspot` | Standalone 3D hotspot viewer over the SUV volume (CT-PET) or raw Dixon FF/T2*/R2* maps. |

## Local vs. SGE dispatch

- **`--submit local`** — everything runs in-process, sequentially per subject.
- **`--submit sge`** — stage 0 becomes one SGE job per subject; each selected sub-pipeline
  then submits one **array job per subject** (tasks = its stages, concurrency-limited, with
  done-markers), held on that subject's stage-0 job id. CT-PET and Dixon arrays for the same
  subject run in parallel once stage 0 finishes. An optional batch-aggregate job (merges
  per-subject stage-3 Excel into a `<batch>_SummaryCodebook.xlsx`) and the stage-4 QC job
  hold on all stage-3 array job IDs.

Cluster defaults live in `.nvitk/sge.json` under `pipelines.pesa_fat_ct_pet` /
`pipelines.pesa_fat_dixon` (log/error subdirs, model/container roots) and
`pipelines.pesa_fat_paths` (local/cluster DICOM/NIfTI/results/model roots) — see
{doc}`../api/cluster-registry`.

## Command reference

```{eval-rst}
.. click:: nvitk.pipes.pesa_fat.run_batch:main
   :prog: nvitk-pesa-fat
   :nested: full

.. click:: nvitk.pipes.pesa_fat.ct_pet_v5.run:main
   :prog: nvitk-pesa-fat-ctpet
   :nested: full

.. click:: nvitk.pipes.pesa_fat.dixon_v5.run:main
   :prog: nvitk-pesa-fat-dixon
   :nested: full

.. click:: nvitk.pipes.pesa_fat.run_hotspot:main
   :prog: nvitk-pesa-fat-hotspot
   :nested: full

.. click:: nvitk.pipes.pesa_fat.common.stage4_qc:main
   :prog: nvitk-pesa-fat-qc
   :nested: full

.. click:: nvitk.pipes.pesa_fat.qc.portal_cli:main
   :prog: nvitk-pesa-fat-qc-portal
   :nested: full

.. click:: nvitk.pipes.pesa_fat.sync_measurements:main
   :prog: nvitk-pesa-fat-sync-measurements
   :nested: full
```

```{seealso}
Full generated reference: [`nvitk.pipes.pesa_fat`](../autoapi/nvitk/pipes/pesa_fat/index).
```
