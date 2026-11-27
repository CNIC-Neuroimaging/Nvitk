# PESA-Health

Whole-body CT/PET and Dixon MRI fat-quantification pipelines (v5). `nvitk-pesa-health` is the
batch driver: shared DICOM→NIfTI conversion, then either or both of two independent
sub-pipelines (CT-PET, Dixon), then an optional local QC report.

```{code-block} bash
nvitk-pesa-health --input-dir /data/raw --output-dir /data/pesa_health \
  --pipelines ct-pet-v5,dixon-v5 --submit local
```

## Stages

| Stage | Command | What it does |
|---|---|---|
| 0 | (inside `nvitk-pesa-health`) | Shared DICOM → NIfTI conversion and renaming (`common/stage0_convert.py`). |
| 1 (CT-PET) | (inside `nvitk-pesa-health-ctpet`) | TotalSegmentator on CT, all configured tasks run sequentially → `res_segmentation_ct/<SUBJECT>/CT/<task>.nii.gz`. |
| 2 (CT-PET) | " | Post-processing into `MO`/`FAT`/`BODY`/`ORGANS`/`MUSCLES.nii.gz` (v5: muscles kept hemisphere-split). |
| 3 (CT-PET) | " | SUV + volume measurement — PET resampled onto each mask grid, written to a per-subject Excel. |
| 1–3 (Dixon) | `nvitk-pesa-health-dixon` | Mirrors the CT-PET stage shape with a `--dixon-regions` selector instead of CT tasks. |
| 4 | `nvitk-pesa-health-qc` | Static per-subject + batch-index HTML QC report. |
| — | `nvitk-pesa-health-qc-portal` | Interactive review portal over those reports (static HTML + Excel/DB-backed review state). |
| — | `nvitk-pesa-health-sync-measurements` | Reads per-subject stage-3 Excel and upserts long-form rows into the `image_measurements` DB table. |
| — | `nvitk-pesa-health-hotspot` | Standalone 3D hotspot viewer over the SUV volume (CT-PET) or raw Dixon FF/T2*/R2* maps. |

## Local vs. SGE dispatch

- **`--submit local`** — everything runs in-process, sequentially per subject.
- **`--submit sge`** — stage 0 becomes one SGE job per subject; each selected sub-pipeline
  then submits one **array job per subject** (tasks = its stages, concurrency-limited, with
  done-markers), held on that subject's stage-0 job id. CT-PET and Dixon arrays for the same
  subject run in parallel once stage 0 finishes. An optional batch-aggregate job (merges
  per-subject stage-3 Excel into a `<batch>_SummaryCodebook.xlsx`) and the stage-4 QC job
  hold on all stage-3 array job IDs.

Cluster defaults live in `sge.json` under `pipelines.pesa_health_ct_pet` /
`pipelines.pesa_health_dixon`: qsub overrides, log/error subdirs, model/container roots, and the
local/cluster DICOM/NIfTI/results/model roots (the two pipelines share one batch layout, so the
roots may be set on either; CT-PET wins when both set one) — see
{doc}`../api/cluster-registry`.

## Command reference

```{eval-rst}
.. click:: nvitk.pipes.pesa_health.run_batch:main
   :prog: nvitk-pesa-health
   :nested: full

.. click:: nvitk.pipes.pesa_health.ct_pet_v5.run:main
   :prog: nvitk-pesa-health-ctpet
   :nested: full

.. click:: nvitk.pipes.pesa_health.dixon_v5.run:main
   :prog: nvitk-pesa-health-dixon
   :nested: full

.. click:: nvitk.pipes.pesa_health.run_hotspot:main
   :prog: nvitk-pesa-health-hotspot
   :nested: full

.. click:: nvitk.pipes.pesa_health.common.stage4_qc:main
   :prog: nvitk-pesa-health-qc
   :nested: full

.. click:: nvitk.pipes.pesa_health.qc.portal_cli:main
   :prog: nvitk-pesa-health-qc-portal
   :nested: full

.. click:: nvitk.pipes.pesa_health.sync_measurements:main
   :prog: nvitk-pesa-health-sync-measurements
   :nested: full
```

```{seealso}
Full generated reference: [`nvitk.pipes.pesa_health`](../autoapi/nvitk/pipes/pesa_health/index).
```
