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
| `stage4` | `segmentation` | Complex-difference crop per vessel + threshold, the ICAs' and basilar's region growing, then the blood flood (Frangi + hysteresis vessel tree on the CD divided by the subject's blood signal; watershed from the ACAs / MCAs / PCAs, the two ACAs sharing theirs in turns behind a midline guard; comms and venous sinuses fill only their own lumen; A1 / M1 / P1 growth labelled A2 / M2 / P2; small islands dropped) → `seg_4dflow`. |
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
`nvitk.segmentation.vessel_refine`). It writes beside eICAB's outputs, which stay untouched:

- `TOF_eICAB_CW_pp.nii.gz`: steps 1–3 and 5, eICAB's labels without the distal growth. Its ACAs /
  MCAs read as the **A1 / M1** (eICAB ids 5–8).
- `TOF_eICAB_WB_pp.nii.gz` (with `--eicab-pp-distal`, the default): steps 1–5. What the ACAs / MCAs
  gain in step 4 is written as their **A2 / M2** (new eICAB ids 21–24), so they split as the PCA
  already does into P1 / P2. Without `--eicab-pp-distal` only the CW_pp mask is written (a stale
  WB_pp is removed).
- `centerlines_mask_pp.nii.gz` (of the CW_pp mask) and `eicab_postprocess.json` (the shared steps
  once, then a `cw` and a `wb` section).

Stage 3 and stage 7 use the WB_pp mask when it exists, else CW_pp (`--eicab-mask cw` for CW_pp).

**Scanner frame.** eICAB resamples the TOF without the rotation of an oblique acquisition's affine
(it keeps the voxel sizes and the origin): its `TOF_resampled` matches the original TOF resampled
*without* its rotation (correlation 1.00; 0.4–0.8 with it). Overlaid on `TOF.nii.gz`, eICAB's
masks therefore sit 2–20 mm off (rotations of 2.5–23° on the 12 test subjects). When stage 1 knows
the original TOF it writes the pp masks with the **scanner-frame affine** (same voxels, true
positions; `scanner_frame_affine`, recorded as `scanner_frame` in the report). Stages 2–3 register
with FLIRT on voxels, so they are unaffected. The GUI QC loader shows eICAB's own files in the
same frame.

In order:

1. **Anterior choroidals dropped** (LACHA / RACHA, labels 17 / 18).
2. **Local re-segmentation.** eICAB's masks are thicker than the lumen. Each label is re-drawn at
   its local half maximum, `T = bg + 0.5 (core − bg)`:
   - `core` is the TOF on the label's skeleton and `bg` the TOF 1–3 mm around it, both spread
     along the vessel (σ 2 mm). Only that vessel's own intensities count, never the whole
     volume's, and the threshold is applied to the image lightly denoised (σ 0.3 mm).
   - The other labels are barriers. A label grows by at most half its radius (≤ 1 mm).
   - A label too faint for a half maximum (contrast < 0.5) keeps its mask.
3. **ICAs to the image floor.** eICAB's box cuts them at its lower face while the lumen goes on
   (the petrous ICA). Since 2026-10-29 the basilar is left as eICAB drew it (re-segmented only):
   no extension, no vertebral arteries.
   - Each ICA follows the minimum-cost path, cheap along bright tubes, through a multi-scale
     Frangi tree from its end down to a vessel cross-section in the image's last slices.
   - The path starts from the label's lowest band and from its lowest skeleton tip (the open end).
     It may run up to 5 mm above the label's lowest voxel, because the petrous ICA often leaves
     eICAB's label sideways at the bottom of its bend before descending.
   - Bright lumen (above the trunk's half maximum) counts as tree, which bridges the Frangi gaps
     at junctions and tight bends. Frangi fades at the volume's edge, so the last 3 mm accept any
     voxel above the trunk's gate.
   - The lumen around the path is drawn at the same local half maximum. The path itself stays
     lumen where it is above the trunk's gate, so one dim bend no longer cuts off the rest.
   - **The ICAs go to the image's floor however far it is** (since 2026-10-28): their crop and
     route reach the lowest slice with real signal (the near-zero resampling slice below it does
     not count), the path may be 2.5× the drop + 20 mm long, and the lowest slice is tried first.
     Past where the route meets the floor, the lumen that runs on along the bottom slices (the
     petrous ICA lying horizontally) is followed for up to 25 mm. On the 12 test subjects both
     ICAs reach the lowest slice (`reached_floor`, `floor_gap_mm` in the report).
   - After every step (re-segmentation, ICAs, distal growth, siphon) each label is reduced to its
     largest connected component, its enclosed holes are filled (`tidy_labels`, unlabelled
     voxels only; an A2 / M2 keeps every piece on its A1 / M1, its branches meeting only there),
     and its pieces under 5 mm³ are dropped (`drop_small_islands`: the loose voxels a growth step
     leaves; real separate M2 pieces are ~20 mm³ and more).
   - An ICA with no path to the floor falls back to following the tree as one tube.
   - **ICA loops below the siphon** (`loop_fill`, since 2026-10-29). In a tight loop the blood
     runs along the outer wall and the inner wall comes out darker on TOF: the mask misses a
     crescent there. The siphon is the cluster of bends (≥ 0.20 mm⁻¹ at a 3 mm scale, as in the
     morphometrics) around the sharpest bend in the ICA's upper 60 %, with the bends up to 8 mm
     before it. From the floor up to 2 mm before it, every 0.5 mm of centerline, the ICA's own
     cross-section is compared with its convex outline, and the gaps at least 0.35 mm deep are
     filled (`segmentation/cross_section_fill`). The staircase along a straight wall is thinner
     than that, so the outer wall does not move. 0–147 voxels per ICA (≤ 18 mm³) on the 12 test
     subjects.
4. **Distal growth** (`--eicab-pp-distal`, on by default; WB_pp only). The ACAs, MCAs, PCAs (P2)
   and SCAs grow along the multi-scale Frangi tree beyond eICAB's box, as the blood flood floods
   its markers. The ACAs' and MCAs' growth is written as A2 / M2 (21–24):
   - all at once and in competition: each voxel goes to the vessel whose front (geodesic, cheaper
     along vessel centres) reaches it first, up to 60 mm, and the other labels are barriers;
   - **ACAs**: the fronts advance together, step for step, so each A2 is reached first by its own
     ACA. Each side of the AComm starts that side's front, since an A2 often begins at the AComm.
     Neither ACA may grow more than 5 mm past a midline plane oriented as the head's mid-sagittal
     plane (from the head outline) and anchored at the AComm.
5. **Siphon correction** (`--eicab-pp-siphon`, on by default), on each mask separately, so the
   distal growth never fills a cut. Where an ICA siphon's limbs touch and close a loop, only the
   contact is cut (`nvitk.morphology.siphon_cut`: pretess, then an
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
- Stage 7 measures longer ICAs (to the image floor), the
  A1 / A2, M1 / M2 and P1 / P2 as separate vessels (WB_pp; with CW_pp the ACAs / MCAs are named
  A1 / M1), and diameters ~15–25 % smaller.
- Stage 3 maps the split into the 4D-flow labels: qvtpy 4–9 are the A1 / M1 / P1 (renamed
  `LACA_A1` … `RPCA_P1`) and 17–22 the A2 / M2 / P2 (15 / 16 unused: the SCAs are not segmented
  in 4D flow). Stage 4 also writes what
  an A1 / M1 / P1 floods beyond its own lumen as its A2 / M2 / P2, so a CW_pp input splits too.
- Stage 7 takes longer: ~285 s instead of ~60 s on PESA17808400 with 6 workers.
- Stage 7's lesion thresholds were calibrated on eICAB's masks ({doc}`qvtpy-morphometrics`).

Known limits:
- The competition can only continue the vessels eICAB started. In one test subject both A2s leave
  the left half of a wide AComm and both go to the left ACA.

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
