# qvtpy methodology

Pipeline code: `src/nvitk/pipes/qvtpy/`. This document tracks **what each stage does**, **inputs/outputs**, **runtime dependencies**, and **compute backend** (CPU vs GPU-capable array code). Update the changelog when behavior or defaults change in a material way.

## Changelog

| Date | Summary |
|------|---------|
| 2026-05-15 | Stages 3–6 MATLAB-aligned overhaul: `--eicab-mask cw\|wb` with fallback; mask cleaning + `cd_vessel_binary_qc.nii.gz`; geometry-based venous centerlines (`venous_SSSV`, …); centerline-backbone `seg_4dflow` (default voxel assembly); QVTplus-style LOC heuristics; per-LOC cross-sectional area and masked-plane flow / PI / RI. |
| 2026-05-12 | Initial stages 2–6, NiPype FLIRT registration module, `measure.hemodynamics` helpers, SGE `hold_jid` chain in `run.py`. |
| 2026-05-12 | `pipes/qvtpy/labels.py`: full eICAB ID table (1–18 plus basilar / AComm), canonical names (LICA, RICA, BASILAR, …), QVT venous name strings (SSSV, LTSV, RTSV, STRV), and qvtpy extension label ints (`QVTPY_VENOUS_UNKNOWN_LABEL`, …). |
| 2026-05-12 | `phase2volume`: single QVTplus-aligned derivative path; VENC from JSON → NIfTI metadata → DICOM → 700 mm/s + warning; default polynomial background correction (`fit_order` default 2). |
| 2026-05-12 | qvtpy stages / `phase2volume` / `hemodynamics` / `_phase2volume_bg`: use `nvitk.core.backend.setup(globals())` and `as_backend_array` so heavy array ops follow `NVITK_BACKEND` (NumPy or CuPy). Venous `skeletonize` (skimage) stays CPU via `to_numpy` before skimage. |
| 2026-05-12 | Stage 2: FLIRT **moving** image is eICAB ``TOF_resampled`` under ``--output-root/<subject>/eicab/`` (not raw ``TOF/TOF.nii.gz``). ``registration_meta.json`` records ``moving_kind: eicab_tof_resampled``. Optional CLI ``--eicab-subdir`` on ``qvtpy-stage2-registration``. |
| 2026-05-12 | Stage 4: ``seg_4dflow`` uses the same CD sliding-threshold vessel mask as stage 3 (via ``util.flow_volume_masks``); venous voxels are only inside the **venous slab** (first third along axis 1), with 0.5% area opening, then the **four largest** connected components receive ids **31–34** (``VENOUS_REGION_BASE``…); no global fill with label 30. |
| 2026-05-12 | Stage 3: global binary mask on ``ComplexDifference_3D`` via sliding-threshold curve (median 3³, step 0.001 up to 0.8×max, FWHM-shifted curvature pick, smooth width 10) + 0.5% foreground area filter; venous slab = first third along axis 1; venous mask = global mask ∧ slab + second 0.5% filter; skeleton venous centerlines. |

## Backend (GPU vs CPU)

nvitk selects the array stack via `NVITK_BACKEND` / `nvitk.using("cupy")` (see [`src/nvitk/core/backend.py`](../../src/nvitk/core/backend.py)). qvtpy modules that touch large volumes call `setup(globals())` and coerce voxel data with `as_backend_array` instead of `numpy.asarray`, matching patterns in [`src/nvitk/morphology/centerline.py`](../../src/nvitk/morphology/centerline.py) and [`src/nvitk/transform/oblique.py`](../../src/nvitk/transform/oblique.py).

| Stage / component | GPU-friendly (CuPy when backend=cupy) | CPU-only or host-bound notes |
|-------------------|----------------------------------------|------------------------------|
| Stage 0 convert + `phase2volume` | Yes: CD/MAG/velocity math, polynomial BG fit (`np.linalg.lstsq`, `ndi` not required there) | DICOM/NIfTI I/O ends on CPU for writers; `imsave` materializes NumPy for NIfTI. |
| Stage 1 eICAB | External container (separate from this table) | Not governed by nvitk `np` proxy. |
| Stage 2 FLIRT | No | FSL `flirt` is a CPU binary; moving image is read from stage 1 eICAB outputs on the host. |
| Stage 3 centerlines | Partial: multilabel arterial on GPU where applicable | ``ndi`` median filter + ``nvitk.morphology.components`` area opening; venous skeleton via ``skeletonize_binary`` (CPU skimage). |
| Stage 4 segmentation | Partial: ``as_backend_array`` for CD read | Cross-section seg + ``nvitk.morphology.components``; oblique reslicing via ``ndi``. |
| Stage 5 LOC | Mostly trivial (reads `.npz` on CPU) | CSV I/O on CPU. |
| Stage 6 measure | Yes: phase volumes, `velocity_mm_s_from_phases`, PI/RI reductions | Per-LOC loop builds small NumPy vectors for dot products; negligible. |

**skimage / non-CuPy libraries:** Any call that cannot accept CuPy arrays must wrap with `to_numpy(...)` before the call and, if the rest of the pipeline should stay on GPU, move results back with `as_backend_array` where it pays off (stage 3 venous branch only needs NumPy coordinates for `np.savez`).

## Data layout (after stage 0)

Per subject under `--nifti-root`:

- `TOF/TOF.nii.gz` — TOF magnitude.
- `4DFlow/{AP,RL,FH}/` — magnitude `*_m.nii.gz`, phase `*_ph.nii.gz`, JSON sidecars.

Optional derivatives under `4DFlow/` from `phase2volume` (see `--compute-phase-derived` on stage 0 / `nvitk-qvtpy`):

- `Angiography_3D`, `ComplexDifference_3D`, `VelocityMagnitude_3D`, …
- `VelocityMeanComponents.nii.gz` (shape `X×Y×Z×3`, mean velocity components in mm/s).

### Core imaging — PC phase to velocity

Encoders store normalized phase; nvitk maps to mm/s consistently with stage 6 and [`hemodynamics.py`](../../src/nvitk/measure/hemodynamics.py):

- \(v_x = -\mathrm{RL}_{\mathrm{phase}} \times 10\) mm/s  
- \(v_y = -\mathrm{AP}_{\mathrm{phase}} \times 10\) mm/s  
- \(v_z = \mathrm{FH}_{\mathrm{phase}} \times 10\) mm/s  

### Derivatives (single recipe)

Time-mean angiography `Angiography_3D` = temporal mean of the AP magnitude stack. Velocity magnitude \(|V| = \sqrt{v_x^2+v_y^2+v_z^2}\). Complex-difference style contrast (QVTplus `calc_angio` intent):

\[
\mathrm{CD} = \mathrm{MAG} \cdot \sin\left(\frac{\pi}{2}\cdot\frac{\min(|V|, \mathrm{VENC})}{\mathrm{VENC}}\right)
\]

Reference implementation: `_calc_angio` in [`phase2volume.py`](../../src/nvitk/io/conversors/phase2volume.py).

### Background phase correction

[`_phase2volume_bg.py`](../../src/nvitk/io/conversors/_phase2volume_bg.py): temporal mean velocity → speed percentile mask → spatial polynomial (order 2 or 3, default **2** to match MATLAB `loadNII` `fit_order`) in normalized voxel coordinates \([-1,1]^3\) → subtract the 3D field from the mean field and from **each** time frame → recompute \(|V|\) and CD. MATLAB’s `cd_thresh` / `noise_thresh` voxel gating is **not** replicated; Python uses `static_percentile` (default **25**) and subsamples at most **12000** voxels for the fit.

### VENC resolution

Order: JSON in `4DFlow/AP/*.json` (`VelocityEncoding` / `PhaseEncodingVelocity` ×10 to mm/s, or `VENC` with heuristic cm/s→mm/s), then NIfTI sidecar metadata on phase/mag, then optional DICOM directory `(0018,9217)` / GE private tags, then default **700** mm/s with a **warning** (MATLAB historically hard-coded 700 when tags were absent).

### Code references (snippets)

```python
# ──────────────────────────────────────────────────────────────────────────────
# phase2volume._calc_angio — CD magnitude
# ──────────────────────────────────────────────────────────────────────────────
vm = np.clip(as_backend_array(v_mag).astype(np.float64), 0.0, float(venc))
return as_backend_array(angio_mag).astype(np.float64) * np.sin((np.pi / 2.0 * vm) / float(venc))
```

```python
# ──────────────────────────────────────────────────────────────────────────────
# phase2volume.compute_phase_derivatives — velocity mm/s
# ──────────────────────────────────────────────────────────────────────────────
vx = -rl_phase * 10.0
vy = -ap_phase * 10.0
vz = fh_phase * 10.0
```

```python
# ──────────────────────────────────────────────────────────────────────────────
# _phase2volume_bg.fit_polynomial_background_3vector (after setup(globals()))
# ──────────────────────────────────────────────────────────────────────────────
cx, *_ = np.linalg.lstsq(A, bvx, rcond=None)
```

## Results layout

Under `--output-root/<subject>/`:

- `eicab/` — stage 1 eICAB outputs (multilabel in TOF space, plus ``TOF_resampled.nii.gz`` used as the FLIRT moving image in stage 2).
- `qvtpy/stage2_registration/` — FLIRT `tof_to_4dflow.mat`, `TOF_warped_to_4dflow_ref.nii.gz`, `registration_meta.json`.
- `qvtpy/stage3_centerline/` — `eicab_in_4dflow.nii.gz`, `centerlines_mask.nii.gz`, `cd_vessel_binary_qc.nii.gz`, `centerline_meta.json`.
- `qvtpy/stage4_4dflow_segmentation/` — `seg_4dflow.nii.gz`, `segmentation_meta.json`.
- `qvtpy/stage5_loc_generation/` — `locs.csv`, `loc_meta.json`.
- `qvtpy/stage6_measure/` — `loc_measurements.csv`, `measure_meta.json`.

Label constants (see `src/nvitk/pipes/qvtpy/labels.py`): eICAB vessel integers `1–18` (see `EICAB_ID_TO_NAME`); qvtpy extensions `QVTPY_VENOUS_UNKNOWN_LABEL` (30), venous region ids **31–34** (`VENOUS_REGION_BASE` + geometry assignment in stage 3), `QVTPY_UNKNOWN_LABEL` (35). Venous branch names SSSV/LTSV/RTSV/STRV are assigned in stage 3 by skeleton geometry.

### qvtpy CLI flags (stages 3–6, `nvitk-qvtpy`)

| Flag | Stage | Default |
|------|-------|---------|
| `--eicab-mask {cw,wb}` | 3 | `cw` (warn + use alternate if missing) |
| `--cd-up-thresh`, `--cd-shift-hm` / `--no-cd-shift-hm` | 3 | auto / on |
| `--venous-min-component-frac`, `--eicab-min-island-fraction`, `--eicab-bridge-open-radius` | 3 | 0.005 / 0.005 / 1 |
| `--venous-min-branch-points` | 3 | 12 |
| `--seg-assembly {voxel,mesh}` | 4 | `voxel` |
| `--seg-interp-level`, `--seg-stride` | 4 | 0 / 1 |
| `--cross-section-res`, `--cross-section-plane-interp` | 4, 6 | 0 / 1 |
| `--loc-arterial-strategy {qvtplus,midpoint}` | 5 | `qvtplus` |
| `--cross-section-radius-vox` | 5, 6 | 10 |
| `--measure-resegment` / `--no-measure-resegment` | 6 | resegment on |

## Stages

### Stage 0 (download / convert)

DICOM acquisition layout, NIfTI conversion, reorganization, optional `phase2volume` (see flags on `qvtpy-stage0` / `nvitk-qvtpy`: `--phase-background-correction`, `--phase-bg-poly-order`, …).

CLI examples:

- `phase2volume -i <patient_dir> [--dicom-dir <subject_dicom>] [--no-background-phase-correction] …`
- `qvtpy-stage0 --subject … --compute-phase-derived …`

### Stage 1 (eICAB)

TOF-based multilabel segmentation; outputs used as moving labels for stage 3.

### Stage 2 — registration (NiPype + FSL FLIRT)

- **Moving:** eICAB ``TOF_resampled.nii.gz`` (or ``TOF_resampled.nii``) under ``--output-root/<subject>/<eicab>/`` (same folder as multilabel masks). Stage 1 must have run first. Override directory name with ``--eicab-subdir`` on ``qvtpy-stage2-registration`` if needed.
- **Fixed:** `4DFlow/Angiography_3D.nii.gz` (default) or `ComplexDifference_3D.nii.gz` (`--stage2-reference cd` on `nvitk-qvtpy`).
- **`--stage2-dof`:** FLIRT degrees of freedom (default **6** = rigid).
- **`--stage2-cost`:** FLIRT cost metric (default **`normmi`**).
- **Implementation:** `nvitk.registration.fsl.flirt` → `nipype.interfaces.fsl.FLIRT`. Optional `searchr_x` exists in the FLIRT wrapper but is **not** exposed on qvtpy CLI yet (FSL default search).
- **Runtime:** FSL binaries and `FSLDIR` on `PATH` in the Python environment (e.g. Singularity on SGE).

### Stage 3 — centerline

Resolve eICAB with ``--eicab-mask`` (`cw` or `wb`); if the requested mask is missing and the other exists, log a warning and continue (recorded in ``centerline_meta.json``). Warp labels to 4D-flow space (nearest neighbour). **Cleaning:** per-label island removal (default 0.5% of label foreground), optional per-label binary opening (`--eicab-bridge-open-radius`) to suppress thin communicating-artery bridges. **CD binary QC:** ``cd_vessel_binary_qc.nii.gz`` from global sliding-threshold on ``ComplexDifference_3D`` (tunable via ``--cd-up-thresh``, ``--cd-shift-hm``). **Arterial:** labels cleared in venous slab → per-label centerlines (`compute_centerlines`, ``min_points=5``). **Venous:** mask = CD binary ∧ superior Y-slab → area opening → skeleton branches scored by position/orientation and assigned to **0–4** of SSSV/STRV/LTSV/RTSV. **IO:** ordered polylines are stored only in ``centerlines_mask.nii.gz`` (multilabel raster); stages 4–5 reload via ``util.centerline_io`` (no NPZ).

### Stage 4 — segmentation

**Centerline-backbone** ``seg_4dflow`` (does not paste raw eICAB). For each arterial/venous polyline loaded from ``centerlines_mask.nii.gz``, sample oblique cross-sections (default radius 10 vox), in-plane segmentation (fused MAG/CD/|V|, sliding threshold, 5% area open, central CC), then assemble into 3D:

- ``--seg-assembly voxel`` (default): stamp masks along the centerline; optional between-plane blend via ``--seg-interp-level``.
- ``--seg-assembly mesh``: sparse voxel stamp + distance-transform fill (~radius).

Post-process with per-label 0.5% island removal. Venous vessels use stage-3 geometry-assigned centerlines and label ids.

### Stage 5 — LOC generation

QVTplus-style heuristics (`--loc-arterial-strategy qvtplus`, default): ICA/BA near common Z with circularity; main arteries at masked midpoint; venous SSSV/STRV/LTSV/RTSV with segment disambiguation (6-part Z-std, SVD alignment vs ``[0,1,1]``, swap validation). ``locs.csv`` includes ``segment_id``, ``centerline_index``, ``loc_circularity``, ``loc_cross_section_area_mm2``.

### Stage 6 — measurements

Per LOC: in-plane segmentation at the LOC station → **per-vessel** ``loc_cross_section_area_mm2``; masked mean through-plane velocity on the plane per cardiac frame; flow \(Q = \bar{v}_\mathrm{plane} \cdot A / 1000\) ml/s; PI/RI on the flow time series. ``--no-measure-resegment`` can reuse area from stage 5 CSV when present.

## Cluster (SGE)

`nvitk-qvtpy --submit sge` emits one bash script per run: per subject, stages run in pipeline order with `-hold_jid` chaining. Stage 0 download remains local-only.

Paths inside the container: `--nifti-root` → `/nvitk/data`, `--output-root` → `/nvitk/output` (see `SingularityBinds`).

## Optional pip extra

`pip install "nvitk[fsl]"` installs the `fsl` extra (NiPype); **FSL itself** remains a system/container install.

## Integration tests with real FSL

Set `NVITK_FSL_TESTS=1` and ensure `flirt` is on `PATH` to enable gated FLIRT tests (not required for default CI).
