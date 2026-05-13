# qvtpy methodology

Pipeline code: `src/nvitk/pipes/qvtpy/`. This document tracks **what each stage does**, **inputs/outputs**, **runtime dependencies**, and **compute backend** (CPU vs GPU-capable array code). Update the changelog when behavior or defaults change in a material way.

## Changelog

| Date | Summary |
|------|---------|
| 2026-05-12 | Initial stages 2–6, NiPype FLIRT registration module, `measure.hemodynamics` helpers, SGE `hold_jid` chain in `run.py`. |
| 2026-05-12 | `pipes/qvtpy/labels.py`: full eICAB ID table (1–18 plus basilar / AComm), canonical names (LICA, RICA, BASILAR, …), QVT venous name strings (SSSV, LTSV, RTSV, STRV), and qvtpy extension label ints (`QVTPY_VENOUS_UNKNOWN_LABEL`, …). |
| 2026-05-12 | `phase2volume`: single QVTplus-aligned derivative path; VENC from JSON → NIfTI metadata → DICOM → 700 mm/s + warning; default polynomial background correction (`fit_order` default 2). |
| 2026-05-12 | qvtpy stages / `phase2volume` / `hemodynamics` / `_phase2volume_bg`: use `nvitk.core.backend.setup(globals())` and `as_backend_array` so heavy array ops follow `NVITK_BACKEND` (NumPy or CuPy). Venous `skeletonize` (skimage) stays CPU via `to_numpy` before skimage. |
| 2026-05-12 | Stage 2: FLIRT **moving** image is eICAB ``TOF_resampled`` under ``--output-root/<subject>/eicab/`` (not raw ``TOF/TOF.nii.gz``). ``registration_meta.json`` records ``moving_kind: eicab_tof_resampled``. Optional CLI ``--eicab-subdir`` on ``qvtpy-stage2-registration``. |
| 2026-05-12 | Stage 3: global binary mask on ``ComplexDifference_3D`` via sliding-threshold curve (median 3³, step 0.001 up to 0.8×max, FWHM-shifted curvature pick, smooth width 10) + 0.5% foreground area filter; venous slab = first third along axis 1; venous mask = global mask ∧ slab + second 0.5% filter; skeleton venous centerlines. |

## Backend (GPU vs CPU)

nvitk selects the array stack via `NVITK_BACKEND` / `nvitk.using("cupy")` (see [`src/nvitk/core/backend.py`](../../src/nvitk/core/backend.py)). qvtpy modules that touch large volumes call `setup(globals())` and coerce voxel data with `as_backend_array` instead of `numpy.asarray`, matching patterns in [`src/nvitk/morphology/centerline.py`](../../src/nvitk/morphology/centerline.py) and [`src/nvitk/transform/oblique.py`](../../src/nvitk/transform/oblique.py).

| Stage / component | GPU-friendly (CuPy when backend=cupy) | CPU-only or host-bound notes |
|-------------------|----------------------------------------|------------------------------|
| Stage 0 convert + `phase2volume` | Yes: CD/MAG/velocity math, polynomial BG fit (`np.linalg.lstsq`, `ndi` not required there) | DICOM/NIfTI I/O ends on CPU for writers; `imsave` materializes NumPy for NIfTI. |
| Stage 1 eICAB | External container (separate from this table) | Not governed by nvitk `np` proxy. |
| Stage 2 FLIRT | No | FSL `flirt` is a CPU binary; moving image is read from stage 1 eICAB outputs on the host. |
| Stage 3 centerlines | Partial: multilabel arterial on GPU where applicable | Global ``slidingThreshold`` + CC filtering and venous ``bwareaopen`` on CPU NumPy; ``scipy.ndimage.median_filter``; ``skimage`` ``remove_small_objects`` / ``skeletonize`` on CPU. |
| Stage 4 segmentation | Yes: percentile, `np.where`, integer labels | Writes NIfTI on CPU. |
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
# phase2volume._calc_angio — CD magnitude
vm = np.clip(as_backend_array(v_mag).astype(np.float64), 0.0, float(venc))
return as_backend_array(angio_mag).astype(np.float64) * np.sin((np.pi / 2.0 * vm) / float(venc))
```

```python
# phase2volume.compute_phase_derivatives — velocity mm/s
vx = -rl_phase * 10.0
vy = -ap_phase * 10.0
vz = fh_phase * 10.0
```

```python
# _phase2volume_bg.fit_polynomial_background_3vector (after setup(globals()))
cx, *_ = np.linalg.lstsq(A, bvx, rcond=None)
```

## Results layout

Under `--output-root/<subject>/`:

- `eicab/` — stage 1 eICAB outputs (multilabel in TOF space, plus ``TOF_resampled.nii.gz`` used as the FLIRT moving image in stage 2).
- `qvtpy/stage2_registration/` — FLIRT `tof_to_4dflow.mat`, `TOF_warped_to_4dflow_ref.nii.gz`, `registration_meta.json`.
- `qvtpy/stage3_centerline/` — `eicab_in_4dflow.nii.gz`, `centerlines.npz`, `centerlines_mask.nii.gz`, `centerline_meta.json`.
- `qvtpy/stage4_4dflow_segmentation/` — `seg_4dflow.nii.gz`, `segmentation_meta.json`.
- `qvtpy/stage5_loc_generation/` — `locs.csv`, `loc_meta.json`.
- `qvtpy/stage6_measure/` — `loc_measurements.csv`, `measure_meta.json`.

Label constants (see `src/nvitk/pipes/qvtpy/labels.py`): eICAB vessel integers `1–18` (see `EICAB_ID_TO_NAME`); qvtpy extensions `QVTPY_VENOUS_UNKNOWN_LABEL` (30), optional venous region base 31, `QVTPY_UNKNOWN_LABEL` (35). Venous branch *names* SSSV/LTSV/RTSV/STRV are strings for LOC, not eICAB voxel values.

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

Apply the stage-2 rigid transform (nearest neighbour) to warp eICAB multilabels into 4D flow space. **Global vessel binary mask** on ``ComplexDifference_3D``: sliding-threshold selection (3³ median pre-filter, normalized threshold axis 0–0.8 in steps of 0.001, smoothed occupancy vs threshold, curvature maximum with optional FWHM-based shift along the axis, mean smooth width 10) followed by removing connected components smaller than 0.5% of foreground voxels (face-adjacent 3D connectivity). **Venous slab** = first ``round(ny/3)`` planes along axis 1; **venous mask** = global mask restricted to that slab, then the same 0.5% component filter. **Arterial multilabel** = warped labels cleared inside the venous slab and 0.5% area opening on the arterial binary hull; per-label centerlines via ``compute_centerlines`` (``min_points=5``). Venous centerline points from the skeleton of the cleaned venous mask. ``centerlines_mask.nii.gz`` stores eICAB ids on arterial centerline voxels and ``VENOUS_UNKNOWN_LABEL`` on venous skeleton voxels.

### Stage 4 — segmentation

Heuristic multilabel from CD percentile and warped eICAB labels.

### Stage 5 — LOC generation

Per arterial label: midpoint voxel and tangent (`centerline_tangents`).

### Stage 6 — measurements

Through-plane velocity time series and PI/RI via `nvitk.measure.hemodynamics`.

## Cluster (SGE)

`nvitk-qvtpy --submit sge` emits one bash script per run: per subject, stages run in pipeline order with `-hold_jid` chaining. Stage 0 download remains local-only.

Paths inside the container: `--nifti-root` → `/nvitk/data`, `--output-root` → `/nvitk/output` (see `SingularityBinds`).

## Optional pip extra

`pip install "nvitk[fsl]"` installs the `fsl` extra (NiPype); **FSL itself** remains a system/container install.

## Integration tests with real FSL

Set `NVITK_FSL_TESTS=1` and ensure `flirt` is on `PATH` to enable gated FLIRT tests (not required for default CI).
