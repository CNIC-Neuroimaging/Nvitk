# Methodological audit

Scope: the imaging and mesh code of nvitk (`src/nvitk`, the Slicer modules and `scripts/`),
at commit `ec706c1`. The vendored nnU-Net / nnssl engines under `pipes/_engines/` were not
reviewed, apart from the nvitk-written trainers in them.

Nothing in this report has been fixed in the code. Every fix below changes numerical outputs,
so each one needs a decision (and, for published cohorts, a re-run plan). The findings marked
**verified** were reproduced on synthetic data; the scripts are in [`scripts/`](scripts/) and
print the numbers quoted here:

```bash
cd audit/scripts
python check_background_polyfit.py   # M-01
python check_cross_section_area.py   # M-03, M-04
python check_hd95.py                 # M-09
python check_nifti_header.py         # M-10
python check_mha_geometry.py         # M-11
python check_image_slicing.py        # M-12
python check_dicom.py                # M-13, M-15
python check_compare.py              # M-20
```

They import single modules from `src/` (see `nvload.py`), so they run with NumPy, SciPy,
nibabel, SimpleITK, pydicom, dicom2nifti and scikit-image; the full nvitk stack is not needed.

Severity scale: **critical** (wrong numbers in a default pipeline path), **high** (wrong numbers
or wrong geometry under common conditions, or a data-governance problem), **medium** (wrong
under specific conditions, or a method choice with a known better alternative), **low**
(robustness, clarity, minor bias).

---

## 1. Summary

| ID | Severity | Area | Finding | Evidence |
|---|---|---|---|---|
| M-01 | critical | 4D flow | Background-phase polynomial is fitted in scrambled coordinates on non-cubic grids | verified |
| M-02 | high | 4D flow | Stage-6 flow uses uncorrected phase; the background correction never reaches the flow numbers | code path |
| M-03 | high | 4D flow | Cross-section area is overestimated by 7.5-15 % with supersampling (qvtpy default) | verified |
| M-04 | high | 4D flow | Cross-section pixel size is a heuristic; +61 % area for an in-plane vessel at 0.5 x 0.5 x 0.8 mm | verified |
| M-05 | high | 4D flow | Velocity components are assumed aligned with the voxel axes (no direction cosines) | code path |
| M-06 | high | 4D flow | Mean flow and PI are computed on per-frame absolute values (noise rectification) | code path |
| M-07 | medium | 4D flow | VENC unit is guessed from its magnitude (< 150 means cm/s) | code path |
| M-08 | medium | 4D flow | Phase is converted as Philips cm/s (x10); VENC never scales phase | code path |
| M-09 | high | metrics | `segmentation_metrics.hd95` uses all foreground voxels, not surfaces | verified |
| M-10 | high | I/O | NIfTI sform and pixdim come from two independent sources; ITK and nibabel can read different spacing | verified |
| M-11 | high | I/O | MetaImage writer prefers stale ITK origin/direction over the updated affine | verified |
| M-12 | high | core | Indexing an `Image` keeps the parent affine; crops are saved misregistered | verified |
| M-13 | high | governance | Patient identifiers from DICOM are written into every derived NIfTI | verified |
| M-14 | high | DICOM | The orientation "fix" merges slices of different planes with an in-plane rotation | code path |
| M-15 | high | DICOM | Per-slice rescale factors are applied in InstanceNumber order to spatially sorted slices | verified |
| M-16 | high | PET | SUV ignores `DecayCorrection` and `Units` | code path |
| M-17 | high | TopBrain | `nvitk-topbrain-infer` without `--submit` raises `TypeError` | code path |
| M-18 | medium | PET | SUV timing and LBM/IBW formulas (SeriesTime, injection date, James LBM) | code path |
| M-19 | medium | metrics | Two inconsistent metric families (Dice of empty masks, HD95, missing structures) | code path |
| M-20 | medium | metrics | `compare.sample_at_physical_points` always raises | verified |
| M-21 | medium | DICOM | Gantry tilt is not resampled; tilted stacks get a sheared affine | code path |
| M-22 | medium | transform | `isotropy()` leaves anisotropic grids and drifts the spacing | code path |
| M-23 | medium | filters | Hessian, Frangi and bilateral filters work in voxels and ignore spacing | code path |
| M-24 | medium | filters | Bilateral "3-D" gives different results on CPU and GPU | code path |
| M-25 | medium | mesh | Marching cubes without padding; binary staircase; per-label meshes with gaps | code path |
| M-26 | medium | registration | ANTs runs are not reproducible by default; CLIs interpolate labels linearly | code path |
| M-27 | medium | core | Backend switching rebinds module globals for all threads | code path |
| M-28 | medium | stats | Convergence warnings are silenced; bootstrap keeps non-converged fits | code path |
| M-29 | medium | segmentation | Midline L/R split assumes voxel axis 0 is left-right | code path |
| M-30 | medium | PESA-Fat | Unit-dependent PET threshold floor; erosions counted in voxels | code path |

Lower-severity items are in section 4, sound choices worth keeping in section 5, and
state-of-the-art options in section 6.

---

## 2. Critical and high findings

### M-01 (critical, verified). Background-phase fit uses scrambled coordinates

`io/conversors/_phase2volume_bg.py:46` (`fit_polynomial_background_3vector`) and `:97`
(`fit_polynomial_background_scalar`) build the design matrix with

```python
zz, yy, xx = np.meshgrid(iz, iy, ix, indexing="ij")      # shape (nz, ny, nx), x varies fastest
coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
...
bvx = vx_mean.ravel()[idx]                                # shape (nx, ny, nz), z varies fastest
```

so row *r* of the design matrix describes a different voxel than value *r* of the data unless
`nx == ny == nz`. The polynomial is then fitted in a coordinate system that is a permutation of
the true one for cubic grids and a non-smooth scramble otherwise.

Measured on a quadratic background of RMS 3.5 mm/s plus noise:

| grid | residual after the phase2volume fit | residual after `polynomial_background` (same file) |
|---|---|---|
| 40 x 32 x 24 | 1.54 mm/s | 0.014 mm/s |
| 32 x 32 x 32 | 0.013 mm/s | 0.013 mm/s |

On a typical non-cubic 4D-flow matrix the "correction" removes only part of the offset and adds a
structured error field. Both the velocity outputs and `ComplexDifference_4D` (via
`subtract_polynomial_background_4d_scalar`) go through these functions, and
`phase2volume(..., background_phase_correction=True)` is the default called by qvtpy stage 0
(`pipes/qvtpy/stage0_convert.py:291`).

**Fix.** Replace both functions with `polynomial_background` (already in the same module, correct
indexing, no full-size design matrix). Re-derive `VelocityMagnitude_*`, `VelocityMeanComponents`
and `ComplexDifference_*` for existing subjects, then re-run what depends on CD (stage 4
segmentation, stage 5 LOCs).

**Method note.** The background voxels are the lowest 25 % of mean speed, which includes air and
other noise-only voxels whose phase is random. The usual practice is static tissue chosen by low
temporal standard deviation inside a magnitude/tissue mask (Walker 1993; Lankhaar 2005; Bissell
2023 consensus). The MATLAB QVT code excluded noise voxels with magnitude and CD thresholds; the
docstring says this was not ported.

### M-02 (high). Flow is measured on uncorrected phase

`pipes/qvtpy/stage6_measure.py:517` reads the AP/RL/FH phase NIfTIs and converts them with
`velocity_mm_s_from_phases`; every LOC and dense-profile flow is computed from those arrays. The
background correction of stage 0 only reaches the derived `VelocityMagnitude` and CD files. So
whatever eddy-current offset is present stays in every flow, PI, RI and PWV.

**Fix.** Fit the background once per subject (M-01 fixed) and subtract it from the three
components before stage 6, or store corrected components in stage 0 and read those in stage 6.
Report which one was used in the measurement provenance.

### M-03 (high, verified). Area bias from the supersampled cross-section grid

`transform/oblique.py:44` samples the plane at `np.linspace(-r, r, n)`, a step of `2r/(n-1)`
voxels. `measure/cross_section.py:397` converts pixel counts with a step of `(2r+1)/n`. The two
agree only when `n = 2r + 1` (no supersampling). With supersampling, `n = 2 r iv + 1`, so every
area is multiplied by `((2r+1) iv / (2 r iv + 1))^2`: +7.5 % at `r = 10, iv = 4` and +15 % at
`r = 5`. `cs_supersampling=True` is the default in stage 6, LOC measurement and the dense profile.

Synthetic cylinder, radius 3 mm, isotropic 0.5 mm: true area 28.27 mm2; nearest grid 28.25 mm2
(-0.1 %); supersampled grid 31.26 mm2 (+10.5 %). Flow (`v * A`) inherits the bias, and because it
depends on `radius_vox` it differs between vessels measured with different radii.

**Fix.** Use the actual sampling step: `step = 2 r / (n - 1)` voxels (or build the grid from the
step and derive `n`). Re-run stage 6.

### M-04 (high, verified). Heuristic pixel size for oblique planes

`tilt_corrected_spacing_mm` (`measure/cross_section.py:67`) sets the in-plane pixel size to
`min(s) + (s_z - min(s)) * sin(angle between tangent and z)` for both plane axes. The exact size
of one plane step along the in-plane unit vectors `u`, `v` (voxel space) is `||u * s||` and
`||v * s||`, and the pixel area is `||(u * s) x (v * s)||`.

Synthetic cylinder along x at 0.5 x 0.5 x 0.8 mm: area 45.44 mm2 for a true 28.27 mm2
(+60.7 %; +80.7 % with supersampling). Isotropic data are unaffected. A second, related problem:
tangents, the plane basis and the plane orientation are all computed in voxel index space, so for
anisotropic voxels the plane is not perpendicular to the vessel in millimetres, which adds a
`1 / cos` area error and mixes in-plane velocity into the through-plane component.

**Fix.** Do the geometry in millimetres: tangent and plane basis from `affine[:3, :3] @ t`, sample
with physical step sizes, and compute the pixel area from the mapped basis vectors. Check which
cohorts were acquired anisotropically before deciding on re-runs.

### M-05 (high, conditional). Velocity components assumed axis-aligned

`velocity_mm_s_from_phases` (`measure/hemodynamics.py`) and `compute_phase_derivatives`
(`io/conversors/phase2volume.py:373`) map `vx = -RL`, `vy = -AP`, `vz = FH` onto array axes 0,
1, 2, and the through-plane velocity is the dot product with a voxel-space tangent. Philips
encodes RL/AP/FH in the patient frame. This holds only for a non-angulated axial slab stored in
the expected axis order and sign. A sagittal, coronal or angulated slab, or a different slice
sort, permutes or flips components and the dot product mixes the wrong ones. Velocity magnitude
is unaffected; flow, PI and PWV are.

**Fix.** Rotate the patient-frame velocity vector into voxel axes with the direction cosines of
the affine (accounting for the RAS/LPS convention of the NIfTI), and assert the expected
orientation codes when the data are read. A one-off check on a phantom or on a subject with
known flow direction in the basilar artery confirms the sign convention.

### M-06 (high). Mean flow and pulsatility from absolute values

`pipes/qvtpy/util/loc/loc_measure.py:85-86` takes `np.abs` of the velocity of every frame, then
of the flow; `mean_flow_ml_min` and `pulsatility_index_qvt` (`measure/hemodynamics.py:47`, `:60`)
also average `|Q(t)|`. The README states the intent: a flipped tangent must not invert the
sign. Two biases follow:

* **Noise rectification.** For a frame mean with noise SD `σ`, `E|v| >= |E v|`; at zero net flow
  the reported mean velocity is `σ * sqrt(2/π) ≈ 0.8 σ` instead of 0. Small and noisy vessels
  (PComm, distal branches, sinuses) are biased upwards, and differences between groups with
  different SNR become partly a noise effect.
* **Retrograde phases are folded.** Bidirectional waveforms get a lower PI and a higher mean.

**Fix.** Choose one sign per station: the sign of the mean through-plane flow (or the expected
anatomical direction), multiply the whole waveform by it, then compute mean, PI and RI on the
signed series. Keep `|mean|` if a magnitude is needed for reporting.

### M-09 (high, verified). HD95 is not the surface HD95

`measure/segmentation_metrics.py:167` computes distances from every foreground voxel of one mask
to the other mask. Voxels inside the overlap contribute zeros, which drags the 95th percentile
down. Benchmarks (medpy, MONAI, the TopCoW/TopBrain evaluation) use boundary voxels.

| sphere r = 20, shifted by | nvitk `hd95` | surface-based HD95 |
|---|---|---|
| 1 voxel | 0.00 | 1.00 |
| 3 voxels | 1.41 | 3.00 |
| 6 voxels | 3.61 | 5.74 |

`measure/surface.py:hausdorff95` already does it the standard way.

**Fix.** Compute `hd95` on boundaries (`mask ^ binary_erosion(mask)`) or call
`surface.hausdorff95`; re-score the TopBrain cross-validation before comparing with leaderboard
numbers.

### M-10 (high, verified). NIfTI header with two sources of geometry

`io/writers/nifti.py:69-115` writes the sform from `metadata["affine"]` (identity when absent)
and pixdim from `x_res`/`y_res`/`z_res`. Nothing checks that they agree. When they do not, the
file has two geometries: nibabel and FSL follow the sform, ITK-based tools (SimpleITK, ANTs,
ITK-SNAP, Slicer) fall back to pixdim:

```
affine diag(2, 2, 3), x_res = 0.5  ->  nibabel spacing (2, 2, 3), SimpleITK spacing (0.5, 0.5, 0.5)
```

It happens whenever an operation updates one of the two (see M-11, M-12, M-22) or when an
array without an affine but with `x_res` is saved (identity sform, 1 mm). The header also gets
`qform_code = 0`, `sform_code = 2` and `xyzt_units = 0` on every write, so source codes and units
(mm, seconds vs milliseconds for `t_res`) are lost.

**Fix.** Treat the affine as the single source of truth: derive zooms from its column norms (and
build an affine from the zooms when there is none), write the source qform/sform codes, and set
`xyzt_units` to mm and s. `Image.spacing` can then be a read-only view of the affine.

### M-11 (high, verified). MetaImage writer uses stale ITK geometry

`io/readers/mha.py` stores ITK `origin`/`direction` (LPS) next to the RAS `affine`.
`io/writers/mha.py:56-60` prefers the ITK keys, but `orient_to`, `resample_to` and `isotropy`
update only the affine. Read a `.mha`, reorient it, write it back: a marked voxel moves from RAS
(-13, -22, 32) to (-10, -23, 32) mm, with its axes mirrored. The NIfTI round trip is correct. The
TopBrain Grand Challenge container reads and writes `.mha`.

**Fix.** Derive origin/direction from the affine on write (drop the stored ITK keys after any
geometry change, or never store them).

### M-12 (high, verified). `Image.__getitem__` keeps the parent affine

`types/image.py:323` (`_clone`) and `:389` (`__getitem__`) carry the metadata, affine included,
into the indexed result. `img[10:20, 5:15, 4:10]` moves a voxel at (6, 3.5, 10) mm to
(1, 1, 2) mm, and the error survives `imsave`/`imread`. Strided slices also keep the old spacing,
and integer indexing keeps a 4 x 4 affine for a 2-D result.

**Fix.** In `__getitem__`, translate the affine origin by the slice start, scale columns by the
step, drop columns of removed axes (or drop the affine when it cannot be expressed).

### M-13 (high, verified). Patient identifiers in derived files

`load_dicom_series` puts the whole first dataset of a series into the metadata, and
`write_nifti` serialises all metadata into a header extension and into the JSON sidecar.
`imread(DICOM)` then `imsave(".nii.gz")` produces a NIfTI with `PatientName`, `PatientID`,
`PatientBirthDate`, `AccessionNumber` and `StudyDate` in its header (also as tag-number
duplicates). `METADATA_TO_SAVE` (`_dicom_conversion.py`) adds more of the same to `dcm2nii`
output.

Derived NIfTIs, masks and meshes are what gets shared with collaborators, uploaded to XNAT
resources or packaged for challenges, so this is a GDPR exposure rather than a style issue.

**Fix.** Default to an allow-list of acquisition parameters (spacing, timing, scaling, PET dose
and decay fields, VENC, modality) and drop identifiers; keep a pseudonymised subject key only
where the pipeline needs one. Scan existing derived trees for the identifier keys.

### M-14 (high). The orientation "fix" fabricates volumes

On `IMAGE_ORIENTATION_INCONSISTENT`, `_fix_orientation_inconsistencies`
(`io/conversors/_dicom_conversion.py:537`) clusters orientations, then gives every outlier slice
the reference orientation:

* the pixel data are rotated in-plane using only the x and y components of the direction
  cosines (`_reorient_pixel_array`). A slice from a different plane cannot be turned into the
  reference plane by an in-plane rotation, so the result is a stack of unrelated images;
* positions are projected onto the reference normal;
* rotated arrays (float64 after `map_coordinates`) are written into `PixelData` without updating
  `Rows`, `Columns`, `BitsAllocated` or the transfer syntax;
* the clustering passes a square distance matrix to `linkage(..., method="ward")`, which expects
  a condensed matrix or observations. SciPy warns about this and the warning is silenced at the
  top of the module (`:84-87`).

The conversion log says "orientation fixing applied" and a plausible-looking volume comes out.

**Fix.** Never merge slices of different orientation. Split them as `split_by_geometry` already
does for size and spacing (and as dcm2niix does), and report the split.

### M-15 (high, verified). Per-slice rescale order

`_read_dicom_conversion` sorts each series by `InstanceNumber` (`:1472`) and collects
`RescaleSlope`/`RescaleIntercept`/`ScaleSlope` in that order; dicom2nifti stacks slices by
position. `_apply_rescale_to_nifti` and `_apply_fp_rescale_to_nifti` then apply factor *k* to
NIfTI slice *k*. With slopes 1..6 and InstanceNumber running against position, `revert_scaling`
returns `[16.7, 40, 75, 133.3, 250, 600]` instead of 100 on every slice; with InstanceNumber
along the position it returns 100. PET almost always has per-slice slopes; Philips MR sometimes
does. The 4-D collapse sorts by `SliceLocation`, whose sign convention is vendor dependent.

**Fix.** Collect the factors after sorting slices the way dicom2nifti does
(`_spatial_sort_ds_list` already reproduces that order).

### M-16 (high). SUV ignores decay-correction mode and units

`measure/suv.py:155` always decays the injected dose from injection to `SeriesTime`.
`DECAY_CORRECTION_TAGS` and `UNITS_TAGS` are defined (`:43`, `:50`) and never read.

* `DecayCorrection = ADMIN`: images are already decay corrected to injection; the extra factor
  inflates SUV by `2^(dt/T½)`, for example +46 % at 60 min with FDG.
* `DecayCorrection = NONE` or dynamic frames: frame-specific decay is needed.
* `Units` other than `BQML` (for example `GML`, already SUV, or Philips `CNTS`) are scaled as if
  they were Bq/mL.

**Fix.** Follow the QIBA vendor-neutral SUV pseudo-code: check `Units`, switch on
`DecayCorrection`, and refuse to compute when the combination is not supported.

### M-17 (high). TopBrain local inference crashes

`pipes/topbrain/stage4_infer.py:1333` calls `run_infer(flood_hyst_low_factor=...,
flood_hyst_high_factor=..., flood_thin_percentile=..., flood_thicken_iter=...)`, but `run_infer`
(`:549`) has no such parameters and uses the names undefined at `:743-746`. Every local run of
`nvitk-topbrain-infer` raises `TypeError` before predicting. Found by `ruff` (F821).

**Fix.** Add the four keyword parameters to `run_infer` and pass them through.

---

## 3. Medium findings

**M-07. VENC units guessed from the value.** `_normalize_venc_to_mm_s`
(`io/conversors/phase2volume.py:133`) multiplies values below 150 by 10. DICOM
`(0018,9217)` and Philips `(2001,101A)` are defined in cm/s, so a VENC of 150-250 cm/s (neck,
aorta) is read as mm/s, a factor of 10 off. Within nvitk VENC only drives the CD contrast, which
feeds segmentation. Use the unit that belongs to each source instead of a threshold.

**M-08. Phase scaling is Philips-specific.** `vx = -rl_phase * 10` assumes the phase NIfTI holds
velocity in cm/s (Philips after rescale). Siemens and GE store phase on a scanner-specific
integer range that has to be mapped to ±VENC. Nothing checks the vendor. Add a vendor check
or an explicit scaling step. Related: no phase unwrapping is done, so aliased voxels stay wrong
(4D Laplacian unwrapping, Loecher 2016); magnitude and phase frame counts that differ are
silently truncated (`phase2volume.py:405`).

**M-18. SUV details.**
* `SeriesTime` is used as scan start. QIBA uses it only when it is not later than the earliest
  `AcquisitionTime`; re-saved or reprocessed series change `SeriesTime`. GE stores the true start
  in `(0009,100D)`.
* `RadiopharmaceuticalStartDateTime (0018,1078)` is not used, so an injection before midnight
  with a scan after midnight gives a negative interval and an error.
* SUVlbm uses the James formula, which gives implausible (even negative) lean mass at high BMI.
  PERCIST and EANM 2015 recommend Janmahasatian 2005 (SUL).
* SUVibw mixes Hamwi (men) and Devine (women) and caps IBW at body weight.
* `revert_scaling=True` turns Bq/mL back into stored integers before the SUV factor is applied,
  which is not an activity concentration.
* No SUVpeak (PERCIST 1 cm3 sphere), which is less noise-dependent than SUVmax.

**M-19. Two metric families disagree.** `segmentation_metrics.dice` returns 1.0 for two empty
masks, `voxel.dice` returns 0.0. `segmentation_metrics.hd95` (all voxels, 290 mm penalty for a
missing structure) differs from `surface.hausdorff95` (surface voxels, and `_distances` returns
`[0.0]`, so HD = 0, when one mask is empty: a missed structure scores perfectly).
`voxel.volume_similarity` is a relative volume difference, not volume similarity. Keep one
implementation per metric, with the Metrics Reloaded conventions (Maier-Hein 2024) for empty
references and predictions.

**M-20 (verified). `compare.sample_at_physical_points` always raises.** `pts_hom =
as_backend_array([pts.T, np.ones((1, n))])` (`measure/compare.py:154`) builds a ragged array;
NumPy refuses it. Use `np.vstack`. In the same module, `percent_diff` divides by intensities that
can be zero or negative (CT, PET background), and voxelwise Pearson/Spearman p-values treat
spatially correlated voxels as independent samples. Bland-Altman bias and limits of agreement on
ROI means are the usual summary.

**M-21. Gantry tilt.** The retry with `reorient_nifti=True` does not enable dicom2nifti's
resampling (`dicom2nifti.settings.enable_resampling()`), so tilted CT falls through to the basic
path, whose affine is sheared. ITK cannot represent shear, so nibabel and ITK place the volume
differently (see M-10). Resample tilted stacks explicitly, as dcm2niix does.

**M-22. `isotropy()` is not isotropic.** `transform/isotropy.py:25` zooms only the coarsest axis
by `max/min`; spacing (0.5, 0.6, 1.0) becomes (0.5, 0.6, 0.5). `ndi.zoom` (with `grid_mode=False`)
maps corner voxel centres, so the real spacing is `s (n-1)/(n'-1)`, not the stored `s / factor`:
up to half an original voxel of drift at the far edge. Resample all three axes onto an explicit
target grid with `resample_to` semantics.

**M-23. Filters ignore spacing.** `filters/hessian.py`, `segmentation/blood_flood.py:80` (Frangi
sigmas 0.5-2.5) and the bilateral filter take scales in voxels. On anisotropic TOF, PET or CT the
filters respond differently along z. The Frangi output is also normalised by its per-volume
maximum, so GMM/hysteresis thresholds depend on the brightest structure of each case. Use
physical sigmas (`sigma_vox = sigma_mm / spacing`) or ITK's `HessianRecursiveGaussian`, and a
fixed normalisation. Jerman (`filters/jerman.py`, Jerman 2016) is already available and responds
more uniformly across vessel sizes than Frangi.

**M-24. Bilateral differs by backend.** `restoration/bilateral.py` filters slice by slice on CPU
even when 3-D is requested (the CPU path has no 3-D filter), and scikit-image bins the range
kernel; the CuPy kernel is true 3-D and exact. The same call gives different images on the
workstation and on the GPU node. `sigma_color` defaults to the global standard deviation, which
the background dominates; a noise estimate (`skimage.restoration.estimate_sigma`, MAD) is the
usual choice.

**M-25. Surface extraction.** `meshlab/marching_cubes.py:82` runs marching cubes without padding,
so masks touching the field of view give open surfaces; the divergence-theorem volume and the
outward-winding test in `_outward` are then unreliable. Binary marching cubes produces a
staircase that inflates area (and biases sphericity); per-label extraction leaves gaps and
overlaps between neighbouring labels. Pad by one voxel and shift back; for multi-label meshes
use `vtkSurfaceNets3D` (VTK >= 9.3, already pinned at 9.6) with windowed-sinc smoothing.

**M-26. Registration reproducibility.** `ants_register` passes `random_seed` only when given and
does not pin `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`; with MI random sampling and multithreading,
two runs differ. The CLIs (`nvitk-ants`, `nvitk-flirt`) default to linear interpolation for label
maps as well (the GUI maps labels to `genericLabel`). Default a seed and a thread count in
pipelines; pick `genericLabel` automatically for integer images.

**M-27. Backend switching is process-wide.** `core/backend.py` keeps the active backend in a
`ContextVar` (per thread) but `_update_backend_proxies` (`:141`) rebinds `np`/`ndi` in the globals
of 77 modules, which all threads share. A GUI worker running under `using("cupy")` changes `np`
for the main thread mid-computation. `map_in_thread_pool` documents the symptom. Resolve the array
module per call (`get_array_module(x)` or an explicit `xp` argument) instead of rebinding
globals. Also, the module docstring lists `NVITK_BACKEND=auto|cupy_required`, but
`_normalize_backend_name` rejects both and `_initial_backend` (`:113`) swallows the error, so
those values are ignored.

**M-28. Statistics convergence.** `stats/mixedlm.py:283` silences `ConvergenceWarning`,
`UserWarning` and `RuntimeWarning` from statsmodels. `converged` is stored in the fit summary,
but the mediation bootstrap keeps non-converged draws (only exceptions count as failures), and
its mixed-model engine defaults to `n_boot=500` with percentile intervals, so each 2.5 % tail
rests on about 12 draws. `sem.py` reports `converged = bool(measures) or not regressions.empty`,
which is not a convergence check. Count non-converged draws as failures, use at least 2000 draws
(BCa intervals where possible), and refit with `reml=False` before any likelihood-ratio or AIC
comparison of fixed effects.

**M-29. Left/right by midline.** `segmentation/hemisphere.py:140` splits along voxel axis 0 and
reads the side from the sign of `affine[0, 0]`, so it assumes axis 0 is left-right
(`lr_axis_and_sign` exists in the same module and is not used). The midline is the centre of the
field of view, not the anatomical midline. The docstrings have been corrected to say this; the
behaviour has not.

**M-30. PESA-Fat details.** `_process_bladder` (`ct_pet_v5/stage2_postprocess.py`) thresholds at
`max(Otsu(PET), 2.5)`; the floor reads like an SUV, but the PET is loaded in Bq/mL, so the floor
never applies. Erosions are counted in voxel iterations of a 6-connected element, so on 3 mm CT
slices one iteration removes three times more tissue along z than in-plane. PET is resampled
trilinearly onto the CT mask grid before SUV statistics; resampling the mask onto the PET grid
(or fractional partial-volume weights) avoids interpolating PET values.

---

## 4. Low findings

| Where | Finding |
|---|---|
| `cluster/sshfs.py:605` | `return` inside `finally` of a context manager swallows any exception raised in the `with` body when the mount is shared. |
| `viz/flowshow.py:1171, 2016, 2145` | `import traceback` later in the same function makes the name local, so the `except` handler raises `UnboundLocalError` and hides the real error. |
| `stats/r_mixedlm.py:445` | Dictionary key `"reml"` given twice; the second wins. |
| `types/mesh.py` | `MeshSeries.__post_init__` reads `self.metadata.get` before turning `None` into a dict. |
| `types/image.py` | `__array__(self, dtype=None)` has no `copy` argument (NumPy 2 deprecation). |
| `io/readers/nifti.py` | A directory input silently picks the first NIfTI alphabetically. |
| `io/imageio.py` | `imshow` uses `aspect="auto"` (ignores spacing); `imsave(axes=...)` permutes the array but not the affine. |
| `io/writers/nifti.py:129` | JSON goes into extension code 16, which the NIfTI registry assigns to PYPICKLE (6 is COMMENT). The comment that claimed otherwise has been corrected. |
| `restoration/n4_bias.py` | With `return_bias_field=True`, N4 runs twice. |
| `measure/hemodynamics.py` | RI uses the minimum instead of end-diastolic flow (common approximation). Plausibility bands are Zarrinkoob 2015 widened by ad hoc factors; label them as heuristic in outputs. |
| `io/conversors/_dicom_conversion.py` | `if is_tissue_segmentation(first_ds): pass` is a dead branch (tissue export is switched off). |
| `morphology/centerline.py` and 3 others | `Any` used in annotations without import (harmless at runtime, breaks `get_type_hints`). |
| `stats/brain_map.py` | Invalid escape `\|` in a docstring (SyntaxWarning on Python 3.12+). Fixed without changing the string value. |

---

## 5. Choices that are sound

Worth keeping, and worth copying elsewhere in the code:

* `normalization/intensity.py`: fixed HU window for CT, robust percentiles over non-zero voxels
  for MR, explicit modality, constant output for degenerate windows.
* `segmentation/losses.py`: soft clDice reproduces Shit et al. 2021 exactly, with chunked
  per-class computation and checkpointing for memory.
* `pipes/topbrain/util/folds.py`: patient-grouped, rare-class-stratified, seeded folds; avoids
  the CT/MR leakage between folds.
* `measure/segmentation_metrics.py`: paired Wilcoxon per case and fold spread instead of cohort
  means; absent classes excluded rather than scored perfect.
* `io/conversors/_dicom_dynamic.py`: careful temporal layout detection, integer-preserving
  rescale, dicom2nifti-compatible affine.
* Philips FP scaling `DV / (RS * SS)` matches Chenevert et al. 2014.
* `measure/hemodynamics.py`: faithful port of the QVT/QVT+ PWV estimators and quality scores.
* `stats/pairwise.py` (Holm), `stats/mediation.py` (cluster bootstrap with relabelled copies).

---

## 6. State of the art and improvement opportunities

| Topic | Current | Options |
|---|---|---|
| DICOM to NIfTI | dicom2nifti plus custom fallbacks | dcm2niix (Li 2016) as the primary converter: BIDS sidecars, de-identification, tilt, vendor-specific scaling; keep the fallbacks for the odd series only. |
| De-identification | none | DICOM PS3.15 basic profile on read; allow-listed sidecars. |
| 4D flow pre-processing | polynomial background, no unwrapping | Static-tissue background fit (temporal SD + magnitude mask), Laplacian unwrapping (Loecher 2016), noise masking, following the 2023 4D flow consensus (Bissell 2023). |
| 4D flow segmentation | CD sliding threshold + Frangi flood | Deep-learning 4D-flow vessel segmentation trained on PC-MRA, or registration of a TOF-derived segmentation; nnU-Net infrastructure is already in the repo. |
| Vesselness | Frangi in voxel units | Jerman (already present) with physical scales; learned vesselness. |
| Surface meshes | per-label marching cubes | SurfaceNets (`vtkSurfaceNets3D`) for shared-boundary multi-label meshes; constrained smoothing that keeps volume. |
| PET quantification | SUVmax/mean/percentiles | QIBA SUV pseudo-code, SUL (Janmahasatian), SUVpeak (PERCIST), partial-volume correction for small structures (e.g. iterative Yang or GTM). |
| Segmentation metrics | in-house | Metrics Reloaded definitions (Maier-Hein 2024); MONAI or `surface-distance` for surface metrics; Betti-1 or topology-matching metrics used by TopCoW. |
| Registration | ANTs SyN, FireANTs | Fixed seeds and thread counts for reproducibility; SynthMorph or learned registration for cross-modality cases. |
| Denoising | bilateral | Non-local means or BM4D with a noise estimate; DL denoisers for PET and ASL. |
| Testing | none | Golden-file tests on small phantoms for every quantitative function (area, flow, SUV, HD95, geometry round trips). The scripts in `audit/scripts` are a starting point. |

---

## 7. Coverage

Read in full or in depth: `core`, `types`, `io` (readers, writers, `imageio`, the DICOM
conversion stack, `phase2volume`), `measure` (volume, intensity, SUV, voxel, surface,
segmentation metrics, compare, hemodynamics, cross-section), `transform` (resampling, isotropy,
oblique), `filters` (Hessian, sliding threshold), `restoration` (bilateral, N4),
`normalization`, `segmentation` (hemisphere, losses, blood flood), `meshlab` (marching cubes,
measurements), `registration/ants.py`, the qvtpy measurement path (stages 0 and 6, LOC and
cross-section code), PESA-Fat stage 2/3 CT-PET, TopBrain folds and inference entry point.

Scanned for specific patterns (static analysis plus targeted reading): the rest of
`segmentation`, `morphology`, `measure/morpho`, `vmtk`, `fsl`, `freesurfer`, `stats`, `db`,
`cluster`, `cli`, `viz`, `gui`, the CViT and TopBrain training stages, `slicer/` and `scripts/`.
Algorithms in those areas (for example `morphology/siphon_cut.py`, the VMTK centerline network,
the stats GUI) were not validated numerically and deserve their own review.

---

## References

* Bissell MM et al. 4D flow cardiovascular magnetic resonance consensus statement: 2023 update. *J Cardiovasc Magn Reson* 2023;25:40.
* Boellaard R et al. FDG PET/CT: EANM procedure guidelines for tumour imaging, version 2.0. *Eur J Nucl Med Mol Imaging* 2015;42:328-354.
* Chenevert TL et al. Errors in quantitative image analysis due to platform-dependent image scaling. *Transl Oncol* 2014;7:65-71.
* Dyverfeldt P et al. 4D flow cardiovascular magnetic resonance consensus statement. *J Cardiovasc Magn Reson* 2015;17:72.
* Janmahasatian S et al. Quantification of lean bodyweight. *Clin Pharmacokinet* 2005;44:1051-1065.
* Jerman T et al. Enhancement of vascular structures in 3D and 2D angiographic images. *IEEE Trans Med Imaging* 2016;35:2107-2118.
* Lankhaar JW et al. Correction of phase offset errors in main pulmonary artery flow quantification. *J Magn Reson Imaging* 2005;22:73-79.
* Li X et al. The first step for neuroimaging data analysis: DICOM to NIfTI conversion. *J Neurosci Methods* 2016;264:47-56.
* Loecher M et al. Phase unwrapping in 4D MR flow with a 4D single-step Laplacian algorithm. *J Magn Reson Imaging* 2016;43:833-842.
* Maier-Hein L et al. Metrics reloaded: recommendations for image analysis validation. *Nat Methods* 2024;21:195-212.
* Schrauben E et al. Fast 4D flow MRI intracranial segmentation and quantification in tortuous arteries. *J Magn Reson Imaging* 2015;42:1458-1464.
* Shit S et al. clDice: a novel topology-preserving loss function for tubular structure segmentation. *CVPR* 2021.
* Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation. *BMC Med Imaging* 2015;15:29.
* Tustison NJ et al. N4ITK: improved N3 bias correction. *IEEE Trans Med Imaging* 2010;29:1310-1320.
* Wahl RL et al. From RECIST to PERCIST. *J Nucl Med* 2009;50 Suppl 1:122S-150S.
* Walker PG et al. Semiautomated method for noise reduction and background phase error correction in MR phase velocity data. *J Magn Reson Imaging* 1993;3:521-530.
* Zarrinkoob L et al. Blood flow distribution in cerebral arteries. *J Cereb Blood Flow Metab* 2015;35:648-654.
