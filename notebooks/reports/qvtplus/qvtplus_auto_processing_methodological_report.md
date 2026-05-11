## Methodological report: QVTplus automatic segmentation & analysis pipeline (source-based)

**Codebase scope**: `src/nvitk/pipes/qvtplus/src` (MATLAB).

**Pipeline entrypoints**:

- Primary automatic pipeline: `src/nvitk/pipes/qvtplus/src/auto_processing/paramMap_auto.m`
- Additional merged step (re-centerline + re-run analysis): `src/nvitk/pipes/qvtplus/src/auto_processing/paramMap_auto_centerline_test.m`

This report is based on the *existing local code* at:

- `/home/imarcoss/nvitk/src/nvitk/pipes/qvtplus/src/auto_processing/paramMap_auto.m`
- `/home/imarcoss/nvitk/src/nvitk/pipes/qvtplus/src/auto_processing/paramMap_auto_centerline_test.m`

---

## 1) What the pipeline computes (high level)

Given a 4D Flow MRI dataset (magnitude + 3-direction velocity phases), the pipeline:

- builds a **3D vascular mask** from a complex-difference-like angiogram (CD),
- extracts **centerlines** and segments the skeleton into branches (segment IDs),
- generates **cross-sectional planes** orthogonal to the centerline,
- performs **in-plane segmentation** for each plane to estimate vessel cross-section and compute:
  - cross-sectional **area**,
  - a “**circularity/diameter**” proxy (stored as `diam_val`),
  - time-averaged and time-resolved **flow** and related indices (PI, RI, etc.),
- (optionally) transfers **eICAB** TOF-derived labels into QVT space to label branches by vessel,
- writes per-vessel and summary outputs (spreadsheets, images, CSV label maps),
- and (centerline-test step) recomputes the entire plane+flow pipeline using an alternative **arterial+venous merged centerline**.

---

## 2) Entrypoint A: `paramMap_auto.m` (automatic processing)

### 2.1 Supported calling conventions

`paramMap_auto` supports:

- **Single patient mode**:
  - `paramMap_auto(path_to_data, eICAB_path, output_path)`
  - `paramMap_auto(path_to_data, eICAB_path, output_path, use_eicab_whole_brain)`
- **Batch mode**:
  - `paramMap_auto('--batch', base_path, patient_id, use_eicab_whole_brain, skip_existing)`
  - `patient_id` may be `'all'`

Batch mode folder structure (as implemented):

- 4DFlow input: `base_path/DATA/Nifti/<patient_id>/4DFlow`
- eICAB results: `base_path/RESULTS/eICAB/<patient_id>`
- QVTplus outputs: `base_path/RESULTS/QVTPlus/<patient_id>`

### 2.2 Main per-subject procedure (`processSinglePatient`)

For each subject, `paramMap_auto` executes, in order:

1) **Load / build QVT dataset**: `loadPreprocessedData(path_to_data, output_path)`
2) **Label transfer from eICAB**: `performLabelTransfer(eICAB_path, output_path, imageData, path_to_data, data_struct, use_eicab_whole_brain)`
3) **Vessel LOC selection**: `generateLOCs(data_struct, correspondenceDict, multiQVT)`
4) **Time-resolved CD enrichment** (best-effort): `computeTimeResolvedCD(...)` and store in `data_struct.timeMIPcrossectionTR`
5) **Per-vessel exports**: `saveVesselData(LOCs, data_struct, output_path)`
6) **QVTplus label table**: `generateQVTplus(correspondenceDict, LOCs, output_path)` → `LabelsQVT.csv`

It also writes a small helper file:

- `output_path/.last_output_path.txt` (for GUI auto-launch)

---

## 3) Stage A1: Data loading and initial segmentation (`loadPreprocessedData` → `loadNII_auto`)

### 3.1 Data loading behavior (`loadPreprocessedData.m`)

`loadPreprocessedData(path_to_data, output_path)`:

- If a `.mat` exists directly in `path_to_data`, it loads the first `.mat` found (assumes preprocessed).
- Else, it loads raw data (NIfTI) via `loadNII_auto(path_to_data)` and saves:
  - `output_path/qvtData_ISOfix_<ddmmmyyyy_HHMM>.mat`

### 3.2 Expected NIfTI layout and metadata (`loadNII_auto.m`)

`loadNII_auto(directory)` expects directional subfolders matching `AP`, `RL`, `FH`, either:

- directly under `directory`, or
- under `directory/scans/`,

and optionally with a nested `NIFTI/` within each direction folder.

It:

- reads a magnitude JSON file from AP (first `*.json` in AP folder),
- loads magnitude time series from the first AP `*.nii.gz` (fallback `*.nii`),
- loads phase volumes `*_ph.nii(.gz)` in AP/RL/FH,
- computes:
  - `MAG = mean(mag,4)` (time-averaged magnitude),
  - `v` (time-resolved velocity) in **mm/s**,
  - `vMean = mean(v,5)` (mean velocity field),
  - `VoxDims` from the NIfTI affine,
  - `res = VoxDims(1)` and `sliceSpace = VoxDims(3)`.

It stores `imageData.OriginalAffine` from the loaded magnitude NIfTI so that outputs preserve orientation.

### 3.3 Complex Difference (angiogram) and background phase correction

`loadNII_auto` computes a CD-like angiogram:

- `timeMIP = calc_angio(MAG, vMean, VENC)` with `VENC = 700` mm/s

It then optionally performs unattended polynomial background phase correction (`unattended_background_phase_correction`) and subtracts the fitted background from `vMean` and each `v(:,:,:,:,frame)`.

### 3.4 3D segmentation: sliding threshold + component filtering

The initial binary vessel segmentation is computed from `timeMIP` by:

- `slidingThreshold(timeMIP, step=0.001, UPthresh=0.8, SMf=10, shiftHM_flag=1, medFilt_flag=1)`
- followed by connected-component size filtering:
  - `areaThresh = round(sum(segment(:)) * 0.005)`
  - `segment = bwareaopen(segment, areaThresh, conn=6)`

It stores:

- `imageData.MAG`, `imageData.CD`, `imageData.V` (= `vMean`), `imageData.v` (= time-resolved `v`), and `imageData.Segmented` (= 3D mask).

---

## 4) Stage A2: Centerlines and branch ordering (`feature_extraction.m`)

`loadNII_auto` calls:

- `feature_extraction(sortingCriteria=3, spurLength=8, vMean, segment)`

Method:

1) **Skeletonization**: `bwskel(segment, 'MinBranchLength', spurLength)` then zero-out skeleton at the volume edges.
2) **Branch graph creation**:
   - `centerlineX` to build branch connectivity and labeling,
   - `centerline_new` to trim junctions (`settings.cl.branchMinLength = 5`).
3) **BranchList creation and flow-consistent direction**:
   - each branch produces rows of `[y, x, z, segment_id]`,
   - direction is reversed if the dot product between displacement and summed local velocity indicates reversed flow (so point order follows flow direction),
   - a within-segment index is appended, producing `[y, x, z, segment_id, point_index]`.
4) **Spline smoothing**:
   - cubic smoothing splines (`csaps`) with `smoothParameter = 0.375` applied per segment, replacing xyz coordinates with smoothed values.

The centerline representation used downstream is:

- `branchList`: \(N \times 5\) matrix with point coordinates, segment ID (`col 4`), and within-segment index (`col 5`).

---

## 5) Stage A3: Cross-sectional planes + in-plane segmentation + hemodynamics (`paramMap_params_threshS.m`)

The main quantitative analysis is implemented in:

- `src/nvitk/pipes/qvtplus/src/functions/initialQVTprocessing/paramMap_params_threshS.m`

It is called in `loadNII_auto` with `SEG_TYPE='thresh'`.

### 5.1 Plane generation (`create_planes`)

Planes are created along `branchList` with:

- plane radius `r = 10` pixels,
- interpolation factor `InterpVals = 4`,
- plane width `width = r * InterpVals * 2 + 1`.

`create_planes` also returns tangent vectors (`Tangent_V`) used to define through-plane direction.

### 5.2 Effective pixel spacing correction (`pixelSpace`)

The code computes a per-plane effective pixel spacing to account for plane tilt:

- `pixelSpace(i) = res + (sliceSpace - res) * sin(angle(Tangent_V, z-axis))`

### 5.3 Interpolation of volumetric data into planes

The pipeline interpolates to each plane:

- mean velocity components (`vMean(:,:,:,1:3)`) then projects them along `Tangent_V`,
- the CD-like angiogram (`timeMIP`),
- the magnitude (`MAG`).

It defines the per-plane “through-plane speed magnitude” used for segmentation as:

- `vTimeFrameave = sqrt(vx_proj^2 + vy_proj^2 + vz_proj^2)` (mm/s)

### 5.4 In-plane segmentation

Cross-section masks are computed by:

- `segment_cross_section_thresh(...)`

and produce:

- `area_val` (used later to compute flow),
- `diam_val` (used downstream as a circularity/shape metric),
- `segmentFull` (flattened per-plane masks).

### 5.5 Time-resolved flow and derived indices (PI/RI)

For each cardiac frame \(j\), it:

- extracts per-frame velocities from `v(:,:,:,:,j)` (NIfTI case),
- interpolates each component into planes,
- projects into through-plane direction using `Tangent_V`,
- applies the in-plane mask `segmentFull` and computes a mean masked velocity per plane,
- converts to flow per frame:
  - `flowPulsatile_val(:,j) = abs(meanVel(:,j) .* area_val)`

It then computes per-plane time-averaged measures:

- `flowPerHeartCycle_val = sum(flowPulsatile_val,2) / nframes`
- `velMean_val = sum(velPulsatile_val,2) / nframes`
- `maxVel_val = max(maxVelFrame,[],2)`

And flow-derived indices:

- **Pulsatility Index (PI)**:
  - `PI_val = abs(max(flowPulsatile_val,[],2) - min(flowPulsatile_val,[],2)) ./ mean(flowPulsatile_val,2)`
- **Resistivity Index (RI)**:
  - `RI_val = abs(max(flowPulsatile_val,[],2) - min(flowPulsatile_val,[],2)) ./ max(flowPulsatile_val,[],2)`

It also computes:

- per-branch mean/std flow (`bnumMeanFlow`, `bnumStdvFlow`) by aggregating all points with the same `segment_id`,
- a pointwise quality-style composite score `StdvFromMean` based on local windowed variability in flow, area, “circularity” (`diam_val`), and waveform tightness.

---

## 6) Stage A4: eICAB → QVT label transfer (`performLabelTransfer.m`)

The automatic pipeline uses eICAB outputs to label QVT branches.

### 6.1 Inputs

`performLabelTransfer(eICAB_path, output_path, imageData, refImage, data_struct, use_eicab_whole_brain)` expects in `eICAB_path`:

- a `*_resampled.nii(.gz)` TOF image (used as the moving image),
- either `*_eICAB_CW.nii(.gz)` or (if `use_eicab_whole_brain`) `*_eICAB_WB.nii(.gz)` (multi-label mask).

### 6.2 QVT-space NIfTI exports written before registration

It writes, in `output_path`:

- `QVT_seg.nii` from `imageData.Segmented`
- `QVT_MAG.nii` from `imageData.MAG`
- `QVT_CD.nii` from `imageData.CD`
- `branch_mask.nii` where each centerline voxel is assigned its `segment_id` (from `data_struct.branchList(:,4)`)

All are written with `data_struct.OriginalAffine` (or `imageData.OriginalAffine`) when available.

### 6.3 Registration (FSL FLIRT) and applying the transform

Registration is performed with **FSL**:

- `flirt -in <tofOrigResampled> -ref <QVT_MAG> -omat transform.mat -out r_<tofOrigResampled> -cost normmi -searchcost normmi -dof 6`
- Then the transform is applied to the eICAB label mask with nearest-neighbour interpolation:
  - `flirt -in <eICAB_mask> -ref <QVT_MAG> -applyxfm -init transform.mat -out r_<eICAB_mask> -interp nearestneighbour`

This produces (in the same folder as the eICAB mask copy):

- `transform.mat`
- registered masks prefixed with `r_...`

### 6.4 Label transfer to the QVT segmentation (kNN in voxel space)

`transferLabels` implements label assignment by:

- extracting all QVT binary voxels (`imageData.Segmented==1`),
- extracting all nonzero voxels from the registered multi-label image,
- doing a `knnsearch` from each QVT voxel to k=5 nearest labeled voxels,
- filtering neighbours by a distance threshold (`distanceThreshold=5`),
- assigning majority labels (via helper functions in `label_transfer.m`),
- then expanding labels (`expandLabels(updatedBinarySegMatrix, 100, 10)`).

It writes:

- `multilabel_QVTseg.nii` (QVT binary mask with transferred integer vessel labels).

### 6.5 Mapping label IDs to named vessels (`generateCorrespondenceDict.m`)

`generateCorrespondenceDict(folderPath, data_struct)`:

- loads `multilabel_QVTseg.nii`,
- samples it at each rounded centerline coordinate from `data_struct.branchList(:,1:3)`,
- builds a mapping from each eICAB label (`good_lab_<id>`) to the set of QVT `segment_id`s intersecting it,
- resolves duplicate mappings (`correspondence_funcs`) and renames keys using an explicit mapping table, including:
  - `good_lab_1→LICA`, `2→RICA`, `3→BASI`, `5→LACA`, `6→RACA`, `7→LMCA`, `8→RMCA`, etc.
- handles PCA variants by comparing the maximum segment length among candidate labels,
- splits `COMM` into `RCOMM` and `LCOMM` based on the x-midpoint in RAS orientation.

---

## 7) Stage A5: LOC selection and outputs (`generateLOCs`, `saveVesselData`, `generateQVTplus`)

### 7.1 LOC selection (`generateLOCs.m`)

`generateLOCs(data_struct, correspondenceDict, multiQVT)` selects “locations of interest” (LOCs) per named vessel. Outputs are stored as:

- `LOCs.<vessel> = [segment_id, point_index]`

Key behaviors:

- ICA/BA LOCs are selected around a slice criterion and refined using circularity-based selection (via `find_LOCs` helpers).
- Venous vessels (SSSV, LTSV, RTSV, STRV) have additional heuristics to disambiguate when multiple vessels share the same segment:
  - splitting the segment into parts and selecting points by vertical/horizontal variability,
  - SVD-based direction alignment to distinguish STRV (diagonal/vertical) from SSSV (more horizontal),
  - validation and potential swapping if alignment indicates misassignment.

### 7.2 Summary and per-vessel reports (`saveVesselData.m`)

`saveVesselData(LOCs, data_struct, output_path)` writes:

- `SummaryParamTool.xls` (via `xlwrite`) including a `Summary_Centerline` sheet with:
  - LOC index, branch number, flow, PI, and a velocity-threshold QC flag.
- For each vessel LOC, two outputs:
  - numeric tables (time-averaged and time-resolved) as additional sheets in `SummaryParamTool.xls`,
  - a cross-sectional montage image:
    - `*_Slicesview.jpg` containing MAG, CD, velocity, and mask panels for a small window of points around the LOC.

### 7.3 Label/LOC table (`generateQVTplus.m`)

`generateQVTplus(correspondenceDict, LOCs, output_path)` writes:

- `LabelsQVT.csv`

with columns:

- `Artery` (human-readable name),
- `Label` (segment IDs in `correspondenceDict`),
- `Loc` (`[segment_id, point_index]` from `LOCs` when present).

---

## 8) Time-resolved CD cross-sections enrichment (`computeTimeResolvedCD.m`)

`paramMap_auto` attempts to compute `data_struct.timeMIPcrossectionTR` (best-effort; warnings only on failure).

Method:

- Reconstruct the same planes as in `paramMap_params_threshS` using:
  - `r = data_struct.r`, `InterpVals=4`, `width = r*InterpVals*2+1`
- For each frame:
  - compute speed `Vmag = sqrt(sum(vFrame.^2,4))`,
  - cap `Vmag` at `VENC`,
  - compute frame CD: `timeMIP_frame = MAG .* sin((pi/2 * Vmag) / VENC)`,
  - interpolate frame CD to planes via `interp_vol_to_planes`.

If time-resolved `imageData.v` is missing, it attempts to reload velocity NIfTIs from disk.

---

## 9) Entrypoint B: `paramMap_auto_centerline_test.m` (arterial+venous centerline merge and re-analysis)

This step is explicitly designed to **not redo registration**: it loads the newest `qvtData_ISOfix_*.mat` from the existing QVT output folder and reconstructs a new `branchList` before re-running the full hemodynamic pipeline.

### 9.1 Venous search region and masks (4D Flow space)

Given `segmentCD = imageData.Segmented`, it defines a “venous region” as the first third of the Y dimension:

- `third_y = round(sz(2)/3)`
- `venous_region(:, 1:third_y, :) = true`

Then:

- `venous_mask = segmentCD & venous_region`

For the arterial side in 4D Flow space, it prefers the registered eICAB label mask (if present):

- If `outputDir/r_TOF_eICAB_CW.nii` exists:
  - `arterial_mask = (multiQVT > 0) & ~venous_region` (resized to match 4D Flow dims if needed)
- Else:
  - `arterial_mask = segmentCD & ~venous_region`

Combined mask used for saving and fallback:

- `segment_combined = arterial_mask | venous_mask`
- small-component removal: `bwareaopen(segment_combined, max(1, round(sum(segment_combined(:))*0.005)), 6)`

### 9.2 Two-path centerline construction (preferred) vs fallback

The method chooses a “two-path” approach if both files exist:

- `outputDir/TOF_eICAB_CW.nii` (TOF/eICAB mask in TOF space)
- `outputDir/transform.mat` (FSL FLIRT transform from TOF to QVT space, produced during label transfer)

If both exist:

#### 9.2.1 Arterial centerline in TOF space

- Load TOF-space binary mask: `mask_tof = spm_read_vols(TOF_eICAB_CW) > 0`
- Area-open: `bwareaopen(mask_tof, max(1, round(sum(mask_tof(:))*0.005)), 6)`
- Centerline extraction in TOF space:
  - `feature_extraction(sortingCriteria=3, spurLength=8, vMean_tof=zeros, mask_tof)`

#### 9.2.2 Transform arterial centerline points into 4D Flow space

It attempts the highest-fidelity mapping first:

- If `img2imgcoord` is available and `outputDir/QVT_MAG.nii` exists:
  - write 0-based voxel coords to a temp file,
  - run `img2imgcoord -src TOF -dest QVT_MAG -xfm transform.mat -vox`,
  - parse output and convert back to 1-based voxel coordinates.

Otherwise it falls back to SPM/world-matrix math:

- Uses `spm_vol` matrices and applies the `transform.mat` 4×4 transform in mm space,
- then converts to reference voxel coordinates using `data_struct.OriginalAffine` (or a derived affine).

**Critical stabilization step**: float coordinates are snapped to integer voxels and de-duplicated, because TOF is higher resolution than 4D Flow and many TOF points map into a single 4D Flow voxel:

- `branchList_arterial_4df(:,1:3) = round(...)`
- clip to bounds
- per segment, remove consecutive duplicates and reset `point_index` to 1..N.

#### 9.2.3 Venous centerline in 4D Flow space

Venous centerlines are extracted directly in QVT space using:

- `venous_clean = bwareaopen(venous_mask, max(1, round(sum(venous_mask(:))*0.005)), 6)`
- `feature_extraction(..., vMean, venous_clean)`

Venous segment IDs are renumbered to follow arterial segments, then concatenated.

#### 9.2.4 Segment sanity constraints and smoothing

Before running plane extraction, it enforces a minimum points-per-segment constraint to avoid indexing failures in plane creation:

- `MIN_POINTS_PER_SEGMENT = 5`
- segments with fewer points are dropped
- segment IDs are renumbered to be contiguous 1..N

It then re-smooths the merged centerline per segment with cubic smoothing splines:

- `smoothParameter = 0.375`
- `csaps` per dimension, same family of smoothing as `feature_extraction`.

If either `TOF_eICAB_CW.nii` or `transform.mat` is missing, it falls back to:

- `feature_extraction(..., vMean, segment_combined)` to extract a single centerline from the combined mask in 4D Flow space.

### 9.3 Re-running the full plane+flow pipeline on the new centerline

With the reconstructed centerline `branchList_new`, it runs:

- `paramMap_params_threshS(filetype='nii', branchList_new, matrix, timeMIP, vMean, back=zeros, BGPCdone=1, directory=outputDir, nframes, res, MAG, v, sliceSpace, Exseg=[])`

and stores outputs into a new `data_struct_new` (copy of the original, but with rederived fields and updated `branchList`).

It also computes a velocity-based pulsatility index `PIvel_val` from `maxVelFrame` if present:

- `PIvel_val = (max(maxVelFrame,[],2) - min(maxVelFrame,[],2)) ./ mean(maxVelFrame,2)`

### 9.4 Outputs written under `outputDir/centerline_test/`

The function writes a dedicated output folder:

- `outputDir/centerline_test/`

and produces:

- **MAT**:
  - `qvtData_ISOfix_centerline_<timestamp>.mat` containing:
    - `data_struct` (with centerline-test values),
    - `Vel_Time_Res` updated with new `VplanesAllx/y/z`,
    - `imageData` (original image data).
- **Masks / centerline volumes (NIfTI)**:
  - `segment_centerline_eICAB_venous.nii` (combined arterial+venous mask in QVT space)
  - `segment_centerline_eICAB_only.nii` (arterial mask only)
  - `segment_centerline_venous_only.nii` (venous mask only)
  - `branch_mask.nii` (centerline volume labeled by `segment_id`)
- **Per-centerline-point table**:
  - `flow_PI_per_centerline_centerline_test.csv`
  - plus `flow_PI_per_centerline_centerline_test.xlsx` (best effort)
  - columns: `vessel_name`, `segment_id`, `centerline_point_index`, `point_index`, `flow_ml_s`, `PI`, `area_val`, `circularity`
- **Re-generated vessel exports** (in the centerline_test folder):
  - copies `multilabel_QVTseg.nii` from the original folder (if present),
  - runs `generateCorrespondenceDict`, `generateLOCs`, then:
    - `saveVesselData(...)` → `SummaryParamTool.xls` + `*_Slicesview.jpg`
    - `generateQVTplus(...)` → `LabelsQVT.csv`

---

## 10) Notes on coordinate conventions and orientation

This codebase treats `branchList(:,1:3)` as `[y, x, z]` indices when building `branch_mask.nii` and when interfacing with FSL tools (see comments in centerline-test code). Throughout, transforms are carefully handled using:

- original NIfTI affines stored in `imageData.OriginalAffine` / `data_struct.OriginalAffine`,
- FSL `transform.mat` from FLIRT for TOF→QVT mapping,
- integer voxel snapping after mapping TOF-derived points into the lower-resolution 4D Flow space.

---

## 11) File products you should expect per subject

At minimum, after running `paramMap_auto` you should see:

- `qvtData_ISOfix_<timestamp>.mat`
- `QVT_seg.nii`, `QVT_MAG.nii`, `QVT_CD.nii`
- `branch_mask.nii`
- `multilabel_QVTseg.nii` (after label transfer)
- `SummaryParamTool.xls`
- `LabelsQVT.csv`

After running `paramMap_auto_centerline_test`, additionally:

- `centerline_test/` directory with:
  - `qvtData_ISOfix_centerline_<timestamp>.mat`
  - `segment_centerline_*.nii`, `branch_mask.nii`
  - `SummaryParamTool.xls`, `LabelsQVT.csv`
  - `flow_PI_per_centerline_centerline_test.csv` (and maybe `.xlsx`)


