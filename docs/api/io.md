# I/O

`nvitk.io` reads and writes every image format the toolkit works with, dispatching on file
extension or an explicit `force_type`, and converting between acquisition formats and NIfTI.

## Reading, writing, and displaying

```{code-block} python
from nvitk.io import imread, imsave, imshow

img = imread("study/pet", force_type="dicom", backend="gpu")
imshow(img, axis=2, index="mid")
imsave("out/pet_copy.nii.gz", img)
```

| Function | Purpose |
|---|---|
| `imread` | Reader dispatch by extension or `force_type` (`nifti`, `dicom`, `tiff`, `mha`, `pil`, `nd2`, `b2nd`, `pkl`, ...) — returns an {class}`~nvitk.types.image.Image`. |
| `imsave` | Writer dispatch, mirrors `imread`'s format set. |
| `imshow` | Slice view, orthogonal view, mosaic, or animation, for quick inspection. |
| `convert_image`, `swapaxes` | Format conversion and axis-order helpers. |

Per-format implementations live in `nvitk.io.readers` (`b2nd`, `dicom`, `mha`, `nd2`,
`nifti`, `pil`, `pkl`, `tiff`) and `nvitk.io.writers` (which cover `nifti`, `mha`, `tiff`, `pil`).

## Browsing, editing and exporting DICOM

`nvitk.io.dicom_index.scan_dicom` indexes a DICOM folder from the headers alone — studies,
series (split by `ImageType` as the loader splits them), files, and what each file is (an
image, a waveform, a report, a Philips spectral base image) — and `load_dicom_series` takes
an explicit list of files, so any series or subset of it can then be loaded.
`nvitk.io.dicom_edit` keeps header changes as pending operations over sets of files and
writes them only when exporting:

```{code-block} python
from nvitk.io.dicom_index import scan_dicom
from nvitk.io.dicom_edit import DicomEditor, export_dicom
from nvitk.io.conversors._dicom_conversion import load_dicom_series

study = scan_dicom("/data/patient01")[0]
series = next(s for s in study.series if s.number == "401")
data, meta = load_dicom_series(series.paths[100:140])          # just those slices

ed = DicomEditor()
ed.anonymize(series.paths, patient_name="SUBJ-001", patient_id="SUBJ-001", dates="shift", shift_days=-100)
ed.set(series.paths, "SeriesDescription", "CT 78%")
ed.delete(series.paths, (0x0020, 0x4000))                     # ImageComments
export_dicom(series.paths, "/data/out", editor=ed, layout="patient / study / series")
```

| Name | Purpose |
|---|---|
| `scan_dicom` | Studies → series → files of a folder or file list, headers only, in parallel; `read_header` for one file. |
| `DicomEditor` | Pending `set` / `delete` / `anonymize` operations, each over a set of files; `header(path)` shows a file as it will be exported. |
| `Anonymize` | De-identification (PS3.15 basic profile, subset): names and IDs replaced, identifying attributes removed, dates kept / shifted / blanked, private tags dropped, consistent new UIDs. Pixels are not inspected. |
| `export_dicom` | Writes chosen files with the edits applied (series folders, patient / study / series, or flat; original or numbered names); pixel data and transfer syntax unchanged. |

## Preprocessed cases (`.b2nd` / `.pkl`)

nnU-Net and nnssl store preprocessed volumes as a Blosc2 array shaped `(C, Z, Y, X)` beside a
`.pkl` sidecar holding the source SimpleITK geometry and the crop/resample bookkeeping — for
example the ToPBrain outputs under
`$nnssl_preprocessed/<Dataset>/<plans>_<config>/.../<session>/TOF.b2nd`. `read_b2nd` reassembles
the world geometry those two files only imply: it takes the target spacing from the
`*Plans*.json` it finds by walking up to the folder named by a configuration's `data_identifier`,
shifts the origin by the crop bounding box, and returns the volume in `XYZ` order with an RAS
affine — so a preprocessed case overlays exactly on the scan it came from.

```{code-block} python
from nvitk.io import imread

img = imread(".../pesa_tof-PESA1521/ses-1/TOF.b2nd")   # (X, Y, Z), affine + orientation
img = imread(".../ses-1")                              # every .b2nd in the folder
img = imread(".../ses-1/TOF.pkl")                      # sidecar → its companion array
```

A length-1 channel axis is dropped by default (`squeeze_channel=False` keeps it as `XYZC`, and
`channel=i` decompresses just one channel of a multi-modality case). The raw sidecar survives in
`img.metadata["preprocessing_properties"]`, with `spacing_source` recording whether the spacing
came from the plans file, the sidecar, or a shape-ratio fallback. In the GUI these files open
like any other volume — file dialog, drag & drop, or `open_paths_with_nvitk`.

## Format conversors

CLI-facing conversion tools, each also a registered entry point:

| Command | Module | Converts |
|---|---|---|
| `dcm2nii` | `nvitk.io.conversors.dcm2nii` | DICOM → NIfTI (incl. RT structs, tissue segmentation, Zeiss-specific handling) |
| `stl2nifti` | `nvitk.io.conversors.stl2nifti` | Surface mesh (STL) → labelmap/NIfTI |
| `phase2volume` | `nvitk.io.conversors.phase2volume` | Phase-contrast MRI → velocity volume and derivatives |
| `nikon2nifti` | `nvitk.io.conversors.nikon2nifti` | Nikon microscopy → NIfTI |

### Dynamic series and mixed geometry in `dcm2nii`

`dcm2nii` (`run_dicom2nifti` / `load_dicom_series`) handles the series a volume-only converter
cannot (`nvitk.io.conversors._dicom_dynamic`):

- **Repeated slice positions → 3D+t.** Bolus tracking (one position, many times), perfusion and
  cine (a stack repeated over time) are written as `X x Y x Z x T` NIfTI — `Z = 1` for a single
  slice — ordered by `TemporalPositionIdentifier` / `TriggerTime` / cardiac phase / acquisition
  time, with the frame interval in `pixdim[4]` and `frame_times_s` in the sidecar. The affine is
  dicom2nifti's formula, so the 4D output sits in the same world frame as the study's 3D series.
  MR keeps dicom2nifti's own (vendor-aware) 4D path first.
- **Mixed image geometry → sub-series.** A series mixing matrix sizes or orientations (surview
  projections, reformats, captures) is split into one file per geometry, suffixed by what differs
  (`_COR` / `_SAG`, `_158x512`).
- Failures fall back to the temporal builder and then to a geometry-aware basic stack (RAS affine,
  largest consistent slice group, HU rescale applied); every message names the series. Single-image
  series (localizers, a locator, captures) go straight to that path.
- **Philips spectral base images (SBI)** — Secondary Captures described `SBI(V5.1)_…|IMR, 78%`,
  `ImageType DERIVED\SECONDARY\SBI\SBI_CSPN`, one per axial slice with a different row count each —
  are compressed proprietary spectral data, not pixels, and are never converted. `--sbi` decides
  what is kept: `manifest` (default) writes `spectral_base_images.json` — per SBI series its
  version, kernel, slice thickness, source files and the reconstruction (and NIfTI) it belongs to;
  `copy` also copies the SBI DICOM into `sbi/<series>/`; `skip` keeps nothing.
- **ECG waveforms** (General ECG Waveform objects, e.g. `For Series: 401 - 40997`) are exported as
  `<name>.csv` (time + leads) and `<name>.json` (sampling rate, start time, R-peaks, RR intervals,
  median heart rate) instead of an image.
- **Colour screen captures** (dose report, tracker graph) are written `X x Y x RGB` and read back as
  RGB images.
- `HeartRate` is filled from Philips private (01F1,1045) *Initial Heart Rate* when the standard tag
  is empty, so phase stacks get a time axis in seconds.
- The RT-Struct scan reads headers only.

A Philips prospective cardiac CT (e.g. ES17774, 4737 files) converts to: the phase/kernel
reconstructions as 3D volumes, the bolus **tracker** as `512 x 512 x 1 x 9` (Δt 1 s, HU), the locator
and the two localizers (COR/SAG) as 2D HU images, the tracker graph and the dose/patient summary as
RGB captures, the ECG as CSV/JSON, and — with `--stack-phases` — one `X x Y x Z x 3` stack per
reconstruction at 73/78/83 % R-R.

### Spectral CT results

The usable spectral data comes from a Philips spectral workstation (IntelliSpace Portal / Spectral
Diagnostic Suite / Spectral Magic Glass): load the original study including the SBI series,
generate the results and export them as DICOM. They then convert like any CT series, and
`nvitk.io.conversors._dicom_spectral` labels them — `spectral_result` (`monoe`,
`iodine_density`, `iodine_no_water`, `vnc`, `z_effective`, `electron_density`, `uric_acid`,
`calcium_suppressed`), `spectral_units`, `spectral_energy_kev` — from the DICOM multi-energy
attributes (`MonoenergeticEnergyEquivalent`, multi-energy `ImageType` values) or, failing those,
the series description. A result found only in the attributes is added to the file name
(`…_monoE70keV`).

`--stack-energies` writes the monoenergetic levels of one reconstruction as a single
`X x Y x Z x E` volume ordered by keV (`…_monoE_40-70-100keV.nii.gz`, `spectral_energies_kev` in
its metadata). In the GUI it opens like a 3D+t layer with an *energy* slider in keV, and the
time–intensity curve tool plots the spectral curve (HU vs keV) of the cursor voxel or a mask.

```{note}
The description patterns follow common vendor naming and were tested on synthetic series; they
have not yet been checked against a real Philips spectral-result export. `spectral_source` in
each sidecar says whether a result came from DICOM attributes or from the description.
```

`--stack-phases` additionally writes cardiac phase reconstructions of one acquisition — e.g.
`IMR, 73%`, `IMR, 78%`, `IMR, 83%` — as one 3D+t volume ordered by R-R phase
(`IMR_CT_phases_73-78-83.nii.gz`), keeping the per-phase files. Different reconstructions
(`IMR` vs `MCR, IMR`) form separate stacks. The same is available on already-converted files:

```{code-block} python
from nvitk.io.conversors import stack_cardiac_phases

stack_cardiac_phases(sorted(Path("ES17").glob("*.nii.gz")), "ES17/phases")
```

## ANTs interop

`nvitk.io.ants_bridge` provides `require_ants`/`require_antspynet` guards and
`to_ants_image` conversion, used wherever a tool needs to hand an `Image` off to ANTsPy or
ANTsPyNet (see {doc}`registration` and {doc}`segmentation`).

```{seealso}
Full generated reference: [`nvitk.io`](../autoapi/nvitk/io/index).
```
