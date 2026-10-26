# nvitk audit (October 2026)

Two parts, on the `feature/audit` branch, starting from commit `ec706c1`:

1. [**Anti-slop audit**](01-anti-slop-audit.md): comments and docstrings rewritten to read like
   a person wrote them, code-quality findings (Anti-slop, Thermos, Karpathy guidelines), and
   three loops made faster without changing a single output.
2. [**Methodological audit**](02-methodological-audit.md): bugs, method flaws and
   state-of-the-art options across the imaging and mesh code, ranked by severity, each with the
   file and line and, where possible, a script that reproduces it.

## What changed in the code

* Comments and docstrings in 556 Python files, plus `pyproject.toml` and the `recipe/` scripts.
  Proof that no code changed: `python audit/scripts/check_code_unchanged.py ec706c1 --allow
  src/nvitk/filters/sliding_threshold.py src/nvitk/measure/segmentation_metrics.py`
  (554 files code-identical, 2 intended edits, 0 unexpected).
* Faster, bit-identical implementations of the sliding threshold
  (`filters/sliding_threshold.py`) and of component matching and label adjacency
  (`measure/segmentation_metrics.py`), with `tests/test_output_preservation.py` comparing them
  against the previous loops.
* The Slicer vendored copy regenerated from the cleaned sources (`vendor_sync.py --check` passes).

Nothing from the methodological audit has been fixed: every fix there changes numbers.

## Most important findings

| ID | Finding | Effect |
|---|---|---|
| M-01 | 4D-flow background-phase fit uses scrambled voxel coordinates on non-cubic grids | Default qvtpy stage 0; residual 1.54 vs 0.01 mm/s on a test background |
| M-02 | Stage-6 flows use uncorrected phase | Eddy-current offsets stay in every flow, PI and PWV |
| M-03 / M-04 | Cross-section area bias: supersampling step (+7.5 to +15 %) and heuristic pixel size on anisotropic grids (+61 %) | Flow = velocity x area inherits it |
| M-05 / M-06 | Velocity frame assumed axis-aligned; mean flow and PI from per-frame absolute values | Wrong components on angulated slabs; upward bias in low-flow vessels |
| M-09 | HD95 computed on all voxels, not surfaces | 1.41 instead of 3.00 for a 3-voxel shift |
| M-10 / M-11 / M-12 | Geometry kept in three places: NIfTI sform vs pixdim, stale MHA origin, crops keep the parent affine | Same file read with different spacing by nibabel and ITK; misplaced outputs |
| M-13 | Patient name, ID, birth date and accession number written into derived NIfTIs | Data-protection exposure |
| M-14 / M-15 | DICOM: mixed-orientation slices merged by in-plane rotation; per-slice rescale factors applied to the wrong slices | Fabricated volumes; wrong intensities with `revert_scaling` / Philips FP |
| M-16 | SUV ignores `DecayCorrection` and `Units` | +46 % SUV for ADMIN-corrected FDG at 60 min |
| M-17 | `nvitk-topbrain-infer` (local) raises `TypeError` | Local inference unusable |

## Reproducing the evidence

```bash
cd audit/scripts
python check_background_polyfit.py
python check_cross_section_area.py
python check_hd95.py
python check_nifti_header.py
python check_mha_geometry.py
python check_image_slicing.py
python check_dicom.py
python check_compare.py
```

The scripts load single modules from `src/` (see `nvload.py`), so NumPy, SciPy, nibabel,
SimpleITK, pydicom, dicom2nifti and scikit-image are enough. `deslop.py` is the comment rewriter
used for part 1; run it without `--write` on new files to see what it would change.
