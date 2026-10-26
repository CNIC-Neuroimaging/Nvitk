"""When the affine and the voxel-size metadata disagree, nibabel and ITK read different geometry.

write_nifti builds the sform from ``metadata['affine']`` (identity when absent) and pixdim from
``x_res``/``y_res``/``z_res``. nibabel and FSL follow the sform; ITK-based tools (SimpleITK, ANTs,
ITK-SNAP, Slicer) fall back to pixdim when the sform scales look inconsistent.
"""

import os
import tempfile

import nibabel as nib
import numpy as np
import SimpleITK as sitk

path = os.path.join(tempfile.mkdtemp(), "t.nii.gz")
img = nib.Nifti1Image(np.zeros((4, 5, 6), np.float32), np.diag([2.0, 2.0, 3.0, 1.0]))
img.header.set_zooms((0.5, 0.5, 0.5))  # what write_nifti does when x_res disagrees with the affine
nib.save(img, path)
sitk.ProcessObject_SetGlobalWarningDisplay(False)
print("nibabel spacing  :", tuple(float(v) for v in np.linalg.norm(nib.load(path).affine[:3, :3], axis=0)))
print("SimpleITK spacing:", sitk.ReadImage(path).GetSpacing())
print("extension code 16 is registered as:", nib.nifti1.extension_codes.label[16])
