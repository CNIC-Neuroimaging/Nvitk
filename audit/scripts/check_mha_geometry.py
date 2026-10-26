"""A MetaImage read, reoriented and written again lands somewhere else.

read_mha stores ITK ``origin``/``direction`` next to the RAS ``affine``; write_mha prefers the
ITK keys, which geometry operations such as ``Image.orient_to`` do not update.
"""

import os
import tempfile

import numpy as np
import SimpleITK as sitk

from nvload import load

io = load("nvitk.io.imageio", stub=("nvitk", "nvitk.io.conversors"))
d = tempfile.mkdtemp()
arr = np.zeros((8, 6, 4), np.int16)
arr[1, 2, 3] = 7
src = sitk.GetImageFromArray(arr)
src.SetOrigin((10.0, 20.0, 30.0))
src.SetSpacing((1.0, 1.0, 2.0))
sitk.WriteImage(src, os.path.join(d, "a.mha"))


def world(image):
    idx = np.argwhere(np.asarray(image.data) == 7)[0]
    return (image.affine @ np.r_[idx, 1.0])[:3]


ras = io.imread(os.path.join(d, "a.mha")).orient_to("RAS")
io.imsave(os.path.join(d, "b.mha"), ras)
io.imsave(os.path.join(d, "b.nii.gz"), ras)
print("marked voxel, RAS mm, after orient_to :", world(ras))
print("after writing and reading .mha        :", world(io.imread(os.path.join(d, "b.mha"))))
print("after writing and reading .nii.gz     :", world(io.imread(os.path.join(d, "b.nii.gz"))))
