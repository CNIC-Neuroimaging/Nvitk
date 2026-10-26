"""Cropping an Image by indexing keeps the parent affine, so the crop is saved in the wrong place."""

import os
import tempfile

import numpy as np

from nvload import load

io = load("nvitk.io.imageio", stub=("nvitk", "nvitk.io.conversors"))
Image = load("nvitk.types.image").Image
data = np.zeros((20, 20, 20), np.int16)
data[12, 7, 5] = 1
img = Image(data, metadata={"affine": np.diag([0.5, 0.5, 2.0, 1.0]), "x_res": 0.5, "y_res": 0.5, "z_res": 2.0})


def world(image):
    idx = np.argwhere(np.asarray(image.data) == 1)[0]
    return (image.affine @ np.r_[idx, 1.0])[:3]


crop = img[10:20, 5:15, 4:10]
path = os.path.join(tempfile.mkdtemp(), "crop.nii.gz")
io.imsave(path, crop)
print("marked voxel (mm) in the full image :", world(img))
print("same voxel in img[10:20, 5:15, 4:10]:", world(crop))
print("after imsave + imread of the crop   :", world(io.imread(path)))
