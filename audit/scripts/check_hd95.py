"""segmentation_metrics.hd95 against the surface-based HD95 used by medpy, MONAI and the challenges."""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

from nvload import load

sm = load("nvitk.measure.segmentation_metrics")


def hd95_surface(a, b):
    cross = generate_binary_structure(3, 1)
    sa, sb = a ^ binary_erosion(a, cross), b ^ binary_erosion(b, cross)
    da, db = distance_transform_edt(~sa), distance_transform_edt(~sb)
    return float(np.percentile(np.concatenate([da[sb], db[sa]]), 95))


z, y, x = np.mgrid[:80, :80, :80]
ref = (z - 40) ** 2 + (y - 40) ** 2 + (x - 40) ** 2 < 20**2
for shift in (1, 3, 6):
    pred = (z - 40 - shift) ** 2 + (y - 40) ** 2 + (x - 40) ** 2 < 20**2
    print(f"sphere r=20 shifted {shift} voxels: nvitk hd95 {sm.hd95(ref, pred, spacing=(1, 1, 1)):.2f}"
          f" | surface-based {hd95_surface(ref, pred):.2f}")
