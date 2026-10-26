"""Cross-section area of a straight cylinder of known radius, as qvtpy stage 6 measures it."""

import numpy as np

from nvload import load

cs = load("nvitk.measure.cross_section")
RADIUS_MM, N = 3.0, 64
TRUE_AREA = np.pi * RADIUS_MM**2


def measured_area(spacing, axis, supersampling):
    sp = np.asarray(spacing)
    idx = np.indices((N, N, N)).astype(float)
    phys = [(idx[d] - N / 2) * sp[d] for d in range(3)]
    a, b = (d for d in range(3) if d != axis)
    seg = (phys[a] ** 2 + phys[b] ** 2) <= RADIUS_MM**2
    tangent = np.zeros(3)
    tangent[axis] = 1.0
    res = cs.cross_section_at_point(
        np.full(3, N / 2), tangent, cd=seg.astype(np.float32), voxel_spacing=tuple(spacing),
        radius_vox=10.0, cs_supersampling=supersampling, measure_resegment=False,
        volume_seg=seg.astype(np.int16), volume_label_id=1,
    )
    return res.area_mm2


print(f"true area {TRUE_AREA:.2f} mm2")
for spacing, axis in [((0.5, 0.5, 0.5), 2), ((0.5, 0.5, 0.8), 2), ((0.5, 0.5, 0.8), 0)]:
    plain, fine = measured_area(spacing, axis, False), measured_area(spacing, axis, True)
    print(f"spacing {spacing}, vessel along axis {axis}: {plain:6.2f} mm2 ({100 * (plain / TRUE_AREA - 1):+5.1f}%)"
          f" | supersampled (qvtpy default) {fine:6.2f} mm2 ({100 * (fine / TRUE_AREA - 1):+5.1f}%)")
