"""4D-flow background correction: the fit used by phase2volume vs the correct one in the same file.

``fit_polynomial_background_3vector`` and ``fit_polynomial_background_scalar`` pair each voxel
value with the coordinates of a different voxel whenever the grid is not cubic. A known smooth
background is added to noise and each fit is asked to recover it.
"""

import numpy as np

from nvload import load

bg = load("nvitk.io.conversors._phase2volume_bg")
for shape in [(40, 32, 24), (32, 32, 32)]:
    i, j, k = np.meshgrid(*(np.linspace(-1, 1, n) for n in shape), indexing="ij")
    truth = 5.0 * i + 3.0 * j**2 - 2.0 * k
    noisy = truth + np.random.default_rng(1).normal(0, 0.5, shape)
    zeros = np.zeros(shape)
    used, _, _ = bg.fit_polynomial_background_3vector(noisy, zeros, zeros, spatial_order=2, static_percentile=100.0)
    right = bg.polynomial_background(noisy, order=2, static_percentile=100.0, signed=True)
    rms = lambda a: float(np.sqrt(np.mean(a**2)))
    print(f"grid {shape}: background RMS {rms(truth):.2f} mm/s | residual after phase2volume fit "
          f"{rms(used - truth):.3f} | after polynomial_background {rms(right - truth):.3f}")
