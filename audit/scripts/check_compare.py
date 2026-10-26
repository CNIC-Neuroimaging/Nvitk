"""measure.compare.sample_at_physical_points raises on any input (ragged homogeneous coordinates)."""

import numpy as np

from nvload import load

compare = load("nvitk.measure.compare")
try:
    compare.sample_at_physical_points(np.zeros((10, 10, 10)), np.eye(4), np.array([[1.0, 2.0, 3.0]]))
    print("sampled without error")
except ValueError as exc:
    print("ValueError:", str(exc).splitlines()[0][:90])
