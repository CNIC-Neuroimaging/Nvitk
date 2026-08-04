"""Nikon ND2 microscopy reader via the ``nd2`` package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvitk.core.exceptions import BackendUnavailableError

from .._common import reorder_axes

try:
    import nd2
except Exception:
    nd2 = None


def _infer_axes_from_sizes(sizes: Any, ndim: int) -> str:
    """Derive an axis-label string from ND2 dimension sizes, falling back to a default order by ndim."""
    if sizes:
        try:
            return "".join(str(k).upper() for k in sizes.keys())
        except Exception:
            pass
    if ndim == 2:
        return "YX"
    if ndim == 3:
        return "ZYX"
    if ndim == 4:
        return "TZYX"
    if ndim == 5:
        return "TCZYX"
    return "".join(f"D{i}" for i in range(ndim))


def read_nd2(path: str, *, axes: str | None = None, **_: Any):
    """Load ND2 to array with inferred axis labels (e.g. ``TCZYX``) and shape metadata."""
    if nd2 is None:
        raise BackendUnavailableError('nd2 is not installed. Please install it with "pip install nd2".')

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    with nd2.ND2File(str(p)) as nd2_file:
        data = nd2_file.asarray()
        axes_prev = _infer_axes_from_sizes(getattr(nd2_file, "sizes", None), data.ndim)

        metadata: dict[str, Any] = {
            "axes": axes_prev,
            "shape": tuple(data.shape),
            "filename": p.name,
            "dtype": str(data.dtype),
        }

        # Best-effort resolution extraction from channel calibration.
        try:
            ch = nd2_file.metadata.channels[0]
            cal = ch.volume.axesCalibration
            if len(cal) > 0:
                metadata["x_res"] = float(cal[0])
            if len(cal) > 1:
                metadata["y_res"] = float(cal[1])
            if len(cal) > 2:
                metadata["z_res"] = float(cal[2])
        except Exception:
            pass

    if axes and axes != metadata["axes"]:
        data = reorder_axes(data, metadata["axes"], axes)
        metadata["axes"] = axes
        metadata["shape"] = tuple(data.shape)

    return data, metadata
