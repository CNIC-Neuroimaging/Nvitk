from __future__ import annotations

from pathlib import Path
from typing import Any

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import BackendUnavailableError

from .._common import reorder_axes


def write_tiff(
    path: str,
    data: Any,
    *,
    axes: str | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    try:
        import tifffile
    except Exception as exc:
        raise BackendUnavailableError('tifffile is not installed. Please install it with "pip install tifffile".') from exc

    metadata = dict(metadata or {})
    arr = to_numpy(data)

    axes_prev = metadata.get("axes")
    if axes and axes_prev and axes_prev != axes:
        arr = reorder_axes(arr, axes_prev, axes)
        metadata["axes"] = axes
    elif axes and "axes" not in metadata:
        metadata["axes"] = axes

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out), arr, metadata=metadata or None, **kwargs)
