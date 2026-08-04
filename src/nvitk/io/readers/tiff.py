"""TIFF reader via ``tifffile``: array load, inferred axes, and common TIFF tags."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvitk.core.exceptions import BackendUnavailableError

from .._common import reorder_axes


def _infer_axes(ndim: int, series_axes: str | None = None) -> str:
    """Return the TIFF series' own axis string if known, else a default order by ndim."""
    if series_axes:
        return series_axes
    if ndim == 2:
        return "YX"
    if ndim == 3:
        return "ZYX"
    if ndim == 4:
        return "TZYX"
    return "".join(f"D{i}" for i in range(ndim))


def read_tiff(path: str, *, axes: str | None = None, **_: Any):
    """
    Load a TIFF stack; returns ``(data, metadata)`` with ``axes``, ``shape``, and optional resolutions.

    *axes* may reorder via :func:`~nvitk.io._common.reorder_axes` when provided.
    """
    try:
        import tifffile
    except Exception as exc:
        raise BackendUnavailableError('tifffile is not installed. Please install it with "pip install tifffile".') from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    with tifffile.TiffFile(str(p)) as tif:
        data = tif.asarray()
        _series_axes = tif.series[0].axes if tif.series else None
        axes_prev = _infer_axes(data.ndim, _series_axes)

        metadata: dict[str, Any] = {
            "axes": axes_prev,
            "shape": tuple(data.shape),
        }

        if tif.pages:
            tags = tif.pages[0].tags
            metadata["tiff_tags"] = {tag.name: tag.value for tag in tags.values()}

            if "XResolution" in metadata["tiff_tags"] and "YResolution" in metadata["tiff_tags"]:
                xr = metadata["tiff_tags"]["XResolution"]
                yr = metadata["tiff_tags"]["YResolution"]
                if isinstance(xr, tuple) and len(xr) == 2 and xr[0] != 0:
                    metadata["x_res"] = float(xr[1]) / float(xr[0])
                if isinstance(yr, tuple) and len(yr) == 2 and yr[0] != 0:
                    metadata["y_res"] = float(yr[1]) / float(yr[0])

            description = metadata["tiff_tags"].get("ImageDescription", "")
            if isinstance(description, bytes):
                description = description.decode("utf-8", "ignore")
            if isinstance(description, str) and "spacing" in description:
                try:
                    spacing_str = description.split("spacing=")[1].split("\n")[0]
                    metadata["z_res"] = float(spacing_str)
                except Exception:
                    pass

    if axes and axes != metadata["axes"]:
        data = reorder_axes(data, metadata["axes"], axes)
        metadata["axes"] = axes
        metadata["shape"] = tuple(data.shape)

    return data, metadata
