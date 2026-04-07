from __future__ import annotations

from pathlib import Path
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.exceptions import ValidationError

from ._common import guess_read_type, guess_write_type, reorder_axes
from .readers import read_dicom, read_mha, read_nd2, read_nifti, read_pil, read_tiff
from .writers import write_mha, write_nifti, write_pil, write_tiff

_READERS = {
    "nifti": read_nifti,
    "dicom": read_dicom,
    "tiff": read_tiff,
    "nd2": read_nd2,
    "mha": read_mha,
    "pil": read_pil,
}

_WRITERS = {
    "nifti": write_nifti,
    "tiff": write_tiff,
    "mha": write_mha,
    "pil": write_pil,
}


def imread(
    path: str | Path,
    *,
    axes: str | None = None,
    force_type: str | None = None,
    backend: str | None = None,
    **kwargs: Any,
):
    """
    Read an image from disk and return Image object(s).

    Returns:
      - Image for single image/series
      - list[Image] for multi-series DICOM sources
    """
    from nvitk.types import Image

    source = Path(path)
    read_type = guess_read_type(path, force_type=force_type)
    reader = _READERS[read_type]
    result = reader(str(path), axes=axes, **kwargs)

    if isinstance(result, list):
        out: list[Image] = []
        for data, metadata in result:
            md = dict(metadata)
            out.append(
                Image(
                    data=as_backend_array(data, backend=backend),
                    metadata=md,
                    axes=md.get("axes"),
                    name=source.stem,
                )
            )
        return out

    data, metadata = result
    md = dict(metadata)
    return Image(
        data=as_backend_array(data, backend=backend),
        metadata=md,
        axes=md.get("axes"),
        name=source.stem,
    )


def imsave(
    path: str | Path,
    image: Any,
    *,
    axes: str | None = None,
    metadata: dict[str, Any] | None = None,
    force_type: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Save image to disk in the requested output format.
    """
    data = image
    merged_meta = dict(metadata or {})

    if hasattr(image, "data"):
        data = image.data
    if hasattr(image, "metadata") and isinstance(getattr(image, "metadata"), dict):
        tmp = dict(image.metadata)
        tmp.update(merged_meta)
        merged_meta = tmp
    if hasattr(image, "axes") and getattr(image, "axes") and "axes" not in merged_meta:
        merged_meta["axes"] = getattr(image, "axes")

    if axes and merged_meta.get("axes") and merged_meta["axes"] != axes:
        data = reorder_axes(data, merged_meta["axes"], axes)
        merged_meta["axes"] = axes

    write_type = guess_write_type(path, force_type=force_type)
    writer = _WRITERS[write_type]
    writer(str(path), data, axes=axes, metadata=merged_meta, **kwargs)


def swapaxes(
    image: Any,
    axes_prev: str,
    axes_new: str,
    metadata: dict[str, Any] | None = None,
):
    data = image.data if hasattr(image, "data") else image
    out = reorder_axes(data, axes_prev, axes_new)
    if metadata is None:
        return out
    meta = dict(metadata)
    meta["axes"] = axes_new
    meta["shape"] = tuple(getattr(out, "shape", ()))
    return out, meta


def imshow(
    image: Any,
    *,
    axis: int = 0,
    index: int | str = "mid",
    show: bool = True,
    **kwargs: Any,
):
    import matplotlib.pyplot as plt

    data = image.data if hasattr(image, "data") else image
    arr = to_numpy(data)

    if arr.ndim not in (2, 3):
        raise ValidationError(f"imshow only supports 2D/3D arrays, got ndim={arr.ndim}")

    if arr.ndim == 3:
        if index == "mid":
            index = arr.shape[axis] // 2
        if not isinstance(index, int):
            raise ValidationError("index must be int or 'mid'")
        view = arr.take(indices=index, axis=axis)
    else:
        view = arr

    handle = plt.imshow(view, **kwargs)
    if show:
        plt.axis("off")
        plt.show()
    return handle


def convert_image(
    src: str | Path,
    dst: str | Path,
    *,
    src_type: str | None = None,
    dst_type: str | None = None,
    axes: str | None = None,
    backend: str | None = None,
    series_index: int = 0,
    **kwargs: Any,
) -> None:
    result = imread(src, force_type=src_type, axes=axes, backend=backend, **kwargs)

    if isinstance(result, list):
        if not (0 <= series_index < len(result)):
            raise ValidationError(f"series_index={series_index} out of range for {len(result)} series")
        image = result[series_index]
    else:
        image = result

    imsave(dst, image, force_type=dst_type, axes=axes, **kwargs)
