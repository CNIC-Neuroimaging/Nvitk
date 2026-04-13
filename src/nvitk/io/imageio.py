from __future__ import annotations

from pathlib import Path
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.exceptions import ValidationError
from nvitk.types.image import Image

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


# def imshow(
#     image: Any,
#     *,
#     axis: int = 0,
#     index: int | str = "mid",
#     show: bool = True,
#     **kwargs: Any,
# ):
#     import matplotlib.pyplot as plt

#     data = image.data if hasattr(image, "data") else image
#     arr = to_numpy(data)

#     if arr.ndim not in (2, 3):
#         raise ValidationError(f"imshow only supports 2D/3D arrays, got ndim={arr.ndim}")

#     if arr.ndim == 3:
#         if index == "mid":
#             index = arr.shape[axis] // 2
#         if not isinstance(index, int):
#             raise ValidationError("index must be int or 'mid'")
#         view = arr.take(indices=index, axis=axis)
#     else:
#         view = arr

#     if 'cmap' not in kwargs: kwargs['cmap'] = 'gray'
#     handle = plt.imshow(view, **kwargs)
#     if show:
#         plt.axis("off")
#         plt.show()
#     return handle

from typing import Any

def imshow(
    image,
    *,
    axis: int = None,
    index: tuple | int | str = "mid",
    show: bool = True,
    display: str = None,  # 'standard', 'orth', 'mosaic', 'animation'
    mosaic_init: int = 0,
    mosaic_fin: int | None = None,
    t_axis: int = 3,
    **kwargs,
):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec
    import numpy as np
    import math

    # 1. Data Prep
    data = image.data if hasattr(image, "data") else image
    arr = to_numpy(data)

    # Handle default axis if None
    if axis is None:
        if hasattr(image, 'axes') and 'Z' in image.axes:
            axis = image.axes.index('Z')
        else:
            axis = 2

    # 2. Setup Dimensions
    spatial_mode = display
    if display is None and arr.ndim == 3: spatial_mode = "standard"
    elif display is None: spatial_mode = "standard"
    elif display == "animation": spatial_mode = "standard"
    elif display in ["animation+orth", "animation+orthogonal"]: spatial_mode = "orthogonal"
    elif display in ["animation+mosaic"]: spatial_mode = "mosaic"
    elif display in ['orth', 'orthogonal', 'full']: spatial_mode = 'orthogonal'
    elif display in ['mosaic', 'series']: spatial_mode = 'mosaic'
    elif display in ['standard', 'default']: spatial_mode = 'standard'
    else: raise ValueError(f"Invalid display mode: {display}")
    
    # 3. Temporal vs Spatial separation
    if display == "animation":
        num_frames = arr.shape[t_axis]
        spatial_arr = arr.take(indices=0, axis=t_axis)
    else:
        spatial_arr = arr

    if "cmap" not in kwargs:
        kwargs["cmap"] = "gray"

    def resolve_index(idx, max_len):
        if idx == "mid": return max_len // 2
        return idx

    handles = []
    fig = None

    # --- BLOCK: FIGURE SETUP ---    
    # CASE: MOSAIC
    if spatial_mode == "mosaic":
        max_slices = spatial_arr.shape[axis]
        start = resolve_index(mosaic_init, max_slices)
        end = max_slices if mosaic_fin is None else resolve_index(mosaic_fin, max_slices)
        num_slices = end - start
        
        cols = math.ceil(math.sqrt(num_slices))
        rows = math.ceil(num_slices / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.atleast_1d(axes).flatten()
        
        slice_indices = list(range(start, end))
        for i, s_idx in enumerate(slice_indices):
            view = spatial_arr.take(indices=s_idx, axis=axis)
            handles.append(axes[i].imshow(view, **kwargs))
            axes[i].axis("off")
        for j in range(num_slices, len(axes)): axes[j].axis("off")

    # CASE: ORTHOGONAL
    elif spatial_mode in ("orth", "orthogonal"):
        if spatial_arr.ndim != 3:
            if arr.ndim == 4:
                spatial_arr = arr.take(indices=0, axis=t_axis)
            else:
                raise ValueError("Orthogonal view requires 3D spatial data.")

        if isinstance(index, (int, str)):
            indices = [resolve_index("mid", spatial_arr.shape[i]) for i in range(3)]
            indices[axis] = resolve_index(index, spatial_arr.shape[axis])
        else:
            indices = [resolve_index(idx, spatial_arr.shape[i]) for i, idx in enumerate(index)]

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
        
        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharey=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        rem = [i for i in range(3) if i != axis]
        axis_top, axis_right = rem[0], rem[1]

        def get_orth_slices(vol):
            if axis == 2:
                return vol[indices[axis_top], :, :].T, vol[:, :, indices[axis]], vol[:, indices[axis_right], :]
            elif axis == 1:
                return vol[:, indices[axis_top], :], vol[:, indices[axis], :], vol[:, :, indices[axis_right]]
            else: # axis 0
                return vol[:, indices[axis_top], :], vol[indices[axis], :, :], vol[:, :, indices[axis_right]]

        top_v, main_v, right_v = get_orth_slices(spatial_arr)
        handles.append(ax_top.imshow(top_v, aspect='auto', **kwargs))
        handles.append(ax_main.imshow(main_v, aspect='auto', **kwargs))
        handles.append(ax_right.imshow(right_v, aspect='auto', **kwargs))
        for a in [ax_main, ax_top, ax_right]: a.axis("off")

    # CASE: STANDARD (2D or single slice)
    else:
        view = spatial_arr
        if spatial_arr.ndim == 3:
            idx = resolve_index(index if isinstance(index, (int, str)) else index[axis], spatial_arr.shape[axis])
            view = spatial_arr.take(indices=idx, axis=axis)
        
        fig, ax = plt.subplots()
        handles = [ax.imshow(view, **kwargs)]
        ax.axis("off")

    # --- BLOCK: ANIMATION EXECUTION ---
    if display == "animation":
        def update(frame):
            vol = arr.take(indices=frame, axis=t_axis)
            if spatial_mode == "mosaic":
                for i, s_idx in enumerate(slice_indices):
                    handles[i].set_data(vol.take(indices=s_idx, axis=axis))
            elif spatial_mode in ("orth", "orthogonal"):
                t_v, m_v, r_v = get_orth_slices(vol)
                handles[0].set_data(t_v)
                handles[1].set_data(m_v)
                handles[2].set_data(r_v)
            else:
                # Standard update
                up_view = vol
                if vol.ndim == 3:
                    idx = resolve_index(index if isinstance(index, (int, str)) else index[axis], vol.shape[axis])
                    up_view = vol.take(indices=idx, axis=axis)
                handles[0].set_data(up_view)
            return handles

        ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True, interval=100)
        if show: 
            try: # Running in Jupyter
                from IPython.display import display, HTML
                display(HTML(ani.to_jshtml()))
                plt.close()
            except Exception:
                plt.show()
        return ani

    if show:
        if spatial_mode == "mosaic": 
            plt.tight_layout()
        plt.show()
    return handles


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
