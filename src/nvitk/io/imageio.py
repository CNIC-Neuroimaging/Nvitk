"""
High-level image load/save and helpers (:func:`imread`, :func:`imsave`, :func:`imshow`, axis tools).

Readers and writers are selected by extension or ``force_type``; see :mod:`nvitk.io._common`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

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


# ──────────────────────────────────────────────────────────────────────────────
# Load / save
# ──────────────────────────────────────────────────────────────────────────────


def imread(
    path: str | Path,
    *,
    axes: str | None = None,
    force_type: str | None = None,
    backend: str | None = None,
    **kwargs: Any,
):
    """
    Load a volume or series from disk into :class:`~nvitk.types.image.Image`.

    Parameters
    ----------
    path
        File or directory (directories are treated as DICOM folders unless ``force_type`` overrides).
    axes
        If set, reorder reader output to this axis string (reader-dependent).
    force_type
        Skip extension sniffing: ``nifti``, ``dicom``, ``tiff``, ``mha``, ``pil``, ``nd2``, …
    backend
        ``numpy`` or ``cupy`` for the returned :class:`~nvitk.types.image.Image` array.
    **kwargs
        Passed to the format reader (e.g. DICOM ``force_ras``, NIfTI ``metadata_json``).

    Returns
    -------
    Image or list[Image]
        A single image, or a list when the reader returns multiple series (e.g. DICOM).
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
                    orientation=md.get("orientation"),
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
        orientation=md.get("orientation"),
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
    Write array or :class:`~nvitk.types.image.Image` to *path* using the writer chosen by extension or *force_type*.

    Merges ``image.metadata`` with *metadata*; optional *axes* reorders data before write.
    """
    data = image
    merged_meta = dict(metadata or {})

    if hasattr(image, "data") and isinstance(image, Image):
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


# ──────────────────────────────────────────────────────────────────────────────
# Axis helpers
# ──────────────────────────────────────────────────────────────────────────────


def swapaxes(
    image: Any,
    axes_prev: str,
    axes_new: str,
    metadata: dict[str, Any] | None = None,
):
    """
    Reorder *image* from *axes_prev* to *axes_new* using :func:`~nvitk.io._common.reorder_axes`.

    If *metadata* is omitted, returns only the reordered array. If provided, returns
    ``(data, metadata)`` with ``axes`` and ``shape`` updated.
    """
    data = image.data if hasattr(image, "data") else image
    out = reorder_axes(data, axes_prev, axes_new)
    if metadata is None:
        return out
    meta = dict(metadata)
    meta["axes"] = axes_new
    meta["shape"] = tuple(getattr(out, "shape", ()))
    return out, meta


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (matplotlib)
# ──────────────────────────────────────────────────────────────────────────────


def _parse_index_token(index: Any, axis: int) -> str | None:
    """Extract a lower-cased string token (e.g. ``\"max\"``, ``\"rgb\"``) from *index* for the given *axis*, if present."""
    if isinstance(index, str):
        return index.strip().lower()
    if isinstance(index, (tuple, list)) and len(index) > axis and isinstance(index[axis], str):
        return str(index[axis]).strip().lower()
    return None


def _is_projection_token(tok: str | None) -> bool:
    """True when *tok* names a supported intensity projection (max/mean/avg/median)."""
    return tok is not None and tok in ("max", "mean", "avg", "median")


def _project_along_axis(vol: np.ndarray, axis: int, how: str) -> np.ndarray:
    """Collapse *vol* along *axis* with the named projection (max/mean/avg/median)."""
    how = how.lower()
    if how == "max":
        return np.max(vol, axis=axis)
    if how in ("mean", "avg"):
        return np.mean(vol, axis=axis)
    if how == "median":
        return np.median(vol, axis=axis)
    raise ValueError(f"Unknown projection {how!r}")


def _rgb_along_axis(vol: np.ndarray, axis: int) -> np.ndarray:
    """Move a length-3 *axis* to the end so it can be shown as RGB by matplotlib."""
    if vol.shape[axis] != 3:
        raise ValueError(
            f"index='rgb' requires length 3 along axis {axis}, got shape {vol.shape[axis]}"
        )
    return np.moveaxis(vol, axis, -1)


def _kwargs_imshow_rgb(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop ``cmap`` from imshow kwargs (invalid/ignored for RGB data)."""
    return {k: v for k, v in kwargs.items() if k != "cmap"}


def _standard_2d_view(
    vol: np.ndarray,
    axis: int,
    index: Any,
    resolve_index,
) -> tuple[np.ndarray, bool]:
    """
    Reduce *vol* to a 2D array for a single ``imshow`` call.

    Returns
    -------
    view : ndarray
    rgb_layout
        If True, omit ``cmap`` (RGB / multi-channel image).
    """
    tok = _parse_index_token(index, axis)
    if vol.ndim < 3:
        return vol, False
    if _is_projection_token(tok):
        if vol.ndim != 3:
            raise ValueError(
                f"Projection index {tok!r} requires a 3D volume; got ndim={vol.ndim}"
            )
        return _project_along_axis(vol, axis, tok), False
    if tok == "rgb":
        if vol.ndim != 3:
            raise ValueError(f"index='rgb' requires a 3D volume; got ndim={vol.ndim}")
        return _rgb_along_axis(vol, axis), True
    idx_raw = index if isinstance(index, (int, str)) else index[axis]
    idx = resolve_index(idx_raw, vol.shape[axis])
    return np.take(vol, idx, axis=axis), False


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
    """
    Display a 2D slice, orthogonal triple, mosaic of slices, or animation via matplotlib.

    *display* selects layout: ``standard`` (single slice), ``orth`` / ``orthogonal`` (three
    linked views), ``mosaic`` (grid of slices along *axis*), or ``animation`` (time or slice
    sweep). Pass ``cmap`` and other ``imshow`` kwargs as needed; default colormap is ``gray``;
    omit or override ``cmap`` for RGB output.

    Parameters
    ----------
    image
        Array or :class:`~nvitk.types.image.Image` with optional ``axes`` (used when *axis* is None).
    axis
        Dimension index along which to slice, mosaic, project, or stack RGB; if None, uses the
        ``Z`` axis index when present, else 2.
    index
        Per-axis slice index, ``"mid"``, a triple for orthogonal mode, a **projection** along
        *axis* (``"max"``, ``"mean"``, ``"avg"``, ``"median"``), or ``"rgb"`` when that axis has
        length 3 (last dimension becomes channels for ``imshow``). Projection and RGB use a
        single 2D panel (orthogonal and mosaic layouts fall back to standard for those modes).
    show
        If True, ``plt.show()`` (or JS HTML in Jupyter for animations).
    display
        Layout name; aliases include ``orth``, ``mosaic``, ``animation``, ``animation+orth``, etc.
    mosaic_init, mosaic_fin
        Inclusive start / exclusive end slice indices for mosaic mode.
    t_axis
        Time dimension for 4D animation when *display* is ``animation``.
    **kwargs
        Forwarded to ``imshow`` (e.g. ``vmin``, ``vmax``, ``cmap``).

    Returns
    -------
    list or matplotlib.animation.FuncAnimation
        Artist handles from static views, or a ``FuncAnimation`` when *display* is ``animation``.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec
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
    if display is None and arr.ndim == 3: spatial_mode = "orthogonal"
    elif display is None: spatial_mode = "standard"
    elif display == "animation": spatial_mode = "standard"
    elif display in ["animation+orth", "animation+orthogonal"]: spatial_mode = "orthogonal"
    elif display in ["animation+mosaic"]: spatial_mode = "mosaic"
    elif display in ['orth', 'orthogonal', 'full']: spatial_mode = 'orthogonal'
    elif display in ['mosaic', 'series']: spatial_mode = 'mosaic'
    elif display in ['standard', 'default']: spatial_mode = 'standard'
    else: raise ValueError(f"Invalid display mode: {display}")

    _idx_tok = _parse_index_token(index, axis)
    if _is_projection_token(_idx_tok) or _idx_tok == "rgb":
        if spatial_mode in ("orth", "orthogonal", "mosaic"):
            spatial_mode = "standard"

    # 3. Temporal vs Spatial separation
    if display == "animation":
        if arr.ndim == 4:
            num_frames = arr.shape[t_axis]
            spatial_arr = arr.take(indices=0, axis=t_axis)
        elif arr.ndim == 3:
            num_frames = arr.shape[axis]
            spatial_arr = arr
        else:
            raise ValueError("Animation requires 3D or 4D data.")
        if (_is_projection_token(_idx_tok) or _idx_tok == "rgb") and arr.ndim == 3:
            num_frames = 1
    else:
        spatial_arr = arr

    if "cmap" not in kwargs:
        kwargs["cmap"] = "gray"

    def resolve_index(idx, max_len):
        """Resolve an index token: ``\"mid\"`` → the middle slice, else pass *idx* through unchanged."""
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
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        rem = [i for i in range(3) if i != axis]
        axis_top, axis_right = rem[0], rem[1]

        def get_orth_slices(vol):
            """Extract the (top, main, right) orthogonal slice views of *vol* for the current axis layout."""
            if axis == 2:
                return vol[indices[axis_top], :, :].T, vol[:, :, indices[axis]], vol[:, indices[axis_right], :]
            elif axis == 1:
                return vol[:, :, indices[axis_top]], vol[:, indices[axis], :], vol[:, :, indices[axis_right]]
            else: # axis 0
                return vol[:, indices[axis_top], :], vol[indices[axis], :, :], vol[indices[axis_right], :, :]

        top_v, main_v, right_v = get_orth_slices(spatial_arr)
        handles.append(ax_top.imshow(top_v, aspect='auto', **kwargs))
        handles.append(ax_main.imshow(main_v, aspect='auto', **kwargs))
        handles.append(ax_right.imshow(right_v, aspect='auto', **kwargs))
        for a in [ax_main, ax_top, ax_right]: a.axis("off")

    # CASE: STANDARD (2D, slice, projection, or RGB)
    else:
        vol_work = spatial_arr
        if vol_work.ndim == 4 and display != "animation":
            vol_work = vol_work.take(indices=0, axis=t_axis)
        if vol_work.ndim >= 3:
            view, rgb_layout = _standard_2d_view(vol_work, axis, index, resolve_index)
            im_kw = _kwargs_imshow_rgb(kwargs) if rgb_layout else kwargs
        else:
            view = vol_work
            im_kw = kwargs
        fig, ax = plt.subplots()
        handles = [ax.imshow(view, **im_kw)]
        ax.axis("off")

    # --- BLOCK: ANIMATION EXECUTION ---
    if display == "animation":
        def update(frame):
            """Matplotlib ``FuncAnimation`` callback: redraw the view(s) for the given time *frame*."""
            if arr.ndim == 4:
                vol = arr.take(indices=frame, axis=t_axis)
            else:
                vol = arr

            if spatial_mode == "mosaic":
                for i, s_idx in enumerate(slice_indices):
                    handles[i].set_data(vol.take(indices=s_idx, axis=axis))
            elif spatial_mode in ("orth", "orthogonal"):
                # If animating a 3D array, update the current orth slice dynamically
                if arr.ndim == 3:
                    indices[axis] = frame
                t_v, m_v, r_v = get_orth_slices(vol)
                handles[0].set_data(t_v)
                handles[1].set_data(m_v)
                handles[2].set_data(r_v)
            else:
                # Standard update (slice, projection, or RGB per frame)
                up_view = vol
                tok = _parse_index_token(index, axis)
                if vol.ndim == 3 and (_is_projection_token(tok) or tok == "rgb"):
                    up_view, _rgb = _standard_2d_view(vol, axis, index, resolve_index)
                elif vol.ndim == 3:
                    if arr.ndim == 3:
                        idx = frame
                    else:
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


# ──────────────────────────────────────────────────────────────────────────────
# Format conversion
# ──────────────────────────────────────────────────────────────────────────────


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
    """
    Read *src* with :func:`imread`, pick one series if multiple, then :func:`imsave` to *dst*.

    Use *series_index* when the reader returns a list (e.g. multi-series DICOM). *src_type* /
    *dst_type* override extension-based format detection; remaining *kwargs* go to read and write.
    """
    result = imread(src, force_type=src_type, axes=axes, backend=backend, **kwargs)

    if isinstance(result, list):
        if not (0 <= series_index < len(result)):
            raise ValidationError(f"series_index={series_index} out of range for {len(result)} series")
        image = result[series_index]
    else:
        image = result

    imsave(dst, image, force_type=dst_type, axes=axes, **kwargs)
