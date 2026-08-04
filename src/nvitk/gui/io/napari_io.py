"""Load images into Napari via :mod:`nvitk.io` (NIfTI, DICOM, TIFF, MHA, ND2, …)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.gui.core.orientation import (
    configure_viewer_for_layer,
    prepare_for_napari,
    suppress_nonorthogonal_slice_warning,
)
from nvitk.gui.core.warnings import install_napari_display_warnings

install_napari_display_warnings()
from nvitk.io import imread
from nvitk.io._common import default_nifti_axes, guess_read_type
from nvitk.types import Image

_NVITK_OPEN_SUFFIXES = frozenset({
    ".nii", ".nii.gz", ".mha", ".mhd", ".tif", ".tiff", ".nd2",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".dcm",
})

LayerData = tuple[Any, dict[str, Any], str]
ReaderFunc = Callable[[str], list[LayerData] | None]


def _normalize_paths(path: str | Path | Sequence[str | Path]) -> list[Path]:
    """Coerce a single path or a sequence of paths into a list of :class:`Path` objects."""
    if isinstance(path, (str, Path)):
        return [Path(path)]
    return [Path(p) for p in path]


def _nvitk_can_open(path: Path) -> bool:
    """True if *path* is a directory (assumed DICOM series) or a file nvitk's readers recognize by
    extension or content sniffing."""
    if not path.exists():
        return False
    if path.is_dir():
        return True
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return True
    if path.suffix.lower() in _NVITK_OPEN_SUFFIXES:
        return True
    try:
        return guess_read_type(path) in ("nifti", "dicom", "tiff", "mha", "pil", "nd2")
    except Exception:
        return False


def _resolution_for_axis(md: dict[str, Any], axis_char: str) -> float | None:
    """Voxel/frame resolution for *axis_char* (``X``/``Y``/``Z``/``T``/``C``) from image metadata
    *md*, or ``None`` if not recorded."""
    key = {
        "X": "x_res",
        "Y": "y_res",
        "Z": "z_res",
        "T": "t_res",
        "C": "t_res",
    }.get(axis_char.upper())
    if key is None:
        return None
    val = md.get(key)
    if val is None and axis_char.upper() in ("T", "C"):
        val = md.get("temporal_resolution")
    if val is None:
        return None
    return float(val)


def _napari_scale(img: Image, ndim: int) -> tuple[float, ...] | None:
    """Per-array-axis scale aligned with ``img.axes`` (e.g. XYZT → x,y,z,t)."""
    axes = (img.axes or default_nifti_axes(ndim)).upper()
    if len(axes) != ndim:
        axes = default_nifti_axes(ndim)
    md = img.metadata or {}
    vals = []
    for ch in axes:
        r = _resolution_for_axis(md, ch)
        if r is None:
            return None
        vals.append(r)
    if len(vals) < ndim:
        vals.extend([1.0] * (ndim - len(vals)))
    return tuple(vals[:ndim])


def _napari_affine(img: Image) -> np.ndarray | None:
    """Raw 4x4 voxel-to-world matrix from the file (before display reorientation)."""
    aff = img.affine
    if aff is None:
        return None
    aff = to_numpy(aff).astype(float)
    if aff.shape != (4, 4):
        return None
    return aff


def _nvitk_layer_metadata(
    img: Image,
    path: Path,
    *,
    affine_source: np.ndarray | None,
) -> dict[str, Any]:
    """Napari ``metadata`` dict (nested nvitk fields only — no invalid layer kwargs)."""
    nvitk_md = dict(img.metadata) if img.metadata else {}
    nvitk_md["source"] = str(path)
    try:
        src_type = guess_read_type(path)
        if src_type:
            nvitk_md["source_type"] = src_type
    except Exception:
        pass
    if affine_source is not None:
        nvitk_md["affine_source"] = to_numpy(affine_source).astype(float)
    out = {"nvitk_metadata": nvitk_md}
    if img.axes:
        out["axes"] = img.axes
    return out


def _axis_labels_for_image(img: Image, ndim: int) -> tuple[str, ...]:
    """*img*'s axis labels if they match *ndim*, else the default NIfTI axis order for *ndim*."""
    axes = img.axes or default_nifti_axes(ndim)
    if len(axes) == ndim:
        return tuple(axes)
    return tuple(default_nifti_axes(ndim))


def _prepare_layer_tuple(img: Image, path: Path) -> LayerData:
    """Build a napari-plugin-style ``(data, layer_kwargs, layer_type)`` tuple for *img*, reorienting
    the array for display and computing scale/affine, axis labels, and nvitk metadata from *path*."""
    data = to_numpy(img.data)
    raw_affine = _napari_affine(img)
    axis_labels = _axis_labels_for_image(img, data.ndim)
    axes_str = "".join(axis_labels)
    data, affine, scale = prepare_for_napari(
        data,
        raw_affine,
        axes=axes_str,
        metadata=img.metadata,
    )
    layer_meta = {
        "name": img.name or path.stem,
        "metadata": _nvitk_layer_metadata(img, path, affine_source=raw_affine),
        "axis_labels": _axis_labels_for_image(img, data.ndim),
    }
    if data.ndim > 3:
        sc = scale or _napari_scale(img, data.ndim)
        if sc is not None:
            layer_meta["scale"] = tuple(sc[: data.ndim])
    elif affine is not None:
        layer_meta["affine"] = affine
    elif scale is not None:
        nd = min(len(scale), data.ndim)
        layer_meta["scale"] = tuple(scale[:nd])
    else:
        sc = _napari_scale(img, data.ndim)
        if sc is not None:
            layer_meta["scale"] = sc
    return (data, layer_meta, "image")


def _read_layer_data(path: str) -> list[LayerData] | None:
    """Read one path into Napari layer tuples."""
    pth = Path(path)
    if not _nvitk_can_open(pth):
        return None
    try:
        result = imread(pth, backend="numpy")
    except Exception:
        return None
    images = result if isinstance(result, list) else [result]
    out = []
    for i, img in enumerate(images):
        if not img.name:
            img.name = f"{pth.stem}_{i}" if len(images) > 1 else pth.stem
        out.append(_prepare_layer_tuple(img, pth))
    return out or None


def read_paths(path: str | list[str]) -> ReaderFunc | list[LayerData] | None:
    """
    Napari npe2 reader command.

    When called with a single path string (npe2 command exec), returns a reader
    function. When that function is invoked, returns layer data tuples.
    """
    if isinstance(path, list):
        layer_data = []
        for p in path:
            chunk = _read_layer_data(p)
            if chunk:
                layer_data.extend(chunk)
        return layer_data or None

    if not _nvitk_can_open(Path(path)):
        return None
    return _read_layer_data


def _add_image_to_viewer(viewer: Any, img: Image, path: Path) -> Any:
    """Add *img* to *viewer* as an Image layer using :func:`_prepare_layer_tuple`'s display kwargs,
    then apply nvitk's viewer/dims configuration for the new layer."""
    data, layer_meta, _ = _prepare_layer_tuple(img, path)
    kwargs = {
        "name": layer_meta["name"],
        "metadata": layer_meta["metadata"],
    }
    if "scale" in layer_meta:
        kwargs["scale"] = layer_meta["scale"]
    elif "affine" in layer_meta:
        kwargs["affine"] = layer_meta["affine"]
    if "axis_labels" in layer_meta:
        kwargs["axis_labels"] = layer_meta["axis_labels"]
    with suppress_nonorthogonal_slice_warning():
        layer = viewer.add_image(data, **kwargs)
    configure_viewer_for_layer(
        viewer,
        layer,
        radiological=False,
        configure_dims=len(viewer.layers) <= 1,
    )
    return layer


def open_paths_with_nvitk(
    viewer: Any,
    paths: str | Path | Sequence[str | Path],
    *,
    stack = False,
    force_type: str | None = None,
) -> list[Any]:
    """Open one or more paths with nvitk.io and add Napari layers.

    *force_type* is forwarded to :func:`~nvitk.io.imread` (e.g. ``\"nifti\"`` when
    opening a directory that must not be treated as DICOM).
    """
    _ = stack
    path_list = _normalize_paths(paths)
    layers = []

    for path in path_list:
        if not _nvitk_can_open(path):
            continue
        try:
            result = imread(path, backend="numpy", force_type=force_type)
        except Exception as exc:
            _notify_error(f"Could not read {path} with nvitk.io:\n{exc}")
            continue

        images = result if isinstance(result, list) else [result]
        for i, img in enumerate(images):
            suffix = f"_{i}" if len(images) > 1 else ""
            if not img.name:
                img.name = f"{path.stem}{suffix}"
            layers.append(_add_image_to_viewer(viewer, img, path))

    return layers


def _notify_error(message: str) -> None:
    """Show *message* via Napari's error notification, falling back to printing it if Napari's UI
    isn't available."""
    try:
        from napari.utils.notifications import show_error
        show_error(message)
    except Exception:
        print(message, flush=True)


def _is_nvitk_layer(layer: Any) -> bool:
    """True if *layer* was loaded through nvitk's reader (carries an ``nvitk_metadata`` entry)."""
    meta = getattr(layer, "metadata", None) or {}
    if isinstance(meta, dict) and "nvitk_metadata" in meta:
        return True
    nv = meta.get("nvitk_metadata") if isinstance(meta, dict) else None
    return isinstance(nv, dict)


def _layer_from_list_event(event: Any) -> Any | None:
    """Layer instance from Napari ``inserted`` / legacy ``added`` events."""
    value = getattr(event, "value", None)
    if value is not None and hasattr(value, "data"):
        return value
    source = getattr(event, "source", None)
    if isinstance(source, (list, tuple)):
        for item in reversed(source):
            if hasattr(item, "data"):
                return item
    return None


def _on_nvitk_layer_inserted(viewer: Any, event: Any) -> None:
    """Configure viewer dims/orientation for a newly inserted layer: repairs the time-dim range for
    overlays and 4D+ layers, and applies nvitk's display setup for 4D+ or nvitk-sourced layers."""
    from nvitk.gui.core.orientation import configure_viewer_for_layer
    from nvitk.gui.viz.layers import repair_time_dim_for_viewer

    layer = _layer_from_list_event(event)
    if layer is None:
        return
    if type(layer).__name__ in ("Vectors", "Points", "Shapes", "Surface"):
        # Overlays right-align onto a 4D layer's time axis; restore the real count.
        repair_time_dim_for_viewer(viewer)
        return
    data = getattr(layer, "data", None)
    ndim = int(getattr(data, "ndim", 0) or 0)
    configure_dims = len(viewer.layers) <= 1
    if ndim > 3:
        configure_viewer_for_layer(
            viewer, layer, radiological=False, configure_dims=configure_dims
        )
        repair_time_dim_for_viewer(viewer)
        return
    if not _is_nvitk_layer(layer):
        repair_time_dim_for_viewer(viewer)
        return
    configure_viewer_for_layer(
        viewer, layer, radiological=False, configure_dims=configure_dims
    )
    repair_time_dim_for_viewer(viewer)


def _on_active_layer_sync_dims(viewer: Any, _event: Any) -> None:
    """Re-apply 4D dims when selecting a 4D layer (3D oblique affines can pollute viewer.dims)."""
    from nvitk.gui.core.orientation import (
        _axes_string_from_layer,
        _synchronize_4d_dims,
        ensure_4d_scale_only_layer,
    )

    if not viewer.layers:
        return
    layer = viewer.layers.selection.active
    if layer is None or getattr(layer.data, "ndim", 0) <= 3:
        return
    layer_type = type(layer).__name__
    if layer_type in ("Vectors", "Points", "Shapes", "Surface"):
        return
    ensure_4d_scale_only_layer(layer)
    axes_str = _axes_string_from_layer(layer)
    _synchronize_4d_dims(viewer, layer, axes_str=axes_str, shape=tuple(layer.data.shape))


def install_nvitk_layer_hooks(viewer: Any) -> None:
    """Configure dims when layers are added by the nvitk-io reader (not only Qt open)."""
    if getattr(viewer, "_nvitk_layer_hooks", False):
        return

    events = viewer.layers.events

    def _callback(event: Any) -> None:
        """Forward a layer-list event to :func:`_on_nvitk_layer_inserted`."""
        _on_nvitk_layer_inserted(viewer, event)

    if hasattr(events, "inserted"):
        events.inserted.connect(_callback)
    elif hasattr(events, "added"):
        events.added.connect(_callback)
    else:
        return

    @viewer.layers.selection.events.active.connect
    def _active_layer_callback(event: Any) -> None:
        """Forward an active-layer-selection event to :func:`_on_active_layer_sync_dims`."""
        _on_active_layer_sync_dims(viewer, event)

    viewer._nvitk_layer_hooks = True


def install_nvitk_io(viewer: Any) -> None:
    """
    Hook Qt open/drop paths to nvitk.io.

    Napari's Viewer model is Pydantic and does not allow patching ``open``;
    we wrap ``QtViewer._qt_open`` instead.
    """
    try:
        _ = viewer.window
    except Exception:
        pass

    try:
        qt = viewer.window._qt_viewer
    except AttributeError:
        return

    if getattr(qt, "_nvitk_io_patched", False):
        return

    original_qt_open = qt._qt_open

    def _qt_open(
        filenames,
        stack = False,
        choose_plugin = False,
        plugin=None,
        layer_type=None,
        **kwargs,
    ):
        """Route file-open requests through nvitk's reader (falling back to the original Qt open
        when the user explicitly picked a plugin or nvitk can't handle any of the files)."""
        if choose_plugin:
            return original_qt_open(
                filenames,
                stack=stack,
                choose_plugin=choose_plugin,
                plugin=plugin,
                layer_type=layer_type,
                **kwargs,
            )

        if isinstance(filenames, str):
            paths = [Path(filenames)]
        else:
            paths = [Path(f) for f in filenames]

        nvitk_paths = [p for p in paths if _nvitk_can_open(p)]
        if nvitk_paths:
            try:
                layers = open_paths_with_nvitk(viewer, nvitk_paths, stack=stack)
                if layers:
                    return
            except Exception as exc:
                _notify_error(f"nvitk I/O failed:\n{exc}")

        return original_qt_open(
            filenames,
            stack=stack,
            choose_plugin=choose_plugin,
            plugin=plugin,
            layer_type=layer_type,
            **kwargs,
        )

    qt._qt_open = _qt_open
    qt._nvitk_io_patched = True
    install_nvitk_layer_hooks(viewer)
