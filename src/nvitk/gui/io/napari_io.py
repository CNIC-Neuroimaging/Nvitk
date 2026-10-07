"""Load images into Napari via :mod:`nvitk.io` (NIfTI, DICOM, TIFF, MHA, ND2, Blosc2, …)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.gui.core.orientation import (
    configure_viewer_for_layer,
    install_affine_ndim_guard,
    DISPLAY_REORDERED_KEY,
    SOURCE_AXES_KEY,
    TIME_LEADING_KEY,
    prepare_for_napari,
    prepare_time_leading_for_napari,
    prepare_world_ordered_for_napari,
    suppress_nonorthogonal_slice_warning,
)
from nvitk.gui.core.warnings import install_napari_display_warnings

install_napari_display_warnings()
install_affine_ndim_guard()
from nvitk.io import imread
from nvitk.io._common import default_nifti_axes, guess_read_type
from nvitk.types import Image

_NVITK_OPEN_SUFFIXES = frozenset({
    ".nii", ".nii.gz", ".mha", ".mhd", ".tif", ".tiff", ".nd2",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".dcm",
    # nnU-Net / nnssl preprocessed cases: the Blosc2 array and its geometry sidecar.
    ".b2nd", ".pkl",
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
        return guess_read_type(path) in ("nifti", "dicom", "tiff", "mha", "pil", "nd2", "b2nd", "pkl")
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


def _rgb_channel_axis(img: Image, ndim: int) -> int | None:
    """Array axis holding colour samples, from the reader's ``rgb`` verdict and axis labels."""
    md = img.metadata or {}
    if md.get("rgb") is False:
        return None
    axes = (img.axes or "").upper()
    if len(axes) == ndim and axes.count("C") == 1 and ndim >= 3:
        return axes.index("C")
    if md.get("rgb") is True and ndim >= 3:
        return ndim - 1
    return None


def _prepare_rgb_layer_tuple(
    img: Image,
    path: Path,
    data: np.ndarray,
    raw_affine: np.ndarray | None,
    channel_axis: int,
) -> LayerData:
    """Layer tuple for a colour image.

    Napari folds the channel axis into ``rgb``, making the layer one dimension smaller than
    the array: only the remaining axes take labels and scale, and a voxel-to-world affine
    sized for the full array no longer fits.
    """
    if channel_axis != data.ndim - 1:
        data = np.moveaxis(data, channel_axis, -1)
    axes = img.axes or default_nifti_axes(data.ndim)
    spatial = "".join(ch for i, ch in enumerate(axes) if i != channel_axis)
    if len(spatial) != data.ndim - 1:
        spatial = default_nifti_axes(data.ndim - 1)
    layer_meta: dict[str, Any] = {
        "name": img.name or path.stem,
        "metadata": _nvitk_layer_metadata(img, path, affine_source=raw_affine),
        "axis_labels": tuple(spatial),
        "rgb": True,
    }
    md = img.metadata or {}
    scale = tuple(_resolution_for_axis(md, ch) or 1.0 for ch in spatial)
    if any(value != 1.0 for value in scale):
        layer_meta["scale"] = scale
    return (np.ascontiguousarray(data), layer_meta, "image")


def _spatial(axes: str) -> str:
    """The spatial letters of *axes*, in order."""
    return "".join(ch for ch in axes.upper() if ch in "XYZ")


def _prepare_layer_tuple(img: Image, path: Path) -> LayerData:
    """Build a napari-plugin-style ``(data, layer_kwargs, layer_type)`` tuple for *img*, reorienting
    the array for display and computing scale/affine, axis labels, and nvitk metadata from *path*."""
    data = to_numpy(img.data)
    raw_affine = _napari_affine(img)
    channel_axis = _rgb_channel_axis(img, data.ndim)
    if channel_axis is not None:
        return _prepare_rgb_layer_tuple(img, path, data, raw_affine, channel_axis)
    axis_labels = _axis_labels_for_image(img, data.ndim)
    axes_str = "".join(axis_labels)

    # 3D+t: shown time-first with its full affine, so 3D layers of the same
    # patient overlay it (see orientation.prepare_time_leading_for_napari).
    timed = prepare_time_leading_for_napari(
        data, raw_affine, axes=axes_str, metadata=img.metadata
    )
    if timed is not None:
        view, affine5, display_axes = timed
        meta = _nvitk_layer_metadata(img, path, affine_source=raw_affine)
        meta["nvitk_metadata"][TIME_LEADING_KEY] = True
        meta["nvitk_metadata"][SOURCE_AXES_KEY] = axes_str
        meta["nvitk_metadata"]["axes"] = display_axes
        meta["axes"] = display_axes
        if _spatial(display_axes) != _spatial(axes_str):
            meta["nvitk_metadata"][DISPLAY_REORDERED_KEY] = True
        return (
            view,
            {
                "name": img.name or path.stem,
                "metadata": meta,
                "axis_labels": tuple(display_axes),
                "affine": affine5,
                "rgb": False,
            },
            "image",
        )

    # A slice or volume whose axes are not in world order (sagittal or coronal
    # storage, any positioned 2D image): shown world-ordered, a view of the file.
    ordered = prepare_world_ordered_for_napari(data, raw_affine, axes=axes_str)
    if ordered is not None:
        view, affine4, display_axes = ordered
        meta = _nvitk_layer_metadata(img, path, affine_source=raw_affine)
        meta["nvitk_metadata"][DISPLAY_REORDERED_KEY] = True
        meta["nvitk_metadata"][SOURCE_AXES_KEY] = axes_str
        meta["nvitk_metadata"]["axes"] = display_axes
        meta["axes"] = display_axes
        return (
            view,
            {
                "name": img.name or path.stem,
                "metadata": meta,
                "axis_labels": tuple(display_axes),
                "affine": affine4,
                "rgb": False,
            },
            "image",
        )

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
        # Napari otherwise reads a trailing axis of 3 or 4 as colour samples.
        "rgb": False,
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


def _dicom_series_name(img: Image) -> str | None:
    """A layer name for a DICOM series: its description, else its protocol, else
    its number — with the part a mixed series was split into (``COR``, ``SAG``,
    ``512x512``) in parentheses. ``None`` when *img* carries no series tags."""
    md = img.metadata or {}

    def _text(*keys: str) -> str:
        for key in keys:
            value = str(md.get(key) or "").strip()
            if value:
                return value
        return ""

    number = _text("series_number", "SeriesNumber")
    name = _text("series_description", "SeriesDescription", "ProtocolName")
    if not name and number:
        name = f"Series {number}"
    if not name:
        return None
    part = _text("geometry_subseries")
    return f"{name} ({part})" if part else name


def _name_image(img: Image, path: Path, suffix: str = "") -> None:
    """Name *img* for its layer.

    A DICOM series (folder or file) is named by its description rather than by the
    folder it was read from: a study folder holds a dozen series, and the folder name made them
    all ``studyid_<uid>``, ``[1]``, ``[2]``… An explicit name the reader chose is
    kept; the folder/file name is only the fallback.
    """
    default = {path.name, path.stem, f"{path.stem}{suffix}"}
    if img.name and img.name not in default:
        return
    series = _dicom_series_name(img) if _is_dicom_source(path) else None
    img.name = series or img.name or f"{path.stem}{suffix}"


def _is_dicom_source(path: Path) -> bool:
    """A DICOM folder or file. A NIfTI converted from DICOM carries the same series
    tags in its sidecar, but its file name was chosen and is kept."""
    if path.is_dir():
        return True
    try:
        return guess_read_type(path) == "dicom"
    except Exception:  # noqa: BLE001
        return False


#: Surface / point files the reader hands to :mod:`nvitk.meshlab.io` (``.csv`` and
#: ``.txt`` are left out: too often tables rather than point lists).
_SURFACE_SUFFIXES = (".stl", ".obj", ".off", ".ply", ".vtk", ".vtp", ".vtu", ".gii", ".xyz", ".pcd", ".pvd")


def _is_surface_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _SURFACE_SUFFIXES


def _read_surface_layer_data(path: Path) -> list[LayerData] | None:
    """A mesh (Surface) or point cloud (Points) file as a layer tuple, in world mm."""
    from nvitk.meshlab.io import read_pvd, read_surface
    from nvitk.types import Mesh

    try:
        if path.suffix.lower() == ".pvd":
            # A time series needs its frame controller: add it through the Mesh panel.
            series = read_pvd(path)
            mesh = series[0]
            return [((mesh.vertices, mesh.faces), {"name": path.stem, "metadata": {"nvitk_mesh": {"space": "world"}}},
                     "surface")]
        obj = read_surface(path)
    except Exception as exc:  # noqa: BLE001
        _notify_error(f"Could not read {path} as a mesh / point cloud:\n{exc}")
        return None
    meta = {"nvitk_mesh": {"space": "world", "source": str(path)}}
    if isinstance(obj, Mesh):
        return [((obj.vertices.astype("float32"), obj.faces.astype("int64")),
                 {"name": obj.name, "metadata": meta, "shading": "smooth"}, "surface")]
    return [(obj.points.astype("float32"), {"name": obj.name, "metadata": meta, "size": 1.0,
                                            "border_width": 0, "face_color": "#5fb8ff"}, "points")]


def _read_layer_data(path: str) -> list[LayerData] | None:
    """Read one path into Napari layer tuples."""
    pth = Path(path)
    if _is_surface_file(pth):
        return _read_surface_layer_data(pth)
    if not _nvitk_can_open(pth):
        return None
    try:
        result = imread(pth, backend="numpy")
    except Exception:
        return None
    images = result if isinstance(result, list) else [result]
    out = []
    for i, img in enumerate(images):
        _name_image(img, pth, f"_{i}" if len(images) > 1 else "")
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

    if not (_nvitk_can_open(Path(path)) or _is_surface_file(Path(path))):
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
    if "rgb" in layer_meta:
        kwargs["rgb"] = layer_meta["rgb"]
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
        if _is_surface_file(path):
            panel = getattr(viewer, "_nvitk_mesh_panel", None)
            try:
                if panel is not None:
                    layers.extend(panel.open_paths([str(path)]))
                else:
                    for data, kwargs, kind in _read_surface_layer_data(path) or []:
                        layers.append(getattr(viewer, f"add_{kind}")(data, **kwargs))
            except Exception as exc:  # noqa: BLE001
                _notify_error(f"Could not read {path}:\n{exc}")
            continue
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
            _name_image(img, path, suffix)
            layers.append(_add_image_to_viewer(viewer, img, path))

    return layers


def open_dicom_files_with_nvitk(viewer: Any, files: Sequence[str | Path], *, source: str | Path | None = None) -> list[Any]:
    """Load exactly *files* (DICOM, e.g. one series picked in the DICOM browser) through
    nvitk's DICOM stack and add their volumes as layers; *source* names the folder they
    came from in the layers' metadata."""
    from nvitk.io.conversors._dicom_conversion import load_dicom_series
    from nvitk.types import Image

    paths = [str(f) for f in files]
    if not paths:
        return []
    origin = Path(source) if source is not None else Path(os.path.commonpath(paths) if len(paths) > 1 else paths[0])
    try:
        result = load_dicom_series(paths, return_all_series=True)
    except Exception as exc:
        _notify_error(f"Could not load the selected DICOM files:\n{exc}")
        return []
    layers = []
    for i, (data, metadata) in enumerate(result):
        md = dict(metadata)
        img = Image(data=np.asarray(data), metadata=md, axes=md.get("axes"), name=md.get("name") or origin.stem,
                    orientation=md.get("orientation"))
        _name_image(img, origin if origin.is_dir() else origin.parent, f"_{i}" if len(result) > 1 else "")
        layers.append(_add_image_to_viewer(viewer, img, origin))
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
    from nvitk.gui.core.orientation import layer_is_time_leading

    if layer_is_time_leading(layer):
        # Its dims come from its affine like any 3D layer's; re-forcing them on
        # every selection would yank the slider back to frame 0.
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

    def _open_folder_dialog(choose_plugin: bool = False) -> None:
        """File ▸ Open Folder… takes several folders at once (Ctrl / Shift-click):
        several DICOM series folders, say, loaded in one go."""
        from nvitk.gui.core.dialogs import choose_directories

        try:
            from napari.utils.history import get_open_history, update_open_history

            start = (get_open_history() or [""])[0]
        except Exception:  # noqa: BLE001
            update_open_history = None
            start = ""
        folders = choose_directories(qt, "Select folder(s)… (Ctrl / Shift-click for several)", start)
        if not folders:
            return
        qt._qt_open(folders, stack=False, choose_plugin=choose_plugin)
        if update_open_history is not None:
            try:
                update_open_history(folders[0])
            except Exception:  # noqa: BLE001
                pass

    qt._qt_open = _qt_open
    qt._open_folder_dialog = _open_folder_dialog
    qt._nvitk_io_patched = True
    install_nvitk_layer_hooks(viewer)
    _add_open_folders_action(viewer, _open_folder_dialog)


def _add_open_folders_action(viewer: Any, opener: Callable[[], None]) -> None:
    """*File ▸ Open Folders… (several)*, right after Napari's single-folder entry.

    Napari's own *Open Folder…* is an app-model command bound to the class method, so
    it keeps its one-folder dialog; this sits beside it.
    """
    try:
        from qtpy.QtWidgets import QAction
    except ImportError:  # Qt 6 moved it
        from qtpy.QtGui import QAction
    try:
        menu = viewer.window.file_menu
    except Exception:  # noqa: BLE001
        return
    action = QAction("Open Folders… (several)", menu)
    action.setToolTip("Choose several folders at once (Ctrl / Shift-click) — e.g. DICOM series folders.")
    action.triggered.connect(lambda _checked=False: opener())
    actions = menu.actions()
    anchor = next((i for i, a in enumerate(actions) if a.text().replace("&", "").startswith("Open Folder")), None)
    if anchor is not None and anchor + 1 < len(actions):
        menu.insertAction(actions[anchor + 1], action)
    else:
        menu.addAction(action)
    viewer.window._nvitk_open_folders_action = action
