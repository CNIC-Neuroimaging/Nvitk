"""Load images into Napari via :mod:`nvitk.io` (NIfTI, DICOM, TIFF, MHA, ND2, …)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.gui.orientation import (
    configure_viewer_for_layer,
    prepare_for_napari,
    suppress_nonorthogonal_slice_warning,
)
from nvitk.gui.warnings import install_napari_display_warnings

install_napari_display_warnings()
from nvitk.io import imread
from nvitk.io._common import guess_read_type
from nvitk.types import Image

_NVITK_OPEN_SUFFIXES = frozenset({
    ".nii", ".nii.gz", ".mha", ".mhd", ".tif", ".tiff", ".nd2",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".dcm",
})

LayerData = tuple[Any, dict[str, Any], str]
ReaderFunc = Callable[[str], list[LayerData] | None]


def _normalize_paths(path: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(path, (str, Path)):
        return [Path(path)]
    return [Path(p) for p in path]


def _nvitk_can_open(path: Path) -> bool:
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


def _napari_scale(img: Image, ndim: int) -> tuple[float, ...] | None:
    sp = img.spacing
    if sp is None:
        sp = img.metadata.get("spacing") if img.metadata else None
    if sp is None:
        return None
    vals = [float(x) for x in sp[:ndim]]
    if len(vals) < ndim:
        vals.extend([1.0] * (ndim - len(vals)))
    return tuple(vals)


def _napari_affine(img: Image) -> np.ndarray | None:
    """Raw 4x4 voxel-to-world matrix from the file (before display reorientation)."""
    aff = img.affine
    if aff is None:
        return None
    aff = np.asarray(aff, dtype=float)
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
    if affine_source is not None:
        nvitk_md["affine_source"] = np.asarray(affine_source, dtype=float)
    out: dict[str, Any] = {"nvitk_metadata": nvitk_md}
    if img.axes:
        out["axes"] = img.axes
    return out


def _prepare_layer_tuple(img: Image, path: Path) -> LayerData:
    data = to_numpy(img.data)
    raw_affine = _napari_affine(img)
    data, affine, scale = prepare_for_napari(data, raw_affine)
    layer_meta: dict[str, Any] = {
        "name": img.name or path.stem,
        "metadata": _nvitk_layer_metadata(img, path, affine_source=raw_affine),
    }
    if affine is not None:
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
    out: list[LayerData] = []
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
        layer_data: list[LayerData] = []
        for p in path:
            chunk = _read_layer_data(p)
            if chunk:
                layer_data.extend(chunk)
        return layer_data or None

    if not _nvitk_can_open(Path(path)):
        return None
    return _read_layer_data


def _add_image_to_viewer(viewer: Any, img: Image, path: Path) -> Any:
    data, layer_meta, _ = _prepare_layer_tuple(img, path)
    kwargs: dict[str, Any] = {
        "name": layer_meta["name"],
        "metadata": layer_meta["metadata"],
    }
    if "affine" in layer_meta:
        kwargs["affine"] = layer_meta["affine"]
    elif "scale" in layer_meta:
        kwargs["scale"] = layer_meta["scale"]
    with suppress_nonorthogonal_slice_warning():
        layer = viewer.add_image(data, **kwargs)
    configure_viewer_for_layer(viewer, layer)
    return layer


def open_paths_with_nvitk(
    viewer: Any,
    paths: str | Path | Sequence[str | Path],
    *,
    stack: bool = False,
) -> list[Any]:
    """Open one or more paths with nvitk.io and add Napari layers."""
    _ = stack
    path_list = _normalize_paths(paths)
    layers: list[Any] = []

    for path in path_list:
        if not _nvitk_can_open(path):
            continue
        try:
            result = imread(path, backend="numpy")
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
    try:
        from napari.utils.notifications import show_error
        show_error(message)
    except Exception:
        print(message, flush=True)


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
        stack: bool = False,
        choose_plugin: bool = False,
        plugin=None,
        layer_type=None,
        **kwargs,
    ):
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
