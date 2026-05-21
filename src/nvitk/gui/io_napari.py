"""Load images into Napari via :mod:`nvitk.io` (NIfTI, DICOM, TIFF, MHA, ND2, …)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.io import imread
from nvitk.io._common import guess_read_type
from nvitk.types import Image

_NVITK_OPEN_SUFFIXES = frozenset({
    ".nii", ".nii.gz", ".mha", ".mhd", ".tif", ".tiff", ".nd2",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".dcm",
})


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
    """4x4 voxel-to-world matrix for Napari (fixes L/R and oblique orientation)."""
    aff = img.affine
    if aff is None:
        return None
    aff = np.asarray(aff, dtype=float)
    if aff.shape != (4, 4):
        return None
    return aff


def _layer_meta(img: Image, path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "name": img.name or path.stem,
        "source": str(path),
    }
    if img.metadata:
        meta["nvitk_metadata"] = dict(img.metadata)
    if img.axes:
        meta["axes"] = img.axes
    return meta


def _add_image_to_viewer(viewer: Any, img: Image, path: Path) -> Any:
    data = to_numpy(img.data)
    meta = _layer_meta(img, path)
    affine = _napari_affine(img)
    kwargs: dict[str, Any] = {"name": meta["name"], "metadata": meta}
    if affine is not None:
        kwargs["affine"] = affine
    else:
        scale = _napari_scale(img, data.ndim)
        if scale is not None:
            kwargs["scale"] = scale
    return viewer.add_image(data, **kwargs)


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


def read_paths(path: str | list[str]) -> list[tuple] | None:
    """Napari npe2 reader — LayerData tuples for drag-and-drop / File › Open."""
    paths = [path] if isinstance(path, str) else list(path)
    layer_data: list[tuple] = []

    for p in paths:
        pth = Path(p)
        if not _nvitk_can_open(pth):
            continue
        try:
            result = imread(pth, backend="numpy")
        except Exception:
            continue
        images = result if isinstance(result, list) else [result]
        for img in images:
            data = to_numpy(img.data)
            meta = _layer_meta(img, pth)
            affine = _napari_affine(img)
            if affine is not None:
                meta["affine"] = affine
            else:
                scale = _napari_scale(img, data.ndim)
                if scale is not None:
                    meta["scale"] = scale
            layer_data.append((data, meta, "image"))

    return layer_data or None


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


def napari_manifest() -> dict[str, Any]:
    """Entry point for ``[project.entry-points.\"napari.manifest\"]``."""
    return {
        "name": "nvitk-io",
        "display_name": "nvitk I/O",
        "version": "0.1.0",
        "contributions": {
            "commands": [
                {
                    "id": "nvitk-read-paths",
                    "title": "Read with nvitk.io",
                    "python_name": "nvitk.gui.io_napari:read_paths",
                }
            ],
            "readers": [
                {
                    "command": "nvitk-read-paths",
                    "accepts": [
                        "*.nii",
                        "*.nii.gz",
                        "*.mha",
                        "*.mhd",
                        "*.tif",
                        "*.tiff",
                        "*.nd2",
                        "*.png",
                        "*.jpg",
                        "*.jpeg",
                        "*.bmp",
                        "*.gif",
                        "*.dcm",
                    ],
                    "priority": 100,
                }
            ],
        },
    }
