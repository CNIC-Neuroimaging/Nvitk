"""Files in and out of the external-suite tools (FastSurfer, FSL).

Those tools are command-line programs: the active layer is written to a run
directory, the program writes its outputs there, and the outputs are opened again
as layers. The helpers here keep that round trip in one place:

* :func:`run_directory` — the tool's *output folder* parameter, or a fresh temporary one;
* :func:`export_layer_file` — a layer as NIfTI in the file's own axis order (a
  time-first 3D+t layer such as a DWI series is written ``XYZT``, as FSL expects);
* :func:`sidecar_file` — a file stored next to the layer's source (``.bval`` / ``.bvec``);
* :func:`add_volume` — an output volume as an Image or Labels layer, placed by its
  own affine, with names and colours from a lookup table for label maps;
* :func:`add_surface` — a mesh already in scanner RAS (world) millimetres.

What a run produced is described by an **import manifest** — the same for a local run
and a cluster job (the SGE worker writes it into its ``.done`` marker), so results open
identically either way::

    {"layers":   [{"file", "title", "labels", "visible", "lut", "schema", "colormap"}],
     "surfaces": [{"file", "title", "values", "visible"}],
     "tables":   [{"title", "payload", "subtitle"}],
     "notes":    ["..."]}

``file`` / ``values`` are relative to the folder the manifest describes (absolute paths
also work); a label layer's ``lut`` holds ``{id: [name, [r, g, b, a]]}`` for the ids it
contains, so no lookup table has to exist where it is opened. :func:`manifest_layer`
and :func:`manifest_files` build and list one (no Qt); :func:`load_manifest` opens it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def run_directory(params: Mapping[str, Any], prefix: str, *, key: str = "output_dir") -> Path:
    """The folder a tool writes into: *params[key]* when set, else a new temporary one."""
    raw = str(params.get(key) or "").strip()
    path = Path(raw).expanduser() if raw else Path(tempfile.mkdtemp(prefix=f"nvitk_{prefix}_"))
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def export_layer_file(layer: Any, path: str | Path) -> Path:
    """Write *layer* to *path* (NIfTI) the way it is stored on disk, and return the path."""
    from nvitk.gui.io.export import export_layer

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_layer(layer, out)
    return out


def layer_source_path(layer: Any) -> Path | None:
    """The file *layer* was read from, when nvitk opened it."""
    meta = getattr(layer, "metadata", None) or {}
    nested = meta.get("nvitk_metadata") if isinstance(meta, dict) else None
    src = (nested or {}).get("source") if isinstance(nested, dict) else None
    src = src or (meta.get("source") if isinstance(meta, dict) else None)
    return Path(str(src)) if src else None


def _stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".mgz", ".mgh"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def sidecar_file(layer: Any, suffixes: tuple[str, ...], fallbacks: tuple[str, ...] = ()) -> Path | None:
    """A file stored beside *layer*'s source: ``<stem><suffix>`` first, then *fallbacks* by name.

    ``sidecar_file(dwi, (".bval", ".bvals"), ("bvals",))`` finds ``dwi.bval`` next to
    ``dwi.nii.gz``, or an FSL-style ``bvals`` in the same folder.
    """
    src = layer_source_path(layer)
    if src is None:
        return None
    folder = src if src.is_dir() else src.parent
    stem = _stem(src)
    for suffix in suffixes:
        candidate = folder / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    for name in fallbacks:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def file_stem(name: str, default: str = "layer") -> str:
    """A layer name as a file stem FSL / FastSurfer accept: no image suffix (napari names a
    ``x.nii.gz`` layer ``x.nii``, which FSL would read as an image name), no dots or spaces."""
    stem = str(name or "").strip()
    for suffix in (".nii.gz", ".nii", ".mgz", ".mgh"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    import re

    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or default


def path_param(params: Mapping[str, Any], key: str) -> Path | None:
    """A path parameter, expanded, or ``None`` when blank."""
    raw = str(params.get(key) or "").strip()
    return Path(raw).expanduser() if raw else None


def _read_lut(lut: Mapping[int, Any] | None) -> tuple[dict[int, str], dict[int, np.ndarray]]:
    """``{id: (name, rgba)}`` (or ``{id: name}``) split into names and RGBA colours."""
    names: dict[int, str] = {}
    colors: dict[int, np.ndarray] = {}
    for key, value in (lut or {}).items():
        lid = int(key)
        if isinstance(value, str):
            names[lid] = value
            continue
        name, rgba = value
        names[lid] = str(name)
        if rgba is not None:
            arr = np.asarray(rgba, dtype=float)
            if arr.max(initial=0.0) > 1.0:
                arr = arr / 255.0
            if arr.size == 3:
                arr = np.append(arr, 1.0)
            colors[lid] = arr
    return names, colors


def add_volume(
    viewer: Any,
    path: str | Path,
    *,
    name: str | None = None,
    labels: bool = False,
    lut: Mapping[int, Any] | None = None,
    schema: str | None = None,
    colormap: str | None = None,
    opacity: float | None = None,
    visible: bool = True,
) -> Any:
    """Open *path* as a layer placed by its own affine (FastSurfer / FSL outputs).

    *labels* makes it a Labels layer; *lut* (``{id: name}`` or ``{id: (name, rgba)}``)
    names its ids and, with colours, paints them; *schema* records the label
    vocabulary the label pickers use.
    """
    from nvitk.gui.io.napari_io import _add_image_to_viewer, _name_image, _prepare_layer_tuple
    from nvitk.io import imread

    p = Path(path)
    img = imread(p, backend="numpy")
    if isinstance(img, list):
        img = img[0]
    _name_image(img, p)
    if name:
        img.name = name
    if not labels:
        layer = _add_image_to_viewer(viewer, img, p)
        if colormap:
            try:
                layer.colormap = colormap
            except Exception:  # noqa: BLE001
                pass
        if opacity is not None:
            layer.opacity = float(opacity)
        layer.visible = bool(visible)
        return layer

    data = np.asarray(img.data)
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data)
    img.data = data.astype(np.int32, copy=False)
    view, kwargs, _kind = _prepare_layer_tuple(img, p)
    kwargs.pop("rgb", None)
    from nvitk.gui.core.orientation import suppress_nonorthogonal_slice_warning

    with suppress_nonorthogonal_slice_warning():
        layer = viewer.add_labels(
            np.asarray(view).astype(np.int32, copy=False),
            opacity=0.7 if opacity is None else float(opacity),
            visible=bool(visible),
            **kwargs,
        )
    layer._nvitk_label_like = True
    names, colors = _read_lut(lut)
    if names:
        from nvitk.gui.labels.catalog import LABEL_NAMES_KEY

        present = set(int(v) for v in np.unique(layer.data)) - {0}
        meta = dict(layer.metadata or {})
        meta[LABEL_NAMES_KEY] = {int(k): v for k, v in names.items() if int(k) in present}
        layer.metadata = meta
    if colors:
        from nvitk.gui.labels.visibility import _apply_label_color_dict

        try:
            _apply_label_color_dict(layer, colors)
        except Exception:  # noqa: BLE001 — colours are cosmetic
            pass
    if schema:
        from nvitk.gui.labels.catalog import remember_layer_schema

        remember_layer_schema(layer, schema)
    return layer


def add_surface(
    viewer: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    name: str,
    source: str | Path | None = None,
    values: np.ndarray | None = None,
    colormap: str = "gray",
    visible: bool = True,
) -> Any:
    """A mesh with *vertices* in scanner-RAS millimetres (world space) as a Surface layer."""
    data: tuple = (np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64))
    if values is not None:
        data = data + (np.asarray(values, dtype=np.float32),)
    meta = {"nvitk_mesh": {"space": "world", "source": str(source) if source else ""}}
    return viewer.add_surface(
        data, name=name, metadata=meta, shading="smooth", colormap=colormap, visible=bool(visible)
    )


# ── where a tool's inputs come from ──────────────────────────────────────────


class ToolInputs:
    """The files and settings an external-suite tool works from.

    Two sources share this interface, so one tool definition serves both runs:
    :class:`ViewerInputs` (local: layers written out on demand from the viewer) and
    :class:`StagedInputs` (a cluster job: the files staged with it). *work* is the folder
    the tool writes into.
    """

    def __init__(self, tool_id: str, params: Mapping[str, Any], work: Path, name: str) -> None:
        self.tool_id = tool_id
        self.params = dict(params)
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        #: The input layer's name as a file stem (output names start with it).
        self.name = file_stem(name, "input")
        #: The input layer's display name (prefix of the layers the results open as).
        self.display_name = str(name or tool_id)

    # sources
    def input(self) -> Path:
        raise NotImplementedError

    def layer_file(self, key: str, *, required: bool = True, what: str = "") -> Path | None:
        raise NotImplementedError

    def named_layer_file(self, name: str) -> Path:
        raise NotImplementedError

    def input_is_labels(self) -> bool:
        raise NotImplementedError

    def sidecar_default(self, suffixes: tuple[str, ...], fallbacks: tuple[str, ...]) -> Path | None:
        return None

    # parameters
    def _missing(self, key: str, what: str) -> ValueError:
        return ValueError(f"Set the {what or key.replace('fsl_', '').replace('fs_', '').replace('_', ' ')}.")

    def path(self, key: str, *, required: bool = False, what: str = "") -> Path | None:
        p = path_param(self.params, key)
        if p is None and required:
            raise self._missing(key, what)
        if p is not None and not p.exists():
            raise ValueError(f"{p} does not exist.")
        return p

    def sidecar(self, key: str, suffixes: tuple[str, ...], fallbacks: tuple[str, ...], what: str) -> Path:
        p = self.path(key) or self.sidecar_default(suffixes, fallbacks)
        if p is None:
            raise ValueError(f"Set the {what} file (none found next to the layer's source).")
        return p

    def bvecs(self) -> Path:
        return self.sidecar("fsl_bvecs", (".bvec", ".bvecs"), ("bvecs",), "bvecs")

    def bvals(self) -> Path:
        return self.sidecar("fsl_bvals", (".bval", ".bvals"), ("bvals",), "bvals")

    def out(self, suffix: str) -> Path:
        return self.work / f"{self.name}_{suffix}"

    def f(self, key: str, default: float) -> float:
        value = self.params.get(key)
        return float(default if value in (None, "") else value)

    def i(self, key: str, default: int) -> int:
        value = self.params.get(key)
        return int(default if value in (None, "") else value)

    def b(self, key: str, default: bool = False) -> bool:
        return bool(self.params.get(key, default))

    def s(self, key: str, default: str = "") -> str:
        return str(self.params.get(key) or default).strip()


class ViewerInputs(ToolInputs):
    """Inputs taken from the viewer: the active *layer* and the layers named in the params."""

    def __init__(self, tool_id: str, viewer: Any, layer: Any, params: Mapping[str, Any], work: Path | None = None) -> None:
        super().__init__(tool_id, params, work or run_directory(params, tool_id),
                         str(getattr(layer, "name", "") or tool_id.split("_", 1)[-1]))
        self.viewer = viewer
        self.layer = layer
        self._exported: dict[int, Path] = {}

    def export(self, layer: Any) -> Path:
        key = id(layer)
        if key not in self._exported:
            self._exported[key] = export_layer_file(layer, self.work / "inputs" / f"{file_stem(layer.name)}.nii.gz")
        return self._exported[key]

    def input(self) -> Path:
        if self.layer is None:
            raise ValueError("Select the input layer first.")
        return self.export(self.layer)

    def _named(self, name: str) -> Any:
        from nvitk.gui.tools.runner import _resolve_layer

        return _resolve_layer(self.viewer, name)

    def layer_file(self, key: str, *, required: bool = True, what: str = "") -> Path | None:
        from nvitk.gui.tools.runner import _layer_param

        name = _layer_param(self.params, key)
        if not name:
            if required:
                raise ValueError(f"Choose the {what or key.replace('fsl_', '').replace('fs_', '').replace('_', ' ')}.")
            return None
        return self.export(self._named(name))

    def named_layer_file(self, name: str) -> Path:
        return self.export(self._named(name.strip()))

    def input_is_labels(self) -> bool:
        from nvitk.gui.labels.visibility import is_label_like_layer

        return bool(self.layer is not None and is_label_like_layer(self.layer))

    def sidecar_default(self, suffixes: tuple[str, ...], fallbacks: tuple[str, ...]) -> Path | None:
        return sidecar_file(self.layer, suffixes, fallbacks) if self.layer is not None else None


class StagedInputs(ToolInputs):
    """Inputs staged with a cluster job: files already on disk, layers by parameter or name."""

    def __init__(
        self,
        tool_id: str,
        params: Mapping[str, Any],
        work: Path,
        *,
        input_file: Path | None,
        input_name: str,
        layer_files: Mapping[str, Path] | None = None,
        named_files: Mapping[str, Path] | None = None,
        input_is_labels: bool = False,
    ) -> None:
        super().__init__(tool_id, params, work, input_name)
        self._input = Path(input_file) if input_file else None
        self._layers = {str(k): Path(v) for k, v in (layer_files or {}).items()}
        self._named = {str(k): Path(v) for k, v in (named_files or {}).items()}
        self._labels = bool(input_is_labels)

    def input(self) -> Path:
        if self._input is None or not self._input.is_file():
            raise ValueError("The job has no input image.")
        return self._input

    def layer_file(self, key: str, *, required: bool = True, what: str = "") -> Path | None:
        path = self._layers.get(key)
        if path is None and required:
            raise ValueError(f"The job has no {what or key} layer.")
        return path

    def named_layer_file(self, name: str) -> Path:
        try:
            return self._named[name.strip()]
        except KeyError:
            raise ValueError(f"The job has no layer named {name!r}.") from None

    def input_is_labels(self) -> bool:
        return self._labels


# ── import manifests ─────────────────────────────────────────────────────────


def _relative(path: Path, base: Path | None) -> str:
    if base is not None:
        try:
            return str(Path(path).resolve().relative_to(Path(base).resolve()))
        except ValueError:
            pass
    return str(path)


def manifest_layer(
    path: str | Path,
    title: str,
    *,
    base: Path | None = None,
    labels: bool = False,
    visible: bool = True,
    lut: Mapping[int, Any] | None = None,
    schema: str | None = None,
    colormap: str | None = None,
) -> dict[str, Any]:
    """One volume entry. For a label map, *lut* (``{id: name}`` or ``{id: (name, rgba)}``) is cut
    down to the ids the file contains, so the entry carries everything needed to name it."""
    entry: dict[str, Any] = {"file": _relative(Path(path), base), "title": title, "labels": bool(labels),
                             "visible": bool(visible)}
    if labels and lut:
        try:
            import nibabel as nib

            present = {int(v) for v in np.unique(np.asarray(nib.load(str(path)).dataobj))} - {0}
        except Exception:  # noqa: BLE001 — unreadable here: keep the whole table
            present = {int(k) for k in lut}
        table: dict[str, Any] = {}
        for key, value in lut.items():
            if int(key) not in present:
                continue
            if isinstance(value, str):
                table[str(int(key))] = [value, None]
            else:
                name, rgba = value
                table[str(int(key))] = [str(name), [float(c) for c in rgba] if rgba is not None else None]
        entry["lut"] = table
    if schema:
        entry["schema"] = schema
    if colormap:
        entry["colormap"] = colormap
    return entry


def manifest_files(manifest: Mapping[str, Any]) -> list[str]:
    """Every file a manifest refers to (what a cluster job has to send back)."""
    out: list[str] = []
    for entry in manifest.get("layers", []) or []:
        out.append(str(entry["file"]))
    for entry in manifest.get("surfaces", []) or []:
        out.append(str(entry["file"]))
        if entry.get("values"):
            out.append(str(entry["values"]))
    out += [str(f) for f in manifest.get("files", []) or []]
    seen: set[str] = set()
    return [f for f in out if not (f in seen or seen.add(f))]


def _lut_from_manifest(table: Mapping[str, Any] | None) -> dict[int, Any] | None:
    if not table:
        return None
    out: dict[int, Any] = {}
    for key, value in table.items():
        name, rgba = (value + [None])[:2] if isinstance(value, list) else (value, None)
        out[int(key)] = (name, rgba) if rgba is not None else str(name)
    return out


def load_manifest(viewer: Any, manifest: Mapping[str, Any], base_dir: str | Path, *, prefix: str = "") -> int:
    """Open what *manifest* describes (files under *base_dir*); returns the number of layers added."""
    base = Path(base_dir)
    count = 0
    lead = f"{prefix} · " if prefix else ""
    for entry in manifest.get("layers", []) or []:
        path = base / entry["file"]
        if not path.is_file():
            continue
        add_volume(viewer, path, name=f"{lead}{entry.get('title') or path.name}", labels=bool(entry.get("labels")),
                   lut=_lut_from_manifest(entry.get("lut")), schema=entry.get("schema"),
                   colormap=entry.get("colormap"), visible=bool(entry.get("visible", True)))
        count += 1
    if manifest.get("surfaces"):
        from nvitk.segmentation.fastsurfer.outputs import read_morph, read_surface

        for entry in manifest["surfaces"]:
            path = base / entry["file"]
            if not path.is_file():
                continue
            vertices, faces = read_surface(path)
            values = None
            if entry.get("values") and (base / entry["values"]).is_file():
                values = read_morph(base / entry["values"])
            add_surface(viewer, vertices, faces, name=f"{lead}{entry.get('title') or path.name}", source=path,
                        values=values, colormap="inferno" if values is not None else "gray",
                        visible=bool(entry.get("visible", True)))
            count += 1
    if manifest.get("tables"):
        from nvitk.gui.viz.results_window import show_results

        for table in manifest["tables"]:
            if table.get("payload"):
                show_results(viewer, f"{lead}{table.get('title', '')}", table["payload"],
                             subtitle=str(table.get("subtitle") or ""), row_header=str(table.get("row_header") or ""))
    for note in manifest.get("notes", []) or []:
        from nvitk.gui.tools.runner import notify

        notify(str(note))
    return count


__all__ = [
    "StagedInputs",
    "ToolInputs",
    "ViewerInputs",
    "add_surface",
    "load_manifest",
    "manifest_files",
    "manifest_layer",
    "add_volume",
    "export_layer_file",
    "file_stem",
    "layer_source_path",
    "path_param",
    "run_directory",
    "sidecar_file",
]
