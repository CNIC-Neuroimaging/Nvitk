"""Scenes: the whole viewer saved to one file, and opened again where it was left.

A scene (``*.nvscene``) holds every layer — images, labels, surfaces, points,
shapes, vectors, tracks — with its data *and* its state: placement (scale,
translate, affine), display (colormap, contrast, opacity, blending, visibility,
label colours and names, the label list's shown labels), metadata (the source
file's affine and axes, so a saved image still exports as it would have), the
layer folders, and nvitk's own per-layer state (a mesh's fields and colouring, a
mesh series, an image used as a label mask). With them, the viewer: 2D/3D, the
axis order, the slice shown, the camera, the grid, the active layer. Not the
tool windows (the vessel CPR, the cross-sections) nor the overlays they draw on
the canvas: those are run again on the restored layers.

The file is a zip archive:

* ``scene.json`` — the manifest: format version, the layers in order (bottom to
  top) with their kind and entries, the viewer's state;
* ``layers/NNN.data.npy`` — a layer's array, a standard ``.npy`` any tool reads
  (``layers/NNN.data.pkl`` for data that is not one array: a surface's
  vertices and faces, shapes, a multiscale pyramid);
* ``layers/NNN.meta.pkl`` — the layer's Napari state (``as_layer_data_tuple``);
  Napari model objects (label colormaps) are stored as their class and fields
  and rebuilt on load, so the file does not depend on how they pickle;
* ``layers/NNN.nvitk.pkl`` — nvitk's own state for the layer, when it has any.

The state is pickled: open scenes you (or someone you trust) saved. Arrays are
stored uncompressed by default (fast); *compress* trades time for size.

Saving and opening run their file work on a worker thread (the activity
indicator shows how far it got); the layers are taken and added on the GUI
thread.
"""

from __future__ import annotations

import datetime
import importlib
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from nvitk.core.array import to_numpy

#: Scene file suffix.
SCENE_SUFFIX = ".nvscene"
#: Bumped when the layout of the archive changes.
FORMAT_VERSION = 1
_MANIFEST = "scene.json"


# ──────────────────────────────────────────────────────────────────────────────
# Napari state ↔ picklable
# ──────────────────────────────────────────────────────────────────────────────


def _picklable(value: Any) -> bool:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception:  # noqa: BLE001
        return False


def _encode(value: Any) -> Any:
    """*value* as something that pickles: arrays and plain values as they are,
    containers element by element (so a large array in the metadata is never
    pickled just to test it), a Napari / pydantic model as its class and fields."""
    if value is None or isinstance(value, (np.ndarray, np.generic, str, bytes, int, float, bool, complex)):
        return value
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and type(value) in (list, tuple):
        return type(value)(_encode(v) for v in value)
    model_fields = getattr(type(value), "model_fields", None) or getattr(type(value), "__fields__", None)
    if model_fields is not None:
        dump = value.model_dump() if hasattr(value, "model_dump") else value.dict()
        return {"__nvitk_model__": f"{type(value).__module__}:{type(value).__qualname__}",
                "fields": {k: _encode(v) for k, v in dump.items()}}
    if _picklable(value):
        return value
    raise TypeError(f"cannot store a {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if "__nvitk_model__" in value:
            module, name = str(value["__nvitk_model__"]).split(":", 1)
            cls: Any = importlib.import_module(module)
            for part in name.split("."):
                cls = getattr(cls, part)
            return cls(**{k: _decode(v) for k, v in dict(value["fields"]).items()})
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_decode(v) for v in value)
    return value


def _jsonable(value: Any) -> Any:
    """Viewer state for the JSON manifest (arrays and tuples become lists)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Collecting (GUI thread)
# ──────────────────────────────────────────────────────────────────────────────


def _nvitk_state(viewer: Any, layer: Any) -> dict[str, Any]:
    """nvitk's own state for *layer* (nothing Napari keeps)."""
    out: dict[str, Any] = {}
    flag = getattr(layer, "_nvitk_label_like", None)
    if flag is not None:
        out["label_like"] = bool(flag)
    try:
        from nvitk.gui.mesh import layers as mesh_layers

        entry = mesh_layers._DATA.get(layer)
        if entry is not None:
            out["mesh"] = entry
        ctrl = mesh_layers.series_controller(layer)
        if ctrl is not None:
            out["mesh_series"] = ctrl.series
    except Exception:  # noqa: BLE001 — a mesh state that will not come along is left out
        pass
    return out


def _viewer_state(viewer: Any) -> dict[str, Any]:
    dims, camera = viewer.dims, viewer.camera
    state: dict[str, Any] = {
        "ndisplay": int(dims.ndisplay),
        "order": [int(v) for v in dims.order],
        "current_step": [int(v) for v in dims.current_step],
        "camera": {
            "center": [float(v) for v in camera.center],
            "zoom": float(camera.zoom),
            "angles": [float(v) for v in camera.angles],
            "perspective": float(getattr(camera, "perspective", 0.0)),
        },
        "grid": {"enabled": bool(viewer.grid.enabled), "shape": list(viewer.grid.shape),
                 "stride": int(viewer.grid.stride)},
    }
    active = viewer.layers.selection.active
    if active is not None:
        state["active"] = str(active.name)
    folders = getattr(viewer, "_nvitk_layer_folders", None)
    if folders is not None:
        state["folders_collapsed"] = sorted(str(f) for f in getattr(folders, "collapsed", ()) or ())
    return state


def _tool_overlay_names() -> set[str]:
    """Layers an open tool window draws for itself (the vessel CPR's centerline,
    control points, station and plane; the cross-sections' centerlines and
    picks): without their window they are dead weight, and the tool draws them
    again when it is run."""
    names: set[str] = set()
    try:
        from nvitk.gui.viz.vessel_cpr_panel import OVERLAY_LAYERS

        names.update(OVERLAY_LAYERS)
    except Exception:  # noqa: BLE001
        pass
    try:
        from nvitk.gui.viz.vessel_cross_sections import XS_CENTERLINES, XS_CL_POINTS, XS_PICK

        names.update((XS_CENTERLINES, XS_CL_POINTS, XS_PICK))
    except Exception:  # noqa: BLE001
        pass
    return names


def collect_scene(viewer: Any) -> dict[str, Any]:
    """Everything a scene holds, taken from *viewer* (GUI thread; the data are the
    layers' arrays, read on the host)."""
    layers = []
    overlays = _tool_overlay_names()
    for layer in viewer.layers:
        if getattr(layer, "_nvitk_internal", False) or str(layer.name) in overlays:
            continue  # previews, outlines and overlays the tools draw for themselves
        data, meta, kind = layer.as_layer_data_tuple()
        if isinstance(data, (list, tuple)):
            data = type(data)(to_numpy(d) if hasattr(d, "__array__") else d for d in data)
        elif hasattr(data, "__array__"):
            data = to_numpy(data)
        encoded, dropped = {}, []
        for key, value in meta.items():
            try:
                encoded[key] = _encode(value)
            except Exception:  # noqa: BLE001 — a property that cannot be stored is reported, not fatal
                dropped.append(key)
        layers.append({"name": str(layer.name), "kind": str(kind), "data": data, "meta": encoded,
                       "nvitk": _nvitk_state(viewer, layer), "dropped": dropped})
    return {"layers": layers, "viewer": _viewer_state(viewer)}


# ──────────────────────────────────────────────────────────────────────────────
# Writing and reading (any thread)
# ──────────────────────────────────────────────────────────────────────────────


def _write_pickle(zf: zipfile.ZipFile, name: str, obj: Any) -> None:
    with zf.open(name, "w", force_zip64=True) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_scene(path: str | Path, scene: dict[str, Any], *, compress: bool = False,
                progress: Callable[[int, int], None] | None = None) -> Path:
    """Write *scene* (:func:`collect_scene`) to *path* (``.nvscene`` added when
    missing). No Qt: safe on a worker thread."""
    path = Path(path)
    if path.suffix.lower() != SCENE_SUFFIX:
        path = path.with_name(path.name + SCENE_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    entries = []
    total = len(scene["layers"])
    with zipfile.ZipFile(tmp, "w", compression=method, compresslevel=1 if compress else None,
                         allowZip64=True) as zf:
        for k, item in enumerate(scene["layers"]):
            if progress is not None:
                progress(k, total)
            stem = f"layers/{k:03d}"
            entry = {"name": item["name"], "kind": item["kind"], "meta": f"{stem}.meta.pkl",
                     "dropped": list(item.get("dropped") or [])}
            data = item["data"]
            if isinstance(data, np.ndarray) and data.dtype != object:
                entry["data"] = f"{stem}.data.npy"
                with zf.open(entry["data"], "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, np.ascontiguousarray(data), allow_pickle=False)
            else:
                entry["data"] = f"{stem}.data.pkl"
                _write_pickle(zf, entry["data"], data)
            _write_pickle(zf, entry["meta"], item["meta"])
            if item.get("nvitk"):
                entry["nvitk"] = f"{stem}.nvitk.pkl"
                _write_pickle(zf, entry["nvitk"], item["nvitk"])
            entries.append(entry)
        manifest = {
            "format": "nvitk-scene",
            "version": FORMAT_VERSION,
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
            "versions": _versions(),
            "layers": entries,
            "viewer": _jsonable(scene.get("viewer") or {}),
        }
        zf.writestr(_MANIFEST, json.dumps(manifest, indent=2))
        if progress is not None:
            progress(total, total)
    tmp.replace(path)
    return path


def _versions() -> dict[str, str]:
    out = {}
    for name in ("nvitk", "napari", "numpy"):
        try:
            out[name] = str(importlib.import_module(name).__version__)
        except Exception:  # noqa: BLE001
            out[name] = "?"
    return out


def read_manifest(path: str | Path) -> dict[str, Any]:
    """The scene's manifest (layers, viewer state) without reading any data."""
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read(_MANIFEST).decode("utf-8"))
    if manifest.get("format") != "nvitk-scene":
        raise ValueError(f"{Path(path).name} is not an nvitk scene.")
    if int(manifest.get("version", 0)) > FORMAT_VERSION:
        raise ValueError(f"{Path(path).name} was saved by a newer nvitk (scene format "
                         f"{manifest.get('version')}); update nvitk to open it.")
    return manifest


def read_scene(path: str | Path, *, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Read a scene file back into the form :func:`collect_scene` makes. No Qt:
    safe on a worker thread."""
    manifest = read_manifest(path)
    layers = []
    with zipfile.ZipFile(path) as zf:
        total = len(manifest["layers"])
        for k, entry in enumerate(manifest["layers"]):
            if progress is not None:
                progress(k, total)
            with zf.open(entry["data"]) as f:
                if entry["data"].endswith(".npy"):
                    data = np.lib.format.read_array(f, allow_pickle=False)
                else:
                    data = pickle.load(f)
            with zf.open(entry["meta"]) as f:
                meta = pickle.load(f)
            extra = {}
            if entry.get("nvitk"):
                with zf.open(entry["nvitk"]) as f:
                    extra = pickle.load(f)
            layers.append({"name": entry["name"], "kind": entry["kind"], "data": data, "meta": meta,
                           "nvitk": extra, "dropped": entry.get("dropped") or []})
        if progress is not None:
            progress(total, total)
    return {"layers": layers, "viewer": manifest.get("viewer") or {}, "manifest": manifest}


# ──────────────────────────────────────────────────────────────────────────────
# Applying (GUI thread)
# ──────────────────────────────────────────────────────────────────────────────

#: What a layer is rebuilt from when its full state will not load (another Napari).
_CORE_KEYS = ("name", "metadata", "scale", "translate", "rotate", "shear", "affine", "opacity",
              "blending", "visible", "colormap", "contrast_limits", "gamma", "axis_labels", "units")


def _create_layer(data: Any, meta: dict[str, Any], kind: str) -> tuple[Any, list[str]]:
    """The Napari layer for *data*/*meta*; falls back to fewer properties when a
    stored one is not understood. Returns the layer and the properties left out."""
    from napari.layers import Layer

    meta = {k: _decode(v) for k, v in meta.items()}
    try:
        return Layer.create(data, meta, kind), []
    except Exception:  # noqa: BLE001 — try with less below
        pass
    kept = {k: v for k, v in meta.items() if k in _CORE_KEYS}
    try:
        return Layer.create(data, kept, kind), sorted(set(meta) - set(kept))
    except Exception:  # noqa: BLE001
        pass
    minimal = {k: meta[k] for k in ("name", "metadata", "scale", "translate", "affine") if k in meta}
    return Layer.create(data, minimal, kind), sorted(set(meta) - set(minimal))


def _apply_nvitk_state(viewer: Any, layer: Any, state: dict[str, Any]) -> None:
    if state.get("label_like") is not None:
        layer._nvitk_label_like = bool(state["label_like"])
    entry = state.get("mesh")
    if entry is not None:
        from nvitk.gui.mesh import layers as mesh_layers

        mesh_layers._DATA[layer] = entry
        try:
            mesh_layers.apply_display(layer)
        except Exception:  # noqa: BLE001
            pass


def _add_series_layer(viewer: Any, item: dict[str, Any]) -> Any:
    """A mesh series comes back through nvitk's own series layer (it animates)."""
    from nvitk.gui.mesh import layers as mesh_layers

    meta = {k: _decode(v) for k, v in item["meta"].items()}
    layer = mesh_layers.add_mesh_series_layer(viewer, item["nvitk"]["mesh_series"], name=item["name"])
    for key in ("opacity", "blending", "visible", "metadata", "scale", "translate", "affine", "shading"):
        if key in meta:
            try:
                setattr(layer, key, meta[key])
            except Exception:  # noqa: BLE001
                pass
    return layer


def _apply_viewer_state(viewer: Any, state: dict[str, Any], layers: list[Any]) -> None:
    dims = viewer.dims
    # Axis order and slice first, while the canvas is still 2D (cheap), then 3D
    # once: in 3D each of them would re-slice and re-upload every volume.
    try:
        order = state.get("order")
        if order and len(order) == int(dims.ndim) and sorted(order) == list(range(int(dims.ndim))):
            dims.order = tuple(int(v) for v in order)
        step = state.get("current_step")
        if step and len(step) == int(dims.ndim):
            dims.current_step = tuple(int(v) for v in step)
        if "ndisplay" in state:
            dims.ndisplay = int(state["ndisplay"])
    except Exception:  # noqa: BLE001 — a viewer shaped otherwise keeps its own dims
        pass
    camera = state.get("camera") or {}
    for key in ("center", "zoom", "angles", "perspective"):
        if key in camera:
            try:
                value = camera[key]
                setattr(viewer.camera, key, tuple(value) if isinstance(value, list) else value)
            except Exception:  # noqa: BLE001
                pass
    grid = state.get("grid") or {}
    try:
        if "shape" in grid:
            viewer.grid.shape = tuple(grid["shape"])
        if "stride" in grid:
            viewer.grid.stride = int(grid["stride"])
        if "enabled" in grid:
            viewer.grid.enabled = bool(grid["enabled"])
    except Exception:  # noqa: BLE001
        pass
    active = state.get("active")
    for layer in layers:
        if str(layer.name) == active:
            viewer.layers.selection.active = layer
            break


def apply_scene(viewer: Any, scene: dict[str, Any], *, replace: bool = True) -> tuple[list[Any], list[str]]:
    """Add *scene*'s layers to *viewer* (after clearing it, with *replace*) and
    restore its view. Returns the layers added and what could not be restored."""
    problems: list[str] = []
    if replace:
        _close_tool_windows(viewer)
    # Layers go in with the canvas in 2D: in 3D every layer added or removed
    # re-slices and re-uploads every volume already there. The view the scene
    # was saved with (3D included) is put back once they are all in.
    _show_2d(viewer)
    if replace:
        _clear_layers(viewer)
    added = []
    # The layer list gives an inserted layer the folder of where it lands; the
    # scene's layers already carry theirs (in their metadata), so that waits.
    folders = getattr(viewer, "_nvitk_layer_folders", None)
    moving = getattr(folders, "_moving", False) if folders is not None else False
    if folders is not None:
        folders._moving = True
    try:
        added = _add_scene_layers(viewer, scene, problems)
    finally:
        if folders is not None:
            folders._moving = moving
            collapsed = (scene.get("viewer") or {}).get("folders_collapsed")
            if collapsed is not None:
                folders.collapsed = set(folders.collapsed) | {str(f) for f in collapsed} if not replace \
                    else {str(f) for f in collapsed}
            folders.invalidate()
    _apply_viewer_state(viewer, scene.get("viewer") or {}, added)
    return added, problems


def _show_2d(viewer: Any) -> None:
    try:
        if int(viewer.dims.ndisplay) != 2:
            viewer.dims.ndisplay = 2
    except Exception:  # noqa: BLE001
        pass


def _close_tool_windows(viewer: Any) -> None:
    """Close the tool windows bound to the layers a replacing scene removes (the
    vessel CPR, the cross-sections): left open, they would keep drawing their
    overlays and following the time slider for layers that are gone."""
    app_state = getattr(viewer, "_nvitk_app_state", None)
    if not isinstance(app_state, dict):
        return
    try:
        from nvitk.gui.viz.vessel_cpr_panel import shutdown_vessel_cpr

        shutdown_vessel_cpr(app_state)
    except Exception:  # noqa: BLE001
        pass
    try:
        from nvitk.gui.viz.vessel_cross_sections import shutdown_vessel_cross_sections

        shutdown_vessel_cross_sections(app_state)
    except Exception:  # noqa: BLE001
        pass


def _clear_layers(viewer: Any) -> None:
    try:
        viewer.layers.clear()
    except Exception:  # noqa: BLE001 — what is left goes one by one below
        pass
    for layer in reversed(list(viewer.layers)):
        _discard(viewer, layer)


def _discard(viewer: Any, layer: Any) -> None:
    """Take *layer* out of the viewer, if it got in."""
    try:
        if layer in viewer.layers:
            viewer.layers.remove(layer)
    except Exception:  # noqa: BLE001
        pass


def _add_scene_layers(viewer: Any, scene: dict[str, Any], problems: list[str]) -> list[Any]:
    added = []
    overlays = _tool_overlay_names()  # in scenes saved before they were left out
    for item in scene["layers"]:
        if str(item["name"]) in overlays:
            continue
        layer = None
        try:
            if (item.get("nvitk") or {}).get("mesh_series") is not None:
                layer = _add_series_layer(viewer, item)
                lost: list[str] = []
            else:
                layer, lost = _create_layer(item["data"], item["meta"], item["kind"])
                viewer.add_layer(layer)
            _apply_nvitk_state(viewer, layer, item.get("nvitk") or {})
        except Exception as exc:  # noqa: BLE001 — one layer must not stop the rest
            # A layer the viewer took but could not draw would sit in the list
            # with no visual, and every layer added after it would fail as well.
            if layer is not None:
                _discard(viewer, layer)
            problems.append(f"“{item['name']}” not restored: {exc}")
            continue
        added.append(layer)
        left = list(item.get("dropped") or []) + list(lost)
        if left:
            problems.append(f"“{item['name']}”: {', '.join(left)} not restored")
    return added


# ──────────────────────────────────────────────────────────────────────────────
# Save / open, with the activity indicator
# ──────────────────────────────────────────────────────────────────────────────


def save_scene(viewer: Any, path: str | Path, *, compress: bool = False) -> Path:
    """Save *viewer* as a scene at *path* (the file work on a worker thread)."""
    from nvitk.gui.core.activity import activity, run_in_background

    with activity(f"Saving scene {Path(path).name}") as task:
        scene = collect_scene(viewer)
        task.update(total=len(scene["layers"]))
        out = run_in_background(write_scene, path, scene, compress=compress,
                                progress=lambda done, total: task.update(done=done, total=total))
    return out


def load_scene(viewer: Any, path: str | Path, *, replace: bool = True) -> tuple[list[Any], list[str]]:
    """Open the scene at *path* in *viewer* (read on a worker thread, layers added
    here). Returns the layers added and what could not be restored."""
    from nvitk.gui.core.activity import activity, run_in_background

    with activity(f"Opening scene {Path(path).name}") as task:
        scene = run_in_background(read_scene, path,
                                  progress=lambda done, total: task.update(done=done, total=total))
        task.update(message=f"Opening scene {Path(path).name}: adding the layers")
        return apply_scene(viewer, scene, replace=replace)


def is_scene_file(path: str | Path) -> bool:
    return str(path).lower().endswith(SCENE_SUFFIX)


def open_scene_interactive(viewer: Any, path: str | Path, parent: Any = None) -> list[Any]:
    """Open a scene from the GUI: ask whether it replaces the layers there are,
    and report what could not be restored."""
    from qtpy.QtWidgets import QMessageBox

    replace = True
    if len(viewer.layers):
        box = QMessageBox(parent)
        box.setWindowTitle("Open scene")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"Open {Path(path).name}?")
        box.setInformativeText("Replace the layers now in the viewer, or add the scene's to them?")
        replace_btn = box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
        add_btn = box.addButton("Add", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (replace_btn, add_btn):
            return []
        replace = clicked is replace_btn
    try:
        layers, problems = load_scene(viewer, path, replace=replace)
    except Exception as exc:  # noqa: BLE001
        _notify(f"Could not open the scene {Path(path).name}: {exc}", error=True)
        return []
    message = f"Scene {Path(path).name}: {len(layers)} layer(s) opened."
    if problems:
        message += " " + "; ".join(problems[:4]) + (" …" if len(problems) > 4 else "")
    _notify(message, error=bool(problems))
    return layers


def build_scene_panel(viewer: Any) -> Any:
    """The Export tab's *Scene* card: save the viewer as a scene, open one."""
    from qtpy.QtWidgets import QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton

    from nvitk.gui.core.design import COLOR_MUTED, Card

    card = Card("Scene")
    card.setToolTip("The whole viewer in one file: every layer with its data, placement, colours and "
                    "metadata, the layer folders, and the view (2D/3D, slice, camera) — to open again "
                    "where you left it.")
    note = QLabel("Every layer and the view, in one file to open again later.")
    note.setWordWrap(True)
    note.setStyleSheet(f"color: {COLOR_MUTED};")
    compress = QCheckBox("Compress (smaller file, slower)")
    status = QLabel("")
    status.setWordWrap(True)
    status.setStyleSheet(f"color: {COLOR_MUTED};")
    save_btn = QPushButton("Save scene…")
    open_btn = QPushButton("Open scene…")
    state = {"dir": str(Path.home())}

    def save() -> None:
        if not len(viewer.layers):
            status.setText("Nothing to save: the viewer has no layers.")
            return
        name, _f = QFileDialog.getSaveFileName(card, "Save scene", str(Path(state["dir"]) / f"scene{SCENE_SUFFIX}"),
                                               f"nvitk scene (*{SCENE_SUFFIX})")
        if not name:
            return
        try:
            out = save_scene(viewer, name, compress=compress.isChecked())
        except Exception as exc:  # noqa: BLE001
            status.setText(f"Could not save: {exc}")
            _notify(f"Could not save the scene: {exc}", error=True)
            return
        state["dir"] = str(out.parent)
        size = out.stat().st_size / 1e6
        status.setText(f"Saved {out.name} ({size:,.1f} MB).")
        _notify(f"Scene saved: {out}")

    def open_() -> None:
        name, _f = QFileDialog.getOpenFileName(card, "Open scene", state["dir"], f"nvitk scene (*{SCENE_SUFFIX})")
        if not name:
            return
        state["dir"] = str(Path(name).parent)
        layers = open_scene_interactive(viewer, name, parent=card)
        if layers:
            status.setText(f"Opened {Path(name).name}: {len(layers)} layer(s).")

    save_btn.clicked.connect(save)
    open_btn.clicked.connect(open_)
    card.add(note)
    row = QHBoxLayout()
    row.addWidget(save_btn)
    row.addWidget(open_btn)
    card.add_layout(row)
    card.add(compress)
    card.add(status)
    return card


def _notify(message: str, *, error: bool = False) -> None:
    try:
        from nvitk.gui.tools.runner import notify

        notify(message, error=error)
    except Exception:  # noqa: BLE001
        print(message, flush=True)


__all__ = [
    "FORMAT_VERSION",
    "SCENE_SUFFIX",
    "apply_scene",
    "build_scene_panel",
    "collect_scene",
    "is_scene_file",
    "load_scene",
    "open_scene_interactive",
    "read_manifest",
    "read_scene",
    "save_scene",
    "write_scene",
]
