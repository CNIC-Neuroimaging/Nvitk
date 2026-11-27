"""Imaging tools on a worker thread, so the GUI stays usable while they compute.

A tool from the Imaging tab mixes computing (an array, a registration, a network) with changing
the viewer (adding the result, sometimes more layers). Only the computing may leave the GUI
thread. :func:`start_background_tool` runs the tool's own code on a worker thread against
*stand-ins*:

* every layer is seen through a read-only view (:class:`_LayerView`): its data, metadata and its
  **real** placement (with resampling off on the main canvas too: the view answers with the
  transform the layer really has, which the foreground run gets from ``world_geometry``). Setting
  an attribute on it, or calling one of its methods that would emit events, stops the run;
* the viewer (:class:`_ViewerView`) lists those views and *records* what the tool does to it:
  ``add_image`` / ``add_labels`` / ... (it gets a placeholder layer back, whose own settings are
  recorded as well), removing a layer, ``reset_view``. The recording is replayed on the GUI
  thread when the tool is done, followed by what the Imaging tab does with the returned array.

A tool that reaches for anything else (a dock, a window, the canvas, a dialog) raises
:class:`ForegroundOnly` there, and is run again on the GUI thread, as before, with a note in
the log. Tools that manage their own background job (FSL, FastSurfer, FreeSurfer), open windows
or plots (visualization, measurements) or run a pipeline never go to the worker.

The worker runs in a copy of the GUI thread's context, so the CPU/GPU backend chosen there
applies, and every log line or notification it makes is shown on the GUI thread.
"""

from __future__ import annotations

import contextvars
import threading
import time
import traceback
from typing import Any, Callable

import numpy as np

#: Preferences key: run eligible tools in the background (default on).
PREF_KEY = "tools.background"

#: Task sections whose tools compute in-process and hand their result back.
BACKGROUND_SECTIONS = frozenset({
    "preprocess", "filters", "morphology", "masks", "segmentation", "registration", "transform",
    "time", "vessels",
})
#: Never on the worker: these manage their own job, open windows, or interact.
FOREGROUND_PREFIXES = ("fsl_", "fastsurfer_", "freesurfer_", "viz_", "export_", "measure_", "qvtpy",
                       "lab_", "pipeline_")
FOREGROUND_IDS = frozenset({
    # Interactive or viewer-driving vessel tools.
    "centerline_detect_junctions", "centerline_to_polyline", "polyline_to_centerline",
    "centerline_cut_junctions", "measure_centerline_arc_length",
})


class ForegroundOnly(RuntimeError):
    """The tool touched something only the GUI thread may: run it there instead."""


def background_enabled() -> bool:
    try:
        from nvitk.gui.core.prefs import load_prefs

        return bool(load_prefs().get(PREF_KEY, True))
    except Exception:  # noqa: BLE001
        return True


def set_background_enabled(on: bool) -> None:
    try:
        from nvitk.gui.core.prefs import save_prefs

        save_prefs({PREF_KEY: bool(on)})
    except Exception:  # noqa: BLE001
        pass


def can_run_in_background(tool_id: str) -> bool:
    """Whether *tool_id* is one that may compute on a worker thread."""
    from nvitk.gui.tools.registry import tool_by_id
    from nvitk.gui.tools.taxonomy import meta_for

    spec = tool_by_id(tool_id)
    # "layer" tools return an array; "notify" ones add their layers themselves (recorded on the
    # worker, replayed here). Pipelines run a command of their own.
    if spec is None or spec.run_mode not in ("layer", "notify"):
        return False
    if tool_id.startswith(FOREGROUND_PREFIXES) or tool_id in FOREGROUND_IDS:
        return False
    try:
        section = meta_for(spec).section
    except Exception:  # noqa: BLE001
        return False
    return section in BACKGROUND_SECTIONS


# stand-ins -------------------------------------------------------------------------------------

#: Layer methods a tool may call from the worker: they only read.
_SAFE_LAYER_METHODS = frozenset({
    "data_to_world", "world_to_data", "get_value", "_world_to_data_ray", "_world_to_displayed_data",
    "_world_to_layer_dims", "_display_bounding_box", "_get_ndim", "as_layer_data_tuple",
    "_get_state", "_get_base_state",
})
_TRANSFORM_PARTS = ("scale", "translate", "rotate", "shear")


class _LayerView:
    """A read-only view of a Napari layer for a tool on the worker thread."""

    __slots__ = ("_real", "_world")

    def __init__(self, real: Any, world: "_World") -> None:
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_world", world)

    # isinstance(view, napari.layers.Labels) holds, as for the layer itself.
    @property
    def __class__(self) -> type:  # type: ignore[override]
        return type(object.__getattribute__(self, "_real"))

    def _parts(self) -> dict[str, Any] | None:
        from nvitk.gui.viz.main_canvas import true_transform

        return true_transform(object.__getattribute__(self, "_real"))

    def _matrix(self) -> np.ndarray:
        from nvitk.gui.viz.main_canvas import real_data_to_world

        return real_data_to_world(object.__getattribute__(self, "_real"))

    def __getattr__(self, name: str) -> Any:
        real = object.__getattribute__(self, "_real")
        parts = self._parts() if name in (*_TRANSFORM_PARTS, "affine", "_data_to_world") else None
        if parts is not None:
            from napari.utils.transforms import Affine

            if name in _TRANSFORM_PARTS:
                return np.array(parts[name], dtype=float)
            if name == "affine":
                return Affine(affine_matrix=np.array(parts["affine"], dtype=float))
            return Affine(affine_matrix=self._matrix())
        if name == "events":
            raise ForegroundOnly(f"layer.{name}")
        value = getattr(real, name)
        if callable(value) and getattr(value, "__self__", None) is real:
            if name in ("data_to_world", "world_to_data") and self._parts() is not None:
                return getattr(self, f"_real_{name}")
            if name not in _SAFE_LAYER_METHODS:
                raise ForegroundOnly(f"layer.{name}()")
        return value

    def _real_data_to_world(self, position: Any) -> tuple[float, ...]:
        m = self._matrix()
        n = m.shape[0] - 1
        p = np.asarray(position, dtype=float)[-n:]
        return tuple(float(v) for v in (m @ np.append(p, 1.0))[:n])

    def _real_world_to_data(self, position: Any) -> tuple[float, ...]:
        m = self._matrix()
        n = m.shape[0] - 1
        p = np.asarray(position, dtype=float)[-n:]
        return tuple(float(v) for v in np.linalg.solve(m, np.append(p, 1.0))[:n])

    def __setattr__(self, name: str, value: Any) -> None:
        raise ForegroundOnly(f"layer.{name} = …")

    def __repr__(self) -> str:
        return f"<background view of {object.__getattribute__(self, '_real')!r}>"

    def __eq__(self, other: Any) -> bool:
        return _unwrap(self) is _unwrap(other)

    def __hash__(self) -> int:
        return id(object.__getattribute__(self, "_real"))


_VIEW_CLASSES: dict[type, type] = {}


def _view_class(layer_type: type) -> type:
    """A :class:`_LayerView` subclass named like *layer_type* (``Image``, ``Labels``...): code
    that checks ``type(layer).__name__`` treats the view as the layer it stands for."""
    cls = _VIEW_CLASSES.get(layer_type)
    if cls is None:
        cls = type(layer_type.__name__, (_LayerView,), {"__slots__": (), "__module__": __name__})
        _VIEW_CLASSES[layer_type] = cls
    return cls


class _Pending:
    """What ``viewer.add_*`` returns on the worker: a layer to be made on the GUI thread.
    Its name, data and metadata read back from the call; settings and calls on it are recorded."""

    def __init__(self, method: str, args: tuple, kwargs: dict[str, Any]) -> None:
        object.__setattr__(self, "_method", method)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "_ops", [])
        object.__setattr__(self, "_real", None)

    def __getattr__(self, name: str) -> Any:
        kwargs = object.__getattribute__(self, "_kwargs")
        args = object.__getattribute__(self, "_args")
        if name == "data":
            return kwargs.get("data", args[0] if args else None)
        if name in kwargs:
            return kwargs[name]
        if name == "metadata":
            kwargs["metadata"] = {}
            return kwargs["metadata"]
        if name.startswith("__"):
            raise AttributeError(name)
        ops = object.__getattribute__(self, "_ops")

        def call(*a: Any, **k: Any) -> None:
            ops.append(("call", name, a, k))

        return call

    def __setattr__(self, name: str, value: Any) -> None:
        object.__getattribute__(self, "_ops").append(("set", name, value))

    def _make(self, viewer: Any) -> Any:
        method = object.__getattribute__(self, "_method")
        args = tuple(_unwrap(a) for a in object.__getattribute__(self, "_args"))
        kwargs = {k: _unwrap(v) for k, v in object.__getattribute__(self, "_kwargs").items()}
        real = getattr(viewer, method)(*args, **kwargs)
        object.__setattr__(self, "_real", real)
        for op in object.__getattribute__(self, "_ops"):
            if op[0] == "set":
                setattr(real, op[1], _unwrap(op[2]))
            else:
                getattr(real, op[1])(*[_unwrap(a) for a in op[2]], **{k: _unwrap(v) for k, v in op[3].items()})
        return real


def _unwrap(value: Any) -> Any:
    try:
        return object.__getattribute__(value, "_real")
    except AttributeError:
        return value


class _SelectionView:
    def __init__(self, world: "_World") -> None:
        self._world = world

    @property
    def active(self) -> Any:
        real = self._world.active
        return self._world.view(real) if real is not None else None

    def __iter__(self):
        return iter([self._world.view(l) for l in self._world.selected])

    def __contains__(self, item: Any) -> bool:
        return _unwrap(item) in self._world.selected

    def __len__(self) -> int:
        return len(self._world.selected)


class _LayerListView:
    def __init__(self, world: "_World") -> None:
        self._world = world
        self.selection = _SelectionView(world)

    def _items(self) -> list[Any]:
        world = self._world
        removed = world.removed
        out = [world.view(l) for l in world.layers if not any(l is r for r in removed)]
        return out + list(world.pending)

    def __iter__(self):
        return iter(self._items())

    def __len__(self) -> int:
        return len(self._items())

    def __getitem__(self, key: Any) -> Any:
        items = self._items()
        if isinstance(key, str):
            for item in items:
                if str(getattr(item, "name", "")) == key:
                    return item
            raise KeyError(key)
        return items[key]

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return any(str(getattr(l, "name", "")) == item for l in self._items())
        target = _unwrap(item)
        return any(_unwrap(l) is target or l is item for l in self._items())

    def index(self, item: Any) -> int:
        target = _unwrap(item)
        for i, l in enumerate(self._items()):
            if _unwrap(l) is target or l is item:
                return i
        raise ValueError(item)

    def remove(self, item: Any) -> None:
        if isinstance(item, _Pending):
            self._world.pending.remove(item)
            self._world.record.append(("drop", item))
            return
        self._world.removed.append(_unwrap(item))
        self._world.record.append(("remove", _unwrap(item)))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise ForegroundOnly(f"viewer.layers.{name}")


class _DimsView:
    """``viewer.dims``, read only (its numbers, not its events)."""

    def __init__(self, dims: Any) -> None:
        self._dims = dims

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._dims, name)
        if name == "events" or callable(value):
            raise ForegroundOnly(f"viewer.dims.{name}")
        return value


_ADD_METHODS = ("add_image", "add_labels", "add_points", "add_shapes", "add_surface", "add_vectors",
                "add_tracks")
#: Viewer attributes a tool may read from the worker (plain Python state, not widgets).
_READABLE_VIEWER_ATTRS = frozenset({"_nvitk_app_state"})


class _ViewerView:
    """The viewer a tool sees on the worker: layers to read, changes recorded for later."""

    def __init__(self, world: "_World") -> None:
        object.__setattr__(self, "_world", world)
        object.__setattr__(self, "layers", _LayerListView(world))
        object.__setattr__(self, "dims", _DimsView(world.viewer.dims))

    def __getattr__(self, name: str) -> Any:
        world = object.__getattribute__(self, "_world")
        if name in _ADD_METHODS:
            def add(*args: Any, **kwargs: Any) -> _Pending:
                pending = _Pending(name, args, kwargs)
                world.pending.append(pending)
                world.record.append(("add", pending))
                return pending

            return add
        if name in ("reset_view", "fit_to_view"):
            return lambda *a, **k: world.record.append(("call", name, a, k))
        if name in _READABLE_VIEWER_ATTRS:
            return getattr(world.viewer, name)
        if name.startswith("_nvitk_"):
            # Optional GUI hooks (panels, the canvas filter): absent here, as in a headless run.
            raise AttributeError(name)
        raise ForegroundOnly(f"viewer.{name}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise ForegroundOnly(f"viewer.{name} = …")


class _World:
    """Snapshot of the viewer's layer list for one background run, and what the run did."""

    def __init__(self, viewer: Any) -> None:
        self.viewer = viewer
        self.layers = list(viewer.layers)
        self.active = viewer.layers.selection.active
        self.selected = [l for l in self.layers if l in viewer.layers.selection]
        self.pending: list[_Pending] = []
        self.removed: list[Any] = []
        self.record: list[tuple[Any, ...]] = []
        self._views: dict[int, _LayerView] = {}

    def view(self, layer: Any) -> Any:
        if isinstance(layer, (_LayerView, _Pending)):
            return layer
        key = id(layer)
        if key not in self._views:
            self._views[key] = _view_class(type(layer))(layer, self)
        return self._views[key]

    def replay(self) -> list[Any]:
        """Do on the real viewer what the tool did to the stand-in (GUI thread); the layers made."""
        made = []
        viewer = self.viewer
        for entry in self.record:
            kind = entry[0]
            if kind == "add":
                made.append(entry[1]._make(viewer))
            elif kind == "remove":
                layer = entry[1]
                if layer in viewer.layers:
                    viewer.layers.remove(layer)
            elif kind == "call":
                getattr(viewer, entry[1])(*entry[2], **entry[3])
        return made


# running ---------------------------------------------------------------------------------------

class BackgroundRun:
    """One tool computing on a worker thread (see :func:`start_background_tool`)."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.started = time.monotonic()
        self.thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


_RUNS: list[BackgroundRun] = []


def running_tools() -> list[BackgroundRun]:
    """Tools computing in the background now."""
    return [run for run in _RUNS if run.running]


def start_background_tool(
    tool_id: str,
    layer: Any,
    viewer: Any,
    *,
    target_mode: str,
    label_ids: list[int] | None,
    params: dict[str, Any],
    title: str,
    on_done: Callable[[Any], None],
    on_error: Callable[[BaseException], None],
    run_foreground: Callable[[], None],
) -> BackgroundRun:
    """Run *tool_id* on a worker thread; then, on the GUI thread, replay its viewer changes and
    call ``on_done(result)`` (or ``on_error(exc)``). A tool that turns out to need the GUI
    thread calls ``run_foreground()`` there instead."""
    from nvitk.gui.core.activity import start_task
    from nvitk.gui.core.log_panel import gui_log
    from nvitk.gui.core.main_thread import call_on_main, install_main_thread_relay
    from nvitk.gui.tools.runner import _run_gui_tool

    install_main_thread_relay()
    world = _World(viewer)
    stand_in = _ViewerView(world)
    target = world.view(layer) if layer is not None else None
    task = start_task(f"{title} (background)")
    run = BackgroundRun(title)
    _RUNS.append(run)
    context = contextvars.copy_context()

    def finish(outcome: str, value: Any) -> None:
        if run in _RUNS:
            _RUNS.remove(run)
        took = time.monotonic() - run.started
        if outcome == "foreground":
            task.finish(True)
            gui_log(f"{title}: needs the GUI thread ({value}); running it in the foreground.")
            run_foreground()
            return
        if outcome == "error":
            task.finish(False)
            on_error(value)
            return
        try:
            world.replay()
            on_done(value)
        except Exception as exc:  # noqa: BLE001
            task.finish(False)
            on_error(exc)
            return
        task.finish(True)
        gui_log(f"{title}: done in the background ({took:.1f} s).")

    def work() -> None:
        try:
            result = context.run(_run_gui_tool, tool_id, target, stand_in, target_mode=target_mode,
                                 label_ids=label_ids, params=params)
        except ForegroundOnly as exc:
            call_on_main(finish, "foreground", str(exc))
            return
        except BaseException as exc:  # noqa: BLE001 (handed to the GUI thread)
            exc.__nvitk_traceback__ = traceback.format_exc()  # type: ignore[attr-defined]
            call_on_main(finish, "error", exc)
            return
        if isinstance(result, (_LayerView, _Pending)):
            result = _unwrap(result)
        call_on_main(finish, "ok", result)

    run.thread = threading.Thread(target=work, name=f"nvitk-tool-{tool_id}", daemon=True)
    run.thread.start()
    return run


__all__ = [
    "BACKGROUND_SECTIONS",
    "BackgroundRun",
    "ForegroundOnly",
    "PREF_KEY",
    "background_enabled",
    "can_run_in_background",
    "running_tools",
    "set_background_enabled",
    "start_background_tool",
]
