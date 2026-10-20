"""What the main Napari canvas draws, apart from the layer list and the orthogonal views.

Three switches act on the main canvas alone:

* **a layer hidden on the canvas only** — ``visible`` stays True, so the layer list,
  the orthogonal views and the tools still see it (the "Main" column of the views'
  layer table);
* **resampling into world space** — Napari draws every layer in world space through
  its affine: an oblique or flipped acquisition is turned and placed where it lies in
  the scanner, and a layer on another grid is resampled onto the screen aligned with
  the others. Off, every image and label layer is drawn **as stored, on its own voxel
  grid**: its voxel size kept, but no rotation, flip, obliquity or offset — the way the
  orthogonal views lay a volume out. Layers on the same grid still overlay exactly;
  layers on other grids are shown, not aligned. Meshes, points and shapes stay in
  world space (they are stored there);
* **the whole canvas** — hidden, the docks (the orthogonal views first) take its
  place and Napari stops rendering it.

Resampling off is a *display* state. Each overridden layer keeps its real transform
on the side (``_nvitk_true_transform``) and Napari gets a scale-only one, so slicing,
picking and painting on the canvas stay consistent with what is drawn. Everything that
computes geometry sees the real one: :func:`~nvitk.gui.core.spatial.layer_affine` and
:func:`~nvitk.gui.core.spatial.layer_spatial_kwargs` read it, and tools, the
orientation buttons and scene saving run inside :func:`world_geometry`, which puts the
real transforms back for their duration.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import numpy as np
from qtpy.QtCore import QObject, QTimer, Signal

from nvitk.gui.core.napari_fixes import CANVAS_HIDDEN_ATTR, canvas_hidden

#: Preference keys (inside the orthogonal views' ``ortho_view`` entry).
PREF_RESAMPLE = "main_resample"
PREF_CANVAS = "main_canvas"

#: The real transform of a layer drawn on its own grid (see :func:`true_transform`).
TRUE_ATTR = "_nvitk_true_transform"
#: The data → world matrix the override gave Napari, to recognise a later outside write.
SHOWN_ATTR = "_nvitk_native_shown"
#: The components the override set (an outside write changes some of them, not all).
SHOWN_PARTS_ATTR = "_nvitk_native_parts"

_TOL = 1e-6


# ── transforms ───────────────────────────────────────────────────────────────


def _parts(layer: Any) -> dict[str, np.ndarray]:
    """Napari's transform components of *layer* (copies)."""
    return {
        "scale": np.array(layer.scale, dtype=float),
        "translate": np.array(layer.translate, dtype=float),
        "rotate": np.array(layer.rotate, dtype=float),
        "shear": np.array(layer.shear, dtype=float),
        "affine": np.array(layer.affine.affine_matrix, dtype=float),
    }


def _composite(layer: Any) -> np.ndarray:
    """*layer*'s full data → world matrix as Napari composes it."""
    return np.asarray(layer._data_to_world.affine_matrix, dtype=float)


def true_transform(layer: Any) -> dict[str, np.ndarray] | None:
    """The real transform components of a layer drawn on its own grid, else ``None``."""
    value = getattr(layer, TRUE_ATTR, None)
    return value if isinstance(value, dict) else None


def _set_parts(layer: Any, parts: dict[str, np.ndarray]) -> None:
    from nvitk.gui.core.orientation import suppress_nonorthogonal_slice_warning

    with suppress_nonorthogonal_slice_warning():
        layer.affine = parts["affine"]
        layer.rotate = parts["rotate"]
        layer.shear = parts["shear"]
        layer.scale = parts["scale"]
        layer.translate = parts["translate"]


def _native_parts(layer: Any, composite: np.ndarray) -> dict[str, np.ndarray]:
    """Scale-only components drawing *layer* on its own voxel grid: each axis's voxel
    size (the length of its column in the real data → world matrix), nothing else."""
    n = composite.shape[0] - 1
    spacing = np.linalg.norm(composite[:n, :n], axis=0)
    spacing = np.where(spacing > 1e-12, spacing, 1.0)
    return {
        "scale": spacing,
        "translate": np.zeros(n),
        "rotate": np.eye(n),
        "shear": np.zeros_like(np.array(layer.shear, dtype=float)),
        "affine": np.eye(n + 1),
    }


def overridable(layer: Any) -> bool:
    """Whether *layer* is drawn on its own grid when resampling is off.

    Image and Labels layers — the voxel grids — and the orthogonal views' box outlines,
    which are in their volume's voxel coordinates. Meshes, points and shapes are stored
    in world space and stay there (the Mesh panel also moves them through their affine).
    """
    kind = type(layer).__name__
    if kind in ("Image", "Labels"):
        return True
    if kind == "Shapes":
        from nvitk.gui.viz.ortho_panel import _BOX_LAYER_SUFFIX

        return _BOX_LAYER_SUFFIX in str(getattr(layer, "name", ""))
    return False


class MainCanvasFilter(QObject):
    """Decides, layer by layer, whether and how the main canvas draws it."""

    #: Something this filter decides changed (a toggle, a layer hidden, a transform).
    changed = Signal()

    def __init__(self, viewer: Any) -> None:
        super().__init__()
        from nvitk.gui.viz.ortho_panel import _ortho_prefs

        self._viewer = viewer
        #: Layers hidden on the canvas by hand, by ``id``.
        self._manual: set[int] = set()
        stored = _ortho_prefs()
        self._resample = bool(stored.get(PREF_RESAMPLE, True))
        self._canvas_shown = bool(stored.get(PREF_CANVAS, True))
        #: Depth of :func:`world_geometry` blocks: while > 0, every layer is in world space.
        self._world_depth = 0
        #: Set while this filter writes transforms, so its own events are not taken
        #: for outside changes.
        self._writing = False
        self._subs: list[tuple[Any, Any]] = []
        self._view_action: Any | None = None
        # Zero-interval: after Napari's own handlers for the same event (a new
        # layer's visual, a rebuilt node) have run.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.apply)

        layers = getattr(viewer, "layers", None)
        if layers is not None:
            for name in ("inserted", "removed"):
                emitter = getattr(layers.events, name, None)
                if emitter is not None:
                    emitter.connect(self._on_layers_changed)
        self._watch_layers()
        self._install_view_action()
        if not self._resample:
            self.schedule()

    # ── per-layer visibility ─────────────────────────────────────────────────

    def is_manually_hidden(self, layer: Any) -> bool:
        return id(layer) in self._manual

    def set_manually_hidden(self, layer: Any, hidden: bool) -> None:
        """Keep *layer* off the canvas (or put it back), leaving its visibility alone."""
        if hidden:
            self._manual.add(id(layer))
        else:
            self._manual.discard(id(layer))
        self.apply()

    # ── resampling ───────────────────────────────────────────────────────────

    def resample(self) -> bool:
        return self._resample

    def set_resample(self, on: bool) -> None:
        """Draw layers aligned in world space (on), or each on its own voxel grid (off)."""
        from nvitk.gui.viz.ortho_panel import _save_ortho_pref

        on = bool(on)
        if on == self._resample:
            return
        self._resample = on
        _save_ortho_pref(PREF_RESAMPLE, on)
        self.apply()
        try:
            self._viewer.reset_view()
        except Exception:  # noqa: BLE001 — a view that cannot be reset keeps its camera
            pass

    def _native_wanted(self) -> bool:
        return not self._resample and self._world_depth == 0

    def _go_native(self, layer: Any) -> bool:
        """Draw *layer* on its own grid; True when its transform was changed."""
        current = _composite(layer)
        shown = getattr(layer, SHOWN_ATTR, None)
        stash = true_transform(layer)
        if stash is not None and shown is not None and np.allclose(current, shown, atol=_TOL):
            return False  # already native, nothing written since
        if stash is None or shown is None:
            # A layer that copied another's *displayed* placement (Napari's new-labels
            # button, a plain ``scale=layer.scale``) inherits that layer's real one.
            stash = self._inherited_truth(layer, current) or _parts(layer)
        else:
            # Written while native (an orientation change, a tool outside
            # :func:`world_geometry`): the components that changed are real values
            # now; the ones still holding what the override set keep the real ones.
            stash = _merge_written(_parts(layer), getattr(layer, SHOWN_PARTS_ATTR, None), stash)
        setattr(layer, TRUE_ATTR, stash)
        true_composite = self._composite_of(layer, stash)
        native = _native_parts(layer, true_composite)
        self._writing = True
        try:
            _set_parts(layer, native)
        finally:
            self._writing = False
        setattr(layer, SHOWN_ATTR, _composite(layer))
        setattr(layer, SHOWN_PARTS_ATTR, native)
        return True

    def _composite_of(self, layer: Any, parts: dict[str, np.ndarray]) -> np.ndarray:
        """The data → world matrix *parts* give *layer* (computed by setting them briefly)."""
        if np.allclose(_parts_matrix(layer), _parts_matrix_from(parts), atol=_TOL):
            return _composite(layer)
        saved = _parts(layer)
        self._writing = True
        try:
            _set_parts(layer, parts)
            out = _composite(layer)
            _set_parts(layer, saved)
        finally:
            self._writing = False
        return out

    def _inherited_truth(self, layer: Any, current: np.ndarray) -> dict[str, np.ndarray] | None:
        shape = _spatial_shape(layer)
        for other in list(getattr(self._viewer, "layers", []) or []):
            if other is layer:
                continue
            shown = getattr(other, SHOWN_ATTR, None)
            stash = true_transform(other)
            if shown is None or stash is None or shown.shape != current.shape:
                continue
            if not np.allclose(shown, current, atol=_TOL):
                continue
            if shape is not None and _spatial_shape(other) not in (None, shape):
                continue
            return {k: v.copy() for k, v in stash.items()}
        return None

    def _go_world(self, layer: Any) -> bool:
        """Put *layer*'s real transform back; True when it was changed."""
        stash = true_transform(layer)
        if stash is None:
            return False
        shown = getattr(layer, SHOWN_ATTR, None)
        if shown is not None and not np.allclose(_composite(layer), shown, atol=_TOL):
            # Written while native and not yet seen: merge the written components in.
            stash = _merge_written(_parts(layer), getattr(layer, SHOWN_PARTS_ATTR, None), stash)
        self._writing = True
        try:
            _set_parts(layer, stash)
        finally:
            self._writing = False
        for attr in (TRUE_ATTR, SHOWN_ATTR, SHOWN_PARTS_ATTR):
            try:
                delattr(layer, attr)
            except AttributeError:
                pass
        return True

    # ── applying ─────────────────────────────────────────────────────────────

    def schedule(self, _event: Any = None) -> None:
        """Re-decide on the next event-loop turn (folds bursts of events)."""
        if not self._writing:
            self._timer.start()

    def apply(self) -> None:
        """Bring every layer's canvas visibility and transform in line with the switches."""
        layers = list(getattr(self._viewer, "layers", []) or [])
        visuals = self._visuals()
        changed = False
        native = self._native_wanted()
        for layer in layers:
            try:
                if native and overridable(layer):
                    changed |= self._go_native(layer)
                elif true_transform(layer) is not None:
                    changed |= self._go_world(layer)
            except Exception:  # noqa: BLE001 — one odd layer must not stop the others
                continue
            hidden = self.is_manually_hidden(layer)
            if canvas_hidden(layer) == hidden:
                continue
            try:
                setattr(layer, CANVAS_HIDDEN_ATTR, hidden)
            except Exception:  # noqa: BLE001
                continue
            changed = True
            visual = visuals.get(layer) if visuals is not None else None
            if visual is not None:
                try:
                    visual._on_visible_change()
                except Exception:  # noqa: BLE001
                    pass
        if changed:
            self.changed.emit()

    @contextlib.contextmanager
    def world(self) -> Iterator[None]:
        """Every layer in its real (world) placement for the duration (nestable)."""
        self._world_depth += 1
        try:
            if self._world_depth == 1 and not self._resample:
                self.apply()
            yield
        finally:
            self._world_depth -= 1
            if self._world_depth == 0 and not self._resample:
                self.apply()

    def _visuals(self) -> dict[Any, Any] | None:
        try:
            return self._viewer.window._qt_viewer.canvas.layer_to_visual
        except Exception:  # noqa: BLE001 — headless viewer without a canvas
            return None

    def _on_layers_changed(self, event: Any = None) -> None:
        if getattr(event, "type", "") == "removed":
            layer = getattr(event, "value", None)
            if layer is not None:
                # ``id`` values are reused once an object is gone.
                self._manual.discard(id(layer))
        self._watch_layers()
        self.schedule()

    def _watch_layers(self) -> None:
        """Notice a transform written from outside while layers are drawn on their own grid."""
        for emitter, callback in self._subs:
            try:
                emitter.disconnect(callback)
            except Exception:  # noqa: BLE001
                pass
        self._subs = []
        for layer in list(getattr(self._viewer, "layers", []) or []):
            for name in ("affine", "scale", "translate", "rotate", "shear"):
                emitter = getattr(getattr(layer, "events", None), name, None)
                if emitter is None:
                    continue
                emitter.connect(self.schedule)
                self._subs.append((emitter, self.schedule))

    # ── the whole canvas ─────────────────────────────────────────────────────

    def canvas_shown(self) -> bool:
        return self._canvas_shown

    def _central(self) -> Any | None:
        try:
            return self._viewer.window._qt_window.centralWidget()
        except Exception:  # noqa: BLE001
            return None

    def set_canvas_shown(self, shown: bool, *, remember: bool = True) -> None:
        """Show the main canvas, or hide it and let the docks take its place."""
        from nvitk.gui.viz.ortho_panel import _save_ortho_pref

        self._canvas_shown = bool(shown)
        central = self._central()
        if central is not None:
            central.setVisible(self._canvas_shown)
        if remember:
            _save_ortho_pref(PREF_CANVAS, self._canvas_shown)
        if self._view_action is not None and self._view_action.isChecked() != self._canvas_shown:
            self._view_action.blockSignals(True)
            self._view_action.setChecked(self._canvas_shown)
            self._view_action.blockSignals(False)
        self.changed.emit()

    def _install_view_action(self) -> None:
        """*View ▸ Show main canvas*: reachable even with the orthogonal views closed."""
        try:
            from qtpy.QtWidgets import QAction
        except ImportError:  # Qt 6 moved it
            from qtpy.QtGui import QAction
        try:
            menu = self._viewer.window.view_menu
        except Exception:  # noqa: BLE001
            return
        action = QAction("Show main canvas", menu)
        action.setCheckable(True)
        action.setChecked(self._canvas_shown)
        action.setToolTip(
            "Hide the main 3D/2D canvas to give the docked panels (the orthogonal "
            "views) its space; the layers stay loaded."
        )
        action.toggled.connect(lambda checked: self.set_canvas_shown(bool(checked)))
        menu.addSeparator()
        menu.addAction(action)
        self._view_action = action


def _merge_written(
    current: dict[str, np.ndarray], shown: dict[str, np.ndarray] | None, real: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """The real transform after an outside write: each component that no longer holds what
    the override set was written, and is taken as it is; the others keep their real value."""
    if shown is None:
        return current
    out: dict[str, np.ndarray] = {}
    for key, value in current.items():
        written = key not in shown or np.shape(value) != np.shape(shown[key]) or not np.allclose(
            value, shown[key], atol=_TOL
        )
        out[key] = value if written or key not in real else real[key].copy()
    return out


def _parts_matrix(layer: Any) -> np.ndarray:
    parts = _parts(layer)
    return _parts_matrix_from(parts)


def _parts_matrix_from(parts: dict[str, np.ndarray]) -> np.ndarray:
    """The components flattened, for an equality test."""
    return np.concatenate([np.ravel(parts[k]) for k in ("scale", "translate", "rotate", "shear", "affine")])


def _spatial_shape(layer: Any) -> tuple[int, ...] | None:
    data = getattr(layer, "data", None)
    shape = getattr(data, "shape", None)
    if shape is None or type(layer).__name__ not in ("Image", "Labels"):
        return None
    return tuple(int(v) for v in shape)


def main_canvas_filter(viewer: Any) -> MainCanvasFilter | None:
    """The filter the orthogonal views installed on *viewer*, if any."""
    return getattr(viewer, "_nvitk_main_canvas", None)


@contextlib.contextmanager
def world_geometry(viewer: Any) -> Iterator[None]:
    """Run a block with every layer in its real (world) placement.

    For code that reads or writes layer transforms directly — tools, orientation
    changes, scene saving. A no-op when resampling is on or there is no filter.
    """
    canvas = main_canvas_filter(viewer) if viewer is not None else None
    if canvas is None:
        yield
        return
    with canvas.world():
        yield


__all__ = [
    "MainCanvasFilter",
    "PREF_CANVAS",
    "PREF_RESAMPLE",
    "TRUE_ATTR",
    "main_canvas_filter",
    "overridable",
    "true_transform",
    "world_geometry",
]
