"""Export Napari viewer screenshots (3D canvas, 4D GIF)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from nvitk.gui.core.orientation import _axes_string_from_layer


def _dims_slot_for_array_axis(viewer: Any, array_axis: int) -> int:
    """Map array-axis index to Napari dims slot (for GIF export only)."""
    order = tuple(int(x) for x in viewer.dims.order)
    ax = int(array_axis)
    if ax in order:
        return order.index(ax)
    return ax


def _array_axis_for_dims_slot(viewer: Any, dims_slot: int) -> int:
    """Inverse of :func:`_dims_slot_for_array_axis`: map a Napari dims slot back to an array axis."""
    order = tuple(int(x) for x in viewer.dims.order)
    slot = int(dims_slot)
    if 0 <= slot < len(order):
        return int(order[slot])
    return slot


@dataclass(frozen=True)
class GifAnimationSpec:
    """Time axis and frame count for viewer GIF export."""

    time_axis: int
    dims_slot: int
    n_frames: int
    phase_layer: Any | None = None


@dataclass(frozen=True)
class _ViewerViewState:
    """Snapshot of a Napari viewer's dims/camera state, for save-and-restore around screenshot/GIF export."""

    ndisplay: int
    order: tuple[int, ...]
    point: tuple[float, ...]
    current_step: tuple[int, ...]
    axis_labels: tuple | None
    camera_center: tuple | None
    camera_zoom: float | None
    camera_angles: tuple | None


def _process_events() -> None:
    """Flush pending Qt events so the viewer redraws before a screenshot is captured."""
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass


def _capture_viewer_state(viewer: Any) -> _ViewerViewState:
    """Snapshot *viewer*'s current dims/camera state into a :class:`_ViewerViewState`."""
    cam = viewer.camera
    angles = tuple(float(a) for a in cam.angles) if hasattr(cam, "angles") else None
    center = tuple(float(c) for c in cam.center) if hasattr(cam, "center") else None
    zoom = float(cam.zoom) if hasattr(cam, "zoom") else None
    labels = viewer.dims.axis_labels
    ax_lab = tuple(labels) if labels is not None else None
    steps = tuple(int(x) for x in viewer.dims.current_step)
    return _ViewerViewState(
        ndisplay=int(viewer.dims.ndisplay),
        order=tuple(int(x) for x in viewer.dims.order),
        point=tuple(float(x) for x in viewer.dims.point),
        current_step=steps,
        axis_labels=ax_lab,
        camera_center=center,
        camera_zoom=zoom,
        camera_angles=angles,
    )


def _restore_viewer_state(viewer: Any, state: _ViewerViewState) -> None:
    """Restore *viewer*'s dims/camera to a previously captured *state*."""
    viewer.dims.ndisplay = state.ndisplay
    viewer.dims.order = state.order
    viewer.dims.point = state.point
    if state.current_step:
        viewer.dims.current_step = state.current_step
    if state.axis_labels is not None:
        viewer.dims.axis_labels = state.axis_labels
    cam = viewer.camera
    if state.camera_center is not None and hasattr(cam, "center"):
        cam.center = state.camera_center
    if state.camera_zoom is not None and hasattr(cam, "zoom"):
        cam.zoom = state.camera_zoom
    if state.camera_angles is not None and hasattr(cam, "angles"):
        cam.angles = state.camera_angles
    _process_events()


def time_axis_index(viewer: Any, layer: Any) -> int | None:
    """Return the array axis used for the time/cardiac slider, or None."""
    data = getattr(layer, "data", None)
    if data is None:
        return None
    ndim = int(data.ndim)
    if ndim < 4:
        return None

    axes = _axes_string_from_layer(layer)
    if axes:
        for i, ch in enumerate(axes):
            if str(ch).upper() in ("T", "C"):
                return i

    ndisplay = int(viewer.dims.ndisplay)
    if ndim <= ndisplay:
        return None
    order = [int(x) for x in viewer.dims.order]
    n_hidden = ndim - ndisplay
    if len(order) >= n_hidden:
        return order[0]
    return ndim - 1


def _gif_spec(
    viewer: Any,
    *,
    array_time_axis: int,
    n_frames: int,
    phase_layer: Any | None,
) -> GifAnimationSpec:
    """Build a :class:`GifAnimationSpec` from an array-space time axis, resolving its Napari dims slot."""
    slot = _dims_slot_for_array_axis(viewer, array_time_axis)
    return GifAnimationSpec(
        time_axis=int(array_time_axis),
        dims_slot=int(slot),
        n_frames=int(n_frames),
        phase_layer=phase_layer,
    )


def _frame_count_for_array_time_axis(
    viewer: Any,
    array_time_axis: int,
    *,
    phase_layer: Any | None = None,
) -> int:
    """Number of frames along *array_time_axis*, preferring *phase_layer*'s own time axis length when
    given, else the viewer dims range for that axis."""
    if phase_layer is not None:
        from nvitk.gui.viz.layers import _time_axis_index_from_layer

        data = getattr(phase_layer, "data", None)
        if data is not None and int(data.ndim) >= 4:
            t_ax = _time_axis_index_from_layer(phase_layer)
            return int(data.shape[t_ax])
    slot = _dims_slot_for_array_axis(viewer, array_time_axis)
    if slot is not None and slot < len(viewer.dims.range):
        rng = viewer.dims.range[slot]
        step = max(abs(float(getattr(rng, "step", 1.0) or 1.0)), 1e-9)
        stop = float(getattr(rng, "stop", 0.0))
        return max(1, int(round(stop / step)) + 1)
    return 1


def resolve_gif_animation(
    viewer: Any,
    layer: Any | None = None,
    *,
    time_axis: int | None = None,
) -> GifAnimationSpec:
    """
    Find how to step the Napari dims slider for GIF export.

    Supports 4D image layers and synced 4D flow-vector overlays (3D vectors layer).
    """
    if time_axis is not None and time_axis >= 0:
        n = _frame_count_for_array_time_axis(viewer, time_axis)
        if n >= 1:
            return _gif_spec(viewer, array_time_axis=time_axis, n_frames=n, phase_layer=None)

    playback = getattr(viewer, "_nvitk_flow_vector_state", None)
    if playback is not None:
        from nvitk.gui.viz.layers import _time_axis_index_from_layer

        phase = playback.phase_layer
        t_ax = _time_axis_index_from_layer(phase)
        n_cache = int(playback.cache.n_time)
        if n_cache < 1:
            raise ValueError("Flow vector cache has no time frames.")
        if getattr(phase, "data", None) is not None:
            n_layer = int(phase.data.shape[t_ax])
            if n_layer > 0 and n_layer != n_cache:
                n_cache = min(n_cache, n_layer)
        if n_cache >= 1:
            return _gif_spec(viewer, array_time_axis=t_ax, n_frames=n_cache, phase_layer=phase)

    if layer is not None:
        t_ax = time_axis_index(viewer, layer)
        if t_ax is not None:
            n = int(layer.data.shape[t_ax])
            if n >= 1:
                return _gif_spec(viewer, array_time_axis=t_ax, n_frames=n, phase_layer=layer)

    for lyr in list(viewer.layers):
        data = getattr(lyr, "data", None)
        if data is None or int(data.ndim) < 4:
            continue
        t_ax = time_axis_index(viewer, lyr)
        if t_ax is not None:
            n = int(data.shape[t_ax])
            if n >= 1:
                return _gif_spec(viewer, array_time_axis=t_ax, n_frames=n, phase_layer=lyr)

    labels = getattr(viewer.dims, "axis_labels", None) or ()
    for i, lab in enumerate(labels):
        if str(lab).upper() in ("T", "C"):
            array_ax = _array_axis_for_dims_slot(viewer, i)
            n = _frame_count_for_array_time_axis(viewer, array_ax)
            if n > 1:
                return _gif_spec(
                    viewer, array_time_axis=array_ax, n_frames=n, phase_layer=None
                )

    raise ValueError(
        "GIF export needs a time-varying view: use a 4D image layer, run "
        "'4D flow vectors', or set the time axis explicitly."
    )


@contextmanager
def _paused_flow_dims_sync(viewer: Any) -> Iterator[None]:
    """Disconnect live flow dims hook while GIF frames are written (avoids re-entrancy)."""
    playback = getattr(viewer, "_nvitk_flow_vector_state", None)
    if playback is None:
        yield
        return
    try:
        viewer.dims.events.current_step.disconnect(playback.dims_callback)
    except Exception:
        pass
    try:
        yield
    finally:
        try:
            viewer.dims.events.current_step.connect(playback.dims_callback)
        except Exception:
            pass
        try:
            playback.dims_callback(None)
        except Exception:
            pass


def _set_dim_slot_index(viewer: Any, dims_slot: int, index: int) -> None:
    """Set time on the Napari dims slot that maps to the cardiac axis (4D image GIFs)."""
    ndim = int(viewer.dims.ndim)
    slot = int(dims_slot)
    steps = [int(x) for x in viewer.dims.current_step]
    if len(steps) < ndim:
        steps.extend([0] * (ndim - len(steps)))
    if slot < len(steps):
        steps[slot] = int(index)
    viewer.dims.current_step = tuple(steps[:ndim])

    pt = list(viewer.dims.point)
    if slot < len(pt) and slot < len(viewer.dims.range):
        rng = viewer.dims.range[slot]
        step = float(getattr(rng, "step", 1.0) or 1.0)
        pt[slot] = float(index) * step
        viewer.dims.point = tuple(pt)
    _process_events()


def _set_gif_frame(viewer: Any, spec: GifAnimationSpec, frame: int) -> None:
    """
    Advance to cardiac phase *frame* for one GIF screenshot.

    Flow overlays: step Napari's cardiac dims slot (volume/MIP) and set vector glyphs
    from the precomputed cache by index (do not use the live dims callback here).
    """
    playback = getattr(viewer, "_nvitk_flow_vector_state", None)
    if playback is not None:
        from nvitk.gui.viz.layers import _update_flow_vector_layer

        _set_dim_slot_index(viewer, spec.dims_slot, frame)
        _update_flow_vector_layer(
            playback.layer,
            playback.cache,
            int(frame),
            viewer=viewer,
            phase_layer=playback.phase_layer,
        )
        _process_events()
        return

    _set_dim_slot_index(viewer, spec.dims_slot, frame)


def _ensure_3d_view(viewer: Any) -> None:
    """Switch *viewer* to 3D display mode if it isn't already, then flush pending Qt events."""
    if int(viewer.dims.ndisplay) != 3:
        viewer.dims.ndisplay = 3
    _process_events()


def export_view_png(
    viewer: Any,
    path: str | Path,
    *,
    canvas_only: bool = True,
    flash: bool = False,
) -> np.ndarray:
    """
    Capture the current Napari 3D canvas to a PNG.

    Temporarily enables ``ndisplay=3`` if needed, then restores the prior view.
    """
    path = Path(path)
    if not viewer.layers:
        raise ValueError("No layers in the viewer.")
    state = _capture_viewer_state(viewer)
    try:
        _ensure_3d_view(viewer)
        _process_events()
        out = viewer.screenshot(
            path=str(path),
            canvas_only=bool(canvas_only),
            flash=bool(flash),
        )
    finally:
        _restore_viewer_state(viewer, state)
    if out is None:
        raise RuntimeError("viewer.screenshot returned no image.")
    return np.asarray(out)


def export_view_gif(
    viewer: Any,
    path: str | Path,
    *,
    fps: float = 8.0,
    time_axis: int | None = None,
    canvas_only: bool = True,
    layer: Any | None = None,
) -> int:
    """
    Export one 3D screenshot per time point as an animated GIF.

    Returns the number of frames written.
    """
    from PIL import Image

    path = Path(path)
    if not viewer.layers:
        raise ValueError("No layers in the viewer.")
    layer = layer or viewer.layers.selection.active or viewer.layers[-1]
    spec = resolve_gif_animation(
        viewer,
        layer,
        time_axis=time_axis if time_axis is not None and time_axis >= 0 else None,
    )
    n_frames = spec.n_frames

    state = _capture_viewer_state(viewer)
    frames: list[Image.Image] = []
    try:
        _ensure_3d_view(viewer)
        with _paused_flow_dims_sync(viewer):
            for t in range(n_frames):
                _set_gif_frame(viewer, spec, t)
                shot = viewer.screenshot(canvas_only=bool(canvas_only))
                frames.append(Image.fromarray(np.asarray(shot)))
    finally:
        _restore_viewer_state(viewer, state)

    duration_ms = max(1, int(1000.0 / max(float(fps), 0.1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return len(frames)
