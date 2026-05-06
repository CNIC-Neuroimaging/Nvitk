"""4DFlow visualization helpers (PyVista + ipywidgets).

This is an initial interactive viewer for 4DFlow phase volumes:
- render vessel masks (multi-label supported) as surfaces
- show velocity glyphs inside a selected label
- optionally show simple streamlines (seeded in-label)
- animate over timepoints

The velocity sign conventions follow `nvitk.io.conversors.phase2volume`:
    vx = -RL * 10
    vy = -AP * 10
    vz =  FH * 10
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import ValidationError
from nvitk.types import Image


def _require_pyvista() -> Any:
    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "flowshow requires 'pyvista'. Install it with: pip install pyvista"
        ) from exc
    return pv


def _require_widgets() -> Any:
    try:
        import ipywidgets as widgets  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "flowshow interactive controls require 'ipywidgets'. "
            "Install it with: pip install ipywidgets"
        ) from exc
    return widgets


def _as_4d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 3:
        return arr[..., None]
    if arr.ndim == 4:
        return arr
    raise ValidationError(f"{name} must be 3D or 4D; got shape {arr.shape}.")


def _velocity_from_phases(ap: np.ndarray, rl: np.ndarray, fh: np.ndarray) -> np.ndarray:
    # returns (X,Y,Z,T,3)
    ap4 = _as_4d(ap, "ap_phase")
    rl4 = _as_4d(rl, "rl_phase")
    fh4 = _as_4d(fh, "fh_phase")
    if not (ap4.shape == rl4.shape == fh4.shape):
        raise ValidationError("AP, RL, and FH phase volumes must have identical shapes.")
    vx = (-rl4 * 10.0).astype(np.float32, copy=False)
    vy = (-ap4 * 10.0).astype(np.float32, copy=False)
    vz = (fh4 * 10.0).astype(np.float32, copy=False)
    return np.stack([vx, vy, vz], axis=-1)


def _add_flow_scene(
    pv: Any,
    plotter: Any,
    *,
    lbl: int,
    tt: int,
    show_mask: bool,
    show_glyphs: bool,
    show_stream: bool,
    vel: np.ndarray,
    mask: np.ndarray,
    stride: int,
    stream_seed: int | None = 42,
) -> None:
    """Populate *plotter* with mask / glyphs / streamlines for one label and timepoint."""
    x, y, z, t, _ = vel.shape
    roi = mask == int(lbl)
    st = max(int(stride), 1)

    if show_mask:
        grid = pv.ImageData(dimensions=roi.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
        grid.point_data["roi"] = roi.astype(np.uint8).flatten(order="F")
        surf = grid.contour([0.5], scalars="roi")
        plotter.add_mesh(surf, color="white", opacity=0.25)

    if show_glyphs:
        coords = np.argwhere(roi)
        if coords.size:
            coords = coords[::st]
            v = vel[coords[:, 0], coords[:, 1], coords[:, 2], tt, :].astype(np.float32, copy=False)
            pts = pv.PolyData(coords.astype(np.float32))
            pts["v"] = v
            glyphs = pts.glyph(orient="v", scale=False, factor=1.0)
            plotter.add_mesh(glyphs, color="#00A6FB", opacity=0.9)

    if show_stream:
        vec = vel[..., tt, :]
        grid = pv.ImageData(dimensions=(x, y, z), spacing=(1, 1, 1), origin=(0, 0, 0))
        grid.point_data["v"] = vec.reshape(-1, 3, order="F")
        coords = np.argwhere(roi)
        if coords.shape[0] > 0:
            rng = np.random.default_rng(stream_seed)
            nseed = min(200, coords.shape[0])
            pick = rng.choice(coords.shape[0], size=nseed, replace=False)
            seeds = coords[pick]
            seed_cloud = pv.PolyData(seeds.astype(np.float32))
            try:
                stream = grid.streamlines_from_source(seed_cloud, vectors="v", max_time=50.0)
                plotter.add_mesh(stream.tube(radius=0.2), color="#F7B801", opacity=0.7)
            except Exception:
                pass

    plotter.add_text(f"Label={lbl}  T={tt}", position="upper_left", font_size=12)
    plotter.view_isometric()


def _flowshow_notebook(
    *,
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    mask: np.ndarray,
    centerline_mask: Image | np.ndarray | None,
    stride: int,
    timepoint: int,
    show: bool,
) -> Any:
    _ = centerline_mask  # reserved for future centerline-based seeding / overlays
    pv = _require_pyvista()
    widgets = _require_widgets()

    vel = _velocity_from_phases(ap, rl, fh)
    x, y, z, t, _ = vel.shape

    labels = sorted(int(v) for v in np.unique(mask) if int(v) != 0)
    if not labels:
        raise ValidationError("vessel_mask has no nonzero labels.")

    w_label = widgets.Dropdown(options=labels, value=labels[0], description="Label")
    w_t = widgets.IntSlider(value=int(np.clip(timepoint, 0, t - 1)), min=0, max=t - 1, step=1, description="T")
    w_show_mask = widgets.Checkbox(value=True, description="Mask surface")
    w_show_glyphs = widgets.Checkbox(value=True, description="Velocity glyphs")
    w_show_stream = widgets.Checkbox(value=False, description="Streamlines")
    w_play = widgets.Play(interval=150, value=w_t.value, min=0, max=t - 1, step=1, description="Play")
    widgets.jslink((w_play, "value"), (w_t, "value"))

    plotter = pv.Plotter(notebook=True)
    plotter.enable_depth_peeling()

    def _clear_dynamic():
        plotter.clear()

    def _render():
        _clear_dynamic()
        _add_flow_scene(
            pv,
            plotter,
            lbl=int(w_label.value),
            tt=int(w_t.value),
            show_mask=bool(w_show_mask.value),
            show_glyphs=bool(w_show_glyphs.value),
            show_stream=bool(w_show_stream.value),
            vel=vel,
            mask=mask,
            stride=stride,
        )
        if show:
            plotter.show(auto_close=False)

    for w in (w_label, w_t, w_show_mask, w_show_glyphs, w_show_stream):
        w.observe(lambda _ch: _render(), names="value")
    _render()

    ui = widgets.VBox(
        [
            widgets.HBox([w_label, w_t, w_play]),
            widgets.HBox([w_show_mask, w_show_glyphs, w_show_stream]),
        ]
    )
    return widgets.VBox([ui])


def _flowshow_desktop(
    *,
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    mask: np.ndarray,
    centerline_mask: Image | np.ndarray | None,
    stride: int,
    timepoint: int,
    show: bool,
) -> Any:
    _ = centerline_mask
    pv = _require_pyvista()

    vel = _velocity_from_phases(ap, rl, fh)
    x, y, z, t, _ = vel.shape

    labels = sorted(int(v) for v in np.unique(mask) if int(v) != 0)
    if not labels:
        raise ValidationError("vessel_mask has no nonzero labels.")

    state: dict[str, Any] = {
        "label_idx": 0,
        "tt": int(np.clip(timepoint, 0, t - 1)),
        "mask": True,
        "glyphs": True,
        "stream": False,
    }

    plotter = pv.Plotter(notebook=False)
    plotter.enable_depth_peeling()

    def _redraw():
        # Remove only scene actors; keep slider/checkbox widgets.
        plotter.clear_actors()
        li = int(state["label_idx"])
        lbl = labels[li]
        _add_flow_scene(
            pv,
            plotter,
            lbl=lbl,
            tt=int(state["tt"]),
            show_mask=bool(state["mask"]),
            show_glyphs=bool(state["glyphs"]),
            show_stream=bool(state["stream"]),
            vel=vel,
            mask=mask,
            stride=stride,
        )
        plotter.render()

    def _on_time(value: float):
        state["tt"] = int(round(value))
        _redraw()

    def _on_label(value: float):
        state["label_idx"] = int(round(value))
        state["label_idx"] = int(np.clip(state["label_idx"], 0, len(labels) - 1))
        _redraw()

    plotter.add_slider_widget(
        _on_time,
        rng=[0, t - 1],
        value=state["tt"],
        title="Time",
        pointa=(0.02, 0.1),
        pointb=(0.35, 0.1),
    )
    if len(labels) > 1:
        plotter.add_slider_widget(
            _on_label,
            rng=[0, len(labels) - 1],
            value=state["label_idx"],
            title="Label idx",
            pointa=(0.02, 0.18),
            pointb=(0.35, 0.18),
        )

    def _cb_mask(val: bool):
        state["mask"] = val
        _redraw()

    def _cb_glyphs(val: bool):
        state["glyphs"] = val
        _redraw()

    def _cb_stream(val: bool):
        state["stream"] = val
        _redraw()

    plotter.add_checkbox_button_widget(_cb_mask, value=True, position=(18, 120), size=20, border_size=1)
    plotter.add_checkbox_button_widget(_cb_glyphs, value=True, position=(18, 90), size=20, border_size=1)
    plotter.add_checkbox_button_widget(_cb_stream, value=False, position=(18, 60), size=20, border_size=1)
    plotter.add_text("Mask / Glyphs / Stream toggles (checkboxes, bottom-left)", position="lower_left", font_size=9)

    _redraw()
    if show:
        plotter.show()
    return plotter


def flowshow(
    ap_phase: Image | np.ndarray,
    rl_phase: Image | np.ndarray,
    fh_phase: Image | np.ndarray,
    vessel_mask: Image | np.ndarray,
    *,
    centerline_mask: Image | np.ndarray | None = None,
    stride: int = 4,
    timepoint: int = 0,
    notebook: bool = True,
    show: bool = True,
) -> Any:
    """Interactive 4DFlow viewer.

    When ``notebook=True`` (default), returns an ipywidgets container for Jupyter.
    When ``notebook=False``, opens a PyVista desktop window with embedded sliders
    and checkbox widgets (suitable for CLI use).

    Parameters
    ----------
    show
        If False, the notebook backend will not call ``plotter.show`` after each
        redraw; the desktop backend will build the plotter but not open a window.
    """
    ap = to_numpy(ap_phase.data) if isinstance(ap_phase, Image) else np.asarray(ap_phase)
    rl = to_numpy(rl_phase.data) if isinstance(rl_phase, Image) else np.asarray(rl_phase)
    fh = to_numpy(fh_phase.data) if isinstance(fh_phase, Image) else np.asarray(fh_phase)
    mask = to_numpy(vessel_mask.data) if isinstance(vessel_mask, Image) else np.asarray(vessel_mask)
    mask = mask.astype(np.int32, copy=False)

    vel = _velocity_from_phases(ap, rl, fh)
    x, y, z, t, _ = vel.shape

    if mask.ndim != 3 or mask.shape != (x, y, z):
        raise ValidationError(f"vessel_mask must be 3D and match spatial dims {(x,y,z)}; got {mask.shape}.")

    if notebook:
        return _flowshow_notebook(
            ap=ap,
            rl=rl,
            fh=fh,
            mask=mask,
            centerline_mask=centerline_mask,
            stride=stride,
            timepoint=timepoint,
            show=show,
        )
    return _flowshow_desktop(
        ap=ap,
        rl=rl,
        fh=fh,
        mask=mask,
        centerline_mask=centerline_mask,
        stride=stride,
        timepoint=timepoint,
        show=show,
    )


__all__ = ["flowshow"]
