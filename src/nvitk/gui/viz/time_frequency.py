"""GUI handlers for 3D+t (time) and k-space (frequency) tools.

The Napari side of :mod:`nvitk.transform.temporal` and
:mod:`nvitk.transform.fourier`: each handler reads the active layer, runs the
base tool on the current compute backend, and adds its result to the viewer on
the right grid — a 3D result of a 3D+t layer on its spatial grid, a 3D+t result
time-first (see :func:`nvitk.gui.core.orientation.prepare_time_leading_for_napari`).

K-space layers
--------------
Napari cannot draw complex numbers, so the k-space tool adds a real *view* of it
(log-magnitude by default, optionally the phase) and keeps the complex array
beside that layer, in :data:`_KSPACE_STORE`. The inverse FFT reads it back from
there: run "Inverse FFT" with the k-space layer active — optionally with a mask
painted on that layer's grid, to keep or remove parts of k-space — and the image
comes back on the original grid. The store holds layers weakly, so deleting the
k-space layer frees its complex array.
"""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.backend import get_global_backend, using
from nvitk.core.logger import Logger
from nvitk.types import Image

log = Logger()

#: Complex k-space behind each k-space display layer (weak: freed with the layer).
_KSPACE_STORE: "weakref.WeakKeyDictionary[Any, Image]" = weakref.WeakKeyDictionary()

#: Kept on the viewer so repeated curve requests share one plot window.
_CURVE_WINDOW_ATTR = "_nvitk_time_curve_window"


# ──────────────────────────────────────────────────────────────────────────────
# Layer helpers
# ──────────────────────────────────────────────────────────────────────────────


def _notify(message: str, *, error: bool = False) -> None:
    """Status line + log, through the tools' notifier when it is importable."""
    try:
        from nvitk.gui.tools.runner import notify

        notify(message, error=error)
    except Exception:  # noqa: BLE001
        (log.error if error else log.info)(message)


def add_image_layer(viewer: Any, img: Image, *, name: str | None = None, source: str = "") -> Any:
    """Add an :class:`~nvitk.types.Image` the way opening a file would.

    A 3D+t image is shown time-first with its full affine; anything else gets
    nvitk's usual display transforms. *source* only labels the layer's origin.
    """
    from nvitk.gui.io.napari_io import _add_image_to_viewer

    if name:
        img.name = name
    return _add_image_to_viewer(viewer, img, Path(source or (img.name or "result")))


def _add_like(viewer: Any, layer: Any, data: Any, *, name: str, colormap: str | None = None,
              contrast: tuple[float, float] | None = None) -> Any:
    """Add *data* on *layer*'s grid (trailing dims), carrying its nvitk metadata."""
    from nvitk.gui.core.spatial import layer_spatial_kwargs
    from nvitk.gui.labels.visibility import copy_layer_metadata_for_output

    arr = to_numpy(data)
    same = arr.ndim == int(layer.data.ndim)
    kwargs: dict[str, Any] = dict(layer_spatial_kwargs(layer) if same else layer_spatial_kwargs(layer, ndim=arr.ndim))
    meta = copy_layer_metadata_for_output(getattr(layer, "metadata", None)) or {}
    if not same and isinstance(meta.get("nvitk_metadata"), dict):
        nested = {k: v for k, v in meta["nvitk_metadata"].items()
                  if k not in ("time_leading", "source_axes", "axes", "frame_times_s", "t_res",
                               "temporal_resolution", "cardiac_phases_percent")}
        meta = dict(meta, nvitk_metadata=nested)
        meta.pop("axes", None)
    if meta:
        kwargs["metadata"] = meta
    if same and getattr(layer, "axis_labels", None) is not None and arr.ndim == 4:
        kwargs["axis_labels"] = tuple(layer.axis_labels)
    if colormap:
        kwargs["colormap"] = colormap
    if contrast is not None:
        kwargs["contrast_limits"] = contrast
    return viewer.add_image(arr, name=name, **kwargs)


def _robust_limits(arr: Any, lo_pct: float = 0.5, hi_pct: float = 99.8) -> tuple[float, float] | None:
    """A display window from a subsample (k-space spans decades; full range hides it)."""
    flat = to_numpy(arr).reshape(-1)
    if flat.size > 500_000:
        flat = flat[:: max(flat.size // 500_000, 1)]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    lo, hi = (float(v) for v in np.percentile(flat, (lo_pct, hi_pct)))
    return (lo, hi) if hi > lo else None


def _require_layer(layer: Any) -> Any:
    """*layer* if it is an image/labels layer with data, else a clear error."""
    if layer is None or getattr(layer, "data", None) is None:
        raise ValueError("Select an image layer first.")
    return layer


def _time_index_now(viewer: Any, layer: Any) -> int:
    """The viewer's current time step for a 3D+t *layer*."""
    from nvitk.gui.core.spatial import _time_axis_index

    t_ax = _time_axis_index(layer)
    offset = int(viewer.dims.ndim) - int(layer.data.ndim)
    try:
        return int(viewer.dims.current_step[offset + t_ax])
    except Exception:  # noqa: BLE001
        return 0


def _cursor_voxel(viewer: Any, layer: Any) -> tuple[int, int, int]:
    """Spatial voxel under the cursor, in the layer's spatial axis order."""
    from nvitk.gui.core.spatial import _time_axis_index

    shape = tuple(int(v) for v in layer.data.shape)
    t_ax = _time_axis_index(layer) if len(shape) == 4 else None
    try:
        pos = list(layer.world_to_data(viewer.cursor.position))[-len(shape):]
    except Exception:  # noqa: BLE001 — no cursor yet: the middle of the volume
        pos = [s / 2.0 for s in shape]
    spatial = [p for i, p in enumerate(pos) if i != t_ax]
    sp_shape = [s for i, s in enumerate(shape) if i != t_ax]
    return tuple(int(np.clip(round(float(p)), 0, n - 1)) for p, n in zip(spatial, sp_shape))  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# K-space
# ──────────────────────────────────────────────────────────────────────────────


def run_kspace(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """FFT of the active layer → k-space display layer(s), complex kept for the inverse."""
    from nvitk.gui.core.spatial import layer_to_image
    from nvitk.transform.fourier import kspace, kspace_component

    layer = _require_layer(layer)
    mode = str(params.get("kspace_mode") or "3d")
    component = str(params.get("kspace_component") or "log_magnitude")
    with using(get_global_backend()):
        kim = kspace(layer_to_image(layer), mode=mode, centered=bool(params.get("kspace_centered", True)))
        view = kspace_component(kim, component)
        phase = kspace_component(kim, "phase") if bool(params.get("kspace_add_phase")) and component != "phase" else None
    base = f"{layer.name}_kspace"
    shown = _add_like(viewer, layer, view.data, name=f"{base}_{component}", colormap="gray",
                      contrast=_robust_limits(view.data) if component != "phase" else (-np.pi, np.pi))
    # A fresh nested dict: the copied metadata may share it with the source layer.
    nested = dict(shown.metadata.get("nvitk_metadata") or {})
    nested["kspace"] = dict(kim.metadata["kspace"])
    nested["kspace_source"] = str(layer.name)
    shown.metadata["nvitk_metadata"] = nested
    _KSPACE_STORE[shown] = kim
    if phase is not None:
        ph = _add_like(viewer, layer, phase.data, name=f"{base}_phase", colormap="twilight",
                       contrast=(-np.pi, np.pi))
        _KSPACE_STORE[ph] = kim
    info = kim.metadata["kspace"]
    _notify(
        f"K-space of “{layer.name}” ({info['mode'].upper()} over {info['axis_labels']}, "
        f"Δk = {', '.join(f'{f:.4g}' for f in info['frequency_spacing'])} cycles/mm). "
        "Run Inverse FFT with it active (optionally with a mask) to go back."
    )


def run_inverse_kspace(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Inverse FFT of the active k-space layer, optionally weighted by a mask layer."""
    from nvitk.gui.tools.runner import _layer_param, _resolve_layer
    from nvitk.transform.fourier import apply_kspace_mask, inverse_kspace

    layer = _require_layer(layer)
    kim = _KSPACE_STORE.get(layer)
    if kim is None:
        raise ValueError(
            f"“{layer.name}” has no complex k-space behind it. Run 'K-space (FFT)' first "
            "and keep its layer active (a saved/reloaded k-space image holds only magnitude)."
        )
    mask_name = _layer_param(params, "kspace_mask_layer")
    weights_note = ""
    with using(get_global_backend()):
        if mask_name:
            mask_layer = _resolve_layer(viewer, mask_name)
            mask = to_numpy(mask_layer.data)
            if bool(params.get("kspace_mask_remove")):
                weights = (mask <= 0).astype(np.float32)
                weights_note = f", removing the k-space painted in “{mask_layer.name}”"
            else:
                weights = (mask > 0).astype(np.float32)
                weights_note = f", keeping only the k-space painted in “{mask_layer.name}”"
            # A 3D mask on a 3D+t k-space applies to every time point.
            if weights.ndim == 3 and kim.ndim == 4:
                from nvitk.transform.temporal import time_axis

                weights = np.expand_dims(weights, time_axis(kim))
            kim = apply_kspace_mask(kim, weights)
        back = inverse_kspace(kim)
    source = layer.metadata.get("nvitk_metadata", {}).get("kspace_source") or layer.name
    _add_like(viewer, layer, back.data, name=f"{source}_ifft")
    _notify(f"Inverse FFT of “{layer.name}”{weights_note}.")


def run_kspace_filter(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Low/high/band-pass filter the active layer in k-space."""
    from nvitk.gui.core.spatial import layer_to_image
    from nvitk.transform.fourier import kspace_filter

    layer = _require_layer(layer)
    kind = str(params.get("kspace_filter_kind") or "lowpass")
    cutoff = float(params.get("kspace_cutoff") or 0.5)
    cutoff_high = params.get("kspace_cutoff_high")
    with using(get_global_backend()):
        out = kspace_filter(
            layer_to_image(layer),
            kind=kind,
            cutoff=cutoff,
            cutoff_high=float(cutoff_high) if cutoff_high not in (None, "") else None,
            window=str(params.get("kspace_window") or "hann"),
            mode=str(params.get("kspace_mode") or "3d"),
        )
    _add_like(viewer, layer, out.data, name=f"{layer.name}_{kind}")
    _notify(f"{kind} k-space filter (cutoff {cutoff:g} × Nyquist) applied to “{layer.name}”.")


# ──────────────────────────────────────────────────────────────────────────────
# Time
# ──────────────────────────────────────────────────────────────────────────────


def _require_4d(layer: Any) -> Any:
    layer = _require_layer(layer)
    if int(getattr(layer.data, "ndim", 0)) != 4:
        raise ValueError(f"“{layer.name}” is not a 3D+t (4D) layer.")
    return layer


def run_extract_frame(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """One time point of the active 3D+t layer as a 3D layer."""
    from nvitk.gui.core.spatial import layer_to_image
    from nvitk.transform.temporal import extract_frame

    layer = _require_4d(layer)
    index = int(params.get("time_frame", -1))
    if index < 0:
        index = _time_index_now(viewer, layer)
    frame = extract_frame(layer_to_image(layer), index)
    _add_like(viewer, layer, frame.data, name=f"{layer.name}_t{index}")
    _notify(f"Frame {index} of “{layer.name}” (t = {frame.metadata.get('frame_time_s', index):.4g}).")


def run_temporal_projection(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Collapse the time axis of the active 3D+t layer (MIP, mean, std, TTP, AUC…)."""
    from nvitk.gui.core.spatial import layer_to_image
    from nvitk.transform.temporal import temporal_projection

    layer = _require_4d(layer)
    method = str(params.get("time_method") or "max")
    with using(get_global_backend()):
        out = temporal_projection(layer_to_image(layer), method)
    colormap = "turbo" if method == "ttp" else None
    _add_like(viewer, layer, out.data, name=f"{layer.name}_t{method}", colormap=colormap,
              contrast=_robust_limits(out.data, 0.0, 100.0) if method == "ttp" else None)
    _notify(f"Temporal {method} of “{layer.name}”.")


def run_time_curve(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Plot the time–intensity curve at the cursor, or over a mask label."""
    from nvitk.gui.core.spatial import layer_to_image
    from nvitk.gui.tools.runner import _layer_param, _resolve_layer
    from nvitk.transform.temporal import time_intensity_curve

    layer = _require_4d(layer)
    img = layer_to_image(layer)
    mask_name = _layer_param(params, "time_mask_layer")
    label_id = int(params.get("time_label_id") or 0)
    if mask_name:
        mask_layer = _resolve_layer(viewer, mask_name)
        mask = to_numpy(mask_layer.data)
        if mask.ndim == 4:
            mask = mask.max(axis=0)
        region = (mask == label_id) if label_id > 0 else (mask > 0)
        times, values = time_intensity_curve(img, mask=region)
        what = f"“{mask_layer.name}”" + (f" label {label_id}" if label_id > 0 else "")
    else:
        voxel = _cursor_voxel(viewer, layer)
        times, values = time_intensity_curve(img, voxel=voxel)
        what = f"voxel {voxel}"
    from nvitk.transform.temporal import frame_axis_unit

    md = img.metadata or {}
    unit = frame_axis_unit(img)
    # A monoenergetic stack plots the spectral attenuation curve: HU against keV.
    y_unit = str(md.get("spectral_units") or "") if unit == "keV" else ""
    window = show_time_curve_window(viewer, times, values, label=f"{layer.name} · {what}",
                                    unit=unit, y_unit=y_unit,
                                    append=bool(params.get("time_append", True)))
    peak = int(np.argmax(values))
    kind = "Spectral curve" if unit == "keV" else "Time curve"
    _notify(f"{kind} of {what}: peak {values[peak]:.4g} at {times[peak]:.4g} {unit}.")
    return window


def run_stack_layers(viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Stack same-grid 3D layers into one 3D+t layer (e.g. cardiac phases opened separately)."""
    from nvitk.gui.core.spatial import layer_to_image, nvitk_metadata_from_layer
    from nvitk.gui.viz.ortho_panel import _same_grid
    from nvitk.io.conversors._dicom_phases import cardiac_phase_percent
    from nvitk.transform.temporal import stack_frames

    layer = _require_layer(layer)
    names = [n.strip() for n in str(params.get("stack_layers") or "").split(",") if n.strip()]
    if names:
        by_name = {l.name: l for l in viewer.layers}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"No layer(s) named {', '.join(missing)}.")
        layers = [by_name[n] for n in names]
    else:
        layers = [
            l for l in viewer.layers
            if type(l).__name__ == "Image" and bool(getattr(l, "visible", True)) and _same_grid(l, layer)
        ]
    layers = [l for l in layers if int(getattr(l.data, "ndim", 0)) == 3 and type(l).__name__ == "Image"]
    if len(layers) < 2:
        raise ValueError("Need at least two visible 3D image layers on the active layer's grid.")
    from nvitk.io.conversors._dicom_spectral import classify_spectral

    # Monoenergetic layers stack along energy (a spectral curve), cardiac phases along R-R.
    metas = [nvitk_metadata_from_layer(l) for l in layers]
    infos = [classify_spectral(md=m) for m in metas]
    energies = [
        (float(m.get("spectral_energy_kev")) if m.get("spectral_energy_kev") not in (None, "")
         else (i.get("energy_kev") if i and i["result"] == "monoe" else None))
        for m, i in zip(metas, infos)
    ]
    phases = [cardiac_phase_percent(m) for m in metas]
    keys = energies if all(e is not None for e in energies) else phases
    if all(k is not None for k in keys):
        order = sorted(range(len(layers)), key=lambda i: keys[i])
        layers = [layers[i] for i in order]
        energies = [energies[i] for i in order]
        phases = [phases[i] for i in order]
    t_res = float(params.get("stack_t_res") or 0.0)
    stacked = stack_frames([layer_to_image(l) for l in layers], t_res=t_res or None,
                           name=f"{layers[0].name}_stack{len(layers)}")
    if all(e is not None for e in energies):
        stacked.metadata["spectral_energies_kev"] = [float(e) for e in energies]
        stacked.metadata["spectral_units"] = metas[0].get("spectral_units", "HU")
        stacked.metadata["t_units"] = "keV"
    elif all(p is not None for p in phases):
        stacked.metadata["cardiac_phases_percent"] = [float(p) for p in phases]
    add_image_layer(viewer, stacked)
    _notify(f"Stacked {len(layers)} layers into 3D+t: {', '.join(l.name for l in layers)}.")


# ──────────────────────────────────────────────────────────────────────────────
# Time–intensity curve window
# ──────────────────────────────────────────────────────────────────────────────


def show_time_curve_window(viewer: Any, times: list[float], values: list[float], *,
                           label: str, unit: str = "s", y_unit: str = "",
                           append: bool = True) -> Any:
    """Plot one curve in the viewer's time-curve window (created on first use)."""
    from qtpy.QtWidgets import (
        QDialog,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )

    from nvitk.gui.core.design import COLOR_MUTED, SPACE_TIGHT, apply_theme, style_image_figure

    window = getattr(viewer, _CURVE_WINDOW_ATTR, None)
    if window is None:
        parent = None
        try:
            parent = viewer.window._qt_window
        except Exception:  # noqa: BLE001
            parent = None
        window = QDialog(parent)
        window.setWindowTitle("Time–intensity curves")
        window.setModal(False)
        window.setMinimumSize(560, 360)
        root = QVBoxLayout(window)
        root.setSpacing(SPACE_TIGHT)
        window._caption = QLabel("")
        window._caption.setStyleSheet(f"color: {COLOR_MUTED};")
        root.addWidget(window._caption)
        window._curves = []
        window._fig = window._ax = window._canvas = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            window._fig = Figure(figsize=(5.6, 3.2), dpi=96, layout="constrained")
            window._ax = window._fig.add_subplot(111)
            window._canvas = FigureCanvasQTAgg(window._fig)
            root.addWidget(window._canvas, stretch=1)
        except Exception as exc:  # noqa: BLE001
            root.addWidget(QLabel(f"Matplotlib unavailable: {exc}"))
        buttons = QHBoxLayout()
        clear = QPushButton("Clear")
        save = QPushButton("Save CSV…")
        buttons.addStretch(1)
        buttons.addWidget(clear)
        buttons.addWidget(save)
        root.addLayout(buttons)

        def _redraw() -> None:
            if window._ax is None:
                return
            ax = window._ax
            ax.clear()
            for c_label, c_times, c_values, _unit, _yu in window._curves:
                ax.plot(c_times, c_values, marker="o", ms=3, lw=1.4, label=c_label)
            if window._curves:
                ax.legend(fontsize=7, loc="best")
                last_unit, last_y = window._curves[-1][3], window._curves[-1][4]
                ax.set_xlabel("energy (keV)" if last_unit == "keV" else f"time ({last_unit})", fontsize=8)
                ax.set_ylabel(f"intensity ({last_y})" if last_y else "intensity", fontsize=8)
            else:
                ax.set_ylabel("intensity", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            style_image_figure(window._fig)
            window._canvas.draw_idle()

        def _clear() -> None:
            window._curves = []
            _redraw()

        def _save() -> None:
            if not window._curves:
                return
            path, _ = QFileDialog.getSaveFileName(window, "Save curves", "time_curves.csv", "CSV (*.csv)")
            if not path:
                return
            lines = ["curve,axis_value,axis_unit,value"]
            for c_label, c_times, c_values, c_unit, _yu in window._curves:
                safe = c_label.replace(",", ";")
                lines += [f"{safe},{t:.6g},{c_unit},{v:.6g}" for t, v in zip(c_times, c_values)]
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

        window._redraw = _redraw
        clear.clicked.connect(_clear)
        save.clicked.connect(_save)
        apply_theme(window)
        try:
            setattr(viewer, _CURVE_WINDOW_ATTR, window)
        except Exception:  # noqa: BLE001
            pass
    if not append:
        window._curves = []
    window._curves.append((label, list(times), list(values), unit, y_unit))
    window._caption.setText(f"{len(window._curves)} curve(s) — latest: {label}")
    window._redraw()
    window.show()
    window.raise_()
    return window


__all__ = [
    "add_image_layer",
    "run_extract_frame",
    "run_inverse_kspace",
    "run_kspace",
    "run_kspace_filter",
    "run_stack_layers",
    "run_temporal_projection",
    "run_time_curve",
    "show_time_curve_window",
]
