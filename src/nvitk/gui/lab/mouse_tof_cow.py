"""Mouse TOF Circle-of-Willis lab workflow (GUI).

Stage 1: N4 → blood flood from-scratch → label connected components.
Stage 2: Interactively assign CCs to Left ICA / Right ICA / Basilar trees,
then finalize with a multilabel blood-flood expand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.backend import using
from nvitk.core.logger import Logger
from nvitk.types import Image

log = Logger()

# Final output label ids.
TREE_SPECS: tuple[tuple[str, int], ...] = (
    ("Left ICA", 1),
    ("Right ICA", 2),
    ("Basilar", 3),
)

_SESSION_ATTR = "_nvitk_mouse_tof_cow_session"
_META_SOURCE_KEY = "nvitk_tof_cow_cc_source"

# Stage-1 recipe (hardcoded; do not change global tool defaults).
_N4_SHRINK = 2
_N4_SPLINE = 6
_FRANGI_SIGMAS = (0.75, 1.0, 1.5, 2.0, 2.5)
_HYST_LOW = 4.0
_HYST_HIGH = 0.5
_THICKEN = 0
_THIN_PERCENTILE = 55.0
_MIN_CC = 125
_CONNECTIVITY = 3
# Final expand (multilabel blood flood) after Stage-2 tree assignment.
_EXPAND_FRANGI_SIGMAS = (0.5, 1.0, 1.5, 2.0, 2.5)
_EXPAND_HYST_LOW = 3.0
_EXPAND_HYST_HIGH = 0.5
_EXPAND_THICKEN = 0
_EXPAND_THIN_PERCENTILE = 25.0
_EXPAND_CONNECTIVITY = 3

_StatusFn = Callable[[str], None]
_DoneFn = Callable[[], None]

_status_hook: _StatusFn | None = None
_visibility_hook: _DoneFn | None = None


def set_ui_hooks(
    *,
    status: _StatusFn | None = None,
    visibility: _DoneFn | None = None,
) -> None:
    """Dock registers callbacks to refresh Stage-2 status / button visibility."""
    global _status_hook, _visibility_hook
    _status_hook = status
    _visibility_hook = visibility


def _emit_status(text: str) -> None:
    """Forward *text* to the registered status hook, if any (errors are swallowed)."""
    if _status_hook is not None:
        try:
            _status_hook(text)
        except Exception:
            pass


def _emit_visibility() -> None:
    """Call the registered visibility-refresh hook, if any (errors are swallowed)."""
    if _visibility_hook is not None:
        try:
            _visibility_hook()
        except Exception:
            pass


def run_stage1(image: Image | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """N4 → blood flood from-scratch → label CCs.

    Returns
    -------
    labeled, intensity
        Connected-component labels (int32) and the raw (pre-N4) intensity used
        later for the final multilabel expand.
    """
    from nvitk.morphology.components import label_connected
    from nvitk.restoration.n4_bias import n4_bias_field_correction
    from nvitk.segmentation.blood_flood import blood_flood_from_scratch

    with using("numpy"):
        arr = to_numpy(image.data if isinstance(image, Image) else image)
        if arr.ndim != 3:
            raise ValueError(f"Mouse TOF CoW Stage 1 expects a 3D volume, got {arr.ndim}D")
        raw = np.asarray(arr, dtype=np.float32)

        log.info("Mouse TOF CoW Stage 1: N4 bias correction")
        corrected = n4_bias_field_correction(
            image if isinstance(image, Image) else arr,
            shrink_factor=_N4_SHRINK,
            spline_param=_N4_SPLINE,
            rescale_intensities=True,
        )
        corrected = to_numpy(corrected)

        log.info("Mouse TOF CoW Stage 1: blood flood from-scratch")
        flood = blood_flood_from_scratch(
            corrected,
            frangi_sigmas=_FRANGI_SIGMAS,
            hyst_low_factor=_HYST_LOW,
            hyst_high_factor=_HYST_HIGH,
            thicken_iter=_THICKEN,
            thin_vesselness_percentile=_THIN_PERCENTILE,
            min_cc_voxels=_MIN_CC,
            connectivity=_CONNECTIVITY,
        )
        tree = to_numpy(flood.tree).astype(bool, copy=False)

        log.info("Mouse TOF CoW Stage 1: label connected components")
        labeled, n_cc = label_connected(tree, connectivity=_CONNECTIVITY)
        labeled = to_numpy(labeled).astype(np.int32, copy=False)
        log.info(f"Mouse TOF CoW Stage 1 done: {n_cc} connected components")
        return labeled, raw


def expand_cow_trees(
    intensity: np.ndarray,
    markers: np.ndarray,
) -> np.ndarray:
    """Final multilabel blood-flood expand of assigned CoW tree seeds."""
    from nvitk.segmentation.blood_flood import blood_flood

    with using("numpy"):
        result = blood_flood(
            np.asarray(intensity, dtype=np.float64),
            np.asarray(markers, dtype=np.int32),
            frangi_sigmas=_EXPAND_FRANGI_SIGMAS,
            hyst_low_factor=_EXPAND_HYST_LOW,
            hyst_high_factor=_EXPAND_HYST_HIGH,
            thicken_iter=_EXPAND_THICKEN,
            thin_vesselness_percentile=_EXPAND_THIN_PERCENTILE,
            connectivity=_EXPAND_CONNECTIVITY,
        )
        labels = to_numpy(result.labels).astype(np.int32, copy=False)
        log.info(
            "Mouse TOF CoW final expand: "
            f"tree_voxels={int(np.count_nonzero(result.tree))}, "
            f"labeled={int(np.count_nonzero(labels))}"
        )
        return labels


def _is_left_mouse_button(event: Any) -> bool:
    """True if the mouse *event* was triggered by the left button (or its identity is unknown)."""
    btn = getattr(event, "button", None)
    if btn in (0, 1, None):
        return True
    name = str(btn).lower()
    return name in ("left", "lbutton", "mouse1")


def _connect_pick_callback(target: Any, callback: Any) -> None:
    """Register *callback* as the first mouse-drag handler on *target* (viewer or layer)."""
    try:
        target.mouse_drag_callbacks.insert(0, callback)
    except Exception:
        target.mouse_drag_callbacks.append(callback)


def _disconnect_pick_callback(target: Any, callback: Any) -> None:
    """Remove *callback* from *target*'s mouse-drag handlers, if present."""
    if target is None or callback is None:
        return
    try:
        if callback in target.mouse_drag_callbacks:
            target.mouse_drag_callbacks.remove(callback)
    except Exception:
        pass


def _sample_label(
    layer: Any,
    event: Any,
    *,
    viewer: Any | None = None,
) -> int | None:
    """Return positive label id under click via Napari's Labels pick API."""
    from nvitk.gui.core.label_pick import label_id_under_mouse

    return label_id_under_mouse(layer, event, viewer=viewer)


@dataclass
class MouseTofCowSession:
    """Interactive Stage-2 assignment of CCs to CoW vessel trees."""

    viewer: Any
    labels_layer: Any
    source_name: str
    intensity: np.ndarray  # raw TOF for final expand
    tree_index: int = 0
    highlight_id: int | None = None
    assigned: dict[int, int] = field(default_factory=dict)
    _callback: Any = field(default=None, repr=False)
    _finished: bool = False

    @property
    def tree_name(self) -> str:
        """Display name of the vessel tree currently being assigned (e.g. ``"Left ICA"``)."""
        return TREE_SPECS[self.tree_index][0]

    @property
    def tree_label(self) -> int:
        """Output label id of the vessel tree currently being assigned."""
        return TREE_SPECS[self.tree_index][1]

    def status_text(self) -> str:
        """Human-readable summary of the current tree, pending highlight, and assignment counts."""
        if self._finished:
            return "Mouse TOF CoW: done."
        hl = f"CC {self.highlight_id}" if self.highlight_id else "none"
        n_tree = sum(1 for v in self.assigned.values() if v == self.tree_label)
        return (
            f"Selecting: {self.tree_name} (label {self.tree_label}) — "
            f"highlight={hl} — in tree: {n_tree} CC(s) — "
            f"assigned total: {len(self.assigned)}"
        )

    def install(self) -> None:
        """Attach the mouse-pick callback to the viewer and select the CC labels layer."""
        self._callback = self._on_mouse_pick
        # Viewer-level pick so clicks work even when an Image is above the Labels.
        _connect_pick_callback(self.viewer, self._callback)
        try:
            self.viewer.layers.selection.active = self.labels_layer
        except Exception:
            pass
        self._refresh_display()
        self._push_status()

    def uninstall(self) -> None:
        """Detach the mouse-pick callback and clear the highlighted-label display."""
        _disconnect_pick_callback(self.viewer, self._callback)
        self._callback = None
        try:
            if hasattr(self.labels_layer, "show_selected_label"):
                self.labels_layer.show_selected_label = False
        except Exception:
            pass

    def _on_mouse_pick(self, layer_or_viewer: Any, event: Any) -> None:
        """Left-click handler: highlight the connected component clicked on the labels layer, unless
        it's already assigned to a tree."""
        if self._finished:
            return
        if getattr(event, "type", None) != "mouse_press":
            return
        if not _is_left_mouse_button(event):
            return
        from nvitk.gui.tools.runner import notify

        lid = _sample_label(
            self.labels_layer,
            event,
            viewer=self.viewer,
        )
        if lid is None:
            return
        if lid in self.assigned:
            notify(f"CC {lid} already assigned to tree label {self.assigned[lid]}.", error=True)
            return
        self.highlight_id = int(lid)
        self._apply_highlight()
        self._push_status()
        yield  # napari mouse_drag generator protocol
        while getattr(event, "type", None) == "mouse_move":
            yield

    def clear_highlight(self) -> None:
        """Clear the pending CC highlight without changing assignments."""
        self.highlight_id = None
        try:
            if hasattr(self.labels_layer, "show_selected_label"):
                self.labels_layer.show_selected_label = False
        except Exception:
            pass
        self._push_status()

    def _apply_highlight(self) -> None:
        """Set the labels layer's selected/shown label to the currently highlighted CC."""
        layer = self.labels_layer
        lid = self.highlight_id
        if lid is None:
            return
        try:
            layer.selected_label = int(lid)
            if hasattr(layer, "show_selected_label"):
                layer.show_selected_label = True
        except Exception:
            pass

    def add_highlighted_cc(self) -> None:
        """Assign the currently highlighted CC to the tree being built and clear the highlight."""
        from nvitk.gui.tools.runner import notify

        if self._finished:
            return
        if self.highlight_id is None:
            notify("Click a connected component on the labeled-CC layer first.", error=True)
            return
        lid = int(self.highlight_id)
        if lid in self.assigned:
            notify(f"CC {lid} is already assigned.", error=True)
            return
        self.assigned[lid] = int(self.tree_label)
        self.highlight_id = None
        try:
            if hasattr(self.labels_layer, "show_selected_label"):
                self.labels_layer.show_selected_label = False
        except Exception:
            pass
        self._refresh_display()
        self._push_status()
        notify(f"Added CC {lid} to {self.tree_name}.")

    def finish_current_tree(self) -> None:
        """Move on to the next vessel tree, or run the final expand once all trees are done."""
        from nvitk.gui.tools.runner import notify

        if self._finished:
            return
        n_tree = sum(1 for v in self.assigned.values() if v == self.tree_label)
        notify(f"{self.tree_name} done ({n_tree} CC(s)).")
        if self.tree_index + 1 < len(TREE_SPECS):
            self.tree_index += 1
            self.highlight_id = None
            try:
                if hasattr(self.labels_layer, "show_selected_label"):
                    self.labels_layer.show_selected_label = False
            except Exception:
                pass
            self._refresh_display()
            self._push_status()
            notify(f"Now select: {self.tree_name}.")
            return
        self._finalize()

    def _finalize(self) -> None:
        """Build multilabel seeds from the assigned CCs, run the final blood-flood expand, add the
        resulting trees Labels layer, and end the interactive session."""
        from nvitk.gui.core.spatial import layer_spatial_kwargs
        from nvitk.gui.tools.runner import notify

        src = to_numpy(self.labels_layer.data)
        meta = getattr(self.labels_layer, "metadata", None) or {}
        cached = meta.get(_META_SOURCE_KEY)
        if cached is not None and np.asarray(cached).shape == src.shape:
            src = np.asarray(cached)

        seeds = np.zeros(src.shape, dtype=np.int32)
        for cc_id, tree_lab in self.assigned.items():
            seeds[src == int(cc_id)] = int(tree_lab)

        notify("Mouse TOF CoW: final multilabel blood-flood expand…")
        out = expand_cow_trees(self.intensity, seeds)

        spatial = layer_spatial_kwargs(self.labels_layer)
        name = f"{self.source_name}_tof_cow_trees"
        trees_layer = self.viewer.add_labels(
            out,
            name=name,
            opacity=0.7,
            **spatial,
        )
        try:
            trees_layer._nvitk_label_like = True
        except Exception:
            pass

        self._finished = True
        self.uninstall()
        _clear_viewer_session(self.viewer)
        self._push_status()
        _emit_visibility()
        notify(
            f"Mouse TOF CoW complete: added {name!r} "
            f"(1=Left ICA, 2=Right ICA, 3=Basilar; {len(self.assigned)} CC(s))."
        )

    def _refresh_display(self) -> None:
        """Tint assigned CCs by tree; keep free CCs on the original palette."""
        from nvitk.gui.labels.visibility import get_label_color, unique_layer_labels

        layer = self.labels_layer
        src = to_numpy(layer.data)
        meta = dict(getattr(layer, "metadata", None) or {})
        cached = meta.get(_META_SOURCE_KEY)
        if cached is not None and np.asarray(cached).shape == src.shape:
            src = np.asarray(cached)
        all_ids = unique_layer_labels(src)
        if not all_ids:
            return

        backup = meta.get("nvitk_tof_cow_color_backup")
        if backup is None:
            backup = {int(lid): get_label_color(layer, int(lid)) for lid in all_ids}
            meta["nvitk_tof_cow_color_backup"] = backup
            layer.metadata = meta
        else:
            backup = {int(k): np.asarray(v, dtype=np.float32) for k, v in dict(backup).items()}

        tree_colors = {
            1: np.array([0.2, 0.6, 1.0, 0.85], dtype=np.float32),
            2: np.array([1.0, 0.35, 0.35, 0.85], dtype=np.float32),
            3: np.array([0.35, 0.9, 0.4, 0.85], dtype=np.float32),
        }
        color_dict: dict[Any, np.ndarray] = {
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            None: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        for lid in all_ids:
            if lid in self.assigned:
                color_dict[int(lid)] = tree_colors.get(
                    int(self.assigned[lid]),
                    np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                )
            else:
                color_dict[int(lid)] = np.asarray(
                    backup.get(lid, [0.9, 0.9, 0.2, 1.0]), dtype=np.float32
                )

        try:
            from napari.utils.colormaps import DirectLabelColormap

            layer.colormap = DirectLabelColormap(color_dict=color_dict)
        except Exception:
            if hasattr(layer, "color"):
                layer.color = {k: v for k, v in color_dict.items() if k not in (None,)}

    def _push_status(self) -> None:
        """Publish the current status text through the registered status hook."""
        _emit_status(self.status_text())


def session_active(viewer: Any | None = None) -> bool:
    """True if *viewer* has an unfinished Stage-2 assignment session."""
    if viewer is not None:
        sess = getattr(viewer, _SESSION_ATTR, None)
        return sess is not None and not getattr(sess, "_finished", True)
    return False


def get_session(viewer: Any) -> MouseTofCowSession | None:
    """The active Stage-2 session on *viewer*, or ``None`` if there isn't one (or it's finished)."""
    sess = getattr(viewer, _SESSION_ATTR, None)
    if sess is None or getattr(sess, "_finished", True):
        return None
    return sess


def _clear_viewer_session(viewer: Any) -> None:
    """Detach the stored Stage-2 session from *viewer*."""
    try:
        setattr(viewer, _SESSION_ATTR, None)
    except Exception:
        pass


def cancel_session(viewer: Any, *, notify_user: bool = True) -> None:
    """Uninstall Stage-2 picking; leave the CC layer unchanged."""
    from nvitk.gui.tools.runner import notify

    sess = getattr(viewer, _SESSION_ATTR, None)
    if sess is None:
        _emit_visibility()
        return
    try:
        sess.uninstall()
    except Exception:
        pass
    _clear_viewer_session(viewer)
    _emit_status("Mouse TOF CoW: cancelled.")
    _emit_visibility()
    if notify_user:
        notify("Mouse TOF CoW Stage 2 cancelled.")


def start_mouse_tof_cow(viewer: Any, layer: Any) -> None:
    """Run Stage 1 on *layer*, add labeled-CC Labels, start Stage 2 session."""
    from nvitk.gui.core.spatial import layer_spatial_kwargs, layer_to_image
    from nvitk.gui.tools.runner import notify

    if layer is None:
        raise ValueError("Select an intensity volume layer first.")
    if type(layer).__name__ == "Labels":
        # Allow re-running only from intensity; Labels are Stage-2 targets.
        raise ValueError(
            "Mouse TOF CoW Stage 1 needs an intensity Image layer "
            "(not a Labels layer). Select the TOF volume."
        )

    # Cancel any open session before starting a new one.
    if getattr(viewer, _SESSION_ATTR, None) is not None:
        cancel_session(viewer, notify_user=False)

    img = layer_to_image(layer)
    if img.ndim != 3:
        raise ValueError(f"Mouse TOF CoW expects a 3D volume, got {img.ndim}D")

    notify("Mouse TOF CoW Stage 1 running (N4 → blood flood → label CCs)…")
    labeled, intensity = run_stage1(img)
    n_cc = int(len(np.unique(labeled)) - (1 if (labeled == 0).any() else 0))
    if n_cc <= 0:
        raise ValueError("Stage 1 produced no connected components.")

    spatial = layer_spatial_kwargs(layer)
    cc_name = f"{layer.name}_tof_cow_cc"
    meta = {_META_SOURCE_KEY: np.array(labeled, copy=True)}
    labels_layer = viewer.add_labels(
        labeled,
        name=cc_name,
        opacity=0.7,
        metadata=meta,
        **spatial,
    )
    try:
        labels_layer._nvitk_label_like = True
    except Exception:
        pass

    sess = MouseTofCowSession(
        viewer=viewer,
        labels_layer=labels_layer,
        source_name=str(layer.name),
        intensity=np.asarray(intensity, dtype=np.float32),
    )
    setattr(viewer, _SESSION_ATTR, sess)
    sess.install()
    _emit_visibility()
    notify(
        f"Stage 1 done: {cc_name!r} ({n_cc} CCs). "
        f"Click components for {sess.tree_name}, then Add CC / Tree done."
    )


__all__ = [
    "MouseTofCowSession",
    "TREE_SPECS",
    "cancel_session",
    "expand_cow_trees",
    "get_session",
    "run_stage1",
    "session_active",
    "set_ui_hooks",
    "start_mouse_tof_cow",
]
