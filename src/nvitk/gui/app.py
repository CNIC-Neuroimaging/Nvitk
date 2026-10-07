"""Napari application shell for the nvitk GUI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy

from nvitk.gui.io.export import export_selected_layer
from nvitk.gui.io.napari_io import (
    install_nvitk_io,
    install_nvitk_layer_hooks,
    open_paths_with_nvitk,
)
from nvitk.gui.core.spatial import attach_orientation_status, find_spatial_reference_layer, layer_spatial_kwargs
from nvitk.gui.core.design import (
    SPACE,
    SPACE_TIGHT,
    apply_theme,
    register_napari_theme,
    set_theme,
    stored_theme,
    theme_toggle_button,
)
from nvitk.gui.core.log_panel import build_log_dock_widget
from nvitk.gui.tools.runner import notify
from nvitk.gui.panels.dicom_tags import DicomTagsPanel, layer_has_dicom_tags
from nvitk.gui.panels.image_properties import ImagePropertiesPanel
from nvitk.gui.viz.ct_window_panel import CTWindowPanel
from nvitk.gui.tools.dock import build_tools_dock
from nvitk.gui.core.notifications import install_notification_timeout
from nvitk.gui.core.warnings import install_napari_display_warnings


def _record_step(state: dict[str, Any], step: dict[str, Any]) -> None:
    """Append a timestamped *step* to the app's pipeline recording, if recording is enabled."""
    if not state.get("record_enabled", False):
        return
    step = dict(step)
    step.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    state["steps"].append(step)


def _layer_display_kwargs(layer: Any, *, name: str, ndim: int | None = None) -> dict[str, Any]:
    """Preserve spatial metadata from a source layer when adding tool outputs.

    *ndim* is the output's dimensionality, when it differs from the source's (a
    3D result of a 3D+t layer): the placement is cut down to the trailing dims.
    """
    from nvitk.gui.labels.visibility import copy_layer_metadata_for_output

    kwargs = {"name": name}
    meta = copy_layer_metadata_for_output(getattr(layer, "metadata", None))
    if meta:
        if ndim is not None and ndim != int(getattr(layer.data, "ndim", ndim)):
            # A 3D output of a time-first layer is not itself time-first.
            nested = meta.get("nvitk_metadata")
            if isinstance(nested, dict):
                nested = dict(nested)
                for key in ("time_leading", "source_axes", "axes"):
                    nested.pop(key, None)
                meta = dict(meta, nvitk_metadata=nested)
            meta.pop("axes", None)
        kwargs["metadata"] = meta
    same = ndim is None or ndim == int(getattr(layer.data, "ndim", ndim))
    kwargs.update(layer_spatial_kwargs(layer) if same else layer_spatial_kwargs(layer, ndim=ndim))
    return kwargs


def _refresh_layer_list(widget: Any, viewer: Any, registry: dict[str, Any]) -> None:
    """Rebuild the Layers tab's summary list widget from the viewer's layers and the app's input/
    output/mesh registry."""
    widget.clear()
    widget.addItem("--- Viewer layers ---")
    for layer in viewer.layers:
        widget.addItem(f"  {layer.name} ({layer.__class__.__name__})")
    widget.addItem("--- Opened inputs (registry) ---")
    for item in registry.get("inputs", []):
        widget.addItem(f"  {item.get('name', '?')}  {item.get('path', '')}")
    widget.addItem("--- Tool outputs (registry) ---")
    for item in registry.get("outputs", []):
        widget.addItem(f"  {item.get('name', '?')}  shape={item.get('shape')}")
    widget.addItem("--- Meshes (registry) ---")
    for item in registry.get("meshes", []):
        widget.addItem(f"  {item.get('name', '?')}")


def _top_aligned(widget: Any) -> Any:
    """*widget* pinned to the top of a container that takes any spare height.

    A magicgui form dropped straight into a resizable scroll area shares the
    dock's extra height out between its rows, so three controls ended up spread
    over the whole panel with large gaps between them.
    """
    from qtpy.QtWidgets import QVBoxLayout, QWidget

    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(SPACE_TIGHT, SPACE, SPACE_TIGHT, SPACE)
    lay.setSpacing(SPACE)
    lay.addWidget(widget)
    lay.addStretch(1)
    return host


def _move_tab_after(window: Any, anchor: Any, dock: Any) -> bool:
    """Put *dock*'s tab right after *anchor*'s (see :func:`nvitk.gui.core.workspace.move_tab_after`)."""
    from nvitk.gui.core.workspace import move_tab_after

    return move_tab_after(window, anchor, dock)


def _scrollable_tab(widget: Any) -> Any:
    """Wrap a panel so its content scrolls instead of setting the dock's floor.

    Tabbed docks share one minimum size, the largest of their pages, and a dock
    passes that minimum up to the window: one tall panel (the data browser wants
    ~1000 px) made the whole panel group refuse to be shorter than it, so the
    window could not be resized down and would not go fullscreen on a shorter
    screen.

    ``widgetResizable`` keeps the page filling the tab whenever it fits, so tables
    and canvases still expand — only content taller than the dock scrolls.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QScrollArea

    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setWidget(widget)
    return area


def run_app() -> None:
    """Build and launch the nvitk Napari GUI: creates the viewer, installs nvitk's I/O hooks, and
    assembles the Imaging (tools)/Labels/Meshlab/Data/QC/Statmodels/Layers/Export/Pipeline panel docks."""
    install_napari_display_warnings()
    install_notification_timeout()
    import napari
    from magicgui import magicgui
    from qtpy.QtWidgets import (
        QFileDialog,
        QLabel,
        QListWidget,
        QVBoxLayout,
        QWidget,
    )

    # The theme the last session chose, in force before a single widget exists:
    # every panel then builds from the palette it will be shown in, rather than
    # being restyled after the fact.
    set_theme(stored_theme())
    # CPU worker budget (and Napari async slicing) before any panel starts work.
    try:
        from nvitk.gui.core.performance import apply_performance

        apply_performance()
    except Exception:  # noqa: BLE001 — a preference must never block a launch
        pass
    # Register before the viewer exists so its chrome is painted from the nvitk
    # palette on the first frame, rather than flashing Napari's default dark.
    theme_id = register_napari_theme()

    viewer = napari.Viewer(title="nvitk")
    _ = viewer.window
    try:
        viewer.theme = theme_id
    except Exception:
        pass
    install_nvitk_io(viewer)
    # The wheel scrolls panels; it never flips a dropdown or nudges an unfocused field.
    from nvitk.gui.core.wheel_guard import install_wheel_guard

    install_wheel_guard()
    install_nvitk_layer_hooks(viewer)

    app_state: dict[str, Any] = {
        "viewer": viewer,
        "record_enabled": False,
        "steps": [],
        "inputs": [],
        "outputs": [],
        "meshes": [],
        "xnat_temp_dirs": [],
        "sge_pending_jobs": [],
        "sge_last_connection": {},
    }
    viewer._nvitk_app_state = app_state

    layer_list = QListWidget()

    def _on_layers_changed() -> None:
        """Refresh the Layers tab's summary list."""
        _refresh_layer_list(layer_list, viewer, app_state)

    tools_widget, tool_panel = build_tools_dock(
        viewer,
        app_state,
        layer_display_kwargs=_layer_display_kwargs,
        on_layers_changed=_on_layers_changed,
        record_step=lambda step: _record_step(app_state, step),
    )
    dicom_tags_panel = DicomTagsPanel()
    image_props_panel = ImagePropertiesPanel()
    ct_window_panel = CTWindowPanel()
    ct_window_panel.set_viewer(viewer)
    image_props_panel.set_viewer(viewer)

    def _on_xnat_inputs_opened(paths: list[str]) -> None:
        """Record newly opened XNAT/data-browser input paths in the app registry."""
        for p in paths:
            app_state["inputs"].append({"path": p, "name": Path(p).name})
        _on_layers_changed()

    try:
        from nvitk.gui.panels.data_browser import DataBrowserPanel

        xnat_panel = DataBrowserPanel(
            viewer,
            app_state,
            on_inputs_opened=_on_xnat_inputs_opened,
        )
        data_tab_label = "Data"
    except Exception as exc:
        xnat_panel = QLabel(f"Data browser unavailable: {exc}")
        xnat_panel.setWordWrap(True)
        data_tab_label = "Data"

    try:
        from nvitk.gui.panels.qc import QcPanel

        qc_panel = QcPanel(
            viewer,
            app_state,
            on_inputs_opened=_on_xnat_inputs_opened,
        )
    except Exception as exc:
        qc_panel = QLabel(f"QC panel unavailable: {exc}")
        qc_panel.setWordWrap(True)

    try:
        from nvitk.gui.panels.statmodels import StatmodelsPanel

        statmodels_panel = StatmodelsPanel()
    except Exception as exc:
        statmodels_panel = QLabel(f"Statmodels unavailable: {exc}")
        statmodels_panel.setWordWrap(True)

    from nvitk.gui.tools.gpu_toggle import backend_label
    from nvitk.gui.core.log_panel import gui_log

    gui_log(f"Compute backend: {backend_label()}")

    @magicgui(
        record_steps={"label": "Record pipeline steps", "value": False},
        labels_opacity={"label": "Labels overlay opacity", "min": 0.0, "max": 1.0, "value": 0.6},
        call_button="Overlay mask as Labels (0=transparent)",
    )
    def layers_panel(record_steps: bool = False, labels_opacity: float = 0.6) -> None:
        """Toggle pipeline-step recording and add a Labels overlay (background transparent) for the
        active layer's data."""
        app_state["record_enabled"] = record_steps
        if not viewer.layers:
            notify("No layer selected.", error=True)
            return
        layer = viewer.layers.selection.active or viewer.layers[-1]
        data = to_numpy(layer.data)
        if data.ndim not in (2, 3):
            notify("Labels overlay needs a 2D or 3D layer.", error=True)
            return
        labels = to_numpy(data).astype(np.int32)
        spatial_src = find_spatial_reference_layer(viewer, layer)
        kwargs = _layer_display_kwargs(spatial_src, name=f"{layer.name}_labels")
        meta = kwargs.pop("metadata", None)
        viewer.add_labels(
            labels,
            name=f"{layer.name}_labels",
            opacity=float(labels_opacity),
            affine=kwargs.get("affine"),
            scale=kwargs.get("scale"),
        )
        if meta is not None:
            viewer.layers[-1].metadata = meta
        notify(
            "Added Labels layer: background (label 0) is transparent; "
            "adjust opacity in the layer controls."
        )
        _on_layers_changed()

    @magicgui(call_button="Refresh layer list")
    def layers_refresh_panel() -> None:
        """Manually refresh the Layers tab's summary list."""
        _on_layers_changed()

    @magicgui(
        record_steps={"label": "Record pipeline steps", "value": False},
        path={"label": "Export path", "value": "pipeline.json"},
        call_button="Export pipeline JSON",
    )
    def export_panel(record_steps: bool = False, path: str = "pipeline.json") -> None:
        """Toggle pipeline-step recording and write the recorded steps/inputs/outputs/meshes/layers to
        a JSON file at *path*."""
        app_state["record_enabled"] = record_steps
        doc = {
            "record_enabled": app_state["record_enabled"],
            "steps": app_state["steps"],
            "inputs": app_state["inputs"],
            "outputs": app_state["outputs"],
            "meshes": app_state["meshes"],
            "layers": [layer.name for layer in viewer.layers],
        }
        out = Path(path)
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        notify(f"Pipeline exported to {out}")

    @magicgui(
        path={"label": "View PNG path", "value": "view.png"},
        view_canvas_only={"label": "Canvas only", "value": True},
        call_button="Export 3D view (PNG)",
    )
    def export_view_png_panel(path: str = "view.png", view_canvas_only: bool = True) -> None:
        """Save a screenshot of the current 3D view to *path*."""
        from nvitk.gui.viz.view_capture import export_view_png

        out = path.strip()
        if not out:
            notify("Set a PNG path.", error=True)
            return
        try:
            export_view_png(viewer, out, canvas_only=view_canvas_only)
        except Exception as exc:
            notify(f"View export failed: {exc}", error=True)
            return
        notify(f"Saved 3D view → {out}")

    @magicgui(
        path={"label": "View GIF path", "value": "view.gif"},
        gif_fps={"label": "Frames per second", "value": 8.0, "min": 0.5, "max": 60.0},
        view_canvas_only={"label": "Canvas only", "value": True},
        call_button="Export 3D view (GIF, 4D)",
    )
    def export_view_gif_panel(
        path: str = "view.gif",
        gif_fps: float = 8.0,
        view_canvas_only: bool = True,
    ) -> None:
        """Save an animated GIF of the active 4D layer's cardiac-phase playback to *path*."""
        from nvitk.gui.viz.view_capture import export_view_gif

        out = path.strip()
        if not out:
            notify("Set a GIF path.", error=True)
            return
        layer = viewer.layers.selection.active or (viewer.layers[-1] if viewer.layers else None)
        try:
            n = export_view_gif(
                viewer,
                out,
                fps=float(gif_fps),
                canvas_only=view_canvas_only,
                layer=layer,
            )
        except Exception as exc:
            notify(f"GIF export failed: {exc}", error=True)
            return
        notify(f"Saved {n}-frame GIF → {out}")

    @magicgui(
        path={"label": "Output path", "value": "output.nii.gz"},
        use_file_affine={
            "label": "Use original file affine (from nvitk metadata)",
            "value": True,
        },
        force_type={
            "label": "Format override (optional)",
            "value": "",
        },
        call_button="Export active layer",
    )
    def save_panel(
        path = "output.nii.gz",
        use_file_affine = True,
        force_type = "",
    ) -> None:
        """Export the active layer to *path*, recording the export as a pipeline step."""
        out = path.strip()
        if not out:
            notify("Set an output path.", error=True)
            return
        ft = force_type.strip() or None
        export_selected_layer(
            viewer,
            out,
            use_file_affine=use_file_affine,
            force_type=ft,
        )
        layer = viewer.layers.selection.active or viewer.layers[-1]
        _record_step(
            app_state,
            {
                "type": "export",
                "path": out,
                "layer": layer.name,
                "use_file_affine": use_file_affine,
                "force_type": ft,
            },
        )
        _refresh_layer_list(layer_list, viewer, app_state)

    def _save_layer_dialog() -> None:
        """Open a native "Save As" dialog for the active layer and export it to the chosen path."""
        layer = viewer.layers.selection.active or (viewer.layers[-1] if viewer.layers else None)
        if layer is None:
            notify("No layer to export.", error=True)
            return
        layer_type = layer.__class__.__name__
        if layer_type == "Surface":
            filt = "STL (*.stl);;PLY (*.ply);;OBJ (*.obj);;VTK PolyData (*.vtp);;OFF (*.off);;GIfTI (*.gii);;All (*)"
            default = f"{layer.name}.stl"
        elif layer_type == "Points":
            filt = "XYZ (*.xyz);;CSV (*.csv);;PLY (*.ply);;VTK PolyData (*.vtp);;All (*)"
            default = f"{layer.name}.xyz"
        else:
            filt = "NIfTI (*.nii *.nii.gz);;TIFF (*.tif *.tiff);;MetaImage (*.mha);;All (*)"
            default = f"{layer.name}.nii.gz"
        dlg = QFileDialog()
        out, _ = dlg.getSaveFileName(None, "Export layer", default, filt)
        if out:
            save_panel.path.value = out
            export_selected_layer(
                viewer,
                out,
                use_file_affine=bool(save_panel.use_file_affine.value),
                force_type=(save_panel.force_type.value.strip() or None),
            )

    def _open_files() -> None:
        """Open a native file-open dialog and load the selected images into the viewer via nvitk's reader."""
        dlg = QFileDialog()
        paths, _ = dlg.getOpenFileNames(
            None,
            "Open image",
            "",
            "Images (*.nii *.nii.gz *.mha *.tif *.tiff *.b2nd);;"
            "Meshes and point clouds (*.stl *.obj *.off *.ply *.vtk *.vtp *.gii *.xyz *.pcd *.pvd);;"
            "Preprocessed (nnU-Net/nnssl) (*.b2nd *.pkl);;"
            "All (*)",
        )
        for p in paths:
            open_paths_with_nvitk(viewer, p)
            app_state["inputs"].append({"path": p, "name": Path(p).stem})
            _record_step(app_state, {"type": "open", "path": p})
        if paths:
            _refresh_layer_list(layer_list, viewer, app_state)

    orientation_label = QLabel("Orientation: —")
    orientation_label.setWordWrap(True)
    attach_orientation_status(viewer, orientation_label)

    layers_tab = QWidget()
    from qtpy.QtCore import Qt

    layers_layout = QVBoxLayout()
    layers_layout.setAlignment(Qt.AlignTop)
    layers_layout.setContentsMargins(SPACE_TIGHT, SPACE, SPACE_TIGHT, SPACE)
    layers_layout.setSpacing(SPACE)
    layers_layout.addWidget(orientation_label)
    layers_layout.addWidget(layer_list)
    layers_layout.addWidget(layers_panel.native)
    layers_layout.addWidget(layers_refresh_panel.native)
    layers_layout.addWidget(ct_window_panel)
    layers_layout.addStretch(1)
    layers_tab.setLayout(layers_layout)

    export_tab = QWidget()
    export_layout = QVBoxLayout()
    export_layout.setAlignment(Qt.AlignTop)
    export_layout.setContentsMargins(SPACE_TIGHT, SPACE, SPACE_TIGHT, SPACE)
    export_layout.setSpacing(SPACE)
    export_layout.addWidget(export_view_png_panel.native)
    export_layout.addWidget(export_view_gif_panel.native)
    export_layout.addWidget(save_panel.native)
    export_layout.addStretch(1)
    export_tab.setLayout(export_layout)

    from nvitk.gui.mesh.panel import build_mesh_panel
    from nvitk.gui.panels.dicom_browser import build_dicom_browser

    mesh_panel = build_mesh_panel(viewer)
    dicom_browser = build_dicom_browser(viewer)

    # One dock per panel rather than tabs in a single dock, so each can be popped
    # out, moved to the workspace window, or split beside another. Tabbed
    # together on the right they look as the old tab widget did. Order: what is
    # used on the active layer first (tools, labels, its properties and tags),
    # then data management, then output and housekeeping.
    from nvitk.gui.core.workspace import install_panel_manager, make_panel_dock
    from nvitk.gui.labels.panel import build_labels_dock

    panel_manager = install_panel_manager(viewer)
    qt_main = viewer.window._qt_window

    def _panel_dock(widget: Any, key: str, title: str, **kwargs: Any) -> Any:
        return make_panel_dock(
            viewer, _scrollable_tab(widget), object_name=f"nvitk:{key}", title=title, **kwargs
        )

    # On the Imaging panel's own title bar: the panel most often on screen, and
    # inside the nvitk stylesheet rather than Napari's, which clamps buttons to 12px.
    theme_button = theme_toggle_button(viewer, None)
    tools_dock = _panel_dock(tools_widget, "tools", "Imaging", extras=[theme_button])
    labels_dock = build_labels_dock(viewer)
    image_props_dock = _panel_dock(image_props_panel, "image_properties", "Image properties")
    dicom_dock = _panel_dock(dicom_tags_panel, "dicom_tags", "DICOM tags")
    dicom_browser_dock = _panel_dock(dicom_browser, "dicom_browser", "DICOM browser")
    mesh_dock = _panel_dock(mesh_panel, "mesh", "Meshlab")
    panel_docks = [
        tools_dock,
        labels_dock,
        mesh_dock,
        image_props_dock,
        dicom_dock,
        dicom_browser_dock,
        _panel_dock(xnat_panel, "data", data_tab_label),
        _panel_dock(qc_panel, "qc", "QC"),
        _panel_dock(statmodels_panel, "statmodels", "Statmodels"),
        _panel_dock(export_tab, "export", "Export"),
        _panel_dock(layers_tab, "layers", "Layers"),
        _panel_dock(_top_aligned(export_panel.native), "pipeline", "Pipeline"),
    ]
    qt_main.addDockWidget(Qt.RightDockWidgetArea, tools_dock)
    for panel_dock in panel_docks[1:]:
        qt_main.tabifyDockWidget(tools_dock, panel_dock)
    tools_dock.raise_()
    if panel_manager is not None:
        panel_manager.set_labels_dock(labels_dock)
        panel_manager.theme_button = theme_button

    log_dock = build_log_dock_widget()
    apply_theme(log_dock)
    viewer.window.add_dock_widget(log_dock, area="bottom", name="nvitk log")

    # A ▾ on each label layer's row in the layer list unfolds it into its labels.
    try:
        from nvitk.gui.labels.layer_list import install_layer_list_label_buttons

        label_delegate = install_layer_list_label_buttons(viewer)
    except Exception as exc:  # noqa: BLE001 — Napari's own layer list still works
        label_delegate = None
        gui_log(f"Layer-list label buttons unavailable: {exc}", error=True)

    # Folders and subfolders in the layer list (headers painted by that delegate).
    try:
        from nvitk.gui.core.layer_folders import install_layer_folders

        install_layer_folders(viewer, label_delegate)
    except Exception as exc:  # noqa: BLE001 — the flat list still works
        gui_log(f"Layer folders unavailable: {exc}", error=True)

    # The orthogonal views open with the window. Created before the saved layout
    # is restored, so a position the user gave the dock last time is honoured.
    ortho_panel = None
    try:
        from nvitk.gui.viz.ortho_panel import open_ortho_views

        ortho_panel = open_ortho_views(viewer, None)
    except Exception as exc:  # noqa: BLE001 — a missing panel must not block a launch
        from nvitk.gui.core.log_panel import gui_log as _gui_log

        _gui_log(f"Orthogonal views unavailable: {exc}", error=True)

    # Both nvitk docks now exist, so a saved layout has something to match
    # against. Napari's own restore ran inside ``napari.Viewer(...)`` far above,
    # when neither of these existed, and ``restoreState`` silently drops entries
    # whose objectName it cannot find — hence a second, explicit restore here.
    try:
        from nvitk.gui.core.prefs import (
            ensure_prefs_file,
            restore_dock_state,
            stored_dock_layout_version,
        )

        # Seed gui.json beside the rest of the configuration the first time a
        # configured install opens the GUI, so there is somewhere for the layout
        # to be remembered rather than it only appearing after a clean exit.
        ensure_prefs_file()
        stored_version = stored_dock_layout_version()
        legacy_layout = stored_version < 2
        # The workspace first: it takes its panels out of the main window, and
        # the main layout is then restored around the ones that stay.
        if panel_manager is not None:
            panel_manager.restore()
        restore_dock_state(qt_main)
        if legacy_layout:
            # Saved when the panels were tabs of one "nvitk" dock: the state
            # places Napari's docks and the ortho views, but knows none of the
            # panel docks, and Qt stacks those one under another. Tab them back
            # together, once — the next save is in the new format.
            for panel_dock in panel_docks[1:]:
                if not panel_dock.isFloating() and qt_main.dockWidgetArea(panel_dock) != Qt.NoDockWidgetArea:
                    qt_main.tabifyDockWidget(tools_dock, panel_dock)
            tools_dock.raise_()
        else:
            # A saved layout keeps its own tab order: tabs added or moved since it
            # was saved are put in place once. Version 3 moved the Meshlab tab next
            # to Labels; version 4 added the DICOM browser after DICOM tags.
            if stored_version < 3:
                _move_tab_after(qt_main, labels_dock, mesh_dock)
            if stored_version < 4:
                _move_tab_after(qt_main, dicom_dock, dicom_browser_dock)
    except Exception:  # noqa: BLE001 — a stored layout must not block a launch
        pass
    if panel_manager is not None:
        panel_manager.watch_all()
    # On screen at launch whatever the stored layout says: a layout saved before
    # the dock existed — or with it closed — would otherwise leave it hidden.
    ortho_dock = getattr(ortho_panel, "_nvitk_dock", None)
    if ortho_dock is not None:
        ortho_dock.show()
        ortho_dock.raise_()

    _refresh_layer_list(layer_list, viewer, app_state)

    # Panels whose contents are rebuilt from the active layer. Refreshing one that
    # nobody is looking at — tabbed behind another, closed, or in a hidden
    # workspace — is pure latency on every click in the layer list, so it only
    # records that it is out of date and catches up when it is next shown. The
    # panels themselves still refresh on demand, so direct callers (and tests)
    # are unaffected.
    _stale_docks: set[str] = set()

    def _on_screen(dock: Any) -> bool:
        """True when *dock*'s contents can be seen."""
        return bool(dock.isVisible() and dock.widget() is not None and dock.widget().isVisible())

    def _refresh_dicom_tags_tab(force: bool = False) -> None:
        """Update the DICOM tags panel for the active layer, greying it out when it has no tags."""
        layer = (
            viewer.layers.selection.active
            if viewer.layers
            else None
        )
        has_tags = layer_has_dicom_tags(layer)
        # The enabled state is visible even from a neighbouring tab, so it is kept
        # current while the panel's own contents wait to be looked at.
        dicom_tags_panel.setEnabled(has_tags)
        dicom_dock.setToolTip(
            "DICOM metadata for the active layer" if has_tags
            else "The active layer has no DICOM tags."
        )
        if not force and not _on_screen(dicom_dock):
            _stale_docks.add("dicom")
            return
        _stale_docks.discard("dicom")
        dicom_tags_panel.refresh_from_layer(layer)

    def _refresh_image_properties_tab(force: bool = False) -> None:
        """Update the Image properties panel for the active layer."""
        if not force and not _on_screen(image_props_dock):
            _stale_docks.add("image_properties")
            return
        _stale_docks.discard("image_properties")
        layer = viewer.layers.selection.active if viewer.layers else None
        image_props_panel.refresh_from_layer(layer)

    def _catch_up(key: str, refresh: Any) -> Any:
        """A ``visibilityChanged`` slot bringing the panel up to date when shown."""

        def _slot(visible: bool) -> None:
            if visible and key in _stale_docks:
                refresh(force=True)

        return _slot

    dicom_dock.visibilityChanged.connect(_catch_up("dicom", _refresh_dicom_tags_tab))
    image_props_dock.visibilityChanged.connect(
        _catch_up("image_properties", _refresh_image_properties_tab)
    )

    @viewer.layers.selection.events.active.connect
    def _on_active_layer_changed(_event: Any) -> None:
        """Refresh the DICOM tags and image properties panels when the active layer selection changes."""
        _refresh_dicom_tags_tab()
        _refresh_image_properties_tab()

    @viewer.layers.events.inserted.connect
    def _on_layer_inserted_dicom(_event: Any) -> None:
        """Refresh the DICOM tags and image properties panels when a new layer is added."""
        _refresh_dicom_tags_tab()
        _refresh_image_properties_tab()

    _refresh_dicom_tags_tab()
    _refresh_image_properties_tab()

    try:
        # The main window, not the viewer widget inside it. Napari's
        # _QtMainWindow.closeEvent tears the session down without ever calling
        # closeEvent on its child QtViewer, and Qt does not deliver close events
        # to children on its own, so a handler installed on the viewer never runs.
        qt_window = viewer.window._qt_window
        _orig_close = qt_window.closeEvent

        def _close_with_xnat_cleanup(event: Any) -> None:
            """Clean up XNAT temp dirs and shut down the SGE monitor / vessel cross-section state
            before delegating to the original Qt close handler."""
            if hasattr(xnat_panel, "cleanup_temp_dirs"):
                xnat_panel.cleanup_temp_dirs()
            from nvitk.gui.sge.poll import shutdown_sge_monitor
            from nvitk.gui.viz.vessel_cross_sections import shutdown_vessel_cross_sections

            shutdown_sge_monitor(app_state)
            shutdown_vessel_cross_sections(app_state)
            # Before delegating: Napari's handler un-floats every floating dock
            # on its way out, so a layout saved afterwards would forget which
            # panels the user had torn off.
            try:
                from nvitk.gui.core.prefs import save_dock_state

                save_dock_state(qt_window)
                if panel_manager is not None:
                    panel_manager.save()
                    # Its own window: left open, it would keep the app running.
                    panel_manager.shutdown()
            except Exception:  # noqa: BLE001 — never block a close on a preference
                pass
            if _orig_close is not None:
                _orig_close(event)

        qt_window.closeEvent = _close_with_xnat_cleanup
    except Exception:
        pass

    @viewer.bind_key("Control-T")
    def _transpose_axes(_viewer) -> None:
        """Same as Napari's transpose axes (Ctrl+T)."""
        try:
            _viewer.dims.transpose()
        except Exception as exc:
            notify(f"Transpose failed: {exc}", error=True)

    @viewer.bind_key("Ctrl+O")
    def _open_key(_viewer) -> None:
        """Keyboard shortcut: open the file-open dialog."""
        _open_files()

    @viewer.bind_key("Ctrl+Shift+S")
    def _save_key(_viewer) -> None:
        """Keyboard shortcut: open the save-active-layer dialog."""
        _save_layer_dialog()

    napari.run()
