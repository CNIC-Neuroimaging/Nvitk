"""Napari application shell for the nvitk GUI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.meshlab import mesh_from_image, marching_cubes_multilabel
from nvitk.types import Image, Mesh

from nvitk.gui.io.export import export_selected_layer
from nvitk.gui.io.napari_io import (
    install_nvitk_io,
    install_nvitk_layer_hooks,
    open_paths_with_nvitk,
)
from nvitk.gui.core.spatial import attach_orientation_status, find_spatial_reference_layer, layer_spatial_kwargs
from nvitk.gui.core.log_panel import build_log_dock_widget
from nvitk.gui.tools.runner import notify
from nvitk.gui.panels.dicom_tags import DicomTagsPanel, layer_has_dicom_tags
from nvitk.gui.tools.dock import build_tools_dock
from nvitk.gui.core.warnings import install_napari_display_warnings


def _record_step(state: dict[str, Any], step: dict[str, Any]) -> None:
    if not state.get("record_enabled", False):
        return
    step = dict(step)
    step.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    state["steps"].append(step)


def _layer_display_kwargs(layer: Any, *, name: str) -> dict[str, Any]:
    """Preserve spatial metadata from a source layer when adding tool outputs."""
    from nvitk.gui.labels.visibility import copy_layer_metadata_for_output

    kwargs = {"name": name}
    meta = copy_layer_metadata_for_output(getattr(layer, "metadata", None))
    if meta:
        kwargs["metadata"] = meta
    kwargs.update(layer_spatial_kwargs(layer))
    return kwargs


def _refresh_layer_list(widget: Any, viewer: Any, registry: dict[str, Any]) -> None:
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


def run_app() -> None:
    install_napari_display_warnings()
    import napari
    from magicgui import magicgui
    from qtpy.QtWidgets import (
        QFileDialog,
        QLabel,
        QListWidget,
        QSizePolicy,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    viewer = napari.Viewer(title="nvitk")
    _ = viewer.window
    install_nvitk_io(viewer)
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
        _refresh_layer_list(layer_list, viewer, app_state)

    tools_widget, tool_panel = build_tools_dock(
        viewer,
        app_state,
        layer_display_kwargs=_layer_display_kwargs,
        on_layers_changed=_on_layers_changed,
        record_step=lambda step: _record_step(app_state, step),
    )
    dicom_tags_panel = DicomTagsPanel()

    def _on_xnat_inputs_opened(paths: list[str]) -> None:
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

    @magicgui(call_button="Reconstruct mesh")
    def mesh_panel(multilabel: bool = False) -> None:
        if not viewer.layers:
            notify("No layers loaded.", error=True)
            return
        layer = viewer.layers.selection.active or viewer.layers[-1]
        from nvitk.gui.core.spatial import layer_to_image

        img = layer_to_image(layer)
        spatial = layer_spatial_kwargs(layer)
        try:
            if multilabel:
                meshes = marching_cubes_multilabel(img, world_space=False) or []
            else:
                m = mesh_from_image(img, multilabel=False, world_space=False)
                meshes = [m] if m is not None else []
        except Exception as exc:
            notify(f"Mesh reconstruction failed: {exc}", error=True)
            return

        if not meshes:
            notify("No surface found (empty mask or no labels).", error=True)
            return

        for mesh in meshes:
            if not isinstance(mesh, Mesh):
                continue
            surf = mesh.to_napari_surface()
            surf_kwargs = {"name": mesh.name, **spatial}
            viewer.add_surface(
                (surf["vertices"], surf["faces"]),
                **surf_kwargs,
            )
            app_state["meshes"].append({"name": mesh.name})
            _record_step(
                app_state,
                {
                    "type": "mesh",
                    "source_layer": layer.name,
                    "mesh_layer": mesh.name,
                    "multilabel": multilabel,
                },
            )
        notify(f"Added {len(meshes)} surface layer(s).")
        _refresh_layer_list(layer_list, viewer, app_state)

    @magicgui(
        record_steps={"label": "Record pipeline steps", "value": False},
        labels_opacity={"label": "Labels overlay opacity", "min": 0.0, "max": 1.0, "value": 0.6},
        call_button="Overlay mask as Labels (0=transparent)",
    )
    def layers_panel(record_steps: bool = False, labels_opacity: float = 0.6) -> None:
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
        _on_layers_changed()

    @magicgui(
        record_steps={"label": "Record pipeline steps", "value": False},
        path={"label": "Export path", "value": "pipeline.json"},
        call_button="Export pipeline JSON",
    )
    def export_panel(record_steps: bool = False, path: str = "pipeline.json") -> None:
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
        layer = viewer.layers.selection.active or (viewer.layers[-1] if viewer.layers else None)
        if layer is None:
            notify("No layer to export.", error=True)
            return
        layer_type = layer.__class__.__name__
        if layer_type == "Surface":
            filt = "STL (*.stl);;All (*)"
            default = f"{layer.name}.stl"
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
        dlg = QFileDialog()
        paths, _ = dlg.getOpenFileNames(
            None,
            "Open image",
            "",
            "Images (*.nii *.nii.gz *.mha *.tif *.tiff);;All (*)",
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
    layers_layout.setSpacing(6)
    layers_layout.addWidget(orientation_label)
    layers_layout.addWidget(layer_list)
    layers_layout.addWidget(layers_panel.native)
    layers_layout.addWidget(layers_refresh_panel.native)
    layers_layout.addStretch(1)
    layers_tab.setLayout(layers_layout)

    dock = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(4, 4, 4, 4)
    tabs = QTabWidget()
    tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    tabs.addTab(tools_widget, "Tools")
    tabs.addTab(xnat_panel, data_tab_label)
    tabs.addTab(qc_panel, "QC")
    tabs.addTab(statmodels_panel, "Statmodels")
    dicom_tab_index = tabs.addTab(dicom_tags_panel, "DICOM tags")
    tabs.addTab(mesh_panel.native, "Mesh")
    tabs.addTab(layers_tab, "Layers")
    export_tab = QWidget()
    export_layout = QVBoxLayout()
    export_layout.setAlignment(Qt.AlignTop)
    export_layout.setSpacing(6)
    export_layout.addWidget(export_view_png_panel.native)
    export_layout.addWidget(export_view_gif_panel.native)
    export_layout.addWidget(save_panel.native)
    export_layout.addStretch(1)
    export_tab.setLayout(export_layout)
    tabs.addTab(export_tab, "Export")
    tabs.addTab(export_panel.native, "Pipeline")
    layout.addWidget(tabs, stretch=1)
    dock.setLayout(layout)
    dock.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    viewer.window.add_dock_widget(dock, area="right", name="nvitk")
    log_dock = build_log_dock_widget()
    viewer.window.add_dock_widget(log_dock, area="bottom", name="nvitk log")

    _refresh_layer_list(layer_list, viewer, app_state)

    def _refresh_dicom_tags_tab() -> None:
        layer = (
            viewer.layers.selection.active
            if viewer.layers
            else None
        )
        dicom_tags_panel.refresh_from_layer(layer)
        has_tags = layer_has_dicom_tags(layer)
        tabs.setTabEnabled(dicom_tab_index, has_tags)
        if has_tags:
            tabs.setTabToolTip(
                dicom_tab_index,
                "DICOM metadata for the active layer",
            )

    @viewer.layers.selection.events.active.connect
    def _on_active_layer_changed(_event: Any) -> None:
        _refresh_dicom_tags_tab()

    @viewer.layers.events.inserted.connect
    def _on_layer_inserted_dicom(_event: Any) -> None:
        _refresh_dicom_tags_tab()

    _refresh_dicom_tags_tab()

    try:
        qt_viewer = viewer.window._qt_viewer
        _orig_close = qt_viewer.closeEvent

        def _close_with_xnat_cleanup(event: Any) -> None:
            if hasattr(xnat_panel, "cleanup_temp_dirs"):
                xnat_panel.cleanup_temp_dirs()
            from nvitk.gui.sge.poll import shutdown_sge_monitor
            from nvitk.gui.viz.vessel_cross_sections import shutdown_vessel_cross_sections

            shutdown_sge_monitor(app_state)
            shutdown_vessel_cross_sections(app_state)
            if _orig_close is not None:
                _orig_close(event)

        qt_viewer.closeEvent = _close_with_xnat_cleanup
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
        _open_files()

    @viewer.bind_key("Ctrl+Shift+S")
    def _save_key(_viewer) -> None:
        _save_layer_dialog()

    napari.run()
