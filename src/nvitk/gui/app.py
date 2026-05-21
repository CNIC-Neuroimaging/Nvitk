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

from nvitk.gui.export_napari import export_selected_layer
from nvitk.gui.io_napari import install_nvitk_io, open_paths_with_nvitk
from nvitk.gui.spatial import attach_orientation_status, find_spatial_reference_layer, layer_spatial_kwargs
from nvitk.gui.log_panel import build_log_dock_widget
from nvitk.gui.tool_runner import notify
from nvitk.gui.tools_dock import build_tools_dock
from nvitk.gui.warnings import install_napari_display_warnings


def _record_step(state: dict[str, Any], step: dict[str, Any]) -> None:
    if not state.get("record_enabled", True):
        return
    step = dict(step)
    step.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    state["steps"].append(step)


def _layer_display_kwargs(layer: Any, *, name: str) -> dict[str, Any]:
    """Preserve spatial metadata from a source layer when adding tool outputs."""
    kwargs: dict[str, Any] = {"name": name}
    meta = dict(getattr(layer, "metadata", None) or {})
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
    from qtpy.QtWidgets import QFileDialog, QLabel, QListWidget, QTabWidget, QVBoxLayout, QWidget

    viewer = napari.Viewer(title="nvitk")
    _ = viewer.window
    install_nvitk_io(viewer)

    app_state: dict[str, Any] = {
        "record_enabled": True,
        "steps": [],
        "inputs": [],
        "outputs": [],
        "meshes": [],
    }

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

    from nvitk.gui.gpu_toggle import backend_label
    from nvitk.gui.log_panel import gui_log

    gui_log(f"Compute backend: {backend_label()}")

    @magicgui(call_button="Reconstruct mesh")
    def mesh_panel(multilabel: bool = False) -> None:
        if not viewer.layers:
            notify("No layers loaded.", error=True)
            return
        layer = viewer.layers.selection.active or viewer.layers[-1]
        from nvitk.gui.spatial import layer_to_image

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
            surf_kwargs: dict[str, Any] = {"name": mesh.name, **spatial}
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
        paths={"label": "Paths (one per line)"},
        call_button="Run batch queue",
    )
    def batch_panel(paths: str = "") -> None:
        opened = 0
        for line in paths.strip().splitlines():
            p = Path(line.strip())
            if not p.exists():
                notify(f"Skipping missing path: {p}", error=True)
                continue
            open_paths_with_nvitk(viewer, p)
            app_state["inputs"].append({"path": str(p), "name": p.stem})
            _record_step(app_state, {"type": "batch_open", "path": str(p)})
            opened += 1
        if opened:
            notify(f"Opened {opened} file(s) from batch queue.")
            _refresh_layer_list(layer_list, viewer, app_state)
        else:
            notify("No valid paths in the batch list.", error=True)

    @magicgui(
        record_steps={"label": "Record pipeline steps", "value": True},
        labels_opacity={"label": "Labels overlay opacity", "min": 0.0, "max": 1.0, "value": 0.6},
        call_button="Overlay mask as Labels (0=transparent)",
    )
    def layers_panel(record_steps: bool = True, labels_opacity: float = 0.6) -> None:
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
        record_steps={"label": "Record pipeline steps", "value": True},
        path={"label": "Export path", "value": "pipeline.json"},
        call_button="Export pipeline JSON",
    )
    def export_panel(record_steps: bool = True, path: str = "pipeline.json") -> None:
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

    @magicgui(call_button="Clear recorded steps")
    def pipeline_panel() -> None:
        app_state["steps"].clear()
        notify("Cleared recorded pipeline steps.")

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
        path: str = "output.nii.gz",
        use_file_affine: bool = True,
        force_type: str = "",
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
    layers_layout = QVBoxLayout()
    layers_layout.addWidget(orientation_label)
    layers_layout.addWidget(layer_list)
    layers_layout.addWidget(layers_panel.native)
    layers_layout.addWidget(layers_refresh_panel.native)
    layers_tab.setLayout(layers_layout)

    dock = QWidget()
    layout = QVBoxLayout()
    tabs = QTabWidget()
    tabs.addTab(tools_widget, "Tools")
    tabs.addTab(mesh_panel.native, "Mesh")
    tabs.addTab(batch_panel.native, "Batch")
    tabs.addTab(layers_tab, "Layers")
    tabs.addTab(save_panel.native, "Export")
    tabs.addTab(export_panel.native, "Pipeline")
    tabs.addTab(pipeline_panel.native, "Pipeline log")
    layout.addWidget(tabs)
    dock.setLayout(layout)
    viewer.window.add_dock_widget(dock, area="right", name="nvitk")
    log_dock = build_log_dock_widget()
    viewer.window.add_dock_widget(log_dock, area="bottom", name="nvitk log")

    _refresh_layer_list(layer_list, viewer, app_state)

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
