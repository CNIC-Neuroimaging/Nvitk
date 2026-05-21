"""Napari application shell for the nvitk GUI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.cli.catalog import build_catalog_tree, parse_pyproject_scripts
from nvitk.meshlab import mesh_from_image, marching_cubes_multilabel
from nvitk.types import Image, Mesh

from nvitk.gui.io_napari import install_nvitk_io, open_paths_with_nvitk
from nvitk.gui.tool_runner import notify, parse_label_ids, run_tool_on_layer


def _catalog_tool_names() -> list[str]:
    roots = build_catalog_tree(parse_pyproject_scripts())
    names: list[str] = []

    def walk(node) -> None:
        for t in node.tools:
            if t.command:
                names.append(t.command)
            else:
                names.append(t.display_label)
        for c in node.children:
            walk(c)

    for r in roots:
        walk(r)
    return names


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
    affine = getattr(layer, "affine", None)
    if affine is not None:
        kwargs["affine"] = np.asarray(affine, dtype=float)
    elif getattr(layer, "scale", None) is not None:
        kwargs["scale"] = layer.scale
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
    import napari
    from magicgui import magicgui
    from qtpy.QtWidgets import QFileDialog, QListWidget, QTabWidget, QVBoxLayout, QWidget

    viewer = napari.Viewer(title="nvitk")
    _ = viewer.window
    install_nvitk_io(viewer)
    tool_names = _catalog_tool_names()

    app_state: dict[str, Any] = {
        "record_enabled": True,
        "steps": [],
        "inputs": [],
        "outputs": [],
        "meshes": [],
    }

    @magicgui(
        tool={"choices": tool_names, "label": "Tool"},
        use_gpu={"label": "GPU backend"},
        target_mode={
            "choices": ["raw", "binary_mask", "label", "all_labels"],
            "label": "Process target",
            "value": "raw",
        },
        label_ids={"label": "Label id(s) (comma-separated)", "value": ""},
        overlay_mode={"choices": ["add_layer", "replace_active"], "label": "Output mode"},
        call_button="Run tool",
    )
    def tool_panel(
        tool: str,
        use_gpu: bool,
        target_mode: str,
        label_ids: str,
        overlay_mode: str,
    ) -> None:
        if not viewer.layers:
            notify("No layers loaded. Open an image first (Ctrl+O or drag-and-drop).", error=True)
            return
        layer = viewer.layers.selection.active or viewer.layers[-1]
        ids = parse_label_ids(label_ids)
        if target_mode == "label" and not ids:
            notify("Label mode requires one or more label ids (e.g. 1 or 1,2,3).", error=True)
            return
        try:
            result = run_tool_on_layer(
                tool,
                layer,
                use_gpu=use_gpu,
                target_mode=target_mode,
                label_ids=ids or None,
            )
        except NotImplementedError as exc:
            notify(str(exc), error=True)
            return
        except Exception as exc:
            notify(f"Tool failed: {exc}", error=True)
            return

        name = f"{layer.name}_{tool.split()[0]}"
        out_kwargs = _layer_display_kwargs(layer, name=name)
        can_replace = (
            overlay_mode == "replace_active"
            and tuple(result.shape) == tuple(layer.data.shape)
        )
        if can_replace:
            try:
                layer.data = result
                layer.name = name
            except Exception:
                viewer.add_image(result, **out_kwargs)
        else:
            viewer.add_image(result, **out_kwargs)

        app_state["outputs"].append({"name": name, "shape": tuple(result.shape)})
        _record_step(
            app_state,
            {
                "type": "tool",
                "tool": tool,
                "source_layer": layer.name,
                "output_layer": name,
                "use_gpu": use_gpu,
                "target_mode": target_mode,
                "label_ids": ids,
                "overlay_mode": overlay_mode,
            },
        )
        notify(f"Applied {tool} → {name}")
        _refresh_layer_list(layer_list, viewer, app_state)

    @magicgui(call_button="Reconstruct mesh")
    def mesh_panel(multilabel: bool = False) -> None:
        if not viewer.layers:
            notify("No layers loaded.", error=True)
            return
        layer = viewer.layers.selection.active or viewer.layers[-1]
        img = Image(data=np.asarray(layer.data), metadata=getattr(layer, "metadata", {}) or {})
        try:
            if multilabel:
                meshes = marching_cubes_multilabel(img) or []
            else:
                m = mesh_from_image(img, multilabel=False)
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
            viewer.add_surface(
                (surf["vertices"], surf["faces"]),
                name=mesh.name,
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
        call_button="Refresh layer list",
    )
    def layers_panel(record_steps: bool = True) -> None:
        app_state["record_enabled"] = record_steps
        _refresh_layer_list(layer_list, viewer, app_state)

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

    dock = QWidget()
    layout = QVBoxLayout()
    tabs = QTabWidget()
    tabs.addTab(tool_panel.native, "Tools")
    tabs.addTab(mesh_panel.native, "Mesh")
    tabs.addTab(batch_panel.native, "Batch")
    layer_list = QListWidget()
    tabs.addTab(layers_panel.native, "Layers")
    tabs.addTab(export_panel.native, "Pipeline")
    tabs.addTab(pipeline_panel.native, "Pipeline log")
    layout.addWidget(tabs)
    dock.setLayout(layout)
    viewer.window.add_dock_widget(dock, area="right", name="nvitk")

    _refresh_layer_list(layer_list, viewer, app_state)

    @viewer.bind_key("Ctrl+O")
    def _open_key(_viewer) -> None:
        _open_files()

    napari.run()
