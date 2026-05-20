"""Napari application shell for the nvitk GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.cli.catalog import build_catalog_tree, parse_pyproject_scripts
from nvitk.core.backend import using
from nvitk.io import imread
from nvitk.meshlab import mesh_from_image, marching_cubes_multilabel
from nvitk.types import Image, Mesh


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


def _run_bilateral(layer_data: np.ndarray, *, use_gpu: bool) -> np.ndarray:
    from nvitk.restoration import bilateral

    img = Image(data=layer_data, metadata={}, axes="XYZ")
    bk = "cupy" if use_gpu else "numpy"
    with using(bk):
        out = bilateral(img)
    return np.asarray(out.data if isinstance(out, Image) else out)


def _run_tool_on_layer(
    tool_name: str,
    layer,
    *,
    use_gpu: bool,
    label: int | None,
) -> np.ndarray | None:
    data = np.asarray(layer.data)
    if label is not None and hasattr(layer, "data") and data.dtype in (np.int32, np.int64, np.uint8, np.uint16):
        data = (data == label).astype(np.float32)

    if "bilateral" in tool_name.lower():
        return _run_bilateral(data, use_gpu=use_gpu)
    return None


def run_app() -> None:
    import napari
    from magicgui import magicgui
    from qtpy.QtWidgets import QFileDialog, QListWidget, QTabWidget, QVBoxLayout, QWidget

    viewer = napari.Viewer(title="nvitk")
    tool_names = _catalog_tool_names()
    layer_registry: dict[str, dict[str, Any]] = {"inputs": [], "outputs": [], "meshes": []}

    @magicgui(
        tool={"choices": tool_names, "label": "Tool"},
        use_gpu={"label": "GPU backend"},
        label_id={"label": "Label (masks)", "value": 0},
        overlay_mode={"choices": ["add_layer", "replace_active"], "label": "Output mode"},
        call_button="Run tool",
    )
    def tool_panel(
        tool: str,
        use_gpu: bool,
        label_id: int,
        overlay_mode: str,
    ) -> None:
        if not viewer.layers:
            return
        layer = viewer.layers.selection.active or viewer.layers[-1]
        result = _run_tool_on_layer(tool, layer, use_gpu=use_gpu, label=label_id or None)
        if result is None:
            return
        name = f"{layer.name}_out"
        if overlay_mode == "replace_active":
            try:
                layer.data = result
            except Exception:
                viewer.add_image(result, name=name)
        else:
            viewer.add_image(result, name=name)
        layer_registry["outputs"].append({"name": name, "shape": result.shape})

    @magicgui(call_button="Reconstruct mesh")
    def mesh_panel(multilabel: bool = False) -> None:
        if not viewer.layers:
            return
        layer = viewer.layers.selection.active or viewer.layers[-1]
        img = Image(data=np.asarray(layer.data), metadata=getattr(layer, "metadata", {}) or {})
        if multilabel:
            meshes = marching_cubes_multilabel(img) or []
        else:
            m = mesh_from_image(img, multilabel=False)
            meshes = [m] if m is not None else []
        for mesh in meshes:
            if not isinstance(mesh, Mesh):
                continue
            surf = mesh.to_napari_surface()
            viewer.add_surface(
                surf["vertices"],
                faces=surf["faces"],
                name=mesh.name,
            )
            layer_registry["meshes"].append({"name": mesh.name})

    @magicgui(
        paths={"label": "Paths (one per line)"},
        call_button="Run batch queue",
    )
    def batch_panel(paths: str = "") -> None:
        for line in paths.strip().splitlines():
            p = Path(line.strip())
            if not p.exists():
                continue
            img = imread(p, backend="numpy")
            arr = np.asarray(img.data)
            viewer.add_image(arr, name=p.stem)
            layer_registry["inputs"].append({"path": str(p), "name": p.stem})

    @magicgui(call_button="Export pipeline JSON")
    def export_panel(path: str = "pipeline.json") -> None:
        doc = {
            "inputs": layer_registry["inputs"],
            "outputs": layer_registry["outputs"],
            "meshes": layer_registry["meshes"],
            "layers": [layer.name for layer in viewer.layers],
        }
        out = Path(path)
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    def _open_files() -> None:
        dlg = QFileDialog()
        paths, _ = dlg.getOpenFileNames(
            None,
            "Open image",
            "",
            "Images (*.nii *.nii.gz *.mha *.tif *.tiff);;All (*)",
        )
        for p in paths:
            img = imread(p, backend="numpy")
            viewer.add_image(np.asarray(img.data), name=Path(p).stem)
            layer_registry["inputs"].append({"path": p, "name": Path(p).stem})

    dock = QWidget()
    layout = QVBoxLayout()
    tabs = QTabWidget()
    tabs.addTab(tool_panel.native, "Tools")
    tabs.addTab(mesh_panel.native, "Mesh")
    tabs.addTab(batch_panel.native, "Batch")
    tabs.addTab(export_panel.native, "Pipeline")
    layer_list = QListWidget()
    layer_list.addItem("--- Inputs ---")
    tabs.addTab(layer_list, "Layers")
    layout.addWidget(tabs)
    dock.setLayout(layout)
    viewer.window.add_dock_widget(dock, area="right", name="nvitk")

    @viewer.bind_key("Ctrl+O")
    def _open_key(_viewer) -> None:
        _open_files()

    napari.run()
