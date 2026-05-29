"""Export Napari layers to disk via :mod:`nvitk.io` writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.io import imsave
from nvitk.io._common import guess_write_type
from nvitk.types import Image, Mesh

from nvitk.gui.spatial import layer_affine, layer_to_image as spatial_layer_to_image
from nvitk.gui.tool_runner import notify


def layer_to_image(layer: Any, *, use_file_affine: bool = True) -> Image:
    """Build an :class:`~nvitk.types.Image` from a Napari layer (Image / Labels)."""
    img = spatial_layer_to_image(layer)
    if use_file_affine:
        meta = dict(img.metadata or {})
        src = meta.get("affine_source")
        if src is not None:
            meta["affine"] = np.asarray(src, dtype=float)
            return Image(data=img.data, metadata=meta, axes=img.axes, name=img.name)
    return img


def layer_to_mesh(layer: Any) -> Mesh:
    """Build a :class:`~nvitk.types.Mesh` from a Napari Surface layer."""
    data = getattr(layer, "data", None)
    if data is None:
        raise ValueError("Surface layer has no data.")
    verts, faces = data[0], data[1]
    meta = dict(getattr(layer, "metadata", None) or {})
    aff = layer_affine(layer)
    if aff is not None:
        meta["affine"] = aff
    return Mesh(
        vertices=to_numpy(verts),
        faces=to_numpy(faces),
        metadata=meta,
    )


def write_mesh_stl(path: str | Path, mesh: Mesh) -> None:
    """Write a mesh to ASCII STL (no extra dependencies)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    name = mesh.name.replace(" ", "_")[:80] or "mesh"
    lines = [f"solid {name}"]
    for tri in faces:
        p0, p1, p2 = verts[int(tri[0])], verts[int(tri[1])], verts[int(tri[2])]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm > 0:
            normal = normal / norm
        else:
            normal = np.array([0.0, 0.0, 0.0])
        lines.append(f"  facet normal {normal[0]} {normal[1]} {normal[2]}")
        lines.append("    outer loop")
        for p in (p0, p1, p2):
            lines.append(f"      vertex {p[0]} {p[1]} {p[2]}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_layer(
    layer: Any,
    path: str | Path,
    *,
    use_file_affine = True,
    force_type = None,
) -> None:
    """Export the active Napari layer using nvitk I/O."""
    path = Path(path)
    layer_type = layer.__class__.__name__

    if layer_type in ("Image", "Labels"):
        img = layer_to_image(layer, use_file_affine=use_file_affine)
        imsave(path, img, force_type=force_type)
        return

    if layer_type == "Surface":
        mesh = layer_to_mesh(layer)
        suffix = path.suffix.lower()
        if force_type == "stl" or suffix == ".stl":
            write_mesh_stl(path, mesh)
            return
        raise ValueError(
            f"Surface export supports .stl (got {path.suffix!r}). "
            "Use a .stl path or force_type='stl'."
        )

    raise ValueError(f"Cannot export layer type {layer_type!r} with nvitk I/O.")


def export_selected_layer(
    viewer: Any,
    path: str,
    *,
    use_file_affine = True,
    force_type = None,
) -> None:
    """Export the selected layer, or the topmost layer if none selected."""
    if not viewer.layers:
        notify("No layers to export.", error=True)
        return
    layer = viewer.layers.selection.active or viewer.layers[-1]
    try:
        if force_type is None and path:
            suffix = Path(path).suffix.lower()
            if suffix == ".stl":
                force_type = "stl"
            else:
                guess_write_type(path)
        export_layer(
            layer,
            path,
            use_file_affine=use_file_affine,
            force_type=force_type,
        )
    except Exception as exc:
        notify(f"Export failed: {exc}", error=True)
        return
    notify(f"Exported {layer.name} → {path}")
