"""Mesh operations in the tool browser: native sections, MeshLab and VMTK as suite badges.

The Meshlab panel's catalogue used to list nvitk's own categories, then a second set of
"MeshLab · ..." groups. Here every operation sits in one task section (Clean & repair, Remesh,
Smooth...), whoever implements it, and the suite shows as a badge: ``VMTK`` for the vmtk
filters, ``MeshLab`` for PyMeshLab's.
"""

from __future__ import annotations

import re
from typing import Any

from nvitk.gui.tools.taxonomy import Section, ToolMeta

#: Native category -> (section key, icon, hue).
_NATIVE: dict[str, tuple[str, str, str]] = {
    "Create": ("create", "plus", "green"),
    "Edit (click on the surface)": ("edit", "brush", "orange"),
    "Select (on the surface)": ("select", "target", "purple"),
    "Clean & repair": ("clean", "tools", "blue"),
    "Smooth": ("smooth", "wave", "cyan"),
    "Remesh": ("remesh", "mesh", "olive"),
    "Transform & align": ("transform", "move", "brown"),
    "Measure": ("measure", "ruler", "red"),
    "Vessels & tubes": ("vessels", "vessel", "red"),
    "Compare": ("compare", "masks", "pink"),
    "Convert (image ↔ mesh ↔ points)": ("convert", "convert", "grey"),
    "Point cloud": ("points", "grid", "cyan"),
    "Time (3D+t)": ("time", "clock", "pink"),
}
#: MeshLab group (after "MeshLab · ") -> (section key, subsection).
_MESHLAB: dict[str, tuple[str, str]] = {
    "Create": ("create", ""),
    "Cleaning & repair": ("clean", ""),
    "Remeshing & simplification": ("remesh", ""),
    "Smoothing & deformation": ("smooth", ""),
    "Reconstruction & point clouds": ("points", "Reconstruction"),
    "Sampling": ("points", "Sampling"),
    "Curvature, normals & quality": ("measure", "Curvature & quality"),
    "Measures & distances": ("measure", "Distances"),
    "Booleans & pieces": ("booleans", ""),
    "Selection": ("select", ""),
    "Transform & align": ("transform", ""),
}

SECTIONS: tuple[Section, ...] = (
    *(Section(key, title, icon, hue) for title, (key, icon, hue) in _NATIVE.items()),
    Section("booleans", "Booleans & pieces", "masks", "green"),
    Section("other", "Other", "more", "grey"),
)


def mesh_meta(op: Any) -> ToolMeta:
    """Section, subsection and suite of a :class:`~nvitk.gui.mesh.operations.MeshOp`."""
    from nvitk.gui.mesh.operations import MESHLAB_PREFIX

    category = str(getattr(op, "category", ""))
    op_id = str(getattr(op, "id", ""))
    if op_id.startswith(MESHLAB_PREFIX) or category.startswith("MeshLab"):
        group = re.sub(r"^MeshLab\s*·\s*", "", category)
        section, sub = _MESHLAB.get(group, ("other", group))
        return ToolMeta(section, sub, suite="MeshLab")
    section = _NATIVE.get(category, ("other", "", ""))[0]
    suite = "VMTK" if op_id.startswith("vmtk") else "nvitk"
    return ToolMeta(section, "", suite=suite)


def mesh_help(op: Any) -> str:
    """Hover text of a mesh operation."""
    return str(getattr(op, "description", "") or "")


__all__ = ["SECTIONS", "mesh_help", "mesh_meta"]
