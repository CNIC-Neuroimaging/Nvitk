"""Anatomical orientation marker: a human figure that rotates with labeled axis arrows.

The plain RGB axes widget tells you which array axis is which, but not how the
subject is lying in the scene. This module builds a marker that does both: three
double-headed arrows (one per array axis, keeping the X=red / Y=green / Z=blue
convention) carrying the anatomical letter of each end -- ``R``/``L``, ``A``/``P``
and ``H``/``F`` (head / feet) -- plus a small human figure posed in the same
frame, so the viewer can read off the subject's pose at a glance.

Everything is built in RAS anatomical space and mapped into the array frame with
a single signed-permutation matrix derived from the image orientation codes. For
data stored in a left-handed frame (e.g. ``LPS``) that matrix is a reflection,
which is exactly right: the rendered scene *is* mirrored, and so is the figure.

The whole marker is a single :class:`pyvista.PolyData` with per-cell RGB colors
wrapped in one actor. Text is real geometry (:func:`pyvista.Text3D`), not caption
or billboard actors, so the marker survives PyVista's HTML scene export the same
way ordinary meshes do.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Anatomical direction of each orientation code, in RAS world axes.
_CODE_TO_RAS: dict[str, tuple[float, float, float]] = {
    "R": (1.0, 0.0, 0.0),
    "L": (-1.0, 0.0, 0.0),
    "A": (0.0, 1.0, 0.0),
    "P": (0.0, -1.0, 0.0),
    "S": (0.0, 0.0, 1.0),
    "I": (0.0, 0.0, -1.0),
}

_OPPOSITE_CODE: dict[str, str] = {
    "R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S",
}

# Letter drawn on each arrow end. Superior/inferior read as Head/Feet, which is
# how the PESA readers describe the scanner axis.
_CODE_TEXT: dict[str, str] = {
    "R": "R", "L": "L", "A": "A", "P": "P", "S": "H", "I": "F",
}

# Per array axis (x, y, z), matching the legacy RGB axes widget.
_AXIS_RGB: tuple[tuple[int, int, int], ...] = (
    (222, 60, 60),
    (60, 175, 95),
    (70, 120, 225),
)

_BODY_RGB: tuple[int, int, int] = (208, 214, 226)
_FRONT_RGB: tuple[int, int, int] = (240, 150, 90)

_ARROW_HALF_LENGTH = 0.92
_FIGURE_SCALE = 1.2
_LABEL_HEIGHT = 0.26
_LABEL_OFFSET = 1.18
_LABEL_RGB: tuple[int, int, int] = (250, 250, 252)
# (right, up, thickness) half-extents of the letter plaque at each arrow end.
_PLAQUE_HALF_SIZE = (0.19, 0.17, 0.028)


def normalize_axcodes(axcodes: Any) -> str | None:
    """Validate 3-letter orientation codes (``\"RAS\"``, ``\"LPS\"``, ...); ``None`` if unusable."""
    if not isinstance(axcodes, str):
        return None
    codes = axcodes.strip().upper()
    if len(codes) != 3 or any(c not in _CODE_TO_RAS for c in codes):
        return None
    axes_seen = {c if c in "RL" else ("AP" if c in "AP" else "SI") for c in codes}
    if len(axes_seen) != 3:
        return None
    return codes


def axcodes_from_image(img: Any) -> str | None:
    """Orientation codes for *img* from its ``orientation`` attribute or its affine."""
    codes = normalize_axcodes(getattr(img, "orientation", None))
    if codes is not None:
        return codes
    affine = getattr(img, "affine", None)
    if affine is None:
        metadata = getattr(img, "metadata", None) or {}
        affine = metadata.get("affine")
    if affine is None:
        return None
    from nvitk.io._common import orientation_codes_from_affine

    return normalize_axcodes(orientation_codes_from_affine(np.asarray(affine)))


def anatomical_axis_labels(axcodes: str) -> tuple[tuple[str, str], ...]:
    """``((pos, neg), ...)`` letters per array axis, e.g. ``RAS`` -> ``(("R","L"), ("A","P"), ("H","F"))``."""
    codes = normalize_axcodes(axcodes)
    if codes is None:
        raise ValueError(f"Invalid orientation codes: {axcodes!r}")
    return tuple(
        (_CODE_TEXT[c], _CODE_TEXT[_OPPOSITE_CODE[c]]) for c in codes
    )


def _ras_to_array_matrix(axcodes: str) -> np.ndarray:
    """3x3 signed permutation mapping an RAS point to the array frame of *axcodes*."""
    return np.asarray([_CODE_TO_RAS[c] for c in axcodes], dtype=float)


def _colored(mesh: Any, rgb: tuple[int, int, int]) -> Any:
    """Tag every cell of *mesh* with a flat RGB color (uint8), in place."""
    colors = np.tile(np.asarray(rgb, dtype=np.uint8), (mesh.n_cells, 1))
    mesh.cell_data["marker_rgb"] = colors
    return mesh


def _scale_axis(mesh: Any, axis: int, factor: float) -> Any:
    """Squash *mesh* along one axis (used to flatten the trunk front-to-back)."""
    pts = np.asarray(mesh.points, dtype=float)
    pts[:, axis] *= float(factor)
    mesh.points = pts
    return mesh


def _human_figure_parts(pv: Any) -> list[tuple[Any, tuple[int, int, int]]]:
    """Stylised standing figure in RAS (+x right, +y anterior, +z superior), ~0.9 tall."""
    head = pv.Sphere(radius=0.085, center=(0.0, 0.0, 0.355), theta_resolution=18, phi_resolution=18)
    nose = pv.Cone(center=(0.0, 0.095, 0.355), direction=(0.0, 1.0, 0.0), height=0.075, radius=0.032, resolution=12)
    neck = pv.Cylinder(center=(0.0, 0.0, 0.275), direction=(0.0, 0.0, 1.0), radius=0.032, height=0.07, resolution=14)
    trunk = _scale_axis(
        pv.Cylinder(center=(0.0, 0.0, 0.105), direction=(0.0, 0.0, 1.0), radius=0.115, height=0.30, resolution=22),
        1,
        0.62,
    )
    pelvis = _scale_axis(
        pv.Cylinder(center=(0.0, 0.0, -0.065), direction=(0.0, 0.0, 1.0), radius=0.10, height=0.09, resolution=22),
        1,
        0.62,
    )
    parts: list[tuple[Any, tuple[int, int, int]]] = [
        (head, _BODY_RGB),
        (nose, _FRONT_RGB),
        (neck, _BODY_RGB),
        (trunk, _BODY_RGB),
        (pelvis, _BODY_RGB),
    ]
    for side in (-1.0, 1.0):
        parts.append(
            (
                pv.Cylinder(
                    center=(side * 0.152, 0.0, 0.09),
                    direction=(0.0, 0.0, 1.0),
                    radius=0.031,
                    height=0.33,
                    resolution=14,
                ),
                _BODY_RGB,
            )
        )
        parts.append(
            (
                pv.Cylinder(
                    center=(side * 0.057, 0.0, -0.25),
                    direction=(0.0, 0.0, 1.0),
                    radius=0.042,
                    height=0.34,
                    resolution=14,
                ),
                _BODY_RGB,
            )
        )
        parts.append(
            (
                pv.Cylinder(
                    center=(side * 0.057, 0.045, -0.435),
                    direction=(0.0, 1.0, 0.0),
                    radius=0.036,
                    height=0.10,
                    resolution=12,
                ),
                _FRONT_RGB,
            )
        )
    for mesh, _rgb in parts:
        mesh.points = np.asarray(mesh.points, dtype=float) * _FIGURE_SCALE
    return parts


def _arrow_parts(pv: Any, axcodes: str) -> list[tuple[Any, tuple[int, int, int]]]:
    """Double-headed arrow per array axis, laid out in RAS."""
    parts: list[tuple[Any, tuple[int, int, int]]] = []
    for axis, code in enumerate(axcodes):
        rgb = _AXIS_RGB[axis]
        direction = np.asarray(_CODE_TO_RAS[code], dtype=float)
        for sign in (1.0, -1.0):
            parts.append(
                (
                    pv.Arrow(
                        start=(0.0, 0.0, 0.0),
                        direction=tuple(sign * direction),
                        tip_length=0.22,
                        tip_radius=0.055,
                        shaft_radius=0.017,
                        scale=_ARROW_HALF_LENGTH,
                    ),
                    rgb,
                )
            )
    return parts


def _label_parts(pv: Any, axcodes: str, matrix: np.ndarray) -> list[tuple[Any, tuple[int, int, int]]]:
    """End letters for every arrow, built straight in the array frame.

    Each end carries a small plaque in the arrow's color with the letter written
    on both faces in white. The plaque is opaque, so whichever face is turned
    away stays hidden and the letter never shows up mirrored -- a backwards ``R``
    reads as a bug, not as anatomy, even though the scene around it genuinely is
    mirrored for left-handed frames.

    Plaques stand upright on the head-feet axis and face outward along their own
    arrow, except the head/feet pair, which would be edge-on under that rule and
    faces anterior instead.
    """
    up = matrix @ np.asarray((0.0, 0.0, 1.0))
    anterior = matrix @ np.asarray((0.0, 1.0, 0.0))
    half_w, half_h, half_t = _PLAQUE_HALF_SIZE
    parts: list[tuple[Any, tuple[int, int, int]]] = []
    for axis, code in enumerate(axcodes):
        rgb = _AXIS_RGB[axis]
        direction = matrix @ np.asarray(_CODE_TO_RAS[code], dtype=float)
        vertical = code in ("S", "I")
        for sign in (1.0, -1.0):
            end_code = code if sign > 0 else _OPPOSITE_CODE[code]
            position = sign * direction * _LABEL_OFFSET
            face = anterior if vertical else sign * direction

            plaque = pv.Box(bounds=(-half_w, half_w, -half_h, half_h, -half_t, half_t))
            rot = np.column_stack((np.cross(up, face), up, face))
            plaque.points = np.asarray(plaque.points, dtype=float) @ rot.T + position
            parts.append((plaque, rgb))

            for outward in (face, -face):
                rot = np.column_stack((np.cross(up, outward), up, outward))
                glyph = pv.Text3D(
                    _CODE_TEXT[end_code],
                    height=_LABEL_HEIGHT,
                    depth=_LABEL_HEIGHT * 0.08,
                    center=(0.0, 0.0, 0.0),
                )
                glyph.points = (
                    np.asarray(glyph.points, dtype=float) @ rot.T
                    + position
                    + outward * (half_t + _LABEL_HEIGHT * 0.03)
                )
                parts.append((glyph, _LABEL_RGB))
    return parts


def build_orientation_marker_mesh(axcodes: str, *, show_figure: bool = True) -> Any:
    """Merged marker mesh for *axcodes*, in array-index space, with ``marker_rgb`` cell colors."""
    import pyvista as pv

    codes = normalize_axcodes(axcodes)
    if codes is None:
        raise ValueError(f"Invalid orientation codes: {axcodes!r}")

    matrix = _ras_to_array_matrix(codes)
    mirrored = float(np.linalg.det(matrix)) < 0

    ras_parts = _arrow_parts(pv, codes)
    if show_figure:
        ras_parts.extend(_human_figure_parts(pv))

    meshes = []
    for mesh, rgb in ras_parts:
        mesh.points = np.asarray(mesh.points, dtype=float) @ matrix.T
        if mirrored:
            # The reflection inverts face winding; recompute so the figure is lit
            # from outside instead of rendering nearly black.
            mesh = mesh.compute_normals(
                consistent_normals=False,
                auto_orient_normals=False,
                flip_normals=True,
            )
        meshes.append(_colored(mesh, rgb))

    for mesh, rgb in _label_parts(pv, codes, matrix):
        meshes.append(_colored(mesh, rgb))

    return meshes[0].merge(meshes[1:], merge_points=False) if len(meshes) > 1 else meshes[0]


def orientation_marker_actor(axcodes: str, *, show_figure: bool = True) -> Any:
    """Single actor rendering the marker mesh with its baked-in RGB cell colors."""
    import pyvista as pv

    mesh = build_orientation_marker_mesh(axcodes, show_figure=show_figure)
    # Make the RGB array the active cell scalars: vtk.js resolves colors through
    # the active array when the exported HTML replays this mapper.
    mesh.set_active_scalars("marker_rgb", preference="cell")
    mapper = pv.DataSetMapper(mesh)
    mapper.set_scalars(
        mesh.cell_data["marker_rgb"],
        "marker_rgb",
        rgb=True,
        preference="cell",
    )
    actor = pv.Actor(mapper=mapper)
    actor.prop.ambient = 0.35
    actor.prop.diffuse = 0.75
    actor.prop.specular = 0.1
    return actor


def add_anatomical_orientation_widget(
    pl: Any,
    axcodes: str,
    *,
    interactive: bool = False,
    viewport: Sequence[float] = (0.0, 0.0, 0.27, 0.27),
    show_figure: bool = True,
) -> bool:
    """Attach the human-figure orientation marker to *pl*; ``False`` if it couldn't be built."""
    codes = normalize_axcodes(axcodes)
    if codes is None:
        return False
    try:
        actor = orientation_marker_actor(codes, show_figure=show_figure)
        pl.add_orientation_widget(actor, interactive=interactive, viewport=tuple(viewport))
    except Exception:
        return False
    return True


__all__ = [
    "add_anatomical_orientation_widget",
    "anatomical_axis_labels",
    "axcodes_from_image",
    "build_orientation_marker_mesh",
    "normalize_axcodes",
    "orientation_marker_actor",
]
