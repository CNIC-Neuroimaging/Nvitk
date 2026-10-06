"""Mesh clean-up and repair: duplicates, degenerate faces, components, holes, normals."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.meshlab.convert import from_pyvista, require_pyvista, to_pyvista
from nvitk.meshlab.topology import face_components
from nvitk.types import Mesh


def _rebuild(mesh: Mesh, vertices: np.ndarray, faces: np.ndarray, vertex_index: np.ndarray | None,
             face_index: np.ndarray | None) -> Mesh:
    """A mesh on new arrays, carrying per-element data through the index maps."""
    point_data = {} if vertex_index is None else {k: v[vertex_index] for k, v in mesh.point_data.items()}
    cell_data = {} if face_index is None else {k: v[face_index] for k, v in mesh.cell_data.items()}
    return Mesh(vertices=vertices, faces=faces, metadata=dict(mesh.metadata),
                point_data=point_data, cell_data=cell_data)


def merge_vertices(mesh: Mesh, tolerance: float = 1e-6) -> Mesh:
    """Weld vertices closer than *tolerance* (duplicates from per-face exports, STL)."""
    if not mesh.n_vertices:
        return mesh.copy()
    if tolerance > 0:
        keys = np.round(mesh.vertices / tolerance).astype(np.int64)
    else:
        keys = mesh.vertices
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    faces = inverse[mesh.faces]
    return _rebuild(mesh, mesh.vertices[first], faces, first, np.arange(mesh.n_faces))


def remove_unreferenced_vertices(mesh: Mesh) -> Mesh:
    """Drop vertices no face uses."""
    used = np.unique(mesh.faces)
    if len(used) == mesh.n_vertices:
        return mesh.copy()
    remap = np.full(mesh.n_vertices, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return _rebuild(mesh, mesh.vertices[used], remap[mesh.faces], used, np.arange(mesh.n_faces))


def remove_degenerate_faces(mesh: Mesh, area_tolerance: float = 0.0) -> Mesh:
    """Drop faces with a repeated vertex or an area at or below *area_tolerance*."""
    f = mesh.faces
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    if mesh.n_faces:
        keep &= mesh.face_areas > float(area_tolerance)
    idx = np.flatnonzero(keep)
    return _rebuild(mesh, mesh.vertices, f[idx], np.arange(mesh.n_vertices), idx)


def remove_duplicate_faces(mesh: Mesh) -> Mesh:
    """Drop faces using the same three vertices as an earlier one (any winding)."""
    if not mesh.n_faces:
        return mesh.copy()
    _, first = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_index=True)
    idx = np.sort(first)
    return _rebuild(mesh, mesh.vertices, mesh.faces[idx], np.arange(mesh.n_vertices), idx)


def clean(mesh: Mesh, *, tolerance: float = 1e-6, area_tolerance: float = 0.0) -> Mesh:
    """Merge duplicate vertices, then drop degenerate / duplicate faces and unused vertices."""
    out = merge_vertices(mesh, tolerance)
    out = remove_degenerate_faces(out, area_tolerance)
    out = remove_duplicate_faces(out)
    return remove_unreferenced_vertices(out)


def keep_components(
    mesh: Mesh,
    *,
    largest: int = 0,
    min_faces: int = 0,
    min_area: float = 0.0,
) -> Mesh:
    """Keep connected pieces: the *largest* N by area, and/or those above a size.

    ``largest=0`` keeps every piece passing the *min_faces* / *min_area* filters.
    """
    n, labels = face_components(mesh)
    if n <= 1 and not (min_faces or min_area):
        return mesh.copy()
    areas = np.bincount(labels, weights=mesh.face_areas, minlength=n)
    counts = np.bincount(labels, minlength=n)
    keep = (counts >= int(min_faces)) & (areas >= float(min_area))
    if largest and largest > 0:
        order = np.argsort(-areas)
        top = np.zeros(n, dtype=bool)
        top[order[: int(largest)]] = True
        keep &= top
    idx = np.flatnonzero(keep[labels])
    out = _rebuild(mesh, mesh.vertices, mesh.faces[idx], np.arange(mesh.n_vertices), idx)
    return remove_unreferenced_vertices(out)


def split_components(mesh: Mesh) -> list[Mesh]:
    """One mesh per connected piece, largest (by area) first."""
    n, labels = face_components(mesh)
    areas = np.bincount(labels, weights=mesh.face_areas, minlength=n)
    pieces = []
    for comp in np.argsort(-areas):
        idx = np.flatnonzero(labels == comp)
        part = _rebuild(mesh, mesh.vertices, mesh.faces[idx], np.arange(mesh.n_vertices), idx)
        pieces.append(remove_unreferenced_vertices(part))
    return pieces


def fill_holes(mesh: Mesh, max_hole_size: float = 1e9) -> Mesh:
    """Cap every hole whose rim fits within *max_hole_size* (mesh units) of its centre.

    Each qualifying boundary loop gets a vertex at its centroid and a fan of
    triangles wound against the rim's existing faces, so a mesh with only small
    holes comes out closed and consistently oriented. Per-vertex data is kept
    for the original vertices (new centre vertices get the loop's mean).
    """
    from nvitk.meshlab.topology import boundary_loops, edges

    loops = boundary_loops(mesh)
    if not loops:
        return mesh.copy()
    half = {(int(a), int(b)) for a, b in edges(mesh)}
    verts = [mesh.vertices]
    faces = [mesh.faces]
    point_data = {k: [v] for k, v in mesh.point_data.items()}
    n = mesh.n_vertices
    for loop in loops:
        if len(loop) < 3:
            continue
        ring = mesh.vertices[loop]
        centre = ring.mean(axis=0)
        if np.linalg.norm(ring - centre, axis=1).max() > float(max_hole_size):
            continue
        nxt = np.roll(loop, -1)
        # The rim's own faces use a→b; the cap must use b→a to wind consistently.
        reverse = (int(loop[0]), int(loop[1])) in half
        tris = np.stack([nxt, loop, np.full(len(loop), n)], axis=1) if reverse else \
            np.stack([loop, nxt, np.full(len(loop), n)], axis=1)
        verts.append(centre[None])
        faces.append(tris)
        for key, arr in mesh.point_data.items():
            point_data[key].append(np.asarray(arr)[loop].mean(axis=0, keepdims=True).astype(arr.dtype))
        n += 1
    return Mesh(
        vertices=np.vstack(verts),
        faces=np.vstack(faces),
        metadata=dict(mesh.metadata),
        point_data={k: np.concatenate(v) for k, v in point_data.items()},
    )


def ensure_outward(mesh: Mesh) -> Mesh:
    """Flip all faces if the (closed) surface's normals point inwards (cheap: one sign)."""
    if not mesh.n_faces:
        return mesh
    tri = mesh.triangles
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        return flip_normals(mesh)
    return mesh


def flip_normals(mesh: Mesh) -> Mesh:
    """Reverse every face's winding (inside ↔ outside)."""
    return _rebuild(mesh, mesh.vertices, mesh.faces[:, ::-1].copy(), np.arange(mesh.n_vertices),
                    np.arange(mesh.n_faces))


def orient_faces(mesh: Mesh, *, outward: bool = True) -> Mesh:
    """Make neighbouring faces wind consistently, pointing out of closed pieces.

    VTK's consistency pass, then each closed piece is flipped if its signed
    volume says its normals point inwards.
    """
    require_pyvista()
    poly = to_pyvista(mesh)
    fixed = poly.compute_normals(
        consistent_normals=True, auto_orient_normals=bool(outward), split_vertices=False,
        cell_normals=False, point_normals=False,
    )
    out = from_pyvista(fixed, metadata=mesh.metadata, keep_data=False)
    if out.n_vertices == mesh.n_vertices:
        out.point_data.update(mesh.point_data)
    if outward:
        from nvitk.meshlab.measure import signed_volume

        if signed_volume(out) < 0:
            out = flip_normals(out)
    return out


def remove_small_components(mesh: Mesh, min_faces: int = 10, min_area: float = 0.0) -> Mesh:
    """Shorthand for :func:`keep_components` with a size floor."""
    return keep_components(mesh, min_faces=min_faces, min_area=min_area)


def mesh_summary_counts(mesh: Mesh) -> dict[str, Any]:
    """Quick defect counts (for a 'check mesh' report)."""
    f = mesh.faces
    degenerate = int(((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])).sum()) if len(f) else 0
    dup = int(mesh.n_faces - len(np.unique(np.sort(f, axis=1), axis=0))) if len(f) else 0
    unused = int(mesh.n_vertices - len(np.unique(f))) if len(f) else mesh.n_vertices
    return {"degenerate_faces": degenerate, "duplicate_faces": dup, "unreferenced_vertices": unused}


__all__ = [
    "clean",
    "ensure_outward",
    "fill_holes",
    "flip_normals",
    "keep_components",
    "merge_vertices",
    "mesh_summary_counts",
    "orient_faces",
    "remove_degenerate_faces",
    "remove_duplicate_faces",
    "remove_small_components",
    "remove_unreferenced_vertices",
    "split_components",
]
