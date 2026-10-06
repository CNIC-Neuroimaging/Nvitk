"""Reading and writing meshes, point clouds and mesh series.

STL (ASCII / binary), OBJ, OFF and XYZ/CSV point lists are read and written
here with NumPy alone; PLY and the VTK formats (``.vtk``, ``.vtp``, ``.vtu``)
go through PyVista. :func:`read_surface` returns a :class:`Mesh` when the file
has faces and a :class:`PointCloud` when it has only points.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nvitk.types import Mesh, MeshSeries, PointCloud

#: Extensions :func:`read_surface` understands (lower case, with the dot).
MESH_EXTENSIONS: tuple[str, ...] = (".stl", ".obj", ".off", ".ply", ".vtk", ".vtp", ".vtu", ".gii")
POINT_EXTENSIONS: tuple[str, ...] = (".xyz", ".pts", ".csv", ".txt", ".pcd")


def _suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".surf.gii"):
        return ".gii"
    return path.suffix.lower()


# ──────────────────────────────────────────────────────────────────────────────
# STL
# ──────────────────────────────────────────────────────────────────────────────


def _read_stl(path: Path) -> Mesh:
    raw = path.read_bytes()
    is_ascii = raw[:5].lower() == b"solid" and b"facet" in raw[:2048]
    if is_ascii:
        nums = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", raw)
        tri = np.asarray(nums, dtype=float).reshape(-1, 3, 3)
    else:
        n = struct.unpack("<I", raw[80:84])[0]
        rec = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                            count=n, offset=84)
        tri = rec["v"].astype(float)
    verts = tri.reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    from nvitk.meshlab.cleaning import merge_vertices

    # STL stores every triangle's corners separately: weld them back.
    return merge_vertices(Mesh(vertices=verts, faces=faces, metadata={"name": path.stem}), tolerance=1e-7)


def _write_stl(path: Path, mesh: Mesh, *, binary: bool = True) -> None:
    tri = mesh.triangles.astype(np.float32)
    normals = mesh.face_normals.astype(np.float32)
    if binary:
        rec = np.zeros(mesh.n_faces, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        rec["n"] = normals
        rec["v"] = tri
        header = (f"nvitk {mesh.name}"[:80]).encode("ascii", "replace").ljust(80, b" ")
        path.write_bytes(header + struct.pack("<I", mesh.n_faces) + rec.tobytes())
        return
    name = re.sub(r"\s+", "_", mesh.name)[:80] or "mesh"
    lines = [f"solid {name}"]
    for nrm, t in zip(normals, tri):
        lines.append(f"  facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}")
        lines.append("    outer loop")
        lines.extend(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}" for p in t)
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ──────────────────────────────────────────────────────────────────────────────
# OBJ / OFF
# ──────────────────────────────────────────────────────────────────────────────


def _read_obj(path: Path) -> Mesh | PointCloud:
    verts, faces = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            faces.extend([idx[0], idx[j], idx[j + 1]] for j in range(1, len(idx) - 1))
    v = np.asarray(verts, dtype=float).reshape(-1, 3)
    if not faces:
        return PointCloud(points=v, metadata={"name": path.stem})
    return Mesh(vertices=v, faces=np.asarray(faces), metadata={"name": path.stem})


def _write_obj(path: Path, mesh: Mesh) -> None:
    lines = [f"# nvitk {mesh.name}"]
    lines.extend(f"v {p[0]:.8g} {p[1]:.8g} {p[2]:.8g}" for p in mesh.vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_off(path: Path) -> Mesh | PointCloud:
    tokens = [t for line in path.read_text().splitlines() for t in line.split("#")[0].split()]
    if tokens and tokens[0].upper().endswith("OFF"):
        tokens = tokens[1:]
    nv, nf = int(tokens[0]), int(tokens[1])
    pos = 3
    v = np.asarray(tokens[pos: pos + 3 * nv], dtype=float).reshape(nv, 3)
    pos += 3 * nv
    faces = []
    for _ in range(nf):
        k = int(tokens[pos])
        idx = [int(x) for x in tokens[pos + 1: pos + 1 + k]]
        faces.extend([idx[0], idx[j], idx[j + 1]] for j in range(1, k - 1))
        pos += 1 + k
    if not faces:
        return PointCloud(points=v, metadata={"name": path.stem})
    return Mesh(vertices=v, faces=np.asarray(faces), metadata={"name": path.stem})


def _write_off(path: Path, mesh: Mesh) -> None:
    lines = ["OFF", f"{mesh.n_vertices} {mesh.n_faces} 0"]
    lines.extend(f"{p[0]:.8g} {p[1]:.8g} {p[2]:.8g}" for p in mesh.vertices)
    lines.extend(f"3 {a} {b} {c}" for a, b, c in mesh.faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ──────────────────────────────────────────────────────────────────────────────
# Point lists
# ──────────────────────────────────────────────────────────────────────────────


def _read_points_text(path: Path) -> PointCloud:
    """XYZ / CSV / TXT: x y z [extra columns], optional header row."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    delim = "," if path.suffix.lower() == ".csv" or (text and text[0].count(",") >= 2) else None
    header: list[str] | None = None
    rows = []
    for line in text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in (line.split(delim) if delim else line.split())]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            if header is None and not rows:
                header = parts
                continue
            raise
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path.name}: expected at least three numeric columns (x, y, z).")
    cols = [c.lower() for c in header] if header else []
    xyz = [cols.index(c) for c in ("x", "y", "z")] if all(c in cols for c in ("x", "y", "z")) else [0, 1, 2]
    data = {}
    for j in range(arr.shape[1]):
        if j in xyz:
            continue
        name = cols[j] if cols and j < len(cols) else f"col{j}"
        data[name] = arr[:, j]
    if all(k in data for k in ("nx", "ny", "nz")):
        data["normals"] = np.stack([data.pop("nx"), data.pop("ny"), data.pop("nz")], axis=1)
    return PointCloud(points=arr[:, xyz], metadata={"name": path.stem}, point_data=data)


def _read_pcd(path: Path) -> PointCloud:
    """ASCII PCD (Point Cloud Library)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    fields: list[str] = []
    start = 0
    for i, line in enumerate(lines):
        if line.upper().startswith("FIELDS"):
            fields = line.split()[1:]
        if line.upper().startswith("DATA"):
            if "ascii" not in line.lower():
                raise ValueError("Only ASCII PCD files are supported (DATA ascii).")
            start = i + 1
            break
    arr = np.loadtxt(lines[start:], ndmin=2)
    idx = [fields.index(c) for c in ("x", "y", "z")]
    data = {f: arr[:, j] for j, f in enumerate(fields) if f not in ("x", "y", "z")}
    return PointCloud(points=arr[:, idx], metadata={"name": path.stem}, point_data=data)


def _write_points_text(path: Path, cloud: PointCloud) -> None:
    cols = ["x", "y", "z"]
    arrays = [cloud.points]
    for key, arr in cloud.point_data.items():
        a = np.asarray(arr, dtype=float).reshape(cloud.n_points, -1)
        if key == "normals" and a.shape[1] == 3:
            cols += ["nx", "ny", "nz"]
        else:
            cols += [key] if a.shape[1] == 1 else [f"{key}_{i}" for i in range(a.shape[1])]
        arrays.append(a)
    table = np.hstack(arrays)
    sep = "," if path.suffix.lower() == ".csv" else " "
    np.savetxt(path, table, delimiter=sep, header=sep.join(cols), comments="", fmt="%.8g")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def read_surface(path: str | Path) -> Mesh | PointCloud:
    """Read a mesh or point cloud; the type follows the file's content."""
    path = Path(path)
    ext = _suffix(path)
    if ext == ".stl":
        return _read_stl(path)
    if ext == ".obj":
        return _read_obj(path)
    if ext == ".off":
        return _read_off(path)
    if ext in (".xyz", ".pts", ".csv", ".txt"):
        return _read_points_text(path)
    if ext == ".pcd":
        return _read_pcd(path)
    if ext == ".gii":
        import nibabel as nib

        gii = nib.load(str(path))
        coords = gii.agg_data("NIFTI_INTENT_POINTSET")
        tris = gii.agg_data("NIFTI_INTENT_TRIANGLE")
        return Mesh(vertices=coords, faces=tris, metadata={"name": path.name.split(".")[0], "space": "world"})
    from nvitk.meshlab.convert import from_pyvista, require_pyvista

    pv = require_pyvista()
    data = pv.read(str(path))
    poly = data if isinstance(data, pv.PolyData) else data.extract_geometry()
    if poly.n_cells == 0 or np.asarray(poly.faces).size == 0:
        cloud = PointCloud(points=np.asarray(poly.points), metadata={"name": path.stem})
        for key in poly.point_data.keys():
            arr = np.asarray(poly.point_data[key])
            if arr.shape[:1] == (cloud.n_points,):
                cloud.point_data[key] = arr
        return cloud
    return from_pyvista(poly, metadata={"name": path.stem})


def read_mesh(path: str | Path) -> Mesh:
    """Read a file that must hold a triangle mesh."""
    out = read_surface(path)
    if not isinstance(out, Mesh):
        raise ValueError(f"{Path(path).name} holds points only, no faces.")
    return out


def read_point_cloud(path: str | Path) -> PointCloud:
    """Read points (a mesh file gives its vertices)."""
    out = read_surface(path)
    if isinstance(out, Mesh):
        from nvitk.meshlab.convert import mesh_to_point_cloud

        return mesh_to_point_cloud(out)
    return out


def write_surface(path: str | Path, obj: Mesh | PointCloud, *, binary: bool = True) -> Path:
    """Write a mesh (STL, OBJ, OFF, PLY, VTK, VTP) or a point cloud (XYZ, CSV, PLY, VTP)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = _suffix(path)
    if isinstance(obj, PointCloud):
        if ext in (".xyz", ".pts", ".csv", ".txt"):
            _write_points_text(path, obj)
            return path
        from nvitk.meshlab.convert import to_pyvista

        to_pyvista(obj).save(str(path), binary=binary)
        return path
    if ext == ".stl":
        _write_stl(path, obj, binary=binary)
    elif ext == ".obj":
        _write_obj(path, obj)
    elif ext == ".off":
        _write_off(path, obj)
    elif ext in (".xyz", ".csv", ".txt", ".pts"):
        from nvitk.meshlab.convert import mesh_to_point_cloud

        _write_points_text(path, mesh_to_point_cloud(obj))
    elif ext == ".gii":
        import nibabel as nib

        arrays = [
            nib.gifti.GiftiDataArray(obj.vertices.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(obj.faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"),
        ]
        nib.save(nib.gifti.GiftiImage(darrays=arrays), str(path))
    else:
        from nvitk.meshlab.convert import to_pyvista

        to_pyvista(obj).save(str(path), binary=binary)
    return path


def read_mesh_series(paths: Sequence[str | Path], *, t_res: float = 1.0) -> MeshSeries:
    """One frame per file, in natural (numeric-aware) file-name order."""
    def _natural(p: Path) -> list[Any]:
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]

    files = sorted((Path(p) for p in paths), key=_natural)
    if not files:
        raise ValueError("No files given.")
    frames = [read_mesh(p) for p in files]
    name = re.sub(r"[_\-.]?\d+$", "", files[0].stem) or files[0].stem
    return MeshSeries(frames=frames, metadata={"name": name, "t_res": float(t_res)})


def write_mesh_series(series: MeshSeries, directory: str | Path, *, ext: str = ".vtp") -> list[Path]:
    """Write one file per frame (``name_000.vtp``…) plus a ParaView ``.pvd`` index."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ext = ext if ext.startswith(".") else f".{ext}"
    stem = re.sub(r"[^\w\-]+", "_", series.name) or "series"
    width = max(3, len(str(len(series) - 1)))
    paths = []
    for k, mesh in enumerate(series):
        paths.append(write_surface(directory / f"{stem}_{k:0{width}d}{ext}", mesh))
    entries = "\n".join(
        f'    <DataSet timestep="{float(t):.6g}" part="0" file="{p.name}"/>' for t, p in zip(series.times, paths)
    )
    (directory / f"{stem}.pvd").write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n'
        f"{entries}\n  </Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
    return paths


def read_pvd(path: str | Path) -> MeshSeries:
    """A ParaView ``.pvd`` time collection as a :class:`MeshSeries`."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    items = re.findall(r'<DataSet[^>]*timestep="([^"]+)"[^>]*file="([^"]+)"', text)
    if not items:
        items = [(m[1], m[0]) for m in re.findall(r'<DataSet[^>]*file="([^"]+)"[^>]*timestep="([^"]+)"', text)]
    items.sort(key=lambda it: float(it[0]))
    frames = [read_mesh(path.parent / f) for _, f in items]
    return MeshSeries(frames=frames, times=np.asarray([float(t) for t, _ in items]), metadata={"name": path.stem})


__all__ = [
    "MESH_EXTENSIONS",
    "POINT_EXTENSIONS",
    "read_mesh",
    "read_mesh_series",
    "read_point_cloud",
    "read_pvd",
    "read_surface",
    "write_mesh_series",
    "write_surface",
]
