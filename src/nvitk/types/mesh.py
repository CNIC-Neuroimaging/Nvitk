"""Surface types: :class:`Mesh` (triangles), :class:`PointCloud`, and :class:`MeshSeries` (3D+t).

Coordinates are physical (mm, world) whenever they come from an image with an
affine — :func:`nvitk.meshlab.mesh_from_image` produces world-space meshes by
default — and voxel indices otherwise; ``metadata["space"]`` says which
(``"world"`` / ``"voxel"``) when known.

Per-element arrays travel with the geometry: ``point_data`` holds one value (or
vector) per vertex / point — curvature, a distance map, normals — and
``cell_data`` one per face. Operations that keep the vertices keep them; ones
that rebuild the mesh drop them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

from nvitk.core.array import to_numpy


def _validated(arrays: dict[str, Any] | None, n: int, what: str) -> dict[str, np.ndarray]:
    """Per-element arrays as NumPy, each with *n* rows."""
    out: dict[str, np.ndarray] = {}
    for key, value in (arrays or {}).items():
        arr = np.asarray(to_numpy(value))
        if arr.shape[:1] != (n,):
            raise ValueError(f"{what} {key!r} has {arr.shape[:1]} rows; expected {n}.")
        out[str(key)] = arr
    return out


@dataclass
class Mesh:
    """Triangle mesh with per-vertex / per-face data and imaging metadata."""

    vertices: np.ndarray
    faces: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce vertices/faces to ``(N, 3)`` float64 / ``(M, 3)`` int32 and validate."""
        self.vertices = np.asarray(to_numpy(self.vertices), dtype=np.float64)
        self.faces = np.asarray(to_numpy(self.faces), dtype=np.int32)
        if self.faces.size == 0:
            self.faces = self.faces.reshape(0, 3)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3); got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3); got {self.faces.shape}")
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError("faces reference vertices that do not exist.")
        self.metadata = dict(self.metadata or {})
        self.point_data = _validated(self.point_data, len(self.vertices), "point_data")
        self.cell_data = _validated(self.cell_data, len(self.faces), "cell_data")

    # -- sizes and geometry ----------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bounds(self) -> np.ndarray:
        """``(2, 3)`` min / max corner (zeros for an empty mesh)."""
        if not self.n_vertices:
            return np.zeros((2, 3))
        return np.stack([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def triangles(self) -> np.ndarray:
        """``(M, 3, 3)`` corner coordinates of every face."""
        return self.vertices[self.faces]

    @property
    def face_normals(self) -> np.ndarray:
        """Unit normal per face (zero for a degenerate face), from the winding."""
        tri = self.triangles
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)

    @property
    def face_areas(self) -> np.ndarray:
        tri = self.triangles
        return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    @property
    def vertex_normals(self) -> np.ndarray:
        """Area-weighted unit normal per vertex."""
        tri = self.triangles
        weighted = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        acc = np.zeros_like(self.vertices)
        for k in range(3):
            np.add.at(acc, self.faces[:, k], weighted)
        norm = np.linalg.norm(acc, axis=1, keepdims=True)
        return np.divide(acc, norm, out=np.zeros_like(acc), where=norm > 0)

    # -- metadata -----------------------------------------------------------------

    @property
    def affine(self) -> np.ndarray | None:
        """4x4 world affine of the source image (float), or ``None`` if unset."""
        aff = self.metadata.get("affine")
        return np.asarray(aff, dtype=float) if aff is not None else None

    @property
    def spacing(self) -> tuple[float, float, float] | None:
        """Source-image voxel spacing in mm (first 3 axes), or ``None``."""
        sp = self.metadata.get("spacing")
        if sp is None:
            return None
        return tuple(float(x) for x in sp[:3])

    @property
    def label_id(self) -> int | None:
        """Label id this mesh was extracted from, if it came from a segmentation."""
        lid = self.metadata.get("label_id")
        return int(lid) if lid is not None else None

    @property
    def name(self) -> str:
        """Human-readable mesh name (defaults to ``\"mesh\"``)."""
        return str(self.metadata.get("name", "mesh"))

    # -- copies -----------------------------------------------------------------

    def copy(self) -> Mesh:
        """Deep copy (arrays and metadata dict)."""
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            metadata=dict(self.metadata),
            point_data={k: v.copy() for k, v in self.point_data.items()},
            cell_data={k: v.copy() for k, v in self.cell_data.items()},
        )

    def with_vertices(self, vertices: np.ndarray) -> Mesh:
        """Same topology and data, moved vertices (smoothing, transforms)."""
        return Mesh(
            vertices=vertices,
            faces=self.faces.copy(),
            metadata=dict(self.metadata),
            point_data=dict(self.point_data),
            cell_data=dict(self.cell_data),
        )

    def with_point_data(self, **arrays: Any) -> Mesh:
        """A copy carrying extra per-vertex arrays."""
        out = self.with_vertices(self.vertices)
        out.point_data.update(_validated(arrays, self.n_vertices, "point_data"))
        return out

    # -- interop ------------------------------------------------------------------

    def to_napari_surface(self) -> dict[str, Any]:
        """Dict suitable for ``napari.layers.Surface``."""
        return {
            "vertices": to_numpy(self.vertices),
            "faces": to_numpy(self.faces),
        }

    @classmethod
    def from_arrays(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Mesh:
        """Build a :class:`Mesh` from raw vertex/face arrays and optional metadata."""
        return cls(vertices=vertices, faces=faces, metadata=dict(metadata or {}))

    def __repr__(self) -> str:
        extra = f", point_data={list(self.point_data)}" if self.point_data else ""
        return f"Mesh({self.name!r}, {self.n_vertices} vertices, {self.n_faces} faces{extra})"


@dataclass
class PointCloud:
    """Unstructured points with per-point data (normals, intensities, labels…)."""

    points: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    point_data: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(to_numpy(self.points), dtype=np.float64)
        if self.points.size == 0:
            self.points = self.points.reshape(0, 3)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3); got {self.points.shape}")
        self.metadata = dict(self.metadata or {})
        self.point_data = _validated(self.point_data, len(self.points), "point_data")

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def bounds(self) -> np.ndarray:
        if not self.n_points:
            return np.zeros((2, 3))
        return np.stack([self.points.min(axis=0), self.points.max(axis=0)])

    @property
    def normals(self) -> np.ndarray | None:
        """``point_data["normals"]`` when present."""
        return self.point_data.get("normals")

    @property
    def name(self) -> str:
        return str(self.metadata.get("name", "points"))

    def copy(self) -> PointCloud:
        return PointCloud(
            points=self.points.copy(),
            metadata=dict(self.metadata),
            point_data={k: v.copy() for k, v in self.point_data.items()},
        )

    def subset(self, index: Any) -> PointCloud:
        """The points (and their data) selected by a boolean mask or index array."""
        return PointCloud(
            points=self.points[index],
            metadata=dict(self.metadata),
            point_data={k: v[index] for k, v in self.point_data.items()},
        )

    def __repr__(self) -> str:
        return f"PointCloud({self.name!r}, {self.n_points} points)"


@dataclass
class MeshSeries:
    """A mesh over time (3D+t): one :class:`Mesh` per frame, with frame times.

    Frames may share one topology (a tracked surface: same faces, moved
    vertices — what displacement, temporal smoothing and frame interpolation
    need) or each have their own (marching cubes per frame).
    """

    frames: list[Mesh]
    times: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frames = list(self.frames)
        if not self.frames:
            raise ValueError("A MeshSeries needs at least one frame.")
        if self.times is None:
            step = float(self.metadata.get("t_res", 1.0) or 1.0)
            self.times = np.arange(len(self.frames), dtype=float) * step
        self.times = np.asarray(self.times, dtype=float)
        if self.times.shape != (len(self.frames),):
            raise ValueError("times needs one value per frame.")
        self.metadata = dict(self.metadata or {})

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Mesh]:
        return iter(self.frames)

    def __getitem__(self, index: int) -> Mesh:
        return self.frames[index]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def shared_topology(self) -> bool:
        """True when every frame has the same faces (vertex correspondence over time)."""
        first = self.frames[0].faces
        return all(f.faces.shape == first.shape and np.array_equal(f.faces, first) for f in self.frames[1:])

    @property
    def name(self) -> str:
        return str(self.metadata.get("name", self.frames[0].name))

    def vertex_stack(self) -> np.ndarray:
        """``(T, N, 3)`` vertex positions; needs a shared topology."""
        if not self.shared_topology:
            raise ValueError("Frames do not share a topology (different faces per frame).")
        return np.stack([f.vertices for f in self.frames])

    @classmethod
    def from_vertex_stack(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        times: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MeshSeries:
        """A shared-topology series from ``(T, N, 3)`` vertex positions and one face array."""
        meta = dict(metadata or {})
        frames = [Mesh(vertices=v, faces=faces, metadata=dict(meta)) for v in np.asarray(vertices)]
        return cls(frames=frames, times=None if times is None else np.asarray(times, float), metadata=meta)

    def __repr__(self) -> str:
        shared = "shared topology" if self.shared_topology else "per-frame topology"
        return f"MeshSeries({self.name!r}, {self.n_frames} frames, {shared})"
