"""The :class:`Mesh` container: vertices, faces, and imaging metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy


@dataclass
class Mesh:
    """Triangle mesh with optional world-space metadata from a source image."""

    vertices: np.ndarray
    faces: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (N, 3); got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (M, 3); got {self.faces.shape}")

    @property
    def affine(self) -> np.ndarray | None:
        aff = self.metadata.get("affine")
        return np.asarray(aff, dtype=float) if aff is not None else None

    @property
    def spacing(self) -> tuple[float, float, float] | None:
        sp = self.metadata.get("spacing")
        if sp is None:
            return None
        return tuple(float(x) for x in sp[:3])

    @property
    def label_id(self) -> int | None:
        lid = self.metadata.get("label_id")
        return int(lid) if lid is not None else None

    @property
    def name(self) -> str:
        return str(self.metadata.get("name", "mesh"))

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
        return cls(vertices=vertices, faces=faces, metadata=dict(metadata or {}))
