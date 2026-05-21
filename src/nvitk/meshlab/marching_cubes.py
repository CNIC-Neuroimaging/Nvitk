"""Spacing- and affine-aware marching cubes for binary and multilabel masks."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import BackendUnavailableError


def _measure():
    try:
        from skimage import measure
    except ImportError as exc:
        raise BackendUnavailableError(
            'scikit-image is required for marching cubes. Install with "pip install scikit-image".'
        ) from exc
    return measure
from nvitk.types import Image, Mesh


def _world_vertices(
    verts_voxel: np.ndarray,
    *,
    affine: np.ndarray | None,
    spacing: Sequence[float] | None,
    origin: Sequence[float] | None,
) -> np.ndarray:
    """Map voxel-index vertices to physical/world coordinates."""
    v = np.asarray(verts_voxel, dtype=np.float64)
    if affine is not None:
        aff = np.asarray(affine, dtype=np.float64)
        if aff.shape == (4, 4):
            hom = np.column_stack([v, np.ones(len(v))])
            return (aff @ hom.T).T[:, :3]
    if spacing is not None:
        sp = np.asarray(spacing[:3], dtype=np.float64)
        off = np.zeros(3, dtype=np.float64)
        if origin is not None:
            off = np.asarray(origin[:3], dtype=np.float64)
        return v * sp + off
    return v


def _metadata_from_image(
    image: Image,
    *,
    label_id: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    meta = dict(image.metadata or {})
    if label_id is not None:
        meta["label_id"] = label_id
    if name:
        meta["name"] = name
    if image.affine is not None:
        meta["affine"] = to_numpy(image.affine)
    if image.spacing is not None:
        meta["spacing"] = tuple(float(x) for x in image.spacing[:3])
    return meta


def marching_cubes_binary(
    mask: Image | Any,
    *,
    level: float = 0.5,
    step_size: int = 1,
    world_space: bool = True,
) -> Mesh | None:
    """Extract a single surface from a binary mask."""
    if isinstance(mask, Image):
        data = to_numpy(mask.data)
        meta = _metadata_from_image(mask, name="binary")
        affine = mask.affine
        spacing = mask.spacing
        origin = meta.get("origin")
    else:
        data = to_numpy(mask)
        meta = {}
        affine = None
        spacing = None
        origin = None

    if not np.any(data > 0):
        return None

    try:
        verts, faces, _normals, _vals = _measure().marching_cubes(
            data.astype(np.float32),
            level=level,
            step_size=step_size,
            allow_degenerate=False,
        )
    except (ValueError, RuntimeError):
        return None

    if world_space:
        verts_out = _world_vertices(
            verts,
            affine=to_numpy(affine) if affine is not None else None,
            spacing=spacing,
            origin=origin,
        )
    else:
        verts_out = verts
    return Mesh(vertices=verts_out, faces=faces.astype(np.int32), metadata=meta)


def marching_cubes_multilabel(
    labels: Image | Any,
    *,
    label_ids: Iterable[int] | None = None,
    level: float = 0.5,
    step_size: int = 1,
    world_space: bool = True,
) -> list[Mesh]:
    """One mesh per nonzero label id."""
    if isinstance(labels, Image):
        data = to_numpy(labels.data)
        base_meta = labels.metadata or {}
        affine = labels.affine
        spacing = labels.spacing
    else:
        data = to_numpy(labels)
        base_meta = {}
        affine = None
        spacing = None

    ids = list(label_ids) if label_ids is not None else sorted(int(x) for x in np.unique(data) if x != 0)
    meshes: list[Mesh] = []
    for lid in ids:
        binary = (data == lid).astype(np.float32)
        if not np.any(binary):
            continue
        try:
            verts, faces, _, _ = _measure().marching_cubes(
                binary,
                level=level,
                step_size=step_size,
                allow_degenerate=False,
            )
        except (ValueError, RuntimeError):
            continue
        meta = dict(base_meta)
        meta["label_id"] = int(lid)
        meta["name"] = f"label_{lid}"
        if isinstance(labels, Image):
            meta.update(_metadata_from_image(labels, label_id=lid, name=meta["name"]))
        origin = meta.get("origin")
        if world_space:
            verts_out = _world_vertices(
                verts,
                affine=to_numpy(affine) if affine is not None else None,
                spacing=spacing,
                origin=origin,
            )
        else:
            verts_out = verts
        meshes.append(Mesh(vertices=verts_out, faces=faces.astype(np.int32), metadata=meta))
    return meshes


def mesh_from_image(
    image: Image | Any,
    *,
    multilabel: bool | None = None,
    label_ids: Iterable[int] | None = None,
    world_space: bool = True,
) -> Mesh | list[Mesh] | None:
    """Convenience: binary (nonzero) or multilabel reconstruction."""
    if isinstance(image, Image):
        data = to_numpy(image.data)
    else:
        data = to_numpy(image)

    unique = np.unique(data)
    is_multi = multilabel if multilabel is not None else len(unique) > 2 or (len(unique) == 2 and 0 not in unique)

    if is_multi:
        return marching_cubes_multilabel(image, label_ids=label_ids, world_space=world_space)
    return marching_cubes_binary(image, world_space=world_space)
