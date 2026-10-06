"""Point clouds: sampling, downsampling, outlier removal, normals, surface reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.meshlab.convert import from_pyvista, require_pyvista, to_pyvista
from nvitk.types import Image, Mesh, PointCloud

#: Surface reconstruction methods of :func:`reconstruct_surface`, most robust first.
RECONSTRUCTION_METHODS: tuple[str, ...] = (
    "poisson", "screened_poisson", "ball_pivoting", "imls", "implicit", "alpha_shape", "convex_hull",
)

#: How :func:`estimate_normals` orients the normals it fits.
NORMAL_ORIENTATIONS: tuple[str, ...] = ("propagate", "outward", "none")


def sample_surface(mesh: Mesh, n_points: int = 10000, *, seed: int = 0, with_normals: bool = True) -> PointCloud:
    """*n_points* uniformly distributed over the surface (area-weighted)."""
    if not mesh.n_faces:
        raise ValueError("The mesh has no faces to sample.")
    rng = np.random.default_rng(seed)
    areas = mesh.face_areas
    prob = areas / areas.sum()
    face = rng.choice(mesh.n_faces, size=int(n_points), p=prob)
    r1 = np.sqrt(rng.random(int(n_points)))
    r2 = rng.random(int(n_points))
    tri = mesh.triangles[face]
    pts = (1 - r1)[:, None] * tri[:, 0] + (r1 * (1 - r2))[:, None] * tri[:, 1] + (r1 * r2)[:, None] * tri[:, 2]
    data = {"normals": mesh.face_normals[face]} if with_normals else {}
    return PointCloud(points=pts, metadata=dict(mesh.metadata), point_data=data)


def points_from_mask(mask: Image, *, label_ids: Any = None, max_points: int = 0, seed: int = 0) -> PointCloud:
    """Voxel centres of a mask (in world mm when it has an affine), one point each.

    The label id of each voxel is kept as ``point_data["label"]``; *max_points*
    subsamples at random (0 = all).
    """
    data = to_numpy(mask.data)
    if data.ndim != 3:
        raise ValueError("points_from_mask needs a 3D mask.")
    sel = data != 0 if label_ids is None else np.isin(data, list(label_ids))
    ijk = np.argwhere(sel)
    labels = data[sel].astype(np.int64)
    if max_points and len(ijk) > max_points:
        keep = np.random.default_rng(seed).choice(len(ijk), int(max_points), replace=False)
        ijk, labels = ijk[keep], labels[keep]
    aff = mask.affine
    if aff is not None:
        a = np.asarray(to_numpy(aff), dtype=float)
        pts = ijk @ a[:3, :3].T + a[:3, 3]
        space = "world"
    else:
        pts = ijk.astype(float)
        space = "voxel"
    meta = {"space": space, "name": f"{mask.name or 'mask'}_points"}
    return PointCloud(points=pts, metadata=meta, point_data={"label": labels})


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    """One point per occupied voxel of *voxel_size*: the centroid of its points
    (point data averaged; integer data takes the first point's value)."""
    if voxel_size <= 0 or not cloud.n_points:
        return cloud.copy()
    keys = np.floor(cloud.points / float(voxel_size)).astype(np.int64)
    _, first, inverse, counts = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n = len(counts)

    def _mean(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(len(values), -1).astype(float)
        acc = np.zeros((n, flat.shape[1]))
        np.add.at(acc, inverse, flat)
        return (acc / counts[:, None]).reshape((n,) + values.shape[1:])

    data = {}
    for key, arr in cloud.point_data.items():
        if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
            data[key] = arr[first]
        else:
            data[key] = _mean(arr)
    if "normals" in data:
        nrm = data["normals"]
        norm = np.linalg.norm(nrm, axis=1, keepdims=True)
        data["normals"] = np.divide(nrm, norm, out=np.zeros_like(nrm), where=norm > 0)
    return PointCloud(points=_mean(cloud.points), metadata=dict(cloud.metadata), point_data=data)


def random_downsample(cloud: PointCloud, n_points: int = 0, *, fraction: float = 0.0, seed: int = 0) -> PointCloud:
    """Keep *n_points* (or *fraction* of the points) chosen at random."""
    n = int(n_points) if n_points else int(round(cloud.n_points * float(fraction)))
    if n <= 0 or n >= cloud.n_points:
        return cloud.copy()
    keep = np.sort(np.random.default_rng(seed).choice(cloud.n_points, n, replace=False))
    return cloud.subset(keep)


def statistical_outliers(cloud: PointCloud, *, k: int = 16, std_ratio: float = 2.0) -> np.ndarray:
    """Boolean mask of outliers: mean distance to the *k* nearest neighbours above
    the cloud's mean by more than *std_ratio* standard deviations."""
    from scipy.spatial import cKDTree

    if cloud.n_points <= k:
        return np.zeros(cloud.n_points, dtype=bool)
    d, _ = cKDTree(cloud.points).query(cloud.points, k=int(k) + 1)
    mean_d = d[:, 1:].mean(axis=1)
    return mean_d > mean_d.mean() + float(std_ratio) * mean_d.std()


def radius_outliers(cloud: PointCloud, *, radius: float = 1.0, min_neighbours: int = 4) -> np.ndarray:
    """Boolean mask of outliers: fewer than *min_neighbours* other points within *radius*."""
    from scipy.spatial import cKDTree

    tree = cKDTree(cloud.points)
    counts = np.asarray([len(n) - 1 for n in tree.query_ball_point(cloud.points, float(radius))])
    return counts < int(min_neighbours)


def remove_outliers(cloud: PointCloud, method: str = "statistical", **kwargs: Any) -> PointCloud:
    """The cloud without the points :func:`statistical_outliers` / :func:`radius_outliers` flag."""
    flags = statistical_outliers(cloud, **kwargs) if method == "statistical" else radius_outliers(cloud, **kwargs)
    return cloud.subset(~flags)


def _pca_normals(pts: np.ndarray, k: int) -> np.ndarray:
    """Unoriented normals: smallest principal direction of each k-neighbourhood."""
    from scipy.spatial import cKDTree

    k = max(3, min(int(k), len(pts)))
    _, idx = cKDTree(pts).query(pts, k=k)
    nbrs = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nbrs, nbrs)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


def orient_normals(points: np.ndarray, normals: np.ndarray, *, k: int = 12) -> np.ndarray:
    """Make normals consistent by propagating along a minimum spanning tree (Hoppe 1992).

    Neighbours whose normals are nearly parallel are linked most cheaply, so the
    orientation travels along the surface rather than jumping across thin parts.
    Each connected piece is seeded at its topmost point (normal up) and finally
    flipped as a whole if most of its normals point towards its centre — outward
    for closed shapes, whatever their form.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=float)
    out = np.asarray(normals, dtype=float).copy()
    n = len(pts)
    if n < 3:
        return out
    k = max(2, min(int(k), n - 1))
    _, idx = cKDTree(pts).query(pts, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    weight = 1.0 - np.abs(np.einsum("ij,ij->i", out[rows], out[cols])) + 1e-6
    graph = sparse.coo_matrix((weight, (rows, cols)), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)
    tree = minimum_spanning_tree(graph)
    tree = tree + tree.T
    n_comp, comp = connected_components(tree, directed=False)
    for c in range(n_comp):
        members = np.flatnonzero(comp == c)
        seed = members[np.argmax(pts[members, 2])]
        if out[seed, 2] < 0:
            out[seed] *= -1
        order, pred = breadth_first_order(tree, seed, directed=False, return_predecessors=True)
        for i in order[1:]:
            if np.dot(out[i], out[pred[i]]) < 0:
                out[i] *= -1
        centre = pts[members].mean(axis=0)
        if np.einsum("ij,ij->i", out[members], pts[members] - centre).mean() < 0:
            out[members] *= -1
    return out


def estimate_normals(
    cloud: PointCloud,
    *,
    k: int = 16,
    orient: str = "propagate",
    orient_outward: bool | None = None,
) -> PointCloud:
    """Per-point normals by PCA of the *k* nearest neighbours.

    *orient*: ``"propagate"`` (consistent along the surface, :func:`orient_normals`
    — right for any shape), ``"outward"`` (away from the centroid: only for
    star-shaped clouds) or ``"none"``. *orient_outward* is the older boolean form.
    """
    if orient_outward is not None:
        orient = "outward" if orient_outward else "none"
    pts = cloud.points
    normals = _pca_normals(pts, k)
    if orient == "propagate":
        normals = orient_normals(pts, normals, k=max(6, int(k) // 2))
    elif orient == "outward":
        flip = np.einsum("ij,ij->i", normals, pts - pts.mean(axis=0)) < 0
        normals[flip] *= -1
    out = cloud.copy()
    out.point_data["normals"] = normals
    return out


def _oriented_normals(cloud: PointCloud, k: int, reuse: bool = True) -> np.ndarray:
    """The cloud's normals when it has usable ones, else estimated and propagated."""
    normals = cloud.normals if reuse else None
    if normals is not None and np.asarray(normals).shape == cloud.points.shape:
        nrm = np.asarray(normals, dtype=float)
        length = np.linalg.norm(nrm, axis=1, keepdims=True)
        if np.all(length > 1e-9):
            return nrm / length
    return estimate_normals(cloud, k=k, orient="propagate").point_data["normals"]


def sample_spacing(points: np.ndarray) -> float:
    """Median distance from each point to its nearest neighbour."""
    from scipy.spatial import cKDTree

    if len(points) < 2:
        return 1.0
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1])) or 1.0


#: :func:`sample_spacing` under a name that functions taking a ``sample_spacing`` argument can still reach.
_point_spacing = sample_spacing


def _grid_for(points: np.ndarray, resolution: int, pad: float) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """``(origin, step, shape)`` of an isotropic grid around the points.

    *resolution* is the number of steps along the longest side; ``0`` picks it
    from the sampling density (a step of ~0.75 sample spacings, 48–256 steps).
    """
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    if not resolution or resolution <= 0:
        resolution = int(np.clip(span.max() / (0.75 * sample_spacing(points)), 48, 256))
    step = float(span.max()) / max(int(resolution) - 1, 8)
    margin = max(3, int(np.ceil(pad * span.max() / step)))
    origin = lo - margin * step
    shape = tuple(int(np.ceil(s / step)) + 2 * margin + 1 for s in span)
    return origin, step, shape  # type: ignore[return-value]


def _trim_far(mesh: Mesh, points: np.ndarray, max_distance: float) -> Mesh:
    """Drop faces whose vertices all lie farther than *max_distance* from every sample."""
    from scipy.spatial import cKDTree

    from nvitk.meshlab.cleaning import remove_unreferenced_vertices

    if max_distance <= 0 or not mesh.n_faces:
        return mesh
    d, _ = cKDTree(points).query(mesh.vertices)
    far = d > float(max_distance)
    keep = ~np.all(far[mesh.faces], axis=1)
    out = Mesh(vertices=mesh.vertices, faces=mesh.faces[keep], metadata=dict(mesh.metadata))
    return remove_unreferenced_vertices(out)


def poisson_reconstruct(
    cloud: PointCloud,
    *,
    resolution: int = 0,
    normals_k: int = 16,
    smoothing: float = 1.0,
    trim: float = 0.0,
    pad: float = 0.1,
) -> Mesh:
    """Poisson surface reconstruction (Kazhdan et al. 2006) on a regular grid.

    The oriented normals are splatted into a vector field, and the indicator
    function whose gradient best matches it is found by solving a Poisson
    equation (with a DCT, i.e. Neumann boundaries). Its level set through the
    samples is the surface: watertight, smooth, and robust to noise and uneven
    sampling — the usual choice for scans and segmentation boundary points.

    *resolution* is the grid size along the longest side (0 = from the sampling
    density); *smoothing* (grid steps) blurs the normal field (more = smoother,
    fewer spurious bumps); *trim* removes surface farther than this many sample
    spacings from any sample — where the solver bridged a gap in the data, e.g.
    the open end of a vessel (0 keeps the closed result).
    """
    from scipy.fft import dctn, idctn
    from scipy.ndimage import gaussian_filter, map_coordinates
    from skimage.measure import marching_cubes

    from nvitk.meshlab.cleaning import ensure_outward

    pts = np.asarray(cloud.points, dtype=float)
    if len(pts) < 10:
        raise ValueError("Poisson reconstruction needs at least 10 points.")
    normals = _oriented_normals(cloud, normals_k)
    origin, step, shape = _grid_for(pts, resolution, pad)
    g = (pts - origin) / step
    base = np.floor(g).astype(np.int64)
    frac = g - base
    field = np.zeros((3, *shape), dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (np.where(dx, frac[:, 0], 1 - frac[:, 0]) * np.where(dy, frac[:, 1], 1 - frac[:, 1])
                     * np.where(dz, frac[:, 2], 1 - frac[:, 2]))
                ix = np.clip(base[:, 0] + dx, 0, shape[0] - 1)
                iy = np.clip(base[:, 1] + dy, 0, shape[1] - 1)
                iz = np.clip(base[:, 2] + dz, 0, shape[2] - 1)
                for c in range(3):
                    np.add.at(field[c], (ix, iy, iz), w * normals[:, c])
    if smoothing > 0:
        for c in range(3):
            field[c] = gaussian_filter(field[c], float(smoothing))
    div = sum(np.gradient(field[c], step, axis=c) for c in range(3))
    coeffs = dctn(div, type=2, norm="ortho")
    eig = [(2.0 * np.cos(np.pi * np.arange(n) / n) - 2.0) / step ** 2 for n in shape]
    lap = eig[0][:, None, None] + eig[1][None, :, None] + eig[2][None, None, :]
    lap[0, 0, 0] = 1.0
    coeffs /= lap
    coeffs[0, 0, 0] = 0.0
    chi = idctn(coeffs, type=2, norm="ortho")
    iso = float(np.mean(map_coordinates(chi, g.T, order=1, mode="nearest")))
    verts, faces, _n, _v = marching_cubes(chi.astype(np.float32), level=iso, allow_degenerate=False)
    mesh = Mesh(vertices=origin + verts * step, faces=faces, metadata=dict(cloud.metadata))
    mesh = ensure_outward(mesh)
    if trim > 0:
        from scipy.spatial import cKDTree

        # The 90th-percentile spacing, not the median: random sampling leaves gaps
        # several median spacings wide that are not holes in the object.
        d, _ = cKDTree(pts).query(pts, k=2)
        mesh = _trim_far(mesh, pts, float(trim) * max(step, float(np.percentile(d[:, 1], 90))))
    return mesh


def imls_reconstruct(
    cloud: PointCloud,
    *,
    resolution: int = 0,
    normals_k: int = 16,
    neighbours: int = 12,
    band: float = 6.0,
    pad: float = 0.05,
) -> Mesh:
    """Implicit moving least squares (Kolluri 2005): the zero set of a smooth signed distance.

    Each grid node near the samples gets the Gaussian-weighted mean of its
    nearest samples' plane distances ``(x − pᵢ)·nᵢ``. Follows the data closely
    (no global smoothing, open surfaces stay open); evaluated only within *band*
    grid steps of the samples.
    """
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes

    from nvitk.meshlab.cleaning import ensure_outward

    pts = np.asarray(cloud.points, dtype=float)
    if len(pts) < 10:
        raise ValueError("IMLS reconstruction needs at least 10 points.")
    normals = _oriented_normals(cloud, normals_k)
    origin, step, shape = _grid_for(pts, resolution, pad)
    tree = cKDTree(pts)
    # Nodes within the band: dilate the occupied cells.
    from scipy.ndimage import binary_dilation

    occ = np.zeros(shape, dtype=bool)
    cells = np.clip(np.round((pts - origin) / step).astype(int), 0, np.asarray(shape) - 1)
    occ[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    near = binary_dilation(occ, iterations=max(1, int(np.ceil(band))))
    # Evaluate one step wider than the band that is meshed: a cube at the band's
    # edge must see real values at all eight corners, or the fill value outside
    # makes a spurious shell there.
    nodes = np.argwhere(binary_dilation(near, iterations=1))
    xyz = origin + nodes * step
    k = max(3, min(int(neighbours), len(pts)))
    dist, idx = tree.query(xyz, k=k)
    spacing = float(np.median(tree.query(pts, k=2)[0][:, 1]))
    sigma = max(step, 2.0 * spacing)
    w = np.exp(-((dist / sigma) ** 2))
    plane = np.einsum("nkj,nkj->nk", xyz[:, None, :] - pts[idx], normals[idx])
    f = (w * plane).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12)
    vol = np.full(shape, np.float32(step * band * 4), dtype=np.float32)
    vol[nodes[:, 0], nodes[:, 1], nodes[:, 2]] = f.astype(np.float32)
    # Only cubes whose eight corners are all in the band.
    from scipy.ndimage import binary_erosion

    cube = np.ones((2, 2, 2), dtype=bool)
    mask = binary_erosion(near, structure=cube, origin=(-1, -1, -1), border_value=0)
    try:
        verts, faces, _n, _v = marching_cubes(vol, level=0.0, mask=mask, allow_degenerate=False)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"No surface found ({exc}).") from None
    mesh = Mesh(vertices=origin + verts * step, faces=faces, metadata=dict(cloud.metadata))
    # Zero crossings far from every sample are artefacts of the local fits (the
    # 90th-percentile spacing: random sampling has gaps wider than the median).
    p90 = float(np.percentile(tree.query(pts, k=2)[0][:, 1], 90))
    mesh = _trim_far(mesh, pts, 3.0 * max(step, p90))
    from nvitk.meshlab.cleaning import remove_small_components

    mesh = remove_small_components(mesh, min_faces=max(20, mesh.n_faces // 200))
    return ensure_outward(mesh)


def reconstruct_surface(
    cloud: PointCloud,
    method: str = "poisson",
    *,
    resolution: int = 0,
    sample_spacing: float = 0.0,
    alpha: float = 0.0,
    smoothing: float = 1.0,
    trim: float = 0.0,
    normals_k: int = 16,
    keep_largest: bool = False,
    smooth_iterations: int = 0,
) -> Mesh:
    """A surface through the points.

    ``"poisson"`` (:func:`poisson_reconstruct`): watertight and smooth, robust to
    noise — the default. ``"screened_poisson"``: MeshLab's screened Poisson
    (PyMeshLab; octree depth from *resolution*, 8 by default). ``"ball_pivoting"``:
    MeshLab's ball pivoting, which interpolates the points and keeps holes open
    (ball radius *alpha*, 0 = automatic). ``"imls"`` (:func:`imls_reconstruct`):
    follows the points closely, keeps open surfaces open. ``"implicit"``: VTK's signed-distance
    reconstruction (*sample_spacing*, 0 = automatic). ``"alpha_shape"``: the
    outer surface of a Delaunay tetrahedralisation without edges longer than
    *alpha* (0 = convex). ``"convex_hull"``. Point normals are used when the
    cloud has them, otherwise estimated and consistently oriented.
    *keep_largest* drops stray pieces; *smooth_iterations* runs Taubin smoothing.
    """
    from nvitk.meshlab.cleaning import clean, keep_components
    from nvitk.meshlab.smoothing import taubin_smooth

    method = {"delaunay": "alpha_shape"}.get(str(method), str(method))
    if method == "convex_hull":
        from nvitk.meshlab.remeshing import convex_hull

        out = convex_hull(cloud)
    elif method == "poisson":
        out = poisson_reconstruct(cloud, resolution=resolution, normals_k=normals_k,
                                  smoothing=smoothing, trim=trim)
    elif method in ("screened_poisson", "ball_pivoting"):
        from nvitk.meshlab.pymeshlab_filters import run_meshlab_filter

        oriented = PointCloud(points=np.asarray(cloud.points, dtype=float), metadata=dict(cloud.metadata),
                              point_data={"normals": _oriented_normals(cloud, normals_k)})
        if method == "screened_poisson":
            depth = 8 if resolution <= 0 else int(np.clip(np.ceil(np.log2(resolution)), 5, 12))
            res = run_meshlab_filter(oriented, "generate_surface_reconstruction_screened_poisson",
                                     {"depth": depth, "preclean": True})
        else:
            radius = f"{float(alpha)}" if alpha and alpha > 0 else "0%"
            res = run_meshlab_filter(oriented, "generate_surface_reconstruction_ball_pivoting", {"ballradius": radius})
        # Screened Poisson adds a new mesh; ball pivoting puts faces on the cloud itself.
        made = next((m for m in res.meshes if isinstance(m, Mesh)), None)
        made = made if made is not None else (res.current if isinstance(res.current, Mesh) else None)
        if made is None or not made.n_faces:
            raise ValueError(f"MeshLab's {method.replace('_', ' ')} produced no surface.")
        out = clean(Mesh(made.vertices, made.faces, metadata=dict(cloud.metadata)))
        if method == "screened_poisson" and trim > 0:
            out = _trim_far(out, oriented.points, float(trim) * _point_spacing(oriented.points))
    elif method == "imls":
        out = imls_reconstruct(cloud, resolution=resolution, normals_k=normals_k)
    elif method in ("implicit", "alpha_shape"):
        require_pyvista()
        poly = to_pyvista(PointCloud(points=cloud.points))
        if method == "implicit":
            kwargs = {}
            if sample_spacing and sample_spacing > 0:
                kwargs["sample_spacing"] = float(sample_spacing)
            surf = poly.reconstruct_surface(**kwargs)
        else:
            surf = poly.delaunay_3d(alpha=float(alpha)).extract_geometry()
        out = clean(from_pyvista(surf, metadata=dict(cloud.metadata), keep_data=False))
    else:
        raise ValueError(f"method must be one of {RECONSTRUCTION_METHODS}.")
    if keep_largest and out.n_faces:
        out = keep_components(out, largest=1)
    if smooth_iterations > 0 and out.n_faces:
        out = taubin_smooth(out, iterations=int(smooth_iterations))
    return out


def points_to_mask(
    points: np.ndarray,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    *,
    radius: float = 0.0,
    fill: bool = False,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Mark the voxels of a grid (shape + affine) that the points fall in.

    *radius* (mm) grows each point into a ball; *fill* turns points sampling a
    closed boundary into the solid inside it (Poisson-reconstructed per label,
    then filled). With *labels* (one per point) the output carries them, else 1.
    """
    from scipy import ndimage

    aff = np.asarray(affine, dtype=float)
    ijk = np.round((np.c_[points, np.ones(len(points))] @ np.linalg.inv(aff).T)[:, :3]).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(shape)), axis=1)
    ijk = ijk[inside]
    vals = np.ones(len(ijk), dtype=np.int32) if labels is None else np.asarray(labels)[inside].astype(np.int32)
    out = np.zeros(shape, dtype=np.int32)
    out[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = vals
    spacing = np.linalg.norm(aff[:3, :3], axis=0)
    if radius > 0:
        dist, (ii, jj, kk) = ndimage.distance_transform_edt(out == 0, sampling=spacing, return_indices=True)
        out = np.where(dist <= float(radius), out[ii, jj, kk], 0).astype(np.int32)
    if fill:
        # A boundary sampling only becomes a solid reliably through a surface:
        # Poisson-reconstruct each label's points and fill the closed result.
        from nvitk.meshlab.voxelize import mesh_to_mask

        pts_in = np.asarray(points, dtype=float)[inside]
        lab_in = vals
        filled = out.copy()
        for lid in [int(v) for v in np.unique(lab_in)]:
            sel = pts_in[lab_in == lid]
            if len(sel) < 20:
                continue
            surf = poisson_reconstruct(PointCloud(points=sel))
            solid = np.asarray(mesh_to_mask(surf, affine=aff, shape=tuple(shape)).data) > 0
            filled[solid & (filled == 0)] = lid
        out = filled
    return out


__all__ = [
    "NORMAL_ORIENTATIONS",
    "RECONSTRUCTION_METHODS",
    "estimate_normals",
    "imls_reconstruct",
    "orient_normals",
    "points_to_mask",
    "poisson_reconstruct",
    "points_from_mask",
    "radius_outliers",
    "random_downsample",
    "reconstruct_surface",
    "remove_outliers",
    "sample_surface",
    "statistical_outliers",
    "voxel_downsample",
]
