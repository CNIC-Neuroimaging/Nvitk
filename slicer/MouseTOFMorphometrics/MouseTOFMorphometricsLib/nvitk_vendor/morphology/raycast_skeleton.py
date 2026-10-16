# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/morphology/raycast_skeleton.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""Ray-casting skeletonization and geometric descriptors (EXPIGEO, *experimental*).

Implements the interior-geometry exploration and descriptor stages of

    M. A. Layachi, A. Nouri, S. Eddarouich, F. Autrusseau, "Exploiting Interior Geometry for
    Intracranial Aneurysm Detection and Segmentation on 3D Point Clouds via GNNs" (EXPIGEO),
    MICCAI 2026, LNCS 16885, pp. 254–263. https://doi.org/10.1007/978-3-032-38091-3_25

re-implemented from the paper's equations (the GNN stage is not part of nvitk):

1. **Ray probing** — every surface point ``o`` (normal ``n_o``) casts ``K`` rays into an inward cone
   ``dᵀ(−n_o) ≥ cos θmax``; a surface point ``t`` is *hit* when its perpendicular distance to the
   ray is ≤ δ = γ · d̄_k (mean k-NN spacing of the cloud). Rays are marched through a KD-tree, all
   at once, and stop at their first hit.
2. **Centerline and radius** (Eq. 1) — ``cᵢ = (oᵢ + t̃ᵢ)/2`` with ``t̃ᵢ`` the spatial (geometric)
   median of the hits, ``rᵢ = ½ · median‖t − oᵢ‖``; kept when the hit dispersion is coherent and the
   local spacing is not an outlier (``d̄ᵢ < μ_d + η σ_d``).
3. **Skeleton** — the centerline cloud is voxelized (vertex = spatial median, density = count,
   re-centred on the median of a ball of centres), joined by a minimum spanning tree, spurious
   branches pruned iteratively, and close vertices merged with the adaptive threshold
   τ_merge = λ · median(nearest-neighbour distance) (Eq. 2).
4. **Descriptors** — radius-gradient magnitude ‖∇R‖ (weighted least squares, Eq. 3), termination
   likelihood ``P_term`` (Eq. 4), blockage ``β`` at leaves, geometric sphericity ``φ`` (Eq. 5) and
   projected flow ``F_proj``, broadcast to every surface point through its nearest skeleton vertex
   (the 9-column feature matrix of the paper's graph).

Status: experimental — available as the ``raycast`` skeleton method and Mesh-panel tool, never a
default. NumPy / SciPy (+ VTK for the mask/mesh helpers) only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RaycastParams:
    """Ray-probing and skeletonization settings (paper defaults where the paper gives them)."""

    #: Rays per source point (paper: K = 8) and cone half-angle (θmax = 15°).
    n_rays: int = 8
    theta_max_deg: float = 15.0
    #: Hit tolerance δ = gamma · mean k-NN spacing (paper: γ = 1/3).
    gamma: float = 1.0 / 3.0
    #: Neighbours for the local spacing d̄ᵢ (paper: k = 10) and the density tolerance η (0.05).
    k_spacing: int = 10
    eta: float = 0.05
    #: Hit coherence: keep a source when std(hit distances) ≤ this fraction of their median.
    max_hit_dispersion: float = 0.25
    #: A source needs at least this many of its rays to hit.
    min_hits: int = 3
    #: Rays are marched up to this length (mm; ``None`` = the cloud's bounding-box diagonal).
    max_ray_length: float | None = None
    #: Skeleton voxel size (mm; ``None`` = half the median centerline radius, at least 2 × spacing).
    voxel_size: float | None = None
    #: Spurious-branch pruning (repeated until stable): a leaf branch (leaf → nearest junction)
    #: shorter than prune_length (``None`` = 2 × median radius, i.e. one vessel diameter) is
    #: removed, and so is a leaf with density below prune_density (``None`` = a quarter of the
    #: median vertex density). Chains joining two leaves are never removed (the vessel's ends).
    prune_length: float | None = None
    prune_density: float | None = None
    prune_iterations: int = 50
    #: τ_merge = merge_lambda · median nearest-neighbour vertex distance (paper: λ = 0.7).
    merge_lambda: float = 0.7
    #: Sources are subsampled to at most this many surface points (``0`` = all).
    max_sources: int = 40000
    seed: int = 0


@dataclass
class RaycastSkeleton:
    """Result of :func:`raycast_skeleton`."""

    #: Centerline cloud (Eq. 1) with radius, and the surface source of each centerline point.
    centerline_points: np.ndarray
    centerline_radius: np.ndarray
    source_index: np.ndarray
    #: Skeleton graph (a tree): vertices, per-vertex radius / density, undirected edges.
    vertices: np.ndarray
    radius: np.ndarray
    density: np.ndarray
    edges: np.ndarray
    params: RaycastParams = field(default_factory=RaycastParams)

    @property
    def degree(self) -> np.ndarray:
        deg = np.zeros(len(self.vertices), dtype=int)
        if len(self.edges):
            np.add.at(deg, self.edges.ravel(), 1)
        return deg

    @property
    def leaves(self) -> np.ndarray:
        return np.flatnonzero(self.degree == 1)

    def adjacency(self) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(len(self.vertices))]
        for a, b in self.edges:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
        return adj

    def branches(self) -> list[np.ndarray]:
        """Vertex-index chains between branch points / leaves (each edge in exactly one chain)."""
        adj = self.adjacency()
        deg = self.degree
        seen: set[tuple[int, int]] = set()
        out: list[np.ndarray] = []
        nodes = [int(v) for v in np.flatnonzero(deg != 2)] or ([0] if len(self.vertices) else [])
        for start in nodes:
            for nxt in adj[start]:
                key = (min(start, nxt), max(start, nxt))
                if key in seen:
                    continue
                chain = [start, nxt]
                seen.add(key)
                prev, cur = start, nxt
                while deg[cur] == 2:
                    a, b = adj[cur]
                    step = b if a == prev else a
                    key = (min(cur, step), max(cur, step))
                    if key in seen:
                        break
                    seen.add(key)
                    chain.append(step)
                    prev, cur = cur, step
                out.append(np.asarray(chain, dtype=np.int64))
        return out

    def polylines(self) -> list[np.ndarray]:
        """Branch polylines (coordinates)."""
        return [self.vertices[c] for c in self.branches()]

    def path_between(self, a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Tree path between the vertices nearest to *a* and *b*: ``(points, radius)``."""
        from scipy.sparse.csgraph import shortest_path

        ia = int(np.argmin(np.linalg.norm(self.vertices - np.asarray(a, dtype=float), axis=1)))
        ib = int(np.argmin(np.linalg.norm(self.vertices - np.asarray(b, dtype=float), axis=1)))
        graph = _edge_graph(self.vertices, self.edges)
        _d, pred = shortest_path(graph, directed=False, indices=ia, return_predecessors=True)
        path = [ib]
        while path[-1] != ia and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        return self.vertices[path], self.radius[path]


# ──────────────────────────────────────────────────────────────────────────────
# 1–2. Ray probing → centerline cloud
# ──────────────────────────────────────────────────────────────────────────────


def _cone_directions(normals_in: np.ndarray, n_rays: int, theta_max: float, rng) -> np.ndarray:
    """``(N, K, 3)`` unit ray directions in the cone of half-angle *theta_max* around each inward normal."""
    n = normals_in / np.maximum(np.linalg.norm(normals_in, axis=1, keepdims=True), 1e-12)
    helper = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    u = np.cross(n, helper)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    v = np.cross(n, u)
    # One ray along the inward normal and the rest evenly spaced on the cone rim (randomly rotated
    # per source): a symmetric pattern keeps the spatial median of the hits on the far wall's
    # antipode, which a few random in-cone rays do not.
    cos_t = np.full((len(n), n_rays), np.cos(theta_max))
    cos_t[:, 0] = 1.0
    sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, None))
    ring = np.arange(n_rays, dtype=float) * 2.0 * np.pi / max(n_rays - 1, 1)
    phi = ring[None, :] + rng.random((len(n), 1)) * 2.0 * np.pi
    return (cos_t[..., None] * n[:, None, :]
            + (sin_t * np.cos(phi))[..., None] * u[:, None, :]
            + (sin_t * np.sin(phi))[..., None] * v[:, None, :])


def _geometric_median(points: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Weiszfeld spatial median of a small point set."""
    y = points.mean(axis=0)
    for _ in range(iterations):
        d = np.linalg.norm(points - y, axis=1)
        if np.any(d < 1e-9):
            return points[int(np.argmin(d))]
        w = 1.0 / d
        y_new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < 1e-7:
            return y_new
        y = y_new
    return y


def ray_probe(
    points: np.ndarray,
    normals: np.ndarray,
    params: RaycastParams = RaycastParams(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray probing (Sec. 2.1, Eq. 1): ``(centerline_points, radius, source_index)``.

    *normals* must point **outward** (rays go along −normal).
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=float).reshape(-1, 3)
    if len(pts) < 10:
        return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(params.seed))
    tree = cKDTree(pts)
    k = max(2, int(params.k_spacing))
    knn_d, _ = tree.query(pts, k=k + 1)
    local_spacing = knn_d[:, 1:].mean(axis=1)                  # d̄ᵢ
    mean_spacing = float(local_spacing.mean())                 # d̄_k
    delta = float(params.gamma) * mean_spacing                 # δ
    mu_d, sd_d = float(local_spacing.mean()), float(local_spacing.std())

    src = np.arange(len(pts))
    if params.max_sources and len(src) > int(params.max_sources):
        src = np.sort(rng.choice(src, size=int(params.max_sources), replace=False))
    dirs = _cone_directions(-nrm[src], int(params.n_rays), np.radians(float(params.theta_max_deg)), rng)
    origin = np.repeat(pts[src], int(params.n_rays), axis=0)
    d = dirs.reshape(-1, 3)

    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    s_max = float(params.max_ray_length) if params.max_ray_length else diag
    step = max(2.0 * delta, 0.5 * mean_spacing)
    radius_q = float(np.hypot(delta, 0.5 * step))
    s_min = 2.0 * mean_spacing                                  # skip the source's own neighbourhood

    hit_idx = np.full(len(d), -1, dtype=np.int64)
    hit_s = np.full(len(d), np.nan)
    active = np.arange(len(d))
    s = s_min
    while len(active) and s <= s_max:
        probe = origin[active] + s * d[active]
        dist, nn = tree.query(probe, k=1, distance_upper_bound=radius_q)
        cand = np.flatnonzero(np.isfinite(dist))
        if len(cand):
            rays = active[cand]
            t = pts[nn[cand]]
            rel = t - origin[rays]
            s_star = (rel * d[rays]).sum(axis=1)
            perp = np.linalg.norm(rel - s_star[:, None] * d[rays], axis=1)
            facing = (nrm[nn[cand]] * d[rays]).sum(axis=1) > 0.0  # far wall: outward normal along the ray
            ok = (perp <= delta) & (s_star > s_min) & facing
            hit_idx[rays[ok]] = nn[cand][ok]
            hit_s[rays[ok]] = s_star[ok]
            active = np.setdiff1d(active, rays[ok], assume_unique=True)
        s += step

    hits = hit_idx.reshape(len(src), int(params.n_rays))
    centers, radii, keep_src = [], [], []
    for i in range(len(src)):
        h = hits[i][hits[i] >= 0]
        if len(h) < int(params.min_hits):
            continue
        o = pts[src[i]]
        dist_h = np.linalg.norm(pts[h] - o, axis=1)
        med = float(np.median(dist_h))
        if med <= 0 or float(dist_h.std()) > float(params.max_hit_dispersion) * med:
            continue                                          # hit coherence σᵢ < τ_σ
        if local_spacing[src[i]] >= mu_d + float(params.eta) * sd_d + 3.0 * sd_d:
            continue                                          # spatial density d̄ᵢ < μ_d + η σ_d (+3σ slack)
        t_med = _geometric_median(pts[h])
        centers.append(0.5 * (o + t_med))
        radii.append(0.5 * med)
        keep_src.append(src[i])
    if not centers:
        return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.int64)
    return np.asarray(centers), np.asarray(radii), np.asarray(keep_src, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Skeleton
# ──────────────────────────────────────────────────────────────────────────────


def _edge_graph(vertices: np.ndarray, edges: np.ndarray):
    from scipy.sparse import coo_matrix

    n = len(vertices)
    if len(edges) == 0:
        return coo_matrix((n, n)).tocsr()
    w = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    w = np.maximum(w, 1e-9)
    return coo_matrix((w, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()


def _mst_edges(vertices: np.ndarray, k: int = 12) -> np.ndarray:
    """Minimum spanning forest of the k-NN graph of *vertices* (undirected edge list)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import cKDTree

    n = len(vertices)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    kk = min(k + 1, n)
    d, idx = cKDTree(vertices).query(vertices, k=kk)
    rows = np.repeat(np.arange(n), kk - 1)
    cols = idx[:, 1:].ravel()
    w = np.maximum(d[:, 1:].ravel(), 1e-9)
    g = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    g = g.maximum(g.T)
    mst = minimum_spanning_tree(g).tocoo()
    return np.column_stack([mst.row, mst.col]).astype(np.int64)


def _voxel_vertices(points: np.ndarray, radius: np.ndarray, voxel: float):
    """Spatial median / median radius / count of the centerline points in each occupied voxel."""
    keys = np.floor((points - points.min(axis=0)) / voxel).astype(np.int64)
    _uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(counts)])
    verts, rads = [], []
    for g in range(len(counts)):
        members = order[bounds[g]:bounds[g + 1]]
        verts.append(_geometric_median(points[members]) if len(members) >= 3 else points[members].mean(axis=0))
        rads.append(float(np.median(radius[members])))
    return np.asarray(verts), np.asarray(rads), counts.astype(float), inv


def _refine_vertices(vertices, centers, radii, ball):
    """Re-estimate each vertex as the spatial median of the centerline points within *ball*.

    Eq. 1's centres sit on a thin ring around the true axis (the cone's rim hits pull ``t̃`` towards
    the source, so every source's centre is displaced towards its own wall); an axis-aligned voxel
    sees only part of that ring, a ball centred on the vertex sees all of it, and the spatial median
    of a full ring is its centre.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(centers)
    out_v, out_r = vertices.copy(), radii[:0].tolist()
    for i, members in enumerate(tree.query_ball_point(vertices, float(ball))):
        if len(members) >= 3:
            out_v[i] = _geometric_median(centers[members])
            out_r.append(float(np.median(radii[members])))
        else:
            out_r.append(np.nan)
    rr = np.asarray(out_r, dtype=float)
    return out_v, rr


def _prune_leaves(vertices, radius, density, edges, length_max, density_max, iterations):
    """Remove spurious branches (paper Sec. 2.1, "Skeletonization").

    The paper prunes leaves by incident edge length (short spurious branches) and density
    (low-confidence outliers). Applied edge by edge, a length test would also erode the vessel's
    real ends, so here the length test applies to whole leaf branches — the chain from a leaf to
    its nearest junction — and only where that junction exists; low-density leaves are dropped
    anywhere. Repeated until nothing changes (or *iterations* passes).
    """
    keep = np.ones(len(vertices), dtype=bool)
    e = edges.copy()
    for _ in range(max(0, int(iterations))):
        adj: list[list[int]] = [[] for _ in range(len(vertices))]
        for a, b in e:
            adj[int(a)].append(int(b))
            adj[int(b)].append(int(a))
        deg = np.asarray([len(x) for x in adj])
        drop: set[int] = set()
        for leaf in np.flatnonzero((deg == 1) & keep):
            if density[leaf] < density_max:
                drop.add(int(leaf))
                continue
            chain, length, prev, cur = [int(leaf)], 0.0, -1, int(leaf)
            while True:
                nxt = [x for x in adj[cur] if x != prev]
                if not nxt:
                    break  # reached another leaf: a whole vessel, keep it
                length += float(np.linalg.norm(vertices[nxt[0]] - vertices[cur]))
                prev, cur = cur, nxt[0]
                if deg[cur] >= 3:
                    if length < length_max:
                        drop.update(chain)
                    break
                chain.append(cur)
        if not drop:
            break
        keep[list(drop)] = False
        e = e[keep[e[:, 0]] & keep[e[:, 1]]]
    return _compact(vertices, radius, density, e, keep)

def _compact(vertices, radius, density, edges, keep):
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    e = remap[edges]
    e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0)] if len(e) else e
    return vertices[keep], radius[keep], density[keep], e


def _merge_close(vertices, radius, density, lam):
    """Merge vertices closer than τ_merge = lam · median NN distance (Eq. 2) by spatial averaging."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if len(vertices) < 3:
        return vertices, radius, density
    tree = cKDTree(vertices)
    nn_d, _ = tree.query(vertices, k=2)
    tau = float(lam) * float(np.median(nn_d[:, 1]))
    pairs = tree.query_pairs(tau, output_type="ndarray")
    if len(pairs) == 0:
        return vertices, radius, density
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(vertices),) * 2)
    n_comp, label = connected_components(g, directed=False)
    w = density
    v_new = np.zeros((n_comp, 3))
    r_new = np.zeros(n_comp)
    d_new = np.zeros(n_comp)
    np.add.at(v_new, label, vertices * w[:, None])
    np.add.at(r_new, label, radius * w)
    np.add.at(d_new, label, w)
    return v_new / d_new[:, None], r_new / d_new, d_new


def raycast_skeleton(
    points: np.ndarray,
    normals: np.ndarray,
    params: RaycastParams = RaycastParams(),
) -> RaycastSkeleton:
    """Full EXPIGEO interior exploration: ray probing → centerline cloud → pruned, merged MST."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    centers, radii, src = ray_probe(pts, normals, params)
    if len(centers) < 2:
        raise ValueError("Ray probing found no centerline points — are the normals outward and the surface closed?")
    from scipy.spatial import cKDTree

    spacing = float(cKDTree(pts).query(pts, k=2)[0][:, 1].mean())
    r_med = float(np.median(radii))
    voxel = float(params.voxel_size) if params.voxel_size else max(0.5 * r_med, 2.0 * spacing)
    verts, rads, dens, _inv = _voxel_vertices(centers, radii, voxel)
    refined, refined_r = _refine_vertices(verts, centers, radii, voxel)
    verts = refined
    rads = np.where(np.isfinite(refined_r), refined_r, rads)
    edges = _mst_edges(verts)
    length_max = float(params.prune_length) if params.prune_length else 2.0 * r_med
    density_max = float(params.prune_density) if params.prune_density else 0.25 * float(np.median(dens))
    verts, rads, dens, edges = _prune_leaves(verts, rads, dens, edges, length_max, density_max,
                                             params.prune_iterations)
    verts, rads, dens = _merge_close(verts, rads, dens, params.merge_lambda)
    edges = _mst_edges(verts)
    # The merge re-links the vertices; prune the spurs that the new tree can sprout.
    verts, rads, dens, edges = _prune_leaves(verts, rads, dens, edges, length_max, density_max,
                                             params.prune_iterations)
    return RaycastSkeleton(
        centerline_points=centers, centerline_radius=radii, source_index=src,
        vertices=verts, radius=rads, density=dens, edges=edges, params=params,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Geometric descriptors
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ExpigeoDescriptors:
    """Per-skeleton-vertex descriptors and their broadcast to the surface points."""

    #: Skeleton vertex arrays.
    radius_gradient: np.ndarray
    termination_likelihood: np.ndarray
    blockage: np.ndarray
    sphericity: np.ndarray
    direction: np.ndarray
    #: Surface-point arrays: nearest skeleton vertex ψ(p) and projected flow.
    nearest_vertex: np.ndarray
    projected_flow: np.ndarray
    #: The paper's node feature matrix ``[p, P_term, β, φ, r, ‖∇R‖, F_proj]`` (N × 9).
    features: np.ndarray

    FEATURE_NAMES: tuple[str, ...] = (
        "x", "y", "z", "termination_likelihood", "blockage", "sphericity", "radius",
        "radius_gradient", "projected_flow",
    )

    def surface_fields(self) -> dict[str, np.ndarray]:
        """Per-surface-point fields (columns 3–8 of :attr:`features`) by name."""
        return {name: self.features[:, i] for i, name in enumerate(self.FEATURE_NAMES) if i >= 3}


def radius_gradient_magnitude(points: np.ndarray, radius: np.ndarray, k: int = 10) -> np.ndarray:
    """‖∇R‖ at each point (Eq. 3): Gaussian-weighted least-squares fit of the radius field over
    its *k* nearest neighbours, σ_k = their mean distance."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=float)
    r = np.asarray(radius, dtype=float)
    kk = min(int(k) + 1, len(pts))
    if kk < 4:
        return np.zeros(len(pts))
    d, idx = cKDTree(pts).query(pts, k=kk)
    d, idx = d[:, 1:], idx[:, 1:]
    sigma = np.maximum(d.mean(axis=1, keepdims=True), 1e-9)
    w = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    a = pts[idx] - pts[:, None, :]                       # (N, k, 3)
    b = r[idx] - r[:, None]                              # (N, k)
    ata = np.einsum("nk,nki,nkj->nij", w, a, a) + 1e-9 * np.eye(3)
    atb = np.einsum("nk,nki,nk->ni", w, a, b)
    g = np.linalg.solve(ata, atb[..., None])[..., 0]
    return np.linalg.norm(g, axis=1)


def _vertex_directions(skel: RaycastSkeleton) -> np.ndarray:
    """Skeletal direction at each vertex (leaves point outward, ``d_v = (v − u)/‖v − u‖``)."""
    adj = skel.adjacency()
    v = skel.vertices
    out = np.zeros_like(v)
    for i, nb in enumerate(adj):
        if len(nb) == 1:
            t = v[i] - v[nb[0]]
        elif len(nb) == 2:
            t = v[nb[1]] - v[nb[0]]
        elif len(nb) > 2:
            t = np.linalg.svd(v[nb] - v[nb].mean(axis=0))[2][0]
        else:
            t = np.zeros(3)
        n = np.linalg.norm(t)
        out[i] = t / n if n > 1e-12 else t
    return out


def expigeo_descriptors(
    skel: RaycastSkeleton,
    surface_points: np.ndarray,
    *,
    termination_sigma: float = 2.0,
    blockage_aperture_deg: float = 30.0,
    epsilon: float = 1e-6,
) -> ExpigeoDescriptors:
    """Descriptors of Sec. 2.2 on the skeleton, broadcast to *surface_points* via ψ."""
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    surf = np.asarray(surface_points, dtype=float).reshape(-1, 3)
    verts = skel.vertices
    n_v = len(verts)

    # ‖∇R‖: on the centerline cloud, aggregated to the vertices by nearest-vertex median.
    grad_c = radius_gradient_magnitude(skel.centerline_points, skel.centerline_radius)
    _d, c_owner = cKDTree(verts).query(skel.centerline_points)
    grad_v = np.zeros(n_v)
    for i in range(n_v):
        m = grad_c[c_owner == i]
        grad_v[i] = float(np.median(m)) if len(m) else np.nan
    if np.isnan(grad_v).any():
        fill = float(np.nanmedian(grad_v)) if np.isfinite(grad_v).any() else 0.0
        grad_v = np.where(np.isnan(grad_v), fill, grad_v)

    # P_term (Eq. 4): Gaussian of the geodesic distance to the nearest leaf.
    leaves = skel.leaves
    if len(leaves):
        dist = dijkstra(_edge_graph(verts, skel.edges), directed=False, indices=leaves).min(axis=0)
        p_term = np.exp(-(dist ** 2) / (2.0 * float(termination_sigma) ** 2))
    else:
        p_term = np.zeros(n_v)

    # Voronoi broadcast ψ and per-vertex surface clusters Ω_v.
    _d, owner = cKDTree(verts).query(surf)
    direction = _vertex_directions(skel)

    # Blockage β (leaves only): any cluster point in front of the leaf within the aperture cone.
    cos_a = np.cos(np.radians(float(blockage_aperture_deg)))
    blockage = np.zeros(n_v)
    for leaf in leaves:
        omega = surf[owner == leaf]
        if len(omega) == 0:
            continue
        rel = omega - verts[leaf]
        nrm = np.linalg.norm(rel, axis=1)
        ok = nrm > 1e-9
        if np.any((rel[ok] @ direction[leaf]) / nrm[ok] >= cos_a):
            blockage[leaf] = 1.0

    # Geometric sphericity φ (Eq. 5).
    sphericity = np.full(n_v, np.nan)
    for i in range(n_v):
        q = surf[owner == i] - verts[i]
        if len(q) < 5:
            continue
        eps_sph = float(np.linalg.norm(q, axis=1).std())
        _u, _s, vt = np.linalg.svd(q - q.mean(axis=0), full_matrices=False)
        eps_cyl = min(float(np.linalg.norm(np.cross(q, vt[a]), axis=1).std()) for a in (0, 2))
        sphericity[i] = eps_cyl / (eps_cyl + eps_sph + float(epsilon))
    sphericity = np.where(np.isnan(sphericity), float(np.nanmedian(sphericity)) if np.isfinite(sphericity).any() else 0.0, sphericity)

    # Projected flow: (p − v_p)ᵀ d_{v_p}.
    f_proj = ((surf - verts[owner]) * direction[owner]).sum(axis=1)

    features = np.column_stack([
        surf,
        p_term[owner],
        blockage[owner],
        sphericity[owner],
        skel.radius[owner],
        grad_v[owner],
        f_proj,
    ])
    return ExpigeoDescriptors(
        radius_gradient=grad_v,
        termination_likelihood=p_term,
        blockage=blockage,
        sphericity=sphericity,
        direction=direction,
        nearest_vertex=owner.astype(np.int64),
        projected_flow=f_proj,
        features=features,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mask / mesh helpers
# ──────────────────────────────────────────────────────────────────────────────


def surface_points_normals(surface: Any) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and consistent **outward** vertex normals of a closed ``vtkPolyData``."""
    import vtk
    from vtk.util import numpy_support

    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(surface)
    nf.ComputePointNormalsOn()
    nf.ComputeCellNormalsOff()
    nf.SplittingOff()
    nf.ConsistencyOn()
    nf.AutoOrientNormalsOn()
    nf.Update()
    out = nf.GetOutput()
    pts = numpy_support.vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    nrm = numpy_support.vtk_to_numpy(out.GetPointData().GetNormals()).astype(float)
    return pts, nrm


def raycast_skeleton_from_surface(surface: Any, params: RaycastParams = RaycastParams()) -> RaycastSkeleton:
    """:func:`raycast_skeleton` on a closed ``vtkPolyData`` (normals computed and oriented outward)."""
    pts, nrm = surface_points_normals(surface)
    return raycast_skeleton(pts, nrm, params)


def raycast_skeleton_from_mask(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    params: RaycastParams = RaycastParams(),
) -> tuple[RaycastSkeleton, Any]:
    """:func:`raycast_skeleton` on the smoothed marching-cubes surface of a binary mask
    (``index × spacing`` frame). Returns ``(skeleton, surface)``."""
    from nvitk_vendor.morphology.centering import mask_surface

    surface = mask_surface(np.asarray(mask).astype(bool), spacing)
    return raycast_skeleton_from_surface(surface, params), surface


__all__ = [
    "ExpigeoDescriptors",
    "RaycastParams",
    "RaycastSkeleton",
    "expigeo_descriptors",
    "radius_gradient_magnitude",
    "ray_probe",
    "raycast_skeleton",
    "raycast_skeleton_from_mask",
    "raycast_skeleton_from_surface",
    "surface_points_normals",
]
