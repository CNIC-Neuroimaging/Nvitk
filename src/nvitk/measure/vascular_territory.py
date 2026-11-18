"""
Vascular territories — which brain regions a segmented vessel irrigates.

Description
-----------
Given a multilabel vessel segmentation (eICAB, TopBrain, QVTPy... any label set whose names
say which artery each label is) and a brain parcellation placed in the same world space, this
estimates the territory of every vessel and how much of each parcellation region it supplies.

Model
-----
1. **Vessel identity from its name.** :func:`classify_vessel` reads side, artery family and
   segment from names as different as ``LMCA_M2`` (eICAB), ``R-P1P2`` (TopBrain) and ``LVA``
   (QVTPy). Veins and extracranial branches are not classified, and take no part.
2. **Supply tree.** :func:`supply_tree` links the classified vessels downstream: ICA -> MCA /
   ACA / AChA / PCom, VA -> basilar + PICA, basilar -> PCAs / SCAs / AICAs, and each segment
   to the next (M1 -> M2 -> M3...). The communicating arteries are collaterals: by default
   they are their own small territories and pass nothing on; with ``follow_collaterals`` the
   PCom feeds the PCA beyond P1 and the ACom both ACAs beyond A1 (fetal PCA, A1 hypoplasia).
3. **Nearest-vessel territories.** Every parenchyma voxel is assigned to the nearest vessel
   voxel (Euclidean, mm) among the *seed* vessels: the ones with parenchymal branches. The
   ICA and vertebral arteries are conduits, not seeds: their cell would claim the temporal
   pole and the medulla that their branches, not they, supply. With ``respect_hemispheres``
   a lateralized region only competes between that side's vessels and the unpaired ones.
4. **Supply.** A vessel's territory is the union of the cells of every seed downstream of it
   (itself included), so clicking an ICA lights up the MCA and ACA territories. A region's
   share is the fraction of its voxels in that union.

This is a geometric proxy, not perfusion: angiography only resolves the proximal arteries,
and the cortex between their visible ends is split by distance. Expect the classical MCA / ACA /
PCA layout, with watershed borders that are approximate to within a gyrus.

Array / axis conventions
------------------------
Everything runs on one *working grid*: an axis-aligned resampling of the vessel grid at about
1 mm (:func:`working_grid`) that also covers the whole parcellation, since an angiography
slab rarely covers the whole brain. Geometry is carried as 4x4 voxel-index -> world (mm)
matrices; vessel, parcellation and working grid must share that world. Backend ``np`` /
``ndi`` after ``setup(globals())``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.core.logger import Logger

setup(globals())

log = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Vessel identity
# ──────────────────────────────────────────────────────────────────────────────

#: Artery families, as :func:`classify_vessel` reports them.
FAMILIES: tuple[str, ...] = (
    "ICA", "MCA", "ACA", "PCA", "ACOM", "PCOM", "ACHA", "BA", "VA", "SCA", "AICA", "PICA", "OA",
)
#: Families that carry blood through to their branches without supplying parenchyma themselves.
CONDUIT_FAMILIES: frozenset[str] = frozenset({"ICA", "VA"})
#: Families that do not supply the brain (the ophthalmic artery supplies the orbit).
EXTRACEREBRAL_FAMILIES: frozenset[str] = frozenset({"OA"})
#: Communicating arteries (collaterals between the anterior and posterior circulations).
COLLATERAL_FAMILIES: frozenset[str] = frozenset({"ACOM", "PCOM"})

#: Family stems, matched as a whole name token (after the side is stripped) or as a substring
#: for the stems long enough to be unambiguous. Ordered: AICA / PICA before ICA, SCA last.
_STEMS: tuple[tuple[str, str], ...] = (
    ("ACOM", "ACOM"), ("PCOM", "PCOM"), ("ACHA", "ACHA"), ("AICA", "AICA"), ("PICA", "PICA"),
    ("ICA", "ICA"), ("MCA", "MCA"), ("ACA", "ACA"), ("PCA", "PCA"), ("BASILAR", "BA"),
    ("VERTEBRAL", "VA"), ("OPHTHALMIC", "OA"), ("SCA", "SCA"),
)
#: Two-letter families: only as a whole token (``BA``, ``R-VA``) or right after a side letter
#: (``LVA``); as substrings they would match inside other names.
_SHORT: dict[str, str] = {"BA": "BA", "VA": "VA", "OA": "OA"}
#: Segment tokens: ``M2`` / ``A1A2`` / ``P3P4`` imply the family; ``C6`` gives an ICA segment.
_SEGMENT_TOKEN = re.compile(r"^([MAPC])(\d)(?:[MAPC]?(\d))?$")
_SEGMENT_FAMILY = {"M": "MCA", "A": "ACA", "P": "PCA", "C": "ICA"}
_NOT_ARTERIES = re.compile(r"VEIN|SINUS|SSS|STS|VOG|ICV|BVR|ECA|STA|MAXA|MMA|BACKGROUND")


@dataclass(frozen=True)
class VesselClass:
    """What a vessel label is: side (``"L"`` / ``"R"`` / ``""``), family and segment span."""

    side: str
    family: str
    seg_min: int = 0
    seg_max: int = 0

    @property
    def is_seed(self) -> bool:
        """Has a territory of its own (not a conduit, supplies the brain)."""
        return self.family not in CONDUIT_FAMILIES and self.family not in EXTRACEREBRAL_FAMILIES


def _split_side(token: str) -> tuple[str, str]:
    """``("L", "MCA")`` from ``"LMCA"``; ``("", token)`` when no side letter leads a known stem."""
    for prefix, side in (("LEFT", "L"), ("RIGHT", "R"), ("LT", "L"), ("RT", "R"), ("L", "L"), ("R", "R")):
        rest = token[len(prefix):]
        if token.startswith(prefix) and rest and (
            any(rest.startswith(stem) for stem, _ in _STEMS) or rest in _SHORT
            or rest.startswith(("PCOMM", "ACOMM"))
        ):
            return side, rest
    return "", token


def classify_vessel(name: str | None) -> VesselClass | None:
    """
    Side, artery family and segment of a vessel label, read from its *name*.

    Handles the eICAB (``LICA``, ``LMCA_M1``, ``LACA_A2``, ``BASILAR``, ``ACOMM``), TopBrain
    (``R-M1``, ``L-P1P2``, ``R-ICA-C6-C7``, ``3rd-A2``, ``BA``) and QVTPy (``LVA``, ``RPCA_P2``)
    vocabularies, and free text such as ``Left vertebral``. Returns ``None`` for veins,
    extracranial branches and names it does not recognise.
    """
    text = str(name or "").strip().upper()
    if not text or _NOT_ARTERIES.search(text.replace(" ", "")):
        return None
    tokens = [t for t in re.split(r"[-_\s.]+", text) if t]
    if not tokens:
        return None

    # ---- Side: a token of its own, or a letter glued to the family stem ------------
    side = ""
    if tokens[0] in ("L", "R", "LEFT", "RIGHT", "LT", "RT"):
        side = tokens.pop(0)[0]
    elif tokens[-1] in ("L", "R", "LEFT", "RIGHT"):
        side = tokens.pop()[0]
    if not side and tokens:
        side, tokens[0] = _split_side(tokens[0])

    # ---- Family and segments ------------------------------------------------------
    family = ""
    segments: list[int] = []
    for token in tokens:
        seg = _SEGMENT_TOKEN.match(token)
        if seg:
            segments += [int(d) for d in seg.groups()[1:] if d]
            family = family or _SEGMENT_FAMILY[seg.group(1)]
            continue
        if not family:
            if token in _SHORT:
                family = _SHORT[token]
                continue
            for stem, fam in _STEMS:
                if stem in token:
                    family = fam
                    break
    if not family:
        return None
    return VesselClass(side, family, min(segments, default=0), max(segments, default=0))


# ──────────────────────────────────────────────────────────────────────────────
# Supply tree
# ──────────────────────────────────────────────────────────────────────────────


def classify_labels(names: Mapping[int, str]) -> dict[int, VesselClass]:
    """``{label id: class}`` for the labels :func:`classify_vessel` recognises."""
    out: dict[int, VesselClass] = {}
    for lid, name in names.items():
        cls = classify_vessel(name)
        if cls is not None and int(lid) > 0:
            out[int(lid)] = cls
    return out


def supply_tree(classes: Mapping[int, VesselClass], *, follow_collaterals: bool = False) -> dict[int, tuple[int, ...]]:
    """
    ``{label id: downstream label ids}``: which present vessels each one feeds directly.

    Only vessels present in *classes* are linked; a missing segment is bridged (M1 feeds the
    M3 when there is no M2 label). See the module docstring for the anatomy.
    """
    children: dict[int, set[int]] = {lid: set() for lid in classes}

    def chain(family: str, side: str) -> list[int]:
        ids = [lid for lid, c in classes.items() if c.family == family and c.side == side]
        return sorted(ids, key=lambda lid: (classes[lid].seg_min, classes[lid].seg_max, lid))

    def heads(family: str, side: str, *, beyond: int = 0) -> list[int]:
        """The most proximal labels of a chain, optionally only those reaching past segment *beyond*."""
        ids = [lid for lid in chain(family, side) if classes[lid].seg_max > beyond or classes[lid].seg_max == 0]
        if not ids:
            return []
        first = classes[ids[0]].seg_min
        return [lid for lid in ids if classes[lid].seg_min == first]

    def link(parents: Sequence[int], kids: Sequence[int]) -> None:
        for p in parents:
            children[p].update(k for k in kids if k != p)

    # ---- Segment chains: each segment feeds the next one present --------------------
    for family in ("ICA", "MCA", "ACA", "PCA", "VA", "BA"):
        for side in ("L", "R", ""):
            ids = chain(family, side)
            groups: list[list[int]] = []
            for lid in ids:
                if groups and classes[groups[-1][0]].seg_min == classes[lid].seg_min:
                    groups[-1].append(lid)
                else:
                    groups.append([lid])
            for proximal, distal in zip(groups, groups[1:]):
                link(proximal, distal)

    def tail(family: str, side: str) -> list[int]:
        ids = chain(family, side)
        if not ids:
            return []
        last = classes[ids[-1]].seg_min
        return [lid for lid in ids if classes[lid].seg_min == last]

    # ---- Branch points --------------------------------------------------------------
    basilar = tail("BA", "")
    for side in ("L", "R"):
        ica = tail("ICA", side)
        link(ica, heads("MCA", side) + heads("ACA", side) + chain("ACHA", side)
             + chain("PCOM", side) + chain("OA", side))
        vertebral = tail("VA", side)
        link(vertebral, heads("BA", "") + chain("PICA", side))
        link(basilar, heads("PCA", side) + chain("SCA", side) + chain("AICA", side))
    link(chain("ACOM", ""), heads("ACA", ""))  # median callosal artery (TopBrain "3rd-A2")

    # ---- Collaterals ----------------------------------------------------------------
    if follow_collaterals:
        for side in ("L", "R"):
            link(chain("PCOM", side), heads("PCA", side, beyond=1))
            link(chain("ACOM", ""), heads("ACA", side, beyond=1))

    # Extracerebral branches have nothing to pass on and nothing to supply.
    for lid, cls in classes.items():
        if cls.family in EXTRACEREBRAL_FAMILIES:
            for kids in children.values():
                kids.discard(lid)
    return {lid: tuple(sorted(kids)) for lid, kids in children.items()}


def downstream(label_id: int, tree: Mapping[int, Sequence[int]]) -> list[int]:
    """*label_id* and every vessel it feeds, transitively, in visiting order."""
    seen: list[int] = []
    stack = [int(label_id)]
    while stack:
        lid = stack.pop()
        if lid in seen:
            continue
        seen.append(lid)
        stack.extend(reversed(tree.get(lid, ())))
    return seen


# ──────────────────────────────────────────────────────────────────────────────
# Grids
# ──────────────────────────────────────────────────────────────────────────────


def working_grid(
    vessel_shape: Sequence[int],
    vessel_to_world: Any,
    extra_boxes: Sequence[Any] = (),
    *,
    target_mm: float = 1.0,
    margin_mm: float = 4.0,
) -> tuple[tuple[int, int, int], Any]:
    """
    An axis-aligned ~*target_mm* grid in the vessel grid's orientation, covering the vessel
    volume and every world-space box in *extra_boxes*.

    Parameters
    ----------
    vessel_shape : sequence of int
        Shape of the vessel label volume.
    vessel_to_world : (4, 4) array
        Vessel voxel index -> world (mm).
    extra_boxes : sequence of (N, 3) arrays
        World points (e.g. a parcellation's bounding-box corners) the grid must also cover.
    target_mm : float
        Working voxel size; never finer than the vessel grid.
    margin_mm : float
        Padding around the covered extent.

    Returns
    -------
    (shape, work_to_world)
        Shape of the working grid and its voxel index -> world matrix.
    """
    with using("cpu"):
        m_v = to_numpy(vessel_to_world).astype(float)[-4:, -4:]
        spacing = np.linalg.norm(m_v[:3, :3], axis=0)
        factor = np.maximum(float(target_mm) / spacing, 1.0)
        shape_v = np.array([int(s) for s in vessel_shape[-3:]], dtype=float)

        corners = [np.array([[i, j, k] for i in (0, shape_v[0] - 1) for j in (0, shape_v[1] - 1)
                             for k in (0, shape_v[2] - 1)], dtype=float)]
        inv_v = np.linalg.inv(m_v)
        for box in extra_boxes:
            pts = to_numpy(box).astype(float).reshape(-1, 3)
            if len(pts):
                corners.append((inv_v[:3, :3] @ pts.T).T + inv_v[:3, 3])
        pts = np.concatenate(corners, axis=0)
        pad = float(margin_mm) / spacing
        lo = np.floor(pts.min(axis=0) - pad)
        hi = np.ceil(pts.max(axis=0) + pad)
        shape = tuple(int(n) for n in np.floor((hi - lo) / factor).astype(int) + 1)

        # Working index w -> vessel index lo + w * factor -> world.
        step = np.eye(4)
        step[:3, :3] = np.diag(factor)
        step[:3, 3] = lo
        return shape, m_v @ step


def resample_labels(labels: Any, labels_to_world: Any, out_shape: Sequence[int], out_to_world: Any,
                    *, world_map: Any | None = None) -> Any:
    """
    Nearest-neighbour resampling of a label map onto another grid, through world space.

    *world_map* is an optional 4x4 world -> world transform applied to the labels first (a
    registration's moving -> fixed matrix), so ``labels`` placed by ``labels_to_world`` in the
    moving world land on ``out_to_world``'s grid in the fixed world.
    """
    with using("cpu"):
        m_src = to_numpy(labels_to_world).astype(float)[-4:, -4:]
        m_out = to_numpy(out_to_world).astype(float)[-4:, -4:]
        w = np.eye(4) if world_map is None else to_numpy(world_map).astype(float)[-4:, -4:]
        # out index -> world (fixed) -> world (moving) -> source index.
        index_map = np.linalg.inv(m_src) @ np.linalg.inv(w) @ m_out
    src = as_backend_array(labels)
    out = ndi.affine_transform(
        src.astype(np.int32),
        as_backend_array(index_map[:3, :3]),
        offset=as_backend_array(index_map[:3, 3]),
        output_shape=tuple(int(s) for s in out_shape),
        order=0,
        mode="constant",
        cval=0,
    )
    return out.astype(np.int32)


def labels_world_box(labels: Any, labels_to_world: Any, *, world_map: Any | None = None) -> Any:
    """World corners (8, 3) of the bounding box of *labels*' non-zero voxels (empty if none)."""
    lab = as_backend_array(labels) > 0
    lo, hi = [], []
    for axis in range(3):
        # Per-axis projections: no coordinate list of a 100-million-voxel volume.
        hits = to_numpy(np.flatnonzero(lab.any(axis=tuple(a for a in range(3) if a != axis))))
        if hits.size == 0:
            with using("cpu"):
                return np.zeros((0, 3))
        lo.append(float(hits[0]))
        hi.append(float(hits[-1]))
    with using("cpu"):
        corners = np.array([[i, j, k, 1.0] for i in (lo[0], hi[0]) for j in (lo[1], hi[1])
                            for k in (lo[2], hi[2])], dtype=float)
        m = to_numpy(labels_to_world).astype(float)[-4:, -4:]
        if world_map is not None:
            m = to_numpy(world_map).astype(float)[-4:, -4:] @ m
        return (m @ corners.T).T[:, :3]


def vessels_on_grid(vessels: Any, vessel_to_world: Any, out_shape: Sequence[int], out_to_world: Any,
                    labels: Sequence[int] | None = None) -> Any:
    """
    Vessel labels splatted onto a (coarser) grid: every vessel voxel marks the working voxel it
    falls in, so a vessel thinner than the working voxel is not lost the way resampling would
    lose it. Where two vessels share a working voxel the higher label id wins (deterministic on
    every backend: fancy assignment with repeated indices is not). *labels* restricts the
    splat to those ids (default: every non-zero label).
    """
    lab = as_backend_array(vessels)
    with using("cpu"):
        index_map = np.linalg.inv(to_numpy(out_to_world).astype(float)[-4:, -4:]) @ \
            to_numpy(vessel_to_world).astype(float)[-4:, -4:]
    out = np.zeros(tuple(int(s) for s in out_shape), dtype=np.int32)
    keep = lab > 0
    if labels is not None:
        top = int(lab.max()) + 1
        wanted = np.zeros(max(top, 1), dtype=bool)
        ids = [int(v) for v in labels if 0 < int(v) < top]
        if ids:
            wanted[as_backend_array(ids)] = True
        keep &= wanted[np.clip(lab, 0, top - 1)]
    idx = np.argwhere(keep)
    if int(idx.shape[0]) == 0:
        return out
    vals = lab[tuple(idx.T)].astype(np.int32)
    # Element-wise 3x3 product, not a matmul: no cuBLAS on the GPU backend, which fails once
    # TensorFlow (ANTsPyNet) has been loaded into the same process.
    coords = idx.astype(np.float64)
    w = np.stack([
        np.rint(sum(coords[:, c] * float(index_map[r, c]) for c in range(3)) + float(index_map[r, 3]))
        for r in range(3)
    ], axis=1).astype(np.int64)
    upper = as_backend_array([int(s) - 1 for s in out_shape])
    inside = np.all((w >= 0) & (w <= upper), axis=1)
    w, vals = w[inside], vals[inside]
    if int(vals.shape[0]) == 0:
        return out
    # One write per working voxel: sort by (voxel, label) and keep each voxel's last entry.
    dims = [int(s) for s in out_shape]
    lin = (w[:, 0] * dims[1] + w[:, 1]) * dims[2] + w[:, 2]
    order = np.argsort(lin * (int(vals.max()) + 1) + vals.astype(np.int64))
    lin, vals = lin[order], vals[order]
    last = np.ones(lin.shape, dtype=bool)
    last[:-1] = lin[1:] != lin[:-1]
    out.ravel()[lin[last]] = vals[last]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Territories
# ──────────────────────────────────────────────────────────────────────────────


def nearest_vessel_map(seeds: Any, spacing: Sequence[float]) -> Any:
    """Label of the nearest non-zero voxel of *seeds*, for every voxel (Euclidean, in mm)."""
    seed = as_backend_array(seeds)
    if not bool((seed > 0).any()):
        return np.zeros(seed.shape, dtype=np.int32)
    _, idx = ndi.distance_transform_edt(seed == 0, sampling=tuple(float(s) for s in spacing),
                                        return_indices=True)
    return seed[tuple(idx)].astype(np.int32)


def territory_map(
    vessel_seeds: Any,
    parenchyma: Any,
    spacing: Sequence[float],
    *,
    seed_sides: Mapping[int, str],
    voxel_sides: Any | None = None,
) -> Any:
    """
    Nearest-seed territory of every parenchyma voxel (0 outside the parenchyma).

    Parameters
    ----------
    vessel_seeds : array
        Seed vessel labels on the working grid (0 elsewhere); only these compete.
    parenchyma : array of bool
        Voxels to assign.
    spacing : sequence of float
        Working voxel size per axis (mm).
    seed_sides : mapping
        ``{seed label: "L" / "R" / ""}``.
    voxel_sides : array of int8, optional
        Per-voxel hemisphere (-1 left, +1 right, 0 either). When given, a lateralized voxel only
        competes between its own side's seeds and the unpaired ones (basilar, ACom...).
    """
    seeds = as_backend_array(vessel_seeds).astype(np.int32)
    brain = as_backend_array(parenchyma).astype(bool)
    out = np.zeros(seeds.shape, dtype=np.int32)
    if voxel_sides is None:
        out[brain] = nearest_vessel_map(seeds, spacing)[brain]
        return out

    sides = as_backend_array(voxel_sides)
    for code, keep in ((0, ("L", "R", "")), (-1, ("L", "")), (1, ("R", ""))):
        where = brain & (sides == code)
        if not bool(where.any()):
            continue
        subset = seeds.copy()
        for lid, side in seed_sides.items():
            if side not in keep:
                subset[subset == int(lid)] = 0
        if not bool((subset > 0).any()):
            subset = seeds  # no vessel on that side at all: fall back to every seed
        out[where] = nearest_vessel_map(subset, spacing)[where]
    return out


@dataclass
class SupplyTable:
    """
    Voxel counts of every (region, seed vessel) pair, for instant per-vessel queries.

    ``counts[r, v]`` is how many parenchyma voxels of ``region_ids[r]`` lie in the cell of
    ``seed_ids[v]``; ``sizes[r]`` is the region's parenchyma voxel count.
    """

    region_ids: tuple[int, ...]
    seed_ids: tuple[int, ...]
    counts: Any
    sizes: Any
    tree: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def region_fractions(self, label_id: int) -> dict[int, float]:
        """``{region id: share of the region in label_id's territory}``, non-zero shares only."""
        with using("cpu"):
            cols = [self.seed_ids.index(s) for s in downstream(label_id, self.tree) if s in self.seed_ids]
            if not cols:
                return {}
            supplied = to_numpy(self.counts)[:, cols].sum(axis=1)
            sizes = to_numpy(self.sizes)
            out: dict[int, float] = {}
            for r, rid in enumerate(self.region_ids):
                if supplied[r] > 0 and sizes[r] > 0:
                    out[int(rid)] = float(supplied[r]) / float(sizes[r])
            return out

    def irrigated(self, label_id: int, min_fraction: float) -> list[tuple[int, float]]:
        """Regions at least *min_fraction* in *label_id*'s territory, largest share first."""
        shares = self.region_fractions(label_id)
        kept = [(rid, f) for rid, f in shares.items() if f >= float(min_fraction)]
        return sorted(kept, key=lambda item: -item[1])


def supply_table(territories: Any, parcellation: Any, seed_ids: Sequence[int],
                 tree: Mapping[int, Sequence[int]]) -> SupplyTable:
    """Count every (region, seed) pair of the parenchyma in one pass."""
    terr = as_backend_array(territories).astype(np.int64)
    parc = as_backend_array(parcellation).astype(np.int64)
    inside = (terr > 0) & (parc > 0)
    seeds = tuple(int(s) for s in seed_ids)
    with using("cpu"):
        regions = tuple(int(r) for r in to_numpy(np.unique(parc[parc > 0])))
    n_r, n_v = len(regions), len(seeds)
    if not n_r or not n_v:
        with using("cpu"):
            return SupplyTable(regions, seeds, np.zeros((n_r, n_v), dtype=np.int64),
                               np.zeros(n_r, dtype=np.int64), dict(tree))

    # Dense index lookups (labels can be sparse, e.g. aparc 2035).
    r_lut = np.full(int(max(regions)) + 1, -1, dtype=np.int64)
    r_lut[as_backend_array(list(regions))] = np.arange(n_r, dtype=np.int64)
    v_lut = np.full(int(max(seeds)) + 1, -1, dtype=np.int64)
    v_lut[as_backend_array(list(seeds))] = np.arange(n_v, dtype=np.int64)

    r_idx = r_lut[parc[inside]]
    t_vals = terr[inside]
    v_idx = v_lut[np.clip(t_vals, 0, v_lut.shape[0] - 1)]
    v_idx = np.where(t_vals < v_lut.shape[0], v_idx, -1)
    ok = (r_idx >= 0) & (v_idx >= 0)
    counts = np.bincount(r_idx[ok] * n_v + v_idx[ok], minlength=n_r * n_v).reshape(n_r, n_v)
    region_vox = r_lut[parc[parc > 0]]
    sizes = np.bincount(region_vox[region_vox >= 0], minlength=n_r)
    with using("cpu"):
        return SupplyTable(regions, seeds, to_numpy(counts), to_numpy(sizes),
                           {int(k): tuple(int(x) for x in v) for k, v in tree.items()})


# ──────────────────────────────────────────────────────────────────────────────
# One call
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TerritoryResult:
    """Everything the brain-vascular map draws and queries, on the working grid."""

    shape: tuple[int, int, int]
    work_to_world: Any
    parcellation: Any
    parenchyma: Any
    territories: Any
    classes: dict[int, VesselClass]
    table: SupplyTable
    #: Vessel labels present in the volume but not classified (veins, unknown names): drawn,
    #: never part of a territory.
    unclassified: tuple[int, ...] = ()


def map_territories(
    vessels: Any,
    vessel_to_world: Any,
    vessel_names: Mapping[int, str],
    parcellation: Any,
    parcellation_to_world: Any,
    region_sides: Mapping[int, str],
    parenchyma_ids: Sequence[int] | None = None,
    *,
    world_map: Any | None = None,
    target_mm: float = 1.0,
    follow_collaterals: bool = False,
    respect_hemispheres: bool = True,
    relabel: Callable[[Any, Sequence[float]], Any] | None = None,
) -> TerritoryResult:
    """
    Vascular territories of a vessel segmentation over a parcellation.

    Parameters
    ----------
    vessels : array
        3D vessel label map.
    vessel_to_world : (4, 4) array
        Its voxel index -> world (mm).
    vessel_names : mapping
        ``{label id: name}``; see :func:`classify_vessel`. When no label is recognised every
        label is a seed and nothing feeds anything (an unnamed label set still maps).
    parcellation : array
        3D brain parcellation, anywhere in the same world (or the moving world of *world_map*).
    parcellation_to_world : (4, 4) array
        Its voxel index -> world (mm).
    region_sides : mapping
        ``{region id: "L" / "R" / ""}`` for the hemisphere constraint.
    parenchyma_ids : sequence of int, optional
        Region ids that count as tissue; default every non-zero id.
    world_map : (4, 4) array, optional
        Parcellation world -> vessel world (a registration result).
    target_mm : float
        Working voxel size (mm).
    follow_collaterals, respect_hemispheres : bool
        See :func:`supply_tree` and :func:`territory_map`.
    relabel : callable, optional
        ``relabel(labels, spacing)`` applied to the parcellation once it is on the working grid,
        e.g. :func:`nvitk.segmentation.brain_parcellation.lobes_from_aparc`: a coarser scheme
        derived at the working resolution instead of on a 100-million-voxel source grid.
    """
    vessel_arr = as_backend_array(vessels)
    present = [int(v) for v in to_numpy(np.unique(vessel_arr)) if int(v) > 0]
    if not present:
        raise ValueError("The vessel mask has no labels.")

    # ---- 1. Identity and supply tree ------------------------------------------------
    classes = classify_labels({lid: vessel_names.get(lid, "") for lid in present})
    if classes:
        unclassified = tuple(lid for lid in present if lid not in classes)
        seeds = [lid for lid, c in classes.items() if c.is_seed]
        tree = supply_tree(classes, follow_collaterals=follow_collaterals)
    else:
        log.warning("No vessel label is named after an artery: every label is its own territory, "
                    "with no supply tree.")
        unclassified = ()
        classes = {lid: VesselClass("", "", 0, 0) for lid in present}
        seeds = list(present)
        tree = {lid: () for lid in present}
    if not seeds:
        raise ValueError("No vessel with a parenchymal territory (only conduits / extracranial labels).")

    # ---- 2. Working grid covering the vessels and the whole parcellation -------------
    box = labels_world_box(parcellation, parcellation_to_world, world_map=world_map)
    shape, work_to_world = working_grid(vessel_arr.shape, vessel_to_world, [box], target_mm=target_mm)
    with using("cpu"):
        spacing = tuple(float(s) for s in np.linalg.norm(to_numpy(work_to_world)[:3, :3], axis=0))
    log.info(f"Territories: working grid {shape} at {', '.join(f'{s:.2f}' for s in spacing)} mm, "
             f"{len(seeds)} seed vessel(s) of {len(present)}")

    parc = resample_labels(parcellation, parcellation_to_world, shape, work_to_world, world_map=world_map)
    if relabel is not None:
        parc = as_backend_array(relabel(parc, spacing)).astype(np.int32)
    top = int(parc.max()) + 1
    if parenchyma_ids is None:
        brain = parc > 0
    else:
        tissue = np.zeros(top, dtype=bool)
        ids = [int(r) for r in parenchyma_ids if 0 < int(r) < top]
        if ids:
            tissue[as_backend_array(ids)] = True
        brain = tissue[parc]
    if not bool(brain.any()):
        raise ValueError("The parcellation does not overlap the vessel space: check the registration.")

    # ---- 3. Nearest-seed territories --------------------------------------------------
    seed_grid = vessels_on_grid(vessel_arr, vessel_to_world, shape, work_to_world, labels=seeds)
    voxel_sides = None
    sided = {int(r): s for r, s in region_sides.items() if s in ("L", "R") and 0 < int(r) < top}
    if respect_hemispheres and sided:
        side_lut = np.zeros(top, dtype=np.int8)
        for code in (-1, 1):
            ids = [r for r, s in sided.items() if s == ("L" if code < 0 else "R")]
            if ids:
                side_lut[as_backend_array(ids)] = code
        voxel_sides = side_lut[parc]
    terr = territory_map(seed_grid, brain, spacing, voxel_sides=voxel_sides,
                         seed_sides={lid: classes[lid].side for lid in seeds})

    # ---- 4. Region x vessel supply ------------------------------------------------------
    parc_brain = np.where(brain, parc, 0)
    table = supply_table(terr, parc_brain, seeds, tree)
    return TerritoryResult(
        shape=shape,
        work_to_world=work_to_world,
        parcellation=parc_brain,
        parenchyma=brain,
        territories=terr,
        classes=classes,
        table=table,
        unclassified=unclassified,
    )


__all__ = [
    "COLLATERAL_FAMILIES",
    "CONDUIT_FAMILIES",
    "EXTRACEREBRAL_FAMILIES",
    "FAMILIES",
    "SupplyTable",
    "TerritoryResult",
    "VesselClass",
    "classify_labels",
    "classify_vessel",
    "downstream",
    "labels_world_box",
    "map_territories",
    "nearest_vessel_map",
    "resample_labels",
    "supply_table",
    "supply_tree",
    "territory_map",
    "vessels_on_grid",
    "working_grid",
]
