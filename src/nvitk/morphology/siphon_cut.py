"""ICA siphon loop repair by a local minimum cut.

Problem
-------
On TOF MRA the two limbs of the cavernous ICA siphon often touch by partial volume, closing the
hook into a loop (:math:`\\beta_1 \\ge 1`). Centerlines then shortcut across the contact. The
earlier correction (removed 2026-10-23) re-thresholded each ICA with its own Otsu threshold,
eroded it and cut a ball at a heuristic anchor: it shrank the whole vessel, and the two sides
differently. The entry point is :func:`nvitk.morphology.centerline_siphon.correct_siphon_centerlines`.

Here the segmentation's boundary is kept and only the contact is removed:

1. :func:`pretess` repairs the digital topology. It removes edge- and corner-only contacts, which
   are loops under 26-connectivity and not under 6-connectivity, the way FreeSurfer's
   ``mri_pretess`` prepares a label for tessellation. FreeSurfer adds a voxel to make such a
   contact face-connected; here the darker voxel goes by default, since a siphon contact is the
   artefact.
2. :func:`loop_candidates` finds the skeleton cycles of at least ``min_loop_mm``. Each cycle is
   split into a *bridge* arc (the shorter one between its anchors) and a *curl* arc. The anchors
   are the cycle's junctions, else its inferior / superior extremes in world space (from the
   affine, not the array's index axes).
3. For each loop, a **minimum s–t vertex cut** on the whole vessel. The ring is opened opposite
   the bridge: a slab of the vessel around the far loop position leaves the graph, and the loop
   just either side of it becomes the source and the sink, so every source → sink path runs
   through the bridge. Each voxel belongs to the loop position of its nearest skeleton voxel. A
   voxel costs its TOF intensity relative to the vessel's centerline intensity, times how close
   it is to a limb's centreline (the skeleton without the bridge arc) relative to the lumen
   radius there (:func:`voxel_weights`). Lumen around a centreline is expensive; a contact
   between two limbs is far from both centrelines and cheap, however deep inside the mask, and
   cheaper still when partial volume makes it dark. Solved with
   :func:`scipy.sparse.csgraph.maximum_flow` on a node-split graph, so the cut is a set of voxels
   separating the two sides under 26-adjacency. A broad contact (limbs touching along a sheet)
   is inside the graph, and has to be cut whole.
4. A cut is **accepted** only if :math:`\\beta_1` drops, the vessel keeps its number of
   components (a ring cut once stays connected), it costs at most ``max_cost_ratio`` of a
   one-voxel slice of the loop's lumen (a cut through healthy lumen costs about one slice), and
   the removed volume stays under ``max_cut_mm3`` and ``max_cut_fraction`` of the vessel. The attempts are, in order: only the
   bridge arc (± the vessel radius) cuttable; any voxel cuttable; windows at the cheapest loop
   positions. Every attempt is logged.

Digital topology: 26-connected foreground and 6-connected background everywhere (the Euler
characteristic, the components, the skeleton graph and the cut). After :func:`pretess` the mask has
no critical configuration, so 6- and 26-connectivity agree and marching cubes, which VMTK's
surface comes from, sees the same loops.

All lengths are in mm.

Backend
-------
``np`` / ``scipy`` / ``ndi`` come from ``setup(globals())``. The algorithms here are host-bound by
design — per-voxel topology tests on 3×3×3 patches, networkx skeleton graphs, SciPy's
``maximum_flow`` (no CuPy equivalent) — on small crops around one vessel, so the public functions
run under ``using("cpu")`` (:func:`_host`). They accept NumPy or CuPy inputs (:func:`to_numpy` at
entry) and return host arrays; :func:`repair_label` writes its in-place result back into a CuPy
label volume.
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import is_cupy_array, setup, using
from nvitk.core.exceptions import ValidationError
from nvitk.core.logger import Logger

setup(globals())

log = Logger()

#: Defaults of :func:`repair_loops` / :func:`correct_siphon_mincut`.
MIN_LOOP_MM = 4.0
#: A cut may cost at most this fraction of a one-voxel slice of the loop's lumen.
MAX_CUT_COST_RATIO = 0.6
MAX_CUT_MM3 = 20.0
MAX_CUT_FRACTION = 0.02
LOCAL_ATTEMPTS = 3
MAX_LOOPS = 8
#: |ΔL − ΔR| (volume change %, both ICAs) above which the run is flagged.
ASYMMETRY_TOLERANCE_PCT = 2.0
RESEGMENT_MODES = ("none", "common")
#: Pieces of at most this many voxels that pretess detaches from the vessel are dropped.
SPECK_VOXELS = 27
PRETESS_MODES = ("remove", "fill")

with using("cpu"):  # host constants: the structuring elements of the host-side labelling
    _CUBE = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity structure
    _FACE = ndi.generate_binary_structure(3, 1)  # 6-connectivity structure
_FWD13 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────


def _host(func: Any) -> Any:
    """Run *func* with the CPU backend (``np`` / ``scipy`` / ``ndi`` = NumPy / SciPy): this
    module's algorithms are host-bound (module docstring)."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with using("cpu"):
            return func(*args, **kwargs)

    return wrapper


def _host_bool(mask: Any) -> Any:
    """*mask* as a host bool array."""
    return to_numpy(mask).astype(bool, copy=False)


def _bbox(mask: np.ndarray, pad: int) -> tuple[slice, slice, slice] | None:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


@_host
def spacing_of(affine: Any = None, spacing: Sequence[float] | None = None) -> Any:
    """Voxel size in mm (host vector): the affine's column norms, else *spacing*, else 1."""
    if affine is not None:
        a = to_numpy(affine).astype(float)
        if a.shape == (4, 4):
            return np.linalg.norm(a[:3, :3], axis=0)
    if spacing is not None and len(spacing) >= 3:
        return to_numpy([float(s) for s in spacing[:3]]).astype(float)
    return np.ones(3)


@_host
def superior_axis(affine: Any = None) -> Any:
    """Per-index-axis change of the world superior coordinate (mm per voxel step).

    The third world coordinate is superior in both RAS and LPS, so this is the affine's third
    row; without an affine, index axis 2 is taken as superior."""
    if affine is not None:
        a = to_numpy(affine).astype(float)
        if a.shape == (4, 4):
            return a[2, :3].copy()
    return np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Betti:
    """Betti numbers of a 3D mask, 26-connected foreground / 6-connected background."""

    b0: int
    b1: int
    b2: int
    n_voxels: int

    def to_dict(self) -> dict[str, int]:
        return {"beta0": self.b0, "beta1": self.b1, "beta2": self.b2, "n_voxels": self.n_voxels}


@_host
def betti(mask: Any) -> Betti:
    """β₀ (components), β₁ (loops) and β₂ (cavities) of *mask*.

    β₁ = β₀ + β₂ − χ, with the Euler characteristic χ for a 26-connected foreground; cavities are
    the 6-connected background components that do not reach the outside."""
    from skimage.measure import euler_number

    m = _host_bool(mask)
    n = int(m.sum())
    if n == 0:
        return Betti(0, 0, 0, 0)
    box = _bbox(m, 1)
    c = np.pad(m[box], 1)
    _, b0 = ndi.label(c, structure=_CUBE)
    _, n_bg = ndi.label(~c, structure=_FACE)
    chi = int(euler_number(c, connectivity=3))
    b2 = max(0, int(n_bg) - 1)
    return Betti(int(b0), max(0, int(b0) + b2 - chi), b2, n)


# ──────────────────────────────────────────────────────────────────────────────
# pretess: critical configurations
# ──────────────────────────────────────────────────────────────────────────────

# Each critical configuration is a pair of voxels that touch only along an edge or at a corner:
# (offset of the first, offset of the second, the offsets of the other voxels of its 2×2 square
# or 2×2×2 cube). The pair is a critical configuration when both are set and all the others are
# not (foreground contact), or the reverse (background contact).


def _square_configs() -> list[tuple[tuple[int, ...], tuple[int, ...], list[tuple[int, ...]]]]:
    out = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        ea = tuple(int(i == a) for i in range(3))
        eb = tuple(int(i == b) for i in range(3))
        eab = tuple(x + y for x, y in zip(ea, eb))
        o = (0, 0, 0)
        out.append((o, eab, [ea, eb]))
        out.append((ea, eb, [o, eab]))
    return out


def _cube_configs() -> list[tuple[tuple[int, ...], tuple[int, ...], list[tuple[int, ...]]]]:
    corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    out = []
    for c in corners:
        anti = tuple(1 - v for v in c)
        if c < anti:
            out.append((c, anti, [x for x in corners if x not in (c, anti)]))
    return out


_SQUARES = _square_configs()
_CUBES = _cube_configs()


def _shifted(m: np.ndarray, off: tuple[int, ...]) -> np.ndarray:
    """``m[p + off]`` for every base position *p* of a 2×2×2 window (shape − 1)."""
    return m[tuple(slice(o, s - 1 + o) for o, s in zip(off, m.shape))]


@_host
def critical_configurations(mask: Any) -> list[dict[str, Any]]:
    """Every edge / corner-only contact of *mask* (host bool, small crop).

    Each entry: ``kind`` (``"fg"`` two set voxels touching only by an edge / corner, ``"bg"`` the
    same for two unset voxels), ``pair`` (their two coordinates) and ``others`` (the rest of their
    square / cube)."""
    m = np.pad(_host_bool(mask), [(0, 1)] * 3)  # the last layers get a window too
    if min(m.shape) < 2:
        return []
    found: list[dict[str, Any]] = []
    for table, is_square in ((_SQUARES, True), (_CUBES, False)):
        for p, q, others in table:
            A, B = _shifted(m, p), _shifted(m, q)
            rest = [_shifted(m, o) for o in others]
            any_rest = np.logical_or.reduce(rest)
            all_rest = np.logical_and.reduce(rest)
            for kind, hit in (("fg", A & B & ~any_rest), ("bg", ~A & ~B & all_rest)):
                if is_square and kind == "bg":
                    continue  # a 2D diagonal is its own complement: the fg entry covers it
                for base in np.argwhere(hit):
                    b = tuple(int(v) for v in base)
                    found.append({
                        "kind": kind,
                        "pair": [tuple(x + y for x, y in zip(b, p)), tuple(x + y for x, y in zip(b, q))],
                        "others": [tuple(x + y for x, y in zip(b, o)) for o in others],
                    })
    return found


with using("cpu"):
    _N18 = np.ones((3, 3, 3), dtype=bool)
    _N18[::2, ::2, ::2] = False  # drop the 8 corners
    _N18[1, 1, 1] = False
    _N26 = np.ones((3, 3, 3), dtype=bool)
    _N26[1, 1, 1] = False
_FACES6 = [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]


def _patch(m: np.ndarray, v: tuple) -> np.ndarray:
    """The 3×3×3 neighbourhood of *v* (outside the array counts as background)."""
    out = np.zeros((3, 3, 3), dtype=bool)
    lo = [max(0, c - 1) for c in v]
    hi = [min(s, c + 2) for c, s in zip(v, m.shape)]
    out[tuple(slice(a - c + 1, b - c + 1) for a, b, c in zip(lo, hi, v))] = m[tuple(slice(a, b) for a, b in zip(lo, hi))]
    return out


@_host
def topology_numbers(m: Any, v: tuple) -> tuple[int, int]:
    """Bertrand's topology numbers of voxel *v*: ``T26`` (26-components of the foreground in its
    26-neighbourhood) and ``T6bar`` (6-components of the background in its 18-neighbourhood that
    touch it by a face). A voxel is simple — removable or addable without changing the
    topology — iff both are 1. Removing a set voxel with ``T6bar != 1`` opens a tunnel or a
    cavity; adding an unset one with ``T26 != 1`` closes a loop or merges components."""
    p = _patch(m, v)
    _, t26 = ndi.label(p & _N26, structure=_CUBE)
    lab, _ = ndi.label(~p & _N18, structure=_FACE)
    t6 = len({int(lab[f]) for f in _FACES6 if lab[f] > 0})
    return int(t26), int(t6)


def _still_critical(m: np.ndarray, cfg: dict[str, Any]) -> bool:
    a, b = (bool(m[v]) for v in cfg["pair"])
    rest = [bool(m[v]) for v in cfg["others"]]
    if cfg["kind"] == "fg":
        return a and b and not any(rest)
    return (not a) and (not b) and all(rest)


@_host
def pretess(
    mask: Any,
    intensity: Any | None = None,
    *,
    mode: str = "remove",
    protect: Any | None = None,
    max_iter: int = 10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove the critical configurations of *mask* (a ``mri_pretess``-like step).

    A critical configuration is two voxels that touch only along an edge or at a corner. Two such
    set voxels are a loop or a link under 26-connectivity and nothing under 6-connectivity
    (marching cubes); two such unset voxels inside the vessel are the reverse, a pin-hole.

    A voxel is only removed when that opens no tunnel or cavity (``T6bar = 1``,
    :func:`topology_numbers`) and only added when that closes no loop (``T26 = 1``); a
    configuration with no such choice stays (``unresolved``). Specks of up to :data:`SPECK_VOXELS`
    voxels a removal detaches from their vessel are dropped (``specks_removed``). If the Betti
    numbers still got worse (more loops or cavities), the input is returned unchanged with
    ``reverted = True``.

    - ``mode="remove"`` (default, for siphon contacts): of a set pair, the darker voxel is removed
      (without intensity, the one nearer the wall). A pin-hole is filled with its brighter voxel.
      Nothing new is connected.
    - ``mode="fill"`` (FreeSurfer's choice for white matter): a set pair is joined by filling the
      brightest of its unset neighbours in the square / cube, and a pin-hole is filled as above.

    *protect* (bool mask) marks voxels that may not be filled, such as other labels. Returns the
    repaired mask (full shape, host bool) and ``{"n_removed", "n_filled", "unresolved",
    "iterations", "removed", "filled"}`` (coordinates in the full frame)."""
    if mode not in PRETESS_MODES:
        raise ValidationError(f"pretess mode must be one of {PRETESS_MODES}, got {mode!r}")
    full = _host_bool(mask).copy()
    info: dict[str, Any] = {"n_removed": 0, "n_filled": 0, "unresolved": 0, "iterations": 0,
                            "removed": [], "filled": [], "reverted": False, "specks_removed": 0}
    box = _bbox(full, 2)
    if box is None:
        return full, info
    m = full[box].copy()
    inten = None if intensity is None else to_numpy(intensity).astype(np.float64, copy=False)[box]
    prot = None if protect is None else _host_bool(protect)[box]
    origin = tuple(s.start for s in box)
    skip: set[tuple] = set()
    for it in range(int(max_iter)):
        cfgs = [c for c in critical_configurations(m) if (c["kind"], tuple(c["pair"])) not in skip]
        info["iterations"] = it + 1
        if not cfgs:
            break
        depth = ndi.distance_transform_edt(m) if inten is None else None

        def value(v: tuple) -> float:
            return float(inten[v]) if inten is not None else float(depth[v])  # type: ignore[index]

        def fill_target(cands: list[tuple]) -> tuple | None:
            free = [v for v in cands if not m[v] and (prot is None or not prot[v])
                    and topology_numbers(m, v)[0] == 1]
            return max(free, key=value) if free else None

        def removal_target(cands: list[tuple]) -> tuple | None:
            ok = [v for v in cands if topology_numbers(m, v)[1] == 1]
            return min(ok, key=value) if ok else None

        # Darkest decisions first: they are the most certain.
        cfgs.sort(key=lambda c: min(value(v) for v in c["pair"]))
        changed = 0
        for cfg in cfgs:
            if not _still_critical(m, cfg):
                continue
            if cfg["kind"] == "fg" and mode == "remove":
                v = removal_target(cfg["pair"])
                if v is None:
                    skip.add((cfg["kind"], tuple(cfg["pair"])))
                    continue
                m[v] = False
                info["removed"].append(v)
                changed += 1
                continue
            v = fill_target(cfg["pair"] if cfg["kind"] == "bg" else cfg["others"])
            if v is None:
                skip.add((cfg["kind"], tuple(cfg["pair"])))
                info["unresolved"] += 1
                continue
            m[v] = True
            info["filled"].append(v)
            changed += 1
        if not changed:
            break
    # A voxel joined to the vessel only by an edge / corner comes off with its contact: drop the
    # specks that removal detached (pieces that were part of a larger input component).
    lab_in, _ = ndi.label(full[box], structure=_CUBE)
    lab_out, n_out = ndi.label(m, structure=_CUBE)
    if n_out > 1:
        sizes = np.bincount(lab_out.ravel())
        owner = np.zeros(n_out + 1, dtype=np.int64)
        np.maximum.at(owner, lab_out.ravel(), lab_in.ravel())
        biggest: dict[int, int] = {}
        for c in range(1, n_out + 1):
            biggest[int(owner[c])] = max(biggest.get(int(owner[c]), 0), int(sizes[c]))
        specks = [c for c in range(1, n_out + 1) if sizes[c] <= SPECK_VOXELS and sizes[c] < biggest[int(owner[c])]]
        if specks:
            drop = np.isin(lab_out, specks)
            info["specks_removed"] = int(drop.sum())
            info["removed"] += [tuple(int(x) for x in v) for v in np.argwhere(drop)]
            m = m & ~drop
    before = betti(full)
    trial = full.copy()
    trial[box] = m
    after = betti(trial)
    info["betti_before"], info["betti_after"] = before.to_dict(), after.to_dict()
    if after.b1 > before.b1 or after.b2 > before.b2:
        info.update(reverted=True, n_removed=0, n_filled=0, removed=[], filled=[],
                    unresolved=len(critical_configurations(full[box])))
        return full, info
    full = trial
    shift = to_numpy(origin)
    info["removed"] = [tuple(int(x) for x in to_numpy(v) + shift) for v in info["removed"]]
    info["filled"] = [tuple(int(x) for x in to_numpy(v) + shift) for v in info["filled"]]
    info["n_removed"] = len(info["removed"])
    info["n_filled"] = len(info["filled"])
    info["unresolved"] = len(critical_configurations(m))
    return full, info


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton loops
# ──────────────────────────────────────────────────────────────────────────────


def _skeleton(m: np.ndarray) -> np.ndarray:
    from nvitk.morphology.centerline import skeletonize_binary

    return _host_bool(skeletonize_binary(m))


def _skeleton_graph(sk: np.ndarray, spacing: np.ndarray) -> Any:
    """26-connected graph of the skeleton voxels, edge weight ``w`` in mm, without the chords
    of 26-adjacency triangles (a diagonal edge with a shorter two-step path beside it)."""
    import networkx as nx

    G = nx.Graph()
    coords = [tuple(int(v) for v in row) for row in np.argwhere(sk)]
    nodes = set(coords)
    G.add_nodes_from(coords)
    for n in coords:
        for d in _FWD13:
            v = (n[0] + d[0], n[1] + d[1], n[2] + d[2])
            if v in nodes:
                G.add_edge(n, v, w=float(np.linalg.norm(to_numpy(d) * spacing)))
    drop = []
    for u, v, w in G.edges(data="w"):
        for x in set(G[u]) & set(G[v]):
            if G[u][x]["w"] < w - 1e-9 and G[x][v]["w"] < w - 1e-9:
                drop.append((u, v))
                break
    G.remove_edges_from(drop)
    return G


def _ordered_cycle(G: Any, cycle: list[tuple]) -> list[tuple]:
    """*cycle*'s nodes in walking order (``minimum_cycle_basis`` gives them unordered)."""
    cyc = set(cycle)
    start = cycle[0]
    order, prev, cur = [start], None, start
    while True:
        nxt = next((u for u in G[cur] if u in cyc and u != prev and (u not in order or u == start)), None)
        if nxt is None or nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(cycle):
            break
    return order if len(order) == len(cycle) else list(cycle)


@dataclass
class Loop:
    """A skeleton cycle (crop coordinates, walking order) split into bridge and curl arcs."""

    nodes: list[tuple]
    arc_mm: np.ndarray  # cumulative arc length at each node (mm)
    length_mm: float
    bridge: list[int]  # positions on ``nodes``
    curl: list[int]
    anchors: list[int]
    anchor_rule: str

    def to_dict(self, origin: Sequence[int] = (0, 0, 0)) -> dict[str, Any]:
        o = to_numpy(origin)
        seg = lambda idx: float(sum(self._step(i) for i in idx))  # noqa: E731
        return {
            "length_mm": round(float(self.length_mm), 2),
            "bridge_mm": round(seg(self.bridge), 2),
            "curl_mm": round(seg(self.curl), 2),
            "anchors": [[int(x) for x in to_numpy(self.nodes[i]) + o] for i in self.anchors],
            "anchor_rule": self.anchor_rule,
        }

    def _step(self, i: int) -> float:
        return float(self.arc_mm[i] - self.arc_mm[i - 1]) if i > 0 else float(self.length_mm - self.arc_mm[-1])


def _split_loop(G: Any, order: list[tuple], spacing: np.ndarray, sup: np.ndarray) -> Loop:
    pts = to_numpy(order).astype(float)
    step = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0) * spacing, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step[:-1])])
    length = float(step.sum())
    height = pts @ sup  # world superior coordinate (up to an offset)
    junctions = [i for i, n in enumerate(order) if G.degree(n) > 2]
    if len(junctions) >= 2:
        if len(junctions) > 2:
            hj = height[junctions]
            anchors, rule = [junctions[int(np.argmin(hj))], junctions[int(np.argmax(hj))]], "junctions (S extremes)"
        else:
            anchors, rule = list(junctions), "junctions"
    elif len(junctions) == 1:
        others = [i for i in range(len(order)) if i != junctions[0]]
        anchors, rule = [junctions[0], others[int(np.argmax(height[others]))]], "junction + most superior"
    else:
        anchors, rule = [int(np.argmin(height)), int(np.argmax(height))], "inferior + superior"
    a, b = sorted(anchors)
    if a == b:
        return Loop(list(order), arc, length, [], list(range(len(order))), [a], rule)
    arc1 = list(range(a + 1, b))
    arc2 = list(range(b + 1, len(order))) + list(range(0, a))

    def mm(idx: list[int]) -> float:
        return float(sum(step[i - 1] if i > 0 else step[-1] for i in idx)) + float(step[a] if idx else 0.0)

    bridge, curl = (arc1, arc2) if mm(arc1) <= mm(arc2) else (arc2, arc1)
    return Loop(list(order), arc, length, bridge, curl, [a, b], rule)


@_host
def loop_candidates(
    mask: Any,
    *,
    spacing: Sequence[float] | np.ndarray = (1.0, 1.0, 1.0),
    affine: Any | None = None,
    min_loop_mm: float = MIN_LOOP_MM,
) -> tuple[list[Loop], dict[str, Any]]:
    """The skeleton loops of *mask* at least *min_loop_mm* long, shortest first.

    Coordinates are those of *mask* (pass a crop to work on a crop). The info dict has the
    skeleton, its graph and the number of shorter cycles left alone."""
    import networkx as nx

    m = _host_bool(mask)
    sp = to_numpy(spacing).astype(float)
    sk = _skeleton(m)
    info: dict[str, Any] = {"skeleton": sk, "graph": None, "n_short_cycles": 0}
    if sk.sum() < 3:
        return [], info
    G = _skeleton_graph(sk, sp)
    info["graph"] = G
    core = nx.k_core(G, 2)
    if core.number_of_edges() == 0:
        return [], info
    try:
        cycles = nx.minimum_cycle_basis(core, weight="w")
    except Exception:  # noqa: BLE001
        cycles = nx.cycle_basis(core)
    sup = superior_axis(affine)
    loops: list[Loop] = []
    for c in cycles:
        order = _ordered_cycle(G, [tuple(n) for n in c])
        lp = _split_loop(G, order, sp, sup)
        if lp.length_mm >= float(min_loop_mm):
            loops.append(lp)
        else:
            info["n_short_cycles"] += 1
    loops.sort(key=lambda lp: lp.length_mm)
    return loops, info


# ──────────────────────────────────────────────────────────────────────────────
# Weights and the minimum cut
# ──────────────────────────────────────────────────────────────────────────────


@_host
def voxel_weights(
    mask: np.ndarray,
    intensity: np.ndarray | None,
    centres: np.ndarray,
    spacing: np.ndarray,
    depth: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Cost of removing each voxel of *mask*: ``clip(I / I_ref, 0.05, 4) · (0.25 + core)``.

    *centres* are the lumen centrelines: the skeleton without the bridge arc of the loop being
    cut. ``core = clip(1 − d_c / R, 0, 1)``, with ``d_c`` the voxel's distance to the nearest
    centre voxel and ``R`` the lumen radius there (its distance to the wall): 1 on a centreline,
    0 at the wall and anywhere farther from a lumen centre than the lumen's radius, such as a
    contact between two limbs however deep inside the mask. ``I_ref`` is the median intensity on
    the centres. Without intensity the first factor is 1."""
    if depth is None:
        depth = ndi.distance_transform_edt(mask, sampling=spacing)
    c = centres & mask
    if not c.any():
        c = mask & (depth >= depth[mask].max(initial=0.0) * 0.5)
    d_c, near = ndi.distance_transform_edt(~c, sampling=spacing, return_indices=True)
    R = np.maximum(depth[tuple(near)], float(spacing.min()))
    core = np.clip(1.0 - d_c / R, 0.0, 1.0)
    w = 0.25 + core
    ref = float("nan")
    if intensity is not None:
        vals = intensity[c]
        ref = float(np.median(vals)) if vals.size else float("nan")
        if np.isfinite(ref) and ref > 0:
            w = w * np.clip(intensity / ref, 0.05, 4.0)
    r = float(np.median(depth[c])) if c.any() else float(spacing.min())
    return np.where(mask, w, 0.0), {"radius_mm": max(r, float(spacing.min())), "intensity_ref": ref}


def _min_vertex_cut(
    region: np.ndarray,
    source: np.ndarray,
    sink: np.ndarray,
    weights: np.ndarray,
    uncuttable: np.ndarray | None = None,
    scale: float = 100.0,
) -> tuple[np.ndarray, float, bool]:
    """Minimum set of *region* voxels whose removal separates *source* from *sink* (26-adjacency).

    Node-split graph: ``in(v) → out(v)`` carries the voxel's cost, ``out(u) → in(v)`` is unbounded
    for 26-neighbours; the seeds and *uncuttable* voxels cannot be cut. Returns the cut (bool,
    *region*'s shape), its cost, and whether a finite cut exists."""
    # SciPy's max flow / BFS (no CuPy equivalent): always reached from a host region.
    maximum_flow = scipy.sparse.csgraph.maximum_flow
    breadth_first_order = scipy.sparse.csgraph.breadth_first_order

    idx = np.full(region.shape, -1, dtype=np.int64)
    coords = np.argwhere(region)
    n = len(coords)
    at = tuple(coords.T)
    idx[at] = np.arange(n)
    cap = np.maximum(np.rint(weights[at] * scale), 1).astype(np.int64)
    fixed = source[at] | sink[at]
    if uncuttable is not None:
        fixed |= uncuttable[at]
    inf = int(cap.sum() + 1)
    if inf >= np.iinfo(np.int32).max // 4:
        raise ValidationError("min-cut region too large for int32 capacities")
    cap[fixed] = inf
    src, snk = 2 * n, 2 * n + 1
    rows, cols, vals = [np.arange(n) * 2], [np.arange(n) * 2 + 1], [cap]
    for d in _FWD13:
        a = idx[tuple(slice(max(0, -o), s - max(0, o)) for o, s in zip(d, region.shape))]
        b = idx[tuple(slice(max(0, o), s - max(0, -o)) for o, s in zip(d, region.shape))]
        both = (a >= 0) & (b >= 0)
        u, v = a[both], b[both]
        rows += [2 * u + 1, 2 * v + 1]
        cols += [2 * v, 2 * u]
        vals += [np.full(len(u), inf), np.full(len(u), inf)]
    s_ids = np.flatnonzero(source[at])
    t_ids = np.flatnonzero(sink[at])
    rows += [np.full(len(s_ids), src), 2 * t_ids + 1]
    cols += [2 * s_ids, np.full(len(t_ids), snk)]
    vals += [np.full(len(s_ids), inf), np.full(len(t_ids), inf)]
    N = 2 * n + 2
    C = scipy.sparse.csr_matrix(
        (np.concatenate(vals).astype(np.int32), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    )
    C.sum_duplicates()
    res = maximum_flow(C, src, snk, method="dinic")
    if res.flow_value >= inf:
        return np.zeros(region.shape, dtype=bool), float("inf"), False
    flow = getattr(res, "flow", None)
    if flow is None:  # SciPy < 1.8
        flow = res.residual
    R = (C.astype(np.int64) - scipy.sparse.csr_matrix(flow).astype(np.int64)).tocsr()
    R.data[R.data < 0] = 0
    R.eliminate_zeros()
    reach = np.zeros(N, dtype=bool)
    reach[breadth_first_order(R, src, directed=True, return_predecessors=False)] = True
    cut_ids = np.flatnonzero(reach[0:2 * n:2] & ~reach[1:2 * n:2])
    cut = np.zeros(region.shape, dtype=bool)
    cut[tuple(coords[cut_ids].T)] = True
    return cut, float(res.flow_value) / scale, True


@dataclass
class _Context:
    m: np.ndarray
    weights: np.ndarray
    arc: np.ndarray  # per voxel: arc position (mm) on the loop of its nearest skeleton voxel, NaN off the loop
    depth: np.ndarray  # distance to the wall (mm)
    spacing: np.ndarray
    radius_mm: float


def _attempt(ctx: _Context, loop: Loop, centre: float, half: float | None, where: str) -> dict[str, Any]:
    """One cut attempt on *loop*: the ring is opened opposite *centre*, so every source → sink
    path runs through *centre*. The opening removes the vessel's whole cross-section around the
    loop position ``centre + L/2`` (a ball one local radius plus a voxel wide, and the voxels of
    the loop positions within 1 mm); the loop positions just beyond it are the source and the
    sink. Only voxels within *half* mm of *centre* along the loop, or off the loop, may be cut
    (``None``: any voxel)."""
    L = float(loop.length_mm)
    sp = ctx.spacing
    vox = float(sp.max())
    seed = max(1.0, 2.0 * vox)
    out: dict[str, Any] = {"where": where, "centre_mm": round(float(np.mod(centre, L)), 2),
                           "half_mm": None if half is None else round(float(half), 2)}
    on = np.isfinite(ctx.arc)
    arc = np.where(on, ctx.arc, 0.0)
    opening = centre + L / 2
    j = int(np.argmin(np.abs(np.mod(loop.arc_mm - opening + L / 2, L) - L / 2)))
    node = to_numpy(loop.nodes[j]).astype(float)
    ball_r = float(ctx.depth[loop.nodes[j]]) + 1.5 * vox
    grid = np.indices(ctx.m.shape, sparse=True)
    ball = sum(((g - c) * s) ** 2 for g, c, s in zip(grid, node, sp)) <= ball_r ** 2
    delta = max(1.0, ball_r + vox)
    rel = np.mod(arc - opening + L / 2, L) - L / 2  # signed distance from the opening along the loop
    slab = ball | (on & (np.abs(rel) <= 1.0))
    src = ctx.m & on & ~slab & (rel > delta) & (rel <= delta + seed)
    snk = ctx.m & on & ~slab & (rel < -delta) & (rel >= -delta - seed)
    out["opening_mm"] = round(float(np.mod(opening, L)), 2)
    if not src.any() or not snk.any():
        out.update(accepted=False, reason="loop too short to open")
        return out
    uncut = None
    if half is not None:
        limit = L / 2 - delta - seed - float(sp.max())
        half = min(float(half), limit)
        if half < float(sp.max()):
            out.update(accepted=False, reason="loop too short for a window")
            return out
        out["half_mm"] = round(half, 2)
        dc = np.abs(np.mod(arc - centre + L / 2, L) - L / 2)
        uncut = on & (dc > half)
    region = ctx.m & ~slab
    cut, cost, finite = _min_vertex_cut(region, src, snk, ctx.weights, uncut)
    if not finite:
        out.update(accepted=False, reason="no cut within the window")
        return out
    out.update(cut=cut, cost=round(cost, 3), n_cut=int(cut.sum()))
    return out


#: A loop position is a contact when its (intensity × radius) relative to the curl's falls below
#: CONTACT_SCORE; the contact run extends over the neighbours below CONTACT_SCORE_EXTEND.
CONTACT_SCORE = 0.6
CONTACT_SCORE_EXTEND = 0.8


@_host
def contact_run(loop: Loop, depth: np.ndarray, intensity: np.ndarray | None) -> tuple[list[int], dict[str, Any]]:
    """The loop positions where the limbs touch: a dip of ``(I / I_curl) · (r / r_curl)``.

    ``r`` is the inscribed radius (distance to the wall) and ``I`` the TOF intensity at each loop
    node, smoothed over ±1 node; ``*_curl`` are their medians on the curl arc. A contact is
    narrower than the lumen (the limbs' walls pinch it) and darker (partial volume). The run is
    looked for on the bridge arc first, then on the whole loop; it is the minimum below
    :data:`CONTACT_SCORE` grown over its neighbours below :data:`CONTACT_SCORE_EXTEND`. Without
    such a dip the run is empty and no centreline is taken out of the cost."""
    n = len(loop.nodes)
    r = np.array([depth[v] for v in loop.nodes], dtype=float)
    smooth = lambda x: (np.roll(x, 1) + x + np.roll(x, -1)) / 3.0  # noqa: E731
    ref_idx = loop.curl or list(range(n))
    score = smooth(r) / max(float(np.median(r[ref_idx])), 1e-6)
    if intensity is not None:
        iv = np.array([intensity[v] for v in loop.nodes], dtype=float)
        i_ref = float(np.median(iv[ref_idx]))
        if i_ref > 0:
            score = score * smooth(iv) / i_ref
    info: dict[str, Any] = {"min_score": None, "where": None, "n_nodes": 0}
    for where, cand in (("bridge", loop.bridge), ("loop", list(range(n)))):
        if not cand:
            continue
        i0 = min(cand, key=lambda i: score[i])
        if score[i0] >= CONTACT_SCORE:
            continue
        allowed = set(cand)
        run = [i0]
        for step in (1, -1):
            j = (i0 + step) % n
            while j in allowed and score[j] < CONTACT_SCORE_EXTEND and j not in run and len(run) < n // 2:
                run.append(j)
                j = (j + step) % n
        run.sort(key=lambda i: (i - i0 + n // 2) % n)  # walking order around i0
        info.update(min_score=round(float(score[i0]), 3), where=where, n_nodes=len(run))
        return run, info
    info["min_score"] = round(float(score[min(loop.bridge or range(n), key=lambda i: score[i])]), 3)
    return [], info


@_host
def repair_loops(
    mask: Any,
    intensity: Any | None = None,
    *,
    affine: Any | None = None,
    spacing: Sequence[float] | None = None,
    min_loop_mm: float = MIN_LOOP_MM,
    max_cost_ratio: float = MAX_CUT_COST_RATIO,
    max_cut_mm3: float = MAX_CUT_MM3,
    max_cut_fraction: float = MAX_CUT_FRACTION,
    local_attempts: int = LOCAL_ATTEMPTS,
    max_loops: int = MAX_LOOPS,
    label_name: str = "vessel",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cut the loops of *mask* at their cheapest contact (module docstring).

    Loops are taken longest first (a long loop's bridge window often holds the contacts of the
    short loops beside it). Per loop the contact is located first (:func:`contact_run`); its nodes
    are taken out of the lumen centrelines that make lumen expensive (:func:`voxel_weights`).
    Then, in order until one is accepted:

    1. *contact window* — only the contact run (± the vessel radius) may be cut; without a
       contact dip, *bridge window*: the bridge arc;
    2. *anywhere* — the cheapest cut anywhere on the loop away from the opening;
    3. *local windows* at the *local_attempts* cheapest loop positions (cross-section cost),
       bridge positions first.

    Returns the repaired mask, the removed voxels (both full shape, host bool) and a JSON-able log:
    Betti numbers before / after, and per loop its arcs and every attempt (window, voxels cut, mm³,
    cost, β₁ after, accepted or why not)."""
    full = _host_bool(mask).copy()
    sp = spacing_of(affine, spacing)
    vox_mm3 = float(np.prod(sp))
    removed = np.zeros_like(full)
    before = betti(full)
    report: dict[str, Any] = {"label_name": label_name, "before": before.to_dict(), "loops": [],
                              "min_loop_mm": float(min_loop_mm), "max_cost_ratio": float(max_cost_ratio),
                              "max_cut_mm3": float(max_cut_mm3), "max_cut_fraction": float(max_cut_fraction)}
    box = _bbox(full, 3)
    if box is None or before.b1 == 0:
        report.update(after=before.to_dict(), removed_mm3=0.0, n_cut_loops=0, unresolved=0)
        return full, removed, report
    origin = tuple(s.start for s in box)
    m = full[box].copy()
    inten = None if intensity is None else to_numpy(intensity).astype(np.float64, copy=False)[box]
    max_cut = min(float(max_cut_mm3), float(max_cut_fraction) * before.n_voxels * vox_mm3)
    report["max_cut_effective_mm3"] = round(max_cut, 3)
    failed: set[tuple] = set()
    n_cut = 0
    t0 = time.time()
    for _ in range(int(max_loops)):
        loops, linfo = loop_candidates(m, spacing=sp, affine=affine, min_loop_mm=min_loop_mm)
        loops = [lp for lp in loops if sum(n in failed for n in lp.nodes) < 0.5 * len(lp.nodes)]
        loops.sort(key=lambda lp: -lp.length_mm)
        if not loops:
            break
        sk = linfo["skeleton"]
        near_idx = ndi.distance_transform_edt(~sk, sampling=sp, return_distances=False, return_indices=True)
        depth = ndi.distance_transform_edt(m, sampling=sp)
        cur = betti(m)
        success = False
        for lp in loops:
            L = float(lp.length_mm)
            pos_of = np.full(m.shape, -1, dtype=np.int64)
            for i, n in enumerate(lp.nodes):
                pos_of[n] = i
            run, cinfo = contact_run(lp, depth, inten)
            centres = sk.copy()
            for i in run:
                centres[lp.nodes[i]] = False
            weights, winfo = voxel_weights(m, inten, centres, sp, depth)
            pos = np.where(m, pos_of[tuple(near_idx)], -1)
            arc = np.where(pos >= 0, lp.arc_mm[np.clip(pos, 0, None)], np.nan)
            ctx = _Context(m, weights, arc, depth, sp, float(winfo["radius_mm"]))
            entry = {**lp.to_dict(origin), "radius_mm": round(ctx.radius_mm, 2), "contact": cinfo,
                     "attempts": [], "resolved": False}
            margin = max(ctx.radius_mm, 1.0)
            window = run or lp.bridge
            if window:
                start = float(lp.arc_mm[window[0]])
                span = float(np.mod(lp.arc_mm[window[-1]] - start, L))
                centre, half_first = start + span / 2, span / 2 + margin
            else:
                centre, half_first = float(lp.arc_mm[lp.anchors[0]]), margin
            windows: list[tuple[float, float | None, str]] = [
                (centre, half_first, "contact window" if run else "bridge window"), (centre, None, "anywhere")]
            # Local windows at the cheapest positions (cross-section cost), bridge first.
            on = pos >= 0
            cell_cost = np.bincount(pos[on], weights=weights[on], minlength=len(lp.nodes))
            bridge_set = set(lp.bridge)
            # A one-voxel slice of the lumen: the curl's median cost per mm × the voxel size.
            steps = np.array([lp._step(i) for i in range(len(lp.nodes))])
            per_mm = cell_cost / np.maximum(steps, 1e-6)
            healthy = [i for i in (lp.curl or range(len(lp.nodes))) if steps[i] > 0]
            slice_cost = float(np.median(per_mm[healthy])) * float(sp.min()) if healthy else float("nan")
            entry["slice_cost"] = round(slice_cost, 3)
            half = max(2.0 * ctx.radius_mm, 1.5)
            picked: list[int] = []
            for i in sorted(range(len(lp.nodes)), key=lambda i: (i not in bridge_set, cell_cost[i])):
                if len(picked) >= int(local_attempts):
                    break
                if all(abs(np.mod(lp.arc_mm[i] - lp.arc_mm[j] + L / 2, L) - L / 2) > half for j in picked):
                    picked.append(i)
            windows += [(float(lp.arc_mm[i]), half, "bridge position" if i in bridge_set else "curl position")
                        for i in picked]
            for c, h, where in windows:
                att = _attempt(ctx, lp, c, h, where)
                cut = att.pop("cut", None)
                if cut is not None:
                    trial = m & ~cut
                    after = betti(trial)
                    cut_mm3 = float(cut.sum() * vox_mm3)
                    ratio = float(att["cost"]) / slice_cost if slice_cost > 0 else float("nan")
                    att.update(cut_mm3=round(cut_mm3, 3), cost_ratio=round(ratio, 3), beta1_after=after.b1,
                               components_after=after.b0)
                    reasons = []
                    if not ratio <= max_cost_ratio:
                        reasons.append(f"cost {ratio:.2f} × a lumen slice > {max_cost_ratio:g}")
                    if after.b1 >= cur.b1:
                        reasons.append("β₁ did not drop")
                    if after.b0 != cur.b0:
                        reasons.append(f"components {cur.b0} → {after.b0}")
                    if cut_mm3 > max_cut + 1e-9:
                        reasons.append(f"cut {cut_mm3:.2f} mm³ > {max_cut:.2f} mm³")
                    att["accepted"] = not reasons
                    if reasons:
                        att["reason"] = "; ".join(reasons)
                    else:
                        att["at_voxel"] = [round(float(v), 1) for v in np.argwhere(cut).mean(0) + to_numpy(origin)]
                        m = trial
                        removed[box] |= cut
                        entry["resolved"] = True
                        n_cut += 1
                        success = True
                entry["attempts"].append(att)
                if success:
                    break
            report["loops"].append(entry)
            if success:
                break
            failed.update(lp.nodes)
            log.warning(f"[{label_name}] loop of {L:.1f} mm not cut: "
                        + "; ".join(f"{a['where']}: {a.get('reason', '')}" for a in entry["attempts"]))
        if not success:
            break
    full[box] = m
    after = betti(full)
    report.update(after=after.to_dict(), removed_mm3=round(float(removed.sum() * vox_mm3), 3), n_cut_loops=n_cut,
                  unresolved=sum(1 for e in report["loops"] if not e["resolved"]), seconds=round(time.time() - t0, 2))
    log.step(f"[{label_name}] loops: β₁ {before.b1} → {after.b1}, {n_cut} cut, "
             f"{report['removed_mm3']:.2f} mm³ removed")
    return full, removed, report


# ──────────────────────────────────────────────────────────────────────────────


@_host
def siphon_centerline(
    mask: Any, *, affine: Any | None = None, spacing: Sequence[float] | None = None, min_loop_mm: float = 0.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Centerline of an ICA mask from its most inferior skeleton end to its most superior one
    (world superior axis), as ``(N, 3)`` voxel coordinates. A loop left in the mask is opened by
    dropping its bridge arc, as the legacy correction does."""
    import networkx as nx

    full = _host_bool(mask)
    sp = spacing_of(affine, spacing)
    info: dict[str, Any] = {}
    box = _bbox(full, 2)
    if box is None:
        return np.empty((0, 3), np.float32), {"error": "empty mask"}
    origin = to_numpy([s.start for s in box])
    m = full[box]
    loops, linfo = loop_candidates(m, spacing=sp, affine=affine, min_loop_mm=min_loop_mm)
    G = linfo.get("graph")
    if G is None or G.number_of_nodes() < 2:
        return np.empty((0, 3), np.float32), {"error": "skeleton too small"}
    G = G.copy()
    dropped = 0
    for lp in loops:
        for i in lp.bridge:
            if lp.nodes[i] in G:
                G.remove_node(lp.nodes[i])
                dropped += 1
    info["bridge_nodes_dropped"] = dropped
    comp = max(nx.connected_components(G), key=len)
    H = G.subgraph(comp)
    sup = superior_axis(affine)
    leaves = [n for n in H.nodes if H.degree(n) == 1] or list(H.nodes)
    h = to_numpy(leaves).astype(float) @ sup
    base, tip = leaves[int(np.argmin(h))], leaves[int(np.argmax(h))]
    if base == tip:
        return np.empty((0, 3), np.float32), {"error": "base == tip"}
    path = nx.shortest_path(H, base, tip, weight="w")
    pts = to_numpy(path).astype(np.float32) + origin
    length = float(np.linalg.norm(np.diff(pts, axis=0) * sp, axis=1).sum()) if len(pts) > 1 else 0.0
    info.update(base=[int(v) for v in pts[0]], tip=[int(v) for v in pts[-1]], n_pts=int(len(pts)),
                length_mm=round(length, 2), n_leaves=len(leaves))
    return pts, info


# ──────────────────────────────────────────────────────────────────────────────
# Common re-segmentation (both ICAs, one threshold)
# ──────────────────────────────────────────────────────────────────────────────


@_host
def common_resegment(
    intensity: Any,
    labels: Any,
    ids: Sequence[int],
    *,
    spacing: Sequence[float] | np.ndarray = (1.0, 1.0, 1.0),
    dilate_mm: float = 1.0,
    max_volume_ratio: float = 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Re-threshold every label in *ids* with **one** Otsu threshold.

    The threshold is computed on the TOF values of all the labels' neighbourhoods together (each
    label dilated by *dilate_mm*, unlabelled voxels only), so both ICAs get the same rule. Each new
    mask is the above-threshold voxels of its own neighbourhood that are connected to the original
    label; one more than *max_volume_ratio* times larger or smaller keeps the original. Returns
    the new label volume and a log (threshold, voxels per label)."""
    from skimage.filters import threshold_otsu

    lab = to_numpy(labels).astype(np.int32, copy=True)
    inten = to_numpy(intensity).astype(np.float64, copy=False)
    sp = to_numpy(spacing).astype(float)
    rad = np.maximum(np.rint(float(dilate_mm) / sp).astype(int), 1)
    ball = np.zeros(tuple(2 * r + 1 for r in rad), dtype=bool)
    zz = np.indices(ball.shape).reshape(3, -1).T - rad
    ball.reshape(-1)[((zz * sp) ** 2).sum(1) <= float(dilate_mm) ** 2 + 1e-9] = True
    hoods: dict[int, np.ndarray] = {}
    for lid in ids:
        own = lab == int(lid)
        if own.any():
            hoods[int(lid)] = ndi.binary_dilation(own, structure=ball) & ((lab == 0) | own)
    info: dict[str, Any] = {"dilate_mm": float(dilate_mm), "labels": {}}
    if not hoods:
        return lab, info
    union = np.logical_or.reduce(list(hoods.values()))
    vals = inten[union]
    vals = vals[vals > 0]
    if vals.size < 16:
        info["warning"] = "too few voxels for a threshold"
        return lab, info
    t = float(threshold_otsu(vals))
    info["threshold"] = t
    claimed = np.zeros(lab.shape, dtype=np.int32)
    new_masks: dict[int, np.ndarray] = {}
    for lid, hood in hoods.items():
        own = lab == lid
        cand = hood & (inten > t)
        cc, _ = ndi.label(cand, structure=_CUBE)
        keep = np.unique(cc[own & cand])
        new = np.isin(cc, keep[keep > 0])
        n0, n1 = int(own.sum()), int(new.sum())
        entry = {"n_voxels_before": n0, "n_voxels_after": n1}
        if n1 == 0 or n1 > max_volume_ratio * n0 or n1 * max_volume_ratio < n0:
            entry["kept_original"] = True
            new = own
        new_masks[lid] = new
        claimed[new] += 1
        info["labels"][str(lid)] = entry
    for lid, new in new_masks.items():
        lab[lab == lid] = 0
        lab[new & (claimed == 1)] = lid
        info["labels"][str(lid)]["n_voxels_after"] = int((lab == lid).sum())
    return lab, info


# ──────────────────────────────────────────────────────────────────────────────
# Outputs: label names, images, rasterised masks, QC figure
# ──────────────────────────────────────────────────────────────────────────────


def siphon_label_name(label_id: int) -> str:
    """Readable name for a siphon label id (numeric: no vessel vocabulary is wired in)."""
    return f"label_{int(label_id)}"


def _jsonable_default(o: Any) -> Any:
    """``json.dump`` *default* hook: NumPy / CuPy scalars and arrays to native Python types."""
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        return o.item()
    if hasattr(o, "shape"):
        return to_numpy(o).tolist()
    return str(o)


def _to_image(obj: Any, *, name: str) -> Any:
    """Coerce a path / array / :class:`~nvitk.types.Image` to an ``Image`` (with metadata)."""
    from nvitk.types import Image

    if isinstance(obj, Image):
        return obj
    if isinstance(obj, (str, Path)):
        from nvitk.io.imageio import imread

        loaded = imread(str(obj))
        if isinstance(loaded, list):
            raise ValidationError(f"{name}: path '{obj}' resolved to multiple series.")
        return loaded
    return Image(data=obj, metadata={})


@_host
def _rasterize_centerlines_mask(shape: tuple[int, int, int], centerlines: dict[int, Any]) -> Any:
    """Host int32 mask with each label's id on its centerline points."""
    mask = np.zeros(shape, dtype=np.int32)
    for vid, pts in sorted(centerlines.items()):
        p = to_numpy(pts)
        if p.size == 0:
            continue
        ijk = np.rint(p[:, :3]).astype(np.int64)
        keep = np.all((ijk >= 0) & (ijk < to_numpy(shape)), axis=1)
        ijk = ijk[keep]
        mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = int(vid)
    return mask


@_host
def _rasterize_bridges_mask(shape: tuple[int, int, int], bridges_by_label: dict[int, list[tuple[int, int, int]]]) -> Any:
    """Host int32 mask of each label's cut voxels."""
    mask = np.zeros(shape, dtype=np.int32)
    for vid, voxels in bridges_by_label.items():
        if not voxels:
            continue
        v = to_numpy(voxels).reshape(-1, 3).astype(np.int64)
        v = v[np.all((v >= 0) & (v < to_numpy(shape)), axis=1)]
        mask[v[:, 0], v[:, 1], v[:, 2]] = int(vid)
    return mask


@_host
def _rasterize_cleared_masks(shape: tuple[int, int, int], cleared_by_label: dict[int, Any]) -> Any:
    """Host int32 mask of every voxel each label lost (pretess and cuts)."""
    mask = np.zeros(shape, dtype=np.int32)
    for vid, cleared in cleared_by_label.items():
        mask[_host_bool(cleared)] = int(vid)
    return mask


@_host
def _save_qc_figure(
    out_path: Path,
    *,
    mask_by_label: dict[int, Any],
    correction_ids: Sequence[int],
    centerlines: dict[int, Any],
    bridges_by_label: dict[int, list[tuple[int, int, int]]],
) -> None:
    """3D QC figure per corrected label: the corrected mask (points), its centerline and the cut
    voxels (headless matplotlib)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = max(1, len(correction_ids))
    fig = plt.figure(figsize=(7 * n, 8))
    for col, lid in enumerate(correction_ids):
        name = siphon_label_name(int(lid))
        ax = fig.add_subplot(1, n, col + 1, projection="3d")
        roi = mask_by_label.get(int(lid))
        if roi is None or not _host_bool(roi).any():
            ax.set_title(f"{name}: empty mask")
            continue
        surf = np.argwhere(_host_bool(roi))
        if surf.shape[0] > 8000:
            surf = surf[np.random.default_rng(0).choice(surf.shape[0], 8000, replace=False)]
        ax.scatter(surf[:, 0], surf[:, 1], surf[:, 2], s=1, c="lightblue", alpha=0.25, label="corrected mask")
        cl = centerlines.get(int(lid))
        if cl is not None and to_numpy(cl).shape[0] > 0:
            c = to_numpy(cl)
            ax.plot(c[:, 0], c[:, 1], c[:, 2], "-", c="red", lw=2.5, label=f"centerline ({c.shape[0]} pts)")
        bv = bridges_by_label.get(int(lid)) or []
        if bv:
            b = to_numpy(bv).reshape(-1, 3)
            ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=60, c="magenta", marker="x", label=f"cut ({b.shape[0]} vox)")
        ax.set_title(f"{name}: corrected mask, centerline, cut voxels")
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        ax.set_zlabel("k")
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# End to end (:func:`nvitk.morphology.centerline_siphon.correct_siphon_centerlines`)
# ──────────────────────────────────────────────────────────────────────────────


def repair_label(
    labels: Any,
    lid: int,
    intensity: Any | None,
    *,
    affine: Any | None,
    spacing: Sequence[float] | None = None,
    pretess_mode: str | None = "remove",
    min_loop_mm: float = MIN_LOOP_MM,
    max_cost_ratio: float = MAX_CUT_COST_RATIO,
    max_cut_mm3: float = MAX_CUT_MM3,
    max_cut_fraction: float = MAX_CUT_FRACTION,
    label_name: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Pretess + loop cuts on label *lid* of *labels* (in place: a CuPy *labels* gets the host
    result written back). Returns the label's repaired mask, every voxel it lost (pretess and
    cuts), both host bool, and the log."""
    host = to_numpy(labels)  # the same array for NumPy input, so the edit lands in place
    out = _repair_label_host(host, lid, intensity, affine=affine, spacing=spacing, pretess_mode=pretess_mode,
                             min_loop_mm=min_loop_mm, max_cost_ratio=max_cost_ratio, max_cut_mm3=max_cut_mm3,
                             max_cut_fraction=max_cut_fraction, label_name=label_name)
    if is_cupy_array(labels):
        labels[...] = as_backend_array(host, backend="cupy")
    return out


@_host
def _repair_label_host(
    labels: Any,
    lid: int,
    intensity: Any | None,
    *,
    affine: Any | None,
    spacing: Sequence[float] | None,
    pretess_mode: str | None,
    min_loop_mm: float,
    max_cost_ratio: float,
    max_cut_mm3: float,
    max_cut_fraction: float,
    label_name: str | None,
) -> tuple[Any, Any, dict[str, Any]]:
    """:func:`repair_label` on a host *labels* (edited in place)."""
    name = label_name or f"label_{int(lid)}"
    sp = spacing_of(affine, spacing)
    own = labels == int(lid)
    n0 = int(own.sum())
    entry: dict[str, Any] = {"label": int(lid), "label_name": name, "n_voxels_before": n0,
                             "volume_before_mm3": round(n0 * float(np.prod(sp)), 3)}
    m = own
    pre: dict[str, Any] = {"mode": None}
    if pretess_mode:
        m, pre = pretess(own, intensity, mode=pretess_mode, protect=(labels != 0) & ~own)
        pre = {"mode": pretess_mode,
               **{k: pre.get(k) for k in ("n_removed", "n_filled", "unresolved", "iterations", "reverted",
                                          "specks_removed")}}
    entry["pretess"] = pre
    m, cut, rep = repair_loops(m, intensity, affine=affine, spacing=sp, min_loop_mm=min_loop_mm,
                               max_cost_ratio=max_cost_ratio, max_cut_mm3=max_cut_mm3,
                               max_cut_fraction=max_cut_fraction, label_name=name)
    entry["repair"] = rep
    lost = own & ~m
    labels[own & ~m] = 0
    labels[m & ~own] = int(lid)
    n1 = int(m.sum())
    entry.update(
        n_voxels_after=n1,
        volume_after_mm3=round(n1 * float(np.prod(sp)), 3),
        volume_change_pct=round(100.0 * (n1 - n0) / n0, 3) if n0 else 0.0,
        cut_mm3=rep.get("removed_mm3", 0.0),
        removed_mm3=round(int(lost.sum()) * float(np.prod(sp)), 3),
        genus_before=betti(own).to_dict(),
        genus_after=betti(m).to_dict(),
    )
    return m, cut, entry


@_host
def correct_siphon_mincut(
    tof: Any,
    vessel_mask: Any,
    *,
    correction_ids: Sequence[int] = (1, 2),
    out_dir: str | Path | None = None,
    save_qc: bool = False,
    min_points: int = 3,
    pretess_mode: str | None = "remove",
    resegment: str = "none",
    min_loop_mm: float = MIN_LOOP_MM,
    max_cost_ratio: float = MAX_CUT_COST_RATIO,
    max_cut_mm3: float = MAX_CUT_MM3,
    max_cut_fraction: float = MAX_CUT_FRACTION,
) -> dict[str, Any]:
    """The minimum-cut siphon correction behind
    :func:`~nvitk.morphology.centerline_siphon.correct_siphon_centerlines` (its keys and files).

    ``removed_bridges`` holds the cut voxels and ``cleared_bridge_region`` every voxel the
    correction removed (pretess and cuts). ``siphon_correction.json`` adds, per label, the pretess
    counts, the loop log, the volume before / after and its change, and over the corrected labels
    ``asymmetry_change_pct`` = |ΔL − ΔR| (volume change %), flagged above
    :data:`ASYMMETRY_TOLERANCE_PCT`."""
    from nvitk.morphology.centerline import compute_centerlines

    if resegment not in RESEGMENT_MODES:
        raise ValidationError(f"resegment must be one of {RESEGMENT_MODES}, got {resegment!r}")
    with using("cpu"):
        tof_img = _to_image(tof, name="tof")
        mask_img = _to_image(vessel_mask, name="vessel_mask")
        inten = to_numpy(tof_img.data).astype(np.float64, copy=False)
        labels = to_numpy(mask_img.data).astype(np.int32, copy=True)
        shape = tuple(int(s) for s in labels.shape[:3])
        if inten.shape[:3] != shape:
            raise ValidationError(f"tof shape {inten.shape[:3]} != vessel_mask shape {shape}")
        affine = tof_img.affine if tof_img.affine is not None else mask_img.affine
        sp = spacing_of(affine, tof_img.spacing or mask_img.spacing)
        ids = [int(v) for v in correction_ids]
        present = sorted(int(v) for v in np.unique(labels) if v > 0)
        log.step(f"correct_siphon_mincut: shape={shape} spacing={np.round(sp, 3).tolist()} ids={ids} "
                 f"pretess={pretess_mode} resegment={resegment}")
        original = labels.copy()
        reseg_info = None
        if resegment == "common":
            labels, reseg_info = common_resegment(inten, labels, [i for i in ids if i in present], spacing=sp)
        details: dict[int, dict[str, Any]] = {}
        cuts: dict[int, list[tuple[int, int, int]]] = {}
        cleared: dict[int, np.ndarray] = {}
        centerlines: dict[int, Any] = {}
        for lid in ids:
            name = siphon_label_name(lid)
            if lid not in present:
                log.warning(f"[{name}] label {lid} not in mask — skipping")
                continue
            t0 = time.time()
            m, cut, entry = repair_label(labels, lid, inten, affine=affine, spacing=sp, pretess_mode=pretess_mode,
                                         min_loop_mm=min_loop_mm, max_cost_ratio=max_cost_ratio,
                                         max_cut_mm3=max_cut_mm3, max_cut_fraction=max_cut_fraction,
                                         label_name=name)
            orig = original == lid
            entry["volume_change_vs_input_pct"] = (
                round(100.0 * (int(m.sum()) - int(orig.sum())) / int(orig.sum()), 3) if orig.any() else 0.0
            )
            path, cl_info = siphon_centerline(m, affine=affine, spacing=sp)
            entry["centerline"] = cl_info
            entry["method"] = "mincut"
            entry["seconds"] = round(time.time() - t0, 2)
            if len(path) >= int(min_points):
                centerlines[lid] = path
            cuts[lid] = [tuple(int(x) for x in v) for v in np.argwhere(cut)]
            cleared[lid] = orig & ~m
            details[lid] = entry
        others = [lid for lid in present if lid not in details]
        if others:
            centerlines.update(compute_centerlines(labels, labels=others, min_points=int(min_points)))
        changes = {str(lid): details[lid]["volume_change_vs_input_pct"] for lid in details}
        asym = None
        if len(details) == 2:
            a, b = (details[k]["volume_change_vs_input_pct"] for k in details)
            asym = round(abs(a - b), 3)
        summary = {
            "method": "mincut",
            "pretess": pretess_mode,
            "resegment": reseg_info if reseg_info is not None else "none",
            "volume_change_pct": changes,
            "asymmetry_change_pct": asym,
            "asymmetry_flag": bool(asym is not None and asym > ASYMMETRY_TOLERANCE_PCT),
            "asymmetry_tolerance_pct": ASYMMETRY_TOLERANCE_PCT,
        }
        if summary["asymmetry_flag"]:
            log.warning(f"siphon correction changed the ICA volumes asymmetrically: |ΔL − ΔR| = {asym:.2f} %")

        ica_only = np.where(np.isin(labels, list(details)), labels, 0).astype(np.int32)
        cl_mask = to_numpy(_rasterize_centerlines_mask(shape, centerlines))
        br_mask = to_numpy(_rasterize_bridges_mask(shape, cuts))
        cleared_mask = to_numpy(_rasterize_cleared_masks(shape, cleared))
        out_paths: dict[str, str] = {}
        if out_dir is not None:
            import nvitk.io as io

            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            meta = dict(tof_img.metadata or {})
            for key, arr, fname in (
                ("centerlines", cl_mask, "corrected_centerlines.nii.gz"),
                ("bridges", br_mask, "removed_bridges.nii.gz"),
                ("vessel_mask", labels, "vessel_mask_corrected.nii.gz"),
                ("seg_ica_repaired", ica_only, "seg_ica_repaired.nii.gz"),
                ("cleared_bridge_region", cleared_mask, "cleared_bridge_region.nii.gz"),
            ):
                io.imsave(out / fname, arr.astype(np.int32), metadata=meta, axes="XYZ")
                out_paths[key] = str(out / fname)
            meta_path = out / "siphon_correction.json"
            meta_path.write_text(json.dumps({
                "correction_ids": ids, "shape": list(shape), "spacing_mm": [float(s) for s in sp],
                "summary": summary, "details": {str(k): v for k, v in details.items()},
            }, indent=2, default=_jsonable_default), encoding="utf-8")
            out_paths["meta"] = str(meta_path)
            if save_qc:
                try:
                    qc = out / "qc_siphon_correction.png"
                    _save_qc_figure(qc, mask_by_label={k: labels == k for k in details}, correction_ids=ids,
                                    centerlines=centerlines, bridges_by_label=cuts)
                    out_paths["qc"] = str(qc)
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"QC figure generation failed: {exc}")
        print_summary(details)
        return {
            "centerlines": {int(k): as_backend_array(v) for k, v in centerlines.items()},
            "bridges": {int(k): v for k, v in cuts.items()},
            "details": details,
            "summary": summary,
            "corrected_centerlines_mask": as_backend_array(cl_mask),
            "removed_bridges_mask": as_backend_array(br_mask),
            "vessel_mask_corrected": as_backend_array(labels),
            "seg_ica_repaired": as_backend_array(ica_only),
            "cleared_bridge_region_mask": as_backend_array(cleared_mask),
            "output_paths": out_paths,
        }


def print_summary(details: dict[int, dict[str, Any]]) -> str:
    """One line per corrected label: voxels, β₁ before → after, pretess, cuts, mm³, volume change."""
    lines = ["", f"{'label':10s} {'vox':>7s} {'→':>7s} {'β₁':>7s} {'pretess −/+':>12s} {'cuts':>5s} "
                 f"{'cut mm³':>8s} {'ΔV %':>7s} {'CL mm':>7s}"]
    for lid, d in details.items():
        g0, g1 = d.get("genus_before", {}), d.get("genus_after", {})
        pre = d.get("pretess") or {}
        rep = d.get("repair") or {}
        lines.append(
            f"{d.get('label_name', lid)!s:10s} {d.get('n_voxels_before', 0):7d} {d.get('n_voxels_after', 0):7d} "
            f"{g0.get('beta1', 0)}→{g1.get('beta1', 0)!s:5s} "
            f"{pre.get('n_removed', 0)!s:>6s}/{pre.get('n_filled', 0)!s:<5s} {rep.get('n_cut_loops', 0):5d} "
            f"{d.get('cut_mm3', 0.0):8.2f} {d.get('volume_change_vs_input_pct', 0.0):7.2f} "
            f"{(d.get('centerline') or {}).get('length_mm', float('nan')):7.1f}"
        )
    text = "\n".join(lines)
    print(text)
    return text


__all__ = [
    "ASYMMETRY_TOLERANCE_PCT",
    "Betti",
    "Loop",
    "MAX_CUT_COST_RATIO",
    "MAX_CUT_FRACTION",
    "MAX_CUT_MM3",
    "MIN_LOOP_MM",
    "PRETESS_MODES",
    "RESEGMENT_MODES",
    "betti",
    "common_resegment",
    "contact_run",
    "correct_siphon_mincut",
    "critical_configurations",
    "loop_candidates",
    "pretess",
    "repair_label",
    "repair_loops",
    "siphon_centerline",
    "siphon_label_name",
    "spacing_of",
    "topology_numbers",
    "voxel_weights",
]
