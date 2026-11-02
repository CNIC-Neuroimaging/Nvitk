"""ICA siphon correction (entry point) and the mask genus check.

Problem
-------
The cavernous siphon of the Internal Carotid Artery (ICA) often closes into a donut on TOF MRA:
two passes of the vessel touch by partial volume and merge in the mask, turning the hook
(:math:`\\beta_1 = 0`) into a torus (:math:`\\beta_1 \\ge 1`). A skeleton centerline then
shortcuts across the contact instead of following the curl.

Correction
----------
:func:`correct_siphon_centerlines` is the stable entry point of the minimum-cut correction in
:mod:`nvitk.morphology.siphon_cut`. The segmentation's boundary is kept: edge- and corner-only
contacts are repaired (pretess), then each remaining loop is opened by removing only the contact
that closes it (an intensity-weighted minimum vertex cut). The earlier Otsu + erosion + ball-cut
correction was removed on 2026-10-23.

Outputs (:func:`correct_siphon_centerlines`, in the TOF's space; files with *out_dir*):

- ``corrected_centerlines.nii.gz``: per-label centerlines (siphon-corrected for
  ``correction_ids``, :func:`~nvitk.morphology.centerline.compute_centerlines` for the rest);
- ``removed_bridges.nii.gz``: the voxels each accepted cut removed;
- ``vessel_mask_corrected.nii.gz``: the multilabel mask after the correction;
- ``seg_ica_repaired.nii.gz``: the corrected labels only;
- ``cleared_bridge_region.nii.gz``: every voxel the correction removed (pretess and cuts);
- ``siphon_correction.json``: per label the pretess counts, every cut attempt, the volume change,
  and |ΔL - ΔR| over the two ICAs;
- ``qc_siphon_correction.png``: a 3D QC figure (``save_qc=True``).

Genus check
-----------
:func:`compute_mask_genus` reports β₀ / β₁ of a mask (Euler characteristic, 6-connected
foreground) and the skeleton's cycles, with small handles (a cycle shorter than
:data:`MIN_SIPHON_CYCLE_LEN` voxels) filtered as noise.

Backend
-------
``np`` / ``scipy`` / ``ndi`` come from ``setup(globals())``. Both functions are host-bound
(networkx, scikit-image, SciPy's max flow) and accept NumPy or CuPy input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from nvitk.core.array import to_numpy
from nvitk.core.backend import setup, using
from nvitk.core.logger import Logger
from nvitk.morphology.centerline import skeletonize_binary
from nvitk.morphology.components import label_connected
from nvitk.morphology.siphon_cut import siphon_label_name

setup(globals())

log = Logger()

#: Skeleton cycles shorter than this (voxels) are noise handles, not a siphon loop.
MIN_SIPHON_CYCLE_LEN = 20

with using("cpu"):
    #: The 26 neighbour offsets of a voxel (host).
    _NEI26 = to_numpy(
        [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    ).astype(np.int32)


# Siphon correction (minimum cut)


def correct_siphon_centerlines(
    tof: Any,
    vessel_mask: Any,
    *,
    correction_ids: Sequence[int] = (1, 2),
    out_dir: str | Path | None = None,
    save_qc: bool = False,
    min_points: int = 3,
    pretess: bool = True,
    resegment: str = "none",
    max_cut_mm3: float | None = None,
    max_cost_ratio: float | None = None,
) -> dict[str, Any]:
    """Open the siphon loops of the labels in *correction_ids* and recompute their centerlines.

    The minimum-cut correction (:func:`nvitk.morphology.siphon_cut.correct_siphon_mincut`):
    *pretess* repairs edge / corner-only contacts first; *resegment* ``"common"`` re-thresholds
    all *correction_ids* with one shared Otsu threshold; *max_cut_mm3* (default 20 mm³) and
    *max_cost_ratio* (default 0.6 of a lumen slice) bound an accepted cut.

    Parameters
    ----------
    tof
        TOF MRA (``Image``, path, or array); its spatial metadata goes to every NIfTI output.
    vessel_mask
        Multilabel vessel mask on the same grid as *tof*.
    correction_ids
        Label ids to correct (default ``(1, 2)``: eICAB's LICA / RICA).
    out_dir
        If set, writes the masks, ``siphon_correction.json`` and, with *save_qc*, the QC figure.
    min_points
        Minimum centerline length (voxels) for inclusion.

    Returns
    -------
    dict
        ``centerlines``, ``bridges``, ``details``, ``summary``, the rasterised masks and
        ``output_paths``.
    """
    from nvitk.morphology import siphon_cut

    extra: dict[str, Any] = {}
    if max_cut_mm3 is not None:
        extra["max_cut_mm3"] = float(max_cut_mm3)
    if max_cost_ratio is not None:
        extra["max_cost_ratio"] = float(max_cost_ratio)
    return siphon_cut.correct_siphon_mincut(
        tof, vessel_mask, correction_ids=correction_ids, out_dir=out_dir, save_qc=save_qc,
        min_points=min_points, pretess_mode="remove" if pretess else None, resegment=resegment, **extra,
    )


# Genus check


@dataclass(frozen=True)
class GenusReport:
    """Topology summary for a 3D binary mask.

    For a tube-like vessel we assume :math:`\\beta_2 \\approx 0` (no internal cavities), so
    :math:`\\beta_1 = \\beta_0 - \\chi`. ``suspect`` flips True iff :math:`\\beta_1 > 0` (a
    topological handle / donut).
    """

    label_name: str
    n_voxels: int
    n_components: int
    euler_chi: float
    beta0: int
    beta1: int
    skeleton_cycles: int
    skeleton_voxels: int
    beta1_raw: int = 0
    max_cycle_len: int = 0
    noise_filtered: bool = False

    @property
    def suspect(self) -> bool:
        """True when the mask has >=1 topological loop (``beta1 > 0``): a likely siphon donut."""
        return self.beta1 > 0

    def to_dict(self) -> dict:
        """JSON-serializable dict of the genus report (adds the derived ``suspect`` flag)."""
        d = asdict(self)
        d["suspect"] = bool(self.suspect)
        return d


def _skeleton_to_graph(sk: Any) -> Any:
    """26-connected ``networkx.Graph`` over the skeleton voxels (host; nodes ``(i, j, k)``)."""
    import networkx as nx

    sk_np = to_numpy(sk).astype(bool, copy=False)
    G = nx.Graph()
    with using("cpu"):
        coords = np.argwhere(sk_np)
    nodes = [tuple(int(v) for v in row) for row in coords]
    node_set = set(nodes)
    for n in nodes:
        G.add_node(n, x=int(n[0]), y=int(n[1]), z=int(n[2]))
    shape = sk_np.shape
    for n in nodes:
        for d in _NEI26:
            v = (int(n[0] + d[0]), int(n[1] + d[1]), int(n[2] + d[2]))
            if not (0 <= v[0] < shape[0] and 0 <= v[1] < shape[1] and 0 <= v[2] < shape[2]):
                continue
            if v in node_set and (v > n):
                G.add_edge(n, v)
    return G


def _skeleton_cycle_lengths(mask: Any) -> list[int]:
    """Lengths of the cycles of the 26-connected skeleton graph."""
    import networkx as nx

    sk = to_numpy(skeletonize_binary(mask)).astype(bool, copy=False)
    if int(sk.sum()) < 3:
        return []
    G = _skeleton_to_graph(sk)
    if G.number_of_edges() == 0:
        return []
    try:
        cycles = list(nx.minimum_cycle_basis(G))
    except Exception:  # noqa: BLE001
        cycles = []
    if not cycles:
        try:
            cycles = list(nx.cycle_basis(G))
        except Exception:  # noqa: BLE001
            cycles = []
    return [int(len(c)) for c in cycles]


def compute_mask_genus(
    mask: Any,
    *,
    label_name: str = "vessel",
    connectivity: int = 1,
    min_cycle_len: int = MIN_SIPHON_CYCLE_LEN,
    filter_small_handles: bool = True,
) -> GenusReport:
    """Return :class:`GenusReport` for *mask* (3D bool / int, NumPy or CuPy).

    When ``filter_small_handles`` is True, β₁ is zeroed if the largest skeleton cycle is shorter
    than ``min_cycle_len`` (noise handle vs cavernous siphon). The mask is moved to NumPy
    internally (``skimage`` / ``networkx`` are CPU only). The report is JSON-serialisable via
    ``report.to_dict()``.
    """
    import networkx as nx
    from skimage.measure import euler_number

    m = to_numpy(mask).astype(bool, copy=False)
    n_vox = int(m.sum())
    if n_vox == 0:
        return GenusReport(label_name, 0, 0, 0.0, 0, 0, 0, 0)
    with using("cpu"):
        labeled, n_cc = label_connected(m, connectivity=connectivity)
    labeled_np = to_numpy(labeled)
    chi_sum = 0.0
    for cid in range(1, int(n_cc) + 1):
        chi_sum += float(euler_number(labeled_np == cid, connectivity=connectivity))
    beta0 = int(n_cc)
    beta1_raw = max(0, int(beta0 - int(round(chi_sum))))
    cycle_lengths = _skeleton_cycle_lengths(m) if filter_small_handles else []
    max_cycle = int(max(cycle_lengths)) if cycle_lengths else 0
    sk = to_numpy(skeletonize_binary(m)).astype(bool, copy=False)
    sk_vox = int(sk.sum())
    if sk_vox < 3:
        cyc = 0
    else:
        G = _skeleton_to_graph(sk)
        cyc = max(0, int(G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)))
    beta1 = int(beta1_raw)
    noise_filtered = False
    if filter_small_handles and beta1 > 0 and 0 < max_cycle < int(min_cycle_len):
        beta1 = 0
        cyc = 0
        noise_filtered = True
    return GenusReport(
        label_name=label_name,
        n_voxels=n_vox,
        n_components=beta0,
        euler_chi=float(chi_sum),
        beta0=beta0,
        beta1=beta1,
        skeleton_cycles=int(cyc),
        skeleton_voxels=int(sk_vox),
        beta1_raw=int(beta1_raw),
        max_cycle_len=int(max_cycle),
        noise_filtered=bool(noise_filtered),
    )


__all__ = [
    "GenusReport",
    "MIN_SIPHON_CYCLE_LEN",
    "compute_mask_genus",
    "correct_siphon_centerlines",
    "siphon_label_name",
]
