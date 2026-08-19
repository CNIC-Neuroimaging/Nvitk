"""
Atlas-based brain visualization via Nilearn (:func:`brainshow`).

Supports volumetric atlases (custom NIfTI or Nilearn presets such as Destrieux 2009)
and Destrieux fsaverage5 surface atlases. Values are mapped to atlas regions using an
explicit ``region_order`` (sequence alignment) or a mapping keyed by region label /
atlas index.

Desikan and vascular presets
----------------------------
Nilearn ships no Desikan–Killiany atlas, and the arterial-territory atlas the ASL pipeline uses is
lab-specific, so both are *located* rather than fetched — see :mod:`nvitk.viz.atlas_sources` for the
search order. Both are registered here as ordinary presets, so a caller says ``atlas="desikan"``
and never learns where the file came from.

Region naming
-------------
The same parcel is spelled several ways across this project — ``ctx-lh-precuneus`` in a published
table, ``ctx_lh_precuneus`` once normalized into the database, ``left_precuneus`` by the T1
pipeline, and just ``precuneus`` inside a FreeSurfer ``.annot``. :func:`normalize_region_key` reduces
all of them to ``(side, stem)``, and :func:`atlas_indices_for_region` resolves that onto atlas
indices — returning a **list**, because an aggregate level such as ``ctx-Left-Frontal-Lobe``,
``ctx-left-hemisphere`` or a hemisphere-melted parcel belongs on several parcels at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

from nvitk.core.exceptions import ValidationError
from nvitk.core.logger import Logger

log = Logger()

AtlasLike = str | Mapping[str, Any]
Mode = Literal["surface", "volume"]
Hemisphere = Literal["both", "left", "right"]
Flavor = Literal["volume_nifti", "surface_fs_vertex"]


@dataclass(frozen=True)
class ResolvedAtlas:
    """Unified atlas representation after resolving presets or user specs."""

    flavor: Flavor
    index_to_label: dict[int, str]
    atlas_img: Nifti1Image | None = None
    map_left: np.ndarray | None = None
    map_right: np.ndarray | None = None
    surf_mesh: str = "fsaverage5"


def _require_nilearn():
    """Raise a clear install hint unless the optional ``nilearn`` dependency is available."""
    try:
        import nilearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "brainshow requires the optional dependency 'nilearn'. "
            "Install it with: pip install nilearn"
        ) from exc


def _norm_label_key(name: str) -> str:
    """Normalize an atlas region label for case-insensitive lookup (strip + casefold)."""
    return name.strip().casefold()


def _effective_region_order(
    region_order: Sequence[Any] | None,
    labels: Sequence[Any] | None,
) -> list[Any] | None:
    """Resolve the region ordering to use: explicit *region_order*, else *labels*, else ``None``."""
    if region_order is not None:
        return list(region_order)
    if labels is not None:
        return list(labels)
    return None


def _invert_label_map(index_to_label: Mapping[int, str]) -> dict[str, int]:
    """Invert an atlas ``index → label`` map to ``normalized_label → index``; raises on duplicate labels."""
    rev: dict[str, int] = {}
    for idx, raw in index_to_label.items():
        key = _norm_label_key(str(raw))
        if key in rev:
            raise ValidationError(
                f"Atlas has duplicate label name {raw!r}; cannot map values unambiguously."
            )
        rev[key] = int(idx)
    return rev


def _sequence_to_mapping(
    values: Sequence[float],
    region_order: Sequence[Any],
    label_to_index: Mapping[str, int],
) -> dict[int, float]:
    """Zip parallel *values* and *region_order* (label names or atlas indices) into an ``index → value`` map."""
    if len(values) != len(region_order):
        raise ValidationError(
            f"values length {len(values)} does not match region_order length {len(region_order)}."
        )
    out: dict[int, float] = {}
    for v, key in zip(values, region_order, strict=False):
        if isinstance(key, bool):
            raise ValidationError(
                "Boolean region keys are not allowed (ambiguous with int indices)."
            )
        if isinstance(key, int):
            idx = int(key)
        elif isinstance(key, str):
            lk = _norm_label_key(key)
            if lk not in label_to_index:
                raise ValidationError(f"Unknown region label {key!r} for this atlas.")
            idx = label_to_index[lk]
        else:
            raise ValidationError(f"Unsupported region key type: {type(key).__name__}")
        out[idx] = float(v)
    return out


def _mapping_to_index_values(
    values: Mapping[Any, float],
    label_to_index: Mapping[str, int],
) -> dict[int, float]:
    """Normalize a ``{region_label_or_index: value}`` dict to ``{atlas_index: value}``."""
    out: dict[int, float] = {}
    for key, v in values.items():
        if isinstance(key, bool):
            raise ValidationError(
                "Boolean dict keys are not allowed (ambiguous with int indices)."
            )
        if isinstance(key, int):
            idx = int(key)
        elif isinstance(key, str):
            lk = _norm_label_key(key)
            if lk not in label_to_index:
                raise ValidationError(f"Unknown region label {key!r} for this atlas.")
            idx = label_to_index[lk]
        else:
            raise ValidationError(f"Unsupported dict key type: {type(key).__name__}")
        out[idx] = float(v)
    return out


def build_index_to_value(
    values: Any,
    *,
    region_order: Sequence[Any] | None,
    index_to_label: Mapping[int, str],
) -> dict[int, float]:
    """
    Turn *values* + *region_order* (or mapping *values*) into atlas index -> float.

    Atlas indices follow the integer labels stored in the volume or surface parcel maps
    (same convention as Nilearn Destrieux atlases).
    """
    label_to_index = _invert_label_map(index_to_label)
    if isinstance(values, Mapping):
        out = _mapping_to_index_values(values, label_to_index)
    else:
        seq = np.ravel(values).tolist()
        ro = region_order
        if ro is None:
            raise ValidationError(
                "When values is a sequence, region_order (or labels) is required "
                "to define the territory ordering."
            )
        out = _sequence_to_mapping(seq, ro, label_to_index)

    unknown = sorted(set(out) - set(map(int, index_to_label.keys())))
    if unknown:
        raise ValidationError(
            "Values reference atlas indices not listed in atlas metadata: "
            + ", ".join(str(i) for i in unknown)
        )
    return out


def _load_nifti_image(maps: Any) -> Nifti1Image:
    """Load an atlas volume from a Nifti1Image, path string, or Path (raises on any other type)."""
    if isinstance(maps, Nifti1Image):
        return maps
    if isinstance(maps, (str, Path)):
        return nib.load(str(maps))
    raise ValidationError(
        "Atlas 'maps' must be a nibabel Nifti1Image, path string, or pathlib.Path."
    )


def _labels_from_spec(spec: Any, unique_indices: Sequence[int]) -> dict[int, str]:
    """Build index -> label from user spec: dict, list (index-aligned), or None (auto)."""
    if spec is None:
        return {int(i): f"region_{int(i)}" for i in sorted(unique_indices)}
    if isinstance(spec, Mapping):
        out: dict[int, str] = {}
        for k, v in spec.items():
            out[int(k)] = str(v)
        return out
    if isinstance(spec, (list, tuple)):
        out_list: dict[int, str] = {}
        for i, name in enumerate(spec):
            out_list[i] = str(name)
        return out_list
    raise ValidationError(
        "Atlas 'labels' must be a dict[int, str], a list of names (index-aligned), or None."
    )


def resolve_atlas_volume_custom(
    maps: Any,
    labels: Any = None,
) -> ResolvedAtlas:
    """Resolve a user-provided volumetric label atlas (integer NIfTI)."""
    img = _load_nifti_image(maps)
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValidationError(f"Atlas volume must be 3D; got shape {data.shape}.")
    uniq = np.unique(data.astype(np.int64, copy=False))
    uniq = [int(u) for u in uniq.tolist()]
    index_to_label = _labels_from_spec(labels, uniq)
    if isinstance(labels, (list, tuple)) and labels:
        max_idx = max(uniq)
        if max_idx >= len(labels):
            raise ValidationError(
                f"Atlas labels list length {len(labels)} cannot encode voxel label index {max_idx}. "
                "Provide a longer labels list or use an explicit dict mapping."
            )
    # Warn if image contains indices missing from explicit dict? We validate coverage at paint time.
    return ResolvedAtlas(
        flavor="volume_nifti",
        index_to_label=index_to_label,
        atlas_img=img,
        map_left=None,
        map_right=None,
    )


def resolve_atlas_preset_destrieux_vol(atlas_kwargs: dict[str, Any] | None) -> ResolvedAtlas:
    """Fetch (via nilearn) and resolve the volumetric Destrieux 2009 atlas."""
    _require_nilearn()
    from nilearn import datasets

    kwargs = dict(atlas_kwargs or {})
    atlas = datasets.fetch_atlas_destrieux_2009(**kwargs)
    img = nib.load(atlas.maps)
    labels_list = list(atlas.labels)
    index_to_label = {i: str(labels_list[i]) for i in range(len(labels_list))}
    return ResolvedAtlas(
        flavor="volume_nifti",
        index_to_label=index_to_label,
        atlas_img=img,
    )


def resolve_atlas_preset_surf_destrieux(atlas_kwargs: dict[str, Any] | None) -> ResolvedAtlas:
    """Fetch (via nilearn) and resolve the surface-based (fsaverage) Destrieux atlas."""
    _require_nilearn()
    from nilearn import datasets

    kwargs = dict(atlas_kwargs or {})
    atlas = datasets.fetch_atlas_surf_destrieux(**kwargs)
    labels_list = list(atlas.labels)
    index_to_label = {i: str(labels_list[i]) for i in range(len(labels_list))}
    ml = np.asarray(atlas.map_left)
    mr = np.asarray(atlas.map_right)
    return ResolvedAtlas(
        flavor="surface_fs_vertex",
        index_to_label=index_to_label,
        atlas_img=None,
        map_left=ml,
        map_right=mr,
    )


# ---------------------------------------------------------------------------
# Desikan–Killiany
# ---------------------------------------------------------------------------
#: Added to the right hemisphere's vertex indices so the two hemispheres address distinct entries of
#: a single ``index_to_label``. A FreeSurfer ``.annot`` numbers each hemisphere from zero with the
#: *same* names, so without an offset a value for ``ctx-lh-precuneus`` would paint both precunei.
#: Larger than any parcel count by a wide margin, so the two ranges can never collide.
RIGHT_HEMI_INDEX_OFFSET: int = 1000

#: Lobes a *cortical surface* atlas can draw. The rest of :data:`DESIKAN_LOBES` covers
#: subcortical structures, white matter, ventricles and whole-head scalars — real groupings for
#: a panel layout, but nothing a surface parcellation contains.
_CORTICAL_LOBES: tuple[str, ...] = (
    "Frontal", "Parietal", "Temporal", "Occipital", "Cingulate", "Insula",
)

#: Annot entries that name no cortical parcel. Painting them would fill the medial wall and the
#: interhemispheric band with whatever value happened to be first.
_ANNOT_NON_PARCELS: frozenset[str] = frozenset({"unknown", "corpuscallosum", "", "none"})

#: Arterial-territory / watershed atlas indices. Published with the atlas file; 9 and 10 are the
#: ventricles and carry no perfusion territory, so they are deliberately absent.
VASCULAR_ATLAS_LABELS: dict[int, str] = {
    1: "Left_ACA", 2: "Right_ACA",
    3: "Left_MCA", 4: "Right_MCA",
    5: "Left_PCA", 6: "Right_PCA",
    7: "Left_Basilar", 8: "Right_Basilar",
    11: "Watershed",
}


def _read_annot(path: Any) -> tuple[np.ndarray, list[str]]:
    """``(per-vertex index, parcel names)`` from a FreeSurfer ``.annot``."""
    labels, _ctab, names = nib.freesurfer.read_annot(str(path))
    decoded = [
        (n.decode("utf-8", "replace") if isinstance(n, (bytes, bytearray)) else str(n))
        for n in names
    ]
    return np.asarray(labels), decoded


def resolve_atlas_desikan_annot(
    left_path: Any, right_path: Any, *, surf_mesh: str = "fsaverage5"
) -> ResolvedAtlas:
    """
    Desikan surface parcellation from a pair of FreeSurfer ``?h.aparc.annot`` files.

    The right hemisphere's indices are shifted by :data:`RIGHT_HEMI_INDEX_OFFSET` so each parcel is
    individually addressable through one ``index_to_label``; labels are emitted in the published
    ``ctx-lh-…`` / ``ctx-rh-…`` spelling so they match the measurement vocabulary directly.

    Vertices belonging to no parcel (``unknown``, ``corpuscallosum``) are left out of the label map
    and therefore never painted — the medial wall is not a region with a value.
    """
    left_map, left_names = _read_annot(left_path)
    right_map, right_names = _read_annot(right_path)

    index_to_label: dict[int, str] = {}
    for side, names in (("lh", left_names), ("rh", right_names)):
        offset = 0 if side == "lh" else RIGHT_HEMI_INDEX_OFFSET
        for index, name in enumerate(names):
            if name.strip().lower() in _ANNOT_NON_PARCELS:
                continue
            index_to_label[index + offset] = f"ctx-{side}-{name}"

    # -1 marks "no label" in an annot; shifting it would make it collide with a real right-hemisphere
    # parcel, so only the genuine indices move.
    shifted_right = np.where(right_map >= 0, right_map + RIGHT_HEMI_INDEX_OFFSET, right_map)

    log.info(
        "Desikan atlas: %d parcels over %s (%d + %d vertices).",
        len(index_to_label), surf_mesh, left_map.size, right_map.size,
    )
    return ResolvedAtlas(
        flavor="surface_fs_vertex",
        index_to_label=index_to_label,
        map_left=left_map,
        map_right=shifted_right,
        surf_mesh=surf_mesh,
    )


def _dkt_on_mni152(cache_path: Path) -> Nifti1Image:
    """
    Generate (once) and cache a DKT parcellation of the MNI152 template.

    Last resort, used only when no Desikan atlas is configured and FreeSurfer is absent. It is
    honest about being a *different* parcellation: DKT drops ``bankssts``, ``frontalpole`` and
    ``temporalpole`` from the Desikan set, so measurements in those three parcels have no geometry
    to paint and are reported as absent rather than approximated onto a neighbour.
    """
    _require_nilearn()
    from nilearn import datasets as nl_datasets

    if cache_path.is_file():
        return nib.load(str(cache_path))

    log.warning(
        "No Desikan atlas configured and no FreeSurfer fsaverage found — generating a DKT "
        "parcellation of the MNI152 template instead (one-off, cached at %s). DKT is not Desikan: "
        "bankssts, frontalpole and temporalpole have no geometry and will be drawn as absent.",
        cache_path,
    )
    from nvitk.io.ants_bridge import to_ants
    from nvitk.segmentation.dkt import desikan_killiany_tourville_labeling
    from nvitk.types import Image

    template = nl_datasets.load_mni152_template(resolution=1)
    volume = Image(
        data=np.asanyarray(template.dataobj).astype(float),
        metadata={"affine": np.asarray(template.affine, dtype=float)},
    )
    labels = desikan_killiany_tourville_labeling(volume, do_preprocessing=True)
    data = np.asarray(labels.data if hasattr(labels, "data") else labels)
    image = nib.Nifti1Image(data.astype(np.int32), np.asarray(template.affine, dtype=float))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(cache_path))
    return image


def _dkt_index_to_label() -> dict[int, str]:
    """ANTsPyNet DKT ids → published ``ctx-lh-…`` / ``ctx-rh-…`` parcel names."""
    from nvitk.segmentation.dkt import DKT_LABELS

    return dict(DKT_LABELS)


def resolve_atlas_preset_desikan(atlas_kwargs: dict[str, Any] | None) -> ResolvedAtlas:
    """
    Resolve the Desikan–Killiany parcellation from wherever this machine has one.

    Order — see :mod:`nvitk.viz.atlas_sources` for what each step reads:

    1. an explicit ``maps=`` / ``annot=`` kwarg, or the configured atlas path (env or settings). A
       ``.annot`` becomes a surface atlas, a NIfTI a volumetric one;
    2. a FreeSurfer install's fsaverage ``?h.aparc.annot`` — the canonical Desikan surface labels;
    3. a cached DKT parcellation of the MNI152 template, generated on demand.

    Raises
    ------
    ValidationError
        When none of the three resolve, naming every place that was searched — the fix is a path,
        and an error that does not say which path to set is not actionable.
    """
    from nvitk.viz import atlas_sources

    kwargs = dict(atlas_kwargs or {})
    explicit = kwargs.get("maps") or kwargs.get("annot") or kwargs.get("path")

    # ---- 1. Configured path -------------------------------------------------------------------
    path = Path(str(explicit)) if explicit is not None else atlas_sources.desikan_atlas_path()
    if path is not None and path.exists():
        if path.suffix.lower() == ".annot":
            # One .annot is one hemisphere; its partner is the same name with the other prefix.
            partner = path.with_name(
                path.name.replace("lh.", "rh.", 1) if path.name.startswith("lh.")
                else path.name.replace("rh.", "lh.", 1)
            )
            if not partner.is_file():
                raise ValidationError(
                    f"{path} is one hemisphere of a Desikan surface atlas, but its counterpart "
                    f"{partner.name} is not beside it. Both hemispheres are needed."
                )
            left, right = (path, partner) if path.name.startswith("lh.") else (partner, path)
            return resolve_atlas_desikan_annot(
                left, right, surf_mesh=str(kwargs.get("surf_mesh") or "fsaverage5")
            )
        return resolve_atlas_volume_custom(path, kwargs.get("labels"))

    # ---- 2. FreeSurfer fsaverage --------------------------------------------------------------
    found = atlas_sources.fsaverage_aparc_annot()
    if found is not None:
        left, right, subject = found
        return resolve_atlas_desikan_annot(left, right, surf_mesh=subject)

    # ---- 3. Generated DKT ----------------------------------------------------------------------
    try:
        image = _dkt_on_mni152(atlas_sources.atlas_cache_dir() / "dkt_mni152.nii.gz")
    except Exception as exc:
        raise ValidationError(
            "No Desikan–Killiany atlas could be resolved, and generating a DKT parcellation of the "
            f"MNI152 template failed ({type(exc).__name__}: {exc}).\n"
            f"Point nvitk at one via: {atlas_sources.describe_search('desikan')}."
        ) from exc
    return ResolvedAtlas(
        flavor="volume_nifti",
        index_to_label=_dkt_index_to_label(),
        atlas_img=image,
    )


def merge_atlas_regions(
    resolved: ResolvedAtlas, groups: Mapping[str, Sequence[int]], *, name: str = "merged"
) -> ResolvedAtlas:
    """
    Collapse an atlas's regions into coarser ones by relabelling its maps.

    The parcels are merged, not merely painted alike: each group becomes a single region with one
    index, so a value lands on it once. That matters because painting a lobe by writing its value
    onto each member parcel makes the *boundaries between members* disappear only if every member
    got the same value — which is true for a lobe-grouped fit and false the moment a caller mixes
    granularities. Merging removes the possibility.

    Parameters
    ----------
    groups : mapping
        ``{new region label: [source atlas index, …]}``. Source indices not named in any group are
        dropped from the merged atlas, so they render as "no estimate" rather than as an
        unnamed leftover region.
    """
    index_to_label: dict[int, str] = {}
    remap: dict[int, int] = {}
    for new_index, (label, sources) in enumerate(groups.items(), start=1):
        index_to_label[new_index] = str(label)
        for source in sources:
            remap[int(source)] = new_index
    if not index_to_label:
        raise ValidationError(f"Cannot build the {name!r} atlas: no source region resolved.")

    def _relabel(arr: np.ndarray | None) -> np.ndarray | None:
        """Vectorised index remap; anything unmapped becomes -1 (drawn as absent)."""
        if arr is None:
            return None
        source = np.asarray(arr)
        # A lookup table beats ``np.isin`` per group: one pass over the map whatever the group count.
        top = int(source.max(initial=0))
        table = np.full(top + 2, -1, dtype=np.int32)
        for old, new in remap.items():
            if 0 <= old <= top:
                table[old] = new
        clipped = np.clip(source, -1, top)
        out = np.where(clipped >= 0, table[np.maximum(clipped, 0)], -1)
        return out.astype(np.int32)

    if resolved.flavor == "surface_fs_vertex":
        return ResolvedAtlas(
            flavor="surface_fs_vertex",
            index_to_label=index_to_label,
            map_left=_relabel(resolved.map_left),
            map_right=_relabel(resolved.map_right),
            surf_mesh=resolved.surf_mesh,
        )

    data = np.asanyarray(resolved.atlas_img.dataobj).astype(np.int64, copy=False)
    merged = _relabel(data)
    return ResolvedAtlas(
        flavor="volume_nifti",
        index_to_label=index_to_label,
        atlas_img=nib.Nifti1Image(merged, resolved.atlas_img.affine),
        surf_mesh=resolved.surf_mesh,
    )


def resolve_atlas_preset_desikan_lobes(atlas_kwargs: dict[str, Any] | None) -> ResolvedAtlas:
    """
    The Desikan parcellation merged into lobes, one region per lobe per hemisphere.

    What a lobe-grouped analysis is actually about: 68 parcels reduced to a handful of structures,
    each drawn as one region with one boundary. Built from the Desikan atlas rather than shipped
    separately, so it follows whatever Desikan source this machine resolved and can never disagree
    with the parcel-level map about where a lobe is.

    Labels are emitted in the pipeline's own published spelling — ``ctx-Left-Frontal-Lobe`` — so a
    frame grouped by the published lobe rows resolves without a translation table. The side-less
    panel name (``Frontal``) also resolves, onto both hemispheres, because that is what the
    ``lobe`` grouping produces.
    """
    from nvitk.stats.region_groups import DESIKAN_LOBES

    base = resolve_atlas_preset_desikan(atlas_kwargs)
    by_key = _atlas_index_by_key(base.index_to_label)

    groups: dict[str, list[int]] = {}
    for lobe, parcels in DESIKAN_LOBES.items():
        if lobe not in _CORTICAL_LOBES:
            continue
        for side, word in (("l", "Left"), ("r", "Right")):
            indices: list[int] = []
            for parcel in parcels:
                indices += by_key.get((side, _squash(parcel)), [])
            if indices:
                groups[f"ctx-{word}-{lobe}-Lobe"] = sorted(set(indices))
    log.info("Desikan lobe atlas: %d lobe region(s) merged from %d parcels.",
             len(groups), len(base.index_to_label))
    return merge_atlas_regions(base, groups, name="desikan_lobes")


def resolve_atlas_preset_vascular(atlas_kwargs: dict[str, Any] | None) -> ResolvedAtlas:
    """
    Resolve the arterial-territory / watershed atlas the ASL pipeline parcellates against.

    A single labelled NIfTI with the published index map :data:`VASCULAR_ATLAS_LABELS`. There is no
    fallback: unlike Desikan this atlas has no standard equivalent to approximate it with, and a
    substitute would silently redraw the territories.
    """
    from nvitk.viz import atlas_sources

    kwargs = dict(atlas_kwargs or {})
    explicit = kwargs.get("maps") or kwargs.get("path")
    path = Path(str(explicit)) if explicit is not None else atlas_sources.vascular_atlas_path()
    if path is None or not path.exists():
        raise ValidationError(
            "No arterial-territory atlas is configured. Point nvitk at the labelled NIfTI via: "
            f"{atlas_sources.describe_search('vascular')}."
        )
    labels = kwargs.get("labels") or VASCULAR_ATLAS_LABELS
    return resolve_atlas_volume_custom(path, labels)


_PRESET_VOLUME = frozenset({"destrieux_2009", "destrieux_vol"})
_PRESET_SURFACE = frozenset({"destrieux_surface", "surf_destrieux"})
_PRESET_DESIKAN = frozenset({"desikan", "desikan_killiany", "aparc", "dk"})
_PRESET_DESIKAN_LOBES = frozenset({"desikan_lobes", "lobes", "aparc_lobes", "lobe"})
_PRESET_VASCULAR = frozenset({"vascular", "arterial", "vascular_territories"})


def resolve_atlas(
    atlas: AtlasLike,
    atlas_kwargs: dict[str, Any] | None,
) -> ResolvedAtlas:
    """
    Normalize *atlas* spec into :class:`ResolvedAtlas`.

    Presets (volume): ``destrieux_2009``, ``destrieux_vol`` →
    :func:`nilearn.datasets.fetch_atlas_destrieux_2009`.

    Presets (surface): ``destrieux_surface``, ``surf_destrieux`` →
    :func:`nilearn.datasets.fetch_atlas_surf_destrieux`.

    Presets (located, not fetched): ``desikan`` / ``aparc`` →
    :func:`resolve_atlas_preset_desikan`; ``vascular`` / ``arterial`` →
    :func:`resolve_atlas_preset_vascular`. See :mod:`nvitk.viz.atlas_sources` for where they are
    searched for.

    Custom mapping atlas::

        {"kind": "volume", "maps": path_or_img, "labels": {1: "L_MCA", ...}}
        # labels optional; list[str] uses integer indices 0..n-1 as atlas IDs.

    Aliases: ``kind`` may be omitted if ``maps`` is present (infer volume).
    """
    atlas_kwargs = dict(atlas_kwargs or {})

    if isinstance(atlas, str):
        key = atlas.strip().casefold().replace("-", "_")
        if key in _PRESET_VOLUME:
            return resolve_atlas_preset_destrieux_vol(atlas_kwargs)
        if key in _PRESET_SURFACE:
            return resolve_atlas_preset_surf_destrieux(atlas_kwargs)
        if key in _PRESET_DESIKAN_LOBES:
            return resolve_atlas_preset_desikan_lobes(atlas_kwargs)
        if key in _PRESET_DESIKAN:
            return resolve_atlas_preset_desikan(atlas_kwargs)
        if key in _PRESET_VASCULAR:
            return resolve_atlas_preset_vascular(atlas_kwargs)
        raise ValidationError(
            f"Unknown atlas preset {atlas!r}. Use one of "
            f"{sorted(_PRESET_VOLUME | _PRESET_SURFACE | _PRESET_DESIKAN | _PRESET_DESIKAN_LOBES | _PRESET_VASCULAR)} "
            f"or a mapping spec."
        )

    if not isinstance(atlas, Mapping):
        raise ValidationError("atlas must be a preset string or a mapping specification.")

    spec = dict(atlas)
    kind = spec.get("kind") or spec.get("type")
    maps = spec.get("maps")

    if maps is not None:
        inferred = "volume"
    elif kind is None:
        raise ValidationError(
            "Atlas mapping must include 'maps' (volume path/image) or a supported 'kind'."
        )
    else:
        inferred = str(kind).casefold()

    if inferred in {"volume", "volume_nifti", "nifti"}:
        return resolve_atlas_volume_custom(maps, spec.get("labels"))

    raise ValidationError(f"Unsupported atlas kind {kind!r}.")


# ---------------------------------------------------------------------------
# Region naming
# ---------------------------------------------------------------------------
#: ``ctx_``/``ctx-`` prefix, then a side token, in either order and with any separator.
_SIDE_PREFIX_RE = re.compile(r"^(?:ctx_)?(?:(lh|rh|bh|left|right|l|r)_)")
_SIDE_SUFFIX_RE = re.compile(r"_(lh|rh|bh|left|right)$")
#: ASL parcels carry the smoothing kernel as a trailing ``_0`` / ``_8`` / ``_12``. Not anatomy.
_KERNEL_SUFFIX_RE = re.compile(r"_(?:0|8|12)$")

_SIDE_CANONICAL: dict[str, str] = {
    "lh": "l", "left": "l", "l": "l",
    "rh": "r", "right": "r", "r": "r",
    # ``bh`` is a bilateral measurement — one number for both sides, so it belongs on both.
    "bh": "", "both": "",
}

#: Stems that mean "every parcel", optionally restricted to one side.
_WHOLE_STEMS: frozenset[str] = frozenset(
    {"wholebrain", "brain", "hemisphere", "cortex", "wholehead"}
)


def _squash(name: Any) -> str:
    """Lowercase, separator-free form of a region name (``ctx-lh-Precuneus`` → ``ctxlhprecuneus``)."""
    return re.sub(r"[^0-9a-z]+", "", str(name or "").strip().lower())


@lru_cache(maxsize=4096)
def _normalize_region_key_cached(name: str) -> tuple[str, str]:
    """Memoized core of :func:`normalize_region_key` (pure string → string)."""
    text = re.sub(r"[\s\-.]+", "_", name.strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    text = _KERNEL_SUFFIX_RE.sub("", text)

    side = ""
    match = _SIDE_PREFIX_RE.match(text)
    if match:
        side = _SIDE_CANONICAL.get(match.group(1), "")
        text = text[match.end():]
    else:
        match = _SIDE_SUFFIX_RE.search(text)
        if match:
            side = _SIDE_CANONICAL.get(match.group(1), "")
            text = text[: match.start()]

    stem = _squash(text)
    # A leading bare ``ctx`` survives when there was no side token at all (``ctx-whole-brain``).
    if stem.startswith("ctx") and len(stem) > 3:
        stem = stem[3:]
    return side, stem


def normalize_region_key(name: Any) -> tuple[str, str]:
    """
    Reduce a region name to ``(side, stem)``, whichever vocabulary it was written in.

    *side* is ``"l"``, ``"r"``, or ``""`` for something bilateral or midline; *stem* is the parcel
    with every separator, side token and smoothing suffix removed, so it compares across spellings.

    Examples
    --------
    >>> normalize_region_key("ctx-lh-precuneus")
    ('l', 'precuneus')
    >>> normalize_region_key("ctx_lh_precuneus") == normalize_region_key("lh.precuneus")
    True
    >>> normalize_region_key("left_precuneus")          # T1 spelling
    ('l', 'precuneus')
    >>> normalize_region_key("ctx_bh_precuneus")        # bilateral: belongs on both sides
    ('', 'precuneus')
    >>> normalize_region_key("Left_MCA-8")              # ASL vascular parcel + smoothing kernel
    ('l', 'mca')
    >>> normalize_region_key("ctx-Left-Frontal-Lobe")
    ('l', 'frontallobe')
    """
    # Memoized on the string: resolving a fit re-normalizes the same few dozen names once per
    # model term, which is thousands of identical regex passes on a 68-parcel atlas.
    return _normalize_region_key_cached(str(name or ""))


def _atlas_index_by_key(index_to_label: Mapping[int, str]) -> dict[tuple[str, str], list[int]]:
    """``(side, stem) -> [atlas index]`` for an atlas's own labels."""
    out: dict[tuple[str, str], list[int]] = {}
    for index, label in index_to_label.items():
        out.setdefault(normalize_region_key(label), []).append(int(index))
    return out


#: Published aggregates that are not lobes. The ASL tables report an ``Anterior Cingulate`` summary
#: alongside the parcels it pools, and the lobe table has no entry for it — ``Cingulate`` there is
#: the whole gyrus, anterior and posterior together.
_EXTRA_AGGREGATES: dict[str, tuple[str, ...]] = {
    "anteriorcingulate": ("rostralanteriorcingulate", "caudalanteriorcingulate"),
    "posteriorcingulate": ("posteriorcingulate", "isthmuscingulate"),
}


@lru_cache(maxsize=1)
def _lobe_stems() -> dict[str, tuple[str, ...]]:
    """``squashed aggregate name -> parcel stems``, from the project's own Desikan lobe grouping."""
    from nvitk.stats.region_groups import DESIKAN_LOBES

    table: dict[str, tuple[str, ...]] = {}
    for lobe, parcels in DESIKAN_LOBES.items():
        stems = tuple(_squash(p) for p in parcels)
        key = _squash(lobe)
        table[key] = stems
        # Published aggregates append "lobe" (``ctx-Left-Frontal-Lobe``); the panel names do not.
        table[f"{key}lobe"] = stems
    # Added last so a genuine parcel of the same name is never shadowed — ``posteriorcingulate`` is
    # both a Desikan parcel and the name of a two-parcel summary, and the parcel must win. The
    # exact-parcel lookup in :func:`atlas_indices_for_region` runs first, so this only ever applies
    # to a level that is not itself a parcel of the atlas in hand.
    for key, stems in _EXTRA_AGGREGATES.items():
        table.setdefault(key, tuple(_squash(s) for s in stems))
    return table


def region_index_resolver(index_to_label: Mapping[int, str]):
    """
    A resolver closure with the atlas's inverse index built **once**.

    :func:`atlas_indices_for_region` is called per model term, and rebuilding a 68-entry inverse map
    on each of those made resolving a single fit the most expensive thing the brain map did — it
    dominated the draw. Callers that resolve many names against one atlas should use this instead.

    Returns
    -------
    callable
        ``resolve(label) -> list[int]``, with the same semantics as
        :func:`atlas_indices_for_region`.
    """
    by_key = _atlas_index_by_key(index_to_label)
    lobes = _lobe_stems()

    def resolve(name: Any) -> list[int]:
        """Atlas indices *name* refers to (possibly several, possibly none)."""
        return _indices_for(name, by_key, lobes)

    return resolve


def atlas_indices_for_region(name: Any, index_to_label: Mapping[int, str]) -> list[int]:
    """
    Atlas indices a region level refers to — **one, several, or none**.

    Several, because the grouping levels a model is fitted at are not all single parcels. A frame
    grouped by hemisphere reports ``precuneus`` for the average of both, a lobe-grouped one reports
    ``ctx-Left-Frontal-Lobe``, and the ASL tables publish ``ctx-whole-brain``. Each is one number
    that belongs on every parcel it was averaged from — painting it on none, which an exact-match
    lookup does, leaves a lobe-grouped fit with an empty brain. This mirrors
    :func:`nvitk.stats.vascular_map.nodes_for_label`, which expands a hemisphere-melted ``ICA`` onto
    both carotids for the same reason.

    Resolution order: exact parcel → whole-brain / hemisphere → Desikan lobe → nothing.

    Returns ``[]`` for anything the atlas does not cover, which is how a caller tells a region that
    was measured but cannot be drawn (a FLAIR white-matter zone, a whole-head scalar such as
    ``etiv``) from one that simply has no value.
    """
    return _indices_for(name, _atlas_index_by_key(index_to_label), _lobe_stems())


def _indices_for(
    name: Any,
    by_key: dict[tuple[str, str], list[int]],
    lobes: dict[str, tuple[str, ...]],
) -> list[int]:
    """Resolution shared by :func:`atlas_indices_for_region` and :func:`region_index_resolver`."""
    side, stem = normalize_region_key(name)
    if not stem:
        return []

    def _sided(want: str) -> list[int]:
        """Indices for stem *want*, on the requested side or on both when none was given."""
        if side:
            return list(by_key.get((side, want), ()))
        found: list[int] = []
        for s in ("l", "r", ""):
            found += by_key.get((s, want), [])
        return found

    # ---- 1. An actual region of this atlas -----------------------------------------------------
    exact = _sided(stem)
    if exact:
        return sorted(set(exact))

    # ---- 2. A lobe *as a region*, on an atlas whose regions are lobes ---------------------------
    # The ``lobe`` grouping emits the bare panel name (``Frontal``), while a lobe atlas names its
    # regions the way the pipeline publishes them (``ctx-Left-Frontal-Lobe`` → stem ``frontallobe``).
    # Without this the one grouping the lobe atlas exists for resolves to nothing.
    if stem in lobes:
        as_region = _sided(f"{stem}lobe")
        if as_region:
            return sorted(set(as_region))

    # ---- 3. Whole brain / one hemisphere ------------------------------------------------------
    if stem in _WHOLE_STEMS:
        return sorted(
            index for (parcel_side, _), indices in by_key.items()
            for index in indices
            if not side or parcel_side == side
        )

    # ---- 4. A lobe expanded into the parcels it pools -------------------------------------------
    # Last, so on a parcel atlas ``Frontal`` paints its eleven parcels, while on a lobe atlas step 2
    # has already matched the lobe region itself.
    stems = lobes.get(stem)
    if stems:
        found: list[int] = []
        for parcel in stems:
            found += _sided(parcel)
        return sorted(set(found))

    return []


def _threshold_for_plot(threshold: float | None, vmin: float | None, vmax: float | None) -> float | None:
    """Normalize threshold: Nilearn hides values below threshold magnitude for symmetric cmap."""
    if threshold is None:
        return None
    if vmin is not None and vmax is not None:
        lo = float(min(vmin, vmax))
        hi = float(max(vmin, vmax))
        if threshold <= lo or threshold >= hi:
            return None
    return float(threshold)


def build_volume_stat_image(
    atlas_img: Nifti1Image,
    index_to_value: Mapping[int, float],
    *,
    hemisphere: Hemisphere = "both",
    background: float = np.nan,
) -> Nifti1Image:
    """Paint voxel-wise stat map from integer atlas labels."""
    data = np.asanyarray(atlas_img.dataobj).astype(np.int64, copy=False)
    stat = np.full(data.shape, background, dtype=float)
    aff = atlas_img.affine

    if hemisphere in {"left", "right"}:
        _require_nilearn()
        from nilearn import image as nl_image

        shape = data.shape
        grids = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        x, _, _ = nl_image.coord_transform(
            grids[0].ravel(), grids[1].ravel(), grids[2].ravel(), aff
        )
        x = np.asarray(x).reshape(shape)
        hemi_mask = x < 0 if hemisphere == "left" else x >= 0
    elif hemisphere == "both":
        hemi_mask = np.ones(data.shape, dtype=bool)
    else:
        raise ValidationError("hemisphere must be 'both', 'left', or 'right'.")

    for idx, val in index_to_value.items():
        mask = (data == int(idx)) & hemi_mask
        stat[mask] = float(val)

    return nib.Nifti1Image(stat, aff)


def _vertex_texture_from_maps(
    map_arr: np.ndarray,
    index_to_value: Mapping[int, float],
) -> np.ndarray:
    """Paint a per-vertex scalar texture by mapping each region-index vertex to its value (NaN elsewhere)."""
    tex = np.full(map_arr.shape, np.nan, dtype=float)
    for idx, val in index_to_value.items():
        tex[map_arr == int(idx)] = float(val)
    return tex


def brainshow(
    values: Any,
    *,
    atlas: AtlasLike,
    mode: Mode = "surface",
    region_order: Sequence[Any] | None = None,
    labels: Sequence[Any] | None = None,
    hemisphere: Hemisphere = "both",
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    threshold: float | None = None,
    title: str | None = None,
    axes: Any | None = None,
    output_file: str | Path | None = None,
    show: bool = True,
    atlas_kwargs: Mapping[str, Any] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """
    Render brain regions colored by *values* on *atlas* using Nilearn.

    Parameters
    ----------
    values
        Per-region numeric values: sequence aligned with ``region_order`` / ``labels``,
        or mapping from region label (str) or atlas index (int) -> float.
    atlas
        Preset name or mapping. Volume presets: ``destrieux_2009``, ``destrieux_vol``.
        Surface presets: ``destrieux_surface``, ``surf_destrieux``.
        Custom volume: ``{"kind": "volume", "maps": path_or_Nifti1Image, "labels": ...}``.
    mode
        ``surface`` (mesh / projected volume) or ``volume`` (glass brain / ortho stat map).
    region_order, labels
        Explicit territory ordering for sequence *values*. ``labels`` is an alias of
        ``region_order``.
    hemisphere
        ``both``, ``left``, or ``right`` (masks volumetric painting by MNI x sign).
    cmap, vmin, vmax, threshold
        Passed to Nilearn plotting (threshold clipped against vmin/vmax when both set).
    title
        Figure title where supported.
    axes
        Optional matplotlib axes for volumetric ``plot_stat_map`` only.
    output_file
        Save figure path when supported by Nilearn.
    show
        Call matplotlib.pyplot.show() when True and backend allows.
    atlas_kwargs
        Extra kwargs for Nilearn fetch functions when using presets.
    plot_kwargs
        Additional kwargs forwarded to Nilearn plotting functions.

    Returns
    -------
    Figure or list
        Nilearn figure handle(s); exact type depends on mode and atlas flavor.

    Raises
    ------
    ImportError
        If Nilearn is not installed.
    ValidationError
        If inputs are inconsistent with the atlas or mapping rules.
    """
    _require_nilearn()
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting, surface

    ro = _effective_region_order(region_order, labels)
    ak = dict(atlas_kwargs or {})
    pk = dict(plot_kwargs or {})

    resolved = resolve_atlas(atlas, ak)
    index_to_value = build_index_to_value(values, region_order=ro, index_to_label=resolved.index_to_label)

    thr = _threshold_for_plot(threshold, vmin, vmax)

    figs: list[Any] = []

    if resolved.flavor == "volume_nifti":
        assert resolved.atlas_img is not None
        stat_img = build_volume_stat_image(
            resolved.atlas_img,
            index_to_value,
            hemisphere=hemisphere,
        )

        if mode == "volume":
            kw = {
                "cmap": cmap,
                "threshold": thr,
                "title": title,
                "output_file": output_file,
                **pk,
            }
            if vmin is not None:
                kw["vmin"] = vmin
            if vmax is not None:
                kw["vmax"] = vmax
            if axes is not None:
                kw["axes"] = axes
            fig = plotting.plot_stat_map(stat_img, cut_coords=None, display_mode="ortho", **kw)
            figs.append(fig)
        else:
            meshes = datasets.fetch_surf_fsaverage(mesh=resolved.surf_mesh)
            surf_kw = {
                "cmap": cmap,
                "threshold": thr,
                **pk,
            }
            if vmin is not None:
                surf_kw["vmin"] = vmin
            if vmax is not None:
                surf_kw["vmax"] = vmax

            pairs: list[tuple[str, str]] = []
            if hemisphere in {"both", "left"}:
                pairs.append(("left", meshes["pial_left"]))
            if hemisphere in {"both", "right"}:
                pairs.append(("right", meshes["pial_right"]))

            for _hemi_name, mesh in pairs:
                texture = surface.vol_to_surf(
                    stat_img,
                    mesh,
                    interpolation="nearest",
                    radius=2,
                )
                fig = plotting.plot_surf_stat_map(
                    mesh,
                    texture,
                    title=title if len(pairs) == 1 else f"{title or ''} ({_hemi_name})".strip(),
                    output_file=str(output_file) if output_file is not None and len(pairs) == 1 else None,
                    **surf_kw,
                )
                figs.append(fig)

    elif resolved.flavor == "surface_fs_vertex":
        if mode != "surface":
            raise ValidationError(
                "Native surface atlases only support mode='surface'. "
                "Use a volumetric atlas preset/custom maps for mode='volume'."
            )
        assert resolved.map_left is not None and resolved.map_right is not None
        meshes = datasets.fetch_surf_fsaverage(mesh=resolved.surf_mesh)
        tex_l = _vertex_texture_from_maps(resolved.map_left, index_to_value)
        tex_r = _vertex_texture_from_maps(resolved.map_right, index_to_value)

        surf_kw = {
            "cmap": cmap,
            "threshold": thr,
            **pk,
        }
        if vmin is not None:
            surf_kw["vmin"] = vmin
        if vmax is not None:
            surf_kw["vmax"] = vmax

        pairs = []
        if hemisphere in {"both", "left"}:
            pairs.append(("left", meshes["pial_left"], tex_l))
        if hemisphere in {"both", "right"}:
            pairs.append(("right", meshes["pial_right"], tex_r))

        for hemi_name, mesh, tex in pairs:
            fig = plotting.plot_surf_stat_map(
                mesh,
                tex,
                title=title if len(pairs) == 1 else f"{title or ''} ({hemi_name})".strip(),
                output_file=str(output_file) if output_file is not None and len(pairs) == 1 else None,
                **surf_kw,
            )
            figs.append(fig)
    else:
        raise ValidationError(f"Unhandled atlas flavor {resolved.flavor!r}.")

    if show:
        plt.show()

    return figs[0] if len(figs) == 1 else figs


__all__ = [
    "RIGHT_HEMI_INDEX_OFFSET",
    "ResolvedAtlas",
    "VASCULAR_ATLAS_LABELS",
    "atlas_indices_for_region",
    "region_index_resolver",
    "brainshow",
    "build_index_to_value",
    "build_volume_stat_image",
    "normalize_region_key",
    "resolve_atlas",
    "resolve_atlas_desikan_annot",
    "merge_atlas_regions",
    "resolve_atlas_preset_desikan",
    "resolve_atlas_preset_desikan_lobes",
    "resolve_atlas_preset_vascular",
]
