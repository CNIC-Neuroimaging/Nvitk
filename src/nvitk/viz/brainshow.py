"""
Atlas-based brain visualization via Nilearn (:func:`brainshow`).

Supports volumetric atlases (custom NIfTI or Nilearn presets such as Destrieux 2009)
and Destrieux fsaverage5 surface atlases. Values are mapped to atlas regions using an
explicit ``region_order`` (sequence alignment) or a mapping keyed by region label /
atlas index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

from nvitk.core.exceptions import ValidationError

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


_PRESET_VOLUME = frozenset({"destrieux_2009", "destrieux_vol"})
_PRESET_SURFACE = frozenset({"destrieux_surface", "surf_destrieux"})


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
        raise ValidationError(
            f"Unknown atlas preset {atlas!r}. "
            f"Use one of {sorted(_PRESET_VOLUME | _PRESET_SURFACE)} or a mapping spec."
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


__all__ = ["brainshow", "ResolvedAtlas", "resolve_atlas", "build_index_to_value", "build_volume_stat_image"]
