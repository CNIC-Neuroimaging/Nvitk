"""Atlases a site configures by path: ``atlas.json`` (and its env vars).

These are the lab's own files nothing can fetch (the Desikan ``aparc+aseg`` in MNI space, the
arterial-territory atlas), located by :mod:`nvitk.viz.atlas_sources`. The Desikan one is named
through FreeSurfer's colour LUT when one is found.
"""

from __future__ import annotations

from pathlib import Path

from nvitk.atlases.base import AtlasData, AtlasInfo, labels_from_names

SOURCE = "configured"


def list_atlases() -> list[AtlasInfo]:
    """The configured Desikan and vascular atlases that resolve to a file."""
    from nvitk.viz import atlas_sources

    out = []
    for key, resolver, title in (
        ("desikan", atlas_sources.desikan_atlas_path, "Desikan-Killiany aparc+aseg (configured)"),
        ("vascular", atlas_sources.vascular_atlas_path, "Arterial territories / watershed (configured)"),
    ):
        try:
            path = resolver()
        except Exception:  # noqa: BLE001
            path = None
        if path is not None and Path(path).is_file():
            out.append(
                AtlasInfo(
                    id=f"{SOURCE}:{key}",
                    name=title,
                    source=SOURCE,
                    kind="labels",
                    space="MNI152",
                    description=f"{path} (atlas.json / {atlas_sources.ENV_ATLAS_ROOT}).",
                    extra={"image": str(path), "key": key},
                )
            )
    return out


def load(info: AtlasInfo, *, variant: str = "", download: bool = False) -> AtlasData:
    """The file, named by the FreeSurfer LUT for Desikan, by id otherwise."""
    del download
    image = Path(info.extra["image"])
    labels = []
    if info.extra.get("key") == "desikan":
        from nvitk.atlases.freesurfer import lut_path, parse_lut

        lut = lut_path()
        labels = parse_lut(lut) if lut is not None else []
    if not labels:
        import nibabel as nib
        import numpy as np

        values = np.unique(np.asarray(nib.load(str(image)).dataobj)).astype(int)
        labels = labels_from_names((int(v), "Background" if v == 0 else f"Region {int(v)}") for v in values[:2000])
    return AtlasData(info=info, image=image, labels=labels, variant=variant)


__all__ = ["SOURCE", "list_atlases", "load"]
