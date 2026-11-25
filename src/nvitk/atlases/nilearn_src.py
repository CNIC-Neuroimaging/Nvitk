"""nilearn's atlas fetchers (``nilearn.datasets.fetch_atlas_*``).

Listing checks nilearn's data directory for each fetcher's folder; a cached atlas loads offline
(nilearn reads its cache before trying the network), anything else is ``downloadable``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvitk.atlases.base import DOWNLOADABLE, INSTALLED, AtlasData, AtlasInfo, labels_from_names

SOURCE = "nilearn"

#: id -> (title, fetcher, kwargs, cache folder, kind); :data:`PROBES` narrows folders several
#: fetchers share (nilearn keeps Harvard-Oxford and Juelich both under ``fsl/``).
CATALOG: dict[str, tuple[str, str, dict[str, Any], str, str]] = {
    "harvard_oxford_cort": ("Harvard-Oxford cortical (max-prob 25 %, 1 mm)", "fetch_atlas_harvard_oxford", {"atlas_name": "cort-maxprob-thr25-1mm"}, "fsl", "labels"),
    "harvard_oxford_sub": ("Harvard-Oxford subcortical (max-prob 25 %, 1 mm)", "fetch_atlas_harvard_oxford", {"atlas_name": "sub-maxprob-thr25-1mm"}, "fsl", "labels"),
    "aal": ("AAL (SPM12)", "fetch_atlas_aal", {}, "aal_SPM12", "labels"),
    "destrieux": ("Destrieux 2009", "fetch_atlas_destrieux_2009", {}, "destrieux_2009", "labels"),
    "schaefer_400": ("Schaefer 2018, 400 parcels / 7 networks", "fetch_atlas_schaefer_2018", {"n_rois": 400}, "schaefer_2018", "labels"),
    "schaefer_100": ("Schaefer 2018, 100 parcels / 7 networks", "fetch_atlas_schaefer_2018", {"n_rois": 100}, "schaefer_2018", "labels"),
    "juelich": ("Juelich (max-prob 25 %, 1 mm)", "fetch_atlas_juelich", {"atlas_name": "maxprob-thr25-1mm"}, "fsl", "labels"),
    "talairach_ba": ("Talairach (Brodmann areas)", "fetch_atlas_talairach", {"level_name": "ba"}, "talairach_atlas", "labels"),
    "yeo_7": ("Yeo 2011, 7 networks", "fetch_atlas_yeo_2011", {}, "yeo_2011", "labels"),
    "pauli_det": ("Pauli 2017 subcortical (deterministic)", "fetch_atlas_pauli_2017", {"atlas_type": "deterministic"}, "pauli_2017", "labels"),
    "difumo_64": ("DiFuMo 64 components", "fetch_atlas_difumo", {"dimension": 64}, "difumo_atlases", "probabilistic"),
}


PROBES: dict[str, str] = {
    "harvard_oxford_cort": "HarvardOxford-cort-maxprob-thr25-1mm*",
    "harvard_oxford_sub": "HarvardOxford-sub-maxprob-thr25-1mm*",
    "juelich": "Juelich-maxprob-thr25-1mm*",
}


def data_dir() -> Path:
    """nilearn's first data directory (``$NILEARN_DATA`` or ``~/nilearn_data``)."""
    try:
        from nilearn.datasets.utils import get_data_dirs

        return Path(get_data_dirs()[0])
    except Exception:  # noqa: BLE001
        return Path.home() / "nilearn_data"


def list_atlases() -> list[AtlasInfo]:
    """The curated nilearn atlases, cached or downloadable."""
    import importlib.util

    if importlib.util.find_spec("nilearn") is None:
        return []
    base = data_dir()
    out = []
    for key, (title, fetcher, kwargs, folder, kind) in CATALOG.items():
        cached = (base / folder).is_dir() and any((base / folder).rglob(PROBES.get(key, "*.nii*")))
        out.append(
            AtlasInfo(
                id=f"{SOURCE}:{key}",
                name=title,
                source=SOURCE,
                kind=kind,
                space="MNI152",
                status=INSTALLED if cached else DOWNLOADABLE,
                description=f"nilearn.datasets.{fetcher}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})",
                extra={"fetcher": fetcher, "kwargs": kwargs},
            )
        )
    return out


def _labels(bunch: Any) -> list[tuple[int, str]]:
    names = [str(n.decode() if isinstance(n, bytes) else n) for n in (bunch.get("labels") or [])]
    indices = bunch.get("indices")
    if indices is not None and len(indices) == len(names):
        return [(int(i), n) for i, n in zip(indices, names)]
    if names and names[0].lower() == "background":
        return list(enumerate(names))
    return [(0, "Background")] + [(i + 1, n) for i, n in enumerate(names)]


def load(info: AtlasInfo, *, variant: str = "", download: bool = False) -> AtlasData:
    """Run the fetcher (offline when cached) and return its map and labels."""
    import nilearn.datasets as datasets

    if not info.available and not download:
        raise FileNotFoundError(f"{info.id} is not cached in {data_dir()}")
    bunch = getattr(datasets, info.extra["fetcher"])(**info.extra["kwargs"])
    maps = bunch.get("maps")
    image = Path(maps) if isinstance(maps, (str, Path)) else None
    if image is None and maps is not None:
        import tempfile

        image = Path(tempfile.mkdtemp()) / f"{info.id.split(':', 1)[1]}.nii.gz"
        maps.to_filename(str(image))
    labels = labels_from_names(_labels(bunch)) if info.kind == "labels" else []
    return AtlasData(info=info, image=image, labels=labels, variant=variant)


__all__ = ["CATALOG", "SOURCE", "data_dir", "list_atlases", "load"]
