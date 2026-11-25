"""ANTsXNet templates and label priors (``antspynet.utilities.get_antsxnet_data``).

Listing reads the cache directory only (``~/.keras/ANTsXNet``, or ``$ANTSXNET_CACHE_DIRECTORY``),
so it neither imports TensorFlow nor touches the network. Anything not cached is
``downloadable``; :func:`load` fetches it through ANTsPyNet only with ``download=True``.
"""

from __future__ import annotations

import os
from pathlib import Path

from nvitk.atlases.base import DOWNLOADABLE, INSTALLED, AtlasData, AtlasInfo, labels_from_names

SOURCE = "antsxnet"

#: Curated ANTsXNet data ids: id -> (title, kind, space, species).
CATALOG: dict[str, tuple[str, str, str, str]] = {
    "mni152": ("MNI152 template (ANTsXNet)", "template", "MNI152", "human"),
    "croppedMni152": ("MNI152, cropped (ANTsXNet brain extraction space)", "template", "MNI152", "human"),
    "croppedMni152Priors": ("MNI152 tissue priors (cropped)", "probabilistic", "MNI152", "human"),
    "biobank": ("UK Biobank T1 template", "template", "Biobank", "human"),
    "adni": ("ADNI T1 template", "template", "ADNI", "human"),
    "ixi": ("IXI T1 template", "template", "IXI", "human"),
    "kirby": ("Kirby T1 template", "template", "Kirby", "human"),
    "nki": ("NKI T1 template", "template", "NKI", "human"),
    "nki10": ("NKI10 T1 template", "template", "NKI", "human"),
    "oasis": ("OASIS T1 template", "template", "OASIS", "human"),
    "priorDktLabels": ("DKT label priors", "probabilistic", "MNI152", "human"),
    "deepFlashTemplateT1SkullStripped": ("DeepFLASH T1 template (skull-stripped)", "template", "DeepFLASH", "human"),
    "deepFlashTemplate2Labels": ("DeepFLASH medial temporal lobe labels", "labels", "DeepFLASH", "human"),
    "hcpyaT1Template": ("HCP-YA T1 template", "template", "HCP-YA", "human"),
    "hcpyaT2Template": ("HCP-YA T2 template", "template", "HCP-YA", "human"),
    "hcpyaFATemplate": ("HCP-YA FA template", "template", "HCP-YA", "human"),
    "hcpyaTemplateBrainMask": ("HCP-YA brain mask", "labels", "HCP-YA", "human"),
    "hcpyaTemplateBrainSegmentation": ("HCP-YA tissue segmentation", "labels", "HCP-YA", "human"),
    "hcpaT1Template": ("HCP-Aging T1 template", "template", "HCP-A", "human"),
    "mraTemplate": ("MRA template", "template", "MRA", "human"),
    "mraTemplateBrainMask": ("MRA template brain mask", "labels", "MRA", "human"),
    "mraTemplateVesselPrior": ("MRA vessel prior", "probabilistic", "MRA", "human"),
    "ctLungTemplate": ("CT lung template", "template", "CT lung", "human"),
    "bsplineT2MouseTemplate": ("Mouse T2 template (B-spline)", "template", "mouse T2", "mouse"),
    "bsplineT2MouseTemplateBrainMask": ("Mouse T2 template brain mask", "labels", "mouse T2", "mouse"),
    "DevCCF_P56_MRI-T2_50um": ("DevCCF P56 mouse T2 (50 µm)", "template", "DevCCF P56", "mouse"),
    "DevCCF_P56_MRI-T2_50um_BrainParcellationNickMask": ("DevCCF P56 brain parcellation", "labels", "DevCCF P56", "mouse"),
}


def cache_dir() -> Path:
    """Where ANTsPyNet keeps its downloads."""
    raw = os.environ.get("ANTSXNET_CACHE_DIRECTORY", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".keras" / "ANTsXNet"


def _cached(data_id: str) -> Path | None:
    path = cache_dir() / f"{data_id}.nii.gz"
    return path if path.is_file() else None


def list_atlases() -> list[AtlasInfo]:
    """The curated ANTsXNet templates and priors, cached or downloadable."""
    out = []
    for data_id, (title, kind, space, species) in CATALOG.items():
        out.append(
            AtlasInfo(
                id=f"{SOURCE}:{data_id}",
                name=title,
                source=SOURCE,
                kind=kind,
                space=space,
                species=species,
                status=INSTALLED if _cached(data_id) else DOWNLOADABLE,
                description=f"ANTsXNet data '{data_id}'.",
                citation="Tustison et al., The ANTsX ecosystem, Sci Rep 2021.",
                extra={"data_id": data_id},
            )
        )
    return out


def load(info: AtlasInfo, *, variant: str = "", download: bool = False) -> AtlasData:
    """The cached file, fetching it first when *download* is set."""
    data_id = info.extra["data_id"]
    path = _cached(data_id)
    if path is None:
        if not download:
            raise FileNotFoundError(f"ANTsXNet data {data_id!r} is not cached in {cache_dir()}")
        from antspynet.utilities import get_antsxnet_data

        path = Path(get_antsxnet_data(data_id))
    labels = []
    if info.kind == "labels":
        import nibabel as nib
        import numpy as np

        values = np.unique(np.asarray(nib.load(str(path)).dataobj)).astype(int)
        labels = labels_from_names((int(v), "Background" if v == 0 else f"Label {int(v)}") for v in values[:1000])
    return AtlasData(info=info, image=path, labels=labels, variant=variant)


__all__ = ["CATALOG", "SOURCE", "cache_dir", "list_atlases", "load"]
