"""FSL's atlases (``$FSLDIR/data/atlases/*.xml``) and standard-space templates.

Each atlas XML names its images per resolution: a probabilistic 4D stack (``imagefile``) and a
summary label image (``summaryimagefile``). FSL's own convention for the summary image's voxel
values: a ``Label`` atlas stores the label's ``index`` itself; a ``Probabilistic`` atlas's
max-probability image stores ``index + 1`` (0 is background). Both are honoured here, so the
label table matches the image.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from nvitk.atlases.base import INSTALLED, AtlasData, AtlasInfo, Label, labels_from_names

SOURCE = "fsl"

#: Standard-space templates offered next to the atlases (name -> file stem under data/standard).
TEMPLATES: dict[str, tuple[str, str]] = {
    "MNI152_T1": ("MNI152 T1 template", "MNI152_T1_{res}"),
    "MNI152_T1_brain": ("MNI152 T1 template, brain", "MNI152_T1_{res}_brain"),
    "MNI152_T1_brain_mask": ("MNI152 brain mask", "MNI152_T1_{res}_brain_mask"),
    "FMRIB58_FA": ("FMRIB58 FA template", "FMRIB58_FA_{res}"),
}


def fsldir() -> Path | None:
    """The FSL installation nvitk uses (``nvitk.fsl.env.resolve_fsldir``), or ``None``."""
    try:
        from nvitk.fsl.env import resolve_fsldir

        return resolve_fsldir()
    except Exception:  # noqa: BLE001
        return None


def _nii(base: Path, stem: str) -> Path | None:
    for suffix in (".nii.gz", ".nii"):
        candidate = base / f"{stem.lstrip('/')}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _resolution(stem: str) -> str:
    found = re.search(r"(\d+(?:\.\d+)?mm)", stem)
    return found.group(1) if found else stem.rsplit("/", 1)[-1]


def _mm(resolution: str) -> float:
    found = re.match(r"(\d+(?:\.\d+)?)mm", resolution)
    return float(found.group(1)) if found else 1e9


def parse_atlas_xml(path: Path) -> dict[str, Any]:
    """Name, type, images per resolution and labels of one FSL atlas XML."""
    root = ET.parse(path).getroot()
    header = root.find("header")
    name = (header.findtext("name") or path.stem).strip() if header is not None else path.stem
    kind = (header.findtext("type") or "Label").strip() if header is not None else "Label"
    images = []
    for block in header.findall("images") if header is not None else []:
        image = (block.findtext("imagefile") or "").strip()
        summary = (block.findtext("summaryimagefile") or image).strip()
        images.append({"resolution": _resolution(summary or image), "image": image, "summary": summary})
    labels = [
        (int(node.get("index", "0")), (node.text or "").strip())
        for node in root.iter("label")
    ]
    return {"name": name, "type": kind, "images": images, "labels": labels,
            "shortname": (header.findtext("shortname") or "").strip() if header is not None else ""}


def list_atlases() -> list[AtlasInfo]:
    """Every FSL atlas and standard template present in the FSL installation."""
    home = fsldir()
    if home is None:
        return []
    out: list[AtlasInfo] = []
    atlas_dir = home / "data" / "atlases"
    for xml in sorted(atlas_dir.glob("*.xml")):
        try:
            meta = parse_atlas_xml(xml)
        except ET.ParseError:
            continue
        present = [img for img in meta["images"] if _nii(atlas_dir, img["summary"])]
        if not present:
            continue
        # Finest resolution first: it is the default a caller gets without asking.
        present.sort(key=lambda img: _mm(img["resolution"]))
        probabilistic = meta["type"].lower().startswith("prob")
        out.append(
            AtlasInfo(
                id=f"{SOURCE}:{xml.stem}",
                name=meta["name"],
                source=SOURCE,
                kind="probabilistic" if probabilistic else "labels",
                space="MNI152",
                status=INSTALLED,
                description=f"FSL atlas ({meta['type']}), {len(meta['labels'])} regions.",
                variants=tuple(img["resolution"] for img in present),
                extra={"xml": str(xml), "images": present, "probabilistic": probabilistic},
            )
        )
    standard = home / "data" / "standard"
    for key, (title, pattern) in TEMPLATES.items():
        variants = tuple(res for res in ("1mm", "2mm", "0.5mm") if _nii(standard, pattern.format(res=res)))
        if variants:
            out.append(
                AtlasInfo(
                    id=f"{SOURCE}:{key}",
                    name=title,
                    source=SOURCE,
                    kind="template" if "mask" not in key else "labels",
                    space="MNI152",
                    description=f"FSL standard-space image ({', '.join(variants)}).",
                    variants=variants,
                    extra={"pattern": pattern, "standard": str(standard)},
                )
            )
    return out


def load(info: AtlasInfo, *, variant: str = "", download: bool = False) -> AtlasData:
    """Image path(s) and label table of an FSL atlas or template."""
    del download
    if "pattern" in info.extra:
        image = _nii(Path(info.extra["standard"]), info.extra["pattern"].format(res=variant))
        labels = [Label(0, "Background", (0, 0, 0)), Label(1, "Brain", (230, 200, 120))] if info.kind == "labels" else []
        return AtlasData(info=info, image=image, labels=labels, variant=variant)
    meta = parse_atlas_xml(Path(info.extra["xml"]))
    atlas_dir = Path(info.extra["xml"]).parent
    chosen = next((img for img in info.extra["images"] if img["resolution"] == variant), info.extra["images"][0])
    offset = 1 if info.extra.get("probabilistic") else 0
    pairs = [(index + offset, name) for index, name in meta["labels"]]
    if offset and all(i != 0 for i, _ in pairs):
        pairs = [(0, "Background")] + pairs
    files: dict[str, Path] = {}
    stack = _nii(atlas_dir, chosen["image"])
    if stack is not None and chosen["image"] != chosen["summary"]:
        files["probabilistic"] = stack
    return AtlasData(
        info=info,
        image=_nii(atlas_dir, chosen["summary"]),
        labels=labels_from_names(pairs),
        variant=variant,
        files=files,
    )


__all__ = ["SOURCE", "fsldir", "list_atlases", "load", "parse_atlas_xml"]
