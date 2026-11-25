"""FreeSurfer / FastSurfer label tables and the fsaverage / cvs_avg35 volumes.

The colour LUT names every id an ``aseg`` / ``aparc+aseg`` / ``aparc.DKTatlas+aseg`` volume holds,
so it is offered as an atlas of its own (``kind="lut"``): open a FastSurfer or FreeSurfer
segmentation and this is its legend. It is read from ``$FREESURFER_HOME/FreeSurferColorLUT.txt``,
or, without FreeSurfer, from the copy FastSurfer ships. With a FreeSurfer installation, the
template subjects' label volumes (``fsaverage``, ``cvs_avg35_inMNI152``) are offered as atlases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from nvitk.atlases.base import INSTALLED, AtlasData, AtlasInfo, Label

SOURCE = "freesurfer"

#: Label volumes of FreeSurfer's template subjects: (subject, file, title, space).
TEMPLATE_VOLUMES: tuple[tuple[str, str, str, str], ...] = (
    ("fsaverage", "aparc+aseg.mgz", "fsaverage aparc+aseg (Desikan)", "MNI305"),
    ("fsaverage", "aparc.a2009s+aseg.mgz", "fsaverage aparc.a2009s+aseg (Destrieux)", "MNI305"),
    ("fsaverage", "aseg.mgz", "fsaverage aseg", "MNI305"),
    ("cvs_avg35_inMNI152", "aparc+aseg.mgz", "cvs_avg35 aparc+aseg (MNI152)", "MNI152"),
    ("cvs_avg35_inMNI152", "aseg.mgz", "cvs_avg35 aseg (MNI152)", "MNI152"),
)


def freesurfer_home() -> Path | None:
    """``$FREESURFER_HOME`` (via :func:`nvitk.viz.atlas_sources.freesurfer_home`), or ``None``."""
    try:
        from nvitk.viz.atlas_sources import freesurfer_home as _home

        home = _home()
        if home is not None:
            return Path(home)
    except Exception:  # noqa: BLE001
        pass
    raw = os.environ.get("FREESURFER_HOME", "").strip()
    return Path(raw) if raw and Path(raw).is_dir() else None


def lut_path() -> Path | None:
    """The colour LUT: FreeSurfer's, else the one shipped with FastSurfer."""
    home = freesurfer_home()
    if home is not None and (home / "FreeSurferColorLUT.txt").is_file():
        return home / "FreeSurferColorLUT.txt"
    candidates = [
        Path(sys.prefix) / "share" / "nvitk" / "fastsurfer" / "FastSurfer" / "FastSurferCNN" / "config" / "FreeSurferColorLUT.txt",
    ]
    fastsurfer = os.environ.get("FASTSURFER_HOME", "").strip()
    if fastsurfer:
        candidates.insert(0, Path(fastsurfer) / "FastSurferCNN" / "config" / "FreeSurferColorLUT.txt")
    return next((c for c in candidates if c.is_file()), None)


def parse_lut(path: Path) -> list[Label]:
    """Labels of a FreeSurfer colour LUT (``id name R G B A`` per line; ``#`` comments)."""
    labels: list[Label] = []
    seen: set[int] = set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) < 5 or not parts[0].lstrip("-").isdigit():
            continue
        label_id = int(parts[0])
        if label_id in seen:
            continue
        seen.add(label_id)
        try:
            rgb = (int(parts[2]), int(parts[3]), int(parts[4]))
        except ValueError:
            continue
        labels.append(Label(label_id, parts[1], rgb))
    return labels


def list_atlases() -> list[AtlasInfo]:
    """The colour LUT, and the template subjects' label volumes when FreeSurfer is installed."""
    out: list[AtlasInfo] = []
    lut = lut_path()
    if lut is not None:
        out.append(
            AtlasInfo(
                id=f"{SOURCE}:ColorLUT",
                name="FreeSurfer colour LUT (aseg / aparc / DKT ids)",
                source=SOURCE,
                kind="lut",
                space="native",
                description=f"Label names and colours for FreeSurfer / FastSurfer outputs ({lut}).",
                extra={"lut": str(lut)},
            )
        )
    home = freesurfer_home()
    if home is not None and lut is not None:
        for subject, filename, title, space in TEMPLATE_VOLUMES:
            path = home / "subjects" / subject / "mri" / filename
            if path.is_file():
                out.append(
                    AtlasInfo(
                        id=f"{SOURCE}:{subject}/{filename.split('.mgz')[0]}",
                        name=title,
                        source=SOURCE,
                        kind="labels",
                        space=space,
                        status=INSTALLED,
                        description=f"{subject} label volume from FreeSurfer.",
                        extra={"image": str(path), "lut": str(lut)},
                    )
                )
    return out


def load(info: AtlasInfo, *, variant: str = "", download: bool = False) -> AtlasData:
    """The label table (and image, for a template volume)."""
    del download
    labels = parse_lut(Path(info.extra["lut"]))
    image = Path(info.extra["image"]) if info.extra.get("image") else None
    return AtlasData(info=info, image=image, labels=labels, variant=variant)


__all__ = ["SOURCE", "freesurfer_home", "list_atlases", "load", "lut_path", "parse_lut"]
