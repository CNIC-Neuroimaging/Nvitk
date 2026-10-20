"""What a FastSurfer subject directory holds, and readers for its tables and surfaces.

The paths are ``run_fastsurfer.sh``'s defaults (``$SUBJECTS_DIR/$SID/...``). With
``--native_image`` the same files are written on the input's own grid instead of the
conformed 1 mm one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import config


@dataclass(frozen=True)
class Volume:
    """One volume FastSurfer writes: where, and how to show it."""

    key: str
    relpath: str
    title: str
    labels: bool = False
    #: Lookup table the ids are named and coloured from (see :func:`lut_for`).
    lut: str | None = None
    #: Label vocabulary key registered in :mod:`nvitk.gui.labels.catalog`.
    schema: str | None = None


#: Every volume worth loading, in the order they are added to a viewer.
VOLUMES: tuple[Volume, ...] = (
    Volume("orig", "mri/orig.mgz", "T1 (conformed)"),
    Volume("orig_nu", "mri/orig_nu.mgz", "T1 bias-corrected"),
    Volume("mask", "mri/mask.mgz", "brain mask", labels=True),
    Volume("aseg", "mri/aseg.auto_noCCseg.mgz", "aseg", labels=True, lut="freesurfer", schema="fastsurfer_aseg"),
    Volume(
        "aparc_aseg",
        "mri/aparc.DKTatlas+aseg.deep.mgz",
        "aparc.DKTatlas+aseg",
        labels=True,
        lut="freesurfer",
        schema="fastsurfer_aparc_aseg",
    ),
    Volume(
        "aparc_aseg_cc",
        "mri/aparc.DKTatlas+aseg.deep.withCC.mgz",
        "aparc.DKTatlas+aseg (with CC)",
        labels=True,
        lut="freesurfer",
        schema="fastsurfer_aparc_aseg",
    ),
    # The sub-module maps are written with FreeSurfer's ids (CC 250-255, cerebellum 601-631,
    # hypothalamus 961-984), not their own LUTs' indices: FreeSurferColorLUT names them.
    Volume("callosum", "mri/callosum.CC.orig.mgz", "corpus callosum", labels=True, lut="freesurfer", schema="fastsurfer_cc"),
    Volume("cereb", "mri/cerebellum.CerebNet.nii.gz", "cerebellum (CerebNet)", labels=True, lut="freesurfer", schema="cerebnet"),
    Volume("hypothal", "mri/hypothalamus.HypVINN.nii.gz", "hypothalamus (HypVINN)", labels=True, lut="freesurfer", schema="hypvinn"),
    # Written by the surface pipeline.
    Volume("aparc_mapped", "mri/aparc.DKTatlas+aseg.mapped.mgz", "aparc.DKTatlas+aseg (surface-mapped)", labels=True, lut="freesurfer", schema="fastsurfer_aparc_aseg"),
    Volume("wmparc", "mri/wmparc.DKTatlas.mapped.mgz", "wmparc", labels=True, lut="freesurfer"),
    Volume("ribbon", "mri/ribbon.mgz", "cortical ribbon", labels=True, lut="freesurfer"),
    Volume("brain", "mri/brain.mgz", "brain"),
    Volume("norm", "mri/norm.mgz", "norm"),
)

#: Volumes each module produces (what a module-only run loads).
MODULE_VOLUMES: dict[str, tuple[str, ...]] = {
    "asegdkt": ("orig_nu", "aparc_aseg_cc", "aparc_aseg", "mask"),
    "cereb": ("cereb",),
    "hypothal": ("hypothal",),
    "cc": ("callosum",),
    "surfaces": ("aparc_mapped",),
}

#: Surfaces worth loading, per hemisphere.
SURFACES: tuple[str, ...] = ("white", "pial")


@dataclass(frozen=True)
class SubjectOutputs:
    """The files of one FastSurfer subject (``<sd>/<sid>``)."""

    subject_dir: Path

    def path(self, relpath: str) -> Path:
        return self.subject_dir / relpath

    def volume(self, key: str) -> Volume | None:
        return next((v for v in VOLUMES if v.key == key), None)

    def existing_volumes(self, keys: tuple[str, ...] | None = None) -> list[tuple[Volume, Path]]:
        """``(volume, path)`` for every volume on disk (only *keys* when given, in that order)."""
        chosen = [self.volume(k) for k in keys] if keys else list(VOLUMES)
        return [(v, self.path(v.relpath)) for v in chosen if v is not None and self.path(v.relpath).is_file()]

    def stats_files(self) -> list[Path]:
        return sorted((self.subject_dir / "stats").glob("*.stats"))

    def surface_files(self) -> list[Path]:
        surf = self.subject_dir / "surf"
        return [surf / f"{h}.{name}" for h in ("lh", "rh") for name in SURFACES if (surf / f"{h}.{name}").is_file()]

    def annot(self, hemi: str) -> Path | None:
        for name in (f"{hemi}.aparc.DKTatlas.mapped.annot", f"{hemi}.aparc.DKTatlas.annot", f"{hemi}.aparc.annot"):
            path = self.subject_dir / "label" / name
            if path.is_file():
                return path
        return None

    def thickness(self, hemi: str) -> Path | None:
        path = self.subject_dir / "surf" / f"{hemi}.thickness"
        return path if path.is_file() else None

    def log_file(self) -> Path:
        return self.subject_dir / "scripts" / "deep-seg.log"

    def has_segmentation(self) -> bool:
        """Whether the whole-brain (asegdkt) segmentation exists, so modules can reuse it."""
        return self.path("mri/aparc.DKTatlas+aseg.deep.mgz").is_file()

    def input_t1(self) -> Path | None:
        """The T1 the subject was segmented from: FastSurfer's copy of the input, else the
        conformed one. ``run_fastsurfer.sh`` wants a T1 even when the asegdkt step is skipped."""
        for rel in ("mri/orig/001.mgz", "mri/orig.mgz"):
            if self.path(rel).is_file():
                return self.path(rel)
        return None


# ── lookup tables ────────────────────────────────────────────────────────────

_LUT_FILES = {
    "freesurfer": "FastSurferCNN/config/FreeSurferColorLUT.txt",
    "fastsurfer": "FastSurferCNN/config/FastSurfer_ColorLUT.tsv",
    "cerebnet": "CerebNet/config/CerebNet_ColorLUT.tsv",
    "hypvinn": "HypVINN/config/HypVINN_ColorLUT.txt",
}


def read_lut(path: str | Path) -> dict[int, tuple[str, tuple[int, int, int, int]]]:
    """``{id: (name, (r, g, b, 255))}`` from a FreeSurfer-style ``.txt`` or a ``.tsv`` LUT.

    FreeSurfer's fourth column is transparency (0 = opaque), not alpha, so every
    entry is returned opaque.
    """
    out: dict[int, tuple[str, tuple[int, int, int, int]]] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            lid = int(parts[0])
            rgb = tuple(int(float(v)) for v in parts[2:5])
        except ValueError:
            continue  # a header row
        out[lid] = (parts[1], (rgb[0], rgb[1], rgb[2], 255))
    return out


def lut_for(name: str | None, home: Path | None = None) -> dict[int, tuple[str, tuple[int, int, int, int]]]:
    """The named lookup table from the FastSurfer tree (empty when unavailable)."""
    if not name:
        return {}
    home = home or config.fastsurfer_home()
    rel = _LUT_FILES.get(name)
    if home is None or rel is None or not (home / rel).is_file():
        return {}
    return read_lut(home / rel)


# ── stats ────────────────────────────────────────────────────────────────────


def read_stats(path: str | Path) -> tuple[list[str], list[dict[str, Any]], dict[str, str]]:
    """``(columns, rows, measures)`` from a FreeSurfer-format ``.stats`` file.

    *measures* are the ``# Measure`` header lines (``{name: "value unit"}``) — eTIV,
    brain volume and similar whole-brain numbers.
    """
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    measures: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("# ColHeaders"):
            columns = line.split()[2:]
            continue
        if line.startswith("# Measure"):
            parts = [p.strip() for p in line[len("# Measure"):].split(",")]
            if len(parts) >= 4:
                measures[parts[1] or parts[0]] = f"{parts[3]} {parts[4] if len(parts) > 4 else ''}".strip()
            continue
        if not line or line.startswith("#") or not columns:
            continue
        values = line.split()
        if len(values) < len(columns):
            continue
        row: dict[str, Any] = {}
        for column, value in zip(columns, values):
            try:
                row[column] = int(value)
            except ValueError:
                try:
                    row[column] = float(value)
                except ValueError:
                    row[column] = value
        rows.append(row)
    return columns, rows, measures


def stats_table(path: str | Path) -> dict[str, dict[str, Any]]:
    """A ``.stats`` file as ``{structure: {column: value}}`` (for a results window)."""
    columns, rows, _measures = read_stats(path)
    name_col = "StructName" if "StructName" in columns else (columns[-1] if columns else "")
    keep = [c for c in columns if c not in (name_col, "Index")]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out[str(row.get(name_col, len(out)))] = {c: row.get(c) for c in keep}
    return out


def cc_measures(subject_dir: str | Path) -> dict[str, Any]:
    """The corpus-callosum module's mid-slice shape measures (scalar entries only)."""
    import json

    path = Path(subject_dir) / "stats" / "callosum.CC.midslice.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, (int, float, str))} if isinstance(data, dict) else {}


# ── surfaces ─────────────────────────────────────────────────────────────────


def read_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)`` of a FreeSurfer surface, vertices in scanner-RAS millimetres.

    Surfaces are stored in tkregister space; their ``cras`` (centre of the conformed
    volume in scanner RAS) brings them back onto the images.
    """
    import nibabel as nib

    coords, faces, meta = nib.freesurfer.read_geometry(str(path), read_metadata=True)
    cras = np.asarray(meta.get("cras", (0.0, 0.0, 0.0)), dtype=float)
    return (np.asarray(coords, dtype=float) + cras).astype(np.float32), np.asarray(faces, dtype=np.int64)


def read_annot_values(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """``(per-vertex label index, names)`` of a FreeSurfer ``.annot``."""
    import nibabel as nib

    labels, _ctab, names = nib.freesurfer.read_annot(str(path))
    return np.asarray(labels), [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def read_morph(path: str | Path) -> np.ndarray:
    """A per-vertex FreeSurfer curv-format file (``?h.thickness`` …)."""
    import nibabel as nib

    return np.asarray(nib.freesurfer.read_morph_data(str(path)))


# ── import manifest (see nvitk.gui.tools.external_io) ─────────────────────────


def subject_manifest(
    subject_dir: str | Path,
    *,
    keys: tuple[str, ...] | None = None,
    surfaces: bool = True,
    stats: bool = True,
    base: str | Path | None = None,
) -> dict[str, Any]:
    """What to open from a FastSurfer subject: its volumes (*keys*, else all on disk), surfaces
    (pial coloured by thickness), ``.stats`` tables and corpus-callosum shape — as an import
    manifest with paths relative to *base* (default: the subject folder). Built without Qt:
    the SGE worker sends it back with the files it lists."""
    from nvitk.gui.tools.external_io import manifest_layer

    outputs = SubjectOutputs(Path(subject_dir))
    base = Path(base) if base is not None else outputs.subject_dir
    luts: dict[str, Any] = {}
    layers = []
    shown_labels = False
    for volume, path in outputs.existing_volumes(keys):
        if volume.lut and volume.lut not in luts:
            luts[volume.lut] = lut_for(volume.lut)
        layers.append(manifest_layer(
            path, volume.title, base=base, labels=volume.labels,
            # One label map in front; the rest a click away in the layer list.
            visible=not volume.labels or not shown_labels,
            lut=luts.get(volume.lut) if volume.lut else None, schema=volume.schema,
        ))
        shown_labels = shown_labels or volume.labels
    surface_entries = []
    if surfaces:
        for path in outputs.surface_files():
            hemi, kind = path.name.split(".", 1)
            thickness = outputs.thickness(hemi) if kind == "pial" else None
            surface_entries.append({
                "file": _rel(path, base), "title": path.name, "visible": kind == "pial",
                "values": _rel(thickness, base) if thickness is not None else None,
            })
    tables = []
    files: list[str] = []
    if stats:
        for path in outputs.stats_files():
            table = stats_table(path)
            if table:
                tables.append({"title": path.name, "payload": table, "subtitle": path.name, "row_header": "Structure"})
                files.append(_rel(path, base))
        shape = cc_measures(outputs.subject_dir)
        if shape:
            tables.append({"title": "corpus callosum shape", "payload": shape,
                           "subtitle": "stats/callosum.CC.midslice.json", "row_header": "Measure"})
    return {"layers": layers, "surfaces": surface_entries, "tables": tables, "files": files}


def _rel(path: Path, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(path)


__all__ = [
    "MODULE_VOLUMES",
    "subject_manifest",
    "cc_measures",
    "SURFACES",
    "SubjectOutputs",
    "VOLUMES",
    "Volume",
    "lut_for",
    "read_annot_values",
    "read_lut",
    "read_morph",
    "read_stats",
    "read_surface",
    "stats_table",
]
