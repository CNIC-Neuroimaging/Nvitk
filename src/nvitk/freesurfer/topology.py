"""FreeSurfer's topology tools on any binary mask: tessellation, Euler number, the spherical
maps and the topology fixers (``mris_fix_topology`` / ``mris_topo_fixer``).

The wrappers return :class:`~nvitk.freesurfer.run.FsCommand` objects and run nothing. The
helpers below move data in and out:

- :func:`write_volume` writes a mask or an image as MGZ, keeping its affine;
- :func:`prepare_subject` builds the minimal subject tree the fixers read, from a mask (and the
  TOF / T1 it came from);
- :func:`read_surface` / :func:`write_surface` read and write FreeSurfer geometry in scanner-RAS
  millimetres;
- :func:`surface_euler` counts V − E + F in Python, so any mesh can be checked without
  FreeSurfer.

The fixers were built for the cortex: they map a closed surface to the sphere, find the defects
as overlaps and re-tessellate each one. They choose between cutting a handle and filling a hole
with a likelihood built from white-matter / grey-matter intensities. Here
:func:`prepare_subject` plays that part with the mask (``wm.mgz`` = mask × 110) and the image
scaled so the mask's median is 110 (``brain.mgz`` / ``norm.mgz``). They are offered to test and
compare, for instance against :mod:`nvitk.morphology.siphon_cut`.

FreeSurfer's tessellation order (``recon-all``, FastSurfer's ``recon-surf``):
``mri_pretess`` → ``mri_mc`` → ``mris_extract_main_component`` → ``mris_smooth`` →
``mris_inflate`` → ``mris_sphere -q`` (or FastSurfer's spectral projection) →
``mris_fix_topology`` (``mris_topo_fixer`` when the first leaves defects) →
``mris_remove_intersection``.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .run import FsCommand, chain

#: ``filled.mgz`` value of each hemisphere (FreeSurfer's convention).
HEMI_VALUE = {"lh": 255, "rh": 127}
#: White-matter intensity FreeSurfer normalises to.
WM_LEVEL = 110
FIXERS = ("mris_fix_topology", "mris_topo_fixer")
SPHERES = ("qsphere", "spectral")
SEED = 1234


def _q(path: Any) -> str:
    return shlex.quote(str(path))


# ── data in and out ──────────────────────────────────────────────────────────


def write_volume(data: Any, affine: Any, path: str | Path, *, dtype: Any = np.uint8) -> Path:
    """*data* as MGZ (or NIfTI, by suffix) with *affine*."""
    import nibabel as nib

    from nvitk.core.array import to_numpy

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(to_numpy(data)).astype(dtype)
    aff = np.asarray(affine, dtype=float)
    img = nib.MGHImage(arr, aff) if out.suffix in (".mgz", ".mgh") else nib.Nifti1Image(arr, aff)
    nib.save(img, str(out))
    return out


def scaled_intensity(intensity: Any | None, mask: np.ndarray) -> np.ndarray:
    """*intensity* scaled so the mask's median is :data:`WM_LEVEL`, as uint8 (``mask × 110``
    without one)."""
    from nvitk.core.array import to_numpy

    if intensity is None:
        return (mask.astype(np.uint8) * WM_LEVEL).astype(np.uint8)
    arr = np.asarray(to_numpy(intensity), dtype=np.float64)
    med = float(np.median(arr[mask])) if mask.any() else float(np.median(arr))
    scale = WM_LEVEL / med if med > 0 else 1.0
    return np.clip(arr * scale, 0, 255).astype(np.uint8)


def prepare_subject(
    mask: Any,
    affine: Any,
    subjects_dir: str | Path,
    subject: str,
    *,
    intensity: Any | None = None,
    hemi: str = "lh",
) -> Path:
    """The subject tree FreeSurfer's topology programs read, from a binary *mask*.

    ``mri/filled.mgz`` (mask × 255 for ``lh``, 127 for ``rh``), ``mri/wm.mgz`` (mask × 110),
    ``mri/brain.mgz`` = ``norm.mgz`` = ``orig.mgz`` (:func:`scaled_intensity`) and empty ``surf``,
    ``label``, ``scripts``, ``stats``, ``tmp`` folders. Returns the subject folder."""
    from nvitk.core.array import to_numpy

    m = np.asarray(to_numpy(mask)).astype(bool)
    sd = Path(subjects_dir) / subject
    for sub in ("mri", "surf", "label", "scripts", "stats", "tmp", "touch"):
        (sd / sub).mkdir(parents=True, exist_ok=True)
    write_volume(m.astype(np.uint8) * HEMI_VALUE[hemi], affine, sd / "mri" / "filled.mgz")
    write_volume(m.astype(np.uint8) * WM_LEVEL, affine, sd / "mri" / "wm.mgz")
    brain = scaled_intensity(intensity, m)
    for name in ("brain.mgz", "norm.mgz", "orig.mgz"):
        write_volume(brain, affine, sd / "mri" / name)
    return sd


def read_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)``, vertices in scanner-RAS millimetres (tkregister + ``cras``)."""
    from nvitk.segmentation.fastsurfer.outputs import read_surface as _read

    return _read(path)


def outward_faces(vertices: Any, faces: Any) -> np.ndarray:
    """*faces* wound so their normals point outwards (FreeSurfer's convention): flipped when the
    signed volume they enclose is negative, as scikit-image's marching cubes leaves them."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    signed = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6.0
    return f[:, ::-1].copy() if signed < 0 else f


def write_surface(path: str | Path, vertices: Any, faces: Any, *, orient: bool = True) -> Path:
    """A FreeSurfer surface with *vertices* (scanner-RAS mm) stored as they are: no volume
    geometry, so tkregister space is scanner space and :func:`read_surface` returns them back.
    With *orient*, the faces are wound outwards first (:func:`outward_faces`)."""
    import nibabel as nib

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = outward_faces(vertices, faces) if orient else np.asarray(faces)
    nib.freesurfer.write_geometry(str(out), np.asarray(vertices, dtype=np.float64), np.asarray(f, dtype=np.int32))
    return out


def surface_euler(vertices: Any, faces: Any) -> dict[str, Any]:
    """Euler characteristic of a triangle mesh, per connected piece: ``χ = V − E + F`` and, for a
    closed piece, its genus ``(2 − χ) / 2`` (handles)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = np.asarray(faces, dtype=np.int64)
    nv = int(len(vertices))
    used = np.unique(f)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(e, axis=0, return_counts=True)
    adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(nv, nv))
    n_cc, lab = connected_components(adj, directed=False)
    pieces = []
    for c in np.unique(lab[used]):
        vmask = lab == c
        V = int(np.isin(np.flatnonzero(vmask), used).sum())
        E = int(vmask[edges[:, 0]].sum())
        F = int(vmask[f[:, 0]].sum())
        boundary = int((counts[vmask[edges[:, 0]]] == 1).sum())
        chi = V - E + F
        pieces.append({"V": V, "E": E, "F": F, "euler": chi, "closed": boundary == 0,
                       "genus": (2 - chi) // 2 if boundary == 0 else None})
    pieces.sort(key=lambda p: -p["V"])
    total = sum(p["euler"] for p in pieces)
    return {"euler": total, "pieces": len(pieces), "genus_main": pieces[0]["genus"] if pieces else None,
            "per_piece": pieces}


# ── single programs ──────────────────────────────────────────────────────────


def pretess(volume: str | Path, label: int, intensity: str | Path, out: str | Path) -> FsCommand:
    """``mri_pretess``: change the voxels of *label* so its edge / corner-only contacts become
    face contacts (FreeSurfer fills; :func:`nvitk.morphology.siphon_cut.pretess` removes)."""
    return FsCommand(["mri_pretess", str(volume), str(int(label)), str(intensity), str(out)],
                     "mri_pretess", outputs={"volume": Path(out)}, label_outputs=("volume",))


def marching_cubes(volume: str | Path, label: int, out: str | Path, *, connectivity: int = 1) -> FsCommand:
    """``mri_mc``: the surface of *label* (connectivity 1 = 6+, 2 = 18, 3 = 6, 4 = 26)."""
    return FsCommand(["mri_mc", str(volume), str(int(label)), str(out), str(int(connectivity))], "mri_mc",
                     outputs={"surface": Path(out)}, surface_outputs=("surface",))


def main_component(surface: str | Path, out: str | Path) -> FsCommand:
    return FsCommand(["mris_extract_main_component", str(surface), str(out)], "mris_extract_main_component",
                     outputs={"surface": Path(out)}, surface_outputs=("surface",))


def euler_number(surface: str | Path) -> FsCommand:
    """``mris_euler_number`` (read the result with :func:`parse_euler`)."""
    return FsCommand(["mris_euler_number", str(surface)], "mris_euler_number")


_EULER = re.compile(r"=\s*(-?\d+)\s*-->\s*(\d+)\s*holes")
_DEFECT = re.compile(r"total defect index\s*=\s*(-?\d+)")


def parse_euler(stdout: str) -> dict[str, Any]:
    """``{"euler", "holes", "defect_index"}`` from ``mris_euler_number``'s output."""
    out: dict[str, Any] = {}
    m = _EULER.search(stdout or "")
    if m:
        out["euler"], out["holes"] = int(m.group(1)), int(m.group(2))
    d = _DEFECT.search(stdout or "")
    if d:
        out["defect_index"] = int(d.group(1))
    return out


def smooth(surface: str | Path, out: str | Path, *, iterations: int = 10, seed: int = SEED) -> FsCommand:
    return FsCommand(["mris_smooth", "-n", str(int(iterations)), "-nw", "-seed", str(int(seed)), str(surface),
                      str(out)], "mris_smooth", outputs={"surface": Path(out)}, surface_outputs=("surface",))


def inflate(surface: str | Path, out: str | Path) -> FsCommand:
    return FsCommand(["mris_inflate", "-no-save-sulc", str(surface), str(out)], "mris_inflate",
                     outputs={"surface": Path(out)}, surface_outputs=("surface",))


def quick_sphere(surface: str | Path, out: str | Path, *, seed: int = SEED) -> FsCommand:
    """``mris_sphere -q``: the quick spherical map the fixers start from (input: inflated)."""
    return FsCommand(["mris_sphere", "-q", "-seed", str(int(seed)), str(surface), str(out)], "mris_sphere -q",
                     outputs={"surface": Path(out)}, surface_outputs=("surface",), long=True)


def remove_intersection(surface: str | Path, out: str | Path) -> FsCommand:
    return FsCommand(["mris_remove_intersection", str(surface), str(out)], "mris_remove_intersection",
                     outputs={"surface": Path(out)}, surface_outputs=("surface",))


def spectral_project(surface: str | Path, out: str | Path, *, fastsurfer_home: str | Path | None = None) -> Path:
    """FastSurfer's spectral spherical projection of *surface* into *out*, for any shape.

    FastSurfer's ``spherically_project_surface`` expects a hemisphere: the first three
    Laplace–Beltrami eigenfunctions running anterior–posterior (y), superior–inferior (z) and
    left–right (x), and it refuses other layouts. The surface is therefore first turned so that
    its principal axes, longest first, lie along y, z and x, then projected; the sphere is turned
    back. The sphere is a parametrisation, so its orientation is free. Runs in this Python (needs
    ``lapy``); no FreeSurfer."""
    import os
    import tempfile

    import nibabel as nib

    from nvitk.segmentation.fastsurfer.config import fastsurfer_home as _home

    home = Path(fastsurfer_home) if fastsurfer_home else _home()
    if home is None:
        from .env import FreeSurferNotFound

        raise FreeSurferNotFound("FastSurfer is not installed (spectral sphere).")
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    os.environ.setdefault("USERNAME", "nvitk")
    from recon_surf.spherically_project import spherically_project_surface

    v, f = nib.freesurfer.read_geometry(str(surface))[:2]
    centre = v.mean(0)
    _u, _s, axes = np.linalg.svd(v - centre, full_matrices=False)  # rows: principal axes, longest first
    rot = np.stack([axes[2], axes[0], axes[1]])  # new x = 3rd axis, y = 1st, z = 2nd
    if np.linalg.det(rot) < 0:
        rot[0] *= -1.0  # a proper rotation keeps the faces' orientation
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "lh.pca"
        dst = Path(tmp) / "lh.pca.sphere"
        nib.freesurfer.write_geometry(str(src), (v - centre) @ rot.T, f)
        spherically_project_surface(str(src), str(dst), use_cholmod=False)
        vs, fs = nib.freesurfer.read_geometry(str(dst))[:2]
    out = Path(out)
    nib.freesurfer.write_geometry(str(out), vs @ rot, fs)
    return out


def spectral_sphere(surface: str | Path, out: str | Path, *, python: str | None = None,
                    fastsurfer_home: str | Path | None = None) -> FsCommand:
    """FastSurfer's spectral spherical projection (:func:`spectral_project`, needs ``lapy``): the
    first three non-constant Laplace–Beltrami eigenfunctions as a sphere, which FastSurfer uses
    instead of ``mris_sphere -q``. Runs in Python, no FreeSurfer license needed. It fails on a
    surface that is not genus 0 (FastSurfer's area checks); :func:`fix_topology` then falls back
    to ``mris_sphere -q`` as FastSurfer does."""
    home = f", fastsurfer_home={str(fastsurfer_home)!r}" if fastsurfer_home else ""
    code = ("from nvitk.freesurfer.topology import spectral_project; "
            f"spectral_project({str(surface)!r}, {str(out)!r}{home})")
    return FsCommand([python or sys.executable, "-c", code], "FastSurfer spectral sphere",
                     outputs={"surface": Path(out)}, surface_outputs=("surface",), requires=())


def rewrite_oriented(surface: str | Path, *, python: str | None = None,
                     fastsurfer_home: str | Path | None = None) -> FsCommand:
    """FastSurfer's ``rewrite_oriented_surface.py``: re-orients the fixed surface's triangles."""
    from nvitk.segmentation.fastsurfer.config import fastsurfer_home as _home

    home = Path(fastsurfer_home) if fastsurfer_home else _home()
    script = (home or Path(".")) / "recon_surf" / "rewrite_oriented_surface.py"
    return FsCommand([python or sys.executable, "-s", str(script), "--file", str(surface), "--backup",
                      f"{surface}.noorient"], "FastSurfer rewrite_oriented_surface", requires=())


# ── the topology fix, as recon-all runs it ───────────────────────────────────


def fix_topology(
    subjects_dir: str | Path,
    subject: str,
    *,
    hemi: str = "lh",
    fixer: str = "mris_fix_topology",
    sphere: str = "qsphere",
    ga: bool = True,
    seed: int = SEED,
    defect_seg: bool = True,
    rewrite_orientation: bool = False,
    python: str | None = None,
    license_file: str | Path | None = None,
) -> FsCommand:
    """The whole tessellation and topology fix on a :func:`prepare_subject` tree, in
    ``recon-all`` order (module docstring), as one background command.

    *fixer* picks ``mris_fix_topology`` (``-mgz -sphere qsphere.nofix [-ga]``, writes
    ``surf/<hemi>.orig``) or ``mris_topo_fixer`` (``-mgz -warnings``, writes
    ``surf/<hemi>.orig_corrected``). *sphere* is ``qsphere`` (``mris_sphere -q``) or ``spectral``
    (FastSurfer's projection, needs FastSurfer and lapy). With *defect_seg*, ``defect-seg`` maps the
    defects back into the volume (``mri/surface.defects.mgz``; needs tcsh).
    *rewrite_orientation* runs FastSurfer's orientation rewrite on the result, as recon-surf does.
    """
    if fixer not in FIXERS:
        raise ValueError(f"fixer must be one of {FIXERS}, got {fixer!r}")
    if sphere not in SPHERES:
        raise ValueError(f"sphere must be one of {SPHERES}, got {sphere!r}")
    sd = Path(subjects_dir) / subject
    mri, surf = sd / "mri", sd / "surf"
    value = HEMI_VALUE[hemi]
    nofix = surf / f"{hemi}.orig.nofix"
    steps: list[FsCommand] = [
        pretess(mri / "filled.mgz", value, mri / "brain.mgz", mri / f"filled-pretess{value}.mgz"),
        marching_cubes(mri / f"filled-pretess{value}.mgz", value, nofix),
        main_component(nofix, nofix),
        smooth(nofix, surf / f"{hemi}.smoothwm.nofix"),
        inflate(surf / f"{hemi}.smoothwm.nofix", surf / f"{hemi}.inflated.nofix"),
    ]
    qsphere = quick_sphere(surf / f"{hemi}.inflated.nofix", surf / f"{hemi}.qsphere.nofix", seed=seed)
    if sphere == "spectral":
        # As FastSurfer does: the spectral map fails on surfaces far from a sphere (its area
        # check), and mris_sphere -q takes over.
        spectral = spectral_sphere(surf / f"{hemi}.smoothwm.nofix", surf / f"{hemi}.qsphere.nofix", python=python)
        steps.append(FsCommand(["bash", "-c", f"{spectral.command_line()} || {qsphere.command_line()}"],
                               "spectral sphere (mris_sphere -q fallback)", requires=("mris_sphere",)))
    else:
        steps.append(qsphere)
    orig = surf / f"{hemi}.orig"
    copy = [f"cp {_q(nofix)} {_q(orig)}", f"cp {_q(surf / f'{hemi}.inflated.nofix')} {_q(surf / f'{hemi}.inflated')}"]
    if fixer == "mris_fix_topology":
        fix = ["mris_fix_topology", "-mgz", "-sphere", "qsphere.nofix"] + (["-ga"] if ga else [])
        fix += ["-seed", str(int(seed)), subject, hemi]
        fixed = orig
    else:
        copy += [f"cp {_q(surf / f'{hemi}.qsphere.nofix')} {_q(surf / f'{hemi}.qsphere')}",
                 f"cp {_q(surf / f'{hemi}.smoothwm.nofix')} {_q(surf / f'{hemi}.smoothwm')}"]
        fix = ["mris_topo_fixer", "-mgz", "-warnings", "-seed", str(int(seed)), subject, hemi]
        fixed = surf / f"{hemi}.orig_corrected"
    steps.append(FsCommand(["bash", "-c", " && ".join(copy)], "copy", requires=()))
    steps.append(FsCommand(fix, fixer, outputs={"fixed": fixed}, surface_outputs=("fixed",), cwd=sd / "scripts"))
    steps.append(remove_intersection(fixed, fixed))
    steps[-1].outputs = {"fixed": fixed}
    steps[-1].surface_outputs = ("fixed",)
    steps.append(FsCommand(["bash", "-c", f"mris_euler_number {_q(nofix)} > {_q(sd / 'scripts' / 'euler_nofix.txt')} 2>&1; "
                            f"mris_euler_number {_q(fixed)} > {_q(sd / 'scripts' / 'euler_fixed.txt')} 2>&1; true"],
                           "mris_euler_number", requires=("mris_euler_number",)))
    if rewrite_orientation:
        steps.append(rewrite_oriented(fixed, python=python))
    if defect_seg:
        steps.append(FsCommand(["defect-seg", "--s", subject, f"--{hemi}-only"], "defect-seg",
                               outputs={"defects": mri / "surface.defects.mgz"}, label_outputs=("defects",),
                               cwd=sd))
    for s in steps:
        s.subjects_dir = Path(subjects_dir)
        s.license_file = Path(license_file) if license_file else None
    cmd = chain(steps, title=f"FreeSurfer topology fix ({fixer}, {sphere})", cwd=sd / "scripts")
    cmd.outputs = {"pretess": mri / f"filled-pretess{value}.mgz", "nofix": nofix, "fixed": fixed}
    if defect_seg:
        cmd.outputs["defects"] = mri / "surface.defects.mgz"
    cmd.surface_outputs = ("nofix", "fixed")
    cmd.label_outputs = ("pretess", "defects")
    return cmd


def topology_report(subjects_dir: str | Path, subject: str, *, hemi: str = "lh",
                    fixer: str = "mris_fix_topology") -> dict[str, Any]:
    """Euler numbers before / after a :func:`fix_topology` run (FreeSurfer's, and
    :func:`surface_euler` on the surfaces) and the defects found."""
    sd = Path(subjects_dir) / subject
    out: dict[str, Any] = {}
    for key, name in (("before", "euler_nofix.txt"), ("after", "euler_fixed.txt")):
        p = sd / "scripts" / name
        if p.is_file():
            out[f"freesurfer_{key}"] = parse_euler(p.read_text(errors="replace"))
    surf = sd / "surf"
    fixed = surf / (f"{hemi}.orig" if fixer == "mris_fix_topology" else f"{hemi}.orig_corrected")
    for key, path in (("before", surf / f"{hemi}.orig.nofix"), ("after", fixed)):
        if path.is_file():
            v, f = read_surface(path)
            e = surface_euler(v, f)
            out[f"nvitk_{key}"] = {k: e[k] for k in ("euler", "pieces", "genus_main")}
    labels = surf / f"{hemi}.defect_labels"
    if labels.is_file():
        import nibabel as nib

        vals = np.asarray(nib.freesurfer.read_morph_data(str(labels)))
        ids = np.unique(vals[vals >= 0])
        out["defects"] = {"n": int(len(ids)), "vertices": int((vals >= 0).sum())}
    return out


__all__ = [
    "FIXERS",
    "HEMI_VALUE",
    "SPHERES",
    "WM_LEVEL",
    "euler_number",
    "fix_topology",
    "inflate",
    "main_component",
    "marching_cubes",
    "outward_faces",
    "parse_euler",
    "prepare_subject",
    "pretess",
    "quick_sphere",
    "read_surface",
    "remove_intersection",
    "rewrite_oriented",
    "scaled_intensity",
    "smooth",
    "spectral_project",
    "spectral_sphere",
    "surface_euler",
    "topology_report",
    "write_surface",
    "write_volume",
]
