"""
Where the brain parcellations used for rendering come from.

Description
-----------
:mod:`nvitk.viz.brainshow` can paint region values onto any atlas it is handed, but two of the
atlases this project reports measurements in are not fetchable from anywhere:

``desikan``
    The Desikan–Killiany cortical parcellation. It is what FreeSurfer's ``aparc`` produces and what
    the ASL and T1 pipelines publish their ``ctx-lh-…`` / ``left_…`` region ids against — but
    ``nilearn.datasets`` ships Destrieux, not Desikan, so there is nothing to download.

``vascular``
    The arterial-territory / watershed atlas the ASL pipeline uses for its ``left_mca_8``-style
    parcels. A lab-specific file with no public distribution.

So the geometry has to be *located* rather than fetched. This module answers "where is it", in one
place, with the same precedence for both:

1. an explicit environment variable — the escape hatch for a one-off run or a cluster job;
2. the ``atlas`` block of ``.nvitk/settings.json``, found by the same search
   :func:`~nvitk.db.settings_paths.settings_json_path` already uses for the ``db`` block;
3. an ``ATLAS/`` directory beside the configured dataset root, since that is where these files
   actually live in the lab tree;
4. for Desikan only, a FreeSurfer installation's ``fsaverage`` surface labels.

Nothing here reads the atlas — that is :mod:`~nvitk.viz.brainshow`'s job. These functions return
paths (or ``None``) and never raise, so a caller can report *every* place it looked when nothing is
configured, which is the difference between an actionable error and "atlas not found".

Settings example
----------------
.. code-block:: json

    {
      "atlas": {
        "root":     "…/DB/ATLAS/",
        "desikan":  "…/ATLAS/aparc+aseg_MNI.nii.gz",
        "vascular": "…/ATLAS/arterial_atlas_watershed_tpm_correct_space.nii.gz"
      }
    }
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path
from typing import Any

from nvitk.core.logger import Logger

log = Logger()

#: Environment overrides, checked before anything in settings.
ENV_DESIKAN: str = "NVITK_DESIKAN_ATLAS"
ENV_VASCULAR: str = "NVITK_VASCULAR_ATLAS"
ENV_ATLAS_ROOT: str = "NVITK_ATLAS_ROOT"

#: Filenames probed inside an atlas directory, most specific first. Deliberately generous: these
#: files are hand-placed by whoever set the study up, and a spelling difference should not be the
#: reason a figure cannot be drawn.
_DESIKAN_FILENAMES: tuple[str, ...] = (
    "desikan.nii.gz", "desikan_killiany.nii.gz", "aparc.nii.gz", "aparc+aseg.nii.gz",
    "aparc_aseg.nii.gz", "aparc+aseg_MNI.nii.gz", "DK.nii.gz", "dkt.nii.gz",
)
_VASCULAR_FILENAMES: tuple[str, ...] = (
    "arterial_atlas_watershed_tpm_correct_space.nii.gz",
    "arterial_atlas_watershed.nii.gz", "vascular_atlas.nii.gz", "arterial_atlas.nii.gz",
)

#: Suffixes accepted as a parcellation. ``.annot`` is a FreeSurfer surface label file and resolves
#: to a vertex atlas; the rest are volumes.
ATLAS_SUFFIXES: tuple[str, ...] = (".nii", ".nii.gz", ".mgz", ".annot")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def load_atlas_settings_block() -> dict[str, Any]:
    """The ``atlas`` section of ``settings.json``, or ``{}``."""
    from nvitk.db.settings_paths import load_settings_document

    block = load_settings_document().get("atlas")
    return block if isinstance(block, dict) else {}


def _expand(raw: Any) -> Path | None:
    """A configured value as an expanded :class:`~pathlib.Path`, or ``None`` when it is blank."""
    text = str(raw or "").strip()
    return Path(os.path.expanduser(text)) if text else None


def atlas_root() -> Path | None:
    """
    Directory the parcellation files live in.

    ``NVITK_ATLAS_ROOT`` → ``atlas.root`` in settings → an ``ATLAS`` directory beside the configured
    dataset root, which is where the lab tree keeps them (``…/DB/ATLAS`` next to
    ``…/DB/nvitk-dataset``).
    """
    from nvitk.db.settings_paths import load_db_settings_block

    candidates: list[Path] = []
    env = _expand(os.environ.get(ENV_ATLAS_ROOT))
    if env is not None:
        candidates.append(env)
    configured = _expand(load_atlas_settings_block().get("root"))
    if configured is not None:
        candidates.append(configured)

    db_root = _expand(load_db_settings_block().get("root"))
    if db_root is not None:
        candidates += [db_root.parent / "ATLAS", db_root / "ATLAS"]

    for path in candidates:
        if path.is_dir():
            return path
    return None


def _resolve_atlas_file(
    *, env_var: str, settings_key: str, filenames: tuple[str, ...]
) -> Path | None:
    """Shared precedence for a named parcellation: env → settings → the atlas directory."""
    env = _expand(os.environ.get(env_var))
    if env is not None:
        if env.exists():
            return env
        log.warning("%s points at %s, which does not exist — ignoring it.", env_var, env)

    configured = _expand(load_atlas_settings_block().get(settings_key))
    if configured is not None:
        if configured.exists():
            return configured
        log.warning(
            "atlas.%s in .nvitk/settings.json points at %s, which does not exist — ignoring it.",
            settings_key, configured,
        )

    root = atlas_root()
    if root is None:
        return None
    for name in filenames:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def desikan_atlas_path() -> Path | None:
    """Configured Desikan–Killiany parcellation (a NIfTI volume or a FreeSurfer ``.annot``)."""
    return _resolve_atlas_file(
        env_var=ENV_DESIKAN, settings_key="desikan", filenames=_DESIKAN_FILENAMES
    )


def vascular_atlas_path() -> Path | None:
    """Configured arterial-territory / watershed parcellation (a NIfTI volume)."""
    return _resolve_atlas_file(
        env_var=ENV_VASCULAR, settings_key="vascular", filenames=_VASCULAR_FILENAMES
    )


# ---------------------------------------------------------------------------
# FreeSurfer fallback
# ---------------------------------------------------------------------------
def freesurfer_home() -> Path | None:
    """A usable FreeSurfer installation, from ``FREESURFER_HOME`` or the conventional locations."""
    env = _expand(os.environ.get("FREESURFER_HOME"))
    candidates = [env] if env is not None else []
    candidates += [Path("/usr/local/freesurfer"), Path("/opt/freesurfer")]
    for path in candidates:
        if path is not None and (path / "subjects").is_dir():
            return path
    return None


#: fsaverage variants to look for, coarsest first. ``fsaverage5`` (10k vertices per hemisphere) is
#: preferred because it is what :mod:`nvitk.viz.brainshow` meshes against by default and is an order
#: of magnitude lighter to render; the full ``fsaverage`` is the fallback and needs the matching
#: full-resolution mesh, which is why the subject name comes back with the paths.
_FSAVERAGE_SUBJECTS: tuple[str, ...] = ("fsaverage5", "fsaverage")


def fsaverage_aparc_annot() -> tuple[Path, Path, str] | None:
    """
    ``(lh, rh, subject)`` paths to an fsaverage ``?h.aparc.annot`` — the canonical Desikan labels.

    Searched under ``$SUBJECTS_DIR`` first, then a FreeSurfer install's own ``subjects`` directory,
    trying each entry of :data:`_FSAVERAGE_SUBJECTS` in turn.

    *subject* matters to the caller: an annot file has one label per **vertex**, so it is only valid
    against the mesh with that vertex count. Returning it lets the caller mesh against the same
    resolution instead of silently producing a texture the mesh cannot index.

    Returns ``None`` unless **both** hemispheres are present: half a parcellation would draw a brain
    with one blank side, which reads as a result rather than as a missing file.
    """
    roots: list[Path] = []
    subjects_dir = _expand(os.environ.get("SUBJECTS_DIR"))
    if subjects_dir is not None:
        roots.append(subjects_dir)
    home = freesurfer_home()
    if home is not None:
        roots.append(home / "subjects")

    for root in roots:
        for subject in _FSAVERAGE_SUBJECTS:
            left = root / subject / "label" / "lh.aparc.annot"
            right = root / subject / "label" / "rh.aparc.annot"
            if left.is_file() and right.is_file():
                return left, right, subject
    return None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def atlas_cache_dir() -> Path:
    """
    Directory for parcellations nvitk generates itself, created on demand.

    Under ``$XDG_CACHE_HOME`` (or ``~/.cache``), *not* in the repo: a generated atlas is a build
    product of this machine, and committing one would ship a file whose provenance the repo cannot
    describe.
    """
    base = _expand(os.environ.get("XDG_CACHE_HOME")) or (Path.home() / ".cache")
    path = base / "nvitk" / "atlases"
    path.mkdir(parents=True, exist_ok=True)
    return path


def describe_search(kind: str) -> str:
    """
    Every place *kind* (``"desikan"`` / ``"vascular"``) was looked for, for an error message.

    A resolver that fails should say where it looked. "No Desikan atlas configured" sends the reader
    to the documentation; this sends them to the line they need to add.
    """
    kind = str(kind).strip().lower()
    env_var = ENV_DESIKAN if kind == "desikan" else ENV_VASCULAR
    root = atlas_root()
    parts = [
        f"${env_var}",
        f'the "atlas.{kind}" key of .nvitk/settings.json',
        f"the atlas directory ({root})" if root else
        f'an ATLAS/ directory beside the dataset root (set "atlas.root" or ${ENV_ATLAS_ROOT})',
    ]
    if kind == "desikan":
        parts.append("$SUBJECTS_DIR / $FREESURFER_HOME fsaverage ?h.aparc.annot")
    return "; ".join(parts)


__all__ = [
    "ATLAS_SUFFIXES",
    "ENV_ATLAS_ROOT",
    "ENV_DESIKAN",
    "ENV_VASCULAR",
    "atlas_cache_dir",
    "atlas_root",
    "describe_search",
    "desikan_atlas_path",
    "freesurfer_home",
    "fsaverage_aparc_annot",
    "load_atlas_settings_block",
    "vascular_atlas_path",
]
