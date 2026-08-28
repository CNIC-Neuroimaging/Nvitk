"""Assemble an unlabeled angiographic corpus into an nnssl ``Collection``.

Description
-----------
nnssl pre-trains from a ``pretrain_data.json`` describing a
``Collection -> Dataset -> Subject -> Session -> Image`` hierarchy. There is no generic
folder-scanner in the upstream repo — its converters all hard-code one cohort's layout — so
this module provides a declarative one: a :class:`CorpusSource` names a root, a glob and a
regex that recovers the subject id from each path.

Why volumes are rewritten rather than referenced
------------------------------------------------
nnssl's experiment planner emits a single normalisation scheme for the whole collection
(``ZScoreNormalization``) and its fingerprint records **only spacings** — it never collects
intensity statistics, so it has no CT normalisation and no way to acquire one. Feeding raw
Hounsfield units and raw TOF arbitrary units into one per-volume z-score therefore trains the
encoder across two incompatible intensity scales. Worse, nnssl crops to the non-zero bounding
box before normalising, which is meaningless on CT where air is −1000 rather than 0.

Harmonising every volume onto a common ``[0, 1]`` range on the way in fixes both, and keeps the
corpus consistent with what stage 0 feeds the segmentation model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from nvitk.core.backend import map_in_thread_pool
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.normalization import harmonize_modality

log = Logger()

#: Extensions nnssl's readers accept. The collection must be format-homogeneous: nnssl picks
#: one reader from the *first* image path and applies it to every volume.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii")


@dataclass(frozen=True)
class CorpusSource:
    """One cohort contributing unlabeled volumes to the pre-training corpus.

    Becomes a ``Dataset`` inside the nnssl ``Collection``, keyed by :attr:`name`.
    """

    name: str
    """Dataset index within the collection, e.g. ``topbrain`` or ``pesa_tof``."""

    root: Path
    """Directory searched with :attr:`pattern`."""

    modality: str
    """``ct``/``mr``, or ``auto`` to read it from the :attr:`subject_regex` ``modality`` group."""

    pattern: str = "**/*.nii.gz"
    """Glob, relative to :attr:`root`."""

    subject_regex: str | None = None
    """Regex over the root-relative path with a ``subject`` group (and optionally ``modality``).

    ``None`` falls back to the filename stem, which is right for a flat directory of volumes.
    """

    subject_template: str | None = None
    """Format string over the :attr:`subject_regex` groups, e.g. ``"{center}_{pid}"``.

    Needed when the subject key is not one contiguous span of the filename — TopAneu writes
    ``topaneu_<center>_<modality>_<patient>``, so the modality sits between the two halves of
    the subject and no single regex group can capture it. ``None`` uses the ``subject`` group.
    """

    session: str = "ses-1"
    """Session id. These cohorts are single-session; nnssl requires the level to exist."""


@dataclass(frozen=True)
class CorpusVolume:
    """One resolved volume of a :class:`CorpusSource`."""

    source: str
    subject_id: str
    session_id: str
    modality: str
    path: Path
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Image name inside the collection — unique within its subject/session."""
        return self.path.name


#: Ready-made sources for the cohorts this pipeline knows about.
BUILTIN_SOURCES: dict[str, dict[str, Any]] = {
    # The challenge's own 50 volumes: in-domain, but far too few on their own.
    "topbrain": {
        "pattern": "imagesTr_topbrain/topcow_*_0000.nii.gz",
        "modality": "auto",
        "subject_regex": r"topcow_(?P<modality>ct|mr)_(?P<subject>\d+)_0000\.nii\.gz$",
    },
    # TopAneu release: the only source here that brings CTA in quantity. Pre-training on a
    # TOF-only corpus biases the encoder toward MR, and half the benchmark is CT.
    "topaneu": {
        "pattern": "images/topaneu_*_0000.nii.gz",
        "modality": "auto",
        "subject_regex": (
            r"topaneu_(?P<center>center\d+)_(?P<modality>ct|mr)_(?P<pid>\d+)"
            r"(?:_(?P<repeat>\d+))?_0000\.nii\.gz$"
        ),
        # Both modalities of one patient — and both scans of a longitudinal pair — share a
        # subject, so the repeat index is deliberately left out of the key.
        "subject_template": "{center}_{pid}",
    },
    # PESA-Brain TOF-MRA: same modality family as the MRA track.
    "pesa_tof": {
        "pattern": "*/TOF/*.nii.gz",
        "modality": "mr",
        "subject_regex": r"^(?P<subject>[^/]+)/TOF/",
    },
}


def make_source(name: str, root: Path, **overrides: Any) -> CorpusSource:
    """Build a :class:`CorpusSource`, starting from :data:`BUILTIN_SOURCES` when *name* is known."""
    settings: dict[str, Any] = dict(BUILTIN_SOURCES.get(name, {}))
    settings.update({k: v for k, v in overrides.items() if v is not None})
    settings.setdefault("modality", "mr")
    return CorpusSource(name=name, root=Path(root), **settings)


def parse_source_spec(spec: str, *, challenge_root: Path | None = None) -> CorpusSource:
    """Parse one ``--corpus-source`` value into a :class:`CorpusSource`.

    Accepted forms::

        topbrain                        # built-in, rooted at the challenge release
        pesa_tof=/path/to/NIFTI         # built-in layout, explicit root
        name:modality=/path[:glob]      # arbitrary cohort

    Raises
    ------
    ValueError
        On a malformed spec, a built-in name given without the root it needs, or a non-built-in
        name without an explicit modality — guessing the modality would apply an HU window to
        arbitrary-unit MR data, which looks plausible and is completely wrong.
    """
    text = spec.strip()
    if not text:
        raise ValueError("Corpus source cannot be empty.")

    head, _, root_part = text.partition("=")
    name, _, modality = head.partition(":")
    name = name.strip()
    root_text, _, glob = root_part.partition(":")
    root_text = root_text.strip()

    if name == "topbrain" and not root_text:
        if challenge_root is None:
            raise ValueError("Source 'topbrain' needs a challenge root.")
        root = Path(challenge_root)
    elif root_text:
        root = Path(root_text).expanduser()
    else:
        raise ValueError(f"Source {name!r} needs a root: use '{name}=/path/to/data'.")

    if name not in BUILTIN_SOURCES and not modality:
        raise ValueError(
            f"Source {name!r} is not built-in ({', '.join(BUILTIN_SOURCES)}), so it needs an "
            f"explicit modality: '{name}:mr=/path'."
        )
    return make_source(
        name, root, modality=modality.strip() or None, pattern=glob.strip() or None
    )


def iter_source_volumes(source: CorpusSource) -> Iterator[CorpusVolume]:
    """Yield the volumes of *source* in sorted order.

    Raises
    ------
    FileNotFoundError
        If the root does not exist. A source that silently contributes nothing would shrink
        the corpus without anyone noticing until the encoder underperforms.
    """
    root = Path(source.root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus source {source.name!r} root does not exist: {root}")

    regex = re.compile(source.subject_regex) if source.subject_regex else None
    found = 0
    for path in sorted(root.glob(source.pattern)):
        if not path.name.endswith(SUPPORTED_SUFFIXES):
            continue
        relative = path.relative_to(root).as_posix()

        if regex is None:
            subject = path.name
            for suffix in SUPPORTED_SUFFIXES:
                if subject.endswith(suffix):
                    subject = subject[: -len(suffix)]
                    break
            modality = source.modality
        else:
            match = regex.search(relative)
            if match is None:
                log.debug("[%s] %s does not match subject_regex; skipping.", source.name, relative)
                continue
            groups = match.groupdict()
            subject = (
                source.subject_template.format(**groups)
                if source.subject_template else match.group("subject")
            )
            modality = (
                groups.get("modality") or source.modality
                if source.modality == "auto"
                else source.modality
            )

        if modality == "auto":
            raise ValueError(
                f"Source {source.name!r} has modality 'auto' but {relative} yielded no "
                f"'modality' regex group; set an explicit modality."
            )

        found += 1
        yield CorpusVolume(
            source=source.name,
            subject_id=f"{source.name}-{subject}",
            session_id=source.session,
            modality=modality,
            path=path,
            info={"source_root": str(root), "relative_path": relative},
        )

    if found == 0:
        log.warning(
            "Corpus source %r matched no volumes under %s with pattern %r.",
            source.name,
            root,
            source.pattern,
        )


def harmonize_volume(volume: CorpusVolume, corpus_root: Path, *, overwrite: bool = False) -> Path:
    """Write an intensity-harmonised copy of *volume* under *corpus_root*; returns its path.

    Geometry is untouched — harmonisation is a voxelwise intensity map.
    """
    destination = Path(corpus_root) / volume.source / volume.subject_id / volume.name
    if destination.is_file() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    image = imread(volume.path)
    harmonised = harmonize_modality(image, volume.modality)
    imsave(destination, harmonised)
    return destination


def build_collection(
    sources: Sequence[CorpusSource],
    *,
    corpus_root: Path,
    collection_index: int,
    collection_name: str,
    harmonize: bool = True,
    overwrite: bool = False,
    workers: int = 1,
) -> tuple[Any, list[CorpusVolume]]:
    """Build an nnssl ``Collection`` from *sources*.

    Requires the vendored nnssl clone to be importable — call
    :func:`~nvitk.pipes.topbrain.util.nnssl_env.apply_nnssl_env` first.

    Returns
    -------
    tuple
        ``(Collection, volumes)``. The volume list is kept for the provenance sidecar, since
        the collection itself discards the source paths once harmonised.
    """
    from nnssl.data.raw_dataset import Collection, Dataset, Image, Session, Subject

    volumes = [volume for source in sources for volume in iter_source_volumes(source)]
    if not volumes:
        raise FileNotFoundError(
            "No volumes found across "
            f"{len(sources)} corpus source(s): {[s.name for s in sources]}."
        )

    if harmonize:
        log.info("Harmonising %d volume(s) -> %s", len(volumes), corpus_root)

        def _harmonize(volume: CorpusVolume) -> Path:
            """Harmonise one volume onto the shared intensity range."""
            return harmonize_volume(volume, corpus_root, overwrite=overwrite)

        written = map_in_thread_pool(_harmonize, volumes, max_workers=int(workers))
    else:
        written = [volume.path for volume in volumes]

    datasets: dict[str, Any] = {}
    for volume, path in zip(volumes, written):
        dataset = datasets.setdefault(
            volume.source,
            Dataset(dataset_index=volume.source, name=volume.source, subjects={}),
        )
        subject = dataset.subjects.setdefault(
            volume.subject_id, Subject(subject_id=volume.subject_id, sessions={})
        )
        session = subject.sessions.setdefault(
            volume.session_id, Session(session_id=volume.session_id, images=[])
        )
        session.images.append(
            Image(
                name=volume.name,
                image_path=str(path),
                modality=volume.modality,
                image_info={"source": volume.source, **volume.info},
            )
        )

    collection = Collection(
        collection_index=collection_index,
        collection_name=collection_name,
        datasets=datasets,
    )
    by_source = {name: sum(len(s.images) for sub in d.subjects.values() for s in sub.sessions.values())
                 for name, d in datasets.items()}
    log.info("Collection %s: %d volume(s) %s", collection_name, len(volumes), by_source)
    return collection, volumes


__all__ = [
    "BUILTIN_SOURCES",
    "SUPPORTED_SUFFIXES",
    "CorpusSource",
    "CorpusVolume",
    "build_collection",
    "harmonize_volume",
    "iter_source_volumes",
    "make_source",
    "parse_source_spec",
]
