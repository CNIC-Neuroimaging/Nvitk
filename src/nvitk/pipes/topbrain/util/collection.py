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

import json
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

    pattern: str | tuple[str, ...] = "**/*.nii.gz"
    """Glob (or several), relative to :attr:`root`.

    A tuple is for cohorts whose wanted volumes live in sibling directories that no single glob
    separates from the unwanted ones -- ``bo_large_ia`` keeps its images in ``internal_Tr*`` and
    ``external_cta_img`` beside annotation directories that must not be swept in. Matches are
    unioned and de-duplicated, so overlapping globs cannot contribute a volume twice.
    """

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
    # BO-Large-IA: ~1400 CTA volumes. 
    # The release keeps its images in numbered batch directories beside annotation directories and
    # CSV manifests, so the globs name the image batches explicitly rather than sweeping the
    # tree -- an annotation volume pulled in as an image would be pre-trained on as anatomy.
    "bo_large_ia": {
        "pattern": (
            "internal_Tr*/*.nii.gz",     # internal_Tr01 .. internal_Tr12
            "internal_Ts*/*.nii.gz",     # internal_Ts01, internal_Ts02
            "external_cta_img/*.nii.gz",
        ),
        "modality": "ct",
        # The batch directory is not part of the subject: the numbering is global across them
        # (Tr01 holds Tr0001-Tr0099, Tr02 holds Tr0100-Tr0199), so the stem alone is unique, and
        # the internal and external cohorts use different prefixes. Deliberately permissive
        # about the stem so a batch whose naming differs is not silently dropped.
        "subject_regex": r"/(?P<subject>[^/]+)\.nii(?:\.gz)?$",
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
    if (
        name in BUILTIN_SOURCES
        and BUILTIN_SOURCES[name].get("modality") == "auto"
        and overrides.get("modality")
    ):
        # Overriding an auto-detecting cohort does not select from it, it relabels all of it:
        # the per-file modality group is only consulted when the source's own modality is
        # 'auto'. On a mixed cohort that sends MR volumes through CT harmonisation.
        log.warning(
            "Source %r detects modality per file, but modality=%r was forced: every volume "
            "will be treated as %s, including the ones that are not. To take one modality of "
            "a mixed cohort, drop the override and use the corpus modality instead.",
            name, overrides["modality"], str(overrides["modality"]).upper(),
        )
    return CorpusSource(name=name, root=Path(root), **settings)


def parse_source_spec(spec: str, *, challenge_root: Path | None = None) -> CorpusSource:
    """Parse one ``--corpus-source`` value into a :class:`CorpusSource`.

    Accepted forms::

        topbrain                        # built-in, rooted at the challenge release
        pesa_tof=/path/to/NIFTI         # built-in layout, explicit root
        bo_large_ia=/path/to/release    # built-in, multi-glob layout
        name:modality=/path[:glob]      # arbitrary cohort

    A ``:glob`` suffix overrides the built-in pattern with a single glob. Built-ins that need
    several (``bo_large_ia``) therefore cannot be narrowed that way -- take them whole, or
    declare an arbitrary cohort per directory.

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


def source_patterns(pattern: str | Sequence[str]) -> tuple[str, ...]:
    """Normalise :attr:`CorpusSource.pattern` to a tuple of globs."""
    return (pattern,) if isinstance(pattern, str) else tuple(pattern)


def _matching_paths(root: Path, pattern: str | Sequence[str]) -> list[Path]:
    """Every file under *root* matching any glob in *pattern*, de-duplicated and sorted.

    Sorting is over the union rather than per glob, so the corpus order does not depend on which
    glob happened to find a volume -- the collection is written in this order and a stable one
    keeps successive builds comparable.
    """
    seen: set[Path] = set()
    for glob in source_patterns(pattern):
        seen.update(root.glob(glob))
    return sorted(seen)


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
    for path in _matching_paths(root, source.pattern):
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


#: Per-file record of how each corpus volume was harmonised, beside the volume itself.
#:
#: Without it the only "has this been done?" test is whether the output exists, which is wrong in
#: both directions: it skips a volume whose window has since changed (silently keeping the old
#: harmonisation), and it redoes every volume when ``--overwrite`` is given for an unrelated
#: reason. With ~1400 CTAs that second case is the bulk of stage 0's runtime.
SIDECAR_SUFFIX: str = ".harmonised.json"


def _sidecar_payload(volume: CorpusVolume, ct_window: Sequence[float] | None,
                     mr_percentiles: Sequence[float] | None) -> dict[str, Any]:
    """Everything that decides the output bytes, so a match means the file is already right."""
    source = Path(volume.path)
    try:
        stat = source.stat()
        fingerprint: dict[str, Any] = {"size": stat.st_size, "mtime": int(stat.st_mtime)}
    except OSError:
        fingerprint = {}
    return {
        "source": str(source),
        "modality": volume.modality,
        "ct_window": list(ct_window) if ct_window else None,
        "mr_percentiles": list(mr_percentiles) if mr_percentiles else None,
        **fingerprint,
    }


def _is_current(destination: Path, expected: dict[str, Any]) -> bool:
    """Whether *destination* was produced from exactly these inputs and parameters."""
    sidecar = destination.with_suffix(destination.suffix + SIDECAR_SUFFIX)
    if not (destination.is_file() and sidecar.is_file()):
        return False
    try:
        return json.loads(sidecar.read_text(encoding="utf-8")) == expected
    except (OSError, ValueError):
        return False


def harmonize_volume(volume: CorpusVolume, corpus_root: Path, *, overwrite: bool = False,
                     ct_window: Sequence[float] | None = None,
                     mr_percentiles: Sequence[float] | None = None) -> Path:
    """Write an intensity-harmonised copy of *volume* under *corpus_root*; returns its path.

    Geometry is untouched — harmonisation is a voxelwise intensity map.

    The windows must be the ones stage 0 applies to the labelled data. Only the *clipping* they
    do survives nnU-Net's per-image z-score — an affine intensity map leaves a z-score unchanged
    — but clipping does survive, so a corpus windowed differently pre-trains the encoder on a
    saturation pattern the segmentation data never shows it. ``None`` keeps
    :func:`~nvitk.normalization.harmonize_modality`'s own defaults.
    """
    destination = Path(corpus_root) / volume.source / volume.subject_id / volume.name
    expected = _sidecar_payload(volume, ct_window, mr_percentiles)
    # Parameters decide, not merely the file's existence. A changed window redoes the volume even
    # without --overwrite (otherwise the corpus silently keeps the old harmonisation), and an
    # unchanged one is skipped even with it.
    if _is_current(destination, expected) and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    image = imread(volume.path)
    harmonised = harmonize_modality(
        image, volume.modality,
        **({"ct_window": ct_window} if ct_window else {}),
        **({"mr_percentiles": mr_percentiles} if mr_percentiles else {}),
    )
    imsave(destination, harmonised)
    destination.with_suffix(destination.suffix + SIDECAR_SUFFIX).write_text(
        json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    only_modality: str | None = None,
    ct_window: Sequence[float] | None = None,
    mr_percentiles: Sequence[float] | None = None,
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
    if only_modality:
        # Filtering belongs here, not in the source spec: writing ``topaneu:ct`` sets the
        # source's modality outright, which *relabels* every volume of a mixed cohort instead
        # of selecting from it -- and then harmonisation clips MR data with an HU window.
        kept = [v for v in volumes if v.modality == only_modality]
        dropped = len(volumes) - len(kept)
        if dropped:
            log.info("Corpus restricted to %s: kept %d volume(s), dropped %d.",
                     only_modality.upper(), len(kept), dropped)
        volumes = kept
    if not volumes:
        raise FileNotFoundError(
            "No volumes found across "
            f"{len(sources)} corpus source(s): {[s.name for s in sources]}."
        )

    if harmonize:
        log.info("Harmonising %d volume(s) -> %s", len(volumes), corpus_root)

        def _harmonize(volume: CorpusVolume) -> Path:
            """Harmonise one volume onto the shared intensity range."""
            return harmonize_volume(
                volume, corpus_root, overwrite=overwrite,
                ct_window=ct_window, mr_percentiles=mr_percentiles,
            )

        already = sum(
            1 for volume in volumes
            if _is_current(
                Path(corpus_root) / volume.source / volume.subject_id / volume.name,
                _sidecar_payload(volume, ct_window, mr_percentiles),
            )
        )
        if already and not overwrite:
            log.info("%d volume(s) already harmonised with these exact parameters; "
                     "re-doing the remaining %d.", already, len(volumes) - already)
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
