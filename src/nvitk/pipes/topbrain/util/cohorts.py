"""
Built-in layouts for additional annotated cohorts merged into the training dataset.

Description
-----------
``--extra-train`` takes ``name=/images:/labels:modality``: two directories and one modality for
the whole cohort. That is enough for a folder somebody assembled by hand, and wrong for a
published release, which typically ships one image directory holding **both** modalities with
the modality encoded in the filename, and its masks under a differently-named sibling.

A :class:`CohortLayout` describes such a release once — where the images and masks live
relative to the release root, how to read the modality and the subject out of a filename, and
whether the masks are human annotation. Callers then name the release and its root::

    --extra-train-only topaneu=/path/to/topaneu_release

instead of spelling out subdirectories and being unable to express "both modalities".

Why the modality has to come per case
-------------------------------------
Stage 0 picks its intensity-harmonisation branch from the modality: a Hounsfield window for CTA,
robust percentiles for TOF. Applying the CT branch to arbitrary-unit MR produces a plausible
looking volume that is completely wrong, so the modality is never guessed — a layout must be
able to state it for every case, or the cohort needs an explicit one.

Pseudo-labelled releases
------------------------
:attr:`CohortLayout.pseudo_labels` marks a cohort whose masks are **model output** rather than
annotation. Those cases may be trained on but must never be validated against: a fold that
scored them would be measuring agreement with whoever produced them, not accuracy. The flag is
enforced in :func:`~nvitk.pipes.topbrain.stage0_dataprep.parse_extra_train`, which refuses the
cohort unless it was passed as ``--extra-train-only``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from nvitk.core.logger import Logger

log = Logger()


@dataclass(frozen=True)
class CohortCase:
    """One image/mask pair discovered in a cohort, with its modality and subject resolved."""

    case_id: str
    modality: str
    subject: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class CohortLayout:
    """Where a published release keeps its images and masks, and how to read its filenames."""

    name: str
    images_subdir: str
    labels_subdir: str
    image_glob: str
    case_regex: str
    """Matched against the image filename. Must expose a ``modality`` group unless the layout
    sets :attr:`modality`, plus whatever groups :attr:`subject_template` refers to."""

    subject_template: str
    """Format string over the regex's named groups, e.g. ``"{center}_{pid}"``.

    Deliberately separate from the case id: a release with longitudinal scans names them
    ``..._008_1`` and ``..._008_2``, which are two cases of **one** subject. Folding them apart
    would split a patient across a split.
    """

    description: str
    modality: str | None = None
    """Fixed modality for releases that do not encode one per file."""

    pseudo_labels: bool = False
    """The masks are model predictions — see the module docstring."""

    def iter_cases(
        self, root: Path, *, only_modality: str | None = None
    ) -> Iterator[CohortCase]:
        """Yield the cohort's cases under *root*, sorted.

        Parameters
        ----------
        only_modality
            Keep just ``"ct"`` or ``"mr"``. A release that mixes modalities is often worth
            taking one half of — 109 CTA cases against a 25-case real CT cohort closes a much
            bigger gap than 307 more MRA does, and it keeps the pseudo-to-real ratio sane.

        Raises
        ------
        FileNotFoundError
            If either directory is absent, naming the expected path. A release laid out
            differently should fail here rather than silently contribute nothing.
        """
        root = Path(root)
        images_dir, labels_dir = root / self.images_subdir, root / self.labels_subdir
        for directory, role in ((images_dir, "images"), (labels_dir, "labels")):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Cohort {self.name!r}: expected {role} under {directory}. Point the spec at "
                    f"the release root, or give explicit directories with "
                    f"'name=/images:/labels:modality'."
                )

        pattern = re.compile(self.case_regex)
        unmatched = 0
        for image in sorted(images_dir.glob(self.image_glob)):
            match = pattern.match(image.name)
            if match is None:
                unmatched += 1
                continue
            groups = match.groupdict()
            stem = image.name[: -len(".nii.gz")]
            base = stem[: -len("_0000")] if stem.endswith("_0000") else stem
            mask = labels_dir / f"{base}.nii.gz"
            if not mask.is_file():
                log.warning("[%s] no mask for %s; skipping.", self.name, image.name)
                continue
            modality = (self.modality or groups.get("modality") or "").strip().lower()
            if modality not in ("ct", "mr"):
                raise ValueError(
                    f"Cohort {self.name!r}: could not read a modality for {image.name} "
                    f"(got {modality!r}). Harmonisation depends on it and it is never guessed."
                )
            if only_modality is not None and modality != only_modality:
                continue
            yield CohortCase(
                case_id=base,
                modality=modality,
                subject=self.subject_template.format(**groups),
                image_path=image,
                label_path=mask,
            )
        if unmatched:
            log.warning(
                "[%s] %d file(s) under %s did not match the expected naming scheme and were "
                "skipped.", self.name, unmatched, images_dir,
            )


#: Releases this pipeline knows how to read by name.
BUILTIN_COHORTS: dict[str, CohortLayout] = {
    "topaneu": CohortLayout(
        name="topaneu",
        images_subdir="images",
        labels_subdir="vessel_masks",
        image_glob="topaneu_*_0000.nii.gz",
        # topaneu_<center>_<modality>_<patient>[_<repeat>]_0000.nii.gz
        case_regex=(
            r"^topaneu_(?P<center>center\d+)_(?P<modality>ct|mr)_(?P<pid>\d+)"
            r"(?:_(?P<repeat>\d+))?_0000\.nii\.gz$"
        ),
        # Longitudinal scans share a patient: center-4 ships several ``_008_1`` / ``_008_2``
        # pairs, and the repeat index is deliberately absent from the subject key.
        subject_template="{center}_{pid}",
        pseudo_labels=True,
        description=(
            "TopAneu release: 416 scans (307 MRA / 109 CTA) from four centres. Its "
            "vessel_masks/ are TA36-labelled but were *predicted by the TopBrain organisers' "
            "model*, not annotated — train-only."
        ),
    ),
}


def describe_builtin_cohorts() -> str:
    """Operator-facing listing of the known releases."""
    lines = ["Built-in annotated cohorts (--extra-train / --extra-train-only):", ""]
    for name, layout in sorted(BUILTIN_COHORTS.items()):
        lines.append(f"  {name}={{release_root}}          (all cases)")
        lines.append(f"  {name}:ct={{release_root}}       (one modality only)")
        lines.append(f"    images: {layout.images_subdir}/   masks: {layout.labels_subdir}/")
        lines.append(f"    {layout.description}")
        if layout.pseudo_labels:
            lines.append("    NOTE: pseudo-labelled — must be passed as --extra-train-only.")
        lines.append("")
    lines.append("  Anything else: 'name=/images:/labels:modality'.")
    return "\n".join(lines)


__all__ = [
    "BUILTIN_COHORTS",
    "CohortCase",
    "CohortLayout",
    "describe_builtin_cohorts",
]
