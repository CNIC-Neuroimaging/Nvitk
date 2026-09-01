"""Filesystem layout for the ToPBrain pipeline (local workstation vs cluster).

Description
-----------
Every root comes from ``sge.json``'s ``pipelines.topbrain_paths`` section — there are no
installation paths in this file. A root that is needed but unconfigured raises
:class:`~nvitk.core.config_paths.ConfigError` naming the key, rather than silently resolving
to somebody else's filesystem. Configuration is read on use, not on import, so ``--config-dir``
and a mid-process :func:`~nvitk.core.config_paths.set_config_dir` are both honoured.

Data layout
-----------
The challenge release is read-only and keeps its published shape::

    <challenge_root>/imagesTr_topbrain/topcow_{ct|mr}_<pid>_0000.nii.gz
    <challenge_root>/labelsTr_topbrain_v2_topaneu36class/topcow_{ct|mr}_<pid>.nii.gz
    <challenge_root>/labelmap_jsons/labels_topbrain_v2_topaneu36class.json

Everything we generate lives under four independent roots, so nnssl and nnU-Net never share a
directory and either can be wiped without touching the other::

    <nnssl_raw|nnssl_preprocessed|nnssl_results>/     # nnssl_* env vars (SSL route only)
    <nnunet_raw|nnunet_preprocessed|nnunet_results>/  # nnUNet_* env vars
    <corpus_root>/                                    # assembled unlabeled pre-training corpus
    <results_root>/<stage*>/                          # adapters, predictions, metrics, docker
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from nvitk.cluster import sge_json as _sj
from nvitk.core import config_paths, lazy_config
from nvitk.pipes.topbrain import labels as _labels

#: ``sge.json`` section holding this pipeline's data roots.
PIPELINE_PATHS_ID = "topbrain_paths"

# ──────────────────────────────────────────────────────────────────────────────
# Release layout constants
# ──────────────────────────────────────────────────────────────────────────────

#: Image subdirectory of the challenge release.
RELEASE_IMAGES_DIR: str = "imagesTr_topbrain"

#: Label-map JSON subdirectory of the challenge release.
RELEASE_LABELMAP_DIR: str = "labelmap_jsons"

#: ``topcow_{modality}_{pid}_0000.nii.gz`` — the release's only image naming scheme.
_IMAGE_RE = re.compile(r"^topcow_(?P<modality>ct|mr)_(?P<pid>\d+)_0000\.nii\.gz$")

#: nnU-Net dataset ids per label set. Chosen in the 5xx range to stay clear of the public
#: MSD/AMOS ids that commonly share an ``nnUNet_raw``.
DATASET_IDS: dict[str, int] = {
    "ta36": 501, "v1_ct": 502, "v1_mr": 503,
    # One binary dataset per multi-class set it seeds — see labels.BINARY_LABEL_SET_FOR.
    "binary": 504, "binary_ct": 505, "binary_mr": 506,
}

#: nnU-Net dataset name suffix per label set.
DATASET_SUFFIXES: dict[str, str] = {
    "ta36": "TopBrainTA36",
    "v1_ct": "TopBrainV1CT",
    "v1_mr": "TopBrainV1MR",
    "binary": "TopBrainVesselBinary",
    "binary_ct": "TopBrainVesselBinaryCT",
    "binary_mr": "TopBrainVesselBinaryMR",
}

#: nnssl collection id/name for the self-supervised pre-training corpus.
CORPUS_DATASET_ID: int = 511
CORPUS_DATASET_SUFFIX: str = "TopBrainCorpus"

# ──────────────────────────────────────────────────────────────────────────────
# Stage output subdirectories (under ``results_root``)
# ──────────────────────────────────────────────────────────────────────────────

STAGE0_DATAPREP_DIR: str = "stage0_dataprep"
STAGE1_PRETRAIN_DIR: str = "stage1_pretrain"
STAGE2_TRAIN_DIR: str = "stage2_train"
STAGE3_EVAL_DIR: str = "stage3_evaluate"
STAGE4_INFER_DIR: str = "stage4_infer"
STAGE5_PACKAGE_DIR: str = "stage5_package"
STAGE6_SELFTRAIN_DIR: str = "stage6_selftrain"

#: TensorBoard event tree, shared by every stage that trains. Lives under ``results_root`` so
#: it is bind-mounted on the cluster and visible through the same mount on the workstation —
#: which is what lets a cluster run be watched locally.
TENSORBOARD_DIR: str = "tensorboard"

#: Root keys of ``pipelines.topbrain_paths``, without the ``local_``/``cluster_`` prefix.
ROOT_KEYS: tuple[str, ...] = (
    "challenge_root",
    "nnssl_raw",
    "nnssl_preprocessed",
    "nnssl_results",
    "nnunet_raw",
    "nnunet_preprocessed",
    "nnunet_results",
    "results_root",
    "model_root",
    "corpus_root",
)


def _pipe_paths() -> dict:
    """The ``pipelines.topbrain_paths`` block, read fresh (and cached) on each use."""
    return _sj.pipeline_section(PIPELINE_PATHS_ID)


def _opt_root(key: str) -> Path | None:
    """The configured root *key* as a path, or ``None`` when unset.

    Deliberately does not raise: these are read at import time as Click option defaults, so an
    unconfigured machine must still be able to print ``--help``. :func:`layout_local` /
    :func:`layout_cluster` raise, naming the key, when a root is actually needed.
    """
    raw = _pipe_paths().get(key)
    if raw is None or not str(raw).strip():
        return None
    return Path(os.path.expanduser(str(raw).strip()))


_RESOLVERS: dict[str, lazy_config.Resolver] = {
    **{f"DEFAULT_{k.upper()}": (lambda k=k: _opt_root(f"cluster_{k}")) for k in ROOT_KEYS},
    **{f"LOCAL_DEFAULT_{k.upper()}": (lambda k=k: _opt_root(f"local_{k}")) for k in ROOT_KEYS},
    "CLUSTER_HOST_ALIASES": lambda: _sj.merge_cluster_host_aliases(
        {}, _sj.paths_section(), _pipe_paths()
    ),
}

__getattr__, __dir__ = lazy_config.module_getattr(_RESOLVERS, module_name=__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Release cases
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReleaseCase:
    """One image of the challenge release, with the label mask of a chosen label set.

    ``patient_id`` is shared by the CT and MR case of the same subject — cross-validation
    folds must be grouped on it or the two halves of a patient land on both sides of a split.
    """

    case_id: str
    """``topcow_ct_001`` — the nnU-Net case identifier (no ``_0000`` channel suffix)."""

    modality: str
    """``"ct"`` (CTA) or ``"mr"`` (MRA/TOF)."""

    patient_id: str
    """Zero-padded subject id, e.g. ``"001"``. Shared across modalities."""

    image_path: Path
    label_path: Path

    @property
    def has_label(self) -> bool:
        """Whether the label mask for this case exists on disk."""
        return self.label_path.is_file()


def iter_release_cases(
    challenge_root: Path,
    *,
    label_set: str = "ta36",
    modality: str = "both",
) -> Iterator[ReleaseCase]:
    """Yield the release's cases in sorted order, filtered by *modality*.

    Parameters
    ----------
    label_set
        Which label directory to pair images with — see :mod:`nvitk.pipes.topbrain.labels`.
        ``v1_ct`` / ``v1_mr`` only cover one modality, so cases of the other modality are
        skipped rather than yielded label-less.
    modality
        ``"both"``, ``"ct"`` or ``"mr"``.

    Raises
    ------
    FileNotFoundError
        If the images directory or the label directory of *label_set* is absent.
    """
    images_dir = challenge_root / RELEASE_IMAGES_DIR
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Release images directory not found: {images_dir}")

    labels_dir = challenge_root / _labels.LABEL_SET_DIRS[label_set]
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Label directory for label set {label_set!r} not found: {labels_dir}"
        )

    covered = _labels.LABEL_SET_MODALITIES[label_set]
    for image_path in sorted(images_dir.glob("topcow_*_0000.nii.gz")):
        match = _IMAGE_RE.match(image_path.name)
        if match is None:
            continue
        mod = match["modality"]
        if modality != "both" and mod != modality:
            continue
        if mod not in covered:
            # e.g. the MR cases under label set v1_ct: no mask exists, so they are not cases.
            continue
        case_id = f"topcow_{mod}_{match['pid']}"
        yield ReleaseCase(
            case_id=case_id,
            modality=mod,
            patient_id=match["pid"],
            image_path=image_path,
            label_path=labels_dir / f"{case_id}.nii.gz",
        )


def parse_case_id(case_id: str) -> tuple[str, str]:
    """Split a case id such as ``topcow_mr_017`` into ``(modality, patient_id)``.

    Raises
    ------
    ValueError
        If *case_id* does not follow the release naming scheme — callers use the modality to
        pick an intensity-normalisation branch, so guessing would be a silent data error.
    """
    parts = case_id.split("_")
    if len(parts) != 3 or parts[0] != "topcow" or parts[1] not in ("ct", "mr"):
        raise ValueError(
            f"Case id {case_id!r} does not match the release scheme 'topcow_{{ct|mr}}_<pid>'."
        )
    return parts[1], parts[2]


# ──────────────────────────────────────────────────────────────────────────────
# Resolved layout
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TopBrainPaths:
    """Resolved roots for one execution context (local workstation or cluster)."""

    challenge_root: Path
    nnssl_raw: Path
    nnssl_preprocessed: Path
    nnssl_results: Path
    nnunet_raw: Path
    nnunet_preprocessed: Path
    nnunet_results: Path
    results_root: Path
    model_root: Path
    corpus_root: Path

    # ---- Release accessors ---------------------------------------------------

    @property
    def release_images_dir(self) -> Path:
        """The release's read-only image directory."""
        return self.challenge_root / RELEASE_IMAGES_DIR

    def release_labels_dir(self, label_set: str) -> Path:
        """The release's read-only label directory for *label_set*."""
        return self.challenge_root / _labels.LABEL_SET_DIRS[label_set]

    def release_labelmap_json(self, label_set: str) -> Path:
        """The release's published label-map JSON for *label_set*."""
        return self.challenge_root / RELEASE_LABELMAP_DIR / _labels.LABEL_SET_JSONS[label_set]

    # ---- nnU-Net dataset accessors -------------------------------------------

    def dataset_name(self, label_set: str) -> str:
        """nnU-Net dataset folder name for *label_set*, e.g. ``Dataset501_TopBrainTA36``."""
        return f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"

    def nnunet_raw_dir(self, label_set: str) -> Path:
        """``<nnunet_raw>/DatasetXXX_...`` for *label_set*."""
        return self.nnunet_raw / self.dataset_name(label_set)

    def nnunet_preprocessed_dir(self, label_set: str) -> Path:
        """``<nnunet_preprocessed>/DatasetXXX_...`` for *label_set*."""
        return self.nnunet_preprocessed / self.dataset_name(label_set)

    def nnunet_results_dir(self, label_set: str) -> Path:
        """``<nnunet_results>/DatasetXXX_...`` for *label_set*."""
        return self.nnunet_results / self.dataset_name(label_set)

    # ---- nnssl collection accessors ------------------------------------------

    @property
    def corpus_dataset_name(self) -> str:
        """nnssl collection folder name, e.g. ``Dataset511_TopBrainCorpus``."""
        return f"Dataset{CORPUS_DATASET_ID:03d}_{CORPUS_DATASET_SUFFIX}"

    @property
    def nnssl_raw_dir(self) -> Path:
        """``<nnssl_raw>/Dataset511_TopBrainCorpus`` — holds ``pretrain_data.json``."""
        return self.nnssl_raw / self.corpus_dataset_name

    # ---- Stage output accessors ----------------------------------------------

    def stage_dir(self, name: str) -> Path:
        """A stage output directory under ``results_root``."""
        return self.results_root / name

    def ensure_dirs(self, *dirs: Path) -> None:
        """``mkdir -p`` each of *dirs*. Never touches the read-only challenge root."""
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)


def _root_from_config(key: str, *, prefix: str, fallback: Path | None) -> Path:
    """Resolve one root, honouring the pipeline's flag-vs-config precedence.

    Under ``--submit local`` the CLI flag wins (*fallback* is preferred). Under ``--submit sge``
    the ``cluster_*`` config wins, because the flag holds a host path that is meaningless on the
    cluster. That inversion is a repo-wide convention — see ``docs/configuration.md``.
    """
    raw = _pipe_paths().get(f"{prefix}_{key}")
    if prefix == "cluster" and raw is not None and str(raw).strip():
        return Path(os.path.expanduser(str(raw).strip()))
    if fallback is not None:
        return Path(fallback)
    return Path(
        os.path.expanduser(
            str(
                config_paths.require(
                    raw,
                    key=f"pipelines.{PIPELINE_PATHS_ID}.{prefix}_{key}",
                    hint="Set it, or pass the matching --*-root flag.",
                )
            ).strip()
        )
    )


def _layout(prefix: str, **overrides: Path | None) -> TopBrainPaths:
    """Build a :class:`TopBrainPaths` from the ``local_``/``cluster_`` half of the config."""
    return TopBrainPaths(
        **{
            key: _root_from_config(key, prefix=prefix, fallback=overrides.get(key))
            for key in ROOT_KEYS
        }
    )


#: Minimum number of path components an existing ancestor must have before it counts as
#: evidence of a mount. ``/data`` existing means nothing; ``/data_lab_MCC/user/LabMCC`` does.
_MOUNT_EVIDENCE_DEPTH: int = 4


def tree_visible(path: Path) -> bool:
    """Whether *path*, or a specific enough ancestor of it, exists on this host.

    A root the cluster has not written to yet exists nowhere, so testing it directly would
    report "not mounted" on a perfectly good mount before the first run. Walking up to the
    deepest existing ancestor answers the question actually being asked: is this filesystem
    reachable from here?
    """
    current = Path(path)
    while len(current.parts) >= _MOUNT_EVIDENCE_DEPTH:
        if current.is_dir():
            return True
        current = current.parent
    return False


def layout_auto(**overrides: Path | None) -> tuple[TopBrainPaths, str]:
    """The layout an interactive tool on this host should read; with which one it picked.

    Cluster storage is commonly NFS-mounted on the workstation at the *same absolute paths*, so
    the ``cluster_*`` roots — the ones jobs actually write to — are usually readable here, while
    the ``local_*`` roots are a separate working copy holding nothing a cluster job produced. A
    tool that defaulted to ``local`` would therefore report "no trained models" while the models
    sat right there.

    Prefers the cluster roots when they are reachable, falls back to local otherwise, and says
    which in the returned string so the choice is never silent.

    Returns
    -------
    tuple
        ``(paths, origin)`` with *origin* one of ``"cluster"`` / ``"local"``.
    """
    local = layout_local(**overrides)
    try:
        cluster = layout_cluster(**overrides)
    except Exception:  # cluster roots are optional on a workstation-only install
        return local, "local"
    if tree_visible(cluster.results_root) or Path(cluster.challenge_root).is_dir():
        return cluster, "cluster"
    return local, "local"


def layout_local(**overrides: Path | None) -> TopBrainPaths:
    """Roots for running on the analysis workstation (``local_*`` config keys).

    Keyword overrides are the CLI flags and take precedence over the configured values.
    """
    return _layout("local", **overrides)


def layout_cluster(**overrides: Path | None) -> TopBrainPaths:
    """Roots for running on the SGE cluster (``cluster_*`` config keys).

    Configured values take precedence here: a host path passed on the workstation's command
    line does not exist on the cluster.
    """
    return _layout("cluster", **overrides)


__all__ = [
    "CORPUS_DATASET_ID",
    "CORPUS_DATASET_SUFFIX",
    "DATASET_IDS",
    "DATASET_SUFFIXES",
    "PIPELINE_PATHS_ID",
    "RELEASE_IMAGES_DIR",
    "RELEASE_LABELMAP_DIR",
    "ROOT_KEYS",
    "STAGE0_DATAPREP_DIR",
    "STAGE1_PRETRAIN_DIR",
    "STAGE2_TRAIN_DIR",
    "STAGE3_EVAL_DIR",
    "STAGE4_INFER_DIR",
    "STAGE5_PACKAGE_DIR",
    "STAGE6_SELFTRAIN_DIR",
    "TENSORBOARD_DIR",
    "ReleaseCase",
    "TopBrainPaths",
    "iter_release_cases",
    "layout_auto",
    "layout_cluster",
    "layout_local",
    "tree_visible",
    "parse_case_id",
]
