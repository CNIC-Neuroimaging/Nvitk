"""ToPBrain stage 0: data preparation.

Turns raw imaging into the two dataset formats the rest of the pipeline consumes, applying the
same modality-aware intensity harmonisation to both so a model never sees the two on different
scales.

``--target train`` *(default)*
    **Labelled training data** → an nnU-Net raw dataset. The challenge's 50 cases, optionally
    plus your own annotated cohorts (``--extra-train``). Writes ``imagesTr/``, ``labelsTr/``,
    ``dataset.json`` and patient-grouped ``splits_final.json``.
``--target corpus``
    **Unlabeled pre-training corpus** → an nnssl collection. Used only by stage 1's
    ``--source scratch`` route, and deliberately *not* the training data: pre-training on the
    same volumes you fine-tune on buys little and muddies the evaluation.
``--target both``
    Both, in one pass.

Why images are rewritten rather than referenced
-----------------------------------------------
nnU-Net chooses an intensity-normalisation scheme **per channel**, not per case, and nnssl emits
a single scheme for a whole collection while recording only spacings in its fingerprint — it has
no CT normalisation and no way to compute one. CTA is calibrated (Hounsfield units, air at
−1000) and TOF-MRA is not (arbitrary units, hard floor at 0), so neither framework can normalise
a mixed set correctly. Mapping both onto a shared ``[0, 1]`` range here is what makes one
modality-agnostic model legitimate.

Geometry
--------
Spacing, affine and orientation (LPS, frequently oblique on MR) pass through untouched:
harmonisation is a voxelwise intensity map. Every label is checked against its image for shape
and affine agreement before either is written — a mismatch means the pair is not co-registered
and training on it would be silently wrong.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO

import click

from nvitk.core.array import to_numpy
from nvitk.core.backend import map_in_thread_pool, setup
from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.normalization import harmonize_modality
from nvitk.pipes.topbrain import config as cfg
from nvitk.pipes.topbrain import labels as lbl
from nvitk.pipes.topbrain.util import collection as corpus_util
from nvitk.pipes.topbrain.util import folds as fold_util
from nvitk.pipes.topbrain.util.nnssl_env import apply_nnssl_env
from nvitk.pipes.topbrain.util.paths import (
    CORPUS_DATASET_ID,
    DATASET_IDS,
    DATASET_SUFFIXES,
    STAGE0_DATAPREP_DIR,
    ReleaseCase,
    TopBrainPaths,
    iter_release_cases,
)
from nvitk.pipes.topbrain.util.sge_backend import sge_backend_cli_args
from nvitk.pipes.topbrain.util.sge_stage import (
    build_stage_command,
    container_layout,
    quote_path,
    submit_stage_job,
)

setup(globals())

log = Logger()

#: Completion marker / provenance sidecar.
DONE_MARKER: str = "topbrain_stage0.json"

#: Tolerance when comparing an image affine with its label affine, in millimetres. The release
#: stores both at float32 precision, so exact equality is too strict.
AFFINE_ATOL: float = 1e-4

#: ``name=/images:/labels[:modality]`` — an extra annotated cohort.
_EXTRA_RE = re.compile(r"^(?P<name>[^=]+)=(?P<images>[^:]+):(?P<labels>[^:]+)(?::(?P<modality>\w+))?$")


@dataclass(frozen=True)
class ExtraCohort:
    """An additional annotated cohort merged into the training dataset."""

    name: str
    images_dir: Path
    labels_dir: Path
    modality: str

    def iter_cases(self) -> Iterator[ReleaseCase]:
        """Pair images with same-named masks; the case id is prefixed with the cohort name.

        Prefixing keeps ids unique against the challenge's ``topcow_*`` cases and makes the
        provenance obvious in ``splits_final.json``.
        """
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Extra cohort {self.name!r}: {self.images_dir} not found")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Extra cohort {self.name!r}: {self.labels_dir} not found")
        for image in sorted(self.images_dir.glob("*.nii.gz")):
            stem = image.name[: -len(".nii.gz")]
            # Tolerate an nnU-Net channel suffix on the image but not on the mask.
            base = stem[: -len("_0000")] if stem.endswith("_0000") else stem
            mask = self.labels_dir / f"{base}.nii.gz"
            if not mask.is_file():
                log.warning("[%s] no mask for %s; skipping.", self.name, image.name)
                continue
            yield ReleaseCase(
                case_id=f"{self.name}_{base}",
                modality=self.modality,
                # One patient per case unless the cohort encodes otherwise; the prefix keeps
                # these from colliding with challenge patient ids during fold grouping.
                patient_id=f"{self.name}-{base}",
                image_path=image,
                label_path=mask,
            )


def parse_extra_train(spec: str) -> ExtraCohort:
    """Parse ``--extra-train name=/images:/labels[:modality]``."""
    match = _EXTRA_RE.match(spec.strip())
    if match is None:
        raise click.BadParameter(
            f"--extra-train {spec!r} is malformed; expected 'name=/images:/labels[:modality]'."
        )
    modality = (match["modality"] or "").strip().lower()
    if modality not in ("ct", "mr"):
        raise click.BadParameter(
            f"--extra-train {spec!r} needs an explicit modality (ct or mr): the intensity "
            f"harmonisation branch depends on it and guessing would silently apply an HU window "
            f"to arbitrary-unit MR data."
        )
    return ExtraCohort(
        name=match["name"].strip(),
        images_dir=Path(match["images"]).expanduser(),
        labels_dir=Path(match["labels"]).expanduser(),
        modality=modality,
    )


def _verify_label_map(challenge_root: Path, label_set: str) -> None:
    """Cross-check :mod:`nvitk.pipes.topbrain.labels` against the release's published JSON.

    The release ships three label maps whose values collide above 34. If a future release
    renumbers a class, this catches it here rather than after training on wrong anatomy.
    """
    published = challenge_root / "labelmap_jsons" / lbl.LABEL_SET_JSONS[label_set]
    if not published.is_file():
        log.warning("No published label map at %s; skipping the cross-check.", published)
        return
    raw = json.loads(published.read_text(encoding="utf-8"))["labels"]
    released = {int(v): k for k, v in raw.items() if int(v) != 0}
    ours = lbl.label_map(label_set)
    if released != ours:
        raise ValueError(
            f"Label map for {label_set!r} disagrees with {published}. "
            f"Only in release: {({k: v for k, v in released.items() if ours.get(k) != v})}. "
            f"Only in nvitk: {({k: v for k, v in ours.items() if released.get(k) != v})}."
        )


def _convert_case(
    case: ReleaseCase,
    *,
    images_dir: Path,
    labels_dir: Path,
    label_set: str,
    ct_window: Sequence[float],
    mr_percentiles: Sequence[float],
    overwrite: bool,
) -> dict[str, Any]:
    """Harmonise and write one labelled case; returns a provenance record.

    Raises
    ------
    FileNotFoundError, ValueError
        Missing mask, geometry disagreement, or label values outside the declared set. All
        unrecoverable data errors, not something to warn past.
    """
    out_image = images_dir / f"{case.case_id}_0000.nii.gz"
    out_label = labels_dir / f"{case.case_id}.nii.gz"
    if not overwrite and out_image.is_file() and out_label.is_file():
        return {"case": case.case_id, "modality": case.modality, "skipped": True}
    if not case.label_path.is_file():
        raise FileNotFoundError(f"[{case.case_id}] label mask not found: {case.label_path}")

    image, label = imread(case.image_path), imread(case.label_path)

    # ---- 1. Geometry agreement: the pair must describe the same voxel grid ----
    if tuple(image.shape) != tuple(label.shape):
        raise ValueError(
            f"[{case.case_id}] image shape {tuple(image.shape)} != label {tuple(label.shape)}."
        )
    # Affines are host metadata, never backend arrays, so plain NumPy comparison is right here
    # — and no backend ``np`` call follows, so nothing can mix.
    if not bool((abs(to_numpy(image.affine) - to_numpy(label.affine)) <= AFFINE_ATOL).all()):
        raise ValueError(
            f"[{case.case_id}] image and label affines differ by more than {AFFINE_ATOL} mm; "
            f"the pair is not on a common grid."
        )

    # ---- 2. Label values must belong to the declared label set ----------------
    # ``values`` stays on the active backend: ``np`` is CuPy under ``--backend gpu``, so pulling
    # it to the host first and then calling ``np.round`` on it would mix a NumPy array into a
    # CuPy ufunc. Convert once, afterwards, for the Python-level iteration that genuinely needs
    # host scalars.
    values = np.unique(label.data)
    if not bool((values == np.round(values)).all()):
        raise ValueError(f"[{case.case_id}] label mask holds non-integral values.")
    present = [int(v) for v in to_numpy(values)]
    unexpected = sorted(v for v in present if v > lbl.max_label(label_set) or v < 0)
    if unexpected:
        raise ValueError(
            f"[{case.case_id}] label values {unexpected} are outside label set {label_set!r} "
            f"(0..{lbl.max_label(label_set)}). Mixing label sets is the usual cause."
        )

    # ---- 3. Harmonise intensities, keep geometry -----------------------------
    harmonised = harmonize_modality(
        image, case.modality, ct_window=ct_window, mr_percentiles=mr_percentiles
    )
    imsave(out_image, harmonised.astype(np.float32))
    imsave(out_label, label.astype(np.uint8))

    foreground = int(to_numpy((label.data > 0).sum()))
    # Plain Python: multiplying three ints does not need a GPU round-trip, and ``np.array`` on a
    # shape tuple would build a device array to do it.
    total = math.prod(int(s) for s in label.shape)
    record = {
        "case": case.case_id,
        "modality": case.modality,
        "patient": case.patient_id,
        "shape": [int(s) for s in image.shape],
        "spacing_mm": [round(float(s), 6) for s in (image.spacing or ())],
        "classes_present": sorted(v for v in present if v != 0),
        "foreground_fraction": round(foreground / total, 6) if total else 0.0,
        "skipped": False,
    }
    log.step(
        f"{case.case_id} ({case.modality}) shape={record['shape']} "
        f"classes={len(record['classes_present'])} fg={record['foreground_fraction']:.4%}"
    )
    return record


def build_training_dataset(
    *,
    challenge_root: Path,
    nnunet_raw: Path,
    label_set: str,
    modality: str,
    extra_cohorts: Sequence[ExtraCohort],
    num_folds: int,
    seed: int,
    ct_window: Sequence[float],
    mr_percentiles: Sequence[float],
    overwrite: bool,
    workers: int,
    include_challenge: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Build the nnU-Net raw dataset; returns ``(dataset_dir, provenance)``."""
    _verify_label_map(challenge_root, label_set)

    dataset_name = f"Dataset{DATASET_IDS[label_set]:03d}_{DATASET_SUFFIXES[label_set]}"
    dataset_dir = Path(nnunet_raw) / dataset_name
    images_dir, labels_dir = dataset_dir / "imagesTr", dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cases: list[ReleaseCase] = []
    if include_challenge:
        cases.extend(iter_release_cases(challenge_root, label_set=label_set, modality=modality))
    for cohort in extra_cohorts:
        cases.extend(cohort.iter_cases())
    if not cases:
        raise FileNotFoundError(
            f"No training cases for label set {label_set!r} / modality {modality!r}."
        )

    log.info(
        "stage0 train | label_set=%s modality=%s cases=%d (challenge=%s, extra=%s) -> %s",
        label_set, modality, len(cases), include_challenge,
        [c.name for c in extra_cohorts] or "none", dataset_dir,
    )

    def _convert(case: ReleaseCase) -> dict[str, Any]:
        """Convert one case with this run's harmonisation settings."""
        return _convert_case(
            case, images_dir=images_dir, labels_dir=labels_dir, label_set=label_set,
            ct_window=ct_window, mr_percentiles=mr_percentiles, overwrite=overwrite,
        )

    records = map_in_thread_pool(_convert, cases, max_workers=int(workers))

    (dataset_dir / "dataset.json").write_text(
        json.dumps(
            {
                # The channel is already modality-harmonised, so a per-case z-score is right;
                # 'ct' here would re-apply an HU-based normalisation to non-HU data.
                "channel_names": {"0": "zscore"},
                "labels": lbl.nnunet_labels(label_set),
                "numTraining": len(cases),
                "file_ending": ".nii.gz",
                "name": dataset_name,
                "description": (
                    f"TopBrain {label_set} vessel segmentation "
                    f"({lbl.num_foreground(label_set)} foreground classes)."
                ),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    splits = fold_util.grouped_folds(cases, num_folds=num_folds, seed=seed)
    fold_util.check_no_patient_leak(splits)
    (dataset_dir / "splits_final.json").write_text(
        json.dumps(splits, indent=2) + "\n", encoding="utf-8"
    )

    provenance = {
        "dataset": dataset_name,
        "label_set": label_set,
        "num_foreground_classes": lbl.num_foreground(label_set),
        "modality": modality,
        "include_challenge": include_challenge,
        "extra_cohorts": [
            {"name": c.name, "images": str(c.images_dir), "labels": str(c.labels_dir),
             "modality": c.modality} for c in extra_cohorts
        ],
        "num_cases": len(cases),
        "num_patients": len(fold_util.group_by_patient(cases)),
        "num_folds": num_folds,
        "fold_seed": seed,
        "ct_window_hu": list(ct_window),
        "mr_percentiles": list(mr_percentiles),
        "cases": records,
    }
    converted = sum(1 for r in records if not r.get("skipped"))
    log.ok(
        f"stage0 train: {converted} converted, {len(cases) - converted} already present, "
        f"{len(splits)} folds -> {dataset_dir}"
    )
    return dataset_dir, provenance


def build_corpus(
    *,
    paths: TopBrainPaths,
    sources: Sequence[str],
    harmonize: bool,
    overwrite: bool,
    workers: int,
) -> tuple[Path, dict[str, Any]]:
    """Build the unlabeled nnssl collection; returns ``(pretrain_data.json, provenance)``."""
    # nnssl binds its roots at import time, so the environment must be set up first.
    apply_nnssl_env(paths, create=True)

    parsed = [
        corpus_util.parse_source_spec(spec, challenge_root=paths.challenge_root)
        for spec in sources
    ]
    log.info(
        "stage0 corpus | %d source(s): %s",
        len(parsed), ", ".join(f"{s.name}({s.modality})@{s.root}" for s in parsed),
    )

    dataset_dir = paths.nnssl_raw_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    built, volumes = corpus_util.build_collection(
        parsed,
        corpus_root=paths.corpus_root,
        collection_index=CORPUS_DATASET_ID,
        collection_name=paths.corpus_dataset_name,
        harmonize=harmonize,
        overwrite=overwrite,
        workers=workers,
    )

    pretrain_json = dataset_dir / "pretrain_data.json"
    pretrain_json.write_text(
        json.dumps(built.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    by_source: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    for volume in volumes:
        by_source[volume.source] = by_source.get(volume.source, 0) + 1
        by_modality[volume.modality] = by_modality.get(volume.modality, 0) + 1

    if len(volumes) < 200:
        log.warning(
            "Corpus holds only %d volume(s). Pre-training from random initialisation needs far "
            "more; prefer stage 1 --source openmind, or add cohorts.", len(volumes),
        )
    log.ok(f"stage0 corpus: {len(volumes)} volumes {by_source} -> {pretrain_json}")
    return pretrain_json, {
        "collection": paths.corpus_dataset_name,
        "pretrain_data_json": str(pretrain_json),
        "corpus_root": str(paths.corpus_root),
        "harmonized": harmonize,
        "num_volumes": len(volumes),
        "num_subjects": len({v.subject_id for v in volumes}),
        "by_source": by_source,
        "by_modality": by_modality,
        "sources": [
            {"name": s.name, "root": str(s.root), "modality": s.modality, "pattern": s.pattern}
            for s in parsed
        ],
    }


def run_dataprep(
    *,
    paths: TopBrainPaths,
    target: str = "train",
    label_set: str = "ta36",
    modality: str = "both",
    extra_train: Sequence[str] = (),
    include_challenge: bool = True,
    corpus_sources: Sequence[str] = (),
    num_folds: int | None = None,
    seed: int | None = None,
    ct_window: Sequence[float] | None = None,
    mr_percentiles: Sequence[float] | None = None,
    harmonize_corpus: bool = True,
    overwrite: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Prepare training data, a pre-training corpus, or both; returns the provenance record."""
    num_folds = cfg.DEFAULT_NUM_FOLDS if num_folds is None else num_folds
    seed = cfg.DEFAULT_FOLD_SEED if seed is None else seed
    ct_window = tuple(ct_window) if ct_window else cfg.DEFAULT_CT_WINDOW
    mr_percentiles = tuple(mr_percentiles) if mr_percentiles else cfg.DEFAULT_MR_PERCENTILES

    if target not in ("train", "corpus", "both"):
        raise ValueError(f"Unknown --target {target!r}; expected train, corpus or both.")

    provenance: dict[str, Any] = {
        "stage": "stage0",
        "created": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "challenge_root": str(paths.challenge_root),
    }

    if target in ("train", "both"):
        cohorts = [parse_extra_train(s) for s in extra_train]
        _, train_meta = build_training_dataset(
            challenge_root=paths.challenge_root,
            nnunet_raw=paths.nnunet_raw,
            label_set=label_set,
            modality=modality,
            extra_cohorts=cohorts,
            num_folds=num_folds,
            seed=seed,
            ct_window=ct_window,
            mr_percentiles=mr_percentiles,
            overwrite=overwrite,
            workers=workers,
            include_challenge=include_challenge,
        )
        provenance["train"] = train_meta

    if target in ("corpus", "both"):
        if not corpus_sources:
            raise ValueError(
                "--target corpus needs at least one --corpus-source. The training data is "
                "deliberately not a valid corpus source; see the module docstring."
            )
        _, corpus_meta = build_corpus(
            paths=paths, sources=corpus_sources, harmonize=harmonize_corpus,
            overwrite=overwrite, workers=workers,
        )
        provenance["corpus"] = corpus_meta

    marker_dir = paths.results_root / STAGE0_DATAPREP_DIR
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / DONE_MARKER).write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _worker_argv(**options) -> list[str]:
    """Worker argv for stage 0, built against the container-side layout."""
    from nvitk.cluster.sge import python_module_argv

    inside = container_layout()
    argv = [
        *python_module_argv("nvitk.pipes.topbrain.stage0_dataprep"),
        *sge_backend_cli_args(options.get("backend", "cpu")),
        "--challenge-root", quote_path(inside.challenge_root),
        "--nnunet-raw", quote_path(inside.nnunet_raw),
        "--nnssl-raw", quote_path(inside.nnssl_raw),
        "--nnssl-preprocessed", quote_path(inside.nnssl_preprocessed),
        "--nnssl-results", quote_path(inside.nnssl_results),
        "--corpus-root", quote_path(inside.corpus_root),
        "--results-root", quote_path(inside.results_root),
        "--target", options.get("target", "train"),
        "--label-set", options.get("label_set", "ta36"),
        "--modality", options.get("modality", "both"),
        "--num-folds", str(int(options.get("num_folds") or cfg.DEFAULT_NUM_FOLDS)),
        "--seed", str(int(options.get("seed") or cfg.DEFAULT_FOLD_SEED)),
        "--workers", str(int(options.get("workers", 1))),
    ]
    for spec in options.get("extra_train", ()):
        argv.extend(["--extra-train", quote_path(spec)])
    for spec in options.get("corpus_sources", ()):
        argv.extend(["--corpus-source", quote_path(spec)])
    if not options.get("include_challenge", True):
        argv.append("--no-challenge")
    if options.get("overwrite"):
        argv.append("--overwrite")
    return argv


def build_sge_command(*, paths, container: Path, src_dir: Path | None = None, **options) -> str:
    """Host shell command for the stage 0 SGE task."""
    return build_stage_command(
        "stage0", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "cpu"),
        # Harmonisation is voxelwise array work; it benefits from CuPy but never needs SGE to
        # reserve a GPU, and asking for one would queue behind the training jobs.
        request_gpu=False, job_suffix=options.get("label_set", ""),
    )


def submit_sge(
    *, paths, container: Path, src_dir: Path | None = None, hold_jid: str | None = None,
    dry_run: bool = False, emit: TextIO | None = None, **options,
) -> str:
    """Emit or submit the stage 0 SGE job."""
    return submit_stage_job(
        "stage0", _worker_argv(**options), paths=paths, container=container, src_dir=src_dir,
        backend=options.get("backend", "cpu"), request_gpu=False,
        job_suffix=options.get("label_set", ""), hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


@click.command("topbrain-stage0-dataprep")
@config_dir_click_option()
@backend_click_option(default="cpu")
@click.option("--challenge-root", type=click.Path(path_type=Path), required=True)
@click.option("--nnunet-raw", type=click.Path(path_type=Path), required=True)
@click.option("--nnssl-raw", type=click.Path(path_type=Path), required=True)
@click.option("--nnssl-preprocessed", type=click.Path(path_type=Path), required=True)
@click.option("--nnssl-results", type=click.Path(path_type=Path), required=True)
@click.option("--corpus-root", type=click.Path(path_type=Path), required=True)
@click.option("--results-root", type=click.Path(path_type=Path), required=True)
@click.option("--target", type=click.Choice(["train", "corpus", "both"]), default="train",
              show_default=True, help="Labelled training data, unlabeled corpus, or both.")
@click.option("--label-set", type=click.Choice(["ta36", "v1_ct", "v1_mr"]), default="ta36",
              show_default=True)
@click.option("--modality", type=click.Choice(["both", "ct", "mr"]), default="both",
              show_default=True)
@click.option("--extra-train", multiple=True,
              help="Extra annotated cohort: 'name=/images:/labels:modality'. Repeatable.")
@click.option("--no-challenge", is_flag=True, default=False,
              help="Exclude the challenge cases; train only on --extra-train cohorts.")
@click.option("--corpus-source", "corpus_sources", multiple=True,
              help="Unlabeled corpus source: 'name:modality=/path[:glob]'. Repeatable.")
@click.option("--num-folds", type=int, default=None)
@click.option("--seed", type=int, default=None, help="Patient-grouped fold assignment seed.")
@click.option("--ct-window", type=float, nargs=2, default=None, help="CT clip window in HU.")
@click.option("--mr-percentiles", type=float, nargs=2, default=None)
@click.option("--no-harmonize-corpus", is_flag=True, default=False,
              help="Reference corpus volumes as-is. Only safe for a single-modality corpus.")
@click.option("--overwrite", is_flag=True, default=True)
@click.option("--workers", type=int, default=1, show_default=True)
def main(
    challenge_root: Path, nnunet_raw: Path, nnssl_raw: Path, nnssl_preprocessed: Path,
    nnssl_results: Path, corpus_root: Path, results_root: Path, target: str, label_set: str,
    modality: str, extra_train: tuple[str, ...], no_challenge: bool,
    corpus_sources: tuple[str, ...], num_folds: int | None, seed: int | None,
    ct_window: tuple[float, float] | None, mr_percentiles: tuple[float, float] | None,
    no_harmonize_corpus: bool, overwrite: bool, workers: int,
) -> None:
    """CLI entry point: prepare training data and/or a pre-training corpus."""
    Logger()
    paths = TopBrainPaths(
        challenge_root=challenge_root, nnssl_raw=nnssl_raw,
        nnssl_preprocessed=nnssl_preprocessed, nnssl_results=nnssl_results,
        nnunet_raw=nnunet_raw, nnunet_preprocessed=results_root, nnunet_results=results_root,
        results_root=results_root, model_root=results_root, corpus_root=corpus_root,
    )
    run_dataprep(
        paths=paths, target=target, label_set=label_set, modality=modality,
        extra_train=extra_train, include_challenge=not no_challenge,
        corpus_sources=corpus_sources, num_folds=num_folds, seed=seed,
        ct_window=ct_window or None, mr_percentiles=mr_percentiles or None,
        harmonize_corpus=not no_harmonize_corpus, overwrite=overwrite, workers=workers,
    )


__all__ = [
    "ExtraCohort", "build_corpus", "build_sge_command", "build_training_dataset",
    "main", "parse_extra_train", "run_dataprep", "submit_sge",
]


if __name__ == "__main__":
    main()
