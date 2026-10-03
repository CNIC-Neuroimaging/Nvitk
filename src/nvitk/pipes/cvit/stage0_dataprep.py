"""
CViT stage 0 — labelled data → validated nnU-Net raw dataset (+ optional nnssl corpus).

Description
-----------
Accepts either an existing nnU-Net-style folder (``imagesTr/<case>_0000.nii.gz`` …,
``labelsTr/<case>.nii.gz``) or a plain pair of folders (``images/<case>.nii.gz``,
``labels/<case>.nii.gz``), validates every case, and writes
``<nnunet_raw>/Dataset<ID>_<Name>/{imagesTr,labelsTr,dataset.json}`` plus a patient-grouped
``splits_final.json`` for nnU-Net's cross-validation.

Validation (per case, fail loudly)
----------------------------------
- every channel and the label have the same **shape** and the same **affine** (``allclose``,
  ``atol=1e-3`` mm) — a mismatched grid trains on misregistered pairs without any error;
- images are finite (no NaN / Inf);
- labels hold integers only and only declared label values; a float label with integral values
  is rewritten as an integer volume (warned), a non-integral one is an error;
- the label has foreground (``--allow-empty`` downgrades this to a warning).

Files already in ``.nii.gz`` are copied byte-for-byte (lossless, header preserved); other
formats are converted through :func:`nvitk.io.imsave`, which writes voxels **and** geometry.

Folds
-----
``--group-regex`` (e.g. ``'^(?P<group>sub-\\d+)'``) keeps all cases of one patient in one fold.
Groups are shuffled with ``--seed`` and greedily assigned to the fold with the fewest cases.

Corpus (optional)
-----------------
``--corpus-source`` (``name:modality=/path[:glob]``, see
:func:`nvitk.pipes.topbrain.util.collection.parse_source_spec`) and/or ``--corpus-from-train``
build an nnssl ``pretrain_data.json`` under ``<nnssl_raw>/Dataset<CID>_<Name>Corpus/``. nnssl
pre-trains single-channel, so ``--corpus-from-train`` uses channel 0.

Array / axis conventions: arrays as returned by :func:`nvitk.io.imread`; backend ``np`` after
``setup(globals())``; host conversion via :func:`~nvitk.core.array.to_numpy`.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import click

from nvitk.core import setup
from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import map_in_thread_pool
from nvitk.core.click_backend import backend_click_option
from nvitk.core.click_config import config_dir_click_option
from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.pipes.cvit import config as cfg
from nvitk.pipes.cvit.util import paths as pth
from nvitk.pipes.cvit.util.cli import paths_from_options, pop_roots, root_options
from nvitk.pipes.cvit.util.paths import CViTPaths
from nvitk.pipes.cvit.util.sge_stage import (
    container_layout,
    python_module_argv,
    quote_path,
    root_args,
    sge_backend_cli_args,
    submit_stage_job,
    to_container_path,
)

setup(globals())
log = Logger()

#: Image extensions recognised, longest first so ``.nii.gz`` wins over ``.gz``.
IMAGE_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".nhdr")
_CHANNEL_RE = re.compile(r"^(?P<case>.+)_(?P<ch>\d{4})$")
STAGE0_MARKER = "cvit_stage0.json"

# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One labelled case: channel image paths (ordered) and its label path."""

    case_id: str
    images: list[Path] = field(default_factory=list)
    label: Path | None = None


def _split_suffix(path: Path) -> tuple[str, str]:
    name = path.name
    for suffix in IMAGE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix
    return path.stem, path.suffix


def default_dirs(data_root: Path) -> tuple[Path, Path]:
    """``imagesTr``/``labelsTr`` if present, else ``images``/``labels`` under *data_root*."""
    for img, lab in (("imagesTr", "labelsTr"), ("images", "labels")):
        if (data_root / img).is_dir():
            return data_root / img, data_root / lab
    raise FileNotFoundError(
        f"No imagesTr/ or images/ folder under {data_root}. Pass --images-dir / --labels-dir."
    )


def discover_cases(images_dir: Path, labels_dir: Path) -> list[Case]:
    """Pair images with labels.

    ``<case>_0000.ext`` names are treated as nnU-Net channels; anything else is a single-channel
    ``<case>.ext``. Every case must have the same channel set and exactly one label.

    Raises
    ------
    FileNotFoundError / ValueError
        On missing folders, missing labels, inconsistent channel sets, or no cases at all.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")
    channels: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in sorted(images_dir.iterdir()):
        stem, suffix = _split_suffix(path)
        if suffix not in IMAGE_SUFFIXES or not path.is_file():
            continue
        m = _CHANNEL_RE.match(stem)
        case_id, ch = (m["case"], int(m["ch"])) if m else (stem, 0)
        if ch in channels[case_id]:
            raise ValueError(f"Case {case_id!r} has two files for channel {ch}.")
        channels[case_id][ch] = path
    if not channels:
        raise ValueError(f"No images with suffixes {IMAGE_SUFFIXES} in {images_dir}.")

    labels: dict[str, Path] = {}
    for path in sorted(labels_dir.iterdir()):
        stem, suffix = _split_suffix(path)
        if suffix in IMAGE_SUFFIXES and path.is_file():
            labels[stem] = path

    expected = sorted(next(iter(channels.values())))
    cases, missing = [], []
    for case_id in sorted(channels):
        chans = channels[case_id]
        if sorted(chans) != expected:
            raise ValueError(f"Case {case_id!r} has channels {sorted(chans)}, expected {expected}.")
        if expected != list(range(len(expected))):
            raise ValueError(f"Channels must be numbered 0..N-1, got {expected}.")
        if case_id not in labels:
            missing.append(case_id)
            continue
        cases.append(Case(case_id, [chans[c] for c in expected], labels[case_id]))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} case(s) have no label in {labels_dir}, e.g. {missing[:5]}."
        )
    return cases


# ---------------------------------------------------------------------------
# Validation + writing
# ---------------------------------------------------------------------------


def _affine_close(a, b, atol: float = 1e-3) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = to_numpy(a), to_numpy(b)
    return a.shape == b.shape and bool(abs(a - b).max() <= atol + 1e-4 * abs(b).max())


def validate_case(case: Case, *, declared: set[int] | None, allow_empty: bool) -> dict[str, Any]:
    """Check one case (see module docstring); returns a record for provenance.

    Raises
    ------
    ValueError
        Naming the case and what is wrong with it.
    """
    label = imread(case.label, backend="cpu")
    lab = as_backend_array(label.data)
    record: dict[str, Any] = {"case": case.case_id, "shape": list(lab.shape)}
    for ch, path in enumerate(case.images):
        img = imread(path, backend="cpu")
        if tuple(img.data.shape) != tuple(lab.shape):
            raise ValueError(
                f"{case.case_id}: channel {ch} shape {tuple(img.data.shape)} != label {tuple(lab.shape)}."
            )
        if not _affine_close(img.affine, label.affine):
            raise ValueError(
                f"{case.case_id}: channel {ch} affine differs from the label's — the pair is "
                f"not on the same grid. Resample the label onto the image (nearest) first."
            )
        if not bool(np.isfinite(as_backend_array(img.data)).all()):
            raise ValueError(f"{case.case_id}: channel {ch} contains NaN or Inf.")
        if ch == 0:
            record["spacing"] = [float(s) for s in (img.spacing or ())]

    # ---- labels: integral, declared, non-empty --------------------------------------------
    values = to_numpy(np.unique(lab))
    if values.dtype.kind == "f":
        if not bool((values == values.round()).all()):
            raise ValueError(f"{case.case_id}: label has non-integer values {values[:8]}.")
        record["label_was_float"] = True
    ints = sorted(int(v) for v in values)
    record["labels_present"] = ints
    if declared is not None:
        unknown = sorted(set(ints) - declared)
        if unknown:
            raise ValueError(f"{case.case_id}: undeclared label value(s) {unknown}.")
    if not any(v != 0 for v in ints):
        if not allow_empty:
            raise ValueError(f"{case.case_id}: label is empty (no foreground). Use --allow-empty to keep it.")
        log.warning("%s: empty label kept (--allow-empty).", case.case_id)
    return record


def _write_case(case: Case, dataset_dir: Path, record: dict[str, Any], overwrite: bool) -> None:
    images_out, labels_out = dataset_dir / "imagesTr", dataset_dir / "labelsTr"
    for ch, src in enumerate(case.images):
        dst = images_out / f"{case.case_id}_{ch:04d}.nii.gz"
        if dst.exists() and not overwrite:
            continue
        if src.name.endswith(".nii.gz"):
            shutil.copyfile(src, dst)
        else:
            imsave(dst, imread(src, backend="cpu"))
    dst = labels_out / f"{case.case_id}.nii.gz"
    if dst.exists() and not overwrite:
        return
    if record.get("label_was_float") or not case.label.name.endswith(".nii.gz"):
        label = imread(case.label, backend="cpu")
        data = as_backend_array(label.data)
        top = int(to_numpy(data.max()))
        dtype = np.uint8 if top < 256 else np.uint16
        imsave(dst, label.with_data(np.rint(data).astype(dtype)))
        if record.get("label_was_float"):
            log.warning("%s: float label rewritten as %s.", case.case_id, np.dtype(dtype).name)
    else:
        shutil.copyfile(case.label, dst)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def grouped_splits(case_ids: Sequence[str], *, num_folds: int, seed: int,
                   group_regex: str | None = None) -> list[dict[str, list[str]]]:
    """Patient-grouped k-fold splits in nnU-Net's ``splits_final.json`` format.

    Raises
    ------
    ValueError
        If there are fewer groups than folds, or the regex does not match a case.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    pattern = re.compile(group_regex) if group_regex else None
    for cid in case_ids:
        if pattern is None:
            key = cid
        else:
            m = pattern.search(cid)
            if m is None:
                raise ValueError(f"--group-regex {group_regex!r} does not match case {cid!r}.")
            key = m.groupdict().get("group") or (m.group(1) if m.groups() else m.group(0))
        groups[key].append(cid)
    if len(groups) < num_folds:
        raise ValueError(f"{len(groups)} group(s) cannot fill {num_folds} folds; lower --num-folds.")
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))              # big groups first, ties keep shuffle
    folds: list[list[str]] = [[] for _ in range(num_folds)]
    for g in order:
        target = min(range(num_folds), key=lambda f: len(folds[f]))
        folds[target].extend(groups[g])
    splits = []
    for f in range(num_folds):
        val = sorted(folds[f])
        train = sorted(c for k, fold in enumerate(folds) if k != f for c in fold)
        splits.append({"train": train, "val": val})
    return splits


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def corpus_folder_name(corpus_id: int, dataset_name: str) -> str:
    return pth.dataset_folder_name(corpus_id, f"{dataset_name}Corpus")


def build_corpus(
    paths: CViTPaths, *, corpus_id: int, dataset_name: str, sources: Sequence[str],
    from_train: Path | None, harmonize: bool, workers: int, overwrite: bool,
) -> dict[str, Any]:
    """Write ``pretrain_data.json`` for nnssl; returns provenance."""
    from nvitk.pipes._engines.env import apply_nnssl_env
    from nvitk.pipes.topbrain.util import collection as corpus_util

    apply_nnssl_env(paths.nnssl_raw, paths.nnssl_preprocessed, paths.nnssl_results)
    parsed = [corpus_util.parse_source_spec(spec) for spec in sources]
    if from_train is not None:
        parsed.append(corpus_util.CorpusSource(
            name="train", root=from_train, modality="mr", pattern="*_0000.nii.gz",
            subject_regex=r"^(?P<subject>.+)_0000\.nii\.gz$",
        ))
    if not parsed:
        raise ValueError("No corpus source given.")
    name = corpus_folder_name(corpus_id, dataset_name)
    out_dir = paths.nnssl_raw / name
    out_dir.mkdir(parents=True, exist_ok=True)
    built, volumes = corpus_util.build_collection(
        parsed, corpus_root=paths.corpus_root, collection_index=corpus_id, collection_name=name,
        harmonize=harmonize, overwrite=overwrite, workers=workers,
    )
    target = out_dir / "pretrain_data.json"
    target.write_text(json.dumps(built.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if len(volumes) < 100:
        log.warning("Corpus holds only %d volume(s); self-supervised pre-training usually needs "
                    "hundreds to help.", len(volumes))
    log.ok(f"stage0 corpus: {len(volumes)} volume(s) -> {target}")
    return {"collection": name, "pretrain_data_json": str(target), "num_volumes": len(volumes),
            "harmonized": harmonize}


# ---------------------------------------------------------------------------
# Library entry point
# ---------------------------------------------------------------------------


def run_dataprep(
    *,
    paths: CViTPaths,
    dataset_id: int = pth.DEFAULT_DATASET_ID,
    dataset_name: str = cfg.DEFAULT_DATASET_NAME,
    images_dir: Path | None = None,
    labels_dir: Path | None = None,
    channel_names: Sequence[str] | None = None,
    labels: dict[str, int] | None = None,
    num_folds: int = cfg.DEFAULT_NUM_FOLDS,
    seed: int = cfg.DEFAULT_FOLD_SEED,
    group_regex: str | None = None,
    allow_empty: bool = False,
    overwrite: bool = False,
    workers: int = 4,
    skip_dataset: bool = False,
    corpus_sources: Sequence[str] = (),
    corpus_from_train: bool = False,
    corpus_id: int = pth.DEFAULT_CORPUS_ID,
    corpus_harmonize: bool = False,
    backend: str = "cpu",
) -> dict[str, Any]:
    """Build (and validate) the nnU-Net raw dataset and optional nnssl corpus.

    Parameters
    ----------
    images_dir, labels_dir
        Default: ``imagesTr``/``labelsTr`` (or ``images``/``labels``) under ``data_root``.
    channel_names
        One name per channel (``dataset.json`` ``channel_names``). nnU-Net normalises a channel
        named ``CT`` with dataset-wide HU statistics and anything else with per-case z-scores.
    labels
        ``{name: value}`` including ``background: 0``. Default: every value found, named
        ``label_<v>``.
    skip_dataset
        Only build the corpus.

    Returns
    -------
    dict
        Provenance (also written to ``<results_root>/stage0_dataprep/<dataset>/cvit_stage0.json``).
    """
    dataset = pth.dataset_folder_name(dataset_id, dataset_name)
    provenance: dict[str, Any] = {
        "stage": "stage0", "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset, "dataset_id": int(dataset_id),
    }
    raw_dir = paths.nnunet_raw / dataset
    if not skip_dataset:
        img_dir, lab_dir = (images_dir, labels_dir) if images_dir else default_dirs(paths.data_root)
        lab_dir = labels_dir or lab_dir
        cases = discover_cases(Path(img_dir), Path(lab_dir))
        n_channels = len(cases[0].images)
        names = list(channel_names or [f"channel{c}" for c in range(n_channels)])
        if len(names) != n_channels:
            raise ValueError(f"{len(names)} channel name(s) given for {n_channels} channel(s).")
        declared = set(labels.values()) if labels else None
        if labels and labels.get("background", 0) != 0:
            raise ValueError("labels must map 'background' to 0.")
        log.info("stage0 | %s: %d case(s), %d channel(s) %s from %s", dataset, len(cases),
                 n_channels, names, img_dir)

        # ---- 1. Validate every case (parallel, fail on the first bad one) ----------------
        records = map_in_thread_pool(
            lambda c: validate_case(c, declared=declared, allow_empty=allow_empty),
            cases, max_workers=int(workers),
        )

        # ---- 2. Write the nnU-Net raw dataset ------------------------------------------
        if overwrite and raw_dir.exists():
            shutil.rmtree(raw_dir)
        (raw_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (raw_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
        map_in_thread_pool(lambda cr: _write_case(cr[0], raw_dir, cr[1], overwrite),
                           list(zip(cases, records)), max_workers=int(workers))
        if labels is None:
            found = sorted({v for r in records for v in r["labels_present"]} | {0})
            labels = {"background": 0, **{f"label_{v}": v for v in found if v != 0}}
        dataset_json = {
            "channel_names": {str(i): n for i, n in enumerate(names)},
            "labels": dict(sorted(labels.items(), key=lambda kv: kv[1])),
            "numTraining": len(cases),
            "file_ending": ".nii.gz",
            "name": dataset,
            "description": "Written by nvitk-cvit stage0.",
        }
        (raw_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")

        # ---- 3. Folds (into nnUNet_preprocessed, where nnU-Net reads them) ---------------
        splits = grouped_splits([c.case_id for c in cases], num_folds=num_folds, seed=seed,
                                group_regex=group_regex)
        pre_dir = paths.nnunet_preprocessed / dataset
        pre_dir.mkdir(parents=True, exist_ok=True)
        for target in (pre_dir / "splits_final.json", raw_dir / "splits_final.json"):
            target.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
        spacings = [r.get("spacing") for r in records if r.get("spacing")]
        provenance.update({
            "raw_dir": str(raw_dir), "images_dir": str(img_dir), "labels_dir": str(lab_dir),
            "num_cases": len(cases), "channel_names": names, "labels": dataset_json["labels"],
            "num_folds": num_folds, "seed": seed, "group_regex": group_regex,
            "fold_sizes": [len(s["val"]) for s in splits],
            "spacing_min": [min(s[a] for s in spacings) for a in range(len(spacings[0]))] if spacings else None,
            "spacing_max": [max(s[a] for s in spacings) for a in range(len(spacings[0]))] if spacings else None,
            "cases": records,
        })
        log.ok(f"stage0: {len(cases)} validated case(s) -> {raw_dir}")

    # ---- 4. Optional nnssl corpus ------------------------------------------------------
    if corpus_sources or corpus_from_train:
        provenance["corpus"] = build_corpus(
            paths, corpus_id=corpus_id, dataset_name=dataset_name, sources=corpus_sources,
            from_train=(raw_dir / "imagesTr") if corpus_from_train else None,
            harmonize=corpus_harmonize, workers=workers, overwrite=overwrite,
        )

    marker_dir = paths.results_root / pth.STAGE0_DATAPREP_DIR / dataset
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / STAGE0_MARKER).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def parse_labels(text: str | None) -> dict[str, int] | None:
    """``"background=0,vessel=1"`` (or a JSON object / file) → ``{name: value}``."""
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
    text = text.strip()
    if text.startswith("{"):
        return {str(k): int(v) for k, v in json.loads(text).items()}
    out: dict[str, int] = {}
    for item in text.split(","):
        name, _, value = item.partition("=")
        if not value:
            raise click.BadParameter(f"Label item {item!r} is not name=value.")
        out[name.strip()] = int(value)
    return out


# ---------------------------------------------------------------------------
# SGE
# ---------------------------------------------------------------------------


def _worker_argv(paths: CViTPaths, **o: Any) -> list[str]:
    inside = container_layout()
    argv = [*python_module_argv("nvitk.pipes.cvit.stage0_dataprep"),
            *sge_backend_cli_args(o.get("backend", "cpu")), *root_args(inside),
            "--dataset-id", str(o.get("dataset_id", pth.DEFAULT_DATASET_ID)),
            "--dataset-name", quote_path(o.get("dataset_name", cfg.DEFAULT_DATASET_NAME)),
            "--num-folds", str(o.get("num_folds", cfg.DEFAULT_NUM_FOLDS)),
            "--seed", str(o.get("seed", cfg.DEFAULT_FOLD_SEED)),
            "--workers", str(o.get("workers", 4))]
    for key, flag in (("images_dir", "--images-dir"), ("labels_dir", "--labels-dir")):
        if o.get(key):
            argv += [flag, quote_path(to_container_path(paths, o[key]) or o[key])]
    if o.get("channel_names"):
        argv += ["--channel-names", quote_path(",".join(o["channel_names"]))]
    if o.get("labels"):
        argv += ["--labels", quote_path(json.dumps(o["labels"]))]
    if o.get("group_regex"):
        argv += ["--group-regex", quote_path(o["group_regex"])]
    for flag, key in (("--allow-empty", "allow_empty"), ("--overwrite", "overwrite"),
                      ("--skip-dataset", "skip_dataset"), ("--corpus-from-train", "corpus_from_train"),
                      ("--corpus-harmonize", "corpus_harmonize")):
        if o.get(key):
            argv.append(flag)
    for spec in o.get("corpus_sources") or ():
        argv += ["--corpus-source", quote_path(spec)]
    argv += ["--corpus-id", str(o.get("corpus_id", pth.DEFAULT_CORPUS_ID))]
    return argv


def _data_paths(paths: CViTPaths, **o: Any) -> list[Path]:
    out = [Path(o[k]) for k in ("images_dir", "labels_dir") if o.get(k)]
    for spec in o.get("corpus_sources") or ():
        root = spec.partition("=")[2].partition(":")[0]
        if root.startswith("/"):
            out.append(Path(root))
    return out


def submit_sge(*, paths: CViTPaths, container: Path, src_dir: Path | None = None,
               hold_jid=None, dry_run: bool = False, emit: TextIO | None = None, **o: Any) -> str:
    """Emit or submit the stage 0 job (CPU)."""
    return submit_stage_job(
        "stage0", _worker_argv(paths, **o), paths=paths, container=container, src_dir=src_dir,
        backend=o.get("backend", "cpu"), request_gpu=False, data_paths=_data_paths(paths, **o),
        job_suffix=str(o.get("dataset_id", "")), hold_jid=hold_jid, dry_run=dry_run, emit=emit,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("cvit-stage0-dataprep")
@config_dir_click_option()
@backend_click_option(default="cpu")
@root_options
@click.option("--dataset-id", type=int, default=pth.DEFAULT_DATASET_ID, show_default=True)
@click.option("--dataset-name", type=str, default=cfg.DEFAULT_DATASET_NAME, show_default=True)
@click.option("--images-dir", type=click.Path(path_type=Path), default=None)
@click.option("--labels-dir", type=click.Path(path_type=Path), default=None)
@click.option("--channel-names", type=str, default=None, help="Comma list, e.g. 'CT' or 'T1,T2'.")
@click.option("--labels", "labels_text", type=str, default=None,
              help="'background=0,vessel=1', a JSON object, or a JSON file.")
@click.option("--num-folds", type=int, default=cfg.DEFAULT_NUM_FOLDS, show_default=True)
@click.option("--seed", type=int, default=cfg.DEFAULT_FOLD_SEED, show_default=True)
@click.option("--group-regex", type=str, default=None,
              help="Regex over case ids whose 'group' (or first) group is the patient key.")
@click.option("--allow-empty", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--workers", type=int, default=4, show_default=True)
@click.option("--skip-dataset", is_flag=True, default=False, help="Only build the corpus.")
@click.option("--corpus-source", "corpus_sources", multiple=True,
              help="Unlabelled cohort 'name:modality=/path[:glob]' (repeatable).")
@click.option("--corpus-from-train", is_flag=True, default=False,
              help="Add the training images (channel 0) to the corpus.")
@click.option("--corpus-id", type=int, default=pth.DEFAULT_CORPUS_ID, show_default=True)
@click.option("--corpus-harmonize", is_flag=True, default=False,
              help="Window/robust-scale corpus volumes to [0,1] on the way in.")
def main(backend: str, labels_text: str | None, channel_names: str | None, **options: Any) -> None:
    """Validate labelled data into an nnU-Net raw dataset (and optionally an nnssl corpus)."""
    paths = paths_from_options(pop_roots(options))
    run_dataprep(
        paths=paths, labels=parse_labels(labels_text), backend=backend,
        channel_names=[c.strip() for c in channel_names.split(",")] if channel_names else None,
        **options,
    )


if __name__ == "__main__":
    main()
