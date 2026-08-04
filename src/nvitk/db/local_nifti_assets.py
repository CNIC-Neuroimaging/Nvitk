"""Register reorganized local NIfTI files (``subject/TOF``, ``subject/4DFlow``) into ``assets``.

Expected layout (same as QVT+/eICAB pipelines)::

    {nifti_root}/{subject_uid}/TOF/*.nii[.gz]
    {nifti_root}/{subject_uid}/4DFlow/**/*.nii[.gz]

Under ``4DFlow/``, supported patterns for ``asset_slot`` (see :func:`_classify_4dflow_path`) include:

- Directional encodings: ``{subject}/4DFlow/AP|FH|RL/<name>_m.nii`` → ``flow_ap_m``, ``flow_fh_ph``, etc.
- Root derivatives: ``Angiography_3D.nii`` / ``Angiography_4D.nii`` → ``flow_angiography_3d`` / ``flow_angiography_4d``,
  and the same for ``ComplexDifference_*`` and ``VelocityMagnitude_*``.

TOF files use ``asset_slot`` ``tof`` when there is a single NIfTI per subject, else ``tof_<stem>``.

Do not store secrets in the dataset; this module only records file paths.

See also :mod:`nvitk.db.xnat_config` for XNAT credentials.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import click
except Exception:
    click = None

from nvitk.core.exceptions import BackendUnavailableError

from .repo import DataRepo
from .storage import utc_now_iso


def _is_nifti_path(path: Path) -> bool:
    """True if *path* has a ``.nii`` or ``.nii.gz`` extension."""
    name = path.name.lower()
    return name.endswith(".nii.gz") or name.endswith(".nii")


def _nifti_stem(filename: str) -> str:
    """Filename without ``.nii`` / ``.nii.gz`` suffix."""
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return filename[:-7]
    if lower.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def _normalize_slot_token(text: str) -> str:
    """Lower-case *text* and collapse runs of non-alphanumeric characters into single underscores, for
    building ``asset_slot`` names."""
    return re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


def _classify_4dflow_path(rel_posix: str, filename: str) -> tuple[str | None, str | None]:
    """Return ``(asset_slot, flow4d_kind)`` for a path under ``.../4DFlow/``.

    ``flow4d_kind`` is ``directional``, ``derived_global``, or ``None`` if unclassified.
    ``asset_slot`` is ``None`` when the layout does not match a known pattern.
    """
    parts = rel_posix.replace("\\", "/").split("/")
    try:
        i = parts.index("4DFlow")
    except ValueError:
        return None, None

    tail = parts[i + 1 :]
    if not tail:
        return None, None

    stem_l = _nifti_stem(filename).lower()
    direction_dirs = {"ap", "fh", "rl"}

    # .../4DFlow/AP/file.nii (exactly one folder after 4DFlow before filename)
    if len(tail) >= 2:
        dir_seg = tail[0].lower()
        if dir_seg in direction_dirs:
            if stem_l.endswith("_ph"):
                comp = "ph"
            elif stem_l.endswith("_m"):
                comp = "m"
            else:
                return None, "directional"
            slot = f"flow_{dir_seg}_{comp}"
            return slot, "directional"

    # Root of 4DFlow: .../4DFlow/Angiography_3D.nii → single segment in tail
    if len(tail) == 1:
        key = _normalize_slot_token(_nifti_stem(filename))
        prefixes = (
            ("angiography_3d", "flow_angiography_3d"),
            ("angiography_4d", "flow_angiography_4d"),
            ("complexdifference_3d", "flow_complex_difference_3d"),
            ("complexdifference_4d", "flow_complex_difference_4d"),
            ("velocitymagnitude_3d", "flow_velocity_magnitude_3d"),
            ("velocitymagnitude_4d", "flow_velocity_magnitude_4d"),
        )
        for prefix, slot in prefixes:
            if key == prefix:
                return slot, "derived_global"

    return None, None


def _tof_slot_name(stem: str, *, multi: bool) -> str:
    """``"tof"`` when there's a single TOF NIfTI per subject, else ``"tof_<normalized stem>"`` to keep
    multiple TOF files distinct."""
    if multi:
        return f"tof_{_normalize_slot_token(stem)}"
    return "tof"


def _iter_niftis_under(root: Path) -> Iterable[Path]:
    """Yield every NIfTI file found anywhere under *root*, sorted; nothing if *root* isn't a directory."""
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and _is_nifti_path(p):
            yield p


def register_nifti_tree(
    nifti_root: str | Path,
    *,
    source: str = "local_nifti",
    pipeline_id: str | None = None,
    pipeline_name: str | None = None,
    source_batch_id: str | None = None,
) -> pd.DataFrame:
    """Build ``assets`` rows for TOF and 4DFlow NIfTI trees under ``nifti_root``."""
    root = Path(nifti_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NIfTI root is not a directory: {root}")

    batch = source_batch_id or f"local_nifti_{utc_now_iso().replace(':', '').replace('-', '')}"
    now = utc_now_iso()
    rows: list[dict[str, Any]] = []

    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subject_uid = subject_dir.name

        tof_dir = subject_dir / "TOF"
        if tof_dir.is_dir():
            tof_files = [fp for fp in sorted(tof_dir.iterdir()) if fp.is_file() and _is_nifti_path(fp)]
            multi_tof = len(tof_files) > 1
            for fp in tof_files:
                rel = fp.relative_to(root).as_posix()
                stem = _nifti_stem(fp.name)
                slot = _tof_slot_name(stem, multi=multi_tof)
                meta_obj: dict[str, Any] = {
                    "relative_path": rel,
                    "basename": fp.name,
                    "folder": "TOF",
                    "asset_slot": slot,
                }
                rows.append(
                    {
                        "asset_uid": f"local:{subject_uid}:tof:{rel}",
                        "subject_uid": subject_uid,
                        "session_uid": pd.NA,
                        "modality": "tof",
                        "asset_type": "nifti",
                        "asset_path": str(fp.resolve()),
                        "resource_label": "TOF",
                        "source": source,
                        "pipeline_name": pipeline_name if pipeline_name is not None else pd.NA,
                        "pipeline_id": pipeline_id if pipeline_id is not None else pd.NA,
                        "exists_locally": fp.exists(),
                        "asset_slot": slot,
                        "metadata_json": json.dumps(meta_obj),
                        "source_batch_id": batch,
                        "updated_at": now,
                    }
                )

        flow_dir = subject_dir / "4DFlow"
        if flow_dir.is_dir():
            for fp in _iter_niftis_under(flow_dir):
                rel = fp.relative_to(root).as_posix()
                slot, flow_kind = _classify_4dflow_path(rel, fp.name)
                parts = rel.split("/")
                try:
                    j = parts.index("4DFlow")
                    seg_after = parts[j + 1] if j + 1 < len(parts) else ""
                except ValueError:
                    seg_after = ""
                if seg_after.upper() in {"AP", "FH", "RL"}:
                    res_label = f"4DFlow/{seg_after.upper()}"
                else:
                    res_label = "4DFlow"
                meta_obj = {
                    "relative_path": rel,
                    "basename": fp.name,
                    "folder": "4DFlow",
                    "asset_slot": slot,
                    "flow4d_kind": flow_kind,
                }
                rows.append(
                    {
                        "asset_uid": f"local:{subject_uid}:4dflow:{rel}",
                        "subject_uid": subject_uid,
                        "session_uid": pd.NA,
                        "modality": "4dflow",
                        "asset_type": "nifti",
                        "asset_path": str(fp.resolve()),
                        "resource_label": res_label,
                        "source": source,
                        "pipeline_name": pipeline_name if pipeline_name is not None else pd.NA,
                        "pipeline_id": pipeline_id if pipeline_id is not None else pd.NA,
                        "exists_locally": fp.exists(),
                        "asset_slot": slot if slot is not None else pd.NA,
                        "metadata_json": json.dumps(meta_obj),
                        "source_batch_id": batch,
                        "updated_at": now,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df


def upsert_nifti_assets(
    repo: DataRepo,
    nifti_root: str | Path,
    *,
    source: str = "local_nifti",
    pipeline_id: str | None = None,
    pipeline_name: str | None = None,
    source_batch_id: str | None = None,
    dry_run: bool = False,
    build_sqlite_index: bool = False,
) -> pd.DataFrame:
    """Compute asset rows and upsert into ``repo`` unless ``dry_run``."""
    df = register_nifti_tree(
        nifti_root,
        source=source,
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_name,
        source_batch_id=source_batch_id,
    )
    if dry_run or df.empty:
        return df
    repo.upsert_table(
        "assets",
        df,
        provenance={"source": "local_nifti", "nifti_root": str(Path(nifti_root).expanduser().resolve())},
        build_sqlite_index=build_sqlite_index,
    )
    return df


def _cli_decorator(*args: Any, **kwargs: Any):
    """No-op stand-in for ``click.command``/``click.option`` when ``click`` isn't installed."""

    def decorator(func: Any) -> Any:
        """Return *func* unchanged."""
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator


@_click_command()
@_click_option(
    "--dataset-root",
    type=click.Path(path_type=Path) if click is not None else None,
    default=Path("dataset"),
    show_default=True,
)
@_click_option(
    "--nifti-root",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    required=True,
    help="Root containing {subject}/TOF and {subject}/4DFlow (reorganized NIfTI tree).",
)
@_click_option("--source", type=str, default="local_nifti", show_default=True)
@_click_option("--pipeline-id", type=str, default=None)
@_click_option("--pipeline-name", type=str, default=None)
@_click_option("--source-batch-id", type=str, default=None)
@_click_option("--dry-run", is_flag=True, help="Print row count only; do not write Parquet.")
@_click_option("--build-sqlite-index", is_flag=True)
def main(
    dataset_root: Path,
    nifti_root: Path,
    source: str,
    pipeline_id: str | None,
    pipeline_name: str | None,
    source_batch_id: str | None,
    dry_run: bool,
    build_sqlite_index: bool,
) -> None:
    """CLI entry point: scaffold/open the dataset at ``dataset_root`` and register the reorganized
    NIfTI tree at ``nifti_root`` into the ``assets`` table."""
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')

    repo = DataRepo(dataset_root, auto_scaffold=True)
    df = upsert_nifti_assets(
        repo,
        nifti_root,
        source=source,
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_name,
        source_batch_id=source_batch_id,
        dry_run=dry_run,
        build_sqlite_index=build_sqlite_index,
    )
    click.echo(f"Registered {len(df)} NIfTI asset row(s).")


if __name__ == "__main__":
    main()
