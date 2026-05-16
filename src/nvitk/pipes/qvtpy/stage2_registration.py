"""qvtpy stage 2: rigid eICAB TOF (resampled) → 4D flow reference registration (FSL FLIRT via NiPype)."""

from __future__ import annotations

import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TextIO

import click

import nvitk
from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    submit_stage,
)
from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.stage1_eicab import find_tof_resampled_volume
from nvitk.registration.fsl.flirt import flirt_register_rigid

log = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

ReferenceKind = Literal["angio", "cd"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _flow_dir(nifti_root: Path, subject: str) -> Path:
    return nifti_root / subject / "4DFlow"


def _reference_volume(flow_dir: Path, kind: ReferenceKind) -> Path:
    if kind == "angio":
        for name in ("Angiography_3D.nii.gz", "Angiography_3D.nii"):
            p = flow_dir / name
            if p.is_file():
                return p
        raise FileNotFoundError(f"No Angiography_3D under {flow_dir}")
    for name in ("ComplexDifference_3D.nii.gz", "ComplexDifference_3D.nii"):
        p = flow_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No ComplexDifference_3D under {flow_dir}")


def _stage2_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE2_REGISTRATION_DIR


def _done_marker(out_dir: Path) -> Path:
    return out_dir / "registration_meta.json"


# ---------------------------------------------------------------------------
# Stage 2: FLIRT rigid registration
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
    reference: ReferenceKind = "angio",
    dof: int = 6,
    cost: str = "normmi",
    eicab_subdir: str | None = None,
) -> Path:
    """Run FLIRT locally. Writes ``registration_meta.json`` and FLIRT outputs."""
    sub = (eicab_subdir or cfg.STAGE1_EICAB_DIR).strip() or "eicab"
    eicab_dir = output_root / subject / sub
    moving = find_tof_resampled_volume(eicab_dir)
    if moving is None:
        raise FileNotFoundError(
            f"No eICAB TOF_resampled NIfTI under {eicab_dir} (run stage 1 first; expected "
            f"TOF_resampled.nii.gz or similar)."
        )
    flow_dir = _flow_dir(nifti_root, subject)
    fixed = _reference_volume(flow_dir, reference)
    out_dir = _stage2_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and _done_marker(out_dir).is_file():
        log.info(f"[{subject}] stage2 registration: skip existing -> {out_dir}")
        return out_dir

    log.info(f"qvtpy stage2 FLIRT | subject={subject}")
    log.info(f"  moving (eICAB TOF resampled): {moving}")
    log.info(f"  fixed ({reference}): {fixed}")

    res = flirt_register_rigid(
        moving,
        fixed,
        out_dir,
        dof=dof,
        cost=cost,
        warped_name="TOF_warped_to_4dflow_ref.nii.gz",
        matrix_name="tof_to_4dflow.mat",
    )
    meta: dict[str, Any] = {
        "subject": subject,
        "moving": str(moving),
        "moving_kind": "eicab_tof_resampled",
        "eicab_dir": str(eicab_dir),
        "fixed": str(fixed),
        "reference_kind": reference,
        "matrix": str(res.matrix_path),
        "warped": str(res.warped_path) if res.warped_path else None,
        "dof": dof,
        "cost": cost,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    _done_marker(out_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def submit_subject_sge(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    reference: ReferenceKind = "angio",
    dof: int = 6,
    cost: str = "normmi",
    eicab_subdir: str | None = None,
    hold_jid: str | None = None,
    emit: TextIO | None = None,
) -> str:
    """Emit or submit one stage2 SGE block (FLIRT inside Singularity)."""
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    script = f"{binds.src}nvitk/pipes/qvtpy/stage2_registration.py"
    parts: list[str] = [
        "python",
        shlex.quote(script),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
        "--reference",
        reference,
        "--dof",
        str(int(dof)),
        "--cost",
        shlex.quote(cost),
    ]
    if skip_existing:
        parts.append("--skip-existing")
    if eicab_subdir:
        parts.extend(["--eicab-subdir", shlex.quote(eicab_subdir.strip())])
    python_cmd = " ".join(parts)

    paths = ClusterPaths(
        src=src_p,
        container=container,
        models=None,
        data_root=nifti_root,
        output_root=output_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    spec = StageSpec(
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage2_{subject}",
        python_cmd=python_cmd,
        resources=SgeResources(
            project=cfg.SGE_PROJECT,
            account=cfg.SGE_ACCOUNT,
            ngpu=0,
            h_vmem=cfg.SGE_H_VMEM,
            queue=cfg.SGE_QUEUE,
        ),
        binds=binds,
        use_nv=False,
        extra_env={"PYTHONPATH": str(binds.src)},
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage2-registration")
@click.option("--subject", required=True)
@click.option("--nifti-root", type=click.Path(path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--reference", type=click.Choice(["angio", "cd"]), default="angio", show_default=True)
@click.option("--dof", type=int, default=6, show_default=True)
@click.option("--cost", type=str, default="normmi", show_default=True)
@click.option(
    "--eicab-subdir",
    type=str,
    default=None,
    help="Subject-relative eICAB output directory under --output-root (default: config STAGE1_EICAB_DIR).",
)
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    reference: ReferenceKind,
    dof: int,
    cost: str,
    eicab_subdir: str | None,
) -> None:
    Logger()
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        reference=reference,
        dof=dof,
        cost=cost,
        eicab_subdir=eicab_subdir,
    )


__all__ = ["main", "run_subject", "submit_subject_sge"]
