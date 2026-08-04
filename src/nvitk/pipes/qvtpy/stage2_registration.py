"""qvtpy stage 2: rigid eICAB TOF (resampled) → 4D-flow reference (FSL FLIRT).

**Inputs**

- eICAB ``TOF_resampled`` (stage 1) and fixed 4D-flow ``Angiography_3D`` or ``ComplexDifference_3D``.

**Outputs**

- ``<output>/<subject>/qvtpy/stage2_registration/`` — FLIRT matrix, warped TOF, ``registration_meta.json``.
"""

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
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.pipes.qvtpy.util.io.sge_backend import (
    sge_backend_cli_args,
    sge_qvtpy_stage_resources,
    sge_stage_extra_env,
    sge_stage_use_nv,
)
from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.util.eicab.eicab_masks import find_tof_resampled_volume
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
    """Repo ``src/`` directory inferred from the installed ``nvitk`` package location."""
    return Path(nvitk.__file__).resolve().parent.parent


def _flow_dir(nifti_root: Path, subject: str) -> Path:
    """*subject*'s 4D-flow NIfTI directory under *nifti_root*."""
    return nifti_root / subject / "4DFlow"


def _reference_volume(flow_dir: Path, kind: ReferenceKind) -> Path:
    """Fixed registration reference volume of *kind* (``"angio"`` or ``"cd"``) under *flow_dir*."""
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
    """Stage 2 output directory for *subject* under *output_root*."""
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE2_REGISTRATION_DIR


def _done_marker(out_dir: Path) -> Path:
    """Path to the stage 2 completion marker (``registration_meta.json``) in *out_dir*."""
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


def _subject_sge_spec(
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
    backend: str = "gpu",
) -> tuple[StageSpec, ClusterPaths]:
    """Build the SGE ``StageSpec``/``ClusterPaths`` pair for one subject's stage 2 registration task."""
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    parts: list[str] = [
        *python_module_argv("nvitk.pipes.qvtpy.stage2_registration"),
        *sge_backend_cli_args(backend),
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
        resources=sge_qvtpy_stage_resources(backend),
        binds=binds,
        use_nv=sge_stage_use_nv(backend),
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    return spec, paths


def build_subject_sge_command(
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
    backend: str = "gpu",
) -> str:
    """Return the host shell command for one stage2 array/SGE task."""
    from nvitk.cluster.sge import build_singularity_command

    spec, paths = _subject_sge_spec(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        reference=reference,
        dof=dof,
        cost=cost,
        eicab_subdir=eicab_subdir,
        backend=backend,
    )
    return build_singularity_command(spec, paths)


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
    backend: str = "gpu",
    emit: TextIO | None = None,
) -> str:
    """Emit or submit one stage2 SGE block (FLIRT inside Singularity)."""
    spec, paths = _subject_sge_spec(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        reference=reference,
        dof=dof,
        cost=cost,
        eicab_subdir=eicab_subdir,
        backend=backend,
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage2-registration")
@backend_click_option()
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
    """CLI entry point (``qvtpy-stage2-registration``): run FLIRT rigid registration for one subject."""
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


__all__ = ["main", "run_subject", "build_subject_sge_command", "submit_subject_sge"]


if __name__ == "__main__":
    main()
