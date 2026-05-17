"""Path resolution for black-blood (no qvtpy / pesa_fat imports)."""

from __future__ import annotations

from pathlib import Path

from nvitk.pipes.pesa_brain.black_blood import config as cfg
from nvitk.pipes.pesa_brain.black_blood.util.eicab_masks import (
    EicabMaskKind,
    EicabMaskResolution,
    resolve_eicab_mask,
)

# Legacy stage1 output names (pre vwi_bb rename).
_LEGACY_WARPED = "WVI_warped_to_tof.nii.gz"
_LEGACY_MATRIX = "wvi_to_tof.mat"


def _stem_without_suffix(p: Path) -> str:
    if p.name.endswith(".nii.gz"):
        return p.name[: -len(".nii.gz")]
    if p.name.lower().endswith(".nii"):
        return p.name[: -len(".nii")]
    return p.stem


def find_tof_resampled_volume(eicab_dir: Path) -> Path | None:
    """Return eICAB ``TOF_resampled`` NIfTI under *eicab_dir*, or None."""
    if not eicab_dir.is_dir():
        return None
    for name in ("TOF_resampled.nii.gz", "TOF_resampled.nii"):
        p = eicab_dir / name
        if p.is_file():
            return p
    hits: list[Path] = []
    for p in sorted(eicab_dir.rglob("*")):
        if not p.is_file():
            continue
        if not (p.suffix == ".nii" or p.name.endswith(".nii.gz")):
            continue
        stem = _stem_without_suffix(p).lower()
        if stem.endswith("_resampled") and "tof" in stem:
            hits.append(p)
    return hits[0] if hits else None


def eicab_mask_resolution(
    eicab_results_root: Path,
    subject: str,
    *,
    eicab_mask: EicabMaskKind = "cw",
    eicab_subdir: str | None = None,
) -> EicabMaskResolution:
    """Resolve CW or WB eICAB multilabel under the subject eICAB output dir."""
    eicab_dir = eicab_subject_dir(eicab_results_root, subject, eicab_subdir=eicab_subdir)
    return resolve_eicab_mask(eicab_dir, preference=eicab_mask)


def eicab_mask_path(
    eicab_results_root: Path,
    subject: str,
    *,
    eicab_mask: EicabMaskKind = "cw",
    eicab_subdir: str | None = None,
) -> Path:
    """Path to requested eICAB CW/WB mask (with fallback logging)."""
    return eicab_mask_resolution(
        eicab_results_root,
        subject,
        eicab_mask=eicab_mask,
        eicab_subdir=eicab_subdir,
    ).path


def eicab_cw_mask_path(
    eicab_results_root: Path,
    subject: str,
    *,
    eicab_subdir: str | None = None,
) -> Path:
    """Deprecated: use :func:`eicab_mask_path` with ``eicab_mask='cw'``."""
    return eicab_mask_path(
        eicab_results_root, subject, eicab_mask="cw", eicab_subdir=eicab_subdir
    )


def require_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(
            f"{name} is not set. Pass --{name.replace('_', '-')} on the CLI or set "
            f"nvitk.pipes.pesa_brain.black_blood.config."
        )
    p = Path(value)
    if not p.exists():
        raise FileNotFoundError(f"{name} does not exist: {p}")
    return p


def resolve_eicab_results_root(eicab_results_root: Path | None) -> Path:
    root = eicab_results_root or cfg.DEFAULT_EICAB_RESULTS_ROOT or cfg.DEFAULT_QVTPY_RESULTS_ROOT
    return require_path(root, "eicab_results_root")


def require_vwi_bb_rel_path(vwi_bb_rel: str | None) -> str:
    rel = (vwi_bb_rel or cfg.VWI_BB_REL_PATH or "").strip()
    if not rel:
        raise ValueError(
            "VWI_BB relative path is not set. Pass --vwi-bb-rel-path or set "
            "nvitk.pipes.pesa_brain.black_blood.config.VWI_BB_REL_PATH."
        )
    return rel


def vwi_bb_path(
    nifti_root: Path,
    subject: str,
    *,
    vwi_bb_rel: str | None = None,
) -> Path:
    rel = require_vwi_bb_rel_path(vwi_bb_rel)
    p = nifti_root / subject / rel
    if not p.is_file():
        raise FileNotFoundError(f"vwi_bb not found for {subject}: {p}")
    return p


def wvi_path(
    nifti_root: Path,
    subject: str,
    *,
    wvi_rel: str | None = None,
) -> Path:
    """Deprecated alias for :func:`vwi_bb_path`."""
    return vwi_bb_path(nifti_root, subject, vwi_bb_rel=wvi_rel)


def require_wvi_rel_path(wvi_rel: str | None) -> str:
    """Deprecated alias for :func:`require_vwi_bb_rel_path`."""
    return require_vwi_bb_rel_path(wvi_rel)


def qvtpy_tof_path(qvtpy_nifti_root: Path, subject: str) -> Path:
    """qvtpy stage0 TOF magnitude (validation / QC)."""
    for name in ("TOF.nii.gz", "TOF.nii"):
        p = qvtpy_nifti_root / subject / "TOF" / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No qvtpy TOF under {qvtpy_nifti_root / subject / 'TOF'}"
    )


def eicab_subject_dir(
    eicab_results_root: Path,
    subject: str,
    *,
    eicab_subdir: str | None = None,
) -> Path:
    sub = (eicab_subdir or cfg.QVTPY_EICAB_SUBDIR or cfg.EICAB_SUBDIR).strip() or "eicab"
    return eicab_results_root / subject / sub


def tof_resampled_path(
    eicab_results_root: Path,
    subject: str,
    *,
    eicab_subdir: str | None = None,
) -> Path:
    eicab_dir = eicab_subject_dir(eicab_results_root, subject, eicab_subdir=eicab_subdir)
    p = find_tof_resampled_volume(eicab_dir)
    if p is None:
        raise FileNotFoundError(
            f"No eICAB TOF_resampled under {eicab_dir} (expected TOF_resampled.nii.gz or similar)."
        )
    return p


def black_blood_root(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.PIPELINE_SUBDIR / cfg.BLACK_BLOOD_SUBDIR


def stage1_dir(output_root: Path, subject: str) -> Path:
    return black_blood_root(output_root, subject) / cfg.STAGE1_REG_DIR


def stage2_dir(output_root: Path, subject: str) -> Path:
    return black_blood_root(output_root, subject) / cfg.STAGE2_SEG_DIR


def registration_meta_path(output_root: Path, subject: str) -> Path:
    return stage1_dir(output_root, subject) / "registration_meta.json"


def vwi_bb_warped_path(output_root: Path, subject: str) -> Path:
    stage1 = stage1_dir(output_root, subject)
    p = stage1 / "vwi_bb_warped_to_tof.nii.gz"
    if p.is_file():
        return p
    legacy = stage1 / _LEGACY_WARPED
    if legacy.is_file():
        return legacy
    return p


def registration_matrix_path(output_root: Path, subject: str) -> Path:
    stage1 = stage1_dir(output_root, subject)
    p = stage1 / "vwi_bb_to_tof.mat"
    if p.is_file():
        return p
    legacy = stage1 / _LEGACY_MATRIX
    if legacy.is_file():
        return legacy
    return p


def wvi_warped_path(output_root: Path, subject: str) -> Path:
    """Deprecated alias for :func:`vwi_bb_warped_path`."""
    return vwi_bb_warped_path(output_root, subject)
