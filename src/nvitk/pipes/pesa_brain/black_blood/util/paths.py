"""Path resolution for black-blood (no qvtpy / pesa_fat imports)."""

from __future__ import annotations

from pathlib import Path

from nvitk.pipes.pesa_brain.black_blood import config as cfg

_CW_PATTERNS = ("*_eICAB_CW.nii.gz", "*_eICAB_CW.nii")


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


def find_eicab_cw_mask(eicab_dir: Path) -> Path | None:
    """Return first ``*_eICAB_CW`` multilabel NIfTI under *eicab_dir*, or None."""
    if not eicab_dir.is_dir():
        return None
    for pat in _CW_PATTERNS:
        hits = sorted(eicab_dir.glob(pat))
        if hits:
            return hits[0]
    for pat in _CW_PATTERNS:
        hits = sorted(eicab_dir.rglob(pat))
        if hits:
            return hits[0]
    return None


def require_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(
            f"{name} is not set. Pass --{name.replace('_', '-')} on the CLI or set "
            f"nvitk.pipes.pesa_brain.black_blood.config.{name.upper()}."
        )
    p = Path(value)
    if not p.exists():
        raise FileNotFoundError(f"{name} does not exist: {p}")
    return p


def require_wvi_rel_path(wvi_rel: str | None) -> str:
    rel = (wvi_rel or cfg.WVI_REL_PATH or "").strip()
    if not rel:
        raise ValueError(
            "WVI relative path is not set. Pass --wvi-rel-path or set "
            "black_blood.config.WVI_REL_PATH (e.g. BlackBlood/WVI.nii.gz)."
        )
    return rel


def wvi_path(nifti_root: Path, subject: str, *, wvi_rel: str | None = None) -> Path:
    rel = require_wvi_rel_path(wvi_rel)
    p = nifti_root / subject / rel
    if not p.is_file():
        alt = p.with_suffix("") if p.suffix == ".gz" else None
        if alt is not None and alt.is_file():
            return alt
        raise FileNotFoundError(f"WVI not found for {subject}: {p}")
    return p


def eicab_subject_dir(
    eicab_results_root: Path,
    subject: str,
    *,
    eicab_subdir: str | None = None,
) -> Path:
    sub = (eicab_subdir or cfg.EICAB_SUBDIR).strip() or "eicab"
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


def eicab_cw_mask_path(
    eicab_results_root: Path,
    subject: str,
    *,
    eicab_subdir: str | None = None,
) -> Path:
    eicab_dir = eicab_subject_dir(eicab_results_root, subject, eicab_subdir=eicab_subdir)
    p = find_eicab_cw_mask(eicab_dir)
    if p is None:
        raise FileNotFoundError(f"No *_eICAB_CW multilabel under {eicab_dir}.")
    return p


def black_blood_root(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.PIPELINE_SUBDIR / cfg.BLACK_BLOOD_SUBDIR


def stage1_dir(output_root: Path, subject: str) -> Path:
    return black_blood_root(output_root, subject) / cfg.STAGE1_REG_DIR


def stage2_dir(output_root: Path, subject: str) -> Path:
    return black_blood_root(output_root, subject) / cfg.STAGE2_SEG_DIR


def registration_meta_path(output_root: Path, subject: str) -> Path:
    return stage1_dir(output_root, subject) / "registration_meta.json"


def wvi_warped_path(output_root: Path, subject: str) -> Path:
    return stage1_dir(output_root, subject) / "WVI_warped_to_tof.nii.gz"


def registration_matrix_path(output_root: Path, subject: str) -> Path:
    return stage1_dir(output_root, subject) / "wvi_to_tof.mat"
