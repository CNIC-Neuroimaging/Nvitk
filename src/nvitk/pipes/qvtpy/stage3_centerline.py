"""qvtpy stage 3: centerlines (eICAB in 4Dflow space + venous proxy from CD)."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, TextIO

import click

import nvitk
from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    submit_stage,
)
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.morphology.centerline import compute_centerlines
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.flow_volume_masks import binary_vessel_segment_cd, venous_search_region
from nvitk.pipes.qvtpy.labels import VENOUS_UNKNOWN_LABEL
from nvitk.registration.fsl.flirt import flirt_apply_rigid

setup(globals())

log = Logger()


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage2_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE2_REGISTRATION_DIR


def _stage3_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _load_stage2_meta(output_root: Path, subject: str) -> dict[str, Any]:
    p = _stage2_dir(output_root, subject) / "registration_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing stage2 meta: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _find_eicab_labels(eicab_dir: Path) -> Path:
    for pat in ("*_eICAB_CW.nii.gz", "*_eICAB_CW.nii", "*_eICAB_WB.nii.gz", "*_eICAB_WB.nii"):
        hits = sorted(eicab_dir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No eICAB CW/WB NIfTI under {eicab_dir}")


def _bwareaopen_bool(mask: np.ndarray, *, min_size: int, connectivity: int = 1) -> np.ndarray:
    """Remove connected foreground components smaller than *min_size* (3D area opening).

    *connectivity=1* selects face-adjacent neighbors in 3D (6-neighborhood).
    """
    from skimage.morphology import remove_small_objects  # type: ignore

    m = to_numpy(mask.astype(bool, copy=False))
    if min_size <= 1:
        return m
    return as_backend_array(remove_small_objects(m, min_size=int(min_size), connectivity=int(connectivity)))


def _rasterize_centerlines_mask(
    shape: tuple[int, int, int],
    arterial: dict[int, Any],
    ven_sk_cpu: np.ndarray,
    *,
    ven_label_id: int,
) -> np.ndarray:
    """Voxel mask: **eICAB vessel_id** on arterial centerline points; *ven_label_id* on venous skeleton."""
    mask = np.zeros(shape, dtype=np.int32)
    for vid, pts in sorted(arterial.items()):
        p = to_numpy(pts)
        for row in p:
            i, j, k = int(round(float(row[0]))), int(round(float(row[1]))), int(round(float(row[2])))
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                mask[i, j, k] = int(vid)
    ven = ven_sk_cpu.astype(bool, copy=False)
    vi, vj, vk = np.nonzero(ven)
    for i, j, k in zip(vi.tolist(), vj.tolist(), vk.tolist()):
        if mask[i, j, k] == 0:
            mask[i, j, k] = int(ven_label_id)
    return mask


def _cd_path(nifti_root: Path, subject: str) -> Path:
    p = nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii.gz"
    if p.is_file():
        return p
    p2 = nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii"
    if p2.is_file():
        return p2
    raise FileNotFoundError(f"Missing ComplexDifference_3D for {subject}")


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None = None,
    skip_existing: bool = False,
) -> Path:
    meta = _load_stage2_meta(output_root, subject)
    mat = Path(meta["matrix"])
    fixed = Path(meta["fixed"])
    subdir = (eicab_subdir or cfg.STAGE1_EICAB_DIR).strip() or "eicab"
    eicab_dir = output_root / subject / subdir
    eicab_labels = _find_eicab_labels(eicab_dir)

    out_dir = _stage3_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "centerline_meta.json"
    if skip_existing and done.is_file():
        log.info(f"[{subject}] stage3 centerline: skip -> {out_dir}")
        return out_dir

    warped_labels = out_dir / "eicab_in_4dflow.nii.gz"
    flirt_apply_rigid(
        eicab_labels,
        fixed,
        mat,
        warped_labels,
        interp="nearestneighbour",
    )

    lab_img = imread(warped_labels)
    labels_arr = as_backend_array(lab_img.data)
    shape3 = tuple(int(x) for x in labels_arr.shape[:3])

    cd_img = imread(_cd_path(nifti_root, subject))
    cd = as_backend_array(cd_img.data).astype(np.float64)

    venous_region = venous_search_region(shape3)
    vessel_bin, sliding_opt_thresh = binary_vessel_segment_cd(cd)
    labels_np = as_backend_array(labels_arr).astype(np.int32, copy=False)
    # Arterial multilabel: eICAB labels cleared in the venous slab, then area opening on the hull.
    arterial_vol = np.where(venous_region, 0, labels_np)
    art_bin = arterial_vol > 0
    min_art = max(1, int(round(0.005 * int(np.count_nonzero(art_bin)))))
    art_bin = _bwareaopen_bool(art_bin, min_size=min_art, connectivity=1)
    arterial_vol = np.where(art_bin, arterial_vol, 0).astype(np.int32, copy=False)

    # Per-vessel centerlines in 4D flow: skeleton per label → ordered polyline; ``vid`` = eICAB vessel id.
    arterial = compute_centerlines(as_backend_array(arterial_vol), min_points=5)

    # Venous: global vessel mask restricted to the venous slab, then area opening.
    venous_mask = vessel_bin & venous_region
    min_ven = max(1, int(round(0.005 * int(np.count_nonzero(venous_mask)))))
    venous_clean = _bwareaopen_bool(venous_mask, min_size=min_ven, connectivity=1)

    from skimage.morphology import skeletonize  # type: ignore

    ven_sk = as_backend_array(skeletonize(to_numpy(venous_clean.astype(np.uint8, copy=False))))
    ven_coords = np.argwhere(ven_sk > 0).astype(np.float32)
    ven_ids = np.full(ven_coords.shape[0], VENOUS_UNKNOWN_LABEL, dtype=np.int32)

    np.savez_compressed(
        out_dir / "centerlines.npz",
        **{f"arterial_{k}": v.astype(np.float32) for k, v in arterial.items()},
        venous_xyz=ven_coords,
        venous_vessel_id=ven_ids,
    )

    cl_mask = _rasterize_centerlines_mask(shape3, arterial, ven_sk, ven_label_id=VENOUS_UNKNOWN_LABEL)
    mask_path = out_dir / "centerlines_mask.nii.gz"
    imsave(mask_path, cl_mask, metadata=dict(lab_img.metadata or {}))

    meta_out = {
        "subject": subject,
        "eicab_in_4dflow": str(warped_labels),
        "arterial_labels": [int(k) for k in sorted(arterial.keys())],
        "n_venous_points": int(ven_coords.shape[0]),
        "centerlines_mask_nifti": str(mask_path),
        "binary_segmentation_sliding_threshold": True,
        "sliding_threshold_step": 0.001,
        "sliding_threshold_up_thresh": 0.8,
        "sliding_threshold_smf": 10,
        "sliding_threshold_shift_hm": True,
        "sliding_threshold_med_filt": True,
        "sliding_threshold_opt_absolute": float(sliding_opt_thresh),
        "global_bwareaopen_fraction_of_foreground": 0.005,
        "venous_bwareaopen_fraction_of_venous_mask": 0.005,
        "venous_region_axis1_third": int(max(1, round(shape3[1] / 3.0))),
        "min_points_per_vessel": 5,
        "mask_value_arterial": "eICAB_label_id",
        "mask_value_venous": int(VENOUS_UNKNOWN_LABEL),
    }
    done.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    log.info(f"[{subject}] stage3 centerline -> {out_dir}")
    return out_dir


def submit_subject_sge(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    hold_jid: str | None = None,
    emit: TextIO | None = None,
) -> str:
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    script = f"{binds.src}nvitk/pipes/qvtpy/stage3_centerline.py"
    parts: list[str] = [
        "python",
        shlex.quote(script),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
    ]
    if skip_existing:
        parts.append("--skip-existing")
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
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage3_{subject}",
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


@click.command("qvtpy-stage3-centerline")
@click.option("--subject", required=True)
@click.option("--nifti-root", type=click.Path(path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--skip-existing", is_flag=True, default=False)
def main(subject: str, nifti_root: Path, output_root: Path, skip_existing: bool) -> None:
    Logger()
    run_subject(subject, nifti_root=nifti_root, output_root=output_root, skip_existing=skip_existing)


__all__ = ["main", "run_subject", "submit_subject_sge"]
