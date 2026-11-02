"""
FastSurfer integration: whole-brain (aparc.DKTatlas+aseg), cerebellum (CerebNet),
hypothalamus (HypVINN) and corpus-callosum segmentation, surfaces (recon-surf, with
FreeSurfer) and longitudinal processing; run from the FastSurfer tree nvitk installs
into the environment (``nvitk-fastsurfer install``).

Usage::

    from nvitk.segmentation.fastsurfer import FastSurferRequest, fastsurfer_argv, run

    argv, env = fastsurfer_argv(FastSurferRequest(sd="subjects", sid="sub01", t1="T1.nii.gz",
                                                  native_image=True))
    run(argv, env)
"""

from __future__ import annotations

from .config import (
    FASTSURFER_VERSION,
    FREESURFER_VERSION,
    doctor,
    fastsurfer_home,
    freesurfer_home,
    fs_license,
    surface_requirements,
)
from .outputs import SubjectOutputs, lut_for, read_lut, read_stats, read_surface, stats_table
from .runner import (
    FastSurferNotInstalled,
    FastSurferRequest,
    conform_argv,
    fastsurfer_argv,
    long_argv,
    reduce_to_aseg_argv,
    run,
    segstats_argv,
    stage_of,
)

__all__ = [
    "FASTSURFER_VERSION",
    "FREESURFER_VERSION",
    "FastSurferNotInstalled",
    "FastSurferRequest",
    "SubjectOutputs",
    "conform_argv",
    "doctor",
    "fastsurfer_argv",
    "fastsurfer_home",
    "freesurfer_home",
    "fs_license",
    "long_argv",
    "lut_for",
    "read_lut",
    "read_stats",
    "read_surface",
    "reduce_to_aseg_argv",
    "run",
    "segstats_argv",
    "stage_of",
    "stats_table",
    "surface_requirements",
]
