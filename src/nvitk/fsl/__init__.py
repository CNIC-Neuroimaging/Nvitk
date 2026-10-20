"""
The FSL suite (FMRIB Software Library) from nvitk: structural (BET, FAST, FIRST, fsl_anat,
SIENA/X …), registration (FNIRT, MCFLIRT, epi_reg … — FLIRT itself is
:mod:`nvitk.registration.fsl`), diffusion (TOPUP, EDDY, DTIFIT, BEDPOSTX, PROBTRACKX2, TBSS,
XTRACT), functional (MELODIC, SUSAN, slice timing, FUGUE), ASL perfusion (BASIL:
:mod:`nvitk.fsl.asl` — oxford_asl, basil, asl_file, asl_calib) and utilities (fslmaths,
fslstats, cluster …).

Every wrapper returns an :class:`~nvitk.fsl.run.FslCommand` — argv, expected outputs, whether
it is long-running — and :meth:`~nvitk.fsl.run.FslCommand.run` runs it on the FSL installation
:func:`~nvitk.fsl.env.resolve_fsldir` picks (``$FSLDIR``, or nvitk's own core from
``nvitk-fsl install``)::

    from nvitk.fsl import structural

    outputs = structural.bet("T1.nii.gz", "T1_brain", f=0.4).run()
    outputs["mask"]   # T1_brain_mask.nii.gz
"""

from __future__ import annotations

from .env import FslNotFound, ensure_fsl_in_process, fsl_env, resolve_fsldir
from .run import FslCommand

__all__ = ["FslCommand", "FslNotFound", "ensure_fsl_in_process", "fsl_env", "resolve_fsldir"]
