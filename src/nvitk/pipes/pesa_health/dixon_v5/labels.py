"""Output label definitions for the dixon-v5 post-processing stage.

Hemisphere-preserving schema
----------------------------

* Autochthon left / right (paravertebral muscle, ``PVM``) are kept **separate**
  in both the HEAD and THORAX masks.
* Kidneys left / right are kept **separate**.
* Quadriceps left / right (``QM``) are kept **separate** in the LEGS mask.
* Deltoid and trapezius are **not** part of the Dixon output contract.

Output files and label schemes
------------------------------

.. code-block:: text

    HEAD.nii
        H_PVM_L = 1        # autochthon_left
        H_PVM_R = 2        # autochthon_right

    THORAX.nii
        LIVER     = 1
        PANCREAS  = 2
        KIDNEY_L  = 3
        KIDNEY_R  = 4
        T_PVM_L   = 5      # autochthon_left
        T_PVM_R   = 6      # autochthon_right
        BN_L3     = 7      # vertebrae_L3 bone narrow
        BN_L4     = 8      # vertebrae_L4 bone narrow

    LEGS.nii
        L_QM_L = 1         # quadriceps_femoris_left
        L_QM_R = 2         # quadriceps_femoris_right
"""

from __future__ import annotations

HEAD_LABELS: dict[str, int] = {
    "H_PVM_L": 1,
    "H_PVM_R": 2,
}

THORAX_LABELS: dict[str, int] = {
    "LIVER": 1,
    "PANCREAS": 2,
    "KIDNEY_L": 3,
    "KIDNEY_R": 4,
    "T_PVM_L": 5,
    "T_PVM_R": 6,
    "BN_L3": 7,
    "BN_L4": 8,
}

LEGS_LABELS: dict[str, int] = {
    "L_QM_L": 1,
    "L_QM_R": 2,
}


__all__ = ["HEAD_LABELS", "THORAX_LABELS", "LEGS_LABELS"]
