"""ToPBrain trainer variants: one per selectable loss, for CNN and transformer encoders.

Lives inside the nnU-Net tree because this build resolves trainers only within its own package
(``recursive_find_python_class`` over ``nnunetv2/training/nnUNetTrainer``) — there is no
external-trainer hook. The loss implementations themselves stay in the installed ``nvitk``
package; these modules only define trainer classes.
"""

from __future__ import annotations

__all__: list[str] = []
