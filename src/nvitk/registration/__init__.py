"""Medical image registration helpers (FSL via NiPype)."""

from nvitk.registration.fsl.flirt import flirt_apply_rigid, flirt_register_rigid
from nvitk.registration.fsl.flirt import FlirtRigidResult

__all__ = [
    "FlirtRigidResult",
    "flirt_apply_rigid",
    "flirt_register_rigid",
]
