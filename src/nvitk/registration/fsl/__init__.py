"""FSL-backed registration (NiPype interfaces)."""

from nvitk.registration.fsl.flirt import flirt_apply_rigid, flirt_register_rigid, FlirtRigidResult

__all__ = ["FlirtRigidResult", "flirt_apply_rigid", "flirt_register_rigid"]
