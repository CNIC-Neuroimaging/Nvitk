"""Medical image registration helpers (FSL + optional ANTsPy / FireANTs)."""

from nvitk.registration.fsl.flirt import flirt_apply_rigid, flirt_register_rigid
from nvitk.registration.fsl.flirt import FlirtRigidResult
from nvitk.registration.ants import AntsRegistrationResult, ants_apply, ants_register
from nvitk.registration.fireants import FireAntsResult, fireants_apply, fireants_register

__all__ = [
    "AntsRegistrationResult",
    "FireAntsResult",
    "FlirtRigidResult",
    "ants_apply",
    "ants_register",
    "fireants_apply",
    "fireants_register",
    "flirt_apply_rigid",
    "flirt_register_rigid",
]
