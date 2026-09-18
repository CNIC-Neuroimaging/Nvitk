"""Cluster / SGE submission utilities shared across nvitk.

All data transfer to and from cluster storage goes through :mod:`nvitk.cluster.sshfs`;
cluster storage is not mounted on the workstation. Paramiko is used only to run commands
on the login node (``qsub``, ``qstat``, ``bash submit.sh``).
"""

from __future__ import annotations

from .remote_submit import run_sge_script_ssh
from .remote_transfer import ClusterSession, cluster_session, ssh_exec
from .sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    build_qsub_command,
    build_singularity_command,
    submit_chain,
    submit_stage,
    write_script_header,
)
from .sshfs import SshfsMount, SshfsRootNotAllowed, assert_no_mountpoint_leak, cluster_mount
from . import sge_json

__all__ = [
    "ClusterPaths",
    "ClusterSession",
    "SshfsMount",
    "SshfsRootNotAllowed",
    "SgeResources",
    "SingularityBinds",
    "StageSpec",
    "assert_no_mountpoint_leak",
    "build_qsub_command",
    "build_singularity_command",
    "cluster_mount",
    "cluster_session",
    "sge_json",
    "ssh_exec",
    "submit_chain",
    "submit_stage",
    "run_sge_script_ssh",
    "write_script_header",
]
