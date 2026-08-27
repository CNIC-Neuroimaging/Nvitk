"""SSH helpers to run an emitted SGE bash script on a login node.

Workstations do not have ``qsub``: SGE submission goes through a login node. The pipelines
therefore emit their whole submission as a bash driver script, SFTP it to the cluster, and run
it there — see :mod:`nvitk.cluster.sge_remote` for the staging half.

:func:`prompt_ssh_credentials` is the shared entry point for obtaining the host, user and
password, so every pipeline asks for them the same way and resolves the same host aliases.
"""

from __future__ import annotations

import getpass
import os
import shlex
import sys
from pathlib import Path
from typing import Mapping


def prompt_ssh_credentials(
    *,
    remote_host: str | None = None,
    remote_user: str | None = None,
    host_aliases: Mapping[str, str] | None = None,
) -> tuple[str, str, str]:
    """Resolve ``(host, user, password)`` from arguments or interactive prompts.

    *remote_host* is looked up in *host_aliases* first, so a pipeline's short name for the
    cluster works wherever a hostname is accepted. The password is always prompted for and
    never read from an argument or the environment — it should not end up in a shell history,
    a process listing, or a log.

    Raises
    ------
    RuntimeError
        If Paramiko is missing, naming the install command. Prompting for a password and only
        then discovering there is no SSH library would be a poor trade.
    """
    try:
        import paramiko  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Cluster submission over SSH requires Paramiko (pip install paramiko)."
        ) from exc

    import click

    aliases = dict(host_aliases or {})
    host_key = remote_host or click.prompt("SSH hostname (short name or IP)")
    host = aliases.get(host_key, host_key)
    user = remote_user or click.prompt("SSH user")
    password = getpass.getpass(f"SSH password for {user}@{host}: ")
    return host, user, password


def run_sge_script_ssh(
    host: str,
    user: str,
    password: str,
    script_path: Path | str,
    *,
    local_script_path: Path | None = None,
    port: int = 22,
    timeout: float | None = None,
) -> bool:
    """Execute ``bash <script>`` on *host* via Paramiko. Returns True on exit 0."""
    code, _out, _err = run_sge_script_ssh_capture(
        host,
        user,
        password,
        script_path,
        local_script_path=local_script_path,
        port=port,
        connect_timeout=timeout,
    )
    return code == 0


def run_sge_script_ssh_capture(
    host: str,
    user: str,
    password: str,
    script_path: Path | str,
    *,
    local_script_path: Path | None = None,
    port: int = 22,
    connect_timeout: float | None = None,
) -> tuple[int, str, str]:
    """Execute ``bash <script>`` on the cluster; return ``(exit_code, stdout, stderr)``.

    *connect_timeout* applies only to the SSH handshake. The remote ``bash`` run
    blocks until the script exits; there is no read timeout while the script runs
    (large cohort submissions may take many minutes).
    """
    from nvitk.core.logger import Logger

    log = Logger()
    try:
        import paramiko
    except ImportError:
        log.warning(
            "paramiko is not installed (pip install 'nvitk[cluster]' or paramiko); "
            "skipping SSH. Run the script on the cluster login node manually."
        )
        return 127, "", "paramiko not installed"

    remote_script = str(script_path)
    remote_cmd = f"bash {shlex.quote(remote_script)}"

    quiet = os.environ.get("NVITK_QUIET_SGE_SUMMARY", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not quiet:
        try:
            from nvitk.cluster.sge import format_sge_driver_script_variables

            summary_src = local_script_path if local_script_path is not None else Path(script_path)
            txt = summary_src.read_text(encoding="utf-8", errors="replace")
            print(
                format_sge_driver_script_variables(txt, Path(remote_script)),
                file=sys.stderr,
                flush=True,
            )
        except OSError as exc:
            log.warning("Could not read SGE driver script for summary: %s", exc)
        print(
            f"[nvitk|SGE] Remote exec via SSH: {user}@{host}:{port} "
            f"bash {shlex.quote(remote_script)}",
            file=sys.stderr,
            flush=True,
        )

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            password=password,
            timeout=connect_timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        _stdin, stdout, stderr = client.exec_command(remote_cmd)
        out_b = stdout.read()
        err_b = stderr.read()
        exit_status = int(stdout.channel.recv_exit_status())
        out_s = out_b.decode(errors="replace")
        err_s = err_b.decode(errors="replace")
        if out_s:
            log.info("SSH connection established | status: %s", exit_status)
            log.info("SSH stdout (tail): \n %s", out_s[-2000:])
        if err_s:
            log.error("SSH connection established | status: %s", exit_status)
            log.error("SSH stderr (tail): \n %s", err_s[-2000:])
        return exit_status, out_s, err_s
    except Exception as exc:
        log.warning("SSH exec failed: %s", exc)
        return 127, "", str(exc)
    finally:
        client.close()


__all__ = [
    "prompt_ssh_credentials","run_sge_script_ssh", "run_sge_script_ssh_capture"]
