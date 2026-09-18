"""Qt dialog for SSH credentials, remote job directory and the SGE resource request."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from nvitk.cluster import sge_json
from nvitk.cluster.sge import SgeResourceOverrides, is_valid_h_vmem


@dataclass(frozen=True)
class SgeConnectionSettings:
    """What :class:`SgeSubmitDialog` collected: where to connect, and what to ask SGE for."""

    host: str
    user: str
    password: str
    remote_job_root: str
    overrides: SgeResourceOverrides = SgeResourceOverrides()


def _default_host() -> str:
    """Default SSH host for the connection dialog: the first configured cluster alias.

    Aliases come from ``sge.json`` ``paths.cluster_host_aliases``; there is no built-in
    hostname, so an unconfigured install shows an empty field for the user to fill rather than
    pre-filling somebody else's cluster.
    """
    aliases = sge_json.merge_cluster_host_aliases({}, sge_json.paths_section(), {})
    return next(iter(aliases), "")


def _default_remote_job_root() -> str:
    """Default remote job root directory for GUI SGE submissions."""
    return sge_json.gui_sge_job_root()


def _configured_resources() -> tuple[str, str, str]:
    """``(project, account, h_vmem)`` the job would use with nothing overridden.

    Read from :mod:`nvitk.cli.config`, which is what actually builds a GUI job's request, so
    the prefilled fields always match what submitting unchanged would do.
    """
    from nvitk.cli import config as cfg

    return str(cfg.SGE_PROJECT or ""), str(cfg.SGE_ACCOUNT or ""), str(cfg.SGE_H_VMEM or "")


class SgeSubmitDialog(QDialog):
    """Collect SSH host, credentials, remote job root and the SGE resource request."""

    def __init__(self, parent=None) -> None:
        """Build the connection and resource form, prefilled from the configured defaults."""
        super().__init__(parent)
        self.setWindowTitle("Run on SGE cluster")
        self.setMinimumWidth(420)

        intro = QLabel(
            "Export the active layer, upload inputs to the cluster, and submit "
            "a Singularity job. Results download and import automatically when the job completes."
        )
        intro.setWordWrap(True)

        self.host = QLineEdit(_default_host())
        self.user = QLineEdit("")
        self.password = QLineEdit("")
        self.password.setEchoMode(QLineEdit.Password)
        default_root = _default_remote_job_root()
        self.remote_job_root = QLineEdit(default_root)
        self.remote_job_root.setPlaceholderText(
            default_root or "set paths.gui_sge_job_root in sge.json, or type a path"
        )

        project, account, h_vmem = _configured_resources()
        self.project = QLineEdit(project)
        self.project.setToolTip(
            "qsub -P. A project starting with L, S or XS also selects the matching "
            "virtual-GPU resource (-l lgpu/sgpu/xsgpu)."
        )
        # Offered alongside the project because this site sets -P and -A together in every
        # pipeline block; changing one without the other submits a mismatched pair.
        self.account = QLineEdit(account)
        self.account.setToolTip("qsub -A.")
        self.h_vmem = QLineEdit(h_vmem)
        self.h_vmem.setToolTip("qsub -l h_vmem. A number with an optional suffix, e.g. 30G.")

        form = QFormLayout()
        form.addRow("SSH host", self.host)
        form.addRow("Username", self.user)
        form.addRow("Password", self.password)
        form.addRow("Remote job directory", self.remote_job_root)
        form.addRow("SGE project (-P)", self.project)
        form.addRow("SGE account (-A)", self.account)
        form.addRow("Memory (h_vmem)", self.h_vmem)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def settings(self) -> SgeConnectionSettings:
        """Read the current form field values into an :class:`SgeConnectionSettings`."""
        project, account, h_vmem = _configured_resources()
        chosen_project = self.project.text().strip()
        chosen_account = self.account.text().strip()
        chosen_h_vmem = self.h_vmem.text().strip()
        return SgeConnectionSettings(
            host=self.host.text().strip(),
            user=self.user.text().strip(),
            password=self.password.text(),
            remote_job_root=self.remote_job_root.text().strip(),
            # Only send what the operator actually changed, so an untouched dialog submits
            # exactly the configured request rather than a copy of it.
            overrides=SgeResourceOverrides(
                project=chosen_project if chosen_project != project else None,
                account=chosen_account if chosen_account != account else None,
                h_vmem=chosen_h_vmem if chosen_h_vmem != h_vmem else None,
            ),
        )

    def accept(self) -> None:
        """Validate before accepting; focus the first invalid field otherwise.

        Host, user and project must be filled, the remote job root falls back to the
        configured default when blank, and the memory request must be a string ``qsub``
        will accept.
        """
        s = self.settings()
        if not s.host:
            self.host.setFocus()
            return
        if not s.user:
            self.user.setFocus()
            return
        if not s.remote_job_root:
            fallback = _default_remote_job_root()
            if fallback:
                self.remote_job_root.setText(fallback)
            else:
                self.remote_job_root.setFocus()
                return
        if not self.project.text().strip():
            self.project.setFocus()
            return
        # Checked here rather than at qsub time: the job is staged and uploaded first, so a
        # rejected memory string would otherwise surface minutes later on the login node.
        if not is_valid_h_vmem(self.h_vmem.text()):
            self.h_vmem.setFocus()
            self.h_vmem.selectAll()
            return
        super().accept()


__all__ = ["SgeConnectionSettings", "SgeSubmitDialog"]
