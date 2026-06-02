"""Qt dialog for SSH credentials and remote job directory."""

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


@dataclass(frozen=True)
class SgeConnectionSettings:
    host: str
    user: str
    password: str
    remote_job_root: str


def _default_host() -> str:
    paths = sge_json.paths_section()
    aliases = sge_json.merge_cluster_host_aliases({}, paths, {})
    for name in ("samwise", "login", "cluster"):
        if name in aliases:
            return name
    return "samwise"


def _default_remote_job_root() -> str:
    return sge_json.gui_sge_job_root()


class SgeSubmitDialog(QDialog):
    """Collect SSH host, credentials, and remote job root."""

    def __init__(self, parent=None) -> None:
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
            default_root or "/data3/BIOIT_IMAGE/nvitk-sge/gui/<job_id>"
        )

        form = QFormLayout()
        form.addRow("SSH host", self.host)
        form.addRow("Username", self.user)
        form.addRow("Password", self.password)
        form.addRow("Remote job directory", self.remote_job_root)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def settings(self) -> SgeConnectionSettings:
        return SgeConnectionSettings(
            host=self.host.text().strip(),
            user=self.user.text().strip(),
            password=self.password.text(),
            remote_job_root=self.remote_job_root.text().strip(),
        )

    def accept(self) -> None:
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
        super().accept()


__all__ = ["SgeConnectionSettings", "SgeSubmitDialog"]
