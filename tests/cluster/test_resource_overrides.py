"""Tests for per-submission SGE resource overrides.

A GUI cluster launch used to be stuck with whatever ``sge.json`` resolved for ``image_tools``.
:class:`~nvitk.cluster.sge.SgeResourceOverrides` lets one submission ask for a different
project or memory size without editing configuration, and these tests pin two things that are
easy to get wrong: an untouched dialog must submit exactly the configured request, and
changing the project must carry the virtual-GPU flag with it.
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
from pathlib import Path

import pytest

from nvitk.cli import _sge
from nvitk.cli import config as cfg
from nvitk.cluster.sge import (
    SgeResourceOverrides,
    SgeResources,
    is_valid_h_vmem,
    qsub_l_resource_args,
)

BASE = SgeResources(project="SGPU", account="SGPU", ngpu=0, h_vmem="30G", queue=None)


class TestHVmemValidation:
    """A bad memory string must be caught before the job is staged and uploaded."""

    @pytest.mark.parametrize("value", ["30G", "4096M", "8", "1.5G", "512k", "2T"])
    def test_accepts_qsub_spellings(self, value):
        assert is_valid_h_vmem(value)

    @pytest.mark.parametrize("value", ["30 GB", "lots", "", "  ", "30GB", "-5G", "G30"])
    def test_rejects_the_rest(self, value):
        assert not is_valid_h_vmem(value)

    def test_apply_refuses_an_invalid_request(self):
        with pytest.raises(ValueError, match="Invalid h_vmem"):
            SgeResourceOverrides(h_vmem="plenty").apply(BASE)


class TestApply:
    """Set fields replace, unset fields are left alone."""

    def test_nothing_set_is_an_exact_passthrough(self):
        assert SgeResourceOverrides().apply(BASE) == BASE

    def test_each_field_replaces_independently(self):
        assert SgeResourceOverrides(project="LGPU").apply(BASE).project == "LGPU"
        assert SgeResourceOverrides(account="Prod").apply(BASE).account == "Prod"
        assert SgeResourceOverrides(h_vmem="120G").apply(BASE).h_vmem == "120G"

    def test_unset_fields_survive_a_change_to_another(self):
        out = SgeResourceOverrides(h_vmem="120G").apply(BASE)
        assert (out.project, out.account, out.queue) == ("SGPU", "SGPU", None)

    def test_ngpu_and_pe_smp_are_never_touched(self):
        base = SgeResources(project="LGPU", account="Prod", ngpu=1, h_vmem="80G",
                            queue=None, pe_smp=8)
        out = SgeResourceOverrides(project="XSGPU", h_vmem="10G").apply(base)
        assert (out.ngpu, out.pe_smp) == (1, 8)

    def test_an_empty_string_is_treated_as_unset(self):
        """A blank field must not blank the configured value."""
        assert SgeResourceOverrides(project="", h_vmem="").apply(BASE) == BASE


class TestProjectCarriesTheGpuFlag:
    """``-l lgpu|sgpu|xsgpu`` follows the project name, so changing it changes the flag."""

    @pytest.mark.parametrize("project,expected", [
        ("SGPU", "sgpu=0"), ("LGPU", "lgpu=0"), ("XSGPU", "xsgpu=0"),
    ])
    def test_virtual_gpu_resource_follows_the_override(self, project, expected):
        args = qsub_l_resource_args(SgeResourceOverrides(project=project).apply(BASE))
        assert expected in args

    def test_a_classic_project_keeps_ngpu(self):
        base = SgeResources(project="GPU", account="Prod", ngpu=2, h_vmem="80G", queue=None)
        args = qsub_l_resource_args(SgeResourceOverrides(project="GPU").apply(base))
        assert "ngpu=2" in args

    def test_switching_to_a_virtual_gpu_project_drops_ngpu(self):
        base = SgeResources(project="GPU", account="Prod", ngpu=2, h_vmem="80G", queue=None)
        args = qsub_l_resource_args(SgeResourceOverrides(project="LGPU").apply(base))
        assert "lgpu=0" in args and "ngpu=2" not in args


class TestDefaultResources:
    """``default_resources`` is where configuration and overrides meet."""

    @pytest.fixture(autouse=True)
    def _pinned_config(self, monkeypatch):
        monkeypatch.setattr(cfg, "SGE_PROJECT", "SGPU")
        monkeypatch.setattr(cfg, "SGE_ACCOUNT", "SGPU")
        monkeypatch.setattr(cfg, "SGE_H_VMEM", "30G")
        monkeypatch.setattr(cfg, "SGE_QUEUE", None)
        monkeypatch.setattr(cfg, "SGE_NGPU", 1)

    def test_without_overrides_it_is_the_configured_request(self):
        assert _sge.default_resources(gpu=False) == BASE

    def test_gpu_requests_a_slot(self):
        assert _sge.default_resources(gpu=True).ngpu == 1

    def test_cpu_requests_none(self):
        assert _sge.default_resources(gpu=False).ngpu == 0

    def test_overrides_are_applied_on_top(self):
        out = _sge.default_resources(
            gpu=True, overrides=SgeResourceOverrides(project="LGPU", h_vmem="120G")
        )
        assert (out.project, out.h_vmem, out.account, out.ngpu) == ("LGPU", "120G", "SGPU", 1)


class TestEmittedScript:
    """The override has to survive all the way into the qsub argv."""

    def _qsub_argv(self, overrides):
        out = Path(tempfile.mkdtemp(prefix="nvitk-override-")) / "submit.sh"
        _sge.emit_submit_script(
            script_path=out,
            stages=[("gui_seg_topbrain", "python -m nvitk.gui.sge.worker")],
            data_root=Path("/BIOIT_IMAGE/d"),
            output_root=Path("/BIOIT_IMAGE/o"),
            gpu=True,
            overrides=overrides,
        )
        # The driver script emits the argv as a bash array, one bare word per line;
        # shlex.quote only adds quotes where the shell needs them, so parse the block.
        block = re.search(r"qsub_\w+=\(\n(.*?)\n\)", out.read_text(), re.S)
        assert block is not None, "no qsub array in the emitted script"
        return [line.strip() for line in block.group(1).splitlines() if line.strip()]

    def test_the_override_reaches_the_script(self):
        argv = self._qsub_argv(SgeResourceOverrides(project="LGPU", account="Prod", h_vmem="120G"))
        assert "LGPU" in argv and "Prod" in argv
        assert "h_vmem=120G" in argv
        assert "lgpu=0" in argv

    def test_without_an_override_the_configured_values_are_emitted(self, monkeypatch):
        monkeypatch.setattr(cfg, "SGE_PROJECT", "SGPU")
        monkeypatch.setattr(cfg, "SGE_ACCOUNT", "SGPU")
        monkeypatch.setattr(cfg, "SGE_H_VMEM", "30G")
        argv = self._qsub_argv(None)
        assert "SGPU" in argv and "h_vmem=30G" in argv


#: Gate only the dialog tests on Qt. A module-level ``importorskip`` would skip the pure
#: resource-logic tests above it too, which need no GUI at all.
_HAS_QT = importlib.util.find_spec("qtpy") is not None


@pytest.mark.skipif(not _HAS_QT, reason="dialog tests need Qt (qtpy)")
class TestSubmitDialog:
    """The dialog must only send what the operator actually changed."""

    @pytest.fixture
    def dialog(self, monkeypatch):
        """A dialog with pinned defaults, closed on teardown.

        The close/deleteLater is not optional: an offscreen Qt widget left alive at
        interpreter shutdown aborts the process *after* the tests have already passed.
        """
        from qtpy.QtWidgets import QApplication

        from nvitk.gui.sge import dialog as dlg_mod

        monkeypatch.setattr(dlg_mod, "_configured_resources", lambda: ("SGPU", "SGPU", "30G"))
        app = QApplication.instance() or QApplication([])
        widget = dlg_mod.SgeSubmitDialog()
        yield widget
        widget.close()
        widget.deleteLater()
        app.processEvents()

    def test_fields_are_prefilled_from_configuration(self, dialog):
        assert (dialog.project.text(), dialog.account.text(), dialog.h_vmem.text()) == (
            "SGPU", "SGPU", "30G",
        )

    def test_an_untouched_dialog_sends_no_overrides(self, dialog):
        assert dialog.settings().overrides == SgeResourceOverrides()

    def test_only_changed_fields_are_sent(self, dialog):
        dialog.h_vmem.setText("120G")
        assert dialog.settings().overrides == SgeResourceOverrides(h_vmem="120G")

    def test_an_invalid_memory_request_blocks_accept(self, dialog):
        dialog.host.setText("samwise")
        dialog.user.setText("imarcoss")
        dialog.remote_job_root.setText("/BIOIT_IMAGE/nvitk-sge/gui")
        dialog.h_vmem.setText("plenty")
        dialog.accept()
        assert not dialog.isVisible() or dialog.result() != dialog.Accepted

    def test_a_blank_project_blocks_accept(self, dialog):
        dialog.host.setText("samwise")
        dialog.user.setText("imarcoss")
        dialog.remote_job_root.setText("/BIOIT_IMAGE/nvitk-sge/gui")
        dialog.project.setText("")
        dialog.accept()
        assert dialog.result() != dialog.Accepted
