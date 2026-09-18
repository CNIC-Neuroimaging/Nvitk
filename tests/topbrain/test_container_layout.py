"""Regression tests for resolving ToPBrain roots inside an SGE container.

``sge.json`` is a workstation file. It is not mounted into a cluster job, and the keys it
holds (``local_*`` especially) describe a machine the job never touches. A GUI cluster launch
used to die in the worker with::

    Required setting "pipelines.topbrain_paths.local_challenge_root" is not configured

because the container path went through ``layout_local``, which resolves *all ten* roots from
config even though the job binds -- and inference reads -- only four. These tests pin the
contract that the container layout resolves nothing it was not handed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nvitk.core import config_paths
from nvitk.gui.tools import topbrain_models as tm
from nvitk.pipes.topbrain.util import paths as tp

#: The four roots an SGE job binds and exports, as container paths.
BOUND = {
    "results_root": Path("/nvitk/models/RESULTS"),
    "nnunet_results": Path("/nvitk/models/nnUNet/results"),
    "nnunet_raw": Path("/nvitk/models/nnUNet/raw"),
    "nnunet_preprocessed": Path("/nvitk/models/nnUNet/preprocessed"),
}

#: Everything else in ``ROOT_KEYS`` -- training-time roots inference never reads.
UNBOUND = tuple(k for k in tp.ROOT_KEYS if k not in BOUND)


@pytest.fixture
def _no_config(monkeypatch):
    """Make every configuration read raise, so a stray one cannot pass silently."""

    def explode(*_args, **_kwargs):
        raise AssertionError("configuration was read inside the container")

    monkeypatch.setattr(tp, "_pipe_paths", explode)
    monkeypatch.setattr(config_paths, "load_json", explode)
    monkeypatch.setattr(config_paths, "require", explode)


class TestLayoutContainer:
    """The container layout is built from bind targets only."""

    def test_bound_roots_are_used_verbatim(self, _no_config):
        paths = tp.layout_container(**BOUND)
        for key, value in BOUND.items():
            assert getattr(paths, key) == value

    def test_unbound_roots_are_marked_unavailable(self, _no_config):
        paths = tp.layout_container(**BOUND)
        for key in UNBOUND:
            root = getattr(paths, key)
            assert root == tp.UNAVAILABLE_ROOT / key
            assert not root.exists()

    def test_reads_no_configuration(self, _no_config):
        """The whole point: no sge.json on the cluster, so none may be consulted."""
        tp.layout_container(**BOUND)  # the fixture raises if anything reads config

    def test_none_valued_overrides_are_treated_as_absent(self, _no_config):
        paths = tp.layout_container(challenge_root=None, **BOUND)
        assert paths.challenge_root == tp.UNAVAILABLE_ROOT / "challenge_root"
        assert paths.results_root == BOUND["results_root"]

    def test_with_no_overrides_every_root_is_unavailable(self, _no_config):
        paths = tp.layout_container()
        for key in tp.ROOT_KEYS:
            assert getattr(paths, key) == tp.UNAVAILABLE_ROOT / key

    def test_the_unavailable_marker_is_recognisable_in_a_traceback(self):
        """A path that fails should say why, not look like a plausible data root."""
        assert "not-bound-in-container" in str(tp.UNAVAILABLE_ROOT)

    def test_layout_local_still_consults_config(self, _no_config):
        """Contrast: the workstation layout must keep resolving from sge.json."""
        with pytest.raises(AssertionError):
            tp.layout_local(**BOUND)


class TestContainerOverrides:
    """Reading the job's exported bind targets out of the environment."""

    def test_reads_every_exported_root(self, monkeypatch):
        for field, variable in tm.CONTAINER_ROOT_ENV.items():
            monkeypatch.setenv(variable, str(BOUND[field]))
        assert tm.container_overrides() == BOUND

    def test_blank_and_whitespace_values_are_ignored(self, monkeypatch):
        monkeypatch.setenv("TOPBRAIN_RESULTS_ROOT", "   ")
        for variable in tm.CONTAINER_ROOT_ENV.values():
            if variable != "TOPBRAIN_RESULTS_ROOT":
                monkeypatch.delenv(variable, raising=False)
        assert tm.container_overrides() == {}

    def test_empty_environment_yields_no_overrides(self, monkeypatch):
        for variable in tm.CONTAINER_ROOT_ENV.values():
            monkeypatch.delenv(variable, raising=False)
        assert tm.container_overrides() == {}


class TestHostLayout:
    """Which layout the GUI tool picks, on the cluster and off it."""

    def test_in_a_container_it_uses_the_bind_targets_and_no_config(
        self, monkeypatch, _no_config
    ):
        for field, variable in tm.CONTAINER_ROOT_ENV.items():
            monkeypatch.setenv(variable, str(BOUND[field]))
        paths, origin = tm.host_layout()
        assert origin == "container"
        assert paths.results_root == BOUND["results_root"]
        assert paths.challenge_root == tp.UNAVAILABLE_ROOT / "challenge_root"

    def test_off_the_cluster_it_falls_back_to_probing(self, monkeypatch):
        for variable in tm.CONTAINER_ROOT_ENV.values():
            monkeypatch.delenv(variable, raising=False)
        sentinel = object()
        monkeypatch.setattr(tp, "layout_auto", lambda **_k: (sentinel, "local"))
        paths, origin = tm.host_layout()
        assert paths is sentinel and origin == "local"

    def test_a_partial_export_still_takes_the_container_path(self, monkeypatch, _no_config):
        """One exported root means a container; the rest must not fall back to config."""
        for variable in tm.CONTAINER_ROOT_ENV.values():
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setenv("TOPBRAIN_RESULTS_ROOT", str(BOUND["results_root"]))
        paths, origin = tm.host_layout()
        assert origin == "container"
        assert paths.results_root == BOUND["results_root"]
        assert paths.nnunet_raw == tp.UNAVAILABLE_ROOT / "nnunet_raw"
