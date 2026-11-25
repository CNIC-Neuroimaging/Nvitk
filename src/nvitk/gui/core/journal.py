"""Recording what is done in the GUI, to replay it or export it as a script.

While recording is on (*Export & record ▸ Record*), each action that changes the data is appended
to the viewer's :class:`Journal` as a :class:`Step`: imaging tools (with their parameters and the
layer they ran on), mesh operations, reorientation, ROI crops and masks, files opened and saved.
The journal can then be

* replayed in the GUI, on the same layers or, with *Replay on the active layer*, on another
  subject's layer (the first step's input is replaced by the active layer, and each later step's
  inputs by the outputs it produced);
* saved / loaded as JSON (``nvitk-journal/2``; the v1 "pipeline JSON" of earlier versions loads too);
* exported as a Python script calling :func:`nvitk.gui.tools.runner.run_gui_tool_headless` on
  files, so a GUI session becomes a batch job.

Producers call :func:`record` (a no-op unless recording), so adding a hook is one line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from qtpy.QtCore import QObject, Signal

FORMAT = "nvitk-journal/2"

#: Kinds of steps the journal knows how to replay.
REPLAYABLE = {"tool", "orient", "roi_crop", "roi_mask"}


@dataclass
class Step:
    """One recorded action."""

    kind: str  # tool | mesh_op | orient | roi_crop | roi_mask | open | export | other
    name: str  # tool id, operation id, target orientation...
    params: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    label: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    enabled: bool = True

    def describe(self) -> str:
        what = self.label or self.name
        ins = ", ".join(self.inputs) or "—"
        outs = f" → {', '.join(self.outputs)}" if self.outputs else ""
        return f"[{self.kind}] {what}  ({ins}){outs}"


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:  # noqa: BLE001
            pass
        return str(value)


class Journal(QObject):
    """The viewer's recording (see the module docstring)."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[Step] = []
        self.recording = False

    def set_recording(self, on: bool) -> None:
        self.recording = bool(on)
        self.changed.emit()

    def record(self, step: Step) -> None:
        if not self.recording or getattr(self, "_replaying", False):
            return
        step.params = _jsonable(step.params)
        self.steps.append(step)
        self.changed.emit()

    def clear(self) -> None:
        self.steps.clear()
        self.changed.emit()

    def move(self, index: int, offset: int) -> None:
        j = index + offset
        if 0 <= index < len(self.steps) and 0 <= j < len(self.steps):
            self.steps[index], self.steps[j] = self.steps[j], self.steps[index]
            self.changed.emit()

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.steps):
            del self.steps[index]
            self.changed.emit()

    # files
    def to_dict(self) -> dict[str, Any]:
        return {"format": FORMAT, "steps": [asdict(s) for s in self.steps]}

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    def load(self, path: str | Path) -> None:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        steps = []
        for raw in doc.get("steps", []):
            if "kind" in raw:
                steps.append(Step(**{k: raw[k] for k in Step.__dataclass_fields__ if k in raw}))
            elif raw.get("type") == "tool":  # v1 pipeline JSON
                steps.append(Step("tool", str(raw.get("tool")), dict(raw.get("params") or {}),
                                  [str(raw.get("source_layer", ""))], [str(raw.get("output_layer", ""))],
                                  label=str(raw.get("operation", "")), timestamp=str(raw.get("timestamp", ""))))
            elif raw.get("type"):
                steps.append(Step(str(raw["type"]), str(raw.get("path") or raw.get("name") or ""), {k: v for k, v in raw.items() if k != "type"}))
        self.steps = steps
        self.changed.emit()

    def export_python(self, path: str | Path) -> Path:
        """A script that runs the recorded tool steps on files, with nvitk's headless runner."""
        lines = [
            '"""Recorded with the nvitk GUI; runs the tool steps headless on NIfTI files.',
            "",
            "Edit INPUTS to point each recorded input layer at a file. Steps other than imaging",
            'tools (mesh operations, opening / saving) are listed as comments."""',
            "",
            "from nvitk.gui.tools.runner import run_gui_tool_headless",
            "from nvitk.io import imread, imsave",
            "",
            "INPUTS = {",
        ]
        first_inputs = []
        produced = set()
        for step in self.steps:
            for name in step.inputs:
                if name and name not in produced and name not in first_inputs:
                    first_inputs.append(name)
            produced.update(step.outputs)
        lines += [f'    {name!r}: "{name}.nii.gz",' for name in first_inputs]
        lines += ["}", "images = {name: imread(path) for name, path in INPUTS.items()}", ""]
        for i, step in enumerate(self.steps, 1):
            if not step.enabled:
                lines.append(f"# step {i} disabled: {step.describe()}")
                continue
            if step.kind != "tool":
                lines.append(f"# step {i} (not scriptable here): {step.describe()}")
                continue
            params = {k: v for k, v in step.params.items() if k not in ("target_mode", "label_ids")}
            src = step.inputs[0] if step.inputs else ""
            out = step.outputs[0] if step.outputs else f"step{i}"
            lines += [
                f"# step {i}: {step.label or step.name}",
                f"result = run_gui_tool_headless({step.name!r}, primary=images[{src!r}], aux={{}},",
                f"    target_mode={str(step.params.get('target_mode') or 'raw')!r}, label_ids={step.params.get('label_ids')!r},",
                f"    params={params!r})",
                "if result is not None:",
                f"    images[{out!r}] = images[{src!r}].with_data(result)",
                f"    imsave({out + '.nii.gz'!r}, images[{out!r}])",
                "",
            ]
        target = Path(path)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    # replay
    def replay(self, viewer: Any, *, on_active: bool = False, log: Callable[[str], Any] | None = None) -> list[str]:
        """Run the enabled, replayable steps in the GUI; returns a line per step.

        With *on_active*, the first step's input is the active layer, and every step's inputs are
        renamed through the outputs produced so far, so the chain applies to another subject.
        """
        self._replaying = True
        try:
            viewer._nvitk_replaying = True
        except Exception:  # noqa: BLE001
            pass
        try:
            return self._replay(viewer, on_active=on_active, log=log)
        finally:
            self._replaying = False
            try:
                viewer._nvitk_replaying = False
            except Exception:  # noqa: BLE001
                pass

    def _replay(self, viewer: Any, *, on_active: bool, log: Callable[[str], Any] | None) -> list[str]:
        report: list[str] = []
        rename: dict[str, str] = {}
        active = viewer.layers.selection.active
        first_input = next((s.inputs[0] for s in self.steps if s.enabled and s.inputs), None)
        if on_active and active is not None and first_input:
            rename[first_input] = str(active.name)
        for i, step in enumerate(self.steps, 1):
            if not step.enabled:
                continue
            if step.kind not in REPLAYABLE:
                report.append(f"{i}: skipped {step.describe()}")
                continue
            names = [rename.get(n, n) for n in step.inputs]
            layers = [viewer.layers[n] for n in names if n in viewer.layers]
            if not layers:
                report.append(f"{i}: input {names} not loaded; stopped")
                break
            try:
                produced = _replay_step(viewer, step, layers)
            except Exception as exc:  # noqa: BLE001
                report.append(f"{i}: {step.label or step.name} failed: {exc}; stopped")
                break
            for old, new in zip(step.outputs, produced):
                rename[old] = new
            report.append(f"{i}: {step.label or step.name} → {', '.join(produced) or 'done'}")
            if log is not None:
                log(report[-1])
        return report


def _replay_step(viewer: Any, step: Step, layers: list[Any]) -> list[str]:
    if step.kind == "tool":
        from nvitk.gui.tools.runner import run_gui_tool

        layer = layers[0]
        params = dict(step.params)
        target_mode = str(params.pop("target_mode", None) or "raw")
        label_ids = params.pop("label_ids", None)
        result = run_gui_tool(step.name, layer, viewer, target_mode=target_mode, label_ids=label_ids, params=params)
        if result is None:
            return []
        from nvitk.gui.core.spatial import layer_spatial_kwargs

        name = f"{layer.name}_{step.name}"
        kind_labels = type(layer).__name__ == "Labels"
        adder = viewer.add_labels if kind_labels and str(getattr(result, "dtype", "")).startswith(("int", "uint")) else viewer.add_image
        try:
            adder(result, name=name, **layer_spatial_kwargs(layer, ndim=int(getattr(result, "ndim", layer.ndim))))
        except Exception:  # noqa: BLE001 (a result on another grid)
            adder(result, name=name)
        return [name]
    if step.kind == "orient":
        from nvitk.gui.core.orientation import apply_target_orientation_many

        apply_target_orientation_many(viewer, layers, step.name)
        return [str(l.name) for l in layers]
    if step.kind in ("roi_crop", "roi_mask"):
        from nvitk.gui.core.roi_world import WorldBox, crop_layers, mask_outside

        box = WorldBox.from_dict(step.params["box"])
        made, _ = (crop_layers if step.kind == "roi_crop" else mask_outside)(viewer, layers, box)
        return [str(l.name) for l in made]
    return []


def journal(viewer: Any) -> Journal:
    """The viewer's journal (created on first use)."""
    existing = getattr(viewer, "_nvitk_journal", None)
    if existing is None:
        existing = Journal()
        try:
            viewer._nvitk_journal = existing
        except Exception:  # noqa: BLE001
            pass
    return existing


def record(viewer: Any, kind: str, name: str, *, params: dict[str, Any] | None = None,
           inputs: list[str] | None = None, outputs: list[str] | None = None, label: str = "") -> None:
    """Append a step to *viewer*'s journal if it is recording (else nothing)."""
    try:
        if getattr(viewer, "_nvitk_replaying", False):
            return  # a replay's own actions are not new steps
        jr = journal(viewer)
        if jr.recording:
            jr.record(Step(kind, str(name), dict(params or {}), [str(n) for n in inputs or []],
                           [str(n) for n in outputs or []], label=label))
    except Exception:  # noqa: BLE001 (recording must never break the action it records)
        pass


__all__ = ["FORMAT", "Journal", "REPLAYABLE", "Step", "journal", "record"]
