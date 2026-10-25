"""FreeSurfer tools of the Tools panel (category *FreeSurfer*), and FastSurfer's equivalents.

FreeSurfer's topology programs on any mask or surface (:mod:`nvitk.freesurfer.topology`):

- *Pretess* and *Marching cubes* on a label / mask layer;
- *Main component*, *Euler number*, *Smooth*, *Inflate* and *Quick sphere* on a Surface layer;
- *Fix topology* (``mris_fix_topology``) and *Topology fixer* (``mris_topo_fixer``) on a mask plus
  its image: the whole tessellation → sphere → fix chain, with the defects as a volume.

The category is listed only when a FreeSurfer installation is found (:func:`available_tool_ids`),
and each tool only when its programs are there. FastSurfer's own parts (category *FastSurfer*):

- *Spectral sphere* on a Surface layer (needs FastSurfer and lapy, no FreeSurfer);
- *Topology chain (recon-surf order)*: the fix chain with FastSurfer's spectral sphere and its
  orientation rewrite (needs FreeSurfer as well).

The layer is written to a run folder; outputs open as layers placed by their own geometry
(surfaces in scanner RAS, volumes by their affine). Short programs run in the foreground; the
sphere and the fix chains run as background jobs (Jobs window). FreeSurfer refuses to run
without a license: the message says where to get one and where to put it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.gui.tools.external_io import ViewerInputs, manifest_layer

#: Tool id → the FreeSurfer programs it needs.
TOOL_PROGRAMS: dict[str, tuple[str, ...]] = {
    "freesurfer_pretess": ("mri_pretess",),
    "freesurfer_mc": ("mri_mc",),
    "freesurfer_main_component": ("mris_extract_main_component",),
    "freesurfer_euler": ("mris_euler_number",),
    "freesurfer_smooth": ("mris_smooth",),
    "freesurfer_inflate": ("mris_inflate",),
    "freesurfer_qsphere": ("mris_sphere",),
    "freesurfer_fix_topology": ("mri_pretess", "mri_mc", "mris_extract_main_component", "mris_smooth",
                                "mris_inflate", "mris_sphere", "mris_fix_topology", "mris_remove_intersection"),
    "freesurfer_topo_fixer": ("mri_pretess", "mri_mc", "mris_extract_main_component", "mris_smooth",
                              "mris_inflate", "mris_sphere", "mris_topo_fixer", "mris_remove_intersection"),
    "fastsurfer_topology_chain": ("mri_pretess", "mri_mc", "mris_extract_main_component", "mris_smooth",
                                  "mris_inflate", "mris_fix_topology", "mris_remove_intersection"),
}
#: FastSurfer-category tools implemented here.
FASTSURFER_TOPOLOGY_IDS: frozenset[str] = frozenset({"fastsurfer_spectral_sphere", "fastsurfer_topology_chain"})
FREESURFER_TOOL_IDS: frozenset[str] = frozenset(k for k in TOOL_PROGRAMS if k.startswith("freesurfer_"))
_SURFACE_INPUT = {"freesurfer_main_component", "freesurfer_euler", "freesurfer_smooth", "freesurfer_inflate",
                  "freesurfer_qsphere", "fastsurfer_spectral_sphere"}
_CHAIN = {"freesurfer_fix_topology", "freesurfer_topo_fixer", "fastsurfer_topology_chain"}


def _notify(message: str, *, error: bool = False) -> None:
    from nvitk.gui.tools.runner import notify

    notify(message, error=error)


def fastsurfer_spectral_available() -> bool:
    """FastSurfer is installed and ``lapy`` imports in this environment."""
    import importlib.util

    try:
        from nvitk.segmentation.fastsurfer.config import fastsurfer_home

        return fastsurfer_home() is not None and importlib.util.find_spec("lapy") is not None
    except Exception:  # noqa: BLE001
        return False


def available_tool_ids() -> set[str]:
    """The tools of this module that can run here (FreeSurfer, its programs, FastSurfer + lapy)."""
    from nvitk.freesurfer import env

    out: set[str] = set()
    home = env.freesurfer_home()
    if home is not None:
        out |= {t for t, progs in TOOL_PROGRAMS.items() if t.startswith("freesurfer_") and not env.missing(progs, home)}
    if fastsurfer_spectral_available():
        out.add("fastsurfer_spectral_sphere")
        if home is not None and not env.missing(TOOL_PROGRAMS["fastsurfer_topology_chain"], home):
            out.add("fastsurfer_topology_chain")
    return out


def default_license() -> str:
    try:
        from nvitk.freesurfer import env

        lic = env.fs_license()
        return str(lic) if lic else ""
    except Exception:  # noqa: BLE001
        return ""


# ── inputs ───────────────────────────────────────────────────────────────────


def _surface_of(layer: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices in world / scanner-RAS mm, faces)`` of a Surface layer."""
    if type(layer).__name__ != "Surface":
        raise ValueError("Select a Surface layer (e.g. from Marching cubes, FastSurfer or the Meshlab panel).")
    vertices = np.asarray(layer.data[0], dtype=np.float64)
    faces = np.asarray(layer.data[1], dtype=np.int64)
    try:
        vertices = np.asarray(layer._data_to_world(vertices), dtype=np.float64)
    except Exception:  # noqa: BLE001 — no transform: data coordinates are world
        pass
    return vertices, faces


def _mask_volume(inputs: ViewerInputs, label: int) -> tuple[np.ndarray, np.ndarray]:
    """The active layer (written in file order) as a binary mask of *label* (0 = every nonzero
    voxel), and its affine."""
    import nibabel as nib

    img = nib.load(str(inputs.input()))
    data = np.asarray(img.dataobj)
    if data.ndim > 3:
        data = data[..., 0]
    m = data != 0 if int(label) == 0 else np.rint(data) == int(label)
    if not m.any():
        raise ValueError(f"Label {label} is empty in '{inputs.display_name}'.")
    return m, img.affine


def _reference(inputs: ViewerInputs) -> np.ndarray | None:
    import nibabel as nib

    path = inputs.layer_file("reference_layer", required=False)
    if path is None:
        return None
    data = np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)
    return data[..., 0] if data.ndim > 3 else data


# ── running ──────────────────────────────────────────────────────────────────


def _license(params: dict[str, Any]) -> Path | None:
    raw = str(params.get("freesurfer_license") or "").strip()
    return Path(raw).expanduser() if raw else None


def _check_license(text: str) -> None:
    from nvitk.freesurfer import env

    problem = env.license_problem(text)
    if problem:
        raise RuntimeError(problem)


def _run_now(cmd: Any) -> str:
    from nvitk.gui.core.log_panel import run_subprocess_logged

    lines: list[str] = []
    code = run_subprocess_logged(cmd.full_argv(), cwd=str(cmd.cwd) if cmd.cwd else None, env=cmd.environment(),
                                 on_line=lines.append)
    text = "\n".join(lines)
    _check_license(text)
    if code != 0:
        raise RuntimeError(f"{cmd.title} failed (exit code {code}); see the log.")
    cmd.stdout = text
    return text


def _manifest(cmd: Any, work: Path, titles: dict[str, str]) -> dict[str, Any]:
    layers, surfaces = [], []
    for role, path in cmd.outputs.items():
        p = Path(path)
        if not p.exists() or role not in titles:
            continue
        if role in cmd.surface_outputs:
            surfaces.append({"file": str(p), "title": titles[role]})
        else:
            layers.append(manifest_layer(p, titles[role], base=work, labels=role in cmd.label_outputs))
    return {"layers": layers, "surfaces": surfaces}


def run_freesurfer_tool(tool_id: str, viewer: Any, layer: Any, params: dict[str, Any]) -> None:
    """Run one FreeSurfer / FastSurfer topology tool of the Tools panel; outputs go to *viewer*."""
    from nvitk.freesurfer import env
    from nvitk.freesurfer import topology as topo
    from nvitk.gui.tools.external_io import load_manifest

    if tool_id not in TOOL_PROGRAMS and tool_id not in FASTSURFER_TOPOLOGY_IDS:
        raise NotImplementedError(f"Tool {tool_id!r} is not implemented.")
    inputs = ViewerInputs(tool_id, viewer, layer, params)
    work = inputs.work
    lic = _license(params)
    name = inputs.name
    label = int(params.get("freesurfer_label") or 0)

    if tool_id in _SURFACE_INPUT:
        v, f = _surface_of(layer)
        src = topo.write_surface(work / f"lh.{name}", v, f)
        if tool_id == "freesurfer_euler":
            own = topo.surface_euler(v, f)
            row: dict[str, Any] = {"nvitk euler": own["euler"], "pieces": own["pieces"],
                                   "genus (largest piece)": own["genus_main"]}
            cmd = topo.euler_number(src)
            cmd.license_file = lic
            note = ""
            try:
                fs = topo.parse_euler(_run_now(cmd))
                row.update({"FreeSurfer euler": fs.get("euler"), "holes": fs.get("holes"),
                            "defect index": fs.get("defect_index")})
            except RuntimeError as exc:  # no license: nvitk's count still stands
                note = str(exc)
            from nvitk.gui.tools.runner import _notify_results

            _notify_results(viewer, "Euler number (FreeSurfer)", row, subtitle=getattr(layer, "name", ""), note=note)
            return
        out = work / f"lh.{name}.{tool_id.split('_', 1)[1]}"
        cmd = {
            "freesurfer_main_component": lambda: topo.main_component(src, out),
            "freesurfer_smooth": lambda: topo.smooth(src, out, iterations=int(params.get("freesurfer_iterations") or 10)),
            "freesurfer_inflate": lambda: topo.inflate(src, out),
            "freesurfer_qsphere": lambda: topo.quick_sphere(src, out),
            "fastsurfer_spectral_sphere": lambda: topo.spectral_sphere(src, out),
        }[tool_id]()
        cmd.license_file = lic
        titles = {"surface": f"{getattr(layer, 'name', name)} {cmd.title}"}
    elif tool_id in ("freesurfer_pretess", "freesurfer_mc"):
        m, aff = _mask_volume(inputs, label)
        mgz = topo.write_volume(m, aff, work / f"{name}_mask.mgz")
        if tool_id == "freesurfer_pretess":
            ref = _reference(inputs)
            norm = topo.write_volume(topo.scaled_intensity(ref, m), aff, work / f"{name}_norm.mgz") if ref is not None else mgz
            cmd = topo.pretess(mgz, 1, norm, work / f"{name}_pretess.mgz")
            titles = {"volume": f"{inputs.display_name} pretess"}
        else:
            cmd = topo.marching_cubes(mgz, 1, work / f"lh.{name}.mc", connectivity=int(params.get("freesurfer_connectivity") or 1))
            titles = {"surface": f"{inputs.display_name} mri_mc"}
        cmd.license_file = lic
    else:  # the fix chains
        m, aff = _mask_volume(inputs, label)
        subject = "nvitk_topology"
        topo.prepare_subject(m, aff, work, subject, intensity=_reference(inputs))
        fastsurfer = tool_id == "fastsurfer_topology_chain"
        cmd = topo.fix_topology(
            work, subject,
            fixer="mris_topo_fixer" if tool_id == "freesurfer_topo_fixer" else "mris_fix_topology",
            sphere="spectral" if fastsurfer else str(params.get("freesurfer_sphere") or "qsphere"),
            defect_seg=bool(params.get("freesurfer_defect_seg", True)) and env.which("defect-seg") is not None,
            rewrite_orientation=fastsurfer, license_file=lic,
        )
        titles = {"nofix": f"{inputs.display_name} orig.nofix", "fixed": f"{inputs.display_name} topology-fixed",
                  "pretess": f"{inputs.display_name} pretess", "defects": f"{inputs.display_name} defects"}

    if not cmd.long:
        _run_now(cmd)
        n = load_manifest(viewer, _manifest(cmd, work, titles), work)
        _notify(f"{cmd.title}: {n} layer(s) from {work}.")
        return

    from nvitk.gui.core.external_jobs import start_external_job

    log_file = work / "nvitk-job.log"

    def finished(code: int) -> None:
        text = log_file.read_text(errors="replace") if log_file.is_file() else ""
        problem = env.license_problem(text)
        if problem:
            _notify(problem, error=True)
            return
        if code != 0:
            _notify(f"{cmd.title} failed (exit code {code}); see the log in {work}.", error=True)
            return
        n = load_manifest(viewer, _manifest(cmd, work, titles), work)
        if tool_id in _CHAIN:
            from nvitk.gui.tools.runner import _notify_results

            report = topo.topology_report(work, "nvitk_topology",
                                          fixer="mris_topo_fixer" if tool_id == "freesurfer_topo_fixer" else "mris_fix_topology")
            flat = {}
            for key, val in report.items():
                for k2, v2 in (val.items() if isinstance(val, dict) else [("", val)]):
                    flat[f"{key} {k2}".strip()] = v2
            _notify_results(viewer, cmd.title, flat, subtitle=inputs.display_name, note=str(work))
        _notify(f"{cmd.title} finished: {n} layer(s) from {work}.")

    start_external_job(cmd.full_argv(), title=f"{cmd.title} · {inputs.display_name}", env=cmd.environment(),
                       cwd=cmd.cwd, on_finished=finished, log_file=log_file)
    _notify(f"{cmd.title} started in the background (Jobs window); outputs go to {work}.")


__all__ = [
    "FASTSURFER_TOPOLOGY_IDS",
    "FREESURFER_TOOL_IDS",
    "TOOL_PROGRAMS",
    "available_tool_ids",
    "default_license",
    "fastsurfer_spectral_available",
    "run_freesurfer_tool",
]
