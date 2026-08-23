"""CLI tool catalog: hierarchical tree for pyhelp, module CLIs, and nvitk-gui."""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class ToolEntry:
    """One installable command or library-only tool reference."""

    command: str
    module: str
    label: str = ""
    supports_gpu: bool = False
    requires_mask: bool = False
    library_only: bool = False

    @property
    def display_label(self) -> str:
        """Human-facing label: explicit ``label``, else the command, else the trailing module segment."""
        return self.label or self.command or self.module.rsplit(".", 1)[-1]


@dataclass
class CatalogNode:
    """Branch or leaf group in the pyhelp tree."""

    id: str
    label: str
    children: list[CatalogNode] = field(default_factory=list)
    tools: list[ToolEntry] = field(default_factory=list)
    expanded: bool = False

    def is_branch(self) -> bool:
        """True if this node has children nodes or tools to display (as opposed to being empty)."""
        return bool(self.children) or bool(self.tools)


# Library-only tools shown under each submodule (not yet separate pyproject scripts).
_LIBRARY_TOOLS: dict[str, list[ToolEntry]] = {
    "filters": [
        ToolEntry("", "nvitk.filters.sliding_threshold", label="sliding-threshold (3d/2d)", supports_gpu=True, library_only=True),
        ToolEntry("", "nvitk.filters.hessian", label="hessian (skimage ridge / vessel filter)", library_only=True),
        ToolEntry("", "nvitk.filters.jerman", label="jerman (vesselness / ridge filter)", library_only=True),
        ToolEntry("", "nvitk.filters.snakes", label="snakes (Kass active contours)", library_only=True, requires_mask=True),
    ],

    "morphology": [
        ToolEntry("", "nvitk.morphology.binary", label="dilate / erode / open / close / fill-holes", supports_gpu=True, requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.morphology.components", label="label-cc, remove-small-components", supports_gpu=True, requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.morphology.centerline", label="centerline / skeletonize", requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.morphology.centerline_siphon", label="ICA siphon correction", requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.morphology.polyline_graph", label="skeleton → polyline / junction graph", requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.morphology.mst_bridge", label="MST bridge of nearby components", supports_gpu=True, requires_mask=True, library_only=True),
    ],
    "restoration": [
        ToolEntry("", "nvitk.restoration.bilateral", label="bilateral 2d/3d", supports_gpu=True, library_only=True),
        ToolEntry("", "nvitk.restoration.n4_bias", label="N4 bias field correction (ANTs)", library_only=True),
        ToolEntry("", "nvitk.restoration.mri_super_resolution", label="MRI super-resolution (ANTsPyNet)", library_only=True),
    ],
    "segmentation": [
        ToolEntry("", "nvitk.segmentation.mouse_brain", label="mouse brain extraction/parcellation (ANTsPyNet)", library_only=True),
        ToolEntry("", "nvitk.segmentation.brain_extraction", label="multi-modal brain extraction (ANTsPyNet)", library_only=True),
        ToolEntry("", "nvitk.segmentation.mra_vessel", label="MRA-TOF vessel segmentation (ANTsPyNet)", library_only=True),
        ToolEntry("", "nvitk.segmentation.dkt", label="Desikan-Killiany-Tourville parcellation (ANTsPyNet)", library_only=True),
        ToolEntry(
            "",
            "nvitk.segmentation.blood_flood",
            label="blood flood (expand + from-scratch vessel tree)",
            library_only=True,
        ),
        ToolEntry("", "nvitk.segmentation.region_growing", label="6-connected intensity region growing", supports_gpu=True, requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.segmentation.mask_ops", label="mask logical ops / apply intensity", supports_gpu=True, requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.segmentation.labels", label="label-map primitives (get / largest-cc / combine)", supports_gpu=True, requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.segmentation.hemisphere", label="split bilateral mask into L/R", requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.segmentation.hull_edt", label="convex hull / distance transform", supports_gpu=True, requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.segmentation.protrusion_filter", label="curvature protrusion (wart) removal", requires_mask=True, library_only=True),
    ],
    "measure": [
        ToolEntry("", "nvitk.measure.volume", label="volume", library_only=True),
        ToolEntry("", "nvitk.measure.suv", label="suv", library_only=True),
        ToolEntry("", "nvitk.measure.voxel", label="dice / jaccard / overlap", library_only=True),
        ToolEntry("", "nvitk.measure.surface", label="surface metrics", library_only=True),
        ToolEntry("", "nvitk.measure.measurer", label="Measurer chain", library_only=True),
        ToolEntry("", "nvitk.measure.intensity", label="masked intensity statistics", requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.measure.hemodynamics", label="4D-flow hemodynamic indices", requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.measure.cross_section", label="oblique cross-section sampling", requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.measure.morphometrics", label="CoW TOF morphometrics", requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.measure.radiomics", label="PyRadiomics wrapper", requires_mask=True, library_only=True),
        ToolEntry("", "nvitk.measure.compare", label="intensity correlation (Pearson / Spearman / RMSE)", library_only=True),
        ToolEntry("", "nvitk.measure.voxelwise", label="voxelwise GLM / permutation tests (FSL randomise)", library_only=True),
    ],
    "transform": [
        ToolEntry("", "nvitk.transform.resampling", label="resample-to", library_only=True),
        ToolEntry("", "nvitk.transform.isotropy", label="isotropy", library_only=True),
        ToolEntry("", "nvitk.transform.oblique", label="oblique-slice", library_only=True),
        ToolEntry("", "nvitk.transform.rotate", label="rotate volume", library_only=True),
        ToolEntry("", "nvitk.transform.swap_axes", label="swap / permute axes", library_only=True),
        ToolEntry("", "nvitk.transform.rotation", label="z-rotation correction", library_only=True),
        ToolEntry("", "nvitk.transform.reorient", label="reorient (axis permute / flips / codes)", library_only=True),
        ToolEntry("", "nvitk.transform.projection", label="intensity projection along an axis", library_only=True),
    ],
    "stats": [
        ToolEntry("", "nvitk.stats.mixedlm", label="mixed-effects models (statsmodels)", library_only=True),
        ToolEntry("", "nvitk.stats.mediation", label="mediation analysis", library_only=True),
        ToolEntry("", "nvitk.stats.violin_hemodynamics", label="cohort hemodynamics violin / strip plots", library_only=True),
    ],
    "viz": [
        ToolEntry("", "nvitk.viz.flowshow", label="4D-flow visualization (PyVista)", library_only=True),
        ToolEntry("", "nvitk.viz.streamlines", label="headless streamlines / pathlines", library_only=True),
        ToolEntry("", "nvitk.viz.brainshow", label="atlas brain views (Nilearn)", library_only=True),
        ToolEntry("", "nvitk.viz.pet_hotspots", label="SUV hotspot 3D visualization", library_only=True),
        ToolEntry("", "nvitk.viz.centerline_pick", label="centerline picking / plane frames", requires_mask=True, library_only=True),
    ],
}

# Map pyproject command names → catalog submodule id under image_processing.
_CMD_TO_SUBMODULE: dict[str, str] = {
    "dcm2nii": "conversion",
    "stl2nifti": "conversion",
    "phase2volume": "conversion",
    "nikon2nifti": "conversion",
    "nvitk-totalseg": "segmentation",
    "nvitk-eicab": "segmentation",
    "nvitk-seg": "segmentation",
    "nvitk-flirt": "registration",
    "nvitk-ants": "registration",
    "nvitk-fireants": "registration",
    "nvitk-morph": "morphology",
    "nvitk-restore": "restoration",
    "nvitk-filter": "filters",
    "nvitk-measure": "measure",
    "nvitk-transform": "transform",
    "nvitk-voxelwise": "measure",
}

_GPU_COMMANDS = frozenset({
    "phase2volume",
    "nvitk-morph",
    "nvitk-restore",
    "nvitk-filter",
    "nvitk-measure",
    "nvitk-transform",
})

_MASK_COMMANDS = frozenset({"nvitk-morph", "nvitk-measure"})


def find_pyproject_toml() -> Path | None:
    """Locate ``pyproject.toml`` by walking up from the current working directory, then from this
    file's own directory; ``None`` if neither search finds it."""
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        p = candidate / "pyproject.toml"
        if p.is_file():
            return p
    file_dir = Path(__file__).resolve().parent
    for anc in [file_dir, *file_dir.parents]:
        p = anc / "pyproject.toml"
        if p.is_file():
            return p
    return None


def parse_pyproject_scripts(pyproject_path: Path | None = None) -> list[tuple[str, str]]:
    """Parse the ``[project.scripts]`` table of *pyproject_path* (or the discovered pyproject.toml)
    into a list of ``(command, module)`` pairs, via a regex scan rather than a full TOML parse."""
    path = pyproject_path or find_pyproject_toml()
    if path is None:
        return []
    content = path.read_text(encoding="utf-8")
    scripts_match = re.search(r"\[project\.scripts\](.*?)(?=\[|$)", content, re.DOTALL)
    if not scripts_match:
        return []
    commands: list[tuple[str, str]] = []
    for line in scripts_match.group(1).strip().split("\n"):
        if "=" not in line or line.strip().startswith("#"):
            continue
        cmd, module = line.split("=", 1)
        commands.append((cmd.strip(), module.strip().strip('"')))
    return commands


def _tool_from_script(cmd: str, module: str) -> ToolEntry:
    """Build a :class:`ToolEntry` for an installed pyproject script, inferring GPU/mask support from
    the known ``_GPU_COMMANDS``/``_MASK_COMMANDS`` sets."""
    return ToolEntry(
        command=cmd,
        module=module,
        supports_gpu=cmd in _GPU_COMMANDS or "phase2volume" in cmd,
        requires_mask=cmd in _MASK_COMMANDS,
    )


def _pipeline_tools(scripts: Iterable[tuple[str, str]]) -> list[ToolEntry]:
    """Tool entries for scripts belonging to a research pipeline (``nvitk-pesa-fat``/``nvitk-qvtpy``/
    ``nvitk-bbtpy``/``nvitk-gpetpy`` prefixes), sorted by command."""
    tools: list[ToolEntry] = []
    for cmd, module in scripts:
        if cmd.startswith("nvitk-pesa-fat") or cmd.startswith("nvitk-qvtpy") or cmd.startswith("nvitk-bbtpy") or cmd.startswith("nvitk-gpetpy"):
            tools.append(_tool_from_script(cmd, module))
    return sorted(tools, key=lambda t: t.command)


def _general_tools(scripts: Iterable[tuple[str, str]]) -> list[ToolEntry]:
    """Tool entries for scripts not claimed by an image-processing submodule or a pipeline prefix,
    sorted by command."""
    known = set(_CMD_TO_SUBMODULE) | {
        c for c, _ in scripts
        if c.startswith("nvitk-pesa-fat") or c.startswith("nvitk-qvtpy")
        or c.startswith("nvitk-bbtpy") or c.startswith("nvitk-gpetpy")
    }
    tools: list[ToolEntry] = []
    for cmd, module in scripts:
        if cmd not in known:
            tools.append(_tool_from_script(cmd, module))
    return sorted(tools, key=lambda t: t.command)


def discover_installed_scripts(distribution_name: str = "nvitk") -> list[tuple[str, str]]:
    """Read *distribution_name*'s registered ``console_scripts`` entry points from installed
    package metadata — works for any install method (conda, pip, an editable/dev checkout)
    since ``entry_points.txt`` is always written at install time, unlike
    :func:`parse_pyproject_scripts`, which needs a source ``pyproject.toml`` on disk and
    finds none in a real (non-source-checkout) installed environment."""
    try:
        dist = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return []
    return [(ep.name, ep.value) for ep in dist.entry_points if ep.group == "console_scripts"]


def discover_scripts() -> list[tuple[str, str]]:
    """Resolve nvitk's registered CLI commands: prefer installed package metadata (accurate
    for any install method), falling back to parsing ``pyproject.toml`` directly only if that
    finds nothing (e.g. running against a source tree with no install at all)."""
    scripts = discover_installed_scripts()
    if scripts:
        return scripts
    return parse_pyproject_scripts()


def build_catalog_tree(scripts: list[tuple[str, str]] | None = None) -> list[CatalogNode]:
    """Build top-level catalog nodes (Image Processing, Pipelines, General)."""
    if scripts is None:
        scripts = discover_scripts()

    submodules = {
        "conversion": CatalogNode("conversion", "conversion"),
        "segmentation": CatalogNode("segmentation", "segmentation"),
        "registration": CatalogNode("registration", "registration"),
        "filters": CatalogNode("filters", "filters"),
        "morphology": CatalogNode("morphology", "morphology"),
        "restoration": CatalogNode("restoration", "restoration"),
        "measure": CatalogNode("measure", "measure"),
        "transform": CatalogNode("transform", "transform"),
        "stats": CatalogNode("stats", "stats"),
        "viz": CatalogNode("viz", "viz"),
    }

    for cmd, module in scripts:
        sub_id = _CMD_TO_SUBMODULE.get(cmd)
        if sub_id and sub_id in submodules:
            submodules[sub_id].tools.append(_tool_from_script(cmd, module))

    for sub_id, lib_tools in _LIBRARY_TOOLS.items():
        existing_cmds = {t.command for t in submodules[sub_id].tools}
        for entry in lib_tools:
            if entry.command and entry.command in existing_cmds:
                continue
            submodules[sub_id].tools.append(entry)

    for node in submodules.values():
        node.tools.sort(key=lambda t: (t.library_only, t.command or t.display_label))

    image_children = [
        submodules[k]
        for k in (
            "conversion",
            "segmentation",
            "registration",
            "filters",
            "morphology",
            "restoration",
            "measure",
            "transform",
            "stats",
            "viz",
        )
    ]

    image_processing = CatalogNode(
        "image_processing",
        "Image Processing",
        children=image_children,
        expanded=True,
    )
    for child in image_children:
        child.expanded = False

    pipelines = CatalogNode(
        "pipelines",
        "Pipelines",
        tools=_pipeline_tools(scripts),
        expanded=False,
    )
    general_tools = _general_tools(scripts)
    if not any(t.command == "pyhelp" for t in general_tools):
        general_tools.insert(0, ToolEntry("pyhelp", "nvitk.util.list_cli_commands:main"))
    general = CatalogNode("general", "General", tools=general_tools, expanded=False)

    return [image_processing, pipelines, general]


def total_tool_count(roots: list[CatalogNode]) -> int:
    """Total number of tools across *roots* and all their descendant nodes."""
    count = 0

    def walk(node: CatalogNode) -> None:
        """Add *node*'s own tool count to the running total, then recurse into its children."""
        nonlocal count
        count += len(node.tools)
        for child in node.children:
            walk(child)

    for root in roots:
        walk(root)
    return count
