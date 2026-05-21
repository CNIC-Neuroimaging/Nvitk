#!/usr/bin/env python3
"""List all CLI commands (pyhelp): hierarchical catalog with interactive expand/collapse."""

from __future__ import annotations

import sys
from pathlib import Path

import click

_project_root = Path(__file__).resolve().parents[3]
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from nvitk.core.click_backend import backend_click_option  # noqa: E402
from nvitk.cli.catalog import (  # noqa: E402
    build_catalog_tree,
    find_pyproject_toml,
    parse_pyproject_scripts,
    total_tool_count,
)
from nvitk.util.colors import bcolors as Colors  # noqa: E402
from nvitk.util.pyhelp_tree import (  # noqa: E402
    print_static_tree,
    run_interactive_pyhelp,
)


def _get_log():
    from nvitk.core.logger import Logger
    return Logger()


def categorize_command(cmd: str, module: str) -> str:
    if any(conv in cmd.lower() for conv in ("dcm2nii", "stl2nifti", "nikon2nifti", "phase2volume")):
        return "Image Conversion"
    if any(seg in cmd for seg in ("nvitk-totalseg", "nvitk-eicab")):
        return "Segmentation"
    if "nvitk-flirt" in cmd:
        return "Registration"
    if cmd.startswith("nvitk-pesa-fat"):
        return "PESA-Fat Analysis"
    if cmd.startswith("nvitk-qvtpy") or cmd.startswith("nvitk-bbtpy") or cmd.startswith("nvitk-gpetpy"):
        return "PESA-Brain Analysis"
    if cmd.startswith(("nvitk-morph", "nvitk-restore", "nvitk-filter", "nvitk-measure", "nvitk-transform")):
        return "Image Processing"
    return "General"


def get_command_color(category: str) -> str:
    color_map = {
        "Image Conversion": Colors.OKCYAN,
        "Segmentation": Colors.OKGREEN,
        "Registration": Colors.HEADER,
        "PESA-Fat Analysis": Colors.WARNING,
        "PESA-Brain Analysis": Colors.OKBLUE,
        "Image Processing": Colors.OKCYAN,
        "General": Colors.WHITE,
    }
    return color_map.get(category, Colors.WHITE)


def list_cli_commands_flat() -> None:
    log = _get_log()
    pyproject_path = find_pyproject_toml()
    if not pyproject_path:
        log.error("Error: pyproject.toml not found")
        return

    commands = parse_pyproject_scripts(pyproject_path)
    if not commands:
        log.warning("No CLI commands found")
        return

    categorized: dict[str, list[tuple[str, str]]] = {}
    for cmd, module in commands:
        category = categorize_command(cmd, module)
        categorized.setdefault(category, []).append((cmd, module))

    display_order = [
        "Image Conversion",
        "Segmentation",
        "Registration",
        "Image Processing",
        "PESA-Fat Analysis",
        "PESA-Brain Analysis",
        "General",
    ]

    log.info("=" * 80)
    log.info(f"{Colors.BOLD}{Colors.OKBLUE}Nvitk CLI Commands{Colors.ENDC}")
    log.info("=" * 80)

    total_commands = 0
    for category in display_order:
        if category not in categorized:
            continue
        items = categorized[category]
        total_commands += len(items)
        log.info(f"{Colors.BOLD}{category}{Colors.ENDC}")
        log.info("─" * len(category))
        cmd_color = get_command_color(category)
        for cmd, module in items:
            colored_cmd = f"{cmd_color}{Colors.BOLD}{cmd}{Colors.ENDC}"
            log.info(f"  {colored_cmd:<20} → {Colors.GRAY}{module}{Colors.ENDC}")
        log.info("")

    log.info("=" * 80)
    log.info(f"{Colors.BOLD}Total commands: {Colors.OKGREEN}{total_commands}{Colors.ENDC}")
    log.info("=" * 80)
    log.info(f"{Colors.BOLD}Backend Options:{Colors.ENDC}")
    log.info(f"   {Colors.WHITE}• --backend cpu|gpu  (array processing){Colors.ENDC}")
    log.info(f"   {Colors.WHITE}• --device cpu|gpu   (TotalSegmentator / eICAB){Colors.ENDC}")
    log.info("=" * 80)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@backend_click_option()
@click.option(
    "--no-interactive", "-n",
    is_flag=True,
    help="Print static expanded tree (for pipes/CI).",
)
@click.option(
    "--flat", "-f",
    is_flag=True,
    help="Legacy flat category listing via Logger.",
)
@click.option(
    "--shell", "-s",
    is_flag=True,
    help="Force bash readline assignments on stdout (for: eval \"$(pyhelp --shell 2>/dev/tty)\").",
)
@click.option(
    "--pick", "-p",
    is_flag=True,
    help="Print only the selected command to stdout (alias for capture mode).",
)
def main(no_interactive: bool, flat: bool, shell: bool, pick: bool) -> None:
    """List nvitk CLI commands organized by module."""
    if flat:
        list_cli_commands_flat()
        return

    scripts = parse_pyproject_scripts()
    roots = build_catalog_tree(scripts)
    installed = len(scripts)

    if sys.stdout.isatty() and not no_interactive:
        run_interactive_pyhelp(
            roots,
            total_cmds=installed,
            shell_mode=shell,
            pick_only=pick,
        )
    else:
        print_static_tree(roots, expanded=True, total_cmds=installed)


if __name__ == "__main__":
    main()
