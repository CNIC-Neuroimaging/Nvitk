"""Interactive pyhelp tree: Rich Live display with keyboard expand/collapse."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import termios
import tty
from dataclasses import dataclass
from typing import TextIO

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from nvitk.cli.catalog import CatalogNode, ToolEntry, total_tool_count

# Rich styles per catalog branch id
_BRANCH_STYLES: dict[str, str] = {
    "image_processing": "bold magenta",
    "conversion": "bold cyan",
    "segmentation": "bold green",
    "registration": "bold blue",
    "filters": "bold yellow",
    "morphology": "bold bright_red",
    "restoration": "bold bright_green",
    "measure": "bold bright_blue",
    "transform": "bold bright_magenta",
    "pipelines": "bold gold1",
    "general": "bold white",
}

_TOP_LEVEL_BORDER = {
    "image_processing": "bright_magenta",
    "pipelines": "gold1",
    "general": "bright_white",
}

# Static view: section panel border + header + command accent
_SECTION_THEME: dict[str, dict[str, str]] = {
    "image_processing": {
        "border": "bright_magenta",
        "header": "bold bright_magenta",
        "rule": "magenta",
    },
    "pipelines": {
        "border": "gold1",
        "header": "bold gold1",
        "rule": "yellow",
    },
    "general": {
        "border": "bright_white",
        "header": "bold bright_white",
        "rule": "white",
    },
}

_SUBMODULE_THEME: dict[str, dict[str, str]] = {
    "conversion": {"header": "bold cyan", "cmd": "cyan", "rule": "cyan"},
    "segmentation": {"header": "bold green", "cmd": "bright_green", "rule": "green"},
    "registration": {"header": "bold blue", "cmd": "bright_blue", "rule": "blue"},
    "filters": {"header": "bold yellow", "cmd": "yellow", "rule": "yellow3"},
    "morphology": {"header": "bold bright_red", "cmd": "bright_red", "rule": "red"},
    "restoration": {"header": "bold bright_green", "cmd": "green", "rule": "green"},
    "measure": {"header": "bold bright_blue", "cmd": "bright_cyan", "rule": "cyan"},
    "transform": {"header": "bold bright_magenta", "cmd": "magenta", "rule": "magenta"},
}


@dataclass
class VisibleRow:
    """One rendered line in the interactive view."""

    kind: str  # "branch" | "tool"
    node: CatalogNode | None = None
    tool: ToolEntry | None = None
    depth: int = 0
    label: str = ""


def flatten_visible(roots: list[CatalogNode]) -> list[VisibleRow]:
    rows: list[VisibleRow] = []

    def walk(node: CatalogNode, depth: int) -> None:
        rows.append(VisibleRow("branch", node=node, depth=depth, label=node.label))
        if not node.expanded:
            return
        for tool in node.tools:
            rows.append(
                VisibleRow(
                    "tool",
                    tool=tool,
                    depth=depth + 1,
                    label=_tool_label(tool),
                )
            )
        for child in node.children:
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return rows


def _tool_label(tool: ToolEntry) -> str:
    return tool.command or tool.display_label


def _prefix(expanded: bool) -> str:
    return "[bright_white][-][/]" if expanded else "[bright_white][+][/]"


def _indent(depth: int) -> str:
    return "  " * depth


def _branch_style(node: CatalogNode) -> str:
    return _BRANCH_STYLES.get(node.id, "bold white")


def _format_branch_row(node: CatalogNode, depth: int) -> str:
    style = _branch_style(node)
    icon = "▸ " if depth == 0 else "├ "
    return (
        f"{_indent(depth)}{_prefix(node.expanded)} {icon}"
        f"[{style}]{node.label}[/]"
    )


def _format_tool_row(tool: ToolEntry, depth: int) -> str:
    indent = _indent(depth)
    if tool.library_only or not tool.command:
        name = f"[italic dim]{tool.display_label}[/]"
        tag = " [dim yellow](library)[/]"
        mod = f"\n{indent}    [dim]↳ {tool.module}[/]" if tool.module else ""
        return f"{indent}  ○ {name}{tag}{mod}"
    cmd_style = "bold bright_cyan"
    if tool.command.startswith("nvitk-pesa") or tool.command.startswith("nvitk-qvt"):
        cmd_style = "bold gold1"
    elif tool.command in ("nvitk-ants", "nvitk-fireants", "nvitk-flirt"):
        cmd_style = "bold bright_blue"
    elif tool.command.startswith("nvitk-"):
        cmd_style = "bold bright_green"
    elif tool.command in ("dcm2nii", "stl2nifti", "phase2volume", "nikon2nifti"):
        cmd_style = "bold cyan"
    gpu = " [dim green]gpu[/]" if tool.supports_gpu else ""
    mask = " [dim magenta]mask[/]" if tool.requires_mask else ""
    mod = f" [dim]→ {tool.module}[/]" if tool.module else ""
    return f"{indent}  ● [{cmd_style}]{tool.command}[/]{gpu}{mask}{mod}"


def _build_table(roots: list[CatalogNode], cursor: int) -> Table:
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        expand=True,
        show_edge=False,
    )
    table.add_column("line", overflow="fold")
    rows = flatten_visible(roots)
    for idx, row in enumerate(rows):
        active = idx == cursor
        style = Style(reverse=True, bold=True) if active else Style()
        if row.kind == "branch" and row.node is not None:
            text = _format_branch_row(row.node, row.depth)
        elif row.tool is not None:
            text = _format_tool_row(row.tool, row.depth)
        else:
            text = row.label
        table.add_row(Text.from_markup(text, style=style))
    return table


def _footer_text(*, interactive: bool = True) -> str:
    keys = (
        "[bold bright_white]↑/↓[/] move  "
        "[bold bright_white]→/+[/] expand  "
        "[bold bright_white]←/-[/] collapse  "
        "[bold bright_white]Enter[/] "
        + ("select cmd (shows --help) · toggle branch  " if interactive else "toggle branch  ")
        + "[bold bright_white]a/z[/] all/none  "
        "[bold bright_white]q[/] quit"
    )
    backend = (
        "[dim]Array:[/] [cyan]--backend cpu|gpu[/]  "
        "[dim]External:[/] [cyan]--device cpu|gpu[/]"
    )
    shell = (
        "[dim]Shell line:[/] [yellow]eval \"$(pyhelp --shell)\"[/] or "
        "[yellow]source scripts/pyhelp-shell.bash[/]"
    )
    return f"{keys}\n{backend}\n{shell if interactive else ''}"


def _set_all_expanded(roots: list[CatalogNode], expanded: bool) -> None:
    def walk(node: CatalogNode) -> None:
        node.expanded = expanded
        for child in node.children:
            walk(child)

    for root in roots:
        walk(root)


def _collapse_all(roots: list[CatalogNode]) -> None:
    for root in roots:
        root.expanded = False
    if roots and roots[0].id == "image_processing":
        roots[0].expanded = True


def _expand_all(roots: list[CatalogNode]) -> None:
    _set_all_expanded(roots, True)


def _ui_console() -> Console:
    """Console for the interactive UI (stderr so stdout stays free for command capture)."""
    return Console(stderr=True, force_terminal=True, color_system="truecolor")


class _KeyReader:
    """Single-key reader with arrow-key decoding (Unix)."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old: list | None = None

    def __enter__(self) -> _KeyReader:
        if sys.stdin.isatty():
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *args: object) -> None:
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read_key(self) -> str:
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            rest = sys.stdin.read(2)
            if rest == "[A":
                return "UP"
            if rest == "[B":
                return "DOWN"
            if rest == "[C":
                return "RIGHT"
            if rest == "[D":
                return "LEFT"
            return "ESC"
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == " ":
            return "SPACE"
        return ch


def _panel_border(roots: list[CatalogNode]) -> str:
    if len(roots) == 1:
        return _TOP_LEVEL_BORDER.get(roots[0].id, "blue")
    return "bright_blue"


def _render_panel(
    roots: list[CatalogNode],
    cursor: int,
    total_cmds: int,
    *,
    interactive: bool,
    console: Console,
) -> Panel:
    table = _build_table(roots, cursor)
    header = Text("Nvitk CLI Commands", style="bold bright_cyan")
    count = Text.assemble(
        ("Installed scripts: ", "dim"),
        (str(total_cmds), "bold bright_green"),
    )
    body = Group(
        header,
        count,
        Text(""),
        table,
        Text(""),
        Text.from_markup(_footer_text(interactive=interactive)),
    )
    return Panel(
        body,
        title="[bold bright_blue]pyhelp[/]",
        border_style=_panel_border(roots),
        padding=(1, 2),
    )


def _default_shell_mode() -> bool:
    """Use bash readline injection when running under bash interactively."""
    shell = os.environ.get("SHELL", "")
    return "bash" in shell and sys.stdin.isatty()


def run_command_help(command: str, *, console: Console | None = None) -> int:
    """Run ``command --help`` after interactive selection. Returns process exit code."""
    cmd = command.strip()
    if not cmd:
        return 1
    out = console or _ui_console()
    exe = shutil.which(cmd) or cmd
    try:
        result = subprocess.run(
            [exe, "--help"],
            stdin=subprocess.DEVNULL,
        )
        return int(result.returncode or 0)
    except FileNotFoundError:
        out.print(f"[bold red]Command not found:[/] [cyan]{cmd}[/]")
        return 127
    except OSError as exc:
        out.print(f"[bold red]Could not run --help for[/] [cyan]{cmd}[/]: {exc}")
        return 1


def _emit_selected_command(command: str, *, shell_mode: bool) -> None:
    """Write selected command to stdout for shell capture / readline."""
    cmd = command.strip()
    if not cmd:
        return
    if shell_mode:
        # Safe for: eval "$(pyhelp --shell)"
        escaped = cmd.replace("\\", "\\\\").replace('"', '\\"')
        sys.stdout.write(f'READLINE_LINE="{escaped}"\n')
        sys.stdout.write(f"READLINE_POINT={len(cmd)}\n")
    else:
        sys.stdout.write(f"{cmd} ")
    sys.stdout.flush()


def run_interactive_pyhelp(
    roots: list[CatalogNode],
    *,
    total_cmds: int | None = None,
    shell_mode: bool = False,
    pick_only: bool = False,
) -> str | None:
    """
    Run interactive TUI. Returns selected command name or None.

    UI renders on /dev/tty (or stderr); selected command is written to stdout.
    """
    if not sys.stdin.isatty():
        print_static_tree(roots, expanded=True, total_cmds=total_cmds)
        return None

    if sys.platform == "win32":
        print_static_tree(roots, expanded=True, total_cmds=total_cmds)
        print("Interactive pyhelp requires a Unix TTY. Use WSL or --no-interactive.", file=sys.stderr)
        return None

    total = total_cmds if total_cmds is not None else total_tool_count(roots)
    cursor = 0
    selected: str | None = None
    ui_console = _ui_console()

    def apply_action(key: str) -> str | None:
        """Return 'quit', 'selected', or None to continue."""
        nonlocal cursor, selected
        rows = flatten_visible(roots)
        if not rows:
            return "quit" if key in ("q", "Q", "ESC") else None

        if key in ("UP", "k"):
            cursor = max(0, cursor - 1)
        elif key in ("DOWN", "j"):
            cursor = min(len(rows) - 1, cursor + 1)
        elif key in ("RIGHT", "+", "SPACE"):
            row = rows[cursor]
            if row.kind == "branch" and row.node is not None:
                row.node.expanded = True
        elif key in ("LEFT", "-"):
            row = rows[cursor]
            if row.kind == "branch" and row.node is not None:
                row.node.expanded = False
        elif key == "ENTER":
            row = rows[cursor]
            if row.kind == "tool" and row.tool is not None and row.tool.command:
                selected = row.tool.command
                return "selected"
            if row.kind == "branch" and row.node is not None:
                row.node.expanded = not row.node.expanded
        elif key in ("a", "A"):
            _expand_all(roots)
        elif key in ("z", "Z"):
            _collapse_all(roots)
        elif key in ("q", "Q", "ESC"):
            return "quit"
        return None

    def _run_live(*, use_screen: bool) -> None:
        nonlocal selected
        with Live(
            console=ui_console,
            refresh_per_second=12,
            screen=use_screen,
            transient=True,
        ) as live:
            with _KeyReader() as reader:
                while True:
                    live.update(
                        _render_panel(
                            roots, cursor, total, interactive=True, console=ui_console
                        )
                    )
                    key = reader.read_key()
                    result = apply_action(key)
                    if result == "selected":
                        break
                    if result == "quit":
                        selected = None
                        break

    try:
        _run_live(use_screen=True)
    except (ValueError, OSError):
        # Some terminals lack alt-screen support or reject stderr alt-screen.
        _run_live(use_screen=False)

    if selected:
        if pick_only:
            sys.stdout.write(f"{selected}\n")
            sys.stdout.flush()
        elif shell_mode:
            _emit_selected_command(selected, shell_mode=True)
        elif _default_shell_mode():
            _emit_selected_command(selected, shell_mode=True)
            ui_console.print(
                "[dim green]Selected[/] [bold cyan]"
                f"{selected}[/][dim green] — inject with:[/]\n"
                "  [yellow]eval \"$(pyhelp --shell 2>/dev/tty)\"[/]\n"
                "  [dim]or once:[/] [yellow]source scripts/pyhelp-shell.bash[/] "
                "[dim]then[/] [yellow]pyhelp-select[/]",
            )
        else:
            _emit_selected_command(selected, shell_mode=False)
            ui_console.print(
                f"[dim green]Selected:[/] [bold cyan]{selected}[/] "
                "[dim](printed to stdout)[/]",
            )
        ui_console.print()
        run_command_help(selected, console=ui_console)
    return selected


def _theme_for_root(node_id: str) -> dict[str, str]:
    return _SECTION_THEME.get(node_id, {
        "border": "blue",
        "header": "bold white",
        "rule": "dim",
    })


def _theme_for_submodule(node_id: str) -> dict[str, str]:
    return _SUBMODULE_THEME.get(node_id, {
        "header": "bold white",
        "cmd": "bright_white",
        "rule": "dim",
    })


def _format_static_tool_line(tool: ToolEntry, submodule_id: str) -> Text:
    """One compact tool line for the static catalog."""
    theme = _theme_for_submodule(submodule_id)

    if tool.library_only or not tool.command:
        return Text.assemble(
            ("    ", "dim"),
            (tool.display_label, "italic dim"),
            ("  library", "dim yellow"),
        )

    badges: list[str] = []
    if tool.supports_gpu:
        badges.append("[dim green]gpu[/]")
    if tool.requires_mask:
        badges.append("[dim magenta]mask[/]")
    badge_str = ("  " + "  ".join(badges)) if badges else ""
    return Text.from_markup(f"  [{theme['cmd']}]{tool.command}[/]{badge_str}")


def _render_static_submodule(node: CatalogNode) -> Group:
    """Submodule block: colored header + tool list."""
    theme = _theme_for_submodule(node.id)
    items: list[RenderableType] = [
        Text(node.label, style=theme["header"]),
        Rule(style=theme["rule"], characters="─"),
    ]
    if not node.tools:
        items.append(Text("  (no tools)", style="dim"))
    else:
        for tool in node.tools:
            items.append(_format_static_tool_line(tool, node.id))
    items.append(Text(""))
    return Group(*items)


def _render_static_pipeline_tools(tools: list[ToolEntry]) -> Group:
    """Group pipeline commands by cohort for readability."""
    pesa = [t for t in tools if "pesa-fat" in t.command]
    brain = [t for t in tools if t.command.startswith(("nvitk-qvtpy", "nvitk-bbtpy", "nvitk-gpetpy"))]
    other = [t for t in tools if t not in pesa and t not in brain]

    def block(label: str, items: list[ToolEntry], cmd_style: str) -> list[RenderableType]:
        if not items:
            return []
        out: list[RenderableType] = [
            Text(label, style="bold gold1"),
            Rule(style="yellow", characters="─"),
        ]
        for tool in items:
            markup = f"  [{cmd_style}]{tool.command}[/]"
            out.append(Text.from_markup(markup))
        out.append(Text(""))
        return out

    parts: list[RenderableType] = []
    parts.extend(block("PESA-Fat", pesa, "yellow"))
    parts.extend(block("PESA-Brain", brain, "bright_blue"))
    parts.extend(block("Other", other, "gold1"))
    return Group(*parts)


def _render_static_root_section(node: CatalogNode) -> Panel:
    """One top-level section (Image Processing, Pipelines, General)."""
    theme = _theme_for_root(node.id)
    body_parts: list[RenderableType] = []

    if node.children:
        for child in node.children:
            body_parts.append(_render_static_submodule(child))
    elif node.id == "pipelines" and node.tools:
        body_parts.append(_render_static_pipeline_tools(node.tools))
    elif node.tools:
        for tool in node.tools:
            body_parts.append(_format_static_tool_line(tool, node.id))
        body_parts.append(Text(""))

    count = len(node.tools) + sum(len(c.tools) for c in node.children)
    title = (
        f"[{theme['header']}]{node.label}[/] "
        f"[dim]({count} tool{'s' if count != 1 else ''})[/]"
    )

    return Panel(
        Group(*body_parts),
        title=title,
        border_style=theme["border"],
        padding=(0, 1),
    )


def _static_footer() -> Group:
    return Group(
        Rule(style="dim"),
        Text.from_markup(
            "[dim]Array processing:[/] [cyan]--backend cpu|gpu[/]   "
            "[dim]External engines:[/] [cyan]--device cpu|gpu[/]"
        ),
        Text.from_markup(
            "[dim]Interactive picker:[/] [yellow]pyhelp[/]   "
            "[dim]Flat list:[/] [yellow]pyhelp --flat[/]"
        ),
    )


def print_static_tree(
    roots: list[CatalogNode],
    *,
    expanded: bool = True,
    total_cmds: int | None = None,
    file: TextIO | None = None,
) -> None:
    """Print a color-grouped static catalog (``pyhelp --no-interactive``)."""
    if expanded:
        _expand_all(roots)
    console = Console(file=file, force_terminal=True, color_system="truecolor")
    installed = total_cmds if total_cmds is not None else len([
        t for r in roots for t in _iter_tools(r)
    ])

    console.print()
    console.print(
        Text.assemble(
            ("Nvitk CLI Commands", "bold bright_cyan"),
            ("\n", ""),
            (f"{installed} installed scripts", "bold bright_green"),
            ("  ·  ", "dim"),
            ("use ", "dim"),
            ("pyhelp", "yellow"),
            (" for interactive mode", "dim"),
        )
    )
    console.print()

    for root in roots:
        console.print(_render_static_root_section(root))

    console.print(_static_footer())
    console.print()


def _iter_tools(node: CatalogNode):
    """Yield all tools under a catalog node."""
    for tool in node.tools:
        yield tool
    for child in node.children:
        yield from _iter_tools(child)
