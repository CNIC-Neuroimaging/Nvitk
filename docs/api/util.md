# Utilities

`nvitk.util` — small helpers shared across the codebase, exposed via a lazy proxy
(`nvitk.util.__init__`) to avoid import cycles with the modules that use them.

| Module | Purpose |
|---|---|
| `list_cli_commands` | Backs the `pyhelp` entry point. |
| `pyhelp_tree` | Builds the interactive/static tree `pyhelp` renders. |
| `colors` | `bcolors` — shared ANSI color codes for console output. |
| `lazy_timer` | A lightweight lazy-start timer used in a few CLIs' progress reporting. |

```{seealso}
Full generated reference: [`nvitk.util`](../autoapi/nvitk/util/index).
```
