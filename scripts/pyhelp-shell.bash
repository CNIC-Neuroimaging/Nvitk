# Shell integration: put the pyhelp-selected command on the current readline line.
#
# Usage:
#   source scripts/pyhelp-shell.bash
#   pyhelp-select          # pick a command → appears on your prompt
#
# Or wrap pyhelp so every run can inject into readline (bash only):
#   alias pyhelp='eval "$(command pyhelp --shell 2>/dev/tty)"'

pyhelp-select() {
  eval "$(command pyhelp --shell 2>/dev/tty)"
}
