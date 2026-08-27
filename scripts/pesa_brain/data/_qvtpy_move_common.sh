#!/usr/bin/env bash
# Shared helpers for qvtpy SGE rsync move scripts.

set -euo pipefail

RSYNC_FLAGS=(-avhP --progress --no-perms --ignore-times)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage_die() {
  echo "$*" >&2
  echo "Run with --help for usage." >&2
  exit 1
}

is_remote_path() {
  [[ "$1" == *:* ]]
}

remote_host_from_path() {
  local path="$1"
  if is_remote_path "$path"; then
    echo "${path%%:*}"
  fi
}

remote_path_from_path() {
  local path="$1"
  if is_remote_path "$path"; then
    echo "${path#*:}"
  else
    echo "$path"
  fi
}

ensure_dir() {
  local target="$1"
  if is_remote_path "$target"; then
    local host path
    host="$(remote_host_from_path "$target")"
    path="$(remote_path_from_path "$target")"
    ssh "$host" "mkdir -p $(printf '%q' "$path")"
  else
    mkdir -p "$target"
  fi
}

read_subjects_csv() {
  local csv="$1"
  [[ -f "$csv" ]] || die "Subjects CSV not found: $csv"
  python3 - "$csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    if not reader.fieldnames:
        sys.exit(0)
    norm = {str(c).strip().lower(): c for c in reader.fieldnames}
    key = None
    for candidate in ("subject_id", "subject", "subject_uid", "pesa", "pesa_id", "id"):
        if candidate in norm:
            key = norm[candidate]
            break
    if key is None:
        key = reader.fieldnames[0]
    seen = set()
    for row in reader:
        value = str(row.get(key, "")).strip().strip('"')
        if value and value not in seen:
            seen.add(value)
            print(value)
PY
}

run_rsync() {
  local dry_run="${DRY_RUN:-0}"
  local -a cmd=(rsync "${RSYNC_FLAGS[@]}")
  if [[ "$dry_run" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  cmd+=("$@")
  echo "+ ${cmd[*]}"
  "${cmd[@]}"
}
