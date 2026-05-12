#!/usr/bin/env bash
# Usage:
#   ./eicab_clean.sh <eicab_root> [--apply] [--subdir <name>] [--keep-aux]
#
# Walks every PESA* subject under <eicab_root> and prunes its eICAB output
# folder so that only the following NIfTIs remain:
#   - Circle-of-Willis segmentation (*eICAB_CW*.nii[.gz])
#   - Whole-brain segmentation (*eICAB_WB* / *eICAB_WHOLE* / *whole_brain* /
#     *TOF_WB* / *_WB_*.nii[.gz]) if present
#   - Resampled inputs (*_resampled.nii[.gz])
#
# Auxiliary artifacts are removed (legacy subdirs original_space/, nn_space/,
# metric_space/, plus .csv/.txt/.log everywhere). Non-kept NIfTIs are deleted.
# Empty directories are cleaned afterwards.
#
# Mirrors src/nvitk/segmentation/eicab/runner.py::prune_eicab_outputs.
#
# Default mode is DRY-RUN (prints what would be removed). Pass --apply to act.

set -euo pipefail

ROOT=""
APPLY=0
KEEP_AUX=0
SUBDIR=""   # if set, look at <subj>/<SUBDIR>/ instead of <subj>/

usage() {
    cat <<EOF
Usage: $0 <eicab_root> [--apply] [--subdir <name>] [--keep-aux]
  <eicab_root>    Directory containing PESA* subject folders.
  --apply         Actually delete files (default: dry-run, prints actions only).
  --subdir NAME   Operate on <subj>/NAME instead of <subj> directly (e.g. "eicab").
                  If omitted, the script auto-detects: uses <subj>/eicab if it
                  exists, otherwise <subj>.
  --keep-aux      Skip the prune entirely (same as runner.py keep_aux_outputs).
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --apply)     APPLY=1; shift ;;
        --keep-aux)  KEEP_AUX=1; shift ;;
        --subdir)    SUBDIR="${2:-}"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        -*)
            echo "Unknown option: $1" >&2; usage; exit 2 ;;
        *)
            if [[ -z "$ROOT" ]]; then ROOT="$1"; shift
            else echo "Unexpected argument: $1" >&2; exit 2
            fi ;;
    esac
done

if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
    usage; exit 1
fi

ACTION="(dry-run)"
(( APPLY == 1 )) && ACTION="(apply)"

shopt -s nullglob nocaseglob
subjects=( "$ROOT"/PESA*/ )
shopt -u nocaseglob
if (( ${#subjects[@]} == 0 )); then
    echo "No PESA* subject folders found under: $ROOT" >&2
    exit 2
fi

# --- helpers -----------------------------------------------------------------

_is_nifti() {
    local n="${1,,}"
    [[ "$n" == *.nii || "$n" == *.nii.gz ]]
}

_nifti_stem() {
    local n="$1"
    case "${n,,}" in
        *.nii.gz) echo "${n%.nii.gz}" ;;
        *.nii)    echo "${n%.nii}" ;;
        *)        echo "$n" ;;
    esac
}

_should_keep_nifti() {
    # Return 0 (true) if the NIfTI at $1 must be kept.
    local p="$1"
    local name; name="$(basename "$p")"
    local lname="${name,,}"
    local stem; stem="$(_nifti_stem "$name")"
    local lstem="${stem,,}"

    # 1) *_resampled stem -> keep (covers TOF_resampled.nii / *_resampled.nii.gz)
    [[ "$lstem" == *_resampled ]] && return 0

    # 2) Any other NIfTI containing "resampled" (not at stem end) -> drop.
    [[ "$lname" == *resampled* ]] && return 1

    # 3) CoW segmentation -> keep
    [[ "$lname" == *eicab_cw* ]] && return 0

    # 4) Whole-brain heuristic -> keep
    [[ "$lname" == *eicab_wb*       \
       || "$lname" == *eicab_whole* \
       || "$lname" == *whole_brain* || "$lname" == *whole-brain* \
       || "$lname" == *tof_wb*      || "$lname" == *tof-wb* \
       || "$lname" == *_wb_*        ]] && return 0

    return 1
}

_rm() {
    local target="$1"
    if (( APPLY == 1 )); then
        if [[ -d "$target" && ! -L "$target" ]]; then
            rm -rf -- "$target"
        else
            rm -f -- "$target"
        fi
    fi
    printf '    rm %s\n' "$target"
}

_rmdir_if_empty() {
    local d="$1"
    if [[ -d "$d" ]] && ! find "$d" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
        if (( APPLY == 1 )); then
            rmdir -- "$d" 2>/dev/null || true
        fi
        printf '    rmdir %s\n' "$d"
    fi
}

# --- main loop ---------------------------------------------------------------

total=0
pruned=0
skipped=0
nokeep=()         # subjects with no CW/WB/_resampled candidate at all

echo "============================================================"
echo "eICAB clean $ACTION"
echo "  root      : $ROOT"
[[ -n "$SUBDIR" ]] && echo "  subdir    : $SUBDIR"
echo "  keep-aux  : $KEEP_AUX"
echo "============================================================"

for subj_dir in "${subjects[@]}"; do
    subj="$(basename "$subj_dir")"
    (( total++ )) || true

    # Resolve actual output dir.
    if [[ -n "$SUBDIR" ]]; then
        out_dir="$subj_dir/$SUBDIR"
    elif [[ -d "$subj_dir/eicab" ]]; then
        out_dir="$subj_dir/eicab"
    else
        out_dir="$subj_dir"
    fi

    echo
    echo "[$subj] -> $out_dir"

    if [[ ! -d "$out_dir" ]]; then
        echo "  (skip: output dir missing)"
        (( skipped++ )) || true
        continue
    fi

    if (( KEEP_AUX == 1 )); then
        echo "  (skip prune: --keep-aux)"
        (( skipped++ )) || true
        continue
    fi

    # 1) Collect keep-set of NIfTIs.
    keep_list=()
    while IFS= read -r -d '' p; do
        if _is_nifti "$p" && _should_keep_nifti "$p"; then
            keep_list+=("$p")
        fi
    done < <(find "$out_dir" -type f -print0)

    if (( ${#keep_list[@]} == 0 )); then
        echo "  WARNING: no CW / WB / *_resampled NIfTI matched; nothing will be kept."
        nokeep+=("$subj")
    else
        echo "  Keep:"
        printf '    %s\n' "${keep_list[@]}"
    fi

    # Build an associative lookup for keeps (resolved paths).
    declare -A keep_set=()
    for k in "${keep_list[@]}"; do
        rk="$(readlink -f -- "$k" 2>/dev/null || echo "$k")"
        keep_set["$rk"]=1
    done

    # 2) Remove legacy subdirs.
    for sub in original_space nn_space metric_space; do
        d="$out_dir/$sub"
        if [[ -d "$d" ]]; then
            _rm "$d"
        fi
    done

    # 3) Remove .csv / .txt / .log + non-kept NIfTIs everywhere under out_dir.
    while IFS= read -r -d '' p; do
        [[ -f "$p" ]] || continue
        case "${p,,}" in
            *.csv|*.txt|*.log)
                _rm "$p"
                continue
                ;;
        esac
        if _is_nifti "$p"; then
            rp="$(readlink -f -- "$p" 2>/dev/null || echo "$p")"
            if [[ -z "${keep_set[$rp]:-}" ]]; then
                _rm "$p"
            fi
        fi
    done < <(find "$out_dir" -type f -print0)

    # 4) Clean up empty directories (deepest first), but don't remove out_dir.
    while IFS= read -r d; do
        [[ "$d" == "$out_dir" ]] && continue
        _rmdir_if_empty "$d"
    done < <(find "$out_dir" -mindepth 1 -depth -type d)

    unset keep_set
    (( pruned++ )) || true
done

echo
echo "============================================================"
echo "Done. subjects=$total  pruned=$pruned  skipped=$skipped"
if (( ${#nokeep[@]} > 0 )); then
    echo "Subjects with no CW/WB/_resampled match (kept set was empty):"
    printf '  %s\n' "${nokeep[@]}" | sort
fi
echo "Mode: $ACTION"
(( ${#nokeep[@]} == 0 ))