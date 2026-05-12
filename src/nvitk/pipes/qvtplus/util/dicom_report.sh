#!/usr/bin/env bash
# Usage: ./check_pesa_dicom.sh <dicom_root>
#
# Scans every PESA* subject under <dicom_root> and reports which subjects
# are missing one or more of: 4Dflow_AP, 4Dflow_FH, 4Dflow_RL, TOF.
set -euo pipefail

ROOT="${1:-}"
if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
    echo "Usage: $0 <dicom_root>" >&2
    echo "  <dicom_root> must contain PESA* subject folders" >&2
    exit 1
fi

REQUIRED=(4Dflow_AP 4Dflow_FH 4Dflow_RL TOF)

complete=()
incomplete=()
declare -A missing_by_subject=()

shopt -s nullglob
subjects=( "$ROOT"/PESA*/ )

if (( ${#subjects[@]} == 0 )); then
    echo "No PESA* subject folders found under: $ROOT" >&2
    exit 2
fi

for subj_dir in "${subjects[@]}"; do
    subj="$(basename "$subj_dir")"
    missing=()

    for seq in "${REQUIRED[@]}"; do
        d="$subj_dir/$seq"
        if [[ ! -d "$d" ]]; then
            missing+=("$seq[missing dir]")
            continue
        fi
        # Any regular file in the first 3 levels counts as "has images".
        if [[ -z "$(find "$d" -maxdepth 3 -type f -print -quit 2>/dev/null)" ]]; then
            missing+=("$seq[empty]")
        fi
    done

    if (( ${#missing[@]} == 0 )); then
        complete+=("$subj")
    else
        incomplete+=("$subj")
        missing_by_subject["$subj"]="${missing[*]}"
    fi
done

total=$(( ${#complete[@]} + ${#incomplete[@]} ))

echo "============================================================"
echo "DICOM completeness report"
echo "  root          : $ROOT"
echo "  sequences req : ${REQUIRED[*]}"
echo "============================================================"
echo "Subjects scanned : $total"
echo "Complete         : ${#complete[@]}"
echo "Incomplete       : ${#incomplete[@]}"
echo

if (( ${#incomplete[@]} > 0 )); then
    echo "-- Incomplete subjects (missing sequences) --"
    # Sort alphabetically for stable reporting.
    while IFS= read -r s; do
        printf "  %-20s  ->  %s\n" "$s" "${missing_by_subject[$s]}"
    done < <(printf '%s\n' "${incomplete[@]}" | sort)
    echo
fi

if (( ${#complete[@]} > 0 )); then
    echo "-- Complete subjects --"
    printf '  %s\n' "${complete[@]}" | sort
fi

# Exit non-zero when any subject is incomplete, useful for CI / pipelines.
(( ${#incomplete[@]} == 0 ))