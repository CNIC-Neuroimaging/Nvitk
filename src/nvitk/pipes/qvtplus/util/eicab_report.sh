#!/usr/bin/env bash
# Usage: ./eicab_report.sh <eicab_root>
#
# Scans every PESA* subject under <eicab_root> and reports, per subject, whether
# the eICAB outputs are present:
#   required : Circle-of-Willis NIfTI (*eICAB_CW*.nii[.gz]) and resampled TOF
#              (*_resampled.nii[.gz])
#   optional : whole-brain NIfTI (*eICAB_WB* / *whole_brain* / *TOF_WB* / *_WB_*)
#
# Layout expected:
#   <eicab_root>/PESA*/<files...>            (flat)
#   <eicab_root>/PESA*/eicab/<files...>      (nested under "eicab/")
#
# Exit non-zero only when a required item is missing on at least one subject.

ROOT="${1:-}"
if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
    echo "Usage: $0 <eicab_root>" >&2
    echo "  <eicab_root> must contain PESA* subject folders" >&2
    exit 1
fi

complete=()
incomplete=()
declare -A missing_by_subject=()
declare -A wb_status=()

shopt -s nullglob nocaseglob
subjects=( "$ROOT"/PESA*/ )
shopt -u nocaseglob

if (( ${#subjects[@]} == 0 )); then
    echo "No PESA* subject folders found under: $ROOT" >&2
    exit 2
fi

_first_match() {
    # Print the first matching file (or nothing). Args: <dir> <glob>...
    local dir="$1"; shift
    [[ -d "$dir" ]] || return 0
    local hit
    hit="$(find "$dir" -maxdepth 2 -type f \( "$@" \) -print -quit 2>/dev/null || true)"
    [[ -n "$hit" ]] && printf '%s\n' "$hit"
}

for subj_dir in "${subjects[@]}"; do
    subj="$(basename "$subj_dir")"

    # Accept either the subject dir itself or a nested eicab/ folder.
    search_dirs=( "$subj_dir" )
    [[ -d "$subj_dir/eicab" ]] && search_dirs+=( "$subj_dir/eicab" )

    cw_hit=""; wb_hit=""; rs_hit=""
    for d in "${search_dirs[@]}"; do
        [[ -z "$cw_hit" ]] && cw_hit="$(_first_match "$d" \
            -iname "*eicab_cw*.nii"     -o -iname "*eicab_cw*.nii.gz")"
        [[ -z "$wb_hit" ]] && wb_hit="$(_first_match "$d" \
            -iname "*eicab_wb*.nii"     -o -iname "*eicab_wb*.nii.gz" \
            -o -iname "*whole*brain*.nii" -o -iname "*whole*brain*.nii.gz" \
            -o -iname "*tof_wb*.nii"    -o -iname "*tof_wb*.nii.gz" \
            -o -iname "*_wb_*.nii"      -o -iname "*_wb_*.nii.gz")"
        [[ -z "$rs_hit" ]] && rs_hit="$(_first_match "$d" \
            -iname "*_resampled.nii"    -o -iname "*_resampled.nii.gz")"
    done

    missing=()
    [[ -z "$cw_hit" ]] && missing+=("CW[missing]")
    [[ -z "$rs_hit" ]] && missing+=("resampled[missing]")
    wb_status["$subj"]=$([[ -n "$wb_hit" ]] && echo "yes" || echo "no")

    if (( ${#missing[@]} == 0 )); then
        complete+=("$subj")
    else
        incomplete+=("$subj")
        missing_by_subject["$subj"]="${missing[*]}"
    fi
done

total=$(( ${#complete[@]} + ${#incomplete[@]} ))

echo "============================================================"
echo "eICAB completeness report"
echo "  root          : $ROOT"
echo "  required      : *eICAB_CW*.nii* , *_resampled.nii*"
echo "  optional      : *eICAB_WB* / *whole_brain* / *TOF_WB* / *_WB_*"
echo "============================================================"
echo "Subjects scanned : $total"
echo "Complete (req.)  : ${#complete[@]}"
echo "Incomplete       : ${#incomplete[@]}"
echo

if (( ${#incomplete[@]} > 0 )); then
    echo "-- Incomplete subjects (missing required) --"
    while IFS= read -r s; do
        printf "  %-20s  ->  %s\n" "$s" "${missing_by_subject[$s]}"
    done < <(printf '%s\n' "${incomplete[@]}" | sort)
    echo
fi

if (( ${#complete[@]} > 0 )); then
    echo "-- Complete subjects (CW + resampled) --"
    while IFS= read -r s; do
        printf "  %-20s  WB=%s\n" "$s" "${wb_status[$s]:-no}"
    done < <(printf '%s\n' "${complete[@]}" | sort)
    echo
fi

echo "-- Whole-brain (optional) summary --"
wb_yes=0
for s in "${!wb_status[@]}"; do
    [[ "${wb_status[$s]}" == "yes" ]] && ((wb_yes++))
done
echo "  WB present   : $wb_yes / $total"

(( ${#incomplete[@]} == 0 ))