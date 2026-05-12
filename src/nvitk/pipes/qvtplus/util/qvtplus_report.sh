#!/usr/bin/env bash
# Usage: ./qvtplus_report.sh <qvtplus_root>
#
# Scans every PESA* subject under <qvtplus_root> and reports:
#   * subject-root required QVTPlus outputs (NIfTI/CSV/MAT/XLS)
#   * optional artifacts (resampled-to-MAG variants, MIP, branch mask, JPG count)
#   * centerline post-processing (centerline_test/) -> absent / partial / complete
#
# Exit non-zero only when any *subject-root required* item is missing.
set -euo pipefail

ROOT="${1:-}"
if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
    echo "Usage: $0 <qvtplus_root>" >&2
    echo "  <qvtplus_root> must contain PESA* subject folders" >&2
    exit 1
fi

# Subject-root required outputs (exact basenames; .nii or .nii.gz accepted via glob).
REQ_FILES=(
    "multilabel_QVTseg.nii"
    "QVT_seg.nii"
    "QVT_MAG.nii"
    "QVT_CD.nii"
    "TOF_eICAB_CW.nii"
    "TOF_resampled.nii"
    "LabelsQVT.csv"
    "SummaryParamTool.xls"
    "transform.mat"
)
# qvtData_ISOfix_*.mat (subject root, NOT under centerline_test)

# Centerline-test required-when-present.
CL_REQ_FILES=(
    "multilabel_QVTseg.nii"
    "branch_mask.nii"
    "LabelsQVT.csv"
    "SummaryParamTool.xls"
    "segment_centerline_eICAB_only.nii"
    "segment_centerline_eICAB_venous.nii"
    "segment_centerline_venous_only.nii"
)

complete=()
incomplete=()
declare -A missing_by_subject=()
declare -A cl_status=()    # absent | partial | complete
declare -A cl_missing=()
declare -A jpg_root=()
declare -A jpg_cl=()
declare -A opt_extras=()   # space-separated tokens for the optional summary

shopt -s nullglob nocaseglob
subjects=( "$ROOT"/PESA*/ )
shopt -u nocaseglob

if (( ${#subjects[@]} == 0 )); then
    echo "No PESA* subject folders found under: $ROOT" >&2
    exit 2
fi

_has() {
    # _has <dir> <basename>  -> 0 if file exists (also accept .gz for .nii)
    local d="$1" name="$2"
    [[ -e "$d/$name" ]] && return 0
    if [[ "$name" == *.nii ]]; then
        [[ -e "$d/${name}.gz" ]] && return 0
    fi
    return 1
}

_has_glob() {
    # _has_glob <dir> <glob>  -> 0 if at least one match exists (no recursion)
    local d="$1" pat="$2"
    shopt -s nullglob
    local matches=( "$d"/$pat )
    shopt -u nullglob
    (( ${#matches[@]} > 0 ))
}

_count_glob() {
    local d="$1" pat="$2"
    shopt -s nullglob
    local m=( "$d"/$pat )
    shopt -u nullglob
    echo "${#m[@]}"
}

for subj_dir in "${subjects[@]}"; do
    subj="$(basename "$subj_dir")"

    # Required subject-root files
    missing=()
    for f in "${REQ_FILES[@]}"; do
        _has "$subj_dir" "$f" || missing+=("$f[missing]")
    done
    # qvtData_ISOfix_*.mat at subject root (excluding centerline_test/ matches)
    _has_glob "$subj_dir" "qvtData_ISOfix_*.mat" \
        || missing+=("qvtData_ISOfix_*.mat[missing]")

    # Optional subject-root artifacts
    extras=()
    _has "$subj_dir" "r_TOF_eICAB_CW.nii"  && extras+=("r_TOF_eICAB_CW")
    _has "$subj_dir" "r_TOF_resampled.nii" && extras+=("r_TOF_resampled")
    _has "$subj_dir" "branch_mask.nii"     && extras+=("branch_mask")
    _has_glob "$subj_dir" "*_MIP.nii"      && extras+=("MIP")
    _has_glob "$subj_dir" "*_MIP.nii.gz"   && extras+=("MIP")
    jpg_root["$subj"]="$(_count_glob "$subj_dir" "*_Vessel_*_Point_*_Slicesview.jpg")"
    opt_extras["$subj"]="${extras[*]}"

    # Centerline post-processing
    cl_dir="$subj_dir/centerline_test"
    if [[ ! -d "$cl_dir" ]]; then
        cl_status["$subj"]="absent"
        cl_missing["$subj"]=""
        jpg_cl["$subj"]="0"
    else
        cl_miss=()
        for f in "${CL_REQ_FILES[@]}"; do
            _has "$cl_dir" "$f" || cl_miss+=("$f")
        done
        _has_glob "$cl_dir" "flow_PI_per_centerline_*.csv" \
            || _has_glob "$cl_dir" "flow_PI_per_centerline_*.xlsx" \
            || cl_miss+=("flow_PI_per_centerline_*.{csv,xlsx}")
        _has_glob "$cl_dir" "qvtData_ISOfix_centerline_*.mat" \
            || cl_miss+=("qvtData_ISOfix_centerline_*.mat")

        jpg_cl["$subj"]="$(_count_glob "$cl_dir" "*_Vessel_*_Point_*_Slicesview.jpg")"

        if (( ${#cl_miss[@]} == 0 )); then
            cl_status["$subj"]="complete"
        else
            cl_status["$subj"]="partial"
            cl_missing["$subj"]="${cl_miss[*]}"
        fi
    fi

    if (( ${#missing[@]} == 0 )); then
        complete+=("$subj")
    else
        incomplete+=("$subj")
        missing_by_subject["$subj"]="${missing[*]}"
    fi
done

total=$(( ${#complete[@]} + ${#incomplete[@]} ))

echo "============================================================"
echo "QVTPlus completeness report"
echo "  root          : $ROOT"
echo "  required (subj root):"
printf '    %s\n' "${REQ_FILES[@]}" "qvtData_ISOfix_*.mat"
echo "  centerline_test/ classified as: absent | partial | complete"
echo "============================================================"
echo "Subjects scanned : $total"
echo "Complete (req.)  : ${#complete[@]}"
echo "Incomplete       : ${#incomplete[@]}"
echo

if (( ${#incomplete[@]} > 0 )); then
    echo "-- Incomplete subjects (missing required at root) --"
    while IFS= read -r s; do
        printf "  %-20s  ->  %s\n" "$s" "${missing_by_subject[$s]}"
    done < <(printf '%s\n' "${incomplete[@]}" | sort)
    echo
fi

if (( ${#complete[@]} > 0 )); then
    echo "-- Complete subjects (root) --"
    while IFS= read -r s; do
        printf "  %-20s  centerline=%-8s  jpg(root)=%s  jpg(centerline)=%s  optional=[%s]\n" \
            "$s" "${cl_status[$s]}" "${jpg_root[$s]:-0}" "${jpg_cl[$s]:-0}" "${opt_extras[$s]:-}"
    done < <(printf '%s\n' "${complete[@]}" | sort)
    echo
fi

echo "-- Centerline post-processing summary --"
cl_complete=0; cl_partial=0; cl_absent=0
for s in "${!cl_status[@]}"; do
    case "${cl_status[$s]}" in
        complete) ((cl_complete++)) ;;
        partial)  ((cl_partial++)) ;;
        absent)   ((cl_absent++)) ;;
    esac
done
echo "  complete : $cl_complete"
echo "  partial  : $cl_partial"
echo "  absent   : $cl_absent"
echo

if (( cl_partial > 0 )); then
    echo "-- Partial centerline subjects (missing items) --"
    for s in $(printf '%s\n' "${!cl_missing[@]}" | sort); do
        [[ -n "${cl_missing[$s]}" ]] || continue
        printf "  %-20s  ->  %s\n" "$s" "${cl_missing[$s]}"
    done
fi

(( ${#incomplete[@]} == 0 ))