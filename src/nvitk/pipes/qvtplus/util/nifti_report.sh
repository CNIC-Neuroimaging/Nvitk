#!/usr/bin/env bash
# Usage: ./nifti_report.sh <nifti_root> [--with-derived]
#
# Scans every PESA* subject under <nifti_root> (stage0_convert output layout)
# and reports which subjects are missing required NIfTI files:
#
#   4DFlow/AP/  -> *_m.nii(.gz) + *_ph.nii(.gz)
#   4DFlow/RL/  -> *_m.nii(.gz) + *_ph.nii(.gz)
#   4DFlow/FH/  -> *_m.nii(.gz) + *_ph.nii(.gz)
#   TOF/TOF.nii(.gz)
#
# With --with-derived, also reports (informational only) missing optional
# derived images under 4DFlow/:
#   Angiography_3D/4D, ComplexDifference_3D/4D, VelocityMagnitude_3D/4D
#
# Exit code is non-zero only when at least one *required* file is missing,
# matching the semantics of dicom_report.sh.
set -euo pipefail

ROOT="${1:-}"
WITH_DERIVED=0
if [[ "${2:-}" == "--with-derived" ]]; then
    WITH_DERIVED=1
fi

if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
    echo "Usage: $0 <nifti_root> [--with-derived]" >&2
    echo "  <nifti_root> must contain PESA* subject folders" >&2
    exit 1
fi

REQ_FLOW=(AP RL FH)
DERIVED=(
    Angiography_3D Angiography_4D
    ComplexDifference_3D ComplexDifference_4D
    VelocityMagnitude_3D VelocityMagnitude_4D
)

complete=()
incomplete=()
declare -A missing_by_subject=()
declare -A derived_missing_by_subject=()

shopt -s nullglob
subjects=( "$ROOT"/PESA*/ )

if (( ${#subjects[@]} == 0 )); then
    echo "No PESA* subject folders found under: $ROOT" >&2
    exit 2
fi

_has_match() {
    # Returns 0 if at least one path matches glob $1, else 1.
    # Works for unquoted globs because we expand them here.
    local pattern=$ROOT
    local matches=( $pattern )
    (( ${#matches[@]} > 0 ))
}

for subj_dir in "${subjects[@]}"; do
    subj="$(basename "$subj_dir")"
    missing=()
    derived_missing=()

    flow_root="$subj_dir/4DFlow"
    for d in "${REQ_FLOW[@]}"; do
        dd="$flow_root/$d"
        if [[ ! -d "$dd" ]]; then
            missing+=("${d}_m[missing dir]" "${d}_ph[missing dir]")
            continue
        fi
        _has_match "$dd"/*_m.nii*  || missing+=("${d}_m[missing]")
        _has_match "$dd"/*_ph.nii* || missing+=("${d}_ph[missing]")
    done

    tof_dir="$subj_dir/TOF"
    if [[ ! -d "$tof_dir" ]]; then
        missing+=("TOF[missing dir]")
    else
        _has_match "$tof_dir"/TOF.nii* || missing+=("TOF[missing]")
    fi

    if (( WITH_DERIVED == 1 )); then
        for f in "${DERIVED[@]}"; do
            if [[ ! -f "$flow_root/$f.nii.gz" && ! -f "$flow_root/$f.nii" ]]; then
                derived_missing+=("$f[missing]")
            fi
        done
    fi

    if (( ${#missing[@]} == 0 )); then
        complete+=("$subj")
    else
        incomplete+=("$subj")
        missing_by_subject["$subj"]="${missing[*]}"
    fi
    if (( ${#derived_missing[@]} > 0 )); then
        derived_missing_by_subject["$subj"]="${derived_missing[*]}"
    fi
done

total=$(( ${#complete[@]} + ${#incomplete[@]} ))

echo "============================================================"
echo "NIfTI completeness report"
echo "  root          : $ROOT"
echo "  required      : ${REQ_FLOW[*]/%/_m+ph} TOF.nii*"
if (( WITH_DERIVED == 1 )); then
    echo "  derived check : ${DERIVED[*]}"
fi
echo "============================================================"
echo "Subjects scanned : $total"
echo "Complete         : ${#complete[@]}"
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
    echo "-- Complete subjects --"
    printf '  %s\n' "${complete[@]}" | sort
    echo
fi

if (( WITH_DERIVED == 1 && ${#derived_missing_by_subject[@]} > 0 )); then
    echo "-- Derived (optional) missing --"
    while IFS= read -r s; do
        printf "  %-20s  ->  %s\n" "$s" "${derived_missing_by_subject[$s]}"
    done < <(printf '%s\n' "${!derived_missing_by_subject[@]}" | sort)
    echo
fi

(( ${#incomplete[@]} == 0 ))
