#!/usr/bin/env bash
# Usage:
#   ./qvtplus_restructure.sh <qvtplus_root> [--apply]
#
# Walks every PESA* subject under <qvtplus_root> and restructures each subject
# folder into:
#
#   <subject>/
#     registration/
#         transform.mat
#         r_TOF_eICAB_*.nii[.gz]
#         r_TOF_resampled.nii[.gz]
#     segmentation/
#         qvt_binary_mask.nii[.gz]       <- QVT_seg.nii(.gz)
#         qvt_multilabel_mask.nii[.gz]   <- multilabel_QVTseg.nii(.gz)
#         qvt_cd_3d.nii[.gz]             <- QVT_CD.nii(.gz)
#         qvt_centerline_mask.nii[.gz]   <- branch_mask.nii(.gz)
#     measure/
#         LabelsQVT.csv
#         SummaryParamTool.xls
#     qvtData_ISOfix_*.mat               (kept at subject root)
#
# Anything else under the subject (vessel JPGs, MIP, QVT_MAG.nii, TOF_eICAB_CW.nii,
# TOF_resampled.nii, the entire centerline_test/ directory after promotion, etc.)
# is removed.
#
# centerline_test/ rule
# ---------------------
# centerline_test/ holds a newer pipeline run of (some of) the same artifacts.
# When that folder exists, for each "to-keep" target the script *prefers* the
# centerline_test version over the subject-root version; targets that don't have
# a centerline_test counterpart (r_TOF_*, transform.mat, QVT_CD.nii, ...) are
# taken from the subject root. After the moves complete, centerline_test/ is
# deleted in full.
#
# Default mode: DRY-RUN. Pass --apply to actually move/delete files.

set -euo pipefail

ROOT=""
APPLY=0

usage() {
    cat <<EOF
Usage: $0 <qvtplus_root> [--apply]
  <qvtplus_root>  Root directory containing PESA* subject folders.
  --apply         Actually perform moves/deletes (default: dry-run).
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --apply)     APPLY=1; shift ;;
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
shopt -u nullglob nocaseglob

if (( ${#subjects[@]} == 0 )); then
    echo "No PESA* subject folders found under: $ROOT" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_print_plan() {
    printf '    %s\n       -> %s\n' "$1" "$2"
}

_ensure_dir() {
    local d="$1"
    if (( APPLY == 1 )); then mkdir -p -- "$d"; fi
}

_mv() {
    local src="$1" dst="$2"
    if [[ "$src" == "$dst" ]]; then
        printf '    keep (in place): %s\n' "$src"
        return 0
    fi
    if [[ ! -e "$src" ]]; then
        printf '    SKIP (source vanished): %s\n' "$src"
        return 0
    fi
    if [[ -e "$dst" ]] && (( APPLY == 1 )); then
        rm -rf -- "$dst"
    fi
    if (( APPLY == 1 )); then
        mkdir -p -- "$(dirname -- "$dst")"
        mv -f -- "$src" "$dst"
    fi
    _print_plan "$src" "$dst"
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

# Resolve the NIfTI extension of a basename: .nii.gz / .nii / "" otherwise.
_nifti_ext() {
    local b="${1,,}"
    if   [[ "$b" == *.nii.gz ]]; then echo ".nii.gz"
    elif [[ "$b" == *.nii    ]]; then echo ".nii"
    else echo ""; fi
}

# Pick a single source preferring centerline_test/<name> over <subj>/<name>.
# For NIfTI source names (ending in .nii), also try the .nii.gz alternative.
# Echoes the chosen path (empty if none found). All candidates are checked
# with -f so non-existent literal paths never bleed through.
_pick_one() {
    local subj_dir="$1" name="$2"
    local cl_dir="$subj_dir/centerline_test"
    local candidates=( "$cl_dir/$name" "$subj_dir/$name" )
    if [[ "$name" == *.nii ]]; then
        candidates=(
            "$cl_dir/$name" "$cl_dir/${name}.gz"
            "$subj_dir/$name" "$subj_dir/${name}.gz"
        )
    fi
    for c in "${candidates[@]}"; do
        [[ -f "$c" ]] && { echo "$c"; return 0; }
    done
    return 1
}

# Pick all matches of a wildcard glob, preferring centerline_test if any are
# there. Echoes one EXISTING path per line. We filter with -f explicitly:
# bash's nullglob does NOT eliminate wildcard-less "patterns", so an array
# could still contain a literal non-existent path. -f guards against that.
_pick_glob() {
    local subj_dir="$1" pat="$2"
    local cl_dir="$subj_dir/centerline_test"

    shopt -s nullglob
    local cl_hits=( $cl_dir/$pat )
    local cl_ok=()
    for h in "${cl_hits[@]}"; do [[ -f "$h" ]] && cl_ok+=("$h"); done
    if (( ${#cl_ok[@]} > 0 )); then
        printf '%s\n' "${cl_ok[@]}"
        shopt -u nullglob
        return 0
    fi

    local rt_hits=( $subj_dir/$pat )
    local rt_ok=()
    for h in "${rt_hits[@]}"; do [[ -f "$h" ]] && rt_ok+=("$h"); done
    shopt -u nullglob
    if (( ${#rt_ok[@]} > 0 )); then
        printf '%s\n' "${rt_ok[@]}"
        return 0
    fi
    return 1
}

# Build target filename: <stem><ext-from-source>. Preserves .nii vs .nii.gz.
_target_with_ext() {
    local src="$1" stem="$2"
    local base; base="$(basename -- "$src")"
    local ext; ext="$(_nifti_ext "$base")"
    [[ -z "$ext" ]] && ext=".${base##*.}"
    echo "${stem}${ext}"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

total=0
restructured=0
missing_required_subjects=()

echo "============================================================"
echo "QVTPlus restructure $ACTION"
echo "  root      : $ROOT"
echo "============================================================"

for subj_dir in "${subjects[@]}"; do
    # Drop trailing slash so paths concatenate cleanly.
    subj_dir="${subj_dir%/}"
    subj="$(basename "$subj_dir")"
    (( total++ )) || true

    echo
    echo "[$subj] -> $subj_dir"

    cl_dir="$subj_dir/centerline_test"
    has_cl=0
    [[ -d "$cl_dir" ]] && has_cl=1
    (( has_cl == 1 )) && echo "  (centerline_test/ present — preferred for shared artifacts)"

    declare -a moves=()
    declare -a missing=()

    # ------- registration/ --------------------------------------------------
    src="$(_pick_one "$subj_dir" "transform.mat" || true)"
    if [[ -n "$src" ]]; then
        moves+=("$src|$subj_dir/registration/transform.mat")
    else
        missing+=("transform.mat")
    fi

    # r_TOF_eICAB_*.nii(.gz) — wildcard, use _pick_glob and keep every match.
    found_eicab=0
    while IFS= read -r src; do
        [[ -z "$src" ]] && continue
        b="$(basename "$src")"
        moves+=("$src|$subj_dir/registration/$b")
        found_eicab=1
    done < <(
        _pick_glob "$subj_dir" "r_TOF_eICAB_*.nii"    || true
        _pick_glob "$subj_dir" "r_TOF_eICAB_*.nii.gz" || true
    )
    (( found_eicab == 0 )) && missing+=("r_TOF_eICAB_*.nii[.gz]")

    # r_TOF_resampled is a literal name (no wildcard): use _pick_one which
    # also handles the .nii / .nii.gz alternative correctly.
    src="$(_pick_one "$subj_dir" "r_TOF_resampled.nii" || true)"
    if [[ -n "$src" ]]; then
        b="$(basename "$src")"
        moves+=("$src|$subj_dir/registration/$b")
    else
        missing+=("r_TOF_resampled.nii[.gz]")
    fi

    # ------- segmentation/ -------------------------------------------------
    declare -A SEG_MAP=(
        ["QVT_seg.nii"]="qvt_binary_mask"
        ["multilabel_QVTseg.nii"]="qvt_multilabel_mask"
        ["QVT_CD.nii"]="qvt_cd_3d"
        ["branch_mask.nii"]="qvt_centerline_mask"
    )
    for src_name in "QVT_seg.nii" "multilabel_QVTseg.nii" "QVT_CD.nii" "branch_mask.nii"; do
        stem="${SEG_MAP[$src_name]}"
        src="$(_pick_one "$subj_dir" "$src_name" || true)"
        if [[ -n "$src" ]]; then
            tgt="$(_target_with_ext "$src" "$stem")"
            moves+=("$src|$subj_dir/segmentation/$tgt")
        else
            missing+=("$src_name (-> segmentation/$stem)")
        fi
    done
    unset SEG_MAP

    # ------- measure/ ------------------------------------------------------
    for src_name in "LabelsQVT.csv" "SummaryParamTool.xls"; do
        src="$(_pick_one "$subj_dir" "$src_name" || true)"
        if [[ -n "$src" ]]; then
            moves+=("$src|$subj_dir/measure/$src_name")
        else
            missing+=("$src_name")
        fi
    done

    # ------- qvtData_ISOfix_*.mat at subject root -------------------------
    qvt_src=""
    while IFS= read -r src; do
        [[ -z "$src" ]] && continue
        qvt_src="$src"          # first wins (centerline_test preferred)
        break
    done < <(_pick_glob "$subj_dir" "qvtData_ISOfix_*.mat" || true)
    if [[ -n "$qvt_src" ]]; then
        b="$(basename "$qvt_src")"
        moves+=("$qvt_src|$subj_dir/$b")
    else
        missing+=("qvtData_ISOfix_*.mat")
    fi

    # ------- Print plan ----------------------------------------------------
    if (( ${#missing[@]} > 0 )); then
        missing_required_subjects+=("$subj")
        echo "  WARNING: missing required source(s):"
        printf '    - %s\n' "${missing[@]}"
    fi

    echo "  Planned moves:"
    declare -A kept_paths=()
    for entry in "${moves[@]}"; do
        src="${entry%%|*}"
        dst="${entry#*|}"
        rd="$(readlink -f -- "$dst" 2>/dev/null || echo "$dst")"
        kept_paths["$rd"]=1
    done

    _ensure_dir "$subj_dir/registration"
    _ensure_dir "$subj_dir/segmentation"
    _ensure_dir "$subj_dir/measure"
    for entry in "${moves[@]}"; do
        src="${entry%%|*}"
        dst="${entry#*|}"
        _mv "$src" "$dst"
    done

    # ------- Cleanup -------------------------------------------------------
    echo "  Cleanup:"
    if [[ -d "$cl_dir" ]]; then
        _rm "$cl_dir"
    fi

    # Sweep subject root: remove every file that isn't a kept target
    # and isn't already inside the new subdirs.
    while IFS= read -r -d '' p; do
        rp="$(readlink -f -- "$p" 2>/dev/null || echo "$p")"
        [[ -n "${kept_paths[$rp]:-}" ]] && continue
        case "$p" in
            "$subj_dir/registration"/*|"$subj_dir/segmentation"/*|"$subj_dir/measure"/*) continue ;;
        esac
        _rm "$p"
    done < <(find "$subj_dir" -mindepth 1 -maxdepth 1 -type f -print0)

    # Drop any stray empty top-level subdirs that aren't one of the three new
    # ones (in case some legacy run wrote e.g. a "logs/" dir).
    while IFS= read -r d; do
        case "$d" in
            "$subj_dir/registration"|"$subj_dir/segmentation"|"$subj_dir/measure") continue ;;
        esac
        if [[ -d "$d" ]] && ! find "$d" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
            _rm "$d"
        fi
    done < <(find "$subj_dir" -mindepth 1 -maxdepth 1 -type d)

    unset kept_paths
    (( restructured++ )) || true
done

echo
echo "============================================================"
echo "Done. subjects=$total  restructured=$restructured"
if (( ${#missing_required_subjects[@]} > 0 )); then
    echo "Subjects with missing required sources (review):"
    printf '  %s\n' "${missing_required_subjects[@]}" | sort
fi
echo "Mode: $ACTION"
(( ${#missing_required_subjects[@]} == 0 ))