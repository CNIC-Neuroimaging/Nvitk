#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# TotalSegmentator inference wrapper used by qsub+Singularity jobs.
# Mirrors the semantics of the BioImaging inference.sh but lives inside nvitk.
# Expected env: TotalSegmentator on PATH and TOTALSEG_HOME_DIR set.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

show_help() {
  cat <<EOF
Usage: $(basename "$0") --input <path> --output <dir> --task <task> [options]

Required:
  --input <path>     NIfTI file or directory with PESA*/*.nii[.gz]
  --output <dir>     Output directory for the multilabel segmentation
  --task <task>      TotalSegmentator task name

Options:
  --subset "<rois>"  Space-separated ROI list for -rs (optional)
  --mode <mode>      "multilabel" (default) or "single_labels"
  --backend <dev>    "gpu" (default) or "cpu"
EOF
}

mode="multilabel"
backend="gpu"
subset=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) show_help; exit 0 ;;
    --input) input="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --subset) subset="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; show_help; exit 1 ;;
  esac
done

if [[ -z "${input:-}" || -z "${output:-}" || -z "${task:-}" ]]; then
  echo "Error: --input, --output and --task are required." >&2
  show_help
  exit 1
fi

case "$backend" in
  gpu|cpu) ;;
  *) echo "Error: --backend must be 'gpu' or 'cpu' (got '$backend')." >&2; exit 1 ;;
esac

mkdir -p "$output"
input="${input%/}"

# If input is a directory, pick the first .nii[.gz] directly under it.
if [[ -d "$input" ]]; then
  image=$(find "$input" -maxdepth 1 -type f \( -name "*.nii" -o -name "*.nii.gz" \) | head -n 1)
  if [[ -z "$image" ]]; then
    echo "Error: no NIfTI file found directly under $input" >&2
    exit 1
  fi
else
  image="$input"
fi

image_id=$(basename "$image" | sed 's/\.\(nii\|nii\.gz\)$//')
out_dir="$output/$task"
mkdir -p "$out_dir"

echo "Model directory: $TOTALSEG_HOME_DIR"

cmd=(TotalSegmentator -i "$image" -o "$out_dir" --statistics --device "$backend" -ta "$task")
if [[ "$mode" == "multilabel" ]]; then
  cmd+=(--ml)
fi
if [[ -n "$subset" ]]; then
  # shellcheck disable=SC2206
  subset_arr=($subset)
  cmd+=(-rs "${subset_arr[@]}")
fi

echo "TotalSegmentator command: ${cmd[*]}"
"${cmd[@]}"

echo "Done. Image: $image_id | Task: $task"
