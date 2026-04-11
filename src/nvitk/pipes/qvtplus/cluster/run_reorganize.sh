#!/bin/bash
#######################################################################################################################
# Reorganize NIfTI Files - HPC Cluster Submission Script
# Reorganizes DICOM2NIfTI outputs into eICAB and QVT+ compatible directory structure
#######################################################################################################################
#######################################################################################################################
# Script variables
#######################################################################################################################

# Input directory (output from dicom2nifti conversion)
# input="/data3/BIOIT_IMAGE/PESA-Brain/DATA/Nifti"
input="/mnt/scratch/imarcoss/NIFTI_UNORGANIZED"

# Output directory (input for eICAB and QVT+)
# output="/data3/BIOIT_IMAGE/PESA-Brain/DATA/Nifti_Reorganized"
output="/mnt/scratch/imarcoss/NIFTI"

skip_existing="false"  # Skip existing outputs [true | false] (default: false)

# Remove trailing slash if present
input="${input%/}"
output="${output%/}"

# Validation: Check if input directory exists (unless waiting for previous step)
hold_jid="${1:-}"
if [ -z "$hold_jid" ] && [ ! -d "$input" ]; then
  echo "Error: Input directory does not exist: $input"
  exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$output" ]; then mkdir -p "$output"; fi

########################################################################################################################
# Submission variables
########################################################################################################################

bioimaging_project="PesaBrain"
task="Reorganize"

sge_project="FSC"
sge_account="Prod"
sge_vmem="8G"              # Memory for reorganization (usually not too memory intensive)

path_sge_log="/data3/BIOIT_IMAGE/BioImaging/env/logs/${bioimaging_project}"
path_sge_err="/data3/BIOIT_IMAGE/BioImaging/env/errs/${bioimaging_project}"
if [ ! -d "$path_sge_log" ]; then mkdir -p "$path_sge_log"; fi
if [ ! -d "$path_sge_err" ]; then mkdir -p "$path_sge_err"; fi

# Path to the reorganization script
reorganize_script="$(cd "$(dirname "${BASH_SOURCE[0]}")/../util/xnat" && pwd)/ReorganizeDICOMs_TOF_4DFlow.sh"

if [ ! -f "$reorganize_script" ]; then
  echo "Error: Reorganization script not found: $reorganize_script"
  exit 1
fi

# Get absolute paths
input_abs=$(realpath -m "$input")  # Use -m to allow non-existent path when waiting for previous step
output_abs=$(realpath -m "$output")

# Remove old err file if it exists
sge_name="${bioimaging_project}_${task}_ALL"
if [ -f "$path_sge_err/${task}_ALL.err" ]; then
  rm "$path_sge_err/${task}_ALL.err"
fi

# Convert skip_existing to format expected by reorganization script (already "true"/"false")
skip_existing_value="$skip_existing"

# SGE command as an array
cmd_sge=(
  qsub -P $sge_project -terse
    -N $sge_name
    -A $sge_account
    -l h_vmem=$sge_vmem
    -o $path_sge_log/${task}_ALL.log
    -e $path_sge_err/${task}_ALL.err
)

# Add hold_jid if provided
if [ -n "$hold_jid" ]; then
  echo "Holding reorganization for job $hold_jid"
  cmd_sge+=(-v "HOLD_PIPELINE=1")
  cmd_sge+=(-hold_jid "$hold_jid")
fi

# Build command to run reorganization script
cmd_run="bash $reorganize_script"

# Set environment variables for the script
env_vars="INPUT_DIR=\"$input_abs\" OUTPUT_DIR=\"$output_abs\" SKIP_EXISTING=\"$skip_existing\""

# Execute with environment variables set
full_cmd="cd $(dirname "$reorganize_script") && export INPUT_DIR=\"$input_abs\" OUTPUT_DIR=\"$output_abs\" SKIP_EXISTING=\"$skip_existing_value\" && $cmd_run"

jid=$( echo "$full_cmd" | "${cmd_sge[@]}" )

cat <<EOF | tee -a $path_sge_log/${task}_ALL.log

############################ Reorganize NIfTI Files ############################

Job:           $jid

Name:          $sge_name
Project:       $sge_project
Account:       $sge_account
Memory:        $sge_vmem
Log dir:       $path_sge_log
Error dir:     $path_sge_err

Input:         $input_abs
Output:        $output_abs
Skip existing: $skip_existing
Hold job ID:   ${hold_jid:-none}

Command:       cd $(dirname "$reorganize_script") && export INPUT_DIR=\"$input_abs\" OUTPUT_DIR=\"$output_abs\" SKIP_EXISTING=\"$skip_existing_value\" && $cmd_run

########################################################################

EOF

echo "Job $jid submitted for reorganization"

# Output job ID for chaining (to stderr so it doesn't interfere)
echo "REORGANIZE_JOB_ID=$jid" >&2

########################################################################################################################
# Summary
########################################################################################################################

echo ""
echo "======================================================================"
echo "Summary: Submitted 1 reorganization job"
echo "======================================================================"
echo "  Job $jid: Reorganizing NIfTI files from $input_abs to $output_abs"
echo ""
echo "logs in: $path_sge_log"
echo "errors in: $path_sge_err"
echo "======================================================================"
echo ""
