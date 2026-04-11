#!/bin/bash
#######################################################################################################################
# QVT+ 4D Flow Analysis - HPC Cluster Submission Script
#######################################################################################################################
#######################################################################################################################
# Script variables
#######################################################################################################################

# Base directory containing DATA and RESULTS
# base_path="/data3/BIOIT_IMAGE/PESA-Brain"
base_path="/mnt/scratch/imarcoss"
subjects2exclude=("")

# QVT+ parameters
use_eicab_whole_brain="0"      # Use eICAB whole brain [0 | 1] (default: 1)
skip_existing="0"              # Skip existing outputs [0 | 1] (default: 1)

# Input/output paths (relative to base_path)
input_data_dir="${base_path}/NIFTI"           # Directory containing subject folders
eicab_results_dir="${base_path}/RESULTS/eICAB"     # Directory containing eICAB results
output_dir="${base_path}/RESULTS/QVTPlus"          # Output directory for QVT+ results

# Remove trailing slash if present
input_data_dir="${input_data_dir%/}"
eicab_results_dir="${eicab_results_dir%/}"
output_dir="${output_dir%/}"

# Accept optional hold_jid parameter and DICOM input directory early (before validation)
# This affects validation - if waiting for previous step, directories don't exist yet
hold_jid="${1:-}"
dicom_input_dir="${2:-}"

# Validation: Check if input directory exists (unless waiting for previous step)
if [ -z "$hold_jid" ] && [ ! -d "$input_data_dir" ]; then
  echo "Error: Input data directory does not exist: $input_data_dir"
  exit 1
fi

if [ -z "$hold_jid" ] && [ ! -d "$eicab_results_dir" ]; then
  echo "Error: eICAB results directory does not exist: $eicab_results_dir"
  exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then mkdir -p "$output_dir"; fi

#######################################################################################################################
# Scan for subjects with 4DFlow data
#######################################################################################################################

declare -a subject_names=()
declare -a path_4dflow=()
declare -a path_eicab=()
declare -a path_output=()

# If hold_jid is provided but no DICOM input directory, try to infer from input_data_dir path
if [ -n "$hold_jid" ] && [ -z "$dicom_input_dir" ]; then
  # Try common patterns: if input_data_dir is .../NIFTI, DICOM might be .../DICOM
  input_parent=$(dirname "$input_data_dir")
  potential_dicom="$input_parent/DICOM"
  if [ -d "$potential_dicom" ]; then
    dicom_input_dir="$potential_dicom"
    echo "Inferred DICOM input directory: $dicom_input_dir"
  else
    echo "Warning: hold_jid provided but DICOM input directory not found. Cannot get subject names."
  fi
fi

echo "=========================================================================="
echo "Scanning for subjects with 4DFlow data..."
if [ -n "$hold_jid" ]; then
  echo "(Deferred validation - waiting for previous step to complete)"
fi
echo "=========================================================================="

# If hold_jid is provided, get subject names from DICOM directory instead
if [ -n "$hold_jid" ] && [ -n "$dicom_input_dir" ] && [ -d "$dicom_input_dir" ]; then
  echo "Getting subject names from DICOM directory: $dicom_input_dir"
  while IFS= read -r -d '' subject_dir; do
    sub_name=$(basename "$subject_dir")
    
    # Check if subject should be excluded
    if [[ " ${subjects2exclude[@]} " =~ " ${sub_name} " ]]; then
      echo "Skipping $sub_name (in exclusion list)"
      continue
    fi
    
    # Construct expected paths (will exist when job runs)
    flow_dir="$input_data_dir/$sub_name/4DFlow"
    eicab_subject_dir="$eicab_results_dir/$sub_name"
    output_subject_dir="$output_dir/$sub_name"
    
    # Add to processing list (deferred validation)
    subject_names+=("$sub_name")
    path_4dflow+=("$flow_dir")
    path_eicab+=("$eicab_subject_dir")
    path_output+=("$output_subject_dir")
    
    echo "  - $sub_name (deferred validation, waiting for previous step)"
  done < <(find "$dicom_input_dir" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
else
  # Scan for subjects with 4DFlow directories (normal mode)
  while IFS= read -r -d '' subject_dir; do
    sub_name=$(basename "$subject_dir")
    
    # Check if subject should be excluded
    if [[ " ${subjects2exclude[@]} " =~ " ${sub_name} " ]]; then
      echo "Skipping $sub_name (in exclusion list)"
      continue
    fi
    
    # Check for 4DFlow directory
    flow_dir="$subject_dir/4DFlow"
    if [ ! -d "$flow_dir" ]; then
      echo "Warning: Skipping $sub_name (missing 4DFlow directory)"
      continue
    fi
    
    # Check for eICAB results
    eicab_subject_dir="${eicab_results_dir}/${sub_name}"
    if [ ! -d "$eicab_subject_dir" ]; then
      echo "Warning: Skipping $sub_name (missing eICAB results at $eicab_subject_dir)"
      continue
    fi
  
  # Check if output already exists (if skip_existing is enabled)
  output_subject_dir="${output_dir}/${sub_name}"
  if [ "$skip_existing" = "1" ] && [ -d "$output_subject_dir" ]; then
    echo "Skipping $sub_name (output already exists at $output_subject_dir)"
    continue
  fi
  
  # Add to processing list
  subject_names+=("$sub_name")
  path_4dflow+=("$flow_dir")
  path_eicab+=("$eicab_subject_dir")
  path_output+=("$output_subject_dir")
  
    echo "  - $sub_name"
  done < <(find "$input_data_dir" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
fi

if [ ${#subject_names[@]} -eq 0 ]; then
  if [ -n "$hold_jid" ]; then
    echo "Error: No subjects found with 4DFlow data (cannot validate eICAB results when waiting for previous step)"
  else
    echo "Error: No subjects found with both 4DFlow data and eICAB results"
  fi
  exit 1
fi

echo "=========================================================================="
if [ -n "$hold_jid" ]; then
  echo "Found ${#subject_names[@]} subject(s) (deferred eICAB validation - waiting for previous step)"
else
  echo "Found ${#subject_names[@]} subject(s) ready for QVT+ processing."
fi
echo "=========================================================================="
echo ""

########################################################################################################################
# Submission variables (common for all jobs)
########################################################################################################################

bioimaging_project="PesaBrain"
task="QVTPlus"

sge_project="FSC"
sge_account="Prod"
sge_vmem="32G"              # QVT+ may need more memory than eICAB

path_sge_log="/data3/BIOIT_IMAGE/BioImaging/env/logs/${bioimaging_project}"
path_sge_err="/data3/BIOIT_IMAGE/BioImaging/env/errs/${bioimaging_project}"
if [ ! -d "$path_sge_log" ]; then mkdir -p "$path_sge_log"; fi
if [ ! -d "$path_sge_err" ]; then mkdir -p "$path_sge_err"; fi

path_container="/data3/BIOIT_IMAGE/Containers/QVTPlus_v2026.01.19.sif"
path_tmp="/data_tmp"

########################################################################################################################
# Execution: Loop through all subjects and submit jobs
########################################################################################################################

submitted_jobs=()

for idx in "${!subject_names[@]}"; do
  subject="${subject_names[$idx]}"
  flow_path="${path_4dflow[$idx]}"
  eicab_path="${path_eicab[$idx]}"
  output_path="${path_output[$idx]}"
  
  sge_name="${bioimaging_project}_${task}_${subject}"
  
  # Get absolute paths
  # If paths don't exist yet (deferred mode), use realpath -m
  if [ -n "$hold_jid" ]; then
    flow_path_abs=$(realpath -m "$flow_path")
    eicab_path_abs=$(realpath -m "$eicab_path")
  else
    flow_path_abs=$(realpath "$flow_path")
    eicab_path_abs=$(realpath "$eicab_path")
  fi
  output_path_abs=$(realpath -m "$output_path")  # -m allows creating path if it doesn't exist
  
  # Create output directory
  mkdir -p "$output_path_abs"
  
  # Remove old err file if it exists
  if [ -f "$path_sge_err/${task}_${subject}.err" ]; then
    rm "$path_sge_err/${task}_${subject}.err"
  fi
  
  # Build singularity flags
  # Bind mount the base_path to /data in container so all paths are accessible
  sing_flags="--cleanenv"
  
  # SGE command as an array
  cmd_sge=(
    qsub -P $sge_project -terse
      -N $sge_name
      -A $sge_account
      -l h_vmem=$sge_vmem
      -o $path_sge_log/${task}_${subject}.log
      -e $path_sge_err/${task}_${subject}.err
  )
  
  # Add hold_jid if provided (can be comma-separated list)
  if [ -n "$hold_jid" ]; then
    echo "Holding pipeline for job $hold_jid"
    cmd_sge+=(-v "HOLD_PIPELINE=1")
    cmd_sge+=(-hold_jid "$hold_jid")
  fi
  
  # Singularity run command
  # The container expects (single patient mode):
  #   path_to_data: /data/NIFTI/subject/4DFlow
  #   eICAB_path: /data/RESULTS/eICAB/subject
  #   output_path: /data/RESULTS/QVTPlus/subject
  #   use_eicab_whole_brain: 0 or 1
  cmd_run="singularity run $sing_flags \
    --bind ${base_path}:/data \
    --bind $path_tmp:/tmp \
    $path_container \
    /data/NIFTI/${subject}/4DFlow \
    /data/RESULTS/eICAB/${subject} \
    /data/RESULTS/QVTPlus/${subject} \
    $use_eicab_whole_brain
  "
  
  jid=$( echo "$cmd_run" | "${cmd_sge[@]}" )
  submitted_jobs+=("$jid")
  
  cat <<EOF | tee -a $path_sge_log/${task}_${subject}.log

############################ QVT+ Processing ############################

Job:           $jid

Name:          $sge_name
Project:       $sge_project
Account:       $sge_account
Memory:        $sge_vmem
Log dir:       $path_sge_log
Error dir:     $path_sge_err

Subject:       $subject
4DFlow data:   $flow_path_abs
eICAB results: $eicab_path_abs
Output:        $output_path_abs
Use eICAB WB:  $use_eicab_whole_brain
Skip existing: $skip_existing

Command:       $cmd_run

########################################################################

EOF

  echo "Job $jid submitted for: $subject"
  
done

########################################################################################################################
# Summary
########################################################################################################################

echo ""
echo "======================================================================"
echo "Summary: Submitted ${#submitted_jobs[@]} job(s)"
echo "======================================================================"
for i in "${!submitted_jobs[@]}"; do
  echo "  [$((i+1))] Job ${submitted_jobs[$i]}: ${subject_names[$i]}"
done
echo ""
echo "logs in: $path_sge_log"
echo "errors in: $path_sge_err"
echo "======================================================================"
echo ""
