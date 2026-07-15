#!/bin/bash
#######################################################################################################################
# eICAB Circle of Willis Segmentation - HPC Cluster Submission Script
#######################################################################################################################
#######################################################################################################################
# Script variables
#######################################################################################################################

# Input: Single MRA image file or directory containing multiple MRA images
# input="/data3/BIOIT_IMAGE/PESA-Brain/DATA/Nifti"
# output="/data3/BIOIT_IMAGE/PESA-Brain/RESULTS/eICAB"
input="/mnt/scratch/imarcoss/NIFTI"
output="/mnt/scratch/imarcoss/RESULTS/eICAB"
subjects2exclude=("")

# eICAB parameters
resolution="0.5"                # Resolution for resampling in mm (isotropic). Default: 0.5
device="cpu"                    # Device to use [cpu | cuda] (default: cpu)
simple_segmentation="False"     # Use simple segmentation [True | False] (default: False)
attention="False"               # Use attention [True | False] (default: False)

# Remove trailing slash if present
input="${input%/}"

# Validation: Check if input exists
if [ ! -e "$input" ]; then
  echo "Error: Input path does not exist: $input"
  exit 1
fi

# Accept optional hold_jid parameter and DICOM input directory (for chaining jobs)
# This affects how we scan for files - if waiting for previous step, files don't exist yet
# When hold_jid is provided, we need the DICOM input directory to get subject names
hold_jid="${1:-}"
dicom_input_dir="${2:-}"

declare -a input_files=()
declare -a dataset_names=()

# If hold_jid is provided but no DICOM input directory, try to infer from input path
if [ -n "$hold_jid" ] && [ -z "$dicom_input_dir" ]; then
  # Try common patterns: if input is .../NIFTI, DICOM might be .../DICOM
  input_parent=$(dirname "$input")
  potential_dicom="$input_parent/DICOM"
  if [ -d "$potential_dicom" ]; then
    dicom_input_dir="$potential_dicom"
    echo "Inferred DICOM input directory: $dicom_input_dir"
  else
    echo "Warning: hold_jid provided but DICOM input directory not found. Cannot get subject names."
  fi
fi

# Determine if input is a file or directory
if [ -d "$input" ]; then
  echo "=========================================================================="
  echo "Input is a directory. Scanning for subject folders with TOF images..."
  echo "=========================================================================="

  subject_mode=false

  # If hold_jid is provided, get subject names from DICOM directory instead
  if [ -n "$hold_jid" ] && [ -n "$dicom_input_dir" ] && [ -d "$dicom_input_dir" ]; then
    echo "Getting subject names from DICOM directory: $dicom_input_dir"
    while IFS= read -r -d '' subject_dir; do
      sub_name=$(basename "$subject_dir")
      # Skip if in exclusion list
      if [[ " ${subjects2exclude[@]} " =~ " ${sub_name} " ]]; then
        continue
      fi
      # Construct expected TOF path in reorganized output
      tof_dir="$input/$sub_name/TOF"
      tof_file="$tof_dir/TOF.nii.gz"
      if [ ! -f "$tof_file" ]; then
        tof_file="$tof_dir/TOF.nii"
      fi
      input_files+=("$tof_file")
      dataset_names+=("$sub_name")
      echo "  - $sub_name (deferred validation, waiting for previous step) --> ${output%/}/$sub_name/"
    done < <(find "$dicom_input_dir" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
    subject_mode=true
  else
    # First pass: look for Subject/TOF/*.nii(.gz) structure (normal mode)
    while IFS= read -r -d '' subject_dir; do
      subject_mode=true
      sub_name=$(basename "$subject_dir")
      tof_dir="$subject_dir/TOF"
      
      # Normal mode: check if TOF directory exists
      if [ ! -d "$tof_dir" ]; then
        echo "Warning: Skipping $sub_name (missing TOF directory)"
        continue
      fi

      # Check if TOF files exist
      tof_file=""
      if compgen -G "$tof_dir/*.nii.gz" > /dev/null; then
        tof_file=$(ls "$tof_dir"/*.nii.gz | sort | head -n1)
      elif compgen -G "$tof_dir/*.nii" > /dev/null; then
        tof_file=$(ls "$tof_dir"/*.nii | sort | head -n1)
      else
        echo "Warning: Skipping $sub_name (no .nii/.nii.gz inside TOF)"
        continue
      fi
    done < <(find "$input" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
  fi

  if ! $subject_mode; then
    echo "No subject folders detected; falling back to flat NIfTI search."
  fi

  if [ ${#input_files[@]} -eq 0 ]; then
    # Fallback: treat directory as flat list of NIfTI files (legacy behavior)
    # Only do this if not waiting for previous step (files should exist)
    if [ -z "$hold_jid" ]; then
      while IFS= read -r -d '' file; do
        input_files+=("$file")
        dataset=$(basename "$file" .nii.gz)
        dataset=$(basename "$dataset" .nii)
        dataset_names+=("$dataset")
      done < <(find "$input" -maxdepth 1 -type f \( -name "*.nii.gz" -o -name "*.nii" \) -print0 | sort -z)

      if [ ${#input_files[@]} -eq 0 ]; then
        echo "Error: No subject TOF directories and no flat NIfTI files found in $input"
        exit 1
      fi
    else
      # If hold_jid is set and no subject directories found, that's an error
      echo "Error: No subject directories found in $input, and cannot scan for flat files when waiting for previous step"
      exit 1
    fi

    echo "Found ${#input_files[@]} flat NIfTI file(s) to process:"
    for idx in "${!input_files[@]}"; do
      filename=$(basename "${input_files[$idx]}")
      dataset="${dataset_names[$idx]}"
      echo "  - $filename --> ${output%/}/${dataset}/"
    done
  else
    echo "=========================================================================="
    if [ -n "$hold_jid" ]; then
      echo "Found ${#input_files[@]} subject(s) (deferred validation - waiting for previous step)"
    else
      echo "Found ${#input_files[@]} subject(s) with TOF data."
    fi
    echo "=========================================================================="
  fi
  echo ""
  
elif [ -f "$input" ]; then
  # Input is a single file
  if [[ ! "$input" =~ \.(nii\.gz|nii)$ ]]; then
    echo "Error: Input file must be a .nii.gz or .nii file: $input"
    exit 1
  fi
  echo "=========================================================================="
  echo "Input is a single file: $(basename "$input")"
  dataset_name=$(basename "$input" .nii.gz)
  dataset_name=$(basename "$dataset_name" .nii)
  echo "Output directory: ${output%/}/${dataset_name}/"
  echo "=========================================================================="
  echo ""
  input_files=("$input")
  dataset=$(basename "$input" .nii.gz)
  dataset=$(basename "$dataset" .nii)
  dataset_names=("$dataset")
else
  echo "Error: Input is neither a file nor a directory: $input"
  exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$output" ]; then mkdir -p "$output"; fi

########################################################################################################################
# Submission variables (common for all jobs)
########################################################################################################################

bioimaging_project="PesaBrain"
task="eICAB_Inference"

sge_project="FSC"
sge_account="Prod"
# sge_ngpu="0"
sge_vmem="25G"

path_sge_log="/data3/BIOIT_IMAGE/BioImaging/env/logs/${bioimaging_project}"
path_sge_err="/data3/BIOIT_IMAGE/BioImaging/env/errs/${bioimaging_project}"
if [ ! -d "$path_sge_log" ]; then mkdir -p "$path_sge_log"; fi
if [ ! -d "$path_sge_err" ]; then mkdir -p "$path_sge_err"; fi

path_container="/images/eicab3.sif"
path_vasculature="/programs/Neuro/vasculature2"

########################################################################################################################
# Execution: Loop through all input files and submit jobs
########################################################################################################################

submitted_jobs=()

for idx in "${!input_files[@]}"; do
  input_file="${input_files[$idx]}"
  
  # Extract dataset name from filename (remove .nii.gz or .nii extension)
  dataset="${dataset_names[$idx]}"
  
  if [[ " ${subjects2exclude[@]} " =~ " ${dataset} " ]]; then
    echo "Skipping $dataset because it is in the subjects2exclude list"
    continue
  fi
  
  sge_name="${bioimaging_project}_${task}_${dataset}"
  
  # Get absolute path of input file
  # If file doesn't exist yet (deferred mode), use realpath -m to create path without checking existence
  if [ -n "$hold_jid" ] && [ ! -f "$input_file" ]; then
    input_file_abs=$(realpath -m "$input_file")
  else
    input_file_abs=$(realpath "$input_file")
  fi
  
  # Create output subdirectory for this file named after the image (without extension)
  path_output="${output%/}/${dataset}/"
  echo "path_output: $path_output"
  mkdir -p "$path_output"
  
  # Determine container input path based on file extension
  # In deferred mode, use .nii.gz as default (most common format)
  if [ -n "$hold_jid" ] && [ ! -f "$input_file_abs" ]; then
    # Deferred mode: try to infer from path, default to .nii.gz
    if [[ "$input_file_abs" =~ \.nii\.gz$ ]]; then
      container_input_path="/TOF.nii.gz"
    elif [[ "$input_file_abs" =~ \.nii$ ]] && [[ ! "$input_file_abs" =~ \.nii\.gz$ ]]; then
      container_input_path="/TOF.nii"
    else
      # Default to .nii.gz for deferred mode
      container_input_path="/TOF.nii.gz"
    fi
  else
    # Normal mode: check actual file extension
    if [[ "$input_file_abs" =~ \.nii\.gz$ ]]; then
      container_input_path="/TOF.nii.gz"
    elif [[ "$input_file_abs" =~ \.nii$ ]]; then
      container_input_path="/TOF.nii"
    else
      echo "Error: Unsupported file extension for $input_file_abs"
      continue
    fi
  fi
  
  # Remove old err file if it exists
  if [ -f "$path_sge_err/${task}_${dataset}.err" ]; then
    rm "$path_sge_err/${task}_${dataset}.err"
  fi
  
  # Per-subject temp (bind-mounted to /tmp in the eICAB container).
  path_tmp="${path_output%/}/.eicab_tmp"
  mkdir -p "$path_tmp"
  
  # Build optional eICAB flags
  eicab_flags=""
  if [ "$simple_segmentation" = "True" ]; then
    eicab_flags="$eicab_flags -s"
  fi
  if [ "$attention" = "True" ]; then
    eicab_flags="$eicab_flags -a"
  fi
  
  # Build singularity flags
  sing_flags="--cleanenv --env PATH='/vessel_segmentation_snaillab:/programs/Neuro/vasculature2:\$PATH'"
  if [ "$device" = "cuda" ]; then
    sing_flags="$sing_flags --nv"
  fi
  
  # SGE command as an array
  cmd_sge=(
    qsub -P $sge_project -terse
      -N $sge_name
      -A $sge_account
      -l h_vmem=$sge_vmem
      -o $path_sge_log/${task}_${dataset}.log
      -e $path_sge_err/${task}_${dataset}.err
  )
  
  # Add hold_jid if provided
  if [ -n "$hold_jid" ]; then
    echo "Holding pipeline for job $hold_jid"
    cmd_sge+=(-v "HOLD_PIPELINE=1")
    cmd_sge+=(-hold_jid "$hold_jid")
  fi

  #      -l ngpu=$sge_ngpu
  
  # Singularity run command (calling eICAB container directly)
  # The container expects:
  #   -t : input TOF image path (inside container)
  #   -o : output directory (inside container)
  #   -r : resolution for resampling
  #   -d : device (cpu or cuda)
  #   -f : force fla
  #   -s : simple segmentation (optional)
  #   -a : attention (optional)
  cmd_run="singularity run $sing_flags \
    --bind $input_file_abs:$container_input_path:ro \
    --bind $path_output:/output \
    --bind $path_vasculature:/programs/Neuro/vasculature2 \
    --bind $path_tmp:/tmp \
    $path_container \
    -t $container_input_path \
    -o /output \
    -r $resolution \
    -d $device \
    -f \
    $eicab_flags
  "
  
  jid=$( echo "$cmd_run" | "${cmd_sge[@]}" )
  submitted_jobs+=("$jid")
  
  cat <<EOF | tee -a $path_sge_log/${task}_${dataset}.log

############################ eICAB Inference ############################

Job:           $jid

Name:          $sge_name
Project:       $sge_project
Account:       $sge_account
GPU:           $sge_ngpu
Memory:        $sge_vmem
Log dir:       $path_sge_log
Error dir:     $path_sge_err

Input file:    $input_file
Output:        $path_output
Resolution:    $resolution mm
Device:        $device
Simple Seg:    $simple_segmentation
Attention:     $attention

Command:       $cmd_run

########################################################################

EOF

  echo "Job $jid submitted for: $(basename "$input_file")"
  
done

########################################################################################################################
# Summary
########################################################################################################################

echo ""
echo "======================================================================"
echo "Summary: Submitted ${#submitted_jobs[@]} job(s)"
echo "======================================================================"
for i in "${!submitted_jobs[@]}"; do
  echo "  [$((i+1))] Job ${submitted_jobs[$i]}: ${dataset_names[$i]}"
done
echo ""
echo "logs in: $path_sge_log"
echo "errors in: $path_sge_err"
echo "======================================================================"
echo ""

# Output last job ID for chaining (to stderr so it doesn't interfere)
# Using only the last job ID ensures all previous jobs have completed
if [ ${#submitted_jobs[@]} -gt 0 ]; then
  last_idx=$((${#submitted_jobs[@]} - 1))
  last_job_id="${submitted_jobs[$last_idx]}"
  echo "EICAB_LAST_JOB_ID=$last_job_id" >&2
fi
