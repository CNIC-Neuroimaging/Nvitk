#!/bin/bash
#######################################################################################################################
# DICOM to NIfTI Conversion - HPC Cluster Submission Script
#######################################################################################################################
#######################################################################################################################
# Script variables
#######################################################################################################################

# Input directory containing subject folders with DICOM files
# input="/data3/BIOIT_IMAGE/PESA-Brain/DATA/DICOM"
input="/mnt/scratch/imarcoss/DICOM"

# Output directory for NIfTI files (will create subject subdirectories)
# output="/data3/BIOIT_IMAGE/PESA-Brain/DATA/Nifti"
output="/mnt/scratch/imarcoss/NIFTI_UNORGANIZED"

subjects2exclude=("")

# DICOM to NIfTI parameters (matching download_cli.py)
naming="AccessionNumber_SeriesDescription_SeriesNumber"  # Naming format for output files
rescale_type="FP"                                        # Rescale type: FP (Floating Point) or DV (Displayed Values)
skip_existing="1"                                        # Skip existing outputs [0 | 1] (default: 0)

# Remove trailing slash if present
input="${input%/}"
output="${output%/}"

# Validation: Check if input directory exists
if [ ! -d "$input" ]; then
  echo "Error: Input directory does not exist: $input"
  exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$output" ]; then mkdir -p "$output"; fi

#######################################################################################################################
# Scan for subjects with DICOM data (for information only)
#######################################################################################################################

declare -a subject_names=()

echo "=========================================================================="
echo "Scanning for subjects with DICOM data..."
echo "=========================================================================="

# Scan for subjects with DICOM directories (just for counting/info)
while IFS= read -r -d '' subject_dir; do
  sub_name=$(basename "$subject_dir")
  
  # Check if subject should be excluded
  if [[ " ${subjects2exclude[@]} " =~ " ${sub_name} " ]]; then
    echo "Skipping $sub_name (in exclusion list)"
    continue
  fi
  
  # Check if directory has DICOM files (look for .dcm files or DICOMDIR)
  has_dicom=false
  if [ -f "$subject_dir/DICOMDIR" ] || [ -f "$subject_dir/DIRFILE" ]; then
    has_dicom=true
  elif find "$subject_dir" -maxdepth 3 -type f \( -name "*.dcm" -o -name "*.DCM" \) 2>/dev/null | head -1 | read; then
    has_dicom=true
  fi
  
  if [ "$has_dicom" = false ]; then
    echo "Warning: Skipping $sub_name (no DICOM files found)"
    continue
  fi
  
  # Add to list for info
  subject_names+=("$sub_name")
  echo "  - $sub_name"
done < <(find "$input" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

if [ ${#subject_names[@]} -eq 0 ]; then
  echo "Error: No subjects found with DICOM data"
  exit 1
fi

echo "=========================================================================="
echo "Found ${#subject_names[@]} subject(s) ready for DICOM to NIfTI conversion."
echo "Will process all subjects in a single job with --multifile flag."
echo "=========================================================================="
echo ""

########################################################################################################################
# Submission variables (common for all jobs)
########################################################################################################################

bioimaging_project="PesaBrain"
task="DICOM2NIfTI"

sge_project="FSC"
sge_account="Prod"
sge_vmem="40G"              # Memory for DICOM to NIfTI conversion
n_procs="1"                 # Number of parallel processes for patient conversion (default: 1, no multiprocessing)

path_sge_log="/data3/BIOIT_IMAGE/BioImaging/env/logs/${bioimaging_project}"
path_sge_err="/data3/BIOIT_IMAGE/BioImaging/env/errs/${bioimaging_project}"
if [ ! -d "$path_sge_log" ]; then mkdir -p "$path_sge_log"; fi
if [ ! -d "$path_sge_err" ]; then mkdir -p "$path_sge_err"; fi

# Use the same container as other pesa_brain scripts, or a general Python container
# Adjust this path to match your container setup
path_container="/data3/BIOIT_IMAGE/Containers/pesa-brain_v2026.1.26.sif"
path_src="/data3/BIOIT_IMAGE/BioImaging/src/"
path_tmp="/data_tmp"

# Bind paths
bind_src="/${bioimaging_project}/src/"
bind_data="/${bioimaging_project}/data/"
bind_out="/${bioimaging_project}/output/"
bind_tmp="/tmp"

script="imaging/conversion/dcm2nii.py"

########################################################################################################################
# Execution: Submit single job for all patients with --multifile flag
########################################################################################################################

# Get absolute paths
input_abs=$(realpath "$input")
output_abs=$(realpath -m "$output")  # -m allows creating path if it doesn't exist

# Create output directory
mkdir -p "$output_abs"

# Remove old err file if it exists
sge_name="${bioimaging_project}_${task}_ALL"
if [ -f "$path_sge_err/${task}_ALL.err" ]; then
  rm "$path_sge_err/${task}_ALL.err"
fi

# Build skip-existing flag
skip_flag=""
if [ "$skip_existing" = "1" ]; then
  skip_flag="--skip-existing"
fi

# Accept optional hold_jid parameter (for chaining jobs)
hold_jid="${1:-}"

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
  cmd_sge+=(-hold_jid "$hold_jid")
  echo "Waiting for job(s) to complete: $hold_jid"
fi

# Singularity run command
# The command matches exactly what's used in download_cli.py with --multifile flag
# All patients will be processed in a single container call
cmd_run="singularity exec \
  -B $path_src:$bind_src \
  -B $input_abs:$bind_data \
  -B $output_abs:$bind_out \
  -B $path_tmp:$bind_tmp \
  $path_container python3 $bind_src$script \
  --input $bind_data \
  --output $bind_out \
  --multifile \
  --save-metadata \
  --force-ras \
  --rescale-type $rescale_type \
  --naming $naming \
  $skip_flag
"

jid=$( echo "$cmd_run" | "${cmd_sge[@]}" )

cat <<EOF | tee -a $path_sge_log/${task}_ALL.log

############################ DICOM to NIfTI Conversion (All Patients) ############################

Job:           $jid

Name:          $sge_name
Project:       $sge_project
Account:       $sge_account
Memory:        $sge_vmem
Log dir:       $path_sge_log
Error dir:     $path_sge_err

Input:         $input_abs
Output:        $output_abs
Total subjects: ${#subject_names[@]}
Naming:        $naming
Rescale type:  $rescale_type
Skip existing: $skip_existing
N processes:   $n_procs

Command:       $cmd_run

########################################################################

EOF

echo "Job $jid submitted for all ${#subject_names[@]} subject(s)"

# Output job ID for chaining (to stderr so it doesn't interfere with normal output)
echo "DICOM2NIfTI_JOB_ID=$jid" >&2

########################################################################################################################
# Summary
########################################################################################################################

echo ""
echo "======================================================================"
echo "Summary: Submitted 1 job for ${#subject_names[@]} subject(s)"
echo "======================================================================"
echo "  Job $jid: Processing all subjects with --multifile flag"
echo ""
echo "logs in: $path_sge_log"
echo "errors in: $path_sge_err"
echo "======================================================================"
echo ""
