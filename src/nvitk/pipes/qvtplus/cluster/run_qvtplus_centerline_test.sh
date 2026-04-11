#!/bin/bash
#######################################################################################################################
# QVT+ Centerline Test - HPC Cluster Submission Script
#######################################################################################################################
#######################################################################################################################
# Script variables
#######################################################################################################################

# Base directory containing NIFTI and RESULTS
# base_path="/data3/BIOIT_IMAGE/PESA-Brain"
base_path="/data_lab_MCC/imarcoss/LabMCC/"
subjects2exclude=("")

# Centerline test parameters
skip_existing="0"   # Skip if centerline_test output folder already exists [0 | 1]

# Input/output paths (relative to base_path)
qvtplus_results_dir="${base_path}/RESULTS/QVTPlus"   # Input: subject-level QVT+ output directories

# Remove trailing slash if present
qvtplus_results_dir="${qvtplus_results_dir%/}"

#######################################################################################################################
# Scan for subjects with QVT+ outputs
#######################################################################################################################

declare -a subject_names=()
declare -a path_qvtplus=()

echo "=========================================================================="
echo "Scanning for subjects with QVT+ outputs..."
echo "=========================================================================="

# Normal mode: scan QVT+ subject directories
while IFS= read -r -d '' subject_dir; do
  sub_name=$(basename "$subject_dir")

  # Check if subject should be excluded
  if [[ " ${subjects2exclude[@]} " =~ " ${sub_name} " ]]; then
    echo "Skipping $sub_name (in exclusion list)"
    continue
  fi

  # Basic validation: subject dir should include qvtData mat outputs
  if ! compgen -G "$subject_dir/qvtData_ISOfix*.mat" > /dev/null; then
    echo "Warning: Skipping $sub_name (no qvtData_ISOfix*.mat found in $subject_dir)"
    continue
  fi

  # Skip if centerline_test exists (if enabled)
  if [ "$skip_existing" = "1" ] && [ -d "$subject_dir/centerline_test" ]; then
    echo "Skipping $sub_name (centerline_test already exists at $subject_dir/centerline_test)"
    continue
  fi

  subject_names+=("$sub_name")
  path_qvtplus+=("$subject_dir")
  echo "  - $sub_name"
done < <(find "$qvtplus_results_dir" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

if [ ${#subject_names[@]} -eq 0 ]; then
  echo "Error: No subjects found with valid QVT+ outputs"
  exit 1
fi

echo "=========================================================================="
echo "Found ${#subject_names[@]} subject(s) ready for centerline test."
echo "=========================================================================="
echo ""

########################################################################################################################
# Submission variables (common for all jobs)
########################################################################################################################

bioimaging_project="PesaBrain"
task="QVTPlus_CenterlineTest"

sge_project="FSC"
sge_account="Prod"
sge_vmem="32G"

path_sge_log="/data3/BIOIT_IMAGE/BioImaging/env/logs/${bioimaging_project}"
path_sge_err="/data3/BIOIT_IMAGE/BioImaging/env/errs/${bioimaging_project}"
if [ ! -d "$path_sge_log" ]; then mkdir -p "$path_sge_log"; fi
if [ ! -d "$path_sge_err" ]; then mkdir -p "$path_sge_err"; fi

path_container="/data3/BIOIT_IMAGE/Containers/QVTV2Plus_v2026.04.09.sif"
path_tmp="/data_tmp"

########################################################################################################################
# Execution: Loop through all subjects and submit jobs
########################################################################################################################

submitted_jobs=()

for idx in "${!subject_names[@]}"; do
  subject="${subject_names[$idx]}"
  qvtplus_path="${path_qvtplus[$idx]}"

  sge_name="${bioimaging_project}_${task}_${subject}"

  # Get absolute path
  qvtplus_path_abs=$(realpath "$qvtplus_path")

  # Remove old err file if it exists
  if [ -f "$path_sge_err/${task}_${subject}.err" ]; then
    rm "$path_sge_err/${task}_${subject}.err"
  fi

  # SGE command as an array
  cmd_sge=(
    qsub -P $sge_project -terse
      -N $sge_name
      -A $sge_account
      -l h_vmem=$sge_vmem
      -o $path_sge_log/${task}_${subject}.log
      -e $path_sge_err/${task}_${subject}.err
  )

  # Run MATLAB function in container.
  # Expected input is subject-level QVT+ output directory.
  cmd_run="singularity run --cleanenv \
    --bind ${base_path}:/data \
    --bind $path_tmp:/tmp \
    $path_container \
    /data/RESULTS/QVTPlus/${subject}
  "

  # Submit the command directly to qsub as a binary job and capture the job id
  # Use -b y to submit a binary and -cwd to run in current working directory
  jid=$( echo "$cmd_run" | "${cmd_sge[@]}" )
  submitted_jobs+=("$jid")

  cat <<EOF | tee -a $path_sge_log/${task}_${subject}.log

######################## QVT+ Centerline Test ########################

Job:           $jid

Name:          $sge_name
Project:       $sge_project
Account:       $sge_account
Memory:        $sge_vmem
Log dir:       $path_sge_log
Error dir:     $path_sge_err

Subject:       $subject
QVT+ output:   $qvtplus_path_abs
Skip existing: $skip_existing

Command:       $cmd_run

######################################################################

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

# Output last job ID for chaining
if [ ${#submitted_jobs[@]} -gt 0 ]; then
  last_idx=$((${#submitted_jobs[@]} - 1))
  last_job_id="${submitted_jobs[$last_idx]}"
  echo "QVTPLUS_CENTERLINE_LAST_JOB_ID=$last_job_id" >&2
fi
