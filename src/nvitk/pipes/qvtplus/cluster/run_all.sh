#!/bin/bash
#######################################################################################################################
# PESA-Brain Complete Pipeline - HPC Cluster Submission Script
# Runs three pipelines sequentially: 1. DICOM to NIfTI, 2. eICAB, 3. QVT+
#######################################################################################################################

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================================="
echo "PESA-Brain Complete Pipeline"
echo "=========================================================================="
echo "This script will run four pipelines sequentially:"
echo "  1. DICOM to NIfTI conversion"
echo "  2. Reorganize NIfTI files into eICAB/QVT+ compatible structure"
echo "  3. eICAB Circle of Willis segmentation"
echo "  4. QVT+ 4D Flow analysis"
echo "=========================================================================="
echo ""

#######################################################################################################################
# Step 1: DICOM to NIfTI Conversion
#######################################################################################################################

echo "=========================================================================="
echo "Step 1/4: DICOM to NIfTI Conversion"
echo "=========================================================================="

# Run dicom2nifti script and capture job ID from stderr
dicom2nifti_output=$(bash "$SCRIPT_DIR/run_dicom2nifti.sh" 2>&1)
dicom2nifti_jid=$(echo "$dicom2nifti_output" | grep "DICOM2NIfTI_JOB_ID=" | cut -d'=' -f2)

echo "DICOM to NIfTI conversion job output: $dicom2nifti_output"

if [ -z "$dicom2nifti_jid" ]; then
  echo "Error: Failed to get job ID from DICOM to NIfTI conversion"
  exit 1
fi 

echo "DICOM to NIfTI conversion job submitted: $dicom2nifti_jid"
echo "Waiting for this job to complete before starting reorganization..."
echo ""

#######################################################################################################################
# Step 2: Reorganize NIfTI Files
#######################################################################################################################

echo "=========================================================================="
echo "Step 2/4: Reorganize NIfTI Files"
echo "=========================================================================="
echo "This step will wait for DICOM to NIfTI conversion (job $dicom2nifti_jid) to complete."
echo ""

# Run reorganize script with hold_jid, and capture job ID from stderr
reorganize_output=$(bash "$SCRIPT_DIR/run_reorganize.sh" "$dicom2nifti_jid" 2>&1)
reorganize_jid=$(echo "$reorganize_output" | grep "REORGANIZE_JOB_ID=" | cut -d'=' -f2)

echo "Reorganization job output: $reorganize_output"

if [ -z "$reorganize_jid" ]; then
  echo "Error: Failed to get job ID from reorganization"
  exit 1
fi

echo "Reorganization job submitted: $reorganize_jid"
echo "Waiting for this job to complete before starting eICAB..."
echo ""

#######################################################################################################################
# Step 3: eICAB Inference
#######################################################################################################################

echo "=========================================================================="
echo "Step 3/4: eICAB Circle of Willis Segmentation"
echo "=========================================================================="
echo "This step will wait for reorganization (job $reorganize_jid) to complete."
echo ""

# Get DICOM input directory from dicom2nifti script
# Extract the active (uncommented) input path
dicom_input=$(grep -E "^input=" "$SCRIPT_DIR/run_dicom2nifti.sh" | grep -v "^#" | head -1 | sed 's/^input="\(.*\)"$/\1/')
if [ -z "$dicom_input" ]; then
  echo "Warning: Could not extract DICOM input directory from run_dicom2nifti.sh"
  echo "eICAB and QVT+ scripts will try to infer it automatically"
fi

# Run eicab script with hold_jid and DICOM input directory, capture last job ID from stderr
eicab_output=$(bash "$SCRIPT_DIR/run_eicab_inference.sh" "$reorganize_jid" "$dicom_input" 2>&1)
eicab_last_jid=$(echo "$eicab_output" | grep "EICAB_LAST_JOB_ID=" | cut -d'=' -f2)

echo "eICAB inference job output: $eicab_output"

if [ -z "$eicab_last_jid" ]; then
  echo "Error: Failed to get last job ID from eICAB inference"
  exit 1
fi

# Count number of eICAB jobs from the summary
eicab_job_count=$(echo "$eicab_output" | grep "Summary: Submitted" | sed 's/.*Submitted \([0-9]*\) job(s).*/\1/' || echo "0")
echo "eICAB inference: $eicab_job_count job(s) submitted (waiting for reorganization job $reorganize_jid)"
echo "Last eICAB job ID: $eicab_last_jid (QVT+ will wait for this job, ensuring all eICAB jobs complete)"
echo "Waiting for all eICAB jobs to complete before starting QVT+..."
echo ""

#######################################################################################################################
# Step 4: QVT+ Processing
#######################################################################################################################

echo "=========================================================================="
echo "Step 4/4: QVT+ 4D Flow Analysis"
echo "=========================================================================="
echo "This step will wait for all eICAB jobs to complete."
echo ""

# Run qvtplus script with hold_jid and DICOM input directory (last eICAB job ID)
qvtplus_output=$(bash "$SCRIPT_DIR/run_qvtplus.sh" "$eicab_last_jid" "$dicom_input" 2>&1)

# Extract QVT+ job IDs from output (they're printed in the summary)
qvtplus_job_ids=$(echo "$qvtplus_output" | grep -E "Job [0-9]+ submitted for:" | sed 's/.*Job \([0-9]*\) submitted for:.*/\1/' | tr '\n' ',' | sed 's/,$//')

echo "QVT+ processing job output: $qvtplus_output"

if [ -n "$qvtplus_job_ids" ]; then
  qvtplus_job_count=$(echo "$qvtplus_job_ids" | tr ',' '\n' | wc -l)
  echo "QVT+ processing: $qvtplus_job_count job(s) submitted (waiting for last eICAB job: $eicab_last_jid)"
  echo "Job IDs: $qvtplus_job_ids"
else
  echo "Warning: Could not extract QVT+ job IDs from output"
fi

echo ""

#######################################################################################################################
# Final Summary
#######################################################################################################################

echo "=========================================================================="
echo "Pipeline Summary"
echo "=========================================================================="
echo "Step 1 - DICOM to NIfTI:"
echo "  Job ID: $dicom2nifti_jid"
echo ""
echo "Step 2 - Reorganize NIfTI Files:"
echo "  Job ID: $reorganize_jid"
echo ""
echo "Step 3 - eICAB Inference:"
echo "  Last Job ID: $eicab_last_jid (used for chaining)"
echo "  Count: $eicab_job_count job(s)"
echo ""
echo "Step 4 - QVT+ Processing:"
if [ -n "$qvtplus_job_ids" ]; then
  echo "  Job IDs: $qvtplus_job_ids"
  echo "  Count: $qvtplus_job_count job(s)"
else
  echo "  (Job IDs could not be extracted)"
fi
echo ""
echo "=========================================================================="
echo "All jobs have been submitted!"
echo "The pipeline will execute sequentially:"
echo "  1. DICOM to NIfTI conversion must complete"
echo "  2. Then reorganization will run"
echo "  3. Then all eICAB jobs will run in parallel"
echo "  4. Then all QVT+ jobs will run in parallel"
echo "=========================================================================="
echo ""
echo "Monitor job status with: qstat"
echo ""
