"""Synthetic DICOM series: identifiers in derived NIfTIs, and per-slice rescale order.

1. imread(DICOM) -> imsave(.nii.gz) keeps patient identifiers in the NIfTI header extension.
2. revert_scaling applies per-slice RescaleSlope values in InstanceNumber order to slices stacked
   by position; a series numbered against its slice order gets the wrong slope on every slice.
"""

import json
import logging
import os
import tempfile

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from nvload import load

io = load("nvitk.io.imageio", stub=("nvitk", "nvitk.io.conversors"))
logging.disable(logging.WARNING)


def write_series(folder, slopes, instance_against_position):
    series, study, n = generate_uid(), generate_uid(), len(slopes)
    for k, slope in enumerate(slopes):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.PatientName, ds.PatientID, ds.PatientBirthDate = "Doe^Jane", "PESA-12345", "19700101"
        ds.AccessionNumber, ds.StudyDate, ds.SeriesDate, ds.SeriesTime = "ACC987", "20240102", "20240102", "101010"
        ds.Modality, ds.SeriesInstanceUID, ds.StudyInstanceUID, ds.SeriesNumber = "CT", series, study, 3
        ds.InstanceNumber = n - k if instance_against_position else k + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, 2.0 * k]
        ds.PixelSpacing, ds.SliceThickness = [0.5, 0.5], 2.0
        ds.Rows = ds.Columns = 8
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = slope, 0
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.PixelData = np.full((8, 8), 100, np.int16).tobytes()
        pydicom.dcmwrite(os.path.join(folder, f"im{k:03d}.dcm"), ds, enforce_file_format=True)


def single(result):
    return result[0] if isinstance(result, list) else result


folder = tempfile.mkdtemp()
write_series(folder, [1] * 6, instance_against_position=True)
out = os.path.join(tempfile.mkdtemp(), "ct.nii.gz")
io.imsave(out, single(io.imread(folder)))
payload = json.loads(nib.load(out).header.extensions[0].get_content().rstrip(b"\0"))
print("identifiers in the derived NIfTI:",
      {k: payload[k] for k in ("PatientName", "PatientID", "PatientBirthDate", "AccessionNumber") if k in payload})

for against in (True, False):
    folder = tempfile.mkdtemp()
    write_series(folder, [1, 2, 3, 4, 5, 6], instance_against_position=against)
    raw = np.asarray(single(io.imread(folder, revert_scaling=True)).data)
    label = "InstanceNumber against position" if against else "InstanceNumber along position"
    print(f"revert_scaling, {label}: {[round(float(raw[..., z].mean()), 2) for z in range(raw.shape[2])]} (expected 100)")
