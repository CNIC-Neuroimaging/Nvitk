from __future__ import annotations

import numpy as np

from nvitk.db import DataRepo
from nvitk.io.conversors.phase2volume import phase2volume
from nvitk.io.readers.nifti import read_nifti
from nvitk.io.writers.nifti import write_nifti


def test_phase2volume_generates_expected_velocity_and_cd(tmp_path):
    dataset_root = tmp_path / "dataset"
    repo = DataRepo(dataset_root, auto_scaffold=True)

    patient_dir = tmp_path / "PESA001"
    ap_dir = patient_dir / "4DFlow" / "AP"
    rl_dir = patient_dir / "4DFlow" / "RL"
    fh_dir = patient_dir / "4DFlow" / "FH"
    ap_dir.mkdir(parents=True)
    rl_dir.mkdir(parents=True)
    fh_dir.mkdir(parents=True)

    angio = np.full((2, 2, 2, 2), 10.0, dtype=float)
    ap_phase = np.full((2, 2, 2, 2), 1.0, dtype=float)
    rl_phase = np.full((2, 2, 2, 2), 2.0, dtype=float)
    fh_phase = np.full((2, 2, 2, 2), 2.0, dtype=float)

    metadata = {"axes": "XYZT"}
    write_nifti(ap_dir / "patient_m.nii", angio, metadata=metadata)
    write_nifti(ap_dir / "patient_ph.nii", ap_phase, metadata=metadata)
    write_nifti(rl_dir / "patient_ph.nii", rl_phase, metadata=metadata)
    write_nifti(fh_dir / "patient_ph.nii", fh_phase, metadata=metadata)
    (ap_dir / "patient.json").write_text('{"VelocityEncoding": 10}', encoding="utf-8")

    outputs = phase2volume(patient_dir, dataset_root=dataset_root)

    assert any(path.name == "VelocityMagnitude_4D.nii" for path in outputs)
    assert any(path.name == "ComplexDifference_4D.nii" for path in outputs)

    vmag, _ = read_nifti(patient_dir / "4DFlow" / "VelocityMagnitude_4D.nii")
    cd, _ = read_nifti(patient_dir / "4DFlow" / "ComplexDifference_4D.nii")

    expected_vmag = np.sqrt((20.0**2) + (10.0**2) + (20.0**2))
    expected_cd = 10.0 * np.sin((np.pi / 2.0 * expected_vmag) / 100.0)

    assert np.allclose(vmag, expected_vmag)
    assert np.allclose(cd, expected_cd)

    assets = repo.assets()
    assert len(assets) == 6
    assert set(assets["pipeline_name"]) == {"phase2volume"}
