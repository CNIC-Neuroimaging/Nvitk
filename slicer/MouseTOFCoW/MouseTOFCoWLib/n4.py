"""N4 bias-field correction via Slicer's built-in N4ITKBiasFieldCorrection CLI.

Avoids in-process SimpleITK N4 (known to hard-crash Slicer with large control grids).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def n4_bias_field_correction(
    input_volume_node,
    *,
    shrink_factor: int = 2,
    spline_distance: float | int | None = None,
    remove_temp_nodes: bool = True,
) -> np.ndarray:
    """Run Slicer ``N4ITKBiasFieldCorrection`` and return the corrected array (k,j,i).

    Parameters
    ----------
    input_volume_node
        ``vtkMRMLScalarVolumeNode`` (TOF intensity).
    shrink_factor
        N4ITK shrink factor (Lab ANTs recipe used 2).
    spline_distance
        Optional N4ITK spline distance in **mm**. ``None`` / ``0`` leaves the
        CLI default (auto). Do not pass the Lab ANTs voxel ``spline_param=6``
        here — units differ.
    remove_temp_nodes
        Delete the temporary output volume and CLI node after reading voxels.
    """
    import slicer

    n4_mod = getattr(slicer.modules, "n4itkbiasfieldcorrection", None)
    if n4_mod is None:
        raise RuntimeError(
            "Slicer module N4ITKBiasFieldCorrection is not available "
            "(expected as a built-in CLI). Check Modules settings."
        )

    src_name = input_volume_node.GetName()
    output = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLScalarVolumeNode",
        f"{src_name}_tof_cow_n4_tmp",
    )
    output.CreateDefaultDisplayNodes()
    # Keep temp output out of the way in the UI.
    try:
        output.SetHideFromEditors(True)
    except Exception:
        pass

    parameters: dict = {
        "inputImageName": input_volume_node.GetID(),
        "outputImageName": output.GetID(),
        "shrinkFactor": int(shrink_factor),
        # integer-vector CLI param (see N4ITKBiasFieldCorrection.xml)
        "numberOfIterations": "50,50,50,50",
        "convergenceThreshold": 1e-7,
        # Keep default initialMeshResolution=1,1,1 (safe small B-spline grid).
    }
    if spline_distance is not None and float(spline_distance) > 0:
        parameters["splineDistance"] = float(spline_distance)

    log.info(
        "N4 (N4ITKBiasFieldCorrection CLI): input=%s shrink=%s splineDistance=%s",
        src_name,
        shrink_factor,
        parameters.get("splineDistance", "default"),
    )
    slicer.app.processEvents()

    run_sync = getattr(slicer.cli, "runSync", None)
    if run_sync is not None:
        cli_node = run_sync(n4_mod, None, parameters)
    else:
        cli_node = slicer.cli.run(
            n4_mod, None, parameters, wait_for_completion=True
        )

    try:
        status = int(cli_node.GetStatus())
        errors_mask = int(getattr(cli_node, "ErrorsMask", 0) or 0)
        if errors_mask and (status & errors_mask):
            err = cli_node.GetErrorText() or "unknown N4ITK error"
            raise RuntimeError(f"N4ITKBiasFieldCorrection failed: {err}")
        # Some Slicer builds only expose Completed / status strings.
        status_str = ""
        try:
            status_str = str(cli_node.GetStatusString())
        except Exception:
            pass
        if status_str and "error" in status_str.lower():
            err = cli_node.GetErrorText() or status_str
            raise RuntimeError(f"N4ITKBiasFieldCorrection failed: {err}")

        out = np.asarray(slicer.util.arrayFromVolume(output), dtype=np.float32).copy()
        if out.ndim != 3:
            raise ValueError(f"N4 output is not 3D: shape={out.shape}")
        log.info("N4 (N4ITK) done: shape=%s", out.shape)
        return out
    finally:
        if remove_temp_nodes:
            try:
                slicer.mrmlScene.RemoveNode(cli_node)
            except Exception:
                pass
            try:
                slicer.mrmlScene.RemoveNode(output)
            except Exception:
                pass
