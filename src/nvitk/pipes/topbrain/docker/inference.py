"""Grand Challenge entry point for the ToPBrain algorithm container.

Reads the single ``.mha`` supplied on the input socket, runs nnU-Net inference plus the
pipeline's topology post-processing, and writes a mask of **identical shape** to the output
socket. Sockets follow the challenge's published interface::

    /input/images/head-{ct,mr}-angio/<uuid>.mha
    /output/images/head-{ct,mr}-angio-segmentation/<uuid>.mha

The track is chosen at run time rather than baked in: TA36 is modality-agnostic, so one
container serves both sockets and picks whichever one actually has an image. That also means a
single build can be submitted to either portal.

Runs offline with no network, weights baked into the image at ``/opt/algorithm/model``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

INPUT_ROOT = Path(os.environ.get("TOPBRAIN_INPUT_ROOT", "/input"))
OUTPUT_ROOT = Path(os.environ.get("TOPBRAIN_OUTPUT_ROOT", "/output"))
MODEL_DIR = Path(os.environ.get("TOPBRAIN_MODEL_DIR", "/opt/algorithm/model"))

#: Input socket → output socket, in the order they are probed.
SOCKETS: tuple[tuple[str, str], ...] = (
    ("head-ct-angio", "head-ct-angio-segmentation"),
    ("head-mr-angio", "head-mr-angio-segmentation"),
)

#: Modality per input socket, for the intensity harmonisation stage 0 applied at training time.
SOCKET_MODALITY: dict[str, str] = {"head-ct-angio": "ct", "head-mr-angio": "mr"}


def find_input() -> tuple[Path, str, str]:
    """Locate the single supplied image; returns ``(path, input_socket, output_socket)``.

    Grand Challenge feeds one image at a time, so exactly one socket is populated.
    """
    for input_socket, output_socket in SOCKETS:
        directory = INPUT_ROOT / "images" / input_socket
        if not directory.is_dir():
            continue
        images = sorted(directory.glob("*.mha")) + sorted(directory.glob("*.mha.gz"))
        if images:
            return images[0], input_socket, output_socket
    raise FileNotFoundError(
        f"No .mha found under {INPUT_ROOT}/images/{{{', '.join(s for s, _ in SOCKETS)}}}."
    )


def main() -> int:
    """Predict for the supplied case and write the mask to the matching output socket."""
    from nvitk.core.array import to_numpy
    from nvitk.normalization import harmonize_modality
    from nvitk.pipes.topbrain import labels as lbl
    from nvitk.segmentation.vessel_postprocess import postprocess_labelmap

    image_path, input_socket, output_socket = find_input()
    modality = SOCKET_MODALITY[input_socket]
    print(f"[topbrain] input={image_path} socket={input_socket} modality={modality}", flush=True)

    image = sitk.ReadImage(str(image_path))
    original_size = image.GetSize()

    # Harmonise exactly as stage 0 did: the model was trained on that intensity range, and a
    # raw HU or raw TOF volume is a different distribution entirely.
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    # to_numpy, not np.asarray: nvitk may hand back a CuPy array when a GPU is visible, and
    # SimpleITK only accepts host memory.
    harmonised = to_numpy(harmonize_modality(array, modality)).astype(np.float32)
    prepared = sitk.GetImageFromArray(harmonised)
    prepared.CopyInformation(image)

    work_in = Path("/tmp/topbrain_in")
    work_out = Path("/tmp/topbrain_out")
    work_in.mkdir(parents=True, exist_ok=True)
    work_out.mkdir(parents=True, exist_ok=True)
    # nnU-Net identifies channels by the _0000 suffix.
    sitk.WriteImage(prepared, str(work_in / "case_0000.mha"))

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,  # labels are lateralised; see the trainer docstring
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        allow_tqdm=False,
    )
    # initialize_from_trained_model_folder reads the trainer name out of the model folder and
    # resolves the class to rebuild the architecture — which only works because the in-tree
    # build (carrying the ToPBrain trainers) is the nnunetv2 on PYTHONPATH here.
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR),
        use_folds=None,  # every fold present in the image
        checkpoint_name="checkpoint_final.pth",
    )
    predictor.predict_from_files(
        str(work_in), str(work_out),
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=1, num_processes_segmentation_export=1,
    )

    produced = sorted(work_out.glob("case.*"))
    if not produced:
        raise RuntimeError(f"nnU-Net produced no output under {work_out}.")
    mask_image = sitk.ReadImage(str(produced[0]))
    mask = sitk.GetArrayFromImage(mask_image)

    cleaned = to_numpy(
        postprocess_labelmap(
            mask,
            labels=sorted(lbl.label_map("ta36")),
            spacing=tuple(reversed(mask_image.GetSpacing())),  # sitk arrays are (z, y, x)
            min_volume_mm3=5.0,
        )
    ).astype(np.uint8)

    output = sitk.GetImageFromArray(cleaned)
    output.CopyInformation(image)
    if output.GetSize() != original_size:
        raise RuntimeError(
            f"Output size {output.GetSize()} != input size {original_size}; the challenge "
            f"requires an identical grid."
        )

    destination = OUTPUT_ROOT / "images" / output_socket
    destination.mkdir(parents=True, exist_ok=True)
    out_path = destination / image_path.name
    sitk.WriteImage(output, str(out_path), useCompression=True)
    print(f"[topbrain] wrote {out_path} labels={sorted(np.unique(cleaned).tolist())[:8]}...",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
