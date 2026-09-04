"""Zeiss OCT raw DICOM storage: extract B-scans / volumes for the conversion pipeline."""

from __future__ import annotations

import io
import os
import struct
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from nvitk.core.logger import Logger

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pydicom
    from pydicom.tag import Tag
except Exception:
    pydicom = None
    Tag = None

# SOP Class UID for Raw Data Storage (often used by Zeiss private data).
RAW_DATA_SOP_UID = "1.2.840.10008.5.1.4.1.1.66"

__all__ = [
    "ZEISS_FIXATION_MM_PER_UNIT",
    "has_obfuscated_cube_payload",
    "decode_obfuscated_frame",
    "is_obfuscated_payload",
    "reconstruct_obfuscated_jp2",
    "derived_analysis_affine",
    "derived_analysis_geometry",
    "zeiss_referenced_sources",
    "ZeissOctAligner",
    "ZeissOctCube",
    "ZeissOctImage",
    "ZEISS_OCT_CUBE_CONTAINERS",
    "ZEISS_OCT_IMAGE_CONTAINERS",
    "extract_zeiss_oct_cubes",
    "extract_zeiss_oct_images",
    "extract_zeiss_raw_oct",
    "has_zeiss_oct_cubes",
    "is_zeiss_raw_storage",
]

log = Logger()


# ---------------------------------------------------------------------------
# Zeiss CIRRUS "CapeCod" private OCT containers
# ---------------------------------------------------------------------------
#
# A CIRRUS Raw Data Storage object wraps every image stack it carries in its own
# private 99CZM_CapeCod_OctGeneral (0407) sequence. Each container holds a
# geometry header plus a per-frame sub-sequence with the encoded frames:
#
#   (0407,1001) US  rows of the decoded frame (A-scans across the B-scan)
#   (0407,1002) US  columns of the decoded frame (depth samples)
#   (0407,1003) US  number of frames (B-scans)
#   (0407,1005) SQ  per-frame sequence; each item carries (0407,1006) OB
#   (0407,100A/100B/100C) DS  device-reported x / depth / y spacing in mm
#
# An OCT-angiography acquisition stores two co-registered cubes in one object -
# the structural ("anatomical") cube and the flow cube - in distinct containers,
# so they split by tag and need no anatomical heuristic.
ZEISS_OCT_CUBE_CONTAINERS: dict[int, str] = {
    0x10A1: "cube_z",  # structural / anatomical OCT cube
    0x10AD: "FlowCube_z",  # OCT-angiography flow (decorrelation) cube
}

# Single-frame containers holding 2-D images rather than a volume, as
# ``element -> (output name, quarter turns to apply to the decoded frame)``.
# The rotations match the orientation the CIRRUS raw export writes.
ZEISS_OCT_IMAGE_CONTAINERS: dict[int, tuple[str, int]] = {
    0x10A6: ("lslo", 3),  # line-scanning ophthalmoscope fundus
    0x10B5: ("iris", 0),  # iris camera
    0x10A7: ("enface", 0),  # depth projection over the cube's en-face grid
}

_TAG_FRAME_ROWS = (0x0407, 0x1001)
_TAG_FRAME_COLS = (0x0407, 0x1002)
_TAG_FRAME_COUNT = (0x0407, 0x1003)
_TAG_FRAME_SEQUENCE = (0x0407, 0x1005)
_TAG_FRAME_DATA = (0x0407, 0x1006)
_TAG_SPACING_X = (0x0407, 0x100A)
_TAG_SPACING_DEPTH = (0x0407, 0x100B)
_TAG_SPACING_Y = (0x0407, 0x100C)
_TAG_SCAN_PATTERN = (0x0405, 0x1001)  # e.g. ANGIO_6MM, MACULAR_CUBE

# Fixation offset: where the scan was centred, relative to the fovea. Zero for every
# macula-centred protocol and a fixed non-zero pair for the disc-centred ones, with the
# horizontal component flipping sign between eyes.
_TAG_FIXATION_X = (0x0405, 0x1032)  # horizontal, along the A-scan axis
_TAG_FIXATION_Y = (0x0405, 0x1033)  # vertical, along the B-scan axis

#: Millimetres per unit of the fixation-offset tags.
#:
#: Calibrated, not documented - Zeiss's private dictionary is not public. 117 units (the
#: disc-centred value) measured 4.40 mm and 4.60 mm on the two eyes of the reference study,
#: by LSLO cross-correlation and by en-face registration, which brackets the expected
#: fovea-to-disc distance. Treat disc-centred placement as accurate to a few tenths of a
#: millimetre: the tags record a fixation-target angle, so the distance it subtends on the
#: retina varies with axial length.
ZEISS_FIXATION_MM_PER_UNIT = 0.038


@dataclass(frozen=True)
class ZeissOctCube:
    """One volumetric cube extracted from a Zeiss private OCT container."""

    kind: str
    array: np.ndarray
    affine: np.ndarray
    meta: dict[str, Any]


@dataclass(frozen=True)
class ZeissOctImage:
    """One 2-D image (fundus, iris or en-face projection) from a Zeiss private OCT container."""

    kind: str
    array: np.ndarray
    meta: dict[str, Any]


def _container_spacing_mm(item: Any) -> tuple[float, float, float] | None:
    """Device-reported ``(x, depth, y)`` voxel spacing in mm from a container header.

    Returns None unless all three of (0407,100A/100B/100C) are present and positive
    (the en-face containers store zeros).
    """
    values = []
    for tag in (_TAG_SPACING_X, _TAG_SPACING_DEPTH, _TAG_SPACING_Y):
        try:
            values.append(float(item[tag].value))
        except Exception:
            return None
    return (values[0], values[1], values[2]) if all(value > 0 for value in values) else None


def _fixation_offset_units(ds: Any) -> tuple[float, float] | None:
    """Scan-centre offset from the fovea in raw tag units, or None when the tags are absent."""
    values = []
    for tag in (_TAG_FIXATION_X, _TAG_FIXATION_Y):
        element = ds.get(tag)
        if element is None:
            return None
        try:
            values.append(float(element.value))
        except Exception:
            return None
    return (values[0], values[1])


def _shared_frame_origin(
    offset_mm: tuple[float, float],
    volume: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    """World coordinate of voxel ``(0, 0, 0)`` for a cube placed at *offset_mm* in the shared frame.

    Only the in-plane axes are placed: the cube is centred on its measured offset from the
    eye's reference scan, so fields of different sizes end up concentric rather than sharing a
    corner. Depth is left at zero - the axial position of an OCT cube is a per-acquisition
    reference-arm setting with no shared anatomical origin, and nothing in the file records it.
    """
    return (
        offset_mm[0] - volume.shape[0] * spacing[0] / 2.0,
        0.0,
        offset_mm[1] - volume.shape[2] * spacing[2] / 2.0,
    )


# ---------------------------------------------------------------------------
# Obfuscated payloads (LuraWave-encoded series)
# ---------------------------------------------------------------------------
#
# Some Zeiss scan types (macular, optic-disc and anterior-segment cubes on this export
# path) store their frames scrambled instead of as a plain JP2: every seventh byte is
# XORed with 0x5A, and the parts of the JP2 file are written out of order. Undoing the
# XOR is proven - it makes the container, the codestream markers and the encoder comment
# parse cleanly - but the byte ordering is only partly understood, so a reconstruction is
# accepted here ONLY when the packet walk lands exactly on the end of the tile-part.
_XOR_STRIDE = 7
_XOR_KEY = 0x5A
_JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  "
_SOC_SIZ = b"\xff\x4f\xff\x51"
_SOD = b"\xff\x93"
_COD = b"\xff\x52"


def _deobfuscate(payload: bytes) -> bytes:
    """Undo the stride-7 XOR that scrambles these payloads."""
    buf = np.frombuffer(payload, np.uint8).copy()
    buf[::_XOR_STRIDE] ^= _XOR_KEY
    return buf.tobytes()


def is_obfuscated_payload(payload: Any) -> bool:
    """True when *payload* is not a plain JP2 but becomes one once the XOR is undone."""
    if not isinstance(payload, (bytes, bytearray)) or len(payload) < 64:
        return False
    if bytes(payload[:8]) == _JP2_SIGNATURE[:8]:
        return False
    return _deobfuscate(bytes(payload)).find(_JP2_SIGNATURE) > 0


def _codestream_parameters(data: bytes, soc: int) -> tuple[int, int, int, int, int, int] | None:
    """``(width, height, levels, layers, xcb, ycb)`` from the SIZ and COD markers."""
    try:
        width, height = struct.unpack(">II", data[soc + 8 : soc + 16])
        cod = data.index(_COD, soc)
        if data[cod + 4] != 0x00:  # precincts / SOP / EPH are outside what the walker covers
            return None
        layers = struct.unpack(">H", data[cod + 6 : cod + 8])[0]
        levels, xcb, ycb, cbstyle = data[cod + 9], data[cod + 10], data[cod + 11], data[cod + 12]
        if data[cod + 5] != 0 or cbstyle != 0x00:  # non-LRCP, or multiple codeword segments
            return None
    except Exception:
        return None
    return width, height, levels, layers, xcb + 2, ycb + 2


def reconstruct_obfuscated_jp2(payload: bytes, require_verified: bool = True) -> bytes | None:
    """Rebuild a plain JP2 from an obfuscated payload, or None when reassembly is impossible.

    Reassembles under the one ordering rule observed to hold: the first packet stays put and
    the remainder is a single circular run starting ``len(header) + len(first packet)`` bytes
    in. With *require_verified* the result is returned only when a full packet walk consumes
    the tile-part exactly; without it the reassembly is returned regardless, which decodes to
    an image whose pixels are known to be wrong.
    """
    from ._j2k_packets import PacketWalker

    data = _deobfuscate(bytes(payload))
    start = data.find(_JP2_SIGNATURE)
    if start <= 0:
        return None
    try:
        soc = data.index(_SOC_SIZ, start)
        sod = data.index(_SOD, soc) + 2
    except ValueError:
        return None
    params = _codestream_parameters(data, soc)
    if params is None:
        return None
    width, height, levels, layers, xcb, ycb = params

    header, leading, trailing, eoc = data[start:sod], data[:start], data[sod : len(data) - 2], data[-2:]
    walker = PacketWalker(width, height, levels, layers, xcb, ycb)
    try:
        header_end, body = walker.packet(trailing, 0, 0, 0)
    except Exception:
        return None
    first = header_end + body
    rest = leading + trailing[first:]
    jump = len(header) + first
    if jump > len(rest):
        return None
    packets = trailing[:first] + rest[jump:] + rest[:jump]

    if require_verified:
        verifier = PacketWalker(width, height, levels, layers, xcb, ycb)
        try:
            walk = verifier.walk(packets)
        except Exception:
            return None
        if len(walk) < layers * (levels + 1) or walk[-1][3] + walk[-1][4] != len(packets):
            return None
    return header + packets + eoc


def decode_obfuscated_frame(payload: Any, expected_shape: tuple[int, int]):
    """Decode one obfuscated frame, preferring a verified reassembly.

    Returns ``(array, status)`` where status is ``verified``, ``unverified`` (the reassembly
    decoded but its packet walk did not check out, so the pixels are wrong) or ``failed``.
    """
    if Image is None:
        return None, "failed"
    for require in (True, False):
        rebuilt = reconstruct_obfuscated_jp2(payload, require_verified=require)
        if rebuilt is None:
            continue
        try:
            img = Image.open(io.BytesIO(rebuilt))
            img.load()
            array = np.asarray(img.convert("L"))
        except Exception:
            continue
        if array.shape == expected_shape:
            return array, ("verified" if require else "unverified")
    return None, "failed"


def _decode_zeiss_frame(
    payload: Any,
    expected_shape: tuple[int, int] | None = None,
    j2k_decode: bool = False,
) -> np.ndarray | None:
    """Decode one private-block payload into a 2-D array.

    Handles JPEG 2000 (via glymur, else PIL), any other codec PIL can read, and -
    when *expected_shape* is known - a raw uncompressed 8-bit buffer. Returns None
    when the payload is not decodable image data.
    """
    if not isinstance(payload, (bytes, bytearray)) or len(payload) < 16:
        return None
    payload = bytes(payload)
    is_jp2_file = len(payload) >= 12 and payload[4:8] == b"jP  "
    is_jp2_codestream = payload.startswith(b"\xff\x4f")

    if is_jp2_file or is_jp2_codestream:
        try:
            import glymur

            temp_file = tempfile.NamedTemporaryFile(suffix=".jp2", delete=False)
            try:
                temp_file.write(payload)
                temp_file.flush()
                temp_file.close()
                arr = glymur.Jp2k(temp_file.name)[:]
                if arr.ndim == 3 and Image is not None:
                    arr = np.asarray(Image.fromarray(arr).convert("L"))
                return np.asarray(arr)
            finally:
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass
        except Exception:
            pass

    if Image is not None:
        try:
            img = Image.open(io.BytesIO(payload))
            img.load()
            return np.asarray(img.convert("L"))
        except Exception:
            pass

    if expected_shape is not None and len(payload) == expected_shape[0] * expected_shape[1]:
        try:
            return np.frombuffer(payload, dtype=np.uint8).reshape(expected_shape)
        except Exception:
            return None

    if j2k_decode and Image is not None and is_obfuscated_payload(payload):
        rebuilt = reconstruct_obfuscated_jp2(payload)
        if rebuilt is not None:
            try:
                img = Image.open(io.BytesIO(rebuilt))
                img.load()
                return np.asarray(img.convert("L"))
            except Exception as e:
                log.warning(e)
                return None
    return None


def _extract_container_cube(
    ds: Any,
    element: int,
    kind: str,
    debug_mode: bool = False,
    aligner: "ZeissOctAligner | None" = None,
    j2k_decode: bool = False,
) -> ZeissOctCube | None:
    """Decode the private container ``(0407,<element>)`` of *ds* into a cube, or None when it is absent/undecodable."""
    container = ds.get((0x0407, element))
    if container is None or not getattr(container, "value", None):
        return None
    item = container.value[0]
    try:
        rows = int(item[_TAG_FRAME_ROWS].value)
        cols = int(item[_TAG_FRAME_COLS].value)
        frame_items = item[_TAG_FRAME_SEQUENCE].value
    except Exception:
        return None
    if rows <= 0 or cols <= 0 or not frame_items:
        return None
    declared = item.get(_TAG_FRAME_COUNT)
    declared = int(getattr(declared, "value", 0) or 0)
    if declared and declared != len(frame_items):
        # A short sequence yields a cube whose slice count disagrees with the
        # header the y-spacing was derived for.
        log.warning(
            f"[Zeiss cubes] container 0407,{element:04X} ({kind}) declares {declared} "
            f"frames but carries {len(frame_items)}"
        )

    frames: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for frame_item in frame_items:
        try:
            payload = frame_item[_TAG_FRAME_DATA].value
        except Exception:
            return None
        decoded = _decode_zeiss_frame(payload, expected_shape=(rows, cols))
        if decoded is None and j2k_decode and is_obfuscated_payload(payload):
            decoded, status = decode_obfuscated_frame(payload, (rows, cols))
            counts[status] = counts.get(status, 0) + 1
            if decoded is None:
                decoded = np.zeros((rows, cols), dtype=np.uint8)
        if decoded is None or decoded.shape != (rows, cols):
            if debug_mode:
                log.debug(
                    f"[Zeiss cubes] container 0407,{element:04X} ({kind}): frame "
                    f"{len(frames)} not decodable as {rows}x{cols}; skipping container"
                )
            return None
        frames.append(decoded)

    # CIRRUS stores the B-scans in reverse of the export order used by the raw
    # .img cubes; reversing here reproduces that slice order.
    volume = np.stack(frames[::-1], axis=2)

    scan_pattern = ds.get(_TAG_SCAN_PATTERN)
    scan_pattern = getattr(scan_pattern, "value", None) if scan_pattern is not None else None

    spacing = _container_spacing_mm(item)
    spacing_source = "dicom_private_tags"
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
        spacing_source = "unknown"
        log.warning(
            f"[Zeiss cubes] container 0407,{element:04X} ({kind}) carries no usable "
            f"spacing in (0407,100A/100B/100C); writing unit spacing"
        )

    offset_units = _fixation_offset_units(ds)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    origin_source = "zero"
    if aligner is not None:
        offset_x, offset_y, origin_source = aligner.scan_offset_mm(ds)
        affine[:3, 3] = _shared_frame_origin((offset_x, offset_y), volume, spacing)
    meta = {
        "method": "zeiss_private_container",
        "zeiss_container_tag": f"0407,{element:04X}",
        "zeiss_cube_kind": kind,
        "zeiss_scan_pattern": str(scan_pattern) if scan_pattern is not None else None,
        "zeiss_spacing_source": spacing_source,
        "zeiss_spacing_mm": list(spacing),
        "zeiss_fixation_offset_units": list(offset_units) if offset_units else None,
        "zeiss_origin_source": origin_source,
        "zeiss_origin_mm": [float(value) for value in affine[:3, 3]],
        "num_slices": int(volume.shape[2]),
        "chosen_shape": tuple(int(size) for size in volume.shape),
    }
    if counts:
        meta["zeiss_j2k_reconstruction"] = "experimental"
        meta["zeiss_j2k_frames_verified"] = counts.get("verified", 0)
        meta["zeiss_j2k_frames_unverified"] = counts.get("unverified", 0)
        meta["zeiss_j2k_frames_blank"] = counts.get("failed", 0)
        log.warning(
            f"[Zeiss cubes] {kind}: EXPERIMENTAL reconstruction - "
            f"{counts.get('verified', 0)} frames verified, "
            f"{counts.get('unverified', 0)} decoded but unverified (wrong pixels), "
            f"{counts.get('failed', 0)} blank. Not fit for analysis."
        )
    if debug_mode:
        log.debug(f"[Zeiss cubes] {kind}: shape={volume.shape} spacing={spacing} source={spacing_source}")
    return ZeissOctCube(kind=kind, array=volume, affine=affine, meta=meta)


def _extract_container_image(
    ds: Any,
    element: int,
    kind: str,
    quarter_turns: int,
    debug_mode: bool = False,
) -> ZeissOctImage | None:
    """Decode the single-frame container ``(0407,<element>)`` of *ds* into a 2-D image, or None when
    it is absent/undecodable."""
    container = ds.get((0x0407, element))
    if container is None or not getattr(container, "value", None):
        return None
    item = container.value[0]
    try:
        rows = int(item[_TAG_FRAME_ROWS].value)
        cols = int(item[_TAG_FRAME_COLS].value)
        frame_items = item[_TAG_FRAME_SEQUENCE].value
    except Exception:
        return None
    if rows <= 0 or cols <= 0 or not frame_items:
        return None

    try:
        payload = frame_items[0][_TAG_FRAME_DATA].value
    except Exception:
        return None
    decoded = _decode_zeiss_frame(payload, expected_shape=(rows, cols))
    if decoded is None or decoded.shape != (rows, cols):
        if debug_mode:
            log.debug(
                f"[Zeiss images] container 0407,{element:04X} ({kind}): "
                f"payload not decodable as {rows}x{cols}; skipping"
            )
        return None

    array = np.rot90(decoded, quarter_turns) if quarter_turns else decoded
    spacing = _container_spacing_mm(item)
    meta = {
        "zeiss_container_tag": f"0407,{element:04X}",
        "zeiss_image_kind": kind,
        "zeiss_spacing_mm": list(spacing) if spacing else None,
        "shape": tuple(int(size) for size in array.shape),
    }
    return ZeissOctImage(kind=kind, array=np.ascontiguousarray(array), meta=meta)


def extract_zeiss_oct_images(ds_list: list[Any], debug_mode: bool = False) -> list[ZeissOctImage]:
    """Extract the 2-D images (fundus, iris, en-face projection) stored alongside the cubes.

    Returns only the containers that are present and decodable, so an acquisition
    missing one - or carrying an unreadable payload - simply yields fewer images.
    """
    images: list[ZeissOctImage] = []
    for ds in ds_list:
        for element, (kind, quarter_turns) in ZEISS_OCT_IMAGE_CONTAINERS.items():
            image = _extract_container_image(ds, element, kind, quarter_turns, debug_mode=debug_mode)
            if image is not None:
                images.append(image)
    return images


# ---------------------------------------------------------------------------
# Cross-scan alignment
# ---------------------------------------------------------------------------

_LSLO_CONTAINER = 0x10A6

#: Lowest normalised cross-correlation accepted from an LSLO registration.
_LSLO_MIN_SCORE = 0.15


def _decode_lslo(ds: Any) -> tuple[np.ndarray, tuple[float, float]] | None:
    """The scan's LSLO fundus image and its ``(row, column)`` spacing in mm, or None."""
    container = ds.get((0x0407, _LSLO_CONTAINER))
    if container is None or not getattr(container, "value", None):
        return None
    item = container.value[0]
    try:
        rows = int(item[_TAG_FRAME_ROWS].value)
        cols = int(item[_TAG_FRAME_COLS].value)
        payload = item[_TAG_FRAME_SEQUENCE].value[0][_TAG_FRAME_DATA].value
        row_mm = float(item[_TAG_SPACING_X].value)
        col_mm = float(item[_TAG_SPACING_DEPTH].value)
    except Exception:
        return None
    if row_mm <= 0 or col_mm <= 0:
        return None
    image = _decode_zeiss_frame(payload, expected_shape=(rows, cols))
    if image is None or image.shape != (rows, cols):
        return None
    return image.astype(np.float32), (row_mm, col_mm)


def _vessel_contrast(image: np.ndarray) -> np.ndarray:
    """High-pass the fundus image so registration follows vessels, not illumination."""
    return image - gaussian_filter(image, 6)


def _ncc(fixed: np.ndarray, moving: np.ndarray, shift: tuple[int, int], min_overlap: tuple[int, int]) -> float | None:
    """Normalised cross-correlation over the overlap when *moving* is displaced by *shift*."""
    d_row, d_col = shift
    height, width = fixed.shape
    r0, r1 = max(0, d_row), min(height, height + d_row)
    c0, c1 = max(0, d_col), min(width, width + d_col)
    if r1 - r0 < min_overlap[0] or c1 - c0 < min_overlap[1]:
        return None
    a = fixed[r0:r1, c0:c1]
    b = moving[r0 - d_row : r1 - d_row, c0 - d_col : c1 - d_col]
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / denominator) if denominator else None


def _register_fundus(fixed: np.ndarray, moving: np.ndarray) -> tuple[int, int, float] | None:
    """Pixel shift ``(row, column)`` aligning *moving* onto *fixed*, with its score.

    Searched coarsely on a 4x decimation and then refined at full resolution, which keeps a
    +/-6 mm search affordable: the fixation jump between a macular and a disc-centred scan is
    around 4.5 mm.
    """
    if fixed.shape != moving.shape:
        return None
    fine_fixed = _vessel_contrast(fixed)
    fine_moving = _vessel_contrast(moving)
    coarse_fixed = gaussian_filter(fine_fixed, 2)[::4, ::4]
    coarse_moving = gaussian_filter(fine_moving, 2)[::4, ::4]

    best = None
    for d_row in range(-75, 76):
        for d_col in range(-30, 31):
            score = _ncc(coarse_fixed, coarse_moving, (d_row, d_col), (40, 75))
            if score is not None and (best is None or score > best[2]):
                best = (d_row, d_col, score)
    if best is None:
        return None

    centre_row, centre_col = best[0] * 4, best[1] * 4
    refined = None
    for d_row in range(centre_row - 5, centre_row + 6):
        for d_col in range(centre_col - 5, centre_col + 6):
            score = _ncc(fine_fixed, fine_moving, (d_row, d_col), (160, 300))
            if score is not None and (refined is None or score > refined[2]):
                refined = (d_row, d_col, score)
    return refined


class ZeissOctAligner:
    """Places the cubes of one study in a per-eye world frame shared across scan sizes.

    Each scan's in-plane offset is measured by registering its LSLO fundus image against the
    eye's reference scan; the fundus images of one session overlap across almost their whole
    field, which makes this far more reliable than registering the cubes themselves (a 6x6
    and a disc-centred 4.5x4.5 share only a ~0.8 mm band).

    The fixation-offset tags are the fallback when a fundus image is missing or the
    registration is too weak to trust. They are nominal - a protocol setting rather than a
    measurement - so they cannot capture eye movement between acquisitions, which on the
    reference study reached 0.16 mm even between two macula-centred scans.
    """

    def __init__(self, debug_mode: bool = False) -> None:
        self._references: dict[str, tuple[np.ndarray, tuple[float, float], tuple[float, float]]] = {}
        self._offsets: dict[str, tuple[float, float, str]] = {}
        self._debug_mode = debug_mode

    @staticmethod
    def _laterality(ds: Any) -> str:
        """Eye label used to keep each eye's scans in their own frame."""
        return str(getattr(ds, "Laterality", "") or "U").strip().upper()

    @staticmethod
    def _nominal_offset_mm(ds: Any) -> tuple[float, float]:
        """Scan-centre offset from the fovea implied by the fixation tags."""
        units = _fixation_offset_units(ds)
        if units is None:
            return (0.0, 0.0)
        return (-units[0] * ZEISS_FIXATION_MM_PER_UNIT, -units[1] * ZEISS_FIXATION_MM_PER_UNIT)

    def add_reference_candidates(self, ds_list: list[Any]) -> None:
        """Choose the scan each eye's other scans are registered against.

        A macula-centred scan is preferred, so the frame is anchored on the fovea; any scan
        with a usable fundus image will do otherwise, with its own nominal offset folded in
        to keep the frame fovea-referenced either way.
        """
        for ds in ds_list:
            if not has_zeiss_oct_cubes(ds):
                continue
            laterality = self._laterality(ds)
            centred = _fixation_offset_units(ds) == (0.0, 0.0)
            if laterality in self._references and not (centred and self._references[laterality][2] != (0.0, 0.0)):
                continue
            decoded = _decode_lslo(ds)
            if decoded is None:
                continue
            image, spacing = decoded
            self._references[laterality] = (image, spacing, self._nominal_offset_mm(ds))
            if self._debug_mode:
                log.debug(f"[Zeiss align] eye {laterality}: reference is {getattr(ds, 'SeriesDescription', '?')}")

    def scan_offset_mm(self, ds: Any) -> tuple[float, float, str]:
        """In-plane offset ``(x, y)`` in mm of this scan's centre from the fovea, and its source."""
        key = str(getattr(ds, "SOPInstanceUID", "") or id(ds))
        cached = self._offsets.get(key)
        if cached is not None:
            return cached
        result = self._measure_offset_mm(ds)
        self._offsets[key] = result
        return result

    def _measure_offset_mm(self, ds: Any) -> tuple[float, float, str]:
        """Register against the eye's reference, falling back to the nominal tag offset."""
        nominal = self._nominal_offset_mm(ds)
        reference = self._references.get(self._laterality(ds))
        decoded = _decode_lslo(ds) if reference is not None else None
        if reference is not None and decoded is not None:
            ref_image, (row_mm, col_mm), ref_nominal = reference
            registered = _register_fundus(ref_image, decoded[0])
            if registered is not None and registered[2] >= _LSLO_MIN_SCORE:
                d_row, d_col, score = registered
                # Validated against the reference study: a fundus row step moves the cube's
                # A-scan axis in the opposite direction, a column step moves the B-scan axis
                # in the same direction.
                offset = (
                    -d_row * row_mm + ref_nominal[0],
                    d_col * col_mm + ref_nominal[1],
                )
                if self._debug_mode:
                    log.debug(
                        f"[Zeiss align] {getattr(ds, 'SeriesDescription', '?')}: "
                        f"measured ({offset[0]:+.3f}, {offset[1]:+.3f}) mm, ncc={score:.3f}"
                    )
                return (offset[0], offset[1], "lslo_registration")
            if self._debug_mode:
                score = registered[2] if registered else float("nan")
                log.debug(
                    f"[Zeiss align] {getattr(ds, 'SeriesDescription', '?')}: registration too weak "
                    f"(ncc={score:.3f}); using the fixation tags"
                )
        return (nominal[0], nominal[1], "fixation_tags")


def has_zeiss_oct_cubes(ds: Any) -> bool:
    """True when *ds* carries at least one private OCT cube container."""
    return any((0x0407, element) in ds for element in ZEISS_OCT_CUBE_CONTAINERS)


def extract_zeiss_oct_cubes(
    ds_list: list[Any],
    debug_mode: bool = False,
    aligner: "ZeissOctAligner | None" = None,
    j2k_decode: bool = False,
) -> list[ZeissOctCube]:
    """Extract every volumetric cube stored in the private containers of *ds_list*.

    An angiography acquisition yields two cubes - the structural ``cube_z`` and the
    flow ``FlowCube_z`` - while a plain cube scan yields only ``cube_z``. Returns an
    empty list when no container is decodable, leaving the caller to fall back to
    :func:`extract_zeiss_raw_oct`.
    """
    cubes: list[ZeissOctCube] = []
    for ds in ds_list:
        for element, kind in ZEISS_OCT_CUBE_CONTAINERS.items():
            cube = _extract_container_cube(
                ds, element, kind, debug_mode=debug_mode, aligner=aligner, j2k_decode=j2k_decode
            )
            if cube is not None:
                cubes.append(cube)
    return cubes


def _is_zeiss_raw_storage(ds: Any) -> bool:
    """Return True if dataset appears to be a Zeiss CIRRUS Raw Data Storage OCT."""
    manu = getattr(ds, "Manufacturer", "") or ""
    sop = getattr(ds, "SOPClassUID", "")
    sop_s = str(sop) if sop is not None else ""
    if "Zeiss" in manu or "Carl Zeiss Meditec" in manu or "CIRRUS" in str(ds.get("ManufacturerModelName", "")):
        return True
    if sop_s == RAW_DATA_SOP_UID or "Raw Data Storage" in sop_s:
        return True
    for tag in ds:
        if getattr(tag, "keyword", None) is None and getattr(tag.tag, "group", 0) >= 0x0400 and "99CZM" in str(tag.value):
            return True
    return False


def _extract_zeiss_raw_oct(
    ds_list: list[Any],
    md: dict[str, Any],
    debug_mode: bool = False,
):
    """
    Robust Zeiss CIRRUS extractor with tile-assembly and output shaped to (X, Y, Z)
    so that nib.save(Nifti1Image(volume, np.eye(4))) mimics dicom2nifti's axis ordering
    (Columns, Rows, Slices).
    """
    from collections import Counter, defaultdict

    def _find_tag_value_anywhere_local(ds: Any, group: int, element: int):
        """Search a dataset (including nested sequences) for a tag's value by group/element."""
        if Tag is None:
            return None
        wanted = Tag(group, element)
        for elem in ds.iterall():
            if elem.tag == wanted:
                return elem.value
        return None

    def _collect_private_blocks_local(ds: Any):
        """Collect every non-empty private OB/OW/OF byte-string element from a dataset (candidate raw-image blocks)."""
        out: list[tuple[Any, str, int, bytes]] = []
        for elem in ds.iterall():
            try:
                if elem.tag.group >= 0x0400 and elem.VR in ("OB", "OW", "OF"):
                    value = getattr(elem, "value", None)
                    if isinstance(value, (bytes, bytearray)) and len(value) > 0:
                        out.append((elem.tag, elem.VR, len(value), bytes(value)))
            except Exception:
                continue
        return out

    first = ds_list[0]
    candidate_width = None
    candidate_height = None
    try:
        if hasattr(first, "Columns") and int(getattr(first, "Columns", 0)) > 0:
            candidate_width = int(first.Columns)
        if hasattr(first, "Rows") and int(getattr(first, "Rows", 0)) > 0:
            candidate_height = int(first.Rows)
    except Exception:
        candidate_width = candidate_height = None

    geometry_candidates = [
        (0x040D, 0x1017),
        (0x040D, 0x1018),
        (0x0407, 0x1001),
        (0x0407, 0x1002),
        (0x0409, 0x10DB),
        (0x0409, 0x10DA),
        (0x0409, 0x1017),
        (0x0409, 0x1018),
    ]
    for group, element in geometry_candidates:
        if not candidate_width:
            value = _find_tag_value_anywhere_local(first, group, element)
            if value is not None:
                try:
                    candidate_width = int(value)
                except Exception:
                    pass
        if not candidate_height:
            value = _find_tag_value_anywhere_local(first, group, element + 1)
            if value is not None:
                try:
                    candidate_height = int(value)
                except Exception:
                    pass

    if not (candidate_width and candidate_height):
        pairs = [
            ((0x0407, 0x1001), (0x0407, 0x1002)),
            ((0x040D, 0x1017), (0x040D, 0x1018)),
            ((0x0409, 0x10DB), (0x0409, 0x10DA)),
        ]
        for (g1, e1), (g2, e2) in pairs:
            v1 = _find_tag_value_anywhere_local(first, g1, e1)
            v2 = _find_tag_value_anywhere_local(first, g2, e2)
            if v1 is not None and v2 is not None:
                try:
                    candidate_width = candidate_width or int(v1)
                    candidate_height = candidate_height or int(v2)
                    break
                except Exception:
                    pass

    if not (candidate_width and candidate_height):
        for ds in ds_list:
            for group, element in geometry_candidates:
                value = _find_tag_value_anywhere_local(ds, group, element)
                if value is None:
                    continue
                try:
                    value_i = int(value)
                except Exception:
                    continue
                if not candidate_width and 16 < value_i < 10000:
                    candidate_width = value_i
                elif not candidate_height and 16 < value_i < 10000:
                    candidate_height = value_i
            if candidate_width and candidate_height:
                break

    try:
        if not candidate_width:
            cw = md.get("Columns") or md.get("(0028,0011)") or md.get("ImageColumns")
            if cw:
                candidate_width = int(cw)
    except Exception:
        candidate_width = None
    try:
        if not candidate_height:
            ch = md.get("Rows") or md.get("(0028,0010)") or md.get("ImageRows")
            if ch:
                candidate_height = int(ch)
    except Exception:
        candidate_height = None

    if debug_mode:
        log.debug(
            "[Zeiss extractor] geometry candidates: "
            f"width={candidate_width}, height={candidate_height}"
        )

    all_blocks: list[tuple[int, Any, str, int, bytes]] = []
    for idx, ds in enumerate(ds_list):
        blocks = _collect_private_blocks_local(ds)
        for tag, vr, length, value in blocks:
            all_blocks.append((idx, tag, vr, length, value))

    if not all_blocks:
        raise RuntimeError("No private OB/OW blocks found across series (all_blocks empty).")

    length_map: dict[int, int] = {}
    for _, _, _, length, _ in all_blocks:
        length_map.setdefault(length, 0)
        length_map[length] += 1

    if debug_mode:
        log.debug(
            "[Zeiss extractor] found "
            f"{len(all_blocks)} private blocks; lengths summary: "
            f"{sorted(length_map.items(), key=lambda item: -item[1])[:10]}"
        )

    def try_decode_image_block(payload: bytes):
        """Attempt to decode a private-block payload as a raw Zeiss OCT image (shape guessed by the decoder)."""
        return _decode_zeiss_frame(payload)

    blocks_by_instance = defaultdict(list)
    for idx, tag, vr, length, value in all_blocks:
        blocks_by_instance[idx].append((tag, vr, length, value))

    assembled_slices: list[tuple[int, np.ndarray]] = []
    for inst_idx in sorted(blocks_by_instance.keys()):
        tiles = blocks_by_instance[inst_idx]
        tiles_by_elem = defaultdict(list)
        for tag, vr, length, value in tiles:
            tiles_by_elem[tag.element].append((tag, length, value))

        decoded_tiles: list[tuple[int, np.ndarray]] = []
        for elem in sorted(tiles_by_elem.keys()):
            candidates = tiles_by_elem[elem]
            decoded = None
            for _, _, value in candidates:
                dec = try_decode_image_block(value)
                if dec is not None:
                    decoded = dec
                    break
            if decoded is not None:
                decoded_tiles.append((elem, decoded))
        if not decoded_tiles:
            continue

        shapes = [decoded.shape for _, decoded in decoded_tiles]
        heights = [shape[0] for shape in shapes]
        widths = [shape[1] for shape in shapes]
        horizontal_ok = len(set(heights)) == 1
        vertical_ok = len(set(widths)) == 1
        assembled = None
        if horizontal_ok:
            total_w = sum(widths)
            h0 = heights[0]
            if candidate_width and candidate_height:
                if abs(total_w - candidate_width) <= max(1, int(candidate_width * 0.05)) and abs(
                    h0 - candidate_height
                ) <= max(1, int(candidate_height * 0.05)):
                    decoded_tiles.sort(key=lambda item: item[0])
                    arrs = [decoded for _, decoded in decoded_tiles]
                    try:
                        assembled = np.concatenate(arrs, axis=1)
                    except Exception:
                        assembled = None
            else:
                if total_w > 16 and h0 > 16:
                    decoded_tiles.sort(key=lambda item: item[0])
                    try:
                        assembled = np.concatenate([decoded for _, decoded in decoded_tiles], axis=1)
                    except Exception:
                        assembled = None

        if assembled is None and vertical_ok:
            total_h = sum(heights)
            w0 = widths[0]
            if candidate_width and candidate_height:
                if abs(total_h - candidate_height) <= max(1, int(candidate_height * 0.05)) and abs(
                    w0 - candidate_width
                ) <= max(1, int(candidate_width * 0.05)):
                    decoded_tiles.sort(key=lambda item: item[0])
                    try:
                        assembled = np.concatenate([decoded for _, decoded in decoded_tiles], axis=0)
                    except Exception:
                        assembled = None
            else:
                if total_h > 16 and w0 > 16:
                    decoded_tiles.sort(key=lambda item: item[0])
                    try:
                        assembled = np.concatenate([decoded for _, decoded in decoded_tiles], axis=0)
                    except Exception:
                        assembled = None

        if assembled is not None:
            h, w = assembled.shape
            th = candidate_height or h
            tw = candidate_width or w

            def normalize(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
                """Center-crop/pad a 2-D array to exactly ``(target_h, target_w)``."""
                ah, aw = arr.shape
                if ah > target_h:
                    sh = (ah - target_h) // 2
                    arr = arr[sh : sh + target_h, :]
                if aw > target_w:
                    sw = (aw - target_w) // 2
                    arr = arr[:, sw : sw + target_w]
                pad_h = target_h - arr.shape[0]
                pad_w = target_w - arr.shape[1]
                if pad_h > 0 or pad_w > 0:
                    pt = pad_h // 2
                    pb = pad_h - pt
                    pl = pad_w // 2
                    pr = pad_w - pl
                    arr = np.pad(arr, ((pt, pb), (pl, pr)), mode="constant", constant_values=0)
                return arr

            assembled = normalize(assembled, th, tw)
            assembled_slices.append((inst_idx, assembled))

    if assembled_slices:
        assembled_slices.sort(key=lambda item: item[0])
        vol_z_h_w = np.stack([slc for _, slc in assembled_slices], axis=0)
        vol = vol_z_h_w.transpose(2, 1, 0)
        affine = np.eye(4)
        meta = {
            "method": "tile_assembled",
            "num_slices": int(vol.shape[2]),
            "chosen_shape": (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])),
        }
        return vol, affine, meta

    def _factor_pairs(n: int):
        """Enumerate integer ``(rows, cols)`` factor pairs of *n*, for tile-grid guessing."""
        pairs = []
        for rows in range(1, int(np.sqrt(n)) + 1):
            if n % rows == 0:
                cols = n // rows
                pairs.append((rows, cols))
                if rows != cols:
                    pairs.append((cols, rows))
        return pairs

    grid_slices: list[tuple[int, np.ndarray]] = []
    for inst_idx in sorted(blocks_by_instance.keys()):
        tiles = blocks_by_instance[inst_idx]
        tiles_by_elem = defaultdict(list)
        for tag, vr, length, value in tiles:
            tiles_by_elem[tag.element].append((tag, length, value))
        decoded_tiles: list[tuple[int, np.ndarray]] = []
        for elem in sorted(tiles_by_elem.keys()):
            candidates = tiles_by_elem[elem]
            decoded = None
            for _, _, value in candidates:
                dec = try_decode_image_block(value)
                if dec is not None:
                    if dec.ndim == 3:
                        dec = dec[..., 0]
                    decoded = dec
                    break
            if decoded is not None:
                decoded_tiles.append((elem, decoded))
        if len(decoded_tiles) < 2:
            continue

        shapes = [decoded.shape for _, decoded in decoded_tiles]
        if not shapes:
            continue
        shape_counts = Counter(shapes)
        tile_h, tile_w = max(shape_counts.items(), key=lambda item: item[1])[0]
        selected = [(elem, decoded) for elem, decoded in decoded_tiles if decoded.shape == (tile_h, tile_w)]
        n_tiles = len(selected)
        if n_tiles < 2:
            continue

        target_h = candidate_height or (tile_h * n_tiles)
        target_w = candidate_width or tile_w
        best = None
        for rows, cols in _factor_pairs(n_tiles):
            height = rows * tile_h
            width = cols * tile_w
            if candidate_height and candidate_width:
                if abs(height - candidate_height) <= max(1, int(0.10 * candidate_height)) and abs(
                    width - candidate_width
                ) <= max(1, int(0.10 * candidate_width)):
                    score = abs(height - candidate_height) + abs(width - candidate_width)
                    if best is None or score < best[0]:
                        best = (score, rows, cols)
            else:
                score = abs(height - target_h) + abs(width - target_w)
                if best is None or score < best[0]:
                    best = (score, rows, cols)
        if best is None:
            continue

        _, rows, cols = best
        selected.sort(key=lambda item: item[0])
        try:
            grid = []
            idx = 0
            for _ in range(rows):
                row_tiles = [selected[idx + cj][1] for cj in range(cols)]
                idx += cols
                row_cat = np.concatenate(row_tiles, axis=1)
                grid.append(row_cat)
            assembled = np.concatenate(grid, axis=0)
            th = candidate_height or assembled.shape[0]
            tw = candidate_width or assembled.shape[1]

            def normalize(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
                """Center-crop/pad a 2-D array to exactly ``(target_h, target_w)``."""
                ah, aw = arr.shape
                if ah > target_h:
                    sh = (ah - target_h) // 2
                    arr = arr[sh : sh + target_h, :]
                if aw > target_w:
                    sw = (aw - target_w) // 2
                    arr = arr[:, sw : sw + target_w]
                pad_h = target_h - arr.shape[0]
                pad_w = target_w - arr.shape[1]
                if pad_h > 0 or pad_w > 0:
                    pt = pad_h // 2
                    pb = pad_h - pt
                    pl = pad_w // 2
                    pr = pad_w - pl
                    arr = np.pad(arr, ((pt, pb), (pl, pr)), mode="constant", constant_values=0)
                return arr

            assembled = normalize(assembled, th, tw)
            grid_slices.append((inst_idx, assembled))
        except Exception:
            continue

    if grid_slices:
        grid_slices.sort(key=lambda item: item[0])
        vol_z_h_w = np.stack([slc for _, slc in grid_slices], axis=0)
        vol = vol_z_h_w.transpose(2, 1, 0)
        affine = np.eye(4)
        meta = {
            "method": "tile_grid_assembled",
            "num_slices": int(vol.shape[2]),
            "chosen_shape": (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])),
        }
        return vol, affine, meta

    decoded = []
    for idx, tag, vr, length, value in sorted(all_blocks, key=lambda item: (item[0], item[1].group, item[1].element)):
        dec = try_decode_image_block(value)
        if dec is None:
            continue
        if dec.ndim == 3:
            dec = dec[..., 0]
        decoded.append((idx, tag, dec))

    if decoded:
        groups = defaultdict(list)
        for idx, tag, arr in decoded:
            groups[arr.shape].append((idx, tag, arr))
        chosen_shape, chosen_list = max(groups.items(), key=lambda item: len(item[1]))
        if debug_mode:
            log.debug(
                "[Zeiss extractor] decoded shapes counts: "
                f"{[(shape, len(items)) for shape, items in groups.items()]}; chosen={chosen_shape}"
            )
        if candidate_height and candidate_width:
            cand_shape = (candidate_height, candidate_width)
            swapped = (candidate_width, candidate_height)
            if cand_shape in groups and len(groups[cand_shape]) >= max(2, len(chosen_list) // 2):
                chosen_shape = cand_shape
                chosen_list = groups[cand_shape]
            elif swapped in groups and len(groups[swapped]) >= max(2, len(chosen_list) // 2):
                chosen_shape = swapped
                chosen_list = groups[swapped]

        h_target, w_target = chosen_shape
        max_val = 0
        for _, _, arr in chosen_list:
            try:
                max_val = max(max_val, int(arr.max()))
            except Exception:
                pass
        target_dtype = np.uint8 if max_val <= 255 else np.uint16

        def normalize_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
            """Center-crop/pad a 2-D array to ``(target_h, target_w)`` and cast to the chosen output dtype."""
            ah, aw = arr.shape
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)
            if ah > target_h:
                sh = (ah - target_h) // 2
                arr = arr[sh : sh + target_h, :]
            if aw > target_w:
                sw = (aw - target_w) // 2
                arr = arr[:, sw : sw + target_w]
            pad_h = target_h - arr.shape[0]
            pad_w = target_w - arr.shape[1]
            if pad_h > 0 or pad_w > 0:
                pt = pad_h // 2
                pb = pad_h - pt
                pl = pad_w // 2
                pr = pad_w - pl
                arr = np.pad(arr, ((pt, pb), (pl, pr)), mode="constant", constant_values=0)
            return arr

        chosen_list.sort(key=lambda item: (item[0], item[1].group, item[1].element))
        arrays = []
        for _, _, arr in chosen_list:
            try:
                arrays.append(normalize_to(arr, h_target, w_target))
            except Exception:
                continue

        if arrays:
            vol_z_h_w = np.stack(arrays, axis=0)
            vol = vol_z_h_w.transpose(2, 1, 0)
            affine = np.eye(4)
            meta = {
                "method": "per-block-decode-majority",
                "chosen_shape": (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])),
                "num_slices": int(vol.shape[2]),
            }
            return vol, affine, meta

    if candidate_width and candidate_height:
        width = candidate_width
        height = candidate_height
        expected1 = width * height
        expected2 = width * height * 2
        usable = []
        bytes_per_pixel = None
        for idx, tag, vr, length, value in all_blocks:
            if not isinstance(value, (bytes, bytearray)):
                continue
            if length == expected1:
                bytes_per_pixel = 1
                usable.append((idx, tag, value, 1))
            elif length == expected2:
                bytes_per_pixel = 2
                usable.append((idx, tag, value, 2))
        if usable:
            usable.sort(key=lambda item: (item[0], item[1].group, item[1].element))
            slices = []
            for idx, tag, bval, bp in usable:
                try:
                    if bp == 1:
                        arr = np.frombuffer(bval, dtype=np.uint8).reshape((height, width))
                    else:
                        arr = np.frombuffer(bval, dtype=np.uint16).reshape((height, width))
                    slices.append(arr)
                except Exception:
                    continue
            if slices:
                vol_z_h_w = np.stack(slices, axis=0)
                vol = vol_z_h_w.transpose(2, 1, 0)
                return vol, np.eye(4), {
                    "method": "per-block-slices",
                    "bytes_per_pixel": bytes_per_pixel,
                    "num_slices": int(vol.shape[2]),
                }

    merged_by_instance = defaultdict(list)
    for idx, tag, vr, length, value in all_blocks:
        if isinstance(value, (bytes, bytearray)):
            merged_by_instance[idx].append((tag, value))

    merged_stream = b""
    for idx in sorted(merged_by_instance.keys()):
        items = sorted(merged_by_instance[idx], key=lambda item: (item[0].group, item[0].element))
        for _, value in items:
            merged_stream += value

    if candidate_width and candidate_height and len(merged_stream) > 0:
        per_slice1 = candidate_width * candidate_height
        if len(merged_stream) % per_slice1 == 0:
            n_slices = len(merged_stream) // per_slice1
            try:
                arr = np.frombuffer(merged_stream, dtype=np.uint8).reshape((n_slices, candidate_height, candidate_width))
                vol = arr.transpose(2, 1, 0)
                return vol, np.eye(4), {"method": "merged-concat-uint8", "num_slices": int(n_slices)}
            except Exception:
                pass
        if len(merged_stream) % (per_slice1 * 2) == 0:
            n_slices = len(merged_stream) // (per_slice1 * 2)
            try:
                arr = np.frombuffer(merged_stream, dtype=np.uint16).reshape(
                    (n_slices, candidate_height, candidate_width)
                )
                vol = arr.transpose(2, 1, 0)
                return vol, np.eye(4), {"method": "merged-concat-uint16", "num_slices": int(n_slices)}
            except Exception:
                pass

    if candidate_width and candidate_height:
        target = candidate_width * candidate_height
        fuzz = []
        for idx, tag, vr, length, value in all_blocks:
            if not isinstance(value, (bytes, bytearray)):
                continue
            if abs(length - target) <= max(1, int(target * 0.10)):
                try:
                    arr = np.frombuffer(value, dtype=np.uint8)
                    if arr.size >= target:
                        arr = arr[:target].reshape((candidate_height, candidate_width))
                        fuzz.append((idx, tag, arr))
                except Exception:
                    pass
        if fuzz:
            fuzz.sort(key=lambda item: (item[0], item[1].group, item[1].element))
            vol_z_h_w = np.stack([item[2] for item in fuzz], axis=0)
            vol = vol_z_h_w.transpose(2, 1, 0)
            return vol, np.eye(4), {"method": "fuzzy-slices", "num_slices": int(vol.shape[2])}

    debug_report = {
        "candidate_width": candidate_width,
        "candidate_height": candidate_height,
        "num_blocks": len(all_blocks),
        "length_map": length_map,
        "sample_blocks": [(int(block[1].group), hex(block[1].element), block[2]) for block in all_blocks[:16]],
    }
    raise RuntimeError(
        "Could not reconstruct volume from Zeiss private tags using heuristics. "
        f"Debug: {debug_report}"
    )


# ---------------------------------------------------------------------------
# Derived analysis series
# ---------------------------------------------------------------------------
#
# Analysis objects (thickness maps, report analyses) carry no cube container. They
# describe a single en-face grid at the top level of the dataset, reusing the same
# private tags the cubes use inside their containers - but with a different meaning:
# here (0407,100A) and (0407,100B) are the two IN-PLANE spacings of the map, not the
# lateral and axial spacings of a B-scan.
#: ReferencedInstanceSequence, with ReferencedImageSequence as the older alternative.
_TAG_REFERENCED_SEQUENCES = ((0x0008, 0x114A), (0x0008, 0x1140))
_TAG_REFERENCED_SOP_INSTANCE = (0x0008, 0x1155)
_TAG_PURPOSE_OF_REFERENCE = (0x0040, 0xA170)
_TAG_CODE_MEANING = (0x0008, 0x0104)


def derived_analysis_geometry(ds: Any) -> tuple[int, int, float, float] | None:
    """En-face grid ``(nx, ny, spacing_x, spacing_y)`` of a derived analysis series, or None.

    Returns None for cube objects, whose geometry lives inside their containers and means
    something else.
    """
    if has_zeiss_oct_cubes(ds):
        return None
    try:
        nx = int(ds[_TAG_FRAME_ROWS].value)
        ny = int(ds[_TAG_FRAME_COLS].value)
        spacing_x = float(ds[_TAG_SPACING_X].value)
        spacing_y = float(ds[_TAG_SPACING_DEPTH].value)
    except Exception:
        return None
    if nx <= 0 or ny <= 0 or spacing_x <= 0 or spacing_y <= 0:
        return None
    return (nx, ny, spacing_x, spacing_y)


def has_obfuscated_cube_payload(ds: Any) -> bool:
    """True when *ds* holds a cube container whose frames are stored obfuscated."""
    for element in ZEISS_OCT_CUBE_CONTAINERS:
        container = ds.get((0x0407, element))
        if container is None or not getattr(container, "value", None):
            continue
        try:
            payload = container.value[0][_TAG_FRAME_SEQUENCE].value[0][_TAG_FRAME_DATA].value
        except Exception:
            continue
        if is_obfuscated_payload(payload):
            return True
    return False


def zeiss_referenced_sources(ds: Any) -> list[dict[str, str]]:
    """Acquisitions this derived object was computed from, as ``{"sop_instance_uid", "scan_type"}``.

    Empty when the object records no reference - which is the case for whole studies, so it
    must not be relied on to identify the parent scan.
    """
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for tag in _TAG_REFERENCED_SEQUENCES:
        for item in getattr(ds.get(tag), "value", None) or []:
            try:
                uid = str(item[_TAG_REFERENCED_SOP_INSTANCE].value)
            except Exception:
                continue
            if uid in seen:
                continue
            seen.add(uid)
            scan_type = ""
            try:
                scan_type = str(item[_TAG_PURPOSE_OF_REFERENCE].value[0][_TAG_CODE_MEANING].value)
            except Exception:
                pass
            out.append({"sop_instance_uid": uid, "scan_type": scan_type})
    return out


def derived_analysis_affine(ds: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    """Voxel-to-world affine for a derived analysis volume, or None when its grid does not match.

    The volume is ``(x, y, map)``: two spatial axes over the parent scan's en-face grid, and a
    stack of analysis maps that carries no physical spacing.
    """
    geometry = derived_analysis_geometry(ds)
    if geometry is None or len(shape) < 2:
        return None
    nx, ny, spacing_x, spacing_y = geometry
    if (int(shape[0]), int(shape[1])) != (nx, ny):
        log.debug(
            f"[Zeiss derived] grid {nx}x{ny} does not match volume {shape[:2]}; leaving unit spacing"
        )
        return None
    return np.diag([spacing_x, spacing_y, 1.0, 1.0])


def is_zeiss_raw_storage(ds: Any) -> bool:
    """Public alias of :func:`_is_zeiss_raw_storage` — True when *ds* is a Zeiss private raw-OCT storage object."""
    return _is_zeiss_raw_storage(ds)


def extract_zeiss_raw_oct(
    ds_list: list[Any],
    md: dict[str, Any],
    debug_mode: bool = False,
):
    """Public alias of :func:`_extract_zeiss_raw_oct` — decode a Zeiss raw-OCT series into a volume."""
    return _extract_zeiss_raw_oct(ds_list, md, debug_mode=debug_mode)
