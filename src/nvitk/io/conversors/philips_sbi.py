"""Philips Spectral Base Images (SBI): read the spectral data of a Philips spectral CT.

A Philips Spectral CT (IQon / Spectral CT 7500) stores, next to each reconstruction, a
series described ``SBI(V5.1)_CB_0.67_C|IMR, 78%``: Secondary Captures with
``ImageType = DERIVED\\SECONDARY\\SBI\\SBI_CSPN``, one instance per slice of the
reconstruction. Their "pixels" are not an image but a container, read here as follows
(reverse-engineered from SBI V5.1 data; not a Philips specification):

* a header of little-endian int32 words: the header length in words (45), the format
  version (51 for 5.1), the number of components, then per component its byte length,
  its id, two reserved words (-1, 0), and its rescale slope and intercept (float32);
* the components' byte streams, one after another. Each is a square image (the
  reconstruction's matrix, 512 x 512) compressed with Philips' ``PMSCT_RLE1`` scheme, the
  one GDCM decodes for Brilliance CT: runs ``A5 <count-1> <byte>`` are expanded first, then
  every byte is a signed step from the previous pixel, except ``5A <lo> <hi>``, a new
  absolute 16-bit value;
* the components, in the order the series names them in ``COMB_SCATT_PE_N_B_U_C`` (private
  ``(01F7,10CC)``): ``4`` the conventional image (HU), ``11`` the Compton-scatter and ``12``
  the photoelectric part of the attenuation, ``13`` a noise estimate, ``26``/``27``/``28``
  binary masks the workstation uses.

The scatter and photoelectric components add up to the conventional image (each is the
part of the attenuation, in HU + 1000, that one process contributes), so they are the
two-material basis every spectral result is derived from; see
:mod:`nvitk.transform.spectral`.

Example::

    from nvitk.io.conversors.philips_sbi import read_sbi_series

    sbi = read_sbi_series(paths)           # the files of one SBI series
    print(sbi.metadata["sbi_recon"], sbi.photo.shape)
    conventional = sbi.image("conventional")

Decoding is exact and runs on the CPU (byte streams); the volumes come back as host
arrays, like any reader's.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy

#: Component ids -> names (the series' ``COMB_SCATT_PE_N_B_U_C``).
COMPONENTS: dict[int, str] = {
    4: "conventional",
    11: "scatter",
    12: "photo",
    13: "noise",
    26: "mask_b",
    27: "mask_u",
    28: "mask_c",
}
#: Components read by default: what the spectral results need.
DEFAULT_COMPONENTS = ("conventional", "photo", "scatter")

_RUN, _ABSOLUTE = 0xA5, 0x5A


# One stream


def _markers(candidates: numpy.ndarray) -> numpy.ndarray:
    """The candidate marker positions that really are markers: one inside the two
    operand bytes of an earlier marker is data, not a marker."""
    if candidates.size < 2:
        return candidates
    keep = numpy.ones(candidates.size, dtype=bool)
    near = numpy.flatnonzero(numpy.diff(candidates) <= 2) + 1
    # Only candidates within two bytes of another need looking at, in order.
    for k in near:
        j = k - 1
        while j >= 0 and candidates[k] - candidates[j] <= 2:
            if keep[j]:
                keep[k] = False
                break
            j -= 1
    return candidates[keep]


def decode_pmsct_rle(stream: bytes | numpy.ndarray) -> numpy.ndarray:
    """Decode one Philips ``PMSCT_RLE1`` stream into its 16-bit pixel values (flat)."""
    raw = numpy.frombuffer(stream, dtype=numpy.uint8) if isinstance(stream, (bytes, bytearray, memoryview)) \
        else numpy.asarray(stream, dtype=numpy.uint8)
    # Runs: A5 <count-1> <value>.
    runs = _markers(numpy.flatnonzero(raw == _RUN))
    runs = runs[runs + 2 < raw.size]
    repeat = numpy.ones(raw.size, dtype=numpy.int64)
    repeat[runs] = 0
    repeat[runs + 1] = 0
    repeat[runs + 2] = raw[runs + 1].astype(numpy.int64) + 1
    expanded = numpy.repeat(raw, repeat)
    # Steps, with 5A <lo> <hi> setting an absolute value.
    absolute = _markers(numpy.flatnonzero(expanded == _ABSOLUTE))
    absolute = absolute[absolute + 2 < expanded.size]
    produces = numpy.ones(expanded.size, dtype=bool)
    produces[absolute + 1] = False
    produces[absolute + 2] = False
    tokens = numpy.flatnonzero(produces)
    steps = expanded.view(numpy.int8).astype(numpy.int64)[tokens]
    is_absolute = numpy.zeros(expanded.size, dtype=bool)
    is_absolute[absolute] = True
    is_absolute = is_absolute[tokens]
    steps[is_absolute] = 0
    running = numpy.cumsum(steps)
    values = expanded[absolute + 1].astype(numpy.int64) + 256 * expanded[absolute + 2].astype(numpy.int64)
    order = numpy.arange(tokens.size)
    last = numpy.maximum.accumulate(numpy.where(is_absolute, order, -1))
    base = numpy.zeros(tokens.size, dtype=numpy.int64)
    if values.size:
        set_at = numpy.zeros(tokens.size, dtype=numpy.int64)
        set_at[is_absolute] = values
        seen = last >= 0
        base[seen] = set_at[last[seen]] - running[last[seen]]
    return ((running + base) & 0xFFFF).astype(numpy.uint16)


# One slice


@dataclass
class SbiSlice:
    """The decoded components of one SBI instance."""

    version: int
    #: Component name -> (pixel values as a 2D array (rows, columns), slope, intercept).
    components: dict[str, tuple[numpy.ndarray, float, float]]


def parse_sbi_payload(payload: bytes, *, wanted: Sequence[str] | None = None) -> SbiSlice:
    """Split an SBI instance's pixel data into its components and decode the *wanted* ones
    (default: all)."""
    words = numpy.frombuffer(payload[:12], dtype="<i4")
    if words.size < 3:
        raise ValueError("Not an SBI payload (too short).")
    header_words, version, count = (int(v) for v in words)
    if not (3 <= header_words <= 1024 and 0 < count <= 64 and header_words >= 3 + 6 * count):
        raise ValueError(f"Not an SBI payload (header {header_words}, {count} components).")
    table = payload[12:12 + 24 * count]
    ints = numpy.frombuffer(table, dtype="<i4").reshape(count, 6)
    floats = numpy.frombuffer(table, dtype="<f4").reshape(count, 6)
    offset = 4 * header_words
    components: dict[str, tuple[numpy.ndarray, float, float]] = {}
    for (length, cid, *_rest), frow in zip(ints.tolist(), floats.tolist()):
        stream = payload[offset: offset + int(length)]
        offset += int(length)
        name = COMPONENTS.get(int(cid), f"component_{int(cid)}")
        if wanted is not None and name not in wanted:
            continue
        flat = decode_pmsct_rle(stream)
        side = int(round(flat.size ** 0.5))
        if side * side != flat.size:
            raise ValueError(f"SBI component {cid} decoded to {flat.size} values, not a square image.")
        components[name] = (flat.reshape(side, side), float(frow[4]), float(frow[5]))
    return SbiSlice(version=version, components=components)


# A series


@dataclass
class SbiVolume:
    """An SBI series as volumes on the reconstruction's grid (``X x Y x Z``, as the
    conventional series loads), each component in its physical units: the conventional
    image in HU, the photoelectric and scatter parts as HU + 1000 contributions minus 1000
    (so air is -1000 in both), masks 0 / 1."""

    components: dict[str, numpy.ndarray]
    affine: numpy.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def conventional(self) -> numpy.ndarray:
        return self.components["conventional"]

    @property
    def photo(self) -> numpy.ndarray:
        return self.components["photo"]

    @property
    def scatter(self) -> numpy.ndarray:
        return self.components["scatter"]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(next(iter(self.components.values())).shape)

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(float(numpy.linalg.norm(self.affine[:3, k])) for k in range(3))  # type: ignore[return-value]

    def image(self, name: str, data: Any = None, *, image_name: str | None = None) -> Any:
        """An :class:`~nvitk.types.Image` of component *name* (or of *data* on this grid)."""
        from nvitk.types import Image

        arr = self.components[name] if data is None else data
        meta = dict(self.metadata)
        meta.update(affine=self.affine.copy(), axes="XYZ", shape=tuple(int(n) for n in numpy.shape(arr)),
                    spacing=self.spacing)
        meta["x_res"], meta["y_res"], meta["z_res"] = self.spacing
        return Image(data=arr, metadata=meta, axes="XYZ",
                     name=image_name or f"{self.metadata.get('sbi_label', 'SBI')} {name}")


def _position(ds: Any) -> numpy.ndarray:
    return numpy.asarray([float(v) for v in ds.ImagePositionPatient], dtype=float)


def _orientation(ds: Any) -> tuple[numpy.ndarray, numpy.ndarray]:
    iop = [float(v) for v in ds.ImageOrientationPatient]
    return numpy.asarray(iop[:3]), numpy.asarray(iop[3:])


def sbi_series_label(ds: Any) -> str:
    """``"IMR, 78%"``: the reconstruction an SBI series belongs to (after the ``|``)."""
    text = str(ds.get("SeriesDescription", "") or ds.get("ImageComments", "") or "")
    return text.split("|", 1)[1].strip() if "|" in text else text.strip()


def _series_metadata(first: Any, n_slices: int, version: int, components: Sequence[str]) -> dict[str, Any]:
    description = str(first.get("SeriesDescription", "") or "")
    match = re.match(r"SBI\(V?([\d.]+)\)", description)
    meta: dict[str, Any] = {
        "Modality": "CT",
        "Manufacturer": str(first.get("Manufacturer", "") or ""),
        "SeriesDescription": description,
        "SeriesNumber": str(first.get("SeriesNumber", "") or ""),
        "SeriesInstanceUID": str(first.get("SeriesInstanceUID", "") or ""),
        "StudyInstanceUID": str(first.get("StudyInstanceUID", "") or ""),
        "FrameOfReferenceUID": str(first.get("FrameOfReferenceUID", "") or ""),
        "PatientID": str(first.get("PatientID", "") or ""),
        "StudyDate": str(first.get("StudyDate", "") or ""),
        "KVP": str(first.get("KVP", "") or ""),
        "ConvolutionKernel": str(first.get("ConvolutionKernel", "") or ""),
        "SliceThickness": str(first.get("SliceThickness", "") or ""),
        "spectral_source": "philips_sbi",
        "sbi_version": match.group(1) if match else f"{version / 10:.1f}",
        "sbi_format": int(version),
        "sbi_recon": sbi_series_label(first),
        "sbi_label": f"SBI {sbi_series_label(first) or description}".strip(),
        "sbi_components": list(components),
        "sbi_slices": int(n_slices),
    }
    try:
        meta["sbi_layout"] = str(first[0x01F7, 0x10CC].value)
    except Exception:  # noqa: BLE001 (informational)
        pass
    for key in ("WindowCenter", "WindowWidth"):
        value = first.get(key, None)
        try:
            meta[key] = float(value[0] if isinstance(value, (list, tuple)) or hasattr(value, "__getitem__")
                              and not isinstance(value, str) else value)
        except (TypeError, ValueError, IndexError):
            pass
    return meta


def read_sbi_series(
    paths: Sequence[str],
    *,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SbiVolume:
    """Decode the SBI instances *paths* (one series) into volumes of *components*.

    The slices are ordered along the slice normal; the volumes are ``X x Y x Z`` with the
    affine (RAS) the conventional series of the same reconstruction loads with, so the two
    overlay exactly.
    """
    import pydicom

    from nvitk.io.conversors._dicom_philips import is_philips_spectral_base_image

    wanted = tuple(components)
    headers = []
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if is_philips_spectral_base_image(ds) and "ImagePositionPatient" in ds:
            headers.append((path, ds))
    if not headers:
        raise ValueError("No Philips spectral base images among the files.")
    uids = {str(ds.get("SeriesInstanceUID", "")) for _p, ds in headers}
    if len(uids) > 1:
        raise ValueError(f"The files hold {len(uids)} SBI series; read one at a time.")
    row_dir, col_dir = _orientation(headers[0][1])
    normal = numpy.cross(row_dir, col_dir)
    headers.sort(key=lambda item: float(_position(item[1]) @ normal))
    positions = numpy.asarray([_position(ds) for _p, ds in headers])

    def decode(index: int) -> tuple[int, SbiSlice]:
        ds = pydicom.dcmread(headers[index][0])
        return index, parse_sbi_payload(bytes(ds.PixelData), wanted=wanted)

    n = len(headers)
    slices: list[SbiSlice | None] = [None] * n
    done = 0
    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
        for index, sl in pool.map(decode, range(n)):
            slices[index] = sl
            done += 1
            if progress is not None:
                progress(done, n)
    first_slice = slices[0]
    assert first_slice is not None
    names = [name for name in wanted if name in first_slice.components]
    if not names:
        raise ValueError(f"None of {list(wanted)} in this SBI series ({list(first_slice.components)}).")
    side = first_slice.components[names[0]][0].shape[0]
    volumes: dict[str, numpy.ndarray] = {}
    for name in names:
        is_mask = name.startswith("mask")
        vol = numpy.empty((side, side, n), dtype=numpy.uint8 if is_mask else numpy.int16)
        for k, sl in enumerate(slices):
            pixels, slope, intercept = sl.components[name]  # type: ignore[union-attr]
            values = pixels.astype(numpy.float32) * slope + intercept
            if name in ("photo", "scatter"):
                # Outside the reconstructed circle (and along its rim) the streams hold 0 or
                # near it ("no data"): a contribution far below air's is not physical;
                # nothing attenuates there.
                values[values < -1100.0] = -1000.0
            # Rows of the decoded image run along the column direction: X is the column index.
            vol[:, :, k] = values.T.astype(vol.dtype) if not is_mask else (values.T > 0)
        volumes[name] = vol

    first = headers[0][1]
    spacing = [float(v) for v in first.PixelSpacing]   # (row spacing, column spacing)
    if n > 1:
        steps = numpy.diff(positions @ normal)
        dz = float(numpy.median(steps))
    else:
        dz = float(first.get("SliceThickness", 1.0) or 1.0)
    lps = numpy.eye(4)
    lps[:3, 0] = row_dir * spacing[1]
    lps[:3, 1] = col_dir * spacing[0]
    lps[:3, 2] = normal * dz
    lps[:3, 3] = positions[0]
    affine = numpy.diag([-1.0, -1.0, 1.0, 1.0]) @ lps     # DICOM LPS -> RAS
    metadata = _series_metadata(first, n, first_slice.version, names)
    return SbiVolume(components=volumes, affine=affine, metadata=metadata)


__all__ = [
    "COMPONENTS",
    "DEFAULT_COMPONENTS",
    "SbiSlice",
    "SbiVolume",
    "decode_pmsct_rle",
    "parse_sbi_payload",
    "read_sbi_series",
    "sbi_series_label",
]
