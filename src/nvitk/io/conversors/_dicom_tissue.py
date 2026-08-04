"""Tissue-segmentation DICOM private tags → numpy volumes."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from nvitk.core.logger import Logger

try:
    import pydicom
except Exception:
    pydicom = None

__all__ = [
    "extract_tissue_segmentation_data",
    "is_tissue_segmentation",
]

_DEBUG = False
log = Logger()


def _warn(message: str) -> None:
    """Log a warning through the module logger."""
    log.warning(message)


def _debug(message: str) -> None:
    """Log *message* at debug level, gated by the module's ``_DEBUG`` flag."""
    if _DEBUG:
        log.debug(message)


def is_tissue_segmentation(ds: Any) -> bool:
    """Return True when a DICOM dataset stores a TISSUE private segmentation mask."""
    if pydicom is None:
        return False

    try:
        image_type = ds.get("ImageType", None)
        if image_type and "TISSUE" in image_type:
            return True

        has_shape_tag = hasattr(ds, "(07A1,1007)") or "(07A1,1007)" in ds
        has_affine_tag = hasattr(ds, "(07A1,1008)") or "(07A1,1008)" in ds
        has_data_tag = hasattr(ds, "(07A1,1009)") or "(07A1,1009)" in ds
        return has_shape_tag and has_affine_tag and has_data_tag
    except Exception as exc:
        _debug(f"Error checking TISSUE segmentation: {exc}")
        return False


def _decode_tissue_data(
    compressed_data: bytes,
    expected_size: int,
    shape_hint: tuple[int, int, int] | None = None,
) -> bytes | None:
    """
    Decode TISSUE segmentation payload using the legacy RLE + delta logic.

    The returned payload is a flat uint8 binary mask stream of length
    ``expected_size``.
    """
    try:
        if not isinstance(compressed_data, (bytes, bytearray)):
            return None

        try:
            checksum = hashlib.md5(compressed_data).hexdigest()
        except Exception:
            checksum = "<md5-failed>"
        _debug(
            "TISSUE decode: "
            f"compressed_len={len(compressed_data)} expected_len={expected_size} md5={checksum}"
        )
        _debug(f"TISSUE decode: compressed_head={compressed_data[:32].hex()}")

        # RLE expand (0xA5 marker).
        temp = bytearray()
        i = 0
        n = len(compressed_data)
        a5_sequences = 0
        a5_truncated = 0
        literal_count = 0
        total_repeats = 0
        while i < n:
            b = compressed_data[i]
            if b == 0xA5:
                if i + 2 >= n:
                    a5_truncated += 1
                    temp.append(b)
                    i += 1
                    continue
                repeat = compressed_data[i + 1] + 1
                value = compressed_data[i + 2]
                if repeat > 0:
                    temp.extend([value] * repeat)
                    total_repeats += repeat
                a5_sequences += 1
                i += 3
            else:
                temp.append(b)
                literal_count += 1
                i += 1

        _debug(
            "TISSUE RLE: "
            f"out_len={len(temp)} a5_sequences={a5_sequences} a5_truncated={a5_truncated} "
            f"literals={literal_count} total_repeats={total_repeats}"
        )
        _debug(f"TISSUE RLE: out_head={bytes(temp[:32]).hex()}")

        # Delta decode into 16-bit values (0x5A absolute markers).
        output_values: list[int] = []
        delta = 0
        i = 0
        m = len(temp)
        abs_markers = 0
        abs_truncated = 0
        delta_literals = 0
        while i < m:
            token = temp[i]
            if token == 0x5A:
                if i + 2 >= m:
                    abs_truncated += 1
                    break
                v1 = temp[i + 1]
                v2 = temp[i + 2]
                value = (v2 * 256) + v1
                output_values.append(value)
                delta = value
                abs_markers += 1
                i += 3
            else:
                value = token + delta
                output_values.append(value)
                delta = value
                delta_literals += 1
                i += 1

        if len(output_values) % 2:
            output_values = output_values[:-1]

        if output_values:
            vmin, vmax = int(min(output_values)), int(max(output_values))
            nonzero = int(sum(1 for value in output_values if value != 0))
        else:
            vmin = vmax = nonzero = 0

        _debug(
            "TISSUE delta: "
            f"values={len(output_values)} abs_markers={abs_markers} abs_truncated={abs_truncated} "
            f"delta_literals={delta_literals} min={vmin} max={vmax} nonzero={nonzero}"
        )
        _debug(f"TISSUE delta: head_vals={list(output_values[:16])}")

        # Continuous Run-Length Toggle (RLT) across full 3D stack in X-fastest order.
        counts = np.array(output_values, dtype=np.int64)
        if (counts < 0).any():
            neg = int(np.count_nonzero(counts < 0))
            _debug(f"TISSUE RLT: found {neg} negative runs; clamping to 0")
            counts[counts < 0] = 0

        total_runs = counts.size
        sum_counts = int(counts.sum())
        _debug(f"TISSUE RLT: runs={total_runs} sum={sum_counts} expected={expected_size}")

        flat = np.zeros(expected_size, dtype=np.uint8)
        idx = 0
        state = 0
        clipped_last = False
        for run_len in counts:
            if run_len <= 0:
                state ^= 1
                continue
            end = idx + int(run_len)
            if end > expected_size:
                end = expected_size
                clipped_last = True
            if state == 1:
                flat[idx:end] = 1
            idx = end
            state ^= 1
            if idx >= expected_size:
                break

        if idx < expected_size:
            _debug(f"TISSUE RLT: stream underfilled by {expected_size - idx} voxels; remaining set to 0")
        if sum_counts != expected_size:
            _debug(
                "TISSUE RLT: "
                f"sum_counts ({sum_counts}) != expected_size ({expected_size}); "
                f"clipped_last={clipped_last}"
            )
        _debug(f"TISSUE RLT: ended at idx={idx}")
        return flat.tobytes()
    except Exception as exc:
        _warn(f"TISSUE decoder failed: {exc}")
        return None


def _debug_log_mask_layout(decoded_bytes: bytes, shape: tuple[int, int, int]) -> None:
    """Emit detailed debug logs to diagnose axis-order issues for TISSUE masks."""
    try:
        arr_flat = np.frombuffer(decoded_bytes, dtype=np.uint8)
        total = arr_flat.size
        expected = int(np.prod(shape))
        _debug(f"MASK DEBUG: flat_len={total} expected={expected} shape={shape}")
        if total != expected:
            _debug("MASK DEBUG: flat length != expected; layout analysis may be misleading")

        def summarize(volume: np.ndarray, name: str) -> None:
            """Debug-log a mask volume's foreground count and its first/last nonzero Z slice."""
            ones_total = int(volume.sum())
            z_counts = volume.sum(axis=(0, 1)).astype(int)
            nonzero_slices = int(np.count_nonzero(z_counts))
            try:
                z_first = int(np.argmax(z_counts > 0)) if nonzero_slices > 0 else -1
                z_last = int(len(z_counts) - 1 - np.argmax(z_counts[::-1] > 0)) if nonzero_slices > 0 else -1
            except Exception:
                z_first = z_last = -1

            xy_any = (volume > 0).any(axis=2)
            xs = np.where(xy_any.any(axis=1))[0]
            ys = np.where(xy_any.any(axis=0))[0]
            if xs.size and ys.size:
                bbox = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
            else:
                bbox = (-1, -1, -1, -1)

            head_slices = [int(c) for c in z_counts[:3]]
            tail_slices = [int(c) for c in z_counts[-3:]]
            _debug(
                f"MASK DEBUG [{name}]: ones_total={ones_total} nonzero_slices={nonzero_slices} "
                f"z_range=[{z_first},{z_last}] z_head={head_slices} "
                f"z_tail={tail_slices} bbox_xy={bbox}"
            )

        x, y, z = shape
        combos = [
            ("xyz_C", (x, y, z), "C"),
            ("zyx_C", (z, y, x), "C"),
            ("yxz_C", (y, x, z), "C"),
            ("xyz_F", (x, y, z), "F"),
        ]
        for name, dims, order in combos:
            try:
                volume = np.array(arr_flat.reshape(dims, order=order), copy=False)
                summarize(volume, name)
            except Exception as exc:
                _debug(f"MASK DEBUG [{name}]: reshape failed for dims={dims} order={order}: {exc}")
    except Exception as exc:
        _debug(f"MASK DEBUG: failed to summarize layout: {exc}")


def extract_tissue_segmentation_data(ds: Any):
    """
    Extract segmentation data from TISSUE DICOM private tags.

    Returns:
        (segmentation_array, affine_matrix, shape_info) or (None, None, None)
    """
    try:
        shape_tag = (0x07A1, 0x1007)
        shape_data = ds.get(shape_tag, None)
        if shape_data:
            shape_data = shape_data.value
            if isinstance(shape_data, (list, tuple)) and len(shape_data) >= 3:
                shape = tuple(int(v) for v in shape_data[:3])
                _debug(f"Extracted shape from private tag: {shape}")
            else:
                _warn(f"Invalid shape data in private tag (07A1,1007): {shape_data}")
                return None, None, None
        else:
            _warn("Shape tag (07A1,1007) not found")
            return None, None, None

        affine_tag = (0x07A1, 0x1008)
        affine_data = ds.get(affine_tag, None)
        if affine_data:
            affine_values = affine_data.value
            if len(affine_values) >= 12:
                points = np.array(affine_values[:12]).reshape(4, 3)
                affine = np.eye(4)
                origin = points[0]
                affine[:3, 3] = origin

                if len(points) >= 3:
                    x_spacing = np.linalg.norm(points[1] - points[0]) / shape[0] if shape[0] > 0 else 1.0
                    y_spacing = np.linalg.norm(points[2] - points[0]) / shape[1] if shape[1] > 0 else 1.0
                    z_spacing = np.linalg.norm(points[3] - points[0]) / shape[2] if shape[2] > 0 else 1.0
                    affine[0, 0] = x_spacing
                    affine[1, 1] = y_spacing
                    affine[2, 2] = z_spacing
                    _debug("Constructed affine matrix from corner points")
                    _debug(f"  Origin: {origin}")
                    _debug(
                        f"  Spacing: x={x_spacing:.2f}, y={y_spacing:.2f}, z={z_spacing:.2f}"
                    )
                else:
                    affine[0, 0] = 1.0
                    affine[1, 1] = 1.0
                    affine[2, 2] = 1.0
            else:
                _warn(f"Invalid affine data in private tag (07A1,1008): {affine_values}")
                affine = np.eye(4)
        else:
            _warn("Affine tag (07A1,1008) not found, using identity matrix")
            affine = np.eye(4)

        data_tag = (0x07A1, 0x1009)
        pixel_data = ds.get(data_tag, None)
        if not pixel_data:
            _warn("Segmentation data tag (07A1,1009) not found")
            return None, None, None

        pixel_bytes = pixel_data.value
        expected_size = int(np.prod(shape))
        decoded = _decode_tissue_data(pixel_bytes, expected_size, shape)
        if decoded is None:
            _warn("Unified tissue decoder failed to decode segmentation payload")
            return None, None, None

        segmentation_array = np.frombuffer(decoded, dtype=np.uint8).reshape(shape)
        _debug(f"Extracted TISSUE segmentation data with shape: {segmentation_array.shape}")
        ones_total = int(segmentation_array.sum())
        z_counts = segmentation_array.sum(axis=(0, 1)).astype(int)
        nz_slices = int(np.count_nonzero(z_counts))
        try:
            z_first = int(np.argmax(z_counts > 0)) if nz_slices > 0 else -1
            z_last = int(len(z_counts) - 1 - np.argmax(z_counts[::-1] > 0)) if nz_slices > 0 else -1
        except Exception:
            z_first = z_last = -1
        _debug(
            "MASK DEBUG (current): "
            f"ones_total={ones_total} nz_slices={nz_slices} z_range=[{z_first},{z_last}] "
            f"z_head={[int(c) for c in z_counts[:3]]} z_tail={[int(c) for c in z_counts[-3:]]}"
        )
        _debug_log_mask_layout(decoded, shape)
        return segmentation_array, affine, shape
    except Exception as exc:
        _warn(f"Error extracting TISSUE segmentation data: {exc}")
        return None, None, None
