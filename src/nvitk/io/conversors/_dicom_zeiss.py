"""Zeiss OCT raw DICOM storage: extract B-scans / volumes for the conversion pipeline."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
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
    "ZeissOctCube",
    "ZEISS_OCT_CUBE_CONTAINERS",
    "extract_zeiss_oct_cubes",
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

_TAG_FRAME_ROWS = (0x0407, 0x1001)
_TAG_FRAME_COLS = (0x0407, 0x1002)
_TAG_FRAME_COUNT = (0x0407, 0x1003)
_TAG_FRAME_SEQUENCE = (0x0407, 0x1005)
_TAG_FRAME_DATA = (0x0407, 0x1006)
_TAG_SPACING_X = (0x0407, 0x100A)
_TAG_SPACING_DEPTH = (0x0407, 0x100B)
_TAG_SPACING_Y = (0x0407, 0x100C)
_TAG_SCAN_PATTERN = (0x0405, 0x1001)  # e.g. ANGIO_6MM, MACULAR_CUBE


@dataclass(frozen=True)
class ZeissOctCube:
    """One volumetric cube extracted from a Zeiss private OCT container."""

    kind: str
    array: np.ndarray
    affine: np.ndarray
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


def _decode_zeiss_frame(payload: Any, expected_shape: tuple[int, int] | None = None) -> np.ndarray | None:
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
    return None


def _extract_container_cube(
    ds: Any,
    element: int,
    kind: str,
    debug_mode: bool = False,
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
    for frame_item in frame_items:
        try:
            payload = frame_item[_TAG_FRAME_DATA].value
        except Exception:
            return None
        decoded = _decode_zeiss_frame(payload, expected_shape=(rows, cols))
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

    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    meta = {
        "method": "zeiss_private_container",
        "zeiss_container_tag": f"0407,{element:04X}",
        "zeiss_cube_kind": kind,
        "zeiss_scan_pattern": str(scan_pattern) if scan_pattern is not None else None,
        "zeiss_spacing_source": spacing_source,
        "zeiss_spacing_mm": list(spacing),
        "num_slices": int(volume.shape[2]),
        "chosen_shape": tuple(int(size) for size in volume.shape),
    }
    if debug_mode:
        log.debug(f"[Zeiss cubes] {kind}: shape={volume.shape} spacing={spacing} source={spacing_source}")
    return ZeissOctCube(kind=kind, array=volume, affine=affine, meta=meta)


def has_zeiss_oct_cubes(ds: Any) -> bool:
    """True when *ds* carries at least one private OCT cube container."""
    return any((0x0407, element) in ds for element in ZEISS_OCT_CUBE_CONTAINERS)


def extract_zeiss_oct_cubes(
    ds_list: list[Any],
    debug_mode: bool = False,
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
            cube = _extract_container_cube(ds, element, kind, debug_mode=debug_mode)
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
