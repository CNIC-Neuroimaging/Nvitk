from __future__ import annotations

import io
from collections import defaultdict
from typing import Any

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from pydicom.tag import Tag
except Exception:
    Tag = None

RAW_DATA_SOP_UID = "1.2.840.10008.5.1.4.1.1.66"


def is_zeiss_raw_storage(ds: Any) -> bool:
    manufacturer = str(getattr(ds, "Manufacturer", "") or "")
    model = str(getattr(ds, "ManufacturerModelName", "") or "")
    sop = str(getattr(ds, "SOPClassUID", "") or "")

    if "ZEISS" in manufacturer.upper() or "CIRRUS" in model.upper():
        return True
    if sop == RAW_DATA_SOP_UID or "RAW DATA STORAGE" in sop.upper():
        return True

    # Known private creator fingerprints.
    try:
        for elem in ds:
            if getattr(elem, "keyword", None):
                continue
            if "99CZM" in str(getattr(elem, "value", "")):
                return True
    except Exception:
        pass
    return False


def _find_candidate_shape(ds_list: list[Any], metadata: dict[str, Any]) -> tuple[int | None, int | None]:
    width = None
    height = None

    for ds in ds_list:
        try:
            c = int(getattr(ds, "Columns", 0) or 0)
            r = int(getattr(ds, "Rows", 0) or 0)
            if c > 0 and width is None:
                width = c
            if r > 0 and height is None:
                height = r
            if width and height:
                return width, height
        except Exception:
            continue

    for key in ("Columns", "(0028,0011)", "ImageColumns"):
        if key in metadata and width is None:
            try:
                width = int(metadata[key])
            except Exception:
                pass
    for key in ("Rows", "(0028,0010)", "ImageRows"):
        if key in metadata and height is None:
            try:
                height = int(metadata[key])
            except Exception:
                pass
    return width, height


def _collect_private_blocks(ds: Any) -> list[bytes]:
    out: list[bytes] = []
    try:
        for elem in ds.iterall():
            if Tag is not None and not isinstance(elem.tag, Tag):
                continue
            if elem.tag.group < 0x0400:
                continue
            if elem.VR not in ("OB", "OW", "OF"):
                continue
            value = getattr(elem, "value", None)
            if isinstance(value, (bytes, bytearray)) and len(value) > 0:
                out.append(bytes(value))
    except Exception:
        return out
    return out


def _decode_image_block(payload: bytes) -> np.ndarray | None:
    if not payload or Image is None:
        return None
    try:
        with Image.open(io.BytesIO(payload)) as img:
            gray = img.convert("L")
            return np.asarray(gray)
    except Exception:
        return None


def _crop_or_pad(array2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    arr = np.asarray(array2d)
    h, w = arr.shape
    if h > target_h:
        off = (h - target_h) // 2
        arr = arr[off : off + target_h, :]
    if w > target_w:
        off = (w - target_w) // 2
        arr = arr[:, off : off + target_w]

    pad_h = target_h - arr.shape[0]
    pad_w = target_w - arr.shape[1]
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        arr = np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    return arr


def extract_zeiss_raw_oct(ds_list: list[Any], metadata: dict[str, Any], debug_mode: bool = False):
    width, height = _find_candidate_shape(ds_list, metadata)

    blocks_by_instance: dict[int, list[np.ndarray]] = defaultdict(list)
    for idx, ds in enumerate(ds_list):
        for payload in _collect_private_blocks(ds):
            arr = _decode_image_block(payload)
            if arr is not None and arr.ndim == 2:
                blocks_by_instance[idx].append(arr)

    # 1) Try instance-level tile assembly.
    assembled: list[tuple[int, np.ndarray]] = []
    for idx in sorted(blocks_by_instance.keys()):
        tiles = blocks_by_instance[idx]
        if not tiles:
            continue
        if len(tiles) == 1:
            assembled2d = tiles[0]
        else:
            same_h = len({t.shape[0] for t in tiles}) == 1
            same_w = len({t.shape[1] for t in tiles}) == 1
            assembled2d = None
            if same_h:
                try:
                    assembled2d = np.concatenate(tiles, axis=1)
                except Exception:
                    assembled2d = None
            if assembled2d is None and same_w:
                try:
                    assembled2d = np.concatenate(tiles, axis=0)
                except Exception:
                    assembled2d = None
            if assembled2d is None:
                assembled2d = tiles[0]
        assembled.append((idx, assembled2d))

    # 2) Fallback to all decoded blocks if instance assembly failed.
    if not assembled:
        for idx, arrs in blocks_by_instance.items():
            for arr in arrs:
                assembled.append((idx, arr))

    if not assembled:
        raise RuntimeError("Zeiss RAW extraction failed: no decodable private image blocks.")

    assembled.sort(key=lambda x: x[0])
    slices = [arr for _, arr in assembled]

    # Choose target shape from most common slice shape, then align to candidate geometry if present.
    shape_counts: dict[tuple[int, int], int] = defaultdict(int)
    for arr in slices:
        shape_counts[arr.shape] += 1
    common_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]
    target_h, target_w = common_shape

    if width and height:
        if abs(target_w - width) > max(4, int(0.1 * width)) or abs(target_h - height) > max(4, int(0.1 * height)):
            # Accept swapped candidate if closer.
            score_native = abs(target_w - width) + abs(target_h - height)
            score_swap = abs(target_w - height) + abs(target_h - width)
            if score_swap < score_native:
                target_h, target_w = width, height
            else:
                target_h, target_w = height, width

    norm = [_crop_or_pad(arr, target_h, target_w) for arr in slices]
    volume_zyx = np.stack(norm, axis=0)  # Z, Y, X
    volume_xyz = volume_zyx.transpose(2, 1, 0)  # X, Y, Z

    meta = {
        "method": "zeiss_private_tags",
        "num_slices": int(volume_xyz.shape[2]),
        "chosen_shape": (int(volume_xyz.shape[0]), int(volume_xyz.shape[1]), int(volume_xyz.shape[2])),
    }
    if debug_mode:
        meta["debug_shape_counts"] = {f"{k[0]}x{k[1]}": int(v) for k, v in shape_counts.items()}

    return volume_xyz, np.eye(4), meta

