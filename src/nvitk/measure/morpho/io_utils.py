"""Filesystem, filename, sheet-name, NIfTI, and mapping I/O helpers."""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple

import nibabel as nib
import numpy as np

from nvitk.measure.morphometrics_config import EXPORT_ANATOMIC_SPLIT_CENTERLINES
from .models import VesselInfo

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_generated_vtp_dir(path: Optional[str]) -> None:
    if not path or not os.path.isdir(path):
        return
    removed = 0
    for name in os.listdir(path):
        if name.endswith(".vtp"):
            os.remove(os.path.join(path, name))
            removed += 1
    if removed:
        print(f"Cleaned {removed} old VTP file(s) from {path}")


def clean_legacy_centerline_vtp_names(path: Optional[str]) -> None:
    if not EXPORT_ANATOMIC_SPLIT_CENTERLINES or not path or not os.path.isdir(path):
        return
    legacy_re = re.compile(
        r"^\d+_.*_(?:trunk|arm\d+|branch\d+|subarm\d+|trunk_plus_.+)(?:_radius)?\.vtp$",
        flags=re.IGNORECASE,
    )
    removed = 0
    for name in os.listdir(path):
        if legacy_re.match(name):
            os.remove(os.path.join(path, name))
            removed += 1
    if removed:
        print(f"Removed {removed} legacy-named centerline VTP file(s) from {path}")


def safe_sheet_name(name: str) -> str:
    name = re.sub(r"[:\\/*?\[\]]", "_", name)
    return name[:31]


def vessel_sheet_sort_key(sheet_name: str) -> Tuple[int, str]:
    match = re.match(r"^(\d+)(?:\D|$)", sheet_name)
    if not match:
        match = re.search(r"(?:^|_)label[_-]?(\d+)(?:\D|$)", sheet_name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), sheet_name
    return 10**9, sheet_name


def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def load_multilabel_nifti(path: str):
    img = nib.load(path)
    data = img.get_fdata().astype(np.int32)
    affine = img.affine
    spacing = img.header.get_zooms()[:3]
    return data, affine, spacing


def load_mapping(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping = {}
    for k, v in raw.items():
        try:
            label = int(k)
        except Exception:
            continue
        mapping[label] = VesselInfo(
            name=str(v.get("name", f"label_{label}")),
            full_name=str(v.get("full_name", "")),
            side=str(v.get("side", "")).upper(),
            pair=v.get("pair"),
            territory=str(v.get("territory", "")),
            flow_from=str(v.get("flow_from", "")),
            flow_to=v.get("flow_to", []),
            no_upstream_start=v.get("no_upstream_start", None),
        )
    return mapping
