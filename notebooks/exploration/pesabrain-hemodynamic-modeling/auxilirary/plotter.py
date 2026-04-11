#!/usr/bin/env python3
"""
Core visualization functions for QVT+ analysis.

This module provides:
- plot_violin: Violin plots for features grouped by vessel groups
- plot_correlation_heatmap: Correlation heatmap with patient annotations
- plot_flow_timeseries: Flow plots over time for individual patients
- plot_flow_timeseries_all: Flow plots for all patients together
- plot_crosssections: Mosaic plots showing cross-sections at vessel LOCs with highlighted measurement pixels
- plot_tree: Dendrogram/tree visualization of Circle of Willis and venous system structure
"""

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import imageio for GIF creation
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    try:
        from PIL import Image
        HAS_PIL = True
    except ImportError:
        HAS_PIL = False

# Try to import h5py for MATLAB v7.3 files
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    warnings.warn("h5py not available. Cannot load MATLAB v7.3 files. Install with: pip install h5py")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------------
# Mappings & constants
# -----------------------------------------------------------------------------

# Vessel code to display name mapping
_VESSEL_CODE_TO_NAME = {
    'LICA': 'Left ICA',
    'RICA': 'Right ICA',
    'BASI': 'Basilar',
    'LMCA': 'Left MCA',
    'RMCA': 'Right MCA',
    'LACA': 'Left ACA',
    'RACA': 'Right ACA',
    'LPCA': 'Left PCA',
    'RPCA': 'Right PCA',
    'SSSV': 'Sagital Sinus',
    'LTSV': 'Left Transverse',
    'RTSV': 'Right Transverse',
    'STRV': 'Straight Sinus',
    'COMM': 'Communicating',
    'LCOMM': 'Left Communicating',
    'RCOMM': 'Right Communicating',
}

# Alternative vessel label names that might appear in Excel files
_VESSEL_LABEL_ALIASES = {
    'Basilar': ['Basilar', 'Basilar Artery', 'Basilar artery', 'basilar', 'BASILAR'],
    'Left ICA': ['Left ICA', 'Left Internal Carotid', 'LICA', 'left ica'],
    'Right ICA': ['Right ICA', 'Right Internal Carotid', 'RICA', 'right ica'],
    'Left MCA': ['Left MCA', 'Left Middle Cerebral', 'LMCA', 'left mca'],
    'Right MCA': ['Right MCA', 'Right Middle Cerebral', 'RMCA', 'right mca'],
    'Left ACA': ['Left ACA', 'Left Anterior Cerebral', 'LACA', 'left aca'],
    'Right ACA': ['Right ACA', 'Right Anterior Cerebral', 'RACA', 'right aca'],
    'Left PCA': ['Left PCA', 'Left Posterior Cerebral', 'LPCA', 'left pca'],
    'Right PCA': ['Right PCA', 'Right Posterior Cerebral', 'RPCA', 'right pca'],
    'Sagital Sinus': ['Sagital Sinus', 'Sagittal Sinus', 'Superior Sagittal Sinus', 'SSSV'],
    'Left Transverse': ['Left Transverse', 'Left Transverse Sinus', 'LTSV', 'left transverse'],
    'Right Transverse': ['Right Transverse', 'Right Transverse Sinus', 'RTSV', 'right transverse'],
    'Straight Sinus': ['Straight Sinus', 'STRV', 'straight sinus'],
    'Communicating': ['Communicating', 'COMM', 'communicating'],
    'Right Communicating': ['Right Communicating', 'RCOMM', 'right communicating'],
    'Left Communicating': ['Left Communicating', 'LCOMM', 'left communicating'],
}


def _match_vessel_label(vessel_label: str) -> Optional[Tuple[str, str]]:
    """
    Match a vessel label from Excel to vessel code and name.
    Uses flexible matching to handle variations in naming.
    
    Args:
        vessel_label: Vessel label string from Excel file
        
    Returns:
        Tuple of (vessel_code, vessel_name) if match found, None otherwise
    """
    if pd.isna(vessel_label) or not vessel_label:
        return None
    
    vessel_label = str(vessel_label).strip()
    
    # First try exact match
    for code, name in _VESSEL_CODE_TO_NAME.items():
        if name == vessel_label:
            return (code, name)
    
    # Try case-insensitive match
    vessel_label_lower = vessel_label.lower()
    for code, name in _VESSEL_CODE_TO_NAME.items():
        if name.lower() == vessel_label_lower:
            return (code, name)
    
    # Try alias matching (check if label contains or matches any alias)
    for code, name in _VESSEL_CODE_TO_NAME.items():
        if name in _VESSEL_LABEL_ALIASES:
            aliases = _VESSEL_LABEL_ALIASES[name]
            for alias in aliases:
                if alias.lower() == vessel_label_lower or vessel_label_lower in alias.lower() or alias.lower() in vessel_label_lower:
                    return (code, name)
    
    # Try partial matching (check if vessel name is contained in label or vice versa)
    for code, name in _VESSEL_CODE_TO_NAME.items():
        name_lower = name.lower()
        if name_lower in vessel_label_lower or vessel_label_lower in name_lower:
            return (code, name)
    
    return None

# Vessel grouping for visualization
_VESSEL_GROUPING = {
    'TCBF & ICAs':           ['TCBF', 'Left ICA', 'Right ICA'],
    'Anterior circulation':  ['Left MCA', 'Right MCA', 'Left ACA', 'Right ACA', 'Right Communicating', 'Left Communicating'],
    'Posterior circulation': ['Basilar', 'Left PCA', 'Right PCA'],
    'Venous drainage':       ['Sagital Sinus', 'Straight Sinus', 'Left Transverse', 'Right Transverse'],
}

# Reverse mapping: display name to group
_VESSEL_NAME_TO_GROUP = {}
for group, vessels in _VESSEL_GROUPING.items():
    for vessel in vessels:
        _VESSEL_NAME_TO_GROUP[vessel] = group

_VIOLIN_ORDER = [
    'TCBF',
    'Left ICA',
    'Right ICA',
    'Left MCA',
    'Right MCA',
    'Left ACA',
    'Right ACA',
    # 'Left Communicating',
    # 'Right Communicating',
    'Basilar',
    'Left PCA',
    'Right PCA',
    'Sagital Sinus',
    'Straight Sinus',
    'Left Transverse',
    'Right Transverse',
]


@dataclass
class ViolinStyle:
    """Style configuration for violin plots."""
    figsize: Tuple[float, float] = (12, 6)
    palette: Optional[List[str]] = None
    inner: str = 'box'  # 'box', 'quartile', 'point', 'stick', None
    scale: str = 'width'  # 'area', 'count', 'width'
    dodge: bool = True
    alpha: float = 0.7
    linewidth: float = 1.0


def _load_mat_v73(mat_file: Path) -> Dict:
    """Load MATLAB v7.3 file using h5py."""
    if not HAS_H5PY:
        raise ImportError("h5py is required to load MATLAB v7.3 files. Install with: pip install h5py")
    
    data_struct = {}
    
    def _read_ref(ref):
        """Read a MATLAB reference."""
        if isinstance(ref, h5py.Reference):
            return f[ref]
        return ref
    
    def _read_dataset(dataset):
        """Read a MATLAB dataset, handling references and transposition."""
        if dataset.dtype == 'object':
            # Handle references
            refs = dataset[()]
            if refs.size == 1:
                ref = refs.flat[0]
                if isinstance(ref, h5py.Reference):
                    return _read_group(f[ref])
            elif refs.size > 1:
                result = []
                for ref in refs.flat:
                    if isinstance(ref, h5py.Reference):
                        result.append(_read_group(f[ref]))
                    else:
                        result.append(ref)
                return np.array(result)
        else:
            value = dataset[()]
            # MATLAB stores matrices transposed
            if value.ndim > 1:
                value = value.T
            # Handle complex numbers
            if np.iscomplexobj(value):
                value = np.abs(value)
            return value
    
    def _read_group(group):
        """Recursively read a MATLAB struct group."""
        result = {}
        for key in group.keys():
            obj = group[key]
            if isinstance(obj, h5py.Dataset):
                result[key] = _read_dataset(obj)
            elif isinstance(obj, h5py.Group):
                result[key] = _read_group(obj)
        return result
    
    with h5py.File(mat_file, 'r') as f:
        # Look for data_struct
        if 'data_struct' in f:
            data_struct = _read_group(f['data_struct'])
        else:
            # Extract all top-level groups (except MATLAB metadata)
            for key in f.keys():
                if not key.startswith('#') and not key.startswith('MATLAB'):
                    obj = f[key]
                    if isinstance(obj, h5py.Group):
                        data_struct[key] = _read_group(obj)
                    elif isinstance(obj, h5py.Dataset):
                        data_struct[key] = _read_dataset(obj)
    
    return {'data_struct': data_struct}


def load_patient_metadata(patient_dir: Union[str, Path]) -> Dict:
    """
    Load only lightweight metadata (no .mat file loading).
    This is much faster and uses minimal memory.
    
    Args:
        patient_dir: Path to patient output directory
        
    Returns:
        Dictionary with: patient_id, patient_dir, LOCs, summary_file
        (No data_struct - .mat files loaded on-demand when needed)
    """
    patient_dir = Path(patient_dir) 
    
    # Load LOCs from SummaryParamTool.xls (preferred source)
    locs = {}
    summary_file = patient_dir / 'centerline_test' / 'SummaryParamTool.xls'
    if summary_file.exists():
        try:
            # Try to read the Summary_Centerline sheet
            summary_df = pd.read_excel(summary_file, sheet_name='Summary_Centerline')
            for _, row in summary_df.iterrows():
                vessel_label = row.get('Vessel Label', '')
                branch_num = row.get('Branch Number', np.nan)
                centerline = row.get('Centerline', np.nan)
                if pd.notna(branch_num) and pd.notna(centerline):
                    # Map vessel label to code using flexible matching
                    match = _match_vessel_label(vessel_label)
                    if match:
                        code, _ = match
                        locs[code] = [int(branch_num), int(centerline)]
        except Exception as e:
            warnings.warn(f"Could not load LOCs from {summary_file}: {e}")
    
    # Fallback to LabelsQVT.csv if LOCs not found
    if not locs:
        labels_file = patient_dir / 'centerline_test' / 'LabelsQVT.csv'
        if labels_file.exists():
            try:
                labels_df = pd.read_csv(labels_file)
                for _, row in labels_df.iterrows():
                    artery = row.get('Artery', '')
                    loc_str = row.get('Loc', '')
                    if pd.notna(loc_str) and loc_str.strip() and loc_str != ' ':
                        # Parse LOC string like "[123, 45]" or "123"
                        loc_str = str(loc_str).strip()
                        if loc_str.startswith('[') and loc_str.endswith(']'):
                            loc_str = loc_str[1:-1]
                        try:
                            loc_parts = [int(x.strip()) for x in loc_str.split(',')]
                            if len(loc_parts) >= 2:
                                # Map artery name to code using flexible matching
                                match = _match_vessel_label(artery)
                                if match:
                                    code, _ = match
                                    locs[code] = loc_parts[:2]
                        except (ValueError, AttributeError):
                            pass
            except Exception as e:
                warnings.warn(f"Could not load LOCs from {labels_file}: {e}")
    
    return {
        'patient_id': patient_dir.name,
        'patient_dir': str(patient_dir),
        'LOCs': locs,
        'summary_file': str(summary_file) if summary_file.exists() else None,
    }


def extract_feature_values_from_mat(
    patient_dir: Union[str, Path],
    feature: str,
    locs: Dict[str, List[int]]
) -> Dict[str, float]:
    """
    Load .mat file, extract feature values for given LOCs, then free memory.
    
    Args:
        patient_dir: Path to patient directory
        feature: Feature name (e.g., 'flowPerHeartCycle_val', 'PI_val')
        locs: Dictionary of vessel LOCs {vessel_code: [segment_id, centerline_idx]}
        
    Returns:
        Dictionary {vessel_code: feature_value}
    """
    # Load .mat file temporarily
    qvt_data = load_qvt_data(patient_dir, minimal=False)
    data_struct = qvt_data['data_struct']
    # Extract feature values
    vessel_values = {}
    for vessel_code, loc in locs.items():
        value = extract_feature_from_loc(data_struct, loc, feature)
        if value is not None:
            vessel_values[vessel_code] = value
    
    # Free memory
    del data_struct
    del qvt_data
    gc.collect()
    
    return vessel_values


def extract_flow_timeseries_from_mat(
    patient_dir: Union[str, Path],
    locs: Dict[str, List[int]]
) -> Dict[str, np.ndarray]:
    """
    Load .mat file, extract flow timeseries for given LOCs, then free memory.
    
    Args:
        patient_dir: Path to patient directory
        locs: Dictionary of vessel LOCs {vessel_code: [segment_id, centerline_idx]}
        
    Returns:
        Dictionary {vessel_code: flow_array}
    """
    # Load .mat file temporarily
    qvt_data = load_qvt_data(patient_dir, minimal=False)
    data_struct = qvt_data['data_struct']

    patient_flow = {}
    
    # Get nframes
    if isinstance(data_struct, dict):
        nframes = int(data_struct.get('nframes', 15))
    else:
        nframes = int(getattr(data_struct, 'nframes', 15))
    
    for vessel_code, loc in locs.items():
        if len(loc) < 2:
            continue
        
        segment_id, centerline_idx = int(loc[0]), int(loc[1])
        
        # Find matching row in branchList
        if isinstance(data_struct, dict):
            branch_list = data_struct.get('branchList', None)
        else:
            branch_list = getattr(data_struct, 'branchList', None)
        
        if branch_list is None:
            continue
        
        if isinstance(branch_list, np.ndarray) and branch_list.ndim == 2 and branch_list.shape[1] >= 5:
            matches = np.where(
                (branch_list[:, 3] == segment_id) & 
                (branch_list[:, 4] == centerline_idx)
            )[0]
            if len(matches) == 0:
                continue
            row_idx = int(matches[0])
        else:
            continue
        
        # Extract flowPulsatile_val
        if isinstance(data_struct, dict):
            flow_pulsatile = data_struct.get('flowPulsatile_val', None)
        else:
            flow_pulsatile = getattr(data_struct, 'flowPulsatile_val', None)
        if flow_pulsatile is not None and isinstance(flow_pulsatile, np.ndarray):
            if flow_pulsatile.ndim == 2 and row_idx < flow_pulsatile.shape[0]:
                flow_array = flow_pulsatile[row_idx, :].copy()  # Copy to avoid reference
                # Handle complex numbers
                if np.iscomplexobj(flow_array):
                    flow_array = np.abs(flow_array)
                # Ensure correct length
                if len(flow_array) == nframes:
                    print(f"Flow array: {flow_array}")
                    patient_flow[vessel_code] = flow_array.astype(float)
    
    # Free memory
    del data_struct
    del qvt_data
    gc.collect()
    
    return patient_flow


def extract_crosssection_data_from_mat(
    patient_dir: Union[str, Path],
    locs: Dict[str, List[int]],
    cross_section_type: str = 'MAG'
) -> Dict[str, Dict]:
    """
    Load .mat file, extract cross-section data for given LOCs, then free memory.
    Only loads if cross-section plotting is needed.
    
    Args:
        patient_dir: Path to patient directory
        locs: Dictionary of vessel LOCs {vessel_code: [segment_id, centerline_idx]}
        cross_section_type: Type of cross-section ('MAG', 'CD', 'VEL', 'combined')
        
    Returns:
        Dictionary {vessel_code: {'mask': array, 'mag': array, 'cd': array, 'vel': array}}
    """
    # Load .mat file temporarily
    qvt_data = load_qvt_data(patient_dir, minimal=False)
    data_struct = qvt_data['data_struct']
    
    # Extract cross-section data
    crosssection_data = {}
    
    if isinstance(data_struct, dict):
        segment_full = data_struct.get('segmentFull', None)
        mag_data = data_struct.get('MAGcrossection', None)
        cd_data = data_struct.get('timeMIPcrossection', None)
        vel_data = data_struct.get('vTimeFrameave', None)
    else:
        segment_full = getattr(data_struct, 'segmentFull', None)
        mag_data = getattr(data_struct, 'MAGcrossection', None)
        cd_data = getattr(data_struct, 'timeMIPcrossection', None)
        vel_data = getattr(data_struct, 'vTimeFrameave', None)
    
    if segment_full is None:
        del data_struct
        del qvt_data
        gc.collect()
        return crosssection_data
    
    imdim = int(np.sqrt(segment_full.shape[1]))
    
    for vessel_code, loc in locs.items():
        row_index = _find_loc_row_index(data_struct, loc)
        if row_index is None or row_index >= segment_full.shape[0]:
            continue
        
        # Extract only what's needed
        vessel_data = {
            'mask': segment_full[row_index, :].reshape(imdim, imdim).copy(),
        }
        
        if cross_section_type in ['MAG', 'combined'] and mag_data is not None:
            vessel_data['mag'] = mag_data[row_index, :].reshape(imdim, imdim).copy()
        if cross_section_type in ['CD', 'combined'] and cd_data is not None:
            vessel_data['cd'] = cd_data[row_index, :].reshape(imdim, imdim).copy()
        if cross_section_type in ['VEL', 'combined'] and vel_data is not None:
            vessel_data['vel'] = vel_data[row_index, :].reshape(imdim, imdim).copy()
        
        crosssection_data[vessel_code] = vessel_data
    
    # Free memory
    del data_struct
    del qvt_data
    gc.collect()
    
    return crosssection_data


def load_qvt_data(patient_dir: Union[str, Path], minimal: bool = False) -> Dict:
    """
    Load QVT+ data from a patient directory.
    
    Args:
        patient_dir: Path to patient output directory containing qvtData_ISOfix_*.mat
        minimal: If True, extract only minimal necessary fields to save memory
        
    Returns:
        Dictionary containing:
            - data_struct: Main data structure from .mat file (as dict-like object, or minimal if minimal=True)
            - LOCs: Dictionary of vessel LOCs {vessel_code: [segment_id, centerline_idx]}
            - patient_id: Patient ID (from directory name)
    """
    patient_dir = Path(patient_dir)
    
    # Find the most recent qvtData_ISOfix_*.mat file
    mat_files = list((patient_dir / 'centerline_test').glob('qvtData_ISOfix_*.mat'))
    if not mat_files:
        raise FileNotFoundError(f"No qvtData_ISOfix_*.mat file found in {patient_dir}")

    
    # Get most recent file
    mat_file = max(mat_files, key=lambda p: p.stat().st_mtime)
    
    # Check if it's a v7.3 file (HDF5 format)
    is_v73 = False
    if HAS_H5PY:
        try:
            with h5py.File(mat_file, 'r') as test_f:
                is_v73 = True
        except (OSError, IOError):
            is_v73 = False
        except Exception:
            is_v73 = False
    
    # Load .mat file
    data_struct = None
    if is_v73:
        # Use h5py for v7.3 files
        if not HAS_H5PY:
            raise ImportError(
                f"MATLAB v7.3 file detected but h5py is not installed. "
                f"Please install h5py: pip install h5py"
            )
        try:
            mat_data = _load_mat_v73(mat_file)
            data_struct = mat_data.get('data_struct', mat_data)
        except Exception as e:
            raise ValueError(
                f"Failed to load MATLAB v7.3 file {mat_file}: {e}\n"
                f"Please ensure h5py is installed: pip install h5py"
            )
    else:
        raise ValueError(f"Unsupported MATLAB file format: {mat_file}")
        pass  # TODO: Implement loading of older MATLAB files
    
    # Load LOCs from SummaryParamTool.xls (preferred source)
    locs = {}
    summary_file = patient_dir / 'centerline_test' / 'SummaryParamTool.xls'
    if summary_file.exists():
        try:
            # Try to read the Summary_Centerline sheet
            summary_df = pd.read_excel(summary_file, sheet_name='Summary_Centerline')
            for _, row in summary_df.iterrows():
                vessel_label = row.get('Vessel Label', '')
                branch_num = row.get('Branch Number', np.nan)
                centerline = row.get('Centerline', np.nan)
                if pd.notna(branch_num) and pd.notna(centerline):
                    # Map vessel label to code using flexible matching
                    match = _match_vessel_label(vessel_label)
                    if match:
                        code, _ = match
                        locs[code] = [int(branch_num), int(centerline)]
        except Exception as e:
            warnings.warn(f"Could not load LOCs from {summary_file}: {e}")
    
    # Fallback to LabelsQVT.csv if LOCs not found
    if not locs:
        labels_file = patient_dir / 'centerline_test' / 'LabelsQVT.csv'
        if labels_file.exists():
            try:
                labels_df = pd.read_csv(labels_file)
                for _, row in labels_df.iterrows():
                    artery = row.get('Artery', '')
                    loc_str = row.get('Loc', '')
                    if pd.notna(loc_str) and loc_str.strip() and loc_str != ' ':
                        # Parse LOC string like "[123, 45]" or "123"
                        loc_str = str(loc_str).strip()
                        if loc_str.startswith('[') and loc_str.endswith(']'):
                            loc_str = loc_str[1:-1]
                        try:
                            loc_parts = [int(x.strip()) for x in loc_str.split(',')]
                            if len(loc_parts) >= 2:
                                # Map artery name to code using flexible matching
                                match = _match_vessel_label(artery)
                                if match:
                                    code, _ = match
                                    locs[code] = loc_parts[:2]
                        except (ValueError, AttributeError) as e:
                            print(f"Error loading LOCs from {labels_file}: {e}")
                            pass
            except Exception as e:
                print(f"Error loading LOCs from {labels_file}: {e}")
                warnings.warn(f"Could not load LOCs from {labels_file}: {e}")
    
    return {
        'data_struct': data_struct,
        'LOCs': locs,
        'patient_id': patient_dir.name,
        'patient_dir': str(patient_dir),
        'summary_file': str(summary_file) if summary_file.exists() else None,
    }


def load_summary_data(patient_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load vessel summary data from SummaryParamTool.xls.
    
    Args:
        patient_dir: Path to patient output directory
        
    Returns:
        DataFrame with vessel data from Summary_Centerline sheet
    """
    patient_dir = Path(patient_dir)
    summary_file = patient_dir / 'centerline_test' / 'SummaryParamTool.xls'

    if not summary_file.exists():
        return pd.DataFrame()
    
    try:
        summary_df = pd.read_excel(summary_file, sheet_name='Summary_Centerline')
        return summary_df
    except Exception as e:
        warnings.warn(f"Could not load summary data from {summary_file}: {e}")
        return pd.DataFrame()


def extract_feature_from_loc(
    data_struct: Union[Dict, object],
    loc: List[int],
    feature: str
) -> Optional[float]:
    """
    Extract a feature value from a LOC (Location of Interest).
    
    Args:
        data_struct: Data structure from QVT+ .mat file (dict or struct-like)
        loc: [segment_id, centerline_idx]
        feature: Feature name (e.g., 'flowPerHeartCycle_val', 'PI_val', 'RI_val')
        
    Returns:
        Feature value or None if not found
    """
    if len(loc) < 2:
        return None
    
    segment_id, centerline_idx = int(loc[0]), int(loc[1])
    
    # Get branchList (handle both dict and struct access)
    if isinstance(data_struct, dict):
        branch_list = data_struct.get('branchList', None)
    else:
        branch_list = getattr(data_struct, 'branchList', None)
    
    if branch_list is None:
        return None
    
    # branchList is [x, y, z, segment_id, centerline_idx, ...]
    # Find row where segment_id and centerline_idx match
    if isinstance(branch_list, np.ndarray):
        # Ensure correct indexing
        if branch_list.ndim == 2 and branch_list.shape[1] >= 5:
            matches = np.where(
                (branch_list[:, 3] == segment_id) & 
                (branch_list[:, 4] == centerline_idx)
            )[0]
            if len(matches) == 0:
                return None
            row_idx = int(matches[0])
        else:
            return None
    else:
        # Handle struct array or other formats
        return None
    
    # Extract feature value (handle both dict and struct access)
    if isinstance(data_struct, dict):
        feature_data = data_struct.get(feature, None)
    else:
        feature_data = getattr(data_struct, feature, None)
    
    if feature_data is None:
        return None
    
    if isinstance(feature_data, np.ndarray):
        if feature_data.ndim == 1 and row_idx < len(feature_data):
            value = feature_data[row_idx]
            # Handle complex numbers
            if np.iscomplexobj(value):
                value = np.abs(value)
            return float(value) if np.isfinite(value) else None
        elif feature_data.ndim == 2 and row_idx < feature_data.shape[0]:
            # For 2D arrays, return first column or mean
            value = feature_data[row_idx, 0] if feature_data.shape[1] > 0 else None
            if value is not None:
                if np.iscomplexobj(value):
                    value = np.abs(value)
                return float(value) if np.isfinite(value) else None
    
    return None


def load_patient_summary(
    results_dir: Union[str, Path],
    summary_xlsx: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load patient summary data (patient_id, mri_id, sex, age_at_mri).
    
    Args:
        results_dir: Directory containing patient subdirectories
        summary_xlsx: Path to summary Excel file. If None, searches for *.xlsx in results_dir.
        
    Returns:
        DataFrame with columns: patient_id, mri_id, sex, age_at_mri
    """
    results_dir = Path(results_dir)
    
    if summary_xlsx is None:
        # Search for .xlsx files in results_dir
        xlsx_files = list((results_dir / 'centerline_test').glob('*.xlsx'))
        if not xlsx_files:
            raise FileNotFoundError(f"No .xlsx file found in {results_dir}")
        summary_xlsx = xlsx_files[0]
    else:
        summary_xlsx = Path(summary_xlsx)
    
    if not summary_xlsx.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_xlsx}")
    
    # Try to read the Excel file
    try:
        # Try first sheet
        df = pd.read_excel(summary_xlsx, sheet_name=0)
        
        # Normalize column names (case-insensitive, handle spaces/underscores)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Map common column name variations
        col_mapping = {}
        for col in df.columns:
            if 'patient' in col or 'subject' in col:
                col_mapping[col] = 'patient_id'
            elif 'mri' in col and 'id' in col:
                col_mapping[col] = 'mri_id'
            elif col in ['sex', 'gender']:
                col_mapping[col] = 'sex'
            elif 'age' in col:
                col_mapping[col] = 'age_at_mri'
        
        df = df.rename(columns=col_mapping)
        
        # Ensure required columns exist
        required_cols = ['patient_id']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error reading summary file {summary_xlsx}: {e}")


def plot_violin(
    data: pd.DataFrame,
    feature: str,
    output_path: Optional[Union[str, Path]] = None,
    style: Optional[ViolinStyle] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_patient_dots: bool = False,
    remove_outliers: bool = False,
    z_score_threshold: float = 6.0,
) -> plt.Figure:
    """
    Create violin plots for a specific feature, grouped by vessel groups.
    
    Args:
        data: DataFrame with columns: 'vessel', 'group', feature, and optionally 'patient_id'
        feature: Feature name to plot (e.g., 'flowPerHeartCycle_val', 'PI_val')
        output_path: Optional path to save figure
        style: Optional ViolinStyle configuration
        figsize: Optional figure size (overrides style)
        show_patient_dots: If True, overlay individual patient dots connected by lines
        remove_outliers: If True, remove outliers based on z-score before plotting
        z_score_threshold: Z-score threshold for outlier removal (default: 3.0, i.e., |z| > 3)
        
    Returns:
        matplotlib Figure
    """
    if style is None:
        style = ViolinStyle()
    
    if figsize is None:
        figsize = style.figsize
    
    # Filter data to remove NaN values for this feature
    plot_data = data.dropna(subset=[feature]).copy()
    
    if plot_data.empty:
        raise ValueError(f"No valid data for feature '{feature}'")
    
    # Calculate outliers VESSEL-WISE (per vessel, not global)
    # For each vessel, find outliers, then mark ALL points from those patients as outliers
    outlier_patients = set()
    outlier_indices = []
    
    if 'patient_id' in plot_data.columns:
        # Group by vessel and calculate outliers per vessel
        for vessel in plot_data['vessel'].unique():
            vessel_data = plot_data[plot_data['vessel'] == vessel]
            vessel_feature_values = vessel_data[feature].values
            
            if len(vessel_feature_values) > 0:
                vessel_mean = np.mean(vessel_feature_values)
                vessel_std = np.std(vessel_feature_values)
                
                if vessel_std > 0:
                    # Calculate z-scores for this vessel
                    vessel_z_scores = np.abs((vessel_feature_values - vessel_mean) / vessel_std)
                    # Find outliers for this vessel
                    vessel_outlier_mask = vessel_z_scores > z_score_threshold
                    vessel_outlier_indices = vessel_data.index[vessel_outlier_mask].tolist()
                    
                    # Track which patients have outliers for this vessel
                    if len(vessel_outlier_indices) > 0:
                        vessel_outlier_patients = set(vessel_data.loc[vessel_outlier_indices, 'patient_id'].unique())
                        outlier_patients.update(vessel_outlier_patients)
                        outlier_indices.extend(vessel_outlier_indices)
    
    # Store original outlier information (before removal)
    original_outlier_indices = outlier_indices.copy()
    original_outlier_patients = outlier_patients.copy()
    
    # Mark ALL points from outlier patients as outliers (not just the specific vessel point)
    plot_data['_is_outlier'] = False
    if len(outlier_patients) > 0 and 'patient_id' in plot_data.columns:
        # Mark all points from patients that have outliers in any vessel
        outlier_patient_mask = plot_data['patient_id'].isin(outlier_patients)
        plot_data.loc[outlier_patient_mask, '_is_outlier'] = True
    
    # Remove outliers based on z-score if requested
    # Remove ALL points from patients that have outliers in any vessel
    if remove_outliers:
        if len(outlier_patients) > 0 and 'patient_id' in plot_data.columns:
            # Remove all points from outlier patients
            outlier_patient_mask = plot_data['patient_id'].isin(outlier_patients)
            plot_data = plot_data[~outlier_patient_mask].copy()


    print(f'Outlier removal (|z| > {z_score_threshold}):')
    print(f'  Total patients: {len(plot_data["patient_id"].unique())}')
    print(f'  Total vessels: {len(plot_data["vessel"].unique())}')
    print(f'  Total points: {len(plot_data)}')
    print(f'  Total outliers (Patients): {len(outlier_patients)}')
    print(f'  Total outliers (Points): {len(outlier_indices)}')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set palette
    if style.palette is None:
        palette = sns.color_palette("Set2", n_colors=len(_VESSEL_GROUPING))
    else:
        palette = style.palette

    # Drop COMMs from plot_data 
    plot_data = plot_data[plot_data['vessel'] != 'Left Communicating']
    plot_data = plot_data[plot_data['vessel'] != 'Right Communicating']

    # Reorder plot_data based on _VIOLIN_ORDER
    plot_data['vessel'] = pd.Categorical(plot_data['vessel'], categories=_VIOLIN_ORDER, ordered=True)
    plot_data = plot_data.sort_values('vessel')

    # _is_outlier column is already set above before potential removal
    
    # Set the values as the absolute value
    plot_data[feature] = np.abs(plot_data[feature])
    
    # Create violin plot
    violin_plot = sns.violinplot(
        data=plot_data,
        x='vessel',
        y=feature,
        hue='group',
        inner=style.inner,
        scale=style.scale,
        palette=palette,
        alpha=style.alpha,
        linewidth=style.linewidth,
        ax=ax,
        dodge=style.dodge,
    )
    
    # Customize plot
    ax.set_xlabel('Vessel', fontsize=12)
    # Format ylabel based on feature name
    if 'flow' in feature.lower() or 'Flow' in feature:
        ylabel = 'Volumetric Flow Rate (mL/min)'
    elif 'PI' in feature or 'pulsatility' in feature.lower():
        ylabel = 'Pulsatility Index'
    else:
        ylabel = feature.replace('_', ' ').title()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, None)
    ax.set_title(f'Violin Plot: {ylabel} by Vessel Group', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    
    # Get vessel group legend
    vessel_group_legend = ax.legend(title='Vessel Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add patient dots and connections if requested
    if show_patient_dots and 'patient_id' in plot_data.columns:
        # Get unique patients
        all_unique_patients = plot_data['patient_id'].unique()
        
        if remove_outliers:
            # When outliers are removed, show dots for ALL remaining patients
            # (outlier patients are already removed, so we show all that remain)
            unique_patients = list(all_unique_patients)
        else:
            # When outliers are NOT removed, only show patients with outliers in legend
            # Use original_outlier_patients (before removal) to know which patients had outliers
            if original_outlier_patients:
                # Show all patients that have outliers
                unique_patients = [pid for pid in all_unique_patients if pid in original_outlier_patients]
            else:
                unique_patients = []  # No outliers, so no patient legend needed
        
        n_patients = len(unique_patients) if unique_patients else 0
        
        # Create color palette for patients
        if n_patients > 0:
            patient_colors = sns.color_palette("husl", n_colors=n_patients)
            patient_color_map = dict(zip(unique_patients, patient_colors))
        else:
            patient_color_map = {}
        
        # Get x-positions for each vessel
        # Extract positions from the violin plot
        vessel_positions = {}
        
        # Get tick positions and labels
        tick_positions = ax.get_xticks()
        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
        
        # When dodge=True, violins are offset by group
        # We need to find the center of all violins for each vessel
        # Get all collections (violin shapes) from the plot
        collections = ax.collections
        
        # Group collections by vessel
        for vessel in _VIOLIN_ORDER:
            if vessel in tick_labels:
                idx = tick_labels.index(vessel)
                if idx < len(tick_positions):
                    base_x = tick_positions[idx]
                    
                    # Find all collections near this position
                    vessel_x_positions = []
                    for collection in collections:
                        try:
                            # Get the x-offset of the collection
                            offsets = collection.get_offsets()
                            if len(offsets) > 0:
                                # Collections in violin plots have offsets relative to base position
                                x_pos = offsets[:, 0].mean() if offsets.ndim > 1 else offsets[0]
                                # Check if this collection is for this vessel (within reasonable range)
                                if abs(x_pos - base_x) < 0.6:  # Adjust threshold based on dodge width
                                    vessel_x_positions.append(x_pos)
                        except:
                            continue
                    
                    # Use average position if found, otherwise use tick position
                    if vessel_x_positions:
                        vessel_positions[vessel] = np.mean(vessel_x_positions)
                    else:
                        # Fallback: use tick position
                        vessel_positions[vessel] = base_x
        
        # For each patient, plot dots and connect them
        for patient_id in unique_patients:
            patient_data = plot_data[plot_data['patient_id'] == patient_id].copy()
            patient_data = patient_data.sort_values('vessel')
            
            # Get x and y coordinates for this patient
            x_coords = []
            y_coords = []
            vessels_with_data = []
            outlier_mask = []
            
            for vessel in _VIOLIN_ORDER:
                vessel_data = patient_data[patient_data['vessel'] == vessel]
                if not vessel_data.empty and vessel in vessel_positions:
                    x_coords.append(vessel_positions[vessel])
                    y_coords.append(vessel_data[feature].iloc[0])
                    vessels_with_data.append(vessel)
                    # Track if this value is an outlier
                    if '_is_outlier' in vessel_data.columns:
                        outlier_mask.append(vessel_data['_is_outlier'].iloc[0])
                    else:
                        outlier_mask.append(False)
            
            if len(x_coords) > 0:
                color = patient_color_map[patient_id]
                
                # Connect dots with lines
                if len(x_coords) > 1:
                    ax.plot(x_coords, y_coords, color=color, alpha=0.4, linewidth=1, linestyle='--')
                
                # Plot dots - separate normal and outlier points
                x_coords_normal = [x for i, x in enumerate(x_coords) if not outlier_mask[i]]
                y_coords_normal = [y for i, y in enumerate(y_coords) if not outlier_mask[i]]
                x_coords_outlier = [x for i, x in enumerate(x_coords) if outlier_mask[i]]
                y_coords_outlier = [y for i, y in enumerate(y_coords) if outlier_mask[i]]
                
                # Plot normal points (normal styling)
                if x_coords_normal:
                    ax.scatter(x_coords_normal, y_coords_normal, color=color, s=50, alpha=0.7, 
                              edgecolors='black', linewidths=0.5, zorder=10)
                
                # Plot outlier points (highlighted with thicker edge and different marker)
                if x_coords_outlier:
                    ax.scatter(x_coords_outlier, y_coords_outlier, color=color, s=70, alpha=0.8,
                              edgecolors='red', linewidths=2.0, marker='s', zorder=11)
        
        # Create patient legend
        if n_patients > 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=patient_color_map[pid], 
                       markersize=8, markeredgecolor='black', markeredgewidth=0.5, label=pid)
                for pid in unique_patients
            ]
            
            # Add legend entry for outlier points (if any and outliers not removed)
            has_outliers = len(original_outlier_indices) > 0 and not remove_outliers
            if has_outliers:
                legend_elements.append(
                    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                           markersize=8, markeredgecolor='red', markeredgewidth=2.0, 
                           label='Outlier', linestyle='None')
                )
            
            # Set legend title based on whether outliers are removed
            if remove_outliers:
                legend_title = 'Patients'
                # If outliers are removed, and there is more than 20 patients, do not show the legend
                if len(unique_patients) > 20:
                    legend_title = None
                    legend_elements, unique_patients = [], []                    
            else:
                legend_title = 'Patients with Outliers'
            
            patient_legend = ax.legend(legend_elements, 
                                       [pid for pid in unique_patients] + (['Outlier'] if has_outliers else []),
                                       title=legend_title, bbox_to_anchor=(1.05, 0.5), loc='center left',
                                       fontsize=8, framealpha=0.9)
            
            # Add back the vessel group legend
            ax.add_artist(vessel_group_legend)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    patient_annotations: Optional[pd.DataFrame] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Patient Correlation Matrix (Flow + PI)",
    color_palette: Optional[List[str]] = None,
    fontsize: int = 10,
) -> plt.Figure:
    """
    Create a correlation heatmap with hierarchical clustering similar to R's pheatmap.
    Uses seaborn.clustermap for automatic clustering and annotations.
    
    Args:
        correlation_matrix: DataFrame with correlation values (patients x patients)
        patient_annotations: Optional DataFrame with patient metadata (age, sex)
        output_path: Optional path to save figure
        figsize: Optional figure size
        title: Plot title
        color_palette: Optional color palette for heatmap
        fontsize: Font size for labels
        
    Returns:
        matplotlib Figure
    """
    if figsize is None:
        # Scale figure size based on number of patients
        n_patients = len(correlation_matrix)
        base_size = max(10, n_patients * 0.4)
        figsize = (base_size, base_size)
    
    # Set color palette for main heatmap
    if color_palette is None:
        # Use a diverging colormap (blue-white-red) similar to pheatmap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
    else:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('custom', color_palette)
    
    # Prepare row and column colors for annotations
    row_colors = None
    col_colors = None
    annotation_colors_dict = {}
    
    if patient_annotations is not None and not patient_annotations.empty:
        # Align annotations with correlation matrix
        annot_df = patient_annotations.reindex(correlation_matrix.index, fill_value=np.nan)
        
        # Prepare color arrays for each annotation column
        row_color_list = []
        col_color_list = []
        
        for col in annot_df.columns:
            if col == 'mri_id':
                continue  # Skip mri_id as it's not useful for visualization
            
            # Determine if numeric or categorical
            if annot_df[col].dtype in ['int64', 'float64']:
                # Numeric: use continuous colormap (e.g., for age)
                # Normalize to 0-1 range
                values = annot_df[col].dropna()
                if len(values) > 0:
                    vmin, vmax = values.min(), values.max()
                    if vmax > vmin:
                        normalized = (annot_df[col] - vmin) / (vmax - vmin)
                    else:
                        normalized = pd.Series(0.5, index=annot_df.index)
                    # Use a colormap (e.g., greens for age)
                    from matplotlib.colors import Normalize
                    try:
                        from matplotlib import colormaps
                        cmap_age = colormaps['Greens']
                    except (ImportError, AttributeError):
                        # Fallback for older matplotlib
                        cmap_age = plt.cm.get_cmap('Greens')
                    
                    # Convert to RGB colors (seaborn prefers RGB over RGBA)
                    colors_list = []
                    for idx in annot_df.index:
                        val = normalized.loc[idx]
                        if pd.notna(val):
                            rgba = cmap_age(val)
                            # Convert RGBA to RGB (take first 3 values)
                            if isinstance(rgba, (list, tuple, np.ndarray)):
                                rgb = tuple(rgba[:3])  # Take first 3 (RGB)
                            else:
                                rgb = (1.0, 1.0, 1.0)  # Fallback
                            colors_list.append(rgb)
                        else:
                            colors_list.append((1.0, 1.0, 1.0))  # White for missing (RGB)
                    
                    row_color_list.append(colors_list)
                    col_color_list.append(colors_list)
                    annotation_colors_dict[col] = {'type': 'continuous', 'cmap': 'Greens', 'vmin': vmin, 'vmax': vmax}
            else:
                # Categorical: use discrete colors (e.g., for sex)
                unique_vals = annot_df[col].dropna().unique()
                n_unique = len(unique_vals)
                
                if n_unique <= 10:
                    # Map categories to colors
                    if col.lower() in ['sex', 'gender', 'deqsex']:
                        # Use blue for Male, red for Female
                        color_map = {}
                        for val in unique_vals:
                            val_str = str(val).lower().strip()
                            # Check for female first (more specific, contains 'm')
                            if 'female' in val_str or (val_str.startswith('f') and len(val_str) <= 2):
                                color_map[val] = (1.0, 0.0, 0.0, 1.0)  # Red
                            elif 'male' in val_str or (val_str.startswith('m') and len(val_str) <= 2):
                                color_map[val] = (0.0, 0.0, 1.0, 1.0)  # Blue
                            else:
                                color_map[val] = (0.5, 0.5, 0.5, 1.0)  # Gray for unknown
                    else:
                        # Use Set3 palette for other categorical variables
                        colors_palette = sns.color_palette("Set3", n_colors=n_unique)
                        # Convert seaborn colors to tuples (they're already in 0-1 range)
                        color_map = {}
                        for val, color in zip(unique_vals, colors_palette):
                            # Ensure color is a tuple with 3 values (RGB)
                            if isinstance(color, (list, tuple)):
                                color_map[val] = tuple(color[:3])  # Take first 3 (RGB)
                            else:
                                color_map[val] = (1.0, 1.0, 1.0)  # Fallback
                    
                    # Create color array - handle missing values properly
                    # Map values first, then convert to list handling NaN
                    mapped_colors = annot_df[col].map(color_map)
                    # Convert to list of tuples, handling NaN values
                    colors_list = []
                    for idx in annot_df.index:
                        val = mapped_colors.loc[idx]
                        if pd.notna(val):
                            # Ensure it's a proper RGBA tuple (3 or 4 values, 0-1 range)
                            if isinstance(val, (list, tuple, np.ndarray)):
                                # Convert to tuple and ensure proper format
                                val_list = list(val)
                                # Ensure we have 3 or 4 values
                                if len(val_list) == 3:
                                    colors_list.append(tuple(val_list))  # RGB
                                elif len(val_list) >= 4:
                                    colors_list.append(tuple(val_list[:4]))  # RGBA
                                else:
                                    # Pad to RGB if needed
                                    while len(val_list) < 3:
                                        val_list.append(1.0)
                                    colors_list.append(tuple(val_list[:3]))
                            else:
                                colors_list.append((1.0, 1.0, 1.0))  # Fallback to white RGB
                        else:
                            colors_list.append((1.0, 1.0, 1.0))  # White for missing (RGB)
                    
                    row_color_list.append(colors_list)
                    col_color_list.append(colors_list)
                    annotation_colors_dict[col] = {'type': 'categorical', 'colors': color_map}
        
        # Combine all annotation colors into arrays
        # seaborn clustermap expects row_colors/col_colors as a list of lists
        # where each inner list represents one annotation column
        # The structure should be: [[color1_patient1, color1_patient2, ...], [color2_patient1, ...]]
        if row_color_list:
            # row_color_list is already in the correct format: each element is a list of colors for one annotation
            # We need to keep it as a list of lists (not convert to numpy array)
            row_colors = row_color_list
            col_colors = col_color_list
    
    # Create clustermap
    g = sns.clustermap(
        correlation_matrix,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        row_cluster=True,
        col_cluster=True,
        method='complete',  # Linkage method
        metric='euclidean',  # Distance metric
        figsize=figsize,
        row_colors=row_colors,
        col_colors=col_colors,
        xticklabels=True,
        yticklabels=True,
        fmt='.2f',
        annot=False,
    )
    
    # Set title
    g.fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust tick labels
    for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
        if ax is not None:
            ax.set_visible(True)
    
    # Set font size for labels
    g.ax_heatmap.tick_params(labelsize=fontsize)
    g.ax_heatmap.set_xlabel('Patient', fontsize=fontsize)
    g.ax_heatmap.set_ylabel('Patient', fontsize=fontsize)
    
    # Create legend for annotations if provided
    if annotation_colors_dict:
        legend_elements = []
        legend_labels = []
        
        for col, color_info in annotation_colors_dict.items():
            if color_info['type'] == 'categorical':
                for val, color in color_info['colors'].items():
                    legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black'))
                    legend_labels.append(f"{col.replace('_', ' ').title()}: {val}")
            elif color_info['type'] == 'continuous':
                # Add a colorbar for continuous variables
                from matplotlib.colors import Normalize
                from matplotlib.cm import ScalarMappable
                try:
                    from matplotlib import colormaps
                    cmap_cont = colormaps[color_info['cmap']]
                except (ImportError, AttributeError):
                    cmap_cont = plt.cm.get_cmap(color_info['cmap'])
                
                sm = ScalarMappable(
                    cmap=cmap_cont,
                    norm=Normalize(vmin=color_info['vmin'], vmax=color_info['vmax'])
                )
                sm.set_array([])
                # Add colorbar to the figure
                cbar = g.fig.colorbar(sm, ax=g.ax_heatmap, orientation='vertical', 
                                     pad=0.05, shrink=0.6, aspect=20)
                cbar.set_label(col.replace('_', ' ').title(), rotation=270, labelpad=20, fontsize=fontsize-2)
        
        if legend_elements:
            # Add legend for categorical variables
            g.fig.legend(legend_elements, legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                        fontsize=fontsize-2, title='Annotations', frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for title
    
    if output_path:
        g.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return g.fig


# Polar plot vessel layout configuration
# Defines the angular position and ring (radius) for each vessel in the polar plot
_POLAR_VESSEL_LAYOUT = {
    # Center point
    'TCBF': {'angle': 0, 'ring': 0, 'abbrev': 'TCBF'},
    
    # Ring 1: ICAs (innermost, mirrored)
    'Left ICA': {'angle': 90, 'ring': 1, 'abbrev': 'LICA'},
    'Right ICA': {'angle': -90, 'ring': 1, 'abbrev': 'RICA'},
    
    # Ring 2: Basilar (center bottom)
    'Basilar': {'angle': 180, 'ring': 2, 'abbrev': 'BASI'},
    
    # Ring 3: Communicating vessels only
    # 'Communicating': {'angle': 180, 'ring': 3, 'abbrev': 'COMM'},  # Bottom center
    'Right Communicating': {'angle': -112.5, 'ring': 3, 'abbrev': 'RCOMM'},  # Right side
    'Left Communicating': {'angle': 112.5, 'ring': 3, 'abbrev': 'LCOMM'},  # Left side
    
    # Ring 4: ACAs (top), MCAs (middle), PCAs (bottom)
    'Left MCA': {'angle': 90, 'ring': 4, 'abbrev': 'LMCA'},  # Top left
    'Right MCA': {'angle': -90, 'ring': 4, 'abbrev': 'RMCA'},  # Top right
    'Left ACA': {'angle': 45, 'ring': 4, 'abbrev': 'LACA'},  # Upper middle left
    'Right ACA': {'angle': -45, 'ring': 4, 'abbrev': 'RACA'},  # Upper middle right
    'Left PCA': {'angle': 135, 'ring': 4, 'abbrev': 'LPCA'},  # Bottom left
    'Right PCA': {'angle': -135, 'ring': 4, 'abbrev': 'RPCA'},  # Bottom right
    
    # Ring 5: Straight Sinus - at bottom
    'Straight Sinus': {'angle': 180, 'ring': 5, 'abbrev': 'STRV'},
    
    # Ring 6: Venous system - at bottom
    'Sagital Sinus': {'angle': 180, 'ring': 6, 'abbrev': 'SSSV'},  # Bottom center
    'Right Transverse': {'angle': -120, 'ring': 6, 'abbrev': 'RTSV'},  # Bottom left
    'Left Transverse': {'angle': 120, 'ring': 6, 'abbrev': 'LTSV'},  # Bottom right
}

# Ring radii (normalized, from 0 to 1)
_POLAR_RING_RADII = {
    0: 0.05,   # Center (TCBF) 
    1: 0.15,   # ICAs 
    2: 0.25,   # Basilar 
    3: 0.35,   # Communicating vessels only
    4: 0.45,   # ACAs (top), MCAs (middle), PCAs (bottom)
    5: 0.65,   # Straight Sinus
    6: 0.85,   # Venous system
}


def _calculate_polar_boundaries(vessel_name: str, ring: int, angle_deg: float) -> Tuple[float, float]:
    """
    Calculate angular boundaries (theta_start, theta_end) for a vessel in a polar plot.
    
    Args:
        vessel_name: Name of the vessel
        ring: Ring number
        angle_deg: Center angle in degrees
        
    Returns:
        Tuple of (theta_start, theta_end) in radians
    """
    # Special handling for ring 5 (STRV) - should cover same space as SSSV
    if ring == 5:
        # STRV should cover same angular width as SSSV (120 degrees)
        angular_width = 60 
        angle_rad = np.deg2rad(angle_deg)
        theta_start = angle_rad - np.deg2rad(angular_width / 2)
        theta_end = angle_rad + np.deg2rad(angular_width / 2)
        return theta_start, theta_end
    
    # Special handling for ring 3 (Communicating vessels)
    if ring == 3 and vessel_name in ['Communicating', 'Right Communicating', 'Left Communicating']:
        # Get all Communicating vessels in this ring
        comm_vessels = [(v, l['angle']) for v, l in _POLAR_VESSEL_LAYOUT.items() 
                       if l['ring'] == ring]
        if len(comm_vessels) == 1:
            # Single vessel gets a reasonable angular width
            angular_width = 25
            angle_rad = np.deg2rad(angle_deg)
            theta_start = angle_rad - np.deg2rad(angular_width / 2)
            theta_end = angle_rad + np.deg2rad(angular_width / 2)
            return theta_start, theta_end
        else:
            # Each vessel with a small angl
            angular_width = 25
            angle_rad = np.deg2rad(angle_deg)
            theta_start = angle_rad - np.deg2rad(angular_width / 2)
            theta_end = angle_rad + np.deg2rad(angular_width / 2)
            return theta_start, theta_end
    
    # Get all vessels in this ring with their angles
    vessels_in_ring = [(v, l['angle']) for v, l in _POLAR_VESSEL_LAYOUT.items() 
                      if l['ring'] == ring]
    vessels_in_ring.sort(key=lambda x: x[1])  # Sort by angle
    
    # Find this vessel's position in the sorted list
    vessel_idx = None
    for idx, (v, a) in enumerate(vessels_in_ring):
        if v == vessel_name:
            vessel_idx = idx
            break
    
    if vessel_idx is None or len(vessels_in_ring) == 0:
        # Fallback to equal division
        angular_width = 360 / len(vessels_in_ring) if vessels_in_ring else 30
        angle_rad = np.deg2rad(angle_deg)
        theta_start = angle_rad - np.deg2rad(angular_width / 2)
        theta_end = angle_rad + np.deg2rad(angular_width / 2)
        return theta_start, theta_end
    
    # Calculate boundaries based on adjacent vessels
    n_vessels = len(vessels_in_ring)
    
    # Special handling for single-vessel rings (e.g., Basilar in Ring 2)
    if n_vessels == 1:
        # Single vessel gets a reasonable angular width (e.g., 60 degrees)
        angular_width = 60
        angle_rad = np.deg2rad(angle_deg)
        theta_start = angle_rad - np.deg2rad(angular_width / 2)
        theta_end = angle_rad + np.deg2rad(angular_width / 2)
        return theta_start, theta_end
    
    # Get previous and next vessel angles
    prev_angle = vessels_in_ring[(vessel_idx - 1) % n_vessels][1]
    next_angle = vessels_in_ring[(vessel_idx + 1) % n_vessels][1]
    
    # Handle wrap-around (angles can cross 180/-180 boundary)
    # Convert all angles to 0-360 range for easier calculation
    def normalize_angle(angle):
        """Normalize angle to 0-360 range"""
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle
    
    current_angle_norm = normalize_angle(angle_deg)
    prev_angle_norm = normalize_angle(prev_angle)
    next_angle_norm = normalize_angle(next_angle)
    
    # Calculate midpoint between previous and current
    if prev_angle_norm > current_angle_norm:
        # Wrap around case
        midpoint_prev = (prev_angle_norm + current_angle_norm + 360) / 2
        if midpoint_prev >= 360:
            midpoint_prev -= 360
    else:
        midpoint_prev = (prev_angle_norm + current_angle_norm) / 2
    
    # Calculate midpoint between current and next
    if next_angle_norm < current_angle_norm:
        # Wrap around case
        midpoint_next = (current_angle_norm + next_angle_norm + 360) / 2
        if midpoint_next >= 360:
            midpoint_next -= 360
    else:
        midpoint_next = (current_angle_norm + next_angle_norm) / 2
    
    # Convert back to -180 to 180 range for polar plot
    def to_polar_range(angle):
        """Convert 0-360 to -180 to 180 range"""
        if angle > 180:
            return angle - 360
        return angle
    
    theta_start = np.deg2rad(to_polar_range(midpoint_prev))
    theta_end = np.deg2rad(to_polar_range(midpoint_next))
    
    # Ensure theta_end > theta_start (handle wrap-around)
    if theta_end < theta_start:
        theta_end += 2 * np.pi
    
    return theta_start, theta_end


def plot_polar_flow(
    patient_flow_data: Dict[str, float],
    patient_id: str,
    feature_name: str = 'Flow',
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'hot',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_abs: bool = True,
) -> plt.Figure:
    """
    Create a polar plot showing feature values for each vessel section.
    
    Args:
        patient_flow_data: Dictionary {vessel_name: feature_value}
        patient_id: Patient identifier for title
        feature_name: Name of the feature being plotted (e.g., 'Flow', 'PI')
        output_path: Optional path to save figure
        figsize: Optional figure size (default: (10, 10))
        cmap: Colormap name (default: 'hot')
        vmin: Minimum value for color scale (default: min of data, excluding TCBF)
        vmax: Maximum value for color scale (default: max of data, excluding TCBF)
        
    Returns:
        matplotlib Figure
    """
    if figsize is None:
        figsize = (10, 10)
    
    # Create figure with polar projection
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection='polar')
    
    # Get colormap (use a fixed palette for consistent scaling)
    if cmap is None or str(cmap).lower() in ["custom_hot_no_green", "custom"]:
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list(
            "custom_hot_no_green",
            [
                (1, 1, 1),      # White
                (1, 1, 0),      # Light Yellow
                (1, 0.5, 0),    # Orange
                (1, 0, 0),      # Red
            ],
            N=1024,
        )
    else:
        try:
            from matplotlib import colormaps
            color_map = colormaps[cmap]
        except (ImportError, AttributeError):
            color_map = plt.cm.get_cmap(cmap)
    
    # Determine color scale - exclude TCBF from min/max calculation
    flow_values = [v for k, v in patient_flow_data.items() 
                   if v is not None and np.isfinite(v) and k != 'TCBF']
    if not flow_values:
        raise ValueError(f"No valid data for patient {patient_id}")

    if use_abs:
        flow_values = np.abs(flow_values)
        patient_flow_data = {k: np.abs(v) for k, v in patient_flow_data.items()}
    
    if vmin is None:
        vmin = min(flow_values)
    if vmax is None:
        vmax = max(flow_values)
    
    # Get TCBF value for text display
    tcbf_value = patient_flow_data.get('TCBF', None)
    
    # Normalize function
    def normalize(value):
        if value is None or not np.isfinite(value):
            return None
        if vmax == vmin:
            return 0.5
        return (value - vmin) / (vmax - vmin)
    
    # Plot each vessel section
    for vessel_name, flow_value in patient_flow_data.items():
        if vessel_name not in _POLAR_VESSEL_LAYOUT:
            continue
        
        layout = _POLAR_VESSEL_LAYOUT[vessel_name]
        angle_deg = layout['angle']
        ring = layout['ring']
        abbrev = layout['abbrev']
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)
        
        # Get radius for this ring
        radius = _POLAR_RING_RADII.get(ring, 0.5)
        
        # Determine color
        if flow_value is None or not np.isfinite(flow_value):
            # Missing vessel: black
            color = 'black'
            alpha = 0.5
        else:
            # Normalize and get color
            # if vmax > 100: 
            norm_val = normalize(flow_value)
            color = color_map(norm_val)
            # else:
            #     # BoundaryNorm to colormap
            #     from matplotlib.colors import BoundaryNorm
            #     norm = BoundaryNorm(np.linspace(vmin, vmax, color_map.N), ncolors=color_map.N)
            #     color = color_map(norm(flow_value))
            #     # color = color_map(flow_value)
            alpha = 1.0
        
        # Calculate section width (angular width for each section)
        # For center point (TCBF), use small circular ring
        if ring == 0:
            # Center circle for TCBF - use fill_between with full 360 degrees
            theta_center = np.linspace(0, 2*np.pi, 100)
            ax.fill_between(theta_center, 0, radius, color=color, alpha=alpha, zorder=10)
            
            # Add red frame around TCBF (arterial)
            frame_color = 'red'
            frame_width = 2.0
            ax.plot(theta_center, [radius] * len(theta_center), color=frame_color, linewidth=frame_width, zorder=11)
            
            # Add label at center with value below
            norm_val_check = normalize(flow_value) if flow_value is not None else None
            ax.text(0, radius/2, abbrev, ha='center', va='center', 
                   fontsize=10, fontweight='bold', 
                   color='white' if (norm_val_check is not None and norm_val_check > 0.5) else 'black',
                   zorder=12)
            # Add TCBF value as text below the name
            if tcbf_value is not None and np.isfinite(tcbf_value):
                if 'flow' in feature_name.lower():
                    value_str = f'{tcbf_value:.1f}'
                else:
                    value_str = f'{tcbf_value:.2f}'
                ax.text(0, radius/3, value_str, ha='center', va='center',
                       fontsize=8, color='white' if (norm_val_check is not None and norm_val_check > 0.5) else 'black',
                       zorder=12)
        else:
            # Calculate angular boundaries based on adjacent vessels
            theta_start, theta_end = _calculate_polar_boundaries(vessel_name, ring, angle_deg)
            
            # Inner and outer radii for this ring
            if ring > 0:
                prev_ring = max([r for r in _POLAR_RING_RADII.keys() if r < ring], default=0)
                r_inner = _POLAR_RING_RADII.get(prev_ring, 0)
            else:
                r_inner = 0
            r_outer = radius
            
            # Draw wedge using fill_between
            theta_wedge = np.linspace(theta_start, theta_end, 50)
            ax.fill_between(theta_wedge, r_inner, r_outer, color=color, alpha=alpha, zorder=ring)
            
            # Add colored frame based on vessel group (venous = blue, arterial = red)
            group = _VESSEL_NAME_TO_GROUP.get(vessel_name, 'Unknown')
            if group == 'Venous drainage':
                frame_color = 'blue'
            else:
                frame_color = 'red'  # All arterial vessels (TCBF & ICAs, Anterior, Posterior)
            
            # Draw frame as outline around the wedge
            frame_width = 2.0
            # Draw outer edge
            ax.plot(theta_wedge, [r_outer] * len(theta_wedge), color=frame_color, linewidth=frame_width, zorder=ring+5)
            # Draw inner edge (if not at center)
            if r_inner > 0:
                ax.plot(theta_wedge, [r_inner] * len(theta_wedge), color=frame_color, linewidth=frame_width, zorder=ring+5)
            # Draw side edges
            ax.plot([theta_start, theta_start], [r_inner, r_outer], color=frame_color, linewidth=frame_width, zorder=ring+5)
            ax.plot([theta_end, theta_end], [r_inner, r_outer], color=frame_color, linewidth=frame_width, zorder=ring+5)
            
            # Add label at center of wedge
            label_angle = angle_rad
            label_radius = (r_inner + r_outer) / 2
            norm_val_check = normalize(flow_value) if flow_value is not None else None
            ax.text(label_angle, label_radius, abbrev, 
                   ha='center', va='center', fontsize=9, 
                   fontweight='bold', 
                   color='white' if (norm_val_check is not None and norm_val_check > 0.5) else 'black',
                   zorder=ring+10)
    
    # Customize plot
    ax.set_theta_zero_location('N')  # 0 degrees at top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])  # Remove radius labels
    ax.set_xticklabels([])  # Remove angle labels
    ax.spines['polar'].set_visible(False)
    ax.grid(False)  # Remove background grid
    
    # Add title
    if 'flow' in feature_name.lower():
        unit = 'mL/min'
    elif 'pi' in feature_name.lower() or 'pulsatility' in feature_name.lower():
        unit = 'PI'
    else:
        unit = ''
    plt.title(f'Polar {feature_name} Map: {patient_id}\n{unit}', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=color_map, 
                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(unit, rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_polar_flow_animation(
    time_resolved_flow_data: Dict[str, np.ndarray],
    patient_id: str,
    feature_name: str = 'Flow',
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = None,
    nframes: int = 15,
    duration: float = 0.2,
) -> None:
    """
    Create an animated GIF showing polar plots for each time point.
    
    Args:
        time_resolved_flow_data: Dictionary {vessel_code: flow_array} where flow_array has nframes time points
        patient_id: Patient identifier
        feature_name: Name of the feature being plotted (e.g., 'Flow', 'PI')
        output_path: Path to save GIF file
        figsize: Optional figure size (default: (10, 10))
        cmap: Colormap name (default: 'hot')
        nframes: Number of time frames (default: 15)
        duration: Duration of each frame in seconds (default: 0.2)
    """
    if not HAS_IMAGEIO and not HAS_PIL:
        raise ImportError("Need imageio or PIL/Pillow to create GIF animations. Install with: pip install imageio or pip install pillow")
    
    # Use non-interactive backend for GIF creation
    import matplotlib
    original_backend = None
    try:
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Non-interactive backend
    except:
        pass
    
    if figsize is None:
        figsize = (10, 10)

    if cmap is None:
        cmap = "custom_hot_no_green"

    # Convert vessel codes to vessel names and build data for each time point
    all_flow_values = []
    
    # Collect all flow values across all time points for consistent color scale
    for vessel_code, flow_array in time_resolved_flow_data.items():
        vessel_name = _VESSEL_CODE_TO_NAME.get(vessel_code, vessel_code)
        if len(flow_array) == nframes:
            flow_vals = np.abs(flow_array)
            # Convert from mL/s to mL/min for flow
            if 'flow' in feature_name.lower():
                flow_vals = flow_vals * 60.0
            all_flow_values.extend(flow_vals.tolist())
    

    time_resolved_flow_data = {k: np.abs(v) for k, v in time_resolved_flow_data.items()}
    
    if not all_flow_values:
        raise ValueError(f"No valid time-resolved data for patient {patient_id}")
    
    # Calculate global color scale (excluding TCBF from calculation)
    vmin = min(all_flow_values)
    vmax = max(all_flow_values)
    
    # Build data for each time point
    all_time_point_data = []
    for t in range(nframes):
        time_point_data = {}
        tcbf_parts = []
        
        for vessel_code, flow_array in time_resolved_flow_data.items():
            vessel_name = _VESSEL_CODE_TO_NAME.get(vessel_code, vessel_code)
            if len(flow_array) == nframes:
                flow_value = float(np.abs(flow_array[t]))
                # Convert from mL/s to mL/min for flow
                if 'flow' in feature_name.lower():
                    flow_value = flow_value * 60.0
                time_point_data[vessel_name] = flow_value
                tcbf_parts.append(flow_value)
        
        # Add TCBF (sum of all vessels for flow)
        if tcbf_parts:
            time_point_data['TCBF'] = sum(tcbf_parts)
        
        all_time_point_data.append(time_point_data)
    
    # Create frames for each time point
    frames = []
    for t, time_point_data in enumerate(all_time_point_data):
        # Create polar plot for this time point
        fig = plot_polar_flow(
            time_point_data,
            patient_id,
            feature_name=feature_name,
            output_path=None,  # Don't save individual frames
            figsize=figsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        
        # Update title to show time point
        if 'flow' in feature_name.lower():
            unit = 'mL/min'
        elif 'pi' in feature_name.lower() or 'pulsatility' in feature_name.lower():
            unit = 'PI'
        else:
            unit = ''
        
        fig.suptitle(f'Polar {feature_name} Map: {patient_id} - Frame {t+1}/{nframes}\n{unit}', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        # Convert figure to image
        # Use savefig to buffer method (most reliable across matplotlib versions)
        from io import BytesIO
        buf_io = BytesIO()
        fig.savefig(buf_io, format='png', bbox_inches='tight', dpi=100)
        buf_io.seek(0)
        
        # Read image from buffer
        if HAS_IMAGEIO:
            buf = imageio.imread(buf_io)
        elif HAS_PIL:
            img = Image.open(buf_io)
            buf = np.array(img)
        else:
            raise RuntimeError("Need imageio to convert figure to array")
        
        frames.append(buf)
        plt.close(fig)
    
    # Save as GIF
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if HAS_IMAGEIO:
            # Use imageio (preferred)
            imageio.mimsave(str(output_path), frames, duration=duration, loop=0)
        elif HAS_PIL:
            # Use PIL/Pillow
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                str(output_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(duration * 1000),  # Convert to milliseconds
                loop=0,
            )
    
    # Restore original backend if it was changed
    if original_backend is not None:
        try:
            import matplotlib
            matplotlib.use(original_backend)
        except:
            pass


def plot_polar_flow_correlation(
    all_patient_flow_data: Dict[str, Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'coolwarm',
) -> plt.Figure:
    """
    Create a polar plot showing correlation of flow values across patients for each vessel.
    
    Args:
        all_patient_flow_data: Dictionary {patient_id: {vessel_name: flow_value}}
        output_path: Optional path to save figure
        figsize: Optional figure size (default: (10, 10))
        cmap: Colormap name for correlation (default: 'coolwarm')
        
    Returns:
        matplotlib Figure
    """
    if figsize is None:
        figsize = (10, 10)
    
    # Build correlation matrix for vessels
    # First, get all unique vessels across all patients
    all_vessels = set()
    for patient_data in all_patient_flow_data.values():
        all_vessels.update(patient_data.keys())
    vessel_names = sorted(list(all_vessels))
    n_vessels = len(vessel_names)
    n_patients = len(all_patient_flow_data)
    
    if n_vessels < 2:
        raise ValueError("Need at least 2 vessels for correlation analysis")
    
    # Build vessel_flows with consistent length (one value per patient for each vessel)
    vessel_flows = {}
    for vessel_name in vessel_names:
        vessel_flows[vessel_name] = []
        for patient_id in sorted(all_patient_flow_data.keys()):
            patient_data = all_patient_flow_data[patient_id]
            flow_value = patient_data.get(vessel_name, None)
            vessel_flows[vessel_name].append(flow_value if flow_value is not None else np.nan)
    
    # Calculate correlation between vessels
    # Build correlation matrix
    correlation_matrix = np.zeros((n_vessels, n_vessels))
    for i, vessel1 in enumerate(vessel_names):
        for j, vessel2 in enumerate(vessel_names):
            flows1 = np.abs(vessel_flows[vessel1])
            flows2 = np.abs(vessel_flows[vessel2])
            
            # Remove NaN pairs
            valid = ~(np.isnan(flows1) | np.isnan(flows2))
            if valid.sum() >= 2:
                corr = np.corrcoef(flows1[valid], flows2[valid])[0, 1]
                correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
            else:
                correlation_matrix[i, j] = 0
    
    # Create figure with polar projection
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection='polar')
    
    # Get colormap
    try:
        from matplotlib import colormaps
        color_map = colormaps[cmap]
    except (ImportError, AttributeError):
        color_map = plt.cm.get_cmap(cmap)
    
    # Normalize correlation values (-1 to 1)
    vmin, vmax = -1, 1
    
    # Plot each vessel section colored by average correlation with other vessels
    for idx, vessel_name in enumerate(vessel_names):
        if vessel_name not in _POLAR_VESSEL_LAYOUT:
            continue
        
        layout = _POLAR_VESSEL_LAYOUT[vessel_name]
        angle_deg = layout['angle']
        ring = layout['ring']
        abbrev = layout['abbrev']
        
        # Calculate average correlation for this vessel
        avg_corr = np.mean(correlation_matrix[idx, :])
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_deg)
        
        # Get radius for this ring
        radius = _POLAR_RING_RADII.get(ring, 0.5)
        
        # Normalize correlation to 0-1 for colormap
        norm_val = (avg_corr - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        color = color_map(norm_val)
        
        # Use same plotting logic as per-patient plot
        if ring == 0:
            # Center circle for TCBF - use fill_between with full 360 degrees
            theta_center = np.linspace(0, 2*np.pi, 100)
            ax.fill_between(theta_center, 0, radius, color=color, alpha=1.0, zorder=10)
            
            # Add red frame around TCBF (arterial)
            frame_color = 'red'
            frame_width = 2.0
            ax.plot(theta_center, [radius] * len(theta_center), color=frame_color, linewidth=frame_width, zorder=11)
            
            # Add label at center
            ax.text(0, radius/2, abbrev, ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white' if norm_val > 0.5 else 'black',
                   zorder=12)
        else:
            # Calculate angular boundaries based on adjacent vessels
            theta_start, theta_end = _calculate_polar_boundaries(vessel_name, ring, angle_deg)
            
            # Get inner and outer radii
            if ring > 0:
                prev_ring = max([r for r in _POLAR_RING_RADII.keys() if r < ring], default=0)
                r_inner = _POLAR_RING_RADII.get(prev_ring, 0)
            else:
                r_inner = 0
            r_outer = radius
            
            # Draw wedge using fill_between
            theta_wedge = np.linspace(theta_start, theta_end, 50)
            ax.fill_between(theta_wedge, r_inner, r_outer, color=color, alpha=1.0, zorder=ring)
            
            # Add colored frame based on vessel group (venous = blue, arterial = red)
            group = _VESSEL_NAME_TO_GROUP.get(vessel_name, 'Unknown')
            if group == 'Venous drainage':
                frame_color = 'blue'
            else:
                frame_color = 'red'  # All arterial vessels
            
            # Draw frame as outline around the wedge
            frame_width = 2.0
            # Draw outer edge
            ax.plot(theta_wedge, [r_outer] * len(theta_wedge), color=frame_color, linewidth=frame_width, zorder=ring+5)
            # Draw inner edge (if not at center)
            if r_inner > 0:
                ax.plot(theta_wedge, [r_inner] * len(theta_wedge), color=frame_color, linewidth=frame_width, zorder=ring+5)
            # Draw side edges
            ax.plot([theta_start, theta_start], [r_inner, r_outer], color=frame_color, linewidth=frame_width, zorder=ring+5)
            ax.plot([theta_end, theta_end], [r_inner, r_outer], color=frame_color, linewidth=frame_width, zorder=ring+5)
            
            # Add label
            label_angle = angle_rad
            label_radius = (r_inner + r_outer) / 2
            ax.text(label_angle, label_radius, abbrev, 
                   ha='center', va='center', fontsize=9, 
                   fontweight='bold', 
                   color='white' if norm_val > 0.5 else 'black',
                   zorder=ring+10)
    
    # Customize plot
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)  # Remove background grid
    
    plt.title('Polar Flow Correlation Map\nAverage Correlation Across Patients', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=color_map, 
                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_flow_timeseries(
    flow_data: Dict[str, np.ndarray],
    patient_id: str,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    nframes: int = 15,
    ylim: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Plot flow over time for each vessel LOC for a single patient.
    
    Args:
        flow_data: Dictionary {vessel_code: flow_array} where flow_array is (nframes,)
        patient_id: Patient ID for title
        output_path: Optional path to save figure
        figsize: Optional figure size
        nframes: Number of time frames
        ylim: Optional y-axis limits (min, max) for consistent scaling
        
    Returns:
        matplotlib Figure
    """
    if figsize is None:
        figsize = (14, 8)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Time points (assuming 0-indexed frames)
    time_points = np.arange(nframes)

    # Set values as the absolute value
    for vessel_code, flow_array in flow_data.items():
        flow_array = np.abs(flow_array)
        flow_data[vessel_code] = flow_array
    
    # Plot each vessel
    colors = sns.color_palette("husl", n_colors=len(flow_data))
    for i, (vessel_code, flow_array) in enumerate(flow_data.items()):
        vessel_name = _VESSEL_CODE_TO_NAME.get(vessel_code, vessel_code)
        if len(flow_array) == nframes:
            ax.plot(time_points, flow_array, label=vessel_name, color=colors[i], linewidth=2, marker='o', markersize=4)
    
    # Set y-limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    ax.set_xlabel('Cardiac Frame', fontsize=12)
    ax.set_ylabel('Flow (mL/s)', fontsize=12)
    ax.set_title(f'Time-Resolved Flow: {patient_id}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_flow_timeseries_all(
    all_flow_data: Dict[str, Dict[str, np.ndarray]],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    nframes: int = 15,
    max_patients_per_plot: int = 10,
) -> List[plt.Figure]:
    """
    Plot flow over time for all patients (may create multiple figures if many patients).
    All patients in the same plot share the same y-limits for visual comparison.
    
    Args:
        all_flow_data: Dictionary {patient_id: {vessel_code: flow_array}}
        output_path: Optional base path to save figures (will append _page_N.png)
        figsize: Optional figure size
        nframes: Number of time frames
        max_patients_per_plot: Maximum number of patients per plot
        
    Returns:
        List of matplotlib Figures
    """
    if figsize is None:
        figsize = (16, 10)
    
    patient_ids = list(all_flow_data.keys())
    n_patients = len(patient_ids)
    
    # Calculate global y-limits across all patients for consistent scaling
    all_flow_values = []
    for patient_flow in all_flow_data.values():
        for flow_array in patient_flow.values():
            if len(flow_array) == nframes:
                all_flow_values.extend(flow_array.tolist())
    
    if all_flow_values:
        y_min = min(all_flow_values)
        y_max = max(all_flow_values)
        # Add some padding
        y_range = y_max - y_min
        ylim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    else:
        ylim = None
    
    # Split into multiple plots if needed
    n_plots = (n_patients + max_patients_per_plot - 1) // max_patients_per_plot
    
    figures = []
    time_points = np.arange(nframes)
    
    for plot_idx in range(n_plots):
        start_idx = plot_idx * max_patients_per_plot
        end_idx = min(start_idx + max_patients_per_plot, n_patients)
        plot_patients = patient_ids[start_idx:end_idx]
        
        fig, axes = plt.subplots(
            len(plot_patients), 1,
            figsize=(figsize[0], figsize[1] * len(plot_patients) / max_patients_per_plot),
            sharex=True,
            sharey=True,  # Share y-axis for consistent scaling
        )
        
        if len(plot_patients) == 1:
            axes = [axes]
        
        for ax_idx, patient_id in enumerate(plot_patients):
            ax = axes[ax_idx]
            flow_data = all_flow_data[patient_id]
            
            # Plot each vessel for this patient
            colors = sns.color_palette("husl", n_colors=len(flow_data))
            for i, (vessel_code, flow_array) in enumerate(flow_data.items()):
                vessel_name = _VESSEL_CODE_TO_NAME.get(vessel_code, vessel_code)
                if len(flow_array) == nframes:
                    ax.plot(time_points, flow_array, label=vessel_name, color=colors[i], linewidth=1.5, marker='o', markersize=3)
            
            # Set consistent y-limits
            if ylim is not None:
                ax.set_ylim(ylim)
            
            ax.set_ylabel('Flow (mL/s)', fontsize=10)
            ax.set_title(f'{patient_id}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if ax_idx == len(plot_patients) - 1:
                ax.set_xlabel('Cardiac Frame', fontsize=12)
            if ax_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        
        plt.suptitle(f'Time-Resolved Flow: All Patients (Page {plot_idx + 1}/{n_plots})', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 0.95, 0.99])
        
        figures.append(fig)
        
        if output_path:
            output_path_obj = Path(output_path)
            if n_plots > 1:
                stem = output_path_obj.stem
                suffix = output_path_obj.suffix
                page_path = output_path_obj.parent / f"{stem}_page_{plot_idx + 1}{suffix}"
            else:
                page_path = output_path_obj
            fig.savefig(page_path, dpi=300, bbox_inches='tight')
    
    return figures


def _find_loc_row_index(
    data_struct: Union[Dict, object],
    loc: List[int]
) -> Optional[int]:
    """
    Find the row index in branchList for a given LOC.
    
    Args:
        data_struct: Data structure from QVT+ .mat file
        loc: [segment_id, centerline_idx]
        
    Returns:
        Row index in branchList, or None if not found
    """
    if len(loc) < 2:
        return None
    
    segment_id, centerline_idx = int(loc[0]), int(loc[1])
    
    # Get branchList
    if isinstance(data_struct, dict):
        branch_list = data_struct.get('branchList', None)
    else:
        branch_list = getattr(data_struct, 'branchList', None)
    
    if branch_list is None:
        return None
    
    # Find row where segment_id and centerline_idx match
    if isinstance(branch_list, np.ndarray):
        if branch_list.ndim == 2 and branch_list.shape[1] >= 5:
            matches = np.where(
                (branch_list[:, 3] == segment_id) & 
                (branch_list[:, 4] == centerline_idx)
            )[0]
            if len(matches) == 0:
                return None
            return int(matches[0])
    
    return None


def plot_crosssections(
    all_patient_metadata: Dict[str, Dict],
    output_path: Optional[Union[str, Path]] = None,
    figsize_per_panel: Tuple[float, float] = (4, 4),
    max_patients_per_figure: int = 6,
    cross_section_type: str = 'MAG',
) -> List[plt.Figure]:
    """
    Create mosaic plots showing cross-sections at vessel LOCs with measurement pixels highlighted.
    Loads .mat files on-demand and frees memory immediately after extraction.
    
    Args:
        all_patient_metadata: Dictionary {patient_id: metadata_dict} where metadata_dict
                             is the output from load_patient_metadata()
        output_path: Optional base path to save figures (will append _page_N.png)
        figsize_per_panel: Size of each cross-section panel (width, height)
        max_patients_per_figure: Maximum number of patients per figure page
        cross_section_type: Type of cross-section to display ('MAG', 'CD', 'VEL', or 'combined')
                           'MAG' = Magnitude, 'CD' = Complex Difference, 'VEL' = Velocity,
                           'combined' = RGB composite (R=MAG, G=CD, B=VEL)
        
    Returns:
        List of matplotlib Figures
    """
    if cross_section_type not in ['MAG', 'CD', 'VEL', 'combined']:
        raise ValueError(f"cross_section_type must be one of ['MAG', 'CD', 'VEL', 'combined'], got '{cross_section_type}'")
    
    patient_ids = list(all_patient_metadata.keys())
    n_patients = len(patient_ids)
    
    if n_patients == 0:
        raise ValueError("No patient metadata provided")
    
    # Collect all vessels across all patients
    all_vessels = set()
    for metadata in all_patient_metadata.values():
        locs = metadata.get('LOCs', {})
        all_vessels.update(locs.keys())
    all_vessels = sorted(list(all_vessels))
    
    # Filter out TCBF if present (it's a calculated value, not a vessel with LOC)
    all_vessels = [v for v in all_vessels if v != 'TCBF']
    
    if not all_vessels:
        raise ValueError("No vessel LOCs found in patient data")
    
    # Split into multiple figures if needed
    n_figures = (n_patients + max_patients_per_figure - 1) // max_patients_per_figure
    figures = []
    
    for fig_idx in range(n_figures):
        start_patient_idx = fig_idx * max_patients_per_figure
        end_patient_idx = min(start_patient_idx + max_patients_per_figure, n_patients)
        page_patient_ids = patient_ids[start_patient_idx:end_patient_idx]
        
        # Calculate figure size based on number of patients and vessels
        n_patients_page = len(page_patient_ids)
        n_vessels = len(all_vessels)
        
        # Layout: rows = patients, cols = vessels
        fig_width = n_vessels * figsize_per_panel[0]
        fig_height = n_patients_page * figsize_per_panel[1] + 1  # Extra space for titles
        
        fig, axes = plt.subplots(
            n_patients_page, n_vessels,
            figsize=(fig_width, fig_height),
            squeeze=False
        )
        
        # Process each patient
        for row_idx, patient_id in enumerate(page_patient_ids):
            metadata = all_patient_metadata[patient_id]
            patient_dir = Path(metadata['patient_dir'])
            locs = metadata.get('LOCs', {})
            
            # Load cross-section data on-demand
            crosssection_data = extract_crosssection_data_from_mat(patient_dir, locs, cross_section_type)
            
            if not crosssection_data:
                # Skip this patient if no cross-section data
                for col_idx in range(n_vessels):
                    axes[row_idx, col_idx].axis('off')
                    if row_idx == 0:
                        vessel_code = all_vessels[col_idx]
                        vessel_name = _VESSEL_CODE_TO_NAME.get(vessel_code, vessel_code)
                        axes[row_idx, col_idx].set_title(vessel_name, fontsize=10)
                continue
            
            # Get image dimension from first vessel's mask
            first_vessel_data = next(iter(crosssection_data.values()))
            mask = first_vessel_data['mask']
            imdim = mask.shape[0]
            
            # Process each vessel
            for col_idx, vessel_code in enumerate(all_vessels):
                ax = axes[row_idx, col_idx]
                
                # Get cross-section data for this vessel
                vessel_data = crosssection_data.get(vessel_code, None)
                
                if vessel_data is None:
                    # No data for this vessel in this patient
                    ax.axis('off')
                    if row_idx == 0:
                        vessel_name = _VESSEL_CODE_TO_NAME.get(vessel_code, vessel_code)
                        ax.set_title(vessel_name, fontsize=10, color='gray')
                    continue
                
                # Extract mask and images
                mask = vessel_data['mask']
                
                # Prepare images based on type
                if cross_section_type == 'MAG' and 'mag' in vessel_data:
                    img = vessel_data['mag']
                    cmap = 'gray'
                    overlay_alpha = 0.6
                elif cross_section_type == 'CD' and 'cd' in vessel_data:
                    img = vessel_data['cd']
                    cmap = 'gray'
                    overlay_alpha = 0.6
                elif cross_section_type == 'VEL' and 'vel' in vessel_data:
                    img = vessel_data['vel']
                    cmap = 'gray'
                    overlay_alpha = 0.6
                elif cross_section_type == 'combined':
                    # Create RGB composite
                    img_r = np.zeros((imdim, imdim))
                    img_g = np.zeros((imdim, imdim))
                    img_b = np.zeros((imdim, imdim))
                    
                    if 'mag' in vessel_data:
                        mag_img = vessel_data['mag']
                        mag_norm = (mag_img - mag_img.min()) / (mag_img.max() - mag_img.min() + 1e-10)
                        img_r = mag_norm
                    
                    if 'cd' in vessel_data:
                        cd_img = vessel_data['cd']
                        cd_norm = (cd_img - cd_img.min()) / (cd_img.max() - cd_img.min() + 1e-10)
                        img_g = cd_norm
                    
                    if 'vel' in vessel_data:
                        vel_img = vessel_data['vel']
                        vel_norm = (vel_img - vel_img.min()) / (vel_img.max() - vel_img.min() + 1e-10)
                        img_b = vel_norm
                    
                    img = np.stack([img_r, img_g, img_b], axis=-1)
                    cmap = None
                    overlay_alpha = 0.5
                else:
                    # Fallback: use mask only if image data unavailable
                    img = mask.astype(float)
                    cmap = 'gray'
                    overlay_alpha = 0.8
                
                # Handle complex numbers
                if np.iscomplexobj(img):
                    img = np.abs(img)
                
                # Normalize image for display (if grayscale)
                if cmap is not None:
                    img_min = img.min() if img.size > 0 else 0
                    img_max = img.max() if img.size > 0 else 1
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
                
                # Display image
                if cmap is not None:
                    ax.imshow(img, cmap=cmap, origin='upper', interpolation='bilinear')
                else:
                    ax.imshow(img, origin='upper', interpolation='bilinear')
                
                # Highlight measurement pixels in red (all pixels in segmentFull mask at LOC)
                # These are the exact pixels where flow and PI are calculated
                mask_binary = (mask > 0).astype(float)
                if mask_binary.sum() > 0:
                    # Create red overlay for measurement pixels (highlighting where flow/PI is measured)
                    # The measurement happens over ALL pixels in the mask at this LOC
                    overlay = np.zeros((imdim, imdim, 4))  # RGBA
                    overlay[:, :, 0] = mask_binary * 1.0  # Red channel (full red for measurement pixels)
                    overlay[:, :, 3] = mask_binary * 0.6  # Alpha channel (60% opacity)
                    
                    ax.imshow(overlay, origin='upper', interpolation='nearest')
                
                # Add red contour outline of mask boundary
                if mask_binary.sum() > 0:
                    try:
                        # Use matplotlib's contour to draw red outline
                        ax.contour(mask_binary, levels=[0.5], colors='red', linewidths=1.5, alpha=0.9)
                    except Exception:
                        # Fallback: just use overlay if contour fails
                        pass
                
                ax.axis('off')
                
                # Set titles
                if row_idx == 0:
                    vessel_name = _VESSEL_CODE_TO_NAME.get(vessel_code, vessel_code)
                    ax.set_title(vessel_name, fontsize=10, fontweight='bold')
                
                if col_idx == 0:
                    ax.text(-0.05, 0.5, patient_id, transform=ax.transAxes,
                           rotation=90, va='center', ha='right', fontsize=9, fontweight='bold')
        
        # Overall title
        page_title = f'Cross-Sections at LOCs ({cross_section_type})'
        if n_figures > 1:
            page_title += f' - Page {fig_idx + 1}/{n_figures}'
        fig.suptitle(page_title, fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        
        figures.append(fig)
        
        # Save if output path provided
        if output_path:
            output_path_obj = Path(output_path)
            if n_figures > 1:
                stem = output_path_obj.stem
                suffix = output_path_obj.suffix
                page_path = output_path_obj.parent / f"{stem}_page_{fig_idx + 1}{suffix}"
            else:
                page_path = output_path_obj
            fig.savefig(page_path, dpi=300, bbox_inches='tight')
    
    return figures


# Tree/dendrogram vessel hierarchy definition for Circle of Willis visualization
_TREE_VESSEL_HIERARCHY = {
    'TCBF': {'children': ['Left ICA', 'Right ICA', 'Basilar'], 'level': 0, 'abbrev': 'TCBF', 'color': 'red'},
    'Left ICA': {'children': ['Left MCA', 'Left ACA'], 'parent': 'TCBF', 'level': 1, 'abbrev': 'LICA', 'color': 'red'},
    'Right ICA': {'children': ['Right MCA', 'Right ACA'], 'parent': 'TCBF', 'level': 1, 'abbrev': 'RICA', 'color': 'red'},
    'Basilar': {'children': ['Left PCA', 'Right PCA'], 'parent': 'TCBF', 'level': 1, 'abbrev': 'BASI', 'color': 'red'},
    'Left MCA': {'children': [], 'parent': 'Left ICA', 'level': 2, 'abbrev': 'LMCA', 'color': 'red'},
    'Left ACA': {'children': [], 'parent': 'Left ICA', 'level': 2, 'abbrev': 'LACA', 'color': 'red'},
    'Right MCA': {'children': [], 'parent': 'Right ICA', 'level': 2, 'abbrev': 'RMCA', 'color': 'red'},
    'Right ACA': {'children': [], 'parent': 'Right ICA', 'level': 2, 'abbrev': 'RACA', 'color': 'red'},
    'Left PCA': {'children': [], 'parent': 'Basilar', 'level': 2, 'abbrev': 'LPCA', 'color': 'red'},
    'Right PCA': {'children': [], 'parent': 'Basilar', 'level': 2, 'abbrev': 'RPCA', 'color': 'red'},
    'Right Communicating': {'children': [], 'parent': 'Left MCA', 'level': 2.5, 'abbrev': 'RCOMM', 'color': 'orange', 'connects': ['Left MCA', 'Left PCA']},
    'Left Communicating': {'children': [], 'parent': 'Right MCA', 'level': 2.5, 'abbrev': 'LCOMM', 'color': 'orange', 'connects': ['Right MCA', 'Right PCA']},
    'Sagital Sinus': {'children': ['Left Transverse', 'Right Transverse'], 'parent': None, 'level': 3, 'abbrev': 'SSSV', 'color': 'blue'},
    'Straight Sinus': {'children': [], 'parent': None, 'level': 3, 'abbrev': 'STRV', 'color': 'blue'},
    'Left Transverse': {'children': [], 'parent': 'Sagital Sinus', 'level': 3, 'abbrev': 'LTSV', 'color': 'blue'},
    'Right Transverse': {'children': [], 'parent': 'Sagital Sinus', 'level': 3, 'abbrev': 'RTSV', 'color': 'blue'},
}


def _build_tree_layout(vessel_hierarchy: Dict) -> Dict:
    """Build circular tree layout positions for vessel dendrogram."""
    vessel_positions = {}
    
    # Use polar coordinates (angle, radius) for circular layout
    # Convert to cartesian for plotting
    
    # Level 0: TCBF at center
    vessel_positions['TCBF'] = (0, 0)  # Center
    
    # Level 1: Inlets (ICAs and Basilar) - arranged in a circle
    level1_radius = 1.0
    level1_angles = {
        'Left ICA': 90,      # Top
        'Right ICA': -90,    # Bottom
        'Basilar': 180,      # Left
    }
    for vessel, angle_deg in level1_angles.items():
        angle_rad = np.deg2rad(angle_deg)
        vessel_positions[vessel] = (level1_radius * np.cos(angle_rad), level1_radius * np.sin(angle_rad))
    
    # Level 2: Major branches - arranged around their parents
    # ACAs and MCAs swapped: ACAs at outer positions, MCAs at inner positions
    level2_radius = 2.0
    level2_angle_offsets = {
        'Left MCA': 30,    # Offset from Left ICA (inner, swapped with ACA)
        'Left ACA': -30,   # Offset from Left ICA (outer, swapped with MCA)
        'Right MCA': -30,  # Offset from Right ICA (inner, swapped with ACA)
        'Right ACA': 30,   # Offset from Right ICA (outer, swapped with MCA)
        'Left PCA': -30,   # Offset from Basilar
        'Right PCA': 30,
    }
    for vessel, info in vessel_hierarchy.items():
        if info.get('level') == 2:
            parent = info.get('parent')
            if parent and parent in vessel_positions:
                # Get parent angle
                parent_x, parent_y = vessel_positions[parent]
                parent_angle = np.arctan2(parent_y, parent_x)
                # Add offset
                offset_deg = level2_angle_offsets.get(vessel, 0)
                child_angle = parent_angle + np.deg2rad(offset_deg)
                vessel_positions[vessel] = (level2_radius * np.cos(child_angle), level2_radius * np.sin(child_angle))
    
    # Level 2.5: Communicating vessels - between MCAs and PCAs
    # RCOMM connects Left MCA to Left PCA (right side of brain)
    # LCOMM connects Right MCA to Right PCA (left side of brain)
    level2_5_radius = 2.3
    # RCOMM connects Left MCA to Left PCA (swapped)
    if 'Left MCA' in vessel_positions and 'Left PCA' in vessel_positions:
        left_mca_x, left_mca_y = vessel_positions['Left MCA']
        left_pca_x, left_pca_y = vessel_positions['Left PCA']
        mca_angle = np.arctan2(left_mca_y, left_mca_x)
        pca_angle = np.arctan2(left_pca_y, left_pca_x)
        # Handle wrap-around
        if abs(pca_angle - mca_angle) > np.pi:
            if pca_angle < 0:
                pca_angle += 2 * np.pi
            else:
                mca_angle += 2 * np.pi
        mid_angle = (mca_angle + pca_angle) / 2
        if 'Right Communicating' in vessel_hierarchy:
            vessel_positions['Right Communicating'] = (level2_5_radius * np.cos(mid_angle), level2_5_radius * np.sin(mid_angle))
    
    # LCOMM connects Right MCA to Right PCA (swapped)
    if 'Right MCA' in vessel_positions and 'Right PCA' in vessel_positions:
        right_mca_x, right_mca_y = vessel_positions['Right MCA']
        right_pca_x, right_pca_y = vessel_positions['Right PCA']
        mca_angle = np.arctan2(right_mca_y, right_mca_x)
        pca_angle = np.arctan2(right_pca_y, right_pca_x)
        # Handle wrap-around
        if abs(pca_angle - mca_angle) > np.pi:
            if pca_angle < 0:
                pca_angle += 2 * np.pi
            else:
                mca_angle += 2 * np.pi
        mid_angle = (mca_angle + pca_angle) / 2
        if 'Left Communicating' in vessel_hierarchy:
            vessel_positions['Left Communicating'] = (level2_5_radius * np.cos(mid_angle), level2_5_radius * np.sin(mid_angle))
    
    # Level 3: Venous system - arranged in outer circle
    # All venous vessels on the same radius level for better visibility
    level3_radius = 3.0
    venous_angles = {
        'Sagital Sinus': 0,        # Top
        'Straight Sinus': 180,     # Bottom (with other veins)
        'Left Transverse': 135,    # Bottom-left
        'Right Transverse': -135,  # Bottom-right
    }
    for vessel, angle_deg in venous_angles.items():
        if vessel in vessel_hierarchy:
            angle_rad = np.deg2rad(angle_deg)
            vessel_positions[vessel] = (level3_radius * np.cos(angle_rad), level3_radius * np.sin(angle_rad))
    
    # Level 3 children (transverse sinuses below sagital)
    level3_child_radius = 3.3  # Slightly further out
    for vessel, info in vessel_hierarchy.items():
        if info.get('level') == 3 and info.get('parent'):
            parent = info.get('parent')
            if parent in vessel_positions and vessel in vessel_hierarchy[parent].get('children', []):
                parent_x, parent_y = vessel_positions[parent]
                parent_angle = np.arctan2(parent_y, parent_x)
                # Position children slightly outward from parent
                child_angle = parent_angle
                vessel_positions[vessel] = (level3_child_radius * np.cos(child_angle), level3_child_radius * np.sin(child_angle))
    
    return vessel_positions


def plot_tree(patient_vessel_data: Dict[str, float], patient_id: str, feature_name: str = 'Flow',
               output_path: Optional[Union[str, Path]] = None, figsize: Optional[Tuple[float, float]] = None,
               cmap: str = None, vmin: Optional[float] = None, vmax: Optional[float] = None) -> plt.Figure:
    """Create a tree/dendrogram plot showing Circle of Willis and venous system structure."""
    if figsize is None:
        figsize = (14, 10)
    fig, ax = plt.subplots(figsize=figsize)

    if cmap is not None:
        try:
            from matplotlib import colormaps
            color_map = colormaps[cmap]
        except (ImportError, AttributeError):
            color_map = plt.cm.get_cmap(cmap)
    else:
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list(
            'custom_hot_no_green',
            [
                (1, 1, 1),      # White
                (1, 1, 0),      # Light Yellow
                (1, 0.5, 0),    # Orange
                (1, 0, 0),      # Red
            ],
            N=1024
        )

    flow_values = [v for k, v in patient_vessel_data.items() if v is not None and np.isfinite(v) and k != 'TCBF']
    if not flow_values:
        raise ValueError(f"No valid data for patient {patient_id}")
    flow_values = np.abs(flow_values)
    patient_vessel_data = {k: np.abs(v) for k, v in patient_vessel_data.items()}
    if vmin is None:
        vmin = min(flow_values)
    if vmax is None:
        vmax = max(flow_values)
    
    def normalize(value):
        if value is None or not np.isfinite(value):
            return None
        return 0.5 if vmax == vmin else (value - vmin) / (vmax - vmin)
    
    vessel_positions = _build_tree_layout(_TREE_VESSEL_HIERARCHY)
    
    # Helper function to convert color name or colormap output to RGB tuple
    def get_rgb_color(color_input):
        """Convert color name or colormap output to RGB tuple."""
        if isinstance(color_input, str):
            # Convert color name to RGB using matplotlib
            from matplotlib.colors import to_rgb
            return to_rgb(color_input)
        elif isinstance(color_input, (tuple, list, np.ndarray)):
            # Colormap returns RGBA, take first 3 for RGB
            return tuple(color_input[:3])
        else:
            return (0.5, 0.5, 0.5)  # Default gray
    
    # Draw vessel connections as colored lines (thickness based on feature value)
    for vessel, info in _TREE_VESSEL_HIERARCHY.items():
        if vessel not in vessel_positions or vessel not in patient_vessel_data:
            continue
        
        x, y = vessel_positions[vessel]  
        value = patient_vessel_data[vessel]
        
        # Determine line color and thickness based on feature value
        if value is None or not np.isfinite(value):
            line_color_rgb = (0.5, 0.5, 0.5)  # Gray
            line_width = 1.0
            alpha = 0.3
        else:
            norm_val = normalize(value)
            cmap_output = color_map(norm_val) if norm_val is not None else (0.5, 0.5, 0.5, 1.0)
            line_color_rgb = get_rgb_color(cmap_output)
            # Line width proportional to value (min 1.5, max 8.0)
            line_width = 1.5 + (norm_val * 6.5) if norm_val is not None else 1.5
            alpha = 1.0
        
        # Draw connection to parent
        parent = info.get('parent')
        if parent and parent in vessel_positions:
            parent_x, parent_y = vessel_positions[parent]
            # Use vessel-specific base color for structure, but overlay with feature color
            base_color_name = info.get('color', 'gray')
            base_color_rgb = get_rgb_color(base_color_name)
            # Blend base color (30%) with feature color (70%)
            blended_color = (
                base_color_rgb[0] * 0.3 + line_color_rgb[0] * 0.7,
                base_color_rgb[1] * 0.3 + line_color_rgb[1] * 0.7,
                base_color_rgb[2] * 0.3 + line_color_rgb[2] * 0.7,
            )
            ax.plot([parent_x, x], [parent_y, y], color=blended_color, linewidth=line_width, 
                   alpha=alpha, zorder=2, solid_capstyle='round')
        
        # Draw connecting lines for communicating vessels
        if 'connects' in info:
            for connected_vessel in info['connects']:
                if connected_vessel in vessel_positions:
                    conn_x, conn_y = vessel_positions[connected_vessel]
                    # Communicating vessels use dashed lines with orange tint
                    comm_color_name = info.get('color', 'orange')
                    comm_color_rgb = get_rgb_color(comm_color_name)
                    # Blend communicating color (40%) with feature color (60%)
                    blended_comm = (
                        comm_color_rgb[0] * 0.4 + line_color_rgb[0] * 0.6,
                        comm_color_rgb[1] * 0.4 + line_color_rgb[1] * 0.6,
                        comm_color_rgb[2] * 0.4 + line_color_rgb[2] * 0.6,
                    )
                    ax.plot([x, conn_x], [y, conn_y], color=blended_comm, 
                           linewidth=max(1.0, line_width * 0.7), linestyle='--', 
                           alpha=alpha * 0.8, zorder=2, solid_capstyle='round')
    
    # Draw vessel labels at endpoints (no dots, just text labels)
    for vessel, (x, y) in vessel_positions.items():
        if vessel not in patient_vessel_data:
            continue
        value = patient_vessel_data[vessel]
        info = _TREE_VESSEL_HIERARCHY.get(vessel, {})
        abbrev = info.get('abbrev', vessel[:5])
        norm_val = None
        if value is None or not np.isfinite(value):
            text_color = 'gray'
            bg_color_rgb = (0.9, 0.9, 0.9)  # Light gray
        else:
            norm_val = normalize(value)
            cmap_output = color_map(norm_val) if norm_val is not None else (0.5, 0.5, 0.5, 1.0)
            bg_color_rgb = get_rgb_color(cmap_output)
            # Text color: white if dark background, black if light
            brightness = (bg_color_rgb[0] + bg_color_rgb[1] + bg_color_rgb[2]) / 3.0
            text_color = 'white' if brightness < 0.5 else 'black'
        
        # Draw small background circle for label
        circle_size = 0.15 if vessel == 'TCBF' else 0.12
        circle = plt.Circle((x, y), circle_size, color=bg_color_rgb, alpha=0.9, zorder=10, 
                           edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        
        # Add label text
        ax.text(x, y, abbrev, ha='center', va='center', fontsize=10 if vessel == 'TCBF' else 8, 
               fontweight='bold', color=text_color, zorder=11)
    
    # Set limits to accommodate circular layout with venous system
    max_radius = 4.0  # Account for venous system at radius 3.3
    margin = 0.8  # Increased margin for better visibility
    ax.set_xlim(-max_radius - margin, max_radius + margin)
    ax.set_ylim(-max_radius - margin, max_radius + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    unit = 'mL/min' if 'flow' in feature_name.lower() else ('PI' if 'pi' in feature_name.lower() or 'pulsatility' in feature_name.lower() else '')
    plt.title(f'Circle of Willis Tree: {patient_id} ({feature_name} {unit})', fontsize=16, fontweight='bold', pad=20)
    
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.05, shrink=0.6, aspect=20)
    cbar.set_label(unit, rotation=270, labelpad=20)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', linewidth=2, label='Arterial'),
                       Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='Communicating'),
                       Line2D([0], [0], color='blue', linewidth=2, label='Venous')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig
