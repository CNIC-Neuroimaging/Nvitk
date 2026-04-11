#!/usr/bin/env python3
"""
PESA-Brain QVT+ Analysis Plotting CLI

Provides CLI interface for generating visualizations from QVT+ outputs:
- Violin plots for vessel features
- Correlation heatmaps
- Flow timeseries plots
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import matplotlib.pyplot as plt

# Ensure the path includes source directory
project_root = Path(__file__).resolve().parents[4]
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

try:
    from imaging.core.logger import Logger
    log = Logger(name='pesa_brain_analysis')
    log.set_level('INFO')
    print, warn, err = log.info, log.warning, log.error
except Exception:
    print = lambda x: click.echo(x)
    warn = lambda x: click.echo(click.style(x, fg='yellow'))
    err = lambda x: click.echo(click.style(x, fg='red'))

import numpy as np
import pandas as pd
from plotter import (
    load_patient_summary,
    load_qvt_data,
    load_patient_metadata,
    load_summary_data,
    extract_feature_from_loc,
    extract_feature_values_from_mat,
    extract_flow_timeseries_from_mat,
    extract_crosssection_data_from_mat,
    plot_violin,
    plot_correlation_heatmap,
    plot_flow_timeseries,
    plot_flow_timeseries_all,
    plot_polar_flow,
    plot_polar_flow_correlation,
    plot_polar_flow_animation,
    plot_crosssections,
    plot_tree,
    _VESSEL_CODE_TO_NAME,
    _VESSEL_NAME_TO_GROUP,
    _match_vessel_label,
)
from correlater import (
    build_patient_feature_table_from_summary,
    load_clinical_summary,
    save_eda_overview,
    save_clinical_correlation_outputs,
    save_flow_pi_cross_correlation,
    save_polar_clinical_correlations,
    save_polar_difference_correlations,
    save_vessel_spatial_correlation_outputs,
)


def load_all_patients(
    results_dir: Path,
    summary_xlsx: Optional[Path] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Load only lightweight metadata for all patients (no .mat files).
    .mat files will be loaded on-demand when needed for specific plots.
    Uses progress bar instead of printing individual patient names.
    
    Args:
        results_dir: Directory containing patient subdirectories
        summary_xlsx: Optional path to summary Excel file
        
    Returns:
        Tuple of (patient_metadata_dict, summary_df)
        patient_metadata_dict: {patient_id: metadata_dict} with patient_id, patient_dir, LOCs, summary_file
        (No data_struct - .mat files loaded on-demand)
    """
    results_dir = Path(results_dir)
    
    # Load patient summary if available
    summary_df = None
    if summary_xlsx is None:
        # Try to find summary file
        xlsx_files = list((results_dir / 'centerline_test').glob('*.xlsx'))
        if xlsx_files:
            summary_xlsx = xlsx_files[0]
    
    if summary_xlsx and Path(summary_xlsx).exists():
        try:
            summary_df = load_patient_summary(results_dir, summary_xlsx)
            print(f"Loaded patient summary from {summary_xlsx}")
        except Exception as e:
            warn(f"Could not load patient summary: {e}")
    
    # Find all patient directories
    patient_dirs = [
        d for d in results_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
        and d.name.startswith('PESA')
    ]
    
    if not patient_dirs:
        raise ValueError(f"No patient directories found in {results_dir}")
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Use progress bar instead of printing each patient name
    try:
        progress_task = log.progress("Loading patient metadata", total=len(patient_dirs))
    except:
        progress_task = None
    
    patient_metadata = {}
    for patient_dir in patient_dirs:
        try:
            metadata = load_patient_metadata(patient_dir)
            patient_metadata[metadata['patient_id']] = metadata
            if progress_task is not None:
                log.update_progress(progress_task, 1)
        except Exception as e:
            warn(f"Failed to load metadata for {patient_dir.name}: {e}")
            if progress_task is not None:
                log.update_progress(progress_task, 1)
    
    if not patient_metadata:
        raise ValueError("No patient metadata could be loaded")
    
    return patient_metadata, summary_df


def build_feature_dataframe_from_summary(
    patient_metadata: Dict,
    feature: str,
) -> pd.DataFrame:
    """
    Build a DataFrame with feature values from SummaryParamTool.xls.
    Includes TCBF calculation as sum of Left ICA + Right ICA.
    
    Args:
        patient_metadata: Dictionary {patient_id: metadata_dict}
        feature: Feature name ('flow', 'pi', etc.)
        
    Returns:
        DataFrame with columns: patient_id, vessel, vessel_name, group, feature
    """
    rows = []
    
    # Map feature names to Excel column names
    feature_to_column = {
        'flow': 'Mean Flow ml/s',
        'flowperheartcycle_val': 'Mean Flow ml/s',
        'flow_rate': 'Mean Flow ml/s',
        'pi': 'Pulsatility Index',
        'pi_val': 'Pulsatility Index',
        'pulsatility': 'Pulsatility Index',
    }
    
    column_name = feature_to_column.get(feature.lower(), feature)
    
    for patient_id, metadata in patient_metadata.items():
        # Get patient directory path
        summary_file = None
        
        # Check if summary_file is stored in metadata
        if 'summary_file' in metadata and metadata['summary_file']:
            summary_file = Path(metadata['summary_file'])
        elif 'patient_dir' in metadata:
            # Use patient_dir to find SummaryParamTool.xls
            patient_dir = Path(metadata['patient_dir'])
            summary_file = patient_dir / 'SummaryParamTool.xls'
        
        if summary_file is None or not summary_file.exists():
            continue
        
        try:
            summary_df = load_summary_data(summary_file.parent)
            if summary_df.empty:
                continue
            
            # Store vessel values for TCBF calculation
            vessel_values = {}
            
            for _, row in summary_df.iterrows():
                vessel_label = row.get('Vessel Label', '')
                if pd.isna(vessel_label):
                    continue
                
                # Map vessel label to code and name using flexible matching
                match = _match_vessel_label(vessel_label)
                if match is None:
                    continue
                vessel_code, vessel_name = match
                
                group = _VESSEL_NAME_TO_GROUP.get(vessel_name, 'Unknown')
                
                # Extract feature value from Excel
                value = row.get(column_name, np.nan)
                
                # Convert flow from mL/s to mL/min if needed
                if 'flow' in feature.lower() and 'Mean Flow' in column_name:
                    if pd.notna(value):
                        value = float(value) * 60.0  # Convert mL/s to mL/min
                
                if pd.notna(value) and np.isfinite(value):
                    vessel_values[vessel_name] = float(value)
                    rows.append({
                        'patient_id': patient_id,
                        'vessel': vessel_name,
                        'vessel_code': vessel_code,
                        'group': group,
                        feature: float(value),
                    })
            
            # Calculate TCBF as sum of all inlet vessels (Left ICA + Right ICA + Basilar)
            if 'flow' in feature.lower():
                lica_value = vessel_values.get('Left ICA', None)
                rica_value = vessel_values.get('Right ICA', None)
                basilar_value = vessel_values.get('Basilar', None)
                
                # Sum all available inlet vessels
                tcbf_parts = []
                if lica_value is not None:
                    tcbf_parts.append(lica_value)
                if rica_value is not None:
                    tcbf_parts.append(rica_value)
                if basilar_value is not None:
                    tcbf_parts.append(basilar_value)
                
                if tcbf_parts:
                    tcbf_value = sum(tcbf_parts)
                    rows.append({
                        'patient_id': patient_id,
                        'vessel': 'TCBF',
                        'vessel_code': 'TCBF',
                        'group': 'TCBF & ICAs',
                        feature: tcbf_value,
                    })
            
        except Exception as e:
            warn(f"Could not load summary data for {patient_id}: {e}")
            continue
    
    if not rows:
        raise ValueError(f"No data found for feature '{feature}' from SummaryParamTool.xls")
    
    df = pd.DataFrame(rows)
    return df


def build_feature_dataframe(
    patient_metadata: Dict,
    feature: str,
    use_summary: bool = False,
) -> pd.DataFrame:
    """
    Build a DataFrame with feature values for all patients and vessels.
    Includes TCBF calculation as sum of Left ICA + Right ICA (for flow features).
    Loads .mat files on-demand and frees memory immediately after extraction.
    
    Args:
        patient_metadata: Dictionary {patient_id: metadata_dict}
        feature: Feature name (e.g., 'flowPerHeartCycle_val', 'PI_val')
        use_summary: If True, load from SummaryParamTool.xls instead of .mat file
        
    Returns:
        DataFrame with columns: patient_id, vessel, vessel_name, group, feature
    """
    if use_summary:
        return build_feature_dataframe_from_summary(patient_metadata, feature)
    
    rows = []
    
    # Use progress bar
    try:
        progress_task = log.progress("Extracting features from .mat files", total=len(patient_metadata))
    except:
        progress_task = None
    
    for patient_id, metadata in patient_metadata.items():
        patient_dir = Path(metadata['patient_dir'])
        locs = metadata.get('LOCs', {})
        
        # Load .mat file, extract features, free memory
        vessel_values_dict = extract_feature_values_from_mat(patient_dir, feature, locs)
        
        # Store vessel values for TCBF calculation
        vessel_values = {}
        
        for vessel_code, value in vessel_values_dict.items():
            vessel_name = _VESSEL_CODE_TO_NAME.get(vessel_code, vessel_code)
            group = _VESSEL_NAME_TO_GROUP.get(vessel_name, 'Unknown')
            
            if value is not None:
                vessel_values[vessel_name] = value
                rows.append({
                    'patient_id': patient_id,
                    'vessel': vessel_name,
                    'vessel_code': vessel_code,
                    'group': group,
                    feature: value,
                })
        
        # Calculate TCBF as sum of all inlet vessels (Left ICA + Right ICA + Basilar)
        if 'flow' in feature.lower():
            lica_value = vessel_values.get('Left ICA', None)
            rica_value = vessel_values.get('Right ICA', None)
            basilar_value = vessel_values.get('Basilar', None)
            
            # Sum all available inlet vessels
            tcbf_parts = []
            if lica_value is not None:
                tcbf_parts.append(lica_value)
            if rica_value is not None:
                tcbf_parts.append(rica_value)
            if basilar_value is not None:
                tcbf_parts.append(basilar_value)
            
            if tcbf_parts:
                tcbf_value = sum(tcbf_parts)
                rows.append({
                    'patient_id': patient_id,
                    'vessel': 'TCBF',
                    'vessel_code': 'TCBF',
                    'group': 'TCBF & ICAs',
                    feature: tcbf_value,
                })
        
        if progress_task is not None:
            log.update_progress(progress_task, 1)
    
    if not rows:
        raise ValueError(f"No data found for feature '{feature}'")
    
    df = pd.DataFrame(rows)
    return df


def build_correlation_matrix(
    patient_metadata: Dict,
    features: List[str],
    normalize_features: bool = False,
) -> pd.DataFrame:
    """
    Build a correlation matrix from patient feature vectors.
    Loads .mat files on-demand and frees memory immediately after extraction.
    
    Args:
        patient_metadata: Dictionary {patient_id: metadata_dict}
        features: List of feature names to include
        normalize_features: If True, normalize each feature type separately (z-score)
        
    Returns:
        DataFrame with correlation matrix (patients x patients)
    """
    import gc
    patient_vectors = {}
    
    # Get consistent vessel ordering across all patients
    all_vessel_codes = set()
    for metadata in patient_metadata.values():
        locs = metadata.get('LOCs', {})
        all_vessel_codes.update(locs.keys())
    vessel_order = sorted(all_vessel_codes)
    
    # Use progress bar
    try:
        progress_task = log.progress("Building correlation matrix from .mat files", total=len(patient_metadata))
    except:
        progress_task = None
    
    for patient_id, metadata in patient_metadata.items():
        try:
            patient_dir = Path(metadata['patient_dir'])
            locs = metadata.get('LOCs', {})
            
            # Load .mat file once per patient
            qvt_data = load_qvt_data(patient_dir, minimal=False)
            data_struct = qvt_data['data_struct']
            
            # Build feature vector for this patient
            vector = []
            
            for vessel_code in vessel_order:
                if vessel_code in locs:
                    loc = locs[vessel_code]
                    # Extract all features for this vessel
                    for feature in features:
                        value = extract_feature_from_loc(data_struct, loc, feature)
                        vector.append(value if value is not None else 0.0)
                else:
                    # Missing vessel: pad with zeros
                    for _ in features:
                        vector.append(0.0)
            
            # Free memory immediately after processing this patient
            del data_struct
            del qvt_data
            gc.collect()
            
            if vector:
                patient_vectors[patient_id] = np.array(vector, dtype=float)
        except Exception as e:
            continue
        finally:
            if progress_task is not None:
                log.update_progress(progress_task, 1)
    
    if not patient_vectors:
        raise ValueError("No patient vectors could be built")
    
    # Build matrix
    patient_ids = sorted(patient_vectors.keys())
    
    # Ensure all vectors have the same length
    vector_length = len(patient_vectors[patient_ids[0]])
    matrix = np.array([
        patient_vectors[pid] if len(patient_vectors[pid]) == vector_length 
        else np.pad(patient_vectors[pid], (0, vector_length - len(patient_vectors[pid])), 'constant')
        for pid in patient_ids
    ])
    
    # Normalize each feature type separately if requested
    if normalize_features and len(features) > 1:
        n_vessels = len(vessel_order)
        normalized_matrix = np.zeros_like(matrix)
        
        for feat_idx, feature in enumerate(features):
            # Extract all values for this feature across all patients and vessels
            feature_indices = [feat_idx + i * len(features) for i in range(n_vessels)]
            feature_values = matrix[:, feature_indices]
            
            # Flatten to get all values for this feature
            flat_values = feature_values.flatten()
            
            # Calculate mean and std (excluding zeros and NaN from missing vessels)
            valid_mask = (flat_values != 0.0) & ~np.isnan(flat_values) & np.isfinite(flat_values)
            if valid_mask.sum() > 1:  # Need at least 2 values for std
                feature_mean = np.mean(flat_values[valid_mask])
                feature_std = np.std(flat_values[valid_mask])
                
                if feature_std > 0:
                    # Normalize (z-score)
                    normalized_feature = (feature_values - feature_mean) / feature_std
                    # Replace zeros and NaN with 0 (don't normalize missing vessels)
                    missing_mask = (feature_values == 0.0) | np.isnan(feature_values) | ~np.isfinite(feature_values)
                    normalized_feature[missing_mask] = 0.0
                    normalized_matrix[:, feature_indices] = normalized_feature
                else:
                    # All values are the same, keep as is
                    normalized_matrix[:, feature_indices] = feature_values
            else:
                # All zeros/NaN or insufficient data, keep as is
                normalized_matrix[:, feature_indices] = feature_values
        
        matrix = normalized_matrix
    
    # Compute correlation
    correlation_matrix = np.corrcoef(matrix)
    
    # Handle NaN values (can occur if all values are the same)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Create DataFrame
    corr_df = pd.DataFrame(correlation_matrix, index=patient_ids, columns=patient_ids)
    
    return corr_df


def build_correlation_matrix_single_feature(
    patient_metadata: Dict,
    feature: str,
) -> pd.DataFrame:
    """
    Build a correlation matrix for a single feature type (flow or PI only).
    
    Args:
        patient_metadata: Dictionary {patient_id: metadata_dict}
        feature: Single feature name (e.g., 'flowPerHeartCycle_val' or 'PI_val')
        
    Returns:
        DataFrame with correlation matrix (patients x patients)
    """
    return build_correlation_matrix(patient_metadata, [feature], normalize_features=False)


def build_flow_timeseries_data(
    patient_metadata: Dict,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build flow timeseries data for all patients.
    Loads .mat files on-demand and frees memory immediately after extraction.
    
    Args:
        patient_metadata: Dictionary {patient_id: metadata_dict}
        
    Returns:
        Dictionary {patient_id: {vessel_code: flow_array}}
    """
    all_flow_data = {}
    
    # Use progress bar
    try:
        progress_task = log.progress("Extracting flow timeseries from .mat files", total=len(patient_metadata))
    except:
        progress_task = None
    
    for patient_id, metadata in patient_metadata.items():
        patient_dir = Path(metadata['patient_dir'])
        locs = metadata.get('LOCs', {})
        
        # Load .mat file, extract flow timeseries, free memory
        patient_flow = extract_flow_timeseries_from_mat(patient_dir, locs)
        
        if patient_flow:
            all_flow_data[patient_id] = patient_flow
        
        if progress_task is not None:
            log.update_progress(progress_task, 1)
    
    return all_flow_data


@click.group()
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
    default='INFO',
    help='Logging level'
)
@click.pass_context
def cli(ctx, log_level):
    """PESA-Brain QVT+ Analysis Plotting CLI."""
    ctx.ensure_object(dict)
    ctx.obj['log_level'] = log_level
    try:
        import debugpy
        debugpy.listen(("localhost", 5678))
    except Exception:
        warn(
            f'debugpy not available. Continuing without debugpy: \n'
            f'Exception: {sys.exc_info()[1]}\n'
        )
    try:
        log.set_level(log_level.upper())
    except Exception:
        pass


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--feature',
    type=str,
    default=None,
    help='Feature to plot. If not specified, plots both flow rate (mL/min) and PI from SummaryParamTool.xls'
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    default=None,
    help='Output path for figure (default: results_dir/violin_{feature}.png)'
)
@click.option(
    '--summary-xlsx',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to summary Excel file with patient metadata'
)
@click.option(
    '--rem-outliers',
    is_flag=True,
    default=False,
    help='Remove outliers based on z-score (|z| > 3) before plotting'
)
@click.pass_context
def violin(ctx, results_dir, feature, output, summary_xlsx, rem_outliers):
    """Create violin plots for a specific feature grouped by vessel groups.
    
    By default (no --feature), plots both volumetric flow rate (mL/min) and 
    pulsatility index from SummaryParamTool.xls.
    
    Examples:
        PB-Analysis violin /path/to/results
        PB-Analysis violin /path/to/results --feature PI_val
        PB-Analysis violin /path/to/results --feature flowPerHeartCycle_val --output plot.png
    """
    try:
        print(f"\n Generating violin plot(s)")
        print(f"   Results directory: {results_dir}")
        
        # Load all patient data
        patient_data, summary_df = load_all_patients(results_dir, summary_xlsx)
        
        # If no feature specified, plot both flow and PI from SummaryParamTool.xls
        if feature is None:
            features_to_plot = [
                ('flow_rate', 'Volumetric Flow Rate (mL/min)'),
                ('pi', 'Pulsatility Index'),
            ]
        else:
            features_to_plot = [(feature, feature)]
        
        for feat, display_name in features_to_plot:
            print(f"\n   Processing feature: {display_name}")
            
            # Use summary data by default for flow and PI
            use_summary = feat.lower() in ['flow', 'pi', 'pulsatility'] or 'flow' in feat.lower() or 'pi' in feat.lower()
            
            try:
                # Build feature dataframe
                df = build_feature_dataframe(patient_data, feat, use_summary=use_summary)
                n_before = len(df.dropna(subset=[feat]))
                print(f"   Loaded {n_before} data points from {len(patient_data)} patients")
                
                # Count outliers if removal is requested (for reporting)
                if rem_outliers:
                    feature_values = df[feat].dropna().values
                    if len(feature_values) > 0:
                        mean_val = np.mean(feature_values)
                        std_val = np.std(feature_values)
                        if std_val > 0:
                            z_scores = np.abs((feature_values - mean_val) / std_val)
                            n_outliers = (z_scores > 6.0).sum()
                            if n_outliers > 0:
                                print(f"   Will remove {n_outliers} outlier(s) (|z| > 3.0)")
                
                # Create plot
                if output is None:
                    output_path = results_dir / f"violin_{feat}.png"
                else:
                    output_path = Path(output)
                    if len(features_to_plot) > 1:
                        # Multiple features: append feature name
                        stem = output_path.stem
                        suffix = output_path.suffix
                        output_path = output_path.parent / f"{stem}_{feat}{suffix}"
                
                # Always save a version without outliers first (to preserve scale)
                output_path_no_outliers = output_path.parent / f"{output_path.stem}_no_outliers{output_path.suffix}"
                fig_no_outliers = plot_violin(df, feat, output_path=output_path_no_outliers, show_patient_dots=False, remove_outliers=True)
                print(f"   [OK] Saved violin plot (no outliers) to {output_path_no_outliers}")
                plt.close(fig_no_outliers)
                
                # Save original violin plot (with or without outliers based on flag)
                fig = plot_violin(df, feat, output_path=output_path, show_patient_dots=False, remove_outliers=rem_outliers)
                print(f"   [OK] Saved violin plot to {output_path}")
                plt.close(fig)
                
                # Save version with patient dots if patient_id is available
                if 'patient_id' in df.columns:
                    # Create output path for patient dots version (with outliers)
                    output_path_dots = output_path.parent / f"{output_path.stem}_with_patient_dots{output_path.suffix}"
                    # Show outliers when displaying patient dots (so outliers can be highlighted)
                    fig_dots = plot_violin(df, feat, output_path=output_path_dots, show_patient_dots=True, remove_outliers=False)
                    print(f"   [OK] Saved violin plot with patient dots to {output_path_dots}")
                    plt.close(fig_dots)
                    
                    # Also save version with patient dots but without outliers
                    output_path_dots_no_outliers = output_path.parent / f"{output_path.stem}_with_patient_dots_no_outliers{output_path.suffix}"
                    fig_dots_no_outliers = plot_violin(df, feat, output_path=output_path_dots_no_outliers, show_patient_dots=True, remove_outliers=True)
                    print(f"   [OK] Saved violin plot with patient dots (no outliers) to {output_path_dots_no_outliers}")
                    plt.close(fig_dots_no_outliers)
            except Exception as e:
                warn(f"   [WARNING] Could not plot {display_name}: {e}")
                continue
        
    except Exception as e:
        err(f"[ERROR] Error generating violin plot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--features',
    type=str,
    default='flowPerHeartCycle_val,PI_val',
    help='Comma-separated list of features for correlation (default: flowPerHeartCycle_val,PI_val)'
)
@click.option(
    '--output',
    type=click.Path(path_type=Path),
    default=None,
    help='Output path for figure (default: results_dir/correlation_heatmap.png)'
)
@click.option(
    '--summary-xlsx',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to summary Excel file with patient metadata (age, sex)'
)
@click.pass_context
def correlation(ctx, results_dir, features, output, summary_xlsx):
    """Create correlation heatmap for patients.
    
    Examples:
        PB-Analysis correlation /path/to/results
        PB-Analysis correlation /path/to/results --features flowPerHeartCycle_val,PI_val,RI_val
    """
    try:
        print(f"\nGenerating correlation heatmaps")
        print(f"   Results directory: {results_dir}")
        
        # Parse features
        feature_list = [f.strip() for f in features.split(',')]
        print(f"   Features: {feature_list}")
        
        # Load all patient data
        patient_data, summary_df = load_all_patients(results_dir, summary_xlsx)
        
        # Prepare patient annotations
        summary_df_idx = None
        annot_cols = []
        if summary_df is not None:
            # Use patient_id as index
            if 'patient_id' in summary_df.columns:
                summary_df_idx = summary_df.set_index('patient_id')
                # Select only annotation columns (age, sex)
                annot_cols = [c for c in summary_df_idx.columns if c in ['age_at_mri', 'sex', 'mri_id']]
                if annot_cols:
                    print(f"   Loaded patient annotations: {list(annot_cols)}")
        
        # 1. Mixed correlation with normalization (Option 1)
        if len(feature_list) > 1:
            print(f"\n   Generating mixed correlation (normalized)...")
            try:
                corr_matrix_mixed = build_correlation_matrix(patient_data, feature_list, normalize_features=True)
                print(f"   Computed mixed correlation matrix: {corr_matrix_mixed.shape}")
                
                if summary_df_idx is not None and annot_cols:
                    patient_annotations_mixed = summary_df_idx[annot_cols].reindex(corr_matrix_mixed.index, fill_value=np.nan)
                else:
                    patient_annotations_mixed = None
                
                if output is None:
                    output_mixed = results_dir / "correlation_heatmap_mixed_normalized.png"
                else:
                    output_path = Path(output)
                    output_mixed = output_path.parent / f"{output_path.stem}_mixed_normalized{output_path.suffix}"
                
                fig = plot_correlation_heatmap(
                    corr_matrix_mixed,
                    patient_annotations=patient_annotations_mixed,
                    output_path=output_mixed,
                    title=f"Patient Correlation Matrix (Mixed: {', '.join(feature_list)}, Normalized)",
                )
                print(f"   [OK] Saved mixed correlation heatmap to {output_mixed}")
                plt.close(fig)
            except Exception as e:
                warn(f"   [WARNING] Could not generate mixed correlation: {e}")
        
        # 2. Separate correlation matrices for each feature (Option 2)
        for feature in feature_list:
            print(f"\n   Generating {feature} correlation...")
            try:
                corr_matrix_single = build_correlation_matrix_single_feature(patient_data, feature)
                print(f"   Computed {feature} correlation matrix: {corr_matrix_single.shape}")
                
                if summary_df_idx is not None and annot_cols:
                    patient_annotations_single = summary_df_idx[annot_cols].reindex(corr_matrix_single.index, fill_value=np.nan)
                else:
                    patient_annotations_single = None
                
                if output is None:
                    output_single = results_dir / f"correlation_heatmap_{feature}.png"
                else:
                    output_path = Path(output)
                    output_single = output_path.parent / f"{output_path.stem}_{feature}{output_path.suffix}"
                
                fig = plot_correlation_heatmap(
                    corr_matrix_single,
                    patient_annotations=patient_annotations_single,
                    output_path=output_single,
                    title=f"Patient Correlation Matrix ({feature})",
                )
                print(f"   [OK] Saved {feature} correlation heatmap to {output_single}")
                plt.close(fig)
            except Exception as e:
                warn(f"   [WARNING] Could not generate {feature} correlation: {e}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        err(f"[ERROR] Error generating correlation heatmap: {e}")
        sys.exit(1)


@cli.command(name="clinical-correlation")
@click.argument("results_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--clinical-summary",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to clinical summary file (.csv or .xlsx) with patient_id and variables",
)
@click.option(
    "--features",
    type=str,
    default="flow,pi",
    help="Comma-separated list of features to correlate (default: flow,pi)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for correlation results (default: results_dir/clinical_correlations)",
)
@click.option(
    "--clinical-columns",
    type=str,
    default=None,
    help="Optional comma-separated list of clinical columns to include",
)
@click.option(
    "--sigma-threshold",
    type=float,
    default=6.0,
    show_default=True,
    help="Sigma threshold for outlier removal in scatter plots",
)
@click.option(
    "--polar-clinical-vars",
    type=str,
    default=None,
    help="Comma-separated list of clinical vars for polar plots (default: all)",
)
@click.pass_context
def clinical_correlation(
    ctx,
    results_dir,
    clinical_summary,
    features,
    output_dir,
    clinical_columns,
    sigma_threshold,
    polar_clinical_vars,
):
    """Correlate clinical variables with flow/PI vessel data and save outputs."""
    try:
        print(f"\nGenerating clinical correlations")
        print(f"   Results directory: {results_dir}")
        print(f"   Clinical summary: {clinical_summary}")

        if output_dir is None:
            output_dir = Path(results_dir) / "clinical_correlations"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_list = [f.strip() for f in features.split(",") if f.strip()]
        clinical_columns_list = None
        if clinical_columns:
            clinical_columns_list = [
                c.strip().lower().replace(" ", "_")
                for c in clinical_columns.split(",")
                if c.strip()
            ]

        patient_data, _ = load_all_patients(results_dir)
        clinical_df = load_clinical_summary(clinical_summary)

        # # EDA overview (using flow feature table as baseline)
        # try:
        #     eda_table = build_patient_feature_table_from_summary(patient_data, "flow")
        #     save_eda_overview(
        #         patient_metadata=patient_data,
        #         clinical_df=clinical_df,
        #         feature_table=eda_table,
        #         output_dir=output_dir / "eda",
        #     )
        # except Exception as e:
        #     warn(f"   [WARNING] Could not generate EDA overview: {e}")

        flow_table = None
        pi_table = None

        polar_vars_list = None
        if polar_clinical_vars:
            polar_vars_list = [
                c.strip().lower().replace(" ", "_")
                for c in polar_clinical_vars.split(",")
                if c.strip()
            ]

        for feature in feature_list:
            feature_key = feature.lower()
            if "flow" in feature_key:
                feature_label = "Flow (mL/min)"
            elif "pi" in feature_key or "pulsatility" in feature_key:
                feature_label = "PI"
            else:
                feature_label = feature

            print(f"\n   Processing feature: {feature_label}")
            feature_table = build_patient_feature_table_from_summary(patient_data, feature)
            if "pi" in feature_key or "pulsatility" in feature_key:
                if "TCBF" in feature_table.columns:
                    feature_table = feature_table.drop(columns=["TCBF"])

            if "flow" in feature_key:
                flow_table = feature_table
            if "pi" in feature_key or "pulsatility" in feature_key:
                pi_table = feature_table

            feature_dir = output_dir / feature_key
            feature_dir.mkdir(parents=True, exist_ok=True)

            save_clinical_correlation_outputs(
                clinical_df=clinical_df,
                feature_table=feature_table,
                feature_label=feature_label,
                output_dir=feature_dir,
                clinical_columns=clinical_columns_list,
                sigma_threshold=sigma_threshold,
            )

            # Spatial (vessel-vessel) correlation summary
            save_vessel_spatial_correlation_outputs(
                feature_table=feature_table,
                output_dir=feature_dir / "vessel_correlations",
                feature_label=feature_label,
            )

            # Per-clinical-variable polar plots
            try:
                save_polar_clinical_correlations(
                    clinical_df=clinical_df,
                    feature_table=feature_table,
                    feature_label=feature_label,
                    output_dir=feature_dir / "polar_clinical",
                    clinical_columns=polar_vars_list or clinical_columns_list,
                )
            except Exception as e:
                warn(f"   [WARNING] Could not generate polar clinical plots for {feature_label}: {e}")

        if flow_table is not None and pi_table is not None:
            save_flow_pi_cross_correlation(
                flow_table=flow_table,
                pi_table=pi_table,
                output_dir=output_dir / "flow_vs_pi",
            )

            # Difference polar: flow vs PI correlations per clinical variable
            try:
                save_polar_difference_correlations(
                    clinical_df=clinical_df,
                    flow_table=flow_table,
                    pi_table=pi_table,
                    output_dir=output_dir / "polar_difference",
                    clinical_columns=polar_vars_list or clinical_columns_list,
                )
            except Exception as e:
                warn(f"   [WARNING] Could not generate polar difference plots: {e}")

        print(f"\n   [OK] Clinical correlations saved to {output_dir}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        err(f"[ERROR] Error generating clinical correlations: {e}")
        sys.exit(1)


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--patient-id',
    type=str,
    default=None,
    help='Specific patient ID to plot (default: all patients, saved individually)'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=None,
    help='Output directory for plots (default: results_dir/flow_timeseries or results_dir)'
)
@click.option(
    '--summary-xlsx',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to summary Excel file with patient metadata'
)
@click.pass_context
def flow(ctx, results_dir, patient_id, output_dir, summary_xlsx):
    """Plot flow timeseries for patient(s).
    
    When plotting all patients, each patient is saved to a separate file.
    
    Examples:
        PB-Analysis flow /path/to/results
        PB-Analysis flow /path/to/results --patient-id PESA001 --output-dir /path/to/output
    """
    try:
        print(f"\n Generating flow timeseries plot(s)")
        print(f"   Results directory: {results_dir}")
        
        # Load all patient data
        patient_data, summary_df = load_all_patients(results_dir, summary_xlsx)
        
        # Build flow timeseries data
        all_flow_data = build_flow_timeseries_data(patient_data)
        
        # Set output directory
        if output_dir is None:
            if patient_id:
                output_dir = results_dir
            else:
                output_dir = results_dir / "flow_timeseries"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if patient_id:
            # Plot single patient
            if patient_id not in all_flow_data:
                raise ValueError(f"Patient {patient_id} not found in data")
            
            flow_data = all_flow_data[patient_id]
            print(f"   Plotting {len(flow_data)} vessels for {patient_id}")
            
            output = output_dir / f"flow_timeseries_{patient_id}.png"
            
            # Get nframes from patient's data
            nframes = len(list(flow_data.values())[0]) if flow_data else 15
            
            fig = plot_flow_timeseries(flow_data, patient_id, output_path=output, nframes=nframes)
            print(f"   [OK] Saved flow timeseries to {output}")
            plt.close(fig)
        else:
            # Plot all patients - save individually
            print(f"   Plotting {len(all_flow_data)} patients (saving individually)")
            
            # Get nframes from first patient's data
            first_patient = list(all_flow_data.keys())[0]
            nframes = len(list(all_flow_data[first_patient].values())[0]) if all_flow_data[first_patient] else 15
            
            # Calculate global y-limits for consistent scaling
            all_flow_values = []
            for patient_flow in all_flow_data.values():
                for flow_array in patient_flow.values():
                    if len(flow_array) == nframes:
                        all_flow_values.extend(flow_array.tolist())
            
            if all_flow_values:
                y_min = min(all_flow_values)
                y_max = max(all_flow_values)
                y_range = y_max - y_min
                ylim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                ylim = None
            
            # Save individual plots for each patient
            saved_count = 0
            for pid, flow_data in all_flow_data.items():
                output = output_dir / f"flow_timeseries_{pid}.png"
                try:
                    fig = plot_flow_timeseries(
                        flow_data, 
                        pid, 
                        output_path=output, 
                        nframes=nframes,
                        ylim=ylim
                    )
                    plt.close(fig)
                    saved_count += 1
                except Exception as e:
                    warn(f"   [WARNING] Could not save plot for {pid}: {e}")
            
            print(f"   [OK] Saved {saved_count} flow timeseries plot(s) to {output_dir}")
        
    except Exception as e:
        err(f"[ERROR] Error generating flow timeseries: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--patient-id',
    type=str,
    default=None,
    help='Specific patient ID to plot (default: all patients, saved individually)'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=None,
    help='Output directory for plots (default: results_dir/polar_plots or results_dir)'
)
@click.option(
    '--summary-xlsx',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to summary Excel file with patient metadata'
)
@click.option(
    '--combined/--no-combined',
    default=True,
    help='Also create a combined correlation plot (default: True)'
)
@click.pass_context
def polar(ctx, results_dir, patient_id, output_dir, summary_xlsx, combined):
    """Create polar flow plots for patients.
    
    Creates circular plots showing flow values for each vessel section.
    Per-patient plots use individual scales, combined plot shows correlation.
    
    Examples:
        PB-Analysis polar /path/to/results
        PB-Analysis polar /path/to/results --patient-id PESA001
        PB-Analysis polar /path/to/results --no-combined
    """
    try:
        print(f"\nGenerating polar flow plots")
        print(f"   Results directory: {results_dir}")
        
        # Load all patient data
        patient_data, summary_df = load_all_patients(results_dir, summary_xlsx)
        
        # Set output directory
        if output_dir is None:
            if patient_id:
                output_dir = results_dir
            else:
                output_dir = results_dir / "polar_plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build flow data from summary (use summary data for flow values)
        all_patient_flow_data = {}
        
        for pid, metadata in patient_data.items():
            try:
                # Get summary data
                summary_file = None
                if 'summary_file' in metadata and metadata['summary_file']:
                    summary_file = Path(metadata['summary_file'])
                elif 'patient_dir' in metadata:
                    patient_dir = Path(metadata['patient_dir'])
                    summary_file = patient_dir / 'SummaryParamTool.xls'
                
                if summary_file is None or not summary_file.exists():
                    continue
                
                summary_df_patient = load_summary_data(summary_file.parent)
                if summary_df_patient.empty:
                    continue
                
                # Extract flow values for all vessels
                patient_flows = {}
                
                # Calculate TCBF (sum of all vessels)
                all_flows = []
                for _, row in summary_df_patient.iterrows():
                    vessel_label = row.get('Vessel Label', '')
                    if pd.isna(vessel_label):
                        continue
                    
                    # Check if this vessel is in our mapping
                    vessel_name = None
                    for code, name in _VESSEL_CODE_TO_NAME.items():
                        if name == vessel_label:
                            vessel_name = name
                            break
                    
                    if vessel_name is None:
                        continue
                    
                    # Get flow value
                    flow_value = row.get('Mean Flow ml/s', np.nan)
                    if pd.notna(flow_value):
                        flow_value = float(flow_value) * 60.0  # Convert to mL/min
                        patient_flows[vessel_name] = flow_value
                        all_flows.append(flow_value)
                
                # Add TCBF (sum of all available vessels)
                if all_flows:
                    patient_flows['TCBF'] = sum(all_flows)
                
                all_patient_flow_data[pid] = patient_flows
                
            except Exception as e:
                warn(f"   [WARNING] Could not load flow data for {pid}: {e}")
                continue
        
        if not all_patient_flow_data:
            raise ValueError("No patient flow data could be loaded")
        
        # Filter by patient_id if specified
        if patient_id:
            if patient_id not in all_patient_flow_data:
                raise ValueError(f"Patient {patient_id} not found in data")
            all_patient_flow_data = {patient_id: all_patient_flow_data[patient_id]}
        
        # Generate per-patient plots for each feature (flow and PI by default)
        features_to_plot = [
            ('flow', 'Mean Flow ml/s', 'Flow', 'mL/min'),
            ('pi', 'Pulsatility Index', 'PI', 'PI'),
        ]
        
        saved_count = 0
        for feat_key, excel_col, feat_name, unit in features_to_plot:
            # Build patient data for this feature
            all_patient_feature_data = {}
            for pid, metadata in patient_data.items():
                try:
                    summary_file = None
                    if 'summary_file' in metadata and metadata['summary_file']:
                        summary_file = Path(metadata['summary_file'])
                    elif 'patient_dir' in metadata:
                        patient_dir = Path(metadata['patient_dir'])
                        summary_file = patient_dir / 'SummaryParamTool.xls'
                    
                    if summary_file is None or not summary_file.exists():
                        continue
                    
                    summary_df_patient = load_summary_data(summary_file.parent)
                    if summary_df_patient.empty:
                        continue
                    
                    patient_feature_values = {}
                    all_values = []
                    
                    for _, row in summary_df_patient.iterrows():
                        vessel_label = row.get('Vessel Label', '')
                        if pd.isna(vessel_label):
                            continue
                        
                        # Use flexible matching for vessel labels
                        match = _match_vessel_label(vessel_label)
                        if match is None:
                            continue
                        _, vessel_name = match
                        
                        # Get feature value
                        value = row.get(excel_col, np.nan)
                        if pd.notna(value):
                            if feat_key == 'flow':
                                value = float(value) * 60.0  # Convert to mL/min
                            else:
                                value = float(value)
                            patient_feature_values[vessel_name] = value
                            all_values.append(value)
                    
                    # Add TCBF (sum for flow, mean for PI)
                    if all_values:
                        if feat_key == 'flow':
                            patient_feature_values['TCBF'] = sum(all_values)
                        else:
                            patient_feature_values['TCBF'] = np.mean(all_values)
                    
                    all_patient_feature_data[pid] = patient_feature_values
                    
                except Exception as e:
                    warn(f"   [WARNING] Could not load {feat_name} data for {pid}: {e}")
                    continue
            
            # Filter by patient_id if specified
            if patient_id:
                if patient_id not in all_patient_feature_data:
                    continue
                all_patient_feature_data = {patient_id: all_patient_feature_data[patient_id]}
            
            # Generate plots for this feature
            for pid, patient_feature_values in all_patient_feature_data.items():
                try:
                    output_path = output_dir / f"polar_{feat_key}_{pid}.png"
                    fig = plot_polar_flow(
                        patient_feature_values,
                        pid,
                        feature_name=feat_name,
                        output_path=output_path,
                    )
                    plt.close(fig)
                    saved_count += 1
                except Exception as e:
                    warn(f"   [WARNING] Could not save polar {feat_name} plot for {pid}: {e}")
        
        print(f"   [OK] Saved {saved_count} polar plot(s) to {output_dir}")
        
        # Generate combined correlation plot if requested (for flow only)
        if combined:
            # Rebuild flow data for correlation
            all_patient_flow_for_corr = {}
            for pid, metadata in patient_data.items():
                try:
                    summary_file = None
                    if 'summary_file' in metadata and metadata['summary_file']:
                        summary_file = Path(metadata['summary_file'])
                    elif 'patient_dir' in metadata:
                        patient_dir = Path(metadata['patient_dir'])
                        summary_file = patient_dir / 'SummaryParamTool.xls'
                    
                    if summary_file is None or not summary_file.exists():
                        continue
                    
                    summary_df_patient = load_summary_data(summary_file.parent)
                    if summary_df_patient.empty:
                        continue
                    
                    patient_flows = {}
                    all_flows = []
                    
                    for _, row in summary_df_patient.iterrows():
                        vessel_label = row.get('Vessel Label', '')
                        if pd.isna(vessel_label):
                            continue
                        
                        # Use flexible matching for vessel labels
                        match = _match_vessel_label(vessel_label)
                        if match is None:
                            continue
                        _, vessel_name = match
                        
                        flow_value = row.get('Mean Flow ml/s', np.nan)
                        if pd.notna(flow_value):
                            flow_value = float(flow_value) * 60.0
                            patient_flows[vessel_name] = flow_value
                            all_flows.append(flow_value)
                    
                    if all_flows:
                        patient_flows['TCBF'] = sum(all_flows)
                    
                    all_patient_flow_for_corr[pid] = patient_flows
                    
                except Exception as e:
                    continue
            
            if len(all_patient_flow_for_corr) > 1:
                try:
                    output_path = output_dir / "polar_flow_correlation.png"
                    fig = plot_polar_flow_correlation(
                        all_patient_flow_for_corr,
                        output_path=output_path,
                    )
                    plt.close(fig)
                    print(f"   [OK] Saved combined correlation plot to {output_path}")
                except Exception as e:
                    warn(f"   [WARNING] Could not save combined correlation plot: {e}")
        
    except Exception as e:
        err(f"[ERROR] Error generating polar plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=None,
    help='Output directory for plots (default: results_dir/crosssections)'
)
@click.option(
    '--summary-xlsx',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to summary Excel file with patient metadata'
)
@click.option(
    '--cross-section-type',
    type=click.Choice(['MAG', 'CD', 'VEL', 'combined'], case_sensitive=False),
    default='combined',
    help='Type of cross-section to display (default: combined)'
)
@click.option(
    '--max-patients-per-figure',
    type=int,
    default=6,
    help='Maximum number of patients per figure page (default: 6)'
)
@click.pass_context
def crosssections(ctx, results_dir, output_dir, summary_xlsx, cross_section_type, max_patients_per_figure):
    """Create mosaic plots showing cross-sections at vessel LOCs with highlighted measurement pixels.
    
    Displays cross-sections for each vessel LOC as a mosaic (rows=patients, cols=vessels).
    Measurement pixels are highlighted with colored overlay and contour outline.
    
    Examples:
        PB-Analysis crosssections /path/to/results
        PB-Analysis crosssections /path/to/results --cross-section-type MAG
        PB-Analysis crosssections /path/to/results --cross-section-type combined --max-patients-per-figure 4
    """
    try:
        print(f"\nGenerating cross-sections mosaic plots")
        print(f"   Results directory: {results_dir}")
        print(f"   Cross-section type: {cross_section_type}")
        
        # Load all patient data
        patient_data, summary_df = load_all_patients(results_dir, summary_xlsx)
        
        # Set output directory
        if output_dir is None:
            output_dir = results_dir / "crosssections"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cross-sections plot
        output_path = output_dir / f"crosssections_{cross_section_type.lower()}.png"
        
        figures = plot_crosssections(
            patient_data,
            output_path=output_path,
            cross_section_type=cross_section_type,
            max_patients_per_figure=max_patients_per_figure,
        )
        
        # Close all figures
        for fig in figures:
            plt.close(fig)
        
        print(f"   [OK] Saved {len(figures)} cross-sections plot(s) to {output_dir}")
        
    except Exception as e:
        err(f"[ERROR] Error generating cross-sections plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=None,
    help='Output directory for plots (default: results_dir/crosssections)'
)
@click.option(
    '--summary-xlsx',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to summary Excel file with patient metadata'
)
@click.option(
    '--cross-section-type',
    type=click.Choice(['MAG', 'CD', 'VEL', 'combined'], case_sensitive=False),
    default='combined',
    help='Type of cross-section to display (default: combined)'
)
@click.option(
    '--max-patients-per-figure',
    type=int,
    default=6,
    help='Maximum number of patients per figure page (default: 6)'
)
@click.pass_context
def crosssections(ctx, results_dir, output_dir, summary_xlsx, cross_section_type, max_patients_per_figure):
    """Create mosaic plots showing cross-sections at vessel LOCs with highlighted measurement pixels.
    
    Displays cross-sections for each vessel LOC as a mosaic (rows=patients, cols=vessels).
    Measurement pixels are highlighted with colored overlay and contour outline.
    
    Examples:
        PB-Analysis crosssections /path/to/results
        PB-Analysis crosssections /path/to/results --cross-section-type MAG
        PB-Analysis crosssections /path/to/results --cross-section-type combined --max-patients-per-figure 4
    """
    try:
        print(f"\nGenerating cross-sections mosaic plots")
        print(f"   Results directory: {results_dir}")
        print(f"   Cross-section type: {cross_section_type}")
        
        # Load all patient data
        patient_data, summary_df = load_all_patients(results_dir, summary_xlsx)
        
        # Set output directory
        if output_dir is None:
            output_dir = results_dir / "crosssections"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cross-sections plot
        output_path = output_dir / f"crosssections_{cross_section_type.lower()}.png"
        
        figures = plot_crosssections(
            patient_data,
            output_path=output_path,
            cross_section_type=cross_section_type,
            max_patients_per_figure=max_patients_per_figure,
        )
        
        # Close all figures
        for fig in figures:
            plt.close(fig)
        
        print(f"   [OK] Saved {len(figures)} cross-sections plot(s) to {output_dir}")
        
    except Exception as e:
        err(f"[ERROR] Error generating cross-sections plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('results_dir', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default=None,
    help='Output directory for all plots (default: results_dir/analysis_plots)'
)
@click.option(
    '--summary-xlsx',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to summary Excel file with patient metadata'
)
@click.option(
    '--features',
    type=str,
    default='flowPerHeartCycle_val,PI_val',
    help='Comma-separated list of features for correlation (default: flowPerHeartCycle_val,PI_val)'
)
@click.pass_context
def all(ctx, results_dir, output_dir, summary_xlsx, features):
    """Run all analysis plots and save to output directory.
    
    Generates:
    - Violin plots for flow rate (mL/min) and pulsatility index
    - Correlation heatmap
    - Flow timeseries plots for all patients
    - Polar plots for flow and PI (per patient and combined correlation)
    - Cross-sections mosaic plots (MAG, CD, VEL, combined)
    - Tree/dendrogram plots for flow and PI (per patient)
    
    Examples:
        PB-Analysis all /path/to/results
        PB-Analysis all /path/to/results --output-dir /path/to/output
    """
    try:
        print(f"\nRunning all analysis plots")
        print(f"   Results directory: {results_dir}")
        
        # Set output directory
        if output_dir is None:
            output_dir = results_dir / "analysis_plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Output directory: {output_dir}")
        
        # Load all patient data
        patient_data, summary_df = load_all_patients(results_dir, summary_xlsx)
        print(f'Total patients loaded: {len(patient_data)} / {len(list(results_dir.glob("PESA*")))}')
        
        # 1. Violin plots (flow and PI from SummaryParamTool.xls)
        print(f"\n1. Generating violin plots...")
        try:
            for feat, display_name in [('flow', 'Volumetric Flow Rate (mL/min)'), ('pi', 'Pulsatility Index')]:
                print(f"   Processing: {display_name}")
                df = build_feature_dataframe(patient_data, feat, use_summary=True)
                output_path = output_dir / f"violin_{feat}.png"
                
                # Always save a version without outliers first (to preserve scale)
                output_path_no_outliers = output_dir / f"violin_{feat}_no_outliers.png"
                fig_no_outliers = plot_violin(df, feat, output_path=output_path_no_outliers, show_patient_dots=False, remove_outliers=True)
                plt.close(fig_no_outliers)
                print(f"   [OK] Saved violin plot (no outliers) to {output_path_no_outliers}")
                
                # Save original violin plot (with outliers)
                fig = plot_violin(df, feat, output_path=output_path, show_patient_dots=False, remove_outliers=False)
                plt.close(fig)
                print(f"   [OK] Saved to {output_path}")
                
                # Save version with patient dots if patient_id is available
                if 'patient_id' in df.columns:
                    # Version with patient dots and outliers
                    output_path_dots = output_dir / f"violin_{feat}_with_patient_dots.png"
                    fig_dots = plot_violin(df, feat, output_path=output_path_dots, show_patient_dots=True, remove_outliers=False)
                    plt.close(fig_dots)
                    print(f"   [OK] Saved to {output_path_dots}")
                    
                    # Version with patient dots but without outliers
                    output_path_dots_no_outliers = output_dir / f"violin_{feat}_with_patient_dots_no_outliers.png"
                    fig_dots_no_outliers = plot_violin(df, feat, output_path=output_path_dots_no_outliers, show_patient_dots=True, remove_outliers=True)
                    plt.close(fig_dots_no_outliers)
                    print(f"   [OK] Saved to {output_path_dots_no_outliers}")
        except Exception as e:
            warn(f"   [WARNING] Error generating violin plots: {e}")
        
        # 2. Correlation heatmaps (mixed normalized + separate for each feature)
        print(f"\n2. Generating correlation heatmaps...")
        try:
            feature_list = [f.strip() for f in features.split(',')]
            
            # Prepare patient annotations
            summary_df_idx = None
            annot_cols = []
            if summary_df is not None and 'patient_id' in summary_df.columns:
                summary_df_idx = summary_df.set_index('patient_id')
                annot_cols = [c for c in summary_df_idx.columns if c in ['age_at_mri', 'sex', 'mri_id']]
                if annot_cols:
                    print(f"   Loaded patient annotations: {list(annot_cols)}")
            
            # 2a. Mixed correlation with normalization (Option 1)
            if len(feature_list) > 1:
                try:
                    corr_matrix_mixed = build_correlation_matrix(patient_data, feature_list, normalize_features=True)
                    if summary_df_idx is not None and annot_cols:
                        patient_annotations_mixed = summary_df_idx[annot_cols].reindex(corr_matrix_mixed.index, fill_value=np.nan)
                    else:
                        patient_annotations_mixed = None
                    
                    output_path = output_dir / "correlation_heatmap_mixed_normalized.png"
                    fig = plot_correlation_heatmap(
                        corr_matrix_mixed,
                        patient_annotations=patient_annotations_mixed,
                        output_path=output_path,
                        title=f"Patient Correlation Matrix (Mixed: {', '.join(feature_list)}, Normalized)",
                    )
                    plt.close(fig)
                    print(f"   [OK] Saved mixed correlation to {output_path}")
                except Exception as e:
                    warn(f"   [WARNING] Could not generate mixed correlation: {e}")
            
            # 2b. Separate correlation for each feature (Option 2)
            for feature in feature_list:
                try:
                    corr_matrix_single = build_correlation_matrix_single_feature(patient_data, feature)
                    if summary_df_idx is not None and annot_cols:
                        patient_annotations_single = summary_df_idx[annot_cols].reindex(corr_matrix_single.index, fill_value=np.nan)
                    else:
                        patient_annotations_single = None
                    
                    output_path = output_dir / f"correlation_heatmap_{feature}.png"
                    fig = plot_correlation_heatmap(
                        corr_matrix_single,
                        patient_annotations=patient_annotations_single,
                        output_path=output_path,
                        title=f"Patient Correlation Matrix ({feature})",
                    )
                    plt.close(fig)
                    print(f"   [OK] Saved {feature} correlation to {output_path}")
                except Exception as e:
                    warn(f"   [WARNING] Could not generate {feature} correlation: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            warn(f"   [WARNING] Error generating correlation heatmaps: {e}")
        
        # 3. Flow timeseries (save individually per patient)
        print(f"\n3. Generating flow timeseries plots (per patient)...")
        try:
            all_flow_data = build_flow_timeseries_data(patient_data)
            first_patient = list(all_flow_data.keys())[0]
            nframes = len(list(all_flow_data[first_patient].values())[0]) if all_flow_data[first_patient] else 15
            
            # Calculate global y-limits for consistent scaling
            all_flow_values = []
            for patient_flow in all_flow_data.values():
                for flow_array in patient_flow.values():
                    if len(flow_array) == nframes:
                        all_flow_values.extend(flow_array.tolist())
            
            if all_flow_values:
                y_min = min(all_flow_values)
                y_max = max(all_flow_values)
                y_range = y_max - y_min
                ylim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            else:
                ylim = None
            
            # Save individual plots for each patient
            flow_output_dir = output_dir / "flow_timeseries"
            flow_output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            for pid, flow_data in all_flow_data.items():
                output_path = flow_output_dir / f"flow_timeseries_{pid}.png"
                try:
                    fig = plot_flow_timeseries(
                        flow_data,
                        pid,
                        output_path=output_path,
                        nframes=nframes,
                        ylim=ylim
                    )
                    plt.close(fig)
                    saved_count += 1
                except Exception as e:
                    warn(f"   [WARNING] Could not save plot for {pid}: {e}")
            
            print(f"   [OK] Saved {saved_count} flow timeseries plot(s) to {flow_output_dir}")
        except Exception as e:
            warn(f"   [WARNING] Error generating flow timeseries: {e}")
        
        # 4. Polar plot GIFs (animated time-resolved)
        print(f"\n4. Generating polar plot GIFs (animated)...")
        try:
            gif_output_dir = output_dir / "polar_gifs"
            gif_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build time-resolved flow data
            all_flow_data = build_flow_timeseries_data(patient_data)
            
            if all_flow_data:
                # Get nframes from first patient
                first_patient = list(all_flow_data.keys())[0]
                nframes = len(list(all_flow_data[first_patient].values())[0]) if all_flow_data[first_patient] else 15
                
                saved_count = 0
                for pid, flow_data in all_flow_data.items():
                    try:
                        output_path = gif_output_dir / f"polar_animation_flow_{pid}.gif"
                        plot_polar_flow_animation(
                            flow_data,
                            pid,
                            feature_name='Flow',
                            output_path=output_path,
                            nframes=nframes,
                        )
                        saved_count += 1
                    except Exception as e:
                        warn(f"   [WARNING] Could not create GIF for {pid}: {e}")
                
                print(f"   [OK] Saved {saved_count} GIF file(s) to {gif_output_dir}")
            else:
                warn(f"   [WARNING] No time-resolved flow data found for GIF generation")
        except Exception as e:
            warn(f"   [WARNING] Error generating polar plot GIFs: {e}")
        
        # 5. Polar plots (flow and PI)
        print(f"\n5. Generating polar plots (per patient)...")
        try:
            polar_output_dir = output_dir / "polar_plots"
            polar_output_dir.mkdir(parents=True, exist_ok=True)
            
            features_to_plot = [
                ('flow', 'Mean Flow ml/s', 'Flow', 'mL/min'),
                ('pi', 'Pulsatility Index', 'PI', 'PI'),
            ]
            
            saved_count = 0
            for feat_key, excel_col, feat_name, unit in features_to_plot:
                # Build patient data for this feature
                all_patient_feature_data = {}
                for pid, metadata in patient_data.items():
                    try:
                        summary_file = None
                        if 'summary_file' in metadata and metadata['summary_file']:
                            summary_file = Path(metadata['summary_file'])
                        elif 'patient_dir' in metadata:
                            patient_dir = Path(metadata['patient_dir'])
                            summary_file = patient_dir / 'SummaryParamTool.xls'
                        
                        if summary_file is None or not summary_file.exists():
                            continue
                        
                        summary_df_patient = load_summary_data(summary_file.parent)
                        if summary_df_patient.empty:
                            continue
                        
                        patient_feature_values = {}
                        all_values = []
                        
                        for _, row in summary_df_patient.iterrows():
                            vessel_label = row.get('Vessel Label', '')
                            if pd.isna(vessel_label):
                                continue
                            
                            vessel_name = None
                            for code, name in _VESSEL_CODE_TO_NAME.items():
                                if name == vessel_label:
                                    vessel_name = name
                                    break
                            
                            if vessel_name is None:
                                continue
                            
                            value = row.get(excel_col, np.nan)
                            if pd.notna(value):
                                if feat_key == 'flow':
                                    value = float(value) * 60.0  # Convert to mL/min
                                else:
                                    value = float(value)
                                patient_feature_values[vessel_name] = value
                                all_values.append(value)
                        
                        # Add TCBF (sum for flow, mean for PI)
                        if all_values:
                            if feat_key == 'flow':
                                patient_feature_values['TCBF'] = sum(all_values)
                            else:
                                patient_feature_values['TCBF'] = np.mean(all_values)
                        
                        all_patient_feature_data[pid] = patient_feature_values
                        
                    except Exception as e:
                        warn(f"   [WARNING] Could not load {feat_name} data for {pid}: {e}")
                        continue
                
                # Generate plots for this feature
                for pid, patient_feature_values in all_patient_feature_data.items():
                    try:
                        output_path = polar_output_dir / f"polar_{feat_key}_{pid}.png"
                        fig = plot_polar_flow(
                            patient_feature_values,
                            pid,
                            feature_name=feat_name,
                            output_path=output_path,
                        )
                        plt.close(fig)
                        saved_count += 1
                    except Exception as e:
                        warn(f"   [WARNING] Could not save polar {feat_name} plot for {pid}: {e}")
            
            print(f"   [OK] Saved {saved_count} polar plot(s) to {polar_output_dir}")
            
            # Generate combined correlation plot (flow only)
            all_patient_flow_for_corr = {}
            for pid, metadata in patient_data.items():
                try:
                    summary_file = None
                    if 'summary_file' in metadata and metadata['summary_file']:
                        summary_file = Path(metadata['summary_file'])
                    elif 'patient_dir' in metadata:
                        patient_dir = Path(metadata['patient_dir'])
                        summary_file = patient_dir / 'SummaryParamTool.xls'
                    
                    if summary_file is None or not summary_file.exists():
                        continue
                    
                    summary_df_patient = load_summary_data(summary_file.parent)
                    if summary_df_patient.empty:
                        continue
                    
                    patient_flows = {}
                    all_flows = []
                    
                    for _, row in summary_df_patient.iterrows():
                        vessel_label = row.get('Vessel Label', '')
                        if pd.isna(vessel_label):
                            continue
                        
                        # Use flexible matching for vessel labels
                        match = _match_vessel_label(vessel_label)
                        if match is None:
                            continue
                        _, vessel_name = match
                        
                        flow_value = row.get('Mean Flow ml/s', np.nan)
                        if pd.notna(flow_value):
                            flow_value = float(flow_value) * 60.0
                            patient_flows[vessel_name] = flow_value
                            all_flows.append(flow_value)
                    
                    if all_flows:
                        patient_flows['TCBF'] = sum(all_flows)
                    
                    all_patient_flow_for_corr[pid] = patient_flows
                    
                except Exception as e:
                    continue
            
            if len(all_patient_flow_for_corr) > 1:
                try:
                    output_path = polar_output_dir / "polar_flow_correlation.png"
                    fig = plot_polar_flow_correlation(
                        all_patient_flow_for_corr,
                        output_path=output_path,
                    )
                    plt.close(fig)
                    print(f"   [OK] Saved combined correlation plot to {output_path}")
                except Exception as e:
                    warn(f"   [WARNING] Could not save combined correlation plot: {e}")
                    
        except Exception as e:
            warn(f"   [WARNING] Error generating polar plots: {e}")
        
        # 6. Cross-sections mosaic plots
        # print(f"\n6. Generating cross-sections mosaic plots...")
        # try:
        #     crosssections_output_dir = output_dir / "crosssections"
        #     crosssections_output_dir.mkdir(parents=True, exist_ok=True)
            
        #     # Generate cross-sections for different types
        #     cross_section_types = ['MAG', 'CD', 'VEL', 'combined']
            
        #     saved_count = 0
        #     for cs_type in cross_section_types:
        #         try:
        #             output_path = crosssections_output_dir / f"crosssections_{cs_type.lower()}.png"
        #             figures = plot_crosssections(
        #                 patient_data,
        #                 output_path=output_path,
        #                 cross_section_type=cs_type,
        #                 max_patients_per_figure=6,
        #             )
        #             # Close all figures
        #             for fig in figures:
        #                 plt.close(fig)
        #             saved_count += len(figures)
        #         except Exception as e:
        #             warn(f"   [WARNING] Could not generate {cs_type} cross-sections: {e}")
            
        #     print(f"   [OK] Saved {saved_count} cross-sections plot(s) to {crosssections_output_dir}")
        # except Exception as e:
        #     warn(f"   [WARNING] Error generating cross-sections plots: {e}")
        
        # 7. Tree/dendrogram plots (flow and PI)
        print(f"\n7. Generating tree/dendrogram plots (per patient)...")
        try:
            tree_output_dir = output_dir / "tree_plots"
            tree_output_dir.mkdir(parents=True, exist_ok=True)
            
            features_to_plot = [
                ('flow', 'Mean Flow ml/s', 'Flow'),
                ('pi', 'Pulsatility Index', 'PI'),
            ]
            
            saved_count = 0
            for feat_key, excel_col, feat_name in features_to_plot:
                # Build patient data for this feature
                all_patient_feature_data = {}
                for pid, metadata in patient_data.items():
                    try:
                        summary_file = None
                        if 'summary_file' in metadata and metadata['summary_file']:
                            summary_file = Path(metadata['summary_file'])
                        elif 'patient_dir' in metadata:
                            patient_dir = Path(metadata['patient_dir'])
                            summary_file = patient_dir / 'SummaryParamTool.xls'
                        
                        if summary_file is None or not summary_file.exists():
                            continue
                        
                        summary_df_patient = load_summary_data(summary_file.parent)
                        if summary_df_patient.empty:
                            continue
                        
                        patient_feature_values = {}
                        all_values = []
                        
                        for _, row in summary_df_patient.iterrows():
                            vessel_label = row.get('Vessel Label', '')
                            if pd.isna(vessel_label):
                                continue
                            
                            # Use flexible matching for vessel labels
                            match = _match_vessel_label(vessel_label)
                            if match is None:
                                continue
                            _, vessel_name = match
                            
                            # Get feature value
                            value = row.get(excel_col, np.nan)
                            if pd.notna(value):
                                if feat_key == 'flow':
                                    value = float(value) * 60.0  # Convert to mL/min
                                else:
                                    value = float(value)
                                patient_feature_values[vessel_name] = value
                                all_values.append(value)
                        
                        # Add TCBF (sum for flow, mean for PI)
                        if all_values:
                            if feat_key == 'flow':
                                patient_feature_values['TCBF'] = sum(all_values)
                            else:
                                patient_feature_values['TCBF'] = np.mean(all_values)
                        
                        all_patient_feature_data[pid] = patient_feature_values
                        
                    except Exception as e:
                        warn(f"   [WARNING] Could not load {feat_name} data for {pid}: {e}")
                        continue
                
                # Generate plots for this feature
                for pid, patient_feature_values in all_patient_feature_data.items():
                    try:
                        output_path = tree_output_dir / f"tree_{feat_key}_{pid}.png"
                        fig = plot_tree(
                            patient_feature_values,
                            pid,
                            feature_name=feat_name,
                            output_path=output_path,
                        )
                        plt.close(fig)
                        saved_count += 1
                    except Exception as e:
                        warn(f"   [WARNING] Could not save tree {feat_name} plot for {pid}: {e}")
            
            print(f"   [OK] Saved {saved_count} tree plot(s) to {tree_output_dir}")
        except Exception as e:
            warn(f"   [WARNING] Error generating tree plots: {e}")
        
        print(f"\n[OK] All analysis plots completed!")
        print(f"   Output directory: {output_dir}")
        
    except Exception as e:
        err(f"[ERROR] Error running all analysis plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    cli()
