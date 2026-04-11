#!/usr/bin/env python3
"""
Correlation utilities for PESA-Brain QVT+ analysis.

This module provides:
- Clinical summary loading/normalization
- Flow/PI feature table construction (patient x vessel)
- Clinical-variable correlations with scatter + heatmap outputs
- Vessel spatial correlations (matrix + polar summary)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plotter import (
    _VESSEL_CODE_TO_NAME,
    _VESSEL_NAME_TO_GROUP,
    _match_vessel_label,
    load_summary_data,
    plot_polar_flow,
)

from scipy import stats
from scipy.stats import linregress


@dataclass
class ClinicalCorrelationResult:
    """Container for correlation outputs."""

    correlations: pd.DataFrame  # long table with r, n
    correlation_matrix: pd.DataFrame  # clinical_vars x vessels
    count_matrix: pd.DataFrame  # clinical_vars x vessels (n)


def _slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_").lower() or "unknown"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def load_clinical_summary(summary_path: Path) -> pd.DataFrame:
    """
    Load clinical summary data (CSV or Excel) and normalize column names.

    Ensures a 'patient_id' column is present when possible.
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Clinical summary not found: {summary_path}")

    if summary_path.suffix.lower() in [".csv", ".tsv"]:
        df = pd.read_csv(summary_path)
    else:
        df = pd.read_excel(summary_path, sheet_name=0)

    df = _normalize_columns(df)

    # Map common patient id column names
    for col in list(df.columns):
        if col in ["patient_id", "subject_id", "subject", "patient", "id"]:
            df = df.rename(columns={col: "patient_id"})
            break
        if "patient" in col or "subject" in col:
            df = df.rename(columns={col: "patient_id"})
            break

    if "patient_id" not in df.columns:
        raise ValueError(
            f"Clinical summary missing 'patient_id'. Available columns: {list(df.columns)}"
        )

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def _coerce_clinical_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce clinical columns to numeric where possible.
    Sex/gender values are mapped to 0/1 when recognized.
    """
    df = df.copy()
    for col in df.columns:
        if 'id' in col:
            continue

        if "sex" in col or "gender" in col:
            mapped = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(
                    {
                        "m": 1,
                        "male": 1,
                        "1": 1,
                        "f": 0,
                        "female": 0,
                        "0": 0,
                        1: 1,
                        0: 0,
                    }
                )
            )
            df[col] = pd.to_numeric(mapped, errors="coerce")
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_patient_feature_table_from_summary(
    patient_metadata: Dict,
    feature: str,
) -> pd.DataFrame:
    """
    Build a patient x vessel table for flow/PI using SummaryParamTool.xls.
    Values are in mL/min for flow and unitless for PI.
    """
    feature_key = feature.lower()
    if "flow" in feature_key:
        column_name = "Mean Flow ml/s"
    elif "pi" in feature_key or "pulsatility" in feature_key:
        column_name = "Pulsatility Index"
    else:
        column_name = feature

    rows: List[Dict[str, float]] = []

    for patient_id, metadata in patient_metadata.items():
        patient_dir = Path(metadata["patient_dir"])
        summary_df = load_summary_data(patient_dir)
        if summary_df.empty:
            continue

        vessel_values: Dict[str, float] = {}
        for _, row in summary_df.iterrows():
            vessel_label = row.get("Vessel Label", "")
            match = _match_vessel_label(vessel_label)
            if match is None:
                continue
            _, vessel_name = match

            value = row.get(column_name, np.nan)
            if pd.isna(value):
                continue

            value = float(value)
            if "flow" in feature_key and "Mean Flow" in column_name:
                value *= 60.0  # mL/s -> mL/min

            vessel_values[vessel_name] = value

        if not vessel_values:
            continue

        row_data = {"patient_id": patient_id}
        row_data.update(vessel_values)

        # Add TCBF summary value
        if "flow" in feature_key:
            tcbf_parts = []
            for vessel in ["Left ICA", "Right ICA", "Basilar"]:
                if vessel in vessel_values:
                    tcbf_parts.append(vessel_values[vessel])
            if tcbf_parts:
                row_data["TCBF"] = float(sum(tcbf_parts))
        else:
            row_data["TCBF"] = float(np.mean(list(vessel_values.values())))

        rows.append(row_data)

    if not rows:
        raise ValueError(f"No {feature} data found in SummaryParamTool.xls files.")

    df = pd.DataFrame(rows)
    df = df.set_index("patient_id")
    return df


def _select_numeric_clinical_columns(
    clinical_df: pd.DataFrame,
    min_non_nan: int = 2,
) -> List[str]:
    numeric_cols = []
    for col in clinical_df.columns:
        if col in ["patient_id", "visit", "seq_num", 'mri_id', 'med_recon_id']:
            continue
        # if clinical_df[col].notna().sum() >= min_non_nan:
        #     numeric_cols.append(col)
        numeric_cols.append(col)
    return numeric_cols


def build_analysis_database(
    patient_metadata: Dict,
    clinical_df: pd.DataFrame,
    output_path: Path,
    patients2exclude: List[str] = None,
) -> pd.DataFrame:
    """
    Build and save a consolidated database file containing:
    - All clinical variables
    - Flow values for all vessels
    - PI values for all vessels
    
    This database can be loaded later for quick analysis without re-processing patient data.
    
    Args:
        patient_metadata: Dictionary of patient metadata from load_all_patients
        clinical_df: Clinical summary dataframe
        output_path: Path to save the database file (CSV or Excel)
    
    Returns:
        Consolidated dataframe with all data
    """
    print("Building flow feature table...")
    flow_table = build_patient_feature_table_from_summary(patient_metadata, "flow")
    
    print("Building PI feature table...")
    pi_table = build_patient_feature_table_from_summary(patient_metadata, "pi")
    # Remove TCBF for PI
    if "TCBF" in pi_table.columns:
        pi_table = pi_table.drop(columns=["TCBF"])
    
    # Prepare clinical data
    clinical_df = clinical_df.copy()
    clinical_df = _coerce_clinical_numeric(clinical_df)
    
    # Merge all data
    clinical_df = clinical_df.set_index("patient_id")
    merged = clinical_df.join(flow_table, how="outer", rsuffix="_flow")
    
    # Add PI columns with suffix
    for col in pi_table.columns:
        merged[f"{col}_PI"] = pi_table[col]
    
    # Reset index to have patient_id as column
    merged = merged.reset_index()
    patients_no_flows = []
    for patient_id in flow_table.index:
        if np.isnan(flow_table.loc[patient_id]['Left ICA']) and np.isnan(flow_table.loc[patient_id]['Right ICA']) and np.isnan(flow_table.loc[patient_id]['Basilar']) and np.isnan(flow_table.loc[patient_id]['Left MCA']):
            patients_no_flows.append(patient_id)
    
    if patients_no_flows:
        print(f'Found {len(patients_no_flows)} patients without flows')
        patients2exclude.extend(patients_no_flows)
    if patients2exclude:
        print(f'Excluding {len(patients2exclude)} patients')
        merged = merged[~merged["patient_id"].isin(patients2exclude)]    
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() in [".xlsx", ".xls"]:
        merged.to_excel(output_path, index=False)
    else:
        merged.to_csv(output_path, index=False)
    
    print(f"Database saved to: {output_path}")
    print(f"Database shape: {merged.shape}")
    print(f"Columns: {len(merged.columns)}")
    
    return merged


def load_analysis_database(
    database_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the consolidated database file and split into:
    - Clinical dataframe
    - Flow feature table
    - PI feature table
    
    Args:
        database_path: Path to the database file (CSV or Excel)
    
    Returns:
        Tuple of (clinical_df, flow_table, pi_table)
    """
    database_path = Path(database_path)
    if not database_path.exists():
        raise FileNotFoundError(f"Database file not found: {database_path}")
    
    if database_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(database_path)
    else:
        df = pd.read_csv(database_path)
    
    # Ensure patient_id is a column
    if "patient_id" not in df.columns:
        raise ValueError("Database file must contain 'patient_id' column")
    
    df = df.set_index("patient_id")
    
    # Identify columns
    # PI columns end with "_PI"
    pi_cols = [c for c in df.columns if c.endswith("_PI")]
    
    # Known vessel names (from _VESSEL_CODE_TO_NAME and common names)
    known_vessels = list(_VESSEL_CODE_TO_NAME.values()) + ["TCBF", "Left ICA", "Right ICA", 
                                                           "Basilar", "Left MCA", "Right MCA",
                                                           "Left PCA", "Right PCA", "Left ACA", "Right ACA",
                                                           "Sagital Sinus", "Straight Sinus", 
                                                           "Left Transverse", "Right Transverse",
                                                           "Left Communicating", "Right Communicating"]
    
    # Flow columns are known vessel names that don't end with "_PI"
    flow_cols = [c for c in df.columns if c in known_vessels and not c.endswith("_PI")]
    
    # Clinical columns are everything else (excluding visit, seq_num)
    clinical_cols = [c for c in df.columns 
                     if c not in flow_cols 
                     and c not in pi_cols 
                     and c not in ["visit", "seq_num"]
                     and c not in ["mri_id", "med_recon_id"]]
    
    # Extract clinical data
    clinical_df = df[clinical_cols].copy()
    clinical_df = clinical_df.reset_index()
    
    # Extract flow table
    flow_table = df[flow_cols].copy()
    flow_table = flow_table.reset_index()
    
    # Extract PI table (remove _PI suffix)
    pi_table = df[pi_cols].copy()
    pi_table.columns = [c.replace("_PI", "") for c in pi_table.columns]
    pi_table = pi_table.reset_index()
    
    print(f"Loaded database: {len(df)} patients")
    print(f"Clinical variables: {len(clinical_cols)}")
    print(f"Flow vessels: {len(flow_cols)}")
    print(f"PI vessels: {len(pi_cols)}")
    
    return clinical_df, flow_table, pi_table


def compute_clinical_correlations(
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    clinical_columns: Optional[Iterable[str]] = None,
) -> ClinicalCorrelationResult:
    """
    Compute correlation (Pearson) between clinical variables and vessel features.
    Returns long table + correlation and count matrices.
    """
    clinical_df = clinical_df.copy()
    feature_table = feature_table.copy()

    clinical_df = _coerce_clinical_numeric(clinical_df)

    clinical_df = clinical_df.set_index("patient_id")
    feature_table = feature_table.set_index("patient_id")
    merged = clinical_df.join(feature_table, how="inner")

    vessel_cols = [c for c in feature_table.columns if c != "patient_id"]

    if clinical_columns is None:
        clinical_columns = _select_numeric_clinical_columns(merged)
    else:
        clinical_columns = [c for c in clinical_columns if c in merged.columns]

    if not clinical_columns:
        raise ValueError("No clinical columns with numeric data found for correlation.")

    corr_matrix = pd.DataFrame(index=clinical_columns, columns=vessel_cols, dtype=float)
    count_matrix = pd.DataFrame(index=clinical_columns, columns=vessel_cols, dtype=float)

    records: List[Dict[str, float]] = []

    for clinical_var in clinical_columns:
        x = merged[clinical_var]
        for vessel in vessel_cols:
            y = merged[vessel]
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n >= 2:
                x_vals = pd.to_numeric(x[mask], errors="coerce")
                y_vals = pd.to_numeric(y[mask], errors="coerce")
                valid = x_vals.notna() & y_vals.notna()
                if int(valid.sum()) >= 2:
                    x_vals = x_vals[valid].to_numpy(dtype=float)
                    y_vals = y_vals[valid].to_numpy(dtype=float)
                    if np.nanstd(x_vals) == 0 or np.nanstd(y_vals) == 0:
                        r = np.nan
                        p_value = np.nan
                    else:
                        r, p_value = stats.pearsonr(x_vals, y_vals)
                        r = float(r)
                        p_value = float(p_value)
                else:
                    r = np.nan
                    p_value = np.nan
            else:
                r = np.nan
                p_value = np.nan
            corr_matrix.loc[clinical_var, vessel] = r
            count_matrix.loc[clinical_var, vessel] = n
            records.append(
                {
                    "clinical_var": clinical_var,
                    "vessel": vessel,
                    "n": n,
                    "r": r,
                    "p_value": p_value,
                }
            )

    return ClinicalCorrelationResult(
        correlations=pd.DataFrame(records),
        correlation_matrix=corr_matrix,
        count_matrix=count_matrix,
    )


def save_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * corr_matrix.shape[1]), max(6, 0.35 * corr_matrix.shape[0])))
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Vessel")
    ax.set_ylabel("Clinical variable")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_scatter_with_fit(
    x: pd.Series,
    y: pd.Series,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    r_value: Optional[float],
    n: int,
    sigma_threshold: Optional[float] = None,
    p_value: Optional[float] = None,
    use_log_scale: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 2:
        return

    x = x[mask]
    y = y[mask]

    if sigma_threshold is not None and sigma_threshold > 0:
        x_std = float(np.nanstd(x))
        y_std = float(np.nanstd(y))
        x_mean = float(np.nanmean(x))
        y_mean = float(np.nanmean(y))
        if x_std > 0 and y_std > 0:
            z_mask = (np.abs((x - x_mean) / x_std) <= sigma_threshold) & (
                np.abs((y - y_mean) / y_std) <= sigma_threshold
            )
            x = x[z_mask]
            y = y[z_mask]
            if int(len(x)) < 2:
                return

    # Apply log scale to y if requested (for PI)
    y_plot = np.log10(y) if use_log_scale else y
    ylabel_plot = f"log10({ylabel})" if use_log_scale else ylabel
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x=x, y=y_plot, ax=ax, scatter_kws={"alpha": 0.7}, line_kws={"color": "black"})
    slope, intercept = np.polyfit(x, y_plot, 1)
    r_text = "nan" if r_value is None or np.isnan(r_value) else f"{r_value:.3f}"
    p_text = ""
    if p_value is not None and not np.isnan(p_value):
        if p_value < 0.001:
            p_text = ", p<0.001"
        else:
            p_text = f", p={p_value:.3f}"
    ax.set_title(
        f"{title}\n r={r_text}{p_text}, n={n}, slope={slope:.4g}, intercept={intercept:.4g}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_plot)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _get_vessel_group(vessel_name: str) -> Optional[str]:
    """
    Get vessel group for stratification.
    Returns: 'Internal Carotid Arteries', 'Venous Drainage', 'Anterior Circulation', 'Posterior Circulation', or None
    """
    vessel_groups = {
        'Internal Carotid Arteries': ['Left ICA', 'Right ICA'],
        'Venous Drainage': ['Sagital Sinus', 'Straight Sinus', 'Left Transverse', 'Right Transverse'],
        'Anterior Circulation': ['Left MCA', 'Right MCA', 'Left ACA', 'Right ACA'],
        'Posterior Circulation': ['Basilar', 'Left PCA', 'Right PCA'],
    }
    
    for group_name, vessels in vessel_groups.items():
        if vessel_name in vessels:
            return group_name
    return None


def _format_hue_series(hue: pd.Series, hue_name: str) -> pd.Series:
    hue_clean = hue.copy()
    if "sex" in hue_name or "gender" in hue_name:
        # Try to map numeric/binary to labels for consistent legend
        mapped = hue_clean.map({0: "Female", 1: "Male", "0": "Female", "1": "Male"})
        if mapped.notna().any():
            hue_clean = mapped
    return hue_clean


def save_scatter_with_fit_hue(
    x: pd.Series,
    y: pd.Series,
    hue: pd.Series,
    hue_name: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    r_value: Optional[float],
    n: int,
    sigma_threshold: Optional[float] = None,
    p_value: Optional[float] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    hue = hue.copy()
    mask = x.notna() & y.notna() & hue.notna()
    if int(mask.sum()) < 2:
        return

    x = x[mask]
    y = y[mask]
    hue = _format_hue_series(hue[mask], hue_name=hue_name)

    if sigma_threshold is not None and sigma_threshold > 0:
        x_std = float(np.nanstd(x))
        y_std = float(np.nanstd(y))
        x_mean = float(np.nanmean(x))
        y_mean = float(np.nanmean(y))
        if x_std > 0 and y_std > 0:
            z_mask = (np.abs((x - x_mean) / x_std) <= sigma_threshold) & (
                np.abs((y - y_mean) / y_std) <= sigma_threshold
            )
            x = x[z_mask]
            y = y[z_mask]
            hue = hue[z_mask]
            if int(len(x)) < 2:
                return

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=x, y=y, hue=hue, ax=ax, alpha=0.7)
    # Per-category linear fits
    unique_hues = pd.Series(hue).dropna().unique()
    palette = sns.color_palette(n_colors=len(unique_hues))
    hue_to_color = {h: palette[i] for i, h in enumerate(unique_hues)}
    for h_val in unique_hues:
        h_mask = hue == h_val
        if int(h_mask.sum()) < 2:
            continue
        sns.regplot(
            x=x[h_mask],
            y=y[h_mask],
            ax=ax,
            scatter=False,
            line_kws={"color": hue_to_color[h_val]},
        )
    slope, intercept = np.polyfit(x, y, 1)
    r_text = "nan" if r_value is None or np.isnan(r_value) else f"{r_value:.3f}"
    p_text = ""
    if p_value is not None and not np.isnan(p_value):
        if p_value < 0.001:
            p_text = ", p<0.001"
        else:
            p_text = f", p={p_value:.3f}"
    ax.set_title(
        f"{title} (hue={hue_name})\n r={r_text}{p_text}, n={n}, slope={slope:.4g}, intercept={intercept:.4g}",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmaps(
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_label: str,
    output_dir: Path,
    clinical_columns: Optional[Iterable[str]] = None,
) -> ClinicalCorrelationResult:
    """
    Compute and save all correlation heatmaps:
    1. Clinical vars vs vessels
    2. Clinical vars and vessels vs vessels (full matrix)
    3. Clinical vars vs clinical vars
    4. Vessels vs vessels
    
    Returns the computed correlation result.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create corr_heatmaps subfolder
    heatmaps_dir = output_dir / "corr_heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute correlations
    result = compute_clinical_correlations(
        clinical_df=clinical_df,
        feature_table=feature_table,
        clinical_columns=clinical_columns,
    )
    
    # Save correlation tables
    result.correlation_matrix.to_csv(heatmaps_dir / f"clinical_corr_matrix_{_slugify(feature_label)}.csv")
    result.count_matrix.to_csv(heatmaps_dir / f"clinical_corr_counts_{_slugify(feature_label)}.csv")
    result.correlations.to_csv(heatmaps_dir / f"clinical_corr_table_{_slugify(feature_label)}.csv", index=False)
    
    # Heatmap 1: Clinical vars vs vessels (exclude vessel rows)
    clinical_corr_matrix = result.correlation_matrix.drop(
        labels=list(_VESSEL_CODE_TO_NAME.values()) + ["TCBF"], 
        axis=0, 
        errors="ignore"
    )
    save_correlation_heatmap(
        clinical_corr_matrix,
        output_path=heatmaps_dir / f"clinical_vars_vs_vessels_{_slugify(feature_label)}.png",
        title=f"Clinical Variables vs Vessels ({feature_label})",
    )
    
    # Heatmap 2: Clinical vars and vessels vs vessels (full matrix)
    save_correlation_heatmap(
        result.correlation_matrix,
        output_path=heatmaps_dir / f"clinical_and_vessels_vs_vessels_{_slugify(feature_label)}.png",
        title=f"Clinical Variables and Vessels vs Vessels ({feature_label})",
    )
    
    # Heatmap 3: Clinical vars vs clinical vars
    clinical_numeric = _coerce_clinical_numeric(clinical_df)
    clinical_numeric = clinical_numeric.set_index("patient_id")
    clinical_cols = _select_numeric_clinical_columns(clinical_numeric.reset_index())
    if clinical_cols:
        clinical_only = clinical_numeric[clinical_cols]
        clinical_corr = clinical_only.corr(method="pearson", min_periods=2)
        clinical_corr.to_csv(heatmaps_dir / f"clinical_vs_clinical_corr_matrix_{_slugify(feature_label)}.csv")
        save_correlation_heatmap(
            clinical_corr,
            output_path=heatmaps_dir / f"clinical_vs_clinical_{_slugify(feature_label)}.png",
            title=f"Clinical Variables vs Clinical Variables ({feature_label})",
        )
    
    # Heatmap 4: Vessels vs vessels (save in vessel_correlations subfolder)
    vessel_corr_dir = heatmaps_dir
    vessel_corr_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure feature_table has patient_id as index, not as column
    feature_table_work = feature_table.copy()
    if "patient_id" in feature_table_work.columns:
        feature_table_work = feature_table_work.set_index("patient_id")
    
    # Select only numeric vessel columns (exclude any non-numeric columns)
    vessel_cols = [c for c in feature_table_work.columns 
                   if pd.api.types.is_numeric_dtype(feature_table_work[c])]
    
    if vessel_cols:
        vessel_table = feature_table_work[vessel_cols]
        vessel_corr = vessel_table.corr(method="pearson", min_periods=2)
        vessel_corr.to_csv(vessel_corr_dir / f"vessel_corr_matrix_{_slugify(feature_label)}.csv")
        save_correlation_heatmap(
            vessel_corr,
            output_path=vessel_corr_dir / f"vessel_corr_heatmap_{_slugify(feature_label)}.png",
            title=f"Vessels vs Vessels ({feature_label})",
        )
    else:
        print(f"Warning: No numeric vessel columns found for {feature_label}")
    
    return result


def save_standard_scatter_plots(
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_label: str,
    output_dir: Path,
    correlation_result: ClinicalCorrelationResult,
    sigma_threshold: Optional[float] = None,
) -> None:
    """
    Save standard scatter plots (one per clinical variable-vessel pair).
    For PI, also saves log-scale versions.
    """
    output_dir = Path(output_dir)
    scatter_dir = output_dir / "scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    
    merged = clinical_df.set_index("patient_id").join(feature_table.set_index("patient_id"), how="inner")
    
    for _, row in correlation_result.correlations.iterrows():
        clinical_var = row["clinical_var"]
        vessel = row["vessel"]
        n = int(row["n"])
        if n < 2:
            continue
        
        x = merged[clinical_var]
        if clinical_var in ["sex", "gender"]:
            x = x.map({"Male": 1, "Female": 0})
        
        y = merged[vessel]
        mask = x.notna() & y.notna()
        if mask.sum() < 2:
            continue
        
        out_path = (
            scatter_dir
            / _slugify(feature_label)
            / _slugify(clinical_var)
            / f"{_slugify(vessel)}.png"
        )
        p_val = row.get("p_value", None)
        save_scatter_with_fit(
            x=x[mask],
            y=y[mask],
            output_path=out_path,
            title=f"{clinical_var} vs {vessel}",
            xlabel=clinical_var,
            ylabel=f"{feature_label} ({vessel})",
            r_value=row["r"],
            n=n,
            sigma_threshold=sigma_threshold,
            p_value=p_val,
        )
        
        # For PI, also save log-scale version
        if "pi" in feature_label.lower() or "pulsatility" in feature_label.lower():
            out_path_log = out_path.parent / f"{out_path.stem}_log.png"
            save_scatter_with_fit(
                x=x[mask],
                y=y[mask],
                output_path=out_path_log,
                title=f"{clinical_var} vs {vessel} (log scale)",
                xlabel=clinical_var,
                ylabel=f"{feature_label} ({vessel})",
                r_value=row["r"],
                n=n,
                sigma_threshold=sigma_threshold,
                p_value=p_val,
                use_log_scale=True,
            )


def save_hued_scatter_plots(
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_label: str,
    output_dir: Path,
    correlation_result: ClinicalCorrelationResult,
    sigma_threshold: Optional[float] = None,
) -> None:
    """
    Save hued scatter plots stratified by:
    - Sex
    - ICA plaque groups
    - Vessel groups (combined plot per clinical variable)
    """
    output_dir = Path(output_dir)
    scatter_hue_dir = output_dir / "scatter_hue"
    scatter_hue_dir.mkdir(parents=True, exist_ok=True)
    
    merged = clinical_df.set_index("patient_id").join(feature_table.set_index("patient_id"), how="inner")
    
    # Detect hue variables
    hue_candidates = ["sex"]
    hue_vars = [h for h in hue_candidates if h in merged.columns]
    
    # Add ICA plaque stratification if variables exist
    plaque_vars = _detect_carotid_plaque_variables(merged)
    if plaque_vars:
        merged = _add_ica_plaque_stratification(merged, plaque_vars)
        if "ica_plaque_group" in merged.columns:
            hue_vars.append("ica_plaque_group")
    
    # Standard hued scatters (by sex and plaque groups)
    for _, row in correlation_result.correlations.iterrows():
        clinical_var = row["clinical_var"]
        vessel = row["vessel"]
        n = int(row["n"])
        if n < 2:
            continue
        
        x = merged[clinical_var]
        if clinical_var in ["sex", "gender"]:
            x = x.map({"Male": 1, "Female": 0})
        
        y = merged[vessel]
        mask = x.notna() & y.notna()
        if mask.sum() < 2:
            continue
        
        for hue_var in hue_vars:
            out_path_hue = (
                scatter_hue_dir
                / _slugify(hue_var)
                / _slugify(feature_label)
                / _slugify(clinical_var)
                / f"{_slugify(vessel)}.png"
            )
            p_val = row.get("p_value", None)
            save_scatter_with_fit_hue(
                x=x[mask],
                y=y[mask],
                hue=merged.loc[mask, hue_var],
                hue_name=hue_var,
                output_path=out_path_hue,
                title=f"{clinical_var} vs {vessel}",
                xlabel=clinical_var,
                ylabel=f"{feature_label} ({vessel})",
                r_value=row["r"],
                n=n,
                sigma_threshold=sigma_threshold,
                p_value=p_val,
            )
    
    # Vessel group stratification (combined plot per clinical variable)
    for clinical_var in correlation_result.correlations['clinical_var'].unique():
        x = merged[clinical_var]
        if clinical_var in ["sex", "gender"]:
            x = x.map({"Male": 1, "Female": 0})
        
        group_vessels = {
            'Internal Carotid Arteries': ['Left ICA', 'Right ICA'],
            'Venous Drainage': ['Sagital Sinus', 'Straight Sinus', 'Left Transverse', 'Right Transverse'],
            'Anterior Circulation': ['Left MCA', 'Right MCA', 'Left ACA', 'Right ACA'],
            'Posterior Circulation': ['Basilar', 'Left PCA', 'Right PCA'],
        }
        
        # Create a single figure with 2x2 subplots for the 4 groups
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        all_groups_x = []
        all_groups_y = []
        
        for idx, (group_name, vessels_in_group) in enumerate(group_vessels.items()):
            ax = axes[idx]
            available_vessels = [v for v in vessels_in_group if v in feature_table.columns]
            if not available_vessels:
                ax.axis('off')
                continue
            
            # Create long-format data for all vessels in this group
            group_data = []
            for v in available_vessels:
                y = merged[v]
                mask = x.notna() & y.notna()
                if mask.sum() >= 2:
                    v_data = pd.DataFrame({
                        clinical_var: x[mask].values,
                        'value': y[mask].values,
                        'vessel': v
                    })
                    group_data.append(v_data)
            
            if group_data:
                group_df = pd.concat(group_data, ignore_index=True)
                group_df = group_df.dropna(subset=[clinical_var, 'value'])
                if len(group_df) >= 2:
                    # Remove outliers using sigma threshold
                    x_vals = group_df[clinical_var].values
                    y_vals = group_df['value'].values
                    
                    if sigma_threshold is not None and sigma_threshold > 0:
                        x_std = float(np.nanstd(x_vals))
                        y_std = float(np.nanstd(y_vals))
                        x_mean = float(np.nanmean(x_vals))
                        y_mean = float(np.nanmean(y_vals))
                        if x_std > 0 and y_std > 0:
                            z_mask = (np.abs((x_vals - x_mean) / x_std) <= sigma_threshold) & (
                                np.abs((y_vals - y_mean) / y_std) <= sigma_threshold
                            )
                            group_df = group_df[z_mask]
                            x_vals = group_df[clinical_var].values
                            y_vals = group_df['value'].values
                    
                    if len(group_df) >= 2 and np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
                        r_group, p_group = stats.pearsonr(x_vals, y_vals)
                        
                        # Store for overall correlation line
                        all_groups_x.append(x_vals)
                        all_groups_y.append(y_vals)
                        
                        # Plot scatter with hue by vessel
                        colors = sns.color_palette("husl", n_colors=len(available_vessels))
                        for i, vessel in enumerate(available_vessels):
                            vessel_data = group_df[group_df['vessel'] == vessel]
                            if len(vessel_data) > 0:
                                ax.scatter(
                                    vessel_data[clinical_var],
                                    vessel_data['value'],
                                    alpha=0.6,
                                    s=30,
                                    color=colors[i],
                                    label=vessel
                                )
                        
                        # Add regression line for this group
                        slope, intercept, _, _, _ = linregress(x_vals, y_vals)
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_line = slope * x_line + intercept
                        ax.plot(x_line, y_line, color='gray', linewidth=2, alpha=0.7,
                               linestyle='--', label=f"Group fit (r={r_group:.3f})")
                        
                        ax.set_xlabel(clinical_var, fontsize=11)
                        ax.set_ylabel(feature_label, fontsize=11)
                        ax.set_title(f"{group_name}\nr={r_group:.3f}, p={p_group:.3g}, n={len(group_df)}", 
                                   fontsize=12, fontweight='bold')
                        ax.legend(loc='best', fontsize=8)
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.axis('off')
                else:
                    ax.axis('off')
            else:
                ax.axis('off')
        
        # Add overall correlation line in black across all groups
        if all_groups_x:
            all_x_combined = np.concatenate(all_groups_x)
            all_y_combined = np.concatenate(all_groups_y)
            if len(all_x_combined) >= 2 and np.nanstd(all_x_combined) > 0 and np.nanstd(all_y_combined) > 0:
                overall_slope, overall_intercept, overall_r, overall_p, _ = linregress(all_x_combined, all_y_combined)
                x_overall = np.linspace(all_x_combined.min(), all_x_combined.max(), 100)
                y_overall = overall_slope * x_overall + overall_intercept
                
                # Add the overall line to each subplot
                for ax in axes:
                    if ax.has_data():
                        ax.plot(x_overall, y_overall, color='black', linewidth=3, 
                               linestyle='-', alpha=0.8, label=f"Overall (r={overall_r:.3f})")
                        ax.legend(loc='best', fontsize=8)
        
        plt.suptitle(f"{clinical_var} vs {feature_label} - Vessel Groups", 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save the combined plot
        out_path_combined = (
            scatter_hue_dir
            / "vessel_groups_combined"
            / _slugify(feature_label)
            / _slugify(clinical_var)
            / f"vessel_groups_{_slugify(clinical_var)}.png"
        )
        out_path_combined.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path_combined, dpi=300, bbox_inches='tight')
        plt.close(fig)


def save_clinical_correlation_outputs(
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_label: str,
    output_dir: Path,
    clinical_columns: Optional[Iterable[str]] = None,
    sigma_threshold: Optional[float] = None,
) -> ClinicalCorrelationResult:
    """
    Compute and save clinical correlations + scatter plots + heatmap.
    This is a convenience function that calls all the separate steps.
    For step-by-step analysis, use the individual functions instead.
    Returns the computed correlation result.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute correlations and save heatmaps
    result = save_correlation_heatmaps(
        clinical_df=clinical_df,
        feature_table=feature_table,
        feature_label=feature_label,
        output_dir=output_dir,
        clinical_columns=clinical_columns,
    )

    # Step 2: Save standard scatter plots
    save_standard_scatter_plots(
        clinical_df=clinical_df,
        feature_table=feature_table,
        feature_label=feature_label,
        output_dir=output_dir,
        correlation_result=result,
        sigma_threshold=sigma_threshold,
    )

    # Step 3: Save hued scatter plots
    save_hued_scatter_plots(
        clinical_df=clinical_df,
        feature_table=feature_table,
        feature_label=feature_label,
        output_dir=output_dir,
        correlation_result=result,
        sigma_threshold=sigma_threshold,
    )

    # Step 4: Mixed-effects model plots
    _save_mixed_effects_by_vessel_plots(
        clinical_df=clinical_df,
        feature_table=feature_table,
        feature_label=feature_label,
        output_dir=output_dir,
        clinical_columns=clinical_columns,
        sigma_threshold=sigma_threshold,
    )

    return result


def _save_mixed_effects_by_vessel_plots(
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_label: str,
    output_dir: Path,
    clinical_columns: Optional[Iterable[str]] = None,
    sigma_threshold: Optional[float] = None,
) -> None:
    """
    Create mixed-effects plots showing random effects (intercepts and slopes) by vessel.
    One plot per clinical variable, with two subplots: intercepts and slopes.
    Similar to Roberts et al. 2023 figures.
    """
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        return
    
    output_dir = Path(output_dir)
    random_effects_dir = output_dir / "random_effects"
    random_effects_dir.mkdir(parents=True, exist_ok=True)
    
    clinical_df = clinical_df.copy()
    feature_table = feature_table.copy().set_index("patient_id")
    clinical_df = _coerce_clinical_numeric(clinical_df)
    clinical_df = clinical_df.set_index("patient_id")
    merged = clinical_df.join(feature_table, how="inner")
    
    vessel_cols = [c for c in feature_table.columns if c != "patient_id"]
    
    if clinical_columns is None:
        clinical_columns = _select_numeric_clinical_columns(merged)
    else:
        clinical_columns = [c for c in clinical_columns if c in merged.columns]
    
    if not clinical_columns:
        return
    
    for clinical_var in clinical_columns:
        # Prepare data in long format
        long_data = _prepare_long_format_data(
            merged, clinical_var, vessel_cols, sigma_threshold
        )
        
        if long_data is None or len(long_data) < 10:
            continue
        
        # Validate data
        n_patients = long_data["patient_id"].nunique()
        n_vessels = long_data["vessel"].nunique()
        
        if n_patients < 3 or n_vessels < 2:
            continue
        
        # Fit mixed-effects model
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
                
                # Model with vessel-specific intercepts and slopes
                model = MixedLM.from_formula(
                    "value ~ clinical_var + C(vessel) + clinical_var:C(vessel)",
                    data=long_data,
                    groups=long_data["patient_id"]
                )
                result = model.fit(reml=True, method=["lbfgs"])
                
                # Extract fixed effects
                fixed_intercept = result.fe_params.get('Intercept', 0.0)
                fixed_slope = result.fe_params.get('clinical_var', 0.0)
                
                # Extract vessel-specific random effects (intercepts and slopes)
                _plot_mixed_effects_by_vessel(
                    long_data, clinical_var, feature_label, random_effects_dir,
                    result, fixed_intercept, fixed_slope, vessel_cols
                )
                
        except Exception:
            continue


def _plot_mixed_effects_by_vessel(
    long_data: pd.DataFrame,
    clinical_var: str,
    feature_label: str,
    output_dir: Path,
    model_result,
    fixed_intercept: float,
    fixed_slope: float,
    vessel_cols: List[str],
) -> None:
    """
    Create plot showing random effects by vessel: intercepts and slopes.
    Two subplots: one for intercepts, one for slopes (age effect).
    """
    # Get vessel-specific intercepts and slopes from model
    vessel_intercepts = {}
    vessel_slopes = {}
    
    # Extract coefficients for each vessel
    fe_params = model_result.fe_params
    
    # Base intercept and slope
    base_intercept = fe_params.get('Intercept', 0.0)
    base_slope = fe_params.get('clinical_var', 0.0)
    
    # Get unique vessels in data, excluding TCBF
    vessels_in_data = [v for v in long_data['vessel'].unique() if v != 'TCBF']
    
    # Define vessel groups and order
    vessel_groups_order = {
        'Internal Carotid Arteries': ['Left ICA', 'Right ICA'],
        'Anterior Circulation': ['Left MCA', 'Right MCA', 'Left ACA', 'Right ACA'],
        'Posterior Circulation': ['Basilar', 'Left PCA', 'Right PCA'],
        'Venous Drainage': ['Sagital Sinus', 'Straight Sinus', 'Left Transverse', 'Right Transverse'],
    }
    
    # Sort vessels by group order
    vessels_sorted = []
    for group_name, group_vessels in vessel_groups_order.items():
        for vessel in group_vessels:
            if vessel in vessels_in_data:
                vessels_sorted.append(vessel)
    
    # Add any remaining vessels not in groups (shouldn't happen, but just in case)
    for vessel in vessels_in_data:
        if vessel not in vessels_sorted:
            vessels_sorted.append(vessel)
    
    for vessel in vessels_in_data:
        # Vessel-specific intercept (base + vessel effect)
        vessel_intercept_key = f"C(vessel)[T.{vessel}]"
        vessel_slope_key = f"clinical_var:C(vessel)[T.{vessel}]"
        
        intercept = base_intercept
        slope = base_slope
        
        if vessel_intercept_key in fe_params:
            intercept += fe_params[vessel_intercept_key]
        if vessel_slope_key in fe_params:
            slope += fe_params[vessel_slope_key]
        
        vessel_intercepts[vessel] = intercept
        vessel_slopes[vessel] = slope
    
    # If we don't have vessel-specific effects, compute them from individual regressions
    if not vessel_intercepts:
        for vessel in vessels_in_data:
            vessel_data = long_data[long_data['vessel'] == vessel]
            if len(vessel_data) < 2:
                continue
            
            x_vals = vessel_data['clinical_var'].values
            y_vals = vessel_data['value'].values
            
            if np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
                slope, intercept, _, _, _ = linregress(x_vals, y_vals)
                vessel_intercepts[vessel] = intercept
                vessel_slopes[vessel] = slope
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(8, len(vessels_sorted) * 0.4)))
    
    # Plot 1: Intercepts
    y_pos = np.arange(len(vessels_sorted))
    intercepts = [vessel_intercepts.get(v, fixed_intercept) for v in vessels_sorted]
    
    ax1.scatter(intercepts, y_pos, s=100, alpha=0.7, color='steelblue', zorder=3)
    ax1.axvline(x=fixed_intercept, color='black', linewidth=2, linestyle='-', 
               label=f'Fixed Effect (at {clinical_var}=0)')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(vessels_sorted, fontsize=10)
    ax1.set_xlabel(f'Intercept ({feature_label})', fontsize=12, fontweight='bold')
    ax1.set_title('Intercept', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.legend(loc='best', fontsize=9)
    ax1.invert_yaxis()  # Top vessel at top
    
    # Plot 2: Slopes (age effect)
    slopes = [vessel_slopes.get(v, fixed_slope) for v in vessels_sorted]
    
    ax2.scatter(slopes, y_pos, s=100, alpha=0.7, color='coral', zorder=3)
    ax2.axvline(x=fixed_slope, color='black', linewidth=2, linestyle='-',
               label=f'Fixed Effect ({clinical_var} slope)')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(vessels_sorted, fontsize=10)
    ax2.set_xlabel(f'{clinical_var} slope ({feature_label}/{clinical_var})', 
                   fontsize=12, fontweight='bold')
    ax2.set_title(f'{clinical_var} ({feature_label}/{clinical_var})', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend(loc='best', fontsize=9)
    ax2.invert_yaxis()  # Top vessel at top
    
    plt.suptitle(f'Random Effects by Vessel - {feature_label}', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save plot
    filename = f"mixed_effects_by_vessel_{_slugify(clinical_var)}_{_slugify(feature_label)}.png"
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _save_random_effects_plots(
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_label: str,
    output_dir: Path,
    clinical_columns: Optional[Iterable[str]] = None,
    sigma_threshold: Optional[float] = None,
) -> None:
    """
    Create random effects plots by vessel using full mixed-effects model (Roberts et al. 2023, Fig 5 & 6).
    
    Implements a linear mixed-effects model:
    flow_ij = β₀ + β₁*clinical_var_i + u_i + v_j + ε_ij
    
    where:
    - β₀, β₁ are fixed effects (intercept, slope for clinical variable)
    - u_i is random intercept per participant
    - v_j is random slope per vessel (vessel-specific relationship with clinical variable)
    - ε_ij is residual error
    
    Shows individual vessel regression lines (random effects) and overall fixed effect.
    When stratifying, creates both random effects plot and individual plots per group.
    """
    from scipy.stats import linregress
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        import warnings
        warnings.warn(
            "statsmodels not available. Falling back to simplified regression visualization. "
            "Install with: pip install statsmodels"
        )
        _save_random_effects_plots_simple(
            clinical_df, feature_table, feature_label, output_dir, 
            clinical_columns, sigma_threshold
        )
        return
    
    output_dir = Path(output_dir)
    random_effects_dir = output_dir / "random_effects"
    random_effects_dir.mkdir(parents=True, exist_ok=True)
    
    clinical_df = clinical_df.copy()
    feature_table = feature_table.copy()
    clinical_df = _coerce_clinical_numeric(clinical_df)
    clinical_df = clinical_df.set_index("patient_id")
    merged = clinical_df.join(feature_table, how="inner")
    
    vessel_cols = [c for c in feature_table.columns if c != "patient_id"]
    
    if clinical_columns is None:
        clinical_columns = _select_numeric_clinical_columns(merged)
    else:
        clinical_columns = [c for c in clinical_columns if c in merged.columns]
    
    if not clinical_columns:
        return
    
    # Stratification variables
    strat_vars = ["sex"]
    strat_vars = [s for s in strat_vars if s in merged.columns]
    
    # Add carotid plaque stratification if variables exist
    plaque_vars = _detect_carotid_plaque_variables(merged)
    if plaque_vars:
        # Create plaque stratification groups
        merged = _add_carotid_plaque_stratification(merged, plaque_vars)
        if "carotid_plaque_group" in merged.columns:
            strat_vars.append("carotid_plaque_group")
    
    for clinical_var in clinical_columns:
        # Prepare data in long format for mixed-effects model
        long_data = _prepare_long_format_data(
            merged, clinical_var, vessel_cols, sigma_threshold
        )
        
        if long_data is None or len(long_data) < 10:  # Need sufficient data
            continue
        
        # Validate data before fitting model
        # Check for sufficient variability and data points per group
        n_patients = long_data["patient_id"].nunique()
        n_vessels = long_data["vessel"].nunique()
        
        if n_patients < 3 or n_vessels < 2:
            # Not enough data for mixed-effects model
            continue
        
        # Check for sufficient data per patient (at least 2 vessels per patient on average)
        min_data_per_patient = len(long_data) / n_patients
        if min_data_per_patient < 1.5:
            # Not enough repeated measures per patient
            continue
        
        # Fit mixed-effects model
        # Model: flow_ij = β₀ + β₁*clinical_var_i + u_i + v_j + (v_j * clinical_var_i) + ε_ij
        # where:
        # - β₀, β₁ are fixed effects
        # - u_i is random intercept per participant
        # - v_j is random intercept per vessel
        # - (v_j * clinical_var_i) is random slope per vessel (vessel-specific relationship)
        
        try:
            # Model: value ~ clinical_var + (1|patient_id) + (clinical_var|vessel)
            # This models:
            # - Fixed effect: intercept and slope for clinical_var
            # - Random intercept per participant (accounts for participant-level variation)
            # - Random intercept and slope per vessel (vessel-specific relationships)
            
            # First, try model with random intercept per participant and vessel-specific slopes
            # We'll use a two-level structure: patient_id as grouping, vessel as variance component
            
            # Approach 1: Random intercept per participant, vessel as fixed effect with interaction
            # This allows vessel-specific slopes while accounting for participant clustering
            # Suppress warnings for singular covariance matrices (common with small datasets)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
                
                model = MixedLM.from_formula(
                    "value ~ clinical_var + C(vessel) + clinical_var:C(vessel)",
                    data=long_data,
                    groups=long_data["patient_id"]
                )
                result = model.fit(reml=True, method=["lbfgs"])
            
            # Extract fixed effects
            fixed_intercept = result.fe_params.get('Intercept', 0.0)
            fixed_slope = result.fe_params.get('clinical_var', 0.0)
            
            # If interaction terms exist, we can extract vessel-specific slopes
            # For visualization, we'll compute them from the model coefficients
            
        except Exception as e:
            # If complex model fails, try simpler model
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                    warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
                    
                    # Simpler model: random intercept per participant, fixed vessel effects
                    model = MixedLM.from_formula(
                        "value ~ clinical_var + C(vessel)",
                        data=long_data,
                        groups=long_data["patient_id"]
                    )
                    result = model.fit(reml=True, method=["lbfgs"])
                fixed_intercept = result.fe_params.get('Intercept', 0.0)
                fixed_slope = result.fe_params.get('clinical_var', 0.0)
            except Exception as e2:
                # If still fails, fall back to visualization without model
                _plot_random_effects_visualization(
                    long_data, clinical_var, feature_label, random_effects_dir, sigma_threshold
                )
                continue
        
        # Extract vessel-specific random effects
        # For visualization, we'll compute vessel-specific regressions
        # and overlay the fixed effect from the model
        _plot_mixed_effects_results(
            long_data, clinical_var, feature_label, random_effects_dir,
            fixed_intercept, fixed_slope, result, sigma_threshold
        )
        
        # Stratified plots
        for strat_var in strat_vars:
            if strat_var not in merged.columns:
                continue
            
            strat_values = merged[strat_var].dropna().unique()
            if len(strat_values) < 2:
                continue
            
            # Prepare stratified long data
            for strat_val in strat_values:
                strat_mask = merged[strat_var] == strat_val
                strat_patients = merged[strat_mask].index
                strat_long_data = long_data[long_data["patient_id"].isin(strat_patients)]
                
                if len(strat_long_data) < 5:
                    continue
                
                # Validate stratified data
                n_patients_strat = strat_long_data["patient_id"].nunique()
                if n_patients_strat < 2:
                    continue
                
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                        warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
                        
                        strat_model = MixedLM.from_formula(
                            "value ~ clinical_var + C(vessel)",
                            data=strat_long_data,
                            groups=strat_long_data["patient_id"]
                        )
                        strat_result = strat_model.fit(reml=True, method=["lbfgs"], reml_start=0)
                    strat_fixed_intercept = strat_result.fe_params.get('Intercept', 0.0)
                    strat_fixed_slope = strat_result.fe_params.get('clinical_var', 0.0)
                    
                    # Create appropriate label based on stratification variable
                    if strat_var == "sex":
                        strat_label = "Male" if (strat_val == 1 or str(strat_val).lower() in ['male', 'm', '1']) else "Female"
                    elif strat_var == "carotid_plaque_group":
                        strat_label = str(strat_val)  # Use the group name directly
                    else:
                        strat_label = str(strat_val)
                    
                    _plot_mixed_effects_results(
                        strat_long_data, clinical_var, feature_label, random_effects_dir,
                        strat_fixed_intercept, strat_fixed_slope, strat_result, sigma_threshold,
                        strat_label=strat_label, strat_var=strat_var
                    )
                except Exception:
                    continue


def _detect_carotid_plaque_variables(merged: pd.DataFrame) -> List[str]:
    """
    Detect carotid plaque variables in the merged dataframe.
    Looks for variables like plaque_left_carotid, plaque_right_carotid, etc.
    """
    plaque_vars = []
    possible_names = [
        "plaque_left_carotid",
        "plaque_right_carotid",
        "plaque_left",
        "plaque_right",
        "left_carotid_plaque",
        "right_carotid_plaque",
        "carotid_plaque_left",
        "carotid_plaque_right",
    ]
    
    for name in possible_names:
        if name in merged.columns:
            plaque_vars.append(name)
    
    return plaque_vars


def _add_ica_plaque_stratification(
    merged: pd.DataFrame,
    plaque_vars: List[str],
) -> pd.DataFrame:
    """
    Add ICA plaque stratification groups to merged dataframe.
    
    Creates groups based on plaque presence in Left ICA, Right ICA, both, or none:
    - "No Plaques": No plaques in either ICA
    - "Left ICA Only": Plaque only in Left ICA
    - "Right ICA Only": Plaque only in Right ICA
    - "Both ICAs": Plaques in both ICAs
    
    Returns merged dataframe with added 'ica_plaque_group' column.
    """
    merged = merged.copy()
    
    # Try to find left and right ICA plaque variables
    left_var = None
    right_var = None
    
    for var in plaque_vars:
        var_lower = var.lower()
        if "left" in var_lower:
            left_var = var
        elif "right" in var_lower:
            right_var = var
    
    # If we have both variables, create stratification
    if left_var and right_var:
        ica_plaque_groups = []
        
        for _, row in merged.iterrows():
            left_val = row[left_var]
            right_val = row[right_var]
            
            # Check if values indicate presence of plaque
            left_has_plaque = False
            right_has_plaque = False
            
            if pd.notna(left_val):
                try:
                    left_val_float = float(left_val)
                    left_has_plaque = left_val_float > 0
                except (ValueError, TypeError):
                    left_has_plaque = False
            
            if pd.notna(right_val):
                try:
                    right_val_float = float(right_val)
                    right_has_plaque = right_val_float > 0
                except (ValueError, TypeError):
                    right_has_plaque = False
            
            # Assign group
            if not left_has_plaque and not right_has_plaque:
                group = "No Plaques"
            elif left_has_plaque and not right_has_plaque:
                group = "Left ICA Only"
            elif not left_has_plaque and right_has_plaque:
                group = "Right ICA Only"
            elif left_has_plaque and right_has_plaque:
                group = "Both ICAs"
            else:
                group = "Unknown"
            
            ica_plaque_groups.append(group)
        
        merged["ica_plaque_group"] = ica_plaque_groups
    
    # If we only have one variable, create simpler stratification
    elif left_var or right_var:
        var = left_var if left_var else right_var
        ica_plaque_groups = []
        
        for _, row in merged.iterrows():
            val = row[var]
            
            if pd.isna(val):
                group = "Unknown"
            else:
                try:
                    val_float = float(val)
                    if val_float > 0:
                        group = "Has Plaques"
                    else:
                        group = "No Plaques"
                except (ValueError, TypeError):
                    group = "Unknown"
            
            ica_plaque_groups.append(group)
        
        merged["ica_plaque_group"] = ica_plaque_groups
    
    return merged


def _add_carotid_plaque_stratification(
    merged: pd.DataFrame,
    plaque_vars: List[str],
) -> pd.DataFrame:
    """
    Add carotid plaque stratification groups to merged dataframe.
    
    Creates groups based on plaque existence and volume:
    - "No Plaques": No plaques in either carotid
    - "Left Carotid Only": Plaque only in left carotid
    - "Right Carotid Only": Plaque only in right carotid
    - "Bilateral Plaques": Plaques in both carotids
    - "High Volume Plaques": If volume data available, patients with high plaque volume
    
    Returns merged dataframe with added 'carotid_plaque_group' column.
    """
    merged = merged.copy()
    
    # Try to find left and right carotid plaque variables
    left_var = None
    right_var = None
    
    for var in plaque_vars:
        var_lower = var.lower()
        if "left" in var_lower:
            left_var = var
        elif "right" in var_lower:
            right_var = var
    
    # If we have both variables, create stratification
    if left_var and right_var:
        plaque_groups = []
        
        for _, row in merged.iterrows():
            left_val = row[left_var]
            right_val = row[right_var]
            
            # Check if values indicate presence of plaque
            # Handle both binary (0/1) and volume (numeric > 0) cases
            left_has_plaque = False
            right_has_plaque = False
            
            if pd.notna(left_val):
                try:
                    left_val_float = float(left_val)
                    left_has_plaque = left_val_float > 0
                except (ValueError, TypeError):
                    left_has_plaque = False
            
            if pd.notna(right_val):
                try:
                    right_val_float = float(right_val)
                    right_has_plaque = right_val_float > 0
                except (ValueError, TypeError):
                    right_has_plaque = False
            
            # Assign group
            if not left_has_plaque and not right_has_plaque:
                group = "No Plaques"
            elif left_has_plaque and not right_has_plaque:
                group = "Left Carotid Only"
            elif not left_has_plaque and right_has_plaque:
                group = "Right Carotid Only"
            elif left_has_plaque and right_has_plaque:
                group = "Bilateral Plaques"
            else:
                group = "Unknown"
            
            plaque_groups.append(group)
        
        merged["carotid_plaque_group"] = plaque_groups
    
    # If we only have one variable, create simpler stratification
    elif left_var or right_var:
        var = left_var if left_var else right_var
        plaque_groups = []
        
        for _, row in merged.iterrows():
            val = row[var]
            
            if pd.isna(val):
                group = "Unknown"
            else:
                try:
                    val_float = float(val)
                    if val_float > 0:
                        group = "Has Plaques"
                    else:
                        group = "No Plaques"
                except (ValueError, TypeError):
                    group = "Unknown"
            
            plaque_groups.append(group)
        
        merged["carotid_plaque_group"] = plaque_groups
    
    return merged


def _prepare_long_format_data(
    merged: pd.DataFrame,
    clinical_var: str,
    vessel_cols: List[str],
    sigma_threshold: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Prepare data in long format for mixed-effects modeling."""
    # Work with merged directly to avoid index mismatches
    merged_work = merged.copy()
    
    # Handle sex/gender mapping
    if clinical_var in ["sex", "gender"]:
        merged_work[clinical_var] = merged_work[clinical_var].map({"Male": 1, "Female": 0})
    
    # Remove outliers from clinical variable
    if sigma_threshold is not None and sigma_threshold > 0:
        x = merged_work[clinical_var]
        x_valid = x.dropna()
        if len(x_valid) > 0:
            x_std = float(np.nanstd(x_valid))
            x_mean = float(np.nanmean(x_valid))
            if x_std > 0 and not pd.isna(x_mean) and np.isfinite(x_mean):
                z_mask = np.abs((x - x_mean) / x_std) <= sigma_threshold
                merged_subset = merged_work.loc[z_mask]
            else:
                merged_subset = merged_work
        else:
            merged_subset = merged_work
    else:
        merged_subset = merged_work
    
    # Create long format: each row is patient × vessel
    long_rows = []
    for patient_id in merged_subset.index:
        try:
            x_val = merged_subset.loc[patient_id, clinical_var]
            # Ensure it's a scalar value
            if isinstance(x_val, pd.Series):
                x_val = x_val.iloc[0] if len(x_val) > 0 else np.nan
            if pd.isna(x_val):
                continue
        except (KeyError, IndexError):
            continue
        
        for vessel in vessel_cols:
            try:
                y_val = merged_subset.loc[patient_id, vessel]
                # Ensure it's a scalar value
                if isinstance(y_val, pd.Series):
                    y_val = y_val.iloc[0] if len(y_val) > 0 else np.nan
                if pd.isna(y_val):
                    continue
                
                # Convert to float, handling any edge cases
                try:
                    x_val_float = float(x_val)
                    y_val_float = float(y_val)
                except (ValueError, TypeError):
                    continue
                
                long_rows.append({
                    'patient_id': patient_id,
                    'vessel': vessel,
                    'clinical_var': x_val_float,
                    'value': y_val_float,
                })
            except (KeyError, IndexError):
                continue
    
    if not long_rows:
        return None
    
    long_data = pd.DataFrame(long_rows)
    
    # Remove outliers per vessel
    if sigma_threshold is not None and sigma_threshold > 0:
        mask_list = []
        for vessel in long_data['vessel'].unique():
            vessel_mask = long_data['vessel'] == vessel
            y_vals = long_data.loc[vessel_mask, 'value'].values
            y_vals_valid = y_vals[~np.isnan(y_vals)]
            
            if len(y_vals_valid) < 2:
                # Keep all data if insufficient for outlier detection
                mask_list.append(pd.Series([True] * vessel_mask.sum(), index=long_data[vessel_mask].index))
                continue
            
            y_std = float(np.nanstd(y_vals_valid, ddof=1))  # Use ddof=1 for sample std
            y_mean = float(np.nanmean(y_vals_valid))
            
            if y_std > 0 and np.isfinite(y_mean) and np.isfinite(y_std):
                z_mask = np.abs((y_vals - y_mean) / y_std) <= sigma_threshold
                z_mask = np.where(np.isnan(y_vals), True, z_mask)  # Keep NaN values
                mask_list.append(pd.Series(z_mask, index=long_data[vessel_mask].index))
            else:
                # Keep all data if cannot compute outliers
                mask_list.append(pd.Series([True] * vessel_mask.sum(), index=long_data[vessel_mask].index))
        
        if mask_list:
            combined_mask = pd.concat(mask_list).reindex(long_data.index, fill_value=True)
            long_data = long_data[combined_mask]
    
    return long_data


def _plot_mixed_effects_results(
    long_data: pd.DataFrame,
    clinical_var: str,
    feature_label: str,
    output_dir: Path,
    fixed_intercept: float,
    fixed_slope: float,
    model_result,
    sigma_threshold: Optional[float] = None,
    strat_label: Optional[str] = None,
    strat_var: Optional[str] = None,
) -> None:
    """Plot mixed-effects model results with vessel-specific random effects."""
    from scipy.stats import linregress
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    vessels = sorted(long_data['vessel'].unique())
    colors = sns.color_palette("husl", n_colors=len(vessels))
    vessel_to_color = {v: colors[i] for i, v in enumerate(vessels)}
    
    # Plot individual vessel regressions (random effects)
    vessel_slopes = {}
    vessel_intercepts = {}
    
    for vessel in vessels:
        vessel_data = long_data[long_data['vessel'] == vessel]
        if len(vessel_data) < 2:
            continue
        
        x_vals = vessel_data['clinical_var'].values
        y_vals = vessel_data['value'].values
        
        if np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
            slope, intercept, _, _, _ = linregress(x_vals, y_vals)
            vessel_slopes[vessel] = slope
            vessel_intercepts[vessel] = intercept
            
            # Plot scatter
            ax.scatter(x_vals, y_vals, alpha=0.3, s=20, color=vessel_to_color[vessel], label=vessel)
            
            # Plot regression line
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=vessel_to_color[vessel], linewidth=2, alpha=0.7,
                   label=f"{vessel} (β={slope:.3f})")
    
    # Plot fixed effect from mixed model
    all_x = long_data['clinical_var'].values
    x_fixed = np.linspace(all_x.min(), all_x.max(), 100)
    y_fixed = fixed_intercept + fixed_slope * x_fixed
    
    # Get p-value for clinical_var
    try:
        p_val = model_result.pvalues.get('clinical_var', np.nan)
        if pd.isna(p_val):
            p_val = model_result.pvalues.get('clinical_var', np.nan)
    except (AttributeError, KeyError):
        p_val = np.nan
    
    ax.plot(x_fixed, y_fixed, color='black', linewidth=3, linestyle='--',
           label=f"Fixed Effect (β={fixed_slope:.4f}, p={p_val:.4f})")
    
    ax.set_xlabel(clinical_var, fontsize=12)
    ax.set_ylabel(feature_label, fontsize=12)
    
    title = f"Mixed-Effects Model - {feature_label}\n{clinical_var}"
    if strat_label:
        title += f" (stratified by {strat_var}: {strat_label})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add model summary text
    try:
        p_val_text = f"{p_val:.4f}" if not pd.isna(p_val) else "N/A"
    except:
        p_val_text = "N/A"
    
    model_text = (
        f"Model: value ~ {clinical_var} + (1|patient_id) + C(vessel)\n"
        f"Fixed Effect: β={fixed_slope:.4f}, p={p_val_text}\n"
        f"AIC: {model_result.aic:.2f}, BIC: {model_result.bic:.2f}"
    )
    ax.text(0.02, 0.98, model_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f"mixed_effects_{_slugify(clinical_var)}_{_slugify(feature_label)}"
    if strat_label:
        filename += f"_strat_{_slugify(strat_var)}_{_slugify(strat_label)}"
    filename += ".png"
    
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_random_effects_visualization(
    long_data: pd.DataFrame,
    clinical_var: str,
    feature_label: str,
    output_dir: Path,
    sigma_threshold: Optional[float] = None,
) -> None:
    """Fallback visualization when mixed-effects model fails."""
    from scipy.stats import linregress
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    vessels = sorted(long_data['vessel'].unique())
    colors = sns.color_palette("husl", n_colors=len(vessels))
    
    all_x_list = []
    all_y_list = []
    
    for i, vessel in enumerate(vessels):
        vessel_data = long_data[long_data['vessel'] == vessel]
        if len(vessel_data) < 2:
            continue
        
        x_vals = vessel_data['clinical_var'].values
        y_vals = vessel_data['value'].values
        
        if np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
            slope, intercept, _, _, _ = linregress(x_vals, y_vals)
            all_x_list.append(x_vals)
            all_y_list.append(y_vals)
            
            ax.scatter(x_vals, y_vals, alpha=0.3, s=20, color=colors[i], label=vessel)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=colors[i], linewidth=2, alpha=0.7,
                   label=f"{vessel} (β={slope:.3f})")
    
    # Overall fixed effect
    if all_x_list:
        all_x = np.concatenate(all_x_list)
        all_y = np.concatenate(all_y_list)
        fixed_slope, fixed_intercept, _, _, _ = linregress(all_x, all_y)
        x_fixed = np.linspace(all_x.min(), all_x.max(), 100)
        y_fixed = fixed_slope * x_fixed + fixed_intercept
        ax.plot(x_fixed, y_fixed, color='black', linewidth=3, linestyle='--',
               label=f"Fixed Effect (β={fixed_slope:.3f})")
    
    ax.set_xlabel(clinical_var, fontsize=12)
    ax.set_ylabel(feature_label, fontsize=12)
    ax.set_title(f"Random Effects by Vessel - {feature_label}\n{clinical_var}", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = output_dir / f"random_effects_{_slugify(clinical_var)}_{_slugify(feature_label)}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _save_random_effects_plots_simple(
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_label: str,
    output_dir: Path,
    clinical_columns: Optional[Iterable[str]] = None,
    sigma_threshold: Optional[float] = None,
) -> None:
    """Simplified version without mixed-effects model (fallback)."""
    from scipy.stats import linregress
    
    output_dir = Path(output_dir)
    random_effects_dir = output_dir / "random_effects"
    random_effects_dir.mkdir(parents=True, exist_ok=True)
    
    clinical_df = clinical_df.copy()
    feature_table = feature_table.copy()
    clinical_df = _coerce_clinical_numeric(clinical_df)
    clinical_df = clinical_df.set_index("patient_id")
    merged = clinical_df.join(feature_table, how="inner")
    
    vessel_cols = [c for c in feature_table.columns if c != "patient_id"]
    
    if clinical_columns is None:
        clinical_columns = _select_numeric_clinical_columns(merged)
    else:
        clinical_columns = [c for c in clinical_columns if c in merged.columns]
    
    if not clinical_columns:
        return
    
    # Stratification variables
    strat_vars = ["sex"]
    strat_vars = [s for s in strat_vars if s in merged.columns]
    
    # Add carotid plaque stratification if variables exist
    plaque_vars = _detect_carotid_plaque_variables(merged)
    if plaque_vars:
        # Create plaque stratification groups
        merged = _add_carotid_plaque_stratification(merged, plaque_vars)
        if "carotid_plaque_group" in merged.columns:
            strat_vars.append("carotid_plaque_group")
    
    for clinical_var in clinical_columns:
        x = merged[clinical_var]
        if clinical_var in ["sex", "gender"]:
            x = x.map({"Male": 1, "Female": 0})
        
        # Remove outliers if requested
        if sigma_threshold is not None and sigma_threshold > 0:
            x_std = float(np.nanstd(x))
            x_mean = float(np.nanmean(x))
            if x_std > 0:
                z_mask = np.abs((x - x_mean) / x_std) <= sigma_threshold
                x = x[z_mask]
                merged_subset = merged.loc[z_mask]
            else:
                merged_subset = merged
        else:
            merged_subset = merged
        
        # Collect data for all vessels
        vessel_data = []
        for vessel in vessel_cols:
            y = merged_subset[vessel]
            mask = x.notna() & y.notna()
            if mask.sum() >= 2:
                x_vals = x[mask].values
                y_vals = y[mask].values
                # Remove outliers from y
                if sigma_threshold is not None and sigma_threshold > 0:
                    y_std = float(np.nanstd(y_vals))
                    y_mean = float(np.nanmean(y_vals))
                    if y_std > 0:
                        z_mask_y = np.abs((y_vals - y_mean) / y_std) <= sigma_threshold
                        x_vals = x_vals[z_mask_y]
                        y_vals = y_vals[z_mask_y]
                
                if len(x_vals) >= 2 and np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
                    vessel_data.append({
                        'vessel': vessel,
                        'x': x_vals,
                        'y': y_vals,
                    })
        
        if not vessel_data:
            continue
        
        # Create random effects plot (all vessels, no stratification)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual vessel regressions (random effects)
        colors = sns.color_palette("husl", n_colors=len(vessel_data))
        vessel_slopes = []
        vessel_intercepts = []
        
        for i, vd in enumerate(vessel_data):
            vessel = vd['vessel']
            x_vals = vd['x']
            y_vals = vd['y']
            
            # Fit regression
            slope, intercept, r_val, p_val, _ = linregress(x_vals, y_vals)
            vessel_slopes.append(slope)
            vessel_intercepts.append(intercept)
            
            # Plot scatter
            ax.scatter(x_vals, y_vals, alpha=0.3, s=20, color=colors[i], label=vessel)
            
            # Plot regression line
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=colors[i], linewidth=2, alpha=0.7, label=f"{vessel} (β={slope:.3f})")
        
        # Overall fixed effect (pooled regression)
        all_x = np.concatenate([vd['x'] for vd in vessel_data])
        all_y = np.concatenate([vd['y'] for vd in vessel_data])
        fixed_slope, fixed_intercept, fixed_r, fixed_p, _ = linregress(all_x, all_y)
        x_fixed = np.linspace(all_x.min(), all_x.max(), 100)
        y_fixed = fixed_slope * x_fixed + fixed_intercept
        ax.plot(x_fixed, y_fixed, color='black', linewidth=3, linestyle='--', label=f"Fixed Effect (β={fixed_slope:.3f})")
        
        ax.set_xlabel(clinical_var, fontsize=12)
        ax.set_ylabel(feature_label, fontsize=12)
        ax.set_title(f"Random Effects by Vessel - {feature_label}\n{clinical_var}", fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = random_effects_dir / f"random_effects_{_slugify(clinical_var)}_{_slugify(feature_label)}.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Stratified plots (if stratification variables available)
        strat_vars_simple = ["sex"]
        strat_vars_simple = [s for s in strat_vars_simple if s in merged_subset.columns]
        # Add plaque stratification if available
        if "carotid_plaque_group" in merged_subset.columns:
            strat_vars_simple.append("carotid_plaque_group")
        for strat_var in strat_vars_simple:
            if strat_var not in merged_subset.columns:
                continue
            
            strat_values = merged_subset[strat_var].dropna().unique()
            if len(strat_values) < 2:
                continue
            
            # For sex (2 groups) or carotid plaque groups, create plots
            if strat_var == "sex" and len(strat_values) == 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for strat_val in strat_values:
                    strat_mask = merged_subset[strat_var] == strat_val
                    strat_label = "Male" if (strat_val == 1 or str(strat_val).lower() in ['male', 'm', '1']) else "Female"
                    
                    # Collect vessel data for this stratum
                    strat_vessel_data = []
                    for vessel in vessel_cols:
                        y = merged_subset.loc[strat_mask, vessel]
                        x_strat = x[strat_mask]
                        mask = x_strat.notna() & y.notna()
                        if mask.sum() >= 2:
                            x_vals = x_strat[mask].values
                            y_vals = y[mask].values
                            if len(x_vals) >= 2 and np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
                                strat_vessel_data.append({
                                    'vessel': vessel,
                                    'x': x_vals,
                                    'y': y_vals,
                                })
                    
                    if not strat_vessel_data:
                        continue
                    
                    # Plot individual vessel regressions for this stratum
                    colors_strat = sns.color_palette("husl", n_colors=len(strat_vessel_data))
                    for i, vd in enumerate(strat_vessel_data):
                        vessel = vd['vessel']
                        x_vals = vd['x']
                        y_vals = vd['y']
                        slope, intercept, _, _, _ = linregress(x_vals, y_vals)
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_line = slope * x_line + intercept
                        ax.plot(x_line, y_line, color=colors_strat[i], linewidth=1.5, alpha=0.6, linestyle='-')
                    
                    # Fixed effect for this stratum
                    all_x_strat = np.concatenate([vd['x'] for vd in strat_vessel_data])
                    all_y_strat = np.concatenate([vd['y'] for vd in strat_vessel_data])
                    if len(all_x_strat) >= 2:
                        fixed_slope_strat, fixed_intercept_strat, _, _, _ = linregress(all_x_strat, all_y_strat)
                        x_fixed_strat = np.linspace(all_x_strat.min(), all_x_strat.max(), 100)
                        y_fixed_strat = fixed_slope_strat * x_fixed_strat + fixed_intercept_strat
                        ax.plot(x_fixed_strat, y_fixed_strat, color='black' if strat_label == "Male" else 'gray', 
                               linewidth=2.5, linestyle='--', label=f"{strat_label} Fixed Effect (β={fixed_slope_strat:.3f})")
                
                ax.set_xlabel(clinical_var, fontsize=12)
                ax.set_ylabel(feature_label, fontsize=12)
                ax.set_title(f"Random Effects by Vessel - {feature_label}\n{clinical_var} (stratified by {strat_var})", 
                           fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                out_path_strat = random_effects_dir / f"random_effects_{_slugify(clinical_var)}_{_slugify(feature_label)}_strat_{_slugify(strat_var)}.png"
                fig.savefig(out_path_strat, dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            # For carotid plaque groups, create separate plots per group
            elif strat_var == "carotid_plaque_group":
                for strat_val in strat_values:
                    strat_mask = merged_subset[strat_var] == strat_val
                    if strat_mask.sum() < 2:
                        continue
                    
                    # Create plot for this plaque group
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    colors_group = sns.color_palette("husl", n_colors=len(vessel_cols))
                    group_x_list = []
                    group_y_list = []
                    
                    for i, vessel in enumerate(vessel_cols):
                        y = merged_subset.loc[strat_mask, vessel]
                        x_strat = x[strat_mask]
                        mask = x_strat.notna() & y.notna()
                        if mask.sum() >= 2:
                            x_vals = x_strat[mask].values
                            y_vals = y[mask].values
                            if len(x_vals) >= 2 and np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
                                slope, intercept, _, _, _ = linregress(x_vals, y_vals)
                                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                                y_line = slope * x_line + intercept
                                ax.plot(x_line, y_line, color=colors_group[i], linewidth=2, alpha=0.7,
                                       label=f"{vessel} (β={slope:.3f})")
                                group_x_list.append(x_vals)
                                group_y_list.append(y_vals)
                    
                    # Fixed effect for this plaque group
                    if group_x_list:
                        all_x_group = np.concatenate(group_x_list)
                        all_y_group = np.concatenate(group_y_list)
                        if len(all_x_group) >= 2:
                            fixed_slope_group, fixed_intercept_group, _, _, _ = linregress(all_x_group, all_y_group)
                            x_fixed_group = np.linspace(all_x_group.min(), all_x_group.max(), 100)
                            y_fixed_group = fixed_slope_group * x_fixed_group + fixed_intercept_group
                            ax.plot(x_fixed_group, y_fixed_group, color='black', linewidth=3, linestyle='--',
                                   label=f"Fixed Effect (β={fixed_slope_group:.3f})")
                    
                    ax.set_xlabel(clinical_var, fontsize=12)
                    ax.set_ylabel(feature_label, fontsize=12)
                    ax.set_title(f"Random Effects by Vessel - {feature_label}\n{clinical_var} ({strat_val})",
                               fontsize=14, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    out_path_group = random_effects_dir / f"random_effects_{_slugify(clinical_var)}_{_slugify(feature_label)}_plaque_{_slugify(str(strat_val))}.png"
                    fig.savefig(out_path_group, dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            # For vessel groups (multiple groups), create separate plots per group
            else:
                vessel_groups = {
                    'Internal Carotid Arteries': ['Left ICA', 'Right ICA'],
                    'Venous Drainage': ['Sagital Sinus', 'Straight Sinus', 'Left Transverse', 'Right Transverse'],
                    'Anterior Circulation': ['Left MCA', 'Right MCA', 'Left ACA', 'Right ACA'],
                    'Posterior Circulation': ['Basilar', 'Left PCA', 'Right PCA'],
                }
                
                for group_name, group_vessels in vessel_groups.items():
                    group_vessels_available = [v for v in group_vessels if v in vessel_cols]
                    if not group_vessels_available:
                        continue
                    
                    # Create plot for this vessel group
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    colors_group = sns.color_palette("husl", n_colors=len(group_vessels_available))
                    for i, vessel in enumerate(group_vessels_available):
                        y = merged_subset[vessel]
                        mask = x.notna() & y.notna()
                        if mask.sum() >= 2:
                            x_vals = x[mask].values
                            y_vals = y[mask].values
                            if len(x_vals) >= 2 and np.nanstd(x_vals) > 0 and np.nanstd(y_vals) > 0:
                                slope, intercept, _, _, _ = linregress(x_vals, y_vals)
                                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                                y_line = slope * x_line + intercept
                                ax.plot(x_line, y_line, color=colors_group[i], linewidth=2, alpha=0.7, 
                                       label=f"{vessel} (β={slope:.3f})")
                    
                    # Fixed effect for this group
                    group_x_list = []
                    group_y_list = []
                    for vessel in group_vessels_available:
                        y = merged_subset[vessel]
                        mask = x.notna() & y.notna()
                        if mask.sum() >= 2:
                            group_x_list.append(x[mask].values)
                            group_y_list.append(y[mask].values)
                    
                    if group_x_list:
                        all_x_group = np.concatenate(group_x_list)
                        all_y_group = np.concatenate(group_y_list)
                        if len(all_x_group) >= 2:
                            fixed_slope_group, fixed_intercept_group, _, _, _ = linregress(all_x_group, all_y_group)
                            x_fixed_group = np.linspace(all_x_group.min(), all_x_group.max(), 100)
                            y_fixed_group = fixed_slope_group * x_fixed_group + fixed_intercept_group
                            ax.plot(x_fixed_group, y_fixed_group, color='black', linewidth=3, linestyle='--', 
                                   label=f"Fixed Effect (β={fixed_slope_group:.3f})")
                    
                    ax.set_xlabel(clinical_var, fontsize=12)
                    ax.set_ylabel(feature_label, fontsize=12)
                    ax.set_title(f"Random Effects by Vessel - {feature_label}\n{clinical_var} ({group_name})", 
                               fontsize=14, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    out_path_group = random_effects_dir / f"random_effects_{_slugify(clinical_var)}_{_slugify(feature_label)}_group_{_slugify(group_name)}.png"
                    fig.savefig(out_path_group, dpi=300, bbox_inches='tight')
                    plt.close(fig)


def save_vessel_spatial_correlation_outputs(
    feature_table: pd.DataFrame,
    output_dir: Path,
    feature_label: str,
) -> None:
    """
    Save vessel correlation matrix and polar spatial correlation plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vessel_corr = feature_table.corr(method="pearson", min_periods=2)
    vessel_corr.to_csv(output_dir / f"vessel_corr_matrix_{_slugify(feature_label)}.csv")

    # Heatmap
    save_correlation_heatmap(
        vessel_corr,
        output_path=output_dir / f"vessel_corr_heatmap_{_slugify(feature_label)}.png",
        title=f"Vessel correlations ({feature_label})",
    )

    # Polar summary intentionally skipped pending new representation


def save_eda_overview(
    patient_metadata: Dict,
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save a simple EDA overview:
    - Patients with .mat outputs
    - LOC coverage per vessel
    - Clinical variable availability
    - Patient intersections with clinical + feature data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Patients with .mat files
    mat_status = []
    for patient_id, metadata in patient_metadata.items():
        patient_dir = Path(metadata["patient_dir"])
        mat_files = list(patient_dir.glob("qvtData_ISOfix_*.mat"))
        mat_status.append({"patient_id": patient_id, "has_mat": len(mat_files) > 0})
    mat_df = pd.DataFrame(mat_status)
    mat_df.to_csv(output_dir / "eda_patients_with_mat.csv", index=False)

    # Vessel LOC coverage
    vessel_counts: Dict[str, int] = {}
    for metadata in patient_metadata.values():
        locs = metadata.get("LOCs", {})
        for vessel_code in locs.keys():
            vessel_name = _VESSEL_CODE_TO_NAME.get(vessel_code, vessel_code)
            vessel_counts[vessel_name] = vessel_counts.get(vessel_name, 0) + 1
    vessel_cov = pd.DataFrame(
        [{"vessel": k, "n_patients_with_loc": v} for k, v in vessel_counts.items()]
    ).sort_values("n_patients_with_loc", ascending=False)
    vessel_cov.to_csv(output_dir / "eda_vessel_loc_coverage.csv", index=False)

    # Clinical availability
    clinical_counts = []
    for col in clinical_df.columns:
        if col == "patient_id":
            continue
        clinical_counts.append(
            {
                "clinical_var": col,
                "n_non_nan": int(clinical_df[col].notna().sum()),
            }
        )
    clinical_cov = pd.DataFrame(clinical_counts).sort_values("n_non_nan", ascending=False)
    clinical_cov.to_csv(output_dir / "eda_clinical_variable_coverage.csv", index=False)

    # Intersection counts
    patients_all = set(patient_metadata.keys())
    patients_mat = set(mat_df[mat_df["has_mat"]]["patient_id"].tolist())
    patients_clinical = set(clinical_df["patient_id"].astype(str))
    patients_feature = set(feature_table.index.astype(str))

    intersection = patients_mat & patients_clinical & patients_feature
    intersection_summary = pd.DataFrame(
        [
            {
                "total_patients": len(patients_all),
                "with_mat": len(patients_mat),
                "with_clinical": len(patients_clinical),
                "with_feature": len(patients_feature),
                "with_all_non_loc": len(intersection),
            }
        ]
    )
    intersection_summary.to_csv(output_dir / "eda_intersection_summary.csv", index=False)


def save_flow_pi_cross_correlation(
    flow_table: pd.DataFrame,
    pi_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save cross-correlation matrix between flow vessels and PI vessels.
    Rows: flow vessels, Columns: PI vessels
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flow = flow_table.copy()
    pi = pi_table.copy()

    # Align patients
    common_patients = flow.index.intersection(pi.index)
    flow = flow.loc[common_patients]
    pi = pi.loc[common_patients]

    if flow.empty or pi.empty:
        return

    flow_cols = list(flow.columns)
    pi_cols = list(pi.columns)

    corr_matrix = pd.DataFrame(index=flow_cols, columns=pi_cols, dtype=float)
    for f_col in flow_cols:
        for p_col in pi_cols:
            x = pd.to_numeric(flow[f_col], errors="coerce")
            y = pd.to_numeric(pi[p_col], errors="coerce")
            mask = x.notna() & y.notna()
            if int(mask.sum()) >= 2:
                x_vals = x[mask].to_numpy(dtype=float)
                y_vals = y[mask].to_numpy(dtype=float)
                if np.nanstd(x_vals) == 0 or np.nanstd(y_vals) == 0:
                    corr_matrix.loc[f_col, p_col] = np.nan
                else:
                    corr_matrix.loc[f_col, p_col] = float(np.corrcoef(x_vals, y_vals)[0, 1])
            else:
                corr_matrix.loc[f_col, p_col] = np.nan

    corr_matrix.to_csv(output_dir / "flow_vs_pi_corr_matrix.csv")
    save_correlation_heatmap(
        corr_matrix,
        output_path=output_dir / "flow_vs_pi_corr_heatmap.png",
        title="Flow vs PI correlations",
    )


def save_polar_clinical_correlations(
    clinical_df: pd.DataFrame,
    feature_table: pd.DataFrame,
    feature_label: str,
    output_dir: Path,
    clinical_columns: Optional[Iterable[str]] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
) -> None:
    """
    Save per-clinical-variable polar plots using vessel correlations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_clinical_correlations(
        clinical_df=clinical_df,
        feature_table=feature_table,
        clinical_columns=clinical_columns,
    )

    for clinical_var in result.correlation_matrix.index:
        row = result.correlation_matrix.loc[clinical_var]
        vessel_values = {
            vessel: float(r)
            for vessel, r in row.items()
            if pd.notna(r)
        }
        if not vessel_values:
            continue

        output_path = output_dir / f"polar_{_slugify(clinical_var)}.png"
        fig = plot_polar_flow(
            vessel_values,
            patient_id=clinical_var,
            feature_name=f"{feature_label} correlation",
            output_path=output_path,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            use_abs=False,
        )
        plt.close(fig)


def save_polar_difference_correlations(
    clinical_df: pd.DataFrame,
    flow_table: pd.DataFrame,
    pi_table: pd.DataFrame,
    output_dir: Path,
    clinical_columns: Optional[Iterable[str]] = None,
    cmap: str = "coolwarm",
) -> None:
    """
    Save per-clinical-variable polar plots for (flow correlation - PI correlation).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_result = compute_clinical_correlations(
        clinical_df=clinical_df,
        feature_table=flow_table,
        clinical_columns=clinical_columns,
    )
    pi_result = compute_clinical_correlations(
        clinical_df=clinical_df,
        feature_table=pi_table,
        clinical_columns=clinical_columns,
    )

    common_clinical = [
        c for c in flow_result.correlation_matrix.index
        if c in pi_result.correlation_matrix.index
    ]

    for clinical_var in common_clinical:
        flow_row = flow_result.correlation_matrix.loc[clinical_var]
        pi_row = pi_result.correlation_matrix.loc[clinical_var]

        common_vessels = [v for v in flow_row.index if v in pi_row.index]
        diff_values = {}
        for vessel in common_vessels:
            f_val = flow_row.get(vessel, np.nan)
            p_val = pi_row.get(vessel, np.nan)
            if pd.notna(f_val) and pd.notna(p_val):
                diff_values[vessel] = float(f_val - p_val)

        if not diff_values:
            continue

        max_abs = max(abs(v) for v in diff_values.values())
        if max_abs == 0:
            max_abs = 1.0

        output_path = output_dir / f"polar_diff_{_slugify(clinical_var)}.png"
        fig = plot_polar_flow(
            diff_values,
            patient_id=clinical_var,
            feature_name="Flow - PI correlation",
            output_path=output_path,
            cmap=cmap,
            vmin=-max_abs,
            vmax=max_abs,
            use_abs=False,
        )
        plt.close(fig)