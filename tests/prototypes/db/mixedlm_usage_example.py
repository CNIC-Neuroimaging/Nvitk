"""Prototype usage examples for nvitk.stats.mixedlm utilities."""

from __future__ import annotations

from pathlib import Path

from nvitk.stats import (
    build_mixedlm_frame_from_repo,
    fit_or_load_mixedlm,
    plot_mixedlm_params,
    print_mixedlm_info,
)


VESSEL_TERRITORIES_MAP = {
    "Internal Carotid Arteries": ["lica", "rica"],
    "Venous Drainage": ["sssv", "strv", "ltsv", "rtsv"],
    "Anterior Circulation": ["lmca", "rmca", "laca", "raca"],
    "Posterior Circulation": ["basi", "lpca", "rpca"],
}

PAIR_SPECS = [
    {"pair_label": "MCA", "macro_key": "MCA", "micro_key": "MCA"},
    {"pair_label": "ACA", "macro_key": "ACA", "micro_key": "ACA"},
    {"pair_label": "PCA", "macro_key": "PCA", "micro_key": "PCA"},
]


def run_example(repo, *, output_dir: str | Path) -> None:
    """
    Minimal end-to-end example:
      1) build model dataframe from repo
      2) fit/load MixedLM
      3) print info
      4) plot params (continuous or grouped)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build long frame from clinical wide table.
    model_df = build_mixedlm_frame_from_repo(
        repo,
        source="clinical",
        value_columns=["age_c", "sex", "hematocrit", "pulse_pressure_map"],
        id_columns=["subject_uid", "visit_id"],
        filters={"variable_id": ["age_c", "sex", "hematocrit", "pulse_pressure_map"]},
        dropna_columns=["subject_uid", "age_c", "sex", "pulse_pressure_map"],
    )
    model_df["group"] = "all_subjects"
    model_df["y"] = model_df["pulse_pressure_map"]

    result, df_fit, _ = fit_or_load_mixedlm(
        model_path=out_dir / "pulse_pressure_map_mm.pkl",
        data=model_df,
        formula="y ~ age_c + sex + hematocrit",
        groups="group",
        re_formula="1",
        vc_formula={"subject": "0 + C(subject_uid)"},
        overwrite=False,
        required_columns=["group", "y", "age_c", "sex", "hematocrit", "subject_uid"],
        dropna_columns=["group", "y", "age_c", "sex", "hematocrit", "subject_uid"],
        fit_kwargs={"reml": True, "method": "lbfgs", "maxiter": 2000},
    )

    print_mixedlm_info(
        result,
        outcome_name="Pulse Pressure MAP",
        group_name="group",
        vc_group_name="subject_uid",
        output_path=out_dir / "pulse_pressure_map_mm_summary.txt",
    )

    fig = plot_mixedlm_params(
        result=result,
        df_fit=df_fit,
        x="age_c",
        y="y",
        group="group",
        mode="continuous",
        include_points=True,
        output_path=out_dir / "pulse_pressure_map_mm_fit.png",
        title="Pulse pressure map mixed model",
        x_label="Age centered",
        y_label="Pulse pressure map",
        covariate_refs={"sex": 0.0, "hematocrit": float(df_fit["hematocrit"].mean())},
    )
    fig.clf()

