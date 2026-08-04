"""General mediation analysis utilities."""

# TODO: GPU implementation Cupy + cuDF

from __future__ import annotations

import numpy as np
import pandas as pd

from .mixedlm import fit_or_load_mixedlm

def bootstrap_mediation_by_subject(
    df,
    formula_m,
    formula_y,
    model_cols,
    n_boot=500,
    seed=42,
):
    """
    Mediation bootstrap (a*b) resampling at the subject level.

    Returns:
        dict with:
            - indirect_dist (np.array)
            - ci_low, ci_high
            - mean
            - p_value (two-sided bootstrap)
       
    """
    rng = np.random.default_rng(seed)
    subjects = df["subject_uid"].unique()

    indirects = []

    for _ in range(n_boot):
        # ---- bootstrap per subject ----
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)

        dfs = []
        for i, subj in enumerate(sampled_subjects):
            df_sub = df[df["subject_uid"] == subj].copy()

            # reindexing key for mixed models
            df_sub["subject_uid"] = f"{subj}_boot{i}"
            dfs.append(df_sub)

        df_boot = pd.concat(dfs, ignore_index=True)

        try:
            # ---- mediator model (a) ----
            res_m, _, _ = fit_or_load_mixedlm(
                model_path=None,
                data=df_boot,
                formula=formula_m,
                groups="territory",
                re_formula="1",
                vc_formula={"subject": "0 + C(subject_uid)"},
                overwrite=True,
                required_columns=model_cols,
            )

            # ---- outcome model (b, c') ----
            res_y, _, _ = fit_or_load_mixedlm(
                model_path=None,
                data=df_boot,
                formula=formula_y,
                groups="territory",
                re_formula="1",
                vc_formula={"subject": "0 + C(subject_uid)"},
                overwrite=True,
                required_columns=model_cols,
            )

            a = res_m.params["pi"]
            b = res_y.params["pp"]

            indirects.append(a * b)

        except Exception as e:
            import traceback
            traceback.print_exc()
            continue  

    indirects = np.array(indirects)

    # ---- summary ----
    ci_low = np.percentile(indirects, 2.5)
    ci_high = np.percentile(indirects, 97.5)
    mean = indirects.mean()

    # two-sided bootstrap p-value
    p_value = 2 * min(
        (indirects <= 0).mean(),
        (indirects >= 0).mean()
    )

    return {
        "indirect_dist": indirects,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean": mean,
        "p_value": p_value,
    }