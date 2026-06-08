#!/usr/bin/env python3
"""
Destructively rebuild the nvitk PESA-Brain dataset from Excel sources, optional XNAT sync,
and follow-up imports (T1, cognitive, ATT, WMH, derivations).

Example::

    python scripts/build_db.py \\
        --dataset-root ~/nvitk/dataset/nvitk-dataset \\
        --db-base-path "/path/to/PESA-Brain/DB/raw/" \\
        --new-vars-path "/path/to/PESA-Brain/DB/new_vars" \\
        --with-xnat \\
        --build-sqlite-index
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from nvitk.db.importers import (
    enrich_sessions_available_scans,
    import_pesabrain_source,
    normalize_measurement_visit_ids,
    normalize_session_visit_labels,
    prune_redundant_local_sessions,
    rebuild_subjects_table,
)
from nvitk.db.local_dicom_assets import upsert_dicom_assets
from nvitk.db.local_nifti_assets import upsert_nifti_assets
from nvitk.db.repo import DataRepo
from nvitk.db.xnat import sync_xnat_project
from nvitk.db.xnat_config import finalize_xnat_connection, load_xnat_profile, resolve_xnat_connection
from nvitk.db.xnat_projects import (
    build_default_xnat_sequences_csv,
    default_sequences_for_project,
    sequences_csv,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from import_new_vars import (
    DEFAULT_PATHS,
    derive_apoe_group,
    derive_pulse_pressure_map,
    import_att_csv,
    import_cognitive_wide,
    import_t1_volumetry,
    import_wmh_csv,
    rename_sys_dias_delta_to_pp,
)

DEFAULT_DATASET_ROOT = Path("~/nvitk/dataset/nvitk-dataset")
DEFAULT_DB_BASE = Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/")
DEFAULT_NEW_VARS = Path("/home/imarcoss/NetVolumes/Tierra/LAB_VF-ICH/LAB/MCC LAB/_IgnacioMarcos/LabVF/PESA-Brain/DB/raw/")
DEFAULT_IMPORT_RUN_ID = "pesabrain_build_v1"
DEFAULT_XNAT_SEQUENCES = build_default_xnat_sequences_csv()
XNAT_BUILD_PROJECTS = ("PESA_Brain", "IA_PET_V5")

CATALOG_TEMPLATE_ROOT = _REPO_ROOT / "dataset" / "catalog"


@dataclass(frozen=True)
class PesabrainImportStep:
    filename: str
    sheet: str | None = None
    source_kind: str | None = None
    pipeline_id: str | None = None
    base: str = "raw"


PESABRAIN_BUILD_STEPS: tuple[PesabrainImportStep, ...] = (
    PesabrainImportStep("PESABrain_All_IDs.xlsx", sheet="Sheet1", source_kind="subject_ids"),
    PesabrainImportStep("PESABrain_All_4DFlow_IDs.xlsx", sheet="Sheet1", source_kind="subject_ids"),
    PesabrainImportStep(
        "PESABrain_SubjectCatalog_AllXNAT_20260216.xlsx",
        sheet="Datos",
        source_kind="subject_catalog",
    ),
    PesabrainImportStep(
        "PESABrain_Clinical_20260216.xlsx",
        sheet="Sheet1",
        source_kind="clinical_wide",
    ),
    PesabrainImportStep("PESABrain_APOE_20260318.xlsx", sheet="Sheet1", source_kind="clinical_wide"),
    PesabrainImportStep("PESABrain_TAC_20260318.xlsx", sheet="Sheet1", source_kind="clinical_wide"),
    PesabrainImportStep(
        "PESABrain_Echography_CarotidePlaque_20260216.xlsx",
        sheet="Sheet1",
        source_kind="clinical_wide",
    ),
    PesabrainImportStep(
        "PESABrain_4DFlow_LocalizedPI_20260216.xlsx",
        sheet="PESABrain_AnalysisDB_Batch1",
        source_kind="image_wide",
        pipeline_id="4dflow_v1",
    ),
    PesabrainImportStep(
        "PESABrain_4DFlow_LocalizedTimeAvgFlow_20260216.xlsx",
        sheet="PESABrain_AnalysisDB_Batch1",
        source_kind="image_wide",
        pipeline_id="4dflow_v1",
    ),
    PesabrainImportStep(
        "PESABrain_4DFlow_LocalizedTimeseriesFlow_Wide_20260216.xlsx",
        sheet="Datos",
        source_kind="image_timeseries_wide",
        pipeline_id="4dflow_v1",
    ),
    PesabrainImportStep(
        "PESABrain_4DFlowv2_LocalizedPI_202602410.xlsx",
        sheet="Sheet1",
        source_kind="image_wide",
        pipeline_id="4dflow_v2",
    ),
    PesabrainImportStep(
        "PESABrain_4DFlowv2_LocalizedTimeAvgFlow_202602410.xlsx",
        sheet="Sheet1",
        source_kind="image_wide",
        pipeline_id="4dflow_v2",
    ),
    PesabrainImportStep(
        "PESABrain_4DFlowv2_LocalizedTimeseriesFlow_Wide_202602410.xlsx",
        sheet="Sheet1",
        source_kind="image_timeseries_wide",
        pipeline_id="4dflow_v2",
    ),
    PesabrainImportStep(
        "PESABrain_ASLPerfusion_ThrMeanCBF_20260216.xlsx",
        sheet="Sheet1",
        source_kind="image_wide",
        pipeline_id="asl_v1",
    ),
    PesabrainImportStep(
        "PESABrain_ASLPerfusion_VascularAtlas_MeanCBF_20260216.xlsx",
        sheet="Sheet1",
        source_kind="image_wide",
        pipeline_id="asl_v1",
    ),
    PesabrainImportStep(
        "PESABrain_ASLPerfusion_CovCBF_20260216.xlsx",
        sheet="Sheet1",
        source_kind="image_wide",
        pipeline_id="asl_v1",
    ),
    PesabrainImportStep(
        "PESABrain_ASLPerfusion_VascularAtlas_MeanATT_20260216.xlsx",
        sheet="Sheet1",
        source_kind="image_wide",
        pipeline_id="asl_v1",
        base="new_vars",
    ),
    PesabrainImportStep(
        "PESABrain_ASLPerfusion_VascularAtlas_MedianATT_20260216.xlsx",
        sheet="Sheet1",
        source_kind="image_wide",
        pipeline_id="asl_v1",
        base="new_vars",
    ),
)

CURATED_VARIABLES: list[dict[str, Any]] = [
    {"variable_id": "age_at_mri", "domain": "clinical", "table": "clinical_measurements", "unit": "Years"},
    {"variable_id": "weight", "domain": "clinical", "table": "clinical_measurements", "unit": "kg"},
    {"variable_id": "height", "domain": "clinical", "table": "clinical_measurements", "unit": "cm"},
    {"variable_id": "bmi", "domain": "clinical", "table": "clinical_measurements", "unit": "kg/m2"},
    {
        "variable_id": "psqto000",
        "domain": "clinical",
        "table": "clinical_measurements",
        "unit": "0: non; 1: active; 2: former; 3: social",
    },
    {"variable_id": "lbxhdd", "domain": "clinical", "table": "clinical_measurements", "unit": "mg/dL"},
    {"variable_id": "lbdlld", "domain": "clinical", "table": "clinical_measurements", "unit": "mg/dL"},
    {"variable_id": "lbxtc", "domain": "clinical", "table": "clinical_measurements", "unit": "mg/dL"},
    {"variable_id": "bpxsym", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "dpxdim", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "bpxpls", "domain": "clinical", "table": "clinical_measurements", "unit": "bpm"},
    {"variable_id": "tas", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "tad", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "sys_dias_delta", "domain": "clinical", "table": "clinical_measurements", "unit": "mmHg"},
    {"variable_id": "hematocrit", "domain": "clinical", "table": "clinical_measurements", "unit": "%"},
    {"variable_id": "tacsctot", "domain": "clinical", "table": "clinical_measurements", "unit": "Agaston Units"},
    {"variable_id": "left_carotid_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "right_carotid_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "total_carotid_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "total_femoral_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "total_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "right_femoral_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {"variable_id": "left_femoral_plaque_vol", "domain": "clinical", "table": "clinical_measurements", "unit": "mm3"},
    {
        "variable_id": "apoe",
        "domain": "clinical",
        "table": "clinical_measurements",
        "unit": "Apolipoprotein E Aplotype Status",
    },
    {"variable_id": "flow_mean", "domain": "image", "table": "image_measurements", "unit": "mL/min"},
    {"variable_id": "flow_tseries", "domain": "image", "table": "image_measurements", "unit": "mL/min"},
    {"variable_id": "mean_cbf", "domain": "image", "table": "image_measurements", "unit": "mL/100g/min"},
]


def _reset_catalog_from_template(dataset_root: Path) -> None:
    """Replace catalog manifests with the clean repo template (long-form schemas only)."""
    if not CATALOG_TEMPLATE_ROOT.is_dir():
        return
    catalog_dst = dataset_root / "catalog"
    catalog_dst.mkdir(parents=True, exist_ok=True)
    for name in ("repository.json", "tables.json", "variables.json", "measurement_pipelines.json"):
        src = CATALOG_TEMPLATE_ROOT / name
        if src.is_file():
            shutil.copy2(src, catalog_dst / name)
    schema_src = CATALOG_TEMPLATE_ROOT / "schema"
    if schema_src.is_dir():
        schema_dst = catalog_dst / "schema"
        schema_dst.mkdir(parents=True, exist_ok=True)
        for schema_file in schema_src.glob("*.json"):
            shutil.copy2(schema_file, schema_dst / schema_file.name)


def _import_pesabrain_steps(
    repo: DataRepo,
    *,
    db_base_path: Path,
    new_vars_path: Path,
    import_run_id: str,
    log: Any,
) -> None:
    bases = {"raw": db_base_path, "new_vars": new_vars_path}
    for step in PESABRAIN_BUILD_STEPS:
        base = bases[step.base]
        log(f"import: {step.filename} ({step.source_kind}) from {base}")
        import_pesabrain_source(
            repo,
            base,
            step.filename,
            sheet=step.sheet,
            source_kind=step.source_kind,
            source_batch_id=import_run_id,
            rebuild_subjects=False,
            build_sqlite_index=False,
            pipeline_id=step.pipeline_id,
        )


def _run_new_vars_steps(
    repo: DataRepo,
    *,
    import_run_id: str,
    paths: dict[str, Path],
    log: Any,
) -> None:
    import_t1_volumetry(
        repo,
        paths["t1_cortical"],
        variable_id="t1_cortical_volume",
        atlas_key="cortical",
        source_batch_id=import_run_id,
        log=log,
    )
    import_t1_volumetry(
        repo,
        paths["t1_subcortical"],
        variable_id="t1_subcortical_volume",
        atlas_key="subcortical",
        source_batch_id=import_run_id,
        log=log,
    )
    import_cognitive_wide(repo, paths["cognitive"], source_batch_id=import_run_id, log=log)
    rename_sys_dias_delta_to_pp(repo, log=log)
    derive_pulse_pressure_map(repo, source_batch_id=import_run_id, log=log)
    derive_apoe_group(repo, source_batch_id=import_run_id, log=log)
    import_att_csv(repo, paths["att"], source_batch_id=import_run_id, log=log)
    import_wmh_csv(repo, paths["wmh"], source_batch_id=import_run_id, log=log)


def build_database(
    *,
    dataset_root: Path,
    db_base_path: Path,
    new_vars_path: Path,
    import_run_id: str,
    with_xnat: bool,
    xnat_server: str,
    xnat_project: str | None,
    xnat_config: Path | None,
    xnat_user: str | None,
    xnat_password: str | None,
    xnat_netrc: str | None,
    xnat_verify: bool,
    xnat_no_prompt: bool,
    xnat_sequences: str | None,
    index_local_dicom: bool,
    dicom_root: Path | None,
    index_local_nifti: bool,
    build_sqlite_index: bool,
    reset_catalog: bool,
    log: Any = print,
) -> DataRepo:
    dataset_root = dataset_root.expanduser().resolve()
    db_base_path = db_base_path.expanduser().resolve()
    new_vars_path = new_vars_path.expanduser().resolve()

    if reset_catalog and dataset_root.exists():
        log(f"Resetting catalog manifests from {CATALOG_TEMPLATE_ROOT}")
        _reset_catalog_from_template(dataset_root)

    repo = DataRepo(dataset_root, auto_scaffold=True, use_sqlite=True)
    log(f"Dropping all tables under {dataset_root}")
    repo.drop_all_tables()

    _import_pesabrain_steps(
        repo,
        db_base_path=db_base_path,
        new_vars_path=new_vars_path,
        import_run_id=import_run_id,
        log=log,
    )

    if index_local_nifti:
        log("Indexing local NIFTI assets")
        upsert_nifti_assets(repo, None, source="local_nifti", pipeline_id="raw")

    if index_local_dicom:
        if dicom_root is None:
            raise ValueError("--dicom-root is required when --index-local-dicom is set")
        log(f"Indexing local DICOM assets from {dicom_root}")
        upsert_dicom_assets(
            repo,
            str(dicom_root.expanduser().resolve()),
            source="local_dicom",
            pipeline_id="raw",
            build_sqlite_index=False,
        )

    if with_xnat:
        profile = load_xnat_profile(xnat_config)
        force_prompt = bool(
            xnat_config is not None
            and not xnat_password
            and not os.environ.get("XNAT_PASSWORD", "").strip()
        )
        base_config = resolve_xnat_connection(
            profile,
            server=xnat_server,
            project=xnat_project or XNAT_BUILD_PROJECTS[0],
            user=xnat_user,
            password=xnat_password,
            netrc_file=xnat_netrc,
            verify=xnat_verify,
        )
        base_config = finalize_xnat_connection(
            base_config,
            prompt_password=not xnat_no_prompt,
            force_prompt_password=force_prompt,
        )
        projects = (xnat_project,) if xnat_project else XNAT_BUILD_PROJECTS
        for project_id in projects:
            seq_filter = xnat_sequences or sequences_csv(default_sequences_for_project(project_id))
            log(f"Syncing XNAT project {project_id} from {xnat_server} (sequences: {seq_filter})")
            config = base_config if project_id == base_config.project else replace(
                base_config, project=project_id
            )
            sync_xnat_project(
                repo,
                config,
                requested_sequences=seq_filter,
                download_dicoms=False,
                download_niftis=False,
                build_sqlite_index=False,
                source_batch_id=import_run_id,
            )

    pruned = prune_redundant_local_sessions(repo)
    log(f"Pruned redundant local_db sessions; remaining sessions: {len(pruned)}")

    paths = dict(DEFAULT_PATHS)
    paths["att"] = db_base_path / "ATT_native_results.csv"
    if not paths["att"].is_file():
        paths["att"] = DEFAULT_PATHS["att"]
    _run_new_vars_steps(repo, import_run_id=import_run_id, paths=paths, log=log)

    normalize_session_visit_labels(repo)
    enrich_sessions_available_scans(repo)
    normalize_measurement_visit_ids(repo)
    log("Normalized visit labels and enriched sessions.available_scans")

    subjects = rebuild_subjects_table(repo)
    log(f"Rebuilt subjects table: {len(subjects)} rows")

    for entry in CURATED_VARIABLES:
        repo.register_variables([entry])

    if build_sqlite_index:
        repo.build_sqlite_index()
        log("SQLite index rebuilt")

    return repo


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root (catalog + tables/); dropped and rebuilt in place",
    )
    parser.add_argument("--db-base-path", type=Path, default=DEFAULT_DB_BASE, help="PESA-Brain Excel/CSV raw directory")
    parser.add_argument("--new-vars-path", type=Path, default=DEFAULT_NEW_VARS, help="Directory with ATT Excel files")
    parser.add_argument("--import-run-id", type=str, default=DEFAULT_IMPORT_RUN_ID, help="Stable provenance batch id")
    parser.add_argument(
        "--with-xnat",
        action="store_true",
        help="Sync XNAT catalog for PESA_Brain and IA_PET_V5 (Dixon/CT/PET + brain MRI incl. VWI_BB)",
    )
    parser.add_argument("--xnat-server", type=str, default="https://xnat.cnic.es")
    parser.add_argument(
        "--xnat-project",
        type=str,
        default=None,
        help="Sync only this XNAT project (default: PESA_Brain and IA_PET_V5)",
    )
    parser.add_argument(
        "--xnat-config",
        type=Path,
        default=None,
        help="Optional XNAT YAML/JSON profile (~/.config/nvitk/xnat.yaml or NVITK_XNAT_CONFIG)",
    )
    parser.add_argument("--xnat-user", type=str, default=None, help="XNAT username (or set XNAT_USER)")
    parser.add_argument(
        "--xnat-password",
        type=str,
        default=None,
        help="XNAT password (prefer XNAT_PASSWORD env or netrc; avoid shell history)",
    )
    parser.add_argument(
        "--xnat-netrc",
        type=str,
        default=None,
        help="netrc file (default: ~/.netrc when present)",
    )
    parser.add_argument(
        "--xnat-verify",
        action="store_true",
        help="Verify TLS certificates (default: off for CNIC self-signed CA)",
    )
    parser.add_argument(
        "--xnat-no-prompt",
        action="store_true",
        help="Do not prompt for XNAT password on the terminal",
    )
    parser.add_argument(
        "--xnat-sequences",
        type=str,
        default=None,
        help=(
            "Override sequence filter for all synced projects; default uses each project's "
            f"catalog defaults (combined: {DEFAULT_XNAT_SEQUENCES})"
        ),
    )
    parser.add_argument("--index-local-dicom", action="store_true")
    parser.add_argument("--dicom-root", type=Path, default=None)
    parser.add_argument("--index-local-nifti", action="store_true")
    parser.add_argument("--build-sqlite-index", action="store_true")
    parser.add_argument(
        "--no-reset-catalog",
        action="store_true",
        help="Keep existing catalog/tables.json instead of copying clean template first",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned steps only")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.dry_run:
        print(f"Would rebuild dataset at {args.dataset_root}")
        print(f"  db-base-path: {args.db_base_path}")
        print(f"  new-vars-path: {args.new_vars_path}")
        print(f"  import-run-id: {args.import_run_id}")
        print(f"  with-xnat: {args.with_xnat}")
        if args.with_xnat:
            projects = (args.xnat_project,) if args.xnat_project else XNAT_BUILD_PROJECTS
            print(f"  xnat-projects: {', '.join(projects)}")
            print(f"  xnat-sequences: {args.xnat_sequences or '(per-project defaults)'}")
            print(f"  default sequence union: {DEFAULT_XNAT_SEQUENCES}")
        print(f"  pesabrain steps: {len(PESABRAIN_BUILD_STEPS)}")
        return

    build_database(
        dataset_root=args.dataset_root,
        db_base_path=args.db_base_path,
        new_vars_path=args.new_vars_path,
        import_run_id=args.import_run_id,
        with_xnat=args.with_xnat,
        xnat_server=args.xnat_server,
        xnat_project=args.xnat_project,
        xnat_config=args.xnat_config,
        xnat_user=args.xnat_user,
        xnat_password=args.xnat_password,
        xnat_netrc=args.xnat_netrc,
        xnat_verify=args.xnat_verify,
        xnat_no_prompt=args.xnat_no_prompt,
        xnat_sequences=args.xnat_sequences,
        index_local_dicom=args.index_local_dicom,
        dicom_root=args.dicom_root,
        index_local_nifti=args.index_local_nifti,
        build_sqlite_index=args.build_sqlite_index,
        reset_catalog=not args.no_reset_catalog,
    )


if __name__ == "__main__":
    main()
