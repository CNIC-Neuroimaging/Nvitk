"""
Configuración compartida — PESA Fat Analysis
"""
import os
from pathlib import Path

BASE                = Path("/data3/BIOIT_IMAGE/PESA_Fat/DATA/Visit-5-DIXON_PET-CT")
BASE_LOCAL          = Path("/home/imarcoss/DATA/BioIT/PESA-Fat/RESULTS/")
OUT_DIR             = BASE_LOCAL / "Validation_202602_v2"

# ── Fuentes manuales ──────────────────────────────────────────────────────────
MANUAL_PATH         = BASE / "MANUAL" / "raw" / "análisis manual médula y fat fraction- participantes febrero 2026_DJC.xlsx" 
MANUAL_PATH2        = BASE / "MANUAL" / "raw" / "RefStd_FAT_PESA_PRUEBAS_manual_DJC.xlsx"

# ── Datos automáticos ─────────────────────────────────────────────────────────
AUTO_BASE           = BASE / "RESULTS"
AUTO_MONTH          = "202602"
AUTO_WEEKS          = [1, 2, 3, 4]              # semanas disponibles
AUTO_WEEK_FMT       = "202602_Week{w}"   # formato del nombre de carpeta/fichero

# ── Ficheros intermedios / de salida ──────────────────────────────────────────
MANUAL_TABLE_PATH   = BASE / "MANUAL" / "Manual_Measurements.xlsx"
COMBINED_TABLE_PATH = OUT_DIR / "Combined_Measurements.xlsx"
METRICS_PATH        = OUT_DIR / "Metricas_Manual_vs_Auto.xlsx"
REPORT_PATH         = OUT_DIR / "Report_Manual_vs_Auto.html"

OUT_DIR.mkdir(parents=True, exist_ok=True)
