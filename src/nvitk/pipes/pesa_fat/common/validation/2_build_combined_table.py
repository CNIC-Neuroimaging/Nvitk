"""
Paso 2 — Construir tabla combinada (manual + automático)
=========================================================
Lee la tabla de medidas manuales (producida por el paso 1) y los codebooks
automáticos semanales, los une y añade columnas derivadas.

Entrada : RESULTS/Manual_Measurements.xlsx
          RESULTS/<tag>/res_measure_{dixon|suv}/<tag>_SummaryCodebook.xlsx
Salida  : RESULTS/Combined_Measurements.xlsx

Columnas derivadas añadidas:
  auto_Medula_SUVmax_avg  — media correcta L3+L4 (el pipeline guarda el MAX)
"""

import sys
import os
import pandas as pd

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from config import (AUTO_BASE, AUTO_MONTH, AUTO_WEEKS,
                    MANUAL_TABLE_PATH, COMBINED_TABLE_PATH)


_STAGE3_DIRS = {"DIXON": "res_measure_dixon", "SUV": "res_measure_suv"}


def load_auto_weeks(modality):
    """Concatena los SummaryCodebook de todas las semanas configuradas."""
    frames = []
    for w in AUTO_WEEKS:
        tag  = f"{AUTO_MONTH}_Week{w}"
        path = AUTO_BASE / tag / _STAGE3_DIRS[modality] / f"{tag}_SummaryCodebook.xlsx"
        df   = pd.read_excel(path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={combined.columns[0]: "pesa_id"})
    return combined.set_index("pesa_id")


if __name__ == "__main__":
    print("Paso 2 — Construyendo tabla combinada…")

    # ── Cargar manual ──────────────────────────────────────────────────────
    if not os.path.exists(MANUAL_TABLE_PATH):
        sys.exit(f"ERROR: No se encuentra {MANUAL_TABLE_PATH}\n"
                 "Ejecuta primero: python 1_build_manual_table.py")

    df_man = pd.read_excel(MANUAL_TABLE_PATH, index_col=0)
    print(f"  Manual: {df_man.shape[0]} participantes, {df_man.shape[1]} variables")

    # ── Cargar automático ──────────────────────────────────────────────────
    df_suv = load_auto_weeks("SUV")
    df_dix = load_auto_weeks("DIXON")
    print(f"  Auto SUV : {df_suv.shape[0]} participantes, {df_suv.shape[1]} variables")
    print(f"  Auto DIXON: {df_dix.shape[0]} participantes, {df_dix.shape[1]} variables")

    # ── Unir ───────────────────────────────────────────────────────────────
    df = df_man.join(df_suv, how="outer").join(df_dix, how="outer")

    # ── Columnas derivadas ─────────────────────────────────────────────────
    # [NOTA 1] MO_SUVMAX es el MAX sobre la unión L3∪L4; la métrica manual es
    # la media de los dos máximos por vértebra.
    df["auto_Medula_SUVmax_avg"] = (df["L3_SUVMAX"] + df["L4_SUVMAX"]) / 2

    # ── Guardar ────────────────────────────────────────────────────────────
    df.index.name = "pesa_id"
    df.to_excel(COMBINED_TABLE_PATH, index=True)
    print(f"  Tabla combinada: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"  OK {COMBINED_TABLE_PATH}")
