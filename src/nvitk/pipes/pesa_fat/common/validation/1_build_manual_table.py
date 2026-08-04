"""
Paso 1 — Construir tabla de medidas manuales
============================================
Lee los dos ficheros manuales y produce una tabla unificada con una fila
por participante (pesa_id como índice).

Salida: RESULTS/Manual_Measurements.xlsx

Fuentes:
  FILE1: análisis manual médula y fat fraction (n≈23)
         SUV L3/L4 (max, mean), médula ósea, hígado Dixon
  FILE2: RefStd_FAT_PESA_PRUEBAS_manual_DJC.xlsx (n=6, semana 1)
         DIXON riñón, páncreas, cuádriceps, paravertebral, médula L3/L4
         PET hígado, bazo, páncreas, cuádriceps, paravertebral
"""

import sys
import numpy as np
import pandas as pd
from nvitk.core.logger import Logger

sys.path.insert(0, __file__.rsplit("\\", 1)[0])   # permite importar config.py
from config import MANUAL_PATH, MANUAL_PATH2, MANUAL_TABLE_PATH

log = Logger()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_spanish_value(val):
    """Extrae el valor numérico de strings 'X,X±Y,Y' (notación española)."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    for sep in ["±", "±", "+"]:
        s = s.split(sep)[0]
    s = s.replace(",", ".").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# FILE1 — médula + fat fraction (n≈23)
# ─────────────────────────────────────────────────────────────────────────────

def load_manual(path):
    """
    Layout FILE1 (columnas relevantes):
      col2:  L3_SUVmax   col3:  L3_SUVmean   col5:  L3_Volume
      col7:  L4_SUVmax   col8:  L4_SUVmean   col10: L4_Volume
      col11: Medula_SUVmax_avg (manual)
      col12: Liver_FF    col13: Liver_VOL
      col14: Liver_T2    col15: Liver_R2
    """
    raw = pd.read_excel(path, header=None)
    rows = []
    for i in range(2, len(raw)):
        row = raw.iloc[i]
        pid = str(row.iloc[0]).strip()
        if not pid.startswith("PESA"):
            continue
        rows.append({
            "pesa_id":               pid,
            "man_L3_SUVmax":         parse_spanish_value(row.iloc[2]),
            "man_L3_SUVmean":        parse_spanish_value(row.iloc[3]),
            "man_L3_Volume":         parse_spanish_value(row.iloc[5]),
            "man_L4_SUVmax":         parse_spanish_value(row.iloc[7]),
            "man_L4_SUVmean":        parse_spanish_value(row.iloc[8]),
            "man_L4_Volume":         parse_spanish_value(row.iloc[10]),
            "man_Medula_SUVmax_avg": parse_spanish_value(row.iloc[11]),
            "man_Liver_FF":          parse_spanish_value(row.iloc[12]),
            "man_Liver_VOL":         parse_spanish_value(row.iloc[13]),
            "man_Liver_T2":          parse_spanish_value(row.iloc[14]),
            "man_Liver_R2":          parse_spanish_value(row.iloc[15]),
        })
    return pd.DataFrame(rows).set_index("pesa_id")


# ─────────────────────────────────────────────────────────────────────────────
# FILE2 — RefStd semana 1 (n=6)
# ─────────────────────────────────────────────────────────────────────────────

def load_manual2(path):
    """
    Layout FILE2 (filas 0-2 = cabeceras, filas 3+ = datos):
      DIXON RIÑON DER   : F(1)  W(2)  T2(3)  FF(4)  Vol(5)
      DIXON RIÑON IZQ   : F(6)  W(7)  T2(8)  FF(9)  Vol(10)
      DIXON PANCREAS    : F(11) W(12) T2(13) FF(14) Vol(15)
      DIXON CUAD DER    : F(16) W(17) T2(18) FF(19) Vol(20)
      DIXON CUAD IZQ    : F(21) W(22) T2(23) FF(24) Vol(25)
      DIXON PARAVERT DER: F(26) W(27) T2(28) FF(29) Vol(30)
      DIXON PARAVERT IZQ: F(31) W(32) T2(33) FF(34) Vol(35)
      DIXON MEDULA L3   : F(36) W(37) T2(38) FF(39) Vol(40)
      DIXON MEDULA L4   : F(41) W(42) T2(43) FF(44) Vol(45)
      PET PARAVERT DER  : MIN(46) MAX(47) MEAN(48) DE(49)
      PET PARAVERT IZQ  : MIN(50) MAX(51) MEAN(52) DE(53)
      PET RINON DER     : MIN(54) MAX(55) MEAN(56) DE(57)
      PET RINON IZQ     : MIN(58) MAX(59) MEAN(60) DE(61)
      PET CUAD DER      : MIN(62) MAX(63) MEAN(64) DE(65)
      PET CUAD IZQ      : MIN(66) MAX(67) MEAN(68) DE(69)
      PET HIGADO        : MIN(70) MAX(71) MEAN(72) DE(73)
      PET BAZO          : MIN(74) MAX(75) MEAN(76) DE(77)
      PET PANCREAS      : MIN(78) MAX(79) MEAN(80) DE(81)
    """
    df_raw = pd.read_excel(path, header=None)
    data = df_raw.iloc[3:].reset_index(drop=True)
    data.columns = range(len(data.columns))

    def g(row, col):
        """Read row[col] as a finite float, or NaN if missing/non-numeric/non-finite."""
        try:
            v = float(row[col])
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError, KeyError):
            return np.nan

    def mean2(row, c1, c2):
        """Mean of the finite values at columns c1/c2 (ignoring NaNs), or NaN if both are missing."""
        a, b = g(row, c1), g(row, c2)
        valid = [x for x in [a, b] if not np.isnan(x)]
        return float(np.mean(valid)) if valid else np.nan

    def sum2(row, c1, c2):
        """Sum of the finite values at columns c1/c2 (ignoring NaNs), or NaN if both are missing."""
        a, b = g(row, c1), g(row, c2)
        if np.isnan(a) and np.isnan(b):
            return np.nan
        return float(np.nansum([a, b]))

    rows = []
    for _, row in data.iterrows():
        pid = str(row[0]).strip()
        if not pid.startswith("PESA"):
            continue
        r = {"pesa_id": pid}

        # ── DIXON Riñón ────────────────────────────────────────────────────
        r["man2_Rinon_DER_T2"]  = g(row, 3);  r["man2_Rinon_DER_FF"]  = g(row, 4)
        r["man2_Rinon_DER_VOL"] = g(row, 5)
        r["man2_Rinon_IZQ_T2"]  = g(row, 8);  r["man2_Rinon_IZQ_FF"]  = g(row, 9)
        r["man2_Rinon_IZQ_VOL"] = g(row, 10)
        r["man2_Rinon_BIL_T2"]  = mean2(row, 3, 8)
        r["man2_Rinon_BIL_FF"]  = mean2(row, 4, 9)
        r["man2_Rinon_BIL_VOL"] = sum2(row, 5, 10)

        # ── DIXON Páncreas ─────────────────────────────────────────────────
        r["man2_Pancreas_T2"]  = g(row, 13)
        r["man2_Pancreas_FF"]  = g(row, 14)
        r["man2_Pancreas_VOL"] = g(row, 15)

        # ── DIXON Cuádriceps ───────────────────────────────────────────────
        r["man2_Cuad_DER_T2"]  = g(row, 18);  r["man2_Cuad_DER_FF"]  = g(row, 19)
        r["man2_Cuad_DER_VOL"] = g(row, 20)
        r["man2_Cuad_IZQ_T2"]  = g(row, 23);  r["man2_Cuad_IZQ_FF"]  = g(row, 24)
        r["man2_Cuad_IZQ_VOL"] = g(row, 25)
        r["man2_Cuad_BIL_T2"]  = mean2(row, 18, 23)
        r["man2_Cuad_BIL_FF"]  = mean2(row, 19, 24)
        r["man2_Cuad_BIL_VOL"] = sum2(row, 20, 25)

        # ── DIXON Paravertebral ────────────────────────────────────────────
        r["man2_Paravert_DER_T2"]  = g(row, 28)
        r["man2_Paravert_DER_FF"]  = g(row, 29)
        r["man2_Paravert_DER_VOL"] = g(row, 30)
        r["man2_Paravert_IZQ_T2"]  = g(row, 33)
        r["man2_Paravert_IZQ_FF"]  = g(row, 34)
        r["man2_Paravert_IZQ_VOL"] = g(row, 35)
        r["man2_Paravert_BIL_T2"]  = mean2(row, 28, 33)
        r["man2_Paravert_BIL_FF"]  = mean2(row, 29, 34)
        r["man2_Paravert_BIL_VOL"] = sum2(row, 30, 35)

        # ── DIXON Médula L3 / L4 ──────────────────────────────────────────
        r["man2_BN_L3_T2"]  = g(row, 38);  r["man2_BN_L3_FF"]  = g(row, 39)
        r["man2_BN_L3_VOL"] = g(row, 40)
        r["man2_BN_L4_T2"]  = g(row, 43);  r["man2_BN_L4_FF"]  = g(row, 44)
        r["man2_BN_L4_VOL"] = g(row, 45)

        # ── PET Paravertebral ─────────────────────────────────────────────
        r["man2_Paravert_DER_SUVmax"] = g(row, 47)
        r["man2_Paravert_DER_SUVmed"] = g(row, 48)
        r["man2_Paravert_IZQ_SUVmax"] = g(row, 51)
        r["man2_Paravert_IZQ_SUVmed"] = g(row, 52)
        r["man2_Paravert_BIL_SUVmax"] = mean2(row, 47, 51)
        r["man2_Paravert_BIL_SUVmed"] = mean2(row, 48, 52)

        # ── PET Riñón (sin equivalente automático actualmente) ────────────
        r["man2_Rinon_DER_SUVmax"] = g(row, 55)
        r["man2_Rinon_DER_SUVmed"] = g(row, 56)
        r["man2_Rinon_IZQ_SUVmax"] = g(row, 59)
        r["man2_Rinon_IZQ_SUVmed"] = g(row, 60)
        r["man2_Rinon_BIL_SUVmax"] = mean2(row, 55, 59)
        r["man2_Rinon_BIL_SUVmed"] = mean2(row, 56, 60)

        # ── PET Cuádriceps ─────────────────────────────────────────────────
        r["man2_Cuad_DER_SUVmax"] = g(row, 63)
        r["man2_Cuad_DER_SUVmed"] = g(row, 64)
        r["man2_Cuad_IZQ_SUVmax"] = g(row, 67)
        r["man2_Cuad_IZQ_SUVmed"] = g(row, 68)
        r["man2_Cuad_BIL_SUVmax"] = mean2(row, 63, 67)
        r["man2_Cuad_BIL_SUVmed"] = mean2(row, 64, 68)

        # ── PET Hígado ─────────────────────────────────────────────────────
        r["man2_Higado_SUVmax"] = g(row, 71)
        r["man2_Higado_SUVmed"] = g(row, 72)

        # ── PET Bazo ───────────────────────────────────────────────────────
        r["man2_Bazo_SUVmax"] = g(row, 75)
        r["man2_Bazo_SUVmed"] = g(row, 76)

        # ── PET Páncreas ───────────────────────────────────────────────────
        r["man2_Pancreas_SUVmax"] = g(row, 79)
        r["man2_Pancreas_SUVmed"] = g(row, 80)

        rows.append(r)

    return pd.DataFrame(rows).set_index("pesa_id")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Paso 1 — Construyendo tabla de medidas manuales…")

    df1 = load_manual(MANUAL_PATH)
    df2 = load_manual2(MANUAL_PATH2)

    df_manual = df1.join(df2, how="outer")

    df_manual.to_excel(MANUAL_TABLE_PATH, index=True)
    log.info(f"  FILE1: {df1.shape[0]} participantes, {df1.shape[1]} variables")
    log.info(f"  FILE2: {df2.shape[0]} participantes, {df2.shape[1]} variables")
    log.info(f"  Tabla unificada: {df_manual.shape[0]} filas × {df_manual.shape[1]} columnas")
    log.info(f"  OK {MANUAL_TABLE_PATH}")
