"""
Configuración compartida — PESA Fat Analysis (validación manual vs automática).

Los directorios se leen de ``sge.json`` (sección ``pipelines.pesa_fat_validation``); no hay
rutas de instalación codificadas aquí. Las constantes se resuelven de forma perezosa mediante
``__getattr__`` (PEP 562), así que importar este módulo no lee configuración ni crea
directorios — sólo lo hace al usar una constante concreta.

Ejemplo de configuración::

    "pipelines": {
      "pesa_fat_validation": {
        "base_root":   "<RAIZ_DATOS_VISITA>",
        "output_root": "<RAIZ_SALIDA_VALIDACION>",
        "run_name":    "Validation_202602_v2",
        "auto_month":  "202602",
        "auto_weeks":  [1, 2, 3, 4]
      }
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvitk.core import config_paths
from nvitk.cluster import sge_json

_PIPELINE_ID = "pesa_fat_validation"

#: Formato del nombre de carpeta/fichero semanal.
AUTO_WEEK_FMT = "{month}_Week{w}"

# Nombres de los libros de origen manual dentro de ``<base_root>/MANUAL/raw/``.
_MANUAL_FILES = (
    "análisis manual médula y fat fraction- participantes febrero 2026_DJC.xlsx",
    "RefStd_FAT_PESA_PRUEBAS_manual_DJC.xlsx",
)


def _section() -> dict[str, Any]:
    """La sección ``pipelines.pesa_fat_validation`` de ``sge.json`` (``{}`` si no existe)."""
    return sge_json.pipeline_section(_PIPELINE_ID)


def _root(key: str) -> Path:
    """Una raíz obligatoria de la sección, o un error indicando qué falta y dónde se buscó."""
    value = config_paths.require(
        _section().get(key),
        key=f"pipelines.{_PIPELINE_ID}.{key}",
        hint="Esta utilidad de validación necesita saber dónde están los datos de entrada y "
             "dónde escribir los resultados.",
    )
    return Path(str(value)).expanduser()


def output_dir(*, create: bool = False) -> Path:
    """Directorio de salida de esta ejecución; se crea sólo si *create* es verdadero.

    La creación es explícita: antes ocurría al importar el módulo, de modo que basta con
    importar la configuración para escribir en disco.
    """
    path = _root("output_root") / str(_section().get("run_name", "validation"))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve(name: str) -> Any:
    """Valor de la constante *name*, calculado en el momento de usarla."""
    if name == "BASE":
        return _root("base_root")
    if name == "BASE_LOCAL":
        return _root("output_root")
    if name == "OUT_DIR":
        return output_dir()
    if name == "MANUAL_PATH":
        return _root("base_root") / "MANUAL" / "raw" / _MANUAL_FILES[0]
    if name == "MANUAL_PATH2":
        return _root("base_root") / "MANUAL" / "raw" / _MANUAL_FILES[1]
    if name == "MANUAL_TABLE_PATH":
        return _root("base_root") / "MANUAL" / "Manual_Measurements.xlsx"
    if name == "AUTO_BASE":
        return _root("base_root") / "RESULTS"
    if name == "AUTO_MONTH":
        return str(_section().get("auto_month", ""))
    if name == "AUTO_WEEKS":
        return list(_section().get("auto_weeks", []))
    if name == "COMBINED_TABLE_PATH":
        return output_dir() / "Combined_Measurements.xlsx"
    if name == "METRICS_PATH":
        return output_dir() / "Metricas_Manual_vs_Auto.xlsx"
    if name == "REPORT_PATH":
        return output_dir() / "Report_Manual_vs_Auto.html"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:
    """Resolución perezosa de las constantes del módulo (PEP 562)."""
    return _resolve(name)


__all__ = [
    "AUTO_BASE", "AUTO_MONTH", "AUTO_WEEK_FMT", "AUTO_WEEKS", "BASE", "BASE_LOCAL",
    "COMBINED_TABLE_PATH", "MANUAL_PATH", "MANUAL_PATH2", "MANUAL_TABLE_PATH", "METRICS_PATH",
    "OUT_DIR", "REPORT_PATH", "output_dir",
]
