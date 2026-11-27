"""Dataset import rules as configuration: identifier columns, visits, sources, local presets.

``dataset.json`` (searched like ``sge.json``; ``$NVITK_DATASET_CONFIG`` names it outright) says
how a dataset is assembled from spreadsheets and folders, so the importers carry no knowledge
of one project's files:

* ``id_columns``: which columns may hold the subject / visit / session / date (first present
  wins, case- and spacing-insensitive), which column names are identifiers rather than
  variables (``namespace_exact``, ``identifier_prefixes``).
* ``visits``, ``default_visit_label``, ``local_project_id``.
* ``clinical_metadata_columns``: clinical columns an image sheet repeats, skipped when its
  measurements are melted.
* ``sources``: one entry per importable (file, sheet): ``filename`` (or a glob), ``sheet``,
  ``source_kind`` (``subject_ids`` / ``subject_catalog`` / ``clinical_wide`` / ``image_wide`` /
  ``image_timeseries_wide``), ``domain``, ``layout``, ``modality``, ``cohort_id``, ``batch_id``,
  ``default_visit_label``, ``default_pipeline_id`` (the measurement pipeline *version* the
  values belong to).
* ``local_presets`` and ``cohort_dir_rules``: how the Data browser walks pipeline folders on disk.

Without a ``dataset.json``, the packaged ``defaults/dataset.json`` applies: generic identifier
columns and no sources, so nothing is imported until a dataset is configured.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

CONFIG_NAME = "dataset.json"
ENV_VAR = "NVITK_DATASET_CONFIG"

_DEFAULTS_DIR = Path(__file__).with_name("defaults")
DEFAULT_PATH = _DEFAULTS_DIR / "dataset.json"
_SCHEMA_PATH = Path(__file__).with_name("schema") / "dataset.schema.json"

#: Fields a source entry may carry, in :class:`~nvitk.db.importers.SourceSpec` order.
SOURCE_FIELDS: tuple[str, ...] = (
    "filename",
    "sheet",
    "source_kind",
    "domain",
    "layout",
    "modality",
    "cohort_id",
    "batch_id",
    "default_visit_label",
    "default_pipeline_id",
)


@dataclass
class DatasetConfig:
    """The parsed ``dataset.json``."""

    name: str
    subject_columns: list[str]
    visit_columns: list[str]
    session_columns: list[str]
    date_columns: list[str]
    namespace_exact: set[str]
    identifier_prefixes: tuple[str, ...]
    local_project_id: str
    default_visit_label: str
    visits: dict[str, str]
    clinical_metadata_columns: set[str]
    sources: list[dict[str, Any]]
    local_presets: list[dict[str, Any]]
    cohort_dir_rules: dict[str, Any] = field(default_factory=dict)
    plaque_source_filename: str = ""
    source: str = ""

    def specs_by_filename(self) -> dict[str, list[dict[str, Any]]]:
        """Source entries grouped by file name, in file order (a file may have several sheets)."""
        out: dict[str, list[dict[str, Any]]] = {}
        for entry in self.sources:
            out.setdefault(str(entry["filename"]), []).append(entry)
        return out

    def specs_for_file(self, filename: str) -> list[dict[str, Any]]:
        """Source entries that apply to *filename*: exact name first, else matching globs."""
        name = Path(str(filename)).name
        exact = [e for e in self.sources if str(e["filename"]) == name]
        if exact:
            return exact
        return [e for e in self.sources if any(ch in str(e["filename"]) for ch in "*?[") and fnmatch.fnmatch(name, str(e["filename"]))]

    def is_cohort_dir_name(self, name: str) -> bool:
        """Whether a folder under a pipeline root is a cohort / batch folder (by ``cohort_dir_rules``)."""
        rules = self.cohort_dir_rules or {}
        if not name:
            return False
        if any(name.startswith(p) for p in rules.get("reject_prefixes", ())):
            return False
        if any(name.startswith(p) for p in rules.get("reject_startswith", ())):
            return False
        upper = name.upper()
        if any(upper.startswith(str(p).upper()) for p in rules.get("reject_upper_prefixes", ())):
            return False
        flags = re.IGNORECASE if "i" in str(rules.get("accept_regex_flags", "i")) else 0
        if any(re.match(p, name, flags) for p in rules.get("accept_regex", ())):
            return True
        if any(upper.startswith(str(p).upper()) for p in rules.get("accept_upper_prefixes", ())):
            return True
        if any(name.startswith(p) for p in rules.get("accept_prefixes", ())):
            return True
        return False


def _parse(doc: Mapping[str, Any], source: str = "") -> DatasetConfig:
    ids = dict(doc.get("id_columns") or {})
    sources = []
    for entry in doc.get("sources") or ():
        clean = {k: entry.get(k) for k in SOURCE_FIELDS}
        clean["filename"] = str(clean["filename"])
        sources.append(clean)
    return DatasetConfig(
        name=str(doc.get("name") or ""),
        subject_columns=[str(c) for c in ids.get("subject", ())],
        visit_columns=[str(c) for c in ids.get("visit", ())],
        session_columns=[str(c) for c in ids.get("session", ())],
        date_columns=[str(c) for c in ids.get("date", ())],
        namespace_exact={str(c) for c in ids.get("namespace_exact", ())},
        identifier_prefixes=tuple(str(c) for c in ids.get("identifier_prefixes", ())),
        local_project_id=str(doc.get("local_project_id") or "local_db"),
        default_visit_label=str(doc.get("default_visit_label") or "1"),
        visits={str(k): str(v) for k, v in dict(doc.get("visits") or {}).items()},
        clinical_metadata_columns={str(c) for c in doc.get("clinical_metadata_columns", ())},
        sources=sources,
        local_presets=[dict(p) for p in doc.get("local_presets") or ()],
        cohort_dir_rules=dict(doc.get("cohort_dir_rules") or {}),
        plaque_source_filename=str(doc.get("plaque_source_filename") or ""),
        source=source,
    )


def config_path() -> Path | None:
    """The ``dataset.json`` the configuration search finds, or ``None``."""
    from nvitk.core import config_paths

    return config_paths.config_file(CONFIG_NAME)


def validate(doc: Mapping[str, Any]) -> list[str]:
    """Problems with *doc* against the schema; empty when valid."""
    problems: list[str] = []
    try:
        import jsonschema

        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        for error in jsonschema.Draft202012Validator(schema).iter_errors(doc):
            where = "/".join(str(p) for p in error.absolute_path) or "(root)"
            problems.append(f"{where}: {error.message}")
    except ImportError:  # pragma: no cover - jsonschema is optional
        pass
    for i, entry in enumerate(doc.get("sources") or ()):
        if not isinstance(entry, Mapping):
            continue
        for key in ("filename", "sheet", "source_kind", "domain", "layout"):
            if not entry.get(key):
                problems.append(f"sources/{i}: '{key}' is required")
    return problems


@lru_cache(maxsize=None)
def _load(path_text: str, mtime: float) -> DatasetConfig:
    del mtime
    path = Path(path_text)
    doc = json.loads(path.read_text(encoding="utf-8"))
    problems = validate(doc)
    if problems:
        raise ValueError(f"{path} is not a valid dataset config:\n  " + "\n  ".join(problems[:20]))
    return _parse(doc, source=str(path))


def load_dataset_config(path: str | Path | None = None) -> DatasetConfig:
    """The dataset config in force: *path*, else the configured file, else the packaged default."""
    target = Path(path).expanduser() if path else (config_path() or DEFAULT_PATH)
    return _load(str(target), target.stat().st_mtime)


def default_document() -> dict[str, Any]:
    """The packaged default as a JSON document (the starting point of a new ``dataset.json``)."""
    return json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))


__all__ = [
    "CONFIG_NAME",
    "DatasetConfig",
    "ENV_VAR",
    "DEFAULT_PATH",
    "SOURCE_FIELDS",
    "config_path",
    "default_document",
    "load_dataset_config",
    "validate",
]
