"""XNAT project mapping as configuration: projects, scan classification rules, asset slots.

What used to be code (``XNAT_PROJECTS``, the ``classify_scan`` regex cascade, the IA_PET_V5
classifier, the sequence → asset-slot table, the pipeline resource maps) is a JSON document,
``xnat_rules.json``, found by the usual configuration search (``$NVITK_XNAT_RULES`` names it
outright). Without one, nvitk uses the packaged ``defaults/pesa_xnat_rules.json``, which
reproduces the original PESA_Brain / IA_PET_V5 behaviour exactly; ``nvitk-config
generate-pesa`` writes that file out to edit, and ``nvitk-config init`` a generic starter.

The document, in short (``schema/xnat_rules.schema.json`` has it all)::

    {
      "projects":   {"<XNAT project id>": {"display_name", "cohort_id", "classifier",
                                           "default_sequences": [...]}},
      "visit_labels": {"<project id>": "<visit label>"},
      "classifiers": {
        "<name>": {
          "quality": {"accept": ["usable"], "strip": false},
          "rules": [                       # first match wins
            {"name": "TOF",
             "match":     {"series_description": "tof|mra"},      # every field must match
             "not_match": {"series_description": "localizer"},    # and none of these
             "ignore_quality": false,      # true: applies even to a scan not accepted by quality
             "extract": {"orientation": {"type": "substring_map", ...}},
             "result": {"modality": "4dflow", "sequence": "4DFLOW_{orientation}"}}
          ],
          "experiment": {"steps": [{"type": "dixon_regions", ...}], "dedupe_sequences": true}
        }
      },
      "sequence_aliases": {"4DFLOW": ["4DFLOW_AP", ...]},
      "asset_slots": {"TOF": "tof", ...}, "asset_slot_labels": {"tof": "TOF", ...},
      "pipeline_resources": {"<resource label>": {"asset_slot", "pipeline_id", "name",
                                                  "bundle", "gui"}},
      "pipeline_filter_slots": [...]
    }

Regexes are Python ``re`` patterns, case-insensitive unless a rule sets ``"flags": ""``. A
scan's fields are whatever the caller passes (``series_description`` and ``quality`` always;
``type``, ``frames``... when known). Extractors compute values a result can use as
``{name}``; they are registered in :data:`EXTRACTORS`, so a site with an unusual naming scheme
adds one function rather than editing the classifier. Experiment-level steps (assigning Dixon
stations by slice location) are registered in :data:`EXPERIMENT_STEPS` the same way.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

#: Configuration file name, searched like ``sge.json`` and friends.
CONFIG_NAME = "xnat_rules.json"
#: Environment variable naming the file outright.
ENV_VAR = "NVITK_XNAT_RULES"

_DEFAULTS_DIR = Path(__file__).with_name("defaults")
PESA_DEFAULT_PATH = _DEFAULTS_DIR / "pesa_xnat_rules.json"
_SCHEMA_PATH = Path(__file__).with_name("schema") / "xnat_rules.schema.json"


# Extractors
def _field(scan: Mapping[str, Any], name: str) -> str:
    value = scan.get(name)
    return "" if value is None else str(value)


def _extract_substring_map(scan: Mapping[str, Any], spec: Mapping[str, Any]) -> str | None:
    """First key of ``map`` one of whose tokens occurs in the field (upper-cased with ``upper``)."""
    text = _field(scan, spec.get("field", "series_description"))
    if spec.get("upper", True):
        text = text.upper()
    for key, tokens in dict(spec.get("map") or {}).items():
        if any(str(token) in text for token in tokens):
            return str(key)
    default = spec.get("default")
    return None if default is None else str(default)


def _extract_regex_group(scan: Mapping[str, Any], spec: Mapping[str, Any]) -> str | None:
    """A regex group of the field; or, failing that, the scan's quality when it is one of
    ``fallback_quality_in`` (BrainVIEW's strong/default/weak variants live in either place)."""
    text = _field(scan, spec.get("field", "series_description")).strip()
    found = re.match(str(spec["pattern"]), text, _flags(spec.get("flags", "i"))) if spec.get(
        "anchored", True
    ) else re.search(str(spec["pattern"]), text, _flags(spec.get("flags", "i")))
    if found:
        value = found.group(int(spec.get("group", 1)))
        return value.lower() if spec.get("lower") else value
    fallback = [str(v).lower() for v in spec.get("fallback_quality_in") or ()]
    quality = _field(scan, "quality").strip().lower()
    if fallback and quality in fallback:
        return quality
    default = spec.get("default")
    return None if default is None else str(default)


#: Extractor type -> ``fn(scan_fields, spec) -> str | None``. ``None`` means "no value": a
#: rule whose extractor is ``"required": true`` then does not apply.
EXTRACTORS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], str | None]] = {
    "substring_map": _extract_substring_map,
    "regex_group": _extract_regex_group,
}


def _dixon_regions_step(
    candidates: list[tuple[Any, str]], step: Mapping[str, Any]
) -> dict[str, str]:
    """Dixon station (HEAD / THORAX / LEGS) per scan id, from the scans' slice locations."""
    from nvitk.pipes.pesa_fat.common.dixon_regions import classify_dixon_scans_by_slice_location

    scan_ids = [scan_id for _, scan_id in candidates]
    scans_by_id = {scan_id: scan for scan, scan_id in candidates}
    return dict(classify_dixon_scans_by_slice_location(scan_ids, scans_by_id))


#: Experiment step type -> ``fn([(scan, scan_id)], step) -> {scan_id: sequence}``, run over
#: the scans a classifier marked with the step's ``candidate_flag``.
EXPERIMENT_STEPS: dict[str, Callable[[list[tuple[Any, str]], Mapping[str, Any]], dict[str, str]]] = {
    "dixon_regions": _dixon_regions_step,
}


def _flags(text: str | None) -> int:
    flags = 0
    for letter in str(text or ""):
        flags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}.get(letter, 0)
    return flags


# Model
@dataclass(frozen=True)
class ScanRule:
    """One classification rule (see the module docstring)."""

    name: str
    match: dict[str, str]
    not_match: dict[str, str] = field(default_factory=dict)
    ignore_quality: bool = False
    extract: dict[str, dict[str, Any]] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    flags: str = "i"

    def apply(self, scan: Mapping[str, Any]) -> dict[str, Any] | None:
        """The classification for *scan*, or ``None`` when this rule does not apply."""
        flags = _flags(self.flags)
        for name, pattern in self.match.items():
            if not re.search(pattern, _field(scan, name), flags):
                return None
        for name, pattern in self.not_match.items():
            if re.search(pattern, _field(scan, name), flags):
                return None
        values: dict[str, Any] = {}
        for name, spec in self.extract.items():
            fn = EXTRACTORS.get(str(spec.get("type")))
            if fn is None:
                raise ValueError(f"Rule {self.name!r}: unknown extractor type {spec.get('type')!r}")
            value = fn(scan, spec)
            if value is None and spec.get("required"):
                return None
            values[name] = value
        out: dict[str, Any] = {}
        for key, value in self.result.items():
            if isinstance(value, str) and "{" in value:
                try:
                    out[key] = value.format(**values)
                except (KeyError, IndexError):
                    out[key] = value
            else:
                out[key] = copy.deepcopy(value)
        return out


@dataclass(frozen=True)
class Classifier:
    """A named rule set: quality gate, ordered rules, optional experiment-level steps."""

    name: str
    rules: tuple[ScanRule, ...]
    quality_accept: tuple[str, ...] = ("usable",)
    quality_strip: bool = False
    experiment_steps: tuple[dict[str, Any], ...] = ()
    dedupe_sequences: bool = False

    @property
    def experiment_level(self) -> bool:
        """Whether scans need the whole experiment to be classified (e.g. Dixon stations)."""
        return bool(self.experiment_steps)

    def quality_ok(self, quality: str | None) -> bool:
        """``None`` (unknown) passes; otherwise the quality must be one accepted."""
        if quality is None:
            return True
        text = str(quality).strip() if self.quality_strip else str(quality)
        return text.lower() in {q.lower() for q in self.quality_accept}

    def classify(self, series_description: str | None, quality: str | None = None, **fields: Any) -> dict[str, Any] | None:
        """Classify one scan: the first rule that applies wins; ``None`` when none does."""
        scan = {"series_description": series_description or "", "quality": quality, **fields}
        accepted = self.quality_ok(quality)
        for rule in self.rules:
            if not accepted and not rule.ignore_quality:
                continue
            result = rule.apply(scan)
            if result is not None:
                return result
        return None

    def classify_experiment(self, scans: list[Any]) -> list[tuple[Any, str, str, str, dict[str, Any]]]:
        """Classify every scan of one experiment, running the experiment steps.

        Returns ``(scan, scan_id, series_description, quality, classification)``; with
        ``dedupe_sequences``, the first scan of each sequence only.
        """
        preliminary: list[tuple[Any, str, str, str, dict[str, Any]]] = []
        flagged: dict[str, list[tuple[Any, str]]] = {}
        flags = [str(step.get("candidate_flag") or "") for step in self.experiment_steps]
        for scan in scans:
            scan_id = _scan_attr(scan, "id", "label", "name")
            desc = _scan_attr(scan, "series_description", "type", "label")
            quality = _scan_attr(scan, "quality")
            cls = self.classify(desc, quality or None, scan_id=scan_id)
            if cls is None:
                continue
            flag = next((f for f in flags if f and cls.get(f)), None)
            if flag:
                flagged.setdefault(flag, []).append((scan, scan_id))
                continue
            preliminary.append((scan, scan_id, desc, quality, cls))
        results = list(preliminary)
        for step in self.experiment_steps:
            candidates = flagged.get(str(step.get("candidate_flag") or ""), [])
            if not candidates:
                continue
            fn = EXPERIMENT_STEPS.get(str(step.get("type")))
            if fn is None:
                raise ValueError(f"Classifier {self.name!r}: unknown experiment step {step.get('type')!r}")
            sequences = fn(candidates, step)
            base = dict(step.get("result") or {})
            for scan, scan_id in candidates:
                sequence = sequences.get(scan_id)
                if not sequence:
                    continue
                results.append(
                    (
                        scan,
                        scan_id,
                        _scan_attr(scan, "series_description", "type", "label"),
                        _scan_attr(scan, "quality"),
                        {**base, "sequence": sequence},
                    )
                )
        if not self.dedupe_sequences:
            return results
        seen: set[str] = set()
        unique = []
        for item in results:
            sequence = str(item[4].get("sequence") or "").strip()
            if not sequence or sequence in seen:
                continue
            seen.add(sequence)
            unique.append(item)
        return unique


def _scan_attr(scan: Any, *names: str) -> str:
    """The first non-``None`` attribute among *names* on *scan* (an XNAT scan object), stripped;
    ``""`` if none."""
    for name in names:
        if hasattr(scan, name):
            value = getattr(scan, name)
            if value is not None:
                return str(value).strip()
    return ""


@dataclass(frozen=True)
class ProjectRules:
    """One XNAT project: how it is shown, which cohort it feeds, which classifier reads it."""

    project_id: str
    display_name: str
    cohort_id: str
    classifier: str
    default_sequences: tuple[str, ...]
    visit_label: str | None = None


@dataclass
class XnatRules:
    """The whole document, parsed."""

    name: str
    projects: dict[str, ProjectRules]
    classifiers: dict[str, Classifier]
    visit_labels: dict[str, str]
    default_visit_label: str
    sequence_aliases: dict[str, tuple[str, ...]]
    asset_slots: dict[str, str]
    asset_slot_labels: dict[str, str]
    pipeline_resources: dict[str, dict[str, Any]]
    pipeline_filter_slots: tuple[str, ...]
    default_classifier: str = ""
    source: str = ""

    # projects
    def project(self, project_id: str) -> ProjectRules:
        """The project registered as *project_id* (``KeyError`` listing the known ones)."""
        key = str(project_id).strip()
        if key not in self.projects:
            raise KeyError(f"Unknown XNAT project {key!r}; known: {', '.join(sorted(self.projects))}")
        return self.projects[key]

    def classifier_for(self, project_id: str) -> Classifier:
        """The classifier a project's scans go through."""
        name = self.project(project_id).classifier
        try:
            return self.classifiers[name]
        except KeyError:
            raise KeyError(f"Project {project_id!r} names classifier {name!r}, which is not defined") from None

    def default(self) -> Classifier:
        """The classifier :func:`~nvitk.db.xnat.classify_scan` uses when no project is named."""
        return self.classifiers[self.default_classifier]

    def visit_label(self, project_id: str) -> str:
        """Default visit label for sessions of *project_id* (or a local project id)."""
        return self.visit_labels.get(str(project_id).strip(), self.default_visit_label)

    # sequences and slots
    def asset_slot(self, sequence_label: str) -> str:
        """Canonical ``asset_slot`` for a sequence label (slugified when unregistered)."""
        key = str(sequence_label).strip().upper().replace("-", "_")
        if key in self.asset_slots:
            return self.asset_slots[key]
        return re.sub(r"[^0-9a-z]+", "_", key.lower()).strip("_") or "unknown"

    def asset_slot_label(self, asset_slot: str) -> str:
        """Human-readable label for an ``asset_slot``."""
        key = str(asset_slot).strip().lower()
        if not key:
            return "?"
        return self.asset_slot_labels.get(key, key.replace("_", " ").upper())

    def expand_sequence(self, token: str) -> tuple[str, ...]:
        """A requested sequence token expanded through ``sequence_aliases``."""
        return self.sequence_aliases.get(token, (token,))


def _parse(doc: Mapping[str, Any], source: str = "") -> XnatRules:
    classifiers: dict[str, Classifier] = {}
    for name, spec in dict(doc.get("classifiers") or {}).items():
        quality = dict(spec.get("quality") or {})
        experiment = dict(spec.get("experiment") or {})
        rules = tuple(
            ScanRule(
                name=str(rule.get("name") or f"rule {i + 1}"),
                match={str(k): str(v) for k, v in dict(rule.get("match") or {}).items()},
                not_match={str(k): str(v) for k, v in dict(rule.get("not_match") or {}).items()},
                ignore_quality=bool(rule.get("ignore_quality", False)),
                extract={str(k): dict(v) for k, v in dict(rule.get("extract") or {}).items()},
                result=dict(rule.get("result") or {}),
                flags=str(rule.get("flags", "i")),
            )
            for i, rule in enumerate(spec.get("rules") or ())
        )
        classifiers[str(name)] = Classifier(
            name=str(name),
            rules=rules,
            quality_accept=tuple(str(q) for q in quality.get("accept", ("usable",))),
            quality_strip=bool(quality.get("strip", False)),
            experiment_steps=tuple(dict(s) for s in experiment.get("steps") or ()),
            dedupe_sequences=bool(experiment.get("dedupe_sequences", False)),
        )
    visit_labels = {str(k): str(v) for k, v in dict(doc.get("visit_labels") or {}).items()}
    projects = {
        str(pid): ProjectRules(
            project_id=str(pid),
            display_name=str(spec.get("display_name") or pid),
            cohort_id=str(spec.get("cohort_id") or pid),
            classifier=str(spec.get("classifier") or ""),
            default_sequences=tuple(str(s) for s in spec.get("default_sequences") or ()),
            visit_label=(str(spec["visit_label"]) if spec.get("visit_label") is not None else None),
        )
        for pid, spec in dict(doc.get("projects") or {}).items()
    }
    for pid, project in projects.items():
        if project.visit_label is not None:
            visit_labels.setdefault(pid, project.visit_label)
    return XnatRules(
        name=str(doc.get("name") or ""),
        projects=projects,
        classifiers=classifiers,
        visit_labels=visit_labels,
        default_visit_label=str(doc.get("default_visit_label") or ""),
        sequence_aliases={str(k): tuple(str(x) for x in v) for k, v in dict(doc.get("sequence_aliases") or {}).items()},
        asset_slots={str(k).upper(): str(v) for k, v in dict(doc.get("asset_slots") or {}).items()},
        asset_slot_labels={str(k).lower(): str(v) for k, v in dict(doc.get("asset_slot_labels") or {}).items()},
        pipeline_resources={str(k).lower(): dict(v) for k, v in dict(doc.get("pipeline_resources") or {}).items()},
        pipeline_filter_slots=tuple(str(s) for s in doc.get("pipeline_filter_slots") or ()),
        default_classifier=str(doc.get("default_classifier") or next(iter(classifiers), "")),
        source=source,
    )


# Loading
def config_path() -> Path | None:
    """The ``xnat_rules.json`` the configuration search finds, or ``None``."""
    from nvitk.core import config_paths

    return config_paths.config_file(CONFIG_NAME)


def validate(doc: Mapping[str, Any]) -> list[str]:
    """Problems with *doc* against the schema and the cross-references; empty when valid."""
    problems: list[str] = []
    try:
        import jsonschema

        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        for error in jsonschema.Draft202012Validator(schema).iter_errors(doc):
            where = "/".join(str(p) for p in error.absolute_path) or "(root)"
            problems.append(f"{where}: {error.message}")
    except ImportError:  # pragma: no cover - jsonschema is optional
        pass
    if problems:
        return problems
    try:
        rules = _parse(doc)
    except Exception as exc:  # noqa: BLE001
        return [f"unreadable: {exc}"]
    for pid, project in rules.projects.items():
        if project.classifier not in rules.classifiers:
            problems.append(f"projects/{pid}: classifier {project.classifier!r} is not defined")
    for name, classifier in rules.classifiers.items():
        for rule in classifier.rules:
            for field_name, pattern in {**rule.match, **rule.not_match}.items():
                try:
                    re.compile(pattern)
                except re.error as exc:
                    problems.append(f"classifiers/{name}/{rule.name}/{field_name}: bad regex: {exc}")
            for key, spec in rule.extract.items():
                if spec.get("type") not in EXTRACTORS:
                    problems.append(f"classifiers/{name}/{rule.name}/extract/{key}: unknown type {spec.get('type')!r}")
        for step in classifier.experiment_steps:
            if step.get("type") not in EXPERIMENT_STEPS:
                problems.append(f"classifiers/{name}/experiment: unknown step {step.get('type')!r}")
    return problems


@lru_cache(maxsize=None)
def _load(path_text: str, mtime: float) -> XnatRules:
    del mtime  # part of the cache key only: an edited file is read again
    path = Path(path_text)
    doc = json.loads(path.read_text(encoding="utf-8"))
    problems = validate(doc)
    if problems:
        raise ValueError(f"{path} is not a valid XNAT rules file:\n  " + "\n  ".join(problems[:20]))
    return _parse(doc, source=str(path))


def load_xnat_rules(path: str | Path | None = None) -> XnatRules:
    """The XNAT rules in force: *path*, else the configured file, else the packaged PESA default."""
    target = Path(path).expanduser() if path else (config_path() or PESA_DEFAULT_PATH)
    return _load(str(target), target.stat().st_mtime)


def default_document() -> dict[str, Any]:
    """The packaged PESA default as a JSON document (what ``generate-pesa`` writes out)."""
    return json.loads(PESA_DEFAULT_PATH.read_text(encoding="utf-8"))


def classify_with(
    classifier: str,
    series_description: str | None,
    quality: str | None = None,
    *,
    rules: XnatRules | None = None,
    **fields: Any,
) -> dict[str, Any] | None:
    """Classify one scan with the named classifier of the rules in force."""
    rules = rules or load_xnat_rules()
    return rules.classifiers[classifier].classify(series_description, quality, **fields)


def iter_projects(rules: XnatRules | None = None) -> Iterable[ProjectRules]:
    """Every configured project, sorted by id."""
    rules = rules or load_xnat_rules()
    return [rules.projects[k] for k in sorted(rules.projects)]


__all__ = [
    "CONFIG_NAME",
    "Classifier",
    "ENV_VAR",
    "EXPERIMENT_STEPS",
    "EXTRACTORS",
    "PESA_DEFAULT_PATH",
    "ProjectRules",
    "ScanRule",
    "XnatRules",
    "classify_with",
    "config_path",
    "default_document",
    "iter_projects",
    "load_xnat_rules",
    "validate",
]
