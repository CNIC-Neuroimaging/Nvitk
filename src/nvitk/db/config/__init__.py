"""Configurable dataset and XNAT setup.

* :mod:`.dataset_config`: ``dataset.json``, how a dataset is assembled from spreadsheets
  (identifier columns, visits, sources and their measurement-pipeline versions) and how the
  Data browser walks pipeline folders.
* :mod:`.xnat_rules`: ``xnat_rules.json``, XNAT projects, the rules that classify a scan's
  series description into a sequence, asset slots and pipeline resources.

Both fall back to packaged defaults (``defaults/pesa_*.json``) that reproduce nvitk's original
PESA-Brain / PESA-Fat behaviour, so an install without either file behaves as before.
``nvitk-config generate-pesa`` writes those defaults out for editing; ``nvitk-config init``
seeds generic starters.
"""

from __future__ import annotations

from nvitk.db.config.dataset_config import DatasetConfig, load_dataset_config
from nvitk.db.config.xnat_rules import Classifier, ScanRule, XnatRules, load_xnat_rules

__all__ = ["Classifier", "DatasetConfig", "ScanRule", "XnatRules", "load_dataset_config", "load_xnat_rules"]
