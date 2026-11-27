"""Atlases from the sources installed on this machine: list, load, export, name labels.

Sources: FSL (``$FSLDIR/data/atlases`` and the MNI152 templates), FreeSurfer / FastSurfer
(the colour LUT, ``fsaverage`` / ``cvs_avg35`` label volumes), ANTsXNet templates and priors,
nilearn's atlas fetchers, and the atlases a site configures in ``atlas.json``.

.. code-block:: python

    from nvitk.atlases import list_atlases, load_atlas
    for info in list_atlases():
        print(info.id, info.kind, info.space)
    ho = load_atlas("fsl:HarvardOxford-Cortical", variant="1mm")
    ho.image, ho.label_names()[1]        # path, 'Frontal Pole'

Nothing downloads unless asked (``include_downloadable=True`` lists more; ``download=True``
fetches). CLI: ``nvitk-atlas list | info | export``.
"""

from __future__ import annotations

from nvitk.atlases.base import (
    DOWNLOADABLE,
    INSTALLED,
    AtlasData,
    AtlasInfo,
    Label,
    get_atlas,
    list_atlases,
    load_atlas,
)

__all__ = ["DOWNLOADABLE", "INSTALLED", "AtlasData", "AtlasInfo", "Label", "get_atlas", "list_atlases", "load_atlas"]
