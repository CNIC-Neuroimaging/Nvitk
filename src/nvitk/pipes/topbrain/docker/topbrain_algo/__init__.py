"""The only first-party code that ships inside the submission container.

Deliberately not ``nvitk``. The image needs three things from the research pipeline — the
intensity harmonisation, connected-component clean-up, and a host-memory cast — and copying the
whole library in to get them dragged 151 MB of unrelated pipelines into an image with a 10 GiB
ceiling, plus every import they pull.

Everything here is numpy/scipy only and duplicates behaviour that is tested in nvitk. It is a
deliberate, small duplication: the alternative is either a slimmer nvitk (a much larger change,
and not one a submission deadline should force) or an image that carries code it never runs.

**Kept in step by hand.** If stage 0's harmonisation changes, :mod:`topbrain_algo.harmonize`
must change with it — the windows themselves travel in ``models.json``, so only the *shape* of
the transform is duplicated here, which is the part that almost never moves.
"""

from __future__ import annotations

__all__: list[str] = []
