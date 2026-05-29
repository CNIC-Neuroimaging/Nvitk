"""Napari GUI for nvitk (``nvitk-gui`` console script).

Package layout:

- ``core/`` — backend, spatial metadata, orientation, logging, warnings
- ``io/`` — Napari open/export integration
- ``labels/`` — label catalog, selector, visibility
- ``tools/`` — registry, panels, runner, presets
- ``pipeline/`` — pipeline catalog, stages, form UI
- ``panels/`` — data browser, DICOM tags, XNAT
- ``sge/`` — cluster job submit/retrieve/worker
- ``viz/`` — vectors, centerline, loc points
"""

__version__ = "0.1.0"
