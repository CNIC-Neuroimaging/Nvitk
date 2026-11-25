"""Atlas label tables as GUI label vocabularies (:class:`nvitk.gui.labels.catalog.LabelSchema`)."""

from __future__ import annotations

from typing import Any

from nvitk.atlases.base import AtlasData


def schema_key(data: AtlasData) -> str:
    """Catalog key for an atlas's labels (``atlas:<source>:<name>[:<variant>]``)."""
    return f"atlas:{data.info.id}" + (f":{data.variant}" if data.variant else "")


def to_label_schema(data: AtlasData) -> Any:
    """A ``LabelSchema`` naming the atlas's label ids (group ``"Atlases"``)."""
    from nvitk.gui.labels.catalog import LabelSchema

    return LabelSchema(
        key=schema_key(data),
        title=data.info.name + (f" ({data.variant})" if data.variant else ""),
        group="Atlases",
        id_to_name={label.id: label.name for label in data.labels},
        description=data.info.description,
    )


def register_atlas_schema(data: AtlasData) -> str:
    """Register the atlas's labels in the GUI catalog; returns the schema key."""
    from nvitk.gui.labels.catalog import register_schema

    return register_schema(to_label_schema(data)).key


def label_colors(data: AtlasData) -> dict[int, tuple[float, float, float, float]]:
    """``{id: (r, g, b, a)}`` in 0-1, for a Napari Labels colormap (background transparent)."""
    return {
        label.id: (label.rgb[0] / 255.0, label.rgb[1] / 255.0, label.rgb[2] / 255.0, 0.0 if label.id == 0 else 1.0)
        for label in data.labels
    }


__all__ = ["label_colors", "register_atlas_schema", "schema_key", "to_label_schema"]
