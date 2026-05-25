"""DICOM tags panel helpers."""

from nvitk.gui.dicom_tags_panel import dicom_tags_from_metadata, layer_has_dicom_tags


class _FakeLayer:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata
        self.name = "test"


def test_dicom_tags_from_metadata():
    md = {
        "source": "/tmp/study",
        "(0010,0020)": "P001",
        "PatientID": "P001",
        "affine": [[1, 0, 0, 0]],
    }
    tags = dicom_tags_from_metadata(md)
    assert "(0010,0020)" in tags
    assert "PatientID" in tags
    assert "affine" not in tags


def test_layer_has_dicom_tags_by_source_type():
    layer = _FakeLayer({"nvitk_metadata": {"source_type": "dicom"}})
    assert layer_has_dicom_tags(layer) is True
