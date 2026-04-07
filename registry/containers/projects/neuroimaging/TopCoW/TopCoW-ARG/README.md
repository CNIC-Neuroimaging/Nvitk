# TopCoW-ARG - Circle of Willis Segmentation

**External Container** - Pre-built container from TopCoW challenge

## 📋 Overview

TopCoW (Topology-preserving anatomical segmentation of the Circle of Willis) team ARG variant for automated segmentation of the Circle of Willis from MR angiography.

**ARG** Team for MRA modality

**Type:** Project Container (External)  
**Source:** Pre-built (no .def file)  
**GPU Required:** Yes  
**Minimum GPU Memory:** Unknown

## ⚠️ External Container Notice

This container was built externally as part of the [TopCoW challenge](https://topcow24.grand-challenge.org/data/). The BioImaging-Containers repository **does not** contain the source definition (`.def`) file.

## 🔧 Available Versions

| Version | Date | Path |
|---------|------|------|
| v2025.10.23 | 2025-10-23 | `.../neuroimaging/TopCoW/TopCoW-ARG/TowCoW-ARG_v2025.10.23.sif` |

**Latest:** v2025.10.23

## 🚀 Usage

### Finding the Container

```bash
# Get latest version
jq -r '.containers.projects["TopCoW-ARG"].latest' registry/containers.json

# Get container path
jq -r '.containers.projects["TopCoW-ARG"].versions["v2025.10.23"].sif_path' registry/containers.json
```

### Running the Container

TBA

### Required Bind Mounts

TBA

## 📝 Capabilities

- ✅ Circle of Willis segmentation from MRA

## 🔬 TopCoW Team Variants

The TopCoW project has different team variants:
- **TopCoW-ARG** (this container): Modality MRA
- **TopCoW-CLAIM**: Modality MRA and CTA

Choose the appropriate variant based on your analysis needs.

## 📊 Container Information

- **Maintainer:** BioImaging Team (imarcoss@cnic.es)
- **Source:** External (TopCoW challenge)
- **GPU Required:** Yes 
- **Type:** Vascular neuroimaging

## 📚 Related Documentation

- [Container Registry](../../../../registry/README.md)
- [Container Guide](../../../../CONTAINER_GUIDE.md)
- [TopCoW challenge documentation](https://topcow24.grand-challenge.org/data/)

## 📞 Support

For issues or questions:
- Container deployment: imarcoss@cnic.es
- TopCoW methodology: Refer to TopCoW challenge documentation

