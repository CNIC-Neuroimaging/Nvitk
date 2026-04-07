# eICAB - Express Intracranial Arteries Breakdown

**External Container** - Pre-built and Fixed container from external source 

## 📋 Overview

eICAB (Express Intracranial Arteries Breakdown) is a tool for automated segmentation and labeling of the main intracranial arteries on MR angiography images.
Further details can be found in the following publication:

[eICAB: A novel deep learning pipeline for Circle of Willis multiclass segmentation and analysis](https://www.sciencedirect.com/science/article/pii/S1053811922005420?via%3Dihub#sec0016)

**Type:** Project Container (External)  
**Source:** Pre-built (no .def file)  
**GPU Required:** No

## ⚠️ External Container Notice

This container was built externally and is provided as-is. The BioImaging-Containers repository **does not** contain the source definition (`.def`) file for this container.

**Available for:**
- Usage and deployment
- Documentation and metadata

**Not available:**
- Source code/definition
- Direct Rebuild & Modification capability

## 🔧 Available Versions

| Version | Date | Path |
|---------|------|------|
| v2023.4.17 | 2023-04-17 | `.../neuroimaging/eICAB/eICAB_v2023.4.17.sif` |
| v2022.10.15 | 2022-10-15 | `.../neuroimaging/eICAB/eICAB_v2022.10.15.sif` |

**Latest:** v2023.4.17

## 🚀 Usage

### Finding the Container

```bash
# Get latest version
jq -r '.containers.projects["eICAB"].latest' registry/containers.json

# Get container path
jq -r '.containers.projects["eICAB"].versions["v2023.4.17"].sif_path' registry/containers.json
```

### Running the Container

TBA

### Required Bind Mounts

TBA

## 📝 Capabilities

- ✅ Circle of Willis segmentation from Raw MRA volumes
    eICAB labels

eICAB automatically segments and labels the Internal Carotid Arteries (ICA), Basilar Artery (BA), Anterior Communicating
Artery (AComm), Anterior Cerebral Arteries (ACA), Middle Cerebral Arteries (MCA), Posterior Communicating Arteries (PComm), Posterior Cerebral Arteries (PCA), Superior Cerebellar Arteries (SCA) and Anterior Choroidal Arteries (AChA).

Labels 15 to 18 are still experimentals, but were annotated and trained using the same scheme as label 1 to 14 (see
article).

| Arteries | Left | Right |
|----------|------|:-----:|
| ICA      | 1    |   2   |
| ACA-A1   | 5    |   6   |
| MCA-M1   | 7    |   8   |
| PComm    | 9    |  10   |
| PCA-P1   | 11   |  12   |
| PCA-P2   | 13   |  14   |
| SCA      | 15   |  16   |
| AChA     | 17   |  18   |

BAS = 3, AComm = 4

## 📊 Container Information

- **Maintainer:** BioImaging Team (imarcoss@cnic.es)
- **Source:** External (pre-built)
- **GPU Required:** No
- **Type:** Neuroimaging analysis

## 📚 Related Documentation

- [Container Registry](../../../registry/README.md)
- [Container Guide](../../../CONTAINER_GUIDE.md)
- [External eICAB documentation](https://gitlab.com/FelixDumais/vessel_segmentation_snaillab)

## 📞 Support

For issues or questions about this container:
- Contact: imarcoss@cnic.es
- GitLab: Open an issue in BioImaging-Containers repository

For eICAB-specific questions, refer to original eICAB documentation.

