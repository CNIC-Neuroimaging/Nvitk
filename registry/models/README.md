# BioImaging-Models

**Git-based Model Registry for Centralized AI Model Management**

---

**Version:** 1.0.0  
**Date:** October 14, 2025  
**Team:** BioImaging - CNIC BioIT Unit  
**Status:** Development

---

## 📋 Overview

BioImaging-Models is a Git-based registry system for managing AI model weights and metadata. This repository serves as the central catalog for all models used in the BioImaging infrastructure.

### What This Repository Contains

```
BioImaging-Models/
├── registry/              # Central registry system
│   ├── models.json        # All models and versions
│   └── schema.json        # JSON schema
│
├── scripts/               # Automation tools
│   ├── update_registry.sh # Update registry
│   └── tag_model.sh       # Create git tags
│
└── imaging/               # Model metadata by category
    ├── Microscopy/        # Cellpose, etc.
    ├── Neuroimaging/      # eICAB, TopCoW
    ├── Oculomics/         # RETFound, VascX
    └── TotalSegmentator/  # Whole-body segmentation
```

### What's NOT in This Repository

- **Model weights** - Stored on server at `/ia_models/` (too large for Git)
- **Training data** - External or project-specific
- **Training artifacts** - Not included in first iteration

---

## 🚀 Quick Start

### Finding a Model

```bash
# Clone the repository
git clone <repo-url> BioImaging-Models
cd BioImaging-Models

# List all models
jq '.models | keys' registry/models.json

# Get model location
jq -r '.models["cellpose3-cyto3"].versions["1.0.0"].location' registry/models.json
# Output: /ia_models/imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0
```

### Using a Model

```python
import json
from pathlib import Path

# Load registry
with open('registry/models.json') as f:
    registry = json.load(f)

# Get model info
model = "cellpose3-cyto3"
version = registry['models'][model]['latest']
info = registry['models'][model]['versions'][version]

# Get model path on server
model_path = Path(info['location']) / info['files'][0]
print(f"Model location: {model_path}")
# /ia_models/imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0/cyto3

# Load model
import torch
model_weights = torch.load(model_path)
```

---

## 📊 Available Models

### Microscopy (2 models)
- **Cellpose3-cyto3** - General cell segmentation
- **CellposeSAM** - Cell segmentation with SAM

### Neuroimaging (4 models)
- **eICAB** v1.0.0, v2.0.0 - Circle of Willis segmentation
- **TopCoW-ARG** - Circle of Willis (ARG method)
- **TopCoW-CLAIM** - Circle of Willis (CLAIM method)

### Oculomics (6 models)
- **RETFound** - Retinal disease classification (8 variants)
- **VascX** - Retinal vessel analysis
  - artery_vein, disc, fovea, quality, vessels

### Whole-Body Segmentation (1 model)
- **TotalSegmentator v2** - 104 anatomical structures (30+ datasets)

**Total:** ~13 distinct models, ~25 model variants/versions

---

## 📖 Documentation

- **[MODEL_GUIDE.md](MODEL_GUIDE.md)** - Complete usage guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to add models
- **[registry/README.md](registry/README.md)** - Registry documentation
- **[scripts/README.md](scripts/README.md)** - Automation scripts

---

## 🔖 Version Control System

### Git Tags

Each model version has a Git tag that links metadata to server files:

```
Git Tag: cellpose3-cyto3-v1.0.0
    │
    ├─→ Points to commit with metadata
    │
    └─→ Metadata references: /ia_models/.../cyto3/v1.0.0/
```

**Tag format:** `<model-name>-v<VERSION>`

**Examples:**
- `cellpose3-cyto3-v1.0.0`
- `eicab-v2.0.0`
- `totalsegmentator-v2.0.0`

### No Checksums

Unlike containers, models do **not** use checksums because:
- Model files are very large (100MB - 15GB)
- SHA256 calculation is impractical (10-30+ minutes per model)
- Git tags provide version traceability
- Server storage is read-only and admin-managed

---

## 🗂️ Model Metadata

Each model version includes:

```
imaging/Category/Model/version/
├── .model-metadata.yml    # Structured metadata
├── README.md              # Usage documentation
└── CHANGELOG.md           # Version history
```

### Metadata Example

```yaml
name: cellpose3-cyto3
version: "1.0.0"
display_name: "Cellpose3 Cyto3"
type: segmentation
framework: pytorch
domain: microscopy

weights:
  location: /ia_models/imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0
  files: [cyto3]
  size_mb: unknown

compatible_containers: [unknown]

source:
  type: pretrained
  origin: "Cellpose official"
  license: "BSD-3-Clause"

maintainers:
  - name: BioImaging Team
    email: imarcoss@cnic.es

tags:
  - cell-segmentation
  - microscopy
```

---

## 🔍 Finding Models

### By Name

```bash
jq -r '.models["cellpose3-cyto3"]' registry/models.json
```

### By Category

```bash
jq -r '.models[] | select(.category | startswith("imaging/Microscopy"))' registry/models.json
```

### By Type

```bash
jq -r '.models[] | select(.type == "segmentation")' registry/models.json
```

### List All Versions

```bash
jq -r '.models["eicab"].versions | keys' registry/models.json
```

---

## 🛠️ For Administrators

### Adding a New Model

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

**Quick workflow:**

```bash
# 1. Create metadata files
mkdir -p imaging/Microscopy/NewModel/v1.0.0
cd imaging/Microscopy/NewModel/v1.0.0
# Create .model-metadata.yml, README.md, CHANGELOG.md

# 2. Commit metadata
git add .
git commit -m "feat(newmodel): Add v1.0.0 metadata"

# 3. Update registry
cd ../../../../
./scripts/update_registry.sh imaging/Microscopy/NewModel v1.0.0

# 4. Create git tag
./scripts/tag_model.sh imaging/Microscopy/NewModel v1.0.0

# 5. Push
git push origin main --tags
```

---

## 📞 Support

**Maintainers:**
- BioImaging Team
- Email: imarcoss@cnic.es

**Related Infrastructure:**
- **BioImaging-Containers** - Container registry
- Server storage: `/ia_models/` (read-only)

---

## 🎯 Design Principles

### 1. Git as Single Source of Truth
- All metadata in Git
- Version controlled
- Distributed copies

### 2. Simplicity Over Completeness
- Focus on essential metadata
- Mark unknowns explicitly
- No unnecessary fields

### 3. Support Diverse Model Types
- Single files (Cellpose)
- Ensembles (eICAB)
- Complex structures (nnU-Net)
- Different domains

### 4. Unknown Fields Policy
When information is not available, use `unknown`:

```yaml
size_mb: unknown
compatible_containers: [unknown]
performance:
  metrics: unknown
```

**Never:** Estimate, guess, or leave empty

---

## 📄 License

Internal CNIC BioIT infrastructure. Check individual model licenses in their metadata.

---

## 🔗 Links

- [BioImaging-Containers](../BioImaging-Containers) - Container registry
- [Model Guide](MODEL_GUIDE.md) - Complete usage guide
- [Contributing Guide](CONTRIBUTING.md) - How to add models
- Registry: `registry/models.json`

---

**Last Updated:** October 14, 2025  
**Repository Version:** 1.0.0
