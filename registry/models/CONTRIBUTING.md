# Contributing to BioImaging-Models

**Guidelines for adding and updating AI models in the registry**

---

## Table of Contents

1. [Overview](#overview)
2. [Before You Start](#before-you-start)
3. [Adding a New Model](#adding-a-new-model)
4. [Required Files](#required-files)
5. [Metadata Guidelines](#metadata-guidelines)
6. [Complete Workflow](#complete-workflow)
7. [Model Update Process](#model-update-process)
8. [Best Practices](#best-practices)

---

## Overview

This guide explains how to add new models or update existing ones in the BioImaging-Models registry.

### Who Can Contribute

- **Administrators** - Can add/update models
- **Users** - Can request model additions via GitLab issues

### What Contributions Include

- Model metadata files (`.model-metadata.yml`)
- Documentation (`README.md`, `CHANGELOG.md`)
- Registry updates (`models.json`)
- Git tags for versions

### What Contributions Do NOT Include

- Model weight files (stored on server, not in Git)
- Training data (too large)
- Training artifacts (first iteration excludes these)

---

## Before You Start

### Prerequisites

1. **Model weights on server** - Weights must already exist at `/ia_models/`
2. **Python with PyYAML** - For registry update script
   ```bash
   pip install pyyaml
   ```
3. **Git access** - Write access to repository
4. **Model information** - Know the model's purpose, source, and usage

### Server Storage Structure

Ensure your model follows this structure:

```
/ia_models/imaging/Category/ModelName/vX.Y.Z/
├── model_weights.pt       # Actual weights
├── config.json            # Optional config
└── ...
```

---

## Adding a New Model

### Step 1: Create Directory Structure

```bash
cd BioImaging-Models

# Create version directory in Git repository
# Pattern: imaging/<Category>/<ModelFamily>/<ModelName>/v<VERSION>
mkdir -p imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0
cd imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0
```

### Step 2: Create `.model-metadata.yml`

```yaml
name: cellpose3-cyto3
version: "1.0.0"
display_name: "Cellpose3 Cyto3 - Cell Segmentation"

# Model type
type: segmentation  # segmentation|classification|detection|embedding
framework: pytorch  # pytorch|nnunet|tensorflow
domain: microscopy  # microscopy|neuroimaging|oculomics|medical-imaging

description: >
  Brief description of what this model does and its intended use.
  Keep this concise but informative.

# Storage information
weights:
  location: /ia_models/imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0
  files:
    - cyto3  # List all weight files
  config_files:
    - config.json  # List config files if any, or empty array
  size_mb: unknown  # Actual size in MB, or "unknown"

# Compatibility
compatible_containers:
  - unknown  # Or specify: "container-name >= vYYYY.M.D"

# Source information
source:
  type: pretrained  # pretrained|trained-inhouse|external|unknown
  origin: "Official source name"
  url: "https://..."  # If available
  license: "License type"  # If known

# Usage information
usage:
  command_example: |
    from cellpose import models
    model = models.CellposeModel(gpu=True, model_type='cyto3')
    masks, flows, styles = model.eval(imgs, diameter=30)
  
  input_format: "Description of expected input"
  output_format: "Description of output format"

# Maintainers
maintainers:
  - name: Your Name
    email: your.email@cnic.es

# Tags for searchability
tags:
  - cell-segmentation
  - microscopy
  - cellpose

# Optional: Performance metrics (use "unknown" if not benchmarked)
performance:
  metrics: unknown
  gpu_memory_mb: unknown
  inference_time_ms: unknown

# Optional: Training information (use "unknown" for external models)
training:
  dataset: "Dataset description or unknown"
  framework_version: unknown
```

### Step 3: Create `README.md`

```markdown
# Cellpose3 Cyto3 v1.0.0

## Overview

Brief description of what this model does and its main purpose.

## Model Details

- **Type:** Segmentation
- **Framework:** PyTorch
- **Input:** 2D/3D microscopy images (grayscale or RGB)
- **Output:** Integer segmentation masks
- **Source:** Cellpose official

## Usage

### Loading the Model

\```python
from cellpose import models

# Load model
model = models.CellposeModel(gpu=True, model_type='cyto3')

# Run inference
masks, flows, styles = model.eval(images, diameter=30)
\```

### Example

\```python
import numpy as np
from cellpose import models

# Load your images
imgs = [np.random.rand(512, 512) for _ in range(10)]

# Initialize model
model = models.CellposeModel(gpu=True, model_type='cyto3')

# Segment
masks, flows, styles = model.eval(
    imgs,
    diameter=30,
    channels=[0,0],  # grayscale
    flow_threshold=0.4,
    cellprob_threshold=0.0
)
\```

## Files on Server

**Location:** `/ia_models/imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0/`

- `cyto3` - Model weights (PyTorch)

## Compatible Containers

- TBD - Check with BioImaging team

## Source

- **Origin:** Cellpose official
- **URL:** https://cellpose.org/
- **License:** BSD-3-Clause

## Performance

- **GPU Memory:** ~2GB
- **Inference Time:** ~45ms per 512x512 image

## Maintainers

- BioImaging Team (imarcoss@cnic.es)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
```

### Step 4: Create `CHANGELOG.md`

```markdown
# Changelog - Cellpose3 Cyto3

## [v1.0.0] - 2025-10-14

### Added
- Initial version of Cellpose3 cyto3 model
- Pretrained weights from official Cellpose release
- Supports 2D and 3D cell segmentation

### Notes
- Model downloaded from official Cellpose repository
- Tested on microscopy images from A4Cell project
```

### Step 5: Commit Metadata Files

```bash
cd ../../../../../..  # Return to repo root
git add imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0
git commit -m "feat(cellpose3-cyto3): Add v1.0.0 metadata"
```

### Step 6: Update Registry

```bash
./scripts/update_registry.sh imaging/Microscopy/Cellpose/Cellpose3/cyto3 v1.0.0
```

This script will:
- Read your metadata file
- Update `registry/models.json`
- Add git commit information

### Step 7: Commit Registry Update

```bash
git add registry/models.json
git commit -m "feat(registry): Add cellpose3-cyto3 v1.0.0"
```

### Step 8: Create Git Tag

```bash
./scripts/tag_model.sh imaging/Microscopy/Cellpose/Cellpose3/cyto3 v1.0.0
```

This creates an annotated tag: `cellpose3-cyto3-v1.0.0`

### Step 9: Push to Remote

```bash
git push origin main --tags
```

---

## Required Files

Each model version **must** have:

```
imaging/Category/Model/version/
├── .model-metadata.yml    # ✅ Required - Structured metadata
├── README.md              # ✅ Required - Usage documentation
└── CHANGELOG.md           # ✅ Required - Version history
```

### `.model-metadata.yml` - Required Fields

```yaml
name:                      # ✅ Required
version:                   # ✅ Required
display_name:              # ✅ Required
type:                      # ✅ Required
framework:                 # ✅ Required
domain:                    # ✅ Required
description:               # ✅ Required
weights:
  location:                # ✅ Required
  files:                   # ✅ Required
source:
  type:                    # ✅ Required
maintainers:               # ✅ Required
  - name:
    email:
```

### Optional Fields

Use `unknown` if information is not available:

```yaml
size_mb: unknown
compatible_containers: [unknown]
performance:
  metrics: unknown
  gpu_memory_mb: unknown
training:
  dataset: unknown
```

---

## Metadata Guidelines

### Version Numbers

Use **Semantic Versioning (SemVer):**

```
vMAJOR.MINOR.PATCH

v1.0.0 - Initial version
v1.0.1 - Bug fix or minor update
v1.1.0 - New features, backward compatible
v2.0.0 - Breaking changes
```

### Model Names

Use lowercase with hyphens:

```
✅ Good:
  cellpose3-cyto3
  eicab
  vascx-vessels
  totalsegmentator

❌ Bad:
  Cellpose3_Cyto3
  eICAB
  VascX_Vessels
```

### Git Tag Format

```
Format: <model-name>-v<VERSION>

Examples:
  cellpose3-cyto3-v1.0.0
  eicab-v2.0.0
  totalsegmentator-v2.0.0
```

### Unknown Fields Policy

When information is **not available**, explicitly use `unknown`:

```yaml
✅ Good:
  size_mb: unknown
  compatible_containers: [unknown]
  performance:
    metrics: unknown

❌ Bad:
  size_mb: 0
  size_mb: null
  size_mb:    # empty
  # Field omitted entirely
```

**Never:**
- Estimate values
- Use placeholder values (0, null, TBD)
- Leave fields empty
- Omit required fields

---

## Complete Workflow

### Full Example: Adding eICAB v2.0.0

```bash
# 1. Create directory
mkdir -p imaging/Neuroimaging/eICAB/v2.0.0
cd imaging/Neuroimaging/eICAB/v2.0.0

# 2. Create .model-metadata.yml
cat > .model-metadata.yml << 'EOF'
name: eicab
version: "2.0.0"
display_name: "eICAB v2 - Circle of Willis Segmentation"
type: segmentation
framework: pytorch
domain: neuroimaging
description: >
  Ensemble model for Circle of Willis segmentation from MRA images.
  Uses 4 models for improved accuracy.
weights:
  location: /ia_models/imaging/Neuroimaging/eICAB/v2.0.0
  files:
    - eICAB_0_236.pt
    - eICAB_1_236.pt
    - eICAB_2_236.pt
    - eICAB_3_236.pt
  size_mb: unknown
compatible_containers: ["eICAB:v2023.4.17"]
source:
  type: external
  origin: "eICAB project"
  url: "https://gitlab.com/FelixDumais/vessel_segmentation_snaillab"
  license: unknown
usage:
  command_example: |
    # Usage via eICAB container
    singularity exec eICAB.sif eicab_cmd --input mra.nii.gz
  input_format: "MR angiography (MRA) images"
  output_format: "Circle of Willis segmentation mask"
maintainers:
  - name: BioImaging Team
    email: djimenez@cnic.es
tags:
  - vascular-segmentation
  - circle-of-willis
  - neuroimaging
performance:
  metrics: unknown
  gpu_memory_mb: unknown
training:
  dataset: unknown
EOF

# 3. Create README.md
cat > README.md << 'EOF'
# eICAB v2.0.0

## Overview
Express Intracranial Arteries Breakdown for automated segmentation
of Circle of Willis from MR angiography.

## Model Details
- **Type:** Segmentation
- **Framework:** PyTorch ensemble (4 models)
- **Input:** MR angiography (MRA) images
- **Output:** Circle of Willis segmentation mask
- **Source:** External (eICAB project)

## Usage
This model is used via the eICAB container.

## Files on Server
- eICAB_0_236.pt
- eICAB_1_236.pt
- eICAB_2_236.pt
- eICAB_3_236.pt

## Compatible Containers
- eICAB:v2023.4.17

## Source
- GitLab: https://gitlab.com/FelixDumais/vessel_segmentation_snaillab

## Maintainers
- BioImaging Team (djimenez@cnic.es)
EOF

# 4. Create CHANGELOG.md
cat > CHANGELOG.md << 'EOF'
# Changelog - eICAB

## [v2.0.0] - 2025-10-14
### Added
- Improved ensemble with 236-layer models
- Better accuracy on Circle of Willis segmentation

### Changed
- Updated model architecture from v1.0.0
- New weight files with "_236" suffix

## [v1.0.0] - 2023-04-17
### Added
- Initial version with 4-model ensemble
EOF

# 5. Return to repo root
cd ../../../..

# 6. Commit metadata
git add imaging/Neuroimaging/eICAB/v2.0.0
git commit -m "feat(eicab): Add v2.0.0 metadata"

# 7. Update registry
./scripts/update_registry.sh imaging/Neuroimaging/eICAB v2.0.0

# 8. Commit registry
git add registry/models.json
git commit -m "feat(registry): Add eicab v2.0.0"

# 9. Create tag
./scripts/tag_model.sh imaging/Neuroimaging/eICAB v2.0.0

# 10. Push
git push origin main --tags
```

---

## Model Update Process

### Adding a New Version

When adding a new version of an existing model:

1. Create new version directory
2. Create metadata files for new version
3. Update CHANGELOG.md with changes
4. Follow same workflow as new model
5. Old versions remain in registry (not deleted)

### Deprecating a Version

To mark a version as deprecated:

1. Edit the version in `registry/models.json`:
   ```json
   "1.0.0": {
     "deprecated": true,
     "deprecation_reason": "Use v2.0.0 with improved accuracy"
   }
   ```

2. Commit the change:
   ```bash
   git add registry/models.json
   git commit -m "feat(eicab): Deprecate v1.0.0"
   ```

---

## Best Practices

### 1. Test Before Adding

- Verify model works on server
- Test with compatible containers
- Confirm file paths are correct

### 2. Complete Documentation

- Provide usage examples
- Document input/output formats
- Include citations if applicable

### 3. Consistent Naming

- Use lowercase with hyphens
- Keep names descriptive
- Follow existing conventions

### 4. Version Carefully

- Use SemVer for models
- Document breaking changes
- Maintain changelog

### 5. Mark Unknowns

- Use `unknown` for missing info
- Never guess or estimate
- Update when information becomes available

### 6. Coordinate with Team

- Announce new models
- Update container dependencies
- Document compatibility

---

## Special Cases

### Ensemble Models (Multiple Files)

```yaml
weights:
  files:
    - model_0.pt
    - model_1.pt
    - model_2.pt
    - model_3.pt
```

### nnU-Net Models

```yaml
weights:
  structure: nnunet_results
  files:
    - "nnunet/results/Dataset*/*/fold_0/checkpoint_final.pth"
  config_files:
    - config.json
    - nnunet/results/Dataset*/dataset.json
```

### Models with Variants

```yaml
# Option A: Single model with multiple files
weights:
  files:
    - variant_a.pth
    - variant_b.pth
    - variant_c.pth

# Option B: Document variants in README
# List each variant and its purpose
```

---

## Getting Help

**Questions or issues?**
- Check existing model metadata for examples
- Review [MODEL_GUIDE.md](MODEL_GUIDE.md)
- Contact: imarcoss@cnic.es

**Request a model addition:**
- Open GitLab issue
- Provide model information
- Admin will add to registry

---

**Last Updated:** October 14, 2025
