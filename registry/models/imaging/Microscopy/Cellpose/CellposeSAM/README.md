# CellposeSAM

**Cell Segmentation with Segment Anything Model Integration**

## Overview

CellposeSAM combines the specialized cell segmentation capabilities of Cellpose with Meta's Segment Anything Model (SAM) to provide enhanced segmentation performance and generalization.

## Model Details

- **Type:** Segmentation
- **Framework:** PyTorch (Cellpose + SAM)
- **Input:** 2D/3D microscopy images (grayscale or RGB)
- **Output:** Integer segmentation masks
- **Source:** Cellpose-SAM official

## Key Features

- **Foundation model integration:** Leverages SAM's powerful features
- **Improved generalization:** Better performance on diverse cell types
- **Cellpose compatibility:** Works with standard Cellpose API version 4.x

## Usage

### Basic Usage

```python
from cellpose import models

# Load CellposeSAM model
model = models.CellposeModel(gpu=True)

# Segment cells
masks, flows, styles = model.eval(images)
```

**Important:** This model only works for Cellpose 4.x versions. The rest of Cellpose models working under 3.x versions are depreciated. 

### Advanced Usage

```python
from cellpose import models
import numpy as np

# Initialize model
model = models.CellposeModel(gpu=True)

# Load images
imgs = [np.random.rand(512, 512) for _ in range(10)]

# Segment with custom parameters
masks, flows, styles = model.eval(
    imgs,
    diameter=None,         # Auto-estimate
    channels=[0,0],        # Grayscale
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    do_3D=False
)
```

## Files on Server

**Location:** `/ia_models/imaging/Microscopy/Cellpose/CellposeSAM/v1.0.0/`

- `cpsam` - Model weights (PyTorch format)

## Comparison with Cellpose3

### CellposeSAM Advantages
- Better generalization to new cell types
- Improved boundary detection
- Foundation model features
- Enhanced performance on challenging images

### Cellpose3 Advantages
- Smaller model size
- Faster inference
- Lower memory requirements

## Compatible Containers

- TBA - Testing with A4Cell and general microscopy containers

## Performance

- **GPU Memory:** TBA
- **Inference Time:** TBA
- **Best For:** TBA

## Source

- **Origin:** Cellpose-SAM project
- **URL:** [Cellpose4 documentation](https://cellpose.readthedocs.io/en/latest/)
- **License:** CC-BY-NC 4.0 International license

## Citation

When using this model, cite both Cellpose and SAM:

- [Cellpose-SAM: superhuman generalization for cellular segmentation. bioRxiv. Pachitariu, M., Rariden, M., & Stringer, C. (2025).](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1)

## Maintainers

- BioImaging Team
- Contact: imarcoss@cnic.es

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

