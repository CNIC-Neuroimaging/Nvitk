# Cellpose3 Cyto3

## Overview

Cellpose3 cyto3 is a general-purpose cell segmentation model for microscopy images. It's trained on diverse cell types and provides robust segmentation for 2D and 3D images.

## Model Details

- **Type:** Segmentation
- **Framework:** PyTorch (Cellpose3)
- **Input:** 2D/3D microscopy images (grayscale or RGB)
- **Output:** Integer segmentation masks (0=background, 1,2,3...=individual cells)
- **Source:** Cellpose official release

## Usage

### Loading the Model

```python
from cellpose import models

# Load cyto3 model
model = models.Cellpose(gpu=True, model_type='cyto3')
```

**Important:** Only working for Cellpose 3.x versions, Cellpose 4.x has only [SAM model](../../CellposeSAM/README.md) available

### Basic Segmentation

```python
from cellpose import models
import numpy as np

# Initialize model
model = models.Cellpose(gpu=True, model_type='cyto3')

# Load your images (can be list or single array)
imgs = [np.random.rand(512, 512) for _ in range(10)]

# Segment cells
masks, flows, styles, diams = model.eval(
    imgs,
    diameter=30,           # Average cell diameter
    channels=[0,0],        # [cytoplasm, nucleus] channels
    flow_threshold=0.4,
    cellprob_threshold=0.0
)
```

### Advanced Usage

```python
# Custom parameters for specific cell types
masks, flows, styles = model.eval(
    imgs,
    diameter=None,         # Auto-estimate diameter
    channels=[2,3],        # Custom channels for multi-channel images
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    do_3D=False,          # Set True for 3D segmentation
    anisotropy=None,      # For anisotropic 3D data
    net_avg=False,        # Average 4 networks for better results
    augment=False,        # Test-time augmentation
    resample=True,        # Resample dynamics
)
```

## Files on Server

**Location:** `/ia_models/imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0/`

- `cyto3` - Model weights (PyTorch format, no .pt extension)

## Compatible Containers

- TBD - Currently testing compatibility with A4Cell and general microscopy containers

## Performance

- **GPU Memory:** Unknown
- **Inference Time:** Unknown
- **Best For:** Unknown

## Source

- **Origin:** Cellpose official repository
- **URL:** [Cellpose documentation](https://cellpose.readthedocs.io/en/v3.1.1.1/models.html)
- **Paper:** [Cellpose 3.0: one-click image restoration for improved cellular segmentation (Pachitariu M, Stringer C. bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.02.10.579780v2)
- **License:** CC-BY-NC 4.0 International license

## Maintainers

- BioImaging Team
- Contact: imarcoss@cnic.es

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Related Models

- `CellposeSAM` - Cellpose combined with Segment Anything Model
- `cellpose3-nuclei` - Specialized for nucleus segmentation (TBA)

