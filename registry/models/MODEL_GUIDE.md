# BioImaging-Models - Complete Usage Guide

**Comprehensive guide for finding, using, and managing AI models**

---

## Table of Contents

1. [Overview](#overview)
2. [Finding Models](#finding-models)
3. [Using Models](#using-models)
4. [Model Types](#model-types)
5. [Python Integration](#python-integration)
6. [Container Compatibility](#container-compatibility)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This guide explains how to discover and use models from the BioImaging-Models registry.

### Key Concepts

**Registry** - `registry/models.json` contains all model metadata  
**Git Tags** - Each version has a permanent Git tag  
**Server Storage** - Actual weights stored at `/ia_models/`  
**Metadata** - Documentation in Git repository

---

## Finding Models

### Method 1: Browse Registry JSON

```bash
cd BioImaging-Models

# List all models
jq '.models | keys' registry/models.json

# Get model categories
jq -r '.models[] | .category' registry/models.json | sort -u

# Get all model types
jq -r '.models[] | .type' registry/models.json | sort -u
```

### Method 2: Browse Git Repository

```bash
# Browse structure
ls -R imaging/

# View model README
cat imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0/README.md

# View metadata
cat imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0/.model-metadata.yml
```

### Method 3: Search by Tag

```bash
# Find models by tag
jq -r '.models[] | select(.versions[].tags[]? | contains("segmentation"))' registry/models.json
```

### Method 4: Use Git Tags

```bash
# List all model versions
git tag | grep cellpose3

# Checkout specific version
git checkout cellpose3-cyto3-v1.0.0

# View metadata at that version
cat imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0/.model-metadata.yml

# Return to latest
git checkout main
```

---

## Using Models

### Get Model Information

```bash
MODEL="cellpose3-cyto3"

# Get latest version
VERSION=$(jq -r ".models[\"$MODEL\"].latest" registry/models.json)
echo "Latest version: $VERSION"

# Get model location
LOCATION=$(jq -r ".models[\"$MODEL\"].versions[\"$VERSION\"].location" registry/models.json)
echo "Location: $LOCATION"

# Get files
jq -r ".models[\"$MODEL\"].versions[\"$VERSION\"].files[]" registry/models.json
```

### Load Model in Python

```python
import json
import torch
from pathlib import Path

# Load registry
with open('registry/models.json') as f:
    registry = json.load(f)

# Get model info
model_name = "cellpose3-cyto3"
version = registry['models'][model_name]['latest']
info = registry['models'][model_name]['versions'][version]

# Construct path to weights
location = Path(info['location'])
weight_file = info['files'][0]
model_path = location / weight_file

# Load model
model = torch.load(model_path)
```

### Use Model with Framework

```python
# Example: Cellpose
from cellpose import models

# Load model (using model type from registry)
model = models.CellposeModel(gpu=True, model_type='cyto3')

# Run inference
masks, flows, styles = model.eval(images, diameter=30)
```

---

## Model Types

### Type 1: Single Weight File

**Example:** Cellpose3-cyto3, VascX models

**Structure:**
```
v1.0.0/
└── cyto3    # Single weight file
```

**Loading:**
```python
info = registry['models']['cellpose3-cyto3']['versions']['1.0.0']
weight_path = Path(info['location']) / info['files'][0]
model = torch.load(weight_path)
```

---

### Type 2: Multiple Files (Ensemble)

**Example:** eICAB (4 ensemble models)

**Structure:**
```
v2.0.0/
├── eICAB_0_236.pt
├── eICAB_1_236.pt
├── eICAB_2_236.pt
└── eICAB_3_236.pt
```

**Loading:**
```python
info = registry['models']['eicab']['versions']['2.0.0']
location = Path(info['location'])

# Load all ensemble models
models = []
for file in info['files']:
    model_path = location / file
    models.append(torch.load(model_path))

print(f"Loaded {len(models)} ensemble models")
```

---

### Type 3: Complex nnU-Net Structure

**Example:** TotalSegmentator (30+ datasets)

**Structure:**
```
v2.0.0/
├── config.json
└── nnunet/results/
    ├── Dataset291_TotalSegmentator_part1_organs_1559subj/
    │   └── nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/
    │       ├── fold_0/checkpoint_final.pth
    │       ├── dataset.json
    │       └── plans.json
    ├── Dataset292_TotalSegmentator_part2_vertebrae_1532subj/
    └── ...
```

**Loading:**
```python
from totalsegmentator.python_api import totalsegmentator

# TotalSegmentator handles model loading internally
# Just point to the base directory
info = registry['models']['totalsegmentator']['versions']['2.0.0']
base_dir = info['location']

# Use via API
totalsegmentator(
    input_path="ct_scan.nii.gz",
    output_path="segmentation.nii.gz",
    ml=True  # Use multi-level segmentation
)
```

---

### Type 4: Multiple Variants

**Example:** RETFound (8 disease-specific checkpoints)

**Structure:**
```
v1.0.0/
├── APTOS2019_checkpoint-best.pth
├── Glaucoma_fundus_checkpoint-best.pth
├── IDRID_checkpoint-best.pth
├── JSIEC_checkpoint-best.pth
├── MESSIDOR2_checkpoint-best.pth
├── OCTID_checkpoint-best.pth
├── PAPILA_checkpoint-best.pth
└── Retina_checkpoint-best.pth
```

**Loading specific variant:**
```python
info = registry['models']['retfound']['versions']['1.0.0']
location = Path(info['location'])

# Load specific disease checkpoint
disease = "APTOS2019"
checkpoint_file = f"{disease}_checkpoint-best.pth"
checkpoint_path = location / checkpoint_file

model = torch.load(checkpoint_path)
```

---

## Python Integration

### Helper Class for Model Loading

```python
import json
from pathlib import Path
from typing import Optional, List

class BioImagingModelRegistry:
    """Helper class for loading models from the registry."""
    
    def __init__(self, registry_path: str = "registry/models.json"):
        with open(registry_path) as f:
            self.registry = json.load(f)
    
    def list_models(self) -> List[str]:
        """List all available models."""
        return list(self.registry['models'].keys())
    
    def get_model_info(self, model_name: str, version: Optional[str] = None):
        """Get model information.
        
        Args:
            model_name: Name of the model
            version: Version (default: latest)
        
        Returns:
            Dictionary with model information
        """
        if model_name not in self.registry['models']:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.registry['models'][model_name]
        
        if version is None:
            version = model['latest']
        
        if version not in model['versions']:
            raise ValueError(f"Version {version} not found for {model_name}")
        
        return model['versions'][version]
    
    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """Get path to model weights on server.
        
        Args:
            model_name: Name of the model
            version: Version (default: latest)
        
        Returns:
            Path to model directory
        """
        info = self.get_model_info(model_name, version)
        return Path(info['location'])
    
    def get_model_files(self, model_name: str, version: Optional[str] = None) -> List[Path]:
        """Get paths to all model weight files.
        
        Args:
            model_name: Name of the model
            version: Version (default: latest)
        
        Returns:
            List of paths to weight files
        """
        info = self.get_model_info(model_name, version)
        location = Path(info['location'])
        return [location / f for f in info['files']]


# Usage
registry = BioImagingModelRegistry()

# List all models
print("Available models:")
for model in registry.list_models():
    print(f"  - {model}")

# Get model path
model_path = registry.get_model_path("cellpose3-cyto3")
print(f"\nCellpose3 location: {model_path}")

# Get all weight files
files = registry.get_model_files("eicab", version="2.0.0")
print(f"\neICAB v2.0.0 files:")
for f in files:
    print(f"  - {f}")
```

---

## Container Compatibility

### Check Compatible Containers

```bash
MODEL="totalsegmentator"
VERSION="2.0.0"

# Get compatible containers
jq -r ".models[\"$MODEL\"].versions[\"$VERSION\"].compatible_containers[]" registry/models.json
```

### Use Model with Container

```bash
# Example: Using TotalSegmentator with pesa-fat container
singularity exec --nv \
    --bind /ia_models:/models:ro \
    /containers/projects/pesa-fat/pesa-fat_latest.sif \
    python3 -c "
from totalsegmentator.python_api import totalsegmentator
totalsegmentator('input.nii.gz', 'output.nii.gz')
"
```

### Environment Variables

Many models require environment variables for model paths:

```bash
# Example: TotalSegmentator
export TOTALSEG_HOME_DIR=/ia_models/imaging/TotalSegmentator/v2.0.0

# Example: Cellpose
export CELLPOSE_MODEL_DIR=/ia_models/imaging/Microscopy/Cellpose
```

---

## Troubleshooting

### Model Not Found

**Problem:** `FileNotFoundError: Model not found`

**Solution:**
```bash
# Check if model exists in registry
jq '.models | keys' registry/models.json | grep <model-name>

# Check server storage
ls -la /ia_models/imaging/...
```

### Wrong Model Version

**Problem:** Using outdated version

**Solution:**
```python
# Always check latest version
registry = BioImagingModelRegistry()
info = registry.get_model_info("cellpose3-cyto3")  # Gets latest
print(f"Using version: {info['git_tag']}")
```

### Permission Denied

**Problem:** Cannot read model files

**Solution:**
```bash
# Check permissions
ls -la /ia_models/imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0/

# Models should be readable by all users
# Contact admin if permissions are incorrect
```

### GPU Memory Issues

**Problem:** Out of memory when loading model

**Solution:**
```python
# Check model memory requirements
info = registry.get_model_info("model-name")
print(f"GPU memory required: {info.get('performance', {}).get('gpu_memory_mb', 'unknown')} MB")

# Load on CPU if necessary
model = torch.load(model_path, map_location='cpu')
```

### Incompatible Container

**Problem:** Model doesn't work in container

**Solution:**
```bash
# Check compatible containers
jq -r '.models["model-name"].versions["version"].compatible_containers[]' registry/models.json

# Use appropriate container
# Or contact admin to add model to container
```

---

## Advanced Usage

### Comparing Model Versions

```python
registry = BioImagingModelRegistry()

model_name = "eicab"
versions = registry.registry['models'][model_name]['versions']

print(f"Versions of {model_name}:")
for version, info in versions.items():
    print(f"\n  Version {version}:")
    print(f"    Added: {info['added_date']}")
    print(f"    Files: {', '.join(info['files'])}")
    print(f"    Deprecated: {info.get('deprecated', False)}")
```

### Finding Models by Criteria

```python
registry = BioImagingModelRegistry()

# Find all segmentation models
segmentation_models = [
    name for name, data in registry.registry['models'].items()
    if data['type'] == 'segmentation'
]

print("Segmentation models:")
for model in segmentation_models:
    print(f"  - {model}")

# Find models from specific category
microscopy_models = [
    name for name, data in registry.registry['models'].items()
    if 'Microscopy' in data['category']
]

print("\nMicroscopy models:")
for model in microscopy_models:
    print(f"  - {model}")
```

---

## See Also

- [README.md](README.md) - Main documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to add models
- [registry/README.md](registry/README.md) - Registry documentation
- [scripts/README.md](scripts/README.md) - Automation scripts

---

**Last Updated:** October 14, 2025

