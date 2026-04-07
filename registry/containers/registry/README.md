# Container Registry

This directory contains the central registry for all BioImaging containers.

## Files

- **`containers.json`** - Main registry file (machine-readable)
- **`schema.json`** - JSON schema defining the registry structure
- **`README.md`** - This file

## Usage

### Finding a Container

```bash
# List all containers
jq '.containers' registry/containers.json

# Get latest version of a container
jq -r '.containers.projects["pesa-fat"].latest' registry/containers.json

# Get container path
jq -r '.containers.projects["pesa-fat"].versions["v2025.5.27"].sif_path' registry/containers.json
```

### Getting Container Information

```bash
# Get all metadata for a container version
jq '.containers.projects["pesa-fat"].versions["v2025.5.27"]' registry/containers.json

# Get dependencies
jq '.containers.projects["pesa-fat"].versions["v2025.5.27"].dependencies' registry/containers.json
```

### Using in Scripts

```python
import json
from pathlib import Path

# Load registry
with open('registry/containers.json') as f:
    registry = json.load(f)

# Get latest container
project = "pesa-fat"
latest_version = registry['containers']['projects'][project]['latest']
container_info = registry['containers']['projects'][project]['versions'][latest_version]
container_path = container_info['sif_path']

print(f"Using {project}:{latest_version}")
print(f"Path: {container_path}")
```

## Registry Format

See `schema.json` for the complete schema definition.

### Key Fields

- **`latest`** - Points to the recommended version (no symlinks needed)
- **`sif_path`** - Absolute path to container on server
- **`sif_sha256`** - Checksum for verification
- **`external`** - Flag for externally-built containers (no .def available)
- **`dependencies`** - Links to base containers and required models
- **`git_tag`** / **`git_commit`** - Version control linkage

## Updating the Registry

Use the provided scripts:

```bash
# Add new container version
./scripts/update_registry.sh <project> <version>

# Verify checksums
./scripts/verify_container.sh <project> <version>
```

## External Containers

Containers marked with `"external": true` are pre-built containers without corresponding `.def` files in this repository. These include:

- **eICAB** - Intracranial Arteries segmentation
- **TopCoW-ARG** - Circle of Willis segmentation (ARG Team)
- **TopCoW-CLAIM** - Circle of Willis segmentation (CLAIM Team)

For external containers, only metadata and usage documentation are maintained.

