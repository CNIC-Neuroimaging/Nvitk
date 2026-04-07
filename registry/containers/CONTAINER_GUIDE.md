# Container Guide

Complete guide for using and contributing to the BioImaging container infrastructure.

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Finding Containers](#finding-containers)
3. [Using Containers](#using-containers)
4. [Building New Versions](#building-new-versions)
5. [Contributing New Containers](#contributing-new-containers)
6. [External Containers](#external-containers)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Infrastructure Components

```
BioImaging Container Infrastructure
├── Git Repository (this repo)
│   ├── Container definitions (.def.template files)
│   ├── Metadata (.container-metadata.yml)
│   ├── Build scripts
│   └── Central registry (containers.json)
│
└── Server Storage
    └── /containers/
        ├── base/          → Base containers
        └── projects/      → Project containers
```

### Key Concepts

- **Templates:** Parameterized container definitions (`.def.template`)
- **Registry:** Central JSON file tracking all containers and versions
- **Versioning:** Calendar versioning (CalVer) for containers: `vYYYY.M.D`
- **Metadata:** YAML files describing container capabilities and requirements

---

## Finding Containers

### Using the Registry

All containers are catalogued in `registry/containers.json`:

```bash
# List all containers
jq '.containers' registry/containers.json

# List base containers
jq '.containers.base | keys' registry/containers.json

# List project containers
jq '.containers.projects | keys' registry/containers.json

# Get latest version of a container
jq -r '.containers.projects["pesa-fat"].latest' registry/containers.json

# Get container path
jq -r '.containers.projects["pesa-fat"].versions["v2025.5.27"].sif_path' registry/containers.json

# Get all versions of a container
jq '.containers.projects["eICAB"].versions | keys' registry/containers.json
```

### Using Python

```python
import json

# Load registry
with open('registry/containers.json') as f:
    registry = json.load(f)

# Get container info
project = "pesa-fat"
latest = registry['containers']['projects'][project]['latest']
info = registry['containers']['projects'][project]['versions'][latest]

print(f"Container: {info['sif_path']}")
print(f"Size: {info['size_mb']} MB")
print(f"SHA256: {info['sif_sha256']}")
```

---

## Using Containers

### Basic Usage

```bash
# Get container path from registry
CONTAINER=$(jq -r '.containers.projects["pesa-fat"].versions["v2025.5.27"].sif_path' \
    registry/containers.json)

# Interactive shell (with GPU)
singularity shell --nv $CONTAINER

# Execute command
singularity exec --nv $CONTAINER python script.py

# Run container's default runscript
singularity run --nv $CONTAINER
```

### With Bind Mounts

```bash
# Bind multiple directories
singularity exec --nv \
    --bind /ia_models:/models \
    --bind /data:/data \
    --bind /results:/results \
    $CONTAINER \
    python pipeline.py
```

### Verifying Integrity

#### Quick Verification (Recommended - Instant)

For daily use, verify using cached checksums:

```bash
./scripts/verify_container.sh projects/pesa-fat v2025.5.27
```

This checks against cached checksums stored in the `.checksums` file (instant, <1 second).

#### Full Verification (Security-Critical - 5-10 minutes)

For complete integrity verification, recalculate SHA256 from the `.sif` file:

```bash
./scripts/verify_container.sh projects/pesa-fat v2025.5.27 --full
```

---

## Building New Versions

### Prerequisites

- singularity/Singularity installed
- Sudo access (or fakeroot capability)
- Write access to container storage location [To be Changed]
- Git repository cloned

### Workflow: Updating an Existing Container

#### 1. Update Requirements

```bash
cd projects/pesa-fat/requirements

# Edit environment.yml
nano environment.yml

# Regenerate requirements files
python ../../../scripts/yml2requirements.py environment.yml
```

#### 2. Update Metadata (if needed)

```bash
cd ..
nano .container-metadata.yml
```

#### 3. Update Changelog

```bash
nano CHANGELOG.md

# Add new version entry:
# ## [v2025.10.13] - 2025-10-13
# ### Added
# - New feature X
# ### Fixed
# - Bug Y
```

#### 4. Commit Changes

```bash
git add .
git commit -m "feat(pesa-fat): Update dependencies for v2025.10.13"
```

#### 5. Build Container

```bash
cd ../../..

# Auto-generate version from date
./scripts/build_container.sh projects/pesa-fat

# Or specify version explicitly
./scripts/build_container.sh projects/pesa-fat v2025.10.13
```

This will:
- Process the template
- Build the container
- Calculate checksum
- Save to storage location

#### 6. Test Container

```bash
# Basic test
CONTAINER_PATH=$(cat /tmp/last_build_path.txt || echo "...")

singularity exec --nv $CONTAINER_PATH python --version

# Run actual tests
singularity exec --nv \
    --bind /ia_models:/models \
    $CONTAINER_PATH \
    python -c "import torch; print(torch.cuda.is_available())"
```

#### 7. Update Registry

```bash
# Get SHA256 and size from build output
SHA256="..."  # From build output
SIZE_MB="..." # From build output

./scripts/update_registry.sh projects/pesa-fat v2025.10.13 $SHA256 $SIZE_MB
```

#### 8. Commit Registry Changes

```bash
git add registry/containers.json
git commit -m "registry: Add pesa-fat:v2025.10.13"
```

#### 9. Create Git Tag

```bash
git tag -a pesa-fat-v2025.10.13 -m "PESA-FAT v2025.10.13

$(head -20 projects/pesa-fat/CHANGELOG.md)"

git push origin main --tags
```

---

## Contributing New Containers

### Creating a New Project Container

#### 1. Create Directory Structure

```bash
mkdir -p projects/my-project/requirements
cd projects/my-project
```

#### 2. Create Template File

Create `singularity-gpu-my-project.def.template`:

```singularity
Bootstrap: localimage
From: ${CONTAINER_STORAGE}/base/gpu-base/gpu-base_${BASE_VERSION}.sif

%files
    ${REPO_ROOT}/projects/my-project/requirements/conda_requirements.txt /conda_req.txt
    ${REPO_ROOT}/projects/my-project/requirements/pip_requirements.txt /pip_req.txt

%environment
    export PATH=/opt/conda/bin:$PATH
    export CONTAINER_VERSION="${CONTAINER_VERSION}"
    export CONTAINER_NAME="my-project"

%post
    . /opt/conda/bin/activate
    conda install --file /conda_req.txt -y
    pip3 install -r /pip_req.txt
    
    # Project-specific setup
    # ...
    
    conda env export > /opt/my-project-environment.yml
    
    # Build metadata
    cat > /BUILD_INFO << EOF
Built: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Version: ${CONTAINER_VERSION}
Base: gpu-base_${BASE_VERSION}
Git Commit: ${GIT_COMMIT}
EOF
    
    conda clean -afy
    pip cache purge

%runscript
    echo "My Project Container ${CONTAINER_VERSION}"
    exec "$@"

%labels
    Version ${CONTAINER_VERSION}
    Project my-project

%help
    My Project Container
    Usage: singularity run --nv my-project.sif
```

#### 3. Create Metadata File

Create `.container-metadata.yml`:

```yaml
name: my-project
display_name: "My Project Pipeline"
type: project
description: |
  Description of what this container does.

maintainers:
  - name: Your Name
    email: your.email@cnic.es

base_image: gpu-base
gpu_required: true
min_gpu_memory_gb: 8

dependencies:
  base_container: gpu-base
  models: []

tags:
  - my-tag
  - analysis
```

#### 4. Create Requirements

Create `requirements/environment.yml`:

```yaml
name: my-project
channels:
  - conda-forge
dependencies:
  - package1
  - package2
  - pip:
    - pip-package1
```

Generate split requirements:
```bash
python ../../../scripts/yml2requirements.py requirements/environment.yml
```

#### 5. Create Documentation

**README.md** - Include:
- Container overview and features
- Usage examples with bind mounts
- Dependencies and requirements
- Troubleshooting

**CHANGELOG.md** - Document:
- All versions with dates
- Added features, fixes, changes
- Known issues
- Migration guides between versions

See `projects/neuroimaging/eICAB/CHANGELOG.md` for example.

#### 6. Build and Test

```bash
./scripts/build_container.sh projects/my-project v2025.10.13
```

#### 7. Submit for Review

```bash
git add projects/my-project
git commit -m "feat: Add my-project container"
git push origin feature/my-project

# Create merge request
```

---

## External Containers

### What are External Containers?

External containers are pre-built containers from external sources where we don't have access to the source `.def` files.

**Examples:**
- eICAB - Circle of Willis segmentation
- TopCoW-ARG - Circle of Willis segmentation (ARG variant)
- TopCoW-CLAIM - Circle of Willis segmentation (CLAIM variant)

### Documenting External Containers

Even without source files, we document these containers:

#### 1. Create Metadata

Create `.container-metadata.yml` with `external: true`:

```yaml
name: external-tool
display_name: "External Tool"
type: project
external: true
description: |
  Description of the external tool.

source:
  type: external
  origin: "URL or source description"

gpu_required: false
maintainers:
  - name: Your Name
    email: your.email@cnic.es
```

#### 2. Create README

Document usage, available versions, and any known issues.

#### 3. Create CHANGELOG

Document all versions, fixes applied, and dependency updates.

**Important:** If you fix broken dependencies in an external container, document this in CHANGELOG.md.

#### 4. Add to Registry

External containers are tracked in `registry/containers.json` with `"external": true` flag.

---

## Best Practices

### Container Development

✅ **Do:**
- Use template variables (`${CONTAINER_VERSION}`, `${BASE_VERSION}`, `${REPO_ROOT}`, `${CONTAINER_STORAGE}`)
- Keep requirements in `environment.yml`
- Document all dependencies in `.container-metadata.yml`
- Update `CHANGELOG.md` for **every** version
- Test thoroughly before committing
- Use descriptive commit messages
- Verify checksums after building (quick and full)
- Document fixes to external containers

❌ **Don't:**
- Hardcode paths (use template variables)
- Skip CHANGELOG updates
- Commit without testing
- Use vague version numbers
- Commit `.sif` files to git

### Versioning

Containers use **Calendar Versioning** (CalVer):

```
Format: vYYYY.M.D[-iteration]

Examples:
  v2025.10.13      # October 13, 2025
  v2025.10.13-2    # Second build same day
  v2025.1.5        # January 5, 2025
```

### Security

- Always verify checksums before using containers
- Don't run untrusted containers with privileged access
- Review external containers before deployment
- Keep containers updated (rebuild regularly with latest dependencies)

---

## Quick Reference

### Common Commands

```bash
# Find container
jq -r '.containers.projects["NAME"].latest' registry/containers.json
jq -r '.containers.projects["NAME"].versions["VERSION"].sif_path' registry/containers.json

# Use container
singularity shell --nv CONTAINER.sif
singularity exec --nv CONTAINER.sif COMMAND
singularity run --nv CONTAINER.sif

# Build container
./scripts/build_container.sh projects/NAME [VERSION]

# Verify container
./scripts/verify_container.sh projects/NAME VERSION

# Update registry
./scripts/update_registry.sh projects/NAME VERSION SHA256 SIZE_MB
```

### File Locations

- **Container storage:** `/run/user/11503/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_FSC/LAB/PERSONAL/imarcoss/containers`
- **Registry:** `registry/containers.json`
- **Scripts:** `scripts/`
- **Templates:** `base/*/` and `projects/*/`

---

## Troubleshooting

### Build Fails

**Problem:** Container build fails

```bash
# Check disk space
df -h

# Check permissions
ls -la /path/to/storage

# Try with explicit sudo
sudo singularity build output.sif input.def

# Check logs
cat build.log
```

### GPU Not Available

**Problem:** CUDA not available in container

```bash
# Check host GPU
nvidia-smi

# Use --nv flag
singularity exec --nv container.sif python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA in container
singularity exec --nv container.sif nvidia-smi
```

### Checksum Mismatch

**Problem:** Verification fails

```bash
# File may be corrupted - rebuild
./scripts/build_container.sh projects/pesa-fat v2025.10.13
```

### Registry Out of Sync

**Problem:** Registry doesn't match actual files

```bash
# Update registry manually
./scripts/update_registry.sh projects/pesa-fat v2025.10.13 <sha256> <size>

# Or recalculate checksum
sha256sum /path/to/container.sif
```

### Missing Dependencies

**Problem:** Container missing required packages

```bash
# Check what's installed
singularity exec container.sif pip list
singularity exec container.sif conda list

# Update requirements and rebuild
cd projects/my-project/requirements
nano environment.yml
python ../../../scripts/yml2requirements.py environment.yml
cd ../../..
./scripts/build_container.sh projects/my-project
```

---

## Additional Resources

- [Scripts Documentation](scripts/README.md)
- [Registry Documentation](registry/README.md)

---

## Support

For questions or issues:
- **Email:** imarcoss@cnic.es
- **GitLab:** Open an issue in this repository
- **Documentation:** Check container-specific READMEs

---

**Last Updated:** October 2025  
**Version:** 1.0

