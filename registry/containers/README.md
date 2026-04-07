# BioImaging Containers

**Centralized container infrastructure for reproducible bioimaging and computational biology workflows**

---

## 📋 Overview

This repository serves as the **container registry** for the BioImaging computational unit. It maintains version-controlled container definitions, metadata, and a central registry linking to built containers on the server.

### Architecture

```
BioImaging Container Infrastructure
│
├── Git Repository (this repo)
│   ├── 📝 Container definitions (.def.template)
│   ├── 📋 Metadata (.container-metadata.yml)
│   ├── 🛠️  Build scripts
│   └── 📊 Central registry (containers.json)
│
└── Server Storage
    └── /containers/
        ├── base/          → Foundation containers
        └── projects/      → Project-specific containers
```

**Key Principle:** Git repository contains **definitions and metadata**; server contains **built images** (`.sif` files).

---

## 🗂️ Repository Structure

```
BioImaging-Containers/
├── README.md                          # This file
├── CONTRIBUTING.md                    # How to contribute
├── CONTAINER_GUIDE.md                 # Complete usage guide
│
├── registry/                          # Central registry
│   ├── containers.json                # Main registry (all containers)
│   ├── schema.json                    # Registry schema
│   └── README.md                      # Registry documentation
│
├── scripts/                           # Build & management tools
│   ├── build_container.sh             # Build from template
│   ├── yml2requirements.py            # Requirements converter
│   ├── update_registry.sh             # Update registry
│   ├── verify_container.sh            # Verify checksums
│   └── README.md                      # Script documentation
│
├── base/                              # Base containers
│   └── gpu-base/                      # GPU-enabled base
│       ├── singularity-gpu-base.def.template
│       ├── .container-metadata.yml
│       ├── README.md
│       ├── CHANGELOG.md
│       └── requirements/
│
├── projects/                          # Project containers
│   ├── pesa-fat/                      # PESA-FAT pipeline
│   │   ├── singularity-gpu-pesa-fat.def.template
│   │   ├── .container-metadata.yml
│   │   ├── README.md
│   │   ├── CHANGELOG.md
│   │   └── requirements/
│   │
│   └── neuroimaging/
│       ├── eICAB/                     # External container
│       └── TopCoW/
│           ├── TopCoW-ARG/            # External container
│           └── TopCoW-CLAIM/          # External container
│
└── docs/                              # Documentation
    └── infrastructure/
        ├── INFRASTRUCTURE_PROPOSAL.md
        ├── IMPLEMENTATION_PLAN.md
        └── EXAMPLES.md
```

---

## 🚀 Quick Start

### Finding a Container

All containers are catalogued in the registry:

```bash
# List all available containers
jq '.containers' registry/containers.json

# Get latest version of a container
jq -r '.containers.projects["pesa-fat"].latest' registry/containers.json

# Get container path
jq -r '.containers.projects["pesa-fat"].versions["v2025.5.27"].sif_path' registry/containers.json
```

### Using a Container

```bash
# Get container path from registry
CONTAINER=$(jq -r '.containers.projects["pesa-fat"].versions["v2025.5.27"].sif_path' \
    registry/containers.json)

# Interactive shell (with GPU)
apptainer shell --nv $CONTAINER

# Run analysis
apptainer exec --nv \
    --bind /ia_models:/models \
    --bind /data:/data \
    $CONTAINER \
    python analysis.py
```

### Building a New Version

```bash
# 1. Update requirements (if needed)
cd base/gpu-base/requirements
nano environment.yml
python ../../../scripts/yml2requirements.py environment.yml

# 2. Update metadata
cd ..
nano .container-metadata.yml

# 3. Update changelog
nano CHANGELOG.md

# 4. Commit changes
git add .
git commit -m "feat(gpu-base): Update dependencies for v2025.10.13"

# 5. Build container
cd ../../..
./scripts/build_container.sh base/gpu-base v2025.10.13

# 6. Test container
apptainer exec /path/to/container/gpu-base_v2025.10.13.sif python --version

# 7. Update registry
./scripts/update_registry.sh base/gpu-base v2025.10.13 <sha256> <size_mb>

# 8. Verify (quick check)
./scripts/verify_container.sh base/gpu-base v2025.10.13

# 9. Full verification (optional, for critical deployments)
./scripts/verify_container.sh base/gpu-base v2025.10.13 --full

# 9. Create git tag
git tag -a gpu-base-v2025.10.13 -m "GPU base container v2025.10.13"
git push origin main --tags
```

---

## 📦 Available Containers

### Base Containers

| Container | Description | GPU | Latest |
|-----------|-------------|-----|--------|
| **gpu-base** | CUDA 12.8 + PyTorch + CuPy | ✅ | v2025.5.13 |

[📄 Base container documentation](base/gpu-base/README.md)

### Project Containers

| Container | Description | GPU | External | Latest |
|-----------|-------------|-----|----------|--------|
| **pesa-fat** | PESA-FAT analysis pipeline | ✅ | No | v2025.5.27 |
| **eICAB** | Brain extraction & parcellation | ❌ | Yes | v2023.4.17 |
| **TopCoW-ARG** | Circle of Willis segmentation | ✅ | Yes | v2025.10.23 |
| **TopCoW-CLAIM** | Circle of Willis segmentation | ✅ | Yes | v2025.10.10 |

**External containers** are pre-built containers from external sources (no `.def` files available).

---

## 📚 Documentation

### For Users

- **[Container Guide](CONTAINER_GUIDE.md)** - Complete guide to using containers
- **[Registry README](registry/README.md)** - How to query the registry
- **Container-specific READMEs** - In each container directory

### For Developers

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute new containers
- **[Scripts Documentation](scripts/README.md)** - Build and management tools
---

## 🔑 Key Features

### 1. Template-Based Builds

Container definitions use templates with variables:

```singularity
Bootstrap: localimage
From: ${CONTAINER_STORAGE}/base/gpu-base/gpu-base_${BASE_VERSION}.sif

%environment
    export CONTAINER_VERSION="${CONTAINER_VERSION}"
    export CONTAINER_NAME="${CONTAINER_NAME}"
```

**Environment Variables Supported:**
- `${REPO_ROOT}` - Repository root path
- `${CONTAINER_VERSION}` - Container version
- `${CONTAINER_NAME}` - Container name
- `${CONTAINER_STORAGE}` - Storage location
- `${BASE_VERSION}` - Base container version (for projects)
- `${GIT_COMMIT}` - Git commit hash
- `${GIT_TAG}` - Git tag name

### 2. Central Registry

The `registry/containers.json` file tracks:
- All container versions
- File paths and checksums
- Dependencies and metadata
- Latest versions (no filesystem symlinks needed)

### 3. Verification System

Every container has a SHA256 checksum:

```bash
./scripts/verify_container.sh projects/pesa-fat v2025.5.27
```

Ensures integrity before use.

### 4. External Container Support

Even containers built externally are documented with:
- Metadata files
- Usage documentation
- Registry entries
- Version tracking

---

## 🛠️ Common Tasks

### Find a Container

```bash
# List all containers
jq '.containers.projects | keys' registry/containers.json
```

### Use a Container

```bash
# Get path
CONTAINER=$(jq -r '.containers.projects["pesa-fat"].versions["v2025.5.27"].sif_path' \
    registry/containers.json)

# Run
apptainer exec --nv --bind /ia_models:/models $CONTAINER python script.py
```

### Build a Container

```bash
./scripts/build_container.sh projects/my-project v2025.10.13
```

### Verify a Container

```bash
# Quick verification (instant, uses cached checksums)
./scripts/verify_container.sh projects/pesa-fat v2025.5.27

# Full verification (5-10 min, recalculates SHA256)
./scripts/verify_container.sh projects/pesa-fat v2025.5.27 --full
```

---

## 🔄 Versioning

Containers use **Calendar Versioning** (CalVer):

```
Format: vYYYY.M.D[-iteration]

Examples:
  v2025.10.13      # October 13, 2025
  v2025.10.13-2    # Second build same day
  v2025.1.5        # January 5, 2025
```
---

## 📊 Container Storage

**Location:** `/run/user/11503/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_FSC/LAB/PERSONAL/imarcoss/containers`

**Structure:**
```
/containers/
.
├── base
│   └── gpu-base
│       ├── .checksums
│       └── gpu-base_v2025.5.13.sif
└── projects
    ├── neuroimaging
    │   ├── eICAB
    │   │   ├── .checksums
    │   │   ├── eICAB_v2022.10.15.sif
    │   │   └── eICAB_v2023.4.17.sif
    │   └── TopCoW
    │       ├── TopCoW-ARG
    │       │   ├── .checksums
    │       │   └── TowCoW-ARG_v2025.10.23.sif
    │       └── TopCoW-CLAIM
    │           ├── .checksums
    │           └── TowCoW-CLAIM_v2025.10.10.sif
    └── pesa-fat
        ├── .checksums
        └── gpu-pesa-fat_v2025.5.27.sif
```
---

## ✅ Best Practices

### For Users

- ✅ Always verify checksums before using containers
- ✅ Use `--nv` flag for GPU containers
- ✅ Bind mount required directories (`/ia_models`, `/data`)
- ✅ Check container README for specific usage instructions

### For Developers

- ✅ Update `CHANGELOG.md` for every version
- ✅ Test containers before committing
- ✅ Use meaningful commit messages
- ✅ Document dependencies in `.container-metadata.yml`
- ✅ Create git tags for releases

---

## 🐛 Troubleshooting

### Container Not Found

```bash
# Verify in registry
jq '.containers.projects["NAME"]' registry/containers.json

# Check actual file exists
ls -la $(jq -r '.containers.projects["NAME"].versions["VERSION"].sif_path' registry/containers.json)
```

### GPU Not Available

```bash
# Use --nv flag
apptainer exec --nv container.sif nvidia-smi

# Check host GPU
nvidia-smi
```
---

## 📝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to add new containers
- Coding standards
- Review process
- Testing requirements

Quick contribution workflow:
1. Create feature branch
2. Add container with all required files
3. Test
4. Submit merge request

---

## 📞 Support

**Maintainer:** Ignacio Marcos Serrano

**Contact:**
- Email: imarcoss@cnic.es
- GitLab: Open an issue in this repository
---

## 🔗 Related Projects

- **BioImaging-Models** - Model weights registry

---

**Last Updated:** October 2025  
**Infrastructure Version:** 1.0

For detailed documentation, see [CONTAINER_GUIDE.md](CONTAINER_GUIDE.md).
