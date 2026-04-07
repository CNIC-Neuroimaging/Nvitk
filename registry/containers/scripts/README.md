# Build and Management Scripts

This directory contains scripts for building, managing, and verifying containers.

## Scripts

### `build_container.sh`

Build a container from a template definition file.

**Usage:**
```bash
./scripts/build_container.sh <container_path> [version]
```

**Examples:**
```bash
# Auto-generate version from current date
./scripts/build_container.sh base/gpu-base

# Specify version explicitly
./scripts/build_container.sh base/gpu-base v2025.10.13

# Build project container
./scripts/build_container.sh projects/pesa-fat v2025.10.13
```

**What it does:**
1. Reads the `.def.template` file from container directory
2. Substitutes environment variables (version, base container, etc.)
3. Builds container with `singularity build`
4. Calculates SHA256 checksum
5. Saves to container storage location
6. Records checksums

**Environment variables available in templates:**
- `${REPO_ROOT}` - Repository root path
- `${CONTAINER_VERSION}` - Container version being built
- `${CONTAINER_NAME}` - Container name
- `${GIT_COMMIT}` - Current git commit hash
- `${GIT_TAG}` - Git tag name for this version
- `${BASE_VERSION}` - Base container version (for project containers)
- `${CONTAINER_STORAGE}` - Container storage root path

---

### `yml2requirements.py`

Convert conda `environment.yml` to separate conda and pip requirements files.

**Usage:**
```bash
python scripts/yml2requirements.py <path/to/environment.yml>
```

**Example:**
```bash
python scripts/yml2requirements.py base/gpu-base/requirements/environment.yml
```

**What it does:**
1. Parses `environment.yml`
2. Splits dependencies into conda and pip packages
3. Creates `conda_requirements.txt`
4. Creates `pip_requirements.txt`

**Input format (environment.yml):**
```yaml
name: my-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.24
  - scipy
  - pip:
    - torch>=2.0
    - transformers
```

**Output:**
- `conda_requirements.txt`: conda packages only
- `pip_requirements.txt`: pip packages only

---

### `update_registry.sh`

Update the central registry with a new container version.

**Usage:**
```bash
./scripts/update_registry.sh <container_path> <version> <sha256> <size_mb>
```

**Example:**
```bash
./scripts/update_registry.sh projects/pesa-fat v2025.10.13 abc123def456... 12500
```

**What it does:**
1. Adds or updates version entry in `registry/containers.json`
2. Sets as "latest" version
3. Records git tag and commit
4. Updates metadata

**Note:** This is automatically called after `build_container.sh` completes.

---

### `verify_container.sh`

Verify container integrity by checking SHA256 checksums against registry.

**Usage:**
```bash
./scripts/verify_container.sh <container_path> <version> [--full]
```

**Modes:**
- **Quick (default):** Uses cached checksums from `.checksums` file (instant)
- **Full (`--full`):** Recalculates SHA256 from `.sif` file (5-10 minutes)

**Examples:**
```bash
# Quick verification (recommended for daily use)
./scripts/verify_container.sh base/gpu-base v2025.10.13

# Full verification (for security-critical checks)
./scripts/verify_container.sh projects/pesa-fat v2025.10.13 --full
```

**What it does:**

**Quick mode:**
1. Reads checksum from `.checksums` file
2. Compares with registry checksum
3. Returns instantly (<1 second)

**Full mode:**
1. Calculates SHA256 from actual `.sif` file
2. Compares with registry checksum
3. Takes 5-10 minutes for large containers
4. Provides absolute verification

**Use cases:**
- **Quick:** Daily workflow, routine checks
- **Full:** After file transfers, suspected corruption, security audits

---

## Complete Workflow Example

### Building a New Container Version

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

## Checksum Files

Containers use a `.checksums` file for fast verification:

- **[CHECKSUMS_FORMAT.md](CHECKSUMS_FORMAT.md)** - Complete documentation of checksum file format

Quick overview:
- `.checksums` - Single file containing all version checksums (format: `SHA256  FILENAME`)

---

## Script Requirements

All scripts require:
- **Bash** (for shell scripts)
- **Python 3** (for Python scripts)
- **jq** - JSON processor (`sudo apt install jq`)
- **apptainer** or **singularity** - Container builder
- **git** - Version control

Python scripts require:
- `pyyaml` - Install with: `pip install pyyaml`

