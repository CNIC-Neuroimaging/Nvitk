# Contributing to BioImaging-Containers

Thank you for contributing! This guide will help you add or modify container definitions in a consistent, validated, and maintainable way.

---

## 📝 Naming Conventions

### Directory Names
- Use **lowercase with hyphens** (kebab-case)
- Be descriptive but concise
- Examples: `gpu-base`, `pesa-fat`, `eICAB`

### Template File Names
- Must end with `.def.template`
- Format: `singularity-gpu-<name>.def.template` or `singularity-<name>.def.template`
- Avoid version numbers in filenames (versions are managed via Git tags and registry)

---

## ➕ Adding a New Container

### Step 1: Create Directory Structure

```bash
mkdir -p projects/my-project/requirements
cd projects/my-project
```

### Step 2: Create Template File

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
    conda env export > /opt/my-project-environment.yml
    conda clean -afy
    pip cache purge

%labels
    Version ${CONTAINER_VERSION}
    Project my-project
    Maintainer your.name@cnic.es

%help
    My Project Container ${CONTAINER_VERSION}
    Usage: apptainer run --nv my-project.sif
```

### Step 3: Create Metadata File

Create `.container-metadata.yml`:

```yaml
name: my-project
display_name: "My Project Pipeline"
type: project
description: |
  Brief description of what this container does.

maintainers:
  - name: Your Name
    email: your.name@cnic.es

base_image: gpu-base
gpu_required: true
min_gpu_memory_gb: 8

dependencies:
  base_container: gpu-base
  models: []

tags:
  - analysis
  - my-domain
```

### Step 4: Create Requirements

Create `requirements/environment.yml`:

```yaml
name: my-project
channels:
  - conda-forge
dependencies:
  - numpy>=1.24
  - scipy>=1.10
  - pip:
    - torch>=2.0
```

Generate split requirements:
```bash
python ../../scripts/yml2requirements.py requirements/environment.yml
```

### Step 5: Create Documentation

Create `README.md`:
- Container overview
- Features
- Usage examples
- Dependencies

Create `CHANGELOG.md`:
```markdown
# Changelog - My Project Container

## [v2025.10.13] - 2025-10-13

### Added
- Initial version
- Feature X
- Feature Y
```

### Step 6: Build and Test

```bash
cd ../../..
./scripts/build_container.sh projects/my-project v2025.10.13
```

### Step 7: Commit and Push

```bash
git add projects/my-project
git commit -m "feat: Add my-project container"
git push origin add-my-project
```

### Step 8: Create Merge Request

Open a merge request on GitLab.

---

## ✅ Required Files

Each container directory must include:
- ✅ `.def.template` - Template definition with variables
- ✅ `.container-metadata.yml` - Container metadata
- ✅ `README.md` - Documentation
- ✅ `CHANGELOG.md` - Version history
- ✅ `requirements/` - Dependencies (if applicable)
  - `environment.yml` - Conda environment
  - `conda_requirements.txt` - Generated from environment.yml
  - `pip_requirements.txt` - Generated from environment.yml

---

## 🎯 Best Practices

1. **Use template variables**: `${CONTAINER_VERSION}`, `${BASE_VERSION}`, `${REPO_ROOT}`, `${CONTAINER_STORAGE}`
2. **Pin dependency versions**: Specify exact versions in `environment.yml`
3. **Update CHANGELOG.md**: Document all changes for each version
4. **Test thoroughly**: Verify container before committing
5. **Verify checksums**: Run `./scripts/verify_container.sh` after building

---

## 🔄 Versioning

Containers use **Calendar Versioning (CalVer)**: `vYYYY.M.D`

```bash
# Build with auto-generated version
./scripts/build_container.sh projects/my-project

# Or specify version explicitly
./scripts/build_container.sh projects/my-project v2025.10.13

# Create git tag after building
git tag -a my-project-v2025.10.13 -m "My Project v2025.10.13"
git push origin main --tags
```

---

## 📋 Checklist for New Contributions

Before creating a merge request:

- [ ] Template file (`.def.template`) created with proper variables
- [ ] Metadata file (`.container-metadata.yml`) complete
- [ ] README.md is comprehensive
- [ ] CHANGELOG.md documents all versions
- [ ] Requirements files (`environment.yml`, split requirements) created
- [ ] Container builds successfully
- [ ] Container tested with actual workload
- [ ] Registry updated with new version
- [ ] No `.sif` files committed
- [ ] Git tag created for version

---

## 📦 External Containers

For containers obtained from external sources without `.def` files:

1. Create directory: `projects/neuroimaging/external-tool/`
2. Add `.container-metadata.yml` with `external: true`
3. Add README.md with usage documentation
4. Add CHANGELOG.md documenting versions and any fixes
5. Add to registry with `"external": true` flag

See `projects/neuroimaging/eICAB/` for example.

---

## 🛠️ Available Scripts

- **`scripts/build_container.sh`** - Build container from template
- **`scripts/verify_container.sh`** - Verify container integrity (quick/full modes)
- **`scripts/update_registry.sh`** - Update registry with new version
- **`scripts/yml2requirements.py`** - Generate split requirements from environment.yml

See [scripts/README.md](scripts/README.md) for details.

---

Thank you for contributing to reproducible science! 🔬
