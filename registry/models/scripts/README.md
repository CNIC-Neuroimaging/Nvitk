# BioImaging-Models Scripts

Automation scripts for managing the model registry.

## Available Scripts

### `update_registry.sh`

Updates the `registry/models.json` file with a new model version.

**Usage:**
```bash
./scripts/update_registry.sh <model_path> <version>
```

**Example:**
```bash
./scripts/update_registry.sh imaging/Neuroimaging/eICAB v2.0.0
```

**What it does:**
1. Reads metadata from `.model-metadata.yml`
2. Extracts model information
3. Updates `registry/models.json`
4. Sets git tag and commit information

**Requirements:**
- Python 3 with `pyyaml` package
- Metadata file must exist at `<model_path>/<version>/.model-metadata.yml`

### `tag_model.sh`

Creates a git tag for a model version.

**Usage:**
```bash
./scripts/tag_model.sh <model_path> <version>
```

**Example:**
```bash
./scripts/tag_model.sh imaging/Neuroimaging/eICAB v2.0.0
```

**What it does:**
1. Creates an annotated git tag
2. Includes CHANGELOG content in tag message
3. Links tag to current commit

**Format:**
```
Tag name: <model-name>-v<VERSION>
Examples: eicab-v2.0.0, cellpose3-cyto3-v1.0.0
```

## Complete Workflow

When adding a new model version:

```bash
# 1. Create version directory and files
mkdir -p imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0
cd imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0

# 2. Create metadata files
# - .model-metadata.yml
# - README.md
# - CHANGELOG.md

# 3. Return to repo root
cd ../../../../..

# 4. Commit metadata
git add imaging/Microscopy/Cellpose/Cellpose3/cyto3/v1.0.0
git commit -m "feat(cellpose3-cyto3): Add v1.0.0 metadata"

# 5. Update registry
./scripts/update_registry.sh imaging/Microscopy/Cellpose/Cellpose3/cyto3 v1.0.0

# 6. Commit registry update
git add registry/models.json
git commit -m "feat(registry): Add cellpose3-cyto3 v1.0.0"

# 7. Create git tag
./scripts/tag_model.sh imaging/Microscopy/Cellpose/Cellpose3/cyto3 v1.0.0

# 8. Push to remote
git push origin main --tags
```

## Script Requirements

### Python Dependencies

Install required Python packages:

```bash
pip install pyyaml
```

### Making Scripts Executable

```bash
chmod +x scripts/*.sh
```

## Troubleshooting

### "Metadata file not found"

Ensure the metadata file exists at the correct path:
```
<model_path>/<version>/.model-metadata.yml
```

### "Tag already exists"

If you need to recreate a tag:
```bash
# Delete local tag
git tag -d <tag-name>

# Delete remote tag (if pushed)
git push origin :refs/tags/<tag-name>

# Recreate tag
./scripts/tag_model.sh <model_path> <version>
```

### Registry Validation

Validate the registry after updates:
```bash
python3 -c "import json; json.load(open('registry/models.json'))"
```

## See Also

- `../CONTRIBUTING.md` - Guidelines for adding models
- `../MODEL_GUIDE.md` - Complete model usage guide
- `../registry/README.md` - Registry documentation

