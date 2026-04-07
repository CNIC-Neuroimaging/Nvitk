# Model Registry

This directory contains the central registry for all models in the BioImaging infrastructure.

## Files

- **`models.json`** - Main registry containing metadata for all models and versions
- **`schema.json`** - JSON schema definition for registry validation
- **`README.md`** - This file

## Registry Structure

The `models.json` file contains:

```json
{
  "schema_version": "1.0",
  "last_updated": "ISO 8601 timestamp",
  "infrastructure": {
    "model_storage": "/ia_models"
  },
  "models": {
    "model-name": {
      "category": "imaging/Category/Model",
      "type": "segmentation|classification|detection|...",
      "latest": "1.0.0",
      "versions": {
        "1.0.0": {
          "git_tag": "model-name-v1.0.0",
          "git_commit": "abc123",
          "added_date": "2025-10-14T00:00:00Z",
          "location": "/ia_models/...",
          "files": ["model.pt"],
          "config_files": ["config.json"],
          "size_mb": "unknown",
          "source": "pretrained|external|trained-inhouse",
          "compatible_containers": ["container-name >= vYYYY.M.D"]
        }
      }
    }
  }
}
```

## Usage

### List All Models

```bash
jq '.models | keys' models.json
```

### Get Model Information

```bash
# Get latest version
jq -r '.models["cellpose3-cyto3"].latest' models.json

# Get model location
jq -r '.models["cellpose3-cyto3"].versions["1.0.0"].location' models.json

# Get all files
jq -r '.models["cellpose3-cyto3"].versions["1.0.0"].files[]' models.json
```

### Find Models by Category

```bash
jq -r '.models[] | select(.category | startswith("imaging/Microscopy"))' models.json
```

## Updating the Registry

Use the provided scripts in `../scripts/`:

```bash
# Add or update a model version
../scripts/update_registry.sh imaging/Microscopy/Cellpose/Cellpose3/cyto3 v1.0.0

# Create git tag for a version
../scripts/tag_model.sh imaging/Microscopy/Cellpose/Cellpose3/cyto3 v1.0.0
```

## Validation

Validate the registry against the schema:

```bash
# Using Python
python3 -c "import json, jsonschema; jsonschema.validate(json.load(open('models.json')), json.load(open('schema.json')))"

# Using ajv-cli (if installed)
ajv validate -s schema.json -d models.json
```

## Version Control

The registry is version controlled in Git. Each model version has:

1. **Git tag** - Permanent label (e.g., `cellpose3-cyto3-v1.0.0`)
2. **Registry entry** - Metadata in `models.json`
3. **Documentation** - README and metadata in model version directory

The git tag points to the commit containing the model's metadata files.

