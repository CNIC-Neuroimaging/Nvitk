#!/bin/bash
# Update models.json with new model version
#
# Usage:
#   ./scripts/update_registry.sh <model_path> <version>
#
# Example:
#   ./scripts/update_registry.sh imaging/Neuroimaging/eICAB v2.0.0

set -e

MODEL_PATH=$1
VERSION=$2

if [ -z "$MODEL_PATH" ] || [ -z "$VERSION" ]; then
    echo "Usage: $0 <model_path> <version>"
    echo "Example: $0 imaging/Neuroimaging/eICAB v2.0.0"
    exit 1
fi

# Get repository root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT"

# Extract model name from path
MODEL_NAME=$(basename "$MODEL_PATH")

# Metadata file path
METADATA_FILE="$MODEL_PATH/$VERSION/.model-metadata.yml"

if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found: $METADATA_FILE"
    exit 1
fi

# Get git info
GIT_COMMIT=$(git rev-parse --short HEAD)
GIT_TAG="${MODEL_NAME}-${VERSION}"
ADDED_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Updating registry for: $MODEL_NAME $VERSION"
echo "  Metadata: $METADATA_FILE"
echo "  Git tag: $GIT_TAG"
echo "  Git commit: $GIT_COMMIT"
echo ""

# Extract fields from metadata using python
python3 - "$METADATA_FILE" "$MODEL_NAME" "$VERSION" "$GIT_TAG" "$GIT_COMMIT" "$ADDED_DATE" <<'PYTHON_SCRIPT'
import sys
import json
import yaml
from pathlib import Path

metadata_file = sys.argv[1]
model_name = sys.argv[2]
version = sys.argv[3]
git_tag = sys.argv[4]
git_commit = sys.argv[5]
added_date = sys.argv[6]

# Load metadata
with open(metadata_file) as f:
    metadata = yaml.safe_load(f)

# Load current registry
registry_file = Path("registry/models.json")
with open(registry_file) as f:
    registry = json.load(f)

# Extract info from metadata
category = metadata.get('domain', 'unknown')
model_type = metadata.get('type', 'unknown')
location = metadata.get('weights', {}).get('location', '')
files = metadata.get('weights', {}).get('files', [])
config_files = metadata.get('weights', {}).get('config_files', [])
size_mb = metadata.get('weights', {}).get('size_mb', 'unknown')
source = metadata.get('source', {}).get('type', 'unknown') if isinstance(metadata.get('source'), dict) else 'unknown'
compatible_containers = metadata.get('compatible_containers', ['unknown'])

# Construct full category path (from MODEL_PATH)
# We need to reconstruct this from the metadata or path
# For now, use a simplified approach
full_category = metadata_file.replace('/.model-metadata.yml', '').replace(f'/{version}', '')

# Create or update model entry
if model_name not in registry['models']:
    registry['models'][model_name] = {
        'category': full_category,
        'type': model_type,
        'latest': version.lstrip('v'),
        'versions': {}
    }
else:
    # Update latest if this is a newer version
    current_latest = registry['models'][model_name].get('latest', '0.0.0')
    new_version = version.lstrip('v')
    if new_version > current_latest:
        registry['models'][model_name]['latest'] = new_version

# Add version entry
version_key = version.lstrip('v')
registry['models'][model_name]['versions'][version_key] = {
    'git_tag': git_tag,
    'git_commit': git_commit,
    'added_date': added_date,
    'location': location,
    'files': files,
    'config_files': config_files,
    'size_mb': size_mb,
    'source': source,
    'compatible_containers': compatible_containers
}

# Add optional fields if present
if 'structure' in metadata.get('weights', {}):
    registry['models'][model_name]['versions'][version_key]['structure'] = metadata['weights']['structure']

if 'notes' in metadata:
    registry['models'][model_name]['versions'][version_key]['notes'] = metadata['notes']

# Update last_updated timestamp
registry['last_updated'] = added_date

# Save registry
with open(registry_file, 'w') as f:
    json.dump(registry, f, indent=2)
    f.write('\n')

print(f"✓ Registry updated successfully")
print(f"  Model: {model_name}")
print(f"  Version: {version_key}")
print(f"  Location: {location}")
PYTHON_SCRIPT

echo ""
echo "Next steps:"
echo "  1. Review changes: git diff registry/models.json"
echo "  2. Commit: git add registry/models.json && git commit -m 'feat($MODEL_NAME): Add $VERSION to registry'"
echo "  3. Create tag: ./scripts/tag_model.sh $MODEL_PATH $VERSION"

