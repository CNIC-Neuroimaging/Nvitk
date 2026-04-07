#!/bin/bash
# Create git tag for model version and update registry
#
# Usage:
#   ./scripts/tag_model.sh <model_path> <version>
#
# Example:
#   ./scripts/tag_model.sh imaging/Neuroimaging/eICAB v2.0.0

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

# Create tag name
TAG_NAME="${MODEL_NAME}-${VERSION}"

# Check if tag already exists
if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    echo "Warning: Tag $TAG_NAME already exists"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    git tag -d "$TAG_NAME"
fi

# Get commit hash
GIT_COMMIT=$(git rev-parse --short HEAD)

# Get changelog content if available (now from model level, not version folder)
MODEL_DIR=$(dirname "$MODEL_PATH")
CHANGELOG_FILE="$MODEL_DIR/CHANGELOG.md"
CHANGELOG_CONTENT=""
if [ -f "$CHANGELOG_FILE" ]; then
    CHANGELOG_CONTENT=$(head -30 "$CHANGELOG_FILE")
fi

# Create annotated tag
git tag -a "$TAG_NAME" -m "Model: $MODEL_NAME $VERSION

Path: $MODEL_PATH
Commit: $GIT_COMMIT

$CHANGELOG_CONTENT"

echo "✓ Created git tag: $TAG_NAME"
echo ""
echo "Tag details:"
git show "$TAG_NAME" --no-patch

# Update registry with git commit information
echo ""
echo "Updating registry with tag information..."

# Use Python to update the registry JSON with the git commit
python3 - "$GIT_COMMIT" "$MODEL_NAME" "$VERSION" <<'PYTHON_SCRIPT'
import sys
import json
from datetime import datetime
from pathlib import Path

git_commit = sys.argv[1]
model_name = sys.argv[2]
version = sys.argv[3].lstrip('v')

# Load registry
registry_file = Path("registry/models.json")
with open(registry_file) as f:
    registry = json.load(f)

# Update git_commit if model and version exist
if model_name in registry['models']:
    if version in registry['models'][model_name]['versions']:
        registry['models'][model_name]['versions'][version]['git_commit'] = git_commit
        registry['last_updated'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Save updated registry
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
            f.write('\n')
        
        print(f"✓ Updated registry: {model_name} v{version} → commit {git_commit}")
    else:
        print(f"⚠ Warning: Version {version} not found in registry for {model_name}")
        print(f"   Run: ./scripts/update_registry.sh {sys.argv[0]} v{version}")
else:
    print(f"⚠ Warning: Model {model_name} not found in registry")
    print(f"   Run: ./scripts/update_registry.sh {sys.argv[0]} v{version}")
PYTHON_SCRIPT

echo ""
echo "Next steps:"
echo "  1. Review changes: git diff registry/models.json"
echo "  2. Commit registry update: git add registry/models.json && git commit -m 'chore(registry): Update ${MODEL_NAME} ${VERSION} git commit'"
echo "  3. Push tag: git push origin $TAG_NAME"
echo "  4. Or push all: git push origin main --tags"
