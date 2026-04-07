#!/bin/bash
# Update registry with new container version
#
# Usage:
#   ./scripts/update_registry.sh <container_path> <version> <sha256> <size_mb>
#
# Examples:
#   ./scripts/update_registry.sh base/gpu-base v2025.10.13 abc123... 8500
#   ./scripts/update_registry.sh projects/pesa-fat v2025.10.13 def456... 12500

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

error() {
    echo -e "${RED}✗ Error:${NC} $1" >&2
    exit 1
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# ════════════════════════════════════════════════════════════
# Argument Parsing
# ════════════════════════════════════════════════════════════

if [ $# -lt 4 ]; then
    echo "Usage: $0 <container_path> <version> <sha256> <size_mb>"
    echo ""
    echo "Examples:"
    echo "  $0 base/gpu-base v2025.10.13 abc123... 8500"
    echo "  $0 projects/pesa-fat v2025.10.13 def456... 12500"
    exit 1
fi

CONTAINER_PATH="$1"
VERSION="$2"
SHA256="$3"
SIZE_MB="$4"

# ════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
cd "$REPO_ROOT"

REGISTRY_FILE="registry/containers.json"

if [ ! -f "$REGISTRY_FILE" ]; then
    error "Registry file not found: $REGISTRY_FILE"
fi

# Determine container type and name
if [[ "$CONTAINER_PATH" == base/* ]]; then
    CONTAINER_TYPE="base"
    CONTAINER_NAME=$(basename "$CONTAINER_PATH")
elif [[ "$CONTAINER_PATH" == projects/* ]]; then
    CONTAINER_TYPE="projects"
    CONTAINER_NAME=$(basename "$CONTAINER_PATH")
else
    error "Invalid container path. Must start with 'base/' or 'projects/'"
fi

# Get storage path from registry
CONTAINER_STORAGE=$(jq -r '.infrastructure.container_storage' "$REGISTRY_FILE")

if [ "$CONTAINER_TYPE" = "base" ]; then
    SIF_PATH="${CONTAINER_STORAGE}/base/${CONTAINER_NAME}/${CONTAINER_NAME}_${VERSION}.sif"
else
    SIF_PATH="${CONTAINER_STORAGE}/projects/${CONTAINER_NAME}/${CONTAINER_NAME}_${VERSION}.sif"
fi

# Get git information
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_TAG="${CONTAINER_NAME}-${VERSION}"
BUILT_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
BUILT_BY=$(git config user.name 2>/dev/null || whoami)

info "Updating registry for: $CONTAINER_NAME:$VERSION"
info "Registry: $REGISTRY_FILE"

# ════════════════════════════════════════════════════════════
# Update Registry JSON
# ════════════════════════════════════════════════════════════

# Create a temporary Python script to update the JSON
cat > /tmp/update_registry.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import json
import sys
from datetime import datetime

registry_file = sys.argv[1]
container_type = sys.argv[2]
container_name = sys.argv[3]
version = sys.argv[4]
sif_path = sys.argv[5]
sha256 = sys.argv[6]
size_mb = int(sys.argv[7])
git_commit = sys.argv[8]
git_tag = sys.argv[9]
built_date = sys.argv[10]
built_by = sys.argv[11]

# Load registry
with open(registry_file, 'r') as f:
    registry = json.load(f)

# Ensure container entry exists
if container_name not in registry['containers'][container_type]:
    registry['containers'][container_type][container_name] = {
        'latest': version,
        'versions': {}
    }

# Create version entry
version_entry = {
    'git_tag': git_tag,
    'git_commit': git_commit,
    'built_date': built_date,
    'built_by': built_by,
    'sif_path': sif_path,
    'sif_sha256': sha256,
    'size_mb': size_mb,
    'metadata': {},
    'dependencies': {}
}

# Add version
registry['containers'][container_type][container_name]['versions'][version] = version_entry

# Update latest
registry['containers'][container_type][container_name]['latest'] = version

# Update last_updated
registry['last_updated'] = datetime.utcnow().isoformat() + 'Z'

# Save registry
with open(registry_file, 'w') as f:
    json.dump(registry, f, indent=2)

print(f"✓ Updated {container_type}/{container_name}:{version}")
PYTHON_SCRIPT

chmod +x /tmp/update_registry.py

python3 /tmp/update_registry.py \
    "$REGISTRY_FILE" \
    "$CONTAINER_TYPE" \
    "$CONTAINER_NAME" \
    "$VERSION" \
    "$SIF_PATH" \
    "$SHA256" \
    "$SIZE_MB" \
    "$GIT_COMMIT" \
    "$GIT_TAG" \
    "$BUILT_DATE" \
    "$BUILT_BY"

rm /tmp/update_registry.py

# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════

success "Registry updated successfully!"
echo ""
echo "Container: $CONTAINER_NAME:$VERSION"
echo "Path:      $SIF_PATH"
echo "SHA256:    $SHA256"
echo "Size:      ${SIZE_MB} MB"
echo ""
echo "Next steps:"
echo "  1. Review the registry:"
echo "     cat $REGISTRY_FILE"
echo ""
echo "  2. Commit the changes:"
echo "     git add $REGISTRY_FILE"
echo "     git commit -m \"registry: Add $CONTAINER_NAME:$VERSION\""
echo ""
echo "  3. Create git tag:"
echo "     git tag -a $GIT_TAG -m \"$CONTAINER_NAME $VERSION\""
echo "     git push origin main --tags"
echo ""

