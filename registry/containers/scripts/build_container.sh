#!/bin/bash
# Build container from template
#
# Usage:
#   ./scripts/build_container.sh <container_path> [version] [output_path] [--base-container <base_name:version>]
#
# Examples:
#   ./scripts/build_container.sh base/gpu-base
#   ./scripts/build_container.sh base/gpu-base v2025.10.13
#   ./scripts/build_container.sh projects/pesa-fat v2025.10.13                    # Standalone build (default)
#   ./scripts/build_container.sh projects/pesa-fat v2025.10.13 /tmp/custom_output  # Standalone build
#   ./scripts/build_container.sh projects/pesa-fat v2025.10.13 /tmp/custom_output --base-container gpu-base:v2025.5.13
#   ./scripts/build_container.sh projects/pesa-fat v2025.10.13 /tmp/custom_output --no-base  # Explicit standalone

# set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════

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

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ════════════════════════════════════════════════════════════
# Argument Parsing
# ════════════════════════════════════════════════════════════

if [ $# -lt 1 ]; then
    echo "Usage: $0 <container_path> [version] [output_path] [--base-container <base_name:version>] [--no-base]"
    echo ""
    echo "Examples:"
    echo "  $0 base/gpu-base"
    echo "  $0 base/gpu-base v2025.10.13"
    echo "  $0 projects/pesa-fat v2025.10.13                    # Standalone build (default)"
    echo "  $0 projects/pesa-fat v2025.10.13 /tmp/custom_output  # Standalone build"
    echo "  $0 projects/pesa-fat v2025.10.13 /tmp/custom_output --base-container gpu-base:v2025.5.13"
    echo "  $0 projects/pesa-fat v2025.10.13 /tmp/custom_output --no-base  # Explicit standalone"
    exit 1
fi

CONTAINER_PATH="$1"
VERSION="${2:-auto}"
OUTPUT_PATH="${3:-}"
BASE_CONTAINER_SPEC=""
NO_BASE=true  # Default to standalone builds for project containers

# Parse additional arguments
# Handle different numbers of positional arguments
if [ $# -ge 3 ]; then
    # We have container_path, version, and output_path
    shift 3
elif [ $# -eq 2 ]; then
    # We have container_path and version
    shift 2
elif [ $# -eq 1 ]; then
    # We have only container_path
    shift 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-container)
            BASE_CONTAINER_SPEC="$2"
            NO_BASE=false  # Override default when base container is specified
            shift 2
            ;;
        --no-base)
            NO_BASE=true
            shift
            ;;
        -*)
            error "Unknown option: $1"
            ;;
        *)
            # Non-option argument - treat as output path if not already set
            if [ -z "$OUTPUT_PATH" ]; then
                OUTPUT_PATH="$1"
            else
                error "Too many arguments: $1"
            fi
            shift
            ;;
    esac
done

# ════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
cd "$REPO_ROOT"

# Load container storage path from registry
CONTAINER_STORAGE=$(jq -r '.infrastructure.container_storage' registry/containers.json)

if [ -z "$CONTAINER_STORAGE" ] || [ "$CONTAINER_STORAGE" = "null" ]; then
    error "Could not read container_storage from registry/containers.json"
fi

# Determine container type and name
if [[ "$CONTAINER_PATH" == base/* ]]; then
    CONTAINER_TYPE="base"
    CONTAINER_NAME=$(basename "$CONTAINER_PATH")
    STORAGE_PATH="${CONTAINER_STORAGE}/base/${CONTAINER_NAME}"
elif [[ "$CONTAINER_PATH" == projects/* ]]; then
    CONTAINER_TYPE="projects"
    CONTAINER_NAME=$(basename "$CONTAINER_PATH")
    STORAGE_PATH="${CONTAINER_STORAGE}/projects/${CONTAINER_NAME}"
else
    error "Invalid container path. Must start with 'base/' or 'projects/'"
fi

# Find template file
TEMPLATE_FILE=$(find "$CONTAINER_PATH" -name "*.def.template" | head -1)

if [ -z "$TEMPLATE_FILE" ]; then
    error "No .def.template file found in $CONTAINER_PATH"
fi

info "Container: $CONTAINER_NAME (type: $CONTAINER_TYPE)"
info "Template: $TEMPLATE_FILE"

# ════════════════════════════════════════════════════════════
# Version Resolution
# ════════════════════════════════════════════════════════════

if [ "$VERSION" = "auto" ]; then
    VERSION="v$(date +%Y.%-m.%-d)"
    info "Auto-generated version: $VERSION"
    
    # Check if this version already exists
    if [ -f "${STORAGE_PATH}/${CONTAINER_NAME}_${VERSION}.sif" ]; then
        warn "Version $VERSION already exists"
        # Try with iteration suffix
        ITERATION=2
        while [ -f "${STORAGE_PATH}/${CONTAINER_NAME}_${VERSION}-${ITERATION}.sif" ]; do
            ITERATION=$((ITERATION + 1))
        done
        VERSION="${VERSION}-${ITERATION}"
        info "Using versioned iteration: $VERSION"
    fi
fi

# ════════════════════════════════════════════════════════════
# Environment Variables for Template
# ════════════════════════════════════════════════════════════

export REPO_ROOT
export CONTAINER_VERSION="$VERSION"
export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
export GIT_TAG="${CONTAINER_NAME}-${VERSION}"
export CONTAINER_NAME
export CONTAINER_STORAGE

# For base containers, no base version needed
# For project containers, determine base container version
if [ "$CONTAINER_TYPE" = "projects" ]; then
    if [ "$NO_BASE" = true ]; then
        info "Building project container without base container (standalone)"
        export BASE_VERSION=""
        export BASE_CONTAINER_NAME=""
    elif [ -n "$BASE_CONTAINER_SPEC" ]; then
        # Parse base container specification (format: name:version)
        if [[ "$BASE_CONTAINER_SPEC" == *":"* ]]; then
            BASE_CONTAINER_NAME=$(echo "$BASE_CONTAINER_SPEC" | cut -d':' -f1)
            BASE_VERSION=$(echo "$BASE_CONTAINER_SPEC" | cut -d':' -f2)
        else
            BASE_CONTAINER_NAME="$BASE_CONTAINER_SPEC"
            BASE_VERSION="latest"
        fi
        
        # Validate base container exists in registry
        if ! jq -e ".containers.base[\"$BASE_CONTAINER_NAME\"]" registry/containers.json > /dev/null; then
            error "Base container '$BASE_CONTAINER_NAME' not found in registry"
        fi
        
        # If version is 'latest', get the latest version from registry
        if [ "$BASE_VERSION" = "latest" ]; then
            BASE_VERSION=$(jq -r ".containers.base[\"$BASE_CONTAINER_NAME\"].latest" registry/containers.json)
        fi
        
        # Validate specific version exists
        if ! jq -e ".containers.base[\"$BASE_CONTAINER_NAME\"].versions[\"$BASE_VERSION\"]" registry/containers.json > /dev/null; then
            error "Base container '$BASE_CONTAINER_NAME' version '$BASE_VERSION' not found in registry"
        fi
        
        export BASE_CONTAINER_NAME
        export BASE_VERSION
        info "Using specified base container: $BASE_CONTAINER_NAME:$BASE_VERSION"
    else
        # Default behavior: standalone build (no base container)
        info "Using default behavior: standalone build (no base container)"
    fi
fi

# ════════════════════════════════════════════════════════════
# Generate .def from Template
# ════════════════════════════════════════════════════════════

DEF_FILE="/tmp/${CONTAINER_NAME}_${VERSION}.def"

info "Generating definition file from template..."

# Check if template supports the selected base container configuration
if [ "$CONTAINER_TYPE" = "projects" ]; then
    if [ "$NO_BASE" = true ]; then
        # Check if template has Bootstrap: localimage (which requires a base)
        if grep -q "Bootstrap: localimage" "$TEMPLATE_FILE"; then
            error "Template uses 'Bootstrap: localimage' but --no-base was specified. Template needs to be updated to use 'Bootstrap: docker' or similar for standalone builds."
        fi
        info "Template appears compatible with standalone build (no base container)"
    else
        # Check if template expects a base container
        if grep -q "Bootstrap: localimage" "$TEMPLATE_FILE"; then
            if [ -z "$BASE_VERSION" ]; then
                error "Template uses 'Bootstrap: localimage' but no base container version was determined"
            fi
            info "Template expects base container: $BASE_CONTAINER_NAME:$BASE_VERSION"
        else
            warn "Template does not use 'Bootstrap: localimage' - base container specification may be ignored"
        fi
    fi
fi

envsubst < "$TEMPLATE_FILE" > "$DEF_FILE"

success "Generated: $DEF_FILE"

# ════════════════════════════════════════════════════════════
# Create Output Directory
# ════════════════════════════════════════════════════════════

# Determine output path
if [ -n "$OUTPUT_PATH" ]; then
    # Use custom output path
    mkdir -p "$OUTPUT_PATH"
    OUTPUT_FILE="${OUTPUT_PATH}/${CONTAINER_NAME}_${VERSION}.sif"
    info "Using custom output path: $OUTPUT_PATH"
else
    # Use default storage path
    mkdir -p "$STORAGE_PATH"
    OUTPUT_FILE="${STORAGE_PATH}/${CONTAINER_NAME}_${VERSION}.sif"
fi

# ════════════════════════════════════════════════════════════
# Build Container
# ════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Building Container"
echo "═══════════════════════════════════════════════════════"
echo "  Container:  $CONTAINER_NAME"
echo "  Version:    $VERSION"
if [ "$CONTAINER_TYPE" = "projects" ] && [ "$NO_BASE" = false ] && [ -n "$BASE_VERSION" ]; then
    echo "  Base:       $BASE_CONTAINER_NAME:$BASE_VERSION"
elif [ "$CONTAINER_TYPE" = "projects" ] && [ "$NO_BASE" = true ]; then
    echo "  Base:       standalone (no base container)"
fi
echo "  Output:     $OUTPUT_FILE"
echo "═══════════════════════════════════════════════════════"
echo ""

# Build with apptainer (requires sudo or fakeroot)
if command -v apptainer &> /dev/null; then
    BUILDER="apptainer"
elif command -v singularity &> /dev/null; then
    BUILDER="singularity"
else
    error "Neither apptainer nor singularity found in PATH"
fi

info "Using builder: $BUILDER"

# Try to build (user may need sudo)
if sudo -n true 2>/dev/null; then
    sudo $BUILDER build "$OUTPUT_FILE" "$DEF_FILE"
else
    info "Attempting build without sudo (may fail)..."
    $BUILDER build --fakeroot "$OUTPUT_FILE" "$DEF_FILE" || \
        error "Build failed. You may need sudo privileges or --fakeroot support"
fi

# ════════════════════════════════════════════════════════════
# Calculate Checksum
# ════════════════════════════════════════════════════════════

echo ""
info "Calculating checksum..."

CHECKSUM=$(sha256sum "$OUTPUT_FILE" | awk '{print $1}')
SIZE_MB=$(du -m "$OUTPUT_FILE" | awk '{print $1}')

success "SHA256: $CHECKSUM"
success "Size: ${SIZE_MB} MB"

# Append to .checksums file only if using default storage path
if [ -z "$OUTPUT_PATH" ]; then
    echo "${CHECKSUM}  ${CONTAINER_NAME}_${VERSION}.sif" >> "${STORAGE_PATH}/.checksums"
fi

# ════════════════════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════════════════════

rm -f "$DEF_FILE"

# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════

echo ""
echo "═══════════════════════════════════════════════════════"
success "Build Complete!"
echo "═══════════════════════════════════════════════════════"
echo "  Container:  $OUTPUT_FILE"
echo "  Size:       ${SIZE_MB} MB"
echo "  SHA256:     $CHECKSUM"
echo "═══════════════════════════════════════════════════════"
echo ""

# Show reminder if using custom output path
if [ -n "$OUTPUT_PATH" ]; then
    echo "⚠️  REMINDER: Container created in custom location"
    echo "   You may want to move it to the registry storage:"
    echo "   mv $OUTPUT_FILE $STORAGE_PATH/"
    echo ""
fi

echo "Next steps:"
echo "  1. Test the container:"
echo "     $BUILDER exec $OUTPUT_FILE python --version"
echo ""

if [ -z "$OUTPUT_PATH" ]; then
    echo "  2. Update the registry:"
    echo "     ./scripts/update_registry.sh $CONTAINER_TYPE/$CONTAINER_NAME $VERSION $CHECKSUM $SIZE_MB"
    echo ""
    echo "  3. Create git tag (optional):"
    echo "     git tag -a ${GIT_TAG} -m \"${CONTAINER_NAME} ${VERSION}\""
    echo "     git push origin main --tags"
else
    echo "  2. Move container to registry storage (if desired):"
    echo "     mv $OUTPUT_FILE $STORAGE_PATH/"
    echo ""
    echo "  3. Update the registry (after moving):"
    echo "     ./scripts/update_registry.sh $CONTAINER_TYPE/$CONTAINER_NAME $VERSION $CHECKSUM $SIZE_MB"
    echo ""
    echo "  4. Create git tag (optional):"
    echo "     git tag -a ${GIT_TAG} -m \"${CONTAINER_NAME} ${VERSION}\""
    echo "     git push origin main --tags"
fi
echo ""

