#!/bin/bash
# Verify container integrity by checking SHA256 checksums
#
# Usage:
#   ./scripts/verify_container.sh <container_path> <version> [--full]
#
# Modes:
#   Quick (default): Uses cached checksums from .checksums file (instant)
#   Full (--full):   Recalculates SHA256 from .sif file (5-10 minutes)
#
# Examples:
#   ./scripts/verify_container.sh base/gpu-base v2025.10.13          # Quick
#   ./scripts/verify_container.sh projects/pesa-fat v2025.10.13 --full  # Full

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# ════════════════════════════════════════════════════════════
# Argument Parsing
# ════════════════════════════════════════════════════════════

if [ $# -lt 2 ]; then
    echo "Usage: $0 <container_path> <version> [--full]"
    echo ""
    echo "Modes:"
    echo "  Quick (default): Uses cached checksums (instant)"
    echo "  Full (--full):   Recalculates SHA256 (5-10 minutes)"
    echo ""
    echo "Examples:"
    echo "  $0 base/gpu-base v2025.10.13          # Quick"
    echo "  $0 projects/pesa-fat v2025.10.13 --full  # Full"
    exit 1
fi

CONTAINER_PATH="$1"
VERSION="$2"
VERIFICATION_MODE="quick"

if [ $# -ge 3 ] && [ "$3" = "--full" ]; then
    VERIFICATION_MODE="full"
fi

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
elif [[ "$CONTAINER_PATH" == projects/* ]]; then
    CONTAINER_TYPE="projects"
else
    error "Invalid container path. Must start with 'base/' or 'projects/'"
fi

CONTAINER_NAME=$(basename "$CONTAINER_PATH")

info "Verifying: $CONTAINER_NAME:$VERSION (mode: $VERIFICATION_MODE)"

# ════════════════════════════════════════════════════════════
# Get Container Information from Registry
# ════════════════════════════════════════════════════════════

CONTAINER_INFO=$(jq -r ".containers.${CONTAINER_TYPE}[\"${CONTAINER_NAME}\"].versions[\"${VERSION}\"]" "$REGISTRY_FILE")

if [ "$CONTAINER_INFO" = "null" ]; then
    error "Container $CONTAINER_NAME:$VERSION not found in registry"
fi

EXPECTED_SHA256=$(echo "$CONTAINER_INFO" | jq -r '.sif_sha256')
SIF_PATH=$(echo "$CONTAINER_INFO" | jq -r '.sif_path')
EXPECTED_SIZE=$(echo "$CONTAINER_INFO" | jq -r '.size_mb')

if [ "$EXPECTED_SHA256" = "null" ] || [ "$EXPECTED_SHA256" = "pending" ]; then
    warn "No checksum in registry for $CONTAINER_NAME:$VERSION"
    info "Cannot verify - checksum not yet recorded"
    exit 0
fi

info "Registry SHA256: $EXPECTED_SHA256"
info "Container path:  $SIF_PATH"

# ════════════════════════════════════════════════════════════
# Verify File Exists
# ════════════════════════════════════════════════════════════

if [ ! -f "$SIF_PATH" ]; then
    error "Container file not found: $SIF_PATH"
fi

success "Container file exists"

# ════════════════════════════════════════════════════════════
# Determine Storage Path and Checksums File
# ════════════════════════════════════════════════════════════

STORAGE_PATH=$(dirname "$SIF_PATH")
CHECKSUMS_FILE="${STORAGE_PATH}/.checksums"

# ════════════════════════════════════════════════════════════
# Quick Verification (default)
# ════════════════════════════════════════════════════════════

if [ "$VERIFICATION_MODE" = "quick" ]; then
    info "Running QUICK verification (using cached checksum)"
    
    # Read from .checksums file
    if [ -f "$CHECKSUMS_FILE" ]; then
        CACHED_ENTRY=$(grep "${CONTAINER_NAME}_${VERSION}.sif" "$CHECKSUMS_FILE" | tail -1)
        if [ -n "$CACHED_ENTRY" ]; then
            CACHED_SHA256=$(echo "$CACHED_ENTRY" | awk '{print $1}')
            info "Source: .checksums file"
        fi
    fi
    
    if [ -z "$CACHED_SHA256" ]; then
        warn "No cached checksum found"
        warn "Falling back to FULL verification..."
        VERIFICATION_MODE="full"
    else
        info "Cached SHA256:   $CACHED_SHA256"
        
        echo ""
        echo "═══════════════════════════════════════════════════════"
        
        if [ "$EXPECTED_SHA256" = "$CACHED_SHA256" ]; then
            success "Quick verification PASSED"
            echo "═══════════════════════════════════════════════════════"
            echo ""
            success "Container checksum matches registry"
            echo ""
            info "Registry: $EXPECTED_SHA256"
            info "Cached:   $CACHED_SHA256"
            echo ""
            info "This is a quick check using cached checksums."
            info "For full verification, run with --full flag:"
            info "  $0 $CONTAINER_PATH $VERSION --full"
            exit 0
        else
            warn "Quick verification FAILED - checksum mismatch!"
            echo "═══════════════════════════════════════════════════════"
            echo ""
            echo "Registry: $EXPECTED_SHA256"
            echo "Cached:   $CACHED_SHA256"
            echo ""
            warn "This could indicate:"
            warn "  1. Registry out of sync with storage"
            warn "  2. Corrupted cache file"
            warn ""
            warn "Running FULL verification to confirm..."
            VERIFICATION_MODE="full"
        fi
    fi
fi

# ════════════════════════════════════════════════════════════
# Full Verification (only if requested or quick failed)
# ════════════════════════════════════════════════════════════

if [ "$VERIFICATION_MODE" = "full" ]; then
    echo ""
    info "Running FULL SHA256 verification"
    warn "Calculating checksum from .sif file..."
    warn "This may take 5-10 minutes for large containers..."
    echo ""
    
    ACTUAL_SHA256=$(sha256sum "$SIF_PATH" | awk '{print $1}')
    ACTUAL_SIZE=$(du -m "$SIF_PATH" | awk '{print $1}')
    
    info "Computed SHA256: $ACTUAL_SHA256"
    info "File size:       ${ACTUAL_SIZE} MB"
    
    # ════════════════════════════════════════════════════════════
    # Compare Checksums
    # ════════════════════════════════════════════════════════════
    
    echo ""
    echo "═══════════════════════════════════════════════════════"
    
    if [ "$EXPECTED_SHA256" = "$ACTUAL_SHA256" ]; then
        success "FULL checksum verification PASSED"
        echo "═══════════════════════════════════════════════════════"
        echo ""
        success "Container is intact and matches registry"
        echo ""
        echo "Registry: $EXPECTED_SHA256"
        echo "Actual:   $ACTUAL_SHA256"
        echo "Size:     ${ACTUAL_SIZE} MB"
        
        # Check size
        if [ "$EXPECTED_SIZE" != "$ACTUAL_SIZE" ] && [ "$EXPECTED_SIZE" != "0" ]; then
            echo ""
            warn "Note: Size differs from registry (expected: ${EXPECTED_SIZE}MB, actual: ${ACTUAL_SIZE}MB)"
            warn "Checksums match - this is fine"
        fi
        
        exit 0
    else
        error "FULL checksum verification FAILED"
        echo "═══════════════════════════════════════════════════════"
        echo ""
        echo "Registry: $EXPECTED_SHA256"
        echo "Actual:   $ACTUAL_SHA256"
        echo ""
        error "Container file is corrupted or does not match registry!"
        echo ""
        echo "Possible causes:"
        echo "  1. File corruption during transfer"
        echo "  2. Wrong container version"
        echo "  3. Registry out of sync"
        echo ""
        echo "Recommended actions:"
        echo "  1. Re-copy or re-download the container"
        echo "  2. Rebuild the container if you have the .def file"
        echo "  3. Update registry if this is the correct file"
        exit 1
    fi
fi

