# Container Checksum Files Format

This document describes the checksum file formats used in the BioImaging-Containers infrastructure.

---

## Overview

A single `.checksums` file is created in each container directory to track all version checksums.

This file is stored in the same directory as the container `.sif` files.

---

## File Locations

```
/containers/base/gpu-base/
├── gpu-base_v2025.5.13.sif       # Container file
├── gpu-base_v2025.10.13.sif      # Newer version
└── .checksums                    # Single checksums file (all versions)
```

---

## Checksums File (`.checksums`)

**Filename:** `.checksums`

**Format:** One line per container version, two fields separated by two spaces

```
<SHA256>  <FILENAME>
```

**Field Descriptions:**
- **SHA256:** 64-character hexadecimal checksum
- **FILENAME:** Container filename (e.g., `gpu-base_v2025.10.13.sif`)

**Example Content:**
```
abc123def456789012345678901234567890123456789012345678901234  gpu-base_v2025.5.13.sif
def456abc789012345678901234567890123456789012345678901234567  gpu-base_v2025.10.13.sif
789012345678901234567890abc123def456789012345678901234567890  gpu-base_v2025.10.14.sif
```

**Usage:**
```bash
# Find checksum for a specific version
grep "gpu-base_v2025.10.13.sif" .checksums

# Extract just the checksum
grep "gpu-base_v2025.10.13.sif" .checksums | awk '{print $1}'

# List all versions
cat .checksums

# Get latest entry
tail -1 .checksums
```

---

## Creation Process

### During Container Build

The `build_container.sh` script creates/updates the `.checksums` file:

```bash
# 1. Build container
apptainer build container.sif definition.def

# 2. Calculate checksum
CHECKSUM=$(sha256sum container.sif | awk '{print $1}')

# 3. Append to .checksums file
echo "${CHECKSUM}  container_v2025.10.13.sif" >> .checksums
```

---

## Verification Process

### Quick Verification (Default)

The `verify_container.sh` script uses the `.checksums` file for instant verification:

```bash
# Read cached checksum from .checksums file (no recalculation needed)
CACHED=$(grep "container_v2025.10.13.sif" .checksums | awk '{print $1}')

# Compare with registry
if [ "$CACHED" = "$REGISTRY_CHECKSUM" ]; then
    echo "✓ Verified (instant)"
fi
```

**Speed:** <1 second  
**Reliability:** Assumes checksum files are intact

### Full Verification (`--full` flag)

Recalculates SHA256 from the actual `.sif` file:

```bash
# Calculate from .sif file
ACTUAL=$(sha256sum container.sif | awk '{print $1}')

# Compare with registry
if [ "$ACTUAL" = "$REGISTRY_CHECKSUM" ]; then
    echo "✓ Verified (full)"
fi
```

**Speed:** 5-10 minutes for large containers  
**Reliability:** 100% verification of file integrity

---

## File Management

### Best Practices

✅ **Do:**
- Keep the `.checksums` file
- Append to `.checksums` (never overwrite)
- Create/update during build
- Verify checksums match registry

❌ **Don't:**
- Manually edit `.checksums` file
- Delete old entries from `.checksums`
- Store this file in version control (it's generated)

---

## Example Workflow

### After Building a Container

```bash
# Build creates/updates .checksums file automatically
./scripts/build_container.sh base/gpu-base v2025.10.13

# File updated:
# - /containers/base/gpu-base/.checksums (appended)
```

### Verifying a Container

```bash
# Quick check (uses .checksum_VERSION file)
./scripts/verify_container.sh base/gpu-base v2025.10.13
# Output: ✓ Quick verification PASSED (instant)

# Full check (recalculates from .sif)
./scripts/verify_container.sh base/gpu-base v2025.10.13 --full
# Output: ✓ FULL checksum verification PASSED (after 8 minutes)
```

### Manual Verification

```bash
cd /containers/base/gpu-base

# Read cached checksum
CACHED=$(grep "gpu-base_v2025.10.13.sif" .checksums | awk '{print $1}')
echo "Cached: $CACHED"

# Manually verify
ACTUAL=$(sha256sum gpu-base_v2025.10.13.sif | awk '{print $1}')
echo "Actual: $ACTUAL"

# Compare
[ "$CACHED" = "$ACTUAL" ] && echo "✓ Match" || echo "✗ Mismatch"
```

---

## Troubleshooting

### Checksums File Missing

If `.checksums` file is missing, recreate it:

```bash
# Recalculate and recreate for all containers in directory
cd /containers/base/gpu-base

# Create new .checksums file
rm -f .checksums

# Add all .sif files
for sif in *.sif; do
    SHA256=$(sha256sum "$sif" | awk '{print $1}')
    echo "${SHA256}  ${sif}" >> .checksums
done
```

### Checksum Mismatch

If quick verification shows mismatch:

```bash
# Run full verification to confirm
./scripts/verify_container.sh base/gpu-base v2025.10.13 --full

# If full verification also fails:
# 1. Container file is corrupted → re-copy or rebuild
# 2. Registry is out of sync → update registry
```

### Checksum Verification Failed

If checksums don't match:

```bash
# Recalculate from .sif file
ACTUAL=$(sha256sum gpu-base_v2025.10.13.sif | awk '{print $1}')
CACHED=$(grep "gpu-base_v2025.10.13.sif" .checksums | awk '{print $1}')

echo "Actual:  $ACTUAL"
echo "Cached:  $CACHED"

# Update .checksums if file is correct
# (Remove old entry and add new one)
grep -v "gpu-base_v2025.10.13.sif" .checksums > .checksums.tmp
echo "${ACTUAL}  gpu-base_v2025.10.13.sif" >> .checksums.tmp
mv .checksums.tmp .checksums
```

---

## Technical Notes

- **Format:** Plain text, Unix line endings (`\n`)
- **Encoding:** ASCII/UTF-8
- **Checksum:** SHA256 (64 hex characters)
- **Separator:** Two spaces between fields in `.checksums`
- **Append-only:** `.checksums` is never edited, only appended
- **Git ignored:** These files are generated, not version-controlled

---

## Related Documentation

- [verify_container.sh](README.md#verify_containersh) - Verification script
- [build_container.sh](README.md#build_containersh) - Build script
- [CONTAINER_GUIDE.md](../CONTAINER_GUIDE.md) - Complete usage guide

