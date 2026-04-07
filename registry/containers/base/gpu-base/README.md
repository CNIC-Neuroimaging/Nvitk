# GPU Base Container

General-purpose GPU-enabled base container for Torch model inference/training and Image Pre/Post-processing.

## 📋 Overview

This is the foundational container for all GPU-enabled project containers. It provides a standardized environment with CuPy (CUDA) and PyTorch.

**Type:** Base Container  
**GPU Required:** Yes  
**Minimum GPU Memory:** Unknown

## 🔧 What's Included

### System
- **OS:** Ubuntu 22.04
- **Conda:** Miniforge3
- **Python:** 3.11

### CUDA Stack
- **CUDA Toolkit:** 12.9
- **cuDNN:** 9
- **NCCL:** 2.26.5
- **cuTENSOR:** 2.2.0
- **Additional:** libcusparse-12-9

### Frameworks & Libraries
- **PyTorch:** 2.8.0 (CUDA 12.9 support)
- **TorchVision:** 0.23.0
- **CuPy:** 13.6.0 (GPU-accelerated NumPy)
- **SciPy:** 1.15.3

## 🚀 Usage

### Finding the Container

```bash
# Get latest version
jq -r '.containers.base["gpu-base"].latest' registry/containers.json

# Get container path
jq -r '.containers.base["gpu-base"].versions["v2025.10.15"].sif_path' registry/containers.json
```

### Running the Container

```bash
# Set container path (adjust version as needed)
CONTAINER="/path/to/containers/base/gpu-base/gpu-base_v2025.10.15.sif"

# Interactive shell with GPU support
singularity shell --nv $CONTAINER

# Execute Python script with GPU
singularity exec --nv $CONTAINER python your_script.py

# Check CUDA availability
singularity exec --nv $CONTAINER python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Using in Project Containers

This base container is designed to be extended by project-specific containers:

```singularity
Bootstrap: localimage
From: /path/to/containers/base/gpu-base/gpu-base_${BASE_VERSION}.sif

%post
    # Add project-specific dependencies
    pip install your-packages
```

## 📦 Building from Source

### Prerequisites
- Apptainer/Singularity with sudo or fakeroot
- Network access for downloading CUDA packages

### Build Command

```bash
# Auto-generate version
./scripts/build_container.sh base/gpu-base

# Specify version
./scripts/build_container.sh base/gpu-base v2025.10.15
```

### Build Process
1. Template is processed with environment variables
2. Container is built with CUDA packages
3. PyTorch and dependencies are installed
4. Build metadata is recorded
5. Checksum is calculated
6. Container is saved to storage location

## 📝 Files in this Directory

```
base/gpu-base/
├── singularity-gpu-base.def.template   # Template definition
├── .container-metadata.yml             # Container metadata
├── README.md                           # This file
├── CHANGELOG.md                        # Version history
```

## 🔍 Verification

Verify container integrity:

```bash
./scripts/verify_container.sh base/gpu-base v2025.10.15
```

## 📊 Container Metadata

- **Maintainer:** Ignacio Marcos Serrano (imarcoss@cnic.es)
- **Base Image:** condaforge/miniforge3
- **GPU Required:** Yes
- **Min GPU Memory:** Unknown

## 🐛 Troubleshooting

### CUDA Not Available

```bash
# Check NVIDIA driver on host
nvidia-smi

# Ensure --nv flag is used
singularity exec --nv container.sif python -c "import torch; print(torch.cuda.is_available())"
```

### Build Fails

- Ensure network access for CUDA package downloads
- Check disk space (build requires ~15GB temporary space)
- Verify sudo access or fakeroot availability

## 📞 Support

For issues or questions:
- Open an issue in the GitLab repository
- Contact: imarcoss@cnic.es

